"""Tests for the Stage E1-P harness (decision 041 v3 §1.6-1.7): the seed check, the plans and the freeze evidence
(stage_e1p_seed_check, stage_e1p_plan), whole sessions against a fake WSL (run_stage_e1p: server log, /metrics, an
OTLP receiver that writes its files, the iteration log, `vllm bench serve` with its seed, the nvidia-smi sampler, 031
idle and spill readings; no GPU), the smoke's checks and masking, the stop rules, the rerun limit, and the reading
(stage_e1p_analysis)."""

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import adv_plan as ap
import adv_smoke
import core_b_plan as bp
import gpu_idle as gi
import run_stage_e1p as r1p
import stage_e1 as se1
import stage_e1p_analysis as an
import stage_e1p_plan as pp
import stage_e1p_seed_check as sc
import test_run_c0_probe as tc0
import test_run_stage_d as td
import test_run_stage_e2 as tre2

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads(pp.CONFIG.read_text())
E1CFG = tomllib.loads((ROOT / "configs" / "stage_e1.toml").read_text())
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u"}
SEED_LABEL = {20261008: "A", 20261023: "B"}
SPIKES = {"A": (32, 64, 88), "B": (40, 120, 160)}          # content-locked: each seed its own positions
SMI = {"pstate": "P1", "sm_mhz": 2800.0, "mem_mhz": 14000.0, "power_w": 250.0, "temp_c": 66.0, "util_pct": 99.0,
       "mem_used_mib": 15000.0, "event_reasons": 0}


def seed_record(**over):
    seeds = CFG["e1p"]["seeds"]
    rec = {"stage": "E1P", "check": "seed", "vllm_version": PINS["vllm"]["version"],
           "script_sha256": pp.seed_check_sha256(), "input_len": 8064,
           "argv": {str(N): {x: pp.seed_check_argv(PINS, x, seed, N) for x, seed in seeds.items()}
                    for N in pp.seed_check_ns(CFG)},
           "checks": [{"N": N, "seeds": {x: {"seed": seed, "n": N, "server_tokens_ok": N, "prompt_len_ok": N,
                                             "prompts_sha256": x.lower() * 64} for x, seed in seeds.items()},
                       "ok": True, "problems": []} for N in pp.seed_check_ns(CFG)],
           "ok": True, "problems": []}
    return {**rec, **over}


def write_seed_check(root, rec=None, **over):
    path = Path(root).joinpath(*pp.SEED_CHECK)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec if rec is not None else seed_record(**over)))


class Receiver:
    """OtlpReceiver's files: a body under raw_dir at start, decoded records appended to spans_path."""

    def __init__(self, world, raw_dir, spans_path):
        self.world, self.raw_dir, self.spans_path = world, Path(raw_dir), Path(spans_path)
        self.records, self.errors = [], []

    def start(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "00001.pb").write_bytes(b"\x0a\x00")
        self.world.receiver = self
        return self

    def add(self, recs):
        self.records += recs
        with open(self.spans_path, "a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in recs)

    def close(self):
        return []

    def raise_if_failed(self):
        pass

    def stats(self):
        return {"posts": 1, "spans": len(self.records), "llm_request": len(self.records), "main": 0,
                "decode_errors": [], "errors": []}

    def read_spans(self):
        return list(self.records)


class TimedSmi(tc0.FakeSmi):
    """FakeSmi that records when it was stopped and returns the world's sampler problems."""

    def stop(self):
        super().stop()
        self.world.smi_stopped_at = self.world.t
        return list(self.world.smi_stop_problems)


class World(td.World):
    """td.World's clock, idle and spill readings with an observed-B0 server. spikes[seed label]: idx whose F is 50 ms
    (else 10 ms); slips[tag]: slipped groups (c0 test's cycle); behave[tag] in {"timeout", "hot", "exit", "unhealthy",
    "no_spans", "short_lens", "no_steps", "raise", "peek"} ("peek" records the committed files of the session mid-run);
    never_ready: boot labels whose /health never answers ("*": all); nd_edit: edits of the logged non-default args;
    smi_temp: the sampler's temperature (None: never reported); smi_dead: the sampler never started."""

    def __init__(self, tmp_path, *, spikes=SPIKES, slips=None, behave=None, never_ready=(), nd_edit=None,
                 smi_temp=66.0, bench_s=None, spill_at=(), smi_dead=False, smi_silent=False, env_fail=None,
                 hot_at_teardown=False, smi_hot_at_boot=None, smi_stop_problems=()):
        super().__init__(tmp_path, spill_at=spill_at)
        self.spikes, self.slips, self.behave = spikes, slips or {}, behave or {}
        self.never_ready, self.nd_edit, self.smi_temp, self.bench_s = set(never_ready), nd_edit or {}, smi_temp, \
            bench_s
        self.smi_dead, self.smi_silent, self.env_fail, self.hot_at_teardown = smi_dead, smi_silent, env_fail, \
            hot_at_teardown
        self.peeked, self.smi_hot_at_boot, self.smi_stop_problems = None, smi_hot_at_boot, smi_stop_problems
        self.smis, self.smi_started_at, self.smi_stopped_at = 0, None, None
        self.step, self.receiver, self.server_log, self.smi, self.unhealthy = 0, None, None, None, False
        self.smi_samples, self.bench_argvs = [], []

    def wall(self):
        return 1.79e9 + self.t

    def start_smi(self, path):
        smi = TimedSmi(self, path)
        self.smis += 1
        self.smi_started_at = self.t
        if self.smi_dead:
            smi.lose("the nvidia-smi sampler did not start: OSError")
        if self.smi_hot_at_boot == self.smis:                   # >= 88 C before the server starts
            smi.flag_hot()
        self.smi_samples = [] if self.smi_silent else [{"t": self.wall(), **SMI, "temp_c": self.smi_temp}]
        return smi

    def after_first_boot(self):
        return bool(self.server_argvs) and self.point == "pre"

    def compute_apps(self):
        if self.env_fail == "gpu_not_idle" and self.after_first_boot():
            return [{"pid": 9, "name": "other", "used_mib": 3000}]
        return []

    def gpu_sample(self):
        if self.env_fail == "gpu_check" and self.after_first_boot():
            raise RuntimeError("nvidia-smi timed out")
        if self.env_fail == "interrupt":
            raise KeyboardInterrupt
        return super().gpu_sample()

    def shared_usage(self):
        if self.env_fail == "spill_unreadable" and self.after_first_boot():
            raise gi.SharedUsageError("no adapter")
        return super().shared_usage()

    def signal_group(self, proc, sig):
        if self.hot_at_teardown and self.smi is not None:
            self.smi.flag_hot()
        super().signal_group(proc, sig)

    def popen(self, argv, env, log_path):
        log_path = Path(log_path)
        self.label, self.point, self.server_log, self.unhealthy = log_path.name.removesuffix(".server.log"), \
            f"ready:{log_path.name}", log_path, False
        self.server_argvs.append(argv)
        a = dict(zip(argv, argv[1:]))
        nd = {"max_num_batched_tokens": int(a["--max-num-batched-tokens"]),
              "kv_cache_memory_bytes": int(a["--kv-cache-memory-bytes"]),
              "otlp_traces_endpoint": a["--otlp-traces-endpoint"], "enable_logging_iteration_details": True,
              **self.nd_edit}
        log_path.write_text(f"{int(self.wall() * 1000)} (APIServer pid=1) INFO non-default args: {nd!r}\n"
                            f"{int(self.wall() * 1000)} (EngineCore pid=2) INFO [cuda.py:478] Using "
                            "AttentionBackendEnum.FLASH_ATTN backend.\n")
        self.proc = td.Proc(77)
        return self.proc

    def http_get(self, url):
        alive = self.proc is not None and self.proc.returncode is None
        if url.endswith("/health"):
            if "*" in self.never_ready or self.label in self.never_ready or self.unhealthy:
                return 0, "refused"
            return (200, "") if alive else (0, "refused")
        lab = 'engine="0",model_name="m"'
        return 200, (f'vllm:cache_config_info{{block_size="528",enable_prefix_caching="False",engine="0",'
                     f'num_gpu_blocks="248"}} 1.0\nvllm:num_requests_running{{{lab}}} 8.0\n'
                     f"vllm:num_requests_waiting{{{lab}}} 0.0\nvllm:kv_cache_usage_perc{{{lab}}} 0.4\n")

    def peek(self, out):
        """What a hard kill now would leave outside raw/ (what fetch_session.sh fetches)."""
        rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines() if x.strip()]
        derived = {p.name: [json.loads(x) for x in p.read_text().splitlines()] for p in (out / "derived").glob(
            "*-requests.jsonl")}
        return {"rows": rows, "derived": derived, "derived_files": sorted(p.name for p in (out / "derived").iterdir())}

    def run_bench(self, argv, log_path, env, timeout_s, stop):
        self.bench_argvs.append(argv)
        a = dict(zip(argv, argv[1:]))
        n, seed, prefix = int(a["--num-prompts"]), int(a["--seed"]), a["--request-id-prefix"]
        tag = Path(a["--result-filename"]).name.removesuffix(".bench.json")
        how = self.behave.get(tag)
        Path(log_path).write_text("log\n")
        t0 = self.wall()
        if how == "peek":
            self.peeked = self.peek(Path(a["--result-dir"]).parent)
        if how == "raise":
            raise r1p.RunError("a harness failure in the bench")
        if how == "timeout":
            self.t += timeout_s
            return -9, True, False
        if how == "hot":
            self.smi.flag_hot()
            self.t += 30
            return -9, False, True
        if how == "lose":
            self.smi.lose("the nvidia-smi sampler exited (code 9)")
            self.t += 30
            return -9, False, True
        if how == "exit":
            self.t += 5
            return 1, False, False
        ttfts = tc0.cycle_s(n, self.slips.get(tag, ()))
        spikes = self.spikes.get(SEED_LABEL[seed], ())
        q, p = 0.01, 0.3
        recs = []
        for i in range(n):
            f = 0.05 if i in spikes else 0.01
            recs.append({"id": f"cmpl-{prefix}{i}-0", "arrival": t0 + 0.5 + 0.2 * i, "Q": q, "P": p,
                         "server_ttft": q + p + f, "e2e": ttfts[i] + 127 * 0.05, "prompt_tokens": 8064,
                         "completion_tokens": 128})
        if how == "no_spans":
            recs = recs[:-1]
        if how == "no_P":
            recs[8] = {k: v for k, v in recs[8].items() if k != "P"}
        if how == "dup_span":
            recs.append(dict(recs[5]))
        self.receiver.add(recs + [{"id": "cmpl-warmup-0", "arrival": t0, "Q": q, "P": p, "server_ttft": 1.0}])
        lens = [8000 if how == "short_lens" and i % 2 else 8064 for i in range(n)]
        res = {"completed": n, "failed": 0, "duration": 0.2 * n, "request_throughput": 5.0, "output_throughput": 300.0,
               "total_token_throughput": 4e4, "input_lens": lens, "output_lens": [128] * n, "ttfts": ttfts,
               "latencies": [t + 127 * 0.05 for t in ttfts], "start_times": [1000.0 + 0.2 * i for i in range(n)]}
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(res))
        if how != "no_steps":
            with open(self.server_log, "a") as fh:
                for j in range(int((0.2 * n + 4) / 0.1)):
                    self.step += 1
                    ctx = 2048 if self.step % 3 == 0 else 0
                    fh.write(f"{int((t0 + 0.05 + 0.1 * j) * 1000)} (EngineCore pid=2) INFO Iteration({self.step}): "
                             f"{int(ctx > 0)} context requests, {ctx} context tokens, 7 generation requests, 7 "
                             "generation tokens, iteration elapsed time: 30.0 ms, GPU KV cache usage: 40.0%\n")
        self.unhealthy = how == "unhealthy"
        self.point = f"bench:{tag}"
        self.t += self.bench_s or 0.2 * n + 5
        if how == "hot_after":                          # the bench finished, then the GPU crossed 88 C
            self.smi.flag_hot()
        return 0, False, False


def setup(tmp_path):
    root = tmp_path / "repo"
    (root / "results" / "stage_e").mkdir(parents=True)
    write_seed_check(root)
    return root


def session(root, plan, *, mode, world=None, sid="s", **kw):
    w = world or World(root.parent / f"w-{sid}")
    stamp = iter(range(10 ** 6))
    args = dict(pins=PINS, pins_sha256="p" * 64, out_dir=root / "results" / "stage_e" / sid, session_id=sid,
                mode=mode, expected_plan=plan, popen=w.popen, signal_group=w.signal_group, group_alive=w.group_alive,
                http_get=w.http_get, run_bench=w.run_bench, gpu_sample=w.gpu_sample, compute_apps=w.compute_apps,
                shared_usage=w.shared_usage, start_smi=w.start_smi,
                receiver_factory=lambda raw, spans, pre: Receiver(w, raw, spans), lock_held_check=lambda: True,
                gpu_lock_held=True, capture_env=tre2.Env(), t_start=w.now(), vllm_bin="/v", base_env=BASE_ENV,
                toolchain_check=lambda env: [], sample_interval=None, now=w.now, sleep=w.sleep, wall=w.wall,
                clock=lambda: f"2026-10-24T{sid}-{next(stamp):06d}", repo_root=root)
    args.update(kw)
    return r1p.run_session(plan, **args), w


def smoke(tmp_path, world=None, sid="smoke-e1p", **kw):
    root = setup(tmp_path)
    s, w = session(root, pp.build_plan(PINS, CFG, mode="smoke", root=root), mode="smoke", world=world, sid=sid, **kw)
    return s, root, w


def frozen(sid="smoke-e1p"):
    return {**CFG, "freeze": {"e1p": True, "smoke_e1p": sid}}


def formal_setup(tmp_path):
    s, root, _ = smoke(tmp_path)
    assert s["smoke"]["ok"] is True, s["smoke"]["checks"]
    return root, pp.build_plan(PINS, frozen(), mode="formal", root=root)


def sdir(root, sid):
    return root / "results" / "stage_e" / sid


def rows_of(root, sid):
    return [json.loads(x) for x in (sdir(root, sid) / "rows.jsonl").read_text().splitlines()]


def lock_stop_after(n):
    """lock_held_check answers True n times (the session's own check, then the boots'), then False."""
    calls = iter([True] * n + [False] * 99)
    return lambda: next(calls)


# ---------------------------------------------------------------- config, seed check and plans

def test_the_repo_config_is_041s_and_unfrozen():
    assert pp.config_problems(CFG) == [] and CFG["freeze"] == {"e1p": False, "smoke_e1p": ""}
    assert pp.config_problems(frozen()) == []
    for sid in ("", "..", ".", "../x", "a/b", "x y"):
        assert pp.config_problems(frozen(sid)), sid
    for key, bad in (("seeds", {"A": 20261008, "B": 20261008}), ("order_even", ["A", "B", "B", "A"]),
                     ("f_spike_ms", 35), ("cap_hours", 3.5), ("a_prior", [32, 64]), ("j_content_max", 0.25)):
        assert any(key in p for p in pp.config_problems({**CFG, "e1p": {**CFG["e1p"], key: bad}})), key
    assert pp.config_problems({**CFG, "e1p": {**CFG["e1p"], "extra": 1}})
    assert pp.e1_config()["freeze"]["e1"] is True and se1.config_problems(pp.e1_config()) == []


def test_e1_d_must_still_be_its_frozen_harness(monkeypatch):
    monkeypatch.setattr(se1, "harness_sha256", lambda root=None: "0" * 64)
    with pytest.raises(pp.PlanError, match="e1_harness_sha256"):
        pp.e1_config()


def test_a_plan_pins_the_configs_content_without_freeze(tmp_path):
    root = setup(tmp_path)
    a, b = frozen(), CFG
    assert pp.config_sha256(a) == pp.config_sha256(b) != pp.config_sha256({**b, "e1p": {**b["e1p"], "boots": 5}})
    p = pp.build_plan(PINS, CFG, mode="smoke", root=root)
    assert p["stage_e1p_sha256"] == pp.config_sha256(CFG) and p["harness_sha256"] == pp.harness_sha256()
    assert p["stage_e1_sha256"] == pp.sha256_json(E1CFG)
    assert p["seed_check_sha256"] == hashlib.sha256(root.joinpath(*pp.SEED_CHECK).read_bytes()).hexdigest()
    assert set(pp.HARNESS_FILES) == {"benchmarks/stage_e1p_plan.py", "benchmarks/run_stage_e1p.py",
                                     "benchmarks/stage_e1p_analysis.py", "benchmarks/run_c0_probe.py",
                                     "benchmarks/c0_analysis.py"}


def _edit(**over):
    return lambda rec: rec.update(over)


def _seed(N, x, **over):
    def edit(rec):
        next(c for c in rec["checks"] if c["N"] == N)["seeds"][x].update(over)
    return edit


def _drop_n(N):
    return lambda rec: rec.update(checks=[c for c in rec["checks"] if c["N"] != N])


@pytest.mark.parametrize("edit, why", [
    (None, "no readable seed check"), (_edit(ok=False, problems=["x"]), "did not pass"),
    (_edit(vllm_version="0.29.0"), "0.29.0"), (_edit(script_sha256="0" * 64), "script_sha256"),
    (_edit(input_len=8000), "input_len"), (_edit(check="other"), "not an E1-P seed check"),
    (_drop_n(16), "one record per N"), (_edit(checks="x"), "one record per N"),
    (lambda rec: rec["checks"].append(dict(rec["checks"][0])), "one record per N"),
    (lambda rec: rec["checks"][0].update(N=[200]), "one record per N"),
    (lambda rec: rec["checks"][0].update(N=200.0), "one record per N"),
    (_seed(200, "B", server_tokens_ok=199), "199 of 200"), (_seed(16, "A", n=15), "of 15 prompts"),
    (_seed(200, "A", n=199, server_tokens_ok=199), "199 of 199"),
    (_seed(200, "B", seed=20261024), "seed 20261024"), (_seed(16, "B", seed=20261024), "N 16, seed B"),
    (_seed(200, "B", prompts_sha256="a" * 64), "same prompts"), (_seed(16, "B", prompts_sha256="a" * 64), "N 16"),
    (_seed(200, "A", prompts_sha256=None), "same prompts"),
    (lambda rec: rec["checks"][0]["seeds"].update(C={}), "no A / B record"),
    (lambda rec: rec["checks"][0]["seeds"].update(A=None), "no A / B record"),
    (lambda rec: rec["argv"].update({"200": ["x"]}), "another bench argv"),
    (lambda rec: rec["argv"]["200"]["B"].append("--random-prefix-len"), "another bench argv")])
def test_every_plan_needs_the_passed_seed_check_of_this_script_argv_and_design(tmp_path, edit, why):
    root = tmp_path / "repo"
    if edit is not None:
        rec = seed_record()
        edit(rec)
        write_seed_check(root, rec)
    for mode in ("smoke", "formal"):
        with pytest.raises(pp.PlanError, match=why):
            pp.build_plan(PINS, frozen(), mode=mode, root=root)


@pytest.mark.parametrize("N, label", [(200, "A"), (200, "B"), (16, "A"), (16, "B")])
def test_the_seed_check_must_have_sampled_the_argv_the_harness_builds_now(tmp_path, N, label):
    root = tmp_path / "repo"
    rec = seed_record()
    rec["argv"][str(N)][label] = rec["argv"][str(N)][label][:-1] + ["other-prefix-"]
    write_seed_check(root, rec)
    with pytest.raises(pp.PlanError, match=f"N {N}, seed {label} sampled another bench argv"):
        pp.build_plan(PINS, CFG, mode="smoke", root=root)


def _sampling_args(argv):
    """A bench argv without what samples nothing: the binary, the request-id prefix and the result paths."""
    drop = {"--request-id-prefix", "--result-dir", "--result-filename"}
    out, skip = [], False
    for a in argv[1:]:
        if skip:
            skip = False
        elif a in drop:
            skip = True
        else:
            out.append(a)
    return out


def test_the_seed_check_needs_every_prompt_at_8064_on_the_server_and_different_prompts_per_seed():
    def prompts(tag, n=200, bad=()):
        return [(f"{tag}{i}" + ("!" if i in bad else ""), 8064) for i in range(n)]

    server = lambda p: 8065 if p.endswith("!") else 8064  # noqa: E731
    ok = sc.check({"A": (1, prompts("a")), "B": (2, prompts("b"))}, N=200, input_len=8064, server_len=server)
    assert ok["ok"] is True and ok["seeds"]["B"]["server_tokens_ok"] == 200 and ok["N"] == 200
    bad = sc.check({"A": (1, prompts("a")), "B": (2, prompts("b", bad=(7,)))}, N=200, input_len=8064,
                   server_len=server)
    assert bad["ok"] is False and bad["seeds"]["B"]["server_tokens_other"] == [8065] and "seed B" in bad["problems"][0]
    assert bad["seeds"]["B"]["prompt_len_ok"] == 200                        # the sampler's own count is descriptive
    same = sc.check({"A": (1, prompts("x")), "B": (2, prompts("x"))}, N=200, input_len=8064, server_len=server)
    assert same["ok"] is False and "same prompts" in same["problems"][0]
    assert sc.check({"A": (1, prompts("a", n=16)), "B": (2, prompts("b"))}, N=200, input_len=8064,
                    server_len=server)["ok"] is False
    assert pp.seed_check_ns(CFG) == (200, 16)
    assert sc.env_problems({"PATH": "/b"}) == [] and sc.env_problems({"VLLM_USE_FASTOKENS": "1"})
    assert sc.env_problems({"PYTORCH_CUDA_ALLOC_CONF": "x"})
    argv = pp.seed_check_argv(PINS, "B", 20261023, 16)
    assert argv[argv.index("--seed") + 1] == "20261023" and argv[argv.index("--num-prompts") + 1] == "16"


def test_the_server_is_advanceds_observed_b0_and_the_client_changes_only_seed_n_and_prefix():
    argv, env = pp.server_argv_for(PINS, vllm_bin="/v", base_env={**BASE_ENV, "PYTORCH_CUDA_ALLOC_CONF": "x"})
    sb = PINS["stage_b"]
    want, _ = ap.server_argv_for("B0", PINS, gdn_prefill_backend=sb["gdn_prefill_backend"],
                                 gpu_memory_utilization=sb["gpu_memory_utilization"], port=8000, vllm_bin="/v",
                                 base_env=BASE_ENV)
    assert argv == want and "--otlp-traces-endpoint" in argv and not any(k.startswith("PYTORCH_") for k in env)
    core = bp.bench_argv(PINS, L=8192, C=8, mix="uniform", result_dir="/r", result_filename="x.json", port=8000,
                         vllm_bin="/v")
    got = pp.bench_argv_for(PINS, seed=20261023, N=16, prefix="e1p-b1-n2-", result_dir="/r", result_filename="x.json",
                            vllm_bin="/v")
    assert got[-2:] == ["--request-id-prefix", "e1p-b1-n2-"] and len(got) == len(core) + 2
    diff = {core[i - 1]: (core[i], got[i]) for i in range(len(core)) if core[i] != got[i]}
    assert diff == {"--num-prompts": ("200", "16"), "--seed": ("20261008", "20261023")}


def test_the_smoke_plan_is_one_boot_of_a_then_b_at_n_16(tmp_path):
    p = pp.build_plan(PINS, CFG, mode="smoke", root=setup(tmp_path))
    assert [(x["tag"], x["seed_label"], x["seed"], x["N"]) for x in p["boots"][0]["benches"]] == [
        ("b1-n1", "A", 20261008, 16), ("b1-n2", "B", 20261023, 16)]
    assert len(p["boots"]) == 1 and p["cap_s"] == 3.0 * 3600 and p["timeout_s"] == 720 and p["smoke"] is None
    assert (p["expected_blocks"], p["block_size"], p["temp_stop_c"], p["max_failed_boots"]) == (248, 528, 88, 2)
    with pytest.raises(pp.PlanError, match="mode"):
        pp.build_plan(PINS, CFG, mode="pilot")


def test_formal_plans_need_the_freeze_and_a_passed_smoke_and_take_its_time_off_the_cap(tmp_path):
    root, p = formal_setup(tmp_path)
    benches = [x for boot in p["boots"] for x in boot["benches"]]
    assert len(p["boots"]) == 6 and len(benches) == 24 and len({x["prefix"] for x in benches}) == 24
    assert [[x["seed_label"] for x in boot["benches"]] for boot in p["boots"]] == [list("ABBA"), list("BAAB")] * 3
    for label in "AB":
        pos = sorted(x["bench"] for x in benches if x["seed_label"] == label)
        assert pos == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert {x["N"] for x in benches} == {200}
    smoke_s = json.loads((sdir(root, "smoke-e1p") / "session.json").read_text())
    assert p["smoke"] == {"session": "smoke-e1p", "elapsed_s": smoke_s["elapsed_s"]}
    assert p["cap_s"] == int(3.0 * 3600 - smoke_s["elapsed_s"]) and 0 < smoke_s["elapsed_s"] < 3600
    with pytest.raises(pp.PlanError, match="freeze.e1p"):
        pp.build_plan(PINS, CFG, mode="formal", root=root)
    with pytest.raises(pp.PlanError, match="smoke first"):
        pp.build_plan(PINS, frozen("none"), mode="formal", root=root)
    assert pp.build_plan(PINS, frozen(), mode="formal", root=root, pins_sha256="p" * 64)["pins_sha256"] == "p" * 64
    with pytest.raises(pp.PlanError, match="pins_sha256"):
        pp.build_plan(PINS, frozen(), mode="formal", root=root, pins_sha256="q" * 64)


def test_the_smoke_evidence_must_be_this_designs_passed_smoke(tmp_path):
    root, _ = formal_setup(tmp_path)
    d = sdir(root, "smoke-e1p")
    s, plan = json.loads((d / "session.json").read_text()), json.loads((d / "plan.json").read_text())
    other = {**plan, "boots": [{**plan["boots"][0], "benches": plan["boots"][0]["benches"][:1]}]}
    for over_s, over_plan, why in (
            ({"harness_sha256": "0" * 64}, None, "another harness"),
            ({"status": "running"}, None, "finished, passed"),
            ({"smoke": {**s["smoke"], "ok": False}}, None, "finished, passed"),
            ({"env_valid": False}, None, "env not valid"),
            ({"host": {"system": "Darwin", "wsl": False, "nvidia": False}}, None, "host"),
            ({"vllm_version": "0.29.0"}, None, "pinned"),
            ({"elapsed_s": None}, None, "elapsed_s"),
            ({"elapsed_s": 4 * 3600}, None, "elapsed_s"),
            ({"plan_sha256": pp.sha256_json(other)}, other, "smoke plan"),
            ({}, other, "plan_sha256")):
        (d / "session.json").write_text(json.dumps({**s, **over_s}))
        (d / "plan.json").write_text(json.dumps(over_plan or plan))
        with pytest.raises(pp.PlanError, match=why):
            pp.build_plan(PINS, frozen(), mode="formal", root=root)
    (d / "session.json").write_text(json.dumps(s))
    (d / "plan.json").write_text(json.dumps(plan))
    assert pp.build_plan(PINS, frozen(), mode="formal", root=root)["boots"]
    write_seed_check(root, prompts_sha256="x")                       # another record: the smoke's plan differs
    with pytest.raises(pp.PlanError, match="smoke plan"):
        pp.build_plan(PINS, frozen(), mode="formal", root=root)


# ---------------------------------------------------------------- smoke sessions

def no_latency(root, sid):
    """Every committed file of a smoke (rows.jsonl, derived/, session.json's boots, smoke.json) without a latency
    key; the session's own elapsed_s / elapsed_at_start_s (the cap's bookkeeping) are its only timing."""
    out = sdir(root, sid)
    saved = json.loads((out / "session.json").read_text())
    found = adv_smoke.latency_keys(rows_of(root, sid)) + adv_smoke.latency_keys(saved["boots"])
    found += adv_smoke.latency_keys(json.loads((out / "smoke.json").read_text()))
    for p in (out / "derived").glob("*-requests.jsonl"):
        found += adv_smoke.latency_keys([json.loads(x) for x in p.read_text().splitlines()])
    top = [k for k in adv_smoke.latency_keys({k: v for k, v in saved.items() if k != "boots"})
           if k.split(".")[0] not in ("elapsed_s", "elapsed_at_start_s")]
    return found + top


def test_a_smoke_runs_seed_a_then_b_checks_041s_conditions_and_masks_the_latencies(tmp_path):
    s, root, w = smoke(tmp_path)
    sm = s["smoke"]
    assert sm["ok"] is True, sm["checks"]
    assert s["status"] == "finished" and s["outcomes"]["valid"] == 2 and s["stopped"] is None
    seeds = [a[a.index("--seed") + 1] for a in w.bench_argvs]
    assert seeds == ["20261008", "20261023"] and {a[a.index("--num-prompts") + 1] for a in w.bench_argvs} == {"16"}
    assert [a[-1] for a in w.bench_argvs] == ["e1p-b1-n1-", "e1p-b1-n2-"] and len(w.server_argvs) == 1
    for argv, (x, seed) in zip(w.bench_argvs, (("A", 20261008), ("B", 20261023))):   # what the seed check sampled
        assert _sampling_args(argv) == _sampling_args(pp.seed_check_argv(PINS, x, seed, 16))
    out = sdir(root, "smoke-e1p")
    assert sorted(p.name for p in (out / "raw").iterdir()) == ["b1-smi.jsonl", "b1.server.log", "env"]
    assert sorted(p.name for p in (out / "raw" / "env").iterdir()) == ["b1.after.json", "b1.before.json"]
    assert sorted(p.name for p in (out / "env").iterdir()) == ["preflight.json"]     # taken before any work
    assert "raw/b1.otlp" in s["masked"] and "raw/b1.spans.jsonl" in s["masked"] and "raw/b1-n1-e1.json" in s["masked"]
    assert sorted(p.name for p in (out / "derived").iterdir()) == ["b1-n1-requests.jsonl", "b1-n2-requests.jsonl"]
    req = (out / "derived" / "b1-n1-requests.jsonl").read_text().splitlines()
    assert len(req) == 16 and set(json.loads(req[0])) == {"idx", "cls"}
    rows = rows_of(root, "smoke-e1p")
    assert all(r["ok"] and r["e1_ok"] for r in rows) and no_latency(root, "smoke-e1p") == []
    saved = json.loads((out / "session.json").read_text())
    assert saved["elapsed_s"] == s["elapsed_s"] > 0 and (out / "smoke.json").is_file()
    b1 = saved["boots"]["b1"]
    assert b1["smi_summary"]["temp_max_c"] == 66.0 and "main" not in b1["receiver"] and b1["smi"] == "raw/b1-smi.jsonl"
    assert not {"t_start", "t_end", "startup_s"} & set(b1) and "lines" not in b1["startup"] and b1["startup"]["counts"]


def test_a_smoke_never_leaves_a_latency_outside_raw_even_mid_run(tmp_path):
    s, root, w = smoke(tmp_path, world=World(tmp_path / "w", behave={"b1-n2": "peek"}))
    seen = w.peeked
    assert s["smoke"]["ok"] is True and adv_smoke.latency_keys(seen["rows"]) == [] and len(seen["rows"]) == 1
    assert seen["derived_files"] == ["b1-n1-requests.jsonl"] and set(seen["derived"]["b1-n1-requests.jsonl"][0]) == {
        "idx", "cls"}


def test_an_aborted_smoke_is_masked_and_fails(tmp_path):
    with pytest.raises(r1p.RunError, match="harness failure"):
        smoke(tmp_path, world=World(tmp_path / "w", behave={"b1-n2": "raise"}))
    root = tmp_path / "repo"
    saved = json.loads((sdir(root, "smoke-e1p") / "session.json").read_text())
    assert saved["status"] == "aborted" and saved["smoke"]["ok"] is False and no_latency(root, "smoke-e1p") == []
    assert not list((sdir(root, "smoke-e1p") / "raw").glob("*-e1.json"))


@pytest.mark.parametrize("world_kw, failed", [
    (dict(behave={"b1-n2": "no_spans"}), "spans"), (dict(behave={"b1-n2": "no_spans"}), "f_computable"),
    (dict(behave={"b1-n2": "short_lens"}), "input_lens_8064"), (dict(behave={"b1-n1": "no_steps"}), "steps_in_window"),
    (dict(behave={"b1-n1": "exit"}), "every_bench_valid"),
    (dict(spill_at=("bench:b1-n1",)), "no_spill")])
def test_each_smoke_condition_can_fail_it(tmp_path, world_kw, failed):
    s, _, _ = smoke(tmp_path, world=World(tmp_path / "w", **world_kw))
    assert s["smoke"]["ok"] is False and s["smoke"]["checks"][failed] is False


@pytest.mark.parametrize("world_kw, rule", [
    (dict(smi_temp=None), "telemetry"), (dict(smi_silent=True), "telemetry"), (dict(smi_dead=True), "telemetry"),
    (dict(smi_hot_at_boot=1), "temperature")])
def test_no_server_starts_without_a_first_temperature(tmp_path, world_kw, rule):
    s, _, w = smoke(tmp_path, world=World(tmp_path / "w", **world_kw))
    assert w.server_argvs == [] and s["status"] == "refused" and s["stopped"]["rule"] == rule
    assert s["smoke"]["ok"] is False
    if world_kw.get("smi_silent"):                                  # it waited exactly SMI_FIRST_S for one
        assert w.smi_stopped_at - w.smi_started_at == r1p.SMI_FIRST_S


def test_88c_before_a_later_boots_server_is_a_valid_temperature_stop(tmp_path):
    root, plan = formal_setup(tmp_path)
    s, w = session(root, plan, mode="formal", world=World(tmp_path / "wf", smi_hot_at_boot=2), sid="f")
    assert s["stopped"]["rule"] == "temperature" and s["stopped"]["boot"] == 2 and len(w.server_argvs) == 1
    assert s["status"] == "finished" and [a["valid"] for a in pp.formal_attempts(root)] == [True]


def test_sampler_problems_fail_the_smoke(tmp_path):
    s, _, _ = smoke(tmp_path, world=World(tmp_path / "w", smi_stop_problems=["the nvidia-smi reader is still open"]))
    assert s["smoke"]["ok"] is False and s["smoke"]["checks"]["smi"] is False


def test_the_smoke_checks_on_rows_cover_coverage_schema_blocks_stops_and_teardown(tmp_path):
    root = setup(tmp_path)
    plan = pp.build_plan(PINS, CFG, mode="smoke", root=root)
    rows = [{"tag": t, "seed_label": x, "N": 16, "ok": True, "spans_main": 16, "f_computable": True,
             "n_steps_window": 5, "input_lens_8064": True, "derived_schema_ok": True, "e1_ok": True,
             "num_gpu_blocks": 248} for t, x in (("b1-n1", "A"), ("b1-n2", "B"))]
    sess = {"boots": {"b1": {"startup_ok": True, "smi_summary": {"n_samples": 3, "fields_missing": []}}},
            "stopped": None}
    assert r1p.smoke_checks(plan, rows, sess)["ok"] is True
    for edit, failed in ((lambda r, s: r.pop(), "covered"), (lambda r, s: r.reverse(), "covered"),
                         (lambda r, s: r[0].update(e1_ok=False), "derived_schema"),
                         (lambda r, s: r[0].update(e1_error="x"), "derived_schema"),
                         (lambda r, s: r[1].update(num_gpu_blocks=200), "blocks"),
                         (lambda r, s: s.update(stopped={"rule": "cap"}), "not_stopped"),
                         (lambda r, s: s.update(teardown_problems=[{"boot": "b1"}]), "teardown_clean"),
                         (lambda r, s: s["boots"]["b1"].update(startup_ok=False), "startup")):
        r, s = copy.deepcopy(rows), copy.deepcopy(sess)
        edit(r, s)
        assert r1p.smoke_checks(plan, r, s)["checks"][failed] is False, failed


def test_a_smoke_whose_server_never_starts_fails_and_records_why(tmp_path):
    s, root, _ = smoke(tmp_path, world=World(tmp_path / "w", never_ready=("*",)), startup_timeout_s=10)
    rows = rows_of(root, "smoke-e1p")
    assert s["smoke"]["ok"] is False and all(r["not_run"] for r in rows)
    assert s["boots"]["b1"]["failed"] is True and s["status"] == "finished"
    s, _, _ = smoke(tmp_path / "x", world=World(tmp_path / "wx", nd_edit={"long_prefill_token_threshold": 512}))
    assert s["smoke"]["checks"]["startup"] is False
    assert any("long_prefill_token_threshold" in why for why in s["boots"]["b1"]["reasons"])


# ---------------------------------------------------------------- formal sessions

def test_a_formal_session_runs_24_benches_with_their_spikes_and_e1_files(tmp_path):
    root, plan = formal_setup(tmp_path)
    world = World(tmp_path / "wf", slips={"b1-n1": (8,), "b2-n3": (16, 24)})
    s, w = session(root, plan, mode="formal", world=world, sid="f")
    rows = rows_of(root, "f")
    assert s["status"] == "finished" and s["stopped"] is None and len(rows) == 24 and all(r["ok"] for r in rows)
    assert len(w.server_argvs) == 6 and s["cap_s"] == plan["cap_s"]
    out = sdir(root, "f")
    e1 = json.loads((out / "derived" / "b1-n1-e1.json").read_text())
    assert e1["f_spikes"]["spike_groups"] == [4, 8, 11]                         # seed A at 32, 64, 88
    assert [x["start_idx"] for x in e1["episodes"]] == [65] and e1["episodes"][0]["links"]["L1"] is True
    b = json.loads((out / "derived" / "b1-n2-e1.json").read_text())
    assert b["f_spikes"]["spike_groups"] == [5, 15, 20]                         # seed B at 40, 120, 160
    r = next(x for x in rows if x["tag"] == "b1-n1")
    assert r["spans_main"] == 200 and r["n_steps_window"] > 0 and r["input_lens_8064"] is True
    assert r["t_before"] < r["t_after"] and set(json.loads((out / "derived" / "b1-n1-requests.jsonl").read_text(
    ).splitlines()[0])) == set(r1p.DERIVED_KEYS)
    assert (out / "raw" / "b1.spans.jsonl").is_file() and (out / "raw" / "b1.otlp" / "00001.pb").is_file()
    assert sorted(p.name for p in (out / "env").iterdir()) == sorted(["preflight.json"] + [
        f"b{b}.{w}.json" for b in range(1, 7) for w in ("after", "before")])
    assert sorted(p.name for p in (out / "derived").glob("*-smi.jsonl")) == [f"b{b}-smi.jsonl" for b in range(1, 7)]
    assert s["boots"]["b1"]["smi"] == "derived/b1-smi.jsonl" and not (out / "raw" / "env").exists()
    seeds = CFG["e1p"]["seeds"]
    for argv in w.bench_argvs:                                      # what the seed check sampled
        x = SEED_LABEL[int(argv[argv.index("--seed") + 1])]
        assert _sampling_args(argv) == _sampling_args(pp.seed_check_argv(PINS, x, seeds[x], 200))
    assert s["boots"]["b1"]["receiver"]["spans"] > 800 and "main" not in s["boots"]["b1"]["receiver"]
    assert pp.formal_attempts(root)[0]["valid"] is True
    with pytest.raises(r1p.RunError, match="already exist"):
        session(root, plan, mode="formal", sid="f2")


def test_the_runs_plan_and_lock_are_checked_before_anything_is_written(tmp_path):
    root, plan = formal_setup(tmp_path)
    for kw, why in ((dict(expected_plan={**plan, "cap_s": 1}), "build_plan"),
                    (dict(gpu_lock_held=False), "GPU lock"), (dict(lock_held_check=lambda: False), "GPU lock"),
                    (dict(base_env={**BASE_ENV, "PYTORCH_CUDA_ALLOC_CONF": "x"}), "PYTORCH_"),
                    (dict(toolchain_check=lambda env: ["nvcc"]), "toolchain")):
        with pytest.raises(r1p.RunError, match=why):
            session(root, plan, mode="formal", sid="g", **kw)
        assert not sdir(root, "g").exists()
    with pytest.raises(r1p.RunError, match="results/stage_e"):
        session(root, plan, mode="formal", sid="g", out_dir=root / "results" / "g")


def test_a_bench_without_every_span_is_invalid_and_two_such_benches_fail_the_boot(tmp_path):
    root, plan = formal_setup(tmp_path)
    world = World(tmp_path / "wf", behave={"b1-n1": "no_spans", "b1-n3": "no_spans", "b2-n2": "no_spans",
                                           "b2-n4": "no_spans"})
    s, _ = session(root, plan, mode="formal", world=world, sid="f")
    rows = {r["tag"]: r for r in rows_of(root, "f")}
    assert rows["b1-n1"]["ok"] is False and rows["b1-n1"]["f_computable"] is False and rows["b1-n2"]["ok"] is True
    assert s["boots"]["b1"]["failed"] is True and s["stopped"]["rule"] == "failed_boots" and s["stopped"]["boot"] == 2
    assert len(rows) == 8 and s["status"] == "finished" and pp.formal_attempts(root)[0]["valid"] is True


@pytest.mark.parametrize("how, rule", [("hot", "temperature"), ("timeout", None), ("unhealthy", None)])
def test_a_hot_gpu_stops_at_once_and_a_timeout_or_unhealthy_server_ends_only_the_boot(tmp_path, how, rule):
    root, plan = formal_setup(tmp_path)
    s, _ = session(root, plan, mode="formal", world=World(tmp_path / "wf", behave={"b2-n3": how}), sid="f")
    rows = {r["tag"]: r for r in rows_of(root, "f")}
    assert rows["b2-n3"]["ok"] is False and rows["b2-n4"]["not_run"] is True
    assert pp.formal_attempts(root)[0]["valid"] is True                # 028's stops: analyzed as they are
    if rule:
        assert s["stopped"]["rule"] == rule and len(rows) == 8 and rows["b2-n3"]["stopped_hot"] is True
    else:
        assert s["stopped"] is None and len(rows) == 24 and s["boots"]["b2"]["failed"] is True
        assert s["failed_boots"] == 1 and rows["b3-n1"]["ok"] is True


def test_the_cap_stops_new_boots_and_benches_by_028s_margins(tmp_path):
    root, plan = formal_setup(tmp_path)
    s, _ = session(root, plan, mode="formal", world=World(tmp_path / "wf", bench_s=1500), sid="f")
    rows = rows_of(root, "f")
    assert s["stopped"]["rule"] == "cap" and s["status"] == "finished"
    ran = [r for r in rows if not r["not_run"]]
    assert ran and all(r["ok"] for r in ran) and s["elapsed_s"] <= plan["cap_s"] + 1500
    assert pp.formal_attempts(root)[0]["valid"] is True


def test_no_server_started_is_refused_and_an_environment_stop_voids_the_session(tmp_path):
    root, plan = formal_setup(tmp_path)
    busy = World(tmp_path / "w1")
    busy.compute_apps = lambda: [{"pid": 9, "name": "other", "used_mib": 3000}]
    s, _ = session(root, plan, mode="formal", world=busy, sid="f1")
    assert s["status"] == "refused" and s["stopped"]["rule"] == "gpu_not_idle" and pp.formal_attempts(root) == []
    s, w = session(root, plan, mode="formal", world=World(tmp_path / "w0", smi_dead=True), sid="f0")
    assert s["status"] == "refused" and s["stopped"]["rule"] == "telemetry" and w.server_argvs == []
    assert pp.formal_attempts(root) == []
    s, _ = session(root, plan, mode="formal", sid="f2", lock_held_check=lock_stop_after(3))  # boot 3 stops
    assert s["status"] == "finished" and s["stopped"]["rule"] == "lock"
    assert [a["valid"] for a in pp.formal_attempts(root)] == [False]
    s, _ = session(root, plan, mode="formal", sid="f3")
    assert s["stopped"] is None and [a["valid"] for a in pp.formal_attempts(root)] == [False, True]


def test_an_interrupt_during_the_preflight_is_refused_not_an_attempt(tmp_path):
    root, plan = formal_setup(tmp_path)

    def interrupted(**kw):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        session(root, plan, mode="formal", sid="f", capture_env=interrupted)
    saved = json.loads((sdir(root, "f") / "session.json").read_text())
    assert saved["status"] == "refused" and "KeyboardInterrupt" in saved["aborted"] and pp.formal_attempts(root) == []


@pytest.mark.parametrize("world_kw, rule", [
    (dict(env_fail="gpu_not_idle"), "gpu_not_idle"), (dict(env_fail="gpu_check"), "gpu_check"),
    (dict(env_fail="spill_unreadable"), "spill_unreadable"), (dict(behave={"b2-n1": "lose"}), "telemetry")])
def test_an_environment_stop_after_a_server_started_voids_the_session(tmp_path, world_kw, rule):
    root, plan = formal_setup(tmp_path)
    s, _ = session(root, plan, mode="formal", world=World(tmp_path / "wf", **world_kw), sid="f")
    assert s["status"] == "finished" and s["stopped"]["rule"] == rule and s["stopped"]["boot"] == 2
    assert [a["valid"] for a in pp.formal_attempts(root)] == [False]
    assert any("environment" in p for p in an.session_problems(sdir(root, "f"), pins=PINS, cfg=frozen(), root=root))


def test_an_interrupt_before_the_first_server_is_refused_not_an_attempt(tmp_path):
    root, plan = formal_setup(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        session(root, plan, mode="formal", world=World(tmp_path / "wf", env_fail="interrupt"), sid="f")
    saved = json.loads((sdir(root, "f") / "session.json").read_text())
    assert saved["status"] == "refused" and "KeyboardInterrupt" in saved["aborted"] and pp.formal_attempts(root) == []


def test_aborted_stub_and_unreadable_formal_sessions_are_void_attempts(tmp_path):
    root, plan = formal_setup(tmp_path)
    with pytest.raises(r1p.RunError, match="harness failure"):
        session(root, plan, mode="formal", world=World(tmp_path / "wa", behave={"b1-n2": "raise"}), sid="a")
    assert json.loads((sdir(root, "a") / "session.json").read_text())["status"] == "aborted"
    assert [a["valid"] for a in pp.formal_attempts(root)] == [False]
    stub = sdir(root, "stub")                       # a hard kill leaves the running stub and the plan
    stub.mkdir()
    (stub / "plan.json").write_text(json.dumps(plan))
    (stub / "session.json").write_text(json.dumps({"stage": "E1P", "mode": "formal", "status": "running",
                                                   "started_at": "z"}))
    assert [a["valid"] for a in pp.formal_attempts(root)] == [False, False]
    with pytest.raises(r1p.RunError, match="already exist"):           # one void attempt and one rerun, no more
        session(root, plan, mode="formal", sid="c")
    with pytest.raises(an.AnalysisError, match="no valid"):
        an.first_valid_session(root)
    other = tmp_path / "other" / "results" / "stage_e"
    (other / "x").mkdir(parents=True)
    (other / "x" / "plan.json").write_text(json.dumps(plan))
    (other / "x" / "session.json").write_text("{torn")                  # unreadable: void (fail closed)
    (other / "y").mkdir()
    (other / "y" / "plan.json").write_text(json.dumps(plan))           # a plan saved alone ran nothing
    assert [a["valid"] for a in pp.formal_attempts(tmp_path / "other")] == [False]


@pytest.mark.parametrize("how, why", [("no_P", "F is not computable"), ("dup_span", "duplicate")])
def test_a_main_request_without_f_or_a_duplicated_span_invalidates_its_bench(tmp_path, how, why):
    s, root, _ = smoke(tmp_path, world=World(tmp_path / "w", behave={"b1-n2": how}))
    r = rows_of(root, "smoke-e1p")[1]
    assert r["ok"] is False and r["f_computable"] is False and any(why in x for x in r["reasons"])
    assert s["smoke"]["checks"]["f_computable"] is False


def test_a_halt_after_a_bench_or_during_the_teardown_is_still_a_stop(tmp_path):
    root, plan = formal_setup(tmp_path)
    s, _ = session(root, plan, mode="formal", world=World(tmp_path / "w1", behave={"b1-n2": "hot_after"}), sid="f1")
    rows = {r["tag"]: r for r in rows_of(root, "f1")}
    assert s["stopped"]["rule"] == "temperature" and rows["b1-n2"]["ok"] is False
    assert any("no 031 spill reading" in x for x in rows["b1-n2"]["reasons"]) and rows["b1-n3"]["not_run"] is True
    root2, plan2 = formal_setup(tmp_path / "two")
    s, _ = session(root2, plan2, mode="formal", world=World(tmp_path / "w2", hot_at_teardown=True), sid="f")
    assert s["stopped"]["rule"] == "temperature" and "teardown of b1" in s["stopped"]["reason"]
    assert all(r["ok"] for r in rows_of(root2, "f")) and len(rows_of(root2, "f")) == 4


def test_two_void_attempts_end_the_formal_runs(tmp_path):
    root, plan = formal_setup(tmp_path)
    for sid in ("v1", "v2"):
        s, _ = session(root, plan, mode="formal", sid=sid, lock_held_check=lock_stop_after(2))
        assert s["stopped"]["rule"] == "lock" and s["status"] == "finished"
    assert [a["valid"] for a in pp.formal_attempts(root)] == [False, False]
    with pytest.raises(r1p.RunError, match="already exist"):
        session(root, plan, mode="formal", sid="v3")


def test_the_per_bench_quantities_use_e1ds_frozen_functions():
    reqs = [{"idx": i, "F_ms": 50.0 if i in (16, 40) else 10.0, "arrival": 100 + 0.2 * i, "server_ttft_ms": 400.0,
             "P_ms": 300.0, "client_ttft_ms": 1000.0 * t, "Q_ms": 10.0, "e2e_ms": 7000.0, "start_time": 1 + 0.2 * i,
             "latency_ms": 7000.0} for i, t in enumerate(tc0.cycle_s(200, (8,)))]
    steps = [{"t": 100 + 0.1 * j, "index": j, "ctx_tok": 2048 if j % 3 == 0 else 0, "ctx_req": 0, "gen_tok": 7,
              "gen_req": 7, "elapsed_ms": 30.0, "u": 40.0} for j in range(450)]
    full = r1p.bench_e1(reqs, steps, E1CFG, full=True)
    assert full["f_spikes"] == se1.f_spikes(reqs, steps, E1CFG) and full["f_spikes"]["spike_groups"] == [2, 5]
    assert [x["k"] for x in full["episodes"]] == [8] and full["episodes"][0]["links"]["L1"] is False
    assert r1p.bench_e1(reqs[:16], steps, E1CFG, full=False)["episodes"] is None
    empty = r1p.bench_e1(reqs, [], E1CFG, full=True)                    # no steps in the window: L2 not evaluated
    assert empty["f_spikes"]["n_dec_anchor"] is None and empty["episodes"][0]["links"]["L2"] is None
    aligned = {"requests": [{"idx": 0, "e2e": 1.5, "F": 0.01}], "missing": [1], "duplicates": [], "unexpected": []}
    assert r1p.span_problems(aligned, [{"F_ms": None}], 2)


# ---------------------------------------------------------------- reading

def summary(repeated, spikes=20):
    return {"repeated": [[i, 3] for i in repeated], "spikes": spikes}


@pytest.mark.parametrize("a, b, spikes, row", [
    ((32, 64), (40,), 4, "few_spikes"), ((32, 64), (), 20, "few_spikes"),
    ((40, 48), (40, 48), 20, "prior_not_reproduced"), ((32, 64, 88), (40, 120), 20, "content_locked"),
    ((32, 64, 88), (32, 64, 120), 20, "position_locked"), ((32, 64, 88), (32, 64, 120, 128), 20, "undetermined")])
def test_the_reading_table_in_order(a, b, spikes, row):
    got = an.reading({"A": summary(a, spikes), "B": summary(b)}, CFG)
    assert got["row"] == row and got["text"] == an.READINGS[row] and got["descriptive_only"] is True
    assert got["J"] == an.jaccard(set(a), set(b))


def test_jaccard_and_spike_idx():
    assert an.jaccard({1, 2}, {2, 3}) == 1 / 3 and an.jaccard(set(), set()) is None
    reqs = [{"F_ms": 36.0 if i in (0, 8, 192, 199) else 35.0} for i in range(200)]
    assert an.spike_idx(reqs, 35.0) == [8, 192]                         # position 0 of k 1-24 only


def test_the_analysis_reads_the_first_valid_formal_session(tmp_path):
    root, _ = formal_setup(tmp_path)
    plan = pp.build_plan(PINS, frozen(), mode="formal", root=root, pins_sha256="p" * 64)
    world = World(tmp_path / "wf", slips={"b1-n1": (8,), "b4-n2": (8,)})
    session(root, plan, mode="formal", world=world, sid="f")
    path = an.first_valid_session(root)
    assert path.name == "f" and an.session_problems(path, pins=PINS, cfg=frozen(), root=root) == []
    assert an.session_problems(path, pins=PINS, cfg=frozen(), root=root, pins_sha256="p" * 64) == []
    assert any("pins_sha256" in p for p in an.session_problems(path, pins=PINS, cfg=frozen(), root=root,
                                                                pins_sha256="q" * 64))
    saved = json.loads((path / "session.json").read_text())
    (path / "session.json").write_text(json.dumps({**saved, "pins_sha256": "q" * 64}))   # the session's own record
    assert any("pins_sha256" in p for p in an.session_problems(path, pins=PINS, cfg=frozen(), root=root,
                                                                pins_sha256="p" * 64))
    (path / "session.json").write_text(json.dumps(saved))
    out = an.analyze(path, frozen(), E1CFG)
    assert json.loads(json.dumps(out)) == out                         # no int keys turn into strings
    assert out["reading"]["row"] == "content_locked" and out["reading"]["J"] == 0.0 and "note" not in out["reading"]
    assert out["reading"]["A_in"] == [32, 64, 88] and out["reading"]["B_in"] == [40, 120, 160]
    a = out["seeds"]["A"]
    assert a["benches"] == 12 and a["spikes"] == 36 and a["repeated"] == [[32, 12], [64, 12], [88, 12]]
    assert a["slips_by_position"][0] == {"position": 1, "benches": 3, "episodes": 1}
    assert a["spike_slip"] == {"spikes": 36, "spikes_in_slip_groups": 2, "slip_groups": 2, "slip_groups_with_spike": 2}
    assert a["episodes"] == 2 and a["chain"]["start_idx"] == [[65, 2]]
    assert out["slip_rate_all"] == 2 / 24 and len(out["inputs_sha256"]) == 3 + 48
    assert out["data"]["planned"] == out["data"]["valid"] == {"A": 12, "B": 12} and out["data"]["stopped"] is None
    assert out["data"]["insufficient_data"] is False and out["data"]["incomplete_session"] is False
    assert out["stage_e1_harness_sha256"] == E1CFG["freeze"]["e1_harness_sha256"]
    same = World(tmp_path / "ws", spikes={"A": (32, 64, 88), "B": (32, 64, 88)})
    root2, plan2 = formal_setup(tmp_path / "two")
    session(root2, plan2, mode="formal", world=same, sid="f")
    assert an.analyze(an.first_valid_session(root2), frozen(), E1CFG)["reading"]["row"] == "position_locked"


def test_a_session_stopped_with_too_few_valid_benches_says_so(tmp_path):
    root, plan = formal_setup(tmp_path)
    no_b = {f"b{b}-n{j}": "no_spans" for b, js in ((1, (2, 3)), (2, (1, 4))) for j in js}
    s, _ = session(root, plan, mode="formal", world=World(tmp_path / "wf", behave=no_b), sid="f")
    assert s["stopped"]["rule"] == "failed_boots"
    out = an.analyze(an.first_valid_session(root), frozen(), E1CFG)
    d = out["data"]
    assert d["stopped"]["rule"] == "failed_boots" and d["valid"] == {"A": 4, "B": 0} and d["planned"]["B"] == 12
    assert d["insufficient_data"] is True and d["incomplete_session"] is True
    assert [x["tag"] for x in d["excluded"]["B"]][:2] == ["b1-n2", "b1-n3"] and len(d["excluded"]["B"]) == 12
    never = [x for x in d["excluded"]["A"] if x["reasons"][0].startswith("its boot never started")]
    assert len(d["excluded"]["A"]) == len(never) == 8 and "2 failed boots" in never[0]["reasons"][0]
    assert all(x["not_run"] is True for x in never)
    assert out["reading"]["row"] == "few_spikes" and "too little data" in out["reading"]["note"]


def test_insufficient_data_means_fewer_valid_benches_than_repeat_min():
    plan = {"boots": [{"benches": [{"tag": f"t{i}", "seed_label": "AB"[i % 2]} for i in range(24)]}]}
    rows = [{"tag": f"t{i}", "seed_label": "AB"[i % 2], "ok": i % 2 == 1 or i < 6, "reasons": []} for i in range(24)]
    d = an.data_summary({"stopped": None, "outcomes": {}}, plan, rows, CFG)
    assert d["valid"] == {"A": 3, "B": 12} and d["insufficient_data"] is False and d["incomplete_session"] is True
    rows[4]["ok"] = False
    assert an.data_summary({"stopped": None}, plan, rows, CFG)["insufficient_data"] is True


def test_first_valid_session_allows_one_void_attempt_only(tmp_path):
    base = tmp_path / "results" / "stage_e"

    def attempt(sid, started, valid):
        (base / sid).mkdir(parents=True)
        (base / sid / "session.json").write_text(json.dumps(
            {"stage": "E1P", "mode": "formal", "status": "finished", "env_valid": valid, "started_at": started}))

    attempt("v1", "1", False)
    with pytest.raises(an.AnalysisError, match="no valid"):
        an.first_valid_session(tmp_path)
    attempt("ok", "2", True)
    assert an.first_valid_session(tmp_path).name == "ok"
    attempt("v0", "0", False)
    with pytest.raises(an.AnalysisError, match="at most one rerun"):
        an.first_valid_session(tmp_path)


def test_the_analysis_refuses_changed_missing_or_inconsistent_inputs(tmp_path):
    root, plan = formal_setup(tmp_path)
    with pytest.raises(an.AnalysisError, match="no valid"):
        an.first_valid_session(root)
    session(root, plan, mode="formal", sid="f")
    path = sdir(root, "f")
    pristine = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}

    def restore():
        for p, data in pristine.items():
            p.write_bytes(data)

    d = path / "derived" / "b3-n2-requests.jsonl"
    d.write_text(d.read_text().replace('"F_ms": 10.0', '"F_ms": 99.0', 1))
    with pytest.raises(an.AnalysisError, match="b3-n2"):
        an.analyze(path, frozen(), E1CFG)
    restore()
    (path / "derived" / "b2-n1-e1.json").unlink()
    with pytest.raises(an.AnalysisError, match="b2-n1.*missing"):
        an.analyze(path, frozen(), E1CFG)
    restore()
    e = path / "derived" / "b4-n3-e1.json"                          # the runner's spikes against the derived F
    doc = json.loads(e.read_text())
    doc["f_spikes"]["spike_groups"] = [1, 2]
    e.write_text(json.dumps(doc))
    rows = rows_of(root, "f")
    for r in rows:
        if r["tag"] == "b4-n3":
            r["e1_sha256"] = hashlib.sha256(e.read_bytes()).hexdigest()
    (path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(an.AnalysisError, match="differ from"):
        an.analyze(path, frozen(), E1CFG)
    assert any("rows_sha256" in p for p in an.session_problems(path, pins=PINS, cfg=frozen(), root=root))
    restore()
    assert an.session_problems(path, pins=PINS, cfg=frozen(), root=root) == []
    assert any("changed" in p for p in an.session_problems(
        path, pins=PINS, cfg={**frozen(), "e1p": {**CFG["e1p"], "boots": 5}}, root=root))
    assert an.out_problems(sdir(root, "f") / "x.json", root=root)
    assert an.out_problems(sdir(root, "new") / "x.json", root=root) == []
