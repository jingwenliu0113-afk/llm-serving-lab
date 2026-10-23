"""Tests for the Stage E1-P harness (decision 041 v3 §1.6-1.7): the plans and the freeze evidence (stage_e1p_plan),
whole sessions against a fake WSL (run_stage_e1p: server log, /metrics, OTLP spans, the iteration log, `vllm bench
serve` with its seed, the nvidia-smi sampler, 031 idle and spill readings; no GPU), the smoke's checks and masking,
the stop rules, the rerun limit, and the reading (stage_e1p_analysis)."""

import json
import tomllib
from pathlib import Path

import pytest

import adv_plan as ap
import core_b_plan as bp
import run_stage_e1p as r1p
import stage_e1 as se1
import stage_e1p_analysis as an
import stage_e1p_plan as pp
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


class Receiver:
    def __init__(self, world):
        self.world, self.records, self.errors = world, [], []

    def start(self):
        self.world.receiver = self
        return self

    def close(self):
        return []

    def raise_if_failed(self):
        pass

    def read_spans(self):
        return list(self.records)


class World(td.World):
    """td.World's clock, idle and spill readings with an observed-B0 server. spikes[seed label]: idx whose F is 50 ms
    (else 10 ms); slips[tag]: slipped groups (c0 test's cycle); behave[tag] in {"timeout", "hot", "exit", "unhealthy",
    "no_spans", "short_lens", "no_steps"}; never_ready: boot labels whose /health never answers ("*": all);
    nd_edit: edits of the logged non-default args; smi_temp: the sampler's temperature (None: never reported)."""

    def __init__(self, tmp_path, *, spikes=SPIKES, slips=None, behave=None, never_ready=(), nd_edit=None,
                 smi_temp=66.0, bench_s=None, spill_at=()):
        super().__init__(tmp_path, spill_at=spill_at)
        self.spikes, self.slips, self.behave = spikes, slips or {}, behave or {}
        self.never_ready, self.nd_edit, self.smi_temp, self.bench_s = set(never_ready), nd_edit or {}, smi_temp, \
            bench_s
        self.step, self.receiver, self.server_log, self.smi, self.unhealthy = 0, None, None, None, False
        self.smi_samples, self.bench_argvs = [], []

    def wall(self):
        return 1.79e9 + self.t

    def start_smi(self, path):
        smi = tc0.FakeSmi(self, path)
        self.smi_samples = [{"t": self.wall(), **SMI, "temp_c": self.smi_temp}]
        return smi

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

    def run_bench(self, argv, log_path, env, timeout_s, stop):
        self.bench_argvs.append(argv)
        a = dict(zip(argv, argv[1:]))
        n, seed, prefix = int(a["--num-prompts"]), int(a["--seed"]), a["--request-id-prefix"]
        tag = Path(a["--result-filename"]).name.removesuffix(".bench.json")
        how = self.behave.get(tag)
        Path(log_path).write_text("log\n")
        t0 = self.wall()
        if how == "timeout":
            self.t += timeout_s
            return -9, True, False
        if how == "hot":
            self.smi.flag_hot()
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
        self.receiver.records += recs + [{"id": "cmpl-warmup-0", "arrival": t0, "Q": q, "P": p, "server_ttft": 1.0}]
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
        return 0, False, False


def setup(tmp_path):
    root = tmp_path / "repo"
    (root / "results" / "stage_e").mkdir(parents=True)
    return root


def session(root, plan, *, mode, world=None, sid="s", **kw):
    w = world or World(root.parent / f"w-{sid}")
    stamp = iter(range(10 ** 6))
    args = dict(pins=PINS, pins_sha256="p" * 64, out_dir=root / "results" / "stage_e" / sid, session_id=sid,
                mode=mode, expected_plan=plan, popen=w.popen, signal_group=w.signal_group, group_alive=w.group_alive,
                http_get=w.http_get, run_bench=w.run_bench, gpu_sample=w.gpu_sample, compute_apps=w.compute_apps,
                shared_usage=w.shared_usage, start_smi=w.start_smi, receiver_factory=lambda raw, spans, pre:
                Receiver(w), lock_held_check=lambda: True, gpu_lock_held=True, capture_env=tre2.Env(), t_start=w.now(),
                vllm_bin="/v", base_env=BASE_ENV, toolchain_check=lambda env: [], sample_interval=None, now=w.now,
                sleep=w.sleep, wall=w.wall, clock=lambda: f"2026-10-24T{sid}-{next(stamp):06d}", repo_root=root)
    args.update(kw)
    return r1p.run_session(plan, **args), w


def smoke(tmp_path, world=None, sid="smoke-e1p", **kw):
    root = setup(tmp_path)
    s, w = session(root, pp.build_plan(PINS, CFG, mode="smoke"), mode="smoke", world=world, sid=sid, **kw)
    return s, root, w


def frozen(sid="smoke-e1p"):
    return {**CFG, "freeze": {"e1p": True, "smoke_e1p": sid}}


def formal_setup(tmp_path):
    s, root, _ = smoke(tmp_path)
    assert s["smoke"]["ok"] is True, s["smoke"]["checks"]
    return root, pp.build_plan(PINS, frozen(), mode="formal", root=root)


def rows_of(root, sid):
    return [json.loads(x) for x in (root / "results" / "stage_e" / sid / "rows.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- config and plans

def test_the_repo_config_is_041s_and_unfrozen():
    assert pp.config_problems(CFG) == [] and CFG["freeze"] == {"e1p": False, "smoke_e1p": ""}
    assert pp.config_problems(frozen()) == []
    assert pp.config_problems({**CFG, "freeze": {"e1p": True, "smoke_e1p": ""}})
    for key, bad in (("seeds", {"A": 20261008, "B": 20261008}), ("order_even", ["A", "B", "B", "A"]),
                     ("f_spike_ms", 35), ("cap_hours", 3.5), ("a_prior", [32, 64]), ("j_content_max", 0.25)):
        assert any(key in p for p in pp.config_problems({**CFG, "e1p": {**CFG["e1p"], key: bad}})), key
    assert pp.config_problems({**CFG, "e1p": {**CFG["e1p"], "extra": 1}})
    assert pp.e1_config()["freeze"]["e1"] is True and se1.config_problems(pp.e1_config()) == []


def test_a_plan_pins_the_configs_content_without_freeze():
    a, b = frozen(), CFG
    assert pp.config_sha256(a) == pp.config_sha256(b) != pp.config_sha256({**b, "e1p": {**b["e1p"], "boots": 5}})
    p = pp.build_plan(PINS, CFG, mode="smoke")
    assert p["stage_e1p_sha256"] == pp.config_sha256(CFG) and p["harness_sha256"] == pp.harness_sha256()
    assert p["stage_e1_sha256"] == pp.sha256_json(E1CFG)
    assert set(pp.HARNESS_FILES) >= {"benchmarks/run_c0_probe.py", "benchmarks/c0_analysis.py"}


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


def test_the_smoke_plan_is_one_boot_of_a_then_b_at_n_16():
    p = pp.build_plan(PINS, CFG, mode="smoke")
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
    smoke_s = json.loads((root / "results" / "stage_e" / "smoke-e1p" / "session.json").read_text())
    assert p["smoke"] == {"session": "smoke-e1p", "elapsed_s": smoke_s["elapsed_s"]}
    assert p["cap_s"] == int(3.0 * 3600 - smoke_s["elapsed_s"]) and 0 < smoke_s["elapsed_s"] < 3600
    with pytest.raises(pp.PlanError, match="freeze.e1p"):
        pp.build_plan(PINS, CFG, mode="formal", root=root)
    with pytest.raises(pp.PlanError, match="smoke first"):
        pp.build_plan(PINS, frozen("none"), mode="formal", root=root)


def test_the_smoke_evidence_must_be_this_designs_passed_smoke(tmp_path):
    root, _ = formal_setup(tmp_path)
    d = root / "results" / "stage_e" / "smoke-e1p"
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


# ---------------------------------------------------------------- smoke sessions

def test_a_smoke_runs_seed_a_then_b_checks_041s_conditions_and_masks_the_latencies(tmp_path):
    s, root, w = smoke(tmp_path)
    sm = s["smoke"]
    assert sm["ok"] is True, sm["checks"]
    assert s["status"] == "finished" and s["outcomes"]["valid"] == 2 and s["stopped"] is None
    seeds = [a[a.index("--seed") + 1] for a in w.bench_argvs]
    assert seeds == ["20261008", "20261023"] and {a[a.index("--num-prompts") + 1] for a in w.bench_argvs} == {"16"}
    assert [a[-1] for a in w.bench_argvs] == ["e1p-b1-n1-", "e1p-b1-n2-"] and len(w.server_argvs) == 1
    out = root / "results" / "stage_e" / "smoke-e1p"
    for pattern in r1p.SMOKE_LATENCY_RAW + ("*.otlp",):
        assert not list((out / "raw").glob(pattern)), pattern
    assert not list((out / "derived").glob("*-e1.json"))
    req = (out / "derived" / "b1-n1-requests.jsonl").read_text().splitlines()
    assert len(req) == 16 and set(json.loads(req[0])) == {"idx", "cls"}
    rows = rows_of(root, "smoke-e1p")
    assert all(set(r) <= set(r1p.MASK_ROW_KEYS) for r in rows) and all(r["ok"] for r in rows)
    saved = json.loads((out / "session.json").read_text())
    assert not any(k in boot for boot in saved["boots"].values() for k in ("startup_s", "t_start", "t_end"))
    assert saved["elapsed_s"] == s["elapsed_s"] > 0 and (out / "smoke.json").is_file()
    assert (out / "derived" / "b1-smi.jsonl").is_file() and saved["boots"]["b1"]["smi_summary"]["temp_max_c"] == 66.0


@pytest.mark.parametrize("world_kw, failed", [
    (dict(behave={"b1-n2": "no_spans"}), "spans"), (dict(behave={"b1-n2": "no_spans"}), "f_computable"),
    (dict(behave={"b1-n2": "short_lens"}), "input_lens_8064"), (dict(behave={"b1-n1": "no_steps"}), "steps_in_window"),
    (dict(smi_temp=None), "smi"), (dict(behave={"b1-n1": "exit"}), "every_bench_valid")])
def test_each_smoke_condition_can_fail_it(tmp_path, world_kw, failed):
    s, _, _ = smoke(tmp_path, world=World(tmp_path / "w", **world_kw))
    assert s["smoke"]["ok"] is False and s["smoke"]["checks"][failed] is False


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
    out = root / "results" / "stage_e" / "f"
    e1 = json.loads((out / "derived" / "b1-n1-e1.json").read_text())
    assert e1["f_spikes"]["spike_groups"] == [4, 8, 11]                         # seed A at 32, 64, 88
    assert [x["start_idx"] for x in e1["episodes"]] == [65] and e1["episodes"][0]["links"]["L1"] is True
    b = json.loads((out / "derived" / "b1-n2-e1.json").read_text())
    assert b["f_spikes"]["spike_groups"] == [5, 15, 20]                         # seed B at 40, 120, 160
    r = next(x for x in rows if x["tag"] == "b1-n1")
    assert r["spans_main"] == 200 and r["n_steps_window"] > 0 and r["input_lens_8064"] is True
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
        assert not (root / "results" / "stage_e" / "g").exists()
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


def test_no_server_started_is_refused_and_an_environment_stop_voids_the_session(tmp_path):
    root, plan = formal_setup(tmp_path)
    busy = World(tmp_path / "w1")
    busy.compute_apps = lambda: [{"pid": 9, "name": "other", "used_mib": 3000}]
    s, _ = session(root, plan, mode="formal", world=busy, sid="f1")
    assert s["status"] == "refused" and s["stopped"]["rule"] == "gpu_not_idle" and pp.formal_attempts(root) == []
    w = World(tmp_path / "w2")
    calls = iter([True] * 3 + [False] * 99)          # the session, boot 1, boot 2; boot 3 stops
    s, _ = session(root, plan, mode="formal", world=w, sid="f2", lock_held_check=lambda: next(calls))
    assert s["status"] == "finished" and s["stopped"]["rule"] == "lock"
    assert [a["valid"] for a in pp.formal_attempts(root)] == [False]
    s, _ = session(root, plan, mode="formal", sid="f3")
    assert s["stopped"] is None and [a["valid"] for a in pp.formal_attempts(root)] == [False, True]


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
    aligned = {"requests": [{"idx": 0, "e2e": 1.5, "F": 0.01}], "missing": [1], "duplicates": [], "unexpected": []}
    assert r1p.span_problems(aligned, [{"F_ms": None}], 2)


# ---------------------------------------------------------------- reading

def summary(repeated, spikes=20):
    return {"repeated": dict.fromkeys(repeated, 3), "spikes": spikes}


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
    root, plan = formal_setup(tmp_path)
    world = World(tmp_path / "wf", slips={"b1-n1": (8,), "b4-n2": (8,)})
    session(root, plan, mode="formal", world=world, sid="f")
    path = an.first_valid_session(root)
    assert path.name == "f" and an.session_problems(path, pins=PINS, cfg=frozen(), root=root) == []
    out = an.analyze(path, frozen(), E1CFG)
    assert out["reading"]["row"] == "content_locked" and out["reading"]["J"] == 0.0
    assert out["reading"]["A_in"] == [32, 64, 88] and out["reading"]["B_in"] == [40, 120, 160]
    a = out["seeds"]["A"]
    assert a["benches"] == 12 and a["spikes"] == 36 and a["repeated"] == {32: 12, 64: 12, 88: 12}
    assert a["slips_by_position"][1] == {"benches": 3, "episodes": 1} and a["spike_slip"]["slips_with_spike"] == 2
    assert out["slip_rate_all"] == 2 / 24 and len(out["inputs_sha256"]) == 3 + 48
    same = World(tmp_path / "ws", spikes={"A": (32, 64, 88), "B": (32, 64, 88)})
    root2, plan2 = formal_setup(tmp_path / "two")
    session(root2, plan2, mode="formal", world=same, sid="f")
    assert an.analyze(an.first_valid_session(root2), frozen(), E1CFG)["reading"]["row"] == "position_locked"


def test_the_analysis_refuses_changed_inputs_and_other_sessions(tmp_path):
    root, plan = formal_setup(tmp_path)
    with pytest.raises(an.AnalysisError, match="no valid"):
        an.first_valid_session(root)
    session(root, plan, mode="formal", sid="f")
    path = root / "results" / "stage_e" / "f"
    d = path / "derived" / "b3-n2-requests.jsonl"
    d.write_text(d.read_text().replace('"F_ms": 10.0', '"F_ms": 99.0', 1))
    with pytest.raises(an.AnalysisError, match="b3-n2"):
        an.analyze(path, frozen(), E1CFG)
    (path / "rows.jsonl").write_text((path / "rows.jsonl").read_text() + "\n")
    assert any("rows_sha256" in p for p in an.session_problems(path, pins=PINS, cfg=frozen(), root=root))
    assert any("changed" in p for p in an.session_problems(
        path, pins=PINS, cfg={**frozen(), "e1p": {**CFG["e1p"], "boots": 5}}, root=root))
    assert an.out_problems(root / "results" / "stage_e" / "f" / "x.json", root=root)
    assert an.out_problems(root / "results" / "stage_e" / "new" / "x.json", root=root) == []
