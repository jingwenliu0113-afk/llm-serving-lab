"""Tests for benchmarks/run_stage_e2.py (decision 041 v3 §2.3-2.10): whole E2 sessions against a fake WSL (server
log, /metrics, OTLP receiver, iteration log, `vllm bench serve`, 031 idle and spill readings; no GPU): the smoke's
pass conditions and masking, the plan binding, the stop rules, the rerun limit and the per-row measurements."""

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
import run_stage_e2 as re2
import stage_e2_plan as e2
import test_run_stage_d as td

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_e2.toml").read_text())
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u"}
IDLE_MIB = 567


class Receiver:
    def __init__(self, world, prefix):
        self.world, self.prefix, self.errors, self.records = world, prefix, [], []

    def start(self):
        self.world.receiver = self
        return self

    def close(self):
        return []

    def raise_if_failed(self):
        pass

    def main_count(self):
        return len(self.records)

    def stats(self):
        return {"main": len(self.records)}

    def read_spans(self):
        return list(self.records)


class World(td.World):
    """td.World's clock, idle and spill readings with an E2 server: its non-default args, blocks, prefix caching and
    mamba_cache_mode follow the argv; drop[label] removes a key from its log; no_hits / hits_off steer the prefix
    cache counter; server_mib is the server's memory after a bench; never_ready lists labels whose /health never
    answers ("*": all); no_queue_hist / no_mamba drop those /metrics series; cached(pc, i) adds a cached-token span
    attribute (decoded spans)."""

    def __init__(self, tmp_path, *, drop=None, no_hits=False, hits_off=False, spill_at=(), server_mib=15000,
                 never_ready=(), no_queue_hist=False, no_mamba=False, cached=None):
        super().__init__(tmp_path, spill_at=spill_at)
        self.drop, self.no_hits, self.hits_off, self.server_mib = drop or {}, no_hits, hits_off, server_mib
        self.never_ready, self.no_queue_hist, self.no_mamba = set(never_ready), no_queue_hist, no_mamba
        self.cached = cached
        self.pc, self.blocks, self.hits, self.receiver, self.server_log = False, 248, 0.0, None, None
        self.q_count, self.q_sum, self.prompt_tokens = 0, 0.0, 0

    def wall(self):
        return 1.79e9 + self.t

    def gpu_sample(self):
        return {"mem_used_mib": IDLE_MIB + (0 if self.point == "pre" else self.server_mib), "util_pct": 0,
                "pstate": "P8"}

    def popen(self, argv, env, log_path):
        log_path = Path(log_path)
        label = log_path.name.removesuffix(".server.log")
        self.label, self.point, self.server_log = label, f"ready:{label}", log_path
        self.server_argvs.append(argv)
        a = dict(zip(argv, argv[1:]))
        self.pc = "--enable-prefix-caching" in argv
        self.blocks = 155 if a["--kv-cache-memory-bytes"] == "2684354560" else 248
        nd = {"max_num_batched_tokens": int(a["--max-num-batched-tokens"]),
              "kv_cache_memory_bytes": int(a["--kv-cache-memory-bytes"]), "enable_prefix_caching": self.pc,
              "otlp_traces_endpoint": a["--otlp-traces-endpoint"], "enable_logging_iteration_details": True}
        if "--long-prefill-token-threshold" in a:
            nd["long_prefill_token_threshold"] = int(a["--long-prefill-token-threshold"])
        for key in self.drop.get(label, ()):
            nd.pop(key, None)
        log_path.write_text(f"{int(self.wall() * 1000)} (APIServer pid=1) INFO non-default args: {nd!r}\n"
                            f"{int(self.wall() * 1000)} (EngineCore pid=2) INFO [cuda.py:478] Using "
                            "AttentionBackendEnum.FLASH_ATTN backend.\n")
        self.threshold = nd.get("long_prefill_token_threshold")
        self.proc = td.Proc(77)
        return self.proc

    def http_get(self, url):
        alive = self.proc is not None and self.proc.returncode is None
        if url.endswith("/health"):
            if "*" in self.never_ready or self.label in self.never_ready:
                return 0, "refused"
            return (200, "") if alive else (0, "refused")
        lab = 'engine="0",model_name="m"'
        mamba = "" if self.no_mamba else f'mamba_cache_mode="{"align" if self.pc else "none"}",'
        text = (f'vllm:cache_config_info{{block_size="528",enable_prefix_caching="{self.pc}",engine="0",{mamba}'
                f'num_gpu_blocks="{self.blocks}"}} 1.0\n'
                f"vllm:num_requests_running{{{lab}}} 3.0\nvllm:num_requests_waiting{{{lab}}} 2.0\n"
                f"vllm:kv_cache_usage_perc{{{lab}}} 0.4\nvllm:num_preemptions_total{{{lab}}} 0.0\n"
                f"vllm:prefix_cache_hits_total{{{lab}}} {self.hits}\n"
                f"vllm:prompt_tokens_total{{{lab}}} {self.prompt_tokens}\n")
        if not self.no_queue_hist:
            q = "vllm:request_queue_time_seconds"
            text += (f'{q}_bucket{{{lab},le="+Inf"}} {self.q_count}\n{q}_sum{{{lab}}} {self.q_sum}\n'
                     f"{q}_count{{{lab}}} {self.q_count}\n")
        return 200, text

    def run_bench(self, argv, log_path, env, timeout_s):
        a = dict(zip(argv, argv[1:]))
        n, warm = int(a["--num-prompts"]), int(a["--num-warmups"])
        prefix = a["--request-id-prefix"]
        dataset = Path(a["--dataset-path"]).name if "--dataset-path" in a else ""
        mixed = dataset.startswith("mixed")
        Path(log_path).write_text("log\n")
        t0 = self.wall()
        lens = [bp.LONG_INPUT if mixed and i % 10 == 5 else 1920 for i in range(n)]
        ttfts = [(5.0 if x == bp.LONG_INPUT else 0.9) for x in lens]
        lats = [t + 127 * 0.05 for t in ttfts]
        res = {"completed": n, "failed": 0, "duration": 10.0, "request_throughput": n / 10.0, "output_throughput": 300.0,
               "total_token_throughput": 1e5, "input_lens": lens, "output_lens": [128] * n, "ttfts": ttfts,
               "latencies": lats, "start_times": [1000.0 + 0.05 * i for i in range(n)]}
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(res))
        thr = self.threshold or 2048
        with open(self.server_log, "a") as fh:
            for i in range(5):
                fh.write(f"{int((t0 + 1 + i) * 1000)} (EngineCore pid=2) INFO Iteration({i}): 1 context requests, "
                         f"{thr} context tokens, 7 generation requests, 7 generation tokens, iteration elapsed time: "
                         f"50.0 ms, GPU KV cache usage: 20.0%\n")
        recs = [{"id": f"cmpl-{prefix}{i}-0", "arrival": t0 + 0.5 + 0.01 * i, "Q": 0.1, "P": 0.5,
                 "server_ttft": ttfts[i], "e2e": lats[i], "prompt_tokens": lens[i], "completion_tokens": 128}
                for i in range(n)]
        if self.cached is not None:                       # decoded spans with a cached-token attribute
            recs = [{"name": "llm_request", "start_time_unix_nano": round(r["arrival"] * 1e9),
                     "attributes": {"gen_ai.request.id": r["id"], "gen_ai.latency.time_in_queue": r["Q"],
                                    "gen_ai.latency.time_in_model_prefill": r["P"],
                                    "gen_ai.latency.time_to_first_token": r["server_ttft"],
                                    "gen_ai.latency.e2e": r["e2e"], "gen_ai.usage.prompt_tokens": r["prompt_tokens"],
                                    "gen_ai.usage.completion_tokens": 128,
                                    "gen_ai.usage.cached_tokens": self.cached(self.pc, i)}}
                    for i, r in enumerate(recs)]
        self.receiver.records = recs
        self.q_count, self.q_sum = self.q_count + n, self.q_sum + 0.1 * n
        self.prompt_tokens += 5 * thr
        if (self.pc and not self.no_hits) or (not self.pc and self.hits_off):
            self.hits += (n + warm) * 1584
        self.point = f"bench:{self.label}"
        self.t += 10.0
        return 0, False


class Env(td.Env):
    def __call__(self, out_path, gpu_lock_held):
        rec = {**super().__call__(out_path, gpu_lock_held), "os": {"system": "Linux", "wsl": True}}
        Path(out_path).write_text(json.dumps(rec))
        return rec


def datasets(root):
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "mixed-2048.jsonl").write_text("mixed\n")
    (d / "shared-2048.jsonl").write_text("shared\n")
    return {"mixed:2048": str(d / "mixed-2048.jsonl"), "shared:2048": str(d / "shared-2048.jsonl")}


def setup(tmp_path):
    """(root, datasets, pins, cfg): the fake datasets pinned (mixed in pins, shared in the config), not frozen."""
    root = tmp_path / "repo"
    ds = datasets(root)
    pins = json.loads(json.dumps(PINS))
    pins["core_b"]["mix_dataset_sha256"]["2048"] = hashlib.sha256(Path(ds["mixed:2048"]).read_bytes()).hexdigest()
    sp = {**CFG["e2"]["shared_prefix"], "sha256": hashlib.sha256(Path(ds["shared:2048"]).read_bytes()).hexdigest(),
          "common_prefix": 1630}
    return root, ds, pins, {**CFG, "e2": {**CFG["e2"], "shared_prefix": sp}}


def session(root, ds, pins, cfg, plan, *, mode, world=None, sid="s", **kw):
    w = world or World(root.parent / f"w-{sid}")
    stamp = iter(range(10 ** 6))
    args = dict(pins=pins, cfg=cfg, pins_sha256="p" * 64, out_dir=root / "results" / "stage_e" / sid, session_id=sid,
                mode=mode, popen=w.popen, signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get,
                run_bench=w.run_bench, gpu_sample=w.gpu_sample, compute_apps=w.compute_apps,
                shared_usage=w.shared_usage, lock_held_check=lambda: True, gpu_lock_held=True, capture_env=Env(),
                datasets=ds, receiver_factory=lambda raw, spans, prefix: Receiver(w, prefix), vllm_bin="/v",
                base_env=BASE_ENV, toolchain_check=lambda env: [], sample_interval=None, now=w.now, sleep=w.sleep,
                wall=w.wall, clock=lambda: f"2026-10-24T{sid}-{next(stamp):06d}", repo_root=root,
                expected_plan=plan)
    args.update(kw)
    return re2.run_session(plan, **args), w


def smoke(tmp_path, world=None, sid="smoke-e2", **kw):
    root, ds, pins, cfg = setup(tmp_path)
    plan = e2.smoke_plan(pins, cfg)
    s, w = session(root, ds, pins, cfg, plan, mode="smoke", world=world, sid=sid, **kw)
    return s, root, w


def test_a_smoke_session_checks_every_condition_and_mix_and_masks_the_latencies(tmp_path):
    s, root, w = smoke(tmp_path)
    sm = s["smoke"]
    assert sm["ok"] is True, sm["checks"]
    assert s["outcomes"] == {"boots": 12, "valid": 12, "skipped": 0} and s["status"] == "finished"
    assert sorted(map(tuple, sm["covered"])) == sorted(e2.SMOKE) and set(sm["checks"]) >= {
        "a10", "teardown_clean", "mamba_cache_mode", "memory", "blocks", "covered"}
    assert sm["measured_blocks"] == {"B0": 248, "T": 248, "T256": 248, "K25": 155, "TK25": 155, "PC": 248}
    assert sm["mamba_cache_mode"] == {"PC-off": "none", "PC-on": "align"}
    assert set(sm["mem_increment_mib"].values()) == {15000} and sm["mem_limit_mib"] == 15157 + 128
    out = root / "results" / "stage_e" / "smoke-e2"
    for pattern in ("*.bench.json", "*.metrics.*.txt", "*.steps.jsonl", "*.series.jsonl"):
        assert not list((out / "raw").glob(pattern)), pattern
    saved = json.loads((out / "session.json").read_text())
    assert not any(k in run for run in saved["runs"].values() for k in ("t_before", "t_after", "startup_s"))
    assert not (out / "session.json.tmp").exists() and saved["status"] == "finished"
    pc_argv = next(a for a in w.server_argvs if "--enable-prefix-caching" in a)
    assert "--no-enable-prefix-caching" not in pc_argv


@pytest.mark.parametrize("world_kw, failed", [
    (dict(no_hits=True), "pc_on_hits"), (dict(hits_off=True), "pc_off_no_hits"),
    (dict(no_queue_hist=True), "a10"), (dict(no_mamba=True), "mamba_cache_mode"),
    (dict(server_mib=15300), "memory")])
def test_each_smoke_condition_can_fail_it(tmp_path, world_kw, failed):
    s, _, _ = smoke(tmp_path, world=World(tmp_path / "w", **world_kw))
    assert s["smoke"]["ok"] is False and s["smoke"]["checks"][failed] is False


def test_smoke_checks_need_full_coverage_and_a_clean_teardown(tmp_path):
    _, root, _ = smoke(tmp_path)
    rows = [{"cell": c, "mix": m, "cond": e2._cell(c)[1], "tag": f"{c}-{m}", "valid": True, "derived": "d",
             "mem_increment_mib": 100, "decomp": {"checks": {"a": True, "b": True, "c": True, "d": True}},
             "manip": {"pc_hits_ok": True, "chunk_ok": True, "num_gpu_blocks": 155 if "8k" in c else 248,
                       "block_size": 528, "mamba_cache_mode": "x"}} for c, m in e2.SMOKE]
    plan = json.loads((root / "results" / "stage_e" / "smoke-e2" / "plan.json").read_text())
    assert re2.smoke_checks(plan, rows, {})["ok"] is True
    assert re2.smoke_checks(plan, rows[:-3] + rows[-2:], {})["checks"]["covered"] is False
    assert re2.smoke_checks(plan, rows, {"teardown_problems": [{"boot": "x"}]})["checks"]["teardown_clean"] is False
    c_false = {"checks": {"a": True, "b": True, "c": False, "d": True}}
    assert re2.smoke_checks(plan, [dict(r, decomp=c_false) for r in rows], {})["checks"]["a10"] is False
    pc_only = [dict(r, decomp=c_false) if r["cond"] == "PC" else r for r in rows]  # (c) is exempt only for PC
    assert re2.smoke_checks(plan, pc_only, {})["checks"]["a10"] is True


def test_the_per_request_cached_tokens_decide_the_hit_check_when_present(tmp_path):
    hit = lambda pc, i: 1584 if pc else 0  # noqa: E731
    s, _, _ = smoke(tmp_path / "a", world=World(tmp_path / "wa", cached=hit))
    assert s["smoke"]["ok"] is True, s["smoke"]["checks"]
    assert {v for v in s["smoke"]["span_cached_key"].values()} == {"gen_ai.usage.cached_tokens"}
    few = lambda pc, i: 1584 if pc and i % 2 else 0  # noqa: E731                 # half the PC-on requests hit
    s, _, _ = smoke(tmp_path / "b", world=World(tmp_path / "wb", cached=few))
    assert s["smoke"]["checks"]["pc_on_hits"] is False
    assert re2.pc_hits_problems("B0", "PC-off", 0, N=4, warmup=1, shared=1584, cached=[0, 0, 7, 0])
    assert re2.pc_hits_problems("PC", "PC-on", 10 ** 9, N=4, warmup=1, shared=1584, cached=[1584] * 4) == []


def test_a_smoke_plan_must_be_build_plans_and_its_datasets_the_pinned_ones(tmp_path):
    root, ds, pins, cfg = setup(tmp_path)
    plan = e2.smoke_plan(pins, cfg)
    trimmed = {**plan, "boots": plan["boots"][-2:]}
    with pytest.raises(re2.RunError, match="exactly"):
        session(root, ds, pins, cfg, trimmed, mode="smoke", expected_plan=plan)
    (root / "other.jsonl").write_text("not the pinned one\n")
    with pytest.raises(re2.RunError, match="shared:2048 sha256"):
        session(root, {**ds, "shared:2048": str(root / "other.jsonl")}, pins, cfg, plan, mode="smoke")
    with pytest.raises(re2.RunError, match="results/stage_e/<session id>"):
        session(root, ds, pins, cfg, plan, mode="smoke", out_dir=root / "results" / "stage_e" / "a" / "b")


def test_three_boots_in_a_row_never_ready_stop_the_session(tmp_path):
    s, _, w = smoke(tmp_path, world=World(tmp_path / "w", never_ready=("*",)))
    assert s["stopped"].startswith("3 boots in a row never ready") and len(w.server_argvs) == 3
    assert s["outcomes"] == {"boots": 12, "valid": 0, "skipped": 9} and s["smoke"]["ok"] is False
    root, ds, pins, cfg = setup(tmp_path / "t")
    plan = e2.smoke_plan(pins, cfg)
    tags = [re2.boot_tag(b) for b in plan["boots"]]
    w = World(tmp_path / "wt", never_ready=tags[:2] + tags[3:5])                     # a ready boot in between
    s, _ = session(root, ds, pins, cfg, plan, mode="smoke", world=w)
    assert not s.get("stopped") and len(w.server_argvs) == 12


def formal_setup(tmp_path):
    root, ds, pins, cfg0 = setup(tmp_path)
    s, _ = session(root, ds, pins, cfg0, e2.smoke_plan(pins, cfg0), mode="smoke", sid="smoke-e2")
    assert s["smoke"]["ok"] is True
    cfg = {**cfg0, "freeze": {"e2": True, "smoke_e2": "smoke-e2"}}
    return root, ds, pins, cfg, e2.build_plan(pins, cfg, mode="formal", root=root)


def rows_of(root, sid):
    return [json.loads(x) for x in (root / "results" / "stage_e" / sid / "rows.jsonl").read_text().splitlines()]


def test_a_formal_session_runs_48_valid_boots_with_their_measurements(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    s, _ = session(root, ds, pins, cfg, plan, mode="formal", sid="f")
    rows = rows_of(root, "f")
    assert len(rows) == 48 and all(r["valid"] for r in rows), [r["invalid_reasons"] for r in rows if not r["valid"]][:3]
    t256 = next(r for r in rows if r["cond"] == "T256")
    assert t256["manip"]["chunk_ok"] is True and t256["beta"] == 1.0 and t256["short_class_p95_ttft_ms"] == 900.0
    assert t256["ctx_tokens_window"] == 5 * 256 and t256["recompute_tokens"] == 0 and t256["budget_full_share"] == 0.0
    b0 = next(r for r in rows if r["cond"] == "B0" and r["mix"] == "uniform" and r["C"] == 25)
    assert b0["budget_full_share"] == 1.0 and b0["lq_short"] == pytest.approx(250 * 0.1 / 10.0, rel=0.01)
    pc = next(r for r in rows if r["cond"] == "PC")
    assert pc["decomp"]["checks"]["a"] is True and pc["mem_increment_mib"] == 15000
    assert all(r.get("derived") for r in rows) and s["status"] == "finished"
    assert s["stage_e2_sha256"] == plan["stage_e2_sha256"] == e2.config_sha256(cfg)
    assert len(list((root / "results" / "stage_e" / "f" / "raw").glob("*.bench.json"))) == 48


def test_an_invalid_boot_skips_the_rest_of_its_cell(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    first = next(b for b in plan["boots"] if b["cell"] == "T256@25" and b["mix"] == "mixed")
    w = World(tmp_path / "wf", drop={re2.boot_tag(first): ("long_prefill_token_threshold",)})
    session(root, ds, pins, cfg, plan, mode="formal", world=w, sid="f")
    rows = rows_of(root, "f")
    cell = [r for r in rows if r["cell"] == "T256@25" and r["mix"] == "mixed"]
    assert [r["valid"] for r in cell] == [False, False, False]
    assert [bool(r.get("skipped")) for r in cell] == [False, True, True] and sum(r["valid"] for r in rows) == 45


def test_the_cap_stops_between_boots(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    plan = {**plan, "cap_s": 60.0}
    s, _ = session(root, ds, pins, cfg, plan, mode="formal", sid="f")
    assert s["cap_reached"] is True and s["outcomes"]["skipped"] > 40


def test_a_spill_at_ready_invalidates_the_boot(tmp_path):
    root, ds, pins, cfg = setup(tmp_path)
    plan = e2.smoke_plan(pins, cfg)
    tag = re2.boot_tag(plan["boots"][0])
    s, _ = session(root, ds, pins, cfg, plan, mode="smoke", world=World(tmp_path / "w", spill_at=(f"ready:{tag}",)))
    assert s["smoke"]["ok"] is False and s["outcomes"]["valid"] == 11


def test_one_valid_formal_session_or_one_void_attempt_and_its_rerun(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    assert session(root, ds, pins, cfg, plan, mode="formal", sid="f1")[0]["status"] == "finished"
    with pytest.raises(re2.RunError, match="already exist"):
        session(root, ds, pins, cfg, plan, mode="formal", sid="f2")
    root, ds, pins, cfg, plan = formal_setup(tmp_path / "void")
    stub = root / "results" / "stage_e" / "v1"
    stub.mkdir(parents=True)
    (stub / "plan.json").write_text(json.dumps(plan))
    (stub / "session.json").write_text('{"stage": "E2", "mo')                       # torn: still an attempt
    refused = root / "results" / "stage_e" / "r1"
    refused.mkdir()
    (refused / "session.json").write_text(json.dumps({"stage": "E2", "mode": "formal", "status": "refused"}))
    assert [(a["dir"].name, a["valid"]) for a in e2.formal_attempts(root)] == [("v1", False)]
    session(root, ds, pins, cfg, plan, mode="formal", sid="v2")
    with pytest.raises(re2.RunError, match="already exist"):
        session(root, ds, pins, cfg, plan, mode="formal", sid="v3")


def test_a_formal_refusal_before_the_first_boot_is_recorded_and_not_an_attempt(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)

    class OldVllm(Env):
        def __call__(self, out_path, gpu_lock_held):
            rec = super().__call__(out_path, gpu_lock_held)
            rec["runtime"]["vllm"]["vllm"] = "0.29.0"
            return rec

    with pytest.raises(re2.RunError, match="refused before it starts"):
        session(root, ds, pins, cfg, plan, mode="formal", sid="x", capture_env=OldVllm())
    assert json.loads((root / "results" / "stage_e" / "x" / "session.json").read_text())["status"] == "refused"
    assert e2.formal_attempts(root) == []


def test_refusals(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    cases = [(dict(gpu_lock_held=False), "GPU lock"), (dict(lock_held_check=lambda: False), "GPU lock"),
             (dict(expected_plan=None), "exactly"),
             (dict(base_env={**BASE_ENV, "PYTORCH_CUDA_ALLOC_CONF": "x"}), "PYTORCH"),
             (dict(toolchain_check=lambda env: ["no ninja"]), "toolchain"),
             (dict(out_dir=tmp_path / "elsewhere"), "results/stage_e")]
    for i, (kw, why) in enumerate(cases):
        with pytest.raises(re2.RunError, match=why):
            session(root, ds, pins, cfg, plan, mode="formal", sid=f"c{i}", **kw)
    (tmp_path / "other.jsonl").write_text("not the pinned one\n")
    with pytest.raises(re2.RunError, match="sha256"):
        session(root, {**ds, "mixed:2048": str(tmp_path / "other.jsonl")}, pins, cfg, plan, mode="formal", sid="d")
