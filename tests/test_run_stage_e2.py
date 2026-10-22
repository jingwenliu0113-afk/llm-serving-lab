"""Tests for benchmarks/run_stage_e2.py (decision 041 v3 §2.3-2.10): whole E2 sessions against a fake WSL (server
log, /metrics, OTLP receiver, iteration log, `vllm bench serve`, 031 idle and spill readings; no GPU)."""

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
import run_capacity as rc
import run_stage_e2 as re2
import stage_e2_plan as e2
import test_run_stage_d as td

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_e2.toml").read_text())
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u"}


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
    """td.World's clock, idle and spill readings with an E2 server: its non-default args, blocks and prefix caching
    follow the argv; drop[label] removes a key from its log; hits_off adds prefix hits when caching is off."""

    def __init__(self, tmp_path, *, drop=None, no_hits=False, hits_off=False, spill_at=()):
        super().__init__(tmp_path, spill_at=spill_at)
        self.drop, self.no_hits, self.hits_off = drop or {}, no_hits, hits_off
        self.pc, self.blocks, self.hits, self.receiver, self.server_log = False, 248, 0.0, None, None

    def wall(self):
        return 1.79e9 + self.t

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
            return (200, "") if alive else (0, "refused")
        lab = 'engine="0",model_name="m"'
        return 200, (f'vllm:cache_config_info{{block_size="528",enable_prefix_caching="{self.pc}",engine="0",'
                     f'num_gpu_blocks="{self.blocks}"}} 1.0\n'
                     f"vllm:num_requests_running{{{lab}}} 3.0\nvllm:num_requests_waiting{{{lab}}} 2.0\n"
                     f"vllm:kv_cache_usage_perc{{{lab}}} 0.4\nvllm:num_preemptions_total{{{lab}}} 0.0\n"
                     f"vllm:prefix_cache_hits_total{{{lab}}} {self.hits}\n")

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
        self.receiver.records = [{"id": f"cmpl-{prefix}{i}-0", "arrival": t0 + 0.5 + 0.01 * i, "Q": 0.1, "P": 0.5,
                                  "server_ttft": ttfts[i], "e2e": lats[i], "prompt_tokens": lens[i],
                                  "completion_tokens": 128} for i in range(n)]
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


def frozen():
    e = {**CFG["e2"], "expected_blocks": {**CFG["e2"]["expected_blocks"], "PC": 248},
         "shared_prefix": {**CFG["e2"]["shared_prefix"], "sha256": "", "common_prefix": 1630}}
    return {**CFG, "freeze": {"e2": True, "smoke_e2": "smoke-e2"}, "e2": e}


def datasets(root, cfg):
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "mixed-2048.jsonl").write_text("mixed\n")
    (d / "shared-2048.jsonl").write_text("shared\n")
    return {"mixed:2048": str(d / "mixed-2048.jsonl"), "shared:2048": str(d / "shared-2048.jsonl")}


def pins_with(ds):
    p = json.loads(json.dumps(PINS))
    p["core_b"]["mix_dataset_sha256"]["2048"] = hashlib.sha256(Path(ds["mixed:2048"]).read_bytes()).hexdigest()
    return p


def session(tmp_path, plan, cfg, *, mode, world=None, pins=None, ds=None, **kw):
    root = tmp_path / "repo"
    ds = ds or datasets(root, cfg)
    pins = pins or pins_with(ds)
    w = world or World(tmp_path)
    args = dict(pins=pins, cfg=cfg, pins_sha256="p" * 64, stage_e2_sha256="s" * 64,
                out_dir=root / "results" / "stage_e" / kw.pop("sid", "s"), session_id="s", mode=mode, popen=w.popen,
                signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get, run_bench=w.run_bench,
                gpu_sample=w.gpu_sample, compute_apps=w.compute_apps, shared_usage=w.shared_usage,
                lock_held_check=lambda: True, gpu_lock_held=True, capture_env=Env(), datasets=ds,
                receiver_factory=lambda raw, spans, prefix: Receiver(w, prefix), vllm_bin="/v", base_env=BASE_ENV,
                toolchain_check=lambda env: [], sample_interval=None, now=w.now, sleep=w.sleep, wall=w.wall,
                clock=lambda: "2026-10-24T00:00:00", repo_root=root)
    args.update(kw)
    return re2.run_session(plan, **args), root, w


def smoke_plan(pins):
    with rc.mini_num_prompts(e2.SMOKE_N):
        return e2.build_plan(pins, CFG, mode="smoke")


def test_a_smoke_session_validates_every_condition_and_masks_the_latencies(tmp_path):
    root = tmp_path / "repo"
    ds = datasets(root, CFG)
    pins = pins_with(ds)
    s, root, w = session(tmp_path, smoke_plan(pins), CFG, mode="smoke", pins=pins, ds=ds)
    assert s["smoke"]["ok"] is True, s["smoke"]
    assert s["smoke"]["measured_blocks"]["K25"] == 155 and s["smoke"]["measured_blocks"]["PC"] == 248
    out = root / "results" / "stage_e" / "s"
    assert not list((out / "raw").glob("*.bench.json"))                     # 030 8.6: latencies discarded
    assert s["outcomes"] == {"boots": 8, "valid": 8, "skipped": 0} and s["host"]["nvidia"] is True
    pc_argv = next(a for a in w.server_argvs if "--enable-prefix-caching" in a)
    assert "--no-enable-prefix-caching" not in pc_argv


def test_a_missing_prefix_cache_hit_or_a_stray_one_invalidates_the_pc_boot(tmp_path):
    root = tmp_path / "repo"
    ds = datasets(root, CFG)
    pins = pins_with(ds)
    s, _, _ = session(tmp_path, smoke_plan(pins), CFG, mode="smoke", pins=pins, ds=ds,
                      world=World(tmp_path, no_hits=True))
    assert s["smoke"]["ok"] is False and s["smoke"]["checks"]["pc_on_hits"] is False
    s, _, _ = session(tmp_path / "b", smoke_plan(pins), CFG, mode="smoke", pins=pins, ds=ds,
                      world=World(tmp_path / "b", hits_off=True))
    assert s["smoke"]["checks"]["pc_off_no_hits"] is False


def formal_setup(tmp_path):
    root = tmp_path / "repo"
    ds = datasets(root, frozen())
    pins = pins_with(ds)
    cfg = frozen()
    cfg["e2"]["shared_prefix"]["sha256"] = hashlib.sha256(Path(ds["shared:2048"]).read_bytes()).hexdigest()
    d = root / "results" / "stage_e" / "smoke-e2"
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps({"stage": "E2", "mode": "smoke", "smoke": {"ok": True},
                                                "harness_sha256": e2.harness_sha256(),
                                                "host": {"system": "Linux", "wsl": True, "nvidia": True}}))
    plan = e2.build_plan(pins, cfg, mode="formal", root=root)
    return root, ds, pins, cfg, plan


def test_a_formal_session_runs_48_valid_boots_with_their_measurements(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    s, _, w = session(tmp_path, plan, cfg, mode="formal", pins=pins, ds=ds, expected_plan=plan, sid="f")
    rows = [json.loads(x) for x in (root / "results" / "stage_e" / "f" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 48 and all(r["valid"] for r in rows), [r["invalid_reasons"] for r in rows if not r["valid"]][:3]
    t256 = next(r for r in rows if r["cond"] == "T256")
    assert t256["manip"]["chunk_ok"] is True and t256["beta"] == 1.0 and t256["short_class_p95_ttft_ms"] == 900.0
    assert all(r.get("derived") for r in rows) and next(r for r in rows if r["cond"] == "PC")["decomp"].startswith("n/a")
    assert len(list((root / "results" / "stage_e" / "f" / "raw").glob("*.bench.json"))) == 48


def test_an_invalid_boot_skips_the_rest_of_its_cell(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    first = next(b for b in plan["boots"] if b["cell"] == "T256@25" and b["mix"] == "mixed")
    w = World(tmp_path, drop={re2.boot_tag(first): ("long_prefill_token_threshold",)})
    s, _, _ = session(tmp_path, plan, cfg, mode="formal", pins=pins, ds=ds, expected_plan=plan, world=w, sid="f")
    rows = [json.loads(x) for x in (root / "results" / "stage_e" / "f" / "rows.jsonl").read_text().splitlines()]
    cell = [r for r in rows if r["cell"] == "T256@25" and r["mix"] == "mixed"]
    assert [r["valid"] for r in cell] == [False, False, False] and [bool(r.get("skipped")) for r in cell] == [False,
                                                                                                             True, True]
    assert sum(r["valid"] for r in rows) == 45


def test_the_cap_stops_between_boots(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    plan = {**plan, "cap_s": 60.0}
    s, _, _ = session(tmp_path, plan, cfg, mode="formal", pins=pins, ds=ds, expected_plan=plan, sid="f")
    assert s["cap_reached"] is True and s["outcomes"]["skipped"] > 40


def test_a_spill_at_ready_invalidates_the_boot(tmp_path):
    root = tmp_path / "repo"
    ds = datasets(root, CFG)
    pins = pins_with(ds)
    plan = smoke_plan(pins)
    tag = re2.boot_tag(plan["boots"][0])
    s, _, _ = session(tmp_path, plan, CFG, mode="smoke", pins=pins, ds=ds, world=World(tmp_path, spill_at=(f"ready:{tag}",)))
    assert s["smoke"]["ok"] is False and s["outcomes"]["valid"] == 7


def test_refusals(tmp_path):
    root, ds, pins, cfg, plan = formal_setup(tmp_path)
    cases = [(dict(gpu_lock_held=False), "GPU lock"), (dict(lock_held_check=lambda: False), "GPU lock"),
             (dict(expected_plan=None), "exactly"), (dict(expected_plan=plan, base_env={**BASE_ENV,
                                                                                       "PYTORCH_CUDA_ALLOC_CONF": "x"}),
                                                     "PYTORCH"),
             (dict(expected_plan=plan, toolchain_check=lambda env: ["no ninja"]), "toolchain"),
             (dict(expected_plan=plan, out_dir=tmp_path / "elsewhere"), "results/stage_e")]
    for i, (kw, why) in enumerate(cases):
        with pytest.raises(re2.RunError, match=why):
            session(tmp_path / f"c{i}", plan, cfg, mode="formal", pins=pins, ds=ds, sid="f", **kw)
    bad = {**ds, "mixed:2048": str(tmp_path / "other.jsonl")}
    (tmp_path / "other.jsonl").write_text("not the pinned one\n")
    with pytest.raises(re2.RunError, match="sha256"):
        session(tmp_path / "d", plan, cfg, mode="formal", pins=pins, ds=bad, expected_plan=plan, sid="f")
