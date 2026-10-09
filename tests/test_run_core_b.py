"""Tests for benchmarks/run_core_b.py: the Core B runner (decision 007; PREREGISTRATION §3 / §6),
with fake server, /metrics and `vllm bench serve`. Real execution happens in WSL after
the S1-S3 smoke gate (008)."""

import json
import signal
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
import run_core_b as rb

ROOT = Path(__file__).resolve().parent.parent

METRICS = """# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="528",cache_dtype="auto",engine="0",num_gpu_blocks="160",prefix_caching_hash_algo="sha256"} 1.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="0",model_name="m"} 7.0
vllm:num_requests_waiting{engine="0",model_name="m"} 2.0
"""


def test_parse_prometheus_and_cache_config():
    m = rb.parse_prometheus(METRICS)
    assert rb.cache_config(m) == {"num_gpu_blocks": 160, "block_size": 528}
    assert rb.gauge(m, "vllm:kv_cache_usage_perc") == 0.42
    assert rb.gauge(m, "vllm:num_preemptions_total") == 3.0


def result_json(L, C, *, n=None, input_len=None, out_len=128, failed=0, mixed=False):
    n = n or bp.num_prompts(C)
    if mixed:
        short, long_ = bp.mixed_counts(n)
        lens = [L - 128] * short + [32640] * long_
    else:
        lens = [input_len or L - 128] * n
    return {"completed": n - failed, "failed": failed, "num_prompts": n, "duration": 10.0,
            "request_throughput": n / 10, "output_throughput": n * 128 / 10, "total_token_throughput": 1e5,
            "input_lens": lens, "output_lens": [out_len] * n, "ttfts": [0.1 + 0.001 * i for i in range(n)],
            "errors": [""] * n,
            **{f"{s}_{m}_ms": 1.0 for s in ("mean", "median", "p50", "p95", "p99")
               for m in ("ttft", "tpot", "itl", "e2el", "client_queue_time")}}


def test_summarize_result_checks_lengths_and_failures():
    ok = rb.summarize_result(result_json(2048, 4), L=2048, C=4, mix="uniform")
    assert ok["ok"] is True and ok["p95_ttft_ms"] == 1.0 and ok["output_throughput"] == 200 * 128 / 10
    assert ok["short_class_p95_ttft_ms"] is not None
    bad = rb.summarize_result(result_json(2048, 4, input_len=2000), L=2048, C=4, mix="uniform")
    assert bad["ok"] is False and any("input_lens" in r for r in bad["reasons"])
    failed = rb.summarize_result(result_json(2048, 4, failed=2), L=2048, C=4, mix="uniform")
    assert failed["ok"] is False and any("failed" in r for r in failed["reasons"])
    mixed = rb.summarize_result(result_json(2048, 22, mixed=True), L=2048, C=22, mix="mixed")
    assert mixed["ok"] is True
    wrong_mix = rb.summarize_result(result_json(2048, 22), L=2048, C=22, mix="mixed")
    assert wrong_mix["ok"] is False


# ----------------------------------------------------------------- session --

class FakeProc:
    pid = 77

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


SERVER_LOG = (
    "(APIServer pid=1) INFO [api_utils.py:286] non-default args: {'model_tag': 'm', 'max_num_batched_tokens': 2048, "
    "'attention_backend': 'FLASH_ATTN', 'async_scheduling': True}\n"
    "(EngineCore pid=2) INFO [cuda.py:478] Using AttentionBackendEnum.FLASH_ATTN backend.\n"
    "(EngineCore pid=2) INFO [kv_cache_utils.py:2395] GPU KV cache size: 66,528 tokens, Maximum concurrency for "
    "32,768 tokens per request: 2.03x\n")


class CoreBEnv:
    """Fake scripts/env_capture.py --vllm-python record."""

    def __init__(self, enforced=300.0, vllm="0.30.0"):
        self.enforced, self.vllm, self.calls = enforced, vllm, []

    def __call__(self, out_path, gpu_lock_held):
        self.calls.append(Path(out_path).name)
        rec = {"runtime": {"vllm": {"vllm": self.vllm}}, "repo": {"commit": "c0ffee", "dirty": False},
               "gpu": {"nvidia": [{"enforced_power_limit_w": self.enforced, "power_default_limit_w": 300.0}]},
               "probe_errors": []}
        Path(out_path).write_text(json.dumps(rec))
        return rec


class World:
    def __init__(self, tmp_path, blocks=160, log=SERVER_LOG):
        self.tmp_path, self.blocks, self.log = tmp_path, blocks, log
        self.started, self.signals, self.bench_calls, self.bench_envs, self.t = [], [], [], [], 0.0
        self.proc = None

    def popen(self, argv, env, log_path):
        Path(log_path).write_text(self.log)
        self.started.append(argv)
        self.server_envs = getattr(self, "server_envs", []) + [env]
        self.proc = FakeProc()
        return self.proc

    def signal_group(self, proc, sig):
        self.signals.append(sig)
        proc.returncode = 0

    def group_alive(self, proc):
        return proc.returncode is None

    def http_get(self, url):
        if url.endswith("/health"):
            return 200, ""
        return 200, METRICS.replace('num_gpu_blocks="160"', f'num_gpu_blocks="{self.blocks}"')

    def run_bench(self, argv, log_path, env):
        self.bench_envs.append(env)
        a = dict(zip(argv, argv[1:]))
        C = int(a["--max-concurrency"])
        L = int(a["--random-input-len"]) + 128 if "--random-input-len" in a else int(Path(a["--dataset-path"]).stem.split("-")[-1])
        self.bench_calls.append((L, C))
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(result_json(L, C, mixed="--dataset-path" in a)))
        return 0

    def gpu_mem_used(self):
        return 400

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def run(world, pins, plan, **kw):
    out = world.tmp_path / "session"
    kw.setdefault("gpu_lock_held", True)
    kw.setdefault("toolchain_check", lambda env: [])
    kw.setdefault("capture_env", CoreBEnv())
    kw.setdefault("gpu_memory_utilization", "0.91")
    summary = rb.run_session(plan, pins=pins, out_dir=out, gdn_prefill_backend="triton",
                             popen=world.popen, signal_group=world.signal_group,
                             group_alive=world.group_alive, http_get=world.http_get, run_bench=world.run_bench,
                             gpu_mem_used=world.gpu_mem_used, gpu_lock_free=lambda: True, now=world.now,
                             sleep=world.sleep, session_id="b1", **kw)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    return out, summary, rows


def small_plan(pins):
    plan = bp.build_uniform_plan(pins, num_gpu_blocks=160, block_size=528)
    plan["runs"] = [r for r in plan["runs"] if r["L"] == 32768]  # grid of C_KV = 2: [1, 2, 3, 4, 8, 16]
    return plan


def test_uniform_session_runs_every_c_once_per_restart(tmp_path, pins):
    w = World(tmp_path)
    out, summary, rows = run(w, pins, small_plan(pins))
    assert len(w.started) == 3  # one server per restart
    assert len(rows) == 3 * len(small_plan(pins)["grids"]["32768"])
    assert all(r["ok"] for r in rows), [r["reasons"] for r in rows if not r["ok"]]
    assert [r["C"] for r in rows if r["restart"] == 1] == small_plan(pins)["runs"][0]["order"]
    assert w.signals.count(signal.SIGINT) == 3
    assert (out / "raw_manifest.json").is_file() and summary["outcomes"]["ok"] == len(rows)
    assert rows[0]["metrics_sampled"]["kv_cache_usage_peak"] == 0.42


def test_cache_config_mismatch_stops_the_restart(tmp_path, pins):
    w = World(tmp_path, blocks=150)
    _, summary, rows = run(w, pins, small_plan(pins))
    n_c = len(small_plan(pins)["grids"]["32768"])
    assert w.bench_calls == [] and summary["outcomes"]["error"] == 3 * n_c  # every planned unit is reported
    assert all("num_gpu_blocks" in r["reasons"][0] for r in rows)


def test_formal_requires_gpu_lock_and_freeze(tmp_path, pins):
    plan = small_plan(pins)
    with pytest.raises(rb.RunError, match="lock"):
        run(World(tmp_path), pins, plan, gpu_lock_held=False, formal=True)
    import copy
    unfrozen = copy.deepcopy(pins)
    unfrozen["freeze"]["nvidia"] = False  # Stage B was frozen on 2026-10-09; the guard itself is what is tested
    with pytest.raises(rb.RunError, match="freeze.nvidia"):
        run(World(tmp_path), unfrozen, plan, formal=True)


def test_existing_session_is_never_overwritten(tmp_path, pins):
    run(World(tmp_path), pins, small_plan(pins))
    with pytest.raises(rb.RunError, match="exists"):
        run(World(tmp_path), pins, small_plan(pins))


def test_bench_runs_offline_with_the_server_hf_cache(tmp_path, pins):
    w = World(tmp_path)
    run(w, pins, small_plan(pins))
    assert w.bench_envs and all(e["HF_HUB_OFFLINE"] == "1" and e["HF_HOME"] == bp.HF_HOME for e in w.bench_envs)


def stage_b_pins(pins, **over):
    import copy

    p = copy.deepcopy(pins)
    p["freeze"]["nvidia"] = True
    p["stage_b"].update({"kv_cache_memory_bytes": 6400 * 2**20, "gpu_memory_utilization": "0.91",
                         "gdn_prefill_backend": "triton", "wsl2_pin_memory": "0", **over})
    return p


def mixed_setup(tmp_path, pins, *, pinned=None):
    import hashlib

    p = stage_b_pins(pins)
    ds = tmp_path / "data" / "mixed-2048.jsonl"
    ds.parent.mkdir(parents=True)
    ds.write_text('{"prompt": "x", "output_tokens": 128}\n')
    digest = hashlib.sha256(ds.read_bytes()).hexdigest()
    p["core_b"]["mix_dataset_sha256"] = {"2048": pinned or digest, "8192": "0" * 64}
    plan = bp.build_mixed_plan(bp.build_uniform_plan(p, num_gpu_blocks=160, block_size=528,
                                                     kv_cache_memory_bytes=6400 * 2**20, gpu_memory_utilization="0.91"),
                               {2048: 22, 8192: None},
                               uniform_session="b-u")
    return p, plan, ds, digest


def test_formal_mixed_run_requires_the_pinned_dataset(tmp_path, pins):
    """The `mixed` file is pinned per L (core_b.mix_dataset_sha256; 007, 013): a formal run checks it first."""
    p, plan, ds, _ = mixed_setup(tmp_path, pins, pinned="f" * 64)
    w = World(tmp_path)
    with pytest.raises(rb.RunError, match="sha256"):
        run(w, p, plan, formal=True, mixed_datasets={2048: str(ds)})
    assert w.started == [] and not (tmp_path / "session").exists()
    with pytest.raises(rb.RunError, match="--mixed-dataset"):
        run(World(tmp_path), p, plan, formal=True)


def test_mixed_run_records_the_dataset_sha256(tmp_path, pins):
    p, plan, ds, digest = mixed_setup(tmp_path, pins)
    out, summary, rows = run(World(tmp_path), p, plan, formal=True, mixed_datasets={2048: str(ds)})
    assert summary["mixed_dataset_sha256"] == {"2048": digest}
    assert len(rows) == 3 and all(r["ok"] for r in rows), [r["reasons"] for r in rows]


def test_server_gets_the_014_env_and_a_toolchain_precheck(tmp_path, pins):
    w = World(tmp_path)
    run(w, pins, small_plan(pins))
    env = w.server_envs[0]
    assert env["CUDA_HOME"] == bp.FLASHINFER_CUDA_HOME and env["PATH"].startswith(bp.VENV_BIN + ":")
    w2 = World(tmp_path / "x")
    (tmp_path / "x").mkdir()
    with pytest.raises(rb.RunError, match="toolchain"):
        run(w2, pins, small_plan(pins), toolchain_check=lambda env: ["missing ninja"])
    assert w2.started == []


def test_formal_run_requires_the_fixed_kv_pool(tmp_path, pins):
    """017: formal Core B runs use --kv-cache-memory-bytes K from the Stage B plan."""
    p = stage_b_pins(pins)
    with pytest.raises(rb.RunError, match="kv_cache_memory_bytes"):
        run(World(tmp_path), p, small_plan(pins), formal=True)
    plan = small_plan(pins)
    plan["kv_cache_memory_bytes"], plan["gpu_memory_utilization"] = 6400 * 2**20, "0.91"
    w = World(tmp_path / "k")
    (tmp_path / "k").mkdir()
    run(w, p, plan, formal=True)
    assert w.started[0][w.started[0].index("--kv-cache-memory-bytes") + 1] == str(6400 * 2**20)



# --------------------------------------------- post-freeze review fixes (018) --

def test_session_records_env_argv_commit_and_log_facts(tmp_path, pins):
    w = World(tmp_path)
    env = CoreBEnv()
    out, summary, rows = run(w, pins, small_plan(pins), capture_env=env)
    assert summary["repo_commit"] == "c0ffee" and summary["preflight"] == "env/preflight.json"
    assert env.calls[0] == "preflight.json" and "uniform-32768-r1.before.json" in env.calls
    assert "uniform-32768-r1.after.json" in env.calls and summary["env_valid"] is True
    run1 = summary["runs"]["uniform-32768-r1"]
    assert run1["argv"] == w.started[0] and run1["log_facts"]["attention_backend"] == "FLASH_ATTN"
    assert run1["max_num_batched_tokens_logged"] == 2048
    assert all(r["kv_cache_tokens"] == 66528 for r in rows)
    assert summary["inherited_unlisted"] == bp.inherited_unlisted(__import__("os").environ)


def test_server_log_checks_stop_a_restart_with_the_wrong_backend_or_batch(tmp_path, pins):
    """§4 / 017: attention backend and max_num_batched_tokens are verified from the log (non-default args)."""
    w = World(tmp_path, log=SERVER_LOG.replace("FLASH_ATTN backend.", "TRITON_ATTN backend."))
    _, _, rows = run(w, pins, small_plan(pins))
    assert rows and all(not r["ok"] and "attention backend" in r["reasons"][0] for r in rows)
    w2 = World(tmp_path / "b", log=SERVER_LOG.replace("'max_num_batched_tokens': 2048", "'max_num_batched_tokens': 8192"))
    (tmp_path / "b").mkdir()
    _, _, rows = run(w2, pins, small_plan(pins))
    assert all(not r["ok"] and "max_num_batched_tokens" in r["reasons"][0] for r in rows)


def test_power_limit_mismatch_makes_the_session_env_invalid(tmp_path, pins):
    _, summary, _ = run(World(tmp_path), pins, small_plan(pins), capture_env=CoreBEnv(enforced=250.0))
    assert summary["env_valid"] is False and "power limit" in summary["env_flags"][0]["reason"]


@pytest.mark.parametrize("over, kw, match", [
    ({"wsl2_pin_memory": ""}, {}, "wsl2_pin_memory"),
    ({}, {"gpu_memory_utilization": "0.85"}, "gpu_memory_utilization"),
    ({"gdn_prefill_backend": "flashinfer"}, {}, "gdn_prefill_backend"),
    ({"kv_cache_memory_bytes": 2**30}, {}, "kv_cache_memory_bytes"),
])
def test_formal_runs_must_match_the_pinned_stage_b_values(tmp_path, pins, over, kw, match):
    p = stage_b_pins(pins, **over)
    plan = small_plan(pins)
    plan["kv_cache_memory_bytes"], plan["gpu_memory_utilization"] = 6400 * 2**20, "0.91"
    with pytest.raises(rb.RunError, match=match):
        run(World(tmp_path), p, plan, formal=True, **kw)


def test_formal_run_refuses_another_vllm_or_power_limit_in_the_preflight(tmp_path, pins):
    p = stage_b_pins(pins)
    plan = small_plan(pins)
    plan["kv_cache_memory_bytes"], plan["gpu_memory_utilization"] = 6400 * 2**20, "0.91"
    with pytest.raises(rb.RunError, match="vLLM"):
        run(World(tmp_path), p, plan, formal=True, capture_env=CoreBEnv(vllm="0.31.0"))
    with pytest.raises(rb.RunError, match="power limit"):
        run(World(tmp_path / "x"), p, plan, formal=True, capture_env=CoreBEnv(enforced=250.0))


# ------------------------------------------------- provenance (020 review) --


class RepoEnv(CoreBEnv):
    """repo record: dirty flag, and a different commit after `after` captures (a mid-session redeploy)."""

    def __init__(self, dirty=False, after=None):
        super().__init__()
        self.dirty, self.after = dirty, after

    def __call__(self, out_path, gpu_lock_held):
        rec = super().__call__(out_path, gpu_lock_held)
        late = self.after is not None and len(self.calls) > self.after
        rec["repo"] = {"commit": "beef" if late else "c0ffee", "dirty": self.dirty}
        Path(out_path).write_text(json.dumps(rec))
        return rec


def formal_plan(pins):
    plan = small_plan(pins)
    plan["kv_cache_memory_bytes"], plan["gpu_memory_utilization"] = 6400 * 2**20, "0.91"
    return plan


def test_formal_core_b_refuses_a_dirty_preflight(tmp_path, pins):
    with pytest.raises(rb.RunError, match="dirty"):
        run(World(tmp_path), stage_b_pins(pins), formal_plan(pins), formal=True, capture_env=RepoEnv(dirty=True))


def test_formal_core_b_flags_a_mid_session_code_change(tmp_path, pins):
    _, summary, _ = run(World(tmp_path), stage_b_pins(pins), formal_plan(pins), formal=True,
                        capture_env=RepoEnv(after=2))
    assert summary["env_valid"] is False and any("mid-session" in f["reason"] for f in summary["env_flags"])
    (tmp_path / "pilot").mkdir()
    _, pilot, _ = run(World(tmp_path / "pilot"), pins, small_plan(pins), capture_env=RepoEnv(after=2))
    assert pilot["env_valid"] is True  # provenance is a formal-session rule
