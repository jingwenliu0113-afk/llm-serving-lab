"""Tests for benchmarks/run_c0_probe.py: the C0 probe runner (decision 028; exploratory, formal = false), with a fake
server, /metrics, nvidia-smi, clock and `vllm bench serve` (no GPU). Real execution happens in WSL under the GPU
lock."""

import ast
import copy
import io
import json
import os
import signal
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

import c0_analysis as ca
import core_b_plan as bp
import run_c0_probe as rp

# ------------------------------------------------------------------ startup-line extractor --

STARTUP_LOG = "".join(f"{stamp} {line}\n" for stamp, line in (
    (1791657700000, "(APIServer pid=11) INFO 10-15 12:00:00 [api_utils.py:286] non-default args: {'model_tag': "
                    "'m', 'max_num_batched_tokens': 2048, 'kv_cache_memory_bytes': 4294967296}"),
    (1791657701000, '(EngineCore pid=12) INFO 10-15 12:00:01 [cuda.py:478] Using '
                    'AttentionBackendEnum.FLASH_ATTN backend.'),
    (1791657702000, '(EngineCore pid=12) INFO 10-15 12:00:02 [qwen_gdn_linear_attn.py:90] Using Triton/FLA GDN '
                    'prefill kernel'),
    (1791657703000, '(EngineCore pid=12) INFO 10-15 12:00:03 [backends.py:610] Using cache directory: '
                    "/home/u/.cache/vllm/torch_compile_cache/ab12/rank_0_0/backbone for vLLM's torch.compile"),
    (1791657704000, '(EngineCore pid=12) INFO 10-15 12:00:04 [backends.py:211] Directly load the compiled '
                    'graph(s) for compile range (1, 2048) from the cache, took 3.21 s'),
    (1791657705000, '(EngineCore pid=12) INFO 10-15 12:00:05 [monitor.py:34] torch.compile takes 6.42 s in '
                    'total'),
    (1791657706000, '(EngineCore pid=12) INFO 10-15 12:00:06 [gpu_worker.py:300] Available KV cache memory: '
                    '4.14 GiB'),
    (1791657707000, '(EngineCore pid=12) INFO 10-15 12:00:07 [kv_cache_utils.py:2395] GPU KV cache size: '
                    '123,128 tokens'),
    (1791657708000, '(EngineCore pid=12) INFO 10-15 12:00:08 [gpu_model_runner.py:4100] Graph capturing '
                    'finished in 5 secs, took 0.52 GiB'),
    (1791657708500, '(EngineCore pid=12) INFO 10-15 12:00:08 [gpu_model_runner.py:4000] Capturing CUDA graphs '
                    '(mixed prefill-decode, PIECEWISE): sizes [1, 2, 4, 8]'),
    (1791657709000, '(EngineCore pid=12) INFO 10-15 12:00:09 [core.py:250] init engine (profile, create kv '
                    'cache, warmup model) took 27.31 seconds'),
    (1791657709500, '(EngineCore pid=12) WARNING 10-15 12:00:09 [utils.py:10] some warning'),
    (1791657709600, '(EngineCore pid=12) ERROR 10-15 12:00:09 [utils.py:11] some error'),
    (1791657710000, '(APIServer pid=11) INFO:     Application startup complete.'),
    (1791657711000, '(APIServer pid=11) WARNING 10-15 12:00:11 [x.py:1] after startup'),
    (1791657712000, '(APIServer pid=11) INFO 10-15 12:00:12 [loggers.py:185] Engine 000: Avg prompt '
                    'throughput: 0.0 tokens/s'),
))


def test_startup_lines_are_cut_at_application_startup_complete_and_categorised():
    got = rp.startup_lines(STARTUP_LOG)
    assert got["nondefault_args"] == [STARTUP_LOG.splitlines()[0].split(" ", 1)[1]]
    assert len(got["backend"]) == 2 and "FLASH_ATTN" in got["backend"][0] and "GDN prefill" in got["backend"][1]
    assert len(got["compile_cache"]) == 3
    assert len(got["cuda_graph"]) == 2
    assert len(got["kv_memory"]) == 2
    assert any("init engine" in x for x in got["init_time"]) and any("startup complete" in x for x in got["init_time"])
    assert got["warning"] == ["(EngineCore pid=12) WARNING 10-15 12:00:09 [utils.py:10] some warning"]
    assert got["error"] == ["(EngineCore pid=12) ERROR 10-15 12:00:09 [utils.py:11] some error"]
    assert not any("after startup" in x for v in got.values() for x in v)


def test_startup_lines_read_unstamped_logs_and_logs_without_a_ready_line():
    plain = "\n".join(x.split(" ", 1)[1] for x in STARTUP_LOG.splitlines()[:5])
    got = rp.startup_lines(plain)
    assert len(got["nondefault_args"]) == 1 and len(got["compile_cache"]) == 2


def test_startup_lines_cap_each_category():
    log = "\n".join(f"WARNING x {i}" for i in range(rp.STARTUP_MAX_LINES + 5))
    assert len(rp.startup_lines(log)["warning"]) == rp.STARTUP_MAX_LINES


def test_normalize_line_drops_stamp_pid_clock_and_decimals_but_keeps_integers():
    a = "1791657708000 (EngineCore pid=12) INFO 10-15 12:00:08 [gpu_model_runner.py:4100] Graph capturing " \
        "finished in 5 secs, took 0.52 GiB, sizes [1, 2, 4, 8]"
    b = "(EngineCore pid=99) INFO 10-16 01:02:03 [gpu_model_runner.py:4100] Graph capturing " \
        "finished in 5 secs, took 0.61 GiB, sizes [1, 2, 4, 8]"
    assert rp.normalize_line(a) == rp.normalize_line(b)
    assert rp.normalize_line(a) == ("(EngineCore) INFO [gpu_model_runner.py:4100] Graph capturing finished in 5 secs, "
                                    "took # GiB, sizes [1, 2, 4, 8]")
    assert rp.normalize_line(a.replace("[1, 2, 4, 8]", "[1, 2, 4]")) != rp.normalize_line(a)


def test_startup_facts_hold_the_structured_values():
    got = rp.startup_facts(STARTUP_LOG)
    assert got["nondefault"] == {"max_num_batched_tokens": 2048, "kv_cache_memory_bytes": 4294967296}
    assert got["log_facts"]["attention_backend"] == "FLASH_ATTN" and got["log_facts"]["kv_cache_tokens"] == 123128
    assert got["available_kv_gib"] == 4.14
    assert got["counts"]["warning"] == 1 and got["counts"]["error"] == 1
    assert set(got["lines"]) == set(rp.STARTUP_PATTERNS)


# ================================================================== the runner (fake world) ==

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_791_700_000.0
SMI_EXITED = "the nvidia-smi sampler exited (code 2)"
PINS_SHA = "cd" * 32
OBSERVATION_FLAGS = ("--otlp-traces-endpoint", "--enable-logging-iteration-details", "--collect-detailed-traces",
                     "--long-prefill-token-threshold", "--no-scheduler-reserve-full-isl", "--request-id-prefix")
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u", "VLLM_LOGGING_LEVEL": "DEBUG"}


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture
def plan(pins):
    return rp.build_plan(pins, pins_sha256=PINS_SHA)


def cycle_s(N, slips=(), C=8):
    """TTFTs (s) with the closed-loop period of 028 (first wave idx < C, slipped groups + 340 ms)."""
    base = [1.485] + [2.477] * (C - 2) + [2.320]
    t = [base[i % C] for i in range(N)]
    for i in range(1, min(C, N)):
        t[i] = 2.8 + 1.2 * i
    for k in slips:
        for p in range(1, C):
            if C * k + p < N:
                t[C * k + p] += 0.340
    return t


class FakeProc:
    pid = 4242

    def __init__(self):
        self.returncode = None
        self.log_reader = FakeReader()

    def poll(self):
        return self.returncode


class FakeReader:
    def __init__(self):
        self.error = None

    def join(self, timeout):
        return True


class FakeSmi:
    def __init__(self, world, path):
        self.world, self.path, self.stopped = world, Path(path), False
        self.hot, self.lost, self.halt, self.lost_reason = threading.Event(), threading.Event(), threading.Event(), None
        world.events.append("smi:start")
        world.smi = self

    def flag_hot(self):
        self.hot.set()
        self.halt.set()

    def lose(self, reason):
        self.lost_reason = reason
        self.lost.set()
        self.halt.set()

    def stop(self):
        self.stopped = True
        self.world.events.append("smi:stop")
        return []

    def samples(self):
        return list(self.world.smi_samples)


class FakeEnv:
    def __init__(self, world, driver="617.42", vllm="0.30.0", torch="2.13.0+cu130"):
        self.world, self.driver, self.vllm, self.torch, self.calls = world, driver, vllm, torch, []

    def __call__(self, out_path, gpu_lock_held):
        self.calls.append(Path(out_path).name)
        self.world.events.append(f"env:{Path(out_path).name}")
        rec = {"runtime": {"vllm": {"vllm": self.vllm, "torch": self.torch}}, "repo": {"commit": "c0ffee"},
               "gpu": {"nvidia": [{"driver_version": self.driver, "enforced_power_limit_w": 300.0,
                                   "power_default_limit_w": 300.0}]}, "probe_errors": []}
        Path(out_path).write_text(json.dumps(rec))
        return rec


class World:
    """Fake WSL for one probe session. behave[(boot, bench)] in {"ok", "timeout", "exit", "fail", "hot",
    "hot_after", "lost", "unhealthy", "raise"}; slips[(boot, bench)] = slipped groups of that bench; blocks /
    nondefault edit the server; mem: memory.used values (the last one repeats); apps: compute processes; lock: lock
    checks in order (the last one repeats); hot_at_start / lost_at_start: boots whose GPU reaches 88 C / whose
    nvidia-smi sampler dies during the startup; halt_at_teardown[boot] in {"hot", "lost"}: the same during its
    teardown; smi: "ok", "none" (the sampler never reports) or "no_temp" (it never reports a temperature)."""

    def __init__(self, tmp_path, *, behave=None, slips=None, blocks=248, nondefault=None, mem=(400,), apps=(),
                 lock=(True,), hot_at_start=(), lost_at_start=(), halt_at_teardown=None, smi="ok", startup_s=32.0,
                 bench_s=330.0, never_ready=False, health_s=0.0):
        self.tmp_path, self.behave, self.slips = tmp_path, behave or {}, slips or {}
        self.blocks, self.nondefault, self.mem, self.apps = blocks, nondefault, list(mem), list(apps)
        self.lock, self.hot_at_start, self.never_ready = list(lock), set(hot_at_start), never_ready
        self.lost_at_start, self.halt_at_teardown, self.smi_mode = set(lost_at_start), halt_at_teardown or {}, smi
        self.startup_s, self.bench_s, self.health_s = startup_s, bench_s, health_s
        self.t, self.events, self.started, self.bench_argvs, self.bench_timeouts = 0.0, [], [], [], []
        self.boot, self.bench, self.boot_t, self.proc, self.smi, self.unhealthy = 0, 0, 0.0, None, None, False
        self.q = {"0.3": 0.0, "0.5": 0.0, "+Inf": 0.0}
        self.smi_samples, self.mem_calls, self.lock_calls = [], 0, 0

    # clocks
    def now(self):
        return self.t

    def sleep(self, s):
        assert s >= 0, f"sleep({s}): time.sleep refuses a negative length"
        self.t += s

    def wall(self):
        return EPOCH + self.t

    # checks
    def lock_held(self):
        self.lock_calls += 1
        return self.lock[min(self.lock_calls, len(self.lock)) - 1]

    def gpu_mem_used(self):
        self.mem_calls += 1
        return self.mem[min(self.mem_calls, len(self.mem)) - 1]

    def compute_apps(self):
        self.events.append("apps")
        return list(self.apps)

    def start_smi(self, path):
        return FakeSmi(self, path)

    # server
    def popen(self, argv, env, log_path):
        self.boot += 1
        self.bench, self.unhealthy, self.boot_t = 0, False, self.t
        self.events.append(f"popen:{self.boot}")
        self.started.append((argv, env))
        a = dict(zip(argv, argv[1:]))
        nd = {"model_tag": argv[2], "max_num_batched_tokens": int(a["--max-num-batched-tokens"]),
              "kv_cache_memory_bytes": int(a["--kv-cache-memory-bytes"]), "max_model_len": 32768,
              "attention_backend": "FLASH_ATTN"}
        if self.nondefault:
            nd = self.nondefault(nd)
        lines = [f"(APIServer pid=1) INFO 10-16 00:00:00 [api_utils.py:286] non-default args: {nd!r}",
                 "(EngineCore pid=2) INFO 10-16 00:00:01 [cuda.py:478] Using AttentionBackendEnum.FLASH_ATTN backend.",
                 "(EngineCore pid=2) INFO 10-16 00:00:02 [kv_cache_utils.py:2395] GPU KV cache size: 123,128 tokens",
                 "(APIServer pid=1) INFO:     Application startup complete."]
        Path(log_path).write_text("".join(f"{int(self.wall() * 1000)} {x}\n" for x in lines))
        if self.boot in self.hot_at_start:
            self.smi.flag_hot()
        if self.boot in self.lost_at_start:
            self.smi.lose(SMI_EXITED)
        self.proc = FakeProc()
        return self.proc

    def signal_group(self, proc, sig):
        self.events.append(f"signal:{int(sig)}")
        kind = self.halt_at_teardown.get(self.boot)
        if kind == "hot":
            self.smi.flag_hot()
        elif kind == "lost":
            self.smi.lose(SMI_EXITED)
        proc.returncode = 0

    def smi_sample(self, temp_c):
        if self.smi_mode != "none":
            self.smi_samples.append({"t": self.wall(), "pstate": "P1", "sm_mhz": 2800.0,
                                     "temp_c": None if self.smi_mode == "no_temp" else temp_c,
                                     "event_reasons": 0x400})

    def group_alive(self, proc):
        return proc.returncode is None

    def metrics_text(self):
        lab = 'engine="0",model_name="m"'
        q = self.q
        return "\n".join([
            'vllm:cache_config_info{block_size="528",cache_dtype="auto",engine="0",'
            f'num_gpu_blocks="{self.blocks}"}} 1.0',
            f"vllm:kv_cache_usage_perc{{{lab}}} 0.5", f"vllm:num_requests_running{{{lab}}} 8.0",
            f"vllm:num_requests_waiting{{{lab}}} 0.0", f"vllm:num_preemptions_total{{{lab}}} 0.0",
            *[f'vllm:request_queue_time_seconds_bucket{{{lab},le="{le}"}} {v}' for le, v in q.items()],
            f"vllm:request_queue_time_seconds_sum{{{lab}}} 1.0",
            f"vllm:request_queue_time_seconds_count{{{lab}}} {q['+Inf']}", ""])

    def http_get(self, url):
        if url.endswith("/health"):
            self.t += self.health_s
            ready = not self.never_ready and self.t - self.boot_t >= self.startup_s and not self.unhealthy
            return (200, "") if ready and self.proc.returncode is None else (0, "refused")
        return 200, self.metrics_text()

    # client
    def run_bench(self, argv, log_path, env, timeout_s, stop):
        self.bench += 1
        key = (self.boot, self.bench)
        self.events.append(f"bench:{self.boot}-{self.bench}")
        self.bench_argvs.append(argv)
        self.bench_timeouts.append(timeout_s)
        how = self.behave.get(key, "ok")
        Path(log_path).write_text("bench log\n")
        self.smi_sample(60.0)
        if how == "raise":
            raise RuntimeError("boom: the client crashed the harness")
        if how == "timeout":
            self.t += timeout_s
            return -9, True, False
        if how in ("hot", "lost"):
            self.t += 100.0
            if how == "hot":
                self.smi.flag_hot()
            else:
                self.smi.lose(SMI_EXITED)
            assert stop.is_set()  # run_bench_watched's stop flag is the sampler's
            return -9, False, True
        if how == "exit":
            self.t += 5.0
            return 1, False, False
        a = dict(zip(argv, argv[1:]))
        N = int(a["--num-prompts"])
        slips = self.slips.get(key, ())
        ttfts = cycle_s(N, slips)
        failed = 5 if how == "fail" else 0
        n_slipped = sum(1 for k in slips for p in range(1, 8) if 8 * k + p < N)
        self.q["0.3"] += N + 8 - n_slipped - 14
        self.q["0.5"] += N + 8 - 14
        self.q["+Inf"] += N + 8
        res = {"completed": N - failed, "failed": failed, "num_prompts": N, "duration": self.bench_s,
               "request_throughput": N / self.bench_s, "output_throughput": N * 128 / self.bench_s,
               "total_token_throughput": 1e4, "input_lens": [8064] * N, "output_lens": [128] * N, "ttfts": ttfts,
               "latencies": [x + 127 * 0.08 for x in ttfts], "errors": [""] * N,
               **{f"{s}_{m}_ms": 1.0 for s in ("mean", "median", "p50", "p95", "p99")
                  for m in ("ttft", "tpot", "itl", "e2el", "client_queue_time")}}
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(res))
        self.t += self.bench_s
        self.smi_sample(61.0)
        if how == "unhealthy":
            self.unhealthy = True
        if how == "hot_after":
            self.smi.flag_hot()
        return 0, False, False


def run(world, plan, pins, out=None, smoke=False, t_start=0.0, **kw):
    out = out or world.tmp_path / ("smoke-c0-20261016" if smoke else "explore-c0-20261016")
    kw.setdefault("gpu_lock_held", True)
    kw.setdefault("toolchain_check", lambda env: [])
    kw.setdefault("capture_env", FakeEnv(world))
    kw.setdefault("base_env", BASE_ENV)
    kw.setdefault("cap_hours", 3.0)
    kw.setdefault("sample_interval", None)
    session = rp.run_session(plan, pins=pins, pins_sha256=PINS_SHA, out_dir=out, session_id=out.name,
                             is_smoke=smoke,
                             popen=world.popen, signal_group=world.signal_group, group_alive=world.group_alive,
                             http_get=world.http_get, run_bench=world.run_bench, gpu_mem_used=world.gpu_mem_used,
                             compute_apps=world.compute_apps, start_smi=world.start_smi,
                             lock_held_check=world.lock_held, t_start=t_start, now=world.now, sleep=world.sleep,
                             wall=world.wall, **kw)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    return out, session, rows


# ------------------------------------------------------------------ plan --

def test_plan_is_deterministic_and_holds_the_028_design(pins):
    a, b = rp.build_plan(pins, pins_sha256=PINS_SHA), rp.build_plan(pins, pins_sha256=PINS_SHA)
    assert a == b and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert {k: a[k] for k in ("stage", "decision", "L", "mix", "C", "N", "warmup", "seed", "boots",
                              "benches_per_boot")} == {
        "stage": "C0", "decision": "028", "L": 8192, "mix": "uniform", "C": 8, "N": 200, "warmup": 8,
        "seed": 20261008, "boots": 6, "benches_per_boot": 4}
    assert a["smoke"] == {"boots": 1, "benches": 1, "N": 16, "C": 8}
    assert (a["kv_cache_memory_bytes"], a["expected_blocks"], a["block_size"]) == (4294967296, 248, 528)
    assert (a["gpu_memory_utilization"], a["gdn_prefill_backend"]) == ("0.91", "triton")
    assert (a["timeout_s"], a["cap_hours"], a["boot_margin_s"], a["temp_stop_c"], a["max_failed_boots"]) == (
        720, 3.0, 1800, 88, 2)
    assert a["C"] == pins["advanced"]["c"]["8192"] and a["N"] == bp.num_prompts(8) and a["warmup"] == bp.num_warmups(8)
    assert [x["tag"] for x in a["order"]] == [f"b{b}-n{j}" for b in range(1, 7) for j in range(1, 5)]
    assert a["pins_sha256"] == PINS_SHA


def test_plan_problems_refuse_any_drift(pins, plan):
    assert rp.plan_problems(plan, pins, pins_sha256=PINS_SHA) == []
    for key, value in (("N", 100), ("boots", 7), ("timeout_s", 900), ("pins_sha256", "ef" * 32)):
        bad = copy.deepcopy(plan)
        bad[key] = value
        assert any(key in p for p in rp.plan_problems(bad, pins, pins_sha256=PINS_SHA)), key


# ------------------------------------------------------------------ a whole session --

def test_full_session_runs_6_boots_of_4_benches_with_core_argv(tmp_path, pins, plan):
    world = World(tmp_path, slips={(1, 1): (8,), (2, 3): (16, 24), (4, 2): (11,)})
    out, session, rows = run(world, plan, pins)
    assert len(rows) == 24 and all(r["ok"] for r in rows) and session["stopped"] is None
    assert [r["tag"] for r in rows] == [x["tag"] for x in plan["order"]]
    assert {k: session[k] for k in ("formal", "stage", "purpose", "decision")} == {
        "formal": False, "stage": "C0", "purpose": "exploratory", "decision": "028"}
    # server: Core's argv and env, K = the Stage B K, no observation flag
    want_argv, want_env = bp.server_argv(pins, gdn_prefill_backend="triton", gpu_memory_utilization="0.91",
                                         port=rp.PORT, vllm_bin=".venv/bin/vllm", base_env=BASE_ENV,
                                         kv_cache_memory_bytes=4294967296)
    assert all(argv == want_argv and env == want_env for argv, env in world.started) and len(world.started) == 6
    assert session["server"]["argv"] == want_argv
    # client: Core's argv, nothing added
    for r, argv in zip(rows, world.bench_argvs):
        assert argv == bp.bench_argv(pins, L=8192, C=8, mix="uniform", result_dir=str(out / "raw"),
                                     result_filename=f"{r['tag']}.bench.json", port=rp.PORT, vllm_bin=".venv/bin/vllm")
        assert r["argv"] == argv
    for argv in [a for a, _ in world.started] + world.bench_argvs:
        assert not any(f in argv for f in OBSERVATION_FLAGS)
    assert world.bench_timeouts == [720] * 24
    # outputs
    assert sorted(p.name for p in out.iterdir()) == ["derived", "env", "plan.json", "raw", "raw_manifest.json",
                                                     "rows.jsonl", "session.json"]
    assert json.loads((out / "plan.json").read_text()) == plan
    assert sorted(p.name for p in (out / "derived").iterdir()) == sorted(
        [f"{r['tag']}-requests.jsonl" for r in rows] + [f"b{b}-smi.jsonl" for b in range(1, 7)])
    assert sorted(p.name for p in (out / "env").iterdir()) == sorted(
        ["preflight.json"] + [f"b{b}.{w}.json" for b in range(1, 7) for w in ("before", "after")])
    manifest = json.loads((out / "raw_manifest.json").read_text())["files"]
    assert "raw/b1-n1.bench.json" in manifest and "raw/b1.server.log" in manifest
    first = rows[0]
    assert first["position"] == 1 and first["t_end"] > first["t_start"] and first["num_gpu_blocks"] == 248
    assert first["scrape_delta"]["vllm:request_queue_time_seconds"]["buckets"]["0.5"] == 194.0
    reqs = [json.loads(x) for x in (out / first["derived"]).read_text().splitlines()]
    assert reqs[9] == {"idx": 9, "input_len": 8064, "output_len": 128, "ttft_ms": pytest.approx(2477.0),
                       "latency_ms": pytest.approx(2477.0 + 127 * 80.0)}
    # the stage-C0 analysis reads it end to end
    a = ca.analyse_session(out)
    assert [b["n_b"] for b in a["boots"]] == [1, 2, 0, 1, 0, 0]
    assert all(x["k_prom_matches"] for x in a["benches"])
    assert a["session"]["outcome"] in ("Rrun", "Rboot")


def test_startup_facts_versions_and_smi_are_recorded(tmp_path, pins, plan):
    world = World(tmp_path)
    out, session, rows = run(world, plan, pins, capture_env=FakeEnv(world, driver="620.10"))
    boot = session["boots"]["b1"]
    assert boot["startup_ok"] is True and boot["startup"]["nondefault"] == {
        "max_num_batched_tokens": 2048, "kv_cache_memory_bytes": 4294967296}
    assert boot["startup"]["log_facts"]["kv_cache_tokens"] == 123128 and boot["startup_s"] == pytest.approx(32.0)
    assert boot["smi"] == "derived/b1-smi.jsonl" and len((out / boot["smi"]).read_text().splitlines()) == 8
    v = session["versions"]
    assert v["driver"] == {"got": "620.10", "phase1": "617.42", "same": False}
    assert v["vllm"]["same"] and v["torch"]["same"]
    assert session["stopped"] is None and len(rows) == 24  # a version drift is recorded, never a stop (rule 9)


def test_boot_order_of_checks_and_teardown(tmp_path, pins, plan):
    world = World(tmp_path)
    run(world, plan, pins)
    ev = world.events
    first = ev.index("popen:1")
    assert ev[:first] == ["env:preflight.json", "apps", "env:b1.before.json", "smi:start"]
    b1 = ev[first:ev.index("popen:2")]
    assert [e for e in b1 if e.startswith("bench")] == ["bench:1-1", "bench:1-2", "bench:1-3", "bench:1-4"]
    assert b1.index("signal:2") > b1.index("bench:1-4") and b1.index("smi:stop") > b1.index("signal:2")
    assert b1[-4:] == ["env:b1.after.json", "apps", "env:b2.before.json", "smi:start"]


# ------------------------------------------------------------------ refusals --

def test_refuses_without_the_lock_or_with_an_unverified_one(tmp_path, pins, plan):
    world = World(tmp_path)
    with pytest.raises(rp.RunError, match="GPU lock"):
        run(world, plan, pins, gpu_lock_held=False)
    world = World(tmp_path, lock=(False,))
    with pytest.raises(rp.RunError, match="does not hold"):
        run(world, plan, pins)
    assert not (tmp_path / "explore-c0-20261016").exists() and world.started == []


def test_refuses_plan_drift_cap_change_toolchain_and_an_existing_session(tmp_path, pins, plan):
    bad = copy.deepcopy(plan)
    bad["N"] = 100
    with pytest.raises(rp.RunError, match="plan"):
        run(World(tmp_path), bad, pins)
    with pytest.raises(rp.RunError, match="cap"):
        run(World(tmp_path), plan, pins, cap_hours=4.0)
    with pytest.raises(rp.RunError, match="toolchain"):
        run(World(tmp_path), plan, pins, toolchain_check=lambda env: ["no ninja"])
    (tmp_path / "explore-c0-20261016").mkdir()
    with pytest.raises(rp.RunError, match="exists"):
        run(World(tmp_path), plan, pins)


def test_refuses_an_output_inside_a_phase_1_results_directory(tmp_path, pins, plan, monkeypatch):
    """028: C0 writes under results/capacity/, never into a Phase 1 results directory (027 §1)."""
    assert set(rp.PHASE1_DIRS) == {"core_a", "core_b", "advanced", "quality", "stage_b", "decision_table", "artifacts"}
    for d in rp.PHASE1_DIRS:
        assert rp.out_problems(ROOT / "results" / d / "explore-c0-20261016"), d
    assert rp.out_problems(ROOT / "results" / "capacity" / "explore-c0-20261016") == []
    monkeypatch.setattr(rp, "ROOT", tmp_path)  # a fake repo: nothing is written into the real results/
    bad, good = tmp_path / "results" / "core_b" / "explore-c0-20261016", tmp_path / "results" / "capacity" / "x"
    world = World(tmp_path)
    for smoke in (False, True):
        with pytest.raises(rp.RunError, match="Phase 1"):
            run(world, plan, pins, out=bad, smoke=smoke)
    with pytest.raises(rp.RunError, match="Phase 1"):  # refused before the smoke, whichever path is wrong
        rp.run_probe(plan, smoke_first=True, smoke_out=good, out=bad, **probe_kwargs(world, pins))
    with pytest.raises(rp.RunError, match="Phase 1"):
        rp.run_probe(plan, smoke_first=True, smoke_out=bad, out=good, **probe_kwargs(world, pins))
    assert not bad.exists() and not good.exists() and world.started == []


# ------------------------------------------------------------------ stop rules --

def test_lock_lost_before_a_boot_stops_the_session(tmp_path, pins, plan):
    world = World(tmp_path, lock=(True, True, False))
    out, session, rows = run(world, plan, pins)
    assert session["stopped"]["rule"] == "lock" and session["stopped"]["boot"] == 2
    assert len(rows) == 4 and len(world.started) == 1


def test_another_compute_process_stops_before_the_boot(tmp_path, pins, plan):
    world = World(tmp_path, apps=("123, python, 900",))
    out, session, rows = run(world, plan, pins)
    assert session["stopped"]["rule"] == "gpu_busy" and rows == [] and world.started == []


def test_memory_must_be_idle_within_60_s(tmp_path, pins, plan):
    world = World(tmp_path, mem=(3000, 3000, 400))
    out, session, rows = run(world, plan, pins)
    assert session["stopped"] is None and len(rows) == 24
    world = World(tmp_path / "busy", mem=(3000,))
    (tmp_path / "busy").mkdir()
    out, session, rows = run(world, plan, pins)
    assert session["stopped"]["rule"] == "gpu_busy" and "memory.used 3000" in session["stopped"]["reason"]
    assert world.t >= rp.IDLE_WAIT_S and world.started == []


def test_no_new_boot_once_elapsed_plus_30_min_exceeds_the_cap(tmp_path, pins, plan):
    world = World(tmp_path, bench_s=600.0)
    out, session, rows = run(world, plan, pins, t_start=-300.0)  # the smoke already used 5 min of the cap
    assert session["stopped"]["rule"] == "cap" and session["stopped"]["boot"] == 5
    assert len(rows) == 16 and all(r["ok"] for r in rows)
    assert session["elapsed_s"] <= 3.0 * 3600


def test_no_bench_that_could_run_past_the_cap(tmp_path, pins, plan):
    world = World(tmp_path, bench_s=600.0)
    out, session, rows = run(world, plan, pins, t_start=-(3.0 * 3600 - 1800 - 1))
    assert session["stopped"]["rule"] == "cap" and len(world.started) == 1
    assert [r["ok"] for r in rows] == [True, True, False, False]
    assert all(r["not_run"] and "cap" in r["reasons"][0] for r in rows[2:])
    assert session["elapsed_s"] <= 3.0 * 3600


@pytest.mark.parametrize("health_s", [0.0, 4.0])
def test_the_server_start_cannot_run_past_the_cap(tmp_path, pins, plan, health_s):
    """028 hard cap: a start that never gets ready is cut at the cap, the idle wait before it included (a /health
    request in flight at the cap still ends; its own duration is the only overrun)."""
    world = World(tmp_path, never_ready=True, mem=(3000,) * 29 + (400,), health_s=health_s)
    out, session, rows = run(world, plan, pins, t_start=-(3.0 * 3600 - 1801))
    assert session["stopped"]["rule"] == "cap" and session["stopped"]["boot"] == 1
    assert session["elapsed_s"] <= 3.0 * 3600 + health_s and world.bench_argvs == []
    assert any("cap" in x for x in session["boots"]["b1"]["reasons"])
    assert len(rows) == 4 and all(r["not_run"] and "cap" in r["reasons"][0] for r in rows)


def test_a_timeout_is_invalid_tears_the_server_down_and_ends_the_boot(tmp_path, pins, plan):
    world = World(tmp_path, behave={(1, 2): "timeout"})
    out, session, rows = run(world, plan, pins)
    r = {x["tag"]: x for x in rows}
    assert r["b1-n1"]["ok"] and not r["b1-n2"]["ok"] and r["b1-n2"]["timeout"] is True
    assert r["b1-n3"]["not_run"] and r["b1-n4"]["not_run"] and not r["b1-n3"]["ok"]
    assert session["boots"]["b1"]["failed"] is True and session["failed_boots"] == 1
    ev = world.events
    assert ev.index("signal:2", ev.index("bench:1-2")) < ev.index("popen:2")
    assert session["stopped"] is None and len(rows) == 24 and sum(x["ok"] for x in rows) == 21


def test_invalid_bench_rules(tmp_path, pins, plan):
    world = World(tmp_path, behave={(1, 1): "exit", (2, 1): "fail", (3, 2): "unhealthy"})
    out, session, rows = run(world, plan, pins)
    r = {x["tag"]: x for x in rows}
    assert not r["b1-n1"]["ok"] and "exit 1" in r["b1-n1"]["reasons"][0] and r["b1-n2"]["ok"]
    assert not r["b2-n1"]["ok"] and any("failed=5" in x for x in r["b2-n1"]["reasons"])
    assert not r["b3-n2"]["ok"] and any("unhealthy" in x for x in r["b3-n2"]["reasons"])
    assert r["b3-n3"]["not_run"] and r["b3-n4"]["not_run"]  # an unhealthy server ends its boot
    assert [session["boots"][f"b{b}"]["failed"] for b in (1, 2, 3)] == [False, False, True]
    assert session["stopped"] is None and len(rows) == 24


def test_wrong_blocks_or_non_default_args_fail_the_startup(tmp_path, pins, plan):
    world = World(tmp_path, blocks=247)
    out, session, rows = run(world, plan, pins)
    assert session["boots"]["b1"]["startup_ok"] is False
    assert any("247" in x for x in session["boots"]["b1"]["reasons"])
    assert world.bench_argvs == [] and all(r["not_run"] for r in rows)
    assert session["stopped"]["rule"] == "failed_boots" and len(world.started) == 2 and len(rows) == 8
    for edit, what in ((lambda nd: {**nd, "otlp_traces_endpoint": "http://127.0.0.1:4318/v1/traces"}, "otlp"),
                       (lambda nd: {**nd, "max_num_batched_tokens": 4096}, "4096"),
                       (lambda nd: {k: v for k, v in nd.items() if k != "kv_cache_memory_bytes"}, "kv_cache")):
        d = tmp_path / what
        d.mkdir()
        world = World(d, nondefault=edit)
        out, session, rows = run(world, plan, pins)
        assert session["boots"]["b1"]["startup_ok"] is False and world.bench_argvs == [], what
        assert any("non-default" in x for x in session["boots"]["b1"]["reasons"]), what


def test_server_never_ready_fails_the_boot(tmp_path, pins, plan):
    world = World(tmp_path, never_ready=True)
    out, session, rows = run(world, plan, pins, startup_timeout_s=60)
    assert session["boots"]["b1"]["startup_ok"] is False and "not ready" in session["boots"]["b1"]["reasons"][0]
    assert session["stopped"]["rule"] == "failed_boots"


def test_one_failed_boot_does_not_stop_two_do(tmp_path, pins, plan):
    world = World(tmp_path, behave={(2, 1): "exit", (2, 3): "exit"})
    out, session, rows = run(world, plan, pins)
    assert session["failed_boots"] == 1 and session["stopped"] is None and len(rows) == 24
    world = World(tmp_path / "two", behave={(2, 1): "exit", (2, 3): "exit", (5, 2): "exit", (5, 4): "fail"})
    (tmp_path / "two").mkdir()
    out, session, rows = run(world, plan, pins)
    assert session["failed_boots"] == 2 and session["stopped"]["rule"] == "failed_boots"
    assert len(rows) == 20 and len(world.started) == 5


def test_temperature_88_c_stops_immediately(tmp_path, pins, plan):
    world = World(tmp_path, behave={(2, 3): "hot"})
    out, session, rows = run(world, plan, pins)
    r = {x["tag"]: x for x in rows}
    assert not r["b2-n3"]["ok"] and r["b2-n3"]["stopped_hot"] is True and r["b2-n4"]["not_run"]
    assert session["stopped"]["rule"] == "temperature" and len(world.started) == 2 and len(rows) == 8
    ev = world.events
    assert ev.index("signal:2", ev.index("bench:2-3")) < ev.index("smi:stop", ev.index("bench:2-3"))
    world = World(tmp_path / "after", behave={(1, 2): "hot_after"})
    (tmp_path / "after").mkdir()
    out, session, rows = run(world, plan, pins)
    assert [x["ok"] for x in rows] == [True, True, False, False] and session["stopped"]["rule"] == "temperature"
    world = World(tmp_path / "start", hot_at_start=(1,))
    (tmp_path / "start").mkdir()
    out, session, rows = run(world, plan, pins)
    assert session["stopped"]["rule"] == "temperature" and world.bench_argvs == []


def test_lost_telemetry_stops_like_the_88_c_rule(tmp_path, pins, plan):
    """028 rule 6 needs the temperature: a sampler that dies or goes stale stops the session ("telemetry"), it is
    never a silent gap."""
    world = World(tmp_path, behave={(2, 3): "lost"})
    out, session, rows = run(world, plan, pins)
    r = {x["tag"]: x for x in rows}
    assert not r["b2-n3"]["ok"] and r["b2-n3"]["stopped_telemetry"] is True and r["b2-n3"]["stopped_hot"] is False
    assert any("telemetry" in x for x in r["b2-n3"]["reasons"]) and r["b2-n4"]["not_run"]
    assert r["b2-n2"]["ok"] and r["b2-n2"]["stopped_telemetry"] is False
    assert session["stopped"]["rule"] == "telemetry" and SMI_EXITED in session["stopped"]["reason"]
    assert len(world.started) == 2 and len(rows) == 8
    ev = world.events
    assert ev.index("signal:2", ev.index("bench:2-3")) < ev.index("smi:stop", ev.index("bench:2-3"))
    world = World(tmp_path / "start", lost_at_start=(1,))
    (tmp_path / "start").mkdir()
    out, session, rows = run(world, plan, pins)
    assert session["stopped"]["rule"] == "telemetry" and world.bench_argvs == []
    assert any("telemetry" in x for x in session["boots"]["b1"]["reasons"])


@pytest.mark.parametrize("kind, rule", [("hot", "temperature"), ("lost", "telemetry")])
def test_a_stop_flag_set_during_the_teardown_still_stops(tmp_path, pins, plan, kind, rule):
    world = World(tmp_path, halt_at_teardown={1: kind})
    out, session, rows = run(world, plan, pins)
    assert session["stopped"]["rule"] == rule and session["stopped"]["boot"] == 1
    assert "teardown" in session["stopped"]["reason"]
    assert len(world.started) == 1 and len(rows) == 4 and all(r["ok"] for r in rows)


def test_a_harness_exception_tears_down_and_records_the_abort(tmp_path, pins, plan):
    world = World(tmp_path, behave={(1, 2): "raise"})
    with pytest.raises(RuntimeError, match="boom"):
        run(world, plan, pins)
    out = tmp_path / "explore-c0-20261016"
    session = json.loads((out / "session.json").read_text())
    assert "boom" in session["aborted"] and session["stopped"]["rule"] == "aborted"
    assert "signal:2" in world.events and world.smi.stopped and (out / "raw_manifest.json").is_file()


def test_no_data_driven_stop(tmp_path, pins, plan):
    """028 rule 7: latency never decides anything; all-high and all-low sessions both run 24 benches."""
    every = {(b, j): (8, 16) for b in range(1, 7) for j in range(1, 5)}
    for name, slips in (("high", every), ("low", {})):
        (tmp_path / name).mkdir()
        out, session, rows = run(World(tmp_path / name, slips=slips), plan, pins)
        assert len(rows) == 24 and session["stopped"] is None
    assert "c0_analysis" not in imported_modules()


# ------------------------------------------------------------------ smoke --

def test_smoke_is_one_boot_one_bench_at_n_16_and_never_analysed(tmp_path, pins, plan):
    world = World(tmp_path)
    out, session, rows = run(world, plan, pins, smoke=True)
    assert out.name == "smoke-c0-20261016" and len(world.started) == 1 and len(rows) == 1 and rows[0]["ok"]
    assert (session["stage"], session["purpose"], session["formal"]) == ("C0-smoke", "smoke", False)
    (argv,) = world.bench_argvs
    core = bp.bench_argv(pins, L=8192, C=8, mix="uniform", result_dir=str(out / "raw"),
                         result_filename="b1-n1.bench.json", port=rp.PORT, vllm_bin=".venv/bin/vllm")
    assert [i for i, (x, y) in enumerate(zip(argv, core)) if x != y] == [core.index("--num-prompts") + 1]
    assert argv[argv.index("--num-prompts") + 1] == "16" and argv[argv.index("--max-concurrency") + 1] == "8"
    assert rows[0]["completed"] == 16 and rows[0]["num_prompts"] == 16
    with pytest.raises(ca.AnalysisError, match="stage"):
        ca.analyse_session(out)


def test_run_probe_smoke_first_then_the_probe_in_one_cap(tmp_path, pins, plan):
    world = World(tmp_path)
    common = probe_kwargs(world, pins)
    smoke_out, out = tmp_path / "smoke-c0-20261016", tmp_path / "explore-c0-20261016"
    sessions = rp.run_probe(plan, smoke_first=True, smoke_out=smoke_out, out=out, **common)
    assert [s["stage"] for s in sessions] == ["C0-smoke", "C0"] and len(world.started) == 7
    probe = json.loads((out / "session.json").read_text())
    assert probe["elapsed_at_start_s"] > 0  # the cap clock started before the smoke


def test_a_failed_smoke_never_starts_the_probe(tmp_path, pins, plan):
    world = World(tmp_path, behave={(1, 1): "exit"})
    common = probe_kwargs(world, pins)
    smoke_out, out = tmp_path / "smoke-c0-20261016", tmp_path / "explore-c0-20261016"
    with pytest.raises(rp.RunError, match="smoke"):
        rp.run_probe(plan, smoke_first=True, smoke_out=smoke_out, out=out, **common)
    assert smoke_out.is_dir() and not out.exists() and len(world.started) == 1


@pytest.mark.parametrize("smi", ["none", "no_temp"])
def test_a_smoke_without_temperature_samples_never_starts_the_probe(tmp_path, pins, plan, smi):
    """The smoke checks the sampler: without a temperature the 88 C stop (028 rule 6) could not work."""
    world = World(tmp_path, smi=smi)
    common = probe_kwargs(world, pins)
    smoke_out, out = tmp_path / "smoke-c0-20261016", tmp_path / "explore-c0-20261016"
    with pytest.raises(rp.RunError, match="temperature"):
        rp.run_probe(plan, smoke_first=True, smoke_out=smoke_out, out=out, **common)
    assert smoke_out.is_dir() and not out.exists() and len(world.started) == 1


def probe_kwargs(world, pins):
    return dict(pins=pins, pins_sha256=PINS_SHA, popen=world.popen, signal_group=world.signal_group,
                group_alive=world.group_alive, http_get=world.http_get, run_bench=world.run_bench,
                gpu_mem_used=world.gpu_mem_used, compute_apps=world.compute_apps, start_smi=world.start_smi,
                lock_held_check=world.lock_held, gpu_lock_held=True, toolchain_check=lambda env: [],
                capture_env=FakeEnv(world), base_env=BASE_ENV, cap_hours=3.0, sample_interval=None, t_start=0.0,
                now=world.now, sleep=world.sleep, wall=world.wall)


def test_main_plan_and_refusal_without_the_lock(tmp_path, capsys):
    assert rp.main(["plan", "--out", str(tmp_path / "c0-plan.json")]) == 0
    p = json.loads((tmp_path / "c0-plan.json").read_text())
    assert p["stage"] == "C0" and len(p["order"]) == 24
    out, smoke_out = tmp_path / "explore", tmp_path / "smoke"
    assert rp.main(["run", "--smoke-first", "--out", str(out), "--smoke-out", str(smoke_out),
                    "--plan", str(tmp_path / "c0-plan.json")]) == 2
    assert "GPU lock" in capsys.readouterr().err and not out.exists() and not smoke_out.exists()
    p["N"] = 100
    (tmp_path / "bad.json").write_text(json.dumps(p))
    assert rp.main(["run", "--gpu-lock-held", "--out", str(out), "--plan", str(tmp_path / "bad.json")]) == 2
    assert not out.exists()


# ------------------------------------------------------------------ helpers: nvidia-smi, bench runner, imports --

def imported_modules():
    tree = ast.parse(Path(rp.__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_imports_only_stdlib_and_the_frozen_modules():
    project = imported_modules() - set(sys.stdlib_module_names)
    assert project <= {"core_b_plan", "run_adv", "run_core_b", "run_core_a", "coreb_smoke"}


def test_parse_smi_line_handles_units_na_and_hex():
    line = "1791700000123 2026/10/16 00:00:00.123, P1, 2850, 14001, 250.31, 65, 100, 14000, 0x0000000000000404"
    assert rp.parse_smi_line(line) == {"t": 1791700000.123, "pstate": "P1", "sm_mhz": 2850.0, "mem_mhz": 14001.0,
                                       "power_w": 250.31, "temp_c": 65.0, "util_pct": 100.0, "mem_used_mib": 14000.0,
                                       "event_reasons": 0x404}
    na = "1791700000123 2026/10/16 00:00:00.123, P8, [N/A], 405, [Not Supported], 40, 0, 600, [N/A]"
    got = rp.parse_smi_line(na)
    assert got["sm_mhz"] is None and got["power_w"] is None and got["event_reasons"] is None and got["temp_c"] == 40.0
    assert rp.parse_smi_line("1791700000123 Failed to initialize NVML") is None
    assert rp.parse_smi_line("no stamp, P1, 1, 2, 3, 4, 5, 6, 0x0") is None
    assert rp.smi_argv() == ["nvidia-smi", "--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,"
                             "temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active",
                             "--format=csv,noheader,nounits", "-lms", "1000"]


class FakeSmiProc:
    def __init__(self, lines, exited=None):
        self.stdout = io.BytesIO("".join(x + "\n" for x in lines).encode())
        self.terminated = False
        self.returncode = exited

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        pass


def test_smi_sampler_stamps_lines_and_flags_88_c(tmp_path):
    rows = ["2026/10/16 00:00:00.000, P1, 2850, 14001, 250.0, 70, 100, 14000, 0x0000000000000000",
            "2026/10/16 00:00:01.000, P1, 2850, 14001, 250.0, 88, 100, 14000, 0x0000000000000020"]
    procs = []

    def popen(argv, **kw):
        assert argv == rp.smi_argv() and kw["start_new_session"] is True
        procs.append(FakeSmiProc(rows))
        return procs[-1]

    smi = rp.SmiSampler(tmp_path / "b1.smi.csv", popen=popen, wall=lambda: 1791700000.5).start()
    assert smi.stop() == [] and procs[0].terminated
    assert smi.hot.is_set() and smi.halt.is_set() and not smi.lost.is_set()
    assert (tmp_path / "b1.smi.csv").read_text().splitlines()[0] == "1791700000500 " + rows[0]
    assert [s["temp_c"] for s in smi.samples()] == [70.0, 88.0]
    cool = rp.SmiSampler(tmp_path / "b2.smi.csv", popen=lambda argv, **kw: FakeSmiProc(rows[:1])).start()
    cool.stop()
    assert not cool.hot.is_set()


def test_smi_sampler_that_cannot_start_is_a_problem_not_a_crash(tmp_path):
    def popen(argv, **kw):
        raise FileNotFoundError("nvidia-smi")

    smi = rp.SmiSampler(tmp_path / "b1.smi.csv", popen=popen).start()
    assert smi.lost.is_set() and smi.halt.is_set() and "did not start" in smi.lost_reason  # no telemetry: a stop
    problems = smi.stop()
    assert problems and "nvidia-smi" in problems[0] and smi.samples() == [] and not smi.hot.is_set()


def test_smi_sampler_that_exits_loses_the_telemetry(tmp_path):
    """An nvidia-smi that rejects a field prints one line and exits: the 88 C stop would silently be off."""
    proc = FakeSmiProc(['Field "clocks_event_reasons.active" is not a valid field to query.'], exited=2)
    smi = rp.SmiSampler(tmp_path / "b1.smi.csv", popen=lambda argv, **kw: proc, poll_s=0.01).start()
    assert smi.halt.wait(5) and smi.lost.is_set() and not smi.hot.is_set()
    assert smi.lost_reason == "the nvidia-smi sampler exited (code 2)"
    assert smi.stop() == [smi.lost_reason] and smi.samples() == []


class PipeSmiProc:
    """An nvidia-smi that stays alive (its pipe open) until terminated."""

    def __init__(self, lines):
        r, self.w = os.pipe()
        os.write(self.w, "".join(x + "\n" for x in lines).encode())
        self.stdout, self.returncode = os.fdopen(r, "rb"), None

    def poll(self):
        return self.returncode

    def terminate(self):
        os.close(self.w)
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def test_smi_sampler_without_a_temperature_for_stale_s_loses_the_telemetry(tmp_path):
    na = "2026/10/16 00:00:00.000, P1, 2850, 14001, 250.0, [N/A], 100, 14000, 0x0000000000000000"
    smi = rp.SmiSampler(tmp_path / "b1.smi.csv", popen=lambda argv, **kw: PipeSmiProc([na]), poll_s=0.01,
                        stale_s=0.2).start()
    assert smi.halt.wait(5) and not smi.hot.is_set()
    assert smi.lost_reason == "no temperature sample from nvidia-smi for more than 0.2 s"
    assert smi.stop() == [smi.lost_reason] and len(smi.samples()) == 1
    ok = "2026/10/16 00:00:00.000, P1, 2850, 14001, 250.0, 60, 100, 14000, 0x0000000000000000"
    alive = rp.SmiSampler(tmp_path / "b2.smi.csv", popen=lambda argv, **kw: PipeSmiProc([ok]), poll_s=0.01,
                          stale_s=5).start()
    time.sleep(0.2)
    assert not alive.halt.is_set() and alive.stop() == [] and not alive.lost.is_set()


def test_run_bench_watched_exit_timeout_and_stop(tmp_path):
    env = {"PATH": "/usr/bin:/bin"}
    stop = threading.Event()
    assert rp.run_bench_watched(["true"], tmp_path / "a.log", env, 5, stop, poll_s=0.05) == (0, False, False)
    t = time.monotonic()
    rc, timed_out, stopped = rp.run_bench_watched(["sleep", "30"], tmp_path / "b.log", env, 0.3, stop, poll_s=0.05)
    assert timed_out and not stopped and rc == -signal.SIGKILL and time.monotonic() - t < 5
    threading.Timer(0.2, stop.set).start()
    rc, timed_out, stopped = rp.run_bench_watched(["sleep", "30"], tmp_path / "c.log", env, 20, stop, poll_s=0.05)
    assert stopped and not timed_out and rc == -signal.SIGKILL


def test_nondefault_problems_need_exactly_mnbt_and_k():
    k = 4294967296
    assert rp.nondefault_problems({"max_num_batched_tokens": 2048, "kv_cache_memory_bytes": k}, kv_bytes=k) == []
    assert rp.nondefault_problems(None, kv_bytes=k)
    assert rp.nondefault_problems({"max_num_batched_tokens": 2048}, kv_bytes=k)
    assert rp.nondefault_problems({"max_num_batched_tokens": 2048, "kv_cache_memory_bytes": k,
                                   "enable_logging_iteration_details": True}, kv_bytes=k)
    assert rp.nondefault_problems({"max_num_batched_tokens": 2048.0, "kv_cache_memory_bytes": k}, kv_bytes=k)
