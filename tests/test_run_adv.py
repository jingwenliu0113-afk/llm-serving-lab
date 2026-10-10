"""Tests for benchmarks/run_adv.py: the Advanced runner (decision 024; PREREGISTRATION §7), with a fake
server, /metrics, OTLP spans, clock and `vllm bench serve` (no GPU). Real execution happens in WSL under the
GPU lock after smoke FS1-FS5."""

import copy
import gzip
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import adv_analysis as aa
import adv_decomp as ad
import adv_plan as ap
import core_b_analysis as ba
import core_b_plan as bp
import coreb_smoke as smoke
import run_adv as ra
import run_core_b as rb

ROOT = Path(__file__).resolve().parent.parent
EPOCH = 1_760_000_000.0
PINS_SHA = "ab" * 32
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u", "VLLM_LOGGING_LEVEL": "DEBUG",
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "grpc", "OTEL_SDK_DISABLED": "true"}
CONTRACT_KEYS = {"cond", "L", "mix", "restart", "block", "C", "N", "ok", "reasons", "timeout", "skipped",
                 "short_class_p95_ttft_ms", "long_class_p95_ttft_ms", "preemptions", "kv_peak", "manip", "decomp"}


# ------------------------------------------------------------------ pins / plans --

@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def pins_k3(pins):
    """The repo pins with the K3 block count measured (smoke FS1 sets it; 186 is the 024 prediction)."""
    p = copy.deepcopy(pins)
    p["advanced"]["num_gpu_blocks_k3"] = 186
    return p


@pytest.fixture(scope="module")
def formal_pins(pins_k3):
    p = copy.deepcopy(pins_k3)
    p["freeze"]["advanced"] = True
    return p


def uniform_session(pins, knee=None):
    """A formal, env-valid Core uniform session whose output throughput saturates at C_knee 25 / 8."""
    knee = knee or {2048: 25, 8192: 8}
    up = bp.build_uniform_plan(pins, num_gpu_blocks=248, block_size=528)
    rows = [{"L": L, "mix": "uniform", "restart": r, "C": C, "ok": True, "output_throughput": 100.0 * min(C, knee[L]),
             "short_class_p95_ttft_ms": 80.0}
            for L in (2048, 8192) for r in (1, 2, 3) for C in up["grids"][str(L)]]
    return {"session_id": "formal-20261009-coreb-uniform", "formal": True, "env_valid": True}, up, rows


def make_plan(pins, sha=PINS_SHA):
    session, up, rows = uniform_session(pins)
    return ap.build_plan(pins, session=session, uniform_plan=up, rows=rows, pins_sha256=sha)


def sub_plan(pins, keep):
    plan = make_plan(pins)
    plan["boots"] = [b for b in plan["boots"] if keep(b)]
    return plan


def key(b):
    return b["cond"], b["L"], b["mix"], b["restart"]


# ------------------------------------------------------------------ fake world --

def stamp(t: float, line: str) -> str:
    return f"{int(t * 1000)} {line}"


def iter_line(j, n_tok, u=50.0):
    return (f"(APIServer pid=1) INFO 10-10 12:00:00 [loggers.py:185] Engine 000: Iteration({j}): 1 context requests, "
            f"{n_tok} context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 10.00 ms, "
            f"GPU KV cache usage: {u:.1f}%")


def nondefault_from_argv(argv):
    """What vLLM 0.30.0 logs as non-default args for a server argv (api_utils.py:286, a dict repr)."""
    a = dict(zip(argv, argv[1:]))
    d = {"model_tag": argv[2], "max_num_batched_tokens": int(a["--max-num-batched-tokens"]),
         "attention_backend": "FLASH_ATTN", "kv_cache_memory_bytes": int(a["--kv-cache-memory-bytes"]),
         "otlp_traces_endpoint": a["--otlp-traces-endpoint"]}
    if "--enable-logging-iteration-details" in argv:
        d["enable_logging_iteration_details"] = True
    if "--long-prefill-token-threshold" in a:
        d["long_prefill_token_threshold"] = int(a["--long-prefill-token-threshold"])
    if "--no-scheduler-reserve-full-isl" in argv:
        d["scheduler_reserve_full_isl"] = False
    return d


def server_log(argv=None, nondefault=None):
    nd = nondefault if nondefault is not None else nondefault_from_argv(argv)
    return [f"(APIServer pid=1) INFO 10-10 12:00:00 [api_utils.py:286] non-default args: {nd!r}",
            "(EngineCore pid=2) INFO 10-10 12:00:01 [cuda.py:478] Using AttentionBackendEnum.FLASH_ATTN backend.",
            "(EngineCore pid=2) INFO 10-10 12:00:02 [kv_cache_utils.py:2395] GPU KV cache size: 130,944 tokens, "
            "Maximum concurrency for 32,768 tokens per request: 4.00x"]


def cond_of(argv):
    a = dict(zip(argv, argv[1:]))
    if a["--kv-cache-memory-bytes"] == "3221225472":
        return "K3"
    t, g = "--long-prefill-token-threshold" in a, "--no-scheduler-reserve-full-isl" in argv
    return {(False, False): "B0", (True, False): "T", (False, True): "G", (True, True): "TG"}[(t, g)]


def json_decode(body, gz):
    return json.loads(gzip.decompress(body) if gz else body)


class FakeProc:
    pid = 77

    def __init__(self):
        self.returncode = None
        self.log_reader = FakeReader()

    def poll(self):
        return self.returncode


class FakeReader:
    """Stands in for run_adv.LogReader (the World writes the server log itself)."""

    def __init__(self):
        self.error = None

    def join(self, timeout):
        return True


DISK_FULL = "OSError: [Errno 28] No space left on device"


class FakeReceiver(ra.OtlpReceiver):
    """The real receiver logic (raw bodies, JSONL, counting) without binding a socket."""

    def __init__(self, world, *a, **kw):
        super().__init__(*a, decode=json_decode, **kw)
        self.world, self.started, self.closed = world, False, False

    def start(self):
        self.world.events.append("receiver:start")
        self.started = True
        return self

    def close(self):
        self.world.events.append("receiver:close")
        self.closed = True


class FakeEnv:
    """Fake scripts/env_capture.py --vllm-python record."""

    def __init__(self, enforced=300.0, vllm="0.30.0", dirty=False, world=None):
        self.enforced, self.vllm, self.dirty, self.world, self.calls = enforced, vllm, dirty, world, []

    def __call__(self, out_path, gpu_lock_held):
        self.calls.append(Path(out_path).name)
        if self.world is not None:
            self.world.events.append(f"env:{Path(out_path).name}")
        rec = {"runtime": {"vllm": {"vllm": self.vllm}}, "repo": {"commit": "c0ffee", "dirty": self.dirty},
               "gpu": {"nvidia": [{"enforced_power_limit_w": self.enforced, "power_default_limit_w": 300.0}]},
               "probe_errors": []}
        Path(out_path).write_text(json.dumps(rec))
        return rec


class World:
    """Fake WSL: one server per boot whose log, /metrics and OTLP spans follow its argv, and a bench that
    writes a --save-detailed result. behave[(cond, L, mix, restart)] in {"ok", "timeout", "exit", "fail",
    "short_spans", "big_chunks", "receiver_write_fails", "log_write_fails"}.
    stubborn: the server group only goes on SIGKILL, and while teardown waits after its SIGINT one interrupt
    arrives: "signal" (a real SIGINT to this process), "raise" (KeyboardInterrupt raised by sleep itself) or
    "sigterm" (a real SIGTERM). mem_fail {call number: exception} makes nvidia-smi fail; log_error ("start",
    "teardown") the server log reader; receiver_fail ("start", "teardown") a receiver write."""

    def __init__(self, tmp_path, *, blocks=None, log_edit=None, behave=None, preempt=None, mem=400,
                 metrics_error=None, stubborn=None, mem_fail=None, log_error=None, receiver_fail=None):
        self.tmp_path, self.blocks, self.log_edit = tmp_path, blocks or {}, log_edit
        self.behave, self.preempt, self.mem, self.metrics_error = behave or {}, preempt or {}, mem, metrics_error
        self.stubborn, self.mem_fail, self.log_error = stubborn, mem_fail or {}, log_error
        self.receiver_fail, self.interrupted, self.mem_calls = receiver_fail, False, 0
        self.t, self.events, self.started, self.server_envs, self.signals = 0.0, [], [], [], []
        self.bench_argvs, self.bench_timeouts, self.bench_envs, self.receivers = [], [], [], []
        self.counters = dict.fromkeys(("q_count", "q_sum", "p_count", "p_sum", "prompt", "gen", "pre"), 0.0)
        self.proc = self.log_path = self.cond = None
        self.iter = 0
        self.metrics_calls = 0

    # clocks
    def wall(self):
        return EPOCH + self.t

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.stubborn and signal.SIGINT in self.signals and not self.interrupted:
            self.interrupted = True  # one more interrupt while teardown waits for the group
            if self.stubborn == "raise":
                raise KeyboardInterrupt
            signal.raise_signal(signal.SIGTERM if self.stubborn == "sigterm" else signal.SIGINT)

    def fail_receiver_post(self):
        """One OTLP POST that the receiver cannot store (its JSONL directory is gone)."""
        rec = self.receivers[-1]
        rec.spans_path = self.tmp_path / "no-such-dir" / "spans.jsonl"
        assert rec.receive(json.dumps([{"name": "llm_request", "attributes": {}}]).encode(), None) == 200

    # server
    def receiver_factory(self, raw_dir, spans_path, prefix):
        r = FakeReceiver(self, raw_dir, spans_path, prefix=prefix)
        self.receivers.append(r)
        return r

    def popen(self, argv, env, log_path):
        self.events.append("popen")
        self.started.append(argv)
        self.server_envs.append(env)
        self.cond, self.log_path, self.iter = cond_of(argv), Path(log_path), 0
        lines = server_log(argv)
        text = "\n".join(lines)
        if self.log_edit:
            text = self.log_edit(self.cond, text)
        self.log_path.write_text("".join(stamp(self.wall(), x) + "\n" for x in text.splitlines()))
        self.proc = FakeProc()
        if self.log_error == "start":
            self.proc.log_reader.error = DISK_FULL
        if self.receiver_fail == "start":
            self.fail_receiver_post()  # e.g. during the ready check
        return self.proc

    def signal_group(self, proc, sig):
        self.events.append(f"signal:{int(sig)}")
        self.signals.append(sig)
        if sig == signal.SIGINT and self.log_error == "teardown":
            proc.log_reader.error = DISK_FULL
        if sig == signal.SIGINT and self.receiver_fail == "teardown":
            self.fail_receiver_post()  # the BatchSpanProcessor flush on shutdown
        if not self.stubborn or sig == signal.SIGKILL:
            proc.returncode = 0

    def group_alive(self, proc):
        return proc.returncode is None

    def gpu_mem_used(self):
        self.mem_calls += 1
        if self.mem_calls in self.mem_fail:
            raise self.mem_fail[self.mem_calls]
        return self.mem

    def metrics_text(self):
        nb = self.blocks.get(self.cond, 186 if self.cond == "K3" else 248)
        c = self.counters
        lab = 'engine="0",model_name="m"'
        return "\n".join([
            "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig",
            f'vllm:cache_config_info{{block_size="528",cache_dtype="auto",engine="0",num_gpu_blocks="{nb}"}} 1.0',
            f"vllm:kv_cache_usage_perc{{{lab}}} 0.5",
            f"vllm:num_requests_running{{{lab}}} 3.0",
            f"vllm:num_requests_waiting{{{lab}}} 1.0",
            f"vllm:num_preemptions_total{{{lab}}} {c['pre']}",
            f"vllm:prompt_tokens_total{{{lab}}} {c['prompt']}",
            f"vllm:generation_tokens_total{{{lab}}} {c['gen']}",
            f'vllm:request_queue_time_seconds_bucket{{{lab},le="1.0"}} {c["q_count"]}',
            f'vllm:request_queue_time_seconds_bucket{{{lab},le="+Inf"}} {c["q_count"]}',
            f"vllm:request_queue_time_seconds_sum{{{lab}}} {c['q_sum']}",
            f"vllm:request_queue_time_seconds_count{{{lab}}} {c['q_count']}",
            f'vllm:request_prefill_time_seconds_bucket{{{lab},le="+Inf"}} {c["p_count"]}',
            f"vllm:request_prefill_time_seconds_sum{{{lab}}} {c['p_sum']}",
            f"vllm:request_prefill_time_seconds_count{{{lab}}} {c['p_count']}", ""])

    def http_get(self, url):
        if url.endswith("/health"):
            self.events.append("health")
            return 200, ""
        self.events.append("metrics")
        self.metrics_calls += 1
        if self.metrics_error is not None and self.metrics_calls == self.metrics_error:
            raise RuntimeError("boom: /metrics broke")
        return 200, self.metrics_text()

    # client
    def run_bench(self, argv, log_path, env, timeout_s):
        self.events.append("bench")
        self.bench_argvs.append(argv)
        self.bench_timeouts.append(timeout_s)
        self.bench_envs.append(env)
        a = dict(zip(argv, argv[1:]))
        prefix = a["--request-id-prefix"]
        _, cond, L, mix, r, _ = prefix.split("-")
        L, k = int(L), (cond, int(L), mix, int(r[1:]))
        how = self.behave.get(k, "ok")
        Path(log_path).write_text("bench log\n")
        if how == "log_write_fails":  # the reader died, the server stalled on its full pipe: the bench timed out
            self.proc.log_reader.error = DISK_FULL
            self.t += timeout_s
            return -9, True
        if how == "timeout":
            self.t += timeout_s
            return -9, True
        if how == "exit":
            return 1, False
        late = {"exit_with_result": (1, False), "timeout_with_result": (-9, True)}.get(how)
        N = int(a["--num-prompts"])
        lens = [bp.LONG_INPUT if mix == "mixed" and i % 10 == 4 else L - bp.N_GEN for i in range(N)]
        t_s = self.t
        reqs = [("cmpl-readycheck-0", L - bp.N_GEN, 0.1), ("cmpl-0a1b2c-0", L - bp.N_GEN, 0.2)]
        reqs += [(ad.request_id(prefix, i), n, 0.5 + 0.01 * i) for i, n in enumerate(lens)]
        chunk = 2048 if how == "big_chunks" or cond not in ("T", "TG") else 512
        lines, k_step = [], 0
        for _, tokens, _ in reqs:
            left = tokens
            while left > 0:
                n = min(chunk, left)
                left -= n
                lines.append(stamp(EPOCH + t_s + 0.3 + 0.0005 * k_step, iter_line(self.iter, n)))
                self.iter += 1
                k_step += 1
        with open(self.log_path, "a") as fh:
            fh.write("".join(x + "\n" for x in lines))
        spans = [{"name": "llm_request", "trace_id": "", "span_id": "",
                  "start_time_unix_nano": int(round((EPOCH + t_s + off) * 1e9)), "end_time_unix_nano": 0,
                  "attributes": {"gen_ai.request.id": rid, "gen_ai.usage.prompt_tokens": tokens,
                                 "gen_ai.usage.completion_tokens": bp.N_GEN, "gen_ai.latency.time_in_queue": 0.0,
                                 "gen_ai.latency.time_in_model_prefill": 0.05,
                                 "gen_ai.latency.time_to_first_token": 0.05, "gen_ai.latency.e2e": 1.0}}
                 for rid, tokens, off in reqs]
        posted = spans[:-3] if how == "short_spans" else spans
        rec = self.receivers[-1]
        if how == "receiver_write_fails":
            rec.spans_path = self.tmp_path / "no-such-dir" / "spans.jsonl"
        half = len(posted) // 2
        assert rec.receive(json.dumps(posted[:half] + [{"name": "other", "attributes": {}}]).encode(), None) == 200
        assert rec.receive(gzip.compress(json.dumps(posted[half:]).encode()), "gzip") == 200
        c = self.counters
        c["q_count"] += len(reqs)
        c["p_count"] += len(reqs)
        c["p_sum"] += 0.05 * len(reqs)
        c["prompt"] += sum(t for _, t, _ in reqs)
        c["gen"] += bp.N_GEN * len(reqs)
        c["pre"] += self.preempt.get(k, 0)
        failed = 5 if how == "fail" else 0
        base = 0.2 if mix == "mixed" else 0.06
        ttfts = [(0.5 if n == bp.LONG_INPUT else base) + 0.0002 * i for i, n in enumerate(lens)]
        res = {"completed": N - failed, "failed": failed, "num_prompts": N, "duration": 12.0,
               "request_throughput": N / 12, "output_throughput": N * 128 / 12, "total_token_throughput": 1e5,
               "input_lens": lens, "output_lens": [bp.N_GEN] * N, "ttfts": ttfts,
               "latencies": [x + 1.0 for x in ttfts], "errors": [""] * N,
               **{f"{s}_{m}_ms": 1.0 for s in ("mean", "median", "p50", "p95", "p99")
                  for m in ("ttft", "tpot", "itl", "e2el", "client_queue_time")}}
        Path(a["--result-dir"], a["--result-filename"]).write_text(json.dumps(res))
        self.t += 12.0
        return late or (0, False)


def datasets(tmp_path):
    """Stand-in mixed dataset files (the world's bench never reads them; formal runs pin their sha256)."""
    d = Path(tmp_path) / "data"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for L in (2048, 8192):
        (d / f"mixed-{L}.jsonl").write_text(f'{{"prompt": "x{L}", "output_tokens": 128}}\n')
        out[L] = str(d / f"mixed-{L}.jsonl")
    return out


def run(world, pins, plan, out=None, **kw):
    out = out or world.tmp_path / "session"
    kw.setdefault("mixed_datasets", datasets(world.tmp_path))
    kw.setdefault("gpu_lock_held", True)
    kw.setdefault("lock_held_check", lambda: True)
    kw.setdefault("toolchain_check", lambda env: [])
    kw.setdefault("capture_env", FakeEnv(world=world))
    kw.setdefault("gpu_memory_utilization", "0.91")
    kw.setdefault("pins_sha256", PINS_SHA)
    kw.setdefault("sample_interval", None)
    kw.setdefault("base_env", BASE_ENV)
    summary = ra.run_session(plan, pins=pins, out_dir=out, gdn_prefill_backend="triton", session_id=out.name,
                             popen=world.popen, signal_group=world.signal_group, group_alive=world.group_alive,
                             http_get=world.http_get, run_bench=world.run_bench, gpu_mem_used=world.gpu_mem_used,
                             gpu_lock_free=lambda: True, receiver_factory=world.receiver_factory,
                             now=world.now, sleep=world.sleep, wall=world.wall, **kw)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    return out, summary, rows


def first_block(pins, n=None, conds=None):
    """The first n boots of block 1 (optionally only some conditions), in plan order."""
    plan = make_plan(pins)
    boots = [b for b in plan["boots"] if b["block"] == 1 and (conds is None or b["cond"] in conds)]
    plan["boots"] = boots[:n] if n else boots
    return plan


# ------------------------------------------------------------------ log reader --

def test_pump_log_stamps_every_line_and_the_core_parsers_still_extract_facts(pins):
    argv, _ = ap.server_argv_for("TG", pins, gdn_prefill_backend="triton", gpu_memory_utilization="0.91", port=8000)
    raw = "\n".join(server_log(argv)) + "\r\n" + iter_line(7, 512) + "\n" + "bad byte \xff\n"
    stream = io.BytesIO(raw.encode("latin-1"))
    ticks = iter([EPOCH + 0.001 * i for i in range(100)])
    out = io.StringIO()
    n = ra.pump_log(stream, out, wall=lambda: next(ticks))
    text = out.getvalue()
    lines = text.splitlines()
    assert n == 5 and len(lines) == 5
    assert all(x.split(" ", 1)[0].isdigit() for x in lines) and lines[0].startswith(f"{int(EPOCH * 1000)} (APIServer")
    assert not any(x.endswith("\r") for x in lines)
    facts = smoke.parse_server_log(text)
    assert facts["attention_backend"] == "FLASH_ATTN" and facts["kv_cache_tokens"] == 130944
    assert int(rb._RE_NONDEFAULT_BATCHED.search(text).group(1)) == 2048
    nd = ra.parse_nondefault(text)
    assert nd["long_prefill_token_threshold"] == 512 and nd["scheduler_reserve_full_isl"] is False
    assert nd["enable_logging_iteration_details"] is True
    step, = ad.parse_iteration_log(lines)
    assert step["index"] == 7 and step["ctx_tok"] == 512 and step["t"] == pytest.approx(EPOCH + 0.003)


def test_pump_log_accepts_text_streams():
    out = io.StringIO()
    assert ra.pump_log(io.StringIO("a\nb"), out, wall=lambda: 1.5) == 2
    assert out.getvalue() == "1500 a\n1500 b\n"


def test_start_server_pipes_output_through_the_stamping_reader(tmp_path):
    """A real local subprocess (no GPU): its own session, output stamped by the reader thread."""
    log = tmp_path / "server.log"
    code = "import os, sys; print('hello'); print('pgid', os.getpgid(0) == os.getpid()); sys.stdout.flush()"
    proc = ra.start_server([sys.executable, "-c", code], dict(os.environ), log, wall=lambda: EPOCH)
    assert proc.wait(timeout=30) == 0
    assert proc.log_reader.join(10) is True
    assert log.read_text().splitlines() == [f"{int(EPOCH * 1000)} hello", f"{int(EPOCH * 1000)} pgid True"]


class FullDiskLog:
    """A log file whose writes fail from the n-th on (ENOSPC)."""

    def __init__(self, ok_writes=5):
        self.ok_writes, self.writes, self.closed = ok_writes, 0, False

    def write(self, text):
        self.writes += 1
        if self.writes > self.ok_writes:
            raise OSError(28, "No space left on device")

    def flush(self):
        pass

    def close(self):
        self.closed = True


def test_log_reader_records_a_write_error_and_keeps_draining_the_pipe():
    """A real local child writing ~1 MB: after the log fails the reader still empties the pipe, so the child (the
    server) never blocks on it; the error is kept for the runner instead of dying with the thread."""
    code = "import sys\nfor i in range(20000): print('x' * 50, i)\nsys.stdout.flush()"
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = FullDiskLog()
    try:
        reader = ra.LogReader(proc.stdout, log, wall=lambda: EPOCH)
        assert proc.wait(timeout=20) == 0  # not stuck on a full pipe
        assert reader.join(10) is True
        assert reader.error is not None and "No space left on device" in reader.error
        assert log.closed is True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    clean = ra.LogReader(io.BytesIO(b"a\nb\n"), io.StringIO(), wall=lambda: EPOCH)
    assert clean.join(10) is True and clean.error is None and clean.lines == 2


def test_start_server_writes_the_log_as_utf8(tmp_path, monkeypatch):
    """The stamped log is UTF-8 whatever the locale (the reader decodes with replacement, so it always encodes)."""
    seen = []

    def spy(path, mode="r", *a, **kw):
        seen.append((Path(path).name, mode, kw.get("encoding")))
        return open(path, mode, *a, **kw)

    monkeypatch.setattr(ra, "open", spy, raising=False)
    proc = ra.start_server([sys.executable, "-c", "print('caf\\u00e9')"], dict(os.environ), tmp_path / "s.log",
                           wall=lambda: EPOCH)
    assert proc.wait(timeout=30) == 0 and proc.log_reader.join(10) is True
    assert seen == [("s.log", "w", "utf-8")]
    assert (tmp_path / "s.log").read_bytes().decode("utf-8").endswith("café\n")


# ------------------------------------------------------------------ manipulation check --

@pytest.mark.parametrize("cond", ap.CONDITIONS)
def test_manipulation_check_passes_on_each_conditions_own_log(pins_k3, cond):
    argv, _ = ap.server_argv_for(cond, pins_k3, gdn_prefill_backend="triton", gpu_memory_utilization="0.91", port=8000)
    facts, nd, problems = ra.manipulation_problems("\n".join(server_log(argv)), cond, pins_k3)
    assert problems == [] and facts["attention_backend"] == "FLASH_ATTN"
    assert nd == {k: v for k, v in nondefault_from_argv(argv).items() if k in ra.WATCHED_KEYS}
    assert ra.expected_nondefault(cond, pins_k3) == nd


def edited(cond, pins, change):
    argv, _ = ap.server_argv_for(cond, pins, gdn_prefill_backend="triton", gpu_memory_utilization="0.91", port=8000)
    nd = nondefault_from_argv(argv)
    change(nd)
    return ra.manipulation_problems("\n".join(server_log(nondefault=nd)), cond, pins)[2]


@pytest.mark.parametrize("cond, change, match", [
    ("B0", lambda d: d.update(long_prefill_token_threshold=512), "long_prefill_token_threshold"),
    ("B0", lambda d: d.update(scheduler_reserve_full_isl=False), "scheduler_reserve_full_isl"),
    ("T", lambda d: d.pop("long_prefill_token_threshold"), "long_prefill_token_threshold"),
    ("T", lambda d: d.update(long_prefill_token_threshold=1024), "long_prefill_token_threshold"),
    ("G", lambda d: d.update(scheduler_reserve_full_isl=True), "scheduler_reserve_full_isl"),
    ("TG", lambda d: d.pop("scheduler_reserve_full_isl"), "scheduler_reserve_full_isl"),
    ("K3", lambda d: d.update(kv_cache_memory_bytes=4294967296), "kv_cache_memory_bytes"),
    ("B0", lambda d: d.update(kv_cache_memory_bytes=3221225472), "kv_cache_memory_bytes"),
    ("G", lambda d: d.pop("enable_logging_iteration_details"), "enable_logging_iteration_details"),
    ("TG", lambda d: d.update(otlp_traces_endpoint="http://127.0.0.1:4317"), "otlp_traces_endpoint"),
    ("B0", lambda d: d.update(max_num_batched_tokens=8192), "max_num_batched_tokens"),
    ("T", lambda d: d.update(long_prefill_token_threshold=True), "long_prefill_token_threshold"),
    ("G", lambda d: d.update(scheduler_reserve_full_isl=0), "scheduler_reserve_full_isl"),  # 0 == False, not a bool
    ("B0", lambda d: d.update(enable_logging_iteration_details=1), "enable_logging_iteration_details"),
])
def test_manipulation_check_detects_extra_missing_or_wrong_keys(pins_k3, cond, change, match):
    problems = edited(cond, pins_k3, change)
    assert problems and any(match in p for p in problems), problems


def test_manipulation_check_needs_the_flash_attn_line_and_the_args_line(pins_k3):
    argv, _ = ap.server_argv_for("B0", pins_k3, gdn_prefill_backend="triton", gpu_memory_utilization="0.91",
                                 port=8000)
    lines = server_log(argv)
    _, _, problems = ra.manipulation_problems("\n".join([lines[0], lines[2]]), "B0", pins_k3)
    assert any("FLASH_ATTN" in p for p in problems)
    _, nd, problems = ra.manipulation_problems("\n".join(lines[1:]), "B0", pins_k3)
    assert nd is None and any("non-default args" in p for p in problems)
    triton = "\n".join(lines).replace("FLASH_ATTN backend.", "TRITON_ATTN backend.")
    assert any("FLASH_ATTN" in p for p in ra.manipulation_problems(triton, "B0", pins_k3)[2])


def test_blocks_check_against_the_frozen_count():
    m = 'vllm:cache_config_info{block_size="528",engine="0",num_gpu_blocks="%d"} 1.0\n'
    assert ra.blocks_problems(m % 248, expected_blocks=248, block_size=528) == (
        {"num_gpu_blocks": 248, "block_size": 528}, [])
    cfg, problems = ra.blocks_problems(m % 186, expected_blocks=248, block_size=528)
    assert cfg["num_gpu_blocks"] == 186 and "num_gpu_blocks" in problems[0]
    assert ra.blocks_problems((m % 248).replace("528", "512"), expected_blocks=248, block_size=528)[1]
    assert ra.blocks_problems("", expected_blocks=248, block_size=528) == (None, ["/metrics has no "
                                                                                "vllm:cache_config_info"])
    _, problems = ra.blocks_problems(m % 186, expected_blocks=0, block_size=528)
    assert problems and "not measured" in problems[0]  # K3 before smoke FS1 froze its block count


def test_chunk_check_for_t_and_tg():
    ok = [{"index": 1, "ctx_req": 2, "ctx_tok": 1024}, {"index": 2, "ctx_req": 1, "ctx_tok": 512},
          {"index": 3, "ctx_req": 0, "ctx_tok": 0}]
    assert ra.chunk_problems(ok, 512) == []
    bad = ok + [{"index": 4, "ctx_req": 1, "ctx_tok": 2048}]
    assert "1 step" in ra.chunk_problems(bad, 512)[0] and "index 4" in ra.chunk_problems(bad, 512)[0]
    assert ra.chunk_problems([], 512) == ["no iteration line in the bench window to check the chunk size"]


# ------------------------------------------------------------------ row fields --

def bench_result(L=2048, C=25, mixed=True):
    N = bp.num_prompts(C)
    lens = [bp.LONG_INPUT if mixed and i % 10 == 4 else L - 128 for i in range(N)]
    ttfts = [(1.0 + 0.01 * i) if n == bp.LONG_INPUT else 0.1 + 0.001 * i for i, n in enumerate(lens)]
    return {"completed": N, "failed": 0, "num_prompts": N, "duration": 10.0, "request_throughput": 25.0,
            "output_throughput": 3200.0, "total_token_throughput": 9e4, "input_lens": lens,
            "output_lens": [128] * N, "ttfts": ttfts, "latencies": [x + 1.27 for x in ttfts],
            **{f"{s}_{m}_ms": 2.0 for s in ("mean", "median", "p50", "p95", "p99")
               for m in ("ttft", "tpot", "itl", "e2el", "client_queue_time")}}


def test_bench_fields_add_the_long_class_and_client_descriptives_to_core_summary():
    res = bench_result()
    f = ra.bench_fields(res, L=2048, C=25, mix="mixed")
    core = rb.summarize_result(res, L=2048, C=25, mix="mixed")
    assert {k: f[k] for k in core} == core and f["ok"] is True
    longs = [t * 1000 for t, n in zip(res["ttfts"], res["input_lens"]) if n == bp.LONG_INPUT]
    shorts = [t * 1000 for t, n in zip(res["ttfts"], res["input_lens"]) if n == 1920]
    assert len(longs) == 25 and f["long_class_p95_ttft_ms"] == pytest.approx(ba.percentile(longs, 95))
    cl = f["client"]
    assert cl["short_p50_ttft_ms"] == pytest.approx(ba.percentile(shorts, 50))
    assert cl["short_p99_ttft_ms"] == pytest.approx(ba.percentile(shorts, 99))
    assert cl["long_p50_ttft_ms"] == pytest.approx(ba.percentile(longs, 50))
    assert cl["short_p95_tpot_ms"] == pytest.approx(1270.0 / 127)  # (latency - ttft) / (output - 1)
    uni = ra.bench_fields(bench_result(mixed=False), L=2048, C=25, mix="uniform")
    assert uni["long_class_p95_ttft_ms"] is None and uni["client"]["long_p50_ttft_ms"] is None
    assert uni["short_class_p95_ttft_ms"] == pytest.approx(ba.short_class_p95_ttft_ms(bench_result(mixed=False),
                                                                                     L=2048))


def test_scrape_deltas_and_series_summary():
    w = World(None)
    w.cond = "B0"
    before = w.metrics_text()
    w.counters.update(q_count=10.0, q_sum=0.5, p_count=10.0, p_sum=1.0, prompt=19200.0, gen=1280.0, pre=2.0)
    after = w.metrics_text()
    d = ra.scrape_deltas(before, after)
    assert d["vllm:request_queue_time_seconds"]["count"] == 10.0 and d["vllm:request_queue_time_seconds"]["sum"] == 0.5
    assert d["vllm:request_queue_time_seconds"]["mean"] == pytest.approx(0.05)
    assert d["vllm:request_prefill_time_seconds"]["mean"] == pytest.approx(0.1)
    assert d["vllm:prompt_tokens_total"] == 19200.0 and d["vllm:num_preemptions_total"] == 2.0
    assert d["vllm:time_to_first_token_seconds"] is None  # absent from this scrape: recorded as missing
    assert "reset" in ra.scrape_deltas(after, before)["errors"][0]
    samples = [{"t": 1.0, "vllm:kv_cache_usage_perc": 0.5, "vllm:num_requests_waiting": 0.0,
                "vllm:num_requests_running": 3.0, "vllm:num_preemptions_total": 0.0},
               {"t": 2.0, "vllm:kv_cache_usage_perc": 0.996, "vllm:num_requests_waiting": 4.0,
                "vllm:num_requests_running": 25.0, "vllm:num_preemptions_total": 1.0}]
    s = ra.series_summary(samples)
    assert s == {"kv_peak": 0.996, "waiting_peak": 4.0, "running_peak": 25.0, "n_samples": 2}
    assert ra.series_summary([])["kv_peak"] is None


def test_time_series_sampler_records_wall_time_and_the_four_series():
    w = World(None)
    w.cond = "B0"
    clock = iter([10.0, 11.0, 12.0])
    s = ra.TimeSeriesSampler(w.http_get, "http://x/metrics", None, wall=lambda: next(clock))
    s.start()
    samples = s.stop()
    assert [x["t"] for x in samples] == [10.0, 11.0]
    assert samples[0] == {"t": 10.0, "vllm:kv_cache_usage_perc": 0.5, "vllm:num_requests_waiting": 1.0,
                          "vllm:num_requests_running": 3.0, "vllm:num_preemptions_total": 0.0}
    dead = ra.TimeSeriesSampler(lambda url: (_ for _ in ()).throw(OSError("down")), "u", None)
    dead.start()
    assert dead.stop() == []


# ------------------------------------------------------------------ OTLP receiver --

PREFIX = "adv-B0-2048-uniform-r1-"


def span(rid, name="llm_request"):
    return {"name": name, "start_time_unix_nano": 1, "attributes": {"gen_ai.request.id": rid}}


def receiver(tmp_path, decode=json_decode, **kw):
    return ra.OtlpReceiver(tmp_path / "otlp", tmp_path / "spans.jsonl", prefix=PREFIX, decode=decode, **kw)


def test_receiver_keeps_only_llm_request_spans_and_counts_main_ids(tmp_path):
    r = receiver(tmp_path)
    body = json.dumps([span(f"cmpl-{PREFIX}0-0"), span(f"cmpl-{PREFIX}1-0"), span("cmpl-0a1b-0"),
                       span(f"cmpl-{PREFIX}2-0", name="load_model"), span(f"cmpl-adv-B0-2048-uniform-r2-3-0")])
    r.handle(body.encode(), None)
    r.handle(json.dumps([span(f"cmpl-{PREFIX}1-0")]).encode(), "identity")  # a duplicate id counts once
    kept = [json.loads(x) for x in (tmp_path / "spans.jsonl").read_text().splitlines()]
    assert [k["attributes"]["gen_ai.request.id"] for k in kept] == [
        f"cmpl-{PREFIX}0-0", f"cmpl-{PREFIX}1-0", "cmpl-0a1b-0", "cmpl-adv-B0-2048-uniform-r2-3-0",
        f"cmpl-{PREFIX}1-0"]
    assert r.main_count() == 2
    assert r.stats() == {"posts": 2, "spans": 6, "llm_request": 5, "main": 2, "decode_errors": [], "errors": []}
    assert sorted(p.name for p in (tmp_path / "otlp").iterdir()) == ["00001.pb", "00002.pb"]
    assert (tmp_path / "otlp" / "00001.pb").read_bytes() == body.encode()


def test_receiver_gunzips_and_keeps_the_raw_compressed_body(tmp_path):
    seen = []

    def decode(body, gz):
        seen.append(gz)
        return json_decode(body, gz)

    r = receiver(tmp_path, decode=decode)
    body = gzip.compress(json.dumps([span(f"cmpl-{PREFIX}0-0")]).encode())
    r.handle(body, "gzip")
    assert seen == [True] and r.main_count() == 1
    assert (tmp_path / "otlp" / "00001.pb.gz").read_bytes() == body


def test_receiver_default_decode_maps_attributes_with_a_fake_protobuf_decoder(tmp_path):
    from types import SimpleNamespace

    class Any_:
        def __init__(self, kind, value):
            self.kind = kind
            setattr(self, kind, value)

        def WhichOneof(self, _):
            return self.kind

    attrs = [SimpleNamespace(key="gen_ai.request.id", value=Any_("string_value", f"cmpl-{PREFIX}4-0")),
             SimpleNamespace(key="gen_ai.latency.time_in_queue", value=Any_("double_value", 0.25))]
    sp = SimpleNamespace(name="llm_request", trace_id=b"\x01", span_id=b"\x02", start_time_unix_nano=2_000_000_000,
                         end_time_unix_nano=3, attributes=attrs)
    msg = SimpleNamespace(resource_spans=[SimpleNamespace(scope_spans=[SimpleNamespace(spans=[sp])])])
    r = receiver(tmp_path, decode=ra.otlp_decode(lambda body: msg))
    r.handle(gzip.compress(b"x"), "gzip")
    rec, = ad.llm_request_records([json.loads(x) for x in (tmp_path / "spans.jsonl").read_text().splitlines()])
    assert rec["id"] == f"cmpl-{PREFIX}4-0" and rec["Q"] == 0.25 and rec["arrival"] == 2.0
    assert r.main_count() == 1


def test_receiver_write_failure_raises_and_is_not_swallowed(tmp_path):
    r = ra.OtlpReceiver(tmp_path / "otlp", tmp_path / "missing" / "spans.jsonl", prefix=PREFIX, decode=json_decode)
    body = json.dumps([span(f"cmpl-{PREFIX}0-0")]).encode()
    with pytest.raises(OSError):
        r.handle(body, None)
    assert r.receive(body, None) == 200  # the exporter always gets 200 ...
    assert r.errors and "FileNotFoundError" in r.errors[0]
    with pytest.raises(ra.ReceiverError, match="FileNotFoundError"):  # ... and the runner gets the error
        r.raise_if_failed()
    assert r.stats()["errors"] == r.errors  # ... and the row's span stats carry it
    ok = receiver(tmp_path / "ok")
    ok.raise_if_failed()


def test_receiver_records_decode_errors_without_failing(tmp_path):
    r = receiver(tmp_path)
    assert r.receive(b"not json", None) == 200
    assert r.receive(b"x", "br") == 200  # an encoding the receiver cannot read: body kept, error recorded
    assert len(r.stats()["decode_errors"]) == 2 and r.errors == []
    r.raise_if_failed()
    assert len(list((tmp_path / "otlp").iterdir())) == 2


def test_receiver_http_roundtrip_answers_200_on_the_traces_path(tmp_path):
    """A real loopback server on an ephemeral port (127.0.0.1 only)."""
    r = receiver(tmp_path, port=0).start()
    try:
        url = f"http://127.0.0.1:{r.port}/v1/traces"
        body = gzip.compress(json.dumps([span(f"cmpl-{PREFIX}0-0")]).encode())
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/x-protobuf",
                                                              "Content-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
        bad = urllib.request.Request(url.replace("/v1/traces", "/v1/logs"), data=b"x")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad, timeout=10)
        assert exc.value.code == 404
        assert r.main_count() == 1 and r.stats()["posts"] == 1
    finally:
        r.close()
    assert r._thread is None or not r._thread.is_alive()
    assert r.close() == []  # idempotent


def test_receiver_timeouts_default_to_10_s(tmp_path):
    r = receiver(tmp_path)
    assert (r.request_timeout_s, r.close_timeout_s) == (10, 10)
    assert ra._handler(r).timeout == 10  # socket timeout of every request (StreamRequestHandler.setup)


def header_only(port):
    """A local client that sends a POST's headers and 3 of its 100 body bytes, then nothing."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall(b"POST /v1/traces HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/x-protobuf\r\n"
                 b"Content-Length: 100\r\n\r\nabc")
    time.sleep(0.3)  # the handler is now blocked reading the body
    return sock


def close_in_thread(r, limit_s):
    """r.close() run with a bound of its own, so a hanging close fails the test instead of hanging it."""
    out = {}
    t = threading.Thread(target=lambda: out.update(problems=r.close()), daemon=True)
    t0 = time.monotonic()
    t.start()
    t.join(limit_s)
    return t, out, time.monotonic() - t0


def test_receiver_request_timeout_frees_close_from_a_header_only_connection(tmp_path):
    """A real loopback receiver: a client that stops mid-body times out (Handler.timeout), so close() returns."""
    r = receiver(tmp_path, port=0, request_timeout_s=0.5).start()
    sock = header_only(r.port)
    try:
        t, out, took = close_in_thread(r, 8)
        assert not t.is_alive() and took < 5
        assert out["problems"] == [] and r.stats()["posts"] == 0 and r.errors == []
    finally:
        sock.close()


def test_receiver_close_is_bounded_and_reports_a_teardown_problem(tmp_path):
    """Even when a request outlives the bound, close() returns, closes the listening socket and says so."""
    r = receiver(tmp_path, port=0, request_timeout_s=30, close_timeout_s=0.5).start()
    port = r.port
    sock = header_only(port)
    try:
        t, out, took = close_in_thread(r, 8)
        assert not t.is_alive() and took < 5
        assert out["problems"] and "did not stop" in out["problems"][0]
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
    finally:
        sock.close()


def test_otlp_protobuf_roundtrip_through_the_receiver(tmp_path):
    pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    req = ExportTraceServiceRequest()
    sp = req.resource_spans.add().scope_spans.add().spans.add()
    sp.name, sp.start_time_unix_nano = "llm_request", 1_000_000_000_000
    a = sp.attributes.add()
    a.key, a.value.string_value = "gen_ai.request.id", f"cmpl-{PREFIX}3-0"
    r = receiver(tmp_path, decode=ra.otlp_decode())
    r.handle(gzip.compress(req.SerializeToString()), "gzip")
    assert r.main_count() == 1


# ------------------------------------------------------------------ bench with timeout --

def test_default_run_bench_returns_the_exit_code(tmp_path):
    log = tmp_path / "b.log"
    rc, timed_out = ra.default_run_bench([sys.executable, "-c", "print('done')"], log, dict(os.environ), 30)
    assert (rc, timed_out) == (0, False) and log.read_text().strip() == "done"


def test_default_run_bench_kills_the_whole_process_group_on_timeout(tmp_path):
    """A real local client that forks a grandchild and hangs: after the timeout the group is gone."""
    pid_file = tmp_path / "child.pid"
    code = ("import subprocess, sys, time; "
            f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            f"open({str(pid_file)!r}, 'w').write(str(p.pid)); time.sleep(60)")
    t0 = time.monotonic()
    rc, timed_out = ra.default_run_bench([sys.executable, "-c", code], tmp_path / "b.log", dict(os.environ), 1.5)
    assert timed_out is True and rc == -signal.SIGKILL and time.monotonic() - t0 < 20
    child = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("grandchild survived the process-group kill")


# ------------------------------------------------------------------ session --

def test_runs_every_boot_in_plan_order_one_bench_per_boot(tmp_path, pins_k3):
    plan = first_block(pins_k3, 6)
    w = World(tmp_path)
    out, summary, rows = run(w, pins_k3, plan)
    assert len(w.started) == len(w.bench_argvs) == len(rows) == 6
    assert [r["prefix"] for r in rows] == [b["prefix"] for b in plan["boots"]]
    assert [a[a.index("--request-id-prefix") + 1] for a in w.bench_argvs] == [b["prefix"] for b in plan["boots"]]
    assert [cond_of(a) for a in w.started] == [b["cond"] for b in plan["boots"]]
    for r, b in zip(rows, plan["boots"]):
        assert CONTRACT_KEYS <= set(r), CONTRACT_KEYS - set(r)
        assert {k: r[k] for k in ("cond", "L", "mix", "restart", "block", "C", "N")} == \
            {k: b[k] for k in ("cond", "L", "mix", "restart", "block", "C", "N")}
        assert r["ok"] is True and r["timeout"] is False and r["skipped"] is False, r["reasons"]
        assert aa.row_valid(r), aa.invalid_reasons(r)
        assert r["manip"] == {"facts_ok": True, "blocks_ok": True,
                              "chunk_ok": True if b["cond"] in ("T", "TG") else None,
                              "num_gpu_blocks": 186 if b["cond"] == "K3" else 248}
        assert r["decomp"]["valid"] is True, r["decomp"]["reasons"]
        assert r["decomp"]["checks"] == {"a": True, "b": True, "c": True, "d": True}
        assert r["preemptions"] == 0 and r["kv_peak"] == 0.5
        assert r["spans"]["main"] == b["N"] and r["spans"]["waited_s"] == 0
    assert summary["outcomes"] == {"boots": 6, "ok": 6, "valid": 6, "invalid": 0, "timeout": 0, "skipped": 0}
    assert summary["env_valid"] is True and (out / "raw_manifest.json").is_file()


def test_each_boot_follows_the_024_step_order(tmp_path, pins_k3):
    w = World(tmp_path)
    run(w, pins_k3, first_block(pins_k3, 1))
    ev = w.events
    tag = ra.boot_tag(first_block(pins_k3, 1)["boots"][0])
    order = [f"env:{tag}.before.json", "receiver:start", "popen", "health", "metrics", "bench", "metrics",
             f"signal:{int(signal.SIGINT)}", "receiver:close", f"env:{tag}.after.json"]
    pos = [ev.index(e) for e in order[:5]] + [ev.index("bench")]
    pos += [max(i for i, e in enumerate(ev) if e == "metrics"), ev.index(order[7]), ev.index(order[8]),
            ev.index(order[9])]
    assert pos == sorted(pos), ev
    assert ev.count("bench") == 1 and ev.count("popen") == 1


def test_server_argv_env_and_bench_argv_follow_the_condition_table(tmp_path, pins_k3):
    plan = first_block(pins_k3, 20)
    ds = {L: tmp_path / f"mixed-{L}.jsonl" for L in (2048, 8192)}
    for p in ds.values():
        p.write_text("{}\n")
    w = World(tmp_path)
    out, summary, rows = run(w, pins_k3, plan, mixed_datasets={L: str(f) for L, f in ds.items()})
    for b, argv, env, bargv, timeout in zip(plan["boots"], w.started, w.server_envs, w.bench_argvs,
                                            w.bench_timeouts):
        want, want_env = ap.server_argv_for(b["cond"], pins_k3, gdn_prefill_backend="triton",
                                            gpu_memory_utilization="0.91", port=ra.PORT, vllm_bin=".venv/bin/vllm",
                                            base_env=BASE_ENV)
        assert argv == want and env == want_env
        assert env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf" and "OTEL_SDK_DISABLED" not in env
        tag = ra.boot_tag(b)
        assert bargv == ap.bench_argv_for(pins_k3, cond=b["cond"], L=b["L"], C=b["C"], mix=b["mix"],
                                          restart=b["restart"], result_dir=str(out / "raw"),
                                          result_filename=f"{tag}.bench.json", port=ra.PORT,
                                          mixed_dataset=str(ds[b["L"]]) if b["mix"] == "mixed" else None,
                                          vllm_bin=".venv/bin/vllm")
        assert timeout == b["timeout_s"]
        assert summary["runs"][tag]["argv"] == argv
        assert summary["runs"][tag]["env"] == ra.recorded_server_env(want_env)
    assert all(e["HF_HUB_OFFLINE"] == "1" and "OTEL_SDK_DISABLED" in e for e in w.bench_envs)  # client env = Core's
    assert summary["inherited_otel"] == ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_SDK_DISABLED"]
    assert summary["mixed_dataset_sha256"] == {str(L): hashlib.sha256(b"{}\n").hexdigest() for L in ds}


def test_session_writes_the_raw_derived_and_env_files(tmp_path, pins_k3):
    plan = first_block(pins_k3, 1)
    out, summary, rows = run(World(tmp_path), pins_k3, plan)
    tag = ra.boot_tag(plan["boots"][0])
    for name in (f"{tag}.server.log", f"{tag}.bench.json", f"{tag}.bench.log", f"{tag}.metrics.before.txt",
                 f"{tag}.metrics.after.txt", f"{tag}.series.jsonl", f"{tag}.steps.jsonl", f"{tag}.spans.jsonl"):
        assert (out / "raw" / name).is_file(), name
    assert (out / "raw" / f"{tag}.otlp").is_dir()
    manifest = json.loads((out / "raw_manifest.json").read_text())["files"]
    assert f"raw/{tag}.spans.jsonl" in manifest and f"raw/{tag}.otlp/00001.pb" in manifest
    derived = [json.loads(x) for x in (out / "derived" / f"{tag}-requests.jsonl").read_text().splitlines()]
    assert len(derived) == plan["boots"][0]["N"]
    assert all(isinstance(v, (int, float, type(None))) for d in derived for k, v in d.items() if k != "cls")
    assert {"preflight.json", f"{tag}.before.json", f"{tag}.after.json"} <= {p.name for p in (out / "env").iterdir()}
    assert rows[0]["result"] == f"raw/{tag}.bench.json" and rows[0]["derived"] == f"derived/{tag}-requests.jsonl"
    meta = json.loads((out / "session.json").read_text())
    assert meta["plan_sha256"] == ra.sha256_json(plan) and meta["pins_sha256"] == PINS_SHA
    assert meta["server"] == {"gdn_prefill_backend": "triton", "gpu_memory_utilization": "0.91", "port": ra.PORT,
                              "vllm_bin": ".venv/bin/vllm"}
    assert meta["result_dir"] == str(out / "raw") and meta["repo_commit"] == "c0ffee"


def stop_case(tmp_path, pins_k3, **world_kw):
    plan = first_block(pins_k3, 1, conds=("T",))
    w = World(tmp_path, **world_kw)
    out, summary, rows = run(w, pins_k3, plan)
    return w, summary, rows


def test_wrong_log_facts_stop_the_boot_before_the_bench(tmp_path, pins_k3):
    w, summary, (row,) = stop_case(tmp_path, pins_k3,
                                   log_edit=lambda c, t: t.replace("FLASH_ATTN backend.", "TRITON_ATTN backend."))
    assert w.bench_argvs == [] and w.signals[0] == signal.SIGINT and w.receivers[0].closed
    assert row["ok"] is False and row["manip"]["facts_ok"] is False and row["manip"]["blocks_ok"] is True
    assert row["manip"]["chunk_ok"] is None and row["decomp"] is None and row["short_class_p95_ttft_ms"] is None
    assert any("FLASH_ATTN" in x for x in row["reasons"]) and not aa.row_valid(row)
    tag = ra.boot_tag(row)
    assert summary["runs"][tag]["log_facts"]["attention_backend"] == "TRITON_ATTN"


def test_an_extra_condition_key_in_the_log_stops_the_boot(tmp_path, pins_k3):
    edit = lambda c, t: t.replace("'enable_logging_iteration_details': True",  # noqa: E731
                                  "'enable_logging_iteration_details': True, 'scheduler_reserve_full_isl': False")
    w, _, (row,) = stop_case(tmp_path, pins_k3, log_edit=edit)
    assert w.bench_argvs == [] and row["manip"]["facts_ok"] is False
    assert any("scheduler_reserve_full_isl" in x for x in row["reasons"])


def test_a_missing_condition_key_in_the_log_stops_the_boot(tmp_path, pins_k3):
    edit = lambda c, t: t.replace(", 'long_prefill_token_threshold': 512", "")  # noqa: E731
    w, _, (row,) = stop_case(tmp_path, pins_k3, log_edit=edit)
    assert w.bench_argvs == [] and any("long_prefill_token_threshold" in x for x in row["reasons"])


def test_a_block_count_mismatch_stops_the_boot(tmp_path, pins_k3):
    w, summary, (row,) = stop_case(tmp_path, pins_k3, blocks={"T": 240})
    assert w.bench_argvs == [] and row["manip"]["blocks_ok"] is False and row["manip"]["facts_ok"] is True
    assert row["manip"]["num_gpu_blocks"] == 240 and any("num_gpu_blocks" in x for x in row["reasons"])


def test_a_chunk_violation_marks_t_rows_invalid_but_keeps_the_measurement(tmp_path, pins_k3):
    plan = first_block(pins_k3, 1, conds=("T",))
    w = World(tmp_path, behave={key(plan["boots"][0]): "big_chunks"})
    _, _, (row,) = run(w, pins_k3, plan)
    assert row["ok"] is True and row["short_class_p95_ttft_ms"] is not None
    assert row["manip"]["chunk_ok"] is False and not aa.row_valid(row)
    assert any("ctx_tokens" in x for x in row["reasons"])


def test_timeout_kills_marks_the_row_and_the_session_continues(tmp_path, pins_k3):
    plan = sub_plan(pins_k3, lambda b: (b["cond"], b["L"], b["mix"]) in {("G", 2048, "mixed"), ("B0", 2048, "uniform")})
    g1 = next(b for b in plan["boots"] if b["cond"] == "G" and b["restart"] == 1)
    w = World(tmp_path, behave={key(g1): "timeout"})
    out, summary, rows = run(w, pins_k3, plan)
    t = next(r for r in rows if key(r) == key(g1))
    assert t["timeout"] is True and t["ok"] is False and any("timeout" in x for x in t["reasons"])
    assert t["decomp"] is None and t["short_class_p95_ttft_ms"] is None and not aa.row_valid(t)
    assert w.signals.count(signal.SIGINT) == len(w.started)  # every server torn down, including the timed-out one
    b0 = [r for r in rows if r["cond"] == "B0"]
    assert len(b0) == 3 and all(aa.row_valid(r) for r in b0)  # the session went on
    assert summary["outcomes"]["timeout"] == 1 and summary["outcomes"]["skipped"] == 2


def test_skip_rule_records_the_remaining_restarts_of_an_invalid_cell_without_running_them(tmp_path, pins_k3):
    cell = ("TG", 8192, "mixed")
    plan = sub_plan(pins_k3, lambda b: (b["cond"], b["L"], b["mix"]) in {cell, ("K3", 8192, "uniform")})
    r1 = next(b for b in plan["boots"] if (b["cond"], b["L"], b["mix"]) == cell and b["restart"] == 1)
    w = World(tmp_path, behave={key(r1): "fail"})
    out, summary, rows = run(w, pins_k3, plan)
    tg = sorted((r for r in rows if (r["cond"], r["L"], r["mix"]) == cell), key=lambda r: r["restart"])
    assert [r["skipped"] for r in tg] == [False, True, True]
    assert tg[0]["ok"] is False and any("failed=5" in x for x in tg[0]["reasons"])
    assert all(r["ok"] is False and "r1" in r["reasons"][0] and r["decomp"] is None for r in tg[1:])
    started = [cond_of(a) for a in w.started]
    assert started.count("TG") == 1 and started.count("K3") == 3
    assert len(rows) == len(plan["boots"]) == 6
    assert [key(r) for r in rows] == [key(b) for b in plan["boots"]]  # skipped rows keep their plan position


def test_missing_spans_only_invalidate_the_decomposition(tmp_path, pins_k3):
    plan = first_block(pins_k3, 1, conds=("B0",))
    w = World(tmp_path, behave={key(plan["boots"][0]): "short_spans"})
    _, _, (row,) = run(w, pins_k3, plan)
    assert row["spans"]["main"] == row["N"] - 3 and row["spans"]["waited_s"] >= ra.SPAN_WAIT_S
    assert row["decomp"]["valid"] is False and row["decomp"]["checks"]["a"] is False
    assert row["ok"] is True and row["short_class_p95_ttft_ms"] is not None and aa.row_valid(row)


def test_a_bench_exit_marks_the_row(tmp_path, pins_k3):
    plan = first_block(pins_k3, 1, conds=("K3",))
    w = World(tmp_path, behave={key(plan["boots"][0]): "exit"})
    _, _, (row,) = run(w, pins_k3, plan)
    assert row["ok"] is False and row["timeout"] is False and "vllm bench serve exit 1" in row["reasons"]
    assert row["decomp"] is None and not aa.row_valid(row)


@pytest.mark.parametrize("how, timeout, reason", [("exit_with_result", False, "vllm bench serve exit 1"),
                                                  ("timeout_with_result", True, "timeout")])
def test_a_result_file_does_not_rescue_a_failed_or_timed_out_bench(tmp_path, pins_k3, how, timeout, reason):
    """As Core: only an exit code 0 within the timeout makes a result (a file left behind is not used)."""
    plan = first_block(pins_k3, 1, conds=("B0",))
    w = World(tmp_path, behave={key(plan["boots"][0]): how})
    _, _, (row,) = run(w, pins_k3, plan)
    assert row["ok"] is False and row["timeout"] is timeout and any(reason in x for x in row["reasons"])
    assert row["short_class_p95_ttft_ms"] is None and row["decomp"] is None and not aa.row_valid(row)


def test_recorded_server_env_keeps_core_fields_otel_and_engine_variable_names(pins_k3):
    _, env = ap.server_argv_for("TG", pins_k3, gdn_prefill_backend="triton", gpu_memory_utilization="0.91",
                                port=8000, base_env=BASE_ENV)
    rec = ra.recorded_server_env(env)
    assert rec == {**bp.recorded_env(env), "OTEL": {"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"},
                   "engine_vars": ["CUDA_HOME", "VLLM_GDN_DECODE_KERNEL", "VLLM_WSL2_ENABLE_PIN_MEMORY"]}
    leaky = {**env, "OTEL_SDK_DISABLED": "true", "VLLM_LOGGING_LEVEL": "DEBUG"}
    assert ra.recorded_server_env(leaky)["OTEL"]["OTEL_SDK_DISABLED"] == "true"
    assert "VLLM_LOGGING_LEVEL" in ra.recorded_server_env(leaky)["engine_vars"]


def test_preemptions_come_from_the_scrape_delta(tmp_path, pins_k3):
    plan = first_block(pins_k3, 1, conds=("G",))
    w = World(tmp_path, preempt={key(plan["boots"][0]): 3})
    _, _, (row,) = run(w, pins_k3, plan)
    assert row["preemptions"] == 3 and row["scrape_delta"]["vllm:num_preemptions_total"] == 3.0
    assert row["decomp"]["checks"]["c"] is None  # A10 (c) exempt with preemptions


def test_any_exception_tears_down_closes_the_receiver_and_records_the_abort(tmp_path, pins_k3):
    plan = first_block(pins_k3, 2)
    w = World(tmp_path, metrics_error=1)
    with pytest.raises(RuntimeError, match="boom"):
        run(w, pins_k3, plan)
    assert w.signals and w.signals[0] == signal.SIGINT and w.receivers[0].closed
    meta = json.loads((tmp_path / "session" / "session.json").read_text())
    assert "boom" in meta["aborted"] and meta["outcomes"]["boots"] == 0 and len(w.started) == 1


def test_a_receiver_write_failure_aborts_the_session_after_teardown(tmp_path, pins_k3):
    plan = first_block(pins_k3, 2)
    w = World(tmp_path, behave={key(plan["boots"][0]): "receiver_write_fails"})
    with pytest.raises(ra.ReceiverError):
        run(w, pins_k3, plan)
    assert w.signals and w.receivers[0].closed and len(w.started) == 1
    assert "ReceiverError" in json.loads((tmp_path / "session" / "session.json").read_text())["aborted"]


def session_meta(tmp_path):
    return json.loads((tmp_path / "session" / "session.json").read_text())


def rows_written(tmp_path):
    return [json.loads(x) for x in (tmp_path / "session" / "rows.jsonl").read_text().splitlines()]


def test_a_receiver_write_failure_on_a_stopped_boot_aborts_the_session(tmp_path, pins_k3):
    """The manipulation check stops the boot (no bench), but a span body could not be stored meanwhile."""
    plan = first_block(pins_k3, 2)
    w = World(tmp_path, blocks={plan["boots"][0]["cond"]: 240}, receiver_fail="start")
    with pytest.raises(ra.ReceiverError, match="FileNotFoundError"):
        run(w, pins_k3, plan)
    assert w.bench_argvs == [] and w.signals[0] == signal.SIGINT and w.receivers[0].closed
    assert len(w.started) == 1 and rows_written(tmp_path) == []
    meta = session_meta(tmp_path)
    assert "ReceiverError" in meta["aborted"]
    assert "FileNotFoundError" in meta["runs"][ra.boot_tag(plan["boots"][0])]["receiver_errors"][0]


def test_a_receiver_write_failure_during_the_teardown_flush_aborts_the_session(tmp_path, pins_k3):
    """The server's last span batch (sent on SIGINT) cannot be stored: recorded, then raised after teardown."""
    plan = first_block(pins_k3, 2)
    w = World(tmp_path, receiver_fail="teardown")
    with pytest.raises(ra.ReceiverError, match="FileNotFoundError"):
        run(w, pins_k3, plan)
    assert w.receivers[0].closed and len(w.started) == 1 and rows_written(tmp_path) == []
    meta = session_meta(tmp_path)
    assert "ReceiverError" in meta["aborted"]
    assert "FileNotFoundError" in meta["runs"][ra.boot_tag(plan["boots"][0])]["receiver_errors"][0]


@pytest.mark.parametrize("log_error, behave", [("start", None), (None, "log_write_fails"), ("teardown", None)])
def test_a_dead_server_log_reader_aborts_the_session_instead_of_writing_a_row(tmp_path, pins_k3, log_error, behave):
    """A log that cannot be written is a harness failure (checked after start, after the bench and at teardown):
    the session stops with RunError; no timeout row lets the skip rule turn it into an NT cell."""
    plan = first_block(pins_k3, 2)
    w = World(tmp_path, log_error=log_error, behave={key(plan["boots"][0]): behave} if behave else None)
    with pytest.raises(ra.RunError, match="server log") as exc:
        run(w, pins_k3, plan)
    assert "No space left on device" in str(exc.value)
    assert len(w.bench_argvs) == (0 if log_error == "start" else 1) and len(w.started) == 1
    assert w.signals[0] == signal.SIGINT and w.receivers[0].closed
    assert rows_written(tmp_path) == []
    assert session_meta(tmp_path)["aborted"].startswith("RunError")


@pytest.mark.parametrize("mode, raised", [("signal", KeyboardInterrupt), ("raise", KeyboardInterrupt),
                                          ("sigterm", SystemExit)])
def test_a_signal_during_teardown_does_not_cut_it_short(tmp_path, pins_k3, mode, raised):
    """The server ignores SIGINT / SIGTERM; while teardown waits, another Ctrl-C (or main's SIGTERM -> SystemExit)
    arrives. The escalation still reaches SIGKILL, the log is joined and the receiver closed; only then the
    interrupt goes on (the previous handlers are back)."""
    plan = first_block(pins_k3, 2)
    w = World(tmp_path, stubborn=mode)
    prev_term = signal.signal(signal.SIGTERM, smoke._exit_on_signal)  # as run_adv.main installs it
    try:
        with pytest.raises(raised) as exc:
            run(w, pins_k3, plan)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert signal.getsignal(signal.SIGTERM) is smoke._exit_on_signal
    finally:
        signal.signal(signal.SIGTERM, prev_term)
    if raised is SystemExit:
        assert exc.value.code == 128 + signal.SIGTERM
    assert w.interrupted and w.signals[-1] == signal.SIGKILL and signal.SIGTERM in w.signals
    assert w.receivers[0].closed and len(w.started) == 1
    meta = session_meta(tmp_path)
    assert meta["aborted"].startswith(raised.__name__)
    assert "server had to be killed" in meta["teardown_problems"][0]["reasons"]


def test_the_held_lock_is_verified_at_session_start_not_trusted(tmp_path, pins_k3):
    """--gpu-lock-held without flock: the runner refuses before anything starts."""
    w = World(tmp_path)
    with pytest.raises(ra.RunError, match="does not hold"):
        run(w, pins_k3, first_block(pins_k3, 1), lock_held_check=lambda: False)
    assert w.started == [] and not (tmp_path / "session").exists()


def test_the_held_lock_is_verified_again_before_every_boot(tmp_path, pins_k3):
    answers = iter([True, True, False])  # session start, before boot 1, before boot 2
    w = World(tmp_path)
    with pytest.raises(ra.RunError, match="does not hold"):
        run(w, pins_k3, first_block(pins_k3, 2), lock_held_check=lambda: next(answers))
    assert len(w.started) == 1 and len(rows_written(tmp_path)) == 1
    assert session_meta(tmp_path)["aborted"].startswith("RunError")


def test_the_lock_check_is_not_consulted_without_gpu_lock_held(tmp_path, pins_k3):
    def never():
        raise AssertionError("lock_held_check called without --gpu-lock-held")

    _, _, rows = run(World(tmp_path), pins_k3, first_block(pins_k3, 1), gpu_lock_held=False, lock_held_check=never)
    assert len(rows) == 1


def test_run_session_checks_the_lock_with_the_real_probe_by_default(tmp_path, pins_k3, monkeypatch):
    calls = []
    monkeypatch.setattr(ra, "held_gpu_lock", lambda: calls.append(1) or False)
    w = World(tmp_path)
    with pytest.raises(ra.RunError, match="does not hold"):
        ra.run_session(first_block(pins_k3, 1), pins=pins_k3, pins_sha256=PINS_SHA, out_dir=tmp_path / "s",
                       gdn_prefill_backend="triton", gpu_memory_utilization="0.91", session_id="s",
                       popen=w.popen, signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get,
                       run_bench=w.run_bench, gpu_mem_used=w.gpu_mem_used, gpu_lock_free=lambda: True,
                       gpu_lock_held=True, receiver_factory=w.receiver_factory, toolchain_check=lambda env: [],
                       base_env=BASE_ENV, now=w.now, sleep=w.sleep, wall=w.wall, sample_interval=None,
                       mixed_datasets=datasets(tmp_path))
    assert calls and w.started == []


def test_held_gpu_lock_needs_a_locked_descriptor_on_the_lock_file(tmp_path):
    """The real probe on a stand-in fd directory (symlinks named by fd, as /proc/self/fd): flock(1) passes its
    locked descriptor to the command, and flock(LOCK_EX | LOCK_NB) on it succeeds; one held through another
    open file description fails with EWOULDBLOCK."""
    import fcntl

    lock = tmp_path / "gpu.lock"
    lock.touch()
    fds = tmp_path / "fd"
    fds.mkdir()
    assert ra.held_gpu_lock(str(lock), fd_dir=str(tmp_path / "no-proc")) is False  # macOS: no /proc
    assert ra.held_gpu_lock(str(lock), fd_dir=str(fds)) is False  # no descriptor on the lock file
    other, ours = os.open(lock, os.O_RDWR), os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(other, fcntl.LOCK_EX)  # another project holds the lock
        (fds / str(ours)).symlink_to(lock)
        (fds / "not-a-number").symlink_to(lock)
        assert ra.held_gpu_lock(str(lock), fd_dir=str(fds)) is False
        os.close(other)
        other = None
        fcntl.flock(ours, fcntl.LOCK_EX)  # as flock -n does before exec
        assert ra.held_gpu_lock(str(lock), fd_dir=str(fds)) is True
        assert ra.held_gpu_lock(str(tmp_path / "another.lock"), fd_dir=str(fds)) is False
    finally:
        for fd in (other, ours):
            if fd is not None:
                os.close(fd)


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc (WSL / Linux)")
def test_held_gpu_lock_reads_proc_self_fd(tmp_path):
    import fcntl

    lock = tmp_path / "gpu.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert ra.held_gpu_lock(str(lock)) is True
    finally:
        os.close(fd)
    assert ra.held_gpu_lock(str(lock)) is False  # no descriptor left on it


def test_a_failing_memory_probe_during_a_boot_is_recorded_not_fatal(tmp_path, pins_k3):
    """memory.used is descriptive (024): nvidia-smi timing out at ready or failing after the bench costs neither
    the row's Y nor the session."""
    plan = first_block(pins_k3, 2)
    w = World(tmp_path, mem_fail={2: subprocess.TimeoutExpired(["nvidia-smi"], 15), 3: ValueError("[N/A]")})
    out, summary, rows = run(w, pins_k3, plan)
    assert len(rows) == 2 and "aborted" not in summary
    r = rows[0]
    assert r["ok"] is True and aa.row_valid(r) and r["short_class_p95_ttft_ms"] is not None
    assert r["mem_ready_mib"] is None and r["mem_after_bench_mib"] is None
    assert len(r["mem_probe_errors"]) == 2
    assert "TimeoutExpired" in r["mem_probe_errors"][0] and "[N/A]" in r["mem_probe_errors"][1]
    assert rows[1]["mem_ready_mib"] == 400 and "mem_probe_errors" not in rows[1]


def test_the_idle_check_before_a_boot_stays_a_hard_check(tmp_path, pins_k3):
    w = World(tmp_path, mem_fail={1: subprocess.TimeoutExpired(["nvidia-smi"], 15)})
    with pytest.raises(subprocess.TimeoutExpired):
        run(w, pins_k3, first_block(pins_k3, 1))
    assert w.started == []


def test_nvidia_mem_used_bounded_times_out_after_15_s(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0, stdout="1234\n", stderr="")

    monkeypatch.setattr(ra.subprocess, "run", fake_run)
    assert ra.nvidia_mem_used_bounded() == 1234
    assert seen["timeout"] == 15 and seen["check"] is True and seen["cmd"][0] == "nvidia-smi"

    def hung(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(ra.subprocess, "run", hung)
    with pytest.raises(subprocess.TimeoutExpired):
        ra.nvidia_mem_used_bounded()


def test_main_wires_the_bounded_memory_probe(tmp_path, monkeypatch):
    seen = {}

    def fake_session(plan, **kw):
        seen.update(kw)
        return {"outcomes": {}}

    monkeypatch.setattr(ra, "run_session", fake_session)
    monkeypatch.setattr(ra.signal, "signal", lambda *a: None)  # main's SIGHUP / SIGTERM handlers stay out of pytest
    (tmp_path / "plan.json").write_text("{}")
    assert ra.main(["--plan", str(tmp_path / "plan.json"), "--out", str(tmp_path / "s"), "--gdn-prefill-backend",
                    "triton", "--gpu-memory-utilization", "0.91"]) == 0
    assert seen["gpu_mem_used"] is ra.nvidia_mem_used_bounded


def test_existing_session_is_never_overwritten_or_resumed(tmp_path, pins_k3):
    plan = first_block(pins_k3, 1)
    run(World(tmp_path), pins_k3, plan)
    w = World(tmp_path)
    with pytest.raises(ra.RunError, match="exists"):
        run(w, pins_k3, plan)
    assert w.started == []


def test_gpu_busy_or_lock_held_stops_before_a_boot(tmp_path, pins_k3):
    plan = first_block(pins_k3, 1)
    w = World(tmp_path, mem=5000)
    with pytest.raises(ra.RunError, match="GPU busy"):
        run(w, pins_k3, plan)
    assert w.started == [] and w.t >= ra.IDLE_WAIT_S
    w2 = World(tmp_path / "b")
    (tmp_path / "b").mkdir()
    with pytest.raises(ra.RunError, match="lock"):
        ra.run_session(plan, pins=pins_k3, pins_sha256=PINS_SHA, out_dir=tmp_path / "b" / "s",
                       gdn_prefill_backend="triton", gpu_memory_utilization="0.91", session_id="s",
                       popen=w2.popen, signal_group=w2.signal_group, group_alive=w2.group_alive,
                       http_get=w2.http_get, run_bench=w2.run_bench, gpu_mem_used=w2.gpu_mem_used,
                       gpu_lock_free=lambda: False, receiver_factory=w2.receiver_factory,
                       toolchain_check=lambda env: [], base_env=BASE_ENV, now=w2.now, sleep=w2.sleep,
                       wall=w2.wall, sample_interval=None, mixed_datasets=datasets(tmp_path / "b"))
    assert w2.started == []


def test_env_capture_before_and_after_each_boot_and_power_flags(tmp_path, pins_k3):
    plan = first_block(pins_k3, 2)
    env = FakeEnv(enforced=250.0)
    _, summary, _ = run(World(tmp_path), pins_k3, plan, capture_env=env)
    tags = [ra.boot_tag(b) for b in plan["boots"]]
    assert env.calls == ["preflight.json"] + [f"{t}.{x}.json" for t in tags for x in ("before", "after")]
    assert summary["env_valid"] is False and "power limit" in summary["env_flags"][0]["reason"]


@pytest.mark.parametrize("change, match", [
    (lambda p: p["boots"][0].update(prefix="adv-X-"), "prefix"),
    (lambda p: p["boots"][0].update(N=10), "N"),
    (lambda p: p["boots"][0].update(timeout_s=1), "timeout"),
    (lambda p: p["boots"][0].update(C=22), "C"),
    (lambda p: p["boots"].append(dict(p["boots"][0])), "duplicate"),
    (lambda p: p["boots"][0].update(cond="X"), "condition"),
    (lambda p: p["expected_blocks"].pop("G"), "expected_blocks"),
    (lambda p: p.update(boots=[]), "no boots"),
])
def test_a_malformed_plan_is_refused_before_anything_starts(tmp_path, pins_k3, change, match):
    plan = first_block(pins_k3, 20)
    change(plan)
    w = World(tmp_path)
    with pytest.raises(ra.RunError, match=match):
        run(w, pins_k3, plan)
    assert w.started == [] and not (tmp_path / "session").exists()


def test_run_core_b_and_opentelemetry_are_imported_lazily():
    """run_core_b imports tomllib (3.11+) and opentelemetry exists only in the WSL venv: neither is a
    module-level import (stdlib only, Python 3.10 compatible)."""
    import ast

    src = Path(ra.__file__).read_text()
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not [ln for ln in top if "opentelemetry" in ln or "run_core_b" in ln or "tomllib" in ln]
    ast.parse(src, feature_version=(3, 10))


def test_the_default_receiver_needs_the_protobuf_decoder_before_any_boot(tmp_path, pins_k3, monkeypatch):
    def missing():
        raise ImportError("No module named 'opentelemetry'")

    monkeypatch.setattr(ad, "_otlp_decoder", missing)
    w = World(tmp_path)
    with pytest.raises(ra.RunError, match="opentelemetry"):
        ra.run_session(first_block(pins_k3, 1), pins=pins_k3, pins_sha256=PINS_SHA, out_dir=tmp_path / "s",
                       gdn_prefill_backend="triton", gpu_memory_utilization="0.91", session_id="s",
                       popen=w.popen, signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get,
                       run_bench=w.run_bench, gpu_mem_used=w.gpu_mem_used, gpu_lock_free=lambda: True,
                       gpu_lock_held=True, lock_held_check=lambda: True, toolchain_check=lambda env: [],
                       base_env=BASE_ENV, now=w.now, sleep=w.sleep, wall=w.wall, sample_interval=None,
                       mixed_datasets=datasets(tmp_path))
    assert w.started == [] and not (tmp_path / "s").exists()


def test_default_receiver_endpoint_comes_from_pins(tmp_path, pins_k3):
    make = ra.receiver_factory_for(pins_k3, decode=json_decode)
    r = make(tmp_path / "o", tmp_path / "s.jsonl", PREFIX)
    assert (r.host, r.port, r.path) == ("127.0.0.1", 4318, "/v1/traces")


# ------------------------------------------------------------------ formal --

def formal_setup(tmp_path, formal_pins):
    p = copy.deepcopy(formal_pins)
    ds = datasets(tmp_path)
    p["core_b"]["mix_dataset_sha256"] = {str(L): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                                         for L, f in ds.items()}
    return p, make_plan(p), ds


@pytest.mark.parametrize("change, kw, match", [
    (None, {"gpu_lock_held": False}, "lock"),
    (lambda p, plan: p["freeze"].update(advanced=False), {}, "freeze.advanced"),
    (None, {"pins_sha256": "cd" * 32}, "pins_sha256"),
    (None, {"gpu_memory_utilization": "0.85"}, "gpu_memory_utilization"),
    (lambda p, plan: plan.update(boots=plan["boots"][:20]), {}, "60-boot"),
    (lambda p, plan: plan["expected_blocks"].update(K3=180), {}, "expected_blocks"),
    (lambda p, plan: p["core_b"]["mix_dataset_sha256"].update({"8192": "f" * 64}), {}, "sha256"),
    (lambda p, plan: p["advanced"].update(long_prefill_token_threshold=1024), {}, "long_prefill_token_threshold"),
])
def test_formal_refusals(tmp_path, formal_pins, change, kw, match):
    p, plan, ds = formal_setup(tmp_path, formal_pins)
    if change:
        change(p, plan)
    w = World(tmp_path)
    with pytest.raises(ra.RunError, match=match):
        run(w, p, plan, formal=True, mixed_datasets=ds, **kw)
    assert w.started == [] and not (tmp_path / "session").exists()


def test_formal_refuses_a_missing_dataset_another_vllm_or_a_dirty_tree(tmp_path, formal_pins):
    p, plan, ds = formal_setup(tmp_path, formal_pins)
    with pytest.raises(ra.RunError, match="--mixed-dataset"):
        run(World(tmp_path), p, plan, formal=True, mixed_datasets={2048: ds[2048]})
    for i, (env, match) in enumerate(((FakeEnv(vllm="0.31.0"), "vLLM"), (FakeEnv(dirty=True), "dirty"),
                                      (FakeEnv(enforced=250.0), "power limit"))):
        w = World(tmp_path / str(i))
        (tmp_path / str(i)).mkdir()
        with pytest.raises(ra.RunError, match=match):
            run(w, p, plan, formal=True, mixed_datasets=ds, capture_env=env)
        assert w.started == []


def test_formal_session_end_to_end_is_accepted_by_adv_report(tmp_path, formal_pins):
    """60 boots in the seeded order under formal rules; the report's A14 checks accept what the runner recorded."""
    import adv_report as ar

    p, plan, ds = formal_setup(tmp_path, formal_pins)
    w = World(tmp_path)
    out, summary, rows = run(w, p, plan, formal=True, mixed_datasets=ds)
    assert summary["formal"] is True and summary["env_valid"] is True and "aborted" not in summary
    assert len(rows) == 60 and all(aa.row_valid(r) for r in rows), [aa.invalid_reasons(r) for r in rows][:3]
    assert all(r["decomp"]["valid"] for r in rows)
    adv = ar.load_session(out)
    session, up, urows = uniform_session(p)
    uniform = {"session": session, "plan": up, "rows": urows}
    assert ar.refusal_problems(adv, uniform, pins=p, pins_sha256=PINS_SHA) == []
    analysis = ar.report(adv, uniform, pins=p, pins_sha256=PINS_SHA)
    assert set(analysis["levels"]) == {"2048", "8192"} and len(analysis["runs"]) == 60


@pytest.mark.parametrize("mode", ["signal", "sigterm"])
def test_real_signals_during_teardown_are_deferred_not_retried(tmp_path, pins_k3, mode):
    """With DeferredSignals the shutdown sequence runs once, uninterrupted; the retry in shutdown_group is only the
    fallback for an interrupt raised directly (mode 'raise'). Removing the handler swap must make this fail."""
    plan = first_block(pins_k3, 2)
    w = World(tmp_path, stubborn=mode)
    prev_term = signal.signal(signal.SIGTERM, smoke._exit_on_signal)
    try:
        with pytest.raises((KeyboardInterrupt, SystemExit)):
            run(w, pins_k3, plan)
    finally:
        signal.signal(signal.SIGTERM, prev_term)
    reasons = session_meta(tmp_path)["teardown_problems"][0]["reasons"]
    assert not any("teardown interrupted" in r for r in reasons), reasons
    assert w.receivers[0].closed


def test_an_error_while_stopping_the_server_still_closes_the_receiver(tmp_path, pins_k3):
    """A non-interrupt failure inside the shutdown (e.g. signalling the group raises) must not skip receiver.close()."""
    w = World(tmp_path)
    real = w.signal_group

    def broken(proc, sig):
        real(proc, sig)
        raise OSError("kill failed")

    w.signal_group = broken
    with pytest.raises(OSError, match="kill failed"):
        run(w, pins_k3, first_block(pins_k3, 1))
    assert w.receivers[0].closed
