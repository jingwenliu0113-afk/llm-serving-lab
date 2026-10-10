#!/usr/bin/env python3
"""Advanced runner (decision 024; PREREGISTRATION §7): one server boot and one bench per plan boot, in the
seeded order of adv_plan.py, with the observability layer that adv_decomp reads. Core modules are imported,
never changed. Runs in WSL with the project venv, only after smoke FS1-FS5.

Per boot, in this order (024 "Harness 修改"):
1. the shared GPU lock is held (formal: --gpu-lock-held) and the GPU is idle (memory.used <= 1000 MiB);
   env_capture before the boot;
2. the OTLP receiver thread (127.0.0.1:4318 /v1/traces, http/protobuf, gzip): every body is saved under
   raw/<tag>.otlp/, llm_request spans are appended to raw/<tag>.spans.jsonl, the exporter always gets 200;
3. the server (adv_plan.server_argv_for) in its own process group, stdout / stderr piped into a reader thread
   that prefixes every line with int(time.time() * 1000) (the spans' wall clock) in raw/<tag>.server.log;
   wait for /health;
4. manipulation check: FLASH_ATTN in the log; the non-default args hold MNBT 2048, both observability keys and
   exactly the condition's keys (long_prefill_token_threshold 512, scheduler_reserve_full_isl False,
   kv_cache_memory_bytes of its K; vLLM never redacts them, api_utils.py:271-286); /metrics num_gpu_blocks /
   block_size equal the plan's frozen values. A mismatch stops the boot (no bench): ok = false, reasons;
5. full /metrics scrape, a 1 Hz sampler (kv_cache_usage_perc, num_requests_waiting / running,
   num_preemptions_total; wall timestamps), the bench (adv_plan.bench_argv_for, own session) with the plan's
   timeout (on timeout the client's process group is killed: ok = false, timeout = true), scrape again, then
   wait <= 30 s until the N main-result spans arrived;
6. the row: run_core_b.summarize_result plus the long class and client descriptives; the T / TG chunk check
   (every step in the bench window has ctx_tokens <= 512 x ctx_requests); preemptions from the scrape delta;
   the KV peak; adv_decomp.summarize_run as row["decomp"] (missing spans only invalidate the decomposition,
   never Y) and derived/<tag>-requests.jsonl;
7. teardown (coreb_smoke._shutdown on the whole process group, SIGKILL escalation included; any exception tears
   down first; SIGINT / SIGTERM / SIGHUP arriving meanwhile are only recorded and re-raised once the group is
   down, the log joined and the receiver closed), env_capture after, append the row.
Skip rule (024; 013): once a (cond, L, mix) cell has a run that is invalid under A1 (adv_analysis.row_valid),
its remaining restarts are recorded as skipped and not run. Harness failures abort the session after teardown
(session.json records "aborted") instead of becoming a row: a receiver that could not store a span body (also on
a stopped boot or during the shutdown flush), and a server log that could not be written (checked after the start,
after the bench and at teardown; the reader keeps draining the pipe so the server never blocks on it).
memory.used during a boot is descriptive (024): nvidia-smi is bounded by a timeout and a failure is recorded in
row["mem_probe_errors"] without touching ok or Y; the idle check before a boot stays a hard check.

--gpu-lock-held is verified, not trusted: at the start and before every boot this process must hold the shared lock
through a descriptor (held_gpu_lock: flock(1) without -o passes its locked fd to the command).

Formal sessions require freeze.advanced = true, --gpu-lock-held, pins [advanced] following 024
(adv_plan.pins_problems), a plan built from these pins bytes (pins_sha256) in the seeded 60-boot order, the
Stage B values (run_core_b.stage_b_problems), the pinned mixed dataset sha256, the pinned vLLM and one clean
commit (preflight). A session directory is never overwritten or resumed. raw/ stays in the local archive;
raw_manifest.json pins it.

Usage (WSL, ~/llm-serving-lab, after freeze.advanced):
    .venv/bin/python benchmarks/adv_plan.py --uniform-session results/core_b/formal-20261009-coreb-uniform \\
        --out adv-plan.json
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/run_adv.py --plan adv-plan.json \\
        --out results/advanced/<session> --gdn-prefill-backend triton --gpu-memory-utilization 0.91 \\
        --mixed-dataset 2048=data/core_b/mixed-2048.jsonl --mixed-dataset 8192=data/core_b/mixed-8192.jsonl \\
        --gpu-lock-held --formal

Stdlib only, Python 3.10 compatible: run_core_b (tomllib) and opentelemetry are imported lazily.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import fcntl
import hashlib
import http.server
import json
import os
import re
import signal
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Callable

import adv_analysis as aa
import adv_decomp as ad
import adv_plan as ap
import core_b_analysis as ba
import core_b_plan as bp
import coreb_smoke as smoke
from run_core_a import env_problems, provenance_problems, sha256_json, write_raw_manifest

SCHEMA_VERSION = 1
PORT = 8000            # as run_core_b
IDLE_MIB = 1000        # as run_core_b; 024 "每次 boot 前檢查 GPU 閒置"
IDLE_WAIT_S = 60       # a just-torn-down server may need a few seconds to release its memory
STARTUP_TIMEOUT_S = 1800
SPAN_WAIT_S = 30       # BatchSpanProcessor exports every 5 s (024)
LOG_JOIN_S = 10
NVIDIA_SMI_TIMEOUT_S = 15
RECEIVER_REQUEST_TIMEOUT_S = 10   # socket timeout of one OTLP request (a client stopping mid-body is dropped)
RECEIVER_CLOSE_TIMEOUT_S = 10     # bound on the receiver's shutdown at teardown
SHUTDOWN_ATTEMPTS = 3
INTERRUPTS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
CHUNKED = ("T", "TG")
GATE_OFF = ("G", "TG")
CONDITION_KEYS = ("long_prefill_token_threshold", "scheduler_reserve_full_isl", "kv_cache_memory_bytes")
WATCHED_KEYS = ("max_num_batched_tokens", "otlp_traces_endpoint", "enable_logging_iteration_details") + CONDITION_KEYS
SERIES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_waiting", "vllm:num_requests_running",
          "vllm:num_preemptions_total")
HISTOGRAMS = ("vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds",
              "vllm:time_to_first_token_seconds", "vllm:request_decode_time_seconds",
              "vllm:request_inference_time_seconds", "vllm:iteration_tokens_total")
COUNTERS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:num_preemptions_total")
BOOT_KEYS = ("block", "cond", "L", "mix", "restart", "C", "N", "warmup", "timeout_s", "prefix")


class RunError(RuntimeError):
    pass


class ReceiverError(RunError):
    pass


def _rb():
    """run_core_b, imported lazily: it imports tomllib (3.11+) at module level."""
    import run_core_b

    return run_core_b


def boot_tag(boot: dict) -> str:
    return f"{boot['cond']}-{boot['L']}-{boot['mix']}-r{boot['restart']}"


def _read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def _write_jsonl(path: Path, items) -> None:
    path.write_text("".join(json.dumps(x) + "\n" for x in items))


# ---------------------------------------------------------------- server log (stamped by the reader thread)

def pump_log(stream, log, *, wall: Callable = time.time) -> int:
    """Copy the server's output into `log` line by line, each prefixed with int(wall() * 1000) and a space (the
    wall clock of the spans; 024 量測). Bytes are decoded as UTF-8 with replacement. Returns the line count."""
    n = 0
    while True:
        raw = stream.readline()
        if not raw:
            return n
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        log.write(f"{int(wall() * 1000)} {text.rstrip(chr(13) + chr(10))}\n")
        log.flush()
        n += 1


def _drain(stream) -> None:
    """Read and discard until EOF, so the server never blocks writing into a full pipe."""
    try:
        while stream.readline():
            pass
    except (OSError, ValueError):
        pass


class LogReader:
    """Daemon thread pumping a server pipe into its (already open) log file; closes it at EOF, i.e. when the
    whole process group (API server and EngineCore share the pipe) is gone. A failure to write the log (ENOSPC,
    EIO, ...) is kept in `error` for the runner (which aborts the session) and the pipe is still drained to EOF:
    a dead reader would leave the server blocked on a full pipe and turn a harness failure into a bench timeout."""

    def __init__(self, stream, log, *, wall: Callable = time.time):
        self.lines, self.error = None, None
        self._thread = threading.Thread(target=self._run, args=(stream, log, wall), daemon=True)
        self._thread.start()

    def _run(self, stream, log, wall):
        try:
            self.lines = pump_log(stream, log, wall=wall)
        except Exception as exc:  # noqa: BLE001 - recorded for the runner, never a silent thread death
            self.error = f"{type(exc).__name__}: {exc}"
            _drain(stream)
        finally:
            try:
                log.close()
            except Exception as exc:  # noqa: BLE001 - the final flush can fail like any write
                self.error = self.error or f"{type(exc).__name__} closing the log: {exc}"

    def join(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()


def start_server(argv, env, log_path, *, wall: Callable = time.time):
    """Popen in a new session (pgid = pid, as coreb_smoke.start_server) with stdout / stderr piped into a
    LogReader; teardown with coreb_smoke._shutdown, then proc.log_reader.join(). The log is UTF-8 whatever the
    locale (the reader decodes with replacement, so every line encodes)."""
    log = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        raise
    proc.log_reader = LogReader(proc.stdout, log, wall=wall)
    return proc


# ---------------------------------------------------------------- manipulation check (024 操弄檢查)

_RE_NONDEFAULT = re.compile(r"non-default args: (\{.*\})")
_RE_VALUE = r"'{key}': ('(?:[^'\\]|\\.)*'|True|False|None|-?\d+(?:\.\d+)?)"


def parse_nondefault(log_text: str) -> dict | None:
    """The watched keys of vLLM's "non-default args: {...}" line (a dict repr, api_utils.py:286); None when the
    log has no such line. A key that is absent from the dict is absent here."""
    m = _RE_NONDEFAULT.search(log_text)
    if not m:
        return None
    out = {}
    for key in WATCHED_KEYS:
        v = re.search(_RE_VALUE.format(key=key), m.group(1))
        if v:
            out[key] = ast.literal_eval(v.group(1))
    return out


def expected_nondefault(cond: str, pins: dict) -> dict:
    """The watched non-default args a condition's server must log, and only these (024)."""
    adv = pins["advanced"]
    want = {"max_num_batched_tokens": bp.MAX_NUM_BATCHED_TOKENS, "otlp_traces_endpoint": adv["otlp_endpoint"],
            "enable_logging_iteration_details": True, "kv_cache_memory_bytes": ap.kv_cache_bytes_for(cond, pins)}
    if cond in CHUNKED:
        want["long_prefill_token_threshold"] = adv["long_prefill_token_threshold"]
    if cond in GATE_OFF:
        want["scheduler_reserve_full_isl"] = False
    return want


def manipulation_problems(log_text: str, cond: str, pins: dict) -> tuple:
    """(log facts, watched non-default args, problems) of one boot's server log."""
    facts = smoke.parse_server_log(log_text)
    problems = []
    if facts.get("attention_backend") != "FLASH_ATTN":
        problems.append(f"attention backend {facts.get('attention_backend')!r} != FLASH_ATTN (003, §4)")
    got = parse_nondefault(log_text)
    if got is None:
        return facts, None, problems + ["no non-default args line in the server log (024 manipulation check)"]
    want = expected_nondefault(cond, pins)
    for k in WATCHED_KEYS:
        if k in want and (k not in got or type(got[k]) is not type(want[k]) or got[k] != want[k]):
            problems.append(f"non-default args {k} = {got.get(k, '<absent>')!r}; condition {cond} needs "
                            f"{want[k]!r} (024)")
        elif k not in want and k in got:
            problems.append(f"non-default args carry {k} = {got[k]!r}, which condition {cond} must not set (024)")
    return facts, got, problems


def blocks_problems(metrics_text: str, *, expected_blocks, block_size) -> tuple:
    """(cache config, problems): /metrics num_gpu_blocks / block_size against the plan's frozen values."""
    rb = _rb()
    cfg = rb.cache_config(rb.parse_prometheus(metrics_text or ""))
    if cfg is None:
        return None, ["/metrics has no vllm:cache_config_info"]
    if not expected_blocks:
        return cfg, [f"num_gpu_blocks {cfg['num_gpu_blocks']}: the frozen count for this K is not measured yet "
                     "(0; smoke FS1 sets it)"]
    want = {"num_gpu_blocks": expected_blocks, "block_size": block_size}
    if cfg != want:
        return cfg, [f"num_gpu_blocks / block_size {cfg} != frozen {want}"]
    return cfg, []


def chunk_problems(steps: list, threshold: int) -> list:
    """T / TG: every step of the bench window has ctx_tokens <= threshold x ctx_requests (024)."""
    if not steps:
        return ["no iteration line in the bench window to check the chunk size"]
    bad = [s for s in steps if s["ctx_tok"] > threshold * s["ctx_req"]]
    if bad:
        return [f"{len(bad)} step(s) with ctx_tokens > {threshold} x ctx_requests "
                f"(first at index {bad[0]['index']}; 024)"]
    return []


# ---------------------------------------------------------------- row fields

def _class_ms(ttfts: list, lens: list, tokens: int) -> list:
    return [t * 1000 for t, n in zip(ttfts, lens) if n == tokens]


def _pct(values: list, q: float):
    return ba.percentile(values, q) if values else None


def client_stats(res: dict, *, L: int) -> dict:
    """024 client descriptives of one run: short-class P50 / P99 TTFT, long-class (32640) P50 / P95 TTFT and the
    short-class P95 TPOT ((latency - ttft) / (output - 1), as `vllm bench serve` defines TPOT)."""
    lens, ttfts = res.get("input_lens") or [], res.get("ttfts") or []
    lat, outs = res.get("latencies") or [], res.get("output_lens") or []
    aligned = len(lens) == len(ttfts)
    short = _class_ms(ttfts, lens, L - bp.N_GEN) if aligned else []
    long_ = _class_ms(ttfts, lens, bp.LONG_INPUT) if aligned else []
    tpot = []
    if aligned and len(lat) == len(outs) == len(lens):
        tpot = [(la - t) / (o - 1) * 1000 for la, t, o, n in zip(lat, ttfts, outs, lens)
                if n == L - bp.N_GEN and o > 1]
    return {"short_p50_ttft_ms": _pct(short, 50), "short_p99_ttft_ms": _pct(short, 99),
            "long_p50_ttft_ms": _pct(long_, 50), "long_p95_ttft_ms": _pct(long_, 95),
            "short_p95_tpot_ms": _pct(tpot, 95)}


def bench_fields(res: dict, *, L: int, C: int, mix: str) -> dict:
    """run_core_b.summarize_result (Core's validation and Y) plus the long class and the client descriptives."""
    out = _rb().summarize_result(res, L=L, C=C, mix=mix)
    cl = client_stats(res, L=L)
    return {**out, "long_class_p95_ttft_ms": cl["long_p95_ttft_ms"], "client": cl}


def scrape_deltas(before: str, after: str) -> dict:
    """Bench-window deltas of the two full scrapes (024 量測; they include warmup and ready-check). A series
    missing from either scrape is None; a counter reset or a bucket change is recorded in "errors"."""
    rb = _rb()
    b, a = rb.parse_prometheus(before or ""), rb.parse_prometheus(after or "")
    out, errors = {}, []
    for name in HISTOGRAMS:
        out[name] = None
        if f"{name}_count" in b and f"{name}_count" in a:
            try:
                d = ad.histogram_delta(b, a, name)
                out[name] = {**d, "mean": d["sum"] / d["count"] if d["count"] else None}
            except ad.DecompError as exc:
                errors.append(str(exc))
    for name in COUNTERS:
        out[name] = None
        if name in b and name in a:
            try:
                out[name] = ad.counter_delta(b, a, name)
            except ad.DecompError as exc:
                errors.append(str(exc))
    out["errors"] = errors
    return out


def series_summary(samples: list) -> dict:
    def col(k):
        return [s[k] for s in samples if s.get(k) is not None]

    return {"kv_peak": max(col("vllm:kv_cache_usage_perc"), default=None),
            "waiting_peak": max(col("vllm:num_requests_waiting"), default=None),
            "running_peak": max(col("vllm:num_requests_running"), default=None), "n_samples": len(samples)}


class TimeSeriesSampler:
    """The 1 Hz /metrics series with wall timestamps (024; a new class, run_core_b.MetricsSampler is unchanged).
    interval None samples only at start and stop."""

    def __init__(self, http_get: Callable, url: str, interval: float | None, *, wall: Callable = time.time):
        self.http_get, self.url, self.interval, self.wall = http_get, url, interval, wall
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def sample(self) -> None:
        rb = _rb()
        try:
            code, text = self.http_get(self.url)
        except OSError:
            return
        if code == 200:
            m = rb.parse_prometheus(text)
            self.samples.append({"t": self.wall(), **{k: rb.gauge(m, k) for k in SERIES}})

    def start(self) -> None:
        self.sample()
        if self.interval:
            def loop():
                while not self._stop.wait(self.interval):
                    self.sample()
            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()

    def stop(self) -> list:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.sample()
        return list(self.samples)


# ---------------------------------------------------------------- OTLP receiver (024 量測)

def otlp_decode(decoder=None) -> Callable:
    """decode(body, gzip) -> decoded span dicts (adv_decomp.decode_otlp_traces). The protobuf decoder
    (opentelemetry.proto, WSL venv only) is resolved here, once, so a missing package fails before any boot."""
    decoder = decoder or ad._otlp_decoder()
    return lambda body, gz: ad.decode_otlp_traces(body, gz, decoder=decoder)


class _Server(http.server.HTTPServer):
    """HTTPServer without server_bind's reverse-DNS lookup (socket.getfqdn)."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def _handler(receiver: "OtlpReceiver"):
    class Handler(http.server.BaseHTTPRequestHandler):
        # socket timeout (StreamRequestHandler.setup): a client that stops mid-request raises TimeoutError, which
        # handle_one_request turns into a closed connection, so the single-threaded server never hangs on it
        timeout = receiver.request_timeout_s

        def do_POST(self):  # noqa: N802 - http.server API
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            status = 404
            if urllib.parse.urlsplit(self.path).path == receiver.path:
                status = receiver.receive(body, self.headers.get("Content-Encoding"))
            self.send_response(status)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")  # an empty ExportTraceServiceResponse
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A002 - no per-request line on the runner's stderr
            pass

    return Handler


class OtlpReceiver:
    """One boot's in-process OTLP/HTTP trace receiver (http/protobuf, identity or gzip). Every body is saved as
    raw_dir/<n>.pb[.gz]; decoded llm_request spans are appended to spans_path (JSONL); main-result ids
    (cmpl-<prefix><i>-0) are counted. The exporter always gets 200; a body that cannot be decoded is recorded in
    decode_errors (its raw copy kept), a failure to store is recorded in errors and raised by raise_if_failed.
    Every request has a socket timeout (request_timeout_s) and close() is bounded (close_timeout_s)."""

    def __init__(self, raw_dir, spans_path, *, prefix: str, decode: Callable, host: str = "127.0.0.1",
                 port: int = 4318, path: str = "/v1/traces", request_timeout_s: float = RECEIVER_REQUEST_TIMEOUT_S,
                 close_timeout_s: float = RECEIVER_CLOSE_TIMEOUT_S):
        self.raw_dir, self.spans_path, self.decode = Path(raw_dir), Path(spans_path), decode
        self.host, self.port, self.path = host, port, path
        self.request_timeout_s, self.close_timeout_s = request_timeout_s, close_timeout_s
        self._main = re.compile("^cmpl-" + re.escape(prefix) + r"\d+-0$")
        self._lock = threading.Lock()
        self._main_ids: set = set()
        self.posts = self.n_spans = self.n_llm = 0
        self.decode_errors, self.errors = [], []
        self._server = self._thread = None

    def handle(self, body: bytes, encoding: str | None = None) -> None:
        """One POST body. Writing the raw copy or the JSONL raises; a decoding failure is recorded."""
        enc = (encoding or "identity").strip().lower()
        with self._lock:
            self.posts += 1
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            (self.raw_dir / f"{self.posts:05d}.pb{'.gz' if enc == 'gzip' else ''}").write_bytes(body)
            if enc not in ("identity", "gzip"):
                self.decode_errors.append(f"post {self.posts}: unsupported Content-Encoding {encoding!r}")
                return
            try:
                spans = self.decode(body, enc == "gzip")
            except Exception as exc:  # noqa: BLE001 - a bad body is recorded and its raw copy kept
                self.decode_errors.append(f"post {self.posts}: {type(exc).__name__}: {exc}")
                return
            keep = [s for s in spans if s.get("name") == ad.LLM_REQUEST]
            with open(self.spans_path, "a") as fh:
                fh.writelines(json.dumps(s) + "\n" for s in keep)
            self.n_spans += len(spans)
            self.n_llm += len(keep)
            ids = (str((s.get("attributes") or {}).get(ad.SPAN_ATTRS["id"]) or "") for s in keep)
            self._main_ids.update(i for i in ids if self._main.match(i))

    def receive(self, body: bytes, encoding: str | None = None) -> int:
        """What the HTTP handler calls: never raises and answers 200, so the exporter neither retries nor stalls."""
        try:
            self.handle(body, encoding)
        except Exception as exc:  # noqa: BLE001 - kept for raise_if_failed, never swallowed
            with self._lock:
                self.errors.append(f"{type(exc).__name__}: {exc}")
        return 200

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ReceiverError(f"the OTLP receiver could not store spans ({len(self.errors)} posts): "
                                + "; ".join(self.errors[:3]))

    def main_count(self) -> int:
        with self._lock:
            return len(self._main_ids)

    def stats(self) -> dict:
        with self._lock:
            return {"posts": self.posts, "spans": self.n_spans, "llm_request": self.n_llm,
                    "main": len(self._main_ids), "decode_errors": list(self.decode_errors),
                    "errors": list(self.errors)}

    def read_spans(self) -> list:
        with self._lock:
            if not self.spans_path.is_file():
                return []
            return [json.loads(x) for x in self.spans_path.read_text().splitlines() if x.strip()]

    def start(self) -> "OtlpReceiver":
        self._server = _Server((self.host, self.port), _handler(self))
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2},
                                        daemon=True)
        self._thread.start()
        return self

    def close(self) -> list:
        """Stop serving within close_timeout_s and close the socket; the teardown problems (empty when clean).
        HTTPServer.shutdown() waits for the request in progress, so it runs in a helper thread with a bound."""
        if self._server is None:
            return []
        server, thread, problems = self._server, self._thread, []
        self._server = self._thread = None
        stopper = threading.Thread(target=server.shutdown, daemon=True)
        stopper.start()
        stopper.join(self.close_timeout_s)
        if stopper.is_alive():
            problems.append(f"OTLP receiver did not stop within {self.close_timeout_s} s (a request still open); "
                            "its socket was closed anyway")
        server.server_close()
        thread.join(0 if problems else 5)
        if thread.is_alive() and not problems:
            problems.append("OTLP receiver thread still alive after close")
        return problems


def receiver_factory_for(pins: dict, decode: Callable | None = None) -> Callable:
    """make(raw_dir, spans_path, prefix) -> OtlpReceiver at pins [advanced].otlp_endpoint."""
    u = urllib.parse.urlsplit(pins["advanced"]["otlp_endpoint"])
    decode = decode or otlp_decode()

    def make(raw_dir, spans_path, prefix):
        return OtlpReceiver(raw_dir, spans_path, prefix=prefix, decode=decode, host=u.hostname, port=u.port,
                            path=u.path)

    return make


def wait_for_spans(receiver, n: int, *, now: Callable, sleep: Callable, limit_s: float = SPAN_WAIT_S,
                   poll_s: float = 1.0) -> tuple:
    """(main-result span count, seconds waited): poll until n main spans arrived or limit_s passed."""
    t = now()
    while receiver.main_count() < n and now() - t < limit_s:
        sleep(poll_s)
    return receiver.main_count(), now() - t


# ---------------------------------------------------------------- bench with a timeout

def default_run_bench(argv: list, log_path, env: dict, timeout_s: float) -> tuple:
    """`vllm bench serve` in its own session (returncode, timed_out). On timeout, or if the runner is interrupted,
    the client's whole process group gets SIGKILL (024: "逾時時 killpg")."""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None or smoke.group_alive(proc):
                smoke.signal_group(proc, signal.SIGKILL)
                proc.wait()
        return proc.returncode, timed_out


# ---------------------------------------------------------------- lock, nvidia-smi, teardown

def held_gpu_lock(lock_path: str = smoke.GPU_LOCK, fd_dir: str = "/proc/self/fd") -> bool:
    """Whether this process holds the shared GPU lock (MACHINE_POLICY), instead of trusting --gpu-lock-held.
    flock(1) without -o passes its locked descriptor to the command: some fd in fd_dir resolves to the lock file
    and flock(LOCK_EX | LOCK_NB) on it succeeds (the same open file description already holds the lock), while a
    lock held through another description fails with EWOULDBLOCK. No fd_dir (macOS), no descriptor on the lock
    file or EWOULDBLOCK: False."""
    want = os.path.realpath(lock_path)
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return False
    for name in names:
        if not name.isdigit():
            continue
        try:
            if os.path.realpath(os.path.join(fd_dir, name)) != want:
                continue
            fcntl.flock(int(name), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:  # EWOULDBLOCK (held elsewhere), or the fd closed meanwhile: not ours
            continue
    return False


def nvidia_mem_used_bounded(timeout_s: float = NVIDIA_SMI_TIMEOUT_S) -> int:
    """coreb_smoke.nvidia_mem_used with a timeout (Core stays unchanged): a hung nvidia-smi raises
    subprocess.TimeoutExpired instead of holding the runner, a live server and the GPU lock forever."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=timeout_s).stdout
    return int(out.split()[0])


class DeferredSignals:
    """While a teardown runs, SIGINT / SIGTERM / SIGHUP are only recorded, so a second Ctrl-C or a hangup cannot
    leave an orphaned server group on the GPU after the runner (and its lock) is gone. On exit the previous handlers
    come back and the first interrupt goes on: an interrupt the caller caught as an exception (`raised`) is
    re-raised, a recorded signal is handed to its previous handler (KeyboardInterrupt from Python's default,
    SystemExit from main's coreb_smoke._exit_on_signal; SystemExit(128 + signum) for SIG_DFL; nothing for SIG_IGN).
    Handlers can only be set in the main thread; elsewhere signals never interrupt the caller anyway."""

    def __init__(self):
        self.signals, self.raised, self._previous = [], [], {}

    def _record(self, signum, frame):
        self.signals.append(signum)

    def __enter__(self) -> "DeferredSignals":
        if threading.current_thread() is threading.main_thread():
            for sig in INTERRUPTS:
                if signal.getsignal(sig) is None:
                    continue  # a handler not installed from Python could not be put back
                self._previous[sig] = signal.signal(sig, self._record)
        return self

    def __exit__(self, *exc_info) -> bool:
        previous, self._previous = self._previous, {}
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if self.raised:
            raise self.raised[0]
        if self.signals:
            signum = self.signals[0]
            handler = previous.get(signum, signal.SIG_DFL)
            if callable(handler):
                handler(signum, None)
            elif handler != signal.SIG_IGN:
                raise SystemExit(128 + signum)
        return False


def shutdown_group(proc, deferred: DeferredSignals, **kw) -> list:
    """coreb_smoke._shutdown to its end (SIGINT, SIGTERM, SIGKILL). An interrupt raised out of it anyway (one that
    arrived before the deferral, or raised by a caller's sleep) is kept in deferred.raised and the shutdown runs
    again, at most SHUTDOWN_ATTEMPTS times."""
    problems = []
    for _ in range(SHUTDOWN_ATTEMPTS):
        try:
            return problems + smoke._shutdown(proc, **kw)
        except (KeyboardInterrupt, SystemExit) as exc:
            deferred.raised.append(exc)
            problems.append(f"teardown interrupted by {type(exc).__name__}; shutdown run again")
    return problems + [f"teardown interrupted {SHUTDOWN_ATTEMPTS} times; the server group may still be alive"]


# ---------------------------------------------------------------- refusals

def plan_problems(plan: dict, pins: dict, *, formal: bool) -> list:
    """Every boot is a 024 cell with Core's N / warmup, the plan's C and timeout and a deterministic prefix;
    formal plans are exactly adv_plan.build_plan's 60-boot seeded order from pins [advanced]."""
    boots = plan.get("boots") or []
    if not boots:
        return ["the plan has no boots"]
    problems, seen = [], set()
    c = {int(L): C for L, C in (plan.get("c") or {}).items()}
    blocks, timeouts = plan.get("expected_blocks") or {}, plan.get("timeouts_s") or {}
    if not isinstance(plan.get("block_size"), int):
        problems.append("the plan has no block_size")
    for i, b in enumerate(boots):
        missing = [k for k in BOOT_KEYS if k not in b]
        if missing:
            problems.append(f"boot {i} lacks {missing}")
            continue
        try:
            prefix = ap.request_id_prefix(b["cond"], b["L"], b["mix"], b["restart"])
        except ap.PlanError as exc:
            problems.append(f"boot {i}: {exc}")
            continue
        tag = boot_tag(b)
        if b["prefix"] != prefix:
            problems.append(f"boot {tag}: prefix {b['prefix']!r} != {prefix!r}")
        if b["C"] != c.get(b["L"]):
            problems.append(f"boot {tag}: C = {b['C']} != plan c {c.get(b['L'])}")
        if b["N"] != bp.num_prompts(b["C"]) or b["warmup"] != bp.num_warmups(b["C"]):
            problems.append(f"boot {tag}: N / warmup {b['N']} / {b['warmup']} != Core's "
                            f"{bp.num_prompts(b['C'])} / {bp.num_warmups(b['C'])} at C = {b['C']}")
        want_t = timeouts.get(f"{b['mix']}_{b['L']}")
        if not (isinstance(want_t, int) and want_t > 0 and b["timeout_s"] == want_t):
            problems.append(f"boot {tag}: timeout_s {b['timeout_s']!r} != plan timeouts_s {want_t!r}")
        if tag in seen:
            problems.append(f"duplicate boot {tag}")
        seen.add(tag)
    for cond in sorted({b.get("cond") for b in boots if b.get("cond") in ap.CONDITIONS}):
        if not isinstance(blocks.get(cond), int):
            problems.append(f"plan expected_blocks has no count for {cond}")
    if formal:
        problems += _formal_plan_problems(plan, pins, boots)
    return problems


def _formal_plan_problems(plan: dict, pins: dict, boots: list) -> list:
    adv, problems = pins.get("advanced") or {}, []
    want = [(r, cell) for r in range(1, ap.RESTARTS + 1) for cell in ap.block_order(adv.get("seed"), r)]
    got = [(b.get("block"), (b.get("cond"), b.get("L"), b.get("mix"))) for b in boots]
    if got != want or any(b.get("restart") != b.get("block") for b in boots):
        problems.append("the boots are not the seeded 60-boot order of adv_plan.build_plan (024)")
    if plan.get("seed") != adv.get("seed"):
        problems.append(f"plan seed {plan.get('seed')!r} != pins {adv.get('seed')!r}")
    if plan.get("derived_from") != ap.DERIVED_FROM:
        problems.append(f"plan derived_from {plan.get('derived_from')!r} != {ap.DERIVED_FROM!r} (024)")
    if {int(L): C for L, C in (plan.get("c") or {}).items()} != ap.C_KNEE:
        problems.append(f"plan c {plan.get('c')!r} != {ap.C_KNEE} (024)")
    blocks = {cond: adv.get("num_gpu_blocks_k3") if cond == "K3" else adv.get("num_gpu_blocks_k4")
              for cond in ap.CONDITIONS}
    if plan.get("expected_blocks") != blocks or not all(isinstance(v, int) and v > 0 for v in blocks.values()):
        problems.append(f"plan expected_blocks {plan.get('expected_blocks')!r} != pins {blocks} (all measured)")
    if plan.get("timeouts_s") != adv.get("timeouts_s"):
        problems.append(f"plan timeouts_s {plan.get('timeouts_s')!r} != pins {adv.get('timeouts_s')!r}")
    if plan.get("block_size") != (pins.get("stage_b") or {}).get("block_size"):
        problems.append(f"plan block_size {plan.get('block_size')!r} != pins [stage_b]")
    return problems


def formal_problems(plan: dict, pins: dict, *, pins_sha256: str, gdn_prefill_backend: str,
                    gpu_memory_utilization: str) -> list:
    problems = [f"pins [advanced]: {p}" for p in ap.pins_problems(pins)]
    if plan.get("pins_sha256") != pins_sha256:
        problems.append(f"plan pins_sha256 {plan.get('pins_sha256')!r} != configs/pins.toml {pins_sha256!r}: "
                        "build the plan from the frozen pins (024 A14)")
    problems += _rb().stage_b_problems(plan, pins, gdn_prefill_backend=gdn_prefill_backend,
                                       gpu_memory_utilization=gpu_memory_utilization)
    return problems


def recorded_server_env(env: dict) -> dict:
    """What a session records of a server env: Core's recorded subset, every OTEL_* variable and the names of
    the engine variables (adv_report checks it against the condition table)."""
    return {**bp.recorded_env(env), "OTEL": {k: env[k] for k in sorted(env) if k.startswith("OTEL_")},
            "engine_vars": sorted(k for k in env if k.startswith(bp.ENGINE_PREFIXES))}


# ---------------------------------------------------------------- rows

def _row_base(boot: dict) -> dict:
    return {**{k: boot[k] for k in ("cond", "L", "mix", "restart", "block", "C", "N", "prefix", "timeout_s")},
            "tag": boot_tag(boot), "ok": False, "reasons": [], "timeout": False, "skipped": False,
            "short_class_p95_ttft_ms": None, "long_class_p95_ttft_ms": None, "preemptions": None, "kv_peak": None,
            "manip": {"facts_ok": None, "blocks_ok": None, "chunk_ok": None, "num_gpu_blocks": None},
            "decomp": None}


def skipped_row(boot: dict, first_invalid: str) -> dict:
    cell = (boot["cond"], boot["L"], boot["mix"])
    return {**_row_base(boot), "skipped": True,
            "reasons": [f"skipped: cell {cell} already has an invalid run ({first_invalid}); the cell is NT "
                        "(024; 013), so its remaining restarts are not run"]}


def outcomes(rows: list) -> dict:
    return {"boots": len(rows), "ok": sum(bool(r["ok"]) for r in rows),
            "valid": sum(aa.row_valid(r) for r in rows), "invalid": sum(not aa.row_valid(r) for r in rows),
            "timeout": sum(bool(r["timeout"]) for r in rows), "skipped": sum(bool(r["skipped"]) for r in rows)}


# ---------------------------------------------------------------- session

def run_session(plan: dict, *, pins: dict, pins_sha256: str, out_dir, gdn_prefill_backend: str,
                gpu_memory_utilization: str, session_id: str, popen: Callable, signal_group: Callable,
                group_alive: Callable, http_get: Callable, run_bench: Callable, gpu_mem_used: Callable,
                gpu_lock_free: Callable, receiver_factory: Callable | None = None, gpu_lock_held: bool = False,
                lock_held_check: Callable | None = None,
                formal: bool = False, vllm_bin: str = ".venv/bin/vllm", mixed_datasets: dict | None = None,
                sample_interval: float | None = 1.0, toolchain_check: Callable = bp.toolchain_problems,
                capture_env: Callable | None = None, decompose: Callable = ad.summarize_run, base_env=None,
                now: Callable = time.monotonic, sleep: Callable = time.sleep, wall: Callable = time.time,
                clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(),
                startup_timeout_s: float = STARTUP_TIMEOUT_S, span_wait_s: float = SPAN_WAIT_S,
                idle_wait_s: float = IDLE_WAIT_S) -> dict:
    base_env = dict(os.environ if base_env is None else base_env)
    lock_held_check = lock_held_check or held_gpu_lock
    if formal and not gpu_lock_held:
        raise RunError("formal Advanced runs must hold the GPU lock: flock -n /tmp/ai-portfolio-gpu.lock ... "
                       "--gpu-lock-held (MACHINE_POLICY)")
    if gpu_lock_held and not lock_held_check():
        raise RunError(f"--gpu-lock-held, but this process does not hold {smoke.GPU_LOCK} (no locked descriptor "
                       f"on it): run under flock -n {smoke.GPU_LOCK} ... (MACHINE_POLICY)")
    if formal and (pins.get("freeze") or {}).get("advanced") is not True:
        raise RunError("formal Advanced runs require freeze.advanced = true (024 A14)")
    problems = plan_problems(plan, pins, formal=formal)
    if formal:
        problems += formal_problems(plan, pins, pins_sha256=pins_sha256, gdn_prefill_backend=gdn_prefill_backend,
                                    gpu_memory_utilization=gpu_memory_utilization)
    if problems:
        raise RunError("; ".join(problems))
    mixed_sha = {}
    for L in sorted({b["L"] for b in plan["boots"] if b["mix"] == "mixed"}):
        path = (mixed_datasets or {}).get(L)
        if not path or not Path(path).is_file():
            raise RunError(f"mixed boots at L={L} need --mixed-dataset {L}=<file> (decision 007)")
        mixed_sha[str(L)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        pinned = (pins["core_b"].get("mix_dataset_sha256") or {}).get(str(L))
        if formal and mixed_sha[str(L)] != pinned:
            raise RunError(f"mixed dataset for L={L}: sha256 {mixed_sha[str(L)]} != pinned {pinned!r} "
                           "(core_b.mix_dataset_sha256)")
    missing = toolchain_check(ap.server_env_for(base_env, pins))
    if missing:
        raise RunError("toolchain incomplete, not starting (014): " + "; ".join(missing))
    if receiver_factory is None:
        try:
            receiver_factory = receiver_factory_for(pins)
        except ImportError as exc:
            raise RunError(f"OTLP decoder unavailable ({exc}); run with the vLLM venv, which has opentelemetry "
                           "(024)") from exc
    out = Path(out_dir)
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten")
    raw, derived = out / "raw", out / "derived"
    raw.mkdir(parents=True)
    (out / "env").mkdir()
    derived.mkdir()
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=gpu_lock_held) if capture_env else {}
    if formal:
        got = ((pre.get("runtime") or {}).get("vllm") or {}).get("vllm")
        if got != pins["vllm"]["version"]:
            raise RunError(f"vLLM {got!r} in the preflight != pinned {pins['vllm']['version']} (003)")
        bad = env_problems(pre, "wsl2-cuda") + provenance_problems(pre, pre)
        if bad:
            raise RunError("preflight: " + "; ".join(bad))
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "formal": formal,
               "gpu_lock_held": gpu_lock_held, "plan_sha256": sha256_json(plan), "pins_sha256": pins_sha256,
               "plan_pins_sha256": plan.get("pins_sha256"), "derived_from": plan.get("derived_from"),
               "started_at": clock(), "preflight": "env/preflight.json" if capture_env else None,
               "repo_commit": (pre.get("repo") or {}).get("commit"),
               "server": {"gdn_prefill_backend": gdn_prefill_backend, "gpu_memory_utilization": gpu_memory_utilization,
                          "port": PORT, "vllm_bin": vllm_bin},
               "result_dir": str(raw), "mixed_datasets": {str(L): str(p) for L, p in (mixed_datasets or {}).items()},
               "mixed_dataset_sha256": mixed_sha, "server_env": recorded_server_env(ap.server_env_for(base_env, pins)),
               "inherited_unlisted": bp.inherited_unlisted(base_env),
               "inherited_otel": sorted(k for k in base_env if k.startswith("OTEL_")), "runs": {}}
    env_flags, rows, invalid_cells = [], [], {}
    base = f"http://127.0.0.1:{PORT}"

    def snap(name: str) -> None:
        if capture_env:
            rec = capture_env(out_path=out / "env" / name, gpu_lock_held=gpu_lock_held)
            env_flags.extend({"env": name, "reason": why} for why in env_problems(rec, "wsl2-cuda"))
            if formal:
                env_flags.extend({"env": name, "reason": why} for why in provenance_problems(rec, pre))

    def wait_idle(tag: str) -> None:
        """Hard checks before a boot: the lock (held by this process, or free for a non-held run) and an idle GPU;
        an nvidia-smi failure here stops the session."""
        if gpu_lock_held:
            if not lock_held_check():
                raise RunError(f"--gpu-lock-held, but this process does not hold {smoke.GPU_LOCK} any more; "
                               f"stopping before {tag} (MACHINE_POLICY)")
        elif not gpu_lock_free():
            raise RunError(f"{smoke.GPU_LOCK} is held by another project; stopping before {tag} (MACHINE_POLICY)")
        t, mem = now(), gpu_mem_used()
        while mem > IDLE_MIB and now() - t < idle_wait_s:
            sleep(2)
            mem = gpu_mem_used()
        if mem > IDLE_MIB:
            raise RunError(f"GPU busy before {tag}: memory.used {mem} MiB > {IDLE_MIB} after {idle_wait_s} s "
                           "(MACHINE_POLICY)")

    def mem_probe(row: dict, what: str):
        """memory.used during a boot is descriptive (024): a failed or timed-out nvidia-smi is recorded in
        row["mem_probe_errors"] and costs neither ok, Y nor the session."""
        try:
            return gpu_mem_used()
        except Exception as exc:  # noqa: BLE001 - subprocess.TimeoutExpired, CalledProcessError, "[N/A]", ...
            row.setdefault("mem_probe_errors", []).append(f"{what}: {type(exc).__name__}: {exc}")
            return None

    def check_log(proc, when: str) -> None:
        """A server log that could not be written is a harness failure: stop the session, write no row."""
        error = getattr(getattr(proc, "log_reader", None), "error", None)
        if error:
            raise RunError(f"the server log could not be written ({error}) {when}: a harness failure, not a "
                           "result; session aborted")

    def checked_start(boot: dict, row: dict, record: dict, proc) -> dict | None:
        """Wait for /health, then the manipulation and block checks; the cache config, or None to stop."""
        cond, tag = boot["cond"], row["tag"]
        t0, ready = now(), False
        while now() - t0 < startup_timeout_s and proc.poll() is None:
            if http_get(f"{base}/health")[0] == 200:
                ready = True
                break
            sleep(2)
        record["startup_s"] = round(now() - t0, 1)
        facts, nondefault, manip = manipulation_problems(_read(raw / f"{tag}.server.log"), cond, pins)
        code, text = http_get(f"{base}/metrics") if ready else (0, "")
        cfg, blocks = blocks_problems(text if code == 200 else "", expected_blocks=plan["expected_blocks"][cond],
                                      block_size=plan["block_size"])
        record.update(log_facts=facts, nondefault=nondefault, cache_config=cfg)
        row["manip"].update(facts_ok=not manip, blocks_ok=not blocks, num_gpu_blocks=(cfg or {}).get("num_gpu_blocks"))
        row["kv_cache_tokens"] = facts.get("kv_cache_tokens")
        stop = ([] if ready else [f"server not ready within {startup_timeout_s} s (exit code {proc.poll()!r})"])
        stop += manip + blocks
        row["reasons"] = stop
        return None if stop else cfg

    def bench_and_measure(boot: dict, row: dict, record: dict, receiver, cfg: dict, proc) -> None:
        cond, L, mix, tag = boot["cond"], boot["L"], boot["mix"], row["tag"]
        url = f"{base}/metrics"
        row["mem_ready_mib"] = mem_probe(row, "at ready")
        t_before = wall()
        code0, before = http_get(url)
        before = before if code0 == 200 else ""
        (raw / f"{tag}.metrics.before.txt").write_text(before)
        bargv = ap.bench_argv_for(pins, cond=cond, L=L, C=boot["C"], mix=mix, restart=boot["restart"],
                                  result_dir=str(raw), result_filename=f"{tag}.bench.json", port=PORT,
                                  mixed_dataset=(mixed_datasets or {}).get(L) if mix == "mixed" else None,
                                  vllm_bin=vllm_bin)
        row["argv"] = bargv
        sampler = TimeSeriesSampler(http_get, url, sample_interval, wall=wall)
        sampler.start()
        try:
            rc, timed_out = run_bench(bargv, raw / f"{tag}.bench.log", {**base_env, **bp.bench_env()},
                                      boot["timeout_s"])
        finally:
            samples = sampler.stop()
        check_log(proc, "during the bench")  # a stalled server's timeout is not a result
        row["healthy_after_bench"] = http_get(f"{base}/health")[0] == 200
        row["mem_after_bench_mib"] = mem_probe(row, "after the bench")
        code1, after = http_get(url)
        after = after if code1 == 200 else ""
        t_after = wall()
        record.update(t_before=t_before, t_after=t_after)
        (raw / f"{tag}.metrics.after.txt").write_text(after)
        _write_jsonl(raw / f"{tag}.series.jsonl", samples)
        path = raw / f"{tag}.bench.json"
        res = json.loads(path.read_text()) if rc == 0 and not timed_out and path.is_file() else None
        waited = wait_for_spans(receiver, boot["N"], now=now, sleep=sleep, limit_s=span_wait_s)[1] \
            if res is not None else 0.0
        receiver.raise_if_failed()
        row["spans"] = {**receiver.stats(), "waited_s": round(waited, 1)}
        reasons = []
        if timed_out:
            row["timeout"] = True
            reasons.append(f"timeout: the bench exceeded {boot['timeout_s']} s; its process group was killed (024)")
        elif res is None:
            reasons.append(f"vllm bench serve exit {rc}")
        if res is not None:
            fields = bench_fields(res, L=L, C=boot["C"], mix=mix)
            reasons = fields.pop("reasons") + reasons
            row.update(fields)
            row["result"] = f"raw/{tag}.bench.json"
        try:
            pre_delta = ad.counter_delta(before, after, ad.PREEMPTIONS)
        except ad.DecompError:
            pre_delta = None
        series = series_summary(samples)
        row.update(preemptions=None if pre_delta is None else int(pre_delta), kv_peak=series["kv_peak"],
                   series=series, one_hz=ad.one_hz_shares(samples, nb=cfg["num_gpu_blocks"], L=L),
                   scrape_delta=scrape_deltas(before, after))
        steps = ad.parse_iteration_log(_read(raw / f"{tag}.server.log").splitlines())
        _write_jsonl(raw / f"{tag}.steps.jsonl", steps)
        window = ad.steps_in_window(steps, t_before, t_after)
        row["n_steps_window"] = len(window)
        if cond in CHUNKED and (window or res is not None):
            chunk = chunk_problems(window, pins["advanced"]["long_prefill_token_threshold"])
            row["manip"]["chunk_ok"] = not chunk
            reasons += chunk
        if res is not None:
            spans = receiver.read_spans()
            try:
                row["decomp"] = decompose(
                    spans=spans, detailed=res, steps=steps, prefix=boot["prefix"], L=L, cond=cond,
                    num_gpu_blocks=cfg["num_gpu_blocks"], metrics_before=before, metrics_after=after,
                    window=(t_before, t_after), block_size=plan["block_size"],
                    long_prefill_token_threshold=pins["advanced"]["long_prefill_token_threshold"])
                records = [r for r in ad.llm_request_records(spans)
                           if r.get("arrival") is not None and t_before <= r["arrival"] <= t_after]
                _write_jsonl(derived / f"{tag}-requests.jsonl",
                             ad.derived_requests(ad.align(records, res, prefix=boot["prefix"], L=L)))
                row["derived"] = f"derived/{tag}-requests.jsonl"
            except Exception as exc:  # noqa: BLE001 - the descriptive layer never costs a measured Y
                row["decomp"] = None
                row["decomp_error"] = f"{type(exc).__name__}: {exc}"
        row["ok"] = bool(res is not None and row.get("ok") and not timed_out)
        row["reasons"] = reasons

    def teardown(tag: str, proc, receiver, record: dict) -> None:
        """Server group down (SIGKILL escalation included), log reader joined, receiver closed (even when the
        shutdown raised), receiver errors and teardown problems recorded; interrupts arriving meanwhile wait until
        all of that is done (DeferredSignals)."""
        with DeferredSignals() as deferred:
            problems = []
            try:
                if proc is not None:
                    problems += shutdown_group(proc, deferred, signal_group=signal_group, group_alive=group_alive,
                                               now=now, sleep=sleep)
                    reader = getattr(proc, "log_reader", None)
                    if reader is not None and not reader.join(LOG_JOIN_S):
                        problems.append("server log reader still open after teardown")
            finally:
                try:
                    problems += receiver.close() or []
                finally:
                    errors = list(getattr(receiver, "errors", None) or [])
                    if errors:
                        record["receiver_errors"] = errors
                    if problems:
                        record["teardown_problems"] = problems
                        session.setdefault("teardown_problems", []).append({"run": tag, "reasons": problems})

    def boot_once(boot: dict) -> dict:
        tag = boot_tag(boot)
        wait_idle(tag)
        snap(f"{tag}.before.json")
        argv, env = ap.server_argv_for(boot["cond"], pins, gdn_prefill_backend=gdn_prefill_backend,
                                       gpu_memory_utilization=gpu_memory_utilization, port=PORT, vllm_bin=vllm_bin,
                                       base_env=base_env)
        record = session["runs"][tag] = {"argv": argv, "env": recorded_server_env(env)}
        row = _row_base(boot)
        receiver = receiver_factory(raw / f"{tag}.otlp", raw / f"{tag}.spans.jsonl", boot["prefix"])
        receiver.start()
        proc = None
        try:
            proc = popen(argv, env, raw / f"{tag}.server.log")
            cfg = checked_start(boot, row, record, proc)
            check_log(proc, "during the server start")
            if cfg is not None:
                bench_and_measure(boot, row, record, receiver, cfg, proc)
        finally:
            teardown(tag, proc, receiver, record)
        # harness failures found at teardown (outside the finally, so they never mask an exception in flight)
        check_log(proc, "at teardown")
        receiver.raise_if_failed()
        snap(f"{tag}.after.json")
        return row

    with open(out / "rows.jsonl", "w") as fh:
        try:
            for boot in plan["boots"]:
                cell = (boot["cond"], boot["L"], boot["mix"])
                row = skipped_row(boot, invalid_cells[cell]) if cell in invalid_cells else boot_once(boot)
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if not aa.row_valid(row) and cell not in invalid_cells:
                    invalid_cells[cell] = boot_tag(boot)
        except BaseException as exc:
            session["aborted"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            session.update({"finished_at": clock(), "outcomes": outcomes(rows), "env_flags": env_flags,
                            "env_valid": not env_flags})
            write_raw_manifest(out)
            (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    return session


def main(argv: list | None = None) -> int:
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gdn-prefill-backend", required=True, choices=["triton", "flashinfer"])
    parser.add_argument("--gpu-memory-utilization", required=True)
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--mixed-dataset", action="append", default=[], help="L=path (decision 007)")
    parser.add_argument("--gpu-lock-held", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    raw = (root / "configs" / "pins.toml").read_bytes()
    rb = _rb()
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, smoke._exit_on_signal)
    try:
        summary = run_session(
            json.loads(Path(args.plan).read_text()), pins=tomllib.loads(raw.decode()),
            pins_sha256=hashlib.sha256(raw).hexdigest(), out_dir=args.out,
            gdn_prefill_backend=args.gdn_prefill_backend, gpu_memory_utilization=args.gpu_memory_utilization,
            session_id=Path(args.out).name, popen=start_server, signal_group=smoke.signal_group,
            group_alive=smoke.group_alive, http_get=rb.default_http_get, run_bench=default_run_bench,
            gpu_mem_used=nvidia_mem_used_bounded, gpu_lock_free=smoke.gpu_lock_free, gpu_lock_held=args.gpu_lock_held,
            lock_held_check=held_gpu_lock,
            formal=args.formal, vllm_bin=args.vllm_bin,
            capture_env=rb.make_vllm_env_capture(root / "scripts" / "env_capture.py"),
            mixed_datasets={int(k): v for k, v in (x.split("=", 1) for x in args.mixed_dataset)})
    except RunError as exc:
        print(f"run_adv: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary["outcomes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
