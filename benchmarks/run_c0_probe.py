#!/usr/bin/env python3
"""C0 probe runner (decision 028; exploratory, formal = false): 6 boots x 4 benches of the Core B server and client at
8192 uniform, C 8, under the shared GPU lock, smoke first. Imports the frozen Core / Advanced modules and changes none.

Subcommands:
- plan: the deterministic plan (build_plan): pins [stage_b] K / blocks / block size / U / GDN backend, the pins
  [advanced] 8K uniform timeout (720 s), L 8192 uniform, C 8, N 200, 8 warmups, seed 20261008, 6 boots x 4 benches,
  smoke 1 boot x 1 bench at N 16, cap 3.0 h, 30 min boot margin, 88 C, 2 failed boots; a --plan that differs is refused;
- smoke: 1 boot, 1 bench (N 16, C 8) into results/capacity/smoke-c0-<YYYYMMDD>/ (stage "C0-smoke": never analysed);
- run [--smoke-first]: the smoke, then the probe into results/capacity/explore-c0-<YYYYMMDD>/ only if the smoke passed
  (no stop, its bench valid, and its sampler reported temperatures without a problem: smoke_problems).

Per boot (028 「每次 boot 都相同的量測」 and 「停止規則」):
1. before it: no new boot when elapsed + 30 min > cap (stop "cap"); run_adv.held_gpu_lock() (else stop "lock");
   memory.used <= 1000 MiB within 60 s, then no compute process on the GPU (else stop "gpu_busy"); env capture;
2. one nvidia-smi process at 1 Hz (pstate, clocks.sm, clocks.mem, power.draw, temperature.gpu, utilization.gpu,
   memory.used, clocks_event_reasons.active) whose lines run_adv.LogReader stamps with the epoch ms into
   raw/<boot>.smi.csv; a sample >= 88 C sets the stop flag, and so does losing that telemetry (the sampler did not
   start, exited, or sent no temperature for 30 s: stop "telemetry", since rule 6 reads nothing else);
3. the server: core_b_plan.server_argv with K (Core's argv and env, no observation flag) through run_adv.start_server
   (epoch-ms stamped log); /health, waited for at most the startup timeout or what is left of the cap (a start cut at
   the cap stops "cap"); startup_facts (non-default args, backends, compile cache, CUDA graphs, KV memory,
   init time, WARNING / ERROR). The start failed when the server is not ready, /metrics blocks / block size are not
   248 / 528 (run_adv.blocks_problems) or the watched non-default args are not exactly {max_num_batched_tokens 2048,
   kv_cache_memory_bytes K} (run_adv.parse_nondefault): then no bench runs;
4. 4 benches in a row (core_b_plan.bench_argv; the smoke changes only --num-prompts): no bench whose 720 s timeout
   could pass the cap (stop "cap"); a full /metrics scrape, run_core_b.MetricsSampler at 1 Hz, the bench (own session;
   its process group is killed at the timeout or as soon as the stop flag is set), /health, a full scrape; the row:
   run_core_b.summarize_result, run_adv.scrape_deltas (c0_analysis reads k_prom from it), the blocks of the scrape
   before, the bench window (wall clock) and position, derived/<boot>-n<j>-requests.jsonl (idx, lengths, TTFT, latency).
   A bench is invalid on exit != 0 or a timeout, completed != N or failed > 0, an unhealthy server after it, blocks
   != 248, or non-default args (the start's check); it is excluded and never replaced. A timeout or an unhealthy
   server ends the boot (the server is torn down); its remaining benches are rows with not_run = true;
5. teardown (run_adv.shutdown_group under run_adv.DeferredSignals, log reader joined), the sampler stopped and its
   samples written to derived/<boot>-smi.jsonl, env capture after. A boot failed when its server did not start or
   >= 2 of its benches are invalid; the 2nd failed boot stops the session ("failed_boots"); 88 C stops it at once
   ("temperature"; "telemetry" for lost telemetry), a flag set during the teardown included. Nothing reads latency
   to stop or extend the session (028 rule 7).

Sessions: refused without --gpu-lock-held verified by run_adv.held_gpu_lock (flock(1) passes its locked descriptor),
for a plan other than build_plan's, pins [stage_b] drift (run_core_b.stage_b_problems), a --cap-hours other than the
plan's, a toolchain problem (core_b_plan.toolchain_problems), an output directory inside a Phase 1 results directory
(results/{core_a,core_b,advanced,quality,stage_b,decision_table,artifacts}; run checks both before the smoke) or an
existing output directory. session.json holds {formal: false, stage: "C0", purpose: "exploratory", decision: "028"},
the server argv and env, the driver / vLLM / torch versions against Phase 1 (recorded, never a stop: 028 rule 9),
every boot and the stop rule that ended the session; plus plan.json, rows.jsonl, derived/, env/ and raw_manifest.json
(raw/ stays git-ignored). The cap (3.0 h of holding the lock, smoke included) counts from this process's start: flock
starts it once it holds the lock.
Harness failures (an exception, a server log that could not be written) tear down first, record "aborted" and raise.

Usage (WSL, ~/llm-serving-lab, after deploying and checking DEPLOYED_COMMIT and tools/rtx/gpu_status.sh FREE):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/run_c0_probe.py run --smoke-first \\
        --gpu-lock-held --cap-hours 3.0

Stdlib only, Python 3.10 compatible (run_core_b and tomllib are imported lazily).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import core_b_plan as bp
import coreb_smoke as smoke
import run_adv as ra
from run_core_a import env_problems, sha256_json, write_raw_manifest

SCHEMA_VERSION = 1
DECISION = "028"
PORT = 8000                 # as run_core_b / run_adv
L, MIX = 8192, "uniform"
C = 8                       # 028: Core's 8K C_knee (pins [advanced].c)
BOOTS, BENCHES_PER_BOOT = 6, 4
SMOKE_N = 16
CAP_HOURS = 3.0
BOOT_MARGIN_S = 1800        # no new boot once elapsed + 30 min > cap
TEMP_STOP_C = 88
MAX_FAILED_BOOTS = 2
IDLE_MIB, IDLE_WAIT_S = ra.IDLE_MIB, ra.IDLE_WAIT_S
STARTUP_TIMEOUT_S = ra.STARTUP_TIMEOUT_S
SMI_INTERVAL_MS = 1000
SMI_STALE_S = 30            # no temperature sample for this long: the telemetry 028 rule 6 needs is lost
SMI_POLL_S = 1.0
SMI_FIELDS = ("timestamp", "pstate", "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "utilization.gpu",
              "memory.used", "clocks_event_reasons.active")
SMI_KEYS = ("pstate", "sm_mhz", "mem_mhz", "power_w", "temp_c", "util_pct", "mem_used_mib", "event_reasons")
# Phase 1 runtime (results/advanced/formal-20261011-adv/env/preflight.json; vLLM is pins [vllm].version)
PHASE1_DRIVER = "617.42"
PHASE1_TORCH = "2.13.0+cu130"
ROOT = Path(__file__).resolve().parent.parent
# Phase 1 results directories: C0 never writes into them (027 §1; 028 writes under results/capacity/)
PHASE1_DIRS = ("core_a", "core_b", "advanced", "quality", "stage_b", "decision_table", "artifacts")


class RunError(RuntimeError):
    pass


class Stop(Exception):
    """A 028 stop rule: the session ends cleanly (teardown done) and session.json records the rule."""

    def __init__(self, rule: str, reason: str):
        super().__init__(reason)
        self.rule, self.reason = rule, reason


def _rb():
    """run_core_b, imported lazily: it imports tomllib (3.11+) at module level."""
    import run_core_b

    return run_core_b


def _read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def boot_tag(b: int) -> str:
    return f"b{b}"


def bench_tag(b: int, j: int) -> str:
    return f"b{b}-n{j}"


def inside(path, parent) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def out_problems(out) -> list:
    """An output inside a Phase 1 results directory (shared with c0_step0)."""
    return [f"{out} is inside the Phase 1 directory {d}" for d in (ROOT / "results" / x for x in PHASE1_DIRS)
            if inside(out, d)]


# ---------------------------------------------------------------- startup-line extractor (028 量測)

STARTUP_MAX_LINES = 40
STARTUP_READY = "Application startup complete"
# matched against the message (the "[file.py:NNN]" header removed, so file names never match)
STARTUP_PATTERNS = {
    "nondefault_args": re.compile(r"non-default args: "),
    "backend": re.compile(r"[Bb]ackend|\bkernel\b"),
    "compile_cache": re.compile(r"torch\.compile|compiled graph|[Cc]ompiling|compile range|compilation"),
    "cuda_graph": re.compile(r"CUDA ?[Gg]raph|cudagraph|[Gg]raph capturing"),
    "kv_memory": re.compile(r"KV cache|Maximum concurrency"),
    "init_time": re.compile(r"init engine|took [\d.]+ ?(?:s|secs|seconds)\b|Starting vLLM|startup complete"),
    "warning": re.compile(r"\bWARNING\b"),
    "error": re.compile(r"\bERROR\b|Traceback"),
}
_RE_STAMP = re.compile(r"^\d{10,} ")              # run_adv.pump_log's epoch-ms prefix
_RE_SOURCE = re.compile(r"\[[\w./-]+\.py:\d+\]")  # vLLM's "[file.py:NNN]"
_RE_PID = re.compile(r" pid=\d+\)")
_RE_CLOCK = re.compile(r"\b\d\d-\d\d \d\d:\d\d:\d\d ")
_RE_DECIMAL = re.compile(r"\d+\.\d+")
_RE_AVAILABLE_KV = re.compile(r"Available KV cache memory: ([\d.]+) GiB")


def _unstamped(text: str) -> list:
    return [_RE_STAMP.sub("", x, count=1) for x in text.splitlines()]


def startup_portion(text: str) -> list:
    """The server's lines (stamp removed) up to and including the first "Application startup complete"."""
    lines = _unstamped(text)
    for i, x in enumerate(lines):
        if STARTUP_READY in x:
            return lines[:i + 1]
    return lines


def _categorised(text: str) -> dict:
    out = {k: [] for k in STARTUP_PATTERNS}
    for line in startup_portion(text):
        msg = _RE_SOURCE.sub("", line)
        for key, pattern in STARTUP_PATTERNS.items():
            if pattern.search(line if key in ("warning", "error") else msg):
                out[key].append(line)
    return out


def startup_lines(text: str) -> dict:
    """The startup lines of a server log by category (non-default args, backends, compile cache, CUDA graphs, KV
    memory, init time, WARNING / ERROR), at most STARTUP_MAX_LINES each."""
    return {k: v[:STARTUP_MAX_LINES] for k, v in _categorised(text).items()}


def normalize_line(line: str) -> str:
    """A startup line without the epoch-ms stamp, pids, vLLM's clock and decimals (durations, GiB); integers such as
    CUDA graph sizes and block counts stay, so two boots compare on what they did, not when."""
    line = _RE_STAMP.sub("", line, count=1)
    return _RE_DECIMAL.sub("#", _RE_CLOCK.sub("", _RE_PID.sub(")", line)))


def startup_facts(text: str) -> dict:
    """Startup lines plus the structured values: run_adv's watched non-default args, coreb_smoke's log facts and
    the available KV cache memory (GiB)."""
    portion = "\n".join(startup_portion(text))
    cats = _categorised(text)
    kv = _RE_AVAILABLE_KV.search(portion)
    return {"lines": {k: v[:STARTUP_MAX_LINES] for k, v in cats.items()},
            "counts": {k: len(v) for k, v in cats.items()}, "nondefault": ra.parse_nondefault(portion),
            "log_facts": smoke.parse_server_log(portion), "available_kv_gib": float(kv.group(1)) if kv else None}


# ---------------------------------------------------------------- plan

def build_plan(pins: dict, *, pins_sha256: str) -> dict:
    """The one C0 plan (028): a single arm, so no randomisation; every bench has its boot and position."""
    sb, adv = pins["stage_b"], pins["advanced"]
    return {"schema_version": SCHEMA_VERSION, "stage": "C0", "decision": DECISION, "L": L, "mix": MIX, "C": C,
            "N": bp.num_prompts(C), "warmup": bp.num_warmups(C), "seed": bp.SEED, "boots": BOOTS,
            "benches_per_boot": BENCHES_PER_BOOT, "smoke": {"boots": 1, "benches": 1, "N": SMOKE_N, "C": C},
            "kv_cache_memory_bytes": sb["kv_cache_memory_bytes"], "expected_blocks": sb["num_gpu_blocks"],
            "block_size": sb["block_size"], "gpu_memory_utilization": sb["gpu_memory_utilization"],
            "gdn_prefill_backend": sb["gdn_prefill_backend"], "timeout_s": adv["timeouts_s"][f"{MIX}_{L}"],
            "cap_hours": CAP_HOURS, "boot_margin_s": BOOT_MARGIN_S, "temp_stop_c": TEMP_STOP_C,
            "max_failed_boots": MAX_FAILED_BOOTS, "pins_sha256": pins_sha256,
            "order": [{"boot": b, "bench": j, "tag": bench_tag(b, j)}
                      for b in range(1, BOOTS + 1) for j in range(1, BENCHES_PER_BOOT + 1)]}


def plan_problems(plan: dict, pins: dict, *, pins_sha256: str) -> list:
    want = build_plan(pins, pins_sha256=pins_sha256)
    return [f"plan {k} = {plan.get(k)!r} != {want.get(k)!r} (build_plan from configs/pins.toml; 028)"
            for k in sorted(set(want) | set(plan)) if plan.get(k) != want.get(k)]


def nondefault_problems(got: dict | None, *, kv_bytes: int) -> list:
    """The watched non-default args (run_adv.parse_nondefault) must be exactly MNBT 2048 and K (028): no observation
    flag, no Advanced condition key."""
    want = {"max_num_batched_tokens": bp.MAX_NUM_BATCHED_TOKENS, "kv_cache_memory_bytes": kv_bytes}
    if got is None:
        return ["no non-default args line in the server log (028)"]
    if got.keys() != want.keys() or any(type(got[k]) is not type(want[k]) or got[k] != want[k] for k in want):
        return [f"non-default args (watched keys) {got!r} != exactly {want!r} (028)"]
    return []


def bench_argv_for(pins: dict, *, N: int, result_dir, result_filename: str, vllm_bin: str) -> list:
    """core_b_plan.bench_argv at L 8192 uniform, C 8; the smoke changes only --num-prompts (028)."""
    argv = bp.bench_argv(pins, L=L, C=C, mix=MIX, result_dir=str(result_dir), result_filename=result_filename,
                         port=PORT, vllm_bin=vllm_bin)
    if N != bp.num_prompts(C):
        argv[argv.index("--num-prompts") + 1] = str(N)
    return argv


# ---------------------------------------------------------------- nvidia-smi at 1 Hz (028 量測)

_RE_SMI = re.compile(r"^(\d{10,}) (.*)$")


def smi_argv(interval_ms: int = SMI_INTERVAL_MS) -> list:
    return ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits",
            "-lms", str(interval_ms)]


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:  # [N/A], [Not Supported]
        return None


def parse_smi_line(line: str) -> dict | None:
    """'<epoch ms> <timestamp>, <pstate>, <sm>, <mem>, <power>, <temp>, <util>, <mem used>, <reasons>' (the
    LogReader stamp, then one CSV row) -> sample with t in epoch seconds; None for any other line."""
    m = _RE_SMI.match(line)
    parts = [x.strip() for x in m.group(2).split(",")] if m else []
    if len(parts) != len(SMI_FIELDS):
        return None
    reasons = parts[8]
    return {"t": int(m.group(1)) / 1000, "pstate": parts[1] if parts[1].startswith("P") else None,
            "sm_mhz": _num(parts[2]), "mem_mhz": _num(parts[3]), "power_w": _num(parts[4]), "temp_c": _num(parts[5]),
            "util_pct": _num(parts[6]), "mem_used_mib": _num(parts[7]),
            "event_reasons": int(reasons, 16) if reasons.startswith("0x") else None}


class _SmiLog:
    """The file run_adv.LogReader writes into: each stamped line is kept and parsed; every temperature goes to the
    sampler (staleness clock, the 88 C flag)."""

    def __init__(self, fh, sampler: "SmiSampler"):
        self.fh, self.sampler = fh, sampler

    def write(self, text: str) -> None:
        self.fh.write(text)
        s = parse_smi_line(text.rstrip("\n"))
        if s is not None and s["temp_c"] is not None:
            self.sampler.saw_temperature(s["temp_c"])

    def flush(self) -> None:
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class SmiSampler:
    """One `nvidia-smi -lms 1000` process per boot (its own session), its lines stamped by run_adv.LogReader. It is
    the only thermometer of 028 stop rule 6, so `halt` (what the runner and the bench watch) is set by a sample
    >= temp_stop_c (`hot`) and also when that telemetry is lost (`lost`, `lost_reason`): the process did not start,
    exited, or sent no temperature for stale_s (a watcher thread checks every poll_s until stop()). Never a crash:
    stop() returns the problems for the boot record."""

    def __init__(self, path, *, temp_stop_c: float = TEMP_STOP_C, popen: Callable = subprocess.Popen,
                 wall: Callable = time.time, clock: Callable = time.monotonic, interval_ms: int = SMI_INTERVAL_MS,
                 stale_s: float = SMI_STALE_S, poll_s: float = SMI_POLL_S):
        self.path, self.temp_stop_c, self.popen, self.wall, self.clock = Path(path), temp_stop_c, popen, wall, clock
        self.interval_ms, self.stale_s, self.poll_s = interval_ms, stale_s, poll_s
        self.hot, self.lost, self.halt, self._stopping = (threading.Event() for _ in range(4))
        self.proc = self.reader = self.watcher = self.lost_reason = None
        self.last_temp = clock()

    def flag_hot(self) -> None:
        self.hot.set()
        self.halt.set()

    def lose(self, reason: str) -> None:
        """The telemetry is gone before stop(): the first reason is kept."""
        if not self._stopping.is_set() and not self.lost.is_set():
            self.lost_reason = reason
            self.lost.set()
            self.halt.set()

    def saw_temperature(self, temp_c: float) -> None:
        self.last_temp = self.clock()
        if temp_c >= self.temp_stop_c:
            self.flag_hot()

    def start(self) -> "SmiSampler":
        log = _SmiLog(open(self.path, "w", encoding="utf-8"), self)
        try:
            self.proc = self.popen(smi_argv(self.interval_ms), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log.close()
            self.lose(f"the nvidia-smi sampler did not start: {type(exc).__name__}: {exc}")
            return self
        self.last_temp = self.clock()
        self.reader = ra.LogReader(self.proc.stdout, log, wall=self.wall)
        self.watcher = threading.Thread(target=self._watch, daemon=True)
        self.watcher.start()
        return self

    def _watch(self) -> None:
        while not self._stopping.wait(self.poll_s):
            code = self.proc.poll()
            if code is not None:
                self.lose(f"the nvidia-smi sampler exited (code {code})")
                return
            if self.clock() - self.last_temp > self.stale_s:
                self.lose(f"no temperature sample from nvidia-smi for more than {self.stale_s} s")
                return

    def stop(self) -> list:
        self._stopping.set()
        if self.watcher is not None:
            self.watcher.join(10)
        problems = [self.lost_reason] if self.lost_reason else []
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.reader is not None:
            if not self.reader.join(10):
                problems.append("the nvidia-smi reader is still open")
            if self.reader.error:
                problems.append(f"the nvidia-smi log could not be written: {self.reader.error}")
        return problems

    def samples(self) -> list:
        return [s for s in (parse_smi_line(x) for x in _read(self.path).splitlines()) if s]


def smi_summary(samples: list) -> dict:
    """What the smoke checks of the sampler (smoke_problems): samples, fields never reported (unsupported under WSL)
    and the peak temperature."""
    temps = [s["temp_c"] for s in samples if s.get("temp_c") is not None]
    missing = [k for k in SMI_KEYS if all(s.get(k) is None for s in samples)]
    return {"n_samples": len(samples), "fields_missing": missing, "temp_max_c": max(temps, default=None)}


def compute_apps_bounded(timeout_s: float = ra.NVIDIA_SMI_TIMEOUT_S) -> list:
    """The GPU's compute processes (028 rule 2); an nvidia-smi that fails or hangs raises (a stop before the boot)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
                         timeout=timeout_s).stdout
    return [x.strip() for x in out.splitlines() if x.strip() and not x.strip().startswith("No running")]


# ---------------------------------------------------------------- bench

def run_bench_watched(argv: list, log_path, env: dict, timeout_s: float, stop: threading.Event, *,
                      poll_s: float = 1.0) -> tuple:
    """`vllm bench serve` in its own session: (returncode, timed_out, stopped). As run_adv.default_run_bench, the
    client's process group gets SIGKILL at the timeout; it also gets it as soon as `stop` is set (88 C, 028 rule 6),
    which default_run_bench cannot do."""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                start_new_session=True)
        t0, timed_out, stopped = time.monotonic(), False, False
        try:
            while True:
                try:
                    proc.wait(timeout=poll_s)
                    break
                except subprocess.TimeoutExpired:
                    if stop.is_set():
                        stopped = True
                        break
                    if time.monotonic() - t0 >= timeout_s:
                        timed_out = True
                        break
        finally:
            if proc.poll() is None or smoke.group_alive(proc):
                smoke.signal_group(proc, signal.SIGKILL)
                proc.wait()
        return proc.returncode, timed_out, stopped


def result_problems(res: dict, *, N: int) -> list:
    """run_core_b.summarize_result's checks at the smoke's own N (Core's are fixed at num_prompts(C))."""
    reasons = []
    if res.get("completed") != N or res.get("failed"):
        reasons.append(f"completed={res.get('completed')} failed={res.get('failed')} (expected {N} / 0)")
    if sorted(res.get("input_lens") or []) != [L - bp.N_GEN] * N:
        reasons.append(f"input_lens do not match the {MIX} mix at L={L} ({len(res.get('input_lens') or [])} values)")
    if (res.get("output_lens") or []) != [bp.N_GEN] * N:
        reasons.append(f"output_lens not all {bp.N_GEN}")
    return reasons


def bench_fields(res: dict, *, N: int) -> dict:
    out = _rb().summarize_result(res, L=L, C=C, mix=MIX)
    if N != out["num_prompts"]:
        out.update(num_prompts=N, reasons=result_problems(res, N=N))
    return out


def derived_requests(res: dict) -> list:
    """derived/<bench>-requests.jsonl rows (numbers only): idx = the bench list index; [] if the arrays disagree."""
    cols = [res.get(k) or [] for k in ("input_lens", "output_lens", "ttfts", "latencies")]
    if any(len(c) != len(cols[0]) for c in cols):
        return []
    return [{"idx": i, "input_len": n, "output_len": o, "ttft_ms": None if t is None else t * 1000,
             "latency_ms": None if la is None else la * 1000} for i, (n, o, t, la) in enumerate(zip(*cols))]


def runtime_versions(pre: dict, pins: dict) -> dict:
    """Driver, vLLM and torch of the preflight against Phase 1 (028 rule 9: recorded, never a stop)."""
    gpus = (pre.get("gpu") or {}).get("nvidia") or [{}]
    vl = (pre.get("runtime") or {}).get("vllm") or {}
    got = {"driver": gpus[0].get("driver_version"), "vllm": vl.get("vllm"), "torch": vl.get("torch")}
    phase1 = {"driver": PHASE1_DRIVER, "vllm": pins["vllm"]["version"], "torch": PHASE1_TORCH}
    return {k: {"got": got[k], "phase1": phase1[k], "same": got[k] == phase1[k]} for k in got}


def _not_run(b: int, j: int, N: int, reason: str) -> dict:
    return {"boot": b, "bench": j, "position": j, "tag": bench_tag(b, j), "L": L, "mix": MIX, "C": C, "N": N,
            "ok": False, "reasons": [f"not run: {reason}"], "timeout": False, "stopped_hot": False,
            "stopped_telemetry": False, "not_run": True}


# ---------------------------------------------------------------- session

def run_session(plan: dict, *, pins: dict, pins_sha256: str, out_dir, session_id: str, is_smoke: bool, popen: Callable,
                signal_group: Callable, group_alive: Callable, http_get: Callable, run_bench: Callable,
                gpu_mem_used: Callable, compute_apps: Callable, start_smi: Callable, lock_held_check: Callable,
                t_start: float, gpu_lock_held: bool = False, cap_hours: float = CAP_HOURS,
                toolchain_check: Callable = bp.toolchain_problems, capture_env: Callable | None = None,
                base_env=None, vllm_bin: str = ".venv/bin/vllm", sample_interval: float | None = 1.0,
                now: Callable = time.monotonic, sleep: Callable = time.sleep, wall: Callable = time.time,
                clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(),
                startup_timeout_s: float = STARTUP_TIMEOUT_S, idle_wait_s: float = IDLE_WAIT_S) -> dict:
    """One C0 session (the smoke when is_smoke). t_start: now() when this process started (the cap's origin)."""
    base_env = dict(os.environ if base_env is None else base_env)
    if not gpu_lock_held:
        raise RunError("C0 must hold the GPU lock: flock -n /tmp/ai-portfolio-gpu.lock ... --gpu-lock-held "
                       "(028 stop rule 1; MACHINE_POLICY)")
    if not lock_held_check():
        raise RunError(f"--gpu-lock-held, but this process does not hold {smoke.GPU_LOCK} (no locked descriptor on "
                       f"it): run under flock -n {smoke.GPU_LOCK} ... (028; MACHINE_POLICY)")
    problems = plan_problems(plan, pins, pins_sha256=pins_sha256)
    problems += _rb().stage_b_problems(plan, pins, gdn_prefill_backend=plan.get("gdn_prefill_backend"),
                                       gpu_memory_utilization=plan.get("gpu_memory_utilization"))
    if cap_hours != plan.get("cap_hours"):
        problems.append(f"--cap-hours {cap_hours} != the plan's cap {plan.get('cap_hours')} h (028)")
    if problems:
        raise RunError("; ".join(problems))
    argv, env = bp.server_argv(pins, gdn_prefill_backend=plan["gdn_prefill_backend"],
                               gpu_memory_utilization=plan["gpu_memory_utilization"], port=PORT, vllm_bin=vllm_bin,
                               base_env=base_env, kv_cache_memory_bytes=plan["kv_cache_memory_bytes"])
    missing = toolchain_check(env)
    if missing:
        raise RunError("toolchain incomplete, not starting (014): " + "; ".join(missing))
    out = Path(out_dir)
    bad = out_problems(out)
    if bad:
        raise RunError(f"refusing {out}: {'; '.join(bad)} (028: C0 writes under results/capacity/, never into a "
                       "Phase 1 results directory; 027 §1)")
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten")
    shape = plan["smoke"] if is_smoke else {"boots": plan["boots"], "benches": plan["benches_per_boot"], "N": plan["N"]}
    n_boots, n_benches, N = shape["boots"], shape["benches"], shape["N"]
    cap_s = cap_hours * 3600
    raw, derived = out / "raw", out / "derived"
    raw.mkdir(parents=True)
    (out / "env").mkdir()
    derived.mkdir()
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=True) if capture_env else {}
    env_flags = [{"env": "preflight.json", "reason": why} for why in env_problems(pre, "wsl2-cuda")] if pre else []

    def elapsed() -> float:
        return now() - t_start

    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "formal": False,
               "stage": "C0-smoke" if is_smoke else "C0", "purpose": "smoke" if is_smoke else "exploratory",
               "decision": DECISION, "gpu_lock_held": True, "plan_sha256": sha256_json(plan),
               "pins_sha256": pins_sha256, "started_at": clock(), "cap_hours": cap_hours,
               "elapsed_at_start_s": round(elapsed(), 1), "boots_planned": n_boots, "benches_per_boot": n_benches,
               "N": N, "preflight": "env/preflight.json" if capture_env else None,
               "repo_commit": (pre.get("repo") or {}).get("commit"), "versions": runtime_versions(pre, pins),
               "server": {"argv": argv, "env": bp.recorded_env(env), "port": PORT, "vllm_bin": vllm_bin,
                          "gdn_prefill_backend": plan["gdn_prefill_backend"],
                          "gpu_memory_utilization": plan["gpu_memory_utilization"],
                          "kv_cache_memory_bytes": plan["kv_cache_memory_bytes"]},
               "inherited_unlisted": bp.inherited_unlisted(base_env),
               "inherited_otel": sorted(k for k in base_env if k.startswith("OTEL_")),
               "boots": {}, "failed_boots": 0, "stopped": None}
    rows = []
    base = f"http://127.0.0.1:{PORT}"
    fh = open(out / "rows.jsonl", "w")

    def emit(row: dict) -> None:
        rows.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    def snap(name: str) -> None:
        if capture_env:
            rec = capture_env(out_path=out / "env" / name, gpu_lock_held=True)
            env_flags.extend({"env": name, "reason": why} for why in env_problems(rec, "wsl2-cuda"))

    def check_log(proc, when: str) -> None:
        """A server log that could not be written is a harness failure: abort, never a result."""
        error = getattr(getattr(proc, "log_reader", None), "error", None)
        if error:
            raise RunError(f"the server log could not be written ({error}) {when}: a harness failure; session aborted")

    def halted(smi, where: str) -> Stop:
        """The stop for a set sampler flag: 88 C (028 rule 6), or the telemetry that rule reads was lost."""
        if smi.hot.is_set():
            return Stop("temperature", f"GPU temperature >= {plan['temp_stop_c']} C {where}: torn down and stopped "
                                       "(028 stop rule 6)")
        return Stop("telemetry", f"nvidia-smi telemetry lost {where} ({smi.lost_reason}): the 88 C stop (028 stop "
                                 "rule 6) cannot be applied; torn down and stopped")

    def pre_boot(b: int) -> None:
        tag = boot_tag(b)
        if elapsed() + plan["boot_margin_s"] > cap_s:
            raise Stop("cap", f"elapsed {elapsed():.0f} s + {plan['boot_margin_s']} s > cap {cap_s:.0f} s: no new "
                              f"boot (028 stop rule 3)")
        if not lock_held_check():
            raise Stop("lock", f"this process does not hold {smoke.GPU_LOCK} any more; stopping before {tag} (028)")
        t0 = now()
        try:
            mem = gpu_mem_used()
            while mem > IDLE_MIB and now() - t0 < idle_wait_s:
                sleep(2)
                mem = gpu_mem_used()
            apps = compute_apps() if mem <= IDLE_MIB else []
        except Exception as exc:  # noqa: BLE001 - nvidia-smi failed or hung: the GPU state is unknown
            raise Stop("gpu_check", f"nvidia-smi failed before {tag}: {type(exc).__name__}: {exc}") from exc
        if mem > IDLE_MIB:
            raise Stop("gpu_busy", f"memory.used {mem} MiB > {IDLE_MIB} after {idle_wait_s} s before {tag} "
                                   "(028 rule 2)")
        if apps:
            raise Stop("gpu_busy", f"other compute process(es) on the GPU before {tag}: {apps} (028 rule 2)")

    def checked_start(b: int, proc, rec: dict, smi) -> None:
        """The wait for /health, bounded by the startup timeout and by what is left of the cap (028 hard cap)."""
        tag = boot_tag(b)
        budget = min(startup_timeout_s, cap_s - elapsed())
        t0, ready = now(), False
        while now() - t0 < budget and proc.poll() is None and not smi.halt.is_set():
            if http_get(f"{base}/health")[0] == 200:
                ready = True
                break
            sleep(max(0, min(2, budget - (now() - t0))))  # the last poll never sleeps past the budget
        rec["startup_s"] = round(now() - t0, 1)
        facts = startup_facts(_read(raw / f"{tag}.server.log"))
        code, text = http_get(f"{base}/metrics") if ready else (0, "")
        cfg, blocks = ra.blocks_problems(text if code == 200 else "", expected_blocks=plan["expected_blocks"],
                                         block_size=plan["block_size"])
        if ready:
            reasons = []
        elif smi.hot.is_set():
            reasons = [f"GPU temperature >= {plan['temp_stop_c']} C during the start"]
        elif smi.halt.is_set():
            reasons = [f"nvidia-smi telemetry lost during the start ({smi.lost_reason})"]
        elif budget < startup_timeout_s and proc.poll() is None:
            rec["cut_at_cap"] = True
            reasons = [f"server not ready within {budget:.0f} s: the start was cut at the cap {cap_s:.0f} s "
                       "(028 hard cap)"]
        else:
            reasons = [f"server not ready within {startup_timeout_s} s (exit code {proc.poll()!r})"]
        reasons += blocks + nondefault_problems(facts["nondefault"], kv_bytes=plan["kv_cache_memory_bytes"])
        rec.update(startup=facts, cache_config=cfg, startup_ok=not reasons, reasons=reasons)

    def bench_once(b: int, j: int, proc, smi) -> tuple:
        """One bench and its row; (row, end the boot)."""
        tag, url = bench_tag(b, j), f"{base}/metrics"
        row = {"boot": b, "bench": j, "position": j, "tag": tag, "L": L, "mix": MIX, "C": C, "N": N, "ok": False,
               "reasons": [], "timeout": False, "stopped_hot": False, "stopped_telemetry": False, "not_run": False,
               "t_start": wall()}
        code0, before = http_get(url)
        before = before if code0 == 200 else ""
        (raw / f"{tag}.metrics.before.txt").write_text(before)
        cfg, blocks = ra.blocks_problems(before, expected_blocks=plan["expected_blocks"], block_size=plan["block_size"])
        bargv = bench_argv_for(pins, N=N, result_dir=raw, result_filename=f"{tag}.bench.json", vllm_bin=vllm_bin)
        row["argv"] = bargv
        sampler = _rb().MetricsSampler(http_get, url, sample_interval)
        sampler.start()
        try:
            rc, timed_out, stopped = run_bench(bargv, raw / f"{tag}.bench.log", {**base_env, **bp.bench_env()},
                                               plan["timeout_s"], smi.halt)
        finally:
            sampled = sampler.stop()
        check_log(proc, "during the bench")
        healthy = http_get(f"{base}/health")[0] == 200
        code1, after = http_get(url)
        after = after if code1 == 200 else ""
        row["t_end"] = wall()
        (raw / f"{tag}.metrics.after.txt").write_text(after)
        path = raw / f"{tag}.bench.json"
        res = json.loads(path.read_text()) if rc == 0 and not timed_out and not stopped and path.is_file() else None
        reasons = []
        if timed_out:
            row["timeout"] = True
            reasons.append(f"timeout: the bench exceeded {plan['timeout_s']} s; its process group was killed and the "
                           "server torn down (028 stop rule 4)")
        elif stopped and smi.hot.is_set():
            row["stopped_hot"] = True
            reasons.append(f"stopped: GPU temperature >= {plan['temp_stop_c']} C (028 stop rule 6)")
        elif stopped:
            row["stopped_telemetry"] = True
            reasons.append(f"stopped: nvidia-smi telemetry lost ({smi.lost_reason}; 028 stop rule 6)")
        elif res is None:
            reasons.append(f"vllm bench serve exit {rc}")
        if res is not None:
            fields = bench_fields(res, N=N)
            reasons += fields.pop("reasons")
            fields.pop("ok")
            row.update(fields, result=f"raw/{tag}.bench.json")
            reqs = derived_requests(res)
            if len(reqs) == N:
                (derived / f"{tag}-requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
                row["derived"] = f"derived/{tag}-requests.jsonl"
            else:
                reasons.append(f"--save-detailed arrays hold {len(reqs)} aligned requests, not {N}")
        if not healthy:
            reasons.append("server unhealthy after the bench (028 stop rule 5); the boot ends")
        reasons += blocks
        row.update(ok=res is not None and not reasons, reasons=reasons, healthy_after_bench=healthy,
                   num_gpu_blocks=(cfg or {}).get("num_gpu_blocks"), metrics_sampled=sampled,
                   scrape_delta=ra.scrape_deltas(before, after))
        return row, timed_out or stopped or not healthy

    def teardown(tag: str, proc, rec: dict, smi) -> None:
        """Server group down (SIGKILL escalation included) and its log joined, then the sampler stopped and its
        samples written; interrupts arriving meanwhile wait until all of that is done (run_adv.DeferredSignals)."""
        with ra.DeferredSignals() as deferred:
            problems = []
            try:
                if proc is not None:
                    problems += ra.shutdown_group(proc, deferred, signal_group=signal_group, group_alive=group_alive,
                                                  now=now, sleep=sleep)
                    reader = getattr(proc, "log_reader", None)
                    if reader is not None and not reader.join(ra.LOG_JOIN_S):
                        problems.append("server log reader still open after teardown")
            finally:
                smi_problems = smi.stop() or []
                samples = smi.samples()
                (derived / f"{tag}-smi.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
                rec.update(smi=f"derived/{tag}-smi.jsonl", smi_summary=smi_summary(samples), smi_problems=smi_problems)
                if problems:
                    rec["teardown_problems"] = problems
                    session.setdefault("teardown_problems", []).append({"boot": tag, "reasons": problems})

    def boot_once(b: int) -> Stop | None:
        tag = boot_tag(b)
        snap(f"{tag}.before.json")
        rec = session["boots"][tag] = {"argv": argv, "t_start": wall(), "elapsed_at_start_s": round(elapsed(), 1),
                                       "startup_ok": False, "reasons": []}
        smi = start_smi(raw / f"{tag}.smi.csv")
        proc, stop, done, why = None, None, 0, None
        try:
            proc = popen(argv, env, raw / f"{tag}.server.log")
            checked_start(b, proc, rec, smi)
            check_log(proc, "during the server start")
            if smi.halt.is_set():
                stop = halted(smi, f"during the start of {tag}")
            elif rec.get("cut_at_cap"):
                stop = Stop("cap", f"the start of {tag} was cut at the cap {cap_s:.0f} s: no bench (028 hard cap)")
            elif not rec["startup_ok"]:
                why = "the server start failed: " + "; ".join(rec["reasons"])
            else:
                for j in range(1, n_benches + 1):
                    if elapsed() + plan["timeout_s"] > cap_s:
                        stop = Stop("cap", f"elapsed {elapsed():.0f} s + the {plan['timeout_s']} s bench timeout > "
                                           f"cap {cap_s:.0f} s: no new bench (028 hard cap)")
                        break
                    row, end = bench_once(b, j, proc, smi)
                    emit(row)
                    done = j
                    if smi.halt.is_set():
                        stop = halted(smi, f"in {row['tag']}")
                        break
                    if end:
                        why = f"the boot ended after {row['tag']} ({'; '.join(row['reasons'])})"
                        break
        finally:
            teardown(tag, proc, rec, smi)
        for j in range(done + 1, n_benches + 1):
            emit(_not_run(b, j, N, stop.reason if stop else why))
        if stop is None and smi.halt.is_set():  # 88 C or lost telemetry while the server went down: still a stop
            stop = halted(smi, f"at the teardown of {tag}")
        check_log(proc, "at teardown")
        snap(f"{tag}.after.json")
        invalid = sum(not r["ok"] for r in rows if r["boot"] == b)
        rec.update(t_end=wall(), n_valid=n_benches - invalid, failed=not rec["startup_ok"] or invalid >= 2)
        return stop

    b = 0
    try:
        for b in range(1, n_boots + 1):
            pre_boot(b)
            stop = boot_once(b)
            if stop is not None:
                raise stop
            if session["boots"][boot_tag(b)]["failed"]:
                session["failed_boots"] += 1
                if session["failed_boots"] >= plan["max_failed_boots"]:
                    raise Stop("failed_boots", f"{session['failed_boots']} failed boots (028 stop rule 5)")
    except Stop as s:
        session["stopped"] = {"rule": s.rule, "reason": s.reason, "boot": b}
    except BaseException as exc:
        session["aborted"] = f"{type(exc).__name__}: {exc}"
        session["stopped"] = {"rule": "aborted", "reason": session["aborted"], "boot": b}
        raise
    finally:
        fh.close()
        session.update(finished_at=clock(), elapsed_s=round(elapsed(), 1), env_flags=env_flags,
                       env_valid=not env_flags,
                       outcomes={"boots": len(session["boots"]), "benches": len(rows),
                                 "valid": sum(bool(r["ok"]) for r in rows), "invalid": sum(not r["ok"] for r in rows),
                                 "timeout": sum(bool(r["timeout"]) for r in rows),
                                 "not_run": sum(bool(r["not_run"]) for r in rows),
                                 "failed_boots": session["failed_boots"]})
        write_raw_manifest(out)
        (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    return session


def smoke_problems(s: dict) -> list:
    """The smoke passed: no stop, its one bench valid, and its sampler reported temperatures without a problem (the
    88 C stop, 028 rule 6, reads nothing else)."""
    problems = []
    if s["stopped"] is not None or s["outcomes"]["valid"] != 1:
        problems.append(f"stopped: {s['stopped']}, valid benches: {s['outcomes']['valid']}")
    for tag, rec in s["boots"].items():
        summary = rec.get("smi_summary") or {}
        if not summary.get("n_samples") or "temp_c" in summary.get("fields_missing", ["temp_c"]):
            problems.append(f"{tag}: the nvidia-smi sampler reported no temperature "
                            f"({summary.get('n_samples', 0)} samples)")
        if rec.get("smi_problems"):
            problems.append(f"{tag}: nvidia-smi sampler problems {rec['smi_problems']}")
    return problems


def run_probe(plan: dict, *, smoke_first: bool, smoke_out, out, **kw) -> list:
    """028 「執行順序」: the smoke (when asked) and, only if it passed, the probe, all inside one cap. Both output
    directories are checked before the smoke starts."""
    bad = out_problems(out) + (out_problems(smoke_out) if smoke_first else [])
    if bad:
        raise RunError(f"refusing {'; '.join(bad)} (028: C0 writes under results/capacity/, never into a Phase 1 "
                       "results directory; 027 §1)")
    sessions = []
    if smoke_first:
        s = run_session(plan, is_smoke=True, out_dir=smoke_out, session_id=Path(smoke_out).name, **kw)
        sessions.append(s)
        problems = smoke_problems(s)
        if problems:
            raise RunError(f"smoke {Path(smoke_out).name} did not pass ({'; '.join(problems)}); the probe was not "
                           "started (028)")
    sessions.append(run_session(plan, is_smoke=False, out_dir=out, session_id=Path(out).name, **kw))
    return sessions


def main(argv: list | None = None) -> int:
    t_start = time.monotonic()  # flock starts this process once it holds the lock: the cap counts from here
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="write the deterministic C0 plan")
    p.add_argument("--out", required=True)
    for name in ("smoke", "run"):
        s = sub.add_parser(name)
        s.add_argument("--plan", help="a plan file; refused unless it equals build_plan's")
        s.add_argument("--out", help="default results/capacity/{smoke,explore}-c0-<YYYYMMDD>")
        s.add_argument("--gpu-lock-held", action="store_true")
        s.add_argument("--cap-hours", type=float, default=CAP_HOURS)
        s.add_argument("--vllm-bin", default=".venv/bin/vllm")
        if name == "run":
            s.add_argument("--smoke-first", action="store_true")
            s.add_argument("--smoke-out", help="default results/capacity/smoke-c0-<YYYYMMDD>")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    raw = (root / "configs" / "pins.toml").read_bytes()
    pins, pins_sha = tomllib.loads(raw.decode()), hashlib.sha256(raw).hexdigest()
    plan = build_plan(pins, pins_sha256=pins_sha)
    if args.cmd == "plan":
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
        print(json.dumps({k: plan[k] for k in ("boots", "benches_per_boot", "L", "C", "N", "cap_hours")}))
        return 0
    if args.plan:
        problems = plan_problems(json.loads(Path(args.plan).read_text()), pins, pins_sha256=pins_sha)
        if problems:
            print(f"run_c0_probe: {'; '.join(problems)}", file=sys.stderr)
            return 2
    date, capacity, rb = dt.datetime.now().strftime("%Y%m%d"), root / "results" / "capacity", _rb()
    kw = dict(pins=pins, pins_sha256=pins_sha, popen=ra.start_server, signal_group=smoke.signal_group,
              group_alive=smoke.group_alive, http_get=rb.default_http_get, run_bench=run_bench_watched,
              gpu_mem_used=ra.nvidia_mem_used_bounded, compute_apps=compute_apps_bounded,
              start_smi=lambda path: SmiSampler(path).start(), lock_held_check=ra.held_gpu_lock,
              gpu_lock_held=args.gpu_lock_held, cap_hours=args.cap_hours, t_start=t_start, vllm_bin=args.vllm_bin,
              capture_env=rb.make_vllm_env_capture(root / "scripts" / "env_capture.py"))
    previous = {sig: signal.signal(sig, smoke._exit_on_signal) for sig in (signal.SIGHUP, signal.SIGTERM)}
    try:
        if args.cmd == "smoke":
            out = Path(args.out or capacity / f"smoke-c0-{date}")
            sessions = [run_session(plan, is_smoke=True, out_dir=out, session_id=out.name, **kw)]
        else:
            out = Path(args.out or capacity / f"explore-c0-{date}")
            smoke_out = Path(args.smoke_out or capacity / f"smoke-c0-{date}")
            sessions = run_probe(plan, smoke_first=args.smoke_first, smoke_out=smoke_out, out=out, **kw)
    except RunError as exc:
        print(f"run_c0_probe: {exc}", file=sys.stderr)
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps({s["session_id"]: {"stopped": s["stopped"], **s["outcomes"]} for s in sessions}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
