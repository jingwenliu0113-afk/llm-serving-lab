#!/usr/bin/env python3
"""GPU idle criterion and spill guard of decision 031 (accepted 2026-10-16), shared by the research runners (C0:
run_c0_probe; Stage C: run_capacity). It replaces the fixed "memory.used <= 1000 MiB within 60 s" of 028 stop rule 2
and 030 8.6 for Phase 2 only; Phase 1's frozen run_core_b / run_adv IDLE_MIB stay 1,000.

Idle (before every boot; the caller has already verified the shared GPU lock): a window of WINDOW_SAMPLES nvidia-smi
samples at WINDOW_INTERVAL_S (memory.used, utilization.gpu, pstate), then the compute-process listing. The window
passes when every sample has utilization.gpu <= UTIL_MAX, nvidia-smi lists no compute process, max - min memory.used
<= RANGE_MAX_MIB (nothing is loading or freeing VRAM) and the idle value (the window's largest memory.used) <= the
headroom bound V_total - F_PEAK_MIB - MARGIN_MIB (pins [stage_b].v_total_mib; 1,018 MiB today), so the measured vLLM
peak still fits on the card. wait_idle tries up to WINDOWS windows (about 90 s); the caller stops with "gpu_not_idle"
when none passes. A value nvidia-smi does not report fails the window (fail closed); an nvidia-smi that fails or hangs
raises to the caller.

Spill guard (during a boot): WDDM does not fail when VRAM runs out, it moves allocations to system memory, which
silently changes latency. read_shared_usage reads the Windows performance counters '\\GPU Adapter Memory(*)\\Dedicated
Usage' and '...\\Shared Usage' through powershell.exe from inside WSL (20 s timeout) and keeps the adapter with the
largest dedicated usage (the NVIDIA card; LUIDs change across Windows boots, so every reading decides afresh).
SpillGuard holds one boot's readings: pre-boot (after the idle window passed), after the server is ready, after each
bench. A reading SPILL_TOL_MIB or more above the pre-boot one is a spill (031 「64 MiB 以上」: 以上 includes 64); a
reading that cannot be taken (timeout, non-zero exit, unparsable output, another adapter than the pre-boot one) raises
SharedUsageError, which the research runners treat as lost telemetry (fail closed).

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import re
import subprocess
import time
from typing import Callable

DECISION = "031"
# 031 調查: vLLM with the Core B argv at K = 4 GiB (U 0.91) peaks at +15,157 MiB over its boot's idle memory.used:
# results/advanced/formal-20261011-adv, the K4 boots (B0 / T / G / TG), max over runs of mem_after_bench_mib minus the
# memory.used of env/<run>.before.json (G-2048-mixed-r2; +14,635 MiB at ready; K3: +14,182). Re-derive it when the
# argv or K changes.
F_PEAK_MIB = 15157
MARGIN_MIB = 128            # 031: headroom left on top of the measured peak
UTIL_MAX = 5                # 031: utilization.gpu <= 5 % in every sample (no compute load)
RANGE_MAX_MIB = 64          # 031: max - min memory.used over the window <= 64 MiB (a stable baseline)
SPILL_TOL_MIB = 64          # 031: a shared-usage reading 64 MiB or more above the boot's pre-boot one is a spill
WINDOWS = 3                 # 031: at most 3 windows (about 90 s), then "gpu_not_idle"
WINDOW_SAMPLES = 30         # 031: 30 samples at 1 Hz
WINDOW_INTERVAL_S = 1.0
NVIDIA_SMI_TIMEOUT_S = 15   # as run_adv.NVIDIA_SMI_TIMEOUT_S: a hung nvidia-smi raises instead of holding the lock
SHARED_USAGE_TIMEOUT_S = 20  # 031: a powershell.exe that does not answer within 20 s is an unreadable reading
SAMPLE_QUERY = ("memory.used", "utilization.gpu", "pstate")
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
COUNTERS = ("\\GPU Adapter Memory(*)\\Dedicated Usage", "\\GPU Adapter Memory(*)\\Shared Usage")
# one line per counter sample: <instance>|<counter, the last path element, lower case>|<MiB, invariant culture>
PS_SCRIPT = ("$ErrorActionPreference = 'Stop'; "
             "(Get-Counter -Counter " + ",".join("'" + c + "'" for c in COUNTERS) + ").CounterSamples | "
             "ForEach-Object { [string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}|{1}|{2:F1}', "
             "$_.InstanceName, ($_.Path -split '\\\\')[-1], $_.CookedValue / 1MB) }")
_RE_COUNTER = re.compile(r"^(?P<inst>[^|]+)\|(?P<counter>dedicated usage|shared usage)\|(?P<mib>\d+(?:\.\d+)?)$",
                         re.IGNORECASE)


class SharedUsageError(RuntimeError):
    """The Windows shared usage could not be read: the spill guard cannot be applied (031: fail closed)."""


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------- the headroom bound (031)

def headroom_bound(v_total_mib: int, *, f_peak_mib: int = F_PEAK_MIB, margin_mib: int = MARGIN_MIB) -> int:
    """V_total - F_peak - margin (031): the largest idle memory.used at which the measured vLLM peak still fits."""
    if not isinstance(v_total_mib, int) or isinstance(v_total_mib, bool):
        raise ValueError(f"v_total_mib {v_total_mib!r} is not an integer MiB (pins [stage_b].v_total_mib)")
    bound = v_total_mib - f_peak_mib - margin_mib
    if bound <= 0:
        raise ValueError(f"V_total {v_total_mib} MiB leaves no headroom for F_peak {f_peak_mib} + margin {margin_mib}")
    return bound


def criterion(v_total_mib: int) -> dict:
    """Every number of the 031 rule, for session.json."""
    return {"decision": DECISION, "v_total_mib": v_total_mib, "f_peak_mib": F_PEAK_MIB, "margin_mib": MARGIN_MIB,
            "bound_mib": headroom_bound(v_total_mib), "util_max_pct": UTIL_MAX, "range_max_mib": RANGE_MAX_MIB,
            "windows": WINDOWS, "window_samples": WINDOW_SAMPLES, "interval_s": WINDOW_INTERVAL_S,
            "spill_tol_mib": SPILL_TOL_MIB}


# ---------------------------------------------------------------- nvidia-smi

def _int_or_none(text: str):
    try:
        return int(float(text))
    except ValueError:  # [N/A], [Not Supported]
        return None


def parse_gpu_sample(text: str) -> dict:
    """One `nvidia-smi --query-gpu=memory.used,utilization.gpu,pstate --format=csv,noheader,nounits` row ->
    {"mem_used_mib", "util_pct", "pstate"} (None for a value nvidia-smi does not report); ValueError for anything but
    exactly one GPU row with a memory value."""
    lines = [x for x in text.splitlines() if x.strip()]
    parts = [x.strip() for x in lines[0].split(",")] if len(lines) == 1 else []
    if len(parts) != len(SAMPLE_QUERY):
        raise ValueError(f"not one nvidia-smi row of {', '.join(SAMPLE_QUERY)}: {text[:200]!r}")
    mem = _int_or_none(parts[0])
    if mem is None:
        raise ValueError(f"nvidia-smi reports no memory.used: {text[:200]!r}")
    return {"mem_used_mib": mem, "util_pct": _int_or_none(parts[1]),
            "pstate": parts[2] if parts[2].startswith("P") else None}


def parse_compute_apps(text: str) -> list:
    """`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits` -> its rows."""
    return [x.strip() for x in text.splitlines() if x.strip() and not x.strip().startswith("No running")]


def nvidia_sample(*, run: Callable | None = None, timeout_s: float = NVIDIA_SMI_TIMEOUT_S) -> dict:
    """One sample of the idle window; an nvidia-smi that fails or hangs raises."""
    out = (run or subprocess.run)(["nvidia-smi", f"--query-gpu={','.join(SAMPLE_QUERY)}",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
                                  timeout=timeout_s).stdout
    return parse_gpu_sample(out)


def nvidia_compute_apps(*, run: Callable | None = None, timeout_s: float = NVIDIA_SMI_TIMEOUT_S) -> list:
    """The GPU's compute processes; an nvidia-smi that fails or hangs raises."""
    out = (run or subprocess.run)(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
                                  timeout=timeout_s).stdout
    return parse_compute_apps(out)


# ---------------------------------------------------------------- the idle window (031)

def idle_window(sample: Callable, n: int = WINDOW_SAMPLES, interval_s: float = WINDOW_INTERVAL_S, *,
                now: Callable = time.monotonic, sleep: Callable = time.sleep) -> list:
    """n samples, the i-th at t0 + i * interval_s (a slow sample delays only itself, never the schedule)."""
    t0, out = now(), []
    for i in range(n):
        if i:
            sleep(max(0.0, t0 + i * interval_s - now()))
        out.append(sample())
    return out


def idle_problems(samples: list, compute_apps: list, *, bound_mib: int) -> list:
    """Why a window is not idle under 031 ([] = idle): utilization.gpu > UTIL_MAX in a sample (or not reported), a
    compute process, a memory.used range > RANGE_MAX_MIB, or the idle value (the largest sample) above the bound."""
    if not samples:
        return ["no nvidia-smi sample in the idle window (031)"]
    mem = [s.get("mem_used_mib") for s in samples]
    util = [s.get("util_pct") for s in samples]
    problems = []
    if not all(_num(u) for u in util):
        problems.append(f"utilization.gpu not reported in {sum(not _num(u) for u in util)} of {len(util)} samples: "
                        "a compute load cannot be ruled out (031)")
    elif max(util) > UTIL_MAX:
        problems.append(f"utilization.gpu up to {max(util)} % (> {UTIL_MAX} % in "
                        f"{sum(u > UTIL_MAX for u in util)} of {len(util)} samples): a compute load (031)")
    if compute_apps:
        problems.append(f"compute process(es) on the GPU: {list(compute_apps)} (031)")
    nums = [m for m in mem if _num(m)]
    if len(nums) != len(mem):
        problems.append(f"memory.used not reported in {len(mem) - len(nums)} of {len(mem)} samples (031)")
    if nums:
        spread = max(nums) - min(nums)
        if spread > RANGE_MAX_MIB:
            problems.append(f"memory.used range {spread} MiB ({min(nums)}-{max(nums)}) > {RANGE_MAX_MIB} MiB over the "
                            "window: VRAM is being loaded or freed (031)")
        if max(nums) > bound_mib:
            problems.append(f"idle memory.used {max(nums)} MiB > the headroom bound {bound_mib} MiB = V_total - "
                            f"F_peak {F_PEAK_MIB} - margin {MARGIN_MIB} (031): the vLLM peak would not fit")
    return problems


def window_report(samples: list, compute_apps: list, *, bound_mib: int, attempt: int) -> dict:
    """One window: its verdict, problems, summary and the samples (memory.used, utilization.gpu) for session.json."""
    problems = idle_problems(samples, compute_apps, bound_mib=bound_mib)
    nums = [s["mem_used_mib"] for s in samples if _num(s.get("mem_used_mib"))]
    utils = [s["util_pct"] for s in samples if _num(s.get("util_pct"))]
    return {"attempt": attempt, "ok": not problems, "problems": problems, "n": len(samples), "bound_mib": bound_mib,
            "idle_mib": max(nums) if nums else None, "mem_min_mib": min(nums) if nums else None,
            "mem_max_mib": max(nums) if nums else None, "mem_range_mib": max(nums) - min(nums) if nums else None,
            "util_max_pct": max(utils) if utils else None,
            "pstates": sorted({s.get("pstate") for s in samples if s.get("pstate")}),
            "compute_apps": list(compute_apps), "mem_used_mib": [s.get("mem_used_mib") for s in samples],
            "util_pct": [s.get("util_pct") for s in samples]}


def wait_idle(sample: Callable, compute_apps: Callable, *, bound_mib: int, windows: int = WINDOWS,
              n: int = WINDOW_SAMPLES, interval_s: float = WINDOW_INTERVAL_S, now: Callable = time.monotonic,
              sleep: Callable = time.sleep) -> tuple:
    """Up to `windows` idle windows, one interval apart: (True, the passing window's report) or (False, the last
    report); each report carries the earlier windows' problems in "earlier". nvidia-smi failures raise."""
    earlier, report = [], None
    for attempt in range(1, windows + 1):
        if attempt > 1:
            sleep(interval_s)
        samples = idle_window(sample, n, interval_s, now=now, sleep=sleep)
        report = window_report(samples, compute_apps(), bound_mib=bound_mib, attempt=attempt)
        report["earlier"] = list(earlier)
        if report["ok"]:
            return True, report
        earlier.append(report["problems"])
    return False, report


# ---------------------------------------------------------------- the spill guard (031)

def parse_shared_usage(text: str) -> dict:
    """powershell output ('<instance>|dedicated usage|<MiB>' and '<instance>|shared usage|<MiB>' lines, any order and
    case) -> {"luid", "dedicated_mib", "shared_mib"} of the adapter with the largest dedicated usage; SharedUsageError
    when no adapter has both values."""
    adapters = {}
    for line in text.splitlines():
        m = _RE_COUNTER.match(line.strip())
        if m:
            key = "dedicated_mib" if m.group("counter").lower().startswith("dedicated") else "shared_mib"
            adapters.setdefault(m.group("inst"), {})[key] = float(m.group("mib"))
    complete = {k: v for k, v in adapters.items() if set(v) == {"dedicated_mib", "shared_mib"}}
    if not complete:
        raise SharedUsageError(f"no GPU adapter with both dedicated and shared usage in the powershell output "
                               f"({text.strip()[:200]!r})")
    luid = max(sorted(complete), key=lambda k: complete[k]["dedicated_mib"])
    return {"luid": luid, **complete[luid]}


def read_shared_usage(*, run: Callable | None = None, timeout_s: float = SHARED_USAGE_TIMEOUT_S) -> dict:
    """The NVIDIA adapter's Windows shared usage, read from WSL through powershell.exe (031); any failure (no
    interop, a timeout, a non-zero exit, unparsable output) raises SharedUsageError. powershell.exe writes stderr in
    the Windows OEM code page (cp950 on a zh-TW Windows): undecodable bytes are replaced, never an error, since
    stdout is ASCII by construction (InvariantCulture numbers, luid_* instance names)."""
    try:
        proc = (run or subprocess.run)([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", PS_SCRIPT],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=timeout_s, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:   # ValueError: a decode error, whatever its cause
        raise SharedUsageError(f"powershell.exe failed: {type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        raise SharedUsageError(f"powershell.exe exited {proc.returncode}: {(proc.stderr or '').strip()[:200]!r}")
    return parse_shared_usage(proc.stdout or "")


def spill(pre_mib: float, reading_mib: float, tol_mib: float = SPILL_TOL_MIB) -> bool:
    """031 「比 boot 前多出 64 MiB 以上即為 spill」: a reading tol_mib or more above the boot's pre-boot one (以上
    includes the bound) is a spill."""
    return reading_mib >= pre_mib + tol_mib


class SpillGuard:
    """One boot's spill guard (031): start() takes the pre-boot reading, check(when) a later one ("ready", or a bench
    tag) and says whether it spilled. A reading that cannot be taken, or one of another adapter than the pre-boot
    reading's, raises SharedUsageError (fail closed) and is recorded with its error."""

    def __init__(self, read: Callable = read_shared_usage, *, tol_mib: float = SPILL_TOL_MIB):
        self.read, self.tol_mib = read, tol_mib
        self.pre, self.readings, self.error = None, [], None

    def _read(self, when: str) -> dict:
        try:
            got = self.read()
        except Exception as exc:  # noqa: BLE001 - whatever failed, the reading is unknown: fail closed
            self.error = f"{when}: {type(exc).__name__}: {exc}"
            raise SharedUsageError(f"shared usage unreadable at {when} ({type(exc).__name__}: {exc})") from exc
        if self.pre is not None and got.get("luid") != self.pre.get("luid"):
            self.error = f"{when}: adapter {got.get('luid')!r} != pre-boot {self.pre.get('luid')!r}"
            raise SharedUsageError(f"shared usage at {when} is of adapter {got.get('luid')!r}, not the pre-boot "
                                   f"reading's {self.pre.get('luid')!r}: not comparable")
        return got

    def start(self) -> dict:
        self.pre = self._read("pre-boot")
        return self.pre

    def check(self, when: str) -> dict:
        if self.pre is None:
            raise SharedUsageError(f"no pre-boot shared usage reading before {when}")
        got = self._read(when)
        entry = {"when": when, **got, "delta_mib": round(got["shared_mib"] - self.pre["shared_mib"], 1),
                 "spilled": spill(self.pre["shared_mib"], got["shared_mib"], self.tol_mib)}
        self.readings.append(entry)
        return entry

    def spilled_at(self):
        return next((e["when"] for e in self.readings if e["spilled"]), None)

    def record(self) -> dict:
        """The boot's readings for session.json."""
        return {"tol_mib": self.tol_mib, "pre": self.pre, "readings": list(self.readings),
                "spilled": self.spilled_at(), "error": self.error}

    def row_record(self) -> dict:
        """What a bench row carries (030 8.6 (g)): the pre-boot value and every later reading of its boot so far."""
        values = [e["shared_mib"] for e in self.readings]
        return {"pre_mib": self.pre["shared_mib"], "readings_mib": values,
                "spilled": any(spill(self.pre["shared_mib"], v, self.tol_mib) for v in values),
                "tol_mib": self.tol_mib}
