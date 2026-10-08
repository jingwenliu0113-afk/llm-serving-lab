"""Parse llama-bench / llama-completion output of the pinned build and classify
each Core A process as ok / oom / error (PREREGISTRATION §3, decision 004).

Stdlib only and Python 3.10 compatible: it also runs inside WSL2 Ubuntu-22.04.
Every pattern below is a literal message from the pinned source (b11146):
ggml-cuda.cu, ggml-metal-*.m/.cpp, ggml-alloc.c, ggml-backend.cpp,
llama-model.cpp, llama-context.cpp, llama-kv-cache.cpp, common.cpp,
tools/completion, tools/llama-bench.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

# Allocation failures = OOM. Deliberately excludes the DEBUG-level
# "failed to allocate ... of pinned memory" (host pinned memory, not VRAM).
OOM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"cudaMalloc failed: out of memory",
        r"CUDA error: out of memory",
        r"failed to allocate \S+ buffer of size",
        r"failed to allocate buffer(,| of size)",
        r"failed to allocate Metal buffer .*out of memory",
        r"unable to allocate \S+ buffer",
        r"failed to allocate buffer for kv cache",
        r"failed to allocate compute (pp |tg )?buffers",
        r"failed to allocate output buffer",
        r"kIOGPUCommandBufferCallbackErrorOutOfMemory",
        r"Insufficient Memory",
    )
]

# Pipeline order: the earliest stage with a marker is where the run failed.
STAGE_MARKERS = [
    ("load", re.compile(r"error loading model|failed to load model", re.IGNORECASE)),
    ("context", re.compile(r"failed to create context|unable to create context|failed to initialize the context",
                           re.IGNORECASE)),
    ("eval", re.compile(r"failed to eval|failed to decode|failed to run (prompt|gen|depth)|command buffer \d+ failed",
                        re.IGNORECASE)),
]

# Settings every Core A llama-bench row must report (PREREGISTRATION §4).
BENCH_FIXED = {
    "n_batch": 2048, "n_ubatch": 512, "type_k": "f16", "type_v": "f16",
    "n_gpu_layers": 999, "flash_attn": 1, "load_mode": "mmap",
}

_RE_N_CTX = re.compile(r"llama_context: n_ctx\s+=\s+(\d+)")
_RE_FA = re.compile(r"llama_context: flash_attn\s+=\s+(\w+)")
_RE_PROMPT = re.compile(r"prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens")
_RE_EVAL = re.compile(r":\s+eval time =\s*[\d.]+ ms /\s*(\d+) runs")


class ParsedLog(NamedTuple):
    logs: list  # [{"level": str, "msg": str}]
    events: dict  # event type -> [data, ...]
    raw_lines: int  # non-JSON lines (generated text)


def parse_log_jsonl(text: str) -> ParsedLog:
    """Split `--log-jsonl` stdout into log entries, typed events and raw lines."""
    logs, events, raw = [], {}, 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            raw += 1
            continue
        if not isinstance(obj, dict):
            raw += 1
        elif obj.get("type") == "log":
            logs.append({"level": obj.get("level"), "msg": str(obj.get("msg", "")).rstrip("\n")})
        else:
            events.setdefault(obj.get("type"), []).append(obj.get("data"))
    return ParsedLog(logs, events, raw)


def parse_bench_jsonl(text: str) -> list:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def check_bench_row(row: dict, expect: dict) -> list:
    """Return protocol violations (empty list = row is valid)."""
    violations = []
    for key, value in BENCH_FIXED.items():
        if row.get(key) != value:
            violations.append(f"{key}={row.get(key)!r} (expected {value!r})")
    for key in ("n_prompt", "n_gen", "n_depth", "n_threads", "poll", "build_commit"):
        if key in expect and row.get(key) != expect[key]:
            violations.append(f"{key}={row.get(key)!r} (expected {expect[key]!r})")
    if "reps" in expect and len(row.get("samples_ns") or []) != expect["reps"]:
        violations.append(f"samples_ns has {len(row.get('samples_ns') or [])} samples (expected {expect['reps']})")
    return violations


def bench_metrics(row: dict) -> dict:
    return {k: row.get(k) for k in ("avg_ns", "stddev_ns", "avg_ts", "stddev_ts", "samples_ns")}


def _last_int(pattern: re.Pattern, msgs: list):
    for msg in reversed(msgs):
        m = pattern.search(msg)
        if m:
            return int(m.group(1))
    return None


def completion_summary(parsed: ParsedLog) -> dict:
    msgs = [entry["msg"] for entry in parsed.logs]
    fa = None
    for msg in msgs:
        m = _RE_FA.search(msg)
        if m:
            fa = m.group(1)
    memory = None
    if parsed.events.get("fit_memory_breakdown"):
        data = parsed.events["fit_memory_breakdown"][-1] or {}
        rows = data.get("rows") or []
        strip = lambda r, keys: {k: v for k, v in r.items() if k not in keys}  # noqa: E731
        host = next((r for r in rows if r.get("kind") == "host"), None)
        memory = {
            "unit": data.get("unit"),
            "devices": [strip(r, ("kind",)) for r in rows if r.get("kind") == "device"],
            "host": strip(host, ("kind", "name")) if host else None,
            "other": [strip(r, ("kind",)) for r in rows if r.get("kind") == "buffer_type"],
        }
    return {
        "n_ctx": _last_int(_RE_N_CTX, msgs),
        "flash_attn": fa,
        "prompt_tokens": _last_int(_RE_PROMPT, msgs),
        "eval_runs": _last_int(_RE_EVAL, msgs),
        "memory": memory,
    }


def find_oom(lines: list):
    for line in lines:
        if any(p.search(line) for p in OOM_PATTERNS):
            return line.strip()
    return None


def find_stage(lines: list):
    for stage, pattern in STAGE_MARKERS:
        if any(pattern.search(line) for line in lines):
            return stage
    return None


def _result(outcome, stage=None, reason=None, evidence=None, **extra):
    return {"outcome": outcome, "stage": stage, "reason": reason, "evidence": evidence, **extra}


def _exit_reason(exit_code) -> str:
    return "timeout" if exit_code is None else f"exit {exit_code}"


def classify_completion(exit_code, parsed: ParsedLog, stderr: str, expected_prompt_tokens: int) -> dict:
    lines = [entry["msg"] for entry in parsed.logs] + stderr.splitlines()
    summary = completion_summary(parsed)
    oom, stage = find_oom(lines), find_stage(lines)

    problems = []
    if summary["prompt_tokens"] != expected_prompt_tokens:
        problems.append(f"prompt_tokens={summary['prompt_tokens']} (expected {expected_prompt_tokens})")
    if not summary["eval_runs"]:
        problems.append("no eval timing line")
    if summary["memory"] is None:
        problems.append("no memory breakdown event")

    if oom and (exit_code != 0 or problems):
        return _result("oom", stage or "unknown", "allocation failure", oom, summary=summary)
    if exit_code != 0:
        return _result("error", stage, _exit_reason(exit_code), next(iter(lines[-1:]), None), summary=summary)
    if oom:
        return _result("error", stage, "allocation failure logged in a completed run (inspect)", oom, summary=summary)
    if problems:
        return _result("error", stage, "; ".join(problems), None, summary=summary)
    return _result("ok", summary=summary)


def classify_bench(exit_code, stdout: str, stderr: str, expect: dict) -> dict:
    lines = stderr.splitlines()
    rows = parse_bench_jsonl(stdout)
    oom, stage = find_oom(lines), find_stage(lines)

    if oom and (exit_code != 0 or len(rows) != 1):
        return _result("oom", stage or "unknown", "allocation failure", oom, metrics=None)
    if exit_code != 0:
        return _result("error", stage, _exit_reason(exit_code), next(iter(lines[-1:]), None), metrics=None)
    if len(rows) != 1:
        return _result("error", stage, f"expected exactly 1 result row, got {len(rows)}", None, metrics=None)
    violations = check_bench_row(rows[0], expect)
    if violations:
        return _result("error", stage, "; ".join(violations), None, metrics=bench_metrics(rows[0]))
    return _result("ok", metrics=bench_metrics(rows[0]))
