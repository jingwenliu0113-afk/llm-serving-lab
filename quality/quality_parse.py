"""Parse llama-perplexity output of the pinned build and classify each quality
process as ok / oom / error (decision 005).

llama-perplexity returns exit 0 on nearly every error path, so a run is ok only
with positive markers, the expected chunk / task counts, and no pinned LOG_ERR
message. All formats are literal printf strings from the pinned
tools/perplexity/perplexity.cpp.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from core_a_parse import find_oom  # noqa: E402

# Every LOG_ERR in the pinned perplexity.cpp (plus the tool-level load / context errors).
LOG_ERR_PATTERNS = [re.compile(p) for p in (
    r"you need at least \d+ tokens", r"tokenizes to only \d+ tokens", r"stride is .* must be greater than zero",
    r"there are only \d+ tokens", r"failed to open .* for writing", r"Unexpected negative standard deviation",
    r"failed to decode the batch", r"number of lines in prompt not a multiple of 6",
    r"does not fit in the context window", r"llama_decode\(\) failed", r"failed to find comma separators",
    r"no _ in <", r"failed to parse answer", r": no tasks", r"failed to read task", r"found bad task",
    r"found empty answer", r"found \d+ malformed tasks", r"requires a higher -np", r"you must provide a name of a file",
    r"failed to open ", r"does not look like a file containing log-probabilities", r"has been computed with \d+",
    r"failed reading n_vocab", r"inconsistent vocabulary", r"failed reading evaluation tokens",
    r": failed to decode", r"failed reading log-probs", r"requires '--ctx-size' > 0",
    r"unable to load model", r"failed to create context",
)]

_NUM = r"(-?\d+(?:\.\d+)?)"
_KLD = {
    "kld_mean": rf"Mean\s+KLD:\s+{_NUM}\s*±\s*{_NUM}",
    "kld_max": rf"Maximum KLD:\s+{_NUM}",
    "kld_p999": rf"99\.9%\s+KLD:\s+{_NUM}",
    "kld_p99": rf"99\.0%\s+KLD:\s+{_NUM}",
    "kld_p95": rf"95\.0%\s+KLD:\s+{_NUM}",
    "kld_p90": rf"90\.0%\s+KLD:\s+{_NUM}",
    "kld_median": rf"Median\s+KLD:\s+{_NUM}",
    "kld_p10": rf"10\.0%\s+KLD:\s+{_NUM}",
    "kld_min": rf"Minimum KLD:\s+{_NUM}",
    "ppl_q": rf"Mean PPL\(Q\)\s+:\s+{_NUM}\s*±\s*{_NUM}",
    "ppl_base": rf"Mean PPL\(base\)\s+:\s+{_NUM}\s*±\s*{_NUM}",
    "ln_ppl_ratio": rf"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s+:\s+{_NUM}\s*±\s*{_NUM}",
    "ppl_ratio": rf"Mean PPL\(Q\)/PPL\(base\)\s+:\s+{_NUM}\s*±\s*{_NUM}",
    "ppl_diff": rf"Mean PPL\(Q\)-PPL\(base\)\s+:\s+{_NUM}\s*±\s*{_NUM}",
    "mean_dp_pct": rf"Mean\s+Δp:\s+{_NUM}\s*±\s*{_NUM} %",
    "rms_dp_pct": rf"RMS Δp\s+:\s+{_NUM}\s*±\s*{_NUM} %",
    "same_top_p_pct": rf"Same top p:\s+{_NUM}\s*±\s*{_NUM} %",
}
_RE_PPL_CHUNKS = re.compile(r"calculating perplexity over (\d+) chunks")
_RE_KLD_CHUNKS = re.compile(r"computing over (\d+) chunks")
_RE_FINAL_PPL = re.compile(rf"Final estimate: PPL = {_NUM} \+/- {_NUM}")
_RE_CHUNK_LINE = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_RE_MC_LINE = re.compile(r"^(\d+)\t(\d+\.\d+)$")
_RE_MC_FINAL = re.compile(rf"Final result: {_NUM} \+/- {_NUM}")
_RE_WG_LINE = re.compile(r"^(\d+)\t(\d+\.\d+)\t\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s*$")
_RE_WG_FINAL = re.compile(rf"Final Winogrande score\((\d+) tasks\): {_NUM} \+/- {_NUM}")


def parse_kld(text: str) -> dict:
    out = {}
    for key, pattern in _KLD.items():
        m = re.search(pattern, text)
        out[key] = float(m.group(1)) if m else None
        if m and m.lastindex and m.lastindex >= 2:
            out[f"{key}_se"] = float(m.group(2))
    m = _RE_KLD_CHUNKS.search(text)
    out["n_chunks"] = int(m.group(1)) if m else None
    return out


def parse_ppl(text: str, scored_per_chunk: int = 255) -> dict:
    m = _RE_PPL_CHUNKS.search(text)
    final = _RE_FINAL_PPL.search(text)
    cumulative = [float(c.group(3)) for c in map(_RE_CHUNK_LINE.match, text.splitlines()) if c]
    chunk_nll, prev = [], 0.0
    for i, av in enumerate(cumulative):
        total = av * (i + 1) * scored_per_chunk
        chunk_nll.append((total - prev) / scored_per_chunk)
        prev = total
    return {
        "n_chunks": int(m.group(1)) if m else None,
        "ppl": float(final.group(1)) if final else None,
        "ppl_se": float(final.group(2)) if final else None,
        "chunk_nll": chunk_nll,
    }


def _correct_from_cumulative(accs: list) -> list:
    correct, prev = [], 0
    for i, acc in enumerate(accs, start=1):
        n = round(acc * i / 100.0)
        correct.append(n - prev)
        prev = n
    return correct


def parse_mc(text: str) -> dict:
    accs = [float(m.group(2)) for m in map(_RE_MC_LINE.match, text.splitlines()) if m]
    final = _RE_MC_FINAL.search(text)
    return {
        "n_tasks": len(accs),
        "accuracy_pct": float(final.group(1)) if final else None,
        "accuracy_se_pct": float(final.group(2)) if final else None,
        "correct": _correct_from_cumulative(accs),
    }


def parse_winogrande(text: str) -> dict:
    rows = [m for m in map(_RE_WG_LINE.match, text.splitlines()) if m]
    final = _RE_WG_FINAL.search(text)
    return {
        "n_tasks": int(final.group(1)) if final else None,
        "accuracy_pct": float(final.group(2)) if final else None,
        "accuracy_se_pct": float(final.group(3)) if final else None,
        "correct": [int(r.group(5) == r.group(6)) for r in rows],
    }


def _log_err(lines: list):
    for line in lines:
        if any(p.search(line) for p in LOG_ERR_PATTERNS):
            return line.strip()
    return None


def classify(kind: str, exit_code, stdout: str, stderr: str, expected: dict) -> dict:
    text = stdout + "\n" + stderr
    lines = text.splitlines()
    problems, metrics = [], None

    if kind in ("kld", "kld_self"):
        metrics = parse_kld(text)
        if "====== KL divergence statistics ======" not in text or metrics["kld_mean"] is None:
            problems.append("no KL divergence statistics")
        if metrics["n_chunks"] != expected.get("n_chunks"):
            problems.append(f"n_chunks={metrics['n_chunks']} (expected {expected.get('n_chunks')})")
    elif kind in ("kld_base", "ppl"):
        metrics = parse_ppl(text)
        if metrics["ppl"] is None:
            problems.append("no 'Final estimate' line")
        if metrics["n_chunks"] != expected.get("n_chunks"):
            problems.append(f"n_chunks={metrics['n_chunks']} (expected {expected.get('n_chunks')})")
        if kind == "ppl" and len(metrics["chunk_nll"]) != expected.get("n_chunks"):
            problems.append(f"{len(metrics['chunk_nll'])} per-chunk lines (expected {expected.get('n_chunks')})")
    elif kind == "mc":
        metrics = parse_mc(text)
        if metrics["accuracy_pct"] is None:
            problems.append("no 'Final result' line")
        if metrics["n_tasks"] != expected.get("n_tasks"):
            problems.append(f"n_tasks={metrics['n_tasks']} (expected {expected.get('n_tasks')})")
    elif kind == "winogrande":
        metrics = parse_winogrande(text)
        if metrics["accuracy_pct"] is None:
            problems.append("no 'Final Winogrande score' line")
        if metrics["n_tasks"] != expected.get("n_tasks") or len(metrics["correct"]) != expected.get("n_tasks"):
            problems.append(f"n_tasks={metrics['n_tasks']} / {len(metrics['correct'])} lines "
                            f"(expected {expected.get('n_tasks')})")
    else:
        raise ValueError(f"unknown kind {kind!r}")

    oom = find_oom(lines)
    if oom and (exit_code != 0 or problems):
        return {"outcome": "oom", "reason": "allocation failure", "evidence": oom, "metrics": metrics}
    err = _log_err(lines)
    if err:
        return {"outcome": "error", "reason": f"llama-perplexity error: {err}", "evidence": err, "metrics": metrics}
    if exit_code != 0:
        return {"outcome": "error", "reason": f"exit {exit_code}", "evidence": None, "metrics": metrics}
    if problems:
        return {"outcome": "error", "reason": "; ".join(problems), "evidence": None, "metrics": metrics}
    return {"outcome": "ok", "reason": None, "evidence": None, "metrics": metrics}
