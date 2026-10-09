#!/usr/bin/env python3
"""Deployment decision table (SPEC; rules in docs/decisions/021).

Reads only valid formal sessions and their frozen analyses, re-verifies every analysis
with the frozen code (fail closed), then derives:
- Core A, per platform and context level: each configuration's status, the Pareto front of
  the candidates over quality / decode / TTFT / memory (source SCCs of the dominance graph),
  and four single-primary-axis workload picks (A-Q / A-D / A-P / A-M). "Beyond noise" is the
  frozen §2 rule (core_a_analysis._beyond_noise); quality relations come only from the H-A5
  Holm results; a pair without a test is "unmeasured" and never acts as a tie.
- Core B (wsl2-cuda only): the curve per L, B-T at the frozen C_knee, B-L facts without a
  concurrency pick (an SLO pick is Advanced), B-X copied from H-B4.
Verdicts are copied verbatim with unit counts and never feed a pick. The two platforms'
Core A values never share a row (SPEC: no Mac vs PC ranking); Core A and Core B never share
a table.

Usage:
    uv run python benchmarks/decision_table.py

No arguments select or skip sessions (021 R1). Writes results/decision_table/decision_table.json
and .md; exit 2 (nothing written) when an input class has two or more valid formal sessions or
a stored analysis does not match its recomputation. Stdlib only, Python >= 3.11 (tomllib).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "quality"))

import core_a_analysis as an  # noqa: E402
import core_a_plan as ap  # noqa: E402
import core_a_report as ar  # noqa: E402
import core_b_analysis as ba  # noqa: E402
import core_b_plan as bp  # noqa: E402
import core_b_report as br  # noqa: E402
import quality_analysis as qa  # noqa: E402

SCHEMA_VERSION = 1
RULES = "docs/decisions/021"
PLATFORMS = ("macos-metal", "wsl2-cuda")
AXES = ("quality", "decode", "ttft", "memory")
HIGHER_IS_BETTER = {"decode": True, "ttft": False, "memory": False}
WORKLOADS = {  # 021 R11: one primary axis per SPEC Core A metric, then quality, then the SPEC order
    "A-Q": ("quality", "ttft", "decode", "memory"),
    "A-D": ("decode", "quality", "ttft", "memory"),
    "A-P": ("ttft", "quality", "decode", "memory"),
    "A-M": ("memory", "quality", "ttft", "decode"),
}
WORKLOAD_TEXT = {
    "A-Q": "quality first (one request at a time)",
    "A-D": "generation speed first (long answers, chat)",
    "A-P": "first token first (long documents, RAG, summaries)",
    "A-M": "memory first (leave room for other programs)",
}
FROZEN_MODULES = ("benchmarks/core_a_analysis.py", "benchmarks/core_a_plan.py", "benchmarks/core_a_report.py",
                  "benchmarks/core_b_analysis.py", "benchmarks/core_b_plan.py", "benchmarks/core_b_report.py",
                  "quality/quality_analysis.py")
HA5_A, HA5_B = (f"{s}-{q}" for s, q in qa.HA5_PAIR)  # 9B-Q4_K_M, 4B-Q8_0
PAIRED_TEST = f"paired dNLL {HA5_A} - {HA5_B} < 0"
TESTABLE = {"H-A2": ("tested",), "H-A3": ("tested",), "H-A4a": ("tested",), "H-A4b": ("tested",),
            "H-B1": ("supported", "refuted", "inconclusive"), "H-B2": ("tested",), "H-B3": ("tested",),
            "H-B4": ("tested",)}
B_METRICS = ("output_throughput", "request_throughput", "total_token_throughput",
             "p50_ttft_ms", "p95_ttft_ms", "p50_tpot_ms", "p95_tpot_ms")
TIE_TEXT = "no resolved quality advantage (no equivalence claimed)"
BL_STATEMENT = "choosing C for a latency target is Advanced (fixed-SLO max concurrency); not done in Core"
BX_STATEMENT = ("measured only at C = uniform C_knee; the mechanism is a conjecture (007); "
                "no remedy evaluated (Advanced)")
CORE_B_QUALITY = "not measured (W4A16; 003/008) -- Core A Q4_K_M quality does not apply"
SCOPE = [
    "one request at a time -> the Core A section of your platform",
    "concurrency > 1 on the RTX -> the Core B section",
    "concurrency > 1 on the Mac -> not measured (SPEC: Core B only on NVIDIA)",
    "Core A (llama.cpp GGUF) and Core B (vLLM W4A16) numbers never share a row or table: different engine, "
    "precision and format; no runtime comparison (SPEC, 003)",
]
CAVEATS = {
    "NO_RANKING": "no Mac vs PC ranking (SPEC): each platform's values are only compared within that platform",
    "CORE_A_B_SEPARATE": "Core A and Core B are not comparable (different engine, precision, format)",
    "TIE_NOT_EQUIVALENCE": "tie = not resolved / within noise; no equivalence is claimed",
    "N3_NOISE_RULE": "beyond noise = |x - y| > 2 sqrt(SE_x^2 + SE_y^2) with n = 3 processes (§2); "
                     "lenient at df = 2, applied as frozen",
    "QUALITY_CTX_512": "quality measured at -c 512 (005); long-context quality not measured",
    "TTFT_ENGINE_ONLY": "TTFT is llama-bench engine time: no tokenize / HTTP / sampling (§3)",
    "NO_IMATRIX_Q4": "Q4_K_M is a plain k-quant without imatrix (004)",
    "QUALITY_PLATFORM_MAC": "quality measured on wsl2-cuda (020), applied as a model x quant property (SPEC); "
                            "not measured on Metal",
    "QUALITY_PLATFORM_RTX": "quality measured on this platform (wsl2-cuda, 020)",
    "CROSS_SIZE_UNMEASURED": "4B vs 9B quality is unmeasured except the paired H-A5(a) test (005)",
    "CONTAMINATION": "the 4B vs 9B WikiText comparison may partly reflect memorisation (005)",
    "MAC_POWER_CONDITION": "Mac session conditions: AC with the Apple 70W USB-C Power Adapter, one powermode per "
                           "session (019)",
    "SPILL_EXCLUDED": "a configuration that spills to system memory is not a fit (004) and is never a candidate",
    "PROVISIONAL": "a configuration at this level is not measured / error / incomplete and could change the pick",
    "NOISE_CYCLE": "dominance within noise forms a cycle; all its members are kept",
    "CONTRARY_TO_PREDICTION": "a Holm-resolved quality relation runs against the H-A5 prediction",
    "KV_USAGE_INCLUDES_GDN": "vLLM KV cache usage % includes the GDN state blocks (§3)",
    "KNEE_POINT_ESTIMATE": "C_knee is a point estimate on the restart means (frozen definition, no noise rule)",
    "KNEE_AT_GRID_TOP": "C_knee is the largest grid C",
    "H_B1_UNREFUTABLE": "the grid does not reach 4 x C_knee, so H-B1 cannot be refuted at this L",
    "CORE_B_FIXED_DEPLOYMENT": "Core B is one model / precision / server config; only client concurrency varies",
    "MIXED_AT_KNEE_ONLY": "the mixed request mix is measured only at the uniform C_knee (007)",
    "NO_TESTABLE_UNITS": "the verdict has no testable unit behind it",
    "DOWNSTREAM_CI_NOT_REVERIFIED": "downstream ci95 / resolved are copied from analysis.json (not re-verified); "
                                    "descriptive, no multiplicity correction (005)",
}


class DecisionTableError(RuntimeError):
    pass


# ------------------------------------------------------------------ inputs --

def _json(p: Path):
    return json.loads(p.read_text())


def _rows(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _norm(x):
    return json.loads(json.dumps(x))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root.parent).as_posix()


def _one(role: str, valid: list) -> dict | None:
    if len(valid) >= 2:
        ids = ", ".join(v["session_id"] for v in valid)
        raise DecisionTableError(f"{role}: {len(valid)} valid formal sessions ({ids}); decide which one with a "
                                 "decision log (021 R1)")
    return valid[0] if valid else None


def quality_problems(meta: dict) -> list:
    """The gate of quality_analysis.main (005, 011, 012, 020)."""
    problems = []
    if meta.get("mode") != "formal":
        problems.append(f"mode {meta.get('mode')!r} is not formal (the pilot report stays masked: 005, 011)")
    if meta.get("env_valid") is not True:
        problems.append(f"session is env-invalid (env_valid={meta.get('env_valid')!r}; rerun it, 012)")
    if meta.get("platform", "macos-metal") != "wsl2-cuda":
        problems.append(f"platform {meta.get('platform', 'macos-metal')!r}: formal quality is measured on wsl2-cuda "
                        "only (020)")
    return problems


def _core_b_role(plan: dict) -> str | None:
    mixes = {r.get("mix") for r in plan.get("runs", [])}
    if not plan.get("derived_from") and mixes == {"uniform"}:
        return "uniform"
    if plan.get("derived_from") and mixes == {"mixed"}:
        return "mixed"
    return None


def _load_b(d: Path) -> dict:
    return {"session": _json(d / "session.json"), "plan": _json(d / "plan.json"), "rows": _rows(d / "rows.jsonl")}


def select_sessions(root: Path) -> dict:
    """021 R1: exactly one valid formal session per input class is used; none -> pending; two or more -> refuse."""
    excluded = []

    def exclude(role, d, sid, problems):
        excluded.append({"role": role, "path": _rel(root, d), "session_id": sid, "problems": problems})

    out = {"core_a": {}}
    for platform in PLATFORMS:
        valid = []
        for sj in sorted((root / "core_a" / platform).glob("*/session.json")):
            s = _json(sj)
            problems = ar.session_problems(s)
            if s.get("platform") != platform:
                problems.append(f"platform {s.get('platform')!r} is not the directory {platform!r}")
            if problems:
                exclude(f"core_a/{platform}", sj.parent, s.get("session_id"), problems)
            else:
                valid.append({"dir": sj.parent, "session_id": s["session_id"], "session": s})
        out["core_a"][platform] = _one(f"core_a/{platform}", valid)

    valid = []
    for sj in sorted((root / "quality").glob("*/session.json")):
        s = _json(sj)
        problems = quality_problems(s)
        if problems:
            exclude("quality", sj.parent, s.get("session_id"), problems)
        else:
            valid.append({"dir": sj.parent, "session_id": s["session_id"], "session": s})
    out["quality"] = _one("quality", valid)

    uniforms, mixeds = [], []
    for sj in sorted((root / "core_b").glob("*/session.json")):
        s = _json(sj)
        pp = sj.parent / "plan.json"
        role = _core_b_role(_json(pp)) if pp.is_file() else None
        if role is None:
            exclude("core_b", sj.parent, s.get("session_id"), ["plan is neither a uniform nor a mixed plan"])
            continue
        if role == "mixed":
            mixeds.append((sj.parent, s))
            continue
        problems = []
        if s.get("formal") is not True:
            problems.append("the uniform session is not formal")
        if s.get("env_valid") is not True:
            problems.append(f"the uniform session is env-invalid (env_valid={s.get('env_valid')!r})")
        if problems:
            exclude("core_b_uniform", sj.parent, s.get("session_id"), problems)
        else:
            uniforms.append({"dir": sj.parent, "session_id": s["session_id"], "session": s})
    out["core_b_uniform"] = uniform = _one("core_b_uniform", uniforms)

    valid = []
    for d, s in mixeds:
        if uniform is None:
            exclude("core_b_mixed", d, s.get("session_id"), ["no selected uniform session"])
            continue
        plan = _json(d / "plan.json")
        if plan.get("derived_from") != uniform["session_id"]:
            exclude("core_b_mixed", d, s.get("session_id"),
                    [f"the mixed plan was derived from {plan.get('derived_from')!r}, not {uniform['session_id']!r}"])
            continue
        try:
            br.report(_load_b(uniform["dir"]), _load_b(d))
        except br.ReportError as exc:
            exclude("core_b_mixed", d, s.get("session_id"), [str(exc)])
            continue
        valid.append({"dir": d, "session_id": s["session_id"], "session": s})
    out["core_b_mixed"] = _one("core_b_mixed", valid)
    out["excluded"] = sorted(excluded, key=lambda e: (e["role"], e["path"]))
    return out


# --------------------------------------------------------------- integrity --

def verify_core_a(platform: str, session: dict, analysis: dict, rows: list, pins: dict) -> None:
    """021 R3: rerun the frozen core_a_report.report and compare field by field."""
    problems = []
    for key, want in (("session", session.get("session_id")), ("mode", session.get("mode")),
                      ("env_valid", session.get("env_valid")), ("sentinel", _norm(session.get("sentinel")))):
        if analysis.get(key) != want:
            problems.append(f"{key} {analysis.get(key)!r} != session.json {want!r}")
    sub = analysis.get("subtracted_embd_bytes") or {}
    want_keys = {f"{m['size']}-{q}" for m in pins["core_a"]["models"] for q in m["files"]}
    if set(sub) != want_keys:
        problems.append(f"subtracted_embd_bytes keys {sorted(sub)} != pins {sorted(want_keys)}")
    else:
        params = None
        if platform == "wsl2-cuda":
            params = {k: pins["stage_b"][k] for k in ("v_total_mib", "b_mib", "c_ctx_mib")}
        try:
            rep = _norm(ar.report(rows, platform=platform, pins=pins,
                                  embd_bytes={tuple(k.split("-", 1)): v for k, v in sub.items()},
                                  tied={m["size"]: False for m in pins["core_a"]["models"]}, platform_params=params))
        except ar.ReportError as exc:
            problems.append(str(exc))
        else:
            for key in ("platform", "platform_params", "subtracted_embd_bytes", "spill", "cells", "hypotheses"):
                if rep[key] != analysis.get(key):
                    problems.append(f"{key} differs from its recomputation")
    if problems:
        raise DecisionTableError(f"core_a/{platform} {session.get('session_id')}: analysis.json does not match "
                                 f"(021 R3): {'; '.join(problems)}; regenerate it with core_a_report.py")


def downstream_cheap(rows: list) -> dict:
    """The fields of quality_analysis.downstream that need no bootstrap, with its filters."""
    cells = {(r["task"], r["size"], r["point"]): r["metrics"]["correct"] for r in rows
             if r["kind"] in ("mc", "winogrande") and r["outcome"] == "ok" and not r.get("repeat_of")}
    out = {}
    for task, size in sorted({(t, s) for t, s, _ in cells}):
        for a, b in qa.DOWNSTREAM_PAIRS:
            x, y = cells.get((task, size, a)), cells.get((task, size, b))
            if not x or not y or len(x) != len(y):
                continue
            d = [j - i for i, j in zip(x, y)]
            out[(task, size, a, b)] = {"n": len(d), "delta_acc": statistics.mean(d),
                                       "flip_rate": sum(v != 0 for v in d) / len(d)}
    return _norm({"|".join(k): v for k, v in out.items()})


def verify_quality(session: dict, analysis: dict, rows: list) -> None:
    problems = []
    if analysis.get("session") != session.get("session_id"):
        problems.append(f"session {analysis.get('session')!r} != {session.get('session_id')!r}")
    if analysis.get("platform") != "wsl2-cuda":
        problems.append(f"platform {analysis.get('platform')!r} is not wsl2-cuda (020)")
    rows = [r for r in rows if r["kind"] != "delete_kld"]
    if _norm(qa.h_a5(rows)) != analysis.get("h_a5"):
        problems.append("h_a5 differs from its recomputation (N_BOOT, SEED defaults)")
    stored = analysis.get("downstream") or []
    have = _norm({f"{d['task']}|{d['size']}|{d['pair'][0]}|{d['pair'][1]}":
                  {"n": d["n"], "delta_acc": d["delta_acc"], "flip_rate": d["flip_rate"]} for d in stored})
    if have != downstream_cheap(rows):
        problems.append("downstream n / delta_acc / flip_rate differ from their recomputation")
    if any(d.get("n_boot") != qa.N_BOOT or d.get("seed") != qa.SEED for d in stored):
        problems.append(f"downstream was not computed with n_boot {qa.N_BOOT} / seed {qa.SEED}")
    if problems:
        raise DecisionTableError(f"quality {session.get('session_id')}: analysis.json does not match (021 R3): "
                                 f"{'; '.join(problems)}; regenerate it with quality_analysis.py")


def verify_core_b(uniform: dict, mixed: dict | None, analysis: dict) -> None:
    if _norm(br.report(uniform, mixed)) != analysis:
        which = "uniform + mixed" if mixed else "uniform only"
        raise DecisionTableError(f"core_b {uniform['session']['session_id']}: analysis.json does not match "
                                 f"core_b_report.report({which}) (021 R3); regenerate it with core_b_report.py")


# --------------------------------------------------------------- relations --

def scalar_rel(axis: str, a: dict | None, b: dict | None) -> str:
    """021 R7: 'better' / 'worse' only beyond the frozen §2 noise rule; otherwise 'tie'."""
    if not a or not b or not an._beyond_noise(a["m"], a["se"], b["m"], b["se"]):
        return "tie"
    a_higher = a["m"] > b["m"]
    return "better" if a_higher == HIGHER_IS_BETTER[axis] else "worse"


def quality_relations(h_a5: dict | None, configs: list) -> dict | None:
    """021 R8: (a, b) -> {rel, source, contrary}, only from the H-A5 Holm results; absent pairs are unmeasured.
    None when the Holm family is unavailable (no session, missing analysis, or not testable)."""
    if not h_a5 or h_a5.get("verdict") == "not testable" or not h_a5.get("tests") \
            or any("reject" not in t for t in h_a5["tests"]):
        return None
    tests = {t["name"]: t for t in h_a5["tests"]}
    rel: dict = {}

    def better(x, y, source, contrary):
        rel[(x, y)] = {"rel": "better", "source": source, "contrary": contrary}
        rel[(y, x)] = {"rel": "worse", "source": source, "contrary": contrary}

    def tie(x, y, source):
        rel[(x, y)] = rel[(y, x)] = {"rel": "tie", "source": source, "contrary": False}

    for size in sorted({c.split("-", 1)[0] for c in configs}):
        direction, names = {}, []
        for lo, hi in qa.KLD_PAIRS:  # predicted KLD(hi) > KLD(lo): z > 0 means lo (more bits) has better quality
            name = f"KLD {size} {hi} > {lo}"
            t = tests.get(name)
            names.append(name)
            if t is None:
                continue
            if not t["reject"]:
                tie(f"{size}-{lo}", f"{size}-{hi}", name)
                direction[(lo, hi)] = "tie"
            elif t["z"] > 0:
                better(f"{size}-{lo}", f"{size}-{hi}", name, False)
                direction[(lo, hi)] = "lo"
            else:
                better(f"{size}-{hi}", f"{size}-{lo}", name, True)
                direction[(lo, hi)] = "hi"
        first, last = qa.KLD_PAIRS[0][0], qa.KLD_PAIRS[-1][1]
        dirs = {direction.get(p) for p in qa.KLD_PAIRS}
        source = "chain:" + "+".join(names)
        if dirs == {"lo"}:  # strict chain only: a tie is never carried across (021 R8)
            better(f"{size}-{first}", f"{size}-{last}", source, False)
        elif dirs == {"hi"}:
            better(f"{size}-{last}", f"{size}-{first}", source, True)
    t = tests.get(PAIRED_TEST)
    if t is not None:
        if not t["reject"]:
            tie(HA5_A, HA5_B, PAIRED_TEST)
        elif t["mean_delta"] < 0:
            better(HA5_A, HA5_B, PAIRED_TEST, False)
        else:
            better(HA5_B, HA5_A, PAIRED_TEST, True)
    return rel


def qrel(rels: dict | None, a: str, b: str) -> str:
    if rels is None:
        return "unmeasured"
    return rels.get((a, b), {"rel": "unmeasured"})["rel"]


def axis_rels(a: str, b: str, vals: dict, rels: dict | None) -> dict:
    out = {"quality": qrel(rels, a, b)}
    for axis in ("decode", "ttft", "memory"):
        out[axis] = scalar_rel(axis, vals[a][axis], vals[b][axis])
    return out


def dominates(r: dict) -> bool:
    """021 R9: no axis unmeasured, none worse, at least one better."""
    v = [r[a] for a in AXES]
    return "unmeasured" not in v and "worse" not in v and "better" in v


def source_sccs(nodes: list, edges: set) -> list:
    """021 R10: the strongly connected components with no edge from outside, in node order (never empty)."""
    reach = {}
    for n in nodes:
        seen, todo = set(), [n]
        while todo:
            x = todo.pop()
            for y in nodes:
                if (x, y) in edges and y not in seen:
                    seen.add(y)
                    todo.append(y)
        reach[n] = seen
    out, placed = [], set()
    for n in nodes:
        if n in placed:
            continue
        comp = [m for m in nodes if m == n or (m in reach[n] and n in reach[m])]
        placed.update(comp)
        if not any((x, m) in edges for m in comp for x in nodes if x not in comp):
            out.append(comp)
    return out


def pick(front: list, order: tuple, rel) -> dict:
    """021 R11: axis by axis, drop a member beaten beyond noise on that axis by another remaining member; once
    quality has been used, the beater's quality relation to it must be 'tie' (unmeasured never acts as a tie)."""
    kept, steps = list(front), []
    for k, axis in enumerate(order):
        q_before = "quality" in order[:k]
        eliminated = []
        for y in kept:
            by = [x for x in kept if x != y and rel(axis, x, y) == "better"
                  and (not q_before or rel("quality", x, y) == "tie")]
            if by:
                eliminated.append({"config": y, "by": by})
        gone = {e["config"] for e in eliminated}
        kept = [y for y in kept if y not in gone]
        steps.append({"axis": axis, "kept": list(kept), "eliminated": eliminated})
    if len(kept) == 1:
        status = "decided"
    elif all(rel("quality", a, b) == "tie" for a in kept for b in kept if a != b):
        status = "tie"
    else:
        status = "incomparable"
    return {"status": status, "configs": kept, "steps": steps}


def _ratio(a: dict | None, b: dict | None):
    if not a or not b:
        return None
    return list(an._ratio(a, b))


# ------------------------------------------------------------------ Core A --

def _cell_status(analysis: dict, rows_by_cell: dict, mem_by_cell: dict, size: str, q: str, L: int) -> tuple:
    c = analysis["cells"].get(f"{size}-{q}-{L}")
    if c is None:
        return "not measured", None
    if c["outcome"] == "oom":
        return "oom", None
    if c["outcome"] == "error":
        bad = next((r for r in rows_by_cell.get((size, q, L), []) if r.get("outcome") != "ok"), {})
        return "error", f"{bad.get('kind')}: {bad.get('stage')}: {bad.get('reason')}"
    if [size, q, L] in analysis["spill"]:
        return "spill", None
    missing = [m for m in ("decode", "ttft") if not c.get(m) or c[m]["n"] != ap.REPS]
    if len(mem_by_cell.get((size, q, L), [])) != ap.REPS:
        missing.append("memory")
    if missing:
        return "incomplete", "missing: " + ", ".join(missing)
    return "candidate", None


def _memory(platform: str, rows_by_cell: dict, pins: dict) -> tuple:
    """021 R6: axis values per fit/ok process (Mac: devices + host self; RTX: devices self = VRAM) and display."""
    axis, display = {}, {}
    for key, rs in rows_by_cell.items():
        fits = [r for r in rs if r["kind"] == "fit" and r["outcome"] == "ok" and r.get("block") in (None, "memory")
                and ((r.get("summary") or {}).get("memory"))]
        if not fits:
            continue
        dev = [sum(d.get("self", 0) for d in r["summary"]["memory"].get("devices", [])) for r in fits]
        host = [((r["summary"]["memory"].get("host") or {}).get("self") or 0) for r in fits]
        axis[key] = [d + h for d, h in zip(dev, host)] if platform == "macos-metal" else dev
        ext = [r.get("external_memory") or {} for r in fits]
        disp = {"total_max": max(d + h for d, h in zip(dev, host)), "device_max": max(dev), "host_max": max(host)}
        if platform == "macos-metal":
            fp = [e["peak_footprint_bytes"] for e in ext if e.get("peak_footprint_bytes") is not None]
            disp["external"] = {"peak_footprint_mib": max(fp) / 2**20 if fp else None,
                                "note": "footprint excludes the mmap'd weights (012)"}
        else:
            sb = pins["stage_b"]
            gp = [e["gpu_mem_used_peak_mib"] for e in ext if e.get("gpu_mem_used_peak_mib") is not None]
            disp["external"] = {"gpu_mem_used_peak_minus_b_mib": max(gp) - sb["b_mib"] if gp else None,
                                "note": "nvidia-smi memory.used peak - B (§3)"}
            disp["spill_margin_mib"] = sb["v_total_mib"] - sb["b_mib"] - sb["c_ctx_mib"] - max(dev)
        display[key] = disp
    return axis, display


def _unit_counts(name: str, h: dict) -> dict:
    units = h.get("units", [])
    if name == "H-A1":
        counts: dict = {}
        for u in units:
            counts[u["observed"]] = counts.get(u["observed"], 0) + 1
        testable = len(units)
    else:
        counts = {}
        for u in units:
            counts[u["status"]] = counts.get(u["status"], 0) + 1
        testable = sum(v for k, v in counts.items() if k in TESTABLE[name])
    return {"counts": dict(sorted(counts.items())), "testable": testable,
            "labels": [] if testable else ["NO_TESTABLE_UNITS"]}


def _verdicts(hyps: dict) -> dict:
    return {name: {"verdict": h["verdict"], "unit_counts": _unit_counts(name, h), "copy": h}
            for name, h in sorted(hyps.items())}


def core_a_section(platform: str, sel: dict | None, pins: dict, rels: dict | None, quality_note: str | None,
                   downstream: list) -> dict:
    """One platform only (021 R18): this platform's Core A inputs plus the shared quality relations."""
    if sel is None:
        return {"status": "pending", "reason": "pending: no valid formal session"}
    if sel.get("analysis") is None:
        return {"status": "pending", "session_id": sel["session_id"],
                "reason": "pending: analysis.json missing (run core_a_report.py)"}
    s, analysis = sel["session"], sel["analysis"]
    sizes = [m["size"] for m in pins["core_a"]["models"]]
    quants = list(pins["core_a"]["quants"])
    levels_L = list(pins["core_a"]["context_lengths"])
    configs = [f"{sz}-{q}" for sz in sizes for q in quants]
    rows_by_cell: dict = {}
    for r in sel["rows_sorted"]:
        if r["kind"] in ("fit", "ttft", "decode"):
            rows_by_cell.setdefault((r["size"], r["quant"], r["L"]), []).append(r)
    mem, mem_disp = _memory(platform, rows_by_cell, pins)
    h_a1 = {tuple(u["cell"]): u for u in analysis["hypotheses"]["H-A1"]["units"]}
    labels = ["QUALITY_PLATFORM_MAC" if platform == "macos-metal" else "QUALITY_PLATFORM_RTX", "QUALITY_CTX_512",
              "TTFT_ENGINE_ONLY", "N3_NOISE_RULE", "CROSS_SIZE_UNMEASURED", "TIE_NOT_EQUIVALENCE"]
    labels += ["MAC_POWER_CONDITION"] if platform == "macos-metal" else ["SPILL_EXCLUDED"]
    conditions = {"sentinel_drift_pct": (s.get("sentinel") or {}).get("drift_pct"),
                  "platform_params": analysis["platform_params"]}
    if platform == "macos-metal":
        conditions.update({"powermode": s.get("powermode"), "adapter_rule": "Apple 70W USB-C Power Adapter (019)",
                           "charge_pauses": len(s.get("charge_pauses") or [])})
    levels, fit_map = {}, {c: {} for c in configs}
    for L in levels_L:
        cfg_out, vals, cands = {}, {}, []
        for cfg in configs:
            size, q = cfg.split("-", 1)
            status, reason = _cell_status(analysis, rows_by_cell, mem, size, q, L)
            cell = analysis["cells"].get(f"{cfg}-{L}") or {}
            m = an._ms(mem[(size, q, L)]) if (size, q, L) in mem else None
            u = h_a1.get((size, q, L))
            cfg_out[cfg] = {"status": status, "reason": reason, "decode": cell.get("decode"),
                            "ttft": cell.get("ttft"), "prefill": cell.get("prefill"), "memory_axis": m,
                            "memory_display": mem_disp.get((size, q, L)),
                            "h_a1": {"predicted": u["predicted"], "observed": u["observed"]} if u else "no unit"}
            fit_map[cfg][str(L)] = status
            if status == "candidate":
                cands.append(cfg)
                vals[cfg] = {"decode": cell["decode"], "ttft": cell["ttft"], "memory": m}
        unknown = [c for c in configs if cfg_out[c]["status"] in ("not measured", "error", "incomplete")]
        excluded = [{"config": c, "status": cfg_out[c]["status"], "reason": cfg_out[c]["reason"]}
                    for c in configs if cfg_out[c]["status"] != "candidate"]
        def h1(cfg):
            v = cfg_out[cfg]["h_a1"]
            return v if isinstance(v, dict) else {"predicted": "no unit", "observed": "no unit"}

        level = {"configs": cfg_out, "excluded": excluded,
                 "fit_boundary": [{**e, "h_a1_predicted": h1(e["config"])["predicted"],
                                   "h_a1_observed": h1(e["config"])["observed"]} for e in excluded],
                 "provisional": bool(unknown), "unknown_configs": unknown, "caveats": []}
        if unknown:
            level["caveats"].append("PROVISIONAL")
        if rels is None:
            reason = f"not decidable: quality relations unavailable ({quality_note})"
            level.update({"frontier": None, "noise_cycles": [], "dominated": [],
                          "picks": {w: {"order": list(o), "status": reason} for w, o in WORKLOADS.items()}})
        elif not cands:
            level.update({"frontier": [], "noise_cycles": [], "dominated": [],
                          "picks": {w: {"order": list(o), "status": "no candidate"} for w, o in WORKLOADS.items()}})
        else:
            level.update(_front_and_picks(cands, vals, rels, downstream, unknown))
            if any("NO_IMATRIX_Q4" in p.get("caveats", []) for p in level["picks"].values()):
                level["caveats"].append("NO_IMATRIX_Q4")
            if HA5_A in cands and HA5_B in cands and qrel(rels, HA5_A, HA5_B) in ("better", "worse"):
                level["caveats"].append("CONTAMINATION")
            if level["noise_cycles"]:
                level["caveats"].append("NOISE_CYCLE")
        level["ha5a"] = _ha5a(cfg_out, vals, rels, sel.get("paired_test"))
        level["caveats"] = sorted(set(level["caveats"]))
        levels[str(L)] = level
    largest = {}
    for c in configs:
        ok_L = [L for L in levels_L if fit_map[c][str(L)] == "candidate"
                or (fit_map[c][str(L)] == "incomplete" and (analysis["cells"].get(f"{c}-{L}") or {}).get("outcome") == "ok")]
        largest[c] = max(ok_L) if ok_L else None
    return {"status": "ok", "session_id": sel["session_id"], "conditions": conditions,
            "memory_axis_definition": ("sum of device self + host self (unified memory)" if platform == "macos-metal"
                                       else "sum of device self = VRAM, the 004 fit / spill budget; host shown apart"),
            "labels": sorted(labels), "verdicts": _verdicts(analysis["hypotheses"]),
            "fit_map": {"configs": fit_map, "largest_fit_L": largest}, "levels": levels}


def _front_and_picks(cands: list, vals: dict, rels: dict, downstream: list, unknown: list) -> dict:
    rel_ab = {(a, b): axis_rels(a, b, vals, rels) for a in cands for b in cands if a != b}
    edges = {k for k, r in rel_ab.items() if dominates(r)}
    sccs = source_sccs(cands, edges)
    front = [c for comp in sccs for c in comp]
    front = [c for c in cands if c in front]
    dominated = []
    for b in cands:
        if b in front:
            continue
        by = [a for a in cands if (a, b) in edges]
        dominated.append({"config": b, "dominated_by": by, "relations": {a: rel_ab[(a, b)] for a in by},
                          "tie_dependent": any(rel_ab[(a, b)]["quality"] == "tie" for a in by),
                          "tie_text": TIE_TEXT if any(rel_ab[(a, b)]["quality"] == "tie" for a in by) else None})

    def rel(axis, x, y):
        return rel_ab[(x, y)][axis]

    picks = {}
    for w, order in WORKLOADS.items():
        p = pick(front, order, rel)
        eliminated_at = {e["config"]: (s["axis"], e["by"]) for s in p["steps"] for e in s["eliminated"]}
        alternatives = []
        for alt in front:
            if alt in p["configs"]:
                continue
            axis, by = eliminated_at[alt]
            alternatives.append({"config": alt, "eliminated_at": axis, "by": by, "vs": {
                pc: {**rel_ab[(alt, pc)],
                     "decode_ratio_alt_over_pick": _ratio(vals[alt]["decode"], vals[pc]["decode"]),
                     "ttft_ratio_alt_over_pick": _ratio(vals[alt]["ttft"], vals[pc]["ttft"]),
                     "memory_ratio_alt_over_pick": _ratio(vals[alt]["memory"], vals[pc]["memory"])}
                for pc in p["configs"]}})
        caveats = ["NO_IMATRIX_Q4"] if any(c.endswith("Q4_K_M") for c in p["configs"]) else []
        if unknown:
            caveats.append("PROVISIONAL")
        if any(rels.get((a, b), {}).get("contrary") for a in p["configs"] for b in front if a != b):
            caveats.append("CONTRARY_TO_PREDICTION")
        picks[w] = {"order": list(order), "status": p["status"], "provisional": bool(unknown),
                    "unknown_configs": unknown, "configs": p["configs"], "steps": p["steps"],
                    "alternatives": alternatives, "downstream": _pick_downstream(p["configs"], downstream),
                    "caveats": sorted(caveats)}
    return {"frontier": sccs, "noise_cycles": [c for c in sccs if len(c) > 1], "dominated": dominated,
            "picks": picks}


def _pick_downstream(configs: list, downstream: list) -> list:
    """Same-size pairs among the picked configs, plus each one vs BF16; cross-size pairs are not compared."""
    out = []
    for d in downstream:
        a, b = (f"{d['size']}-{x}" for x in d["pair"])
        if (d["pair"][0] == "BF16" and b in configs) or (a in configs and b in configs):
            out.append({**d, "note": "descriptive, no multiplicity correction (005); ci95 not re-verified"})
    if len({c.split("-", 1)[0] for c in configs}) > 1:
        out.append({"cross_size": "not compared"})
    return out


def _ha5a(cfg_out: dict, vals: dict, rels: dict | None, paired: dict | None) -> dict:
    """021 R14 / §2 H-A5(a): 9B-Q4_K_M and 4B-Q8_0 side by side; no conclusion drawn."""
    side = {c: {k: cfg_out[c][k] for k in ("status", "memory_axis", "decode", "ttft")} for c in (HA5_A, HA5_B)}
    relations = None
    if HA5_A in vals and HA5_B in vals:
        relations = axis_rels(HA5_A, HA5_B, vals, rels)
    test = None
    if paired:
        test = {k: paired.get(k) for k in ("mean_delta", "ci95", "p_adj", "reject", "direction_ok")}
        test["ci95_note"] = "descriptive only (§2)"
    return {"configs": side, "relations_9B_vs_4B": relations, "paired_test": test}


# ----------------------------------------------------------------- quality --

def quality_section(sel: dict | None, configs: list) -> tuple:
    """-> (section, relations or None, reason when unavailable, downstream list, paired test)."""
    if sel is None:
        reason = "no valid formal quality session"
        return {"status": "pending", "reason": reason}, None, reason, [], None
    if sel.get("analysis") is None:
        reason = "quality analysis.json missing (run quality_analysis.py)"
        return {"status": "pending", "session_id": sel["session_id"], "reason": reason}, None, reason, [], None
    a = sel["analysis"]
    rows = [r for r in sel["rows_sorted"] if r["kind"] != "delete_kld"]
    per = {}
    for r in rows:
        if r["kind"] == "kld" and r["outcome"] == "ok" and not r.get("variant") and not r.get("repeat_of"):
            per[f"{r['size']}-{r['point']}"] = {k: r["metrics"].get(k) for k in
                                                ("kld_mean", "kld_mean_se", "ppl_q", "ppl_q_se", "same_top_p_pct",
                                                 "kld_p99")}
    floor = {r["size"]: r["metrics"].get("kld_mean") for r in rows if r["kind"] == "kld_self" and r["outcome"] == "ok"
             and not r.get("repeat_of")}
    bf16 = {r["size"]: [r["metrics"].get("ppl"), r["metrics"].get("ppl_se")] for r in rows
            if r["kind"] == "kld_base" and r["outcome"] == "ok" and not r.get("repeat_of")}
    h = a["h_a5"]
    rels = quality_relations(h, configs)
    reason = None
    if rels is None:
        reason = f"h_a5 not testable: missing {h.get('missing')}"
    listed = []
    for i, x in enumerate(configs):
        for y in configs[i + 1:]:
            r = (rels or {}).get((x, y))
            listed.append({"a": x, "b": y, "rel": r["rel"] if r else "unmeasured",
                           "source": r["source"] if r else None, "contrary": r["contrary"] if r else False})
    paired = next((t for t in h.get("tests", []) if t["name"] == PAIRED_TEST), None)
    downstream = [{**d, "ci95_reverified": False} for d in a.get("downstream") or []]
    sec = {"status": "ok" if rels is not None else "unavailable", "reason": reason, "platform": "wsl2-cuda",
           "ctx": 512, "session_id": sel["session_id"], "per_config": dict(sorted(per.items())),
           "floor_kld_self": dict(sorted(floor.items())), "bf16_ppl": dict(sorted(bf16.items())),
           "h_a5": {"verdict": h["verdict"], "missing": h.get("missing"), "tests": h.get("tests"),
                    "labels": ["NO_TESTABLE_UNITS"] if h["verdict"] == "not testable" else []},
           "relations": listed, "downstream": downstream, "labels": ["DOWNSTREAM_CI_NOT_REVERIFIED", "QUALITY_CTX_512"]}
    return sec, rels, reason, downstream, paired


# ------------------------------------------------------------------ Core B --

def _b_ms(values: list) -> dict | None:
    if any(v is None for v in values):
        return None
    return {"m": statistics.mean(values), "se": statistics.stdev(values) / math.sqrt(len(values))}


def core_b_section(sel_u: dict | None, sel_m: dict | None, analysis: dict | None) -> dict:
    if sel_u is None:
        return {"status": "pending", "reason": "pending: no valid formal uniform session"}
    if analysis is None:
        return {"status": "pending", "session_id": sel_u["session_id"],
                "reason": "pending: analysis.json missing (run core_b_report.py)"}
    plan, rows = sel_u["loaded"]["plan"], sel_u["rows_sorted"]
    hyp = analysis["hypotheses"]
    unit = {name: {u["L"]: u for u in hyp[name]["units"]} for name in ("H-B1", "H-B2", "H-B3", "H-B4")}
    levels = {}
    for L in sorted(int(x) for x in plan["grids"]):
        grid = sorted(plan["grids"][str(L)])
        curve, by_c = [], {}
        for C in grid:
            rs = [r for r in rows if r["L"] == L and r["mix"] == "uniform" and r["C"] == C and r.get("ok") is True]
            n_valid = len({r["restart"] for r in rs})
            complete = n_valid == ba.RESTARTS and len(rs) == ba.RESTARTS
            entry = {"C": C, "n_valid": n_valid, "complete": complete}
            if complete:
                entry.update({k: _b_ms([r.get(k) for r in rs]) for k in B_METRICS})
                cell = analysis["cells"].get(f"uniform-{L}-{C}") or {}
                entry.update({"kv_cache_usage_peak_max": cell.get("kv_cache_usage_peak_max"),
                              "preemptions_total": cell.get("preemptions_total")})
                by_c[C] = entry
            curve.append(entry)
        h1 = unit["H-B1"].get(L)
        level = {"c_kv": plan["c_kv"][str(L)], "grid": grid, "c_max": max(grid), "curve": curve,
                 "caveats": ["CORE_B_FIXED_DEPLOYMENT", "KV_USAGE_INCLUDES_GDN"]}
        if h1 is None or h1["status"] == "not testable":
            level.update({"status": "no operating point", "missing_c": (h1 or {}).get("missing_c"),
                          "c_knee": None, "picks": None})
        else:
            k = ba.knee(rows, L, grid)
            h2 = unit["H-B2"].get(L) or {}
            if not (k["c_knee"] == h1["c_knee"] == h2.get("c_knee")):
                raise DecisionTableError(f"core_b L={L}: C_knee {k['c_knee']} / H-B1 {h1['c_knee']} / "
                                         f"H-B2 {h2.get('c_knee')} disagree (021 R19)")
            c_knee, c_max = k["c_knee"], max(grid)
            thr = {C: by_c[C]["output_throughput"]["m"] for C in grid}
            c_top = min(C for C in grid if thr[C] == max(thr.values()))
            flags = ["KNEE_POINT_ESTIMATE"]
            top_note = None
            if c_knee == c_max:
                flags.append("KNEE_AT_GRID_TOP")
                top_note = (f"grid top = GRID_CAP = max_num_seqs {bp.GRID_CAP} (019); throughput may still rise "
                            "beyond the grid" if c_max == bp.GRID_CAP else
                            "grid top from the 007 rule (smallest power of 2 >= max(8, 8 x C_KV))")
            if c_max < ba.H_B1_COVER * c_knee:
                flags.append("H_B1_UNREFUTABLE")
            latency = {}
            for label, C in (("C=1", 1), ("C_knee", c_knee), ("C_max", c_max)):
                e = by_c.get(C)
                latency[label] = {"C": C, **({k: e[k] for k in ("p50_ttft_ms", "p95_ttft_ms", "p50_tpot_ms",
                                                                   "p95_tpot_ms")} if e else {"n/a": True})}
            h3 = unit["H-B3"].get(L) or {}
            level.update({"status": "ok", "c_knee": c_knee, "picks": {
                "B-T": {"C": c_knee, "throughput_at_knee": by_c[c_knee]["output_throughput"],
                        "throughput_max": by_c[c_top]["output_throughput"], "c_at_max": c_top,
                        "observed_gain_pct": (thr[c_top] / thr[c_knee] - 1) * 100,
                        "gain_bound_pct": (1 / ba.KNEE_FRACTION - 1) * 100, "flags": flags,
                        "grid_top_note": top_note},
                "B-L": {"gain_bound_pct": (1 / ba.KNEE_FRACTION - 1) * 100, "latency": latency,
                        "h_b1": {"status": h1["status"], "first_c_at_3x": h1.get("first_c_at_3x")},
                        "h_b3": {k: h3.get(k) for k in ("status", "g_tpot", "g_ttft")},
                        "statement": BL_STATEMENT}}})
            level["caveats"] += flags
        if L in bp.MIXED_LEVELS:
            if sel_m is None:
                bx = {"status": "pending", "reason": "no valid formal mixed session"}
            else:
                u4 = unit["H-B4"].get(L)
                C = (u4 or {}).get("C") or level.get("c_knee")
                bx = {"status": "ok", "h_b4_unit": u4, "mixed_cell": analysis["cells"].get(f"mixed-{L}-{C}"),
                      "statement": BX_STATEMENT}
                level["caveats"].append("MIXED_AT_KNEE_ONLY")
        else:
            bx = {"status": "not defined (007)"}
        if level.get("picks") is not None:
            level["picks"]["B-X"] = bx
        else:
            level["b_x"] = bx
        level["caveats"] = sorted(set(level["caveats"]))
        levels[str(L)] = level
    pins_b = sel_u["pins_core_b"]
    return {"status": "ok", "platform": "wsl2-cuda", "uniform_session": sel_u["session_id"],
            "mixed_session": sel_m["session_id"] if sel_m else None,
            "scope_line": {"model": f"{pins_b['repo']}@{pins_b['revision']}", "precision": pins_b["precision"],
                           "vllm": sel_u["vllm_version"], "gpu_memory_utilization": plan.get("gpu_memory_utilization"),
                           "kv_cache_memory_bytes": plan.get("kv_cache_memory_bytes"),
                           "num_gpu_blocks": plan.get("num_gpu_blocks"), "block_size": plan.get("block_size"),
                           "max_num_seqs": bp.MAX_NUM_SEQS, "max_num_batched_tokens": bp.MAX_NUM_BATCHED_TOKENS,
                           "uniform_request": "(L - 128, 128)", "temperature": 0, "ignore_eos": True},
            "quality": CORE_B_QUALITY, "verdicts": _verdicts(hyp), "levels": levels}


# ------------------------------------------------------------------- build --

def _not_supported(table: dict) -> list:
    out = [{"item": "quality on Metal", "reason": "not measured (020)"},
           {"item": "long-context quality", "reason": "not measured (-c 512, 005)"},
           {"item": "Core B quality", "reason": "not measured (W4A16; 003/008)"},
           {"item": "multi-user serving on the Mac", "reason": "not measured (SPEC: Core B only on NVIDIA)"}]
    sections = [(f"core_a/{p}", table["core_a"][p]) for p in PLATFORMS] + [("core_b", table["core_b"])]
    for where, sec in sections:
        if sec.get("status") != "ok":
            out.append({"item": where, "reason": sec.get("reason")})
            continue
        for name, v in sec["verdicts"].items():
            if "NO_TESTABLE_UNITS" in v["unit_counts"]["labels"]:
                out.append({"item": f"{where} {name}", "reason": f"verdict {v['verdict']!r} with 0 testable units"})
            for st in ("not testable", "inconclusive", "descriptive"):
                if v["unit_counts"]["counts"].get(st):
                    out.append({"item": f"{where} {name}", "reason": f"{v['unit_counts']['counts'][st]} unit(s) {st}"})
        for L, lv in sec["levels"].items():
            for code in ("PROVISIONAL", "NOISE_CYCLE", "KNEE_AT_GRID_TOP", "H_B1_UNREFUTABLE"):
                if code in lv.get("caveats", []):
                    out.append({"item": f"{where} L={L}", "reason": code})
            if where == "core_b" and lv["status"] != "ok":
                out.append({"item": f"{where} L={L}", "reason": lv["status"]})
            for w, p in (lv.get("picks") or {}).items():
                if where != "core_b" and p.get("status") not in ("decided",):
                    out.append({"item": f"{where} L={L} {w}", "reason": p.get("status")})
                if where == "core_b" and w == "B-X" and p.get("status") not in ("ok", "not defined (007)"):
                    out.append({"item": f"{where} L={L} B-X", "reason": p.get("status")})
            for d in lv.get("dominated", []):
                if d["tie_dependent"]:
                    out.append({"item": f"{where} L={L} {d['config']} dominated", "reason": TIE_TEXT})
            if lv.get("b_x", {}).get("status") == "pending":
                out.append({"item": f"{where} L={L} B-X", "reason": "pending"})
    q = table["quality"]
    if q.get("status") != "ok":
        out.append({"item": "quality", "reason": q.get("reason")})
    else:
        for r in q["relations"]:
            if r["rel"] == "unmeasured":
                out.append({"item": f"quality {r['a']} vs {r['b']}", "reason": "unmeasured"})
        out.append({"item": "downstream ci95", "reason": "copied, not re-verified"})
    return out


def build(root: Path, pins: dict) -> dict:
    """The decision table as one JSON object; raises DecisionTableError for exit 2 (021 R1, R3)."""
    sel = select_sessions(root)
    configs = [f"{m['size']}-{q}" for m in pins["core_a"]["models"] for q in pins["core_a"]["quants"]]
    inputs = []

    def record(role, platform, d: Path, names):
        files = {_rel(root, d / n): _sha(d / n) for n in names if (d / n).is_file()}
        inputs.append({"role": role, "platform": platform, "session_id": d.name, "files": files})

    q = sel["quality"]
    if q is not None:
        q["rows_file"] = _rows(q["dir"] / "rows.jsonl")
        q["rows_sorted"] = sorted(q["rows_file"], key=lambda r: json.dumps(
            [r.get("kind"), r.get("size"), r.get("task") or "", r.get("point"), r.get("job_id")]))
        ap_path = q["dir"] / "analysis.json"
        q["analysis"] = _json(ap_path) if ap_path.is_file() else None
        if q["analysis"] is not None:
            verify_quality(q["session"], q["analysis"], q["rows_file"])
        record("quality", "wsl2-cuda", q["dir"], ("session.json", "rows.jsonl", "analysis.json"))
    q_sec, rels, q_reason, downstream, paired = quality_section(q, configs)

    core_a = {}
    for platform in PLATFORMS:
        s = sel["core_a"][platform]
        if s is not None:
            s["rows_file"] = _rows(s["dir"] / "rows.jsonl")
            s["rows_sorted"] = sorted(s["rows_file"], key=lambda r: json.dumps(
                [r.get("kind"), r.get("size"), r.get("quant"), r.get("L"), r.get("rep"), r.get("job_id")]))
            p = s["dir"] / "analysis.json"
            s["analysis"] = _json(p) if p.is_file() else None
            if s["analysis"] is not None:
                verify_core_a(platform, s["session"], s["analysis"], s["rows_file"], pins)
            s["paired_test"] = paired
            record("core_a", platform, s["dir"], ("session.json", "rows.jsonl", "analysis.json"))
        core_a[platform] = core_a_section(platform, s, pins, rels, q_reason, downstream)

    u, m = sel["core_b_uniform"], sel["core_b_mixed"]
    b_analysis = None
    if u is not None:
        u["loaded"] = _load_b(u["dir"])
        u["rows_sorted"] = sorted(u["loaded"]["rows"], key=lambda r: (r["L"], r["mix"], r["C"], r["restart"]))
        u["pins_core_b"] = pins["core_b"]
        u["vllm_version"] = pins["vllm"]["version"]
        p = u["dir"] / "analysis.json"
        if p.is_file():
            b_analysis = _json(p)
            verify_core_b(u["loaded"], _load_b(m["dir"]) if m else None, b_analysis)
        record("core_b_uniform", "wsl2-cuda", u["dir"], ("session.json", "plan.json", "rows.jsonl", "analysis.json"))
        if m is not None:
            record("core_b_mixed", "wsl2-cuda", m["dir"], ("session.json", "plan.json", "rows.jsonl"))
    core_b = core_b_section(u, m, b_analysis)

    code = {"benchmarks/decision_table.py": _sha(Path(__file__))}
    code.update({f: _sha(ROOT / f) for f in FROZEN_MODULES})
    table = {
        "schema_version": SCHEMA_VERSION, "rules": RULES,
        "provenance": {"pins_canonical_sha256": hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest(),
                       "code": code, "inputs": sorted(inputs, key=lambda i: (i["role"], i["platform"])),
                       "excluded": sel["excluded"]},
        "global_caveats": ["NO_RANKING", "CORE_A_B_SEPARATE", "TIE_NOT_EQUIVALENCE", "N3_NOISE_RULE",
                           "QUALITY_CTX_512", "TTFT_ENGINE_ONLY", "NO_IMATRIX_Q4"],
        "caveat_text": CAVEATS, "scope": SCOPE, "workloads": {w: {"order": list(o), "text": WORKLOAD_TEXT[w]}
                                                            for w, o in WORKLOADS.items()},
        "quality": q_sec, "core_a": core_a, "core_b": core_b,
    }
    table["not_supported"] = _not_supported(table)
    return _norm(table)


# ---------------------------------------------------------------- markdown --

def _n(x, fmt: str) -> str:
    return "n/a" if x is None else f"{x:{fmt}}"


def _pm(m, se, fmt: str) -> str:
    return "n/a" if m is None else (_n(m, fmt) if se is None else f"{m:{fmt}} ± {se:{fmt}}")


def _f(d, fmt: str) -> str:
    if not d:
        return "n/a"
    if d.get("se") is None:
        return f"{d['m']:{fmt}}"
    return f"{d['m']:{fmt}} ± {d['se']:{fmt}}"


def _cell_text(p: dict) -> str:
    st = p.get("status", "")
    if st in ("decided", "tie", "incomparable"):
        sep = {"decided": "", "tie": " = ", "incomparable": " ‖ "}[st]
        txt = sep.join(p["configs"]) if st != "decided" else p["configs"][0]
        return txt + (" †" if p.get("provisional") else "")
    return st


def _md_core_a(platform: str, sec: dict, q: dict) -> list:
    out = [f"## Core A — {platform}", ""]
    if sec["status"] != "ok":
        return out + [f"{sec['reason']}", ""]
    c = sec["conditions"]
    out.append(f"Session `{sec['session_id']}`; sentinel drift {c['sentinel_drift_pct']} %.")
    if platform == "macos-metal":
        out.append(f"Power: powermode {c['powermode']}, {c['adapter_rule']}, charge pauses {c['charge_pauses']}.")
    out.append(f"Memory axis: {sec['memory_axis_definition']}.")
    out.append("Labels: " + ", ".join(sec["labels"]) + ".")
    out += ["", "### Recommendations", "", "| workload | " + " | ".join(f"L = {L}" for L in sec["levels"]) + " |",
            "|---|" + "---|" * len(sec["levels"])]
    for w in WORKLOADS:
        out.append(f"| {w} {WORKLOAD_TEXT[w]} | " + " | ".join(_cell_text(lv["picks"][w])
                                                         for lv in sec["levels"].values()) + " |")
    out.append("| A-F not deployable | " + " | ".join(
        "; ".join(f"{e['config']} ({e['status']})" for e in lv["excluded"]) or "none"
        for lv in sec["levels"].values()) + " |")
    out.append("")
    out.append("† provisional: an unknown configuration (not measured / error / incomplete) could change the pick. "
               "`X = Y`: tie (no equivalence claimed). `X ‖ Y`: incomparable (quality not measured across sizes).")
    for L, lv in sec["levels"].items():
        out += ["", f"### L = {L}: evidence", "",
                "| config | state | KLD ± SE | PPL(Q) ± SE | decode tok/s | TTFT s | prefill tok/s | memory axis MiB | "
                "total max / host max MiB |", "|---|---|---|---|---|---|---|---|---|"]
        front = {x for comp in (lv.get("frontier") or []) for x in comp}
        dom = {d["config"]: d for d in lv.get("dominated", [])}
        for cfg, v in lv["configs"].items():
            if v["status"] != "candidate":
                state = f"excluded: {v['status']}" + (f" ({v['reason']})" if v["reason"] else "")
            elif cfg in front:
                state = "front"
            elif cfg in dom:
                state = "dominated by " + ", ".join(
                    f"{a} ({'/'.join(f'{ax} {r}' for ax, r in dom[cfg]['relations'][a].items())})"
                    for a in dom[cfg]["dominated_by"])
            else:
                state = "candidate"
            qq = (q.get("per_config") or {}).get(cfg) or {}
            kld = _pm(qq.get("kld_mean"), qq.get("kld_mean_se"), ".3g")
            ppl = _pm(qq.get("ppl_q"), qq.get("ppl_q_se"), ".4f")
            md = v["memory_display"] or {}
            tot = "n/a" if not md else f"{md['total_max']:.0f} / {md['host_max']:.0f}"
            out.append(f"| {cfg} | {state} | {kld} | {ppl} | {_f(v['decode'], '.1f')} | {_f(v['ttft'], '.4g')} | "
                       f"{_f(v['prefill'], '.1f')} | {_f(v['memory_axis'], '.0f')} | {tot} |")
        if lv.get("picks") and isinstance(next(iter(lv["picks"].values())).get("configs"), list):
            out += ["", f"Alternatives and costs (ratios = alternative ÷ pick, ± SE):", ""]
            for w, p in lv["picks"].items():
                for alt in p.get("alternatives", []):
                    for pc, r in alt["vs"].items():
                        rat = ", ".join(f"{k.split('_')[0]} {x[0]:.3f}" + (f" ± {x[1]:.3f}" if x[1] is not None else "")
                                        for k, x in r.items() if k.endswith("_over_pick") and x)
                        axes = ", ".join(f"{a} {r[a]}" for a in AXES)
                        out.append(f"- {w}: {alt['config']} vs {pc} — dropped at {alt['eliminated_at']} by "
                                   f"{', '.join(alt['by'])}; {axes}; {rat}")
        h = lv["ha5a"]
        out += ["", f"H-A5(a) side by side (§2), L = {L}:", "", "| config | state | memory axis MiB | decode tok/s | "
                "TTFT s |", "|---|---|---|---|---|"]
        for cfg, v in h["configs"].items():
            out.append(f"| {cfg} | {v['status']} | {_f(v['memory_axis'], '.0f')} | {_f(v['decode'], '.1f')} | "
                       f"{_f(v['ttft'], '.4g')} |")
        if h["relations_9B_vs_4B"]:
            out.append("")
            out.append(f"{HA5_A} vs {HA5_B}: " + ", ".join(f"{a} {r}" for a, r in h["relations_9B_vs_4B"].items()))
        if h["paired_test"]:
            t = h["paired_test"]
            out.append(f"Paired ΔNLL test: mean {t['mean_delta']:.4g}, ci95 {t['ci95']} ({t['ci95_note']}), "
                       f"p_adj {t['p_adj']}, reject {t['reject']}, direction_ok {t['direction_ok']}.")
        if lv["caveats"]:
            out.append("")
            out.append("Caveats: " + ", ".join(lv["caveats"]) + ".")
    out += ["", "### Fit map", "", "| config | " + " | ".join(f"L = {L}" for L in sec["levels"]) + " | largest fit L |",
            "|---|" + "---|" * (len(sec["levels"]) + 1)]
    for cfg, row in sec["fit_map"]["configs"].items():
        out.append(f"| {cfg} | " + " | ".join(row.values()) + f" | {sec['fit_map']['largest_fit_L'][cfg]} |")
    out += ["", "### Verdicts (copied verbatim; 'not refuted' only means the test did not refute)", ""]
    for name, v in sec["verdicts"].items():
        lab = f" [{', '.join(v['unit_counts']['labels'])}]" if v["unit_counts"]["labels"] else ""
        out.append(f"- {name}: {v['verdict']}{lab}; units {v['unit_counts']['counts']}")
    return out + [""]


def _md_core_b(sec: dict) -> list:
    out = ["## Core B — wsl2-cuda / vLLM / W4A16", ""]
    if sec["status"] != "ok":
        return out + [sec["reason"], ""]
    s = sec["scope_line"]
    out.append("Scope: " + "; ".join(f"{k} {v}" for k, v in s.items()) + ".")
    out.append(f"Quality: {sec['quality']}.")
    out += ["", "| L | C_KV | C_knee | C_max | throughput at knee / max (tok/s) | observed gain / bound | "
            "P95 TTFT ms (C=1 / knee / max) | P95 TPOT ms (C=1 / knee / max) | flags |",
            "|---|---|---|---|---|---|---|---|---|"]
    for L, lv in sec["levels"].items():
        if lv["status"] != "ok":
            out.append(f"| {L} | {lv['c_kv']} | {lv['status']} (missing C {lv.get('missing_c')}) | {lv['c_max']} "
                       "| | | | | |")
            continue
        bt, bl = lv["picks"]["B-T"], lv["picks"]["B-L"]
        lat = bl["latency"]
        out.append(f"| {L} | {lv['c_kv']} | {lv['c_knee']} | {lv['c_max']} | {_f(bt['throughput_at_knee'], '.1f')} / "
                   f"{_f(bt['throughput_max'], '.1f')} (C {bt['c_at_max']}) | {bt['observed_gain_pct']:.1f} % / "
                   f"{bt['gain_bound_pct']:.1f} % | "
                   + " / ".join(_f(lat[k].get("p95_ttft_ms"), ".1f") for k in ("C=1", "C_knee", "C_max")) + " | "
                   + " / ".join(_f(lat[k].get("p95_tpot_ms"), ".1f") for k in ("C=1", "C_knee", "C_max")) + " | "
                   + ", ".join(bt["flags"]) + " |")
    for L, lv in sec["levels"].items():
        out += ["", f"### L = {L}", ""]
        if lv["status"] == "ok":
            bt, bl = lv["picks"]["B-T"], lv["picks"]["B-L"]
            out.append(f"- B-T (throughput): C = C_knee = {bt['C']}" + (f"; {bt['grid_top_note']}" if bt["grid_top_note"]
                                                                          else "") + ".")
            out.append(f"- B-L (latency-sensitive): {bl['statement']}. H-B1 {bl['h_b1']}; H-B3 {bl['h_b3']}.")
            bx = lv["picks"]["B-X"]
        else:
            bx = lv["b_x"]
        out.append(f"- B-X (mixed traffic): {bx.get('status')}" + (f"; {bx['statement']}; H-B4 unit {bx['h_b4_unit']}"
                                                                   if bx.get("status") == "ok" else "") + ".")
        out += ["", "| C | n valid | output tok/s | req/s | P50 TTFT ms | P95 TTFT ms | P50 TPOT ms | P95 TPOT ms | "
                "KV usage peak | preemptions |", "|---|---|---|---|---|---|---|---|---|---|"]
        for e in lv["curve"]:
            if not e["complete"]:
                out.append(f"| {e['C']} | {e['n_valid']} (incomplete) | | | | | | | | |")
                continue
            out.append(f"| {e['C']} | {e['n_valid']} | {_f(e['output_throughput'], '.1f')} | "
                       f"{_f(e['request_throughput'], '.3f')} | {_f(e['p50_ttft_ms'], '.1f')} | "
                       f"{_f(e['p95_ttft_ms'], '.1f')} | {_f(e['p50_tpot_ms'], '.1f')} | {_f(e['p95_tpot_ms'], '.1f')} | "
                       f"{e['kv_cache_usage_peak_max']} | {e['preemptions_total']} |")
        out.append("")
        out.append("Caveats: " + ", ".join(lv["caveats"]) + ".")
    out += ["", "### Verdicts (copied)", ""]
    for name, v in sec["verdicts"].items():
        lab = f" [{', '.join(v['unit_counts']['labels'])}]" if v["unit_counts"]["labels"] else ""
        out.append(f"- {name}: {v['verdict']}{lab}; units {v['unit_counts']['counts']}")
    return out + [""]


def _md_quality(q: dict) -> list:
    out = ["## Quality — wsl2-cuda, -c 512", ""]
    if q.get("status") == "pending":
        return out + [q["reason"], ""]
    out += ["| config | KLD ± SE | PPL(Q) ± SE | same top-p % | KLD p99 |", "|---|---|---|---|---|"]
    for cfg, v in q["per_config"].items():
        out.append(f"| {cfg} | {_pm(v['kld_mean'], v['kld_mean_se'], '.3g')} | {_pm(v['ppl_q'], v['ppl_q_se'], '.4f')} "
                   f"| {_n(v['same_top_p_pct'], '.2f')} | {_n(v['kld_p99'], '.3g')} |")
    out.append("")
    out.append(f"Self-KLD floor: {q['floor_kld_self']}; BF16 PPL ± SE: {q['bf16_ppl']}.")
    out.append(f"H-A5: {q['h_a5']['verdict']}" + (f" (missing {q['h_a5']['missing']})" if q["h_a5"]["missing"] else "")
               + ".")
    for t in q["h_a5"]["tests"] or []:
        out.append(f"- {t['name']}: p {_n(t.get('p'), '.3g')}, p_adj {_n(t.get('p_adj'), '.3g')}, reject {t.get('reject')}, "
                   f"direction_ok {t['direction_ok']}")
    out += ["", "Relations (Holm only; unmeasured is never a tie):", "", "| a | b | relation of a | source |",
            "|---|---|---|---|"]
    for r in q["relations"]:
        out.append(f"| {r['a']} | {r['b']} | {r['rel']}{' (contrary)' if r['contrary'] else ''} | {r['source']} |")
    out += ["", "Downstream (descriptive, no multiplicity correction; ci95 copied, not re-verified):", "",
            "| task | size | pair | n | Δacc | ci95 | resolved | flip rate |", "|---|---|---|---|---|---|---|---|"]
    for d in q["downstream"]:
        out.append(f"| {d['task']} | {d['size']} | {d['pair'][0]} → {d['pair'][1]} | {d['n']} | {d['delta_acc']:.4f} | "
                   f"[{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | {d['resolved']} | {d['flip_rate']:.4f} |")
    return out + [""]


def render_md(t: dict) -> str:
    out = ["# Deployment decision table", "",
           f"Rules: `{t['rules']}` (schema {t['schema_version']}). Generated from the JSON next to this file.", "",
           "## How to read", ""]
    out += [f"- {s}" for s in t["scope"]]
    out += [f"- {CAVEATS[c]}" for c in t["global_caveats"]]
    out += ["- Workloads (one primary axis each, then quality, then the SPEC metric order):"]
    out += [f"  - {w}: {v['text']}; axes {' → '.join(v['order'])}" for w, v in t["workloads"].items()]
    out += ["", "Inputs:"]
    out += [f"- {i['role']} {i['platform']}: `{i['session_id']}`" for i in t["provenance"]["inputs"]]
    out += ["", "Excluded sessions:"]
    out += [f"- {e['role']}: `{e['session_id']}` — {'; '.join(e['problems'])}" for e in t["provenance"]["excluded"]]
    if not t["provenance"]["excluded"]:
        out.append("- none")
    out.append("")
    for p in PLATFORMS:
        out += _md_core_a(p, t["core_a"][p], t["quality"])
    out += _md_core_b(t["core_b"])
    out += _md_quality(t["quality"])
    out += ["## What the evidence does not support", ""]
    out += [f"- {x['item']}: {x['reason']}" for x in t["not_supported"]]
    out += ["", "## Caveat codes", ""]
    out += [f"- {k}: {v}" for k, v in sorted(t["caveat_text"].items())]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------- CLI --

def write_outputs(out_dir: Path, table: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"decision_table.json": json.dumps(table, sort_keys=True, indent=1, ensure_ascii=False) + "\n",
            "decision_table.md": render_md(table)}
    tmp = {}
    for name, text in data.items():
        p = out_dir / f".{name}.tmp"
        p.write_text(text)
        tmp[name] = p
    for name, p in tmp.items():
        os.replace(p, out_dir / name)


def run(root: Path, pins: dict, out_dir: Path) -> int:
    try:
        table = build(root, pins)
    except DecisionTableError as exc:
        print(f"decision_table: refusing: {exc}", file=sys.stderr)
        return 2
    write_outputs(out_dir, table)
    print(json.dumps({"written": [f"{out_dir.name}/decision_table.json", f"{out_dir.name}/decision_table.md"]}))
    return 0


def main(argv: list | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)  # no selection args (021 R1)
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    return run(ROOT / "results", pins, ROOT / "results" / "decision_table")


if __name__ == "__main__":
    raise SystemExit(main())
