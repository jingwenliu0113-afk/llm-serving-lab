#!/usr/bin/env python3
"""Core B analysis report (rules in core_b_analysis.py; decisions 007, 011, 013).

Inputs: the formal `uniform` session (H-B1..H-B3, and the uniform side of H-B4) and the
formal `mixed` session whose plan was derived from it (core_b_plan.py mixed). Only runs
that passed validation count. H-B4 uses each run's short-class P95 TTFT, which the runner
computed from the --save-detailed per-request data. The report refuses non-formal
sessions, a mixed plan derived from another session, or a mixed C that is not the uniform
C_knee recomputed here. Output: analysis.json (verdicts plus a per-(mix, L, C) table).

Usage:
    python benchmarks/core_b_report.py --uniform results/core_b/<uniform> --mixed results/core_b/<mixed> \\
        --out results/core_b/<uniform>/analysis.json

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import core_b_analysis as ba
import core_b_plan as bp


class ReportError(RuntimeError):
    pass


def _cells(rows: list) -> dict:
    by: dict = {}
    for r in rows:
        if r.get("ok") is True:
            by.setdefault((r["mix"], r["L"], r["C"]), []).append(r)
    out = {}
    for (mix, L, C), rs in sorted(by.items()):
        sampled = [r.get("metrics_sampled") or {} for r in rs]
        peaks = [s["kv_cache_usage_peak"] for s in sampled if s.get("kv_cache_usage_peak") is not None]
        pre = [s["preemptions"] for s in sampled if s.get("preemptions") is not None]
        out[f"{mix}-{L}-{C}"] = {
            "n_valid": len(rs),
            **{k: statistics.mean(r[k] for r in rs) if all(r.get(k) is not None for r in rs) else None
               for k in ("output_throughput", "p95_ttft_ms", "p95_tpot_ms", "short_class_p95_ttft_ms")},
            "kv_cache_usage_peak_max": max(peaks, default=None),
            "preemptions_total": sum(pre) if pre else None,
        }
    return out


def report(uniform: dict, mixed: dict | None) -> dict:
    """uniform / mixed: {"session": session.json, "plan": plan.json, "rows": rows}."""
    problems = []
    if uniform["session"].get("formal") is not True:
        problems.append("the uniform session is not formal")
    if mixed is not None:
        if mixed["session"].get("formal") is not True:
            problems.append("the mixed session is not formal")
        if mixed["plan"].get("derived_from") != uniform["session"]["session_id"]:
            problems.append(f"the mixed plan was derived from {mixed['plan'].get('derived_from')!r}, "
                            f"not {uniform['session']['session_id']!r}")
        for key in ("server", "server_env"):  # 007: one shared server config for every level and mix
            if mixed["session"].get(key) != uniform["session"].get(key):
                problems.append(f"the mixed session's {key} {mixed['session'].get(key)!r} differs from the uniform "
                                f"session's {uniform['session'].get(key)!r} (007)")
    rows, plan = uniform["rows"], uniform["plan"]
    knees = {L: ba.knee(rows, L, plan["grids"][str(L)])["c_knee"] for L in bp.MIXED_LEVELS}
    if mixed is not None:
        for L, C in mixed["plan"]["c_knee"].items():
            if knees.get(int(L)) != C:
                problems.append(f"mixed C = {C} at L = {L} is not the uniform C_knee {knees.get(int(L))}")
    if problems:
        raise ReportError("; ".join(problems))

    def short_p95(rs: list, mix: str, L: int, C) -> dict:
        return {r["restart"]: r["short_class_p95_ttft_ms"] for r in rs
                if r["mix"] == mix and r["L"] == L and r["C"] == C and r.get("ok") is True
                and r.get("short_class_p95_ttft_ms") is not None}

    per_level = {L: {"C": knees[L], "uniform": short_p95(rows, "uniform", L, knees[L]) if knees[L] else {},
                     "mixed": short_p95(mixed["rows"], "mixed", L, knees[L]) if mixed and knees[L] else {}}
                 for L in bp.MIXED_LEVELS}
    return {
        "uniform_session": uniform["session"]["session_id"],
        "mixed_session": mixed["session"]["session_id"] if mixed else None,
        "c_kv": plan["c_kv"], "grids": plan["grids"],
        "hypotheses": {
            "H-B1": ba.h_b1(rows, plan["grids"]),
            "H-B2": ba.h_b2(rows, plan["c_kv"], plan["grids"]),
            "H-B3": ba.h_b3(rows, plan["grids"]),
            "H-B4": ba.h_b4_values(per_level),
        },
        "cells": _cells(rows + (mixed["rows"] if mixed else [])),
    }


def _load(path: str) -> dict:
    d = Path(path)
    return {"session": json.loads((d / "session.json").read_text()), "plan": json.loads((d / "plan.json").read_text()),
            "rows": [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines()]}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uniform", required=True)
    parser.add_argument("--mixed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        rep = report(_load(args.uniform), _load(args.mixed) if args.mixed else None)
    except ReportError as exc:
        print(f"core_b_report: refusing: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v["verdict"] for k, v in rep["hypotheses"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
