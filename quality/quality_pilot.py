#!/usr/bin/env python3
"""Masked report of a quality pilot session (decision 005 'Pilot'; PREREGISTRATION §5).

Shows only what the pilot is for, never a hypothesis contrast:
- throughput -> estimated formal duration (calibrates the ~19.5 h estimate);
- determinism: each repeat job vs its original (identical or not);
- sanity: KLD of BF16 vs itself, Q8_0 KLD with -fa on / off (expected ~1e-3);
- SE of every KLD scaled to the formal chunk count (means of Q6_K / Q4_K_M hidden);
- downstream discordance b + c per adjacent pair -> CI half-width at the formal N
  (005: "the pilot script prints only b + c and N, never an accuracy");
- H-A5(a): SD of the paired per-chunk NLL difference -> CI half-width (no mean, no sign).

Usage:
    uv run python quality/quality_pilot.py --session results/quality/<pilot session>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tomllib
from pathlib import Path

PAIRS = (("BF16", "Q8_0"), ("Q8_0", "Q6_K"), ("Q6_K", "Q4_K_M"))
MC_EXPECTED_KEY = {"hellaswag": "hellaswag_tasks", "mmlu": "mmlu_tasks", "arc": "arc_tasks",
                   "winogrande": "winogrande_tasks"}


def _main_rows(rows: list) -> list:
    return [r for r in rows if not r.get("variant") and not r.get("repeat_of")]


def determinism(rows: list) -> list:
    by_id = {r["job_id"]: r for r in rows}
    out = []
    for r in rows:
        if r.get("repeat_of"):
            orig = by_id.get(r["repeat_of"])
            same = (orig is not None and r["outcome"] == "ok" and orig["outcome"] == "ok"
                    and r["metrics"] == orig["metrics"])
            out.append({"job": r["job_id"], "of": r["repeat_of"], "identical": same})
    return out


def discordance(rows: list, n_formal: dict) -> list:
    cells = {}
    for r in _main_rows(rows):
        if r["kind"] in ("mc", "winogrande") and r["outcome"] == "ok":
            cells[(r["task"], r["size"], r["point"])] = r["metrics"]["correct"]
    out = []
    for (task, size, point) in sorted(cells):
        for a, b in PAIRS:
            if point != a or (task, size, b) not in cells:
                continue
            x, y = cells[(task, size, a)], cells[(task, size, b)]
            if len(x) != len(y) or not x:
                continue
            disc = sum(i != j for i, j in zip(x, y))
            rate = disc / len(x)
            n_f = n_formal.get(task)
            out.append({"task": task, "size": size, "pair": [a, b], "n": len(x), "b_plus_c": disc, "rate": rate,
                        # paired-difference 95% CI half-width at the formal N, in percentage points
                        "ci_halfwidth_pp_formal": 196 * math.sqrt(rate / n_f) if n_f else None})
    return out


def report(rows: list, *, pins_expected: dict, pilot: dict) -> dict:
    main = _main_rows(rows)
    chunk_scale = pins_expected["wikitext_chunks"] / pilot["chunks"]

    sanity = {"kld_self": {}, "q8_0_kld": {}}
    se_formal: dict = {}
    for r in rows:
        if r["outcome"] != "ok":
            continue
        m = r.get("metrics") or {}
        if r["kind"] == "kld_self":
            sanity["kld_self"][r["size"]] = m.get("kld_mean")
        if r["kind"] == "kld" and r["point"] == "Q8_0" and not r.get("repeat_of"):
            key = "fa_off" if r.get("variant") == "fa_off" else "fa_on"
            sanity["q8_0_kld"].setdefault(r["size"], {})[key] = m.get("kld_mean")
        if r["kind"] == "kld" and not r.get("variant") and not r.get("repeat_of") and m.get("kld_mean_se"):
            # only the largest SE per size: per-quant SEs track the KLD means and would reveal the
            # H-A5(b) ordering (013); the max bounds the z-test denominators for the power check
            se = m["kld_mean_se"] / math.sqrt(chunk_scale)
            se_formal[r["size"]] = {"max": max(se, se_formal.get(r["size"], {}).get("max", 0.0))}

    ppl = {(r["size"], r["point"]): r["metrics"]["chunk_nll"] for r in main
           if r["kind"] == "ppl" and r["outcome"] == "ok"}
    ha5 = {"n_chunks": None, "sd_chunk_diff": None, "ci_halfwidth_formal": None}
    if len(ppl) == 2:
        (_, a), (_, b) = sorted(ppl.items())
        if len(a) == len(b) and len(a) > 1:
            sd = statistics.stdev(x - y for x, y in zip(a, b))
            ha5 = {"n_chunks": len(a), "sd_chunk_diff": sd,
                   "ci_halfwidth_formal": 1.96 * sd / math.sqrt(pins_expected["wikitext_chunks"])}

    secs = 0.0
    for r in main:
        if r.get("wall_s") is None:
            continue
        if r["kind"] in ("kld_base", "kld_self", "kld", "ppl"):
            secs += r["wall_s"] * chunk_scale
        elif r["kind"] in ("mc", "winogrande"):
            secs += r["wall_s"] * pins_expected[MC_EXPECTED_KEY[r["task"]]] / pilot["tasks"]

    return {
        "throughput": {"formal_hours_est": round(secs / 3600, 2),
                       "method": "pilot wall time scaled by formal / pilot chunks or tasks (sizes in this pilot)"},
        "determinism": determinism(rows),
        "sanity": sanity,
        "kld_se_formal_est": se_formal,
        "discordance": discordance(rows, {t: pins_expected[k] for t, k in MC_EXPECTED_KEY.items()}),
        "ha5_paired_nll": ha5,
        "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in ("ok", "oom", "error", "skipped")},
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    import run_quality as rq

    root = Path(__file__).resolve().parent.parent
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    session = Path(args.session)
    rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines()]
    rep = report([r for r in rows if r["kind"] != "delete_kld"], pins_expected=pins["quality"]["expected"],
                 pilot=rq.PILOT)
    (session / "pilot_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
