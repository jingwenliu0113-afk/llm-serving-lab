#!/usr/bin/env python3
"""Core A analysis report for one platform session (rules in core_a_analysis.py).

Inputs: the session rows, pins.toml (file sizes), token_embd bytes per GGUF (read from
the pinned files with scripts/gguf_tensors.py), which sizes are tied (no output.weight),
and on the RTX the Stage B measurements (V_total, B, C_ctx). Output: analysis.json next
to the rows.

Usage:
    uv run python benchmarks/core_a_report.py --session results/core_a/macos-metal/<session> --models models/gguf
    uv run python benchmarks/core_a_report.py --session results/core_a/wsl2-cuda/<session> --models models/gguf \\
        --v-total-mib 16303 --b-mib <B> --c-ctx-mib <C_ctx>
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import core_a_analysis as an

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gguf_tensors  # noqa: E402

MAC_PARAMS = {"v_total_mib": 38338, "b_mib": 0}  # Metal working set (004); B = 0 on unified memory
MiB = 2**20


class ReportError(RuntimeError):
    pass


def report(rows: list, *, platform: str, pins: dict, embd_bytes: dict, tied: dict,
           platform_params: dict | None = None) -> dict:
    if platform == "macos-metal":
        params = dict(MAC_PARAMS)
    else:
        if not platform_params or any(platform_params.get(k) is None for k in ("v_total_mib", "b_mib", "c_ctx_mib")):
            raise ReportError("RTX reports need the Stage B measurements: v_total_mib, b_mib, c_ctx_mib")
        params = dict(platform_params)
    files = {(m["size"], q): f["size_bytes"] for m in pins["core_a"]["models"] for q, f in m["files"].items()}
    sub = {k: (0 if tied[k[0]] else embd_bytes[k]) for k in files}  # untied only (008)
    cells = an.cell_stats(rows)
    spill = an.spill_cells(cells, files, sub, params) if platform == "wsl2-cuda" else set()

    predicted = {}
    for (size, q, L) in cells:
        n = an.fit_need_bytes(file_bytes=files[(size, q)], embd_bytes=sub[(size, q)], L=L, platform=platform)
        predicted[(size, q, L)] = an.predict_fit(n_bytes=n, v_total=params["v_total_mib"] * MiB,
                                                 b=params["b_mib"] * MiB)
    observed = {k: c["outcome"] for k, c in cells.items()}
    hyps = {
        "H-A1": an.h_a1(predicted, observed, spill=spill),
        "H-A2": an.h_a2(cells, files, sub, spill=spill),
        "H-A3": an.h_a3(cells, spill=spill),
        "H-A4a": an.h_a4a(cells, spill=spill),
        "H-A4b": an.h_a4b(cells, spill=spill),
    }
    return {
        "platform": platform, "platform_params": params,
        "subtracted_embd_bytes": {f"{s}-{q}": v for (s, q), v in sorted(sub.items())},
        "spill": sorted(list(k) for k in spill),
        "cells": {f"{s}-{q}-{L}": {k: c[k] for k in ("outcome", "decode", "ttft", "prefill")}
                  for (s, q, L), c in sorted(cells.items())},
        "hypotheses": hyps,
    }


def session_problems(s: dict) -> list:
    """Only a valid formal session is analysed: pilots stay masked (011); an env-invalid or drifted
    session is rerun as a whole (§4, 012), so its verdicts are never computed (013)."""
    problems = []
    if s.get("mode") != "formal":
        problems.append(f"mode {s.get('mode')!r} is not formal (pilots stay masked)")
    if s.get("env_valid") is not True:
        problems.append(f"session is env-invalid (env_valid={s.get('env_valid')!r})")
    drift = (s.get("sentinel") or {}).get("drift")
    if drift is not False:
        problems.append(f"sentinel drift is {drift!r} (must be False)")
    if s.get("seed") != 20261008:
        problems.append(f"seed {s.get('seed')!r} is not the frozen 20261008 (§4)")
    if s.get("platform") == "wsl2-cuda" and (s.get("session_b") or {}).get("drift") is not False:
        problems.append(f"session-start B drift is {(s.get('session_b') or {}).get('drift')!r} (§4 step 4: must be False)")
    return problems


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--v-total-mib", type=float)
    parser.add_argument("--b-mib", type=float)
    parser.add_argument("--c-ctx-mib", type=float)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    session = Path(args.session)
    s = json.loads((session / "session.json").read_text())
    problems = session_problems(s)
    if problems:
        print(f"core_a_report: refusing {session}: " + "; ".join(problems), file=sys.stderr)
        return 2
    rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines()]
    embd, tied = {}, {}
    for m in pins["core_a"]["models"]:
        for q, f in m["files"].items():
            info = gguf_tensors.read_header(Path(args.models) / f["filename"])
            names = {t["name"]: t for t in info["tensors"]}
            embd[(m["size"], q)] = names["token_embd.weight"]["bytes"]
            tied[m["size"]] = "output.weight" not in names
    params = None
    if s["platform"] == "wsl2-cuda":
        params = {"v_total_mib": args.v_total_mib, "b_mib": args.b_mib, "c_ctx_mib": args.c_ctx_mib}
    try:
        rep = report(rows, platform=s["platform"], pins=pins, embd_bytes=embd, tied=tied, platform_params=params)
    except ReportError as exc:
        print(f"core_a_report: {exc}", file=sys.stderr)
        return 2
    rep.update({"session": s["session_id"], "mode": s["mode"], "env_valid": s["env_valid"],
                "sentinel": s.get("sentinel")})
    (session / "analysis.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v["verdict"] for k, v in rep["hypotheses"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
