#!/usr/bin/env python3
"""Core A figures from a session's rows (SPEC Week 1 #6: first memory / latency figure).

memory: llama.cpp memory breakdown (device self + host self, MiB) of the fit runs vs context,
one line per (size, quant). This is the §3 peak-memory metric, not a hypothesis contrast,
so it may be drawn from pilot data. Latency figures are drawn only from formal sessions.

Usage:
    uv run python benchmarks/core_a_figures.py memory --session results/core_a/macos-metal/<session> \\
        --platform-label "M4 Pro 48GB (Metal)" --out results/core_a/figures/<name>.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import svg_chart  # noqa: E402


def memory_series(rows: list, *, platform_label: str) -> dict:
    series: dict = {}
    for r in rows:
        if r["kind"] != "fit" or r["outcome"] != "ok" or r.get("block") not in (None, "memory"):
            continue
        mem = (r.get("summary") or {}).get("memory") or {}
        if not mem:
            continue
        total = sum(d.get("self", 0) for d in mem.get("devices", [])) + ((mem.get("host") or {}).get("self") or 0)
        series.setdefault(f"{r['size']}-{r['quant']}", {}).setdefault(r["L"], []).append(total)
    return {k: [(L, max(v)) for L, v in sorted(d.items())] for k, d in sorted(series.items())}


def memory_figure(rows: list, *, platform_label: str) -> str:
    return svg_chart.line_chart(
        title=f"Core A peak memory vs context — {platform_label}",
        x_label="context L (tokens, log2 scale)", y_label="MiB (device self + host self)",
        series=memory_series(rows, platform_label=platform_label), log2_x=True,
        note="llama.cpp memory breakdown of the fit run (llama-completion -c L, f16 KV, -fa on); max over processes",
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figure", choices=["memory"])
    parser.add_argument("--session", required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    session = Path(args.session)
    plan = json.loads((session / "plan.json").read_text())
    blocks = {j["id"]: j.get("block") for j in plan["jobs"]}
    rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines()]
    for r in rows:
        r.setdefault("block", blocks.get(r["job_id"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(memory_figure(rows, platform_label=args.platform_label))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
