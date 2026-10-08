#!/usr/bin/env python3
"""Core B figures from a core_b_report.py analysis.json (SPEC final Core B deliverables:
concurrency vs TTFT / TPOT, throughput, P95). One line per context level (uniform mix), log2 C axis;
each level's C_knee (H-B1 units) is written in the note. The analysis exists only for formal sessions
(core_b_report refuses anything else).

Usage:
    python benchmarks/core_b_figures.py --analysis results/core_b/<uniform>/analysis.json \\
        --metric p95_ttft_ms|p95_tpot_ms|output_throughput --platform-label "RTX 5070 Ti 16GB" --out <svg>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import svg_chart  # noqa: E402

LABELS = {"p95_ttft_ms": ("P95 TTFT (ms)", "P95 TTFT vs concurrency"),
          "p95_tpot_ms": ("P95 TPOT (ms)", "P95 TPOT vs concurrency"),
          "output_throughput": ("output tok/s", "Output throughput vs concurrency")}


RESTARTS = 3  # = core_b_analysis.RESTARTS (007 / §2: a point is the mean of 3 valid server restarts)


def series(analysis: dict, metric: str) -> tuple:
    """Uniform-mix points with all 3 restarts valid; also the (L, C) cells left out because fewer were valid."""
    out: dict = {}
    dropped = []
    for key, cell in sorted(analysis["cells"].items()):
        mix, L, C = key.split("-")
        if mix != "uniform" or cell.get(metric) is None:
            continue
        if cell.get("n_valid") != RESTARTS:
            dropped.append(f"L={L} C={C} (n={cell.get('n_valid')})")
            continue
        out.setdefault(int(L), []).append((int(C), cell[metric]))
    return {f"L={L}": sorted(v) for L, v in sorted(out.items())}, dropped


def figure(analysis: dict, metric: str, *, platform_label: str) -> str:
    y_label, title = LABELS[metric]
    s, dropped = series(analysis, metric)
    knees = ", ".join(f"L={u['L']}: " + ("not testable" if u.get("status") == "not testable" else str(u.get("c_knee")))
                      for u in analysis["hypotheses"]["H-B1"]["units"])
    note = f"uniform mix, mean of 3 valid server restarts; C_knee {knees}"
    if dropped:
        note += "; left out (fewer valid restarts): " + ", ".join(dropped)
    return svg_chart.line_chart(title=f"Core B {title} — {platform_label}", x_label="concurrency C (log2 scale)",
                                y_label=y_label, series=s, log2_x=True, note=note)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--metric", required=True, choices=sorted(LABELS))
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    analysis = json.loads(Path(args.analysis).read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(figure(analysis, args.metric, platform_label=args.platform_label))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
