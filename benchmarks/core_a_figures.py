#!/usr/bin/env python3
"""Core A figures from a session's rows (SPEC Week 1 #6 and the final Core A deliverables).

memory:   llama.cpp memory breakdown (device self + host self, MiB) of the fit runs vs context,
          one line per (size, quant). The §3 peak-memory metric, not a hypothesis contrast, so it
          may be drawn from pilot data.
decode /  process-level mean decode or prefill tok/s (core_a_analysis.cell_stats) vs context, one
prefill   line per (size, quant). Drawn only from valid formal sessions (011, 013).
frontier  quality vs throughput: decode tok/s at L = 2048 (Core A) against KLD vs BF16 (quality), one
          point per configuration. Both sessions must be valid formal sessions.

Usage:
    uv run python benchmarks/core_a_figures.py memory --session results/core_a/macos-metal/<session> \\
        --platform-label "M4 Pro 48GB (Metal)" --out results/core_a/figures/<name>.svg
    uv run python benchmarks/core_a_figures.py decode|prefill --session <formal session> ...
    uv run python benchmarks/core_a_figures.py frontier --session <formal Core A> --quality-session <formal quality> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import svg_chart  # noqa: E402

import core_a_analysis as an  # noqa: E402


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


SPEED = {"decode": ("decode", "decode tok/s (process-level mean)", "llama-bench -p 0 -n 128 -d L-128, avg_ts"),
         "prefill": ("prefill", "prefill tok/s (process-level mean)", "L / TTFT per process (llama-bench -p L -n 0)")}


def speed_series(rows: list, *, metric: str) -> dict:
    key = SPEED[metric][0]
    series: dict = {}
    for (size, quant, L), c in sorted(an.cell_stats(rows).items()):
        if c["outcome"] == "ok" and c[key]:
            series.setdefault(f"{size}-{quant}", []).append((L, c[key]["m"]))
    return series


def speed_figure(rows: list, *, metric: str, platform_label: str) -> str:
    _, y_label, note = SPEED[metric]
    return svg_chart.line_chart(title=f"Core A {metric} speed vs context — {platform_label}",
                                x_label="context L (tokens, log2 scale)", y_label=y_label,
                                series=speed_series(rows, metric=metric), log2_x=True, note=note)


def frontier_points(core_a_rows: list, quality_rows: list, *, L: int = 2048) -> dict:
    cells = an.cell_stats(core_a_rows)
    kld = {(r["size"], r["point"]): r["metrics"]["kld_mean"] for r in quality_rows
           if r["kind"] == "kld" and r["outcome"] == "ok" and not r.get("variant") and not r.get("repeat_of")}
    out = {}
    for (size, quant), k in sorted(kld.items()):
        c = cells.get((size, quant, L))
        if c and c["outcome"] == "ok" and c["decode"]:
            out[f"{size}-{quant}"] = [(c["decode"]["m"], k)]
    return out


def frontier_figure(core_a_rows: list, quality_rows: list, *, platform_label: str, L: int = 2048,
                    quality_platform: str = "wsl2-cuda") -> str:
    return svg_chart.line_chart(title=f"Quality vs decode throughput — {platform_label}",
                                x_label=f"decode tok/s at L = {L}", y_label="KLD vs BF16 (WikiText-2, mean)",
                                series=frontier_points(core_a_rows, quality_rows, L=L), y_from_zero=True,
                                note="one point per configuration; lower KLD and higher tok/s are better; "
                                     f"KLD measured on {quality_platform} (020)")


def _load(session: Path) -> tuple:
    meta = json.loads((session / "session.json").read_text())
    rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines() if x.strip()]
    return meta, rows


def main(argv: list | None = None) -> int:
    import core_a_report as cr

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figure", choices=["memory", "decode", "prefill", "frontier"])
    parser.add_argument("--session", required=True)
    parser.add_argument("--quality-session")
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    session = Path(args.session)
    meta, rows = _load(session)
    if args.figure == "memory":
        plan = json.loads((session / "plan.json").read_text())
        blocks = {j["id"]: j.get("block") for j in plan["jobs"]}
        for r in rows:
            r.setdefault("block", blocks.get(r["job_id"]))
        svg = memory_figure(rows, platform_label=args.platform_label)
    else:
        problems = cr.session_problems(meta)
        if args.figure == "frontier":
            if not args.quality_session:
                problems.append("frontier needs --quality-session")
            else:
                qmeta, qrows = _load(Path(args.quality_session))
                if qmeta.get("mode") != "formal" or qmeta.get("env_valid") is not True:
                    problems.append("the quality session is not a valid formal session")
                if qmeta.get("platform", "macos-metal") != "wsl2-cuda":
                    problems.append("the quality session is not on wsl2-cuda (020: formal quality is measured there)")
        if problems:
            print("core_a_figures: refusing: " + "; ".join(problems), file=sys.stderr)
            return 2
        svg = (frontier_figure(rows, qrows, platform_label=args.platform_label) if args.figure == "frontier"
               else speed_figure(rows, metric=args.figure, platform_label=args.platform_label))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
