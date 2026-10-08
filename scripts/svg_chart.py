"""Minimal stdlib SVG line charts for the result figures (no plotting dependency)."""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

W, H = 720, 440
LEFT, RIGHT, TOP, BOTTOM = 80, 170, 50, 60
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f")


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def line_chart(*, title: str, x_label: str, y_label: str, series: dict, log2_x: bool = False,
               y_from_zero: bool = True, note: str | None = None) -> str:
    pts = [p for s in series.values() for p in s]
    xs = sorted({x for x, _ in pts})
    fx = (lambda v: math.log2(v)) if log2_x else (lambda v: v)
    x_lo, x_hi = fx(min(xs)), fx(max(xs))
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 1, x_hi + 1
    y_vals = [y for _, y in pts]
    y_ticks = _nice_ticks(0 if y_from_zero else min(y_vals), max(y_vals))
    y_lo, y_hi = y_ticks[0], y_ticks[-1]
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    sx = lambda v: LEFT + (fx(v) - x_lo) / (x_hi - x_lo) * pw  # noqa: E731
    sy = lambda v: TOP + ph - (v - y_lo) / (y_hi - y_lo) * ph  # noqa: E731

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W / 2}" y="24" text-anchor="middle" font-size="15">{escape(title)}</text>',
           f'<line x1="{LEFT}" y1="{TOP + ph}" x2="{LEFT + pw}" y2="{TOP + ph}" stroke="black"/>',
           f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + ph}" stroke="black"/>']
    for x in xs:
        out.append(f'<line x1="{sx(x):.1f}" y1="{TOP + ph}" x2="{sx(x):.1f}" y2="{TOP + ph + 5}" stroke="black"/>')
        out.append(f'<text x="{sx(x):.1f}" y="{TOP + ph + 18}" text-anchor="middle">{_fmt(x)}</text>')
    for y in y_ticks:
        out.append(f'<line x1="{LEFT}" y1="{sy(y):.1f}" x2="{LEFT + pw}" y2="{sy(y):.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{LEFT - 8}" y="{sy(y) + 4:.1f}" text-anchor="end">{_fmt(y)}</text>')
    out.append(f'<text x="{LEFT + pw / 2}" y="{H - 18}" text-anchor="middle">{escape(x_label)}</text>')
    out.append(f'<text x="18" y="{TOP + ph / 2}" text-anchor="middle" '
               f'transform="rotate(-90 18 {TOP + ph / 2})">{escape(y_label)}</text>')
    for i, (name, s) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in sorted(s))
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        for x, y in s:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>')
        ly = TOP + 14 + 18 * i
        out.append(f'<line x1="{LEFT + pw + 15}" y1="{ly - 4}" x2="{LEFT + pw + 35}" y2="{ly - 4}" '
                   f'stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{LEFT + pw + 40}" y="{ly}">{escape(name)}</text>')
    if note:
        out.append(f'<text x="{LEFT}" y="{H - 4}" font-size="10" fill="#555">{escape(note)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
