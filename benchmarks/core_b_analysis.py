#!/usr/bin/env python3
"""Pre-registered Core B decision rules (PREREGISTRATION §2 H-B1..H-B4; decision 007).
Written and tested before any NVIDIA data; frozen with Stage A.

Unit = server restart (n = 3). Throughput is averaged over restarts before C_knee is
computed; P95 values are averaged over restarts. C_knee = smallest grid C whose mean
output throughput reaches 95% of the grid maximum; C_max = largest grid C.
Only runs that passed validation (row["ok"]) enter a mean; a grid C without 3 valid
restarts makes that L's unit not testable -- a failed C is never replaced by another (013).
Grids and C_KV come from the plan JSON (str(L) keys) and are normalised to int L.
Percentiles over per-request values use linear interpolation (numpy's default, as
`vllm bench serve` reports them). Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import math
import statistics

KNEE_FRACTION = 0.95
H_B1_FACTOR, H_B1_COVER = 3.0, 4.0
H_B4_MIN_INCREASE_PCT = 20.0
N_GEN = 128
RESTARTS = 3


def percentile(values: list, q: float) -> float:
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] + (v[c] - v[f]) * (k - f)


def _by_l(d: dict) -> dict:
    return {int(L): v for L, v in d.items()}


def _valid(rows: list, L: int, mix: str = "uniform") -> list:
    return [r for r in rows if r["L"] == L and r["mix"] == mix and r.get("ok") is True]


def _mean_by_c(rows: list, L: int, key: str, mix: str = "uniform") -> dict:
    by: dict = {}
    for r in _valid(rows, L, mix):
        if r.get(key) is not None:
            by.setdefault(r["C"], []).append(r[key])
    return {C: statistics.mean(v) for C, v in sorted(by.items())}


def missing_c(rows: list, L: int, grid: list, mix: str = "uniform") -> list:
    """Grid C values without a valid run in every one of the 3 restarts."""
    have: dict = {}
    for r in _valid(rows, L, mix):
        have.setdefault(r["C"], set()).add(r["restart"])
    return [C for C in sorted(grid) if len(have.get(C, ())) < RESTARTS]


def knee(rows: list, L: int, grid: list) -> dict:
    gone = missing_c(rows, L, grid)
    if gone:
        return {"c_knee": None, "c_max": max(grid), "missing_c": gone}
    thr = _mean_by_c(rows, L, "output_throughput")
    top = max(thr[C] for C in grid)
    c_knee = min(C for C in grid if thr[C] >= KNEE_FRACTION * top)
    return {"c_knee": c_knee, "c_max": max(grid), "max_throughput": top}


def h_b1(rows: list, grids: dict) -> dict:
    units = []
    for L, grid in sorted(_by_l(grids).items()):
        k = knee(rows, L, grid)
        if k["c_knee"] is None:
            units.append({"L": L, "status": "not testable", "missing_c": k["missing_c"]})
            continue
        ttft = _mean_by_c(rows, L, "p95_ttft_ms")
        base = ttft[k["c_knee"]]
        hits = [C for C in grid if C > k["c_knee"] and ttft[C] >= H_B1_FACTOR * base]
        if hits:
            status = "supported"
        elif k["c_max"] >= H_B1_COVER * k["c_knee"]:
            status = "refuted"
        else:
            status = "inconclusive"
        units.append({"L": L, "status": status, "c_knee": k["c_knee"], "first_c_at_3x": min(hits) if hits else None})
    verdict = "refuted" if any(u["status"] == "refuted" for u in units) else "not refuted"
    return {"verdict": verdict, "units": units}


def h_b2(rows: list, c_kv: dict, grids: dict) -> dict:
    units, grids = [], _by_l(grids)
    for L, ckv in sorted(_by_l(c_kv).items()):
        grid = grids[L]
        k = knee(rows, L, grid)
        if k["c_knee"] is None:
            units.append({"L": L, "status": "not testable", "missing_c": k["missing_c"]})
            continue
        interval = [ckv / 2, 2 * ckv]
        if not (min(grid) <= ckv <= max(grid)):
            units.append({"L": L, "status": "descriptive", "c_kv": ckv, "c_knee": k["c_knee"]})
            continue
        inside = interval[0] <= k["c_knee"] <= interval[1]
        units.append({"L": L, "status": "tested", "c_kv": ckv, "c_knee": k["c_knee"],
                      "interval": [x if x != int(x) else int(x) for x in interval], "refutes": not inside})
    return {"verdict": "refuted" if any(u.get("refutes") for u in units) else "not refuted", "units": units}


def h_b3(rows: list, grids: dict) -> dict:
    units = []
    for L, grid in sorted(_by_l(grids).items()):
        c_max = max(grid)
        gone = missing_c(rows, L, [1, c_max])
        if gone:
            units.append({"L": L, "status": "not testable", "missing_c": gone})
            continue
        ttft, tpot = _mean_by_c(rows, L, "p95_ttft_ms"), _mean_by_c(rows, L, "p95_tpot_ms")
        g_tpot, g_ttft = tpot[c_max] / tpot[1], ttft[c_max] / ttft[1]
        units.append({"L": L, "status": "tested", "c_max": c_max, "g_tpot": g_tpot, "g_ttft": g_ttft,
                      "refutes": g_tpot >= g_ttft})
    return {"verdict": "refuted" if any(u.get("refutes") for u in units) else "not refuted", "units": units}


def short_class_p95_ttft_ms(detailed: dict, *, L: int) -> float:
    """P95 TTFT (ms) over the short-class requests (input = L - 128) of one --save-detailed result."""
    short = [t * 1000 for t, n in zip(detailed["ttfts"], detailed["input_lens"]) if n == L - N_GEN]
    return percentile(short, 95)


def h_b4(per_level: dict) -> dict:
    """per_level: L -> {"C": uniform C_knee, "uniform": {restart: detailed}, "mixed": {restart: detailed}},
    only runs that passed validation; each mix needs all 3 restarts (013)."""
    return h_b4_values({L: {"C": d["C"], **{mix: {r: short_class_p95_ttft_ms(x, L=L) for r, x in d[mix].items()}
                                              for mix in ("uniform", "mixed")}}
                        for L, d in per_level.items()})


def h_b4_values(per_level: dict) -> dict:
    """per_level: L -> {"C", "uniform": {restart: short-class P95 TTFT ms}, "mixed": {...}}; valid runs only."""
    units = []
    for L, d in sorted(per_level.items()):
        u = [v for _, v in sorted(d["uniform"].items())]
        m = [v for _, v in sorted(d["mixed"].items())]
        if len(u) < RESTARTS or len(m) < RESTARTS:
            units.append({"L": L, "status": "not testable"})
            continue
        mu, mm = statistics.mean(u), statistics.mean(m)
        se_u, se_m = statistics.stdev(u) / math.sqrt(len(u)), statistics.stdev(m) / math.sqrt(len(m))
        inc = (mm / mu - 1) * 100
        beyond = abs(mm - mu) > 2 * math.sqrt(se_u ** 2 + se_m ** 2)
        units.append({"L": L, "status": "tested", "C": d["C"], "uniform_p95_ms": mu, "mixed_p95_ms": mm,
                      "increase_pct": inc, "beyond_noise": beyond,
                      "refutes": inc < H_B4_MIN_INCREASE_PCT or not beyond})
    return {"verdict": "refuted" if any(u.get("refutes") for u in units) else "not refuted", "units": units}
