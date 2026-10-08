#!/usr/bin/env python3
"""Pre-registered Core B decision rules (PREREGISTRATION §2 H-B1..H-B4; decision 007).
Written and tested before any NVIDIA data; frozen with Stage A.

Unit = server restart (n = 3). Throughput is averaged over restarts before C_knee is
computed; P95 values are averaged over restarts. C_knee = smallest grid C whose mean
output throughput reaches 95% of the grid maximum; C_max = largest grid C.
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


def percentile(values: list, q: float) -> float:
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] + (v[c] - v[f]) * (k - f)


def _mean_by_c(rows: list, L: int, key: str, mix: str = "uniform") -> dict:
    by: dict = {}
    for r in rows:
        if r["L"] == L and r["mix"] == mix and r.get(key) is not None:
            by.setdefault(r["C"], []).append(r[key])
    return {C: statistics.mean(v) for C, v in sorted(by.items())}


def knee(rows: list, L: int) -> dict:
    thr = _mean_by_c(rows, L, "output_throughput")
    if not thr:
        return {"c_knee": None, "c_max": None}
    top = max(thr.values())
    c_knee = min(C for C, t in thr.items() if t >= KNEE_FRACTION * top)
    return {"c_knee": c_knee, "c_max": max(thr), "max_throughput": top}


def h_b1(rows: list, levels: list) -> dict:
    units = []
    for L in levels:
        k = knee(rows, L)
        ttft = _mean_by_c(rows, L, "p95_ttft_ms")
        if k["c_knee"] is None or k["c_knee"] not in ttft:
            units.append({"L": L, "status": "not testable"})
            continue
        base = ttft[k["c_knee"]]
        hits = [C for C, t in ttft.items() if C > k["c_knee"] and t >= H_B1_FACTOR * base]
        if hits:
            status = "supported"
        elif max(ttft) >= H_B1_COVER * k["c_knee"]:
            status = "refuted"
        else:
            status = "inconclusive"
        units.append({"L": L, "status": status, "c_knee": k["c_knee"], "first_c_at_3x": min(hits) if hits else None})
    verdict = "refuted" if any(u["status"] == "refuted" for u in units) else "not refuted"
    return {"verdict": verdict, "units": units}


def h_b2(rows: list, c_kv: dict) -> dict:
    units = []
    for L, ckv in sorted(c_kv.items()):
        k = knee(rows, L)
        grid = sorted(_mean_by_c(rows, L, "output_throughput"))
        if k["c_knee"] is None:
            units.append({"L": L, "status": "not testable"})
            continue
        interval = [ckv / 2, 2 * ckv]
        if not grid or not (grid[0] <= ckv <= grid[-1]):
            units.append({"L": L, "status": "descriptive", "c_kv": ckv, "c_knee": k["c_knee"]})
            continue
        inside = interval[0] <= k["c_knee"] <= interval[1]
        units.append({"L": L, "status": "tested", "c_kv": ckv, "c_knee": k["c_knee"],
                      "interval": [x if x != int(x) else int(x) for x in interval], "refutes": not inside})
    return {"verdict": "refuted" if any(u.get("refutes") for u in units) else "not refuted", "units": units}


def h_b3(rows: list, levels: list) -> dict:
    units = []
    for L in levels:
        ttft, tpot = _mean_by_c(rows, L, "p95_ttft_ms"), _mean_by_c(rows, L, "p95_tpot_ms")
        if 1 not in ttft or 1 not in tpot:
            units.append({"L": L, "status": "not testable"})
            continue
        c_max = max(ttft)
        g_tpot, g_ttft = tpot[c_max] / tpot[1], ttft[c_max] / ttft[1]
        units.append({"L": L, "status": "tested", "c_max": c_max, "g_tpot": g_tpot, "g_ttft": g_ttft,
                      "refutes": g_tpot >= g_ttft})
    return {"verdict": "refuted" if any(u.get("refutes") for u in units) else "not refuted", "units": units}


def short_class_p95_ttft_ms(detailed: dict, *, L: int) -> float:
    """P95 TTFT (ms) over the short-class requests (input = L - 128) of one --save-detailed result."""
    short = [t * 1000 for t, n in zip(detailed["ttfts"], detailed["input_lens"]) if n == L - N_GEN]
    return percentile(short, 95)


def h_b4(per_level: dict) -> dict:
    """per_level: L -> {"C": uniform C_knee, "uniform": {restart: detailed}, "mixed": {restart: detailed}}."""
    units = []
    for L, d in sorted(per_level.items()):
        u = [short_class_p95_ttft_ms(x, L=L) for _, x in sorted(d["uniform"].items())]
        m = [short_class_p95_ttft_ms(x, L=L) for _, x in sorted(d["mixed"].items())]
        if len(u) < 2 or len(m) < 2:
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
