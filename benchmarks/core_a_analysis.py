#!/usr/bin/env python3
"""Pre-registered Core A decision rules (PREREGISTRATION §2 H-A1..H-A4; 004 spill as
revised by 009). Written and tested before any Core A data exists; frozen with Stage A.

Unit of analysis: the process-level mean (one llama-bench process = mean of -r 5).
Each cell has n = 3 processes: m = mean, SE = s / sqrt(n). "Beyond noise":
|x - y| > 2 sqrt(SE_x^2 + SE_y^2); ratio SE by the delta method. A comparison unit
that needs an OOM / spill / failed cell is "not testable" for that unit only.

Stdlib only, Python 3.10 compatible (also runs in WSL).
"""

from __future__ import annotations

import itertools
import math
import statistics

MiB, GiB = 2**20, 2**30
KV_PER_TOKEN = 32768  # f16, 8 full-attention layers x 4 KV heads x 256 x 2 (K, V) x 2 B (008)
S_BYTES = 50.25 * MiB  # recurrent GDN state per sequence, f32 (008)
N_GEN = 128
O_LOW, O_HIGH = 0.8 * GiB, 1.5 * GiB  # non-weight / non-KV overhead bounds (004)
H_A2_TOL = 0.20
H_A4B_THRESHOLD = 4.0
SPILL_SPEED_FRACTION = 0.5
ANCHOR = ("4B", "Q4_K_M")  # 009: smallest size, lowest bit (= the sentinel config)


def _ms(values: list) -> dict | None:
    if not values:
        return None
    n = len(values)
    m = statistics.mean(values)
    return {"m": m, "se": statistics.stdev(values) / math.sqrt(n) if n > 1 else None, "n": n}


def cell_stats(rows: list) -> dict:
    """(size, quant, L) -> outcome + decode tok/s, TTFT seconds, prefill tok/s (process-level)."""
    cells: dict = {}
    for r in rows:
        if r["kind"] not in ("fit", "ttft", "decode"):
            continue
        key = (r["size"], r["quant"], r["L"])
        c = cells.setdefault(key, {"outcomes": [], "decode": [], "ttft": [], "prefill": [], "fit": []})
        c["outcomes"].append(r["outcome"])
        if r["outcome"] != "ok":
            continue
        if r["kind"] == "decode":
            c["decode"].append(r["metrics"]["avg_ts"])
        elif r["kind"] == "ttft":
            ttft_s = r["metrics"]["avg_ns"] / 1e9
            c["ttft"].append(ttft_s)
            c["prefill"].append(key[2] / ttft_s)  # process-level L / TTFT, then the cell mean (§3; 013)
        else:
            c["fit"].append(r.get("summary"))
    out = {}
    for key, c in cells.items():
        outcome = "oom" if "oom" in c["outcomes"] else ("error" if any(o != "ok" for o in c["outcomes"]) else "ok")
        out[key] = {"outcome": outcome, "decode": _ms(c["decode"]), "ttft": _ms(c["ttft"]),
                    "prefill": _ms(c["prefill"]), "fit_summaries": [s for s in c["fit"] if s]}
    return out


def _usable(cells: dict, key, spill: set, need: tuple = ("decode",)) -> bool:
    """ok, not spilled, and every metric the comparison needs was measured (a partial cell is not testable)."""
    c = cells.get(key)
    return bool(c) and c["outcome"] == "ok" and key not in spill and all(c[m] for m in need)


def _ratio(a: dict, b: dict) -> tuple:
    r = a["m"] / b["m"]
    if a["se"] is None or b["se"] is None:
        return r, None
    return r, r * math.sqrt((a["se"] / a["m"]) ** 2 + (b["se"] / b["m"]) ** 2)


def _beyond_noise(x: float, se_x, y: float, se_y) -> bool:
    if se_x is None or se_y is None:
        return False
    return abs(x - y) > 2 * math.sqrt(se_x ** 2 + se_y ** 2)


def _verdict(units: list) -> str:
    return "refuted" if any(u.get("refutes") for u in units) else "not refuted"


# -------------------------------------------------------------------- H-A1 --

def fit_need_bytes(*, file_bytes: float, embd_bytes: float, L: int, platform: str) -> float:
    """N = file - token_embd (untied only, CUDA only; Mac unified memory does not subtract) + KV(L) + S."""
    sub = embd_bytes if platform == "wsl2-cuda" else 0
    return file_bytes - sub + KV_PER_TOKEN * L + S_BYTES


def predict_fit(*, n_bytes: float, v_total: float, b: float) -> str:
    if n_bytes + O_HIGH <= v_total - b:
        return "fit"
    if n_bytes + O_LOW > v_total - b:
        return "oom"
    return "boundary"


def h_a1(predicted: dict, observed: dict, *, spill: set) -> dict:
    """observed: cell -> 'ok' | 'oom' | 'error'. Boundary cells are reported only."""
    units, refuting = [], []
    for key in sorted(predicted):
        pred, obs = predicted[key], observed.get(key)
        state = "spill" if key in spill else {"ok": "fit"}.get(obs, obs)
        refutes = (pred == "fit" and state in ("oom", "spill")) or (pred == "oom" and state == "fit")
        units.append({"cell": list(key), "predicted": pred, "observed": state, "refutes": refutes})
        if refutes:
            refuting.append(list(key))
    return {"verdict": "refuted" if refuting else "not refuted", "refuting": refuting, "units": units}


# -------------------------------------------------------------------- H-A2 --

def decode_bytes(file_bytes: float, embd_bytes: float, L: int) -> float:
    """H-A2: (file - token_embd, untied only) + KV(L - 64) + 2 S (read + write of the recurrent state)."""
    return file_bytes - embd_bytes + KV_PER_TOKEN * (L - N_GEN // 2) + 2 * S_BYTES


def h_a2(cells: dict, file_bytes: dict, embd_bytes: dict, *, spill: set, L: int = 2048) -> dict:
    configs = sorted({(s, q) for (s, q, l) in cells if l == L})
    units = []
    for a, b in itertools.combinations(configs, 2):
        ka, kb = (*a, L), (*b, L)
        unit = {"pair": f"{a[0]}-{a[1]} / {b[0]}-{b[1]}"}
        if not (_usable(cells, ka, spill) and _usable(cells, kb, spill)):
            units.append({**unit, "status": "not testable", "refutes": False})
            continue
        observed = cells[ka]["decode"]["m"] / cells[kb]["decode"]["m"]
        predicted = decode_bytes(file_bytes[b], embd_bytes[b], L) / decode_bytes(file_bytes[a], embd_bytes[a], L)
        dev = observed / predicted - 1
        units.append({**unit, "status": "tested", "observed": observed, "predicted": predicted,
                      "deviation": dev, "refutes": abs(dev) > H_A2_TOL})
    return {"verdict": _verdict(units), "tolerance": H_A2_TOL, "units": units}


# -------------------------------------------------------------------- H-A3 --

def h_a3(cells: dict, *, spill: set, quants: tuple = ("Q8_0", "Q6_K", "Q4_K_M"),
         l_short: int = 2048, l_long: int = 32768) -> dict:
    """D = 1 - tok/s(32K) / tok/s(2K); predicted D(Q4_K_M) > D(Q6_K) > D(Q8_0). Refuted only by a
    reversal of an adjacent pair that is beyond noise; "adjacent" = neighbours in that order among the
    configs that fit at 32K (013). Fewer than 2 configs at 32K -> not testable."""
    units = []
    for size in sorted({k[0] for k in cells}):
        d = {}
        for q in quants:
            ks, kl = (size, q, l_short), (size, q, l_long)
            if _usable(cells, ks, spill) and _usable(cells, kl, spill):
                r, se = _ratio(cells[kl]["decode"], cells[ks]["decode"])
                d[q] = (1 - r, se)
        unit = {"size": size, "D": {q: v[0] for q, v in d.items()}, "D_se": {q: v[1] for q, v in d.items()}}
        if len(d) < 2:
            units.append({**unit, "status": "not testable", "refutes": False, "pairs": []})
            continue
        pairs, refutes = [], False
        present = [q for q in quants if q in d]  # neighbours among the configs that fit at 32K (013)
        for lo, hi in zip(present, present[1:]):  # predicted D(hi-bit) < D(lower-bit)
            reversed_ = d[lo][0] > d[hi][0]
            noisy = not _beyond_noise(d[lo][0], d[lo][1], d[hi][0], d[hi][1])
            pairs.append({"pair": [lo, hi], "reversed": reversed_, "within_noise": noisy})
            refutes = refutes or (reversed_ and not noisy)
        units.append({**unit, "status": "tested", "pairs": pairs, "refutes": refutes})
    return {"verdict": _verdict(units), "units": units}


# -------------------------------------------------------------------- H-A4 --

def h_a4a(cells: dict, *, spill: set, quants: tuple = ("Q8_0", "Q6_K", "Q4_K_M")) -> dict:
    units = []
    for size, L in sorted({(k[0], k[2]) for k in cells}):
        keys = [(size, q, L) for q in quants]
        if not all(_usable(cells, k, spill, need=("decode", "prefill")) for k in keys):
            units.append({"size": size, "L": L, "status": "not testable", "refutes": False})
            continue
        pre = [cells[k]["prefill"]["m"] for k in keys]
        dec = [cells[k]["decode"]["m"] for k in keys]
        sp, sd = max(pre) / min(pre), max(dec) / min(dec)
        units.append({"size": size, "L": L, "status": "tested", "spread_prefill": sp, "spread_decode": sd,
                      "refutes": sp >= sd})
    return {"verdict": _verdict(units), "units": units}


def h_a4b(cells: dict, *, spill: set, l_mid: int = 8192, l_long: int = 32768) -> dict:
    units = []
    for size, q in sorted({(k[0], k[1]) for k in cells}):
        km, kl = (size, q, l_mid), (size, q, l_long)
        if not _usable(cells, kl, spill, need=()):
            continue  # only configurations that fit at 32K are units
        if not (_usable(cells, kl, spill, need=("ttft",)) and _usable(cells, km, spill, need=("ttft",))):
            units.append({"config": f"{size}-{q}", "status": "not testable", "refutes": False})
            continue
        ratio = cells[kl]["ttft"]["m"] / cells[km]["ttft"]["m"]
        units.append({"config": f"{size}-{q}", "status": "tested", "ratio": ratio,
                      "refutes": ratio <= H_A4B_THRESHOLD})
    return {"verdict": _verdict(units), "threshold": H_A4B_THRESHOLD, "units": units}


# ------------------------------------------------------------------- spill --

def spill_cells(cells: dict, file_bytes: dict, embd_bytes: dict, params: dict) -> set:
    """RTX only (004, 009). params: v_total_mib, b_mib, c_ctx_mib (Stage B measurements).
    1. memory: max over fit runs of GPU device self + C_ctx + B > V_total;
    2. speed: decode < 0.5 x anchor decode x bytes(anchor) / bytes(cell); anchor = same-L 4B-Q4_K_M,
       or 4B-Q4_K_M @ 2048 when the same-L anchor itself spills."""
    out = set()
    for key, c in cells.items():
        selfs = [sum(d.get("self", 0) for d in (s.get("memory") or {}).get("devices", []))
                 for s in c["fit_summaries"]]
        if selfs and max(selfs) + params["c_ctx_mib"] + params["b_mib"] > params["v_total_mib"]:
            out.add(key)

    def speed_spill(key, anchor_key) -> bool:
        c, a = cells.get(key), cells.get(anchor_key)
        if not (c and a and c["decode"] and a["decode"]):
            return False
        expected = a["decode"]["m"] * (decode_bytes(file_bytes[ANCHOR], embd_bytes[ANCHOR], anchor_key[2])
                                       / decode_bytes(file_bytes[key[:2]], embd_bytes[key[:2]], key[2]))
        return c["decode"]["m"] < SPILL_SPEED_FRACTION * expected

    fallback = (*ANCHOR, min(k[2] for k in cells))
    anchors = {}
    for L in sorted({k[2] for k in cells}):
        same = (*ANCHOR, L)
        anchors[L] = fallback if (same in out or (same != fallback and speed_spill(same, fallback))) else same
    for key in cells:
        if key in out or key[:2] == ANCHOR and anchors[key[2]] == key:
            continue
        if speed_spill(key, anchors[key[2]]):
            out.add(key)
    return out
