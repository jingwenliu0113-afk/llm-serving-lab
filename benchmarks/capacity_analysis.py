#!/usr/bin/env python3
"""Pre-registered Stage C (C1) analysis: maximum closed-loop concurrency under a fixed SLO (decision 030 8.1-8.6;
PREREGISTRATION §8). Written and tested before any C1 data; frozen with freeze.capacity.

Inputs: the session {"session", "plan", "confirm_plan", "rows"} (run_capacity) and the per-request derived data
{tag: [{"idx", "cls", "ttft_ms", "latency_ms", "out"}]} (derived/<tag>-requests.jsonl, 030 8.7).
- session["boots"]: [{"boot" (0..n-1 in session order), "phase" ("sentinel" | "S" | "C"), "workload", "block"
  (Phase C r, else None), "attempt" (Phase S 1 | 2, else None), "position" (sentinel "start" | "end", else None),
  "reached_first_bench"}]; session["confirm_plan_sha256"], recorded before Phase C.
- a row per bench: {"phase", "workload", "L", "mix", "C", "N", "block", "position", "boot", "tag"
  (capacity_plan.run_tag), "valid" (the runner's 8.6 (a)-(g); (g) no spill, 031, which capacity_report recomputes
  from the row's "spill" record), "reasons", "timeout_s", "output_throughput",
  "metrics" (run_core_b.MetricsSampler), "search_pass" (Phase S, search_bits) | "outcome" (Phase C,
  phase_c_outcome)}: the runner decides with the same functions the replay re-runs. A Phase S boot that fails
  before its first bench has no row (it is retried once at the end of Phase S); a failed sentinel or Phase C boot
  records each of its planned runs as an invalid row (013: never replaced).

Per run (030 8.1), over W = idx in [C, N - C) (W' = [2C, N - C) for WINDOW_SENSITIVE, all requests for C^full):
- valid = the runner's verdict and clean derived data: N requests with idx 0..N-1 once, out = 128, cls short / long
  with N // 10 long (mixed, 007) or none (uniform), and at least one long request in W (mixed, 8.6 (e)).
- Y_short = P95 TTFT of the short class, Y_TPOT = P95 over every request of (latency - ttft) / (out - 1), Y_long =
  P95 TTFT of the 32,640-token class (mixed); core_b_analysis.percentile (linear). R_j = Y_j / S_j with S_short the
  L tier, S_long the 32768 tier (mixed_long "budget"; "best_effort" leaves it unjudged), S_TPOT for every request.
- maxR_run = max(R_short, R_TPOT). A run FAILs when maxR_run > 1; that is the "valid FAIL" of the Phase C skip rule
  (8.4 names it beside "an invalid run": a run, not the point's class), and a Phase C run's own conclusion ("met" or
  not, STRADDLE) reads only it: Y_long is pooled because one run has only about 18-20 long requests, so the pooled long
  class never triggers a skip and is judged once over the point's runs.
- A Phase S pass (navigation only, 8.4) also needs the run's own R_long <= 1. C^_g's band reads maxR_run <= 1 - m only
  (8.4 step 3): a mixed run whose own R_long fails is still inside it; such a run lies above C^, so K is unchanged
  (c_low uses C^_g only when C^_g < C^ - 1), while the recorded C^_g may then exceed C^.
- Every comparison is exact (Fraction): R = 1 is not a FAIL and R = 1 - m is inside the band.
Per point C of K over its n (030 8.1), FAIL first: FAIL if a valid run has maxR_run > 1 or the long class pooled over
every valid run has R_long > 1; else NT with fewer than n valid runs (or, mixed, no pooled long value under the
window); PASS if every run's R_short, R_TPOT and the pooled R_long are <= 1 - m; else MARGINAL.
Estimands over K ascending (downward closure): C_SLO = the largest c with every c' <= c in K PASS or MARGINAL; C_PROD
the same with PASS only. Status: CENSORED (c = C_cap) before EXACT (c + 1 in K and FAIL; for C_PROD FAIL or
MARGINAL) before LOWER (c + 1 untested or NT); without such a c: INFEASIBLE (value 0) if K's lowest point is 1 and
FAILs, UNSTABLE if it is above 1 and FAILs, else NT. C_PROD reads MARGINAL as FAIL in all of these rows, as in its EXACT
row (030 implementation note), so a MARGINAL C 1 is a verified 0 (refuse the class), never NT.
Flags: NONMONOTONE (a PASS / MARGINAL point above a FAIL point), STRADDLE (valid runs at one C disagree on FAIL, i.e.
maxR_run > 1 in some and not in others), STRICT_COARSE (C_PROD < C_SLO - 1), SEARCH_MISMATCH (|C_SLO - C^| > 1),
WINDOW_SENSITIVE (value or status of C_SLO or C_PROD differs under W'), ABOVE_CKV (C_PROD > C_KV(L), also for mixed).
Descriptive only: C^full and the W' estimands, joint per-request attainment, pooled P95s, the first-wave maximum, the
binding terms (those above 1 at the smallest FAIL point of K), margins (1 - worst R at C_SLO / C_PROD), KV peak,
preemptions and waiting peak, the 1 Hz MetricsSampler check running + waiting <= C (8.6: inflight_peak, inflight_le_c
per run, per point and per sentinel), C_SLO beside C_knee of its L (8.5: 25 / 8), the secondary SLOs S1 / S3 on the
same runs (8.2), and under C0 branch B / C the per-run slip count of 028 (slip_episodes) for the plan's c0_covariate
workloads (8.1: C0's marker recorded as a covariate; uniform only, where 028 defines it).
Hypotheses (030 8.5, rules, no p-values; per unit, only S2's predictions are frozen):
- H-S1 (L in 2048 / 8192): NT if either side is NT / UNSTABLE or uniform is INFEASIBLE; supported if mixed (EXACT, or
  0 if INFEASIBLE) < uniform (EXACT / LOWER / CENSORED); refuted if mixed (EXACT / LOWER, as 030's table lists them)
  >= uniform EXACT; else inconclusive (overlapping bounds, and the cases the table leaves open, e.g. a CENSORED mixed).
- H-S2 (U2K, U8K): NT if NT / UNSTABLE; refuted if CENSORED or an EXACT / LOWER value > C_KV; supported if EXACT or
  INFEASIBLE <= C_KV; else (a lower bound <= C_KV) inconclusive.
Session checks this module owns (8.6; the report adds env, freeze and sha256 refusals): the sentinel gate (output
throughput and W-mean TTFT of the end sentinel within the tolerance of the start's, |end / start - 1| <= 0.03) and
the replay (replay_problems: every search decision, K, n, skip and order recomputed from the rows; any mismatch makes
the session invalid). CROSS_SESSION_SHIFT (> 5 % from the Core / Advanced references) is descriptive.

Usage (capacity_report runs the refusals first and writes analysis.json):
    import capacity_analysis as ca
    analysis = ca.analyze({"session": ..., "plan": ..., "confirm_plan": ..., "rows": [...]}, requests)

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import statistics
from fractions import Fraction

import capacity_plan as cp
import core_b_analysis as ba
import core_b_plan as bp

SCHEMA_VERSION = 1
STAGE = cp.STAGE
CLASSES = ("FAIL", "NT", "PASS", "MARGINAL")
ACCEPT = {"c_slo": ("PASS", "MARGINAL"), "c_prod": ("PASS",)}
EXACT_NEXT = {"c_slo": ("FAIL",), "c_prod": ("FAIL", "MARGINAL")}
STATUSES = ("EXACT", "LOWER", "CENSORED", "INFEASIBLE", "UNSTABLE", "NT")
FLAGS = ("NONMONOTONE", "STRADDLE", "STRICT_COARSE", "SEARCH_MISMATCH", "WINDOW_SENSITIVE", "ABOVE_CKV")
WINDOWS = ("steady", "late", "full")
TERMS = ("short", "tpot", "long")
PCT = 95
SLIP_MS = 170.0                     # 028: a request > 170 ms above its own position's baseline has slipped
CROSS_SESSION_LIMIT = 0.05
C_KNEE = {"2048": 25, "8192": 8}    # 030 8.5 (descriptive): Core's C_knee (024: core_b_analysis.knee, uniform session)
H_S1_PAIRS = {"2048": ("M2K", "U2K"), "8192": ("M8K", "U8K")}
H_S2_UNITS = ("U2K", "U8K")
PREDICTIONS = {"H-S1": {"2048": "supported", "8192": "supported"},   # 030 8.5, frozen for the chosen SLO (S2) only
               "H-S2": {"U2K": "supported", "U8K": "supported"}}
NOT_SUPPORTED = (
    "uniform-32K (option O-32K, not chosen)",
    "--long-prefill-token-threshold 512 (option O-T, not chosen; a separate argv fingerprint)",
    "open-loop or variable-length traffic, outputs other than 128 tokens (8.9)",
    "the start-up burst (idx < C) is outside the estimand; only its maximum is reported (D1)",
    "queueing before admission (client semaphore wait is excluded from TTFT)",
)


class AnalysisError(ValueError):
    pass


# ---------------------------------------------------------------- per run (030 8.1)

def window_bounds(kind: str, C: int, N: int) -> tuple:
    if kind == "steady":
        return C, N - C
    if kind == "late":
        return 2 * C, N - C
    if kind == "full":
        return 0, N
    raise AnalysisError(f"unknown window {kind!r}; 030 uses {WINDOWS}")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def derived_problems(reqs, *, mix: str, C: int, N: int) -> list:
    """030 8.6 (e) on one run's derived requests; [] when they are complete and consistent."""
    if reqs is None:
        return ["no derived requests file"]
    if not all(isinstance(q, dict) and isinstance(q.get("idx"), int) for q in reqs):
        return ["derived requests without an integer idx"]
    problems = []
    if sorted(q["idx"] for q in reqs) != list(range(N)):
        problems.append(f"idx is not 0..{N - 1} exactly once ({len(reqs)} requests)")
    if not all(_num(q.get(k)) for q in reqs for k in ("ttft_ms", "latency_ms", "out")):
        problems.append("ttft_ms / latency_ms / out missing or not numbers")
    if any(q.get("out") != bp.N_GEN for q in reqs):
        problems.append(f"out not all {bp.N_GEN}")
    classes = [q.get("cls") for q in reqs]
    if any(c not in ("short", "long") for c in classes):
        problems.append("cls not all short / long")
    want = bp.mixed_counts(N)[1] if mix == "mixed" else 0
    if classes.count("long") != want:
        problems.append(f"{classes.count('long')} long requests, the {mix} mix has {want} (007)")
    lo, hi = window_bounds("steady", C, N)
    in_w = [c for q, c in zip(reqs, classes) if lo <= q["idx"] < hi]
    if "short" not in in_w:
        problems.append("no short request in W")
    if mix == "mixed" and "long" not in in_w:
        problems.append("no long request in W (030 8.6 (e))")
    return problems


def run_valid(row: dict, reqs, *, mix: str, C: int, N: int) -> tuple:
    """(valid, reasons): the runner's 8.6 verdict, then the derived data (an invalid run is never evidence)."""
    if row.get("valid") is not True:
        return False, ["runner: " + "; ".join(row.get("reasons") or ["not valid"])]
    problems = derived_problems(reqs, mix=mix, C=C, N=N)
    return not problems, problems


def thresholds(slo: dict, L: int, mix: str) -> dict:
    """S_j in ms: the L tier for short requests, the 32K tier for a mixed workload's long ones (budget), TPOT for
    every request."""
    t = slo["ttft_p95_ms"]
    long_ = t[str(cp.LONG_L)] if mix == "mixed" and slo["mixed_long"] == "budget" else None
    return {"short": t[str(L)], "tpot": slo["tpot_p95_ms"], "long": long_}


def _over(y: float, s: int, bound: Fraction) -> bool:
    """y / s > bound, exactly."""
    return Fraction(y) > s * bound


def _p95(values: list):
    return ba.percentile(values, PCT) if values else None


def _tpot_ms(q: dict) -> float:
    return (q["latency_ms"] - q["ttft_ms"]) / (q["out"] - 1)


def _band(guard_band: float) -> Fraction:
    return 1 - Fraction(str(guard_band))


def run_eval(reqs: list, *, L: int, mix: str, C: int, N: int, slo: dict, guard_band: float,
             window: str = cp.WINDOW) -> dict:
    """Y, R and the pass / band bits of one valid run under one SLO and window (keys with "_" are the raw values the
    point pools)."""
    lo, hi = window_bounds(window, C, N)
    W = [q for q in reqs if lo <= q["idx"] < hi]
    thr, band = thresholds(slo, L, mix), _band(guard_band)
    short = [q["ttft_ms"] for q in W if q["cls"] == "short"]
    long_ = [q["ttft_ms"] for q in W if q["cls"] == "long"]
    tpot = [_tpot_ms(q) for q in W]
    Y = {"short": _p95(short), "tpot": _p95(tpot), "long": _p95(long_) if mix == "mixed" else None}
    R, over, over_band = {}, {}, {}
    for k in TERMS:
        judged = thr[k] is not None and Y[k] is not None
        R[k] = Y[k] / thr[k] if judged else None
        over[k] = _over(Y[k], thr[k], Fraction(1)) if judged else None
        over_band[k] = _over(Y[k], thr[k], band) if judged else None
    fail = bool(over["short"] or over["tpot"])

    def met(q):   # joint per-request attainment (descriptive): its TTFT tier (none: best_effort long) and TPOT
        tier = thr["long"] if q["cls"] == "long" else thr["short"]
        return (tier is None or q["ttft_ms"] <= tier) and _tpot_ms(q) <= thr["tpot"]

    burst = [q["ttft_ms"] for q in reqs if q["idx"] < C]
    rs = [v for v in (R["short"], R["tpot"]) if v is not None]
    # fail: 8.1's per-run FAIL (maxR_run > 1). search_pass: 8.4's Phase S pass, which alone also reads the run's own
    # R_long. within_band: 8.4 step 3's maxR <= 1 - m, maxR_run's terms only.
    return {"window": window, "Y": Y, "R": R, "over": over, "over_band": over_band, "max_r": max(rs) if rs else None,
            "fail": fail, "search_pass": not fail and not over["long"],
            "within_band": not (over_band["short"] or over_band["tpot"]),
            "attainment": sum(map(met, W)) / len(W) if W else None, "burst_max_ttft_ms": max(burst) if burst else None,
            "n": {"short": len(short), "long": len(long_), "all": len(W)},
            "_short_ms": short, "_tpot_ms": tpot, "_long_ms": long_}


def slip_episodes(reqs: list, C: int) -> int:
    """028's per-run slip count, the C0 branch B / C covariate (descriptive): b_j = median TTFT over idx >= C with
    idx = j (mod C); idx >= C is slipped when TTFT - b_(idx mod C) > 170 ms; a slip is a maximal run of consecutive
    slipped idx. Re-implemented here because the frozen Stage C modules import only Core / Advanced code (030 8.6)."""
    t = [q["ttft_ms"] for q in sorted(reqs, key=lambda q: q["idx"])]
    base = [ba.percentile(t[C + j::C], 50) for j in range(C)]
    slipped = [i for i in range(C, len(t)) if t[i] - base[i % C] > SLIP_MS]
    return sum(1 for k, i in enumerate(slipped) if k == 0 or slipped[k - 1] != i - 1)


# ---------------------------------------------------------------- per point (030 8.1)

def classify(evals: list, *, n: int, L: int, mix: str, slo: dict, guard_band: float) -> dict:
    """Class of one point from the evals of its valid runs (FAIL first), with the pooled and worst-run values."""
    thr, band = thresholds(slo, L, mix), _band(guard_band)
    pooled = {k: [x for e in evals for x in e[f"_{k}_ms"]] for k in TERMS}
    y_long = _p95(pooled["long"]) if mix == "mixed" else None
    long_judged = thr["long"] is not None
    long_over = long_judged and y_long is not None and _over(y_long, thr["long"], Fraction(1))
    over = {k: any(e["over"][k] for e in evals) for k in ("short", "tpot")}
    over["long"] = bool(long_over)
    if over["short"] or over["tpot"] or long_over:
        klass = "FAIL"
    elif len(evals) < n or (long_judged and y_long is None):
        klass = "NT"
    elif not any(e["over_band"]["short"] or e["over_band"]["tpot"] for e in evals) \
            and not (long_judged and _over(y_long, thr["long"], band)):
        klass = "PASS"
    else:
        klass = "MARGINAL"

    def worst(k):
        vals = [e["R"][k] for e in evals if e["R"][k] is not None]
        return max(vals) if vals else None

    r_long = y_long / thr["long"] if long_judged and y_long is not None else None
    # a run's own conclusion is 8.1's per-run FAIL (maxR_run > 1), never its own long P95 (pooled above): "met in k of
    # n" (8.1's rare modes) and STRADDLE read it
    return {"class": klass, "n_valid": len(evals), "over": over,
            "worst_R": {"short": worst("short"), "tpot": worst("tpot"), "long": r_long},
            "pooled": {"Y_short": _p95(pooled["short"]), "Y_tpot": _p95(pooled["tpot"]), "Y_long": y_long,
                       "R_long": r_long},
            "met": sum(1 for e in evals if not e["fail"]),
            "conclusions": sorted({_conclusion(e) for e in evals})}


def _conclusion(e: dict) -> str:
    return "fail" if e["fail"] else "pass"


# ---------------------------------------------------------------- estimands, flags (030 8.1)

def estimand(classes: dict, *, c_cap: int, which: str) -> dict:
    """C_SLO (which = "c_slo") or C_PROD ("c_prod") over K = classes' keys, with its status."""
    K, value = sorted(classes), None
    for c in K:
        if classes[c] not in ACCEPT[which]:
            break
        value = c
    if value is None:
        if K and classes[K[0]] in EXACT_NEXT[which]:     # FAIL; for C_PROD also MARGINAL (030 implementation note)
            return {"value": 0, "status": "INFEASIBLE"} if K[0] == 1 else {"value": None, "status": "UNSTABLE"}
        return {"value": None, "status": "NT"}
    if value == c_cap:
        return {"value": value, "status": "CENSORED"}
    if classes.get(value + 1) in EXACT_NEXT[which]:
        return {"value": value, "status": "EXACT"}
    return {"value": value, "status": "LOWER"}


def estimands(classes: dict, *, c_cap: int) -> dict:
    return {which: estimand(classes, c_cap=c_cap, which=which) for which in ("c_slo", "c_prod")}


def _pairs(est: dict) -> dict:
    return {k: (est[k]["value"], est[k]["status"]) for k in ("c_slo", "c_prod")}


def flags(classes: dict, est: dict, *, conclusions: dict, c_hat, c_kv: int, late: dict) -> list:
    K = sorted(classes)
    slo_v, prod_v = est["c_slo"]["value"], est["c_prod"]["value"]
    on = {
        "NONMONOTONE": any(classes[a] == "FAIL" and classes[b] in ACCEPT["c_slo"]
                           for i, a in enumerate(K) for b in K[i + 1:]),
        "STRADDLE": any(len(set(v)) > 1 for v in conclusions.values()),
        "STRICT_COARSE": slo_v is not None and prod_v is not None and prod_v < slo_v - 1,
        "SEARCH_MISMATCH": slo_v is not None and c_hat is not None and abs(slo_v - c_hat) > 1,
        "WINDOW_SENSITIVE": _pairs(late) != _pairs(est),
        "ABOVE_CKV": prod_v is not None and prod_v > c_kv,
    }
    return [f for f in FLAGS if on[f]]


# ---------------------------------------------------------------- hypotheses (030 8.5)

def h_s1(mixed: dict, uniform: dict) -> str:
    """C_SLO(M_L) < C_SLO(U_L)."""
    if mixed["status"] in ("NT", "UNSTABLE") or uniform["status"] in ("NT", "UNSTABLE", "INFEASIBLE"):
        return "NT"
    m, u = mixed["value"], uniform["value"]
    if mixed["status"] in ("EXACT", "INFEASIBLE") and uniform["status"] in ("EXACT", "LOWER", "CENSORED") and m < u:
        return "supported"
    if mixed["status"] in ("EXACT", "LOWER") and uniform["status"] == "EXACT" and m >= u:   # 030 8.5's refuted row
        return "refuted"
    return "inconclusive"


def h_s2(uniform: dict, *, c_kv: int) -> str:
    """C_SLO(U_L) <= C_KV(L)."""
    status, value = uniform["status"], uniform["value"]
    if status in ("NT", "UNSTABLE"):
        return "NT"
    if status == "CENSORED" or (status in ("EXACT", "LOWER") and value > c_kv):
        return "refuted"
    if status in ("EXACT", "INFEASIBLE"):
        return "supported"
    return "inconclusive"


def hypotheses(c_slo: dict, c_kv: dict, *, predictions: dict | None) -> dict:
    """c_slo / c_kv: workload -> estimand / C_KV. predictions: PREDICTIONS for the chosen SLO, None for S1 / S3."""
    out = {"H-S1": {}, "H-S2": {}}
    for L, (m, u) in H_S1_PAIRS.items():
        if m in c_slo and u in c_slo:
            out["H-S1"][L] = {"mixed": c_slo[m], "uniform": c_slo[u], "verdict": h_s1(c_slo[m], c_slo[u])}
    for w in H_S2_UNITS:
        if w in c_slo:
            out["H-S2"][w] = {"c_slo": c_slo[w], "c_kv": c_kv[w], "verdict": h_s2(c_slo[w], c_kv=c_kv[w])}
    if predictions:
        for h, units in out.items():
            for unit, d in units.items():
                d["prediction"] = predictions[h][unit]
    return out


# ---------------------------------------------------------------- sentinel (030 8.4 / 8.6)

def inflight(row) -> dict:
    """030 8.6 (descriptive only): the run's 1 Hz MetricsSampler peak of running + waiting (run_capacity's
    metrics.inflight_peak) and whether it stayed <= C; None without samples."""
    peak, C = ((row or {}).get("metrics") or {}).get("inflight_peak"), (row or {}).get("C")
    return {"inflight_peak": peak, "inflight_le_c": peak <= C if _num(peak) and isinstance(C, int) else None}


def sentinel_summary(row, reqs, *, C: int, N: int) -> dict:
    if row is None:
        return {"valid": False, "reasons": ["sentinel run missing"], "output_throughput": None,
                "w_mean_ttft_ms": None, "steady_p95_ttft_ms": None, **inflight(None)}
    ok, reasons = run_valid(row, reqs, mix="uniform", C=C, N=N)
    if ok and not (_num(row.get("output_throughput")) and row["output_throughput"] > 0):
        ok, reasons = False, ["no output throughput"]
    if not ok:
        return {"valid": False, "reasons": reasons, "output_throughput": row.get("output_throughput"),
                "w_mean_ttft_ms": None, "steady_p95_ttft_ms": None, **inflight(row)}
    lo, hi = window_bounds("steady", C, N)
    w = [q["ttft_ms"] for q in reqs if lo <= q["idx"] < hi]
    return {"valid": True, "reasons": [], "output_throughput": row["output_throughput"],
            "w_mean_ttft_ms": statistics.fmean(w), "steady_p95_ttft_ms": _p95(w), **inflight(row)}


def sentinel_gate(start, end, *, tolerance: float) -> dict:
    """Output throughput and W-mean TTFT each within tolerance between start and end (Core A's rule; not the P95,
    which flips on the TTFT lattice)."""
    reasons = [f"{name} sentinel missing or invalid ({'; '.join((s or {}).get('reasons') or ['missing'])})"
               for name, s in (("start", start), ("end", end)) if not (s and s.get("valid"))]
    if reasons:
        return {"ok": False, "drift": None, "tolerance": tolerance, "reasons": reasons}
    drift = {}
    for key in ("output_throughput", "w_mean_ttft_ms"):
        d = abs(Fraction(end[key]) / Fraction(start[key]) - 1)
        drift[key] = float(d)
        if d > Fraction(str(tolerance)):
            reasons.append(f"sentinel {key} drifted {float(d):.4f} > {tolerance} between start and end")
    return {"ok": not reasons, "drift": drift, "tolerance": tolerance, "reasons": reasons}


def sentinel_references(core_rows: list, adv_runs: list, *, L: int = 2048, C: int = 25) -> dict:
    """The committed references of CROSS_SESSION_SHIFT: Core's mean output throughput at (L, C) uniform over its valid
    restarts, and the mean steady short-class P95 of the Advanced B0 runs (adv_runs in the 030 derived schema)."""
    thr = [r["output_throughput"] for r in core_rows
           if r.get("L") == L and r.get("mix") == "uniform" and r.get("C") == C and r.get("ok") is True]
    p95 = [_p95([q["ttft_ms"] for q in reqs if C <= q["idx"] < len(reqs) - C and q["cls"] == "short"])
           for reqs in adv_runs]
    return {"core_output_throughput": statistics.mean(thr), "advanced_steady_p95_ttft_ms": statistics.mean(p95)}


def cross_session_shift(sentinels: list, references, *, limit: float = CROSS_SESSION_LIMIT) -> dict:
    """Descriptive: a sentinel's throughput or steady P95 more than `limit` from the Core / Advanced references."""
    if references is None:
        return {"shift": None, "limit": limit, "values": []}
    values = [{"output_throughput": s["output_throughput"] / references["core_output_throughput"] - 1,
               "steady_p95_ttft_ms": s["steady_p95_ttft_ms"] / references["advanced_steady_p95_ttft_ms"] - 1}
              for s in sentinels if s and s.get("valid")]
    shift = any(abs(v) > limit for d in values for v in d.values()) if values else None
    return {"shift": shift, "limit": limit, "values": values, "references": references}


# ---------------------------------------------------------------- replay (030 8.6)

class _Stop(Exception):
    """A structural mismatch: what follows cannot be replayed."""


def evaluate(row: dict, reqs, plan: dict, slo: dict | None = None, window: str = cp.WINDOW) -> tuple:
    """(valid, reasons, eval or None) of one run {"workload", "C", "valid", "reasons"} and its derived requests, under
    slo (default: the plan's chosen SLO)."""
    L, mix = cp.workload_spec(row.get("workload"))
    C = row.get("C")
    N = bp.num_prompts(C)
    ok, reasons = run_valid(row, reqs, mix=mix, C=C, N=N)
    if not ok:
        return False, reasons, None
    return True, [], run_eval(reqs, L=L, mix=mix, C=C, N=N, slo=slo or plan["slo"], guard_band=plan["guard_band"],
                              window=window)


def search_bits(row: dict, reqs, plan: dict) -> dict:
    """A Phase S run's navigation bits (030 8.4): pass = valid, maxR_run <= 1 and (mixed) its own R_long <= 1 (an
    invalid run is a fail); within_band = valid with maxR_run <= 1 - m (C^_g, step 3: R_short and R_TPOT, not the long
    class). run_capacity records both; the replay recomputes them with this function."""
    ok, _, e = evaluate(row, reqs, plan)
    return {"pass": bool(ok and e["search_pass"]), "within_band": bool(ok and e["within_band"])}


def phase_c_outcome(row: dict, reqs, plan: dict) -> str:
    """A Phase C run for the skip rule (030 8.4): "invalid", "fail" (valid with maxR_run > 1) or "ok". The long class
    is not a run's outcome: it is pooled over the point's runs and judged once (8.1), so it never triggers a skip."""
    ok, _, e = evaluate(row, reqs, plan)
    return "invalid" if not ok else ("fail" if e["fail"] else "ok")


def _eval_row(row: dict, requests: dict, plan: dict, slo: dict, window: str = cp.WINDOW) -> tuple:
    return evaluate(row, requests.get(row.get("tag")), plan, slo, window)


def _row_plan_problems(plan: dict, row: dict) -> list:
    tag = row.get("tag")
    phase, w, C = row.get("phase"), row.get("workload"), row.get("C")
    if phase not in cp.PHASES or w not in plan["workloads"] or not isinstance(C, int) or C < 1:
        return [f"row {tag}: phase {phase!r} / workload {w!r} / C {C!r} outside the plan"]
    problems = []
    try:
        want_tag = cp.run_tag(phase, w, C, block=row.get("block"), position=row.get("position"))
    except cp.PlanError as e:
        want_tag = str(e)
    if tag != want_tag:
        problems.append(f"row {tag}: tag != {want_tag!r}")
    wl = plan["workloads"][w]
    if (row.get("L"), row.get("mix")) != (wl["L"], wl["mix"]):
        problems.append(f"row {tag}: L / mix {row.get('L')} / {row.get('mix')} != {w}'s")
    if row.get("N") != bp.num_prompts(C):
        problems.append(f"row {tag}: N {row.get('N')!r} != core_b_plan.num_prompts({C}) = {bp.num_prompts(C)}")
    want_t = plan["sentinel"]["timeout_s"] if phase == "sentinel" else wl["timeouts_s"].get(str(C))
    if row.get("timeout_s") != want_t:
        problems.append(f"row {tag}: timeout_s {row.get('timeout_s')!r} != planned {want_t!r}")
    return problems


def replay_problems(plan: dict, confirm_plan: dict, boots: list, rows: list, requests: dict, *,
                    confirm_sha256: str | None = None) -> list:
    """Every boot, search step, the confirm plan (K, n, orders), every skip and Phase C order recomputed from the
    rows; [] when the session ran exactly what 030 prescribes for its own data."""
    problems = []
    if [b.get("boot") for b in boots] != list(range(len(boots))):
        return ["boots are not numbered 0 .. n-1 in session order"]
    by_boot, tags = {}, set()
    for i, row in enumerate(rows):
        b = row.get("boot")
        if not (isinstance(b, int) and 0 <= b < len(boots)):
            problems.append(f"row {i} ({row.get('tag')}) names no recorded boot")
            continue
        if row.get("tag") in tags:
            problems.append(f"duplicate run tag {row.get('tag')}")
        tags.add(row.get("tag"))
        by_boot.setdefault(b, []).append(row)
        problems += _row_plan_problems(plan, row)
    if problems:
        return problems
    cursor = [0]

    def take(phase, w, *, block=None, attempt=None, position=None) -> dict:
        want = {"phase": phase, "workload": w, "block": block, "attempt": attempt, "position": position}
        if cursor[0] >= len(boots):
            problems.append(f"session ends before boot {want} (030 order)")
            raise _Stop
        b = boots[cursor[0]]
        if {k: b.get(k) for k in want} != want:
            problems.append(f"boot {cursor[0]} is {({k: b.get(k) for k in want})}; the 030 order needs {want}")
            raise _Stop
        cursor[0] += 1
        return b

    def sentinel(position: str) -> None:
        s = plan["sentinel"]
        b = take("sentinel", s["workload"], position=position)
        got = [(r.get("phase"), r.get("workload"), r.get("C"), r.get("position")) for r in by_boot.get(b["boot"], [])]
        if got != [("sentinel", s["workload"], s["C"], position)]:
            problems.append(f"{position} sentinel boot holds {got}, not one {s['workload']} C {s['C']} run")
            raise _Stop

    def search(w: str, b: dict):
        rows_b = by_boot.get(b["boot"], [])
        if b.get("reached_first_bench") is not True:
            if rows_b:
                problems.append(f"Phase S boot {b['boot']} ({w}) failed before its first bench but holds rows")
                raise _Stop
            return None
        c_cap, hist = plan["workloads"][w]["c_cap"], []
        for row in rows_b:
            want = cp.search_next(c_cap, [(h["C"], h["pass"]) for h in hist])
            if (row.get("phase"), row.get("workload"), row.get("C")) != ("S", w, want):
                problems.append(f"Phase S {w} step {len(hist) + 1} ran {row.get('phase')} {row.get('workload')} "
                                f"C {row.get('C')}; the 030 search runs C {want}")
                raise _Stop
            bits = search_bits(row, requests.get(row["tag"]), plan)
            if row.get("search_pass") is not bits["pass"]:
                problems.append(f"Phase S {row['tag']}: recorded search_pass {row.get('search_pass')!r}, "
                                f"recomputed {bits['pass']}")
            hist.append({"C": row["C"], **bits})
        nxt = cp.search_next(c_cap, [(h["C"], h["pass"]) for h in hist])
        if nxt is not None:
            problems.append(f"Phase S {w} stopped before C {nxt}: the search is incomplete")
            raise _Stop
        return hist

    try:
        sentinel("start")
        searches, retry = {}, []
        for w in plan["workloads"]:
            hist = search(w, take("S", w, attempt=1))
            if hist is None:
                retry.append(w)
            else:
                searches[w] = hist
        for w in cp.phase_s_retries(plan["workloads"], set(retry)):
            searches[w] = search(w, take("S", w, attempt=2))
        try:
            confirm = cp.build_confirm_plan(plan, searches)
        except cp.PlanError as e:
            problems.append(f"confirm plan cannot be recomputed: {e}")
            raise _Stop
        if confirm != confirm_plan:
            problems.append("the recorded confirm plan is not the one the Phase S rows determine")
        if confirm_sha256 is not None and confirm_sha256 != cp.confirm_plan_sha256(confirm_plan):
            problems.append("confirm_plan_sha256 recorded before Phase C != sha256 of the confirm plan")
        outcomes = {w: [] for w in plan["workloads"]}
        for r in range(1, confirm["blocks"] + 1):
            new = []
            for w, points in cp.phase_c_boots(confirm, r, outcomes):
                b = take("C", w, block=r)
                rows_b = by_boot.get(b["boot"], [])
                got = [(row.get("phase"), row.get("workload"), row.get("block"), row.get("C")) for row in rows_b]
                if got != [("C", w, r, C) for C in points]:
                    problems.append(f"Phase C block {r} {w} ran {[x[3] for x in got]}; the 030 order is {points}")
                    raise _Stop
                for row in rows_b:
                    outcome = phase_c_outcome(row, requests.get(row["tag"]), plan)
                    if row.get("outcome") != outcome:
                        problems.append(f"Phase C {row['tag']}: recorded outcome {row.get('outcome')!r}, "
                                        f"recomputed {outcome!r}")
                    new.append((w, (r, row["C"], outcome)))
            for w, o in new:
                outcomes[w].append(o)
        sentinel("end")
        if cursor[0] != len(boots):
            problems.append(f"{len(boots) - cursor[0]} boot(s) after the end sentinel")
    except _Stop:
        pass
    return problems


# ---------------------------------------------------------------- the analysis

def _run_summary(row: dict, ok: bool, reasons: list, e, full, slips=None) -> dict:
    m = row.get("metrics") or {}
    out = {"tag": row.get("tag"), "block": row.get("block"), "valid": ok, "reasons": reasons,
           "output_throughput": row.get("output_throughput"), "kv_cache_usage_peak": m.get("kv_cache_usage_peak"),
           "preemptions": m.get("preemptions"), "waiting_peak": m.get("waiting_peak"), **inflight(row)}
    if ok:
        out.update({"Y": e["Y"], "R": e["R"], "max_r": e["max_r"], "conclusion": _conclusion(e),
                    "attainment": e["attainment"], "burst_max_ttft_ms": e["burst_max_ttft_ms"], "n": e["n"],
                    "full": {"Y_short": full["Y"]["short"], "Y_tpot": full["Y"]["tpot"], "Y_long": full["Y"]["long"]}})
        if slips is not None:
            out["slip_episodes"] = slips
    return out


def workload_analysis(plan: dict, confirm_plan: dict, w: str, rows: list, requests: dict, *, slo: dict) -> dict:
    wl = plan["workloads"][w]
    L, mix, m = wl["L"], wl["mix"], plan["guard_band"]
    x = confirm_plan["workloads"].get(w) or {}
    points = {p["C"]: p for p in x.get("points") or []}
    mine = [r for r in rows if r.get("phase") == "C" and r.get("workload") == w]
    classes, point_out, steady_sum = {win: {} for win in WINDOWS}, {}, {}
    for C in sorted(points):
        evals, runs = {win: [] for win in WINDOWS}, []
        for row in (r for r in mine if r.get("C") == C):
            ok, reasons, e = _eval_row(row, requests, plan, slo)
            full = slips = None
            if ok:
                for win in WINDOWS:
                    evals[win].append(e if win == "steady" else _eval_row(row, requests, plan, slo, win)[2])
                full = evals["full"][-1]
                if wl.get("c0_covariate"):
                    slips = slip_episodes(requests[row["tag"]], C)
            runs.append(_run_summary(row, ok, reasons, e, full, slips))
        summary = {win: classify(evals[win], n=points[C]["n"], L=L, mix=mix, slo=slo, guard_band=m) for win in WINDOWS}
        for win in WINDOWS:
            classes[win][C] = summary[win]["class"]
        s = steady_sum[C] = summary["steady"]
        sampled = [r["inflight_le_c"] for r in runs if r["inflight_le_c"] is not None]
        point_out[str(C)] = {"role": points[C]["role"], "n": points[C]["n"], "n_valid": s["n_valid"],
                             "class": s["class"], "class_late": summary["late"]["class"],
                             "class_full": summary["full"]["class"], "met": s["met"], "worst_R": s["worst_R"],
                             "pooled": s["pooled"], "conclusions": s["conclusions"],
                             "inflight_le_c": all(sampled) if sampled else None, "runs": runs}
    est = {win: estimands(classes[win], c_cap=wl["c_cap"]) for win in WINDOWS}
    e = est["steady"]
    fails = [C for C in sorted(points) if classes["steady"][C] == "FAIL"]
    binding = None
    if fails:
        s = steady_sum[fails[0]]
        binding = sorted((k for k in TERMS if s["over"][k]), key=lambda k: -(s["worst_R"][k] or 0))
    margins = {k: ({t: 1 - v for t, v in steady_sum[e[k]["value"]]["worst_R"].items() if v is not None}
                   if e[k]["value"] in steady_sum else None) for k in ("c_slo", "c_prod")}
    search_rows = [r for r in rows if r.get("phase") == "S" and r.get("workload") == w]
    search_runs = []
    for row in search_rows:
        ok, reasons, ev = _eval_row(row, requests, plan, slo)
        search_runs.append({"C": row.get("C"), "valid": ok, "reasons": reasons, "Y": ev and ev["Y"],
                            "R": ev and ev["R"], "search_pass": bool(ok and ev["search_pass"]), **inflight(row)})
    knee = C_KNEE[str(L)]
    return {"L": L, "mix": mix, "c_kv": wl["c_kv"], "c_cap": wl["c_cap"], "n_near": wl["n_near"],
            "search": {"c_hat": x.get("c_hat"), "c_hat_g": x.get("c_hat_g"), "runs": search_runs},
            "K": sorted(points), "points": point_out, "c_slo": e["c_slo"], "c_prod": e["c_prod"],
            "binding": binding, "margins": margins,
            "flags": flags(classes["steady"], e, conclusions={C: steady_sum[C]["conclusions"] for C in points},
                           c_hat=x.get("c_hat"), c_kv=wl["c_kv"], late=est["late"]),
            "descriptive": {"c_full": est["full"], "c_late": est["late"],
                            "knee": {"c_knee": knee, "c_slo": e["c_slo"],
                                     "c_slo_minus_knee": None if e["c_slo"]["value"] is None
                                     else e["c_slo"]["value"] - knee}}}


def _sentinel(plan: dict, rows: list, requests: dict, position: str) -> dict:
    s = plan["sentinel"]
    row = next((r for r in rows if r.get("phase") == "sentinel" and r.get("position") == position), None)
    return sentinel_summary(row, requests.get(row.get("tag")) if row else None, C=s["C"], N=s["N"])


def analyze(cap: dict, requests: dict, *, references: dict | None = None) -> dict:
    """analysis.json body (schema_version 1) for one Stage C session; the report adds provenance and platform."""
    plan, confirm, session, rows = cap["plan"], cap["confirm_plan"], cap["session"], cap["rows"]
    replay = replay_problems(plan, confirm, session.get("boots") or [], rows, requests,
                             confirm_sha256=session.get("confirm_plan_sha256"))
    sent = {pos: _sentinel(plan, rows, requests, pos) for pos in ("start", "end")}
    workloads = {w: workload_analysis(plan, confirm, w, rows, requests, slo=plan["slo"]) for w in plan["workloads"]}
    c_kv = {w: x["c_kv"] for w, x in plan["workloads"].items()}
    secondary = {}
    for sid, slo in plan["secondary_slo"].items():
        res = {w: workload_analysis(plan, confirm, w, rows, requests, slo=slo) for w in plan["workloads"]}
        secondary[sid] = {"slo": slo, "descriptive": True,
                          "workloads": {w: {k: r[k] for k in ("c_slo", "c_prod", "binding", "flags")}
                                        for w, r in res.items()},
                          "hypotheses": hypotheses({w: r["c_slo"] for w, r in res.items()}, c_kv, predictions=None)}
    runs = []
    for row in rows:
        ok, reasons = False, ["not a sentinel / S / C row of this plan"]
        if row.get("workload") in plan["workloads"] and isinstance(row.get("C"), int):
            ok, reasons, _ = _eval_row(row, requests, plan, plan["slo"])
        runs.append({k: row.get(k) for k in ("phase", "workload", "C", "block", "position", "tag")}
                    | {"valid": ok, "reasons": reasons})
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "slo": plan["slo"],
            "protocol": {"seed": plan["seed"], "c0_branch": plan["c0_branch"], "window": plan["window"],
                         "guard_band": plan["guard_band"], "n_low": plan["n_low"],
                         "n_near": {w: x["n_near"] for w, x in plan["workloads"].items()}},
            "replay": {"ok": not replay, "problems": replay},
            "sentinel": {**sent, "gate": sentinel_gate(sent["start"], sent["end"],
                                                       tolerance=plan["sentinel"]["tolerance"]),
                         "cross_session_shift": cross_session_shift([sent["start"], sent["end"]], references)},
            "workloads": workloads, "secondary_slo": secondary,
            "hypotheses": hypotheses({w: r["c_slo"] for w, r in workloads.items()}, c_kv, predictions=PREDICTIONS),
            "not_supported": list(NOT_SUPPORTED), "runs": runs}
