#!/usr/bin/env python3
"""C0 descriptive analysis (decision 028; exploratory, formal = false): the 8K uniform two-mode probe. Not a hypothesis
test: the only thing it decides is which branch (A-C) PREREGISTRATION §8 takes. Written before any C0 data.

Per bench (valid benches only; L 8192 uniform, C 8, N 200); idx is the bench list index (= dispatch order, 028):
- Y = core_b_analysis.short_class_p95_ttft_ms; m_A = median TTFT over idx >= C;
- b_j = median TTFT over idx >= C with idx = j (mod C), each position's own baseline (request 8k+7 of a slipped group
  is only 11-16 ms above a global m_A + 170 ms threshold, 028);
- slipped: idx >= C and TTFT - b_(idx mod C) > 170 ms; an episode is a maximal run of consecutive slipped idx, recorded
  as its start group k = idx // C, position idx mod C and length;
- k_prom: the (0.3, 0.5] s bucket delta of vllm:request_queue_time_seconds between the full scrapes before and after
  the bench (run_adv.scrape_deltas), checked against the slipped count; None when the histogram is missing;
- mode, tested in this order: "other" if Y - m_A > 600 ms, "high" if Y - m_A > 170 ms, else "low";
- also: P95, P50 and median TPOT over idx >= C, the bench's output throughput, and the 1 Hz nvidia-smi samples inside
  the bench window (SM clock min / median, pstates, OR of clocks_event_reasons, OR of its throttle bits, max temp).
Per boot: n_b = episodes over its valid benches; the median of its valid benches' P50. A boot failed when its server
did not start or >= 2 of its benches are invalid (028 stop rule 5).
Session, first match: a stopped session is "cap" (the cap rule), "failed" (2 failed boots) or "indeterminate" (any
other stop, or fewer boots than planned); R0 if there is no episode; Rboot if the dispersion D = sum (n_b - n_bar)^2 /
n_bar over the boots with 4 valid benches exceeds chi^2_0.95(those boots - 1), or any boot with >= 1 valid bench (the
4-valid limit is D's alone, 028) has 0 episodes while the session has >= 1 episode per valid bench; "indeterminate"
if D cannot be computed (< 2 boots with 4 valid benches, or n_bar = 0); else Rrun. §8 branch: R0 / Rrun -> A,
Rboot -> B, cap / failed / indeterminate -> C. Branch D (a harness or client defect) is set only by a human decision
log, never here.
Reported only: boots showing both low and high, start k of bench 1 vs benches 2-4, the k in {8, 16, 24} concentration,
the high share against Phase 1's 8 / 12. Flags: Fhw (throttle bits, or a median SM clock < 0.95 x the boot's median)
cross-tabulated with episodes; Foff (boot-median P50 spread > 1 %).

Refuses any session whose session.json stage is not "C0".

Usage (Mac): .venv/bin/python benchmarks/c0_analysis.py --session results/capacity/explore-c0-<YYYYMMDD>

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import core_b_analysis as ba

SCHEMA_VERSION = 1
SLIP_MS = 170.0
HIGH_MS = 170.0
OTHER_MS = 600.0
QUEUE_HIST = "vllm:request_queue_time_seconds"
PROM_BUCKET = ("0.3", "0.5")  # le bounds of the (0.3, 0.5] s bucket
# Upper 5 % critical values of the chi-square distribution, df 1-5: NIST/SEMATECH e-Handbook of Statistical Methods,
# table 1.3.6.7.4 "Critical Values of the Chi-Square Distribution" (6 boots -> df 5 -> 11.070, 028)
CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070}
# nvml.h clocks event reasons that slow the clocks: SwPowerCap 0x4, HwSlowdown 0x8, SwThermalSlowdown 0x20,
# HwThermalSlowdown 0x40, HwPowerBrakeSlowdown 0x80. GpuIdle 0x1, ApplicationsClocksSetting 0x2, SyncBoost 0x10 and
# DisplayClockSetting 0x100 do not; 0x400 (every Phase 1 idle capture) is undocumented (028 線索 10): OR only.
THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
FHW_CLOCK_FRACTION = 0.95
FOFF_PCT = 1.0
K_REPLICATION = (8, 16, 24)
PHASE1_HIGH = (8, 12)  # 028: high in 8 of the 12 Phase 1 runs at 8K C = 8 (Core 3 / 3, Advanced B0 / G / K3 5 / 9)
BRANCH = {"R0": "A", "Rrun": "A", "Rboot": "B", "cap": "C", "failed": "C", "indeterminate": "C"}
BRANCH_D = "branch D (a harness or client defect) is set only by a human decision log; it is never computed here"


class AnalysisError(ValueError):
    pass


def _median(values) -> float:
    return ba.percentile(list(values), 50)


def _read_jsonl(path: Path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------- per bench

def baselines(ttfts_ms: list, C: int) -> list:
    """b_j, j = 0..C-1: the median TTFT over idx >= C with idx = j (mod C)."""
    return [_median(ttfts_ms[i] for i in range(C + j, len(ttfts_ms), C)) for j in range(C)]


def slipped(ttfts_ms: list, C: int, base: list) -> list:
    return [i for i in range(C, len(ttfts_ms)) if ttfts_ms[i] - base[i % C] > SLIP_MS]


def episodes(slipped_idx: list, C: int) -> list:
    """Maximal runs of consecutive slipped idx: start idx, start group k, position, length."""
    out = []
    for i in slipped_idx:
        if out and out[-1]["start_idx"] + out[-1]["length"] == i:
            out[-1]["length"] += 1
        else:
            out.append({"start_idx": i, "k": i // C, "position": i % C, "length": 1})
    return out


def mode(y_ms: float, m_a_ms: float) -> str:
    """028, in this order: other (> 600 ms), high (> 170 ms), low."""
    d = y_ms - m_a_ms
    if d > OTHER_MS:
        return "other"
    if d > HIGH_MS:
        return "high"
    return "low"


def classify_bench(ttfts_ms: list, *, C: int, L: int, input_lens: list | None = None) -> dict:
    """028 per-bench rules over the TTFTs (ms) in bench list order; input_lens default to the uniform short class."""
    lens = list(input_lens) if input_lens is not None else [L - ba.N_GEN] * len(ttfts_ms)
    if len(lens) != len(ttfts_ms):
        raise AnalysisError(f"{len(ttfts_ms)} TTFTs but {len(lens)} input lengths")
    if len(ttfts_ms) < 2 * C:
        raise AnalysisError(f"{len(ttfts_ms)} TTFTs: the per-position baselines need at least 2 C = {2 * C}")
    base = baselines(ttfts_ms, C)
    m_a = _median(ttfts_ms[C:])
    y = ba.short_class_p95_ttft_ms({"ttfts": [t / 1000 for t in ttfts_ms], "input_lens": lens}, L=L)
    s = slipped(ttfts_ms, C, base)
    eps = episodes(s, C)
    return {"y_ms": y, "m_a_ms": m_a, "mode": mode(y, m_a), "baselines_ms": base, "slipped": s, "n_slipped": len(s),
            "episodes": eps, "n_episodes": len(eps), "p95_excl_first_c_ms": ba.percentile(ttfts_ms[C:], 95),
            "p50_excl_first_c_ms": ba.percentile(ttfts_ms[C:], 50)}


def tpot_excl_first_c_ms(requests: list, C: int) -> float | None:
    """Median per-request TPOT ((latency - ttft) / (output - 1), as `vllm bench serve`) over idx >= C."""
    v = [(r["latency_ms"] - r["ttft_ms"]) / (r["output_len"] - 1) for r in requests
         if r["idx"] >= C and r.get("output_len", 0) > 1]
    return _median(v) if v else None


def k_prom(scrape_delta: dict | None) -> int | None:
    """The (0.3, 0.5] s bucket count of the queue-time histogram's bench-window delta; None without it."""
    buckets = (((scrape_delta or {}).get(QUEUE_HIST)) or {}).get("buckets") or {}
    lo, hi = PROM_BUCKET
    if lo not in buckets or hi not in buckets:
        return None
    return int(round(buckets[hi] - buckets[lo]))


def bench_telemetry(samples: list, t0: float, t1: float) -> dict:
    """The 1 Hz nvidia-smi samples with t0 <= t <= t1 (wall seconds); missing values ([N/A]) are None and skipped."""
    win = [s for s in samples if t0 <= s["t"] <= t1]
    sm = [s["sm_mhz"] for s in win if s.get("sm_mhz") is not None]
    temps = [s["temp_c"] for s in win if s.get("temp_c") is not None]
    reasons = [int(s["event_reasons"]) for s in win if s.get("event_reasons") is not None]
    ored = 0
    for r in reasons:
        ored |= r
    return {"n_samples": len(win), "sm_min_mhz": min(sm, default=None), "sm_median_mhz": _median(sm) if sm else None,
            "pstates": sorted({s["pstate"] for s in win if s.get("pstate")}),
            "event_reasons_or": ored if reasons else None, "throttle_or": ored & THROTTLE_MASK if reasons else None,
            "temp_max_c": max(temps, default=None)}


# ---------------------------------------------------------------- session

def binomial_tail(n: int, m: int, p: float) -> float:
    """P(X >= m) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(m, n + 1))


def session_outcome(boots: list, *, stop_rule: str | None, complete: bool, benches_per_boot: int = 4) -> dict:
    """boots: {"boot", "n_valid", "n_b"} each. The 028 session outcome, first match (module docstring)."""
    total = sum(b["n_b"] for b in boots)
    n_valid = sum(b["n_valid"] for b in boots)
    q = [b for b in boots if b["n_valid"] == benches_per_boot]
    n_bar = d = df = crit = None
    if len(q) >= 2:
        df = len(q) - 1
        if df not in CHI2_95:
            raise AnalysisError(f"{len(q)} boots: no hard-coded chi-square critical value for df {df}")
        crit = CHI2_95[df]
        n_bar = sum(b["n_b"] for b in q) / len(q)
        if n_bar > 0:
            d = sum((b["n_b"] - n_bar) ** 2 for b in q) / n_bar
    per_bench = total / n_valid if n_valid else None
    # the 4-valid limit is D's alone (028); a boot without a valid bench observed nothing, so it is never empty
    zero = per_bench is not None and per_bench >= 1 and any(b["n_b"] == 0 and b["n_valid"] >= 1 for b in boots)
    if stop_rule == "cap":
        outcome = "cap"
    elif stop_rule == "failed_boots":
        outcome = "failed"
    elif stop_rule is not None or not complete or not n_valid:
        outcome = "indeterminate"
    elif total == 0:
        outcome = "R0"
    elif (d is not None and d > crit) or zero:
        outcome = "Rboot"
    elif d is None:
        outcome = "indeterminate"
    else:
        outcome = "Rrun"
    return {"outcome": outcome, "branch": BRANCH[outcome], "total_episodes": total, "valid_benches": n_valid,
            "episodes_per_valid_bench": per_bench, "qualifying_boots": [b["boot"] for b in q], "n_bar": n_bar,
            "D": d, "df": df, "critical": crit, "zero_rule": zero, "stop_rule": stop_rule}


def _bench(d: Path, row: dict, *, C: int, L: int, samples: list) -> dict:
    rec = {"tag": row["tag"], "boot": row["boot"], "bench": row["bench"], "valid": row.get("ok") is True,
           "reasons": row.get("reasons") or []}
    if not rec["valid"]:
        return rec
    if not row.get("derived") or not (d / row["derived"]).is_file():
        raise AnalysisError(f"valid bench {row['tag']} has no derived requests file")
    reqs = sorted(_read_jsonl(d / row["derived"]), key=lambda r: r["idx"])
    if [r["idx"] for r in reqs] != list(range(len(reqs))):
        raise AnalysisError(f"{row['derived']}: idx is not 0..{len(reqs) - 1}")
    cls = classify_bench([r["ttft_ms"] for r in reqs], C=C, L=L, input_lens=[r["input_len"] for r in reqs])
    kp = k_prom(row.get("scrape_delta"))
    rec.update(cls, k_prom=kp, k_prom_matches=None if kp is None else kp == cls["n_slipped"],
               tpot_median_excl_first_c_ms=tpot_excl_first_c_ms(reqs, C),
               output_throughput=row.get("output_throughput"),
               telemetry=bench_telemetry(samples, row["t_start"], row["t_end"]))
    return rec


def _boot(b: int, record: dict, benches: list, per_boot: int) -> dict:
    valid = [x for x in benches if x["boot"] == b and x["valid"]]
    startup_ok = record.get("startup_ok") is True
    p50 = [x["p50_excl_first_c_ms"] for x in valid]
    sm = [x["telemetry"]["sm_median_mhz"] for x in valid if x["telemetry"]["sm_median_mhz"] is not None]
    return {"boot": b, "startup_ok": startup_ok, "n_valid": len(valid), "n_invalid": per_boot - len(valid),
            "failed": not startup_ok or per_boot - len(valid) >= 2, "n_b": sum(x["n_episodes"] for x in valid),
            "p50_median_ms": _median(p50) if p50 else None, "sm_median_mhz": _median(sm) if sm else None,
            "modes": sorted({x["mode"] for x in valid})}


def _fhw(valid: list, boots: list) -> dict:
    ref = {b["boot"]: b["sm_median_mhz"] for b in boots}
    table = dict.fromkeys(("flag_with_episodes", "flag_without_episodes", "clean_with_episodes",
                           "clean_without_episodes"), 0)
    flagged, no_tel = [], 0
    for x in valid:
        tel = x["telemetry"]
        if not tel["n_samples"]:
            no_tel += 1
            continue
        sm, boot_sm = tel["sm_median_mhz"], ref.get(x["boot"])
        hw = bool(tel["throttle_or"]) or (sm is not None and boot_sm is not None and sm < FHW_CLOCK_FRACTION * boot_sm)
        table[("flag" if hw else "clean") + ("_with_episodes" if x["n_episodes"] else "_without_episodes")] += 1
        if hw:
            flagged.append(x["tag"])
    return {"flagged": flagged, "table": table, "no_telemetry": no_tel}


def _foff(boots: list) -> dict:
    p50 = {f"b{b['boot']}": b["p50_median_ms"] for b in boots if b["p50_median_ms"] is not None}
    spread = (max(p50.values()) / min(p50.values()) - 1) * 100 if len(p50) >= 2 else None
    return {"boot_p50_median_ms": p50, "spread_pct": spread, "flag": spread is not None and spread > FOFF_PCT}


def _reported(valid: list, boots: list, *, C: int, N: int) -> dict:
    eps = [e for x in valid for e in x["episodes"]]
    at = sum(e["k"] in K_REPLICATION for e in eps)
    chance = len(K_REPLICATION) / ((N - 1) // C)  # start groups k = 1 .. (N - 1) // C can hold an episode
    high = sum(x["mode"] == "high" for x in valid)
    return {
        "both_modes_boots": sum({"low", "high"} <= set(b["modes"]) for b in boots),
        "start_k": {"bench1": sorted(e["k"] for x in valid if x["bench"] == 1 for e in x["episodes"]),
                    "bench2_4": sorted(e["k"] for x in valid if x["bench"] > 1 for e in x["episodes"])},
        "k_8_16_24": {"k": list(K_REPLICATION), "episodes": len(eps), "at": at,
                      "share": at / len(eps) if eps else None, "chance": chance,
                      "binomial_tail_p": binomial_tail(len(eps), at, chance)},
        "high_share": {"high": high, "other": sum(x["mode"] == "other" for x in valid), "valid": len(valid),
                       "share": high / len(valid) if valid else None, "phase1": list(PHASE1_HIGH),
                       "phase1_share": PHASE1_HIGH[0] / PHASE1_HIGH[1]}}


def analyse_session(session_dir) -> dict:
    d = Path(session_dir)
    session = json.loads((d / "session.json").read_text())
    if session.get("stage") != "C0":
        raise AnalysisError(f"session {d.name}: stage {session.get('stage')!r} != 'C0' (028: C0 sessions only)")
    plan = json.loads((d / "plan.json").read_text())
    C, L, per_boot = plan["C"], plan["L"], plan["benches_per_boot"]
    records = session.get("boots") or {}
    samples = {}
    for key, rec in records.items():
        samples[int(key[1:])] = _read_jsonl(d / rec["smi"]) if rec.get("smi") and (d / rec["smi"]).is_file() else []
    benches = [_bench(d, r, C=C, L=L, samples=samples.get(r["boot"], [])) for r in _read_jsonl(d / "rows.jsonl")]
    boots = [_boot(int(k[1:]), records[k], benches, per_boot) for k in sorted(records, key=lambda k: int(k[1:]))]
    stopped = session.get("stopped")
    outcome = session_outcome(boots, stop_rule=(stopped or {}).get("rule"),
                              complete=stopped is None and len(boots) == plan["boots"], benches_per_boot=per_boot)
    valid = [x for x in benches if x["valid"]]
    return {"schema_version": SCHEMA_VERSION, "session_id": session.get("session_id"), "stage": "C0",
            "decision": session.get("decision"), "formal": session.get("formal"),
            "rules": {"slip_ms": SLIP_MS, "high_ms": HIGH_MS, "other_ms": OTHER_MS, "prom_bucket_s": list(PROM_BUCKET),
                      "chi2_95": CHI2_95, "throttle_mask": THROTTLE_MASK, "fhw_clock_fraction": FHW_CLOCK_FRACTION,
                      "foff_pct": FOFF_PCT},
            "benches": benches, "boots": boots, "session": outcome,
            "reported": _reported(valid, boots, C=C, N=plan["N"]),
            "flags": {"Fhw": _fhw(valid, boots), "Foff": _foff(boots)}, "branch_d": BRANCH_D}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    try:
        a = analyse_session(args.session)
    except AnalysisError as exc:
        print(f"c0_analysis: {exc}", file=sys.stderr)
        return 2
    (Path(args.session) / "analysis.json").write_text(json.dumps(a, indent=1) + "\n")
    print(json.dumps({k: a["session"][k] for k in ("outcome", "branch", "total_episodes", "D", "df", "critical")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
