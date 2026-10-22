#!/usr/bin/env python3
"""Stage D analysis (decision 039; PREREGISTRATION §9), fixed before any Stage D data. The statistics are Stage C's
own frozen functions (capacity_analysis.run_eval, classify, estimands, flags, h_s2, the sentinel gate and the
cross-session shift); this module only supplies what 030 did not have:

- a run's spec from the Stage D plan: the S2 tier of its prompt class (tier), its mix ("var" judged as uniform: every
  request is short) and its output length (a 512-token workload's TPOT is (latency - TTFT) / 511, run_eval's own
  formula on each request's out);
- N: core_b_plan.num_prompts(C) closed loop (and D-C, and the sentinel), the grid point's N open loop;
- the window: closed loop W = [C, N - C) as 030; open loop W = [ceil(0.1 N), N - ceil(0.1 N)), which is run_eval's
  steady window with C := ceil(0.1 N) (so its "burst" is the first 10 %, descriptive);
- derived_problems with the workload's output length and its mix (mixed: 007's long count and a long request in W;
  uniform / var: every request short);
- the replay of D-A (capacity_analysis.replay_problems' rules with Stage D's tags and N), of D-B (the same with
  039's scan and confirm set: stage_d_plan.open_search_next / build_open_confirm_plan) and of D-C (blocks r = 1..n in
  capacity_plan.block_order, one boot and one run per active point; 030's skip rule: a point leaves later blocks
  after a valid FAIL or an invalid run);
- open loop (039 FS-order, per run): the derived start_s must follow idx (no request sent more than ORDER_TOL_S before
  a lower idx), so the trimmed window by idx is the window by send time;
- D-B values are grid indices in the estimands; the report adds their rates;
- the 039 hypotheses: H-D1 / H-D3 = 030's H-S2 (C_SLO <= C_KV; "refuted" written "not supported"), H-D2a C_PROD(M8K-T)
  > 2 and H-D2b C_PROD(U8K-T) < 8 (c1b's values, 035), H-D4 lambda_PROD <= 0.9 lambda_ref, H-D6 / H-D7 PASS;
- D5 (descriptive, c1b only): the start-up wave's largest first-wave TTFT by C.
"""

from __future__ import annotations

import json
import statistics
from fractions import Fraction
from pathlib import Path

import capacity_analysis as ca
import capacity_plan as cp
import core_b_plan as bp
import stage_d_plan as sd

SCHEMA_VERSION = 1
STAGE = sd.STAGE
WINDOWS = ca.WINDOWS
C1B_C_PROD = {"M8K": 2, "U8K": 8}          # 035, the c1b values H-D2a / H-D2b compare with (never re-estimated)
NO_CKV = 10 ** 9                           # D-B: a grid index has no C_KV, so ABOVE_CKV never applies
ORDER_TOL_S = 0.05                         # D-B: how much earlier than any lower idx a request may be sent
NOT_SUPPORTED = (
    "Stage D measures one machine, one model and Core B's server (or T); open loop is vLLM's own capacity without the "
    "gateway's admission; D-C checks only C_PROD itself; mixed open loop, other output lengths and T on other "
    "workloads are not measured; the start-up wave (D5) is descriptive.",
)


class _Stop(Exception):
    pass


# ---------------------------------------------------------------- per run

def spec(plan: dict, w: str) -> dict:
    """The workload's plan entry; the sentinel's (U2K at C 25 on the Core B server) when w is only the sentinel."""
    if w in plan["workloads"]:
        return plan["workloads"][w]
    if w == plan["sentinel"]["workload"]:
        return {**sd.WORKLOADS[w], "kv_len": sd.kv_len(w)}
    raise ca.AnalysisError(f"workload {w!r} is not in this plan")


def eval_mix(x: dict) -> str:
    return "mixed" if x["mix"] == "mixed" else "uniform"


def run_n(plan: dict, row: dict) -> int:
    """N of a run: the grid point's N for an open-loop S / C run, else core_b_plan.num_prompts(C)."""
    if plan["kind"] == "open" and row.get("phase") in ("S", "C"):
        return plan["workloads"][row["workload"]]["points"][str(row["C"])]["N"]
    return bp.num_prompts(row["C"])


def window_c(plan: dict, row: dict, n: int) -> int:
    """run_eval's C: W = [c, n - c) is 030's steady window closed loop and D-B's trimmed window open loop."""
    if plan["kind"] == "open" and row.get("phase") in ("S", "C"):
        return sd.open_window(n, plan["open_loop"]["trim"])[0]
    return row["C"]


def derived_problems(reqs, *, x: dict, c: int, n: int, ordered: bool = False) -> list:
    """030 8.6 (e) for a Stage D run: complete idx, numbers, every out = the workload's output length, the mix's
    classes and a short (mixed: and a long) request in W = [c, n - c); ordered (open loop): start_s follows idx."""
    if reqs is None:
        return ["no derived requests file"]
    if not all(isinstance(q, dict) and isinstance(q.get("idx"), int) for q in reqs):
        return ["derived requests without an integer idx"]
    problems = []
    if sorted(q["idx"] for q in reqs) != list(range(n)):
        problems.append(f"idx is not 0..{n - 1} exactly once ({len(reqs)} requests)")
    if not all(ca._num(q.get(k)) for q in reqs for k in ("ttft_ms", "latency_ms", "out")):
        problems.append("ttft_ms / latency_ms / out missing or not numbers")
    if any(q.get("out") != x["out"] for q in reqs):
        problems.append(f"out not all {x['out']}")
    classes = [q.get("cls") for q in reqs]
    want_long = bp.mixed_counts(n)[1] if x["mix"] == "mixed" else 0
    if any(cl not in ("short", "long") for cl in classes) or classes.count("long") != want_long:
        problems.append(f"{classes.count('long')} long requests of {len(classes)}; the {x['mix']} mix has {want_long}")
    in_w = [cl for q, cl in zip(reqs, classes) if c <= q["idx"] < n - c]
    if "short" not in in_w:
        problems.append("no short request in W")
    if x["mix"] == "mixed" and "long" not in in_w:
        problems.append("no long request in W (030 8.6 (e))")
    if ordered and not problems:
        starts = [q.get("start_s") for q in sorted(reqs, key=lambda q: q["idx"])]
        if not all(ca._num(t) for t in starts):
            problems.append("start_s missing: the send order cannot be checked (039 D-B)")
        else:
            late, peak = 0.0, starts[0]                  # against every lower idx: the running maximum
            for t in starts[1:]:
                late, peak = max(late, peak - t), max(peak, t)
            if late > ORDER_TOL_S:
                problems.append(f"a request was sent {late:.3f} s before a lower idx: idx is not the send order "
                                "(039 D-B window)")
    return problems


def run_valid(row: dict, reqs, plan: dict) -> tuple:
    if row.get("valid") is not True:
        return False, ["runner: " + "; ".join(row.get("reasons") or ["not valid"])]
    n = run_n(plan, row)
    ordered = plan["kind"] == "open" and row.get("phase") in ("S", "C")
    problems = derived_problems(reqs, x=spec(plan, row["workload"]), c=window_c(plan, row, n), n=n, ordered=ordered)
    return not problems, problems


def evaluate(row: dict, reqs, plan: dict, slo: dict | None = None, window: str = cp.WINDOW) -> tuple:
    """(valid, reasons, eval or None) of one run under slo (default: the plan's S2), with run_eval itself."""
    ok, reasons = run_valid(row, reqs, plan)
    if not ok:
        return False, reasons, None
    x, n = spec(plan, row["workload"]), run_n(plan, row)
    return True, [], ca.run_eval(reqs, L=x["tier"], mix=eval_mix(x), C=window_c(plan, row, n), N=n,
                                 slo=slo or plan["slo"], guard_band=plan["guard_band"], window=window)


def search_bits(row: dict, reqs, plan: dict) -> dict:
    ok, _, e = evaluate(row, reqs, plan)
    return {"pass": bool(ok and e["search_pass"]), "within_band": bool(ok and e["within_band"])}


def phase_c_outcome(row: dict, reqs, plan: dict) -> str:
    ok, _, e = evaluate(row, reqs, plan)
    return "invalid" if not ok else ("fail" if e["fail"] else "ok")


# ---------------------------------------------------------------- replay

def verify_order(plan: dict) -> list:
    """D-C's boots before skips: [(block, workload), ...], blocks r = 1..n in capacity_plan.block_order."""
    n = max(x["n"] for x in plan["workloads"].values())
    return [(r, w) for r in range(1, n + 1) for w in cp.block_order(plan["seed"], r, list(plan["workloads"]))
            if r <= plan["workloads"][w]["n"]]


def _row_plan_problems(plan: dict, row: dict) -> list:
    tag, phase, w, x = row.get("tag"), row.get("phase"), row.get("workload"), row.get("C")
    sentinel = phase == "sentinel" and w == plan["sentinel"]["workload"]
    if phase not in sd.PHASES or not (sentinel or w in plan["workloads"]) or not isinstance(x, int) or x < 1:
        return [f"row {tag}: phase {phase!r} / workload {w!r} / C {x!r} outside the plan"]
    try:
        want_tag = sd.run_tag(phase, w, x, block=row.get("block"), position=row.get("position"), kind=plan["kind"])
    except sd.PlanError as e:
        want_tag = str(e)
    problems = [] if tag == want_tag else [f"row {tag}: tag != {want_tag!r}"]
    if row.get("N") != run_n(plan, row):
        problems.append(f"row {tag}: N {row.get('N')!r} != {run_n(plan, row)}")
    if phase == "sentinel":
        want_t = plan["sentinel"]["timeout_s"]
    elif plan["kind"] == "closed":
        want_t = plan["workloads"][w]["timeouts_s"].get(str(x))
    elif plan["kind"] == "open":
        want_t = (plan["workloads"][w]["points"].get(str(x)) or {}).get("timeout_s")
    else:
        want_t = plan["workloads"][w]["timeout_s"]
    if row.get("timeout_s") != want_t:
        problems.append(f"row {tag}: timeout_s {row.get('timeout_s')!r} != planned {want_t!r}")
    return problems


def replay_problems(plan: dict, confirm_plan, boots: list, rows: list, requests: dict, *,
                    confirm_sha256: str | None = None) -> list:
    """Every boot and step recomputed from the rows: D-A / D-B as capacity_analysis.replay_problems (Phase S with
    capacity_plan.search_next, the confirm plan, the Phase C skips and orders), D-C as one boot per point and block
    in capacity_plan.block_order; [] when the session ran exactly what 039 prescribes for its own data."""
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
            problems.append(f"session ends before boot {want} (039 order)")
            raise _Stop
        b = boots[cursor[0]]
        if {k: b.get(k) for k in want} != want:
            problems.append(f"boot {cursor[0]} is {({k: b.get(k) for k in want})}; the 039 order needs {want}")
            raise _Stop
        cursor[0] += 1
        return b

    def no_valid_on_failed_boot(b: dict, rows_b: list) -> None:
        if b.get("reached_first_bench") is not True and any(r.get("valid") is True for r in rows_b):
            problems.append(f"boot {b['boot']} failed before its first bench but holds a valid run")
            raise _Stop

    def sentinel(position: str) -> None:
        s = plan["sentinel"]
        b = take("sentinel", s["workload"], position=position)
        no_valid_on_failed_boot(b, by_boot.get(b["boot"], []))
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
            want = search_next(c_cap, [(h["C"], h["pass"]) for h in hist])
            if (row.get("phase"), row.get("workload"), row.get("C")) != ("S", w, want):
                problems.append(f"Phase S {w} step {len(hist) + 1} ran {row.get('phase')} {row.get('workload')} "
                                f"{row.get('C')}; the 030 search runs {want}")
                raise _Stop
            bits = search_bits(row, requests.get(row["tag"]), plan)
            if row.get("search_pass") is not bits["pass"]:
                problems.append(f"Phase S {row['tag']}: recorded search_pass {row.get('search_pass')!r}, "
                                f"recomputed {bits['pass']}")
            hist.append({"C": row["C"], **bits})
        nxt = search_next(c_cap, [(h["C"], h["pass"]) for h in hist])
        if nxt is not None:
            problems.append(f"Phase S {w} stopped before {nxt}: the search is incomplete")
            raise _Stop
        return hist

    search_next = sd.open_search_next if plan["kind"] == "open" else cp.search_next
    build_confirm = sd.build_open_confirm_plan if plan["kind"] == "open" else cp.build_confirm_plan
    try:
        sentinel("start")
        if plan["kind"] == "verify":
            dropped = set()
            for r, w in verify_order(plan):
                if w in dropped:
                    continue
                x = plan["workloads"][w]
                b = take("V", w, block=r)
                rows_b = by_boot.get(b["boot"], [])
                no_valid_on_failed_boot(b, rows_b)
                got = [(row.get("phase"), row.get("workload"), row.get("block"), row.get("C")) for row in rows_b]
                if got != [("V", w, r, x["C"])]:
                    problems.append(f"D-C block {r} {w} ran {got}; 039 runs one V run at C {x['C']}")
                    raise _Stop
                outcome = phase_c_outcome(rows_b[0], requests.get(rows_b[0]["tag"]), plan)
                if rows_b[0].get("outcome") != outcome:
                    problems.append(f"D-C {rows_b[0]['tag']}: recorded outcome {rows_b[0].get('outcome')!r}, "
                                    f"recomputed {outcome!r}")
                if outcome != "ok":
                    dropped.add(w)
        else:
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
                confirm = build_confirm(plan, searches)
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
                    no_valid_on_failed_boot(b, rows_b)
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


# ---------------------------------------------------------------- per workload

def _eval_row(row: dict, requests: dict, plan: dict, slo: dict, window: str = cp.WINDOW) -> tuple:
    return evaluate(row, requests.get(row.get("tag")), plan, slo, window)


def _point(plan: dict, rows: list, requests: dict, *, n: int, x: dict, slo: dict) -> tuple:
    """(steady summary, {window: class}, runs) of one point's rows."""
    evals, runs = {win: [] for win in WINDOWS}, []
    for row in rows:
        ok, reasons, e = _eval_row(row, requests, plan, slo)
        full = None
        if ok:
            for win in WINDOWS:
                evals[win].append(e if win == "steady" else _eval_row(row, requests, plan, slo, win)[2])
            full = evals["full"][-1]
        # open loop: the grid index is not a concurrency, so inflight <= C does not apply
        runs.append(ca._run_summary({**row, "C": None} if plan["kind"] == "open" else row, ok, reasons, e, full))
    summary = {win: ca.classify(evals[win], n=n, L=x["tier"], mix=eval_mix(x), slo=slo,
                                guard_band=plan["guard_band"]) for win in WINDOWS}
    return summary["steady"], {win: summary[win]["class"] for win in WINDOWS}, runs


def search_analysis(plan: dict, confirm_plan: dict, w: str, rows: list, requests: dict, *, slo: dict) -> dict:
    """A D-A / D-B workload: classes on K, C_SLO / C_PROD (D-B: grid indices) with 030's statuses and flags."""
    x = plan["workloads"][w]
    cw = confirm_plan["workloads"].get(w) or {}
    points = {p["C"]: p for p in cw.get("points") or []}
    mine = [r for r in rows if r.get("phase") == "C" and r.get("workload") == w]
    classes, point_out, steady = {win: {} for win in WINDOWS}, {}, {}
    for C in sorted(points):
        s, cls, runs = _point(plan, [r for r in mine if r.get("C") == C], requests, n=points[C]["n"], x=x, slo=slo)
        steady[C] = s
        for win in WINDOWS:
            classes[win][C] = cls[win]
        point_out[str(C)] = {"role": points[C]["role"], "n": points[C]["n"], "n_valid": s["n_valid"],
                             "class": s["class"], "class_late": cls["late"], "class_full": cls["full"],
                             "met": s["met"], "worst_R": s["worst_R"], "pooled": s["pooled"],
                             "conclusions": s["conclusions"], "runs": runs}
        if plan["kind"] == "open":
            point_out[str(C)]["rate"] = x["points"][str(C)]["rate"]
    est = {win: ca.estimands(classes[win], c_cap=x["c_cap"]) for win in WINDOWS}
    e = est["steady"]
    fails = [C for C in sorted(points) if classes["steady"][C] == "FAIL"]
    binding = None
    if fails:
        s = steady[fails[0]]
        binding = sorted((k for k in ca.TERMS if s["over"][k]), key=lambda k: -(s["worst_R"][k] or 0))
    search_runs = []
    for row in (r for r in rows if r.get("phase") == "S" and r.get("workload") == w):
        ok, reasons, ev = _eval_row(row, requests, plan, slo)
        search_runs.append({"C": row.get("C"), "valid": ok, "reasons": reasons, "Y": ev and ev["Y"],
                            "R": ev and ev["R"], "search_pass": bool(ok and ev["search_pass"])})
    out = {"tier": x["tier"], "mix": x["mix"], "out": x["out"], "server": x["server"], "c_kv": x["c_kv"],
           "c_cap": x["c_cap"], "n_near": x["n_near"],
           "search": {"c_hat": cw.get("c_hat"), "c_hat_g": cw.get("c_hat_g"), "runs": search_runs},
           "K": sorted(points), "points": point_out, "c_slo": e["c_slo"], "c_prod": e["c_prod"], "binding": binding,
           "flags": ca.flags(classes["steady"], e, conclusions={C: steady[C]["conclusions"] for C in points},
                             c_hat=cw.get("c_hat"), c_kv=NO_CKV if plan["kind"] == "open" else x["c_kv"],
                             late=est["late"]),
           "descriptive": {"c_full": est["full"], "c_late": est["late"]}}
    if plan["kind"] == "open":   # a grid index's rate and factor; INFEASIBLE (0) is below the lowest grid point
        grid = plan["open_loop"]["grid"]

        def at(k, field):
            v = e[k]["value"]
            return None if v is None else 0 if v == 0 else (grid[v - 1] if field == "factor" else
                                                              x["points"][str(v)]["rate"])

        out.update(lambda_ref=x["lambda_ref"], rate_slo=at("c_slo", "rate"), rate_prod=at("c_prod", "rate"),
                   factor_slo=at("c_slo", "factor"), factor_prod=at("c_prod", "factor"))
    return out


def verify_analysis(plan: dict, w: str, rows: list, requests: dict, *, slo: dict) -> dict:
    x = plan["workloads"][w]
    s, cls, runs = _point(plan, [r for r in rows if r.get("phase") == "V" and r.get("workload") == w], requests,
                          n=x["n"], x=x, slo=slo)
    return {"tier": x["tier"], "mix": x["mix"], "via": x["via"], "C": x["C"], "n": x["n"], "n_valid": s["n_valid"],
            "class": s["class"], "class_late": cls["late"], "class_full": cls["full"], "met": s["met"],
            "worst_R": s["worst_R"], "pooled": s["pooled"], "conclusions": s["conclusions"], "runs": runs}


# ---------------------------------------------------------------- hypotheses (039)

def _ckv_verdict(est: dict, c_kv: int) -> str:
    v = ca.h_s2(est, c_kv=c_kv)
    return "not supported" if v == "refuted" else v


def h_greater(est: dict, bound: int) -> str:
    """C_PROD > bound: a value above it, exact or a lower bound, supports; an exact or infeasible value at or below
    it does not; a lower bound at or below it is inconclusive."""
    status, value = est["status"], est["value"]
    if status in ("NT", "UNSTABLE"):
        return "NT"
    if value > bound:
        return "supported"
    return "not supported" if status in ("EXACT", "INFEASIBLE") else "inconclusive"


def h_less(est: dict, bound: int) -> str:
    """C_PROD < bound: an exact or infeasible value below it supports; any value at or above it (a lower bound
    included) does not; a lower bound below it is inconclusive."""
    status, value = est["status"], est["value"]
    if status in ("NT", "UNSTABLE"):
        return "NT"
    if value >= bound:
        return "not supported"
    return "supported" if status in ("EXACT", "INFEASIBLE") else "inconclusive"


def h_open(x: dict) -> str:
    """lambda_PROD < lambda_ref on the grid's own factors (the rates are those factors x lambda_ref rounded to 4
    decimals): supported when exact (or infeasible) at a factor of at most 0.9, not supported at a factor of at least 1
    (a lower bound included), else inconclusive."""
    est, factor = x["c_prod"], x["factor_prod"]
    if est["status"] in ("NT", "UNSTABLE"):
        return "NT"
    if Fraction(str(factor)) >= 1:
        return "not supported"
    if est["status"] in ("EXACT", "INFEASIBLE") and Fraction(str(factor)) <= Fraction(9, 10):
        return "supported"
    return "inconclusive"


def h_verify(klass: str) -> str:
    return {"PASS": "supported", "FAIL": "not supported", "MARGINAL": "inconclusive"}.get(klass, "NT")


def hypotheses(kind: str, workloads: dict) -> dict:
    out = {}
    if kind == "closed":
        if "U32K" in workloads:
            out["H-D1"] = {"U32K": {"c_slo": workloads["U32K"]["c_slo"], "c_kv": workloads["U32K"]["c_kv"],
                                    "verdict": _ckv_verdict(workloads["U32K"]["c_slo"], workloads["U32K"]["c_kv"]),
                                    "prediction": "supported"}}
        if "M8K-T" in workloads:
            out["H-D2a"] = {"M8K-T": {"c_prod": workloads["M8K-T"]["c_prod"], "c1b_c_prod": C1B_C_PROD["M8K"],
                                      "verdict": h_greater(workloads["M8K-T"]["c_prod"], C1B_C_PROD["M8K"]),
                                      "prediction": "supported"}}
        if "U8K-T" in workloads:
            out["H-D2b"] = {"U8K-T": {"c_prod": workloads["U8K-T"]["c_prod"], "c1b_c_prod": C1B_C_PROD["U8K"],
                                      "verdict": h_less(workloads["U8K-T"]["c_prod"], C1B_C_PROD["U8K"]),
                                      "prediction": "supported"}}
        o512 = {w: r for w, r in workloads.items() if r["out"] == 512}
        if o512:
            out["H-D3"] = {w: {"c_slo": r["c_slo"], "c_kv": r["c_kv"], "verdict": _ckv_verdict(r["c_slo"], r["c_kv"]),
                               "prediction": "supported"} for w, r in o512.items()}
    elif kind == "open":
        out["H-D4"] = {w: {"rate_prod": r["rate_prod"], "factor_prod": r["factor_prod"],
                           "lambda_ref": r["lambda_ref"]["lambda_ref"], "verdict": h_open(r), "prediction": "supported"}
                       for w, r in workloads.items()}
    else:
        out["H-D6"] = {w: {"class": r["class"], "verdict": h_verify(r["class"]), "prediction": "supported"}
                       for w, r in workloads.items() if r["mix"] == "var"}
        out["H-D7"] = {w: {"class": r["class"], "verdict": h_verify(r["class"]), "prediction": "supported"}
                       for w, r in workloads.items() if r["via"] == "gateway"}
    return out


# ---------------------------------------------------------------- the analysis

def _sentinel(plan: dict, rows: list, requests: dict, position: str) -> dict:
    s = plan["sentinel"]
    row = next((r for r in rows if r.get("phase") == "sentinel" and r.get("position") == position), None)
    return ca.sentinel_summary(row, requests.get(row.get("tag")) if row else None, C=s["C"], N=s["N"])


def _workloads(plan: dict, confirm: dict, rows: list, requests: dict, slo: dict) -> dict:
    if plan["kind"] == "verify":
        return {w: verify_analysis(plan, w, rows, requests, slo=slo) for w in plan["workloads"]}
    return {w: search_analysis(plan, confirm, w, rows, requests, slo=slo) for w in plan["workloads"]}


def analyze(cap: dict, requests: dict, *, references: dict | None = None) -> dict:
    """analysis.json body (schema_version 1) of one Stage D session; the report adds provenance and platform."""
    plan, confirm, session, rows = cap["plan"], cap.get("confirm_plan"), cap["session"], cap["rows"]
    replay = replay_problems(plan, confirm, session.get("boots") or [], rows, requests,
                             confirm_sha256=session.get("confirm_plan_sha256"))
    sent = {pos: _sentinel(plan, rows, requests, pos) for pos in ("start", "end")}
    workloads = _workloads(plan, confirm, rows, requests, plan["slo"])
    secondary = {}
    for sid, slo in plan["secondary_slo"].items():
        res = _workloads(plan, confirm, rows, requests, slo)
        keys = ("class",) if plan["kind"] == "verify" else ("c_slo", "c_prod", "binding", "flags")
        secondary[sid] = {"slo": slo, "descriptive": True,
                          "workloads": {w: {k: r[k] for k in keys} for w, r in res.items()}}
    runs = []
    for row in rows:
        ok, reasons = False, ["not a row of this plan"]
        if (row.get("workload") in plan["workloads"] or row.get("phase") == "sentinel") and isinstance(row.get("C"),
                                                                                                      int):
            ok, reasons, _ = _eval_row(row, requests, plan, plan["slo"])
        runs.append({k: row.get(k) for k in ("phase", "workload", "C", "block", "position", "tag")}
                    | {"valid": ok, "reasons": reasons})
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "session": plan["session"], "kind": plan["kind"],
            "slo": plan["slo"],
            "protocol": {"seed": plan["seed"], "window": plan["window"], "guard_band": plan["guard_band"],
                         "n_low": plan["n_low"], "open_loop": plan["open_loop"]},
            "replay": {"ok": not replay, "problems": replay},
            "sentinel": {**sent, "gate": ca.sentinel_gate(sent["start"], sent["end"],
                                                          tolerance=plan["sentinel"]["tolerance"]),
                         "cross_session_shift": ca.cross_session_shift([sent["start"], sent["end"]], references)},
            "workloads": workloads, "secondary_slo": secondary, "hypotheses": hypotheses(plan["kind"], workloads),
            "not_supported": list(NOT_SUPPORTED), "runs": runs}


# ---------------------------------------------------------------- D5: the start-up wave (descriptive, c1b only)

def startup_wave(c1b_dir) -> dict:
    """For each c1b workload and each C with a valid run (Phase S or C): the largest first-wave TTFT (idx < C) per run,
    their mean and max, and (uniform) a least-squares line of the per-C mean against C. Descriptive and post hoc: 030
    reported the start-up wave only at C_PROD; nothing here is a hypothesis test."""
    d = Path(c1b_dir)
    plan = json.loads((d / "plan.json").read_text())
    rows = [json.loads(line) for line in (d / "rows.jsonl").read_text().splitlines() if line.strip()]
    out = {}
    for w, wl in plan["workloads"].items():
        by_c = {}
        for row in rows:
            if row.get("workload") != w or row.get("phase") not in ("S", "C") or not row.get("derived"):
                continue
            reqs = [json.loads(x) for x in (d / row["derived"]).read_text().splitlines() if x.strip()]
            ok, _ = ca.run_valid(row, reqs, mix=wl["mix"], C=row["C"], N=row["N"])
            if ok:
                by_c.setdefault(row["C"], []).append(max(q["ttft_ms"] for q in reqs if q["idx"] < row["C"]))
        points = {str(C): {"runs": len(v), "mean_ms": statistics.fmean(v), "max_ms": max(v)}
                  for C, v in sorted(by_c.items())}
        fit = None
        if wl["mix"] == "uniform" and len(by_c) >= 2:
            xs = sorted(by_c)
            ys = [statistics.fmean(by_c[C]) for C in xs]
            slope, intercept = statistics.linear_regression(xs, ys)
            resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
            fit = {"slope_ms_per_request": slope, "intercept_ms": intercept,
                   "max_abs_residual_ms": max(abs(r) for r in resid), "points": len(xs)}
        out[w] = {"L": wl["L"], "mix": wl["mix"], "points": points, "fit": fit}
    return {"schema_version": SCHEMA_VERSION, "source_session": d.name, "descriptive": True, "post_hoc": True,
            "definition": "largest TTFT of idx < C per valid run (Phase S and C), its mean and max over runs at each C",
            "workloads": out}



D5_OUT = ("results", "stage_d", "d5-startup-wave", "analysis.json")


def main(argv: list | None = None) -> int:
    """`python3 benchmarks/stage_d_analysis.py startup-wave --c1b results/capacity/formal-20261017-c1b`: writes D5's
    description to results/stage_d/d5-startup-wave/analysis.json (039 D5; no GPU, descriptive)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Stage D analysis helpers (039)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d5 = sub.add_parser("startup-wave", help="039 D5: the start-up wave of the c1b session (descriptive)")
    d5.add_argument("--c1b", required=True)
    args = parser.parse_args(argv)
    c1b = Path(args.c1b)
    if c1b.resolve().name != sd.C1B_SESSION:
        print(f"stage_d_analysis: --c1b must be the {sd.C1B_SESSION} session (039 D5)", file=sys.stderr)
        return 2
    out = sd.ROOT.joinpath(*D5_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(startup_wave(c1b), indent=1) + "\n")
    print(json.dumps({"out": str(out.relative_to(sd.ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
