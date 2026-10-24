#!/usr/bin/env python3
"""Stage V's pre-freeze design check (decision 044 V4 「凍結前的設計檢查」): a desk discrete-event simulation of
weighted-v1's strict FIFO at W-90 and W-50, estimating U_W per mix from committed per-request timings only (c1b and
D-C). Descriptive only: no Stage V data, no hypothesis test and no SLO prediction (044 揭露: it only checks that the
non-trivial threshold is reachable). Its output, results/stage_v/prereg-sim/sim.json, is committed with the freeze.

- A run (simulate, simulate_run): the V-ADM plan's W point (C 32 clients, N 320, weighted-v1's budget, units and
  caps) on one of its five pinned sequences (stage_v_analysis.planned: the block's dataset, pure in the seed). The
  main run starts empty (the bench awaits its warmups); idx 0..C-1 arrive at 0 in idx order; idx k >= C arrives a
  client turnaround after the (k - C + 1)-th release (vLLM's bench wakes its semaphore waiters in idx order). The
  gateway (capserve's PolicyController, never imported): one strict FIFO in arrival order, re-pumped after every
  event; the head is sent when its units fit in the free budget and its class is below its cap, and nothing overtakes
  a head that does not fit (head-of-line blocking). The gateway's own parse and tokenize time is in no committed file
  and is left out.
- In vLLM: a request of the 2K class (v-uniform-2k) stays a draw from c1b's U2K at C_PROD 32, one of the 8K class
  (v-uniform-8k) a draw from c1b's U8K at C_PROD 8 (duration_pools: the e2e latencies of the valid Phase C runs at
  C_PROD, run_valid, as stage_d_plan.lambda_ref; direct, so the time in vLLM): a request sent at 0 (the start-up
  wave) from idx < C, every other from W = [C, N - C). This is 043 7.3 A's premise, each class alone filling the
  budget at its C_PROD (a measured state); no mixed batch was ever measured.
- The turnaround (turnaround_pool): D-C's closed loop through the gateway at C 32 (U2K-gw, the W points' client
  concurrency), start_s of idx k minus the (k - C + 1)-th earliest client finish (start_s + latency) over its valid
  runs.
- U_W, N_W, (j) and the non-trivial condition are stage_v_analysis's own (in_vllm on the derived rows' times, rounded
  to the microsecond as gateway_requests rounds them; inflight_problems; nontrivial): the estimator V4 judges with. A
  simulated run that breaks (j) is a SimError.
- Variants (VARIANTS): "committed" (the above; the design check), "steady_only" (every duration from W), and two
  what-ifs on the one premise no committed file covers, an 8K request's time in vLLM x 0.8 / x 1.25 against a 2K
  one's. REPS runs per block, each (variant, point, block) on its own seeded Random (_rng).
- The design check (design_check, 044): per mix, the mean U_W over the committed variant's runs >= 0.93 x the budget
  (29.76); below it a design decision is needed before the freeze. This script never changes a threshold or the
  design.

Usage (Mac, CPU only):
    .venv/bin/python benchmarks/stage_v_prereg_sim.py

Stdlib only, Python 3.10 compatible (tomllib only in the CLI).
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import random
import statistics
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Callable

import capacity_analysis as ca
import capacity_plan as cp
import stage_d_plan as sd
import stage_v_analysis as sva
import stage_v_datasets as svd
import stage_v_plan as sv

SCHEMA_VERSION = 1
ROOT = sv.ROOT
SESSION = "V-ADM"
POINTS = ("W-90", "W-50")
C1B_DIR = ROOT / "results" / "capacity" / sd.C1B_SESSION
D_C_DIR = ROOT.joinpath(*sd.RESULTS, "formal-20261023-d-c")
OUT = ("results", "stage_v", "prereg-sim", "sim.json")
DURATION_BASES = {svd.PROMPT_2K: "U2K", svd.PROMPT_8K: "U8K"}   # a W prompt -> the c1b workload of its class
TURNAROUND_BASE = "U2K-gw"
REPS = 20
DESIGN_SHARE = Fraction(93, 100)
VARIANTS = {"committed": {"first_wave": True, "scale_8k": 1.0},
            "steady_only": {"first_wave": False, "scale_8k": 1.0},
            "8k_x0.8": {"first_wave": True, "scale_8k": 0.8},
            "8k_x1.25": {"first_wave": True, "scale_8k": 1.25}}
DESIGN_VARIANT = "committed"
NOTE = ("044 V4's pre-freeze design check: a desk simulation on committed c1b and D-C timings; not Stage V data, not "
        "a test, no SLO prediction")


class SimError(Exception):
    pass


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rng(name: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{sv.SEED}:stage_v:prereg-sim:{name}".encode()).hexdigest(), 16))


# ---------------------------------------------------------------- committed timings

def duration_pools(c1b_dir) -> dict:
    """{workload: {c_prod, runs, first, steady}} for U2K and U8K: the e2e latencies (s) of the valid c1b Phase C runs
    at C_PROD, idx < C (the start-up wave) and W = [C, N - C) apart."""
    d = Path(c1b_dir)
    analysis = json.loads((d / "analysis.json").read_text())
    rows = _read_jsonl(d / "rows.jsonl")
    pools = {}
    for w in sorted(set(DURATION_BASES.values())):
        est = analysis["workloads"][w]["c_prod"]
        if est.get("status") != "EXACT" or not isinstance(est.get("value"), int) or est["value"] < 1:
            raise SimError(f"c1b C_PROD({w}) {est!r} is not an EXACT positive value")
        C = est["value"]
        runs, first, steady = [], [], []
        for row in rows:
            if row.get("phase") != "C" or row.get("workload") != w or row.get("C") != C or not row.get("derived"):
                continue
            reqs = _read_jsonl(d / row["derived"])
            if ca.run_valid(row, reqs, mix="uniform", C=C, N=row["N"])[0]:
                runs.append(row["derived"])
                first += [q["latency_ms"] / 1000 for q in reqs if q["idx"] < C]
                steady += [q["latency_ms"] / 1000 for q in reqs if C <= q["idx"] < row["N"] - C]
        if not runs:
            raise SimError(f"no valid c1b Phase C run of {w} at C_PROD {C}")
        pools[w] = {"c_prod": C, "runs": runs, "first": first, "steady": steady}
    return pools


def turnaround_pool(d_c_dir) -> dict:
    """{C, runs, gaps}: per valid D-C run of TURNAROUND_BASE, start_s of idx k >= C minus the (k - C + 1)-th earliest
    client finish (start_s + latency), in s."""
    d = Path(d_c_dir)
    Cs, runs, gaps = set(), [], []
    for row in _read_jsonl(d / "rows.jsonl"):
        if row.get("phase") != "V" or row.get("workload") != TURNAROUND_BASE or not row.get("derived"):
            continue
        reqs = _read_jsonl(d / row["derived"])
        C, N = row["C"], row["N"]
        if not ca.run_valid(row, reqs, mix="uniform", C=C, N=N)[0]:
            continue
        if not all(ca._num(q.get("start_s")) for q in reqs):
            raise SimError(f"{row['tag']}: start_s missing")
        by = sorted(reqs, key=lambda q: q["idx"])
        finish = sorted(q["start_s"] + q["latency_ms"] / 1000 for q in by)
        run = [by[k]["start_s"] - finish[k - C] for k in range(C, N)]
        if min(run) < 0:
            raise SimError(f"{row['tag']}: an idx was sent before the release that frees its client: not a closed "
                           f"loop of C {C} clients")
        Cs.add(C)
        runs.append(row["derived"])
        gaps += run
    if len(Cs) != 1:
        raise SimError(f"no valid D-C run of {TURNAROUND_BASE} at one C (C {sorted(Cs)})")
    return {"C": Cs.pop(), "runs": runs, "gaps": gaps}


# ---------------------------------------------------------------- the simulation

def simulate(demands: list, *, C: int, budget: int, caps: dict, duration: Callable, turnaround: Callable) -> dict:
    """One closed-loop run of weighted-v1's strict FIFO. demands[i] = (class name, units) of idx i; duration(i, sent)
    -> its seconds in vLLM; turnaround() -> the seconds from a release to the next idx's arrival. -> {"times":
    [(arrival, sent, released)] per idx, "peak_queue": the most requests ever waiting after a pump}."""
    n = len(demands)
    arrival, sent, released = [None] * n, [None] * n, [None] * n
    events, order = [], itertools.count()
    queue, free, per = deque(), budget, dict.fromkeys(caps, 0)
    done = peak = 0
    for i in range(min(C, n)):
        heapq.heappush(events, (0.0, next(order), "arrive", i))
    while events:
        now, _, kind, i = heapq.heappop(events)
        name, units = demands[i]
        if kind == "release":
            released[i] = now
            free += units
            per[name] -= 1
            done += 1
            k = C - 1 + done
            if k < n:
                heapq.heappush(events, (now + turnaround(), next(order), "arrive", k))
        else:
            arrival[i] = now
            queue.append(i)
        while queue:
            name, units = demands[queue[0]]
            if units > free or per[name] >= caps[name]:
                break
            h = queue.popleft()
            sent[h] = now
            free -= units
            per[name] += 1
            heapq.heappush(events, (now + duration(h, now), next(order), "release", h))
        peak = max(peak, len(queue))
    return {"times": list(zip(arrival, sent, released)), "peak_queue": peak}


def simulate_run(plan: dict, point: str, block: int, *, rng: random.Random, pools: dict, gaps: list,
                 variant: str) -> dict:
    """One simulated run of a W point's block: {"reqs": the derived rows' idx, units and gw_* times (s from the first
    arrival, rounded to the microsecond), "peak_queue"}."""
    x = plan["points"][point]
    rows = sva.planned(plan, point, block)
    classes = dict(zip((svd.PROMPT_2K, svd.PROMPT_8K), x["admission_params"]["classes"]))
    demands = [(classes[p]["name"], classes[p]["units"]) for p, _ in rows]
    v = VARIANTS[variant]

    def duration(i: int, at: float) -> float:
        pool = pools[DURATION_BASES[rows[i][0]]]
        s = rng.choice(pool["first"] if v["first_wave"] and at == 0 else pool["steady"])
        return s * v["scale_8k"] if rows[i][0] == svd.PROMPT_8K else s

    out = simulate(demands, C=x["C"], budget=x["admission_params"]["budget"],
                   caps={c["name"]: c["max_concurrency"] for c in x["admission_params"]["classes"]},
                   duration=duration, turnaround=lambda: rng.choice(gaps))
    reqs = [{"idx": i, "units": u, "gw_arrival_s": round(a, 6), "gw_sent_s": round(s, 6),
             "gw_released_s": round(r, 6)} for i, ((_, u), (a, s, r)) in enumerate(zip(demands, out["times"]))]
    return {"reqs": reqs, "peak_queue": out["peak_queue"]}


def run_stats(plan: dict, point: str, run: dict) -> dict:
    """U_W and N_W (exact), trivial (U_W < 0.9 x the budget), the peak queue and the bench time of one simulated run,
    by stage_v_analysis's own functions; (j) broken is a SimError."""
    x = plan["points"][point]
    iv = sva.in_vllm(run["reqs"], C=x["C"], N=x["N"])
    problems = sva.inflight_problems(iv, x=x)
    if problems:
        raise SimError(f"{point}: the simulation broke the policy: {'; '.join(problems)}")
    return {"u_w": iv["_u_w"], "n_w": iv["_n_w"], "trivial": not sva.nontrivial(iv, x=x, plan=plan)["ok"],
            "peak_queue": run["peak_queue"], "bench_s": max(q["gw_released_s"] for q in run["reqs"])}


def _summary(stats: list) -> dict:
    def spread(key: str) -> dict:
        values = [s[key] for s in stats]
        return {"mean": round(float(sum(values) / len(values)), 4), "min": round(float(min(values)), 4),
                "max": round(float(max(values)), 4)}

    return {"runs": len(stats), "u_w": spread("u_w"), "n_w": spread("n_w"),
            "trivial_runs": sum(s["trivial"] for s in stats), "peak_queue": max(s["peak_queue"] for s in stats),
            "bench_s": round(statistics.fmean(s["bench_s"] for s in stats), 1)}


def simulate_point(plan: dict, point: str, *, pools: dict, gaps: list, variant: str, reps: int = REPS) -> dict:
    """reps simulated runs of each block's pinned sequence: the summary over all of them and per block, and the
    exact mean U_W ("_mean_u_w")."""
    x = plan["points"][point]
    blocks, every = {}, []
    for r in range(1, x["n"] + 1):
        rng = _rng(f"{variant}:{point}:{r}")
        stats = [run_stats(plan, point, simulate_run(plan, point, r, rng=rng, pools=pools, gaps=gaps,
                                                     variant=variant)) for _ in range(reps)]
        key = x["datasets"][str(r)]
        blocks[str(r)] = {"dataset": key, "long_in_w": svd.window_counts(key, x["C"])["long"], **_summary(stats)}
        every += stats
    return {**_summary(every), "blocks": blocks, "_mean_u_w": sum(s["u_w"] for s in every) / len(every)}


def design_check(plan: dict, means: dict) -> dict:
    """044: each mix's simulated mean U_W >= 0.93 x the budget, exactly; else a design decision before the freeze."""
    points, threshold = {}, None
    for p in POINTS:
        threshold = DESIGN_SHARE * plan["points"][p]["admission_params"]["budget"]
        points[p] = {"mean_u_w": round(float(means[p]), 4), "ok": means[p] >= threshold}
    return {"variant": DESIGN_VARIANT, "rule": "mean U_W of the variant's runs >= 0.93 x the budget, per mix (044)",
            "threshold": float(threshold), "points": points,
            "decision_needed": not all(x["ok"] for x in points.values())}


def _stats_s(values: list, scale: float = 1.0, digits: int = 4) -> dict:
    return {"n": len(values), "mean": round(statistics.fmean(values) * scale, digits),
            "min": round(min(values) * scale, digits), "max": round(max(values) * scale, digits)}


def prereg_sim(plan: dict, *, c1b_dir=C1B_DIR, d_c_dir=D_C_DIR, reps: int = REPS) -> dict:
    """The whole design check on plan's W points (sim.json)."""
    pools, turn = duration_pools(c1b_dir), turnaround_pool(d_c_dir)
    for p in POINTS:
        if turn["C"] != plan["points"][p]["C"]:
            raise SimError(f"the turnaround's {TURNAROUND_BASE} runs are at C {turn['C']}, {p}'s clients at C "
                           f"{plan['points'][p]['C']}")
    points, means = {}, {}
    for p in POINTS:
        x = plan["points"][p]
        adm = x["admission_params"]
        variants = {}
        for v in VARIANTS:
            res = simulate_point(plan, p, pools=pools, gaps=turn["gaps"], variant=v, reps=reps)
            if v == DESIGN_VARIANT:
                means[p] = res["_mean_u_w"]
            variants[v] = {k: res[k] for k in res if not k.startswith("_")}
        points[p] = {"dataset": x["dataset"], "mix": svd.MIXES[x["dataset"]], "C": x["C"], "N": x["N"],
                     "budget": adm["budget"], "classes": adm["classes"],
                     "trivial_threshold": float(Fraction(str(plan["nontrivial"]["u_w"])) * adm["budget"]),
                     "effective_concurrency_043": sva.W_EFFECTIVE[x["dataset"]], "bench_est_s": x["bench_est_s"],
                     "variants": variants}
    c1b, dc = Path(c1b_dir), Path(d_c_dir)
    return {"schema_version": SCHEMA_VERSION, "stage": sv.STAGE, "decision": "044", "descriptive_only": True,
            "note": NOTE,
            "inputs": {"c1b": {"session": c1b.name, "files": {f: _sha256(c1b / f) for f in sorted(
                           ["analysis.json", "rows.jsonl"] + pools["U2K"]["runs"] + pools["U8K"]["runs"])}},
                       "d_c": {"session": dc.name, "files": {f: _sha256(dc / f) for f in sorted(
                           ["rows.jsonl"] + turn["runs"])}}},
            "model": {"reps_per_block": reps, "variants": VARIANTS, "duration_bases": DURATION_BASES,
                      "turnaround_base": TURNAROUND_BASE},
            "pools": {**{w: {"c_prod": x["c_prod"], "runs": x["runs"], "first_s": _stats_s(x["first"]),
                             "steady_s": _stats_s(x["steady"])} for w, x in pools.items()},
                      "turnaround": {"C": turn["C"], "runs": turn["runs"],
                                     "gaps_ms": _stats_s(turn["gaps"], 1000, 3)}},
            "points": points, "design_check": design_check(plan, means)}


def main(argv: list | None = None) -> int:
    import argparse
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    cfg_raw, sd_raw, pins_raw = sv.load_stage_v_bytes(), sd.load_stage_d_bytes(), cp.load_pins_bytes()
    cfg, sd_cfg, pins = (tomllib.loads(raw.decode()) for raw in (cfg_raw, sd_raw, pins_raw))
    plan = sv.build_plan(cfg, pins, sd_cfg, session_name=SESSION, pins_sha256=hashlib.sha256(pins_raw).hexdigest(),
                         stage_v_sha256=hashlib.sha256(cfg_raw).hexdigest(),
                         stage_d_sha256=hashlib.sha256(sd_raw).hexdigest())
    res = prereg_sim(plan)
    out = ROOT.joinpath(*OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"out": str(out.relative_to(ROOT)), "design_check": res["design_check"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
