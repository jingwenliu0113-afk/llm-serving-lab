"""Tests for benchmarks/stage_d_analysis.py (decision 039): Stage D runs judged with Stage C's frozen statistics
(tiers, output lengths, var mixes, the open-loop window), the replay of D-A / D-B / D-C sessions, the 039
hypotheses and the D5 start-up wave. simulate() plays a whole session from per-point latencies, with the same
capacity_plan functions the runner uses, so every estimand below comes from a replayable session."""

import copy
import json
import math
import tomllib
from pathlib import Path

import pytest

import capacity_plan as cp
import core_b_plan as bp
import stage_d_analysis as sda
import stage_d_plan as sd

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_d.toml").read_text())
C1B = ROOT / "results" / "capacity" / "formal-20261017-c1b"
UNIFORM_PLAN = {"num_gpu_blocks": PINS["stage_b"]["num_gpu_blocks"], "block_size": PINS["stage_b"]["block_size"]}
REFS = {"U2K": {"workload": "U2K", "c_prod": 32, "lambda_ref": 2.5, "mean_latency_s": 12.8, "runs": [], "requests": 0},
        "U8K": {"workload": "U8K", "c_prod": 8, "lambda_ref": 0.6, "mean_latency_s": 13.3, "runs": [], "requests": 0}}


def plan(session):
    return sd.build_plan(CFG, PINS, session_name=session, uniform_plan=UNIFORM_PLAN, refs=REFS, pins_sha256="p",
                         stage_d_sha256="s")


def requests(x, n, *, ttft, tpot, long_ttft=None):
    """N derived requests: mixed puts a long one at idx = 5 (mod 10), as 007's blocks of ten."""
    out = []
    for idx in range(n):
        cls = "long" if x["mix"] == "mixed" and idx % 10 == 5 else "short"
        t = long_ttft if cls == "long" else ttft
        out.append({"idx": idx, "cls": cls, "ttft_ms": float(t), "latency_ms": float(t) + (x["out"] - 1) * tpot,
                    "out": x["out"]})
    return out


def simulate(p, latency, *, invalid=()):
    """Run p's state machine (039: 030's for D-A / D-B, blocks for D-C). latency(w, point) -> (ttft_ms, tpot_ms,
    long_ttft_ms); a (w, point, phase) in invalid makes that run invalid. Returns the cap dict analyze() takes and
    the derived requests."""
    boots, rows, reqs_by_tag = [], [], {}

    def boot(phase, w, *, block=None, attempt=None, position=None):
        b = {"boot": len(boots), "phase": phase, "workload": w, "block": block, "attempt": attempt,
             "position": position, "reached_first_bench": True, "problems": []}
        boots.append(b)
        return b

    def run(b, x_point):
        w, phase = b["workload"], b["phase"]
        x = sda.spec(p, w)
        row = {"phase": phase, "workload": w, "C": x_point, "block": b["block"], "position": b["position"],
               "boot": b["boot"], "valid": True, "reasons": [], "output_throughput": 300.0, "metrics": {}}
        row["tag"] = sd.run_tag(phase, w, x_point, block=b["block"], position=b["position"], kind=p["kind"])
        row["N"] = sda.run_n(p, row)
        row["timeout_s"] = (p["sentinel"]["timeout_s"] if phase == "sentinel" else
                            p["workloads"][w]["timeouts_s"][str(x_point)] if p["kind"] == "closed" else
                            p["workloads"][w]["points"][str(x_point)]["timeout_s"] if p["kind"] == "open" else
                            p["workloads"][w]["timeout_s"])
        if (w, x_point, phase) in invalid:
            row.update(valid=False, reasons=["simulated invalid run"])
            reqs = None
        else:
            ttft, tpot, long_ttft = latency(w, x_point) if phase != "sentinel" else (800.0, 40.0, None)
            reqs = requests(x, row["N"], ttft=ttft, tpot=tpot, long_ttft=long_ttft)
            reqs_by_tag[row["tag"]] = reqs
        if phase == "S":
            bits = sda.search_bits(row, reqs, p)
            row["search_pass"], row["within_band"] = bits["pass"], bits["within_band"]
        elif phase == "C":
            row["outcome"] = sda.phase_c_outcome(row, reqs, p)
        rows.append(row)
        return row

    s = p["sentinel"]
    run(boot("sentinel", s["workload"], position="start"), s["C"])
    confirm = None
    if p["kind"] == "verify":
        n = max(x["n"] for x in p["workloads"].values())
        for r in range(1, n + 1):
            for w in cp.block_order(p["seed"], r, list(p["workloads"])):
                run(boot("V", w, block=r), p["workloads"][w]["C"])
    else:
        searches = {}
        for w, x in p["workloads"].items():
            b, hist = boot("S", w, attempt=1), []
            while (c := cp.search_next(x["c_cap"], [(h["C"], h["pass"]) for h in hist])) is not None:
                row = run(b, c)
                hist.append({"C": c, "pass": row["search_pass"], "within_band": row["within_band"]})
            searches[w] = hist
        confirm = cp.build_confirm_plan(p, searches)
        outcomes = {w: [] for w in p["workloads"]}
        for r in range(1, confirm["blocks"] + 1):
            new = []
            for w, points in cp.phase_c_boots(confirm, r, outcomes):
                b = boot("C", w, block=r)
                for c in points:
                    new.append((w, (r, c, run(b, c)["outcome"])))
            for w, o in new:
                outcomes[w].append(o)
    run(boot("sentinel", s["workload"], position="end"), s["C"])
    session = {"boots": boots, "confirm_plan_sha256": cp.confirm_plan_sha256(confirm) if confirm else None}
    return {"plan": p, "confirm_plan": confirm, "session": session, "rows": rows}, reqs_by_tag


def step(limit, *, ok=(500.0, 50.0), bad=(500.0, 120.0)):
    """A latency function: pass (well inside the band) up to `limit`, fail (TPOT 120 ms > 100) above."""
    return lambda w, c: (*(ok if c <= limit[w] else bad), 5000.0)       # a mixed long request: inside 20 s


# ------------------------------------------------------------------ per run --

def test_a_512_token_run_is_judged_on_its_tier_with_511_token_tpot():
    p = plan("D-A2")
    x = p["workloads"]["U2K-O512"]
    row = {"phase": "S", "workload": "U2K-O512", "C": 4, "N": 200, "valid": True, "reasons": []}
    reqs = requests(x, 200, ttft=1500.0, tpot=96.0)
    ok, _, e = sda.evaluate(row, reqs, p)
    assert ok and e["Y"]["short"] == pytest.approx(1500.0) and e["Y"]["tpot"] == pytest.approx(96.0)
    assert e["R"]["short"] == pytest.approx(0.75) and e["R"]["tpot"] == pytest.approx(0.96) and not e["fail"]
    assert e["within_band"] is False                                       # 0.96 > 1 - m
    bad = [dict(q, out=128) for q in reqs]
    assert sda.evaluate(row, bad, p)[1] == ["out not all 512"]


def test_open_loop_runs_use_the_trimmed_window():
    p = plan("D-B")
    x = p["workloads"]["U2K-P"]
    n = x["points"]["6"]["N"]
    row = {"phase": "S", "workload": "U2K-P", "C": 6, "N": n, "valid": True, "reasons": []}
    reqs = requests(x, n, ttft=500.0, tpot=50.0)
    k = math.ceil(0.1 * n)
    for q in reqs[:k] + reqs[n - k:]:                                       # outside W: never judged
        q["ttft_ms"] = 99999.0
    ok, _, e = sda.evaluate(row, reqs, p)
    assert ok and e["Y"]["short"] == pytest.approx(500.0) and e["n"]["all"] == n - 2 * k
    assert sda.window_c(p, row, n) == k and sda.run_n(p, row) == n


def test_var_and_mixed_classes():
    pv, pm = plan("D-C"), plan("D-A1")
    assert sda.derived_problems(requests(pv["workloads"]["U2K-var"], 320, ttft=1, tpot=1), x=pv["workloads"]["U2K-var"],
                                c=32, n=320) == []
    m = requests(pm["workloads"]["M8K-T"], 200, ttft=1, tpot=1, long_ttft=5)
    assert sda.derived_problems(m, x=pm["workloads"]["M8K-T"], c=8, n=200) == []
    m[5]["cls"] = "short"
    assert any("long requests" in p for p in sda.derived_problems(m, x=pm["workloads"]["M8K-T"], c=8, n=200))


# ------------------------------------------------------------------ D-A --

def test_a_closed_session_replays_and_yields_030_estimands_and_039_hypotheses():
    p = plan("D-A1")
    cap, reqs = simulate(p, step({"U32K": 2, "U8K-T": 5, "M8K-T": 3}))
    a = sda.analyze(cap, reqs)
    assert a["replay"] == {"ok": True, "problems": []} and a["sentinel"]["gate"]["ok"]
    w = a["workloads"]
    assert (w["U32K"]["c_prod"], w["U8K-T"]["c_prod"], w["M8K-T"]["c_prod"]) == (
        {"value": 2, "status": "EXACT"}, {"value": 5, "status": "EXACT"}, {"value": 3, "status": "EXACT"})
    assert w["U32K"]["binding"] == ["tpot"] and w["U32K"]["K"] == [1, 2, 3]
    h = a["hypotheses"]
    assert h["H-D1"]["U32K"]["verdict"] == "supported" and h["H-D2a"]["M8K-T"]["verdict"] == "supported"
    assert h["H-D2b"]["U8K-T"]["verdict"] == "supported" and "H-D3" not in h
    assert set(a["secondary_slo"]) == {"S1", "S3"}


def test_the_512_token_session_judges_h_d3_against_c_kv():
    p = plan("D-A2")
    cap, reqs = simulate(p, step({"U2K-O512": 40, "U8K-O512": 9}))
    a = sda.analyze(cap, reqs)
    assert a["replay"]["ok"]
    h = a["hypotheses"]["H-D3"]
    assert h["U2K-O512"]["verdict"] == "not supported"                      # C_SLO 40 > C_KV 30
    assert h["U8K-O512"]["verdict"] == "supported" and a["workloads"]["U8K-O512"]["c_slo"]["value"] == 9


def test_the_replay_catches_a_wrong_record():
    p = plan("D-A1")
    cap, reqs = simulate(p, step({"U32K": 2, "U8K-T": 5, "M8K-T": 3}))
    bad = copy.deepcopy(cap)
    next(r for r in bad["rows"] if r["phase"] == "S")["search_pass"] = False
    assert any("search_pass" in x for x in sda.replay_problems(p, bad["confirm_plan"], bad["session"]["boots"],
                                                                 bad["rows"], reqs))
    bad = copy.deepcopy(cap)
    bad["rows"][1]["timeout_s"] += 60
    assert any("timeout_s" in x for x in sda.replay_problems(p, bad["confirm_plan"], bad["session"]["boots"],
                                                               bad["rows"], reqs))
    bad = copy.deepcopy(cap)
    bad["confirm_plan"]["workloads"]["U32K"]["points"][0]["n"] = 1
    assert any("confirm plan" in x for x in sda.replay_problems(p, bad["confirm_plan"], bad["session"]["boots"],
                                                                  bad["rows"], reqs))


def test_an_invalid_phase_c_run_drops_its_point_and_above():
    p = plan("D-A1")
    cap, reqs = simulate(p, step({"U32K": 2, "U8K-T": 5, "M8K-T": 3}), invalid={("U32K", 2, "C")})
    a = sda.analyze(cap, reqs)
    assert a["replay"]["ok"]
    u = a["workloads"]["U32K"]
    assert u["points"]["2"]["class"] == "NT" and u["c_prod"] == {"value": 1, "status": "LOWER"}


# ------------------------------------------------------------------ D-B --

@pytest.mark.parametrize("limit, rate_factor, verdict", [(5, 0.9, "supported"), (6, 1.0, "not supported"),
                                                         (9, 1.5, "not supported")])
def test_an_open_session_reports_rates_and_h_d4(limit, rate_factor, verdict):
    p = plan("D-B")
    cap, reqs = simulate(p, step({"U2K-P": limit, "U8K-P": limit}))
    a = sda.analyze(cap, reqs)
    assert a["replay"]["ok"]
    u = a["workloads"]["U2K-P"]
    assert u["c_prod"]["value"] == limit and u["rate_prod"] == pytest.approx(round(rate_factor * 2.5, 4))
    assert a["hypotheses"]["H-D4"]["U2K-P"]["verdict"] == verdict
    assert "ABOVE_CKV" not in u["flags"]
    if limit == 9:
        assert u["c_prod"]["status"] == "CENSORED"


def test_an_open_session_below_the_grid_is_infeasible():
    p = plan("D-B")
    cap, reqs = simulate(p, step({"U2K-P": 0, "U8K-P": 0}))
    u = sda.analyze(cap, reqs)["workloads"]["U2K-P"]
    assert u["c_prod"] == {"value": 0, "status": "INFEASIBLE"} and u["rate_prod"] == 0


# ------------------------------------------------------------------ D-C --

def test_a_verify_session_runs_n_boots_per_point_and_judges_h_d6_h_d7():
    p = plan("D-C")
    lat = lambda w, c: (500.0, 120.0 if w == "U8K-gw" else 50.0, None)   # noqa: E731
    cap, reqs = simulate(p, lat)
    a = sda.analyze(cap, reqs)
    assert a["replay"]["ok"] and len(cap["session"]["boots"]) == 2 + 4 * 5
    w = a["workloads"]
    assert {k: v["class"] for k, v in w.items()} == {"U2K-var": "PASS", "U8K-var": "PASS", "U2K-gw": "PASS",
                                                     "U8K-gw": "FAIL"}
    h = a["hypotheses"]
    assert h["H-D6"]["U2K-var"]["verdict"] == "supported" and h["H-D7"]["U8K-gw"]["verdict"] == "not supported"


def test_a_verify_session_out_of_order_is_caught():
    p = plan("D-C")
    cap, reqs = simulate(p, lambda w, c: (500.0, 50.0, None))
    boots = copy.deepcopy(cap["session"]["boots"])
    boots[1]["workload"], boots[2]["workload"] = boots[2]["workload"], boots[1]["workload"]
    assert sda.replay_problems(p, None, boots, cap["rows"], reqs)


# ------------------------------------------------------------------ hypotheses --

@pytest.mark.parametrize("est, gt, lt", [
    ({"value": 3, "status": "EXACT"}, "supported", "supported"),
    ({"value": 2, "status": "EXACT"}, "not supported", "supported"),
    ({"value": 2, "status": "LOWER"}, "inconclusive", "inconclusive"),
    ({"value": 9, "status": "LOWER"}, "supported", "not supported"),
    ({"value": 0, "status": "INFEASIBLE"}, "not supported", "supported"),
    ({"value": None, "status": "NT"}, "NT", "NT"),
])
def test_h_d2_rules(est, gt, lt):
    assert sda.h_greater(est, 2) == gt and sda.h_less(est, 8) == lt


def test_h_verify_rules():
    assert [sda.h_verify(k) for k in ("PASS", "FAIL", "MARGINAL", "NT")] == [
        "supported", "not supported", "inconclusive", "NT"]


# ------------------------------------------------------------------ D5 --

def test_the_start_up_wave_is_described_from_c1b():
    out = sda.startup_wave(C1B)
    u2k = out["workloads"]["U2K"]
    assert out["descriptive"] and out["post_hoc"] and out["source_session"] == "formal-20261017-c1b"
    assert u2k["points"]["32"]["max_ms"] == pytest.approx(10350, abs=10)    # 035: uniform-2k 10.35 s at C_PROD
    assert u2k["fit"]["slope_ms_per_request"] > 0 and out["workloads"]["M2K"]["fit"] is None
    assert out["workloads"]["U8K"]["points"]["8"]["max_ms"] == pytest.approx(11040, abs=10)
