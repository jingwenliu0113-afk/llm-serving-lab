"""Tests for benchmarks/capacity_analysis.py: the Stage C (C1) per-run statistics, point classes, estimands, flags,
hypotheses, sentinel gate and replay check of decision 030 (PREREGISTRATION §8), written before any C1 data."""

import copy
import json
import tomllib
from pathlib import Path

import pytest

import capacity_analysis as ca
import capacity_plan as cp
import core_b_analysis as ba

ROOT = Path(__file__).resolve().parent.parent
ADV = ROOT / "results" / "advanced" / "formal-20261011-adv" / "derived"
CORE_UNIFORM = ROOT / "results" / "core_b" / "formal-20261009-coreb-uniform"
S2 = cp.SLO_S2
M = 0.05


def adv_requests(name, tpot_ms=30.0):
    """An Advanced derived file in the 030 8.7 schema (idx, cls, ttft_ms, latency_ms, out). The Advanced files carry
    no latency, so latency_ms = ttft + 127 x tpot_ms: TPOT is synthetic here, TTFT / idx / cls are measured."""
    path = ADV / f"{name}-requests.jsonl"
    if not path.is_file():
        pytest.skip(f"{path.name} not present in this checkout")
    out = []
    for x in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
        t = x["client_ttft_ms"]
        out.append({"idx": x["idx"], "cls": x["cls"], "ttft_ms": t, "latency_ms": t + 127 * tpot_ms,
                    "out": x["completion_tokens"]})
    return out


def _val(v, idx):
    return v(idx) if callable(v) else v


def make_requests(N, C, mix, *, short, tpot, long=None, burst=None, head=None, tpot_long=None):
    """Synthetic derived requests: long requests (mixed) at idx = 5 (mod 10), so N / 10 of them as in 007; the first C
    (the start-up wave) at `burst`, idx in [C, 2C) at `head`; short / long / tpot may be functions of idx."""
    reqs = []
    for idx in range(N):
        cls = "long" if mix == "mixed" and idx % 10 == 5 else "short"
        t = _val(long if cls == "long" else short, idx)
        if burst is not None and idx < C:
            t = burst
        elif head is not None and C <= idx < 2 * C:
            t = head
        tp = _val(tpot_long if cls == "long" and tpot_long is not None else tpot, idx)
        reqs.append({"idx": idx, "cls": cls, "ttft_ms": float(t), "latency_ms": float(t) + 127 * tp, "out": 128})
    return reqs


def ev(reqs, L=2048, mix="uniform", C=8, N=None, slo=S2, window="steady"):
    return ca.run_eval(reqs, L=L, mix=mix, C=C, N=N or len(reqs), slo=slo, guard_band=M, window=window)


# ------------------------------------------------------------------ windows and per-run statistics (030 8.1)

def test_windows():
    assert ca.window_bounds("steady", 25, 250) == (25, 225)
    assert ca.window_bounds("late", 25, 250) == (50, 225)       # W' of WINDOW_SENSITIVE
    assert ca.window_bounds("full", 25, 250) == (0, 250)        # C^full (Core-style, descriptive)
    with pytest.raises(ca.AnalysisError):
        ca.window_bounds("p99", 25, 250)


@pytest.mark.parametrize("cond", ["B0", "G", "K3"])
@pytest.mark.parametrize("r", [1, 2, 3])
def test_steady_window_removes_the_start_up_burst_at_2k_c25(cond, r):
    """030 D1: 2K uniform C 25, full-run P95 4,258-4,262 ms, steady 996-997 ms, first-wave max about 8.0 s."""
    reqs = adv_requests(f"{cond}-2048-uniform-r{r}")
    steady, full = ev(reqs, C=25), ev(reqs, C=25, window="full")
    assert 995.5 <= steady["Y"]["short"] <= 996.8
    assert 4258.0 <= full["Y"]["short"] <= 4262.5
    assert 8018.0 <= steady["burst_max_ttft_ms"] <= 8025.0
    assert steady["R"]["short"] == pytest.approx(steady["Y"]["short"] / 2000)
    assert steady["search_pass"] and steady["within_band"] and not steady["fail"]
    assert steady["Y"]["long"] is None and steady["n"]["long"] == 0 and steady["n"]["all"] == 200
    assert ca.derived_problems(reqs, mix="uniform", C=25, N=250) == []


def test_8k_slips_stay_visible_in_w_only_when_they_fall_inside_it():
    """030 D2: B0 r2 is the only run with two slips inside W; G r1's slip is outside W (high full-run, low steady)."""
    b0r1, b0r2, gr1 = (adv_requests(n) for n in ("B0-8192-uniform-r1", "B0-8192-uniform-r2", "G-8192-uniform-r1"))
    assert ev(b0r1, L=8192)["Y"]["short"] == pytest.approx(2480.1, abs=0.1)
    assert ev(b0r2, L=8192)["Y"]["short"] == pytest.approx(2815.3, abs=0.1)
    assert ev(gr1, L=8192, window="full")["Y"]["short"] == pytest.approx(2815.6, abs=0.1)
    assert ev(gr1, L=8192)["Y"]["short"] == pytest.approx(2481.3, abs=0.1)
    assert all(ev(x, L=8192)["search_pass"] for x in (b0r1, b0r2, gr1))   # 6000 ms tier


def test_mixed_short_requests_queue_behind_the_32k_prefill():
    """030 8.2 conditioning: mixed fails S2 already at the knee (2K: 23.3 s steady; 8K: 7.67 s); long class pooled."""
    m2 = ev(adv_requests("B0-2048-mixed-r1"), mix="mixed", C=25)
    assert m2["Y"]["short"] == pytest.approx(23309.1, abs=0.1) and m2["fail"]
    assert m2["n"]["long"] == 21 and m2["Y"]["long"] == pytest.approx(23427.8, abs=0.1)
    assert m2["over"]["long"] is True and m2["R"]["long"] == pytest.approx(m2["Y"]["long"] / 20000)
    m8 = ev(adv_requests("B0-8192-mixed-r1"), L=8192, mix="mixed")
    assert m8["Y"]["short"] == pytest.approx(7670.4, abs=0.1) and m8["fail"] and m8["n"]["long"] == 18


def test_tpot_is_over_every_request_in_w_including_the_long_class():
    """Y_TPOT = P95 over all requests in W of (latency - ttft) / (out - 1): a new statistic (run_adv: short only)."""
    reqs = make_requests(200, 8, "mixed", short=500, long=7000, tpot=50, tpot_long=200)
    e = ev(reqs, mix="mixed")
    assert e["Y"]["tpot"] == pytest.approx(200.0)               # 10 % long > 5 %: the P95 is a long request's
    assert e["fail"] and e["over"]["tpot"] and not e["over"]["short"]
    short_only = [(q["latency_ms"] - q["ttft_ms"]) / (q["out"] - 1) for q in reqs if q["cls"] == "short"]
    assert ba.percentile(short_only, 95) == pytest.approx(50.0)
    reqs[100]["latency_ms"] = reqs[100]["ttft_ms"] + 127 * 1000.0   # one outlier: (lat - ttft) / 127 = 1000 ms
    vals = sorted((q["latency_ms"] - q["ttft_ms"]) / 127 for q in reqs if 8 <= q["idx"] < 192)
    assert ev(reqs, mix="mixed")["Y"]["tpot"] == pytest.approx(ba.percentile(vals, 95))


def test_thresholds_follow_the_tiers():
    assert ca.thresholds(S2, 2048, "uniform") == {"short": 2000, "tpot": 100, "long": None}
    assert ca.thresholds(S2, 8192, "uniform") == {"short": 6000, "tpot": 100, "long": None}
    assert ca.thresholds(S2, 2048, "mixed") == {"short": 2000, "tpot": 100, "long": 20000}
    assert ca.thresholds(S2, 8192, "mixed") == {"short": 6000, "tpot": 100, "long": 20000}
    best = {**S2, "mixed_long": "best_effort"}
    assert ca.thresholds(best, 8192, "mixed") == {"short": 6000, "tpot": 100, "long": None}


def test_the_long_class_is_not_part_of_maxR_run_but_blocks_a_search_pass():
    """030: maxR_run = max(R_short, R_TPOT); FAIL (and the skip rule) read it; a Phase S pass also needs R_long <= 1,
    while C^_g's band reads maxR only (8.4 step 3)."""
    reqs = make_requests(200, 1, "mixed", short=500, long=21000, tpot=20)
    e = ev(reqs, mix="mixed", C=1)
    assert e["max_r"] == pytest.approx(0.25) and not e["fail"]
    assert e["over"]["long"] and not e["search_pass"] and e["within_band"]
    best = ev(reqs, mix="mixed", C=1, slo={**S2, "mixed_long": "best_effort"})
    assert best["search_pass"] and best["Y"]["long"] == pytest.approx(21000.0) and best["R"]["long"] is None


def test_the_guard_band_reads_maxR_run_only(plan):
    """030 8.4 step 3: C^_g = the largest tested c whose every tested c' <= c has maxR <= 1 - m, and 8.1 defines
    maxR_run = max(R_short, R_TPOT). M8K: C 1, 2, 4 pass at maxR 0.25 (C 2's own R_long 0.975: a pass, outside the
    band if the band also read R_long), 8 fails, 6 passes at R_short 0.97 (outside the band), 7 fails: C^ = 6 and
    C^_g = 4, so c_low = 4 (an R_long band would give C^_g = 1 and c_low = 1)."""
    spec = {1: (1500, 6000), 2: (1500, 19500), 4: (1500, 6000), 8: (7000, 6000), 6: (5820, 6000), 7: (7000, 6000)}
    hist = []
    for C in (1, 2, 4, 8, 6, 7):
        short, long_ = spec[C]
        reqs = make_requests(200, C, "mixed", short=short, long=long_, tpot=20)
        bits = ca.search_bits({"workload": "M8K", "C": C, "valid": True, "reasons": []}, reqs, plan)
        hist.append({"C": C, "pass": bits["pass"], "within_band": bits["within_band"]})
    assert [(h["C"], h["pass"], h["within_band"]) for h in hist] == [
        (1, True, True), (2, True, True), (4, True, True), (8, False, False), (6, True, False), (7, False, False)]
    got = cp.build_confirm_plan(plan, {"M8K": hist})["workloads"]["M8K"]
    assert (got["c_hat"], got["c_hat_g"]) == (6, 4)
    assert [(p["C"], p["role"]) for p in got["points"]] == [(4, "low"), (5, "near"), (6, "near"), (7, "near")]


def test_r_boundaries_are_exact_and_inclusive():
    at1 = ev(make_requests(200, 8, "uniform", short=2000, tpot=10))
    assert at1["R"]["short"] == 1.0 and not at1["over"]["short"] and at1["search_pass"] and not at1["within_band"]
    band = ev(make_requests(200, 8, "uniform", short=1900, tpot=95))
    assert not band["over_band"]["short"] and not band["over_band"]["tpot"] and band["within_band"]
    above = ev(make_requests(200, 8, "uniform", short=1900.0000001, tpot=10))
    assert above["over_band"]["short"] and not above["within_band"]
    over = ev(make_requests(200, 8, "uniform", short=2000.000001, tpot=10))
    assert over["fail"] and not over["search_pass"]


def test_attainment_and_burst_are_descriptive():
    reqs = make_requests(200, 8, "uniform", short=lambda i: 2500 if i % 4 == 0 else 1000, tpot=50, burst=9000)
    e = ev(reqs)
    assert e["attainment"] == pytest.approx(138 / 184)          # W = [8, 192): 46 of 184 miss the 2 s tier
    assert e["burst_max_ttft_ms"] == 9000.0


@pytest.mark.parametrize("cond,want", [("B0", (0, 2, 0)), ("G", (1, 0, 0)), ("K3", (2, 1, 2))])
def test_slip_count_is_028s_per_position_rule(cond, want):
    """The C0 branch B / C covariate (030 8.1): 028 found 8 slips in these 9 runs (線索 4)."""
    got = tuple(ca.slip_episodes(adv_requests(f"{cond}-8192-uniform-r{r}"), 8) for r in (1, 2, 3))
    assert got == want


def test_slip_count_uses_each_positions_own_baseline():
    reqs = make_requests(200, 8, "uniform", short=lambda i: 1485 if i % 8 == 0 else (2320 if i % 8 == 7 else 2477),
                         tpot=80)
    assert ca.slip_episodes(reqs, 8) == 0
    for q in reqs[65:72]:            # group k = 8 slips: requests 8k+1 .. 8k+7 wait one more step
        q["ttft_ms"] += 340.0
    assert ca.slip_episodes(reqs, 8) == 1
    for q in reqs[0:7]:              # the start-up wave (idx < C) is never a slip
        q["ttft_ms"] += 5000.0
    assert ca.slip_episodes(reqs, 8) == 1


@pytest.mark.parametrize("bump, want", [(200.0, 1), (171.0, 1), (170.0, 0), (150.0, 0)])
def test_a_slip_is_more_than_170_ms_above_its_positions_baseline(bump, want):
    """028: a request slipped when its TTFT is more than 170 ms above its position's baseline (the threshold itself,
    not only the +340 ms of a real lattice slip)."""
    reqs = make_requests(200, 8, "uniform", short=lambda i: 1485 if i % 8 == 0 else (2320 if i % 8 == 7 else 2477),
                         tpot=80)
    for q in reqs[65:72]:
        q["ttft_ms"] += bump
    assert ca.slip_episodes(reqs, 8) == want


# ------------------------------------------------------------------ run validity (030 8.6 (e))

def good(N=200, C=8, mix="uniform"):
    return make_requests(N, C, mix, short=500, long=7000, tpot=20)


@pytest.mark.parametrize("mutate,mix", [
    (lambda r: r.pop(), "uniform"),                                              # 199 of 200
    (lambda r: r.__setitem__(3, {**r[3], "idx": 4}), "uniform"),                 # duplicate idx
    (lambda r: r.__setitem__(7, {**r[7], "out": 127}), "uniform"),
    (lambda r: r.__setitem__(7, {**r[7], "cls": "long"}), "uniform"),
    (lambda r: r.__setitem__(15, {**r[15], "cls": "short"}), "mixed"),           # 19 long of 200
    (lambda r: r.__setitem__(7, {**r[7], "cls": "medium"}), "mixed"),
    (lambda r: r.__setitem__(7, {**r[7], "ttft_ms": None}), "uniform"),
])
def test_derived_problems(mutate, mix):
    reqs = good(mix=mix)
    assert ca.derived_problems(reqs, mix=mix, C=8, N=200) == []
    mutate(reqs)
    assert ca.derived_problems(reqs, mix=mix, C=8, N=200)


def test_a_mixed_run_with_no_long_request_in_w_is_invalid():
    """030 8.6 (e): mixed needs at least one long request in W."""
    reqs = make_requests(200, 10, "uniform", short=500, tpot=20)
    for q in reqs:   # the 20 long requests all in the first or the last C = 10
        q["cls"] = "long" if q["idx"] < 10 or q["idx"] >= 190 else "short"
    assert any("no long request in W" in p for p in ca.derived_problems(reqs, mix="mixed", C=10, N=200))
    ok, reasons = ca.run_valid({"valid": True}, reqs, mix="mixed", C=10, N=200)
    assert not ok and reasons


def test_run_valid_needs_the_runner_verdict_and_its_derived_file():
    assert ca.run_valid({"valid": True}, good(), mix="uniform", C=8, N=200) == (True, [])
    ok, reasons = ca.run_valid({"valid": False, "reasons": ["timeout"]}, good(), mix="uniform", C=8, N=200)
    assert not ok and any("timeout" in x for x in reasons)
    ok, reasons = ca.run_valid({"valid": True}, None, mix="uniform", C=8, N=200)
    assert not ok and any("derived" in x for x in reasons)
    assert not ca.run_valid({"valid": "true"}, good(), mix="uniform", C=8, N=200)[0]


# ------------------------------------------------------------------ point classes (030 8.1)

def evals_of(*specs, L=2048, mix="uniform", C=8, N=200):
    return [ev(make_requests(N, C, mix, **s), L=L, mix=mix, C=C, N=N) for s in specs]


def cls(evals, n=5, L=2048, mix="uniform"):
    return ca.classify(evals, n=n, L=L, mix=mix, slo=S2, guard_band=M)


def test_fail_takes_precedence_over_nt():
    out = cls(evals_of({"short": 500, "tpot": 20}, {"short": 500, "tpot": 101}))
    assert out["class"] == "FAIL" and out["n_valid"] == 2


def test_nt_without_n_valid_runs():
    assert cls(evals_of(*[{"short": 500, "tpot": 20}] * 4))["class"] == "NT"
    assert cls([])["class"] == "NT"


def test_pass_and_marginal_with_exact_boundaries():
    assert cls(evals_of(*[{"short": 500, "tpot": 20}] * 5))["class"] == "PASS"
    assert cls(evals_of(*[{"short": 1900, "tpot": 95}] * 5))["class"] == "PASS"            # R = 0.95: inside
    marginal = evals_of(*[{"short": 500, "tpot": 20}] * 4, {"short": 500, "tpot": 97})
    assert cls(marginal)["class"] == "MARGINAL"
    assert cls(evals_of(*[{"short": 2000, "tpot": 20}] * 5))["class"] == "MARGINAL"         # R = 1: not FAIL
    assert cls(evals_of(*[{"short": 500, "tpot": 20}] * 7), n=7)["class"] == "PASS"


def test_pooled_long_class_fails_although_every_run_passes_alone():
    """030 8.1: Y_long pools the 32,640-token requests of every valid run at C (about 18-20 per run)."""
    runs = []
    for _ in range(5):   # C 10, W = [10, 190): 18 long requests per run, one at 30 s
        reqs = make_requests(200, 10, "mixed", short=500, long=lambda i: 30000 if i == 15 else 10000, tpot=20)
        runs.append(ev(reqs, mix="mixed", C=10))
    assert all(e["n"]["long"] == 18 and e["R"]["long"] == pytest.approx(0.65) and e["search_pass"] for e in runs)
    out = cls(runs, mix="mixed")
    assert out["pooled"]["Y_long"] == pytest.approx(21000.0) and out["pooled"]["R_long"] == pytest.approx(1.05)
    assert out["class"] == "FAIL"
    two = cls(runs[:2], mix="mixed")
    assert two["class"] == "NT" and two["pooled"]["Y_long"] == pytest.approx(ba.percentile(
        [30000.0] * 2 + [10000.0] * 34, 95))


def test_pooled_long_in_the_band_margin_is_marginal():
    runs = [ev(make_requests(200, 1, "mixed", short=500, long=19500, tpot=20), mix="mixed", C=1) for _ in range(5)]
    assert cls(runs, mix="mixed")["class"] == "MARGINAL"


def test_mixed_without_any_pooled_long_value_cannot_pass():
    runs = [ev(make_requests(200, 10, "mixed", short=500, long=7000, tpot=20), mix="mixed", C=10)] * 5
    for e in runs:
        assert e["n"]["long"] == 18
    empty = [{**e, "_long_ms": [], "n": {**e["n"], "long": 0}} for e in runs]   # e.g. under W'
    assert cls(empty, mix="mixed")["class"] == "NT"


def test_classify_reports_worst_runs_and_met_counts():
    out = cls(evals_of({"short": 500, "tpot": 20}, {"short": 1500, "tpot": 99}, {"short": 900, "tpot": 101},
                       {"short": 500, "tpot": 20}, {"short": 500, "tpot": 20}))
    assert out["class"] == "FAIL" and out["met"] == 4 and out["n_valid"] == 5
    assert out["worst_R"] == {"short": 0.75, "tpot": 1.01, "long": None}
    assert out["conclusions"] == ["fail", "pass"]


def one_run_with_two_slow_long_requests(C=10, r_slow=0, runs=5):
    """Mixed runs at C (W = [C, N - C)): run r_slow has two long requests at 21 s (idx 15 and 25), every other long
    request takes 15 s; short 500 ms, TPOT 20 ms (maxR_run 0.25 everywhere)."""
    out = []
    for r in range(runs):
        slow = {15, 25} if r == r_slow else set()
        out.append(make_requests(200, C, "mixed", short=500, long=lambda i, s=slow: 21000 if i in s else 15000,
                                 tpot=20))
    return out


def test_a_phase_c_runs_conclusion_is_its_maxR_run_never_its_own_long_p95():
    """030 8.1: a Phase C run's own statistics are R_short and R_TPOT (maxR_run); Y_long is pooled over the point's
    valid runs because one run holds only about 18-20 long requests. "Met in k of n" and STRADDLE read maxR_run: a run
    whose own long P95 is 21 s (2 of its 18 long requests) agrees with the others when the pooled R_long passes."""
    runs = [ev(reqs, mix="mixed", C=10) for reqs in one_run_with_two_slow_long_requests()]
    assert runs[0]["R"]["long"] == pytest.approx(1.05) and not runs[0]["search_pass"]   # a Phase S fail, but ...
    assert all(e["max_r"] == pytest.approx(0.25) and not e["fail"] for e in runs)      # ... no run FAILs
    out = cls(runs, mix="mixed")
    assert out["class"] == "PASS" and out["pooled"]["R_long"] == pytest.approx(0.75)
    assert out["met"] == 5 and out["conclusions"] == ["pass"]
    e = ca.estimands({10: out["class"]}, c_cap=70)
    assert "STRADDLE" not in ca.flags({10: out["class"]}, e, conclusions={10: out["conclusions"]}, c_hat=10, c_kv=35,
                                      late=e)


# ------------------------------------------------------------------ estimands and statuses (030 8.1)

def est(classes, c_cap=70):
    return ca.estimands({int(k): v for k, v in classes.items()}, c_cap=c_cap)


def vs(e):
    return (e["value"], e["status"])


def test_exact_and_lower():
    e = est({16: "PASS", 32: "MARGINAL", 33: "MARGINAL", 34: "FAIL"})
    assert vs(e["c_slo"]) == (33, "EXACT")
    assert vs(e["c_prod"]) == (16, "LOWER")                  # 17 untested
    e = est({4: "PASS", 7: "PASS", 8: "MARGINAL", 9: "FAIL"}, c_cap=26)
    assert vs(e["c_slo"]) == (8, "EXACT") and vs(e["c_prod"]) == (7, "EXACT")   # C_PROD: c + 1 MARGINAL


def test_lower_when_the_next_point_is_nt_or_untested():
    assert vs(est({32: "PASS", 33: "PASS", 34: "NT"})["c_slo"]) == (33, "LOWER")
    assert vs(est({24: "PASS", 25: "PASS", 26: "PASS"})["c_slo"]) == (26, "LOWER")   # 27 untested, below the cap


def test_censored_at_the_cap():
    assert vs(est({69: "PASS", 70: "PASS"})["c_slo"]) == (70, "CENSORED")
    assert vs(est({25: "MARGINAL", 26: "MARGINAL"}, c_cap=26)["c_slo"]) == (26, "CENSORED")
    assert vs(est({25: "MARGINAL", 26: "MARGINAL"}, c_cap=26)["c_prod"]) == (None, "UNSTABLE")


def test_infeasible_is_zero():
    e = est({1: "FAIL"})
    assert vs(e["c_slo"]) == (0, "INFEASIBLE") and vs(e["c_prod"]) == (0, "INFEASIBLE")
    assert vs(est({1: "FAIL", 2: "PASS", 3: "PASS"})["c_slo"]) == (0, "INFEASIBLE")


def test_unstable_when_the_lowest_point_above_1_fails():
    e = est({16: "FAIL", 32: "PASS", 33: "PASS", 34: "FAIL"})
    assert vs(e["c_slo"]) == (None, "UNSTABLE") and vs(e["c_prod"]) == (None, "UNSTABLE")


def test_nt_points_stop_the_downward_closure():
    assert vs(est({16: "NT", 32: "PASS", 33: "PASS", 34: "FAIL"})["c_slo"]) == (None, "NT")
    assert vs(est({16: "PASS", 32: "NT", 33: "PASS", 34: "FAIL"})["c_slo"]) == (16, "LOWER")
    assert vs(est({7: "PASS", 8: "NT", 9: "PASS"}, c_cap=26)["c_slo"]) == (7, "LOWER")
    assert vs(est({})["c_slo"]) == (None, "NT")


def test_c_prod_treats_a_marginal_lowest_point_as_failing():
    """030 implementation note (2026-10-16, before any C1 data): C_PROD reads MARGINAL as FAIL in every status row, as
    its EXACT row already does, so a MARGINAL C 1 is a verified 0 (refuse the class), not NT."""
    assert vs(est({1: "MARGINAL", 2: "FAIL"})["c_prod"]) == (0, "INFEASIBLE")
    assert vs(est({1: "MARGINAL"})["c_prod"]) == (0, "INFEASIBLE")
    assert vs(est({16: "MARGINAL", 32: "PASS"})["c_prod"]) == (None, "UNSTABLE")
    assert vs(est({1: "MARGINAL", 2: "FAIL"})["c_slo"]) == (1, "EXACT")
    assert vs(est({1: "NT", 2: "PASS"})["c_prod"]) == (None, "NT")


# ------------------------------------------------------------------ flags (030 8.1)

def flags(classes, *, conclusions=None, c_hat=33, c_kv=35, late=None, c_cap=70):
    e = est(classes, c_cap=c_cap)
    return ca.flags(classes, e, conclusions=conclusions or {}, c_hat=c_hat, c_kv=c_kv,
                    late=late if late is not None else e)


def test_nonmonotone():
    assert "NONMONOTONE" in flags({1: "FAIL", 2: "PASS"}, c_hat=1)
    assert "NONMONOTONE" in flags({32: "PASS", 33: "FAIL", 34: "MARGINAL"})
    assert "NONMONOTONE" not in flags({32: "PASS", 33: "NT", 34: "FAIL"})


def test_straddle():
    assert "STRADDLE" in flags({33: "FAIL"}, conclusions={33: ["fail", "pass"]})
    assert "STRADDLE" not in flags({33: "FAIL"}, conclusions={33: ["fail"]})


def test_strict_coarse_search_mismatch_and_above_ckv():
    f = flags({16: "PASS", 32: "MARGINAL", 33: "MARGINAL", 34: "FAIL"}, c_hat=33)
    assert "STRICT_COARSE" in f and "SEARCH_MISMATCH" not in f
    assert "STRICT_COARSE" not in flags({32: "PASS", 33: "MARGINAL", 34: "FAIL"})   # 32 = 33 - 1
    assert "SEARCH_MISMATCH" in flags({32: "PASS", 33: "MARGINAL", 34: "FAIL"}, c_hat=35)
    assert "SEARCH_MISMATCH" not in flags({32: "PASS", 33: "MARGINAL", 34: "FAIL"}, c_hat=34)
    assert "SEARCH_MISMATCH" not in flags({16: "FAIL", 32: "PASS"}, c_hat=33)       # no C_SLO value
    assert "ABOVE_CKV" in flags({35: "PASS", 36: "PASS", 37: "FAIL"}, c_hat=36)
    assert "ABOVE_CKV" not in flags({34: "PASS", 35: "PASS", 36: "MARGINAL"}, c_hat=35)


def test_window_sensitive_compares_both_estimands():
    steady = {32: "PASS", 33: "PASS", 34: "FAIL"}
    assert "WINDOW_SENSITIVE" not in flags(steady, late=est(steady))
    assert "WINDOW_SENSITIVE" in flags(steady, late=est({32: "PASS", 33: "PASS", 34: "PASS"}))
    assert "WINDOW_SENSITIVE" in flags(steady, late=est({32: "PASS", 33: "MARGINAL", 34: "FAIL"}))  # C_PROD moves


# ------------------------------------------------------------------ hypotheses (030 8.5)

def E(value, status):
    return {"value": value, "status": status}


@pytest.mark.parametrize("mixed,uniform,want", [
    (E(1, "EXACT"), E(33, "EXACT"), "supported"),
    (E(0, "INFEASIBLE"), E(8, "EXACT"), "supported"),
    (E(1, "EXACT"), E(20, "LOWER"), "supported"),
    (E(1, "EXACT"), E(70, "CENSORED"), "supported"),
    (E(0, "INFEASIBLE"), E(3, "LOWER"), "supported"),
    (E(33, "EXACT"), E(33, "EXACT"), "refuted"),
    (E(34, "LOWER"), E(33, "EXACT"), "refuted"),
    (E(70, "CENSORED"), E(33, "EXACT"), "inconclusive"),   # 030 8.5 refutes with a mixed EXACT or LOWER value only
    (E(5, "LOWER"), E(20, "LOWER"), "inconclusive"),
    (E(25, "EXACT"), E(20, "LOWER"), "inconclusive"),
    (E(5, "LOWER"), E(20, "EXACT"), "inconclusive"),
    (E(None, "NT"), E(8, "EXACT"), "NT"),
    (E(1, "EXACT"), E(None, "UNSTABLE"), "NT"),
    (E(None, "UNSTABLE"), E(8, "EXACT"), "NT"),
    (E(0, "INFEASIBLE"), E(0, "INFEASIBLE"), "NT")])
def test_h_s1(mixed, uniform, want):
    assert ca.h_s1(mixed, uniform) == want


@pytest.mark.parametrize("uniform,want", [
    (E(33, "EXACT"), "supported"), (E(35, "EXACT"), "supported"), (E(0, "INFEASIBLE"), "supported"),
    (E(36, "EXACT"), "refuted"), (E(36, "LOWER"), "refuted"), (E(70, "CENSORED"), "refuted"),
    (E(26, "CENSORED"), "refuted"), (E(30, "LOWER"), "inconclusive"), (E(35, "LOWER"), "inconclusive"),
    (E(None, "NT"), "NT"), (E(None, "UNSTABLE"), "NT")])
def test_h_s2(uniform, want):
    assert ca.h_s2(uniform, c_kv=35) == want


# ------------------------------------------------------------------ sentinel gate (030 8.6)

def sentinel(thr, ttft, valid=True, slip=False):
    reqs = make_requests(250, 25, "uniform", short=ttft, tpot=40, burst=8000)
    if slip:   # two lattice slips of 7 requests (+340 ms, 030 D2): 7 % of W, enough to move the P95
        for q in reqs[100:114]:
            q["ttft_ms"] += 340.0
    row = {"phase": "sentinel", "valid": valid, "output_throughput": thr, "C": 25, "N": 250}
    return ca.sentinel_summary(row, reqs if valid else None, C=25, N=250)


def test_sentinel_summary_uses_the_w_mean_ttft():
    s = sentinel(315.0, 996.0)
    assert s["valid"] and s["output_throughput"] == 315.0 and s["w_mean_ttft_ms"] == pytest.approx(996.0)
    assert s["steady_p95_ttft_ms"] == pytest.approx(996.0)


def test_sentinel_gate_within_three_percent_inclusive():
    g = ca.sentinel_gate(sentinel(300.0, 1000.0), sentinel(309.0, 970.0), tolerance=0.03)
    assert g["ok"] and g["drift"]["output_throughput"] == pytest.approx(0.03)
    g = ca.sentinel_gate(sentinel(300.0, 1000.0), sentinel(309.3, 1000.0), tolerance=0.03)
    assert not g["ok"] and g["reasons"]
    g = ca.sentinel_gate(sentinel(300.0, 1000.0), sentinel(300.0, 1040.0), tolerance=0.03)
    assert not g["ok"]


def test_sentinel_gate_ignores_a_lattice_slip_that_flips_the_p95():
    start, end = sentinel(315.0, 996.0), sentinel(315.0, 996.0, slip=True)
    assert end["steady_p95_ttft_ms"] > 1.3 * start["steady_p95_ttft_ms"]
    assert ca.sentinel_gate(start, end, tolerance=0.03)["ok"]


def test_sentinel_gate_fails_on_a_missing_or_invalid_sentinel():
    assert not ca.sentinel_gate(sentinel(315.0, 996.0), None, tolerance=0.03)["ok"]
    assert not ca.sentinel_gate(sentinel(315.0, 996.0), sentinel(315.0, 996.0, valid=False), tolerance=0.03)["ok"]


@pytest.mark.parametrize("thr", [None, 0.0, "315"])
def test_a_sentinel_without_an_output_throughput_is_invalid_not_a_crash(thr):
    """The gate compares output throughput (8.6); a sentinel row without a positive number cannot enter it."""
    s = sentinel(thr, 996.0)
    assert s["valid"] is False and any("throughput" in x for x in s["reasons"])
    g = ca.sentinel_gate(sentinel(315.0, 996.0), s, tolerance=0.03)
    assert g["ok"] is False and g["drift"] is None


def test_cross_session_shift_is_descriptive():
    refs = {"core_output_throughput": 315.0, "advanced_steady_p95_ttft_ms": 996.0}
    ok = ca.cross_session_shift([sentinel(316.0, 996.0), sentinel(314.0, 996.0)], refs)
    assert ok["shift"] is False
    far = ca.cross_session_shift([sentinel(316.0, 996.0), sentinel(290.0, 996.0)], refs)
    assert far["shift"] is True
    assert ca.cross_session_shift([sentinel(316.0, 996.0)], None)["shift"] is None


def test_references_recomputed_from_committed_core_and_advanced_data():
    rows = [json.loads(x) for x in (CORE_UNIFORM / "rows.jsonl").read_text().splitlines() if x.strip()]
    adv = [adv_requests(f"B0-2048-uniform-r{r}") for r in (1, 2, 3)]
    refs = ca.sentinel_references(rows, adv)
    assert refs["core_output_throughput"] == pytest.approx(314.94, abs=0.01)      # "Core 315.0 tok/s"
    assert refs["advanced_steady_p95_ttft_ms"] == pytest.approx(996.0, abs=0.1)   # "Advanced steady 996 ms"


# ------------------------------------------------------------------ a simulated session (what run_capacity does)

def scenario(w, C, phase, block):
    """S2 truth: U2K TPOT 3C ms (C^ 33), U8K TPOT 12C ms (C^ 8), M2K short TTFT 6 s from C 2 on (C^ 1), M8K long
    TTFT 21 s at every C (infeasible)."""
    if w == "U2K":
        return {"short": 900, "tpot": 3 * C, "burst": 8000}
    if w == "U8K":
        return {"short": 2500, "tpot": 12 * C, "burst": 11000}
    if w == "M2K":
        return {"short": 600 if C == 1 else 6000, "long": 6300, "tpot": 20}
    return {"short": 1500, "long": 21000, "tpot": 30}


@pytest.fixture(scope="module")
def cfg():
    return tomllib.loads((ROOT / "configs" / "capacity.toml").read_text())


@pytest.fixture(scope="module")
def pins():
    return tomllib.loads((ROOT / "configs" / "pins.toml").read_text())


@pytest.fixture(scope="module")
def plan(cfg, pins):
    session, up = cp.load_uniform_session(CORE_UNIFORM)
    return cp.build_plan(cfg, pins, session=session, uniform_plan=up, pins_sha256="ab" * 32,
                         capacity_sha256="cd" * 32)


def simulate(plan, truth=scenario, *, boot_fails=(), invalid=(), sentinel_thr=(315.0, 314.0)):
    """Drive the plan as run_capacity would. boot_fails: {(phase, w, block_or_attempt)} boots that fail before their
    first bench; invalid: {(phase, w, C, block)} runs the runner marks invalid."""
    boots, rows, requests = [], [], {}

    def boot(phase, w, *, block=None, attempt=None, position=None, reached=True):
        boots.append({"boot": len(boots), "phase": phase, "workload": w, "block": block, "attempt": attempt,
                      "position": position, "reached_first_bench": reached})
        return len(boots) - 1

    def run(b, phase, w, C, *, block=None, position=None, valid=True, thr=1000.0):
        L, mix = cp.WORKLOAD_SPEC[w]
        N = cp.bp.num_prompts(C)
        tag = cp.run_tag(phase, w, C, block=block, position=position)
        t = truth(w, C, phase, block)
        requests[tag] = make_requests(N, C, mix, **t)
        timeout = plan["sentinel"]["timeout_s"] if phase == "sentinel" else plan["workloads"][w]["timeouts_s"][str(C)]
        row = {"phase": phase, "workload": w, "L": L, "mix": mix, "C": C, "N": N, "block": block,
               "position": position, "boot": b, "tag": tag, "valid": valid,
               "reasons": [] if valid else ["server unhealthy after the bench"], "timeout_s": timeout,
               "output_throughput": thr, "metrics": {"kv_cache_usage_peak": 0.5, "preemptions": 0.0,
                                                     "running_peak": C, "waiting_peak": 0, "n_samples": 10,
                                                     "inflight_peak": C}}
        bits = ca.search_bits(row, requests[tag], plan)
        if phase == "S":
            row["search_pass"] = bits["pass"]
        if phase == "C":
            row["outcome"] = ca.phase_c_outcome(row, requests[tag], plan)
        rows.append(row)
        return row, bits

    s = plan["sentinel"]
    run(boot("sentinel", s["workload"], position="start"), "sentinel", s["workload"], s["C"], position="start",
        thr=sentinel_thr[0])
    searches, retry = {}, []

    def search(w, attempt):
        if ("S", w, attempt) in boot_fails:
            boot("S", w, attempt=attempt, reached=False)
            return False
        b, hist = boot("S", w, attempt=attempt), []
        c_cap = plan["workloads"][w]["c_cap"]
        while (C := cp.search_next(c_cap, [(h["C"], h["pass"]) for h in hist])) is not None:
            row, bits = run(b, "S", w, C, valid=("S", w, C, None) not in invalid)
            hist.append({"C": C, "pass": row["search_pass"], "within_band": bits["within_band"]})
        searches[w] = hist
        return True

    for w in plan["workloads"]:
        if not search(w, 1):
            retry.append(w)
    for w in retry:
        if not search(w, 2):
            searches[w] = None
    confirm = cp.build_confirm_plan(plan, searches)
    outcomes = {w: [] for w in plan["workloads"]}
    for r in range(1, confirm["blocks"] + 1):
        new = []
        for w, points in cp.phase_c_boots(confirm, r, outcomes):
            failed = ("C", w, r) in boot_fails
            b = boot("C", w, block=r, reached=not failed)
            for C in points:
                row, _ = run(b, "C", w, C, block=r, valid=not failed and ("C", w, C, r) not in invalid)
                new.append((w, (r, C, row["outcome"])))
        for w, o in new:
            outcomes[w].append(o)
    run(boot("sentinel", s["workload"], position="end"), "sentinel", s["workload"], s["C"], position="end",
        thr=sentinel_thr[1])
    session = {"session_id": "formal-test", "boots": boots, "confirm_plan_sha256": cp.confirm_plan_sha256(confirm)}
    return {"session": session, "plan": plan, "confirm_plan": confirm, "rows": rows}, requests


@pytest.fixture(scope="module")
def sim(plan):
    return simulate(plan)


@pytest.fixture(scope="module")
def analysis(sim):
    cap, requests = sim
    return ca.analyze(cap, requests)


def replay(cap, requests):
    return ca.replay_problems(cap["plan"], cap["confirm_plan"], cap["session"]["boots"], cap["rows"], requests,
                              confirm_sha256=cap["session"]["confirm_plan_sha256"])


def test_runner_decisions_are_the_functions_replay_uses(plan):
    """run_capacity records search_pass / within_band (Phase S) and the skip outcome (Phase C) with these."""
    def row(w, C, valid=True):
        return {"workload": w, "C": C, "valid": valid, "reasons": [] if valid else ["timeout"]}

    ok = make_requests(200, 8, "uniform", short=900, tpot=90)
    assert ca.search_bits(row("U2K", 8), ok, plan) == {"pass": True, "within_band": True}
    assert ca.phase_c_outcome(row("U2K", 8), ok, plan) == "ok"
    margin = make_requests(200, 8, "uniform", short=900, tpot=99)
    assert ca.search_bits(row("U2K", 8), margin, plan) == {"pass": True, "within_band": False}
    slow = make_requests(200, 8, "uniform", short=900, tpot=101)
    assert ca.search_bits(row("U2K", 8), slow, plan) == {"pass": False, "within_band": False}
    assert ca.phase_c_outcome(row("U2K", 8), slow, plan) == "fail"
    assert ca.search_bits(row("U2K", 8, valid=False), ok, plan) == {"pass": False, "within_band": False}
    assert ca.phase_c_outcome(row("U2K", 8, valid=False), ok, plan) == "invalid"
    assert ca.phase_c_outcome(row("U2K", 8), ok[:-1], plan) == "invalid"            # derived data incomplete
    long_slow = make_requests(200, 1, "mixed", short=500, long=21000, tpot=20)
    assert ca.search_bits(row("M8K", 1), long_slow, plan) == {"pass": False, "within_band": True}   # band: maxR only
    assert ca.phase_c_outcome(row("M8K", 1), long_slow, plan) == "ok"              # the long class is pooled later
    long_near = make_requests(200, 1, "mixed", short=500, long=19500, tpot=20)    # own R_long 0.975: pass, band
    assert ca.search_bits(row("M8K", 1), long_near, plan) == {"pass": True, "within_band": True}


def test_the_simulated_session_replays_cleanly(sim):
    cap, requests = sim
    assert replay(cap, requests) == []
    phases = [b["phase"] for b in cap["session"]["boots"]]
    assert phases[0] == phases[-1] == "sentinel" and phases[1:5] == ["S"] * 4


def test_end_to_end_estimands_flags_and_hypotheses(analysis):
    a = analysis
    assert a["replay"] == {"ok": True, "problems": []}
    assert a["sentinel"]["gate"]["ok"] is True
    wl = a["workloads"]
    u2 = wl["U2K"]
    assert (u2["search"]["c_hat"], u2["search"]["c_hat_g"], u2["K"]) == (33, 16, [16, 32, 33, 34])
    assert {C: p["class"] for C, p in u2["points"].items()} == {"16": "PASS", "32": "MARGINAL", "33": "MARGINAL",
                                                                "34": "FAIL"}
    assert vs(u2["c_slo"]) == (33, "EXACT") and vs(u2["c_prod"]) == (16, "LOWER")
    assert u2["flags"] == ["STRICT_COARSE"] and u2["binding"] == ["tpot"]
    assert u2["margins"]["c_slo"]["tpot"] == pytest.approx(0.01)
    assert u2["points"]["16"]["n"] == 3 and u2["points"]["16"]["n_valid"] == 3 and u2["points"]["33"]["n_valid"] == 5
    u8 = wl["U8K"]
    assert u8["K"] == [4, 7, 8, 9] and vs(u8["c_slo"]) == (8, "EXACT") and vs(u8["c_prod"]) == (7, "EXACT")
    assert u8["flags"] == []
    m2 = wl["M2K"]
    assert m2["K"] == [1, 2] and vs(m2["c_slo"]) == (1, "EXACT") and m2["binding"] == ["short"]
    m8 = wl["M8K"]
    assert m8["K"] == [1] and vs(m8["c_slo"]) == (0, "INFEASIBLE") and vs(m8["c_prod"]) == (0, "INFEASIBLE")
    assert m8["binding"] == ["long"] and m8["points"]["1"]["n"] == 5
    h = a["hypotheses"]
    assert h["H-S1"]["2048"]["verdict"] == h["H-S1"]["8192"]["verdict"] == "supported"
    assert h["H-S2"]["U2K"]["verdict"] == h["H-S2"]["U8K"]["verdict"] == "supported"
    assert h["H-S1"]["2048"]["prediction"] == h["H-S2"]["U8K"]["prediction"] == "supported"
    assert h["H-S2"]["U2K"]["c_kv"] == 35 and h["H-S2"]["U8K"]["c_kv"] == 13


def test_full_run_statistics_are_reported_beside_the_steady_window(analysis):
    """C^full keeps the start-up burst (8 s first wave): at U2K every confirmation point fails Core-style."""
    u2 = analysis["workloads"]["U2K"]
    assert u2["points"]["16"]["class_full"] == "FAIL"
    assert vs(u2["descriptive"]["c_full"]["c_slo"]) == (None, "UNSTABLE")
    run = u2["points"]["33"]["runs"][0]
    assert run["full"]["Y_short"] == pytest.approx(8000.0) and run["Y"]["short"] == pytest.approx(900.0)
    assert run["burst_max_ttft_ms"] == 8000.0 and run["kv_cache_usage_peak"] == 0.5 and run["preemptions"] == 0.0


def test_secondary_slos_are_descriptive_bounds_on_the_same_runs(analysis):
    s1 = analysis["secondary_slo"]["S1"]
    assert s1["slo"] == cp.SLO_S1 and s1["descriptive"] is True
    assert vs(s1["workloads"]["U2K"]["c_slo"]) == (16, "LOWER")          # TPOT 48 ms of 50
    assert vs(s1["workloads"]["U2K"]["c_prod"]) == (None, "UNSTABLE")   # lowest point MARGINAL (030 impl. note)
    assert "prediction" not in s1["hypotheses"]["H-S1"]["2048"]
    s3 = analysis["secondary_slo"]["S3"]
    # TPOT 102 ms of 200 passes S3 at 34, but S2's skip rule left 34 with one run: NT, so 33 is a lower bound
    assert vs(s3["workloads"]["U2K"]["c_slo"]) == (33, "LOWER")


def test_analysis_is_json_and_lists_every_run(analysis, sim):
    cap, _ = sim
    assert json.loads(json.dumps(analysis)) == analysis
    assert len(analysis["runs"]) == len(cap["rows"])
    assert analysis["schema_version"] == 1 and analysis["stage"] == "C1" and analysis["slo"] == S2
    assert analysis["protocol"]["n_near"] == dict.fromkeys(cp.WORKLOADS, 5)
    assert analysis["not_supported"]


def test_c0_covariate_is_recorded_only_for_the_branch_workloads(cfg, pins):
    session, up = cp.load_uniform_session(CORE_UNIFORM)
    c = copy.deepcopy(cfg)
    c["capacity"]["c0_branch"] = "C"
    plan_c = cp.build_plan(c, pins, session=session, uniform_plan=up, pins_sha256="ab" * 32, capacity_sha256="cd" * 32)
    cap, requests = simulate(plan_c)
    assert replay(cap, requests) == []
    a = ca.analyze(cap, requests)
    u8 = a["workloads"]["U8K"]
    assert u8["n_near"] == 7 and u8["points"]["8"]["n"] == 7 and u8["points"]["8"]["n_valid"] == 7
    assert all(run["slip_episodes"] == 0 for p in u8["points"].values() for run in p["runs"] if run["valid"])
    assert all("slip_episodes" not in run for p in a["workloads"]["U2K"]["points"].values() for run in p["runs"])
    assert a["protocol"]["c0_branch"] == "C" and a["protocol"]["n_near"]["U8K"] == 7


def test_window_sensitive_end_to_end(plan):
    """A head of slow requests in [C, 2C) puts c_low in the margin under W but not under W' = [2C, N - C)."""
    def truth(w, C, phase, block):
        t = scenario(w, C, phase, block)
        return {**t, "head": 1980} if (w == "U2K" and phase == "C" and C == 16) else t

    cap, requests = simulate(plan, truth)
    a = ca.analyze(cap, requests)
    u2 = a["workloads"]["U2K"]
    assert u2["points"]["16"]["class"] == "MARGINAL" and u2["points"]["16"]["class_late"] == "PASS"
    assert vs(u2["c_slo"]) == (33, "EXACT") and vs(u2["c_prod"]) == (None, "UNSTABLE")
    assert vs(u2["descriptive"]["c_late"]["c_prod"]) == (16, "LOWER")
    assert "WINDOW_SENSITIVE" in u2["flags"]


def test_skip_rule_and_invalid_runs_end_to_end(plan):
    """An invalid Phase C run (never replaced, 013) drops its point and every higher one from later blocks."""
    cap, requests = simulate(plan, invalid={("C", "U8K", 8, 1)})
    assert replay(cap, requests) == []
    u8 = [r for r in cap["rows"] if r["phase"] == "C" and r["workload"] == "U8K"]
    assert sorted({r["C"] for r in u8 if r["block"] >= 2}) == [4, 7]
    a = ca.analyze(cap, requests)
    p = a["workloads"]["U8K"]["points"]
    assert p["8"]["class"] == "NT" and p["8"]["n_valid"] == 0 and p["9"]["class"] == "FAIL"   # 9 failed in block 1
    assert vs(a["workloads"]["U8K"]["c_slo"]) == (7, "LOWER")


def two_slow_long_requests_in_m2k_c1_block_1(w, C, phase, block):
    """scenario, except that M2K's C 1 run of block 1 has two of its 20 long requests in W at 21 s: its own long P95 is
    21 s (R_long 1.05), while the long class pooled over the point's 5 runs stays at 6.3 s."""
    t = scenario(w, C, phase, block)
    if (w, C, phase, block) == ("M2K", 1, "C", 1):
        return {**t, "long": lambda i: 21000 if i in (15, 25) else 6300}
    return t


def test_run_summaries_and_straddle_read_maxR_run_end_to_end(plan):
    cap, requests = simulate(plan, two_slow_long_requests_in_m2k_c1_block_1)
    assert replay(cap, requests) == []
    m2 = ca.analyze(cap, requests)["workloads"]["M2K"]
    p = m2["points"]["1"]
    assert p["runs"][0]["R"]["long"] == pytest.approx(1.05)
    assert p["class"] == "PASS" and p["pooled"]["R_long"] == pytest.approx(0.315)
    assert p["met"] == 5 and p["conclusions"] == ["pass"] and [r["conclusion"] for r in p["runs"]] == ["pass"] * 5
    assert "STRADDLE" not in m2["flags"] and vs(m2["c_prod"]) == (1, "EXACT")


def test_the_pooled_long_class_never_triggers_the_skip_rule(plan):
    """030 8.4 skip rule: "a valid FAIL" (beside "an invalid run") is a valid run with maxR_run > 1, 8.1's per-run
    FAIL. The long class is judged once, pooled over the point's runs (one run has only about 18-20 long requests), so
    a point whose long class pooled over block 1 alone is above 1 keeps running and is classified on all its runs."""
    cap, requests = simulate(plan, two_slow_long_requests_in_m2k_c1_block_1)
    first = _rows(cap, phase="C", workload="M2K", C=1, block=1)[0]
    e1 = ca.evaluate(first, requests[first["tag"]], plan)[2]
    assert first["outcome"] == "ok"
    assert cls([e1], mix="mixed")["class"] == "FAIL"                 # the point's class after block 1 alone
    assert [r["block"] for r in _rows(cap, phase="C", workload="M2K", C=1)] == [1, 2, 3, 4, 5]
    assert ca.analyze(cap, requests)["workloads"]["M2K"]["points"]["1"]["class"] == "PASS"


def test_descriptive_inflight_check_and_knee_comparison(plan):
    """030 8.6 (descriptive): the 1 Hz MetricsSampler's running + waiting <= C, per run and per point; 8.5
    (descriptive): C_SLO beside C_knee (25 / 8)."""
    cap, requests = simulate(plan)
    _rows(cap, phase="C", workload="U2K", C=33, block=1)[0]["metrics"]["inflight_peak"] = 40
    _rows(cap, phase="C", workload="U8K", C=8, block=2)[0]["metrics"].pop("inflight_peak")
    a = ca.analyze(cap, requests)
    u2 = a["workloads"]["U2K"]
    runs = u2["points"]["33"]["runs"]
    assert [(r["block"], r["inflight_peak"], r["inflight_le_c"]) for r in runs] == [
        (1, 40, False), (2, 33, True), (3, 33, True), (4, 33, True), (5, 33, True)]
    assert u2["points"]["33"]["inflight_le_c"] is False and u2["points"]["32"]["inflight_le_c"] is True
    u8_runs = a["workloads"]["U8K"]["points"]["8"]["runs"]
    assert [r["inflight_le_c"] for r in u8_runs] == [True, None, True, True, True]
    assert a["workloads"]["U8K"]["points"]["8"]["inflight_le_c"] is True       # a run without samples is no evidence
    assert a["sentinel"]["start"]["inflight_peak"] == 25 and a["sentinel"]["end"]["inflight_le_c"] is True
    assert all(r["inflight_le_c"] is True for r in u2["search"]["runs"])
    assert u2["descriptive"]["knee"] == {"c_knee": 25, "c_slo": {"value": 33, "status": "EXACT"},
                                         "c_slo_minus_knee": 8}
    assert a["workloads"]["U8K"]["descriptive"]["knee"]["c_knee"] == 8
    assert a["workloads"]["U8K"]["descriptive"]["knee"]["c_slo_minus_knee"] == 0
    assert a["workloads"]["M8K"]["descriptive"]["knee"] == {"c_knee": 8, "c_slo": {"value": 0, "status": "INFEASIBLE"},
                                                            "c_slo_minus_knee": -8}


def test_a_failed_phase_c_boot_is_invalid_runs(plan):
    cap, requests = simulate(plan, boot_fails={("C", "M2K", 1)})
    assert replay(cap, requests) == []
    m2 = [r for r in cap["rows"] if r["workload"] == "M2K" and r["phase"] == "C"]
    assert all(not r["valid"] for r in m2) and {r["block"] for r in m2} == {1}
    assert vs(ca.analyze(cap, requests)["workloads"]["M2K"]["c_slo"]) == (None, "NT")


def test_phase_s_boot_retry_at_the_end_of_phase_s(plan):
    cap, requests = simulate(plan, boot_fails={("S", "U8K", 1)})
    assert replay(cap, requests) == []
    s = [(b["phase"], b["workload"], b["attempt"], b["reached_first_bench"]) for b in cap["session"]["boots"][1:7]]
    assert s == [("S", "U2K", 1, True), ("S", "U8K", 1, False), ("S", "M2K", 1, True), ("S", "M8K", 1, True),
                 ("S", "U8K", 2, True), ("C", cap["confirm_plan"]["block_orders"]["1"][0], None, True)]
    assert cap["confirm_plan"]["workloads"]["U8K"]["c_hat"] == 8


def test_a_search_whose_boot_fails_twice_is_not_testable(plan):
    cap, requests = simulate(plan, boot_fails={("S", "U8K", 1), ("S", "U8K", 2)})
    assert replay(cap, requests) == []
    a = ca.analyze(cap, requests)
    assert a["workloads"]["U8K"]["K"] == [] and vs(a["workloads"]["U8K"]["c_slo"]) == (None, "NT")
    assert a["hypotheses"]["H-S1"]["8192"]["verdict"] == "NT" and a["hypotheses"]["H-S2"]["U8K"]["verdict"] == "NT"


def test_an_invalid_search_run_counts_as_a_fail_for_navigation(plan):
    cap, requests = simulate(plan, invalid={("S", "U2K", 16, None)})
    assert replay(cap, requests) == []
    assert cap["confirm_plan"]["workloads"]["U2K"]["c_hat"] == 15


# ------------------------------------------------------------------ replay mismatches (030 8.6)

def mutated(sim, fn):
    cap, requests = copy.deepcopy(sim)
    fn(cap, requests)
    return replay(cap, requests)


def _rows(cap, **kw):
    return [r for r in cap["rows"] if all(r.get(k) == v for k, v in kw.items())]


def swap_two_phase_c_points(cap, requests):
    rows = cap["rows"]
    i = next(i for i, r in enumerate(rows) if r["phase"] == "C" and r["workload"] == "U2K" and r["block"] == 1)
    rows[i], rows[i + 1] = rows[i + 1], rows[i]


def change_a_search_point(cap, requests):
    """C 64 -> 63 with a consistent tag, N, timeout and derived file: only the search rule can object."""
    r = _rows(cap, phase="S", workload="U2K", C=64)[0]
    old = r["tag"]
    r.update(C=63, N=630, tag=cp.run_tag("S", "U2K", 63), timeout_s=cap["plan"]["workloads"]["U2K"]["timeouts_s"]["63"])
    requests[r["tag"]] = make_requests(630, 63, "uniform", short=900, tpot=3 * 63)
    del requests[old]


def flip_a_search_pass(cap, requests):
    r = _rows(cap, phase="S", workload="U8K")[3]
    r["search_pass"] = not r["search_pass"]


def slow_down_a_search_run(cap, requests):
    r = _rows(cap, phase="S", workload="U2K", C=32)[0]
    for q in requests[r["tag"]]:
        q["latency_ms"] = q["ttft_ms"] + 127 * 150.0   # its recomputed pass becomes a fail


def drop_a_phase_c_boot(cap, requests):
    """A whole block-2 boot missing, boots renumbered and rows following them: only the order can object."""
    gone = next(b["boot"] for b in cap["session"]["boots"] if b["phase"] == "C" and b["block"] == 2)
    cap["session"]["boots"] = [x for x in cap["session"]["boots"] if x["boot"] != gone]
    cap["rows"] = [r for r in cap["rows"] if r["boot"] != gone]
    for x in cap["session"]["boots"]:
        x["boot"] -= x["boot"] > gone
    for r in cap["rows"]:
        r["boot"] -= r["boot"] > gone


def tamper_the_confirm_plan(cap, requests):
    cap["confirm_plan"]["workloads"]["U2K"]["points"][0]["n"] = 5


def tamper_the_confirm_sha(cap, requests):
    cap["session"]["confirm_plan_sha256"] = "00" * 32


def flip_a_phase_c_outcome(cap, requests):
    r = _rows(cap, phase="C", workload="U2K", C=34)[0]
    r["outcome"] = "ok"


def run_a_skipped_point(cap, requests):
    r = copy.deepcopy(_rows(cap, phase="C", workload="M2K", C=2, block=1)[0])
    r["block"] = 2
    r["tag"] = cp.run_tag("C", "M2K", 2, block=2)
    requests[r["tag"]] = requests[_rows(cap, phase="C", workload="M2K", C=2, block=1)[0]["tag"]]
    last = _rows(cap, phase="C", workload="M2K", block=2)[-1]
    r["boot"] = last["boot"]
    cap["rows"].insert(cap["rows"].index(last) + 1, r)


def swap_two_phase_c_boots(cap, requests):
    boots = cap["session"]["boots"]
    i = next(i for i, b in enumerate(boots) if b["phase"] == "C" and b["block"] == 1)
    a, b = boots[i], boots[i + 1]
    ida, idb = a["boot"], b["boot"]
    boots[i], boots[i + 1] = b, a
    b["boot"], a["boot"] = ida, idb        # consistent numbering, rows follow their boot: only the order is wrong
    for r in cap["rows"]:
        r["boot"] = {ida: idb, idb: ida}.get(r["boot"], r["boot"])


def wrong_n(cap, requests):
    _rows(cap, phase="C", workload="U8K")[0]["N"] = 250


def wrong_timeout(cap, requests):
    _rows(cap, phase="S", workload="M8K")[0]["timeout_s"] = 600


def retry_out_of_place(cap, requests):
    s = [b for b in cap["session"]["boots"] if b["phase"] == "S"]
    s[0]["attempt"] = 2


def missing_derived(cap, requests):
    del requests[_rows(cap, phase="S", workload="U2K")[0]["tag"]]


def rows_in_a_failed_search_boot(cap, requests):
    next(b for b in cap["session"]["boots"] if b["phase"] == "S")["reached_first_bench"] = False


def missing_end_sentinel(cap, requests):
    end = cap["session"]["boots"].pop()
    cap["rows"] = [r for r in cap["rows"] if r["boot"] != end["boot"]]


def end_sentinel_at_another_workload(cap, requests):
    """A U8K C 25 run in the end sentinel boot, its tag, L and derived file consistent with U8K (N and the sentinel
    timeout are the same): 030 8.4 runs U2K there, so only the sentinel's own workload can object."""
    r = _rows(cap, phase="sentinel", position="end")[0]
    old = r["tag"]
    r.update(workload="U8K", L=8192, tag=cp.run_tag("sentinel", "U8K", 25, position="end"))
    requests[r["tag"]] = requests.pop(old)


def append_a_boot_after_the_end_sentinel(cap, requests):
    """A Phase C boot after the end sentinel (e.g. a re-run of a dropped point, which 013 forbids): consistent rows,
    tag and derived file; without the check its run would enter the estimands (workload_analysis reads every C row)."""
    b = len(cap["session"]["boots"])
    cap["session"]["boots"].append({"boot": b, "phase": "C", "workload": "U2K", "block": 6, "attempt": None,
                                    "position": None, "reached_first_bench": True})
    C, N = 34, cp.bp.num_prompts(34)
    tag = cp.run_tag("C", "U2K", C, block=6)
    requests[tag] = make_requests(N, C, "uniform", short=900, tpot=90, burst=8000)
    cap["rows"].append({"phase": "C", "workload": "U2K", "L": 2048, "mix": "uniform", "C": C, "N": N, "block": 6,
                        "position": None, "boot": b, "tag": tag, "valid": True, "reasons": [],
                        "timeout_s": cap["plan"]["workloads"]["U2K"]["timeouts_s"][str(C)], "outcome": "ok",
                        "output_throughput": 1.0, "metrics": {}})


@pytest.mark.parametrize("fn", [swap_two_phase_c_points, change_a_search_point, flip_a_search_pass,
                                slow_down_a_search_run, drop_a_phase_c_boot, tamper_the_confirm_plan,
                                tamper_the_confirm_sha, flip_a_phase_c_outcome, run_a_skipped_point,
                                swap_two_phase_c_boots, wrong_n, wrong_timeout, retry_out_of_place, missing_derived,
                                rows_in_a_failed_search_boot, missing_end_sentinel, end_sentinel_at_another_workload,
                                append_a_boot_after_the_end_sentinel])
def test_replay_catches_every_deviation(sim, fn):
    assert mutated(sim, fn), fn.__name__


def test_a_boot_after_the_end_sentinel_is_a_replay_mismatch_not_evidence(sim):
    cap, requests = copy.deepcopy(sim)
    append_a_boot_after_the_end_sentinel(cap, requests)
    assert any("after the end sentinel" in p for p in replay(cap, requests))
    a = ca.analyze(cap, requests)
    assert a["replay"]["ok"] is False and a["workloads"]["U2K"]["points"]["34"]["n_valid"] == 2


def test_a_replay_mismatch_is_reported_by_analyze(sim):
    cap, requests = copy.deepcopy(sim)
    flip_a_search_pass(cap, requests)
    a = ca.analyze(cap, requests)
    assert a["replay"]["ok"] is False and a["replay"]["problems"]
