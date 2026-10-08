"""Tests for benchmarks/core_b_analysis.py: PREREGISTRATION §2 Core B rules (H-B1..H-B4),
written before any NVIDIA data. Unit = server restart (n = 3)."""

import pytest

import core_b_analysis as ba


def runs_for(L, grid, *, thr, ttft, tpot, restarts=3, mix="uniform"):
    """thr / ttft / tpot: f(C) -> value; same for every restart (restart index adds tiny noise)."""
    out = []
    for r in range(1, restarts + 1):
        for C in grid:
            eps = 1 + 0.001 * (r - 2)
            out.append({"L": L, "mix": mix, "restart": r, "C": C, "ok": True, "output_throughput": thr(C) * eps,
                        "p95_ttft_ms": ttft(C) * eps, "p95_tpot_ms": tpot(C) * eps})
    return out


GRID = [1, 2, 4, 8, 11, 16, 22, 31, 32, 44, 64, 128, 256]


def saturating(C, knee=22):
    return 100.0 * min(C, knee)


def test_knee_is_smallest_c_reaching_95pct_of_the_max():
    rows = runs_for(2048, GRID, thr=saturating, ttft=lambda C: 100, tpot=lambda C: 10)
    k = ba.knee(rows, 2048, GRID)
    assert k["c_knee"] == 22 and k["c_max"] == 256
    rows = runs_for(2048, GRID, thr=lambda C: 100.0 * min(C, 21.5), ttft=lambda C: 100, tpot=lambda C: 10)
    assert ba.knee(rows, 2048, GRID)["c_knee"] == 22  # 0.95 x 2150 = 2042.5 -> 22 (2150) not 16 (1600)


def queueing_ttft(C, knee=22):
    return 100.0 if C <= knee else 100.0 * C / knee


def test_h_b1_supported_refuted_inconclusive():
    rows = runs_for(2048, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10)
    assert ba.h_b1(rows, {2048: GRID})["units"][0]["status"] == "supported"
    flat = runs_for(2048, GRID, thr=saturating, ttft=lambda C: 100, tpot=lambda C: 10)
    assert ba.h_b1(flat, {2048: GRID})["verdict"] == "refuted"  # grid reaches 256 >= 4 x 22, no 3x
    short = [g for g in GRID if g <= 64]
    flat = runs_for(2048, short, thr=saturating, ttft=lambda C: 100, tpot=lambda C: 10)
    assert ba.h_b1(flat, {2048: short})["units"][0]["status"] == "inconclusive"  # 64 < 88


def test_h_b2_interval_and_out_of_grid():
    rows = runs_for(2048, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10)
    res = ba.h_b2(rows, {2048: 22}, {2048: GRID})
    assert res["verdict"] == "not refuted" and res["units"][0]["interval"] == [11, 44]
    res = ba.h_b2(rows, {2048: 8}, {2048: GRID})  # knee 22 > 16
    assert res["verdict"] == "refuted"
    res = ba.h_b2(rows, {2048: 400}, {2048: GRID})  # C_KV outside the grid -> descriptive only
    assert res["units"][0]["status"] == "descriptive" and res["verdict"] == "not refuted"


def test_h_b3_tpot_grows_slower():
    rows = runs_for(8192, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10 + C / 20)
    assert ba.h_b3(rows, {8192: GRID})["verdict"] == "not refuted"
    rows = runs_for(8192, GRID, thr=saturating, ttft=lambda C: 100 + C, tpot=lambda C: 10 * C)
    assert ba.h_b3(rows, {8192: GRID})["verdict"] == "refuted"


def detailed(ttfts_short, n_long=0, L=2048):
    lens = [L - 128] * len(ttfts_short) + [32640] * n_long
    return {"input_lens": lens, "ttfts": [t / 1000 for t in ttfts_short] + [5.0] * n_long}


def test_short_class_p95_uses_only_short_requests():
    d = detailed([100.0] * 95 + [200.0] * 5, n_long=10)
    assert ba.short_class_p95_ttft_ms(d, L=2048) == pytest.approx(ba.percentile([100.0] * 95 + [200.0] * 5, 95))


def test_h_b4_increase_and_noise():
    uni = {r: detailed([100.0 + r] * 200) for r in (1, 2, 3)}
    mixed = {r: detailed([130.0 + r] * 180, n_long=20) for r in (1, 2, 3)}
    res = ba.h_b4({2048: {"uniform": uni, "mixed": mixed, "C": 22}})
    assert res["verdict"] == "not refuted" and res["units"][0]["increase_pct"] == pytest.approx((132 / 102 - 1) * 100)
    small = {r: detailed([110.0 + r] * 180, n_long=20) for r in (1, 2, 3)}
    assert ba.h_b4({2048: {"uniform": uni, "mixed": small, "C": 22}})["verdict"] == "refuted"  # +9.8%
    noisy_u = {1: detailed([60.0] * 200), 2: detailed([100.0] * 200), 3: detailed([140.0] * 200)}
    noisy_m = {1: detailed([60.0] * 180, 20), 2: detailed([150.0] * 180, 20), 3: detailed([240.0] * 180, 20)}
    res = ba.h_b4({2048: {"uniform": noisy_u, "mixed": noisy_m, "C": 22}})
    assert res["units"][0]["beyond_noise"] is False and res["verdict"] == "refuted"


# ------------------------------------------------- pre-freeze review fixes (013) --

def test_runs_that_failed_validation_never_enter_the_means():
    rows = runs_for(2048, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10)
    rows.append({"L": 2048, "mix": "uniform", "restart": 4, "C": 256, "ok": False, "output_throughput": 1e9,
                 "p95_ttft_ms": 1.0, "p95_tpot_ms": 1.0})  # e.g. failed > 0 or output_lens != 128
    assert ba.knee(rows, 2048, GRID)["c_knee"] == 22


def test_a_grid_c_without_three_valid_restarts_makes_the_unit_not_testable():
    """n = 3 restarts per C (§2); a C_max that failed is never replaced by a smaller C (013)."""
    rows = runs_for(2048, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10 + C / 20)
    for r in rows:
        if r["C"] == 256 and r["restart"] == 2:
            r["ok"] = False
    k = ba.knee(rows, 2048, GRID)
    assert k["c_knee"] is None and k["missing_c"] == [256]
    assert ba.h_b1(rows, {2048: GRID})["units"][0] == {"L": 2048, "status": "not testable", "missing_c": [256]}
    assert ba.h_b2(rows, {2048: 22}, {2048: GRID})["units"][0]["status"] == "not testable"
    b3 = ba.h_b3(rows, {2048: GRID})
    assert b3["units"][0]["status"] == "not testable" and b3["verdict"] == "not refuted"
    gone = [r for r in runs_for(2048, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10) if r["C"] != 256]
    assert ba.h_b3(gone, {2048: GRID})["units"][0]["missing_c"] == [256]


def test_plan_json_string_keys_are_accepted():
    """core_b_plan stores c_kv / grids with str(L) keys (JSON); rows carry int L (013)."""
    rows = runs_for(2048, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10 + C / 20)
    res = ba.h_b2(rows, {"2048": 22}, {"2048": GRID})
    assert res["units"][0]["status"] == "tested" and res["units"][0]["L"] == 2048
    assert ba.h_b1(rows, {"2048": GRID})["units"][0]["status"] == "supported"
    assert ba.h_b3(rows, {"2048": GRID})["units"][0]["c_max"] == 256


def test_h_b4_needs_all_three_restarts_of_each_mix():
    uni = {r: detailed([100.0 + r] * 200) for r in (1, 2)}
    mixed = {r: detailed([130.0 + r] * 180, n_long=20) for r in (1, 2, 3)}
    assert ba.h_b4({2048: {"uniform": uni, "mixed": mixed, "C": 22}})["units"][0]["status"] == "not testable"


def test_h_b4_values_matches_the_detailed_form():
    uni = {r: detailed([100.0 + r] * 200) for r in (1, 2, 3)}
    mixed = {r: detailed([130.0 + r] * 180, n_long=20) for r in (1, 2, 3)}
    values = {2048: {"C": 22, "uniform": {r: ba.short_class_p95_ttft_ms(x, L=2048) for r, x in uni.items()},
                     "mixed": {r: ba.short_class_p95_ttft_ms(x, L=2048) for r, x in mixed.items()}}}
    assert ba.h_b4_values(values) == ba.h_b4({2048: {"uniform": uni, "mixed": mixed, "C": 22}})
