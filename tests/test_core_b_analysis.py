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
            out.append({"L": L, "mix": mix, "restart": r, "C": C, "output_throughput": thr(C) * eps,
                        "p95_ttft_ms": ttft(C) * eps, "p95_tpot_ms": tpot(C) * eps})
    return out


GRID = [1, 2, 4, 8, 11, 16, 22, 31, 32, 44, 64, 128, 256]


def saturating(C, knee=22):
    return 100.0 * min(C, knee)


def test_knee_is_smallest_c_reaching_95pct_of_the_max():
    rows = runs_for(2048, GRID, thr=saturating, ttft=lambda C: 100, tpot=lambda C: 10)
    k = ba.knee(rows, 2048)
    assert k["c_knee"] == 22 and k["c_max"] == 256
    rows = runs_for(2048, GRID, thr=lambda C: 100.0 * min(C, 21.5), ttft=lambda C: 100, tpot=lambda C: 10)
    assert ba.knee(rows, 2048)["c_knee"] == 22  # 0.95 x 2150 = 2042.5 -> 22 (2150) not 16 (1600)


def queueing_ttft(C, knee=22):
    return 100.0 if C <= knee else 100.0 * C / knee


def test_h_b1_supported_refuted_inconclusive():
    rows = runs_for(2048, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10)
    assert ba.h_b1(rows, [2048])["units"][0]["status"] == "supported"
    flat = runs_for(2048, GRID, thr=saturating, ttft=lambda C: 100, tpot=lambda C: 10)
    assert ba.h_b1(flat, [2048])["verdict"] == "refuted"  # grid reaches 256 >= 4 x 22, no 3x
    short = [g for g in GRID if g <= 64]
    flat = runs_for(2048, short, thr=saturating, ttft=lambda C: 100, tpot=lambda C: 10)
    assert ba.h_b1(flat, [2048])["units"][0]["status"] == "inconclusive"  # 64 < 88


def test_h_b2_interval_and_out_of_grid():
    rows = runs_for(2048, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10)
    res = ba.h_b2(rows, {2048: 22})
    assert res["verdict"] == "not refuted" and res["units"][0]["interval"] == [11, 44]
    res = ba.h_b2(rows, {2048: 8})  # knee 22 > 16
    assert res["verdict"] == "refuted"
    res = ba.h_b2(rows, {2048: 400})  # C_KV outside the grid -> descriptive only
    assert res["units"][0]["status"] == "descriptive" and res["verdict"] == "not refuted"


def test_h_b3_tpot_grows_slower():
    rows = runs_for(8192, GRID, thr=saturating, ttft=queueing_ttft, tpot=lambda C: 10 + C / 20)
    assert ba.h_b3(rows, [8192])["verdict"] == "not refuted"
    rows = runs_for(8192, GRID, thr=saturating, ttft=lambda C: 100 + C, tpot=lambda C: 10 * C)
    assert ba.h_b3(rows, [8192])["verdict"] == "refuted"


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
