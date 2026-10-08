"""Tests for benchmarks/core_a_analysis.py: the pre-registered Core A decision rules
(PREREGISTRATION §2 H-A1..H-A4, 004 spill as revised by 009), written before any data."""

import math

import pytest

import core_a_analysis as an

GiB = 2**30
QUANTS = ("Q8_0", "Q6_K", "Q4_K_M")
LS = (2048, 8192, 32768)


def bench(kind, size, quant, L, rep, *, ts=None, ns=None, outcome="ok"):
    m = None
    if outcome == "ok":
        m = {"avg_ts": ts, "avg_ns": ns}
    return {"kind": kind, "size": size, "quant": quant, "L": L, "rep": rep, "outcome": outcome, "metrics": m,
            "summary": None}


def fit(size, quant, L, rep, outcome="ok", self_mib=None):
    s = None
    if self_mib is not None:
        s = {"memory": {"devices": [{"name": "CUDA0", "self": self_mib}], "host": {"self": 100}, "other": []}}
    return {"kind": "fit", "size": size, "quant": quant, "L": L, "rep": rep, "outcome": outcome, "metrics": None,
            "summary": s}


def grid(decode, ttft, *, sizes=("4B", "9B"), noise=(1.0, 1.0, 1.0)):
    """decode(size, quant, L) -> tok/s and ttft(size, quant, L) -> seconds; 3 reps with multiplicative noise."""
    rows = []
    for size in sizes:
        for q in QUANTS:
            for L in LS:
                for rep, f in enumerate(noise, start=1):
                    rows.append(fit(size, q, L, rep))
                    rows.append(bench("decode", size, q, L, rep, ts=decode(size, q, L) * f))
                    rows.append(bench("ttft", size, q, L, rep, ns=ttft(size, q, L) * 1e9 * f))
    return rows


FILE = {("4B", "Q8_0"): 4.48e9, ("4B", "Q6_K"): 3.46e9, ("4B", "Q4_K_M"): 2.71e9,
        ("9B", "Q8_0"): 9.51e9, ("9B", "Q6_K"): 7.35e9, ("9B", "Q4_K_M"): 5.62e9}
EMBD = {("4B", q): 0 for q in QUANTS} | {("9B", "Q8_0"): 1.08e9, ("9B", "Q6_K"): 0.83e9, ("9B", "Q4_K_M"): 0.6e9}


def ideal_decode(size, q, L):
    return 300e9 / an.decode_bytes(FILE[(size, q)], EMBD[(size, q)], L)  # perfectly bandwidth bound


def ideal_ttft(size, q, L):
    p = {"4B": 3.57e9, "9B": 6.92e9}[size]
    return (2 * p * L + 65536 * L * L) / 20e12


def test_cell_stats_process_level_mean_and_se():
    rows = grid(lambda *a: 100.0, lambda *a: 1.0, noise=(0.99, 1.0, 1.01))
    c = an.cell_stats(rows)[("4B", "Q8_0", 2048)]
    assert c["outcome"] == "ok"
    assert c["decode"]["m"] == pytest.approx(100.0) and c["decode"]["n"] == 3
    assert c["decode"]["se"] == pytest.approx(1.0 / math.sqrt(3))
    assert c["prefill"]["m"] == pytest.approx(2048 / 1.0)


def test_any_oom_process_makes_the_cell_oom():
    rows = grid(lambda *a: 100.0, lambda *a: 1.0)
    rows.append(bench("ttft", "9B", "Q8_0", 32768, 4, outcome="oom"))
    assert an.cell_stats(rows)[("9B", "Q8_0", 32768)]["outcome"] == "oom"


def test_h_a1_prediction_and_refutation():
    assert an.predict_fit(n_bytes=8.9 * GiB, v_total=15.92 * GiB, b=1.0 * GiB) == "fit"
    assert an.predict_fit(n_bytes=14.5 * GiB, v_total=15.92 * GiB, b=1.0 * GiB) == "oom"
    assert an.predict_fit(n_bytes=13.8 * GiB, v_total=15.92 * GiB, b=1.0 * GiB) == "boundary"
    obs = {("9B", "Q8_0", 32768): "ok", ("4B", "Q8_0", 2048): "oom"}
    pred = {("9B", "Q8_0", 32768): "fit", ("4B", "Q8_0", 2048): "fit"}
    res = an.h_a1(pred, obs, spill=set())
    assert res["verdict"] == "refuted" and res["refuting"] == [["4B", "Q8_0", 2048]]
    assert an.h_a1(pred, {k: "ok" for k in obs}, spill={("9B", "Q8_0", 32768)})["verdict"] == "refuted"
    assert an.h_a1(pred, {k: "ok" for k in obs}, spill=set())["verdict"] == "not refuted"


def test_h_a1_n_subtracts_token_embd_only_on_cuda():
    n_cuda = an.fit_need_bytes(file_bytes=9.51e9, embd_bytes=1.08e9, L=32768, platform="wsl2-cuda")
    n_mac = an.fit_need_bytes(file_bytes=9.51e9, embd_bytes=1.08e9, L=32768, platform="macos-metal")
    assert n_mac - n_cuda == pytest.approx(1.08e9)
    assert n_cuda == pytest.approx(9.51e9 - 1.08e9 + 32768 * 32768 + an.S_BYTES)


def test_h_a2_ideal_bandwidth_model_is_not_refuted():
    cells = an.cell_stats(grid(ideal_decode, ideal_ttft))
    res = an.h_a2(cells, FILE, EMBD, spill=set())
    assert res["verdict"] == "not refuted" and len(res["units"]) == 15


def test_h_a2_refuted_when_one_config_is_25pct_off():
    cells = an.cell_stats(grid(lambda s, q, L: ideal_decode(s, q, L) * (0.75 if (s, q) == ("9B", "Q6_K") else 1),
                               ideal_ttft))
    res = an.h_a2(cells, FILE, EMBD, spill=set())
    assert res["verdict"] == "refuted"
    assert all("9B-Q6_K" in u["pair"] for u in res["units"] if u["refutes"])


def test_h_a3_order_and_noise():
    cells = an.cell_stats(grid(ideal_decode, ideal_ttft))
    res = an.h_a3(cells, spill=set())
    assert res["verdict"] == "not refuted"
    d = {u["size"]: u["D"] for u in res["units"]}
    assert d["4B"]["Q4_K_M"] > d["4B"]["Q6_K"] > d["4B"]["Q8_0"]

    def swapped(s, q, L):  # Q8_0 loses much more at 32K than Q6_K -> reversed and beyond noise
        return ideal_decode(s, q, L) * (0.5 if (q, L) == ("Q8_0", 32768) else 1)

    res = an.h_a3(an.cell_stats(grid(swapped, ideal_ttft, noise=(0.999, 1.0, 1.001))), spill=set())
    assert res["verdict"] == "refuted"


def test_h_a3_reversal_within_noise_does_not_refute():
    def tiny(s, q, L):
        return ideal_decode(s, q, L) * (0.96 if (q, L) == ("Q8_0", 32768) and s == "9B" else 1)  # reversed by ~0.01

    res = an.h_a3(an.cell_stats(grid(tiny, ideal_ttft, noise=(0.95, 1.0, 1.05))), spill=set())
    assert res["verdict"] == "not refuted"


def test_h_a3_not_testable_with_fewer_than_two_configs_at_32k():
    rows = [r for r in grid(ideal_decode, ideal_ttft)
            if not (r["size"] == "9B" and r["L"] == 32768 and r["quant"] != "Q4_K_M")]
    rows += [bench("decode", "9B", q, 32768, 1, outcome="oom") for q in ("Q8_0", "Q6_K")]
    res = an.h_a3(an.cell_stats(rows), spill=set())
    nine = next(u for u in res["units"] if u["size"] == "9B")
    assert nine["status"] == "not testable"


def test_h_a4_ideal_compute_model():
    cells = an.cell_stats(grid(ideal_decode, ideal_ttft))
    a = an.h_a4a(cells, spill=set())
    b = an.h_a4b(cells, spill=set())
    assert a["verdict"] == "not refuted" and len(a["units"]) == 6  # (size, L) units on one platform
    assert b["verdict"] == "not refuted"
    r = {u["config"]: u["ratio"] for u in b["units"]}
    assert r["9B-Q8_0"] == pytest.approx(4.448, abs=0.01) and r["4B-Q8_0"] == pytest.approx(4.839, abs=0.01)


def test_h_a4b_refuted_when_ttft_is_linear():
    cells = an.cell_stats(grid(ideal_decode, lambda s, q, L: L / 1000.0))
    assert an.h_a4b(cells, spill=set())["verdict"] == "refuted"


def test_spill_rules_rtx():
    cells = an.cell_stats(grid(ideal_decode, ideal_ttft)
                          + [fit("9B", "Q8_0", 32768, 9, self_mib=15000)])
    params = {"v_total_mib": 16303, "b_mib": 1000, "c_ctx_mib": 400}
    spill = an.spill_cells(cells, FILE, EMBD, params)
    assert ("9B", "Q8_0", 32768) in spill  # 15000 + 400 + 1000 > 16303
    slow = an.cell_stats(grid(lambda s, q, L: ideal_decode(s, q, L) * (0.4 if (s, q, L) == ("9B", "Q6_K", 8192)
                                                                      else 1), ideal_ttft))
    assert an.spill_cells(slow, FILE, EMBD, params) == {("9B", "Q6_K", 8192)}
