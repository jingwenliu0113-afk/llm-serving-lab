"""Tests for benchmarks/adv_analysis.py: decision 024 A1-A13 and the interpretation matrix R0-R7
(PREREGISTRATION §7), written before any Advanced data. Unit = server restart (n = 3)."""

import copy
import math
import random

import pytest

import adv_analysis as aa
import core_b_analysis as ba

LEVELS = {2048: 25, 8192: 8}
# (U, M) short-class P95 TTFT means per condition; B0 rho = ln 4. T / G / TG default to category N
# (f = 1 - ln 3.8 / ln 4 = 0.037), K3 to "no dose" (a = 0).
BASE = {"B0": (100.0, 400.0), "T": (100.0, 380.0), "G": (100.0, 380.0), "TG": (100.0, 380.0), "K3": (100.0, 400.0)}
GRIDS = {2048: [1, 8, 16, 25, 32, 64], 8192: [1, 4, 8, 16, 32]}
BLOCKED = {"HOL-long": 0.5, "KV-short": 0.2, "no-queue": 0.3}


def decomp(shares=None, sens=None, Q=100.0, valid=True):
    s = dict.fromkeys(aa.Q_CATEGORIES, 0.0)
    s.update(shares or {"no-queue": 1.0})
    t = dict(s)
    if sens is not None:
        t = dict.fromkeys(aa.Q_CATEGORIES, 0.0)
        t.update(sens)
    return {"valid": valid, "checks": {"a": valid, "b": True, "c": True, "d": True},
            "tail": {"O": 1.0, "Q": Q, "P": 50.0}, "n_tail": 20,
            "tail95": {"O": 1.5, "Q": 1.5 * Q, "P": 60.0, "F": 0.0}, "n_tail95": 10, "q_shares": s,
            "q_shares_sens": t, "residual_bm_share": 0.0, "unexplained_break_share": 0.01}


QUEUE_HIST, PREFILL_HIST = "vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds"


def hist(mean):
    """One run_adv.scrape_deltas histogram entry (count 200)."""
    return None if mean is None else {"buckets": {"+Inf": 200.0}, "sum": 200 * mean, "count": 200.0, "mean": mean}


def mkrow(cond, L, mix, r, y, **over):
    mixed = mix == "mixed"
    row = {"cond": cond, "L": L, "mix": mix, "restart": r, "block": r, "C": LEVELS[L], "N": max(10 * LEVELS[L], 200),
           "ok": True, "reasons": [], "timeout": False, "skipped": False, "completed": max(10 * LEVELS[L], 200),
           "failed": 0, "short_class_p95_ttft_ms": y, "output_throughput": 1000.0, "total_token_throughput": 9000.0,
           "p95_tpot_ms": 30.0, "long_class_p95_ttft_ms": 5000.0 if mixed else None, "preemptions": 0,
           "kv_peak": 0.9,
           # run_adv's per-run descriptive fields (client_stats, one_hz_shares, scrape_deltas)
           "client": {"short_p50_ttft_ms": y / 2, "short_p99_ttft_ms": 1.2 * y,
                      "long_p50_ttft_ms": 4000.0 if mixed else None, "long_p95_ttft_ms": 5000.0 if mixed else None,
                      "short_p95_tpot_ms": 30.0},
           "one_hz": {"kv_full_while_waiting": 0.1, "gate_zone": 0.2, "n_samples": 100},
           "scrape_delta": {QUEUE_HIST: hist(0.05), PREFILL_HIST: hist(0.2), "errors": []},
           "manip": {"facts_ok": True, "blocks_ok": True, "chunk_ok": True if cond in ("T", "TG") else None,
                     "num_gpu_blocks": 186 if cond == "K3" else 248},
           "decomp": decomp(Q=100.0 if mixed else 10.0)}
    row.update(over)
    return row


def dataset(spec=None):
    """spec: {L: {cond: (U, M)}} overriding BASE. 3 restarts per cell: m - m/256, m, m + m/256 (exact mean m)."""
    rows = []
    for L in LEVELS:
        cells = {**BASE, **(spec or {}).get(L, {})}
        for cond, (u, m) in cells.items():
            for mix, v in (("uniform", u), ("mixed", m)):
                d = v / 256
                for r, y in zip((1, 2, 3), (v - d, v, v + d)):
                    rows.append(mkrow(cond, L, mix, r, y))
    return rows


def patch(rows, cond, L, mix, r=None, **over):
    for row in rows:
        if row["cond"] == cond and row["L"] == L and row["mix"] == mix and (r is None or row["restart"] == r):
            row.update(copy.deepcopy(over))
    return rows


def uniform_core(knees=None):
    knees = knees or LEVELS
    rows = [{"L": L, "mix": "uniform", "restart": r, "C": C, "ok": True, "output_throughput": 100.0 * min(C, knees[L]),
             "short_class_p95_ttft_ms": 80.0}
            for L, grid in GRIDS.items() for r in (1, 2, 3) for C in grid]
    return {"session": {"session_id": "formal-20261009-coreb-uniform", "formal": True, "env_valid": True},
            "plan": {"grids": {str(L): g for L, g in GRIDS.items()}}, "rows": rows}


def adv(rows, plan_c=None, **session):
    return {"session": {"session_id": "formal-20261011-adv", "formal": True, "env_valid": True, **session},
            "plan": {"schema_version": 1, "seed": 20261008, "derived_from": "formal-20261009-coreb-uniform",
                     "c": plan_c or {"2048": 25, "8192": 8}, "conditions": list(aa.CONDITIONS)},
            "rows": rows}


def run(rows, **kw):
    return aa.analyze(adv(rows), uniform_core(), **kw)


def lv(res, L=2048):
    return res["levels"][str(L)]


def cat(res, cond, L=2048):
    return lv(res, L)["conditions"][cond]["category"]


def row_of(res, L=2048):
    return lv(res, L)["interpretation"]["row"]


# ---------------------------------------------------------------- A2 / A3 statistics


def test_rho_and_delta_method_se_match_hand_calculation():
    U = aa.cell_stats({1: 99.0, 2: 100.0, 3: 101.0})
    M = aa.cell_stats({3: 402.0, 1: 398.0, 2: 400.0})
    assert U["mean"] == 100.0 and U["se"] == pytest.approx(1 / math.sqrt(3))
    assert M["mean"] == 400.0 and M["se"] == pytest.approx(2 / math.sqrt(3))
    rho, se = aa.rho_stats(U, M)
    assert rho == pytest.approx(math.log(4))
    assert se == pytest.approx(math.sqrt((2 / math.sqrt(3) / 400) ** 2 + (1 / math.sqrt(3) / 100) ** 2))
    assert aa.cell_stats({1: 1.0, 2: 2.0}) is None  # fewer than 3 restarts: no mean (013)
    assert aa.rho_stats(None, M) == (None, None)


def test_ratio_se_is_the_section_2_delta_method():
    a, se_a, b, se_b = 0.6, 0.01, 1.4, 0.02
    r = a / b
    assert aa.ratio_se(a, se_a, b, se_b) == pytest.approx(r * math.sqrt((se_a / a) ** 2 + (se_b / b) ** 2))
    assert aa.ratio_se(0.0, 0.01, 1.4, 0.02) == pytest.approx(0.01 / 1.4)  # a = 0 allowed


def test_noise_rule_matches_section_2():
    assert aa.beyond_noise(10.0, 0.0, 6.0, 2.0) is False  # |x - y| = 2 sqrt(SE^2 + SE^2) exactly: within noise
    assert aa.beyond_noise(10.1, 0.0, 6.0, 2.0) is True
    assert aa.beyond_noise(6.0, 2.0, 10.1, 0.0) is True
    assert aa.exceeds(10.1, 0.0, 6.0, 2.0) is True
    assert aa.exceeds(6.0, 2.0, 10.1, 0.0) is False  # directional: hi must be the larger one
    assert aa.exceeds(10.0, 0.0, 6.0, 2.0) is False


# ---------------------------------------------------------------- A5 categories


def test_category_boundaries_go_to_the_higher_category():
    c = aa.removal_category
    assert c(0.5, True, True, False) == "M"
    assert c(0.4999, True, True, False) == "P"
    assert c(0.2, True, True, False) == "P"
    assert c(0.1999, True, True, False) == "N"
    assert c(0.2, True, False, False) == "R"
    assert c(0.1999, True, False, False) == "N"
    assert c(0.9, True, False, False) == "R"
    assert c(-0.2, False, False, True) == "W"
    assert c(-0.1999, False, False, True) == "N"
    assert c(0.9, False, True, False) == "N"  # no rho drop beyond noise
    assert c(-0.9, False, False, False) == "N"  # rho rise within noise


def test_category_boundaries_end_to_end_with_exact_values():
    # 2048: rho_0 = ln 16 = 4 ln 2; G rho = ln 4 -> f = 0.5 exactly -> M.
    # 8192: rho_0 = ln 32 = 5 ln 2; T rho = ln 16 -> f = 0.2 -> P; TG rho = ln 64 -> f = -0.2 -> W.
    res = run(dataset({2048: {"B0": (100.0, 1600.0), "G": (100.0, 400.0)},
                       8192: {"B0": (100.0, 3200.0), "T": (100.0, 1600.0), "TG": (100.0, 6400.0)}}))
    assert lv(res)["conditions"]["G"]["f"] == pytest.approx(0.5)
    assert cat(res, "G") == "M"
    assert lv(res, 8192)["conditions"]["T"]["f"] == pytest.approx(0.2)
    assert cat(res, "T", 8192) == "P"
    assert lv(res, 8192)["conditions"]["TG"]["f"] == pytest.approx(-0.2)
    assert cat(res, "TG", 8192) == "W"


def test_m_requires_m_to_drop_r_class_and_uniform_inflation():
    # T: U doubles, M unchanged -> rho halves (f = 0.5) but M did not drop -> R, never M.
    # G: U 3x, M rises -> f = 0.76 -> R.
    res = run(dataset({2048: {"T": (200.0, 400.0), "G": (300.0, 420.0)}}))
    t = lv(res)["conditions"]["T"]
    assert t["f"] == pytest.approx(0.5) and t["d_rho"] is True and t["d_m"] is False
    assert cat(res, "T") == "R" and cat(res, "G") == "R"
    h1 = lv(res)["hypotheses"]["H-C1"]
    assert h1["status"] == "not supported" and h1["category"] == "R"
    assert "chunk 上限的副作用或 uniform 位移遮蔽" in h1["note"]
    assert row_of(res) == "R7"


def test_w_class():
    res = run(dataset({2048: {"G": (100.0, 800.0)}}))  # rho_G = ln 8, f = -0.5, rise beyond noise
    g = lv(res)["conditions"]["G"]
    assert g["f"] == pytest.approx(-0.5) and g["u_rho"] is True and g["category"] == "W"
    assert lv(res)["hypotheses"]["H-C2"]["status"] == "not supported"


def test_f_se_and_g_and_a_are_reported():
    res = run(dataset({2048: {"T": (100.0, 180.0)}}))
    b0, t = lv(res)["conditions"]["B0"], lv(res)["conditions"]["T"]
    r = t["rho"] / b0["rho"]
    assert t["f"] == pytest.approx(1 - r) and t["a"] == pytest.approx(r - 1)
    assert t["f_se"] == pytest.approx(r * math.sqrt((t["rho_se"] / t["rho"]) ** 2 + (b0["rho_se"] / b0["rho"]) ** 2))
    assert t["g"] == pytest.approx(1 - (180 - 100) / (400 - 100))
    assert t["ln_u_vs_b0"] == pytest.approx(0.0)


# ---------------------------------------------------------------- A1 validity and NT


@pytest.mark.parametrize("over", [
    {"ok": False}, {"timeout": True}, {"skipped": True}, {"short_class_p95_ttft_ms": None},
    {"manip": {"facts_ok": False, "blocks_ok": True, "chunk_ok": True, "num_gpu_blocks": 248}},
    {"manip": {"facts_ok": True, "blocks_ok": False, "chunk_ok": True, "num_gpu_blocks": 247}},
    {"manip": {"facts_ok": True, "blocks_ok": True, "chunk_ok": False, "num_gpu_blocks": 248}},
])
def test_an_invalid_run_makes_the_cell_nt_without_substitution(over):
    rows = patch(dataset({2048: {"T": (100.0, 180.0)}}), "T", 2048, "mixed", 2, **over)
    if "short_class_p95_ttft_ms" not in over:
        patch(rows, "T", 2048, "mixed", 2, short_class_p95_ttft_ms=1.0)  # an invalid run's Y is never used
    res = run(rows)
    t = lv(res)["conditions"]["T"]
    assert t["n_valid"] == {"uniform": 3, "mixed": 2}
    assert t["M"] is None and t["rho"] is None and t["category"] == "NT"
    assert lv(res)["hypotheses"]["H-C1"]["status"] == "not testable"
    assert cat(res, "T", 8192) == "N"  # other cells unaffected


def test_row_valid_follows_the_contract():
    good = mkrow("B0", 2048, "uniform", 1, 100.0)
    assert aa.row_valid(good) is True  # chunk_ok None (not T / TG) counts
    assert aa.row_valid({**good, "manip": {**good["manip"], "chunk_ok": True}}) is True
    assert aa.invalid_reasons({**good, "timeout": True, "ok": False}) == ["not ok", "timeout"]


def test_malformed_rows_raise():
    rows = dataset()
    with pytest.raises(aa.AnalysisError, match="duplicate"):
        run(rows + [copy.deepcopy(rows[0])])
    with pytest.raises(aa.AnalysisError, match="restart"):
        run(rows + [mkrow("T", 2048, "mixed", 4, 180.0)])  # a 4th restart never replaces a failed one
    with pytest.raises(aa.AnalysisError, match="condition table"):
        run(rows + [mkrow("X", 2048, "mixed", 1, 180.0)])


# ---------------------------------------------------------------- A6 uniform specificity


def test_uniform_specificity_qualifies_g_tg_and_blocks_m():
    # G: U +20 % (|ln 1.2| = 0.18 > 0.10), rho = ln 2 -> raw M but qualified -> not R2.
    res = run(dataset({2048: {"G": (120.0, 240.0)}, 8192: {"TG": (115.0, 230.0)}}))
    g = lv(res)["conditions"]["G"]
    assert g["category"] == "M" and g["qualified"] is True and g["effective"] == "qualified"
    assert g["specificity"]["ref"] == "B0" and g["specificity"]["ln_ratio"] == pytest.approx(math.log(1.2))
    assert lv(res)["hypotheses"]["H-C2"]["status"] == "qualified"
    assert row_of(res) == "R7"
    tg = lv(res, 8192)["conditions"]["TG"]  # TG is checked against T (U_T = 100)
    assert tg["category"] == "M" and tg["specificity"]["ref"] == "T" and tg["effective"] == "qualified"
    assert lv(res, 8192)["hypotheses"]["H-C3"]["status"] == "qualified"
    assert row_of(res, 8192) == "R7"
    assert lv(res)["conditions"]["T"]["specificity"] is None  # T legitimately moves U: no check
    assert lv(res)["conditions"]["T"]["qualified"] is None


def test_uniform_specificity_is_inclusive_at_0_10():
    res = run(dataset({2048: {"G": (100.0 * math.exp(0.1), 100.0 * math.exp(0.1) * 1.8)}}))
    g = lv(res)["conditions"]["G"]
    assert g["specificity"]["ok"] is True and g["effective"] == "M" and row_of(res) == "R2"


def test_tg_m_with_unknown_specificity_is_not_used_as_m():
    rows = patch(dataset({2048: {"TG": (100.0, 180.0)}}), "T", 2048, "uniform", 1, timeout=True)
    res = run(rows)
    tg = lv(res)["conditions"]["TG"]
    assert tg["category"] == "M" and tg["specificity"]["ok"] is None and tg["effective"] == "NT"
    assert row_of(res) == "R7" and lv(res)["interpretation"]["incomplete"] is True


# ---------------------------------------------------------------- A7 e_KV


def test_kv_dose_component():
    res = run(dataset({2048: {"K3": (100.0, 600.0)},                       # a = ln6/ln4 - 1 = 0.29
                       8192: {"K3": (120.0, 720.0)}}))                     # dose but U_K3 +20 %
    i = lv(res)["kv"]["i"]
    assert i["status"] == "holds" and i["holds"] is True and i["a"] == pytest.approx(math.log(6) / math.log(4) - 1)
    i8 = lv(res, 8192)["kv"]["i"]
    assert i8["status"] == "qualified" and i8["holds"] is False
    assert lv(res, 8192)["hypotheses"]["H-C4"]["qualified_i"] is True
    res = run(dataset({2048: {"K3": (100.0, 450.0)}}))  # a = 0.085 < 0.20
    assert lv(res)["kv"]["i"]["status"] == "not"
    rows = patch(dataset({2048: {"K3": (100.0, 600.0)}}), "K3", 2048, "mixed", 3, timeout=True)
    i = lv(run(rows))["kv"]["i"]
    assert i["status"] == "NT" and i["holds"] is False  # a timed-out K3 run is no evidence


def test_kv_observation_component():
    rows = patch(dataset(), "B0", 2048, "mixed", decomp=decomp(BLOCKED))
    ii = lv(run(rows))["kv"]["ii"]
    assert ii["status"] == "holds" and ii["share"] == pytest.approx(0.7)
    rows = patch(dataset(), "B0", 2048, "mixed", decomp=decomp({"HOL-long": 0.2, "KV-short": 0.1, "no-queue": 0.7}))
    assert lv(run(rows))["kv"]["ii"]["status"] == "not"
    rows = patch(dataset(), "B0", 2048, "mixed", decomp=decomp({"HOL-long": 0.25, "KV-short": 0.25, "no-queue": 0.5}))
    assert lv(run(rows))["kv"]["ii"]["status"] == "holds"  # 0.5 is inclusive
    rows = patch(dataset(), "B0", 2048, "mixed", decomp=decomp({"unexplained": 0.9, "no-queue": 0.1}))
    assert lv(run(rows))["kv"]["ii"]["status"] == "not"  # only HOL-long + KV-short count
    for bad in ({"decomp": decomp(BLOCKED, valid=False)}, {"decomp": None}):
        rows = patch(dataset(), "B0", 2048, "mixed", decomp=decomp(BLOCKED))
        res = run(patch(rows, "B0", 2048, "mixed", 2, **bad))
        assert lv(res)["kv"]["ii"]["status"] == "NT" and lv(res)["kv"]["ii"]["holds"] is False
        assert lv(res)["hypotheses"]["H-C4"]["incomplete"] is True
    # an invalid B0-mixed run already makes H-C0 not testable (R0), so (ii) is never read from it
    rows = patch(dataset(), "B0", 2048, "mixed", decomp=decomp(BLOCKED))
    res = run(patch(rows, "B0", 2048, "mixed", 2, timeout=True))
    assert lv(res)["kv"] is None and row_of(res) == "R0"


def test_kv_observation_nt_marks_h_c4_incomplete():
    rows = patch(dataset(), "B0", 2048, "mixed", decomp=decomp(BLOCKED))
    res = run(patch(rows, "B0", 2048, "mixed", 1, decomp=decomp(BLOCKED, valid=False)))
    h4 = lv(res)["hypotheses"]["H-C4"]
    assert h4["incomplete"] is True and h4["status"] == "not supported"


def test_kv_gate_protective_component():
    rows = patch(dataset({2048: {"G": (100.0, 800.0)}}), "G", 2048, "mixed", preemptions=3)
    res = run(rows)
    assert lv(res)["kv"]["iii"]["status"] == "holds" and lv(res)["kv"]["iii"]["preemptions"] == 9
    h2 = lv(res)["hypotheses"]["H-C2"]
    assert h2["status"] == "not supported" and "閘門具保護作用" in h2["note"]
    res = run(dataset({2048: {"G": (100.0, 800.0)}}))  # W without preemption
    assert lv(res)["kv"]["iii"]["status"] == "not"
    res = run(patch(dataset({2048: {"G": (100.0, 380.0)}}), "G", 2048, "mixed", preemptions=5))  # N
    assert lv(res)["kv"]["iii"]["status"] == "not"
    rows = patch(dataset({2048: {"G": (100.0, 800.0)}}), "G", 2048, "mixed", preemptions=3)
    res = run(patch(rows, "G", 2048, "mixed", 2, timeout=True, preemptions=50))
    assert lv(res)["kv"]["iii"]["status"] == "NT" and lv(res)["kv"]["e_kv"] == 0


def test_e_kv_counts_the_components():
    rows = patch(dataset({2048: {"K3": (100.0, 600.0), "G": (100.0, 800.0)}}), "G", 2048, "mixed", preemptions=1)
    rows = patch(rows, "B0", 2048, "mixed", decomp=decomp(BLOCKED))
    kv = lv(run(rows))["kv"]
    assert (kv["i"]["holds"], kv["ii"]["holds"], kv["iii"]["holds"], kv["e_kv"]) == (True, True, True, 3)
    assert lv(run(dataset()))["kv"]["e_kv"] == 0


# ---------------------------------------------------------------- A4 H-C0


def test_h_c0_failure_makes_the_level_uninterpretable_via_h_b4_values(monkeypatch):
    calls = []
    real = ba.h_b4_values

    def spy(per_level):
        calls.append(copy.deepcopy(per_level))
        return real(per_level)

    monkeypatch.setattr(ba, "h_b4_values", spy)
    res = run(dataset({2048: {"B0": (100.0, 110.0), "T": (100.0, 30.0)}, 8192: {"T": (100.0, 180.0)}}))
    assert len(calls) == 1 and set(calls[0]) == {2048, 8192}
    assert calls[0][2048]["C"] == 25 and calls[0][8192]["C"] == 8
    assert calls[0][2048]["uniform"] == {1: 100 - 100 / 256, 2: 100.0, 3: 100 + 100 / 256}
    two = lv(res)
    assert two["h_c0"]["passed"] is False and two["h_c0"]["increase_pct"] == pytest.approx(10.0)
    assert two["interpretation"]["row"] == "R0"
    assert two["interpretation"]["conclusion"] == "該 L 在 session 內沒有重現 H-B4，不做歸因"
    assert two["interpretation"]["incomplete"] is False and two["interpretation"]["incomplete_flips"] == []
    assert two["interpretation"]["notes"] == []
    assert set(two["hypotheses"]) == {"H-C1", "H-C2", "H-C3", "H-C4", "H-C5"}
    assert all(h["status"] == "uninterpretable" for h in two["hypotheses"].values())
    assert two["conditions"]["T"]["category"] is None and two["kv"] is None
    assert row_of(res, 8192) == "R1"  # the other level is analysed as usual


H_C0_NT_CONCLUSION = "該 L 的 H-C0 不可檢定（B0 缺 3 次有效 restart），無法判定是否重現 H-B4，不做歸因"


@pytest.mark.parametrize("mix", ["uniform", "mixed"])
def test_h_c0_not_testable_is_not_reported_as_not_reproduced(mix):
    """Missing B0 data is not the finding that H-B4 did not reproduce (024 R0 = H-C0 failed; A1 / 013: NT)."""
    res = run(patch(dataset(), "B0", 2048, mix, 1, ok=False))
    two = lv(res)
    assert two["h_c0"]["status"] == "not testable" and two["h_c0"]["passed"] is False
    it = two["interpretation"]
    assert it["row"] == "R0"  # first match: H-C0 did not pass, so H-C1..H-C5 stay uninterpretable
    assert it["conclusion"] == H_C0_NT_CONCLUSION and "沒有重現" not in it["conclusion"]
    assert it["incomplete"] is True and it["incomplete_flips"] == ["H-C0"]  # a passing H-C0 would move the row
    assert it["notes"] == ["H-C0 不可檢定：B0 缺 3 次有效 restart，不做歸因"]
    assert all(h["status"] == "uninterpretable" for h in two["hypotheses"].values())
    assert two["kv"] is None and two["conditions"]["T"]["category"] is None
    assert row_of(res, 8192) == "R7" and lv(res, 8192)["interpretation"]["incomplete"] is False


def test_h_c0_noise_rule_failure_is_r0():
    rows = dataset({2048: {"B0": (100.0, 130.0)}})
    patch(rows, "B0", 2048, "mixed", 1, short_class_p95_ttft_ms=60.0)
    patch(rows, "B0", 2048, "mixed", 3, short_class_p95_ttft_ms=200.0)
    res = run(rows)
    assert lv(res)["h_c0"]["beyond_noise"] is False and row_of(res) == "R0"


# ---------------------------------------------------------------- A11 H-C5


def h_c5_rows(spec=None, mixed=None, sens=None):
    rows = dataset(spec)
    return patch(rows, "B0", 2048, "mixed", decomp=decomp(mixed, sens=sens))


def test_h_c5_budget_long_maps_to_t():
    res = run(h_c5_rows({2048: {"T": (100.0, 180.0)}}, {"budget-long": 0.8, "no-queue": 0.2}))
    h5 = lv(res)["hypotheses"]["H-C5"]
    assert h5["c_star"] == "budget-long" and h5["share"] == pytest.approx(80 / 90)
    assert h5["status"] == "consistent"
    assert lv(res)["interpretation"]["h_c5"]["status"] == "consistent"
    res = run(h_c5_rows({2048: {"T": (100.0, 300.0)}}, {"budget-long": 0.8, "no-queue": 0.2}))  # P also maps
    assert lv(res)["hypotheses"]["H-C5"]["status"] == "consistent"
    res = run(h_c5_rows(None, {"budget-long": 0.8, "no-queue": 0.2}))  # T = N
    assert lv(res)["hypotheses"]["H-C5"]["status"] == "inconsistent"
    assert "機制歸因不一致，以操弄的結論為準" in lv(res)["interpretation"]["notes"]
    assert row_of(res) == "R7"  # the manipulation decides the row


def test_h_c5_hol_and_kv_short_map_to_kv_dose():
    res = run(h_c5_rows({2048: {"K3": (100.0, 600.0)}}, BLOCKED))
    assert lv(res)["hypotheses"]["H-C5"]["c_star"] == "HOL-long"
    assert lv(res)["hypotheses"]["H-C5"]["status"] == "consistent"
    res = run(h_c5_rows({2048: {"K3": (100.0, 600.0)}}, {"KV-short": 0.6, "HOL-long": 0.1, "no-queue": 0.3}))
    assert lv(res)["hypotheses"]["H-C5"]["c_star"] == "KV-short"
    assert lv(res)["hypotheses"]["H-C5"]["status"] == "consistent"
    res = run(h_c5_rows(None, BLOCKED))  # (i) does not hold
    assert lv(res)["hypotheses"]["H-C5"]["status"] == "inconsistent"


@pytest.mark.parametrize("mixed, sens, reason", [
    ({"budget-long": 0.4, "HOL-long": 0.3, "no-queue": 0.3}, None, "no dominant"),     # 40 / 90 < 0.5
    ({"budget-long": 0.3, "HOL-long": 0.3, "KV-short": 0.3, "no-queue": 0.1}, None, "no dominant"),  # tie
    (None, None, "no dominant"),                                                         # all zero
    ({"budget-long": 0.8, "no-queue": 0.2}, {"HOL-long": 0.8, "no-queue": 0.2}, "τ"),     # c* moves
])
def test_h_c5_not_testable_cases(mixed, sens, reason):
    res = run(h_c5_rows({2048: {"T": (100.0, 180.0)}}, mixed, sens))
    h5 = lv(res)["hypotheses"]["H-C5"]
    assert h5["status"] == "NT" and reason in h5["reason"]
    assert lv(res)["interpretation"]["h_c5"]["status"] == "NT"


def test_h_c5_nt_when_any_b0_decomposition_is_invalid():
    rows = h_c5_rows({2048: {"T": (100.0, 180.0)}}, {"budget-long": 0.8, "no-queue": 0.2})
    res = run(patch(rows, "B0", 2048, "uniform", 2, decomp=decomp(valid=False, Q=10.0)))
    h5 = lv(res)["hypotheses"]["H-C5"]
    assert h5["status"] == "NT" and "decomposition" in h5["reason"]


def test_h_c5_dominance_is_inclusive_at_one_half():
    res = run(h_c5_rows({2048: {"T": (100.0, 180.0)}}, {"budget-long": 0.45, "HOL-long": 0.2, "no-queue": 0.35}))
    h5 = lv(res)["hypotheses"]["H-C5"]  # dQ = 90, dQ_budget-long = 45 -> exactly 0.5
    assert h5["share"] == pytest.approx(0.5) and h5["status"] == "consistent"


# ---------------------------------------------------------------- matrix R0..R7

R5_ROWS = lambda: h_c5_rows({2048: {"K3": (100.0, 600.0)}}, BLOCKED)  # noqa: E731

SCENARIOS = {
    "R0": (lambda: dataset({2048: {"B0": (100.0, 110.0)}}), "該 L 在 session 內沒有重現 H-B4，不做歸因"),
    "R1": (lambda: dataset({2048: {"T": (100.0, 180.0)}}),
           "主因是 batching，即長 request 的 chunked prefill 壟斷每個 step 的 budget（H-C1 成立；007 的 conjecture 得到支持）"),
    "R2": (lambda: dataset({2048: {"G": (100.0, 180.0)}}),
           "主因是 scheduler 的准入閘門，即 full-ISL 檢查加上 FCFS break，由 KV 占用觸發（H-C2 成立）"),
    "R3": (lambda: dataset({2048: {"T": (100.0, 180.0), "G": (100.0, 180.0)}}),
           "兩個機制都是必要條件（conjunctive），移除任一個都消掉過半"),
    "R4": (lambda: dataset({2048: {"TG": (100.0, 180.0)}}), "串聯瓶頸，batching 與准入閘門一起作用（H-C3 成立）"),
    "R5": (R5_ROWS, "最可能是 KV 容量經 FCFS 准入造成（推論級：由排除、容量敏感、觀測三者推得；KV 無法往上調，無法用移除檢驗確認）"),
    "R6": (lambda: h_c5_rows(None, BLOCKED), "KV 可能有關，但未確立"),
    "R7": (lambda: dataset(), "在 KV、batching、scheduler 之間沒有隔離出成因"),
}


@pytest.mark.parametrize("row", sorted(SCENARIOS))
def test_each_matrix_row_is_reachable(row):
    make, conclusion = SCENARIOS[row]
    res = run(make())
    it = lv(res)["interpretation"]
    assert it["row"] == row and it["conclusion"] == conclusion
    assert it["incomplete"] is False


def test_hypothesis_outcomes_follow_the_rows():
    h = lv(run(SCENARIOS["R1"][0]()))["hypotheses"]
    assert h["H-C1"]["status"] == "supported" and h["H-C3"]["status"] == "not applicable"
    h = lv(run(SCENARIOS["R2"][0]()))["hypotheses"]
    assert h["H-C2"]["status"] == "supported" and h["H-C1"]["status"] == "not supported"
    h = lv(run(SCENARIOS["R4"][0]()))["hypotheses"]
    assert h["H-C3"]["status"] == "supported"
    h = lv(run(SCENARIOS["R5"][0]()))["hypotheses"]
    assert h["H-C4"]["status"] == "supported"
    h = lv(run(SCENARIOS["R6"][0]()))["hypotheses"]
    assert h["H-C4"]["status"] == "partial evidence"
    h = lv(run(SCENARIOS["R7"][0]()))["hypotheses"]
    assert h["H-C4"]["status"] == "not supported" and h["H-C4"]["note"] == "KV 未顯示受限或敏感"
    assert h["H-C3"]["status"] == "not supported"


R4_H_C3_NT_CONCLUSION = "移除 TG 消掉過半，但 cat(T) 或 cat(G) 為 NT，無法確認是串聯瓶頸（H-C3 不可檢定）"


@pytest.mark.parametrize("cond", ["T", "G"])
def test_r4_conclusion_does_not_claim_h_c3_when_h_c3_is_not_testable(cond):
    # TG: f = 1 - ln 1.5 / ln 4 = 0.71 -> M; every T- (or G-) mixed run invalid -> cat = NT, which is "not M":
    # the first match is R4 (024), but H-C3 is not testable, so the conclusion must not say "H-C3 成立".
    rows = patch(dataset({2048: {"TG": (100.0, 150.0)}}), cond, 2048, "mixed", ok=False)
    two = lv(run(rows))
    it, h3 = two["interpretation"], two["hypotheses"]["H-C3"]
    assert it["row"] == "R4" and h3["status"] == "not testable"
    assert it["conclusion"] == R4_H_C3_NT_CONCLUSION and "H-C3 成立" not in it["conclusion"]
    assert it["incomplete"] is True and it["incomplete_flips"] == [cond]
    # H-C3 supported keeps the 024 R4 conclusion
    res = run(dataset({2048: {"TG": (100.0, 150.0)}}))
    assert lv(res)["hypotheses"]["H-C3"]["status"] == "supported"
    assert lv(res)["interpretation"]["conclusion"] == "串聯瓶頸，batching 與准入閘門一起作用（H-C3 成立）"


def test_r1_notes():
    res = run(dataset({2048: {"T": (100.0, 180.0), "G": (100.0, 300.0)}}))  # G: f = 0.21 -> P
    it = lv(res)["interpretation"]
    assert it["row"] == "R1" and "准入閘門也有貢獻" in it["notes"] and "KV 也受限（次要）" not in it["notes"]
    rows = patch(dataset({2048: {"T": (100.0, 180.0), "K3": (100.0, 600.0)}}), "B0", 2048, "mixed",
                 decomp=decomp(BLOCKED))
    it = lv(run(rows))["interpretation"]
    assert it["row"] == "R1" and "KV 也受限（次要）" in it["notes"]


def test_r2_note_without_kv_evidence():
    it = lv(run(SCENARIOS["R2"][0]()))["interpretation"]
    assert "閘門有效，但缺少 KV 證據，不一致" in it["notes"]
    rows = patch(dataset({2048: {"G": (100.0, 180.0)}}), "B0", 2048, "mixed", decomp=decomp(BLOCKED))
    it = lv(run(rows))["interpretation"]
    assert it["row"] == "R2" and "閘門有效，但缺少 KV 證據，不一致" not in it["notes"]


def test_r5_sub_label_and_requires_ii_or_iii():
    it = lv(run(R5_ROWS()))["interpretation"]
    assert it["sub_label"] == "HOL-long 主導"
    rows = h_c5_rows({2048: {"K3": (100.0, 600.0)}}, {"KV-short": 0.6, "HOL-long": 0.1, "no-queue": 0.3})
    assert lv(run(rows))["interpretation"]["sub_label"] == "KV-short 主導"
    rows = patch(dataset({2048: {"K3": (100.0, 600.0), "G": (100.0, 800.0)}}), "G", 2048, "mixed", preemptions=2)
    it = lv(run(rows))["interpretation"]  # (i) + (iii)
    assert it["row"] == "R5" and it["sub_label"] is None
    it = lv(run(dataset({2048: {"K3": (100.0, 600.0)}})))["interpretation"]  # (i) alone
    assert it["row"] == "R6"
    rows = patch(dataset({2048: {"G": (100.0, 800.0)}}), "G", 2048, "mixed", preemptions=2)
    rows = patch(rows, "B0", 2048, "mixed", decomp=decomp(BLOCKED))
    assert row_of(run(rows)) == "R6"  # (ii) + (iii) without (i)


def test_r6_r7_list_p_and_r_contributors_and_r7_compute_load():
    res = run(dataset({2048: {"T": (100.0, 300.0), "G": (200.0, 400.0)}}))  # T = P, G = R
    it = lv(res)["interpretation"]
    assert it["row"] == "R7"
    assert it["contributors"] == [{"cond": "T", "category": "P", "qualified": None},
                                  {"cond": "G", "category": "R", "qualified": True}]
    assert "殘餘部分以 conjecture 歸給固定 C 下的封閉迴路運算負載（Little's law）" in it["notes"]
    it = lv(run(SCENARIOS["R6"][0]()))["interpretation"]
    assert it["contributors"] == [] and all("Little" not in n for n in it["notes"])


def test_every_row_carries_the_attachments():
    rows = dataset({2048: {"T": (100.0, 180.0), "G": (100.0, 800.0), "TG": (100.0, 150.0), "K3": (100.0, 600.0)}})
    patch(rows, "T", 2048, "mixed", output_throughput=1200.0, long_class_p95_ttft_ms=6000.0)
    patch(rows, "T", 2048, "uniform", output_throughput=900.0)
    patch(rows, "G", 2048, "mixed", preemptions=3, kv_peak=0.99)
    res = run(rows)
    it = lv(res)["interpretation"]
    att, cs = it["attachments"], lv(res)["conditions"]
    assert att["residual_rho_ratio_tg"] == pytest.approx(cs["TG"]["rho"] / cs["B0"]["rho"])
    assert set(att["f"]) == set(att["g"]) == set(att["a"]) == {"T", "G", "TG", "K3"}
    assert att["a"]["K3"] == pytest.approx(lv(res)["kv"]["i"]["a"])
    assert att["m_p_deltas"]["T"] == {"d_output_throughput_mixed": pytest.approx(200.0),
                                      "d_output_throughput_uniform": pytest.approx(-100.0),
                                      "d_long_class_p95_ttft_ms": pytest.approx(1000.0)}
    assert set(att["m_p_deltas"]) == {"T", "TG"}
    assert att["preemptions"]["G"] == {"uniform": 0, "mixed": 9} and att["kv_peak"]["G"]["mixed"] == 0.99
    assert {(n["cond"], n["category"]) for n in att["r_w_notes"]} == {("G", "W")}
    assert "閘門具保護作用" in att["r_w_notes"][0]["note"]
    alloc = it["allocator"]
    assert alloc["status"] == "未操弄，未檢驗" and alloc["waste_upper_bound"] == 0.031
    assert alloc["unexplained_break_share"] == pytest.approx(0.01)
    r0 = lv(run(SCENARIOS["R0"][0]()))["interpretation"]
    assert r0["attachments"]["f"] and r0["allocator"]["status"] == "未操弄，未檢驗"


# ---------------------------------------------------------------- modifiers


def test_incomplete_when_flipping_an_nt_to_m_changes_the_row():
    res = run(patch(dataset(), "T", 2048, "mixed", 3, timeout=True))
    it = lv(res)["interpretation"]
    assert it["row"] == "R7" and it["incomplete"] is True and it["incomplete_flips"] == ["T"]
    rows = patch(dataset({2048: {"T": (100.0, 180.0)}}), "TG", 2048, "mixed", 1, timeout=True)
    it = lv(run(rows))["interpretation"]  # TG = NT cannot move R1
    assert it["row"] == "R1" and it["incomplete"] is False and it["incomplete_flips"] == []
    rows = patch(dataset({2048: {"G": (130.0, 260.0)}}), "G", 2048, "mixed", 2, timeout=True)
    res = run(rows)  # G NT, but its uniform check already fails: an M would be qualified
    assert lv(res)["conditions"]["G"]["category"] == "NT" and lv(res)["conditions"]["G"]["qualified"] is True
    assert lv(res)["interpretation"]["incomplete"] is False


def test_kv_nt_flips_are_reported_separately():
    rows = h_c5_rows({2048: {"K3": (100.0, 600.0)}}, BLOCKED)
    res = run(patch(rows, "B0", 2048, "mixed", 1, decomp=decomp(BLOCKED, valid=False)))
    it = lv(res)["interpretation"]
    assert it["row"] == "R6" and it["incomplete"] is False and it["kv_nt_flips"] == ["ii"]


# ---------------------------------------------------------------- A12 cross-L


def test_cross_l_agreement_and_disagreement():
    res = run(dataset({2048: {"T": (100.0, 180.0)}, 8192: {"T": (100.0, 180.0)}}))
    x = res["cross_l"]
    assert x["agree"] is True and x["row"] == "R1" and x["rows"] == {"2048": "R1", "8192": "R1"}
    assert x["conclusion"] == SCENARIOS["R1"][1] and x["not_testable"] == []
    res = run(dataset({2048: {"T": (100.0, 180.0)}, 8192: {"G": (100.0, 180.0)}}))
    x = res["cross_l"]
    assert x["agree"] is False and x["row"] is None and x["conclusion"] == "成因依 L 而異"
    assert x["rows"] == {"2048": "R1", "8192": "R2"} and x["not_testable"] == []


CROSS_L_NT = "有 L 的 H-C0 不可檢定，跨 L 不可檢定，不寫成 H-B4 的一般結論"


def test_cross_l_failed_h_c0_at_both_levels_agrees_on_r0():
    x = run(dataset({2048: {"B0": (100.0, 110.0)}, 8192: {"B0": (100.0, 110.0)}}))["cross_l"]
    assert x["agree"] is True and x["row"] == "R0" and x["conclusion"] == SCENARIOS["R0"][1]
    assert x["not_testable"] == []


@pytest.mark.parametrize("other", [
    None,                                   # both levels not testable
    {8192: {"B0": (100.0, 110.0)}},         # the other level is a real R0 (H-C0 failed)
    {8192: {"T": (100.0, 180.0)}},          # the other level is R1
])
def test_cross_l_never_counts_a_not_testable_h_c0(other):
    rows = patch(dataset(other), "B0", 2048, "mixed", 1, ok=False)
    if other is None:
        patch(rows, "B0", 8192, "mixed", 1, ok=False)
    x = run(rows)["cross_l"]
    assert x["agree"] is None and x["row"] is None
    assert x["conclusion"] == CROSS_L_NT and "沒有重現" not in x["conclusion"]
    assert x["not_testable"] == (["2048", "8192"] if other is None else ["2048"])


def test_cross_l_r4_general_conclusion_does_not_claim_h_c3_unless_both_levels_support_it():
    rows = patch(dataset({2048: {"TG": (100.0, 150.0)}, 8192: {"TG": (100.0, 150.0)}}), "T", 2048, "mixed",
                 ok=False)
    x = run(rows)["cross_l"]
    assert x["agree"] is True and x["row"] == "R4"
    assert x["conclusion"] == R4_H_C3_NT_CONCLUSION and "H-C3 成立" not in x["conclusion"]
    x = run(dataset({2048: {"TG": (100.0, 150.0)}, 8192: {"TG": (100.0, 150.0)}}))["cross_l"]
    assert x["row"] == "R4" and x["conclusion"] == SCENARIOS["R4"][1]


def test_levels_are_judged_separately():
    res = run(dataset({8192: {"T": (100.0, 180.0)}}))
    assert row_of(res, 2048) == "R7" and row_of(res, 8192) == "R1"


# ---------------------------------------------------------------- A13 descriptive never decides


def verdicts(res):
    out = {"cross_l": res["cross_l"]}
    for L, level in res["levels"].items():
        out[L] = {"h_c0": level["h_c0"],
                  "cats": {c: {k: level["conditions"][c].get(k) for k in ("category", "qualified", "effective")}
                           for c in aa.CONDITIONS},
                  "kv": level["kv"], "hyp": level["hypotheses"],
                  "interp": {k: v for k, v in level["interpretation"].items() if k not in ("attachments", "allocator")}}
    return out


DESCRIPTIVE = ("long_class_p95_ttft_ms", "output_throughput", "total_token_throughput", "request_throughput",
               "p95_tpot_ms", "p95_ttft_ms", "duration", "kv_peak")
CLIENT = ("short_p50_ttft_ms", "short_p99_ttft_ms", "long_p50_ttft_ms", "long_p95_ttft_ms", "short_p95_tpot_ms")


@pytest.mark.parametrize("make", [SCENARIOS[k][0] for k in sorted(SCENARIOS)] + [
    lambda: patch(dataset({2048: {"G": (100.0, 800.0), "T": (100.0, 300.0)}}), "G", 2048, "mixed", preemptions=2)])
def test_verdicts_never_depend_on_descriptive_fields(make):
    rows = make()
    base = verdicts(run(rows))
    rng = random.Random(20261008)
    for row in rows:
        for k in DESCRIPTIVE:
            row[k] = rng.uniform(0, 1e5)
        row["client"] = {k: rng.uniform(0, 1e5) for k in CLIENT}
        row["one_hz"] = {"kv_full_while_waiting": rng.random(), "gate_zone": rng.random(), "n_samples": 7}
        row["scrape_delta"] = {QUEUE_HIST: hist(rng.uniform(0, 60)), PREFILL_HIST: hist(rng.uniform(0, 60)),
                               "errors": []}
        if row.get("decomp"):
            row["decomp"]["tail"]["O"] = rng.uniform(0, 1e4)
            row["decomp"]["tail"]["P"] = rng.uniform(0, 1e4)
            row["decomp"]["tail95"] = {p: rng.uniform(0, 1e4) for p in "OQPF"}
            row["decomp"]["residual_bm_share"] = rng.random()
            row["decomp"]["unexplained_break_share"] = rng.random()
    assert verdicts(run(rows)) == base


def test_g_category_is_shown_beside_f_but_f_decides():
    # T: U halves, M drops to 160 -> rho = ln 3.2, f = 0.16 (N); g = 1 - 110/300 = 0.63.
    res = run(dataset({2048: {"T": (50.0, 160.0)}}))
    t = lv(res)["conditions"]["T"]
    assert t["category"] == "N" and t["g"] == pytest.approx(1 - 110 / 300)
    assert t["g_category"] == "M" and t["g_differs"] is True
    assert row_of(res) == "R7"


def test_descriptive_quantities():
    rows = dataset({2048: {"T": (100.0, 180.0), "G": (100.0, 300.0), "TG": (100.0, 150.0)}})
    res = run(rows)
    d, cs = lv(res)["descriptive"], lv(res)["conditions"]
    inter = d["interaction"]
    assert inter["I"] == pytest.approx((cs["B0"]["rho"] - cs["T"]["rho"]) - (cs["G"]["rho"] - cs["TG"]["rho"]))
    assert inter["se"] == pytest.approx(math.sqrt(sum(cs[c]["rho_se"] ** 2 for c in ("B0", "T", "G", "TG"))))
    cell = d["cells"]["B0-mixed"]
    assert cell["n_valid"] == 3 and cell["long_class_p95_ttft_ms"] == 5000.0 and cell["decomp_valid"] == 3
    assert cell["tail"]["Q"]["mean"] == 100.0 and cell["tail"]["Q"]["se"] == 0.0
    tail = d["tail_delta"]["B0"]
    assert tail["Q"]["delta"] == pytest.approx(90.0) and tail["O"]["delta"] == pytest.approx(0.0)
    assert d["core_ratio"]["uniform"] == pytest.approx(100.0 / 80.0) and d["core_ratio"]["mixed"] is None


def st(mean, se, n=3):
    return {"mean": pytest.approx(mean), "se": None if se is None else pytest.approx(se), "n": n}


def test_a13_a8_descriptive_quantities_are_aggregated_per_cell():
    """024 A13 (histogram queue / prefill means, 1 Hz shares), A8 (the >= P95 tail subset) and the 量測 client
    descriptives (short P50 / P99, long P50, short P95 TPOT): mean / SE over the cell's valid restarts."""
    rows = dataset()
    for r, x in ((1, 1.0), (2, 2.0), (3, 3.0)):
        patch(rows, "B0", 2048, "mixed", r,
              scrape_delta={QUEUE_HIST: hist(x), PREFILL_HIST: hist(10 * x), "errors": []},
              one_hz={"kv_full_while_waiting": x / 10, "gate_zone": x / 5, "n_samples": 90},
              client={"short_p50_ttft_ms": 100 * x, "short_p99_ttft_ms": 1000 * x, "long_p50_ttft_ms": 4000 * x,
                      "long_p95_ttft_ms": 5000.0, "short_p95_tpot_ms": 20 * x},
              decomp={**decomp(Q=100.0), "tail95": {"O": x, "Q": 100 * x, "P": 10 * x, "F": 0.0}})
    d = lv(run(rows))["descriptive"]
    se = 1 / math.sqrt(3)  # stdev(1, 2, 3) = 1
    cell = d["cells"]["B0-mixed"]
    assert cell["hist"] == {"queue_time_mean_s": st(2.0, se), "prefill_time_mean_s": st(20.0, 10 * se)}
    assert cell["one_hz"] == {"kv_full_while_waiting": st(0.2, se / 10), "gate_zone": st(0.4, se / 5)}
    assert cell["client"] == {"short_p50_ttft_ms": st(200.0, 100 * se), "short_p99_ttft_ms": st(2000.0, 1000 * se),
                              "long_p50_ttft_ms": st(8000.0, 4000 * se), "short_p95_tpot_ms": st(40.0, 20 * se)}
    assert cell["tail95"] == {"O": st(2.0, se), "Q": st(200.0, 100 * se), "P": st(20.0, 10 * se)}
    u = d["cells"]["B0-uniform"]  # the fixture's defaults
    assert u["hist"]["queue_time_mean_s"] == st(0.05, 0.0) and u["one_hz"]["gate_zone"] == st(0.2, 0.0)
    assert u["client"]["long_p50_ttft_ms"] is None  # no long class in uniform
    assert u["tail95"]["Q"] == st(15.0, 0.0)
    t95 = d["tail95_delta"]["B0"]
    assert t95["Q"] == {"delta": pytest.approx(185.0), "se": pytest.approx(100 * se)}
    assert t95["O"]["delta"] == pytest.approx(0.5)
    assert set(d["tail95_delta"]) == set(aa.CONDITIONS)


def test_descriptive_aggregates_use_valid_restarts_only_and_never_fill_gaps():
    rows = dataset()
    for r, x in ((1, 1.0), (2, 2.0), (3, 30.0)):
        patch(rows, "K3", 2048, "mixed", r, scrape_delta={QUEUE_HIST: hist(x), PREFILL_HIST: hist(x), "errors": []})
    patch(rows, "K3", 2048, "mixed", 3, ok=False)  # the invalid run's values are never used
    patch(rows, "K3", 2048, "mixed", 1, scrape_delta={QUEUE_HIST: hist(1.0), PREFILL_HIST: None, "errors": []},
          one_hz={"kv_full_while_waiting": None, "gate_zone": None, "n_samples": 0})
    patch(rows, "G", 2048, "uniform", 2, client=None, scrape_delta=None, one_hz=None)
    cells = lv(run(rows))["descriptive"]["cells"]
    k3 = cells["K3-mixed"]
    assert k3["n_valid"] == 2
    assert k3["hist"]["queue_time_mean_s"] == st(1.5, 0.5, n=2)  # stdev(1, 2) / sqrt 2
    assert k3["hist"]["prefill_time_mean_s"] is None  # a histogram missing from one valid run: no partial mean
    assert k3["one_hz"] == {"kv_full_while_waiting": None, "gate_zone": None}
    assert k3["client"]["short_p95_tpot_ms"] == st(30.0, 0.0, n=2)
    assert k3["tail95"] is None and k3["tail"] is None  # A8: 3 valid decompositions per cell
    g = cells["G-uniform"]
    assert g["client"] == dict.fromkeys(("short_p50_ttft_ms", "short_p99_ttft_ms", "long_p50_ttft_ms",
                                         "short_p95_tpot_ms"))
    assert g["hist"] == {"queue_time_mean_s": None, "prefill_time_mean_s": None}
    assert g["one_hz"] == {"kv_full_while_waiting": None, "gate_zone": None}
    patch(rows, "K3", 2048, "mixed", 2, timeout=True)
    k3 = lv(run(rows))["descriptive"]["cells"]["K3-mixed"]
    assert k3["n_valid"] == 1 and k3["hist"]["queue_time_mean_s"] == st(1.0, None, n=1)  # no SE from one run
    empty = lv(run(patch(rows, "K3", 2048, "mixed", 1, skipped=True)))["descriptive"]["cells"]["K3-mixed"]
    assert empty["n_valid"] == 0 and empty["hist"]["queue_time_mean_s"] is None and empty["tail95"] is None


def test_descriptive_keys_are_the_ones_run_adv_records():
    import adv_decomp as ad
    import run_adv as ra
    assert set(aa.CLIENT_KEYS) <= set(ra.client_stats({}, L=2048))
    assert set(aa.HIST_MEANS.values()) <= set(ra.HISTOGRAMS)
    assert set(aa.HIST_MEANS.values()) <= set(ra.scrape_deltas("", ""))
    assert set(aa.ONE_HZ_KEYS) <= set(ad.one_hz_shares([], nb=248, L=2048))


def test_core_mixed_ratio_when_given():
    core_mixed = {"rows": [{"L": L, "mix": "mixed", "restart": r, "C": C, "ok": True, "short_class_p95_ttft_ms": 320.0}
                           for L, C in LEVELS.items() for r in (1, 2, 3)]}
    res = run(dataset(), core_mixed=core_mixed)
    assert lv(res)["descriptive"]["core_ratio"]["mixed"] == pytest.approx(400.0 / 320.0)


# ---------------------------------------------------------------- A12 run table / inputs


def test_runs_table_lists_every_run_with_its_status():
    rows = patch(dataset(), "T", 2048, "mixed", 1, timeout=True, ok=False, reasons=["timeout"])
    patch(rows, "T", 2048, "mixed", 2, skipped=True, ok=False, short_class_p95_ttft_ms=None)
    res = run(rows)
    assert len(res["runs"]) == 60
    bad = {(r["cond"], r["L"], r["mix"], r["restart"]): r for r in res["runs"] if not r["valid"]}
    assert set(bad) == {("T", 2048, "mixed", 1), ("T", 2048, "mixed", 2)}
    assert bad[("T", 2048, "mixed", 1)]["timeout"] is True and "timeout" in bad[("T", 2048, "mixed", 1)]["invalid"]
    assert bad[("T", 2048, "mixed", 2)]["skipped"] is True


@pytest.mark.parametrize("change, match", [
    (lambda a, u: a["session"].update(formal=False), "not formal"),
    (lambda a, u: a["session"].update(env_valid=None), "env-invalid"),
    (lambda a, u: a["plan"].update(c={"2048": 22, "8192": 8}), "plan C"),
    (lambda a, u: u.update(uniform_core({2048: 16, 8192: 8})), "C_knee"),
    (lambda a, u: a["rows"][0].update(C=22), "C ="),
])
def test_refuses_invalid_session_and_c_mismatch(change, match):
    a, u = adv(dataset()), uniform_core()
    change(a, u)
    with pytest.raises(aa.AnalysisError, match=match):
        aa.analyze(a, u)


def test_output_is_json_serialisable():
    import json
    res = run(R5_ROWS())
    assert json.loads(json.dumps(res)) == res
    assert res["schema_version"] == 1 and res["c"] == {"2048": 25, "8192": 8}
