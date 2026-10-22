"""Tests for quality/stage_q_analysis.py (decision 040): the Stage Q hypotheses from synthetic rows and Phase 1's
committed quality rows."""

from pathlib import Path

import pytest

import stage_q_analysis as qa2

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = qa2.load_rows(ROOT / "results" / "quality" / "formal-20261010-quality")
BASE_KLD = {("4B", "Q8_0"): 0.00203, ("4B", "Q6_K"): 0.00546, ("4B", "Q4_K_M"): 0.0458,
            ("9B", "Q8_0"): 0.00202, ("9B", "Q6_K"): 0.00726, ("9B", "Q4_K_M"): 0.0574}


def kld_row(size, q, mean, se=0.0001, ctx=None):
    return {"kind": "kld", "outcome": "ok", "size": size, "point": q, "ctx": ctx,
            "metrics": {"kld_mean": mean, "kld_mean_se": se}}


def ppl_row(size, p, ppl):
    return {"kind": "ppl", "outcome": "ok", "size": size, "point": p, "ctx": 512, "metrics": {"ppl": ppl}}


def test_phase1_rows_give_the_cuda_references():
    assert qa2.kld(PHASE1)[("9B", 512, "Q4_K_M")][0] == pytest.approx(0.0574, abs=1e-4)
    ppl = qa2.cuda_ppl(PHASE1)
    assert ppl[("9B", "BF16")] == pytest.approx(8.186, abs=1e-3) and ppl[("4B", "Q4_K_M")] == pytest.approx(10.392, abs=1e-3)


def metal(scale=1.0, ppl_scale=1.0):
    rows = [kld_row(s, q, v * scale, ctx=512) for (s, q), v in BASE_KLD.items()]
    ppl = qa2.cuda_ppl(PHASE1)
    rows += [ppl_row(s, p, ppl[(s, p)] * ppl_scale) for s in ("4B", "9B") for p in ("BF16", "Q8_0", "Q6_K", "Q4_K_M")]
    return rows


def test_h_q1_supported_when_metal_matches_cuda():
    h = qa2.h_q1(metal(), [kld_row(s, q, v, ctx=512) for (s, q), v in BASE_KLD.items()], PHASE1)
    assert [h[k]["verdict"] for k in ("H-Q1a", "H-Q1b", "H-Q1c")] == ["supported"] * 3


def test_h_q1_bounds_fail_outside_and_are_nt_when_missing():
    cuda = [kld_row(s, q, v, ctx=512) for (s, q), v in BASE_KLD.items()]
    h = qa2.h_q1(metal(scale=1.3, ppl_scale=1.01), cuda, PHASE1)
    assert h["H-Q1b"]["verdict"] == "not supported" and h["H-Q1c"]["verdict"] == "not supported"
    h = qa2.h_q1(metal(), cuda[:-1], PHASE1)
    assert h["H-Q1b"]["verdict"] == "NT" and h["H-Q1b"]["ratios"]["9B Q4_K_M"] is None


def test_the_ordering_family_is_holm_and_three_valued():
    good = {("4B", 512, q): (v, 0.0001) for (s, q), v in BASE_KLD.items() if s == "4B"}
    assert qa2.ordering(good, [("4B", 512)])["verdict"] == "supported"
    swapped = {**good, ("4B", 512, "Q6_K"): (0.0010, 0.0001)}       # Q6_K below Q8_0
    assert qa2.ordering(swapped, [("4B", 512)])["verdict"] == "not supported"
    noisy = {k: (v[0], 1.0) for k, v in good.items()}
    assert qa2.ordering(noisy, [("4B", 512)])["verdict"] == "inconclusive"
    assert qa2.ordering({}, [("4B", 512)])["verdict"] == "NT"


def test_h_q2_compares_each_context_with_phase1_at_512():
    rows = [kld_row(s, q, v * 1.4, ctx=c) for c in (2048, 8192) for (s, q), v in BASE_KLD.items()]
    h = qa2.h_q2(rows, PHASE1)
    assert h["H-Q2a"]["verdict"] == "supported" and len(h["H-Q2a"]["tests"]) == 8
    assert h["H-Q2b"]["verdict"] == "supported"
    rows[0] = kld_row("4B", "Q8_0", BASE_KLD[("4B", "Q8_0")] * 2.5, ctx=2048)
    assert qa2.h_q2(rows, PHASE1)["H-Q2b"]["verdict"] == "not supported"
