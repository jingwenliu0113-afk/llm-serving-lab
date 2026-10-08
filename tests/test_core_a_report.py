"""Tests for benchmarks/core_a_report.py: one analysis.json per platform session."""

import copy
import tomllib
from pathlib import Path

import pytest

import core_a_report as cr
from test_core_a_analysis import EMBD, FILE, grid, ideal_decode, ideal_ttft

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        p = tomllib.load(fh)
    p = copy.deepcopy(p)
    for m in p["core_a"]["models"]:
        for q, f in m["files"].items():
            f["size_bytes"] = int(FILE[(m["size"], q)])
    return p


def test_mac_report_has_all_hypotheses(pins):
    rows = grid(ideal_decode, ideal_ttft)
    rep = cr.report(rows, platform="macos-metal", pins=pins, embd_bytes=EMBD, tied={"4B": True, "9B": False})
    assert rep["platform_params"] == {"v_total_mib": 38338, "b_mib": 0}
    assert set(rep["hypotheses"]) == {"H-A1", "H-A2", "H-A3", "H-A4a", "H-A4b"}
    assert rep["hypotheses"]["H-A1"]["verdict"] == "not refuted"
    assert all(u["predicted"] == "fit" for u in rep["hypotheses"]["H-A1"]["units"])
    assert rep["spill"] == []
    assert rep["hypotheses"]["H-A2"]["verdict"] == "not refuted"


def test_rtx_report_needs_stage_b_params(pins):
    rows = grid(ideal_decode, ideal_ttft)
    with pytest.raises(cr.ReportError, match="Stage B"):
        cr.report(rows, platform="wsl2-cuda", pins=pins, embd_bytes=EMBD, tied={"4B": True, "9B": False})
    rep = cr.report(rows, platform="wsl2-cuda", pins=pins, embd_bytes=EMBD, tied={"4B": True, "9B": False},
                    platform_params={"v_total_mib": 16303, "b_mib": 1000, "c_ctx_mib": 400})
    assert rep["hypotheses"]["H-A1"]["verdict"] == "not refuted"


def test_tied_size_never_subtracts_token_embd(pins):
    rows = grid(ideal_decode, ideal_ttft)
    embd = dict(EMBD)
    embd[("4B", "Q8_0")] = 6.75e8  # the tied 4B still has a token_embd tensor
    rep = cr.report(rows, platform="macos-metal", pins=pins, embd_bytes=embd, tied={"4B": True, "9B": False})
    assert rep["subtracted_embd_bytes"]["4B-Q8_0"] == 0
