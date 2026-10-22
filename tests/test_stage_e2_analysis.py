"""Tests for benchmarks/stage_e2_analysis.py (decision 041 v3 §2.2, 2.6, 2.7) on synthetic session rows: the
in-session contrasts, E / I, the H-C0 gate, beta_ok, KV_ok, H-E2k, H-E2e and the interpretation rows."""

import tomllib
from pathlib import Path

import pytest

import stage_e2_analysis as ea

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "configs" / "stage_e2.toml").read_text())
SPREAD = (0.99, 1.0, 1.01)


def rows_for(cells, *, beta=None, gate_zone=None, missing=(), preempt=None):
    """cells: {cell: (U, M)} (shared cells: (Y,)); every restart valid with a 1 % spread."""
    beta, gate_zone, preempt = beta or {}, gate_zone or {}, preempt or {}
    out = []
    for name, values in cells.items():
        mixes = ("shared",) if name.startswith("PC") else ("uniform", "mixed")
        for mix, y in zip(mixes, values):
            for r, k in zip((1, 2, 3), SPREAD):
                if (name, mix, r) in missing:
                    continue
                out.append({"cell": name, "mix": mix, "restart": r, "valid": True, "short_class_p95_ttft_ms": y * k,
                            "beta": beta.get(name, 0.5), "preemptions": preempt.get(name, 0),
                            "one_hz": {"gate_zone": gate_zone.get(name, 0.1) * k}})
    return out


BASE = {"B0@25": (1000, 5000), "T@25": (1050, 4900), "T256@25": (1100, 4800),
        "B0@8": (970, 4000), "T@8": (1000, 1500), "B0@8k": (975, 4500), "T@8k": (1000, 4400),
        "PC-off": (1000,), "PC-on": (400,)}


def analyze(cells=None, **kw):
    return ea.analyze(rows_for({**BASE, **(cells or {})}, **kw), CFG)


def test_t_failing_only_under_kv_pressure_reads_r3():
    a = analyze(beta={"B0@25": 0.9, "B0@8": 0.3}, gate_zone={"B0@8k": 0.6, "B0@8": 0.1}, preempt={"T@8k": 40})
    assert a["versus"]["T@25"]["verdict"] == "I" and a["versus"]["T@8"]["verdict"] == "E"
    assert a["versus"]["T@8k"]["verdict"] == "I" and a["kv_ok"] is True and a["beta_ok"] is True
    assert a["row"]["row"] == "R3" and a["row"]["path"] == "over-commit"
    h = a["hypotheses"]
    assert (h["H-E2-R"]["verdict"], h["H-E2a"]["verdict"], h["H-E2b"]["verdict"]) == ("supported",) * 3
    assert h["H-E2c"]["verdict"] == "not supported" and a["row"]["t256_note"].startswith("no evidence")
    assert h["H-E2e"]["verdict"] == "supported" and h["H-E2e"]["ratio"] == pytest.approx(0.4)
    assert a["interaction_I_KV"]["value"] > 0


def test_t_failing_at_c8_too_reads_r2():
    a = analyze({"T@8": (1000, 3950)}, beta={"B0@25": 0.9, "B0@8": 0.3})
    assert a["versus"]["T@8"]["verdict"] == "I" and a["row"]["row"] == "R2" and "neither S nor V" in a["row"]["reading"]
    a = analyze({"T@8": (1000, 3950)}, beta={"B0@25": 0.6, "B0@8": 0.5})
    assert a["row"]["row"] == "R2" and "not enough" in a["row"]["reading"]


def test_mostly_uniform_slower_is_not_effective():
    a = analyze({"T@8": (1500, 3700)})                                   # rho falls because U rises, M barely moves
    v = a["versus"]["T@8"]
    assert v["category"] in ("M", "P") and v["verdict"] == "I" and v["mostly_uniform_slower"] is True


def test_premise_rows_come_first():
    assert analyze({"B0@25": (1000, 1010)})["row"]["row"] == "R0"
    assert analyze({"T@25": (1000, 1300)})["row"]["row"] == "R0b"
    a = ea.analyze(rows_for(BASE, missing=(("T@25", "mixed", 3),)), CFG)
    assert a["row"]["row"] == "R0c" and a["hypotheses"]["H-E2-R"]["verdict"] == "NT"
    assert analyze({"B0@8": (970, 980)})["row"]["row"] == "R1"


def test_r4_and_r5():
    a = analyze({"T@8k": (1000, 1600)}, beta={"B0@25": 0.9, "B0@8": 0.3}, gate_zone={"B0@8k": 0.6, "B0@8": 0.1})
    assert a["row"]["row"] == "R4" and "consistent with S" in a["row"]["reading"]
    a = analyze(gate_zone={"B0@8k": 0.1, "B0@8": 0.1})
    assert a["kv_ok"] is False and a["row"]["row"] == "R5"


def test_t256_working_at_c25_is_noted():
    a = analyze({"T256@25": (1000, 1800)}, gate_zone={"B0@8k": 0.6, "B0@8": 0.1})
    assert a["hypotheses"]["H-E2c"]["verdict"] == "supported" and a["row"]["t256_note"].startswith("a smaller")


def test_the_kv_dose_follows_a7_i():
    a = analyze({"B0@8k": (975, 6000)})
    hk = a["hypotheses"]["H-E2k"]
    assert hk["a"] > 0.2 and hk["u_rho"] and hk["m_up"] and hk["specific"] and hk["verdict"] == "supported"
    a = analyze({"B0@8k": (1300, 6000)})                                 # uniform moved too: not specific
    assert a["hypotheses"]["H-E2k"]["verdict"] == "not supported" and a["kv_ok"] is False


def test_prefix_caching_that_barely_helps_is_not_supported():
    a = analyze({"PC-on": (900,)})
    assert a["hypotheses"]["H-E2e"]["verdict"] == "not supported"
