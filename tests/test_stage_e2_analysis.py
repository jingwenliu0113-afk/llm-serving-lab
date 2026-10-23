"""Tests for benchmarks/stage_e2_analysis.py (decision 041 v3 §2.2, 2.6, 2.7) on synthetic session rows: the
in-session contrasts, E / I, the H-C0 gate, beta_ok, KV_ok, H-E2k, H-E2-0, H-E2e, the interpretation rows with their
incomplete flag, R3's paths, the T256 caveat, the descriptive block and the formal-session checks."""

import json
import tomllib
from pathlib import Path

import pytest

import stage_e2_analysis as ea
import stage_e2_plan as e2p
import test_run_stage_e2 as tre2

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "configs" / "stage_e2.toml").read_text())
SPREAD = (0.99, 1.0, 1.01)


def rows_for(cells, *, beta=None, gate_zone=None, missing=(), preempt=None, recompute=None, shares=None):
    """cells: {cell: (U, M)} (shared cells: (Y,)); every restart valid with a 1 % spread; recompute: R per cell
    (default 0); shares: B0 cells' HOL-long + KV-short (split evenly)."""
    beta, gate_zone, preempt = beta or {}, gate_zone or {}, preempt or {}
    recompute, shares = recompute or {}, shares or {}
    out = []
    for name, values in cells.items():
        mixes = ("shared",) if name.startswith("PC") else ("uniform", "mixed")
        for mix, y in zip(mixes, values):
            for r, k in zip((1, 2, 3), SPREAD):
                if (name, mix, r) in missing:
                    continue
                h = shares.get(name, 0.2) / 2
                out.append({"cell": name, "mix": mix, "restart": r, "valid": True, "short_class_p95_ttft_ms": y * k,
                            "beta": beta.get(name, 0.5), "preemptions": preempt.get(name, 0),
                            "recompute_tokens": recompute.get(name, 0), "kv_peak": 0.5, "budget_full_share": 0.4,
                            "lq_short": 2.0, "total_token_throughput": 5e3,
                            "decomp": {"q_shares": {"budget-long": 0.5, "HOL-long": h, "KV-short": h},
                                       "unexplained_break_share": 0.01},
                            "one_hz": {"gate_zone": gate_zone.get(name, 0.1) * k, "kv_full_while_waiting": 0.0}})
    return out


BASE = {"B0@25": (1000, 5000), "T@25": (1050, 4900), "T256@25": (1100, 4800),
        "B0@8": (970, 4000), "T@8": (1000, 1500), "B0@8k": (975, 4500), "T@8k": (1000, 4400),
        "PC-off": (1000,), "PC-on": (400,)}


def analyze(cells=None, **kw):
    return ea.analyze(rows_for({**BASE, **(cells or {})}, **kw), CFG)


R3_KW = dict(beta={"B0@25": 0.9, "B0@8": 0.3}, gate_zone={"B0@8k": 0.6, "B0@8": 0.1})


def test_t_failing_only_under_kv_pressure_reads_r3():
    a = analyze(preempt={"T@8k": 40}, recompute={"T@8k": 5000}, **R3_KW)
    assert a["versus"]["T@25"]["verdict"] == "I" and a["versus"]["T@8"]["verdict"] == "E"
    assert a["versus"]["T@8k"]["verdict"] == "I" and a["kv_ok"] is True and a["beta_ok"] is True
    assert a["row"]["row"] == "R3" and a["row"]["path"] == "over-commit"
    h = a["hypotheses"]
    assert (h["H-E2-R"]["verdict"], h["H-E2a"]["verdict"], h["H-E2b"]["verdict"]) == ("supported",) * 3
    assert h["H-E2c"]["verdict"] == "not supported" and a["row"]["t256_note"].startswith("no evidence")
    assert h["H-E2e"]["verdict"] == "supported" and h["H-E2e"]["ratio"] == pytest.approx(0.4)
    assert a["interaction_I_KV"]["value"] > 0 and a["row"]["incomplete"] is False and "t512_only" not in a["row"]
    assert h["H-E2-0"] == {"verdict": "supported", "by_reference": {"B0@25": "passes", "B0@8": "passes",
                                                                     "B0@8k": "passes"}}


@pytest.mark.parametrize("kw, path", [
    (dict(preempt={"T@8k": 40}, recompute={"T@8k": 0}), "unresolved"),             # preemptions but no recompute
    (dict(shares={"B0@8k": 0.7}), "HOL"), (dict(shares={"B0@8k": 0.3}), "unresolved"),
    (dict(recompute={"T@8k": None}), "NT")])
def test_r3s_paths_follow_041(kw, path):
    a = analyze(**R3_KW, **kw)
    assert a["row"]["row"] == "R3" and a["row"]["path"] == path


def test_the_descriptive_block_and_the_allocator_bound():
    a = analyze(preempt={"T@8k": 40}, recompute={"T@8k": 5000}, **R3_KW)
    d = a["descriptive"]["T@8k/mixed"]
    assert d["n_valid"] == 3 and d["recompute_tokens"]["mean"] == 5000 and d["preemptions"]["mean"] == 40
    assert d["budget_full_share"]["mean"] == 0.4 and d["lq_short"]["mean"] == 2.0 and d["q_share_hol_long"]["mean"] == 0.1
    assert d["long_class_p95_ttft_ms"] is None and set(a["descriptive"]) >= {"PC-on/shared", "B0@8k/uniform"}
    assert a["allocator"] == {"status": "未操弄，未檢驗", "waste_upper_bound": 0.031, "source": "024 A13"}


def test_t_failing_at_c8_too_reads_r2():
    a = analyze({"T@8": (1000, 3950)}, beta={"B0@25": 0.9, "B0@8": 0.3})
    assert a["versus"]["T@8"]["verdict"] == "I" and a["row"]["row"] == "R2" and "neither S nor V" in a["row"]["reading"]
    a = analyze({"T@8": (1000, 3950)}, beta={"B0@25": 0.6, "B0@8": 0.5})
    assert a["row"]["row"] == "R2" and "not enough" in a["row"]["reading"] and a["row"]["incomplete"] is False
    a = analyze({"T@8": (1000, 3950)}, beta={"B0@25": 0.9, "B0@8": None})              # the gauge is missing
    assert a["beta_ok"] is None and "beta_ok NT" in a["row"]["reading"] and a["row"]["incomplete"] is True


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


def test_t256_working_at_c25_is_noted_and_qualifies_r3_and_r4():
    a = analyze({"T256@25": (1000, 1800), "T@8k": (1000, 1600)}, gate_zone={"B0@8k": 0.6, "B0@8": 0.1})
    assert a["hypotheses"]["H-E2c"]["verdict"] == "supported" and a["row"]["t256_note"].startswith("a smaller")
    assert a["row"]["row"] == "R4" and a["row"]["t512_only"] == ea.T512_ONLY
    a = analyze({"T256@25": (1000, 1800)}, **R3_KW)
    assert a["row"]["row"] == "R3" and a["row"]["t512_only"] == ea.T512_ONLY


def test_incomplete_only_when_an_nt_could_change_the_row():
    nt = dict(missing=(("T@8k", "mixed", 3),))                                       # T@8k NT
    assert analyze({"B0@25": (1000, 1010)})["row"] == {**analyze({"B0@25": (1000, 1010)})["row"], "row": "R0",
                                                        "incomplete": False}
    a = analyze({"B0@8": (970, 980)}, **nt)
    assert (a["row"]["row"], a["row"]["incomplete"]) == ("R1", False)
    a = analyze({"T@8": (1000, 3950)}, **nt)
    assert (a["row"]["row"], a["row"]["incomplete"]) == ("R2", False)                # T@8k cannot move R2
    a = analyze(**R3_KW, missing=(("T256@25", "mixed", 3),))
    assert (a["row"]["row"], a["row"]["incomplete"], a["row"]["t256_note"]) == ("R3", False, "incomplete")
    a = analyze(**nt, gate_zone={"B0@8k": 0.6, "B0@8": 0.1})
    assert (a["row"]["row"], a["row"]["incomplete"]) == ("R6", True)
    a = analyze(missing=(("B0@8k", "mixed", 3),))                                    # kv_ok and B0@8k's gate NT
    assert a["kv_ok"] is None and a["row"]["incomplete"] is True
    a = analyze({"B0@8k": (975, 980)})                                               # R5 through B0@8k's gate
    assert (a["row"]["row"], a["row"]["incomplete"]) == ("R5", False)
    a = ea.analyze(rows_for(BASE, missing=(("T@25", "mixed", 3),)), CFG)
    assert (a["row"]["row"], a["row"]["incomplete"]) == ("R0c", True)


def test_h_e2k_is_nt_without_a_c8_penalty_and_h_e2_0_aggregates_the_gates():
    a = analyze({"B0@8": (1000, 990), "B0@8k": (1000, 1400)})
    assert a["hypotheses"]["H-E2k"]["verdict"] == "NT" and "undefined" in a["hypotheses"]["H-E2k"]["why"]
    assert a["hypotheses"]["H-E2-0"]["verdict"] == "not supported"
    a = ea.analyze(rows_for(BASE, missing=(("B0@8", "mixed", 3),)), CFG)
    assert a["hypotheses"]["H-E2-0"]["verdict"] == "NT"


def test_the_kv_dose_follows_a7_i():
    a = analyze({"B0@8k": (975, 6000)})
    hk = a["hypotheses"]["H-E2k"]
    assert hk["a"] > 0.2 and hk["u_rho"] and hk["m_up"] and hk["specific"] and hk["verdict"] == "supported"
    a = analyze({"B0@8k": (1300, 6000)})                                 # uniform moved too: not specific
    assert a["hypotheses"]["H-E2k"]["verdict"] == "not supported" and a["kv_ok"] is False


def test_prefix_caching_that_barely_helps_is_not_supported():
    a = analyze({"PC-on": (900,)})
    assert a["hypotheses"]["H-E2e"]["verdict"] == "not supported"


def test_only_a_finished_first_valid_formal_session_is_analyzed(tmp_path):
    root, ds, pins, cfg, plan = tre2.formal_setup(tmp_path)
    tre2.session(root, ds, pins, cfg, plan, mode="formal", sid="f1")
    base = root / "results" / "stage_e"
    kw = dict(pins=pins, cfg=cfg, root=root)
    assert ea.session_problems(base / "f1", **kw) == []
    both = {**cfg, "freeze": {"e2": True, "smoke_e2": "smoke-e2"}}                   # the [freeze] never matters
    assert ea.session_problems(base / "f1", pins=pins, cfg=both, root=root) == []
    assert any("not a finished formal" in p for p in ea.session_problems(base / "smoke-e2", **kw))
    s = json.loads((base / "f1" / "session.json").read_text())
    for over, why in (({"stopped": "3 boots"}, "void"), ({"env_valid": False}, "void"),
                      ({"harness_sha256": "0" * 64}, "harness_sha256"), ({"stage_e2_sha256": "0" * 64}, "content")):
        (base / "f1" / "session.json").write_text(json.dumps({**s, **over}))
        assert any(why in p for p in ea.session_problems(base / "f1", **kw)), over
    (base / "f1" / "session.json").write_text(json.dumps(s))
    for name, started in (("old1", "2000"), ("old2", "2001")):                      # two earlier attempts
        (base / name).mkdir()
        (base / name / "session.json").write_text(json.dumps({"stage": "E2", "mode": "formal", "status": "aborted",
                                                               "started_at": started}))
    assert any("earlier formal E2 attempts" in p for p in ea.session_problems(base / "f1", **kw))
    (base / "old2" / "session.json").unlink()
    assert ea.session_problems(base / "f1", **kw) == []                             # one void attempt is fine
    (base / "x").mkdir()
    import shutil
    shutil.copytree(base / "f1", base / "x" / "f1")
    assert any("not results/stage_e/<session id>" in p for p in ea.session_problems(base / "x" / "f1", **kw))
    assert ea.out_problems(base / "new" / "analysis.json", root=root) == []
    assert ea.out_problems(base / "f1" / "analysis.json", root=root)                 # never into an existing session
    assert ea.out_problems(base / "analysis.json", root=root)
    (base / "saved").mkdir()
    (base / "saved" / "plan.json").write_text((base / "f1" / "plan.json").read_text())   # a saved plan ran nothing
    assert sorted(str(a["dir"].relative_to(base)) for a in e2p.formal_attempts(root)) == ["f1", "old1", "x/f1"]
