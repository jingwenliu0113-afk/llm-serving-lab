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
    assert qa2.ordering(noisy, [("4B", 512)])["verdict"] == "not supported"     # 005: not significant -> not supported
    assert qa2.ordering({}, [("4B", 512)])["verdict"] == "NT"


def test_h_q2_compares_each_context_with_phase1_at_512():
    rows = [kld_row(s, q, v * 1.4, ctx=c) for c in (2048, 8192) for (s, q), v in BASE_KLD.items()]
    h = qa2.h_q2(rows, PHASE1)
    assert h["H-Q2a"]["verdict"] == "supported" and len(h["H-Q2a"]["tests"]) == 8
    assert h["H-Q2b"]["verdict"] == "supported"
    rows[0] = kld_row("4B", "Q8_0", BASE_KLD[("4B", "Q8_0")] * 2.5, ctx=2048)
    assert qa2.h_q2(rows, PHASE1)["H-Q2b"]["verdict"] == "not supported"


# ---------------------------------------------------------------- the analysis of formal sessions (040 凍結)

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tomllib  # noqa: E402

import stage_q as sq  # noqa: E402
import test_stage_q as tq  # noqa: E402
from run_core_a import sha256_json  # noqa: E402

CFG_SHA = hashlib.sha256((ROOT / "configs" / "stage_q.toml").read_bytes()).hexdigest()
PINS_SHA = hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
HOSTS = {"macos-metal": {"system": "Darwin", "wsl": False, "nvidia": False},
         "wsl2-cuda": {"system": "Linux", "wsl": True, "nvidia": True}}


@pytest.fixture(scope="module")
def pilot_root(tmp_path_factory):
    root = tq.pilots(tmp_path_factory.mktemp("pilots") / "repo")
    (root / "results" / "quality").mkdir(parents=True)
    (root / "results" / "quality" / "formal-20261010-quality").symlink_to(
        ROOT / "results" / "quality" / "formal-20261010-quality")
    return root


def frozen(tmp_path, pilot_root):
    root = tmp_path / "repo"
    shutil.copytree(pilot_root, root, symlinks=True)
    return root, tq.FROZEN


def formal_session(root, part, rows, plan_over=None, session_over=None, sid=None, started="2026-10-24T00:00:00"):
    d = root / "results" / "stage_q" / (sid or f"formal-{part}")
    d.mkdir(parents=True)
    plan = {**sq.build_plan(PINS, part=part, mode="formal", pins_sha256=PINS_SHA, cfg=tq.FROZEN,
                            stage_q_sha256=CFG_SHA, root=root), **(plan_over or {})}
    session = {"stage": "Q", "part": part, "mode": "formal", "platform": sq.PARTS[part], "env_valid": True,
               "plan_sha256": sha256_json(plan), "harness_sha256": sq.harness_sha256(),
               "host": HOSTS[sq.PARTS[part]], "inputs_verified": True, "started_at": started, **(session_over or {})}
    (d / "plan.json").write_text(json.dumps(plan))
    (d / "session.json").write_text(json.dumps(session))
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def sessions(root):
    cuda = [kld_row(s, q, v, ctx=512) for (s, q), v in BASE_KLD.items()]
    q2 = [kld_row(s, q, v * 1.4, ctx=c) for c in (2048, 8192) for (s, q), v in BASE_KLD.items()]
    return {"q1-metal": formal_session(root, "q1-metal", metal()), "q1-cuda": formal_session(root, "q1-cuda", cuda),
            "q2": formal_session(root, "q2", q2)}


def analyze(root, cfg, paths, **kw):
    return qa2.analyze_sessions(paths, cfg=cfg, stage_q_sha256=CFG_SHA, pins_sha256=PINS_SHA, pins=PINS, root=root,
                                **kw)


def test_the_formal_sessions_give_every_hypothesis(tmp_path, pilot_root):
    root, cfg = frozen(tmp_path, pilot_root)
    out = analyze(root, cfg, sessions(root))
    assert {k: v["verdict"] for k, v in out["hypotheses"].items()} == {
        "H-Q1a": "supported", "H-Q1b": "supported", "H-Q1c": "supported", "H-Q2a": "supported", "H-Q2b": "supported"}
    assert out["sessions"]["q2"] == "formal-q2" and out["phase1_session"] == "formal-20261010-quality"
    assert out["harness_sha256"] == sq.harness_sha256() and out["stage_q_sha256"] == CFG_SHA


def test_a_part_not_yet_run_leaves_its_hypotheses_nt(tmp_path, pilot_root):
    root, cfg = frozen(tmp_path, pilot_root)
    paths = sessions(root)
    out = analyze(root, cfg, {"q2": paths["q2"]})
    v = {k: h["verdict"] for k, h in out["hypotheses"].items()}
    assert v == {"H-Q1a": "NT", "H-Q1b": "NT", "H-Q1c": "NT", "H-Q2a": "supported", "H-Q2b": "supported"}
    assert out["sessions"]["q1-metal"] is None
    v = {k: h["verdict"] for k, h in analyze(root, cfg, {"q1-metal": paths["q1-metal"]})["hypotheses"].items()}
    assert (v["H-Q1a"], v["H-Q1b"], v["H-Q1c"]) == ("supported", "NT", "supported")      # Q1c reads Phase 1


@pytest.mark.parametrize("plan_over, session_over, why", [
    ({}, {"mode": "pilot"}, "not formal"),
    ({}, {"env_valid": False}, "env-invalid"),
    ({"part": "q1-cuda"}, {}, "part"),
    ({}, {"harness_sha256": "0" * 64}, "another harness"),
    ({"harness_sha256": "0" * 64}, {}, "another harness"),
    ({"stage_q_sha256": "0" * 64}, {}, "configs/stage_q.toml"),
    ({"pins_sha256": "0" * 64}, {}, "pins.toml"),
    ({}, {"plan_sha256": "0" * 64}, "plan.json"),
    ({"params": {"n_thr": 4, "poll": 0}}, {}, "not build_plan's formal q2 plan"),
    ({}, {"host": HOSTS["macos-metal"]}, "not a wsl2-cuda host"),
    ({}, {"inputs_verified": False}, "inputs_verified"),
])
def test_a_session_that_is_not_a_frozen_formal_one_is_refused(tmp_path, pilot_root, plan_over, session_over, why):
    root, cfg = frozen(tmp_path, pilot_root)
    d = formal_session(root, "q2", [], plan_over, session_over)
    with pytest.raises(qa2.AnalysisError, match=why):
        analyze(root, cfg, {"q2": d})


def test_only_the_first_formal_session_of_a_part_is_analyzed(tmp_path, pilot_root):
    root, cfg = frozen(tmp_path, pilot_root)
    rows = [kld_row(s, q, v * 1.4, ctx=c) for c in (2048, 8192) for (s, q), v in BASE_KLD.items()]
    first = formal_session(root, "q2", rows, sid="formal-q2-a", started="2026-10-24T00:00:00")
    second = formal_session(root, "q2", rows, sid="formal-q2-b", started="2026-10-25T00:00:00")
    assert analyze(root, cfg, {"q2": first})["sessions"]["q2"] == "formal-q2-a"
    with pytest.raises(qa2.AnalysisError, match="earlier formal q2 sessions"):
        analyze(root, cfg, {"q2": second})


def test_no_analysis_before_the_freeze_or_without_its_evidence(tmp_path, pilot_root):
    root, cfg = frozen(tmp_path, pilot_root)
    paths = sessions(root)
    with pytest.raises(qa2.AnalysisError, match="freeze.stage_q"):
        analyze(root, {**cfg, "freeze": {**cfg["freeze"], "stage_q": False}}, paths)
    shutil.rmtree(root / "results" / "stage_q" / "pilot-q1c")
    with pytest.raises(qa2.AnalysisError, match="pilot_q1_cuda"):
        analyze(root, cfg, paths)


def test_the_cli_refuses_the_unfrozen_repo_config(tmp_path, capsys):
    assert qa2.main(["--q2", str(tmp_path / "x"), "--out", str(ROOT / "results" / "stage_q" / "analysis.json")]) == 2
    assert "freeze.stage_q" in capsys.readouterr().err
    assert not (ROOT / "results" / "stage_q" / "analysis.json").exists()
