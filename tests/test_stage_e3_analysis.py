"""Tests for benchmarks/stage_e3_analysis.py (decision 041 v3 §3.3-3.5): the same-session fit, intercepts and their
differences, the shared-slope fit and phi (checked against the preregistration tally on Phase 1 data), the Q4_0 format
contrast against kappa, the premise, the verdicts, the interpretation table, NT and the formal-session checks."""

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

import stage_e3 as e3
import stage_e3_analysis as ea
import test_stage_e3 as te

ROOT = Path(__file__).resolve().parent.parent
PINS = te.PINS
FILES = e3.config_files(PINS, te.pinned())
N_PARAMS = te.CFG["e3"]["n_params"]
PHASE1 = {"macos-metal": ROOT / "results/core_a/macos-metal/formal-20261009-a2/analysis.json",
          "wsl2-cuda": ROOT / "results/core_a/wsl2-cuda/formal-20261009-rtx-a1/analysis.json"}


def analyze(rows, platform="wsl2-cuda"):
    return ea.analyze(rows, FILES, platform=platform, n_params=N_PARAMS)


def rows_for(model, se_frac=0.002, missing=(), errors=()):
    """Three decode rows per config whose tok/s follow model(key) ms/token, with a small spread (SE ~ se_frac)."""
    rows = []
    for key in FILES:
        s, q = key.split("-", 1)
        if key in missing:
            continue
        t = model((s, q))
        for rep, k in enumerate((-1, 0, 1), start=1):
            tps = 1000 / (t * (1 + k * se_frac * math.sqrt(3)))
            ok = not (key in errors and rep == 1)
            rows.append({"kind": "decode", "size": s, "quant": q, "L": 2048, "rep": rep, "outcome": "ok" if ok else "error",
                         "metrics": {"avg_ts": tps} if ok else None})
    return rows


def affine(a=2.0, b=1.2, fmt=None):
    fmt = fmt or {}
    return lambda k: a + b * ea.x_gb(FILES, k, 2048) + fmt.get(k, 0.0)


def test_a_purely_affine_session_supports_h_e3a_and_leaves_d_inside():
    out = analyze(rows_for(affine()))
    assert out["fit"]["a"] == pytest.approx(2.0, rel=1e-3) and out["fit"]["b"] == pytest.approx(1.2, rel=1e-3)
    assert out["H-E3a"]["verdict"] == "supported"
    assert {v["state"] for v in out["format_contrast"]["by_size"].values()} == {"inside"}
    assert out["H-E3b"]["verdict"] == "not supported"
    assert out["row"].startswith("non-proportional component")
    assert out["premise"]["holds"] is True and out["premise"]["bytes_ratio"] > out["premise"]["observed_ratio"]


def test_a_cheaper_q4_0_in_both_sizes_supports_h_e3b():
    model = affine(fmt={("4B", "Q4_0"): -0.4, ("9B", "Q4_0"): -0.5, ("4B", "Q6_K"): 0.02, ("9B", "Q4_K_M"): -0.02})
    out = analyze(rows_for(model), "macos-metal")
    assert out["H-E3b"]["verdict"] == "supported" and out["H-E3a"]["verdict"] == "supported"
    assert out["row"].startswith("both")
    d = out["format_contrast"]
    assert d["kappa"] > 0 and all(v["threshold"] >= d["kappa"] for v in d["by_size"].values())


def test_kappa_sets_the_bar_when_k_quants_scatter():
    scatter = {("4B", "Q6_K"): 0.5, ("9B", "Q6_K"): -0.5}
    model = affine(fmt={**scatter, ("4B", "Q4_0"): -0.3, ("9B", "Q4_0"): -0.3})
    out = analyze(rows_for(model))
    assert out["format_contrast"]["kappa"] > 0.3 and out["H-E3b"]["verdict"] == "not supported"


def test_no_intercept_and_mixed_signs():
    zero = analyze(rows_for(affine(a=0.0)))
    assert zero["H-E3a"]["verdict"] == "not supported" and zero["premise"]["holds"] is False
    assert zero["row"].startswith("premise fails")
    mixed = analyze(rows_for(affine(fmt={("4B", "Q4_0"): -0.6, ("9B", "Q4_0"): 0.6})))
    assert mixed["row"].startswith("inconsistent format effect")


def test_a_missing_or_failed_cell_makes_the_dependent_hypotheses_nt():
    out = analyze(rows_for(affine(), missing=("9B-Q4_0",)))
    assert out["H-E3a"]["verdict"] == "supported" and out["H-E3b"]["verdict"] == "NT" and out["row"] == "incomplete"
    out = analyze(rows_for(affine(), errors=("9B-Q6_K",)))
    assert out["H-E3a"]["verdict"] == "NT" and out["fit"] is None and out["row"] == "incomplete"


def test_the_decode_bytes_subtract_token_embd_only_for_the_untied_9b():
    f9, f4 = FILES["9B-Q4_K_M"], FILES["4B-Q4_K_M"]
    kv = 32768 * (2048 - 64) + 2 * 50.25 * 2 ** 20
    assert ea.x_gb(FILES, ("9B", "Q4_K_M"), 2048) == pytest.approx((f9["size_bytes"] - f9["token_embd_bytes"] + kv) / 1e9)
    assert ea.x_gb(FILES, ("4B", "Q4_K_M"), 2048) == pytest.approx((f4["size_bytes"] + kv) / 1e9)


def test_two_platforms_make_a_general_statement_only_on_the_same_definite_row():
    a = analyze(rows_for(affine()), "macos-metal")
    b = analyze(rows_for(affine(fmt={("4B", "Q4_0"): -0.4, ("9B", "Q4_0"): -0.5})))
    assert ea.combine(a, a)["general"] == a["row"]
    assert ea.combine(a, b)["general"] is None and ea.combine(a, b)["statement"] == "the cause differs by platform"
    nt = analyze(rows_for(affine(), missing=("9B-Q4_0",)))
    for c in (ea.combine(nt, nt), ea.combine(a, nt), ea.combine(a, None)):
        assert c["general"] is None and c["statement"].startswith("incomplete")


def phase1(platform):
    """Phase 1's L 2048 decode cells as E3 times."""
    A = json.loads(PHASE1[platform].read_text())
    out = {}
    for key in FILES:
        if key in e3.NEW:
            continue
        c = A["cells"][f"{key}-2048"]["decode"]
        out[tuple(key.split("-", 1))] = {"t": 1000 / c["m"], "se": 1000 * c["se"] / c["m"] ** 2}
    return out


@pytest.mark.parametrize("platform, b, a4, a9, a0, c, phi_in, phi_lpo", [
    ("macos-metal", 3.275, 7.485, 9.948, 5.299, 0.519, 0.91, 0.455),
    ("wsl2-cuda", 1.139, 2.006, 2.212, 1.822, 0.043, 0.94, 0.730)])
def test_the_shared_slope_fit_and_phi_reproduce_the_tally_on_phase_1(platform, b, a4, a9, a0, c, phi_in, phi_lpo):
    T = phase1(platform)
    x = lambda k: ea.x_gb(FILES, k, 2048)  # noqa: E731
    points = {s: [(x((s, q)), T[(s, q)]["t"]) for q in ea.OLD_QUANTS] for s in e3.SIZES}
    tally_p = {"4B": 8424393664 / 2, "9B": 17920697312 / 2}                  # the tally's BF16 bytes / 2
    fit = ea.shared_slope(points, tally_p)
    assert (fit["b"], fit["a_by_size"]["4B"], fit["a_by_size"]["9B"]) == pytest.approx((b, a4, a9), abs=1e-3)
    assert fit["df"] == 3 and (fit["a0"], fit["c_ms_per_gparam"]) == pytest.approx((a0, c), abs=1e-3)
    exact = ea.shared_slope(points, N_PARAMS)                                 # the pinned header counts move a0 a little
    assert exact["a0"] == pytest.approx(a0, abs=5e-3) and exact["c_ms_per_gparam"] == pytest.approx(c, abs=1e-3)
    six = ea.ols([x(k) for k in ea.OLD], [T[k]["t"] for k in ea.OLD])
    rest = [k for k in ea.OLD if k not in ea.PAIR]
    four = ea.ols([x(k) for k in rest], [T[k]["t"] for k in rest])
    assert ea.phi(six, FILES, T, 2048) == pytest.approx(phi_in, abs=5e-3)
    assert ea.phi(four, FILES, T, 2048) == pytest.approx(phi_lpo, abs=5e-4)


def test_the_analysis_reports_the_intercept_differences_shared_fit_and_both_phis():
    out = analyze(rows_for(affine(fmt={("4B", "Q4_0"): -0.4, ("9B", "Q4_0"): -0.5})))
    a = out["intercepts"]["by_format"]
    d = out["intercepts"]["differences"]
    assert set(d) == {"Q8_0 - Q6_K", "Q8_0 - Q4_K_M", "Q8_0 - Q4_0", "Q6_K - Q4_K_M", "Q6_K - Q4_0", "Q4_K_M - Q4_0"}
    assert d["Q8_0 - Q4_0"]["diff"] == pytest.approx(a["Q8_0"]["a_f"] - a["Q4_0"]["a_f"])
    assert d["Q8_0 - Q4_0"]["se"] == pytest.approx(math.hypot(a["Q8_0"]["se"], a["Q4_0"]["se"]))
    assert out["shared_slope"]["b"] == pytest.approx(1.2, rel=1e-3) and out["shared_slope"]["df"] == 3
    assert set(out["premise"]["phi"]) == {"in_sample", "leave_pair_out"}
    assert out["premise"]["phi"]["leave_pair_out"] == pytest.approx(1.0, abs=0.02)     # a purely affine session


def test_only_the_first_valid_formal_session_after_at_most_one_void_attempt_is_analyzed(te_frozen, tmp_path):
    root, cfg = te_frozen
    base = root / "results" / "stage_e"
    pins_sha = hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()
    plan = e3.build_plan(PINS, cfg, platform="wsl2-cuda", mode="formal", pins_sha256=pins_sha, root=root)
    te.run(root, plan, "formal-a", expected_plan=plan, cfg=cfg, measure_b=lambda f: {"b_mib": f + 900, "drift": True})
    second = te.run(root, plan, "formal-b", expected_plan=plan, cfg=cfg)
    kw = dict(pins=PINS, cfg=cfg, root=root)
    assert any("void" in p for p in ea.session_problems(base / "formal-a", **kw))
    assert second["env_valid"] and ea.session_problems(base / "formal-b", **kw) == []
    assert any("not a finished formal" in p for p in ea.session_problems(base / "pilot-e3_cuda", **kw))
    shutil.copytree(base / "formal-b", base / "x" / "formal-c")                 # a nested copy is never analyzed
    assert any("not results/stage_e/<session id>" in p for p in ea.session_problems(base / "x" / "formal-c", **kw))
    s = json.loads((base / "formal-b" / "session.json").read_text())
    (base / "x" / "formal-c" / "session.json").write_text(json.dumps({**s, "started_at": "2000-01-01T00:00:00"}))
    assert any("earlier formal" in p for p in ea.session_problems(base / "formal-b", **kw))  # hidden attempts count


@pytest.fixture
def te_frozen(tmp_path):
    root = te.pilots(tmp_path / "repo")
    return root, te.pinned(e3.PLATFORMS)


def test_a_session_stays_analyzable_when_only_freeze_flags_or_pins_bytes_change(tmp_path):
    root = te.pilots(tmp_path / "repo")
    both = te.pinned(e3.PLATFORMS)
    plan = e3.build_plan(PINS, both, platform="wsl2-cuda", mode="formal", pins_sha256="old pins", root=root)
    assert te.run(root, plan, "formal-rtx", expected_plan=plan, cfg=both)["env_valid"]
    renamed = {**both, "freeze": {**both["freeze"], "pilot_e3_metal": "pilot-e3_metal"}}  # pins.toml moved on too
    assert ea.session_problems(root / "results" / "stage_e" / "formal-rtx", pins=PINS, cfg=renamed, root=root) == []
    (root / "results" / "stage_e" / "saved").mkdir()                              # a plan saved alone ran nothing
    (root / "results" / "stage_e" / "saved" / "plan.json").write_text(json.dumps(plan))
    (root / "results" / "stage_e" / "plan.json").write_text(json.dumps(plan))
    assert [a["dir"].name for a in e3.formal_attempts(root, "wsl2-cuda")] == ["formal-rtx"]
    assert ea.out_problems(root / "results" / "stage_e" / "new" / "e3.json", root=root) == []
    assert ea.out_problems(root / "results" / "stage_e" / "formal-rtx" / "e3.json", root=root)
