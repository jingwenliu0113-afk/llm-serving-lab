"""Tests for benchmarks/stage_e3_analysis.py (decision 041 v3 §3.3-3.5): the same-session fit, intercepts, the Q4_0
format contrast against kappa, the premise, the verdicts, the interpretation table, NT and the formal-session checks."""

import hashlib
import json
import math
import tomllib
from pathlib import Path

import pytest

import stage_e3 as e3
import stage_e3_analysis as ea
import test_stage_e3 as te

ROOT = Path(__file__).resolve().parent.parent
PINS = te.PINS
FILES = e3.config_files(PINS, te.pinned())


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
    out = ea.analyze(rows_for(affine()), FILES, platform="wsl2-cuda")
    assert out["fit"]["a"] == pytest.approx(2.0, rel=1e-3) and out["fit"]["b"] == pytest.approx(1.2, rel=1e-3)
    assert out["H-E3a"]["verdict"] == "supported"
    assert {v["state"] for v in out["format_contrast"]["by_size"].values()} == {"inside"}
    assert out["H-E3b"]["verdict"] == "not supported"
    assert out["row"].startswith("non-proportional component")
    assert out["premise"]["holds"] is True and out["premise"]["bytes_ratio"] > out["premise"]["observed_ratio"]


def test_a_cheaper_q4_0_in_both_sizes_supports_h_e3b():
    model = affine(fmt={("4B", "Q4_0"): -0.4, ("9B", "Q4_0"): -0.5, ("4B", "Q6_K"): 0.02, ("9B", "Q4_K_M"): -0.02})
    out = ea.analyze(rows_for(model), FILES, platform="macos-metal")
    assert out["H-E3b"]["verdict"] == "supported" and out["H-E3a"]["verdict"] == "supported"
    assert out["row"].startswith("both")
    d = out["format_contrast"]
    assert d["kappa"] > 0 and all(v["threshold"] >= d["kappa"] for v in d["by_size"].values())


def test_kappa_sets_the_bar_when_k_quants_scatter():
    scatter = {("4B", "Q6_K"): 0.5, ("9B", "Q6_K"): -0.5}
    model = affine(fmt={**scatter, ("4B", "Q4_0"): -0.3, ("9B", "Q4_0"): -0.3})
    out = ea.analyze(rows_for(model), FILES, platform="wsl2-cuda")
    assert out["format_contrast"]["kappa"] > 0.3 and out["H-E3b"]["verdict"] == "not supported"


def test_no_intercept_and_mixed_signs():
    zero = ea.analyze(rows_for(affine(a=0.0)), FILES, platform="wsl2-cuda")
    assert zero["H-E3a"]["verdict"] == "not supported" and zero["premise"]["holds"] is False
    assert zero["row"].startswith("premise fails")
    mixed = ea.analyze(rows_for(affine(fmt={("4B", "Q4_0"): -0.6, ("9B", "Q4_0"): 0.6})), FILES, platform="wsl2-cuda")
    assert mixed["row"].startswith("inconsistent format effect")


def test_a_missing_or_failed_cell_makes_the_dependent_hypotheses_nt():
    out = ea.analyze(rows_for(affine(), missing=("9B-Q4_0",)), FILES, platform="wsl2-cuda")
    assert out["H-E3a"]["verdict"] == "supported" and out["H-E3b"]["verdict"] == "NT" and out["row"] == "incomplete"
    out = ea.analyze(rows_for(affine(), errors=("9B-Q6_K",)), FILES, platform="wsl2-cuda")
    assert out["H-E3a"]["verdict"] == "NT" and out["fit"] is None and out["row"] == "incomplete"


def test_the_decode_bytes_subtract_token_embd_only_for_the_untied_9b():
    f9, f4 = FILES["9B-Q4_K_M"], FILES["4B-Q4_K_M"]
    kv = 32768 * (2048 - 64) + 2 * 50.25 * 2 ** 20
    assert ea.x_gb(FILES, ("9B", "Q4_K_M"), 2048) == pytest.approx((f9["size_bytes"] - f9["token_embd_bytes"] + kv) / 1e9)
    assert ea.x_gb(FILES, ("4B", "Q4_K_M"), 2048) == pytest.approx((f4["size_bytes"] + kv) / 1e9)


def test_two_platforms_make_a_general_statement_only_on_the_same_row():
    a = ea.analyze(rows_for(affine()), FILES, platform="macos-metal")
    b = ea.analyze(rows_for(affine(fmt={("4B", "Q4_0"): -0.4, ("9B", "Q4_0"): -0.5})), FILES, platform="wsl2-cuda")
    assert ea.combine(a, a)["general"] == a["row"]
    assert ea.combine(a, b)["general"] is None and ea.combine(a, None)["general"] is None


def test_only_the_first_valid_formal_session_of_a_platform_is_analyzed(te_frozen, tmp_path):
    root, cfg = te_frozen
    pins_sha = hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()
    plan = e3.build_plan(PINS, cfg, platform="wsl2-cuda", mode="formal", pins_sha256=pins_sha, stage_e3_sha256="e",
                         root=root)
    first = te.run(root, plan, "formal-a", expected_plan=plan, cfg=cfg)
    kw = dict(pins=PINS, cfg=cfg, pins_sha256=pins_sha, stage_e3_sha256="e", root=root)
    assert first["env_valid"] and ea.session_problems(root / "results" / "stage_e" / "formal-a", **kw) == []
    te.run(root, plan, "formal-b", expected_plan=plan, cfg=cfg)
    assert any("earlier valid formal" in p for p in ea.session_problems(root / "results" / "stage_e" / "formal-b", **kw))
    assert any("formal" in p for p in ea.session_problems(root / "results" / "stage_e" / "pilot-e3_cuda", **kw))


@pytest.fixture
def te_frozen(tmp_path):
    root = te.pilots(tmp_path / "repo")
    return root, te.pinned(e3.PLATFORMS)
