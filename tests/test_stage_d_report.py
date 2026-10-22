"""Tests for benchmarks/stage_d_report.py (decision 039): a formal Stage D session (from test_run_stage_d's fake WSL)
becomes analysis.json only when every 039 refusal is clear."""

import copy
import json
from pathlib import Path

import pytest

import capacity_report as cr
import stage_d_report as sr
import test_run_stage_d as t

ROOT = Path(__file__).resolve().parent.parent
UNIFORM = ROOT / "results" / "core_b" / "formal-20261009-coreb-uniform"
ADVANCED = ROOT / "results" / "advanced" / "formal-20261011-adv"


def configs(root):
    """The frozen stage_d.toml / pins the fake session ran with (its evidence already exists under root)."""
    import hashlib
    sha = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in
           {"mixed:8192": root / "data" / "mixed-8192.jsonl", "var:2048": root / "data" / "var-2048.jsonl",
            "var:8192": root / "data" / "var-8192.jsonl"}.items()}
    c, p = copy.deepcopy(t.CFG), copy.deepcopy(t.PINS)
    c["freeze"].update(stage_d=True, mini_closed="smoke-d-a", mini_open="smoke-d-b", mini_verify="smoke-d-c")
    c["stage_d"]["mix_dataset_sha256"] = {"8192": sha["mixed:8192"]}
    p["core_b"]["mix_dataset_sha256"] = {**p["core_b"]["mix_dataset_sha256"], "8192": sha["mixed:8192"]}
    c["stage_d"]["var_dataset_sha256"] = {"2048": sha["var:2048"], "8192": sha["var:8192"]}
    return c, p


def report(out, root, **kw):
    c, p = configs(root)
    cap = cr.load_session(out)
    for k, v in kw.items():
        v(cap) if callable(v) else None
    return sr.report(cap, cfg=c, pins=p, stage_d_sha256=t.SD_SHA, pins_sha256=t.PINS_SHA,
                     uniform=cr.load_uniform(UNIFORM), advanced_runs=cr.load_advanced_references(ADVANCED),
                     repo_root=root)


@pytest.mark.parametrize("name", ["D-A1", "D-B", "D-C"])
def test_a_formal_session_is_reported(tmp_path, name):
    out, s, w = t.session(tmp_path, name)
    rep = report(out, tmp_path / "repo")
    assert rep["replay"]["ok"] and rep["sentinel"]["gate"]["ok"] and rep["kind"] == t.sd.KINDS[name]
    assert rep["provenance"]["session"] == name and rep["provenance"]["c1b_session"] == "formal-20261017-c1b"
    assert rep["report"]["advanced_references"] is True
    assert rep["sentinel"]["cross_session_shift"]["references"]["core_output_throughput"] > 0
    assert set(rep["platform"]["server_argv_sha256"]) >= {"U2K"}


def test_refusals(tmp_path):
    out, s, w = t.session(tmp_path, "D-C")
    root = tmp_path / "repo"

    def bench_argv(cap):
        r = next(r for r in cap["rows"] if r.get("argv"))
        r["argv"] = r["argv"] + ["--extra"]

    def gateway(cap):
        rec = next(v for v in cap["session"]["runs"].values() if v.get("gateway"))
        rec["gateway"]["settings"]["tokenizer"] = "vllm"

    def server(cap):
        rec = next(iter(cap["session"]["runs"].values()))
        rec["argv"] = rec["argv"] + ["--long-prefill-token-threshold", "512"]

    def informal(cap):
        cap["session"]["mode"] = "mini"

    def rows_sha(cap):
        cap["rows_file_sha256"] = "0" * 64

    for change, match in ((bench_argv, "bench argv"), (gateway, "gateway"), (server, "server argv"),
                          (informal, "not formal"), (rows_sha, "rows.jsonl")):
        with pytest.raises(sr.ReportError, match=match):
            report(out, root, change=change)


def test_an_unfrozen_config_is_refused(tmp_path):
    out, s, w = t.session(tmp_path, "D-A1")
    c, p = configs(tmp_path / "repo")
    c["freeze"]["stage_d"] = False
    with pytest.raises(sr.ReportError, match="freeze.stage_d"):
        sr.report(cr.load_session(out), cfg=c, pins=p, stage_d_sha256=t.SD_SHA, pins_sha256=t.PINS_SHA,
                  uniform=None, advanced_runs=None, repo_root=tmp_path / "repo")


def test_the_cli_writes_only_the_sessions_analysis(tmp_path, capsys):
    assert sr.main(["--session", str(tmp_path / "x"), "--uniform-session", str(UNIFORM), "--advanced-session",
                    str(ADVANCED), "--out", str(tmp_path / "x" / "analysis.json")]) == 2
    assert "results/stage_d" in capsys.readouterr().err
