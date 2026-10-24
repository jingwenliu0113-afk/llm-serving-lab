"""Tests for benchmarks/stage_v_report.py (decision 044): a formal Stage V session (from test_run_stage_v's fake WSL)
becomes <session>/analysis.json only when every 044 refusal is clear, and a re-run writes the same bytes (021 R3)."""

import ast
import copy
import json
from pathlib import Path

import pytest

import capacity_report as cr
import stage_v_plan as sv
import stage_v_report as svr
import test_run_stage_v as t

ROOT = Path(__file__).resolve().parent.parent
UNIFORM = ROOT / "results" / "core_b" / "formal-20261009-coreb-uniform"
ADVANCED = ROOT / "results" / "advanced" / "formal-20261011-adv"


def formal(tmp_path, name):
    """A formal session in a repo that also holds the checkout's service/ (the profiles the report reads)."""
    repo = t.make_repo(tmp_path)
    (repo[0] / "service").symlink_to(ROOT / "service")
    out, s, w = t.session(tmp_path, name, repo=repo)
    c, d, p = t.configs(repo[0], repo[2], formal=True)
    return out, s, repo[0], (c, d, p)


def report(out, root, cfgs, change=None, **kw):
    c, d, p = cfgs
    cap = cr.load_session(out)
    if change:
        change(cap)
    return svr.report(cap, cfg=c, stage_d_cfg=d, pins=p, stage_v_sha256=t.SV_SHA, stage_d_sha256=t.SD_SHA,
                      pins_sha256=t.PINS_SHA, uniform=kw.get("uniform", cr.load_uniform(UNIFORM)),
                      advanced_runs=kw.get("advanced", cr.load_advanced_references(ADVANCED)), repo_root=root)


@pytest.mark.parametrize("name", ["V-ADM", "V-CLS"])
def test_a_formal_session_is_reported(tmp_path, name):
    out, s, root, cfgs = formal(tmp_path, name)
    rep = report(out, root, cfgs)
    assert rep["replay"]["ok"] and rep["sentinel"]["gate"]["ok"] and rep["stage"] == "V" and rep["session"] == name
    assert all(u["holds"] for u in rep["units"].values())
    prov = rep["provenance"]
    assert prov["session_id"] == out.name and prov["session"] == name and prov["deployed_commit"] == t.td.COMMIT
    assert (prov["stage_v_sha256"], prov["stage_d_sha256"], prov["pins_sha256"]) == (t.SV_SHA, t.SD_SHA, t.PINS_SHA)
    assert prov["gateway_path_sha256"] == sv.gateway_path_sha256() == cfgs[0]["freeze"]["gateway_path_sha256"]
    assert prov["harness_sha256"] == sv.harness_sha256() and prov["capserve_sha256"] == s["capserve_sha256"]
    assert prov["freeze_mini"] == t.MINI_ID and prov["profile_sha256"] == s["profile_sha256"]
    assert rep["report"]["advanced_references"] is True
    assert rep["sentinel"]["cross_session_shift"]["references"]["core_output_throughput"] > 0
    assert set(rep["platform"]["server_argv_sha256"]) == {"U2K", *json.loads((out / "plan.json").read_text())["points"]}
    assert json.loads(json.dumps(rep)) == rep


def _first(cap, pred):
    return next(x for x in cap if pred(x))


def _gw(cap, pred=lambda b, x: True):
    """The gateway record of the first V boot that ran a bench and satisfies pred(boot, point)."""
    plan = cap["plan"]
    b = _first(cap["session"]["boots"], lambda b: b["phase"] == "V" and b["reached_first_bench"]
               and pred(b, plan["points"][b["workload"]]))
    return cap["session"]["runs"][b["tag"]]["gateway"]


def _valid_v(cap):
    return _first(cap["rows"], lambda r: r["phase"] == "V" and r["valid"])


def _hide(cap):
    """A V boot marked as failed before its first bench although it holds a valid run, with another profile."""
    v = _valid_v(cap)
    b = cap["session"]["boots"][v["boot"]]
    b["reached_first_bench"] = False
    cap["session"]["runs"][b["tag"]]["gateway"]["profile"] = "service/profiles/verified-formal-20261017-c1b.json"


CHANGES = {
    "location": lambda cap: cap.update(dir=cap["dir"].parent.parent),
    "product": lambda cap: cap.update(meta={"purpose": "product-test"}),
    "session name": lambda cap: cap["session"].update(session="V-CLS"),
    "no gateway record": lambda cap: cap["session"]["runs"][cap["session"]["boots"][_valid_v(cap)["boot"]]["tag"]]
    .pop("gateway"),
    "hidden gateway": _hide,
    "server env": lambda cap: next(iter(cap["session"]["runs"].values())).update(env={}),
    "session server env": lambda cap: cap["session"].update(server_env={}),
    "sentinel argv": lambda cap: _first(cap["rows"], lambda r: r["phase"] == "sentinel")["argv"].append("--extra"),
    "started_at": lambda cap: cap["session"].update(started_at=None),
    "bench argv": lambda cap: _valid_v(cap).update(argv=_valid_v(cap)["argv"] + ["--extra"]),
    "no argv": lambda cap: _valid_v(cap).pop("argv"),
    "gateway settings": lambda cap: _gw(cap)["settings"].update(tokenizer="vllm"),
    "gateway profile": lambda cap: _gw(cap).update(profile="service/profiles/verified-formal-20261017-c1b.json"),
    "gateway toml": lambda cap: _gw(cap).update(toml_sha256="0" * 64),
    "gateway argv": lambda cap: _gw(cap, lambda b, x: x["provisional"])["argv"].remove("--allow-provisional"),
    "readyz": lambda cap: _gw(cap)["readyz"]["queue"].update(max_queue_wait_s=2.0),
    "server argv": lambda cap: next(iter(cap["session"]["runs"].values()))["argv"].extend(
        ["--long-prefill-token-threshold", "512"]),
    "informal": lambda cap: cap["session"].update(mode="mini"),
    "stage": lambda cap: cap["session"].update(stage="V-mini"),
    "aborted": lambda cap: cap["session"].update(aborted="RunError: x"),
    "lock": lambda cap: cap["session"].update(gpu_lock_held=False),
    "env": lambda cap: cap["session"].update(env_valid=False),
    "deployed": lambda cap: cap["session"].update(repo_source="WORKTREE"),
    "vllm": lambda cap: cap["session"].update(vllm="0.31.0"),
    "inherited": lambda cap: cap["session"].update(inherited_unlisted=["VLLM_X"]),
    "rows sha": lambda cap: cap.update(rows_file_sha256="0" * 64),
    "derived sha": lambda cap: cap["derived_files"].update({_valid_v(cap)["tag"]: "0" * 64}),
    "plan sha": lambda cap: cap["session"].update(plan_sha256="0" * 64),
    "stage_v sha": lambda cap: cap["session"].update(stage_v_sha256="0" * 64),
    "dataset sha": lambda cap: cap["session"]["dataset_sha256"].update({"var-2048": "0" * 64}),
    "harness": lambda cap: cap["session"].update(harness_sha256="0" * 64),
    "request path": lambda cap: cap["session"].update(gateway_path_sha256="0" * 64),
    "capserve": lambda cap: cap["session"].update(capserve_sha256=None),
    "profile": lambda cap: cap["session"]["profile_sha256"].update({"W-90": "0" * 64}),
    "access log": lambda cap: _valid_v(cap)["access_log"].update(problems=["x"]),
    "access log missing": lambda cap: _valid_v(cap).pop("access_log"),
    "access log count": lambda cap: _valid_v(cap)["access_log"].update(joined=1),
    "sentinel gateway": lambda cap: cap["session"]["runs"][cap["session"]["boots"][0]["tag"]].update(gateway={}),
    "replay": lambda cap: _valid_v(cap).update(outcome="fail"),
    "spill": lambda cap: _valid_v(cap)["spill"].update(spilled=True),
    "gate": lambda cap: cap["requests"].update({
        r["tag"]: [dict(q, ttft_ms=q["ttft_ms"] * 1.05) for q in cap["requests"][r["tag"]]]
        for r in cap["rows"] if r.get("position") == "end"}),
}
MATCH = {"location": "not a session directory under results/stage_v", "product": "product-test",
         "session name": "is not one of", "no gateway record": "no gateway record",
         "hidden gateway": "settings / profile", "server env": "server env is not", "session server env":
         "session's server env", "sentinel argv": "sentinel-start-U2K-c25: bench argv", "started_at": "started_at",
         "bench argv": "bench argv", "no argv": "without its bench argv", "gateway settings": "settings / profile",
         "gateway profile": "settings / profile", "gateway toml": "settings / profile", "gateway argv": "gateway argv",
         "readyz": "readyz", "server argv": "server argv", "informal": "not formal", "stage": "stage",
         "aborted": "aborted", "lock": "lock", "env": "env-invalid", "deployed": "DEPLOYED_COMMIT", "vllm": "vLLM",
         "inherited": "inherited", "rows sha": "rows.jsonl", "derived sha": "derived file", "plan sha": "plan_sha256",
         "stage_v sha": "stage_v.toml", "dataset sha": "var-2048", "harness": "harness_sha256",
         "request path": "gateway_path_sha256", "capserve": "capserve_sha256", "profile": "profile_sha256",
         "access log": "access log", "access log missing": "access log", "access log count": "access log",
         "sentinel gateway": "a gateway on", "replay": "replay", "spill": "spill", "gate": "sentinel gate"}


def test_every_refusal(tmp_path):
    out, s, root, cfgs = formal(tmp_path, "V-ADM")
    assert report(out, root, cfgs)["replay"]["ok"]
    assert set(CHANGES) == set(MATCH)
    for name, change in CHANGES.items():
        with pytest.raises(svr.ReportError, match=MATCH[name]):
            report(out, root, cfgs, change=change)


def test_a_verified_point_never_gets_the_flag(tmp_path):
    out, s, root, cfgs = formal(tmp_path, "V-CLS")

    def flag(cap):
        _gw(cap, lambda b, x: not x["provisional"])["argv"].append("--allow-provisional")

    with pytest.raises(svr.ReportError, match="gateway argv"):
        report(out, root, cfgs, change=flag)


def test_config_refusals(tmp_path):
    out, s, root, (c, d, p) = formal(tmp_path, "V-ADM")
    unfrozen = copy.deepcopy(c)
    unfrozen["freeze"]["stage_v"] = False
    with pytest.raises(svr.ReportError, match="freeze.stage_v is not true"):
        report(out, root, (unfrozen, d, p))
    broken = copy.deepcopy(c)
    broken["stage_v"]["n"] = 4
    with pytest.raises(svr.ReportError, match=r"(^|; )stage_v.toml: n "):
        report(out, root, (broken, d, p))
    sd_broken = copy.deepcopy(d)
    sd_broken["stage_d"]["guard_band"] = 0.04
    with pytest.raises(svr.ReportError, match=r"(^|; )stage_d.toml: guard_band"):
        report(out, root, (c, sd_broken, p))
    other = copy.deepcopy(c)
    other["freeze"]["mini"] = "smoke-v-other"
    with pytest.raises(svr.ReportError, match="freeze evidence"):
        report(out, root, (other, d, p))
    plan = copy.deepcopy(c)
    plan["stage_v"]["datasets"]["v4-90-r2"] = "e" * 64
    with pytest.raises(svr.ReportError, match="plan.json is not"):
        report(out, root, (plan, d, p))
    (root / "results" / "stage_d").unlink()                                   # no committed D5 file to recompute from
    with pytest.raises(svr.ReportError, match="pacing"):
        report(out, root, (c, d, p))


def test_a_profile_that_is_not_the_pinned_bytes_is_refused(tmp_path):
    out, s, root, (c, d, p) = formal(tmp_path, "V-ADM")
    (root / "service").unlink()
    (root / "service" / "profiles").mkdir(parents=True)
    for f in (ROOT / "service" / "profiles").glob("*.json"):
        (root / "service" / "profiles" / f.name).write_bytes(f.read_bytes())
    prof = root / "service" / "profiles" / "provisional-stage-v.json"
    prof.write_bytes(prof.read_bytes() + b"\n")
    with pytest.raises(svr.ReportError, match="provisional-stage-v.json sha256"):
        report(out, root, (c, d, p))


def test_formal_sessions_are_at_least_2_hours_apart(tmp_path):
    out, s, root, cfgs = formal(tmp_path, "V-ADM")
    import datetime as dt
    start = dt.datetime.fromisoformat(s["started_at"])
    prev = root.joinpath(*sv.RESULTS, "formal-t-earlier")
    prev.mkdir()

    def write(finished, started=start - dt.timedelta(hours=3)):
        (prev / "session.json").write_text(json.dumps({"started_at": started.isoformat(),
                                                       "finished_at": finished.isoformat()}))

    write(start - dt.timedelta(hours=2))
    assert report(out, root, cfgs)["replay"]["ok"]
    write(start - dt.timedelta(hours=2) + dt.timedelta(seconds=1))
    with pytest.raises(svr.ReportError, match="2 h"):
        report(out, root, cfgs)
    write(start + dt.timedelta(hours=9), started=start + dt.timedelta(hours=8))     # a later session: its own gap
    assert report(out, root, cfgs)["replay"]["ok"]


def test_the_cli_writes_only_the_sessions_analysis_and_a_rerun_is_byte_identical(tmp_path, monkeypatch, capsys):
    out, s, root, (c, d, p) = formal(tmp_path, "V-CLS")
    args = ["--session", str(out), "--uniform-session", str(UNIFORM), "--advanced-session", str(ADVANCED)]
    assert svr.main(args + ["--out", str(tmp_path / "x" / "analysis.json")]) == 2
    assert "analysis.json" in capsys.readouterr().err
    assert svr.main(args + ["--out", str(out / "analysis.json")]) == 2           # not this checkout's results/stage_v
    assert not (out / "analysis.json").exists()
    monkeypatch.setattr(svr, "ROOT", root)
    monkeypatch.setattr(svr, "load_configs", lambda: (c, d, p, t.SV_SHA, t.SD_SHA, t.PINS_SHA))
    before = sorted(x.relative_to(out).as_posix() for x in out.rglob("*"))
    assert svr.main(args + ["--out", str(out / "analysis.json")]) == 0
    first = (out / "analysis.json").read_bytes()
    assert sorted(x.relative_to(out).as_posix() for x in out.rglob("*")) == sorted(before + ["analysis.json"])
    assert svr.main(args + ["--out", str(out / "analysis.json")]) == 0
    assert (out / "analysis.json").read_bytes() == first                      # 021 R3: the same bytes again
    (out / "analysis.json").write_bytes(first.replace(b'"ok": true', b'"ok": false', 1))
    assert svr.main(args + ["--out", str(out / "analysis.json")]) == 2
    assert "021 R3" in capsys.readouterr().err
    assert json.loads(first)["provenance"]["session_id"] == out.name


def test_the_report_imports_only_research_modules():
    tree = ast.parse((ROOT / "benchmarks" / "stage_v_report.py").read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(n == "capserve" or n.startswith("capserve.") for n in names)
    assert names <= {"__future__", "argparse", "hashlib", "json", "sys", "pathlib", "tomllib", "capacity_analysis",
                     "capacity_plan", "capacity_report", "core_b_plan", "run_adv", "run_core_a", "run_stage_v",
                     "stage_d_plan", "stage_v_analysis", "stage_v_plan"}
