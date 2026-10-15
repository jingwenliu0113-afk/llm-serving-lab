"""Tests for benchmarks/capacity_report.py: the Stage C (C1) report CLI and every refusal of decision 030 8.7
(PREREGISTRATION §8), on sessions the runner writes with the fake world of test_run_capacity (no GPU)."""

import copy
import hashlib
import json
import shutil

import pytest

import capacity_plan as cp
import capacity_report as cr
from run_core_a import sha256_json
from test_run_capacity import (BASE_ENV, CAP_SHA, CORE_UNIFORM, PINS_SHA, ROOT, World, datasets, frozen,
                               make_plan, run)

ADV = ROOT / "results" / "advanced" / "formal-20261011-adv"


def uniform_session():
    return {"session": json.loads((CORE_UNIFORM / "session.json").read_text()),
            "plan": json.loads((CORE_UNIFORM / "plan.json").read_text()),
            "rows": [json.loads(x) for x in (CORE_UNIFORM / "rows.jsonl").read_text().splitlines() if x.strip()]}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    import tomllib

    cfg = tomllib.loads((ROOT / "configs" / "capacity.toml").read_text())
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    tmp = tmp_path_factory.mktemp("report")
    ds, sha = datasets(tmp)
    c, p = frozen(cfg, pins, sha)
    plan = make_plan(c, p, formal=True)
    out, summary, rows = run(World(tmp), plan, c, p, mixed_datasets=ds)
    return {"tmp": tmp, "out": out, "cfg": c, "pins": p, "plan": plan, "ds": ds, "uniform": uniform_session()}


def rep(setup, cap=None, **kw):
    kw.setdefault("cfg", setup["cfg"])
    kw.setdefault("pins", setup["pins"])
    kw.setdefault("capacity_sha256", CAP_SHA)
    kw.setdefault("pins_sha256", PINS_SHA)
    kw.setdefault("uniform", setup["uniform"])
    kw.setdefault("repo_root", setup["tmp"] / "repo")
    return cr.report(cap if cap is not None else cr.load_session(setup["out"]), **kw)


def refused(setup, cap=None, **kw):
    with pytest.raises(cr.ReportError) as exc:
        rep(setup, cap, **kw)
    return str(exc.value)


def loaded(setup):
    return copy.deepcopy(cr.load_session(setup["out"]))


# ------------------------------------------------------------------ the accepted session

def test_a_formal_session_is_accepted_and_analysed(setup):
    a = rep(setup)
    assert json.loads(json.dumps(a)) == a
    assert a["schema_version"] == 1 and a["stage"] == "C1" and a["slo"] == cp.SLO_S2
    assert a["replay"]["ok"] and a["sentinel"]["gate"]["ok"]
    wl = a["workloads"]
    assert (wl["U2K"]["c_slo"]["value"], wl["U2K"]["c_slo"]["status"]) == (33, "EXACT")
    assert (wl["M8K"]["c_slo"]["value"], wl["M8K"]["c_slo"]["status"]) == (0, "INFEASIBLE")
    assert a["hypotheses"]["H-S1"]["2048"]["verdict"] == "supported"


def test_analysis_carries_provenance_and_platform(setup):
    a = rep(setup)
    s = json.loads((setup["out"] / "session.json").read_text())
    pv = a["provenance"]
    assert pv["session_id"] == "s1" and pv["deployed_commit"] == s["repo_commit"]
    assert pv["pins_sha256"] == PINS_SHA and pv["capacity_sha256"] == CAP_SHA
    assert pv["plan_sha256"] == sha256_json(setup["plan"]) and pv["confirm_plan_sha256"] == s["confirm_plan_sha256"]
    assert pv["c0_branch"] == "A" and pv["n_near"] == dict.fromkeys(cp.WORKLOADS, 5)
    assert pv["mixed_dataset_sha256"] == s["mixed_dataset_sha256"]
    pf = a["platform"]
    assert pf["gpu"] == "NVIDIA GeForce RTX 5070 Ti" and pf["driver"] == "617.42" and pf["vllm"] == "0.30.0"
    assert pf["model"] == "RedHatAI/Qwen3.5-9B-quantized.w4a16@a398088c4228b0ae0c8c78df88fd1e4bf445f068"
    assert pf["K"] == 4294967296 and pf["U"] == "0.91" and pf["num_gpu_blocks"] == 248 and pf["block_size"] == 528
    argv, _ = cp.server_argv_for(setup["pins"], port=8000, vllm_bin=".venv/bin/vllm", base_env=BASE_ENV)
    assert pf["server_argv_sha256"] == cr.server_argv_sha256(argv) and pf["env"] == s["server_env"]


def test_server_argv_sha256_strips_host_port_and_the_binary_path():
    a = ["/x/vllm", "serve", "m", "--host", "127.0.0.1", "--port", "8000", "--dtype", "auto"]
    b = [".venv/bin/vllm", "serve", "--dtype", "auto", "m", "--host", "0.0.0.0", "--port", "8100"]
    assert cr.server_argv_sha256(a) != cr.server_argv_sha256(b)       # the other arguments' order still counts
    c = [".venv/bin/vllm", "serve", "m", "--host", "0.0.0.0", "--port", "8100", "--dtype", "auto"]
    assert cr.server_argv_sha256(a) == cr.server_argv_sha256(c)
    assert cr.server_argv_sha256(a) == sha256_json(["serve", "m", "--dtype", "auto"])


def test_cross_session_shift_uses_the_committed_references(setup):
    if not (ADV / "derived" / "B0-2048-uniform-r1-requests.jsonl").is_file():
        pytest.skip("Advanced derived files not present")
    a = rep(setup, advanced_runs=cr.load_advanced_references(ADV))
    css = a["sentinel"]["cross_session_shift"]
    assert css["references"]["core_output_throughput"] == pytest.approx(314.94, abs=0.01)
    assert css["shift"] is True                   # the fake world's throughput is nothing like Core's
    assert rep(setup)["sentinel"]["cross_session_shift"]["shift"] is None


# ------------------------------------------------------------------ refusals (030 8.7)

@pytest.mark.parametrize("mutate, match", [
    (lambda c: c["session"].update(formal=False), "not formal"),
    (lambda c: c["session"].update(mode="mini"), "mode"),
    (lambda c: c["session"].update(stage="C1-mini"), "stage"),
    (lambda c: c["plan"].update(stage="C1-mini"), "stage"),
    (lambda c: c["session"].update(env_valid=False), "env-invalid"),
    (lambda c: c["session"].update(aborted="KeyboardInterrupt: "), "aborted"),
    (lambda c: c["session"].update(gpu_lock_held=False), "lock"),
    (lambda c: c["session"].update(inherited_unlisted=["VLLM_USE_V1"]), "engine variables"),
    (lambda c: c["session"].update(repo_source="git"), "DEPLOYED_COMMIT"),
    (lambda c: c["session"].update(vllm="0.31.0"), "vLLM"),
    (lambda c: c["session"].update(capacity_sha256="00" * 32), "capacity.toml"),
    (lambda c: c["session"].update(plan_capacity_sha256="00" * 32), "capacity.toml"),
    (lambda c: c["session"].update(pins_sha256="00" * 32), "pins.toml"),
    (lambda c: c["session"].update(plan_sha256="00" * 32), "plan.json"),
    (lambda c: c["session"].update(confirm_plan_sha256="00" * 32), "confirm"),
    (lambda c: c.update(confirm_plan=None), "confirm"),
    (lambda c: c["session"]["mixed_dataset_sha256"].update({"8192": "f" * 64}), "mixed dataset"),
    (lambda c: c["session"]["mixed_dataset_sha256"].pop("2048"), "mixed dataset"),
    (lambda c: c["derived_files"].update({next(iter(c["derived_files"])): "00" * 32}), "derived"),
    (lambda c: c["session"]["derived_sha256"].pop(next(iter(c["session"]["derived_sha256"]))), "derived"),
    (lambda c: c.update(rows_file_sha256="00" * 32), "rows.jsonl"),
    (lambda c: c["plan"]["slo"].update(tpot_p95_ms=200), "SLO"),
    (lambda c: c["plan"]["workloads"]["U2K"]["timeouts_s"].update({"33": 60}), "plan"),
    (lambda c: next(iter(c["session"]["runs"].values()))["argv"].append("--enforce-eager"), "server argv"),
    (lambda c: next(iter(c["session"]["runs"].values()))["env"].update(VLLM_WSL2_ENABLE_PIN_MEMORY="1"), "server env"),
    (lambda c: c["rows"][3]["argv"].append("--request-id-prefix"), "bench argv"),
    (lambda c: c["session"]["server"].update(gpu_memory_utilization="0.85"), "gpu_memory_utilization"),
    (lambda c: c["rows"][5].update(search_pass=not c["rows"][5].get("search_pass")), "replay"),
    (lambda c: c.update(meta={"purpose": "product-test", "formal": False, "research_valid": False}), "product-test"),
    (lambda c: c["session"].update(purpose="product-test"), "product-test"),
    (lambda c: c["session"].update(research_valid=False), "product-test"),
    (lambda c: c["session"]["server_env"]["OTEL"].update(OTEL_SERVICE_NAME="x"), "session's server env"),
    (lambda c: c["session"]["runs"].pop(c["session"]["boots"][1]["tag"]), "no server record"),
])
def test_every_refusal(setup, mutate, match):
    cap = loaded(setup)
    mutate(cap)
    msg = refused(setup, cap)
    assert match in msg, msg


@pytest.mark.parametrize("kw, match", [
    ({"capacity_sha256": "00" * 32}, "capacity.toml"),
    ({"pins_sha256": "00" * 32}, "pins.toml"),
])
def test_refuses_when_the_current_config_bytes_differ(setup, kw, match):
    assert match in refused(setup, **kw)


def test_refuses_before_the_freeze_or_with_a_config_that_breaks_030(setup):
    c = copy.deepcopy(setup["cfg"])
    c["freeze"]["capacity"] = False
    assert "freeze.capacity" in refused(setup, cfg=c)
    c = copy.deepcopy(setup["cfg"])
    c["slo"]["tpot_p95_ms"] = 200
    msg = refused(setup, cfg=c)
    assert "capacity.toml" in msg and "SLO" in msg


def test_refuses_a_plan_that_is_not_built_from_the_uniform_session(setup):
    u = copy.deepcopy(setup["uniform"])
    u["session"]["session_id"] = "formal-20261010-coreb-mixed"
    assert "plan" in refused(setup, uniform=u)
    u = copy.deepcopy(setup["uniform"])
    u["session"]["env_valid"] = False
    assert "plan" in refused(setup, uniform=u)


@pytest.mark.parametrize("edit", [
    lambda p: p.update(guard_band=0.10),                                       # 030 8.1 m = 0.05
    lambda p: p["sentinel"].update(tolerance=0.05),                            # 030 8.6: 3 %
    lambda p: [w.update(n_near=3) for w in p["workloads"].values()],           # 030 8.4: n_near 5
])
def test_a_self_consistent_plan_that_is_not_build_plan_is_refused(setup, edit):
    """The replay reads the guard band, the sentinel tolerance and n from plan.json without recomputing them; only the
    build_plan comparison can refuse a plan.json whose session plan_sha256 was recorded consistently with it."""
    cap = loaded(setup)
    edit(cap["plan"])
    cap["session"]["plan_sha256"] = sha256_json(cap["plan"])   # what a runner given this plan would have recorded
    msg = refused(setup, cap)
    assert "build_plan" in msg, msg


def test_refuses_a_drifted_session(tmp_path, setup):
    """8.6 sentinel gate: the end sentinel's output throughput 4 % below the start's."""
    w = World(tmp_path, thr_scale={"sentinel-end-U2K-c25": 0.96})
    out, _, _ = run(w, setup["plan"], setup["cfg"], setup["pins"], mixed_datasets=setup["ds"])
    msg = refused(setup, cr.load_session(out), repo_root=tmp_path / "repo")
    assert "drift" in msg and "output_throughput" in msg


def test_refuses_a_non_formal_smoke_session(tmp_path, setup):
    import tomllib

    cfg = tomllib.loads((ROOT / "configs" / "capacity.toml").read_text())
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    plan = make_plan(cfg, pins)
    out, s, _ = run(World(tmp_path), plan, cfg, pins, mode="mini")
    assert s["smoke"]["ok"] is True
    msg = refused(setup, cr.load_session(out), repo_root=tmp_path / "repo")
    assert "not formal" in msg and "stage" in msg


def test_refuses_a_session_outside_results_capacity_or_under_runs(tmp_path, setup):
    for rel in ("runs/pt-20261017", "results/advanced/s1"):
        dst = tmp_path / "repo" / rel
        shutil.copytree(setup["out"], dst)
        assert "results/capacity" in refused(setup, cr.load_session(dst), repo_root=tmp_path / "repo")


def test_refuses_a_product_test_meta_in_the_session_directory(tmp_path, setup):
    dst = tmp_path / "repo" / "results" / "capacity" / "s1"
    shutil.copytree(setup["out"], dst)
    (dst / "meta.json").write_text(json.dumps({"purpose": "product-test", "formal": False, "research_valid": False}))
    assert "product-test" in refused(setup, cr.load_session(dst), repo_root=tmp_path / "repo")


def test_a_tampered_derived_file_or_rows_file_on_disk_is_refused(tmp_path, setup):
    dst = tmp_path / "repo" / "results" / "capacity" / "s1"
    shutil.copytree(setup["out"], dst)
    f = sorted((dst / "derived").iterdir())[0]
    f.write_text(f.read_text().replace('"out": 128', '"out": 127', 1))
    assert "derived" in refused(setup, cr.load_session(dst), repo_root=tmp_path / "repo")
    shutil.rmtree(dst)
    shutil.copytree(setup["out"], dst)
    (dst / "rows.jsonl").write_text((dst / "rows.jsonl").read_text() + "\n")
    assert "rows.jsonl" in refused(setup, cr.load_session(dst), repo_root=tmp_path / "repo")


def test_load_session_reads_every_file(setup):
    cap = cr.load_session(setup["out"])
    assert set(cap) >= {"dir", "session", "plan", "confirm_plan", "rows", "requests", "derived_files", "preflight",
                        "rows_file_sha256", "meta"}
    assert cap["meta"] is None and cap["preflight"]["repo"]["source"] == "DEPLOYED_COMMIT"
    assert set(cap["requests"]) == {r["tag"] for r in cap["rows"]} == set(cap["derived_files"])


# ------------------------------------------------------------------ CLI

def test_main_refuses_with_exit_2_and_writes_nothing(setup, monkeypatch, capsys):
    """The repo's capacity.toml is not frozen yet, so the real CLI refuses every session today."""
    monkeypatch.setattr(cr, "ROOT", setup["tmp"] / "repo")
    out = setup["out"] / "analysis.json"
    assert cr.main(["--session", str(setup["out"]), "--uniform-session", str(CORE_UNIFORM), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert not out.exists() and "refusing" in err and "freeze.capacity" in err


@pytest.mark.parametrize("name", ["rows.jsonl", "session.json", "plan.json", "confirm_plan.json", "derived",
                                  "analysis-2.json"])
def test_main_writes_only_the_sessions_own_analysis_json(setup, monkeypatch, capsys, name):
    """030 8.7: the report writes <session>/analysis.json and nothing else; any other --out under results/capacity/ (a
    session file the report has just checked included) is refused with exit 2 and every file stays as it was."""
    monkeypatch.setattr(cr, "ROOT", setup["tmp"] / "repo")
    monkeypatch.setattr(cr, "load_configs", lambda: (setup["cfg"], setup["pins"], CAP_SHA, PINS_SHA))
    target = sorted((setup["out"] / "derived").iterdir())[0] if name == "derived" else setup["out"] / name
    before = {p: p.read_bytes() for p in setup["out"].rglob("*") if p.is_file()}
    assert cr.main(["--session", str(setup["out"]), "--uniform-session", str(CORE_UNIFORM), "--out", str(target)]) == 2
    assert {p: p.read_bytes() for p in setup["out"].rglob("*") if p.is_file()} == before
    assert "analysis.json" in capsys.readouterr().err


def test_main_writes_analysis_json_for_an_accepted_session(setup, monkeypatch, capsys):
    monkeypatch.setattr(cr, "ROOT", setup["tmp"] / "repo")
    monkeypatch.setattr(cr, "load_configs", lambda: (setup["cfg"], setup["pins"], CAP_SHA, PINS_SHA))
    out = setup["out"] / "analysis.json"
    assert cr.main(["--session", str(setup["out"]), "--uniform-session", str(CORE_UNIFORM), "--out", str(out)]) == 0
    a = json.loads(out.read_text())
    assert a["provenance"]["session_id"] == "s1" and a["replay"]["ok"]
    printed = json.loads(capsys.readouterr().out)
    u2 = printed["workloads"]["U2K"]
    assert u2["c_slo"] == [33, "EXACT"] and u2["c_prod"] == [16, "LOWER"] and u2["flags"] == ["STRICT_COARSE"]
    assert printed["hypotheses"]["H-S2"]["U2K"] == "supported"
    out.unlink()


def test_main_refuses_an_out_path_outside_results_capacity(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(cr, "ROOT", setup["tmp"] / "repo")
    monkeypatch.setattr(cr, "load_configs", lambda: (setup["cfg"], setup["pins"], CAP_SHA, PINS_SHA))
    bad = setup["tmp"] / "repo" / "results" / "core_b" / "analysis.json"
    assert cr.main(["--session", str(setup["out"]), "--uniform-session", str(CORE_UNIFORM), "--out", str(bad)]) == 2
    assert not bad.exists()


def test_load_configs_hashes_the_repo_bytes():
    cfg, pins, cap_sha, pins_sha = cr.load_configs()
    assert cap_sha == hashlib.sha256((ROOT / "configs" / "capacity.toml").read_bytes()).hexdigest()
    assert pins_sha == hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()
    assert cfg["slo"]["id"] == "S2" and pins["vllm"]["version"] == "0.30.0"


def test_the_report_refuses_without_the_freeze_evidence(setup, monkeypatch):
    monkeypatch.setattr(cr.cp, "freeze_evidence_problems", lambda cfg, root: ["no passed mini smoke"])
    assert "freeze evidence" in refused(setup) and "no passed mini smoke" in refused(setup)
