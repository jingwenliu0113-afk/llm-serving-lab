"""Tests for quality/stage_q.py (decision 040): Stage Q plans in run_quality's job format, and the runner's platform
rules (Mac lock, the 140W adapter, one powermode; GPU lock for the CUDA parts)."""

import json
import tomllib
from pathlib import Path

import pytest

import stage_q as sq

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())


def jobs(plan, kind=None):
    return [j for j in plan["jobs"] if kind is None or j["kind"] == kind]


def test_q1_metal_compares_the_first_200_chunks_and_runs_the_full_ppl():
    p = sq.build_plan(PINS, part="q1-metal", mode="formal")
    assert (p["platform"], p["params"]) == ("macos-metal", {"n_thr": 8, "poll": 0})
    base = jobs(p, "kld_base")
    assert len(base) == 2 and all(j["argv"][j["argv"].index("--chunks") + 1] == "200" for j in base)
    assert all(j["expect"] == {"n_chunks": 200} for j in jobs(p, "kld"))
    ppl = jobs(p, "ppl")
    assert len(ppl) == 8 and all("--chunks" not in j["argv"] and j["expect"] == {"n_chunks": 580} for j in ppl)
    assert base[0]["need_gib"] < 40                                     # about 25 GB: fits the Mac (037 §4)
    flags = base[0]["argv"]
    assert flags[flags.index("-c") + 1] == "512" and flags[flags.index("-t") + 1] == "8"


def test_q1_cuda_runs_the_same_kld_jobs_without_ppl():
    m, c = sq.build_plan(PINS, part="q1-metal", mode="formal"), sq.build_plan(PINS, part="q1-cuda", mode="formal")
    assert c["platform"] == "wsl2-cuda" and c["params"] == {"n_thr": 6, "poll": 0} and not jobs(c, "ppl")
    strip = lambda j: [a for a in j["argv"] if a not in ("8", "6")]    # noqa: E731 - the thread count differs
    assert [strip(j) for j in jobs(c, "kld")] == [strip(j) for j in jobs(m, "kld")]


def test_q2_runs_each_context_over_the_whole_test_set():
    p = sq.build_plan(PINS, part="q2", mode="formal")
    base = jobs(p, "kld_base")
    assert [(j["size"], j["ctx"], j["expect"]["n_chunks"]) for j in base] == [
        ("4B", 2048, 145), ("9B", 2048, 145), ("4B", 8192, 36), ("9B", 8192, 36)]
    assert all("--chunks" not in j["argv"] for j in base)
    for j in jobs(p):
        if "argv" in j:
            assert j["argv"][j["argv"].index("-c") + 1] == str(j["ctx"])
    assert len(jobs(p, "kld")) == 4 * 3 and len(jobs(p, "delete_kld")) == 4
    assert sq.expected_chunks(PINS, 512) == 580


def test_pilots_run_ten_chunks_everywhere():
    p = sq.build_plan(PINS, part="q2", mode="pilot")
    assert all(j["expect"]["n_chunks"] == 10 for j in jobs(p) if "expect" in j)
    with pytest.raises(sq.StageQError):
        sq.build_plan(PINS, part="q9", mode="formal")


def test_the_mac_rules_are_the_140w_adapter_and_one_powermode():
    ok = {"power": {"powermode": 2, "ac_adapter": {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"}},
          "hardware": {"iogpu_wired_limit_mb": 0}}
    assert sq.mac_problems(ok) == []
    old = {**ok, "power": {**ok["power"], "ac_adapter": {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"}}}
    assert any("140W" in p for p in sq.mac_problems(old))
    assert sq.mac_problems({**ok, "power": {**ok["power"], "powermode": 1}})


def env(ok_power=True):
    def capture(llama_bin, out_path, gpu_lock_held):
        rec = {"runtime": {"llama_cpp": {"build_number": PINS["llama_cpp"]["build_number"],
                                         "commit": PINS["llama_cpp"]["commit"]}},
               "repo": {"commit": "c", "dirty": False, "status": [], "source": "git"}, "probe_errors": [],
               "power": {"source": "AC", "battery_pct": 90, "thermal": {}, "powermode": 2,
                         "ac_adapter": {"manufacturer": "Apple Inc.",
                                        "name": "140W USB-C Power Adapter" if ok_power else "70W USB-C Power Adapter"}},
               "hardware": {"iogpu_wired_limit_mb": 0}}
        Path(out_path).write_text(json.dumps(rec))
        return rec
    return capture


def test_the_runner_refuses_without_its_lock_or_outside_results(tmp_path):
    plan = sq.build_plan(PINS, part="q1-metal", mode="pilot")
    kw = {"bin_dir": "/b", "models_dir": "/m", "data_dir": "/d", "capture_env": env(), "session_id": "s",
          "repo_root": tmp_path}
    with pytest.raises(sq.StageQError, match="Mac heavy lock"):
        sq.run_stage_q(plan, out_dir=tmp_path / "results" / "stage_q" / "s", **kw)
    with pytest.raises(sq.StageQError, match="results/stage_q"):
        sq.run_stage_q(plan, out_dir=tmp_path / "elsewhere", mac_lock_held=True, **kw)
    cuda = sq.build_plan(PINS, part="q2", mode="pilot")
    with pytest.raises(sq.StageQError, match="GPU lock"):
        sq.run_stage_q(cuda, out_dir=tmp_path / "results" / "stage_q" / "t", **kw)


def test_a_formal_metal_run_refuses_the_70w_adapter(tmp_path):
    plan = sq.build_plan(PINS, part="q1-metal", mode="formal")
    with pytest.raises(sq.StageQError, match="140W"):
        sq.run_stage_q(plan, bin_dir="/b", models_dir="/m", data_dir="/d", capture_env=env(ok_power=False),
                       session_id="s", mac_lock_held=True, out_dir=tmp_path / "results" / "stage_q" / "s",
                       repo_root=tmp_path)


def test_a_pilot_session_runs_every_job_and_deletes_the_bases(tmp_path):
    plan = sq.build_plan(PINS, part="q1-cuda", mode="pilot")
    deleted = []

    def run_proc(argv, out, err, timeout):
        kind = "kld" if "--kl-divergence" in argv else ("kld_base" if "--kl-divergence-base" in argv else "ppl")
        text = ("computing over 10 chunks\\n" "====== KL divergence statistics ======\\n"
                "Mean    KLD:   0.002000 ±   0.000200\\n") if kind == "kld" else "computing over 10 chunks\\n"
        Path(out).write_text(text)
        Path(err).write_text("")
        return 0, 1.0

    s = sq.run_stage_q(plan, bin_dir="/b", models_dir="/m", data_dir=str(tmp_path / "d"), capture_env=env(),
                       session_id="s", gpu_lock_held=True, out_dir=tmp_path / "results" / "stage_q" / "s",
                       repo_root=tmp_path, disk_ok=lambda path, need: {"ok": True}, run_proc=run_proc,
                       delete_file=deleted.append)
    rows = [json.loads(x) for x in (tmp_path / "results" / "stage_q" / "s" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == len(plan["jobs"]) and s["stage"] == "Q" and s["part"] == "q1-cuda"
