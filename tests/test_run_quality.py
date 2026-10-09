"""Tests for quality/run_quality.py: the quality protocol of decision 005 as a
saved plan executed one llama-perplexity process per job, with KLD scratch
handled per docs/STORAGE_PLAN.md."""

import copy
import json
import stat
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

import run_quality as rq

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def plan(pins, **kw):
    args = dict(mode="pilot", n_thr=8, poll=0)
    args.update(kw)
    return rq.build_quality_plan(pins, **args)


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_job_order_per_size_follows_005(pins):
    frozen = rtx_frozen(pins)
    rtx = dict(platform="wsl2-cuda", n_thr=frozen["stage_b"]["rtx_n_thr"], poll=frozen["stage_b"]["rtx_poll"])
    for mode, p, extra_kld, extra_mc, kw in (("formal", frozen, 0, 0, rtx), ("pilot", pins, 2, 1, {})):
        jobs = plan(p, mode=mode, **kw)["jobs"]
        for size in [m["size"] for m in pins["core_a"]["models"]]:
            kinds = [j["kind"] for j in jobs if j["size"] == size]
            n = 6 + extra_kld
            assert kinds[:n] == ["kld_base", "kld_self"] + ["kld"] * (3 + extra_kld) + ["delete_kld"]
            assert Counter(kinds[n:])["mc"] == 4 * 3 + extra_mc and Counter(kinds[n:])["winogrande"] == 4


def test_ha5_pair_gets_per_chunk_ppl(pins):
    ppl = [(j["size"], j["point"]) for j in plan(pins)["jobs"] if j["kind"] == "ppl"]
    big, small = pins["core_a"]["models"][-1]["size"], pins["core_a"]["models"][0]["size"]
    q = pins["core_a"]["quants"]
    assert sorted(ppl) == sorted([(big, q[-1]), (small, q[0])])
    j = next(j for j in plan(pins)["jobs"] if j["kind"] == "ppl")
    assert arg(j["argv"], "--ppl-output-type") == "1"


def test_flags_match_005(pins):
    for j in plan(pins, n_thr=8, poll=50)["jobs"]:
        if j["kind"] == "delete_kld":
            continue
        a = j["argv"]
        assert a[0] == "{bin}/llama-perplexity"
        for flag, value in rq.QUALITY_FLAGS.items():
            want = "off" if (flag, j.get("variant")) == ("-fa", "fa_off") else value
            assert arg(a, flag) == want, (j["id"], flag)
        assert (arg(a, "-t"), arg(a, "--poll")) == ("8", "50")
        if j["kind"] in ("mc", "winogrande"):
            assert arg(a, "-np") == "16"


def test_kld_jobs_share_the_size_kld_file(pins):
    jobs = [j for j in plan(pins)["jobs"] if j["size"] == "4B"]
    base = next(j for j in jobs if j["kind"] == "kld_base")
    path = arg(base["argv"], "--kl-divergence-base")
    assert path == "{data}/kld/4B.kld"
    for j in jobs:
        if j["kind"] in ("kld", "kld_self"):
            assert arg(j["argv"], "--kl-divergence-base") == path and "--kl-divergence" in j["argv"]


def test_pilot_vs_formal_counts(pins):
    pilot = plan(pins)
    assert pilot["expected"]["n_chunks"] == 20
    assert all(arg(j["argv"], "--chunks") == "20" for j in pilot["jobs"] if j["kind"] in ("kld_base", "ppl"))
    frozen = rtx_frozen(pins)
    formal = plan(frozen, mode="formal", platform="wsl2-cuda", n_thr=frozen["stage_b"]["rtx_n_thr"],
                  poll=frozen["stage_b"]["rtx_poll"])
    assert formal["expected"]["n_chunks"] == pins["quality"]["expected"]["wikitext_chunks"]
    assert all("--chunks" not in j["argv"] for j in formal["jobs"] if j["kind"] != "delete_kld")
    mmlu = next(j for j in formal["jobs"] if j.get("task") == "mmlu")
    assert mmlu["expect"]["n_tasks"] == pins["quality"]["expected"]["mmlu_tasks"]


def test_formal_quality_runs_only_on_the_rtx(pins):
    """020: formal quality is measured once, on wsl2-cuda; the Mac keeps only the pilot path (012)."""
    frozen = rtx_frozen(pins)
    with pytest.raises(rq.QualityError, match="020"):
        plan(frozen, mode="formal")  # macos-metal
    unfrozen = copy.deepcopy(frozen)
    unfrozen["freeze"]["nvidia"] = False
    with pytest.raises(rq.QualityError, match="freeze.nvidia"):
        plan(unfrozen, mode="formal", platform="wsl2-cuda", n_thr=6, poll=0)


# --------------------------------------------------------------- runner --

FAKE_PPL = r'''
import os, sys
a = sys.argv
g = lambda f: a[a.index(f) + 1] if f in a else None
chunks = int(g("--chunks") or os.environ["FAKE_FULL_CHUNKS"])
if "--kl-divergence" in a:
    if os.environ.get("FAKE_KLD_FAIL") and "Q6_K" in g("-m"):
        print("kl_divergence: inconsistent vocabulary (1 vs 2)", file=sys.stderr); sys.exit(0)
    print(f"kl_divergence: computing over {chunks} chunks, n_ctx=512, batch_size=2048, n_seq=4")
    print("====== KL divergence statistics ======")
    print("Mean    KLD:   0.001400 ±   0.000010")
elif "--multiple-choice" in a or "--winogrande" in a:
    n = int(os.environ["FAKE_TASKS"])
    if "--winogrande" in a:
        for i in range(1, n + 1): print(f"{i}\t100.0000\t -1.0   -2.0  1  1")
        print(f"Final Winogrande score({n} tasks): 100.0000 +/- 0.0000", file=sys.stderr)
    else:
        for i in range(1, n + 1): print(f"{i}\t100.00000000")
        print("Final result: 100.0000 +/- 0.0000", file=sys.stderr)
else:
    if g("--kl-divergence-base"):
        open(g("--kl-divergence-base"), "wb").write(b"_logits_")
    print(f"perplexity: calculating perplexity over {chunks} chunks, n_ctx=512, batch_size=2048, n_seq=4", file=sys.stderr)
    for i in range(chunks): print(f"{i*512:8d}  9.0000  2.197225  0.100000")
    print("Final estimate: PPL = 9.0000 +/- 0.05000", file=sys.stderr)
'''


class Env:
    def __init__(self):
        self.calls = []

    def __call__(self, llama_bin, out_path, gpu_lock_held):
        self.calls.append(Path(out_path).name)
        rec = {"runtime": {"llama_cpp": {"build_number": 11146, "commit": COMMIT}}, "probe_errors": [],
               "power": {"source": "AC", "battery_pct": 80, "powermode": 0,  # 019
                         "ac_adapter": {"wattage_w": 68, "manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"},
                         "thermal": {"thermal_warning_recorded": False, "performance_warning_recorded": False}},
               "hardware": {"iogpu_wired_limit_mb": 0}}
        Path(out_path).write_text(json.dumps(rec))
        return rec


def run(tmp_path, monkeypatch, plan_obj, **kw):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "llama-perplexity"
    exe.write_text(f"#!{sys.executable}\n{FAKE_PPL}")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    data = tmp_path / "data"
    (data / "kld").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FAKE_FULL_CHUNKS", str(plan_obj["expected"]["n_chunks"]))
    monkeypatch.setenv("FAKE_TASKS", str(plan_obj["expected"]["pilot_tasks"]))
    kw.setdefault("path_exists", lambda p: False)
    kw.setdefault("disk_ok", lambda path, need_gib: {"ok": True})
    kw.setdefault("read_power", lambda: {"source": "AC", "battery_pct": 100})
    kw.setdefault("sleep", lambda s: None)
    summary = rq.run_quality(plan_obj, bin_dir=bin_dir, models_dir=tmp_path / "models", data_dir=data,
                             out_dir=tmp_path / "out", capture_env=Env(), session_id="q1", **kw)
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    return summary, rows, data


def small(pins):
    p = copy.deepcopy(pins)
    p["core_a"]["models"] = p["core_a"]["models"][:1]
    return plan(p)


def test_run_all_ok_and_kld_deleted(tmp_path, monkeypatch, pins):
    summary, rows, data = run(tmp_path, monkeypatch, small(pins))
    assert summary["outcomes"]["error"] == 0, [r for r in rows if r["outcome"] != "ok"]
    deletions = [r for r in rows if r["kind"] == "delete_kld"]
    assert deletions and deletions[0]["deleted"] is True
    assert not (data / "kld" / "4B.kld").exists()
    mc = next(r for r in rows if r["kind"] == "mc")
    assert mc["metrics"]["n_tasks"] == 200 and sum(mc["metrics"]["correct"]) == 200


def test_failed_kld_keeps_the_file(tmp_path, monkeypatch, pins):
    monkeypatch.setenv("FAKE_KLD_FAIL", "1")
    summary, rows, data = run(tmp_path, monkeypatch, small(pins))
    bad = [r for r in rows if r["kind"] == "kld" and r["outcome"] == "error"]
    assert bad and "inconsistent vocabulary" in bad[0]["reason"]
    deletion = next(r for r in rows if r["kind"] == "delete_kld")
    assert deletion["deleted"] is False
    assert (data / "kld" / "4B.kld").exists()


def test_disk_guard_failure_skips_kld_jobs(tmp_path, monkeypatch, pins):
    summary, rows, _ = run(tmp_path, monkeypatch, small(pins), disk_ok=lambda path, need_gib: {"ok": False})
    kld_rows = [r for r in rows if r["kind"] in ("kld_base", "kld_self", "kld")]
    assert all(r["outcome"] == "skipped" for r in kld_rows)
    assert any(r["outcome"] == "ok" for r in rows if r["kind"] == "mc")


def test_refuses_when_mac_lock_held_by_other(tmp_path, monkeypatch, pins):
    with pytest.raises(rq.QualityError, match="Mac heavy lock"):
        run(tmp_path, monkeypatch, small(pins), path_exists=lambda p: True)


def test_a_formal_mac_plan_is_refused_at_run_time_too(tmp_path, monkeypatch, pins):
    """020: an old or hand-made formal plan without platform (= macos-metal) never runs."""
    formal_mac = {**small(pins), "mode": "formal"}
    formal_mac.pop("platform")
    with pytest.raises(rq.QualityError, match="020"):
        run(tmp_path, monkeypatch, formal_mac, mac_lock_held=True)


def test_quality_session_writes_a_raw_manifest(tmp_path, monkeypatch, pins):
    summary, rows, data = run(tmp_path, monkeypatch, small(pins))
    manifest = json.loads((tmp_path / "out" / "raw_manifest.json").read_text())
    assert sorted(manifest["files"]) == sorted(p.relative_to(tmp_path / "out").as_posix()
                                               for p in (tmp_path / "out" / "raw").iterdir())


def test_pilot_can_be_limited_to_available_sizes(pins):
    p = rq.build_quality_plan(pins, mode="pilot", n_thr=8, poll=0, sizes=["4B"])
    assert {j["size"] for j in p["jobs"]} == {"4B"}
    assert any(j["kind"] == "ppl" for j in p["jobs"])  # the 4B half of the H-A5 pair
    with pytest.raises(rq.QualityError, match="all sizes"):
        rq.build_quality_plan(rtx_frozen(pins), mode="formal", n_thr=6, poll=0, sizes=["4B"], platform="wsl2-cuda")


def test_quality_session_reports_env_validity(tmp_path, monkeypatch, pins):
    summary, rows, data = run(tmp_path, monkeypatch, small(pins))
    assert summary["env_valid"] is True and summary["env_flags"] == []


def test_quality_session_pauses_to_charge(tmp_path, monkeypatch, pins):
    """016 applies to quality sessions too (they run on the Mac only)."""
    levels = iter([100, 30, 95] + [95] * 1000)
    summary, rows, _ = run(tmp_path, monkeypatch, small(pins),
                           read_power=lambda: {"source": "AC", "battery_pct": next(levels)})
    assert len(summary["charge_pauses"]) == 1 and summary["charge_pauses"][0]["to_pct"] == 95


# ------------------------------------------------------------ 020: on the RTX --


def rtx_frozen(pins):
    p = copy.deepcopy(pins)
    p["freeze"]["mac"] = p["freeze"]["nvidia"] = True
    return p


def test_rtx_plan_records_the_platform_and_needs_the_stage_b_freeze_and_params(pins):
    """020: quality runs on wsl2-cuda; formal needs freeze.nvidia and the §4 RTX N_thr / P from pins [stage_b]."""
    p = rtx_frozen(pins)
    sb = p["stage_b"]
    formal = plan(p, mode="formal", platform="wsl2-cuda", n_thr=sb["rtx_n_thr"], poll=sb["rtx_poll"])
    assert formal["platform"] == "wsl2-cuda" and formal["params"] == {"n_thr": sb["rtx_n_thr"], "poll": sb["rtx_poll"]}
    assert plan(pins)["platform"] == "macos-metal"  # default unchanged
    unfrozen = copy.deepcopy(p)
    unfrozen["freeze"]["nvidia"] = False
    with pytest.raises(rq.QualityError, match="freeze.nvidia"):
        plan(unfrozen, mode="formal", platform="wsl2-cuda", n_thr=sb["rtx_n_thr"], poll=sb["rtx_poll"])
    with pytest.raises(rq.QualityError, match="N_thr / P"):
        plan(p, mode="formal", platform="wsl2-cuda", n_thr=8, poll=0)
    with pytest.raises(rq.QualityError, match="platform"):
        plan(p, platform="cuda-native")
    # the job list is the same protocol on both platforms (only -t / --poll differ)
    mac, rtx = plan(p, n_thr=6, poll=0), plan(p, platform="wsl2-cuda", n_thr=6, poll=0)
    assert [j["id"] for j in mac["jobs"]] == [j["id"] for j in rtx["jobs"]]
    assert [j.get("argv") for j in mac["jobs"]] == [j.get("argv") for j in rtx["jobs"]]


class RtxEnv:
    def __init__(self, enforced=300.0, redeploy_after=None):
        self.calls, self.enforced, self.redeploy_after = [], enforced, redeploy_after

    def repo_commit(self):
        late = self.redeploy_after is not None and len(self.calls) > self.redeploy_after
        return ("f" if late else "a") * 40

    def __call__(self, llama_bin, out_path, gpu_lock_held):
        self.calls.append((Path(out_path).name, gpu_lock_held))
        rec = {"runtime": {"llama_cpp": {"build_number": 11146, "commit": COMMIT}}, "probe_errors": [],
               "repo": {"commit": self.repo_commit(), "dirty": False},
               "power": {"source": None, "battery_pct": None, "thermal": None},
               "gpu": {"nvidia": [{"enforced_power_limit_w": self.enforced, "power_default_limit_w": 300.0}]}}
        Path(out_path).write_text(json.dumps(rec))
        return rec


def rtx_small(pins, mode="pilot"):
    p = rtx_frozen(pins)
    p["core_a"]["models"] = p["core_a"]["models"][:1]
    return plan(p, mode=mode, platform="wsl2-cuda", n_thr=p["stage_b"]["rtx_n_thr"], poll=p["stage_b"]["rtx_poll"])


def run_rtx(tmp_path, monkeypatch, plan_obj, env, **kw):
    battery = []
    kw.setdefault("read_power", lambda: battery.append(1) or {"source": "AC", "battery_pct": 10})
    kw.setdefault("path_exists", lambda p: True)  # a Mac lock path must not matter on the RTX
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "llama-perplexity"
    exe.write_text(f"#!{sys.executable}\n{FAKE_PPL}")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    data = tmp_path / "data"
    (data / "kld").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FAKE_FULL_CHUNKS", str(plan_obj["expected"]["n_chunks"]))
    monkeypatch.setenv("FAKE_TASKS", str(plan_obj["expected"]["pilot_tasks"] or 200))
    summary = rq.run_quality(plan_obj, bin_dir=bin_dir, models_dir=tmp_path / "models", data_dir=data,
                             out_dir=tmp_path / "out", capture_env=env, session_id="q-rtx",
                             disk_ok=lambda path, need_gib: {"ok": True}, sleep=lambda s: None, **kw)
    return summary, battery


def test_rtx_quality_runs_under_the_gpu_lock_without_battery_or_mac_lock_rules(tmp_path, monkeypatch, pins):
    env = RtxEnv()
    summary, battery = run_rtx(tmp_path, monkeypatch, rtx_small(pins), env, gpu_lock_held=True)
    assert summary["outcomes"]["error"] == 0 and summary["env_valid"] is True
    assert summary["platform"] == "wsl2-cuda" and summary["gpu_lock_held"] is True
    assert battery == [] and summary["charge_pauses"] == []  # no battery, no 016 gate on the desktop
    assert all(held is True for _, held in env.calls)


def test_rtx_formal_quality_needs_the_gpu_lock_and_the_default_power_limit(tmp_path, monkeypatch, pins):
    with pytest.raises(rq.QualityError, match="GPU lock"):
        run_rtx(tmp_path, monkeypatch, rtx_small(pins, mode="formal"), RtxEnv())
    with pytest.raises(rq.QualityError, match="GPU lock"):  # pilots too (MACHINE_POLICY, 020)
        run_rtx(tmp_path, monkeypatch, rtx_small(pins), RtxEnv())
    (tmp_path / "b").mkdir()
    with pytest.raises(rq.QualityError, match="power limit"):
        run_rtx(tmp_path / "b", monkeypatch, rtx_small(pins, mode="formal"), RtxEnv(enforced=250.0),
                gpu_lock_held=True)


def test_rtx_env_flags_use_the_rtx_rule(tmp_path, monkeypatch, pins):
    summary, _ = run_rtx(tmp_path, monkeypatch, rtx_small(pins), RtxEnv(enforced=250.0), gpu_lock_held=True)
    assert summary["env_valid"] is False
    assert all("power limit" in f["reason"] for f in summary["env_flags"])


def test_kld_directory_is_created_when_missing(tmp_path, monkeypatch, pins):
    """The WSL copy has no data/kld (git-ignored, not deployed); kld_base must create it."""
    plan_obj = rtx_small(pins)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "llama-perplexity"
    exe.write_text(f"#!{sys.executable}\n{FAKE_PPL}")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    data = tmp_path / "data"
    data.mkdir()  # no kld/ inside
    monkeypatch.setenv("FAKE_FULL_CHUNKS", str(plan_obj["expected"]["n_chunks"]))
    monkeypatch.setenv("FAKE_TASKS", str(plan_obj["expected"]["pilot_tasks"]))
    summary = rq.run_quality(plan_obj, bin_dir=bin_dir, models_dir=tmp_path / "models", data_dir=data,
                             out_dir=tmp_path / "out", capture_env=RtxEnv(), session_id="q", gpu_lock_held=True,
                             disk_ok=lambda path, need_gib: {"ok": True}, sleep=lambda s: None)
    assert summary["outcomes"]["error"] == 0 and (data / "kld").is_dir()


def test_formal_rtx_quality_flags_a_mid_session_code_change(tmp_path, monkeypatch, pins):
    summary, _ = run_rtx(tmp_path, monkeypatch, rtx_small(pins, mode="formal"), RtxEnv(redeploy_after=3),
                         gpu_lock_held=True)
    assert summary["env_valid"] is False and any("mid-session" in f["reason"] for f in summary["env_flags"])
