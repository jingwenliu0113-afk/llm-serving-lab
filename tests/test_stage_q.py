"""Tests for quality/stage_q.py (decision 040): Stage Q plans in run_quality's job format, the runner's platform
rules (Mac lock, the 140W adapter, one powermode; GPU lock for the CUDA parts; both verified), the freeze and its
pilot evidence."""

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

import pytest

import stage_q as sq

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_q.toml").read_text())
PILOT_IDS = {"pilot_q1_metal": "pilot-q1m", "pilot_q1_cuda": "pilot-q1c", "pilot_q2": "pilot-q2"}
FROZEN = {**CFG, "freeze": {"stage_q": True, **PILOT_IDS}}


def jobs(plan, kind=None):
    return [j for j in plan["jobs"] if kind is None or j["kind"] == kind]


# ---------------------------------------------------------------- fakes shared with test_stage_q_analysis

def env(platform="macos-metal", adapter="140W USB-C Power Adapter"):
    def capture(llama_bin, out_path, gpu_lock_held):
        rec = {"runtime": {"llama_cpp": {"build_number": PINS["llama_cpp"]["build_number"],
                                         "commit": PINS["llama_cpp"]["commit"]}},
               "repo": {"commit": "c", "dirty": False, "status": [], "source": "git"}, "probe_errors": []}
        if platform == "macos-metal":
            rec.update(os={"system": "Darwin", "wsl": False}, hardware={"iogpu_wired_limit_mb": 0},
                       power={"source": "AC", "battery_pct": 90, "thermal": {}, "powermode": 2,
                              "ac_adapter": {"manufacturer": "Apple Inc.", "name": adapter}})
        else:
            rec.update(os={"system": "Linux", "wsl": True},
                       gpu={"nvidia": [{"enforced_power_limit_w": 300.0, "power_default_limit_w": 300.0}]})
        Path(out_path).write_text(json.dumps(rec))
        return rec
    return capture


def fake_llama(n=10, kld=0.002, ppl=8.0, fail=()):
    """llama-perplexity output that quality_parse accepts for every job kind; argv containing any of `fail` exits 1."""
    def run_proc(argv, out, err, timeout):
        if any(f in a for f in fail for a in argv):
            Path(out).write_text("")
            Path(err).write_text("error: something failed\n")
            return 1, 1.0
        if "--kl-divergence" in argv:
            text = (f"computing over {n} chunks\n====== KL divergence statistics ======\n"
                    f"Mean PPL(Q)                   :  {ppl:.6f} ±   0.010000\n"
                    f"Mean    KLD:   {kld:.6f} ±   0.000200\n")
        else:
            text = f"calculating perplexity over {n} chunks\n"
            if "--ppl-output-type" in argv:
                text += "".join(f"{i + 1:5d}  8.0000  2.000000  0.010000\n" for i in range(n))
            text += f"Final estimate: PPL = {ppl:.4f} +/- 0.0100\n"
        Path(out).write_text(text)
        Path(err).write_text("")
        return 0, 1.0
    return run_proc


def pinned_hash(path):
    """hash_file for formal runs: the pinned sha256 of the file name (wiki.test.raw and the GGUFs)."""
    name = Path(path).name
    if name == "wiki.test.raw":
        return PINS["quality"]["files"]["wiki_test_raw"]["sha256"]
    for m in PINS["core_a"]["models"]:
        for f in (m["reference"], *m["files"].values()):
            if f["filename"] == name:
                return f["sha256"]
    raise OSError(f"no such file {path}")


def run(root, plan, sid, *, capture=None, run_proc=None, **kw):
    args = dict(bin_dir="/b", models_dir="/m", data_dir=str(root / "d"), session_id=sid, mac_lock_held=True,
                gpu_lock_held=True, lock_check=lambda: True, out_dir=root / "results" / "stage_q" / sid,
                repo_root=root, disk_ok=lambda path, need: {"ok": True}, delete_file=lambda path: None,
                read_power=lambda: {"source": "AC", "battery_pct": 90}, sleep=lambda s: None,
                capture_env=capture or env(sq.PARTS[plan["part"]]), run_proc=run_proc or fake_llama())
    args.update(kw)
    return sq.run_stage_q(plan, **args)


def pilots(root, parts=tuple(sq.PARTS)):
    """Real pilot sessions (fake llama) under root, named as FROZEN names them."""
    ids = {part: PILOT_IDS[key] for key, part in sq.FREEZE_EVIDENCE.items()}
    for part in parts:
        run(root, sq.build_plan(PINS, part=part, mode="pilot"), ids[part])
    return root


def formal(part, root):
    return sq.build_plan(PINS, part=part, mode="formal", cfg=FROZEN, stage_q_sha256="q" * 64, root=root)


@pytest.fixture(scope="module")
def frozen_root(tmp_path_factory):
    return pilots(tmp_path_factory.mktemp("frozen") / "repo")


# ---------------------------------------------------------------- plans

def test_q1_metal_compares_the_first_200_chunks_and_runs_the_full_ppl(frozen_root):
    p = formal("q1-metal", frozen_root)
    assert (p["platform"], p["params"]) == ("macos-metal", {"n_thr": 8, "poll": 0})
    base = jobs(p, "kld_base")
    assert len(base) == 2 and all(j["argv"][j["argv"].index("--chunks") + 1] == "200" for j in base)
    assert all(j["expect"] == {"n_chunks": 200} and j["base"] == f"kld_base-{j['size']}" for j in jobs(p, "kld"))
    ppl = jobs(p, "ppl")
    assert len(ppl) == 8 and all("--chunks" not in j["argv"] and j["expect"] == {"n_chunks": 580} for j in ppl)
    assert base[0]["need_gib"] < 40                                     # about 25 GB: fits the Mac (037 §4)
    flags = base[0]["argv"]
    assert flags[flags.index("-c") + 1] == "512" and flags[flags.index("-t") + 1] == "8"
    assert "{data}/kld/{session}/stage-q-4B.kld" in base[0]["argv"]     # one base per session


def test_q1_cuda_runs_the_same_kld_jobs_without_ppl(frozen_root):
    m, c = formal("q1-metal", frozen_root), formal("q1-cuda", frozen_root)
    assert c["platform"] == "wsl2-cuda" and c["params"] == {"n_thr": 6, "poll": 0} and not jobs(c, "ppl")
    strip = lambda j: [a for a in j["argv"] if a not in ("8", "6")]    # noqa: E731 - the thread count differs
    assert [strip(j) for j in jobs(c, "kld")] == [strip(j) for j in jobs(m, "kld")]


def test_q2_runs_each_context_over_the_whole_test_set(frozen_root):
    p = formal("q2", frozen_root)
    base = jobs(p, "kld_base")
    assert [(j["size"], j["ctx"], j["expect"]["n_chunks"]) for j in base] == [
        ("4B", 2048, 145), ("9B", 2048, 145), ("4B", 8192, 36), ("9B", 8192, 36)]
    assert all("--chunks" not in j["argv"] for j in base)
    for j in jobs(p):
        if "argv" in j:
            assert j["argv"][j["argv"].index("-c") + 1] == str(j["ctx"])
    assert len(jobs(p, "kld")) == 4 * 3 and len(jobs(p, "delete_kld")) == 4
    assert sq.expected_chunks(PINS, 512) == 580


def test_the_base_size_follows_the_scored_tokens():
    assert [sq.scored_per_chunk(c) for c in (512, 2048, 8192)] == [255, 1023, 4095]
    assert sq.kld_need_gib(580, 512) == pytest.approx(73.5 * 1.1 / 1.073741824, abs=0.1)
    assert sq.kld_need_gib(145, 2048) == pytest.approx(73.5 / 580 * 145 * 1023 / 255 * 1.1 / 1.073741824, abs=0.1)


def test_pilots_run_ten_chunks_everywhere():
    p = sq.build_plan(PINS, part="q2", mode="pilot")
    assert all(j["expect"]["n_chunks"] == 10 for j in jobs(p) if "expect" in j)
    with pytest.raises(sq.StageQError):
        sq.build_plan(PINS, part="q9", mode="pilot")


# ---------------------------------------------------------------- the Mac rules (037 §4)

def test_the_mac_rules_are_the_140w_adapter_and_one_powermode():
    ok = {"power": {"source": "AC", "battery_pct": 90, "thermal": {}, "powermode": 2,
                    "ac_adapter": {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"}},
          "hardware": {"iogpu_wired_limit_mb": 0}}
    assert sq.mac_problems(ok) == [] and sq.platform_problems(ok, "macos-metal") == []    # no 70 W rule (019)
    old = {**ok, "power": {**ok["power"], "ac_adapter": {"manufacturer": "Apple Inc.",
                                                         "name": "70W USB-C Power Adapter"}}}
    assert [p for p in sq.platform_problems(old, "macos-metal") if "adapter" in p] == [
        "AC adapter 'Apple Inc.' '70W USB-C Power Adapter' (037 §4 requires Apple Inc. '140W USB-C Power Adapter')"]
    assert sq.platform_problems({**ok, "power": {**ok["power"], "powermode": 1}}, "macos-metal")
    assert sq.platform_problems({**ok, "power": {**ok["power"], "source": "Battery"}}, "macos-metal")


def test_a_formal_metal_run_starts_with_the_140w_adapter_and_refuses_the_70w(frozen_root, tmp_path):
    plan = formal("q1-metal", frozen_root)
    root = tmp_path / "repo"
    s = run(root, plan, "f1", expected_plan=plan, pins=PINS, hash_file=pinned_hash, run_proc=fake_llama(n=200))
    assert s["env_valid"] is True and s["inputs_verified"] is True and s["host"]["system"] == "Darwin"
    assert s["status"] == "finished"
    root = tmp_path / "repo2"
    with pytest.raises(sq.StageQError, match="037 §4 requires"):
        run(root, plan, "f2", expected_plan=plan, pins=PINS, hash_file=pinned_hash,
            capture=env(adapter="70W USB-C Power Adapter"))
    refused = json.loads((root / "results" / "stage_q" / "f2" / "session.json").read_text())
    assert refused["status"] == "refused" and sq.formal_attempts(root, "q1-metal") == []


def test_a_formal_run_needs_build_plans_plan_and_the_pinned_inputs(frozen_root, tmp_path):
    plan = formal("q2", frozen_root)
    root = tmp_path / "repo"
    edited = {**plan, "jobs": plan["jobs"][:5]}
    for bad in (None, edited):
        with pytest.raises(sq.StageQError, match="exactly build_plan's"):
            run(root, plan if bad is None else bad, "f", expected_plan=None if bad is None else plan, pins=PINS,
                hash_file=pinned_hash)
    with pytest.raises(sq.StageQError, match="not the pinned"):
        run(root, plan, "g", expected_plan=plan, pins=PINS, hash_file=lambda path: "0" * 64)
    with pytest.raises(sq.StageQError, match="another harness"):
        run(root, {**plan, "harness_sha256": "0" * 64}, "h", expected_plan=plan, pins=PINS, hash_file=pinned_hash)
    assert not (root / "results" / "stage_q" / "f").exists() and not (root / "results" / "stage_q" / "h").exists()


# ---------------------------------------------------------------- the runner

def test_the_runner_refuses_without_its_lock_or_outside_results(tmp_path):
    plan = sq.build_plan(PINS, part="q1-metal", mode="pilot")
    with pytest.raises(sq.StageQError, match="Mac heavy lock"):
        run(tmp_path, plan, "s", mac_lock_held=False)
    with pytest.raises(sq.StageQError, match="Mac heavy lock"):
        run(tmp_path, plan, "s", lock_check=lambda: False)                  # the flag alone is never trusted
    with pytest.raises(sq.StageQError, match="results/stage_q"):
        run(tmp_path, plan, "s", out_dir=tmp_path / "elsewhere")
    cuda = sq.build_plan(PINS, part="q2", mode="pilot")
    for kw in ({"gpu_lock_held": False}, {"lock_check": lambda: False}):
        with pytest.raises(sq.StageQError, match="GPU lock"):
            run(tmp_path, cuda, "t", **kw)


def test_the_mac_lock_must_be_held_by_an_ancestor(tmp_path):
    lock = tmp_path / "lock"
    assert not sq.mac_lock_owned(str(lock), pid=10, parent_of=lambda pid: pid - 1)
    lock.mkdir()
    (lock / "owner.txt").write_text("pid=7\nstarted_at=x\ncwd=/y\n")
    parents = {10: 9, 9: 7, 7: 1, 5: 1}
    assert sq.mac_lock_owned(str(lock), pid=10, parent_of=parents.get)
    assert not sq.mac_lock_owned(str(lock), pid=5, parent_of=parents.get)


def test_a_pilot_session_runs_every_job_and_deletes_the_bases(tmp_path):
    plan = sq.build_plan(PINS, part="q1-cuda", mode="pilot")
    base = tmp_path / "d" / "kld" / "s"
    base.mkdir(parents=True)
    for size in ("4B", "9B"):                                           # the fake llama writes no base file
        (base / f"stage-q-{size}.kld").write_text("x")
    deleted = []
    s = run(tmp_path, plan, "s", delete_file=deleted.append)
    rows = [json.loads(x) for x in (tmp_path / "results" / "stage_q" / "s" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == len(plan["jobs"]) and all(r["outcome"] == "ok" for r in rows)
    assert s["stage"] == "Q" and s["part"] == "q1-cuda" and s["harness_sha256"] == sq.harness_sha256()
    assert s["host"] == {"system": "Linux", "wsl": True, "nvidia": True} and s["inputs_verified"] is False
    assert deleted == [str(base / f"stage-q-{size}.kld") for size in ("4B", "9B")]


def test_a_failed_base_skips_its_kld_runs(tmp_path):
    plan = sq.build_plan(PINS, part="q1-cuda", mode="pilot")
    run(tmp_path, plan, "s", run_proc=fake_llama(fail=("stage-q-4B.kld",)))
    rows = {json.loads(x)["job_id"]: json.loads(x)
            for x in (tmp_path / "results" / "stage_q" / "s" / "rows.jsonl").read_text().splitlines()}
    assert rows["kld_base-4B"]["outcome"] == "error"
    assert {rows[k]["outcome"] for k in ("kld_self-4B", "kld-4B-Q8_0", "kld-4B-Q6_K", "kld-4B-Q4_K_M")} == {"skipped"}
    assert rows["delete_kld-4B"]["outcome"] == "error" and rows["kld-9B-Q8_0"]["outcome"] == "ok"


# ---------------------------------------------------------------- config, freeze and evidence

def test_the_repo_config_is_well_formed_and_not_yet_frozen():
    assert sq.config_problems(CFG) == [] and CFG["freeze"]["stage_q"] is False
    assert CFG["stage_q"]["phase1_session"] == "formal-20261010-quality"
    assert sq.config_problems({**CFG, "freeze": {**CFG["freeze"], "stage_q": True}})      # true needs the pilot ids
    assert sq.config_problems({**CFG, "extra": 1})


def test_formal_plans_need_the_freeze_this_parts_pilot_and_record_the_harness(frozen_root, tmp_path):
    for cfg in (None, CFG):
        with pytest.raises(sq.StageQError, match="freeze.stage_q"):
            sq.build_plan(PINS, part="q2", mode="formal", cfg=cfg, root=frozen_root)
    with pytest.raises(sq.StageQError, match="pilot first"):
        formal("q2", tmp_path)
    q2_only = pilots(tmp_path / "q2only", parts=("q2",))
    assert formal("q2", q2_only)["part"] == "q2"                       # the other parts' pilots: the analysis
    p = formal("q2", frozen_root)
    assert p["stage_q_sha256"] == "q" * 64 and p["harness_sha256"] == sq.harness_sha256()
    assert sq.build_plan(PINS, part="q2", mode="pilot")["harness_sha256"] == sq.harness_sha256()


def test_the_harness_fingerprint_covers_the_stage_q_modules_and_their_unguarded_imports():
    assert len(sq.harness_sha256()) == 64 and all((ROOT / f).is_file() for f in sq.HARNESS_FILES)
    assert {Path(f).name for f in sq.HARNESS_FILES} == {"stage_q.py", "stage_q_analysis.py", "run_quality.py",
                                                         "quality_parse.py", "disk_guard.py"}


def test_the_freeze_needs_a_passed_pilot_of_each_part_from_this_harness(frozen_root, tmp_path):
    assert sq.freeze_evidence_problems(CFG, tmp_path) == []                          # nothing to check before it
    assert len(sq.freeze_evidence_problems(FROZEN, tmp_path)) == 3
    assert sq.freeze_evidence_problems(FROZEN, frozen_root, PINS) == []
    root = tmp_path / "copy"
    shutil.copytree(frozen_root, root, symlinks=True)
    d = root / "results" / "stage_q" / "pilot-q2"
    session, plan = json.loads((d / "session.json").read_text()), json.loads((d / "plan.json").read_text())

    def tamper(s_over=None, plan_jobs=None):
        p = {**plan, "jobs": plan["jobs"] if plan_jobs is None else plan_jobs}
        (d / "plan.json").write_text(json.dumps(p))
        (d / "session.json").write_text(json.dumps({**session, "plan_sha256": sq.sha256_json(p), **(s_over or {})}))
        return sq.freeze_evidence_problems(FROZEN, root, PINS)

    assert tamper() == []
    assert any("not a q2 pilot" in p for p in tamper({"part": "q1-cuda"}))
    assert any("not a q2 pilot" in p for p in tamper({"mode": "formal"}))
    assert any("another harness" in p for p in tamper({"harness_sha256": "0" * 64}))
    assert any("not a passed q2 pilot" in p for p in tamper({"outcomes": {"ok": 23, "oom": 1, "error": 0,
                                                                          "skipped": 0}}))
    assert any("not build_plan's pilot plan" in p for p in tamper(plan_jobs=plan["jobs"][:1]))
    assert any("not a wsl2-cuda host" in p for p in tamper({"host": {"system": "Darwin", "wsl": False,
                                                                      "nvidia": False}}))
    (d / "session.json").write_text(json.dumps({**session, "plan_sha256": "0" * 64}))
    (d / "plan.json").write_text(json.dumps(plan))
    assert any("plan_sha256" in p for p in sq.freeze_evidence_problems(FROZEN, root, PINS))


# ---------------------------------------------------------------- attempts (040 補記 14)

def drifting_powermode():
    calls = iter(range(10 ** 6))

    def capture(llama_bin, out_path, gpu_lock_held):
        rec = env()(llama_bin, out_path, gpu_lock_held)
        if next(calls) > 2:
            rec["power"]["powermode"] = 0
        return rec
    return capture


def test_one_valid_formal_session_or_one_void_attempt_and_its_rerun(frozen_root, tmp_path):
    plan = formal("q1-metal", frozen_root)
    kw = dict(expected_plan=plan, pins=PINS, hash_file=pinned_hash, run_proc=fake_llama(n=200))
    root = tmp_path / "valid"
    run(root, plan, "a", **kw)
    with pytest.raises(sq.StageQError, match="already has formal attempts"):
        run(root, plan, "b", **kw)
    root = tmp_path / "void"
    assert run(root, plan, "a", capture=drifting_powermode(), **kw)["env_valid"] is False
    assert run(root, plan, "b", **kw)["env_valid"] is True
    with pytest.raises(sq.StageQError, match="already has formal attempts"):
        run(root, plan, "c", **kw)
    root = tmp_path / "interrupted"
    calls = iter(range(10 ** 6))

    def dies(argv, out, err, timeout):
        if next(calls) == 3:
            raise KeyboardInterrupt
        return fake_llama(n=200)(argv, out, err, timeout)

    with pytest.raises(KeyboardInterrupt):
        run(root, plan, "a", **{**kw, "run_proc": dies})
    assert [(a["id"], a["status"], a["valid"]) for a in sq.formal_attempts(root, "q1-metal")] == [("a", "running", False)]
    run(root, plan, "b", **kw)
    with pytest.raises(sq.StageQError, match="already has formal attempts"):
        run(root, plan, "c", **kw)


def test_a_superseded_attempt_needs_a_committed_decision_log_naming_both_sessions(frozen_root, tmp_path):
    plan = formal("q1-metal", frozen_root)
    kw = dict(expected_plan=plan, pins=PINS, hash_file=pinned_hash, run_proc=fake_llama(n=200))
    root = tmp_path / "repo"
    run(root, plan, "old", **kw)
    log = root / "docs" / "decisions" / "999-supersede.md"
    log.parent.mkdir(parents=True)
    log.write_text("old is superseded by new because ...\n")
    sup = {"old": "docs/decisions/999-supersede.md"}
    with pytest.raises(sq.StageQError, match="not committed"):
        run(root, plan, "new", superseded=sup, tracked=lambda p: False, **kw)
    with pytest.raises(sq.StageQError, match="does not name both"):
        run(root, plan, "other", superseded=sup, tracked=lambda p: True, **kw)
    with pytest.raises(sq.StageQError, match="not a repo-relative decision log"):
        run(root, plan, "new", superseded={"old": "README.md"}, tracked=lambda p: True, **kw)
    with pytest.raises(sq.StageQError, match="not a repo-relative decision log"):          # read on both machines
        run(root, plan, "new", superseded={"old": str(log)}, tracked=lambda p: True, **kw)
    with pytest.raises(sq.StageQError, match="cannot supersede itself"):
        run(root, plan, "new", superseded={"new": "docs/decisions/999-supersede.md"}, tracked=lambda p: True, **kw)
    s = run(root, plan, "new", superseded=sup, tracked=lambda p: True, **kw)
    assert s["status"] == "finished" and s["superseded"] == sup


def test_sessions_are_never_nested_and_plans_never_written_into_results(tmp_path):
    plan = sq.build_plan(PINS, part="q1-cuda", mode="pilot")
    with pytest.raises(sq.StageQError, match="results/stage_q/<session id>"):
        run(tmp_path, plan, "s", out_dir=tmp_path / "results" / "stage_q" / "a" / "s")
    out = ROOT / "results" / "stage_q" / "plan-test-should-not-exist.json"
    assert sq.main(["plan", "--part", "q2", "--mode", "pilot", "--out", str(out)]) == 2 and not out.exists()
    (tmp_path / "results" / "stage_q" / "saved").mkdir(parents=True)
    (tmp_path / "results" / "stage_q" / "saved" / "plan.json").write_text(json.dumps({**plan, "mode": "formal"}))
    assert sq.formal_attempts(tmp_path, "q1-cuda") == []                            # a saved plan ran nothing


def test_the_pilot_records_each_kld_base_size(tmp_path):
    plan = sq.build_plan(PINS, part="q1-cuda", mode="pilot")
    base = tmp_path / "d" / "kld" / "s"
    base.mkdir(parents=True)
    for size in ("4B", "9B"):
        (base / f"stage-q-{size}.kld").write_bytes(b"x" * 1234)
    run(tmp_path, plan, "s", delete_file=lambda path: None)
    rows = {json.loads(x)["job_id"]: json.loads(x)
            for x in (tmp_path / "results" / "stage_q" / "s" / "rows.jsonl").read_text().splitlines()}
    assert rows["kld_base-4B"]["base_bytes"] == 1234 and rows["kld_base-4B"]["need_gib"] > 0
    assert rows["delete_kld-9B"]["base_bytes"] == 1234 and rows["delete_kld-9B"]["deleted"] is True


def test_a_log_counts_as_committed_on_the_wsl_copy_through_its_deploy_manifest(tmp_path):
    log = tmp_path / "docs" / "decisions" / "999-x.md"
    log.parent.mkdir(parents=True)
    log.write_text("a supersedes b\n")
    assert sq.git_tracked(log, root=tmp_path) is False                                # no .git, no manifest
    sha = hashlib.sha256(log.read_bytes()).hexdigest()
    (tmp_path / "DEPLOY_MANIFEST.json").write_text(json.dumps({"files": {"docs/decisions/999-x.md": sha}}))
    assert sq.git_tracked(log, root=tmp_path) is True
    log.write_text("edited after the deploy\n")
    assert sq.git_tracked(log, root=tmp_path) is False



def test_an_interrupt_before_the_first_job_is_refused_not_an_attempt(frozen_root, tmp_path):
    plan = formal("q1-metal", frozen_root)

    def interrupted(**kw):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, plan, "a", capture=interrupted, expected_plan=plan, pins=PINS, hash_file=pinned_hash,
            run_proc=fake_llama(n=200))
    s = json.loads((tmp_path / "results" / "stage_q" / "a" / "session.json").read_text())
    assert s["status"] == "refused" and "KeyboardInterrupt" in s["refused"]
    assert sq.formal_attempts(tmp_path, "q1-metal") == []


def test_the_refused_window_ends_when_the_first_job_process_starts(frozen_root, tmp_path):
    plan = formal("q1-metal", frozen_root)
    kw = dict(expected_plan=plan, pins=PINS, hash_file=pinned_hash)

    def power():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):                     # the first job's charge gate: nothing ran yet
        run(tmp_path / "a", plan, "a", read_power=power, run_proc=fake_llama(n=200), **kw)
    s = json.loads((tmp_path / "a" / "results" / "stage_q" / "a" / "session.json").read_text())
    assert s["status"] == "refused" and s["refused"].startswith("interrupted before the first job")
    assert sq.formal_attempts(tmp_path / "a", "q1-metal") == []
    inner, calls = fake_llama(n=200), []

    def second_interrupted(*a):
        calls.append(a)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return inner(*a)

    with pytest.raises(KeyboardInterrupt):                     # after the first job: an attempt
        run(tmp_path / "b", plan, "b", run_proc=second_interrupted, **kw)
    assert [a["status"] for a in sq.formal_attempts(tmp_path / "b", "q1-metal")] == ["running"]
    assert sq.refused_why(RuntimeError("x")).startswith("failed before the first job: RuntimeError")
    with pytest.raises(SystemExit, match="143"):
        sq._exit_on_signal(15, None)
