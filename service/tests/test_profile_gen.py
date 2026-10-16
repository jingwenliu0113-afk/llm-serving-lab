"""Tests for capserve/profile_gen.py (decision 029 "產生器", M2; decision 030 8.7): the verified-profile generator.

The Stage C sessions are written by the real runner (run_capacity) in the fake world of tests/test_run_capacity.py,
and their analysis.json by the real capacity_report CLI, so every input is schema-faithful: nothing is hand-written.
Scenario A gives C_PROD LOWER / EXACT / EXACT / INFEASIBLE (U2K / U8K / M2K / M8K), scenario B CENSORED / UNSTABLE /
NT / EXACT, so every 030 8.7 mapping is exercised end to end. git is faked except in the tests of the git helpers,
which use a real repository."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import capacity_report as cr
import core_b_plan as bp
from capserve import profile as cp
from capserve import profile_gen as pg

SERVICE = Path(__file__).resolve().parent.parent
ROOT = SERVICE.parent
if str(ROOT / "tests") not in sys.path:      # the Stage C fake world lives with the research tests
    sys.path.insert(0, str(ROOT / "tests"))
import test_run_capacity as trc  # noqa: E402
from test_capacity_isolation import CAPACITY_MODULES  # noqa: E402
from test_run_capacity import COMMIT, CORE_UNIFORM, PINS_SHA, World, datasets, frozen, make_plan, run  # noqa: E402

SID = "formal-20261101-c1"
FREEZE = "fe" * 20
BEFORE = "be" * 20         # the commit before the freeze that last touched capacity.toml (unfrozen)
GEN = "9a" * 20
CLEAN_HEAD = "7b" * 20
CAPACITY = "configs/capacity.toml"
# capacity.toml at the freeze commit. The configs are injected as dicts, so these bytes stand in for the frozen file:
# the test sessions record their sha256, as a real session records the sha256 of the bytes it ran.
FROZEN_TOML = b"[freeze]\ncapacity = true\n# the frozen capacity.toml of the test sessions (configs are injected)\n"
UNFROZEN_TOML = b"[freeze]\ncapacity = false\n"
CAP_SHA = hashlib.sha256(FROZEN_TOML).hexdigest()


def module_bytes(name: str) -> bytes:
    """A Stage C module as this generator imports it (the frozen bytes, unless a test says otherwise)."""
    return (ROOT / "benchmarks" / f"{name}.py").read_bytes()


def scenario_b(w, C, phase, block):
    """U2K passes everywhere (CENSORED at C_cap 70), U8K passes Phase S but fails every Phase C point (UNSTABLE), M2K's
    Phase S boots both fail (no points: NT), M8K TPOT 30 C ms (C 3 PASS, C 4 FAIL: EXACT 3)."""
    if w == "U2K":
        return {"short": 900, "tpot": 1, "burst": 3000}
    if w == "U8K":
        return {"short": 2500, "tpot": 10 if phase == "S" else 200}
    if w == "M2K":
        return {"short": 600, "long": 6300, "tpot": 20}
    return {"short": 1500, "long": 6300, "tpot": 30 * C}


class FakeGit:
    """The git answers profile_gen asks for: files at a commit (capacity.toml at the freeze commit, its first parent
    and the earlier commits of its history; the Stage C modules at the freeze commit and at the session's
    DEPLOYED_COMMIT), capacity.toml's history, ancestry, HEAD and the tracked-file status. files overrides or adds
    {(rev, path): bytes or None}; history is capacity.toml's commits up to the freeze commit, newest first."""

    def __init__(self, *, freeze=FREEZE, parent_frozen=False, ancestor=True, head=CLEAN_HEAD, dirty="", files=None,
                 history=None):
        self.freeze, self.ancestor = freeze, ancestor
        self.head, self.dirty = head, dirty
        self.files = {(freeze, CAPACITY): FROZEN_TOML, (f"{freeze}^", CAPACITY):
                      FROZEN_TOML if parent_frozen else UNFROZEN_TOML, (BEFORE, CAPACITY): UNFROZEN_TOML}
        self.files.update({(rev, f"benchmarks/{m}.py"): module_bytes(m) for rev in (freeze, COMMIT)
                           for m in CAPACITY_MODULES})
        self.files.update(files or {})
        self.history = [freeze, BEFORE] if history is None else history
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "show":
            rev, _, path = args[1].partition(":")
            return self.files.get((rev, path))
        if args[:2] == ("rev-list", "--full-history"):
            assert args[3:] == ("--", CAPACITY)
            return "".join(f"{c}\n" for c in self.history).encode() if args[2] == self.freeze else None
        if args[:2] == ("merge-base", "--is-ancestor"):
            return b"" if self.ancestor and args[2:] == (self.freeze, COMMIT) else None
        if args == ("rev-parse", "HEAD"):
            return None if self.head is None else self.head.encode() + b"\n"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return self.dirty.encode()
        raise AssertionError(args)


def make_session(tmp: Path, **world) -> dict:
    """A formal session at <tmp>/repo/results/capacity/<SID> with its analysis.json, the uniform session the plan
    derives from at results/core_b/, and the freeze evidence (passed smokes, C0's log) under the same repo root."""
    cfg = tomllib.loads((ROOT / "configs" / "capacity.toml").read_text())
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    ds, sha = datasets(tmp)
    c, p = frozen(cfg, pins, sha)
    repo = tmp / "repo"
    configs = (c, p, CAP_SHA, PINS_SHA)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trc, "CAP_SHA", CAP_SHA)          # the session records the sha256 of FROZEN_TOML
        out, _, _ = run(World(tmp, **world), make_plan(c, p, formal=True), c, p, mixed_datasets=ds,
                        out=repo / "results" / "capacity" / SID)
    (repo / "results" / "core_b").mkdir(parents=True)
    (repo / "results" / "core_b" / CORE_UNIFORM.name).symlink_to(CORE_UNIFORM)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cr, "ROOT", repo)
        mp.setattr(cr, "load_configs", lambda: configs)
        assert cr.main(["--session", str(out), "--uniform-session", str(CORE_UNIFORM),
                        "--out", str(out / "analysis.json")]) == 0
    return {"repo": repo, "out": out, "configs": configs, "cfg": c, "pins": p}


@pytest.fixture(scope="module")
def sess_a(tmp_path_factory):
    return make_session(tmp_path_factory.mktemp("gen-a"))


@pytest.fixture(scope="module")
def sess_b(tmp_path_factory):
    return make_session(tmp_path_factory.mktemp("gen-b"), truth=scenario_b,
                        boots={"S-M2K-a1": "not_ready", "S-M2K-a2": "not_ready"})


def clone(s: dict, tmp_path: Path) -> dict:
    """A private copy of a session's repo root, to break on disk."""
    repo = tmp_path / "repo"
    shutil.copytree(s["repo"], repo, symlinks=True)
    return {**s, "repo": repo, "out": repo / "results" / "capacity" / SID}


def gen(s: dict, **kw) -> dict:
    kw.setdefault("freeze_commit", FREEZE)
    kw.setdefault("generator_commit", GEN)
    kw.setdefault("root", s["repo"])
    kw.setdefault("configs", s["configs"])
    kw.setdefault("git", FakeGit())
    return pg.generate(kw.pop("session", s["out"]), **kw)


def refused(s: dict, **kw) -> str:
    with pytest.raises(pg.GenerationError) as exc:
        gen(s, **kw)
    return str(exc.value)


def classes(doc: dict) -> dict:
    return {c["name"]: c for c in doc["classes"]}


def summary(c: dict) -> tuple:
    return (c["status"], c["max_prompt_tokens"], c["max_output_tokens"], c["max_concurrency"], c["max_queue"],
            c["bound"], c["source"])


# ------------------------------------------------------------------ 030 8.7 mapping, end to end --

def test_exact_lower_infeasible_map_to_verified_numbers_and_a_verified_0(sess_a):
    doc = gen(sess_a)
    got = classes(doc)
    assert list(got) == ["uniform-2k", "uniform-8k", "mixed-2k", "mixed-8k"]
    assert summary(got["uniform-2k"]) == ("verified", 1920, 128, 16, 16, "lower", "stage_c:U2K:c_prod:LOWER")
    assert summary(got["uniform-8k"]) == ("verified", 8064, 128, 7, 7, "exact", "stage_c:U8K:c_prod:EXACT")
    assert summary(got["mixed-2k"]) == ("verified", 32640, 128, 1, 1, "exact", "stage_c:M2K:c_prod:EXACT")
    assert summary(got["mixed-8k"]) == ("verified", 32640, 128, 0, 0, "exact", "stage_c:M8K:c_prod:INFEASIBLE")
    assert all(c["unavailable_reason"] is None for c in got.values())


def test_each_class_carries_its_stage_c_evidence(sess_a):
    analysis = json.loads((sess_a["out"] / "analysis.json").read_text())
    got = classes(gen(sess_a))
    for name, w in (("uniform-2k", "U2K"), ("uniform-8k", "U8K"), ("mixed-2k", "M2K"), ("mixed-8k", "M8K")):
        x, e = analysis["workloads"][w], got[name]["evidence"]
        assert e["workload"] == w and e["c_prod"] == x["c_prod"] and e["c_slo"] == x["c_slo"]
        assert (e["c_kv"], e["c_cap"], e["flags"], e["binding"]) == (x["c_kv"], x["c_cap"], x["flags"], x["binding"])
        assert e["margins"] == x["margins"]["c_prod"]
    u2 = got["uniform-2k"]["evidence"]
    assert u2["c_slo"] == {"value": 33, "status": "EXACT"} and u2["flags"] == ["STRICT_COARSE"]
    assert u2["binding"] == ["tpot"] and (u2["c_kv"], u2["c_cap"]) == (35, 70)
    assert u2["burst_max_ttft_ms"] == 8000.0                     # the start-up wave at C_PROD 16 (D1)
    assert "1920-token" in u2["scope"] and "steady state" in u2["scope"] and "after admission" in u2["scope"]
    m8 = got["mixed-8k"]["evidence"]
    assert m8["binding"] == ["long"] and m8["burst_max_ttft_ms"] is None and m8["margins"] is None
    assert "32640-token" in got["mixed-2k"]["evidence"]["scope"]


def test_censored_unstable_nt_map_to_a_censored_number_and_unavailable_classes(sess_b):
    got = classes(gen(sess_b))
    assert summary(got["uniform-2k"]) == ("verified", 1920, 128, 70, 70, "censored", "stage_c:U2K:c_prod:CENSORED")
    assert got["uniform-2k"]["evidence"]["flags"] == ["ABOVE_CKV"]
    assert got["uniform-2k"]["evidence"]["burst_max_ttft_ms"] == 3000.0
    assert summary(got["mixed-8k"]) == ("verified", 32640, 128, 3, 3, "exact", "stage_c:M8K:c_prod:EXACT")
    for name, w, status, why in (("uniform-8k", "U8K", "UNSTABLE", "C 13"),
                                 ("mixed-2k", "M2K", "NT", "no confirmation")):
        c = got[name]
        assert summary(c) == ("unavailable", c["max_prompt_tokens"], 128, None, None, None,
                              f"stage_c:{w}:c_prod:{status}")
        assert status in c["unavailable_reason"] and why in c["unavailable_reason"]
        assert "never a number" in c["unavailable_reason"]
        assert c["evidence"]["c_prod"] == {"value": None, "status": status}
        assert c["evidence"]["burst_max_ttft_ms"] is None


def test_the_profile_states_the_slo_server_and_provenance(sess_a):
    doc = gen(sess_a)
    raw = (sess_a["out"] / "analysis.json").read_bytes()
    slo, pins = sess_a["cfg"]["slo"], sess_a["pins"]
    assert (doc["schema_version"], doc["profile_id"], doc["status"], doc["provisional_reason"]) == (
        1, f"verified-{SID}", "verified", None)
    assert doc["slo"] == {"id": "S2", "terms": {k: v for k, v in slo.items() if k != "id"}}       # verbatim
    s = json.loads((sess_a["out"] / "session.json").read_text())
    rec = s["runs"][s["boots"][0]["tag"]]
    assert doc["server"] == {
        "vllm_version": "0.30.0", "model_repo": pins["core_b"]["repo"], "model_revision": pins["core_b"]["revision"],
        "fingerprint_sha256": cp.server_fingerprint(rec["argv"], {k: rec["env"][k] for k in bp.recorded_env({})}),
        "num_gpu_blocks": 248, "block_size": 528}
    assert doc["server"]["fingerprint_sha256"] == cp.stage_b_fingerprint(pins)   # the server deploy starts
    assert doc["provenance"] == {
        "source_session": SID, "analysis_path": f"results/capacity/{SID}/analysis.json",
        "analysis_sha256": hashlib.sha256(raw).hexdigest(), "prereg_section": "§8", "freeze_commit": FREEZE,
        "generator_commit": GEN, "deployed_commit": COMMIT, "capacity_sha256": CAP_SHA, "pins_sha256": PINS_SHA,
        "gpu": "NVIDIA GeForce RTX 5070 Ti", "gpu_driver": "617.42"}
    assert doc["content_sha256"] == cp.content_sha256(doc)


@pytest.mark.parametrize("which", ["sess_a", "sess_b"])
def test_the_generated_profile_passes_the_loader_as_verified(which, request):
    s = request.getfixturevalue(which)
    prof = cp.parse_profile(pg.render(gen(s)), source="generated", allow_provisional=False, pins=s["pins"],
                            frozen=cp.FrozenConfigs(s["cfg"], CAP_SHA, PINS_SHA))
    assert prof.verified and len(prof.classes) == 4


def test_the_output_is_deterministic_and_only_the_generator_commit_varies(sess_a):
    a, b = pg.render(gen(sess_a)), pg.render(gen(sess_a))
    assert a == b and a.endswith(b"\n") and json.loads(a) == gen(sess_a)
    other = gen(sess_a, generator_commit="1c" * 20)
    mine = gen(sess_a)
    diff = {k for k in mine if mine[k] != other[k]}
    assert diff == {"provenance", "content_sha256"}
    assert {k for k in mine["provenance"] if mine["provenance"][k] != other["provenance"][k]} == {"generator_commit"}


def test_an_unavailable_class_never_gets_a_fallback_number(sess_b):
    """030 8.7: UNSTABLE / NT give no number at all, never C_SLO, Ĉ, C_KV or a provisional value."""
    c = classes(gen(sess_b))["uniform-8k"]
    assert (c["max_concurrency"], c["max_queue"], c["bound"]) == (None, None, None)
    prof = cp.parse_profile(pg.render(gen(sess_b)), source="g", allow_provisional=False, pins=sess_b["pins"],
                            frozen=cp.FrozenConfigs(sess_b["cfg"], CAP_SHA, PINS_SHA))
    with pytest.raises(cp.ProfileError):
        prof.serving_class("uniform-8k")


# ------------------------------------------------------------------ refusals (exit 2, nothing written) --

def test_refused_before_the_freeze(sess_a):
    c = copy.deepcopy(sess_a["cfg"])
    c["freeze"]["capacity"] = False
    msg = refused(sess_a, configs=(c, *sess_a["configs"][1:]))
    assert "[freeze] capacity" in msg


def test_refused_without_the_freeze_evidence(sess_a, tmp_path):
    s = clone(sess_a, tmp_path)
    shutil.rmtree(s["repo"] / "results" / "capacity" / "smoke-mini-t")
    msg = refused(s)
    assert "freeze evidence" in msg and "mini" in msg


@pytest.mark.parametrize("commit, git, match", [
    ("fe" * 10, FakeGit(), "40 lowercase hex"),
    ("FE" * 20, FakeGit(), "40 lowercase hex"),
    ("ab" * 20, FakeGit(), "no readable configs/capacity.toml"),
    (FREEZE, FakeGit(parent_frozen=True), "already frozen"),
    (FREEZE, FakeGit(ancestor=False), "DEPLOYED_COMMIT"),
])
def test_the_freeze_commit_must_be_the_commit_that_set_the_freeze(sess_a, commit, git, match):
    assert match in refused(sess_a, freeze_commit=commit, git=git)


def test_a_freeze_commit_whose_capacity_toml_is_not_frozen_is_refused(sess_a):
    assert "not the freeze commit" in refused(sess_a, git=FakeGit(files={(FREEZE, CAPACITY): UNFROZEN_TOML}))


OTHER_FREEZE = (b'[freeze]\ncapacity = true\n[capacity]\nc0_branch = "B"\nguard_band = 0.10\n'
                b'[slo]\nid = "S1"\ntpot_p95_ms = 50\n')


def test_a_capacity_toml_changed_after_the_freeze_is_refused(sess_a):
    """The freeze commit's capacity.toml must be the bytes the session ran and the generator re-runs with (029:
    the frozen Stage C rules), not another frozen config (review finding: only the freeze flag was compared)."""
    msg = refused(sess_a, git=FakeGit(files={(FREEZE, CAPACITY): OTHER_FREEZE}))
    assert "freeze commit" in msg and CAPACITY in msg and "changed after the freeze" in msg


@pytest.mark.parametrize("module", CAPACITY_MODULES)
def test_a_stage_c_module_changed_after_the_freeze_is_refused(sess_a, module):
    path = f"benchmarks/{module}.py"
    msg = refused(sess_a, git=FakeGit(files={(COMMIT, path): module_bytes(module) + b"# edited after the freeze\n"}))
    assert path in msg and "DEPLOYED_COMMIT" in msg and "frozen" in msg
    msg = refused(sess_a, git=FakeGit(files={(FREEZE, path): module_bytes(module) + b"# not what runs now\n"}))
    assert path in msg and "imported by this generator" in msg      # the analysis would be re-run by other code
    assert "no " + path in refused(sess_a, git=FakeGit(files={(FREEZE, path): None}))


class NoHistory(FakeGit):
    def __call__(self, *args):
        if args[:2] == ("rev-list", "--full-history"):
            return None
        return super().__call__(*args)


def test_a_refreeze_after_an_unfreeze_is_refused(sess_a):
    """The freeze commit is the first one that set [freeze] capacity = true: an unfreeze / change / re-freeze is
    not the preregistered freeze (030 8.9), even when the re-freeze commit's bytes are the ones that ran."""
    first, unfrozen = "f1" * 20, "f2" * 20
    git = FakeGit(history=[FREEZE, unfrozen, first, BEFORE],
                  files={(first, CAPACITY): OTHER_FREEZE, (unfrozen, CAPACITY): UNFROZEN_TOML})
    msg = refused(sess_a, git=git)
    assert "already frozen at " + first in msg and "re-freeze" in msg
    assert "cannot list" in refused(sess_a, git=NoHistory())


def test_the_frozen_modules_are_the_stage_c_modules():
    """The modules whose bytes must not change after the freeze are test_capacity_isolation's Stage C modules."""
    assert [m.__name__ for m in pg.STAGE_C_MODULES] == list(CAPACITY_MODULES)


@pytest.mark.parametrize("edit, match", [
    (lambda s: s.update(formal=False), "not formal"),
    (lambda s: s.update(mode="mini"), "mode"),
    (lambda s: s.update(env_valid=False), "env-invalid"),
    (lambda s: s.update(stage="C1-mini"), "stage"),
    (lambda s: s.update(aborted="KeyboardInterrupt: "), "aborted"),
    (lambda s: s.update(session_id="pilot-20261101-c1"), "formal-"),
])
def test_refused_unless_the_session_is_formal_env_valid_c1(sess_a, tmp_path, edit, match):
    s = clone(sess_a, tmp_path)
    path = s["out"] / "session.json"
    doc = json.loads(path.read_text())
    edit(doc)
    path.write_text(json.dumps(doc))
    assert match in refused(s)


def test_refused_unless_the_session_is_results_capacity_formal_id(sess_a, tmp_path):
    s = clone(sess_a, tmp_path)
    other = s["repo"] / "results" / "capacity" / "formal-20261101-c1-copy"
    shutil.copytree(s["out"], other)
    assert "results/capacity/<session_id>" in refused(s, session=other)
    nested = s["repo"] / "results" / "capacity" / "x" / SID
    shutil.copytree(s["out"], nested)
    assert "results/capacity/<session_id>" in refused(s, session=nested)
    outside = tmp_path / "elsewhere" / SID
    shutil.copytree(s["out"], outside)
    assert "results/capacity" in refused(s, session=outside)


def test_refused_when_the_session_cannot_be_read(sess_a, tmp_path):
    assert "cannot read the session" in refused(sess_a, session=tmp_path / "missing")


def test_refused_without_an_analysis_json(sess_a, tmp_path):
    s = clone(sess_a, tmp_path)
    (s["out"] / "analysis.json").write_text("[]\n")
    assert "not a JSON object" in refused(s)
    (s["out"] / "analysis.json").unlink()
    assert "analysis.json" in refused(s) and "run capacity_report" in refused(s)


def test_an_analysis_json_with_other_bytes_is_refused_even_with_the_same_content(sess_a, tmp_path):
    """021 R3: the frozen Stage C modules must reproduce analysis.json byte for byte."""
    s = clone(sess_a, tmp_path)
    path = s["out"] / "analysis.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2) + "\n")
    msg = refused(s)
    assert "byte for byte" in msg


def test_an_edited_analysis_json_is_refused(sess_a, tmp_path):
    s = clone(sess_a, tmp_path)
    path = s["out"] / "analysis.json"
    doc = json.loads(path.read_text())
    doc["workloads"]["U2K"]["c_prod"]["value"] = 33
    path.write_text(json.dumps(doc, indent=1) + "\n")
    msg = refused(s)
    assert "byte for byte" in msg and "workloads" in msg


ADV = ROOT / "results" / "advanced" / pg.ADVANCED_SESSION


@pytest.mark.skipif(not (ADV / "derived" / "B0-2048-uniform-r1-requests.jsonl").is_file(),
                    reason="Advanced derived files not present")
def test_an_analysis_made_with_the_advanced_references_is_reproduced_with_them(sess_a, tmp_path):
    """capacity_report's optional --advanced-session (CROSS_SESSION_SHIFT, descriptive) changes analysis.json, so the
    generator re-runs it with the Advanced session's references when analysis.json says it used them."""
    s = clone(sess_a, tmp_path)
    (s["out"] / "analysis.json").unlink()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cr, "ROOT", s["repo"])
        mp.setattr(cr, "load_configs", lambda: s["configs"])
        assert cr.main(["--session", str(s["out"]), "--uniform-session", str(CORE_UNIFORM), "--advanced-session",
                        str(ADV), "--out", str(s["out"] / "analysis.json")]) == 0
    msg = refused(s)
    assert "Advanced references" in msg
    (s["repo"] / "results" / "advanced").mkdir()
    (s["repo"] / "results" / "advanced" / ADV.name).symlink_to(ADV)
    doc = gen(s)
    assert doc["provenance"]["analysis_sha256"] == hashlib.sha256((s["out"] / "analysis.json").read_bytes()).hexdigest()
    assert classes(doc)["uniform-2k"]["max_concurrency"] == 16


def test_refused_with_every_capacity_report_refusal(sess_a, tmp_path):
    s = clone(sess_a, tmp_path)
    rows = s["out"] / "rows.jsonl"
    rows.write_text(rows.read_text() + "\n")
    assert "capacity_report" in refused(s) and "rows.jsonl" in refused(s)


def test_a_malformed_session_is_refused_not_a_crash(sess_a, tmp_path, monkeypatch, capsys):
    s = clone(sess_a, tmp_path)
    rows = s["out"] / "rows.jsonl"
    rows.write_text(rows.read_text() + "1\n")                      # a row that is not an object
    assert "malformed" in refused(s)
    out = tmp_path / "verified.json"
    assert cli(s, monkeypatch, out) == 2
    assert not out.exists() and "malformed" in capsys.readouterr().err


def test_refused_with_another_slo(sess_a):
    c = copy.deepcopy(sess_a["cfg"])
    c["slo"]["id"] = "S3"
    msg = refused(sess_a, configs=(c, *sess_a["configs"][1:]))
    assert "SLO" in msg and "S3" in msg


def test_refused_without_the_uniform_session(sess_a, tmp_path):
    s = clone(sess_a, tmp_path)
    (s["repo"] / "results" / "core_b" / CORE_UNIFORM.name).unlink()
    assert "uniform session" in refused(s)


def test_refused_with_other_config_bytes(sess_a):
    assert "capacity.toml" in refused(sess_a, configs=(*sess_a["configs"][:2], "00" * 32, PINS_SHA))


def test_the_session_fingerprint_is_the_recorded_server_and_must_be_the_product_server(sess_a):
    s = json.loads((sess_a["out"] / "session.json").read_text())
    fp, problems = pg.session_fingerprint(s, sess_a["pins"])
    assert problems == [] and fp == cp.stage_b_fingerprint(sess_a["pins"])
    other = copy.deepcopy(s)
    other["runs"][other["boots"][0]["tag"]]["argv"].append("--enforce-eager")
    fp, problems = pg.session_fingerprint(other, sess_a["pins"])
    assert fp != cp.stage_b_fingerprint(sess_a["pins"]) and "product server" in problems[0]
    other = copy.deepcopy(s)
    other["runs"][other["boots"][0]["tag"]]["env"].pop("CUDA_HOME")
    assert "CUDA_HOME" in pg.session_fingerprint(other, sess_a["pins"])[1][0]
    other["boots"] = []
    assert "no server argv" in pg.session_fingerprint(other, sess_a["pins"])[1][0]


# ------------------------------------------------------------------ git helpers (a real repository) --

def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def commit_files(root, files: dict) -> str:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", ", ".join(files))
    return git(root, "rev-parse", "HEAD").strip()


STAGE_C_FILES = {f"benchmarks/{m}.py": module_bytes(m) for m in CAPACITY_MODULES}
EDITED_ANALYSIS = {"benchmarks/capacity_analysis.py": module_bytes("capacity_analysis")
                   + b"\nACCEPT_PROD = ('PASS', 'MARGINAL')   # a rule changed after the freeze\n"}
REFROZEN_TOML = b"[freeze]\ncapacity = true\n[capacity]\nguard_band = 0.0\n"


@pytest.fixture
def repo(tmp_path):
    """before (unfrozen) -> freeze -> deployed (an unrelated commit): the Stage C modules are this generator's."""
    root = tmp_path / "r"
    root.mkdir()
    git(root, "init", "-q")
    before = commit_files(root, {CAPACITY: UNFROZEN_TOML, **STAGE_C_FILES})
    freeze = commit_files(root, {CAPACITY: FROZEN_TOML})
    deployed = commit_files(root, {"x.txt": b"after the freeze\n"})
    return {"root": root, "before": before, "freeze": freeze, "deployed": deployed}


def problems_at(repo, commit, deployed, capacity_sha256=CAP_SHA):
    return pg.freeze_commit_problems(commit, deployed_commit=deployed, capacity_sha256=capacity_sha256,
                                     git=pg.git_runner(repo["root"]))


def test_freeze_commit_problems_with_a_real_repository(repo):
    assert problems_at(repo, repo["freeze"], repo["deployed"]) == []
    assert "not the freeze commit" in problems_at(repo, repo["before"], repo["deployed"])[0]
    assert "already frozen" in problems_at(repo, repo["deployed"], repo["deployed"])[0]
    assert "DEPLOYED_COMMIT" in problems_at(repo, repo["freeze"], repo["before"])[0]
    assert "no readable" in problems_at(repo, "0" * 40, repo["deployed"])[0]


def test_a_stage_c_module_edited_after_the_freeze_is_refused_with_a_real_repository(repo):
    """Review finding: the session's DEPLOYED_COMMIT ran a capacity_analysis edited after the freeze."""
    edited = commit_files(repo["root"], EDITED_ANALYSIS)
    msg = "; ".join(problems_at(repo, repo["freeze"], edited))
    assert "benchmarks/capacity_analysis.py" in msg and "DEPLOYED_COMMIT" in msg
    assert problems_at(repo, repo["freeze"], repo["deployed"]) == []      # an unrelated later commit is fine


def test_an_unfreeze_change_and_refreeze_is_refused_with_a_real_repository(repo):
    """Review finding (the probe): freeze, unfreeze, change the rules, re-freeze; the session ran the re-frozen code.
    Neither the first freeze (its bytes are not what ran) nor the re-freeze (not the first freeze) is accepted."""
    commit_files(repo["root"], {CAPACITY: UNFROZEN_TOML})
    refrozen = commit_files(repo["root"], {CAPACITY: REFROZEN_TOML, **EDITED_ANALYSIS})
    ran = hashlib.sha256(REFROZEN_TOML).hexdigest()
    msg = "; ".join(problems_at(repo, repo["freeze"], refrozen, ran))
    assert "changed after the freeze" in msg and "benchmarks/capacity_analysis.py" in msg
    msg = "; ".join(problems_at(repo, refrozen, refrozen, ran))
    assert f"already frozen at {repo['freeze']}" in msg and "re-freeze" in msg


def test_a_refreeze_with_unchanged_code_is_still_not_the_freeze_with_a_real_repository(repo):
    commit_files(repo["root"], {CAPACITY: UNFROZEN_TOML})
    refrozen = commit_files(repo["root"], {CAPACITY: REFROZEN_TOML})
    problems = problems_at(repo, refrozen, refrozen, hashlib.sha256(REFROZEN_TOML).hexdigest())
    assert len(problems) == 1 and f"already frozen at {repo['freeze']}" in problems[0]


def test_the_generator_commit_is_head_of_a_clean_tree(repo):
    run_git = pg.git_runner(repo["root"])
    assert pg.generator_commit(run_git) == (repo["deployed"], [])
    (repo["root"] / "x.txt").write_text("hot patch\n")
    sha, problems = pg.generator_commit(run_git)
    assert sha == repo["deployed"] and "uncommitted" in problems[0]
    (repo["root"] / "x.txt").write_text("after the freeze\n")
    (repo["root"] / "untracked.json").write_text("{}")             # e.g. the profile about to be written
    assert pg.generator_commit(run_git) == (repo["deployed"], [])
    assert "git rev-parse HEAD" in pg.generator_commit(pg.git_runner(repo["root"] / "nowhere"))[1][0]


# ------------------------------------------------------------------ CLI --

def cli_root(tmp_path, capacity_text):
    """A repo root with configs/capacity.toml (unless None) and the repo's pins.toml."""
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "pins.toml", root / "configs" / "pins.toml")
    if capacity_text is not None:
        (root / "configs" / "capacity.toml").write_text(capacity_text)
    return root


@pytest.mark.parametrize("text", [None, "", "[slo]\nid = 'S2'\n", "[freeze]\n", "[freeze]\ncapacity = false\n",
                                  "[freeze]\ncapacity = 'true'\n", "[freeze]\ncapacity = 1\n"])
def test_main_refuses_until_capacity_is_frozen(tmp_path, capsys, text):
    out = tmp_path / "profile.json"
    code = pg.main(["--session", f"results/capacity/{SID}", "--freeze-commit", FREEZE, "--out", str(out)],
                   root=cli_root(tmp_path, text), git=FakeGit())
    err = capsys.readouterr().err
    assert code == 2 and not out.exists()
    assert "capacity.toml" in err and "[freeze] capacity" in err


def test_main_refuses_an_unreadable_capacity_toml(tmp_path, capsys):
    out = tmp_path / "profile.json"
    code = pg.main(["--session", "s", "--freeze-commit", FREEZE, "--out", str(out)],
                   root=cli_root(tmp_path, "[freeze\n"), git=FakeGit())
    assert code == 2 and not out.exists() and "cannot read" in capsys.readouterr().err


def cli(s, monkeypatch, out, *, git=None, freeze=FREEZE):
    monkeypatch.setattr(pg, "load_configs", lambda root: s["configs"])
    return pg.main(["--session", str(s["out"]), "--freeze-commit", freeze, "--out", str(out)], root=s["repo"],
                   git=git or FakeGit())


def test_main_writes_the_profile_and_prints_a_summary(sess_a, tmp_path, monkeypatch, capsys):
    out = tmp_path / "verified.json"
    assert cli(sess_a, monkeypatch, out) == 0
    assert out.read_bytes() == pg.render(gen(sess_a, generator_commit=CLEAN_HEAD))
    printed = json.loads(capsys.readouterr().out)
    assert printed["profile_id"] == f"verified-{SID}" and printed["out"] == str(out)
    assert printed["classes"]["uniform-2k"] == ["verified", 16, "lower"]
    assert printed["classes"]["mixed-8k"] == ["verified", 0, "exact"]


@pytest.mark.parametrize("git, match", [
    (FakeGit(dirty=" M service/capserve/profile_gen.py\n"), "uncommitted"),
    (FakeGit(head=None), "git rev-parse HEAD"),
    (FakeGit(ancestor=False), "DEPLOYED_COMMIT"),
])
def test_main_refuses_and_writes_nothing(sess_a, tmp_path, monkeypatch, capsys, git, match):
    out = tmp_path / "verified.json"
    assert cli(sess_a, monkeypatch, out, git=git) == 2
    assert not out.exists() and match in capsys.readouterr().err


def test_main_never_overwrites_and_never_writes_under_results(sess_a, tmp_path, monkeypatch, capsys):
    out = tmp_path / "verified.json"
    out.write_text("keep")
    assert cli(sess_a, monkeypatch, out) == 2
    assert out.read_text() == "keep" and "exists" in capsys.readouterr().err
    research = sess_a["repo"] / "results" / "capacity" / SID / "profile.json"
    assert cli(sess_a, monkeypatch, research) == 2
    assert not research.exists() and "results/" in capsys.readouterr().err


def test_no_override_options():
    """021 R1: the only inputs are the session, the freeze commit (verified against git) and the output path."""
    assert pg.option_names() == {"--session", "--out", "--freeze-commit"}


def test_default_paths_are_the_repo():
    assert pg.ROOT == ROOT and pg.CAPACITY_TOML == ROOT / "configs" / "capacity.toml"


def test_module_entry_point(tmp_path):
    env = {**os.environ, "PYTHONPATH": f"{SERVICE}{os.pathsep}{ROOT / 'benchmarks'}"}
    r = subprocess.run([sys.executable, "-m", "capserve.profile_gen", "--session", "s", "--freeze-commit", FREEZE,
                        "--out", str(tmp_path / "p.json")], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=60)
    assert r.returncode == 2 and "refused" in r.stderr and not (tmp_path / "p.json").exists()   # frozen or not


# ------------------------------------------------------------------ committed profiles (029) --

def _regenerable(doc: dict) -> dict:
    """The profile without what a regeneration may change (029: generator_commit, and the hash over it)."""
    out = {k: v for k, v in doc.items() if k != "content_sha256"}
    out["provenance"] = {k: v for k, v in (doc.get("provenance") or {}).items() if k != "generator_commit"}
    return out


def tracked_verified_profiles(root) -> list:
    """Every tracked file outside results/ that is a JSON document with status "verified", whatever its path or
    extension (git ls-files: what deploy_bundle puts in a release's DEPLOY_MANIFEST.json)."""
    names = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    found = []
    for rel in sorted(n for n in names.decode().split("\0") if n and not n.startswith("results/")):
        path = Path(root) / rel
        try:
            data = path.read_bytes()
            doc = json.loads(data) if b'"verified"' in data else None
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("status") == "verified":
            found.append(path)
    return found


def test_the_scan_finds_a_verified_profile_at_any_tracked_path(tmp_path):
    """Review finding: the regeneration test scanned only service/profiles/*.json."""
    root = tmp_path / "r"
    root.mkdir()
    git(root, "init", "-q")
    doc = json.dumps({"status": "verified", "profile_id": "x"}).encode()
    commit_files(root, {"service/profiles/verified/a.json": doc, "service/deploy/b.json": doc,
                        "service/profiles/c.JSON": doc, "d.txt": doc, "results/capacity/s/e.json": doc,
                        "service/profiles/provisional.json": json.dumps({"status": "provisional"}).encode(),
                        "x.py": b'status = "verified"\n'})
    (root / "untracked.json").write_bytes(doc)
    got = [p.relative_to(root).as_posix() for p in tracked_verified_profiles(root)]
    assert got == ["d.txt", "service/deploy/b.json", "service/profiles/c.JSON", "service/profiles/verified/a.json"]


def test_every_committed_verified_profile_is_what_the_generator_writes():
    assert sorted((SERVICE / "profiles").glob("*.json"))
    for path in tracked_verified_profiles(ROOT):
        doc = json.loads(path.read_text())
        prov = doc.get("provenance") or {}
        session = ROOT / "results" / "capacity" / str(prov.get("source_session"))
        try:
            regenerated = pg.generate(session, freeze_commit=str(prov.get("freeze_commit")),
                                      generator_commit=str(prov.get("generator_commit")))
        except pg.GenerationError as exc:
            pytest.fail(f"{path.name} is verified, but the generator refuses it: {exc}")
        assert _regenerable(regenerated) == _regenerable(doc), path.name
