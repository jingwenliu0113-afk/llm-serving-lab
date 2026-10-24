"""Tests for capserve/profile_gen.py (decision 029 "產生器", M2; decision 030 8.7): the verified-profile generator.

The Stage C sessions are written by the real runner (run_capacity) in the fake world of tests/test_run_capacity.py,
and their analysis.json by the real capacity_report CLI, so every input is schema-faithful: nothing is hand-written.
Scenario A gives C_PROD LOWER / EXACT / EXACT / INFEASIBLE (U2K / U8K / M2K / M8K), scenario B CENSORED / UNSTABLE /
NT / EXACT, so every 030 8.7 mapping is exercised end to end. git is faked except in the tests of the git helpers,
which use a real repository.

Stage D (decision 043, PREREGISTRATION §9) is tested on the committed formal sessions (results/stage_d/: D-A1b and
D-A2 give profiles; D-B, D-C and the aborted D-A1 never do) with this repository's own git, so the freeze commit, the
sessions' DEPLOYED_COMMIT and the frozen files are the real ones (RepoGit only fakes HEAD / the tree status, or one
answer a test breaks). A session broken on disk is a copy under a private repo root (d_root)."""

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
import stage_d_plan as sdplan
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
    """021 R1: the only inputs are the session, the freeze commit (verified against git) and the output path (043: a
    file for Stage C, a directory for every profile of a Stage D session; no server selector)."""
    assert pg.option_names() == {"--session", "--out", "--out-dir", "--freeze-commit"}


def test_default_paths_are_the_repo():
    assert pg.ROOT == ROOT and pg.CAPACITY_TOML == ROOT / "configs" / "capacity.toml"
    assert pg.STAGE_D_TOML == ROOT / "configs" / "stage_d.toml"


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


def regenerate(doc: dict) -> dict:
    """What the generator writes for a committed verified profile, by its prereg_section (043 step 5): §8 from its
    results/capacity session (generate); §9 the profile with its profile_id among those of its results/stage_d
    session (generate_stage_d)."""
    prov = doc.get("provenance") or {}
    sid, kw = str(prov.get("source_session")), {"freeze_commit": str(prov.get("freeze_commit")),
                                                "generator_commit": str(prov.get("generator_commit"))}
    if prov.get("prereg_section") == cp.STAGE_D_SECTION:
        docs = pg.generate_stage_d(ROOT / "results" / "stage_d" / sid, **kw)
        if doc.get("profile_id") not in docs:
            raise pg.GenerationError([f"{sid} gives the profiles {list(docs)}, not {doc.get('profile_id')!r}"])
        return docs[doc["profile_id"]]
    return pg.generate(ROOT / "results" / "capacity" / sid, **kw)


def test_every_committed_verified_profile_is_what_the_generator_writes():
    assert sorted((SERVICE / "profiles").glob("*.json"))
    for path in tracked_verified_profiles(ROOT):
        doc = json.loads(path.read_text())
        try:
            regenerated = regenerate(doc)
        except pg.GenerationError as exc:
            pytest.fail(f"{path.name} is verified, but the generator refuses it: {exc}")
        assert _regenerable(regenerated) == _regenerable(doc), path.name


def test_the_pinned_c1b_profile_regenerates_byte_for_byte_but_its_generator_commit():
    """043 §3 / §9: the Stage D schema and generator change nothing a §8 profile is written with."""
    path = SERVICE / "profiles" / "verified-formal-20261017-c1b.json"
    doc = json.loads(path.read_text())
    assert doc["provenance"]["prereg_section"] == cp.PREREG_SECTION
    regenerated = regenerate(doc)
    regenerated["provenance"]["generator_commit"] = doc["provenance"]["generator_commit"]
    regenerated["content_sha256"] = cp.content_sha256(regenerated)
    assert pg.render(regenerated) == path.read_bytes()


# ------------------------------------------------------------------ Stage D (043 §1, §2; PREREGISTRATION §9) --

STAGE_D = ROOT / "results" / "stage_d"
D_A1B, D_A2, D_A1, D_B, D_C = (STAGE_D / f"formal-20261023-{x}" for x in ("d-a1b", "d-a2", "d-a1", "d-b", "d-c"))
D_FREEZE = "fd95164f19cb261ccdf54b22ae93359f74b8f6be"     # set configs/stage_d.toml [freeze] stage_d = true (039)
D_DEPLOYED = "0ec7b9208ea339b302cf43f8658829f4a9995545"   # the D-A sessions' DEPLOYED_COMMIT
STAGE_D_TOML = "configs/stage_d.toml"
STAGE_D_SHA = "be7909b1e98183c337182f9559953e2613cad96f06d03899815fc0af88cba2a5"
FROZEN_D, UNFROZEN_D = b"[freeze]\nstage_d = true\n", b"[freeze]\nstage_d = false\n"
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
S2_TERMS = {"ttft_p95_ms": {"2048": 2000, "8192": 6000, "32768": 20000}, "tpot_p95_ms": 100, "mixed_long": "budget"}
NOT_COVERED = ["prompt_length_variation", "via_gateway", "open_loop", "output_length_other_than_measured",
               "start_up_wave", "concurrent_classes"]                     # 043 §5: every D-A class
# 039 凍結: stage_d_plan.HARNESS_FILES, then the Stage C modules Stage D runs
D_FROZEN_PATHS = list(sdplan.HARNESS_FILES) + [f"benchmarks/{m}.py" for m in CAPACITY_MODULES]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RepoGit:
    """This repository's own git, so the Stage D freeze commit, the sessions' DEPLOYED_COMMIT and the frozen files are
    the real ones; HEAD is CLEAN_HEAD of a clean tree unless dirty is given. files overrides `git show` {(rev, path):
    bytes or None}, history overrides stage_d.toml's history up to the freeze commit (newest first), and ancestor=False
    answers no to every ancestry question."""

    def __init__(self, *, files=None, history=None, ancestor=True, dirty=""):
        self.real, self.files, self.history = pg.git_runner(ROOT), files or {}, history
        self.ancestor, self.dirty = ancestor, dirty

    def __call__(self, *args):
        if args[0] == "show":
            rev, _, path = args[1].partition(":")
            if (rev, path) in self.files:
                return self.files[(rev, path)]
        if args[:2] == ("rev-list", "--full-history") and self.history is not None:
            return "".join(f"{c}\n" for c in self.history).encode()
        if args[:2] == ("merge-base", "--is-ancestor") and not self.ancestor:
            return None
        if args == ("rev-parse", "HEAD"):
            return CLEAN_HEAD.encode() + b"\n"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return self.dirty.encode()
        return self.real(*args)


def gen_d(session=D_A1B, **kw) -> dict:
    kw.setdefault("freeze_commit", D_FREEZE)
    kw.setdefault("generator_commit", GEN)
    kw.setdefault("git", RepoGit())
    return pg.generate_stage_d(session, **kw)


def refused_d(session=D_A1B, **kw) -> str:
    with pytest.raises(pg.GenerationError) as exc:
        gen_d(session, **kw)
    return str(exc.value)


def main_d(session, out, *, option="--out-dir", root=ROOT, git=None, freeze=D_FREEZE) -> int:
    return pg.main(["--session", str(session), "--freeze-commit", freeze, option, str(out)], root=root,
                   git=git or RepoGit())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def t_boots(s: dict) -> list:
    return [b["tag"] for b in s["boots"] if sdplan.WORKLOADS[b["workload"]]["server"] == "T"]


def d_root(tmp_path: Path, session: Path = D_A1B) -> dict:
    """A private repo root with a copy of a Stage D session to break on disk: the repo's configs, and links to what the
    frozen report reads besides the session (the mini smokes, the uniform and Advanced sessions, c1b's lambda_ref)."""
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    for name in ("capacity.toml", "pins.toml", "stage_d.toml"):
        shutil.copy(ROOT / "configs" / name, root / "configs" / name)
    cfg = tomllib.loads((ROOT / STAGE_D_TOML).read_text())
    links = [("stage_d", cfg["freeze"][k]) for k in ("mini_closed", "mini_open", "mini_verify")]
    links += [("core_b", cfg["stage_d"]["derived_from"]), ("advanced", pg.ADVANCED_SESSION),
              ("capacity", cfg["stage_d"]["c1b_session"])]
    for kind, sid in links:
        (root / "results" / kind).mkdir(parents=True, exist_ok=True)
        (root / "results" / kind / sid).symlink_to(ROOT / "results" / kind / sid)
    out = root / "results" / "stage_d" / session.name
    shutil.copytree(session, out)
    return {"root": root, "out": out}


@pytest.fixture(scope="module")
def a1b():
    return gen_d(D_A1B)


@pytest.fixture(scope="module")
def a2():
    return gen_d(D_A2)


def by_name(docs: dict) -> dict:
    return {c["name"]: c for doc in docs.values() for c in doc["classes"]}


def rounded(margins: dict) -> dict:
    return {t: round(v, 3) for t, v in margins.items()}


# ---------------------------------------------------- the committed D-A sessions

def test_d_a1b_gives_a_b0_and_a_t_profile_and_the_sentinel_no_class(a1b):
    """043 §1.1 / §2: one profile per (session, server): U32K on B0, U8K-T and M8K-T on T; the U2K sentinel ran on B0
    but never becomes a class."""
    assert list(a1b) == ["verified-formal-20261023-d-a1b-b0", "verified-formal-20261023-d-a1b-t"]
    b0, t = a1b.values()
    assert [summary(c) for c in b0["classes"]] == [
        ("verified", 32640, 128, 2, 2, "exact", "stage_d:U32K:c_prod:EXACT")]
    assert [summary(c) for c in t["classes"]] == [
        ("verified", 8064, 128, 4, 4, "exact", "stage_d:U8K-T:c_prod:EXACT"),
        ("verified", 32640, 128, 4, 4, "exact", "stage_d:M8K-T:c_prod:EXACT")]
    assert list(by_name(a1b)) == ["uniform-32k", "uniform-8k-t", "mixed-8k-t"]


def test_each_stage_d_profile_states_its_own_server(a1b):
    """043 §1.4: T's classes carry T's fingerprint, never the first boot's (b000-sentinel-start-U2K, on B0), which Stage
    C's session_fingerprint would have stamped on them, and passed."""
    b0, t = a1b.values()
    assert b0["server"]["fingerprint_sha256"] == cp.stage_b_fingerprint(PINS) == \
        "98b9f41be7b75b6bf431e6c090124156360edfc929ccd6164001468da05b6d4f"
    assert t["server"]["fingerprint_sha256"] == cp.t_fingerprint(PINS) == \
        "564d27ecfc45f162c564185cf8c9b4566f84208e788808aa6d302187474dd7cc"
    assert (b0["server"]["long_prefill_token_threshold"], t["server"]["long_prefill_token_threshold"]) == (None, 512)
    s = read_json(D_A1B / "session.json")
    first, problems = pg.session_fingerprint(s, PINS)
    assert s["boots"][0]["tag"] == "b000-sentinel-start-U2K" and problems == []
    assert first == b0["server"]["fingerprint_sha256"] != t["server"]["fingerprint_sha256"]
    for doc in (b0, t):
        assert {k: v for k, v in doc["server"].items() if k not in ("fingerprint_sha256", "long_prefill_token_threshold")} \
            == {"vllm_version": "0.30.0", "model_repo": PINS["core_b"]["repo"],
                "model_revision": PINS["core_b"]["revision"], "num_gpu_blocks": 248, "block_size": 528}


def test_a_stage_d_profile_states_the_slo_and_its_provenance(a1b):
    raw = (D_A1B / "analysis.json").read_bytes()
    for suffix, doc in zip(("b0", "t"), a1b.values()):
        assert (doc["schema_version"], doc["profile_id"], doc["status"], doc["provisional_reason"]) == (
            1, f"verified-formal-20261023-d-a1b-{suffix}", "verified", None)
        assert doc["slo"] == {"id": "S2", "terms": S2_TERMS}
        assert doc["provenance"] == {
            "source_session": "formal-20261023-d-a1b",
            "analysis_path": "results/stage_d/formal-20261023-d-a1b/analysis.json", "analysis_sha256": sha(raw),
            "prereg_section": "§9", "freeze_commit": D_FREEZE, "generator_commit": GEN, "deployed_commit": D_DEPLOYED,
            "capacity_sha256": sha((ROOT / CAPACITY).read_bytes()),
            "pins_sha256": sha((ROOT / "configs" / "pins.toml").read_bytes()), "stage_d_sha256": STAGE_D_SHA,
            "gpu": "NVIDIA GeForce RTX 5070 Ti", "gpu_driver": "617.42"}
        assert doc["content_sha256"] == cp.content_sha256(doc)


def test_the_d_a1b_classes_carry_the_analysis_evidence(a1b):
    """043 §1.5 / §3: only analysis.json's fields; margins 1 - worst R at C_PROD (043 §2's table); the hypotheses'
    verdicts verbatim; at C_PROD the valid runs' worst attainment, preemptions and KV peak."""
    analysis = read_json(D_A1B / "analysis.json")
    got = {n: c["evidence"] for n, c in by_name(a1b).items()}
    for name, w in (("uniform-32k", "U32K"), ("uniform-8k-t", "U8K-T"), ("mixed-8k-t", "M8K-T")):
        x, e = analysis["workloads"][w], got[name]
        point = x["points"][str(x["c_prod"]["value"])]
        runs = [r for r in point["runs"] if r["valid"]]
        assert e["workload"] == w and e["c_prod"] == x["c_prod"] and e["c_slo"] == x["c_slo"]
        assert (e["c_kv"], e["c_cap"], e["flags"], e["binding"]) == (x["c_kv"], x["c_cap"], x["flags"], x["binding"])
        assert e["margins"] == {t: 1 - R for t, R in point["worst_R"].items() if R is not None}
        assert e["burst_max_ttft_ms"] == max(r["burst_max_ttft_ms"] for r in runs)
        assert e["at_c_prod"] == {"attainment_min": min(r["attainment"] for r in runs),
                                  "preemptions_max": max(r["preemptions"] for r in runs),
                                  "kv_cache_usage_peak_max": max(r["kv_cache_usage_peak"] for r in runs)}
        assert (e["server"], e["output_tokens"]) == (x["server"], 128)
        assert e["descriptive"] == {k: x["descriptive"][k]["c_prod"] for k in ("c_full", "c_late")}
        assert e["not_covered"] == NOT_COVERED
        assert "exactly 128 output tokens (ignore-eos)" in e["scope"] and "after admission" in e["scope"]
    assert {n: rounded(e["margins"]) for n, e in got.items()} == {
        "uniform-32k": {"short": 0.488, "tpot": 0.358}, "uniform-8k-t": {"short": 0.06, "tpot": 0.827},
        "mixed-8k-t": {"short": 0.114, "tpot": 0.212, "long": 0.42}}
    assert {n: e["hypotheses"] for n, e in got.items()} == {
        "uniform-32k": {"H-D1": "supported"}, "uniform-8k-t": {"H-D2b": "supported"},
        "mixed-8k-t": {"H-D2a": "supported"}}
    assert got["mixed-8k-t"]["flags"] == ["WINDOW_SENSITIVE"]
    assert got["mixed-8k-t"]["descriptive"]["c_late"] == {"value": 5, "status": "EXACT"}
    assert "32640-token (10 %)" in got["mixed-8k-t"]["scope"] and "T server" in got["mixed-8k-t"]["scope"]
    assert got["uniform-32k"]["c_kv"] == 3 and round(got["uniform-32k"]["burst_max_ttft_ms"] / 1000, 1) == 12.8
    assert "32640-token prompts" in got["uniform-32k"]["scope"] and "B0" in got["uniform-32k"]["scope"]


def test_d_a2_gives_one_b0_profile_with_the_two_o512_classes(a2):
    """043 §2: B0 only; U2K-O512 ABOVE_CKV (C_KV 30), WINDOW_SENSITIVE (c_full UNSTABLE), attainment 0.96875 with 30
    preemptions at a KV peak of 1.0, H-D3 not supported; U8K-O512 12 at C_KV 12, its first wave 16.95 s."""
    assert list(a2) == ["verified-formal-20261023-d-a2-b0"]
    (doc,) = a2.values()
    assert doc["server"]["fingerprint_sha256"] == cp.stage_b_fingerprint(PINS)
    assert doc["server"]["long_prefill_token_threshold"] is None
    got = by_name(a2)
    assert summary(got["uniform-2k-o512"]) == ("verified", 1920, 512, 32, 32, "exact", "stage_d:U2K-O512:c_prod:EXACT")
    assert summary(got["uniform-8k-o512"]) == ("verified", 8064, 512, 12, 12, "exact", "stage_d:U8K-O512:c_prod:EXACT")
    u2, u8 = got["uniform-2k-o512"]["evidence"], got["uniform-8k-o512"]["evidence"]
    assert (u2["flags"], u2["c_kv"], u2["output_tokens"], u2["server"]) == (["WINDOW_SENSITIVE", "ABOVE_CKV"], 30, 512,
                                                                            "B0")
    assert u2["at_c_prod"] == {"attainment_min": 0.96875, "preemptions_max": 30.0, "kv_cache_usage_peak_max": 1.0}
    assert (u2["hypotheses"], u8["hypotheses"]) == ({"H-D3": "not supported"}, {"H-D3": "supported"})
    assert u2["descriptive"] == {"c_full": {"value": None, "status": "UNSTABLE"},
                                 "c_late": {"value": 32, "status": "LOWER"}}
    assert u2["not_covered"] == NOT_COVERED + ["in_vllm_wait_beyond_c_kv"] and u8["not_covered"] == NOT_COVERED
    assert rounded(u2["margins"]) == {"short": 0.292, "tpot": 0.557}
    assert rounded(u8["margins"]) == {"short": 0.575, "tpot": 0.526}
    assert u8["c_kv"] == 12 and round(u8["burst_max_ttft_ms"] / 1000, 2) == 16.95
    for e in (u2, u8):
        assert "exactly 512 output tokens (ignore-eos)" in e["scope"] and "whole bench process" in e["scope"]


def test_the_stage_d_profiles_pass_the_loader_and_serving_class_applies_043(a1b, a2):
    """Every generated profile loads as verified against this checkout (§9); serving_class then refuses U8K-T (030
    8.8: report only) and both O512 classes (043 §4); uniform-32k and mixed-8k-t are servable."""
    frozen_cfgs = cp.frozen_configs(ROOT)
    profs = [cp.parse_profile(pg.render(doc), source=pid, allow_provisional=False, pins=PINS, frozen=frozen_cfgs)
             for pid, doc in {**a1b, **a2}.items()]
    b0, t, o512 = profs
    assert all(p.verified and p.provenance.prereg_section == "§9" for p in profs)
    assert (b0.server_variant, t.server_variant, o512.server_variant) == ("B0", "T", "B0")
    assert b0.serving_class("uniform-32k").max_concurrency == 2
    assert t.serving_class("mixed-8k-t").max_concurrency == 4
    with pytest.raises(cp.ProfileError, match="030 8.8"):
        t.serving_class("uniform-8k-t")
    for name in ("uniform-2k-o512", "uniform-8k-o512"):
        with pytest.raises(cp.ProfileError, match="043 §4"):
            o512.serving_class(name)


def test_the_stage_d_output_is_deterministic_and_only_the_generator_commit_varies(a1b):
    again, other = gen_d(), gen_d(generator_commit="1c" * 20)
    assert {k: pg.render(v) for k, v in again.items()} == {k: pg.render(v) for k, v in a1b.items()}
    for pid, doc in a1b.items():
        assert {k for k in doc if doc[k] != other[pid][k]} == {"provenance", "content_sha256"}
        assert {k for k in doc["provenance"] if doc["provenance"][k] != other[pid]["provenance"][k]} == {
            "generator_commit"}


def test_stage_d_margins_are_capacity_analysis_margins_on_c1b():
    """043 §1.5: the Stage D analysis has no margins; 1 - worst R at C_PROD is capacity_analysis's own margins.c_prod,
    so the same function on c1b's four workloads gives their margins.c_prod (U2K short: 1 - 0.5892 = 0.4108)."""
    analysis = read_json(ROOT / "results" / "capacity" / "formal-20261017-c1b" / "analysis.json")
    for w, x in analysis["workloads"].items():
        assert pg.stage_d_margins(x) == x["margins"]["c_prod"], w
    assert pg.stage_d_margins(analysis["workloads"]["U2K"])["short"] == 1 - 0.5891562507490562
    x = copy.deepcopy(analysis["workloads"]["U2K"])
    for est in ({"value": None, "status": "UNSTABLE"}, {"value": None, "status": "NT"},
                {"value": 0, "status": "INFEASIBLE"}):
        x["c_prod"] = est
        assert pg.stage_d_margins(x) is None       # no C_PROD point, as capacity_analysis writes it


# ---------------------------------------------------- server groups (043 §1.4)

def test_server_groups_fingerprint_every_boot_by_its_workloads_server():
    """B0 has 8 boots in D-A1b (U32K 6, the sentinel 2), T 12; D-A2 runs B0 only."""
    groups, problems = pg.server_groups(read_json(D_A1B / "session.json"), PINS)
    assert problems == []
    assert {v: (g["fingerprint"], len(g["boots"]), g["workloads"]) for v, g in groups.items()} == {
        "B0": (cp.stage_b_fingerprint(PINS), 8, ["U2K", "U32K"]), "T": (cp.t_fingerprint(PINS), 12, ["U8K-T", "M8K-T"])}
    groups, problems = pg.server_groups(read_json(D_A2 / "session.json"), PINS)
    assert problems == [] and list(groups) == ["B0"]
    assert groups["B0"]["workloads"] == ["U2K", "U2K-O512", "U8K-O512"] and len(groups["B0"]["boots"]) == 14


def _one_t_boot_another_server(s):
    s["runs"][t_boots(s)[3]]["argv"].append("--enforce-eager")


def _t_boots_on_b0(s):
    for tag in t_boots(s):
        s["runs"][tag]["argv"] = s["runs"][tag]["argv"][:-2]          # without --long-prefill-token-threshold 512


def _b0_boots_on_t(s):
    for b in s["boots"]:
        if b["tag"] not in t_boots(s):
            s["runs"][b["tag"]]["argv"] += ["--long-prefill-token-threshold", "512"]


def _no_env_key(s):
    s["runs"][s["boots"][0]["tag"]]["env"].pop("CUDA_HOME")


def _no_record(s):
    del s["runs"][s["boots"][5]["tag"]]


def _unknown_workload(s):
    s["boots"][1]["workload"] = "U64K"


def _no_boots(s):
    s["boots"] = []


@pytest.mark.parametrize("edit, match", [
    (_one_t_boot_another_server, "server T: its boots ran 2 different servers"),
    (_t_boots_on_b0, "not the product T server's"),
    (_b0_boots_on_t, "not the product B0 server's"),
    (_no_env_key, "CUDA_HOME"),
    (_no_record, "no server argv / env"),
    (_unknown_workload, "U64K"),
    (_no_boots, "no boots"),
])
def test_server_groups_refuse_a_server_that_is_not_one_product_server(edit, match):
    s = read_json(D_A1B / "session.json")
    edit(s)
    _, problems = pg.server_groups(s, PINS)
    assert any(match in p for p in problems), problems


def test_the_analysis_records_one_server_argv_per_server():
    """043 §1.4's cross-check: analysis.platform.server_argv_sha256 is one value per server (B0 54614663..., T
    56fcc296...)."""
    analysis = read_json(D_A1B / "analysis.json")
    assert pg.argv_sha256_problems(analysis) == []
    assert set(analysis["platform"]["server_argv_sha256"].values()) == {
        "54614663c11ad0a8805227c2e0e1b12fdc010209c85e143cf723c868e7640c7d",
        "56fcc2960d9f4186b83a343b2193ec8105ebcc0a45a8129c7fb4f55bd186a915"}
    analysis["platform"]["server_argv_sha256"]["M8K-T"] = "00" * 32
    assert "server T" in "; ".join(pg.argv_sha256_problems(analysis))
    analysis["platform"]["server_argv_sha256"] = {"U64K": "00" * 32}
    assert "U64K" in "; ".join(pg.argv_sha256_problems(analysis))


def test_a_session_whose_t_boots_ran_two_servers_is_refused(tmp_path):
    r = d_root(tmp_path)
    path = r["out"] / "session.json"
    s = read_json(path)
    _one_t_boot_another_server(s)
    path.write_text(json.dumps(s, indent=1) + "\n")
    msg = refused_d(r["out"], root=r["root"])
    assert "server T: its boots ran 2 different servers" in msg


# ---------------------------------------------------- 030 8.7 on a synthetic analysis

def test_unstable_nt_and_infeasible_map_to_unavailable_classes_and_a_verified_0(tmp_path, monkeypatch):
    """030 8.7 for §9 on a synthetic analysis.json: the frozen report (its byte reproduction is tested on the real
    sessions) is replaced by one that returns it. UNSTABLE / NT give unavailable classes without any number, INFEASIBLE
    a verified 0; the profiles still pass the loader."""
    r = d_root(tmp_path)
    path = r["out"] / "analysis.json"
    analysis = read_json(path)
    ws = analysis["workloads"]
    ws["U32K"]["c_prod"] = {"value": 0, "status": "INFEASIBLE"}
    ws["U8K-T"]["c_prod"] = {"value": None, "status": "UNSTABLE"}
    ws["M8K-T"]["c_prod"] = {"value": None, "status": "NT"}
    path.write_text(json.dumps(analysis, indent=1) + "\n")
    monkeypatch.setattr(pg.sr, "report", lambda *a, **k: copy.deepcopy(analysis))
    docs = gen_d(r["out"], root=r["root"])
    got = by_name(docs)
    assert summary(got["uniform-32k"]) == ("verified", 32640, 128, 0, 0, "exact", "stage_d:U32K:c_prod:INFEASIBLE")
    for name, w, status in (("uniform-8k-t", "U8K-T", "UNSTABLE"), ("mixed-8k-t", "M8K-T", "NT")):
        c = got[name]
        assert summary(c) == ("unavailable", c["max_prompt_tokens"], 128, None, None, None,
                              f"stage_d:{w}:c_prod:{status}")
        assert f"Stage D {w}: C_PROD is {status}" in c["unavailable_reason"]
        assert "never a number" in c["unavailable_reason"]
    for c in got.values():
        assert (c["evidence"]["margins"], c["evidence"]["at_c_prod"], c["evidence"]["burst_max_ttft_ms"]) == (
            None, None, None)
    assert got["uniform-32k"]["evidence"]["c_prod"] == {"value": 0, "status": "INFEASIBLE"}
    prof = cp.parse_profile(pg.render(docs["verified-formal-20261023-d-a1b-t"]), source="synthetic",
                            allow_provisional=False, pins=PINS, frozen=cp.frozen_configs(r["root"]))
    with pytest.raises(cp.ProfileError, match="unavailable"):
        prof.serving_class("mixed-8k-t")


def test_at_c_prod_and_the_first_wave_read_only_the_valid_runs(tmp_path, monkeypatch):
    """An invalid run at C_PROD (synthetic, as above) changes neither at_c_prod nor burst_max_ttft_ms."""
    r = d_root(tmp_path)
    path = r["out"] / "analysis.json"
    analysis = read_json(path)
    run = analysis["workloads"]["U32K"]["points"]["2"]["runs"][0]
    run.update(valid=False, attainment=0.5, preemptions=99.0, kv_cache_usage_peak=1.0, burst_max_ttft_ms=99999.0)
    path.write_text(json.dumps(analysis, indent=1) + "\n")
    monkeypatch.setattr(pg.sr, "report", lambda *a, **k: copy.deepcopy(analysis))
    e = by_name(gen_d(r["out"], root=r["root"]))["uniform-32k"]["evidence"]
    valid = [x for x in analysis["workloads"]["U32K"]["points"]["2"]["runs"] if x["valid"]]
    assert len(valid) == 4 and e["at_c_prod"] == {"attainment_min": 1.0, "preemptions_max": 0.0,
                                                  "kv_cache_usage_peak_max": max(x["kv_cache_usage_peak"] for x in valid)}
    assert e["burst_max_ttft_ms"] == max(x["burst_max_ttft_ms"] for x in valid) < 99999.0
    assert pg.at_c_prod(analysis["workloads"]["U32K"], 0) is None and pg.at_c_prod(analysis["workloads"]["U32K"], 7) \
        is None


# ---------------------------------------------------- refusals (exit 2, nothing written)

@pytest.mark.parametrize("session, match", [
    (D_B, "D-B and D-C produce no profile"), (D_C, "D-B and D-C produce no profile"), (D_A1, "aborted"),
])
def test_d_b_d_c_and_the_aborted_d_a1_give_no_profile(tmp_path, capsys, session, match):
    """042: D-B (open) and D-C (verify) only go to service/README.md's scope; the aborted D-A1 has no confirm plan
    and no analysis.json."""
    out = tmp_path / "profiles"
    out.mkdir()
    assert main_d(session, out) == 2
    err = capsys.readouterr().err
    assert list(out.iterdir()) == [] and "refused" in err and match in err
    if session == D_A1:
        assert "confirm_plan" in err and "analysis.json" in err


def test_the_plan_and_the_analysis_must_both_be_closed_loop(tmp_path):
    """Each kind is checked on its own: a D-B copy whose analysis.json claims closed loop is still refused by its plan,
    and a D-A copy whose analysis.json claims open loop by the analysis."""
    assert pg.D_A_KIND == sdplan.KINDS["D-A1"] == sdplan.KINDS["D-A2"] == "closed"
    assert {sdplan.KINDS["D-B"], sdplan.KINDS["D-C"]} == {"open", "verify"}
    r = d_root(tmp_path / "b", D_B)
    path = r["out"] / "analysis.json"
    doc = read_json(path)
    doc["provenance"]["kind"] = "closed"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    msg = refused_d(r["out"], root=r["root"])
    assert "plan.json kind is 'open'" in msg and "analysis.json kind" not in msg
    r = d_root(tmp_path / "a", D_A1B)
    path = r["out"] / "analysis.json"
    doc = read_json(path)
    doc["provenance"]["kind"] = "open"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    msg = refused_d(r["out"], root=r["root"])
    assert "analysis.json kind is 'open'" in msg and "plan.json kind" not in msg


def test_an_analysis_with_two_server_argv_for_one_server_is_refused(tmp_path):
    r = d_root(tmp_path)
    path = r["out"] / "analysis.json"
    doc = read_json(path)
    doc["platform"]["server_argv_sha256"]["M8K-T"] = doc["platform"]["server_argv_sha256"]["U32K"]
    path.write_text(json.dumps(doc, indent=1) + "\n")
    assert "2 server argv for server T" in refused_d(r["out"], root=r["root"])


def test_the_stage_d_freeze_commit_must_be_the_commit_that_set_the_freeze():
    parent = git(ROOT, "rev-parse", f"{D_FREEZE}^").strip()
    assert "has no [freeze] stage_d = true: not the freeze commit" in refused_d(freeze_commit=parent)
    assert "already frozen" in refused_d(freeze_commit=D_DEPLOYED)        # frozen in its parent too
    assert "40 lowercase hex" in refused_d(freeze_commit=f"{D_FREEZE}^")
    assert "DEPLOYED_COMMIT" in refused_d(git=RepoGit(ancestor=False))
    first, unfrozen = "f1" * 20, "f2" * 20                                    # freeze, unfreeze, re-freeze
    msg = refused_d(git=RepoGit(history=[D_FREEZE, unfrozen, first],
                                files={(first, STAGE_D_TOML): FROZEN_D, (unfrozen, STAGE_D_TOML): UNFROZEN_D}))
    assert f"already frozen at {first}" in msg and "re-freeze" in msg


def test_a_stage_d_toml_changed_after_the_freeze_is_refused():
    msg = refused_d(git=RepoGit(files={(D_FREEZE, STAGE_D_TOML): (ROOT / STAGE_D_TOML).read_bytes() + b"# x\n"}))
    assert STAGE_D_TOML in msg and "changed after the freeze" in msg


def test_the_stage_d_frozen_files_are_the_harness_and_the_stage_c_modules():
    """043 §1.2: stage_d_plan.HARNESS_FILES (stage_d_datasets, never imported, is this checkout's file) and the Stage C
    modules; each the file the generator runs."""
    files = [pg.frozen_file(m) for m in pg.STAGE_D_MODULES]
    assert [rel for rel, _ in files] == D_FROZEN_PATHS
    assert all(local.resolve() == (ROOT / rel).resolve() for rel, local in files)
    assert [m for m in pg.STAGE_D_MODULES if isinstance(m, str)] == ["benchmarks/stage_d_datasets.py"]
    assert pg.frozen_file(pg.STAGE_C_MODULES[0]) == ("benchmarks/capacity_plan.py", Path(pg.cplan.__file__))


@pytest.mark.parametrize("path", D_FROZEN_PATHS)
def test_a_stage_d_frozen_file_changed_after_the_freeze_is_refused(path):
    edited = (ROOT / path).read_bytes() + b"# edited after the freeze\n"
    msg = refused_d(git=RepoGit(files={(D_DEPLOYED, path): edited}))
    assert path in msg and "DEPLOYED_COMMIT" in msg and "frozen Stage D code" in msg
    msg = refused_d(git=RepoGit(files={(D_FREEZE, path): edited}))
    where = "in this checkout" if path == "benchmarks/stage_d_datasets.py" else "imported by this generator"
    assert path in msg and where in msg
    assert "no " + path in refused_d(git=RepoGit(files={(D_FREEZE, path): None}))


def test_a_copied_root_generates_the_same_profiles(tmp_path, a1b):
    """The private root of the on-disk refusals below is a faithful copy: unbroken, it gives the same bytes."""
    r = d_root(tmp_path)
    got = gen_d(r["out"], root=r["root"])
    assert {k: pg.render(v) for k, v in got.items()} == {k: pg.render(v) for k, v in a1b.items()}


def test_a_stage_d_analysis_json_with_other_bytes_or_content_is_refused(tmp_path):
    """021 R3: stage_d_report.report, re-run read-only, reproduces analysis.json byte for byte."""
    r = d_root(tmp_path)
    path = r["out"] / "analysis.json"
    doc = read_json(path)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    assert "byte for byte" in refused_d(r["out"], root=r["root"])
    doc["workloads"]["U32K"]["c_prod"]["value"] = 3
    path.write_text(json.dumps(doc, indent=1) + "\n")
    msg = refused_d(r["out"], root=r["root"])
    assert "byte for byte" in msg and "workloads" in msg
    path.unlink()
    assert "stage_d_report" in refused_d(r["out"], root=r["root"])


def test_a_stage_d_session_elsewhere_or_with_another_slo_is_refused(tmp_path):
    r = d_root(tmp_path)
    other = r["out"].parent / "formal-20261023-d-a1b-copy"
    shutil.copytree(r["out"], other)
    assert "results/stage_d/<session_id>" in refused_d(other, root=r["root"])
    configs = pg.load_stage_d_configs(r["root"])
    s3 = copy.deepcopy(configs[0])
    s3["slo"]["tpot_p95_ms"] = 200
    msg = refused_d(r["out"], root=r["root"], stage_d_configs=(s3, *configs[1:]))
    assert "SLO" in msg and "stage_d.toml" in msg


@pytest.mark.parametrize("git, match", [
    (RepoGit(dirty=" M service/capserve/profile_gen.py\n"), "uncommitted"),
    (RepoGit(ancestor=False), "DEPLOYED_COMMIT"),
])
def test_main_refuses_a_stage_d_session_and_writes_nothing(tmp_path, capsys, git, match):
    out = tmp_path / "profiles"
    out.mkdir()
    assert main_d(D_A1B, out, git=git) == 2
    assert list(out.iterdir()) == [] and match in capsys.readouterr().err


def test_main_writes_every_profile_of_a_d_a_session(tmp_path, capsys, a1b):
    out = tmp_path / "profiles"
    out.mkdir()
    assert main_d(D_A1B, out) == 0
    assert sorted(p.name for p in out.iterdir()) == [f"{pid}.json" for pid in a1b]
    for pid, doc in gen_d(generator_commit=CLEAN_HEAD).items():
        assert (out / f"{pid}.json").read_bytes() == pg.render(doc)
    printed = json.loads(capsys.readouterr().out)
    assert [p["profile_id"] for p in printed["profiles"]] == list(a1b)
    assert printed["profiles"][1]["out"] == str(out / "verified-formal-20261023-d-a1b-t.json")
    assert printed["profiles"][1]["classes"] == {"uniform-8k-t": ["verified", 4, "exact"],
                                                 "mixed-8k-t": ["verified", 4, "exact"]}


def test_main_writes_all_or_nothing_and_never_overwrites(tmp_path, capsys):
    out = tmp_path / "profiles"
    out.mkdir()
    keep = out / "verified-formal-20261023-d-a1b-t.json"
    keep.write_text("keep")
    assert main_d(D_A1B, out) == 2
    assert [p.name for p in out.iterdir()] == [keep.name] and keep.read_text() == "keep"
    assert "already exists" in capsys.readouterr().err


def test_main_never_writes_stage_d_profiles_under_results_or_into_no_directory(tmp_path, capsys):
    r = d_root(tmp_path)
    out = r["root"] / "results" / "profiles"
    out.mkdir()
    assert main_d(r["out"], out, root=r["root"]) == 2
    assert list(out.iterdir()) == [] and "results/" in capsys.readouterr().err
    assert main_d(D_A1B, tmp_path / "missing") == 2
    assert not (tmp_path / "missing").exists() and "not an existing directory" in capsys.readouterr().err


def test_write_all_removes_what_it_wrote_when_a_write_fails(tmp_path):
    a, keep = tmp_path / "a.json", tmp_path / "keep.json"
    with pytest.raises(pg.GenerationError, match="all or none"):
        pg.write_all({a: b"a", tmp_path / "missing" / "b.json": b"b"})
    assert not a.exists()
    keep.write_text("keep")                                     # created after the check (a race): never overwritten
    with pytest.raises(pg.GenerationError, match="all or none"):
        pg.write_all({a: b"a", keep: b"other"})
    assert not a.exists() and keep.read_text() == "keep"
    pg.write_all({a: b"a"})
    assert a.read_bytes() == b"a"


def test_out_and_out_dir_follow_the_sessions_path(sess_a, tmp_path, monkeypatch, capsys):
    """043 §1.1: results/stage_d/ takes --out-dir, results/capacity/ --out; exactly one of the two, else exit 2."""
    out, d = tmp_path / "p.json", tmp_path / "dir"
    d.mkdir()
    assert main_d(D_A1B, out, option="--out") == 2
    assert not out.exists() and "--out-dir" in capsys.readouterr().err
    monkeypatch.setattr(pg, "load_configs", lambda root: sess_a["configs"])
    assert pg.main(["--session", str(sess_a["out"]), "--freeze-commit", FREEZE, "--out-dir", str(d)],
                   root=sess_a["repo"], git=FakeGit()) == 2
    assert list(d.iterdir()) == [] and "--out" in capsys.readouterr().err
    for argv in (["--out", str(out), "--out-dir", str(d)], []):
        with pytest.raises(SystemExit) as exc:
            pg.main(["--session", str(D_A1B), "--freeze-commit", D_FREEZE, *argv], root=ROOT, git=RepoGit())
        assert exc.value.code == 2
    assert not out.exists() and list(d.iterdir()) == []


@pytest.mark.parametrize("text", [None, "", "[freeze]\nstage_d = false\n", "[freeze]\nstage_d = 'true'\n"])
def test_main_refuses_a_stage_d_session_until_stage_d_is_frozen(tmp_path, capsys, text):
    root = cli_root(tmp_path, (ROOT / CAPACITY).read_text())
    if text is not None:
        (root / STAGE_D_TOML).write_text(text)
    out = tmp_path / "out"
    out.mkdir()
    assert main_d(root / "results" / "stage_d" / D_A1B.name, out, root=root) == 2
    assert list(out.iterdir()) == [] and "[freeze] stage_d" in capsys.readouterr().err
    (root / STAGE_D_TOML).write_text("[freeze\n")
    assert main_d(root / "results" / "stage_d" / D_A1B.name, out, root=root) == 2
    assert "cannot read" in capsys.readouterr().err


def test_load_stage_d_configs_reads_stage_d_toml_and_pins(tmp_path):
    """043 §1.2: (stage_d.toml, pins, their sha256); Stage C's load_configs four-tuple is unchanged."""
    raw_d, raw_p = (ROOT / STAGE_D_TOML).read_bytes(), (ROOT / "configs" / "pins.toml").read_bytes()
    assert pg.load_stage_d_configs(ROOT) == (tomllib.loads(raw_d.decode()), tomllib.loads(raw_p.decode()), STAGE_D_SHA,
                                             sha(raw_p))
    assert pg.load_configs(ROOT)[2:] == (sha((ROOT / CAPACITY).read_bytes()), sha(raw_p))
    assert pg.load_stage_d_configs(cli_root(tmp_path, None)) is None


def test_freeze_commit_problems_for_stage_d_with_a_real_repository(tmp_path):
    """The parameterized freeze check (043 §1.2) on stage_d.toml, with the Stage C defaults unchanged."""
    root = tmp_path / "r"
    root.mkdir()
    git(root, "init", "-q")
    files = {rel: (ROOT / rel).read_bytes() for rel in D_FROZEN_PATHS}
    before = commit_files(root, {STAGE_D_TOML: UNFROZEN_D, CAPACITY: FROZEN_TOML, **files})
    freeze = commit_files(root, {STAGE_D_TOML: FROZEN_D})
    deployed = commit_files(root, {"x.txt": b"after the freeze\n"})
    kw = {"git": pg.git_runner(root), "freeze_file": STAGE_D_TOML, "freeze_key": "stage_d",
          "modules": pg.STAGE_D_MODULES}
    assert pg.freeze_commit_problems(freeze, deployed_commit=deployed, capacity_sha256=sha(FROZEN_D), **kw) == []
    assert "[freeze] stage_d = true: not the freeze commit" in pg.freeze_commit_problems(
        before, deployed_commit=deployed, capacity_sha256=sha(FROZEN_D), **kw)[0]
    assert "already frozen" in pg.freeze_commit_problems(deployed, deployed_commit=deployed,
                                                         capacity_sha256=sha(FROZEN_D), **kw)[0]
    assert "DEPLOYED_COMMIT" in pg.freeze_commit_problems(freeze, deployed_commit=before,
                                                          capacity_sha256=sha(FROZEN_D), **kw)[0]
    # capacity.toml, frozen since `before`, is what the Stage C defaults still read
    assert "already frozen" not in "; ".join(pg.freeze_commit_problems(before, deployed_commit=deployed,
                                                                       capacity_sha256=CAP_SHA, git=kw["git"]))
    commit_files(root, {STAGE_D_TOML: UNFROZEN_D})
    refrozen = commit_files(root, {STAGE_D_TOML: FROZEN_D + b"# again\n"})
    msg = "; ".join(pg.freeze_commit_problems(refrozen, deployed_commit=refrozen,
                                              capacity_sha256=sha(FROZEN_D + b"# again\n"), **kw))
    assert f"already frozen at {freeze}" in msg and "re-freeze" in msg


def test_the_regeneration_dispatches_on_the_prereg_section(a1b):
    """043 step 5: a §9 profile is regenerated from its results/stage_d session and picked by its profile_id."""
    for doc in a1b.values():
        committed = copy.deepcopy(doc)
        committed["provenance"]["generator_commit"] = "5e" * 20
        committed["content_sha256"] = cp.content_sha256(committed)
        assert _regenerable(regenerate(committed)) == _regenerable(committed)
    with pytest.raises(pg.GenerationError, match="not 'verified-formal-20261023-d-a1b-x'"):
        regenerate({**doc, "profile_id": "verified-formal-20261023-d-a1b-x"})
