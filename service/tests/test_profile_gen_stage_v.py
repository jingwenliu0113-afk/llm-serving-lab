"""Tests for capserve/profile_gen.py's Stage V path (decision 044 產品對應; PREREGISTRATION §12): `python -m
capserve.profile_gen --validation-session results/stage_v/<session> --freeze-commit <sha> --out-dir service/profiles`.

There is no Stage V data (044: none before the freeze), so every session here is synthetic, and written by the real
harness: run_stage_v.run_session in the fake WSL of tests/test_run_stage_v.py (the closed-loop FIFO gateway of
tests/test_stage_v_analysis.py drives each run to a PASS, or to a FAIL with fail_kw), and its analysis.json by the real
stage_v_report CLI. So the units, provenance and platform the generator reads have their real shape, and every unit
holds or not exactly as the frozen analysis decides. git is faked (VGit): the Stage V freeze commit and its stage_v.toml
(FROZEN_V stands in for the frozen file: the configs are injected and the sessions record its sha256), the frozen
modules at it and at the sessions' DEPLOYED_COMMIT, the request path at DEPLOYED_COMMIT and the base profiles at the
freeze commit are this checkout's bytes unless a test says otherwise; the git helpers are also tested on a real
repository."""

import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

import capacity_report as cr
import run_stage_v as rv
import stage_v_plan as sv
import stage_v_report as svr
from capserve import config as cfg
from capserve import gateway_path as gpath
from capserve import profile as cp
from capserve import profile_gen as pg

SERVICE = Path(__file__).resolve().parent.parent
ROOT = SERVICE.parent
if str(ROOT / "tests") not in sys.path:      # the Stage V fake world lives with the research tests
    sys.path.insert(0, str(ROOT / "tests"))
import test_run_stage_v as t  # noqa: E402
import test_stage_v_analysis as tva  # noqa: E402
import test_profile_gen as tpg  # noqa: E402

FREEZE_V = "f5" * 20          # set configs/stage_v.toml [freeze] stage_v = true (faked)
BEFORE_V = "b5" * 20          # the commit before it that last touched stage_v.toml (unfrozen)
GEN, CLEAN_HEAD = "9a" * 20, "7b" * 20
DEPLOYED = t.td.COMMIT        # the synthetic sessions' DEPLOYED_COMMIT
T0 = t.T0                     # their start (test_run_stage_v's; make_sessions moves it)
STAGE_V_TOML = "configs/stage_v.toml"
# stage_v.toml at the freeze commit. The configs are injected as dicts, so these bytes stand in for the frozen file:
# the test sessions record their sha256, as a real session records the sha256 of the bytes it ran.
FROZEN_V = b"[freeze]\nstage_v = true\n# the frozen stage_v.toml of the test sessions (configs are injected)\n"
UNFROZEN_V = b"[freeze]\nstage_v = false\n"
UNIFORM = ROOT / "results" / "core_b" / "formal-20261009-coreb-uniform"   # stage_d.toml's derived_from
ADVANCED = ROOT / "results" / "advanced" / pg.ADVANCED_SESSION
CAPACITY_CONFIGS = pg.load_configs(ROOT)                  # the frozen capacity.toml (the loader and the provenance)
C1B = json.loads((SERVICE / "profiles" / "verified-formal-20261017-c1b.json").read_text())
D_A2_B0 = json.loads((SERVICE / "profiles" / "verified-formal-20261023-d-a2-b0.json").read_text())
SLOW = {"max_queue_wait_s": 120.0, "request_deadline_s": 600.0}                         # 044 V2 / V4
S2_TERMS = {"ttft_p95_ms": {"2048": 2000, "8192": 6000, "32768": 20000}, "tpot_p95_ms": 100, "mixed_long": "budget"}
LOCAL = {"tokenizer": "local", "tokenizer_path": "/snap"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SV_SHA = sha(FROZEN_V)
PINS_SHA = sha((ROOT / "configs" / "pins.toml").read_bytes())


def capserve_files() -> list:
    return sorted(p.relative_to(ROOT).as_posix() for p in (SERVICE / "capserve").glob("*.py"))


class VGit:
    """The git answers the Stage V path asks for: stage_v.toml at the freeze commit, its first parent and the earlier
    commits of its history; the frozen modules (pg.STAGE_V_MODULES) at the freeze commit and at DEPLOYED; the tracked
    service/capserve files at DEPLOYED (ls-tree and show: the request path); the base profiles at the freeze commit;
    stage_v.toml's history, ancestry, HEAD and the tracked-file status. files overrides or adds {(rev, path): bytes or
    None}; listing replaces DEPLOYED's service/capserve listing (None: ls-tree fails); calls records every question."""

    def __init__(self, *, parent_frozen=False, ancestor=True, head=CLEAN_HEAD, dirty="", files=None, history=None,
                 listing="checkout"):
        self.ancestor, self.head, self.dirty = ancestor, head, dirty
        self.files = {(FREEZE_V, STAGE_V_TOML): FROZEN_V, (BEFORE_V, STAGE_V_TOML): UNFROZEN_V,
                      (f"{FREEZE_V}^", STAGE_V_TOML): FROZEN_V if parent_frozen else UNFROZEN_V}
        for m in pg.STAGE_V_MODULES:
            path, local = pg.frozen_file(m)
            self.files[(FREEZE_V, path)] = self.files[(DEPLOYED, path)] = local.read_bytes()
        self.listing = capserve_files() if listing == "checkout" else listing
        for rel in capserve_files():
            self.files[(DEPLOYED, rel)] = (ROOT / rel).read_bytes()
        for rel in (pg.COPY_BASE, pg.V1_BASE):
            self.files[(FREEZE_V, rel)] = (ROOT / rel).read_bytes()
        self.files.update(files or {})
        self.history = [FREEZE_V, BEFORE_V] if history is None else history
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "show":
            rev, _, path = args[1].partition(":")
            return self.files.get((rev, path))
        if args[:2] == ("rev-list", "--full-history"):
            assert args[3:] == ("--", STAGE_V_TOML)
            return "".join(f"{c}\n" for c in self.history).encode() if args[2] == FREEZE_V else None
        if args[:2] == ("merge-base", "--is-ancestor"):
            return b"" if self.ancestor and args[2:] == (FREEZE_V, DEPLOYED) else None
        if args[:3] == ("ls-tree", "-r", "--name-only"):
            assert args[4:] == ("--", "service/capserve")
            ok = args[3] == DEPLOYED and self.listing is not None
            return "".join(f"{p}\n" for p in self.listing).encode() if ok else None
        if args == ("rev-parse", "HEAD"):
            return None if self.head is None else self.head.encode() + b"\n"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return self.dirty.encode()
        raise AssertionError(args)


def link_references(root: Path) -> None:
    """What the report reads besides the session: the checkout's service/ (the pinned profiles), and the Core B uniform
    and Advanced sessions (the sentinel's descriptive references; Stage D's D5 is linked by make_repo)."""
    (root / "service").symlink_to(SERVICE)
    for ref in (UNIFORM, ADVANCED):
        (root / "results" / ref.parent.name).mkdir(exist_ok=True)
        (root / "results" / ref.parent.name / ref.name).symlink_to(ref)


def make_session(tmp: Path, name: str, fails=(), *, mode: str = "formal") -> dict:
    """A Stage V session (formal V-ADM / V-CLS, or the mini) at <tmp>/repo/results/stage_v/<id> written by the real
    runner; fails: the (point, block) runs that FAIL (test_stage_v_analysis.fail_kw), every other run PASSes
    comfortably. A formal session gets its analysis.json from the real stage_v_report CLI."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t, "SV_SHA", SV_SHA)              # the session records the sha256 of FROZEN_V
        mp.setattr(t, "PINS_SHA", PINS_SHA)          # and of this checkout's pins.toml (the base profiles' provenance)
        repo = t.make_repo(tmp)
        link_references(repo[0])
        out, s, _ = t.session(tmp, name, repo=repo, kw=tva.fail_kw(set(fails)), mode=mode)
        c, d, p = t.configs(repo[0], repo[2], formal=True)
    configs = (c, d, p, SV_SHA, t.SD_SHA, PINS_SHA)
    if mode == "formal":
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(svr, "ROOT", repo[0])
            mp.setattr(svr, "load_configs", lambda: configs)
            assert svr.main(["--session", str(out), "--uniform-session", str(UNIFORM), "--advanced-session",
                             str(ADVANCED), "--out", str(out / "analysis.json")]) == 0
    return {"root": repo[0], "out": out, "session": s, "configs": CAPACITY_CONFIGS, "stage_v_configs": configs,
            "dataset_sha256": repo[2]}


class BadStartSentinel(t.World):
    """The start sentinel's bench result loses a TTFT: the runner aborts the session after its first boot."""

    def run_bench(self, argv, log_path, env, timeout_s):
        out = super().run_bench(argv, log_path, env, timeout_s)
        a = dict(zip(argv, argv[1:]))
        if a["--result-filename"].startswith("sentinel-start"):
            path = Path(a["--result-dir"], a["--result-filename"])
            res = json.loads(path.read_text())
            res["ttfts"][3] = None
            path.write_text(json.dumps(res))
        return out


class WrongVllm(t.td.Env):
    """The preflight records another vLLM: the runner aborts in its preflight, before it records the freeze sha256
    and previous_formal."""

    def __call__(self, out_path, gpu_lock_held):
        rec = super().__call__(out_path, gpu_lock_held)
        rec["runtime"]["vllm"]["vllm"] = "0.31.0"
        return rec


def port_taken() -> Callable:
    """The gateway port is free for the first V boot's check and taken at the next one."""
    calls = []

    def port_free(port):
        calls.append(port)
        return len(calls) != 2
    return port_free


# The runner's own aborts (run_stage_v.RunError): test_run_stage_v.session keywords, given the session's tmp dir.
ABORTS = {
    "port busy": lambda d: {"port_free": lambda port: port != rv.GATEWAY_PORT},     # before the first V boot
    "start sentinel": lambda d: {"world": lambda plan: BadStartSentinel(d, plan)},
    "vLLM": lambda d: {"capture_env": WrongVllm()},                                 # in the preflight
    "port taken": lambda d: {"port_free": port_taken()},                            # at a later V boot
}


def make_sessions(tmp: Path, *specs) -> dict:
    """Formal Stage V sessions in one repo root, in order, each written by the real runner and reported by the real
    stage_v_report CLI: a spec is (plan session, session id, the runs that FAIL or an ABORTS key, its start in hours
    after test_run_stage_v.T0). An aborted session is left as the runner leaves it, never reported. {session id: what
    make_session returns} of the reported ones."""
    got = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t, "SV_SHA", SV_SHA)
        mp.setattr(t, "PINS_SHA", PINS_SHA)
        repo = t.make_repo(tmp)
        link_references(repo[0])
        for name, sid, fails, hours in specs:
            mp.setattr(t, "T0", T0 + dt.timedelta(hours=hours))
            (tmp / sid).mkdir()
            if isinstance(fails, str):
                with pytest.raises(rv.RunError):
                    t.session(tmp / sid, name, repo=repo, out=repo[0].joinpath(*sv.RESULTS, sid),
                              **ABORTS[fails](tmp / sid))
                continue
            out, s, _ = t.session(tmp / sid, name, repo=repo, kw=tva.fail_kw(set(fails)),
                                  out=repo[0].joinpath(*sv.RESULTS, sid))
            configs = (*t.configs(repo[0], repo[2], formal=True), SV_SHA, t.SD_SHA, PINS_SHA)
            mp.setattr(svr, "ROOT", repo[0])
            mp.setattr(svr, "load_configs", lambda: configs)
            assert svr.main(["--session", str(out), "--uniform-session", str(UNIFORM), "--advanced-session",
                             str(ADVANCED), "--out", str(out / "analysis.json")]) == 0
            got[sid] = {"root": repo[0], "out": out, "session": s, "configs": CAPACITY_CONFIGS,
                        "stage_v_configs": configs, "dataset_sha256": repo[2]}
    return got



# The synthetic sessions: (session, the runs that FAIL). Every point runs in every block, so a FAIL in block 1 makes
# the point FAIL (it is skipped afterwards) and its unit not hold.
SCENARIOS = {
    "adm": ("V-ADM", ()),                                                     # PACE-2K, PACE-8K, W1
    "adm-w1": ("V-ADM", (("P-U2K", 1), ("P-U8K", 1))),                        # W1 without PACE
    "adm-pace": ("V-ADM", (("W-90", 1),)),                                     # PACE-2K, PACE-8K without W1
    "adm-veto": ("V-ADM", (("P-U2K-var", 1),)),                                # PACE-2K vetoed; PACE-8K, W1
    "adm-none": ("V-ADM", (("P-U2K", 1), ("P-U8K", 1), ("W-50", 1))),         # no unit holds
    "cls": ("V-CLS", ()),                                                     # VG-32K, VG-M8K-T, V1-2K, V1-8K
    "cls-8k": ("V-CLS", (("GW-U2K-O512", 1),)),                               # V1-8K (and the VG units)
    "cls-vg": ("V-CLS", (("L-U2K", 1), ("L-U8K", 1))),                         # only VG-32K, VG-M8K-T: no file
}


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    cache = {}

    def get(key):
        if key not in cache:
            cache[key] = make_session(tmp_path_factory.mktemp(key), *SCENARIOS[key])
        return cache[key]
    return get


def gen(s: dict, **kw) -> tuple:
    kw.setdefault("freeze_commit", FREEZE_V)
    kw.setdefault("generator_commit", GEN)
    kw.setdefault("root", s["root"])
    kw.setdefault("configs", s["configs"])
    kw.setdefault("stage_v_configs", s["stage_v_configs"])
    kw.setdefault("git", VGit())
    return pg.generate_stage_v(kw.pop("session", s["out"]), **kw)


def one(s: dict, **kw) -> dict:
    docs, _ = gen(s, **kw)
    (doc,) = docs.values()
    return doc


def problems(s: dict, **kw) -> list:
    with pytest.raises(pg.GenerationError) as exc:
        gen(s, **kw)
    return exc.value.problems


def refused(s: dict, **kw) -> str:
    return "; ".join(problems(s, **kw))


def clone(s: dict, tmp_path: Path) -> dict:
    """A private copy of a session's repo root, to break on disk (links kept: service/, the references)."""
    root = tmp_path / "repo"
    shutil.copytree(s["root"], root, symlinks=True)
    return {**s, "root": root, "out": root / s["out"].relative_to(s["root"])}


def frozen_configs(s: dict) -> cp.FrozenConfigs:
    cap, _, cap_sha, _ = s["configs"]
    c, d, p, sv_sha, sd_sha, pins_sha = s["stage_v_configs"]
    return cp.FrozenConfigs(cap, cap_sha, pins_sha, d, sd_sha, c, sv_sha)


def loaded(s: dict, doc: dict) -> cp.CapacityProfile:
    return cp.parse_profile(pg.render(doc), source=doc["profile_id"], allow_provisional=False,
                            pins=s["stage_v_configs"][2], frozen=frozen_configs(s))


def analysis(s: dict) -> dict:
    return json.loads((s["out"] / "analysis.json").read_text())


def validation(s: dict, unit: str) -> dict:
    """044 / 實作補記一第 9 項: a §12 validation is the unit's {study, unit, points, measured_classes, worst_r, n}."""
    v = analysis(s)["units"][unit]["validation"]
    return {k: v[k] for k in ("study", "unit", "points", "measured_classes", "worst_r", "n")}


def base_class(doc: dict, name: str) -> dict:
    return copy.deepcopy(next(c for c in doc["classes"] if c["name"] == name))


def paced(s: dict, w: str) -> dict:
    """044 PACE-2K / PACE-8K: c1b's class verbatim, with the pacing 044 V2 froze, the slow queue and the validation."""
    name, unit, params = {"U2K": ("uniform-2k", "PACE-2K", (9600, 5920)),
                          "U8K": ("uniform-8k", "PACE-8K", (32256, 5800))}[w]
    return {**base_class(C1B, name), "pacing": {"burst_tokens": params[0], "rate_tokens_per_s": params[1],
                                                "queue": SLOW}, "validation": validation(s, unit)}


def admission(s: dict) -> dict:
    """044 W1: weighted-v1's parameters over the copies (the study's v-uniform-2k / v-uniform-8k are c1b's classes),
    the slow queue, the two mixes, the frozen request path and V4's validation."""
    return {"policy": "weighted-v1", "budget": 32, "max_queue": 32,
            "classes": [{"name": "uniform-2k", "units": 1, "max_concurrency": 32},
                        {"name": "uniform-8k", "units": 4, "max_concurrency": 8}],
            "queue": SLOW, "measured_mixes": [0.9, 0.5], "gateway_path_sha256": sv.gateway_path_sha256(),
            "validation": validation(s, "W1")}


def v1(s: dict, w: str) -> dict:
    """044 V1: the D-A O512 class of d-a2-b0 under V1's name at V1's C, its evidence with not_covered recomputed, the
    output distribution, the L point's five datasets and via_gateway; validated by its unit."""
    name, base, c, unit, tier = {"U2K-O512": ("uniform-2k-o1-512", "uniform-2k-o512", 32, "V1-2K", 2048),
                                 "U8K-O512": ("uniform-8k-o1-512", "uniform-8k-o512", 8, "V1-8K", 8192)}[w]
    cls = base_class(D_A2_B0, base)
    above = ["in_vllm_wait_beyond_c_kv"] if w == "U2K-O512" else []           # U2K-O512 is ABOVE_CKV (043 §2)
    datasets = [s["dataset_sha256"][f"v1-{tier}-r{r}"] for r in range(1, 6)]           # the pinned sequences
    ev = {**cls["evidence"], "not_covered": ["prompt_length_variation", "open_loop", "start_up_wave",
                                             "concurrent_classes", *above, "output_length_distribution"],
          "output_distribution": "uniform[1,512]", "datasets_sha256": datasets, "via_gateway": True}
    return {**cls, "name": name, "max_concurrency": c, "max_queue": c, "source": f"stage_v:{unit}",
            "bound": "validated", "evidence": ev, "validation": validation(s, unit)}


def classes(doc: dict) -> dict:
    return {c["name"]: c for c in doc["classes"]}


# ------------------------------------------------------------------ units -> artifacts (044 產品對應) --

def test_v_adm_with_every_unit_holding_gives_the_paced_copies_and_the_admission_section(sessions):
    s = sessions("adm")
    docs, holds = gen(s)
    assert holds == ["PACE-2K", "PACE-8K", "W1"] and list(docs) == ["verified-formal-t-v-adm-b0"]
    doc = docs["verified-formal-t-v-adm-b0"]
    assert classes(doc) == {"uniform-2k": paced(s, "U2K"), "uniform-8k": paced(s, "U8K")}
    assert list(classes(doc)) == ["uniform-2k", "uniform-8k"]
    assert doc["admission"] == admission(s)
    assert [p["id"] for p in doc["provenance"]["base_profiles"]] == ["verified-formal-20261017-c1b"]


def test_w1_without_pacing_names_plain_copies(sessions):
    """044 W1: an admission member whose PACE unit does not hold is c1b's class verbatim (no pacing, no validation)."""
    s = sessions("adm-w1")
    docs, holds = gen(s)
    assert holds == ["W1"]
    (doc,) = docs.values()
    assert classes(doc) == {"uniform-2k": base_class(C1B, "uniform-2k"), "uniform-8k": base_class(C1B, "uniform-8k")}
    assert doc["admission"] == admission(s)


def test_pacing_without_w1_writes_no_admission_section(sessions):
    s = sessions("adm-pace")
    docs, holds = gen(s)
    assert holds == ["PACE-2K", "PACE-8K"]
    (doc,) = docs.values()
    assert classes(doc) == {"uniform-2k": paced(s, "U2K"), "uniform-8k": paced(s, "U8K")} and doc["admission"] is None


def test_a_vetoed_pace_2k_leaves_uniform_2k_a_plain_admission_member(sessions):
    """044 V2: F-U2K-var PASS and P-U2K-var not: PACE-2K is vetoed although P-U2K PASSed; PACE-8K and W1 hold."""
    s = sessions("adm-veto")
    units = analysis(s)["units"]
    assert units["PACE-2K"]["classes"] == {"P-U2K": "PASS"} and units["PACE-2K"]["vetoed"] is True
    docs, holds = gen(s)
    assert holds == ["PACE-8K", "W1"]
    (doc,) = docs.values()
    assert classes(doc) == {"uniform-2k": base_class(C1B, "uniform-2k"), "uniform-8k": paced(s, "U8K")}
    assert doc["admission"] == admission(s)


@pytest.mark.parametrize("key, holds", [("adm-none", []), ("cls-vg", ["VG-32K", "VG-M8K-T"])])
def test_no_profile_without_a_unit_that_writes_one(sessions, key, holds):
    """044 產品對應: VG-32K / VG-M8K-T make candidates only (no new file); with no other unit, nothing is written."""
    docs, got = gen(sessions(key))
    assert docs == {} and got == holds


def test_v_cls_gives_v1s_validated_classes(sessions):
    s = sessions("cls")
    docs, holds = gen(s)
    assert holds == ["VG-32K", "VG-M8K-T", "V1-2K", "V1-8K"] and list(docs) == ["verified-formal-t-v-cls-b0"]
    doc = docs["verified-formal-t-v-cls-b0"]
    assert classes(doc) == {"uniform-2k-o1-512": v1(s, "U2K-O512"), "uniform-8k-o1-512": v1(s, "U8K-O512")}
    assert list(classes(doc)) == ["uniform-2k-o1-512", "uniform-8k-o1-512"] and doc["admission"] is None
    assert doc["provenance"]["base_profiles"] == [{"id": "verified-formal-20261023-d-a2-b0",
                                                   "content_sha256": D_A2_B0["content_sha256"]}]
    v8 = classes(doc)["uniform-8k-o1-512"]["validation"]
    assert v8["points"] == ["L-U8K", "GW-U8K-O512"] and v8["measured_classes"] == ["v-uniform-8k-o512-c8",
                                                                                     "v-uniform-8k-o512"]


def test_one_v1_unit_gives_its_class_only(sessions):
    s = sessions("cls-8k")
    docs, holds = gen(s)
    assert holds == ["VG-32K", "VG-M8K-T", "V1-8K"]
    (doc,) = docs.values()
    assert classes(doc) == {"uniform-8k-o1-512": v1(s, "U8K-O512")}


def test_the_profile_states_the_slo_the_b0_server_and_the_section_12_provenance(sessions):
    s = sessions("adm")
    doc = one(s)
    raw = (s["out"] / "analysis.json").read_bytes()
    pins = s["stage_v_configs"][2]
    assert (doc["schema_version"], doc["profile_id"], doc["status"], doc["provisional_reason"]) == (
        1, "verified-formal-t-v-adm-b0", "verified", None)
    assert doc["slo"] == {"id": "S2", "terms": S2_TERMS}
    assert doc["server"] == {"vllm_version": "0.30.0", "model_repo": pins["core_b"]["repo"],
                             "model_revision": pins["core_b"]["revision"],
                             "fingerprint_sha256": cp.stage_b_fingerprint(pins), "num_gpu_blocks": 248,
                             "block_size": 528, "long_prefill_token_threshold": None}
    assert doc["provenance"] == {
        "source_session": "formal-t-v-adm", "analysis_path": "results/stage_v/formal-t-v-adm/analysis.json",
        "analysis_sha256": sha(raw), "prereg_section": "§12", "freeze_commit": FREEZE_V, "generator_commit": GEN,
        "deployed_commit": DEPLOYED, "capacity_sha256": CAPACITY_CONFIGS[2], "pins_sha256": PINS_SHA,
        "stage_d_sha256": t.SD_SHA, "stage_v_sha256": SV_SHA, "gateway_path_sha256": sv.gateway_path_sha256(),
        "capserve_sha256": s["session"]["capserve_sha256"],
        "base_profiles": [{"id": "verified-formal-20261017-c1b", "content_sha256": C1B["content_sha256"]}],
        "gpu": "NVIDIA GeForce RTX 5070 Ti", "gpu_driver": "617.42"}
    assert doc["content_sha256"] == cp.content_sha256(doc)


@pytest.mark.parametrize("key", ["adm", "adm-w1", "adm-pace", "adm-veto", "cls", "cls-8k"])
def test_every_generated_profile_loads_as_section_12(sessions, key):
    s = sessions(key)
    doc = one(s)
    prof = loaded(s, doc)
    assert prof.verified and prof.section == cp.STAGE_V_SECTION and prof.server_variant == "B0"
    assert [c.name for c in prof.classes] == [c["name"] for c in doc["classes"]]
    for c in prof.classes:
        assert prof.serving_class(c.name) is c
        assert prof.validated(c) is (c.bound == "validated")


def test_the_output_is_deterministic_and_only_the_generator_commit_varies(sessions):
    s = sessions("adm")
    doc, again, other = one(s), one(s), one(s, generator_commit="1c" * 20)
    assert pg.render(again) == pg.render(doc)
    assert {k for k in doc if doc[k] != other[k]} == {"provenance", "content_sha256"}
    assert {k for k in doc["provenance"] if doc["provenance"][k] != other["provenance"][k]} == {"generator_commit"}


def test_only_units_that_hold_with_every_point_pass_are_read():
    """044 產品對應: a unit is written only when analysis.json says it holds, each of its points PASSes and (PACE-2K) it
    is not vetoed; the order is analysis.json's."""
    units = {"VG-32K": {"classes": {"GW-U32K": "PASS"}, "vetoed": None, "holds": True},
             "V1-2K": {"classes": {"L-U2K": "PASS", "GW-U2K-O512": "PASS"}, "vetoed": None, "holds": False},
             "V1-8K": {"classes": {"L-U8K": "PASS", "GW-U8K-O512": "MARGINAL"}, "vetoed": None, "holds": True},
             "PACE-2K": {"classes": {"P-U2K": "PASS"}, "vetoed": True, "holds": True},
             "PACE-8K": {"classes": {"P-U8K": "PASS"}, "vetoed": None, "holds": True},
             "W1": {"classes": {"W-90": "PASS", "W-50": "TRIVIAL"}, "vetoed": None, "holds": False}}
    assert pg.holding_units({"units": units}) == ["VG-32K", "PACE-8K"]


def test_the_units_are_044s():
    """Every unit of configs/stage_v.toml is mapped: VG-32K / VG-M8K-T to no file, the rest to §12's items."""
    assert set(pg.NO_FILE_UNITS) | set(cp.COPY_UNITS.values()) | set(cp.V1_UNITS.values()) | {cp.ADMISSION_UNIT} == \
        set(sv.UNITS)
    assert (pg.COPY_BASE, pg.V1_BASE) == ("service/profiles/verified-formal-20261017-c1b.json",
                                          "service/profiles/verified-formal-20261023-d-a2-b0.json")


def test_an_unknown_unit_that_holds_is_refused(sessions, monkeypatch):
    s = sessions("adm")
    real = pg.holding_units
    monkeypatch.setattr(pg, "holding_units", lambda a: real(a) + ["V3"])
    assert "V3" in refused(s)


# ------------------------------------------------------------------ the served profile (044 服務規則) --

def settings(**over) -> cfg.Settings:
    return cfg.Settings.model_validate({**LOCAL, **over})


def test_the_paced_copy_serves_only_with_its_identical_pacing_and_queue(sessions):
    s = sessions("adm")
    prof, path = loaded(s, one(s)), sv.gateway_path_sha256()
    on = {"active_class": "uniform-2k", "prefill_pacing": {"burst_tokens": 9600, "rate_tokens_per_s": 5920}, **SLOW}
    for study in (True, False):                                   # capserve serve and deploy up alike
        assert [c.name for c in cfg.served_classes(settings(**on), prof, gateway_path_sha256=path,
                                                   study=study)] == ["uniform-2k"]
    assert [c.name for c in cfg.served_classes(
        settings(active_class="uniform-8k", prefill_pacing={"burst_tokens": 32256, "rate_tokens_per_s": 5800}, **SLOW),
        prof, gateway_path_sha256=path)] == ["uniform-8k"]
    for over in ({"prefill_pacing": {"burst_tokens": 9600, "rate_tokens_per_s": 5930}},
                 {"prefill_pacing": {"burst_tokens": 32256, "rate_tokens_per_s": 5800}},
                 {"max_queue_wait_s": 2.0, "request_deadline_s": 300.0}):
        with pytest.raises(cp.ProfileError, match="044"):
            cfg.served_classes(settings(**{**on, **over}), prof, gateway_path_sha256=path)
    with pytest.raises(cp.ProfileError, match="request path"):
        cfg.served_classes(settings(**on), prof, gateway_path_sha256="0" * 64)


def test_the_admission_section_serves_only_with_identical_parameters(sessions):
    s = sessions("adm-w1")
    prof, path = loaded(s, one(s)), sv.gateway_path_sha256()
    section = {k: v for k, v in admission(s).items() if k in ("policy", "budget", "max_queue", "classes")}
    got = cfg.served_classes(settings(admission=section, **SLOW), prof, gateway_path_sha256=path)
    assert [c.name for c in got] == ["uniform-2k", "uniform-8k"]
    for over in ({"admission": {**section, "max_queue": 16}}, {"max_queue_wait_s": 2.0, "request_deadline_s": 300.0},
                 {"active_class": "uniform-2k", "admission": None,
                  "prefill_pacing": {"burst_tokens": 9600, "rate_tokens_per_s": 5920}}):
        with pytest.raises(cp.ProfileError, match="044"):
            cfg.served_classes(settings(**{"admission": section, **SLOW, **over}), prof, gateway_path_sha256=path)
    with pytest.raises(cp.ProfileError, match="request path"):
        cfg.served_classes(settings(admission=section, **SLOW), prof, gateway_path_sha256="0" * 64)


def test_v1s_validated_classes_serve_512_output_tokens_without_a_policy(sessions):
    s = sessions("cls")
    prof = loaded(s, one(s))
    for name, c in (("uniform-2k-o1-512", 32), ("uniform-8k-o1-512", 8)):
        (got,) = cfg.served_classes(settings(active_class=name), prof, gateway_path_sha256="0" * 64)
        assert (got.max_concurrency, got.max_output_tokens) == (c, 512)
        with pytest.raises(cp.ProfileError, match="044"):
            cfg.served_classes(settings(active_class=name, prefill_pacing={"burst_tokens": 9600,
                                                                          "rate_tokens_per_s": 5920}),
                               prof, gateway_path_sha256=sv.gateway_path_sha256())


# ------------------------------------------------------------------ refusals --

@pytest.mark.parametrize("which, match", [
    ("stage_v", "[freeze] stage_v is not true"),
    ("stage_d", "[freeze] stage_d is not true in"),
    ("capacity", "[freeze] capacity is not true in"),
])
def test_refused_unless_stage_v_stage_d_and_capacity_are_frozen(sessions, which, match):
    """The generator's own freeze checks, besides the loader's (a base profile then no longer loads either)."""
    s = sessions("adm")
    cap, c, d = (copy.deepcopy(x) for x in (s["configs"][0], *s["stage_v_configs"][:2]))
    {"capacity": cap, "stage_v": c, "stage_d": d}[which]["freeze"][which] = False
    got = problems(s, configs=(cap, *s["configs"][1:]), stage_v_configs=(c, d, *s["stage_v_configs"][2:]))
    assert any(p.startswith(match) for p in got)


def test_refused_without_the_freeze_evidence(sessions, tmp_path):
    s = clone(sessions("adm"), tmp_path)
    (s["root"] / "results" / "stage_v" / t.MINI_ID / "session.json").unlink()
    assert any(p.startswith("freeze evidence: freeze.mini: no readable") for p in problems(s))   # besides the report's


@pytest.mark.parametrize("commit, git, match", [
    ("f5" * 19, VGit(), "40 lowercase hex"),
    (FREEZE_V, VGit(parent_frozen=True), "already frozen"),
    (FREEZE_V, VGit(history=[FREEZE_V, "e1" * 20, BEFORE_V], files={("e1" * 20, STAGE_V_TOML): FROZEN_V}),
     "re-freeze"),
    (FREEZE_V, VGit(files={(FREEZE_V, STAGE_V_TOML): UNFROZEN_V}), "not the freeze commit"),
    (FREEZE_V, VGit(ancestor=False), "DEPLOYED_COMMIT"),
    (FREEZE_V, VGit(files={(FREEZE_V, STAGE_V_TOML): FROZEN_V + b"# changed\n"}), "changed after the freeze"),
])
def test_the_freeze_commit_must_be_the_commit_that_set_stage_v_and_the_session_ran_it(sessions, commit, git, match):
    assert match in refused(sessions("adm"), freeze_commit=commit, git=git)


@pytest.mark.parametrize("path", ["benchmarks/stage_v_report.py", "benchmarks/stage_v_analysis.py",
                                  "benchmarks/stage_d_plan.py", "benchmarks/capacity_analysis.py"])
def test_a_frozen_module_changed_after_the_freeze_is_refused(sessions, path):
    other = (ROOT / path).read_bytes() + b"# changed\n"
    msg = refused(sessions("adm"), git=VGit(files={(DEPLOYED, path): other}))
    assert f"{path} at the session's DEPLOYED_COMMIT" in msg and "Stage V" in msg
    msg = refused(sessions("adm"), git=VGit(files={(FREEZE_V, path): other, (DEPLOYED, path): other}))
    assert f"{path} imported by this generator" in msg


def test_the_frozen_modules_are_the_stage_v_harness_and_the_stage_d_and_stage_c_modules():
    assert [pg.frozen_file(m)[0] for m in pg.STAGE_V_MODULES] == list(sv.HARNESS_FILES) + [
        pg.frozen_file(m)[0] for m in pg.STAGE_D_MODULES]


def test_the_request_path_at_the_deployed_commit_must_be_the_frozen_one(sessions):
    """044 產品對應: the units were validated on the frozen request path; DEPLOYED_COMMIT's own closure is hashed (a
    file outside it, e.g. this generator or deploy, does not count)."""
    s = sessions("adm")
    for rel in ("service/capserve/proxy.py", "service/capserve/__init__.py", "service/capserve/gateway_path.py"):
        changed = (ROOT / rel).read_bytes() + b"# changed\n"
        assert "request path at the session's DEPLOYED_COMMIT" in refused(s, git=VGit(files={(DEPLOYED, rel): changed}))
    for rel in ("service/capserve/profile_gen.py", "service/capserve/deploy.py"):
        changed = (ROOT / rel).read_bytes() + b"# changed\n"
        assert one(s, git=VGit(files={(DEPLOYED, rel): changed}))["profile_id"] == "verified-formal-t-v-adm-b0"
    imported = (ROOT / "service/capserve/__main__.py").read_bytes() + b"import capserve.deploy\n"
    assert "request path" in refused(s, git=VGit(files={(DEPLOYED, "service/capserve/__main__.py"): imported}))
    assert "cannot list service/capserve" in refused(s, git=VGit(listing=None))
    missing = [f for f in capserve_files() if not f.endswith("/proxy.py")]
    assert "request path" in refused(s, git=VGit(listing=missing))
    c = copy.deepcopy(s["stage_v_configs"][0])
    c["freeze"]["gateway_path_sha256"] = "0" * 64
    assert "request path at the session's DEPLOYED_COMMIT" in refused(s, stage_v_configs=(c, *s["stage_v_configs"][1:]))


def test_deployed_gateway_path_sha256_with_a_real_repository(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    tpg.git(root, "init", "-q")
    files = {rel: (ROOT / rel).read_bytes() for rel in capserve_files()}
    first = tpg.commit_files(root, {**files, "service/README.md": b"x\n"})
    second = tpg.commit_files(root, {"service/capserve/deploy.py": files["service/capserve/deploy.py"] + b"# x\n"})
    third = tpg.commit_files(root, {"service/capserve/proxy.py": files["service/capserve/proxy.py"] + b"# x\n"})
    git = pg.git_runner(root)
    want = gpath.gateway_path_sha256(ROOT)
    assert pg.deployed_gateway_path_sha256(first, git=git) == (want, [])
    assert pg.deployed_gateway_path_sha256(second, git=git) == (want, [])
    sha3, problems = pg.deployed_gateway_path_sha256(third, git=git)
    assert problems == [] and sha3 != want
    assert pg.deployed_gateway_path_sha256("0" * 40, git=git)[0] is None


@pytest.mark.parametrize("data", [b"def (:\n", b"\xff\xfe not UTF-8\n", b"x = 1\0\n"])
def test_an_unparsable_request_path_file_at_the_deployed_commit_is_refused_not_a_crash(sessions, tmp_path, monkeypatch,
                                                                                      capsys, data):
    """044 實作補記十第 3 項: a closure file ast cannot parse (bad syntax, not UTF-8, a NUL byte) is a refusal (exit 2,
    nothing written), never a traceback."""
    files = {(DEPLOYED, "service/capserve/proxy.py"): data}
    sha256, got = pg.deployed_gateway_path_sha256(DEPLOYED, git=VGit(files=files))
    assert sha256 is None and len(got) == 1 and "cannot hash the request path at DEPLOYED_COMMIT" in got[0]
    assert "cannot hash the request path at DEPLOYED_COMMIT" in refused(sessions("adm"), git=VGit(files=files))
    out = tmp_path / "profiles"
    out.mkdir()
    assert cli(sessions("adm"), monkeypatch, out, git=VGit(files=files)) == 2
    assert list(out.iterdir()) == [] and "cannot hash the request path" in capsys.readouterr().err


def test_the_deployed_request_path_is_hashed_by_the_frozen_research_copy(sessions, monkeypatch):
    """044 實作補記十第 4 項: capserve (gateway_path.py too) may change after V-CLS; DEPLOYED_COMMIT's closure is hashed
    by stage_v_plan's frozen copy of the algorithm, so a later change of the product module never refuses a valid
    session (both agree on this checkout: test_the_generator_stays_outside_the_request_path)."""
    monkeypatch.setattr(gpath, "gateway_path_sha256", lambda root=gpath.ROOT: "0" * 64)
    assert one(sessions("adm"))["profile_id"] == "verified-formal-t-v-adm-b0"


def stage_v_files() -> dict:
    """What a Stage V freeze commit holds besides stage_v.toml, this checkout's bytes: the frozen modules, the tracked
    service/capserve files and the base profiles."""
    files = {pg.frozen_file(m)[0]: pg.frozen_file(m)[1].read_bytes() for m in pg.STAGE_V_MODULES}
    return {**files, **{rel: (ROOT / rel).read_bytes() for rel in (*capserve_files(), pg.COPY_BASE, pg.V1_BASE)}}


def real_git(root: Path, deployed: str):
    """git of a real repository, the synthetic sessions' DEPLOYED_COMMIT read as the commit deployed."""
    git = pg.git_runner(root)
    return lambda *args: git(*(a.replace(DEPLOYED, deployed) for a in args))


FIRST_FREEZE_V = b"[freeze]\nstage_v = true\n# the first freeze: its mini, its request path\n"


@pytest.mark.parametrize("path", ["unfreeze and re-freeze", "frozen throughout"])
def test_a_session_under_the_stop_rules_new_freeze_is_refused_until_its_decision_amends_the_rule(sessions, tmp_path,
                                                                                                 path):
    """044 實作補記十第 2 項 (option B, fail closed): after a gateway-caused invalid run the stop rule fixes capserve and
    makes a new freeze and mini; the generator accepts only the first freeze, so a session under the new one is
    refused with either freeze commit, the one refusal pointing at the note (that freeze's decision must amend this
    rule first). Real git."""
    root = tmp_path / "r"
    root.mkdir()
    tpg.git(root, "init", "-q")
    proxy, files = "service/capserve/proxy.py", stage_v_files()
    tpg.commit_files(root, {**files, STAGE_V_TOML: UNFROZEN_V, proxy: files[proxy] + b"# the defect\n"})
    first = tpg.commit_files(root, {STAGE_V_TOML: FIRST_FREEZE_V})
    fix = {proxy: files[proxy], STAGE_V_TOML: UNFROZEN_V} if path.startswith("unfreeze") else {proxy: files[proxy]}
    tpg.commit_files(root, fix)
    new = tpg.commit_files(root, {STAGE_V_TOML: FROZEN_V})           # the new mini and request path: the session's
    git = real_git(root, tpg.commit_files(root, {"x.txt": b"deployed\n"}))
    for commit, why in ((new, "already frozen"), (first, "changed after the freeze")):
        (msg,) = problems(sessions("adm"), freeze_commit=commit, git=git)
        assert why in msg and "if this is the stop rule's new freeze, " in msg and "044 實作補記十" in msg


@pytest.mark.parametrize("freeze", [{"gateway_path_sha256": "1" * 64}, {"mini": "smoke-v-new"},
                                    {"gateway_path_sha256": "1" * 64, "mini": "smoke-v-new"}])
def test_after_a_new_freeze_the_old_sessions_profile_no_longer_loads(sessions, freeze):
    """044 停止規則 「舊結果只對舊的 sha 有效，依服務規則不能上線」 holds mechanically for every class, V1's too (實作補記十
    第 2 項): a new freeze changes configs/stage_v.toml ([freeze] mini, gateway_path_sha256), and the §12 loader
    compares the profile's stage_v_sha256 and gateway_path_sha256 with the checkout's."""
    s = sessions("cls")
    doc = one(s)
    cap, _, cap_sha, _ = s["configs"]
    c, d, p, _, sd_sha, pins_sha = s["stage_v_configs"]
    c = copy.deepcopy(c)
    c["freeze"].update(freeze)
    new = cp.FrozenConfigs(cap, cap_sha, pins_sha, d, sd_sha, c, sha(FROZEN_V + b"# the new freeze\n"))
    with pytest.raises(cp.ProfileError) as exc:
        cp.parse_profile(pg.render(doc), source=doc["profile_id"], allow_provisional=False, pins=p, frozen=new)
    assert "provenance.stage_v_sha256" in str(exc.value)
    assert ("provenance.gateway_path_sha256" in str(exc.value)) is ("gateway_path_sha256" in freeze)


@pytest.mark.parametrize("rel", [pg.COPY_BASE, pg.V1_BASE])
def test_a_base_profile_must_be_its_bytes_at_the_freeze_commit(sessions, tmp_path, rel):
    s = sessions("cls")
    other = (ROOT / rel).read_bytes().replace(b'"gpu_driver"', b'"gpu_driver" ', 1)
    msg = refused(s, git=VGit(files={(FREEZE_V, rel): other}))
    assert f"base profile {rel} is not its bytes at the Stage V freeze commit" in msg
    assert f"no {rel} at the Stage V freeze commit" in refused(s, git=VGit(files={(FREEZE_V, rel): None}))
    c = clone(s, tmp_path)
    (c["root"] / "service").unlink()
    shutil.copytree(SERVICE / "profiles", c["root"] / "service" / "profiles")
    doc = json.loads((ROOT / rel).read_text())
    doc["provenance"]["gpu_driver"] = "999.0"
    (c["root"] / rel).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    msg = refused(c)
    assert "not its bytes at the Stage V freeze commit" in msg and "content_sha256" in msg   # and it no longer loads


def test_refused_unless_the_session_ran_this_harness(sessions, tmp_path):
    s = clone(sessions("adm"), tmp_path)
    path = s["out"] / "session.json"
    rec = json.loads(path.read_text())
    rec["harness_sha256"] = "0" * 64
    path.write_text(json.dumps(rec))
    msg = refused(s)
    assert "stage_v_report:" in msg and "harness_sha256" in msg


def test_refused_unless_the_session_is_results_stage_v_formal(sessions, tmp_path):
    s = clone(sessions("adm"), tmp_path)
    elsewhere = s["root"] / "results" / "other" / s["out"].name
    shutil.copytree(s["out"], elsewhere.parent / elsewhere.name)
    assert "is not results/stage_v/<session_id>" in refused(s, session=elsewhere)


def test_a_mini_session_never_gives_a_profile(tmp_path):
    s = make_session(tmp_path, "mini", mode="mini")
    assert s["out"].name.startswith("smoke-v-")
    msg = refused(s)
    assert "not a formal- session" in msg and "stage_v_report:" in msg and "analysis.json" in msg


def test_refused_without_an_analysis_json(sessions, tmp_path):
    s = clone(sessions("adm"), tmp_path)
    (s["out"] / "analysis.json").unlink()
    assert "run stage_v_report on the session first" in refused(s)


def test_an_analysis_json_with_other_bytes_or_content_is_refused(sessions, tmp_path):
    """021 R3: the frozen stage_v_report re-run must reproduce analysis.json byte for byte; the generator reads its
    units only after that."""
    s = clone(sessions("adm"), tmp_path)
    path = s["out"] / "analysis.json"
    doc = json.loads(path.read_text())
    path.write_text(json.dumps(doc, indent=2) + "\n")                       # the same content, other bytes
    assert "byte for byte" in refused(s) and "none, only the bytes" in refused(s)
    doc["units"]["W1"]["holds"] = False                                     # edited content
    path.write_text(json.dumps(doc, indent=1) + "\n")
    assert "differing top-level keys: ['units']" in refused(s)


def test_refused_unless_every_b0_boot_ran_the_product_server(sessions, tmp_path):
    s = clone(sessions("adm"), tmp_path)
    path = s["out"] / "session.json"
    rec = json.loads(path.read_text())
    tag = next(b["tag"] for b in rec["boots"] if b["phase"] == "V")
    rec["runs"][tag]["argv"].append("--enforce-eager")
    path.write_text(json.dumps(rec))
    msg = refused(s)
    assert "server B0" in msg and "different servers" in msg


def test_refused_with_another_slo(sessions):
    s = sessions("adm")
    c = copy.deepcopy(s["stage_v_configs"][0])
    c["slo"]["tpot_p95_ms"] = 200
    assert "stage_v.toml's SLO" in refused(s, stage_v_configs=(c, *s["stage_v_configs"][1:]))


def test_a_malformed_session_is_refused_not_a_crash(sessions, tmp_path):
    s = clone(sessions("adm"), tmp_path)
    (s["out"] / "session.json").write_text("[]")
    assert "malformed" in refused(s) or "cannot read" in refused(s)


# ------------------------------------------------------------------ 044 停止規則: no rerun of a valid session --

FIRST, BETWEEN, RERUN = "formal-t-v-adm", "formal-t-v-cls", "formal-t-v-adm-rerun"


@pytest.fixture(scope="module")
def reruns(tmp_path_factory):
    """The review's case (044 實作補記十): a valid V-ADM where no unit holds, a V-CLS 6 h later, then a second V-ADM
    under the same freeze 6 h after that where every unit holds."""
    return make_sessions(tmp_path_factory.mktemp("reruns"), ("V-ADM", FIRST, SCENARIOS["adm-none"][1], 0),
                         ("V-CLS", BETWEEN, (), 6), ("V-ADM", RERUN, (), 12))


def earlier(s: dict) -> Path:
    """The first V-ADM in a (cloned) reruns root."""
    return s["root"].joinpath(*sv.RESULTS, FIRST)


def edit_session(d: Path, **fields) -> None:
    path = d / "session.json"
    rec = json.loads(path.read_text())
    rec.update(fields)
    path.write_text(json.dumps(rec))


def slow_end_sentinel(d: Path) -> None:
    """The end sentinel 5 % slower than the start, on disk, its derived files' sha256 rerecorded in session.json:
    stage_v_report's sentinel gate fails (drift, 030 8.6), nothing else does."""
    rec = json.loads((d / "session.json").read_text())
    for row in map(json.loads, (d / "rows.jsonl").read_text().splitlines()):
        if row.get("position") == "end" and row.get("derived"):
            path = d / row["derived"]
            reqs = [dict(q, ttft_ms=q["ttft_ms"] * 1.05) for q in map(json.loads, path.read_text().splitlines())]
            path.write_text("".join(json.dumps(q) + "\n" for q in reqs))
            rec["derived_sha256"][row["tag"]] = sha(path.read_bytes())
    (d / "session.json").write_text(json.dumps(rec))


def report_refusal(s: dict, d: Path) -> str:
    """Why stage_v_report refuses the session at d ("" when it accepts it), with s's configs and references."""
    c, sd_cfg, pins, sv_sha, sd_sha, pins_sha = s["stage_v_configs"]
    cap = cr.load_session(d)
    kw = {"cfg": c, "stage_d_cfg": sd_cfg, "pins": pins, "stage_v_sha256": sv_sha, "stage_d_sha256": sd_sha,
          "pins_sha256": pins_sha, "repo_root": s["root"]}
    try:
        svr.report(cap, uniform=cr.load_uniform(UNIFORM), advanced_runs=None, **kw)
    except svr.ReportError as exc:
        return str(exc)
    return ""


def stop_rule(got: list) -> list:
    return [p for p in got if "044 停止規則" in p]


def test_a_rerun_of_a_valid_session_of_its_plan_is_refused(reruns, tmp_path, monkeypatch, capsys):
    """044 停止規則: 不重跑有效的 session 來換 PASS. The review's case: the first V-ADM is valid and no unit holds; the
    rerun under the same freeze, where every unit holds, gives no profile (refused, nothing written)."""
    rerun = reruns[RERUN]
    assert report_refusal(rerun, earlier(rerun)) == ""
    assert all(x["holds"] for x in analysis(rerun)["units"].values())
    (msg,) = problems(rerun)
    assert f"the earlier V-ADM session {FIRST} (started " in msg and f"{RERUN} " in msg
    assert "044 停止規則" in msg and "不重跑有效的 session 來換 PASS" in msg
    out = tmp_path / "profiles"
    out.mkdir()
    assert cli(rerun, monkeypatch, out) == 2
    assert list(out.iterdir()) == [] and "不重跑有效的 session 來換 PASS" in capsys.readouterr().err


def test_the_earlier_session_and_the_other_plan_still_give_their_results(reruns):
    """The rule looks back only, and only at the session's own plan: the first V-ADM is still a result (no unit
    holds) although a later rerun exists, and the V-CLS between the two V-ADM sessions writes V1's classes."""
    assert gen(reruns[FIRST]) == ({}, [])
    docs, holds = gen(reruns[BETWEEN])
    assert holds == ["VG-32K", "VG-M8K-T", "V1-2K", "V1-8K"] and list(docs) == [f"verified-{BETWEEN}-b0"]


@pytest.mark.parametrize("how, why", [
    ("aborted", "aborted"),
    ("env", "env-invalid"),
    ("gate", "sentinel gate failed"),
])
def test_an_earlier_session_that_is_not_valid_does_not_block_the_rerun(reruns, tmp_path, how, why):
    """044 停止規則: an aborted or env-invalid session, or one whose sentinel gate failed, unlocks nothing and is rerun
    under a new id; the rerun's units are the result."""
    rerun = clone(reruns[RERUN], tmp_path)
    {"aborted": lambda d: edit_session(d, aborted="RunError: x"), "env": lambda d: edit_session(d, env_valid=False),
     "gate": slow_end_sentinel}[how](earlier(rerun))
    assert why in report_refusal(rerun, earlier(rerun))
    docs, holds = gen(rerun)
    assert holds == ["PACE-2K", "PACE-8K", "W1"] and list(docs) == [f"verified-{RERUN}-b0"]


@pytest.mark.parametrize("key", ["stage_v_sha256", "harness_sha256", "gateway_path_sha256", "preflight"])
def test_an_earlier_session_under_another_freeze_does_not_block_the_rerun(reruns, tmp_path, key):
    """Only the same freeze counts (the recorded stage_v_sha256, harness_sha256 and gateway_path_sha256): after the
    stop rule's new freeze both sessions rerun. preflight: a session aborted in its preflight records none of them and
    has no rows.jsonl (never read)."""
    rerun = clone(reruns[RERUN], tmp_path)
    d = earlier(rerun)
    if key == "preflight":
        rec = json.loads((d / "session.json").read_text())
        for k in ("stage_v_sha256", "harness_sha256", "gateway_path_sha256"):
            rec.pop(k)
        (d / "session.json").write_text(json.dumps({**rec, "aborted": "RunError: preflight"}))
        (d / "rows.jsonl").unlink()
    else:
        edit_session(d, **{key: "0" * 64})
    docs, holds = gen(rerun)
    assert holds == ["PACE-2K", "PACE-8K", "W1"] and list(docs) == [f"verified-{RERUN}-b0"]


@pytest.mark.parametrize("how", ["no session.json", "not JSON", "no started_at", "no offset", "no plan.json",
                                 "empty directory"])
def test_an_earlier_session_of_the_plan_that_cannot_be_read_fails_closed(reruns, tmp_path, how):
    """A formal-* directory of the session's plan, or of a plan that cannot be read, without a readable session.json
    and started_at (with an offset, 044 實作補記八), or one stage_v_report cannot load: whether it is an earlier valid
    session is unknown, so the rerun is refused (besides the report's own fail-closed 2 h gap)."""
    rerun = clone(reruns[RERUN], tmp_path)
    d, name = earlier(rerun), FIRST
    if how == "no session.json":
        (d / "session.json").unlink()
    elif how == "not JSON":
        (d / "session.json").write_text("{")
    elif how == "no started_at":
        rec = json.loads((d / "session.json").read_text())
        rec.pop("started_at")
        (d / "session.json").write_text(json.dumps(rec))
    elif how == "no offset":
        edit_session(d, started_at="2026-11-02T09:00:00")
    elif how == "no plan.json":
        (d / "plan.json").unlink()
    else:
        name = "formal-t-v-x"
        rerun["root"].joinpath(*sv.RESULTS, name).mkdir()
    got = stop_rule(problems(rerun))
    assert got and all(name in p and "fail closed" in p for p in got)


def test_a_session_without_a_readable_start_fails_the_stop_rule_closed(reruns, tmp_path):
    rerun = clone(reruns[RERUN], tmp_path)
    rec = json.loads((rerun["out"] / "session.json").read_text())
    rec.pop("started_at")
    (rerun["out"] / "session.json").write_text(json.dumps(rec))
    (msg,) = stop_rule(problems(rerun))
    assert "cannot tell which V-ADM sessions came before it" in msg and "fail closed" in msg


def test_an_unreadable_session_of_the_other_plan_is_no_earlier_session_of_this_one(reruns, tmp_path):
    """The V-CLS without a session.json still refuses the rerun (the report's 2 h gap, 044 實作補記七), and it is no
    earlier V-ADM: the stop rule never asks whether it is valid. It is the rerun's previous_formal, though, so the walk
    to the earlier formal sessions stops there (fail closed, 044 實作補記十第 10 項; the first V-ADM aborted, so that
    only the walk refuses)."""
    rerun = clone(reruns[RERUN], tmp_path)
    (rerun["root"].joinpath(*sv.RESULTS, BETWEEN) / "session.json").unlink()
    edit_session(earlier(rerun), aborted="RunError: x")
    got = problems(rerun)
    assert any(BETWEEN in p and "044 停止規則" not in p for p in got)
    (msg,) = stop_rule(got)
    assert f"past {BETWEEN}" in msg and "no readable session.json" in msg and "fail closed" in msg
    assert "is an earlier valid" not in msg


def test_the_runner_records_each_formal_sessions_previous_formal(reruns):
    """What the walk reads (run_stage_v.gap_problems at the start): the formal session that finished last among those
    that started before, of either plan; null for the first."""
    def previous(sid):
        return json.loads((reruns[RERUN]["root"].joinpath(*sv.RESULTS, sid) / "session.json").read_text())[
            "previous_formal"]
    assert previous(FIRST) is None
    assert previous(BETWEEN)["session_id"] == FIRST and previous(RERUN)["session_id"] == BETWEEN


@pytest.mark.parametrize("how, why", [
    ("derived file deleted", "derived file"),
    ("rows.jsonl edited", "rows.jsonl"),
    ("plan.json edited", "plan.json is not the plan the session ran"),
    ("pins_sha256 edited", "pins.toml sha256"),
    ("stage_d_sha256 edited", "stage_d.toml sha256"),
    ("aborted, derived file deleted", "derived file"),
])
def test_an_earlier_session_whose_files_are_not_its_records_fails_closed(reruns, tmp_path, how, why):
    """044 實作補記十第 10 項: only positive evidence that an earlier session was invalid lets the rerun through. A file
    of the earlier valid session that is not what its session.json records (stage_v_report's sha256 checks, against
    the session's own records) is no such evidence: the review's probes (a derived file deleted, a newline appended to
    rows.jsonl, an edited sha256) gave a profile before. The files are checked before aborted / env_valid are read."""
    rerun = clone(reruns[RERUN], tmp_path)
    d = earlier(rerun)
    if how.startswith("aborted"):
        edit_session(d, aborted="RunError: x")
    if how.endswith("derived file deleted"):
        next(iter(sorted((d / "derived").glob("*-requests.jsonl")))).unlink()
    elif how == "rows.jsonl edited":
        (d / "rows.jsonl").write_text((d / "rows.jsonl").read_text() + "\n")
    elif how == "plan.json edited":
        plan = json.loads((d / "plan.json").read_text())
        (d / "plan.json").write_text(json.dumps({**plan, "hour_cap_h": 99}))
    else:
        edit_session(d, **{how.split()[0]: "0" * 64})                    # as if pins.toml / stage_d.toml changed
    got = stop_rule(problems(rerun))
    assert got and all(FIRST in p and "fail closed" in p for p in got)
    assert any("are not what its session.json records" in p and why in p for p in got)
    assert not any("which ran under the same freeze and is valid" in p for p in got)


def test_a_renamed_earlier_session_fails_closed(reruns, tmp_path):
    """The review's probe: the earlier valid V-ADM copied back under another name (its session_id unchanged) gave a
    profile (the report then counts it as its own previous session). A directory that is not its session_id is
    refused (fail closed)."""
    rerun = clone(reruns[RERUN], tmp_path)
    renamed = earlier(rerun).with_name(f"{FIRST}-from-wsl")
    earlier(rerun).rename(renamed)
    got = stop_rule(problems(rerun))
    assert got and all("fail closed" in p for p in got)
    assert any(renamed.name in p and f"records session_id {FIRST!r}" in p for p in got)


@pytest.mark.parametrize("missing", [FIRST, BETWEEN])
def test_a_formal_session_that_is_not_here_fails_closed(reruns, tmp_path, missing):
    """The review's probe: without the earlier valid V-ADM's directory on the Mac the rerun gave a profile. Every
    formal session the previous_formal chain names must be here, of either plan (the missing V-CLS could hide an
    earlier V-ADM behind it; the first V-ADM aborted then, so that only the walk refuses)."""
    rerun = clone(reruns[RERUN], tmp_path)
    shutil.rmtree(rerun["root"].joinpath(*sv.RESULTS, missing))
    if missing == BETWEEN:
        edit_session(earlier(rerun), aborted="RunError: x")
    got = stop_rule(problems(rerun))
    assert got and all(missing in p and "fail closed" in p for p in got)
    assert any("fetch every formal Stage V session before generating" in p for p in got)


@pytest.mark.parametrize("prev, why", [({"session_id": RERUN}, f"returns to {RERUN}"), (FIRST, "names no session")])
def test_a_previous_formal_chain_that_cannot_be_followed_fails_closed(reruns, tmp_path, prev, why):
    """A chain that returns to a session, or a previous_formal that is no record (hand-edited records): refused."""
    rerun = clone(reruns[RERUN], tmp_path)
    edit_session(rerun["root"].joinpath(*sv.RESULTS, BETWEEN), previous_formal=prev)
    edit_session(earlier(rerun), aborted="RunError: x")                  # only the walk refuses
    (msg,) = stop_rule(problems(rerun))
    assert why in msg and "fail closed" in msg


@pytest.mark.parametrize("how, why", [
    ("not under the GPU lock", "did not run under the held GPU lock"),
    ("no env_valid", "env-invalid"),
])
def test_an_earlier_session_the_report_cannot_evaluate_cleanly_fails_closed(reruns, tmp_path, how, why):
    """stage_v_report.refusal_problems of the earlier session (besides its files) is no evidence that it was invalid:
    the rerun is refused, the problems named. Only aborted or env_valid false are (a missing env_valid is not)."""
    rerun = clone(reruns[RERUN], tmp_path)
    d = earlier(rerun)
    if how == "not under the GPU lock":
        edit_session(d, gpu_lock_held=False)
    else:
        rec = json.loads((d / "session.json").read_text())
        rec.pop("env_valid")
        (d / "session.json").write_text(json.dumps(rec))
    got = stop_rule(problems(rerun))
    assert got and all(FIRST in p and "fail closed" in p for p in got)
    assert any("stage_v_report refuses to evaluate" in p and why in p for p in got)


def test_an_earlier_session_whose_report_crashes_fails_closed(reruns, tmp_path, monkeypatch):
    """Only stage_v_report's own ReportError (with no refusal problem) says the earlier session is invalid; anything
    else it raises (a record nobody foresaw) is refused, never read as invalid."""
    rerun = clone(reruns[RERUN], tmp_path)
    report = svr.report

    def crashes(cap, **kw):
        if cap["dir"] == earlier(rerun):
            raise TypeError("an unforeseen record")
        return report(cap, **kw)
    monkeypatch.setattr(svr, "report", crashes)
    got = stop_rule(problems(rerun))
    assert got and all(FIRST in p and "fail closed" in p and "TypeError: an unforeseen record" in p for p in got)


def test_an_earlier_session_whose_refusal_check_crashes_fails_closed(reruns, tmp_path):
    """An earlier same-plan session whose plan.json lost its points (its plan_sha256 rerecorded, so its files are its
    records): stage_v_report.refusal_problems itself raises. Refused (fail closed), never a crash."""
    rerun = clone(reruns[RERUN], tmp_path)
    d = earlier(rerun)
    plan = json.loads((d / "plan.json").read_text())
    plan.pop("points")
    (d / "plan.json").write_text(json.dumps(plan))
    edit_session(d, plan_sha256=svr.sha256_json(plan))
    got = stop_rule(problems(rerun))
    assert got and all(FIRST in p and "fail closed" in p for p in got)
    assert any("KeyError" in p for p in got)


def test_a_committed_directory_that_is_no_session_never_blocks(sessions, tmp_path):
    """results/stage_v/prereg-sim (committed, no session.json) is no formal session: the first valid V-ADM still writes
    its profile."""
    s = clone(sessions("adm"), tmp_path)
    sim = s["root"].joinpath(*sv.RESULTS, "prereg-sim")
    sim.mkdir()
    (sim / "sim.json").write_text("{}")
    docs, holds = gen(s)
    assert holds == ["PACE-2K", "PACE-8K", "W1"] and list(docs) == ["verified-formal-t-v-adm-b0"]


# The runner's own aborted attempts before the first valid V-ADM (044 實作補記十第 10 項).
ATTEMPTS = (("V-ADM", "formal-t-v-a1", "port busy", 0), ("V-ADM", "formal-t-v-a2", "start sentinel", 3),
            ("V-ADM", "formal-t-v-a3", "vLLM", 6), ("V-ADM", "formal-t-v-a4", "port taken", 9),
            ("V-CLS", "formal-t-v-c1", "port busy", 12), ("V-ADM", "formal-t-v-b", (), 15))


@pytest.fixture(scope="module")
def attempts(tmp_path_factory):
    return make_sessions(tmp_path_factory.mktemp("attempts"), *ATTEMPTS)["formal-t-v-b"]


def test_the_runners_aborted_attempts_do_not_block_the_first_valid_session(attempts):
    """Aborted before the first V boot (port busy), at the start sentinel, in the preflight (it records no
    previous_formal: the walk goes on from the session the runner's own rule finds here) and at a later boot: none is
    valid, so the V-ADM after them writes its profile."""
    rec = {sid: json.loads((attempts["root"].joinpath(*sv.RESULTS, sid) / "session.json").read_text())
           for _, sid, _, _ in ATTEMPTS}
    assert all(rec[sid]["aborted"] for _, sid, how, _ in ATTEMPTS if isinstance(how, str))
    assert "previous_formal" not in rec["formal-t-v-a3"] and "stage_v_sha256" not in rec["formal-t-v-a3"]
    assert rec["formal-t-v-a4"]["previous_formal"]["session_id"] == "formal-t-v-a3"
    docs, holds = gen(attempts)
    assert holds == ["PACE-2K", "PACE-8K", "W1"] and list(docs) == ["verified-formal-t-v-b-b0"]


@pytest.mark.parametrize("how", ["behind the preflight abort", "the preflight abort not aborted"])
def test_the_walk_goes_past_a_preflight_abort_fail_closed(attempts, tmp_path, how):
    """A session aborted in its preflight records no previous_formal; the walk goes on from the formal session that
    finished last among those here that started before it (run_stage_v.gap_problems), so a missing session behind it
    is still found. A session that records no previous_formal and was not aborted is refused (fail closed)."""
    s = clone(attempts, tmp_path)
    results = s["root"].joinpath(*sv.RESULTS)
    if how == "behind the preflight abort":
        shutil.rmtree(results / "formal-t-v-a1")
        missing = "formal-t-v-a1"
    else:
        rec = json.loads((results / "formal-t-v-a3" / "session.json").read_text())
        rec.pop("aborted")
        (results / "formal-t-v-a3" / "session.json").write_text(json.dumps(rec))
        missing = "formal-t-v-a3"
    got = stop_rule(problems(s))
    assert got and all(missing in p and "fail closed" in p for p in got)


# ------------------------------------------------------------------ the CLI --

def cli(s: dict, monkeypatch, out_dir, *, git=None, freeze=FREEZE_V, extra=()) -> int:
    monkeypatch.setattr(pg, "load_configs", lambda root: s["configs"])
    monkeypatch.setattr(pg, "load_stage_v_configs", lambda root: s["stage_v_configs"])
    return pg.main(["--validation-session", str(s["out"]), "--freeze-commit", freeze, "--out-dir", str(out_dir),
                    *extra], root=s["root"], git=git or VGit())


def test_main_writes_the_sessions_profile_and_prints_a_summary(sessions, tmp_path, monkeypatch, capsys):
    s = sessions("adm")
    out = tmp_path / "profiles"
    out.mkdir()
    capsys.readouterr()                                                     # the session's own report output
    assert cli(s, monkeypatch, out) == 0
    doc = one(s, generator_commit=CLEAN_HEAD)
    assert [p.name for p in out.iterdir()] == ["verified-formal-t-v-adm-b0.json"]
    assert (out / "verified-formal-t-v-adm-b0.json").read_bytes() == pg.render(doc)
    assert cp.load_profile(out / "verified-formal-t-v-adm-b0.json", allow_provisional=False,
                           pins=s["stage_v_configs"][2], frozen=frozen_configs(s)).section == "§12"
    printed = json.loads(capsys.readouterr().out)
    assert printed["holds"] == ["PACE-2K", "PACE-8K", "W1"]
    assert [p["profile_id"] for p in printed["profiles"]] == ["verified-formal-t-v-adm-b0"]
    assert printed["profiles"][0]["admission"] == "weighted-v1"
    assert printed["profiles"][0]["classes"] == {
        "uniform-2k": {"status": "verified", "max_concurrency": 32, "bound": "exact",
                       "pacing": {"burst_tokens": 9600, "rate_tokens_per_s": 5920}, "unit": "PACE-2K"},
        "uniform-8k": {"status": "verified", "max_concurrency": 8, "bound": "exact",
                       "pacing": {"burst_tokens": 32256, "rate_tokens_per_s": 5800}, "unit": "PACE-8K"}}


@pytest.mark.parametrize("key, admission, classes", [
    ("adm-w1", "weighted-v1", {"uniform-2k": (32, "exact", None, None), "uniform-8k": (8, "exact", None, None)}),
    ("cls", None, {"uniform-2k-o1-512": (32, "validated", None, "V1-2K"),
                   "uniform-8k-o1-512": (8, "validated", None, "V1-8K")}),
])
def test_the_summary_says_what_a_section_12_profile_adds(sessions, tmp_path, monkeypatch, capsys, key, admission,
                                                         classes):
    """044 實作補記十第 5 項: per profile its admission policy (or null), per class its pacing (or null) and unit (its
    validation's, or null), so a W1-only profile is not read as a plain copy of c1b."""
    s, out = sessions(key), tmp_path / "profiles"
    out.mkdir()
    capsys.readouterr()                                                     # the session's own report output
    assert cli(s, monkeypatch, out) == 0
    (profile,) = json.loads(capsys.readouterr().out)["profiles"]
    assert profile["admission"] == admission
    assert profile["classes"] == {name: {"status": "verified", "max_concurrency": c, "bound": bound, "pacing": pacing,
                                         "unit": unit} for name, (c, bound, pacing, unit) in classes.items()}


@pytest.mark.parametrize("key, holds", [("adm-none", []), ("cls-vg", ["VG-32K", "VG-M8K-T"])])
def test_main_writes_nothing_and_says_so_when_no_unit_writes_a_profile(sessions, tmp_path, monkeypatch, capsys, key,
                                                                       holds):
    """Decision (044 實作補記九): a reproduced session whose units write nothing is a result, not a refusal: exit 0,
    nothing written, and the summary says so."""
    s, out = sessions(key), tmp_path / "profiles"
    out.mkdir()
    capsys.readouterr()                                                     # the session's own report output
    assert cli(s, monkeypatch, out) == 0
    assert list(out.iterdir()) == []
    text = capsys.readouterr().out
    printed = json.loads(text)
    assert printed["profiles"] == [] and printed["holds"] == holds
    assert "no profile" in printed["written"] and "044 產品對應" in text      # printed as is, not \u escapes



@pytest.mark.parametrize("git, match", [
    (VGit(dirty=" M service/capserve/profile_gen.py\n"), "uncommitted"),
    (VGit(head=None), "git rev-parse HEAD"),
    (VGit(ancestor=False), "DEPLOYED_COMMIT"),
])
def test_main_refuses_and_writes_nothing(sessions, tmp_path, monkeypatch, capsys, git, match):
    out = tmp_path / "profiles"
    out.mkdir()
    assert cli(sessions("adm"), monkeypatch, out, git=git) == 2
    assert list(out.iterdir()) == [] and match in capsys.readouterr().err


def test_main_never_overwrites_and_never_writes_under_results(sessions, tmp_path, monkeypatch, capsys):
    s = sessions("adm")
    out = tmp_path / "profiles"
    out.mkdir()
    keep = out / "verified-formal-t-v-adm-b0.json"
    keep.write_text("keep")
    assert cli(s, monkeypatch, out) == 2
    assert keep.read_text() == "keep" and "already exists" in capsys.readouterr().err
    research = s["root"] / "results" / "profiles"
    research.mkdir()
    assert cli(s, monkeypatch, research) == 2
    assert list(research.iterdir()) == [] and "results/" in capsys.readouterr().err
    research.rmdir()
    assert cli(s, monkeypatch, tmp_path / "missing") == 2
    assert "not an existing directory" in capsys.readouterr().err


def test_main_writes_nothing_when_the_generated_profile_fails_the_loader(sessions, tmp_path, monkeypatch, capsys):
    """044 產品對應 / 043 §1.1: the document passes the loader before anything is written (synthetic defect: a V1
    class at the O512 C_PROD 12, rehashed)."""
    build = pg.build_stage_v_profile

    def defective(*args, **kwargs):
        doc = build(*args, **kwargs)
        doc["classes"][-1].update(max_concurrency=12, max_queue=12)
        doc["content_sha256"] = cp.content_sha256(doc)
        return doc

    monkeypatch.setattr(pg, "build_stage_v_profile", defective)
    out = tmp_path / "profiles"
    out.mkdir()
    assert cli(sessions("cls"), monkeypatch, out) == 2
    err = capsys.readouterr().err
    assert list(out.iterdir()) == [] and "fails the loader" in err and "12" in err


def test_the_validation_session_takes_out_dir_and_the_session_option_never_reads_stage_v(sessions, tmp_path,
                                                                                          monkeypatch, capsys):
    s = sessions("adm")
    out = tmp_path / "profiles"
    out.mkdir()
    monkeypatch.setattr(pg, "load_configs", lambda root: s["configs"])
    monkeypatch.setattr(pg, "load_stage_v_configs", lambda root: s["stage_v_configs"])
    assert pg.main(["--validation-session", str(s["out"]), "--freeze-commit", FREEZE_V, "--out",
                    str(out / "p.json")], root=s["root"], git=VGit()) == 2
    assert "--out-dir" in capsys.readouterr().err
    for option in ("--out", "--out-dir"):
        target = out / "p.json" if option == "--out" else out
        assert pg.main(["--session", str(s["out"]), "--freeze-commit", FREEZE_V, option, str(target)],
                       root=s["root"], git=VGit()) == 2
        assert "--validation-session" in capsys.readouterr().err
    for argv in (["--session", str(s["out"]), "--validation-session", str(s["out"])], []):
        with pytest.raises(SystemExit) as exc:
            pg.main([*argv, "--freeze-commit", FREEZE_V, "--out-dir", str(out)], root=s["root"], git=VGit())
        assert exc.value.code == 2
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("text", [None, "", "[freeze]\nstage_v = false\n", "[freeze]\nstage_v = 'true'\n"])
def test_main_refuses_until_stage_v_is_frozen(tmp_path, capsys, text):
    root = tpg.cli_root(tmp_path, (ROOT / "configs" / "capacity.toml").read_text())
    shutil.copy(ROOT / "configs" / "stage_d.toml", root / "configs" / "stage_d.toml")
    if text is not None:
        (root / STAGE_V_TOML).write_text(text)
    out = tmp_path / "out"
    out.mkdir()
    argv = ["--validation-session", str(root / "results" / "stage_v" / "formal-x"), "--freeze-commit", FREEZE_V,
            "--out-dir", str(out)]
    git = VGit()
    assert pg.main(argv, root=root, git=git) == 2
    assert list(out.iterdir()) == [] and "[freeze] stage_v" in capsys.readouterr().err
    assert git.calls == []                         # refused before asking git or reading the session
    (root / STAGE_V_TOML).write_text("[freeze\n")
    assert pg.main(argv, root=root, git=VGit()) == 2
    assert "cannot read" in capsys.readouterr().err


def test_load_stage_v_configs_reads_stage_v_stage_d_and_pins(tmp_path):
    """As stage_v_report.load_configs: (stage_v.toml, stage_d.toml, pins.toml, their sha256)."""
    raws = [(ROOT / "configs" / n).read_bytes() for n in ("stage_v.toml", "stage_d.toml", "pins.toml")]
    got = pg.load_stage_v_configs(ROOT)
    assert got[3:] == tuple(sha(r) for r in raws) and got[:3] == svr.load_configs()[:3]
    assert pg.load_stage_v_configs(tpg.cli_root(tmp_path, None)) is None


def test_freeze_commit_problems_for_stage_v_with_a_real_repository(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    tpg.git(root, "init", "-q")
    files = {pg.frozen_file(m)[0]: pg.frozen_file(m)[1].read_bytes() for m in pg.STAGE_V_MODULES}
    before = tpg.commit_files(root, {STAGE_V_TOML: UNFROZEN_V, **files})
    freeze = tpg.commit_files(root, {STAGE_V_TOML: FROZEN_V})
    deployed = tpg.commit_files(root, {"x.txt": b"after the freeze\n"})
    kw = {"git": pg.git_runner(root), "freeze_file": STAGE_V_TOML, "freeze_key": "stage_v",
          "modules": pg.STAGE_V_MODULES}
    assert pg.freeze_commit_problems(freeze, deployed_commit=deployed, capacity_sha256=SV_SHA, **kw) == []
    assert "[freeze] stage_v = true: not the freeze commit" in pg.freeze_commit_problems(
        before, deployed_commit=deployed, capacity_sha256=SV_SHA, **kw)[0]
    assert "DEPLOYED_COMMIT" in pg.freeze_commit_problems(freeze, deployed_commit=before, capacity_sha256=SV_SHA,
                                                          **kw)[0]
    edited = tpg.commit_files(root, {"benchmarks/stage_v_report.py": files["benchmarks/stage_v_report.py"] + b"#\n"})
    msg = "; ".join(pg.freeze_commit_problems(freeze, deployed_commit=edited, capacity_sha256=SV_SHA, **kw))
    assert "benchmarks/stage_v_report.py at the session's DEPLOYED_COMMIT" in msg and "044 凍結" in msg


def test_the_help_names_every_stage():
    assert pg._parser().description == "verified capacity profiles from a frozen Stage C, D or V analysis (M2; 044 §12)"


def test_module_entry_point(tmp_path):
    env = {**os.environ, "PYTHONPATH": f"{SERVICE}{os.pathsep}{ROOT / 'benchmarks'}"}
    (tmp_path / "d").mkdir()
    r = subprocess.run([sys.executable, "-m", "capserve.profile_gen", "--validation-session", "s", "--freeze-commit",
                        FREEZE_V, "--out-dir", str(tmp_path / "d")], capture_output=True, text=True, env=env,
                       cwd=ROOT, timeout=60)
    assert r.returncode == 2 and "refused" in r.stderr and list((tmp_path / "d").iterdir()) == []


# ------------------------------------------------------------------ regeneration, request path (044) --

def test_a_section_12_profile_regenerates_through_the_committed_profiles_dispatch(sessions):
    """044 實作補記一第 9 項: a committed §12 profile is trusted as §8 / §9 are, by the Mac regeneration test, which
    dispatches on prereg_section to the validation-session path and picks the profile by its id."""
    s = sessions("cls")
    doc = one(s)
    committed = copy.deepcopy(doc)
    committed["provenance"]["generator_commit"] = "5e" * 20
    committed["content_sha256"] = cp.content_sha256(committed)
    kw = {"root": s["root"], "git": VGit(), "configs": s["configs"], "stage_v_configs": s["stage_v_configs"]}
    assert tpg._regenerable(tpg.regenerate(committed, **kw)) == tpg._regenerable(committed)
    with pytest.raises(pg.GenerationError, match="not 'verified-formal-t-v-cls-x'"):
        tpg.regenerate({**committed, "profile_id": "verified-formal-t-v-cls-x"}, **kw)


def test_the_generator_stays_outside_the_request_path():
    """044: profile_gen is not in the gateway's import closure, so writing it never moves gateway_path_sha256."""
    assert "service/capserve/profile_gen.py" not in gpath.gateway_path_files(ROOT)
    assert gpath.gateway_path_sha256(ROOT) == sv.gateway_path_sha256(ROOT)
