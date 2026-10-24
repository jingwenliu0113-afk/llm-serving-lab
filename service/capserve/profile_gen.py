"""python -m capserve.profile_gen --session results/capacity/<formal session> --freeze-commit <sha> --out <profile.json>
python -m capserve.profile_gen --session results/stage_d/<formal D-A session> --freeze-commit <sha> --out-dir <dir>
python -m capserve.profile_gen --validation-session results/stage_v/<formal-id> --freeze-commit <sha> --out-dir <dir>

A verified profile is the only kind that may become a product limit (027 §4), and only this generator writes one,
from a formal, env-valid Stage C session analysed by the frozen Stage C modules (decision 029 "產生器", M2; decision
030 8.7). It has no override options (021 R1): its inputs are the session, the commit that froze Stage C (checked
against git, never trusted) and the output path. It refuses (exit 2) and writes nothing unless:
- configs/capacity.toml has [freeze] capacity = true (a missing file or key is false; 030 implementation note) and
  capacity_plan.freeze_evidence_problems is empty (the passed FS-order / mini smokes and C0's result log);
- --freeze-commit is the commit that set [freeze] capacity = true: 40 lowercase hex, frozen in that commit, not in its
  first parent nor in any earlier commit of its capacity.toml history (an unfreeze / change / re-freeze is not the
  preregistered freeze, 030 8.9), and an ancestor of the session's DEPLOYED_COMMIT (session.json does not record the
  freeze commit, so it is passed explicitly and verified here). The frozen bytes are the ones that ran: capacity.toml
  at the freeze commit is the capacity.toml the session recorded and this generator reads (its sha256), and every
  Stage C module (capacity_plan, capacity_analysis, run_capacity, capacity_report, gpu_idle) at the freeze commit is
  the one at the session's DEPLOYED_COMMIT and the file this generator imports, which re-runs the analysis (029);
- the session is results/capacity/<session_id> with a formal- id, and capacity_report.refusal_problems is empty
  (formal, mode formal, stage C1, env-valid, not aborted, the GPU lock, one DEPLOYED_COMMIT, every sha256, the plan
  rebuilt from its uniform session results/core_b/<derived_from>, the Core B argv / env, the 031 spill records);
- capacity_report.report, re-run read-only on the session (with the Advanced references when analysis.json says it
  used them), reproduces <session>/analysis.json byte for byte (021 R3 style; json.dumps(indent=1) + newline, the
  report's own serialization), and its SLO is capacity.toml's;
- the session's server (its recorded argv / env, capserve.profile.server_fingerprint) is the product server
  (profile.stage_b_fingerprint), which capserve.deploy checks again at every boot;
- the generator commit is HEAD of a clean tree (no uncommitted change to tracked files), and --out does not exist
  and lies outside results/ (research data is never written by the product).
Each Stage C workload becomes one class (030 8.7): U2K / U8K accept prompts of L - 128 tokens (1920 / 8064), the mixed
M2K / M8K accept the 007 mix's 32,640-token prompts; max_output_tokens 128.
- C_PROD EXACT / LOWER / CENSORED -> "verified", max_concurrency = max_queue (029 product policy) = C_PROD, bound
  "exact" / "lower" / "censored";
- C_PROD INFEASIBLE -> "verified" 0 with bound "exact": refuse this class (the gateway answers 503
  class_not_serviceable);
- C_PROD UNSTABLE / NT -> "unavailable" with the reason, never a number and never a fallback (no C_SLO, Ĉ, C_KV or
  provisional value).
Every class carries its evidence (scope, C_PROD, C_SLO, C_KV, C_cap, flags, binding terms, margins at C_PROD, the
start-up burst at C_PROD); the profile carries the SLO terms verbatim, the server (vLLM, model, fingerprint, KV
geometry) and the provenance (session, analysis path and sha256, freeze / generator / deployed commits, capacity /
pins sha256, GPU). The output is deterministic (no timestamps): regenerating a committed verified profile gives the
same document except generator_commit (029's Mac regeneration test). Research code never imports this module.

A Stage D session (decision 043; PREREGISTRATION §9) is dispatched on its path: results/stage_d/<session> takes
--out-dir, results/capacity/ (and anything else) --out; exactly one of the two, the other one is refused (exit 2). It
writes every server's profile of the session at <out-dir>/<profile_id>.json, all or none (each document passes the
loader and no target exists before the first write; a failed write removes what was written), and refuses unless:
- configs/stage_d.toml has [freeze] stage_d = true and stage_d_plan.freeze_evidence_problems is empty, and
  capacity.toml is frozen too (Stage D runs S2 and the frozen Stage C modules);
- --freeze-commit is the commit that first set [freeze] stage_d = true (the same rules as above, on stage_d.toml), an
  ancestor of DEPLOYED_COMMIT, its stage_d.toml the one the session recorded and this generator reads, and every
  stage_d_plan.HARNESS_FILES file and Stage C module there is the one at DEPLOYED_COMMIT and the one this generator
  runs (stage_d_datasets, never imported, is compared with this checkout's file);
- the session is results/stage_d/<session_id> with a formal- id, a D-A session (plan and analysis kind "closed": D-B
  and D-C never give a profile, 039 / 042; the aborted D-A1 has no confirm plan nor analysis.json), and
  stage_d_report.refusal_problems is empty; stage_d_report.report, re-run read-only (with the Advanced references when
  analysis.json used them), reproduces analysis.json byte for byte, and its SLO is stage_d.toml's and capacity.toml's;
- every boot's recorded server is its workload's (server_groups: B0 or T, each boot's own argv / env, never only the
  first boot's), one fingerprint per server, the product B0 (stage_b_fingerprint) or T (t_fingerprint) server, and
  analysis.json records one server argv per server;
- the generator commit is HEAD of a clean tree, and --out-dir is an existing directory outside results/.
One profile per server (verified-<session_id>-<b0|t>, the T one with pins [advanced]'s long_prefill_token_threshold),
one class per D-A workload of that server (the U2K sentinel runs on B0 but is never a class), mapped as 030 8.7 with a
stage_d source, named and limited by capserve.profile.stage_d_class (max_output_tokens: the measured output length).
Its evidence is Stage C's (margins: stage_d_margins, 1 - worst R at C_PROD, capacity_analysis's own formula) plus
043 §3's, all copied from analysis.json: server, output_tokens, descriptive c_full / c_late, at_c_prod, hypotheses,
not_covered. The provenance is §9 (stage_d_sha256; capacity_sha256 of this checkout).

A Stage V session (decision 044 產品對應; PREREGISTRATION §12) is the --validation-session, never --session, and is
written into --out-dir: at most one profile, verified-<session_id>-b0.json, and none when no unit that writes one holds
(exit 0: a result, not a refusal; the summary says so). Only analysis.json's units decide what is written, and only a
unit that holds with every point PASS (holding_units): VG-32K and VG-M8K-T write nothing (candidates for the pinned
profile only); PACE-2K / PACE-8K a copy of c1b's uniform-2k / uniform-8k (COPY_BASE, the class verbatim) with the unit's
pacing, its queue settings and its validation; W1 the admission section (weighted-v1's parameters over the copies, the
queue settings, measured_mixes, the frozen gateway_path_sha256, V4's validation), its member copies plain unless their
PACE unit holds; V1-2K / V1-8K V1's validated class (uniform-2k-o1-512 / uniform-8k-o1-512: d-a2-b0's O512 class
(V1_BASE) at V1_C, bound validated, source stage_v:<unit>, its evidence with not_covered = v1_not_covered, the output
distribution, the L point's five datasets and via_gateway; V1's validation). A validation is the unit's {study, unit,
points, measured_classes, worst_r, n}. The provenance is §12's (stage_v_sha256, gateway_path_sha256, capserve_sha256,
base_profiles, besides §9's). It refuses (exit 2) and writes nothing unless:
- configs/stage_v.toml has [freeze] stage_v = true and stage_v_plan.freeze_evidence_problems is empty, and stage_d.toml
  and capacity.toml are frozen (the loader's §12 rules);
- --freeze-commit is the commit that first set [freeze] stage_v = true (the rules above, on stage_v.toml), an ancestor
  of DEPLOYED_COMMIT, its stage_v.toml the one the session recorded and this generator reads, and every
  stage_v_plan.HARNESS_FILES file, Stage D frozen file and Stage C module there is the one at DEPLOYED_COMMIT and the
  one this generator runs (a session under the stop rule's new freeze is therefore refused until that freeze's
  decision amends this rule: 044 實作補記十第 2 項);
- the request path at DEPLOYED_COMMIT (its tracked service/capserve files, hashed by stage_v_plan's frozen copy of
  capserve.gateway_path's algorithm: deployed_gateway_path_sha256; a file it cannot parse refuses too) is
  stage_v.toml's frozen gateway_path_sha256;
- each base profile (COPY_BASE, V1_BASE) loads as a verified profile of this checkout and is byte for byte its file at
  the freeze commit;
- the session is results/stage_v/<session_id> with a formal- id (never a mini), stage_v_report.refusal_problems is
  empty, and stage_v_report.report, re-run read-only (stage_d.toml's Core B uniform session, the Advanced references
  when analysis.json used them), reproduces analysis.json byte for byte, whose SLO is stage_v.toml's;
- every boot's recorded server is its point's server_base's (server_groups, B0 the product fingerprint);
- no other results/stage_v/formal-* session of its plan session (V-ADM / V-CLS) that started before it under the same
  freeze (the same recorded stage_v_sha256, harness_sha256 and gateway_path_sha256) is valid, i.e. accepted by
  stage_v_report (044 停止規則: 不重跑有效的 session 來換 PASS; rerun_problems), checked fail closed (044 實作補記十第
  10 項): such an earlier session lets the rerun through only when its session.json records it aborted or env_valid
  false, or stage_v_report.refusal_problems is empty and report() raises ReportError; a directory that is not its
  session_id, files that are not what its session.json records, refusal problems or any other error refuse, as do a
  formal-* directory of its plan, or of a plan that cannot be read, without a readable session.json and started_at,
  and a formal session of its previous_formal chain that is not here (fetch every formal Stage V session first). The
  2-attempt cap stays with the operator (044 實作補記四第 4 項, 實作補記十);
- the generator commit is HEAD of a clean tree, --out-dir an existing directory outside results/, the target does not
  exist, and the document passes the loader before anything is written.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from fractions import Fraction
from pathlib import Path
from typing import Callable

import capacity_analysis as ca
import capacity_plan as cplan
import capacity_report as cr
import core_b_plan as bp
import gpu_idle as gi
import run_capacity as rc
import run_stage_d as rd
import run_stage_v as rv
import stage_d_analysis as sda
import stage_d_plan as sdplan
import stage_d_report as sr
import stage_v_analysis as sva
import stage_v_datasets as svd
import stage_v_plan as svplan
import stage_v_report as svr
from capserve import profile as cp

ROOT = Path(__file__).resolve().parents[2]
CAPACITY_TOML = ROOT / "configs" / "capacity.toml"
FREEZE_FILE = "configs/capacity.toml"
STAGE_D_TOML = ROOT / "configs" / "stage_d.toml"
STAGE_D_FREEZE_FILE = "configs/stage_d.toml"
STAGE_D_RESULTS = sdplan.RESULTS          # ("results", "stage_d"): a Stage D session's place (043 §1.1 dispatch)
STAGE_V_TOML = ROOT / "configs" / "stage_v.toml"
STAGE_V_FREEZE_FILE = "configs/stage_v.toml"
STAGE_V_RESULTS = svplan.RESULTS          # ("results", "stage_v"): a Stage V session's place (044)
D_A_KIND = sdplan.KINDS["D-A1"]           # "closed", D-A2's too; D-B is open, D-C verify (no profile, 042)
ADVANCED_SESSION = "formal-20261011-adv"   # the CROSS_SESSION_SHIFT references capacity_report may have used (8.4)
# frozen with capacity.toml (030 凍結; tests/test_capacity_isolation CAPACITY_MODULES): their bytes at the freeze commit
# are the ones the session ran and the ones that re-run its analysis here
STAGE_C_MODULES = (cplan, ca, rc, cr, gi)
# frozen with stage_d.toml (039 凍結): stage_d_plan.HARNESS_FILES, the modules stage_d_report re-runs the analysis with
# (stage_d_datasets is never imported: its path, compared with this checkout's file), and the Stage C modules Stage D
# runs (043 §1.2)
_STAGE_D_IMPORTED = {m.__name__: m for m in (sdplan, sda, rd, sr)}
STAGE_D_MODULES = tuple(_STAGE_D_IMPORTED.get(Path(rel).stem, rel) for rel in sdplan.HARNESS_FILES) + STAGE_C_MODULES
# frozen with stage_v.toml (044 凍結): stage_v_plan.HARNESS_FILES, the modules stage_v_report re-runs the analysis with,
# then the Stage D and Stage C modules the Stage V harness runs
_STAGE_V_IMPORTED = {m.__name__: m for m in (svplan, svd, sva, rv, svr)}
STAGE_V_MODULES = tuple(_STAGE_V_IMPORTED[Path(rel).stem] for rel in svplan.HARNESS_FILES) + STAGE_D_MODULES
# 044 產品對應: the committed verified profiles a §12 class copies, read at their bytes of the Stage V freeze commit
COPY_BASE = "service/profiles/verified-formal-20261017-c1b.json"      # PACE-2K / PACE-8K / W1: c1b's uniform-2k / -8k
V1_BASE = "service/profiles/verified-formal-20261023-d-a2-b0.json"    # V1-2K / V1-8K: D-A2's O512 classes (B0)
NO_FILE_UNITS = ("VG-32K", "VG-M8K-T")    # 044: candidates for the pinned profile only, never a new file
STUDY_COPIES = {"v-uniform-2k": "U2K", "v-uniform-8k": "U8K"}   # the study classes W1 measured: c1b's, by workload
# what each freeze guards, in the refusals (the Stage C wording unchanged); refreeze: what a refusal of a later freeze
# adds (Stage V: the stop rule's new freeze, 044 實作補記十第 2 項)
_FREEZES = {"capacity": {"stage": "Stage C", "rule": "030 凍結", "data": "C1 data", "change": "030 8.9",
                         "refreeze": ""},
            "stage_d": {"stage": "Stage D", "rule": "039 凍結", "data": "Stage D data", "change": "039 凍結, 030 8.9",
                        "refreeze": ""},
            "stage_v": {"stage": "Stage V", "rule": "044 凍結", "data": "Stage V data", "change": "044 凍結, 030 8.9",
                        "refreeze": "; if this is the stop rule's new freeze, a session under it gives a profile only "
                                    "once that freeze's decision amends this first-freeze rule (044 實作補記十第 2 項)"}}
_HEX40 = re.compile(r"[0-9a-f]{40}")
NOT_FROZEN = ("[freeze] capacity is not true in {path} (a missing file or key is false). PREREGISTRATION §8 is not "
              "frozen, so no verified capacity exists yet (027 §4, 029); develop against a provisional profile with "
              "--allow-provisional.")
NOT_FROZEN_D = ("[freeze] stage_d is not true in {path} (a missing file or key is false). PREREGISTRATION §9 is not "
                "frozen, so no Stage D capacity exists (039, 043).")
NOT_FROZEN_V = ("[freeze] stage_v is not true in {path} (a missing file or key is false). PREREGISTRATION §12 is not "
                "frozen, so no Stage V unit is validated (044).")
NO_PROFILE = ("no profile: no unit that writes one holds (044 產品對應: VG-32K / VG-M8K-T only make candidates for the "
              "pinned profile; every other unit needs each of its points PASS)")
# 044 停止規則: what makes an earlier session "the same freeze" (rerun_problems)
STAGE_V_FREEZE_SHAS = ("stage_v_sha256", "harness_sha256", "gateway_path_sha256")
RERUN_RULE = "044 停止規則: 不重跑有效的 session 來換 PASS"
# How the frozen stage_v_report.report begins the ReportError of its own validity verdict (after the analysis: a replay
# mismatch, a failed sentinel gate); its other one, "the analysis failed: ...", is no verdict (044 實作補記十第 11 項).
REPORT_VERDICTS = ("replay mismatch: ", "the sentinel gate failed (drift, 030 8.6): ")
NOT_D_A = ("039 / 042: D-B and D-C produce no profile (their facts are service/README.md's scope, 043 §5); the {what} "
           "kind is {kind!r}, not D-A's {want!r}")

Git = Callable[..., "bytes | None"]


class GenerationError(RuntimeError):
    def __init__(self, problems):
        self.problems = list(dict.fromkeys(problems))
        super().__init__("; ".join(self.problems))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capserve.profile_gen",
                                     description="verified capacity profiles from a frozen Stage C, D or V "
                                                 "analysis (M2; 044 §12)")
    session = parser.add_mutually_exclusive_group(required=True)
    session.add_argument("--session", help="results/capacity/<formal session> or results/stage_d/<formal D-A session> "
                         "(with its analysis.json)")
    session.add_argument("--validation-session", help="Stage V (044): results/stage_v/<formal session> (with its "
                         "analysis.json)")
    parser.add_argument("--freeze-commit", required=True,
                        help="the commit that set configs/capacity.toml [freeze] capacity = true, or (Stage D) "
                             "configs/stage_d.toml [freeze] stage_d = true, or (Stage V) configs/stage_v.toml [freeze] "
                             "stage_v = true (40 hex; verified)")
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--out", help="Stage C: the verified profile to write (never overwritten)")
    out.add_argument("--out-dir", help="Stage D / V: the directory of the session's profiles, each <profile_id>.json "
                                       "(never overwritten; all or none)")
    return parser


def option_names() -> set:
    return {s for a in _parser()._actions for s in a.option_strings if s not in ("-h", "--help")}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- inputs

def git_runner(root) -> Git:
    """git -C root <args>: stdout bytes, or None when git fails (unknown commit, no repository, no git)."""
    def git(*args: str):
        try:
            r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None
    return git


def frozen(cfg: dict, key: str = "capacity") -> bool:
    """[freeze] <key> is exactly true (capacity: Stage C's capacity.toml, stage_d: Stage D's stage_d.toml)."""
    return (cfg.get("freeze") or {}).get(key) is True


def load_configs(root) -> tuple | None:
    """(capacity.toml, pins.toml, their sha256) under root, as capacity_report.load_configs reads the repo's; None
    without a capacity.toml (a missing file is not frozen). OSError / TOMLDecodeError propagate."""
    path = Path(root) / FREEZE_FILE
    if not path.exists():
        return None
    cap_raw, pins_raw = path.read_bytes(), (Path(root) / "configs" / "pins.toml").read_bytes()
    return tomllib.loads(cap_raw.decode()), tomllib.loads(pins_raw.decode()), _sha256(cap_raw), _sha256(pins_raw)


def _configs(root) -> tuple:
    path = Path(root) / FREEZE_FILE
    try:
        configs = load_configs(root)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise GenerationError([f"cannot read {path} or configs/pins.toml: {exc}"]) from exc
    if configs is None or not frozen(configs[0]):
        raise GenerationError([NOT_FROZEN.format(path=path)])
    return configs


def load_stage_d_configs(root) -> tuple | None:
    """(stage_d.toml, pins.toml, their sha256) under root, as stage_d_report.load_configs reads the repo's; None
    without a stage_d.toml (a missing file is not frozen). Stage C's four-tuple (load_configs) is unchanged (043 §1.2).
    OSError / TOMLDecodeError propagate."""
    path = Path(root) / STAGE_D_FREEZE_FILE
    if not path.exists():
        return None
    sd_raw, pins_raw = path.read_bytes(), (Path(root) / "configs" / "pins.toml").read_bytes()
    return tomllib.loads(sd_raw.decode()), tomllib.loads(pins_raw.decode()), _sha256(sd_raw), _sha256(pins_raw)


def _stage_d_configs(root) -> tuple:
    path = Path(root) / STAGE_D_FREEZE_FILE
    try:
        configs = load_stage_d_configs(root)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise GenerationError([f"cannot read {path} or configs/pins.toml: {exc}"]) from exc
    if configs is None or not frozen(configs[0], "stage_d"):
        raise GenerationError([NOT_FROZEN_D.format(path=path)])
    return configs


def load_stage_v_configs(root) -> tuple | None:
    """(stage_v.toml, stage_d.toml, pins.toml, their sha256) under root, as stage_v_report.load_configs reads the
    repo's; None without a stage_v.toml (a missing file is not frozen). OSError / TOMLDecodeError propagate."""
    path = Path(root) / STAGE_V_FREEZE_FILE
    if not path.exists():
        return None
    raws = (path.read_bytes(), (Path(root) / STAGE_D_FREEZE_FILE).read_bytes(),
            (Path(root) / "configs" / "pins.toml").read_bytes())
    return (*(tomllib.loads(raw.decode()) for raw in raws), *(_sha256(raw) for raw in raws))


def _stage_v_configs(root) -> tuple:
    path = Path(root) / STAGE_V_FREEZE_FILE
    try:
        configs = load_stage_v_configs(root)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise GenerationError([f"cannot read {path}, configs/stage_d.toml or configs/pins.toml: {exc}"]) from exc
    if configs is None or not frozen(configs[0], "stage_v"):
        raise GenerationError([NOT_FROZEN_V.format(path=path)])
    return configs


def _frozen_at(git: Git, rev: str, *, freeze_file: str = FREEZE_FILE, freeze_key: str = "capacity"):
    """Whether freeze_file at rev has [freeze] <freeze_key> = true; None when git cannot show a readable one."""
    data = git("show", f"{rev}:{freeze_file}")
    if data is None:
        return None
    try:
        return frozen(tomllib.loads(data.decode()), freeze_key)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None


def frozen_file(module) -> tuple:
    """(its path in the repository, the local file whose bytes must be the frozen ones) of a frozen module: an imported
    module's own file, or, given as a repository path (never imported), this checkout's file."""
    if isinstance(module, str):
        return module, ROOT / module
    return f"benchmarks/{module.__name__}.py", Path(module.__file__)


def freeze_commit_problems(commit, *, deployed_commit, capacity_sha256: str, git: Git, freeze_file: str = FREEZE_FILE,
                           freeze_key: str = "capacity", modules=STAGE_C_MODULES) -> list:
    """[] when commit is the full sha of the commit that first set [freeze] <freeze_key> = true in freeze_file (frozen
    there, never before in its history), an ancestor of (or equal to) the session's DEPLOYED_COMMIT, and its frozen
    bytes are the ones that ran: freeze_file there has sha256 capacity_sha256 (what the session recorded and this
    generator reads), and every one of modules there is the one at DEPLOYED_COMMIT and the file this generator runs
    (frozen_file). The defaults are Stage C's (capacity.toml, its modules); Stage D passes stage_d.toml, "stage_d" and
    STAGE_D_MODULES (043 §1.2)."""
    if not (isinstance(commit, str) and _HEX40.fullmatch(commit)):
        return [f"--freeze-commit {commit!r} is not a full commit (40 lowercase hex)"]
    where = {"freeze_file": freeze_file, "freeze_key": freeze_key}
    rule = _FREEZES[freeze_key]
    problems, at = [], _frozen_at(git, commit, **where)
    if at is None:
        problems.append(f"{commit} has no readable {freeze_file} (not a commit of this repository?)")
    elif not at:
        problems.append(f"{freeze_file} at {commit} has no [freeze] {freeze_key} = true: not the freeze commit")
    elif _frozen_at(git, f"{commit}^", **where):
        problems.append(f"{freeze_file} was already frozen at {commit}^: {commit} is not the commit that set "
                        f"[freeze] {freeze_key} = true{rule['refreeze']}")
    else:
        problems += _earlier_freeze_problems(commit, git, **where)
    if not isinstance(deployed_commit, str) or git("merge-base", "--is-ancestor", commit, deployed_commit) is None:
        problems.append(f"{commit} is not an ancestor of the session's DEPLOYED_COMMIT {deployed_commit!r}: the "
                        f"session did not run the frozen code ({rule['rule']}: {rule['data']} only after the freeze)")
    if at is None:
        return problems
    return problems + _frozen_bytes_problems(commit, deployed_commit=deployed_commit,
                                             capacity_sha256=capacity_sha256, git=git, modules=modules, **where)


def _earlier_freeze_problems(commit: str, git: Git, *, freeze_file: str = FREEZE_FILE,
                             freeze_key: str = "capacity") -> list:
    """The freeze is the first commit that set [freeze] <freeze_key> = true: a re-freeze after an unfreeze is a rule
    change after the freeze (030 8.9: its own decision and a new session), never the preregistered freeze."""
    history = git("rev-list", "--full-history", commit, "--", freeze_file)
    if history is None:
        return [f"cannot list the history of {freeze_file} up to {commit} (git rev-list failed): cannot tell whether "
                "it is the first freeze"]
    earlier = [c for c in history.decode().split()
               if c != commit and _frozen_at(git, c, freeze_file=freeze_file, freeze_key=freeze_key)]
    if earlier:
        rule = _FREEZES[freeze_key]
        return [f"{freeze_file} was already frozen at {earlier[-1]}, before {commit}: an unfreeze and re-freeze is "
                f"not the preregistered freeze ({rule['change']}: a rule change after the freeze needs its own "
                f"decision{rule['refreeze']})"]
    return []


def _frozen_bytes_problems(commit: str, *, deployed_commit, capacity_sha256: str, git: Git,
                           freeze_file: str = FREEZE_FILE, freeze_key: str = "capacity",
                           modules=STAGE_C_MODULES) -> list:
    problems, rule = [], _FREEZES[freeze_key]
    toml = git("show", f"{commit}:{freeze_file}")
    if toml is not None and _sha256(toml) != capacity_sha256:
        problems.append(f"{freeze_file} at the freeze commit {commit} (sha256 {_sha256(toml)}) is not the one the "
                        f"session ran and this generator reads ({capacity_sha256}): it changed after the freeze"
                        f"{rule['refreeze']}")
    for module in modules:
        path, local = frozen_file(module)
        frozen_bytes = git("show", f"{commit}:{path}")
        if frozen_bytes is None:
            problems.append(f"no {path} at the freeze commit {commit}")
            continue
        if isinstance(deployed_commit, str):
            ran = git("show", f"{deployed_commit}:{path}")
            if ran is None or _sha256(ran) != _sha256(frozen_bytes):
                problems.append(f"{path} at the session's DEPLOYED_COMMIT {deployed_commit} is not the frozen one at "
                                f"{commit}: the session did not run the frozen {rule['stage']} code ({rule['rule']})")
        try:
            mine = local.read_bytes()
        except OSError as exc:
            mine = None
            problems.append(f"cannot read {local} ({exc})")
        if mine is None or _sha256(mine) == _sha256(frozen_bytes):
            continue
        if isinstance(module, str):
            problems.append(f"{path} in this checkout ({local}) is not the frozen one at {commit}: the harness changed "
                            f"after the freeze ({rule['rule']})")
        else:
            problems.append(f"{path} imported by this generator ({local}) is not the frozen one at {commit}: the "
                            f"analysis would not be re-run by the frozen {rule['stage']} modules (029)")
    return problems


def generator_commit(git: Git) -> tuple:
    """(HEAD, problems): the generator commit must name the code that generates, so the tracked files are clean."""
    head = git("rev-parse", "HEAD")
    sha = head.decode().strip() if head else ""
    problems = [] if _HEX40.fullmatch(sha) else ["cannot name the generator commit (git rev-parse HEAD failed)"]
    status = git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        problems.append("git status failed: cannot tell whether the generator's code is committed")
    elif status.strip():
        problems.append(f"uncommitted changes in tracked files: generator_commit {sha[:12] or '?'} would not be the "
                        "code that generates the profile; commit first")
    return sha, problems


def deployed_gateway_path_sha256(commit, *, git: Git) -> tuple:
    """(gateway_path_sha256, problems) of the request path at commit (044 產品對應: the session's DEPLOYED_COMMIT): its
    tracked service/capserve files, written to a scratch directory and hashed by stage_v_plan's frozen research copy of
    the algorithm (the AST closure of __main__, as capserve.gateway_path, which /readyz discloses and may change after
    V-CLS: 044 實作補記十第 4 項), never this checkout's files. A file the AST cannot parse is a problem, not a crash
    (實作補記十第 3 項)."""
    package = "/".join(svplan.PACKAGE_DIR)
    listing = git("ls-tree", "-r", "--name-only", commit, "--", package) if isinstance(commit, str) else None
    names = [] if listing is None else [n for n in listing.decode().splitlines() if n]
    if not names:
        return None, [f"cannot list {package} at the session's DEPLOYED_COMMIT {commit!r}: its request path is unknown"]
    with tempfile.TemporaryDirectory() as tmp:
        for rel in names:
            data = git("show", f"{commit}:{rel}")
            if data is None:
                return None, [f"cannot read {rel} at the session's DEPLOYED_COMMIT {commit}: its request path is "
                              "unknown"]
            path = Path(tmp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        try:
            return svplan.gateway_path_sha256(tmp), []
        except (SyntaxError, ValueError, OSError, UnicodeDecodeError) as exc:
            return None, [f"cannot hash the request path at DEPLOYED_COMMIT {commit}: {type(exc).__name__}: {exc}"]


def base_profiles(root, freeze_commit, *, git: Git, pins: dict, frozen_cfgs: cp.FrozenConfigs) -> tuple:
    """({path: document}, problems) of COPY_BASE and V1_BASE (044 產品對應: a copy's base is a committed verified
    profile): each loads as a verified profile of this checkout and is byte for byte its file at the Stage V freeze
    commit (the bytes the study profile's v- classes copy, which the mini and the sessions then served)."""
    docs, problems = {}, []
    for rel in (COPY_BASE, V1_BASE):
        try:
            raw = (Path(root) / rel).read_bytes()
        except OSError as exc:
            problems.append(f"cannot read the base profile {rel} ({exc})")
            continue
        committed = git("show", f"{freeze_commit}:{rel}") if isinstance(freeze_commit, str) and _HEX40.fullmatch(
            freeze_commit) else None
        if committed is None:
            problems.append(f"no {rel} at the Stage V freeze commit {freeze_commit!r}: a copy's base is a committed "
                            "verified profile (044 產品對應)")
        elif committed != raw:
            problems.append(f"the base profile {rel} is not its bytes at the Stage V freeze commit {freeze_commit} "
                            f"(sha256 {_sha256(raw)}, there {_sha256(committed)}): the classes a copy copies changed "
                            "after the freeze (044 產品對應)")
        try:
            cp.parse_profile(raw, source=rel, allow_provisional=False, pins=pins, frozen=frozen_cfgs)
        except cp.ProfileError as exc:
            problems.append(f"the base profile does not load as verified: {exc}")
            continue
        docs[rel] = json.loads(raw)
    return docs, problems


def _out_problems(out: Path, root) -> list:
    out = Path(out)
    problems = []
    if out.exists():
        problems.append(f"{out} already exists: a profile is never overwritten (remove it first)")
    if out.resolve().is_relative_to(Path(root).resolve() / "results"):
        problems.append(f"{out} is under results/: the product never writes research data (027 §3)")
    return problems


def _out_dir_problems(out_dir: Path, root) -> list:
    out_dir = Path(out_dir)
    problems = [] if out_dir.is_dir() else [f"--out-dir {out_dir} is not an existing directory"]
    if out_dir.resolve().is_relative_to(Path(root).resolve() / "results"):
        problems.append(f"{out_dir} is under results/: the product never writes research data (027 §3)")
    return problems


def write_all(files: dict) -> None:
    """Write every {path: bytes} or none (043 §1.1): each file is created exclusively, so nothing is ever overwritten
    (not even one created after the checks), and a failure removes every file this call created."""
    written = []
    try:
        for path, data in files.items():
            with open(path, "xb") as f:
                written.append(Path(path))
                f.write(data)
    except OSError as exc:
        for done in written:
            done.unlink(missing_ok=True)
        raise GenerationError([f"cannot write {path} ({exc}); removed the {len(written)} profile(s) this run had "
                               "created: all or none (043 §1.1)"]) from exc


def session_fingerprint(session: dict, pins: dict) -> tuple:
    """(fingerprint, problems): capserve.profile.server_fingerprint of the first boot's recorded server argv and env,
    the env projected onto core_b_plan.recorded_env's keys (a session also records OTEL_* and the engine variable
    names, which capacity_report requires to be Core B's); it must be the product server's
    (profile.stage_b_fingerprint), since capacity_report checks every boot ran that one argv / env."""
    boots = session.get("boots") or []
    rec = (session.get("runs") or {}).get(boots[0].get("tag")) if boots else None
    if not (isinstance(rec, dict) and isinstance(rec.get("argv"), list) and isinstance(rec.get("env"), dict)):
        return None, ["the session records no server argv / env for its first boot"]
    keys = list(bp.recorded_env({}))
    missing = [k for k in keys if k not in rec["env"]]
    if missing:
        return None, [f"the session's recorded server env lacks {missing} (core_b_plan.recorded_env)"]
    fp = cp.server_fingerprint(rec["argv"], {k: rec["env"][k] for k in keys})
    want = cp.stage_b_fingerprint(pins)
    if fp != want:
        return fp, [f"the session's server fingerprint {fp} is not the product server's {want} (the Stage B argv on "
                    "the product port; capserve.deploy would refuse this profile at every boot)"]
    return fp, []


def server_groups(session: dict, pins: dict) -> tuple:
    """({server: {"fingerprint", "boots", "workloads"}}, problems) of a Stage D session (043 §1.4). Every boot belongs
    to its workload's server (stage_d_plan.WORKLOADS[w]["server"], B0 or T; the U2K sentinel's is B0) and is
    fingerprinted from its own recorded argv and env (the env projected onto core_b_plan.recorded_env's keys, as
    session_fingerprint does); a server is grouped only when all its boots have one fingerprint, the product's
    (capserve.profile.server_variant_fingerprint: B0 stage_b_fingerprint, T t_fingerprint). Stage C's
    session_fingerprint reads the first boot only, which in D-A1b is B0's sentinel: it would stamp T's classes with
    B0's fingerprint, and pass."""
    runs, keys, problems = session.get("runs") or {}, list(bp.recorded_env({})), []
    boots = session.get("boots") or []
    if not boots:
        problems.append("the session records no boots: no server to fingerprint")
    seen = {}                                # server -> fingerprint -> its boots, in boot order
    for b in boots:
        tag, w = b.get("tag"), b.get("workload")
        if w not in sdplan.WORKLOADS:
            problems.append(f"boot {tag}: workload {w!r} is not a Stage D workload (stage_d_plan.WORKLOADS)")
            continue
        rec = runs.get(tag)
        if not (isinstance(rec, dict) and isinstance(rec.get("argv"), list) and isinstance(rec.get("env"), dict)):
            problems.append(f"boot {tag}: the session records no server argv / env")
            continue
        missing = [k for k in keys if k not in rec["env"]]
        if missing:
            problems.append(f"boot {tag}: the recorded server env lacks {missing} (core_b_plan.recorded_env)")
            continue
        fp = cp.server_fingerprint(rec["argv"], {k: rec["env"][k] for k in keys})
        seen.setdefault(sdplan.WORKLOADS[w]["server"], {}).setdefault(fp, []).append(b)
    groups = {}
    for server in cp.SERVERS:
        fps, want = seen.get(server) or {}, cp.server_variant_fingerprint(server, pins)
        if len(fps) > 1:
            ran = {fp[:12]: [b.get("tag") for b in bs] for fp, bs in fps.items()}
            problems.append(f"server {server}: its boots ran {len(fps)} different servers {ran}: a profile states one "
                            "server's fingerprint, and every boot of its classes must be that server (043 §1.4)")
        elif fps and want not in fps:
            problems.append(f"server {server}: the fingerprint {next(iter(fps))} of its boots is not the product "
                            f"{server} server's {want} (capserve.deploy would refuse its profile at every boot)")
        elif fps:
            bs = fps[want]
            groups[server] = {"fingerprint": want, "boots": [b.get("tag") for b in bs],
                              "workloads": list(dict.fromkeys(b["workload"] for b in bs))}
    return groups, problems


def argv_sha256_problems(analysis: dict) -> list:
    """043 §1.4's cross-check: analysis.platform.server_argv_sha256 records one server argv per server (every workload
    of B0 one value, every workload of T one value)."""
    by_server, problems = {}, []
    for w, value in (analysis["platform"].get("server_argv_sha256") or {}).items():
        if w not in sdplan.WORKLOADS:
            problems.append(f"analysis.json's server_argv_sha256 names {w!r}, not a Stage D workload")
            continue
        by_server.setdefault(sdplan.WORKLOADS[w]["server"], {}).setdefault(value, []).append(w)
    for server, values in by_server.items():
        if len(values) > 1:
            problems.append(f"analysis.json records {len(values)} server argv for server {server} "
                            f"({ {v[:12]: ws for v, ws in values.items()} }): one server per profile (043 §1.4)")
    return problems


# ---------------------------------------------------------------- 030 8.7 mapping

def scope(L: int, mix: str) -> str:
    prompts = (f"uniform {L - bp.N_GEN}-token prompts" if mix == "uniform" else
               f"the 007 mix of {L - bp.N_GEN}-token (90 %) and {bp.LONG_INPUT}-token (10 %) prompts")
    return (f"{prompts}, {bp.N_GEN} output tokens (ignore-eos); closed loop at C admitted requests, steady state "
            "(requests idx in [C, N - C)); latency after admission (030 8.1 / 8.7)")


def burst_max_ttft_ms(x: dict, c):
    """The start-up wave (idx < C, D1) at C_PROD: the largest first-wave TTFT over the point's valid runs."""
    if not c:
        return None
    runs = (x["points"].get(str(c)) or {}).get("runs") or []
    values = [r["burst_max_ttft_ms"] for r in runs if r.get("valid") and r.get("burst_max_ttft_ms") is not None]
    return max(values) if values else None


def unavailable_reason(w: str, x: dict, *, stage: str = "Stage C") -> str:
    status, K = x["c_prod"]["status"], x["K"]
    point = {C: x["points"][str(C)]["class"] for C in K}
    if status == "UNSTABLE":
        why = f"its lowest confirmation point C {K[0]} is above 1 and {point[K[0]]} (C_PROD reads MARGINAL as FAIL)"
    elif not K:
        why = "it has no confirmation points (its Phase S boots failed before a first bench)"
    else:
        why = "its confirmation points are not testable (" + ", ".join(f"C {C} {k}" for C, k in point.items()) + ")"
    return (f"{stage} {w}: C_PROD is {status}: {why}; no admission limit is verified, so the class is never served "
            "(030 8.7: never a number, never a fallback)")


def _mapping(w: str, x: dict, *, stage: str = "Stage C") -> tuple:
    """030 8.7: (max_concurrency, bound, unavailable_reason) of workload w's C_PROD."""
    status = x["c_prod"]["status"]
    if status in cp.BOUNDS:
        return x["c_prod"]["value"], cp.BOUNDS[status], None
    if status == cp.REFUSE_STATUS:
        return 0, "exact", None
    if status in cp.UNAVAILABLE_STATUSES:
        return None, None, unavailable_reason(w, x, stage=stage)
    raise GenerationError([f"{w}: unknown C_PROD status {status!r}"])


def class_entry(w: str, x: dict) -> dict:
    """One class of the verified profile from analysis.json's workload w (030 8.7)."""
    L, mix = x["L"], x["mix"]
    est = {"value": x["c_prod"]["value"], "status": x["c_prod"]["status"]}
    limit, bound, reason = _mapping(w, x)
    return {"name": cp.CLASS_NAMES[w], "max_prompt_tokens": bp.LONG_INPUT if mix == "mixed" else L - bp.N_GEN,
            "max_output_tokens": bp.N_GEN, "max_concurrency": limit, "max_queue": limit,   # 029: queue = concurrency
            "source": cp.stage_c_source(w, est["status"]), "status": "unavailable" if reason else "verified",
            "bound": bound, "unavailable_reason": reason,
            "evidence": {"workload": w, "scope": scope(L, mix), "c_prod": est,
                         "c_slo": {"value": x["c_slo"]["value"], "status": x["c_slo"]["status"]},
                         "c_kv": x["c_kv"], "c_cap": x["c_cap"], "flags": list(x["flags"]), "binding": x["binding"],
                         "margins": (x["margins"] or {}).get("c_prod"),
                         "burst_max_ttft_ms": burst_max_ttft_ms(x, limit)}}


def build_profile(analysis: dict, *, analysis_path: str, analysis_sha256: str, fingerprint: str, freeze_commit: str,
                  generator_commit: str) -> dict:
    """The verified profile of one reproduced analysis.json (deterministic: no timestamps)."""
    pf, pv, slo = analysis["platform"], analysis["provenance"], analysis["slo"]
    repo, _, revision = pf["model"].partition("@")
    doc = {"schema_version": cp.SCHEMA_VERSION, "profile_id": f"verified-{pv['session_id']}", "status": "verified",
           "provisional_reason": None, "slo": {"id": slo["id"], "terms": {k: v for k, v in slo.items() if k != "id"}},
           "server": {"vllm_version": pf["vllm"], "model_repo": repo, "model_revision": revision,
                      "fingerprint_sha256": fingerprint, "num_gpu_blocks": pf["num_gpu_blocks"],
                      "block_size": pf["block_size"]},
           "classes": [class_entry(w, x) for w, x in analysis["workloads"].items()],
           "provenance": {"source_session": pv["session_id"], "analysis_path": analysis_path,
                          "analysis_sha256": analysis_sha256, "prereg_section": cp.PREREG_SECTION,
                          "freeze_commit": freeze_commit, "generator_commit": generator_commit,
                          "deployed_commit": pv["deployed_commit"], "capacity_sha256": pv["capacity_sha256"],
                          "pins_sha256": pv["pins_sha256"], "gpu": pf["gpu"], "gpu_driver": pf["driver"]}}
    doc["content_sha256"] = cp.content_sha256(doc)
    return doc


def render(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _analysis_bytes(rep: dict) -> bytes:
    """capacity_report.main's (and stage_d_report.main's) serialization of analysis.json."""
    return (json.dumps(rep, indent=1) + "\n").encode()


# ---------------------------------------------------------------- Stage D (043 §1.5, §2, §3)

def stage_d_margins(x: dict):
    """1 - worst R per term at C_PROD (043 §1.5): a Stage D analysis records each point's worst_R but no margins, and
    this is capacity_analysis's own margins.c_prod formula, so no new statistic; None when C_PROD is no confirmation
    point (unavailable, INFEASIBLE), as capacity_analysis writes it."""
    point = x["points"].get(str(x["c_prod"]["value"]))
    if point is None:
        return None
    return {t: 1 - R for t, R in point["worst_R"].items() if R is not None}


def at_c_prod(x: dict, c):
    """043 §3: the C_PROD point's lowest attainment, most preemptions and highest KV peak over its valid runs (None
    without the point or a valid run). Preemptions and KV are the whole bench process's (042), not per request."""
    runs = [r for r in ((x["points"].get(str(c)) or {}).get("runs") or []) if r.get("valid")] if c else []
    if not runs:
        return None

    def pick(f, key):
        values = [r[key] for r in runs if r.get(key) is not None]
        return f(values) if values else None
    return {"attainment_min": pick(min, "attainment"), "preemptions_max": pick(max, "preemptions"),
            "kv_cache_usage_peak_max": pick(max, "kv_cache_usage_peak")}


def stage_d_scope(w: str, x: dict, pins: dict) -> str:
    _, prompt, out = cp.stage_d_class(w)
    prompts = (f"uniform {prompt}-token prompts" if x["mix"] == "uniform" else
               f"the 007 mix of {x['tier'] - bp.N_GEN}-token (90 %) and {bp.LONG_INPUT}-token (10 %) prompts")
    server = ("the Core B server (B0)" if x["server"] == "B0" else "the T server (B0 plus "
              f"--long-prefill-token-threshold {pins['advanced']['long_prefill_token_threshold']})")
    return (f"{prompts}, exactly {out} output tokens (ignore-eos), on {server}; closed loop at C admitted requests, "
            "steady state (requests idx in [C, N - C)); latency after admission (030 8.1 / 8.7, 039 D-A); at_c_prod's "
            "preemptions and KV peak are the whole bench process's (042)")


def stage_d_class_entry(w: str, x: dict, analysis: dict, pins: dict) -> dict:
    """One class of a §9 profile from analysis.json's D-A workload w (030 8.7 via 043): stage_d_class's name and
    limits; Stage C's evidence (Stage C's class_entry needs L and margins, which a Stage D analysis does not have) plus
    043 §3's, every value copied from analysis.json, never computed from the derived files."""
    name, max_prompt, max_output = cp.stage_d_class(w)
    est = {"value": x["c_prod"]["value"], "status": x["c_prod"]["status"]}
    limit, bound, reason = _mapping(w, x, stage="Stage D")
    return {"name": name, "max_prompt_tokens": max_prompt, "max_output_tokens": max_output,
            "max_concurrency": limit, "max_queue": limit,   # 029: queue = concurrency
            "source": cp.stage_d_source(w, est["status"]), "status": "unavailable" if reason else "verified",
            "bound": bound, "unavailable_reason": reason,
            "evidence": {"workload": w, "scope": stage_d_scope(w, x, pins), "c_prod": est,
                         "c_slo": {"value": x["c_slo"]["value"], "status": x["c_slo"]["status"]},
                         "c_kv": x["c_kv"], "c_cap": x["c_cap"], "flags": list(x["flags"]), "binding": x["binding"],
                         "margins": stage_d_margins(x), "burst_max_ttft_ms": burst_max_ttft_ms(x, limit),
                         "server": x["server"], "output_tokens": x["out"],
                         "descriptive": {k: {"value": x["descriptive"][k]["c_prod"]["value"],
                                             "status": x["descriptive"][k]["c_prod"]["status"]}
                                         for k in ("c_full", "c_late")},
                         "at_c_prod": at_c_prod(x, limit),
                         "hypotheses": {h: units[w]["verdict"] for h, units in analysis["hypotheses"].items()
                                        if w in units},
                         "not_covered": cp.not_covered_for(w, x["flags"])}}


def build_stage_d_profiles(analysis: dict, *, analysis_path: str, analysis_sha256: str, groups: dict,
                           freeze_commit: str, generator_commit: str, capacity_sha256: str, pins: dict) -> dict:
    """{profile_id: profile} of one reproduced D-A analysis.json, one per server with a D-A workload, B0 first
    (deterministic: no timestamps). groups is server_groups' (each server's fingerprint)."""
    pf, pv, slo = analysis["platform"], analysis["provenance"], analysis["slo"]
    repo, _, revision = pf["model"].partition("@")
    docs = {}
    for server in cp.SERVERS:
        ws = [w for w in analysis["workloads"] if sdplan.WORKLOADS[w]["server"] == server]
        if not ws:
            continue
        doc = {"schema_version": cp.SCHEMA_VERSION, "profile_id": f"verified-{pv['session_id']}-{server.lower()}",
               "status": "verified", "provisional_reason": None,
               "slo": {"id": slo["id"], "terms": {k: v for k, v in slo.items() if k != "id"}},
               "server": {"vllm_version": pf["vllm"], "model_repo": repo, "model_revision": revision,
                          "fingerprint_sha256": groups[server]["fingerprint"], "num_gpu_blocks": pf["num_gpu_blocks"],
                          "block_size": pf["block_size"],
                          "long_prefill_token_threshold": (pins["advanced"]["long_prefill_token_threshold"]
                                                           if server == "T" else None)},
               "classes": [stage_d_class_entry(w, analysis["workloads"][w], analysis, pins) for w in ws],
               "provenance": {"source_session": pv["session_id"], "analysis_path": analysis_path,
                              "analysis_sha256": analysis_sha256, "prereg_section": cp.STAGE_D_SECTION,
                              "freeze_commit": freeze_commit, "generator_commit": generator_commit,
                              "deployed_commit": pv["deployed_commit"], "capacity_sha256": capacity_sha256,
                              "pins_sha256": pv["pins_sha256"], "stage_d_sha256": pv["stage_d_sha256"],
                              "gpu": pf["gpu"], "gpu_driver": pf["driver"]}}
        doc["content_sha256"] = cp.content_sha256(doc)
        docs[doc["profile_id"]] = doc
    return docs


# ---------------------------------------------------------------- Stage V (044 產品對應; PREREGISTRATION §12)

def holding_units(analysis: dict) -> list:
    """044 產品對應: the units of analysis.json that hold, in its order: holds (a valid formal session), each of its
    points PASS and (PACE-2K) not vetoed. Only these write anything."""
    return [u for u, x in analysis["units"].items()
            if x["holds"] is True and x["vetoed"] is not True and all(k == "PASS" for k in x["classes"].values())]


def stage_v_validation(v: dict) -> dict:
    """A §12 validation (capserve.profile.Validation) from a unit's analysis.json validation: {study, unit, points,
    measured_classes, worst_r, n}; its queue, pacing, admission, datasets and mixes go where §12 states them."""
    return {k: copy.deepcopy(v[k]) for k in cp.Validation.model_fields}


def _base_class(base: dict, name: str) -> dict:
    for c in base["classes"]:
        if c["name"] == name:
            return copy.deepcopy(c)
    raise GenerationError([f"base profile {base['profile_id']} has no class {name!r} (044 產品對應)"])


def build_stage_v_profile(analysis: dict, *, analysis_path: str, analysis_sha256: str, fingerprint: str,
                          freeze_commit: str, generator_commit: str, capacity_sha256: str, bases: dict) -> dict | None:
    """The §12 profile of one reproduced Stage V analysis.json (044 產品對應; deterministic: no timestamps), or None
    when no unit that writes one holds. bases is base_profiles' {path: document}."""
    units, holds = analysis["units"], holding_units(analysis)
    known = set(NO_FILE_UNITS) | set(cp.COPY_UNITS.values()) | set(cp.V1_UNITS.values()) | {cp.ADMISSION_UNIT}
    unknown = [u for u in holds if u not in known]
    if unknown:
        raise GenerationError([f"analysis.json units {unknown} hold, but 044 產品對應 maps no artifact to them"])
    pf, pv, slo = analysis["platform"], analysis["provenance"], analysis["slo"]
    copies = {}                                          # workload -> class: c1b's, verbatim, plus a PACE unit
    for w, u in cp.COPY_UNITS.items():
        if u in holds:
            v = units[u]["validation"]
            c = _base_class(bases[COPY_BASE], cp.CLASS_NAMES[w])
            c.update(pacing={**v["pacing"], "queue": dict(v["queue"])}, validation=stage_v_validation(v))
            copies[w] = c
    admission = None
    if cp.ADMISSION_UNIT in holds:
        v = units[cp.ADMISSION_UNIT]["validation"]
        adm = v["admission"]
        unmapped = [a["name"] for a in adm["classes"] if a["name"] not in STUDY_COPIES]
        if unmapped:
            raise GenerationError([f"W1 measured the study classes {unmapped}, which are no copy of c1b's "
                                   f"{sorted(STUDY_COPIES)} (044 V4)"])
        members = [STUDY_COPIES[a["name"]] for a in adm["classes"]]
        for w in members:                                # a member without its PACE unit: c1b's class, plain
            copies.setdefault(w, _base_class(bases[COPY_BASE], cp.CLASS_NAMES[w]))
        admission = {"policy": adm["policy"], "budget": adm["budget"], "max_queue": adm["max_queue"],
                     "classes": [{**a, "name": cp.CLASS_NAMES[w]} for a, w in zip(adm["classes"], members)],
                     "queue": dict(v["queue"]), "measured_mixes": list(v["measured_mixes"]),
                     "gateway_path_sha256": pv["gateway_path_sha256"], "validation": stage_v_validation(v)}
    v1 = []
    for w, u in cp.V1_UNITS.items():
        if u in holds:
            v = units[u]["validation"]
            c = _base_class(bases[V1_BASE], cp.STAGE_D_CLASS_NAMES[w])
            ev = c["evidence"]
            ev.update(not_covered=cp.v1_not_covered(w, ev["flags"]), output_distribution=cp.V1_DISTRIBUTION,
                      datasets_sha256=list(v["datasets_sha256"][cp.V1_POINTS[w][0]]), via_gateway=True)
            c.update(name=cp.V1_CLASS_NAMES[w], max_concurrency=cp.V1_C[w], max_queue=cp.V1_C[w],
                     source=cp.stage_v_source(u), status="verified", bound=cp.VALIDATED, unavailable_reason=None,
                     validation=stage_v_validation(v))
            v1.append(c)
    classes = [copies[w] for w in cp.COPY_UNITS if w in copies] + v1
    if not classes:
        return None
    used = [rel for rel, on in ((COPY_BASE, copies), (V1_BASE, v1)) if on]
    repo, _, revision = pf["model"].partition("@")
    doc = {"schema_version": cp.SCHEMA_VERSION, "profile_id": f"verified-{pv['session_id']}-b0", "status": "verified",
           "provisional_reason": None, "slo": {"id": slo["id"], "terms": {k: v for k, v in slo.items() if k != "id"}},
           "server": {"vllm_version": pf["vllm"], "model_repo": repo, "model_revision": revision,
                      "fingerprint_sha256": fingerprint, "num_gpu_blocks": pf["num_gpu_blocks"],
                      "block_size": pf["block_size"], "long_prefill_token_threshold": None},
           "classes": classes, "admission": admission,
           "provenance": {"source_session": pv["session_id"], "analysis_path": analysis_path,
                          "analysis_sha256": analysis_sha256, "prereg_section": cp.STAGE_V_SECTION,
                          "freeze_commit": freeze_commit, "generator_commit": generator_commit,
                          "deployed_commit": pv["deployed_commit"], "capacity_sha256": capacity_sha256,
                          "pins_sha256": pv["pins_sha256"], "stage_d_sha256": pv["stage_d_sha256"],
                          "stage_v_sha256": pv["stage_v_sha256"], "gateway_path_sha256": pv["gateway_path_sha256"],
                          "capserve_sha256": pv["capserve_sha256"],
                          "base_profiles": [{"id": bases[rel]["profile_id"],
                                             "content_sha256": bases[rel]["content_sha256"]} for rel in used],
                          "gpu": pf["gpu"], "gpu_driver": pf["driver"]}}
    doc["content_sha256"] = cp.content_sha256(doc)
    return doc


# ---------------------------------------------------------------- the generator

def _session_problems(d: Path, cap: dict, root: Path) -> list:
    sid = cap["session"].get("session_id")
    problems = []
    if not (isinstance(sid, str) and sid.startswith("formal-")):
        problems.append(f"session_id {sid!r} is not a formal- session (029)")
    if d.resolve() != root.resolve() / "results" / "capacity" / str(sid):
        problems.append(f"{d} is not results/capacity/<session_id> ({sid!r}) of {root}")
    return problems


def generate(session_dir, *, freeze_commit: str, generator_commit: str, root=ROOT, configs: tuple | None = None,
             git: Git | None = None) -> dict:
    """The verified profile of a formal Stage C session, or GenerationError with every problem found. A session or
    analysis.json too malformed for the read-only Stage C checks to run is refused too, never a crash."""
    root = Path(root)
    configs = configs if configs is not None else _configs(root)
    try:
        return _generate(Path(session_dir), freeze_commit=freeze_commit, generator_commit=generator_commit,
                         root=root, configs=configs, git=git or git_runner(root))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise GenerationError([f"malformed session or analysis.json ({type(exc).__name__}: {exc})"]) from exc


def _generate(d: Path, *, freeze_commit: str, generator_commit: str, root: Path, configs: tuple, git: Git) -> dict:
    cfg, pins, cap_sha, pins_sha = configs
    if not frozen(cfg):
        raise GenerationError([NOT_FROZEN.format(path=root / FREEZE_FILE)])
    problems = [f"freeze evidence: {p}" for p in cplan.freeze_evidence_problems(cfg, root)]
    try:
        cap = cr.load_session(d)
    except (OSError, ValueError) as exc:
        raise GenerationError(problems + [f"cannot read the session {d}: {exc}"]) from exc
    s = cap["session"]
    problems += _session_problems(d, cap, root)
    problems += [f"freeze commit: {p}" for p in
                 freeze_commit_problems(freeze_commit, deployed_commit=s.get("repo_commit"), capacity_sha256=cap_sha,
                                        git=git)]
    analysis_path = d / cr.ANALYSIS_NAME
    raw = analysis = uniform = advanced = None
    try:
        raw = analysis_path.read_bytes()
        analysis = json.loads(raw)
    except (OSError, ValueError) as exc:
        problems.append(f"no readable {analysis_path} ({exc}): run capacity_report on the session first")
    if raw is not None and not isinstance(analysis, dict):
        problems.append(f"{analysis_path} is not a JSON object")
    try:
        uniform = cr.load_uniform(root / "results" / "core_b" / str(cap["plan"].get("derived_from")))
    except (OSError, ValueError) as exc:
        problems.append(f"cannot read the uniform session the plan derives from ({exc})")
    if isinstance(analysis, dict) and (analysis.get("report") or {}).get("advanced_references") is True:
        try:
            advanced = cr.load_advanced_references(root / "results" / "advanced" / ADVANCED_SESSION)
        except (OSError, ValueError) as exc:
            problems.append(f"analysis.json used the Advanced references, which cannot be read ({exc})")
    if uniform is not None:
        problems += [f"capacity_report: {p}" for p in
                     cr.refusal_problems(cap, cfg=cfg, pins=pins, capacity_sha256=cap_sha, pins_sha256=pins_sha,
                                         uniform=uniform, repo_root=root)]
    if isinstance(analysis, dict) and analysis.get("slo") != cfg.get("slo"):
        problems.append(f"the analysis SLO {(analysis.get('slo') or {}).get('id')!r} is not capacity.toml's SLO "
                        f"{(cfg.get('slo') or {}).get('id')!r} (030 8.7)")
    fingerprint, fp_problems = session_fingerprint(s, pins)
    problems += fp_problems
    if problems:
        raise GenerationError(problems)
    try:
        rep = cr.report(cap, cfg=cfg, pins=pins, capacity_sha256=cap_sha, pins_sha256=pins_sha, uniform=uniform,
                        advanced_runs=advanced, repo_root=root)
    except cr.ReportError as exc:
        raise GenerationError([f"capacity_report refuses the session: {exc}"]) from exc
    if _analysis_bytes(rep) != raw:
        keys = sorted(k for k in set(rep) | set(analysis) if rep.get(k) != analysis.get(k))
        raise GenerationError([f"{analysis_path} is not what the frozen Stage C modules reproduce byte for byte "
                               f"(021 R3; differing top-level keys: {keys or 'none, only the bytes'})"])
    doc = build_profile(analysis, analysis_path=analysis_path.resolve().relative_to(root.resolve()).as_posix(),
                        analysis_sha256=_sha256(raw), fingerprint=fingerprint, freeze_commit=freeze_commit,
                        generator_commit=generator_commit)
    try:
        cp.parse_profile(render(doc), source="the generated profile", allow_provisional=False, pins=pins,
                         frozen=cp.FrozenConfigs(cfg, cap_sha, pins_sha))
    except cp.ProfileError as exc:
        raise GenerationError([f"the generated profile fails the loader (a generator defect): {exc}"]) from exc
    return doc


def is_stage_d_session(session) -> bool:
    """043 §1.1's dispatch: a session at results/stage_d/<id> is Stage D's; anything else goes to Stage C, which
    refuses what is not results/capacity/<id>."""
    return Path(session).resolve().parent.parts[-2:] == STAGE_D_RESULTS


def _stage_d_session_problems(d: Path, cap: dict, root: Path) -> list:
    sid, kind = cap["session"].get("session_id"), cap["plan"].get("kind")
    problems = []
    if not (isinstance(sid, str) and sid.startswith("formal-")):
        problems.append(f"session_id {sid!r} is not a formal- session (029)")
    if d.resolve() != root.resolve().joinpath(*STAGE_D_RESULTS, str(sid)):
        problems.append(f"{d} is not results/stage_d/<session_id> ({sid!r}) of {root}")
    if kind != D_A_KIND:
        problems.append(NOT_D_A.format(what="plan.json", kind=kind, want=D_A_KIND))
    return problems


def _d_a_workload_problems(analysis: dict, groups: dict) -> list:
    """Every workload of the analysis is a D-A one (a class: never the sentinel, D-B or D-C), and its server has a
    fingerprint group."""
    problems = []
    for w in analysis["workloads"]:
        if w not in cp.STAGE_D_CLASS_NAMES:
            problems.append(f"analysis.json workload {w!r} is not a D-A workload {list(cp.STAGE_D_CLASS_NAMES)} (043)")
        elif sdplan.WORKLOADS[w]["server"] not in groups:
            problems.append(f"{w} runs on server {sdplan.WORKLOADS[w]['server']}, whose boots give no product "
                            "fingerprint (043 §1.4)")
    return problems


def generate_stage_d(session_dir, *, freeze_commit: str, generator_commit: str, root=ROOT,
                     configs: tuple | None = None, stage_d_configs: tuple | None = None, git: Git | None = None) -> dict:
    """{profile_id: verified profile} of a formal Stage D D-A session, one per server (043 §1), or GenerationError with
    every problem found. configs is load_configs' (capacity.toml: frozen too, its sha256 in the provenance),
    stage_d_configs load_stage_d_configs'. A session or analysis.json too malformed for the read-only Stage D checks
    is refused too, never a crash."""
    root = Path(root)
    configs = configs if configs is not None else _configs(root)
    stage_d_configs = stage_d_configs if stage_d_configs is not None else _stage_d_configs(root)
    try:
        return _generate_stage_d(Path(session_dir), freeze_commit=freeze_commit, generator_commit=generator_commit,
                                 root=root, configs=configs, stage_d_configs=stage_d_configs,
                                 git=git or git_runner(root))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise GenerationError([f"malformed session or analysis.json ({type(exc).__name__}: {exc})"]) from exc


def _generate_stage_d(d: Path, *, freeze_commit: str, generator_commit: str, root: Path, configs: tuple,
                      stage_d_configs: tuple, git: Git) -> dict:
    cfg, _, cap_sha, _ = configs
    sd_cfg, pins, sd_sha, pins_sha = stage_d_configs
    if not frozen(sd_cfg, "stage_d"):
        raise GenerationError([NOT_FROZEN_D.format(path=root / STAGE_D_FREEZE_FILE)])
    problems = [] if frozen(cfg) else [
        f"[freeze] capacity is not true in {root / FREEZE_FILE}: Stage D runs S2 and the frozen Stage C modules (039), "
        "so no Stage D profile exists before Stage C's freeze"]
    problems += [f"freeze evidence: {p}" for p in sdplan.freeze_evidence_problems(sd_cfg, root)]
    try:
        cap = cr.load_session(d)
    except (OSError, ValueError) as exc:
        raise GenerationError(problems + [f"cannot read the session {d}: {exc}"]) from exc
    s = cap["session"]
    problems += _stage_d_session_problems(d, cap, root)
    problems += [f"freeze commit: {p}" for p in
                 freeze_commit_problems(freeze_commit, deployed_commit=s.get("repo_commit"), capacity_sha256=sd_sha,
                                        git=git, freeze_file=STAGE_D_FREEZE_FILE, freeze_key="stage_d",
                                        modules=STAGE_D_MODULES)]
    analysis_path = d / sr.ANALYSIS_NAME
    raw = analysis = uniform = advanced = None
    try:
        raw = analysis_path.read_bytes()
        analysis = json.loads(raw)
    except (OSError, ValueError) as exc:
        problems.append(f"no readable {analysis_path} ({exc}): run stage_d_report on the session first")
    if raw is not None and not isinstance(analysis, dict):
        problems.append(f"{analysis_path} is not a JSON object")
        analysis = None
    if analysis is not None and (analysis.get("provenance") or {}).get("kind") != D_A_KIND:
        problems.append(NOT_D_A.format(what="analysis.json", kind=(analysis.get("provenance") or {}).get("kind"),
                                       want=D_A_KIND))
    try:
        uniform = cr.load_uniform(root / "results" / "core_b" / str(cap["plan"].get("derived_from")))
    except (OSError, ValueError) as exc:
        problems.append(f"cannot read the uniform session the plan derives from ({exc})")
    if analysis is not None and (analysis.get("report") or {}).get("advanced_references") is True:
        try:
            advanced = cr.load_advanced_references(root / "results" / "advanced" / ADVANCED_SESSION)
        except (OSError, ValueError) as exc:
            problems.append(f"analysis.json used the Advanced references, which cannot be read ({exc})")
    problems += [f"stage_d_report: {p}" for p in
                 sr.refusal_problems(cap, cfg=sd_cfg, pins=pins, stage_d_sha256=sd_sha, pins_sha256=pins_sha,
                                     uniform=uniform, repo_root=root)]
    if sd_cfg.get("slo") != cfg.get("slo"):
        problems.append(f"stage_d.toml's SLO {(sd_cfg.get('slo') or {}).get('id')!r} is not capacity.toml's "
                        f"{(cfg.get('slo') or {}).get('id')!r} (037 §3: both S2)")
    if analysis is not None and analysis.get("slo") != sd_cfg.get("slo"):
        problems.append(f"the analysis SLO {(analysis.get('slo') or {}).get('id')!r} is not stage_d.toml's SLO "
                        f"{(sd_cfg.get('slo') or {}).get('id')!r} (043 §1.3)")
    groups, group_problems = server_groups(s, pins)
    problems += group_problems
    if analysis is not None:
        problems += argv_sha256_problems(analysis)
        if cap["plan"].get("kind") == D_A_KIND:
            problems += _d_a_workload_problems(analysis, groups)
    if problems:
        raise GenerationError(problems)
    try:
        rep = sr.report(cap, cfg=sd_cfg, pins=pins, stage_d_sha256=sd_sha, pins_sha256=pins_sha, uniform=uniform,
                        advanced_runs=advanced, repo_root=root)
    except sr.ReportError as exc:
        raise GenerationError([f"stage_d_report refuses the session: {exc}"]) from exc
    if _analysis_bytes(rep) != raw:
        keys = sorted(k for k in set(rep) | set(analysis) if rep.get(k) != analysis.get(k))
        raise GenerationError([f"{analysis_path} is not what the frozen Stage D modules reproduce byte for byte "
                               f"(021 R3; differing top-level keys: {keys or 'none, only the bytes'})"])
    docs = build_stage_d_profiles(analysis, groups=groups, freeze_commit=freeze_commit,
                                  analysis_path=analysis_path.resolve().relative_to(root.resolve()).as_posix(),
                                  analysis_sha256=_sha256(raw), generator_commit=generator_commit,
                                  capacity_sha256=cap_sha, pins=pins)
    for pid, doc in docs.items():
        try:
            cp.parse_profile(render(doc), source=f"the generated profile {pid}", allow_provisional=False, pins=pins,
                             frozen=cp.FrozenConfigs(cfg, cap_sha, pins_sha, sd_cfg, sd_sha))
        except cp.ProfileError as exc:
            raise GenerationError([f"the generated profile fails the loader (a generator defect): {exc}"]) from exc
    return docs


def is_stage_v_session(session) -> bool:
    """A session at results/stage_v/<id> is Stage V's: --validation-session reads it, --session never does (044)."""
    return Path(session).resolve().parent.parts[-2:] == STAGE_V_RESULTS


def stage_v_server_groups(session: dict, plan: dict, pins: dict) -> tuple:
    """server_groups (043 §1.4) of a Stage V session: a V boot's workload is its point, so it is grouped by the point's
    server_base (B0, or T for GW-M8K-T; the U2K sentinel runs on B0)."""
    points = plan.get("points") or {}
    boots = [{**b, "workload": points[b["workload"]]["server_base"]}
             if b.get("phase") == "V" and b.get("workload") in points else b for b in session.get("boots") or []]
    return server_groups({**session, "boots": boots}, pins)


def _stage_v_session_problems(d: Path, cap: dict, root: Path) -> list:
    sid = cap["session"].get("session_id")
    problems = []
    if not (isinstance(sid, str) and sid.startswith("formal-")):
        problems.append(f"session_id {sid!r} is not a formal- session (044: a mini or smoke never gives a profile)")
    if d.resolve() != root.resolve().joinpath(*STAGE_V_RESULTS, str(sid)):
        problems.append(f"{d} is not results/stage_v/<session_id> ({sid!r}) of {root}")
    return problems


def _started_at(session) -> dt.datetime:
    """A session record's started_at, a time with a zone offset (044 實作補記八); KeyError / TypeError / ValueError
    when it records none."""
    when = dt.datetime.fromisoformat(session["started_at"])
    if when.utcoffset() is None:
        raise ValueError(f"started_at {session['started_at']!r} has no time zone offset")
    return when


def _closed(why: str) -> str:
    return f"{why}; {RERUN_RULE}, checked fail closed"


def previous_formal_problems(d: Path, s: dict, *, root: Path, dirs: list) -> list:
    """044 停止規則 reads only the directories here, so every earlier formal session must be here (044 實作補記十第 10
    項): walk the session's previous_formal (run_stage_v records, at the start, the formal session of either plan that
    finished last among those that started at or before it: gap_problems) back to the first one (null). A session it
    names with no results/stage_v/formal-* directory whose session.json records that session_id, a chain that cannot
    be followed or that returns to a session, is a problem (fail closed). A session aborted in its preflight records no
    previous_formal (the runner records it after the preflight): the walk goes on from the session the runner's own
    rule finds among the directories here; one that records none and was not aborted is a problem. A session no record
    leads to (one just before a preflight abort) is hours_problems' to find (044 實作補記十第 11 項)."""
    named, by_id = {}, {}
    for e in dirs:
        try:
            rec = json.loads((e / "session.json").read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        named[e.name] = (e, rec)
        if isinstance(rec.get("session_id"), str) and (rec["session_id"] == e.name or rec["session_id"] not in by_id):
            by_id[rec["session_id"]] = (e, rec)
    e, rec, seen = d, s, {d.resolve()}
    while True:
        who = rec.get("session_id")
        if "previous_formal" in rec:
            prev = rec["previous_formal"]
            if prev is None:
                return []
            pid = prev.get("session_id") if isinstance(prev, dict) else None
            if not isinstance(pid, str):
                return [_closed(f"{who}'s previous_formal {prev!r} names no session: the formal sessions before it "
                                "cannot be told")]
            if pid not in by_id:
                here = root.joinpath(*STAGE_V_RESULTS, pid).is_dir()
                return [_closed(f"cannot follow the previous_formal chain past {pid} ({who}'s previous_formal): "
                                f"results/stage_v/{pid} has no readable session.json" if here else
                                f"the formal Stage V session {pid} ({who}'s previous_formal) has no directory here (no "
                                "results/stage_v/formal-* whose session.json records that session_id), so whether it "
                                "is an earlier valid session of this plan under this freeze is unknown: fetch every "
                                "formal Stage V session before generating")]
            e, rec = by_id[pid]
        elif rec.get("aborted"):
            try:                                          # its own 2 h gap problems are the report's (044 實作補記七)
                prev, _ = rv.gap_problems(root, rec["started_at"], session_id=e.name)
            except (KeyError, TypeError, ValueError) as exc:
                return [_closed(f"{who} (aborted in its preflight) records no previous_formal and no usable start "
                                f"({type(exc).__name__}: {exc}): the formal sessions before it cannot be told")]
            if prev is None:
                return []
            e, rec = named[prev["session_id"]]
        else:
            return [_closed(f"{who} records no previous_formal and was not aborted (a session past its preflight "
                            "records one): the formal sessions before it cannot be told")]
        if e.resolve() in seen:
            return [_closed(f"the previous_formal chain returns to {e.name}: the formal sessions before it cannot be "
                            "told")]
        seen.add(e.resolve())


def hours_problems(d: Path, s: dict, *, root: Path) -> list:
    """044 停止規則 reads only the directories here, and the previous_formal walk finds only the sessions a record leads
    to, so every Stage V directory the runner counted must be here (044 實作補記十第 11 項): run_stage_v records in
    session.json hours_used_h, float(run_stage_v.hours_used) over every results/stage_v session directory at its start
    (the mini, formal ones, reruns and aborted ones alike). The directories here that started by its start (hours_used
    with not_after, as stage_v_report counts them), without its own hours, must add up to exactly that with no problem
    counting them; a session missing, moved out of results/stage_v, its start moved past this one's or one directory
    too many (a copy, a decoy) does not (fail closed)."""
    problems, before = [], None
    try:
        here, problems = rv.hours_used(root, not_after=s["started_at"])
        began, ended, _ = rv._times(d)                  # its own hours, as hours_used counts them
        before = float(here - Fraction((ended - began) // dt.timedelta(microseconds=1), 3600 * 10 ** 6))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        problems = [*problems, f"{d.name}: {type(exc).__name__}: {exc}"]
    if problems or before != s.get("hours_used_h"):
        return [_closed(f"fetch every Stage V session before generating; the directories on the Mac do not add up to "
                        f"the hours this session counted at its start ({s.get('session_id')}'s hours_used_h "
                        f"{s.get('hours_used_h')!r}; the results/stage_v sessions here that started by its start, "
                        f"without it: {before!r} h" + "".join(f"; {p}" for p in problems) + ")")]
    return []


def _other_plan(e: Path):
    """The plan session (V-ADM / V-CLS) of the formal-* directory e when its plan.json is the plan its session.json
    records it ran (plan_sha256) and of the session.json's plan session; else None: maybe this plan (044 實作補記十第
    11 項)."""
    try:
        plan, rec = json.loads((e / "plan.json").read_text()), json.loads((e / "session.json").read_text())
        if svr.sha256_json(plan) == rec.get("plan_sha256") and rec.get("session") == plan.get("session"):
            return plan.get("session")
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return None


def report_verdict(exc: Exception) -> bool:
    """Whether stage_v_report.report's ReportError is its own verdict that the session is invalid (REPORT_VERDICTS)."""
    return str(exc).startswith(REPORT_VERDICTS)


def rerun_problems(d: Path, cap: dict, *, root: Path, evaluate: Callable) -> list:
    """044 停止規則 (不重跑有效的 session 來換 PASS), mechanized here, outside the request path and the harness: no
    results/stage_v/formal-* session other than this one, of its plan session (V-ADM / V-CLS), that started before it
    under the same freeze (the same recorded STAGE_V_FREEZE_SHAS) is valid. A directory is another plan's only when its
    plan.json is the plan its session.json records it ran (_other_plan). Such an earlier session that has an
    analysis.json blocks the rerun outright: stage_v_report writes one only for a session it accepts (044 實作補記十第
    11 項). Otherwise it releases the rerun only on positive evidence that it was invalid (第 10 項): its session.json
    records it aborted or env_valid false, or, with stage_v_report.refusal_problems empty, report() raises its own
    validity verdict (REPORT_VERDICTS: the replay, the sentinel gate; 第 11 項). evaluate(earlier cap) is (what keeps
    stage_v_report from judging it: refusal_problems, or the analysis failing; whether report() accepts it).
    Everything else fails closed: a directory that is not its session_id (renamed), files that are not what its
    session.json records (stage_v_report's sha256 checks against its own records), refusal problems, the analysis
    failing, any other error; a formal-* directory of this plan, or of a plan that cannot be confirmed, without a
    readable session.json and started_at; a formal session of the previous_formal chain that is not here
    (previous_formal_problems); directories here that do not add up to the hours the session counted at its start
    (hours_problems). The 2-attempt cap stays with the operator (044 實作補記四第 4 項, 實作補記十). It guards against
    honest mistakes, not deliberate edits of a session.json (it has no sha256 of its own; 實作補記十第 11 項)."""
    s, plan = cap["session"], cap["plan"].get("session")
    dirs = sorted(p for p in root.joinpath(*STAGE_V_RESULTS).glob("formal-*") if p.is_dir())
    problems = previous_formal_problems(d, s, root=root, dirs=dirs)
    try:
        start = _started_at(s)
    except (KeyError, TypeError, ValueError) as exc:
        return problems + [f"the session records no readable started_at ({exc}): cannot tell which {plan} sessions "
                           f"came before it ({RERUN_RULE}, checked fail closed)"]
    problems += hours_problems(d, s, root=root)
    for e in dirs:
        if e.resolve() == d.resolve():
            continue
        other = _other_plan(e)
        if other is not None and other != plan:
            continue
        try:
            rec = json.loads((e / "session.json").read_text())
            began = _started_at(rec)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            problems.append(f"cannot tell whether {e.name} is an earlier valid {plan} session under this freeze (no "
                            f"readable session.json / started_at: {type(exc).__name__}: {exc}): {RERUN_RULE}, checked "
                            "fail closed")
            continue
        if began >= start or any(rec.get(k) != s.get(k) for k in STAGE_V_FREEZE_SHAS):
            continue
        if (e / svr.ANALYSIS_NAME).exists():
            problems.append(f"{s.get('session_id')} reruns the earlier {plan} session {e.name} (started "
                            f"{rec['started_at']}), which ran under the same freeze and has an analysis.json "
                            "(stage_v_report writes one only for a session it accepts): its units are the result, "
                            f"{RERUN_RULE} (044 實作補記十第 11 項)")
            continue
        if rec.get("session_id") != e.name:
            problems.append(_closed(f"{e.name} records session_id {rec.get('session_id')!r}: a renamed directory "
                                    f"(results/stage_v/<session_id>) is not read as an earlier {plan} session under "
                                    "this freeze"))
            continue
        try:
            earlier = cr.load_session(e)
            files = svr._sha_problems(earlier, **{k: rec.get(k) for k in ("stage_v_sha256", "stage_d_sha256",
                                                                          "pins_sha256")})
        except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            problems.append(_closed(f"cannot tell whether the earlier {plan} session {e.name} under this freeze is "
                                    f"valid ({type(exc).__name__}: {exc})"))
            continue
        if files:
            problems.append(_closed(f"the files of the earlier {plan} session {e.name} under this freeze are not what "
                                    f"its session.json records ({'; '.join(files)}), so whether it is valid is "
                                    "unknown"))
            continue
        if rec.get("aborted") or rec.get("env_valid") is False:
            continue                                     # positive evidence: invalid, rerun under a new id (013)
        try:
            refusals, valid = evaluate(earlier)
        except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            problems.append(_closed(f"cannot tell whether the earlier {plan} session {e.name} under this freeze is "
                                    f"valid ({type(exc).__name__}: {exc})"))
            continue
        if refusals:
            problems.append(_closed(f"stage_v_report refuses to evaluate the earlier {plan} session {e.name} under "
                                    f"this freeze ({'; '.join(refusals)}), so whether it is valid is unknown"))
        elif valid:
            problems.append(f"{s.get('session_id')} reruns the earlier {plan} session {e.name} (started "
                            f"{rec['started_at']}), which ran under the same freeze and is valid (stage_v_report "
                            f"accepts it): its units are the result, {RERUN_RULE} (044 實作補記十)")
    return problems


def generate_stage_v(session_dir, *, freeze_commit: str, generator_commit: str, root=ROOT, configs: tuple | None = None,
                     stage_v_configs: tuple | None = None, git: Git | None = None) -> tuple:
    """({profile_id: §12 profile}, the units that hold) of a formal Stage V session (044 產品對應): at most one profile,
    none when no unit that writes one holds; or GenerationError with every problem found. configs is load_configs'
    (capacity.toml: frozen, its sha256 in the provenance), stage_v_configs load_stage_v_configs'. A session or
    analysis.json too malformed for the read-only Stage V checks is refused too, never a crash."""
    root = Path(root)
    configs = configs if configs is not None else _configs(root)
    stage_v_configs = stage_v_configs if stage_v_configs is not None else _stage_v_configs(root)
    try:
        return _generate_stage_v(Path(session_dir), freeze_commit=freeze_commit, generator_commit=generator_commit,
                                 root=root, configs=configs, stage_v_configs=stage_v_configs,
                                 git=git or git_runner(root))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise GenerationError([f"malformed session or analysis.json ({type(exc).__name__}: {exc})"]) from exc


def _generate_stage_v(d: Path, *, freeze_commit: str, generator_commit: str, root: Path, configs: tuple,
                      stage_v_configs: tuple, git: Git) -> tuple:
    cfg, _, cap_sha, _ = configs
    sv_cfg, sd_cfg, pins, sv_sha, sd_sha, pins_sha = stage_v_configs
    if not frozen(sv_cfg, "stage_v"):
        raise GenerationError([NOT_FROZEN_V.format(path=root / STAGE_V_FREEZE_FILE)])
    problems = [] if frozen(cfg) else [
        f"[freeze] capacity is not true in {root / FREEZE_FILE}: Stage V runs S2 and the frozen Stage C modules, and a "
        "§12 profile loads only after Stage C's freeze"]
    if not frozen(sd_cfg, "stage_d"):
        problems.append(f"[freeze] stage_d is not true in {root / STAGE_D_FREEZE_FILE}: Stage V reads the frozen "
                        "stage_d.toml (044), and a §12 profile loads only after Stage D's freeze")
    problems += [f"freeze evidence: {p}" for p in svplan.freeze_evidence_problems(sv_cfg, root)]
    try:
        cap = cr.load_session(d)
    except (OSError, ValueError) as exc:
        raise GenerationError(problems + [f"cannot read the session {d}: {exc}"]) from exc
    s = cap["session"]
    problems += _stage_v_session_problems(d, cap, root)
    problems += [f"freeze commit: {p}" for p in
                 freeze_commit_problems(freeze_commit, deployed_commit=s.get("repo_commit"), capacity_sha256=sv_sha,
                                        git=git, freeze_file=STAGE_V_FREEZE_FILE, freeze_key="stage_v",
                                        modules=STAGE_V_MODULES)]
    deployed_path, path_problems = deployed_gateway_path_sha256(s.get("repo_commit"), git=git)
    problems += path_problems
    frozen_path = (sv_cfg.get("freeze") or {}).get("gateway_path_sha256")
    if deployed_path is not None and deployed_path != frozen_path:
        problems.append(f"the request path at the session's DEPLOYED_COMMIT {s.get('repo_commit')} "
                        f"(gateway_path_sha256 {deployed_path}) is not configs/stage_v.toml's frozen {frozen_path!r}: "
                        "the units were validated on the frozen request path only (044 產品對應)")
    frozen_cfgs = cp.FrozenConfigs(cfg, cap_sha, pins_sha, sd_cfg, sd_sha, sv_cfg, sv_sha)
    bases, base_problems = base_profiles(root, freeze_commit, git=git, pins=pins, frozen_cfgs=frozen_cfgs)
    problems += base_problems
    analysis_path = d / svr.ANALYSIS_NAME
    raw = analysis = uniform = advanced = None
    try:
        raw = analysis_path.read_bytes()
        analysis = json.loads(raw)
    except (OSError, ValueError) as exc:
        problems.append(f"no readable {analysis_path} ({exc}): run stage_v_report on the session first")
    if raw is not None and not isinstance(analysis, dict):
        problems.append(f"{analysis_path} is not a JSON object")
        analysis = None
    try:
        uniform = cr.load_uniform(root / "results" / "core_b" / str(sd_cfg["stage_d"]["derived_from"]))
    except (OSError, ValueError) as exc:
        problems.append(f"cannot read stage_d.toml's Core B uniform session (the sentinel's references) ({exc})")
    if analysis is not None and (analysis.get("report") or {}).get("advanced_references") is True:
        try:
            advanced = cr.load_advanced_references(root / "results" / "advanced" / ADVANCED_SESSION)
        except (OSError, ValueError) as exc:
            problems.append(f"analysis.json used the Advanced references, which cannot be read ({exc})")
    report_kw = {"cfg": sv_cfg, "stage_d_cfg": sd_cfg, "pins": pins, "stage_v_sha256": sv_sha,
                 "stage_d_sha256": sd_sha, "pins_sha256": pins_sha, "repo_root": root}
    problems += [f"stage_v_report: {p}" for p in svr.refusal_problems(cap, **report_kw)]

    def evaluate(earlier: dict) -> tuple:
        """(what keeps stage_v_report from judging an earlier session, whether report() accepts it): its
        refusal_problems; else report() raising its own verdict (report_verdict: the replay, the sentinel gate) is
        ([], False), any other ReportError (the analysis failed) is no verdict ([its message], False), rerun_problems."""
        refusals = svr.refusal_problems(earlier, **report_kw)
        if refusals:
            return refusals, False
        try:
            svr.report(earlier, uniform=uniform, advanced_runs=advanced, **report_kw)
        except svr.ReportError as exc:
            return ([] if report_verdict(exc) else [str(exc)]), False
        return [], True

    problems += rerun_problems(d, cap, root=root, evaluate=evaluate)
    if analysis is not None and analysis.get("slo") != sv_cfg.get("slo"):
        problems.append(f"the analysis SLO {(analysis.get('slo') or {}).get('id')!r} is not stage_v.toml's SLO "
                        f"{(sv_cfg.get('slo') or {}).get('id')!r} (044)")
    groups, group_problems = stage_v_server_groups(s, cap["plan"], pins)
    problems += group_problems
    if "B0" not in groups and not group_problems:
        problems.append("no boot of the session ran the product B0 server: a §12 profile is B0's (044)")
    if problems:
        raise GenerationError(problems)
    try:
        rep = svr.report(cap, uniform=uniform, advanced_runs=advanced, **report_kw)
    except svr.ReportError as exc:
        raise GenerationError([f"stage_v_report refuses the session: {exc}"]) from exc
    if _analysis_bytes(rep) != raw:
        keys = sorted(k for k in set(rep) | set(analysis) if rep.get(k) != analysis.get(k))
        raise GenerationError([f"{analysis_path} is not what the frozen Stage V modules reproduce byte for byte "
                               f"(021 R3; differing top-level keys: {keys or 'none, only the bytes'})"])
    doc = build_stage_v_profile(analysis, analysis_path=analysis_path.resolve().relative_to(root.resolve()).as_posix(),
                                analysis_sha256=_sha256(raw), fingerprint=groups["B0"]["fingerprint"],
                                freeze_commit=freeze_commit, generator_commit=generator_commit,
                                capacity_sha256=cap_sha, bases=bases)
    docs = {} if doc is None else {doc["profile_id"]: doc}
    for pid, doc in docs.items():
        try:
            cp.parse_profile(render(doc), source=f"the generated profile {pid}", allow_provisional=False, pins=pins,
                             frozen=frozen_cfgs)
        except cp.ProfileError as exc:
            raise GenerationError([f"the generated profile fails the loader (a generator defect): {exc}"]) from exc
    return docs, holding_units(analysis)


def _summary(doc: dict, out: Path) -> dict:
    return {"profile_id": doc["profile_id"], "out": str(out),
            "classes": {c["name"]: [c["status"], c["max_concurrency"], c["bound"]] for c in doc["classes"]}}


def _stage_v_summary(doc: dict, out: Path) -> dict:
    """What a §12 profile adds (044 實作補記十第 5 項): its admission policy (or None), and per class its pacing
    parameters (or None) and its validation's unit (or None: a plain admission member)."""
    pacing = ("burst_tokens", "rate_tokens_per_s")
    return {"profile_id": doc["profile_id"], "out": str(out), "admission": (doc["admission"] or {}).get("policy"),
            "classes": {c["name"]: {"status": c["status"], "max_concurrency": c["max_concurrency"], "bound": c["bound"],
                                    "pacing": {k: c["pacing"][k] for k in pacing} if c.get("pacing") else None,
                                    "unit": (c.get("validation") or {}).get("unit")} for c in doc["classes"]}}


def main(argv: list | None = None, *, root=ROOT, git: Git | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(root)
    git = git or git_runner(root)
    try:
        if args.validation_session is not None:
            return _main_stage_v(args, root=root, git=git)
        if is_stage_v_session(args.session):
            raise GenerationError([f"{args.session} is a Stage V session (results/stage_v/): it is read with "
                                   "--validation-session, never --session (044 產品對應)"])
        if is_stage_d_session(args.session):
            return _main_stage_d(args, root=root, git=git)
        if args.out is None:
            raise GenerationError([f"{args.session} is not a Stage D session (results/stage_d/<session_id>): a Stage C "
                                   "session (results/capacity/) writes its one profile with --out, never --out-dir "
                                   "(043 §1.1)"])
        out = Path(args.out)
        configs = _configs(root)
        commit, problems = generator_commit(git)
        problems += _out_problems(out, root)
        if problems:
            raise GenerationError(problems)
        doc = generate(args.session, freeze_commit=args.freeze_commit, generator_commit=commit, root=root,
                       configs=configs, git=git)
    except GenerationError as exc:
        print(f"profile_gen: refused: {exc}", file=sys.stderr)
        return 2
    out.write_bytes(render(doc))
    print(json.dumps(_summary(doc, out)))
    return 0


def _main_stage_d(args, *, root: Path, git: Git) -> int:
    """043 §1.1: every server's profile of a D-A session into --out-dir, all or none (GenerationError: refused)."""
    if args.out_dir is None:
        raise GenerationError([f"{args.session} is a Stage D session (results/stage_d/): it writes every server's "
                               "profile with --out-dir, never --out (043 §1.1)"])
    out_dir = Path(args.out_dir)
    configs, stage_d_configs = _configs(root), _stage_d_configs(root)
    commit, problems = generator_commit(git)
    problems += _out_dir_problems(out_dir, root)
    if problems:
        raise GenerationError(problems)
    docs = generate_stage_d(args.session, freeze_commit=args.freeze_commit, generator_commit=commit, root=root,
                            configs=configs, stage_d_configs=stage_d_configs, git=git)
    targets = {out_dir / f"{pid}.json": doc for pid, doc in docs.items()}
    problems = [p for out in targets for p in _out_problems(out, root)]
    if problems:
        raise GenerationError(problems)
    write_all({out: render(doc) for out, doc in targets.items()})
    print(json.dumps({"profiles": [_summary(doc, out) for out, doc in targets.items()]}))
    return 0


def _main_stage_v(args, *, root: Path, git: Git) -> int:
    """044 產品對應: the session's §12 profile into --out-dir, or nothing when no unit that writes one holds (exit 0, and
    the summary says so); GenerationError: refused."""
    if args.out_dir is None:
        raise GenerationError([f"{args.validation_session}: a Stage V session writes its profile with --out-dir, never "
                               "--out (044 產品對應)"])
    out_dir = Path(args.out_dir)
    configs, stage_v_configs = _configs(root), _stage_v_configs(root)
    commit, problems = generator_commit(git)
    problems += _out_dir_problems(out_dir, root)
    if problems:
        raise GenerationError(problems)
    docs, holds = generate_stage_v(args.validation_session, freeze_commit=args.freeze_commit, generator_commit=commit,
                                   root=root, configs=configs, stage_v_configs=stage_v_configs, git=git)
    targets = {out_dir / f"{pid}.json": doc for pid, doc in docs.items()}
    problems = [p for out in targets for p in _out_problems(out, root)]
    if problems:
        raise GenerationError(problems)
    write_all({out: render(doc) for out, doc in targets.items()})
    summary = {"holds": holds, "profiles": [_stage_v_summary(doc, out) for out, doc in targets.items()]}
    if not targets:
        summary["written"] = NO_PROFILE
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
