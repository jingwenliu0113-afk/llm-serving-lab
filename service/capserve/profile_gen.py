"""python -m capserve.profile_gen --session results/capacity/<formal session> --freeze-commit <sha> --out <profile.json>
python -m capserve.profile_gen --session results/stage_d/<formal D-A session> --freeze-commit <sha> --out-dir <dir>

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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable

import capacity_analysis as ca
import capacity_plan as cplan
import capacity_report as cr
import core_b_plan as bp
import gpu_idle as gi
import run_capacity as rc
import run_stage_d as rd
import stage_d_analysis as sda
import stage_d_plan as sdplan
import stage_d_report as sr
from capserve import profile as cp

ROOT = Path(__file__).resolve().parents[2]
CAPACITY_TOML = ROOT / "configs" / "capacity.toml"
FREEZE_FILE = "configs/capacity.toml"
STAGE_D_TOML = ROOT / "configs" / "stage_d.toml"
STAGE_D_FREEZE_FILE = "configs/stage_d.toml"
STAGE_D_RESULTS = sdplan.RESULTS          # ("results", "stage_d"): a Stage D session's place (043 §1.1 dispatch)
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
# what each freeze guards, in the refusals (the Stage C wording unchanged)
_FREEZES = {"capacity": {"stage": "Stage C", "rule": "030 凍結", "data": "C1 data", "change": "030 8.9"},
            "stage_d": {"stage": "Stage D", "rule": "039 凍結", "data": "Stage D data", "change": "039 凍結, 030 8.9"}}
_HEX40 = re.compile(r"[0-9a-f]{40}")
NOT_FROZEN = ("[freeze] capacity is not true in {path} (a missing file or key is false). PREREGISTRATION §8 is not "
              "frozen, so no verified capacity exists yet (027 §4, 029); develop against a provisional profile with "
              "--allow-provisional.")
NOT_FROZEN_D = ("[freeze] stage_d is not true in {path} (a missing file or key is false). PREREGISTRATION §9 is not "
                "frozen, so no Stage D capacity exists (039, 043).")
NOT_D_A = ("039 / 042: D-B and D-C produce no profile (their facts are service/README.md's scope, 043 §5); the {what} "
           "kind is {kind!r}, not D-A's {want!r}")

Git = Callable[..., "bytes | None"]


class GenerationError(RuntimeError):
    def __init__(self, problems):
        self.problems = list(dict.fromkeys(problems))
        super().__init__("; ".join(self.problems))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capserve.profile_gen",
                                     description="verified capacity profiles from a frozen Stage C or D analysis (M2)")
    parser.add_argument("--session", required=True, help="results/capacity/<formal session> or "
                        "results/stage_d/<formal D-A session> (with its analysis.json)")
    parser.add_argument("--freeze-commit", required=True,
                        help="the commit that set configs/capacity.toml [freeze] capacity = true, or (Stage D) "
                             "configs/stage_d.toml [freeze] stage_d = true (40 hex; verified)")
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--out", help="Stage C: the verified profile to write (never overwritten)")
    out.add_argument("--out-dir", help="Stage D: the directory of the session's profiles, each <profile_id>.json "
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
                        f"[freeze] {freeze_key} = true")
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
        return [f"{freeze_file} was already frozen at {earlier[-1]}, before {commit}: an unfreeze and re-freeze is "
                f"not the preregistered freeze ({_FREEZES[freeze_key]['change']}: a rule change after the freeze needs "
                "its own decision)"]
    return []


def _frozen_bytes_problems(commit: str, *, deployed_commit, capacity_sha256: str, git: Git,
                           freeze_file: str = FREEZE_FILE, freeze_key: str = "capacity",
                           modules=STAGE_C_MODULES) -> list:
    problems, rule = [], _FREEZES[freeze_key]
    toml = git("show", f"{commit}:{freeze_file}")
    if toml is not None and _sha256(toml) != capacity_sha256:
        problems.append(f"{freeze_file} at the freeze commit {commit} (sha256 {_sha256(toml)}) is not the one the "
                        f"session ran and this generator reads ({capacity_sha256}): it changed after the freeze")
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


def _summary(doc: dict, out: Path) -> dict:
    return {"profile_id": doc["profile_id"], "out": str(out),
            "classes": {c["name"]: [c["status"], c["max_concurrency"], c["bound"]] for c in doc["classes"]}}


def main(argv: list | None = None, *, root=ROOT, git: Git | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(root)
    git = git or git_runner(root)
    try:
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


if __name__ == "__main__":
    raise SystemExit(main())
