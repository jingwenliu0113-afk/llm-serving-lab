"""python -m capserve.profile_gen --session results/capacity/<formal session> --freeze-commit <sha> --out <profile.json>

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
from capserve import profile as cp

ROOT = Path(__file__).resolve().parents[2]
CAPACITY_TOML = ROOT / "configs" / "capacity.toml"
FREEZE_FILE = "configs/capacity.toml"
ADVANCED_SESSION = "formal-20261011-adv"   # the CROSS_SESSION_SHIFT references capacity_report may have used (8.4)
# frozen with capacity.toml (030 凍結; tests/test_capacity_isolation CAPACITY_MODULES): their bytes at the freeze commit
# are the ones the session ran and the ones that re-run its analysis here
STAGE_C_MODULES = (cplan, ca, rc, cr, gi)
_HEX40 = re.compile(r"[0-9a-f]{40}")
NOT_FROZEN = ("[freeze] capacity is not true in {path} (a missing file or key is false). PREREGISTRATION §8 is not "
              "frozen, so no verified capacity exists yet (027 §4, 029); develop against a provisional profile with "
              "--allow-provisional.")

Git = Callable[..., "bytes | None"]


class GenerationError(RuntimeError):
    def __init__(self, problems):
        self.problems = list(dict.fromkeys(problems))
        super().__init__("; ".join(self.problems))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capserve.profile_gen",
                                     description="verified capacity profile from a frozen Stage C analysis (M2)")
    parser.add_argument("--session", required=True, help="results/capacity/<formal session> (with its analysis.json)")
    parser.add_argument("--freeze-commit", required=True,
                        help="the commit that set configs/capacity.toml [freeze] capacity = true (40 hex; verified)")
    parser.add_argument("--out", required=True, help="the verified profile to write (never overwritten)")
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


def frozen(cfg: dict) -> bool:
    """[freeze] capacity is exactly true."""
    return (cfg.get("freeze") or {}).get("capacity") is True


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


def _frozen_at(git: Git, rev: str):
    """Whether configs/capacity.toml at rev has [freeze] capacity = true; None when git cannot show a readable one."""
    data = git("show", f"{rev}:{FREEZE_FILE}")
    if data is None:
        return None
    try:
        return frozen(tomllib.loads(data.decode()))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None


def freeze_commit_problems(commit, *, deployed_commit, capacity_sha256: str, git: Git) -> list:
    """[] when commit is the full sha of the commit that first set [freeze] capacity = true (frozen there, never before
    in its history), an ancestor of (or equal to) the session's DEPLOYED_COMMIT, and its frozen bytes are the ones that
    ran: capacity.toml there has sha256 capacity_sha256 (what the session recorded and this generator reads), and every
    Stage C module there is the one at DEPLOYED_COMMIT and the file this generator imports."""
    if not (isinstance(commit, str) and _HEX40.fullmatch(commit)):
        return [f"--freeze-commit {commit!r} is not a full commit (40 lowercase hex)"]
    problems, at = [], _frozen_at(git, commit)
    if at is None:
        problems.append(f"{commit} has no readable {FREEZE_FILE} (not a commit of this repository?)")
    elif not at:
        problems.append(f"{FREEZE_FILE} at {commit} has no [freeze] capacity = true: not the freeze commit")
    elif _frozen_at(git, f"{commit}^"):
        problems.append(f"{FREEZE_FILE} was already frozen at {commit}^: {commit} is not the commit that set "
                        "[freeze] capacity = true")
    else:
        problems += _earlier_freeze_problems(commit, git)
    if not isinstance(deployed_commit, str) or git("merge-base", "--is-ancestor", commit, deployed_commit) is None:
        problems.append(f"{commit} is not an ancestor of the session's DEPLOYED_COMMIT {deployed_commit!r}: the "
                        "session did not run the frozen code (030 凍結: C1 data only after the freeze)")
    if at is None:
        return problems
    return problems + _frozen_bytes_problems(commit, deployed_commit=deployed_commit,
                                             capacity_sha256=capacity_sha256, git=git)


def _earlier_freeze_problems(commit: str, git: Git) -> list:
    """The freeze is the first commit that set [freeze] capacity = true: a re-freeze after an unfreeze is a rule change
    after the freeze (030 8.9: its own decision and a new session), never the preregistered freeze."""
    history = git("rev-list", "--full-history", commit, "--", FREEZE_FILE)
    if history is None:
        return [f"cannot list the history of {FREEZE_FILE} up to {commit} (git rev-list failed): cannot tell whether "
                "it is the first freeze"]
    earlier = [c for c in history.decode().split() if c != commit and _frozen_at(git, c)]
    if earlier:
        return [f"{FREEZE_FILE} was already frozen at {earlier[-1]}, before {commit}: an unfreeze and re-freeze is "
                "not the preregistered freeze (030 8.9: a rule change after the freeze needs its own decision)"]
    return []


def _frozen_bytes_problems(commit: str, *, deployed_commit, capacity_sha256: str, git: Git) -> list:
    problems = []
    toml = git("show", f"{commit}:{FREEZE_FILE}")
    if toml is not None and _sha256(toml) != capacity_sha256:
        problems.append(f"{FREEZE_FILE} at the freeze commit {commit} (sha256 {_sha256(toml)}) is not the one the "
                        f"session ran and this generator reads ({capacity_sha256}): it changed after the freeze")
    for module in STAGE_C_MODULES:
        path = f"benchmarks/{module.__name__}.py"
        frozen_bytes = git("show", f"{commit}:{path}")
        if frozen_bytes is None:
            problems.append(f"no {path} at the freeze commit {commit}")
            continue
        if isinstance(deployed_commit, str):
            ran = git("show", f"{deployed_commit}:{path}")
            if ran is None or _sha256(ran) != _sha256(frozen_bytes):
                problems.append(f"{path} at the session's DEPLOYED_COMMIT {deployed_commit} is not the frozen one at "
                                f"{commit}: the session did not run the frozen Stage C code (030 凍結)")
        try:
            imported = Path(module.__file__).read_bytes()
        except OSError as exc:
            imported = None
            problems.append(f"cannot read {module.__file__} ({exc})")
        if imported is not None and _sha256(imported) != _sha256(frozen_bytes):
            problems.append(f"{path} imported by this generator ({module.__file__}) is not the frozen one at {commit}: "
                            "the analysis would not be re-run by the frozen Stage C modules (029)")
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


def unavailable_reason(w: str, x: dict) -> str:
    status, K = x["c_prod"]["status"], x["K"]
    point = {C: x["points"][str(C)]["class"] for C in K}
    if status == "UNSTABLE":
        why = f"its lowest confirmation point C {K[0]} is above 1 and {point[K[0]]} (C_PROD reads MARGINAL as FAIL)"
    elif not K:
        why = "it has no confirmation points (its Phase S boots failed before a first bench)"
    else:
        why = "its confirmation points are not testable (" + ", ".join(f"C {C} {k}" for C, k in point.items()) + ")"
    return (f"Stage C {w}: C_PROD is {status}: {why}; no admission limit is verified, so the class is never served "
            "(030 8.7: never a number, never a fallback)")


def class_entry(w: str, x: dict) -> dict:
    """One class of the verified profile from analysis.json's workload w (030 8.7)."""
    L, mix = x["L"], x["mix"]
    est = {"value": x["c_prod"]["value"], "status": x["c_prod"]["status"]}
    reason = None
    if est["status"] in cp.BOUNDS:
        limit, bound = est["value"], cp.BOUNDS[est["status"]]
    elif est["status"] == cp.REFUSE_STATUS:
        limit, bound = 0, "exact"
    elif est["status"] in cp.UNAVAILABLE_STATUSES:
        limit, bound, reason = None, None, unavailable_reason(w, x)
    else:
        raise GenerationError([f"{w}: unknown C_PROD status {est['status']!r}"])
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
    """capacity_report.main's serialization of analysis.json."""
    return (json.dumps(rep, indent=1) + "\n").encode()


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


def main(argv: list | None = None, *, root=ROOT, git: Git | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(root)
    git = git or git_runner(root)
    out = Path(args.out)
    try:
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
    print(json.dumps({"profile_id": doc["profile_id"], "out": str(out),
                      "classes": {c["name"]: [c["status"], c["max_concurrency"], c["bound"]] for c in doc["classes"]}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
