#!/usr/bin/env python3
"""Stage V report (decision 044; rules in stage_v_analysis.py; PREREGISTRATION §12).

Inputs: one formal Stage V session directory (run_stage_v.py), the formal Core B uniform session and the formal
Advanced session (the CROSS_SESSION_SHIFT references, descriptive, as 030), configs/stage_v.toml, configs/stage_d.toml
and pins.toml. It refuses (exit 2) and writes nothing, as stage_d_report does for Stage D, when:
- the session is not formal, its stage is not "V", it is env-invalid, aborted, ran without the held GPU lock, inherited
  an engine variable, did not run one DEPLOYED_COMMIT or the pinned vLLM;
- it is not results/stage_v/formal-<id> or lies under runs/, or --out is not <session>/analysis.json;
- freeze.stage_v is not true, stage_v.toml breaks 044 (or stage_d.toml 039), the freeze evidence is missing, the
  pinned pacing is not stage_v_plan.pacing_params of the committed D5 file, or a point's committed profile is not its
  pinned bytes;
- the session ran another harness (harness_sha256), another request path than the frozen gateway_path_sha256, records
  no capserve_sha256 or other profile bytes than the plan's;
- any sha256 differs: stage_v.toml, stage_d.toml and pins.toml (session, plan) against the current bytes, plan.json,
  the datasets against the plan's pins, every derived file and rows.jsonl against session.json;
- plan.json is not stage_v_plan.build_plan of these bytes (formal);
- a server argv / env is not its point's (B0 or T), a bench argv is not stage_v_plan.bench_argv of its run (the
  sentinel's: stage_d_plan.bench_argv_for), a valid run lacks its argv; a gateway is not the point's (044: its rendered
  settings and their sha256, its profile, `capserve serve` with --allow-provisional exactly on a provisional point, the
  recorded /readyz equal to the point's frozen tuple), or a sentinel boot ran one; a valid V run's access-log join
  recorded a problem or not its N requests;
- another formal V session finished less than 2 h before this one started (run_stage_v.gap_problems on the committed
  results/stage_v/formal-*), or a formal-* / smoke-v-* directory has no finished session.json (run_stage_v.hours_used:
  a session killed hard, its hours and end unknown; 044 實作補記六); both only over the directories whose session.json
  started_at is not after this session's (one whose start cannot be read counts: fail closed), so a later or
  unrelated unfinished directory never blocks this report or its re-run (044 實作補記七; the runner checks them all);
- run validity (g): capacity_report.spill_problems on the rows (031 and its amendment);
- after stage_v_analysis.analyze: a replay mismatch or a failed sentinel gate.
Otherwise analysis.json is stage_v_analysis.analyze's body plus provenance (what the generator's §12 provenance reads:
the session, the deployed commit, the config, harness, request-path, capserve and profile sha256s, the freeze's mini),
platform and the report's inputs. It writes only <session>/analysis.json, and an existing one must be byte-identical to
the re-run (021 R3), else nothing is written.

Usage:
    python3 benchmarks/stage_v_report.py --session results/stage_v/<session> \\
        --uniform-session results/core_b/formal-20261009-coreb-uniform \\
        --advanced-session results/advanced/formal-20261011-adv --out results/stage_v/<session>/analysis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import capacity_analysis as ca
import capacity_plan as cp
import capacity_report as cr
import core_b_plan as bp
import run_adv as ra
import run_stage_v as rv
import stage_d_plan as sd
import stage_v_analysis as sva
import stage_v_plan as sv
from run_core_a import sha256_json

ROOT = sv.ROOT
ANALYSIS_NAME = "analysis.json"
FORMAL_SESSIONS = ("V-ADM", "V-CLS")


class ReportError(RuntimeError):
    pass


def load_configs() -> tuple:
    """(stage_v.toml, stage_d.toml, pins.toml, their sha256) as the repo holds them."""
    import tomllib

    raw = (sv.load_stage_v_bytes(), sd.load_stage_d_bytes(), cp.load_pins_bytes())
    return (*(tomllib.loads(x.decode()) for x in raw), *(hashlib.sha256(x).hexdigest() for x in raw))


# ---------------------------------------------------------------- refusals

def _session_problems(cap: dict, cfg: dict, pins: dict, repo_root) -> list:
    s, plan = cap["session"], cap["plan"]
    problems = rv.output_problems(cap["dir"], mode="formal", repo_root=repo_root)
    if cap.get("meta") is not None or s.get("purpose") == "product-test":
        problems.append("the session directory is a product-test run (029 runs/), never research data")
    if s.get("formal") is not True or s.get("mode") != "formal":
        problems.append(f"the session is not formal (mode {s.get('mode')!r}): smokes are never reported (044)")
    for what, stage in (("session", s.get("stage")), ("plan", plan.get("stage"))):
        if stage != sv.STAGE:
            problems.append(f"the {what} stage is {stage!r}, not {sv.STAGE!r}")
    if plan.get("session") not in FORMAL_SESSIONS or s.get("session") != plan.get("session"):
        problems.append(f"the session {s.get('session')!r} / plan {plan.get('session')!r} is not one of "
                        f"{FORMAL_SESSIONS}")
    if s.get("env_valid") is not True:
        problems.append(f"the session is env-invalid ({len(s.get('env_flags') or [])} env flags); rerun it (013)")
    if s.get("aborted"):
        problems.append(f"the session was aborted ({s['aborted']}); rerun it under a new id, never patch it (013)")
    if s.get("gpu_lock_held") is not True:
        problems.append("the session did not run under the held GPU lock (MACHINE_POLICY)")
    if s.get("inherited_unlisted"):
        problems.append(f"the session inherited engine variables {s['inherited_unlisted']} (030 8.6: none)")
    if s.get("repo_source") != "DEPLOYED_COMMIT" or not s.get("repo_commit"):
        problems.append(f"the session ran {s.get('repo_source')!r} code, not one DEPLOYED_COMMIT")
    if s.get("vllm") != pins["vllm"]["version"]:
        problems.append(f"the session's vLLM {s.get('vllm')!r} != pinned {pins['vllm']['version']}")
    if s.get("harness_sha256") != sv.harness_sha256():
        problems.append(f"the session ran harness_sha256 {s.get('harness_sha256')!r}, not this (frozen) harness's "
                        "(044 凍結)")
    frozen = (cfg.get("freeze") or {}).get("gateway_path_sha256")
    if s.get("gateway_path_sha256") != frozen or plan.get("gateway_path_sha256") != frozen:
        problems.append(f"the session ran request path gateway_path_sha256 {s.get('gateway_path_sha256')!r} (plan "
                        f"{plan.get('gateway_path_sha256')!r}), not the frozen {frozen!r} (044 凍結)")
    if not sv._sha(s.get("capserve_sha256")):
        problems.append("the session records no capserve_sha256: which gateway code it served is unknown (044)")
    points = plan.get("points") or {}
    if s.get("profile_sha256") != {p: x.get("profile_sha256") for p, x in points.items()}:
        problems.append("the session's profile_sha256 are not the plan's pinned profile bytes (044)")
    return problems


def _config_problems(cfg: dict, stage_d_cfg: dict, pins: dict, repo_root) -> list:
    problems = []
    if (cfg.get("freeze") or {}).get("stage_v") is not True:
        problems.append("freeze.stage_v is not true in configs/stage_v.toml (044 凍結: no Stage V report before it)")
    problems += [f"stage_v.toml: {p}" for p in sv.stage_v_problems(cfg, pins, stage_d_cfg)]
    problems += [f"stage_d.toml: {p}" for p in sd.stage_d_problems(stage_d_cfg, pins)]
    if not problems:
        problems += [f"freeze evidence: {p}" for p in sv.freeze_evidence_problems(cfg, repo_root)]
        problems += [f"stage_v.toml: {p}" for p in sv.pacing_problems(cfg, repo_root, pins)]
        problems += [f"profile: {p}" for p in sv.profile_problems(cfg, repo_root)]
    return problems


def _sha_problems(cap: dict, *, stage_v_sha256: str, stage_d_sha256: str, pins_sha256: str) -> list:
    s, plan, problems = cap["session"], cap["plan"], []
    for name, key, want in (("stage_v.toml", "stage_v_sha256", stage_v_sha256),
                            ("stage_d.toml", "stage_d_sha256", stage_d_sha256),
                            ("pins.toml", "pins_sha256", pins_sha256)):
        problems += [f"the {what} {name} sha256 {value!r} != configs/{name} {want!r}" for what, value in (
            ("session", s.get(key)), ("session's plan", s.get(f"plan_{key}")), ("plan", plan.get(key)))
                     if value != want]
    if s.get("plan_sha256") != sha256_json(plan):
        problems.append("plan.json is not the plan the session ran (plan_sha256)")
    pinned, got = plan.get("datasets") or {}, s.get("dataset_sha256") or {}
    for key in sorted(set(pinned) | set(got)):
        if got.get(key) != pinned.get(key):
            problems.append(f"dataset {key}: session sha256 {got.get(key)!r} != pinned {pinned.get(key)!r}")
    recorded, files = s.get("derived_sha256") or {}, cap["derived_files"]
    for tag in sorted(set(recorded) | set(files)):
        if recorded.get(tag) != files.get(tag):
            problems.append(f"derived file {tag}: missing, unrecorded or sha256 differs from session.json")
    if s.get("rows_sha256") != cap["rows_file_sha256"]:
        problems.append("rows.jsonl: sha256 differs from session.json rows_sha256")
    return problems


def _plan_problems(cap: dict, cfg: dict, stage_d_cfg: dict, pins: dict, *, stage_v_sha256: str, stage_d_sha256: str,
                   pins_sha256: str) -> list:
    try:
        want = sv.build_plan(cfg, pins, stage_d_cfg, session_name=cap["plan"].get("session"), pins_sha256=pins_sha256,
                             stage_v_sha256=stage_v_sha256, stage_d_sha256=stage_d_sha256, formal=True)
    except (bp.PlanError, OSError, KeyError, ValueError, TypeError) as exc:
        return [f"the plan cannot be rebuilt: {exc}"]
    if want != cap["plan"]:
        keys = sorted(k for k in set(want) | set(cap["plan"]) if want.get(k) != cap["plan"].get(k))
        return [f"plan.json is not stage_v_plan.build_plan of these stage_v.toml / stage_d.toml / pins bytes (differs "
                f"in {keys})"]
    return []


def _gateway_problems(tag: str, point: str, gw, *, s: dict, plan: dict, repo_root) -> list:
    """One V boot's gateway against the point (044): its settings, their rendered sha256 and its profile, its argv,
    its recorded /readyz."""
    x = plan["points"][point]
    if not isinstance(gw, dict):
        return [f"boot {tag}: no gateway record for a V boot (044)"]
    problems = []
    if (gw.get("settings"), gw.get("toml_sha256"), gw.get("profile")) != (x["gateway"], x["gateway_toml_sha256"],
                                                                            x["profile"]):
        problems.append(f"boot {tag}: the gateway settings / profile are not the point's (044)")
    want = rv.gateway_argv(s["gateway_python"], Path(s["result_dir"]) / f"{tag}.gateway.toml",
                           Path(s["root"]) / x["profile"], provisional=x["provisional"])
    if gw.get("argv") != want:
        problems.append(f"boot {tag}: gateway argv is not the point's `capserve serve` (044: --allow-provisional "
                        "exactly on a provisional point)")
    try:
        profile_id = json.loads((Path(repo_root) / x["profile"]).read_text())["profile_id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return problems + [f"boot {tag}: no profile id from {x['profile']} ({exc}): its /readyz cannot be checked"]
    if gw.get("readyz") != rv.readyz_want(plan, point, profile_id=profile_id,
                                          gateway_path_sha256=s.get("gateway_path_sha256")):
        problems.append(f"boot {tag}: its recorded /readyz is not the point's frozen tuple (044 boot)")
    return problems


def _argv_problems(cap: dict, pins: dict, repo_root) -> list:
    """Every boot's server argv / env (its point's B0 or T), every V boot's gateway, every bench argv, every valid V
    run's access-log join (044)."""
    s, plan, problems = cap["session"], cap["plan"], []
    server = s.get("server") or {}
    if not (isinstance(server.get("port"), int) and server.get("vllm_bin") and s.get("result_dir") and s.get("root")
            and s.get("gateway_python")):
        return ["the session records no server port / vllm_bin / result_dir / root / gateway_python"]
    runs, points = s.get("runs") or {}, plan["points"]
    want_env = None
    for b in s.get("boots") or []:
        tag, w = b.get("tag"), b.get("workload")
        rec = runs.get(tag)
        if rec is None:
            problems.append(f"boot {tag}: no server record in session.json")
            continue
        v = b.get("phase") == "V" and w in points
        argv, env = sd.server_argv_for(pins, points[w]["server_base"] if v else w, port=server["port"],
                                       vllm_bin=server["vllm_bin"], base_env=cr.CHECK_BASE_ENV)
        want_env = ra.recorded_server_env(env)
        if rec.get("argv") != argv:
            problems.append(f"boot {tag}: server argv is not {w}'s (044: its server_base, B0 or T)")
        if rec.get("env") != want_env:
            problems.append(f"boot {tag}: server env is not the Core B one (engine / OTEL variables)")
        has_valid = any(r.get("boot") == b.get("boot") and r.get("valid") is True for r in cap["rows"])
        if v and (b.get("reached_first_bench") or has_valid):
            problems += _gateway_problems(tag, w, rec.get("gateway"), s=s, plan=plan, repo_root=repo_root)
        elif not v and rec.get("gateway") is not None:
            problems.append(f"boot {tag}: a gateway on a boot that does not run one (the sentinel is direct)")
    if want_env is not None and s.get("server_env") != want_env:
        problems.append("the session's server env is not the Core B one (engine / OTEL variables)")
    for r in cap["rows"]:
        tag = r.get("tag")
        if r.get("argv") is None:
            if r.get("valid") is True:
                problems.append(f"{tag}: a valid run without its bench argv (044)")
            continue
        kw = {"result_dir": s["result_dir"], "result_filename": f"{tag}.bench.json", "vllm_bin": server["vllm_bin"]}
        try:
            if r.get("phase") == "V":
                want = sv.bench_argv(plan, pins, r["workload"], block=r["block"], datasets=s.get("datasets"), **kw)
            else:
                want = sd.bench_argv_for(pins, r["workload"], C=r["C"], **kw)
        except (bp.PlanError, KeyError) as exc:
            problems.append(f"{tag}: {exc}")
            continue
        if r["argv"] != want:
            problems.append(f"{tag}: bench argv is not stage_v_plan.bench_argv of this run (044)")
        if r.get("phase") == "V" and r.get("valid") is True:
            log = r.get("access_log") or {}
            if log.get("problems") != [] or log.get("joined") != r.get("N"):
                problems.append(f"{tag}: a valid run whose access log did not join its {r.get('N')} requests cleanly "
                                "(044 (h))")
    return problems


def refusal_problems(cap: dict, *, cfg: dict, stage_d_cfg: dict, pins: dict, stage_v_sha256: str,
                     stage_d_sha256: str, pins_sha256: str, repo_root=ROOT) -> list:
    config = _config_problems(cfg, stage_d_cfg, pins, repo_root)
    problems = _session_problems(cap, cfg, pins, repo_root) + config
    problems += _sha_problems(cap, stage_v_sha256=stage_v_sha256, stage_d_sha256=stage_d_sha256,
                              pins_sha256=pins_sha256)
    if not config:
        problems += _plan_problems(cap, cfg, stage_d_cfg, pins, stage_v_sha256=stage_v_sha256,
                                   stage_d_sha256=stage_d_sha256, pins_sha256=pins_sha256)
    problems += _argv_problems(cap, pins, repo_root) + cr.spill_problems(cap["rows"])
    started = cap["session"].get("started_at")
    if not isinstance(started, str):
        return problems + rv.hours_used(repo_root)[1] + [
            f"the session records no started_at ({started!r}): the 2 h gap cannot be checked (044)"]
    # 044 實作補記六: every session directory recorded its end; 實作補記七: only those that started by this session's
    # start (a later or unrelated unfinished one never blocks this report; one whose start is unreadable still does)
    try:
        hours = rv.hours_used(repo_root, not_after=started)[1]
        gap = rv.gap_problems(repo_root, started, session_id=cap["session"].get("session_id"), not_after=started)[1]
    except ValueError as exc:
        return problems + rv.hours_used(repo_root)[1] + [f"started_at {started!r}: {exc} (044: the 2 h gap)"]
    return problems + hours + gap


# ---------------------------------------------------------------- analysis.json

def provenance(cap: dict, cfg: dict, *, stage_v_sha256: str, stage_d_sha256: str, pins_sha256: str) -> dict:
    s, plan = cap["session"], cap["plan"]
    return {"session_id": s.get("session_id"), "session": plan.get("session"), "kind": plan.get("kind"),
            "deployed_commit": s.get("repo_commit"), "stage_v_sha256": stage_v_sha256,
            "stage_d_sha256": stage_d_sha256, "pins_sha256": pins_sha256, "plan_sha256": s.get("plan_sha256"),
            "rows_sha256": s.get("rows_sha256"), "dataset_sha256": s.get("dataset_sha256"),
            "harness_sha256": s.get("harness_sha256"), "gateway_path_sha256": s.get("gateway_path_sha256"),
            "capserve_sha256": s.get("capserve_sha256"), "profile_sha256": s.get("profile_sha256"),
            "freeze_mini": cfg["freeze"]["mini"], "started_at": s.get("started_at"),
            "finished_at": s.get("finished_at")}


def platform(cap: dict, pins: dict) -> dict:
    s, plan, pre = cap["session"], cap["plan"], cap.get("preflight") or {}
    gpu = (((pre.get("gpu") or {}).get("nvidia")) or [{}])[0]
    vllm = (pre.get("runtime") or {}).get("vllm") or {}
    runs, b = s.get("runs") or {}, pins["core_b"]
    argvs = {bt["workload"]: cr.server_argv_sha256(runs[bt["tag"]]["argv"]) for bt in s.get("boots") or []
             if bt.get("tag") in runs}
    return {"gpu": gpu.get("name"), "driver": gpu.get("driver_version"), "vllm": vllm.get("vllm"),
            "torch": vllm.get("torch"), "model": f"{b['repo']}@{b['revision']}", "server_argv_sha256": argvs,
            "env": s.get("server_env"), "K": plan["kv_cache_memory_bytes"], "U": plan["gpu_memory_utilization"],
            "num_gpu_blocks": plan["expected_blocks"], "block_size": plan["block_size"]}


def report(cap: dict, *, cfg: dict, stage_d_cfg: dict, pins: dict, stage_v_sha256: str, stage_d_sha256: str,
           pins_sha256: str, uniform: dict | None, advanced_runs: list | None, repo_root=ROOT) -> dict:
    shas = {"stage_v_sha256": stage_v_sha256, "stage_d_sha256": stage_d_sha256, "pins_sha256": pins_sha256}
    problems = refusal_problems(cap, cfg=cfg, stage_d_cfg=stage_d_cfg, pins=pins, repo_root=repo_root, **shas)
    if problems:
        raise ReportError("; ".join(problems))
    refs = ca.sentinel_references(uniform["rows"], advanced_runs) if uniform and advanced_runs else None
    try:
        analysis = sva.analyze({k: cap[k] for k in ("session", "plan", "rows")}, cap["requests"], references=refs)
    except (ca.AnalysisError, bp.PlanError, KeyError) as exc:
        raise ReportError(f"the analysis failed: {exc}") from exc
    after = []
    if not analysis["replay"]["ok"]:
        after.append("replay mismatch: " + "; ".join(analysis["replay"]["problems"][:5]))
    gate = analysis["sentinel"]["gate"]
    if not gate["ok"]:
        after.append("the sentinel gate failed (drift, 030 8.6): " + "; ".join(gate["reasons"]))
    if after:
        raise ReportError("; ".join(after))
    analysis["provenance"] = provenance(cap, cfg, **shas)
    analysis["platform"] = platform(cap, pins)
    analysis["report"] = {"session_id": cap["session"].get("session_id"),
                          "uniform_session": (uniform or {}).get("session", {}).get("session_id"),
                          "advanced_references": bool(advanced_runs), **shas}
    return analysis


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="the formal Stage V session directory (run_stage_v.py)")
    parser.add_argument("--uniform-session", required=True, help="the formal Core B uniform session")
    parser.add_argument("--advanced-session", required=True, help="the formal Advanced session")
    parser.add_argument("--out", required=True, help="results/stage_v/<session>/analysis.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        problems = rv.output_problems(args.session, mode="formal", repo_root=ROOT)
        if out.resolve() != Path(args.session).resolve() / ANALYSIS_NAME:
            problems.append(f"--out {args.out} is not <--session>/{ANALYSIS_NAME}")
        if problems:
            raise ReportError("; ".join(problems))
        cfg, sd_cfg, pins, sv_sha, sd_sha, pins_sha = load_configs()
        rep = report(cr.load_session(args.session), cfg=cfg, stage_d_cfg=sd_cfg, pins=pins, stage_v_sha256=sv_sha,
                     stage_d_sha256=sd_sha, pins_sha256=pins_sha, uniform=cr.load_uniform(args.uniform_session),
                     advanced_runs=cr.load_advanced_references(args.advanced_session), repo_root=ROOT)
        text = json.dumps(rep, indent=1) + "\n"
        if out.exists() and out.read_text() != text:
            raise ReportError(f"{args.out} exists and differs from this re-run (021 R3: a report re-run is "
                              "byte-identical); not overwritten")
    except (ReportError, OSError, ValueError) as exc:
        print(f"stage_v_report: refusing: {exc}", file=sys.stderr)
        return 2
    out.write_text(text)
    print(json.dumps({"session": rep["provenance"]["session_id"],
                      "units": {u: x["holds"] for u, x in rep["units"].items()},
                      "hypotheses": {h: x["verdict"] for h, x in rep["hypotheses"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
