#!/usr/bin/env python3
"""Stage D report (decision 039; rules in stage_d_analysis.py; PREREGISTRATION §9).

Inputs: one formal Stage D session directory (run_stage_d.py), the formal Core B uniform session and the formal
Advanced session (the CROSS_SESSION_SHIFT references, descriptive, as 030), configs/stage_d.toml and pins.toml. It
refuses (exit 2) and writes nothing, as capacity_report does for Stage C, when:
- the session is not formal, its stage is not "D", it is env-invalid, aborted, ran without the held GPU lock, inherited
  an engine variable, did not run one DEPLOYED_COMMIT or the pinned vLLM, or (D-A / D-B) never reached Phase C;
- it lies outside results/stage_d/ or under runs/, or --out is not <session>/analysis.json;
- freeze.stage_d is not true, stage_d.toml breaks 039, or the freeze evidence is missing;
- any sha256 differs: stage_d.toml and pins.toml (session, plan) against the current bytes, plan.json, the confirm
  plan, the datasets against stage_d.toml, every derived file and rows.jsonl against session.json;
- plan.json is not stage_d_plan.build_plan of these bytes (lambda_ref recomputed from the committed c1b files);
- a server argv / env is not its workload's (B0 or T), a bench argv is not stage_d_plan.bench_argv_for of its run,
  or a D-C gateway is not the one 039 runs (settings, verified profile);
- run validity (g): capacity_report.spill_problems on the rows (031 and its amendment);
- after stage_d_analysis.analyze: a replay mismatch or a failed sentinel gate.
Otherwise analysis.json is stage_d_analysis.analyze's body plus provenance, platform and the report's inputs.

Usage:
    python3 benchmarks/stage_d_report.py --session results/stage_d/<session> \\
        --uniform-session results/core_b/formal-20261009-coreb-uniform \\
        --advanced-session results/advanced/formal-20261011-adv --out results/stage_d/<session>/analysis.json
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
import run_capacity as rc
import run_stage_d as rd
import stage_d_analysis as sda
import stage_d_plan as sd
from run_core_a import sha256_json

ROOT = sd.ROOT
ANALYSIS_NAME = "analysis.json"


class ReportError(RuntimeError):
    pass


def load_configs() -> tuple:
    import tomllib

    cfg_raw, pins_raw = sd.load_stage_d_bytes(), cp.load_pins_bytes()
    return (tomllib.loads(cfg_raw.decode()), tomllib.loads(pins_raw.decode()), hashlib.sha256(cfg_raw).hexdigest(),
            hashlib.sha256(pins_raw).hexdigest())


# ---------------------------------------------------------------- refusals

def _session_problems(cap: dict, pins: dict, repo_root) -> list:
    s, plan = cap["session"], cap["plan"]
    problems = rd.output_problems(cap["dir"], repo_root=repo_root)
    if cap.get("meta") is not None or s.get("purpose") == "product-test":
        problems.append("the session directory is a product-test run (029 runs/), never research data")
    if s.get("formal") is not True or s.get("mode") != "formal":
        problems.append(f"the session is not formal (mode {s.get('mode')!r}): smokes are never reported (039)")
    for what, stage in (("session", s.get("stage")), ("plan", plan.get("stage"))):
        if stage != sd.STAGE:
            problems.append(f"the {what} stage is {stage!r}, not {sd.STAGE!r}")
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
    if plan.get("kind") != "verify" and cap.get("confirm_plan") is None:
        problems.append("no confirm_plan.json: the session never reached Phase C")
    return problems


def _config_problems(cap: dict, cfg: dict, pins: dict, repo_root) -> list:
    problems = []
    if (cfg.get("freeze") or {}).get("stage_d") is not True:
        problems.append("freeze.stage_d is not true in configs/stage_d.toml (039 凍結: no Stage D report before it)")
    problems += [f"stage_d.toml: {p}" for p in sd.stage_d_problems(cfg, pins)]
    problems += [f"freeze evidence: {p}" for p in sd.freeze_evidence_problems(cfg, repo_root)]
    return problems


def _sha_problems(cap: dict, *, stage_d_sha256: str, pins_sha256: str) -> list:
    s, plan, problems = cap["session"], cap["plan"], []
    for name, want, got in (
            ("stage_d.toml", stage_d_sha256, (("session", s.get("stage_d_sha256")),
                                              ("session's plan", s.get("plan_stage_d_sha256")),
                                              ("plan", plan.get("stage_d_sha256")))),
            ("pins.toml", pins_sha256, (("session", s.get("pins_sha256")),
                                        ("session's plan", s.get("plan_pins_sha256")),
                                        ("plan", plan.get("pins_sha256"))))):
        problems += [f"the {what} {name} sha256 {value!r} != configs/{name} {want!r}" for what, value in got
                     if value != want]
    if s.get("plan_sha256") != sha256_json(plan):
        problems.append("plan.json is not the plan the session ran (plan_sha256)")
    confirm = cap.get("confirm_plan")
    if confirm is not None and s.get("confirm_plan_sha256") != sha256_json(confirm):
        problems.append("confirm_plan.json is not the confirm plan whose sha256 was recorded before Phase C")
    for x in plan["workloads"].values():
        if x["mix"] in ("mixed", "var"):
            key, tier = f"{x['mix']}:{x['tier']}", str(x["tier"])
            pinned = (plan["mix_dataset_sha256"] if x["mix"] == "mixed" else plan["var_dataset_sha256"]).get(tier)
            if (s.get("dataset_sha256") or {}).get(key) != pinned:
                problems.append(f"dataset {key}: session sha256 {(s.get('dataset_sha256') or {}).get(key)!r} != "
                                f"pinned {pinned!r}")
    recorded, files = s.get("derived_sha256") or {}, cap["derived_files"]
    for tag in sorted(set(recorded) | set(files)):
        if recorded.get(tag) != files.get(tag):
            problems.append(f"derived file {tag}: missing, unrecorded or sha256 differs from session.json")
    if s.get("rows_sha256") != cap["rows_file_sha256"]:
        problems.append("rows.jsonl: sha256 differs from session.json rows_sha256")
    return problems


def _plan_problems(cap: dict, cfg: dict, pins: dict, *, stage_d_sha256: str, pins_sha256: str, repo_root) -> list:
    try:
        want = rd.expected_plan(cfg, pins, session_name=cap["plan"].get("session"), pins_sha256=pins_sha256,
                                stage_d_sha256=stage_d_sha256, formal=True, repo_root=repo_root)
    except (sd.PlanError, OSError, KeyError, ValueError) as exc:
        return [f"the plan cannot be rebuilt: {exc}"]
    if want != cap["plan"]:
        keys = sorted(k for k in set(want) | set(cap["plan"]) if want.get(k) != cap["plan"].get(k))
        return [f"plan.json is not stage_d_plan.build_plan of these stage_d.toml / pins bytes (differs in {keys})"]
    return []


def _argv_problems(cap: dict, pins: dict) -> list:
    """Every boot's server argv / env (its workload's B0 or T), every D-C gateway, every bench argv (039)."""
    s, plan, problems = cap["session"], cap["plan"], []
    server = s.get("server") or {}
    if not isinstance(server.get("port"), int) or not server.get("vllm_bin") or not s.get("result_dir"):
        return ["the session records no server port / vllm_bin / result_dir"]
    runs = s.get("runs") or {}
    want_env = None
    for b in s.get("boots") or []:
        rec = runs.get(b.get("tag"))
        if rec is None:
            problems.append(f"boot {b.get('tag')}: no server record in session.json")
            continue
        argv, env = sd.server_argv_for(pins, b["workload"], port=server["port"], vllm_bin=server["vllm_bin"],
                                       base_env=cr.CHECK_BASE_ENV)
        want_env = ra.recorded_server_env(env)
        if rec.get("argv") != argv:
            problems.append(f"boot {b.get('tag')}: server argv is not {b['workload']}'s (039: B0 or T)")
        if rec.get("env") != want_env:
            problems.append(f"boot {b.get('tag')}: server env is not the Core B one (engine / OTEL variables)")
        gw = rec.get("gateway")
        is_gw = b["workload"] in plan["workloads"] and plan["workloads"][b["workload"]].get("via") == "gateway"
        if is_gw and b.get("reached_first_bench"):
            want_settings = rd.gateway_settings(plan, pins, b["workload"])
            if not gw or gw.get("settings") != want_settings or gw.get("profile") != plan["verify"]["gateway_profile"]:
                problems.append(f"boot {b.get('tag')}: the gateway is not the one 039 runs (settings / profile)")
        elif gw is not None and not is_gw:
            problems.append(f"boot {b.get('tag')}: a gateway on a workload that does not use one")
    if want_env is not None and s.get("server_env") != want_env:
        problems.append("the session's server env is not the Core B one (engine / OTEL variables)")
    datasets = s.get("datasets") or {}
    for r in cap["rows"]:
        if r.get("argv") is None:
            continue
        x = sda.spec(plan, r["workload"])
        dataset = datasets.get(f"{x['mix']}:{x['tier']}") if x["mix"] in ("mixed", "var") else None
        kw = {"result_dir": s["result_dir"], "result_filename": f"{r['tag']}.bench.json", "dataset": dataset,
              "vllm_bin": server["vllm_bin"]}
        try:
            if plan["kind"] == "open" and r["phase"] in ("S", "C"):
                want = sd.bench_argv_for(pins, r["workload"], rate=r.get("rate"), N=r["N"],
                                         warmups=plan["open_loop"]["warmups"], **kw)
            else:
                want = sd.bench_argv_for(pins, r["workload"], C=r["C"], **kw)
        except (bp.PlanError, KeyError) as exc:
            problems.append(f"{r.get('tag')}: {exc}")
            continue
        if r["argv"] != want:
            problems.append(f"{r.get('tag')}: bench argv is not stage_d_plan.bench_argv_for of this run (039)")
        if plan["kind"] == "open" and r["phase"] in ("S", "C") and \
                r.get("rate") != plan["workloads"][r["workload"]]["points"][str(r["C"])]["rate"]:
            problems.append(f"{r.get('tag')}: rate {r.get('rate')!r} is not the grid point's")
    return problems


def refusal_problems(cap: dict, *, cfg: dict, pins: dict, stage_d_sha256: str, pins_sha256: str,
                     repo_root=ROOT) -> list:
    return (_session_problems(cap, pins, repo_root) + _config_problems(cap, cfg, pins, repo_root)
            + _sha_problems(cap, stage_d_sha256=stage_d_sha256, pins_sha256=pins_sha256)
            + _plan_problems(cap, cfg, pins, stage_d_sha256=stage_d_sha256, pins_sha256=pins_sha256,
                             repo_root=repo_root)
            + _argv_problems(cap, pins) + cr.spill_problems(cap["rows"]))


# ---------------------------------------------------------------- analysis.json

def provenance(cap: dict, *, stage_d_sha256: str, pins_sha256: str) -> dict:
    s, plan = cap["session"], cap["plan"]
    return {"session_id": s.get("session_id"), "session": plan.get("session"), "kind": plan.get("kind"),
            "deployed_commit": s.get("repo_commit"), "stage_d_sha256": stage_d_sha256, "pins_sha256": pins_sha256,
            "plan_sha256": s.get("plan_sha256"), "confirm_plan_sha256": s.get("confirm_plan_sha256"),
            "rows_sha256": s.get("rows_sha256"), "dataset_sha256": s.get("dataset_sha256"),
            "derived_from": plan.get("derived_from"), "c1b_session": plan.get("c1b_session")}


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


def report(cap: dict, *, cfg: dict, pins: dict, stage_d_sha256: str, pins_sha256: str, uniform: dict | None,
           advanced_runs: list | None, repo_root=ROOT) -> dict:
    problems = refusal_problems(cap, cfg=cfg, pins=pins, stage_d_sha256=stage_d_sha256, pins_sha256=pins_sha256,
                                repo_root=repo_root)
    if problems:
        raise ReportError("; ".join(problems))
    refs = ca.sentinel_references(uniform["rows"], advanced_runs) if uniform and advanced_runs else None
    try:
        analysis = sda.analyze({k: cap[k] for k in ("session", "plan", "confirm_plan", "rows")}, cap["requests"],
                               references=refs)
    except (ca.AnalysisError, sd.PlanError, KeyError) as exc:
        raise ReportError(f"the analysis failed: {exc}") from exc
    after = []
    if not analysis["replay"]["ok"]:
        after.append("replay mismatch: " + "; ".join(analysis["replay"]["problems"][:5]))
    gate = analysis["sentinel"]["gate"]
    if not gate["ok"]:
        after.append("the sentinel gate failed (drift, 030 8.6): " + "; ".join(gate["reasons"]))
    if after:
        raise ReportError("; ".join(after))
    analysis["provenance"] = provenance(cap, stage_d_sha256=stage_d_sha256, pins_sha256=pins_sha256)
    analysis["platform"] = platform(cap, pins)
    analysis["report"] = {"session_id": cap["session"].get("session_id"),
                          "uniform_session": (uniform or {}).get("session", {}).get("session_id"),
                          "advanced_references": bool(advanced_runs), "stage_d_sha256": stage_d_sha256,
                          "pins_sha256": pins_sha256}
    return analysis


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="the formal Stage D session directory (run_stage_d.py)")
    parser.add_argument("--uniform-session", required=True, help="the formal Core B uniform session")
    parser.add_argument("--advanced-session", required=True, help="the formal Advanced session")
    parser.add_argument("--out", required=True, help="results/stage_d/<session>/analysis.json")
    args = parser.parse_args(argv)
    try:
        problems = rd.output_problems(args.out, repo_root=ROOT)
        if Path(args.out).resolve() != Path(args.session).resolve() / ANALYSIS_NAME:
            problems.append(f"--out {args.out} is not <--session>/{ANALYSIS_NAME}")
        if problems:
            raise ReportError("; ".join(problems))
        cfg, pins, sd_sha, pins_sha = load_configs()
        rep = report(cr.load_session(args.session), cfg=cfg, pins=pins, stage_d_sha256=sd_sha, pins_sha256=pins_sha,
                     uniform=cr.load_uniform(args.uniform_session),
                     advanced_runs=cr.load_advanced_references(args.advanced_session), repo_root=ROOT)
    except (ReportError, OSError, ValueError) as exc:
        print(f"stage_d_report: refusing: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"session": rep["provenance"]["session_id"],
                      "hypotheses": {h: {u: d["verdict"] for u, d in units.items()}
                                     for h, units in rep["hypotheses"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
