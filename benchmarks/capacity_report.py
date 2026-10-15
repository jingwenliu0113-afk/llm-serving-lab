#!/usr/bin/env python3
"""Stage C (C1) report (decision 030 8.7; rules in capacity_analysis.py; PREREGISTRATION §8).

Inputs: the formal Stage C session directory (run_capacity.py: session.json, plan.json, confirm_plan.json, rows.jsonl,
derived/, env/preflight.json), the formal Core B uniform session the plan derives from (the plan is rebuilt from it),
optionally the formal Advanced session (the CROSS_SESSION_SHIFT references, descriptive), and configs/capacity.toml
and configs/pins.toml. It refuses (exit 2) and writes nothing when (030 8.7, with the session rules of 8.6):
- the session is not formal (session formal / mode), its stage or its plan's is not "C1", it is env-invalid, it was
  aborted (an aborted session reruns under a new id, never patched; 013), it did not run under the held GPU lock, it
  inherited an engine variable, it did not run one DEPLOYED_COMMIT, its vLLM is not pins [vllm].version, or it never
  reached Phase C (no confirm_plan.json);
- it lies outside results/capacity/ or under runs/, or its directory holds a product-test meta.json (029 runs/);
- --out is not <session>/analysis.json: the report writes that file only, never over a session's own files (a
  session is never overwritten, 030 8.4 / 8.7);
- freeze.capacity is not true, or capacity.toml breaks 030 (capacity_plan.capacity_problems), or the freeze evidence
  (passed FS-order / mini smokes, C0's result log: capacity_plan.freeze_evidence_problems) is missing, or the plan's SLO (or
  the secondary S1 / S3) is not capacity.toml's;
- any sha256 differs: capacity.toml (session, plan) and pins.toml (session, plan) against the current bytes, plan.json
  (plan_sha256), confirm_plan.json (the sha256 recorded before Phase C), the mixed datasets against capacity.toml
  mix_dataset_sha256, every derived/<tag>-requests.jsonl and rows.jsonl against session.json;
- plan.json is not capacity_plan.build_plan of the uniform session and these capacity.toml / pins bytes;
- a recorded server argv / env is not the Core B one with the Stage B values (capacity_plan.server_argv_for), or a
  bench argv is not core_b_plan.bench_argv (capacity_plan.bench_argv_for);
- after capacity_analysis.analyze: a replay mismatch (every search decision, K, n, skip and order recomputed from the
  rows) or a failed sentinel gate (drift > 3 % in output throughput or W-mean TTFT, or a sentinel invalid).
Otherwise analysis.json (schema_version 1) is capacity_analysis.analyze's body plus provenance (session, the
DEPLOYED_COMMIT, every sha256, c0_branch, n_near), platform (GPU, driver, vLLM, model@revision, server_argv_sha256
without host / port, the server env, K, U, blocks) and the report's inputs. capserve.profile_gen (029) is the only
consumer that turns it into a verified profile; this module never imports capserve.

Usage:
    python3 benchmarks/capacity_report.py --session results/capacity/<session> \\
        --uniform-session results/core_b/formal-20261009-coreb-uniform \\
        [--advanced-session results/advanced/formal-20261011-adv] --out results/capacity/<session>/analysis.json

Stdlib only, Python 3.10 compatible (tomllib is imported only by load_configs, as in adv_plan).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import capacity_analysis as ca
import capacity_plan as cp
import core_b_plan as bp
import run_adv as ra
import run_capacity as rc
from run_core_a import sha256_json

ROOT = cp.ROOT
CHECK_BASE_ENV = rc.CHECK_BASE_ENV   # the recorded server env does not depend on the inherited env (as adv_report)
MIXED_LEVELS = ("2048", "8192")
ADV_REFERENCE_RUNS = tuple(f"B0-2048-uniform-r{r}" for r in (1, 2, 3))   # 030 8.4: Advanced steady 996 ms
DERIVED_SUFFIX = "-requests.jsonl"
ANALYSIS_NAME = "analysis.json"                                        # 030 8.7: <session>/analysis.json


class ReportError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _optional_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


# ---------------------------------------------------------------- inputs

def load_session(path) -> dict:
    """A run_capacity session directory: its JSON files, rows, derived requests and the sha256 of what is on disk."""
    d = Path(path)
    rows_bytes = (d / "rows.jsonl").read_bytes()
    requests, files = {}, {}
    if (d / "derived").is_dir():
        for p in sorted((d / "derived").glob(f"*{DERIVED_SUFFIX}")):
            tag, data = p.name[:-len(DERIVED_SUFFIX)], p.read_bytes()
            files[tag] = _sha256_bytes(data)
            requests[tag] = [json.loads(x) for x in data.decode().splitlines() if x.strip()]
    return {"dir": d, "session": json.loads((d / "session.json").read_text()),
            "plan": json.loads((d / "plan.json").read_text()), "confirm_plan": _optional_json(d / "confirm_plan.json"),
            "rows": [json.loads(x) for x in rows_bytes.decode().splitlines() if x.strip()], "requests": requests,
            "derived_files": files, "rows_file_sha256": _sha256_bytes(rows_bytes),
            "preflight": _optional_json(d / "env" / "preflight.json"), "meta": _optional_json(d / "meta.json")}


def load_uniform(path) -> dict:
    d = Path(path)
    return {"session": json.loads((d / "session.json").read_text()), "plan": json.loads((d / "plan.json").read_text()),
            "rows": [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()]}


def load_advanced_references(path) -> list:
    """The Advanced B0 2K uniform runs in the fields capacity_analysis.sentinel_references reads (idx, cls, TTFT)."""
    out = []
    for name in ADV_REFERENCE_RUNS:
        p = Path(path) / "derived" / f"{name}{DERIVED_SUFFIX}"
        out.append([{"idx": x["idx"], "cls": x["cls"], "ttft_ms": x["client_ttft_ms"]}
                    for x in map(json.loads, p.read_text().splitlines()) if x])
    return out


def load_configs() -> tuple:
    """(capacity.toml, pins.toml, their sha256) as the repo holds them."""
    import tomllib

    cap_raw, pins_raw = cp.load_capacity_bytes(), cp.load_pins_bytes()
    return (tomllib.loads(cap_raw.decode()), tomllib.loads(pins_raw.decode()), _sha256_bytes(cap_raw),
            _sha256_bytes(pins_raw))


def server_argv_sha256(argv: list) -> str:
    """sha256 of a server argv without the binary path and without --host / --port and their values (030 8.7)."""
    out, skip = [], False
    for a in argv[1:]:
        if skip:
            skip = False
        elif a in ("--host", "--port"):
            skip = True
        else:
            out.append(a)
    return sha256_json(out)


# ---------------------------------------------------------------- refusals (030 8.7)

def _location_problems(cap: dict, repo_root) -> list:
    problems = rc.output_problems(cap["dir"], repo_root=repo_root)
    meta, s = cap.get("meta"), cap["session"]
    if meta is not None:
        problems.append(f"the session directory holds a meta.json (purpose {meta.get('purpose')!r}): a product-test "
                        "run of capserve (029 runs/), never research data")
    if s.get("purpose") == "product-test" or s.get("research_valid") is False:
        problems.append("the session is marked as a product-test run (029), never research data")
    return problems


def _session_problems(cap: dict, pins: dict) -> list:
    s, plan, problems = cap["session"], cap["plan"], []
    if s.get("formal") is not True:
        problems.append("the session is not formal (030 8.7)")
    if s.get("mode") != "formal":
        problems.append(f"the session mode is {s.get('mode')!r}, not formal (smokes are never reported, 030 8.6)")
    for what, stage in (("session", s.get("stage")), ("plan", plan.get("stage"))):
        if stage != cp.STAGE:
            problems.append(f"the {what} stage is {stage!r}, not {cp.STAGE!r}")
    if s.get("env_valid") is not True:
        problems.append(f"the session is env-invalid (env_valid={s.get('env_valid')!r}, "
                        f"{len(s.get('env_flags') or [])} env flags); rerun it (013)")
    if s.get("aborted"):
        problems.append(f"the session was aborted ({s['aborted']}); rerun it under a new id, never patch it "
                        "(030 8.4; 013)")
    if s.get("gpu_lock_held") is not True:
        problems.append("the session did not run under the held GPU lock (030 8.4; MACHINE_POLICY)")
    if s.get("inherited_unlisted"):
        problems.append(f"the session inherited engine variables {s['inherited_unlisted']} (030 8.6: none)")
    if s.get("repo_source") != "DEPLOYED_COMMIT" or not s.get("repo_commit"):
        problems.append(f"the session ran {s.get('repo_source')!r} code {s.get('repo_commit')!r}, not one "
                        "DEPLOYED_COMMIT (030 8.4)")
    if s.get("vllm") != pins["vllm"]["version"]:
        problems.append(f"the session's vLLM {s.get('vllm')!r} != pinned {pins['vllm']['version']} (030 8.6)")
    if cap.get("confirm_plan") is None:
        problems.append("no confirm_plan.json: the session never reached Phase C")
    return problems


def _config_problems(cap: dict, cfg: dict, pins: dict, repo_root) -> list:
    problems, plan = [], cap["plan"]
    if (cfg.get("freeze") or {}).get("capacity") is not True:
        problems.append("freeze.capacity is not true in configs/capacity.toml (030 凍結: no C1 report before it)")
    problems += [f"capacity.toml: {p}" for p in cp.capacity_problems(cfg, pins)]
    problems += [f"freeze evidence: {p}" for p in cp.freeze_evidence_problems(cfg, repo_root)]
    if plan.get("slo") != cfg.get("slo") or plan.get("secondary_slo") != cfg.get("secondary_slo"):
        problems.append(f"the plan's SLO {(plan.get('slo') or {}).get('id')!r} (or its secondary S1 / S3) differs from "
                        "capacity.toml [slo] / [secondary_slo] (030 8.7)")
    return problems


def _sha_problems(cap: dict, cfg: dict, *, capacity_sha256: str, pins_sha256: str) -> list:
    s, plan, problems = cap["session"], cap["plan"], []
    for name, want, got in (
            ("capacity.toml", capacity_sha256, (("session", s.get("capacity_sha256")),
                                                ("session's plan", s.get("plan_capacity_sha256")),
                                                ("plan", plan.get("capacity_sha256")))),
            ("pins.toml", pins_sha256, (("session", s.get("pins_sha256")),
                                        ("session's plan", s.get("plan_pins_sha256")),
                                        ("plan", plan.get("pins_sha256"))))):
        problems += [f"the {what} {name} sha256 {value!r} != configs/{name} {want!r}" for what, value in got
                     if value != want]
    if s.get("plan_sha256") != sha256_json(plan):
        problems.append("plan.json is not the plan the session ran (plan_sha256)")
    confirm = cap.get("confirm_plan")
    if confirm is not None and s.get("confirm_plan_sha256") != sha256_json(confirm):
        problems.append("confirm_plan.json is not the confirm plan whose sha256 was recorded before Phase C (030 8.4)")
    pinned = (cfg.get("capacity") or {}).get("mix_dataset_sha256") or {}
    got = s.get("mixed_dataset_sha256") or {}
    problems += [f"mixed dataset L={L}: session sha256 {got.get(L)!r} != capacity.toml {pinned.get(L)!r}"
                 for L in MIXED_LEVELS if got.get(L) is None or got.get(L) != pinned.get(L)]
    recorded, files = s.get("derived_sha256") or {}, cap["derived_files"]
    for tag in sorted(set(recorded) | set(files)):
        if tag not in files:
            problems.append(f"derived file {tag}{DERIVED_SUFFIX} recorded in session.json is missing")
        elif tag not in recorded:
            problems.append(f"derived file {tag}{DERIVED_SUFFIX} is not recorded in session.json")
        elif recorded[tag] != files[tag]:
            problems.append(f"derived file {tag}{DERIVED_SUFFIX}: sha256 differs from session.json")
    if s.get("rows_sha256") != cap["rows_file_sha256"]:
        problems.append("rows.jsonl: sha256 differs from session.json rows_sha256")
    return problems


def _plan_problems(cap: dict, cfg: dict, pins: dict, uniform: dict, *, capacity_sha256: str,
                   pins_sha256: str) -> list:
    """plan.json against capacity_plan.build_plan of the uniform session (its C_KV source) and the current bytes."""
    try:
        want = cp.build_plan(cfg, pins, session=uniform["session"], uniform_plan=uniform["plan"],
                             pins_sha256=pins_sha256, capacity_sha256=capacity_sha256, formal=True)
    except cp.PlanError as exc:
        return [f"the plan cannot be rebuilt from the uniform session: {exc}"]
    if want != cap["plan"]:
        keys = sorted(k for k in set(want) | set(cap["plan"]) if want.get(k) != cap["plan"].get(k))
        return [f"plan.json is not capacity_plan.build_plan of uniform session "
                f"{uniform['session'].get('session_id')!r} and these capacity.toml / pins bytes (differs in {keys})"]
    return []


def _argv_problems(cap: dict, pins: dict) -> list:
    """Every boot's server argv / env and every bench argv: the Core B ones with the Stage B values (030 8.1)."""
    s, problems = cap["session"], []
    server, sb = s.get("server") or {}, pins.get("stage_b") or {}
    for k in ("gdn_prefill_backend", "gpu_memory_utilization"):
        if server.get(k) != sb.get(k):
            problems.append(f"server {k} {server.get(k)!r} != pinned [stage_b] {sb.get(k)!r}")
    if not isinstance(server.get("port"), int) or not server.get("vllm_bin"):
        problems.append("the session records no server port / vllm_bin")
    if problems:
        return problems
    argv, env = cp.server_argv_for(pins, port=server["port"], vllm_bin=server["vllm_bin"], base_env=CHECK_BASE_ENV)
    want_env, runs = ra.recorded_server_env(env), s.get("runs") or {}
    if s.get("server_env") != want_env:
        problems.append("the session's server env is not the Core B one (engine / OTEL variables)")
    for b in s.get("boots") or []:
        rec = runs.get(b.get("tag"))
        if rec is None:
            problems.append(f"boot {b.get('tag')}: no server record in session.json")
            continue
        if rec.get("argv") != argv:
            problems.append(f"boot {b.get('tag')}: server argv is not the Core B argv with the Stage B values "
                            "(030 8.1)")
        if rec.get("env") != want_env:
            problems.append(f"boot {b.get('tag')}: server env is not the Core B one (engine / OTEL variables)")
    datasets = s.get("mixed_datasets") or {}
    for r in cap["rows"]:
        if r.get("argv") is None:
            continue
        try:
            want = cp.bench_argv_for(pins, workload=r["workload"], C=r["C"], result_dir=s.get("result_dir"),
                                     result_filename=f"{r['tag']}.bench.json", port=server["port"],
                                     mixed_dataset=datasets.get(str(r.get("L"))) if r.get("mix") == "mixed" else None,
                                     vllm_bin=server["vllm_bin"])
        except (bp.PlanError, KeyError) as exc:
            problems.append(f"{r.get('tag')}: {exc}")
            continue
        if r["argv"] != want:
            problems.append(f"{r.get('tag')}: bench argv is not core_b_plan.bench_argv (030 8.1)")
    return problems


def refusal_problems(cap: dict, *, cfg: dict, pins: dict, capacity_sha256: str, pins_sha256: str, uniform: dict,
                     repo_root=ROOT) -> list:
    """Every refusal decidable before the analysis; [] when capacity_analysis.analyze may run."""
    return (_location_problems(cap, repo_root) + _session_problems(cap, pins) + _config_problems(cap, cfg, pins, repo_root)
            + _sha_problems(cap, cfg, capacity_sha256=capacity_sha256, pins_sha256=pins_sha256)
            + _plan_problems(cap, cfg, pins, uniform, capacity_sha256=capacity_sha256, pins_sha256=pins_sha256)
            + _argv_problems(cap, pins))


# ---------------------------------------------------------------- analysis.json

def provenance(cap: dict, *, capacity_sha256: str, pins_sha256: str) -> dict:
    s, plan = cap["session"], cap["plan"]
    return {"session_id": s.get("session_id"), "deployed_commit": s.get("repo_commit"),
            "capacity_sha256": capacity_sha256, "pins_sha256": pins_sha256, "plan_sha256": s.get("plan_sha256"),
            "confirm_plan_sha256": s.get("confirm_plan_sha256"), "rows_sha256": s.get("rows_sha256"),
            "c0_branch": plan.get("c0_branch"), "n_near": {w: x["n_near"] for w, x in plan["workloads"].items()},
            "derived_from": plan.get("derived_from"), "mixed_dataset_sha256": s.get("mixed_dataset_sha256")}


def platform(cap: dict, pins: dict) -> dict:
    s, plan, pre = cap["session"], cap["plan"], cap.get("preflight") or {}
    gpu = (((pre.get("gpu") or {}).get("nvidia")) or [{}])[0]
    vllm = (pre.get("runtime") or {}).get("vllm") or {}
    boots = s.get("boots") or []
    argv = ((s.get("runs") or {}).get(boots[0].get("tag")) or {}).get("argv") if boots else None
    b = pins["core_b"]
    return {"gpu": gpu.get("name"), "driver": gpu.get("driver_version"), "vllm": vllm.get("vllm"),
            "torch": vllm.get("torch"), "model": f"{b['repo']}@{b['revision']}",
            "server_argv_sha256": server_argv_sha256(argv) if argv else None, "env": s.get("server_env"),
            "K": plan["kv_cache_memory_bytes"], "U": plan["gpu_memory_utilization"],
            "num_gpu_blocks": plan["expected_blocks"], "block_size": plan["block_size"]}


def report(cap: dict, *, cfg: dict, pins: dict, capacity_sha256: str, pins_sha256: str, uniform: dict,
           advanced_runs: list | None = None, repo_root=ROOT) -> dict:
    """analysis.json of one formal session, or ReportError with every refusal."""
    problems = refusal_problems(cap, cfg=cfg, pins=pins, capacity_sha256=capacity_sha256, pins_sha256=pins_sha256,
                                uniform=uniform, repo_root=repo_root)
    if problems:
        raise ReportError("; ".join(problems))
    refs = ca.sentinel_references(uniform["rows"], advanced_runs) if advanced_runs else None
    try:
        analysis = ca.analyze({k: cap[k] for k in ("session", "plan", "confirm_plan", "rows")}, cap["requests"],
                              references=refs)
    except (ca.AnalysisError, cp.PlanError) as exc:
        raise ReportError(f"the analysis failed: {exc}") from exc
    after = []
    if not analysis["replay"]["ok"]:
        after.append("replay mismatch (030 8.6: the rows do not replay to what the session ran): "
                     + "; ".join(analysis["replay"]["problems"][:5]))
    gate = analysis["sentinel"]["gate"]
    if not gate["ok"]:
        after.append("the sentinel gate failed (drift, 030 8.6): " + "; ".join(gate["reasons"]))
    if after:
        raise ReportError("; ".join(after))
    analysis["provenance"] = provenance(cap, capacity_sha256=capacity_sha256, pins_sha256=pins_sha256)
    analysis["platform"] = platform(cap, pins)
    analysis["report"] = {"session_id": cap["session"].get("session_id"),
                          "uniform_session": uniform["session"].get("session_id"),
                          "advanced_references": bool(advanced_runs), "capacity_sha256": capacity_sha256,
                          "pins_sha256": pins_sha256}
    return analysis


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="the formal Stage C session directory (run_capacity.py)")
    parser.add_argument("--uniform-session", required=True, help="the formal Core B uniform session (derived_from)")
    parser.add_argument("--advanced-session", help="the formal Advanced session (CROSS_SESSION_SHIFT, descriptive)")
    parser.add_argument("--out", required=True, help="results/capacity/<session>/analysis.json")
    args = parser.parse_args(argv)
    try:
        problems = rc.output_problems(args.out, repo_root=ROOT)
        if Path(args.out).resolve() != Path(args.session).resolve() / ANALYSIS_NAME:
            problems.append(f"--out {args.out} is not <--session>/{ANALYSIS_NAME}: the report writes only that file, "
                            "never over a session's own files (030 8.4 / 8.7)")
        if problems:
            raise ReportError("; ".join(problems))
        cfg, pins, cap_sha, pins_sha = load_configs()
        rep = report(load_session(args.session), cfg=cfg, pins=pins, capacity_sha256=cap_sha, pins_sha256=pins_sha,
                     uniform=load_uniform(args.uniform_session), repo_root=ROOT,
                     advanced_runs=load_advanced_references(args.advanced_session) if args.advanced_session else None)
    except (ReportError, OSError, ValueError) as exc:
        print(f"capacity_report: refusing: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    wl = {w: {k: [x[k]["value"], x[k]["status"]] for k in ("c_slo", "c_prod")} | {"flags": x["flags"]}
          for w, x in rep["workloads"].items()}
    print(json.dumps({"session": rep["provenance"]["session_id"], "workloads": wl,
                      "hypotheses": {h: {u: d["verdict"] for u, d in units.items()}
                                     for h, units in rep["hypotheses"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
