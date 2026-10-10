#!/usr/bin/env python3
"""Advanced analysis report (decision 024 A14; rules in adv_analysis.py; PREREGISTRATION §7).

Inputs: the formal Advanced session (run_adv.py: session.json, plan.json, rows.jsonl), the formal Core uniform
session it derives from (the C check), optionally the formal Core mixed session (the descriptive B0 / Core
ratio of A4), and configs/pins.toml. It refuses, and writes nothing, when (A14):
- the Advanced session is not formal, is env-invalid, or was aborted, or its rows do not cover the plan's boots
  one to one (an incomplete session is rerun, never patched; 013);
- freeze.advanced is not true, or pins [advanced] break the 024 rules (adv_plan.pins_problems);
- the plan's derived_from is not formal-20261009-coreb-uniform, the uniform session given is another one or is
  not formal and env-valid, or the plan is not adv_plan's seeded 60-boot plan of these pins (run_adv.plan_problems);
- C (plan and rows) is not the C_knee recomputed here from the uniform session (core_b_analysis.knee), 25 / 8;
- a recorded server argv or env is outside the condition table (adv_plan.server_argv_for with the session's
  pinned server values; OTEL_* and engine variables included), or a bench argv is not adv_plan.bench_argv_for;
- the recorded mixed dataset sha256 is not the pinned one;
- plan.json is not the plan the session ran (plan_sha256), or the plan's or the session's pins sha256 is not the
  sha256 of configs/pins.toml.
Otherwise adv_analysis.analyze produces analysis.json: the per-L verdicts and interpretation rows, the
per-(condition, L, mix) descriptive table (levels[L].descriptive.cells: mean / SE over the valid restarts of the
A13 histogram queue / prefill means and 1 Hz shares, the A8 >= P95 tail and the client descriptives, beside
throughput, long-class P95, preemptions and KV peak; never read by a verdict) and every run (valid, invalid,
timed out, skipped), plus the inputs' identities under "report". stdout gets each level's row, conclusion and
incomplete flag and the cross-L outcome.

Usage:
    python3 benchmarks/adv_report.py --session results/advanced/<session> \\
        --uniform-session results/core_b/formal-20261009-coreb-uniform \\
        [--core-mixed-session results/core_b/formal-20261010-coreb-mixed2] \\
        --out results/advanced/<session>/analysis.json

Stdlib only, Python 3.10 compatible (tomllib is imported only by the CLI, as in adv_plan).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import adv_analysis as aa
import adv_plan as ap
import core_b_analysis as ba
import core_b_plan as bp
import run_adv as ra
from run_core_a import sha256_json

PINS_PATH = Path(__file__).resolve().parent.parent / "configs" / "pins.toml"
CHECK_BASE_ENV = {"PATH": "/usr/bin"}  # the recorded server env does not depend on the inherited env


class ReportError(RuntimeError):
    pass


def _session_problems(adv: dict) -> list:
    s, problems = adv["session"], []
    if s.get("formal") is not True:
        problems.append("the Advanced session is not formal")
    if s.get("env_valid") is not True:
        problems.append(f"the Advanced session is env-invalid (env_valid={s.get('env_valid')!r}); rerun it (013)")
    if s.get("aborted"):
        problems.append(f"the Advanced session was aborted ({s['aborted']}); rerun it, never patch it (013)")
    plan_keys = [(b.get("cond"), b.get("L"), b.get("mix"), b.get("restart")) for b in adv["plan"].get("boots") or []]
    row_keys = [(r.get("cond"), r.get("L"), r.get("mix"), r.get("restart")) for r in adv["rows"]]
    if sorted(row_keys, key=str) != sorted(plan_keys, key=str):
        missing, extra = set(plan_keys) - set(row_keys), set(row_keys) - set(plan_keys)
        problems.append(f"the rows do not cover the plan's {len(plan_keys)} boots one to one ({len(missing)} "
                        f"missing, {len(extra)} outside the plan, {len(row_keys) - len(set(row_keys))} duplicated)")
    return problems


def _pins_problems(adv: dict, pins: dict, pins_sha256: str) -> list:
    s, plan, problems = adv["session"], adv["plan"], []
    if (pins.get("freeze") or {}).get("advanced") is not True:
        problems.append("freeze.advanced is not true (024 A14)")
    problems += [f"pins [advanced]: {p}" for p in ap.pins_problems(pins)]
    if s.get("plan_sha256") != sha256_json(plan):
        problems.append("plan.json is not the plan the session ran (plan_sha256)")
    if plan.get("pins_sha256") != pins_sha256:
        problems.append(f"the plan was built from pins {plan.get('pins_sha256')!r}, not configs/pins.toml "
                        f"{pins_sha256!r}")
    if s.get("pins_sha256") != pins_sha256:
        problems.append(f"the session ran with pins {s.get('pins_sha256')!r}, not configs/pins.toml {pins_sha256!r}")
    problems += [f"plan: {p}" for p in ra.plan_problems(plan, pins, formal=True)]
    return problems


def _c_problems(adv: dict, uniform: dict) -> list:
    plan, us, problems = adv["plan"], uniform["session"], []
    if plan.get("derived_from") != ap.DERIVED_FROM:
        problems.append(f"the plan's derived_from {plan.get('derived_from')!r} != {ap.DERIVED_FROM!r} (024)")
    if us.get("session_id") != plan.get("derived_from"):
        problems.append(f"the uniform session given is {us.get('session_id')!r}, not the plan's derived_from "
                        f"{plan.get('derived_from')!r}")
    if us.get("formal") is not True or us.get("env_valid") is not True:
        problems.append(f"the uniform session is not formal and env-valid (formal={us.get('formal')!r}, "
                        f"env_valid={us.get('env_valid')!r})")
    knees = {}
    for L in ap.LEVELS:
        grid = (uniform["plan"].get("grids") or {}).get(str(L))
        knees[L] = ba.knee(uniform["rows"], L, grid)["c_knee"] if grid else None
    plan_c = {int(L): C for L, C in (plan.get("c") or {}).items()}
    if plan_c != knees or knees != ap.C_KNEE:
        problems.append(f"plan C {plan_c}, the C_knee recomputed from the uniform session {knees} and the frozen "
                        f"C {ap.C_KNEE} differ (024)")
    for L, C in sorted({(r.get("L"), r.get("C")) for r in adv["rows"] if r.get("C") != plan_c.get(r.get("L"))},
                       key=str):
        problems.append(f"rows at L = {L} ran at C = {C}, not the plan's C {plan_c.get(L)}")
    return problems


def _dataset_problems(adv: dict, pins: dict) -> list:
    pinned = pins["core_b"].get("mix_dataset_sha256") or {}
    got = adv["session"].get("mixed_dataset_sha256") or {}
    return [f"mixed dataset L={L}: session sha256 {got.get(str(L))!r} != pinned {pinned.get(str(L))!r}"
            for L in ap.LEVELS if got.get(str(L)) is None or got.get(str(L)) != pinned.get(str(L))]


def _argv_problems(adv: dict, pins: dict) -> list:
    """Every started boot's server argv / env and every bench argv against the condition table."""
    s = adv["session"]
    server, sb, problems = s.get("server") or {}, pins.get("stage_b") or {}, []
    for k in ("gdn_prefill_backend", "gpu_memory_utilization"):
        if server.get(k) != sb.get(k):
            problems.append(f"server {k} {server.get(k)!r} != pinned [stage_b] {sb.get(k)!r}")
    if not isinstance(server.get("port"), int) or not server.get("vllm_bin"):
        problems.append("the session records no server port / vllm_bin")
    if problems:
        return problems
    kw = {"gdn_prefill_backend": server["gdn_prefill_backend"],
          "gpu_memory_utilization": server["gpu_memory_utilization"], "port": server["port"],
          "vllm_bin": server["vllm_bin"]}
    runs, datasets = s.get("runs") or {}, s.get("mixed_datasets") or {}
    for r in adv["rows"]:
        if r.get("skipped"):
            continue
        tag = ra.boot_tag(r)
        rec = runs.get(tag)
        if rec is None:
            problems.append(f"{tag}: no server record in session.json")
            continue
        try:
            argv, env = ap.server_argv_for(r["cond"], pins, base_env=CHECK_BASE_ENV, **kw)
            bench = ap.bench_argv_for(pins, cond=r["cond"], L=r["L"], C=r["C"], mix=r["mix"], restart=r["restart"],
                                      result_dir=s.get("result_dir"), result_filename=f"{tag}.bench.json",
                                      port=kw["port"], vllm_bin=kw["vllm_bin"],
                                      mixed_dataset=datasets.get(str(r["L"])) if r["mix"] == "mixed" else None)
        except bp.PlanError as exc:
            problems.append(f"{tag}: {exc}")
            continue
        if rec.get("argv") != argv:
            problems.append(f"{tag}: server argv is not condition {r['cond']}'s (024 condition table)")
        if rec.get("env") != ra.recorded_server_env(env):
            problems.append(f"{tag}: server env is not the condition table's (OTEL_* / engine variables)")
        if r.get("argv") is not None and r["argv"] != bench:
            problems.append(f"{tag}: bench argv is not Core's plus the request-id prefix")
    return problems


def _core_mixed_problems(core_mixed: dict | None) -> list:
    if core_mixed is None:
        return []
    s, problems = core_mixed["session"], []
    if s.get("formal") is not True or s.get("env_valid") is not True:
        problems.append("the Core mixed session is not formal and env-valid")
    if core_mixed["plan"].get("derived_from") != ap.DERIVED_FROM:
        problems.append(f"the Core mixed session derives from {core_mixed['plan'].get('derived_from')!r}, "
                        f"not {ap.DERIVED_FROM!r}")
    return problems


def refusal_problems(adv: dict, uniform: dict, *, pins: dict, pins_sha256: str, core_mixed: dict | None = None) -> list:
    """Every A14 refusal (plus the coverage rule); [] when the report may run."""
    return (_session_problems(adv) + _pins_problems(adv, pins, pins_sha256) + _c_problems(adv, uniform)
            + _dataset_problems(adv, pins) + _argv_problems(adv, pins) + _core_mixed_problems(core_mixed))


def report(adv: dict, uniform: dict, *, pins: dict, pins_sha256: str, core_mixed: dict | None = None) -> dict:
    """adv / uniform / core_mixed: {"session": session.json, "plan": plan.json, "rows": rows}."""
    problems = refusal_problems(adv, uniform, pins=pins, pins_sha256=pins_sha256, core_mixed=core_mixed)
    if problems:
        raise ReportError("; ".join(problems))
    try:
        analysis = aa.analyze(adv, uniform, core_mixed)
    except aa.AnalysisError as exc:
        raise ReportError(str(exc)) from exc
    analysis["report"] = {"session_id": adv["session"].get("session_id"),
                          "uniform_session": uniform["session"].get("session_id"),
                          "core_mixed_session": core_mixed["session"].get("session_id") if core_mixed else None,
                          "plan_sha256": adv["session"].get("plan_sha256"), "pins_sha256": pins_sha256,
                          "mixed_dataset_sha256": adv["session"].get("mixed_dataset_sha256")}
    return analysis


def load_session(path) -> dict:
    d = Path(path)
    return {"session": json.loads((d / "session.json").read_text()), "plan": json.loads((d / "plan.json").read_text()),
            "rows": [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()]}


def load_pins_bytes() -> bytes:
    return PINS_PATH.read_bytes()


def main(argv: list | None = None) -> int:
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="the formal Advanced session directory (run_adv.py)")
    parser.add_argument("--uniform-session", required=True, help="the formal Core B uniform session (C check)")
    parser.add_argument("--core-mixed-session", help="the formal Core B mixed session (descriptive ratio only, A4)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    raw = load_pins_bytes()
    try:
        rep = report(load_session(args.session), load_session(args.uniform_session),
                     pins=tomllib.loads(raw.decode()), pins_sha256=hashlib.sha256(raw).hexdigest(),
                     core_mixed=load_session(args.core_mixed_session) if args.core_mixed_session else None)
    except ReportError as exc:
        print(f"adv_report: refusing: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    its = {L: lv["interpretation"] for L, lv in rep["levels"].items()}
    print(json.dumps({"session": rep["session_id"], "rows": {L: it["row"] for L, it in its.items()},
                      "conclusions": {L: it["conclusion"] for L, it in its.items()},
                      "incomplete": {L: it["incomplete"] for L, it in its.items()},
                      "cross_l": rep["cross_l"]["row"], "cross_l_conclusion": rep["cross_l"]["conclusion"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
