#!/usr/bin/env python3
"""Core A pilot (PREREGISTRATION §5 Stage A checklist). Pilot data are never formal data.

Blocks, executed by run_core_a.py like any plan (one process per job):
  fa_probe  fit run per model @ shortest L with `-fa auto`: the log must say
            "Flash Attention enabled" (002); graph splits expose a silent fused-GDN
            CPU fallback (008).
  poll      sentinel decode, --poll 0 / 50 alternating, POLL_REPS each -> the §4 rule.
  memory    fit run per (quant, L) of the given sizes -> breakdown vs external peak
            memory (§3 cross-check).
  calib     ttft + decode at the longest L for the heaviest model -> upper bound of
            the formal session duration.
plus the sentinel at start and end (drift).

The report is masked: it prints the poll medians, the noise level and MDE, the
memory cross-check and a duration bound, but never speeds of different cells side
by side, so no hypothesis effect (H-A2..H-A4) is seen before the Stage A freeze.

Usage:
    python benchmarks/core_a_pilot.py plan --platform macos-metal --sizes 4B --n-thr 8 --out pilot.json
    python benchmarks/run_core_a.py --plan pilot.json ... --out results/core_a/macos-metal/pilot-<date>
    python benchmarks/core_a_pilot.py report --session results/core_a/macos-metal/pilot-<date>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tomllib
from pathlib import Path

import core_a_plan as cp

POLL_REPS = 3
POLL_VALUES = (0, 50)
POLL_SLOWER_PCT = 3.0  # §4: use 50 only if poll 0 is > 3% slower (median decode tok/s)
MiB = 2**20


def _files(pins: dict) -> dict:
    return {(m["size"], q): f["filename"] for m in pins["core_a"]["models"] for q, f in m["files"].items()}


def _fit_job(jid, block, size, quant, L, model, n_thr, platform, fa="on"):
    ext_kind, prefix = cp.EXTERNAL_MEMORY[platform]
    argv = prefix + cp.completion_argv(model, L, n_thr, 0)
    argv[argv.index("-fa") + 1] = fa
    return {"id": jid, "block": block, "kind": "fit", "size": size, "quant": quant, "L": L, "rep": 1,
            "model": model, "argv": argv, "external_memory": ext_kind,
            "prompt": {"unit": cp.PROMPT_UNIT, "count": L - cp.N_GEN}, "expect": {"prompt_tokens": L - cp.N_GEN}}


def build_pilot_plan(pins: dict, *, platform: str, sizes: list, n_thr: int, pins_sha256: str | None = None) -> dict:
    files = _files(pins)
    quants, contexts = pins["core_a"]["quants"], pins["core_a"]["context_lengths"]
    known = [m["size"] for m in pins["core_a"]["models"]]
    sizes = [s for s in known if s in sizes]  # pins order: smallest first
    if not sizes:
        raise cp.PlanError(f"no known sizes in {sizes}; expected some of {known}")
    jobs = []
    for size in sizes:
        for q in quants:
            jobs.append(_fit_job(f"pilot-fa-{size}-{q}", "fa_probe", size, q, min(contexts), files[(size, q)],
                                 n_thr, platform, fa="auto"))
    s_size, s_quant, s_L = cp.sentinel_cell(pins)
    for rep in range(1, POLL_REPS + 1):
        for poll in POLL_VALUES:
            j = cp._bench_job("decode", s_size, s_quant, s_L, rep, files[(s_size, s_quant)], n_thr, poll)
            j.update({"id": f"pilot-poll{poll}-r{rep}", "block": "poll"})
            j["expect"] = {**j["expect"], "poll": poll}
            jobs.append(j)
    for size in sizes:
        for q in quants:
            for L in contexts:
                jobs.append(_fit_job(f"pilot-mem-{size}-{q}-{L}", "memory", size, q, L, files[(size, q)],
                                     n_thr, platform))
    heavy_size, heavy_q, L_max = sizes[-1], quants[0], max(contexts)
    for kind in ("ttft", "decode"):
        j = cp._bench_job(kind, heavy_size, heavy_q, L_max, 1, files[(heavy_size, heavy_q)], n_thr, 0)
        j.update({"id": f"pilot-calib-{kind}-{heavy_size}-{heavy_q}-{L_max}", "block": "calib"})
        jobs.append(j)
    sentinel = lambda pos: {**cp._bench_job("sentinel", s_size, s_quant, s_L, 1, files[(s_size, s_quant)],  # noqa: E731
                                             n_thr, 0, position=pos), "block": "sentinel"}
    return {
        "schema_version": cp.SCHEMA_VERSION, "platform": platform, "mode": "pilot", "seed": None,
        "pilot": {"sizes": sizes, "poll_reps": POLL_REPS, "poll_values": list(POLL_VALUES)},
        "pins_sha256": pins_sha256,
        "params": {"n_thr": n_thr, "poll": 0, "reps": 1, "bench_r": cp.BENCH_R, "delay_s": cp.DELAY_S,
                   "idle_s": cp.IDLE_S},
        "expect": {"build_number": pins["llama_cpp"]["build_number"], "build_commit": pins["llama_cpp"]["commit"]},
        "jobs": [sentinel("start"), *jobs, sentinel("end")],
    }


# ------------------------------------------------------------------ report --


def _poll_of(r: dict) -> int:
    a = r["argv"]
    return int(a[a.index("--poll") + 1])


def poll_decision(rows: list) -> dict:
    by = {p: [r["metrics"]["avg_ts"] for r in rows if r.get("block") == "poll" and r["outcome"] == "ok"
              and _poll_of(r) == p] for p in POLL_VALUES}
    med = {str(p): statistics.median(v) for p, v in by.items() if v}
    if len(med) < 2:
        return {"poll": None, "median_tok_s": med, "note": "poll block incomplete"}
    slower = (1 - med["0"] / med["50"]) * 100
    return {"poll": 50 if slower > POLL_SLOWER_PCT else 0, "median_tok_s": med,
            "poll0_slower_pct": round(slower, 3), "rule": f"50 only if poll 0 is > {POLL_SLOWER_PCT}% slower"}


def noise_and_mde(rows: list, poll: int) -> dict:
    v = [r["metrics"]["avg_ts"] for r in rows if r.get("block") == "poll" and r["outcome"] == "ok"
         and _poll_of(r) == poll]
    if len(v) < 2:
        return {"n": len(v), "cv_pct": None, "mde_pct": None}
    cv = statistics.stdev(v) / statistics.mean(v) * 100
    # §2: beyond noise if |x - y| > 2 sqrt(SE_x^2 + SE_y^2), SE = s / sqrt(3); equal CV assumed
    return {"n": len(v), "cv_pct": cv, "mde_pct": 2 * (2 / 3) ** 0.5 * cv,
            "note": "process-level CV of the sentinel cell; MDE = relative difference two cells need to be "
                    "'beyond noise' with n = 3 each"}


def memory_cross_check(rows: list) -> list:
    out = []
    for r in rows:
        if r.get("block") not in ("memory", "fa_probe") or r["kind"] != "fit":
            continue
        mem = (r.get("summary") or {}).get("memory") or {}
        dev = sum(d.get("self", 0) for d in mem.get("devices", []))
        host = (mem.get("host") or {}).get("self")
        ext = r.get("external_memory") or {}
        fp = ext.get("peak_footprint_bytes")
        out.append({
            "size": r["size"], "quant": r["quant"], "L": r["L"], "outcome": r["outcome"],
            "device_self_mib": dev if mem else None, "host_self_mib": host,
            "footprint_mib": fp / MiB if fp else None,
            "max_rss_mib": ext["max_rss_bytes"] / MiB if ext.get("max_rss_bytes") else None,
            "footprint_over_self": (fp / MiB) / (dev + (host or 0)) if fp and mem else None,
            "graph_splits": ((r.get("summary") or {}).get("graph") or {}).get("splits"),
        })
    return out


def fa_probe(rows: list) -> list:
    return [{"size": r["size"], "quant": r["quant"], "outcome": r["outcome"],
             "fused_ops": (r.get("summary") or {}).get("fused_ops"),
             "graph": (r.get("summary") or {}).get("graph"),
             "fa_enabled": "Flash Attention enabled" in ((r.get("summary") or {}).get("fused_ops") or [])}
            for r in rows if r.get("block") == "fa_probe"]


def report(rows: list, *, n_formal_jobs: dict, idle_s: float) -> dict:
    poll = poll_decision(rows)
    walls = {k: max((r["wall_s"] for r in rows if r.get("block") == "calib" and r["kind"] == k), default=None)
             for k in ("ttft", "decode")}
    fit_wall = max((r["wall_s"] for r in rows if r["kind"] == "fit"), default=None)
    per_kind = {"fit": fit_wall, "ttft": walls["ttft"], "decode": walls["decode"], "sentinel": walls["decode"]}
    bound = None
    if all(per_kind.values()):
        bound = sum(n * (per_kind[k] + idle_s) for k, n in n_formal_jobs.items()) / 3600
    return {
        "fa_probe": fa_probe(rows),
        "poll": poll,
        "noise": noise_and_mde(rows, poll["poll"] if poll["poll"] is not None else 0),
        "memory": memory_cross_check(rows),
        # masked: only the heaviest-job bound, never per-cell speeds
        "calibration": {"upper_bound_hours": round(bound, 2) if bound else None,
                        "method": "n_jobs x (slowest pilot wall time of that kind + idle) per kind"},
        "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in ("ok", "oom", "error")},
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--pins", default=str(Path(__file__).resolve().parent.parent / "configs" / "pins.toml"))
    p.add_argument("--platform", required=True, choices=sorted(cp.PLATFORMS))
    p.add_argument("--sizes", required=True, nargs="+")
    p.add_argument("--n-thr", required=True, type=int)
    p.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--session", required=True)
    r.add_argument("--pins", default=str(Path(__file__).resolve().parent.parent / "configs" / "pins.toml"))
    args = parser.parse_args(argv)

    if args.cmd == "plan":
        raw = Path(args.pins).read_bytes()
        plan = build_pilot_plan(tomllib.loads(raw.decode()), platform=args.platform, sizes=args.sizes,
                                n_thr=args.n_thr, pins_sha256=hashlib.sha256(raw).hexdigest())
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
        return 0

    session = Path(args.session)
    plan = json.loads((session / "plan.json").read_text())
    blocks = {j["id"]: j.get("block") for j in plan["jobs"]}
    rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines()]
    for row in rows:
        row.setdefault("block", blocks.get(row["job_id"]))
    pins = tomllib.loads(Path(args.pins).read_text())
    formal = cp.build_plan({**pins, "freeze": {"mac": True, "nvidia": True}}, platform=plan["platform"],
                           mode="formal", seed=0, n_thr=plan["params"]["n_thr"], poll=0)
    n_jobs: dict = {}
    for j in formal["jobs"]:
        n_jobs[j["kind"]] = n_jobs.get(j["kind"], 0) + 1
    rep = report(rows, n_formal_jobs=n_jobs, idle_s=plan["params"]["idle_s"])
    out = session / "pilot_report.json"
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
