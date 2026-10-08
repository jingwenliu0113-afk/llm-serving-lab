#!/usr/bin/env python3
"""Plan and run the Core A quality protocol (decision 005) with llama-perplexity.

Per size, in this order:
  kld_base (BF16 -> data/kld/<size>.kld) -> kld_self (floor) -> kld x 3 quants
  -> delete_kld (only if all four KLD runs are ok; docs/STORAGE_PLAN.md)
  -> downstream (HellaSwag / MMLU / ARC via --multiple-choice, WinoGrande) for BF16 + 3 quants
  -> per-chunk PPL (--ppl-output-type 1) for the H-A5(a) pair.
One process per job; ok only with positive markers and expected counts
(quality_parse). pilot = 20 chunks / 200 tasks; formal requires freeze.mac and
running under tools/with-mac-lock.

Usage:
    uv run python quality/run_quality.py plan --mode pilot --n-thr 8 --poll 0 --out plan.json
    tools/with-mac-lock uv run python quality/run_quality.py run --plan plan.json \\
        --bin third_party/llama.cpp/build-metal/bin --models models/gguf --data data --out results/quality/<session>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_guard  # noqa: E402
import quality_parse as qp  # noqa: E402
from run_core_a import default_run_proc, make_env_capture, sha256_json  # noqa: E402

SCHEMA_VERSION = 1
MAC_HEAVY_LOCK = "/tmp/ai-portfolio-mac-heavy.lock"
QUALITY_FLAGS = {"-ngl": "999", "-fa": "on", "--fit": "off", "-ctk": "f16", "-ctv": "f16",
                 "-b": "2048", "-ub": "512", "-c": "512", "-lm": "mmap"}
PILOT = {"chunks": 20, "tasks": 200}
KLD_NEED_GIB = {"pilot": 4, "formal": 70}  # ~73.5 GB per size at 580 chunks (005 / 008)
MC_TASKS = ("hellaswag", "mmlu", "arc")
FORMAL_FILES = {
    "hellaswag": "{data}/datasets/hellaswag-val-4000.bin",
    "mmlu": "{data}/datasets/mmlu-test-4000.bin",
    "arc": "{data}/datasets/arc-combined-validation.bin",
    "winogrande": "{data}/datasets/winogrande-debiased-eval.csv",
}
PILOT_FILES = {t: f"{{data}}/datasets/pilot/{t}-200.bin" for t in MC_TASKS}
PILOT_FILES["winogrande"] = FORMAL_FILES["winogrande"]


class QualityError(RuntimeError):
    pass


def _flags(n_thr: int, poll: int) -> list:
    return [x for kv in QUALITY_FLAGS.items() for x in kv] + ["-t", str(n_thr), "--poll", str(poll)]


def build_quality_plan(pins: dict, *, mode: str, n_thr: int, poll: int, pins_sha256: str | None = None) -> dict:
    if mode not in ("pilot", "formal"):
        raise QualityError(f"unknown mode {mode!r}")
    if mode == "formal" and not pins["freeze"]["mac"]:
        raise QualityError("formal quality runs require freeze.mac = true in pins.toml (decision 006)")
    pilot = mode == "pilot"
    exp = pins["quality"]["expected"]
    n_chunks = PILOT["chunks"] if pilot else exp["wikitext_chunks"]
    n_tasks = {t: PILOT["tasks"] if pilot else exp[f"{t}_tasks"] for t in (*MC_TASKS, "winogrande")}
    files = PILOT_FILES if pilot else FORMAL_FILES
    chunks_arg = ["--chunks", str(n_chunks)] if pilot else []
    flags = _flags(n_thr, poll)
    models = pins["core_a"]["models"]
    quants = pins["core_a"]["quants"]
    ha5_pair = {(models[-1]["size"], quants[-1]), (models[0]["size"], quants[0])}
    exe = "{bin}/llama-perplexity"

    jobs = []
    for m in models:
        size = m["size"]
        gguf = {"BF16": m["reference"]["filename"], **{q: m["files"][q]["filename"] for q in quants}}
        model = lambda p: ["-m", "{models}/" + gguf[p]]  # noqa: E731
        kld = f"{{data}}/kld/{size}.kld"

        def add(jid, kind, point, argv, expect, **extra):
            jobs.append({"id": jid, "kind": kind, "size": size, "point": point, "argv": argv, "expect": expect, **extra})

        add(f"kld_base-{size}", "kld_base", "BF16",
            [exe, *model("BF16"), "-f", "{data}/datasets/wiki.test.raw", "--kl-divergence-base", kld, *chunks_arg, *flags],
            {"n_chunks": n_chunks})
        add(f"kld_self-{size}", "kld_self", "BF16",
            [exe, *model("BF16"), "--kl-divergence-base", kld, "--kl-divergence", *flags], {"n_chunks": n_chunks})
        for q in quants:
            add(f"kld-{size}-{q}", "kld", q,
                [exe, *model(q), "--kl-divergence-base", kld, "--kl-divergence", *flags], {"n_chunks": n_chunks})
        jobs.append({"id": f"delete_kld-{size}", "kind": "delete_kld", "size": size, "point": None, "path": kld,
                     "after": [f"kld_base-{size}", f"kld_self-{size}", *(f"kld-{size}-{q}" for q in quants)]})
        for point in ("BF16", *quants):
            for task in MC_TASKS:
                add(f"mc-{task}-{size}-{point}", "mc", point,
                    [exe, *model(point), "-bf", files[task], "--multiple-choice", *flags, "-np", "16"],
                    {"n_tasks": n_tasks[task]}, task=task)
            wg_n = ["--winogrande-tasks", str(PILOT["tasks"])] if pilot else []
            add(f"winogrande-{size}-{point}", "winogrande", point,
                [exe, *model(point), "-f", files["winogrande"], "--winogrande", *wg_n, *flags, "-np", "16"],
                {"n_tasks": n_tasks["winogrande"]}, task="winogrande")
        for q in quants:
            if (size, q) in ha5_pair:
                add(f"ppl-{size}-{q}", "ppl", q,
                    [exe, *model(q), "-f", "{data}/datasets/wiki.test.raw", "--ppl-output-type", "1", *chunks_arg, *flags],
                    {"n_chunks": n_chunks})

    return {
        "schema_version": SCHEMA_VERSION, "mode": mode, "pins_sha256": pins_sha256,
        "params": {"n_thr": n_thr, "poll": poll},
        "expected": {"n_chunks": n_chunks, "pilot_tasks": PILOT["tasks"] if pilot else None},
        "expect_build": {"build_number": pins["llama_cpp"]["build_number"], "build_commit": pins["llama_cpp"]["commit"]},
        "jobs": jobs,
    }


def _default_disk_ok(path, need_gib: float) -> dict:
    return disk_guard.check(str(path), need_bytes=int(need_gib * disk_guard.GiB),
                            reserve_bytes=disk_guard.DEFAULT_RESERVE_GIB * disk_guard.GiB)


def run_quality(
    plan: dict,
    *,
    bin_dir,
    models_dir,
    data_dir,
    out_dir,
    capture_env: Callable,
    session_id: str,
    mac_lock_held: bool = False,
    path_exists: Callable = os.path.exists,
    disk_ok: Callable = _default_disk_ok,
    run_proc: Callable = default_run_proc,
    delete_file: Callable = os.remove,
    timeout_s: float = 6 * 3600,
    now: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(),
) -> dict:
    if not mac_lock_held and path_exists(MAC_HEAVY_LOCK):
        raise QualityError(f"Mac heavy lock is held by another job ({MAC_HEAVY_LOCK}); do not run now (MACHINE_POLICY)")
    if plan["mode"] == "formal" and not mac_lock_held:
        raise QualityError("formal quality runs must run under tools/with-mac-lock (MACHINE_POLICY 'Mac Hard Lock')")
    out = Path(out_dir)
    if out.exists():
        raise QualityError(f"{out} exists; sessions are never resumed or overwritten")
    for sub in ("raw", "env"):
        (out / sub).mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")

    bin_dir, models_dir, data_dir = str(bin_dir), str(models_dir), str(data_dir)
    exe = f"{bin_dir}/llama-perplexity"
    pre = capture_env(llama_bin=exe, out_path=out / "env" / "preflight.json", gpu_lock_held=False)
    lc = (pre.get("runtime") or {}).get("llama_cpp") or {}
    want = plan["expect_build"]
    if lc.get("build_number") != want["build_number"] or lc.get("commit") != want["build_commit"]:
        raise QualityError(f"llama.cpp build mismatch: {lc} vs {want}")

    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "mode": plan["mode"],
               "plan_sha256": sha256_json(plan), "pins_sha256": plan.get("pins_sha256"),
               "mac_lock_held": mac_lock_held, "started_at": now()}
    rows, by_id, skipped_sizes = [], {}, set()
    resolve = lambda a: a.replace("{bin}", bin_dir).replace("{models}", models_dir).replace("{data}", data_dir)  # noqa: E731
    with open(out / "rows.jsonl", "w") as fh:
        for job in plan["jobs"]:
            jid, kind, size = job["id"], job["kind"], job["size"]
            base = {"job_id": jid, "kind": kind, "size": size, "point": job.get("point"), "task": job.get("task")}
            if kind == "delete_kld":
                path = resolve(job["path"])
                all_ok = all(by_id.get(a, {}).get("outcome") == "ok" for a in job["after"])
                deleted = False
                if all_ok and os.path.exists(path):
                    delete_file(path)
                    deleted = True
                row = {**base, "outcome": "ok" if all_ok else "error", "deleted": deleted, "path": path,
                       "reason": None if all_ok else "KLD runs not all ok; KLD file kept for diagnosis"}
            elif kind in ("kld_base", "kld_self", "kld") and size in skipped_sizes:
                row = {**base, "outcome": "skipped", "reason": "disk guard failed for this size's KLD file"}
            else:
                guard = None
                if kind == "kld_base":
                    guard = disk_ok(Path(data_dir) / "kld", KLD_NEED_GIB[plan["mode"]])
                    if not guard.get("ok"):
                        skipped_sizes.add(size)
                        row = {**base, "outcome": "skipped", "reason": f"disk guard failed: {guard}"}
                        rows.append(row); by_id[jid] = row; fh.write(json.dumps(row) + "\n"); fh.flush()
                        continue
                argv = [resolve(a) for a in job["argv"]]
                rel = {"stdout": f"raw/{jid}.stdout.txt", "stderr": f"raw/{jid}.stderr.txt",
                       "env_before": f"env/{jid}.before.json", "env_after": f"env/{jid}.after.json"}
                capture_env(llama_bin=exe, out_path=out / rel["env_before"], gpu_lock_held=False)
                code, wall = run_proc(argv, out / rel["stdout"], out / rel["stderr"], timeout_s)
                capture_env(llama_bin=exe, out_path=out / rel["env_after"], gpu_lock_held=False)
                res = qp.classify(kind, code, (out / rel["stdout"]).read_text(errors="replace"),
                                  (out / rel["stderr"]).read_text(errors="replace"), job["expect"])
                row = {**base, "argv": argv, "exit_code": code, "wall_s": wall, **res, "disk_guard": guard, **rel}
            rows.append(row)
            by_id[jid] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    outcomes = {"ok": 0, "oom": 0, "error": 0, "skipped": 0}
    for r in rows:
        outcomes[r["outcome"]] += 1
    session.update({"finished_at": now(), "outcomes": outcomes})
    (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    return session


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--mode", required=True, choices=["pilot", "formal"])
    p_plan.add_argument("--n-thr", type=int, required=True)
    p_plan.add_argument("--poll", type=int, required=True)
    p_plan.add_argument("--out", required=True)
    p_run = sub.add_parser("run")
    for a in ("--plan", "--bin", "--models", "--data", "--out"):
        p_run.add_argument(a, required=True)
    p_run.add_argument("--session-id", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "plan":
        raw = (ROOT / "configs" / "pins.toml").read_bytes()
        plan = build_quality_plan(tomllib.loads(raw.decode()), mode=args.mode, n_thr=args.n_thr, poll=args.poll,
                                  pins_sha256=hashlib.sha256(raw).hexdigest())
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
        return 0
    plan = json.loads(Path(args.plan).read_text())
    try:
        summary = run_quality(
            plan, bin_dir=Path(args.bin).resolve(), models_dir=Path(args.models).resolve(),
            data_dir=Path(args.data).resolve(), out_dir=args.out,
            capture_env=make_env_capture(ROOT / "scripts" / "env_capture.py"),
            session_id=args.session_id or Path(args.out).name,
            mac_lock_held=os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") == "1",
        )
    except QualityError as exc:
        print(f"run_quality: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary["outcomes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
