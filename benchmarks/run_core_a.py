#!/usr/bin/env python3
"""Execute a saved Core A plan (core_a_plan.py), one process per job.

Stdlib only and Python 3.10 compatible, so the same file runs on the Mac and in
WSL2 Ubuntu-22.04. Output layout and row schema: docs/RESULTS_FORMAT.md.

Guards (MACHINE_POLICY, decision 006):
- formal mode on wsl2-cuda must run under `flock -n /tmp/ai-portfolio-gpu.lock`
  and be told so with --gpu-lock-held;
- formal mode on macos-metal must run under tools/with-mac-lock (which sets
  LLM_SERVING_LAB_MAC_LOCK_HELD=1);
- on the Mac, any run is refused while another project holds the Mac heavy lock;
- the pinned llama.cpp build is verified before the first job;
- an existing output directory is never resumed or overwritten.

Usage:
    python benchmarks/run_core_a.py --plan plan.json --bin <llama.cpp bin dir> \\
        --models <gguf dir> --out results/core_a/<platform>/<session> --session-id <id> \\
        [--gpu-lock-held]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import core_a_parse as cap

SCHEMA_VERSION = 1
MAC_HEAVY_LOCK = "/tmp/ai-portfolio-mac-heavy.lock"
SENTINEL_DRIFT_PCT = 3.0  # PREREGISTRATION §4
JOB_TIMEOUT_S = 7200


class RunError(RuntimeError):
    pass


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def default_run_proc(argv: list, stdout_path: Path, stderr_path: Path, timeout: float):
    """Run one job; returns (exit_code or None on timeout, wall seconds)."""
    t0 = time.monotonic()
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        try:
            code = subprocess.run(argv, stdout=out, stderr=err, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = None
        except OSError as exc:
            err.write(f"{type(exc).__name__}: {exc}\n")
            code = 127
    return code, round(time.monotonic() - t0, 3)


def make_env_capture(script: Path, python: str = sys.executable) -> Callable:
    """env_capture.py as a subprocess; never raises, failures land in probe_errors."""

    def capture(llama_bin: str, out_path: Path, gpu_lock_held: bool) -> dict:
        cmd = [python, str(script), "--llama-bin", llama_bin, "--out", str(out_path)]
        if gpu_lock_held:
            cmd.append("--gpu-lock-held")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not Path(out_path).is_file():
            rec = {"probe_errors": [f"env_capture failed (exit {res.returncode}): {res.stderr.strip()[:300]}"]}
            Path(out_path).write_text(json.dumps(rec))
            return rec
        return json.loads(Path(out_path).read_text())

    return capture


def _resolve(argv: list, bin_dir: str, models_dir: str, prompt: str) -> list:
    return [a.replace("{bin}", bin_dir).replace("{models}", models_dir).replace("{prompt}", prompt) for a in argv]


def _sentinel_summary(rows: list) -> dict:
    by_pos = {r["position"]: r for r in rows if r["kind"] == "sentinel"}
    start, end = by_pos.get("start"), by_pos.get("end")
    if not (start and end and start["outcome"] == "ok" and end["outcome"] == "ok"):
        return {"start_tok_s": None, "end_tok_s": None, "drift_pct": None, "drift": None,
                "note": "sentinel missing or failed; session must be re-run"}
    s, e = start["metrics"]["avg_ts"], end["metrics"]["avg_ts"]
    pct = abs(e / s - 1.0) * 100.0
    return {"start_tok_s": s, "end_tok_s": e, "drift_pct": round(pct, 4), "drift": pct > SENTINEL_DRIFT_PCT,
            "threshold_pct": SENTINEL_DRIFT_PCT}


def run_session(
    plan: dict,
    *,
    bin_dir,
    models_dir,
    out_dir,
    capture_env: Callable,
    session_id: str,
    gpu_lock_held: bool = False,
    mac_lock_held: bool = False,
    run_proc: Callable = default_run_proc,
    sleep: Callable = time.sleep,
    path_exists: Callable = os.path.exists,
    timeout_s: float = JOB_TIMEOUT_S,
    now: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(),
) -> dict:
    platform, mode = plan["platform"], plan["mode"]
    if platform == "macos-metal" and not mac_lock_held and path_exists(MAC_HEAVY_LOCK):
        raise RunError(f"Mac heavy lock is held by another job ({MAC_HEAVY_LOCK}); do not run now (MACHINE_POLICY)")
    if mode == "formal":
        if platform == "wsl2-cuda" and not gpu_lock_held:
            raise RunError("formal NVIDIA runs must hold the GPU lock: flock -n /tmp/ai-portfolio-gpu.lock ... "
                           "--gpu-lock-held")
        if platform == "macos-metal" and not mac_lock_held:
            raise RunError("formal Mac runs must run under tools/with-mac-lock (MACHINE_POLICY 'Mac Hard Lock')")

    out = Path(out_dir)
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten")
    for sub in ("raw", "env", "prompts"):
        (out / sub).mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")

    bin_dir, models_dir = str(bin_dir), str(models_dir)
    llama_bin = f"{bin_dir}/llama-completion"
    preflight = capture_env(llama_bin=llama_bin, out_path=out / "env" / "preflight.json", gpu_lock_held=gpu_lock_held)
    lc = (preflight.get("runtime") or {}).get("llama_cpp") or {}
    want = plan["expect"]
    if lc.get("build_number") != want["build_number"] or lc.get("commit") != want["build_commit"]:
        raise RunError(f"llama.cpp build mismatch: got build {lc.get('build_number')} commit {lc.get('commit')}, "
                       f"plan expects build {want['build_number']} commit {want['build_commit']}")

    session = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "platform": platform,
        "mode": mode,
        "seed": plan["seed"],
        "plan_sha256": sha256_json(plan),
        "pins_sha256": plan.get("pins_sha256"),
        "gpu_lock_held": gpu_lock_held,
        "mac_lock_held": mac_lock_held,
        "bin_dir": bin_dir,
        "models_dir": models_dir,
        "preflight": "env/preflight.json",
        "started_at": now(),
    }

    params = plan["params"]
    bench_expect = {"build_commit": want["build_commit"], "n_threads": params["n_thr"], "poll": params["poll"],
                    "reps": params["bench_r"]}
    rows = []
    with open(out / "rows.jsonl", "w") as rows_file:
        for i, job in enumerate(plan["jobs"]):
            if i:
                sleep(params["idle_s"])
            prompt_path = ""
            if job.get("prompt"):
                p = job["prompt"]
                prompt_file = out / "prompts" / f"unit_x{p['count']}.txt"
                if not prompt_file.exists():
                    prompt_file.write_text(p["unit"] * p["count"])
                prompt_path = str(prompt_file)
            argv = _resolve(job["argv"], bin_dir, models_dir, prompt_path)

            jid = job["id"]
            rel = {
                "stdout": f"raw/{jid}.stdout.txt", "stderr": f"raw/{jid}.stderr.txt",
                "env_before": f"env/{jid}.before.json", "env_after": f"env/{jid}.after.json",
            }
            capture_env(llama_bin=llama_bin, out_path=out / rel["env_before"], gpu_lock_held=gpu_lock_held)
            exit_code, wall_s = run_proc(argv, out / rel["stdout"], out / rel["stderr"], timeout_s)
            capture_env(llama_bin=llama_bin, out_path=out / rel["env_after"], gpu_lock_held=gpu_lock_held)

            stdout = (out / rel["stdout"]).read_text(errors="replace")
            stderr = (out / rel["stderr"]).read_text(errors="replace")
            if job["kind"] == "fit":
                result = cap.classify_completion(exit_code, cap.parse_log_jsonl(stdout), stderr,
                                                 job["expect"]["prompt_tokens"])
            else:
                result = cap.classify_bench(exit_code, stdout, stderr, {**bench_expect, **job["expect"]})

            row = {
                "job_id": jid, "kind": job["kind"], "position": job.get("position"),
                "size": job["size"], "quant": job["quant"], "L": job["L"], "rep": job["rep"], "model": job["model"],
                "argv": argv, "exit_code": exit_code, "wall_s": wall_s,
                "outcome": result["outcome"], "stage": result["stage"], "reason": result["reason"],
                "evidence": result["evidence"], "metrics": result.get("metrics"), "summary": result.get("summary"),
                **rel,
            }
            rows.append(row)
            rows_file.write(json.dumps(row) + "\n")
            rows_file.flush()

    outcomes = {"ok": 0, "oom": 0, "error": 0}
    for row in rows:
        outcomes[row["outcome"]] += 1
    session.update({"finished_at": now(), "outcomes": outcomes, "sentinel": _sentinel_summary(rows)})
    (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    return session


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--bin", required=True, help="llama.cpp build bin directory")
    parser.add_argument("--models", required=True, help="directory with the pinned GGUF files")
    parser.add_argument("--out", required=True, help="new session directory")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--gpu-lock-held", action="store_true", help="set when running under the WSL GPU flock")
    args = parser.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text())
    script = Path(__file__).resolve().parent.parent / "scripts" / "env_capture.py"
    try:
        summary = run_session(
            plan, bin_dir=Path(args.bin).resolve(), models_dir=Path(args.models).resolve(), out_dir=args.out,
            capture_env=make_env_capture(script), session_id=args.session_id, gpu_lock_held=args.gpu_lock_held,
            mac_lock_held=os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") == "1",
        )
    except RunError as exc:
        print(f"run_core_a: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary[k] for k in ("session_id", "outcomes", "sentinel")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
