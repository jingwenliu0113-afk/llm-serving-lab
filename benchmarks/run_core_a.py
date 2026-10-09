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
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import core_a_parse as cap

SCHEMA_VERSION = 2  # v2 (decision 010): rows carry external_memory
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


class NvidiaSmiSampler:
    """Samples nvidia-smi memory.used (MiB) every 200 ms while a fit process runs
    (PREREGISTRATION §3: RTX peak memory = sampled peak - B; B is subtracted in analysis)."""

    def __init__(self, path: Path, interval_ms: int = 200):
        self.path, self.interval_ms, self.proc, self.fh = Path(path), interval_ms, None, None

    def start(self) -> None:
        self.fh = open(self.path, "w")
        try:
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                          "-lms", str(self.interval_ms)], stdout=self.fh, stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None

    def stop(self) -> dict | None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.fh is not None:
            self.fh.close()
        values = [int(x) for x in self.path.read_text().split() if x.strip().isdigit()] if self.path.exists() else []
        if not values:
            return None
        return {"gpu_mem_used_peak_mib": max(values), "gpu_mem_used_first_mib": values[0], "n_samples": len(values)}


BATTERY_MIN_PCT = 20  # decision 012


# 019 (user, 2026-10-09; §4 had 2): automatic mode; user addendum the same day: high power (2) is allowed too.
# §4 still holds one powermode for a whole session (checked against the preflight in run_session).
POWERMODES_ALLOWED = (0, 2)
# 019: the Apple 70 W adapter (`pmset -g ac`); the negotiated wattage (68 W on 2026-10-09) is recorded, not required
AC_ADAPTER_REQUIRED = {"manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"}


def mac_config_problems(rec: dict) -> list:
    """§4 / 004 / 019: powermode 0 or 2, the Apple 70 W adapter and iogpu.wired_limit_mb = 0 (unknown counts as a
    problem, 013)."""
    power = (rec or {}).get("power") or {}
    wired = ((rec or {}).get("hardware") or {}).get("iogpu_wired_limit_mb")
    problems = []
    if power.get("powermode") not in POWERMODES_ALLOWED:
        problems.append(f"powermode {power.get('powermode')!r} (019 allows 0 or 2)")
    adapter = power.get("ac_adapter") or {}
    got = {k: adapter.get(k) for k in AC_ADAPTER_REQUIRED}
    if got != AC_ADAPTER_REQUIRED:
        problems.append(f"AC adapter {got['manufacturer']!r} {got['name']!r} (019 requires Apple Inc. "
                        f"{AC_ADAPTER_REQUIRED['name']!r})")
    if wired != 0:
        problems.append(f"iogpu.wired_limit_mb {wired!r} (§4 requires 0)")
    return problems


def env_problems(rec: dict, platform: str) -> list:
    """State that invalidates a session. Mac (012): AC, battery >= 20 %, no thermal / performance warning; a
    failed or partial probe cannot show them, so unknown counts as a problem (013); a Mac without a battery
    reports no percentage. RTX (§4): no battery; enforced power limit == default power limit (018)."""
    if platform == "wsl2-cuda":  # §4 RTX: no battery; the power limit must be the default one
        gpus = ((rec or {}).get("gpu") or {}).get("nvidia") or []
        if not gpus:
            return ["nvidia-smi power limits unknown (§4)"]
        out = []
        for g in gpus:
            e, d = g.get("enforced_power_limit_w"), g.get("power_default_limit_w")
            if e is None or d is None:
                out.append("power limit unknown (§4)")
            elif e != d:
                out.append(f"enforced power limit {e} W != default {d} W (§4)")
        return out
    if platform != "macos-metal":
        return []
    power = (rec or {}).get("power") or {}
    problems = []
    if power.get("source") is None:
        problems.append("power source unknown (AC required)")
    elif power["source"] != "AC":
        problems.append(f"power source {power['source']} (AC required)")
    if power.get("battery_pct") is not None and power["battery_pct"] < BATTERY_MIN_PCT:
        problems.append(f"battery {power['battery_pct']}% < {BATTERY_MIN_PCT}%")
    thermal = power.get("thermal")
    if thermal is None:
        problems.append("thermal state unknown")
    elif thermal.get("thermal_warning_recorded") or thermal.get("performance_warning_recorded"):
        problems.append("thermal / performance warning recorded")
    return problems + mac_config_problems(rec)


# decision 016: on the Mac, a process starts only with charge to spare. Under full load the 70 W adapter
# loses ~25-40 %/h (pilots), so long sessions would cross the 20 % validity floor (012); below the pause
# level the harness idles (charging) until the resume level. Pauses are recorded in session.json.
CHARGE_PAUSE_BELOW_PCT = 40
CHARGE_RESUME_AT_PCT = 90
CHARGE_POLL_S = 60
CHARGE_MAX_WAIT_S = 4 * 3600
_PMSET_SOURCE = re.compile(r"Now drawing from '(AC|Battery) Power'")
_PMSET_PCT = re.compile(r"\t(\d+)%;")


def parse_pmset_batt(text: str) -> dict:
    """Same parse as scripts/env_capture.py (kept local: the runner must not depend on that script's path)."""
    m, pct = _PMSET_SOURCE.search(text), _PMSET_PCT.search(text)
    return {"source": m.group(1) if m else None, "battery_pct": int(pct.group(1)) if pct else None}


def read_mac_power() -> dict:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {"source": None, "battery_pct": None}
    return parse_pmset_batt(out)


def charge_gate(read_power: Callable, sleep: Callable, pause_below: int = CHARGE_PAUSE_BELOW_PCT,
                resume_at: int = CHARGE_RESUME_AT_PCT) -> dict | None:
    """Before a Mac process (016): on AC below the pause level, idle until the resume level (or the max
    wait). Off AC or unknown -> no pause (env_problems flags it). Returns the pause record or None."""
    p = read_power()
    pct = p.get("battery_pct")
    if p.get("source") != "AC" or pct is None or pct >= pause_below:
        return None
    start, waited = pct, 0
    while pct is not None and pct < resume_at and waited < CHARGE_MAX_WAIT_S:
        sleep(CHARGE_POLL_S)
        waited += CHARGE_POLL_S
        pct = read_power().get("battery_pct")
    return {"from_pct": start, "to_pct": pct, "waited_s": waited,
            "timed_out": pct is None or pct < resume_at}


def provenance_problems(rec: dict, pre: dict) -> list:
    """A formal session runs one commit, unmodified, from start to end. A WSL redeploy swaps DEPLOYED_COMMIT and the
    manifest together (dirty stays False) and a Mac commit moves HEAD, so every capture is compared with the
    preflight; an unknown commit or dirty flag counts as a problem (013)."""
    r, p = (rec or {}).get("repo") or {}, (pre or {}).get("repo") or {}
    out = []
    if r.get("commit") is None or r.get("commit") != p.get("commit"):
        out.append(f"repo commit {r.get('commit')!r} != preflight {p.get('commit')!r} (mid-session code change)")
    if r.get("dirty") is not False:
        out.append(f"repo dirty={r.get('dirty')!r} (the code differs from the recorded commit)")
    return out


def write_raw_manifest(out: Path) -> dict:
    """raw/ is kept in the local archive, not in git; this committed manifest pins its content."""
    files = {}
    for path in sorted((out / "raw").rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            files[path.relative_to(out).as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(),
                                                       "size_bytes": len(data)}
    manifest = {"note": "raw/ is not in git (docs/STORAGE_PLAN.md); sha256 of every raw file", "files": files}
    (out / "raw_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


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
    gpu_sampler: Callable | None = None,
    sleep: Callable = time.sleep,
    path_exists: Callable = os.path.exists,
    read_power: Callable = read_mac_power,
    measure_b: Callable | None = None,
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
    if mode == "formal" and platform == "macos-metal":
        bad = mac_config_problems(preflight)
        if bad:
            raise RunError("formal Mac run refused before it starts: " + "; ".join(bad))
    if mode == "formal":
        bad = provenance_problems(preflight, preflight)
        if bad:
            raise RunError("formal run refused before it starts: " + "; ".join(bad))
    session_b = None
    if mode == "formal" and platform == "wsl2-cuda":  # §4 step 4: re-measure B at every formal session start
        if measure_b is None or not want.get("frozen_b_mib"):
            raise RunError("formal RTX Core A needs the session-start B measurement (§4 step 4; stage_b.py)")
        session_b = measure_b(want["frozen_b_mib"])

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
        "session_b": session_b,
        "powermode": ((preflight or {}).get("power") or {}).get("powermode") if platform == "macos-metal" else None,
        "started_at": now(),
    }

    params = plan["params"]
    bench_expect = {"build_commit": want["build_commit"], "n_threads": params["n_thr"], "poll": params["poll"],
                    "reps": params["bench_r"]}
    rows, env_flags, charge_pauses = [], [], []
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
            if platform == "macos-metal":
                pause = charge_gate(read_power, sleep)
                if pause:
                    charge_pauses.append({"job_id": jid, **pause})
            ext_kind = job.get("external_memory")
            sampler = None
            if ext_kind == "nvidia-smi" and gpu_sampler is not None:
                rel["gpu_mem_samples"] = f"raw/{jid}.gpumem.txt"
                sampler = gpu_sampler(out / rel["gpu_mem_samples"])
            env_before = capture_env(llama_bin=llama_bin, out_path=out / rel["env_before"], gpu_lock_held=gpu_lock_held)
            if sampler is not None:
                sampler.start()
            try:
                exit_code, wall_s = run_proc(argv, out / rel["stdout"], out / rel["stderr"], timeout_s)
            finally:
                sampled = sampler.stop() if sampler is not None else None
            env_after = capture_env(llama_bin=llama_bin, out_path=out / rel["env_after"], gpu_lock_held=gpu_lock_held)
            for phase, rec in (("before", env_before), ("after", env_after)):
                env_flags.extend({"job_id": jid, "phase": phase, "reason": why} for why in env_problems(rec, platform))
                if mode == "formal":
                    env_flags.extend({"job_id": jid, "phase": phase, "reason": why}
                                     for why in provenance_problems(rec, preflight))
                if mode == "formal" and platform == "macos-metal":  # §4: one powermode for the whole session
                    pm = ((rec or {}).get("power") or {}).get("powermode")
                    if pm != session["powermode"]:
                        env_flags.append({"job_id": jid, "phase": phase,
                                          "reason": f"powermode {pm!r} != preflight {session['powermode']!r} (§4)"})

            stdout = (out / rel["stdout"]).read_text(errors="replace")
            stderr = (out / rel["stderr"]).read_text(errors="replace")
            external = None
            if ext_kind == "time-l":
                stderr, external = cap.split_time_l(stderr)
            elif sampled is not None:
                external = {**sampled, "samples": rel["gpu_mem_samples"]}
            if job["kind"] == "fit":
                result = cap.classify_completion(exit_code, cap.parse_log_jsonl(stdout), stderr,
                                                 job["expect"]["prompt_tokens"])
            else:
                result = cap.classify_bench(exit_code, stdout, stderr, {**bench_expect, **job["expect"]})

            row = {
                "job_id": jid, "kind": job["kind"], "position": job.get("position"), "block": job.get("block"),
                "size": job["size"], "quant": job["quant"], "L": job["L"], "rep": job["rep"], "model": job["model"],
                "argv": argv, "exit_code": exit_code, "wall_s": wall_s,
                "outcome": result["outcome"], "stage": result["stage"], "reason": result["reason"],
                "evidence": result["evidence"], "metrics": result.get("metrics"), "summary": result.get("summary"),
                "external_memory": external,
                **rel,
            }
            rows.append(row)
            rows_file.write(json.dumps(row) + "\n")
            rows_file.flush()

    outcomes = {"ok": 0, "oom": 0, "error": 0}
    for row in rows:
        outcomes[row["outcome"]] += 1
    session.update({"finished_at": now(), "outcomes": outcomes, "sentinel": _sentinel_summary(rows),
                    "env_flags": env_flags, "env_valid": not env_flags, "charge_pauses": charge_pauses})
    write_raw_manifest(out)
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
    measure_b = None
    if plan["platform"] == "wsl2-cuda" and plan["mode"] == "formal":
        import stage_b  # imported here: stage_b itself imports this module

        def measure_b(frozen_b_mib: float) -> dict:  # §4 step 4 (018)
            return stage_b.measure_baseline(query=stage_b.nvidia_query, compute_apps=stage_b.nvidia_compute_apps,
                                            sleep=time.sleep, frozen_b_mib=frozen_b_mib)
    try:
        summary = run_session(
            plan, bin_dir=Path(args.bin).resolve(), models_dir=Path(args.models).resolve(), out_dir=args.out,
            capture_env=make_env_capture(script), session_id=args.session_id, gpu_lock_held=args.gpu_lock_held,
            mac_lock_held=os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") == "1",
            gpu_sampler=NvidiaSmiSampler if plan["platform"] == "wsl2-cuda" else None, measure_b=measure_b,
        )
    except RunError as exc:
        print(f"run_core_a: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary.get(k) for k in ("session_id", "outcomes", "sentinel", "env_valid", "powermode")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
