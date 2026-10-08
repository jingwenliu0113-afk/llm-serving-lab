#!/usr/bin/env python3
"""Stage B measurements on the RTX (PREREGISTRATION §4 / §6; decisions 003, 004, 008, 009).

Every subcommand runs in WSL under `flock -n /tmp/ai-portfolio-gpu.lock` (--gpu-lock-held is
required) and refuses while another process uses the GPU. Results are new JSON files; nothing is
ever overwritten.

  baseline  B (§4): idle 5 min after the last GPU process, then nvidia-smi memory.used at 1 Hz for
            60 s -> median. Also records V_total, display_active / display_mode (is a monitor on this
            card), HAGS (Windows registry, read-only) and .wslconfig.
  cctx      C_ctx (004 / 009): peak memory.used during the 4B-Q4_K_M @ 2048 fit run - B - that run's
            device self from the llama.cpp memory breakdown.
  overflow  sysmem test (004 / 008): 9B-Q8_0 with -c 262144 (outside the 18 cells; N ~ 15.9 GiB > 16 GB
            for any B). The KV is allocated at context creation, so a 1024-token prompt shows whether
            WSL2 hard-OOMs or spills to system memory.

Usage (WSL, ~/llm-serving-lab):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/stage_b.py baseline --gpu-lock-held \\
        --out results/stage_b/<date>/baseline.json
    flock -n ... benchmarks/stage_b.py cctx --gpu-lock-held --b-mib <B> --bin <llama.cpp bin> --models <gguf dir> \\
        --n-thr <N_thr> --poll <P> --out results/stage_b/<date>/cctx
    flock -n ... benchmarks/stage_b.py overflow --gpu-lock-held --b-mib <B> ... --out results/stage_b/<date>/overflow

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable

import core_a_parse as cap
import core_a_plan as cp
from run_core_a import NvidiaSmiSampler, _resolve, default_run_proc, make_env_capture

IDLE_S = 300
IDLE_POLL_S = 5          # GPU processes are checked throughout the idle wait (a new one restarts it)
MAX_IDLE_WAIT_S = 3600   # give up when the GPU never stays quiet for IDLE_S
N_SAMPLES = 60
DRIFT_MIB = 256          # §4 step 4: > 0.25 GiB above the frozen B marks the session drift
CCTX_CELL = ("4B", "Q4_K_M", 2048)  # 009
OVERFLOW_CELL = ("9B", "Q8_0")
OVERFLOW_CTX = 262144
OVERFLOW_PROMPT_TOKENS = 1024
GPU_QUERY = "memory.used,memory.total,display_active,display_mode"


class StageBError(RuntimeError):
    pass


def parse_gpu_query(text: str) -> dict:
    line = next((x for x in text.splitlines() if x.strip()), "")
    parts = [p.strip() for p in line.split(",")]
    try:
        used, total = int(parts[0]), int(parts[1])
    except (ValueError, IndexError) as exc:
        raise StageBError(f"unexpected nvidia-smi output: {line!r}") from exc
    return {"memory_used_mib": used, "memory_total_mib": total,
            "display_active": parts[2] if len(parts) > 2 else None,
            "display_mode": parts[3] if len(parts) > 3 else None}


def parse_compute_apps(text: str) -> list:
    apps = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        apps.append({"pid": int(parts[0]), "name": parts[1],
                     "used_mib": int(parts[2]) if parts[2].isdigit() else None})
    return apps


def nvidia_query() -> dict:  # pragma: no cover - needs nvidia-smi
    out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return parse_gpu_query(out)


def nvidia_compute_apps() -> list:  # pragma: no cover - needs nvidia-smi
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    return parse_compute_apps(out)


def windows_gpu_settings() -> dict:  # pragma: no cover - needs WSL interop
    """Read-only: HAGS (HwSchMode: 2 = on, 1 = off) and the user's .wslconfig. None = not read (see probe_errors);
    wslconfig "absent" = no such file."""
    out: dict = {"hags": None, "wslconfig": None, "probe_errors": []}
    try:
        r = subprocess.run(["reg.exe", "query", r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
                            "/v", "HwSchMode"], capture_output=True, text=True, timeout=30)
        vals = [line for line in r.stdout.splitlines() if "HwSchMode" in line]
        if vals:
            out["hags"] = str(int(vals[-1].split()[-1], 16))
        else:
            out["probe_errors"].append(f"reg.exe rc={r.returncode}: {r.stderr.strip()[:200]}")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        out["probe_errors"].append(f"reg.exe: {exc!r}")
    try:
        home = subprocess.run(["cmd.exe", "/c", "echo %USERPROFILE%"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
        path = subprocess.run(["wslpath", "-u", home], capture_output=True, text=True, timeout=30).stdout.strip()
        if not home or not path:
            raise ValueError(f"cannot resolve %USERPROFILE%: {home!r} -> {path!r}")
        cfg = Path(path) / ".wslconfig"
        out["wslconfig"] = cfg.read_text(errors="replace") if cfg.is_file() else "absent"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        out["probe_errors"].append(f"wslconfig: {exc!r}")
    return out


def measure_baseline(*, query: Callable, compute_apps: Callable, sleep: Callable, idle_s: int = IDLE_S,
                     n_samples: int = N_SAMPLES, frozen_b_mib: float | None = None) -> dict:
    """§4: no other GPU process; B is sampled only after IDLE_S without any GPU process (checked every
    IDLE_POLL_S; a process restarts the wait); a process during the 1 Hz sampling makes B invalid."""
    apps = compute_apps()
    if apps:
        raise StageBError(f"another GPU process is running, not measuring B: {apps}")
    idle = waited = 0
    while idle < idle_s:
        if waited >= MAX_IDLE_WAIT_S:
            raise StageBError(f"the GPU never stayed free of other processes for {idle_s} s within {MAX_IDLE_WAIT_S} s")
        sleep(IDLE_POLL_S)
        waited += IDLE_POLL_S
        idle = 0 if compute_apps() else idle + IDLE_POLL_S
    samples, info = [], None
    for i in range(n_samples):
        apps = compute_apps()
        if apps:
            raise StageBError(f"a GPU process appeared during the measurement, B is invalid: {apps}")
        info = query()
        samples.append(info["memory_used_mib"])
        if i < n_samples - 1:
            sleep(1)
    rec = {"b_mib": statistics.median(samples), "samples_mib": samples, "idle_s": idle_s, "idle_waited_s": waited,
           "v_total_mib": info["memory_total_mib"], "display_active": info["display_active"],
           "display_mode": info["display_mode"]}
    if frozen_b_mib is not None:
        rec.update({"frozen_b_mib": frozen_b_mib, "drift": rec["b_mib"] - frozen_b_mib > DRIFT_MIB})
    return rec


def device_self_mib(summary: dict) -> int | None:
    memory = (summary or {}).get("memory")
    if not memory:
        return None
    return sum(d.get("self", 0) for d in memory.get("devices", []))


def c_ctx_mib(samples_mib: list, *, b_mib: float, self_mib: float) -> float:
    return max(samples_mib) - b_mib - self_mib


def classify_overflow(result: dict, *, self_mib, v_total_mib: float, b_mib: float) -> str:
    if result["outcome"] == "oom":
        return "hard-oom"
    if result["outcome"] != "ok" or self_mib is None:
        return "error"
    return "spill" if self_mib + b_mib > v_total_mib else "fit"


def _model(pins: dict, size: str, quant: str) -> str:
    m = next(m for m in pins["core_a"]["models"] if m["size"] == size)
    return m["files"][quant]["filename"]


def cctx_job(*, pins: dict, n_thr: int, poll: int) -> dict:
    size, quant, L = CCTX_CELL
    return {"argv": cp.completion_argv(_model(pins, size, quant), L, n_thr, poll), "prompt_tokens": L - cp.N_GEN}


def overflow_job(*, pins: dict, n_thr: int, poll: int) -> dict:
    return {"argv": cp.completion_argv(_model(pins, *OVERFLOW_CELL), OVERFLOW_CTX, n_thr, poll),
            "prompt_tokens": OVERFLOW_PROMPT_TOKENS}


def run_fit_job(job: dict, *, out: Path, bin_dir: str, models_dir: str, capture_env: Callable,
                run_proc: Callable = default_run_proc, sampler_cls=NvidiaSmiSampler, timeout_s: float = 3600) -> dict:
    out.mkdir(parents=True)
    prompt = out / "prompt.txt"
    prompt.write_text(cp.PROMPT_UNIT * job["prompt_tokens"])
    argv = _resolve(job["argv"], bin_dir, models_dir, str(prompt))
    capture_env(llama_bin=f"{bin_dir}/llama-completion", out_path=out / "env.before.json", gpu_lock_held=True)
    sampler = sampler_cls(out / "gpumem.txt")
    sampler.start()
    try:
        exit_code, wall_s = run_proc(argv, out / "stdout.txt", out / "stderr.txt", timeout_s)
    finally:
        sampler.stop()
    capture_env(llama_bin=f"{bin_dir}/llama-completion", out_path=out / "env.after.json", gpu_lock_held=True)
    stdout = (out / "stdout.txt").read_text(errors="replace")
    stderr = (out / "stderr.txt").read_text(errors="replace")
    result = cap.classify_completion(exit_code, cap.parse_log_jsonl(stdout), stderr, job["prompt_tokens"])
    samples = [int(x) for x in (out / "gpumem.txt").read_text().split() if x.strip().isdigit()] \
        if (out / "gpumem.txt").exists() else []
    return {"argv": argv, "exit_code": exit_code, "wall_s": wall_s, "samples_mib": samples,
            **{k: result.get(k) for k in ("outcome", "stage", "reason", "evidence", "summary")}}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("baseline", "cctx", "overflow"):
        p = sub.add_parser(name)
        p.add_argument("--gpu-lock-held", action="store_true")
        p.add_argument("--out", required=True)
        if name == "baseline":
            p.add_argument("--frozen-b-mib", type=float, help="§4 step 4: mark drift when B exceeds it by > 0.25 GiB")
        if name != "baseline":
            p.add_argument("--b-mib", type=float, required=True)
            p.add_argument("--bin", required=True)
            p.add_argument("--models", required=True)
            p.add_argument("--n-thr", type=int, required=True)
            p.add_argument("--poll", type=int, required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if not args.gpu_lock_held:
            raise StageBError("Stage B steps run only under flock -n /tmp/ai-portfolio-gpu.lock (--gpu-lock-held)")
        if out.exists():
            raise StageBError(f"{out} exists; Stage B results are never overwritten")
        if args.cmd == "baseline":
            rec = measure_baseline(query=nvidia_query, compute_apps=nvidia_compute_apps, sleep=time.sleep,
                                   frozen_b_mib=args.frozen_b_mib)
            rec["windows"] = windows_gpu_settings()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=1) + "\n")
            print(json.dumps({k: rec[k] for k in ("b_mib", "v_total_mib", "display_active")}))
            return 0
        apps = nvidia_compute_apps()
        if apps:
            raise StageBError(f"another GPU process is running: {apps}")
        root = Path(__file__).resolve().parent.parent
        pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
        job = (cctx_job if args.cmd == "cctx" else overflow_job)(pins=pins, n_thr=args.n_thr, poll=args.poll)
        capture_env = make_env_capture(root / "scripts" / "env_capture.py")
        rec = run_fit_job(job, out=out, bin_dir=args.bin, models_dir=args.models, capture_env=capture_env)
        self_mib = device_self_mib(rec.get("summary"))
        v_total = nvidia_query()["memory_total_mib"]
        rec.update({"b_mib": args.b_mib, "v_total_mib": v_total, "device_self_mib": self_mib})
        if args.cmd == "cctx":
            if rec["outcome"] != "ok" or not rec["samples_mib"] or self_mib is None:
                raise StageBError(f"C_ctx fit run not usable: {rec['outcome']} {rec['reason']}")
            rec["c_ctx_mib"] = c_ctx_mib(rec["samples_mib"], b_mib=args.b_mib, self_mib=self_mib)
        else:
            rec["overflow"] = classify_overflow(rec, self_mib=self_mib, v_total_mib=v_total, b_mib=args.b_mib)
        (out / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("outcome", "c_ctx_mib", "overflow", "device_self_mib")}))
        return 0
    except StageBError as exc:
        print(f"stage_b: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
