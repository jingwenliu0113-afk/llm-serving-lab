#!/usr/bin/env python3
"""Core B runner (decision 007; PREREGISTRATION §3 / §6). Runs in WSL with the project venv,
only after the S1-S3 smoke gate (008) has passed.

For each run of a plan (core_b_plan.py): start one server (shared config), check the
GPU is idle and the shared GPU lock is not held by another project, read
vllm:cache_config_info and require the frozen num_gpu_blocks / block_size, then run
`vllm bench serve` for every C in the run's stored order, sampling /metrics
(KV cache usage peak, preemptions, running / waiting) around and during each C.
Every result is checked (completed = num_prompts, no failures, exact input / output
lengths per mix). Teardown signals the server's whole process group (coreb_smoke).

Formal runs require freeze.nvidia = true and running under
`flock -n /tmp/ai-portfolio-gpu.lock` (--gpu-lock-held). raw/ (bench JSON with
per-request data, server logs) stays local; raw_manifest.json pins it.

Usage (WSL, ~/llm-serving-lab):
    .venv/bin/python benchmarks/core_b_plan.py uniform --num-gpu-blocks <N> --block-size <B> --out uniform.json
    .venv/bin/python benchmarks/run_core_b.py --plan uniform.json --out results/core_b/<uniform> \\
        --gdn-prefill-backend triton --gpu-memory-utilization 0.90 [--gpu-lock-held --formal]
    # H-B4 (007): `mixed` at each level's uniform C_knee, then the report (core_b_report.py)
    .venv/bin/python benchmarks/core_b_plan.py mixed --uniform-session results/core_b/<uniform> --out mixed.json
    .venv/bin/python benchmarks/run_core_b.py --plan mixed.json --out results/core_b/<mixed> ... \\
        --mixed-dataset 2048=data/core_b/mixed-2048.jsonl --mixed-dataset 8192=data/core_b/mixed-8192.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Callable

import core_b_analysis as ba
import core_b_plan as bp
import coreb_smoke as smoke
from run_core_a import sha256_json, write_raw_manifest

SCHEMA_VERSION = 1
PORT = 8000
IDLE_MIB = 1000
_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+([-+0-9.eEinfNa]+)\s*$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


class RunError(RuntimeError):
    pass


def parse_prometheus(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m or line.startswith("#"):
            continue
        labels = dict(_LABEL.findall(m.group(3) or ""))
        out.setdefault(m.group(1), []).append((labels, float(m.group(4))))
    return out


def cache_config(metrics: dict) -> dict | None:
    for labels, _ in metrics.get("vllm:cache_config_info", []):
        if "num_gpu_blocks" in labels and "block_size" in labels:
            return {"num_gpu_blocks": int(labels["num_gpu_blocks"]), "block_size": int(labels["block_size"])}
    return None


def gauge(metrics: dict, name: str) -> float | None:
    vals = [v for _, v in metrics.get(name, [])]
    return sum(vals) if vals else None


class MetricsSampler:
    """Polls /metrics while a bench runs; interval None = sample only at start and stop."""

    def __init__(self, http_get: Callable, url: str, interval: float | None):
        self.http_get, self.url, self.interval = http_get, url, interval
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def _sample(self):
        try:
            code, text = self.http_get(self.url)
            if code == 200:
                m = parse_prometheus(text)
                self.samples.append({k: gauge(m, k) for k in ("vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
                                                              "vllm:num_requests_running", "vllm:num_requests_waiting")})
        except OSError:
            pass

    def start(self):
        self._sample()
        if self.interval:
            def loop():
                while not self._stop.wait(self.interval):
                    self._sample()
            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._sample()
        col = lambda k: [s[k] for s in self.samples if s.get(k) is not None]  # noqa: E731
        pre = col("vllm:num_preemptions_total")
        return {"kv_cache_usage_peak": max(col("vllm:kv_cache_usage_perc"), default=None),
                "preemptions": (pre[-1] - pre[0]) if len(pre) >= 2 else None,
                "running_peak": max(col("vllm:num_requests_running"), default=None),
                "waiting_peak": max(col("vllm:num_requests_waiting"), default=None),
                "n_samples": len(self.samples)}


METRIC_KEYS = [f"{s}_{m}_ms" for m in ("ttft", "tpot", "itl", "e2el", "client_queue_time")
               for s in ("mean", "median", "p50", "p95", "p99")]


def summarize_result(res: dict, *, L: int, C: int, mix: str) -> dict:
    n = bp.num_prompts(C)
    reasons = []
    if res.get("completed") != n or res.get("failed"):
        reasons.append(f"completed={res.get('completed')} failed={res.get('failed')} (expected {n} / 0)")
    lens, outs = res.get("input_lens") or [], res.get("output_lens") or []
    if mix == "uniform":
        want = [L - bp.N_GEN] * n
    else:
        short, long_ = bp.mixed_counts(n)
        want = sorted([L - bp.N_GEN] * short + [bp.LONG_INPUT] * long_)
    if sorted(lens) != sorted(want):
        reasons.append(f"input_lens do not match the {mix} mix at L={L} ({len(lens)} values)")
    if outs != [bp.N_GEN] * n:
        reasons.append(f"output_lens not all {bp.N_GEN}")
    short_p95 = None
    if lens and res.get("ttfts") and len(lens) == len(res["ttfts"]):
        try:
            short_p95 = ba.short_class_p95_ttft_ms({"input_lens": lens, "ttfts": res["ttfts"]}, L=L)
        except (IndexError, ValueError):
            short_p95 = None
    return {"ok": not reasons, "reasons": reasons, "num_prompts": n,
            **{k: res.get(k) for k in ("completed", "failed", "duration", "request_throughput", "output_throughput",
                                       "total_token_throughput", *METRIC_KEYS)},
            "short_class_p95_ttft_ms": short_p95}


def default_http_get(url: str, timeout: float = 5) -> tuple:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except OSError as exc:
        return 0, str(exc)


def default_run_bench(argv: list, log_path, env: dict) -> int:
    with open(log_path, "w") as log:
        return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def run_session(plan: dict, *, pins: dict, out_dir, gdn_prefill_backend: str, gpu_memory_utilization: str,
                session_id: str, popen: Callable, signal_group: Callable, group_alive: Callable,
                http_get: Callable, run_bench: Callable, gpu_mem_used: Callable, gpu_lock_free: Callable,
                gpu_lock_held: bool = False, formal: bool = False, vllm_bin: str = ".venv/bin/vllm",
                mixed_datasets: dict | None = None, sample_interval: float | None = None,
                now: Callable = time.monotonic, sleep: Callable = time.sleep, startup_timeout_s: float = 1800,
                clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat()) -> dict:
    if formal and not gpu_lock_held:
        raise RunError("formal Core B runs must hold the GPU lock: flock -n /tmp/ai-portfolio-gpu.lock ... "
                       "--gpu-lock-held (MACHINE_POLICY)")
    if formal and not pins["freeze"]["nvidia"]:
        raise RunError("formal Core B runs require freeze.nvidia = true (decision 006)")
    mixed_sha = {}
    for L in sorted({r["L"] for r in plan["runs"] if r["mix"] == "mixed"}):
        path = (mixed_datasets or {}).get(L)
        if not path or not Path(path).is_file():
            raise RunError(f"mixed runs at L={L} need --mixed-dataset {L}=<file> (decision 007)")
        mixed_sha[str(L)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        pinned = (pins["core_b"].get("mix_dataset_sha256") or {}).get(str(L))
        if formal and mixed_sha[str(L)] != pinned:
            raise RunError(f"mixed dataset for L={L}: sha256 {mixed_sha[str(L)]} != pinned {pinned!r} "
                           "(core_b.mix_dataset_sha256)")
    out = Path(out_dir)
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten")
    (out / "raw").mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "formal": formal,
               "gpu_lock_held": gpu_lock_held, "plan_sha256": sha256_json(plan), "started_at": clock(),
               "server": {"gdn_prefill_backend": gdn_prefill_backend, "gpu_memory_utilization": gpu_memory_utilization},
               "mixed_dataset_sha256": mixed_sha}
    rows = []
    base = f"http://127.0.0.1:{PORT}"
    with open(out / "rows.jsonl", "w") as fh:
        def emit(row):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()

        for run in plan["runs"]:
            L, mix, restart = run["L"], run["mix"], run["restart"]
            tag = f"{mix}-{L}-r{restart}"
            if not gpu_lock_held and not gpu_lock_free():
                raise RunError(f"{smoke.GPU_LOCK} is held by another project; stopping (MACHINE_POLICY)")
            mem0 = gpu_mem_used()
            if mem0 > IDLE_MIB:
                raise RunError(f"GPU busy before {tag}: memory.used {mem0} MiB > {IDLE_MIB}")
            argv, env = bp.server_argv(pins, gdn_prefill_backend=gdn_prefill_backend,
                                       gpu_memory_utilization=gpu_memory_utilization, port=PORT, vllm_bin=vllm_bin)
            proc = popen(argv, {**os.environ, **env}, out / "raw" / f"server-{tag}.log")
            stop_reasons = []
            try:
                t0 = now()
                ready = False
                while now() - t0 < startup_timeout_s and proc.poll() is None:
                    if http_get(f"{base}/health")[0] == 200:
                        ready = True
                        break
                    sleep(2)
                code, text = http_get(f"{base}/metrics") if ready else (0, "")
                cfg = cache_config(parse_prometheus(text)) if code == 200 else None
                want = {"num_gpu_blocks": plan["num_gpu_blocks"], "block_size": plan["block_size"]}
                if not ready:
                    stop_reasons.append("server not ready")
                elif cfg != want:
                    stop_reasons.append(f"num_gpu_blocks / block_size {cfg} != frozen {want}")
                for C in run["order"]:
                    row = {"L": L, "mix": mix, "restart": restart, "C": C, "server_cache_config": cfg}
                    if stop_reasons:
                        emit({**row, "ok": False, "reasons": list(stop_reasons)})
                        continue
                    name = f"bench-{tag}-c{C}.json"
                    bargv = bp.bench_argv(pins, L=L, C=C, mix=mix, result_dir=str(out / "raw"), result_filename=name,
                                          port=PORT, vllm_bin=vllm_bin,
                                          mixed_dataset=(mixed_datasets or {}).get(L) if mix == "mixed" else None)
                    sampler = MetricsSampler(http_get, f"{base}/metrics", sample_interval)
                    sampler.start()
                    rc = run_bench(bargv, out / "raw" / f"bench-{tag}-c{C}.log",
                                   {**os.environ, **bp.bench_env()})
                    sampled = sampler.stop()
                    path = out / "raw" / name
                    if rc != 0 or not path.is_file():
                        emit({**row, "ok": False, "reasons": [f"vllm bench serve exit {rc}"], "metrics_sampled": sampled})
                        continue
                    emit({**row, **summarize_result(json.loads(path.read_text()), L=L, C=C, mix=mix),
                          "metrics_sampled": sampled, "argv": bargv, "result": f"raw/{name}"})
            finally:
                teardown = smoke._shutdown(proc, signal_group=signal_group, group_alive=group_alive, now=now,
                                           sleep=sleep)
            if teardown:
                session.setdefault("teardown_problems", []).append({"run": tag, "reasons": teardown})

    session.update({"finished_at": clock(), "outcomes": {"ok": sum(r["ok"] for r in rows),
                                                         "error": sum(not r["ok"] for r in rows)}})
    write_raw_manifest(out)
    (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    return session


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gdn-prefill-backend", required=True, choices=["triton", "flashinfer"])
    parser.add_argument("--gpu-memory-utilization", required=True)
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--mixed-dataset", action="append", default=[], help="L=path (decision 007)")
    parser.add_argument("--gpu-lock-held", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, smoke._exit_on_signal)
    try:
        summary = run_session(
            json.loads(Path(args.plan).read_text()), pins=pins, out_dir=args.out,
            gdn_prefill_backend=args.gdn_prefill_backend, gpu_memory_utilization=args.gpu_memory_utilization,
            session_id=Path(args.out).name, popen=smoke.start_server, signal_group=smoke.signal_group,
            group_alive=smoke.group_alive, http_get=default_http_get, run_bench=default_run_bench,
            gpu_mem_used=smoke.nvidia_mem_used, gpu_lock_free=smoke.gpu_lock_free, gpu_lock_held=args.gpu_lock_held,
            formal=args.formal, vllm_bin=args.vllm_bin, sample_interval=1.0,
            mixed_datasets={int(k): v for k, v in (x.split("=", 1) for x in args.mixed_dataset)},
        )
    except RunError as exc:
        print(f"run_core_b: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary["outcomes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
