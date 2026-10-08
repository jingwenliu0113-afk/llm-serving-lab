#!/usr/bin/env python3
"""vLLM Stage B steps (decision 003 smoke gate 3-4, details fixed in decision 017). WSL, project
venv, under `flock -n /tmp/ai-portfolio-gpu.lock` (--gpu-lock-held required); refuses while another
GPU process runs; never overwrites.

  boots   Two boots of the Core B server (core_b_plan.server_argv, U from B by the 017 rule, no
          --kv-cache-memory-bytes) with fresh Triton / FlashInfer / vLLM caches under <out>/cache:
          boot 1 is cold, boot 2 warm (same caches). Records the time from start to /health 200
          (003 #3: JIT time of the Triton / FlashInfer / vLLM caches, cache effective on the second boot; the CUDA
          driver's own PTX JIT cache is not reset, so FA2's compute_80 PTX JIT may already be cached) and K = the warm boot's
          'Available KV cache memory' floored to 256 MiB (017; the cold boot is the discarded one).
  bench-check  003 gate step 5 on the 008 smoke server (S1 settings): one uniform L = 2048, C = 4 `vllm bench
          serve` run completes and validates, and /metrics has the KV-usage / preemption / cache-config metrics.
  pin-ab  VLLM_WSL2_ENABLE_PIN_MEMORY 0 then 1 (003 #4), each one boot with U and K: uniform L = 2048,
          C = 1 and C = 16, 3 `vllm bench serve` runs each; 1 only if >= 5 % faster at both C and every
          group's run-to-run CV < 3 % (003 rule). Also records /metrics cache_config (num_gpu_blocks,
          block_size) for the Stage B plan.

Usage (WSL, ~/llm-serving-lab):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/coreb_stage_b.py boots --gpu-lock-held \\
        --b-mib <B> --v-total-mib <V> --gdn-prefill-backend triton --out results/stage_b/<date>/boots
    flock -n ... benchmarks/coreb_stage_b.py pin-ab --gpu-lock-held --gpu-memory-utilization <U> \\
        --kv-cache-memory-bytes <K> --gdn-prefill-backend triton --out results/stage_b/<date>/pin-ab
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable

import core_b_plan as bp
import coreb_smoke as smoke
import run_core_b as rb
from stage_b import nvidia_compute_apps

PORT = 8766
CACHE_VARS = ("TRITON_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "VLLM_CACHE_ROOT")
AB_L = 2048
AB_CS = (1, 16)
AB_RUNS = 3
AB_MIN_GAIN_PCT = 5.0
AB_MAX_CV_PCT = 3.0


class CoreBStageBError(RuntimeError):
    pass


def server_log_checks(text: str) -> tuple:
    """(log facts, problems): attention backend and max_num_batched_tokens from the server log (§4 / 017 / 018)."""
    facts = smoke.parse_server_log(text)
    m = rb._RE_NONDEFAULT_BATCHED.search(text)
    facts["max_num_batched_tokens_logged"] = int(m.group(1)) if m else None
    problems = []
    if facts.get("attention_backend") != "FLASH_ATTN":
        problems.append(f"attention backend {facts.get('attention_backend')!r} != FLASH_ATTN")
    if facts["max_num_batched_tokens_logged"] != bp.MAX_NUM_BATCHED_TOKENS:
        problems.append(f"max_num_batched_tokens {facts['max_num_batched_tokens_logged']!r} != {bp.MAX_NUM_BATCHED_TOKENS}")
    return facts, problems


def pin_memory_choice(results: dict, *, runs: int = AB_RUNS) -> dict:
    """results: (setting, C) -> output throughputs of the valid runs."""
    groups = [(v, C) for v in (0, 1) for C in AB_CS]
    missing = [f"{v}@C{C}" for v, C in groups if len(results.get((v, C), [])) < runs]
    if missing:
        return {"choice": 0, "reason": f"incomplete A/B (fewer than {runs} valid runs: {missing}); keep the default 0"}
    mean = {g: statistics.mean(results[g]) for g in groups}
    cv = {f"{v}@C{C}": 100 * statistics.stdev(results[(v, C)]) / mean[(v, C)] for v, C in groups}
    gain = {C: 100 * (mean[(1, C)] / mean[(0, C)] - 1) for C in AB_CS}
    ok = all(g >= AB_MIN_GAIN_PCT for g in gain.values()) and all(c < AB_MAX_CV_PCT for c in cv.values())
    return {"choice": 1 if ok else 0, "gain_pct": gain, "cv_pct": cv,
            "reason": "1 is >= 5 % faster at C = 1 and 16 with CV < 3 %" if ok else "rule not met; keep the default 0"}


def _precheck(env: dict, *, toolchain_check: Callable, compute_apps: Callable, out: Path) -> None:
    if out.exists():
        raise CoreBStageBError(f"{out} exists; Stage B results are never overwritten")
    apps = compute_apps()
    if apps:
        raise CoreBStageBError(f"another GPU process is running, not starting: {apps}")
    missing = toolchain_check(env)
    if missing:
        raise CoreBStageBError("toolchain incomplete (014): " + "; ".join(missing))


def _boot(argv, env, log_path, *, popen, http_get, now, sleep, timeout_s: float, signal_group: Callable,
          group_alive: Callable) -> tuple:
    """Start the server and wait for /health. Any exception during the wait (SIGTERM / SIGHUP via
    _exit_on_signal, Ctrl-C) tears the server down before propagating: the callers never get `proc` back
    then, and a server left running would hold the GPU after the lock is released."""
    t0 = now()
    proc = popen(argv, env, log_path)
    try:
        while now() - t0 < timeout_s and proc.poll() is None:
            if http_get(f"http://127.0.0.1:{PORT}/health")[0] == 200:
                return proc, now() - t0
            sleep(2)
        return proc, None
    except BaseException:
        smoke._shutdown(proc, signal_group=signal_group, group_alive=group_alive, now=now, sleep=sleep)
        raise


def run_boots(pins: dict, *, out: Path, v_total_mib: float, b_mib: float, gdn_prefill_backend: str,
              popen: Callable, signal_group: Callable, group_alive: Callable, http_get: Callable, now: Callable,
              sleep: Callable, toolchain_check: Callable = bp.toolchain_problems,
              compute_apps: Callable = nvidia_compute_apps, base_env=None, vllm_bin: str = ".venv/bin/vllm",
              startup_timeout_s: float = 3600) -> dict:
    out = Path(out)
    env = bp.server_env(dict(os.environ if base_env is None else base_env))
    env.update({var: str(out / "cache" / var.lower()) for var in CACHE_VARS})
    _precheck(env, toolchain_check=toolchain_check, compute_apps=compute_apps, out=out)
    (out / "cache").mkdir(parents=True)
    util = bp.gpu_memory_utilization(v_total_mib=v_total_mib, b_mib=b_mib)
    argv, _ = bp.server_argv(pins, gdn_prefill_backend=gdn_prefill_backend, gpu_memory_utilization=util, port=PORT,
                             vllm_bin=vllm_bin)
    boots = []
    for i, label in enumerate(("cold", "warm")):
        log = out / f"server-{label}.log"
        proc, startup_s = _boot(argv, env, log, popen=popen, http_get=http_get, now=now, sleep=sleep,
                                timeout_s=startup_timeout_s, signal_group=signal_group,
                                group_alive=group_alive)
        teardown = smoke._shutdown(proc, signal_group=signal_group, group_alive=group_alive, now=now, sleep=sleep)
        text = log.read_text(errors="replace") if log.exists() else ""
        boots.append({"boot": label, "startup_s": startup_s, "teardown_problems": teardown,
                      "log_facts": smoke.parse_server_log(text), "kv_cache_bytes": bp.kv_cache_bytes_from_log(text),
                      "log": log.name})
        if startup_s is None:
            break
    warm = boots[1] if len(boots) == 2 and boots[1]["startup_s"] is not None else None
    rec = {"v_total_mib": v_total_mib, "b_mib": b_mib, "gpu_memory_utilization": util, "argv": argv,
           "server_env": bp.recorded_env(env), "caches": {v: env[v] for v in CACHE_VARS}, "boots": boots,
           "cache_effective": bool(warm and warm["startup_s"] < boots[0]["startup_s"]),
           # the CUDA driver's own JIT cache (FA2 compute_80 PTX -> sm_120) is not reset: CUDA_CACHE_PATH is not
           # among the §4-listed variables, so the cold boot may reuse PTX JIT results of earlier boots
           "driver_jit_cache": "shared default (CUDA_CACHE_PATH unset, not reset; earlier boots may have filled it)",
           "kv_cache_memory_bytes": warm["kv_cache_bytes"] if warm else None}
    (out / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def run_pin_ab(pins: dict, *, out: Path, gpu_memory_utilization: str, kv_cache_memory_bytes: int,
               gdn_prefill_backend: str, popen: Callable, signal_group: Callable, group_alive: Callable,
               http_get: Callable, run_bench: Callable, now: Callable, sleep: Callable,
               toolchain_check: Callable = bp.toolchain_problems, compute_apps: Callable = nvidia_compute_apps,
               base_env=None, vllm_bin: str = ".venv/bin/vllm", startup_timeout_s: float = 3600) -> dict:
    out = Path(out)
    base = bp.server_env(dict(os.environ if base_env is None else base_env))
    _precheck(base, toolchain_check=toolchain_check, compute_apps=compute_apps, out=out)
    (out / "raw").mkdir(parents=True)
    argv, _ = bp.server_argv(pins, gdn_prefill_backend=gdn_prefill_backend,
                             gpu_memory_utilization=gpu_memory_utilization, port=PORT, vllm_bin=vllm_bin,
                             kv_cache_memory_bytes=kv_cache_memory_bytes)
    results, runs, cache_cfg, facts, checks = {}, [], {}, {}, {}
    for v in (0, 1):
        env = {**base, "VLLM_WSL2_ENABLE_PIN_MEMORY": str(v)}
        proc, startup_s = _boot(argv, env, out / "raw" / f"server-pin{v}.log", popen=popen, http_get=http_get,
                                now=now, sleep=sleep, timeout_s=startup_timeout_s, signal_group=signal_group,
                                group_alive=group_alive)
        try:
            if startup_s is None:
                runs.append({"setting": v, "ok": False, "reasons": ["server not ready"]})
                continue
            log = out / "raw" / f"server-pin{v}.log"
            facts[str(v)], checks[str(v)] = server_log_checks(log.read_text(errors="replace") if log.exists() else "")
            code, text = http_get(f"http://127.0.0.1:{PORT}/metrics")
            cache_cfg[str(v)] = rb.cache_config(rb.parse_prometheus(text)) if code == 200 else None
            for C in AB_CS:
                for r in range(1, AB_RUNS + 1):
                    name = f"bench-pin{v}-c{C}-r{r}.json"
                    bargv = bp.bench_argv(pins, L=AB_L, C=C, mix="uniform", result_dir=str(out / "raw"),
                                          result_filename=name, port=PORT, vllm_bin=vllm_bin)
                    rc = run_bench(bargv, out / "raw" / f"bench-pin{v}-c{C}-r{r}.log", {**os.environ, **bp.bench_env()})
                    path = out / "raw" / name
                    summary = (rb.summarize_result(json.loads(path.read_text()), L=AB_L, C=C, mix="uniform")
                               if rc == 0 and path.is_file() else {"ok": False, "reasons": [f"bench exit {rc}"]})
                    runs.append({"setting": v, "C": C, "run": r, **summary})
                    if summary["ok"]:
                        results.setdefault((v, C), []).append(summary["output_throughput"])
        finally:
            smoke._shutdown(proc, signal_group=signal_group, group_alive=group_alive, now=now, sleep=sleep)
    rec = {"gpu_memory_utilization": gpu_memory_utilization, "kv_cache_memory_bytes": kv_cache_memory_bytes,
           "argv": argv, "server_env": bp.recorded_env(base), "L": AB_L, "C": list(AB_CS), "runs_per_point": AB_RUNS,
           "cache_config": cache_cfg, "log_facts": facts, "server_checks": checks, "runs": runs,
           "decision": pin_memory_choice(results) if not any(checks.values()) else
           {"choice": 0, "reason": f"server log checks failed: {checks}; keep the default 0"}}
    rb.write_raw_manifest(out)
    (out / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


BENCH_CHECK_L, BENCH_CHECK_C = 2048, 4
REQUIRED_METRICS = ("vllm:kv_cache_usage_perc", "vllm:num_preemptions_total", "vllm:cache_config_info")


def run_bench_check(pins: dict, *, out: Path, gdn_prefill_backend: str, popen: Callable, signal_group: Callable,
                    group_alive: Callable, http_get: Callable, run_bench: Callable, now: Callable, sleep: Callable,
                    toolchain_check: Callable = bp.toolchain_problems, compute_apps: Callable = nvidia_compute_apps,
                    base_env=None, vllm_bin: str = ".venv/bin/vllm", startup_timeout_s: float = 3600) -> dict:
    """003 gate step 5: `vllm bench serve` completes and writes its JSON, and /metrics exposes the KV-usage,
    preemption and cache-config metrics (names used by run_core_b). Runs on the 008 smoke server (S1 settings:
    --max-model-len 2048, --max-num-seqs 8, --gpu-memory-utilization 0.85): a tool check, not a measurement."""
    out = Path(out)
    argv, env = smoke.serve_command(vllm_bin=vllm_bin, model=pins["core_b"]["repo"], revision=pins["core_b"]["revision"],
                                    stage=smoke.STAGES[0], gdn_prefill_backend=gdn_prefill_backend, port=PORT,
                                    base_env=dict(os.environ if base_env is None else base_env))
    _precheck(env, toolchain_check=toolchain_check, compute_apps=compute_apps, out=out)
    (out / "raw").mkdir(parents=True)
    proc, startup_s = _boot(argv, env, out / "raw" / "server.log", popen=popen, http_get=http_get, now=now,
                            sleep=sleep, timeout_s=startup_timeout_s, signal_group=signal_group,
                                group_alive=group_alive)
    bench, found, sampled = {"ok": False, "reasons": ["server not ready"]}, {}, None
    try:
        if startup_s is not None:
            name = "bench-check.json"
            bargv = bp.bench_argv(pins, L=BENCH_CHECK_L, C=BENCH_CHECK_C, mix="uniform", result_dir=str(out / "raw"),
                                  result_filename=name, port=PORT, vllm_bin=vllm_bin)
            sampler = rb.MetricsSampler(http_get, f"http://127.0.0.1:{PORT}/metrics", 1.0)
            sampler.start()
            rc = run_bench(bargv, out / "raw" / "bench-check.log", {**os.environ, **bp.bench_env()})
            sampled = sampler.stop()
            path = out / "raw" / name
            bench = (rb.summarize_result(json.loads(path.read_text()), L=BENCH_CHECK_L, C=BENCH_CHECK_C, mix="uniform")
                     if rc == 0 and path.is_file() else {"ok": False, "reasons": [f"bench exit {rc}"]})
            code, text = http_get(f"http://127.0.0.1:{PORT}/metrics")
            names = {line.split("{")[0].split(" ")[0] for line in text.splitlines()
                     if line and not line.startswith("#")} if code == 200 else set()
            found = {m: m in names for m in REQUIRED_METRICS}
    finally:
        teardown = smoke._shutdown(proc, signal_group=signal_group, group_alive=group_alive, now=now, sleep=sleep)
    rec = {"argv": argv, "server_env": bp.recorded_env(env), "startup_s": startup_s, "teardown_problems": teardown,
           "bench": bench, "metrics_found": found, "metrics_sampled": sampled,
           "gate_step5_ok": bool(bench.get("ok") and found and all(found.values()))}
    rb.write_raw_manifest(out)
    (out / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("boots")
    p.add_argument("--b-mib", type=float, required=True)
    p.add_argument("--v-total-mib", type=float, required=True)
    c = sub.add_parser("bench-check")
    q = sub.add_parser("pin-ab")
    q.add_argument("--gpu-memory-utilization", required=True)
    q.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    for x in (p, q, c):
        x.add_argument("--gpu-lock-held", action="store_true")
        x.add_argument("--gdn-prefill-backend", required=True, choices=["triton", "flashinfer"])
        x.add_argument("--vllm-bin", default=".venv/bin/vllm")
        x.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if not args.gpu_lock_held:
            raise CoreBStageBError("Stage B steps run only under flock -n /tmp/ai-portfolio-gpu.lock (--gpu-lock-held)")
        pins = tomllib.loads((Path(__file__).resolve().parent.parent / "configs" / "pins.toml").read_text())
        for sig in (signal.SIGHUP, signal.SIGTERM):
            signal.signal(sig, smoke._exit_on_signal)
        common = dict(gdn_prefill_backend=args.gdn_prefill_backend, popen=smoke.start_server,
                      signal_group=smoke.signal_group, group_alive=smoke.group_alive, http_get=rb.default_http_get,
                      now=time.monotonic, sleep=time.sleep, vllm_bin=args.vllm_bin)
        if args.cmd == "boots":
            rec = run_boots(pins, out=Path(args.out), v_total_mib=args.v_total_mib, b_mib=args.b_mib, **common)
            print(json.dumps({k: rec[k] for k in ("gpu_memory_utilization", "cache_effective", "kv_cache_memory_bytes")}
                             | {"startup_s": [b["startup_s"] for b in rec["boots"]]}))
        elif args.cmd == "bench-check":
            rec = run_bench_check(pins, out=Path(args.out), run_bench=rb.default_run_bench, **common)
            print(json.dumps({k: rec[k] for k in ("gate_step5_ok", "metrics_found", "startup_s")}
                             | {"bench_ok": rec["bench"].get("ok"), "reasons": rec["bench"].get("reasons")}))
        else:
            rec = run_pin_ab(pins, out=Path(args.out), gpu_memory_utilization=args.gpu_memory_utilization,
                             kv_cache_memory_bytes=args.kv_cache_memory_bytes, run_bench=rb.default_run_bench, **common)
            print(json.dumps({"decision": rec["decision"], "cache_config": rec["cache_config"]}))
        return 0
    except CoreBStageBError as exc:
        print(f"coreb_stage_b: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
