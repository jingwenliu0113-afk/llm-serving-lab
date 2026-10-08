#!/usr/bin/env python3
"""Staged Core B smoke test (decision 008): S1 2K -> S2 8K -> S3 32K.

Goal: the fixed Core B checkpoint starts, serves ONE text-only request and
releases its GPU memory on the RTX 5070 Ti. No throughput measurement, no high
concurrency; results are pass/fail only (decision 006). Each stage runs only if
the previous stage's report is ok. Runs inside WSL with the project venv;
stdlib only (Python 3.10+).

Usage (WSL, ~/llm-serving-lab):
    .venv/bin/python benchmarks/coreb_smoke.py --stage S1 --out results/core_b/smoke/<session> \\
        --model <snapshot dir> --revision a398088c... --gdn-prefill-backend triton
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple


class Stage(NamedTuple):
    name: str
    max_model_len: int
    prompt_tokens: int


STAGES = (Stage("S1", 2048, 1800), Stage("S2", 8192, 7900), Stage("S3", 32768, 32500))
MAX_TOKENS = 128
PROMPT_UNIT = " the"  # one Qwen3.5 token per unit (checked against tokenizer.json)
HF_HOME = "/home/wenn0113/.cache/huggingface"
REQUIRED_FACTS = ("kv_cache_tokens", "attn_block_size", "gdn_prefill_kernel", "gdn_decode_kernel")

# Literal formats from vLLM v0.30.0 (kv_cache_utils.py, platforms/interface.py,
# platforms/cuda.py, layers/mamba/gdn/qwen_gdn_linear_attn.py).
_FACTS = {
    "attention_backend": (re.compile(r"Using (\S+) backend\."), str),
    "attn_block_size": (re.compile(r"Setting attention block size to (\d+) tokens"), int),
    "gdn_prefill_kernel": (re.compile(r"Using (\S+) GDN prefill kernel"), str),
    "gdn_prefill_requested": (re.compile(r"GDN prefill kernel \(requested=([^,]+),"), str),
    "gdn_decode_kernel": (re.compile(r"GDN decode kernel: (\S+)"), str),
    "kv_cache_tokens": (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), lambda s: int(s.replace(",", ""))),
    "max_concurrency": (re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"), float),
    "max_model_len_logged": (re.compile(r"Maximum concurrency for ([\d,]+) tokens per request"),
                             lambda s: int(s.replace(",", ""))),
}


class SmokeError(RuntimeError):
    pass


def serve_command(*, vllm_bin: str, model: str, revision: str, stage: Stage, gdn_prefill_backend: str,
                  port: int, base_env: dict) -> tuple:
    argv = [
        vllm_bin, "serve", model, "--revision", revision,
        "--language-model-only",  # no vision tower: this study never uses it
        "--dtype", "bfloat16",
        "--max-model-len", str(stage.max_model_len),
        "--max-num-seqs", "8",
        "--gpu-memory-utilization", "0.85",
        "--no-enable-prefix-caching",
        "--gdn-prefill-backend", gdn_prefill_backend,
        "--generation-config", "vllm",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    env = {**base_env, "HF_HOME": HF_HOME, "HF_HUB_OFFLINE": "1", "VLLM_GDN_DECODE_KERNEL": "triton"}
    return argv, env


def parse_server_log(text: str) -> dict:
    facts = {}
    for key, (pattern, cast) in _FACTS.items():
        m = pattern.search(text)
        facts[key] = cast(m.group(1)) if m else None
    return facts


def _previous_ok(stage_index: int, out: Path) -> None:
    if stage_index == 0:
        return
    prev = STAGES[stage_index - 1].name
    report = out / f"{prev}.json"
    if not report.is_file() or not json.loads(report.read_text()).get("ok"):
        raise SmokeError(f"{prev} must pass before {STAGES[stage_index].name} (decision 008)")


def run_stage(
    *,
    stage_name: str,
    out_dir,
    vllm_bin: str,
    model: str,
    revision: str,
    gdn_prefill_backend: str,
    gpu_mem_used: Callable,
    popen: Callable,
    http_get_ok: Callable,
    http_post_json: Callable,
    now: Callable = time.monotonic,
    sleep: Callable = time.sleep,
    idle_mib: int = 3000,
    port: int = 8765,
    startup_timeout_s: float = 1800,
    request_timeout_s: float = 900,
    release_timeout_s: float = 60,
    release_margin_mib: int = 256,
    base_env: dict | None = None,
) -> dict:
    names = [s.name for s in STAGES]
    if stage_name not in names:
        raise SmokeError(f"unknown stage {stage_name}; expected one of {names}")
    idx = names.index(stage_name)
    stage = STAGES[idx]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _previous_ok(idx, out)

    mem_before = gpu_mem_used()
    if mem_before > idle_mib:
        raise SmokeError(f"GPU busy: memory.used {mem_before} MiB > {idle_mib} MiB; not starting (MACHINE_POLICY)")

    argv, env = serve_command(vllm_bin=vllm_bin, model=model, revision=revision, stage=stage,
                              gdn_prefill_backend=gdn_prefill_backend, port=port,
                              base_env=dict(os.environ if base_env is None else base_env))
    log_path = out / f"{stage.name}.server.log"
    reasons, timings = [], {}
    mem_during = None

    t0 = now()
    proc = popen(argv, env, log_path)
    ready = False
    while now() - t0 < startup_timeout_s:
        if proc.poll() is not None:
            reasons.append(f"server exited during startup (code {proc.poll()})")
            break
        if http_get_ok(f"http://127.0.0.1:{port}/health"):
            ready = True
            break
        sleep(2)
    else:
        reasons.append(f"server not ready within {startup_timeout_s} s")
    timings["startup_s"] = round(now() - t0, 1)

    if ready:
        payload = {"model": model, "prompt": PROMPT_UNIT * stage.prompt_tokens, "max_tokens": MAX_TOKENS,
                   "temperature": 0, "ignore_eos": True}
        t1 = now()
        try:
            resp = http_post_json(f"http://127.0.0.1:{port}/v1/completions", payload, request_timeout_s)
            usage = resp.get("usage") or {}
            if not resp.get("choices") or not resp["choices"][0].get("text"):
                reasons.append("empty completion")
            if usage.get("prompt_tokens") != stage.prompt_tokens:
                reasons.append(f"prompt_tokens={usage.get('prompt_tokens')} (expected {stage.prompt_tokens})")
            if usage.get("completion_tokens") != MAX_TOKENS:
                reasons.append(f"completion_tokens={usage.get('completion_tokens')} (expected {MAX_TOKENS})")
        except Exception as exc:  # noqa: BLE001 - any request failure fails the stage
            reasons.append(f"request failed: {type(exc).__name__}: {exc}")
        timings["request_s"] = round(now() - t1, 1)
        mem_during = gpu_mem_used()

    # shutdown: SIGINT -> TERM -> KILL
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        for _ in range(15):
            if proc.poll() is not None:
                break
            sleep(2)
        if proc.poll() is None:
            proc.terminate()
            sleep(5)
        if proc.poll() is None:
            proc.kill()
            reasons.append("server had to be killed")

    t2 = now()
    mem_after = gpu_mem_used()
    while mem_after > mem_before + release_margin_mib and now() - t2 < release_timeout_s:
        sleep(2)
        mem_after = gpu_mem_used()
    timings["release_s"] = round(now() - t2, 1)
    if mem_after > mem_before + release_margin_mib:
        reasons.append(f"GPU memory not released within {release_timeout_s} s "
                       f"({mem_after} MiB vs {mem_before} MiB before)")

    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    facts = parse_server_log(log_text)
    if ready:
        reasons.extend(f"missing log fact: {k}" for k in REQUIRED_FACTS if facts.get(k) is None)

    report = {
        "stage": stage.name, "max_model_len": stage.max_model_len, "prompt_tokens": stage.prompt_tokens,
        "ok": not reasons, "reasons": reasons, "argv": argv,
        "env": {k: env[k] for k in ("HF_HOME", "HF_HUB_OFFLINE", "VLLM_GDN_DECODE_KERNEL")},
        "timings": timings, "log_facts": facts,
        "mem_before_mib": mem_before, "mem_during_mib": mem_during, "mem_after_mib": mem_after,
        "server_log": log_path.name, "log_tail": "\n".join(log_text.splitlines()[-40:]),
    }
    (out / f"{stage.name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


# ----------------------------------------------------------- real backends --


def nvidia_mem_used() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def start_server(argv, env, log_path):
    log = open(log_path, "w")
    return subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def http_get_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=[s.name for s in STAGES])
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True, help="pinned snapshot dir or repo id")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--gdn-prefill-backend", required=True, choices=["triton", "flashinfer", "cutedsl"])
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capture = [sys.executable, str(root / "scripts" / "env_capture.py"), "--vllm-python", sys.executable]
    subprocess.run(capture + ["--out", str(out / f"{args.stage}.env.before.json")], check=False)
    try:
        report = run_stage(
            stage_name=args.stage, out_dir=out, vllm_bin=args.vllm_bin, model=args.model, revision=args.revision,
            gdn_prefill_backend=args.gdn_prefill_backend, gpu_mem_used=nvidia_mem_used, popen=start_server,
            http_get_ok=http_get_ok, http_post_json=http_post_json, port=args.port,
        )
    except SmokeError as exc:
        print(f"coreb_smoke: {exc}", file=sys.stderr)
        return 2
    finally:
        subprocess.run(capture + ["--out", str(out / f"{args.stage}.env.after.json")], check=False)
    print(json.dumps({k: report[k] for k in ("stage", "ok", "reasons", "timings", "log_facts")}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
