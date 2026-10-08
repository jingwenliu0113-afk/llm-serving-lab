#!/usr/bin/env python3
"""Core B plan rules (decision 007, hybrid C_KV of 008 / 009), fixed before any NVIDIA data.

- C_KV(L) = floor((num_gpu_blocks - 1) / (ceil(L / block_size) + 3)), num_gpu_blocks and
  block_size from /metrics vllm:cache_config_info (Stage B values).
- grid = {1, 2, 4, ..., C_hi} with C_hi = smallest power of two >= max(8, 8 x C_KV), cap 256,
  plus round(C_KV x 2^(k/2)) for k in -2..2 (>= 1, deduplicated). "round" is half-up
  (x.5 -> up), never Python's banker's rounding.
- every C: num-prompts = max(10 C, 200), warmups = C; 3 server restarts per (L, mix);
  C order per restart is a seeded permutation stored in the plan.
- one shared server config for all levels and mixes: --max-model-len 32768, --max-num-seqs 264.
- `mixed` (H-B4 only, L in {2048, 8192}): 90% (L - 128, 128) + 10% (32640, 128), a custom
  dataset built with vLLM's own RandomDataset (same generator, seed and input length as
  `uniform`, so the short prompts follow the uniform distribution; token content can differ
  because RandomDataset's re-encode padding draws depend on num_requests, 013) in blocks of
  ten (9 short + 1 long, seeded order inside each block), MIXED_ROWS = num-prompts at the
  grid cap. It is read in file order (--disable-shuffle), so the first N rows hold exactly
  N / 10 long prompts for any C; its sha256 is pinned at Stage B (core_b.mix_dataset_sha256),
  before C_knee is known.
- client (013): --temperature 0 (vLLM 0.30.0 `bench serve` no longer sends it), the pinned
  snapshot as --tokenizer (`bench serve` has no --revision) with HF_HUB_OFFLINE=1, and for
  `mixed` --custom-output-len 128 (its default 256 overrides the rows' output_tokens).

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import hashlib
import math
import random
import re

LEVELS = (2048, 8192, 32768)
MIXED_LEVELS = (2048, 8192)
N_GEN = 128
LONG_INPUT = 32640  # 32768 - 128
SEED = 20261008
RESTARTS = 3
MAX_MODEL_LEN = 32768
MAX_NUM_SEQS = 264
MAX_NUM_BATCHED_TOKENS = 2048  # 003: the 16 GB default, written explicitly
UTIL_MARGIN_MIB = 512          # 017
KV_ROUND_MIB = 256             # 017
GRID_CAP = 256
HF_HOME = "/home/wenn0113/.cache/huggingface"
PROJECT_DIR = "/home/wenn0113/llm-serving-lab"
VENV_BIN = f"{PROJECT_DIR}/.venv/bin"
# decision 014: user-space CUDA_HOME for FlashInfer's sampler JIT, pointing at the hash-locked pip CUDA 13
# (nvidia/cu13: nvcc 13.4.92, cudart 13, curand headers); created by scripts/wsl_cuda_home_shim.sh
FLASHINFER_CUDA_HOME = f"{PROJECT_DIR}/.cuda-home-cu13"
MIXED_ROWS = 10 * GRID_CAP  # = num_prompts(GRID_CAP)


class PlanError(ValueError):
    pass


def c_kv(num_gpu_blocks: int, block_size: int, L: int) -> int:
    return (num_gpu_blocks - 1) // (math.ceil(L / block_size) + 3)


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def concurrency_grid(ckv: int) -> list:
    target = max(8, 8 * ckv)
    c_hi = 1
    while c_hi < target:
        c_hi *= 2
    c_hi = min(c_hi, GRID_CAP)
    grid = {2 ** i for i in range(int(math.log2(c_hi)) + 1)}
    grid |= {max(1, round_half_up(ckv * 2 ** (k / 2))) for k in range(-2, 3)}
    return sorted(c for c in grid if c <= GRID_CAP)


def num_prompts(C: int) -> int:
    return max(10 * C, 200)


def num_warmups(C: int) -> int:
    return C


def snapshot_dir(pins: dict) -> str:
    b = pins["core_b"]
    return f"{HF_HOME}/hub/models--{b['repo'].replace('/', '--')}/snapshots/{b['revision']}"


def bench_env() -> dict:
    return {"HF_HOME": HF_HOME, "HF_HUB_OFFLINE": "1"}


ENGINE_PREFIXES = ("VLLM_", "CUDA_", "GGML_")


def inherited_unlisted(base) -> list:
    """Engine variables in the caller's env that server_env drops (§4: none are set unless listed)."""
    return sorted(k for k in base if k.startswith(ENGINE_PREFIXES))


def server_env(base, pin_memory: str = "0") -> dict:
    """The Core B server env on top of the caller's (§4; 008; 014; 018). Inherited VLLM_* / CUDA_* / GGML_*
    variables are dropped (§4 lists every one the server gets); VLLM_WSL2_ENABLE_PIN_MEMORY is the Stage B
    A/B choice (003 / 017; "0" = vLLM's default until it is pinned). nvcc is never put on PATH: vLLM's
    has_flashinfer() gate depends on it, while FlashInfer's own JIT finds nvcc through CUDA_HOME."""
    clean = {k: v for k, v in base.items() if not k.startswith(ENGINE_PREFIXES)}
    path = clean.get("PATH", "")
    return {**clean, **bench_env(), "VLLM_GDN_DECODE_KERNEL": "triton", "CUDA_HOME": FLASHINFER_CUDA_HOME,
            "VLLM_WSL2_ENABLE_PIN_MEMORY": pin_memory, "PATH": f"{VENV_BIN}:{path}" if path else VENV_BIN}


def recorded_env(env: dict) -> dict:
    """What a report records of the server env (the inherited PATH itself is machine noise)."""
    out = {k: env.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE", "VLLM_GDN_DECODE_KERNEL", "CUDA_HOME",
                                    "VLLM_WSL2_ENABLE_PIN_MEMORY")}
    out["PATH_PREFIX"] = VENV_BIN if env.get("PATH", "").split(":")[0] == VENV_BIN else None
    return out


def toolchain_problems(env: dict, *, exists=None, which=None) -> list:
    """What the server would miss at startup (Triton C stub, FlashInfer sampler JIT; 014), checked
    before any model is loaded."""
    import os
    import shutil

    exists = exists or os.path.exists
    which = which or shutil.which
    path, home = env.get("PATH", ""), env.get("CUDA_HOME") or ""
    problems = []
    if not (which("cc", path=path) or which("gcc", path=path)):
        problems.append("no C compiler (cc / gcc) on PATH: Triton cannot build its launcher stub")
    if not which("c++", path=path):
        problems.append("no C++ compiler (c++) on PATH: FlashInfer's JIT links with it")
    if not which("ninja", path=path):
        problems.append("no ninja on PATH (expected in the project venv): FlashInfer's JIT builds with it")
    for rel in ("bin/nvcc", "include/curand.h", "lib64/libcudart.so", "lib64/stubs/libcuda.so"):
        if not exists(f"{home}/{rel}"):
            problems.append(f"CUDA_HOME={home!r} lacks {rel} (scripts/wsl_cuda_home_shim.sh)")
    if which("nvcc", path=path):
        problems.append("nvcc is on PATH: it would change vLLM's has_flashinfer() gate (014)")
    return problems


def gpu_memory_utilization(*, v_total_mib: float, b_mib: float) -> str:
    """017: the largest multiple of 0.01 with U x V_total <= V_total - B - UTIL_MARGIN_MIB (vLLM refuses to
    start when free VRAM < U x total; the margin absorbs Windows idle VRAM fluctuation)."""
    return f"{int(100 * (v_total_mib - b_mib - UTIL_MARGIN_MIB) // v_total_mib) / 100:.2f}"


_RE_AVAILABLE_KV = re.compile(r"Available KV cache memory: ([\d.]+) GiB")


def kv_cache_bytes_from_log(text: str) -> int | None:
    """017: K = the warm boot's 'Available KV cache memory' (gpu_worker.py), floored to 256 MiB."""
    m = _RE_AVAILABLE_KV.search(text)
    if not m:
        return None
    mib = float(m.group(1)) * 1024
    return int(mib // KV_ROUND_MIB) * KV_ROUND_MIB * 2**20


def server_argv(pins: dict, *, gdn_prefill_backend: str, gpu_memory_utilization: str, port: int,
                vllm_bin: str = "{vllm}", base_env=None, kv_cache_memory_bytes: int | None = None) -> tuple:
    """Shared Core B server: 007 / 008 flags plus the 003 fixed scheduler / backend / KV-pool flags (017).
    Only the Stage B boots that determine K run without --kv-cache-memory-bytes."""
    b = pins["core_b"]
    argv = [vllm_bin, "serve", b["repo"], "--revision", b["revision"],
            "--language-model-only", "--dtype", b["dtype"],
            "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
            "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
            "--performance-mode", "balanced", "--async-scheduling", "--attention-backend", "FLASH_ATTN",
            "--gpu-memory-utilization", gpu_memory_utilization,
            "--no-enable-prefix-caching", "--gdn-prefill-backend", gdn_prefill_backend,
            "--generation-config", "vllm", "--host", "127.0.0.1", "--port", str(port)]
    if kv_cache_memory_bytes is not None:
        argv += ["--kv-cache-memory-bytes", str(int(kv_cache_memory_bytes))]
    pin_memory = (pins.get("stage_b") or {}).get("wsl2_pin_memory") or "0"
    return argv, server_env(base_env or {}, pin_memory=pin_memory)


def bench_argv(pins: dict, *, L: int, C: int, mix: str, result_dir: str, result_filename: str, port: int,
               mixed_dataset: str | None = None, vllm_bin: str = "{vllm}") -> list:
    argv = [vllm_bin, "bench", "serve", "--backend", "vllm", "--base-url", f"http://127.0.0.1:{port}",
            "--endpoint", "/v1/completions", "--model", pins["core_b"]["repo"], "--tokenizer", snapshot_dir(pins)]
    if mix == "uniform":
        argv += ["--dataset-name", "random", "--random-input-len", str(L - N_GEN),
                 "--random-output-len", str(N_GEN), "--random-range-ratio", "0"]
    elif mix == "mixed":
        if L not in MIXED_LEVELS or not mixed_dataset:
            raise PlanError(f"mixed is defined only for L in {MIXED_LEVELS} with a pinned dataset (007)")
        argv += ["--dataset-name", "custom", "--dataset-path", mixed_dataset, "--skip-chat-template",
                 "--disable-shuffle", "--custom-output-len", str(N_GEN)]
    else:
        raise PlanError(f"unknown mix {mix!r}")
    argv += ["--num-prompts", str(num_prompts(C)), "--max-concurrency", str(C), "--num-warmups", str(num_warmups(C)),
             "--request-rate", "inf", "--ignore-eos", "--temperature", "0", "--seed", str(SEED),
             "--percentile-metrics", "ttft,tpot,itl,e2el,client_queue_time", "--metric-percentiles", "50,95,99",
             "--save-result", "--save-detailed", "--result-dir", result_dir, "--result-filename", result_filename]
    return argv


def _order(grid: list, seed: int, L: int, restart: int, mix: str) -> list:
    digest = hashlib.sha256(f"{seed}:{mix}:{L}:{restart}".encode()).hexdigest()
    order = list(grid)
    random.Random(int(digest, 16)).shuffle(order)
    return order


def build_uniform_plan(pins: dict, *, num_gpu_blocks: int, block_size: int, seed: int = SEED,
                       kv_cache_memory_bytes: int | None = None, gpu_memory_utilization: str | None = None) -> dict:
    ckv = {str(L): c_kv(num_gpu_blocks, block_size, L) for L in LEVELS}
    grids = {L: concurrency_grid(v) for L, v in ckv.items()}
    runs = [{"L": L, "mix": "uniform", "restart": r, "order": _order(grids[str(L)], seed, L, r, "uniform")}
            for L in LEVELS for r in range(1, RESTARTS + 1)]
    return {"schema_version": 1, "seed": seed, "num_gpu_blocks": num_gpu_blocks, "block_size": block_size,
            "kv_cache_memory_bytes": kv_cache_memory_bytes, "gpu_memory_utilization": gpu_memory_utilization,
            "c_kv": ckv, "grids": grids, "runs": runs}


def build_mixed_plan(uniform_plan: dict, c_knee: dict, *, uniform_session: str) -> dict:
    """007: `mixed` only at L in {2048, 8192}, one C = that level's uniform C_knee (from the
    uniform session, core_b_analysis.knee), 3 restarts. A level without a C_knee is skipped."""
    knees = {int(L): C for L, C in c_knee.items()}
    levels = [L for L in MIXED_LEVELS if knees.get(L) is not None]
    runs = [{"L": L, "mix": "mixed", "restart": r, "order": [knees[L]]}
            for L in levels for r in range(1, RESTARTS + 1)]
    return {**{k: uniform_plan.get(k) for k in ("schema_version", "seed", "num_gpu_blocks", "block_size",
                                                "kv_cache_memory_bytes", "gpu_memory_utilization", "c_kv", "grids")},
            "derived_from": uniform_session, "c_knee": {str(L): knees[L] for L in levels},
            "skipped": {str(L): "uniform C_knee not available" for L in MIXED_LEVELS if L not in levels},
            "runs": runs}


def mixed_counts(n: int) -> tuple:
    n_long = n // 10
    return n - n_long, n_long


def mixed_rows(*, short: list, long: list, seed: int = SEED) -> list:
    """Blocks of ten: 9 short + 1 long in a seeded order per block; short prompts keep their order."""
    if len(short) != 9 * len(long):
        raise PlanError("mixed dataset needs exactly 9 short prompts per long prompt")
    rng = random.Random(seed)
    rows, s = [], iter(short)
    for lp in long:
        pos = rng.randrange(10)
        block = [next(s) for _ in range(9)]
        block.insert(pos, lp)
        rows.extend({"prompt": p, "output_tokens": N_GEN} for p in block)
    return rows


def load_pins() -> dict:
    import tomllib
    from pathlib import Path

    return tomllib.loads((Path(__file__).resolve().parent.parent / "configs" / "pins.toml").read_text())


def main(argv: list | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Core B plans (decision 007)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("uniform", help="uniform plan from the Stage B values in pins.toml [stage_b] (017)")
    u.add_argument("--out", required=True)
    m = sub.add_parser("mixed", help="mixed plan at each level's uniform C_knee")
    m.add_argument("--uniform-session", required=True)
    m.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "uniform":
        pins = load_pins()
        sb = pins["stage_b"]
        missing = [k for k in ("num_gpu_blocks", "block_size", "kv_cache_memory_bytes", "gpu_memory_utilization")
                   if not sb.get(k)]
        if missing:
            parser.error(f"pins.toml [stage_b] lacks {missing} (measure them at Stage B first, 017)")
        plan = build_uniform_plan(pins, num_gpu_blocks=sb["num_gpu_blocks"], block_size=sb["block_size"],
                                  kv_cache_memory_bytes=sb["kv_cache_memory_bytes"],
                                  gpu_memory_utilization=sb["gpu_memory_utilization"])
    else:
        import core_b_analysis as ba

        session = Path(args.uniform_session)
        up = json.loads((session / "plan.json").read_text())
        rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines()]
        sid = json.loads((session / "session.json").read_text())["session_id"]
        knees = {L: ba.knee(rows, L, up["grids"][str(L)])["c_knee"] for L in MIXED_LEVELS}
        plan = build_mixed_plan(up, knees, uniform_session=sid)
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"runs": len(plan["runs"]), "c_knee": plan.get("c_knee"), "c_kv": plan["c_kv"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
