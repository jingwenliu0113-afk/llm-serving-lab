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

LEVELS = (2048, 8192, 32768)
MIXED_LEVELS = (2048, 8192)
N_GEN = 128
LONG_INPUT = 32640  # 32768 - 128
SEED = 20261008
RESTARTS = 3
MAX_MODEL_LEN = 32768
MAX_NUM_SEQS = 264
GRID_CAP = 256
HF_HOME = "/home/wenn0113/.cache/huggingface"
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


def server_argv(pins: dict, *, gdn_prefill_backend: str, gpu_memory_utilization: str, port: int,
                vllm_bin: str = "{vllm}") -> tuple:
    b = pins["core_b"]
    argv = [vllm_bin, "serve", b["repo"], "--revision", b["revision"],
            "--language-model-only", "--dtype", b["dtype"],
            "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
            "--gpu-memory-utilization", gpu_memory_utilization,
            "--no-enable-prefix-caching", "--gdn-prefill-backend", gdn_prefill_backend,
            "--generation-config", "vllm", "--host", "127.0.0.1", "--port", str(port)]
    env = {**bench_env(), "VLLM_GDN_DECODE_KERNEL": "triton"}
    return argv, env


def bench_argv(pins: dict, *, L: int, C: int, mix: str, result_dir: str, result_filename: str, port: int,
               mixed_dataset: str | None = None, vllm_bin: str = "{vllm}") -> list:
    argv = [vllm_bin, "bench", "serve", "--backend", "openai", "--base-url", f"http://127.0.0.1:{port}",
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


def build_uniform_plan(pins: dict, *, num_gpu_blocks: int, block_size: int, seed: int = SEED) -> dict:
    ckv = {str(L): c_kv(num_gpu_blocks, block_size, L) for L in LEVELS}
    grids = {L: concurrency_grid(v) for L, v in ckv.items()}
    runs = [{"L": L, "mix": "uniform", "restart": r, "order": _order(grids[str(L)], seed, L, r, "uniform")}
            for L in LEVELS for r in range(1, RESTARTS + 1)]
    return {"schema_version": 1, "seed": seed, "num_gpu_blocks": num_gpu_blocks, "block_size": block_size,
            "c_kv": ckv, "grids": grids, "runs": runs}


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
