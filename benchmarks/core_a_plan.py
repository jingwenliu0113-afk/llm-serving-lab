#!/usr/bin/env python3
"""Build the Core A job plan from configs/pins.toml (PREREGISTRATION §4).

One job = one process = one configuration:
  fit     llama-completion -c L, prompt of L-128 tokens, 128 generated (memory / fit)
  ttft    llama-bench -p L -n 0                                         (engine TTFT)
  decode  llama-bench -p 0 -n 128 -d L-128                              (decode tok/s)
Each (size, quant, L) gets 3 independent processes per kind, in a seeded random
order, between a sentinel decode run at the start and at the end (drift check).

The plan is generated on the Mac and saved; the runner (run_core_a.py, stdlib
only) executes the saved plan on either platform, so the order is reproducible
from plan.json itself, not from the seed. Formal mode is refused until the
matching pre-registration stage is frozen (decision 006).

Usage:
    python benchmarks/core_a_plan.py --platform macos-metal --mode pilot \\
        --seed 20261008 --n-thr 8 --poll 0 --out plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tomllib
from pathlib import Path

SCHEMA_VERSION = 2  # v2 (decision 010): fit jobs carry external_memory; Mac fit runs under /usr/bin/time -l

# platform -> pins.toml [freeze] key (decision 006)
PLATFORMS = {"macos-metal": "mac", "wsl2-cuda": "nvidia"}
# PREREGISTRATION §3 peak memory, external measure per platform (in addition to the
# llama.cpp breakdown): Mac = /usr/bin/time -l peak memory footprint (the fit argv is
# wrapped); RTX = nvidia-smi memory.used sampled by the runner during the fit run.
EXTERNAL_MEMORY = {"macos-metal": ("time-l", ["/usr/bin/time", "-l"]), "wsl2-cuda": ("nvidia-smi", [])}

N_GEN = 128
REPS = 3
BENCH_R = 5
DELAY_S = 5
IDLE_S = 30
# Prompt for fit runs: exactly one Qwen3.5 token per unit (id 279; checked against the
# pinned tokenizer.json by tests/test_core_a_plan.py); the runner checks the logged
# prompt token count equals L - 128 (decisions 004 / 009).
PROMPT_UNIT = " the"

BENCH_COMMON = {
    "-ngl": "999", "-fa": "on", "-ctk": "f16", "-ctv": "f16", "-b": "2048", "-ub": "512",
    "-lm": "mmap", "-r": str(BENCH_R), "--delay": str(DELAY_S), "-o": "jsonl",
}
COMPLETION_COMMON = {
    "-ngl": "999", "-fa": "on", "-ctk": "f16", "-ctv": "f16", "-b": "2048", "-ub": "512", "-lm": "mmap",
}


class PlanError(ValueError):
    pass


def sentinel_cell(pins: dict) -> tuple:
    """Smallest size x lowest-bit quant x shortest context (PREREGISTRATION §4).

    Models are listed smallest first and quants highest-bit first in pins.toml.
    """
    core_a = pins["core_a"]
    return core_a["models"][0]["size"], core_a["quants"][-1], min(core_a["context_lengths"])


def _flags(d: dict) -> list:
    return [x for kv in d.items() for x in kv]


def bench_argv(model: str, p: int, n: int, d: int, n_thr: int, poll: int) -> list:
    argv = ["{bin}/llama-bench", "-m", "{models}/" + model, "-p", str(p), "-n", str(n)]
    if d:
        argv += ["-d", str(d)]
    return argv + _flags(BENCH_COMMON) + ["-t", str(n_thr), "--poll", str(poll), "-v"]


def completion_argv(model: str, L: int, n_thr: int, poll: int) -> list:
    return [
        "{bin}/llama-completion", "-m", "{models}/" + model, "-c", str(L), "-np", "1", "--fit", "off",
        *_flags(COMPLETION_COMMON), "-t", str(n_thr), "--poll", str(poll),
        "-f", "{prompt}", "-n", str(N_GEN), "--ignore-eos", "-no-cnv", "--log-jsonl",
        "-lv", "4",  # decision 012: library INFO logs (n_ctx, flash_attn, graph splits) are verbosity 4 here
    ]


def _bench_job(kind, size, quant, L, rep, model, n_thr, poll, position=None):
    if kind == "ttft":
        p, n, d = L, 0, 0
    else:  # decode / sentinel
        p, n, d = 0, N_GEN, L - N_GEN
    job = {
        "id": f"sentinel-{position}" if kind == "sentinel" else f"{kind}-{size}-{quant}-{L}-r{rep}",
        "kind": kind, "size": size, "quant": quant, "L": L, "rep": rep, "model": model,
        "argv": bench_argv(model, p, n, d, n_thr, poll),
        "expect": {"n_prompt": p, "n_gen": n, "n_depth": d},
    }
    if position:
        job["position"] = position
    return job


def build_plan(pins: dict, *, platform: str, mode: str, seed: int, n_thr: int, poll: int,
               pins_sha256: str | None = None) -> dict:
    if platform not in PLATFORMS:
        raise PlanError(f"unknown platform {platform!r}; expected one of {sorted(PLATFORMS)}")
    if mode not in ("pilot", "formal"):
        raise PlanError(f"unknown mode {mode!r}")
    stage = PLATFORMS[platform]
    if mode == "formal" and not pins["freeze"][stage]:
        raise PlanError(f"formal mode on {platform} requires freeze.{stage} = true in pins.toml (decision 006)")

    core_a = pins["core_a"]
    files = {(m["size"], q): f["filename"] for m in core_a["models"] for q, f in m["files"].items()}
    ext_kind, ext_prefix = EXTERNAL_MEMORY[platform]

    jobs = []
    for (size, quant), model in files.items():
        for L in core_a["context_lengths"]:
            for rep in range(1, REPS + 1):
                jobs.append({
                    "id": f"fit-{size}-{quant}-{L}-r{rep}", "kind": "fit", "size": size, "quant": quant, "L": L,
                    "rep": rep, "model": model, "argv": ext_prefix + completion_argv(model, L, n_thr, poll),
                    "external_memory": ext_kind,
                    "prompt": {"unit": PROMPT_UNIT, "count": L - N_GEN},
                    "expect": {"prompt_tokens": L - N_GEN},
                })
                for kind in ("ttft", "decode"):
                    jobs.append(_bench_job(kind, size, quant, L, rep, model, n_thr, poll))
    random.Random(seed).shuffle(jobs)

    s_size, s_quant, s_L = sentinel_cell(pins)
    s_model = files[(s_size, s_quant)]
    jobs = (
        [_bench_job("sentinel", s_size, s_quant, s_L, 1, s_model, n_thr, poll, position="start")]
        + jobs
        + [_bench_job("sentinel", s_size, s_quant, s_L, 1, s_model, n_thr, poll, position="end")]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "mode": mode,
        "seed": seed,
        "pins_sha256": pins_sha256,
        "params": {"n_thr": n_thr, "poll": poll, "reps": REPS, "bench_r": BENCH_R, "delay_s": DELAY_S,
                   "idle_s": IDLE_S},
        "expect": {"build_number": pins["llama_cpp"]["build_number"], "build_commit": pins["llama_cpp"]["commit"]},
        "jobs": jobs,
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pins", default=str(Path(__file__).resolve().parent.parent / "configs" / "pins.toml"))
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    parser.add_argument("--mode", required=True, choices=["pilot", "formal"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n-thr", required=True, type=int, help="-t value (PREREGISTRATION §4)")
    parser.add_argument("--poll", required=True, type=int, help="--poll value (PREREGISTRATION §4)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    raw = Path(args.pins).read_bytes()
    plan = build_plan(
        tomllib.loads(raw.decode()), platform=args.platform, mode=args.mode, seed=args.seed,
        n_thr=args.n_thr, poll=args.poll, pins_sha256=hashlib.sha256(raw).hexdigest(),
    )
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
