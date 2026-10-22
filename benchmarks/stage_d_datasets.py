#!/usr/bin/env python3
"""Build the seeded var datasets of decision 039 D-C (U2K-var, U8K-var): prompts whose lengths are uniform on
[64, tier - 128], each made by vLLM's own RandomDataset (as 007's mixed dataset, core_b_mixed_dataset.py) and each
re-tokenized, as vLLM's custom dataset counts it, to exactly its target length, or nothing is written (013).

- lengths(tier, rows, seed): random.Random(int(sha256("<seed>:stage_d:var:<tier>"), 16)).randint(64, tier - 128) for
  each of `rows` rows (rows = core_b_plan.num_prompts(C) + its warmups, so every request and warmup has its own row).
- Each prompt is RandomDataset(random_seed = seed + i).sample(input_len = lengths[i], output_len 128, range_ratio 0).
- Rows are {"prompt": ..., "output_tokens": 128} JSONL in order (the bench runs them with --disable-shuffle). The file's
  sha256 is pinned in configs/stage_d.toml var_dataset_sha256 before the freeze.

Usage (WSL, project venv, offline HF cache):
    .venv/bin/python benchmarks/stage_d_datasets.py --tier 2048 --out data/stage_d/var-2048.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Callable

import core_b_plan as bp
import stage_d_plan as sd

VAR_POINTS = {2048: 32, 8192: 8}          # 039 D-C: the C each var dataset serves (its rows cover N + warmups)


def rows_for(tier: int) -> int:
    C = VAR_POINTS[tier]
    return bp.num_prompts(C) + bp.num_warmups(C)


def lengths(tier: int, rows: int, seed: int = bp.SEED) -> list:
    rng = random.Random(int(hashlib.sha256(f"{seed}:stage_d:var:{tier}".encode()).hexdigest(), 16))
    return [rng.randint(sd.VAR_MIN_PROMPT, tier - bp.N_GEN) for _ in range(rows)]


def build(tier: int, out: Path, *, sample: Callable, count_tokens: Callable, seed: int = bp.SEED) -> dict:
    if tier not in VAR_POINTS:
        raise sd.PlanError(f"var datasets exist only for tiers {sorted(VAR_POINTS)} (039 D-C)")
    want = lengths(tier, rows_for(tier), seed)
    prompts = [sample(input_len=n, seed=seed + i) for i, n in enumerate(want)]
    bad = sum(count_tokens(p) != n for p, n in zip(prompts, want))
    if bad:
        raise sd.PlanError(f"{bad} of {len(want)} prompts do not re-tokenize to their target length")
    data = "".join(json.dumps({"prompt": p, "output_tokens": bp.N_GEN}, ensure_ascii=False) + "\n"
                   for p in prompts).encode()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    return {"tier": tier, "rows": len(want), "min": min(want), "max": max(want),
            "sha256": hashlib.sha256(data).hexdigest(), "path": str(out)}


def vllm_sampler(model: str, revision: str) -> tuple:  # pragma: no cover - needs vLLM (WSL)
    from vllm.benchmarks.datasets.datasets import RandomDataset
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(model, revision=revision)

    def sample(*, input_len: int, seed: int) -> str:
        (req,) = RandomDataset(random_seed=seed).sample(tokenizer=tokenizer, num_requests=1, prefix_len=0,
                                                         input_len=input_len, output_len=bp.N_GEN, range_ratio=0.0)
        return req.prompt

    def count_tokens(prompt: str) -> int:  # as CustomDataset.sample counts prompt_len (vLLM 0.30.0)
        return len(tokenizer(prompt).input_ids)

    return sample, count_tokens


def main(argv: list | None = None) -> int:  # pragma: no cover - WSL only
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", type=int, required=True, choices=sorted(VAR_POINTS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    pins = bp.load_pins()
    sample, count = vllm_sampler(pins["core_b"]["repo"], pins["core_b"]["revision"])
    print(json.dumps(build(args.tier, Path(args.out), sample=sample, count_tokens=count)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
