#!/usr/bin/env python3
"""Build the seeded `mixed` dataset of decision 007 (H-B4 only; L in {2048, 8192}).

Short prompts come from vLLM's own RandomDataset with the same seed and input length as
the `uniform` runs (so they are the uniform prompts); long prompts (32640 tokens) from the
same generator with seed + 1. Rows are laid out by core_b_plan.mixed_rows (blocks of ten).
The sha256 of each file is pinned at Stage B (pins.toml core_b.mix_dataset_sha256).

Usage (WSL, project venv, offline HF cache):
    .venv/bin/python benchmarks/core_b_mixed_dataset.py --L 2048 --out data/core_b/mixed-2048.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import core_b_plan as bp


def build(L: int, out: Path, *, sample: Callable) -> dict:
    if L not in bp.MIXED_LEVELS:
        raise bp.PlanError(f"mixed is defined only for L in {bp.MIXED_LEVELS}")
    n_short, n_long = bp.mixed_counts(bp.MIXED_ROWS)
    short = sample(num_requests=n_short, input_len=L - bp.N_GEN, seed=bp.SEED)
    long_ = sample(num_requests=n_long, input_len=bp.LONG_INPUT, seed=bp.SEED + 1)
    rows = bp.mixed_rows(short=short, long=long_, seed=bp.SEED)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    return {"L": L, "rows": len(rows), "sha256": hashlib.sha256(data).hexdigest(), "path": str(out)}


def vllm_sampler(model: str, revision: str) -> Callable:  # pragma: no cover - needs vLLM (WSL)
    from vllm.benchmarks.datasets.datasets import RandomDataset
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(model, revision=revision)

    def sample(*, num_requests: int, input_len: int, seed: int) -> list:
        reqs = RandomDataset(random_seed=seed).sample(tokenizer=tokenizer, num_requests=num_requests,
                                                      prefix_len=0, input_len=input_len, output_len=bp.N_GEN,
                                                      range_ratio=0.0)
        return [r.prompt for r in reqs]

    return sample


def main(argv: list | None = None) -> int:  # pragma: no cover - WSL only
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--L", type=int, required=True, choices=list(bp.MIXED_LEVELS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    pins = tomllib.loads((Path(__file__).resolve().parent.parent / "configs" / "pins.toml").read_text())
    info = build(args.L, Path(args.out), sample=vllm_sampler(pins["core_b"]["repo"], pins["core_b"]["revision"]))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
