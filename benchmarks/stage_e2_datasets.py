#!/usr/bin/env python3
"""Build E2's shared-prefix dataset (decision 041 v3 §2.8) for the PC cells: `rows` prompts of exactly `total`
(1,920) tokens that share one RandomDataset prefix, written as {"prompt", "output_tokens": 128} JSONL in order (the
bench runs it with --disable-shuffle).

- One RandomDataset(random_seed = Core B's seed).sample(num_requests = rows, prefix_len, input_len = total -
  prefix_len, output_len 128, range_ratio 0) call, so every row has the same prefix.
- Every prompt must re-tokenize (as vLLM's custom dataset counts it) to exactly `total` tokens, and the token-id prefix
  common to all rows must be >= min_shared (3 x 528-token blocks, so prefix caching can reuse whole blocks); otherwise
  nothing is written. The file's sha256 and the measured common prefix are pinned in configs/stage_e2.toml.

Usage (WSL, project venv, offline HF cache):
    .venv/bin/python benchmarks/stage_e2_datasets.py --out data/stage_e/shared-2048.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import core_b_plan as bp  # noqa: E402


class DatasetError(RuntimeError):
    pass


def common_prefix_len(id_lists: list) -> int:
    if not id_lists:
        return 0
    n = min(len(x) for x in id_lists)
    for i in range(n):
        first = id_lists[0][i]
        if any(x[i] != first for x in id_lists[1:]):
            return i
    return n


def build(out, *, spec: dict, sample_all: Callable, encode: Callable) -> dict:
    """spec: the config's shared_prefix table; sample_all(rows, prefix_len, input_len) -> prompts; encode(prompt) ->
    token ids."""
    prompts = sample_all(rows=spec["rows"], prefix_len=spec["prefix_len"], input_len=spec["total"] - spec["prefix_len"])
    if len(prompts) != spec["rows"]:
        raise DatasetError(f"{len(prompts)} prompts, {spec['rows']} expected")
    ids = [list(encode(p)) for p in prompts]
    bad = sum(len(x) != spec["total"] for x in ids)
    if bad:
        raise DatasetError(f"{bad} of {len(ids)} prompts do not re-tokenize to {spec['total']} tokens; nothing written")
    common = common_prefix_len(ids)
    if common < spec["min_shared"]:
        raise DatasetError(f"the common token prefix is {common} < {spec['min_shared']}; nothing written")
    data = "".join(json.dumps({"prompt": p, "output_tokens": bp.N_GEN}, ensure_ascii=False) + "\n"
                   for p in prompts).encode()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    return {"rows": len(prompts), "total": spec["total"], "common_prefix": common,
            "sha256": hashlib.sha256(data).hexdigest(), "path": str(out)}


def vllm_sampler(model: str, revision: str) -> tuple:  # pragma: no cover - needs vLLM (WSL)
    from vllm.benchmarks.datasets.datasets import RandomDataset
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(model, revision=revision)

    def sample_all(*, rows: int, prefix_len: int, input_len: int) -> list:
        reqs = RandomDataset(random_seed=bp.SEED).sample(tokenizer=tokenizer, num_requests=rows, prefix_len=prefix_len,
                                                          input_len=input_len, output_len=bp.N_GEN, range_ratio=0.0)
        return [r.prompt for r in reqs]

    def encode(prompt: str) -> list:  # as CustomDataset.sample counts prompt_len (vLLM 0.30.0)
        return tokenizer(prompt).input_ids

    return sample_all, encode


def main(argv: list | None = None) -> int:  # pragma: no cover - WSL only
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    pins = bp.load_pins()
    spec = tomllib.loads((ROOT / "configs" / "stage_e2.toml").read_text())["e2"]["shared_prefix"]
    sample_all, encode = vllm_sampler(pins["core_b"]["repo"], pins["core_b"]["revision"])
    try:
        print(json.dumps(build(Path(args.out), spec=spec, sample_all=sample_all, encode=encode)))
    except DatasetError as exc:
        print(f"stage_e2_datasets: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
