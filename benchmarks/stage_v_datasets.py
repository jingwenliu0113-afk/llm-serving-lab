#!/usr/bin/env python3
"""Build the seeded datasets of decision 044 (V1, V4 and the two mini datasets): every row's prompt is made by vLLM's
own RandomDataset (random_seed = seed + i, as D-C's var datasets, stage_d_datasets) at the row's target length and
re-tokenized, as vLLM's custom dataset counts it, to exactly that length, or nothing is written (013).

- V1 (v1-<tier>-r<r>, r = 1..5, block r reads r): N = core_b_plan.num_prompts(C) rows (L-U2K 320, L-U8K 200) of the
  O512 base's prompt (U2K-O512 1,920 / U8K-O512 8,064 tokens); row 0's output is 128 (the bench's ready check and its
  warmups resend row 0, so they are U2K's / U8K's), every other row's
  random.Random(int(sha256("<seed>:stage_v:v1:<tier>:<r>"), 16)).randint(1, 512) in row order. The bench reads them
  with --custom-output-len -1 (each row's output_tokens) and --ignore-eos.
- V4 (v4-<mix>-r<r>): 320 rows; row 0 is a 2K prompt (the warmup is U2K's 32 x 2K); rows 1..319 hold exactly 287 x 2K
  and 32 x 8K (90/10: 288 / 32 in all) or 159 x 2K and 160 x 8K (50/50: 160 / 160) in the order
  random.Random(int(sha256("<seed>:stage_v:v4:<mix>:<r>"), 16)).shuffle gives them; every output 128. The number of
  8K requests in W is the dataset's (window_counts), known before any data.
- The minis (20 rows, the mini's N): v1-mini-8192 is V1's rule on 8,064-token prompts with rows 5 / 6 / 7 pinned to 1 /
  2 / 512 output tokens (the mini's L-U8K at C 4 has them in W = [4, 16)); v4-mini has row 0 2K, rows 1-8 8K (the
  mini's first wave at C 9 is 1 + 8 x 4 = 33 units > 32, so one request must queue) and its other 11 rows 6 x 2K +
  5 x 8K shuffled. 044 names no seed for them: "<seed>:stage_v:<key>".
- spec(key) is pure in the seed (no tokenizer): stage_v_analysis recomputes every pinned output and every V4 class
  from it. rows_for(key) is fixed at import, so run_stage_d.mini_num_prompts' patch never changes a formal dataset.
- Rows are {"prompt", "output_tokens"} JSONL in order (the bench reads them with --disable-shuffle); each file's
  sha256 is pinned in configs/stage_v.toml [stage_v.datasets] before the mini.

Usage (WSL, CPU only, the project venv, offline HF cache, under flock -n like E1-P's seed check):
    .venv/bin/python benchmarks/stage_v_datasets.py --out-dir data/stage_v [--dataset v1-2048-r1 ...]
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
import stage_v_plan as sv
from stage_d_datasets import vllm_sampler

DATASETS = sv.DATASETS
PROMPT_2K = sd.WORKLOADS["U2K"]["prompt"]          # 1,920: v-uniform-2k (V4), U2K-O512's prompt (V1)
PROMPT_8K = sd.WORKLOADS["U8K-gw"]["prompt"]       # 8,064: v-uniform-8k (V4), U8K-O512's prompt (V1)
V1_TIERS = {"2048": ("L-U2K", sd.WORKLOADS["U2K-O512"]["prompt"]),      # tier -> its point, its prompt
            "8192": ("L-U8K", sd.WORKLOADS["U8K-O512"]["prompt"])}
V4_LONG = {"90": 32, "50": 160}                    # 8K rows among rows 1..N-1 (the rest 2K): 288 / 32, 160 / 160
MIXES = {"v4-90": 0.9, "v4-50": 0.5}               # the 2K share of each V4 mix (044: measured_mixes)
OUT_RANGE = (1, 512)
ROW0_OUT = bp.N_GEN
MINI_OUTS = {5: 1, 6: 2, 7: 512}                   # v1-mini-8192: the shortest, the shortest with a TPOT, the longest
MINI_8K_HEAD = 8                                   # v4-mini rows 1-8
MINI_V4_REST = (6, 5)                              # v4-mini rows 9..19: 2K, 8K
_N = {**{f"v1-{t}": bp.num_prompts(sv.POINTS[p]["C"]) for t, (p, _) in V1_TIERS.items()},
      "v4": bp.num_prompts(sv.POINTS["W-90"]["C"]), "mini": sv.MINI["N"]}
_BLOCKS = range(1, sv.N_RUNS + 1)


def _rng(seed: int, name: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}:stage_v:{name}".encode()).hexdigest(), 16))


def _parse(key: str) -> tuple:
    """(kind, tier or mix, block) of a 044 dataset key; mini keys have no block."""
    if key in ("v1-mini-8192", "v4-mini"):
        return key, None, None
    parts = key.split("-")
    if key in DATASETS and len(parts) == 3 and parts[2] in {f"r{r}" for r in _BLOCKS}:
        return parts[0], parts[1], int(parts[2][1:])
    raise sv.PlanError(f"no Stage V dataset {key!r}; 044 defines {list(DATASETS)}")


def rows_for(key: str) -> int:
    kind, tier, _ = _parse(key)
    return _N["mini"] if tier is None else _N[f"v1-{tier}"] if kind == "v1" else _N["v4"]


def _v1(prompt: int, n: int, rng: random.Random) -> list:
    return [(prompt, ROW0_OUT)] + [(prompt, rng.randint(*OUT_RANGE)) for _ in range(n - 1)]


def spec(key: str, seed: int = bp.SEED) -> list:
    """[(prompt tokens, output tokens)] of every row of dataset key, in row order (044)."""
    kind, tier, block = _parse(key)
    n = rows_for(key)
    if key == "v1-mini-8192":
        rows = _v1(PROMPT_8K, n, _rng(seed, key))
        for i, out in MINI_OUTS.items():
            rows[i] = (PROMPT_8K, out)
        return rows
    if key == "v4-mini":
        rest = [PROMPT_2K] * MINI_V4_REST[0] + [PROMPT_8K] * MINI_V4_REST[1]
        _rng(seed, key).shuffle(rest)
        return [(p, bp.N_GEN) for p in [PROMPT_2K] + [PROMPT_8K] * MINI_8K_HEAD + rest]
    if kind == "v1":
        return _v1(V1_TIERS[tier][1], n, _rng(seed, f"v1:{tier}:{block}"))
    rest = [PROMPT_2K] * (n - 1 - V4_LONG[tier]) + [PROMPT_8K] * V4_LONG[tier]
    _rng(seed, f"v4:{tier}:{block}").shuffle(rest)
    return [(p, bp.N_GEN) for p in [PROMPT_2K] + rest]


def window_counts(key: str, C: int, seed: int = bp.SEED) -> dict:
    """The 2K ("short") and 8K ("long") rows of dataset key in W = [C, N - C)."""
    rows = spec(key, seed)
    w = [p for p, _ in rows[C:len(rows) - C]]
    return {"short": w.count(PROMPT_2K), "long": w.count(PROMPT_8K)}


def build(key: str, out: Path, *, sample: Callable, count_tokens: Callable, seed: int = bp.SEED) -> dict:
    """Write dataset key to out: row i's prompt is sample(input_len = its length, seed = seed + i), and every prompt
    must re-tokenize to exactly that length, else nothing is written (013)."""
    rows = spec(key, seed)
    prompts = [sample(input_len=p, seed=seed + i) for i, (p, _) in enumerate(rows)]
    bad = sum(count_tokens(text) != p for text, (p, _) in zip(prompts, rows))
    if bad:
        raise sv.PlanError(f"{key}: {bad} of {len(rows)} prompts do not re-tokenize to their target length (013)")
    data = "".join(json.dumps({"prompt": text, "output_tokens": o}, ensure_ascii=False) + "\n"
                   for text, (_, o) in zip(prompts, rows)).encode()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    lens = [p for p, _ in rows]
    outs = [o for _, o in rows]
    return {"dataset": key, "rows": len(rows), "prompts": {str(p): lens.count(p) for p in sorted(set(lens))},
            "outputs": [min(outs), max(outs)], "sha256": hashlib.sha256(data).hexdigest(), "path": str(out)}


def main(argv: list | None = None) -> int:  # pragma: no cover - WSL only (vLLM's RandomDataset and tokenizer)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", action="append", choices=list(DATASETS), help="default: all 22 (044)")
    args = parser.parse_args(argv)
    pins = bp.load_pins()
    sample, count = vllm_sampler(pins["core_b"]["repo"], pins["core_b"]["revision"])
    for key in args.dataset or DATASETS:
        print(json.dumps(build(key, Path(args.out_dir) / f"{key}.jsonl", sample=sample, count_tokens=count)),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
