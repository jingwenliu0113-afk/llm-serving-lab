#!/usr/bin/env python3
"""Structural MMLU exclusion list for llama-perplexity --multiple-choice (decision 005).

Rule (score-blind, decided before any scoring): exclude a task if some option's
token sequence `question + " " + answer` is entirely inside the token prefix
shared by all options, i.e. min_len == common_prefix. The pinned reader
(tools/perplexity/perplexity.cpp, multiple_choice_score) reads
seq_tokens[s][common_prefix] for every option and goes out of bounds on such
tasks. Tokenization mirrors llama.cpp common_tokenize(ctx, text, add_special=true):
Qwen3.5 adds no BOS/EOS, so the HF tokenizer is used without special tokens.

Usage (needs the `tokenizers` package; the converter venv has it):
    third_party/convert-venv/bin/python quality/mmlu_exclusions.py \\
        --src data/datasets/mmlu-test.bin --indices quality/subsets/mmlu-test-4000.json \\
        --tokenizer models/hf/Qwen__Qwen3.5-9B@c2022362/tokenizer.json \\
        --out quality/subsets/mmlu-test-4000.exclude.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_subset  # noqa: E402

RULE = ("exclude a task if min_s len(tok(question + ' ' + answer_s)) == common_prefix over all options "
        "(pinned perplexity.cpp reads seq_tokens[s][common_prefix]; out of bounds otherwise)")


def common_prefix_len(seqs: list) -> int:
    n = 0
    for column in zip(*seqs):
        if any(tok != column[0] for tok in column):
            break
        n += 1
    return n


def is_excluded(seqs: list) -> bool:
    return min(len(s) for s in seqs) == common_prefix_len(seqs)


def compute_exclusions(data: bytes, indices: list, tokenize: Callable) -> list:
    tasks = mc_subset.read_tasks(data)
    excluded = []
    for i in sorted(indices):
        t = mc_subset.parse_task(tasks[i])
        seqs = [tokenize(t["question"] + " " + a) for a in t["answers"]]
        if is_excluded(seqs):
            excluded.append(i)
    return excluded


def make_record(*, excluded: list, n_subset: int, tokenizer_desc: dict, source_sha256: str,
                indices_file: str) -> dict:
    return {
        "rule": RULE,
        "tokenizer": tokenizer_desc,
        "source_sha256": source_sha256,
        "indices_file": indices_file,
        "n_subset": n_subset,
        "excluded": excluded,
        "n_after": n_subset - len(excluded),
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json of the pinned Core A snapshot")
    parser.add_argument("--tokenizer-repo", default="Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer  # lazy: only the CLI needs it

    root = Path(__file__).resolve().parent.parent
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    data = Path(args.src).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pins["quality"]["files"]["mmlu"]["sha256"]:
        raise SystemExit(f"source sha256 {digest} != pinned mmlu sha256")
    tok = Tokenizer.from_file(args.tokenizer)
    record_in = json.loads(Path(args.indices).read_text())
    excluded = compute_exclusions(data, record_in["indices"],
                                  lambda text: tok.encode(text, add_special_tokens=False).ids)
    rec = make_record(
        excluded=excluded, n_subset=len(record_in["indices"]),
        tokenizer_desc={"repo": args.tokenizer_repo,
                        "sha256": hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),
                        "library": "tokenizers", "add_special_tokens": False},
        source_sha256=digest, indices_file=args.indices,
    )
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("n_subset", "n_after", "excluded")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
