#!/usr/bin/env python3
"""Frozen task subsets for llama-perplexity --multiple-choice (decision 005).

Binary layout (pinned tools/perplexity/perplexity.cpp, little-endian):
    u32 n_task | u32 task_pos[n_task] | task...
    task    = str question | answers mc1 | answers mc2
    answers = u32 n | str[n] | i32 labels[n]
    str     = u32 len | bytes

The committed artifact is the index list (plus the sha256 of the built .bin),
not the seed: Python's random is not guaranteed stable across versions.
Sampling needs only the task count, so it is model-independent; the
tokenizer-dependent MMLU exclusion is applied afterwards from an explicit list.

Usage:
    python quality/mc_subset.py indices --n-total 13943 --n 4000 --seed 20261008 --out quality/subsets/mmlu-test-4000.json
    python quality/mc_subset.py build --src data/mmlu-test.bin --src-sha256 <pins> \\
        --indices quality/subsets/mmlu-test-4000.json --out data/mmlu-test-4000.bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

MAX_ANSWERS = 100  # same bound as the pinned reader


class SubsetError(ValueError):
    pass


class _Reader:
    def __init__(self, data: bytes, pos: int):
        self.data, self.pos = data, pos

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise SubsetError(f"truncated file at offset {self.pos}")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def string(self) -> str:
        return self.take(self.u32()).decode("utf-8", errors="replace")

    def answers(self) -> tuple:
        n = self.u32()
        if n > MAX_ANSWERS:
            raise SubsetError(f"{n} answers at offset {self.pos} (> {MAX_ANSWERS})")
        items = [self.string() for _ in range(n)]
        labels = list(struct.unpack(f"<{n}i", self.take(4 * n)))
        return items, labels


def _task_end(data: bytes, start: int) -> int:
    r = _Reader(data, start)
    r.string()
    r.answers()
    r.answers()
    return r.pos


def read_tasks(data: bytes) -> list:
    """Raw serialized bytes of every task, in file order."""
    r = _Reader(data, 0)
    n_task = r.u32()
    if n_task == 0:
        raise SubsetError("no tasks")
    positions = list(struct.unpack(f"<{n_task}I", r.take(4 * n_task)))
    tasks = []
    for i, start in enumerate(positions):
        end = _task_end(data, start)
        if i + 1 < n_task and end != positions[i + 1]:
            raise SubsetError(f"task {i} ends at {end}, next task starts at {positions[i + 1]}")
        tasks.append(data[start:end])
    return tasks


def parse_task(raw: bytes) -> dict:
    r = _Reader(raw, 0)
    question = r.string()
    items, labels = r.answers()
    r.answers()  # mc2: not used by llama.cpp scoring
    return {"question": question, "answers": items, "labels": labels}


def write_tasks(tasks: list) -> bytes:
    offset = 4 + 4 * len(tasks)
    positions = []
    for t in tasks:
        positions.append(offset)
        offset += len(t)
    return struct.pack("<I", len(tasks)) + struct.pack(f"<{len(tasks)}I", *positions) + b"".join(tasks)


def subset(data: bytes, indices: list) -> bytes:
    tasks = read_tasks(data)
    if len(set(indices)) != len(indices):
        raise SubsetError("duplicate indices")
    if any(i < 0 or i >= len(tasks) for i in indices):
        raise SubsetError(f"index out of range (n_task={len(tasks)})")
    return write_tasks([tasks[i] for i in sorted(indices)])


def select_indices(n_total: int, n: int, seed: int) -> list:
    return sorted(random.Random(seed).sample(range(n_total), n))


def apply_exclusions(indices: list, exclude: list) -> list:
    excluded = set(exclude)
    return [i for i in indices if i not in excluded]


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_idx = sub.add_parser("indices", help="sample an index list (model-independent)")
    p_idx.add_argument("--n-total", type=int, required=True)
    p_idx.add_argument("--n", type=int, required=True)
    p_idx.add_argument("--seed", type=int, required=True)
    p_idx.add_argument("--out", required=True)
    p_build = sub.add_parser("build", help="build the subset .bin from a pinned source file")
    p_build.add_argument("--src", required=True)
    p_build.add_argument("--src-sha256", required=True)
    p_build.add_argument("--indices", required=True)
    p_build.add_argument("--exclude", help="JSON list of source indices to drop (structural rule, decision 005)")
    p_build.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "indices":
        record = {"n_total": args.n_total, "n": args.n, "seed": args.seed,
                  "indices": select_indices(args.n_total, args.n, args.seed)}
        Path(args.out).write_text(json.dumps(record) + "\n")
        return 0

    data = Path(args.src).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.src_sha256:
        raise SubsetError(f"source sha256 {digest} != pinned {args.src_sha256}")
    record = json.loads(Path(args.indices).read_text())
    n_task = len(read_tasks(data))
    if record["n_total"] != n_task:
        raise SubsetError(f"index list n_total={record['n_total']} but source has {n_task} tasks")
    indices = record["indices"]
    if args.exclude:
        indices = apply_exclusions(indices, json.loads(Path(args.exclude).read_text()))
    out = subset(data, indices)
    Path(args.out).write_bytes(out)
    print(json.dumps({"out": args.out, "n_tasks": len(indices), "sha256": hashlib.sha256(out).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
