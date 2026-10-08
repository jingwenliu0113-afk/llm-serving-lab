"""Tests for quality/mmlu_exclusions.py: the structural, score-blind MMLU exclusion
rule of decision 005, mirroring the pinned llama.cpp reader.

tools/perplexity/perplexity.cpp (pinned) tokenizes `question + " " + answer` per
option, computes the shared token prefix, then reads
`seq_tokens[s][common_prefix]` for every option s. If some option's full
sequence IS the shared prefix (min length == common prefix), that read is out of
bounds. Such tasks are excluded before any scoring.
"""

import json
import struct

import mmlu_exclusions as mx


def s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<I", len(b)) + b


def task(q, opts):
    ans = struct.pack("<I", len(opts)) + b"".join(s(a) for a in opts) + struct.pack(f"<{len(opts)}i", *([1] + [0] * (len(opts) - 1)))
    return s(q) + ans + struct.pack("<I", 0)


def build_bin(tasks):
    off, pos = 4 + 4 * len(tasks), []
    for t in tasks:
        pos.append(off)
        off += len(t)
    return struct.pack("<I", len(tasks)) + struct.pack(f"<{len(tasks)}I", *pos) + b"".join(tasks)


def word_tokenize(text):
    return text.split(" ")  # one "token" per space-separated word


def test_common_prefix_len():
    assert mx.common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    assert mx.common_prefix_len([[1], [2]]) == 0
    assert mx.common_prefix_len([[7, 7], [7, 7]]) == 2


def test_rule_matches_out_of_bounds_condition():
    assert mx.is_excluded([[1, 2, 3], [1, 2, 4]]) is False
    assert mx.is_excluded([[1, 2], [1, 2, 4]]) is True  # option 0 lies inside the shared prefix
    assert mx.is_excluded([[5, 6], [5, 6]]) is True  # duplicate options


def test_compute_exclusions_over_subset_indices():
    tasks = [
        task("Q0 is", ["a", "b"]),          # fine
        task("Q1 is", ["x", "x y"]),         # "Q1 is x" is a prefix of "Q1 is x y" -> excluded
        task("Q2 is", ["same", "same"]),     # duplicates -> excluded
        task("Q3 is", ["c", "d"]),           # fine, but not in the subset
    ]
    out = mx.compute_exclusions(build_bin(tasks), indices=[0, 1, 2], tokenize=word_tokenize)
    assert out == [1, 2]


def test_record_contains_rule_and_provenance(tmp_path):
    rec = mx.make_record(excluded=[1, 2], n_subset=3, tokenizer_desc={"repo": "Qwen/Qwen3.5-9B", "sha256": "ab" * 32},
                         source_sha256="cd" * 32, indices_file="quality/subsets/x.json")
    assert rec["excluded"] == [1, 2] and rec["n_subset"] == 3 and rec["n_after"] == 1
    assert "common_prefix" in rec["rule"]
    assert rec["tokenizer"]["sha256"] == "ab" * 32
    json.dumps(rec)
