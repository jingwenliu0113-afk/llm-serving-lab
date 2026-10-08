"""Tests for quality/mc_subset.py: frozen task subsets for llama-perplexity
--multiple-choice (decision 005).

The binary layout mirrors the pinned reader (tools/perplexity/perplexity.cpp,
multiple_choice_score / multiple_choice_task::deserialize):
  u32 n_task | u32 task_pos[n_task] | tasks...
  task = str question | answers mc1 | answers mc2
  answers = u32 n | str[n] | i32 labels[n]      str = u32 len | bytes
Sampling depends only on the task count (model-independent); the structural MMLU
exclusion is tokenizer-dependent and applied afterwards from an explicit list.
"""

import hashlib
import json
import struct

import pytest

import mc_subset as mcs


def s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<I", len(b)) + b


def answers(items, labels) -> bytes:
    return struct.pack("<I", len(items)) + b"".join(s(a) for a in items) + struct.pack(f"<{len(labels)}i", *labels)


def task(q, opts, gold) -> bytes:
    return s(q) + answers(opts, [int(i == gold) for i in range(len(opts))]) + answers([], [])


def build_bin(tasks: list) -> bytes:
    header = 4 + 4 * len(tasks)
    pos, off = [], header
    for t in tasks:
        pos.append(off)
        off += len(t)
    return struct.pack("<I", len(tasks)) + struct.pack(f"<{len(tasks)}I", *pos) + b"".join(tasks)


TASKS = [task(f"Q{i}?", [f"a{i}", f"b{i}", f"c{i}", f"d{i}"], i % 4) for i in range(10)]
BIN = build_bin(TASKS)


def test_read_tasks_returns_raw_task_bytes():
    assert mcs.read_tasks(BIN) == TASKS


def test_parse_task():
    t = mcs.parse_task(TASKS[3])
    assert t == {"question": "Q3?", "answers": ["a3", "b3", "c3", "d3"], "labels": [0, 0, 0, 1]}


def test_write_tasks_round_trip():
    assert mcs.write_tasks(mcs.read_tasks(BIN)) == BIN


def test_subset_keeps_task_bytes_in_index_order():
    sub = mcs.subset(BIN, [7, 2, 5])
    assert mcs.read_tasks(sub) == [TASKS[2], TASKS[5], TASKS[7]]


def test_subset_rejects_bad_indices():
    with pytest.raises(mcs.SubsetError):
        mcs.subset(BIN, [1, 1])
    with pytest.raises(mcs.SubsetError):
        mcs.subset(BIN, [10])


def test_select_indices_deterministic_sorted_unique():
    a = mcs.select_indices(13943, 4000, seed=20261008)
    assert a == mcs.select_indices(13943, 4000, seed=20261008)
    assert a == sorted(set(a)) and len(a) == 4000 and 0 <= a[0] and a[-1] < 13943
    assert a != mcs.select_indices(13943, 4000, seed=1)


def test_apply_exclusions_after_sampling():
    assert mcs.apply_exclusions([1, 4, 6, 9], exclude=[4, 5, 9]) == [1, 6]


def test_corrupt_header_rejected():
    with pytest.raises(mcs.SubsetError):
        mcs.read_tasks(BIN[:-3])
    with pytest.raises(mcs.SubsetError):
        mcs.read_tasks(struct.pack("<I", 0))


def test_cli_indices_then_build(tmp_path):
    idx_file = tmp_path / "idx.json"
    assert mcs.main(["indices", "--n-total", "10", "--n", "4", "--seed", "3", "--out", str(idx_file)]) == 0
    record = json.loads(idx_file.read_text())
    assert record["n_total"] == 10 and record["seed"] == 3 and len(record["indices"]) == 4

    src = tmp_path / "src.bin"
    src.write_bytes(BIN)
    out = tmp_path / "sub.bin"
    rc = mcs.main(["build", "--src", str(src), "--src-sha256", hashlib.sha256(BIN).hexdigest(),
                   "--indices", str(idx_file), "--out", str(out)])
    assert rc == 0
    assert mcs.read_tasks(out.read_bytes()) == [TASKS[i] for i in record["indices"]]


def test_cli_build_refuses_wrong_source_hash(tmp_path):
    idx_file = tmp_path / "idx.json"
    idx_file.write_text(json.dumps({"n_total": 10, "seed": 0, "indices": [0, 1]}))
    src = tmp_path / "src.bin"
    src.write_bytes(BIN)
    with pytest.raises(mcs.SubsetError, match="sha256"):
        mcs.main(["build", "--src", str(src), "--src-sha256", "0" * 64, "--indices", str(idx_file),
                  "--out", str(tmp_path / "o.bin")])


def test_cli_build_refuses_task_count_mismatch(tmp_path):
    idx_file = tmp_path / "idx.json"
    idx_file.write_text(json.dumps({"n_total": 11, "seed": 0, "indices": [0, 1]}))
    src = tmp_path / "src.bin"
    src.write_bytes(BIN)
    with pytest.raises(mcs.SubsetError, match="n_total"):
        mcs.main(["build", "--src", str(src), "--src-sha256", hashlib.sha256(BIN).hexdigest(),
                  "--indices", str(idx_file), "--out", str(tmp_path / "o.bin")])
