"""Tests for benchmarks/stage_e2_datasets.py (decision 041 v3 §2.8): the shared-prefix dataset is written only when
every row has exactly 1,920 tokens and the rows share >= 1,584 leading token ids."""

import json
import tomllib
from pathlib import Path

import pytest

import stage_e2_datasets as sd

ROOT = Path(__file__).resolve().parent.parent
SPEC = tomllib.loads((ROOT / "configs" / "stage_e2.toml").read_text())["e2"]["shared_prefix"]


def fake(prefix_shift=0, short=False):
    """Prompts are space-separated ids; the first prefix_len - prefix_shift ids are shared."""
    def sample_all(*, rows, prefix_len, input_len):
        shared = list(range(prefix_len - prefix_shift))
        out = []
        for r in range(rows):
            tail = [100000 + r * 10000 + i for i in range(input_len + prefix_shift - (1 if short and r == 3 else 0))]
            out.append(" ".join(map(str, shared + tail)))
        return out
    return sample_all, lambda p: [int(x) for x in p.split()]


def test_a_shared_prefix_dataset_is_written_with_its_common_prefix(tmp_path):
    sample_all, encode = fake()
    info = sd.build(tmp_path / "s.jsonl", spec=SPEC, sample_all=sample_all, encode=encode)
    rows = [json.loads(x) for x in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert info["rows"] == len(rows) == 275 and info["common_prefix"] == 1600 and rows[0]["output_tokens"] == 128
    assert len(info["sha256"]) == 64 and SPEC["min_shared"] == 1584 and SPEC["total"] == 1920


def test_nothing_is_written_when_a_row_is_short_or_the_prefix_too_small(tmp_path):
    sample_all, encode = fake(short=True)
    with pytest.raises(sd.DatasetError, match="re-tokenize"):
        sd.build(tmp_path / "a.jsonl", spec=SPEC, sample_all=sample_all, encode=encode)
    sample_all, encode = fake(prefix_shift=40)
    with pytest.raises(sd.DatasetError, match="common token prefix is 1560"):
        sd.build(tmp_path / "b.jsonl", spec=SPEC, sample_all=sample_all, encode=encode)
    assert not list(tmp_path.iterdir())


def test_the_common_prefix():
    assert sd.common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2 and sd.common_prefix_len([]) == 0
