"""Tests for benchmarks/stage_d_datasets.py (decision 039 D-C): seeded prompt lengths uniform on [64, tier - 128],
one RandomDataset prompt per row, exact re-tokenization or nothing written."""

import hashlib
import json

import pytest

import core_b_plan as bp
import stage_d_datasets as ds
import stage_d_plan as sd


def fake_sampler():
    def sample(*, input_len, seed):
        return " ".join(f"w{seed}" for _ in range(input_len))

    return sample, lambda p: len(p.split())


def test_lengths_are_seeded_uniform_on_the_range():
    a, b = ds.lengths(2048, 352), ds.lengths(2048, 352)
    assert a == b and min(a) >= 64 and max(a) <= 1920 and len(set(a)) > 200
    assert ds.lengths(8192, 10) != ds.lengths(2048, 10) and max(ds.lengths(8192, 400)) <= 8064


def test_rows_cover_every_request_and_warmup():
    assert ds.rows_for(2048) == bp.num_prompts(32) + bp.num_warmups(32) == 352
    assert ds.rows_for(8192) == bp.num_prompts(8) + bp.num_warmups(8) == 208
    assert ds.VAR_POINTS == {2048: sd.VERIFY["points"]["U2K-var"], 8192: sd.VERIFY["points"]["U8K-var"]}


def test_build_writes_exact_prompts_and_reports_the_sha256(tmp_path):
    sample, count = fake_sampler()
    out = ds.build(2048, tmp_path / "var-2048.jsonl", sample=sample, count_tokens=count)
    rows = [json.loads(x) for x in (tmp_path / "var-2048.jsonl").read_text().splitlines()]
    assert len(rows) == 352 and all(r["output_tokens"] == 128 for r in rows)
    assert [count(r["prompt"]) for r in rows] == ds.lengths(2048, 352)
    assert out["sha256"] == hashlib.sha256((tmp_path / "var-2048.jsonl").read_bytes()).hexdigest()


def test_a_prompt_that_does_not_re_tokenize_writes_nothing(tmp_path):
    sample, count = fake_sampler()
    with pytest.raises(sd.PlanError, match="re-tokenize"):
        ds.build(2048, tmp_path / "x.jsonl", sample=sample, count_tokens=lambda p: count(p) + 1)
    assert not (tmp_path / "x.jsonl").exists()
    with pytest.raises(sd.PlanError):
        ds.build(4096, tmp_path / "y.jsonl", sample=sample, count_tokens=count)
