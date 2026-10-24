"""Tests for benchmarks/stage_v_datasets.py (decision 044): the V1 output sequences (row 0 128, the rest seeded uniform
on [1, 512]), the V4 class sequences (row 0 2K, exact counts 288 / 32 and 160 / 160 in seeded order), the two mini
datasets, one RandomDataset prompt per row and exact re-tokenization or nothing written. A stub sampler stands in for
vLLM, which only the WSL CLI loads."""

import hashlib
import json
import random

import pytest

import core_b_plan as bp
import stage_d_datasets as sdd
import stage_d_plan as sd
import stage_v_datasets as svd
import stage_v_plan as sv

FORMAL = [k for k in sv.DATASETS if "mini" not in k]


def fake_sampler():
    def sample(*, input_len, seed):
        return " ".join(f"w{seed}" for _ in range(input_len))

    return sample, lambda p: len(p.split())


def test_the_module_builds_every_dataset_044_pins_and_nothing_else():
    assert set(svd.DATASETS) == set(sv.DATASETS) and len(sv.DATASETS) == 22
    for key in sv.DATASETS:
        assert len(svd.spec(key)) == svd.rows_for(key)
    for key in ("var-2048", "mixed-8192", "v1-2048-r6", "v1-4096-r1", "v4-75-r1", "v4-mini-2", ""):
        with pytest.raises(sv.PlanError, match="044"):
            svd.spec(key)
    assert svd.vllm_sampler is sdd.vllm_sampler                     # vLLM's own RandomDataset, as D-C's var datasets


def test_rows_are_each_points_n():
    assert {k: svd.rows_for(k) for k in ("v1-2048-r1", "v1-8192-r5", "v4-90-r3", "v4-50-r1", "v1-mini-8192",
                                          "v4-mini")} == {"v1-2048-r1": 320, "v1-8192-r5": 200, "v4-90-r3": 320,
                                                          "v4-50-r1": 320, "v1-mini-8192": 20, "v4-mini": 20}
    assert svd.rows_for("v1-2048-r1") == bp.num_prompts(sv.POINTS["L-U2K"]["C"])
    assert svd.rows_for("v1-8192-r1") == bp.num_prompts(sv.POINTS["L-U8K"]["C"])
    assert svd.rows_for("v4-50-r1") == bp.num_prompts(sv.POINTS["W-50"]["C"]) == bp.num_prompts(sv.POINTS["W-90"]["C"])
    assert svd.rows_for("v4-mini") == svd.rows_for("v1-mini-8192") == sv.MINI["N"]


def test_rows_do_not_follow_the_minis_num_prompts_patch(monkeypatch):
    monkeypatch.setattr(bp, "num_prompts", lambda C: 20)              # run_stage_d.mini_num_prompts during a mini
    assert len(svd.spec("v1-2048-r1")) == 320 and len(svd.spec("v4-90-r1")) == 320


def test_v1_prompts_are_the_o512_bases_and_outputs_are_seeded_uniform_on_1_512():
    for tier, base in (("2048", "U2K-O512"), ("8192", "U8K-O512")):
        for r in range(1, 6):
            rows = svd.spec(f"v1-{tier}-r{r}")
            assert {p for p, _ in rows} == {sd.WORKLOADS[base]["prompt"]}
            assert rows[0][1] == 128                                  # the warmups resend row 0 (no ready check)
            outs = [o for _, o in rows[1:]]
            assert min(outs) >= 1 and max(outs) <= 512 and len(set(outs)) > 100
            rng = random.Random(int(hashlib.sha256(f"20261008:stage_v:v1:{tier}:{r}".encode()).hexdigest(), 16))
            assert outs == [rng.randint(1, 512) for _ in range(len(rows) - 1)]
    seqs = [tuple(svd.spec(f"v1-2048-r{r}")) for r in range(1, 6)]
    assert len(set(seqs)) == 5                                        # block r reads its own sequence
    assert svd.spec("v1-2048-r1") == svd.spec("v1-2048-r1")
    assert svd.spec("v1-2048-r1", seed=1) != svd.spec("v1-2048-r1")


def test_v4_has_the_exact_counts_row_0_2k_and_a_seeded_order():
    for mix, (short, long_) in (("90", (288, 32)), ("50", (160, 160))):
        seqs = []
        for r in range(1, 6):
            rows = svd.spec(f"v4-{mix}-r{r}")
            prompts = [p for p, _ in rows]
            assert rows[0] == (1920, 128)                             # the warmup is U2K's 32 x 2K
            assert prompts.count(1920) == short and prompts.count(8064) == long_ and len(rows) == 320
            assert {o for _, o in rows} == {128}
            rest = [1920] * (short - 1) + [8064] * long_
            random.Random(int(hashlib.sha256(f"20261008:stage_v:v4:{mix}:{r}".encode()).hexdigest(), 16)).shuffle(rest)
            assert prompts[1:] == rest
            seqs.append(tuple(prompts))
        assert len(set(seqs)) == 5
    assert (sd.WORKLOADS["U2K"]["prompt"], sd.WORKLOADS["U8K-gw"]["prompt"]) == (svd.PROMPT_2K, svd.PROMPT_8K)
    assert svd.MIXES == {"v4-90": 0.9, "v4-50": 0.5}


def test_the_8k_count_in_w_is_fixed_by_the_datasets():
    """044 V4: the number of 8K requests in W = [32, 288) is the dataset's, known before any data (pinned here)."""
    got = {k: svd.window_counts(k, 32) for k in FORMAL if k.startswith("v4-")}
    assert got == {"v4-90-r1": {"short": 230, "long": 26}, "v4-90-r2": {"short": 232, "long": 24},
                   "v4-90-r3": {"short": 231, "long": 25}, "v4-90-r4": {"short": 229, "long": 27},
                   "v4-90-r5": {"short": 229, "long": 27}, "v4-50-r1": {"short": 128, "long": 128},
                   "v4-50-r2": {"short": 129, "long": 127}, "v4-50-r3": {"short": 130, "long": 126},
                   "v4-50-r4": {"short": 127, "long": 129}, "v4-50-r5": {"short": 130, "long": 126}}
    rows = svd.spec("v4-90-r1")
    assert got["v4-90-r1"]["long"] == sum(p == 8064 for p, _ in rows[32:288])


def test_the_mini_datasets_first_rows():
    v1 = svd.spec("v1-mini-8192")
    assert len(v1) == 20 and {p for p, _ in v1} == {8064}
    assert v1[0][1] == 128 and [o for _, o in v1[5:8]] == [1, 2, 512]
    assert all(1 <= o <= 512 for _, o in v1)
    rng = random.Random(int(hashlib.sha256(b"20261008:stage_v:v1-mini-8192").hexdigest(), 16))
    drawn = [rng.randint(1, 512) for _ in range(19)]
    assert [o for i, (_, o) in enumerate(v1) if i not in (0, 5, 6, 7)] == [o for i, o in enumerate(drawn, 1)
                                                                          if i not in (5, 6, 7)]
    v4 = svd.spec("v4-mini")
    assert [p for p, _ in v4[:9]] == [1920] + [8064] * 8              # first wave at C 9: 1 + 8 x 4 = 33 units > 32
    assert sum(1 if p == 1920 else 4 for p, _ in v4[:9]) == 33
    rest = [p for p, _ in v4[9:]]
    assert sorted(rest) == [1920] * 6 + [8064] * 5 and {o for _, o in v4} == {128}
    want = [1920] * 6 + [8064] * 5
    random.Random(int(hashlib.sha256(b"20261008:stage_v:v4-mini").hexdigest(), 16)).shuffle(want)
    assert rest == want


def test_build_writes_one_exact_random_prompt_per_row_and_reports_the_sha256(tmp_path):
    sample, count = fake_sampler()
    out = svd.build("v4-50-r2", tmp_path / "d" / "v4-50-r2.jsonl", sample=sample, count_tokens=count)
    raw = (tmp_path / "d" / "v4-50-r2.jsonl").read_bytes()
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    want = svd.spec("v4-50-r2")
    assert [(count(r["prompt"]), r["output_tokens"]) for r in rows] == want
    assert [r["prompt"].split()[0] for r in rows] == [f"w{bp.SEED + i}" for i in range(320)]   # RandomDataset(seed + i)
    assert set(rows[0]) == {"prompt", "output_tokens"}
    assert out == {"dataset": "v4-50-r2", "rows": 320, "prompts": {"1920": 160, "8064": 160}, "outputs": [128, 128],
                   "sha256": hashlib.sha256(raw).hexdigest(), "path": str(tmp_path / "d" / "v4-50-r2.jsonl")}
    v1 = svd.build("v1-mini-8192", tmp_path / "m.jsonl", sample=sample, count_tokens=count)
    assert v1["rows"] == 20 and v1["prompts"] == {"8064": 20} and v1["outputs"] == [1, 512]


def test_prompts_are_written_as_utf_8_text_as_d_cs_var_datasets(tmp_path):
    def sample(*, input_len, seed):
        return " ".join("é" for _ in range(input_len))

    svd.build("v4-mini", tmp_path / "u.jsonl", sample=sample, count_tokens=lambda p: len(p.split()))
    raw = (tmp_path / "u.jsonl").read_bytes()
    assert "é".encode() in raw and b"\\u00e9" not in raw


def test_an_inexact_re_tokenization_writes_nothing(tmp_path):
    sample, count = fake_sampler()
    one_off = lambda p: count(p) + (1 if p.startswith(f"w{bp.SEED + 7} ") else 0)   # noqa: E731 - a single row
    with pytest.raises(sv.PlanError, match="1 of 320 prompts"):
        svd.build("v1-2048-r1", tmp_path / "x.jsonl", sample=sample, count_tokens=one_off)
    assert not (tmp_path / "x.jsonl").exists()
    with pytest.raises(sv.PlanError):
        svd.build("var-2048", tmp_path / "y.jsonl", sample=sample, count_tokens=count)
    assert not (tmp_path / "y.jsonl").exists()


def test_every_dataset_keeps_the_row_0_rule_and_outputs_in_1_512():
    for key in sv.DATASETS:
        rows = svd.spec(key)
        assert rows[0] == ((8064 if "8192" in key else 1920), 128), key     # V4's row 0 is 2K
        assert all(1 <= o <= 512 for _, o in rows) and all(p in (1920, 8064) for p, _ in rows)
        if key.startswith("v4-"):
            assert {o for _, o in rows} == {128}
    outs = [o for k in sv.DATASETS if k.startswith("v1-") for _, o in svd.spec(k)[1:]]
    assert min(outs) == 1 and max(outs) == 512                        # both ends drawn: the range is [1, 512]
