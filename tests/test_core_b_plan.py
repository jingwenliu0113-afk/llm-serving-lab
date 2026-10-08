"""Tests for benchmarks/core_b_plan.py: decision 007 grid rules (hybrid C_KV of 008 / 009)
and the `vllm bench serve` commands, fixed before any NVIDIA data."""

import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp

ROOT = Path(__file__).resolve().parent.parent


def test_c_kv_hybrid_formula():
    # 009 example: 160 blocks, block_size 528 -> 22 / 8 / 2
    assert [bp.c_kv(160, 528, L) for L in (2048, 8192, 32768)] == [22, 8, 2]


def test_round_half_up_not_bankers():
    assert [bp.round_half_up(x) for x in (2.5, 3.5, 2.4999, 0.5)] == [3, 4, 2, 1]


def test_grid_rule_007():
    g = bp.concurrency_grid(22)
    # powers of two up to the smallest >= max(8, 8 x 22) = 176 -> 256 (cap 256)
    assert {1, 2, 4, 8, 16, 32, 64, 128, 256} <= set(g)
    # refinement round(22 x 2^(k/2)), k = -2..2 -> 11, 16 (15.56), 22, 31 (31.11), 44
    assert {11, 16, 22, 31, 44} <= set(g)
    assert g == sorted(set(g)) and max(g) == 256


def test_grid_small_c_kv_and_cap():
    assert bp.concurrency_grid(1) == [1, 2, 4, 8]  # C_hi = max(8, 8) = 8; refinement 1 (0.5 -> 1), 1, 1, 1, 2
    assert bp.concurrency_grid(2) == [1, 2, 3, 4, 8, 16]  # refinement 1, 1 (1.41), 2, 3 (2.83), 4
    assert max(bp.concurrency_grid(200)) == 256  # 8 x 200 = 1600 -> capped at 256
    assert all(c <= 256 for c in bp.concurrency_grid(200))


def test_prompts_and_warmups():
    assert bp.num_prompts(1) == 200 and bp.num_prompts(32) == 320
    assert bp.num_warmups(16) == 16


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_uniform_bench_argv(pins):
    argv = bp.bench_argv(pins, L=8192, C=16, mix="uniform", result_dir="/r", result_filename="x.json", port=8000)
    assert argv[:3] == ["{vllm}", "bench", "serve"]
    assert arg(argv, "--dataset-name") == "random"
    assert (arg(argv, "--random-input-len"), arg(argv, "--random-output-len")) == ("8064", "128")
    assert arg(argv, "--random-range-ratio") == "0"
    assert (arg(argv, "--max-concurrency"), arg(argv, "--num-prompts"), arg(argv, "--num-warmups")) == ("16", "200", "16")
    assert arg(argv, "--request-rate") == "inf" and "--ignore-eos" in argv
    assert arg(argv, "--percentile-metrics") == "ttft,tpot,itl,e2el,client_queue_time"
    assert arg(argv, "--metric-percentiles") == "50,95,99"
    assert {"--save-result", "--save-detailed"} <= set(argv)
    assert arg(argv, "--seed") == "20261008"
    assert arg(argv, "--model") == pins["core_b"]["repo"]


def test_mixed_bench_argv_uses_the_seeded_custom_dataset(pins):
    argv = bp.bench_argv(pins, L=2048, C=8, mix="mixed", result_dir="/r", result_filename="x.json", port=8000,
                         mixed_dataset="/d/mixed-2048.jsonl")
    assert arg(argv, "--dataset-name") == "custom" and arg(argv, "--dataset-path") == "/d/mixed-2048.jsonl"
    assert "--skip-chat-template" in argv and "--random-input-len" not in argv
    with pytest.raises(bp.PlanError):
        bp.bench_argv(pins, L=32768, C=8, mix="mixed", result_dir="/r", result_filename="x.json", port=8000,
                      mixed_dataset="/d/x.jsonl")  # mixed only at 2048 / 8192 (007)


def test_server_argv_is_shared_by_all_levels(pins):
    argv, env = bp.server_argv(pins, gdn_prefill_backend="triton", gpu_memory_utilization="0.90", port=8000)
    assert arg(argv, "--max-model-len") == "32768" and arg(argv, "--max-num-seqs") == "264"
    assert {"--language-model-only", "--no-enable-prefix-caching"} <= set(argv)
    assert arg(argv, "--revision") == pins["core_b"]["revision"]
    assert env["VLLM_GDN_DECODE_KERNEL"] == "triton"


def test_uniform_plan_orders_c_by_seed_per_restart(pins):
    plan = bp.build_uniform_plan(pins, num_gpu_blocks=160, block_size=528, seed=20261008)
    assert plan["c_kv"] == {"2048": 22, "8192": 8, "32768": 2}
    runs = plan["runs"]
    assert {(r["L"], r["restart"]) for r in runs} == {(L, k) for L in (2048, 8192, 32768) for k in (1, 2, 3)}
    for r in runs:
        assert sorted(r["order"]) == plan["grids"][str(r["L"])]
    orders = [tuple(r["order"]) for r in runs if r["L"] == 2048]
    assert len(set(orders)) > 1  # each restart gets its own seeded order
    assert bp.build_uniform_plan(pins, num_gpu_blocks=160, block_size=528, seed=20261008) == plan


def test_mixed_dataset_is_blocks_of_ten_so_any_prefix_is_exactly_ten_percent_long():
    rows = bp.mixed_rows(short=[f"s{i}" for i in range(9 * 256)], long=[f"l{i}" for i in range(256)], seed=20261008)
    assert len(rows) == bp.MIXED_ROWS == 2560
    for n in (200, 320, 2560):  # num-prompts is always a multiple of 10
        head = rows[:n]
        assert sum(r["prompt"].startswith("l") for r in head) == n // 10
    assert all(r["output_tokens"] == 128 for r in rows)
    assert [r["prompt"] for r in rows if r["prompt"].startswith("s")][:5] == ["s0", "s1", "s2", "s3", "s4"]
    assert rows == bp.mixed_rows(short=[f"s{i}" for i in range(9 * 256)], long=[f"l{i}" for i in range(256)],
                                 seed=20261008)
    assert bp.mixed_counts(200) == (180, 20) and bp.mixed_counts(320) == (288, 32)


def test_mixed_bench_reads_the_file_in_order(pins):
    argv = bp.bench_argv(pins, L=8192, C=8, mix="mixed", result_dir="/r", result_filename="x.json", port=8000,
                         mixed_dataset="/d/mixed-8192.jsonl")
    assert "--disable-shuffle" in argv


def test_mixed_dataset_builder_uses_uniform_generator_for_short_prompts(tmp_path):
    import json

    import core_b_mixed_dataset as md

    calls = []

    def fake_sample(*, num_requests, input_len, seed):
        calls.append((num_requests, input_len, seed))
        return [f"{input_len}:{seed}:{i}" for i in range(num_requests)]

    out = tmp_path / "mixed-2048.jsonl"
    info = md.build(2048, out, sample=fake_sample)
    assert calls == [(2304, 1920, 20261008), (256, 32640, 20261009)]
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 2560 and info["rows"] == 2560 and len(info["sha256"]) == 64
    short = [r["prompt"] for r in rows if r["prompt"].startswith("1920:")]
    assert short[:3] == ["1920:20261008:0", "1920:20261008:1", "1920:20261008:2"]
