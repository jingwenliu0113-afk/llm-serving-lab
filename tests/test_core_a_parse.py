"""Tests for benchmarks/core_a_parse.py.

Real fixtures (tests/fixtures/*.metal.*) come from the pinned build (b11146,
Metal). Success-path and OOM texts below are SYNTHETIC: they follow the exact
printf formats in the pinned source (llama-bench jsonl_printer, common_perf_print,
common_memory_breakdown_print, ggml-cuda / ggml-metal allocation errors) and must
be replaced by real captures once models are available (NEXT.md).
"""

import json
from pathlib import Path

import pytest

import core_a_parse as cap

FIX = Path(__file__).resolve().parent / "fixtures"
COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"


def bench_line(**over):
    row = {
        "build_commit": COMMIT, "build_number": 11146, "cpu_info": "Apple M4 Pro", "gpu_info": "Apple M4 Pro",
        "backends": "MTL,BLAS", "model_filename": "/m/Qwen3-8B-Q4_K_M.gguf", "model_type": "qwen3 8B Q4_K - Medium",
        "model_size": 5027783488, "model_n_params": 8190735360, "n_batch": 2048, "n_ubatch": 512, "n_threads": 8,
        "cpu_mask": "0x0", "cpu_strict": False, "poll": 0, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 999,
        "n_cpu_moe": 0, "split_mode": "layer", "main_gpu": 0, "no_kv_offload": False, "flash_attn": 1,
        "devices": "auto", "tensor_split": "0.00", "tensor_buft_overrides": "none", "load_mode": "mmap",
        "lazy_mode": "off", "embeddings": False, "no_op_offload": 0, "no_host": False, "fit_target": 0,
        "fit_min_ctx": 0, "n_prompt": 2048, "n_gen": 0, "n_depth": 0, "test_time": "2026-10-08T08:00:00Z",
        "avg_ns": 2500000000, "stddev_ns": 10000000, "avg_ts": 819.2, "stddev_ts": 3.1,
        "samples_ns": [2490000000, 2500000000, 2510000000, 2495000000, 2505000000],
        "samples_ts": [822.5, 819.2, 816.0, 820.8, 817.6],
    }
    row.update(over)
    return json.dumps(row)


EXPECT_TTFT = {
    "n_prompt": 2048, "n_gen": 0, "n_depth": 0, "build_commit": COMMIT, "n_threads": 8, "poll": 0, "reps": 5,
}


def log(msg, level="info"):
    return json.dumps({"type": "log", "time": 1, "level": level, "msg": msg + "\n"})


BREAKDOWN = json.dumps({"type": "fit_memory_breakdown", "data": {"unit": "MiB", "rows": [
    {"kind": "device", "name": "MTL0", "description": "Apple M4 Pro", "total": 38338, "free": 28000,
     "self": 6100, "model": 4790, "context": 288, "compute": 1022, "unaccounted": 4238},
    {"kind": "host", "name": "Host", "self": 340, "model": 333, "context": 0, "compute": 7},
]}})

COMPLETION_OK = "\n".join([
    log("llama_context: n_ctx                 = 2048"),
    log("llama_context: flash_attn            = enabled"),
    log("generate: n_ctx = 2048, n_batch = 2048, n_predict = 128, n_keep = 0"),
    "the the the",  # generated text is printed raw, not as JSON
    log("common_perf_print: prompt eval time =     812.33 ms /  1920 tokens (    0.42 ms per token,  2363.58 tokens per second)"),
    log("common_perf_print:        eval time =    2911.20 ms /   127 runs   (   22.92 ms per token,    43.62 tokens per second)"),
    BREAKDOWN,
])

CUDA_LOAD_OOM = "\n".join([
    log("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 15012.00 MiB on device 0: cudaMalloc failed: out of memory", "error"),
    log("alloc_tensor_range: failed to allocate CUDA0 buffer of size 15741026304", "error"),
    log("llama_model_load: error loading model: unable to allocate CUDA0 buffer", "error"),
    log("llama_model_load_from_file_impl: failed to load model", "error"),
    log("cmn  common_init_: failed to load model '/m/Qwen3-14B-Q8_0.gguf'", "error"),
    log("llama_completion: error: unable to create context", "error"),
])

CUDA_CONTEXT_OOM = "\n".join([
    log("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5120.00 MiB on device 0: cudaMalloc failed: out of memory", "error"),
    log("llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache", "error"),
    log("cmn  common_init_: failed to create context with model '/m/Qwen3-14B-Q6_K.gguf'", "error"),
    log("llama_completion: error: unable to create context", "error"),
])

METAL_EVAL_OOM = "\n".join([
    log("llama_context: flash_attn            = enabled"),
    log("generate: n_ctx = 32768, n_batch = 2048, n_predict = 128, n_keep = 0"),
    log("ggml_metal_synchronize: error: command buffer 0 failed with status 5", "error"),
    log("error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)", "error"),
    log("llama_completion : failed to eval", "error"),
])


# ------------------------------------------------------------- llama-bench --


def test_parse_bench_jsonl_ignores_non_json_lines():
    rows = cap.parse_bench_jsonl("some banner\n" + bench_line() + "\n")
    assert len(rows) == 1 and rows[0]["avg_ns"] == 2500000000


def test_bench_metrics_and_no_violations():
    (row,) = cap.parse_bench_jsonl(bench_line())
    assert cap.check_bench_row(row, EXPECT_TTFT) == []
    m = cap.bench_metrics(row)
    assert m == {
        "avg_ns": 2500000000, "stddev_ns": 10000000, "avg_ts": 819.2, "stddev_ts": 3.1,
        "samples_ns": [2490000000, 2500000000, 2510000000, 2495000000, 2505000000],
    }


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"n_prompt": 512}, "n_prompt"),
        ({"flash_attn": -1}, "flash_attn"),
        ({"type_k": "q8_0"}, "type_k"),
        ({"n_gpu_layers": 20}, "n_gpu_layers"),
        ({"load_mode": "none"}, "load_mode"),
        ({"build_commit": "deadbeef"}, "build_commit"),
        ({"n_threads": 4}, "n_threads"),
        ({"samples_ns": [1, 2, 3]}, "samples"),
        ({"n_batch": 512}, "n_batch"),
    ],
)
def test_bench_violations_detected(over, needle):
    (row,) = cap.parse_bench_jsonl(bench_line(**over))
    violations = cap.check_bench_row(row, EXPECT_TTFT)
    assert any(needle in v for v in violations), violations


# ------------------------------------------------------- llama-completion --


def test_parse_log_jsonl_splits_logs_events_and_raw():
    parsed = cap.parse_log_jsonl(COMPLETION_OK)
    assert len(parsed.logs) == 5
    assert parsed.events["fit_memory_breakdown"][0]["unit"] == "MiB"
    assert parsed.raw_lines == 1


def test_completion_summary():
    s = cap.completion_summary(cap.parse_log_jsonl(COMPLETION_OK))
    assert s["n_ctx"] == 2048
    assert s["flash_attn"] == "enabled"
    assert s["prompt_tokens"] == 1920
    assert s["eval_runs"] == 127
    assert s["memory"] == {
        "unit": "MiB",
        "devices": [{"name": "MTL0", "description": "Apple M4 Pro", "total": 38338, "free": 28000, "self": 6100,
                     "model": 4790, "context": 288, "compute": 1022, "unaccounted": 4238}],
        "host": {"self": 340, "model": 333, "context": 0, "compute": 7},
        "other": [],
    }


def test_completion_summary_missing_everything():
    s = cap.completion_summary(cap.parse_log_jsonl(""))
    assert s == {"n_ctx": None, "flash_attn": None, "prompt_tokens": None, "eval_runs": None, "memory": None}


# ---------------------------------------------------------- classification --


def test_classify_completion_success():
    parsed = cap.parse_log_jsonl(COMPLETION_OK)
    out = cap.classify_completion(0, parsed, "", expected_prompt_tokens=1920)
    assert out["outcome"] == "ok", out


def test_classify_completion_wrong_prompt_length_is_error():
    parsed = cap.parse_log_jsonl(COMPLETION_OK)
    out = cap.classify_completion(0, parsed, "", expected_prompt_tokens=8064)
    assert out["outcome"] == "error"
    assert "prompt_tokens" in out["reason"]


def test_classify_completion_exit0_without_breakdown_is_error():
    text = COMPLETION_OK.replace(BREAKDOWN, "")
    out = cap.classify_completion(0, cap.parse_log_jsonl(text), "", expected_prompt_tokens=1920)
    assert out["outcome"] == "error" and "memory breakdown" in out["reason"]


@pytest.mark.parametrize(
    "text, stage",
    [(CUDA_LOAD_OOM, "load"), (CUDA_CONTEXT_OOM, "context"), (METAL_EVAL_OOM, "eval")],
)
def test_classify_completion_oom_stages(text, stage):
    out = cap.classify_completion(1, cap.parse_log_jsonl(text), "", expected_prompt_tokens=1920)
    assert out["outcome"] == "oom"
    assert out["stage"] == stage
    assert out["evidence"]


def test_classify_real_missing_model_is_not_oom():
    text = (FIX / "completion_missing_model.metal.jsonl").read_text()
    out = cap.classify_completion(1, cap.parse_log_jsonl(text), "", expected_prompt_tokens=1920)
    assert out["outcome"] == "error"
    assert out["stage"] == "load"


def test_pinned_memory_debug_message_is_not_oom():
    text = log("ggml_cuda_host_malloc: failed to allocate 512.00 MiB of pinned memory: out of memory", "debug")
    assert cap.find_oom([json.loads(text)["msg"]]) is None


def test_classify_bench_success():
    out = cap.classify_bench(0, bench_line(), "", EXPECT_TTFT)
    assert out["outcome"] == "ok" and out["metrics"]["avg_ns"] == 2500000000


def test_classify_bench_real_missing_model():
    stderr = (FIX / "bench_missing_model.metal.stderr.txt").read_text()
    out = cap.classify_bench(1, "", stderr, EXPECT_TTFT)
    assert out["outcome"] == "error" and out["stage"] == "load"


def test_classify_bench_oom_from_stderr():
    stderr = "ggml_metal_buffer_init: error: failed to allocate buffer, size =  5120.00 MiB\n" \
             "llama_bench: error: failed to create context with model '/m/x.gguf'\n"
    out = cap.classify_bench(1, "", stderr, EXPECT_TTFT)
    assert out["outcome"] == "oom" and out["stage"] == "context"


def test_classify_bench_exit0_but_violation_is_error():
    out = cap.classify_bench(0, bench_line(flash_attn=0), "", EXPECT_TTFT)
    assert out["outcome"] == "error" and "flash_attn" in out["reason"]


def test_classify_crash_signal_without_oom_is_error():
    out = cap.classify_bench(-6, "", "GGML_ASSERT(x) failed\n", EXPECT_TTFT)
    assert out["outcome"] == "error" and "-6" in out["reason"]
