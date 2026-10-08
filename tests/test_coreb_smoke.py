"""Tests for benchmarks/coreb_smoke.py: the staged Core B smoke test (decision 008).

S1 (2K) -> S2 (8K) -> S3 (32K), one request each, text-only. A stage runs only
if the previous stage's report says ok; the GPU must be idle before start; the
server must release its memory within 60 s after shutdown.
"""

import json

import pytest

import coreb_smoke as cs

LOG_OK = """
INFO 10-08 12:00:01 [cuda.py:478] Using FLASH_ATTN backend.
INFO 10-08 12:00:02 [interface.py:934] Setting attention block size to 528 tokens to ensure that attention page size is >= mamba page size.
INFO 10-08 12:00:03 [qwen_gdn_linear_attn.py:176] Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).
INFO 10-08 12:00:03 [qwen_gdn_linear_attn.py:528] GDN decode kernel: triton
INFO 10-08 12:00:09 [kv_cache_utils.py:2396] GPU KV cache size: 87,648 tokens, Maximum concurrency for 2,048 tokens per request: 24.71x
"""


def test_stages_are_the_preregistered_ladder():
    assert [(s.name, s.max_model_len, s.prompt_tokens) for s in cs.STAGES] == [
        ("S1", 2048, 1800), ("S2", 8192, 7900), ("S3", 32768, 32500),
    ]
    assert all(s.prompt_tokens + cs.MAX_TOKENS <= s.max_model_len for s in cs.STAGES)


def test_serve_argv_and_env_are_text_only_and_pinned():
    argv, env = cs.serve_command(
        vllm_bin="/v/bin/vllm", model="/snap", revision="a398", stage=cs.STAGES[0],
        gdn_prefill_backend="triton", port=8765, base_env={"PATH": "/usr/bin"},
    )
    assert argv[:3] == ["/v/bin/vllm", "serve", "/snap"]
    for flag in ("--language-model-only", "--no-enable-prefix-caching"):
        assert flag in argv
    pairs = dict(zip(argv, argv[1:]))
    assert pairs["--dtype"] == "bfloat16"
    assert pairs["--max-model-len"] == "2048"
    assert pairs["--max-num-seqs"] == "8"
    assert pairs["--gpu-memory-utilization"] == "0.85"
    assert pairs["--gdn-prefill-backend"] == "triton"
    assert pairs["--generation-config"] == "vllm"
    assert pairs["--host"] == "127.0.0.1"
    assert env["VLLM_GDN_DECODE_KERNEL"] == "triton"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HOME"] == "/home/wenn0113/.cache/huggingface"
    assert env["PATH"] == "/usr/bin"


def test_parse_server_log():
    assert cs.parse_server_log(LOG_OK) == {
        "attention_backend": "FLASH_ATTN",
        "attn_block_size": 528,
        "gdn_prefill_kernel": "Triton/FLA",
        "gdn_prefill_requested": "triton",
        "gdn_decode_kernel": "triton",
        "kv_cache_tokens": 87648,
        "max_concurrency": 24.71,
        "max_model_len_logged": 2048,
    }


def test_parse_server_log_missing_lines():
    facts = cs.parse_server_log("nothing useful")
    assert set(facts.values()) == {None}


# ------------------------------------------------------------------ fakes --


class FakeProc:
    def __init__(self, log_path, log_text, dies=False):
        self.log_path, self.dies, self.killed = log_path, dies, False
        log_path.write_text(log_text)
        self.returncode = 1 if dies else None

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.returncode = 0

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class World:
    """Scripted GPU memory, HTTP and clock."""

    def __init__(self, tmp_path, *, mem=None, response=None, log=LOG_OK, dies=False, healthy=True):
        self.tmp_path = tmp_path
        self.mem = list(mem or [1000, 14000, 1100])  # before start, while serving, after shutdown
        self.response = response or {"choices": [{"text": " the" * 128}],
                                     "usage": {"prompt_tokens": 1800, "completion_tokens": 128}}
        self.log, self.dies, self.healthy = log, dies, healthy
        self.t = 0.0
        self.requests = []

    def gpu_mem_used(self):
        return self.mem.pop(0) if len(self.mem) > 1 else self.mem[0]

    def popen(self, argv, env, log_path):
        self.argv, self.env = argv, env
        return FakeProc(log_path, self.log, dies=self.dies)

    def http_get_ok(self, url):
        return self.healthy

    def http_post_json(self, url, payload, timeout):
        self.requests.append(payload)
        return self.response

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run(world, stage="S1", out=None, **kw):
    out = out or world.tmp_path / "smoke"
    return cs.run_stage(
        stage_name=stage, out_dir=out, vllm_bin="/v/bin/vllm", model="/snap", revision="a398",
        gdn_prefill_backend="triton", gpu_mem_used=world.gpu_mem_used, popen=world.popen,
        http_get_ok=world.http_get_ok, http_post_json=world.http_post_json, now=world.now,
        sleep=world.sleep, **kw,
    )


def test_stage_success_writes_ok_report(tmp_path):
    w = World(tmp_path)
    report = run(w)
    assert report["ok"] is True, report["reasons"]
    assert report["log_facts"]["attn_block_size"] == 528
    assert report["mem_before_mib"] == 1000 and report["mem_after_mib"] == 1100
    assert w.requests[0]["prompt"] == cs.PROMPT_UNIT * 1800
    assert w.requests[0]["max_tokens"] == cs.MAX_TOKENS and w.requests[0]["temperature"] == 0
    saved = json.loads((tmp_path / "smoke" / "S1.json").read_text())
    assert saved == report
    assert "--language-model-only" in report["argv"]


def test_next_stage_requires_previous_ok(tmp_path):
    with pytest.raises(cs.SmokeError, match="S1"):
        run(World(tmp_path), stage="S2")
    run(World(tmp_path))  # S1 ok
    w = World(tmp_path, response={"choices": [{"text": "x"}],
                                  "usage": {"prompt_tokens": 7900, "completion_tokens": 128}})
    assert run(w, stage="S2")["ok"] is True


def test_failed_previous_stage_blocks_next(tmp_path):
    run(World(tmp_path, response={"choices": [], "usage": {}}))  # S1 fails
    with pytest.raises(cs.SmokeError, match="S1"):
        run(World(tmp_path), stage="S2")


def test_busy_gpu_refuses_to_start(tmp_path):
    w = World(tmp_path, mem=[6000, 6000])
    with pytest.raises(cs.SmokeError, match="busy"):
        run(w, idle_mib=3000)
    assert not hasattr(w, "argv")


def test_token_count_mismatch_fails(tmp_path):
    w = World(tmp_path, response={"choices": [{"text": "x"}],
                                  "usage": {"prompt_tokens": 1799, "completion_tokens": 128}})
    report = run(w)
    assert report["ok"] is False
    assert any("prompt_tokens" in r for r in report["reasons"])


def test_memory_not_released_fails(tmp_path):
    w = World(tmp_path, mem=[1000, 1000] + [9000] * 200)
    report = run(w)
    assert report["ok"] is False
    assert any("release" in r for r in report["reasons"])


def test_server_dies_during_startup_fails_with_log_tail(tmp_path):
    w = World(tmp_path, dies=True, log="RuntimeError: Failed to find C compiler. Please specify via CC\n")
    report = run(w)
    assert report["ok"] is False
    assert any("exited" in r for r in report["reasons"])
    assert "C compiler" in report["log_tail"]


def test_missing_log_facts_fail(tmp_path):
    report = run(World(tmp_path, log="INFO started\n"))
    assert report["ok"] is False
    assert any("kv_cache_tokens" in r for r in report["reasons"])
