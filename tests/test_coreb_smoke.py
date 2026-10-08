"""Tests for benchmarks/coreb_smoke.py: the staged Core B smoke test (decision 008).

S1 (2K) -> S2 (8K) -> S3 (32K), one request each, text-only. A stage runs only
if the previous stage's report says ok; the GPU must be idle before start; the
server must release its memory within 60 s after shutdown.
"""

import json
import signal
from pathlib import Path

import pytest

import coreb_smoke as cs

ROOT = Path(__file__).resolve().parent.parent
S1_LOG = ROOT / "results" / "core_b" / "smoke" / "20261008-s1-no-cc" / "S1.server.log"

# Lines 11 / 23 / 24 / 26 / 37 are verbatim from S1_LOG (vLLM 0.30.0 on the RTX); the
# KV cache line follows kv_cache_utils.py (S1 died before reaching it).
LOG_OK = """
(APIServer pid=3428) INFO 10-08 17:12:45 [registry.py:146] All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.
(EngineCore pid=3552) INFO 10-08 17:12:54 [qwen_gdn_linear_attn.py:176] Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).
(EngineCore pid=3552) INFO 10-08 17:12:54 [qwen_gdn_linear_attn.py:528] GDN decode kernel: triton
(EngineCore pid=3552) INFO 10-08 17:13:00 [cuda.py:538] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=3552) INFO 10-08 17:13:28 [interface.py:933] Setting attention block size to 528 tokens to ensure that attention page size is >= mamba page size.
(EngineCore pid=3552) INFO 10-08 17:13:40 [kv_cache_utils.py:2396] GPU KV cache size: 87,648 tokens, Maximum concurrency for 2,048 tokens per request: 24.71x
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
        "text_only": True,
        "gdn_prefill_kernel": "Triton/FLA",
        "gdn_prefill_requested": "triton",
        "gdn_decode_kernel": "triton",
        "gdn_warmup_failed": None,
        "kv_cache_tokens": 87648,
        "max_concurrency": 24.71,
        "max_model_len_logged": 2048,
    }


def test_parse_committed_s1_log():
    """Review F3: the attention-backend regex must match what vLLM 0.30.0 really prints."""
    facts = cs.parse_server_log(S1_LOG.read_text())
    assert facts["attention_backend"] == "FLASH_ATTN"
    assert facts["text_only"] is True
    assert facts["attn_block_size"] == 528
    assert (facts["gdn_prefill_kernel"], facts["gdn_decode_kernel"]) == ("Triton/FLA", "triton")


def test_parse_explicit_backend_line():
    assert cs.parse_server_log("[cuda.py:478] Using AttentionBackendEnum.FLASH_ATTN backend.")[
        "attention_backend"] == "FLASH_ATTN"


def test_parse_server_log_missing_lines():
    facts = cs.parse_server_log("nothing useful")
    assert set(facts.values()) == {None}


# ------------------------------------------------------------------ fakes --


class FakeProc:
    """API-server leader of a process group that also contains an EngineCore child."""

    pid = 4242

    def __init__(self, log_path, log_text, dies=False):
        self.log_path, self.dies = log_path, dies
        log_path.write_text(log_text)
        self.returncode = 1 if dies else None

    def poll(self):
        return self.returncode


class World:
    """Scripted GPU memory, HTTP, clock and process group."""

    def __init__(self, tmp_path, *, mem=None, response=None, log=LOG_OK, dies=False, healthy=True,
                 ignores=(), engine_survives=(), lock_free=True, post_raises=None, mem_raises_at=None):
        self.tmp_path = tmp_path
        self.mem = list(mem or [1000, 14000, 1100])  # before start, while serving, after shutdown
        self.response = response or {"choices": [{"text": " the" * 128}],
                                     "usage": {"prompt_tokens": 1800, "completion_tokens": 128}}
        self.log, self.dies, self.healthy = log, dies, healthy
        self.ignores, self.engine_survives = set(ignores), set(engine_survives)
        self.lock_free, self.post_raises, self.mem_raises_at = lock_free, post_raises, mem_raises_at
        self.t = 0.0
        self.requests, self.signals, self.mem_calls = [], [], 0
        self.engine_alive = False
        self.proc = None

    def gpu_mem_used(self):
        self.mem_calls += 1
        if self.mem_raises_at == self.mem_calls:
            raise RuntimeError("nvidia-smi: [N/A]")
        return self.mem.pop(0) if len(self.mem) > 1 else self.mem[0]

    def gpu_lock_free(self):
        return self.lock_free

    def popen(self, argv, env, log_path):
        self.argv, self.env = argv, env
        self.proc = FakeProc(log_path, self.log, dies=self.dies)
        self.engine_alive = True
        return self.proc

    def signal_group(self, proc, sig):
        self.signals.append(sig)
        if sig in self.ignores:
            return
        proc.returncode = 0 if proc.returncode is None else proc.returncode
        if sig not in self.engine_survives:
            self.engine_alive = False

    def group_alive(self, proc):
        return proc.returncode is None or self.engine_alive

    def http_get_ok(self, url):
        return self.healthy

    def http_post_json(self, url, payload, timeout):
        self.requests.append(payload)
        if self.post_raises:
            raise self.post_raises
        return self.response

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run(world, stage="S1", out=None, backend="triton", **kw):
    out = out or world.tmp_path / "smoke"
    return cs.run_stage(
        stage_name=stage, out_dir=out, vllm_bin="/v/bin/vllm", model="/snap", revision="a398",
        gdn_prefill_backend=backend, gpu_mem_used=world.gpu_mem_used, gpu_lock_free=world.gpu_lock_free,
        popen=world.popen, signal_group=world.signal_group, group_alive=world.group_alive,
        http_get_ok=world.http_get_ok, http_post_json=world.http_post_json, now=world.now,
        sleep=world.sleep, **kw,
    )


def test_stage_success_writes_ok_report(tmp_path):
    w = World(tmp_path)
    report = run(w)
    assert report["ok"] is True, report["reasons"]
    assert w.signals == [signal.SIGINT]  # clean SIGINT to the whole group is enough
    assert report["idle_mib"] == 1000
    assert report["log_facts"]["attn_block_size"] == 528
    assert report["mem_before_mib"] == 1000 and report["mem_after_mib"] == 1100
    assert w.requests[0]["prompt"] == cs.PROMPT_UNIT * 1800
    assert w.requests[0]["max_tokens"] == cs.MAX_TOKENS and w.requests[0]["temperature"] == 0
    saved = json.loads((tmp_path / "smoke" / "S1.json").read_text())
    assert saved == report
    assert "--language-model-only" in report["argv"]


S2_LOG = LOG_OK.replace("2,048 tokens per request", "8,192 tokens per request")
S2_OK = {"choices": [{"text": "x"}], "usage": {"prompt_tokens": 7900, "completion_tokens": 128}}


def test_next_stage_requires_previous_ok(tmp_path):
    with pytest.raises(cs.SmokeError, match="S1"):
        run(World(tmp_path), stage="S2")
    run(World(tmp_path))  # S1 ok
    w = World(tmp_path, response=S2_OK, log=S2_LOG)
    assert run(w, stage="S2")["ok"] is True


def test_next_stage_requires_the_same_configuration(tmp_path):
    """Review F9: S2 may only follow an S1 that passed with the same flags and env."""
    run(World(tmp_path))  # S1 ok with triton
    with pytest.raises(cs.SmokeError, match="configuration"):
        run(World(tmp_path, response=S2_OK, log=S2_LOG), stage="S2", backend="flashinfer")


def test_refuses_to_overwrite_existing_stage_evidence(tmp_path):
    """Review F9: a re-run must not replace an earlier failure report."""
    run(World(tmp_path, response={"choices": [], "usage": {}}))  # S1 fails
    before = (tmp_path / "smoke" / "S1.json").read_text()
    with pytest.raises(cs.SmokeError, match="exists"):
        run(World(tmp_path))
    assert (tmp_path / "smoke" / "S1.json").read_text() == before


def test_failed_previous_stage_blocks_next(tmp_path):
    run(World(tmp_path, response={"choices": [], "usage": {}}))  # S1 fails
    with pytest.raises(cs.SmokeError, match="S1"):
        run(World(tmp_path), stage="S2")


def test_busy_gpu_refuses_to_start(tmp_path):
    w = World(tmp_path, mem=[6000, 6000])
    with pytest.raises(cs.SmokeError, match="busy"):
        run(w, idle_mib=3000)
    assert not hasattr(w, "argv")


def test_default_idle_threshold_is_near_the_measured_baseline(tmp_path):
    """Review F8: 0.85 x 16303 MiB leaves ~2.4 GB; 3000 MiB would let a doomed start through."""
    w = World(tmp_path, mem=[1500, 1500])
    with pytest.raises(cs.SmokeError, match="busy"):
        run(w)
    assert not hasattr(w, "argv")


def test_gpu_lock_held_by_another_project_refuses_to_start(tmp_path):
    w = World(tmp_path, lock_free=False)
    with pytest.raises(cs.SmokeError, match="ai-portfolio-gpu.lock"):
        run(w)
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


def test_missing_text_only_line_fails(tmp_path):
    """Review F4: decision 008 requires the log to show text-only mode."""
    log = "\n".join(l for l in LOG_OK.splitlines() if "text-only" not in l)
    report = run(World(tmp_path, log=log))
    assert report["ok"] is False
    assert any("text_only" in r for r in report["reasons"])


def test_logged_max_model_len_must_match_stage(tmp_path):
    report = run(World(tmp_path, log=S2_LOG))  # S1 server that reports 8,192
    assert report["ok"] is False
    assert any("max_model_len" in r for r in report["reasons"])


def test_gdn_prefill_fallback_fails(tmp_path):
    """Review F10: requested flashinfer but vLLM silently ran Triton/FLA."""
    log = LOG_OK.replace("requested=triton", "requested=flashinfer")
    report = run(World(tmp_path, log=log), backend="flashinfer")
    assert report["ok"] is False
    assert any("fell back" in r for r in report["reasons"])


def test_gdn_warmup_failure_fails(tmp_path):
    log = LOG_OK + "WARNING [qwen_gdn_linear_attn.py:1173] GDN prefill kernel warmup (T=64) failed for layer x\n"
    report = run(World(tmp_path, log=log))
    assert report["ok"] is False
    assert any("warmup" in r for r in report["reasons"])


# ---------------------------------------------------------------- teardown --


def test_shutdown_escalates_to_group_kill(tmp_path):
    """Review F2: SIGINT/SIGTERM go to the whole process group, then SIGKILL."""
    w = World(tmp_path, ignores={signal.SIGINT, signal.SIGTERM})
    report = run(w)
    assert w.signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert report["ok"] is False and any("killed" in r for r in report["reasons"])


def test_orphaned_engine_core_is_not_missed(tmp_path):
    """The API server exits on SIGINT but EngineCore survives: the group is still alive."""
    w = World(tmp_path, engine_survives={signal.SIGINT})
    report = run(w)
    assert w.signals == [signal.SIGINT, signal.SIGTERM]
    assert report["ok"] is True, report["reasons"]
    assert not w.engine_alive


def test_residual_process_after_sigkill_is_reported(tmp_path):
    w = World(tmp_path, engine_survives={signal.SIGINT, signal.SIGTERM, signal.SIGKILL})
    report = run(w)
    assert report["ok"] is False
    assert any("residual" in r for r in report["reasons"])


def test_harness_error_after_popen_still_tears_down(tmp_path):
    w = World(tmp_path, mem_raises_at=2)  # nvidia-smi fails while the server is up
    report = run(w)
    assert w.signals and w.signals[0] == signal.SIGINT
    assert report["ok"] is False and any("harness error" in r for r in report["reasons"])
    assert (tmp_path / "smoke" / "S1.json").is_file()


def test_keyboard_interrupt_still_tears_down(tmp_path):
    w = World(tmp_path, post_raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run(w)
    assert w.signals == [signal.SIGINT] and not w.group_alive(w.proc)


# -------------------------------------------------------------------- main --


def _main(tmp_path, monkeypatch, run_stage):
    written = []

    def fake_run(cmd, check=False):
        out = cmd[cmd.index("--out") + 1]
        Path(out).write_text("{}")
        written.append(Path(out).name)

    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    monkeypatch.setattr(cs, "run_stage", run_stage)
    monkeypatch.setattr(cs.signal, "signal", lambda *a: None)
    rc = cs.main(["--stage", "S1", "--out", str(tmp_path / "s"), "--model", "/snap", "--revision", "a398",
                  "--gdn-prefill-backend", "triton"])
    return rc, written


def test_main_refuses_existing_evidence_before_capturing_env(tmp_path, monkeypatch):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "S1.json").write_text("{}")
    rc, written = _main(tmp_path, monkeypatch, lambda **kw: pytest.fail("must not run"))
    assert rc == 2 and written == []


def test_main_refusal_leaves_the_stage_retryable(tmp_path, monkeypatch):
    def refuse(**kw):
        raise cs.SmokeError("GPU busy")

    rc, _ = _main(tmp_path, monkeypatch, refuse)
    assert rc == 2
    assert not list((tmp_path / "s").glob("S1.*"))
