"""Tests for capserve/itest.py, the driver of decision 029's RTX integration tests IT-0..IT-6 (product tests only).
On the Mac: IT-0 against a fake vLLM source tree, IT-1..IT-5 against the fake vLLM behind the real gateway, IT-6
with a fake `find` and a real one in a temporary tree, and the orchestration (one supervisor boot, stop, results)
with every process effect faked. IT-5's GPU check is decision 031's: back to the supervisor run's pre-boot baseline
+ 64 MiB."""

import asyncio
import dataclasses
import datetime as dt
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

import core_b_plan as bp
import coreb_smoke as smoke
import gpu_idle as gi
from capserve import deploy, itest, runs
from capserve.config import Settings
from fakes.fake_vllm import FakeConfig
from fakes.servers import make_profile, stack

PINS = bp.load_pins()
ROOT = Path(__file__).resolve().parents[2]
NOW = dt.datetime(2026, 10, 16, 3, 4, 5, tzinfo=dt.timezone.utc)

PROTOCOL = '''
from pydantic import BaseModel

class OpenAIBaseModel(BaseModel):
    pass

class ChatCompletionRequest(OpenAIBaseModel):
    messages: list
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    chat_template_kwargs: dict | None = None

class TokenizeCompletionRequest(OpenAIBaseModel):
    model: str | None = None
    prompt: str
    add_special_tokens: bool = True

class TokenizeChatRequest(ChatTemplateMixin):
    model: str | None = None
    messages: list
    add_special_tokens: bool = False

class TokenizeResponse(OpenAIBaseModel):
    count: int
    tokens: list[int]
'''
MIXIN = '''
class ChatTemplateMixin(OpenAIBaseModel):
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template_kwargs: dict | None = None
'''
ENVS = '''
environment_variables = {
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": lambda: int(os.environ.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5")),
}
'''


def fake_vllm_tree(root: Path, *, protocol=PROTOCOL, mixin=MIXIN, envs=ENVS, launcher=None) -> Path:
    pkg = root / "vllm"
    (pkg / "entrypoints" / "openai").mkdir(parents=True)
    (pkg / "entrypoints" / "openai" / "protocol.py").write_text(protocol)
    (pkg / "entrypoints" / "chat_utils.py").write_text(mixin)
    (pkg / "entrypoints" / "openai" / "api_server.py").write_text('@router.post("/tokenize")\nasync def t(): pass\n')
    (pkg / "entrypoints" / "launcher.py").write_text(
        launcher if launcher is not None else "config = uvicorn.Config(app, timeout_keep_alive="
                                              "envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE)\n")
    (pkg / "envs.py").write_text(envs)
    return pkg


def check(result, name):
    (item,) = [c for c in result["checks"] if c["name"] == name]
    return item


# ------------------------------------------------------------------ IT-0 --

def test_it0_passes_on_the_expected_sources(tmp_path):
    res = itest.it0(locate=lambda: (fake_vllm_tree(tmp_path), "0.30.0"), pins=PINS)
    assert res["ok"] is True, [c for c in res["checks"] if not c["ok"]]
    assert res["data"]["keep_alive_s"] == 5 and res["data"]["vllm_version"] == "0.30.0"
    assert set(res["data"]["tokenize_chat_fields"]) >= {"messages", "add_generation_prompt", "add_special_tokens"}
    assert check(res, "import capserve.deploy")["ok"]


@pytest.mark.parametrize("change, failing", [
    (dict(mixin=MIXIN.replace("add_generation_prompt", "add_gen_prompt")),
     "/tokenize chat request accepts add_generation_prompt and add_special_tokens"),
    (dict(protocol=PROTOCOL.replace("    max_completion_tokens: int | None = None\n", "")),
     "chat completions accept max_completion_tokens"),
    (dict(protocol=PROTOCOL.replace("    add_special_tokens: bool = True\n", "")),
     "/tokenize completion request accepts add_special_tokens"),
    (dict(mixin=MIXIN.replace("    chat_template_kwargs: dict | None = None\n", "")),
     "/tokenize chat request accepts chat_template_kwargs"),
    (dict(protocol=PROTOCOL.replace("    chat_template_kwargs: dict | None = None\n", "")),
     "chat completions accept chat_template_kwargs"),
    (dict(envs=ENVS.replace('"5"', '"1"')), "vLLM keep-alive outlives the gateway pool's keepalive_expiry"),
    (dict(envs="x = 1\n"), "vLLM keep-alive outlives the gateway pool's keepalive_expiry"),
])
def test_it0_fails_when_vllm_differs(tmp_path, change, failing):
    res = itest.it0(locate=lambda: (fake_vllm_tree(tmp_path, **change), "0.30.0"), pins=PINS)
    assert res["ok"] is False and check(res, failing)["ok"] is False


def test_it0_keep_alive_falls_back_to_uvicorn_when_vllm_never_sets_it(tmp_path):
    res = itest.it0(locate=lambda: (fake_vllm_tree(tmp_path, envs="x = 1\n", launcher="pass\n"), "0.30.0"),
                    pins=PINS)
    assert res["data"]["keep_alive_s"] == 5 and res["data"]["keep_alive_source"] == "uvicorn default"
    assert check(res, "vLLM keep-alive outlives the gateway pool's keepalive_expiry")["ok"]


def test_it0_version_and_missing_vllm(tmp_path):
    res = itest.it0(locate=lambda: (fake_vllm_tree(tmp_path), "0.29.1"), pins=PINS)
    assert check(res, "vLLM version is the pinned 0.30.0")["ok"] is False
    res = itest.it0(locate=lambda: (None, None), pins=PINS)
    assert res["ok"] is False and check(res, "vLLM sources found")["ok"] is False


# ------------------------------------------------------------------ booted checks against the fake stack --

def ctx_for(st, tmp_path, *, lock=None, vllm_pid=None, gateway_pid=None):
    vport = int(st.gateway.settings.backend_url.rsplit(":", 1)[1])
    gport = int(st.url.rsplit(":", 1)[1])
    cfg = deploy.DeployConfig.model_validate({"service_root": str(tmp_path / "svc"), "vllm_port": vport,
                                              "gateway_port": gport, "gateway": {"active_class": "test"}})
    return itest.Ctx(cfg=cfg, settings=st.gateway.settings, run_dir=tmp_path, vllm_pid=vllm_pid,
                     gateway_pid=gateway_pid, lock_path=str(lock or tmp_path / "gpu.lock"),
                     proc_root=str(tmp_path / "proc"), poll_s=0.01, idle_wait_s=5.0, token_watch=(0.2, 0.6))


def fake_proc(tmp_path, pid, lock: Path):
    fd = tmp_path / "proc" / str(pid) / "fd"
    fd.mkdir(parents=True)
    os.symlink(lock, fd / "7")


@pytest.mark.anyio
async def test_it1_against_the_fake(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("")
    fake_proc(tmp_path, 11, lock)
    fake_proc(tmp_path, 12, lock)
    async with stack(FakeConfig(n_tokens=4)) as st:
        res = await itest.it1(ctx_for(st, tmp_path, lock=lock, vllm_pid=11, gateway_pid=12))
    assert res["ok"] is True, [c for c in res["checks"] if not c["ok"]]
    names = [c["name"] for c in res["checks"]]
    for want in ("gateway /readyz is 200", "/v1/models lists only the served model", "chat non-stream",
                 "chat stream ends with [DONE]", "completion non-stream", "completion stream ends with [DONE]",
                 "gateway /metrics has capserve_profile_info", "vLLM holds the GPU lock descriptor",
                 "gateway holds the GPU lock descriptor"):
        assert want in names
    sent = [s.body for s in st.fake.generations]
    assert any("max_completion_tokens" in b for b in sent)     # chat's bound reaches vLLM as max_completion_tokens


@pytest.mark.anyio
async def test_it1_fails_when_vllm_does_not_hold_the_lock(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("")
    fake_proc(tmp_path, 12, lock)
    async with stack(FakeConfig(n_tokens=2)) as st:
        res = await itest.it1(ctx_for(st, tmp_path, lock=lock, vllm_pid=11, gateway_pid=12))
    assert res["ok"] is False and check(res, "vLLM holds the GPU lock descriptor")["ok"] is False


@pytest.mark.anyio
async def test_it2_overload_against_the_fake(tmp_path):
    prof = make_profile(max_concurrency=2, max_queue=1)
    async with stack(FakeConfig(ttft_s=0.02, itl_s=0.005, n_tokens=8), prof, max_queue_wait_s=0.05) as st:
        res = await itest.it2(ctx_for(st, tmp_path))
    assert res["ok"] is True, [c for c in res["checks"] if not c["ok"]]
    assert check(res, "vLLM running + waiting never above max_concurrency")["ok"]
    assert (tmp_path / "it2" / "summary.json").is_file() and (tmp_path / "it2" / "requests.jsonl").is_file()


@pytest.mark.anyio
async def test_it3_disconnects_free_vllm_within_2s(tmp_path):
    prof = make_profile(max_output_tokens=1024)
    async with stack(FakeConfig(itl_s=0.01, n_tokens=1024, honor_max_tokens=True), prof) as st:
        res = await itest.it3(ctx_for(st, tmp_path))
        assert st.fake.cancelled_ids                                  # both requests were cancelled upstream
    assert res["ok"] is True, [c for c in res["checks"] if not c["ok"]]
    for case in ("stream", "non-stream"):
        assert check(res, f"{case}: vLLM running + waiting back to 0 within 2 s")["ok"]
        assert check(res, f"{case}: the generation was cut short (decisive)")["ok"]
        assert check(res, f"{case}: no tokens generated after the disconnect")["ok"]
        data = res["data"][case.replace("-", "_")]
        assert 0 < data["tokens_generated"] < data["max_tokens"] == 1024 and data["remaining_s"] is not None


@pytest.mark.anyio
async def test_it3_is_decisive_within_a_128_token_bound(tmp_path):
    """036's limitation: 128 tokens take ~2.6 s here, less than the old 4 s of remaining generation, so the time-based
    check could never decide; the token count does (038 §2)."""
    async with stack(FakeConfig(itl_s=0.02, n_tokens=1024, honor_max_tokens=True)) as st:   # profile bound: 128
        res = await itest.it3(ctx_for(st, tmp_path))
    assert res["ok"] is True, [c for c in res["checks"] if not c["ok"]]
    for case in ("stream", "non_stream"):
        assert res["data"][case]["max_tokens"] == 128 and res["data"][case]["tokens_generated"] < 128
        assert res["data"][case]["remaining_s"] < itest.DECISIVE_FACTOR * itest.DISCONNECT_WINDOW_S


@pytest.mark.anyio
async def test_it3_fails_when_the_whole_output_was_generated(tmp_path):
    """Delta = max_tokens: nothing shows the abort reached vLLM (too late, or never), so IT-3 fails (038 §2)."""
    async with stack(FakeConfig(itl_s=0.0, n_tokens=1024, honor_max_tokens=True)) as st:    # 128 tokens at once
        res = await itest.it3(ctx_for(st, tmp_path))
    assert res["ok"] is False
    for case in ("stream", "non-stream"):
        item = check(res, f"{case}: the generation was cut short (decisive)")
        assert item["ok"] is False and item["detail"].startswith("128 of 128 tokens")


@pytest.mark.anyio
async def test_it3_fails_when_generation_tokens_are_unreadable(tmp_path, monkeypatch):
    orig = itest._vllm_load

    async def no_tokens(vc):
        load, _ = await orig(vc)
        return load, None

    monkeypatch.setattr(itest, "_vllm_load", no_tokens)
    async with stack(FakeConfig(itl_s=0.01, n_tokens=1024, honor_max_tokens=True)) as st:
        res = await itest.it3(ctx_for(st, tmp_path))
    assert check(res, "stream: the generation was cut short (decisive)")["ok"] is False


@pytest.mark.anyio
async def test_it4_tokenize_parity_against_the_fake(tmp_path):
    async with stack() as st:
        res = await itest.it4(ctx_for(st, tmp_path))
    assert res["ok"] is True, [c for c in res["checks"] if not c["ok"]]
    assert check(res, "' the' x 256 is 256 tokens")["ok"]
    assert any(c["name"].startswith("chat bound holds") for c in res["checks"])
    assert res["data"]["chat"] and all("count" in m and "bound" in m for m in res["data"]["chat"])
    assert {m["thinking"] for m in res["data"]["chat"]} == {False, True}
    assert len(res["data"]["chat"]) == 2 * len(itest.CHAT_CASES)
    sent = [s.body for s in st.fake.seen if s.path == "/tokenize" and "messages" in s.body]
    assert {json.dumps(b["chat_template_kwargs"]) for b in sent} == {'{"enable_thinking": false}',
                                                                     '{"enable_thinking": true}'}
    assert res["data"]["settings_tokens"] == Settings.model_fields["tokens"].default.model_dump()


@pytest.mark.anyio
async def test_it4_fails_when_a_count_exceeds_the_bound(tmp_path, monkeypatch):
    async with stack() as st:
        ctx = ctx_for(st, tmp_path)
        ctx = dataclasses.replace(ctx, settings=st.gateway.settings.model_copy(
            update={"tokens": st.gateway.settings.tokens.model_copy(update={"chat_overhead": 0, "per_message": 0})}))
        orig = itest.PromptBudget.count

        async def inflated(self, route, body):
            return await orig(self, route, body) + (1000 if route == "chat" else 0)

        monkeypatch.setattr(itest.PromptBudget, "count", inflated)
        res = await itest.it4(ctx)
    assert res["ok"] is False and any(not c["ok"] and c["name"].startswith("chat bound holds")
                                      for c in res["checks"])


async def it4_with_template(tmp_path, monkeypatch, template: int) -> dict:
    """IT-4 against the fake, whose /tokenize counts words with no chat template, plus `template` chat tokens."""
    orig = itest.PromptBudget.count

    async def with_template(self, route, body):
        return await orig(self, route, body) + (template if route == "chat" else 0)

    monkeypatch.setattr(itest.PromptBudget, "count", with_template)
    async with stack() as st:
        return await itest.it4(ctx_for(st, tmp_path))


@pytest.mark.anyio
async def test_it4_passes_a_template_exactly_at_the_constants(tmp_path, monkeypatch):
    tok = Settings.model_fields["tokens"].default
    res = await it4_with_template(tmp_path, monkeypatch, tok.chat_overhead + tok.per_message + len("user"))
    assert res["ok"] is True, [c for c in res["checks"] if not c["ok"]]


@pytest.mark.anyio
async def test_it4_fails_a_template_just_above_the_constants(tmp_path, monkeypatch):
    """029 IT-4: the chat overhead must be within the configured constants (chat_overhead + per_message + role bytes
    per message), measured exactly: the chat count minus the messages' own tokens. The byte bound alone cannot fail
    it, because ' the' gives 4 bytes per token of slack."""
    tok = Settings.model_fields["tokens"].default
    res = await it4_with_template(tmp_path, monkeypatch, tok.chat_overhead + tok.per_message + len("user") + 1)
    assert res["ok"] is False
    failing = {c["name"] for c in res["checks"] if not c["ok"]}
    for thinking in ("off", "on"):
        assert f"chat template within the overhead constants: one user message (' the' x 16), thinking {thinking}" \
            in failing
        assert f"chat template within the overhead constants: empty user message, thinking {thinking}" in failing
        # the bound alone would have passed
        assert f"chat bound holds: one user message (' the' x 16), thinking {thinking}" not in failing
    ones = [m for m in res["data"]["chat"] if m["name"] == "one user message (' the' x 16)"]
    assert [(m["thinking"], m["content_tokens"], m["template_tokens"], m["allowed"]) for m in ones] == \
        [(False, 16, 53, 52), (True, 16, 53, 52)]


@pytest.mark.anyio
async def test_it4_measures_per_message_with_many_messages(tmp_path, monkeypatch):
    res = await it4_with_template(tmp_path, monkeypatch, 0)
    (many,) = [m for m in res["data"]["chat"] if m["name"].startswith("many messages") and not m["thinking"]]
    assert many["messages"] >= 8 and many["template_tokens"] == 0
    tok = Settings.model_fields["tokens"].default
    msgs = dict(itest.CHAT_CASES)[many["name"]]
    assert many["allowed"] == tok.chat_overhead + sum(tok.per_message + len(m["role"]) for m in msgs)
    assert res["ok"] is True


@pytest.mark.anyio
async def test_it4_counts_message_text_without_special_tokens(tmp_path):
    async with stack() as st:
        await itest.it4(ctx_for(st, tmp_path))
        texts = [s.body for s in st.fake.seen if s.path == "/tokenize" and "prompt" in s.body
                 and s.body.get("add_special_tokens") is False]
    assert {"model": PINS["core_b"]["repo"], "prompt": "Hi", "add_special_tokens": False} in texts


@pytest.mark.anyio
async def test_it5_stop_under_load_drains(tmp_path):
    prof = make_profile(max_concurrency=2, max_queue=2, max_output_tokens=1024)
    async with stack(FakeConfig(itl_s=0.002, n_tokens=150), prof) as st:
        def stop():
            fut = asyncio.run_coroutine_threadsafe(_drain(st), loop)
            fut.result(30)
            return {"stop_exit": 0, "supervisor_exit": 143,
                    "outcome": {"teardown": {"problems": [], "gateway": {"returncode": -15, "reasons": []},
                                             "vllm": {"returncode": -2, "shutdown_reasons": []},
                                             "gpu_baseline_mib": 600, "gpu_return_limit_mib": 664,
                                             "gpu_mem_after_mib": 640}}}

        loop = asyncio.get_running_loop()
        res = await itest.it5(ctx_for(st, tmp_path), stop=stop)
    assert res["ok"] is True, [c for c in res["checks"] if not c["ok"]]
    assert check(res, "requests running at the stop completed with 200 (drain)")["ok"]
    assert res["data"]["in_flight_at_stop"] == 2


async def _drain(st):
    st.gateway.set_draining()
    await st.gateway.ctrl.wait_idle(20)


def test_it5_evaluation_flags_a_dirty_teardown_and_cut_requests():
    records = [{"ended_after_stop": True, "status": 200, "error": None, "retry_after": None},
               {"ended_after_stop": True, "status": None, "error": "RemoteProtocolError", "retry_after": None},
               {"ended_after_stop": True, "status": 503, "error": None, "retry_after": None}]
    stopped = {"stop_exit": 0, "supervisor_exit": 1,
               "outcome": {"teardown": {"problems": ["gateway had to be killed"], "gpu_mem_after_mib": 1500,
                                        "gpu_baseline_mib": 1385, "gpu_return_limit_mib": 1449,
                                        "gateway": {"returncode": -9}, "vllm": {"shutdown_reasons": []}}}}
    checks = {c["name"]: c["ok"] for c in itest.evaluate_it5(records, stopped, saturated=True,
                                                               in_flight_at_stop=2)}
    assert checks["teardown had no problems"] is False
    assert checks[IT5_GPU] is False
    assert checks["the supervisor exited 0 or 143 (SIGTERM)"] is False
    assert checks["no request was cut (no transport error)"] is False
    assert checks["every 429 / 503 carries Retry-After"] is False
    assert checks["requests running at the stop completed with 200 (drain)"] is False


IT5_GPU = "GPU memory back <= the pre-boot baseline + 64 MiB (031)"


def _clean_stop(teardown: dict) -> dict:
    return {"stop_exit": 0, "supervisor_exit": 143, "outcome": {"teardown": {"problems": [], **teardown}}}


def _it5_gpu(stopped: dict) -> dict:
    records = [{"ended_after_stop": True, "status": 200, "error": None, "retry_after": None}]
    (item,) = [c for c in itest.evaluate_it5(records, stopped, saturated=True, in_flight_at_stop=1)
               if c["name"] == IT5_GPU]
    return item


def test_it5_gpu_check_is_the_runs_baseline_plus_64():
    """031: IT-5 needs GPU memory back <= the supervisor run's pre-boot baseline + 64 MiB (its teardown waited up to
    60 s), not the fixed 1,000 MiB: 1,440 MiB passes on a 1,385 MiB desktop baseline, 1,450 does not."""
    ok = _it5_gpu(_clean_stop({"gpu_baseline_mib": 1385, "gpu_return_limit_mib": 1449, "gpu_mem_after_mib": 1440}))
    assert ok["ok"] is True and ok["detail"] == {"after_mib": 1440, "baseline_mib": 1385, "limit_mib": 1449}
    over = _it5_gpu(_clean_stop({"gpu_baseline_mib": 1385, "gpu_return_limit_mib": 1449, "gpu_mem_after_mib": 1450}))
    assert over["ok"] is False


@pytest.mark.parametrize("teardown", [
    {"gpu_mem_after_mib": 600},                                                       # no 031 baseline recorded
    {"gpu_baseline_mib": 500, "gpu_return_limit_mib": 564, "gpu_mem_after_mib": None},  # nvidia-smi failed
    {"gpu_baseline_mib": 500, "gpu_return_limit_mib": 9999, "gpu_mem_after_mib": 900},  # IT-5 adds the 64 itself
])
def test_it5_gpu_check_fails_without_a_consistent_baseline_limit(teardown):
    assert _it5_gpu(_clean_stop(teardown))["ok"] is False


# ------------------------------------------------------------------ IT-6 --

def test_it6_with_a_real_find(tmp_path):
    research, cache = tmp_path / "llm-serving-lab", tmp_path / ".cache" / "vllm"
    research.mkdir()
    cache.mkdir(parents=True)
    (research / "old.txt").write_text("x")
    ref = tmp_path / "meta.json"
    ref.write_text("{}")
    past = time.time() - 100
    for p in (research / "old.txt", research, cache):
        os.utime(p, (past, past))
    res = itest.it6(reference=ref, paths=[research, cache, tmp_path / ".triton"])
    assert res["ok"] is True and res["data"]["newer"] == [] and res["data"]["missing"] == [str(tmp_path / ".triton")]
    (research / "results.json").write_text("{}")
    res = itest.it6(reference=ref, paths=[research, cache])
    assert res["ok"] is False and str(research / "results.json") in res["data"]["newer"]


def test_it6_paths_are_the_029_list(tmp_path):
    cfg = deploy.DeployConfig.model_validate({"service_root": str(tmp_path / "svc"), "gateway": {"active_class": "x"}})
    paths = itest.research_paths(cfg, home=Path("/home/wenn0113"))
    assert paths == [Path("/home/wenn0113/llm-serving-lab"), Path("/home/wenn0113/.cache/vllm"),
                     Path("/home/wenn0113/.cache/flashinfer"), Path("/home/wenn0113/.cache/huggingface"),
                     Path("/home/wenn0113/.triton")]


def test_it6_a_failing_find_is_not_a_pass(tmp_path):
    ref = tmp_path / "meta.json"
    ref.write_text("{}")

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "find: permission denied")

    res = itest.it6(reference=ref, paths=[tmp_path], run=run)
    assert res["ok"] is False


# ------------------------------------------------------------------ orchestration --

class Sup:
    def __init__(self):
        self.pid, self.returncode = 777, None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class Orchestra:
    """Fake effects for itest.main: the 'supervisor' writes state.json when started; stop() ends it."""

    def __init__(self, tmp_path):
        self.svc = tmp_path / "svc"
        self.cfg_path = tmp_path / "rtx.toml"
        self.cfg_path.write_text(f'[deploy]\nservice_root = "{self.svc}"\nmax_hold_s = 100\n'
                                 '[gateway]\nactive_class = "rtx-it"\n')
        self.cfg = deploy.load_config(self.cfg_path)
        self.started, self.ran, self.stops = [], [], 0
        self.sup = Sup()
        self.lock_held = True
        self.ready = True
        self.ready_at = None       # the supervisor's state.json says ready at this fake time (None: at once)
        self.t = 0.0
        self.installed = []

    def _become_ready(self):
        sup_run = self.svc / "runs" / "pt-20261016T030406Z"
        sup_run.mkdir(parents=True)
        (sup_run / "meta.json").write_text(json.dumps({"purpose": "product-test"}))
        (sup_run / "gateway.toml").write_text('active_class = "rtx-it"\nport = 8080\n')
        deploy.write_state(self.cfg, {"phase": "ready", "supervisor_pid": self.sup.pid, "vllm_pid": 1,
                                      "gateway_pid": 2, "run_id": sup_run.name, "run_dir": str(sup_run)},
                           wall=lambda: NOW)

    def start_group(self, argv, env, log_path, *, cwd, keep_fds=()):
        self.started.append({"argv": argv, "env": env, "keep_fds": tuple(keep_fds), "cwd": cwd})
        Path(log_path).write_text("")
        if not self.ready:
            self.sup.returncode = 2
        elif self.ready_at is None:
            self._become_ready()
        return self.sup

    def stop(self, cfg):
        self.stops += 1
        self.sup.returncode = 143
        run_dir = Path(deploy.read_state(cfg)["run_dir"])
        runs_meta = json.loads((run_dir / "meta.json").read_text())
        runs_meta["outcome"] = {"teardown": {"problems": [], "gpu_mem_after_mib": 600}}
        (run_dir / "meta.json").write_text(json.dumps(runs_meta))
        return 0

    def check(self, name):
        async def fn(ctx, **kw):
            self.ran.append((name, ctx.vllm_pid, ctx.gateway_pid))
            if "stop" in kw:
                kw["stop"]()
            return {"test": name, "ok": True, "checks": [{"name": "fake", "ok": True, "detail": None}]}
        return fn

    def fx(self):
        return itest.Effects(base_env={"PATH": "/usr/bin"}, start_group=self.start_group,
                             install_signals=lambda: self.installed.append("handlers"),
                             held_gpu_lock=lambda: self.lock_held, lock_fd=lambda: 9, python="/venv/bin/python",
                             wall=lambda: NOW, now=self.now, sleep=self.sleep, stop_supervisor=self.stop,
                             checks={t: self.check(t) for t in itest.BOOTED},
                             it0=lambda: {"test": "IT-0", "ok": True, "checks": []},
                             it6=lambda reference, paths: {"test": "IT-6", "ok": True, "checks": [],
                                                           "data": {"reference": str(reference)}})

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.ready_at is not None and self.t >= self.ready_at and deploy.read_state(self.cfg) is None:
            self._become_ready()


def production_orchestra(tmp_path):
    """Orchestra with a production config (036): [deploy] production_profile pins a verified profile."""
    o = Orchestra(tmp_path)
    o.cfg_path.write_text(f'[deploy]\nservice_root = "{o.svc}"\nmax_hold_s = 100\n'
                          'production_profile = "service/profiles/verified-formal-20261017-c1b.json"\n'
                          '[gateway]\nactive_class = "uniform-2k"\n')
    o.cfg = deploy.load_config(o.cfg_path)
    return o


def test_a_production_config_boots_its_pinned_profile_by_default(tmp_path):
    o = production_orchestra(tmp_path)
    assert itest.main(["IT-1", "--config", str(o.cfg_path)], fx=o.fx()) == 0
    (sup,) = o.started
    argv = sup["argv"]
    assert argv[argv.index("--profile") + 1] == str(deploy.RELEASE_ROOT / o.cfg.production_profile)
    assert "--allow-provisional" not in argv


@pytest.mark.parametrize("extra", [["--allow-provisional"],
                                   ["--profile", str(ROOT / "service" / "profiles" / "provisional-rtx.json")]])
def test_a_production_config_refuses_provisional_runs_before_anything_starts(tmp_path, capsys, extra):
    o = production_orchestra(tmp_path)
    assert itest.main(["IT-1", "--config", str(o.cfg_path), *extra], fx=o.fx()) == 2
    assert o.started == [] and not (o.svc / "runs").exists()
    assert "production_profile" in capsys.readouterr().err


def test_a_production_config_refuses_a_leftover_allow_provisional_env_before_anything_starts(tmp_path, capsys):
    """Review finding (036): itest checks the same effective value as deploy.main, flag or CAPSERVE_ALLOW_PROVISIONAL=1."""
    o = production_orchestra(tmp_path)
    fx = o.fx()
    fx.base_env = {**fx.base_env, "CAPSERVE_ALLOW_PROVISIONAL": "1"}
    assert itest.main(["IT-1", "--config", str(o.cfg_path)], fx=fx) == 2
    assert o.started == [] and not (o.svc / "runs").exists()
    err = capsys.readouterr().err
    assert "production_profile" in err and "CAPSERVE_ALLOW_PROVISIONAL" in err


def test_a_production_config_refuses_a_copy_of_the_pinned_profile_elsewhere(tmp_path, capsys):
    """Same file name, other directory: the pin is the release file itself, never its name."""
    o = production_orchestra(tmp_path)
    pinned = deploy.RELEASE_ROOT / o.cfg.production_profile
    copy = tmp_path / "elsewhere" / pinned.name
    copy.parent.mkdir()
    copy.write_bytes(pinned.read_bytes())
    assert itest.main(["IT-1", "--config", str(o.cfg_path), "--profile", str(copy)], fx=o.fx()) == 2
    assert o.started == [] and "production_profile" in capsys.readouterr().err


def test_an_integration_config_still_defaults_to_the_provisional_rtx_profile(tmp_path):
    o = Orchestra(tmp_path)
    assert itest.main(["IT-1", "--config", str(o.cfg_path), "--allow-provisional"], fx=o.fx()) == 0
    (sup,) = o.started
    argv = sup["argv"]
    assert argv[argv.index("--profile") + 1] == str(itest.DEFAULT_PROFILE.resolve())


def test_one_boot_runs_every_booted_test_in_order_then_stops(tmp_path, capsys):
    o = Orchestra(tmp_path)
    code = itest.main(["IT-6", "IT-2", "IT-1", "IT-0", "--config", str(o.cfg_path), "--allow-provisional"],
                      fx=o.fx())
    assert code == 0
    (sup,) = o.started
    assert sup["argv"][:5] == ["/venv/bin/python", "-m", "capserve.deploy", "up", "--config"]
    assert "--allow-provisional" in sup["argv"] and sup["keep_fds"] == (9,)
    assert sup["env"]["PYTHONPATH"].split(os.pathsep) == [str(ROOT / "service"), str(ROOT / "benchmarks")]
    assert [r[0] for r in o.ran] == ["IT-1", "IT-2"] and o.ran[0][1:] == (1, 2)
    assert o.stops == 1
    (run_dir,) = [d for d in (o.svc / "runs").iterdir() if d.name.endswith("-itest")]
    out = json.loads((run_dir / "itest.json").read_text())
    assert list(out["tests"]) == ["IT-0", "IT-1", "IT-2", "IT-6"] and out["ok"] is True
    assert out["supervisor"]["exit"] == 143 and out["supervisor"]["ok"] is True
    assert out["tests"]["IT-6"]["data"]["reference"] == str(run_dir / "meta.json")   # this run's start
    meta = runs.read_meta(run_dir)
    assert meta["kind"] == "itest" and meta["purpose"] == "product-test" and meta["outcome"]["ok"] is True


def test_it5_stops_the_supervisor_itself(tmp_path):
    o = Orchestra(tmp_path)
    assert itest.main(["IT-5", "IT-1", "--config", str(o.cfg_path), "--allow-provisional"], fx=o.fx()) == 0
    assert [r[0] for r in o.ran] == ["IT-1", "IT-5"] and o.stops == 1


def test_booted_tests_need_the_lock(tmp_path, capsys):
    o = Orchestra(tmp_path)
    o.lock_held = False
    assert itest.main(["IT-1", "--config", str(o.cfg_path)], fx=o.fx()) == 2
    assert o.started == [] and not (o.svc / "runs").exists() and "flock" in capsys.readouterr().err


def test_it0_needs_no_lock_and_no_boot(tmp_path):
    o = Orchestra(tmp_path)
    o.lock_held = False
    assert itest.main(["IT-0", "--config", str(o.cfg_path)], fx=o.fx()) == 0
    assert o.started == []


def test_it6_needs_the_lock_but_no_boot(tmp_path, capsys):
    """029: the RTX integration tests run under the GPU lock and only IT-0 is exempt; IT-6 must not interleave with
    a formal research session (027 §5), whose own writes it would blame on the product."""
    o = Orchestra(tmp_path)
    o.lock_held = False
    assert itest.main(["IT-6", "--config", str(o.cfg_path)], fx=o.fx()) == 2
    assert o.started == [] and not (o.svc / "runs").exists() and "flock" in capsys.readouterr().err
    o.lock_held = True
    assert itest.main(["IT-0", "IT-6", "--config", str(o.cfg_path)], fx=o.fx()) == 0
    assert o.started == []


def test_it6_since_another_run(tmp_path):
    o = Orchestra(tmp_path)
    earlier = runs.create_run(service_root=o.svc, research_dir=bp.PROJECT_DIR, run_id="pt-20261015T000000Z",
                              kind="deploy", release=None, profile=None, now_utc=NOW)
    assert itest.main(["IT-6", "--since", "pt-20261015T000000Z", "--config", str(o.cfg_path)], fx=o.fx()) == 0
    (run_dir,) = [d for d in (o.svc / "runs").iterdir() if d.name.endswith("-itest")]
    out = json.loads((run_dir / "itest.json").read_text())
    assert out["tests"]["IT-6"]["data"]["reference"] == str(earlier / "meta.json")
    assert itest.main(["IT-6", "--since", "formal-20261009-coreb-uniform", "--config", str(o.cfg_path)],
                      fx=o.fx()) == 2


def test_the_readiness_wait_covers_the_031_preflight(tmp_path):
    """The supervisor spends the 031 idle window (30 s) and two shared-usage readings (<= 20 s each) before it is
    ready, on top of its boot waits: itest waits for them too instead of failing a slow but healthy boot."""
    o = Orchestra(tmp_path)
    assert deploy.READY_OVERHEAD_S == 30 + 2 * gi.SHARED_USAGE_TIMEOUT_S
    o.ready_at = (o.cfg.startup_timeout_s + deploy.GATEWAY_READY_TIMEOUT_S + itest.SUPERVISOR_EXTRA_S
                  + deploy.READY_OVERHEAD_S - 5)
    assert itest.main(["IT-1", "--config", str(o.cfg_path), "--allow-provisional"], fx=o.fx()) == 0
    assert [r[0] for r in o.ran] == ["IT-1"]


def test_a_supervisor_that_never_gets_ready_fails_every_booted_test(tmp_path):
    o = Orchestra(tmp_path)
    o.ready = False
    assert itest.main(["IT-1", "IT-3", "--config", str(o.cfg_path), "--allow-provisional"], fx=o.fx()) == 1
    (run_dir,) = [d for d in (o.svc / "runs").iterdir() if d.name.endswith("-itest")]
    out = json.loads((run_dir / "itest.json").read_text())
    assert out["tests"]["IT-1"]["ok"] is False and out["tests"]["IT-3"]["ok"] is False and o.ran == []
    assert "supervisor.log" in out["tests"]["IT-1"]["error"]


def test_a_crashing_check_is_a_failed_test_and_the_supervisor_still_stops(tmp_path):
    o = Orchestra(tmp_path)
    fx = o.fx()

    async def boom(ctx):
        raise RuntimeError("bug")

    fx = dataclasses.replace(fx, checks={**fx.checks, "IT-2": boom})
    assert itest.main(["IT-1", "IT-2", "--config", str(o.cfg_path), "--allow-provisional"], fx=fx) == 1
    (run_dir,) = [d for d in (o.svc / "runs").iterdir() if d.name.endswith("-itest")]
    out = json.loads((run_dir / "itest.json").read_text())
    assert out["tests"]["IT-2"]["ok"] is False and "RuntimeError" in out["tests"]["IT-2"]["error"]
    assert o.stops == 1


def test_sigterm_and_sighup_end_the_itest_through_its_finally_blocks():
    """An ssh hang-up (SIGHUP) or a kill (SIGTERM) raises SystemExit(128 + signum), as in `capserve.deploy up`, so
    run_booted stops the supervisor and itest.json / meta.json are written."""
    calls = []
    itest.install_exit_handlers(set_handler=lambda sig, handler: calls.append((sig, handler)))
    assert calls == [(signal.SIGTERM, smoke._exit_on_signal), (signal.SIGHUP, smoke._exit_on_signal)]
    assert itest.Effects().install_signals is itest.install_exit_handlers


def test_main_installs_the_exit_handlers_before_anything(tmp_path):
    o = Orchestra(tmp_path)
    fx = dataclasses.replace(o.fx(), install_signals=lambda: o.installed.append(len(o.started)))
    assert itest.main(["IT-1", "--config", str(o.cfg_path), "--allow-provisional"], fx=fx) == 0
    assert o.installed == [0]


def test_an_interrupt_keeps_the_finished_results_and_stops_the_supervisor(tmp_path):
    o = Orchestra(tmp_path)
    fx = o.fx()

    async def hangup(ctx):
        raise SystemExit(128 + signal.SIGHUP)   # what the handler raises while IT-2 runs

    fx = dataclasses.replace(fx, checks={**fx.checks, "IT-2": hangup})
    with pytest.raises(SystemExit) as info:
        itest.main(["IT-1", "IT-2", "IT-3", "--config", str(o.cfg_path), "--allow-provisional"], fx=fx)
    assert info.value.code == 129 and o.stops == 1
    (run_dir,) = [d for d in (o.svc / "runs").iterdir() if d.name.endswith("-itest")]
    out = json.loads((run_dir / "itest.json").read_text())
    assert out["ok"] is False and out["interrupted"] == "SystemExit(129)"
    assert out["tests"]["IT-1"]["ok"] is True                                   # finished before the hang-up
    assert out["tests"]["IT-2"]["ok"] is False and out["tests"]["IT-3"]["ok"] is False
    assert "interrupted" in out["tests"]["IT-3"]["error"]
    assert out["supervisor"]["exit"] == 143 and out["supervisor"]["stop_exit"] == 0
    outcome = runs.read_meta(run_dir)["outcome"]
    assert outcome["ok"] is False and outcome["interrupted"] == "SystemExit(129)"


def test_cli_has_no_formal_option():
    for action in itest.parser()._actions:
        assert not any("formal" in s for s in action.option_strings)
    assert itest.DEFAULT_PROFILE == ROOT / "service" / "profiles" / "provisional-rtx.json"


# ------------------------------------------------------------------ module entry points (real subprocesses) --

def _module_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSERVE_")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "service"), str(ROOT / "benchmarks")])
    return env


def test_deploy_render_and_itest_it0_run_as_modules(tmp_path):
    import sys
    r = subprocess.run([sys.executable, "-m", "capserve.deploy", "render"], capture_output=True, text=True,
                       env=_module_env(), cwd=tmp_path, timeout=120)
    assert r.returncode == 0 and json.loads(r.stdout)["fingerprint"]
    cfg = tmp_path / "rtx.toml"
    cfg.write_text(f'[deploy]\nservice_root = "{tmp_path / "svc"}"\n[gateway]\nactive_class = "rtx-it"\n')
    r = subprocess.run([sys.executable, "-m", "capserve.itest", "IT-0", "--config", str(cfg)], capture_output=True,
                       text=True, env=_module_env(), cwd=tmp_path, timeout=300)
    (run_dir,) = (tmp_path / "svc" / "runs").iterdir()
    out = json.loads((run_dir / "itest.json").read_text())
    assert out["purpose"] == "product-test" and out["research_valid"] is False
    found = {c["name"]: c["ok"] for c in out["tests"]["IT-0"]["checks"]}
    assert found["import capserve.deploy"] is True
    if found["vLLM sources found"] is False:          # the Mac has no vLLM: IT-0 fails there, by design
        assert r.returncode == 1
    r = subprocess.run([sys.executable, "-m", "capserve.itest", "IT-1", "--config", str(cfg)], capture_output=True,
                       text=True, env=_module_env(), cwd=tmp_path, timeout=120)
    assert r.returncode == 2 and "flock" in r.stderr     # no GPU lock on the Mac: nothing starts
