"""Test servers for capserve: real uvicorn with the production protocol stack (http="h11", loop="asyncio"; decision
029), bound to 127.0.0.1:0 and run as tasks in the test's own event loop, so the fake backend's asyncio Events can be
awaited directly. Signal handlers are left alone (the test process owns them).

stack() starts a FakeVLLM and a gateway (capserve.server.build_config, the production uvicorn config, on port 0)
in front of it, and waits until the gateway's BackendMonitor is ready (unless the backend is scripted unhealthy).
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

import httpx
import uvicorn

from capserve import server as srv
from capserve.app import Gateway, create_app
from capserve.config import Settings, loopback_client_context
from capserve.profile import CapacityProfile, content_sha256
from fakes.fake_vllm import MODEL, FakeConfig, FakeVLLM

import core_b_plan as bp

STARTUP_LIMIT_S = 10.0


def make_profile(*, max_concurrency=4, max_queue=4, max_prompt_tokens=1920, max_output_tokens=128,
                 classes: list | None = None) -> CapacityProfile:
    """A provisional profile with one class "test", or the given classes [(name, max_prompt_tokens,
    max_concurrency)] (128 output tokens, max_queue = max_concurrency)."""
    pins = bp.load_pins()
    if classes is None:
        classes = [{"name": "test", "max_prompt_tokens": max_prompt_tokens, "max_output_tokens": max_output_tokens,
                    "max_concurrency": max_concurrency, "max_queue": max_queue, "source": "test"}]
    else:
        classes = [{"name": n, "max_prompt_tokens": p, "max_output_tokens": 128, "max_concurrency": c,
                    "max_queue": c, "source": "test"} for n, p, c in classes]
    doc = {"schema_version": 1, "profile_id": "provisional-test", "status": "provisional",
           "provisional_reason": "tests only; not a capacity", "slo": None,
           "server": {"vllm_version": pins["vllm"]["version"], "model_repo": MODEL,
                      "model_revision": pins["core_b"]["revision"], "fingerprint_sha256": None,
                      "num_gpu_blocks": None, "block_size": None},
           "classes": classes, "provenance": None}
    doc["content_sha256"] = content_sha256(doc)
    return CapacityProfile.model_validate(doc)


def gateway_settings(backend_url: str, **over) -> Settings:
    base = {"active_class": "test", "backend_url": backend_url, "host": "127.0.0.1", "port": 0,
            "disconnect_poll_s": 0.01, "monitor_interval_s": 0.01, "monitor_timeout_s": 0.5, "drain_timeout_s": 2,
            "max_queue_wait_s": 30.0}
    return Settings.model_validate({**base, **over})



def _in_loop(server_cls):
    class InLoop(server_cls):
        def capture_signals(self):
            return contextlib.nullcontext()
    return InLoop


@contextlib.asynccontextmanager
async def serve_config(config: uvicorn.Config, server: uvicorn.Server, scheme: str = "http"):
    task = asyncio.create_task(server.serve())
    loop = asyncio.get_running_loop()
    limit = loop.time() + STARTUP_LIMIT_S
    while not server.started:
        if task.done():
            task.result()
            raise RuntimeError("server exited during startup")
        if loop.time() > limit:
            raise RuntimeError("server did not start")
        await asyncio.sleep(0.005)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"{scheme}://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


@contextlib.asynccontextmanager
async def serve(app):
    """Any ASGI app on 127.0.0.1:0 with h11 / asyncio; yields its base URL."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, http="h11", loop="asyncio", ws="none", lifespan="on",
                            log_level="warning", timeout_graceful_shutdown=1)
    async with serve_config(config, _in_loop(uvicorn.Server)(config)) as url:
        yield url


@dataclass
class Stack:
    fake: FakeVLLM
    gateway: Gateway
    url: str

    def client(self, **kw) -> httpx.AsyncClient:
        tls = self.gateway.settings.tls
        if tls is not None:
            kw.setdefault("verify", loopback_client_context(tls))
        return httpx.AsyncClient(base_url=self.url, trust_env=False, timeout=10.0,
                                 limits=httpx.Limits(max_connections=100), **kw)


@contextlib.asynccontextmanager
async def stack(fake_cfg: FakeConfig | None = None, profile: CapacityProfile | None = None, *,
                api_key: bytes | None = None, wait_ready: bool = True, local_tokenizer=None, keyring=None,
                gateway_path_sha256: str | None = None, **settings_over):
    fake = FakeVLLM(fake_cfg)
    async with serve(fake.app) as fake_url:
        settings = gateway_settings(fake_url, **settings_over)
        prof = profile or make_profile()
        gw = create_app(settings, prof, api_key=api_key, local_tokenizer=local_tokenizer, keyring=keyring,
                        gateway_path_sha256=gateway_path_sha256)
        config = srv.build_config(gw, settings)
        scheme = "https" if settings.tls else "http"
        async with serve_config(config, _in_loop(srv.CapserveServer)(config, gw), scheme) as url:
            if wait_ready:
                for _ in range(1000):
                    if gw.monitor.ready:
                        break
                    await asyncio.sleep(0.005)
                assert gw.monitor.ready, gw.monitor.reason
            try:
                yield Stack(fake, gw, url)
            finally:
                fake.release.set()   # never leave a held generation behind at teardown
