"""Tests for capserve/upstream.py: the data-plane httpx client configuration and BackendMonitor (own client,
3 consecutive failures before not-ready, /v1/models only at startup and on recovery; decision 029)."""

import asyncio

import httpx
import pytest

from capserve import upstream as up

M = "served/model"


def test_data_client_overrides_every_httpx_default(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:1")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:1")
    c = up.data_client("http://127.0.0.1:8100", max_concurrency=4)
    pool = c._transport._pool
    assert c.trust_env is False and c._mounts == {}          # no proxy, even for 127.0.0.1
    assert (pool._max_connections, pool._max_keepalive_connections) == (8, 8)
    assert pool._keepalive_expiry == 2.0                      # below uvicorn's 5 s keep-alive
    assert c.timeout == httpx.Timeout(connect=2.0, read=None, write=10.0, pool=2.0)   # deadline bounds reads
    assert str(c.base_url) == "http://127.0.0.1:8100"


def test_monitor_client_is_small_and_short(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:1")
    m = up.BackendMonitor("http://127.0.0.1:8100", M, timeout_s=2.0)
    pool = m.client._transport._pool
    assert m.client.trust_env is False and m.client._mounts == {}
    assert (pool._max_connections, pool._max_keepalive_connections) == (2, 2)
    assert m.client.timeout == httpx.Timeout(2.0)


class Backend:
    """A scripted vLLM: /health answers from `health` (status or exception), /v1/models lists `models`."""

    def __init__(self):
        self.health = [200]
        self.models = [M]
        self.calls = []

    def handler(self, request):
        self.calls.append(request.url.path)
        if request.url.path == "/health":
            outcome = self.health.pop(0) if len(self.health) > 1 else self.health[0]
            if isinstance(outcome, Exception):
                raise outcome
            return httpx.Response(outcome)
        return httpx.Response(200, json={"object": "list", "data": [{"id": x} for x in self.models]})


def monitor(backend, downs=None, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend.handler), base_url="http://vllm")
    on_down = (lambda: downs.append(1)) if downs is not None else None
    return up.BackendMonitor("http://vllm", M, client=client, on_down=on_down, **kw)


@pytest.mark.anyio
async def test_ready_only_after_health_and_the_served_model():
    b = Backend()
    m = monitor(b)
    assert (m.ready, m.reason) == (False, "starting")
    await m.probe_once()
    assert (m.ready, m.reason) == (True, None) and b.calls == ["/health", "/v1/models"]


@pytest.mark.anyio
async def test_wrong_model_is_not_ready():
    b = Backend()
    b.models = ["other"]
    m = monitor(b)
    await m.probe_once()
    assert m.ready is False and "model" in m.reason


@pytest.mark.anyio
async def test_three_consecutive_failures_before_not_ready_and_models_only_on_recovery():
    b = Backend()
    downs = []
    m = monitor(b, downs, fail_threshold=3)
    await m.probe_once()
    b.health = [503, httpx.ReadTimeout("slow"), 200, 503, httpx.ConnectError("refused"), 500, 200]
    for _ in range(5):
        await m.probe_once()
        assert m.ready and downs == []        # a success resets the count; two failures are not enough
    await m.probe_once()
    assert (m.ready, downs) == (False, [1]) and "health" in m.reason
    await m.probe_once()
    assert m.ready is True
    assert b.calls.count("/v1/models") == 2   # startup and recovery only
    await m.probe_once()
    assert b.calls.count("/v1/models") == 2


@pytest.mark.anyio
async def test_on_down_fires_once_per_outage():
    b = Backend()
    downs = []
    m = monitor(b, downs, fail_threshold=1)
    await m.probe_once()
    b.health = [503]
    for _ in range(3):
        await m.probe_once()
    assert downs == [1]


@pytest.mark.anyio
async def test_run_probes_until_cancelled():
    b = Backend()
    m = monitor(b, interval_s=0.001)
    task = asyncio.create_task(m.run())
    for _ in range(1000):
        if b.calls.count("/health") >= 3:
            break
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert m.ready and b.calls.count("/health") >= 3
    await m.aclose()
