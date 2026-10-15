"""Clients for the vLLM server (decision 029; review findings 4 and 9).

data_client: every httpx default that would break a proxy is overridden.
- trust_env=False: no HTTP(S)_PROXY / ALL_PROXY, even for 127.0.0.1.
- max_connections = max_concurrency + 4: a permit covers each request, so the pool is never the bottleneck
  (httpx's 100 would also be below vLLM's --max-num-seqs 128).
- keepalive_expiry 2 s, below uvicorn's 5 s keep-alive, so the pool never reuses a connection the server has just
  closed (a spurious "Server disconnected" 502); vLLM's actual keep-alive is checked at RTX IT-0.
- Timeout(connect=2, read=None, write=10, pool=2): reads are bounded by the request deadline (non-stream) or the
  per-read deadline (stream), never by httpx's 5 s default.

BackendMonitor: probes /health with its own 2-connection client and a 2 s timeout, so probes never wait behind
data-plane traffic. It turns not-ready only after fail_threshold (3) consecutive failures, so one slow /health under
load does not reject the whole queue; then on_down runs once (the app rejects the queue with 503). /v1/models must
list the served id; it is checked at startup and on each recovery only.
"""

from __future__ import annotations

import asyncio

import httpx

CONNECT_TIMEOUT_S = 2.0
WRITE_TIMEOUT_S = 10.0
POOL_TIMEOUT_S = 2.0
KEEPALIVE_EXPIRY_S = 2.0
SPARE_CONNECTIONS = 4
MONITOR_CONNECTIONS = 2


def data_client(base_url: str, *, max_concurrency: int) -> httpx.AsyncClient:
    n = max_concurrency + SPARE_CONNECTIONS
    return httpx.AsyncClient(
        base_url=base_url, trust_env=False,
        limits=httpx.Limits(max_connections=n, max_keepalive_connections=n, keepalive_expiry=KEEPALIVE_EXPIRY_S),
        timeout=httpx.Timeout(connect=CONNECT_TIMEOUT_S, read=None, write=WRITE_TIMEOUT_S, pool=POOL_TIMEOUT_S))


class BackendMonitor:
    def __init__(self, base_url: str, served_model: str, *, interval_s: float = 2.0, timeout_s: float = 2.0,
                 fail_threshold: int = 3, on_down=None, client: httpx.AsyncClient | None = None):
        self.served_model = served_model
        self.interval_s = interval_s
        self.fail_threshold = fail_threshold
        self.on_down = on_down
        self.client = client or httpx.AsyncClient(
            base_url=base_url, trust_env=False, timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(max_connections=MONITOR_CONNECTIONS, max_keepalive_connections=MONITOR_CONNECTIONS))
        self.ready = False
        self.reason: str | None = "starting"
        self._failures = 0

    async def _healthy(self) -> bool:
        try:
            return (await self.client.get("/health")).status_code == 200
        except httpx.HTTPError:
            return False

    async def _serves_model(self) -> bool:
        try:
            resp = await self.client.get("/v1/models")
            ids = [m.get("id") for m in resp.json().get("data", [])] if resp.status_code == 200 else []
        except (httpx.HTTPError, ValueError, AttributeError):
            return False
        return self.served_model in ids

    async def probe_once(self) -> None:
        if await self._healthy():
            self._failures = 0
            if not self.ready:   # startup or recovery: the server must serve our model
                if await self._serves_model():
                    self.ready, self.reason = True, None
                else:
                    self.reason = f"backend does not serve the model {self.served_model!r}"
            return
        self._failures += 1
        if self.ready and self._failures >= self.fail_threshold:
            self.ready, self.reason = False, f"health failed {self._failures} times in a row"
            if self.on_down is not None:
                self.on_down()

    async def run(self) -> None:
        while True:
            await self.probe_once()
            await asyncio.sleep(self.interval_s)

    async def aclose(self) -> None:
        await self.client.aclose()
