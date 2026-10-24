"""Running the gateway under uvicorn (decision 029 "Drain"; review findings 3, 11 and 12).

- The protocol stack is pinned: http="h11", loop="asyncio", ws="none". uvicorn's "auto" picks httptools / uvloop
  when importable (they are in the WSL research venv, not on the Mac), so tests and production would otherwise run
  different HTTP stacks.
- One process: workers=1 explicitly (uvicorn would read WEB_CONCURRENCY) and Server.run directly, never uvicorn.run
  or reload; with more workers every in-process limit would multiply.
- No limit_concurrency (029 had 2 x max_pending + 16 as a backstop; review fix): uvicorn counts open connections,
  idle keep-alive sockets and sockets that never send a request included, and past the limit answers every request
  with its own plain 503 (no Retry-After, no x-request-id, never seen by the guard or the metrics). The guard's
  pending cap is the only request limit.
- Client keep-alive (043 §7.2): timeout_keep_alive = keep_alive_s (75 s), not uvicorn's 5 s, which a client pool
  reusing its connections longer (vllm bench's aiohttp pool: keepalive_timeout 60 s, FIFO) or as long (httpx 5 s)
  races: a request on a connection the gateway is just closing fails before its first token (the race 042 infers for
  D-B's invalid run). Only a pool whose idle limit is below 75 s (aiohttp's default 15 s, httpx's 5 s, vllm bench's
  60 s) is out of the race; a longer or unlimited one (Go's default 90 s, urllib3) can still hit it. Upstream, httpx's
  keepalive_expiry 2 s already stays below vLLM's 5 s (capserve/upstream.py).
- Drain: handle_exit marks the gateway draining on its event loop (call_soon_threadsafe; the signal handler must not
  touch the queue directly), then lets uvicorn shut down: listeners close at once, queued requests get 503 draining,
  running requests finish or are cancelled after timeout_graceful_shutdown = drain_timeout_s (their shielded cleanups
  still release). uvicorn re-raises the captured signal on exit, so exit status -15 is the normal outcome of SIGTERM.
"""

from __future__ import annotations

import uvicorn

from capserve.app import Gateway, create_app
from capserve.config import ConfigError, Settings, check_bind
from capserve.keys import KeyRing
from capserve.profile import CapacityProfile
from capserve.tokens import LocalTokenizer

__all__ = ["CapserveServer", "ConfigError", "build_config", "run"]


def build_config(app: Gateway, settings: Settings) -> uvicorn.Config:
    return uvicorn.Config(app, host=settings.host, port=settings.port, http="h11", loop="asyncio", ws="none",
                          lifespan="on", workers=1, reload=False,
                          timeout_graceful_shutdown=settings.drain_timeout_s, limit_concurrency=None,
                          timeout_keep_alive=settings.keep_alive_s,   # 043 §7.2: not uvicorn's 5 s
                          access_log=False,   # 038 §6: HTTPS when [tls] is set
                          ssl_certfile=settings.tls.certfile if settings.tls else None,
                          ssl_keyfile=settings.tls.keyfile if settings.tls else None)


class CapserveServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, gateway: Gateway):
        super().__init__(config)
        self.gateway = gateway

    def handle_exit(self, sig, frame):
        loop = self.gateway.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self.gateway.set_draining)
        super().handle_exit(sig, frame)


def run(settings: Settings, profile: CapacityProfile, *, api_key: bytes | None,
        local_tokenizer: LocalTokenizer | None = None, keyring: KeyRing | None = None) -> None:
    """Serve until SIGINT / SIGTERM (the caller has already loaded an allowed profile and, for tokenizer "local",
    the tokenizer)."""
    check_bind(settings.host, api_key or keyring)
    gateway = create_app(settings, profile, api_key=api_key, local_tokenizer=local_tokenizer, keyring=keyring)
    config = build_config(gateway, settings)
    CapserveServer(config, gateway).run()
