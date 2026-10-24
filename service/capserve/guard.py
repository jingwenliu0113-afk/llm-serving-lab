"""The outermost pure-ASGI middleware (decision 029 steps 1 and 10; review findings 1, 7 and 13).

Per HTTP request, before anything else runs:
- a fresh x-request-id (never the client's), added with x-capserve-profile: <id>; status=<verified|provisional> to
  every response, including the guard's own 429 and errors produced inside the app;
- the pending cap: pending counts requests from arrival until the response finished; at max_pending
  (max_concurrency + max_queue + pre_admission_slots) a request gets 429 queue_full before its body is read.
  /healthz, /readyz and /metrics are exempt;
- a RequestRecord on the scope ("capserve.record") that the endpoint fills in; at the end the guard hands it to
  Metrics.finish and on_finish (the Retry-After policy's observe, 038 §3) and writes one JSON access-log line (no
  prompt or completion text, ever). The guard's own 429 takes its Retry-After from retry_after_s("queue_full").
  044: the line also carries the client's x-request-id as client_request_id (one [A-Za-z0-9._:-]{1,128} value, else
  null; never echoed: the response keeps the gateway's id), so a client can join its own requests to the log, and the
  gateway's monotonic times t_arrival_s (arrival), t_sent_s (upstream send) and t_released_s (the admission permit's
  release, recorded before the slot moves on), with prefill_wait_s, admission_class and admission_units once set.
Cleanups (on_cleanup) are registered on the scope by whoever acquires something (the admission permit, the upstream
response) right after acquiring it; the guard runs them in a shielded finally, LIFO, each bounded by
CLEANUP_TIMEOUT_S, whether the endpoint returned, raised, or was cancelled, and whether or not the response object
was ever called. Release functions are idempotent, so an endpoint may also release early.

Pure ASGI on purpose: BaseHTTPMiddleware would reroute receive / send through memory streams, which changes
disconnect detection and backpressure.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

import anyio

from capserve.openai_api import openai_error

EXEMPT_PATHS = ("/healthz", "/readyz", "/metrics")
COMPLETION_ROUTES = {"/v1/chat/completions": "chat", "/v1/completions": "completions"}
CLEANUP_TIMEOUT_S = 5.0
RECORD_KEY = "capserve.record"
CLEANUPS_KEY = "capserve.cleanups"
CLIENT_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")   # 044: what the access log keeps of a client's id

access_log = logging.getLogger("capserve.access")
log = logging.getLogger("capserve")


@dataclass
class RequestRecord:
    request_id: str
    method: str
    path: str
    arrival: float
    route: str | None = None            # "chat" / "completions" on the completion routes
    stream: bool | None = None          # None until the body is parsed
    status: int | None = None
    outcome: str | None = None          # ok, client_error, rejected, upstream_error, deadline, cancelled, server_error
    reason: str | None = None           # rejection reason
    error_code: str | None = None       # client error code
    upstream_error: str | None = None   # connect, timeout, status_5xx, stream_broken, stream_error, protocol
    disconnect_phase: str | None = None  # queued, waiting_response, streaming
    queue_wait_s: float | None = None
    upstream_sent: float | None = None
    ttft_s: float | None = None
    upstream_ttft_s: float | None = None
    duration_s: float | None = None
    key: str | None = None              # the API key's name (038 §5 keys file); never the key
    # 044: logged so a client can join the log (client_request_id) and the in-vLLM interval [t_sent_s, t_released_s)
    # is exact; arrival and upstream_sent are logged as t_arrival_s and t_sent_s
    client_request_id: str | None = None
    t_released_s: float | None = None   # the admission permit's release (clock), before the slot moves on
    prefill_wait_s: float | None = None  # prefill pacing: at the head with a free slot, waiting for tokens
    admission_class: str | None = None  # weighted-v1: the class the prompt was admitted as, and its units
    admission_units: int | None = None


def client_request_id(headers) -> str | None:
    """044: the client's x-request-id when it sent exactly one, matching CLIENT_REQUEST_ID; else None."""
    values = [v for n, v in headers if n == b"x-request-id"]
    if len(values) != 1:
        return None
    try:
        value = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    return value if CLIENT_REQUEST_ID.fullmatch(value) else None


def on_cleanup(scope, fn: Callable[[], Awaitable[None] | None]) -> None:
    """Run fn when the request ends, however it ends (LIFO, shielded, bounded)."""
    scope.setdefault(CLEANUPS_KEY, []).append(fn)


async def run_cleanups(scope, *, timeout_s: float = CLEANUP_TIMEOUT_S) -> None:
    cleanups = scope.get(CLEANUPS_KEY) or []
    scope[CLEANUPS_KEY] = []
    with anyio.CancelScope(shield=True):
        for fn in reversed(cleanups):
            try:
                with anyio.move_on_after(timeout_s):
                    result = fn()
                    if inspect.isawaitable(result):
                        await result
            except Exception:
                log.exception("cleanup failed")


class Guard:
    def __init__(self, app, *, max_pending: int, profile_header: str, retry_after_s: Callable[[str], float], metrics,
                 on_finish: Callable[[RequestRecord], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.app = app
        self.max_pending = max_pending
        self.profile_header = profile_header.encode()
        self.retry_after_s = retry_after_s
        self.metrics = metrics
        self.on_finish = on_finish
        self.clock = clock
        self.pending = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        rec = RequestRecord(request_id=uuid.uuid4().hex, method=scope["method"], path=path, arrival=self.clock(),
                            route=COMPLETION_ROUTES.get(path), client_request_id=client_request_id(scope["headers"]))
        scope[RECORD_KEY] = rec
        scope[CLEANUPS_KEY] = []
        extra = [(b"x-request-id", rec.request_id.encode()), (b"x-capserve-profile", self.profile_header)]

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                rec.status = message["status"]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        limited = path not in EXEMPT_PATHS
        if limited and self.pending >= self.max_pending:
            rec.outcome, rec.reason = "rejected", "queue_full"
            try:
                await openai_error(429, "rate_limit_error", "queue_full", "too many pending requests; retry later",
                                   retry_after_s=self.retry_after_s("queue_full"))(scope, receive, send_with_headers)
            finally:
                self._finish(rec)
            return
        if limited:
            self.pending += 1
        try:
            await self.app(scope, receive, send_with_headers)
        except asyncio.CancelledError:
            rec.outcome = rec.outcome or "cancelled"
            raise
        except Exception:
            rec.outcome = "server_error"
            raise
        finally:
            await run_cleanups(scope)
            if limited:
                self.pending -= 1
            self._finish(rec)

    def _finish(self, rec: RequestRecord) -> None:
        rec.duration_s = max(self.clock() - rec.arrival, 0.0)
        try:
            self.metrics.finish(rec)
        except Exception:
            log.exception("metrics failed")
        if self.on_finish is not None:
            try:
                self.on_finish(rec)
            except Exception:
                log.exception("on_finish failed")
        entry = {k: v for k, v in asdict(rec).items() if v is not None and k not in ("arrival", "upstream_sent")}
        entry["t_arrival_s"] = rec.arrival
        if rec.upstream_sent is not None:
            entry["t_sent_s"] = rec.upstream_sent
        access_log.info(json.dumps({"ts": round(time.time(), 3), **entry}, sort_keys=True))
