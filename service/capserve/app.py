"""The gateway application (decision 029 "Request 流程"): plain Starlette routes behind the pure-ASGI Guard; no
FastAPI (no /docs or /openapi.json, no {"detail": ...} errors: 404 / 405 / 500 are OpenAI-shaped).

Routes: POST /v1/chat/completions, POST /v1/completions, GET /v1/models (the served id only), /healthz, /readyz,
/metrics. There is no catch-all passthrough, so vLLM's /tokenize and admin endpoints are never exposed.

The active class must be servable (profile.serving_class): an unavailable class of a verified profile (Stage C
C_PROD UNSTABLE / NT, decision 030 8.7), a verified class of more than 128 output tokens, and on a T profile any class
but M8K-T with H-D2a supported (decision 043) never build a gateway (ProfileError), whatever flag is given. A verified 0
(C_PROD INFEASIBLE: refuse the class) builds one with a closed admission controller: every completion request gets
503 class_not_serviceable right after auth (no Retry-After, x-should-retry false: retrying cannot help), /readyz is
503 with not_ready_reason class_not_serviceable, and /v1/models, /healthz and /metrics work as usual.

A completion request, after the guard (request id, pending cap):
 2. auth: optional bearer key, compared as bytes with hmac.compare_digest, or a keys file's key by sha256 (401);
 3. gate: a verified 0 class -> 503 class_not_serviceable; draining or backend not ready -> 503 with Retry-After;
    all before the body is read and never queued; then the key's quota (038 §5): 429 quota_exceeded;
 4. body: at most max_body_bytes (413) within body_read_timeout_s (408);
 5. validation (openai_api): allowlist, served model, exactly one prompt, max_tokens injected / bounded;
 6. byte fast path (tokens): whether a count is needed at all; with the local tokenizer (038 §4) the count happens
    here, before admission and without vLLM: too long -> 400 context_length_exceeded, a template error -> 400;
 7. admission: the gate is checked again (the backend may have gone down while the body arrived), then a FIFO permit
    raced against a client-disconnect watcher; the release is registered on the scope at once;
 8. /tokenize only if needed, under the permit; too long -> 400 context_length_exceeded and release; a 4xx from
    /tokenize passes through like a generation error (502 only for 5xx or a malformed answer);
 9. upstream (Authorization never forwarded; body re-serialized), all under the request deadline and the watcher:
    non-stream passes through (504 at the deadline, 502 when vLLM is unreachable); stream -> ProxyStream;
10. cleanup: the guard's shielded scope cleanups (LIFO: the upstream response is closed, then the permit released),
    metrics and the access log.
A disconnected client gets no response (499 is recorded), and its upstream request is cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import time
from typing import Callable

import anyio
import httpx
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from capserve.admission import NOT_SERVICEABLE, STATUS, AdmissionController, Rejection
from capserve.config import ConfigError, Settings
from capserve.guard import RECORD_KEY, Guard, on_cleanup
from capserve.keys import KeyRing, QuotaDenied
from capserve.metrics import Metrics
from capserve.openai_api import ApiError, openai_error, parse_request
from capserve.profile import CapacityProfile
from capserve.proxy import ClientDisconnected, ProxyStream, cancel_on_disconnect, passthrough, read_capped
from capserve.retry import RetryAfterPolicy
from capserve.tokens import LocalTokenizer, PromptBudget, TemplateRejected, TokenizeError, TokenizerLoadError
from capserve.upstream import BackendMonitor, data_client

UPSTREAM_PATHS = {"chat": "/v1/chat/completions", "completions": "/v1/completions"}
CLIENT_CLOSED = 499   # recorded only: the client is gone and never sees it
REJECTION_MESSAGES = {
    "queue_full": "the server is at capacity and its queue is full; retry later",
    "queue_timeout": "the request waited too long for capacity; retry later",
    "backend_unavailable": "the model server is not available; retry later",
    "draining": "the server is shutting down; retry later",
    "class_not_serviceable": "this deployment's workload class has a verified capacity of 0 under its SLO (Stage C "
                             "INFEASIBLE): every request is refused; do not retry",
}


class _TooLarge(Exception):
    pass


class _TooLong(Exception):
    def __init__(self, n: int):
        self.n = n


class Gateway:
    """The ASGI app plus its state (admission, backend monitor, clients, metrics)."""

    def __init__(self, settings: Settings, profile: CapacityProfile, *, api_key: bytes | None = None,
                 clock: Callable[[], float] = time.monotonic, local_tokenizer: LocalTokenizer | None = None,
                 keyring: KeyRing | None = None):
        self.settings = settings
        self.profile = profile
        self.cls = profile.serving_class(settings.active_class)   # ProfileError: 030 8.7, 043
        self.served_model = profile.server.model_repo
        self.api_key = api_key
        self.keyring = keyring
        self.clock = clock
        self.draining = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ctrl = AdmissionController(max_concurrency=self.cls.max_concurrency, max_queue=self.cls.max_queue,
                                        max_queue_wait_s=settings.max_queue_wait_s,
                                        retry_after_s=settings.retry_after_s.fixed_values(), clock=clock)
        self.retry = RetryAfterPolicy(settings.retry_after_s, max_concurrency=self.cls.max_concurrency,
                                      queued=lambda: self.ctrl.queued)
        self.client = data_client(settings.backend_url, max_concurrency=self.cls.max_concurrency)
        self.monitor = BackendMonitor(settings.backend_url, self.served_model, interval_s=settings.monitor_interval_s,
                                      timeout_s=settings.monitor_timeout_s,
                                      fail_threshold=settings.monitor_fail_threshold, on_down=self._backend_down)
        local = None
        if settings.tokenizer == "local":
            try:
                local = local_tokenizer or LocalTokenizer.load(settings.tokenizer_path)
            except TokenizerLoadError as exc:
                raise ConfigError(str(exc)) from exc
        self.budget = PromptBudget(self.client, **settings.tokens.model_dump(), local=local)
        self.metrics = Metrics(profile, self.cls, in_flight=lambda: self.ctrl.in_flight,
                               queued=lambda: self.ctrl.queued, pending=lambda: self.guard.pending,
                               backend_ready=lambda: self.monitor.ready)
        routes = [Route("/v1/chat/completions", self.chat, methods=["POST"]),
                  Route("/v1/completions", self.completions, methods=["POST"]),
                  Route("/v1/models", self.models, methods=["GET"]),
                  Route("/healthz", self.healthz, methods=["GET"]),
                  Route("/readyz", self.readyz, methods=["GET"]),
                  Route("/metrics", self.metrics_endpoint, methods=["GET"])]
        handlers = {404: self._not_found, 405: self._method_not_allowed, 500: self._server_error}
        self.starlette = Starlette(routes=routes, exception_handlers=handlers, lifespan=self._lifespan)
        self.guard = Guard(self.starlette, max_pending=settings.max_pending(self.cls),
                           profile_header=f"{profile.profile_id}; status={profile.status}",
                           retry_after_s=self.retry.seconds, metrics=self.metrics, on_finish=self.retry.observe,
                           clock=clock)

    async def __call__(self, scope, receive, send):
        await self.guard(scope, receive, send)

    # ------------------------------------------------------------ lifecycle

    @contextlib.asynccontextmanager
    async def _lifespan(self, app):
        self.loop = asyncio.get_running_loop()
        monitor = self.loop.create_task(self.monitor.run())
        try:
            yield
        finally:
            monitor.cancel()
            await asyncio.wait([monitor])
            await self.monitor.aclose()
            await self.client.aclose()

    def _backend_down(self) -> None:
        self.ctrl.reject_waiters("backend_unavailable")

    def set_draining(self) -> None:
        self.draining = True
        self.ctrl.set_draining()

    def _unavailable(self) -> str | None:
        """The 503 reason, if any. Checked at the gate and again right before admission: the backend can go down
        (and on_down reject the queue) while a body is still arriving, and nothing awaits between this check and
        acquire, so a request is never queued or sent upstream after the backend was declared down."""
        if not self.cls.serviceable:
            return "class_not_serviceable"
        if self.draining:
            return "draining"
        if not self.monitor.ready:
            return "backend_unavailable"
        return None

    # ------------------------------------------------------------ responses

    def _reject(self, rec, reason: str) -> Response:
        rec.outcome, rec.reason = "rejected", reason
        type_ = "rate_limit_error" if STATUS[reason] == 429 else "server_error"
        if reason == NOT_SERVICEABLE:   # permanent: no Retry-After, and OpenAI clients must not retry
            resp = openai_error(STATUS[reason], type_, reason, REJECTION_MESSAGES[reason])
            resp.headers["x-should-retry"] = "false"
            return resp
        return openai_error(STATUS[reason], type_, reason, REJECTION_MESSAGES[reason],
                            retry_after_s=self.retry.seconds(reason))

    def _client_error(self, rec, status: int, code: str, message: str, *, param: str | None = None) -> Response:
        rec.outcome, rec.error_code = "client_error", code
        return openai_error(status, "invalid_request_error", code, message, param=param)

    def _too_long(self, rec, route: str, n: int) -> Response:
        return self._client_error(rec, 400, "context_length_exceeded",
                                  f"the prompt has {n} tokens; this server accepts at most {self.cls.max_prompt_tokens}",
                                  param="messages" if route == "chat" else "prompt")

    def _upstream_error(self, rec, status: int, kind: str, code: str, message: str,
                        outcome: str = "upstream_error") -> Response:
        rec.outcome, rec.upstream_error = outcome, kind
        return openai_error(status, "server_error", code, message)

    def _gone(self, rec, phase: str | None) -> Response:
        rec.outcome, rec.disconnect_phase = "cancelled", phase
        return Response(status_code=CLIENT_CLOSED)

    @staticmethod
    def _classify_status(rec, status: int) -> None:
        if status == 200:
            rec.outcome = "ok"
        elif status >= 500:
            rec.outcome, rec.upstream_error = "upstream_error", "status_5xx"
        else:
            rec.outcome, rec.error_code = "client_error", f"upstream_{status}"

    async def _not_found(self, request, exc) -> Response:
        return openai_error(404, "invalid_request_error", "not_found", f"no route {request.url.path}")

    async def _method_not_allowed(self, request, exc) -> Response:
        resp = openai_error(405, "invalid_request_error", "method_not_allowed",
                            f"{request.method} is not allowed on {request.url.path}")
        for k, v in (getattr(exc, "headers", None) or {}).items():
            resp.headers[k] = v
        return resp

    async def _server_error(self, request, exc) -> Response:
        return openai_error(500, "server_error", "internal_error", "internal gateway error")

    # ------------------------------------------------------------ simple routes

    def _identify(self, scope) -> str | None:
        """The caller: "" without keys or with the single key, the key's name with a keys file, None when refused."""
        if self.api_key is None and self.keyring is None:
            return ""
        value = next((v for n, v in scope["headers"] if n == b"authorization"), None)
        if value is None:
            return None
        if self.api_key is not None:
            return "" if hmac.compare_digest(value, b"Bearer " + self.api_key) else None
        return self.keyring.match(value[len(b"Bearer "):]) if value.startswith(b"Bearer ") else None

    def _quota_denied(self, rec, exc: QuotaDenied) -> Response:
        rec.outcome, rec.reason = "rejected", "quota_exceeded"
        retry = exc.retry_after_s if exc.retry_after_s is not None else self.retry.seconds("queue_full")
        return openai_error(429, "rate_limit_error", "quota_exceeded",
                            f"this API key's {exc.limit} quota is used up; retry later", retry_after_s=retry)

    def _unauthorized(self, rec) -> Response:
        return self._client_error(rec, 401, "invalid_api_key", "missing or invalid API key")

    async def healthz(self, request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    def readiness(self) -> dict:
        """The /readyz body. Besides the limits it discloses (043 §5) the server variant (B0 / T) and, from the
        active class's evidence, its flags and what its limit was never measured under (not_covered: null for a §8
        class such as c1b's, whose conditions are the README's; both null for a provisional class)."""
        p, c = self.profile, self.cls
        ev = c.evidence
        return {"ready": c.serviceable and self.monitor.ready and not self.draining,
                "not_ready_reason": self._unavailable(),
                "profile": {"id": p.profile_id, "status": p.status, "provisional": not p.verified},
                "active_class": c.name, "class_status": c.status, "bound": c.bound, "serviceable": c.serviceable,
                "server_variant": p.server_variant,
                "flags": list(ev.flags) if ev is not None else None,
                "not_covered": list(ev.not_covered) if ev is not None and ev.not_covered is not None else None,
                "backend": {"ready": self.monitor.ready, "reason": self.monitor.reason},
                "draining": self.draining, "in_flight": self.ctrl.in_flight, "queued": self.ctrl.queued,
                "pending": self.guard.pending,
                "limits": {"max_concurrency": c.max_concurrency, "max_queue": c.max_queue,
                           "max_pending": self.settings.max_pending(c), "max_prompt_tokens": c.max_prompt_tokens,
                           "max_output_tokens": c.max_output_tokens}}

    async def readyz(self, request: Request) -> Response:
        body = self.readiness()
        return JSONResponse(body, status_code=200 if body["ready"] else 503)

    async def metrics_endpoint(self, request: Request) -> Response:
        return Response(self.metrics.exposition(), media_type=CONTENT_TYPE_LATEST)

    async def models(self, request: Request) -> Response:
        if self._identify(request.scope) is None:
            return self._unauthorized(request.scope[RECORD_KEY])
        return JSONResponse({"object": "list",
                             "data": [{"id": self.served_model, "object": "model", "owned_by": "capserve"}]})

    # ------------------------------------------------------------ completions

    async def chat(self, request: Request) -> Response:
        return await self._complete(request, "chat")

    async def completions(self, request: Request) -> Response:
        return await self._complete(request, "completions")

    async def _read_body(self, request: Request) -> bytes:
        limit = self.settings.max_body_bytes
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise _TooLarge
        body = bytearray()
        with anyio.fail_after(self.settings.body_read_timeout_s):
            async for chunk in request.stream():
                body += chunk
                if len(body) > limit:
                    raise _TooLarge
        return bytes(body)

    def _watch(self, request: Request):
        return cancel_on_disconnect(request.is_disconnected, self.settings.disconnect_poll_s)

    async def _complete(self, request: Request, route: str) -> Response:
        scope = request.scope
        rec = scope[RECORD_KEY]
        cls = self.cls
        key = self._identify(scope)
        if key is None:
            return self._unauthorized(rec)
        rec.key = key or None
        if reason := self._unavailable():
            return self._reject(rec, reason)
        if self.keyring is not None:
            try:
                on_cleanup(scope, self.keyring.admit(key, self.clock()))
            except QuotaDenied as exc:
                return self._quota_denied(rec, exc)
        try:
            raw = await self._read_body(request)
        except _TooLarge:
            return self._client_error(rec, 413, "request_too_large",
                                      f"the request body exceeds {self.settings.max_body_bytes} bytes")
        except TimeoutError:
            return self._client_error(rec, 408, "request_timeout", "the request body was not received in time")
        except ClientDisconnect:
            return self._gone(rec, None)
        try:
            fwd = parse_request(route, raw, served_model=self.served_model, max_output_tokens=cls.max_output_tokens)
        except ApiError as exc:
            rec.outcome, rec.error_code = "client_error", exc.code
            return exc.response()
        rec.stream = fwd.stream
        needs_count = not self.budget.fast_fits(route, fwd.body, cls.max_prompt_tokens)
        deadline = rec.arrival + self.settings.request_deadline_s
        if needs_count and self.budget.local is not None:   # 038 §4: no vLLM call, so no permit
            try:
                n = await self.budget.count(route, fwd.body)
            except TemplateRejected:
                return self._client_error(rec, 400, "invalid_value", "the chat template rejected these messages",
                                          param="messages")
            if n > cls.max_prompt_tokens:
                return self._too_long(rec, route, n)
            needs_count = False

        if reason := self._unavailable():   # again: the backend may have gone down while the body arrived
            return self._reject(rec, reason)
        try:
            async with self._watch(request):
                permit = await self.ctrl.acquire(deadline=deadline)
                on_cleanup(scope, permit.release)   # registered before anything else can interrupt
        except Rejection as exc:
            return self._reject(rec, exc.reason)
        except ClientDisconnected:
            return self._gone(rec, "queued")
        rec.queue_wait_s = permit.queued_s

        try:
            with anyio.fail_after(max(deadline - self.clock(), 0.0)):
                async with self._watch(request):
                    if needs_count:
                        n = await self.budget.count(route, fwd.body)
                        if n > cls.max_prompt_tokens:
                            raise _TooLong(n)
                    rec.upstream_sent = self.clock()
                    req = self.client.build_request("POST", UPSTREAM_PATHS[route], json=fwd.body)
                    upstream = await self.client.send(req, stream=fwd.stream)
                    on_cleanup(scope, upstream.aclose)
        except _TooLong as exc:
            permit.release()
            return self._too_long(rec, route, exc.n)
        except TimeoutError:
            permit.release()
            return self._upstream_error(rec, 504, "timeout", "deadline_exceeded", "the request deadline passed",
                                        outcome="deadline")
        except ClientDisconnected:
            # possibly noticed only once the upstream answered: no early release, so the scope cleanups close that
            # response first (LIFO) and only then release the permit
            return self._gone(rec, "waiting_response")
        except (httpx.ConnectError, httpx.ConnectTimeout):
            permit.release()
            return self._upstream_error(rec, 502, "connect", "upstream_unavailable", "the model server is unreachable")
        except TokenizeError as exc:
            permit.release()
            if 400 <= exc.status < 500:   # vLLM refused the prompt (e.g. its chat template raised): as for generation
                self._classify_status(rec, exc.status)
                return passthrough(exc.response)
            kind = "status_5xx" if exc.status >= 500 else "protocol"
            return self._upstream_error(rec, 502, kind, "upstream_unavailable", "the model server failed to tokenize")
        except httpx.TransportError:
            permit.release()
            return self._upstream_error(rec, 502, "protocol", "upstream_unavailable", "the model server failed")

        if not fwd.stream:
            permit.release()   # vLLM has finished this request
            self._classify_status(rec, upstream.status_code)
            return passthrough(upstream)
        if upstream.status_code != 200:
            content = b""
            try:
                with anyio.move_on_after(max(deadline - self.clock(), 0.0)):
                    content = await read_capped(upstream)
            except httpx.TransportError:
                pass
            finally:
                await upstream.aclose()
                permit.release()
            self._classify_status(rec, upstream.status_code)
            return passthrough(upstream, content)
        return ProxyStream(upstream, rec=rec, deadline=deadline, clock=self.clock, cleanup=permit.release)


def create_app(settings: Settings, profile: CapacityProfile, *, api_key: bytes | None = None,
               clock: Callable[[], float] = time.monotonic, local_tokenizer: LocalTokenizer | None = None,
               keyring: KeyRing | None = None) -> Gateway:
    return Gateway(settings, profile, api_key=api_key, clock=clock, local_tokenizer=local_tokenizer, keyring=keyring)
