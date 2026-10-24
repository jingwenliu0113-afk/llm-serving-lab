"""Forwarding vLLM responses (decision 029 "非串流" / "串流"; review findings 7, 8 and 19).

- Non-stream: the upstream status and raw bytes pass through with only their content-type (no hop-by-hop or other
  upstream headers). A non-200 body passes through when it is OpenAI-shaped ({"error": {...}}) and is wrapped
  otherwise, without echoing it.
- cancel_on_disconnect: while the gateway waits (queue, /tokenize, upstream), a watcher polls the client every
  disconnect_poll_s and cancels the block when the client is gone, which cancels the httpx request and closes its
  connection, so vLLM aborts the request (checked at RTX IT-3).
- Stream (ProxyStream): only complete SSE events are forwarded (SSEFramer), so an error can never be appended to half
  an event. The request deadline is applied around each upstream read and each response write, never across a yield
  (an anyio cancel scope spanning a yield breaks when the generator is closed from another task). When the upstream
  breaks or the deadline passes, one error event and `data: [DONE]` end the stream; cancellation is never caught. A
  slow client pauses the response writes and with them the upstream reads (backpressure); a client that stops
  reading is cut at the deadline (outcome deadline, not an upstream error). An in-band error event from vLLM is
  forwarded as is and recorded as upstream_error / stream_error. The permit is held until the last byte, a disconnect
  or the deadline. Cleanup runs in a shielded, bounded finally: the body generator, then the upstream response, then
  the permit release.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import Awaitable, Callable

import anyio
import httpx
from starlette.responses import Response, StreamingResponse

from capserve.openai_api import error_body, openai_error

SSE_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}
CLEANUP_TIMEOUT_S = 5.0
WATCHER_STOP_TIMEOUT_S = 1.0
ERROR_BODY_LIMIT = 64 * 1024
_EVENT_END = re.compile(rb"\r\n\r\n|\n\n")


class ClientDisconnected(Exception):
    pass


class SSEFramer:
    """Splits an SSE byte stream into complete events (each with its blank-line terminator)."""

    def __init__(self):
        self.pending = b""

    def feed(self, data: bytes) -> list:
        self.pending += data
        events = []
        while (m := _EVENT_END.search(self.pending)) is not None:
            events.append(self.pending[:m.end()])
            self.pending = self.pending[m.end():]
        return events


def has_content(event: bytes) -> bool:
    """Whether an event carries generated text (chat delta.content or completion text): the content checks of itest
    and loadtest (never the gateway's first-token mark: is_token_event)."""
    for line in event.splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return False
        try:
            choices = json.loads(payload).get("choices") or []
            choice = choices[0]
            return bool((choice.get("delta") or {}).get("content") or choice.get("text"))
        except (ValueError, AttributeError, IndexError, TypeError):
            return False
    return False


def is_token_event(event: bytes) -> bool:
    """Whether an event is a token event, the gateway's first-token mark (ttft_s, upstream_ttft_s; 044 實作補記六): its
    first data line is JSON with a non-empty choices list: vLLM 0.30 bench's TTFT rule (endpoint_request_func:
    `if choices := data.get("choices")`, whatever the text, which may be empty for a special or partial UTF-8 token).
    [DONE], the usage-only chunk ("choices": []), an in-band error and a comment are not."""
    for line in event.splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return False
        try:
            doc = json.loads(payload)
        except ValueError:
            return False
        choices = doc.get("choices") if isinstance(doc, dict) else None
        return isinstance(choices, list) and len(choices) > 0
    return False


def is_error(event: bytes) -> bool:
    """Whether an event is an in-band error: vLLM reports a failure after the stream started (engine error, abort)
    as `data: {"error": {...}}` on the HTTP 200 stream, then `data: [DONE]`."""
    if b'"error"' not in event:   # skip the JSON parse for ordinary events
        return False
    for line in event.splitlines():
        if line.startswith(b"data:"):
            return _openai_shaped(line[5:].strip())
    return False


def sse_error(code: str, message: str) -> list:
    """The two events that end a broken stream: an OpenAI-shaped error, then [DONE]."""
    body = error_body("server_error", code, message)
    return [b"data: " + json.dumps(body).encode() + b"\n\n", b"data: [DONE]\n\n"]


@contextlib.asynccontextmanager
async def cancel_on_disconnect(is_disconnected: Callable[[], Awaitable[bool]], poll_s: float):
    """Cancel the block and raise ClientDisconnected when the client goes away.

    The watcher also stops on a flag: Starlette's is_disconnected polls under a pre-cancelled anyio scope, and a
    task.cancel() that lands in that window can be absorbed with it, so cancel() alone could leave the watcher
    polling after the block ended (seen as a rare hang before the flag; test_proxy covers it)."""
    gone = stopped = False
    watcher = None
    try:
        with anyio.CancelScope() as scope:
            async def watch():
                nonlocal gone
                while not stopped:
                    if await is_disconnected():
                        gone = True
                        scope.cancel()
                        return
                    if stopped:
                        return
                    await asyncio.sleep(poll_s)

            watcher = asyncio.get_running_loop().create_task(watch())
            yield
    finally:
        stopped = True
        if watcher is not None:
            watcher.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.wait([watcher], timeout=WATCHER_STOP_TIMEOUT_S)   # bounded: never hang a request
    if gone:
        raise ClientDisconnected


def _openai_shaped(content: bytes) -> bool:
    try:
        doc = json.loads(content)
    except ValueError:
        return False
    return isinstance(doc, dict) and isinstance(doc.get("error"), dict)


def passthrough(upstream: httpx.Response, content: bytes | None = None) -> Response:
    content = upstream.content if content is None else content
    status = upstream.status_code
    if status == 200 or _openai_shaped(content):
        return Response(content, status_code=status,
                        media_type=upstream.headers.get("content-type", "application/json"))
    type_ = "server_error" if status >= 500 else "invalid_request_error"
    return openai_error(status, type_, f"upstream_{status}", f"the model server returned {status}")


async def read_capped(upstream: httpx.Response, limit: int = ERROR_BODY_LIMIT) -> bytes:
    out = b""
    async for chunk in upstream.aiter_bytes():
        out += chunk
        if len(out) >= limit:
            break
    return out[:limit]


class ProxyStream(StreamingResponse):
    def __init__(self, upstream: httpx.Response, *, rec, deadline: float, clock: Callable[[], float],
                 cleanup: Callable[[], None]):
        self.upstream = upstream
        self.rec = rec
        self.deadline = deadline
        self.clock = clock
        self._cleanup = cleanup
        self._ended = False
        super().__init__(self._events(), media_type="text/event-stream", headers=SSE_HEADERS)

    def _fail(self, outcome: str, kind: str | None, code: str, message: str) -> list:
        self.rec.outcome, self.rec.upstream_error = outcome, kind
        return sse_error(code, message)

    async def _events(self):
        framer = SSEFramer()
        chunks = self.upstream.aiter_bytes()
        ending = []
        try:
            while True:
                remaining = self.deadline - self.clock()
                if remaining <= 0:   # passed while writing to the client (reads are bounded): not an upstream error
                    ending = self._fail("deadline", None, "deadline_exceeded", "the request deadline passed")
                    break
                try:
                    with anyio.fail_after(remaining):   # this read only; every yield is outside the scope
                        chunk = await chunks.__anext__()
                except StopAsyncIteration:
                    if framer.pending.strip():
                        ending = self._fail("upstream_error", "stream_broken", "upstream_stream_broken",
                                            "the model server stream ended inside an event")
                    break
                except TimeoutError:
                    ending = self._fail("deadline", "timeout", "deadline_exceeded", "the request deadline passed")
                    break
                except httpx.TransportError:
                    ending = self._fail("upstream_error", "stream_broken", "upstream_stream_broken",
                                        "the model server stream broke")
                    break
                for event in framer.feed(chunk):
                    if self.rec.outcome is None and is_error(event):   # forwarded as is, but never counted ok
                        self.rec.outcome, self.rec.upstream_error = "upstream_error", "stream_error"
                    if self.rec.ttft_s is None and is_token_event(event):   # vLLM bench's first token
                        now = self.clock()
                        self.rec.ttft_s = now - self.rec.arrival
                        if self.rec.upstream_sent is not None:
                            self.rec.upstream_ttft_s = now - self.rec.upstream_sent
                    yield event
            for event in ending:
                yield event
            if self.rec.outcome is None:
                self.rec.outcome = "ok"
            self._ended = True
        finally:
            with contextlib.suppress(Exception):
                await chunks.aclose()

    async def _send(self, send, message) -> None:
        with anyio.fail_after(max(self.deadline - self.clock(), 0.0)):   # this send only, never across a yield
            await send(message)

    async def stream_response(self, send) -> None:
        """Starlette's, with every write bounded by the request deadline: a client that stays connected but stops
        reading blocks send on flow control, which neither the upstream read deadline nor the disconnect watcher
        would end. At the deadline the response is abandoned unfinished (uvicorn closes the connection) and the
        shielded cleanup releases the permit. A write that need not wait still goes out after the deadline (the
        error events)."""
        try:
            await self._send(send, {"type": "http.response.start", "status": self.status_code,
                                    "headers": self.raw_headers})
            async for chunk in self.body_iterator:
                await self._send(send, {"type": "http.response.body", "body": chunk, "more_body": True})
            await self._send(send, {"type": "http.response.body", "body": b"", "more_body": False})
        except TimeoutError:
            if self.rec.outcome in (None, "ok"):   # the client stopped reading; not an upstream error
                self.rec.outcome = "deadline"

    async def __call__(self, scope, receive, send):
        interrupted = False
        try:
            await super().__call__(scope, receive, send)
        except BaseException:
            interrupted = True
            raise
        finally:
            with anyio.CancelScope(shield=True):
                with anyio.move_on_after(CLEANUP_TIMEOUT_S):
                    with contextlib.suppress(Exception):
                        await self.body_iterator.aclose()
                    with contextlib.suppress(Exception):
                        await self.upstream.aclose()
                self._cleanup()
            if not self._ended and self.rec.outcome is None:
                self.rec.outcome = "cancelled"
                if not interrupted:   # Starlette stopped the stream because the client disconnected
                    self.rec.disconnect_phase = "streaming"
