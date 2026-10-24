"""Tests for capserve/proxy.py: SSEFramer (complete events only), the per-read deadline, ProxyStream's shielded
cleanup, the disconnect watcher, and pass-through of non-stream responses (decision 029 "非串流" / "串流")."""

import asyncio
import json
import random
import time

import httpx
import pytest

from capserve import proxy as px
from capserve.guard import RequestRecord

EVENTS = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
          b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
          b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n',
          b"data: [DONE]\n\n"]


# ------------------------------------------------------------------ SSEFramer --

def test_framer_yields_only_complete_events_at_any_split():
    stream = b"".join(EVENTS)
    rng = random.Random(3)
    splits = [[stream[i:i + 1] for i in range(len(stream))], [stream]]
    for _ in range(50):
        cuts = sorted(rng.sample(range(1, len(stream)), rng.randrange(1, 12)))
        splits.append([stream[a:b] for a, b in zip([0] + cuts, cuts + [len(stream)])])
    for parts in splits:
        f = px.SSEFramer()
        out = [e for p in parts for e in f.feed(p)]
        assert out == EVENTS and f.pending == b""


def test_framer_handles_crlf_and_keeps_the_partial_tail():
    f = px.SSEFramer()
    assert f.feed(b"data: a\r\n\r\ndata: b\r\n") == [b"data: a\r\n\r\n"]
    assert f.pending == b"data: b\r\n"
    assert f.feed(b"\r\n") == [b"data: b\r\n\r\n"] and f.pending == b""


@pytest.mark.parametrize("event, want", [
    (EVENTS[0], False), (EVENTS[1], True), (EVENTS[3], False),
    (b'data: {"choices":[{"text":""}]}\n\n', False), (b'data: {"choices":[{"text":"x"}]}\n\n', True),
    (b"data: not json\n\n", False), (b": keep-alive\n\n", False), (b'data: {"choices":[]}\n\n', False),
])
def test_first_content_detection(event, want):
    assert px.has_content(event) is want


USAGE_ONLY = b'data: {"id":"cmpl-1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1}}\n\n'


@pytest.mark.parametrize("event, want", [
    (EVENTS[0], True),                                                     # chat's role chunk carries a choice
    (EVENTS[1], True), (EVENTS[3], False),                                 # [DONE] is no token
    (b'data: {"choices":[{"text":""}]}\n\n', True),                        # empty text is still a token
    (b'data: {"choices":[{"text":"","finish_reason":"length"}]}\n\n', True),
    (b'data: {"choices":[{"delta":{"content":""}}]}\n\n', True),
    (b'data: {"choices":[{"text":"x"}]}\n\n', True),
    (USAGE_ONLY, False), (b'data: {"choices":[]}\n\n', False),             # the usage-only chunk is no token
    (b'data: {"usage":{"completion_tokens":1}}\n\n', False), (b'data: {"choices":null}\n\n', False),
    (b'data: {"error":{"message":"boom","type":"x","code":500}}\n\n', False),   # an in-band error is no token
    (b"data: not json\n\n", False), (b": keep-alive\n\n", False), (b'data: [1, 2]\n\n', False),
    (b'data: {"choices":{"0":{}}}\n\n', False),
])
def test_the_first_token_event_is_vllm_benchs(event, want):
    """044 實作補記六: the gateway's first-token mark is vLLM 0.30 bench's (endpoint_request_func: the first data event
    whose JSON has a non-empty choices list; not [DONE], not the usage-only chunk), not the first non-empty text."""
    assert px.is_token_event(event) is want


class Timed(httpx.AsyncByteStream):
    """An upstream body whose events arrive at scripted times of a fake clock: (t, bytes) sets now to t, then yields."""

    def __init__(self, steps, now):
        self.steps, self.now = steps, now

    async def __aiter__(self):
        for t, data in self.steps:
            self.now[0] = t
            yield data

    async def aclose(self):
        pass


def cmpl(text, finish=None) -> bytes:
    return b"data: " + json.dumps({"id": "cmpl-1", "choices": [{"index": 0, "text": text,
                                                                 "finish_reason": finish}]}).encode() + b"\n\n"


async def timed_marks(steps) -> tuple:
    """(ttft_s, upstream_ttft_s, outcome) of a completions stream: arrival at 0, upstream send at 0.25."""
    now = [0.0]
    rec = RequestRecord(request_id="r", method="POST", path="/v1/completions", arrival=0.0, route="completions",
                        stream=True, upstream_sent=0.25)
    resp = px.ProxyStream(httpx.Response(200, stream=Timed(steps, now)), rec=rec, deadline=1e9,
                          clock=lambda: now[0], cleanup=lambda: None)
    peer = Peer()
    await resp(SCOPE, peer.receive, peer.send)
    assert b"".join(peer.chunks) == b"".join(data for _, data in steps)   # the bytes pass through unchanged
    return rec.ttft_s, rec.upstream_ttft_s, rec.outcome


DONE = b"data: [DONE]\n\n"


@pytest.mark.anyio
@pytest.mark.parametrize("steps, want", [
    # an empty-text first chunk (a partial UTF-8 byte-BPE token, a skipped special token), then text: the first chunk
    ([(1.0, cmpl("")), (2.0, cmpl("x")), (2.5, cmpl("y", "length")), (2.6, USAGE_ONLY), (2.7, DONE)], (1.0, 0.75)),
    # out = 1 whose only chunk has empty text
    ([(1.5, cmpl("", "length")), (1.6, USAGE_ONLY), (1.7, DONE)], (1.5, 1.25)),
    # every chunk empty, then usage and [DONE]
    ([(1.0, cmpl("")), (2.0, cmpl("")), (3.0, cmpl("", "length")), (3.1, USAGE_ONLY), (3.2, DONE)], (1.0, 0.75)),
    # the usage-only chunk (and [DONE]) never mark
    ([(1.0, USAGE_ONLY), (2.0, cmpl("x", "length")), (3.0, DONE)], (2.0, 1.75)),
    ([(1.0, USAGE_ONLY), (2.0, DONE)], (None, None)),
])
async def test_ttft_marks_the_first_token_event_as_vllm_bench_does(steps, want):
    ttft, upstream_ttft, outcome = await timed_marks(steps)
    assert (ttft, upstream_ttft) == want and outcome == "ok"


def test_error_event_then_done():
    events = px.sse_error("upstream_stream_broken", "the model server stream broke")
    assert px.SSEFramer().feed(b"".join(events)) == events
    assert len(events) == 2 and events[1] == b"data: [DONE]\n\n"
    body = json.loads(events[0].removeprefix(b"data: "))
    assert body["error"]["code"] == "upstream_stream_broken" and body["error"]["type"] == "server_error"


# ------------------------------------------------------------------ ProxyStream --

class Script(httpx.AsyncByteStream):
    """An upstream body: bytes are yielded, an Exception is raised, an asyncio.Event is waited for."""

    def __init__(self, steps):
        self.steps = steps
        self.closed = False

    async def __aiter__(self):
        for step in self.steps:
            if isinstance(step, Exception):
                raise step
            if isinstance(step, asyncio.Event):
                await step.wait()
                continue
            yield step

    async def aclose(self):
        self.closed = True


class Peer:
    """The client side of one ASGI response (spec 2.3): send collects, receive blocks until disconnect()."""

    def __init__(self):
        self.sent = []
        self.gone = asyncio.Event()
        self.started = asyncio.Event()

    async def receive(self):
        await self.gone.wait()
        return {"type": "http.disconnect"}

    async def send(self, message):
        self.sent.append(message)
        if message.get("body"):
            self.started.set()

    @property
    def chunks(self):
        return [m["body"] for m in self.sent if m["type"] == "http.response.body" and m["body"]]


SCOPE = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}}


def make_stream(steps, *, deadline_in=30.0):
    script = Script(steps)
    upstream = httpx.Response(200, stream=script)
    released = []
    rec = RequestRecord(request_id="r", method="POST", path="/v1/chat/completions", arrival=time.monotonic() - 1.0,
                        route="chat", stream=True, upstream_sent=time.monotonic() - 0.5)
    resp = px.ProxyStream(upstream, rec=rec, deadline=time.monotonic() + deadline_in, clock=time.monotonic,
                          cleanup=lambda: released.append(script.closed))   # 029: the upstream closes first
    return resp, script, rec, released


@pytest.mark.anyio
async def test_stream_passes_complete_events_and_cleans_up_once():
    stream = b"".join(EVENTS)
    resp, script, rec, released = make_stream([stream[:10], stream[10:70], stream[70:]])
    peer = Peer()
    await resp(SCOPE, peer.receive, peer.send)
    assert peer.chunks == EVENTS
    start = peer.sent[0]
    headers = {k.decode(): v.decode() for k, v in start["headers"]}
    assert headers["content-type"].startswith("text/event-stream")
    assert (headers["cache-control"], headers["x-accel-buffering"]) == ("no-cache", "no")
    assert "content-length" not in headers
    assert rec.outcome == "ok" and released == [True] and script.closed
    assert 0.9 < rec.ttft_s < 5 and 0.4 < rec.upstream_ttft_s < 5


@pytest.mark.anyio
async def test_a_mid_event_drop_gives_one_error_event_and_done():
    resp, script, rec, released = make_stream([EVENTS[0] + EVENTS[1][:15], httpx.RemoteProtocolError("peer closed")])
    peer = Peer()
    await resp(SCOPE, peer.receive, peer.send)
    assert peer.chunks[0] == EVENTS[0] and len(peer.chunks) == 3 and peer.chunks[2] == b"data: [DONE]\n\n"
    assert json.loads(peer.chunks[1].removeprefix(b"data: "))["error"]["code"] == "upstream_stream_broken"
    assert (rec.outcome, rec.upstream_error) == ("upstream_error", "stream_broken") and released == [True]


@pytest.mark.anyio
async def test_an_in_band_upstream_error_is_forwarded_and_is_not_ok():
    # vLLM reports a failure after the stream started (engine error, abort) on the HTTP 200 stream
    error = (b'data: {"error": {"message": "EngineCore encountered an issue", "type": "InternalServerError", '
             b'"param": null, "code": 500}}\n\n')
    resp, _, rec, released = make_stream([EVENTS[0] + EVENTS[1] + error + EVENTS[3]])
    peer = Peer()
    await resp(SCOPE, peer.receive, peer.send)
    assert peer.chunks == [EVENTS[0], EVENTS[1], error, EVENTS[3]]
    assert (rec.outcome, rec.upstream_error) == ("upstream_error", "stream_error") and released == [True]


@pytest.mark.parametrize("event, want", [
    (b'data: {"error": {"message": "x", "type": "InternalServerError", "param": null, "code": 500}}\n\n', True),
    (b'data: {"choices":[{"delta":{"content":"\\"error\\": {}"}}]}\n\n', False), (EVENTS[3], False),
    (b'data: {"error": "not an object"}\n\n', False), (b": error\n\n", False),
])
def test_in_band_error_detection(event, want):
    assert px.is_error(event) is want


@pytest.mark.anyio
async def test_eof_inside_an_event_is_a_broken_stream():
    resp, _, rec, _ = make_stream([EVENTS[0] + b"data: {"])
    peer = Peer()
    await resp(SCOPE, peer.receive, peer.send)
    assert peer.chunks[0] == EVENTS[0] and peer.chunks[-1] == b"data: [DONE]\n\n" and len(peer.chunks) == 3
    assert rec.upstream_error == "stream_broken"


@pytest.mark.anyio
async def test_the_deadline_bounds_each_upstream_read():
    never = asyncio.Event()
    resp, script, rec, released = make_stream([EVENTS[0], never], deadline_in=0.05)
    peer = Peer()
    await resp(SCOPE, peer.receive, peer.send)
    assert peer.chunks[0] == EVENTS[0] and peer.chunks[-1] == b"data: [DONE]\n\n"
    assert json.loads(peer.chunks[1].removeprefix(b"data: "))["error"]["code"] == "deadline_exceeded"
    assert (rec.outcome, rec.upstream_error) == ("deadline", "timeout") and released == [True] and script.closed


@pytest.mark.anyio
async def test_a_deadline_spent_writing_to_the_client_is_not_an_upstream_timeout():
    now = [0.0]
    script = Script([EVENTS[0], EVENTS[1], EVENTS[2]])
    rec = RequestRecord(request_id="r", method="POST", path="/v1/chat/completions", arrival=0.0, route="chat",
                        stream=True)
    resp = px.ProxyStream(httpx.Response(200, stream=script), rec=rec, deadline=10.0, clock=lambda: now[0],
                          cleanup=lambda: None)
    peer = Peer()

    async def slow_send(message):
        await peer.send(message)
        if message.get("body"):
            now[0] = 11.0          # this write took the client past the deadline; no upstream read was waiting

    await resp(SCOPE, peer.receive, slow_send)
    assert peer.chunks[0] == EVENTS[0] and peer.chunks[-1] == b"data: [DONE]\n\n" and len(peer.chunks) == 3
    assert json.loads(peer.chunks[1].removeprefix(b"data: "))["error"]["code"] == "deadline_exceeded"
    assert (rec.outcome, rec.upstream_error) == ("deadline", None) and script.closed


@pytest.mark.anyio
async def test_a_client_that_stops_reading_is_cut_at_the_deadline():
    resp, script, rec, released = make_stream([EVENTS[0], EVENTS[1], EVENTS[2]], deadline_in=0.05)
    peer = Peer()
    blocked = asyncio.Event()

    async def stalled_send(message):
        await peer.send(message)
        if message.get("body"):
            await blocked.wait()   # flow control: the client's buffers are full and it never reads again

    await asyncio.wait_for(resp(SCOPE, peer.receive, stalled_send), 5)
    assert peer.chunks == [EVENTS[0]] and not any(m.get("more_body") is False for m in peer.sent)
    assert (rec.outcome, rec.upstream_error) == ("deadline", None) and released == [True]


@pytest.mark.anyio
async def test_client_disconnect_mid_stream_closes_upstream_and_releases():
    never = asyncio.Event()
    resp, script, rec, released = make_stream([EVENTS[0], EVENTS[1], never])
    peer = Peer()
    task = asyncio.create_task(resp(SCOPE, peer.receive, peer.send))
    await peer.started.wait()
    peer.gone.set()
    await task                                   # no RuntimeError from closing the generator in another task
    assert script.closed and released == [True]
    assert (rec.outcome, rec.disconnect_phase) == ("cancelled", "streaming")


@pytest.mark.anyio
async def test_server_cancellation_still_releases():
    never = asyncio.Event()
    resp, script, rec, released = make_stream([EVENTS[0], never])
    peer = Peer()
    task = asyncio.create_task(resp(SCOPE, peer.receive, peer.send))
    await peer.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert script.closed and released == [True]
    assert rec.outcome == "cancelled" and rec.disconnect_phase is None


# ------------------------------------------------------------------ disconnect watcher --

@pytest.mark.anyio
async def test_cancel_on_disconnect_cancels_the_block_and_raises():
    state = {"gone": False}
    entered = asyncio.Event()

    async def is_disconnected():
        return state["gone"]

    async def body():
        async with px.cancel_on_disconnect(is_disconnected, 0.001):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(body())
    await entered.wait()
    state["gone"] = True
    with pytest.raises(px.ClientDisconnected):
        await task


@pytest.mark.anyio
async def test_cancel_on_disconnect_is_transparent_otherwise():
    async def connected():
        return False

    async with px.cancel_on_disconnect(connected, 0.001):
        await asyncio.sleep(0.005)
    with pytest.raises(KeyError):
        async with px.cancel_on_disconnect(connected, 0.001):
            raise KeyError("x")


@pytest.mark.anyio
async def test_cancel_on_disconnect_exits_even_if_a_poll_absorbs_the_cancel():
    entered, absorbed = asyncio.Event(), []

    async def absorbing():
        entered.set()
        if not absorbed:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                absorbed.append(1)  # what a pre-cancelled anyio scope can do with a colliding task.cancel()
        return False

    async def body():
        async with px.cancel_on_disconnect(absorbing, 0.001):
            await entered.wait()

    await asyncio.wait_for(body(), 0.5)   # the watcher stops at once, not after a 10 s poll


# ------------------------------------------------------------------ pass-through --

def test_passthrough_200_copies_only_body_and_content_type():
    up = httpx.Response(200, content=b'{"id":"x"}', headers={"content-type": "application/json",
                                                             "transfer-encoding": "chunked", "connection": "close",
                                                             "x-vllm": "1"})
    r = px.passthrough(up)
    assert r.status_code == 200 and r.body == b'{"id":"x"}'
    assert r.headers["content-type"] == "application/json" and "x-vllm" not in r.headers
    assert "transfer-encoding" not in r.headers and "connection" not in r.headers


def test_passthrough_keeps_openai_shaped_errors_and_wraps_others():
    shaped = {"error": {"message": "too long", "type": "BadRequestError", "param": None, "code": 400}}
    r = px.passthrough(httpx.Response(400, json=shaped))
    assert r.status_code == 400 and json.loads(r.body) == shaped
    r = px.passthrough(httpx.Response(500, text="Internal Server Error"))
    assert r.status_code == 500
    assert json.loads(r.body)["error"] == {"message": "the model server returned 500", "type": "server_error",
                                           "param": None, "code": "upstream_500"}
    r = px.passthrough(httpx.Response(422, json={"detail": "x"}))
    assert json.loads(r.body)["error"]["type"] == "invalid_request_error"


@pytest.mark.anyio
async def test_read_capped_stops_at_the_limit():
    up = httpx.Response(500, stream=Script([b"a" * 40, b"b" * 40, b"c" * 40]))
    assert await px.read_capped(up, 64) == b"a" * 40 + b"b" * 24
