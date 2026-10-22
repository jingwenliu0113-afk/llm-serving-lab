"""Tests for capserve/guard.py: the outermost pure-ASGI middleware (request id, pending cap enforced before the body
is read, shielded LIFO cleanups registered on the scope, one JSON access-log line; decision 029 steps 1 and 10)."""

import asyncio
import json
import logging

import pytest

from capserve import guard as gd
from capserve.admission import AdmissionController

pytestmark = pytest.mark.anyio
HEADER = "provisional-x; status=provisional"


class Finished:
    def __init__(self):
        self.records = []

    def finish(self, rec):
        self.records.append(rec)


def make_guard(app, max_pending=2, metrics=None):
    return gd.Guard(app, max_pending=max_pending, profile_header=HEADER, retry_after_s=lambda reason: 2,
                    metrics=metrics or Finished())


def scope(path="/v1/completions", method="POST"):
    return {"type": "http", "method": method, "path": path, "headers": [], "query_string": b"",
            "asgi": {"version": "3.0", "spec_version": "2.3"}}


class Client:
    def __init__(self, body=b"{}"):
        self.body = body
        self.receive_calls = 0
        self.sent = []

    async def receive(self):
        self.receive_calls += 1
        return {"type": "http.request", "body": self.body, "more_body": False}

    async def send(self, message):
        self.sent.append(message)

    @property
    def status(self):
        return next(m["status"] for m in self.sent if m["type"] == "http.response.start")

    @property
    def headers(self):
        start = next(m for m in self.sent if m["type"] == "http.response.start")
        return {k.decode(): v.decode() for k, v in start["headers"]}

    @property
    def json(self):
        return json.loads(b"".join(m.get("body", b"") for m in self.sent if m["type"] == "http.response.body"))


async def ok_app(scope, receive, send):
    await receive()
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": b"ok"})


def blocking_app(gate, entered):
    async def app(scope, receive, send):
        entered.append(scope["path"])
        await gate.wait()
        await ok_app(scope, receive, send)
    return app


async def until(pred):
    for _ in range(1000):
        if pred():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


async def test_every_response_carries_request_id_and_profile_headers():
    g = make_guard(ok_app)
    c = Client()
    await g(scope(), c.receive, c.send)
    assert c.status == 200 and c.headers["x-capserve-profile"] == HEADER
    rid = c.headers["x-request-id"]
    assert len(rid) == 32 and int(rid, 16) >= 0
    c2 = Client()
    await g(scope(), c2.receive, c2.send)
    assert c2.headers["x-request-id"] != rid


async def test_a_client_supplied_request_id_is_never_used():
    g = make_guard(ok_app)
    c = Client()
    await g({**scope(), "headers": [(b"x-request-id", b"client-value")]}, c.receive, c.send)
    rid = c.headers["x-request-id"]
    assert rid != "client-value" and len(rid) == 32 and int(rid, 16) >= 0
    assert [v for k, v in c.sent[0]["headers"] if k == b"x-request-id"] == [rid.encode()]


async def test_pending_cap_rejects_before_the_body_is_read():
    gate, entered = asyncio.Event(), []
    fin = Finished()
    g = make_guard(blocking_app(gate, entered), max_pending=2, metrics=fin)
    clients = [Client(), Client()]
    tasks = [asyncio.create_task(g(scope(), c.receive, c.send)) for c in clients]
    await until(lambda: len(entered) == 2)
    assert g.pending == 2
    late = Client()
    await g(scope(), late.receive, late.send)
    assert late.receive_calls == 0 and late.status == 429
    assert late.json == {"error": {"message": late.json["error"]["message"], "type": "rate_limit_error",
                                   "param": None, "code": "queue_full"}}
    assert (late.headers["retry-after"], late.headers["x-should-retry"]) == ("2", "true")
    assert "x-request-id" in late.headers and late.headers["x-capserve-profile"] == HEADER
    assert (fin.records[-1].outcome, fin.records[-1].reason, fin.records[-1].route) == (
        "rejected", "queue_full", "completions")
    gate.set()
    await asyncio.gather(*tasks)
    assert g.pending == 0 and [c.status for c in clients] == [200, 200]


async def test_health_ready_and_metrics_are_exempt_from_the_cap():
    gate, entered = asyncio.Event(), []
    g = make_guard(blocking_app(gate, entered), max_pending=1)
    first = Client()
    t = asyncio.create_task(g(scope(), first.receive, first.send))
    await until(lambda: len(entered) == 1)
    gate.set()
    for path in ("/healthz", "/readyz", "/metrics"):
        c = Client()
        await g(scope(path, "GET"), c.receive, c.send)
        assert c.status == 200
    await t


async def test_cleanup_runs_when_the_endpoint_raises_and_the_permit_returns():
    ctrl = AdmissionController(max_concurrency=1, max_queue=0, max_queue_wait_s=1, retry_after_s={"queue_full": 1})
    fin = Finished()

    async def app(scope, receive, send):
        permit = await ctrl.acquire(deadline=1e12)
        gd.on_cleanup(scope, permit.release)
        raise RuntimeError("boom")

    g = make_guard(app, metrics=fin)
    with pytest.raises(RuntimeError):
        await g(scope(), Client().receive, Client().send)
    assert (ctrl.in_flight, g.pending) == (0, 0) and fin.records[-1].outcome == "server_error"


async def test_cleanup_runs_when_the_response_raises_after_it_started():
    ran = []

    async def app(scope, receive, send):
        gd.on_cleanup(scope, lambda: ran.append("sync"))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise OSError("client gone")

    with pytest.raises(OSError):
        await make_guard(app)(scope(), Client().receive, Client().send)
    assert ran == ["sync"]


async def test_cleanup_is_shielded_when_the_request_task_is_cancelled():
    ran, gate, entered = [], asyncio.Event(), []

    async def slow_cleanup():
        await asyncio.sleep(0.01)          # needs the loop although its task is being cancelled
        ran.append("async")

    async def app(scope, receive, send):
        gd.on_cleanup(scope, slow_cleanup)
        entered.append(1)
        await gate.wait()

    fin = Finished()
    g = make_guard(app, metrics=fin)
    t = asyncio.create_task(g(scope(), Client().receive, Client().send))
    await until(lambda: entered)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert ran == ["async"] and g.pending == 0 and fin.records[-1].outcome == "cancelled"


async def test_cleanups_run_lifo_once_each_even_if_one_fails():
    ran = []
    s = scope()

    def bad():
        ran.append("bad")
        raise ValueError("x")

    gd.on_cleanup(s, lambda: ran.append(1))
    gd.on_cleanup(s, bad)
    gd.on_cleanup(s, lambda: ran.append(3))
    await gd.run_cleanups(s)
    await gd.run_cleanups(s)
    assert ran == [3, "bad", 1]


async def test_a_hanging_cleanup_is_bounded():
    s = scope()
    ran = []

    async def hang():
        await asyncio.Event().wait()

    gd.on_cleanup(s, lambda: ran.append("after"))
    gd.on_cleanup(s, hang)
    await gd.run_cleanups(s, timeout_s=0.01)
    assert ran == ["after"]


async def test_one_json_access_log_line_without_request_text(caplog):
    async def app(scope, receive, send):
        await receive()
        scope["capserve.record"].outcome = "ok"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"COMPLETION TEXT"})

    caplog.set_level(logging.INFO, logger="capserve.access")
    c = Client(body=b'{"prompt": "SECRET PROMPT"}')
    await make_guard(app)(scope(), c.receive, c.send)
    lines = [r.getMessage() for r in caplog.records if r.name == "capserve.access"]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert (entry["status"], entry["outcome"], entry["path"], entry["request_id"]) == (
        200, "ok", "/v1/completions", c.headers["x-request-id"])
    assert "SECRET" not in lines[0] and "COMPLETION" not in lines[0] and entry["duration_s"] >= 0


async def test_non_http_scopes_pass_through():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    await make_guard(app)({"type": "lifespan"}, None, None)
    assert seen == ["lifespan"]
