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


def access_entries(caplog) -> list:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "capserve.access"]


async def test_the_access_log_has_the_044_fields_on_the_injected_monotonic_clock(caplog):
    """044 item 1: t_arrival_s / t_sent_s / t_released_s on the gateway's monotonic clock (arrival, upstream send, the
    permit's release), prefill_wait_s, admission_class and admission_units; each only once it is set."""
    clock = iter([10.0, 13.5])   # the guard reads its clock at arrival and at the end (duration_s)

    async def app(scope, receive, send):
        rec = scope["capserve.record"]
        rec.upstream_sent, rec.t_released_s, rec.prefill_wait_s = 11.25, 12.75, 0.5
        rec.admission_class, rec.admission_units, rec.outcome = "v-uniform-8k", 4, "ok"
        await ok_app(scope, receive, send)

    caplog.set_level(logging.INFO, logger="capserve.access")
    g = gd.Guard(app, max_pending=2, profile_header=HEADER, retry_after_s=lambda reason: 2, metrics=Finished(),
                 clock=lambda: next(clock))
    await g(scope(), Client().receive, Client().send)
    (entry,) = access_entries(caplog)
    assert (entry["t_arrival_s"], entry["t_sent_s"], entry["t_released_s"], entry["duration_s"]) == (
        10.0, 11.25, 12.75, 3.5)
    assert (entry["prefill_wait_s"], entry["admission_class"], entry["admission_units"]) == (0.5, "v-uniform-8k", 4)
    assert "arrival" not in entry and "upstream_sent" not in entry
    caplog.clear()
    await make_guard(ok_app)(scope("/readyz", "GET"), Client().receive, Client().send)
    (entry,) = access_entries(caplog)
    assert "t_arrival_s" in entry and not {"t_sent_s", "t_released_s", "prefill_wait_s", "admission_class",
                                            "admission_units", "client_request_id"} & set(entry)


@pytest.mark.parametrize("headers, want", [
    ([(b"x-request-id", b"V-P-U2K-c32-r1-17")], "V-P-U2K-c32-r1-17"),
    ([(b"x-request-id", b"bench-1a2b3c4d-0")], "bench-1a2b3c4d-0"),
    ([(b"x-request-id", b"a.b_c:d-E9")], "a.b_c:d-E9"),
    ([(b"x-request-id", b"x" * 128)], "x" * 128),
    ([(b"x-request-id", b"x" * 129)], None),
    ([(b"x-request-id", b"")], None),
    ([(b"x-request-id", b"has space")], None),
    ([(b"x-request-id", b"tag\n")], None),
    ([(b"x-request-id", b"q\"uote")], None),
    ([(b"x-request-id", "é".encode())], None),
    ([(b"x-request-id", b"a"), (b"x-request-id", b"b")], None),
    ([], None),
])
async def test_the_client_request_id_is_sanitised_and_never_echoed(caplog, headers, want):
    """044 item 1: the client's x-request-id, only as one [A-Za-z0-9._:-]{1,128} value (else null); the response keeps
    the gateway's own x-request-id."""
    caplog.set_level(logging.INFO, logger="capserve.access")
    fin = Finished()
    c = Client()
    await make_guard(ok_app, metrics=fin)({**scope(), "headers": headers}, c.receive, c.send)
    assert fin.records[-1].client_request_id == want
    (entry,) = access_entries(caplog)
    assert entry.get("client_request_id") == want
    rid = c.headers["x-request-id"]
    assert len(rid) == 32 and rid == entry["request_id"] and rid != want
    assert [v for k, v in c.sent[0]["headers"] if k == b"x-request-id"] == [rid.encode()]


async def test_non_http_scopes_pass_through():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    await make_guard(app)({"type": "lifespan"}, None, None)
    assert seen == ["lifespan"]
