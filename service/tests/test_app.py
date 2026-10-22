"""Integration tests for capserve/app.py against the fake vLLM backend, both on real uvicorn (h11 / asyncio) in the
test's event loop (decision 029 "整合測試（假後端）"): non-stream and stream pass-through, disconnects cancel the
upstream request and free the slot, overload never exceeds the limit, readiness, auth, /tokenize only under a permit,
OpenAI-shaped errors everywhere, and the class statuses of a verified profile (decision 030 8.7: a verified 0 answers
503 class_not_serviceable, an unavailable class is never served)."""

import asyncio
import contextlib
import json
import socket
import urllib.parse

import httpx
import pytest

from capserve.app import create_app
from capserve.profile import CapacityProfile, ProfileError
from capserve.proxy import ClientDisconnected
from capserve.keys import load_keyring
from capserve.tokens import LocalTokenizer
from fakes.fake_tokenizer import WordsHF
from fakes.fake_vllm import MODEL, FakeConfig
from fakes.profiles import verified_class, verified_doc
from fakes.servers import gateway_settings, make_profile, stack

pytestmark = pytest.mark.anyio
CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "hello there"}]}
COMP = {"model": MODEL, "prompt": "hello there"}
OLD_BACKSTOP_SLOTS = 16   # uvicorn limit_concurrency was 2 x max_pending + 16 open connections


def address(st):
    url = urllib.parse.urlsplit(st.url)
    return url.hostname, url.port


async def until(pred, limit_s=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + limit_s
    while not pred():
        if loop.time() > end:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.002)


def error(resp):
    return resp.json()["error"]


def sse_events(text):
    return [e for e in text.split("\n\n") if e]


def verified_profile(**cls_kw) -> CapacityProfile:
    """A verified profile whose one class "test" (the stack's active class) comes from the given C_PROD."""
    return CapacityProfile.model_validate(verified_doc(classes=[verified_class("test", **cls_kw)]))


def metric(gw, name, **labels):
    from prometheus_client.parser import text_string_to_metric_families
    for fam in text_string_to_metric_families(gw.metrics.exposition().decode()):
        for s in fam.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return None


# ------------------------------------------------------------------ non-stream --

async def test_non_stream_chat_passes_through_with_the_bound_injected():
    async with stack() as st, st.client() as c:
        r = await c.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer client-secret"})
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == " t0 t1 t2 t3"
        assert r.headers["x-capserve-profile"] == "provisional-test; status=provisional"
        assert len(r.headers["x-request-id"]) == 32
        seen = st.fake.generations[0]
        assert seen.path == "/v1/chat/completions" and seen.body == {**CHAT, "max_completion_tokens": 128,
                                                                    "chat_template_kwargs": {"enable_thinking": False}}
        assert "authorization" not in seen.headers
        assert [s.path for s in st.fake.seen] == ["/v1/chat/completions"]   # fast path: no /tokenize
        assert (st.gateway.ctrl.in_flight, st.gateway.guard.pending) == (0, 0)
        assert metric(st.gateway, "capserve_requests_total", route="chat", stream="false", outcome="ok") == 1


@pytest.mark.parametrize("status, body, want_type", [
    (400, None, "BadRequestError"),
    (500, b"Internal Server Error", "server_error"),
])
async def test_upstream_errors_keep_their_status(status, body, want_type):
    async with stack(FakeConfig(fail_status=status, fail_body=body)) as st, st.client() as c:
        r = await c.post("/v1/completions", json=COMP)
        assert r.status_code == status and error(r)["type"] == want_type
        r = await c.post("/v1/completions", json={**COMP, "stream": True})
        assert r.status_code == status and error(r)["type"] == want_type
        assert st.gateway.ctrl.in_flight == 0
        if status == 500:
            assert metric(st.gateway, "capserve_upstream_errors_total", kind="status_5xx") == 2


async def test_the_deadline_gives_504_and_cancels_upstream():
    async with stack(FakeConfig(hang=True), request_deadline_s=0.3) as st, st.client() as c:
        r = await c.post("/v1/completions", json=COMP)
        assert r.status_code == 504 and error(r)["code"] == "deadline_exceeded"
        await asyncio.wait_for(st.fake.cancelled.wait(), 5)
        assert st.gateway.ctrl.in_flight == 0
        assert metric(st.gateway, "capserve_requests_total", route="completions", stream="false",
                      outcome="deadline") == 1


async def test_non_stream_disconnect_cancels_upstream_and_frees_the_slot():
    async with stack(FakeConfig(hang=True)) as st, st.client() as c:
        task = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        assert st.gateway.ctrl.in_flight == 1
        task.cancel()
        await asyncio.wait_for(st.fake.cancelled.wait(), 5)
        await until(lambda: st.gateway.ctrl.in_flight == 0 and st.gateway.guard.pending == 0)
        assert metric(st.gateway, "capserve_client_disconnects_total", phase="waiting_response") == 1


async def test_disconnect_while_queued_leaves_the_queue_and_the_running_request():
    prof = make_profile(max_concurrency=1, max_queue=1)
    async with stack(FakeConfig(hang=True), prof) as st, st.client() as c:
        running = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        queued = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await until(lambda: st.gateway.ctrl.queued == 1)
        queued.cancel()
        await until(lambda: st.gateway.ctrl.queued == 0 and st.gateway.guard.pending == 1)
        assert st.gateway.ctrl.in_flight == 1 and len(st.fake.generations) == 1
        assert metric(st.gateway, "capserve_client_disconnects_total", phase="queued") == 1
        st.fake.release.set()
        assert (await running).status_code == 200


async def test_a_slow_body_is_408():
    async def slow_body():
        yield b'{"model": "'
        await asyncio.sleep(0.5)
        yield MODEL.encode() + b'", "prompt": "x"}'

    async with stack(body_read_timeout_s=0.1) as st, st.client() as c:
        try:
            r = await c.post("/v1/completions", content=slow_body())
            assert r.status_code == 408 and error(r)["code"] == "request_timeout"
        except httpx.TransportError:
            pass            # the gateway may close the connection before the client finished writing
        await until(lambda: metric(st.gateway, "capserve_client_errors_total", code="request_timeout") == 1)
        assert st.fake.seen == [] and st.gateway.ctrl.in_flight == 0


# ------------------------------------------------------------------ stream --

async def test_stream_arrives_incrementally_and_ends_with_done():
    async with stack(FakeConfig(pause_after_chunks=2, split_events=True)) as st, st.client() as c:
        async with c.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            chunks, got = r.aiter_text(), ""
            while got.count("\n\n") < 2:
                got += await chunks.__anext__()
            assert "[DONE]" not in got and not st.fake.completed_ids   # the fake is still paused
            st.fake.release.set()
            async for text in chunks:
                got += text
        events = sse_events(got)
        assert events[-1] == "data: [DONE]" and all(e.startswith("data: ") for e in events)
        assert all(json.loads(e[6:]) for e in events[:-1])     # every forwarded event is complete JSON
        await until(lambda: st.gateway.ctrl.in_flight == 0)
        assert metric(st.gateway, "capserve_requests_total", route="chat", stream="true", outcome="ok") == 1
        assert metric(st.gateway, "capserve_ttft_seconds_count", route="chat") == 1
        assert metric(st.gateway, "capserve_upstream_ttft_seconds_count", route="chat") == 1


async def test_stream_drop_gives_one_error_event_and_frees_the_slot():
    async with stack(FakeConfig(drop_after_chunks=2)) as st, st.client() as c:
        r = await c.post("/v1/completions", json={**COMP, "stream": True})
        events = sse_events(r.text)
        assert r.status_code == 200 and events[-1] == "data: [DONE]"
        assert json.loads(events[-2][6:])["error"]["code"] == "upstream_stream_broken"
        assert all(json.loads(e[6:]) for e in events[:-1])
        assert len(events) == 2 + 2
        await until(lambda: st.gateway.ctrl.in_flight == 0)
        assert metric(st.gateway, "capserve_upstream_errors_total", kind="stream_broken") == 1


async def test_stream_disconnect_cancels_upstream_and_frees_the_slot():
    async with stack(FakeConfig(pause_after_chunks=1)) as st, st.client() as c:
        async with c.stream("POST", "/v1/completions", json={**COMP, "stream": True}) as r:
            async for _ in r.aiter_bytes():
                break
        await asyncio.wait_for(st.fake.cancelled.wait(), 5)
        await until(lambda: st.gateway.ctrl.in_flight == 0 and st.gateway.guard.pending == 0)
        assert metric(st.gateway, "capserve_client_disconnects_total", phase="streaming") == 1


async def test_a_stream_client_that_stops_reading_frees_the_slot_at_the_deadline():
    # once the socket buffers are full, every response write waits on flow control: neither the upstream read
    # deadline nor the disconnect watcher would ever end it
    prof = make_profile(max_concurrency=1, max_queue=0)
    async with stack(FakeConfig(n_tokens=40000), prof, request_deadline_s=0.5) as st:   # ~3 MB of SSE
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        sock.setblocking(False)
        await asyncio.get_running_loop().sock_connect(sock, address(st))
        reader, writer = await asyncio.open_connection(sock=sock, limit=1024)   # stops reading at 2 KiB buffered
        body = json.dumps({**COMP, "stream": True}).encode()
        writer.write(b"POST /v1/completions HTTP/1.1\r\nhost: gw\r\ncontent-type: application/json\r\n"
                     b"content-length: %d\r\n\r\n%s" % (len(body), body))
        try:
            assert (await reader.readline()).startswith(b"HTTP/1.1 200")
            await until(lambda: (st.gateway.ctrl.in_flight, st.gateway.guard.pending) == (0, 0))
            assert metric(st.gateway, "capserve_requests_total", route="completions", stream="true",
                          outcome="deadline") == 1
            assert metric(st.gateway, "capserve_upstream_errors_total", kind="timeout") is None   # not vLLM's fault
        finally:
            writer.close()


# ------------------------------------------------------------------ overload --

async def test_overload_never_exceeds_the_limit_and_every_429_has_retry_after():
    prof = make_profile(max_concurrency=4, max_queue=4)
    async with stack(FakeConfig(hang=True), prof) as st, st.client() as c:
        tasks = [asyncio.create_task(c.post("/v1/completions", json=COMP)) for _ in range(40)]
        await until(lambda: sum(t.done() for t in tasks) == 32 and st.fake.in_flight == 4)
        assert (st.fake.in_flight, st.gateway.ctrl.in_flight, st.gateway.ctrl.queued) == (4, 4, 4)
        st.fake.release.set()
        responses = await asyncio.gather(*tasks)
        codes = [r.status_code for r in responses]
        assert codes.count(200) == 8 and codes.count(429) == 32
        assert all(r.headers.get("retry-after") == "2" and r.headers.get("x-should-retry") == "true"
                   for r in responses if r.status_code == 429)
        assert {error(r)["code"] for r in responses if r.status_code == 429} <= {"queue_full"}
        assert st.fake.peak_in_flight <= 4 and st.gateway.ctrl.in_flight == 0


async def test_more_connections_than_the_old_uvicorn_backstop_get_only_200_or_429():
    # uvicorn's limit_concurrency counted open connections and answered past it with its own plain 503 (no
    # Retry-After, no x-request-id, never seen by the guard); the guard's pending cap must be the only limit
    prof = make_profile(max_concurrency=1, max_queue=1)
    async with stack(FakeConfig(hang=True), prof, pre_admission_slots=0) as st, st.client() as c:
        n = 2 * st.gateway.settings.max_pending(st.gateway.cls) + OLD_BACKSTOP_SLOTS + 4
        tasks = [asyncio.create_task(c.post("/v1/completions", json=COMP)) for _ in range(n)]
        await until(lambda: sum(t.done() for t in tasks) == n - 2 and st.fake.in_flight == 1)
        st.fake.release.set()
        responses = await asyncio.gather(*tasks)
        codes = [r.status_code for r in responses]
        assert codes.count(200) == 2 and codes.count(429) == n - 2
        assert all(r.headers.get("retry-after") == "2" and error(r)["code"] == "queue_full"
                   for r in responses if r.status_code == 429)


async def test_idle_connections_do_not_block_service():
    prof = make_profile(max_concurrency=1, max_queue=1)
    async with stack(FakeConfig(), prof, pre_admission_slots=0) as st, st.client() as c:
        n = 2 * st.gateway.settings.max_pending(st.gateway.cls) + OLD_BACKSTOP_SLOTS + 4
        idle = [(await asyncio.open_connection(*address(st)))[1] for _ in range(n)]   # never send a byte
        try:
            r = await c.post("/v1/completions", json=COMP)
            assert r.status_code == 200 and (st.gateway.guard.pending, st.gateway.ctrl.in_flight) == (0, 0)
        finally:
            for writer in idle:
                writer.close()


async def test_retry_after_follows_the_observed_service_time():
    """038 §3: once requests have completed, a 429's Retry-After is (queued + 1) x mean service / max_concurrency,
    clamped to [min_s, max_s]; before that it is the fixed value. 503s keep their fixed values."""
    prof = make_profile(max_concurrency=1, max_queue=0)
    retry = {"mode": "adaptive", "min_s": 0.001, "max_s": 60.0}
    async with stack(FakeConfig(ttft_s=0.05), prof, retry_after_s=retry) as st, st.client() as c:
        assert (await c.post("/v1/completions", json=COMP)).status_code == 200
        mean = st.gateway.retry.mean_service_s
        assert mean is not None and mean >= 0.05
        st.fake.cfg.hang = True
        first = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        await until(lambda: st.gateway.ctrl.in_flight == 1)
        r = await c.post("/v1/completions", json=COMP)
        assert r.status_code == 429 and error(r)["code"] == "queue_full"
        assert int(r.headers["retry-after-ms"]) == round(mean * 1000) and r.headers["retry-after"] == "1"
        st.fake.release.set()
        assert (await first).status_code == 200


async def test_fixed_retry_after_mode_keeps_the_mvp_values():
    prof = make_profile(max_concurrency=1, max_queue=0)
    async with stack(FakeConfig(), prof, retry_after_s={"mode": "fixed"}) as st, st.client() as c:
        assert (await c.post("/v1/completions", json=COMP)).status_code == 200
        st.fake.cfg.hang = True
        first = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        await until(lambda: st.gateway.ctrl.in_flight == 1)
        r = await c.post("/v1/completions", json=COMP)
        assert r.status_code == 429 and (r.headers["retry-after"], r.headers["retry-after-ms"]) == ("2", "2000")
        st.fake.release.set()
        await first


async def test_queue_timeout_is_429():
    prof = make_profile(max_concurrency=1, max_queue=1)
    async with stack(FakeConfig(hang=True), prof, max_queue_wait_s=0.05) as st, st.client() as c:
        first = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        r = await c.post("/v1/completions", json=COMP)
        assert r.status_code == 429 and error(r)["code"] == "queue_timeout" and r.headers["retry-after"] == "2"
        st.fake.release.set()
        assert (await first).status_code == 200


# ------------------------------------------------------------------ readiness --

async def test_ready_body_reports_profile_backend_and_limits():
    async with stack() as st, st.client() as c:
        r = await c.get("/readyz")
        body = r.json()
        assert r.status_code == 200 and body["ready"] is True
        assert body["profile"] == {"id": "provisional-test", "status": "provisional", "provisional": True}
        assert (body["serviceable"], body["class_status"], body["bound"]) == (True, None, None)
        assert body["limits"]["max_concurrency"] == 4 and body["limits"]["max_pending"] == 4 + 4 + 16
        assert (await c.get("/healthz")).json() == {"status": "ok"}


async def test_a_verified_class_is_ready_and_reports_its_status_and_bound():
    async with stack(profile=verified_profile(status="LOWER", value=4)) as st, st.client() as c:
        body = (await c.get("/readyz")).json()
        assert body["ready"] is True and body["profile"]["status"] == "verified"
        assert (body["serviceable"], body["class_status"], body["bound"]) == (True, "verified", "lower")
        assert body["limits"]["max_concurrency"] == 4 and body["limits"]["max_queue"] == 4
        r = await c.post("/v1/completions", json=COMP)
        assert r.status_code == 200
        assert r.headers["x-capserve-profile"] == "verified-formal-20261101-c1; status=verified"


async def test_a_verified_0_class_answers_every_completion_503_class_not_serviceable():
    """030 8.7: INFEASIBLE -> verified 0, refuse the class. The gateway starts, but every completion request gets 503
    class_not_serviceable before its body is read: never queued, never sent upstream, no Retry-After and
    x-should-retry false (retrying cannot help). /readyz says why it is not ready."""
    prof = verified_profile(workload="M8K", status="INFEASIBLE", value=0, max_prompt_tokens=32640)
    async with stack(profile=prof) as st, st.client() as c:
        for path, body in (("/v1/completions", COMP), ("/v1/chat/completions", {**CHAT, "stream": True}),
                           ("/v1/completions", {"model": "other", "prompt": "x", "tools": []})):
            r = await c.post(path, json=body)
            assert r.status_code == 503, r.text
            assert error(r)["code"] == "class_not_serviceable" and error(r)["type"] == "server_error"
            assert "retry-after" not in r.headers and "retry-after-ms" not in r.headers
            assert r.headers["x-should-retry"] == "false"
        assert st.fake.seen == [] and (st.gateway.ctrl.in_flight, st.gateway.ctrl.queued) == (0, 0)
        assert metric(st.gateway, "capserve_rejections_total", reason="class_not_serviceable") == 3
        assert metric(st.gateway, "capserve_concurrency_limit") == 0
        r = await c.get("/readyz")
        body = r.json()
        assert r.status_code == 503 and body["ready"] is False and body["backend"]["ready"] is True
        assert (body["serviceable"], body["class_status"], body["bound"]) == (False, "verified", "exact")
        assert body["not_ready_reason"] == "class_not_serviceable"
        assert body["limits"]["max_concurrency"] == 0 and body["limits"]["max_queue"] == 0
        assert (await c.get("/healthz")).status_code == 200 and (await c.get("/v1/models")).status_code == 200


async def test_a_verified_0_class_still_checks_the_key_first():
    prof = verified_profile(workload="M8K", status="INFEASIBLE", value=0, max_prompt_tokens=32640)
    async with stack(profile=prof, api_key=b"k") as st, st.client() as c:
        assert (await c.post("/v1/completions", json=COMP)).status_code == 401
        r = await c.post("/v1/completions", json=COMP, headers={"Authorization": "Bearer k"})
        assert r.status_code == 503 and error(r)["code"] == "class_not_serviceable"


@pytest.mark.parametrize("status", ["UNSTABLE", "NT"])
def test_an_unavailable_active_class_never_builds_a_gateway(status):
    """030 8.7: no verified number, so no gateway serves it (no flag exists that could override this)."""
    prof = verified_profile(workload="U8K", status=status, max_prompt_tokens=8064)
    with pytest.raises(ProfileError, match="unavailable"):
        create_app(gateway_settings("http://127.0.0.1:9"), prof)


async def test_backend_down_is_503_with_retry_after():
    async with stack(FakeConfig(healthy=False), wait_ready=False) as st, st.client() as c:
        r = await c.get("/readyz")
        assert r.status_code == 503 and r.json()["backend"]["ready"] is False
        r = await c.post("/v1/completions", json=COMP)
        assert r.status_code == 503 and error(r)["code"] == "backend_unavailable" and r.headers["retry-after"] == "5"
        assert st.fake.generations == []


async def test_backend_down_is_503_before_the_body_is_read():
    async with stack(FakeConfig(healthy=False), wait_ready=False) as st:
        reader, writer = await asyncio.open_connection(*address(st))
        writer.write(b"POST /v1/completions HTTP/1.1\r\nhost: gw\r\ncontent-type: application/json\r\n"
                     b"content-length: 1000\r\n\r\n{")                  # the rest of the body never comes
        try:
            status = await asyncio.wait_for(reader.readline(), 2.0)   # long before body_read_timeout_s (10 s)
        finally:
            writer.close()
        assert status.startswith(b"HTTP/1.1 503")


async def test_wrong_model_is_not_ready():
    async with stack(FakeConfig(model_id="other/model"), wait_ready=False) as st, st.client() as c:
        await until(lambda: "model" in (st.gateway.monitor.reason or ""))
        assert (await c.get("/readyz")).status_code == 503
        assert (await c.post("/v1/completions", json=COMP)).status_code == 503


async def test_draining_is_503_and_rejects_the_queue():
    prof = make_profile(max_concurrency=1, max_queue=2)
    async with stack(FakeConfig(hang=True), prof) as st, st.client() as c:
        running = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        queued = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await until(lambda: st.gateway.ctrl.queued == 1)
        st.gateway.set_draining()
        r = await queued
        assert r.status_code == 503 and error(r)["code"] == "draining" and r.headers["retry-after"] == "5"
        assert (await c.get("/readyz")).status_code == 503
        assert (await c.post("/v1/completions", json=COMP)).status_code == 503
        st.fake.release.set()
        assert (await running).status_code == 200          # requests already running finish


async def test_backend_down_rejects_queued_requests():
    prof = make_profile(max_concurrency=1, max_queue=2)
    async with stack(FakeConfig(hang=True), prof) as st, st.client() as c:
        running = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        queued = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await until(lambda: st.gateway.ctrl.queued == 1)
        st.fake.cfg.healthy = False
        r = await queued
        assert r.status_code == 503 and error(r)["code"] == "backend_unavailable"
        st.fake.release.set()
        await running


@pytest.mark.parametrize("occupied", [False, True])
async def test_a_body_that_arrives_after_the_backend_went_down_is_503(occupied):
    # the gate passed while the backend was ready; on_down rejected the queue while the body was still arriving
    prof = make_profile(max_concurrency=1, max_queue=2)
    async with stack(FakeConfig(hang=occupied), prof, max_queue_wait_s=0.5) as st, st.client() as c:
        running = asyncio.create_task(c.post("/v1/completions", json=COMP)) if occupied else None
        if occupied:
            await asyncio.wait_for(st.fake.started.wait(), 5)
        rest = asyncio.Event()

        async def body():
            yield b'{"model": "' + MODEL.encode()
            await rest.wait()
            yield b'", "prompt": "hello there"}'

        late = asyncio.create_task(c.post("/v1/completions", content=body()))
        await until(lambda: st.gateway.guard.pending == 1 + occupied)   # past the gate, reading the body
        st.fake.cfg.healthy = False
        await until(lambda: not st.gateway.monitor.ready)
        rest.set()
        r = await late
        assert r.status_code == 503 and error(r)["code"] == "backend_unavailable" and r.headers["retry-after"] == "5"
        assert len(st.fake.generations) == int(occupied) and st.gateway.ctrl.queued == 0
        if running:
            st.fake.release.set()
            await running


# ------------------------------------------------------------------ prompt budget --

async def test_tokenize_only_when_needed_and_under_the_permit():
    prof = make_profile(max_concurrency=1, max_queue=0, max_prompt_tokens=60)
    long_ok = {"model": MODEL, "prompt": "w " * 50}          # 100 bytes > 60, but 50 words <= 60
    too_long = {"model": MODEL, "prompt": "w " * 70}         # 70 words > 60
    async with stack(FakeConfig(), prof) as st, st.client() as c:
        r = await c.post("/v1/completions", json=long_ok)
        assert r.status_code == 200 and [s.path for s in st.fake.seen] == ["/tokenize", "/v1/completions"]
        r = await c.post("/v1/completions", json=too_long)
        assert r.status_code == 400 and error(r)["code"] == "context_length_exceeded" and error(r)["param"] == "prompt"
        assert [s.path for s in st.fake.seen][-1] == "/tokenize" and st.gateway.ctrl.in_flight == 0
        r = await c.post("/v1/completions", json={**COMP, "prompt": "short"})
        assert r.status_code == 200 and st.fake.seen[-1].path == "/v1/completions"
        assert st.fake.peak_in_flight == 1


async def test_tokenize_waits_for_the_permit():
    # 029 step 8: every HTTP request to vLLM, /tokenize included, holds a permit, so vLLM never sees more than C
    prof = make_profile(max_concurrency=1, max_queue=1, max_prompt_tokens=60)
    async with stack(FakeConfig(hang=True), prof) as st, st.client() as c:
        running = asyncio.create_task(c.post("/v1/completions", json=COMP))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        long_ok = asyncio.create_task(c.post("/v1/completions", json={"model": MODEL, "prompt": "w " * 50}))
        await until(lambda: st.gateway.ctrl.queued == 1)
        assert [s.path for s in st.fake.seen] == ["/v1/completions"]      # no /tokenize while queued
        st.fake.release.set()
        assert (await running).status_code == 200 and (await long_ok).status_code == 200
        assert [s.path for s in st.fake.seen] == ["/v1/completions", "/tokenize", "/v1/completions"]
        assert st.fake.peak_in_flight == 1


async def test_upstream_ttft_starts_at_the_upstream_send_after_tokenize():
    # upstream_ttft is the research TTFT definition (upstream send -> first content); ttft counts from arrival
    prof = make_profile(max_prompt_tokens=60)
    async with stack(FakeConfig(tokenize_delay_s=0.3), prof) as st, st.client() as c:
        r = await c.post("/v1/completions", json={"model": MODEL, "prompt": "w " * 50, "stream": True})
        assert r.status_code == 200 and [s.path for s in st.fake.seen] == ["/tokenize", "/v1/completions"]
        await until(lambda: metric(st.gateway, "capserve_upstream_ttft_seconds_count", route="completions") == 1)
        assert metric(st.gateway, "capserve_ttft_seconds_sum", route="completions") >= 0.3
        assert metric(st.gateway, "capserve_upstream_ttft_seconds_sum", route="completions") < 0.3


@pytest.mark.parametrize("status, want, code, counter, label", [
    (400, 400, "fake tokenize failure", "capserve_client_errors_total", {"code": "upstream_400"}),
    (500, 502, "upstream_unavailable", "capserve_upstream_errors_total", {"kind": "status_5xx"}),
])
async def test_a_tokenize_client_error_keeps_its_status_like_generation(status, want, code, counter, label):
    # vLLM's /tokenize answers 400 when the chat template raises (e.g. a system message after a user one): the same
    # client error the generation request would get on the byte fast path, never a retryable 502
    prof = make_profile(max_prompt_tokens=60)
    late_system = {"model": MODEL, "messages": [{"role": "user", "content": "w " * 50},
                                                {"role": "system", "content": "late"}]}
    async with stack(FakeConfig(tokenize_fail_status=status), prof) as st, st.client() as c:
        r = await c.post("/v1/chat/completions", json=late_system)
        assert r.status_code == want
        assert (error(r)["message"] if status == 400 else error(r)["code"]) == code
        assert [s.path for s in st.fake.seen] == ["/tokenize"] and st.gateway.ctrl.in_flight == 0
        await until(lambda: metric(st.gateway, counter, **label) == 1)
        if status == 400:
            assert metric(st.gateway, "capserve_upstream_errors_total", kind="protocol") is None
            assert metric(st.gateway, "capserve_requests_total", route="chat", stream="false",
                          outcome="client_error") == 1


async def test_validation_errors_never_reach_the_backend():
    async with stack() as st, st.client() as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "tools": [{"type": "function"}]})
        assert r.status_code == 400 and error(r)["code"] == "unsupported_parameter" and error(r)["param"] == "tools"
        r = await c.post("/v1/completions", json={**COMP, "prompt": ["a", "b"]})
        assert r.status_code == 400 and error(r)["param"] == "prompt"
        r = await c.post("/v1/completions", json={**COMP, "model": "gpt-4"})
        assert r.status_code == 404 and error(r)["code"] == "model_not_found"
        r = await c.post("/v1/completions", content=b"{bad")
        assert r.status_code == 400 and error(r)["code"] == "invalid_json"
        assert st.fake.seen == []
        assert metric(st.gateway, "capserve_client_errors_total", code="unsupported_parameter") == 2   # tools, batch


async def test_bodies_that_cannot_be_forwarded_are_400_not_500():
    nan = ('{"model": "%s", "prompt": "hi", "temperature": NaN}' % MODEL).encode()
    surrogate = ('{"model": "%s", "prompt": "a\\ud800b"}' % MODEL).encode()
    deep = b"[" * 100_000 + b"]" * 100_000
    async with stack() as st, st.client() as c:
        for raw in (nan, surrogate, deep):
            r = await c.post("/v1/completions", content=raw)
            assert r.status_code == 400 and error(r)["code"] in ("invalid_value", "invalid_json")
        assert st.fake.seen == [] and st.gateway.ctrl.in_flight == 0
        assert metric(st.gateway, "capserve_requests_total", route="completions", stream="unknown",
                      outcome="server_error") is None


async def test_body_limit_is_413():
    async with stack(max_body_bytes=100) as st, st.client() as c:
        r = await c.post("/v1/completions", json={**COMP, "prompt": "x" * 200})
        assert r.status_code == 413 and error(r)["code"] == "request_too_large"


# ------------------------------------------------------------------ auth, routes --

async def test_bearer_key_guards_the_v1_routes_only():
    async with stack(api_key="sé-cret".encode()) as st, st.client() as c:
        assert (await c.post("/v1/completions", json=COMP)).status_code == 401
        r = await c.post("/v1/completions", json=COMP, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401 and error(r)["code"] == "invalid_api_key"
        r = await c.post("/v1/completions", json=COMP, headers=[(b"authorization", b"Bearer \xff\xfe")])
        assert r.status_code == 401
        assert (await c.get("/v1/models")).status_code == 401
        good = {"Authorization": b"Bearer " + "sé-cret".encode()}
        r = await c.post("/v1/completions", json=COMP, headers=good)
        assert r.status_code == 200 and "authorization" not in st.fake.generations[0].headers
        for path in ("/healthz", "/readyz", "/metrics"):
            assert (await c.get(path)).status_code == 200


async def test_models_404_405_and_metrics():
    async with stack() as st, st.client() as c:
        r = await c.get("/v1/models")
        assert r.json()["data"] == [{"id": MODEL, "object": "model", "owned_by": "capserve"}]
        r = await c.get("/tokenize")
        assert r.status_code == 404 and error(r)["code"] == "not_found" and "x-request-id" in r.headers
        r = await c.get("/v1/completions")
        assert r.status_code == 405 and error(r)["code"] == "method_not_allowed"
        for path in ("/docs", "/openapi.json"):
            assert (await c.get(path)).status_code == 404
        text = (await c.get("/metrics")).text
        assert 'capserve_profile_info{active_class="test"' in text and "_created" not in text


async def test_unreachable_backend_is_502():
    settings = gateway_settings("http://127.0.0.1:9")
    gw = create_app(settings, make_profile())
    gw.monitor.ready = True     # the monitor would say otherwise; this checks the data-plane error path
    transport = httpx.ASGITransport(app=gw)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.post("/v1/completions", json=COMP)
    assert r.status_code == 502 and error(r)["code"] == "upstream_unavailable"
    assert gw.ctrl.in_flight == 0 and metric(gw, "capserve_upstream_errors_total", kind="connect") == 1
    await gw.client.aclose()
    await gw.monitor.aclose()


async def test_a_disconnect_noticed_after_the_upstream_answered_closes_it_before_releasing():
    # the watcher can report the disconnect only as the upstream block ends, with the stream already open; the
    # permit must still be released after that stream is closed, or the next request would join it in vLLM
    class Body(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            Body.closed = True

    gw = create_app(gateway_settings("http://vllm"), make_profile())
    gw.monitor.ready = True
    await gw.client.aclose()
    gw.client = httpx.AsyncClient(base_url="http://vllm",
                                  transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Body())))
    watches, released_after_close = [], []

    @contextlib.asynccontextmanager
    async def watch(request):
        watches.append(request)
        yield
        if len(watches) == 2:   # the /tokenize + upstream block
            raise ClientDisconnected

    gw._watch = watch
    release = gw.ctrl._release
    gw.ctrl._release = lambda: (released_after_close.append(Body.closed), release())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw), base_url="http://gw") as c:
        r = await c.post("/v1/completions", json={**COMP, "stream": True})
    assert r.status_code == 499 and released_after_close == [True] and gw.ctrl.in_flight == 0
    assert metric(gw, "capserve_client_disconnects_total", phase="waiting_response") == 1
    await gw.client.aclose()
    await gw.monitor.aclose()


# ------------------------------------------------------------------ local tokenizer (038 §4) --

def local_stack(prof, hf=None, **kw):
    return stack(FakeConfig(), prof, tokenizer="local", tokenizer_path="/fake-snapshot",
                 local_tokenizer=LocalTokenizer(hf or WordsHF()), **kw)


async def test_a_local_tokenizer_counts_before_admission_and_never_calls_vllm():
    prof = make_profile(max_concurrency=1, max_queue=0, max_prompt_tokens=60)
    hf = WordsHF()
    async with local_stack(prof, hf) as st, st.client() as c:
        r = await c.post("/v1/completions", json={"model": MODEL, "prompt": "w " * 50})   # 100 bytes, 50 tokens
        assert r.status_code == 200 and [s.path for s in st.fake.seen] == ["/v1/completions"] and hf.calls == 1
        r = await c.post("/v1/chat/completions", json={"model": MODEL, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "w " * 30}, {"type": "text", "text": "w " * 20}]}]})
        assert r.status_code == 200 and [s.path for s in st.fake.seen] == ["/v1/completions", "/v1/chat/completions"]
        assert st.fake.seen[-1].body["messages"] == [{"role": "user", "content": "w " * 50}]


async def test_a_local_tokenizer_rejects_too_long_prompts_without_a_permit():
    prof = make_profile(max_concurrency=1, max_queue=0, max_prompt_tokens=60)
    async with stack(FakeConfig(hang=True), prof, tokenizer="local", tokenizer_path="/fake-snapshot",
                     local_tokenizer=LocalTokenizer(WordsHF())) as st, st.client() as c:
        running = asyncio.create_task(c.post("/v1/completions", json=COMP))   # holds the only slot
        await asyncio.wait_for(st.fake.started.wait(), 5)
        r = await c.post("/v1/completions", json={"model": MODEL, "prompt": "w " * 70})
        assert r.status_code == 400 and error(r)["code"] == "context_length_exceeded" and error(r)["param"] == "prompt"
        assert [s.path for s in st.fake.seen] == ["/v1/completions"]           # no /tokenize, no second request
        st.fake.release.set()
        assert (await running).status_code == 200


async def test_a_template_the_local_tokenizer_rejects_is_400():
    prof = make_profile(max_prompt_tokens=60)
    async with local_stack(prof, WordsHF(fail=True)) as st, st.client() as c:
        r = await c.post("/v1/chat/completions", json={"model": MODEL, "messages": [
            {"role": "user", "content": "w " * 50}]})
        assert r.status_code == 400 and (error(r)["code"], error(r)["param"]) == ("invalid_value", "messages")
        assert st.fake.seen == []
        assert metric(st.gateway, "capserve_client_errors_total", code="invalid_value") == 1


def test_local_mode_needs_a_tokenizer(tmp_path):
    """create_app loads settings.tokenizer_path when no tokenizer is given; a missing snapshot is a ConfigError."""
    from capserve.config import ConfigError
    settings = gateway_settings("http://127.0.0.1:9", tokenizer="local", tokenizer_path=str(tmp_path / "none"))
    with pytest.raises(ConfigError, match="tokenizer"):
        create_app(settings, make_profile())


# ------------------------------------------------------------------ per-key quota (038 §5) --

def keyring(tmp_path):
    import hashlib
    sha = {k: hashlib.sha256(k.encode()).hexdigest() for k in ("key-a", "key-b")}
    path = tmp_path / "keys.toml"
    path.write_text(f'[[keys]]\nname = "team-a"\nsha256 = "{sha["key-a"]}"\nrequests_per_minute = 2\n'
                    f'max_in_flight = 1\n\n[[keys]]\nname = "team-b"\nsha256 = "{sha["key-b"]}"\n')
    return load_keyring(path)


A = {"Authorization": "Bearer key-a"}
B = {"Authorization": "Bearer key-b"}


async def test_each_key_has_its_own_rate_limit(tmp_path):
    async with stack(keyring=keyring(tmp_path)) as st, st.client() as c:
        assert [(await c.post("/v1/completions", json=COMP, headers=A)).status_code for _ in range(2)] == [200, 200]
        r = await c.post("/v1/completions", json=COMP, headers=A)
        assert r.status_code == 429 and (error(r)["type"], error(r)["code"]) == ("rate_limit_error", "quota_exceeded")
        assert r.headers["retry-after"] == "30" and r.headers["x-should-retry"] == "true"
        assert (await c.post("/v1/completions", json=COMP, headers=B)).status_code == 200
        for headers in ({}, {"Authorization": "Bearer key-c"}, {"Authorization": "key-a"}):
            assert (await c.post("/v1/completions", json=COMP, headers=headers)).status_code == 401
        assert (await c.get("/v1/models", headers=B)).status_code == 200
        assert (await c.get("/v1/models")).status_code == 401
        assert metric(st.gateway, "capserve_key_requests_total", key="team-a", outcome="ok") == 2
        assert metric(st.gateway, "capserve_key_requests_total", key="team-a", outcome="rejected") == 1
        assert metric(st.gateway, "capserve_key_requests_total", key="team-b", outcome="ok") == 1
        assert metric(st.gateway, "capserve_rejections_total", reason="quota_exceeded") == 1
        assert "authorization" not in st.fake.generations[0].headers


async def test_a_key_at_its_in_flight_cap_is_429_while_others_proceed(tmp_path):
    async with stack(FakeConfig(hang=True), keyring=keyring(tmp_path)) as st, st.client() as c:
        first = asyncio.create_task(c.post("/v1/completions", json=COMP, headers=A))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        r = await c.post("/v1/completions", json=COMP, headers=A)
        assert r.status_code == 429 and error(r)["code"] == "quota_exceeded" and "retry-after" in r.headers
        other = asyncio.create_task(c.post("/v1/completions", json=COMP, headers=B))
        await until(lambda: st.fake.in_flight == 2)
        st.fake.release.set()
        assert ((await first).status_code, (await other).status_code) == (200, 200)
        assert (await c.post("/v1/completions", json=COMP, headers=A)).status_code == 200   # released at the end
        assert st.gateway.keyring.in_flight("team-a") == 0


async def test_the_quota_is_checked_after_the_gate(tmp_path):
    """A request the gateway cannot serve anyway (backend down) costs no quota."""
    async with stack(FakeConfig(healthy=False), keyring=keyring(tmp_path), wait_ready=False) as st, st.client() as c:
        for _ in range(3):
            r = await c.post("/v1/completions", json=COMP, headers=A)
            assert r.status_code == 503 and error(r)["code"] == "backend_unavailable"
