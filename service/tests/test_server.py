"""Tests for capserve/server.py: the uvicorn configuration (h11 / asyncio pinned, one process, drain timeout, no
limit_concurrency, the client keep-alive of decision 043 §7.2) and handle_exit setting draining before uvicorn's own
shutdown (decision 029 "Drain")."""

import asyncio
import json
import re
import signal
import urllib.parse

import pytest

from capserve import server as srv
from capserve.app import create_app
from fakes.fake_vllm import MODEL
from fakes.servers import gateway_settings, make_profile, stack

IDLE_S = 6.0   # 043 step 8: longer than uvicorn's default keep-alive (5 s), what build_config used before 043
BODY = json.dumps({"model": MODEL, "prompt": "hello there"}).encode()
COMPLETION = (b"POST /v1/completions HTTP/1.1\r\nhost: gw\r\ncontent-type: application/json\r\n"
              b"content-length: %d\r\n\r\n%s" % (len(BODY), BODY))


def gateway(**over):
    prof = make_profile(max_concurrency=4, max_queue=6)
    settings = gateway_settings("http://127.0.0.1:9", port=8080, drain_timeout_s=45, pre_admission_slots=16, **over)
    return create_app(settings, prof), settings


def test_config_pins_the_protocol_stack_and_one_process(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "4")     # uvicorn would read workers from it
    gw, settings = gateway()
    c = srv.build_config(gw, settings)
    assert (c.http, c.loop, c.ws, c.lifespan) == ("h11", "asyncio", "none", "on")
    assert (c.workers, c.reload, c.use_subprocess) == (1, False, False)
    assert (c.host, c.port) == ("127.0.0.1", 8080)
    assert c.timeout_graceful_shutdown == 45
    assert c.limit_concurrency is None   # uvicorn would count connections, not requests: the guard is the limit
    assert c.access_log is False and c.app is gw
    assert c.timeout_keep_alive == 75    # 043 §7.2: Settings.keep_alive_s, not uvicorn's 5 s


def test_keep_alive_s_reaches_uvicorn():
    gw, settings = gateway(keep_alive_s=120)
    assert srv.build_config(gw, settings).timeout_keep_alive == 120


@pytest.mark.anyio
async def test_handle_exit_drains_on_the_loop_then_exits():
    gw, settings = gateway()
    server = srv.CapserveServer(srv.build_config(gw, settings), gw)
    gw.loop = asyncio.get_running_loop()
    server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit and server._captured_signals == [signal.SIGTERM]
    await asyncio.sleep(0)
    assert gw.draining and gw.ctrl.draining and gw.readiness()["ready"] is False


def test_handle_exit_before_startup_only_exits():
    gw, settings = gateway()
    server = srv.CapserveServer(srv.build_config(gw, settings), gw)
    server.handle_exit(signal.SIGINT, None)
    assert server.should_exit and not gw.draining


def test_run_refuses_a_public_bind_without_a_key():
    gw, settings = gateway()
    public = settings.model_copy(update={"host": "0.0.0.0"})
    with pytest.raises(srv.ConfigError):
        srv.run(public, make_profile(), api_key=None)


def test_tls_settings_reach_uvicorn(tmp_path):
    gw, settings = gateway(tls={"certfile": "/c.pem", "keyfile": "/k.pem"})
    c = srv.build_config(gw, settings)
    assert (c.ssl_certfile, c.ssl_keyfile) == ("/c.pem", "/k.pem")
    gw, settings = gateway()
    c = srv.build_config(gw, settings)
    assert (c.ssl_certfile, c.ssl_keyfile) == (None, None)


async def exchange(reader, writer) -> bytes | None:
    """One completion request on this very connection: the response's status line, or None when the gateway has
    closed the connection (what a pooled client reusing it meets: D-B's failure before the first token, 042)."""
    writer.write(COMPLETION)
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    await asyncio.wait_for(reader.readexactly(int(re.search(rb"(?i)\r\ncontent-length: *(\d+)", head)[1])), 5)
    return head.split(b"\r\n", 1)[0]


@pytest.mark.anyio
@pytest.mark.parametrize("over, reused", [({}, True), ({"keep_alive_s": 5}, False)])
async def test_a_client_connection_idle_longer_than_uvicorns_default_is_reused(monkeypatch, over, reused):
    """043 §7.2: a client connection idle for IDLE_S (longer than uvicorn's default 5 s, which aiohttp's FIFO pool of
    60 s and httpx's of 5 s race) is still open with the default keep_alive_s, and a completion on it completes
    normally; with keep_alive_s = 5 (uvicorn's default, before 043) the gateway has closed it.

    The idle time is the event loop's clock jumping IDLE_S ahead: uvicorn closes an idle connection with
    loop.call_later(timeout_keep_alive), so every timer due within the jump fires at once and the test takes no wall
    time. The fake backend's own 5 s keep-alive and the monitor's probe timeout fire too; the monitor's threshold is
    raised so that neither turns the backend not-ready (they are not under test)."""
    loop = asyncio.get_running_loop()
    real, skew = loop.time, [0.0]
    monkeypatch.setattr(loop, "time", lambda: real() + skew[0])
    async with stack(monitor_fail_threshold=1000, **over) as st:
        url = urllib.parse.urlsplit(st.url)
        reader, writer = await asyncio.open_connection(url.hostname, url.port)
        try:
            assert await exchange(reader, writer) == b"HTTP/1.1 200 OK"
            skew[0] += IDLE_S
            await asyncio.sleep(0.05)   # the due timers fire, and a close reaches this socket
            assert await exchange(reader, writer) == (b"HTTP/1.1 200 OK" if reused else None)
        finally:
            writer.close()
        assert len(st.fake.generations) == (2 if reused else 1)
