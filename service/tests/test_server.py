"""Tests for capserve/server.py: the uvicorn configuration (h11 / asyncio pinned, one process, drain timeout, no
limit_concurrency) and handle_exit setting draining before uvicorn's own shutdown (decision 029 "Drain")."""

import asyncio
import signal

import pytest

from capserve import server as srv
from capserve.app import create_app
from fakes.servers import gateway_settings, make_profile


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
