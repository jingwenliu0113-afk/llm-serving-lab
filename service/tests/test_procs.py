"""Tests for capserve/procs.py (decision 029 "部署管理" step 4; review findings 5 and 6): product processes start in
their own session with the GPU-lock descriptor passed on (pass_fds) and PR_SET_PDEATHSIG on Linux, so the shared lock
stays held while vLLM lives even if the supervisor is SIGKILLed; vLLM's env is scrubbed and gets its own HOME / TMPDIR
so product JIT / autotune caches never land in the research caches."""

import fcntl
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import core_b_plan as bp
from capserve import procs

PINS = bp.load_pins()


class FakePopen:
    def __init__(self, argv, **kw):
        self.argv, self.kw = argv, kw
        self.pid = 4242


def cfg(tmp_path):
    return SimpleNamespace(service_root=str(tmp_path / "svc"))


# ------------------------------------------------------------------ start_group --

def test_start_group_new_session_passes_only_the_kept_fds_and_logs_to_a_new_file(tmp_path):
    log = tmp_path / "vllm.log"
    proc = procs.start_group(["vllm", "serve", "m"], {"A": "1"}, log, cwd=tmp_path, keep_fds=(7,),
                             popen=FakePopen, preexec=None)
    kw = proc.kw
    assert proc.argv == ["vllm", "serve", "m"] and kw["env"] == {"A": "1"} and kw["cwd"] == str(tmp_path)
    assert kw["start_new_session"] is True and kw["close_fds"] is True and kw["pass_fds"] == (7,)
    assert kw["stdin"] is subprocess.DEVNULL and kw["stderr"] is subprocess.STDOUT
    assert kw["stdout"].name == str(log) and kw["stdout"].closed   # the child has its own copy
    assert kw["preexec_fn"] is None and proc.log_path == log and log.exists()


def test_start_group_never_overwrites_a_log(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("earlier evidence\n")
    with pytest.raises(FileExistsError):
        procs.start_group(["x"], {}, log, cwd=tmp_path, popen=FakePopen, preexec=None)
    assert log.read_text() == "earlier evidence\n"


def test_start_group_closes_the_log_when_popen_fails(tmp_path):
    seen = {}

    def boom(argv, **kw):
        seen["log"] = kw["stdout"]
        raise OSError("no such binary")

    with pytest.raises(OSError):
        procs.start_group(["x"], {}, tmp_path / "l.log", cwd=tmp_path, popen=boom, preexec=None)
    assert seen["log"].closed


def test_start_group_uses_the_pdeathsig_preexec_by_default(tmp_path, monkeypatch):
    marker = object()
    monkeypatch.setattr(procs, "pdeathsig_preexec", lambda: marker)
    proc = procs.start_group(["x"], {}, tmp_path / "l.log", cwd=tmp_path, popen=FakePopen)
    assert proc.kw["preexec_fn"] is marker


def test_a_real_child_inherits_exactly_the_kept_descriptor(tmp_path):
    lock = tmp_path / "gpu.lock"
    kept = os.open(lock, os.O_RDWR | os.O_CREAT)
    other = os.open(tmp_path / "other", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(kept, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = ("import os, sys, fcntl\n"
                f"os.fstat({kept})\n"
                f"fcntl.flock({kept}, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"   # same open file description: still ours
                f"try:\n    os.fstat({other})\n    sys.exit(3)\nexcept OSError:\n    pass\n"
                "print(os.getpgid(0) == os.getpid())\n")
        proc = procs.start_group([sys.executable, "-c", code], dict(os.environ), tmp_path / "child.log", cwd=tmp_path,
                                 keep_fds=(kept,), preexec=None)
        assert proc.wait(timeout=60) == 0
        assert (tmp_path / "child.log").read_text().strip() == "True"   # its own session / process group
    finally:
        os.close(kept)
        os.close(other)


# ------------------------------------------------------------------ PR_SET_PDEATHSIG --

def test_pdeathsig_is_a_no_op_off_linux():
    assert procs.pdeathsig_preexec(system="Darwin", prctl=lambda *a: 0) is None


def test_pdeathsig_on_linux_sets_sigterm_and_checks_the_parent():
    calls = []
    pre = procs.pdeathsig_preexec(system="Linux", prctl=lambda *a: calls.append(a) or 0, getpid=lambda: 100,
                                  getppid=lambda: 100, kill=lambda *a: calls.append(("kill",) + a))
    pre()
    assert calls == [(procs.PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)]


def test_pdeathsig_kills_the_child_when_the_parent_died_before_prctl():
    calls, pids = [], iter([100, 555])   # parent pid at creation, then the child's own pid
    pre = procs.pdeathsig_preexec(system="Linux", prctl=lambda *a: 0, getpid=lambda: next(pids),
                                  getppid=lambda: 1, kill=lambda *a: calls.append(a))
    pre()
    assert calls == [(555, signal.SIGTERM)]


def test_pdeathsig_failure_raises_so_the_start_fails():
    pre = procs.pdeathsig_preexec(system="Linux", prctl=lambda *a: -1, getpid=lambda: 1, getppid=lambda: 1,
                                  kill=lambda *a: None)
    with pytest.raises(OSError, match="PR_SET_PDEATHSIG"):
        pre()


# ------------------------------------------------------------------ environments --

def test_vllm_env_scrubs_python_capserve_and_cache_variables_then_applies_core_b(tmp_path):
    base = {"PATH": "/usr/bin", "PYTHONPATH": "service:benchmarks", "PYTHONDONTWRITEBYTECODE": "1",
            "CAPSERVE_API_KEY": "s3cr3t", "CAPSERVE_ALLOW_PROVISIONAL": "1", "VLLM_LOGGING_LEVEL": "DEBUG",
            "CUDA_VISIBLE_DEVICES": "1", "XDG_CACHE_HOME": "/home/u/.cache", "XDG_CONFIG_HOME": "/home/u/.config",
            "TRITON_CACHE_DIR": "/home/u/.triton", "TORCHINDUCTOR_CACHE_DIR": "/tmp/ti", "HOME": "/home/u",
            "TMPDIR": "/tmp", "LANG": "C.UTF-8"}
    env = procs.vllm_env(base, cfg(tmp_path), pins=PINS)
    svc = tmp_path / "svc"
    assert env["HOME"] == str(svc / ".home") and env["TMPDIR"] == str(svc / ".tmp")
    for gone in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "CAPSERVE_API_KEY", "CAPSERVE_ALLOW_PROVISIONAL",
                 "VLLM_LOGGING_LEVEL", "CUDA_VISIBLE_DEVICES", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
                 "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        assert gone not in env, gone
    assert "s3cr3t" not in repr(env)
    core = bp.server_env({"PATH": "/usr/bin", "HOME": "/home/u", "TMPDIR": "/tmp", "LANG": "C.UTF-8"},
                         pin_memory=PINS["stage_b"]["wsl2_pin_memory"])
    assert env == {**core, "HOME": str(svc / ".home"), "TMPDIR": str(svc / ".tmp")}
    assert env["VLLM_WSL2_ENABLE_PIN_MEMORY"] == PINS["stage_b"]["wsl2_pin_memory"] == "0"


def test_vllm_env_records_like_the_research_server():
    """recorded_env (and so the fingerprint) equals the Stage B server's: HOME / TMPDIR are not recorded."""
    env = procs.vllm_env({"PATH": "/usr/bin"}, SimpleNamespace(service_root="/srv/x"), pins=PINS)
    _, research_env = bp.server_argv(PINS, gdn_prefill_backend="triton", gpu_memory_utilization="0.91", port=8000)
    assert bp.recorded_env(env) == bp.recorded_env(research_env)


def test_vllm_env_pin_memory_follows_pins_like_server_argv():
    pins = {**PINS, "stage_b": {**PINS["stage_b"], "wsl2_pin_memory": "1"}}
    assert procs.vllm_env({}, SimpleNamespace(service_root="/s"), pins=pins)["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"
    pins = {k: v for k, v in PINS.items() if k != "stage_b"}
    assert procs.vllm_env({}, SimpleNamespace(service_root="/s"), pins=pins)["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "0"


def test_gateway_env_imports_the_release_and_keeps_the_api_key(tmp_path):
    base = {"PATH": "/usr/bin", "PYTHONPATH": "/elsewhere", "CAPSERVE_API_KEY": "k", "XDG_CACHE_HOME": "/home/u/.c"}
    env = procs.gateway_env(base, cfg(tmp_path), release_root=Path("/rel/abc"))
    assert env["PYTHONPATH"] == os.pathsep.join(["/rel/abc/service", "/rel/abc/benchmarks"])
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["CAPSERVE_API_KEY"] == "k"
    assert env["HOME"] == str(tmp_path / "svc" / ".home") and env["TMPDIR"] == str(tmp_path / "svc" / ".tmp")
    assert "XDG_CACHE_HOME" not in env and env["PATH"] == "/usr/bin"


# ------------------------------------------------------------------ lock descriptors, ports --

def fake_fd_dir(tmp_path, links: dict) -> Path:
    d = tmp_path / "fd"
    d.mkdir()
    for name, target in links.items():
        os.symlink(target, d / name)
    return d


def test_lock_fd_finds_the_descriptor_holding_the_lock(tmp_path):
    lock = tmp_path / "gpu.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        d = fake_fd_dir(tmp_path, {"0": "/dev/null", str(fd): lock, "notanfd": lock})
        assert procs.lock_fd(str(lock), fd_dir=str(d)) == fd
        assert procs.lock_fd(str(tmp_path / "other.lock"), fd_dir=str(d)) is None
    finally:
        os.close(fd)


def test_lock_fd_refuses_a_descriptor_whose_lock_is_held_elsewhere(tmp_path):
    lock = tmp_path / "gpu.lock"
    holder = os.open(lock, os.O_RDWR | os.O_CREAT)
    mine = os.open(lock, os.O_RDWR)   # another open file description of the same file
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        d = fake_fd_dir(tmp_path, {str(mine): lock})
        assert procs.lock_fd(str(lock), fd_dir=str(d)) is None
    finally:
        os.close(holder)
        os.close(mine)


def test_lock_fd_without_a_proc_fd_dir_is_none(tmp_path):
    assert procs.lock_fd(str(tmp_path / "l"), fd_dir=str(tmp_path / "missing")) is None


def test_fds_on_lists_a_process_descriptors_that_point_at_a_path(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("")
    proc = tmp_path / "proc"
    (proc / "77" / "fd").mkdir(parents=True)
    os.symlink(lock, proc / "77" / "fd" / "9")
    os.symlink("/dev/null", proc / "77" / "fd" / "1")
    assert procs.fds_on(77, str(lock), proc_root=str(proc)) == [9]
    assert procs.fds_on(78, str(lock), proc_root=str(proc)) == []


def test_port_free():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        assert procs.port_free(port) is False
    finally:
        s.close()
    assert procs.port_free(port) is True
