"""Product processes: start them, give them an isolated environment, find the GPU-lock descriptor (decision 029
"部署管理" step 4; design review findings 5 and 6).

start_group (vLLM, the gateway, the itest's supervisor):
- start_new_session: each child leads its own process group, so teardown signals the whole group (EngineCore too);
- pass_fds=keep_fds with close_fds: the shared GPU-lock descriptor that flock(1) handed to the supervisor is kept
  open in vLLM and the gateway (and nothing else is), so the lock stays held while they live even if the
  supervisor is SIGKILLed or OOM-killed. Without it the lock would be free while ~14.6 GB of vLLM is still on the
  GPU, and another project that checks only the lock could start;
- PR_SET_PDEATHSIG(SIGTERM) on Linux (prctl through ctypes, resolved in the parent before the fork): a dead
  supervisor's children are told to shut down instead of holding the GPU and the lock indefinitely; if the parent
  died before prctl took effect the child signals itself. A no-op elsewhere (macOS has no prctl);
- stdout and stderr go to a new log file; an existing log (earlier evidence) is never overwritten.

vllm_env: the research server env (core_b_plan.server_env, the pinned pin-memory value as server_argv picks it) on
top of a scrubbed base: PYTHONPATH and PYTHONDONTWRITEBYTECODE (the supervisor's, pointing into the release),
CAPSERVE_* (the API key never reaches vLLM) and the variables that relocate caches (XDG_*_HOME, TRITON_CACHE_DIR,
TORCHINDUCTOR_CACHE_DIR, FLASHINFER_WORKSPACE_BASE) are dropped, then HOME=<service_root>/.home and
TMPDIR=<service_root>/.tmp, so product triton / flashinfer / vllm / inductor caches never land in the research's
~/.triton, ~/.cache/{vllm,flashinfer} or /tmp (a C0 / C1 confounder; checked by RTX IT-6). HF_HOME stays the shared,
read-only model cache (HF_HUB_OFFLINE=1). HOME and TMPDIR are not part of core_b_plan.recorded_env, so the server
fingerprint equals the research server's.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import os
import platform
import signal
import socket
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Sequence

import core_b_plan as bp
import coreb_smoke as smoke

PR_SET_PDEATHSIG = 1   # <linux/prctl.h>
SCRUBBED = ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
            "XDG_STATE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE")
SCRUBBED_PREFIXES = ("CAPSERVE_",)
_DEFAULT = object()


# ---------------------------------------------------------------- starting a process group

def _libc_prctl() -> Callable:
    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True).prctl


def pdeathsig_preexec(*, system: str | None = None, prctl: Callable | None = None, sig=signal.SIGTERM,
                      getpid: Callable = os.getpid, getppid: Callable = os.getppid,
                      kill: Callable = os.kill) -> Callable | None:
    """A preexec_fn that sets PR_SET_PDEATHSIG(sig) in the child (Linux only; None elsewhere). Everything it needs is
    resolved here, in the parent: after the fork only prctl, getppid and kill run."""
    if (platform.system() if system is None else system) != "Linux":
        return None
    prctl = prctl or _libc_prctl()
    parent = getpid()

    def preexec() -> None:
        if prctl(PR_SET_PDEATHSIG, int(sig), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if getppid() != parent:   # the parent died between fork and prctl: nobody would ever signal us
            kill(getpid(), sig)

    return preexec


def start_group(argv: Sequence[str], env: Mapping[str, str], log_path, *, cwd, keep_fds: Sequence[int] = (),
                popen: Callable = subprocess.Popen, preexec=_DEFAULT):
    """Popen in a new session with only keep_fds inherited, PDEATHSIG (Linux) and output into a new log_path."""
    if preexec is _DEFAULT:
        preexec = pdeathsig_preexec()
    log = open(log_path, "x")
    try:
        proc = popen(list(argv), env=dict(env), cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=log,
                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True, pass_fds=tuple(keep_fds),
                     preexec_fn=preexec)
    finally:
        log.close()   # the child has its own descriptor
    proc.log_path = Path(log_path)
    return proc


# ---------------------------------------------------------------- environments

def scrubbed(base: Mapping[str, str]) -> dict:
    return {k: v for k, v in base.items() if k not in SCRUBBED and not k.startswith(SCRUBBED_PREFIXES)}


def home_dir(cfg) -> Path:
    return Path(cfg.service_root) / ".home"


def tmp_dir(cfg) -> Path:
    return Path(cfg.service_root) / ".tmp"


def vllm_env(base: Mapping[str, str], cfg, *, pins: dict | None = None) -> dict:
    pins = bp.load_pins() if pins is None else pins
    pin_memory = (pins.get("stage_b") or {}).get("wsl2_pin_memory") or "0"   # exactly as core_b_plan.server_argv
    env = bp.server_env(scrubbed(base), pin_memory=pin_memory)
    return {**env, "HOME": str(home_dir(cfg)), "TMPDIR": str(tmp_dir(cfg))}


def gateway_env(base: Mapping[str, str], cfg, *, release_root: Path) -> dict:
    """The gateway imports capserve and the read-only research modules from the release; it keeps
    CAPSERVE_API_KEY (its only secret), writes no bytecode into the release and never uses the real HOME."""
    keep = {k: v for k, v in base.items() if k not in SCRUBBED}
    root = Path(release_root)
    return {**keep, "PYTHONPATH": os.pathsep.join([str(root / "service"), str(root / "benchmarks")]),
            "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(home_dir(cfg)), "TMPDIR": str(tmp_dir(cfg)),
            "HF_HUB_OFFLINE": "1"}   # 038 §4: a local tokenizer loads from the snapshot, never the Hub


# ---------------------------------------------------------------- the GPU lock, ports

def lock_fd(lock_path: str = smoke.GPU_LOCK, fd_dir: str = "/proc/self/fd") -> int | None:
    """The descriptor through which this process holds the shared GPU lock (run_adv.held_gpu_lock's rule, which
    only answers yes / no): it resolves to the lock file and flock(LOCK_EX | LOCK_NB) on it succeeds, because its
    open file description already holds the lock. None when there is no such descriptor (or no fd_dir: macOS)."""
    want = os.path.realpath(lock_path)
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return None
    for name in sorted(names, key=lambda n: (len(n), n)):
        if not name.isdigit():
            continue
        try:
            if os.path.realpath(os.path.join(fd_dir, name)) != want:
                continue
            fcntl.flock(int(name), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return int(name)
        except OSError:   # EWOULDBLOCK (held through another description) or closed meanwhile
            continue
    return None


def fds_on(pid: int, path: str, *, proc_root: str = "/proc") -> list:
    """The descriptors of process pid that point at path (RTX IT-1: vLLM holds the lock file open)."""
    want = os.path.realpath(path)
    fd_dir = os.path.join(proc_root, str(pid), "fd")
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return []
    out = []
    for name in names:
        try:
            if name.isdigit() and os.path.realpath(os.path.join(fd_dir, name)) == want:
                out.append(int(name))
        except OSError:
            continue
    return sorted(out)


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Whether a listener could bind host:port now (SO_REUSEADDR like uvicorn and vLLM, so TIME_WAIT is free)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True
