"""python -m capserve.deploy render | up | status | stop   (decision 029 "部署管理"; design review findings 5, 6, 15-17)

render (works on the Mac): the vLLM argv (core_b_plan.server_argv with the Stage B values in pins, the research
venv's vllm binary, port 8100), its core_b_plan.recorded_env and the server fingerprint (capserve.profile).

up: a foreground supervisor (WSL kills detached jobs), run from a release directory under the GPU lock:
    cd ~/llm-serving-service/releases/<sha> && flock -n /tmp/ai-portfolio-gpu.lock env PYTHONDONTWRITEBYTECODE=1 \\
        PYTHONPATH=service:benchmarks ~/llm-serving-lab/.venv/bin/python -m capserve.deploy up \\
        --config service/deploy/rtx.toml --profile service/profiles/provisional-rtx.json --allow-provisional \\
        [--loadtest smoke|steady|burst|overload]
 1. the release: DEPLOYED_COMMIT names its directory releases/<sha> under service_root (never the research
    checkout) and every file matches DEPLOY_MANIFEST.json; pins with measured Stage B values; the profile, read once
    (a provisional one only when allowed; a verified one checked by the loader against the release's frozen configs
    (capserve.profile: [freeze] capacity, the capacity / pins sha256, the SLO, the session ids, the Stage C classes)
    and served only as the release's service/profiles/<name>.json listed in DEPLOY_MANIFEST.json with those bytes,
    i.e. committed, so the Mac test that regenerates every committed verified profile covers its numbers; 027 §4);
 2. preflight, every problem reported and nothing started (when this process holds the lock, state.json says phase
    preflight meanwhile, so stop finds it during the 30 s idle window, and refused afterwards; without the lock it
    never writes state.json, a running supervisor's stays): this process holds the shared GPU lock
    (run_adv.held_gpu_lock verifies the descriptor, it does not trust a flag) through a descriptor it can pass on;
    the GPU idle by decision 031's product rule (the research's helpers, benchmarks/gpu_idle.py, read-only): one
    window of 30 nvidia-smi samples at 1 Hz with utilization.gpu <= 5 % in every sample, no compute process and a
    memory.used range <= 64 MiB; its idle value (the largest memory.used) is the run's baseline. A baseline above
    the headroom bound (pins [stage_b].v_total_mib - F_peak 15157 - margin 128 = 1018 MiB) is only a warning in
    meta.json and on stderr: product tests check function, not latency (031 「產品的最終規格」);
    core_b_plan.toolchain_problems empty; ports 8100 and 8080 free; the run dir allowed by runs.py; the profile's
    fingerprint and KV geometry (if it states them; a verified profile always does) equal to this server's; the
    active class in the profile, servable and serviceable (030 8.7: never an unavailable class, nor a verified 0,
    whose gateway would answer every request 503 class_not_serviceable and never be ready);
 3. the pre-boot shared usage (below); runs/pt-<UTC>[-<scenario>]/meta.json (purpose product-test, formal false,
    research_valid false, the preflight's idle window and baseline, its warnings), the profile bytes preflight
    checked as runs/<id>/profile.json (the gateway serves that copy, never the mutable path) and
    <service_root>/state.json (phase preflight -> starting -> ready -> stopping -> stopped; status and stop read
    it; a stop before vLLM starts ends the supervisor with phase stopped and nothing started);
 4. vLLM: procs.start_group with keep_fds=(lock fd,) and PDEATHSIG, env procs.vllm_env, log runs/<id>/vllm.log;
    wait for /health (startup_timeout_s, cut to what is left of max_hold_s); then the server must be the Core B
    server, else teardown and exit 2:
    /metrics cache_config num_gpu_blocks / block_size = Stage B, and the log facts FLASH_ATTN, the pinned GDN
    prefill kernel and max_num_batched_tokens 2048 in the non-default args (as run_core_b / run_adv check them);
 5. the gateway: python -m capserve serve with runs/<id>/gateway.toml (rtx.toml [gateway] plus the deploy-owned
    backend_url, host 127.0.0.1, port and drain_timeout_s) and runs/<id>/profile.json, the same keep_fds; wait for
    /readyz (at most GATEWAY_READY_TIMEOUT_S, cut to what is left of max_hold_s), which must name the profile id,
    status, active class and limits preflight checked, else teardown and exit 2;
 6. the workload: the capserve.loadtest scenario (a fixed module, never an arbitrary command), or hold until a
    signal, max_hold_s after the start or the exit of either child. max_hold_s (finite, <= 1 h: 029 allows no
    resident daemon) bounds how long the shared lock is held before the teardown starts, the preflight's idle
    window and the boot included; the teardown then adds at most drain_timeout_s (<= MAX_DRAIN_S) plus the kill
    waits, the after-workload reading and the idle wait;
 7. teardown, in a finally under run_adv.DeferredSignals (a second Ctrl-C cannot orphan a GPU process group), and
    always: state.json / meta.json writes and stderr around it are best effort (a full disk or a permission error
    becomes a problem, never a skipped teardown or a replaced interrupt). SIGTERM the gateway group (it drains),
    SIGKILL after drain_timeout_s + 10 s; the after-workload shared usage; run_adv.shutdown_group on the vLLM group
    (SIGINT, SIGTERM, SIGKILL); GPU memory.used back <= the run's pre-boot baseline + 64 MiB within 60 s (031; the
    fixed run_core_b.IDLE_MIB of Phase 1 is not the product's rule any more); problems and the outcome go into
    meta.json (its start mtime kept for IT-6), then an interrupt goes on.
Shared usage (031, descriptive only: no spill verdict for product tests): the Windows shared usage of the NVIDIA
adapter (gpu_idle.read_shared_usage, through powershell.exe, <= 20 s) before vLLM starts, once vLLM is the verified
Core B server, and after the workload (the gateway stopped, vLLM still holding its memory), each with its change
since the pre-boot reading, in meta.json shared_usage; a reading that fails is recorded with its error, never fatal.
Exit status: 0 ok, 1 a problem after the start (workload, readiness, teardown, bookkeeping), 2 refused, not the
Core B server or a gateway serving another profile than preflight checked, 128 + signal when stopped by SIGTERM /
SIGHUP (coreb_smoke._exit_on_signal).

status: state.json plus whether the supervisor is alive and the endpoints answer. stop: SIGTERM to the supervisor,
only after /proc/<pid>/cmdline shows `capserve.deploy up` (a reused pid is never signalled), then wait for it.

The product never writes ~/llm-serving-lab: research modules are imported read-only from the release, vLLM's caches
live under <service_root>/.home and .tmp (procs.vllm_env), outputs under <service_root>/runs. Port 8100 is not
research's 8000. Changing any vLLM flag changes the fingerprint and needs its own decision (029).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import core_b_plan as bp
import coreb_smoke as smoke
import gpu_idle as gi
import run_adv
import run_core_b
from capserve import loadtest, procs, runs
from capserve.config import API_KEY_ENV, Settings, allow_provisional
from capserve.profile import FrozenConfigs, ProfileError, parse_profile, server_fingerprint

RESEARCH_PORT = run_core_b.PORT
RELEASE_ROOT = runs.REPO_ROOT          # the release this module runs from: releases/<sha>/ on WSL
DEFAULT_CONFIG = RELEASE_ROOT / "service" / "deploy" / "rtx.toml"
STATE_FILE = "state.json"
OWNED_GATEWAY_KEYS = ("backend_url", "host", "port", "drain_timeout_s")
LOOPBACK = "127.0.0.1"
HTTP_TIMEOUT_S = 5.0
HEALTH_POLL_S = 2.0
GATEWAY_READY_TIMEOUT_S = 120.0
HOLD_POLL_S = 1.0
GATEWAY_KILL_EXTRA_S = 10.0   # 029: SIGKILL the gateway drain_timeout_s + 10 s after SIGTERM
KILL_WAIT_S = 10.0
IDLE_RETURN_S = 60.0          # 029 / 031: GPU memory back within 60 s
IDLE_RETURN_TOL_MIB = 64      # 031: ... to <= the run's pre-boot baseline + 64 MiB (replaces run_core_b.IDLE_MIB)
# 031 addendum: above this idle value the resident memory is not the desktop (the heaviest desktop baseline seen was
# 1,396 MiB, 2026-10-16) but likely another project's model kept in VRAM at 0 % utilization, which WSL may not list as
# a compute process: a product boot never starts on top of it (不影響其他專案). Between the 031 headroom bound and this
# cap a product test only warns.
PRODUCT_IDLE_CAP_MIB = 2048
IDLE_POLL_S = 2.0
IDLE_WINDOW_S = gi.WINDOW_SAMPLES * gi.WINDOW_INTERVAL_S   # 031: the preflight's idle window (30 s)
# 031: what up() spends before it is ready besides its boot waits: the idle window, the pre-boot and ready readings
READY_OVERHEAD_S = IDLE_WINDOW_S + 2 * gi.SHARED_USAGE_TIMEOUT_S
MAX_HOLD_S = 3600.0           # 029: product tests at most 1 h per session; longer is a resident daemon (user decision)
MAX_DRAIN_S = 300             # a drain longer than the gateway's default request_deadline_s (300 s) only holds the lock
PROFILE_COPY = "profile.json"
SERVED_LIMITS = ("max_concurrency", "max_queue", "max_prompt_tokens", "max_output_tokens")
_SHA = re.compile(r"[0-9a-f]{40}")
# the only place a served verified profile lives in a release: a visible file at the top level of service/profiles
# with a lowercase .json suffix (the Mac regeneration test reads it, with every other tracked verified document; 029)
VERIFIED_PROFILE_PATH = re.compile(r"service/profiles/[^/.][^/]*\.json")


class DeployError(ValueError):
    pass


# ---------------------------------------------------------------- configuration

class DeployConfig(BaseModel):
    """service/deploy/rtx.toml: [deploy] (these fields) and [gateway] (capserve Settings minus the owned keys)."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    service_root: str = "/home/wenn0113/llm-serving-service"   # releases/<sha>/, runs/, state.json, .home/, .tmp/
    research_dir: str = bp.PROJECT_DIR                         # read-only: its venv and CUDA_HOME shim
    vllm_port: int = Field(8100, ge=1, le=65535)
    gateway_port: int = Field(8080, ge=1, le=65535)
    # Stage B saw a 164 s cold boot; the first product boot is cold too (own HOME). max_hold_s (finite, <= 1 h)
    # bounds how long the shared lock is held before the teardown starts, the boot included.
    startup_timeout_s: float = Field(900.0, gt=0, allow_inf_nan=False)
    max_hold_s: float = Field(3600.0, gt=0, le=MAX_HOLD_S, allow_inf_nan=False)
    drain_timeout_s: int = Field(60, ge=0, le=MAX_DRAIN_S)
    # 036: a production config pins the one verified profile it serves (release-relative service/profiles/<name>.json);
    # `up` then refuses any other --profile and any run with provisional profiles allowed. None: an integration config.
    production_profile: str | None = None
    gateway: dict = Field(default_factory=dict)

    @property
    def vllm_url(self) -> str:
        return f"http://{LOOPBACK}:{self.vllm_port}"

    @property
    def gateway_url(self) -> str:
        return f"http://{LOOPBACK}:{self.gateway_port}"

    @property
    def vllm_bin(self) -> str:
        return f"{self.research_dir}/.venv/bin/vllm"

    @classmethod
    def checked(cls, doc: dict) -> "DeployConfig":
        try:
            cfg = cls.model_validate(doc)
        except ValidationError as exc:
            raise DeployError(str(exc)) from exc
        problems = cfg.problems()
        if problems:
            raise DeployError("; ".join(problems))
        return cfg

    def problems(self) -> list:
        problems = []
        if self.research_dir != bp.PROJECT_DIR:
            problems.append(f"research_dir {self.research_dir!r} != {bp.PROJECT_DIR!r}: the Core B server env "
                            "(PATH, CUDA_HOME) points into core_b_plan.PROJECT_DIR")
        if self.vllm_port == RESEARCH_PORT:
            problems.append(f"vllm_port {self.vllm_port} is the research server's port {RESEARCH_PORT}")
        if self.vllm_port == self.gateway_port:
            problems.append("vllm_port and gateway_port must differ")
        svc, research = Path(self.service_root).resolve(), Path(self.research_dir).resolve()
        if svc == research or svc.is_relative_to(research) or research.is_relative_to(svc):
            problems.append(f"service_root {svc} overlaps the research directory {research}")
        if self.production_profile is not None and not VERIFIED_PROFILE_PATH.fullmatch(self.production_profile):
            problems.append(f"production_profile {self.production_profile!r} is not service/profiles/<name>.json, the "
                            "only place a served verified profile lives in a release (036)")
        owned = sorted(set(self.gateway) & set(OWNED_GATEWAY_KEYS))
        if owned:
            problems.append(f"[gateway] must not set {owned}: the deploy owns them")
        else:
            try:
                Settings.model_validate(gateway_settings(self))
            except ValidationError as exc:
                problems.append(f"[gateway]: {exc}")
        return problems


def load_config(path) -> DeployConfig:
    try:
        data = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DeployError(f"{path}: cannot read the deploy config: {exc}") from exc
    unknown = sorted(set(data) - {"deploy", "gateway"})
    try:
        if unknown:
            raise DeployError(f"unknown tables {unknown} (only [deploy] and [gateway])")
        return DeployConfig.checked({**data.get("deploy", {}), "gateway": data.get("gateway", {})})
    except DeployError as exc:
        raise DeployError(f"{path}: {exc}") from exc


def gateway_settings(cfg: DeployConfig) -> dict:
    """The gateway's settings: rtx.toml [gateway] plus what the deploy owns (loopback only: no API key needed)."""
    return {**cfg.gateway, "backend_url": cfg.vllm_url, "host": LOOPBACK, "port": cfg.gateway_port,
            "drain_timeout_s": cfg.drain_timeout_s}


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key, ensure_ascii=False)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)   # JSON string escapes are valid TOML basic-string escapes
    raise DeployError(f"cannot write {type(value).__name__} values to the gateway settings")


def toml_dumps(doc: dict) -> str:
    """Scalars, then one level of tables: exactly the shape of capserve Settings."""
    lines = [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in doc.items() if not isinstance(v, dict)]
    for k, v in doc.items():
        if isinstance(v, dict):
            lines += ["", f"[{_toml_key(k)}]"] + [f"{_toml_key(k2)} = {_toml_value(v2)}" for k2, v2 in v.items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- render, pins, release

def stage_b_problems(pins: dict) -> list:
    sb = pins.get("stage_b") or {}
    problems = []
    if sb.get("gdn_prefill_backend") not in smoke.GDN_KERNEL_NAMES:
        problems.append(f"pins stage_b.gdn_prefill_backend {sb.get('gdn_prefill_backend')!r} is not measured")
    if not sb.get("gpu_memory_utilization"):
        problems.append("pins stage_b.gpu_memory_utilization is not measured")
    for key in ("kv_cache_memory_bytes", "num_gpu_blocks", "block_size"):
        value = sb.get(key)
        if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            problems.append(f"pins stage_b.{key} is not measured ({value!r})")
    return problems


def render_vllm(pins: dict, cfg: DeployConfig, base_env: Mapping[str, str]) -> tuple:
    """(argv, env): the Stage B server (core_b_plan.server_argv, read-only) on the product port, the research venv's
    vllm, in procs.vllm_env."""
    sb = pins["stage_b"]
    argv, _ = bp.server_argv(pins, gdn_prefill_backend=sb["gdn_prefill_backend"],
                             gpu_memory_utilization=sb["gpu_memory_utilization"], port=cfg.vllm_port,
                             vllm_bin=cfg.vllm_bin, kv_cache_memory_bytes=sb["kv_cache_memory_bytes"])
    return argv, procs.vllm_env(base_env, cfg, pins=pins)


def fingerprint(argv, env) -> str:
    return server_fingerprint(argv, bp.recorded_env(env))


def release_problems(release_root, cfg: DeployConfig) -> tuple:
    """(sha, problems) of the release directory this supervisor runs from (scripts/deploy_bundle.py's files)."""
    root = Path(release_root).resolve()
    releases = (Path(cfg.service_root) / "releases").resolve()
    research = Path(cfg.research_dir).resolve()
    problems = []
    if root.parent != releases:
        problems.append(f"{root} is not a release directory {releases}/<sha> (tools/rtx/deploy_service.sh)")
    if root == research or root.is_relative_to(research):
        problems.append(f"{root} is inside the research directory {research}: the product never runs from it")
    try:
        sha = (root / "DEPLOYED_COMMIT").read_text().split()[0]
    except (OSError, IndexError):
        return None, problems + [f"{root}/DEPLOYED_COMMIT is missing or empty"]
    if not _SHA.fullmatch(sha):
        problems.append(f"DEPLOYED_COMMIT {sha!r} is not a full commit sha")
    if root.name != sha:
        problems.append(f"release directory name {root.name!r} != DEPLOYED_COMMIT {sha}")
    try:
        manifest = json.loads((root / "DEPLOY_MANIFEST.json").read_text())
    except (OSError, ValueError) as exc:
        return sha, problems + [f"DEPLOY_MANIFEST.json is unreadable: {exc}"]
    if manifest.get("commit") != sha:
        problems.append(f"DEPLOY_MANIFEST.json commit {manifest.get('commit')!r} != DEPLOYED_COMMIT {sha}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return sha, problems + ["DEPLOY_MANIFEST.json lists no files"]
    changed = []
    for rel, want in sorted(files.items()):
        try:
            got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            changed.append(f"D {rel}")
            continue
        if got != want:
            changed.append(f"M {rel}")
    if changed:
        more = " ..." if len(changed) > 20 else ""
        problems.append(f"{len(changed)} release file(s) differ from DEPLOY_MANIFEST.json: "
                        + ", ".join(changed[:20]) + more)
    return sha, problems


def verified_profile_problems(profile_path, data: bytes, release_root) -> list:
    """027 §4: a verified profile comes only from the generator. Besides the loader's run-time checks (the frozen
    configs, the session ids and the Stage C classes; capserve.profile), deploy serves one only as this release's
    service/profiles/<name>.json (top level, lowercase .json) listed in DEPLOY_MANIFEST.json with exactly these
    bytes: a committed file, whose numbers the Mac test that regenerates every committed verified profile covers."""
    root = Path(release_root).resolve()
    path = Path(profile_path).resolve()
    why = "a verified profile is served only as a committed release file listed in DEPLOY_MANIFEST.json (027 §4)"
    if not path.is_relative_to(root):
        return [f"{path} is outside the release {root}: {why}"]
    try:
        files = json.loads((root / "DEPLOY_MANIFEST.json").read_text()).get("files")
    except (OSError, ValueError, AttributeError) as exc:
        return [f"DEPLOY_MANIFEST.json is unreadable ({exc}): {why}"]
    rel = path.relative_to(root).as_posix()
    if not VERIFIED_PROFILE_PATH.fullmatch(rel):
        return [f"{rel} is not service/profiles/<name>.json: {why}"]
    want = files.get(rel) if isinstance(files, dict) else None
    if want is None:
        return [f"{rel} is not in DEPLOY_MANIFEST.json: {why}"]
    if hashlib.sha256(data).hexdigest() != want:
        return [f"{rel} differs from its DEPLOY_MANIFEST.json hash: {why}"]
    return []


def production_profile_problems(cfg: DeployConfig, profile_path, *, allow_provisional: bool, release_root) -> list:
    """036: a production config (production_profile set) serves exactly its pinned verified profile and never runs
    with provisional profiles allowed, so a provisional profile, another committed profile or a leftover
    CAPSERVE_ALLOW_PROVISIONAL=1 cannot replace the product limits. [] for an integration config (no pin)."""
    if cfg.production_profile is None:
        return []
    problems = []
    if allow_provisional:
        problems.append(f"this production config (production_profile {cfg.production_profile}) never runs with "
                        "provisional profiles allowed: drop --allow-provisional and unset CAPSERVE_ALLOW_PROVISIONAL "
                        "(036)")
    pinned = Path(release_root).resolve() / cfg.production_profile
    if Path(profile_path).resolve() != pinned:
        problems.append(f"--profile {profile_path} is not this production config's production_profile "
                        f"{cfg.production_profile} ({pinned}); a production deploy serves only that file (036)")
    return problems


# ---------------------------------------------------------------- effects (all injectable)

def default_http_get(url: str, timeout: float = HTTP_TIMEOUT_S) -> tuple:
    """(status, text); status 0 when nothing answered. No proxy, even if the environment names one."""
    try:
        r = httpx.get(url, timeout=timeout, trust_env=False)
        return r.status_code, r.text
    except httpx.HTTPError as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


@dataclass
class Effects:
    base_env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    start_group: Callable = procs.start_group
    http_get: Callable = default_http_get
    now: Callable = time.monotonic
    wall: Callable = utcnow
    sleep: Callable = time.sleep
    held_gpu_lock: Callable = run_adv.held_gpu_lock
    lock_fd: Callable = procs.lock_fd
    gpu_mem_used: Callable = run_adv.nvidia_mem_used_bounded      # the teardown's return to the baseline
    gpu_sample: Callable = gi.nvidia_sample                       # 031: the idle window's samples
    compute_apps: Callable = gi.nvidia_compute_apps
    shared_usage: Callable = gi.read_shared_usage                 # 031: descriptive for product tests
    toolchain_problems: Callable = bp.toolchain_problems
    port_free: Callable = procs.port_free
    signal_group: Callable = smoke.signal_group
    group_alive: Callable = smoke.group_alive
    pid: Callable = os.getpid
    python: str = sys.executable


@dataclass
class Running:
    """What a workload gets once vLLM and the gateway are ready."""
    cfg: DeployConfig
    run_dir: Path
    vllm: object
    gateway: object
    hold_deadline: float
    fx: Effects
    api_key: str | None

    @property
    def gateway_url(self) -> str:
        return self.cfg.gateway_url

    @property
    def vllm_url(self) -> str:
        return self.cfg.vllm_url


# ---------------------------------------------------------------- checks

def profile_server_problems(prof, *, pins: dict, fingerprint: str) -> list:
    """029: a profile is refused for a server other than the one it states (a verified profile always states it)."""
    sb, server = pins["stage_b"], prof.server
    problems = []
    if server.fingerprint_sha256 is not None and server.fingerprint_sha256 != fingerprint:
        problems.append(f"profile {prof.profile_id} fingerprint {server.fingerprint_sha256} != this server's "
                        f"{fingerprint} (029: refused)")
    for key in ("num_gpu_blocks", "block_size"):
        value = getattr(server, key)
        if value is not None and value != sb[key]:
            problems.append(f"profile server.{key} {value} != pins stage_b.{key} {sb[key]}")
    return problems


def idle_record(samples: list, compute_apps: list, *, bound_mib: int | None) -> dict:
    """031's product rule on one idle window, for meta.json. Its "problems" (a refusal) are gpu_idle's: a compute
    load (utilization.gpu > 5 % or not reported in a sample, a compute process) or an unstable baseline (memory.used
    range > 64 MiB). The baseline is the window's idle value (its largest memory.used), which the teardown and IT-5
    return to. A baseline above the headroom bound is only a warning: product tests check function, not latency
    (preflight refuses a baseline above PRODUCT_IDLE_CAP_MIB, the 031 addendum)."""
    window = gi.window_report(samples, compute_apps, bound_mib=sys.maxsize, attempt=1)   # the headroom: see below
    baseline = window["idle_mib"]
    warnings = []
    if bound_mib is not None and baseline is not None and baseline > bound_mib:
        warnings.append(f"idle memory.used {baseline} MiB > the 031 headroom bound {bound_mib} MiB (V_total - F_peak "
                        f"{gi.F_PEAK_MIB} - margin {gi.MARGIN_MIB}): the vLLM peak may not fit in VRAM; a product "
                        "test still runs (031), with its shared usage recorded")
    return {"decision": gi.DECISION, "baseline_mib": baseline, "bound_mib": bound_mib,
            "return_limit_mib": None if baseline is None else baseline + IDLE_RETURN_TOL_MIB, "warnings": warnings,
            "window": {**window, "bound_mib": bound_mib}}


def preflight(cfg: DeployConfig, prof, *, pins: dict, env: dict, fingerprint: str, run_id: str,
              fx: Effects) -> tuple:
    """(problems, facts) before anything starts; every problem is reported, not only the first."""
    problems, facts = [], {}
    if not fx.held_gpu_lock():
        problems.append(f"this process does not hold the shared GPU lock {smoke.GPU_LOCK} (run it under "
                        f"`flock -n {smoke.GPU_LOCK} ...`; 027 §5)")
    facts["lock_fd"] = fd = fx.lock_fd()
    if fd is None:
        problems.append("no descriptor holding the GPU lock to pass to vLLM and the gateway (029: pass_fds)")
    try:
        bound = gi.headroom_bound(pins["stage_b"].get("v_total_mib"))
    except ValueError as exc:
        bound = None
        problems.append(f"pins stage_b.v_total_mib: {exc} (031 headroom bound)")
    try:
        samples = gi.idle_window(fx.gpu_sample, now=fx.now, sleep=fx.sleep)
        facts["gpu_idle"] = idle = idle_record(samples, fx.compute_apps(), bound_mib=bound)
        problems += [f"the GPU is not idle: {p}" for p in idle["window"]["problems"]]
        if idle["baseline_mib"] is not None and idle["baseline_mib"] > PRODUCT_IDLE_CAP_MIB:
            problems.append(f"idle memory.used {idle['baseline_mib']} MiB > {PRODUCT_IDLE_CAP_MIB} MiB: resident memory "
                            "that is not the desktop, likely another project's model; never boot on top of other "
                            "tenants (031 addendum)")
    except Exception as exc:  # noqa: BLE001 - nvidia-smi missing, failing, hung or unparsable: refuse, never guess
        problems.append(f"nvidia-smi failed: {type(exc).__name__}: {exc}")
    problems += [f"toolchain: {p}" for p in fx.toolchain_problems(env)]
    for name, port in (("vllm_port", cfg.vllm_port), ("gateway_port", cfg.gateway_port)):
        if not fx.port_free(port):
            problems.append(f"port {port} ({name}) is in use")
    problems += runs.run_path_problems(run_id, service_root=cfg.service_root, research_dir=cfg.research_dir)
    if prof is not None:
        problems += profile_server_problems(prof, pins=pins, fingerprint=fingerprint)
        try:
            cls = prof.serving_class(cfg.gateway.get("active_class", ""))   # an unavailable class: refused (030 8.7)
            if not cls.serviceable:
                problems.append(f"active class {cls.name} of profile {prof.profile_id} is a verified 0 (refuse this "
                                "class, 030 8.7): its gateway answers every request 503 class_not_serviceable and "
                                "is never ready, so there is nothing to boot vLLM for")
        except ProfileError as exc:
            problems.append(str(exc))
    return problems, facts


def shared_reading(fx: Effects, when: str, pre: dict | None = None) -> dict:
    """031, descriptive only for product tests: the NVIDIA adapter's Windows shared usage at `when`
    (gpu_idle.read_shared_usage), with delta_mib since the pre-boot reading when both were read on the same adapter.
    A reading that fails is recorded with its error, never fatal."""
    try:
        entry = {"when": when, **fx.shared_usage(), "error": None}
    except Exception as exc:  # noqa: BLE001 - no interop, a timeout, unparsable output: recorded, never fatal
        return {"when": when, "error": f"{type(exc).__name__}: {exc}"}
    if pre is not None and pre.get("error") is None and pre.get("luid") == entry.get("luid"):
        entry["delta_mib"] = round(entry["shared_mib"] - pre["shared_mib"], 1)
    return entry


def server_problems(metrics_text: str, log_text: str, *, pins: dict) -> tuple:
    """(facts, problems): is the running vLLM the Core B server (run_core_b / run_adv's checks, read-only)?"""
    sb = pins["stage_b"]
    cache = run_core_b.cache_config(run_core_b.parse_prometheus(metrics_text or ""))
    facts = smoke.parse_server_log(log_text)
    nondefault = run_adv.parse_nondefault(log_text)
    problems = []
    want = {"num_gpu_blocks": sb["num_gpu_blocks"], "block_size": sb["block_size"]}
    if cache is None:
        problems.append("vLLM /metrics has no vllm:cache_config_info")
    elif cache != want:
        problems.append(f"num_gpu_blocks / block_size {cache} != Stage B {want}")
    if facts.get("attention_backend") != "FLASH_ATTN":
        problems.append(f"attention backend {facts.get('attention_backend')!r} != FLASH_ATTN (003, §4)")
    kernel = smoke.GDN_KERNEL_NAMES.get(sb["gdn_prefill_backend"])
    if facts.get("gdn_prefill_kernel") != kernel:
        problems.append(f"GDN prefill kernel {facts.get('gdn_prefill_kernel')!r} != {kernel!r} (008)")
    if nondefault is None:
        problems.append("no non-default args line in the vLLM log")
    elif nondefault.get("max_num_batched_tokens") != bp.MAX_NUM_BATCHED_TOKENS:
        problems.append(f"max_num_batched_tokens {nondefault.get('max_num_batched_tokens')!r} in the non-default "
                        f"args != {bp.MAX_NUM_BATCHED_TOKENS} (003, §4)")
    return {"cache_config": cache, "log_facts": facts, "nondefault": nondefault}, problems


def served_profile_problems(readyz_text: str, prof, active_class: str) -> list:
    """The gateway's /readyz must name the profile id, status, active class and limits preflight checked."""
    try:
        body = json.loads(readyz_text)
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return [f"gateway /readyz body is not a JSON object: {readyz_text[:200]!r}"]
    cls = prof.workload_class(active_class)
    want = {"profile_id": prof.profile_id, "status": prof.status, "active_class": cls.name,
            **{k: getattr(cls, k) for k in SERVED_LIMITS}}
    profile, limits = body.get("profile") or {}, body.get("limits") or {}
    got = {"profile_id": profile.get("id"), "status": profile.get("status"), "active_class": body.get("active_class"),
           **{k: limits.get(k) for k in SERVED_LIMITS}}
    if got != want:
        return [f"gateway /readyz serves {got} != what preflight checked {want} (refused)"]
    return []


# ---------------------------------------------------------------- state.json

def write_state(cfg: DeployConfig, state: dict, *, wall: Callable = utcnow) -> dict:
    doc = {"schema_version": 1, **state, "updated_utc": wall().astimezone(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")}
    path = Path(cfg.service_root) / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return doc


def read_state(cfg: DeployConfig) -> dict | None:
    try:
        return json.loads((Path(cfg.service_root) / STATE_FILE).read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- workloads

def hold(run: Running) -> dict:
    """Serve until max_hold_s after the start or until either child exits; a signal ends it by an exception."""
    while True:
        for name, proc in (("vLLM", run.vllm), ("gateway", run.gateway)):
            code = proc.poll()
            if code is not None:
                return {"ok": False, "ended_by": f"{name.lower()}_exit",
                        "problems": [f"{name} exited (code {code}) while serving"]}
        if run.fx.now() >= run.hold_deadline:
            return {"ok": True, "ended_by": "max_hold_s", "problems": []}
        run.fx.sleep(HOLD_POLL_S)


def loadtest_workload(scenario: str) -> Callable:
    if scenario not in loadtest.SCENARIOS:
        raise DeployError(f"unknown load-test scenario {scenario!r}")

    def run(r: Running) -> dict:
        summary = loadtest.run_scenario(scenario, gateway_url=r.gateway_url, vllm_url=r.vllm_url, run_dir=r.run_dir,
                                        time_limit_s=max(r.hold_deadline - r.fx.now(), 0.0), api_key=r.api_key)
        return {"ok": bool(summary.get("ok")), "ended_by": "loadtest", "problems": list(summary.get("problems", [])),
                "loadtest": {"scenario": scenario, "ok": summary.get("ok"), "summary": "summary.json"}}

    return run


# ---------------------------------------------------------------- up

def _wait(ok: Callable[[], bool], *, dead: Callable[[], str | None], timeout_s: float, poll_s: float,
          fx: Effects) -> str | None:
    """None once ok() holds within timeout_s; otherwise why not (a child exited, or the timeout)."""
    t = fx.now()
    while True:
        if ok():
            return None
        why = dead()
        if why:
            return why
        if fx.now() - t >= timeout_s:
            return f"not within {timeout_s:g} s"
        fx.sleep(poll_s)


def _exited(*named) -> str | None:
    for name, proc in named:
        code = proc.poll()
        if code is not None:
            return f"{name} exited (code {code})"
    return None


def _best_effort(fn: Callable[[], object], what: str) -> list:
    """[] once fn ran, else the problem: bookkeeping around the teardown (state.json, meta.json) never skips it nor
    replaces an interrupt on its way out (029 step 7: the teardown always runs)."""
    try:
        fn()
        return []
    except Exception as exc:  # noqa: BLE001 - a full disk or a permission error is recorded, never raised here
        return [f"{what} failed: {type(exc).__name__}: {exc}"]


def up(cfg: DeployConfig, profile_path, *, allow_provisional: bool, loadtest: str | None = None,
       workload: Callable | None = None, fx: Effects | None = None, pins: dict | None = None,
       frozen: FrozenConfigs | None = None, release_root=RELEASE_ROOT, err=None) -> int:
    fx = fx or Effects()
    err = err or sys.stderr
    pins = bp.load_pins() if pins is None else pins
    t0 = fx.now()

    def say(msg: str) -> None:
        try:
            print(f"capserve.deploy: {msg}", file=err, flush=True)
        except OSError:   # stderr gone (an ssh hang-up): never a reason to stop a start or a teardown
            pass

    if workload is None:
        workload = loadtest_workload(loadtest) if loadtest else hold
    mode = f"loadtest:{loadtest}" if loadtest else ("hold" if workload is hold else "workload")

    # 1-2. release, pins, profile, preflight: refuse with every problem before anything starts
    sha, problems = release_problems(release_root, cfg)
    problems += stage_b_problems(pins)
    problems += production_profile_problems(cfg, profile_path, allow_provisional=allow_provisional,
                                            release_root=release_root)
    prof = profile_bytes = None
    try:   # read once: these bytes are checked, recorded and served (runs/<id>/profile.json)
        profile_bytes = Path(profile_path).read_bytes()
        prof = parse_profile(profile_bytes, source=profile_path, pins=pins, frozen=frozen,
                             allow_provisional=allow_provisional and cfg.production_profile is None)
    except OSError as exc:
        problems.append(f"{profile_path}: cannot read the profile: {exc}")
    except ProfileError as exc:
        problems.append(str(exc))
    if prof is not None and prof.verified:
        problems += verified_profile_problems(profile_path, profile_bytes, release_root)
    argv, env = render_vllm(pins, cfg, fx.base_env)
    fp = fingerprint(argv, env)
    run_id = runs.new_run_id(fx.wall(), loadtest)
    # 031 made the preflight last 30 s or more (the idle window, then the pre-boot shared usage): state.json names this
    # supervisor as soon as it holds the lock, so `stop` (and `status`) find it meanwhile. Only the lock holder writes
    # state.json: a refused `up` without the lock never replaces a running supervisor's.
    early = {"supervisor_pid": fx.pid(), "run_id": run_id, "mode": mode}
    owns_state = fx.held_gpu_lock()
    state: dict = {}

    def early_phase(phase: str, **fields) -> list:
        """A state.json write before vLLM can start: [] once written (or not ours to write), else the problem."""
        if not owns_state:
            return []
        return _best_effort(lambda: write_state(cfg, {**early, **state, **fields, "phase": phase}, wall=fx.wall),
                            f"state.json write (phase {phase})")

    def set_phase(phase: str, **fields) -> None:
        state.update(fields)
        write_state(cfg, {**state, "phase": phase}, wall=fx.wall)

    try:
        problems += early_phase("preflight")
        pre_problems, pre = preflight(cfg, prof, pins=pins, env=env, fingerprint=fp, run_id=run_id, fx=fx)
        problems += pre_problems
        if problems:
            problems += early_phase("refused", problems=list(problems))
            say("refused; nothing was started:")
            for p in problems:
                say(f"  - {p}")
            return 2
        warnings = pre["gpu_idle"]["warnings"]
        for w in warnings:
            say(f"warning: {w}")
        shared = [shared_reading(fx, "pre-boot")]   # 031: after the idle window passed, before vLLM starts

        # 3. run dir and state
        profile_meta = {"id": prof.profile_id, "status": prof.status, "content_sha256": prof.content_sha256,
                        "path": str(Path(profile_path).resolve())}
        run_dir = runs.create_run(service_root=cfg.service_root, research_dir=cfg.research_dir, run_id=run_id,
                                  kind="deploy", release=sha, profile=profile_meta, now_utc=fx.wall(),
                                  extra={"mode": mode, "supervisor_pid": fx.pid(), "vllm_argv": argv,
                                         "recorded_env": bp.recorded_env(env), "fingerprint": fp, "preflight": pre,
                                         "warnings": warnings, "shared_usage": list(shared),
                                         "deploy_config": cfg.model_dump()})
        with open(run_dir / PROFILE_COPY, "xb") as f:
            f.write(profile_bytes)
        state.update({"supervisor_pid": fx.pid(), "run_id": run_id, "run_dir": str(run_dir), "release": sha,
                      "mode": mode, "profile": profile_meta, "vllm_url": cfg.vllm_url, "gateway_url": cfg.gateway_url,
                      "vllm_pid": None, "gateway_pid": None})
        set_phase("starting")
    except BaseException as exc:   # a `stop` meanwhile (SIGTERM -> SystemExit), or a failure: nothing was started
        early_phase("stopped", exit=None, interrupted=f"{type(exc).__name__}({exc})")
        raise
    say(f"run {run_dir}")
    started, info, problems = {}, {"timeline": {}, "shared_usage": shared}, []
    code, interrupted = 1, None
    try:
        code = _serve(cfg, prof, argv=argv, env=env, run_dir=run_dir,
                      allow_provisional=allow_provisional, workload=workload, started=started, info=info,
                      problems=problems, set_phase=set_phase, fx=fx, release_root=release_root, pins=pins,
                      lock_fd=pre["lock_fd"], t0=t0, say=say)
    except BaseException as exc:
        interrupted = f"{type(exc).__name__}({exc})"
        raise
    finally:
        with run_adv.DeferredSignals() as deferred:
            # the teardown always runs: every write around it is best effort and only becomes a problem
            failed = _best_effort(lambda: set_phase("stopping"), "state.json write (phase stopping)")
            info["timeline"]["teardown_start_s"] = fx.now() - t0
            td = teardown(started, cfg, deferred, fx, baseline_mib=pre["gpu_idle"]["baseline_mib"], shared=shared)
            if (td["problems"] or failed) and code == 0:
                code = 1
            stopped = _best_effort(lambda: set_phase("stopped", exit=code if interrupted is None else None,
                                                     interrupted=interrupted), "state.json write (phase stopped)")
            if stopped and code == 0:
                code = 1
            failed += stopped
            problems.extend(failed)
            outcome = {"ok": code == 0 and interrupted is None, "exit": code if interrupted is None else None,
                       "interrupted": interrupted, "problems": problems, "server": info.get("server"),
                       "workload": info.get("workload"), "teardown": td, "timeline": info["timeline"],
                       "ended_utc": fx.wall().astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            meta_failed = _best_effort(lambda: runs.update_meta(run_dir, outcome=outcome, shared_usage=shared),
                                       "meta.json outcome write")
            if meta_failed and code == 0:
                code = 1
            say(f"stopped: exit {code if interrupted is None else None}, "
                f"{len(problems) + len(td['problems']) + len(meta_failed)} problem(s); see {run_dir / 'meta.json'}")
            for p in failed + meta_failed:
                say(f"  - {p}")
    return code


def _serve(cfg, prof, *, argv, env, run_dir: Path, allow_provisional, workload, started: dict, info: dict,
           problems: list, set_phase, fx: Effects, release_root, pins, lock_fd: int, t0: float, say) -> int:
    timeline = info["timeline"]
    hold_deadline = t0 + cfg.max_hold_s

    def within_hold(limit_s: float) -> float:
        """A boot wait never runs past max_hold_s: it bounds the lock, the boot included."""
        return max(0.0, min(limit_s, hold_deadline - fx.now()))

    def held_too_long(why: str) -> str:
        return f"{why} (max_hold_s {cfg.max_hold_s:g} s reached)" if fx.now() >= hold_deadline else why

    for d in (procs.home_dir(cfg), procs.tmp_dir(cfg)):
        d.mkdir(parents=True, exist_ok=True)
    keep = (lock_fd,)

    # 4. vLLM
    vllm = started["vllm"] = fx.start_group(argv, env, run_dir / "vllm.log", cwd=run_dir, keep_fds=keep)
    set_phase("starting", vllm_pid=vllm.pid)
    limit = within_hold(cfg.startup_timeout_s)
    say(f"vLLM started (pid {vllm.pid}); waiting for /health (at most {limit:g} s)")
    why = _wait(lambda: fx.http_get(f"{cfg.vllm_url}/health")[0] == 200, dead=lambda: _exited(("vLLM", vllm)),
                timeout_s=limit, poll_s=HEALTH_POLL_S, fx=fx)
    if why:
        problems.append(f"vLLM /health: {held_too_long(why)}")
        return 1
    timeline["vllm_healthy_s"] = fx.now() - t0
    status, metrics_text = fx.http_get(f"{cfg.vllm_url}/metrics")
    log_text = (run_dir / "vllm.log").read_text(errors="replace")
    info["server"], server_bad = server_problems(metrics_text if status == 200 else "", log_text, pins=pins)
    if server_bad:
        problems.extend(server_bad)
        return 2
    shared = info["shared_usage"]
    shared.append(shared_reading(fx, "ready", shared[0]))   # 031: vLLM holds its memory now

    # 5. gateway
    gw_toml = run_dir / "gateway.toml"
    gw_toml.write_text(toml_dumps(gateway_settings(cfg)))
    gw_argv = [fx.python, "-m", "capserve", "serve", "--config", str(gw_toml), "--profile",
               str(run_dir / PROFILE_COPY)]   # the bytes preflight checked, never the mutable --profile path
    if allow_provisional and not prof.verified:
        gw_argv.append("--allow-provisional")
    gw = started["gateway"] = fx.start_group(gw_argv, procs.gateway_env(fx.base_env, cfg, release_root=release_root),
                                             run_dir / "gateway.log", cwd=run_dir, keep_fds=keep)
    set_phase("starting", gateway_pid=gw.pid)
    why = _wait(lambda: fx.http_get(f"{cfg.gateway_url}/readyz")[0] == 200,
                dead=lambda: _exited(("vLLM", vllm), ("gateway", gw)), timeout_s=within_hold(GATEWAY_READY_TIMEOUT_S),
                poll_s=HEALTH_POLL_S, fx=fx)
    if why:
        problems.append(f"gateway /readyz: {held_too_long(why)}")
        return 1
    status, readyz_text = fx.http_get(f"{cfg.gateway_url}/readyz")
    if status != 200:
        problems.append(f"gateway /readyz returned {status} right after it was ready")
        return 1
    served_bad = served_profile_problems(readyz_text, prof, cfg.gateway.get("active_class", ""))
    if served_bad:   # the gateway serves another profile than preflight checked: refused, like a non-Core-B server
        problems.extend(served_bad)
        return 2
    timeline["gateway_ready_s"] = fx.now() - t0
    set_phase("ready")
    say(f"ready: gateway {cfg.gateway_url} (profile {prof.profile_id}, {prof.status}), vLLM {cfg.vllm_url}")

    # 6. workload
    run = Running(cfg=cfg, run_dir=run_dir, vllm=vllm, gateway=gw, hold_deadline=hold_deadline, fx=fx,
                  api_key=fx.base_env.get(API_KEY_ENV) or None)
    result = info["workload"] = workload(run)
    problems.extend(result.get("problems", []))
    return 0 if result.get("ok") and not result.get("problems") else 1


# ---------------------------------------------------------------- teardown

def _guarded(fn: Callable[[], list], deferred, what: str) -> list:
    """fn to its end; an interrupt raised out of it anyway is kept for DeferredSignals and fn runs again."""
    problems = []
    for _ in range(run_adv.SHUTDOWN_ATTEMPTS):
        try:
            return problems + fn()
        except (KeyboardInterrupt, SystemExit) as exc:
            deferred.raised.append(exc)
            problems.append(f"{what} interrupted by {type(exc).__name__}; run again")
    return problems + [f"{what} interrupted {run_adv.SHUTDOWN_ATTEMPTS} times; it may still be alive"]


def stop_gateway(proc, *, grace_s: float, fx: Effects) -> list:
    """SIGTERM the gateway's group (uvicorn drains for drain_timeout_s), SIGKILL after grace_s."""
    def gone_within(limit: float) -> bool:
        t = fx.now()
        while True:
            proc.poll()
            if not fx.group_alive(proc):
                return True
            if fx.now() - t >= limit:
                return False
            fx.sleep(1)

    if gone_within(0):
        return []
    fx.signal_group(proc, signal.SIGTERM)
    if gone_within(grace_s):
        code = proc.poll()
        return [] if code in (None, 0, -signal.SIGTERM) else [
            f"gateway exit status {code} after SIGTERM (expected 0 or -15)"]
    fx.signal_group(proc, signal.SIGKILL)
    reasons = [f"gateway had to be killed ({grace_s:g} s after SIGTERM)"]
    if not gone_within(KILL_WAIT_S):
        reasons.append("residual process in the gateway group after SIGKILL")
    return reasons


def wait_gpu_idle(fx: Effects, baseline_mib: int, limit_s: float = IDLE_RETURN_S) -> tuple:
    """031: GPU memory.used back <= the run's pre-boot baseline + IDLE_RETURN_TOL_MIB within limit_s."""
    limit_mib = baseline_mib + IDLE_RETURN_TOL_MIB
    t = fx.now()
    while True:
        try:
            used, error = fx.gpu_mem_used(), None
        except Exception as exc:  # noqa: BLE001 - recorded as a teardown problem
            used, error = None, f"{type(exc).__name__}: {exc}"
        if used is not None and used <= limit_mib:
            return used, []
        if fx.now() - t >= limit_s:
            if error:
                return None, [f"nvidia-smi failed after teardown: {error}"]
            return used, [f"GPU memory.used {used} MiB > {limit_mib} MiB (the pre-boot baseline {baseline_mib} + "
                          f"{IDLE_RETURN_TOL_MIB}; 031) {limit_s:g} s after teardown"]
        fx.sleep(IDLE_POLL_S)


def teardown(started: dict, cfg: DeployConfig, deferred, fx: Effects, *, baseline_mib: int, shared: list) -> dict:
    """The gateway first (it drains), then the after-workload shared usage (031, only once vLLM was ready: it still
    holds its memory), then the vLLM group, then GPU memory back to the pre-boot baseline + 64 MiB."""
    out = {"problems": []}
    gw = started.get("gateway")
    if gw is not None:
        reasons = _guarded(lambda: stop_gateway(gw, grace_s=cfg.drain_timeout_s + GATEWAY_KILL_EXTRA_S, fx=fx),
                           deferred, "gateway teardown")
        out["gateway"] = {"returncode": gw.poll(), "reasons": reasons}
        out["problems"] += reasons
    vllm = started.get("vllm")
    if vllm is not None:
        if any(r["when"] == "ready" for r in shared):
            shared.append(shared_reading(fx, "after", shared[0]))
        reasons = run_adv.shutdown_group(vllm, deferred, signal_group=fx.signal_group, group_alive=fx.group_alive,
                                         now=fx.now, sleep=fx.sleep)
        out["vllm"] = {"returncode": vllm.poll(), "shutdown_reasons": reasons}
        out["problems"] += reasons
        out["gpu_baseline_mib"] = baseline_mib
        out["gpu_return_limit_mib"] = baseline_mib + IDLE_RETURN_TOL_MIB
        out["gpu_mem_after_mib"], idle_problems = wait_gpu_idle(fx, baseline_mib)
        out["problems"] += idle_problems
    return out


# ---------------------------------------------------------------- status, stop

def _cmdline(pid, proc_root: str) -> list | None:
    """The argv of a live process (None if there is none)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        raw = (Path(proc_root) / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return [a.decode(errors="replace") for a in raw.split(b"\0") if a]


def is_supervisor(args: list) -> bool:
    return "capserve.deploy" in args and "up" in args


def status(cfg: DeployConfig, *, proc_root: str = "/proc", http_get: Callable = default_http_get) -> dict:
    state = read_state(cfg)
    if state is None:
        return {"running": False, "state": None}
    args = _cmdline(state.get("supervisor_pid"), proc_root)
    out = {"running": args is not None and is_supervisor(args), "state": state}
    if out["running"]:
        out["vllm_healthy"] = http_get(f"{cfg.vllm_url}/health")[0] == 200
        out["gateway_ready"] = http_get(f"{cfg.gateway_url}/readyz")[0] == 200
    return out


def default_stop_wait_s(cfg: DeployConfig) -> float:
    """Long enough for the whole teardown: gateway drain + kill, the after-workload shared-usage reading, vLLM
    SIGINT / SIGTERM / SIGKILL, the idle check."""
    return (cfg.drain_timeout_s + GATEWAY_KILL_EXTRA_S + KILL_WAIT_S + 2 * 30 + KILL_WAIT_S
            + gi.SHARED_USAGE_TIMEOUT_S + IDLE_RETURN_S + 30)


def stop(cfg: DeployConfig, *, proc_root: str = "/proc", kill: Callable = os.kill, now: Callable = time.monotonic,
         sleep: Callable = time.sleep, wait_s: float | None = None, err=None) -> int:
    """SIGTERM the supervisor recorded in state.json, only if /proc/<pid>/cmdline shows `capserve.deploy up`.
    0 stopped or not running, 1 still running after wait_s, 2 refused (the pid is some other process)."""
    err = err or sys.stderr
    wait_s = default_stop_wait_s(cfg) if wait_s is None else wait_s

    def say(msg: str) -> None:
        print(f"capserve.deploy stop: {msg}", file=err, flush=True)

    state = read_state(cfg)
    if state is None:
        say("no state.json: nothing to stop")
        return 0
    pid = state.get("supervisor_pid")
    args = _cmdline(pid, proc_root)
    if args is None:
        say(f"supervisor {pid} is not running (last phase {state.get('phase')!r})")
        return 0
    if not is_supervisor(args):
        say(f"pid {pid} is not a capserve.deploy supervisor ({' '.join(args)[:200]!r}); not signalled")
        return 2
    try:
        kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        say(f"supervisor {pid} exited meanwhile")
        return 0
    t = now()
    while True:
        args = _cmdline(pid, proc_root)
        if args is None or not is_supervisor(args):
            say(f"supervisor {pid} stopped")
            return 0
        if now() - t >= wait_s:
            say(f"supervisor {pid} is still running {wait_s:g} s after SIGTERM")
            return 1
        sleep(1)


# ---------------------------------------------------------------- CLI

def _build_parser() -> tuple:
    parser = argparse.ArgumentParser(prog="python -m capserve.deploy",
                                     description="deploy capserve and its vLLM on the RTX (product tests only)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    subs = {}
    for name, help_ in (("render", "print the vLLM argv, recorded env and fingerprint (works on the Mac)"),
                        ("up", "foreground supervisor: preflight, vLLM, gateway, workload, teardown"),
                        ("status", "state.json and whether the supervisor and endpoints are alive"),
                        ("stop", "SIGTERM the supervisor (only a verified capserve.deploy up process)")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", default=str(DEFAULT_CONFIG), help="deploy config (TOML)")
        subs[name] = p
    subs["up"].add_argument("--profile", required=True, help="capacity profile (JSON)")
    subs["up"].add_argument("--allow-provisional", action="store_true",
                            help="allow a provisional profile (integration / product tests; not product limits)")
    subs["up"].add_argument("--loadtest", choices=sorted(loadtest.SCENARIOS),
                            help="run this capserve.loadtest scenario instead of holding")
    return parser, subs


def parsers() -> list:
    parser, subs = _build_parser()
    return [parser, *subs.values()]


def main(argv: list | None = None, *, env: Mapping[str, str] | None = None) -> int:
    parser, _ = _build_parser()
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    try:
        cfg = load_config(args.config)
    except DeployError as exc:
        print(f"capserve.deploy: {exc}", file=sys.stderr)
        return 2
    if args.cmd == "render":
        pins = bp.load_pins()
        argv_, env_ = render_vllm(pins, cfg, env)
        print(json.dumps({"argv": argv_, "recorded_env": bp.recorded_env(env_), "fingerprint": fingerprint(argv_, env_),
                          "vllm_url": cfg.vllm_url, "gateway_url": cfg.gateway_url,
                          "stage_b_problems": stage_b_problems(pins)}, indent=1))
        return 0
    if args.cmd == "status":
        st = status(cfg)
        print(json.dumps(st, indent=1))
        return 0 if st["running"] else 1
    if args.cmd == "stop":
        return stop(cfg)
    for sig in (signal.SIGTERM, signal.SIGHUP):   # SystemExit(128 + signum): the finally tears down
        signal.signal(sig, smoke._exit_on_signal)
    return up(cfg, args.profile, allow_provisional=allow_provisional(args.allow_provisional, env),
              loadtest=args.loadtest)


if __name__ == "__main__":
    raise SystemExit(main())
