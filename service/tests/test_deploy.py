"""Tests for capserve/deploy.py (decision 029 "部署管理"): render on the Mac, and the foreground supervisor `up` with
every OS / HTTP / process effect faked (fake start_group / processes / clock / nvidia-smi / powershell shared usage /
lock descriptor): release manifest, preflight refusals (nothing started, no run dir), the 031 idle window and its
baseline (a headroom shortfall only warns), vLLM with the lock fd, server checks, gateway, state.json, hold /
workload, the descriptive shared-usage readings, and the teardown order and checks (GPU memory back to the pre-boot
baseline + 64 MiB) in every exit path; status and stop."""

import datetime as dt
import hashlib
import inspect
import json
import shutil
import signal
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
import coreb_smoke as smoke
import gpu_idle as gi
import run_adv
import run_core_b
from capserve import deploy, procs, runs
from capserve.config import Settings
from capserve.profile import content_sha256, stage_b_fingerprint
from fakes import profiles as fake_profiles
from fakes.profiles import FROZEN, frozen_capacity, frozen_configs, verified_class

PINS = bp.load_pins()
SB = PINS["stage_b"]
SHA = "a" * 40
NOW = dt.datetime(2026, 10, 16, 3, 4, 5, tzinfo=dt.timezone.utc)
RUN_ID = "pt-20261016T030405Z"
LOG_OK = ("INFO non-default args: {'model': 'RedHatAI/Qwen3.5-9B-quantized.w4a16', 'max_num_batched_tokens': 2048, "
          "'kv_cache_memory_bytes': 4294967296}\n"
          "INFO Using FLASH_ATTN backend.\n"
          "INFO Using Triton/FLA GDN prefill kernel\n")
METRICS_OK = 'vllm:cache_config_info{block_size="528",num_gpu_blocks="248"} 1.0\n'
ROOT = Path(__file__).resolve().parents[2]
LUID = "luid_0x00000000_0x000103ae_phys_0"
SHARED = {"luid": LUID, "dedicated_mib": 1396.0, "shared_mib": 64.4}


# ------------------------------------------------------------------ fixtures --

def profile_doc(*, fingerprint=None, blocks=None, block_size=None, name="test"):
    """A provisional profile (verified ones: verified_doc)."""
    doc = {"schema_version": 1, "profile_id": "provisional-test", "status": "provisional",
           "provisional_reason": "tests only; not a capacity", "slo": None,
           "server": {"vllm_version": PINS["vllm"]["version"], "model_repo": PINS["core_b"]["repo"],
                      "model_revision": PINS["core_b"]["revision"], "fingerprint_sha256": fingerprint,
                      "num_gpu_blocks": blocks, "block_size": block_size},
           "classes": [{"name": name, "max_prompt_tokens": 1920, "max_output_tokens": 128, "max_concurrency": 4,
                        "max_queue": 4, "source": "test"}],
           "provenance": None}
    doc["content_sha256"] = content_sha256(doc)
    return doc


def write_json(path, doc):
    path.write_text(json.dumps(doc))
    return path


def make_release(svc: Path, sha=SHA, files=None) -> Path:
    root = svc / "releases" / sha
    files = files or {"service/capserve/x.py": b"print(1)\n", "benchmarks/y.py": b"y = 2\n"}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    manifest = {"commit": sha, "files": {rel: hashlib.sha256(d).hexdigest() for rel, d in files.items()}}
    (root / "DEPLOYED_COMMIT").write_text(sha + "\n")
    (root / "DEPLOY_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return root


def add_to_release(root: Path, rel: str, data: bytes) -> Path:
    """A committed file of the release: written and listed in its DEPLOY_MANIFEST.json."""
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(data)
    manifest = json.loads((root / "DEPLOY_MANIFEST.json").read_text())
    manifest["files"][rel] = hashlib.sha256(data).hexdigest()
    (root / "DEPLOY_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return root / rel


VERIFIED_ACTIVE = {"gateway": {"active_class": "uniform-2k"}}   # the verified class verified_doc serves


def verified_doc(**kw):
    """A verified profile for exactly this server (Stage B fingerprint and KV geometry), class uniform-2k."""
    return fake_profiles.verified_doc(PINS, **kw)


class FakeProc:
    def __init__(self, name, pid):
        self.name, self.pid = name, pid
        self.returncode = None
        self.alive = True
        self.dies_on = {signal.SIGINT, signal.SIGTERM}

    def poll(self):
        return self.returncode

    def die(self, code):
        self.returncode, self.alive = code, False


class World:
    """Every effect up() has, scripted; events records the order of starts and signals."""

    def __init__(self, tmp_path):
        self.svc = tmp_path / "svc"
        self.release = make_release(self.svc)
        self.t = 0.0
        self.started, self.events = [], []
        self.health_after = 3          # /health calls that fail before vLLM is healthy
        self.health_calls = 0
        self.vllm_log, self.metrics, self.readyz = LOG_OK, METRICS_OK, 200
        self.readyz_body = None        # None: what the gateway was started with (served())
        self.idle = [(500, 0)]         # the 031 window's (memory.used, utilization.gpu) samples; the last one repeats
        self.idle_times = []           # when each window sample was taken
        self.apps = []                 # nvidia-smi compute processes
        self.gpu = [500]               # the teardown's memory.used answers in order; the last one repeats
        self.shared = [dict(SHARED)]   # powershell shared-usage readings in order; the last one repeats
        self.shared_reads = []         # len(events) at each shared-usage reading: where it fell in the boot
        self.lock_held, self.fd = True, 9
        self.toolchain, self.busy_ports = [], set()
        self.on_sleep = None
        self.procs = {}

    # effects
    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.on_sleep:
            self.on_sleep(self)

    def start_group(self, argv, env, log_path, *, cwd, keep_fds=()):
        name = "vllm" if argv[1:2] == ["serve"] else "gateway"
        Path(log_path).write_text(self.vllm_log if name == "vllm" else "")
        proc = FakeProc(name, 1000 + len(self.started))
        self.started.append({"name": name, "argv": argv, "env": env, "log": Path(log_path), "cwd": cwd,
                             "keep_fds": tuple(keep_fds)})
        self.events.append(("start", name))
        self.procs[name] = proc
        return proc

    def http_get(self, url, timeout=5.0):
        vllm, gw = self.procs.get("vllm"), self.procs.get("gateway")
        if url == "http://127.0.0.1:8100/health":
            self.health_calls += 1
            ok = vllm is not None and vllm.alive and self.health_calls > self.health_after
            return (200, "") if ok else (0, "connection refused")
        if url == "http://127.0.0.1:8100/metrics":
            return (200, self.metrics)
        if url == "http://127.0.0.1:8080/readyz":
            if gw is None or not gw.alive:
                return (0, "connection refused")
            return (self.readyz, json.dumps(self.readyz_body if self.readyz_body is not None else self.served()))
        raise AssertionError(url)

    def served(self) -> dict:
        """What a real gateway's /readyz says: the profile file and active class it was started with."""
        argvs = [s["argv"] for s in self.started if s["name"] == "gateway"]
        if not argvs:
            return {}
        argv = argvs[-1]
        doc = json.loads(Path(argv[argv.index("--profile") + 1]).read_text())
        active = tomllib.loads(Path(argv[argv.index("--config") + 1]).read_text())["active_class"]
        (cls,) = [c for c in doc["classes"] if c["name"] == active]
        limits = ("max_concurrency", "max_queue", "max_prompt_tokens", "max_output_tokens")
        return {"ready": True, "active_class": active, "limits": {k: cls[k] for k in limits},
                "profile": {"id": doc["profile_id"], "status": doc["status"],
                            "provisional": doc["status"] != "verified"}}

    @staticmethod
    def _next(answers):
        value = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(value, Exception):
            raise value
        return value

    def gpu_mem(self):
        return self._next(self.gpu)

    def gpu_sample(self):
        self.idle_times.append(self.t)
        mem, util = self._next(self.idle)
        return {"mem_used_mib": mem, "util_pct": util, "pstate": "P8"}

    def shared_usage(self):
        self.shared_reads.append(len(self.events))
        return dict(self._next(self.shared))

    def signal_group(self, proc, sig):
        self.events.append(("signal", proc.name, sig))
        if sig == signal.SIGKILL or sig in proc.dies_on:
            proc.die(-sig)

    def group_alive(self, proc):
        return proc.alive

    def fx(self, **over):
        kw = dict(base_env={"PATH": "/usr/bin", "PYTHONPATH": "/old", "CAPSERVE_API_KEY": "k", "HOME": "/home/u"},
                  start_group=self.start_group, http_get=self.http_get, now=self.now, wall=lambda: NOW,
                  sleep=self.sleep, held_gpu_lock=lambda: self.lock_held, lock_fd=lambda: self.fd,
                  gpu_mem_used=self.gpu_mem, gpu_sample=self.gpu_sample, compute_apps=lambda: list(self.apps),
                  shared_usage=self.shared_usage, toolchain_problems=lambda env: list(self.toolchain),
                  port_free=lambda port: port not in self.busy_ports, signal_group=self.signal_group,
                  group_alive=self.group_alive, pid=lambda: 4321, python="/venv/bin/python")
        kw.update(over)
        return deploy.Effects(**kw)

    def cfg(self, **over):
        doc = {"service_root": str(self.svc), "max_hold_s": 100.0, "gateway": {"active_class": "test"}, **over}
        return deploy.DeployConfig.model_validate(doc)

    def run_dir(self):
        return self.svc / "runs" / RUN_ID

    def state(self):
        return json.loads((self.svc / "state.json").read_text())


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture
def prof(tmp_path):
    return write_json(tmp_path / "profile.json", profile_doc())


def up(world, prof, **kw):
    kw.setdefault("allow_provisional", True)
    kw.setdefault("pins", PINS)
    kw.setdefault("frozen", FROZEN)
    fx = kw.pop("fx", None) or world.fx()
    cfg = kw.pop("cfg", None) or world.cfg()
    return deploy.up(cfg, prof, fx=fx, release_root=world.release, **kw)


# ------------------------------------------------------------------ helpers used from research --

def test_the_research_helpers_capserve_uses_exist_with_these_signatures():
    """029 M1: every imported research helper's real name and signature (they are frozen; this catches a typo)."""
    assert list(inspect.signature(bp.server_argv).parameters) == [
        "pins", "gdn_prefill_backend", "gpu_memory_utilization", "port", "vllm_bin", "base_env",
        "kv_cache_memory_bytes"]
    assert list(inspect.signature(bp.server_env).parameters) == ["base", "pin_memory"]
    assert list(inspect.signature(bp.toolchain_problems).parameters) == ["env", "exists", "which"]
    assert list(inspect.signature(run_adv.held_gpu_lock).parameters) == ["lock_path", "fd_dir"]
    assert list(inspect.signature(run_adv.nvidia_mem_used_bounded).parameters) == ["timeout_s"]
    assert list(inspect.signature(run_adv.shutdown_group).parameters) == ["proc", "deferred", "kw"]
    assert list(inspect.signature(smoke._shutdown).parameters) == [
        "proc", "signal_group", "group_alive", "now", "sleep", "grace_s", "kill_wait_s"]
    assert callable(run_adv.parse_nondefault) and callable(smoke.parse_server_log)
    assert run_core_b.IDLE_MIB == run_adv.IDLE_MIB == 1000           # Phase 1's frozen rule stays as it was
    assert not hasattr(deploy, "IDLE_MIB")                            # 031: the product no longer uses it
    assert smoke.GDN_KERNEL_NAMES[SB["gdn_prefill_backend"]] == "Triton/FLA"


def test_the_product_reads_the_gpu_with_the_031_helpers():
    """031: the product's idle window and shared-usage readings are gpu_idle's (read-only), the research's own."""
    fx = deploy.Effects()
    assert fx.gpu_sample is gi.nvidia_sample and fx.compute_apps is gi.nvidia_compute_apps
    assert fx.shared_usage is gi.read_shared_usage and fx.gpu_mem_used is run_adv.nvidia_mem_used_bounded
    assert list(inspect.signature(gi.idle_window).parameters) == ["sample", "n", "interval_s", "now", "sleep"]
    assert list(inspect.signature(gi.window_report).parameters) == ["samples", "compute_apps", "bound_mib", "attempt"]
    assert deploy.IDLE_RETURN_TOL_MIB == 64 and deploy.IDLE_RETURN_S == 60


# ------------------------------------------------------------------ config --

def test_the_committed_rtx_config_loads_with_the_029_defaults():
    cfg = deploy.load_config(ROOT / "service" / "deploy" / "rtx.toml")
    assert cfg.service_root == "/home/wenn0113/llm-serving-service"
    assert cfg.research_dir == bp.PROJECT_DIR == "/home/wenn0113/llm-serving-lab"
    assert (cfg.vllm_port, cfg.gateway_port) == (8100, 8080)
    assert (cfg.startup_timeout_s, cfg.max_hold_s, cfg.drain_timeout_s) == (900, 3600, 60)
    assert cfg.vllm_bin == "/home/wenn0113/llm-serving-lab/.venv/bin/vllm"
    assert cfg.gateway["active_class"] == "rtx-it"
    assert deploy.DEFAULT_CONFIG == ROOT / "service" / "deploy" / "rtx.toml"


@pytest.mark.parametrize("over, match", [
    ({"research_dir": "/home/wenn0113/other"}, "research_dir"),
    ({"vllm_port": 8000}, "8000"),
    ({"gateway_port": 8100}, "differ"),
    ({"service_root": "/home/wenn0113/llm-serving-lab/svc"}, "research"),
    ({"gateway": {"active_class": "x", "port": 9}}, "port"),
    ({"gateway": {"active_class": "x", "backend_url": "http://a"}}, "backend_url"),
    ({"gateway": {"active_class": "x", "host": "0.0.0.0"}}, "host"),
    ({"gateway": {"active_class": "x", "workers": 2}}, "workers"),
    ({"gateway": {}}, "active_class"),
    ({"surprise": 1}, "surprise"),
    ({"max_hold_s": 3601.0}, "max_hold_s"),                 # 029: at most 1 h per session; no resident daemon
    ({"max_hold_s": float("inf")}, "max_hold_s"),
    ({"max_hold_s": float("nan")}, "max_hold_s"),
    ({"startup_timeout_s": float("inf")}, "startup_timeout_s"),
    ({"startup_timeout_s": float("nan")}, "startup_timeout_s"),
    ({"drain_timeout_s": 100_000_000}, "drain_timeout_s"),
])
def test_bad_deploy_configs_are_refused(tmp_path, over, match):
    doc = {"service_root": str(tmp_path / "svc"), "gateway": {"active_class": "x"}, **over}
    with pytest.raises(deploy.DeployError, match=match):
        deploy.DeployConfig.checked(doc)


def test_load_config_reports_the_file(tmp_path):
    path = tmp_path / "d.toml"
    path.write_text("[deploy]\nvllm_port = 8000\n[gateway]\nactive_class = 'x'\n")
    with pytest.raises(deploy.DeployError, match=str(path)):
        deploy.load_config(path)
    path.write_text("[deploy\n")
    with pytest.raises(deploy.DeployError, match="cannot read"):
        deploy.load_config(path)


def test_an_infinite_hold_in_the_toml_is_refused(tmp_path):
    """`max_hold_s = inf` would make `up` a resident daemon holding the shared lock forever (029: a user decision)."""
    path = tmp_path / "d.toml"
    for key in ("max_hold_s", "startup_timeout_s"):
        path.write_text(f"[deploy]\n{key} = inf\n[gateway]\nactive_class = 'x'\n")
        with pytest.raises(deploy.DeployError, match=key):
            deploy.load_config(path)


def test_gateway_settings_are_owned_by_the_deploy(world):
    cfg = world.cfg(gateway={"active_class": "test", "max_queue_wait_s": 1.5, "tokens": {"chat_overhead": 40}})
    doc = deploy.gateway_settings(cfg)
    assert doc == {"active_class": "test", "max_queue_wait_s": 1.5, "tokens": {"chat_overhead": 40},
                   "backend_url": "http://127.0.0.1:8100", "host": "127.0.0.1", "port": 8080, "drain_timeout_s": 60}
    assert tomllib.loads(deploy.toml_dumps(doc)) == doc
    Settings.model_validate(doc)


def test_toml_dumps_round_trips_strings_with_quotes_and_unicode():
    doc = {"a": 'x"y\\z\n', "b": True, "c": 2, "d": 0.25, "t": {"u": "測試"}}
    assert tomllib.loads(deploy.toml_dumps(doc)) == doc


# ------------------------------------------------------------------ render --

def test_render_is_the_stage_b_server_on_the_product_port(world):
    cfg = world.cfg()
    argv, env = deploy.render_vllm(PINS, cfg, {"PATH": "/usr/bin", "PYTHONPATH": "x"})
    want, _ = bp.server_argv(PINS, gdn_prefill_backend=SB["gdn_prefill_backend"],
                             gpu_memory_utilization=SB["gpu_memory_utilization"], port=8100,
                             vllm_bin="/home/wenn0113/llm-serving-lab/.venv/bin/vllm",
                             kv_cache_memory_bytes=SB["kv_cache_memory_bytes"])
    assert argv == want and argv[argv.index("--port") + 1] == "8100"
    assert env == procs.vllm_env({"PATH": "/usr/bin", "PYTHONPATH": "x"}, cfg, pins=PINS)
    assert deploy.fingerprint(argv, env) == stage_b_fingerprint(PINS)


def test_render_cli_prints_argv_recorded_env_and_fingerprint(capsys):
    assert deploy.main(["render"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["argv"][0] == "/home/wenn0113/llm-serving-lab/.venv/bin/vllm" and "8100" in out["argv"]
    assert out["fingerprint"] == stage_b_fingerprint(PINS)
    assert out["recorded_env"]["PATH_PREFIX"] == bp.VENV_BIN and "HOME" not in out["recorded_env"]


def test_no_formal_option_anywhere():
    for parser in deploy.parsers():
        for action in parser._actions:
            assert not any("formal" in s for s in action.option_strings), action.option_strings


def test_stage_b_values_must_be_measured():
    pins = {**PINS, "stage_b": {**SB, "kv_cache_memory_bytes": 0, "num_gpu_blocks": 0}}
    problems = deploy.stage_b_problems(pins)
    assert any("kv_cache_memory_bytes" in p for p in problems) and any("num_gpu_blocks" in p for p in problems)
    assert deploy.stage_b_problems(PINS) == []


# ------------------------------------------------------------------ release --

def test_a_clean_release_verifies(world):
    sha, problems = deploy.release_problems(world.release, world.cfg())
    assert (sha, problems) == (SHA, [])


def test_a_hot_patched_or_missing_release_file_is_refused(world):
    (world.release / "service" / "capserve" / "x.py").write_text("print(2)\n")
    (world.release / "benchmarks" / "y.py").unlink()
    _, problems = deploy.release_problems(world.release, world.cfg())
    assert any("service/capserve/x.py" in p for p in problems) and any("benchmarks/y.py" in p for p in problems)


def test_a_release_outside_releases_or_misnamed_is_refused(world, tmp_path):
    other = make_release(tmp_path / "elsewhere")
    _, problems = deploy.release_problems(other, world.cfg())
    assert any("releases" in p for p in problems)
    misnamed = world.svc / "releases" / ("b" * 40)
    misnamed.mkdir(parents=True)
    for f in ("DEPLOYED_COMMIT", "DEPLOY_MANIFEST.json"):
        (misnamed / f).write_bytes((world.release / f).read_bytes())
    _, problems = deploy.release_problems(misnamed, world.cfg())
    assert any("name" in p for p in problems)


def test_a_release_without_manifest_is_refused(world):
    (world.release / "DEPLOY_MANIFEST.json").unlink()
    _, problems = deploy.release_problems(world.release, world.cfg())
    assert any("DEPLOY_MANIFEST.json" in p for p in problems)


# ------------------------------------------------------------------ preflight refusals --

@pytest.mark.parametrize("setup, match", [
    (lambda w: setattr(w, "lock_held", False), "GPU lock"),
    (lambda w: setattr(w, "fd", None), "descriptor"),
    (lambda w: setattr(w, "idle", [(500, 0)] * 12 + [(500, 37), (500, 0)]), "utilization.gpu up to 37 %"),
    (lambda w: setattr(w, "idle", [(500, 0)] * 12 + [(600, 0)]), "memory.used range 100 MiB"),
    (lambda w: setattr(w, "apps", ["4242, ollama, 900"]), "compute process"),
    (lambda w: setattr(w, "idle", [(2049, 0)]), "2048 MiB"),           # 031 addendum: another tenant's memory
    (lambda w: setattr(w, "idle", [(15900, 0)]), "2048 MiB"),
    (lambda w: setattr(w, "idle", [(500, 0)] * 3 + [RuntimeError("nvidia-smi hung")]), "nvidia-smi"),
    (lambda w: setattr(w, "toolchain", ["no ninja on PATH"]), "ninja"),
    (lambda w: w.busy_ports.add(8100), "8100"),
    (lambda w: w.busy_ports.add(8080), "8080"),
    (lambda w: (w.release / "benchmarks" / "y.py").write_text("patched"), "y.py"),
])
def test_preflight_refusals_start_nothing_and_create_no_run(world, prof, capsys, setup, match):
    setup(world)
    assert up(world, prof) == 2
    assert world.started == [] and not (world.svc / "runs").exists()
    state = deploy.read_state(world.cfg())
    if world.lock_held:                                # the lock holder named itself in state.json for the preflight
        assert state["phase"] == "refused" and state["supervisor_pid"] == 4321 and state["run_id"] == RUN_ID
        assert any(match in p for p in state["problems"])
    else:
        assert state is None
    assert world.shared_reads == []                    # the pre-boot reading is taken only for a boot
    assert match in capsys.readouterr().err


def test_stop_during_the_031_preflight_stops_the_supervisor_before_anything_starts(world, prof, tmp_path):
    """The 031 idle window makes the preflight last 30 s or more: state.json names the supervisor as soon as it holds
    the lock, so a `stop` meanwhile signals it (SIGTERM -> SystemExit, main()'s handler) and nothing starts."""
    cfg = world.cfg()
    proc_root = tmp_path / "proc"
    write_cmdline(proc_root, 4321, SUPERVISOR)
    seen, kills = {}, []

    def kill(pid, sig):
        kills.append((pid, sig))
        shutil.rmtree(proc_root / str(pid))            # the supervisor exits on it (below)

    def sample():
        if len(world.idle_times) == 10:                # 10 s into the window
            seen["phase"] = (deploy.read_state(cfg) or {}).get("phase")
            seen["stop"] = deploy.stop(cfg, proc_root=str(proc_root), kill=kill, now=world.now, sleep=world.sleep)
            if kills:
                raise SystemExit(128 + signal.SIGTERM)   # coreb_smoke._exit_on_signal, as main() installs it
        return world.gpu_sample()

    with pytest.raises(SystemExit) as info:
        up(world, prof, cfg=cfg, fx=world.fx(gpu_sample=sample))
    assert info.value.code == 143
    assert seen == {"phase": "preflight", "stop": 0} and kills == [(4321, signal.SIGTERM)]
    assert world.started == [] and world.shared_reads == [] and not (world.svc / "runs").exists()
    state = world.state()
    assert state["phase"] == "stopped" and state["supervisor_pid"] == 4321 and state["run_id"] == RUN_ID
    assert state["interrupted"] == "SystemExit(143)"


def test_a_stop_during_the_pre_boot_shared_usage_reading_starts_nothing(world, prof):
    def interrupted():
        raise SystemExit(128 + signal.SIGTERM)

    with pytest.raises(SystemExit):
        up(world, prof, fx=world.fx(shared_usage=interrupted))
    assert world.started == [] and world.state()["phase"] == "stopped"


def test_an_up_without_the_lock_never_touches_a_running_supervisors_state(world, prof):
    """Only the lock holder writes state.json: a refused `up` without the lock leaves the running supervisor's."""
    cfg = world.cfg()
    deploy.write_state(cfg, {"phase": "ready", "supervisor_pid": 555}, wall=lambda: NOW)
    world.lock_held = False
    assert up(world, prof) == 2
    assert world.state()["phase"] == "ready" and world.state()["supervisor_pid"] == 555


def test_a_failing_preflight_state_write_refuses(world, prof, monkeypatch, capsys):
    """Without state.json naming it, `stop` could not find the supervisor: refused, nothing started."""
    failing_state_write(monkeypatch, lambda st: st["phase"] == "preflight")
    assert up(world, prof) == 2
    assert world.started == [] and "No space left" in capsys.readouterr().err


def test_preflight_samples_a_30s_window_at_1hz_and_records_the_baseline(world, prof):
    """031 product preflight: 30 nvidia-smi samples at 1 Hz before anything starts; the idle baseline (the window's
    largest memory.used) is recorded and is what the teardown returns to."""
    world.idle = [(480, 0), (512, 3), (500, 5)]
    assert up(world, prof) == 0
    assert world.idle_times == [float(i) for i in range(30)]
    idle = runs.read_meta(world.run_dir())["preflight"]["gpu_idle"]
    assert idle["decision"] == "031" and idle["baseline_mib"] == 512 and idle["warnings"] == []
    assert idle["bound_mib"] == SB["v_total_mib"] - gi.F_PEAK_MIB - gi.MARGIN_MIB == 1018
    assert idle["return_limit_mib"] == 512 + 64
    window = idle["window"]
    assert window["n"] == 30 and window["mem_range_mib"] == 32 and window["util_max_pct"] == 5
    assert window["ok"] is True and window["problems"] == [] and window["compute_apps"] == []


def test_an_idle_value_at_the_product_cap_still_boots_with_a_warning(world, prof):
    """031 addendum: up to 2,048 MiB idle (above the 1,018 MiB headroom bound) a product test boots with a warning."""
    world.idle, world.gpu = [(2048, 0)], [2048 + 64]
    assert up(world, prof) == 0
    assert any("1018" in w for w in runs.read_meta(world.run_dir())["warnings"])


def test_an_idle_value_above_the_headroom_bound_only_warns(world, prof, capsys):
    """031: product tests check function, not latency, so a headroom shortfall (idle 1,385 MiB > 1,018) is a warning
    in meta.json and on stderr, never a refusal; the teardown returns to that baseline + 64 MiB, not to 1,000."""
    world.idle, world.gpu = [(1385, 0)], [1449]
    assert up(world, prof) == 0
    meta = runs.read_meta(world.run_dir())
    (warning,) = meta["warnings"]
    assert "1385" in warning and "1018" in warning and meta["preflight"]["gpu_idle"]["warnings"] == [warning]
    assert meta["preflight"]["gpu_idle"]["baseline_mib"] == 1385
    td = meta["outcome"]["teardown"]
    assert (td["gpu_baseline_mib"], td["gpu_return_limit_mib"], td["gpu_mem_after_mib"]) == (1385, 1449, 1449)
    assert meta["outcome"]["problems"] == [] and td["problems"] == []
    assert "warning" in capsys.readouterr().err


def test_pins_without_v_total_are_refused(world, prof, capsys):
    pins = {**PINS, "stage_b": {k: v for k, v in SB.items() if k != "v_total_mib"}}
    assert up(world, prof, pins=pins) == 2
    assert world.started == [] and "v_total_mib" in capsys.readouterr().err


def test_a_provisional_profile_needs_the_flag(world, prof, capsys):
    assert up(world, prof, allow_provisional=False) == 2
    assert world.started == [] and "--allow-provisional" in capsys.readouterr().err


def test_an_unknown_active_class_is_refused_before_any_boot(world, prof, capsys):
    assert up(world, prof, cfg=world.cfg(gateway={"active_class": "nope"})) == 2
    assert world.started == [] and "nope" in capsys.readouterr().err


def test_a_verified_profile_with_another_fingerprint_is_refused(world, capsys):
    doc = verified_doc(fingerprint="e" * 64)
    path = add_to_release(world.release, "service/profiles/v.json", json.dumps(doc).encode())
    assert up(world, path, allow_provisional=False, cfg=world.cfg(**VERIFIED_ACTIVE)) == 2
    err = capsys.readouterr().err
    assert world.started == [] and "fingerprint" in err and "DEPLOY_MANIFEST.json" not in err


def test_a_verified_profile_is_checked_against_the_frozen_configs(world, capsys):
    """029 M2: the loader checks a verified profile's provenance at run time; deploy passes its frozen configs on."""
    path = add_to_release(world.release, "service/profiles/v.json", json.dumps(verified_doc()).encode())
    assert up(world, path, allow_provisional=False, cfg=world.cfg(**VERIFIED_ACTIVE),
              frozen=frozen_configs(frozen_capacity(capacity=False))) == 2
    assert world.started == [] and "[freeze] capacity" in capsys.readouterr().err
    assert up(world, path, allow_provisional=False, cfg=world.cfg(**VERIFIED_ACTIVE),
              frozen=frozen_configs(pins_sha256="00" * 32)) == 2
    assert world.started == [] and "pins_sha256" in capsys.readouterr().err


# ------------------------------------------------------------------ verified profiles (027 §4) --

def test_a_handmade_verified_profile_outside_the_release_is_refused(world, tmp_path, capsys):
    """027 §4: verified only from the generator. Deploy serves a verified profile only as a committed release file
    (service/profiles/<name>.json listed in DEPLOY_MANIFEST.json), which the Mac test that regenerates every committed
    verified profile covers; a hand-written one anywhere else is refused before anything starts."""
    (tmp_path / "anywhere").mkdir()
    path = write_json(tmp_path / "anywhere" / "handmade.json", verified_doc())
    assert up(world, path, allow_provisional=False, cfg=world.cfg(**VERIFIED_ACTIVE)) == 2
    assert world.started == [] and not (world.svc / "runs").exists()
    assert "DEPLOY_MANIFEST.json" in capsys.readouterr().err


def test_a_verified_profile_in_the_release_but_not_in_its_manifest_is_refused(world, capsys):
    path = world.release / "service" / "profiles" / "uncommitted.json"
    path.parent.mkdir(parents=True)
    write_json(path, verified_doc())
    assert up(world, path, allow_provisional=False, cfg=world.cfg(**VERIFIED_ACTIVE)) == 2
    assert world.started == [] and "DEPLOY_MANIFEST.json" in capsys.readouterr().err


@pytest.mark.parametrize("rel", ["service/profiles/verified/forged.json", "service/deploy/forged.json",
                                 "service/profiles/forged.JSON", "forged.json", "service/profiles/.json"])
def test_a_committed_verified_profile_elsewhere_than_service_profiles_is_refused(world, capsys, rel):
    """Review finding: a verified profile is served only from service/profiles/<name>.json (top level, lowercase
    .json), never from any other committed path of the release."""
    path = add_to_release(world.release, rel, json.dumps(verified_doc()).encode())
    assert deploy.verified_profile_problems(path, path.read_bytes(), world.release) != []
    assert up(world, path, allow_provisional=False, cfg=world.cfg(**VERIFIED_ACTIVE)) == 2
    assert world.started == [] and "service/profiles/<name>.json" in capsys.readouterr().err


def test_a_committed_verified_profile_is_served_as_verified(world):
    path = add_to_release(world.release, "service/profiles/verified-test.json", json.dumps(verified_doc()).encode())
    assert up(world, path, allow_provisional=False, cfg=world.cfg(**VERIFIED_ACTIVE)) == 0
    (gw,) = [s for s in world.started if s["name"] == "gateway"]
    assert "--allow-provisional" not in gw["argv"]
    assert runs.read_meta(world.run_dir())["profile"]["status"] == "verified"


@pytest.mark.parametrize("cls, match", [
    (verified_class(workload="U8K", status="UNSTABLE"), "unavailable"),
    (verified_class(workload="U8K", status="NT"), "unavailable"),
    (verified_class(workload="M8K", status="INFEASIBLE", value=0), "class_not_serviceable"),
])
def test_an_active_class_without_a_verified_limit_is_refused_before_any_boot(world, capsys, cls, match):
    """030 8.7: an unavailable class is never served, and a verified 0 has nothing to serve (its gateway answers 503
    class_not_serviceable and is never ready), so neither boots vLLM; no flag overrides it."""
    doc = verified_doc(classes=[cls])
    path = add_to_release(world.release, "service/profiles/v.json", json.dumps(doc).encode())
    assert up(world, path, allow_provisional=True, cfg=world.cfg(gateway={"active_class": cls["name"]})) == 2
    assert world.started == [] and not (world.svc / "runs").exists()
    assert match in capsys.readouterr().err


def test_profile_server_geometry_must_equal_the_pins(world, tmp_path, capsys):
    path = write_json(tmp_path / "p.json", profile_doc(blocks=186, block_size=528))
    assert up(world, path) == 2
    assert world.started == [] and "num_gpu_blocks" in capsys.readouterr().err


# ------------------------------------------------------------------ the happy path (hold) --

def test_up_holds_until_max_hold_then_tears_down_in_order(world, prof):
    assert up(world, prof) == 0
    vllm, gw = world.started
    cfg = world.cfg()
    argv, env = deploy.render_vllm(PINS, cfg, world.fx().base_env)
    assert vllm["argv"] == argv and vllm["env"] == env and vllm["keep_fds"] == (9,)
    assert "CAPSERVE_API_KEY" not in vllm["env"] and vllm["env"]["HOME"] == str(world.svc / ".home")
    assert vllm["log"] == world.run_dir() / "vllm.log" and Path(vllm["cwd"]) == world.run_dir()
    gw_toml = world.run_dir() / "gateway.toml"
    assert gw["argv"] == ["/venv/bin/python", "-m", "capserve", "serve", "--config", str(gw_toml),
                          "--profile", str(world.run_dir() / "profile.json"), "--allow-provisional"]
    assert (world.run_dir() / "profile.json").read_bytes() == Path(prof).read_bytes()
    assert gw["keep_fds"] == (9,) and gw["env"]["CAPSERVE_API_KEY"] == "k"
    assert gw["env"]["PYTHONPATH"].split(":") == [str(world.release / "service"), str(world.release / "benchmarks")]
    assert tomllib.loads(gw_toml.read_text()) == deploy.gateway_settings(cfg)
    assert (world.svc / ".home").is_dir() and (world.svc / ".tmp").is_dir()
    # teardown: the gateway first (SIGTERM only), then the vLLM group (SIGINT first), never the reverse
    assert world.events == [("start", "vllm"), ("start", "gateway"), ("signal", "gateway", signal.SIGTERM),
                            ("signal", "vllm", signal.SIGINT)]
    meta = runs.read_meta(world.run_dir())
    assert meta["purpose"] == "product-test" and meta["formal"] is False and meta["research_valid"] is False
    assert meta["release"] == SHA and meta["profile"]["status"] == "provisional" and meta["mode"] == "hold"
    assert meta["fingerprint"] == stage_b_fingerprint(PINS)
    out = meta["outcome"]
    assert out["ok"] is True and out["problems"] == [] and out["exit"] == 0 and out["interrupted"] is None
    assert out["workload"]["ended_by"] == "max_hold_s"
    assert out["server"]["cache_config"] == {"num_gpu_blocks": 248, "block_size": 528}
    assert out["server"]["nondefault"]["max_num_batched_tokens"] == 2048
    assert out["teardown"]["gpu_mem_after_mib"] == 500 and out["teardown"]["gateway"]["returncode"] == -15
    assert (out["teardown"]["gpu_baseline_mib"], out["teardown"]["gpu_return_limit_mib"]) == (500, 564)
    assert meta["warnings"] == []
    state = world.state()
    assert state["phase"] == "stopped" and state["supervisor_pid"] == 4321 and state["run_id"] == RUN_ID
    assert state["vllm_pid"] == 1000 and state["gateway_pid"] == 1001


def test_state_is_ready_while_the_workload_runs(world, prof):
    seen = {}

    def workload(run):
        seen["state"] = world.state()
        seen["urls"] = (run.gateway_url, run.vllm_url, run.run_dir)
        return {"ok": True, "problems": [], "ended_by": "workload"}

    assert up(world, prof, workload=workload) == 0
    assert seen["state"]["phase"] == "ready" and seen["state"]["gateway_url"] == "http://127.0.0.1:8080"
    assert seen["urls"] == ("http://127.0.0.1:8080", "http://127.0.0.1:8100", world.run_dir())


def test_shared_usage_is_read_before_the_boot_once_vllm_is_ready_and_after_the_workload(world, prof):
    """031 (descriptive for product tests): the Windows shared usage before vLLM starts, once vLLM is ready (before
    the gateway starts) and after the workload (the gateway drained, vLLM still holding its memory), with the change
    since the pre-boot reading; meta.json has the pre-boot reading from the start and all three at the end."""
    world.shared = [dict(SHARED), {**SHARED, "shared_mib": 70.1}, {**SHARED, "shared_mib": 180.9}]
    seen = {}

    def workload(run):
        seen["meta"] = runs.read_meta(run.run_dir)
        return {"ok": True, "problems": [], "ended_by": "workload"}

    assert up(world, prof, workload=workload) == 0
    assert world.events == [("start", "vllm"), ("start", "gateway"), ("signal", "gateway", signal.SIGTERM),
                            ("signal", "vllm", signal.SIGINT)]
    assert world.shared_reads == [0, 1, 3]       # before vLLM; before the gateway; after the gateway, before vLLM
    assert seen["meta"]["shared_usage"] == [{"when": "pre-boot", **SHARED, "error": None}]
    meta = runs.read_meta(world.run_dir())
    assert [(r["when"], r["shared_mib"], r.get("delta_mib")) for r in meta["shared_usage"]] == [
        ("pre-boot", 64.4, None), ("ready", 70.1, 5.7), ("after", 180.9, 116.5)]
    assert meta["outcome"]["ok"] is True and meta["outcome"]["problems"] == []   # descriptive: never a verdict


def test_a_failed_shared_usage_reading_is_recorded_not_fatal(world, prof):
    world.shared = [gi.SharedUsageError("powershell.exe exited 1"), dict(SHARED), TimeoutError("20 s")]
    assert up(world, prof) == 0
    readings = runs.read_meta(world.run_dir())["shared_usage"]
    assert [r["when"] for r in readings] == ["pre-boot", "ready", "after"]
    assert "powershell.exe exited 1" in readings[0]["error"] and "TimeoutError" in readings[2]["error"]
    assert readings[1]["error"] is None and "delta_mib" not in readings[1]   # nothing to compare with


def test_a_shared_usage_reading_of_another_adapter_has_no_delta(world, prof):
    world.shared = [dict(SHARED), {**SHARED, "luid": "luid_0x0_0x2_phys_0", "shared_mib": 900.0}]
    assert up(world, prof) == 0
    ready = runs.read_meta(world.run_dir())["shared_usage"][1]
    assert ready["luid"] == "luid_0x0_0x2_phys_0" and "delta_mib" not in ready


def test_no_after_reading_when_vllm_never_got_ready(world, prof):
    world.health_after = 10 ** 9
    assert up(world, prof, cfg=world.cfg(startup_timeout_s=30.0)) == 1
    assert [r["when"] for r in runs.read_meta(world.run_dir())["shared_usage"]] == ["pre-boot"]


def test_a_loadtest_names_the_run_and_a_failed_workload_exits_1(world, prof):
    def workload(run):
        return {"ok": False, "problems": ["a 502 was seen"], "ended_by": "loadtest"}

    assert up(world, prof, loadtest="smoke", workload=workload) == 1
    d = world.svc / "runs" / f"{RUN_ID}-smoke"
    meta = runs.read_meta(d)
    assert meta["mode"] == "loadtest:smoke" and meta["outcome"]["problems"] == ["a 502 was seen"]
    assert world.events[-1] == ("signal", "vllm", signal.SIGINT)   # torn down anyway


def test_the_default_workload_of_a_loadtest_is_the_loadtest_module(world, prof, monkeypatch):
    calls = []

    def fake_run(scenario, **kw):
        calls.append((scenario, kw))
        return {"ok": True, "problems": []}

    monkeypatch.setattr(deploy.loadtest, "run_scenario", fake_run)
    assert up(world, prof, loadtest="overload") == 0
    scenario, kw = calls[0]
    assert scenario == "overload" and kw["gateway_url"] == "http://127.0.0.1:8080"
    assert kw["vllm_url"] == "http://127.0.0.1:8100" and kw["run_dir"] == world.svc / "runs" / f"{RUN_ID}-overload"
    assert kw["api_key"] == "k" and 0 < kw["time_limit_s"] <= 100


def test_the_gateway_serves_the_bytes_preflight_checked(world, prof):
    """The profile is read once: the gateway gets runs/<id>/profile.json (those bytes), so a profile file swapped
    during the boot is neither served nor described by meta.json."""
    original = Path(prof).read_bytes()
    other = profile_doc()
    other["classes"][0].update(max_concurrency=128, max_queue=512)
    other["server"]["fingerprint_sha256"] = "f" * 64          # a server preflight would refuse
    other["content_sha256"] = content_sha256(other)

    def swap(w):
        if "vllm" in w.procs and "gateway" not in w.procs:
            Path(prof).write_text(json.dumps(other))

    world.on_sleep = swap
    assert up(world, prof) == 0
    (gw,) = [s for s in world.started if s["name"] == "gateway"]
    served = Path(gw["argv"][gw["argv"].index("--profile") + 1])
    assert served == world.run_dir() / "profile.json" and served.read_bytes() == original
    assert runs.read_meta(world.run_dir())["profile"]["content_sha256"] == json.loads(original)["content_sha256"]


def test_a_gateway_serving_other_limits_than_preflight_checked_is_torn_down_with_exit_2(world, prof):
    served = {"ready": True, "active_class": "test",
              "profile": {"id": "provisional-test", "status": "provisional", "provisional": True},
              "limits": {"max_concurrency": 128, "max_queue": 512, "max_prompt_tokens": 1920, "max_output_tokens": 128}}
    world.readyz_body = served
    assert up(world, prof) == 2
    assert world.events[-2:] == [("signal", "gateway", signal.SIGTERM), ("signal", "vllm", signal.SIGINT)]
    out = runs.read_meta(world.run_dir())["outcome"]
    assert out["exit"] == 2 and any("readyz" in p and "128" in p for p in out["problems"])
    assert out["workload"] is None


def test_max_hold_s_bounds_a_slow_vllm_boot(world, prof):
    """max_hold_s bounds how long the shared lock is held before the teardown starts, the boot included."""
    world.health_after = 300           # healthy only ~600 s after the start (startup_timeout_s is 900)
    reached = []
    assert up(world, prof, cfg=world.cfg(max_hold_s=100.0),
              workload=lambda run: reached.append(run) or {"ok": True, "problems": []}) == 1
    out = runs.read_meta(world.run_dir())["outcome"]
    assert reached == [] and out["timeline"]["teardown_start_s"] <= 100 + deploy.HEALTH_POLL_S
    assert any("max_hold_s" in p for p in out["problems"])
    assert [e for e in world.events if e[0] == "start"] == [("start", "vllm")]


def test_max_hold_s_bounds_the_gateway_readiness_wait(world, prof):
    world.readyz = 503                 # never ready: GATEWAY_READY_TIMEOUT_S (120 s) would end the wait at ~156 s
    assert up(world, prof, cfg=world.cfg(max_hold_s=60.0)) == 1   # the 30 s idle window counts: the lock is held
    out = runs.read_meta(world.run_dir())["outcome"]
    assert out["timeline"]["teardown_start_s"] <= 60 + deploy.HEALTH_POLL_S
    assert any("readyz" in p and "max_hold_s" in p for p in out["problems"])


# ------------------------------------------------------------------ failures after the start --

def test_vllm_never_healthy_tears_down_and_exits_1(world, prof):
    world.health_after = 10 ** 9
    assert up(world, prof, cfg=world.cfg(startup_timeout_s=30.0)) == 1
    assert [e for e in world.events if e[0] == "start"] == [("start", "vllm")]
    assert ("signal", "vllm", signal.SIGINT) in world.events
    out = runs.read_meta(world.run_dir())["outcome"]
    assert out["ok"] is False and any("health" in p for p in out["problems"])


def test_vllm_dying_during_startup_is_reported_at_once(world, prof):
    def die(w):
        if "vllm" in w.procs and w.procs["vllm"].alive:
            w.procs["vllm"].die(1)

    world.health_after, world.on_sleep = 10 ** 9, die
    assert up(world, prof) == 1
    out = runs.read_meta(world.run_dir())["outcome"]
    assert any("exited" in p for p in out["problems"]) and world.t < 100


@pytest.mark.parametrize("change, match", [
    (dict(metrics='vllm:cache_config_info{block_size="528",num_gpu_blocks="186"} 1.0\n'), "num_gpu_blocks"),
    (dict(metrics=""), "cache_config_info"),
    (dict(vllm_log=LOG_OK.replace("FLASH_ATTN", "FLASHINFER")), "FLASH_ATTN"),
    (dict(vllm_log=LOG_OK.replace("Triton/FLA", "FlashInfer")), "GDN prefill"),
    (dict(vllm_log=LOG_OK.replace("'max_num_batched_tokens': 2048", "'max_num_batched_tokens': 8192")), "2048"),
    (dict(vllm_log=LOG_OK.split("\n", 1)[1]), "non-default args"),
])
def test_a_server_that_is_not_the_core_b_server_is_torn_down_with_exit_2(world, prof, change, match):
    for k, v in change.items():
        setattr(world, k, v)
    assert up(world, prof) == 2
    assert [e for e in world.events if e[0] == "start"] == [("start", "vllm")]   # no gateway
    out = runs.read_meta(world.run_dir())["outcome"]
    assert any(match in p for p in out["problems"]) and out["exit"] == 2


def test_gateway_never_ready_tears_both_down(world, prof):
    world.readyz = 503
    assert up(world, prof) == 1
    assert ("signal", "gateway", signal.SIGTERM) in world.events and ("signal", "vllm", signal.SIGINT) in world.events
    assert any("readyz" in p for p in runs.read_meta(world.run_dir())["outcome"]["problems"])


def test_a_child_exiting_while_holding_ends_the_hold(world, prof):
    def die(w):
        if w.t >= 50 and w.procs.get("gateway") and w.procs["gateway"].alive:
            w.procs["gateway"].die(1)

    world.on_sleep = die
    assert up(world, prof) == 1
    out = runs.read_meta(world.run_dir())["outcome"]
    assert out["workload"]["ended_by"] == "gateway_exit" and any("gateway exited" in p for p in out["problems"])


def test_an_interrupt_while_holding_tears_down_and_propagates(world, prof):
    def interrupt(w):
        if w.t >= 30 and w.procs.get("gateway"):
            w.on_sleep = None
            raise SystemExit(128 + signal.SIGTERM)

    world.on_sleep = interrupt
    with pytest.raises(SystemExit) as info:
        up(world, prof)
    assert info.value.code == 143
    assert world.events[-2:] == [("signal", "gateway", signal.SIGTERM), ("signal", "vllm", signal.SIGINT)]
    out = runs.read_meta(world.run_dir())["outcome"]
    assert out["interrupted"] == "SystemExit(143)" and out["ok"] is False and world.state()["phase"] == "stopped"


def test_an_unexpected_error_still_tears_down(world, prof):
    def workload(run):
        raise RuntimeError("bug in a workload")

    with pytest.raises(RuntimeError):
        up(world, prof, workload=workload)
    assert world.events[-1] == ("signal", "vllm", signal.SIGINT)
    assert "RuntimeError" in runs.read_meta(world.run_dir())["outcome"]["interrupted"]


# ------------------------------------------------------------------ bookkeeping never skips the teardown --

def failing_state_write(monkeypatch, when):
    real = deploy.write_state

    def flaky(cfg, state, *, wall=deploy.utcnow):
        if when(state):
            raise OSError(28, "No space left on device")
        return real(cfg, state, wall=wall)

    monkeypatch.setattr(deploy, "write_state", flaky)


def test_a_failing_stopping_state_write_still_tears_down_in_order(world, prof, monkeypatch):
    """029 step 7: the teardown always runs; a state.json write that fails (ENOSPC, EACCES) is only a problem."""
    failing_state_write(monkeypatch, lambda st: st["phase"] == "stopping")
    assert up(world, prof) == 1
    assert world.events == [("start", "vllm"), ("start", "gateway"), ("signal", "gateway", signal.SIGTERM),
                            ("signal", "vllm", signal.SIGINT)]
    out = runs.read_meta(world.run_dir())["outcome"]
    assert out["teardown"]["gpu_mem_after_mib"] == 500 and out["exit"] == 1 and out["ok"] is False
    assert any("state.json" in p and "No space left" in p for p in out["problems"])
    assert world.state()["phase"] == "stopped"


def test_a_failing_state_write_after_the_vllm_start_still_tears_vllm_down(world, prof, monkeypatch):
    failing_state_write(monkeypatch, lambda st: st.get("vllm_pid") is not None)   # every later write fails too
    with pytest.raises(OSError):
        up(world, prof)
    assert world.events == [("start", "vllm"), ("signal", "vllm", signal.SIGINT)]
    out = runs.read_meta(world.run_dir())["outcome"]
    assert "OSError" in out["interrupted"] and out["teardown"]["gpu_mem_after_mib"] == 500


def test_a_failing_meta_write_neither_skips_the_teardown_nor_replaces_the_interrupt(world, prof, monkeypatch,
                                                                                     capsys):
    def no_meta(*a, **kw):
        raise OSError(13, "Permission denied")

    def interrupt(w):
        if w.t >= 30 and w.procs.get("gateway"):
            w.on_sleep = None
            raise SystemExit(128 + signal.SIGTERM)

    monkeypatch.setattr(deploy.runs, "update_meta", no_meta)
    world.on_sleep = interrupt
    with pytest.raises(SystemExit) as info:
        up(world, prof)
    assert info.value.code == 143
    assert world.events[-2:] == [("signal", "gateway", signal.SIGTERM), ("signal", "vllm", signal.SIGINT)]
    assert world.state()["phase"] == "stopped"
    err = capsys.readouterr().err
    assert "meta.json" in err and "Permission denied" in err


def test_a_failing_meta_write_after_a_clean_run_exits_1(world, prof, monkeypatch, capsys):
    def no_meta(*a, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(deploy.runs, "update_meta", no_meta)
    assert up(world, prof) == 1
    assert world.events[-2:] == [("signal", "gateway", signal.SIGTERM), ("signal", "vllm", signal.SIGINT)]
    assert "No space left" in capsys.readouterr().err


# ------------------------------------------------------------------ teardown checks --

def test_a_gateway_that_ignores_sigterm_is_killed_after_drain_plus_10s(world, prof):
    t_before = {}

    def workload(run):
        run.gateway.dies_on = set()
        t_before["t"] = world.t
        return {"ok": True, "problems": [], "ended_by": "workload"}

    assert up(world, prof, workload=workload) == 1
    gw_signals = [e[2] for e in world.events if e[:2] == ("signal", "gateway")]
    assert gw_signals == [signal.SIGTERM, signal.SIGKILL]
    td = runs.read_meta(world.run_dir())["outcome"]["teardown"]
    assert any("killed" in p for p in td["problems"])
    assert world.t - t_before["t"] >= 60 + 10


def test_gpu_memory_not_returning_to_the_baseline_is_a_problem(world, prof):
    """031: back to <= the pre-boot baseline (500) + 64 MiB within 60 s; 900 MiB passed the old 1,000 rule."""
    world.gpu = [900] * 1000
    assert up(world, prof) == 1
    td = runs.read_meta(world.run_dir())["outcome"]["teardown"]
    assert td["gpu_mem_after_mib"] == 900 and td["gpu_return_limit_mib"] == 564
    (problem,) = td["problems"]
    assert "900" in problem and "564" in problem and "baseline 500" in problem


def test_gpu_memory_returning_within_60s_is_fine(world, prof):
    world.gpu = [9000, 3000, 564]
    assert up(world, prof) == 0
    assert runs.read_meta(world.run_dir())["outcome"]["teardown"]["gpu_mem_after_mib"] == 564


def test_a_failing_nvidia_smi_after_teardown_is_a_problem(world, prof):
    world.gpu = [RuntimeError("nvidia-smi hung")]
    assert up(world, prof) == 1
    td = runs.read_meta(world.run_dir())["outcome"]["teardown"]
    assert td["gpu_mem_after_mib"] is None and any("nvidia-smi failed" in p for p in td["problems"])


def test_the_stop_wait_covers_the_after_reading():
    cfg = deploy.DeployConfig.model_validate({"gateway": {"active_class": "x"}})
    assert deploy.default_stop_wait_s(cfg) == (cfg.drain_timeout_s + deploy.GATEWAY_KILL_EXTRA_S + deploy.KILL_WAIT_S
                                               + 2 * 30 + deploy.KILL_WAIT_S + gi.SHARED_USAGE_TIMEOUT_S
                                               + deploy.IDLE_RETURN_S + 30)


def test_a_vllm_group_needing_sigkill_is_a_problem(world, prof):
    def stubborn(w):
        v = w.procs.get("vllm")
        if v is not None:
            v.dies_on = set()

    world.on_sleep = stubborn
    assert up(world, prof) == 1
    td = runs.read_meta(world.run_dir())["outcome"]["teardown"]
    assert "server had to be killed" in td["vllm"]["shutdown_reasons"]


# ------------------------------------------------------------------ status / stop --

def write_cmdline(proc_root: Path, pid: int, args):
    (proc_root / str(pid)).mkdir(parents=True)
    (proc_root / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")


SUPERVISOR = ["/venv/bin/python", "-m", "capserve.deploy", "up", "--config", "rtx.toml"]


def test_stop_signals_only_a_verified_supervisor(world, tmp_path, capsys):
    cfg = world.cfg()
    deploy.write_state(cfg, {"phase": "ready", "supervisor_pid": 555}, wall=lambda: NOW)
    proc_root = tmp_path / "proc"
    write_cmdline(proc_root, 555, SUPERVISOR)
    kills = []

    def kill(pid, sig):
        kills.append((pid, sig))
        import shutil
        shutil.rmtree(proc_root / str(pid))

    assert deploy.stop(cfg, proc_root=str(proc_root), kill=kill, now=world.now, sleep=world.sleep) == 0
    assert kills == [(555, signal.SIGTERM)]


def test_stop_refuses_a_reused_pid(world, tmp_path, capsys):
    cfg = world.cfg()
    deploy.write_state(cfg, {"phase": "ready", "supervisor_pid": 556}, wall=lambda: NOW)
    write_cmdline(tmp_path / "proc", 556, ["/usr/bin/python3", "-m", "http.server"])
    kills = []
    assert deploy.stop(cfg, proc_root=str(tmp_path / "proc"), kill=lambda *a: kills.append(a), now=world.now,
                       sleep=world.sleep) == 2
    assert kills == [] and "not a capserve.deploy supervisor" in capsys.readouterr().err


def test_stop_with_nothing_running(world, tmp_path, capsys):
    cfg = world.cfg()
    kills = []
    assert deploy.stop(cfg, proc_root=str(tmp_path / "proc"), kill=lambda *a: kills.append(a)) == 0
    deploy.write_state(cfg, {"phase": "stopped", "supervisor_pid": 557}, wall=lambda: NOW)
    assert deploy.stop(cfg, proc_root=str(tmp_path / "proc"), kill=lambda *a: kills.append(a)) == 0
    assert kills == []


def test_stop_reports_a_supervisor_that_does_not_exit(world, tmp_path):
    cfg = world.cfg()
    deploy.write_state(cfg, {"phase": "ready", "supervisor_pid": 558}, wall=lambda: NOW)
    write_cmdline(tmp_path / "proc", 558, SUPERVISOR)
    assert deploy.stop(cfg, proc_root=str(tmp_path / "proc"), kill=lambda *a: None, now=world.now,
                       sleep=world.sleep, wait_s=30) == 1


def test_status_reports_liveness_and_endpoints(world, tmp_path):
    cfg = world.cfg()
    assert deploy.status(cfg, proc_root=str(tmp_path / "proc"), http_get=world.http_get)["running"] is False
    deploy.write_state(cfg, {"phase": "ready", "supervisor_pid": 559}, wall=lambda: NOW)
    write_cmdline(tmp_path / "proc", 559, SUPERVISOR)
    world.procs["vllm"], world.procs["gateway"] = FakeProc("vllm", 1), FakeProc("gateway", 2)
    world.health_after = 0
    st = deploy.status(cfg, proc_root=str(tmp_path / "proc"), http_get=world.http_get)
    assert st["running"] is True and st["state"]["phase"] == "ready"
    assert st["gateway_ready"] is True and st["vllm_healthy"] is True


def test_state_writes_are_atomic_and_stamped(world):
    cfg = world.cfg()
    deploy.write_state(cfg, {"phase": "starting", "supervisor_pid": 1}, wall=lambda: NOW)
    state = json.loads((world.svc / "state.json").read_text())
    assert state == {"schema_version": 1, "phase": "starting", "supervisor_pid": 1,
                     "updated_utc": "2026-10-16T03:04:05Z"}
    assert not list(world.svc.glob("*.tmp"))


# ------------------------------------------------------------------ production config (036) --

PROD_REL = "service/profiles/verified-prod.json"


def production(world, **gateway):
    """A release holding a committed verified profile, and a deploy config that pins it as its production profile."""
    path = add_to_release(world.release, PROD_REL, json.dumps(verified_doc()).encode())
    return path, world.cfg(production_profile=PROD_REL, gateway=gateway or VERIFIED_ACTIVE["gateway"])


def test_a_production_config_serves_its_pinned_verified_profile(world):
    path, cfg = production(world)
    assert up(world, path, allow_provisional=False, cfg=cfg) == 0
    (gw,) = [s for s in world.started if s["name"] == "gateway"]
    assert "--allow-provisional" not in gw["argv"]
    assert runs.read_meta(world.run_dir())["profile"]["status"] == "verified"


def test_a_production_config_refuses_any_other_profile_even_a_committed_verified_one(world, capsys):
    _, cfg = production(world)
    other = add_to_release(world.release, "service/profiles/verified-other.json", json.dumps(verified_doc()).encode())
    assert up(world, other, allow_provisional=False, cfg=cfg) == 2
    err = capsys.readouterr().err
    assert world.started == [] and not (world.svc / "runs").exists()
    assert "production_profile" in err and PROD_REL in err


def test_a_production_config_refuses_a_provisional_profile_whatever_the_flag(world, prof, capsys):
    _, cfg = production(world, active_class="test")
    assert up(world, prof, allow_provisional=True, cfg=cfg) == 2
    err = capsys.readouterr().err
    assert world.started == [] and "production_profile" in err
    assert "is provisional" in err   # second guard: under a pin the loader never allows a provisional profile


def test_a_production_config_refuses_allow_provisional_even_with_its_own_profile(world, capsys):
    """A CAPSERVE_ALLOW_PROVISIONAL=1 left in the shell (main merges it into allow_provisional) must not slip through:
    a production deploy never runs with provisional profiles allowed."""
    path, cfg = production(world)
    assert up(world, path, allow_provisional=True, cfg=cfg) == 2
    err = capsys.readouterr().err
    assert world.started == [] and "--allow-provisional" in err and "CAPSERVE_ALLOW_PROVISIONAL" in err


def test_deploy_main_merges_a_leftover_allow_provisional_env_into_up(monkeypatch, tmp_path):
    """Review finding (036): `deploy up` under the production config gets allow_provisional=True from a leftover
    CAPSERVE_ALLOW_PROVISIONAL=1 even without the flag, so production_profile_problems refuses it."""
    seen = {}
    monkeypatch.setattr(deploy, "up", lambda cfg, profile, **kw: seen.update(cfg=cfg, **kw) or 2)
    monkeypatch.setattr(deploy.signal, "signal", lambda *a: None)
    config = str(ROOT / "service" / "deploy" / "rtx-verified.toml")
    profile = str(ROOT / "service" / "profiles" / "verified-formal-20261017-c1b.json")
    assert deploy.main(["up", "--config", config, "--profile", profile], env={"CAPSERVE_ALLOW_PROVISIONAL": "1"}) == 2
    assert seen["allow_provisional"] is True and seen["cfg"].production_profile is not None
    assert deploy.main(["up", "--config", config, "--profile", profile], env={}) == 2
    assert seen["allow_provisional"] is False


def test_production_profile_problems_names_every_reason(world):
    path, cfg = production(world)
    assert deploy.production_profile_problems(cfg, path, allow_provisional=False, release_root=world.release) == []
    assert deploy.production_profile_problems(world.cfg(), "/anything.json", allow_provisional=True,
                                              release_root=world.release) == []   # no pin: an integration config
    both = deploy.production_profile_problems(cfg, world.release / "x.json", allow_provisional=True,
                                              release_root=world.release)
    assert len(both) == 2


@pytest.mark.parametrize("bad", ["profiles/x.json", "service/profiles/sub/x.json", "service/profiles/x.JSON",
                                 "/abs/service/profiles/x.json", "service/profiles/.json", "",
                                 "service/profiles/x.json.bak", "service/profiles/x.jsonl"])
def test_production_profile_must_be_a_top_level_service_profiles_file(world, bad):
    with pytest.raises(deploy.DeployError, match="production_profile"):
        deploy.DeployConfig.checked({"service_root": str(world.svc), "production_profile": bad,
                                     "gateway": {"active_class": "uniform-2k"}})


def test_the_repo_production_config_pins_the_final_verified_profile():
    """036: rtx-verified.toml is the production config; it serves only verified-formal-20261017-c1b.json, whose active
    class is verified with a positive limit. rtx.toml (integration tests, provisional-rtx) pins nothing."""
    from capserve.profile import load_profile
    cfg = deploy.load_config(ROOT / "service" / "deploy" / "rtx-verified.toml")
    assert cfg.production_profile == "service/profiles/verified-formal-20261017-c1b.json"
    prof = load_profile(ROOT / cfg.production_profile, allow_provisional=False)
    cls = prof.serving_class(cfg.gateway["active_class"])
    assert prof.verified and cls.serviceable and (cls.name, cls.max_concurrency, cls.bound) == ("uniform-2k", 32, "exact")
    assert deploy.load_config(deploy.DEFAULT_CONFIG).production_profile is None


def test_both_repo_configs_count_prompts_in_the_gateway():
    """038 §4: switched to the local tokenizer after RTX IT-4 (release c9202c5) found every local count equal to
    vLLM's /tokenize; the deploy points it at the snapshot vLLM serves."""
    for name in ("rtx.toml", "rtx-verified.toml"):
        cfg = deploy.load_config(ROOT / "service" / "deploy" / name)
        assert cfg.gateway["tokenizer"] == "local"
        assert deploy.gateway_settings(cfg)["tokenizer_path"] == bp.snapshot_dir(PINS)


# ------------------------------------------------------------------ the committed RTX profile --

def test_the_rtx_profile_is_provisional_conservative_and_for_this_server():
    from capserve.profile import load_profile, ProfileError
    path = ROOT / "service" / "profiles" / "provisional-rtx.json"
    with pytest.raises(ProfileError, match="provisional"):
        load_profile(path, allow_provisional=False)
    prof = load_profile(path, allow_provisional=True)
    assert prof.status == "provisional" and prof.provenance is None and prof.slo is None
    reason = prof.provisional_reason
    assert "not a capacity" in reason.lower() and "C_knee" in reason and "not product limits" in reason
    cls = prof.workload_class(deploy.load_config(deploy.DEFAULT_CONFIG).gateway["active_class"])
    assert (cls.max_concurrency, cls.max_queue) == (4, 4)          # deliberately small; Core's 2K C_knee was 25
    assert cls.max_prompt_tokens + cls.max_output_tokens <= bp.MAX_MODEL_LEN and "unverified" in cls.source
    assert prof.server.fingerprint_sha256 == stage_b_fingerprint(PINS)
    assert (prof.server.num_gpu_blocks, prof.server.block_size) == (SB["num_gpu_blocks"], SB["block_size"])
    assert deploy.profile_server_problems(prof, pins=PINS, fingerprint=stage_b_fingerprint(PINS)) == []


def test_a_local_tokenizer_gets_the_served_snapshot(world):
    """038 §4: with [gateway] tokenizer = "local" the deploy sets tokenizer_path to the pinned snapshot vLLM serves
    (core_b_plan.snapshot_dir); [gateway] may not set it itself."""
    cfg = world.cfg(gateway={"active_class": "test", "tokenizer": "local"})
    doc = deploy.gateway_settings(cfg)
    assert doc["tokenizer_path"] == bp.snapshot_dir(bp.load_pins())
    Settings.model_validate(doc)
    assert "tokenizer_path" not in deploy.gateway_settings(world.cfg(gateway={"active_class": "test"}))
    with pytest.raises(deploy.DeployError, match="tokenizer_path"):
        deploy.DeployConfig.checked({"service_root": str(world.svc), "gateway": {
            "active_class": "test", "tokenizer": "local", "tokenizer_path": "/elsewhere"}})
