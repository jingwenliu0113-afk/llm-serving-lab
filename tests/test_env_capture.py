"""Tests for scripts/env_capture.py.

EXPERIMENT_RULES.md "Benchmark Environment" requires every run to record:
git commit, OS, runtime version, CUDA / driver, power state, background load,
benchmark config. These tests pin down the parsers (fixtures are real command
outputs where available) and the capture() contract: it must never raise when
a probe is missing or misbehaves, and must surface every failed probe instead
of silently returning null.
"""

import datetime as dt
import hashlib
import json
import re
import sys

import pytest

import env_capture as ec


# ---------------------------------------------------------------- fixtures --

# Real output, M4 Pro MacBook Pro on AC (2026-10-08).
PMSET_BATT_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=24576099)\t80%; AC attached; not charging present: true\n"
)
PMSET_BATT_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=24576099)\t55%; discharging; 3:10 remaining present: true\n"
)
PMSET_BATT_DESKTOP = "Now drawing from 'AC Power'\n"

# Real excerpt of `pmset -g` (2026-10-08).
PMSET_G_HIGH = (
    "System-wide power settings:\n"
    "Currently in use:\n"
    " standby              1\n"
    " sleep                0 (sleep prevented by Claude, powerd)\n"
    " powermode            2\n"
    " womp                 1\n"
)
PMSET_G_OLD_LOWPOWER = "Currently in use:\n lowpowermode         1\n"
PMSET_G_NONE = "Currently in use:\n standby              1\n"

# Real output of `pmset -g therm` with nothing recorded (2026-10-08).
PMSET_THERM_CLEAN = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)
# Format when the OS has recorded CPU power limits (throttling).
PMSET_THERM_LIMITED = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "2026-10-08 12:00:00 +0800 CPU Power notify\n"
    "\tCPU_Scheduler_Limit \t= 100\n"
    "\tCPU_Available_CPUs \t= 12\n"
    "\tCPU_Speed_Limit \t= 71\n"
)

# Real output (2026-10-08).
SW_VERS = "ProductName:\t\tmacOS\nProductVersion:\t\t27.0.1\nBuildVersion:\t\t26A434\n"

OS_RELEASE_UBUNTU = (
    'PRETTY_NAME="Ubuntu 24.04.1 LTS"\n'
    'NAME="Ubuntu"\n'
    'VERSION_ID="24.04"\n'
    "ID=ubuntu\n"
)
PROC_VERSION_WSL = (
    "Linux version 6.6.36.6-microsoft-standard-WSL2 (root@1234) "
    "(gcc (GCC) 11.2.0, GNU ld (GNU Binutils) 2.37) #1 SMP PREEMPT_DYNAMIC\n"
)
PROC_VERSION_NATIVE = "Linux version 6.8.0-45-generic (buildd@lcy02) (gcc 13.2.0) #45-Ubuntu SMP\n"

# Format of: nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits
NVIDIA_IDENTITY_CSV = "NVIDIA GeForce RTX 5070 Ti, 580.97, 12.0, 16303, 512, [N/A], P8, 12.34\n"
NVIDIA_POWER_CSV = "41, 180, 3105, 300.00, 300.00, 300.00, 0x0000000000000001\n"

# Real output from the RTX 5070 Ti host (WSL2 Ubuntu 22.04.5, driver 617.42, 2026-10-08).
NVIDIA_IDENTITY_CSV_REAL = "NVIDIA GeForce RTX 5070 Ti, 617.42, 12.0, 16303, 1026, 0, P8, 4.64\n"
NVIDIA_POWER_CSV_REAL = "36, 180, 3090, 300.00, 300.00, 300.00, 0x0000000000000400\n"

NVCC_VERSION = (
    "nvcc: NVIDIA (R) Cuda compiler driver\n"
    "Copyright (c) 2005-2025 NVIDIA Corporation\n"
    "Cuda compilation tools, release 12.8, V12.8.93\n"
    "Build cuda_12.8.r12.8/compiler.35583870_0\n"
)

# Real output of the pinned build (b11146 / v0.5.0, scripts/build_llama_cpp.sh metal).
LLAMA_VERSION = (
    "version: 0.5.0 (build 11146, commit 7fe450e19305b828c199d602c23a8337aaa1f03b)\n"
    "built with AppleClang 21.0.0.21000334 for Darwin arm64\n"
)
# Pre-semver banner format (older builds); still recognised.
LLAMA_VERSION_LEGACY = (
    "version: 6500 (abc1234)\n"
    "built with Apple clang version 21.0.0 (clang-2100.3.34.2) for arm64-apple-darwin27.0.0\n"
)

VLLM_STACK = {
    "vllm": "0.30.0",
    "torch": "2.13.0",
    "torch_cuda": "13.0",
    "transformers": "5.16.1",
    "triton": "3.7.1",
    "tokenizers": "0.23.1",
    "huggingface_hub": "1.31.0",
    "nvidia_wheels": {"nvidia-cuda-runtime": "13.0.96", "nvidia-nvjitlink": "13.0.88"},
    "freeze_sha256": "ab" * 32,
}
VLLM_PROBE_OUT = json.dumps(VLLM_STACK)

MAC_LOCK_OWNER = "pid=4242\nstarted_at=2026-10-08T12:00:00+08:00\ncwd=/x/rust-hnsw\ncommand=cargo bench\n"


# ------------------------------------------------------------- fake system --


class FakeSystem:
    """Scripted stand-in for subprocess / filesystem probes."""

    def __init__(self, commands=None, files=None, existing_paths=()):
        self.commands = commands or {}
        self.files = files or {}
        self.existing_paths = set(existing_paths)
        self.calls = []

    def run(self, cmd):
        self.calls.append(tuple(cmd))
        key = tuple(cmd)
        if key not in self.commands:
            return None  # executable not found
        value = self.commands[key]
        if isinstance(value, ec.CmdResult):
            return value
        return ec.CmdResult(returncode=0, stdout=value, stderr="")

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def path_exists(self, path):
        return path in self.existing_paths


FIXED_NOW = dt.datetime(2026, 10, 8, 3, 30, 0, tzinfo=dt.timezone.utc)

GIT_HEAD = ("git", "-C", "REPO", "rev-parse", "HEAD")
GIT_STATUS = ("git", "-C", "REPO", "status", "--porcelain")
NVIDIA_IDENTITY_CMD = ("nvidia-smi", f"--query-gpu={ec.NVIDIA_IDENTITY.query}", "--format=csv,noheader,nounits")
NVIDIA_POWER_CMD = ("nvidia-smi", f"--query-gpu={ec.NVIDIA_POWER.query}", "--format=csv,noheader,nounits")
VLLM_PY = "/opt/vllm-venv/bin/python"


def darwin_system(**overrides):
    commands = {
        GIT_HEAD: "0123456789abcdef0123456789abcdef01234567\n",
        GIT_STATUS: "",
        ("sw_vers",): SW_VERS,
        ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M4 Pro\n",
        ("sysctl", "-n", "hw.memsize"): "51539607552\n",
        ("sysctl", "-n", "iogpu.wired_limit_mb"): "0\n",
        ("pmset", "-g", "batt"): PMSET_BATT_AC,
        ("pmset", "-g"): PMSET_G_HIGH,
        ("pmset", "-g", "therm"): PMSET_THERM_CLEAN,
    }
    commands.update(overrides.pop("commands", {}))
    return FakeSystem(commands=commands, **overrides)


def wsl_system(**overrides):
    commands = {
        GIT_HEAD: "0123456789abcdef0123456789abcdef01234567\n",
        GIT_STATUS: "",
        NVIDIA_IDENTITY_CMD: NVIDIA_IDENTITY_CSV,
        NVIDIA_POWER_CMD: NVIDIA_POWER_CSV,
        ("nvcc", "--version"): NVCC_VERSION,
    }
    commands.update(overrides.pop("commands", {}))
    files = {
        "/etc/os-release": OS_RELEASE_UBUNTU,
        "/proc/version": PROC_VERSION_WSL,
        "/proc/cpuinfo": "processor\t: 0\nmodel name\t: AMD Ryzen 7 9700X 8-Core Processor\n",
        "/proc/meminfo": "MemTotal:       32768000 kB\nMemFree:  1000 kB\n",
    }
    return FakeSystem(commands=commands, files=files, **overrides)


def capture(system, system_name, **kwargs):
    kwargs.setdefault("env", {})
    return ec.capture(
        repo_dir="REPO",
        system_name=system_name,
        run=system.run,
        read_file=system.read_file,
        path_exists=system.path_exists,
        now=lambda: FIXED_NOW,
        loadavg=lambda: (2.5, 2.0, 1.5),
        cpu_count=lambda: 12,
        **kwargs,
    )


# ----------------------------------------------------------------- parsers --


@pytest.mark.parametrize(
    "text, expected",
    [
        (PMSET_BATT_AC, {"source": "AC", "battery_pct": 80}),
        (PMSET_BATT_BATTERY, {"source": "Battery", "battery_pct": 55}),
        (PMSET_BATT_DESKTOP, {"source": "AC", "battery_pct": None}),
        ("", {"source": None, "battery_pct": None}),
    ],
)
def test_parse_pmset_batt(text, expected):
    assert ec.parse_pmset_batt(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (PMSET_G_HIGH, {"powermode": 2, "label": "high_power"}),
        ("Currently in use:\n powermode            1\n", {"powermode": 1, "label": "low_power"}),
        ("Currently in use:\n powermode            0\n", {"powermode": 0, "label": "automatic"}),
        (PMSET_G_OLD_LOWPOWER, {"powermode": 1, "label": "low_power"}),
        (PMSET_G_NONE, {"powermode": None, "label": None}),
    ],
)
def test_parse_pmset_powermode(text, expected):
    assert ec.parse_pmset_powermode(text) == expected


def test_parse_pmset_therm_clean():
    assert ec.parse_pmset_therm(PMSET_THERM_CLEAN) == {
        "thermal_warning_recorded": False,
        "performance_warning_recorded": False,
        "cpu_limits": {},
    }


def test_parse_pmset_therm_limited():
    assert ec.parse_pmset_therm(PMSET_THERM_LIMITED) == {
        "thermal_warning_recorded": False,
        "performance_warning_recorded": False,
        "cpu_limits": {"CPU_Scheduler_Limit": 100, "CPU_Available_CPUs": 12, "CPU_Speed_Limit": 71},
    }


def test_parse_sw_vers():
    assert ec.parse_sw_vers(SW_VERS) == {
        "product_name": "macOS",
        "product_version": "27.0.1",
        "build_version": "26A434",
    }


def test_parse_os_release():
    assert ec.parse_os_release(OS_RELEASE_UBUNTU) == {
        "pretty_name": "Ubuntu 24.04.1 LTS",
        "id": "ubuntu",
        "version_id": "24.04",
    }


def test_detect_wsl():
    assert ec.detect_wsl(PROC_VERSION_WSL) is True
    assert ec.detect_wsl(PROC_VERSION_NATIVE) is False
    assert ec.detect_wsl("") is False


def test_parse_nvidia_identity():
    gpus, errors = ec.parse_nvidia_smi_csv(NVIDIA_IDENTITY_CSV, ec.NVIDIA_IDENTITY)
    assert errors == []
    assert gpus == [
        {
            "name": "NVIDIA GeForce RTX 5070 Ti",
            "driver_version": "580.97",
            "compute_cap": "12.0",
            "memory_total_mib": 16303,
            "memory_used_mib": 512,
            "utilization_gpu_pct": None,  # "[N/A]" (WSL2 cannot report it)
            "pstate": "P8",
            "power_draw_w": 12.34,
        }
    ]


def test_parse_nvidia_power():
    gpus, errors = ec.parse_nvidia_smi_csv(NVIDIA_POWER_CSV, ec.NVIDIA_POWER)
    assert errors == []
    assert gpus == [
        {
            "temperature_c": 41,
            "clocks_sm_mhz": 180,
            "clocks_max_sm_mhz": 3105,
            "power_limit_w": 300.0,
            "enforced_power_limit_w": 300.0,
            "power_default_limit_w": 300.0,
            "clocks_event_reasons_active": "0x0000000000000001",
        }
    ]


def test_parse_nvidia_real_wsl_output():
    (ident,), errors = ec.parse_nvidia_smi_csv(NVIDIA_IDENTITY_CSV_REAL, ec.NVIDIA_IDENTITY)
    assert errors == []
    assert ident["compute_cap"] == "12.0"
    assert ident["memory_total_mib"] == 16303
    assert ident["utilization_gpu_pct"] == 0
    (power,), errors = ec.parse_nvidia_smi_csv(NVIDIA_POWER_CSV_REAL, ec.NVIDIA_POWER)
    assert errors == []
    assert power["enforced_power_limit_w"] == power["power_default_limit_w"] == 300.0
    assert power["clocks_event_reasons_active"] == "0x0000000000000400"


def test_parse_nvidia_multi_gpu():
    text = NVIDIA_IDENTITY_CSV + NVIDIA_IDENTITY_CSV.replace("5070 Ti", "5060")
    gpus, errors = ec.parse_nvidia_smi_csv(text, ec.NVIDIA_IDENTITY)
    assert [g["name"] for g in gpus] == ["NVIDIA GeForce RTX 5070 Ti", "NVIDIA GeForce RTX 5060"]
    assert errors == []


def test_parse_nvidia_malformed_line_is_reported_not_dropped():
    gpus, errors = ec.parse_nvidia_smi_csv("No devices were found\n", ec.NVIDIA_IDENTITY)
    assert gpus == []
    assert len(errors) == 1 and "No devices were found" in errors[0]


def test_parse_nvcc_version():
    assert ec.parse_nvcc_version(NVCC_VERSION) == "12.8"
    assert ec.parse_nvcc_version("garbage") is None


def test_parse_llama_version():
    assert ec.parse_llama_version(LLAMA_VERSION) == {
        "version": "0.5.0",
        "build_number": 11146,
        "commit": "7fe450e19305b828c199d602c23a8337aaa1f03b",
        "built_with": "AppleClang 21.0.0.21000334 for Darwin arm64",
    }


def test_parse_llama_version_legacy_format():
    assert ec.parse_llama_version(LLAMA_VERSION_LEGACY) == {
        "version": None,
        "build_number": 6500,
        "commit": "abc1234",
        "built_with": "Apple clang version 21.0.0 (clang-2100.3.34.2) for arm64-apple-darwin27.0.0",
    }


def test_parse_llama_version_unrecognised_returns_none():
    assert ec.parse_llama_version("something else") is None


@pytest.mark.parametrize(
    "porcelain, dirty",
    [
        ("", False),
        (" M scripts/env_capture.py\n", True),  # tracked modification
        ("?? benchmarks/new_harness.py\n", True),  # untracked code counts
        ("?? results/core_a/run1.json\n", False),  # uncommitted results do not
        ("?? results/core_a/run1.json\nM  configs/pins.toml\n", True),
    ],
)
def test_git_dirty_ignores_only_untracked_results(porcelain, dirty):
    assert ec.git_dirty(porcelain) is dirty


def test_select_env_vars_redacts_credentials():
    env = {
        "LLAMA_API_KEY": "sk-secret",
        "VLLM_API_KEY": "sk-2",
        "LLAMA_ARG_HF_TOKEN": "hf_x",
        "VLLM_AUTH_PASSWORD": "pw",
        "CUDA_CREDENTIALS": "c",
        "GGML_CUDA_PDL": "0",
    }
    out = ec.select_env_vars(env)
    assert out == {
        "CUDA_CREDENTIALS": ec.REDACTED,
        "GGML_CUDA_PDL": "0",
        "LLAMA_API_KEY": ec.REDACTED,
        "LLAMA_ARG_HF_TOKEN": ec.REDACTED,
        "VLLM_API_KEY": ec.REDACTED,
        "VLLM_AUTH_PASSWORD": ec.REDACTED,
    }
    assert not any(secret in json.dumps(out) for secret in ("sk-secret", "sk-2", "hf_x", "pw"))


def test_select_env_vars():
    env = {
        "GGML_CUDA_PDL": "0",
        "GGML_METAL_TENSOR_DISABLE": "1",
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "SOURCE_DATE_EPOCH": "0",
        "HOME": "/Users/x",
        "AWS_SECRET_ACCESS_KEY": "nope",
    }
    assert ec.select_env_vars(env) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "GGML_CUDA_PDL": "0",
        "GGML_METAL_TENSOR_DISABLE": "1",
        "HF_HUB_OFFLINE": "1",
        "SOURCE_DATE_EPOCH": "0",
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
    }


# ----------------------------------------------------------------- capture --


def test_capture_darwin_core_fields():
    rec = capture(darwin_system(), "Darwin")

    assert rec["schema_version"] == ec.SCHEMA_VERSION
    assert rec["captured_at_utc"] == "2026-10-08T03:30:00+00:00"
    assert rec["repo"] == {
        "commit": "0123456789abcdef0123456789abcdef01234567", "dirty": False, "status": [], "source": "git",
    }
    assert rec["os"]["system"] == "Darwin"
    assert rec["os"]["product_version"] == "27.0.1"
    assert rec["os"]["wsl"] is False
    assert rec["hardware"] == {
        "cpu": "Apple M4 Pro",
        "logical_cpus": 12,
        "ram_bytes": 51539607552,
        "iogpu_wired_limit_mb": 0,
    }
    assert rec["power"] == {
        "source": "AC",
        "battery_pct": 80,
        "powermode": 2,
        "powermode_label": "high_power",
        "thermal": {"thermal_warning_recorded": False, "performance_warning_recorded": False, "cpu_limits": {}},
    }
    assert rec["load"] == {
        "loadavg_1m": 2.5,
        "loadavg_5m": 2.0,
        "loadavg_15m": 1.5,
        "mac_heavy_lock_present": False,
        "mac_heavy_lock_owner": None,
        "gpu_lock_held": None,  # GPU lock lives in WSL; meaningless on the Mac
    }
    # No NVIDIA stack on a Mac: explicit nulls, not missing keys.
    assert rec["gpu"] == {"nvidia": None, "cuda_toolkit_on_path": None}
    assert rec["runtime"] == {"llama_cpp": None, "vllm": None}
    assert rec["config"] is None
    assert rec["env"] == {}


def test_capture_is_json_serialisable():
    rec = capture(darwin_system(), "Darwin")
    assert json.loads(json.dumps(rec)) == rec


def test_capture_reports_dirty_tree_with_paths():
    system = darwin_system(commands={GIT_STATUS: " M scripts/env_capture.py\n?? results/x.json\n"})
    repo = capture(system, "Darwin")["repo"]
    assert repo["dirty"] is True
    assert repo["status"] == [" M scripts/env_capture.py", "?? results/x.json"]


def test_capture_reports_mac_heavy_lock_and_owner():
    owner_path = f"{ec.MAC_HEAVY_LOCK}/owner.txt"
    system = darwin_system(existing_paths={ec.MAC_HEAVY_LOCK})
    system.files[owner_path] = MAC_LOCK_OWNER
    load = capture(system, "Darwin")["load"]
    assert load["mac_heavy_lock_present"] is True
    assert load["mac_heavy_lock_owner"] == MAC_LOCK_OWNER


def test_capture_records_env_vars():
    rec = capture(darwin_system(), "Darwin", env={"GGML_CUDA_PDL": "0", "HOME": "/x"})
    assert rec["env"] == {"GGML_CUDA_PDL": "0"}


def test_capture_wsl_with_nvidia():
    system = wsl_system()
    rec = capture(system, "Linux")

    assert rec["os"]["wsl"] is True
    assert rec["os"]["pretty_name"] == "Ubuntu 24.04.1 LTS"
    assert rec["hardware"]["cpu"] == "AMD Ryzen 7 9700X 8-Core Processor"
    assert rec["hardware"]["ram_bytes"] == 32768000 * 1024
    (gpu,) = rec["gpu"]["nvidia"]
    assert gpu["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert gpu["enforced_power_limit_w"] == 300.0  # identity + power merged per GPU
    assert rec["gpu"]["cuda_toolkit_on_path"] == "12.8"
    # pmset / sysctl do not exist on Linux and must not be probed there.
    assert not any(call[0] in ("pmset", "sysctl") for call in system.calls)
    assert rec["power"]["source"] is None
    # mkdir-style Mac lock is only meaningful on macOS.
    assert rec["load"]["mac_heavy_lock_present"] is None
    assert rec["probe_errors"] == []


@pytest.mark.parametrize("held", [True, False])
def test_capture_records_gpu_lock_state_on_linux(held):
    # The harness runs under `flock -n /tmp/ai-portfolio-gpu.lock` and reports it.
    rec = capture(wsl_system(), "Linux", gpu_lock_held=held)
    assert rec["load"]["gpu_lock_held"] is held


def test_capture_gpu_lock_unknown_when_not_reported():
    assert capture(wsl_system(), "Linux")["load"]["gpu_lock_held"] is None


def test_capture_nvidia_power_query_failure_keeps_identity():
    system = wsl_system(commands={NVIDIA_POWER_CMD: ec.CmdResult(6, "", "Field 'x' is not a valid field")})
    rec = capture(system, "Linux")
    (gpu,) = rec["gpu"]["nvidia"]
    assert gpu["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert "enforced_power_limit_w" not in gpu
    assert any("nvidia-smi power" in e for e in rec["probe_errors"])


def test_capture_nvidia_zero_gpus_is_an_error():
    system = wsl_system(commands={NVIDIA_IDENTITY_CMD: ""})
    rec = capture(system, "Linux")
    assert rec["gpu"]["nvidia"] == []
    assert any("no GPU" in e for e in rec["probe_errors"])


def test_capture_llama_runtime(tmp_path):
    binary = str(tmp_path / "llama-cli")
    system = darwin_system(
        commands={(binary, "--version"): ec.CmdResult(returncode=0, stdout="", stderr=LLAMA_VERSION)}
    )
    rec = capture(system, "Darwin", llama_bin=binary)
    assert rec["runtime"]["llama_cpp"] == {
        "binary": binary,
        "version": "0.5.0",
        "build_number": 11146,
        "commit": "7fe450e19305b828c199d602c23a8337aaa1f03b",
        "built_with": "AppleClang 21.0.0.21000334 for Darwin arm64",
    }


def test_capture_vllm_runtime_from_its_own_interpreter():
    system = wsl_system(commands={(VLLM_PY, "-c", ec.VLLM_PROBE): VLLM_PROBE_OUT})
    rec = capture(system, "Linux", vllm_python=VLLM_PY)
    assert rec["runtime"]["vllm"] == {"python": VLLM_PY, **VLLM_STACK}
    assert rec["probe_errors"] == []


def test_capture_vllm_requested_but_missing_is_an_error():
    rec = capture(wsl_system(), "Linux", vllm_python=VLLM_PY)
    assert rec["runtime"]["vllm"] is None
    assert any("vllm" in e for e in rec["probe_errors"])


def test_capture_vllm_not_requested_is_silent():
    rec = capture(wsl_system(), "Linux")
    assert rec["runtime"]["vllm"] is None
    assert rec["probe_errors"] == []


def test_capture_config_hash(tmp_path):
    cfg = tmp_path / "bench.toml"
    cfg.write_bytes(b"[bench]\nctx = 2048\n")
    rec = capture(darwin_system(), "Darwin", config_path=str(cfg))
    assert rec["config"] == {
        "path": str(cfg),
        "sha256": hashlib.sha256(b"[bench]\nctx = 2048\n").hexdigest(),
    }


def test_capture_missing_probes_never_raise_and_are_reported():
    # Nothing available at all: not a git repo, no sw_vers, no pmset, no sysctl.
    rec = capture(FakeSystem(), "Darwin", llama_bin="/nope/llama-cli")

    assert rec["repo"] == {"commit": None, "dirty": None, "status": None, "source": None}
    assert rec["hardware"]["cpu"] is None
    assert rec["power"]["source"] is None
    assert rec["runtime"]["llama_cpp"] is None
    failed = " ".join(rec["probe_errors"])
    for probe in ("git", "sw_vers", "sysctl", "pmset", "therm", "iogpu", "llama"):
        assert probe in failed


def test_capture_failed_command_is_reported():
    system = darwin_system(
        commands={GIT_HEAD: ec.CmdResult(returncode=128, stdout="", stderr="fatal: not a git repository")}
    )
    rec = capture(system, "Darwin")
    assert rec["repo"]["commit"] is None
    assert any("git" in e and "not a git repository" in e for e in rec["probe_errors"])


def test_capture_happy_path_has_no_probe_errors():
    assert capture(darwin_system(), "Darwin")["probe_errors"] == []


# ------------------------------------------------- default_run robustness --


def test_default_run_missing_executable_returns_none(tmp_path):
    assert ec.default_run([str(tmp_path / "does-not-exist")]) is None


def test_default_run_exec_format_error_does_not_raise(tmp_path):
    bogus = tmp_path / "llama-cli"
    bogus.write_bytes(b"\x00\x01not an executable")
    bogus.chmod(0o755)
    res = ec.default_run([str(bogus), "--version"])
    assert res is not None and res.returncode != 0
    assert "Error" in res.stderr


def test_default_run_undecodable_output_does_not_raise():
    res = ec.default_run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'ok \\xff')"])
    assert res.returncode == 0
    assert res.stdout.startswith("ok ")


def test_capture_with_real_runner_and_broken_binary_reports_llama(tmp_path):
    bogus = tmp_path / "llama-cli"
    bogus.write_bytes(b"\x00\x01not an executable")
    bogus.chmod(0o755)
    rec = ec.capture(repo_dir=str(tmp_path), llama_bin=str(bogus), env={})
    assert rec["runtime"]["llama_cpp"] is None
    assert any("llama" in e for e in rec["probe_errors"])


# --------------------------------------------------------------------- CLI --


def test_main_writes_json(tmp_path, monkeypatch):
    out = tmp_path / "env.json"
    monkeypatch.setattr(ec, "capture", lambda **kwargs: {"schema_version": ec.SCHEMA_VERSION, "kw": sorted(kwargs)})
    argv = ["--out", str(out), "--llama-bin", "/x/llama-cli", "--config", "c.toml", "--vllm-python", VLLM_PY]
    assert ec.main(argv + ["--gpu-lock-held"]) == 0
    data = json.loads(out.read_text())
    assert data["schema_version"] == ec.SCHEMA_VERSION
    assert {"llama_bin", "config_path", "vllm_python", "gpu_lock_held"} <= set(data["kw"])


def test_vllm_probe_runs_and_is_parsed(tmp_path):
    # Stub torch / vllm so the probe string actually executes in a real interpreter.
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text(
        "__version__ = '2.13.0'\nclass version:\n    cuda = '13.0'\n"
    )
    (tmp_path / "vllm.py").write_text("__version__ = '0.30.0'\n")
    wrapper = tmp_path / "python"
    wrapper.write_text(f"#!/bin/sh\nPYTHONPATH={tmp_path} exec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)

    rec = ec.capture(repo_dir=str(tmp_path), system_name="Linux", vllm_python=str(wrapper), env={},
                     read_file=lambda p: "", run=lambda cmd: ec.default_run(cmd) if cmd[0] == str(wrapper) else None)
    stack = rec["runtime"]["vllm"]
    assert stack["vllm"] == "0.30.0" and stack["torch"] == "2.13.0" and stack["torch_cuda"] == "13.0"
    assert set(stack) == {"python", *ec._VLLM_KEYS}
    assert re.fullmatch(r"[0-9a-f]{64}", stack["freeze_sha256"])
    assert not any(e.startswith("vllm") for e in rec["probe_errors"])


SHA = "1d952c2" + "0" * 33


def test_capture_falls_back_to_deployed_commit_file():
    # Remote copies are deployed as a bundle with DEPLOYED_COMMIT (+ manifest), no .git.
    system = FakeSystem(files={"REPO/DEPLOYED_COMMIT": SHA + "\n"})
    rec = capture(system, "Linux")
    assert rec["repo"] == {"commit": SHA, "dirty": None, "status": None, "source": "DEPLOYED_COMMIT"}
    # the git failure is still reported, not hidden
    assert any("git rev-parse" in e for e in rec["probe_errors"])


def test_deployed_commit_free_text_is_not_a_commit():
    """Review F12: a hot-patch note must not be recorded as a clean commit."""
    system = FakeSystem(files={"REPO/DEPLOYED_COMMIT": SHA + " + uncommitted benchmarks/coreb_smoke.py\n"})
    repo = capture(system, "Linux")["repo"]
    assert repo["commit"] == SHA and repo["dirty"] is True
    assert repo["deployed_note"] == "+ uncommitted benchmarks/coreb_smoke.py"
    system = FakeSystem(files={"REPO/DEPLOYED_COMMIT": "1d952c2 + patch\n"})
    repo = capture(system, "Linux")["repo"]
    assert repo["commit"] is None and repo["dirty"] is True and repo["deployed_note"] == "1d952c2 + patch"


def _manifest(files, commit=SHA):
    return json.dumps({"commit": commit, "files": files})


def test_deploy_manifest_verifies_files():
    files = {"benchmarks/coreb_smoke.py": "aa" * 32, "scripts/env_capture.py": "bb" * 32}
    hashes = {"REPO/benchmarks/coreb_smoke.py": "aa" * 32, "REPO/scripts/env_capture.py": "bb" * 32}
    system = FakeSystem(files={"REPO/DEPLOYED_COMMIT": SHA + "\n", "REPO/DEPLOY_MANIFEST.json": _manifest(files)})
    repo = capture(system, "Linux", file_sha256=hashes.__getitem__)["repo"]
    assert repo["dirty"] is False and repo["status"] == []

    hashes["REPO/benchmarks/coreb_smoke.py"] = "cc" * 32  # hot-patched after deploy
    repo = capture(system, "Linux", file_sha256=hashes.__getitem__)["repo"]
    assert repo["dirty"] is True and repo["status"] == [" M benchmarks/coreb_smoke.py"]


def test_deploy_manifest_for_another_commit_is_dirty():
    system = FakeSystem(files={"REPO/DEPLOYED_COMMIT": SHA + "\n",
                               "REPO/DEPLOY_MANIFEST.json": _manifest({}, commit="f" * 40)})
    rec = capture(system, "Linux", file_sha256=lambda p: "")
    assert rec["repo"]["dirty"] is True
    assert any("DEPLOY_MANIFEST" in e for e in rec["probe_errors"])


def test_capture_repo_source_is_git_when_available():
    assert capture(darwin_system(), "Darwin")["repo"]["source"] == "git"
