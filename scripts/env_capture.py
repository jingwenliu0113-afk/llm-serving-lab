#!/usr/bin/env python3
"""Capture the benchmark environment record required by docs/EXPERIMENT_RULES.md.

Records: git commit (+ dirty flag and status), OS, hardware, runtime version
(llama.cpp / vLLM stack), CUDA / driver, power + thermal state, background
load, run-relevant environment variables, benchmark config hash.

Stdlib only, so the same file runs on the Mac and inside WSL2. Every probe is
best-effort: a missing or misbehaving tool never raises, it yields null plus an
entry in `probe_errors`, so an incomplete record is visible rather than
silently clean.

Usage:
    python scripts/env_capture.py [--llama-bin PATH] [--vllm-python PATH]
                                  [--config PATH] [--out FILE]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, NamedTuple

SCHEMA_VERSION = 2

# Shared Mac heavy-work lock from the portfolio policy (mkdir-style: exists == held).
MAC_HEAVY_LOCK = "/tmp/ai-portfolio-mac-heavy.lock"

_POWERMODE_LABELS = {0: "automatic", 1: "low_power", 2: "high_power"}

# Variables that change kernels, devices or engine behaviour (decisions 002 / 003).
_ENV_VAR_RE = re.compile(r"^(GGML_|LLAMA_|CUDA_|VLLM_|PYTORCH_|NCCL_|OMP_)|^(HF_HUB_OFFLINE|SOURCE_DATE_EPOCH)$")
# Names that may hold credentials (e.g. LLAMA_API_KEY, VLLM_API_KEY): keep the name, never the value.
_SECRET_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|PASSW|AUTH|CRED", re.IGNORECASE)
REDACTED = "<redacted>"

# Printed as one JSON line by the vLLM venv's own interpreter (decision 003).
# freeze_sha256 hashes the sorted name==version list of every installed distribution,
# so a run record can be checked against the frozen lock file.
VLLM_PROBE = (
    "import hashlib, json, importlib.metadata as m, torch, vllm\n"
    "def v(p):\n"
    "    try:\n"
    "        return m.version(p)\n"
    "    except m.PackageNotFoundError:\n"
    "        return None\n"
    "dists = sorted({(d.metadata['Name'] or '').lower(): d.version for d in m.distributions()}.items())\n"
    "print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,"
    " 'transformers': v('transformers'), 'triton': v('triton'), 'tokenizers': v('tokenizers'),"
    " 'huggingface_hub': v('huggingface-hub'),"
    " 'nvidia_wheels': {n: ver for n, ver in dists if n.startswith('nvidia-')},"
    " 'freeze_sha256': hashlib.sha256('\\n'.join(f'{n}=={ver}' for n, ver in dists).encode()).hexdigest()}))"
)
_VLLM_KEYS = (
    "vllm",
    "torch",
    "torch_cuda",
    "transformers",
    "triton",
    "tokenizers",
    "huggingface_hub",
    "nvidia_wheels",
    "freeze_sha256",
)


class CmdResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


class NvidiaQuery(NamedTuple):
    query: str  # comma-separated --query-gpu fields
    fields: tuple  # (record key, cast) per field, same order


def _str(value: str) -> str:
    return value


NVIDIA_IDENTITY = NvidiaQuery(
    "name,driver_version,compute_cap,memory.total,memory.used,utilization.gpu,pstate,power.draw",
    (
        ("name", _str),
        ("driver_version", _str),
        ("compute_cap", _str),
        ("memory_total_mib", int),
        ("memory_used_mib", int),
        ("utilization_gpu_pct", int),
        ("pstate", _str),
        ("power_draw_w", float),
    ),
)
NVIDIA_POWER = NvidiaQuery(
    "temperature.gpu,clocks.sm,clocks.max.sm,power.limit,enforced.power.limit,power.default_limit,"
    "clocks_event_reasons.active",
    (
        ("temperature_c", int),
        ("clocks_sm_mhz", int),
        ("clocks_max_sm_mhz", int),
        ("power_limit_w", float),
        ("enforced_power_limit_w", float),
        ("power_default_limit_w", float),
        ("clocks_event_reasons_active", _str),
    ),
)

Runner = Callable[[list], "CmdResult | None"]


def default_run(cmd: list) -> CmdResult | None:
    """Run a command; None if the executable does not exist. Never raises."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return CmdResult(returncode=-1, stdout="", stderr="timeout after 30s")
    except OSError as exc:  # permission denied, exec format error, not a directory, ...
        return CmdResult(returncode=-1, stdout="", stderr=f"{type(exc).__name__}: {exc}")
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_deployed_commit(text: str) -> tuple:
    """DEPLOYED_COMMIT = full 40-hex sha. Anything else is kept as a note, never as the commit."""
    text = text.strip()
    token, _, rest = text.partition(" ")
    if re.fullmatch(r"[0-9a-f]{40}", token):
        return token, (rest.strip() or None)
    return None, (text or None)


def _read_text(path: str) -> str:
    return Path(path).read_text(errors="replace")


# ------------------------------------------------------------------ parsers --


_BATT_TIME = re.compile(r"\d+:\d+ remaining|\(no estimate\)")


def parse_pmset_batt(text: str) -> dict:
    """Source, percentage and the state field(s) after it ("charged", "charging", "discharging",
    "AC attached; not charging", ...), without the time estimate (019 records the power state)."""
    m = re.search(r"Now drawing from '(AC|Battery) Power'", text)
    pct = re.search(r"\t(\d+)%;(.*?)(?:\s+present:.*)?$", text, re.MULTILINE)
    state = None
    if pct:
        parts = [x.strip() for x in pct.group(2).split(";")]
        state = "; ".join(x for x in parts if x and not _BATT_TIME.fullmatch(x)) or None
    return {
        "source": m.group(1) if m else None,
        "battery_pct": int(pct.group(1)) if pct else None,
        "battery_state": state,
    }


def parse_pmset_ac(text: str) -> dict:
    """`pmset -g ac`: wattage, manufacturer and adapter name (019). The adapter serial is never recorded."""
    def field(name: str) -> str | None:
        m = re.search(rf"^[ \t]*{re.escape(name)}[ \t]*=[ \t]*(.*?)[ \t]*$", text, re.MULTILINE)  # one line only
        return m.group(1) if m and m.group(1) else None

    watt = field("Wattage")
    w = re.fullmatch(r"(\d+)W", watt or "")
    return {"wattage_w": int(w.group(1)) if w else None, "manufacturer": field("Manufacturer"),
            "name": field("Adapter Name")}


def parse_pmset_powermode(text: str) -> dict:
    m = re.search(r"^\s*powermode\s+(\d+)", text, re.MULTILINE)
    if m is None:
        # Older macOS only exposes a boolean low power mode.
        m = re.search(r"^\s*lowpowermode\s+(\d+)", text, re.MULTILINE)
    if m is None:
        return {"powermode": None, "label": None}
    mode = int(m.group(1))
    return {"powermode": mode, "label": _POWERMODE_LABELS.get(mode)}


def parse_pmset_therm(text: str) -> dict:
    return {
        "thermal_warning_recorded": "No thermal warning level has been recorded" not in text,
        "performance_warning_recorded": "No performance warning level has been recorded" not in text,
        "cpu_limits": {k: int(v) for k, v in re.findall(r"^\s*(CPU_\w+)\s*=\s*(\d+)", text, re.MULTILINE)},
    }


def parse_sw_vers(text: str) -> dict:
    fields = dict(re.findall(r"^(\w+):\s*(.+?)\s*$", text, re.MULTILINE))
    return {
        "product_name": fields.get("ProductName"),
        "product_version": fields.get("ProductVersion"),
        "build_version": fields.get("BuildVersion"),
    }


def parse_os_release(text: str) -> dict:
    fields = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip().strip('"')
    return {
        "pretty_name": fields.get("PRETTY_NAME"),
        "id": fields.get("ID"),
        "version_id": fields.get("VERSION_ID"),
    }


def detect_wsl(proc_version: str) -> bool:
    return "microsoft" in proc_version.lower()


def _num(value: str, cast):
    value = value.strip()
    if not value or value.startswith("["):  # "[N/A]", "[Not Supported]"
        return None
    try:
        return cast(value)
    except ValueError:
        return None


def parse_nvidia_smi_csv(text: str, query: NvidiaQuery) -> tuple[list, list]:
    """Parse `--format=csv,noheader,nounits` output. Returns (gpus, errors)."""
    gpus, errors = [], []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(query.fields):
            errors.append(f"unparsed nvidia-smi line: {line.strip()[:200]}")
            continue
        gpus.append({key: _num(raw, cast) for (key, cast), raw in zip(query.fields, parts)})
    return gpus, errors


def parse_nvcc_version(text: str) -> str | None:
    m = re.search(r"release (\d+\.\d+)", text)
    return m.group(1) if m else None


def parse_llama_version(text: str) -> dict | None:
    # Semver banner (v0.x releases): "version: 0.5.0 (build 11146, commit <sha>)"
    m = re.search(r"version:\s*(\d+\.\d+\.\d+\S*)\s*\(build (\d+), commit ([0-9a-f]+)\)", text)
    if m:
        version, build, commit = m.group(1), m.group(2), m.group(3)
    else:
        # Legacy banner: "version: 6500 (abc1234)"
        m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text)
        if m is None:
            return None
        version, build, commit = None, m.group(1), m.group(2)
    built = re.search(r"^built with (.+?)\s*$", text, re.MULTILINE)
    return {
        "version": version,
        "build_number": int(build),
        "commit": commit,
        "built_with": built.group(1) if built else None,
    }


def git_dirty(porcelain: str) -> bool:
    """Tracked changes or untracked files count; untracked files under results/ do not."""
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        if line.startswith("?? ") and line[3:].startswith("results/"):
            continue
        return True
    return False


def select_env_vars(env: Mapping[str, str]) -> dict:
    return {
        k: (REDACTED if _SECRET_NAME_RE.search(k) else env[k]) for k in sorted(env) if _ENV_VAR_RE.search(k)
    }


# ------------------------------------------------------------------ capture --


def capture(
    *,
    repo_dir: str,
    system_name: str | None = None,
    llama_bin: str | None = None,
    vllm_python: str | None = None,
    config_path: str | None = None,
    gpu_lock_held: bool | None = None,
    env: Mapping[str, str] | None = None,
    run: Runner = default_run,
    read_file: Callable[[str], str] = _read_text,
    file_sha256: Callable[[str], str] = _file_sha256,
    path_exists: Callable[[str], bool] = os.path.exists,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    loadavg: Callable[[], tuple] = os.getloadavg,
    cpu_count: Callable[[], int | None] = os.cpu_count,
) -> dict:
    system_name = system_name or platform.system()
    env = os.environ if env is None else env
    errors: list[str] = []

    def probe(label: str, cmd: list) -> str | None:
        res = run(cmd)
        if res is None:
            errors.append(f"{label}: command not found ({cmd[0]})")
            return None
        if res.returncode != 0:
            errors.append(f"{label}: exit {res.returncode}: {res.stderr.strip()[:200]}")
            return None
        return res.stdout

    def read(label: str, path: str) -> str | None:
        try:
            return read_file(path)
        except OSError as exc:
            errors.append(f"{label}: cannot read {path}: {exc}")
            return None

    def as_int(label: str, text: str | None) -> int | None:
        if text is None:
            return None
        if text.strip().isdigit():
            return int(text.strip())
        errors.append(f"{label}: unexpected output {text.strip()[:80]!r}")
        return None

    # repo
    head = probe("git rev-parse", ["git", "-C", repo_dir, "rev-parse", "HEAD"])
    status = probe("git status", ["git", "-C", repo_dir, "status", "--porcelain"]) if head else None
    repo = {
        "commit": head.strip() if head else None,
        "dirty": git_dirty(status) if status is not None else None,
        "status": [line for line in status.splitlines() if line.strip()] if status is not None else None,
        "source": "git" if head else None,
    }
    if not head:
        # Remote copies are deployed as a bundle (scripts/deploy_bundle.py): DEPLOYED_COMMIT
        # holds the full sha, DEPLOY_MANIFEST.json the sha256 of every deployed file.
        try:
            deployed = read_file(f"{repo_dir}/DEPLOYED_COMMIT").strip()
        except OSError:
            deployed = ""
        if deployed:
            commit, note = parse_deployed_commit(deployed)
            repo.update({"commit": commit, "source": "DEPLOYED_COMMIT"})
            if note:
                repo.update({"deployed_note": note, "dirty": True})
            try:
                manifest = json.loads(read_file(f"{repo_dir}/DEPLOY_MANIFEST.json"))
            except OSError:
                manifest = None
            except ValueError as exc:
                errors.append(f"DEPLOY_MANIFEST.json: unreadable: {exc}")
                manifest = None
            if manifest is not None:
                changed = []
                for rel, want in sorted(manifest.get("files", {}).items()):
                    try:
                        got = file_sha256(f"{repo_dir}/{rel}")
                    except (OSError, KeyError):
                        changed.append(f" D {rel}")
                        continue
                    if got != want:
                        changed.append(f" M {rel}")
                if manifest.get("commit") != commit:
                    errors.append(f"DEPLOY_MANIFEST.json commit {manifest.get('commit')} != DEPLOYED_COMMIT {commit}")
                repo["status"] = changed
                repo["dirty"] = bool(changed) or bool(note) or manifest.get("commit") != commit

    # os / hardware / power
    os_info: dict = {"system": system_name, "release": platform.release(), "machine": platform.machine(), "wsl": False}
    hardware: dict = {"cpu": None, "logical_cpus": cpu_count(), "ram_bytes": None}
    power: dict = {"source": None, "battery_pct": None, "battery_state": None, "ac_adapter": None, "powermode": None,
                   "powermode_label": None, "thermal": None}
    mac_lock: bool | None = None
    mac_lock_owner: str | None = None

    if system_name == "Darwin":
        os_info.update(parse_sw_vers(probe("sw_vers", ["sw_vers"]) or ""))
        cpu = probe("sysctl cpu", ["sysctl", "-n", "machdep.cpu.brand_string"])
        hardware["cpu"] = cpu.strip() if cpu else None
        hardware["ram_bytes"] = as_int("sysctl memsize", probe("sysctl memsize", ["sysctl", "-n", "hw.memsize"]))
        # Governs the Metal working-set limit, i.e. the Mac fit boundary (decision 004: must stay 0).
        hardware["iogpu_wired_limit_mb"] = as_int(
            "sysctl iogpu", probe("sysctl iogpu", ["sysctl", "-n", "iogpu.wired_limit_mb"])
        )
        power.update(parse_pmset_batt(probe("pmset batt", ["pmset", "-g", "batt"]) or ""))
        ac = probe("pmset ac", ["pmset", "-g", "ac"])
        power["ac_adapter"] = parse_pmset_ac(ac) if ac is not None else None
        mode = parse_pmset_powermode(probe("pmset settings", ["pmset", "-g"]) or "")
        power["powermode"], power["powermode_label"] = mode["powermode"], mode["label"]
        therm = probe("pmset therm", ["pmset", "-g", "therm"])
        power["thermal"] = parse_pmset_therm(therm) if therm is not None else None
        mac_lock = path_exists(MAC_HEAVY_LOCK)
        if mac_lock:
            try:
                mac_lock_owner = read_file(f"{MAC_HEAVY_LOCK}/owner.txt")
            except OSError:
                mac_lock_owner = None  # lock held but owner not written yet
    elif system_name == "Linux":
        os_info.update(parse_os_release(read("os-release", "/etc/os-release") or ""))
        os_info["wsl"] = detect_wsl(read("proc version", "/proc/version") or "")
        m = re.search(r"^model name\s*:\s*(.+?)\s*$", read("cpuinfo", "/proc/cpuinfo") or "", re.MULTILINE)
        hardware["cpu"] = m.group(1) if m else None
        m = re.search(r"^MemTotal:\s*(\d+)\s*kB", read("meminfo", "/proc/meminfo") or "", re.MULTILINE)
        hardware["ram_bytes"] = int(m.group(1)) * 1024 if m else None

    # gpu (NVIDIA only; the Apple GPU is described by the SoC name)
    gpu: dict = {"nvidia": None, "cuda_toolkit_on_path": None, "host_cc": None, "cuda_home": None,
                 "cuda_home_nvcc": None}
    if system_name == "Linux":

        def nvidia(label: str, query: NvidiaQuery) -> list | None:
            out = probe(label, ["nvidia-smi", f"--query-gpu={query.query}", "--format=csv,noheader,nounits"])
            if out is None:
                return None
            rows, row_errors = parse_nvidia_smi_csv(out, query)
            errors.extend(f"{label}: {e}" for e in row_errors)
            return rows

        identity = nvidia("nvidia-smi identity", NVIDIA_IDENTITY)
        if identity is not None:
            if not identity:
                errors.append("nvidia-smi identity: no GPU reported")
            power_rows = nvidia("nvidia-smi power", NVIDIA_POWER) if identity else None
            if power_rows is not None and len(power_rows) == len(identity):
                identity = [{**a, **b} for a, b in zip(identity, power_rows)]
            elif power_rows is not None:
                errors.append("nvidia-smi power: GPU count differs from identity query")
        gpu["nvidia"] = identity
        # System toolkit on PATH at capture time; vLLM's own CUDA runtime is in runtime.vllm.
        nvcc = probe("nvcc", ["nvcc", "--version"])
        gpu["cuda_toolkit_on_path"] = parse_nvcc_version(nvcc) if nvcc else None
        # The JIT toolchain of the Core B server (decision 014): host C compiler and the nvcc FlashInfer
        # finds through CUDA_HOME (nvcc itself is kept off PATH).
        cc = probe("cc", ["cc", "--version"])
        gpu["host_cc"] = cc.splitlines()[0].strip() if cc and cc.strip() else None
        gpu["cuda_home"] = env.get("CUDA_HOME") or None
        if gpu["cuda_home"]:
            home_nvcc = probe("CUDA_HOME nvcc", [f"{gpu['cuda_home']}/bin/nvcc", "--version"])
            gpu["cuda_home_nvcc"] = parse_nvcc_version(home_nvcc) if home_nvcc else None

    # load
    try:
        l1, l5, l15 = loadavg()
    except OSError as exc:
        errors.append(f"loadavg: {exc}")
        l1 = l5 = l15 = None
    load = {
        "loadavg_1m": l1,
        "loadavg_5m": l5,
        "loadavg_15m": l15,
        "mac_heavy_lock_present": mac_lock,
        "mac_heavy_lock_owner": mac_lock_owner,
        # Reported by the harness that wraps the run in `flock -n /tmp/ai-portfolio-gpu.lock` (WSL).
        "gpu_lock_held": gpu_lock_held if system_name == "Linux" else None,
    }

    # runtimes
    llama = None
    if llama_bin:
        # llama.cpp prints its version banner on stderr.
        res = run([llama_bin, "--version"])
        if res is None:
            errors.append(f"llama.cpp --version: command not found ({llama_bin})")
        else:
            parsed = parse_llama_version(res.stderr + res.stdout)
            if parsed is None:
                errors.append(f"llama.cpp --version: unrecognised output (exit {res.returncode}): {res.stderr.strip()[:200]}")
            else:
                llama = {"binary": llama_bin, **parsed}

    vllm = None
    if vllm_python:
        out = probe("vllm stack", [vllm_python, "-c", VLLM_PROBE])
        if out is not None:
            try:
                data = json.loads(out.strip().splitlines()[-1])
                vllm = {"python": vllm_python, **{k: data[k] for k in _VLLM_KEYS}}
            except (ValueError, KeyError, IndexError) as exc:
                errors.append(f"vllm stack: unparsable probe output ({exc}): {out.strip()[:200]}")

    # config
    config = None
    if config_path:
        try:
            config = {"path": config_path, "sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest()}
        except OSError as exc:
            errors.append(f"config: cannot read {config_path}: {exc}")

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": now().isoformat(),
        "repo": repo,
        "os": os_info,
        "hardware": hardware,
        "gpu": gpu,
        "power": power,
        "load": load,
        "runtime": {"llama_cpp": llama, "vllm": vllm},
        "env": select_env_vars(env),
        "capture_python": platform.python_version(),
        "config": config,
        "probe_errors": errors,
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llama-bin", help="llama.cpp binary to query with --version")
    parser.add_argument("--vllm-python", help="python interpreter of the vLLM venv to probe (Core B)")
    parser.add_argument("--config", dest="config_path", help="benchmark config file to hash")
    parser.add_argument(
        "--gpu-lock-held", action="store_true", help="set by the harness when running under the WSL GPU flock"
    )
    parser.add_argument("--out", help="write JSON here instead of stdout")
    args = parser.parse_args(argv)

    repo_dir = str(Path(__file__).resolve().parent.parent)
    record = capture(
        repo_dir=repo_dir,
        llama_bin=args.llama_bin,
        vllm_python=args.vllm_python,
        config_path=args.config_path,
        gpu_lock_held=True if args.gpu_lock_held else None,
    )
    text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
