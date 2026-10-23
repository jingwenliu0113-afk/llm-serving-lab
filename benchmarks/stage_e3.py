#!/usr/bin/env python3
"""Stage E3 plans and runs (decision 041 v3 §3): H-A2's cause from Core A's decode protocol on eight GGUFs (4B and 9B
in Q8_0, Q6_K, Q4_K_M and the new Q4_0) at L 2048, three repetitions, between two sentinels, one session per platform
with every comparison inside it.

- Jobs are core_a_plan._bench_job("decode", ...) (llama-bench, 005 / §4 flags, -r 5), in Random(sha256("<seed>:stage_e:
  E3:<platform>")) order; n_thr / poll are Core A's (Mac FORMAL_MAC; RTX pins [stage_b]). A pilot runs the two Q4_0s
  and the sentinel cell once.
- configs/stage_e3.toml freezes each platform on its own (freeze.e3_metal / e3_cuda) with its passed pilot (mode
  pilot, finished, its platform's host, build_plan's pilot plan, every job ok, env valid with the Mac's 140W adapter
  and its watts or the RTX's undrifted B, all eight GGUFs' sha256 and headers verified, this harness). The two
  platforms share the harness and [e3], so a formal plan on either needs both freezes and both pilots (041 v3 §3.7):
  no platform's formal session can be stranded by a harness fix the other platform's pilot forces. A plan records
  stage_e3_sha256 = the config's content without [freeze]. A pilot commits no speed: its full rows go to
  raw/rows.full.jsonl and its per-job env captures to raw/env/ (git-ignored; their timestamps give each job's wall
  time), rows.jsonl keeps the job identity and outcome only.
- The runner: the Mac under the Mac heavy lock (owner verified) with 037 §4's rules (the Apple 140W adapter, one
  powermode, AC, battery, thermal; run_core_a's 70 W rule does not apply to Phase 3), the RTX under the GPU lock
  (verified) with the session-start B (stage_b.measure_baseline, before the session directory exists). Before the
  first job: all eight GGUFs' sha256 against their pins and their headers (produce_gguf_e3.header_facts) against
  configs/stage_e3.toml; a formal run also needs at most one earlier void attempt on its platform (a session.json stub
  is written first, so an interrupted run counts). Core A's 30 s idle between jobs, charge gate (Mac, also once
  before the clock starts), env capture around each job, core_a_parse.classify_bench and each row's model_filename /
  model_size / model_type against the config; sentinel drift > 3 % marks the session. The time cap counts
  measurement only (from the first job, charge pauses subtracted); it skips what is left, never the end sentinel.

Usage:
    python benchmarks/stage_e3.py plan --platform macos-metal --mode pilot --out plan.json
    tools/with-mac-lock python benchmarks/stage_e3.py run --plan plan.json --bin <llama bin> --models models/gguf \\
        --out results/stage_e/<session>                                                  # Mac
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/stage_e3.py run ... --gpu-lock-held   # RTX
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
import core_a_parse as cap  # noqa: E402
import core_a_plan as ap  # noqa: E402
import gguf_tensors  # noqa: E402
import produce_gguf_e3 as pe  # noqa: E402
import stage_b  # noqa: E402
from run_core_a import (POWERMODES_ALLOWED, SENTINEL_DRIFT_PCT, _sentinel_summary, charge_gate,  # noqa: E402
                        default_run_proc, env_problems, mac_config_problems, make_env_capture, provenance_problems,
                        read_mac_power, sha256_json, write_raw_manifest)

SCHEMA_VERSION = 1
STAGE = "E3"
PLATFORMS = ("macos-metal", "wsl2-cuda")
FREEZE_KEYS = {"macos-metal": "e3_metal", "wsl2-cuda": "e3_cuda"}
QUANTS = ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0")
SIZES = ("4B", "9B")
NEW = ("4B-Q4_0", "9B-Q4_0")
CONFIG = ROOT / "configs" / "stage_e3.toml"
RESULTS = ("results", "stage_e")
# stage_e3's own modules and the imports no guard test pins (041 v3 §0)
HARNESS_FILES = ("benchmarks/stage_e3.py", "benchmarks/stage_e3_analysis.py", "scripts/produce_gguf_e3.py",
                 "benchmarks/core_a_plan.py", "benchmarks/stage_b.py", "scripts/gguf_tensors.py",
                 "scripts/produce_gguf.py", "scripts/sync_gguf_to_wsl.py", "scripts/disk_guard.py")
MAC_ADAPTER_REQUIRED = {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"}   # 037 §4
MAC_LOCK_DIR = "/tmp/ai-portfolio-mac-heavy.lock"
CONFIG_KEYS = {"schema_version": None, "freeze": {"e3_metal", "e3_cuda", "pilot_e3_metal", "pilot_e3_cuda"},
               "e3": {"seed", "L", "reps", "sentinel", "sentinel_drift_max", "cap_hours", "n_params", "files"}}
PILOT_ROW_KEYS = ("job_id", "kind", "position", "size", "quant", "L", "rep", "model", "outcome", "stage", "reason")
HEADER_KEYS = ("token_embd_bytes", "token_embd_type", "tied", "output_type", "tensor_types", "model_size")
FILE_KEYS = {"filename", "model_type", *HEADER_KEYS}                  # an old config (sha256 / size from pins)
NEW_FILE_KEYS = FILE_KEYS | {"sha256", "size_bytes"}
CAP_HOURS = {"macos-metal": 1.5, "wsl2-cuda": 1.0}                    # 041 v3 §3.6
CAP_MARGIN_S = 600                                                     # a job's timeout past the cap


class StageE3Error(RuntimeError):
    pass


def harness_sha256(root=ROOT) -> str:
    h = hashlib.sha256()
    for rel in HARNESS_FILES:
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- config

def config_problems(cfg: dict) -> list:
    """configs/stage_e3.toml: exactly its keys; the eight configs; a frozen platform needs its pilot and the new GGUFs
    pinned."""
    if set(cfg) != set(CONFIG_KEYS):
        return [f"top-level keys {sorted(cfg)} are not {sorted(CONFIG_KEYS)}"]
    problems = [f"[{t}] keys {sorted(cfg[t])} are not {sorted(k)}" for t, k in CONFIG_KEYS.items()
                if k is not None and set(cfg[t]) != k]
    if cfg["schema_version"] != 1:
        problems.append(f"schema_version {cfg['schema_version']!r} is not 1")
    if problems:
        return problems
    e, f = cfg["e3"], cfg["freeze"]
    if (e["seed"], e["L"], e["reps"], e["sentinel"], e["sentinel_drift_max"]) != \
            (ap.FORMAL_SEED, 2048, ap.REPS, {"size": "4B", "quant": "Q4_K_M"}, SENTINEL_DRIFT_PCT / 100):
        problems.append("e3 seed / L / reps / sentinel / drift are not 041 v3 §3.2's (20261008, 2048, 3, 4B-Q4_K_M, 3 %)")
    if e["cap_hours"] != CAP_HOURS:
        problems.append(f"e3.cap_hours {e['cap_hours']} are not 041 v3 §3.6's {CAP_HOURS}")
    if set(e["n_params"]) != set(SIZES) or not all(isinstance(v, int) and v > 0 for v in e["n_params"].values()):
        problems.append("e3.n_params must give 4B's and 9B's parameter counts")
    want = {f"{s}-{q}" for s in SIZES for q in QUANTS}
    if set(e["files"]) != want:
        return problems + [f"e3.files {sorted(e['files'])} are not the eight configs {sorted(want)}"]
    for key, c in sorted(e["files"].items()):
        keys = NEW_FILE_KEYS if key in NEW else FILE_KEYS
        if set(c) != keys:
            problems.append(f"e3.files.{key} keys {sorted(c)} are not {sorted(keys)}")
        elif key not in NEW and not _file_complete(c):
            problems.append(f"e3.files.{key} is incomplete (every header value is pinned)")
    frozen = any(f[k] is True for k in FREEZE_KEYS.values())
    for platform, key in FREEZE_KEYS.items():
        if not isinstance(f[key], bool) or not isinstance(f[f"pilot_{key}"], str):
            problems.append(f"freeze.{key} must be a bool and freeze.pilot_{key} a string")
        elif f[key] and not f[f"pilot_{key}"]:
            problems.append(f"freeze.pilot_{key} must name the passed {platform} pilot once freeze.{key} is true")
    if frozen:
        problems += new_file_problems(cfg)
    return problems


def _file_complete(c: dict) -> bool:
    filled = all(c.get(k) for k in ("filename", "token_embd_bytes", "token_embd_type", "tensor_types", "model_size",
                                    "model_type"))
    return filled and isinstance(c.get("tied"), bool) and c["tied"] == (c.get("output_type") == "")


def new_file_problems(cfg: dict) -> list:
    out = []
    for key in NEW:
        f = cfg["e3"]["files"][key]
        if len(str(f.get("sha256") or "")) != 64 or not f.get("size_bytes") or not _file_complete(f):
            out.append(f"e3.files.{key} is not pinned yet (scripts/produce_gguf_e3.py produce prints its entry)")
    return out


def config_files(pins: dict, cfg: dict) -> dict:
    """{"<size>-<quant>": {filename, sha256, size_bytes, model_type, header values}}: sha256 and size from pins core_a
    for the six old ones."""
    out = {}
    for m in pins["core_a"]["models"]:
        for q, f in m["files"].items():
            key = f"{m['size']}-{q}"
            c = cfg["e3"]["files"][key]
            if c["filename"] != f["filename"]:
                raise StageE3Error(f"e3.files.{key} is {c['filename']}, pins core_a has {f['filename']}")
            out[key] = {"sha256": f["sha256"], "size_bytes": f["size_bytes"], **{k: c[k] for k in sorted(FILE_KEYS)}}
    for key in NEW:
        c = cfg["e3"]["files"][key]
        out[key] = {k: c.get(k) for k in sorted(NEW_FILE_KEYS)}
    return out


# ---------------------------------------------------------------- plans and evidence

def _params(pins: dict, platform: str) -> tuple:
    if platform == "macos-metal":
        return ap.FORMAL_MAC["n_thr"], ap.FORMAL_MAC["poll"], None
    sb = pins["stage_b"]
    return sb["rtx_n_thr"], sb["rtx_poll"], sb["b_mib"]


def config_sha256(cfg: dict) -> str:
    """configs/stage_e3.toml's content without [freeze] (comments and the freeze flags never change a plan)."""
    return sha256_json({k: v for k, v in cfg.items() if k != "freeze"})


def build_plan(pins: dict, cfg: dict, *, platform: str, mode: str, pins_sha256: str | None = None,
               root=ROOT) -> dict:
    """Pilot plans need the new GGUFs pinned; formal plans also need both platforms' freezes and passed pilots under
    root (041 v3 §3.7)."""
    if platform not in PLATFORMS or mode not in ("pilot", "formal"):
        raise StageE3Error(f"unknown platform / mode {platform!r} / {mode!r}")
    problems = config_problems(cfg) + new_file_problems(cfg)
    if problems:
        raise StageE3Error("configs/stage_e3.toml: " + "; ".join(dict.fromkeys(problems)))
    if mode == "formal":
        unfrozen = [f"freeze.{k}" for k in FREEZE_KEYS.values() if cfg["freeze"][k] is not True]
        if unfrozen:
            raise StageE3Error(f"formal E3 plans need {' and '.join(unfrozen)} = true: both platforms freeze before "
                               "either runs formally (041 v3 §3.7)")
        evidence = freeze_evidence_problems(cfg, root, pins)
        if evidence:
            raise StageE3Error("formal E3 plans need both platforms' pilots first: " + "; ".join(evidence))
    e, files = cfg["e3"], config_files(pins, cfg)
    n_thr, poll, frozen_b = _params(pins, platform)
    L, s = e["L"], e["sentinel"]
    job = lambda kind, size, quant, rep, position=None: ap._bench_job(  # noqa: E731
        kind, size, quant, L, rep, files[f"{size}-{quant}"]["filename"], n_thr, poll, position)
    if mode == "pilot":
        body = [job("decode", "4B", "Q4_0", 1), job("decode", "9B", "Q4_0", 1), job("decode", "4B", "Q4_K_M", 1)]
    else:
        body = [job("decode", size, q, rep) for size in SIZES for q in QUANTS for rep in range(1, e["reps"] + 1)]
        random.Random(int(hashlib.sha256(f"{e['seed']}:stage_e:E3:{platform}".encode()).hexdigest(), 16)).shuffle(body)
    jobs = [job("sentinel", s["size"], s["quant"], 1, "start"), *body, job("sentinel", s["size"], s["quant"], 1, "end")]
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "platform": platform, "mode": mode, "seed": e["seed"],
            "pins_sha256": pins_sha256, "stage_e3_sha256": config_sha256(cfg), "harness_sha256": harness_sha256(),
            "files": files, "params": {"n_thr": n_thr, "poll": poll, "reps": e["reps"], "bench_r": ap.BENCH_R,
                                       "delay_s": ap.DELAY_S, "idle_s": ap.IDLE_S},
            "cap_s": e["cap_hours"][platform] * 3600, "n_params": e["n_params"],
            "expect": {"build_number": pins["llama_cpp"]["build_number"], "build_commit": pins["llama_cpp"]["commit"],
                       "frozen_b_mib": frozen_b},
            "jobs": jobs}


def host_of(rec: dict) -> dict:
    os_info = (rec or {}).get("os") or {}
    return {"system": os_info.get("system"), "wsl": os_info.get("wsl"),
            "nvidia": bool(((rec or {}).get("gpu") or {}).get("nvidia"))}


def host_problems(host: dict | None, platform: str) -> list:
    want = {"system": "Darwin", "wsl": False, "nvidia": False} if platform == "macos-metal" else \
        {"system": "Linux", "wsl": True, "nvidia": True}
    return [] if host == want else [f"ran on host {host!r}, not {platform}'s {want}"]


def pilot_problems(path, platform: str, pins: dict, cfg: dict) -> list:
    path = Path(path)
    try:
        s = json.loads((path / "session.json").read_text())
        plan = json.loads((path / "plan.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"{path.name}: no readable session.json / plan.json ({exc})"]
    problems = []
    if (s.get("stage"), s.get("mode"), s.get("platform"), s.get("status")) != (STAGE, "pilot", platform, "finished"):
        problems.append(f"{path.name}: not a finished {platform} E3 pilot")
    problems += [f"{path.name}: {p}" for p in host_problems(s.get("host"), platform)]
    if s.get("plan_sha256") != sha256_json(plan):
        problems.append(f"{path.name}: plan.json is not the plan the pilot ran (plan_sha256)")
    want = build_plan(pins, cfg, platform=platform, mode="pilot")
    if any(plan.get(k) != want[k] for k in ("files", "params", "expect", "jobs", "cap_s", "n_params")):
        problems.append(f"{path.name}: the pilot's plan is not build_plan's {platform} pilot plan under this config")
    out = s.get("outcomes") or {}
    if out.get("ok") != len(plan.get("jobs") or [None]) or any(out.get(k) for k in ("oom", "error", "skipped")):
        problems.append(f"{path.name}: not every job ok ({out})")
    if s.get("env_valid") is not True:
        problems.append(f"{path.name}: env not valid ({s.get('env_flags')})")
    if platform == "macos-metal":
        adapter = s.get("ac_adapter") or {}
        w = adapter.get("wattage_w")
        if {k: adapter.get(k) for k in MAC_ADAPTER_REQUIRED} != MAC_ADAPTER_REQUIRED or \
                not (isinstance(w, int) and not isinstance(w, bool) and w > 0):
            problems.append(f"{path.name}: the pilot's adapter {adapter} is not the Apple 140W with its watts recorded")
    elif not (isinstance(s.get("session_b"), dict) and s["session_b"].get("drift") is False):
        problems.append(f"{path.name}: no undrifted session B ({s.get('session_b')})")
    if s.get("ggufs_sha256") != {f["filename"]: f["sha256"] for f in want["files"].values()}:
        problems.append(f"{path.name}: not all eight GGUFs were verified against their pins")
    if s.get("header_verified") != {k: True for k in want["files"]}:
        problems.append(f"{path.name}: not all eight GGUF headers matched configs/stage_e3.toml")
    if s.get("harness_sha256") != harness_sha256():
        problems.append(f"{path.name}: ran another harness (harness_sha256)")
    return problems


def freeze_evidence_problems(cfg: dict, root, pins: dict, platforms=PLATFORMS) -> list:
    problems = []
    for platform in platforms:
        key = FREEZE_KEYS[platform]
        if cfg["freeze"].get(key) is True:
            path = Path(root).joinpath(*RESULTS, str(cfg["freeze"].get(f"pilot_{key}")))
            problems += [f"freeze.pilot_{key}: {p}" for p in pilot_problems(path, platform, pins, cfg)]
    return problems


# ---------------------------------------------------------------- platform rules

def mac_problems(rec: dict) -> list:
    """037 §4: powermode 0 or 2, the Apple 140W adapter (its negotiated watts are recorded, not required),
    iogpu.wired_limit_mb 0."""
    power = (rec or {}).get("power") or {}
    wired = ((rec or {}).get("hardware") or {}).get("iogpu_wired_limit_mb")
    problems = []
    if power.get("powermode") not in POWERMODES_ALLOWED:
        problems.append(f"powermode {power.get('powermode')!r} (0 or 2)")
    adapter = power.get("ac_adapter") or {}
    got = {k: adapter.get(k) for k in MAC_ADAPTER_REQUIRED}
    if got != MAC_ADAPTER_REQUIRED:
        problems.append(f"AC adapter {got['manufacturer']!r} {got['name']!r} (037 §4 requires Apple Inc. "
                        f"{MAC_ADAPTER_REQUIRED['name']!r})")
    if wired != 0:
        problems.append(f"iogpu.wired_limit_mb {wired!r} (§4 requires 0)")
    return problems


def platform_problems(rec: dict, platform: str) -> list:
    problems = env_problems(rec, platform)
    if platform != "macos-metal":
        return problems
    phase1 = mac_config_problems(rec)
    return [p for p in problems if p not in phase1] + mac_problems(rec)


def mac_lock_owned(lock_dir: str = MAC_LOCK_DIR, pid: int | None = None, parent_of: Callable | None = None) -> bool:
    """tools/with-mac-lock's owner.txt names an ancestor of this process (the env flag alone is never trusted)."""
    try:
        text = (Path(lock_dir) / "owner.txt").read_text()
        owner = int(next(x for x in text.splitlines() if x.startswith("pid=")).split("=", 1)[1])
    except (OSError, ValueError, StopIteration):
        return False
    parent_of = parent_of or _parent_pid
    pid = os.getpid() if pid is None else pid
    for _ in range(64):
        if pid == owner:
            return True
        if pid in (0, 1) or pid is None:
            return False
        pid = parent_of(pid)
    return False


def _parent_pid(pid: int):  # pragma: no cover - ps
    import subprocess
    try:
        return int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True,
                                  timeout=10).stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def default_lock_check(platform: str) -> Callable:  # pragma: no cover - /proc or ps
    if platform == "macos-metal":
        return mac_lock_owned
    import run_adv
    return run_adv.held_gpu_lock


def output_problems(out_dir, *, repo_root=ROOT) -> list:
    """A session directory is exactly results/stage_e/<session id> (no nesting: every attempt is visible)."""
    root = Path(repo_root).resolve().joinpath(*RESULTS)
    if Path(out_dir).resolve().parent != root:
        return [f"{out_dir} is not results/stage_e/<session id> (041)"]
    return []


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def formal_attempts(repo_root, platform: str) -> list:
    """Every formal E3 attempt of the platform under results/stage_e, oldest first: a directory whose session.json
    (final or the running stub) or, when that is unreadable, plan.json is a formal E3 one for the platform; runs refused
    before their first job are left out. valid = finished, env valid and no sentinel drift; an unreadable
    session.json counts as a void attempt, oldest (fail closed)."""
    out = []
    base = Path(repo_root).joinpath(*RESULTS)
    dirs = {f.parent for name in ("session.json", "plan.json") for f in base.rglob(name)} - {base}
    for d in sorted(dirs):
        s = _read_json(d / "session.json")
        if not isinstance(s, dict) and not any(x.name != "plan.json" for x in d.iterdir()):
            continue                                        # a saved plan alone: nothing ran here
        head = s if isinstance(s, dict) else _read_json(d / "plan.json")
        if not isinstance(head, dict) or (head.get("stage"), head.get("mode"), head.get("platform")) != \
                (STAGE, "formal", platform):
            continue
        s = s if isinstance(s, dict) else {}
        if s.get("status") == "refused":
            continue
        valid = s.get("status") == "finished" and s.get("env_valid") is True and s.get("sentinel_ok") is True
        out.append({"dir": d, "started_at": s.get("started_at") or "", "valid": valid})
    return sorted(out, key=lambda a: (a["started_at"], str(a["dir"])))


def header_problems(files: dict, models_dir, read_header: Callable) -> tuple:
    """({config: header matches}, problems): each GGUF's header against its pinned values."""
    verified, bad = {}, []
    for key, f in sorted(files.items()):
        try:
            got = pe.header_facts(read_header(Path(str(models_dir)) / f["filename"]))
        except (OSError, ValueError) as exc:
            verified[key] = False
            bad.append(f"{key}: header unreadable ({exc})")
            continue
        diff = {k: (got[k], f[k]) for k in HEADER_KEYS if got[k] != f[k]}
        verified[key] = not diff
        if diff:
            bad.append(f"{key}: header (got, pinned) {diff}")
    return verified, bad


def identity_problems(stdout: str, job: dict, files: dict) -> list:
    """041 v3 §3.6: the llama-bench row's model_filename, model_size and model_type are the config's."""
    rows = cap.parse_bench_jsonl(stdout)
    if len(rows) != 1:
        return [f"{len(rows)} result rows"]
    r, f = rows[0], files[f"{job['size']}-{job['quant']}"]
    got = {"model_filename": Path(str(r.get("model_filename") or "")).name, "model_size": r.get("model_size"),
           "model_type": r.get("model_type")}
    want = {"model_filename": job["model"], "model_size": f["model_size"], "model_type": f["model_type"]}
    return [f"{k} {got[k]!r} is not {want[k]!r}" for k in want if got[k] != want[k]]


# ---------------------------------------------------------------- the runner

def run_e3(plan: dict, *, bin_dir, models_dir, out_dir, capture_env: Callable, session_id: str, lock_held: bool,
           pins: dict, cfg: dict, lock_check: Callable | None = None, run_proc: Callable = default_run_proc,
           sleep: Callable = time.sleep, read_power: Callable = read_mac_power, measure_b: Callable | None = None,
           hash_file: Callable = file_sha256, read_header: Callable = gguf_tensors.read_header,
           expected_plan: dict | None = None, repo_root=ROOT, clock: Callable = time.monotonic, timeout_s: float = 7200,
           now: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat()) -> dict:
    platform, mode = plan.get("platform"), plan.get("mode")
    if plan.get("stage") != STAGE or platform not in PLATFORMS or mode not in ("pilot", "formal"):
        raise StageE3Error(f"not an E3 pilot or formal plan ({plan.get('stage')!r}, {platform!r}, {mode!r})")
    problems = output_problems(out_dir, repo_root=repo_root)
    if problems:
        raise StageE3Error("; ".join(problems))
    mac = platform == "macos-metal"
    if not (lock_held and (lock_check or default_lock_check(platform))()):
        raise StageE3Error("E3 on the Mac runs under the Mac heavy lock (tools/with-mac-lock)" if mac else
                           "E3 on the RTX holds the GPU lock (flock -n ... --gpu-lock-held; verified, not trusted)")
    formal = mode == "formal"
    if plan.get("harness_sha256") != harness_sha256():
        raise StageE3Error(f"the {mode} plan was built by another harness (harness_sha256)")
    if expected_plan is None or plan != expected_plan:
        raise StageE3Error(f"a {mode} plan must be exactly build_plan's for this platform, pins and "
                           "configs/stage_e3.toml (041 v3 §3.6, 3.7)")
    out = Path(out_dir)
    if out.exists():
        raise StageE3Error(f"{out} exists; sessions are never resumed or overwritten")
    if formal:
        prior = formal_attempts(repo_root, platform)
        if len(prior) >= 2 or any(a["valid"] for a in prior):
            raise StageE3Error(f"{platform} already has formal E3 attempts {[a['dir'].name for a in prior]}: one valid "
                               "session, or one void attempt and its rerun, at most (041 v3 §3.6)")
    shas, bad = {}, []
    for f in plan["files"].values():                        # all eight, used or not (041 v3 §3.6)
        try:
            shas[f["filename"]] = hash_file(Path(str(models_dir)) / f["filename"])
        except OSError as exc:
            bad.append(f"{f['filename']}: unreadable ({exc})")
            continue
        if shas[f["filename"]] != f["sha256"]:
            bad.append(f"{f['filename']}: sha256 {shas[f['filename']]} is not the pinned {f['sha256']}")
    verified, header_bad = header_problems(plan["files"], models_dir, read_header)
    if bad or header_bad:
        raise StageE3Error("GGUF check failed before the session: " + "; ".join(bad + header_bad))
    want = plan["expect"]
    session_b = None
    if not mac:
        if measure_b is None:
            raise StageE3Error("E3 on the RTX measures B at the session start (§4; stage_b.measure_baseline)")
        try:
            session_b = measure_b(want["frozen_b_mib"])
        except stage_b.StageBError as exc:
            raise StageE3Error(f"session B not measured, nothing started: {exc}") from exc
    for sub in ("raw", "env"):
        (out / sub).mkdir(parents=True)
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": STAGE, "platform": platform,
               "mode": mode, "seed": plan["seed"], "plan_sha256": sha256_json(plan),
               "pins_sha256": plan.get("pins_sha256"), "harness_sha256": harness_sha256(), "status": "running",
               "lock_held": lock_held, "ggufs_sha256": shas, "header_verified": verified, "session_b": session_b,
               "started_at": now()}

    def write_session():                                    # atomic: a torn write never hides the attempt
        tmp = out / "session.json.tmp"
        tmp.write_text(json.dumps(session, indent=1) + "\n")
        os.replace(tmp, out / "session.json")

    write_session()                                         # the stub: an interrupted run is still an attempt
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    bin_dir, models_dir = str(bin_dir), str(models_dir)
    llama_bin = f"{bin_dir}/llama-bench"
    pre = capture_env(llama_bin=llama_bin, out_path=out / "env" / "preflight.json", gpu_lock_held=not mac)
    session.update({"host": host_of(pre), "powermode": ((pre.get("power") or {}).get("powermode")) if mac else None,
                    "ac_adapter": ((pre.get("power") or {}).get("ac_adapter")) if mac else None})
    lc = (pre.get("runtime") or {}).get("llama_cpp") or {}
    refuse = [] if lc.get("build_number") == want["build_number"] and lc.get("commit") == want["build_commit"] else \
        [f"llama.cpp build mismatch: {lc} vs {want}"]
    if formal:
        refuse += platform_problems(pre, platform) + provenance_problems(pre, pre) + \
            host_problems(host_of(pre), platform)
    if refuse:
        session.update({"status": "refused", "refused": refuse})
        write_session()
        raise StageE3Error("E3 run refused before its first job: " + "; ".join(refuse))
    powermode = session["powermode"]
    params = plan["params"]
    bench_expect = {"build_commit": want["build_commit"], "n_threads": params["n_thr"], "poll": params["poll"],
                    "reps": params["bench_r"]}
    rows, env_flags, charge_pauses = [], [], []
    if session_b and session_b.get("drift"):
        env_flags.append({"job_id": None, "phase": "preflight",
                          "reason": f"session B {session_b.get('b_mib')} MiB drifted from the frozen B (§4)"})
    if mac:
        pause = charge_gate(read_power, sleep)              # before the clock starts
        if pause:
            charge_pauses.append({"job_id": None, **pause})
    t0, paused, cap_reached = clock(), 0.0, False
    full = open(out / "raw" / "rows.full.jsonl", "w") if not formal else None
    env_dir = "env" if formal else "raw/env"               # a pilot's per-job captures carry its job times
    (out / env_dir).mkdir(exist_ok=True)

    def emit(fh, row):                                      # a pilot commits no speed (041 v3 §3.7: 結果遮蔽)
        if full is not None:
            full.write(json.dumps(row) + "\n")
            full.flush()
            row = {k: row[k] for k in PILOT_ROW_KEYS if k in row}
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    with open(out / "rows.jsonl", "w") as fh:
        for i, job in enumerate(plan["jobs"]):
            jid, end = job["id"], job.get("position") == "end"
            base = {"job_id": jid, "kind": job["kind"], "position": job.get("position"), "size": job["size"],
                    "quant": job["quant"], "L": job["L"], "rep": job["rep"], "model": job["model"]}
            if not (cap_reached and not end):
                if i:
                    sleep(params["idle_s"])
                if mac:
                    pause = charge_gate(read_power, sleep)
                    if pause:
                        charge_pauses.append({"job_id": jid, **pause})
                        paused += pause["waited_s"]
                elapsed = clock() - t0 - paused             # measurement time: charge pauses do not count
                cap_reached = cap_reached or (not end and elapsed > plan["cap_s"])
            if cap_reached and not end:                     # the end sentinel always runs: the drift is measured
                row = {**base, "outcome": "skipped", "reason": f"time cap {plan['cap_s']:.0f} s reached (041 v3 §3.6)"}
                rows.append(row)
                emit(fh, row)
                continue
            job_timeout = timeout_s if end else min(timeout_s, max(plan["cap_s"] - elapsed, 0) + CAP_MARGIN_S)
            argv = [a.replace("{bin}", bin_dir).replace("{models}", models_dir) for a in job["argv"]]
            rel = {"stdout": f"raw/{jid}.stdout.txt", "stderr": f"raw/{jid}.stderr.txt",
                   "env_before": f"{env_dir}/{jid}.before.json", "env_after": f"{env_dir}/{jid}.after.json"}
            env_before = capture_env(llama_bin=llama_bin, out_path=out / rel["env_before"], gpu_lock_held=not mac)
            code, wall = run_proc(argv, out / rel["stdout"], out / rel["stderr"], job_timeout)
            env_after = capture_env(llama_bin=llama_bin, out_path=out / rel["env_after"], gpu_lock_held=not mac)
            for phase, rec in (("before", env_before), ("after", env_after)):
                why = platform_problems(rec, platform)
                if formal:
                    why += provenance_problems(rec, pre)
                if mac and ((rec.get("power") or {}).get("powermode")) != powermode:
                    why.append(f"powermode {(rec.get('power') or {}).get('powermode')!r} != preflight {powermode!r} "
                               "(one powermode per session, §4)")
                env_flags.extend({"job_id": jid, "phase": phase, "reason": w} for w in why)
            stdout = (out / rel["stdout"]).read_text(errors="replace")
            result = cap.classify_bench(code, stdout, (out / rel["stderr"]).read_text(errors="replace"),
                                        {**bench_expect, **job["expect"]})
            if result["outcome"] == "ok":
                wrong = identity_problems(stdout, job, plan["files"])
                if wrong:
                    result = {**result, "outcome": "error", "stage": "identity", "reason": "; ".join(wrong)}
            row = {**base, "argv": argv, "exit_code": code, "wall_s": wall, "timeout_s": job_timeout,
                   "outcome": result["outcome"], "stage": result["stage"], "reason": result["reason"],
                   "evidence": result["evidence"], "metrics": result.get("metrics"), "summary": result.get("summary"),
                   **rel}
            rows.append(row)
            emit(fh, row)
    if full is not None:
        full.close()
    outcomes = {"ok": 0, "oom": 0, "error": 0, "skipped": 0}
    for r in rows:
        outcomes[r["outcome"]] += 1
    sentinel = _sentinel_summary([r for r in rows if r["outcome"] != "skipped"])
    if not formal:
        sentinel = {k: sentinel.get(k) for k in ("drift", "drift_pct", "threshold_pct", "note") if k in sentinel}
    session.update({"status": "finished", "finished_at": now(), "outcomes": outcomes, "sentinel": sentinel,
                    "sentinel_ok": sentinel.get("drift") is False, "cap_reached": cap_reached,
                    "measured_s": clock() - t0 - paused, "env_flags": env_flags, "env_valid": not env_flags,
                    "charge_pauses": charge_pauses})
    write_raw_manifest(out)
    write_session()
    return session


# ---------------------------------------------------------------- CLI

def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--mode", required=True, choices=["pilot", "formal"])
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    for a in ("--plan", "--bin", "--models", "--out"):
        p.add_argument(a, required=True)
    p.add_argument("--gpu-lock-held", action="store_true")
    args = parser.parse_args(argv)
    raw, cfg_raw = (ROOT / "configs" / "pins.toml").read_bytes(), CONFIG.read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(cfg_raw.decode())

    def plan_for(platform: str, mode: str) -> dict:
        return build_plan(pins, cfg, platform=platform, mode=mode, pins_sha256=hashlib.sha256(raw).hexdigest())

    try:
        if args.cmd == "plan":
            out = Path(args.out).resolve()
            if ROOT.joinpath(*RESULTS).resolve() in (out, *out.parents):
                raise StageE3Error("write plans outside results/stage_e (a session directory is the runner's)")
            out.write_text(json.dumps(plan_for(args.platform, args.mode), indent=1) + "\n")
            return 0
        plan = json.loads(Path(args.plan).read_text())
        platform = plan.get("platform")
        measure_b = None
        if platform == "wsl2-cuda":
            import stage_b
            measure_b = lambda frozen: stage_b.measure_baseline(  # noqa: E731
                query=stage_b.nvidia_query, compute_apps=stage_b.nvidia_compute_apps, sleep=time.sleep,
                frozen_b_mib=frozen)
        s = run_e3(plan, bin_dir=Path(args.bin).resolve(), models_dir=Path(args.models).resolve(), out_dir=args.out,
                   capture_env=make_env_capture(ROOT / "scripts" / "env_capture.py"), session_id=Path(args.out).name,
                   lock_held=(os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") == "1") if platform == "macos-metal"
                   else args.gpu_lock_held, pins=pins, cfg=cfg, measure_b=measure_b,
                   expected_plan=plan_for(platform, plan.get("mode")))
    except StageE3Error as exc:
        print(f"stage_e3: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"outcomes": s["outcomes"], "sentinel_ok": s["sentinel_ok"], "env_valid": s["env_valid"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
