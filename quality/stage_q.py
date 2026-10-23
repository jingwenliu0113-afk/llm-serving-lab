#!/usr/bin/env python3
"""Stage Q plans and runs (decision 040; PREREGISTRATION §10) with llama-perplexity, in run_quality's job format and
with its rules (quality_parse markers and counts, one process per job, the KLD base deleted once its runs are all ok,
env capture around every job). run_quality itself (Phase 1) is imported, never changed.

Parts (each its own session):
- q1-metal (macos-metal, Mac lock): per size, a KLD base of the first Q1_CHUNKS chunks of WikiText (BF16, Metal) ->
  self-KLD -> Q8_0 / Q6_K / Q4_K_M KLD -> delete the base; then the full-test PPL (580 chunks) of BF16 and the three
  quants. Mac rules: AC, battery >= 20 %, no thermal warning (012), the 016 charge gate, one powermode (0 or 2) for the
  session, iogpu.wired_limit_mb 0 and the Apple 140W USB-C Power Adapter (037 §4: Phase 3's Mac runs; Phase 1's 70 W
  rule stays with its sessions).
- q1-cuda (wsl2-cuda, GPU lock): the same KLD jobs on the same first Q1_CHUNKS chunks (the Metal comparison).
- q2 (wsl2-cuda, GPU lock): per context (2048, 8192) and size, a full-test KLD base -> self-KLD -> three quants ->
  delete, with -c set to the context (the rest of 005's flags unchanged).
Formal plans need pins freeze.nvidia (as run_quality) and, from the freeze on, configs/stage_q.toml's freeze; a pilot
runs PILOT_CHUNKS per job and records markers, time and disk (each KLD base's bytes).
Attempts (040 補記 14): a session directory is exactly results/stage_q/<id>; a session.json stub (status running) is
written before the first job and replaced atomically, so an interrupted session is still an attempt; a run refused
before its first job is status refused and is not one. A formal run refuses when its part already has a valid formal
session (finished, env valid) or two attempts, except attempts superseded by a git-tracked decision log under
docs/decisions/ that names both session ids (--superseded <id>=<log>).

Usage:
    python quality/stage_q.py plan --part q1-metal --mode formal --out plan.json
    tools/with-mac-lock python quality/stage_q.py run --plan plan.json --bin <llama bin> --models models/gguf \\
        --data data --out results/stage_q/<session>                        # q1-metal
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python quality/stage_q.py run ... --gpu-lock-held   # q1-cuda, q2
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_guard  # noqa: E402
import quality_parse as qp  # noqa: E402
import run_quality as rq  # noqa: E402
from run_core_a import (POWERMODES_ALLOWED, charge_gate, default_run_proc, env_problems,  # noqa: E402
                        mac_config_problems, make_env_capture, provenance_problems, read_mac_power, sha256_json,
                        write_raw_manifest)

SCHEMA_VERSION = 1
STAGE = "Q"
PARTS = {"q1-metal": "macos-metal", "q1-cuda": "wsl2-cuda", "q2": "wsl2-cuda"}
Q1_CHUNKS = 200
Q2_CONTEXTS = (2048, 8192)
PILOT_CHUNKS = 10
N_THR = {"macos-metal": 8, "wsl2-cuda": None}          # Mac: PREREGISTRATION §4; RTX: pins [stage_b] rtx_n_thr
POLL = {"macos-metal": 0, "wsl2-cuda": None}
# 005 / 008: 73.5 GB per size at 580 chunks of 255 scored tokens; a base scales with its scored tokens
KLD_GB_PER_CHUNK_512 = 73.5 / 580
SCORED_512 = 255                                       # llama-perplexity scores n_ctx - 1 - n_ctx / 2 tokens per chunk
MAC_ADAPTER_REQUIRED = {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"}   # 037 §4
RESULTS = ("results", "stage_q")
CONFIG = ROOT / "configs" / "stage_q.toml"
# stage_q's own modules plus the Phase 1 modules it runs through that no guard test pins (040 review): the flags
# (run_quality), the success markers and counts (quality_parse) and the disk guard
HARNESS_FILES = ("quality/stage_q.py", "quality/stage_q_analysis.py", "quality/run_quality.py",
                 "quality/quality_parse.py", "scripts/disk_guard.py")
MAC_LOCK_DIR = "/tmp/ai-portfolio-mac-heavy.lock"
FREEZE_EVIDENCE = {"pilot_q1_metal": "q1-metal", "pilot_q1_cuda": "q1-cuda", "pilot_q2": "q2"}
CONFIG_KEYS = {"schema_version": None, "freeze": {"stage_q", *FREEZE_EVIDENCE}, "stage_q": {"phase1_session"}}


class StageQError(RuntimeError):
    pass


def harness_sha256(root=ROOT) -> str:
    """sha256 over HARNESS_FILES (path and bytes): the harness a pilot or a formal session ran."""
    h = hashlib.sha256()
    for rel in HARNESS_FILES:
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


def config_problems(cfg: dict) -> list:
    """configs/stage_q.toml: exactly its keys; once frozen, every pilot named."""
    problems = []
    if set(cfg) != set(CONFIG_KEYS):
        problems.append(f"top-level keys {sorted(cfg)} are not {sorted(CONFIG_KEYS)}")
        return problems
    if cfg["schema_version"] != 1:
        problems.append(f"schema_version {cfg['schema_version']!r} is not 1")
    for table, keys in CONFIG_KEYS.items():
        if keys is not None and set(cfg[table]) != keys:
            problems.append(f"[{table}] keys {sorted(cfg[table])} are not {sorted(keys)}")
    if problems:
        return problems
    freeze = cfg["freeze"]
    if not isinstance(freeze["stage_q"], bool):
        problems.append("freeze.stage_q is not a bool")
    for key in FREEZE_EVIDENCE:
        if not isinstance(freeze[key], str) or (freeze["stage_q"] is True and not freeze[key]):
            problems.append(f"freeze.{key} must name the passed pilot once freeze.stage_q is true")
    return problems


def platform_problems(rec: dict, platform: str) -> list:
    """run_core_a.env_problems with 037 §4's Mac rules: on the Mac its Phase 1 configuration check (019's 70 W
    adapter) gives way to mac_problems (powermode, the 140W adapter, wired limit); AC, battery and thermal stay."""
    problems = env_problems(rec, platform)
    if platform != "macos-metal":
        return problems
    phase1 = mac_config_problems(rec)
    return [p for p in problems if p not in phase1] + mac_problems(rec)


def host_of(rec: dict) -> dict:
    """The machine a capture saw: OS, WSL and whether nvidia-smi reported a GPU."""
    os_info = (rec or {}).get("os") or {}
    return {"system": os_info.get("system"), "wsl": os_info.get("wsl"),
            "nvidia": bool(((rec or {}).get("gpu") or {}).get("nvidia"))}


def host_problems(host: dict | None, platform: str) -> list:
    want = {"system": "Darwin", "wsl": False, "nvidia": False} if platform == "macos-metal" else \
        {"system": "Linux", "wsl": True, "nvidia": True}
    return [] if host == want else [f"ran on {host!r}, not a {platform} host {want}"]


def pilot_problems(path, part: str, pins: dict | None = None) -> list:
    """A passed pilot of `part` (040 凍結): mode pilot on its platform's host, plan.json the session ran (plan_sha256)
    and, given pins, build_plan's pilot jobs for the part; one ok row per job; the current harness."""
    path = Path(path)
    try:
        s = json.loads((path / "session.json").read_text())
        plan = json.loads((path / "plan.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"{path.name}: no readable session.json / plan.json ({exc})"]
    out, problems = s.get("outcomes") or {}, []
    if (s.get("mode"), s.get("part"), s.get("platform")) != ("pilot", part, PARTS.get(part)):
        problems.append(f"{path.name}: not a {part} pilot (mode / part / platform)")
    problems += [f"{path.name}: {p}" for p in host_problems(s.get("host"), PARTS.get(part, ""))]
    if s.get("plan_sha256") != sha256_json(plan):
        problems.append(f"{path.name}: plan.json is not the plan the pilot ran (plan_sha256)")
    if pins is not None:
        want = build_plan(pins, part=part, mode="pilot")
        if any(plan.get(k) != want[k] for k in ("part", "mode", "platform", "params", "expect_build", "jobs")):
            problems.append(f"{path.name}: the pilot's plan is not build_plan's pilot plan for {part}")
    if out.get("ok") != len(plan.get("jobs") or [None]) or any(out.get(k) for k in ("oom", "error", "skipped")):
        problems.append(f"{path.name}: not a passed {part} pilot (one ok row per job; outcomes {out})")
    if s.get("harness_sha256") != harness_sha256():
        problems.append(f"{path.name}: ran another harness (harness_sha256 {s.get('harness_sha256')!r})")
    return problems


def freeze_evidence_problems(cfg: dict, root, pins: dict | None = None, parts=None) -> list:
    """Once freeze.stage_q is true: each named pilot (of `parts`, default all) exists under root/results/stage_q and
    passes pilot_problems. [] before the freeze."""
    freeze = cfg.get("freeze") or {}
    if freeze.get("stage_q") is not True:
        return []
    problems = []
    for key, part in sorted(FREEZE_EVIDENCE.items()):
        if parts is None or part in parts:
            path = Path(root).joinpath(*RESULTS, str(freeze.get(key)))
            problems += [f"freeze.{key}: {p}" for p in pilot_problems(path, part, pins)]
    return problems


def flags(n_thr: int, poll: int, ctx: int = 512) -> list:
    out = rq._flags(n_thr, poll)
    out[out.index("-c") + 1] = str(ctx)
    return out


WIKITEXT_TOKENS = 297_194                              # pins [quality.expected] comment: 297,194 / 512 = 580


def expected_chunks(pins: dict, ctx: int) -> int:
    """WikiText test chunks at this context (llama-perplexity: tokens // n_ctx): 580 at 512, 145 at 2048, 36 at 8192."""
    if WIKITEXT_TOKENS // 512 != pins["quality"]["expected"]["wikitext_chunks"]:
        raise StageQError("WIKITEXT_TOKENS disagrees with pins [quality.expected] wikitext_chunks")
    return WIKITEXT_TOKENS // ctx


def scored_per_chunk(ctx: int) -> int:
    """llama-perplexity scores positions n_ctx / 2 .. n_ctx - 2 (predicting the next token): 255 / 1,023 / 4,095."""
    return ctx - 1 - ctx // 2


def kld_need_gib(chunks: int, ctx: int) -> float:
    """The base file's size for chunks of ctx (it stores the scored positions' logits), plus 10 %."""
    gb = KLD_GB_PER_CHUNK_512 * chunks * scored_per_chunk(ctx) / SCORED_512
    return round(gb * 1.1 * 1e9 / disk_guard.GiB, 1)


def build_plan(pins: dict, *, part: str, mode: str, pins_sha256: str | None = None, cfg: dict | None = None,
               stage_q_sha256: str | None = None, root=ROOT) -> dict:
    """Pilot plans need nothing; formal plans need freeze.stage_q and this part's pilot (it ran on this machine) under
    root to pass pilot_problems (the other parts' pilots are checked by the analysis, on the Mac)."""
    if part not in PARTS:
        raise StageQError(f"unknown part {part!r}; 040 defines {sorted(PARTS)}")
    if mode not in ("pilot", "formal"):
        raise StageQError(f"unknown mode {mode!r}")
    platform = PARTS[part]
    if mode == "formal" and not pins["freeze"]["nvidia"]:
        raise StageQError("formal Stage Q plans need pins freeze.nvidia = true (as run_quality, 020)")
    if mode == "formal" and (cfg is None or config_problems(cfg) or cfg["freeze"]["stage_q"] is not True):
        raise StageQError("formal Stage Q plans need freeze.stage_q = true in configs/stage_q.toml (040 凍結)"
                          + (f": {'; '.join(config_problems(cfg))}" if cfg is not None and config_problems(cfg) else ""))
    if mode == "formal":
        evidence = freeze_evidence_problems(cfg, root, pins, parts=(part,))
        if evidence:
            raise StageQError("formal Stage Q plans need this part's passed pilot first (040 凍結): " + "; ".join(evidence))
    sb = pins.get("stage_b") or {}
    n_thr = N_THR[platform] or sb.get("rtx_n_thr")
    poll = POLL[platform] if POLL[platform] is not None else sb.get("rtx_poll")
    pilot = mode == "pilot"
    models, quants = pins["core_a"]["models"], pins["core_a"]["quants"]
    exe = "{bin}/llama-perplexity"
    jobs = []
    contexts = Q2_CONTEXTS if part == "q2" else (512,)
    for ctx in contexts:
        f = flags(n_thr, poll, ctx)
        for m in models:
            size = m["size"]
            gguf = {"BF16": m["reference"]["filename"], **{q: m["files"][q]["filename"] for q in quants}}
            model = lambda p: ["-m", "{models}/" + gguf[p]]  # noqa: E731
            tag = f"{size}-c{ctx}" if part == "q2" else size
            kld = f"{{data}}/kld/{{session}}/stage-q-{tag}.kld"                    # one base per session
            n_kld = PILOT_CHUNKS if pilot else (Q1_CHUNKS if part.startswith("q1") else expected_chunks(pins, ctx))
            chunks = ["--chunks", str(n_kld)] if (pilot or part.startswith("q1")) else []

            def add(jid, kind, point, argv, expect, **extra):
                jobs.append({"id": jid, "kind": kind, "size": size, "point": point, "ctx": ctx, "argv": argv,
                             "expect": expect, **extra})

            add(f"kld_base-{tag}", "kld_base", "BF16",
                [exe, *model("BF16"), "-f", "{data}/datasets/wiki.test.raw", "--kl-divergence-base", kld, *chunks, *f],
                {"n_chunks": n_kld}, need_gib=kld_need_gib(n_kld, ctx))
            add(f"kld_self-{tag}", "kld_self", "BF16",
                [exe, *model("BF16"), "--kl-divergence-base", kld, "--kl-divergence", *f], {"n_chunks": n_kld},
                base=f"kld_base-{tag}")
            for q in quants:
                add(f"kld-{tag}-{q}", "kld", q,
                    [exe, *model(q), "--kl-divergence-base", kld, "--kl-divergence", *f], {"n_chunks": n_kld},
                    base=f"kld_base-{tag}")
            jobs.append({"id": f"delete_kld-{tag}", "kind": "delete_kld", "size": size, "point": None, "ctx": ctx,
                         "path": kld, "after": [f"kld_base-{tag}", f"kld_self-{tag}",
                                                *(f"kld-{tag}-{q}" for q in quants)]})
            if part == "q1-metal":
                n_ppl = PILOT_CHUNKS if pilot else pins["quality"]["expected"]["wikitext_chunks"]
                for point in ("BF16", *quants):
                    add(f"ppl-{size}-{point}", "ppl", point,
                        [exe, *model(point), "-f", "{data}/datasets/wiki.test.raw", "--ppl-output-type", "1",
                         *(["--chunks", str(n_ppl)] if pilot else []), *f], {"n_chunks": n_ppl})
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "part": part, "mode": mode, "platform": platform,
            "pins_sha256": pins_sha256, "stage_q_sha256": stage_q_sha256, "harness_sha256": harness_sha256(),
            "params": {"n_thr": n_thr, "poll": poll},
            "expect_build": {"build_number": pins["llama_cpp"]["build_number"],
                             "build_commit": pins["llama_cpp"]["commit"]},
            "jobs": jobs}


def mac_problems(rec: dict) -> list:
    """037 §4 / §4 / 012: powermode 0 or 2, the Apple 140W adapter, iogpu.wired_limit_mb 0 (unknown is a problem)."""
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


def output_problems(out_dir, *, repo_root=ROOT) -> list:
    """A session directory is exactly results/stage_q/<id> (no nesting: every attempt is visible)."""
    if Path(out_dir).resolve().parent != Path(repo_root).resolve().joinpath(*RESULTS):
        return [f"{out_dir} is not results/stage_q/<session id> (040)"]
    return []


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def formal_attempts(root, part: str) -> list:
    """Every formal attempt of `part` under results/stage_q, oldest first (040 補記 14): a directory whose session.json
    (final or the running stub) or, when that is unreadable, plan.json is a formal one of the part; a plan saved alone
    and runs refused before their first job are left out. valid = finished and env valid; an unreadable session.json
    is a void attempt, oldest (fail closed)."""
    base = Path(root).joinpath(*RESULTS)
    out = []
    for d in sorted({f.parent for name in ("session.json", "plan.json") for f in base.rglob(name)} - {base}):
        s = _read_json(d / "session.json")
        if not isinstance(s, dict) and not any(x.name != "plan.json" for x in d.iterdir()):
            continue
        head = s if isinstance(s, dict) else _read_json(d / "plan.json")
        if not isinstance(head, dict) or (head.get("stage"), head.get("mode"), head.get("part")) != \
                (STAGE, "formal", part):
            continue
        s = s if isinstance(s, dict) else {}
        if s.get("status") == "refused":
            continue
        out.append({"dir": d, "id": d.name, "started_at": s.get("started_at") or "", "status": s.get("status"),
                    "valid": s.get("status") == "finished" and s.get("env_valid") is True,
                    "superseded": dict(s.get("superseded") or {})})
    return sorted(out, key=lambda a: (a["started_at"], str(a["dir"])))


def git_tracked(path, root=ROOT) -> bool:
    """Committed: tracked by git here; on the WSL copy (deploy_bundle: no .git), listed in DEPLOY_MANIFEST.json with
    exactly these bytes."""
    root, path = Path(root).resolve(), Path(path).resolve()
    if (root / ".git").exists():  # pragma: no cover - git
        return subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(path)],
                              capture_output=True).returncode == 0
    manifest = _read_json(root / "DEPLOY_MANIFEST.json")
    if not isinstance(manifest, dict) or root not in path.parents:
        return False
    return (manifest.get("files") or {}).get(path.relative_to(root).as_posix()) == file_sha256(path)


def superseded_problems(superseded: dict, new_id: str, *, root=ROOT, tracked: Callable = git_tracked) -> list:
    """040 補記 14's exception: each superseded session id names a git-tracked decision log under docs/decisions/ whose
    text names both that id and new_id (the decision lives outside every hashed file)."""
    problems = []
    for sid, log in sorted((superseded or {}).items()):
        path = (Path(root) / log).resolve()
        if sid == new_id:
            problems.append(f"--superseded {sid}: a session cannot supersede itself")
        elif not str(log).startswith("docs/decisions/") or Path(root).resolve().joinpath("docs", "decisions") \
                not in path.parents or path.relative_to(Path(root).resolve()).as_posix() != log or not path.is_file():
            problems.append(f"--superseded {sid}: {log} is not a repo-relative decision log docs/decisions/<file> "
                            "(recorded as typed, read on both machines)")
        elif not tracked(path):
            problems.append(f"--superseded {sid}: {log} is not committed (git ls-files)")
        elif not (_names(text := path.read_text(), sid) and _names(text, new_id)):
            problems.append(f"--superseded {sid}: {log} does not name both {sid} and {new_id}")
    return problems


def _names(text: str, sid: str) -> bool:
    """sid as a whole token (a rerun's id often extends the first one's, e.g. -c1 -> -c1b)."""
    return re.search(rf"(?<![\w-]){re.escape(sid)}(?![\w-])", text) is not None


def recorded_supersedes(attempts: list) -> dict:
    """{superseded id: (decision log, the attempt that recorded it)} from the attempts' session.json (each run records
    its --superseded), so the analysis cannot choose them afterwards."""
    return {sid: (log, a["id"]) for a in attempts for sid, log in a["superseded"].items()}


def recorded_supersede_problems(attempts: list, *, root=ROOT, tracked: Callable = git_tracked) -> list:
    """Every recorded supersede checks out: its log (superseded_problems) and an earlier attempt it names."""
    started = {a["id"]: a["started_at"] for a in attempts}
    out = []
    for sid, (log, by) in sorted(recorded_supersedes(attempts).items()):
        out += superseded_problems({sid: log}, by, root=root, tracked=tracked)
        if sid != by and not (sid in started and started[sid] < started[by]):
            out.append(f"{by} supersedes {sid}, which is not an earlier formal attempt of this part")
    return out


def default_gpu_lock_check() -> bool:  # pragma: no cover - /proc (WSL)
    import run_adv
    return run_adv.held_gpu_lock()


def mac_lock_owned(lock_dir: str = MAC_LOCK_DIR, pid: int | None = None, parent_of: Callable | None = None) -> bool:
    """Whether tools/with-mac-lock's owner.txt names an ancestor of this process (the flag alone is never trusted)."""
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
        if pid in (0, 1):
            return False
        pid = parent_of(pid)
        if pid is None:
            return False
    return False


def _parent_pid(pid: int):  # pragma: no cover - ps
    import subprocess
    try:
        return int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True,
                                  timeout=10).stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def input_problems(plan: dict, pins: dict, *, data_dir, models_dir, hash_file: Callable = file_sha256) -> list:
    """Before a formal session: wiki.test.raw and every GGUF the plan reads are pins' bytes (sha256), so the two
    platforms of H-Q1b / H-Q1c read the same inputs."""
    want = {"{data}/datasets/wiki.test.raw": pins["quality"]["files"]["wiki_test_raw"]["sha256"]}
    for m in pins["core_a"]["models"]:
        want["{models}/" + m["reference"]["filename"]] = m["reference"]["sha256"]
        want.update({"{models}/" + f["filename"]: f["sha256"] for f in m["files"].values()})
    used = sorted({a for j in plan["jobs"] for a in j.get("argv", []) if a in want})
    problems = []
    for a in used:
        path = a.replace("{data}", str(data_dir)).replace("{models}", str(models_dir))
        try:
            got = hash_file(path)
        except OSError as exc:
            problems.append(f"{path}: unreadable ({exc})")
            continue
        if got != want[a]:
            problems.append(f"{path}: sha256 {got} is not the pinned {want[a]}")
    return problems


def _default_disk_ok(path, need_gib: float) -> dict:
    return disk_guard.check(str(path), need_bytes=int(need_gib * disk_guard.GiB),
                            reserve_bytes=disk_guard.DEFAULT_RESERVE_GIB * disk_guard.GiB)


def run_stage_q(plan: dict, *, bin_dir, models_dir, data_dir, out_dir, capture_env: Callable, session_id: str,
                mac_lock_held: bool = False, gpu_lock_held: bool = False, disk_ok: Callable = _default_disk_ok,
                run_proc: Callable = default_run_proc, delete_file: Callable = os.remove,
                read_power: Callable = read_mac_power, sleep: Callable = time.sleep, timeout_s: float = 6 * 3600,
                now: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(), repo_root=ROOT,
                expected_plan: dict | None = None, pins: dict | None = None, lock_check: Callable | None = None,
                hash_file: Callable = file_sha256, size_of: Callable = os.path.getsize,
                superseded: dict | None = None, tracked: Callable = git_tracked) -> dict:
    """One Stage Q session: run_quality's loop with 040's platform rules (q1-metal on the Mac under the Mac lock with
    the 140W adapter and one powermode; the CUDA parts under the GPU lock, both verified, not trusted). A formal plan
    must equal build_plan's (expected_plan) and its inputs pins' bytes."""
    platform, part = plan["platform"], plan["part"]
    mac = platform == "macos-metal"
    if plan.get("stage") != STAGE or part not in PARTS or PARTS[part] != platform:
        raise StageQError(f"not a Stage Q plan ({plan.get('stage')!r}, {part!r}, {platform!r})")
    if plan["mode"] == "formal" and plan.get("harness_sha256") != harness_sha256():
        raise StageQError("the formal plan was built by another harness (harness_sha256); rebuild it with this one")
    problems = output_problems(out_dir, repo_root=repo_root)
    if problems:
        raise StageQError("; ".join(problems))
    lock_check = lock_check or (mac_lock_owned if mac else default_gpu_lock_check)
    if mac and not (mac_lock_held and lock_check()):
        raise StageQError("q1-metal runs only under the Mac heavy lock: tools/with-mac-lock ... (MACHINE_POLICY; 037 §4)")
    if not mac and not (gpu_lock_held and lock_check()):
        raise StageQError("the CUDA parts must hold the GPU lock: flock -n /tmp/ai-portfolio-gpu.lock ... "
                          "--gpu-lock-held (MACHINE_POLICY, 020)")
    formal = plan["mode"] == "formal"
    if formal and (expected_plan is None or plan != expected_plan):
        raise StageQError("a formal plan must be exactly build_plan's for this part, pins and configs/stage_q.toml "
                          "(040 凍結); rebuild it with stage_q.py plan --mode formal")
    out = Path(out_dir)
    if out.exists():
        raise StageQError(f"{out} exists; sessions are never resumed or overwritten")
    if formal:
        bad = superseded_problems(superseded or {}, session_id, root=repo_root, tracked=tracked)
        attempts = formal_attempts(repo_root, part)
        gone = set(superseded or {}) | set(recorded_supersedes(attempts))
        prior = [a for a in attempts if a["id"] not in gone]
        if len(prior) >= 2 or any(a["valid"] for a in prior):
            bad.append(f"{part} already has formal attempts {[a['id'] for a in prior]}: one valid session, or one "
                       "void attempt and its rerun, at most (040 補記 14; --superseded needs a decision log)")
        if bad:
            raise StageQError("; ".join(bad))
    for sub in ("raw", "env"):
        (out / sub).mkdir(parents=True)
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": STAGE, "part": part,
               "mode": plan["mode"], "platform": platform, "status": "running", "plan_sha256": sha256_json(plan),
               "pins_sha256": plan.get("pins_sha256"), "mac_lock_held": mac_lock_held, "gpu_lock_held": gpu_lock_held,
               "harness_sha256": harness_sha256(), "inputs_verified": formal, "superseded": dict(superseded or {}),
               "started_at": now()}

    def write_session():                                    # atomic: a torn write never hides the attempt
        tmp = out / "session.json.tmp"
        tmp.write_text(json.dumps(session, indent=1) + "\n")
        os.replace(tmp, out / "session.json")

    def refuse(why: str):
        session.update(status="refused", refused=why)
        write_session()
        raise StageQError(why)

    write_session()                                         # the stub: an interrupted session is still an attempt
    try:                                    # before the first job nothing ran: any end here is refused
        (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
        bin_dir, models_dir, data_dir = str(bin_dir), str(models_dir), str(data_dir)
        exe = f"{bin_dir}/llama-perplexity"
        pre = capture_env(llama_bin=exe, out_path=out / "env" / "preflight.json", gpu_lock_held=gpu_lock_held)
        lc = (pre.get("runtime") or {}).get("llama_cpp") or {}
        want = plan["expect_build"]
        if lc.get("build_number") != want["build_number"] or lc.get("commit") != want["build_commit"]:
            refuse(f"llama.cpp build mismatch: {lc} vs {want}")
        if formal:
            bad = platform_problems(pre, platform) + provenance_problems(pre, pre) + \
                host_problems(host_of(pre), platform)
            if not bad:
                bad = input_problems(plan, pins, data_dir=data_dir, models_dir=models_dir, hash_file=hash_file)
            if bad:
                refuse("formal Stage Q run refused before it starts: " + "; ".join(bad))
        powermode = ((pre.get("power") or {}).get("powermode")) if mac else None
        session.update(powermode=powermode, host=host_of(pre))
    except BaseException as exc:
        if session["status"] != "refused":
            session.update(status="refused", refused=f"interrupted before the first job: {type(exc).__name__}: {exc}")
            write_session()
        raise
    rows, by_id, skipped, env_flags, charge_pauses = [], {}, set(), [], []
    resolve = lambda a: a.replace("{bin}", bin_dir).replace("{models}", models_dir).replace(  # noqa: E731
        "{data}", data_dir).replace("{session}", session_id)
    with open(out / "rows.jsonl", "w") as fh:
        for job in plan["jobs"]:
            jid, kind, size = job["id"], job["kind"], job["size"]
            group = (size, job.get("ctx"))
            base = {"job_id": jid, "kind": kind, "size": size, "point": job.get("point"), "ctx": job.get("ctx")}
            if kind == "delete_kld":
                path = resolve(job["path"])
                all_ok = all(by_id.get(a, {}).get("outcome") == "ok" for a in job["after"])
                deleted = False
                base_bytes = size_of(path) if os.path.exists(path) else None
                if all_ok and base_bytes is not None:
                    delete_file(path)
                    deleted = True
                row = {**base, "outcome": "ok" if all_ok else "error", "deleted": deleted, "path": path,
                       "base_bytes": base_bytes,
                       "reason": None if all_ok else "KLD runs not all ok; KLD file kept for diagnosis"}
            elif kind in ("kld_base", "kld_self", "kld") and group in skipped:
                row = {**base, "outcome": "skipped", "reason": "disk guard failed for this group's KLD file"}
            elif kind in ("kld_self", "kld") and by_id.get(job["base"], {}).get("outcome") != "ok":
                row = {**base, "outcome": "skipped", "reason": f"its base job {job['base']} is not ok"}
            else:
                guard = None
                if kind == "kld_base":
                    guard = disk_ok(Path(data_dir) / "kld", job["need_gib"])
                    if not guard.get("ok"):
                        skipped.add(group)
                        row = {**base, "outcome": "skipped", "reason": f"disk guard failed: {guard}"}
                        rows.append(row); by_id[jid] = row; fh.write(json.dumps(row) + "\n"); fh.flush()
                        continue
                argv = [resolve(a) for a in job["argv"]]
                if kind == "kld_base":
                    Path(argv[argv.index("--kl-divergence-base") + 1]).parent.mkdir(parents=True, exist_ok=True)
                rel = {"stdout": f"raw/{jid}.stdout.txt", "stderr": f"raw/{jid}.stderr.txt",
                       "env_before": f"env/{jid}.before.json", "env_after": f"env/{jid}.after.json"}
                pause = charge_gate(read_power, sleep) if mac else None
                if pause:
                    charge_pauses.append({"job_id": jid, **pause})
                env_before = capture_env(llama_bin=exe, out_path=out / rel["env_before"], gpu_lock_held=gpu_lock_held)
                code, wall = run_proc(argv, out / rel["stdout"], out / rel["stderr"], timeout_s)
                env_after = capture_env(llama_bin=exe, out_path=out / rel["env_after"], gpu_lock_held=gpu_lock_held)
                for phase, rec in (("before", env_before), ("after", env_after)):
                    flags_ = platform_problems(rec, platform)
                    if formal:
                        flags_ += provenance_problems(rec, pre)
                        if mac:
                            if ((rec.get("power") or {}).get("powermode")) != powermode:
                                flags_.append(f"powermode changed to {(rec.get('power') or {}).get('powermode')!r} "
                                              f"from {powermode!r} (one powermode per session, 019)")
                    env_flags.extend({"job_id": jid, "phase": phase, "reason": why} for why in flags_)
                res = qp.classify(kind, code, (out / rel["stdout"]).read_text(errors="replace"),
                                  (out / rel["stderr"]).read_text(errors="replace"), job["expect"])
                row = {**base, "argv": argv, "exit_code": code, "wall_s": wall, **res, "disk_guard": guard, **rel}
                if kind == "kld_base":                      # 040 補記 5: the pilot confirms the disk use
                    kld_file = argv[argv.index("--kl-divergence-base") + 1]
                    row.update(base_bytes=size_of(kld_file) if os.path.exists(kld_file) else None,
                               need_gib=job["need_gib"])
            rows.append(row)
            by_id[jid] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    outcomes = {"ok": 0, "oom": 0, "error": 0, "skipped": 0}
    for r in rows:
        outcomes[r["outcome"]] += 1
    session.update({"status": "finished", "finished_at": now(), "outcomes": outcomes, "env_flags": env_flags,
                    "env_valid": not env_flags, "charge_pauses": charge_pauses})
    write_raw_manifest(out)
    write_session()
    return session


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--part", required=True, choices=sorted(PARTS))
    p_plan.add_argument("--mode", required=True, choices=["pilot", "formal"])
    p_plan.add_argument("--out", required=True)
    p_run = sub.add_parser("run")
    for a in ("--plan", "--bin", "--models", "--data", "--out"):
        p_run.add_argument(a, required=True)
    p_run.add_argument("--gpu-lock-held", action="store_true")
    p_run.add_argument("--superseded", action="append", default=[], help="<session id>=docs/decisions/<log>.md")
    args = parser.parse_args(argv)
    raw, cfg_raw = (ROOT / "configs" / "pins.toml").read_bytes(), CONFIG.read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(cfg_raw.decode())

    def plan_for(part: str, mode: str) -> dict:
        return build_plan(pins, part=part, mode=mode, pins_sha256=hashlib.sha256(raw).hexdigest(), cfg=cfg,
                          stage_q_sha256=hashlib.sha256(cfg_raw).hexdigest())

    try:
        if args.cmd == "plan":
            out = Path(args.out).resolve()
            if ROOT.joinpath(*RESULTS).resolve() in (out, *out.parents):
                raise StageQError("write plans outside results/stage_q (a session directory is the runner's)")
            out.write_text(json.dumps(plan_for(args.part, args.mode), indent=1) + "\n")
            return 0
        plan = json.loads(Path(args.plan).read_text())
        expected = plan_for(plan["part"], "formal") if plan.get("mode") == "formal" else None
        summary = run_stage_q(plan, bin_dir=Path(args.bin).resolve(), models_dir=Path(args.models).resolve(),
                              data_dir=Path(args.data).resolve(), out_dir=args.out,
                              capture_env=make_env_capture(ROOT / "scripts" / "env_capture.py"),
                              session_id=Path(args.out).name,
                              mac_lock_held=os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") == "1",
                              gpu_lock_held=args.gpu_lock_held, expected_plan=expected, pins=pins,
                              superseded=dict(x.split("=", 1) for x in args.superseded))
    except StageQError as exc:
        print(f"stage_q: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary["outcomes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
