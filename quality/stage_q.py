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
runs PILOT_CHUNKS per job and records markers, time and disk.

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
                        make_env_capture, provenance_problems, read_mac_power, sha256_json, write_raw_manifest)

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
MAC_ADAPTER_REQUIRED = {"manufacturer": "Apple Inc.", "name": "140W USB-C Power Adapter"}   # 037 §4
RESULTS = ("results", "stage_q")


class StageQError(RuntimeError):
    pass


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


def kld_need_gib(chunks: int, ctx: int) -> float:
    """The base file's size for chunks of ctx (scored: the second half of each chunk), plus 10 %."""
    gb = KLD_GB_PER_CHUNK_512 * chunks * (ctx // 512)
    return round(gb * 1.1 * 1e9 / disk_guard.GiB, 1)


def build_plan(pins: dict, *, part: str, mode: str, pins_sha256: str | None = None) -> dict:
    if part not in PARTS:
        raise StageQError(f"unknown part {part!r}; 040 defines {sorted(PARTS)}")
    if mode not in ("pilot", "formal"):
        raise StageQError(f"unknown mode {mode!r}")
    platform = PARTS[part]
    if mode == "formal" and not pins["freeze"]["nvidia"]:
        raise StageQError("formal Stage Q plans need pins freeze.nvidia = true (as run_quality, 020)")
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
            kld = f"{{data}}/kld/stage-q-{tag}.kld"
            n_kld = PILOT_CHUNKS if pilot else (Q1_CHUNKS if part.startswith("q1") else expected_chunks(pins, ctx))
            chunks = ["--chunks", str(n_kld)] if (pilot or part.startswith("q1")) else []

            def add(jid, kind, point, argv, expect, **extra):
                jobs.append({"id": jid, "kind": kind, "size": size, "point": point, "ctx": ctx, "argv": argv,
                             "expect": expect, **extra})

            add(f"kld_base-{tag}", "kld_base", "BF16",
                [exe, *model("BF16"), "-f", "{data}/datasets/wiki.test.raw", "--kl-divergence-base", kld, *chunks, *f],
                {"n_chunks": n_kld}, need_gib=kld_need_gib(n_kld, ctx))
            add(f"kld_self-{tag}", "kld_self", "BF16",
                [exe, *model("BF16"), "--kl-divergence-base", kld, "--kl-divergence", *f], {"n_chunks": n_kld})
            for q in quants:
                add(f"kld-{tag}-{q}", "kld", q,
                    [exe, *model(q), "--kl-divergence-base", kld, "--kl-divergence", *f], {"n_chunks": n_kld})
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
            "pins_sha256": pins_sha256, "params": {"n_thr": n_thr, "poll": poll},
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
    root = Path(repo_root).resolve().joinpath(*RESULTS)
    out = Path(out_dir).resolve()
    if out == root or root not in out.parents:
        return [f"{out_dir} is not a session directory under results/stage_q/ (040)"]
    return []


def _default_disk_ok(path, need_gib: float) -> dict:
    return disk_guard.check(str(path), need_bytes=int(need_gib * disk_guard.GiB),
                            reserve_bytes=disk_guard.DEFAULT_RESERVE_GIB * disk_guard.GiB)


def run_stage_q(plan: dict, *, bin_dir, models_dir, data_dir, out_dir, capture_env: Callable, session_id: str,
                mac_lock_held: bool = False, gpu_lock_held: bool = False, disk_ok: Callable = _default_disk_ok,
                run_proc: Callable = default_run_proc, delete_file: Callable = os.remove,
                read_power: Callable = read_mac_power, sleep: Callable = time.sleep, timeout_s: float = 6 * 3600,
                now: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(), repo_root=ROOT) -> dict:
    """One Stage Q session: run_quality's loop with 040's platform rules (q1-metal formal on the Mac under the Mac lock
    with the 140W adapter and one powermode; the CUDA parts under the GPU lock)."""
    platform, part = plan["platform"], plan["part"]
    mac = platform == "macos-metal"
    if plan.get("stage") != STAGE or part not in PARTS or PARTS[part] != platform:
        raise StageQError(f"not a Stage Q plan ({plan.get('stage')!r}, {part!r}, {platform!r})")
    problems = output_problems(out_dir, repo_root=repo_root)
    if problems:
        raise StageQError("; ".join(problems))
    if mac and not mac_lock_held:
        raise StageQError("q1-metal runs only under the Mac heavy lock: tools/with-mac-lock ... (MACHINE_POLICY; 037 §4)")
    if not mac and not gpu_lock_held:
        raise StageQError("the CUDA parts must hold the GPU lock: flock -n /tmp/ai-portfolio-gpu.lock ... "
                          "--gpu-lock-held (MACHINE_POLICY, 020)")
    out = Path(out_dir)
    if out.exists():
        raise StageQError(f"{out} exists; sessions are never resumed or overwritten")
    for sub in ("raw", "env"):
        (out / sub).mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    bin_dir, models_dir, data_dir = str(bin_dir), str(models_dir), str(data_dir)
    exe = f"{bin_dir}/llama-perplexity"
    pre = capture_env(llama_bin=exe, out_path=out / "env" / "preflight.json", gpu_lock_held=gpu_lock_held)
    lc = (pre.get("runtime") or {}).get("llama_cpp") or {}
    want = plan["expect_build"]
    if lc.get("build_number") != want["build_number"] or lc.get("commit") != want["build_commit"]:
        raise StageQError(f"llama.cpp build mismatch: {lc} vs {want}")
    if plan["mode"] == "formal":
        bad = env_problems(pre, platform) + provenance_problems(pre, pre) + (mac_problems(pre) if mac else [])
        if bad:
            raise StageQError("formal Stage Q run refused before it starts: " + "; ".join(bad))
    powermode = ((pre.get("power") or {}).get("powermode")) if mac else None
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": STAGE, "part": part,
               "mode": plan["mode"], "platform": platform, "plan_sha256": sha256_json(plan),
               "pins_sha256": plan.get("pins_sha256"), "mac_lock_held": mac_lock_held, "gpu_lock_held": gpu_lock_held,
               "powermode": powermode, "started_at": now()}
    rows, by_id, skipped, env_flags, charge_pauses = [], {}, set(), [], []
    resolve = lambda a: a.replace("{bin}", bin_dir).replace("{models}", models_dir).replace("{data}", data_dir)  # noqa: E731
    with open(out / "rows.jsonl", "w") as fh:
        for job in plan["jobs"]:
            jid, kind, size = job["id"], job["kind"], job["size"]
            group = (size, job.get("ctx"))
            base = {"job_id": jid, "kind": kind, "size": size, "point": job.get("point"), "ctx": job.get("ctx")}
            if kind == "delete_kld":
                path = resolve(job["path"])
                all_ok = all(by_id.get(a, {}).get("outcome") == "ok" for a in job["after"])
                deleted = False
                if all_ok and os.path.exists(path):
                    delete_file(path)
                    deleted = True
                row = {**base, "outcome": "ok" if all_ok else "error", "deleted": deleted, "path": path,
                       "reason": None if all_ok else "KLD runs not all ok; KLD file kept for diagnosis"}
            elif kind in ("kld_base", "kld_self", "kld") and group in skipped:
                row = {**base, "outcome": "skipped", "reason": "disk guard failed for this group's KLD file"}
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
                    flags_ = env_problems(rec, platform)
                    if plan["mode"] == "formal":
                        flags_ += provenance_problems(rec, pre)
                        if mac:
                            flags_ += mac_problems(rec)
                            if ((rec.get("power") or {}).get("powermode")) != powermode:
                                flags_.append(f"powermode changed to {(rec.get('power') or {}).get('powermode')!r} "
                                              f"from {powermode!r} (one powermode per session, 019)")
                    env_flags.extend({"job_id": jid, "phase": phase, "reason": why} for why in flags_)
                res = qp.classify(kind, code, (out / rel["stdout"]).read_text(errors="replace"),
                                  (out / rel["stderr"]).read_text(errors="replace"), job["expect"])
                row = {**base, "argv": argv, "exit_code": code, "wall_s": wall, **res, "disk_guard": guard, **rel}
            rows.append(row)
            by_id[jid] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    outcomes = {"ok": 0, "oom": 0, "error": 0, "skipped": 0}
    for r in rows:
        outcomes[r["outcome"]] += 1
    session.update({"finished_at": now(), "outcomes": outcomes, "env_flags": env_flags, "env_valid": not env_flags,
                    "charge_pauses": charge_pauses})
    write_raw_manifest(out)
    (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
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
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        raw = (ROOT / "configs" / "pins.toml").read_bytes()
        plan = build_plan(tomllib.loads(raw.decode()), part=args.part, mode=args.mode,
                          pins_sha256=hashlib.sha256(raw).hexdigest())
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
        return 0
    plan = json.loads(Path(args.plan).read_text())
    try:
        summary = run_stage_q(plan, bin_dir=Path(args.bin).resolve(), models_dir=Path(args.models).resolve(),
                              data_dir=Path(args.data).resolve(), out_dir=args.out,
                              capture_env=make_env_capture(ROOT / "scripts" / "env_capture.py"),
                              session_id=Path(args.out).name,
                              mac_lock_held=os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") == "1",
                              gpu_lock_held=args.gpu_lock_held)
    except StageQError as exc:
        print(f"stage_q: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary["outcomes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
