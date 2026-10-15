#!/usr/bin/env python3
"""Stage C (C1) session runner (decision 030 8.4 / 8.6 / 8.7; PREREGISTRATION §8): one process, under a GPU lock it
already holds, runs sentinel -> Phase S -> confirm plan -> Phase C -> sentinel and decides every next step with the
functions capacity_report's replay re-runs (capacity_plan.search_next / build_confirm_plan / phase_c_boots,
capacity_analysis.search_bits / phase_c_outcome). Core and Advanced modules are imported, never changed. Runs in WSL
with the project venv, as one ssh-foreground command on one DEPLOYED_COMMIT; no deploy and no product test meanwhile.

Refusals before any server starts (RunError; nothing is written except, for a refused formal preflight, plan.json
and env/preflight.json):
- the output is not a session directory under results/capacity/ (never results/core_a|core_b|quality|advanced|
  decision_table, never anything under runs/: output_problems), or it exists (never resumed or overwritten; an
  aborted session reruns under a new id);
- --gpu-lock-held is missing or not true: held_gpu_lock is verified, not trusted, at the start and before every
  boot (MACHINE_POLICY; every mode boots a server, so the smokes need the lock too);
- capacity.toml breaks 030 (capacity_plan.capacity_problems) or the plan is not capacity_plan.build_plan of these
  capacity.toml and pins bytes (plan_problems, everything recomputed; both sha256 included);
- a mixed workload without its --mixed-dataset, or an incomplete toolchain (014);
- formal: freeze.capacity is not true, run_core_b.stage_b_problems is not empty, the caller's env carries an engine
  variable (030 8.6: none inherited), the server env it gives is not the one capacity_report checks every boot against
  (server_env_problems: an inherited OTEL_* variable stays in the Core B env), or a mixed dataset is not the pinned
  sha256;
- formal preflight: vLLM != pins [vllm].version, a power limit off its default, a dirty tree, or code that is not a
  deployed bundle (repo source DEPLOYED_COMMIT).

Session (030 8.4):
1. sentinel start: one boot, U2K at C 25. An invalid start sentinel stops the session: the 8.6 sentinel gate needs
   both sentinels valid, so nothing after it could ever be reported.
2. Phase S: one boot per workload in plan order; bracket then bisection by capacity_plan.search_next on the recorded
   search_pass (= capacity_analysis.search_bits: valid, maxR_run <= 1, mixed R_long <= 1; invalid = fail). A boot
   that fails before its first bench (a failed check, or a server already gone when the first bench would start) has
   no row and is retried once at the end of Phase S (phase_s_retries); a workload whose two boots fail gets no
   points. A server dying after its first bench makes the remaining search runs invalid (= fail) without a bench.
3. confirm plan: capacity_plan.build_confirm_plan of the Phase S histories -> confirm_plan.json, and its sha256 is
   written into session.json on disk before the first Phase C boot.
4. Phase C: blocks r = 1 .. blocks; capacity_plan.phase_c_boots gives the boots and their seeded point order from
   the earlier blocks' outcomes (capacity_analysis.phase_c_outcome: "ok" | "fail" | "invalid"), so a valid FAIL or an
   invalid run (never replaced, 013) drops its point and every higher one, and a boot without points is skipped.
   A Phase C or sentinel boot that fails its checks records each planned run as an invalid row.
5. sentinel end.

Per boot (030 8.6; run_adv's order): lock held (held_gpu_lock), memory.used <= 1000 MiB within 60 s, no compute
process on the GPU (any of the three stops the session), env capture before; the Core B server
(capacity_plan.server_argv_for: the Stage B values, no observation flag) through run_adv.start_server in its own
process group; /health within server_ready_timeout_s (1800 s); checks (b) ready and FLASH_ATTN, (c)
run_adv.parse_nondefault holds max_num_batched_tokens 2048 and kv_cache_memory_bytes K and no observation, T or G
key, (d) /metrics num_gpu_blocks / block_size = 248 / 528; then the runs; teardown (run_adv.shutdown_group under
run_adv.DeferredSignals: SIGINT -> SIGTERM -> SIGKILL on the whole group, interrupts deferred until it is down, the
log reader joined); env capture after. A server log that cannot be written aborts the session (a harness failure,
not a result).
Per run: a server that already exited makes the run invalid without a bench; else capacity_plan.bench_argv_for
(Core's bench_argv, no request-id prefix) through run_bench with the plan's timeout (run_adv.default_run_bench: the
client's process group is killed at the timeout), run_core_b.MetricsSampler at 1 Hz (descriptive: KV peak,
preemptions, running + waiting peak), then (a) run_core_b.summarize_result ok, (e) ttfts / latencies / input_lens /
output_lens of length N and capacity_analysis.derived_problems (mixed: a long request in W), (f) no timeout (a result
file never rescues a timed-out or failed bench). derived/<tag>-requests.jsonl (idx, cls, ttft_ms, latency_ms, out; no
prompt text) comes from the --save-detailed result, its sha256 goes into session.json.

Modes (exactly one):
- --formal: the C1 session; session.json formal = true (it needs freeze.capacity = true), stage "C1".
- --mini (030 8.6 smoke, before the freeze; non-formal, stage "C1-mini"): the same state machine on a mini plan
  (build_mini_plan: U2K with C_cap 6 and a forced 5 s timeout at C 6, M2K with C_cap 2, n_near 2, n_low 1, a sentinel
  at C 2 and a dummy SLO no valid run can miss, so whatever the latencies U2K brackets 1, 2, 4, 6 (timeout), bisects
  at 5, gets C^ 5 and K = {3 (c_low), 4, 5, 6}, and the timeout drops 6 from block 2) with N = MINI_N:
  mini_num_prompts makes core_b_plan.num_prompts return MINI_N for the session, so the unchanged bench argv,
  summarize_result, timeouts and capacity_analysis replay see one N. The verdict (mini_checks: replay clean, derived
  schema, the forced timeouts hit, a point skipped, a bisection step and a c_low point, no other invalid run, clean
  teardowns) goes to smoke.json; then every latency is discarded (mask_smoke). The Phase S boot retry is not forced
  here (the fake-world tests cover it).
- --smoke-fs-order (030 8.6 smoke; D3): one boot of the Core B server WITH the 024 observation flags
  (adv_plan.server_argv_for("B0")), U2K at C 2 and C 70 with Core's N and a request-id prefix per bench, run_adv's
  OTLP receiver; fs_order_check on the span arrivals (adv_decomp.align): the first C arrivals are exactly idx 0..C-1
  and at most 3 of the last C arrivals lie outside idx N-C..N-1. Verdict in smoke.json; latencies, spans and bench
  results discarded.

Outputs: results/capacity/<session>/ only: session.json (with the sha256 of the confirm plan, of every derived file
and of rows.jsonl), plan.json, confirm_plan.json, rows.jsonl, env/, raw/ (git-ignored: server logs, bench results and
logs) with raw_manifest.json, derived/, smoke.json (smokes).

Usage (WSL, ~/llm-serving-lab, the plan from benchmarks/capacity_plan.py; from the Mac as one ssh-foreground command,
ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- bash -lc "cd ~/llm-serving-lab && <the command below>"'):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/run_capacity.py --plan capacity-plan.json \\
        --out results/capacity/<session> --mixed-dataset 2048=data/core_b/mixed-2048.jsonl \\
        --mixed-dataset 8192=data/core_b/mixed-8192.jsonl --gpu-lock-held (--formal | --mini | --smoke-fs-order)
Exit: 0 (a formal session that ran, or a passed smoke), 1 (a failed smoke), 2 (refused before anything started, or
stopped by a hard check such as a lost lock or a busy GPU; session.json then records "aborted").

Stdlib only, Python 3.10 compatible: run_core_b (tomllib) is imported lazily, opentelemetry only by the FS-order
receiver.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import adv_decomp as ad
import adv_plan as ap
import capacity_analysis as ca
import capacity_plan as cp
import core_b_plan as bp
import coreb_smoke as smoke
import run_adv as ra
from run_core_a import env_problems, provenance_problems, sha256_json, write_raw_manifest

SCHEMA_VERSION = 1
MODES = ("formal", "mini", "fs-order")
PORT = cp.PORT
IDLE_MIB, IDLE_WAIT_S = ra.IDLE_MIB, ra.IDLE_WAIT_S
SPAN_WAIT_S = ra.SPAN_WAIT_S
PLATFORM = "wsl2-cuda"
ROOT = cp.ROOT
RESULTS = ("results", "capacity")                 # 030 8.7: every Stage C file lives here
PRODUCT_RUNS = "runs"                             # 027 / 029: product-test outputs, never research data
DERIVED_KEYS = ("idx", "cls", "ttft_ms", "latency_ms", "out")
RUNNING, WAITING = "vllm:num_requests_running", "vllm:num_requests_waiting"
# mini smoke (030 8.6): a tiny replayable session; the dummy SLO is so loose that only the forced timeout fails
MINI_N = 20
MINI_WORKLOADS = ("U2K", "M2K")                   # 2K: the shortest benches; M2K exercises the mixed derived schema
# U2K's cap 6 with the forced fail at 6: bracket 1, 2, 4, 6, then bisection (F - P = 2) at 5, C^ 5, and K gets c_low =
# ceil(5 / 2) = 3 (< C^ - 1) with n_low; W = [C, N - C) keeps N - 2 C = 8 requests at C 6. M2K stays at C <= 2.
MINI_C_CAP = {"U2K": 6, "M2K": 2}
MINI_N_NEAR, MINI_N_LOW = 2, 1
MINI_SENTINEL_C = 2                               # W = [C, N - C) needs N > 2 C
MINI_FORCED_TIMEOUT_S = 5                         # below any bench's own duration: the timeout path, then the skip rule
MINI_FORCED = {"U2K": (6,)}
MINI_SLO = {"id": "MINI", "ttft_p95_ms": {"2048": 10 ** 9, "8192": 10 ** 9, "32768": 10 ** 9},
            "tpot_p95_ms": 10 ** 9, "mixed_long": "budget"}
# FS-order smoke (030 8.6, disclosure D3)
FS_WORKLOAD, FS_COND, FS_C, FS_LAST_MAX = "U2K", "B0", (2, 70), 3
FS_PREFIX = "capfs-c{C}-"
# what a smoke keeps of a row (counts and flags; 030 8.6: smoke latencies are discarded)
MASK_ROW_KEYS = ("phase", "workload", "L", "mix", "C", "N", "block", "position", "boot", "tag", "valid", "reasons",
                 "timeout", "timeout_s", "search_pass", "within_band", "outcome", "derived")
MASK_DERIVED_KEYS = ("idx", "cls", "out")


class RunError(RuntimeError):
    pass


def _rb():
    """run_core_b, imported lazily: it imports tomllib (3.11+) at module level."""
    import run_core_b

    return run_core_b


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n")


# ---------------------------------------------------------------- output location (030 8.7)

def output_problems(out_dir, *, repo_root=ROOT) -> list:
    """[] when out_dir (symlinks resolved) is a session directory below <repo>/results/capacity/ and outside any
    runs/ directory."""
    root = Path(repo_root).resolve().joinpath(*RESULTS)
    out = Path(out_dir).resolve()
    if out == root or root not in out.parents:
        return [f"{out_dir} is not a session directory under results/capacity/ (030 8.7: Stage C writes only there, "
                "never results/core_a|core_b|quality|advanced|decision_table or runs/)"]
    if PRODUCT_RUNS in out.relative_to(root).parts:
        return [f"{out_dir} is inside a runs/ directory: product-test outputs are never research data (027 §3)"]
    return []


# ---------------------------------------------------------------- server env (030 8.6)

CHECK_BASE_ENV = {"PATH": "/usr/bin"}   # capacity_report rebuilds every recorded server env from this neutral base


def server_env_problems(pins: dict, base_env: dict) -> list:
    """[] when the server env built on the caller's env records exactly what capacity_report rebuilds from
    CHECK_BASE_ENV (ra.recorded_server_env: Core's subset, every OTEL_* variable, the engine variable names).
    core_b_plan.server_env drops inherited engine variables but keeps OTEL_* ones, which the report would refuse only
    after the whole session."""
    got = ra.recorded_server_env(cp.server_argv_for(pins, base_env=base_env)[1])
    want = ra.recorded_server_env(cp.server_argv_for(pins, base_env=CHECK_BASE_ENV)[1])
    if got == want:
        return []
    diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
    otel = sorted(k for k in base_env if k.startswith("OTEL_"))
    return [f"the server env differs from the Core B one capacity_report checks (recorded {diff}; inherited OTEL "
            f"variables {otel}): a formal session inherits none (030 8.6)"]


# ---------------------------------------------------------------- plan check

def expected_plan(cfg: dict, pins: dict, *, pins_sha256: str, capacity_sha256: str, formal: bool) -> dict:
    """capacity_plan.build_plan of these capacity.toml / pins values. The derived_from session can only hold the pins
    [stage_b] blocks (build_plan refuses another) and so the C_KV recomputed from them; the report rebuilds the plan
    from the committed session itself."""
    sb = pins["stage_b"]
    session = {"session_id": cfg["capacity"]["derived_from"], "formal": True, "env_valid": True}
    uniform_plan = {"num_gpu_blocks": sb["num_gpu_blocks"], "block_size": sb["block_size"],
                    "c_kv": {str(L): cp.c_kv_for(pins, L) for L in bp.LEVELS}}
    return cp.build_plan(cfg, pins, session=session, uniform_plan=uniform_plan, pins_sha256=pins_sha256,
                         capacity_sha256=capacity_sha256, formal=formal)


def dict_diff(got, want) -> str:
    """Where two values differ: the differing keys of two dicts, one level deep (" at k, k2.x"), else both values."""
    if isinstance(got, dict) and isinstance(want, dict):
        keys = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        inner = []
        for k in keys:
            g, w = got.get(k), want.get(k)
            if isinstance(g, dict) and isinstance(w, dict):
                inner += [f"{k}.{x}" for x in sorted(y for y in set(g) | set(w) if g.get(y) != w.get(y))]
            else:
                inner.append(str(k))
        return " at " + ", ".join(inner[:8])
    return f": {got!r} != {want!r}"


def plan_problems(plan: dict, cfg: dict, pins: dict, *, pins_sha256: str, capacity_sha256: str, formal: bool) -> list:
    """Every plan value against capacity_plan.build_plan of these capacity.toml and pins bytes; [] when equal."""
    try:
        want = expected_plan(cfg, pins, pins_sha256=pins_sha256, capacity_sha256=capacity_sha256, formal=formal)
    except cp.PlanError as exc:
        return [f"the plan cannot be rebuilt: {exc}"]
    return [f"plan {k} differs from capacity_plan.build_plan of capacity.toml and pins"
            f"{dict_diff(plan.get(k), want.get(k))}" for k in sorted(set(plan) | set(want), key=str)
            if plan.get(k) != want.get(k)]


# ---------------------------------------------------------------- boot checks (030 8.6 (b) (c) (d))

def nondefault_problems(log_text: str, *, kv_bytes: int) -> tuple:
    """(watched non-default args, problems): the Core B server logs max_num_batched_tokens 2048 and
    kv_cache_memory_bytes K, and none of run_adv's observation, T or G keys."""
    got = ra.parse_nondefault(log_text)
    if got is None:
        return None, ["no non-default args line in the server log (030 8.6 (c))"]
    want = {"max_num_batched_tokens": bp.MAX_NUM_BATCHED_TOKENS, "kv_cache_memory_bytes": kv_bytes}
    problems = []
    for k in ra.WATCHED_KEYS:
        if k in want and (k not in got or type(got[k]) is not type(want[k]) or got[k] != want[k]):
            problems.append(f"non-default args {k} = {got.get(k, '<absent>')!r}; the Core B server needs "
                            f"{want[k]!r} (030 8.6 (c))")
        elif k not in want and k in got:
            problems.append(f"non-default args carry {k} = {got[k]!r}: an observation, T or G key the Core B "
                            "server must not set (030 8.1)")
    return got, problems


def server_checks(log_text: str, metrics_text: str, *, ready: bool, plan: dict, detail: str = "") -> tuple:
    """(record, problems) of one formal boot: (b) ready with FLASH_ATTN, (c) the non-default args, (d) the blocks."""
    facts = smoke.parse_server_log(log_text)
    problems = [] if ready else [f"server not ready{detail} (030 8.6 (b))"]
    if facts.get("attention_backend") != "FLASH_ATTN":
        problems.append(f"attention backend {facts.get('attention_backend')!r} != FLASH_ATTN (030 8.6 (b); 003)")
    nondefault, nd = nondefault_problems(log_text, kv_bytes=plan["kv_cache_memory_bytes"])
    cfg, blocks = ra.blocks_problems(metrics_text, expected_blocks=plan["expected_blocks"],
                                     block_size=plan["block_size"])
    return {"log_facts": facts, "nondefault": nondefault, "cache_config": cfg}, problems + nd + blocks


def fs_server_checks(log_text: str, metrics_text: str, *, ready: bool, plan: dict, pins: dict,
                     detail: str = "") -> tuple:
    """The FS-order boot: the B0 server of 024 (observation flags) passes run_adv's manipulation and block checks."""
    facts, nondefault, manip = ra.manipulation_problems(log_text, FS_COND, pins)
    cfg, blocks = ra.blocks_problems(metrics_text, expected_blocks=plan["expected_blocks"],
                                     block_size=plan["block_size"])
    problems = ([] if ready else [f"server not ready{detail} (030 8.6 (b))"]) + manip + blocks
    return {"log_facts": facts, "nondefault": nondefault, "cache_config": cfg}, problems


# ---------------------------------------------------------------- per run (030 8.6 (a) (e) (f); 8.7)

def array_problems(res: dict, *, N: int) -> list:
    """8.6 (e): ttfts, latencies, input_lens and output_lens each hold N values."""
    out = []
    for k in ("ttfts", "latencies", "input_lens", "output_lens"):
        v = res.get(k)
        if not isinstance(v, list) or len(v) != N:
            out.append(f"{k} has {len(v) if isinstance(v, list) else 'no'} values, not N = {N} (030 8.6 (e))")
    return out


def derived_requests(res: dict, *, L: int) -> list | None:
    """derived/<tag>-requests.jsonl rows of a --save-detailed result (030 8.7): idx = the bench list index (the
    dispatch order, D3), cls by input length (short L - 128, long 32640, else "other"), TTFT and latency in ms, output
    tokens; never prompt or generated text. None when the four arrays disagree in length."""
    cols = [res.get(k) for k in ("input_lens", "output_lens", "ttfts", "latencies")]
    if not all(isinstance(c, list) for c in cols) or len({len(c) for c in cols}) != 1:
        return None
    out = []
    for i, (n, o, t, la) in enumerate(zip(*cols)):
        cls = "short" if n == L - bp.N_GEN else ("long" if n == bp.LONG_INPUT else "other")
        out.append({"idx": i, "cls": cls, "ttft_ms": None if t is None else t * 1000,
                    "latency_ms": None if la is None else la * 1000, "out": o})
    return out


def inflight_peak(samples: list):
    """Descriptive (030 8.6): the largest running + waiting of the 1 Hz samples; None without one."""
    vals = [(s.get(RUNNING) or 0) + (s.get(WAITING) or 0) for s in samples
            if s.get(RUNNING) is not None or s.get(WAITING) is not None]
    return max(vals) if vals else None


def boot_label(boot: dict) -> str:
    phase, w = boot["phase"], boot["workload"]
    if phase == "sentinel":
        return f"sentinel-{boot['position']}-{w}"
    if phase == "S":
        return f"S-{w}-a{boot['attempt']}"
    if phase == "C":
        return f"C-{w}-r{boot['block']}"
    return f"fs-{w}"


def boot_tag(boot: dict) -> str:
    """raw/<boot tag>.server.log, env/<boot tag>.before|after.json: b<NNN>-<label>."""
    return f"b{boot['boot']:03d}-{boot_label(boot)}"


def compute_apps_bounded(timeout_s: float = ra.NVIDIA_SMI_TIMEOUT_S) -> list:
    """The GPU's compute processes; an nvidia-smi that fails or hangs raises (the session stops before the boot)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
                         timeout=timeout_s).stdout
    return [x.strip() for x in out.splitlines() if x.strip() and not x.strip().startswith("No running")]


# ---------------------------------------------------------------- smokes (030 8.6, before the freeze)

@contextlib.contextmanager
def mini_num_prompts(n: int = MINI_N):
    """The mini smoke's N: core_b_plan.num_prompts returns n while the block runs (the module's bytes never change),
    so bench_argv, run_core_b.summarize_result, capacity_plan's timeouts and capacity_analysis's replay all see the
    same N; the real function always comes back. run_session uses it only for --mini, never for a formal session."""
    real = bp.num_prompts
    bp.num_prompts = lambda C: n
    try:
        yield n
    finally:
        bp.num_prompts = real


def build_mini_plan(plan: dict) -> dict:
    """The mini smoke's plan from the real one (call it under mini_num_prompts): U2K and M2K with their MINI_C_CAP,
    n_near 2, n_low 1, timeouts at the mini N except the forced one, a sentinel at C 2 and the dummy SLO, so every valid
    run passes and the search (bracket and bisection), the confirm set (with a c_low), the timeout and the skip rule all
    happen without revealing any latency."""
    workloads = {}
    for w in MINI_WORKLOADS:
        x, cap = plan["workloads"][w], MINI_C_CAP[w]
        timeouts = {str(C): cp.bench_timeout_s(x["L"], x["mix"], C, cp.TAU1_S) for C in range(1, cap + 1)}
        timeouts.update({str(C): MINI_FORCED_TIMEOUT_S for C in MINI_FORCED.get(w, ())})
        workloads[w] = {**x, "c_cap": cap, "n_near": MINI_N_NEAR, "c0_covariate": False,
                        "bracket": cp.bracket_points(cap), "timeouts_s": timeouts}
    s = plan["sentinel"]
    sentinel = {**s, "C": MINI_SENTINEL_C, "N": bp.num_prompts(MINI_SENTINEL_C),
                "timeout_s": cp.bench_timeout_s(s["L"], s["mix"], MINI_SENTINEL_C, cp.TAU1_S)}
    return {**plan, "stage": "C1-mini", "slo": MINI_SLO, "secondary_slo": {}, "n_low": MINI_N_LOW,
            "workloads": workloads, "sentinel": sentinel,
            "mini": {"base_plan_sha256": sha256_json(plan), "n_prompts": bp.num_prompts(1),
                     "forced_timeouts_s": {w: {str(C): MINI_FORCED_TIMEOUT_S for C in Cs}
                                           for w, Cs in MINI_FORCED.items()},
                     "note": "030 8.6 mini smoke: non-formal, dummy SLO, latencies discarded"}}


def _forced(row: dict) -> bool:
    return row.get("phase") in ("S", "C") and row.get("C") in MINI_FORCED.get(row.get("workload"), ())


def mini_checks(plan: dict, confirm: dict, session: dict, rows: list, requests: dict) -> dict:
    """The mini smoke's verdict: replay clean, the derived schema, every forced point timed out, the skip rule removed
    a point, a Phase S bisection step (a C outside its bracket) and a c_low point in K happened, no other run invalid,
    clean teardowns."""
    replay = ca.replay_problems(plan, confirm, session["boots"], rows, requests,
                                confirm_sha256=session.get("confirm_plan_sha256"))
    derived = []
    for r in rows:
        reqs = requests.get(r.get("tag"))
        if reqs is not None and any(set(q) != set(DERIVED_KEYS) for q in reqs):
            derived.append(r["tag"])
        elif r.get("valid") and ca.derived_problems(reqs, mix=r["mix"], C=r["C"], N=r["N"]):
            derived.append(r["tag"])
    counts = collections.Counter((r["workload"], r["C"]) for r in rows if r.get("phase") == "C")
    skipped = sorted([w, p["C"]] for w, x in confirm["workloads"].items() for p in x["points"]
                     if counts[(w, p["C"])] < p["n"])
    bisected = sorted([w, h["C"]] for w, x in confirm["workloads"].items() for h in x.get("search") or []
                      if h["C"] not in plan["workloads"][w]["bracket"])
    low = sorted([w, p["C"]] for w, x in confirm["workloads"].items() for p in x["points"] if p["role"] == "low")
    checks = {"replay": replay, "derived": derived, "timeouts": sum(bool(r.get("timeout")) for r in rows),
              "forced_missed": [r["tag"] for r in rows if _forced(r) and not r.get("timeout")],
              "skipped": skipped, "bisected": bisected, "low_points": low,
              "unexpected_invalid": [r["tag"] for r in rows if not r.get("valid") and not _forced(r)],
              "teardown_problems": session.get("teardown_problems") or []}
    ok = (not replay and not derived and checks["timeouts"] >= 1 and not checks["forced_missed"] and bool(skipped)
          and bool(bisected) and bool(low) and not checks["unexpected_invalid"] and not checks["teardown_problems"])
    return {"ok": ok, "checks": checks}


def fs_order_check(arrivals: dict, *, C: int, N: int, last_max: int = FS_LAST_MAX) -> dict:
    """030 8.6 FS-order on one bench's server arrivals {idx: time}: the first C arrivals are exactly idx 0..C-1 and at
    most last_max of the last C arrivals lie outside idx N-C..N-1 (ties keep idx order)."""
    missing = sum(1 for i in range(N) if arrivals.get(i) is None)
    if missing:
        return {"C": C, "N": N, "missing": missing, "first_c_equal": None, "last_c_diff": None, "ok": False}
    order = sorted(range(N), key=lambda i: (arrivals[i], i))
    first = set(order[:C]) == set(range(C))
    last = len(set(order[N - C:]) - set(range(N - C, N)))
    return {"C": C, "N": N, "missing": 0, "first_c_equal": first, "last_c_diff": last, "ok": first and last <= last_max}


def mask_smoke(out) -> list:
    """Discard a smoke's latencies (030 8.6): bench results and logs, spans and OTLP bodies go; derived rows keep
    idx / cls / out and rows.jsonl keeps counts and flags. Server logs stay in raw/ (git-ignored). Returns what went."""
    out, gone = Path(out), []
    raw = out / "raw"
    for pattern in ("*.bench.json", "*.bench.log", "*.spans.jsonl"):
        for p in sorted(raw.glob(pattern)):
            p.unlink()
            gone.append(p.relative_to(out).as_posix())
    for d in sorted(raw.glob("*.otlp")):
        shutil.rmtree(d)
        gone.append(d.relative_to(out).as_posix())
    if (out / "derived").is_dir():
        for p in sorted((out / "derived").glob("*-requests.jsonl")):
            keep = [{k: q.get(k) for k in MASK_DERIVED_KEYS} for q in map(json.loads, p.read_text().splitlines())]
            p.write_text("".join(json.dumps(q) + "\n" for q in keep))
    rows_path = out / "rows.jsonl"
    if rows_path.is_file():
        rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
        rows_path.write_text("".join(json.dumps({k: r[k] for k in MASK_ROW_KEYS if k in r}) + "\n" for r in rows))
    return gone


def _count_ids(receiver, prefix: str) -> int:
    pat = re.compile("^cmpl-" + re.escape(prefix) + r"\d+-0$")
    return len({str(r.get("id")) for r in ad.llm_request_records(receiver.read_spans())
                if pat.match(str(r.get("id") or ""))})


def wait_for_prefix_spans(receiver, prefix: str, n: int, *, now: Callable, sleep: Callable,
                          limit_s: float = SPAN_WAIT_S, poll_s: float = 1.0) -> int:
    """The bench's main-result spans (cmpl-<prefix><i>-0) received after polling until n or limit_s."""
    t = now()
    got = _count_ids(receiver, prefix)
    while got < n and now() - t < limit_s:
        sleep(poll_s)
        got = _count_ids(receiver, prefix)
    return got


# ---------------------------------------------------------------- one session's steps

class _Harness:
    """The per-boot and per-run steps of 030 8.6 for one session (state machine and FS-order smoke alike)."""

    def __init__(self, *, plan, pins, out, session, formal, pre, rows_fh, popen, signal_group, group_alive, http_get,
                 run_bench, gpu_mem_used, compute_apps, lock_held_check, capture_env, base_env, vllm_bin,
                 mixed_datasets, sample_interval, now, sleep, clock, idle_wait_s):
        self.plan, self.pins, self.out, self.session, self.formal, self.pre = plan, pins, out, session, formal, pre
        self.raw, self.derived, self.rows_fh = out / "raw", out / "derived", rows_fh
        self.popen, self.signal_group, self.group_alive, self.http_get = popen, signal_group, group_alive, http_get
        self.run_bench, self.gpu_mem_used, self.compute_apps = run_bench, gpu_mem_used, compute_apps
        self.lock_held_check, self.capture_env, self.base_env, self.vllm_bin = lock_held_check, capture_env, base_env, \
            vllm_bin
        self.mixed_datasets, self.sample_interval = mixed_datasets or {}, sample_interval
        self.now, self.sleep, self.clock, self.idle_wait_s = now, sleep, clock, idle_wait_s
        self.base = f"http://127.0.0.1:{PORT}"
        self.rows, self.requests, self.env_flags = [], {}, []

    # ---------------------------------------------------------- bookkeeping
    def emit(self, row: dict) -> dict:
        self.rows.append(row)
        self.rows_fh.write(json.dumps(row) + "\n")
        self.rows_fh.flush()
        return row

    def new_boot(self, phase: str, w: str, *, block=None, attempt=None, position=None) -> dict:
        boot = {"boot": len(self.session["boots"]), "phase": phase, "workload": w, "block": block, "attempt": attempt,
                "position": position, "reached_first_bench": None, "problems": []}
        boot["tag"] = boot_tag(boot)
        self.session["boots"].append(boot)
        return boot

    def snap(self, name: str) -> None:
        if self.capture_env:
            rec = self.capture_env(out_path=self.out / "env" / name, gpu_lock_held=True)
            self.env_flags.extend({"env": name, "reason": why} for why in env_problems(rec, PLATFORM))
            if self.formal:
                self.env_flags.extend({"env": name, "reason": why} for why in provenance_problems(rec, self.pre))

    # ---------------------------------------------------------- boot
    def pre_boot(self, tag: str) -> None:
        """Hard checks before a boot (030 8.6): the lock still held by this process, memory.used <= 1000 MiB within
        60 s, no compute process on the GPU; any failure stops the session."""
        if not self.lock_held_check():
            raise RunError(f"--gpu-lock-held, but this process does not hold {smoke.GPU_LOCK} any more; stopping "
                           f"before {tag} (MACHINE_POLICY)")
        t, mem = self.now(), self.gpu_mem_used()
        while mem > IDLE_MIB and self.now() - t < self.idle_wait_s:
            self.sleep(2)
            mem = self.gpu_mem_used()
        if mem > IDLE_MIB:
            raise RunError(f"GPU busy before {tag}: memory.used {mem} MiB > {IDLE_MIB} after {self.idle_wait_s} s "
                           "(030 8.6; MACHINE_POLICY)")
        apps = self.compute_apps()
        if apps:
            raise RunError(f"other compute process(es) on the GPU before {tag}: {apps} (030 8.6; MACHINE_POLICY)")

    def wait_ready(self, proc, record: dict) -> tuple:
        t0, ready, limit = self.now(), False, self.plan["server_ready_timeout_s"]
        while self.now() - t0 < limit and proc.poll() is None:
            if self.http_get(f"{self.base}/health")[0] == 200:
                ready = True
                break
            self.sleep(2)
        record["startup_s"] = round(self.now() - t0, 1)
        return ready, f" within {limit} s (exit code {proc.poll()!r})"

    def metrics_text(self, ready: bool) -> str:
        if not ready:
            return ""
        code, text = self.http_get(f"{self.base}/metrics")
        return text if code == 200 else ""

    def check_log(self, proc, when: str) -> None:
        """A server log that could not be written is a harness failure: stop the session, write no row."""
        error = getattr(getattr(proc, "log_reader", None), "error", None)
        if error:
            raise RunError(f"the server log could not be written ({error}) {when}: a harness failure, not a result; "
                           "session aborted")

    def teardown(self, tag: str, proc, record: dict, receiver=None) -> None:
        """Server group down (SIGKILL escalation included), log reader joined, receiver closed; interrupts arriving
        meanwhile wait until all of that is done (run_adv.DeferredSignals)."""
        with ra.DeferredSignals() as deferred:
            problems = []
            try:
                if proc is not None:
                    problems += ra.shutdown_group(proc, deferred, signal_group=self.signal_group,
                                                  group_alive=self.group_alive, now=self.now, sleep=self.sleep)
                    reader = getattr(proc, "log_reader", None)
                    if reader is not None and not reader.join(ra.LOG_JOIN_S):
                        problems.append("server log reader still open after teardown")
            finally:
                try:
                    if receiver is not None:
                        problems += receiver.close() or []
                finally:
                    if problems:
                        record["teardown_problems"] = problems
                        self.session.setdefault("teardown_problems", []).append({"boot": tag, "reasons": problems})

    def boot(self, boot: dict, next_point: Callable, planned: list | None = None) -> list | None:
        """One server boot running next_point(rows so far) -> C | None; the rows, or None for a Phase S boot that
        failed before its first bench (it has no row). A failed sentinel / Phase C boot records `planned` as invalid."""
        tag = boot["tag"]
        self.pre_boot(tag)
        self.snap(f"{tag}.before.json")
        argv, env = cp.server_argv_for(self.pins, port=PORT, vllm_bin=self.vllm_bin, base_env=self.base_env)
        record = self.session["runs"][tag] = {"argv": argv, "env": ra.recorded_server_env(env)}
        proc, rows = None, []
        try:
            proc = self.popen(argv, env, self.raw / f"{tag}.server.log")
            ready, detail = self.wait_ready(proc, record)
            metrics = self.metrics_text(ready)
            facts, problems = server_checks(_read(self.raw / f"{tag}.server.log"), metrics, ready=ready,
                                            plan=self.plan, detail=detail)
            record.update(facts)
            self.check_log(proc, "during the server start")
            if not problems and proc.poll() is not None:     # gone between its checks and its first bench (8.4 step 4)
                problems = [f"server exited (code {proc.poll()!r}) before its first bench (030 8.4)"]
            boot["problems"], boot["reached_first_bench"] = problems, not problems
            if problems:
                for C in planned or []:
                    rows.append(self.not_run(boot, C, problems))
            else:
                while (C := next_point(rows)) is not None:
                    rows.append(self.run_point(boot, proc, C))
        finally:
            self.teardown(tag, proc, record)
        self.check_log(proc, "at teardown")
        self.snap(f"{tag}.after.json")
        if boot["phase"] == "S" and not boot["reached_first_bench"]:
            return None
        return rows

    # ---------------------------------------------------------- run
    def _spec(self, boot: dict) -> tuple:
        w = boot["workload"]
        wl = self.plan["workloads"][w]
        return w, wl["L"], wl["mix"]

    def _row(self, boot: dict, C: int) -> dict:
        w, L, mix = self._spec(boot)
        phase = boot["phase"]
        timeout_s = self.plan["sentinel"]["timeout_s"] if phase == "sentinel" else \
            self.plan["workloads"][w]["timeouts_s"][str(C)]
        return {"phase": phase, "workload": w, "L": L, "mix": mix, "C": C, "N": bp.num_prompts(C),
                "block": boot["block"], "position": boot["position"], "boot": boot["boot"],
                "tag": cp.run_tag(phase, w, C, block=boot["block"], position=boot["position"]), "valid": False,
                "reasons": [], "timeout": False, "timeout_s": timeout_s, "output_throughput": None, "metrics": None}

    def decide(self, row: dict, reqs) -> dict:
        """The navigation bits with the replay's own functions: search_pass / within_band (Phase S), outcome (C)."""
        if row["phase"] == "S":
            bits = ca.search_bits(row, reqs, self.plan)
            row["search_pass"], row["within_band"] = bits["pass"], bits["within_band"]
        elif row["phase"] == "C":
            row["outcome"] = ca.phase_c_outcome(row, reqs, self.plan)
        return self.emit(row)

    def not_run(self, boot: dict, C: int, reasons: list) -> dict:
        row = self._row(boot, C)
        row["reasons"] = list(reasons)
        return self.decide(row, None)

    def bench(self, row: dict, argv: list) -> tuple:
        """Run one bench with its timeout and the 1 Hz sampler; (result or None, reasons) with (a) (e) (f)."""
        tag, L, mix, C, N = row["tag"], row["L"], row["mix"], row["C"], row["N"]
        row["argv"] = argv
        sampler = _rb().MetricsSampler(self.http_get, f"{self.base}/metrics", self.sample_interval)
        sampler.start()
        try:
            rc, timed_out = self.run_bench(argv, self.raw / f"{tag}.bench.log", {**self.base_env, **bp.bench_env()},
                                           row["timeout_s"])
        finally:
            sampled = sampler.stop()
        row["metrics"] = {**sampled, "inflight_peak": inflight_peak(sampler.samples)}
        row["healthy_after_bench"] = self.http_get(f"{self.base}/health")[0] == 200
        path, res, reasons = self.raw / f"{tag}.bench.json", None, []
        if rc == 0 and not timed_out and path.is_file():
            try:
                res = json.loads(path.read_text())
            except ValueError as exc:
                reasons.append(f"unreadable bench result ({exc})")
        if timed_out:
            row["timeout"] = True
            reasons.append(f"timeout: the bench exceeded {row['timeout_s']} s; its process group was killed "
                           "(030 8.4; 8.6 (f))")
        elif res is None and not reasons:
            reasons.append(f"vllm bench serve exit {rc} (no result)")
        if res is not None:
            summary = _rb().summarize_result(res, L=L, C=C, mix=mix)
            reasons += summary.pop("reasons")
            row.update(summary=summary, output_throughput=summary.get("output_throughput"),
                       result=f"raw/{tag}.bench.json")
            reasons += array_problems(res, N=N)
        return res, reasons

    def run_point(self, boot: dict, proc, C: int) -> dict:
        row = self._row(boot, C)
        if proc.poll() is not None:
            row["reasons"] = [f"server exited (code {proc.poll()!r}) before this bench; not run (030 8.6 (b))"]
            return self.decide(row, None)
        w, L, mix = row["workload"], row["L"], row["mix"]
        argv = cp.bench_argv_for(self.pins, workload=w, C=C, result_dir=str(self.raw),
                                 result_filename=f"{row['tag']}.bench.json", port=PORT,
                                 mixed_dataset=self.mixed_datasets.get(L) if mix == "mixed" else None,
                                 vllm_bin=self.vllm_bin)
        res, reasons = self.bench(row, argv)
        self.check_log(proc, "during the bench")
        reqs = derived_requests(res, L=L) if res is not None else None
        if reqs is not None:
            path = self.derived / f"{row['tag']}-requests.jsonl"
            path.write_text("".join(json.dumps(q) + "\n" for q in reqs))
            row["derived"] = path.relative_to(self.out).as_posix()
            self.session["derived_sha256"][row["tag"]] = _sha256_file(path)
            self.requests[row["tag"]] = reqs
            reasons += [p for p in ca.derived_problems(reqs, mix=mix, C=C, N=row["N"]) if p not in reasons]
        row["reasons"], row["valid"] = reasons, not reasons
        return self.decide(row, reqs)


# ---------------------------------------------------------------- the state machine (030 8.4)

def _fixed(points: list) -> Callable:
    return lambda rows: points[len(rows)] if len(rows) < len(points) else None


def _state_machine(h: _Harness, write_session: Callable) -> dict:
    plan = h.plan
    s = plan["sentinel"]
    start = h.boot(h.new_boot("sentinel", s["workload"], position="start"), _fixed([s["C"]]), [s["C"]])
    if not start[0]["valid"]:
        raise RunError(f"the start sentinel is invalid ({'; '.join(start[0]['reasons'])}): the 8.6 sentinel gate needs "
                       "both sentinels valid, so the session stops here (030 8.6)")
    searches, retry = {}, []

    def search(w: str, attempt: int):
        c_cap = plan["workloads"][w]["c_cap"]
        rows = h.boot(h.new_boot("S", w, attempt=attempt),
                      lambda done: cp.search_next(c_cap, [(r["C"], r["search_pass"]) for r in done]))
        return None if rows is None else [{"C": r["C"], "pass": r["search_pass"], "within_band": r["within_band"]}
                                          for r in rows]

    for w in plan["workloads"]:
        hist = search(w, 1)
        if hist is None:
            retry.append(w)
        else:
            searches[w] = hist
    for w in cp.phase_s_retries(plan["workloads"], set(retry)):
        searches[w] = search(w, 2)
    confirm = cp.build_confirm_plan(plan, searches)
    _write_json(h.out / "confirm_plan.json", confirm)
    h.session.update(confirm_plan_sha256=cp.confirm_plan_sha256(confirm), confirm_plan_recorded_at=h.clock())
    write_session()                    # 030 8.4: the confirm plan's sha256 is on disk before Phase C
    outcomes = {w: [] for w in plan["workloads"]}
    for r in range(1, confirm["blocks"] + 1):
        new = []
        for w, points in cp.phase_c_boots(confirm, r, outcomes):
            for row in h.boot(h.new_boot("C", w, block=r), _fixed(points), points):
                new.append((w, (r, row["C"], row["outcome"])))
        for w, o in new:
            outcomes[w].append(o)
    h.boot(h.new_boot("sentinel", s["workload"], position="end"), _fixed([s["C"]]), [s["C"]])
    return confirm


def _fs_order(h: _Harness, receiver_factory: Callable, span_wait_s: float) -> dict:
    """030 8.6 FS-order: one B0 boot (024 observation flags), U2K at C 2 and C 70 with Core's N."""
    pins, plan, sb = h.pins, h.plan, h.pins["stage_b"]
    boot = h.new_boot("fs-order", FS_WORKLOAD)
    tag = boot["tag"]
    h.pre_boot(tag)
    h.snap(f"{tag}.before.json")
    argv, env = ap.server_argv_for(FS_COND, pins, gdn_prefill_backend=sb["gdn_prefill_backend"],
                                   gpu_memory_utilization=sb["gpu_memory_utilization"], port=PORT,
                                   vllm_bin=h.vllm_bin, base_env=h.base_env)
    record = h.session["runs"][tag] = {"argv": argv, "env": ra.recorded_server_env(env)}
    receiver = receiver_factory(h.raw / f"{tag}.otlp", h.raw / f"{tag}.spans.jsonl", "capfs-")
    receiver.start()
    proc, benches, problems = None, [], []
    try:
        proc = h.popen(argv, env, h.raw / f"{tag}.server.log")
        ready, detail = h.wait_ready(proc, record)
        facts, problems = fs_server_checks(_read(h.raw / f"{tag}.server.log"), h.metrics_text(ready), ready=ready,
                                           plan=plan, pins=pins, detail=detail)
        record.update(facts)
        h.check_log(proc, "during the server start")
        boot["problems"], boot["reached_first_bench"] = problems, not problems
        if not problems:
            for C in FS_C:
                benches.append(_fs_bench(h, boot, proc, receiver, C, span_wait_s))
    finally:
        h.teardown(tag, proc, record, receiver)
    h.check_log(proc, "at teardown")
    receiver.raise_if_failed()
    h.snap(f"{tag}.after.json")
    return {"ok": not problems and len(benches) == len(FS_C) and all(b["ok"] for b in benches),
            "boot_problems": problems, "benches": benches,
            "rule": "first C span orders = idx 0..C-1 exactly; at most 3 of the last C outside idx N-C..N-1 (030 8.6)"}


def _fs_bench(h: _Harness, boot: dict, proc, receiver, C: int, span_wait_s: float) -> dict:
    N, prefix = bp.num_prompts(C), FS_PREFIX.format(C=C)
    wl = h.plan["workloads"][FS_WORKLOAD]
    row = {"phase": "fs-order", "workload": FS_WORKLOAD, "L": wl["L"], "mix": wl["mix"], "C": C, "N": N,
           "block": None, "position": None, "boot": boot["boot"], "tag": f"fs-{FS_WORKLOAD}-c{C}", "valid": False,
           "reasons": [], "timeout": False, "timeout_s": wl["timeouts_s"][str(C)]}
    argv = cp.bench_argv_for(h.pins, workload=FS_WORKLOAD, C=C, result_dir=str(h.raw),
                             result_filename=f"{row['tag']}.bench.json", port=PORT, vllm_bin=h.vllm_bin)
    res, reasons = h.bench(row, argv + ["--request-id-prefix", prefix])
    h.check_log(proc, "during the bench")
    spans = wait_for_prefix_spans(receiver, prefix, N, now=h.now, sleep=h.sleep, limit_s=span_wait_s) \
        if res is not None else 0
    receiver.raise_if_failed()
    order = {}
    if res is not None:
        aligned = ad.align(ad.llm_request_records(receiver.read_spans()), res, prefix=prefix, L=wl["L"])
        order = {r["idx"]: r["arrival"] for r in aligned["requests"] if r["arrival"] is not None}
    check = fs_order_check(order, C=C, N=N)
    row["reasons"], row["valid"] = reasons, not reasons
    h.emit(row)
    return {**check, "bench_ok": not reasons, "spans": spans, "reasons": reasons, "ok": check["ok"] and not reasons}


# ---------------------------------------------------------------- session

def run_session(plan: dict, *, cfg: dict, pins: dict, pins_sha256: str, capacity_sha256: str, out_dir, session_id: str,
                mode: str = "formal", popen: Callable, signal_group: Callable, group_alive: Callable,
                http_get: Callable, run_bench: Callable, gpu_mem_used: Callable, compute_apps: Callable,
                lock_held_check: Callable | None = None, gpu_lock_held: bool = False,
                vllm_bin: str = ".venv/bin/vllm", mixed_datasets: dict | None = None,
                sample_interval: float | None = 1.0, toolchain_check: Callable = bp.toolchain_problems,
                capture_env: Callable | None = None, base_env=None, receiver_factory: Callable | None = None,
                now: Callable = time.monotonic, sleep: Callable = time.sleep,
                clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(), repo_root=ROOT,
                idle_wait_s: float = IDLE_WAIT_S, span_wait_s: float = SPAN_WAIT_S) -> dict:
    """One Stage C session in `mode` ("formal", "mini" or "fs-order"); the session record (also session.json)."""
    if mode not in MODES:
        raise RunError(f"unknown mode {mode!r}; run_capacity runs {MODES}")
    base_env = dict(os.environ if base_env is None else base_env)
    lock_held_check = lock_held_check or ra.held_gpu_lock
    _refusals(plan, cfg=cfg, pins=pins, pins_sha256=pins_sha256, capacity_sha256=capacity_sha256, out_dir=out_dir,
              mode=mode, lock_held_check=lock_held_check, gpu_lock_held=gpu_lock_held, base_env=base_env,
              repo_root=repo_root)                       # with Core's N: the plan given is the real one
    with mini_num_prompts() if mode == "mini" else contextlib.nullcontext():
        return _run_session(plan, cfg=cfg, pins=pins, pins_sha256=pins_sha256, capacity_sha256=capacity_sha256,
                            out_dir=out_dir, session_id=session_id, mode=mode, popen=popen, signal_group=signal_group,
                            group_alive=group_alive, http_get=http_get, run_bench=run_bench,
                            gpu_mem_used=gpu_mem_used, compute_apps=compute_apps, lock_held_check=lock_held_check,
                            gpu_lock_held=gpu_lock_held, vllm_bin=vllm_bin, mixed_datasets=mixed_datasets,
                            sample_interval=sample_interval, toolchain_check=toolchain_check, capture_env=capture_env,
                            base_env=base_env, receiver_factory=receiver_factory, now=now, sleep=sleep, clock=clock,
                            idle_wait_s=idle_wait_s, span_wait_s=span_wait_s)


def _refusals(plan, *, cfg, pins, pins_sha256, capacity_sha256, out_dir, mode, lock_held_check, gpu_lock_held,
              base_env, repo_root) -> None:
    problems = output_problems(out_dir, repo_root=repo_root)
    if problems:
        raise RunError("; ".join(problems))
    if not gpu_lock_held:
        raise RunError(f"Stage C boots (formal and smoke alike) must hold the GPU lock: flock -n {smoke.GPU_LOCK} ... "
                       "--gpu-lock-held (MACHINE_POLICY; 030 8.4 / 8.6)")
    if not lock_held_check():
        raise RunError(f"--gpu-lock-held, but this process does not hold {smoke.GPU_LOCK} (no locked descriptor on "
                       f"it): run under flock -n {smoke.GPU_LOCK} ... (MACHINE_POLICY)")
    formal = mode == "formal"
    if formal and ((cfg or {}).get("freeze") or {}).get("capacity") is not True:
        raise RunError("formal Stage C sessions require freeze.capacity = true in configs/capacity.toml (030 凍結)")
    problems = [f"capacity.toml: {p}" for p in cp.capacity_problems(cfg, pins)]
    if not problems:
        problems = plan_problems(plan, cfg, pins, pins_sha256=pins_sha256, capacity_sha256=capacity_sha256,
                                 formal=formal)
    if formal:
        sb = pins.get("stage_b") or {}
        problems += _rb().stage_b_problems(plan, pins, gdn_prefill_backend=sb.get("gdn_prefill_backend"),
                                           gpu_memory_utilization=sb.get("gpu_memory_utilization"))
        inherited = bp.inherited_unlisted(base_env)
        if inherited:
            problems.append(f"inherited engine variables {inherited}: a formal session inherits none (030 8.6)")
        problems += server_env_problems(pins, base_env)
        problems += [f"freeze evidence: {p}" for p in cp.freeze_evidence_problems(cfg, repo_root)]
    if problems:
        raise RunError("; ".join(problems))


def _run_session(plan, *, cfg, pins, pins_sha256, capacity_sha256, out_dir, session_id, mode, popen, signal_group,
                 group_alive, http_get, run_bench, gpu_mem_used, compute_apps, lock_held_check, gpu_lock_held,
                 vllm_bin, mixed_datasets, sample_interval, toolchain_check, capture_env, base_env, receiver_factory,
                 now, sleep, clock, idle_wait_s, span_wait_s) -> dict:
    formal = mode == "formal"
    session_plan = build_mini_plan(plan) if mode == "mini" else plan
    mixed_sha = {}
    if mode != "fs-order":
        for L in sorted({x["L"] for x in session_plan["workloads"].values() if x["mix"] == "mixed"}):
            path = (mixed_datasets or {}).get(L)
            if not path or not Path(path).is_file():
                raise RunError(f"mixed workloads at L={L} need --mixed-dataset {L}=<file> (decision 007)")
            mixed_sha[str(L)] = _sha256_file(path)
            pinned = plan["mix_dataset_sha256"].get(str(L))
            if formal and mixed_sha[str(L)] != pinned:
                raise RunError(f"mixed dataset for L={L}: sha256 {mixed_sha[str(L)]} != pinned {pinned!r} "
                               "(capacity.toml mix_dataset_sha256 = pins core_b.mix_dataset_sha256)")
    sb = pins["stage_b"]
    if mode == "fs-order":
        server_env = ap.server_env_for(base_env, pins)
        if receiver_factory is None:
            try:
                receiver_factory = ra.receiver_factory_for(pins)
            except ImportError as exc:
                raise RunError(f"OTLP decoder unavailable ({exc}); run with the vLLM venv, which has opentelemetry "
                               "(024)") from exc
    else:
        server_env = cp.server_argv_for(pins, port=PORT, vllm_bin=vllm_bin, base_env=base_env)[1]
    missing = toolchain_check(server_env)
    if missing:
        raise RunError("toolchain incomplete, not starting (014): " + "; ".join(missing))
    out = Path(out_dir)
    if out.exists():
        raise RunError(f"{out} exists; sessions are never resumed or overwritten (030 8.4: rerun under a new id)")
    (out / "raw").mkdir(parents=True)
    (out / "env").mkdir()
    if mode != "fs-order":
        (out / "derived").mkdir()
    _write_json(out / "plan.json", session_plan)
    pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=True) if capture_env else {}
    repo = pre.get("repo") or {}
    vllm = ((pre.get("runtime") or {}).get("vllm") or {}).get("vllm")
    if formal:
        if vllm != pins["vllm"]["version"]:
            raise RunError(f"vLLM {vllm!r} in the preflight != pinned {pins['vllm']['version']} (003; 030 8.6)")
        bad = env_problems(pre, PLATFORM) + provenance_problems(pre, pre)
        if repo.get("source") != "DEPLOYED_COMMIT":
            bad.append(f"repo source {repo.get('source')!r}: the code is not a deployed bundle (030 8.4: one "
                       "DEPLOYED_COMMIT)")
        if bad:
            raise RunError("preflight: " + "; ".join(bad))
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id,
               "stage": session_plan["stage"] if mode != "fs-order" else "C1-fs-order", "mode": mode, "formal": formal,
               "gpu_lock_held": gpu_lock_held, "plan_sha256": sha256_json(session_plan), "pins_sha256": pins_sha256,
               "capacity_sha256": capacity_sha256, "plan_pins_sha256": plan.get("pins_sha256"),
               "plan_capacity_sha256": plan.get("capacity_sha256"), "derived_from": plan.get("derived_from"),
               "c0_branch": plan.get("c0_branch"), "started_at": clock(),
               "preflight": "env/preflight.json" if capture_env else None, "repo_commit": repo.get("commit"),
               "repo_source": repo.get("source"), "vllm": vllm,
               "server": {"gdn_prefill_backend": sb["gdn_prefill_backend"],
                          "gpu_memory_utilization": sb["gpu_memory_utilization"], "port": PORT, "vllm_bin": vllm_bin},
               "result_dir": str(out / "raw"),
               "mixed_datasets": {str(L): str(p) for L, p in (mixed_datasets or {}).items()},
               "mixed_dataset_sha256": mixed_sha, "server_env": ra.recorded_server_env(server_env),
               "inherited_unlisted": bp.inherited_unlisted(base_env), "boots": [], "runs": {}, "derived_sha256": {}}
    fh = open(out / "rows.jsonl", "w")
    h = _Harness(plan=session_plan, pins=pins, out=out, session=session, formal=formal, pre=pre, rows_fh=fh,
                 popen=popen, signal_group=signal_group, group_alive=group_alive, http_get=http_get,
                 run_bench=run_bench, gpu_mem_used=gpu_mem_used, compute_apps=compute_apps,
                 lock_held_check=lock_held_check, capture_env=capture_env, base_env=base_env, vllm_bin=vllm_bin,
                 mixed_datasets=mixed_datasets, sample_interval=sample_interval, now=now, sleep=sleep, clock=clock,
                 idle_wait_s=idle_wait_s)

    def write_session() -> None:
        _write_json(out / "session.json", session)

    try:
        try:
            if mode == "fs-order":
                session["smoke"] = _fs_order(h, receiver_factory, span_wait_s)
            else:
                confirm = _state_machine(h, write_session)
                if mode == "mini":
                    session["smoke"] = mini_checks(session_plan, confirm, session, h.rows, h.requests)
        finally:
            fh.close()
    except BaseException as exc:
        session["aborted"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        session.update(finished_at=clock(), outcomes=_outcomes(h.rows, session), env_flags=h.env_flags,
                       env_valid=not h.env_flags)
        if mode != "formal":
            if "smoke" not in session:
                session["smoke"] = {"ok": False, "aborted": session.get("aborted")}
            _write_json(out / "smoke.json", session["smoke"])
            mask_smoke(out)
            for rec in session["runs"].values():
                rec.pop("startup_s", None)
        session["rows_sha256"] = _sha256_file(out / "rows.jsonl")     # the rows as kept (a smoke's after masking)
        write_raw_manifest(out)
        write_session()
    return session


def _outcomes(rows: list, session: dict) -> dict:
    return {"boots": len(session["boots"]), "runs": len(rows), "valid": sum(bool(r.get("valid")) for r in rows),
            "invalid": sum(not r.get("valid") for r in rows), "timeout": sum(bool(r.get("timeout")) for r in rows)}


def main(argv: list | None = None) -> int:
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, help="capacity_plan.py output")
    parser.add_argument("--out", required=True, help="results/capacity/<session>")
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--mixed-dataset", action="append", default=[], help="L=path (decision 007)")
    parser.add_argument("--gpu-lock-held", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--formal", dest="mode", action="store_const", const="formal")
    modes.add_argument("--mini", dest="mode", action="store_const", const="mini")
    modes.add_argument("--smoke-fs-order", dest="mode", action="store_const", const="fs-order")
    args = parser.parse_args(argv)
    cap_raw, pins_raw = cp.load_capacity_bytes(), cp.load_pins_bytes()
    rb = _rb()
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, smoke._exit_on_signal)
    try:
        summary = run_session(
            json.loads(Path(args.plan).read_text()), cfg=tomllib.loads(cap_raw.decode()),
            pins=tomllib.loads(pins_raw.decode()), pins_sha256=hashlib.sha256(pins_raw).hexdigest(),
            capacity_sha256=hashlib.sha256(cap_raw).hexdigest(), out_dir=args.out, session_id=Path(args.out).name,
            mode=args.mode, popen=ra.start_server, signal_group=smoke.signal_group, group_alive=smoke.group_alive,
            http_get=rb.default_http_get, run_bench=ra.default_run_bench, gpu_mem_used=ra.nvidia_mem_used_bounded,
            compute_apps=compute_apps_bounded, lock_held_check=ra.held_gpu_lock, gpu_lock_held=args.gpu_lock_held,
            vllm_bin=args.vllm_bin, capture_env=rb.make_vllm_env_capture(ROOT / "scripts" / "env_capture.py"),
            mixed_datasets={int(k): v for k, v in (x.split("=", 1) for x in args.mixed_dataset)},
            sample_interval=1.0)
    except (RunError, ra.RunError) as exc:      # ra.RunError: run_adv's receiver could not store spans (FS-order)
        print(f"run_capacity: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": summary.get("mode"), "outcomes": summary.get("outcomes"),
                      "smoke": summary.get("smoke")}))
    if args.mode == "formal":
        return 0
    return 0 if (summary.get("smoke") or {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
