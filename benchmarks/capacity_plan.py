#!/usr/bin/env python3
"""Stage C (C1) plan rules (decision 030; PREREGISTRATION §8): the configs/capacity.toml checks, the deterministic
Phase S search, the confirmation set and the Phase C order, fixed before any C1 data. Core modules are imported,
never changed.

- capacity.toml (030 implementation note: never pins.toml) must hold exactly the 030 values (capacity_problems):
  SLO = S2 (TTFT P95 2000 / 6000 / 20000 ms at 2048 / 8192 / 32768, TPOT P95 100 ms, mixed_long "budget"), the
  secondary S1 / S3, steady window, m = 0.05, n_near 5 (7 under C0 branch B / C for n_near_bc_workloads, which is at
  least U8K and exactly U8K under C), n_low 3, C_cap = min(128, 2 C_KV), the tau1 table, the sentinel and the pinned
  mixed datasets. freeze.capacity = true needs c0_branch in A / B / C (028: D or "" keeps §8 unfrozen) and pins
  freeze.mac / nvidia / advanced. Under C0 branch B / C the uniform n_near_bc workloads also carry c0_covariate:
  capacity_analysis records C0's per-run slip count (028) for their runs, descriptively. 028 defines the slip on a
  uniform lattice (each position's own baseline over idx = j mod C), so a mixed workload C0's log names gets
  n_near_bc (030 8.1: n never drops) but no covariate.
- C_KV(L) = core_b_plan.c_kv on the derived_from session's pool (248 x 528: 35 / 13), checked against that session's
  plan and pins [stage_b]; C_cap(L) = min(128, 2 C_KV(L)) = 70 / 26; a mixed workload uses its L's values.
- Phase S (one boot per workload, canonical order U2K, U8K, M2K, M8K; a boot that fails before its first bench is
  retried once at the end of Phase S, phase_s_retries): bracket 1, 2, 4, ... (< C_cap) then C_cap until the first
  fail F (P = the last pass), then bisection mid = (P + F) // 2 while F - P > 1; C^ = P (0 when C 1 fails); a
  workload whose boot fails twice gets no points (its estimands are NT). search_next is the pure step
  function and refuses a history the rule would not have run (replay). A run's pass / band bits come from
  capacity_analysis (valid, maxR <= 1 and, for mixed, the run's own R_long <= 1; band: maxR <= 1 - m, 8.4 step 3,
  with maxR = max(R_short, R_TPOT) as in 8.1).
- C^_g = the largest tested c whose every tested c' <= c is inside the band (0 if none).
- K: C^ = 0 -> {1} (n_near); else {C^ - 1, C^, C^ + 1} (n_near) within [1, C_cap] plus c_low (n_low) = C^_g if
  1 <= C^_g < C^ - 1, else ceil(C^ / 2) unless that is >= C^ - 1. build_confirm_plan is a pure function of the
  Phase S histories; its sha256 (confirm_plan_sha256) is recorded before Phase C.
- Phase C: blocks r = 1 .. max n; block r boots the workloads in random.Random(int(sha256("<seed>:capacity:block:r"),
  16)) order, and a boot runs its active points in the order of sorted(K) shuffled by
  random.Random(int(sha256("<seed>:capacity:<w>:<r>"), 16)) (the full order is fixed in the confirm plan; skips only
  filter it). A point is active in block r if r <= its n and no earlier block has, at it or below it, a valid FAIL
  (a valid run with maxR_run > 1: 8.4 names it beside "an invalid run") or an invalid run (013: never replaced). A
  mixed point's pooled long class is not a run's outcome and never drops a point: it is judged once over the point's
  runs (8.1). A boot with no active point is skipped.
- Every bench: N = core_b_plan.num_prompts(C), timeout = ceil60(2 N tau1 + 60 s) with tau1 = tau1(L) (uniform) or
  0.9 tau1(L) + 0.1 tau1(32768) (mixed), computed exactly (Fraction); server ready timeout 1800 s.
- argv: core_b_plan.server_argv with the [stage_b] values and core_b_plan.bench_argv, unchanged: no observation flags,
  no request-id prefix (030 8.1).

Usage (CPU only, Mac or WSL):
    python3 benchmarks/capacity_plan.py --uniform-session results/core_b/formal-20261009-coreb-uniform \\
        --out capacity-plan.json [--formal]

Stdlib only, Python 3.10 compatible (tomllib is imported only by the CLI, as in adv_plan).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import random
from fractions import Fraction
from pathlib import Path

import core_b_plan as bp
from run_core_a import sha256_json

SCHEMA_VERSION = 1
STAGE = "C1"
SEED = bp.SEED
DERIVED_FROM = "formal-20261009-coreb-uniform"
WORKLOADS = ("U2K", "U8K", "M2K", "M8K")          # canonical order: Phase S boots, block shuffles start from it
WORKLOAD_SPEC = {"U2K": (2048, "uniform"), "U8K": (8192, "uniform"), "M2K": (2048, "mixed"), "M8K": (8192, "mixed")}
LONG_L = bp.MAX_MODEL_LEN                         # the 32K tier of a mixed workload's 32,640-token requests
SLO_S1 = {"id": "S1", "ttft_p95_ms": {"2048": 1000, "8192": 3000, "32768": 10000}, "tpot_p95_ms": 50,
          "mixed_long": "budget"}
SLO_S2 = {"id": "S2", "ttft_p95_ms": {"2048": 2000, "8192": 6000, "32768": 20000}, "tpot_p95_ms": 100,
          "mixed_long": "budget"}
SLO_S3 = {"id": "S3", "ttft_p95_ms": {"2048": 2000, "8192": 6000, "32768": 20000}, "tpot_p95_ms": 200,
          "mixed_long": "budget"}
SECONDARY_SLO = {"S1": SLO_S1, "S3": SLO_S3}
WINDOW = "steady"
GUARD_BAND = 0.05
N_NEAR, N_NEAR_BC, N_LOW = 5, 7, 3
C0_BRANCHES = ("", "A", "B", "C", "D")
FREEZE_BRANCHES = ("A", "B", "C")
# 030 implementation note: evidence the freeze must record (key -> required form once freeze.capacity is true)
FREEZE_EVIDENCE = {"fs_order_smoke": r"smoke-[A-Za-z0-9._-]+", "mini_smoke": r"smoke-[A-Za-z0-9._-]+",
                   "c0_decision": r"[0-9]{3}"}
FREEZE_KEYS = {"capacity", *FREEZE_EVIDENCE}
BC_BRANCHES = ("B", "C")
BC_REQUIRED = "U8K"                               # 030 8.1: branch B / C applies at least to uniform-8K
C_CAP = {"max": 128, "kv_factor": 2}
TAU1_S = {"2048": 1.845, "8192": 2.95, "32768": 7.92}
LONG_SHARE = Fraction(1, 10)                      # 007: 10 % (32640, 128) in a mixed workload
SERVER_READY_TIMEOUT_S = 1800
SENTINEL = {"L": 2048, "mix": "uniform", "C": 25, "tolerance": 0.03}
PORT = 8000                                       # as run_core_b / run_adv (capserve's product vLLM uses 8100, 029)
OUTCOMES = ("ok", "fail", "invalid")              # a Phase C run for the skip rule
PHASES = ("sentinel", "S", "C")
ROOT = Path(__file__).resolve().parent.parent
CAPACITY_PATH = ROOT / "configs" / "capacity.toml"
PINS_PATH = ROOT / "configs" / "pins.toml"
TOP_KEYS = {"schema_version", "freeze", "capacity", "slo", "secondary_slo"}
CAPACITY_KEYS = {"c0_branch", "seed", "derived_from", "workloads", "window", "guard_band", "n_near", "n_near_bc",
                 "n_near_bc_workloads", "n_low", "c_cap", "tau1_s", "server_ready_timeout_s", "mix_dataset_sha256",
                 "sentinel"}


class PlanError(bp.PlanError):
    pass


def workload_spec(w: str) -> tuple:
    if w not in WORKLOAD_SPEC:
        raise PlanError(f"unknown workload {w!r}; 030 8.3 defines {WORKLOADS}")
    return WORKLOAD_SPEC[w]


# ---------------------------------------------------------------- capacity.toml (030)

def _same(a, b) -> bool:
    """Equal values of the same TOML types all the way down (5 != 5.0 != True)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _expect(problems: list, where: str, got, want, why: str) -> None:
    if not _same(got, want):
        problems.append(f"{where} {got!r} != {want!r} ({why})")


def _keys(problems: list, where: str, table, want: set) -> bool:
    if not isinstance(table, dict):
        problems.append(f"{where} is not a table")
        return False
    if set(table) != want:
        problems.append(f"{where} keys {sorted(table)} != {sorted(want)} (030: every value present, nothing else)")
    return True


def capacity_problems(cfg: dict, pins: dict) -> list:
    """capacity.toml against the 030 rules; [] when build_plan may use it."""
    problems = []
    if not _keys(problems, "capacity.toml", cfg, TOP_KEYS):
        return problems
    _expect(problems, "schema_version", cfg.get("schema_version"), SCHEMA_VERSION, "capacity.toml schema")
    freeze = cfg.get("freeze")
    if _keys(problems, "[freeze]", freeze, FREEZE_KEYS):
        if type(freeze.get("capacity")) is not bool:
            problems.append(f"freeze.capacity {freeze.get('capacity')!r} is not a bool")
        for key in sorted(FREEZE_EVIDENCE):
            if not isinstance(freeze.get(key), str):
                problems.append(f"freeze.{key} {freeze.get(key)!r} is not a string")
    c = cfg.get("capacity")
    if _keys(problems, "[capacity]", c, CAPACITY_KEYS):
        problems += _capacity_table_problems(c, pins)
    _expect(problems, "[slo]", cfg.get("slo"), SLO_S2, "030 使用者決定: S2, mixed long requests on the 32K tier")
    _expect(problems, "[secondary_slo]", cfg.get("secondary_slo"), SECONDARY_SLO, "030 8.2: S1 and S3, descriptive")
    if isinstance(freeze, dict) and freeze.get("capacity") is True:
        branch = (c or {}).get("c0_branch") if isinstance(c, dict) else None
        if branch not in FREEZE_BRANCHES:
            problems.append(f"freeze.capacity = true needs c0_branch in {FREEZE_BRANCHES}, not {branch!r} "
                            "(028: D or no C0 result keeps §8 unfrozen)")
        unfrozen = [k for k in ("mac", "nvidia", "advanced") if (pins.get("freeze") or {}).get(k) is not True]
        if unfrozen:
            problems.append(f"freeze.capacity = true needs pins freeze {unfrozen} true")
        for key, pattern in sorted(FREEZE_EVIDENCE.items()):
            value = freeze.get(key)
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                problems.append(f"freeze.capacity = true needs freeze.{key} matching {pattern}, not {value!r} "
                                "(030 implementation note: passed smokes and C0's result log before the freeze)")
    return problems


def freeze_evidence_problems(cfg: dict, root) -> list:
    """Once freeze.capacity is true: the named FS-order and mini smoke sessions exist under root/results/capacity with
    their own mode, formal false and smoke.ok true, and C0's result decision log exists (030 implementation note).
    [] before the freeze."""
    freeze = cfg.get("freeze") or {}
    if freeze.get("capacity") is not True:
        return []
    root, problems = Path(root), []
    for key, mode in (("fs_order_smoke", "fs-order"), ("mini_smoke", "mini")):
        sid = freeze.get(key)
        path = root / "results" / "capacity" / str(sid) / "session.json"
        try:
            s = json.loads(path.read_text())
        except (OSError, ValueError):
            problems.append(f"freeze.{key}: no readable {path}")
            continue
        if s.get("mode") != mode or s.get("formal") is not False or (s.get("smoke") or {}).get("ok") is not True:
            problems.append(f"freeze.{key}: {sid} is not a passed non-formal {mode} smoke (mode {s.get('mode')!r}, "
                            f"formal {s.get('formal')!r}, smoke.ok {(s.get('smoke') or {}).get('ok')!r})")
    if not list((root / "docs" / "decisions").glob(f"{freeze.get('c0_decision')}-*.md")):
        problems.append(f"freeze.c0_decision: no docs/decisions/{freeze.get('c0_decision')}-*.md")
    return problems


def _capacity_table_problems(c: dict, pins: dict) -> list:
    problems = []
    if c.get("c0_branch") not in C0_BRANCHES:
        problems.append(f"c0_branch {c.get('c0_branch')!r} not in {C0_BRANCHES} (028)")
    _expect(problems, "seed", c.get("seed"), SEED, "Core B seed (030 8.4)")
    _expect(problems, "derived_from", c.get("derived_from"), DERIVED_FROM, "C_KV source (030 8.4)")
    _expect(problems, "workloads", c.get("workloads"), list(WORKLOADS), "030 8.3")
    _expect(problems, "window", c.get("window"), WINDOW, "030 使用者決定")
    _expect(problems, "guard_band", c.get("guard_band"), GUARD_BAND, "030 8.1 m")
    _expect(problems, "n_near", c.get("n_near"), N_NEAR, "030 8.4")
    _expect(problems, "n_near_bc", c.get("n_near_bc"), N_NEAR_BC, "030 8.1 C0 branch B / C")
    _expect(problems, "n_low", c.get("n_low"), N_LOW, "030 8.4")
    _expect(problems, "c_cap", c.get("c_cap"), C_CAP, "030 8.4: min(128, 2 C_KV)")
    _expect(problems, "tau1_s", c.get("tau1_s"), TAU1_S, "030 8.4 timeouts")
    _expect(problems, "server_ready_timeout_s", c.get("server_ready_timeout_s"), SERVER_READY_TIMEOUT_S, "030 8.4")
    _expect(problems, "sentinel", c.get("sentinel"), SENTINEL, "030 8.4 / 8.6")
    _expect(problems, "mix_dataset_sha256", c.get("mix_dataset_sha256"),
            (pins.get("core_b") or {}).get("mix_dataset_sha256"), "the pinned 007 mix, pins [core_b]")
    bc = c.get("n_near_bc_workloads")
    if not (isinstance(bc, list) and all(w in WORKLOADS for w in bc) and len(set(bc)) == len(bc)
            and BC_REQUIRED in bc):
        problems.append(f"n_near_bc_workloads {bc!r} must be distinct workloads including {BC_REQUIRED} (030 8.1)")
    elif c.get("c0_branch") == "C" and bc != [BC_REQUIRED]:
        problems.append(f"c0_branch C applies n_near_bc to {BC_REQUIRED} only, not {bc} (030 8.1)")
    return problems


def n_near_for(cfg: dict, w: str) -> int:
    """n at C^ - 1, C^, C^ + 1 and K = {1}: n_near_bc for the branch B / C workloads, else n_near (030 8.1)."""
    c = cfg["capacity"]
    if c["c0_branch"] in BC_BRANCHES and w in c["n_near_bc_workloads"]:
        return c["n_near_bc"]
    return c["n_near"]


# ---------------------------------------------------------------- C_KV, cap, timeouts

def c_kv_for(pins: dict, L: int) -> int:
    sb = pins["stage_b"]
    return bp.c_kv(sb["num_gpu_blocks"], sb["block_size"], L)


def c_cap_for(c_kv: int) -> int:
    return min(C_CAP["max"], C_CAP["kv_factor"] * c_kv)


def per_request_s(L: int, mix: str, tau1_s: dict) -> Fraction:
    """tau1 of a workload, exactly: Core's C 1 time per request; mixed = 0.9 tau1(L) + 0.1 tau1(32768) (030 8.4)."""
    tau = {int(k): Fraction(str(v)) for k, v in tau1_s.items()}
    if mix == "uniform":
        return tau[L]
    return (1 - LONG_SHARE) * tau[L] + LONG_SHARE * tau[LONG_L]


def bench_timeout_s(L: int, mix: str, C: int, tau1_s: dict) -> int:
    """ceil60(2 N tau1 + 60 s), a serial upper bound (030 8.4); exact, so a multiple of 60 is never rounded up."""
    t = 2 * bp.num_prompts(C) * per_request_s(L, mix, tau1_s) + 60
    return 60 * math.ceil(t / 60)


# ---------------------------------------------------------------- Phase S (030 8.4 steps 1-3)

def bracket_points(c_cap: int) -> list:
    out, c = [], 1
    while c < c_cap:
        out.append(c)
        c *= 2
    return out + [c_cap]


def _next(bracket: list, P: int, F) -> int | None:
    if F is None:
        later = [c for c in bracket if c > P]
        return later[0] if later else None
    return (P + F) // 2 if F - P > 1 else None


def _search_state(c_cap: int, history) -> tuple:
    """(P, F, next) after replaying history = [(C, passed), ...]; refuses a C the rule would not have run."""
    bracket, P, F = bracket_points(c_cap), 0, None
    for i, (C, passed) in enumerate(history):
        want = _next(bracket, P, F)
        if C != want:
            raise PlanError(f"Phase S step {i + 1} ran C = {C!r}; the 030 search runs {want!r} there")
        if passed:
            P = C
        else:
            F = C
    return P, F, _next(bracket, P, F)


def search_next(c_cap: int, history) -> int | None:
    """The next Phase S point after history = [(C, passed), ...] in run order, or None when the search is over."""
    return _search_state(c_cap, history)[2]


def phase_s_retries(workloads, failed_before_first_bench) -> list:
    """The workloads whose first Phase S boot failed before its first bench, retried once each at the end of Phase S
    in canonical order (030 8.4 step 4; no data has been seen for them)."""
    return [w for w in workloads if w in failed_before_first_bench]


def guard_point(history: list) -> int:
    """C^_g: the largest tested c whose every tested c' <= c is inside the guard band; 0 if none (030 8.4 step 3)."""
    best = 0
    for h in sorted(history, key=lambda h: h["C"]):
        if not h["within_band"]:
            break
        best = h["C"]
    return best


def search_summary(c_cap: int, history: list) -> dict:
    """C^, C^_g and the final F of one complete Phase S history [{"C", "pass", "within_band"}, ...]."""
    P, F, nxt = _search_state(c_cap, [(h["C"], h["pass"]) for h in history])
    if nxt is not None:
        raise PlanError(f"Phase S history is incomplete: the 030 search still runs C = {nxt}")
    return {"c_hat": P, "c_hat_g": guard_point(history), "f": F}


# ---------------------------------------------------------------- confirmation set (030 8.4)

def confirm_points(c_hat: int, c_hat_g: int, c_cap: int, *, n_near: int, n_low: int) -> list:
    """K with each point's role and n, ascending."""
    if c_hat == 0:
        return [{"C": 1, "role": "one", "n": n_near}]
    out = {c: "near" for c in (c_hat - 1, c_hat, c_hat + 1) if 1 <= c <= c_cap}
    if 1 <= c_hat_g < c_hat - 1:
        out[c_hat_g] = "low"
    elif math.ceil(c_hat / 2) < c_hat - 1:
        out[math.ceil(c_hat / 2)] = "low"
    return [{"C": c, "role": role, "n": n_low if role == "low" else n_near} for c, role in sorted(out.items())]


def _shuffled(items: list, key: str) -> list:
    order = list(items)
    random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16)).shuffle(order)
    return order


def block_order(seed: int, r: int, workloads) -> list:
    return _shuffled(list(workloads), f"{seed}:capacity:block:{r}")


def point_order(seed: int, w: str, r: int, points) -> list:
    return _shuffled(sorted(points), f"{seed}:capacity:{w}:{r}")


def build_confirm_plan(plan: dict, searches: dict) -> dict:
    """The confirm plan from each workload's Phase S history (None: both boots failed before a bench, no points)."""
    workloads, per_w = list(plan["workloads"]), {}
    for w in workloads:
        wl, history = plan["workloads"][w], searches.get(w)
        if history is None:
            per_w[w] = {"c_hat": None, "c_hat_g": None, "search": None, "points": []}
            continue
        history = [{"C": h["C"], "pass": bool(h["pass"]), "within_band": bool(h["within_band"])} for h in history]
        s = search_summary(wl["c_cap"], history)
        per_w[w] = {"c_hat": s["c_hat"], "c_hat_g": s["c_hat_g"], "search": history,
                    "points": confirm_points(s["c_hat"], s["c_hat_g"], wl["c_cap"], n_near=wl["n_near"],
                                             n_low=plan["n_low"])}
    blocks = max((p["n"] for x in per_w.values() for p in x["points"]), default=0)
    for w, x in per_w.items():
        x["orders"] = {str(r): point_order(plan["seed"], w, r, [p["C"] for p in x["points"]])
                       for r in range(1, blocks + 1)} if x["points"] else {}
    return {"schema_version": SCHEMA_VERSION, "seed": plan["seed"], "blocks": blocks,
            "block_orders": {str(r): block_order(plan["seed"], r, workloads) for r in range(1, blocks + 1)},
            "workloads": per_w}


def confirm_plan_sha256(confirm_plan: dict) -> str:
    return sha256_json(confirm_plan)


# ---------------------------------------------------------------- Phase C (030 8.4)

def _drop_from(outcomes: list):
    """The lowest point with a valid FAIL or an invalid run: it and every higher point leave later blocks."""
    bad = [C for _, C, o in outcomes if o != "ok"]
    return min(bad) if bad else None


def phase_c_boots(confirm_plan: dict, r: int, outcomes: dict) -> list:
    """Block r's boots [(w, [C in run order]), ...]; outcomes = {w: [(block, C, "ok" | "fail" | "invalid")]} of the
    blocks before r ("fail" = a valid run with maxR > 1)."""
    if not 1 <= r <= confirm_plan["blocks"]:
        return []
    boots = []
    for w in confirm_plan["block_orders"][str(r)]:
        x, done = confirm_plan["workloads"][w], outcomes.get(w) or []
        for block, C, o in done:
            if o not in OUTCOMES or not 1 <= block < r:
                raise PlanError(f"{w}: outcome {(block, C, o)!r} is not an earlier block's ok / fail / invalid")
        n = {p["C"]: p["n"] for p in x["points"]}
        cut = _drop_from(done)
        active = {C for C in n if r <= n[C] and (cut is None or C < cut)}
        points = [C for C in x["orders"].get(str(r), []) if C in active]
        if points:
            boots.append((w, points))
    return boots


# ---------------------------------------------------------------- argv (030 8.1)

def server_argv_for(pins: dict, *, port: int = PORT, vllm_bin: str = "{vllm}", base_env=None) -> tuple:
    """(argv, env): the Core B server with the Stage B values, unchanged (no observation flags)."""
    sb = pins["stage_b"]
    return bp.server_argv(pins, gdn_prefill_backend=sb["gdn_prefill_backend"],
                          gpu_memory_utilization=sb["gpu_memory_utilization"], port=port, vllm_bin=vllm_bin,
                          base_env=base_env, kv_cache_memory_bytes=sb["kv_cache_memory_bytes"])


def bench_argv_for(pins: dict, *, workload: str, C: int, result_dir: str, result_filename: str, port: int = PORT,
                   mixed_dataset: str | None = None, vllm_bin: str = "{vllm}") -> list:
    L, mix = workload_spec(workload)
    return bp.bench_argv(pins, L=L, C=C, mix=mix, result_dir=result_dir, result_filename=result_filename, port=port,
                         mixed_dataset=mixed_dataset, vllm_bin=vllm_bin)


def run_tag(phase: str, workload: str, C: int, *, block: int | None = None, position: str | None = None) -> str:
    """derived/<tag>-requests.jsonl names (030 8.7): S-<w>-c<C>, C-<w>-c<C>-r<block>, sentinel-<pos>-<w>-c<C>."""
    workload_spec(workload)
    if phase == "S" and block is None and position is None:
        return f"S-{workload}-c{C}"
    if phase == "C" and isinstance(block, int) and position is None:
        return f"C-{workload}-c{C}-r{block}"
    if phase == "sentinel" and position in ("start", "end") and block is None:
        return f"sentinel-{position}-{workload}-c{C}"
    raise PlanError(f"no run tag for phase {phase!r} block {block!r} position {position!r}")


# ---------------------------------------------------------------- plan

def build_plan(cfg: dict, pins: dict, *, session: dict, uniform_plan: dict, pins_sha256: str, capacity_sha256: str,
               formal: bool = False) -> dict:
    """The Stage C plan: per-workload C_KV / C_cap / n_near / bracket / timeouts, the SLOs and the sentinel."""
    problems = capacity_problems(cfg, pins)
    if problems:
        raise PlanError("capacity.toml: " + "; ".join(problems))
    c = cfg["capacity"]
    if formal and cfg["freeze"]["capacity"] is not True:
        raise PlanError("a formal plan needs freeze.capacity = true (030 凍結)")
    sid = session.get("session_id")
    if session.get("formal") is not True or session.get("env_valid") is not True:
        raise PlanError(f"uniform session {sid} is not formal and env-valid (formal={session.get('formal')!r}, "
                        f"env_valid={session.get('env_valid')!r}); C_KV comes only from a valid formal session")
    if sid != c["derived_from"]:
        raise PlanError(f"uniform session {sid!r} != derived_from {c['derived_from']!r} (030)")
    sb = pins["stage_b"]
    for key in ("num_gpu_blocks", "block_size"):
        if uniform_plan.get(key) != sb[key]:
            raise PlanError(f"derived_from {key} {uniform_plan.get(key)!r} != pins [stage_b] {sb[key]!r}: "
                            "the server Stage C boots is not the one C_KV was computed for")
    workloads = {}
    for w in c["workloads"]:
        L, mix = workload_spec(w)
        ckv = bp.c_kv(uniform_plan["num_gpu_blocks"], uniform_plan["block_size"], L)
        if (uniform_plan.get("c_kv") or {}).get(str(L)) != ckv:
            raise PlanError(f"recomputed C_KV({L}) = {ckv} != derived_from plan "
                            f"{(uniform_plan.get('c_kv') or {}).get(str(L))!r}")
        cap = c_cap_for(ckv)
        workloads[w] = {"L": L, "mix": mix, "c_kv": ckv, "c_cap": cap, "n_near": n_near_for(cfg, w),
                        "c0_covariate": (c["c0_branch"] in BC_BRANCHES and w in c["n_near_bc_workloads"]
                                         and mix == "uniform"),     # 028's slip marker: uniform lattices only
                        "bracket": bracket_points(cap),
                        "timeouts_s": {str(C): bench_timeout_s(L, mix, C, c["tau1_s"]) for C in range(1, cap + 1)}}
    s = c["sentinel"]
    sentinel_w = next(w for w, spec in WORKLOAD_SPEC.items() if spec == (s["L"], s["mix"]))
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "seed": c["seed"], "derived_from": sid,
            "c0_branch": c["c0_branch"], "slo": cfg["slo"], "secondary_slo": cfg["secondary_slo"],
            "window": c["window"], "guard_band": c["guard_band"], "n_low": c["n_low"], "workloads": workloads,
            "sentinel": {"workload": sentinel_w, "L": s["L"], "mix": s["mix"], "C": s["C"],
                         "N": bp.num_prompts(s["C"]), "tolerance": s["tolerance"],
                         "timeout_s": bench_timeout_s(s["L"], s["mix"], s["C"], c["tau1_s"])},
            "server_ready_timeout_s": c["server_ready_timeout_s"], "expected_blocks": sb["num_gpu_blocks"],
            "block_size": sb["block_size"], "gpu_memory_utilization": sb["gpu_memory_utilization"],
            "kv_cache_memory_bytes": sb["kv_cache_memory_bytes"], "mix_dataset_sha256": c["mix_dataset_sha256"],
            "pins_sha256": pins_sha256, "capacity_sha256": capacity_sha256}


def load_capacity_bytes() -> bytes:
    return CAPACITY_PATH.read_bytes()


def load_pins_bytes() -> bytes:
    return PINS_PATH.read_bytes()


def load_uniform_session(path) -> tuple:
    import json

    d = Path(path)
    return json.loads((d / "session.json").read_text()), json.loads((d / "plan.json").read_text())


def main(argv: list | None = None) -> int:
    import argparse
    import json
    import tomllib

    parser = argparse.ArgumentParser(description="Stage C plan (decision 030)")
    parser.add_argument("--uniform-session", required=True, help="the formal Core B uniform session (derived_from)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--formal", action="store_true", help="refuse unless freeze.capacity = true")
    args = parser.parse_args(argv)
    cap_raw, pins_raw = load_capacity_bytes(), load_pins_bytes()
    session, uniform_plan = load_uniform_session(args.uniform_session)
    try:
        plan = build_plan(tomllib.loads(cap_raw.decode()), tomllib.loads(pins_raw.decode()), session=session,
                          uniform_plan=uniform_plan, pins_sha256=hashlib.sha256(pins_raw).hexdigest(),
                          capacity_sha256=hashlib.sha256(cap_raw).hexdigest(), formal=args.formal)
    except PlanError as e:
        parser.error(str(e))
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({w: {k: x[k] for k in ("c_kv", "c_cap", "n_near")} for w, x in plan["workloads"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
