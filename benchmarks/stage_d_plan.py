#!/usr/bin/env python3
"""Stage D plan rules (decision 039; PREREGISTRATION §9), fixed before any Stage D data. Core and Stage C modules are
imported, never changed: the Phase S search, the confirmation set and the Phase C order are capacity_plan's own
functions (search_next, bracket_points, build_confirm_plan, phase_c_boots), so D-A and D-B run 030's protocol.

- configs/stage_d.toml must hold exactly the 039 values (stage_d_problems): S2 and the secondary S1 / S3 of 030, m 0.05,
  n_near 5, n_low 3, C_cap = min(128, 2 C_KV), the 030 tau1 table, the sessions and their workloads, the D-B grid and
  window, the D-C points, the sentinel (U2K at C 25 on the Core B server) and the pinned datasets. freeze.stage_d =
  true needs the three passed mini smokes (freeze_evidence_problems) and every var dataset's sha256.
- Workloads (WORKLOADS): tier (the S2 TTFT tier of the prompt class), prompt (tokens; None for a mixed or var mix),
  out (output tokens), mix ("uniform", the 007 "mixed" mix, or "var": a pinned dataset of prompt lengths uniform on
  [64, tier - 128]), server ("B0": core_b_plan.server_argv with the Stage B values, as 030; "T": B0 plus
  --long-prefill-token-threshold <pins [advanced]>, Advanced's condition T) and via ("vllm" or "gateway": D-C's
  capserve gateway at port 8080, which the runner starts as a subprocess and never imports).
- C_KV = core_b_plan.c_kv(blocks, block size, prompt + out) (mixed: L = 8192 as 030; var: the tier);
  C_cap = min(128, 2 C_KV).
- tau1 (timeouts only): 030's tau1 of the tier (mixed: 0.9 tau1(L) + 0.1 tau1(32768)), x t_tau_factor for T, x out /
  128 for longer outputs. Closed loop: ceil60(2 N tau1 + 60 s) with N = core_b_plan.num_prompts(C), as 030; open
  loop: the same serial bound with the point's own N.
- D-B (open loop): lambda_ref(w) = C_PROD(w) / mean e2e latency over W = [C, N - C) of the c1b Phase C runs at C_PROD
  that capacity_analysis.run_valid accepts (lambda_ref, from the committed c1b files: U2K 2.5459, U8K 0.6297 req/s);
  point i (1-based) of the grid has rate round(grid[i - 1] x lambda_ref, 4) and N = max(min_n, ceil(min_duration_s x
  rate)). The index i plays 030's "C" (c_cap = len(grid)), but 039 scans it: Phase S runs i = 1, 2, ... up to the
  first fail (open_search_next, not 030's bracket and bisection), K = {i^ - 1, i^, i^ + 1} with n_near each and no
  c_low (build_open_confirm_plan, in capacity_plan's confirm-plan shape so phase_c_boots orders and skips it). Every
  open-loop run has its own client seed (open_seed): vLLM 0.30.0's bench seeds numpy with --seed and draws the
  arrival gaps from it, so one seed would replay one Poisson path in every run.
- D-C (verify): each point runs n times at its C, one boot per run, blocks r = 1..n in seeded order.
- argv: B0 / T servers; clients = core_b_plan.bench_argv where it applies unchanged (128-token outputs at port 8000
  or the gateway's 8080), else the same flags with --random-output-len 512, a custom var dataset, or (D-B)
  --request-rate rate --burstiness 1.0 without --max-concurrency and with warmups 10.

Usage (CPU only, Mac or WSL):
    python3 benchmarks/stage_d_plan.py --session D-A1 --out stage-d-plan.json [--formal]

Stdlib only, Python 3.10 compatible (tomllib only in the CLI).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import capacity_analysis as ca
import capacity_plan as cp
import core_b_plan as bp
from run_core_a import sha256_json

SCHEMA_VERSION = 1
STAGE = "D"
ROOT = Path(__file__).resolve().parent.parent
STAGE_D_PATH = ROOT / "configs" / "stage_d.toml"
RESULTS = ("results", "stage_d")
PORT = cp.PORT                       # the research vLLM (8000)
GATEWAY_PORT = 8080
SEED = bp.SEED
DERIVED_FROM = cp.DERIVED_FROM
C1B_SESSION = "formal-20261017-c1b"
GUARD_BAND, N_NEAR, N_LOW = cp.GUARD_BAND, cp.N_NEAR, cp.N_LOW
TAU1_S = {"2048": 1.845, "8192": 2.95, "32768": 7.92}
T_TAU_FACTOR = 2
VAR_MIN_PROMPT = 64

WORKLOADS = {
    # the sentinel: U2K at C 25 on the Core B server, exactly 030's
    "U2K": {"tier": 2048, "prompt": 1920, "out": 128, "mix": "uniform", "server": "B0", "via": "vllm"},
    # D-A (closed loop)
    "U32K": {"tier": 32768, "prompt": 32640, "out": 128, "mix": "uniform", "server": "B0", "via": "vllm"},
    "U8K-T": {"tier": 8192, "prompt": 8064, "out": 128, "mix": "uniform", "server": "T", "via": "vllm"},
    "M8K-T": {"tier": 8192, "prompt": None, "out": 128, "mix": "mixed", "server": "T", "via": "vllm"},
    "U2K-O512": {"tier": 2048, "prompt": 1920, "out": 512, "mix": "uniform", "server": "B0", "via": "vllm"},
    "U8K-O512": {"tier": 8192, "prompt": 8064, "out": 512, "mix": "uniform", "server": "B0", "via": "vllm"},
    # D-B (open loop)
    "U2K-P": {"tier": 2048, "prompt": 1920, "out": 128, "mix": "uniform", "server": "B0", "via": "vllm"},
    "U8K-P": {"tier": 8192, "prompt": 8064, "out": 128, "mix": "uniform", "server": "B0", "via": "vllm"},
    # D-C (verify)
    "U2K-var": {"tier": 2048, "prompt": None, "out": 128, "mix": "var", "server": "B0", "via": "vllm"},
    "U8K-var": {"tier": 8192, "prompt": None, "out": 128, "mix": "var", "server": "B0", "via": "vllm"},
    "U2K-gw": {"tier": 2048, "prompt": 1920, "out": 128, "mix": "uniform", "server": "B0", "via": "gateway",
               "gateway_class": "uniform-2k"},
    "U8K-gw": {"tier": 8192, "prompt": 8064, "out": 128, "mix": "uniform", "server": "B0", "via": "gateway",
               "gateway_class": "uniform-8k"},
}
OPEN_BASE = {"U2K-P": "U2K", "U8K-P": "U8K"}        # D-B: the c1b workload lambda_ref comes from
SESSIONS = {"D-A1": ["U32K", "U8K-T", "M8K-T"], "D-A2": ["U2K-O512", "U8K-O512"], "D-B": ["U2K-P", "U8K-P"],
            "D-C": ["U2K-var", "U8K-var", "U2K-gw", "U8K-gw"]}
KINDS = {"D-A1": "closed", "D-A2": "closed", "D-B": "open", "D-C": "verify"}
OPEN_LOOP = {"grid": [0.4, 0.55, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5], "min_n": 300, "min_duration_s": 300,
             "trim": 0.1, "warmups": 10, "burstiness": 1.0}
VERIFY = {"n": 5, "points": {"U2K-var": 32, "U8K-var": 8, "U2K-gw": 32, "U8K-gw": 8}, "gateway_port": GATEWAY_PORT,
          "gateway_profile": "service/profiles/verified-formal-20261017-c1b.json"}
SENTINEL = {"workload": "U2K", "C": 25, "tolerance": 0.03}
FREEZE_EVIDENCE = {"mini_closed": "closed", "mini_open": "open", "mini_verify": "verify"}
TOP_KEYS = {"schema_version", "freeze", "stage_d", "slo", "secondary_slo"}
STAGE_D_KEYS = {"seed", "derived_from", "c1b_session", "guard_band", "n_near", "n_low", "c_cap", "tau1_s",
                "t_tau_factor", "server_ready_timeout_s", "mix_dataset_sha256", "var_dataset_sha256", "sessions",
                "open_loop", "verify", "sentinel"}
PHASES = ("sentinel", "S", "C", "V")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SESSION_ID = re.compile(r"smoke-d-[A-Za-z0-9._-]+")


class PlanError(bp.PlanError):
    pass


def workload(w: str) -> dict:
    if w not in WORKLOADS:
        raise PlanError(f"unknown workload {w!r}; 039 defines {sorted(WORKLOADS)}")
    return WORKLOADS[w]


# ---------------------------------------------------------------- stage_d.toml (039)

def stage_d_problems(cfg: dict, pins: dict) -> list:
    """stage_d.toml against the 039 rules; [] when build_plan may use it."""
    problems = []
    if not cp._keys(problems, "stage_d.toml", cfg, TOP_KEYS):
        return problems
    cp._expect(problems, "schema_version", cfg.get("schema_version"), SCHEMA_VERSION, "stage_d.toml schema")
    freeze = cfg.get("freeze")
    if cp._keys(problems, "[freeze]", freeze, {"stage_d", *FREEZE_EVIDENCE}):
        if type(freeze.get("stage_d")) is not bool:
            problems.append(f"freeze.stage_d {freeze.get('stage_d')!r} is not a bool")
        for key in sorted(FREEZE_EVIDENCE):
            if not isinstance(freeze.get(key), str):
                problems.append(f"freeze.{key} {freeze.get(key)!r} is not a string")
    d = cfg.get("stage_d")
    if cp._keys(problems, "[stage_d]", d, STAGE_D_KEYS):
        want = {"seed": SEED, "derived_from": DERIVED_FROM, "c1b_session": C1B_SESSION, "guard_band": GUARD_BAND,
                "n_near": N_NEAR, "n_low": N_LOW, "c_cap": cp.C_CAP, "tau1_s": TAU1_S, "t_tau_factor": T_TAU_FACTOR,
                "server_ready_timeout_s": cp.SERVER_READY_TIMEOUT_S, "sessions": SESSIONS, "open_loop": OPEN_LOOP,
                "verify": VERIFY, "sentinel": SENTINEL,
                "mix_dataset_sha256": {"8192": ((pins.get("core_b") or {}).get("mix_dataset_sha256") or {}).get("8192")}}
        for key, value in want.items():
            cp._expect(problems, key, d.get(key), value, "039")
        var = d.get("var_dataset_sha256")
        if not (isinstance(var, dict) and set(var) == {"2048", "8192"}
                and all(v == "" or (isinstance(v, str) and _HEX64.fullmatch(v)) for v in var.values())):
            problems.append(f"var_dataset_sha256 {var!r} must map 2048 and 8192 to a sha256 (or \"\" before the freeze)")
    cp._expect(problems, "[slo]", cfg.get("slo"), cp.SLO_S2, "037 §3: S2, the user's Stage C choice")
    cp._expect(problems, "[secondary_slo]", cfg.get("secondary_slo"), cp.SECONDARY_SLO, "030 8.2: descriptive")
    if isinstance(freeze, dict) and freeze.get("stage_d") is True:
        for key in sorted(FREEZE_EVIDENCE):
            if not (isinstance(freeze.get(key), str) and _SESSION_ID.fullmatch(freeze[key])):
                problems.append(f"freeze.stage_d = true needs freeze.{key} naming a smoke-d-* session, not "
                                f"{freeze.get(key)!r}")
        var = (d or {}).get("var_dataset_sha256") if isinstance(d, dict) else None
        if not (isinstance(var, dict) and all(isinstance(v, str) and _HEX64.fullmatch(v) for v in var.values())):
            problems.append("freeze.stage_d = true needs every var_dataset_sha256 pinned")
    return problems


def freeze_evidence_problems(cfg: dict, root) -> list:
    """Once freeze.stage_d is true: each named mini smoke exists under root/results/stage_d with its kind, mode mini,
    formal false and smoke.ok true. [] before the freeze."""
    freeze = cfg.get("freeze") or {}
    if freeze.get("stage_d") is not True:
        return []
    problems = []
    for key, kind in sorted(FREEZE_EVIDENCE.items()):
        sid = freeze.get(key)
        path = Path(root).joinpath(*RESULTS, str(sid), "session.json")
        try:
            s = json.loads(path.read_text())
        except (OSError, ValueError):
            problems.append(f"freeze.{key}: no readable {path}")
            continue
        if (s.get("mode"), s.get("kind"), s.get("formal"), (s.get("smoke") or {}).get("ok")) != ("mini", kind, False,
                                                                                                True):
            problems.append(f"freeze.{key}: {sid} is not a passed non-formal {kind} mini smoke")
    return problems


# ---------------------------------------------------------------- C_KV, cap, timeouts

def kv_len(w: str) -> int:
    spec = workload(w)
    if spec["mix"] == "uniform":
        return spec["prompt"] + spec["out"]
    return spec["tier"]                                  # mixed: 030's L; var: the longest prompt + 128


def c_kv_for(pins: dict, w: str) -> int:
    return cp.c_kv_for(pins, kv_len(w))


def tau1(cfg: dict, w: str) -> Fraction:
    """Exact tau1 of a workload (timeouts only, 039)."""
    spec, d = workload(w), cfg["stage_d"]
    mix = "mixed" if spec["mix"] == "mixed" else "uniform"
    t = cp.per_request_s(spec["tier"], mix, d["tau1_s"])
    if spec["server"] == "T":
        t *= d["t_tau_factor"]
    return t * Fraction(spec["out"], bp.N_GEN)


def timeout_s(t1: Fraction, n: int) -> int:
    """ceil60(2 N tau1 + 60 s): 030's serial upper bound, exact."""
    return 60 * math.ceil((2 * n * t1 + 60) / 60)


# ---------------------------------------------------------------- D-B: lambda_ref from c1b (039)

def lambda_ref(c1b_dir, w: str) -> dict:
    """lambda_ref of a c1b workload from its committed files: C_PROD (analysis.json) over the mean e2e latency, in
    seconds, of every request in W = [C, N - C) of the Phase C runs at C_PROD that run_valid accepts."""
    d = Path(c1b_dir)
    analysis = json.loads((d / "analysis.json").read_text())
    est = analysis["workloads"][w]["c_prod"]
    if est.get("status") != "EXACT" or not isinstance(est.get("value"), int) or est["value"] < 1:
        raise PlanError(f"c1b C_PROD({w}) {est!r} is not an EXACT positive value: no lambda_ref (039 D-B)")
    C = est["value"]
    rows = [json.loads(line) for line in (d / "rows.jsonl").read_text().splitlines() if line.strip()]
    lat, tags = [], []
    for row in rows:
        if row.get("phase") != "C" or row.get("workload") != w or row.get("C") != C:
            continue
        reqs = None
        if row.get("derived"):
            reqs = [json.loads(x) for x in (d / row["derived"]).read_text().splitlines() if x.strip()]
        ok, _ = ca.run_valid(row, reqs, mix="uniform", C=C, N=row["N"])
        if ok:
            tags.append(row["tag"])
            lat += [Fraction(str(q["latency_ms"])) for q in reqs if C <= q["idx"] < row["N"] - C]
    if not lat:
        raise PlanError(f"no valid c1b Phase C run of {w} at C_PROD {C}: no lambda_ref (039 D-B)")
    mean_s = sum(lat) / len(lat) / 1000
    return {"workload": w, "c_prod": C, "runs": tags, "requests": len(lat), "mean_latency_s": round(float(mean_s), 6),
            "lambda_ref": round(float(C / mean_s), 6)}


def open_points(cfg: dict, w: str, ref: dict) -> dict:
    """{str(i): {"rate", "N", "timeout_s"}} for grid index i = 1..len(grid) (039 D-B)."""
    ol, t1 = cfg["stage_d"]["open_loop"], tau1(cfg, w)
    out = {}
    for i, f in enumerate(ol["grid"], start=1):
        rate = round(f * ref["lambda_ref"], 4)
        n = max(ol["min_n"], math.ceil(ol["min_duration_s"] * rate))
        out[str(i)] = {"rate": rate, "N": n, "timeout_s": timeout_s(t1, n)}
    return out


def open_search_next(c_cap: int, history) -> int | None:
    """039 D-B Phase S: the grid indices 1, 2, ... in order up to the first fail (or c_cap); history = [(i, passed),
    ...] in run order. Refuses a history the scan would not have run."""
    for k, (i, passed) in enumerate(history):
        if i != k + 1 or (not passed and k != len(history) - 1):
            raise PlanError(f"D-B Phase S step {k + 1} ran {i!r}; the 039 scan runs {k + 1} and stops at a fail")
    if history and (not history[-1][1] or len(history) >= c_cap):
        return None
    return len(history) + 1 if c_cap >= 1 else None


def open_search_summary(c_cap: int, history: list) -> dict:
    """i^ (the last passing index; 0 when index 1 fails) of one complete D-B Phase S history."""
    pairs = [(h["C"], h["pass"]) for h in history]
    nxt = open_search_next(c_cap, pairs)
    if nxt is not None:
        raise PlanError(f"D-B Phase S history is incomplete: the 039 scan still runs {nxt}")
    passed = [i for i, ok in pairs if ok]
    return {"c_hat": max(passed) if passed else 0, "c_hat_g": None, "f": next((i for i, ok in pairs if not ok), None)}


def build_open_confirm_plan(plan: dict, searches: dict) -> dict:
    """039 D-B's confirm plan in capacity_plan.build_confirm_plan's shape: K = {i^ - 1, i^, i^ + 1} within the grid
    ({1} when index 1 failed), n_near each, no c_low; orders from capacity_plan.point_order / block_order."""
    workloads, per_w = list(plan["workloads"]), {}
    for w in workloads:
        x, history = plan["workloads"][w], searches.get(w)
        if history is None:
            per_w[w] = {"c_hat": None, "c_hat_g": None, "search": None, "points": []}
            continue
        history = [{"C": h["C"], "pass": bool(h["pass"]), "within_band": bool(h["within_band"])} for h in history]
        c_hat = open_search_summary(x["c_cap"], history)["c_hat"]
        near = [1] if c_hat == 0 else [i for i in (c_hat - 1, c_hat, c_hat + 1) if 1 <= i <= x["c_cap"]]
        per_w[w] = {"c_hat": c_hat, "c_hat_g": None, "search": history,
                    "points": [{"C": i, "role": "one" if c_hat == 0 else "near", "n": x["n_near"]} for i in near]}
    blocks = max((p["n"] for x in per_w.values() for p in x["points"]), default=0)
    for w, x in per_w.items():
        x["orders"] = {str(r): cp.point_order(plan["seed"], w, r, [p["C"] for p in x["points"]])
                       for r in range(1, blocks + 1)} if x["points"] else {}
    return {"schema_version": cp.SCHEMA_VERSION, "seed": plan["seed"], "blocks": blocks,
            "block_orders": {str(r): cp.block_order(plan["seed"], r, workloads) for r in range(1, blocks + 1)},
            "workloads": per_w}


def open_seed(w: str, i: int, *, phase: str, block: int | None) -> int:
    """039 D-B: the client seed of one open-loop run (its prompts and its Poisson arrival path)."""
    key = f"{SEED}:stage_d:open:{w}:{i}:{phase}:{block}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 31)


def open_window(n: int, trim: float) -> tuple:
    """D-B's W = idx in [ceil(trim N), N - ceil(trim N))."""
    k = math.ceil(Fraction(str(trim)) * n)
    return k, n - k


# ---------------------------------------------------------------- argv (039)

def server_argv_for(pins: dict, w: str, *, port: int = PORT, vllm_bin: str = "{vllm}", base_env=None) -> tuple:
    """(argv, env): 030's Core B server (B0), or B0 plus Advanced's --long-prefill-token-threshold (T)."""
    argv, env = cp.server_argv_for(pins, port=port, vllm_bin=vllm_bin, base_env=base_env)
    if workload(w)["server"] == "T":
        argv = argv + ["--long-prefill-token-threshold", str(pins["advanced"]["long_prefill_token_threshold"])]
    return argv, env


def bench_argv_for(pins: dict, w: str, *, C: int | None = None, rate: float | None = None, N: int | None = None,
                   warmups: int | None = None, seed: int = SEED, result_dir: str, result_filename: str,
                   dataset: str | None = None, vllm_bin: str = "{vllm}") -> list:
    """The client of one run: closed loop at C (C given) or open loop at rate (rate, N and warmups given).
    core_b_plan.bench_argv itself where it applies (128-token outputs of a uniform or mixed workload, closed loop,
    port 8000 or the gateway's 8080); otherwise its flags in its order with only these changes: --random-output-len
    512, a var custom dataset, or (open loop) --request-rate rate --burstiness 1.0, no --max-concurrency and the
    run's own seed (open_seed)."""
    spec = workload(w)
    port = GATEWAY_PORT if spec["via"] == "gateway" else PORT
    if (C is None) == (rate is None):
        raise PlanError("a run is closed loop (C) or open loop (rate), exactly one")
    if C is not None and seed != SEED:
        raise PlanError("a closed-loop run uses the Core B seed (030)")
    if C is not None and spec["mix"] in ("uniform", "mixed") and spec["out"] == bp.N_GEN:
        L = spec["prompt"] + spec["out"] if spec["mix"] == "uniform" else spec["tier"]
        return bp.bench_argv(pins, L=L, C=C, mix=spec["mix"], result_dir=result_dir, result_filename=result_filename,
                             port=port, mixed_dataset=dataset, vllm_bin=vllm_bin)
    argv = [vllm_bin, "bench", "serve", "--backend", "vllm", "--base-url", f"http://127.0.0.1:{port}", "--endpoint",
            "/v1/completions", "--model", pins["core_b"]["repo"], "--tokenizer", bp.snapshot_dir(pins)]
    if spec["mix"] == "uniform":
        argv += ["--dataset-name", "random", "--random-input-len", str(spec["prompt"]), "--random-output-len",
                 str(spec["out"]), "--random-range-ratio", "0"]
    elif spec["mix"] == "var":
        if not dataset:
            raise PlanError(f"{w} needs its pinned var dataset (039 D-C)")
        argv += ["--dataset-name", "custom", "--dataset-path", dataset, "--skip-chat-template", "--disable-shuffle",
                 "--custom-output-len", str(spec["out"])]
    else:
        raise PlanError(f"{w}: no client for mix {spec['mix']!r} with {spec['out']} output tokens")
    if C is not None:
        argv += ["--num-prompts", str(bp.num_prompts(C)), "--max-concurrency", str(C), "--num-warmups",
                 str(bp.num_warmups(C)), "--request-rate", "inf"]
    else:
        argv += ["--num-prompts", str(N), "--num-warmups", str(warmups), "--request-rate", str(rate), "--burstiness",
                 str(OPEN_LOOP["burstiness"])]
    return argv + ["--ignore-eos", "--temperature", "0", "--seed", str(seed), "--percentile-metrics",
                   "ttft,tpot,itl,e2el,client_queue_time", "--metric-percentiles", "50,95,99", "--save-result",
                   "--save-detailed", "--result-dir", result_dir, "--result-filename", result_filename]


def run_tag(phase: str, w: str, x: int, *, block: int | None = None, position: str | None = None,
            kind: str = "closed") -> str:
    """derived/<tag>-requests.jsonl names: S-<w>-c<C>, C-<w>-c<C>-r<block>, sentinel-<pos>-<w>-c<C>, V-<w>-c<C>-r<block>;
    an open-loop point is its grid index: S-<w>-i<i>, C-<w>-i<i>-r<block>."""
    workload(w)
    unit = "i" if kind == "open" and phase in ("S", "C") else "c"
    if phase == "S" and block is None and position is None:
        return f"S-{w}-{unit}{x}"
    if phase in ("C", "V") and isinstance(block, int) and position is None:
        return f"{phase}-{w}-{unit}{x}-r{block}"
    if phase == "sentinel" and position in ("start", "end") and block is None:
        return f"sentinel-{position}-{w}-c{x}"
    raise PlanError(f"no run tag for phase {phase!r} block {block!r} position {position!r}")


# ---------------------------------------------------------------- plan

def build_plan(cfg: dict, pins: dict, *, session_name: str, uniform_plan: dict, refs: dict | None, pins_sha256: str,
               stage_d_sha256: str, formal: bool = False) -> dict:
    """The plan of one Stage D session (D-A1, D-A2, D-B or D-C). refs: lambda_ref of OPEN_BASE's workloads (D-B)."""
    problems = stage_d_problems(cfg, pins)
    if problems:
        raise PlanError("stage_d.toml: " + "; ".join(problems))
    if session_name not in SESSIONS:
        raise PlanError(f"unknown session {session_name!r}; 039 defines {sorted(SESSIONS)}")
    if formal and cfg["freeze"]["stage_d"] is not True:
        raise PlanError("a formal plan needs freeze.stage_d = true (039 凍結)")
    sb, d = pins["stage_b"], cfg["stage_d"]
    for key in ("num_gpu_blocks", "block_size"):
        if uniform_plan.get(key) != sb[key]:
            raise PlanError(f"derived_from {key} {uniform_plan.get(key)!r} != pins [stage_b] {sb[key]!r}")
    kind, workloads = KINDS[session_name], {}
    for w in d["sessions"][session_name]:
        spec = workload(w)
        ckv = c_kv_for(pins, w)
        x = {**spec, "kv_len": kv_len(w), "c_kv": ckv}
        if kind == "closed":
            cap = cp.c_cap_for(ckv)
            x.update(c_cap=cap, n_near=d["n_near"], bracket=cp.bracket_points(cap),
                     timeouts_s={str(C): timeout_s(tau1(cfg, w), bp.num_prompts(C)) for C in range(1, cap + 1)})
        elif kind == "open":
            ref = (refs or {}).get(OPEN_BASE[w])
            if not ref:
                raise PlanError(f"{w} needs lambda_ref of {OPEN_BASE[w]} from {d['c1b_session']} (039 D-B)")
            grid = d["open_loop"]["grid"]
            x.update(c_cap=len(grid), n_near=d["n_near"], scan=list(range(1, len(grid) + 1)), lambda_ref=ref,
                     points=open_points(cfg, w, ref))
        else:
            C = d["verify"]["points"][w]
            x.update(C=C, n=d["verify"]["n"], timeout_s=timeout_s(tau1(cfg, w), bp.num_prompts(C)))
        workloads[w] = x
    s = d["sentinel"]
    sentinel = {"workload": s["workload"], "L": 2048, "mix": "uniform", "C": s["C"], "N": bp.num_prompts(s["C"]),
                "tolerance": s["tolerance"], "timeout_s": timeout_s(tau1(cfg, s["workload"]), bp.num_prompts(s["C"]))}
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "session": session_name, "kind": kind, "seed": d["seed"],
            "derived_from": d["derived_from"], "c1b_session": d["c1b_session"], "slo": cfg["slo"],
            "secondary_slo": cfg["secondary_slo"], "window": cp.WINDOW, "guard_band": d["guard_band"],
            "n_low": d["n_low"], "open_loop": d["open_loop"] if kind == "open" else None,
            "verify": {k: d["verify"][k] for k in ("gateway_port", "gateway_profile")} if kind == "verify" else None,
            "workloads": workloads, "sentinel": sentinel, "server_ready_timeout_s": d["server_ready_timeout_s"],
            "expected_blocks": sb["num_gpu_blocks"], "block_size": sb["block_size"],
            "gpu_memory_utilization": sb["gpu_memory_utilization"], "kv_cache_memory_bytes": sb["kv_cache_memory_bytes"],
            "long_prefill_token_threshold": pins["advanced"]["long_prefill_token_threshold"],
            "mix_dataset_sha256": d["mix_dataset_sha256"], "var_dataset_sha256": d["var_dataset_sha256"],
            "pins_sha256": pins_sha256, "stage_d_sha256": stage_d_sha256}


def plan_sha256(plan: dict) -> str:
    return sha256_json(plan)


def load_stage_d_bytes() -> bytes:
    return STAGE_D_PATH.read_bytes()


def main(argv: list | None = None) -> int:
    import argparse
    import tomllib

    parser = argparse.ArgumentParser(description="Stage D plan (decision 039)")
    parser.add_argument("--session", required=True, choices=sorted(SESSIONS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--formal", action="store_true", help="refuse unless freeze.stage_d = true")
    args = parser.parse_args(argv)
    cfg_raw, pins_raw = load_stage_d_bytes(), cp.load_pins_bytes()
    cfg, pins = tomllib.loads(cfg_raw.decode()), tomllib.loads(pins_raw.decode())
    sb = pins["stage_b"]
    refs = None
    if KINDS[args.session] == "open":
        c1b = ROOT / "results" / "capacity" / cfg["stage_d"]["c1b_session"]
        refs = {w: lambda_ref(c1b, w) for w in sorted(set(OPEN_BASE.values()))}
    try:
        plan = build_plan(cfg, pins, session_name=args.session,
                          uniform_plan={"num_gpu_blocks": sb["num_gpu_blocks"], "block_size": sb["block_size"]},
                          refs=refs, pins_sha256=hashlib.sha256(pins_raw).hexdigest(),
                          stage_d_sha256=hashlib.sha256(cfg_raw).hexdigest(), formal=args.formal)
    except PlanError as e:
        parser.error(str(e))
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({w: {k: x.get(k) for k in ("c_kv", "c_cap", "C", "n")} for w, x in plan["workloads"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
