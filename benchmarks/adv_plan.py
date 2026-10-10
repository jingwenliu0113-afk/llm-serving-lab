#!/usr/bin/env python3
"""Advanced plan (decision 024; PREREGISTRATION §7): the five server conditions that isolate the cause of H-B4
and the 60-boot order, fixed before any Advanced data. Core modules are imported, never changed.

- conditions: every one is the Core B server (core_b_plan.server_argv: same U, gdn backend, port, Stage B K) plus
  the same observability flags `--otlp-traces-endpoint <[advanced].otlp_endpoint> --enable-logging-iteration-details`.
  T appends `--long-prefill-token-threshold 512`, G appends `--no-scheduler-reserve-full-isl`, TG appends both, and
  K3 passes [advanced].kv_low_bytes (3 GiB) as Core's kv_cache_memory_bytes instead of the Stage B K.
- server env = core_b_plan.server_env minus every inherited OTEL_* plus OTEL_EXPORTER_OTLP_TRACES_PROTOCOL =
  [advanced].otel_protocol (vLLM's otel.py defaults to grpc; the runner's stdlib receiver speaks http/protobuf).
- client = core_b_plan.bench_argv plus `--request-id-prefix adv-<cond>-<L>-<mix>-r<restart>-` (bench serve
  otherwise draws a random prefix), so server spans align with client requests.
- rules checked against pins [advanced] (pins_problems):
  - the conditions table [advanced.conditions] = each condition's argv delta, exactly condition_flags (no drift).
  - otlp_endpoint = OTLP_ENDPOINT, otel_protocol = OTEL_PROTOCOL and timeouts_s = TIMEOUTS_S, the exact 024 values.
  - T = MNBT // (C_KV(32768) + 1) = 2048 // 4 = 512.
  - K3 = the smallest multiple of 256 MiB below the Stage B K whose pool gives C_KV(L) >= C_knee(L) at 2K and 8K,
    >= 128 blocks (019) and >= 67 blocks (one 32K request). vLLM 0.30.0 sizes the pool as K // bytes_per_block
    (kv_cache_utils.py:1695 / 1750), so 248 blocks at 4 GiB bracket the pool of any K (blocks_bounds): 3 GiB is
    exactly 186 blocks (C_KV 26 / 9), 2.75 GiB is 170-171 (C_KV(2K) = 24 < 25).
    After smoke FS1 the measured count must pass, and the next lower 256 MiB multiple must provably fail.
- plan: C = the uniform C_knee recomputed with core_b_analysis.knee from the formal, env-valid
  [advanced].derived_from session and required to equal [advanced].c (25 / 8). Three blocks, block r = restart r,
  each a permutation of the 20 (cond, L, mix) cells (canonical order CONDITIONS x LEVELS x MIXES) shuffled by
  random.Random(int(sha256("<seed>:advanced:block:<r>"), 16)). N = max(10 C, 200), warmup = C (Core);
  timeout_s = [advanced].timeouts_s["<mix>_<L>"]; expected num_gpu_blocks per condition; pins_sha256 = sha256 of
  the pins.toml bytes. The plan also carries the Stage B U / K / block_size, so run_core_b.stage_b_problems applies.

Usage (CPU only, Mac or WSL):
    python3 benchmarks/adv_plan.py --uniform-session results/core_b/formal-20261009-coreb-uniform --out adv-plan.json

Stdlib only, Python 3.10 compatible (tomllib is imported only by the CLI, as in core_b_plan).
"""

from __future__ import annotations

import hashlib
import itertools
import random
from pathlib import Path

import core_b_analysis as ba
import core_b_plan as bp

SCHEMA_VERSION = 1
CONDITIONS = ("B0", "T", "G", "TG", "K3")
LEVELS = bp.MIXED_LEVELS            # (2048, 8192): H-B4's levels
MIXES = ("uniform", "mixed")
RESTARTS = ba.RESTARTS              # 3; block r holds restart r of every cell
CELLS = tuple(itertools.product(CONDITIONS, LEVELS, MIXES))  # canonical order before each block's shuffle
DERIVED_FROM = "formal-20261009-coreb-uniform"
C_KNEE = {2048: 25, 8192: 8}        # 024: Core's uniform C_knee; build_plan recomputes it
OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"   # 024: run_adv's in-process receiver
TIMEOUTS_S = {"uniform_2048": 300, "mixed_2048": 600, "uniform_8192": 720, "mixed_8192": 960}  # 024, per <mix>_<L>
OTEL_PROTOCOL = "http/protobuf"
OTEL_PROTOCOL_VAR = "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"
KV_STEP = bp.KV_ROUND_MIB * 2**20   # 256 MiB
PINS_PATH = Path(__file__).resolve().parent.parent / "configs" / "pins.toml"


class PlanError(bp.PlanError):
    pass


def _check_cond(cond: str) -> None:
    if cond not in CONDITIONS:
        raise PlanError(f"unknown condition {cond!r}; 024 defines {CONDITIONS}")


# ----------------------------------------------------------------- conditions --

def condition_flags(cond: str, pins: dict) -> list:
    """Server flags a condition appends after the observability flags (K3 changes the K value instead)."""
    _check_cond(cond)
    t = ["--long-prefill-token-threshold", str(pins["advanced"]["long_prefill_token_threshold"])]
    g = ["--no-scheduler-reserve-full-isl"]
    return {"B0": [], "T": t, "G": g, "TG": t + g, "K3": []}[cond]


def observability_flags(pins: dict) -> list:
    return ["--otlp-traces-endpoint", pins["advanced"]["otlp_endpoint"], "--enable-logging-iteration-details"]


def kv_cache_bytes_for(cond: str, pins: dict) -> int:
    _check_cond(cond)
    return pins["advanced"]["kv_low_bytes"] if cond == "K3" else pins["stage_b"]["kv_cache_memory_bytes"]


def server_env_for(base_env, pins: dict) -> dict:
    """core_b_plan.server_env (same pin-memory choice as core_b_plan.server_argv) without any inherited OTEL_*,
    plus the OTLP protocol."""
    pin_memory = (pins.get("stage_b") or {}).get("wsl2_pin_memory") or "0"
    env = bp.server_env(base_env or {}, pin_memory=pin_memory)
    return {**{k: v for k, v in env.items() if not k.startswith("OTEL_")},
            OTEL_PROTOCOL_VAR: pins["advanced"]["otel_protocol"]}


def server_argv_for(cond: str, pins: dict, *, gdn_prefill_backend: str, gpu_memory_utilization: str, port: int,
                    vllm_bin: str = "{vllm}", base_env=None) -> tuple:
    """(argv, env) of one condition's server: Core's argv + observability flags + the condition's flags."""
    argv, _ = bp.server_argv(pins, gdn_prefill_backend=gdn_prefill_backend,
                             gpu_memory_utilization=gpu_memory_utilization, port=port, vllm_bin=vllm_bin,
                             base_env=base_env, kv_cache_memory_bytes=kv_cache_bytes_for(cond, pins))
    return argv + observability_flags(pins) + condition_flags(cond, pins), server_env_for(base_env, pins)


def request_id_prefix(cond: str, L: int, mix: str, restart: int) -> str:
    _check_cond(cond)
    if L not in LEVELS or mix not in MIXES or restart not in range(1, RESTARTS + 1):
        raise PlanError(f"no Advanced cell ({cond}, {L}, {mix}, r{restart}) (024)")
    return f"adv-{cond}-{L}-{mix}-r{restart}-"


def bench_argv_for(pins: dict, *, cond: str, L: int, C: int, mix: str, restart: int, result_dir: str,
                   result_filename: str, port: int, mixed_dataset: str | None = None, vllm_bin: str = "{vllm}") -> list:
    """Core's `vllm bench serve` argv plus the deterministic request-id prefix (the only client difference)."""
    argv = bp.bench_argv(pins, L=L, C=C, mix=mix, result_dir=result_dir, result_filename=result_filename, port=port,
                         mixed_dataset=mixed_dataset, vllm_bin=vllm_bin)
    return argv + ["--request-id-prefix", request_id_prefix(cond, L, mix, restart)]


# ---------------------------------------------------------------------- rules --

def long_prefill_threshold(*, num_gpu_blocks: int, block_size: int) -> int:
    """024 H-C1: T = MNBT // (n + 1), n = C_KV(32768) = long requests the pool holds at once."""
    return bp.MAX_NUM_BATCHED_TOKENS // (bp.c_kv(num_gpu_blocks, block_size, bp.MAX_MODEL_LEN) + 1)


def k3_problems(num_gpu_blocks: int, *, block_size: int, c: dict) -> list:
    """024 H-C4 rule for the KV dose; [] when a pool of num_gpu_blocks satisfies it."""
    problems = [f"C_KV({int(L)}) = {bp.c_kv(num_gpu_blocks, block_size, int(L))} < C_knee {C}"
                for L, C in sorted(c.items(), key=lambda x: int(x[0]))
                if bp.c_kv(num_gpu_blocks, block_size, int(L)) < C]
    if num_gpu_blocks < bp.MAX_NUM_SEQS:
        problems.append(f"num_gpu_blocks {num_gpu_blocks} < max_num_seqs {bp.MAX_NUM_SEQS} (019)")
    if bp.c_kv(num_gpu_blocks, block_size, bp.MAX_MODEL_LEN) < 1:
        problems.append(f"num_gpu_blocks {num_gpu_blocks} cannot hold one {bp.MAX_MODEL_LEN}-token request")
    return problems


def blocks_bounds(kv_bytes: int, *, ref_bytes: int, ref_blocks: int) -> tuple:
    """Pool size at kv_bytes given ref_blocks at ref_bytes, for blocks(K) = K // s with an unknown s:
    ref_blocks = ref_bytes // s puts s in (ref_bytes / (ref_blocks + 1), ref_bytes / ref_blocks], hence
    floor(K ref_blocks / ref_bytes) <= blocks(K) <= ceil(K (ref_blocks + 1) / ref_bytes) - 1."""
    return kv_bytes * ref_blocks // ref_bytes, -(-kv_bytes * (ref_blocks + 1) // ref_bytes) - 1


def kv_low_bytes_rule(*, ref_bytes: int, ref_blocks: int, block_size: int, c: dict) -> int:
    """024 K3: the smallest multiple of 256 MiB below ref_bytes whose pool provably satisfies k3_problems while
    every smaller multiple provably fails; a multiple the bounds cannot decide is refused (measure it instead)."""
    for k in range(KV_STEP, ref_bytes, KV_STEP):
        lo, hi = blocks_bounds(k, ref_bytes=ref_bytes, ref_blocks=ref_blocks)
        if not k3_problems(lo, block_size=block_size, c=c):
            return k
        if not k3_problems(hi, block_size=block_size, c=c):
            raise PlanError(f"{k} B gives {lo}-{hi} blocks, which straddles the K3 rule: measure it")
    raise PlanError(f"no multiple of 256 MiB below {ref_bytes} B satisfies the K3 rule")


def _k3_pin_problems(adv: dict, sb: dict, freeze_advanced: bool) -> list:
    k4, nb4, bs = sb["kv_cache_memory_bytes"], sb["num_gpu_blocks"], sb["block_size"]
    kv, k3 = adv.get("kv_low_bytes"), adv.get("num_gpu_blocks_k3")
    problems = []
    measured = isinstance(k3, int) and not isinstance(k3, bool) and k3 > 0
    if not measured and k3 != 0:
        problems.append(f"num_gpu_blocks_k3 {k3!r} is neither a measured count nor 0 (not measured yet)")
    elif k3 == 0 and freeze_advanced:
        problems.append("freeze.advanced = true needs the measured num_gpu_blocks_k3 (smoke FS1)")
    if not (isinstance(kv, int) and 0 < kv < k4 and kv % KV_STEP == 0):
        problems.append(f"kv_low_bytes {kv!r} is not a multiple of 256 MiB below the Stage B K {k4}")
    elif measured:
        problems += [f"K3 measured {k3} blocks: {p}" for p in k3_problems(k3, block_size=bs, c=C_KNEE)]
        lo, hi = blocks_bounds(kv, ref_bytes=k4, ref_blocks=nb4)  # the count must be possible at kv_low_bytes
        if not lo <= k3 <= hi:
            problems.append(f"num_gpu_blocks_k3 {k3} is impossible at kv_low_bytes {kv} B "
                            f"({lo}-{hi} blocks given {nb4} at {k4} B)")
        below = blocks_bounds(kv - KV_STEP, ref_bytes=kv, ref_blocks=k3)[1]
        if not k3_problems(below, block_size=bs, c=C_KNEE):
            problems.append(f"kv_low_bytes {kv} is not the smallest 256 MiB multiple: {kv - KV_STEP} B "
                            f"(<= {below} blocks) may already pass the K3 rule")
    else:
        try:
            want = kv_low_bytes_rule(ref_bytes=k4, ref_blocks=nb4, block_size=bs, c=C_KNEE)
        except PlanError as e:
            want = str(e)
        if kv != want:
            problems.append(f"kv_low_bytes {kv} != the K3 rule's {want} (024 H-C4)")
    return problems


def pins_problems(pins: dict) -> list:
    """pins [advanced] against the 024 rules; [] when build_plan may use it."""
    adv, sb = pins.get("advanced"), pins.get("stage_b") or {}
    if not adv:
        return ["pins.toml has no [advanced] table (decision 024)"]
    if not all(isinstance(sb.get(k), int) and sb[k] > 0
               for k in ("num_gpu_blocks", "block_size", "kv_cache_memory_bytes")):
        return ["pins [stage_b] lacks num_gpu_blocks / block_size / kv_cache_memory_bytes (§6)"]
    problems = []
    if adv.get("seed") != bp.SEED:
        problems.append(f"seed {adv.get('seed')!r} != {bp.SEED} (024)")
    if adv.get("derived_from") != DERIVED_FROM:
        problems.append(f"derived_from {adv.get('derived_from')!r} != {DERIVED_FROM!r} (024)")
    if {int(L): C for L, C in (adv.get("c") or {}).items()} != C_KNEE:
        problems.append(f"c {adv.get('c')!r} != Core's C_knee {C_KNEE} (024)")
    t = long_prefill_threshold(num_gpu_blocks=sb["num_gpu_blocks"], block_size=sb["block_size"])
    if adv.get("long_prefill_token_threshold") != t:
        problems.append(f"long_prefill_token_threshold {adv.get('long_prefill_token_threshold')!r} != rule {t} (024)")
    if adv.get("num_gpu_blocks_k4") != sb["num_gpu_blocks"]:
        problems.append(f"num_gpu_blocks_k4 {adv.get('num_gpu_blocks_k4')!r} != [stage_b] {sb['num_gpu_blocks']}")
    problems += _k3_pin_problems(adv, sb, bool((pins.get("freeze") or {}).get("advanced")))
    if adv.get("otel_protocol") != OTEL_PROTOCOL:
        problems.append(f"otel_protocol {adv.get('otel_protocol')!r} != {OTEL_PROTOCOL!r} (the stdlib receiver)")
    if adv.get("otlp_endpoint") != OTLP_ENDPOINT:
        problems.append(f"otlp_endpoint {adv.get('otlp_endpoint')!r} != {OTLP_ENDPOINT!r}, the runner's receiver (024)")
    timeouts = adv.get("timeouts_s")
    if not (isinstance(timeouts, dict) and timeouts == TIMEOUTS_S
            and all(type(v) is int for v in timeouts.values())):
        problems.append(f"timeouts_s {timeouts!r} != {TIMEOUTS_S} (024)")
    # the conditions table must be what the code appends, so the two cannot drift
    want = ({cond: condition_flags(cond, pins) for cond in CONDITIONS}
            if "long_prefill_token_threshold" in adv else None)
    if want is None or adv.get("conditions") != want:
        problems.append(f"conditions {adv.get('conditions')!r} != adv_plan.condition_flags {want!r} "
                        f"(024: each condition's argv delta)")
    return problems


# ----------------------------------------------------------------------- plan --

def block_order(seed: int, block: int) -> list:
    digest = hashlib.sha256(f"{seed}:advanced:block:{block}".encode()).hexdigest()
    order = list(CELLS)
    random.Random(int(digest, 16)).shuffle(order)
    return order


def build_plan(pins: dict, *, session: dict, uniform_plan: dict, rows: list, pins_sha256: str) -> dict:
    """The 60-boot plan from the uniform Core B session (its session.json, plan.json and rows)."""
    problems = pins_problems(pins)
    if problems:
        raise PlanError("pins [advanced]: " + "; ".join(problems))
    adv, sb = pins["advanced"], pins["stage_b"]
    sid = session.get("session_id")
    if session.get("formal") is not True or session.get("env_valid") is not True:  # as core_b_plan mixed (018)
        raise PlanError(f"uniform session {sid} is not formal and env-valid (formal={session.get('formal')!r}, "
                        f"env_valid={session.get('env_valid')!r}); C_knee comes only from a valid formal session")
    if sid != adv["derived_from"]:
        raise PlanError(f"uniform session {sid!r} != pinned derived_from {adv['derived_from']!r} (024)")
    c = {}
    for L in LEVELS:
        grid = (uniform_plan.get("grids") or {}).get(str(L))
        c[L] = ba.knee(rows, L, grid)["c_knee"] if grid else None
        if c[L] != int(adv["c"][str(L)]):
            raise PlanError(f"recomputed C_knee({L}) = {c[L]!r} != pinned {adv['c'][str(L)]} (024)")
    boots = []
    for r in range(1, RESTARTS + 1):
        for cond, L, mix in block_order(adv["seed"], r):
            C = c[L]
            boots.append({"block": r, "cond": cond, "L": L, "mix": mix, "restart": r, "C": C,
                          "N": bp.num_prompts(C), "warmup": bp.num_warmups(C),
                          "timeout_s": adv["timeouts_s"][f"{mix}_{L}"], "prefix": request_id_prefix(cond, L, mix, r)})
    return {"schema_version": SCHEMA_VERSION, "seed": adv["seed"], "derived_from": sid,
            "c": {str(L): C for L, C in c.items()}, "conditions": list(CONDITIONS), "boots": boots,
            "expected_blocks": {cond: adv["num_gpu_blocks_k3"] if cond == "K3" else adv["num_gpu_blocks_k4"]
                                for cond in CONDITIONS},
            "timeouts_s": dict(adv["timeouts_s"]), "pins_sha256": pins_sha256,
            "gpu_memory_utilization": sb["gpu_memory_utilization"],
            "kv_cache_memory_bytes": sb["kv_cache_memory_bytes"], "block_size": sb["block_size"]}


def load_pins_bytes() -> bytes:
    return PINS_PATH.read_bytes()


def load_uniform_session(path) -> tuple:
    import json

    d = Path(path)
    rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()]
    return json.loads((d / "session.json").read_text()), json.loads((d / "plan.json").read_text()), rows


def main(argv: list | None = None) -> int:
    import argparse
    import json
    import tomllib

    parser = argparse.ArgumentParser(description="Advanced plan (decision 024)")
    parser.add_argument("--uniform-session", required=True, help="the formal Core B uniform session directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    raw = load_pins_bytes()
    session, uniform_plan, rows = load_uniform_session(args.uniform_session)
    try:
        plan = build_plan(tomllib.loads(raw.decode()), session=session, uniform_plan=uniform_plan, rows=rows,
                          pins_sha256=hashlib.sha256(raw).hexdigest())
    except PlanError as e:
        parser.error(str(e))
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"boots": len(plan["boots"]), "c": plan["c"], "expected_blocks": plan["expected_blocks"],
                      "pins_sha256": plan["pins_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
