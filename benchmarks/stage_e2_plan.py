#!/usr/bin/env python3
"""Stage E2 plans (decision 041 v3 §2): why T fails at 2K C 25. Every server is 024's observed B0 (Core B's argv,
--kv-cache-memory-bytes, the OTLP and iteration-log flags) plus its condition; the client is Core B's `vllm bench
serve` plus a request-id prefix; one bench per boot (024).

Conditions:
- B0 and T: exactly adv_plan.server_argv_for("B0" / "T") (T = 512, pins [advanced]).
- T256: B0 + --long-prefill-token-threshold 256.
- K25 / TK25: B0 / T with --kv-cache-memory-bytes 2.5 GiB (155 blocks).
- PC: B0 with its single --no-enable-prefix-caching replaced by --enable-prefix-caching (041 §6 A3).
Cells (L 2048, 3 restarts, uniform and mixed unless noted): B0@25, T@25, T256@25, B0@8, T@8, B0@8k (K25), T@8k (TK25),
PC-off / PC-on (B0 / PC at C 25 on the shared-prefix dataset only). 48 boots, each restart block shuffled with
Random(sha256("<seed>:stage_e:E2:block:<r>")).
Timeouts: capacity_plan.bench_timeout_s(2048, mix, C, TAU1_S) (shared as uniform), doubled for T, T256 and TK25 (039).
Mixed: the pinned data/core_b/mixed-2048.jsonl (--disable-shuffle; C 8 reads its first 200 rows). Shared: the pinned
shared-prefix dataset (stage_e2_datasets.py).
Formal plans need freeze.e2, every expected block count measured and the passed smoke (mode smoke, smoke.ok, this
harness, a WSL CUDA host).

Usage:
    python benchmarks/stage_e2_plan.py --mode smoke --out plan.json
    python benchmarks/stage_e2_plan.py --mode formal --out plan.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import adv_plan as ap  # noqa: E402
import capacity_plan as cp  # noqa: E402
import core_b_plan as bp  # noqa: E402
import coreb_smoke as smoke  # noqa: E402
import run_adv as ra  # noqa: E402

SCHEMA_VERSION = 1
STAGE = "E2"
PORT = 8000
CONFIG = ROOT / "configs" / "stage_e2.toml"
RESULTS = ("results", "stage_e")
CONDITIONS = ("B0", "T", "T256", "K25", "TK25", "PC")
CHUNKED = ("T", "T256", "TK25")
MIX2 = ("uniform", "mixed")
# (cell, condition, C, mixes) of 041 v3 §2.5
CELLS = (("B0@25", "B0", 25, MIX2), ("T@25", "T", 25, MIX2), ("T256@25", "T256", 25, MIX2),
         ("B0@8", "B0", 8, MIX2), ("T@8", "T", 8, MIX2), ("B0@8k", "K25", 8, MIX2), ("T@8k", "TK25", 8, MIX2),
         ("PC-off", "B0", 25, ("shared",)), ("PC-on", "PC", 25, ("shared",)))
# the smoke: one boot per condition and mix type, each at its formal C (041 v3 §2.10)
SMOKE = (("B0@8", "uniform"), ("B0@25", "mixed"), ("T@25", "mixed"), ("T256@25", "mixed"), ("B0@8k", "mixed"),
         ("T@8k", "mixed"), ("PC-off", "shared"), ("PC-on", "shared"))
SMOKE_N = 20
WATCHED_KEYS = ra.WATCHED_KEYS + ("enable_prefix_caching",)
HARNESS_FILES = ("benchmarks/stage_e2_plan.py", "benchmarks/run_stage_e2.py", "benchmarks/stage_e2_analysis.py",
                 "benchmarks/stage_e2_datasets.py")
CONFIG_KEYS = {"schema_version": None, "freeze": {"e2", "smoke_e2"},
               "e2": {"seed", "L", "restarts", "kv25_bytes", "t256", "expected_blocks", "m_effective", "beta_gap",
                      "specificity", "cap_hours", "shared_prefix"}}


class PlanError(bp.PlanError):
    pass


def harness_sha256(root=ROOT) -> str:
    h = hashlib.sha256()
    for rel in HARNESS_FILES:
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------- config

def config_problems(cfg: dict, pins: dict) -> list:
    if set(cfg) != set(CONFIG_KEYS):
        return [f"top-level keys {sorted(cfg)} are not {sorted(CONFIG_KEYS)}"]
    problems = [f"[{t}] keys {sorted(cfg[t])} are not {sorted(k)}" for t, k in CONFIG_KEYS.items()
                if k is not None and set(cfg[t]) != k]
    if problems:
        return problems
    e, sb = cfg["e2"], pins["stage_b"]
    if (e["seed"], e["L"], e["restarts"], e["t256"]) != (bp.SEED, 2048, ap.RESTARTS, 256):
        problems.append("e2 seed / L / restarts / t256 are not 041 v3 §2's (20261008, 2048, 3, 256)")
    lo, hi = ap.blocks_bounds(e["kv25_bytes"], ref_bytes=sb["kv_cache_memory_bytes"], ref_blocks=sb["num_gpu_blocks"])
    if e["kv25_bytes"] != 10 * ap.KV_STEP or (lo, hi) != (155, 155):
        problems.append(f"e2.kv25_bytes {e['kv25_bytes']} is not 2.5 GiB with exactly 155 blocks ({lo}, {hi})")
    want = {"B0": sb["num_gpu_blocks"], "T": sb["num_gpu_blocks"], "T256": sb["num_gpu_blocks"], "K25": 155,
            "TK25": 155}
    eb = e["expected_blocks"]
    if set(eb) != set(CONDITIONS) or any(eb[k] != v for k, v in want.items()) or not isinstance(eb["PC"], int):
        problems.append(f"e2.expected_blocks {eb} are not {want} plus PC (the smoke measures PC)")
    if (e["m_effective"], e["beta_gap"], e["specificity"]) != (0.20, 0.20, 0.10):
        problems.append("e2.m_effective / beta_gap / specificity are not 041 v3 §2.2's (0.20, 0.20, 0.10)")
    sp = e["shared_prefix"]
    if (sp.get("total"), sp.get("min_shared")) != (2048 - bp.N_GEN, 3 * sb["block_size"]):
        problems.append("e2.shared_prefix total / min_shared are not 1,920 / 3 blocks")
    f = cfg["freeze"]
    if not isinstance(f["e2"], bool) or not isinstance(f["smoke_e2"], str):
        problems.append("freeze.e2 must be a bool and freeze.smoke_e2 a string")
    elif f["e2"]:
        if not f["smoke_e2"]:
            problems.append("freeze.smoke_e2 must name the passed smoke once freeze.e2 is true")
        if not eb["PC"]:
            problems.append("e2.expected_blocks.PC must be measured (smoke) before the freeze")
        if len(str(sp.get("sha256") or "")) != 64 or (sp.get("common_prefix") or 0) < sp["min_shared"]:
            problems.append("e2.shared_prefix sha256 / common_prefix must be pinned before the freeze")
    return problems


# ---------------------------------------------------------------- conditions

def kv_bytes(cond: str, pins: dict, cfg: dict) -> int:
    return cfg["e2"]["kv25_bytes"] if cond in ("K25", "TK25") else pins["stage_b"]["kv_cache_memory_bytes"]


def threshold(cond: str, pins: dict, cfg: dict) -> int | None:
    if cond in ("T", "TK25"):
        return pins["advanced"]["long_prefill_token_threshold"]
    return cfg["e2"]["t256"] if cond == "T256" else None


def strip_env(base_env) -> dict:
    """The caller's env without any inherited PYTORCH_* (core_b_plan.server_env drops only VLLM_ / CUDA_ / GGML_)."""
    return {k: v for k, v in dict(base_env or {}).items() if not k.startswith("PYTORCH_")}


def server_argv_for(cond: str, pins: dict, cfg: dict, *, gdn_prefill_backend: str, gpu_memory_utilization: str,
                    port: int = PORT, vllm_bin: str = "{vllm}", base_env=None) -> tuple:
    if cond not in CONDITIONS:
        raise PlanError(f"unknown condition {cond!r}; 041 v3 defines {CONDITIONS}")
    env_in = strip_env(base_env)
    argv, _ = bp.server_argv(pins, gdn_prefill_backend=gdn_prefill_backend,
                             gpu_memory_utilization=gpu_memory_utilization, port=port, vllm_bin=vllm_bin,
                             base_env=env_in, kv_cache_memory_bytes=kv_bytes(cond, pins, cfg))
    argv = argv + ap.observability_flags(pins)
    t = threshold(cond, pins, cfg)
    if t is not None:
        argv += ["--long-prefill-token-threshold", str(t)]
    if cond == "PC":
        if argv.count("--no-enable-prefix-caching") != 1:
            raise PlanError("PC: Core B's argv must carry exactly one --no-enable-prefix-caching to replace")
        argv[argv.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
    return argv, ap.server_env_for(env_in, pins)


def expected_nondefault(cond: str, pins: dict, cfg: dict) -> dict:
    """The watched non-default args a condition's server must log, and only these. PC may log
    enable_prefix_caching True or leave it out (vLLM's default); /metrics' cache_config_info decides it."""
    want = {"max_num_batched_tokens": bp.MAX_NUM_BATCHED_TOKENS,
            "otlp_traces_endpoint": pins["advanced"]["otlp_endpoint"], "enable_logging_iteration_details": True,
            "kv_cache_memory_bytes": kv_bytes(cond, pins, cfg), "enable_prefix_caching": cond == "PC"}
    t = threshold(cond, pins, cfg)
    if t is not None:
        want["long_prefill_token_threshold"] = t
    return want


def parse_watched(log_text: str) -> dict | None:
    m = ra._RE_NONDEFAULT.search(log_text)
    if not m:
        return None
    out = {}
    for key in WATCHED_KEYS:
        v = re.search(ra._RE_VALUE.format(key=key), m.group(1))
        if v:
            out[key] = ast.literal_eval(v.group(1))
    return out


def manipulation_problems(log_text: str, cond: str, pins: dict, cfg: dict) -> tuple:
    """(log facts, watched non-default args, problems): FLASH_ATTN and exactly the condition's non-default args."""
    facts = smoke.parse_server_log(log_text)
    problems = []
    if facts.get("attention_backend") != "FLASH_ATTN":
        problems.append(f"attention backend {facts.get('attention_backend')!r} != FLASH_ATTN (003, §4)")
    got = parse_watched(log_text)
    if got is None:
        return facts, None, problems + ["no non-default args line in the server log (024 manipulation check)"]
    want = expected_nondefault(cond, pins, cfg)
    for k in WATCHED_KEYS:
        if cond == "PC" and k == "enable_prefix_caching":
            if got.get(k, True) is not True:
                problems.append(f"non-default args enable_prefix_caching = {got[k]!r}; PC needs True or the default")
        elif k in want and (k not in got or type(got[k]) is not type(want[k]) or got[k] != want[k]):
            problems.append(f"non-default args {k} = {got.get(k, '<absent>')!r}; condition {cond} needs {want[k]!r}")
        elif k not in want and k in got:
            problems.append(f"non-default args carry {k} = {got[k]!r}, which condition {cond} must not set")
    return facts, got, problems


_RE_CACHE_INFO = re.compile(r"vllm:cache_config_info\{([^}]*)\}")


def prefix_caching_label(metrics_text: str):
    """enable_prefix_caching of /metrics' vllm:cache_config_info (True / False), or None when absent."""
    m = _RE_CACHE_INFO.search(metrics_text or "")
    if not m:
        return None
    v = re.search(r'enable_prefix_caching="(\w+)"', m.group(1))
    return None if not v else v.group(1) == "True"


# ---------------------------------------------------------------- clients

def request_id_prefix(cell: str, mix: str, restart: int) -> str:
    return f"e2-{cell.replace('@', '-c')}-{mix}-r{restart}-"


def bench_argv_for(pins: dict, *, mix: str, C: int, prefix: str, result_dir: str, result_filename: str,
                   datasets: dict, vllm_bin: str = "{vllm}") -> list:
    """Core B's client (uniform / mixed); shared: the same flags with the pinned shared-prefix custom dataset."""
    if mix in MIX2:
        argv = bp.bench_argv(pins, L=2048, C=C, mix=mix, result_dir=result_dir, result_filename=result_filename,
                             port=PORT, mixed_dataset=datasets.get("mixed:2048") if mix == "mixed" else None,
                             vllm_bin=vllm_bin)
    elif mix == "shared":
        path = datasets.get("shared:2048")
        if not path:
            raise PlanError("the PC cells need the pinned shared-prefix dataset (041 v3 §2.8)")
        argv = bp.bench_argv(pins, L=2048, C=C, mix="uniform", result_dir=result_dir,
                             result_filename=result_filename, port=PORT, vllm_bin=vllm_bin)
        i = argv.index("--dataset-name")
        argv = argv[:i] + ["--dataset-name", "custom", "--dataset-path", path, "--skip-chat-template",
                           "--disable-shuffle", "--custom-output-len", str(bp.N_GEN)] + argv[i + 8:]
    else:
        raise PlanError(f"unknown mix {mix!r}")
    return argv + ["--request-id-prefix", prefix]


def timeout_s(cond: str, mix: str, C: int) -> int:
    t = cp.bench_timeout_s(2048, "uniform" if mix == "shared" else mix, C, cp.TAU1_S)
    return 2 * t if cond in CHUNKED else t


# ---------------------------------------------------------------- plans

def _cell(name: str) -> tuple:
    return next(c for c in CELLS if c[0] == name)


def _boot(cell: str, mix: str, restart: int, block: int, cfg: dict) -> dict:
    _, cond, C, _ = _cell(cell)
    return {"block": block, "cell": cell, "cond": cond, "C": C, "L": 2048, "mix": mix, "restart": restart,
            "N": bp.num_prompts(C), "warmup": bp.num_warmups(C), "timeout_s": timeout_s(cond, mix, C),
            "prefix": request_id_prefix(cell, mix, restart), "dataset": {"mixed": "mixed:2048",
                                                                        "shared": "shared:2048"}.get(mix)}


def build_plan(pins: dict, cfg: dict, *, mode: str, pins_sha256: str | None = None,
               stage_e2_sha256: str | None = None, root=ROOT) -> dict:
    """mode "smoke" (SMOKE, one boot each, built under run_capacity.mini_num_prompts(SMOKE_N) by the runner) or
    "formal" (every cell x mix x 3 restarts; needs freeze.e2 and the passed smoke under root)."""
    if mode not in ("smoke", "formal"):
        raise PlanError(f"unknown mode {mode!r}")
    problems = config_problems(cfg, pins)
    if problems:
        raise PlanError("configs/stage_e2.toml: " + "; ".join(problems))
    e = cfg["e2"]
    if mode == "formal":
        if cfg["freeze"]["e2"] is not True:
            raise PlanError("formal E2 plans need freeze.e2 = true (041 凍結)")
        evidence = smoke_problems(Path(root).joinpath(*RESULTS, cfg["freeze"]["smoke_e2"]))
        if evidence:
            raise PlanError("formal E2 plans need the passed smoke first: " + "; ".join(evidence))
        boots = []
        for r in range(1, e["restarts"] + 1):
            block = [(cell, mix) for cell, _, _, mixes in CELLS for mix in mixes]
            random.Random(int(hashlib.sha256(f"{e['seed']}:stage_e:E2:block:{r}".encode()).hexdigest(), 16)).shuffle(
                block)
            boots += [_boot(cell, mix, r, r, cfg) for cell, mix in block]
    else:
        boots = [_boot(cell, mix, 1, 1, cfg) for cell, mix in SMOKE]
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "mode": mode, "decision": "041 v3 §2", "seed": e["seed"],
            "derived_from": ap.DERIVED_FROM, "pins_sha256": pins_sha256, "stage_e2_sha256": stage_e2_sha256,
            "harness_sha256": harness_sha256(), "expected_blocks": dict(e["expected_blocks"]),
            "block_size": pins["stage_b"]["block_size"],
            "thresholds": {c: threshold(c, pins, cfg) for c in CONDITIONS if threshold(c, pins, cfg)},
            "kv_cache_memory_bytes": {c: kv_bytes(c, pins, cfg) for c in CONDITIONS},
            "datasets_sha256": {"mixed:2048": pins["core_b"]["mix_dataset_sha256"]["2048"],
                                "shared:2048": e["shared_prefix"]["sha256"] or None},
            "cap_s": e["cap_hours"] * 3600 if mode == "formal" else None, "boots": boots}


def host_problems(host: dict | None) -> list:
    want = {"system": "Linux", "wsl": True, "nvidia": True}
    return [] if host == want else [f"ran on host {host!r}, not a WSL CUDA host"]


def smoke_problems(path) -> list:
    path = Path(path)
    try:
        s = json.loads((path / "session.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"{path.name}: no readable session.json ({exc})"]
    problems = []
    if (s.get("stage"), s.get("mode"), (s.get("smoke") or {}).get("ok")) != (STAGE, "smoke", True):
        problems.append(f"{path.name}: not a passed E2 smoke")
    problems += [f"{path.name}: {p}" for p in host_problems(s.get("host"))]
    if s.get("harness_sha256") != harness_sha256():
        problems.append(f"{path.name}: ran another harness (harness_sha256)")
    return problems


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    import run_capacity as rc
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=["smoke", "formal"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    raw, cfg_raw = (ROOT / "configs" / "pins.toml").read_bytes(), CONFIG.read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(cfg_raw.decode())
    try:
        if args.mode == "smoke":
            with rc.mini_num_prompts(SMOKE_N):
                plan = build_plan(pins, cfg, mode="smoke", pins_sha256=hashlib.sha256(raw).hexdigest(),
                                  stage_e2_sha256=hashlib.sha256(cfg_raw).hexdigest())
        else:
            plan = build_plan(pins, cfg, mode="formal", pins_sha256=hashlib.sha256(raw).hexdigest(),
                              stage_e2_sha256=hashlib.sha256(cfg_raw).hexdigest())
    except PlanError as exc:
        print(f"stage_e2_plan: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"boots": len(plan["boots"]), "mode": plan["mode"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
