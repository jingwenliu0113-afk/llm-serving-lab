#!/usr/bin/env python3
"""Stage E1-P plans (decision 041 v3 §1.6-1.7): the seed-swap probe. Is a position-0 F spike locked to the prompt's
content or to its position (time) in the closed-loop bench? Exploratory (formal plans and sessions, descriptive
reading only).

Server: 024's observed B0, exactly adv_plan.server_argv_for("B0") (Core B's argv, --kv-cache-memory-bytes, the OTLP
and iteration-log flags; F needs the spans). It is not C0's flag-off server: no comparison with C0.
Client: core_b_plan.bench_argv at L 8192 uniform, C 8 (Core's N 200, 8 warmups) with --seed set to the bench's seed
and a request-id prefix e1p-b<boot>-n<bench>-; the smoke changes only --num-prompts (16, as C0's).
Formal: 6 boots x 4 benches; odd boots A B B A, even boots B A A B (seed A 20261008, seed B 20261023), so each seed
has 12 benches, three at each position. Smoke: 1 boot, benches A then B.
Stop rules (028 / C0, run_c0_probe's constants): 031 idle window and spill guard before every boot, 720 s bench
timeout (pins [advanced]), a boot fails when its server does not start or >= 2 of its benches are invalid, the 2nd
failed boot stops the session, no new boot when elapsed + 30 min > cap, no bench whose timeout could pass the cap,
GPU >= 88 C stops at once. The cap is 3.0 h of holding the lock with the smoke included: a formal plan's cap_s is
3.0 h minus the passed smoke's elapsed_s (041 v3 §1.6, §4).
Every plan needs the passed seed check (results/stage_e/e1p-seed-check/seed_check.json from
stage_e1p_seed_check.py: vLLM's own sampler, under the bench argv built now, gave both seeds' prompts at the formal N
and the smoke's 8,064 server-side tokens and A != B) and E1-D's frozen stage_e1 (config and code pin). Formal plans
also need freeze.e1p and the passed smoke (freeze.smoke_e1p, a session id): finished, smoke.ok, env valid, a WSL CUDA
host, the pinned vLLM, these pins (when given), this harness, its plan exactly the smoke plan under this config. A plan records stage_e1p_sha256 (the config without [freeze]),
stage_e1_sha256 (E1-D's frozen config, whose links it uses) and seed_check_sha256.

Usage:
    python benchmarks/stage_e1p_plan.py --mode smoke --out plan.json
    python benchmarks/stage_e1p_plan.py --mode formal --out plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
import adv_plan as ap  # noqa: E402
import core_b_plan as bp  # noqa: E402
import run_adv as ra  # noqa: E402
import run_c0_probe as rcp  # noqa: E402
import stage_e1 as se1  # noqa: E402
from run_core_a import sha256_json  # noqa: E402

SCHEMA_VERSION = 1
STAGE = "E1P"
PORT = 8000
L, MIX, C = rcp.L, rcp.MIX, rcp.C
CONFIG = ROOT / "configs" / "stage_e1p.toml"
RESULTS = ("results", "stage_e")
# 041 v3 §0: imported modules without a guard test belong to the part's harness (stage_e1 is pinned by its own test)
HARNESS_FILES = ("benchmarks/stage_e1p_plan.py", "benchmarks/run_stage_e1p.py", "benchmarks/stage_e1p_analysis.py",
                 "benchmarks/run_c0_probe.py", "benchmarks/c0_analysis.py")
CONFIG_KEYS = {"schema_version": None, "freeze": {"e1p", "smoke_e1p"},
               "e1p": {"seeds", "boots", "order_odd", "order_even", "smoke_order", "smoke_N", "cap_hours",
                       "boot_margin_s", "temp_stop_c", "max_failed_boots", "f_spike_ms", "repeat_min", "min_spikes",
                       "a_prior", "j_content_max", "j_position_min"}}
E1P_VALUES = {"seeds": {"A": bp.SEED, "B": 20261023}, "boots": rcp.BOOTS, "order_odd": ["A", "B", "B", "A"],
              "order_even": ["B", "A", "A", "B"], "smoke_order": ["A", "B"], "smoke_N": rcp.SMOKE_N,
              "cap_hours": rcp.CAP_HOURS, "boot_margin_s": rcp.BOOT_MARGIN_S, "temp_stop_c": rcp.TEMP_STOP_C,
              "max_failed_boots": rcp.MAX_FAILED_BOOTS, "f_spike_ms": se1.E1_VALUES["f_spike_ms"], "repeat_min": 3,
              "min_spikes": 5, "a_prior": [32, 64, 88, 128, 176, 192], "j_content_max": 0.2, "j_position_min": 0.5}
# a stop of the environment, not of the probe (041 v3 §1.6): such a session is void, like an env-invalid one
ENV_STOPS = ("lock", "gpu_not_idle", "gpu_check", "spill_unreadable", "telemetry")
SEED_CHECK = ("results", "stage_e", "e1p-seed-check", "seed_check.json")
SEED_CHECK_SCRIPT = "benchmarks/stage_e1p_seed_check.py"
_RE_SESSION_ID = re.compile(r"(?!\.+$)[A-Za-z0-9._-]+")


class PlanError(bp.PlanError):
    pass


def harness_sha256(root=ROOT) -> str:
    h = hashlib.sha256()
    for rel in HARNESS_FILES:
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------- config

def _same(a, b) -> bool:
    """Equal with the same types all the way down (True is not 1, 3 is not 3.0)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def config_problems(cfg: dict) -> list:
    if set(cfg) != set(CONFIG_KEYS):
        return [f"top-level keys {sorted(cfg)} are not {sorted(CONFIG_KEYS)}"]
    problems = [f"[{t}] keys {sorted(cfg[t])} are not {sorted(k)}" for t, k in CONFIG_KEYS.items()
                if k is not None and set(cfg[t]) != k]
    if problems:
        return problems
    problems = [f"e1p.{k} {cfg['e1p'][k]!r} is not 041 v3 §1.6's {v!r}" for k, v in E1P_VALUES.items()
                if not _same(cfg["e1p"][k], v)]
    f = cfg["freeze"]
    if not isinstance(f["e1p"], bool) or not isinstance(f["smoke_e1p"], str):
        problems.append("freeze.e1p must be a bool and freeze.smoke_e1p a string")
    elif f["e1p"] and not _RE_SESSION_ID.fullmatch(f["smoke_e1p"]):
        problems.append("freeze.smoke_e1p must name the passed smoke (a session id under results/stage_e) once "
                        "freeze.e1p is true")
    return problems


def config_sha256(cfg: dict) -> str:
    """configs/stage_e1p.toml's content without [freeze] (comments and the freeze never change a plan)."""
    return sha256_json({k: v for k, v in cfg.items() if k != "freeze"})


def e1_config(root=ROOT) -> dict:
    """E1-D's frozen configs/stage_e1.toml (L1-L5's thresholds), checked and read only, with stage_e1.py (whose
    functions E1-P runs) still the frozen harness it pins."""
    cfg = tomllib.loads((Path(root) / "configs" / "stage_e1.toml").read_text())
    problems = se1.config_problems(cfg)
    if problems or cfg["freeze"]["e1"] is not True:
        raise PlanError("configs/stage_e1.toml is not E1-D's frozen config: " + "; ".join(problems or ["not frozen"]))
    if se1.harness_sha256(root) != cfg["freeze"]["e1_harness_sha256"]:
        raise PlanError("benchmarks/stage_e1.py / c0_analysis.py are not the harness E1-D froze (e1_harness_sha256)")
    return cfg


# ---------------------------------------------------------------- the seed check (041 v3 §1.7, before the smoke)

def seed_check_sha256(root=ROOT) -> str:
    return hashlib.sha256((Path(root) / SEED_CHECK_SCRIPT).read_bytes()).hexdigest()


def seed_check_ns(cfg: dict) -> tuple:
    """The N the seed check covers: the formal benches' and the smoke's (vLLM's padding draws depend on N, 013 §2)."""
    return bp.num_prompts(C), cfg["e1p"]["smoke_N"]


def seed_check_argv(pins: dict, label: str, seed: int, N: int) -> list:
    """The bench argv the seed check samples for one seed and N (bench_argv_for; the prefix and paths sample
    nothing)."""
    return bench_argv_for(pins, seed=seed, N=N, prefix=f"e1p-seedcheck-{label}-", result_dir="/unused",
                          result_filename="unused.json", vllm_bin="vllm")


def seed_check_problems(pins: dict, cfg: dict, root=ROOT) -> list:
    """The committed seed check (stage_e1p_seed_check.py, CPU only on WSL): vLLM's own sampler, for both seeds at the
    formal N and the smoke's, under the bench argv the harness builds now, gave every prompt 8,064 server-side tokens
    and different prompts for A and B. The verdict is re-derived from the per-seed counts, never taken from the
    record's ok alone. Without it no plan is built: one bad seed-B prompt among the formal 200 would invalidate every
    seed-B bench and stop the only formal session, and identical prompts would read as position-locked."""
    rec = _read_json(Path(root).joinpath(*SEED_CHECK))
    if not isinstance(rec, dict):
        return [f"no readable seed check at {Path(*SEED_CHECK)} (run benchmarks/stage_e1p_seed_check.py on WSL)"]
    problems = []
    if rec.get("ok") is not True:
        problems.append(f"the seed check did not pass ({rec.get('problems')})")
    if rec.get("vllm_version") != pins["vllm"]["version"]:
        problems.append(f"the seed check ran vLLM {rec.get('vllm_version')!r}, not {pins['vllm']['version']}")
    if rec.get("script_sha256") != seed_check_sha256():
        problems.append("the seed check was made by another stage_e1p_seed_check.py (script_sha256)")
    if rec.get("input_len") != L - bp.N_GEN:
        problems.append(f"the seed check's input_len {rec.get('input_len')!r} is not {L - bp.N_GEN}")
    checks = rec.get("checks")
    by_n = {c.get("N"): c for c in checks if isinstance(c, dict)} if isinstance(checks, list) else {}
    argv = rec.get("argv") if isinstance(rec.get("argv"), dict) else {}
    for N in seed_check_ns(cfg):
        c = by_n.get(N)
        seeds = c.get("seeds") if isinstance(c, dict) else None
        if not isinstance(seeds, dict) or set(seeds) != {"A", "B"} or not all(isinstance(v, dict)
                                                                                for v in seeds.values()):
            problems.append(f"the seed check has no A / B record at N {N}")
            continue
        for x, seed in cfg["e1p"]["seeds"].items():
            got = seeds[x]
            if (got.get("seed"), got.get("n"), got.get("server_tokens_ok")) != (seed, N, N):
                problems.append(f"the seed check at N {N}, seed {x}: seed {got.get('seed')!r}, "
                                f"{got.get('server_tokens_ok')!r} of {got.get('n')!r} prompts at {L - bp.N_GEN} "
                                f"tokens (needs seed {seed}, all {N})")
            if (argv.get(str(N)) or {}).get(x) != seed_check_argv(pins, x, seed, N):
                problems.append(f"the seed check at N {N}, seed {x} sampled another bench argv than the harness "
                                "builds now")
        if seeds["A"].get("prompts_sha256") in (None, seeds["B"].get("prompts_sha256")):
            problems.append(f"the seed check at N {N}: seeds A and B drew the same prompts")
    return problems


# ---------------------------------------------------------------- server and client

def strip_env(base_env) -> dict:
    """The caller's env without any inherited PYTORCH_* (core_b_plan.server_env drops only VLLM_ / CUDA_ / GGML_)."""
    return {k: v for k, v in dict(base_env or {}).items() if not k.startswith("PYTORCH_")}


def server_argv_for(pins: dict, *, vllm_bin: str = "{vllm}", base_env=None) -> tuple:
    """024's observed B0, exactly Advanced's (adv_plan.server_argv_for)."""
    sb = pins["stage_b"]
    return ap.server_argv_for("B0", pins, gdn_prefill_backend=sb["gdn_prefill_backend"],
                              gpu_memory_utilization=sb["gpu_memory_utilization"], port=PORT, vllm_bin=vllm_bin,
                              base_env=strip_env(base_env))


def request_id_prefix(boot: int, bench: int) -> str:
    return f"e1p-b{boot}-n{bench}-"


def bench_argv_for(pins: dict, *, seed: int, N: int, prefix: str, result_dir: str, result_filename: str,
                   vllm_bin: str = "{vllm}") -> list:
    """Core B's client at 8192 uniform, C 8 with this bench's --seed, N and request-id prefix (nothing else changes)."""
    argv = bp.bench_argv(pins, L=L, C=C, mix=MIX, result_dir=result_dir, result_filename=result_filename, port=PORT,
                         vllm_bin=vllm_bin)
    if argv.count("--seed") != 1 or argv.count("--num-prompts") != 1:
        raise PlanError("Core B's client must carry exactly one --seed and one --num-prompts")
    argv[argv.index("--seed") + 1] = str(seed)
    argv[argv.index("--num-prompts") + 1] = str(N)
    return argv + ["--request-id-prefix", prefix]


# ---------------------------------------------------------------- plans

def _benches(boot: int, order: list, seeds: dict, N: int) -> list:
    return [{"boot": boot, "bench": j, "tag": rcp.bench_tag(boot, j), "seed_label": x, "seed": seeds[x], "N": N,
             "prefix": request_id_prefix(boot, j)} for j, x in enumerate(order, 1)]


def build_plan(pins: dict, cfg: dict, *, mode: str, pins_sha256: str | None = None, root=ROOT) -> dict:
    """mode "smoke" (1 boot, smoke_order at smoke_N) or "formal" (6 boots; needs freeze.e1p and the passed smoke
    under root, whose elapsed_s comes off the 3.0 h cap)."""
    if mode not in ("smoke", "formal"):
        raise PlanError(f"unknown mode {mode!r}")
    problems = config_problems(cfg)
    if problems:
        raise PlanError("configs/stage_e1p.toml: " + "; ".join(problems))
    problems = seed_check_problems(pins, cfg, root)
    if problems:
        raise PlanError("E1-P plans need the passed seed check first (041 v3 §1.7): " + "; ".join(problems))
    e, sb, adv = cfg["e1p"], pins["stage_b"], pins["advanced"]
    cap_s, smoke_ref = e["cap_hours"] * 3600, None
    if mode == "formal":
        if cfg["freeze"]["e1p"] is not True:
            raise PlanError("formal E1-P plans need freeze.e1p = true (041 v3 §1.7 凍結)")
        path = Path(root).joinpath(*RESULTS, cfg["freeze"]["smoke_e1p"])
        evidence = smoke_problems(path, pins, cfg, root=root, pins_sha256=pins_sha256)
        if evidence:
            raise PlanError("formal E1-P plans need the passed smoke first: " + "; ".join(evidence))
        elapsed = _read_json(path / "session.json")["elapsed_s"]
        smoke_ref = {"session": path.name, "elapsed_s": elapsed}
        cap_s = int(cap_s - elapsed)                                   # 041 v3 §1.6: the cap includes the smoke
        boots = [{"boot": b, "benches": _benches(b, e["order_odd"] if b % 2 else e["order_even"], e["seeds"],
                                                  bp.num_prompts(C))} for b in range(1, e["boots"] + 1)]
    else:
        boots = [{"boot": 1, "benches": _benches(1, e["smoke_order"], e["seeds"], e["smoke_N"])}]
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "mode": mode, "decision": "041 v3 §1.6",
            "derived_from": ap.DERIVED_FROM, "L": L, "mix": MIX, "C": C, "warmup": bp.num_warmups(C),
            "seeds": dict(e["seeds"]), "pins_sha256": pins_sha256, "stage_e1p_sha256": config_sha256(cfg),
            "stage_e1_sha256": sha256_json(e1_config()), "harness_sha256": harness_sha256(),
            "seed_check_sha256": hashlib.sha256(Path(root).joinpath(*SEED_CHECK).read_bytes()).hexdigest(),
            "kv_cache_memory_bytes": sb["kv_cache_memory_bytes"], "expected_blocks": sb["num_gpu_blocks"],
            "block_size": sb["block_size"], "gpu_memory_utilization": sb["gpu_memory_utilization"],
            "gdn_prefill_backend": sb["gdn_prefill_backend"], "timeout_s": adv["timeouts_s"][f"{MIX}_{L}"],
            "span_wait_s": ra.SPAN_WAIT_S, "cap_s": cap_s, "boot_margin_s": e["boot_margin_s"],
            "temp_stop_c": e["temp_stop_c"], "max_failed_boots": e["max_failed_boots"], "smoke": smoke_ref,
            "boots": boots}


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _without(plan: dict, *keys) -> dict:
    return {k: v for k, v in plan.items() if k not in keys}


def host_problems(host: dict | None) -> list:
    want = {"system": "Linux", "wsl": True, "nvidia": True}
    return [] if host == want else [f"ran on host {host!r}, not a WSL CUDA host"]


def smoke_problems(path, pins: dict, cfg: dict, *, root=ROOT, pins_sha256: str | None = None) -> list:
    """041 v3 §1.7's freeze evidence: the smoke at path is a finished, passed run of this design and harness (and,
    when pins_sha256 is given, of these pins)."""
    path = Path(path)
    s, plan = _read_json(path / "session.json"), _read_json(path / "plan.json")
    if not isinstance(s, dict) or not isinstance(plan, dict):
        return [f"{path.name}: no readable session.json / plan.json"]
    problems = []
    if pins_sha256 is not None and s.get("pins_sha256") != pins_sha256:
        problems.append(f"{path.name}: ran under another configs/pins.toml (pins_sha256)")
    smoke = s.get("smoke") or {}
    if (s.get("stage"), s.get("mode"), s.get("status"), smoke.get("ok")) != (STAGE, "smoke", "finished", True):
        problems.append(f"{path.name}: not a finished, passed E1-P smoke")
    if s.get("env_valid") is not True:
        problems.append(f"{path.name}: env not valid ({s.get('env_flags')})")
    if s.get("vllm_version") != pins["vllm"]["version"]:
        problems.append(f"{path.name}: ran vLLM {s.get('vllm_version')!r}, not the pinned {pins['vllm']['version']}")
    problems += [f"{path.name}: {p}" for p in host_problems(s.get("host"))]
    if s.get("harness_sha256") != harness_sha256():
        problems.append(f"{path.name}: ran another harness (harness_sha256)")
    if s.get("plan_sha256") != sha256_json(plan):
        problems.append(f"{path.name}: plan.json is not the plan the smoke ran (plan_sha256)")
    elapsed = s.get("elapsed_s")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not 0 < elapsed < cfg["e1p"]["cap_hours"] \
            * 3600:
        problems.append(f"{path.name}: no elapsed_s to take off the cap ({elapsed!r})")
    try:
        want = build_plan(pins, cfg, mode="smoke", root=root)
    except PlanError as exc:
        return problems + [f"{path.name}: no smoke plan under this config ({exc})"]
    if _without(plan, "pins_sha256") != _without(want, "pins_sha256"):
        problems.append(f"{path.name}: its plan is not build_plan's smoke plan under this config")
    return problems


def formal_attempts(root=ROOT) -> list:
    """Every formal E1-P attempt under results/stage_e, oldest first (041 v3 §0: one rerun of a void session, only the
    first valid one analyzed): a directory whose session.json (final or the running stub) or, when that is unreadable,
    plan.json is a formal E1-P one; runs refused before any server started are left out. valid = finished, env valid
    and not stopped by an ENV_STOPS rule (028's other stops end a valid session: it is analyzed as it is, never rerun);
    an unreadable session.json is a void attempt (fail closed)."""
    out = []
    base = Path(root).joinpath(*RESULTS)
    dirs = {f.parent for name in ("session.json", "plan.json") for f in base.rglob(name)} - {base}
    for d in sorted(dirs):
        s = _read_json(d / "session.json")
        if not isinstance(s, dict) and not any(x.name != "plan.json" for x in d.iterdir()):
            continue                                        # a saved plan alone: nothing ran here
        head = s if isinstance(s, dict) else _read_json(d / "plan.json")
        if not isinstance(head, dict) or (head.get("stage"), head.get("mode")) != (STAGE, "formal"):
            continue
        s = s if isinstance(s, dict) else {}
        if s.get("status") == "refused":
            continue
        stop = (s.get("stopped") or {}).get("rule")
        valid = s.get("status") == "finished" and s.get("env_valid") is True and stop not in ENV_STOPS
        out.append({"dir": d, "started_at": s.get("started_at") or "", "valid": valid})
    return sorted(out, key=lambda a: (a["started_at"], str(a["dir"])))


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=["smoke", "formal"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    raw = (ROOT / "configs" / "pins.toml").read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(CONFIG.read_text())
    out = Path(args.out).resolve()
    if ROOT.joinpath(*RESULTS).resolve() in (out, *out.parents):
        print("stage_e1p_plan: write plans outside results/stage_e (a session directory is the runner's)",
              file=sys.stderr)
        return 2
    try:
        plan = build_plan(pins, cfg, mode=args.mode, pins_sha256=hashlib.sha256(raw).hexdigest())
    except PlanError as exc:
        print(f"stage_e1p_plan: {exc}", file=sys.stderr)
        return 2
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"mode": plan["mode"], "boots": len(plan["boots"]), "cap_s": plan["cap_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
