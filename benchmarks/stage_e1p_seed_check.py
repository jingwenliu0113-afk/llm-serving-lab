#!/usr/bin/env python3
"""E1-P's seed check (decision 041 v3 §1.7), before the smoke. CPU only, on WSL with the project venv, while
tools/rtx/gpu_status.sh prints FREE (it reads WSL's disk; run it under flock -n so no GPU session starts meanwhile).
For seed A and seed B, at the formal N (200) and at the smoke's N (16), it parses the bench argv
(stage_e1p_plan.seed_check_argv: bench_argv_for with that seed and N) with vLLM's own `vllm bench serve` parser, seeds
random / numpy as main_async does and draws the prompts with vLLM's get_samples, exactly as the bench will, in the
bench's env (the caller's env without PYTORCH_* or inherited engine variables, refused like a formal session's, plus
core_b_plan.bench_env()).
Per seed and N it records how many prompts the server will count as 8,064 tokens (server_tokens_ok: the prompt
tokenized with special tokens, as vLLM's completions endpoint does; the bench saves the server's usage.prompt_tokens
as input_lens and run_core_b.summarize_result checks them), how many have the sampler's prompt_len 8,064 (descriptive:
the sampler counts without special tokens; for this tokenizer, which adds none, the two agree), the other lengths and
the sha256 of the prompts. ok = at both N every prompt of both seeds counts 8,064 on the server side and the two seeds
drew different prompts. The record goes to results/stage_e/e1p-seed-check/seed_check.json (never overwritten) and is
committed (tools/rtx/fetch_seed_check.sh); stage_e1p_plan builds no plan without a passed record of this script, the
pinned vLLM and the bench argv the harness builds now.

vLLM 0.30.0 (read 2026-10-23 at the v0.30.0 tag): main_async seeds random and numpy with --seed; get_samples builds
RandomDataset(random_seed=args.seed) and samples num_prompts requests (input_len less the tokenizer's special tokens;
prompt_len is the re-encoded length without special tokens); the warmups resend input_requests[0];
async_request_openai_completions replaces each request's prompt_len by the server's usage.prompt_tokens, which the
saved input_lens are; _align_prompts_to_server_tokenizer compares prompt 0 only and truncates longer prompts.

Usage (WSL, research checkout):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/stage_e1p_seed_check.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
import core_b_plan as bp  # noqa: E402
import stage_e1p_plan as pp  # noqa: E402


def check(samples: dict, *, N: int, input_len: int, server_len: Callable) -> dict:
    """samples: {label: (seed, [(prompt, prompt_len)])} at one N -> per-seed counts, ok and problems."""
    seeds, problems = {}, []
    for x, (seed, reqs) in samples.items():
        n_server = sum(server_len(p) == input_len for p, _ in reqs)
        seeds[x] = {"seed": seed, "n": len(reqs), "server_tokens_ok": n_server,
                    "prompt_len_ok": sum(n == input_len for _, n in reqs),
                    "server_tokens_other": sorted({server_len(p) for p, _ in reqs} - {input_len}),
                    "prompts_sha256": hashlib.sha256(json.dumps([p for p, _ in reqs],
                                                                ensure_ascii=False).encode()).hexdigest()}
        if len(reqs) != N or n_server != N:
            problems.append(f"N {N}, seed {x} ({seed}): {n_server} of {len(reqs)} prompts count {input_len} tokens "
                            "on the server; those benches would be invalid")
    if len(seeds) != 2 or len({s["prompts_sha256"] for s in seeds.values()}) != len(seeds):
        problems.append(f"N {N}: the seeds drew the same prompts (--seed does not change the content; J would be "
                        "meaningless)")
    return {"N": N, "seeds": seeds, "ok": not problems, "problems": problems}


def env_problems(env) -> list:
    """The bench's env (041 v3 §1.7): no inherited engine or PYTORCH_* variable, as a formal session requires."""
    bad = bp.inherited_unlisted(env) + sorted(k for k in env if k.startswith("PYTORCH_"))
    return [f"inherited variables {bad}: the bench runs without them"] if bad else []


def vllm_sampler() -> Callable:  # pragma: no cover - needs vLLM (WSL)
    import random

    import numpy as np
    from vllm.benchmarks import serve as vs
    from vllm.benchmarks.datasets.datasets import get_samples
    from vllm.tokenizers import get_tokenizer

    def sample(argv: list) -> tuple:
        """(the [(prompt, prompt_len)] `vllm bench serve` would send for argv, the server-side token count)."""
        parser = argparse.ArgumentParser()
        vs.add_cli_args(parser)
        args = parser.parse_args(argv[3:])                         # after "<vllm> bench serve"
        if getattr(args, "input_len", None) is not None:           # as main_async
            args.random_input_len = args.input_len
        tokenizer = get_tokenizer(args.tokenizer if args.tokenizer is not None else args.model,
                                  tokenizer_mode=args.tokenizer_mode, trust_remote_code=args.trust_remote_code)
        random.seed(args.seed)
        np.random.seed(args.seed)
        reqs = get_samples(args, tokenizer)
        return [(r.prompt, r.prompt_len) for r in reqs], lambda p: len(tokenizer(p).input_ids)

    return sample


def main(argv: list | None = None) -> int:  # pragma: no cover - WSL only
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT.joinpath(*pp.SEED_CHECK)))
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.parent.exists():
        print(f"stage_e1p_seed_check: {out.parent} exists; a seed check is never overwritten", file=sys.stderr)
        return 2
    pins, cfg = bp.load_pins(), tomllib.loads(pp.CONFIG.read_text())
    problems = pp.config_problems(cfg) + env_problems(os.environ)
    if problems:
        print("stage_e1p_seed_check: " + "; ".join(problems), file=sys.stderr)
        return 2
    os.environ.update(bp.bench_env())
    import vllm
    if vllm.__version__ != pins["vllm"]["version"]:                # no record that no plan could use
        print(f"stage_e1p_seed_check: vLLM {vllm.__version__}, not the pinned {pins['vllm']['version']}; nothing "
              "written", file=sys.stderr)
        return 2
    sample, input_len = vllm_sampler(), pp.L - bp.N_GEN
    checks, argvs = [], {}
    for N in pp.seed_check_ns(cfg):
        samples, argvs[str(N)], server_len = {}, {}, None
        for x, seed in cfg["e1p"]["seeds"].items():
            argvs[str(N)][x] = pp.seed_check_argv(pins, x, seed, N)
            reqs, server_len = sample(argvs[str(N)][x])
            samples[x] = (seed, reqs)
        checks.append(check(samples, N=N, input_len=input_len, server_len=server_len))
    rec = {"stage": pp.STAGE, "check": "seed", "decision": "041 v3 §1.7", "vllm_version": vllm.__version__,
           "script_sha256": pp.seed_check_sha256(), "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "input_len": input_len, "argv": argvs, "checks": checks, "ok": all(c["ok"] for c in checks),
           "problems": [p for c in checks for p in c["problems"]]}
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"ok": rec["ok"], "vllm_version": rec["vllm_version"], "problems": rec["problems"],
                      "checks": [{"N": c["N"], **{x: s["server_tokens_ok"] for x, s in c["seeds"].items()}}
                                 for c in checks]}))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
