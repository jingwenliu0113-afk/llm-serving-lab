#!/usr/bin/env python3
"""E1-P's seed check (decision 041 v3 §1.7), before the smoke. CPU only, on WSL with the project venv and the offline
HF cache, while tools/rtx/gpu_status.sh prints FREE (it reads WSL's disk; run it under flock -n so no GPU session
starts meanwhile). For seed A and seed B it parses the formal bench argv (stage_e1p_plan.bench_argv_for at N 200) with
vLLM's own `vllm bench serve` parser, seeds random / numpy as its main_async does, and draws the prompts with vLLM's
get_samples, exactly as the bench will. Per seed it records how many prompts have prompt_len 8,064 (what the bench
reports as input_lens and run_core_b.summarize_result checks), how many re-tokenize to 8,064 (descriptive: the bench
aligns longer prompts to the server's tokenizer), the other lengths, and the sha256 of the prompts. ok = every prompt
of both seeds has prompt_len 8,064 and the two seeds drew different prompts. The record goes to
results/stage_e/e1p-seed-check/seed_check.json (never overwritten) and is committed; stage_e1p_plan builds no plan
without a passed record made by this script under the pinned vLLM.

vLLM 0.30.0 (read 2026-10-23 at the v0.30.0 tag): main_async seeds random and numpy with --seed; get_samples builds
RandomDataset(random_seed=args.seed) and samples num_prompts requests; the warmups resend input_requests[0]; the
saved input_lens are the requests' prompt_len, the length of the re-encoded token sequence.

Usage (WSL, research checkout):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/stage_e1p_seed_check.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
import core_b_plan as bp  # noqa: E402
import stage_e1p_plan as pp  # noqa: E402


def check(samples: dict, *, N: int, input_len: int, encode_len: Callable) -> dict:
    """samples: {label: (seed, [(prompt, prompt_len)])} -> the per-seed record, ok and problems."""
    seeds, problems = {}, []
    for x, (seed, reqs) in samples.items():
        n_ok = sum(n == input_len for _, n in reqs)
        seeds[x] = {"seed": seed, "n": len(reqs), "prompt_len_ok": n_ok,
                    "prompt_lens_other": sorted({n for _, n in reqs if n != input_len}),
                    "retokenized_ok": sum(encode_len(p) == input_len for p, _ in reqs),
                    "prompts_sha256": hashlib.sha256(json.dumps([p for p, _ in reqs],
                                                                ensure_ascii=False).encode()).hexdigest()}
        if len(reqs) != N or n_ok != N:
            problems.append(f"seed {x} ({seed}): {n_ok} of {len(reqs)} prompts have prompt_len {input_len} "
                            f"(N {N}); its benches would all be invalid")
    if len(seeds) != 2 or len({s["prompts_sha256"] for s in seeds.values()}) != len(seeds):
        problems.append("the seeds drew the same prompts: --seed does not change the content (J would be meaningless)")
    return {"seeds": seeds, "ok": not problems, "problems": problems}


def vllm_sampler() -> Callable:  # pragma: no cover - needs vLLM (WSL)
    import random

    import numpy as np
    from vllm.benchmarks import serve as vs
    from vllm.benchmarks.datasets.datasets import get_samples
    from vllm.tokenizers import get_tokenizer

    def sample(argv: list) -> tuple:
        """(the [(prompt, prompt_len)] `vllm bench serve` would send for argv, a re-tokenizing length)."""
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
    problems = pp.config_problems(cfg)
    if problems:
        print("stage_e1p_seed_check: configs/stage_e1p.toml: " + "; ".join(problems), file=sys.stderr)
        return 2
    import vllm
    sample, N = vllm_sampler(), bp.num_prompts(pp.C)
    samples, argvs, encode_len = {}, {}, None
    for x, seed in cfg["e1p"]["seeds"].items():
        argvs[x] = pp.bench_argv_for(pins, seed=seed, N=N, prefix=f"e1p-seedcheck-{x}-", result_dir="/unused",
                                     result_filename="unused.json", vllm_bin="vllm")
        reqs, encode_len = sample(argvs[x])
        samples[x] = (seed, reqs)
    rec = {"stage": pp.STAGE, "check": "seed", "decision": "041 v3 §1.7", "vllm_version": vllm.__version__,
           "script_sha256": pp.seed_check_sha256(), "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "N": N, "input_len": pp.L - bp.N_GEN, "argv": argvs,
           **check(samples, N=N, input_len=pp.L - bp.N_GEN, encode_len=encode_len)}
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"ok": rec["ok"], "problems": rec["problems"],
                      "seeds": {x: {k: s[k] for k in ("n", "prompt_len_ok", "retokenized_ok")}
                                for x, s in rec["seeds"].items()}}))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
