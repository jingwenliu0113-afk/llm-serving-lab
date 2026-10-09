#!/usr/bin/env python3
"""Pre-registered quality analysis (decision 005; PREREGISTRATION §2 H-A5). Written and
tested before any quality data exists; frozen with Stage A.

H-A5 confirmatory family (alpha 0.05, two-sided, Holm step-down over 5 p-values):
- per size, KLD(Q6_K) > KLD(Q8_0) and KLD(Q4_K_M) > KLD(Q6_K): z test on the tool's
  mean +- SE, denominator sqrt(SE_a^2 + SE_b^2);
- across sizes, paired per-chunk delta NLL (9B-Q4_K_M - 4B-Q8_0) < 0: chunk bootstrap,
  p = 2 min(P*(d >= 0), P*(d <= 0)), floored at 1 / n_boot.
Supported only if all 5 are significant after Holm and in the predicted direction;
otherwise refuted. A missing cell makes H-A5 not testable.

Downstream (descriptive): paired delta accuracy (second - first) for each quant vs BF16 and
for adjacent quants, 95% paired-bootstrap CI, flip rate; "resolved" iff the CI excludes 0.

Bootstrap: random.Random(seed) (Mersenne Twister, stable across CPython versions), seed
20261008, resampling indices with floor(random() * n). Stdlib only.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist, mean

SEED = 20261008
N_BOOT = 10_000
ALPHA = 0.05
KLD_PAIRS = (("Q8_0", "Q6_K"), ("Q6_K", "Q4_K_M"))
HA5_PAIR = (("9B", "Q4_K_M"), ("4B", "Q8_0"))  # delta = first - second, predicted < 0
DOWNSTREAM_PAIRS = (("BF16", "Q8_0"), ("BF16", "Q6_K"), ("BF16", "Q4_K_M"), ("Q8_0", "Q6_K"), ("Q6_K", "Q4_K_M"))


def kld_test(size: str, lo_q: str, lo: tuple, hi_q: str, hi: tuple) -> dict:
    """Predicted KLD(hi_q) > KLD(lo_q); lo / hi = (mean, se)."""
    z = (hi[0] - lo[0]) / math.sqrt(lo[1] ** 2 + hi[1] ** 2)
    p = 2 * (1 - NormalDist().cdf(abs(z)))
    return {"name": f"KLD {size} {hi_q} > {lo_q}", "z": z, "p": p, "direction_ok": z > 0}


def _bootstrap_means(values: list, n_boot: int, seed: int) -> list:
    rng, n = random.Random(seed), len(values)
    return [sum(values[math.floor(rng.random() * n)] for _ in range(n)) / n for _ in range(n_boot)]


def _percentile(sorted_vals: list, q: float) -> float:
    k = (len(sorted_vals) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def paired_nll_test(first: list, second: list, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    if len(first) != len(second) or not first:
        raise ValueError("paired per-chunk NLL lists must be non-empty and the same length")
    d = [a - b for a, b in zip(first, second)]
    boots = sorted(_bootstrap_means(d, n_boot, seed))
    ge = sum(x >= 0 for x in boots) / n_boot
    le = sum(x <= 0 for x in boots) / n_boot
    p = min(1.0, max(2 * min(ge, le), 1 / n_boot))
    m = mean(d)
    return {"name": "paired dNLL 9B-Q4_K_M - 4B-Q8_0 < 0", "mean_delta": m, "p": p, "direction_ok": m < 0,
            "ci95": [_percentile(boots, 0.025), _percentile(boots, 0.975)], "n_chunks": len(d),
            "n_boot": n_boot, "seed": seed}


def holm(pvalues: dict, alpha: float = ALPHA) -> dict:
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(order)
    out, running, stopped = {}, 0.0, False
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        reject = not stopped and pvalues[name] <= alpha / (m - i)
        stopped = stopped or not reject
        out[name] = {"p": pvalues[name], "p_adj": running, "reject": reject}
    return out


def h_a5(rows: list, *, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = ALPHA) -> dict:
    kld = {(r["size"], r["point"]): (r["metrics"]["kld_mean"], r["metrics"]["kld_mean_se"]) for r in rows
           if r["kind"] == "kld" and r["outcome"] == "ok" and not r.get("variant") and not r.get("repeat_of")}
    ppl = {(r["size"], r["point"]): r["metrics"]["chunk_nll"] for r in rows
           if r["kind"] == "ppl" and r["outcome"] == "ok"}
    sizes = sorted({s for s, _ in kld} | {"4B", "9B"})
    tests, missing = [], []
    for size in sizes:
        for lo, hi in KLD_PAIRS:
            if (size, lo) in kld and (size, hi) in kld:
                tests.append(kld_test(size, lo, kld[(size, lo)], hi, kld[(size, hi)]))
            else:
                missing.append(f"KLD {size} {hi} > {lo}")
    if all(k in ppl for k in HA5_PAIR):
        tests.append(paired_nll_test(ppl[HA5_PAIR[0]], ppl[HA5_PAIR[1]], n_boot=n_boot, seed=seed))
    else:
        missing.append("paired dNLL")
    if missing:
        return {"verdict": "not testable", "missing": missing, "tests": tests}
    adj = holm({t["name"]: t["p"] for t in tests}, alpha)
    for t in tests:
        t.update({"p_adj": adj[t["name"]]["p_adj"], "reject": adj[t["name"]]["reject"]})
    ok = all(t["reject"] and t["direction_ok"] for t in tests)
    return {"verdict": "not refuted" if ok else "refuted", "alpha": alpha, "correction": "Holm", "tests": tests}


def downstream(rows: list, *, n_boot: int = N_BOOT, seed: int = SEED) -> list:
    cells = {(r["task"], r["size"], r["point"]): r["metrics"]["correct"] for r in rows
             if r["kind"] in ("mc", "winogrande") and r["outcome"] == "ok" and not r.get("repeat_of")}
    out = []
    for task, size in sorted({(t, s) for t, s, _ in cells}):
        for a, b in DOWNSTREAM_PAIRS:
            x, y = cells.get((task, size, a)), cells.get((task, size, b))
            if not x or not y or len(x) != len(y):
                continue
            d = [j - i for i, j in zip(x, y)]
            boots = sorted(_bootstrap_means(d, n_boot, seed))
            lo, hi = _percentile(boots, 0.025), _percentile(boots, 0.975)
            out.append({"task": task, "size": size, "pair": [a, b], "n": len(d), "delta_acc": mean(d),
                        "ci95": [lo, hi], "resolved": lo > 0 or hi < 0,
                        "flip_rate": sum(v != 0 for v in d) / len(d), "n_boot": n_boot, "seed": seed})
    return out


def main(argv: list | None = None) -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="H-A5 and downstream analysis of a quality session")
    parser.add_argument("--session", required=True)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    session = Path(args.session)
    meta = json.loads((session / "session.json").read_text())
    problems = []
    if meta.get("mode") != "formal":
        problems.append(f"mode {meta.get('mode')!r} is not formal (the pilot report stays masked: 005, 011)")
    if meta.get("env_valid") is not True:
        problems.append(f"session is env-invalid (env_valid={meta.get('env_valid')!r}; rerun it, 012)")
    if meta.get("platform", "macos-metal") != "wsl2-cuda":
        problems.append(f"platform {meta.get('platform', 'macos-metal')!r}: formal quality is measured on wsl2-cuda "
                        "only (020)")
    if problems:
        print(f"quality_analysis: refusing {session}: " + "; ".join(problems), file=sys.stderr)
        return 2
    rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["kind"] != "delete_kld"]
    out = {"session": meta.get("session_id"), "mode": meta.get("mode"), "env_valid": meta.get("env_valid"),
           "platform": meta.get("platform"),
           "h_a5": h_a5(rows, n_boot=args.n_boot), "downstream": downstream(rows, n_boot=args.n_boot)}
    (session / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"h_a5": out["h_a5"]["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
