#!/usr/bin/env python3
"""Stage E3 analysis (decision 041 v3 §3.3-3.5), one formal session per platform, every comparison inside it.

- t_c = 1000 / m_c ms/token from core_a_analysis.cell_stats (decode at L 2048), SE(t) = 1000 se / m^2; a cell is
  usable with 3 ok processes and no RTX speed spill (core_a_analysis.spill_cells).
- x_c = core_a_analysis.decode_bytes(file, token_embd if untied else 0, L) in GB.
- OLS t = a + b x on the six old configs; residuals r = t - (a + b x) for all eight.
- a_f per format (Q8_0, Q6_K, Q4_K_M, and Q4_0 when usable): the 4B-9B line's intercept with its SE; every pairwise
  difference with SE = sqrt(SE_1^2 + SE_2^2) (independent cells); the inverse-variance homogeneity chi^2.
- Shared-slope fit on the six old configs: t = a_size + b x (3 df); a_size = a0 + c P_size / 1e9 with the pinned
  parameter counts gives a0 and c (ms per billion parameters), with no degrees of freedom left.
- D_s = r(s, Q4_0) - r(s, Q4_K_M), SE_D = sqrt(SE_t(Q4_0)^2 + SE_t(Q4_K_M)^2 + (dx SE_b)^2); kappa = max over s of
  |r(s, Q6_K) - r(s, Q4_K_M)|; D_s is "negative beyond" / "positive beyond" when |D_s| > max(2 SE_D, kappa).
- Premise: the H-A2 pair (4B-Q4_K_M / 9B-Q8_0) is slower than its bytes ratio by > 2 SE (delta method).
  phi = (R_bytes - R_affine) / (R_bytes - R_observed), the share of the pair's deviation an affine t(x) explains, both
  in-sample (the six-config fit) and leave-pair-out (fit on the four configs outside the pair, 2 df).
- H-E3a: all three old a_f > 2 SE(a_f); needs the six old configs. H-E3b: both D_s negative beyond; needs the six old
  configs (the fit, kappa) and both Q4_0s. NT when a needed cell is not usable.
- The interpretation table of §3.5 per platform; two platforms make a general statement only when both have a
  definite row and it is the same one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import core_a_analysis as caa  # noqa: E402
import stage_e3 as e3  # noqa: E402

OLD_QUANTS = ("Q8_0", "Q6_K", "Q4_K_M")
OLD = tuple((s, q) for s in e3.SIZES for q in OLD_QUANTS)
PAIR = (("4B", "Q4_K_M"), ("9B", "Q8_0"))              # H-A2's refuting pair


class AnalysisError(RuntimeError):
    pass


def _bytes(files: dict, L: int) -> tuple:
    fb = {tuple(k.split("-", 1)): f["size_bytes"] for k, f in files.items()}
    eb = {tuple(k.split("-", 1)): (0 if f["tied"] else f["token_embd_bytes"]) for k, f in files.items()}
    return fb, eb


def times(rows: list, files: dict, *, L: int, platform: str, spill_params: dict | None = None) -> dict:
    """{(size, quant): {t, se, n, usable, spill}} at L from the session's decode rows."""
    cells = caa.cell_stats([r for r in rows if r.get("kind") == "decode"])
    fb, eb = _bytes(files, L)
    spill = caa.spill_cells(cells, fb, eb, spill_params) if platform == "wsl2-cuda" and spill_params else set()
    out = {}
    for (s, q, l), c in cells.items():
        if l != L:
            continue
        d = c["decode"]
        usable = bool(d and d["n"] >= 3 and d["se"] is not None and c["outcome"] == "ok" and (s, q, l) not in spill)
        out[(s, q)] = {"t": 1000 / d["m"] if d else None,
                       "se": 1000 * d["se"] / d["m"] ** 2 if d and d["se"] is not None else None,
                       "n": d["n"] if d else 0, "usable": usable, "spill": (s, q, l) in spill}
    return out


def x_gb(files: dict, key: tuple, L: int) -> float:
    f = files[f"{key[0]}-{key[1]}"]
    return caa.decode_bytes(f["size_bytes"], 0 if f["tied"] else f["token_embd_bytes"], L) / 1e9


def ols(xs: list, ys: list) -> dict:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    s = math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    return {"a": a, "b": b, "s": s, "se_a": s * math.sqrt(1 / n + mx ** 2 / sxx), "se_b": s / math.sqrt(sxx), "n": n}


def intercept(t4: dict, t9: dict, x4: float, x9: float) -> dict:
    a = (x9 * t4["t"] - x4 * t9["t"]) / (x9 - x4)
    se = math.sqrt((x9 * t4["se"]) ** 2 + (x4 * t9["se"]) ** 2) / (x9 - x4)
    return {"a_f": a, "se": se, "beyond_2se": a > 2 * se}


def shared_slope(points: dict, n_params: dict) -> dict:
    """points {size: [(x, t)]}: per-size intercepts with one common slope, then a0 and c from the parameter counts."""
    means = {s: (sum(x for x, _ in v) / len(v), sum(t for _, t in v) / len(v)) for s, v in points.items()}
    sxx = sum((x - means[s][0]) ** 2 for s, v in points.items() for x, _ in v)
    b = sum((x - means[s][0]) * (t - means[s][1]) for s, v in points.items() for x, t in v) / sxx
    a = {s: means[s][1] - b * means[s][0] for s in points}
    resid = [t - (a[s] + b * x) for s, v in points.items() for x, t in v]
    df = len(resid) - len(points) - 1
    sd = math.sqrt(sum(r * r for r in resid) / df)
    p4, p9 = n_params["4B"], n_params["9B"]
    return {"b": b, "se_b": sd / math.sqrt(sxx), "a_by_size": a, "s": sd, "df": df,
            "a0": (p9 * a["4B"] - p4 * a["9B"]) / (p9 - p4), "c_ms_per_gparam": (a["9B"] - a["4B"]) / (p9 - p4) * 1e9}


def phi(fit: dict, files: dict, T: dict, L: int) -> float | None:
    pa, pb = PAIR
    rb = x_gb(files, pb, L) / x_gb(files, pa, L)
    ro = T[pb]["t"] / T[pa]["t"]
    affine = (fit["a"] + fit["b"] * x_gb(files, pb, L)) / (fit["a"] + fit["b"] * x_gb(files, pa, L))
    return (rb - affine) / (rb - ro) if rb != ro else None


def _state(d: float, thr: float) -> str:
    if abs(d) <= thr:
        return "inside"
    return "negative beyond" if d < 0 else "positive beyond"


def analyze(rows: list, files: dict, *, platform: str, n_params: dict, L: int = 2048,
            spill_params: dict | None = None) -> dict:
    T = times(rows, files, L=L, platform=platform, spill_params=spill_params)
    use = lambda k: T.get(k, {}).get("usable", False)  # noqa: E731
    out = {"platform": platform, "L": L,
           "cells": {f"{s}-{q}": {**v, "x_gb": x_gb(files, (s, q), L)} for (s, q), v in sorted(T.items())}}
    # premise: the H-A2 pair is slower than its bytes ratio
    a4, b9 = PAIR
    if use(a4) and use(b9):
        ro = T[b9]["t"] / T[a4]["t"]
        se_ro = ro * math.hypot(T[a4]["se"] / T[a4]["t"], T[b9]["se"] / T[b9]["t"])
        rb = x_gb(files, b9, L) / x_gb(files, a4, L)
        out["premise"] = {"observed_ratio": ro, "bytes_ratio": rb, "se": se_ro, "holds": rb - ro > 2 * se_ro}
    else:
        out["premise"] = {"holds": None, "reason": "the H-A2 pair is not usable"}
    if not all(use(k) for k in OLD):
        out.update({"fit": None, "H-E3a": {"verdict": "NT", "reason": "an old config is not usable"},
                    "H-E3b": {"verdict": "NT", "reason": "no fit"}})
        out["row"] = interpret(out)
        return out
    fit = ols([x_gb(files, k, L) for k in OLD], [T[k]["t"] for k in OLD])
    resid = {k: T[k]["t"] - (fit["a"] + fit["b"] * x_gb(files, k, L)) for k in T if use(k)}
    out["fit"] = {**fit, "residuals_ms": {f"{s}-{q}": r for (s, q), r in sorted(resid.items())},
                  "residuals_pct": {f"{s}-{q}": 100 * r / T[(s, q)]["t"] for (s, q), r in sorted(resid.items())}}
    a_f = {q: intercept(T[("4B", q)], T[("9B", q)], x_gb(files, ("4B", q), L), x_gb(files, ("9B", q), L))
           for q in OLD_QUANTS}
    if use(("4B", "Q4_0")) and use(("9B", "Q4_0")):
        a_f["Q4_0"] = intercept(T[("4B", "Q4_0")], T[("9B", "Q4_0")], x_gb(files, ("4B", "Q4_0"), L),
                                x_gb(files, ("9B", "Q4_0"), L))
    w = {q: 1 / v["se"] ** 2 for q, v in a_f.items() if q in OLD_QUANTS}
    mean_a = sum(a_f[q]["a_f"] * w[q] for q in w) / sum(w.values())
    qs = list(a_f)
    out["intercepts"] = {"by_format": a_f, "weighted_mean": mean_a,
                         "homogeneity_chi2": sum((a_f[q]["a_f"] - mean_a) ** 2 * w[q] for q in w), "df": len(w) - 1,
                         "differences": {f"{q1} - {q2}": {"diff": a_f[q1]["a_f"] - a_f[q2]["a_f"],
                                                          "se": math.hypot(a_f[q1]["se"], a_f[q2]["se"])}
                                         for i, q1 in enumerate(qs) for q2 in qs[i + 1:]}}
    out["shared_slope"] = shared_slope({s: [(x_gb(files, (s, q), L), T[(s, q)]["t"]) for q in OLD_QUANTS]
                                        for s in e3.SIZES}, n_params)
    out["H-E3a"] = {"verdict": "supported" if all(a_f[q]["beyond_2se"] for q in OLD_QUANTS) else "not supported",
                    "prediction": "supported"}
    kappa = max(abs(resid[(s, "Q6_K")] - resid[(s, "Q4_K_M")]) for s in e3.SIZES)
    D = {}
    for s in e3.SIZES:
        if not use((s, "Q4_0")):
            D[s] = {"state": "NT"}
            continue
        d = resid[(s, "Q4_0")] - resid[(s, "Q4_K_M")]
        dx = x_gb(files, (s, "Q4_0"), L) - x_gb(files, (s, "Q4_K_M"), L)
        se_d = math.sqrt(T[(s, "Q4_0")]["se"] ** 2 + T[(s, "Q4_K_M")]["se"] ** 2 + (dx * fit["se_b"]) ** 2)
        thr = max(2 * se_d, kappa)
        D[s] = {"D": d, "se": se_d, "threshold": thr, "state": _state(d, thr)}
    out["format_contrast"] = {"kappa": kappa, "by_size": D}
    if any(v["state"] == "NT" for v in D.values()):
        out["H-E3b"] = {"verdict": "NT", "reason": "a Q4_0 cell is not usable"}
    else:
        both = all(v["state"] == "negative beyond" for v in D.values())
        out["H-E3b"] = {"verdict": "supported" if both else "not supported", "prediction": "none"}
    if out["premise"].get("holds") is not None:
        rest = [k for k in OLD if k not in PAIR]
        lpo = ols([x_gb(files, k, L) for k in rest], [T[k]["t"] for k in rest])
        out["premise"]["phi"] = {"in_sample": phi(fit, files, T, L), "leave_pair_out": phi(lpo, files, T, L)}
    out["row"] = interpret(out)
    return out


def interpret(a: dict) -> str:
    """041 v3 §3.5, first matching row."""
    if a["premise"].get("holds") is False:
        return "premise fails: H-A2's deviation not reproduced in the session; no attribution"
    if a["premise"].get("holds") is None or "NT" in (a["H-E3a"]["verdict"], a["H-E3b"]["verdict"]):
        return "incomplete"
    ea, eb = a["H-E3a"]["verdict"] == "supported", a["H-E3b"]["verdict"] == "supported"
    states = {v["state"] for v in a["format_contrast"]["by_size"].values()}
    if ea and eb:
        return "both: a non-proportional component and a 4-bit k-quant format cost"
    if ea and states == {"inside"}:
        return "non-proportional component; no evidence of a format cost (not excluded)"
    if ea and states == {"positive beyond"}:
        return "Q4_0 slower than its bytes predict: matches neither candidate (output.weight stays Q6_K)"
    if ea:
        return "inconsistent format effect across sizes: H-E3b not interpreted"
    if eb:
        return "no measurable non-proportional component; the format cost is the only supported mechanism"
    return "neither candidate supported; H-A2's deviation unexplained"


def combine(mac: dict | None, rtx: dict | None) -> dict:
    rows = {p: a["row"] for p, a in (("macos-metal", mac), ("wsl2-cuda", rtx)) if a}
    if len(rows) < 2 or "incomplete" in rows.values():
        return {"rows": rows, "general": None, "statement": "incomplete (a platform is NT or missing)"}
    general = len(set(rows.values())) == 1
    return {"rows": rows, "general": rows["macos-metal"] if general else None,
            "statement": "the same row on both platforms" if general else "the cause differs by platform"}


# ---------------------------------------------------------------- formal sessions

def session_problems(path, *, pins: dict, cfg: dict, root=ROOT) -> list:
    path = Path(path)
    try:
        s = json.loads((path / "session.json").read_text())
        plan = json.loads((path / "plan.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"{path}: no readable session.json / plan.json ({exc})"]
    platform = s.get("platform")
    problems = []
    if (s.get("stage"), s.get("mode"), s.get("status")) != (e3.STAGE, "formal", "finished") or \
            platform not in e3.PLATFORMS:
        return [f"{path.name}: not a finished formal E3 session"]
    if path.resolve().parent != Path(root).resolve().joinpath(*e3.RESULTS):
        problems.append(f"{path}: not results/stage_e/<session id>")
    if s.get("env_valid") is not True or s.get("sentinel_ok") is not True:
        problems.append(f"{path.name}: env-invalid or sentinel drift (void; one rerun under a new id, 041 v3 §3.6)")
    problems += [f"{path.name}: {p}" for p in e3.host_problems(s.get("host"), platform)]
    if s.get("plan_sha256") != e3.sha256_json(plan) or s.get("harness_sha256") != e3.harness_sha256():
        problems.append(f"{path.name}: plan_sha256 / harness_sha256 do not match")
    try:
        want = e3.build_plan(pins, cfg, platform=platform, mode="formal", root=root)
        drop = lambda p: {k: v for k, v in p.items() if k != "pins_sha256"}  # noqa: E731 - content pins it
        if drop(plan) != drop(want):
            problems.append(f"{path.name}: plan.json is not build_plan's formal {platform} plan")
    except e3.StageE3Error as exc:
        problems.append(f"{path.name}: build_plan refuses its formal plan ({exc})")
    earlier = [a for a in e3.formal_attempts(root, platform)
               if a["dir"].resolve() != path.resolve() and a["started_at"] < (s.get("started_at") or "")]
    if len(earlier) > 1 or any(a["valid"] for a in earlier):
        problems.append(f"{path.name}: earlier formal {platform} attempts {[a['dir'].name for a in earlier]} (only the "
                        "first valid session, after at most one void attempt, is analyzed)")
    return problems


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", action="append", required=True, help="results/stage_e/<formal E3 session>")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    raw, cfg_raw = (ROOT / "configs" / "pins.toml").read_bytes(), e3.CONFIG.read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(cfg_raw.decode())
    by_platform = {}
    for sess in map(Path, args.session):
        problems = session_problems(sess, pins=pins, cfg=cfg)
        if problems:
            print("stage_e3_analysis: " + "; ".join(problems), file=sys.stderr)
            return 2
        s = json.loads((sess / "session.json").read_text())
        if s["platform"] in by_platform:
            print(f"stage_e3_analysis: one session per platform ({s['platform']} twice)", file=sys.stderr)
            return 2
        plan = json.loads((sess / "plan.json").read_text())
        rows = [json.loads(x) for x in (sess / "rows.jsonl").read_text().splitlines() if x.strip()]
        sb = pins["stage_b"]
        params = {"v_total_mib": sb["v_total_mib"], "b_mib": (s.get("session_b") or {}).get("b_mib", sb["b_mib"]),
                  "c_ctx_mib": sb["c_ctx_mib"]}
        by_platform[s["platform"]] = {"session": sess.name, **analyze(rows, plan["files"], platform=s["platform"],
                                                                      n_params=plan["n_params"], spill_params=params)}
    result = {"by_platform": by_platform, "cross_platform": combine(by_platform.get("macos-metal"),
                                                                    by_platform.get("wsl2-cuda"))}
    Path(args.out).write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({p: a["row"] for p, a in by_platform.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
