#!/usr/bin/env python3
"""Stage Q analysis (decision 040): the hypotheses H-Q1a-c and H-Q2a-b from the Stage Q sessions' rows and Phase 1's
formal quality session (formal-20261010-quality, the CUDA PPL and the KLD at -c 512), with quality_analysis's frozen
kld_test and holm (imported, never changed).

- KLD of a (size, ctx, quant): the ok kld row's tool mean +/- SE (005: token-independent SE, reported as such).
- H-Q1a / H-Q2a (ordering): per size (and context) KLD(Q8_0) < KLD(Q6_K) < KLD(Q4_K_M), z tests, Holm over the family
  (4 tests for Q1a, 8 for Q2a), with 005's rule (quality_analysis.h_a5): NT when a KLD is missing (an OOM included),
  else supported only when every test rejects in the predicted direction, else not supported.
- H-Q1b: KLD_Metal / KLD_CUDA on the same first 200 chunks in [0.8, 1.25] for each size and quant.
- H-Q1c: PPL_Metal (the full-test ppl job) / PPL_CUDA (Phase 1: PPL(Q) of the quant's KLD row, of the self-KLD row for BF16)
  in [0.995, 1.005] for each of the 8 configurations.
- H-Q2b: KLD(ctx) / KLD(512, Phase 1) in [0.5, 2.0] for each size, quant and context.
- Bounds (Q1b, Q1c, Q2b): not supported when an observed ratio is outside (even if another is missing: the observed
  violation stands), else NT when one is missing, else supported. They are 040's substantive thresholds, not
  statistical tests.

Formal sessions only (040 凍結): configs/stage_q.toml frozen with its pilot evidence. The analysis finds each part's
session itself (040 補記 14; no part can be left out): its formal attempts under results/stage_q
(stage_q.formal_attempts, minus those superseded by a committed decision log naming both ids), the first valid one,
after at most one void attempt; that session must be finished, env-valid, of its part on its platform's host,
directly under results/stage_q, with plan.json exactly build_plan's formal plan. A part with no formal attempt leaves
its hypotheses NT ("not run"); a part whose attempts are all void leaves them NT and lists the attempts. Bounds
compare exact decimal ratios.

Usage:
    python quality/stage_q_analysis.py --out results/stage_q/analysis-<date>.json \
        [--superseded <id>=docs/decisions/<log>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_analysis as qa  # noqa: E402
import stage_q as sq  # noqa: E402

QUANTS = ("Q8_0", "Q6_K", "Q4_K_M")
SIZES = ("4B", "9B")
CONTEXTS = (2048, 8192)
Q1B_BOUNDS = (Fraction(4, 5), Fraction(5, 4))
Q1C_BOUNDS = (Fraction(995, 1000), Fraction(1005, 1000))
Q2B_BOUNDS = (Fraction(1, 2), Fraction(2))


def kld(rows: list, *, default_ctx: int = 512) -> dict:
    """{(size, ctx, quant): (mean, se)} of the ok KLD rows (Phase 1 rows carry no ctx: -c 512)."""
    return {(r["size"], r.get("ctx") or default_ctx, r["point"]): (r["metrics"]["kld_mean"], r["metrics"]["kld_mean_se"])
            for r in rows if r.get("kind") == "kld" and r.get("outcome") == "ok" and not r.get("variant")
            and not r.get("repeat_of") and r["metrics"].get("kld_mean") is not None}


def cuda_ppl(phase1_rows: list) -> dict:
    """{(size, point): PPL} from Phase 1: PPL(Q) of each quant's KLD row and, for BF16, of the self-KLD row (the same
    evaluation as the quants; README's 9.945 / 8.186). PPL(base), from the base run, differs by about 0.04 %."""
    out = {}
    for r in phase1_rows:
        if r.get("kind") in ("kld", "kld_self") and r.get("outcome") == "ok" and not r.get("variant") \
                and not r.get("repeat_of"):
            out[(r["size"], r["point"])] = r["metrics"]["ppl_q"]
    return out


def metal_ppl(rows: list) -> dict:
    return {(r["size"], r["point"]): r["metrics"]["ppl"] for r in rows
            if r.get("kind") == "ppl" and r.get("outcome") == "ok" and r["metrics"].get("ppl") is not None}


def ordering(table: dict, units: list) -> dict:
    """units: [(size, ctx)]; the KLD ordering tests of all units as one Holm family."""
    tests, missing = [], []
    for size, ctx in units:
        for lo, hi in qa.KLD_PAIRS:
            a, b = table.get((size, ctx, lo)), table.get((size, ctx, hi))
            if a is None or b is None:
                missing.append(f"KLD {size} c{ctx} {hi} > {lo}")
                continue
            t = qa.kld_test(f"{size} c{ctx}", lo, a, hi, b)
            tests.append(t)
    if missing:                                         # 005: a missing cell makes the family not testable
        return {"verdict": "NT", "missing": missing, "tests": tests}
    adj = qa.holm({t["name"]: t["p"] for t in tests})
    for t in tests:
        t.update(p_adj=adj[t["name"]]["p_adj"], reject=adj[t["name"]]["reject"])
    verdict = "supported" if all(t["reject"] and t["direction_ok"] for t in tests) else "not supported"
    return {"verdict": verdict, "correction": "Holm", "alpha": qa.ALPHA, "tests": tests}


def bounded(ratios: dict, bounds: tuple) -> dict:
    """ratios: {name: exact Fraction or None}; supported when all inside bounds (inclusive), not supported when one is
    outside (even if another is missing), NT when one is missing and none is outside."""
    lo, hi = bounds
    inside = {k: None if v is None else lo <= v <= hi for k, v in ratios.items()}
    if any(v is False for v in inside.values()):
        verdict = "not supported"
    elif any(v is None for v in inside.values()):
        verdict = "NT"
    else:
        verdict = "supported"
    return {"verdict": verdict, "bounds": [float(lo), float(hi)],
            "ratios": {k: None if v is None else float(v) for k, v in ratios.items()}, "inside": inside}


def _ratio(a, b):
    """The exact ratio of the two printed decimals (040 補記 9: the bounds include their endpoints)."""
    return None if a is None or b is None or b == 0 else Fraction(str(a)) / Fraction(str(b))


def h_q1(metal_rows: list, cuda_rows: list, phase1_rows: list) -> dict:
    m, c = kld(metal_rows), kld(cuda_rows)
    mp, cp_ = metal_ppl(metal_rows), cuda_ppl(phase1_rows)
    return {
        "H-Q1a": {**ordering(m, [(s, 512) for s in SIZES]), "prediction": "supported"},
        "H-Q1b": {**bounded({f"{s} {q}": _ratio((m.get((s, 512, q)) or (None,))[0], (c.get((s, 512, q)) or (None,))[0])
                             for s in SIZES for q in QUANTS}, Q1B_BOUNDS), "prediction": "supported"},
        "H-Q1c": {**bounded({f"{s} {p}": _ratio(mp.get((s, p)), cp_.get((s, p)))
                             for s in SIZES for p in ("BF16", *QUANTS)}, Q1C_BOUNDS), "prediction": "supported"},
    }


def h_q2(q2_rows: list, phase1_rows: list) -> dict:
    k, base = kld(q2_rows), kld(phase1_rows)
    return {
        "H-Q2a": {**ordering(k, [(s, c) for c in CONTEXTS for s in SIZES]), "prediction": "supported"},
        "H-Q2b": {**bounded({f"{s} {q} c{c}": _ratio((k.get((s, c, q)) or (None,))[0],
                                                       (base.get((s, 512, q)) or (None,))[0])
                             for c in CONTEXTS for s in SIZES for q in QUANTS}, Q2B_BOUNDS),
                  "prediction": "supported"},
    }


def load_rows(path) -> list:
    return [json.loads(x) for x in (Path(path) / "rows.jsonl").read_text().splitlines() if x.strip()]


class AnalysisError(RuntimeError):
    pass


def session_problems(path, part: str, *, stage_q_sha256: str, pins_sha256: str, pins: dict, cfg: dict,
                     root=sq.ROOT) -> list:
    """A finished, env-valid formal session of this part on its platform's host, directly under results/stage_q, that
    ran build_plan's formal plan (the frozen harness, configs/stage_q.toml and pins.toml)."""
    path = Path(path)
    try:
        meta = json.loads((path / "session.json").read_text())
        plan = json.loads((path / "plan.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"{path}: no readable session.json / plan.json ({exc})"]
    harness = sq.harness_sha256()
    problems = []
    if meta.get("mode") != "formal" or plan.get("mode") != "formal":
        problems.append(f"{path.name}: mode {meta.get('mode')!r} is not formal (pilots stay masked, 040)")
    if meta.get("status") != "finished":
        problems.append(f"{path.name}: status {meta.get('status')!r}, not a finished session")
    if path.resolve().parent != Path(root).resolve().joinpath(*sq.RESULTS):
        problems.append(f"{path}: not results/stage_q/<session id>")
    if meta.get("env_valid") is not True:
        problems.append(f"{path.name}: the session is env-invalid (void: one rerun under a new id, 012)")
    if (meta.get("stage"), meta.get("part"), plan.get("part"), meta.get("platform")) != \
            (sq.STAGE, part, part, sq.PARTS[part]):
        problems.append(f"{path.name}: not a {part} session (stage / part / platform)")
    if meta.get("plan_sha256") != sq.sha256_json(plan):
        problems.append(f"{path.name}: plan.json is not the plan the session ran (plan_sha256)")
    if meta.get("harness_sha256") != harness or plan.get("harness_sha256") != harness:
        problems.append(f"{path.name}: ran another harness (harness_sha256; 040 凍結 freezes it)")
    if plan.get("stage_q_sha256") != stage_q_sha256:
        problems.append(f"{path.name}: planned with another configs/stage_q.toml (stage_q_sha256)")
    if plan.get("pins_sha256") != pins_sha256:
        problems.append(f"{path.name}: planned with another configs/pins.toml (pins_sha256)")
    problems += [f"{path.name}: {p}" for p in sq.host_problems(meta.get("host"), sq.PARTS[part])]
    if meta.get("inputs_verified") is not True:
        problems.append(f"{path.name}: its inputs were not checked against pins (inputs_verified)")
    if not problems:
        try:
            want = sq.build_plan(pins, part=part, mode="formal", pins_sha256=pins_sha256, cfg=cfg,
                                 stage_q_sha256=stage_q_sha256, root=root)
        except sq.StageQError as exc:
            want = None
            problems.append(f"{path.name}: build_plan refuses its formal plan ({exc})")
        if want is not None and plan != want:
            problems.append(f"{path.name}: plan.json is not build_plan's formal {part} plan")
    return problems


def part_session(part: str, *, root, superseded: dict, tracked) -> tuple:
    """(the session dir to analyze or None, record): the part's first valid formal attempt after at most one void one,
    superseded attempts left out once their decision log names both ids (040 補記 14)."""
    attempts = sq.formal_attempts(root, part)
    kept = [a for a in attempts if a["id"] not in superseded]
    gone = [a["id"] for a in attempts if a["id"] in superseded]
    valid = [i for i, a in enumerate(kept) if a["valid"]]
    record = {"attempts": [a["id"] for a in kept], "superseded": gone}
    if not kept:
        return None, {**record, "status": "not run"}
    if not valid:
        return None, {**record, "status": "no valid formal session (every attempt is void)"}
    first = kept[valid[0]]
    if valid[0] > 1:
        raise AnalysisError(f"{part}: {valid[0]} void attempts {record['attempts'][:valid[0]]} before {first['id']}; "
                            "040 allows one rerun")
    bad = sq.superseded_problems({k: superseded[k] for k in gone}, first["id"], root=root, tracked=tracked)
    if bad:
        raise AnalysisError("; ".join(bad))
    return first["dir"], {**record, "status": "analyzed", "session": first["id"],
                          "void_before": [a["id"] for a in kept[:valid[0]]]}


def phase1_problems(path) -> list:
    """Phase 1's formal quality session (005 / 020): formal, env-valid, on wsl2-cuda."""
    path = Path(path)
    try:
        meta = json.loads((path / "session.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"{path}: no readable session.json ({exc})"]
    if (meta.get("mode"), meta.get("env_valid"), meta.get("platform"), meta.get("session_id")) != \
            ("formal", True, "wsl2-cuda", path.name):
        return [f"{path.name}: not Phase 1's formal, env-valid wsl2-cuda quality session"]
    return []


def analyze_sessions(*, cfg: dict, stage_q_sha256: str, pins_sha256: str, pins: dict, root=sq.ROOT,
                     superseded: dict | None = None, tracked=sq.git_tracked) -> dict:
    """Every part's session found under root (part_session). Refuses (AnalysisError) unless 040 凍結 holds."""
    problems = [f"configs/stage_q.toml: {p}" for p in sq.config_problems(cfg)]
    if not problems and cfg["freeze"]["stage_q"] is not True:
        problems.append("freeze.stage_q is not true in configs/stage_q.toml (040 凍結: no Stage Q analysis before it)")
    if problems:
        raise AnalysisError("; ".join(problems))
    problems += [f"freeze evidence: {p}" for p in sq.freeze_evidence_problems(cfg, root, pins)]
    paths, records = {}, {}
    for part in sq.PARTS:
        path, records[part] = part_session(part, root=root, superseded=superseded or {}, tracked=tracked)
        if path is not None:
            paths[part] = path
            problems += session_problems(path, part, stage_q_sha256=stage_q_sha256, pins_sha256=pins_sha256,
                                         pins=pins, cfg=cfg, root=root)
    phase1 = Path(root) / "results" / "quality" / cfg["stage_q"]["phase1_session"]
    problems += phase1_problems(phase1)
    if problems:
        raise AnalysisError("; ".join(problems))
    rows = {part: load_rows(paths[part]) if part in paths else [] for part in sq.PARTS}
    base = load_rows(phase1)
    return {"stage": sq.STAGE,
            "sessions": {part: Path(paths[part]).name if part in paths else None for part in sq.PARTS},
            "parts": records, "phase1_session": phase1.name, "harness_sha256": sq.harness_sha256(),
            "stage_q_sha256": stage_q_sha256, "pins_sha256": pins_sha256,
            "hypotheses": {**h_q1(rows["q1-metal"], rows["q1-cuda"], base), **h_q2(rows["q2"], base)}}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage Q hypotheses (040) from its formal sessions")
    parser.add_argument("--out", required=True, help="results/stage_q/<name>.json (a new file)")
    parser.add_argument("--superseded", action="append", default=[], help="<session id>=docs/decisions/<log>.md")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    cfg_raw, pins_raw = sq.CONFIG.read_bytes(), (sq.ROOT / "configs" / "pins.toml").read_bytes()
    try:
        if out.parent != sq.ROOT.joinpath(*sq.RESULTS).resolve() or out.suffix != ".json" or out.exists():
            raise AnalysisError(f"--out {out} is not a new .json file directly under results/stage_q/")
        result = analyze_sessions(cfg=tomllib.loads(cfg_raw.decode()),
                                  stage_q_sha256=hashlib.sha256(cfg_raw).hexdigest(),
                                  pins_sha256=hashlib.sha256(pins_raw).hexdigest(),
                                  pins=tomllib.loads(pins_raw.decode()),
                                  superseded=dict(x.split("=", 1) for x in args.superseded))
    except AnalysisError as exc:
        print(f"stage_q_analysis: {exc}", file=sys.stderr)
        return 2
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v["verdict"] for k, v in result["hypotheses"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
