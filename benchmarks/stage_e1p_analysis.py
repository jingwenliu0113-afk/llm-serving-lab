#!/usr/bin/env python3
"""Stage E1-P analysis (decision 041 v3 §1.6): the seed-swap probe's reading. Descriptive, not a test.

Input: the first valid formal E1-P session under results/stage_e (stage_e1p_plan.formal_attempts; after at most one
void attempt), fetched to the Mac without raw/ (tools/rtx/fetch_all_sessions.sh stage_e); its valid benches' derived
requests and e1 files, each checked against the sha256 its row recorded (rows.jsonl against session.json).
Per valid bench: spike idx = 8k (k 1-24) with F(8k) > f_spike_ms (recomputed from the derived F and checked against
the runner's stage_e1.f_spikes). Per seed: the spike total, the repeated positions (idx spiking in >= repeat_min of
that seed's valid benches; A_in, B_in), E1-D's chain summary (stage_e1.chain_summary) and the agreement of F spikes
and slips, the slip rate (episodes per bench) and the slips by bench position.
Reading (041 v3 §1.6, the first row that holds):
- either seed with fewer than min_spikes spikes, or A_in or B_in empty: 無法判定（觸發太少）;
- A_in and a_prior disjoint: the earlier repeated positions did not recur in the session, 不判定;
- J(A_in, B_in) <= j_content_max: content-locked;
- J >= j_position_min: position- / time-locked;
- otherwise 無法判定.
Beside the reading (041 v3 §1.7): the data it stands on (data_summary: the session's stop and outcomes, planned and
valid benches per seed, the benches left out and why, valid benches without E1-D's quantities); when a seed has fewer
valid benches than repeat_min, a few_spikes row carries a note that the data, not the spikes, are too few. Int-keyed
counts are written as [key, value] pairs.

Usage:
    tools/rtx/fetch_all_sessions.sh stage_e                       # first: every WSL session, without raw/
    python benchmarks/stage_e1p_analysis.py --out results/stage_e/e1p-analysis-<date>/e1p.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import stage_e1 as se1  # noqa: E402
import stage_e1p_plan as pp  # noqa: E402

C = pp.C
READINGS = {"few_spikes": "無法判定（觸發太少）",
            "prior_not_reproduced": "既有的位置集中沒有在 session 內重現，不判定",
            "content_locked": "和 prompt 內容鎖定：換 seed 之後，觸發位置改變",
            "position_locked": "和位置或時間鎖定，與內容無關",
            "undetermined": "無法判定"}


class AnalysisError(RuntimeError):
    pass


# ---------------------------------------------------------------- pure

def spike_idx(reqs: list, f_spike_ms: float) -> list:
    """idx 8k (k 1 .. N / 8 - 1: 1-24 at N 200) whose position-0 F exceeds f_spike_ms."""
    return [C * k for k in range(1, len(reqs) // C) if reqs[C * k]["F_ms"] > f_spike_ms]


def jaccard(a: set, b: set):
    union = a | b
    return len(a & b) / len(union) if union else None


def _pairs(d: dict) -> list:
    """An int-keyed count as sorted [key, value] pairs (JSON keeps them ints)."""
    return [[k, v] for k, v in sorted(d.items())]


def seed_summary(benches: list, cfg: dict, e1cfg: dict) -> dict:
    """benches: [{"tag", "position", "spikes": [idx], "e1": the runner's e1 file or None, "e1_error"}] of one seed's
    valid benches. Episodes (段) and slip groups are counted separately: one group can hold two episodes."""
    e = cfg["e1p"]
    counts = collections.Counter(i for b in benches for i in set(b["spikes"]))
    with_e1 = [b for b in benches if b["e1"] and b["e1"].get("episodes") is not None]
    by_pos, agree = {}, {"spikes": 0, "spikes_in_slip_groups": 0, "slip_groups": 0, "slip_groups_with_spike": 0}
    for b in with_e1:
        ks = {x["k"] for x in b["e1"]["episodes"]}
        agree["spikes"] += len(b["spikes"])
        agree["spikes_in_slip_groups"] += sum(i // C in ks for i in b["spikes"])
        agree["slip_groups"] += len(ks)
        agree["slip_groups_with_spike"] += sum(C * k in b["spikes"] for k in ks)
        by_pos.setdefault(b["position"], []).append(len(b["e1"]["episodes"]))
    n_eps = sum(len(b["e1"]["episodes"]) for b in with_e1)
    chain = se1.chain_summary([b["e1"] for b in with_e1], e1cfg) if with_e1 else None
    if chain:
        chain = {**chain, "start_idx": _pairs(chain["start_idx"])}
    return {"benches": len(benches), "spikes": sum(len(b["spikes"]) for b in benches),
            "spike_idx_counts": _pairs(counts),
            "repeated": [[i, c] for i, c in sorted(counts.items()) if c >= e["repeat_min"]],
            "per_bench": [{"tag": b["tag"], "position": b["position"], "spike_idx": b["spikes"],
                           "episodes": [x["start_idx"] for x in b["e1"]["episodes"]] if b in with_e1 else None,
                           "e1_error": b.get("e1_error")} for b in benches],
            "benches_with_episodes": len(with_e1), "episodes": n_eps,
            "slip_rate": n_eps / len(with_e1) if with_e1 else None,
            "slips_by_position": [{"position": p, "benches": len(v), "episodes": sum(v)} for p, v in
                                  sorted(by_pos.items())],
            "spike_slip": agree, "chain": chain}


def reading(seeds: dict, cfg: dict) -> dict:
    """041 v3 §1.6's table over the two seeds' summaries, in order."""
    e = cfg["e1p"]
    a, b = {i for i, _ in seeds["A"]["repeated"]}, {i for i, _ in seeds["B"]["repeated"]}
    prior = set(e["a_prior"])
    j = jaccard(a, b)
    if any(s["spikes"] < e["min_spikes"] for s in seeds.values()) or not a or not b:
        row = "few_spikes"
    elif not a & prior:
        row = "prior_not_reproduced"
    elif j <= e["j_content_max"]:
        row = "content_locked"
    elif j >= e["j_position_min"]:
        row = "position_locked"
    else:
        row = "undetermined"
    return {"row": row, "text": READINGS[row], "J": j, "A_in": sorted(a), "B_in": sorted(b),
            "A_in_and_prior": sorted(a & prior), "descriptive_only": True}


def data_summary(session: dict, plan: dict, rows: list, cfg: dict) -> dict:
    """What the reading stands on (041 v3 §1.7): the session's stop and outcomes, the planned and valid benches per
    seed, the benches left out with their reasons, the valid ones without E1-D's quantities. insufficient_data: a seed
    with fewer valid benches than repeat_min can have no repeated position, so a few_spikes row then says too little
    data, not too few spikes."""
    benches = [x for boot in plan["boots"] for x in boot["benches"]]
    planned = collections.Counter(x["seed_label"] for x in benches)
    valid = collections.Counter(r["seed_label"] for r in rows if r.get("ok") and not r.get("not_run"))
    by_tag = {r["tag"]: r for r in rows}
    stop = (session.get("stopped") or {}).get("reason")

    def why(x):                                     # a planned bench: its row's reasons, or the stop before its boot
        r = by_tag.get(x["tag"])
        if r is None:
            return {"tag": x["tag"], "not_run": True, "reasons": [f"its boot never started: {stop}"]}
        return {"tag": x["tag"], "not_run": bool(r.get("not_run")), "reasons": r.get("reasons")}

    out = {"stopped": session.get("stopped"), "outcomes": session.get("outcomes"),
           "planned": {x: planned[x] for x in ("A", "B")}, "valid": {x: valid[x] for x in ("A", "B")},
           "excluded": {lab: [why(x) for x in benches if x["seed_label"] == lab
                              and not (by_tag.get(x["tag"], {}).get("ok") and not by_tag[x["tag"]].get("not_run"))]
                        for lab in ("A", "B")},
           "without_e1": {x: [{"tag": r["tag"], "e1_error": r.get("e1_error")} for r in rows
                              if r["seed_label"] == x and r.get("ok") and not r.get("e1")] for x in ("A", "B")}}
    out["insufficient_data"] = any(out["valid"][x] < cfg["e1p"]["repeat_min"] for x in ("A", "B"))
    out["incomplete_session"] = out["stopped"] is not None or out["valid"] != out["planned"]
    return out


# ---------------------------------------------------------------- inputs

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_problems(path, *, pins: dict, cfg: dict, root=ROOT, pins_sha256: str | None = None) -> list:
    """The session is a finished, env-valid formal E1-P session of this harness, config, design (and, when
    pins_sha256 is given, these pins), directly under results/stage_e, not stopped by an environment rule, and its
    rows.jsonl the one it recorded."""
    path = Path(path)
    s, plan = pp._read_json(path / "session.json"), pp._read_json(path / "plan.json")
    if not isinstance(s, dict) or not isinstance(plan, dict):
        return [f"{path}: no readable session.json / plan.json"]
    if (s.get("stage"), s.get("mode"), s.get("status")) != (pp.STAGE, "formal", "finished"):
        return [f"{path.name}: not a finished formal E1-P session"]
    problems = []
    if path.resolve().parent != Path(root).resolve().joinpath(*pp.RESULTS):
        problems.append(f"{path}: not results/stage_e/<session id>")
    if s.get("env_valid") is not True or (s.get("stopped") or {}).get("rule") in pp.ENV_STOPS:
        problems.append(f"{path.name}: env-invalid or stopped by the environment ({s.get('stopped')}): void")
    if s.get("harness_sha256") != pp.harness_sha256() or s.get("plan_sha256") != pp.sha256_json(plan):
        problems.append(f"{path.name}: harness_sha256 / plan_sha256 do not match")
    if s.get("stage_e1p_sha256") != pp.config_sha256(cfg):
        problems.append(f"{path.name}: configs/stage_e1p.toml's content changed since the session")
    if pins_sha256 is not None and (s.get("pins_sha256"), plan.get("pins_sha256")) != (pins_sha256, pins_sha256):
        problems.append(f"{path.name}: ran under another configs/pins.toml (pins_sha256)")
    rows = path / "rows.jsonl"
    if not rows.is_file() or s.get("rows_sha256") != _sha(rows):
        problems.append(f"{path.name}: rows.jsonl is not the one the session recorded (rows_sha256)")
    try:
        want = pp.build_plan(pins, cfg, mode="formal", root=root)
        if pp._without(plan, "pins_sha256") != pp._without(want, "pins_sha256"):
            problems.append(f"{path.name}: plan.json is not build_plan's formal plan")
    except pp.PlanError as exc:
        problems.append(f"{path.name}: build_plan refuses its formal plan ({exc})")
    return problems


def first_valid_session(root=ROOT) -> Path:
    """041 v3 §0: the first valid formal session, after at most one void attempt."""
    attempts = pp.formal_attempts(root)
    for n, a in enumerate(attempts):
        if a["valid"]:
            if n > 1:
                raise AnalysisError(f"{n} void attempts before {a['dir'].name}: at most one rerun (041 v3 §0)")
            return a["dir"]
    raise AnalysisError(f"no valid formal E1-P session under results/stage_e ({[a['dir'].name for a in attempts]})")


def _rows(path: Path) -> list:
    return [json.loads(x) for x in (path / "rows.jsonl").read_text().splitlines() if x.strip()]


def load_benches(path: Path, rows: list, cfg: dict, used: dict) -> tuple:
    """(valid benches by seed label, problems): every valid row's derived requests (and e1 file) present and equal
    to the sha256 its row recorded; a runner spike set that differs from the derived F's is a problem."""
    f = cfg["e1p"]["f_spike_ms"]
    by_seed, problems = {"A": [], "B": []}, []
    for r in rows:
        if not r.get("ok") or r.get("not_run"):
            continue
        dpath = path / r["derived"]
        if not dpath.is_file() or _sha(dpath) != r.get("derived_sha256"):
            problems.append(f"{r['tag']}: {r['derived']} is missing or not the file the row recorded")
            continue
        used[r["derived"]] = r["derived_sha256"]
        reqs = sorted((json.loads(x) for x in dpath.read_text().splitlines() if x.strip()), key=lambda q: q["idx"])
        spikes = spike_idx(reqs, f)
        e1 = None
        if r.get("e1"):
            epath = path / r["e1"]
            if not epath.is_file() or _sha(epath) != r.get("e1_sha256"):
                problems.append(f"{r['tag']}: {r['e1']} is missing or not the file the row recorded")
            else:
                used[r["e1"]] = r["e1_sha256"]
                e1 = json.loads(epath.read_text())
                if [C * g for g in e1["f_spikes"]["spike_groups"]] != spikes:
                    problems.append(f"{r['tag']}: the runner's f_spikes {e1['f_spikes']['spike_groups']} differ from "
                                    f"the derived F's spikes {spikes}")
        by_seed[r["seed_label"]].append({"tag": r["tag"], "position": r["position"], "spikes": spikes, "e1": e1,
                                         "e1_error": r.get("e1_error")})
    return by_seed, problems


def analyze(path: Path, cfg: dict, e1cfg: dict) -> dict:
    used, rows = {}, _rows(path)
    by_seed, problems = load_benches(path, rows, cfg, used)
    if problems:
        raise AnalysisError("; ".join(problems))
    seeds = {x: seed_summary(by_seed[x], cfg, e1cfg) for x in ("A", "B")}
    data = data_summary(pp._read_json(path / "session.json"), pp._read_json(path / "plan.json"), rows, cfg)
    read = reading(seeds, cfg)
    if read["row"] == "few_spikes" and data["insufficient_data"]:
        read["note"] = "a seed has fewer valid benches than repeat_min: too little data, not too few spikes"
    n = sum(s["benches_with_episodes"] for s in seeds.values())
    return {"stage": pp.STAGE, "decision": "041 v3 §1.6", "descriptive_only": True, "session": path.name,
            "reading": read, "data": data, "seeds": seeds,
            "slip_rate_all": sum(s["episodes"] for s in seeds.values()) / n if n else None,
            "earlier_slip_rates": {"C0": 0.21, "Advanced": 0.89, "Core": 2.67,          # 041 v3 §1.6: context only
                                   "note": "episodes per bench in other sessions and on other servers"},
            "stage_e1_harness_sha256": se1.harness_sha256(),
            "inputs_sha256": {**{x: _sha(path / x) for x in ("session.json", "plan.json", "rows.jsonl")}, **used}}


def out_problems(out, root=ROOT) -> list:
    """--out is a file in a new session directory results/stage_e/<id>/ (041 v3 §0: no existing session is changed)."""
    out = Path(out).resolve()
    if out.parent.parent != Path(root).resolve().joinpath(*pp.RESULTS):
        return [f"{out} is not results/stage_e/<new session>/<file>"]
    return [f"{out.parent} already exists (a new session directory only)"] if out.parent.exists() else []


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="results/stage_e/<new id>/e1p.json")
    args = parser.parse_args(argv)
    bad = out_problems(args.out)
    if bad:
        print("stage_e1p_analysis: --out " + "; ".join(bad), file=sys.stderr)
        return 2
    raw = (ROOT / "configs" / "pins.toml").read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(pp.CONFIG.read_text())
    try:
        session = first_valid_session()
        problems = session_problems(session, pins=pins, cfg=cfg, pins_sha256=hashlib.sha256(raw).hexdigest())
        if problems:
            raise AnalysisError("; ".join(problems))
        out = {**analyze(session, cfg, pp.e1_config()), "harness_sha256": pp.harness_sha256(),
               "stage_e1p_sha256": pp.config_sha256(cfg)}
    except (AnalysisError, pp.PlanError) as exc:
        print(f"stage_e1p_analysis: {exc}", file=sys.stderr)
        return 2
    Path(args.out).parent.mkdir(parents=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"session": out["session"], "reading": out["reading"]["row"], "J": out["reading"]["J"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
