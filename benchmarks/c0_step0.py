#!/usr/bin/env python3
"""C0 step 0 (decision 028 「步驟 0」): desk checks S1-S5 over Phase 1 raw files, no GPU. Descriptive only.

Committed before any raw file is fetched, so S1 is a replication of 028 clue 6 and not a rule written after seeing the
data. The raw files come read-only from WSL through tools/rtx/fetch_raw.sh into a git-ignored directory (default
results/capacity/c0-step0/raw/<session id>/); every file is checked against its session's committed
raw_manifest.json first, and a missing or mismatching file is recorded and left out (028: a missing archive is
recorded and the probe goes ahead; C0's purpose does not need the mechanism).

- S1 (Core formal-20261009-coreb-uniform, bench-uniform-8192-r{1,2,3}-c{4,7,8,9,13}.json): the c0_analysis classifier
  on every run (episodes with start group k, position and length; mode). Pre-stated replication: the share of the
  C = 8 episodes that start at idx = 1 (mod 64), i.e. in groups 8, 16, 24 of 1..24, against chance 1 / 8 (with the
  binomial upper tail). C = 9 starts are reported descriptively only: 9k + 1 is never 1 (mod 64) below N = 200, so a
  chance rate of 1 / 8 does not apply to them (028 implementation note).
- S2 (Core C = 4, C = 7): is an elevated run (Y - m_A > 170 ms) slip-driven, i.e. has episodes and falls to
  Y - m_A <= 170 ms once its slipped requests are left out of the short-class P95?
- S3 (Advanced formal-20261011-adv, <tag>.steps.jsonl of B0 / G / K3 at 8192 uniform): for every episode, the steps
  of its group (arrival(8k) <= t < arrival(8k + 8)) against the neighbouring non-slip groups k - 1 and k + 1; the
  extra queue time (median Q of the slipped requests minus that of the same positions in the neighbours) over the
  median prefill step of the slip group.
- S4 (Advanced <tag>.server.log, <tag>.bench.log): the server log lines within +-1 s of arrival(8k + 1) by kind, slip
  groups against the other groups, and the non-iteration lines near slips verbatim; startup lines
  (run_c0_probe.startup_lines, normalised) only in the high runs or only in the low runs; WARNING / ERROR lines of the
  client logs.
- S5 (Core server-uniform-8192-r{1,2,3}.log): startup lines only in r2 (about 1 % slower at every C) or missing from
  r2 while in both r1 and r3.

Writes <out>/step0.json (default results/capacity/c0-step0/); refuses an --out inside a Phase 1 results directory or
an input session, and never writes an analysis.json or a verdict (027 §1).

Usage (Mac):
    tools/rtx/fetch_raw.sh core_b formal-20261009-coreb-uniform \\
        $(.venv/bin/python benchmarks/c0_step0.py --list-files core)
    tools/rtx/fetch_raw.sh advanced formal-20261011-adv $(.venv/bin/python benchmarks/c0_step0.py --list-files adv)
    .venv/bin/python benchmarks/c0_step0.py

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import c0_analysis as ca
import core_b_analysis as ba
import run_c0_probe as rp

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent
CORE_SID = "formal-20261009-coreb-uniform"
ADV_SID = "formal-20261011-adv"
L = 8192
RESTARTS = (1, 2, 3)
CORE_C = (4, 7, 8, 9, 13)
S2_C = (4, 7)
ADV_TAGS = tuple(f"{cond}-{L}-uniform-r{r}" for cond in ("B0", "G", "K3") for r in RESTARTS)
ADV_C = 8  # the Advanced 8K C_knee
REPLICATION_C = (8,)
REPLICATION_MOD, REPLICATION_RESIDUE, REPLICATION_CHANCE = 64, 1, 1 / 8
LOG_WINDOW_S = 1.0
LINE_CAP = 20
DEFAULT_RAW = ROOT / "results" / "capacity" / "c0-step0" / "raw"
_RE_STAMPED = re.compile(r"^(\d{10,}) (.*)$")
_RE_CLIENT_FLAG = re.compile(r"WARNING|ERROR|Traceback|Exception")


def core_bench_name(r: int, C: int) -> str:
    return f"bench-uniform-{L}-r{r}-c{C}.json"


def core_log_name(r: int) -> str:
    return f"server-uniform-{L}-r{r}.log"


def core_files() -> list:
    return [core_bench_name(r, C) for r in RESTARTS for C in CORE_C] + [core_log_name(r) for r in RESTARTS]


def adv_files() -> list:
    return [f"{tag}.{ext}" for tag in ADV_TAGS for ext in ("steps.jsonl", "server.log", "bench.log")]


def _read_jsonl(path: Path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _median(values):
    v = [x for x in values if x is not None]
    return ba.percentile(v, 50) if v else None


def verify_raw(session_dir, raw_dir, names: list) -> dict:
    """name -> "ok", "missing", "sha256 mismatch" or "not in manifest" (the session's committed raw_manifest.json)."""
    path = Path(session_dir) / "raw_manifest.json"
    files = json.loads(path.read_text())["files"] if path.is_file() else {}
    out = {}
    for name in names:
        want, local = files.get(f"raw/{name}"), Path(raw_dir) / name
        if want is None:
            out[name] = "not in manifest"
        elif not local.is_file():
            out[name] = "missing"
        elif hashlib.sha256(local.read_bytes()).hexdigest() != want["sha256"]:
            out[name] = "sha256 mismatch"
        else:
            out[name] = "ok"
    return out


# ---------------------------------------------------------------- S1 / S2 (Core bench JSONs)

def _core_run(raw: Path, status: dict, r: int, C: int) -> dict:
    name = core_bench_name(r, C)
    rec = {"restart": r, "C": C, "file": name, "status": status.get(name, "missing")}
    if rec["status"] != "ok":
        return rec
    res = json.loads((raw / name).read_text())
    lens = res["input_lens"]
    cls = ca.classify_bench([t * 1000 for t in res["ttfts"]], C=C, L=L, input_lens=lens)
    gone = set(cls["slipped"])
    keep = [i for i in range(len(lens)) if i not in gone]
    y_wo = ba.short_class_p95_ttft_ms({"ttfts": [res["ttfts"][i] for i in keep],
                                       "input_lens": [lens[i] for i in keep]}, L=L)
    rec.update(cls, y_without_slipped_ms=y_wo)
    return rec


def s1_core(raw, status: dict) -> dict:
    runs = [_core_run(Path(raw), status, r, C) for r in RESTARTS for C in CORE_C]
    starts = sorted(e["start_idx"] for x in runs if x["status"] == "ok" and x["C"] in REPLICATION_C
                    for e in x["episodes"])
    at = sum(i % REPLICATION_MOD == REPLICATION_RESIDUE for i in starts)
    return {"runs": runs, "replication": {
        "C": list(REPLICATION_C), "mod": REPLICATION_MOD, "residue": REPLICATION_RESIDUE, "start_idx": starts,
        "episodes": len(starts), "at": at, "share": at / len(starts) if starts else None,
        "chance": REPLICATION_CHANCE, "binomial_tail_p": ca.binomial_tail(len(starts), at, REPLICATION_CHANCE)},
        "c9_start_idx": sorted(e["start_idx"] for x in runs if x["status"] == "ok" and x["C"] == 9
                               for e in x["episodes"])}


def s2_core(s1: dict) -> dict:
    out = []
    for x in s1["runs"]:
        if x["C"] not in S2_C:
            continue
        rec = {"restart": x["restart"], "C": x["C"], "status": x["status"]}
        if x["status"] != "ok":
            out.append({**rec, "elevated": None, "slip_driven": None})
            continue
        d, d_wo = x["y_ms"] - x["m_a_ms"], x["y_without_slipped_ms"] - x["m_a_ms"]
        elevated = x["mode"] != "low"
        out.append({**rec, "mode": x["mode"], "n_episodes": x["n_episodes"], "n_slipped": x["n_slipped"],
                    "y_minus_m_a_ms": d, "y_without_slipped_minus_m_a_ms": d_wo,
                    "p95_excl_first_c_minus_m_a_ms": x["p95_excl_first_c_ms"] - x["m_a_ms"], "elevated": elevated,
                    "slip_driven": elevated and x["n_episodes"] > 0 and d_wo <= ca.HIGH_MS})
    return {"runs": out}


# ---------------------------------------------------------------- S3 / S4 (Advanced steps and logs)

def _adv_requests(session: Path, tag: str) -> tuple:
    reqs = sorted(_read_jsonl(session / "derived" / f"{tag}-requests.jsonl"), key=lambda r: r["idx"])
    cls = ca.classify_bench([r["client_ttft_ms"] for r in reqs], C=ADV_C, L=L,
                            input_lens=[r["prompt_tokens"] for r in reqs])
    return reqs, cls


def _group_steps(steps: list, reqs: list, k: int) -> dict:
    t0 = reqs[ADV_C * k]["arrival"]
    t1 = reqs[ADV_C * (k + 1)]["arrival"] if ADV_C * (k + 1) < len(reqs) else math.inf
    win = [s for s in steps if t0 <= s["t"] < t1]
    ms = [s["elapsed_ms"] for s in win]
    return {"n_steps": len(win), "n_prefill_steps": sum(s["ctx_tok"] > 0 for s in win), "step_ms_median": _median(ms),
            "prefill_step_ms_median": _median(s["elapsed_ms"] for s in win if s["ctx_tok"] > 0),
            "step_ms_max": max(ms, default=None), "step_ms_sum": sum(ms)}


def s3_steps(adv_session, adv_raw, status: dict) -> dict:
    session, raw = Path(adv_session), Path(adv_raw)
    episodes, missing = [], []
    for tag in ADV_TAGS:
        name = f"{tag}.steps.jsonl"
        if status.get(name) != "ok":
            missing.append(tag)
            continue
        reqs, cls = _adv_requests(session, tag)
        steps = _read_jsonl(raw / name)
        slip_k = {e["k"] for e in cls["episodes"]}
        for e in cls["episodes"]:
            k, idx = e["k"], range(e["start_idx"], e["start_idx"] + e["length"])
            positions = {i % ADV_C for i in idx}
            neigh = [n for n in (k - 1, k + 1) if n >= 1 and ADV_C * n < len(reqs) and n not in slip_k]
            q_slip = _median(reqs[i].get("Q_ms") for i in idx)
            q_base = _median(reqs[ADV_C * n + p].get("Q_ms") for n in neigh for p in positions
                             if ADV_C * n + p < len(reqs))
            extra = q_slip - q_base if q_slip is not None and q_base is not None else None
            group = _group_steps(steps, reqs, k)
            step = group["prefill_step_ms_median"]
            episodes.append({"tag": tag, "k": k, "start_idx": e["start_idx"], "length": e["length"],
                             "slip_group": group, "neighbours": {str(n): _group_steps(steps, reqs, n) for n in neigh},
                             "q_slipped_median_ms": q_slip, "q_neighbours_median_ms": q_base, "extra_q_ms": extra,
                             "extra_q_over_step": extra / step if extra is not None and step else None})
    return {"episodes": episodes, "missing": missing}


def line_kind(text: str) -> str:
    if "Iteration(" in text:
        return "iteration"
    if "Avg prompt throughput" in text or "Avg generation throughput" in text:
        return "stats"
    if '"POST ' in text or "POST /v1" in text:
        return "access"
    if re.search(r"\bERROR\b|Traceback", text):
        return "error"
    if re.search(r"\bWARNING\b", text):
        return "warning"
    return "other"


LINE_KINDS = ("iteration", "stats", "access", "warning", "error", "other")


def _stamped(text: str) -> list:
    out = []
    for line in text.splitlines():
        m = _RE_STAMPED.match(line)
        if m:
            out.append((int(m.group(1)) / 1000, m.group(2)))
    return sorted(out, key=lambda x: x[0])


def _compare(facts: dict, left: list, right: list) -> tuple:
    """Per startup category, the normalised lines in some `left` log but no `right` log, and the reverse."""
    if not left or not right:
        return None, None

    def union(tags, cat):
        return {rp.normalize_line(x) for t in tags for x in facts[t]["lines"][cat]}

    cats = list(rp.STARTUP_PATTERNS)
    return ({c: sorted(union(left, c) - union(right, c)) for c in cats},
            {c: sorted(union(right, c) - union(left, c)) for c in cats})


def _facts_summary(f: dict) -> dict:
    return {k: f[k] for k in ("counts", "nondefault", "log_facts", "available_kv_gib")}


def s4_logs(adv_session, adv_raw, status: dict) -> dict:
    session, raw = Path(adv_session), Path(adv_raw)
    with_ = {k: {"slip": 0, "non_slip": 0} for k in LINE_KINDS}
    lines, n_slip, n_other, facts, high, bench_logs = [], 0, 0, {}, [], {}
    for tag in ADV_TAGS:
        if status.get(f"{tag}.server.log") == "ok":
            text = (raw / f"{tag}.server.log").read_text(errors="replace")
            reqs, cls = _adv_requests(session, tag)
            slip_k = {e["k"] for e in cls["episodes"]}
            stamped = _stamped(text)
            times = [t for t, _ in stamped]
            for k in range(1, (len(reqs) - 2) // ADV_C + 1):  # every group with an 8k + 1
                m = reqs[ADV_C * k + 1].get("arrival")
                if m is None:
                    continue
                near = stamped[bisect.bisect_left(times, m - LOG_WINDOW_S):bisect.bisect_right(times, m + LOG_WINDOW_S)]
                side = "slip" if k in slip_k else "non_slip"
                for kind in {line_kind(x) for _, x in near}:
                    with_[kind][side] += 1
                if k in slip_k:
                    n_slip += 1
                    lines += [{"tag": tag, "k": k, "offset_s": round(t - m, 3), "line": x} for t, x in near
                              if line_kind(x) not in ("iteration", "access")][:LINE_CAP]
                else:
                    n_other += 1
            facts[tag] = rp.startup_facts(text)
            if cls["episodes"]:
                high.append(tag)
        if status.get(f"{tag}.bench.log") == "ok":
            flagged = [x for x in (raw / f"{tag}.bench.log").read_text(errors="replace").splitlines()
                       if _RE_CLIENT_FLAG.search(x)]
            if flagged:
                bench_logs[tag] = flagged[:LINE_CAP]
    low = [t for t in facts if t not in high]
    only_high, only_low = _compare(facts, high, low)
    return {"near_slips": {"window_s": LOG_WINDOW_S, "slip_groups": n_slip, "non_slip_groups": n_other,
                           "with": with_, "lines": lines},
            "startup": {"high": high, "low": low, "only_in_high": only_high, "only_in_low": only_low,
                        "facts": {t: _facts_summary(f) for t, f in facts.items()}},
            "bench_logs": bench_logs}


# ---------------------------------------------------------------- S5 (Core server logs)

def s5_core_startup(core_raw, status: dict) -> dict:
    raw, facts, st = Path(core_raw), {}, {}
    for r in RESTARTS:
        name = core_log_name(r)
        st[f"r{r}"] = status.get(name, "missing")
        if st[f"r{r}"] == "ok":
            facts[f"r{r}"] = rp.startup_facts((raw / name).read_text(errors="replace"))
    others = [t for t in ("r1", "r3") if t in facts]
    only, missing = (None, None)
    if "r2" in facts and others:
        def norm(t, cat):
            return {rp.normalize_line(x) for x in facts[t]["lines"][cat]}

        cats = list(rp.STARTUP_PATTERNS)
        only = {c: sorted(norm("r2", c) - set().union(*(norm(t, c) for t in others))) for c in cats}
        missing = {c: sorted(set.intersection(*(norm(t, c) for t in others)) - norm("r2", c)) for c in cats}
    return {"status": st, "only_in_r2": only, "missing_in_r2": missing,
            "facts": {t: _facts_summary(f) for t, f in facts.items()}}


# ---------------------------------------------------------------- driver

def run_step0(*, core_session, core_raw, adv_session, adv_raw) -> dict:
    core_status = verify_raw(core_session, core_raw, core_files())
    adv_status = verify_raw(adv_session, adv_raw, adv_files())
    s1 = s1_core(core_raw, core_status)
    return {"schema_version": SCHEMA_VERSION, "decision": "028", "descriptive_only": True,
            "note": "028 step 0: descriptive desk checks on Phase 1 raw files; no verdict or analysis.json changes",
            "inputs": {"core_session": Path(core_session).name, "adv_session": Path(adv_session).name},
            "files": {"core": core_status, "adv": adv_status}, "S1": s1, "S2": s2_core(s1),
            "S3": s3_steps(adv_session, adv_raw, adv_status), "S4": s4_logs(adv_session, adv_raw, adv_status),
            "S5": s5_core_startup(core_raw, core_status)}


def out_problems(out, *, sessions) -> list:
    """run_c0_probe's Phase 1 directory rule, plus the input sessions."""
    return rp.out_problems(out) + [f"{out} is inside {s}" for s in sessions if rp.inside(out, s)]


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list-files", choices=["core", "adv"], help="print the raw files to fetch and exit")
    parser.add_argument("--core-session", default=str(ROOT / "results" / "core_b" / CORE_SID))
    parser.add_argument("--adv-session", default=str(ROOT / "results" / "advanced" / ADV_SID))
    parser.add_argument("--core-raw", default=str(DEFAULT_RAW / CORE_SID))
    parser.add_argument("--adv-raw", default=str(DEFAULT_RAW / ADV_SID))
    parser.add_argument("--out", default=str(ROOT / "results" / "capacity" / "c0-step0"))
    args = parser.parse_args(argv)
    if args.list_files:
        print("\n".join(core_files() if args.list_files == "core" else adv_files()))
        return 0
    bad = out_problems(args.out, sessions=(args.core_session, args.adv_session))
    if bad:
        print(f"c0_step0: refusing --out: {'; '.join(bad)} (028: step 0 never writes into Phase 1 directories)",
              file=sys.stderr)
        return 2
    res = run_step0(core_session=args.core_session, core_raw=args.core_raw, adv_session=args.adv_session,
                    adv_raw=args.adv_raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "step0.json").write_text(json.dumps(res, indent=1) + "\n")
    files = {k: sum(v == "ok" for v in res["files"][k].values()) for k in ("core", "adv")}
    print(json.dumps({"files_ok": files, "replication": res["S1"]["replication"],
                      "s3_episodes": len(res["S3"]["episodes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
