#!/usr/bin/env python3
"""Stage E2 analysis (decision 041 v3 §2.2, 2.6, 2.7) of one formal session: 024's quantities with adv_analysis's
frozen functions, every contrast inside the session.

- Per (cell, mix): Y = short-class P95 TTFT of the valid rows; cell_stats needs all 3 restarts (013), else NT.
- U, M, rho = ln(M / U) (rho_stats); against the reference B0 of the same C and K: f = 1 - rho_c / rho_0, d_rho /
  u_rho / d_m / m_up by `exceeds`, the A5 category (removal_category); m = ln(M_0 / M_c), u = ln(U_c / U_0).
- E: category M / P and m >= m_effective; I: category N / R / W, or M / P with m < m_effective ("mostly uniform
  slower"); NT when the category is NT.
- H-C0 gate per reference: core_b_analysis.h_b4_values (>= 20 % and beyond noise).
- beta_ok: beta(B0@25) - beta(B0@8) >= beta_gap (beta: share of 1 Hz samples with waiting > 0, mixed).
- KV_ok: B0@8k's mixed gate-zone share exceeds B0@8's beyond noise and |ln(U_8k / U_8)| <= specificity.
- H-E2k: 024 A7(i) at C 8: a = rho_8k / rho_8 - 1 >= 0.20, u_rho, m_up and the uniform specificity; NT when
  rho_8 <= 0 (the relative dose is undefined).
- H-E2-0: not supported if a reference's H-C0 fails, else NT if one is NT, else supported (statuses kept).
- H-E2e: shared-prefix short-class P95 TTFT with PC on <= 0.8 x off and beyond noise.
- The interpretation table R0-R6 (first match); incomplete when a row's NT input could change the row (every value it
  could take is substituted), for R0c, and for R6 with any NT; beta_ok NT gives its own R2 / R4 reading. R3's path:
  over-commit when T@8k's mixed preemptions > 0 and its mean R > 0, HOL when they are 0 and B0@8k's mean mixed
  HOL-long + KV-short share >= 0.5, else unresolved (NT on a missing input). T256: its note, and the T512-only
  caveat on R3 / R4.
- Descriptive per (cell, mix), mean / SE over the valid restarts: throughput, long-class P95, KV peak, the 1 Hz
  shares, preemptions, R, Lambda, Lq, the B0 tail shares and the unexplained-break share; the allocator bound is
  reported as 024 A13's, 未操弄，未檢驗.
- A formal session is analyzed only if it is the first valid one after at most one void attempt (041 v3 §0); the
  result goes to a new results/stage_e/<id>/ with the inputs' sha256.

Usage:
    python benchmarks/stage_e2_analysis.py --session results/stage_e/<formal> --out results/stage_e/<new>/analysis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import adv_analysis as aa  # noqa: E402
import core_b_analysis as ba  # noqa: E402
import stage_e2_plan as e2p  # noqa: E402

REFERENCE = {"T@25": "B0@25", "T256@25": "B0@25", "T@8": "B0@8", "T@8k": "B0@8k"}
PC_FACTOR = 0.8
HOL_SHARE = 0.5                                         # 041 v3 §2.7: R3's HOL path
DESCRIPTIVE = {"total_token_throughput": ("total_token_throughput",),
               "long_class_p95_ttft_ms": ("long_class_p95_ttft_ms",), "kv_peak": ("kv_peak",),
               "kv_full_while_waiting": ("one_hz", "kv_full_while_waiting"), "gate_zone": ("one_hz", "gate_zone"),
               "preemptions": ("preemptions",), "recompute_tokens": ("recompute_tokens",),
               "budget_full_share": ("budget_full_share",), "lq_short": ("lq_short",),
               "unexplained_break_share": ("decomp", "unexplained_break_share"),
               "q_share_budget_long": ("decomp", "q_shares", "budget-long"),
               "q_share_hol_long": ("decomp", "q_shares", "HOL-long"),
               "q_share_kv_short": ("decomp", "q_shares", "KV-short")}


class AnalysisError(RuntimeError):
    pass


def _values(rows: list, cell: str, mix: str, key: str) -> dict:
    return {r["restart"]: r[key] for r in rows if r["cell"] == cell and r["mix"] == mix and r.get("valid")
            and r.get(key) is not None}


def cell(rows: list, name: str) -> dict:
    U = aa.cell_stats(_values(rows, name, "uniform", "short_class_p95_ttft_ms"))
    M = aa.cell_stats(_values(rows, name, "mixed", "short_class_p95_ttft_ms"))
    rho, rho_se = aa.rho_stats(U, M)
    gz = [r["one_hz"]["gate_zone"] for r in rows if r["cell"] == name and r["mix"] == "mixed" and r.get("valid")
          and (r.get("one_hz") or {}).get("gate_zone") is not None]
    beta = [r["beta"] for r in rows if r["cell"] == name and r["mix"] == "mixed" and r.get("valid")
            and r.get("beta") is not None]
    pre = [r.get("preemptions") for r in rows if r["cell"] == name and r.get("valid")]
    return {"U": U and U["mean"], "U_se": U and U["se"], "M": M and M["mean"], "M_se": M and M["se"], "rho": rho,
            "rho_se": rho_se, "gate_zone": aa.cell_stats(dict(enumerate(gz))) if len(gz) >= aa.RESTARTS else None,
            "beta": statistics.mean(beta) if len(beta) >= aa.RESTARTS else None,
            "preemptions_total": sum(pre) if len(pre) == 2 * aa.RESTARTS and None not in pre else None,
            "n_valid": {mix: len(_values(rows, name, mix, "short_class_p95_ttft_ms")) for mix in e2p.MIX2}}


def _valid(rows: list, name: str, mix: str) -> list:
    return [r for r in rows if r["cell"] == name and r["mix"] == mix and r.get("valid")]


def describe(rows: list, name: str, mix: str) -> dict:
    """041 v3 §2.2's descriptive quantities of one (cell, mix): mean / SE over its 3 valid restarts, None with fewer
    or when one lacks the value; the decomposition's fields only when all 3 pass A10 (024:64, adv_analysis)."""
    v = _valid(rows, name, mix)
    full, dec_ok = len(v) == aa.RESTARTS, len(v) == aa.RESTARTS and all(aa._decomp_ok(r) for r in v)
    return {"n_valid": len(v), **{k: aa._mean_se([aa._field(r, *path) for r in v])
                                  if (dec_ok if path[0] == "decomp" else full) else None
                                  for k, path in DESCRIPTIVE.items()}}


def r3_path(rows: list) -> str:
    """041 v3 §2.7's R3 sub-label from T@8k's and B0@8k's mixed restarts."""
    t = _valid(rows, "T@8k", "mixed")
    if len(t) < aa.RESTARTS or any(r.get("preemptions") is None or r.get("recompute_tokens") is None for r in t):
        return "NT"
    pre, R = sum(r["preemptions"] for r in t), statistics.mean(r["recompute_tokens"] for r in t)
    if pre > 0 and R > 0:
        return "over-commit"
    if pre == 0:
        b = _valid(rows, "B0@8k", "mixed")
        shares = [aa._field(r, "decomp", "q_shares") for r in b]
        if len(shares) < aa.RESTARTS or not all(aa._decomp_ok(r) for r in b) or not all(
                isinstance(q, dict) and q.get("HOL-long") is not None and q.get("KV-short") is not None
                for q in shares):
            return "NT"
        if statistics.mean(q["HOL-long"] + q["KV-short"] for q in shares) >= HOL_SHARE - aa.TOL:
            return "HOL"
    return "unresolved"


def gate(rows: list, name: str, C: int) -> dict:
    per = {2048: {"C": C, "uniform": _values(rows, name, "uniform", "short_class_p95_ttft_ms"),
                  "mixed": _values(rows, name, "mixed", "short_class_p95_ttft_ms")}}
    u = ba.h_b4_values(per)["units"][0]
    if u["status"] != "tested":
        return {"status": "NT"}
    return {**u, "status": "passes" if not u["refutes"] else "fails"}


def versus(c: dict, ref: dict, m_eff: float) -> dict:
    out = {"f": None, "category": "NT", "m": None, "u": None, "verdict": "NT"}
    if ref["U"] and c["U"]:
        out["u"] = math.log(c["U"] / ref["U"])
    if ref["M"] and c["M"]:
        out["m"] = math.log(ref["M"] / c["M"])
    if ref["rho"] is None or c["rho"] is None or ref["rho"] == 0:
        return out
    f = 1 - c["rho"] / ref["rho"]
    d_rho = aa.exceeds(ref["rho"], ref["rho_se"], c["rho"], c["rho_se"])
    u_rho = aa.exceeds(c["rho"], c["rho_se"], ref["rho"], ref["rho_se"])
    d_m = aa.exceeds(ref["M"], ref["M_se"], c["M"], c["M_se"])
    cat = aa.removal_category(f, d_rho, d_m, u_rho)
    if cat in ("M", "P") and out["m"] >= m_eff - aa.TOL:
        verdict = "E"
    else:
        verdict = "I"
    out.update(f=f, d_rho=d_rho, u_rho=u_rho, d_m=d_m, category=cat, verdict=verdict,
               mostly_uniform_slower=cat in ("M", "P") and verdict == "I")
    return out


def analyze(rows: list, cfg: dict) -> dict:
    e = cfg["e2"]
    cells = {name: cell(rows, name) for name, _, _, mixes in e2p.CELLS if mixes == e2p.MIX2}
    gates = {name: gate(rows, name, C) for name, C in (("B0@25", 25), ("B0@8", 8), ("B0@8k", 8))}
    vs = {c: versus(cells[c], cells[r], e["m_effective"]) for c, r in REFERENCE.items()}
    for c, r in REFERENCE.items():
        if gates[r]["status"] != "passes":
            vs[c] = {**vs[c], "verdict": "NT" if gates[r]["status"] == "NT" else "gate fails"}
    b25, b8 = cells["B0@25"]["beta"], cells["B0@8"]["beta"]
    beta_ok = None if b25 is None or b8 is None else b25 - b8 >= e["beta_gap"] - aa.TOL
    g8, g8k = cells["B0@8"]["gate_zone"], cells["B0@8k"]["gate_zone"]
    spec = None
    if cells["B0@8"]["U"] and cells["B0@8k"]["U"]:
        spec = abs(math.log(cells["B0@8k"]["U"] / cells["B0@8"]["U"])) <= e["specificity"] + aa.TOL
    kv_ok = None if g8 is None or g8k is None or spec is None else \
        aa.exceeds(g8k["mean"], g8k["se"], g8["mean"], g8["se"]) and spec
    hk = {"verdict": "NT"}
    c8, c8k = cells["B0@8"], cells["B0@8k"]
    if c8["rho"] is not None and c8["rho"] <= 0:
        hk["why"] = "rho_8 <= 0: the relative dose a = rho_8k / rho_8 - 1 is undefined"
    elif c8["rho"] is not None and c8k["rho"] is not None and spec is not None:
        a = c8k["rho"] / c8["rho"] - 1
        u_rho = aa.exceeds(c8k["rho"], c8k["rho_se"], c8["rho"], c8["rho_se"])
        m_up = aa.exceeds(c8k["M"], c8k["M_se"], c8["M"], c8["M_se"])
        hk = {"a": a, "u_rho": u_rho, "m_up": m_up, "specific": spec,
              "verdict": "supported" if a >= aa.KV_DOSE - aa.TOL and u_rho and m_up and spec else "not supported"}
    off = aa.cell_stats(_values(rows, "PC-off", "shared", "short_class_p95_ttft_ms"))
    on = aa.cell_stats(_values(rows, "PC-on", "shared", "short_class_p95_ttft_ms"))
    he = {"verdict": "NT"} if off is None or on is None else {
        "Y_off": off["mean"], "Y_on": on["mean"], "ratio": on["mean"] / off["mean"],
        "verdict": "supported" if on["mean"] <= PC_FACTOR * off["mean"] and aa.exceeds(off["mean"], off["se"],
                                                                                       on["mean"], on["se"])
        else "not supported"}
    statuses = {r: gates[r]["status"] for r in gates}
    h0 = "not supported" if "fails" in statuses.values() else "NT" if "NT" in statuses.values() else "supported"
    hyp = {"H-E2-0": {"verdict": h0, "by_reference": statuses},
           "H-E2-R": _verdict(vs["T@25"], want="I"), "H-E2a": _verdict(vs["T@8"], want="E"),
           "H-E2b": _verdict(vs["T@8k"], want="I"), "H-E2c": _verdict(vs["T256@25"], want="E"), "H-E2k": hk,
           "H-E2e": he}
    out = {"cells": cells, "gates": gates, "versus": vs, "beta_ok": beta_ok, "kv_ok": kv_ok, "hypotheses": hyp,
           "interaction_I_KV": _interaction(cells),
           "descriptive": {f"{name}/{mix}": describe(rows, name, mix) for name, _, _, mixes in e2p.CELLS
                           for mix in mixes},
           "allocator": {"status": aa.ALLOCATOR_STATUS, "waste_upper_bound": aa.ALLOCATOR_WASTE_BOUND,
                         "source": "024 A13"}}
    out["row"] = interpret(out, r3=r3_path(rows))
    return out


def _verdict(v: dict, *, want: str) -> dict:
    if v["verdict"] in ("NT", "gate fails"):
        return {"verdict": "NT", "why": v["verdict"]}
    return {"verdict": "supported" if v["verdict"] == want else "not supported", "E_or_I": v["verdict"]}


def _interaction(cells: dict):
    keys = ("B0@8", "T@8", "B0@8k", "T@8k")
    if any(cells[k]["rho"] is None for k in keys):
        return None
    i = (cells["B0@8"]["rho"] - cells["T@8"]["rho"]) - (cells["B0@8k"]["rho"] - cells["T@8k"]["rho"])
    return {"value": i, "se": math.sqrt(sum(cells[k]["rho_se"] ** 2 for k in keys))}


def table(g: dict, ev: dict, kv_ok) -> str:
    """041 v3 §2.7's first matching row for given reference statuses, verdicts and KV_ok."""
    if g["B0@25"] == "fails":
        return "R0"
    if ev["T@25"] == "E":
        return "R0b"
    if "NT" in (g["B0@25"], ev["T@25"]) or ev["T@25"] == "gate fails":
        return "R0c"
    if g["B0@8"] == "fails":
        return "R1"
    if ev["T@8"] == "I":
        return "R2"
    if ev["T@8"] == "E" and kv_ok and ev["T@8k"] == "I":
        return "R3"
    if ev["T@8"] == "E" and kv_ok and ev["T@8k"] == "E":
        return "R4"
    if ev["T@8"] == "E" and (g["B0@8k"] == "fails" or kv_ok is False):
        return "R5"
    return "R6"


def _options(status: str, verdict: str) -> list:
    """Every determined (reference status, verdict) an NT pair could turn out to be."""
    if status == "NT":
        return [("passes", "E"), ("passes", "I"), ("fails", "gate fails")]
    if verdict == "NT":
        return [(status, "E"), (status, "I")]
    return [(status, verdict)]


def possible_rows(g: dict, ev: dict, kv_ok) -> set:
    out = set()
    for s25, v25 in _options(g["B0@25"], ev["T@25"]):
        for s8, v8 in _options(g["B0@8"], ev["T@8"]):
            for s8k, v8k in _options(g["B0@8k"], ev["T@8k"]):
                for kv in ((True, False) if kv_ok is None else (kv_ok,)):
                    out.add(table({"B0@25": s25, "B0@8": s8, "B0@8k": s8k},
                                  {"T@25": v25, "T@8": v8, "T@8k": v8k}, kv))
    return out


READINGS = {
    "R0": "C 25 did not reproduce H-B4 in the session; no attribution",
    "R0b": "025's N was not reproduced in the session; E2's question does not arise",
    "R0c": "incomplete",
    "R1": "no 2K penalty at C 8; neither the load nor the KV contrast can be tested",
    "R2": {True: "T fails at C 8 too, with less load and KV unconstrained: neither S nor V is necessary",
           False: "T fails at C 8 too; the load difference is not enough",
           None: "T fails at C 8 too; beta_ok NT, the load difference is not readable"},
    "R3": "KV pressure alone makes T fail at C 8's load: supports V; S not necessary",
    "R4": {True: "C 8's rebuilt KV structure is not enough; C 25's failure needs a C-related factor; consistent with S "
                 "(not separable from a load x KV interaction)",
           False: "C 8's rebuilt KV structure is not enough; C 25's failure needs a C-related factor; S not "
                  "interpreted (beta_ok fails)",
           None: "C 8's rebuilt KV structure is not enough; C 25's failure needs a C-related factor; S not "
                 "interpreted (beta_ok NT)"},
    "R5": "the KV contrast fails; only 'T works at C 8'",
    "R6": "other: reported, not interpreted",
}
T512_ONLY = "the mechanism reading applies to T512 only"


def interpret(a: dict, r3: str = "NT") -> dict:
    g = {k: v["status"] for k, v in a["gates"].items()}
    ev = {k: x["verdict"] for k, x in a["versus"].items()}
    row = table(g, ev, a["kv_ok"])
    reading = READINGS[row]
    if isinstance(reading, dict):
        reading = reading[a["beta_ok"]]
    nt = "NT" in g.values() or any(v == "NT" for k, v in ev.items() if k != "T256@25") or a["kv_ok"] is None
    incomplete = row == "R0c" or len(possible_rows(g, ev, a["kv_ok"])) > 1 or (row == "R6" and nt) or \
        (row in ("R2", "R4") and a["beta_ok"] is None)
    out = {"row": row, "reading": reading, "incomplete": incomplete}
    if row == "R3":
        out["path"] = r3
    note = ev["T256@25"]
    out["t256_note"] = {"E": "a smaller threshold works at C 25: T512's failure depends on the threshold size (matches "
                             "neither S nor V)", "I": "no evidence on the threshold size (T256 has its own costs)"}.get(
        note, "incomplete")
    if note == "E" and row in ("R3", "R4"):
        out["t512_only"] = T512_ONLY
    return out


def session_problems(path, *, pins: dict, cfg: dict, root=ROOT) -> list:
    """The session is a finished, env-valid formal E2 session of this harness, config and design, not stopped, directly
    under results/stage_e, and the first valid one after at most one void attempt (041 v3 §0)."""
    path = Path(path)
    s, plan = e2p._read_json(path / "session.json"), e2p._read_json(path / "plan.json")
    if not isinstance(s, dict) or not isinstance(plan, dict):
        return [f"{path}: no readable session.json / plan.json"]
    problems = []
    if (s.get("stage"), s.get("mode"), s.get("status")) != (e2p.STAGE, "formal", "finished"):
        return [f"{path.name}: not a finished formal E2 session"]
    if path.resolve().parent != Path(root).resolve().joinpath(*e2p.RESULTS):
        problems.append(f"{path}: not results/stage_e/<session id>")
    if s.get("env_valid") is not True or s.get("stopped"):
        problems.append(f"{path.name}: env-invalid or stopped ({s.get('stopped')}): void, one rerun (041 v3 §0)")
    if s.get("harness_sha256") != e2p.harness_sha256() or s.get("plan_sha256") != e2p.sha256_json(plan):
        problems.append(f"{path.name}: harness_sha256 / plan_sha256 do not match")
    if s.get("stage_e2_sha256") != e2p.config_sha256(cfg):
        problems.append(f"{path.name}: configs/stage_e2.toml's content changed since the session")
    try:
        want = e2p.build_plan(pins, cfg, mode="formal", root=root)
        if e2p._without(plan, "pins_sha256") != e2p._without(want, "pins_sha256"):
            problems.append(f"{path.name}: plan.json is not build_plan's formal plan")
    except e2p.PlanError as exc:
        problems.append(f"{path.name}: build_plan refuses its formal plan ({exc})")
    earlier = [x for x in e2p.formal_attempts(root)
               if x["dir"].resolve() != path.resolve() and x["started_at"] < (s.get("started_at") or "")]
    if len(earlier) > 1 or any(x["valid"] for x in earlier):
        problems.append(f"{path.name}: earlier formal E2 attempts {[x['dir'].name for x in earlier]} (only the first "
                        "valid session, after at most one void attempt, is analyzed)")
    return problems


def out_problems(out, root=ROOT) -> list:
    """--out is a file in a new session directory results/stage_e/<id>/ (041 v3 §0: no existing session is changed)."""
    out = Path(out).resolve()
    if out.parent.parent != Path(root).resolve().joinpath(*e2p.RESULTS):
        return [f"{out} is not results/stage_e/<new session>/<file>"]
    return [f"{out.parent} already exists (a new session directory only)"] if out.parent.exists() else []


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", required=True, help="results/stage_e/<new id>/analysis.json")
    args = parser.parse_args(argv)
    session = Path(args.session)
    bad = out_problems(args.out)
    if bad:
        print("stage_e2_analysis: --out " + "; ".join(bad), file=sys.stderr)
        return 2
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    cfg = tomllib.loads(e2p.CONFIG.read_text())
    problems = session_problems(session, pins=pins, cfg=cfg)
    if problems:
        print("stage_e2_analysis: " + "; ".join(problems), file=sys.stderr)
        return 2
    rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines() if x.strip()]
    out = {**analyze(rows, cfg), "session": session.name, "harness_sha256": e2p.harness_sha256(),
           "inputs_sha256": {n: hashlib.sha256((session / n).read_bytes()).hexdigest()
                             for n in ("session.json", "plan.json", "rows.jsonl")}}
    Path(args.out).parent.mkdir(parents=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"row": out["row"]["row"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
