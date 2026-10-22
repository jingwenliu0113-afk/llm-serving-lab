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
- H-E2k: 024 A7(i) at C 8: a = rho_8k / rho_8 - 1 >= 0.20, u_rho, m_up and the uniform specificity.
- H-E2e: shared-prefix short-class P95 TTFT with PC on <= 0.8 x off and beyond noise.
- The interpretation table R0-R6 (first match), R3's sub-labels and the T256 note.
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
    pre = [r["preemptions"] for r in rows if r["cell"] == name and r.get("valid") and r.get("preemptions") is not None]
    return {"U": U and U["mean"], "U_se": U and U["se"], "M": M and M["mean"], "M_se": M and M["se"], "rho": rho,
            "rho_se": rho_se, "gate_zone": aa.cell_stats(dict(enumerate(gz))) if len(gz) >= aa.RESTARTS else None,
            "beta": statistics.mean(beta) if len(beta) >= aa.RESTARTS else None,
            "preemptions_total": sum(pre) if pre else None,
            "n_valid": {mix: len(_values(rows, name, mix, "short_class_p95_ttft_ms")) for mix in e2p.MIX2}}


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
    if c8["rho"] and c8k["rho"] is not None and spec is not None:
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
    hyp = {"H-E2-0": {r: gates[r]["status"] for r in gates},
           "H-E2-R": _verdict(vs["T@25"], want="I"), "H-E2a": _verdict(vs["T@8"], want="E"),
           "H-E2b": _verdict(vs["T@8k"], want="I"), "H-E2c": _verdict(vs["T256@25"], want="E"), "H-E2k": hk,
           "H-E2e": he}
    out = {"cells": cells, "gates": gates, "versus": vs, "beta_ok": beta_ok, "kv_ok": kv_ok, "hypotheses": hyp,
           "interaction_I_KV": _interaction(cells)}
    out["row"] = interpret(out)
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


def interpret(a: dict) -> dict:
    g, v = a["gates"], a["versus"]
    ev = {k: x["verdict"] for k, x in v.items()}
    if g["B0@25"]["status"] == "fails":
        row = ("R0", "C 25 did not reproduce H-B4 in the session; no attribution")
    elif ev["T@25"] == "E":
        row = ("R0b", "025's N was not reproduced in the session; E2's question does not arise")
    elif "NT" in (g["B0@25"]["status"], ev["T@25"]) or ev["T@25"] == "gate fails":
        row = ("R0c", "incomplete")
    elif g["B0@8"]["status"] == "fails":
        row = ("R1", "no 2K penalty at C 8; neither the load nor the KV contrast can be tested")
    elif ev["T@8"] == "I":
        row = ("R2", "T fails at C 8 too, with less load and KV unconstrained: neither S nor V is necessary"
               if a["beta_ok"] else "T fails at C 8 too; the load difference is not enough")
    elif ev["T@8"] == "E" and a["kv_ok"] and ev["T@8k"] == "I":
        row = ("R3", "KV pressure alone makes T fail at C 8's load: supports V; S not necessary")
    elif ev["T@8"] == "E" and a["kv_ok"] and ev["T@8k"] == "E":
        row = ("R4", "C 8's rebuilt KV structure is not enough; C 25's failure needs a C-related factor"
               + ("; consistent with S (not separable from a load x KV interaction)" if a["beta_ok"] else
                  "; S not interpreted (beta_ok fails)"))
    elif ev["T@8"] == "E" and (g["B0@8k"]["status"] == "fails" or a["kv_ok"] is False):
        row = ("R5", "the KV contrast fails; only 'T works at C 8'")
    else:
        row = ("R6", "other: reported, not interpreted")
    nt = any(x in ("NT", "gate fails") for x in ev.values())
    out = {"row": row[0], "reading": row[1], "incomplete": nt and row[0] not in ("R0c",)}
    if row[0] == "R3":
        t8k = a["cells"]["T@8k"]
        out["path"] = "over-commit" if (t8k["preemptions_total"] or 0) > 0 else "unresolved"
    note = ev["T256@25"]
    out["t256_note"] = {"E": "a smaller threshold works at C 25: T512's failure depends on the threshold size (matches "
                             "neither S nor V)", "I": "no evidence on the threshold size (T256 has its own costs)"}.get(
        note, "incomplete")
    return out


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    args = parser.parse_args(argv)
    session = Path(args.session)
    s = json.loads((session / "session.json").read_text())
    cfg_raw = e2p.CONFIG.read_bytes()
    if (s.get("mode"), s.get("env_valid"), s.get("harness_sha256"), s.get("stage_e2_sha256")) != \
            ("formal", True, e2p.harness_sha256(), hashlib.sha256(cfg_raw).hexdigest()):
        print("stage_e2_analysis: not a formal, env-valid session of this harness and configs/stage_e2.toml",
              file=sys.stderr)
        return 2
    rows = [json.loads(x) for x in (session / "rows.jsonl").read_text().splitlines() if x.strip()]
    out = analyze(rows, tomllib.loads(cfg_raw.decode()))
    (session / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"row": out["row"]["row"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
