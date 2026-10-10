#!/usr/bin/env python3
"""Pre-registered Advanced analysis: isolating the cause of H-B4 (decision 024 A1-A13 and the
interpretation matrix R0-R7; PREREGISTRATION §7). Written and tested before any Advanced data;
frozen with freeze.advanced.

Inputs: the Advanced session {"session", "plan", "rows"} (run_adv.py, one row per boot) and the
formal Core `uniform` session, from which C is recomputed with core_b_analysis.knee and must equal
the plan's (and the frozen) C = 25 / 8. Unit = server restart (n = 3).
- A1: a row counts only if ok, not timed out or skipped, its log facts and num_gpu_blocks match the
  condition table and (T / TG) no step failed the chunk check. A cell (condition, L, mix) without 3
  valid restarts is NT; nothing is substituted (013). Invalid runs are never evidence.
- A2 / A3: U, M = mean of the 3 short-class P95 TTFTs, SE = s / sqrt(3); rho = ln(M / U) with the §2
  delta-method SE; "beyond noise" is |x - y| > 2 sqrt(SE_x^2 + SE_y^2) (strict, as in §2).
- A4: H-C0 calls core_b_analysis.h_b4_values on B0; a level where it does not pass is R0 and
  H-C1..H-C5 are uninterpretable there. An H-C0 that is not testable (B0 without 3 valid restarts,
  A1 / 013) is not "H-C0 failed": the level keeps row R0 but gets its own not-testable conclusion
  and is incomplete, and cross-L reports such a level as not testable, never as agreement.
  A5-A7, A11 and the matrix (first match per L, the incomplete / H-C5 / allocator modifiers, the
  cross-L rule) follow 024 verbatim; R4 claims H-C3 only when H-C3 is supported.
- Thresholds are inclusive and a boundary value goes to the higher category (A5); TOL = 1e-9 only
  absorbs floating-point rounding of an exact boundary (e.g. 1 - ln16/ln32 != 0.2 in binary).
- A13 descriptive quantities (g, interaction, long-class TTFT, throughput, KV peak, tail O / P,
  histogram queue / prefill means, 1 Hz KV-full-while-waiting and gate-zone shares, residual-BM and
  unexplained-break shares, allocator waste bound), the A8 >= P95 tail subset and the client
  descriptives (short-class P50 / P99 TTFT, long-class P50 TTFT, short-class P95 TPOT) are reported
  per (condition, L, mix) as mean / SE over the valid restarts and never read by a verdict.

Usage (adv_report.py runs the A14 refusals first and writes analysis.json):
    import adv_analysis as aa
    analysis = aa.analyze(adv, uniform)   # each {"session": ..., "plan": ..., "rows": [...]}

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import math
import statistics

import core_b_analysis as ba

SCHEMA_VERSION = 1
CONDITIONS = ("B0", "T", "G", "TG", "K3")
REMOVALS = ("T", "G", "TG")
LEVELS = {2048: 25, 8192: 8}
MIXES = ("uniform", "mixed")
RESTARTS = ba.RESTARTS
MAJOR, SUBSTANTIVE = 0.5, 0.2  # A5: "removes more than half"; H-B4's substantive 20 % on the log scale
SPECIFICITY = 0.10  # A6 |ln(U_c / U_ref)|
KV_DOSE = 0.20  # A7 (i)
BLOCKED_SHARE = 0.5  # A7 (ii)
DOMINANT = 0.5  # A11
TOL = 1e-9
SPEC_REF = {"G": "B0", "TG": "T", "K3": "B0"}
Q_CATEGORIES = ("no-queue", "budget-long", "budget-other", "HOL-long", "KV-short", "unexplained", "unclassified")
BLOCKED = ("HOL-long", "KV-short")
C5_CANDIDATES = ("budget-long", "HOL-long", "KV-short")
ALLOCATOR_WASTE_BOUND = 0.031
DESCRIPTIVE_KEYS = ("long_class_p95_ttft_ms", "output_throughput", "total_token_throughput", "request_throughput",
                    "p95_tpot_ms", "p95_ttft_ms", "duration")
# run_adv row fields aggregated per cell as mean / SE (A13, A8, 量測): row["scrape_delta"][histogram]["mean"] (s),
# row["one_hz"][share], row["client"][key] (ms) and row["decomp"]["tail95"][O / Q / P] (ms).
HIST_MEANS = {"queue_time_mean_s": "vllm:request_queue_time_seconds",
              "prefill_time_mean_s": "vllm:request_prefill_time_seconds"}
ONE_HZ_KEYS = ("kv_full_while_waiting", "gate_zone")
CLIENT_KEYS = ("short_p50_ttft_ms", "short_p99_ttft_ms", "long_p50_ttft_ms", "short_p95_tpot_ms")
TAIL_PARTS = ("O", "Q", "P")

CONCLUSIONS = {
    "R0": "該 L 在 session 內沒有重現 H-B4，不做歸因",
    "R1": "主因是 batching，即長 request 的 chunked prefill 壟斷每個 step 的 budget（H-C1 成立；007 的 conjecture 得到支持）",
    "R2": "主因是 scheduler 的准入閘門，即 full-ISL 檢查加上 FCFS break，由 KV 占用觸發（H-C2 成立）",
    "R3": "兩個機制都是必要條件（conjunctive），移除任一個都消掉過半",
    "R4": "串聯瓶頸，batching 與准入閘門一起作用（H-C3 成立）",
    "R5": "最可能是 KV 容量經 FCFS 准入造成（推論級：由排除、容量敏感、觀測三者推得；KV 無法往上調，無法用移除檢驗確認）",
    "R6": "KV 可能有關，但未確立",
    "R7": "在 KV、batching、scheduler 之間沒有隔離出成因",
}
# R0 is "H-C0 failed" (024); an H-C0 that is not testable (A1 / 013) also stops attribution, but is not that finding.
CONCLUSION_H_C0_NT = "該 L 的 H-C0 不可檢定（B0 缺 3 次有效 restart），無法判定是否重現 H-B4，不做歸因"
# R4 with cat(T) or cat(G) NT (first match reads NT as "not M"): H-C3 is not testable, so it is not claimed.
CONCLUSION_R4_H_C3_NT = "移除 TG 消掉過半，但 cat(T) 或 cat(G) 為 NT，無法確認是串聯瓶頸（H-C3 不可檢定）"
NOTE_GATE_ALSO = "准入閘門也有貢獻"
NOTE_KV_SECONDARY = "KV 也受限（次要）"
NOTE_GATE_NO_KV = "閘門有效，但缺少 KV 證據，不一致"
NOTE_COMPUTE_LOAD = "殘餘部分以 conjecture 歸給固定 C 下的封閉迴路運算負載（Little's law）"
NOTE_H_C5_INCONSISTENT = "機制歸因不一致，以操弄的結論為準"
NOTE_H_C0_NT = "H-C0 不可檢定：B0 缺 3 次有效 restart，不做歸因"
NOTE_R = "只有相對下降（ρ 下降但 M 沒下降），不算支持"
NOTE_R_T = "chunk 上限的副作用或 uniform 位移遮蔽"
NOTE_W = "移除後 ρ 上升並超出 noise"
NOTE_GATE_PROTECTIVE = "閘門具保護作用（KV 證據 iii），不是 HOL 的證據"
ALLOCATOR_STATUS = "未操弄，未檢驗"
CROSS_L_DIFFER = "成因依 L 而異"
CROSS_L_NT = "有 L 的 H-C0 不可檢定，跨 L 不可檢定，不寫成 H-B4 的一般結論"


class AnalysisError(RuntimeError):
    pass


# ---------------------------------------------------------------- A1 validity


def invalid_reasons(row: dict) -> list:
    m = row.get("manip") or {}
    checks = (("not ok", row.get("ok") is not True), ("timeout", bool(row.get("timeout"))),
              ("skipped", bool(row.get("skipped"))), ("log facts", m.get("facts_ok") is not True),
              ("num_gpu_blocks", m.get("blocks_ok") is not True), ("chunk check", m.get("chunk_ok") is False),
              ("no short-class P95", row.get("short_class_p95_ttft_ms") is None))
    return [name for name, bad in checks if bad]


def row_valid(row: dict) -> bool:
    return not invalid_reasons(row)


def _index(rows: list) -> dict:
    """(cond, L, mix) -> {restart: row} over valid rows. Rows outside the plan's cells raise."""
    seen, valid = set(), {}
    for row in rows:
        key = (row.get("cond"), row.get("L"), row.get("mix"))
        if key[0] not in CONDITIONS or key[1] not in LEVELS or key[2] not in MIXES:
            raise AnalysisError(f"row outside the condition table: {key}")
        r = row.get("restart")
        if r not in range(1, RESTARTS + 1):
            raise AnalysisError(f"row {key} has restart {r!r}, not 1..{RESTARTS}")
        if (key, r) in seen:
            raise AnalysisError(f"duplicate row {key} restart {r}")
        seen.add((key, r))
        if row_valid(row):
            valid.setdefault(key, {})[r] = row
    return valid


def _decomp_ok(row: dict) -> bool:
    return isinstance(row.get("decomp"), dict) and row["decomp"].get("valid") is True


# ---------------------------------------------------------------- A2 / A3 statistics


def cell_stats(values: dict) -> dict | None:
    """values: restart -> Y. None unless all 3 restarts are present (013)."""
    if len(values) < RESTARTS:
        return None
    v = [values[r] for r in sorted(values)]
    return {"mean": statistics.mean(v), "se": statistics.stdev(v) / math.sqrt(len(v)), "n": len(v)}


def rho_stats(U: dict | None, M: dict | None) -> tuple:
    """rho = ln(M / U) and its §2 delta-method SE."""
    if U is None or M is None:
        return None, None
    return (math.log(M["mean"] / U["mean"]),
            math.sqrt((M["se"] / M["mean"]) ** 2 + (U["se"] / U["mean"]) ** 2))


def ratio_se(a: float, se_a: float, b: float, se_b: float) -> float:
    """SE of r = a / b by the §2 delta method r sqrt((SE_a/a)^2 + (SE_b/b)^2), written so that a = 0 is allowed."""
    r = a / b
    return math.sqrt(se_a ** 2 + (r * se_b) ** 2) / abs(b)


def beyond_noise(x: float, se_x: float, y: float, se_y: float) -> bool:
    return abs(x - y) > 2 * math.sqrt(se_x ** 2 + se_y ** 2)


def exceeds(hi: float, se_hi: float, lo: float, se_lo: float) -> bool:
    """hi - lo is positive and beyond noise (A3; never paired by run order)."""
    return hi > lo and beyond_noise(hi, se_hi, lo, se_lo)


# ---------------------------------------------------------------- A5 / A6


def removal_category(f: float, d_rho: bool, d_m: bool, u_rho: bool) -> str:
    """A5 (first match; a boundary value goes to the higher category)."""
    if f >= MAJOR - TOL and d_rho and d_m:
        return "M"
    if f >= SUBSTANTIVE - TOL and d_rho and d_m:
        return "P"
    if f >= SUBSTANTIVE - TOL and d_rho:
        return "R"
    if f <= -SUBSTANTIVE + TOL and u_rho:
        return "W"
    return "N"


def _cell(idx: dict, cond: str, L: int) -> dict:
    stats = {mix: cell_stats({r: row["short_class_p95_ttft_ms"] for r, row in idx.get((cond, L, mix), {}).items()})
             for mix in MIXES}
    U, M = stats["uniform"], stats["mixed"]
    rho, rho_se = rho_stats(U, M)
    return {"n_valid": {mix: len(idx.get((cond, L, mix), {})) for mix in MIXES},
            "U": U and U["mean"], "U_se": U and U["se"], "M": M and M["mean"], "M_se": M and M["se"],
            "rho": rho, "rho_se": rho_se}


def _versus_b0(cs: dict, c: str) -> dict:
    """f, a, g, the noise flags and ln(U_c / U_0) of condition c against B0 (None where a cell is missing)."""
    b, x = cs["B0"], cs[c]
    out = {k: None for k in ("f", "f_se", "a", "g", "d_rho", "u_rho", "d_m", "m_up", "ln_u_vs_b0")}
    if b["U"] is not None and x["U"] is not None:
        out["ln_u_vs_b0"] = math.log(x["U"] / b["U"])
    if b["rho"] is None or x["rho"] is None:
        return out
    if b["rho"] != 0:
        r = x["rho"] / b["rho"]
        out.update(f=1 - r, a=r - 1, f_se=ratio_se(x["rho"], x["rho_se"], b["rho"], b["rho_se"]))
    if b["M"] != b["U"]:
        out["g"] = 1 - (x["M"] - x["U"]) / (b["M"] - b["U"])
    out.update(d_rho=exceeds(b["rho"], b["rho_se"], x["rho"], x["rho_se"]),
               u_rho=exceeds(x["rho"], x["rho_se"], b["rho"], b["rho_se"]),
               d_m=exceeds(b["M"], b["M_se"], x["M"], x["M_se"]),
               m_up=exceeds(x["M"], x["M_se"], b["M"], b["M_se"]))
    return out


def specificity(cs: dict, c: str) -> dict | None:
    """A6: |ln(U_c / U_ref)| <= 0.10 for G, TG (ref T) and K3; None for T (it legitimately moves U)."""
    ref = SPEC_REF.get(c)
    if ref is None:
        return None
    if cs[c]["U"] is None or cs[ref]["U"] is None:
        return {"ref": ref, "ln_ratio": None, "ok": None}
    x = math.log(cs[c]["U"] / cs[ref]["U"])
    return {"ref": ref, "ln_ratio": x, "ok": abs(x) <= SPECIFICITY + TOL}


def effective(category: str, spec: dict | None) -> str:
    """The category the matrix reads: a qualified M is not M; an M whose check cannot be made is NT."""
    if category == "M" and spec is not None:
        if spec["ok"] is False:
            return "qualified"
        if spec["ok"] is None:
            return "NT"
    return category


# ---------------------------------------------------------------- A7 e_KV


def kv_dose(cs: dict) -> dict:
    """(i): a >= 0.20, rho and M up beyond noise, K3 uniform specificity."""
    k = cs["K3"]
    if k["a"] is None:
        return {"status": "NT", "holds": False}
    dose = k["a"] >= KV_DOSE - TOL and k["u_rho"] and k["m_up"]
    status = ("holds" if k["specificity"]["ok"] else "qualified") if dose else "not"
    return {"status": status, "holds": status == "holds", "a": k["a"], "u_rho": k["u_rho"], "m_up": k["m_up"],
            "specific": k["specificity"]["ok"]}


def kv_observed(idx: dict, L: int) -> dict:
    """(ii): B0-mixed tail Q share in HOL-long + KV-short, mean of 3 runs with valid decompositions."""
    runs = idx.get(("B0", L, "mixed"), {})
    if len(runs) < RESTARTS or not all(_decomp_ok(r) for r in runs.values()):
        return {"status": "NT", "holds": False}
    share = statistics.mean(sum(runs[r]["decomp"]["q_shares"][c] for c in BLOCKED) for r in sorted(runs))
    holds = share >= BLOCKED_SHARE - TOL
    return {"status": "holds" if holds else "not", "holds": holds, "share": share}


def kv_gate(idx: dict, L: int, cat_g: str) -> dict:
    """(iii): cat(G) = W (all cells valid) and G-mixed preemptions > 0."""
    if cat_g == "NT":
        return {"status": "NT", "holds": False}
    if cat_g != "W":
        return {"status": "not", "holds": False}
    pre = [r.get("preemptions") for r in idx[("G", L, "mixed")].values()]
    known = [p for p in pre if p is not None]
    total = sum(known)
    status = "holds" if total > 0 else ("NT" if len(known) < len(pre) else "not")
    return {"status": status, "holds": status == "holds", "preemptions": total}


# ---------------------------------------------------------------- A11 H-C5


def _delta_q(runs: dict, key: str) -> tuple:
    def mean_q(mix, c=None):
        return statistics.mean(r["decomp"]["tail"]["Q"] * (r["decomp"][key][c] if c else 1.0)
                               for r in runs[mix].values())
    return (mean_q("mixed") - mean_q("uniform"),
            {c: mean_q("mixed", c) - mean_q("uniform", c) for c in C5_CANDIDATES})


def _argmax(d: dict) -> str | None:
    top = max(d.values())
    winners = [c for c, v in d.items() if v == top]
    return winners[0] if len(winners) == 1 else None


def b0_delta_q(idx: dict, L: int) -> dict | None:
    """Tail Q by category, B0 mixed - uniform (primary tau = t_{j-2} and sensitivity t_{j-1});
    None unless all 6 B0 runs are valid with valid decompositions."""
    runs = {mix: idx.get(("B0", L, mix), {}) for mix in MIXES}
    if any(len(v) < RESTARTS or not all(_decomp_ok(r) for r in v.values()) for v in runs.values()):
        return None
    total, by = _delta_q(runs, "q_shares")
    _, by_sens = _delta_q(runs, "q_shares_sens")
    return {"total": total, "by": by, "by_sens": by_sens}


def h_c5(dq: dict | None, cat_t: str, kv_i: dict) -> dict:
    if dq is None:
        return {"status": "NT", "reason": "B0 decomposition invalid or missing in a run"}
    out = {"delta_q": dq["total"], "delta_q_by": dq["by"], "delta_q_by_sens": dq["by_sens"]}
    c_star = _argmax(dq["by"])
    share = dq["by"][c_star] / dq["total"] if c_star is not None and dq["total"] > 0 else None
    out.update(c_star=c_star, share=share)
    if share is None or share < DOMINANT - TOL:
        return {**out, "status": "NT", "reason": "no dominant category (ΔQ_c*/ΔQ < 0.5, tie, or ΔQ <= 0)"}
    sens = _argmax(dq["by_sens"])
    out["c_star_sens"] = sens
    if sens != c_star:
        return {**out, "status": "NT", "reason": f"c* changes under τ = t_(j-1) ({c_star} -> {sens})"}
    if c_star == "budget-long":
        if cat_t == "NT":
            return {**out, "status": "NT", "reason": "c* = budget-long but cat(T) is NT"}
        ok = cat_t in ("M", "P")
    else:
        if kv_i["status"] == "NT":
            return {**out, "status": "NT", "reason": f"c* = {c_star} but H-C4 (i) is NT"}
        ok = kv_i["holds"]
    return {**out, "status": "consistent" if ok else "inconsistent"}


# ---------------------------------------------------------------- hypotheses


def _h_c1(cat: str) -> dict:
    status, note = {
        "M": ("supported", "BM 是主因"),
        "P": ("not supported", "BM 是貢獻者，不是主因"),
        "R": ("not supported", f"{NOTE_R}；{NOTE_R_T}"),
        "N": ("not supported", "在該 L，BM 不是實質成因"),
        "W": ("not supported", "在該 L，BM 不是實質成因"),
        "NT": ("not testable", "不可檢定"),
    }[cat]
    return {"status": status, "category": cat, "note": note}


def _h_c2(cat: str, eff: str, kv_iii: dict) -> dict:
    if cat == "M" and eff != "M":
        return {"status": "qualified", "category": cat, "note": "uniform 檢查失敗，不能觸發 R2"}
    status, note = {
        "M": ("supported", "准入閘門是主因"),
        "P": ("not supported", "閘門是貢獻者，不是主因"),
        "R": ("not supported", NOTE_R),
        "N": ("not supported", "閘門不是實質成因"),
        "W": ("not supported", NOTE_GATE_PROTECTIVE if kv_iii["holds"] else NOTE_W),
        "NT": ("not testable", "逾時、crash 或缺有效 restart：NT，不當成任何證據"),
    }[cat]
    return {"status": status, "category": cat, "note": note}


def _h_c3(eff: dict) -> dict:
    t, g, tg = (eff[c] for c in REMOVALS)
    if "M" in (t, g):
        return {"status": "not applicable", "note": "單一移除已是 M：成因是單一的（H-C1 或 H-C2）"}
    if tg == "NT":
        return {"status": "not testable", "note": "cat(TG) 為 NT"}
    if tg == "qualified":
        return {"status": "qualified", "note": "TG 的 uniform 檢查失敗"}
    if tg != "M":
        return {"status": "not supported", "note": "沒有串聯的證據"}
    if "NT" in (t, g):
        return {"status": "not testable", "note": "cat(TG) = M 但 cat(T) 或 cat(G) 為 NT"}
    return {"status": "supported", "note": "串聯瓶頸"}


def _h_c4(kv: dict) -> dict:
    n = kv["i"]["holds"] + kv["ii"]["holds"]
    status = {2: "supported", 1: "partial evidence", 0: "not supported"}[n]
    return {"status": status, "note": {2: "KV 受限且敏感", 1: "部分證據", 0: "KV 未顯示受限或敏感"}[n],
            "i": kv["i"]["status"], "ii": kv["ii"]["status"], "qualified_i": kv["i"]["status"] == "qualified",
            "incomplete": kv["ii"]["status"] == "NT"}


# ---------------------------------------------------------------- interpretation matrix


def interpret(h_c0_passed: bool, eff: dict, kv: dict) -> str:
    """First match of R0..R7. eff: T / G / TG -> effective category; kv: i / ii / iii -> holds."""
    if not h_c0_passed:
        return "R0"
    t, g, tg = (eff[c] == "M" for c in REMOVALS)
    if t and not g:
        return "R1"
    if g and not t:
        return "R2"
    if t and g:
        return "R3"
    if tg:
        return "R4"
    if kv["i"] and (kv["ii"] or kv["iii"]):
        return "R5"
    return "R6" if any(kv.values()) else "R7"


def _flips(eff: dict, spec_ok: dict, kv: dict, kv_status: dict, row: str) -> tuple:
    """Single NT inputs whose supportive value would move the row: NT -> M (a known failed uniform check
    keeps it qualified) for the removals; NT -> holds for the KV components (reported apart)."""
    nt = []
    for c in REMOVALS:
        if eff[c] == "NT":
            alt = {**eff, c: "qualified" if spec_ok.get(c) is False else "M"}
            if interpret(True, alt, kv) != row:
                nt.append(c)
    kv_nt = [k for k in ("i", "ii", "iii") if kv_status[k] == "NT" and interpret(True, eff, {**kv, k: True}) != row]
    return nt, kv_nt


def cross_l(levels: dict) -> dict:
    """A12: the same row at both L is the general H-B4 conclusion. A level whose H-C0 is not testable is never
    agreement or disagreement (agree None). Under one row the general conclusion takes a level's qualified variant
    (R4 without H-C3 supported) whenever one level has it."""
    rows = {L: lv["interpretation"]["row"] for L, lv in levels.items()}
    nt = sorted(L for L, lv in levels.items() if lv["h_c0"].get("status") != "tested")
    if nt:
        return {"rows": rows, "agree": None, "row": None, "conclusion": CROSS_L_NT, "not_testable": nt}
    if len(set(rows.values())) == 1:
        row = next(iter(rows.values()))
        texts = [levels[L]["interpretation"]["conclusion"] for L in sorted(levels)]
        conclusion = next((t for t in texts if t != CONCLUSIONS[row]), CONCLUSIONS[row])
        return {"rows": rows, "agree": True, "row": row, "conclusion": conclusion, "not_testable": []}
    return {"rows": rows, "agree": False, "row": None, "conclusion": CROSS_L_DIFFER, "not_testable": []}


# ---------------------------------------------------------------- A8 / A13 descriptive


def _mean_or_none(values: list):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def _field(row: dict, *path):
    """row[path[0]][path[1]]..., or None where a level is missing or not a dict."""
    for key in path:
        if not isinstance(row, dict):
            return None
        row = row.get(key)
    return row


def _mean_se(values: list) -> dict | None:
    """Mean / SE (s / sqrt n) over the cell's valid restarts; None if any of them lacks the value (never a partial
    mean) or there is none; SE None from a single run."""
    if not values or any(v is None for v in values):
        return None
    n = len(values)
    return {"mean": statistics.mean(values), "se": statistics.stdev(values) / math.sqrt(n) if n > 1 else None,
            "n": n}


def _tail_stats(dec: list, key: str) -> dict | None:
    """A8: per-run tail means of O / Q / P, then mean / SE over the 3 restarts (key "tail" or "tail95")."""
    tails = [d.get(key) for d in dec]
    if len(tails) < RESTARTS or not all(isinstance(t, dict) for t in tails):
        return None
    return {p: cell_stats({i: t[p] for i, t in enumerate(tails)}) for p in TAIL_PARTS}


def _cell_descriptive(runs: dict) -> dict:
    rs = [runs[r] for r in sorted(runs)]
    out = {"n_valid": len(rs), **{k: _mean_or_none([r.get(k) for r in rs]) for k in DESCRIPTIVE_KEYS}}
    pre = [r["preemptions"] for r in rs if r.get("preemptions") is not None]
    peaks = [r["kv_peak"] for r in rs if r.get("kv_peak") is not None]
    out.update(preemptions_total=sum(pre) if pre else None, kv_peak_max=max(peaks, default=None))
    out["hist"] = {k: _mean_se([_field(r, "scrape_delta", h, "mean") for r in rs]) for k, h in HIST_MEANS.items()}
    out["one_hz"] = {k: _mean_se([_field(r, "one_hz", k) for r in rs]) for k in ONE_HZ_KEYS}
    out["client"] = {k: _mean_se([_field(r, "client", k) for r in rs]) for k in CLIENT_KEYS}
    dec = [r["decomp"] for r in rs if _decomp_ok(r)]
    out["decomp_valid"] = len(dec)
    out["tail"] = out["tail95"] = None
    if len(dec) == len(rs) == RESTARTS:
        out["tail"], out["tail95"] = _tail_stats(dec, "tail"), _tail_stats(dec, "tail95")
        for k in ("residual_bm_share", "unexplained_break_share"):
            out[k] = _mean_or_none([d.get(k) for d in dec])
    return out


def _tail_delta(cells: dict, c: str, key: str) -> dict | None:
    """Delta = mixed - uniform of one condition's tail means (A8), SE = sqrt(SE_m^2 + SE_u^2)."""
    u, m = cells[f"{c}-uniform"][key], cells[f"{c}-mixed"][key]
    if u is None or m is None:
        return None
    return {p: {"delta": m[p]["mean"] - u[p]["mean"], "se": math.sqrt(m[p]["se"] ** 2 + u[p]["se"] ** 2)}
            for p in TAIL_PARTS}


def _core_mean(session: dict | None, L: int, mix: str):
    if session is None:
        return None
    vals = [r["short_class_p95_ttft_ms"] for r in session["rows"]
            if r.get("L") == L and r.get("mix") == mix and r.get("C") == LEVELS[L] and r.get("ok") is True
            and r.get("short_class_p95_ttft_ms") is not None]
    return statistics.mean(vals) if vals else None


def _descriptive(idx: dict, L: int, cs: dict, uniform: dict, core_mixed: dict | None) -> dict:
    cells = {f"{c}-{mix}": _cell_descriptive(idx.get((c, L, mix), {})) for c in CONDITIONS for mix in MIXES}
    tail_delta = {c: _tail_delta(cells, c, "tail") for c in CONDITIONS}
    tail95_delta = {c: _tail_delta(cells, c, "tail95") for c in CONDITIONS}
    rho = {c: (cs[c]["rho"], cs[c]["rho_se"]) for c in ("B0", "T", "G", "TG")}
    interaction = None
    if all(v[0] is not None for v in rho.values()):
        interaction = {"I": (rho["B0"][0] - rho["T"][0]) - (rho["G"][0] - rho["TG"][0]),
                       "se": math.sqrt(sum(v[1] ** 2 for v in rho.values()))}
    core = {"uniform": _core_mean(uniform, L, "uniform"), "mixed": _core_mean(core_mixed, L, "mixed")}
    ratio = {mix: (cs["B0"][k] / core[mix] if cs["B0"][k] is not None and core[mix] else None)
             for mix, k in (("uniform", "U"), ("mixed", "M"))}
    return {"cells": cells, "tail_delta": tail_delta, "tail95_delta": tail95_delta, "interaction": interaction,
            "core_ratio": ratio}


def _attachments(cs: dict, desc: dict, kv: dict | None) -> dict:
    """"所有列都要附上": residual rho ratio, f / g / a, R / W notes, M / P deltas, preemptions, KV peak."""
    b0 = cs["B0"]["rho"]
    cells = desc["cells"]
    att = {"residual_rho_ratio_tg": cs["TG"]["rho"] / b0 if b0 and cs["TG"]["rho"] is not None else None,
           **{k: {c: cs[c][k] for c in CONDITIONS[1:]} for k in ("f", "g", "a")},
           "r_w_notes": [], "m_p_deltas": {},
           "preemptions": {c: {mix: cells[f"{c}-{mix}"]["preemptions_total"] for mix in MIXES} for c in CONDITIONS},
           "kv_peak": {c: {mix: cells[f"{c}-{mix}"]["kv_peak_max"] for mix in MIXES} for c in CONDITIONS}}

    def delta(c, mix, k):
        x, y = cells[f"{c}-{mix}"][k], cells[f"B0-{mix}"][k]
        return None if x is None or y is None else x - y

    for c in REMOVALS:
        category = cs[c]["category"]
        if category in ("M", "P"):
            att["m_p_deltas"][c] = {"d_output_throughput_mixed": delta(c, "mixed", "output_throughput"),
                                    "d_output_throughput_uniform": delta(c, "uniform", "output_throughput"),
                                    "d_long_class_p95_ttft_ms": delta(c, "mixed", "long_class_p95_ttft_ms")}
        elif category == "R":
            att["r_w_notes"].append({"cond": c, "category": "R", "note": NOTE_R + ("；" + NOTE_R_T if c == "T" else "")})
        elif category == "W":
            protective = c == "G" and kv is not None and kv["iii"]["holds"]
            att["r_w_notes"].append({"cond": c, "category": "W",
                                     "note": NOTE_GATE_PROTECTIVE if protective else NOTE_W})
    return att


# ---------------------------------------------------------------- per level


def _level(idx: dict, L: int, unit: dict, uniform: dict, core_mixed: dict | None) -> dict:
    cs = {c: _cell(idx, c, L) for c in CONDITIONS}
    for c in CONDITIONS[1:]:
        cs[c].update(_versus_b0(cs, c), specificity=specificity(cs, c))
    for c in CONDITIONS[1:]:
        spec = cs[c]["specificity"]
        cs[c]["qualified"] = None if spec is None or spec["ok"] is None else not spec["ok"]
    passed = unit.get("status") == "tested" and not unit.get("refutes")
    for c in REMOVALS:
        x = cs[c]
        if passed:
            x["category"] = "NT" if x["f"] is None else removal_category(x["f"], x["d_rho"], x["d_m"], x["u_rho"])
            x["effective"] = effective(x["category"], x["specificity"])
        else:
            x["category"] = x["effective"] = None
        x["g_category"] = (removal_category(x["g"], x["d_rho"], x["d_m"], x["u_rho"])
                           if passed and x["g"] is not None and x["category"] != "NT" else None)
        x["g_differs"] = None if x["g_category"] is None else x["g_category"] != x["category"]
    desc = _descriptive(idx, L, cs, uniform, core_mixed)
    b0_mixed = desc["cells"]["B0-mixed"]
    allocator = {"status": ALLOCATOR_STATUS, "waste_upper_bound": ALLOCATOR_WASTE_BOUND,
                 "unexplained_break_share": b0_mixed.get("unexplained_break_share")}
    h_c0 = {**unit, "passed": passed}
    if not passed:
        # R0 (first match). An H-C0 that is not testable is not "H-C0 failed": its own conclusion, and incomplete,
        # since a passing H-C0 would move the row.
        nt = unit.get("status") != "tested"
        return {"L": L, "C": LEVELS[L], "h_c0": h_c0, "conditions": cs, "kv": None,
                "hypotheses": {h: {"status": "uninterpretable"} for h in ("H-C1", "H-C2", "H-C3", "H-C4", "H-C5")},
                "interpretation": {"row": "R0", "conclusion": CONCLUSION_H_C0_NT if nt else CONCLUSIONS["R0"],
                                   "notes": [NOTE_H_C0_NT] if nt else [], "incomplete": nt,
                                   "incomplete_flips": ["H-C0"] if nt else [], "kv_nt_flips": [], "sub_label": None,
                                   "contributors": [], "h_c5": {"status": "uninterpretable"}, "allocator": allocator,
                                   "attachments": _attachments(cs, desc, None)},
                "descriptive": desc}

    cats = {c: cs[c]["category"] for c in REMOVALS}
    eff = {c: cs[c]["effective"] for c in REMOVALS}
    kv = {"i": kv_dose(cs), "ii": kv_observed(idx, L), "iii": kv_gate(idx, L, cats["G"])}
    kv["e_kv"] = sum(kv[k]["holds"] for k in ("i", "ii", "iii"))
    dq = b0_delta_q(idx, L)
    hyps = {"H-C1": _h_c1(cats["T"]), "H-C2": _h_c2(cats["G"], eff["G"], kv["iii"]), "H-C3": _h_c3(eff),
            "H-C4": _h_c4(kv), "H-C5": h_c5(dq, cats["T"], kv["i"])}

    holds = {k: kv[k]["holds"] for k in ("i", "ii", "iii")}
    row = interpret(True, eff, holds)
    flips, kv_flips = _flips(eff, {c: (cs[c]["specificity"] or {}).get("ok") for c in REMOVALS}, holds,
                             {k: kv[k]["status"] for k in ("i", "ii", "iii")}, row)
    notes, sub_label, contributors = [], None, []
    if row == "R1":
        if cats["G"] == "P":
            notes.append(NOTE_GATE_ALSO)
        if kv["e_kv"] >= 2:
            notes.append(NOTE_KV_SECONDARY)
    if row == "R2" and kv["e_kv"] == 0:
        notes.append(NOTE_GATE_NO_KV)
    if row == "R5" and dq is not None and dq["by"]["HOL-long"] != dq["by"]["KV-short"] \
            and max(dq["by"]["HOL-long"], dq["by"]["KV-short"]) > 0:
        sub_label = ("HOL-long" if dq["by"]["HOL-long"] > dq["by"]["KV-short"] else "KV-short") + " 主導"
    if row in ("R6", "R7"):
        contributors = [{"cond": c, "category": cats[c], "qualified": cs[c]["qualified"]}
                        for c in REMOVALS if cats[c] in ("P", "R")]
    if row == "R7":
        notes.append(NOTE_COMPUTE_LOAD)
    c5 = hyps["H-C5"]
    if c5["status"] == "inconsistent":
        notes.append(NOTE_H_C5_INCONSISTENT)
    conclusion = CONCLUSIONS[row]
    if row == "R4" and hyps["H-C3"]["status"] != "supported":
        # R4 means eff(TG) = M with no M among T / G, so _h_c3 is "supported" or (T or G NT) "not testable"
        conclusion = CONCLUSION_R4_H_C3_NT
    return {"L": L, "C": LEVELS[L], "h_c0": h_c0, "conditions": cs, "kv": kv, "hypotheses": hyps,
            "interpretation": {"row": row, "conclusion": conclusion, "notes": notes,
                               "incomplete": bool(flips), "incomplete_flips": flips, "kv_nt_flips": kv_flips,
                               "sub_label": sub_label, "contributors": contributors,
                               "h_c5": {k: c5.get(k) for k in ("status", "c_star", "reason")},
                               "allocator": allocator, "attachments": _attachments(cs, desc, kv)},
            "descriptive": desc}


# ---------------------------------------------------------------- entry point


def input_problems(adv: dict, uniform: dict) -> list:
    """Session validity (A1) and the C check: plan C = frozen C = knee recomputed from the uniform session."""
    problems, s = [], adv["session"]
    if s.get("formal") is not True:
        problems.append("the Advanced session is not formal")
    if s.get("env_valid") is not True:
        problems.append(f"the Advanced session is env-invalid (env_valid={s.get('env_valid')!r})")
    plan_c = {int(L): C for L, C in (adv["plan"].get("c") or {}).items()}
    if plan_c != LEVELS:
        problems.append(f"plan C = {plan_c} is not the frozen C {LEVELS}")
    for L, C in LEVELS.items():
        k = ba.knee(uniform["rows"], L, uniform["plan"]["grids"][str(L)])["c_knee"]
        if k != C:
            problems.append(f"the uniform C_knee at L = {L} recomputes to {k}, not {C}")
    bad = sorted({(r.get("L"), r.get("C")) for r in adv["rows"]
                  if r.get("L") in LEVELS and r.get("C") != LEVELS[r["L"]]})
    for L, C in bad:
        problems.append(f"rows at L = {L} ran at C = {C}, not C = {LEVELS[L]}")
    return problems


def analyze(adv: dict, uniform: dict, core_mixed: dict | None = None) -> dict:
    """adv: the Advanced {"session", "plan", "rows"}; uniform: the formal Core uniform session (C check,
    descriptive B0 / Core ratio); core_mixed: optional Core mixed session (descriptive ratio only)."""
    problems = input_problems(adv, uniform)
    if problems:
        raise AnalysisError("; ".join(problems))
    idx = _index(adv["rows"])
    b0 = {L: {"C": C, **{mix: {r: row["short_class_p95_ttft_ms"] for r, row in idx.get(("B0", L, mix), {}).items()}
                         for mix in MIXES}}
          for L, C in LEVELS.items()}
    units = {u["L"]: u for u in ba.h_b4_values(b0)["units"]}
    levels = {str(L): _level(idx, L, units[L], uniform, core_mixed) for L in LEVELS}
    runs = [{"cond": r.get("cond"), "L": r.get("L"), "mix": r.get("mix"), "restart": r.get("restart"),
             "block": r.get("block"), "valid": row_valid(r), "invalid": invalid_reasons(r),
             "reasons": r.get("reasons") or [], "timeout": bool(r.get("timeout")), "skipped": bool(r.get("skipped"))}
            for r in adv["rows"]]
    return {"schema_version": SCHEMA_VERSION, "session_id": adv["session"].get("session_id"),
            "derived_from": adv["plan"].get("derived_from"), "c": {str(L): C for L, C in LEVELS.items()},
            "levels": levels, "cross_l": cross_l(levels), "runs": runs}
