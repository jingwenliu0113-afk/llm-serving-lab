"""Tests for benchmarks/decision_table.py (rules in docs/decisions/021), on synthetic results trees."""

import copy
import functools
import json
import random
import tomllib
from pathlib import Path

import pytest

import core_a_analysis as an
import core_a_report as ar
import core_b_report as br
import decision_table as dt
import quality_analysis as qa

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
SIZES, QUANTS, LS = ("4B", "9B"), ("Q8_0", "Q6_K", "Q4_K_M"), (2048, 8192, 32768)
CONFIGS = [f"{s}-{q}" for s in SIZES for q in QUANTS]
EMBD = {**{("4B", q): 0 for q in QUANTS},
        **{("9B", q): v for q, v in zip(QUANTS, (1080688640, 834355200, 572129280))}}
TIED = {"4B": True, "9B": False}
DECODE = {"4B-Q8_0": 130, "4B-Q6_K": 160, "4B-Q4_K_M": 185, "9B-Q8_0": 70, "9B-Q6_K": 100, "9B-Q4_K_M": 120}
PREFILL = {"4B-Q8_0": 8500, "4B-Q6_K": 7300, "4B-Q4_K_M": 8300, "9B-Q8_0": 5000, "9B-Q6_K": 4400, "9B-Q4_K_M": 5500}
DEVICE = {"4B-Q8_0": 4400, "4B-Q6_K": 3600, "4B-Q4_K_M": 2900, "9B-Q8_0": 9000, "9B-Q6_K": 7200, "9B-Q4_K_M": 5700}
HOST = {"4B": 500, "9B": 1000}


# --------------------------------------------------------------- fixtures --

def core_a_rows(spec: dict | None = None) -> list:
    """spec[(cfg, L)]: {"skip"}, {"oom"}, {"error"}, {"decode": [3]}, {"ttft": [3]}, {"device": [3]},
    {"host": [3]}, {"drop_decode_rep": 3}, {"drop_fit_rep": 3}."""
    rows = []
    for cfg in CONFIGS:
        size, q = cfg.split("-", 1)
        for L in LS:
            o = (spec or {}).get((cfg, L), {})
            if o.get("skip"):
                continue
            dec = o.get("decode", [DECODE[cfg] * (1 - L / 400000) + d for d in (-0.5, 0.0, 0.5)])
            ttft = o.get("ttft", [L / PREFILL[cfg] * f for f in (0.999, 1.0, 1.001)])
            dev = o.get("device", [DEVICE[cfg] + L // 16] * 3)
            host = o.get("host", [HOST[size]] * 3)
            for i, rep in enumerate((1, 2, 3)):
                base = {"size": size, "quant": q, "L": L, "rep": rep, "block": None, "stage": None, "reason": None}
                fit_out = "oom" if o.get("oom") else "ok"
                if o.get("drop_fit_rep") != rep:
                    rows.append({**base, "kind": "fit", "job_id": f"fit-{cfg}-{L}-r{rep}", "outcome": fit_out,
                                 "summary": {"memory": {"devices": [{"self": dev[i]}], "host": {"self": host[i]}}}
                                 if fit_out == "ok" else None,
                                 "external_memory": {"peak_footprint_bytes": 2**30, "gpu_mem_used_peak_mib": dev[i] + 462}})
                if o.get("oom"):
                    continue
                rows.append({**base, "kind": "ttft", "job_id": f"ttft-{cfg}-{L}-r{rep}", "outcome": "ok",
                             "metrics": {"avg_ns": ttft[i] * 1e9}})
                if o.get("drop_decode_rep") == rep:
                    continue
                bad = o.get("error") and rep == 2
                rows.append({**base, "kind": "decode", "job_id": f"decode-{cfg}-{L}-r{rep}",
                             "outcome": "error" if bad else "ok", "stage": "bench" if bad else None,
                             "reason": "exit 1" if bad else None, "metrics": {} if bad else {"avg_ts": dec[i]}})
    return rows


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n")


def _write_rows(p: Path, rows: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def core_a_analysis(rows: list, platform: str, session: dict) -> dict:
    params = None if platform == "macos-metal" else {k: PINS["stage_b"][k] for k in ("v_total_mib", "b_mib", "c_ctx_mib")}
    rep = ar.report(rows, platform=platform, pins=PINS, embd_bytes=EMBD, tied=TIED, platform_params=params)
    rep.update({"session": session["session_id"], "mode": session["mode"], "env_valid": session["env_valid"],
                "sentinel": session.get("sentinel")})
    return json.loads(json.dumps(rep))


def write_core_a(results: Path, platform: str, sid: str, rows: list | None = None, *, session: dict | None = None,
                 analysis: bool = True) -> Path:
    rows = core_a_rows() if rows is None else rows
    d = results / "core_a" / platform / sid
    s = {"session_id": sid, "platform": platform, "mode": "formal", "seed": 20261008, "env_valid": True,
         "sentinel": {"drift": False, "drift_pct": 0.1}, "battery_drain": None, "charge_pauses": []}
    if platform == "macos-metal":
        s["powermode"] = 2
    else:
        s["session_b"] = {"drift": False}
    s.update(session or {})
    _write(d / "session.json", s)
    _write_rows(d / "rows.jsonl", rows)
    (d / "raw").mkdir(exist_ok=True)
    (d / "raw" / "x.log").write_text("raw")
    (d / "env").mkdir(exist_ok=True)
    (d / "env" / "x.json").write_text("{}")
    if analysis:
        _write(d / "analysis.json", core_a_analysis(rows, platform, s))
    return d


KLD = {"4B-Q8_0": (0.002, 0.0001), "4B-Q6_K": (0.006, 0.0002), "4B-Q4_K_M": (0.03, 0.001),
       "9B-Q8_0": (0.001, 0.0001), "9B-Q6_K": (0.004, 0.0002), "9B-Q4_K_M": (0.02, 0.001)}


def quality_rows(kld: dict | None = None, *, paired: str = "9B-better", n_chunks: int = 30, n_items: int = 12,
                 drop_kld: tuple = ()) -> list:
    kld = {**KLD, **(kld or {})}
    rng = random.Random(7)
    base = [2.0 + rng.random() for _ in range(n_chunks)]
    if paired == "9B-better":
        other = [b - 0.05 + 0.001 * rng.random() for b in base]
    else:  # no resolved difference
        other = [b + (0.1 if i % 2 else -0.1) for i, b in enumerate(base)]
    rows = []
    for cfg, (m, se) in kld.items():
        size, q = cfg.split("-", 1)
        if cfg in drop_kld:
            continue
        rows.append({"kind": "kld", "size": size, "point": q, "outcome": "ok", "job_id": f"kld-{cfg}",
                     "metrics": {"kld_mean": m, "kld_mean_se": se, "ppl_q": 8.0 + m, "ppl_q_se": 0.05,
                                 "same_top_p_pct": 99.0 - m, "kld_p99": 10 * m}})
    for size in SIZES:
        rows.append({"kind": "kld_self", "size": size, "point": "BF16", "outcome": "ok", "job_id": f"self-{size}",
                     "metrics": {"kld_mean": 0.0}})
        rows.append({"kind": "kld_base", "size": size, "point": "BF16", "outcome": "ok", "job_id": f"base-{size}",
                     "metrics": {"ppl": 8.0, "ppl_se": 0.05, "chunk_nll": base}})
        rows.append({"kind": "delete_kld", "size": size, "point": "BF16", "outcome": "ok", "job_id": f"del-{size}"})
        for point in ("BF16", *QUANTS):
            correct = [int((i * 7 + len(point) + len(size)) % 5 != 0) for i in range(n_items)]
            rows.append({"kind": "mc", "task": "hellaswag", "size": size, "point": point, "outcome": "ok",
                         "job_id": f"mc-{size}-{point}", "metrics": {"correct": correct}})
    rows.append({"kind": "ppl", "size": "9B", "point": "Q4_K_M", "outcome": "ok", "job_id": "ppl-9B",
                 "metrics": {"chunk_nll": other}})
    rows.append({"kind": "ppl", "size": "4B", "point": "Q8_0", "outcome": "ok", "job_id": "ppl-4B",
                 "metrics": {"chunk_nll": base}})
    return rows


@functools.lru_cache(maxsize=None)
def _quality_analysis_cached(rows_json: str) -> str:
    rows = [r for r in json.loads(rows_json) if r["kind"] != "delete_kld"]
    return json.dumps({"h_a5": qa.h_a5(rows), "downstream": qa.downstream(rows)})


def write_quality(results: Path, sid: str, rows: list | None = None, *, session: dict | None = None,
                  analysis: bool = True) -> Path:
    rows = quality_rows() if rows is None else rows
    d = results / "quality" / sid
    s = {"session_id": sid, "mode": "formal", "env_valid": True, "platform": "wsl2-cuda", **(session or {})}
    _write(d / "session.json", s)
    _write_rows(d / "rows.jsonl", rows)
    if analysis:
        a = json.loads(_quality_analysis_cached(json.dumps(rows)))
        _write(d / "analysis.json", {"session": sid, "mode": s["mode"], "env_valid": s["env_valid"],
                                     "platform": s["platform"], **a})
    return d


GRIDS = {"2048": [1, 2, 4, 8], "8192": [1, 2, 4], "32768": [1, 2]}
C_KV = {"2048": 4, "8192": 2, "32768": 1}
CAP = {2048: 4, 8192: 2, 32768: 1}


def b_rows(mix: str = "uniform", *, grids: dict | None = None, cap: dict | None = None, knees: dict | None = None,
           drop: tuple = ()) -> list:
    grids, cap = grids or GRIDS, {**CAP, **(cap or {})}
    rows = []
    for Ls, grid in sorted(grids.items(), key=lambda x: int(x[0])):
        L = int(Ls)
        if mix == "mixed":
            if not knees or str(L) not in knees:
                continue
            grid = [knees[str(L)]]
        for r in (1, 2, 3):
            for C in grid:
                if (L, C, r) in drop:
                    continue
                thr = 100 * min(C, cap[L]) + 0.1 * r
                ttft = 100 * L / 2048 * (1 + max(0, C - cap[L])) * (1 + 0.01 * r)
                tpot = 20 + 0.5 * C + 0.01 * r
                rows.append({"L": L, "mix": mix, "restart": r, "C": C, "ok": True, "output_throughput": thr,
                             "request_throughput": thr / 128, "total_token_throughput": thr * L / 128,
                             "p50_ttft_ms": 0.6 * ttft, "p95_ttft_ms": ttft, "p50_tpot_ms": 0.8 * tpot,
                             "p95_tpot_ms": tpot, "short_class_p95_ttft_ms": ttft * (1.5 if mix == "mixed" else 1.0),
                             "metrics_sampled": {"kv_cache_usage_peak": 0.5, "preemptions": 0}})
    return rows


B_SESSION = {"formal": True, "env_valid": True, "server": ["vllm", "serve", "m"], "server_env": {"HF_HUB_OFFLINE": "1"}}


def _b_plan(grids, c_kv, mix="uniform", derived_from=None, knees=None):
    runs = [{"L": int(L), "mix": mix, "restart": r, "order": (g if mix == "uniform" else [knees[L]])}
            for L, g in grids.items() if mix == "uniform" or L in (knees or {}) for r in (1, 2, 3)]
    plan = {"grids": grids, "c_kv": c_kv, "runs": runs, "num_gpu_blocks": 248, "block_size": 528,
            "kv_cache_memory_bytes": 2**32, "gpu_memory_utilization": "0.91"}
    if derived_from:
        plan.update({"derived_from": derived_from, "c_knee": knees})
    return plan


def write_core_b(results: Path, sid: str = "b-uniform", *, rows: list | None = None, grids: dict | None = None,
                 c_kv: dict | None = None, session: dict | None = None, mixed_sid: str | None = None,
                 analysis: str = "auto") -> Path:
    grids, c_kv = grids or GRIDS, c_kv or C_KV
    rows = b_rows(grids=grids) if rows is None else rows
    d = results / "core_b" / sid
    _write(d / "session.json", {"session_id": sid, **B_SESSION, **(session or {})})
    _write(d / "plan.json", _b_plan(grids, c_kv))
    _write_rows(d / "rows.jsonl", rows)
    if analysis != "none":
        regenerate_core_b(results, sid, None if analysis == "uniform-only" else mixed_sid)
    return d


def write_mixed(results: Path, sid: str = "b-mixed", uniform_sid: str = "b-uniform", knees: dict | None = None,
                session: dict | None = None, regenerate: bool = True) -> Path:
    knees = {"2048": 4, "8192": 2} if knees is None else knees
    d = results / "core_b" / sid
    _write(d / "session.json", {"session_id": sid, **B_SESSION, **(session or {})})
    _write(d / "plan.json", _b_plan(GRIDS, C_KV, "mixed", uniform_sid, knees))
    _write_rows(d / "rows.jsonl", b_rows("mixed", knees=knees))
    if regenerate:
        regenerate_core_b(results, uniform_sid, sid)
    return d


def regenerate_core_b(results: Path, uniform_sid: str, mixed_sid: str | None) -> None:
    u = br._load(str(results / "core_b" / uniform_sid))
    m = br._load(str(results / "core_b" / mixed_sid)) if mixed_sid else None
    _write(results / "core_b" / uniform_sid / "analysis.json", br.report(u, m))


def full_tree(tmp_path: Path, **kw) -> Path:
    results = tmp_path / "results"
    write_core_a(results, "macos-metal", "mac-a", kw.get("mac_rows"))
    write_core_a(results, "wsl2-cuda", "rtx-a", kw.get("rtx_rows"))
    if kw.get("quality", True):
        write_quality(results, "q-formal", kw.get("q_rows"))
    write_core_b(results)
    if kw.get("mixed", True):
        write_mixed(results)
    return results


def build(results: Path) -> dict:
    return dt.build(results, PINS)


@pytest.fixture(scope="module")
def base_table(tmp_path_factory):
    results = full_tree(tmp_path_factory.mktemp("base"))
    return results, build(results)


# -------------------------------------------------------------- relations --

def test_scalar_rel_delegates_to_frozen_noise_rule():
    a, b = {"m": 10.0, "se": 0.3}, {"m": 9.0, "se": 0.4}
    edge = 2 * (0.3 ** 2 + 0.4 ** 2) ** 0.5  # = 1.0
    assert edge == pytest.approx(1.0)
    assert dt.scalar_rel("decode", a, b) == ("better" if an._beyond_noise(10.0, 0.3, 9.0, 0.4) else "tie")
    assert dt.scalar_rel("decode", {"m": 10.0 + 1e-9, "se": 0.3}, b) == "better"
    assert dt.scalar_rel("decode", b, {"m": 10.0 + 1e-9, "se": 0.3}) == "worse"
    assert dt.scalar_rel("ttft", {"m": 1.0, "se": 0.0}, {"m": 2.0, "se": 0.0}) == "better"  # lower is better
    assert dt.scalar_rel("memory", {"m": 101.0, "se": 0.0}, {"m": 100.0, "se": 0.0}) == "worse"  # SE 0: 1 MiB counts
    assert dt.scalar_rel("decode", {"m": 10.0, "se": None}, {"m": 1.0, "se": 0.1}) == "tie"


def _h(tests):
    return {"verdict": "refuted", "tests": tests}


def _kt(size, hi, lo, z, reject):
    return {"name": f"KLD {size} {hi} > {lo}", "z": z, "p": 0.01, "direction_ok": z > 0, "p_adj": 0.02, "reject": reject}


def _pt(mean_delta, reject):
    return {"name": dt.PAIRED_TEST, "mean_delta": mean_delta, "p": 0.01, "direction_ok": mean_delta < 0,
            "ci95": [-1, 1], "p_adj": 0.02, "reject": reject}


def test_quality_relations_come_only_from_holm():
    tests = [_kt("4B", "Q6_K", "Q8_0", 5.0, True), _kt("4B", "Q4_K_M", "Q6_K", -5.0, True),
             _kt("9B", "Q6_K", "Q8_0", 0.5, False), _kt("9B", "Q4_K_M", "Q6_K", 5.0, True), _pt(-0.05, True)]
    rels = dt.quality_relations(_h(tests), CONFIGS)
    assert rels[("4B-Q8_0", "4B-Q6_K")] == {"rel": "better", "source": "KLD 4B Q6_K > Q8_0", "contrary": False}
    assert rels[("4B-Q6_K", "4B-Q8_0")]["rel"] == "worse"
    assert rels[("4B-Q4_K_M", "4B-Q6_K")] == {"rel": "better", "source": "KLD 4B Q4_K_M > Q6_K", "contrary": True}
    assert rels[("9B-Q8_0", "9B-Q6_K")]["rel"] == "tie"
    assert rels[("9B-Q4_K_M", "4B-Q8_0")]["rel"] == "better"
    # other cross-size pairs are unmeasured, never a tie
    assert dt.qrel(rels, "4B-Q4_K_M", "9B-Q8_0") == "unmeasured"
    assert dt.qrel(None, "4B-Q8_0", "4B-Q6_K") == "unmeasured"
    # the paired test the other way round is contrary
    rels = dt.quality_relations(_h(tests[:4] + [_pt(0.05, True)]), CONFIGS)
    assert rels[("4B-Q8_0", "9B-Q4_K_M")] == {"rel": "better", "source": dt.PAIRED_TEST, "contrary": True}
    # not testable / missing reject -> unavailable
    assert dt.quality_relations({"verdict": "not testable", "missing": ["x"], "tests": tests[:2]}, CONFIGS) is None
    assert dt.quality_relations(_h([{k: v for k, v in t.items() if k != "reject"} for t in tests]), CONFIGS) is None


def test_quality_strict_chain_never_carries_a_tie():
    base = [_kt("9B", "Q6_K", "Q8_0", 5.0, True), _kt("9B", "Q4_K_M", "Q6_K", 5.0, True), _pt(-0.05, True)]
    chain = [_kt("4B", "Q6_K", "Q8_0", 5.0, True), _kt("4B", "Q4_K_M", "Q6_K", 0.3, False)]
    rels = dt.quality_relations(_h(chain + base), CONFIGS)
    assert dt.qrel(rels, "4B-Q8_0", "4B-Q4_K_M") == "unmeasured"
    rels = dt.quality_relations(_h([_kt("4B", "Q6_K", "Q8_0", 5.0, True), _kt("4B", "Q4_K_M", "Q6_K", 5.0, True)]
                                   + base), CONFIGS)
    assert rels[("4B-Q8_0", "4B-Q4_K_M")]["rel"] == "better"
    assert rels[("4B-Q8_0", "4B-Q4_K_M")]["source"].startswith("chain:")
    rels = dt.quality_relations(_h([_kt("4B", "Q6_K", "Q8_0", 5.0, True), _kt("4B", "Q4_K_M", "Q6_K", -5.0, True)]
                                   + base), CONFIGS)
    assert dt.qrel(rels, "4B-Q8_0", "4B-Q4_K_M") == "unmeasured"  # opposite directions


def test_source_scc_front_keeps_noise_cycles_and_is_never_empty():
    nodes = ["a", "b", "c", "d"]
    assert dt.source_sccs(nodes, {("a", "b"), ("b", "c"), ("c", "a"), ("a", "d")}) == [["a", "b", "c"]]
    assert dt.source_sccs(nodes, {("a", "b"), ("c", "d")}) == [["a"], ["c"]]
    assert dt.source_sccs(nodes, set()) == [["a"], ["b"], ["c"], ["d"]]


def test_lexicographic_pick_is_nonempty_and_keeps_maximal_members():
    rng = random.Random(1)
    for _ in range(300):
        cfgs = CONFIGS[:]
        vals = {c: {ax: {"m": rng.choice([1.0, 1.2, 2.0, 3.0]), "se": rng.choice([0.0, 0.05, 0.5])}
                    for ax in ("decode", "ttft", "memory")} for c in cfgs}
        tests = [_kt(s, hi, lo, rng.choice([-5.0, 5.0]), rng.random() < 0.6) for s in SIZES
                 for lo, hi in qa.KLD_PAIRS] + [_pt(rng.choice([-0.1, 0.1]), rng.random() < 0.6)]
        rels = dt.quality_relations(_h(tests), cfgs)
        rel_ab = {(a, b): dt.axis_rels(a, b, vals, rels) for a in cfgs for b in cfgs if a != b}
        for order in dt.WORKLOADS.values():
            p = dt.pick(cfgs, order, lambda ax, x, y: rel_ab[(x, y)][ax])
            assert p["configs"]
            for k, step in enumerate(p["steps"]):
                assert step["kept"]
                for y in step["kept"]:  # the kept set is an antichain under this step's rule
                    assert not any(rel_ab[(x, y)][step["axis"]] == "better"
                                   and ("quality" not in order[:k] or rel_ab[(x, y)]["quality"] == "tie")
                                   for x in step["kept"] if x != y)
                for e in step["eliminated"]:  # every eliminator was unbeaten when it eliminated
                    assert e["by"] and all(rel_ab[(b, e["config"])][step["axis"]] == "better" and b in step["kept"]
                                           for b in e["by"])
            assert p["configs"] == [c for c in cfgs if c in p["configs"]]  # canonical order, no tie-break


# ----------------------------------------------------------- whole table --

def test_default_tree_front_and_picks(base_table):
    _, t = base_table
    mac = t["core_a"]["macos-metal"]
    lv = mac["levels"]["2048"]
    assert [c for comp in lv["frontier"] for c in comp] == CONFIGS  # every config trades something
    # 9B-Q4_K_M beats 4B-Q8_0 (paired dNLL) but is itself below 9B-Q8_0 (KLD): no cross-metric chain (021 R11 fix)
    assert lv["picks"]["A-Q"]["status"] == "incomparable" and lv["picks"]["A-Q"]["configs"] == ["4B-Q8_0", "9B-Q8_0"]
    assert lv["picks"]["A-D"]["configs"] == ["4B-Q4_K_M"]
    assert lv["picks"]["A-P"]["configs"] == ["4B-Q8_0"]
    assert lv["picks"]["A-M"]["configs"] == ["4B-Q4_K_M"]
    assert "NO_IMATRIX_Q4" in lv["picks"]["A-D"]["caveats"] and "CONTAMINATION" in lv["caveats"]
    assert t["quality"]["status"] == "ok" and t["core_b"]["status"] == "ok"


def test_cross_size_unmeasured_keeps_slower_size_and_never_acts_as_tie(tmp_path):
    results = full_tree(tmp_path, q_rows=quality_rows(paired="no-difference"))
    lv = build(results)["core_a"]["macos-metal"]["levels"]["2048"]
    a_q = lv["picks"]["A-Q"]
    assert a_q["status"] == "incomparable" and a_q["configs"] == ["4B-Q8_0", "9B-Q8_0"]
    # 4B-Q8_0 is faster in TTFT beyond noise but cannot remove 9B-Q8_0 (quality unmeasured across sizes)
    assert dt.scalar_rel("ttft", lv["configs"]["4B-Q8_0"]["ttft"], lv["configs"]["9B-Q8_0"]["ttft"]) == "better"


def test_tie_dependent_dominance(tmp_path):
    q = quality_rows({"4B-Q6_K": (0.0021, 0.0001)})  # Q8_0 vs Q6_K not resolved
    spec = {("4B-Q6_K", L): {"ttft": [L / PREFILL["4B-Q8_0"] * f for f in (0.999, 1.0, 1.001)]} for L in LS}
    results = full_tree(tmp_path, q_rows=q, mac_rows=core_a_rows(spec))
    lv = build(results)["core_a"]["macos-metal"]["levels"]["2048"]
    dom = {d["config"]: d for d in lv["dominated"]}
    assert dom["4B-Q8_0"]["dominated_by"] == ["4B-Q6_K"] and dom["4B-Q8_0"]["tie_dependent"]
    assert dom["4B-Q8_0"]["tie_text"] == dt.TIE_TEXT
    assert lv["configs"]["4B-Q8_0"]["status"] == "candidate"


def test_quality_unavailable_means_not_decidable_but_evidence_stays(tmp_path):
    for kw in ({"quality": False}, {"q_rows": quality_rows(drop_kld=("9B-Q6_K",))}):
        results = full_tree(tmp_path / str(len(kw)) / next(iter(kw)), **kw)
        t = build(results)
        lv = t["core_a"]["wsl2-cuda"]["levels"]["8192"]
        for w in dt.WORKLOADS:
            assert lv["picks"][w]["status"].startswith("not decidable: quality relations unavailable")
            assert "configs" not in lv["picks"][w]
        assert lv["frontier"] is None and lv["configs"]["4B-Q8_0"]["decode"] and lv["ha5a"]["configs"]
    assert "9B Q6_K" in t["core_a"]["macos-metal"]["levels"]["2048"]["picks"]["A-Q"]["status"]


def test_status_precedence_and_provisional(tmp_path):
    spec = {("9B-Q8_0", 32768): {"oom": True}, ("9B-Q6_K", 32768): {"error": True},
            ("4B-Q6_K", 32768): {"drop_decode_rep": 3}, ("4B-Q8_0", 32768): {"drop_fit_rep": 2},
            ("4B-Q4_K_M", 32768): {"skip": True}}
    results = full_tree(tmp_path, rtx_rows=core_a_rows({**spec, ("9B-Q4_K_M", 32768): {"device": [16000] * 3}}))
    lv = build(results)["core_a"]["wsl2-cuda"]["levels"]["32768"]
    st = {c: (v["status"], v["reason"]) for c, v in lv["configs"].items()}
    assert st["9B-Q8_0"] == ("oom", None)
    assert st["9B-Q6_K"] == ("error", "decode: bench: exit 1")
    assert st["4B-Q6_K"] == ("incomplete", "missing: decode")
    assert st["4B-Q8_0"] == ("incomplete", "missing: memory")
    assert st["4B-Q4_K_M"] == ("not measured", None)
    assert st["9B-Q4_K_M"] == ("spill", None)
    assert lv["unknown_configs"] == ["4B-Q8_0", "4B-Q6_K", "4B-Q4_K_M", "9B-Q6_K"]
    assert lv["picks"]["A-Q"]["status"] == "no candidate" and "PROVISIONAL" in lv["caveats"]
    lv2 = build(results)["core_a"]["wsl2-cuda"]["levels"]["2048"]
    assert not lv2["provisional"]  # oom / spill elsewhere never make another level provisional


def test_memory_axis_per_platform(tmp_path):
    spec = {("9B-Q4_K_M", 2048): {"host": [9000] * 3}}
    results = full_tree(tmp_path, mac_rows=core_a_rows(spec), rtx_rows=core_a_rows(spec))
    t = build(results)
    mac = t["core_a"]["macos-metal"]["levels"]["2048"]["configs"]["9B-Q4_K_M"]
    rtx = t["core_a"]["wsl2-cuda"]["levels"]["2048"]["configs"]["9B-Q4_K_M"]
    assert mac["memory_axis"]["m"] == DEVICE["9B-Q4_K_M"] + 128 + 9000
    assert rtx["memory_axis"]["m"] == DEVICE["9B-Q4_K_M"] + 128  # VRAM only; host shown apart
    assert rtx["memory_display"]["host_max"] == 9000
    assert rtx["memory_display"]["spill_margin_mib"] == 16303 - 811.0 + 89.0 - (DEVICE["9B-Q4_K_M"] + 128)
    base = build(full_tree(tmp_path / "b"))["core_a"]["wsl2-cuda"]["levels"]["2048"]
    assert base["picks"] == t["core_a"]["wsl2-cuda"]["levels"]["2048"]["picks"]


# ------------------------------------------------------------- selection --

def test_session_selection_zero_one_two(tmp_path):
    results = full_tree(tmp_path)
    write_core_a(results, "macos-metal", "mac-pilot", session={"mode": "pilot"})
    write_core_a(results, "macos-metal", "mac-drain", session={"battery_drain": {"pct": 15}})
    write_core_a(results, "wsl2-cuda", "rtx-bdrift", session={"session_b": {"drift": True}})
    write_core_a(results, "wsl2-cuda", "rtx-wrongdir", session={"platform": "macos-metal"})
    write_quality(results, "q-mac", session={"platform": "macos-metal"})
    t = build(results)
    ex = {e["session_id"]: e for e in t["provenance"]["excluded"]}
    assert "pilot" in ex["mac-pilot"]["problems"][0] and "battery drain" in ex["mac-drain"]["problems"][0]
    assert "B drift" in ex["rtx-bdrift"]["problems"][0]
    assert any("not the directory" in p for p in ex["rtx-wrongdir"]["problems"])
    assert "wsl2-cuda" in ex["q-mac"]["problems"][0]
    assert t["core_a"]["macos-metal"]["session_id"] == "mac-a"
    write_core_a(results, "macos-metal", "mac-b")
    with pytest.raises(dt.DecisionTableError, match="2 valid formal sessions"):
        build(results)
    empty = tmp_path / "empty" / "results"
    empty.mkdir(parents=True)
    t = build(empty)
    assert t["core_a"]["macos-metal"]["status"] == "pending" and t["core_b"]["status"] == "pending"
    assert t["quality"]["status"] == "pending"


def test_excluded_sessions_rows_are_never_read(tmp_path, monkeypatch):
    results = full_tree(tmp_path)
    write_core_a(results, "macos-metal", "mac-pilot", session={"mode": "pilot"})
    seen = []
    orig = Path.read_text

    def spy(self, *a, **k):
        seen.append(str(self))
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    build(results)
    assert not [p for p in seen if "mac-pilot" in p and not p.endswith("session.json")]
    assert not [p for p in seen if "/raw/" in p or "/env/" in p]


def test_mixed_selection(tmp_path):
    results = full_tree(tmp_path, mixed=False)
    write_mixed(results, "m-other", uniform_sid="someone-else", regenerate=False)
    write_mixed(results, "m-server", session={"server": ["other"]}, regenerate=False)
    t = build(results)
    ex = {e["session_id"]: e for e in t["provenance"]["excluded"]}
    assert "derived from 'someone-else'" in ex["m-other"]["problems"][0]
    assert "server" in ex["m-server"]["problems"][0]
    assert t["core_b"]["levels"]["2048"]["picks"]["B-X"]["status"] == "pending"
    write_mixed(results, "m1")
    write_mixed(results, "m2", regenerate=False)
    with pytest.raises(dt.DecisionTableError, match="core_b_mixed: 2 valid"):
        build(results)


# ------------------------------------------------------------- integrity --

def _tamper(p: Path, fn) -> None:
    a = json.loads(p.read_text())
    fn(a)
    p.write_text(json.dumps(a))


@pytest.mark.parametrize("fn", [
    lambda a: a["cells"]["4B-Q8_0-2048"]["decode"].update(m=1.0),
    lambda a: a["spill"].append(["4B", "Q8_0", 2048]),
    lambda a: a["hypotheses"]["H-A2"].update(verdict="not refuted"),
    lambda a: a["platform_params"].update(b_mib=400.0),
    lambda a: a.update(session="other"),
])
def test_integrity_core_a(tmp_path, fn):
    results = full_tree(tmp_path)
    _tamper(results / "core_a" / "wsl2-cuda" / "rtx-a" / "analysis.json", fn)
    with pytest.raises(dt.DecisionTableError, match="analysis.json does not match"):
        build(results)


@pytest.mark.parametrize("fn", [
    lambda a: a["h_a5"]["tests"][0].update(p=0.5),
    lambda a: a["downstream"][0].update(delta_acc=0.5),
    lambda a: a["downstream"][0].update(n_boot=1000),
    lambda a: a.update(platform="macos-metal"),
])
def test_integrity_quality(tmp_path, fn):
    results = full_tree(tmp_path)
    _tamper(results / "quality" / "q-formal" / "analysis.json", fn)
    with pytest.raises(dt.DecisionTableError, match="quality q-formal: analysis.json does not match"):
        build(results)


def test_integrity_quality_ci95_is_copied_and_labelled(tmp_path):
    results = full_tree(tmp_path)
    _tamper(results / "quality" / "q-formal" / "analysis.json", lambda a: a["downstream"][0].update(ci95=[9, 9]))
    t = build(results)
    assert t["quality"]["downstream"][0]["ci95"] == [9, 9] and not t["quality"]["downstream"][0]["ci95_reverified"]
    assert "DOWNSTREAM_CI_NOT_REVERIFIED" in t["quality"]["labels"]


def test_integrity_core_b_stale_analysis_without_mixed(tmp_path):
    results = full_tree(tmp_path)
    regenerate_core_b(results, "b-uniform", None)  # made before the mixed session existed
    with pytest.raises(dt.DecisionTableError, match="uniform \\+ mixed"):
        build(results)


def test_refusal_writes_nothing(tmp_path):
    results = full_tree(tmp_path)
    out = tmp_path / "out"
    assert dt.run(results, PINS, out) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    write_core_a(results, "macos-metal", "mac-b")
    assert dt.run(results, PINS, out) == 2
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


# ------------------------------------------------------------- isolation --

def _strip_prov(t):
    t = copy.deepcopy(t)
    t.pop("provenance")
    return t


def test_platform_isolation(tmp_path):
    a = build(full_tree(tmp_path / "a"))
    spec = {(c, L): {"decode": [1.0, 1.1, 1.2]} for c in CONFIGS for L in LS}
    b = build(full_tree(tmp_path / "b", mac_rows=core_a_rows(spec)))
    assert a["core_a"]["wsl2-cuda"] == b["core_a"]["wsl2-cuda"]
    assert a["core_a"]["macos-metal"] != b["core_a"]["macos-metal"]
    md_a, md_b = dt.render_md(a), dt.render_md(b)
    seg = lambda md: md[md.index("## Core A — wsl2-cuda"):md.index("## Core B")]  # noqa: E731
    assert seg(md_a) == seg(md_b)


def test_core_a_and_core_b_never_share_a_section(base_table):
    _, t = base_table
    core_b = json.dumps(t["core_b"])
    assert not any(c in core_b for c in CONFIGS)
    md = dt.render_md(t)
    b_md = md[md.index("## Core B"):md.index("## Quality")]
    assert not any(c in b_md for c in CONFIGS)
    a_md = md[md.index("## Core A — macos-metal"):md.index("## Core B")]
    assert "C_knee" not in a_md and "W4A16" not in a_md


def test_verdicts_are_verbatim_counted_and_inert(tmp_path, base_table):
    results, t = base_table
    a = json.loads((results / "core_a" / "macos-metal" / "mac-a" / "analysis.json").read_text())
    v = t["core_a"]["macos-metal"]["verdicts"]
    assert {k: x["copy"] for k, x in v.items()} == a["hypotheses"]
    assert all(x["verdict"] in ("refuted", "not refuted") for x in v.values())
    assert "confirmed" not in json.dumps(t) and "confirmed" not in dt.render_md(t)
    sec = dt.core_a_section("macos-metal", None, PINS, None, "x", [])
    assert sec["status"] == "pending"
    # picks never read verdicts: flipping every verdict leaves the picks unchanged
    q = t["quality"]
    rels = dt.quality_relations(json.loads((results / "quality" / "q-formal" / "analysis.json").read_text())["h_a5"],
                                CONFIGS)
    rows = [json.loads(x) for x in (results / "core_a" / "macos-metal" / "mac-a" / "rows.jsonl").read_text().splitlines()]
    flipped = copy.deepcopy(a)
    for h in flipped["hypotheses"].values():
        h["verdict"] = "refuted" if h["verdict"] == "not refuted" else "not refuted"
    sel = {"session_id": "mac-a", "session": json.loads((results / "core_a" / "macos-metal" / "mac-a" / "session.json")
                                                           .read_text()), "rows_sorted": rows}
    s1 = dt.core_a_section("macos-metal", {**sel, "analysis": a}, PINS, rels, None, q["downstream"])
    s2 = dt.core_a_section("macos-metal", {**sel, "analysis": flipped}, PINS, rels, None, q["downstream"])
    assert {L: lv["picks"] for L, lv in s1["levels"].items()} == {L: lv["picks"] for L, lv in s2["levels"].items()}
    # H-B4 without a mixed session: 0 testable units is labelled
    t2 = build(full_tree(tmp_path, mixed=False))
    assert t2["core_b"]["verdicts"]["H-B4"]["unit_counts"]["labels"] == ["NO_TESTABLE_UNITS"]


# ----------------------------------------------------------------- Core B --

def test_core_b_bt_fields_and_flags(base_table, tmp_path):
    _, t = base_table
    lv = t["core_b"]["levels"]["2048"]
    bt = lv["picks"]["B-T"]
    assert bt["C"] == lv["c_knee"] == 4 and lv["c_max"] == 8
    assert bt["gain_bound_pct"] == pytest.approx((1 / 0.95 - 1) * 100)
    assert "H_B1_UNREFUTABLE" in bt["flags"] and "KNEE_AT_GRID_TOP" not in bt["flags"]
    # knee at the grid top from the 007 rule (L = 32768 grid ends at 2 here) vs at GRID_CAP 128
    results = tmp_path / "results"
    write_core_b(results, rows=b_rows(cap={32768: 2}), analysis="uniform-only")
    lv32 = build(results)["core_b"]["levels"]["32768"]["picks"]["B-T"]
    assert "KNEE_AT_GRID_TOP" in lv32["flags"] and "007 rule" in lv32["grid_top_note"]
    grids = {"2048": [1, 64, 128], "8192": [1, 2], "32768": [1, 2]}
    r2 = tmp_path / "r2" / "results"
    write_core_b(r2, rows=b_rows(grids=grids, cap={2048: 128}), grids=grids, analysis="uniform-only")
    top = build(r2)["core_b"]["levels"]["2048"]["picks"]["B-T"]
    assert top["C"] == 128 and "GRID_CAP" in top["grid_top_note"]


def test_b_l_has_no_concurrency_pick_and_latency_never_moves_b_t(tmp_path):
    rows = b_rows()
    for r in rows:
        r["p95_ttft_ms"] *= 50 if r["C"] == 4 else 1
        r["p95_tpot_ms"] *= 7
    results = tmp_path / "results"
    write_core_b(results, rows=rows, analysis="uniform-only")
    lv = build(results)["core_b"]["levels"]["2048"]
    assert "C" not in lv["picks"]["B-L"] and lv["picks"]["B-L"]["statement"] == dt.BL_STATEMENT
    assert lv["picks"]["B-T"]["C"] == 4
    assert set(lv["picks"]["B-L"]["latency"]) == {"C=1", "C_knee", "C_max"}


def test_core_b_incomplete_level_has_no_operating_point(tmp_path):
    results = tmp_path / "results"
    write_core_b(results, rows=b_rows(drop=((8192, 2, 3),)), analysis="uniform-only")
    lv = build(results)["core_b"]["levels"]["8192"]
    assert lv["status"] == "no operating point" and lv["missing_c"] == [2] and lv["picks"] is None
    assert [e["n_valid"] for e in lv["curve"]] == [3, 2, 3]
    assert "output_throughput" not in lv["curve"][1]


def test_bx_states(base_table, tmp_path):
    results, t = base_table
    a = json.loads((results / "core_b" / "b-uniform" / "analysis.json").read_text())
    for L in ("2048", "8192"):
        bx = t["core_b"]["levels"][L]["picks"]["B-X"]
        assert bx["status"] == "ok" and bx["h_b4_unit"] == next(u for u in a["hypotheses"]["H-B4"]["units"]
                                                                 if str(u["L"]) == L)
        assert bx["statement"] == dt.BX_STATEMENT
    assert t["core_b"]["levels"]["32768"]["picks"]["B-X"] == {"status": "not defined (007)"}
    assert t["core_b"]["quality"] == dt.CORE_B_QUALITY


# ---------------------------------------------------------------- labels --

def test_labels_and_conditions(base_table):
    _, t = base_table
    mac, rtx = t["core_a"]["macos-metal"], t["core_a"]["wsl2-cuda"]
    assert "QUALITY_PLATFORM_MAC" in mac["labels"] and "MAC_POWER_CONDITION" in mac["labels"]
    assert "QUALITY_PLATFORM_RTX" in rtx["labels"] and "SPILL_EXCLUDED" in rtx["labels"]
    assert mac["conditions"]["powermode"] == 2 and "powermode" not in rtx["conditions"]
    assert all("QUALITY_CTX_512" in s["labels"] for s in (mac, rtx))
    md = dt.render_md(t)
    assert "not measured on Metal" in md and "-c 512" in md


def test_ha5a_side_by_side_on_every_platform_and_level(base_table):
    _, t = base_table
    for p in dt.PLATFORMS:
        for L, lv in t["core_a"][p]["levels"].items():
            h = lv["ha5a"]
            assert set(h["configs"]) == {"9B-Q4_K_M", "4B-Q8_0"}
            assert h["paired_test"]["ci95_note"] == "descriptive only (§2)"
            assert h["relations_9B_vs_4B"]["quality"] == "better"


def test_picks_are_on_the_front_and_never_bf16(base_table):
    _, t = base_table
    for p in dt.PLATFORMS:
        for lv in t["core_a"][p]["levels"].values():
            front = {c for comp in lv["frontier"] for c in comp}
            for w in dt.WORKLOADS:
                assert set(lv["picks"][w]["configs"]) <= front
                assert not any("BF16" in c for c in lv["picks"][w]["configs"])


def test_not_supported_lists_gaps(tmp_path):
    t = build(full_tree(tmp_path, mixed=False, q_rows=quality_rows(paired="no-difference")))
    items = {(x["item"], x["reason"]) for x in t["not_supported"]}
    assert ("quality on Metal", "not measured (020)") in items
    assert any(i.startswith("core_b H-B4") for i, _ in items)
    assert any(i.startswith("core_a/macos-metal L=2048 A-Q") and r == "incomparable" for i, r in items)
    assert any(i == "quality 4B-Q8_0 vs 9B-Q8_0" for i, _ in items)


# ----------------------------------------------------------- determinism --

def test_determinism_and_no_host_details(tmp_path):
    results = full_tree(tmp_path)
    out = tmp_path / "out"
    assert dt.run(results, PINS, out) == 0
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert dt.run(results, PINS, out) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first
    text = b"".join(first.values()).decode()
    assert str(tmp_path) not in text and str(ROOT) not in text
    # the row order in the file does not matter
    rows = core_a_rows()
    random.Random(3).shuffle(rows)
    shuffled = tmp_path / "s" / "results"
    write_core_a(shuffled, "macos-metal", "mac-a", rows)
    write_core_a(shuffled, "wsl2-cuda", "rtx-a")
    write_quality(shuffled, "q-formal")
    write_core_b(shuffled)
    write_mixed(shuffled)
    assert _strip_prov(build(shuffled)) == _strip_prov(build(results))


def test_cli_has_no_selection_args():
    with pytest.raises(SystemExit):
        dt.main(["--session", "x"])
    with pytest.raises(SystemExit):
        dt.main(["--out", "x"])


def test_markdown_renders_every_section(base_table):
    _, t = base_table
    md = dt.render_md(t)
    for h in ("## How to read", "## Core A — macos-metal", "## Core A — wsl2-cuda", "## Core B",
              "## Quality", "## What the evidence does not support", "### Fit map"):
        assert h in md
    assert "| A-Q quality first" in md and "C = C_knee = 4" in md
    assert "A-Q: 4B-Q6_K vs 4B-Q8_0 — dropped at quality by 4B-Q8_0;" in md  # the eliminator is named
    assert "`X ‖ Y`: incomparable (a quality relation among them is unmeasured" in md


# ------------------------------------------------- review fixes (2026-10-10) --

def test_pick_never_eliminates_through_an_eliminated_member():
    """x beats y and w beats x on TTFT (quality ties); w vs y quality is unmeasured -> y stays (021 R11 fix)."""
    q = {("w", "x"): "tie", ("x", "y"): "tie", ("w", "y"): "unmeasured"}
    t = {("w", "x"): "better", ("x", "y"): "better", ("w", "y"): "better"}

    def rel(axis, a, b):
        if axis == "quality":
            return q.get((a, b)) or q.get((b, a))
        if (a, b) in t:
            return t[(a, b)]
        return "worse" if (b, a) in t else "tie"

    p = dt.pick(["w", "x", "y"], ("quality", "ttft"), rel)
    assert p["configs"] == ["w", "y"] and p["status"] == "incomparable"
    assert p["steps"][1]["eliminated"] == [{"config": "x", "by": ["w"]}]


def test_h_b1_inconclusive_and_h_a1_boundary_are_not_testable(tmp_path):
    rows = b_rows()
    for r in rows:
        r["p95_ttft_ms"] = 100.0 + 0.01 * r["restart"]  # no C reaches 3 x the knee; every grid stops below 4 x C_knee
    results = tmp_path / "results"
    write_core_b(results, rows=rows, analysis="uniform-only")
    h = build(results)["core_b"]["verdicts"]["H-B1"]
    assert h["unit_counts"]["counts"] == {"inconclusive": 3} and h["unit_counts"]["labels"] == ["NO_TESTABLE_UNITS"]
    assert dt._unit_counts("H-A1", {"units": [{"predicted": "boundary", "observed": "fit"},
                                               {"predicted": "fit", "observed": "error"}]})["testable"] == 0


def test_contamination_note_is_always_on_the_ha5a_pair(tmp_path):
    t = build(full_tree(tmp_path, q_rows=quality_rows(paired="no-difference")))
    pair = next(r for r in t["quality"]["relations"] if {r["a"], r["b"]} == {"9B-Q4_K_M", "4B-Q8_0"})
    assert pair["rel"] == "tie" and pair["caveats"] == ["CONTAMINATION"]
    for p in dt.PLATFORMS:
        for lv in t["core_a"][p]["levels"].values():
            assert "CONTAMINATION" in lv["ha5a"]["caveats"]


def test_alternatives_carry_same_size_downstream_and_mark_cross_size(base_table):
    _, t = base_table
    a_d = t["core_a"]["macos-metal"]["levels"]["2048"]["picks"]["A-D"]  # decided 4B-Q4_K_M
    alts = {a["config"]: a for a in a_d["alternatives"]}
    same = alts["4B-Q8_0"]["vs"]["4B-Q4_K_M"]["downstream"]
    assert same and all(d["size"] == "4B" for d in same)
    assert alts["9B-Q8_0"]["vs"]["4B-Q4_K_M"]["downstream"] == "not compared"
    a_q = t["core_a"]["macos-metal"]["levels"]["2048"]["picks"]["A-Q"]  # incomparable 4B-Q8_0 / 9B-Q8_0
    assert a_q["relations_among_picks"] == [{"a": "4B-Q8_0", "b": "9B-Q8_0", **{
        "quality": "unmeasured", "decode": "better", "ttft": "better", "memory": "better"}}]


def test_tie_dependent_is_per_edge(tmp_path):
    q = quality_rows({"4B-Q4_K_M": (0.0061, 0.0002)})  # 4B Q6_K vs Q4_K_M not resolved
    spec = {}
    for L in LS:
        same = {"decode": [DECODE["4B-Q8_0"] * (1 - L / 400000) + d for d in (-0.5, 0.0, 0.5)],
                "ttft": [L / PREFILL["4B-Q8_0"] * f for f in (0.999, 1.0, 1.001)], "device": [DEVICE["4B-Q8_0"] + L // 16] * 3}
        spec[("4B-Q6_K", L)] = same
        spec[("4B-Q4_K_M", L)] = {"ttft": same["ttft"]}
    lv = build(full_tree(tmp_path, q_rows=q, mac_rows=core_a_rows(spec)))["core_a"]["macos-metal"]["levels"]["2048"]
    d = next(x for x in lv["dominated"] if x["config"] == "4B-Q6_K")
    assert d["dominated_by"] == ["4B-Q8_0", "4B-Q4_K_M"]
    assert d["relations"]["4B-Q8_0"]["tie_dependent"] is False and d["relations"]["4B-Q4_K_M"]["tie_dependent"] is True
    assert d["tie_dependent"] is False  # dominated anyway by a resolved quality advantage
    t = build(full_tree(tmp_path / "b", q_rows=q, mac_rows=core_a_rows(spec)))
    assert not any(x["item"].startswith("core_a/macos-metal") and "4B-Q6_K dominated" in x["item"]
                   for x in t["not_supported"])
    assert any(x["item"] == "core_a/wsl2-cuda L=2048 4B-Q6_K dominated" for x in t["not_supported"])  # tie only


def test_not_supported_keeps_downstream_ci_item_when_quality_unavailable(tmp_path):
    t = build(full_tree(tmp_path, q_rows=quality_rows(drop_kld=("9B-Q6_K",))))
    assert t["quality"]["status"] == "unavailable" and t["quality"]["downstream"]
    assert {"item": "downstream ci95", "reason": "copied, not re-verified"} in t["not_supported"]


def test_excluded_mixed_rows_are_never_read_and_paths_are_redacted(tmp_path, monkeypatch):
    results = full_tree(tmp_path, mixed=False)
    write_mixed(results, "m-env", session={"env_valid": False}, regenerate=False)
    write_mixed(results, "m-srv", session={"server_env": {"HF_HOME": "/home/someone/.cache/huggingface"}},
                regenerate=False)
    seen, orig = [], Path.read_text

    def spy(self, *a, **k):
        seen.append(str(self))
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    t = build(results)
    assert not [x for x in seen if ("m-env" in x or "m-srv" in x) and x.endswith("rows.jsonl")]
    text = json.dumps(t) + dt.render_md(t)
    assert "/home/someone" not in text and "<path>" in text


def test_bx_is_not_ok_without_a_tested_h_b4_unit(tmp_path):
    results = tmp_path / "results"
    write_core_b(results, rows=b_rows(drop=((8192, 2, 3),)), analysis="none")
    write_mixed(results, knees={"2048": 4})  # the frozen mixed plan skips 8192 (no uniform C_knee there)
    lv = build(results)["core_b"]["levels"]
    assert lv["8192"]["b_x"]["status"] == "not measured (no uniform C_knee at this L; the mixed plan skipped it)"
    assert lv["2048"]["picks"]["B-X"]["status"] == "ok"
    r2 = tmp_path / "r2" / "results"
    write_core_b(r2, analysis="none")
    m = write_mixed(r2, regenerate=False)
    rows = [json.loads(x) for x in (m / "rows.jsonl").read_text().splitlines()]
    _write_rows(m / "rows.jsonl", [r for r in rows if not (r["L"] == 2048 and r["restart"] == 2)])
    regenerate_core_b(r2, "b-uniform", "b-mixed")
    t = build(r2)
    bx = t["core_b"]["levels"]["2048"]["picks"]["B-X"]
    assert bx["status"] == "not testable" and "MIXED_AT_KNEE_ONLY" not in t["core_b"]["levels"]["2048"]["caveats"]
    assert {"item": "core_b L=2048 B-X", "reason": "not testable"} in t["not_supported"]


def test_zero_run_mixed_plan_is_still_the_mixed_session(tmp_path):
    results = tmp_path / "results"
    write_core_b(results, rows=b_rows(drop=((2048, 2, 3), (8192, 2, 3))), analysis="none")
    write_mixed(results, knees={})
    t = build(results)
    assert t["core_b"]["mixed_session"] == "b-mixed"
    assert all(t["core_b"]["levels"][L]["b_x"]["status"].startswith("not measured") for L in ("2048", "8192"))


def test_unknown_configs_are_not_called_infeasible(tmp_path):
    spec = {("9B-Q6_K", 8192): {"error": True}, ("9B-Q8_0", 8192): {"oom": True}}
    md = dt.render_md(build(full_tree(tmp_path, rtx_rows=core_a_rows(spec))))
    seg = md[md.index("## Core A — wsl2-cuda"):md.index("## Core B")]
    assert "not deployable" not in seg
    assert "| not feasible (oom / spill) | none | 9B-Q8_0 (oom) | none |" in seg
    assert "| unknown † (not measured / error / incomplete) | none | 9B-Q6_K (error) | none |" in seg


def test_contrary_relation_is_flagged_where_it_acts(tmp_path):
    rng = random.Random(7)
    base = [2.0 + rng.random() for _ in range(30)]
    q = quality_rows()
    for r in q:
        if r["kind"] == "ppl" and r["size"] == "9B":
            r["metrics"]["chunk_nll"] = [b + 0.05 for b in base]  # 9B-Q4_K_M worse: contrary to H-A5(a)
    t = build(full_tree(tmp_path, q_rows=q))
    lv = t["core_a"]["macos-metal"]["levels"]["2048"]
    assert "CONTRARY_TO_PREDICTION" in lv["caveats"]
    seg = dt.render_md(t)
    assert seg.count("CONTRARY_TO_PREDICTION") > 1  # not only in the glossary
    assert any(x["reason"] == "CONTRARY_TO_PREDICTION" for x in t["not_supported"])


def test_provenance_records_the_pins_file_and_each_session_pins(tmp_path):
    import hashlib
    results = full_tree(tmp_path)
    sha = hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()
    t = dt.build(results, PINS, pins_sha256=sha)
    assert t["provenance"]["pins"] == {"configs/pins.toml": sha}
    assert all("session_pins_sha256" in i for i in t["provenance"]["inputs"])
