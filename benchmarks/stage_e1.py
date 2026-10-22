#!/usr/bin/env python3
"""Stage E1-D (decision 041 v3 §1.3-1.5): where an 8K slip starts, described on existing data with the predefined
links L1-L5 of the slip chain. Exploratory and descriptive only (descriptive_only = true); no GPU.

Data (every input's sha256 is recorded; every raw file is checked against its own session's raw_manifest.json before
it is read; raw copies, local or fetched, live where tools/rtx/fetch_raw.sh puts them,
results/capacity/c0-step0/raw/<session>/):
- advanced: formal-20261011-adv 8K uniform. B0, G, K3 (the slip conditions): committed derived requests (server
  arrival, Q, P, F, TTFT), steps.jsonl (local copies), spans.jsonl (fetched: e2e, so completion = arrival + e2e; else
  the closed-loop proxy completion(8g + j) ~ arrival(8(g + 1) + j)), bench.json (fetched: client start_times and
  latencies for the client-side send lag, never combined with the server clock). T, TG: auxiliary only (their TTFT
  structure differs; no slip classification). Every run: the position-0 F spikes and, with steps (T / TG fetched),
  the steps stamped between 8g's arrival and its own scheduling s = arrival + server_ttft - P (plus ANCHOR_TAU_MS for
  the stamps' logging delay).
- c0: explore-c0-20261016's 24 benches: committed derived TTFT / latency; bench.json (fetched) for start_times.
- core: formal-20261009-coreb-uniform's three C 8 benches (local copies): TTFT, latency, start_times.
A missing raw file blocks only what depends on it (listed in "blocked"); a file that fails its manifest stops the
analysis. Nothing real is read before freeze.e1 = true with freeze.e1_harness_sha256 equal to harness_sha256().

Per bench: slips by c0_analysis.classify_bench (028); group g = idx 8g..8g+7; slip group k = start idx // 8; baselines
over g 1..23 that are neither slipped nor a slip's precursor (k - 1); a slip whose k - 1 also slipped is "continuous"
(counted once, L3 not evaluated); slips outside g 1..23 (k 24) are listed but not counted in the chain verdict
(041 v3 §1.4).
- L1 (advanced): F(8k) > f_spike_ms.
- L2 (advanced): n_dec(k) = steps stamped after 8k's arrival before the first step with ctx_tok > 0; holds when it is
  the median over g 1..23 plus 1.
- L3: min_j over j 1..7 of (lat(8(k-1)+j) - lat(8(k-1))) - that position's baseline median < -slip_ms.
- L4: gap(k) = time(8k+1) - time(8k) < gap_ms (server arrival for advanced, client start for c0 / core).
- L5 (advanced): mean over j 1..7 of Q(8k+j) - that position's baseline median > slip_ms.
- send lag (c0 / core / advanced client, one clock each): the (C + m)-th send in time order minus the m-th
  completion in time order; per slip the lags of 8k+1..8k+7.
A dataset's chain holds when at most chain_tolerance evaluable, counted episodes have a failing link and L1 holds in
at most l1_false_max of the non-slipped groups; the three datasets are reported separately.

Usage:
    python benchmarks/stage_e1.py fetch-list | while read -r args; do tools/rtx/fetch_raw.sh $args; done
                                                    # after the freeze, only while gpu_status.sh prints FREE
    python benchmarks/stage_e1.py analyze --out results/stage_e/e1-<date>/e1.json      # a new file, after the freeze
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import adv_decomp as ad  # noqa: E402
import adv_plan as ap  # noqa: E402
import c0_analysis as ca  # noqa: E402

C, L = 8, 8192
BASE_GROUPS = range(1, 24)
CONFIG = ROOT / "configs" / "stage_e1.toml"
# steps.jsonl stamps are the runner's floored wall-clock milliseconds when it read each iteration log line
# (run_adv.pump_log), not the step's end: the stamp of the step after which 8g is scheduled lands -0.9 to +3.1 ms
# from s on the seen B0/G/K3 data, so the window ends ANCHOR_TAU_MS after s (decode steps take >= ~11 ms).
STAMP_RESOLUTION_MS = 1.0
ANCHOR_TAU_MS = 5.0
HARNESS_FILES = ("benchmarks/stage_e1.py", "benchmarks/c0_analysis.py")      # c0_analysis has no guard test
RESULTS = ("results", "stage_e")
ADV = Path("results/advanced/formal-20261011-adv")
C0 = Path("results/capacity/explore-c0-20261016")
CORE = Path("results/core_b/formal-20261009-coreb-uniform")
COPIES = Path("results/capacity/c0-step0/raw")           # tools/rtx/fetch_raw.sh's DEST root
SLIP_CONDS = ("B0", "G", "K3")
AUX_CONDS = ("T", "TG")
ADV_TAGS = tuple(f"{c}-8192-uniform-r{r}" for c in SLIP_CONDS + AUX_CONDS for r in (1, 2, 3))
C0_BENCHES = tuple(f"b{b}-n{n}" for b in range(1, 7) for n in range(1, 5))
# fetched read-only after freeze.e1 (041 v3 §1.7), exactly what the readers open: (results subdir, session, raw files)
FETCH = (("advanced", "formal-20261011-adv",
          tuple(f"{t}.{x}" for t in ADV_TAGS if t.split("-")[0] in SLIP_CONDS for x in ("spans.jsonl", "bench.json"))
          + tuple(f"{t}.steps.jsonl" for t in ADV_TAGS if t.split("-")[0] in AUX_CONDS)),
         ("capacity", "explore-c0-20261016", tuple(f"{b}.bench.json" for b in C0_BENCHES)))
CONFIG_KEYS = {"schema_version": None, "freeze": {"e1", "e1_harness_sha256"},
               "e1": {"f_spike_ms", "slip_ms", "gap_ms", "chain_tolerance", "l1_false_max"}}
E1_VALUES = {"f_spike_ms": 35.0, "slip_ms": ca.SLIP_MS, "gap_ms": 200.0, "chain_tolerance": 1, "l1_false_max": 0.02}


class StageE1Error(RuntimeError):
    pass


def harness_sha256(root=ROOT) -> str:
    h = hashlib.sha256()
    for rel in HARNESS_FILES:
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


def config_problems(cfg: dict) -> list:
    if set(cfg) != set(CONFIG_KEYS):
        return [f"top-level keys {sorted(cfg)} are not {sorted(CONFIG_KEYS)}"]
    problems = [f"[{t}] keys {sorted(cfg[t])} are not {sorted(k)}" for t, k in CONFIG_KEYS.items()
                if k is not None and set(cfg[t]) != k]
    if problems:
        return problems
    problems = [f"e1.{k} {cfg['e1'][k]!r} is not 041 v3's {v!r}" for k, v in E1_VALUES.items()
                if cfg["e1"][k] != v or type(cfg["e1"][k]) is not type(v)]
    f = cfg["freeze"]
    pin = f["e1_harness_sha256"]
    if f["e1"] is True and not (isinstance(pin, str) and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin)):
        problems.append("freeze.e1 = true needs freeze.e1_harness_sha256 (64 hex)")
    elif f["e1"] is not True and pin != "":
        problems.append("freeze.e1_harness_sha256 is set before freeze.e1")
    return problems


def out_problems(out: Path, root=ROOT) -> list:
    """--out is a file in a new session directory results/stage_e/<session>/ (041 v3 §0, 1.7: no existing session is
    changed)."""
    out = Path(out).resolve()
    if out.parent.parent != Path(root).joinpath(*RESULTS).resolve():
        return [f"{out} is not results/stage_e/<session>/<file>"]
    return [f"{out.parent} already exists (a new session directory only)"] if out.parent.exists() else []


# ---------------------------------------------------------------- per-bench quantities (pure)

def episodes(ttfts_ms: list) -> list:
    return [{**e, "k": e["start_idx"] // C} for e in ca.classify_bench(ttfts_ms, C=C, L=L)["episodes"]]


def baseline_groups(n: int, slip_ks: set) -> list:
    last = n // C - 1
    return [g for g in BASE_GROUPS if g < last and g not in slip_ks and g + 1 not in slip_ks]


def _position_median(values: list, groups: list, j: int, rel: bool) -> float:
    return statistics.median((values[C * g + j] - values[C * g]) if rel else values[C * g + j] for g in groups)


def l3_dev(lat: list, k: int, base: list) -> float | None:
    """min over j 1..7 of the k - 1 group's within-group latency offset minus that position's baseline (None when a
    latency it needs is missing)."""
    if any(lat[C * g + j] is None for g in [k - 1, *base] for j in range(C)):
        return None
    return min((lat[C * (k - 1) + j] - lat[C * (k - 1)]) - _position_median(lat, base, j, True) for j in range(1, C))


def delta_q(q: list, k: int, base: list) -> float:
    return statistics.mean(q[C * k + j] - _position_median(q, base, j, False) for j in range(1, C))


def gap_ms(times_s: list, k: int) -> float:
    return (times_s[C * k + 1] - times_s[C * k]) * 1000


def n_dec(step_t: list, step_ctx: list, arrival: float) -> int | None:
    n = 0
    for t, ctx in zip(step_t, step_ctx):
        if t <= arrival:
            continue
        if ctx > 0:
            return n
        n += 1
    return None


def send_lags_by_idx(starts_s: list, completions_s: list) -> dict:
    """{sent idx: lag ms}: the (C + m)-th send in time order is released by the m-th completion in time order (the
    closed loop's semaphore), whatever their indices."""
    sends = sorted(range(len(starts_s)), key=lambda i: starts_s[i])
    done = sorted(completions_s)
    return {sends[C + m]: (starts_s[sends[C + m]] - done[m]) * 1000 for m in range(len(sends) - C)}


def bench_episodes(ttfts_ms: list, *, lat_ms: list | None, times_s: list | None, cfg: dict, F_ms: list | None = None,
                   Q_ms: list | None = None, steps: list | None = None, arrivals_s: list | None = None,
                   send_lags: dict | None = None) -> dict:
    """The links of every slip of one bench (None where the data cannot evaluate a link)."""
    e = cfg["e1"]
    eps = episodes(ttfts_ms)
    ks = {x["k"] for x in eps}
    base = baseline_groups(len(ttfts_ms), ks)
    ndec = None
    if steps is not None and arrivals_s is not None:
        st, sc = [s["t"] for s in steps], [s["ctx_tok"] for s in steps]
        ndec = {g: n_dec(st, sc, arrivals_s[C * g]) for g in range(1, len(ttfts_ms) // C)}
    med_ndec = statistics.median(v for g, v in (ndec or {}).items() if g in BASE_GROUPS and v is not None) \
        if ndec else None
    out = []
    for x in eps:
        k = x["k"]
        continuous = k - 1 in ks
        links = {"L1": None, "L2": None, "L3": None, "L4": None, "L5": None}
        values = {}
        if F_ms is not None:
            values["F0_ms"] = F_ms[C * k]
            links["L1"] = F_ms[C * k] > e["f_spike_ms"]
        if ndec is not None and ndec.get(k) is not None:
            values["n_dec"], values["n_dec_median"] = ndec[k], med_ndec
            links["L2"] = ndec[k] == med_ndec + 1
        dev = l3_dev(lat_ms, k, base) if lat_ms is not None and not continuous and k >= 1 else None
        if dev is not None:
            values["L3_dev_ms"] = dev
            links["L3"] = dev < -e["slip_ms"]
        if times_s is not None and C * k + 1 < len(times_s):
            values["gap_ms"] = gap_ms(times_s, k)
            links["L4"] = values["gap_ms"] < e["gap_ms"]
        if Q_ms is not None:
            values["dQ_ms"] = delta_q(Q_ms, k, base)
            links["L5"] = values["dQ_ms"] > e["slip_ms"]
        if send_lags is not None:
            lags = [send_lags.get(C * k + j) for j in range(1, C)]
            values["send_lag_ms"] = lags
            values["send_lag_ms_max"] = max((v for v in lags if v is not None), default=None)
        out.append({**x, "continuous": continuous, "in_base_range": k in BASE_GROUPS, "links": links,
                    "values": values})
    l1_false = None
    if F_ms is not None:
        normal = [g for g in BASE_GROUPS if g not in ks and C * g < len(F_ms)]
        l1_false = {"groups": len(normal), "spikes": sum(F_ms[C * g] > e["f_spike_ms"] for g in normal)}
    return {"episodes": out, "l1_in_normal_groups": l1_false, "n_dec_median": med_ndec}


def chain_summary(benches: list, cfg: dict) -> dict:
    eps = [x for b in benches for x in b["episodes"] if not x["continuous"] and x["in_base_range"]]
    evaluable = [x for x in eps if any(v is not None for v in x["links"].values())]
    failing = [x for x in evaluable if any(v is False for v in x["links"].values())]
    l1 = [b["l1_in_normal_groups"] for b in benches if b["l1_in_normal_groups"]]
    rate = sum(x["spikes"] for x in l1) / sum(x["groups"] for x in l1) if l1 else None
    holds = len(failing) <= cfg["e1"]["chain_tolerance"] and (rate is None or rate <= cfg["e1"]["l1_false_max"])
    starts = {}
    for b in benches:
        for x in b["episodes"]:
            starts[x["start_idx"]] = starts.get(x["start_idx"], 0) + 1
    return {"episodes": len(eps),
            "continuous": sum(x["continuous"] for b in benches for x in b["episodes"]),
            "outside_base_range": sum(not x["in_base_range"] for b in benches for x in b["episodes"]),
            "evaluable": len(evaluable), "with_a_failing_link": len(failing), "l1_rate_in_normal_groups": rate,
            "chain": "holds" if evaluable and holds else ("incomplete" if evaluable else "not evaluable"),
            "start_idx": dict(sorted(starts.items()))}


def admission_s(row: dict) -> float:
    """8g's own prefill start on the server clock: arrival + (server_ttft - P) (041 v3 §1.4)."""
    return row["arrival"] + (row["server_ttft_ms"] - row["P_ms"]) / 1000


def n_dec_anchored(steps: list, row: dict) -> dict:
    """The step count n stamped in (arrival, s + ANCHOR_TAU_MS] and its decode-only part (T / TG chunks can make some
    of them mixed). The distance from each window edge to the nearest stamp on either side shows how close the count
    is to gaining or losing one step; below the stamps' resolution (1 ms) the count is ambiguous."""
    a, s = row["arrival"], admission_s(row)
    end = s + ANCHOR_TAU_MS / 1000
    t = [x["t"] for x in steps]
    inside = [x for x in steps if a < x["t"] <= end]
    before_a, after_a = [v for v in t if v <= a], [v for v in t if v > a]
    before_e, after_e = [v for v in t if v <= end], [v for v in t if v > end]
    nearest = min(t, key=lambda v: abs(v - s)) if t else None
    edges = {"arrival_minus_stamp_ms": (a - before_a[-1]) * 1000 if before_a else None,
             "stamp_minus_arrival_ms": (after_a[0] - a) * 1000 if after_a else None,
             "end_minus_stamp_ms": (end - before_e[-1]) * 1000 if before_e else None,
             "stamp_minus_end_ms": (after_e[0] - end) * 1000 if after_e else None}
    known = [v for v in edges.values() if v is not None]
    margin = min(known) if known else None
    return {"n": len(inside), "decode_only": sum(x["ctx_tok"] == 0 for x in inside), "edges": edges,
            "nearest_stamp_minus_s_ms": (nearest - s) * 1000 if nearest is not None else None,
            "margin_ms": margin, "ambiguous": margin is None or margin < STAMP_RESOLUTION_MS}


def f_spikes(rows: list, steps: list | None, cfg: dict) -> dict:
    """Every advanced run (041 v3 §1.4; the only description for T / TG, whose TTFT structure differs and which get no
    slip classification): per g 1..24 with F(8g) > f_spike_ms, F and (with steps) the anchored decode-only step count
    against the run's median over g 1..23; the rate of spikes among g 1..23."""
    f = cfg["e1"]["f_spike_ms"]
    groups = range(1, len(rows) // C)
    spikes = [g for g in groups if rows[C * g]["F_ms"] > f]
    in_base = [g for g in BASE_GROUPS if g in groups]
    out = {"spike_groups": spikes, "F_ms": {g: rows[C * g]["F_ms"] for g in spikes},
           "rate_g1_23": sum(g in spikes for g in in_base) / len(in_base), "n_dec_anchor": None}
    if steps is not None:
        st = sorted(steps, key=lambda x: x["t"])
        nd = {g: n_dec_anchored(st, rows[C * g]) for g in groups}
        med = statistics.median(nd[g]["n"] for g in in_base)
        plus1 = [g for g in spikes if nd[g]["n"] == med + 1]
        near = [nd[g]["nearest_stamp_minus_s_ms"] for g in in_base if nd[g]["nearest_stamp_minus_s_ms"] is not None]
        out["n_dec_anchor"] = {"median_g1_23": med, "spikes": {g: nd[g] for g in spikes},
                               "n_counts_g1_23": dict(sorted(collections.Counter(nd[g]["n"] for g in in_base).items())),
                               "nearest_stamp_minus_s_ms_range_g1_23": [min(near), max(near)] if near else None,
                               "spikes_at_median_plus_1": len(plus1),
                               "spikes_at_median_plus_1_unambiguous": sum(not nd[g]["ambiguous"] for g in plus1),
                               "ambiguous_groups_g1_23": sum(nd[g]["ambiguous"] for g in in_base)}
    return out


# ---------------------------------------------------------------- inputs (manifest-checked)

def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest(root: Path, session: Path) -> dict:
    return json.loads((root / session / "raw_manifest.json").read_text())["files"]


def raw_input(root: Path, session: Path, name: str, used: dict, *, required: bool):
    """The local copy COPIES/<session id>/<name> checked against <session>/raw_manifest.json; None when it is absent
    and not required (the dependent quantities are then blocked)."""
    path = COPIES / session.name / name
    m = _manifest(root, session)
    key = f"raw/{name}"
    if key not in m:
        raise StageE1Error(f"{key} is not in {session}/raw_manifest.json")
    if (root / path).with_name(name + ".sha256-mismatch").exists():                 # fetch_raw.sh's exit 5
        raise StageE1Error(f"{path} failed its raw_manifest.json sha256 when it was fetched")
    if not (root / path).is_file():
        if required:
            raise StageE1Error(f"{path} is missing")
        return None
    got = file_sha256(root / path)
    if got != m[key]["sha256"]:
        raise StageE1Error(f"{path} does not match {session}/raw_manifest.json")
    used[str(path)] = got
    return root / path


def _jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def committed(root: Path, path: Path, used: dict) -> Path:
    used[str(path)] = file_sha256(root / path)
    return root / path


def advanced_run(root: Path, tag: str, cfg: dict, used: dict) -> dict:
    cond, r = tag.split("-")[0], int(tag[-1])
    rows = sorted(_jsonl(committed(root, ADV / "derived" / f"{tag}-requests.jsonl", used)), key=lambda x: x["idx"])
    blocked = []
    steps_path = raw_input(root, ADV, f"{tag}.steps.jsonl", used, required=False)
    steps = _jsonl(steps_path) if steps_path else None
    if steps is None:
        blocked.append("steps (the anchored n_dec)" if cond in AUX_CONDS else "steps (L2 and the anchored n_dec)")
    spikes = f_spikes(rows, steps, cfg)
    if cond in AUX_CONDS:
        return {"run": tag, "slip_condition": False, "f_spikes": spikes, "blocked": blocked}
    arrivals = [x["arrival"] for x in rows]
    spans_path = raw_input(root, ADV, f"{tag}.spans.jsonl", used, required=False)
    completion, source = None, "proxy"
    if spans_path:
        by = {rec["id"]: rec for rec in ad.llm_request_records(_jsonl(spans_path))}
        ids = [ad.request_id(ap.request_id_prefix(cond, L, "uniform", r), i) for i in range(len(rows))]
        if all(i in by and by[i].get("e2e") is not None for i in ids):
            completion, source = [by[i]["arrival"] + by[i]["e2e"] for i in ids], "e2e"
        else:
            blocked.append("spans lack an e2e (L3 uses the proxy completion)")
    else:
        blocked.append("spans (L3 uses the proxy completion)")
    if source == "proxy":
        completion = [arrivals[i + C] if i + C < len(arrivals) else None for i in range(len(rows))]
    lat = [(c - a) * 1000 if c is not None else None for c, a in zip(completion, arrivals)]
    bench_path = raw_input(root, ADV, f"{tag}.bench.json", used, required=False)
    lags = None
    if bench_path:
        d = json.loads(bench_path.read_text())
        st, la = d["start_times"], d["latencies"]
        lags = send_lags_by_idx(st, [s + v for s, v in zip(st, la)])      # the client clock only
    else:
        blocked.append("bench.json (client send lag)")
    out = bench_episodes([x["client_ttft_ms"] for x in rows], lat_ms=lat, times_s=arrivals, cfg=cfg,
                         F_ms=[x["F_ms"] for x in rows], Q_ms=[x["Q_ms"] for x in rows], steps=steps,
                         arrivals_s=arrivals, send_lags=lags)
    return {"run": tag, "slip_condition": True, "completion_source": source, "f_spikes": spikes, "blocked": blocked,
            **out}


def c0_bench(root: Path, name: str, cfg: dict, used: dict) -> dict:
    rows = sorted(_jsonl(committed(root, C0 / "derived" / f"{name}-requests.jsonl", used)), key=lambda x: x["idx"])
    lat = [x["latency_ms"] for x in rows]
    path = raw_input(root, C0, f"{name}.bench.json", used, required=False)
    starts, lags, blocked = None, None, []
    if path:
        starts = json.loads(path.read_text())["start_times"]
        lags = send_lags_by_idx(starts, [s + v / 1000 for s, v in zip(starts, lat)])
    else:
        blocked.append("bench.json (L4 and send lag)")
    out = bench_episodes([x["ttft_ms"] for x in rows], lat_ms=lat, times_s=starts, cfg=cfg, send_lags=lags)
    if lags:
        out["send_lag_ms_max"] = max(lags.values())
    return {"run": name, "blocked": blocked, **out}


def core_run(root: Path, r: int, cfg: dict, used: dict) -> dict:
    name = f"bench-uniform-8192-r{r}-c8.json"
    d = json.loads(raw_input(root, CORE, name, used, required=True).read_text())
    ttft, lat, starts = [x * 1000 for x in d["ttfts"]], [x * 1000 for x in d["latencies"]], d["start_times"]
    lags = send_lags_by_idx(starts, [s + v / 1000 for s, v in zip(starts, lat)])
    out = bench_episodes(ttft, lat_ms=lat, times_s=starts, cfg=cfg, send_lags=lags)
    out["send_lag_ms_max"] = max(lags.values())
    return {"run": f"r{r}", "blocked": [], **out}


def analyze(root: Path, cfg: dict, *, config_sha256: str) -> dict:
    problems = config_problems(cfg)
    if problems:
        raise StageE1Error("configs/stage_e1.toml: " + "; ".join(problems))
    if cfg["freeze"]["e1"] is not True:
        raise StageE1Error("E1-D reads no real input before freeze.e1 = true (041 v3 §1.7)")
    if harness_sha256() != cfg["freeze"]["e1_harness_sha256"]:
        raise StageE1Error("the harness differs from freeze.e1_harness_sha256")
    used = {}
    adv = [advanced_run(root, t, cfg, used) for t in ADV_TAGS]
    c0 = [c0_bench(root, b, cfg, used) for b in C0_BENCHES]
    core = [core_run(root, r, cfg, used) for r in (1, 2, 3)]
    slips = [x for x in adv if x["slip_condition"]]

    def rate(runs):
        return statistics.mean(x["f_spikes"]["rate_g1_23"] for x in runs)

    return {"descriptive_only": True, "decision": "041 v3 §1", "harness_sha256": harness_sha256(),
            "stage_e1_sha256": config_sha256, "inputs_sha256": dict(sorted(used.items())),
            "datasets": {
                "advanced": {"runs": slips, "summary": chain_summary(slips, cfg),
                             "auxiliary_T_TG": [x for x in adv if not x["slip_condition"]],
                             "f_spike_rate_g1_23": {"B0_G_K3": rate(slips),
                                                    "T_TG": rate([x for x in adv if not x["slip_condition"]])},
                             "blocked": sorted({b for x in adv for b in x["blocked"]})},
                "c0": {"runs": c0, "summary": chain_summary(c0, cfg),
                       "blocked": sorted({b for x in c0 for b in x["blocked"]})},
                "core": {"runs": core, "summary": chain_summary(core, cfg), "blocked": []}}}


def gate_problems(cfg: dict) -> list:
    """fetch-list and analyze both need a valid frozen config whose harness pin matches (041 v3 §1.7)."""
    problems = config_problems(cfg)
    if not problems and cfg["freeze"]["e1"] is not True:
        problems.append("freeze.e1 must be true first (the module and its tests are committed before any fetch)")
    if not problems and harness_sha256() != cfg["freeze"]["e1_harness_sha256"]:
        problems.append("the harness differs from freeze.e1_harness_sha256")
    return problems


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-list")
    p = sub.add_parser("analyze")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cfg_raw = CONFIG.read_bytes()
    cfg = tomllib.loads(cfg_raw.decode())
    problems = gate_problems(cfg)
    if problems:
        print("stage_e1: " + "; ".join(problems), file=sys.stderr)
        return 2
    if args.cmd == "fetch-list":
        bad = [f"{s}/{n}" for _, s, names in FETCH for n in names
               if (ROOT / COPIES / s / f"{n}.sha256-mismatch").exists()]
        if bad:
            print(f"stage_e1: fetched files failed their sha256, the analysis stops (041 v3 §1.5): {bad}",
                  file=sys.stderr)
            return 2
        print("stage_e1: fetch only while tools/rtx/gpu_status.sh prints FREE (041 v3 §1.7: never during a GPU "
              "session; the fetch takes no lock)", file=sys.stderr)
        for sub_dir, session, names in FETCH:                 # one fetch_raw.sh argument line per session left
            missing = [n for n in names if not (ROOT / COPIES / session / n).exists()]
            if missing:
                print(" ".join([sub_dir, session, *missing]))
        return 0
    out = Path(args.out)
    problems = out_problems(out)
    if problems:
        print("stage_e1: --out " + "; ".join(problems), file=sys.stderr)
        return 2
    try:
        result = analyze(ROOT, cfg, config_sha256=hashlib.sha256(cfg_raw).hexdigest())
    except StageE1Error as exc:
        print(f"stage_e1: {exc}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v["summary"]["chain"] for k, v in result["datasets"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
