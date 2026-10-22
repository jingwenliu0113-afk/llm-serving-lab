#!/usr/bin/env python3
"""Stage E1-D (decision 041 v3 §1.3-1.5): where an 8K slip starts, described on existing data with the predefined
links L1-L5 of the slip chain. Exploratory and descriptive only (descriptive_only = true); no GPU.

Data (each checked against its own session's raw_manifest.json before it is read):
- advanced: formal-20261011-adv 8K uniform B0, G, K3 (slips) and T, TG (auxiliary): committed derived requests (server
  arrival, Q, P, F, client TTFT); steps.jsonl (B0 / G / K3 from the c0-step0 copies, T / TG fetched); spans.jsonl
  (fetched) for each request's e2e, so completion = arrival + e2e. Without spans, completion is proxied by the closed
  loop: completion(8g + j) ~ arrival(8(g + 1) + j) (labelled "proxy").
- c0: explore-c0-20261016's 24 benches: committed derived TTFT / latency; bench.json (fetched) for start_times.
- core: formal-20261009-coreb-uniform's three C 8 benches (c0-step0 copies): TTFT, latency, start_times.

Per bench: slips by c0_analysis.classify_bench (028); group g = idx 8g..8g+7; slip group k = start idx // 8; baselines
over g 1..23 that are neither slipped nor a slip's precursor (k - 1); a slip whose k - 1 also slipped is "continuous"
(counted once, L3 not evaluated).
- L1 (advanced): F(8k) > f_spike_ms.
- L2 (advanced): n_dec(k) = steps stamped after 8k's arrival before the first step with ctx_tok > 0; holds when it is
  the median over all groups plus 1.
- L3: min_j over j 1..7 of (lat(8(k-1)+j) - lat(8(k-1))) - that position's baseline median < -slip_ms.
- L4: gap(k) = time(8k+1) - time(8k) < gap_ms (server arrival for advanced, client start for c0 / core).
- L5 (advanced): mean over j 1..7 of Q(8k+j) - that position's baseline median > slip_ms.
- send lag (c0 with start_times, core): start of the (C + m)-th send minus the m-th completion, in time order.
A dataset's chain holds when at most chain_tolerance evaluable episodes have a failing link and L1 holds in at most
l1_false_max of the non-slipped groups; the three datasets are reported separately.

Usage:
    python benchmarks/stage_e1.py fetch-list                  # what tools/rtx/fetch_raw.sh must fetch first
    python benchmarks/stage_e1.py analyze --out results/stage_e/e1-<date>/e1.json      # needs freeze.e1 = true
"""

from __future__ import annotations

import argparse
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
HARNESS_FILES = ("benchmarks/stage_e1.py", "benchmarks/c0_analysis.py")      # c0_analysis has no guard test
ADV = Path("results/advanced/formal-20261011-adv")
C0 = Path("results/capacity/explore-c0-20261016")
CORE = Path("results/core_b/formal-20261009-coreb-uniform")
COPIES = Path("results/capacity/c0-step0/raw")
SLIP_CONDS = ("B0", "G", "K3")
AUX_CONDS = ("T", "TG")
ADV_TAGS = tuple(f"{c}-8192-uniform-r{r}" for c in SLIP_CONDS + AUX_CONDS for r in (1, 2, 3))
C0_BENCHES = tuple(f"b{b}-n{n}" for b in range(1, 7) for n in range(1, 5))
# fetched read-only after freeze.e1 (041 v3 §1.7): (results subdir, session, raw file names)
FETCH = (("advanced", "formal-20261011-adv",
          tuple(f"{t}.{x}" for t in ADV_TAGS for x in ("spans.jsonl", "bench.json"))
          + tuple(f"{t}.steps.jsonl" for t in ADV_TAGS if t.split("-")[0] in AUX_CONDS)),
         ("capacity", "explore-c0-20261016", tuple(f"{b}.bench.json" for b in C0_BENCHES)))
CONFIG_KEYS = {"schema_version": None, "freeze": {"e1"},
               "e1": {"f_spike_ms", "slip_ms", "gap_ms", "chain_tolerance", "l1_false_max"}}


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
    if not problems and cfg["e1"]["slip_ms"] != ca.SLIP_MS:
        problems.append(f"e1.slip_ms {cfg['e1']['slip_ms']} is not 028's SLIP_MS {ca.SLIP_MS}")
    return problems


# ---------------------------------------------------------------- per-bench quantities (pure)

def episodes(ttfts_ms: list) -> list:
    return [{**e, "k": e["start_idx"] // C} for e in ca.classify_bench(ttfts_ms, C=C, L=L)["episodes"]]


def baseline_groups(n: int, slip_ks: set) -> list:
    last = n // C - 1
    return [g for g in BASE_GROUPS if g < last and g not in slip_ks and g + 1 not in slip_ks]


def _position_median(values: list, groups: list, j: int, rel: bool) -> float:
    return statistics.median((values[C * g + j] - values[C * g]) if rel else values[C * g + j] for g in groups)


def l3_dev(lat: list, k: int, base: list) -> float:
    """min over j 1..7 of the k - 1 group's within-group latency offset minus that position's baseline."""
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


def send_lags_ms(starts_s: list, completions_s: list) -> list:
    """Closed loop with C slots: the (C + m)-th send (time order) is released by the m-th completion (time order)."""
    s, c = sorted(starts_s), sorted(completions_s)
    return [(s[C + m] - c[m]) * 1000 for m in range(len(s) - C)]


def bench_episodes(ttfts_ms: list, *, lat_ms: list | None, times_s: list | None, cfg: dict, F_ms: list | None = None,
                   Q_ms: list | None = None, steps: list | None = None, arrivals_s: list | None = None) -> dict:
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
        if lat_ms is not None and not continuous and k >= 1:
            values["L3_dev_ms"] = l3_dev(lat_ms, k, base)
            links["L3"] = values["L3_dev_ms"] < -e["slip_ms"]
        if times_s is not None and C * k + 1 < len(times_s):
            values["gap_ms"] = gap_ms(times_s, k)
            links["L4"] = values["gap_ms"] < e["gap_ms"]
        if Q_ms is not None:
            values["dQ_ms"] = delta_q(Q_ms, k, base)
            links["L5"] = values["dQ_ms"] > e["slip_ms"]
        out.append({**x, "continuous": continuous, "in_base_range": k in BASE_GROUPS, "links": links,
                    "values": values})
    l1_false = None
    if F_ms is not None:
        normal = [g for g in BASE_GROUPS if g not in ks and C * g < len(F_ms)]
        l1_false = {"groups": len(normal), "spikes": sum(F_ms[C * g] > e["f_spike_ms"] for g in normal)}
    return {"episodes": out, "l1_in_normal_groups": l1_false, "n_dec_median": med_ndec}


def chain_summary(benches: list, cfg: dict) -> dict:
    eps = [x for b in benches for x in b["episodes"] if not x["continuous"]]
    evaluable = [x for x in eps if any(v is not None for v in x["links"].values())]
    failing = [x for x in evaluable if any(v is False for v in x["links"].values())]
    l1 = [b["l1_in_normal_groups"] for b in benches if b["l1_in_normal_groups"]]
    rate = sum(x["spikes"] for x in l1) / sum(x["groups"] for x in l1) if l1 else None
    holds = len(failing) <= cfg["e1"]["chain_tolerance"] and (rate is None or rate <= cfg["e1"]["l1_false_max"])
    starts = {}
    for b in benches:
        for x in b["episodes"]:
            starts[x["start_idx"]] = starts.get(x["start_idx"], 0) + 1
    return {"episodes": len(eps), "continuous": sum(x["continuous"] for b in benches for x in b["episodes"]),
            "evaluable": len(evaluable), "with_a_failing_link": len(failing), "l1_rate_in_normal_groups": rate,
            "chain": "holds" if evaluable and holds else ("incomplete" if evaluable else "not evaluable"),
            "start_idx": dict(sorted(starts.items()))}


# ---------------------------------------------------------------- inputs (manifest-checked)

def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(root: Path, path: Path, manifest_session: Path, key: str) -> Path:
    """Path of an input under root whose sha256 equals its entry (key) in manifest_session's raw_manifest.json."""
    m = json.loads((root / manifest_session / "raw_manifest.json").read_text())["files"]
    if key not in m:
        raise StageE1Error(f"{key} is not in {manifest_session}/raw_manifest.json")
    if not (root / path).is_file():
        raise StageE1Error(f"{path} is missing (fetch it first: tools/rtx/fetch_raw.sh)")
    if file_sha256(root / path) != m[key]["sha256"]:
        raise StageE1Error(f"{path} does not match {manifest_session}/raw_manifest.json")
    return root / path


def _jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def advanced_run(root: Path, tag: str, cfg: dict, *, fetched: bool) -> dict:
    cond, r = tag.split("-")[0], int(tag[-1])
    rows = sorted(_jsonl(root / ADV / "derived" / f"{tag}-requests.jsonl"), key=lambda x: x["idx"])
    steps_path = (COPIES / "formal-20261011-adv" / f"{tag}.steps.jsonl") if cond in SLIP_CONDS else \
        (ADV / "raw" / f"{tag}.steps.jsonl")
    steps = None
    if cond in SLIP_CONDS or fetched:
        steps = _jsonl(checked(root, steps_path, ADV, f"raw/{tag}.steps.jsonl"))
    arrivals = [x["arrival"] for x in rows]
    completion, source = None, "proxy"
    if fetched:
        recs = ad.llm_request_records(_jsonl(checked(root, ADV / "raw" / f"{tag}.spans.jsonl", ADV,
                                                     f"raw/{tag}.spans.jsonl")))
        prefix = ap.request_id_prefix(cond, L, "uniform", r)
        by = {rec["id"]: rec for rec in recs}
        completion = [by[ad.request_id(prefix, i)]["arrival"] + by[ad.request_id(prefix, i)]["e2e"]
                      if ad.request_id(prefix, i) in by and by[ad.request_id(prefix, i)].get("e2e") is not None
                      else None for i in range(len(rows))]
        source = "e2e" if all(v is not None for v in completion) else "proxy"
    if source == "proxy":
        completion = [arrivals[i + C] if i + C < len(arrivals) else None for i in range(len(rows))]
    lat = [(c - a) * 1000 if c is not None else None for c, a in zip(completion, arrivals)]
    usable_lat = lat if all(v is not None for v in lat[: len(lat) - C]) else None
    out = bench_episodes([x["client_ttft_ms"] for x in rows], lat_ms=usable_lat, times_s=arrivals, cfg=cfg,
                         F_ms=[x["F_ms"] for x in rows], Q_ms=[x["Q_ms"] for x in rows], steps=steps,
                         arrivals_s=arrivals)
    return {"run": tag, "completion_source": source, "slip_condition": cond in SLIP_CONDS, **out}


def c0_bench(root: Path, name: str, cfg: dict, *, fetched: bool) -> dict:
    rows = sorted(_jsonl(root / C0 / "derived" / f"{name}-requests.jsonl"), key=lambda x: x["idx"])
    starts = None
    if fetched:
        d = json.loads(checked(root, C0 / "raw" / f"{name}.bench.json", C0, f"raw/{name}.bench.json").read_text())
        starts = d.get("start_times")
    lat = [x["latency_ms"] for x in rows]
    out = bench_episodes([x["ttft_ms"] for x in rows], lat_ms=lat, times_s=starts, cfg=cfg)
    if starts:
        out["send_lag_ms_max"] = max(send_lags_ms(starts, [s + v / 1000 for s, v in zip(starts, lat)]))
    return {"run": name, **out}


def core_run(root: Path, r: int, cfg: dict) -> dict:
    name = f"bench-uniform-8192-r{r}-c8.json"
    d = json.loads(checked(root, COPIES / "formal-20261009-coreb-uniform" / name, CORE, f"raw/{name}").read_text())
    ttft, lat, starts = [x * 1000 for x in d["ttfts"]], [x * 1000 for x in d["latencies"]], d["start_times"]
    out = bench_episodes(ttft, lat_ms=lat, times_s=starts, cfg=cfg)
    out["send_lag_ms_max"] = max(send_lags_ms(starts, [s + v / 1000 for s, v in zip(starts, lat)]))
    return {"run": f"r{r}", **out}


def analyze(root: Path, cfg: dict, *, fetched: bool) -> dict:
    problems = config_problems(cfg)
    if problems:
        raise StageE1Error("configs/stage_e1.toml: " + "; ".join(problems))
    if cfg["freeze"]["e1"] is not True:
        raise StageE1Error("E1-D reads no real input before freeze.e1 = true (041 v3 §1.7)")
    adv = [advanced_run(root, t, cfg, fetched=fetched) for t in ADV_TAGS]
    c0 = [c0_bench(root, b, cfg, fetched=fetched) for b in C0_BENCHES]
    core = [core_run(root, r, cfg) for r in (1, 2, 3)]
    return {"descriptive_only": True, "decision": "041 v3 §1", "harness_sha256": harness_sha256(), "fetched": fetched,
            "datasets": {
                "advanced": {"runs": adv, "summary": chain_summary([x for x in adv if x["slip_condition"]], cfg)},
                "c0": {"runs": c0, "summary": chain_summary(c0, cfg)},
                "core": {"runs": core, "summary": chain_summary(core, cfg)}}}


def main(argv: list | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-list")
    p = sub.add_parser("analyze")
    p.add_argument("--out", required=True)
    p.add_argument("--no-fetched", action="store_true", help="the WSL raw is unavailable: proxies, links blocked")
    args = parser.parse_args(argv)
    if args.cmd == "fetch-list":
        for sub_dir, session, names in FETCH:
            print(json.dumps({"subdir": sub_dir, "session": session, "files": list(names)}))
        return 0
    cfg_raw = CONFIG.read_bytes()
    try:
        out = analyze(ROOT, tomllib.loads(cfg_raw.decode()), fetched=not args.no_fetched)
    except StageE1Error as exc:
        print(f"stage_e1: {exc}", file=sys.stderr)
        return 2
    out["stage_e1_sha256"] = hashlib.sha256(cfg_raw).hexdigest()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["summary"]["chain"] for k, v in out["datasets"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
