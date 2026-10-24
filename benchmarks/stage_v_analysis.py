#!/usr/bin/env python3
"""Stage V analysis (decision 044; PREREGISTRATION §12), fixed before any Stage V data. The statistics are Stage C's
and Stage D's own frozen functions (capacity_analysis.run_eval, classify, window_bounds, the sentinel summary, gate
and cross-session shift; stage_d_analysis.derived_problems for the fixed-output points and h_verify); this module
only supplies what 039 did not have:

- the access-log join (join_access_log, 044 (h) and (i)), read once per boot after gateway_down (the log reader has
  joined, so the log is complete): on the completion route (POST /v1/completions) exactly one record per
  client_request_id <tag>-<idx> for idx 0..N-1, each status 200 and outcome ok with queue_wait_s, ttft_s,
  upstream_ttft_s, duration_s, t_arrival_s <= t_sent_s <= t_released_s (pacing: prefill_wait_s; weighted-v1:
  admission_class and its admission_units); no other client_request_id; at least C + 1 records without an id (the
  bench's ready check, which may retry, and its C warmups), every one status 200 and outcome ok, all arriving before
  the first main request; /healthz, /readyz and /metrics are never read; any other route is a problem; pacing off:
  no completion record waited for tokens (E);
- the derived requests (gateway_requests): run_stage_d.derived_requests' columns with the gateway's gw_queue_wait_ms,
  gw_prefill_wait_ms, gw_ttft_ms, gw_upstream_ttft_ms, gw_arrival_s / gw_sent_s / gw_released_s (seconds from the
  run's earliest main arrival), units (weighted-v1: the record's admission_units, else 1) and out_planned; a W
  point's cls is long for an 8,064-token row. Their checks (derived_problems): D-C's for a fixed-output point; for an
  L or W point idx 0..N-1 once, out = the dataset's pinned output (044 (k)) and cls / units of the dataset's class
  (044 (l), N2), from stage_v_datasets.spec, which is pure in the seed; every gateway field a number, the times in
  order, every gw ttft_s <= the client's TTFT + join_tolerance_ms (044 (h));
- per run (044 共通的定義), over W = [C, N - C): run_eval_v1 is capacity_analysis.run_eval with Y_TPOT only over out
  >= 2 (044 差異 10; the L points'); the upstream view (upstream_view: ttft := upstream_ttft, latency := upstream_ttft
  + (latency - ttft), so TPOT is unchanged); the W SLO view (slo_view_w: run_eval at L 2048, mix "mixed" with the 8K
  tier on the long class's 32K slot, so 2K is judged per run and 8K pooled over the runs on 6 s, never a skip);
- in vLLM (in_vllm): the half-open intervals [t_sent, t_released), exact (Fraction): over T_W = [t_arrival(idx C),
  t_arrival(idx N - C)) the time averages of units (U_W) and of requests (N1's), and over the whole run the peaks
  (044 (j): requests <= C; weighted-v1 units <= the budget and each class <= its cap); the 1 Hz sampler's peak is
  reported beside them;
- E (manipulation_e, V2): pacing on, at least C - b - 1 first-wave requests (idx < C) waited for tokens, b the most of
  the first wave's shortest prompts that fit in the burst (uniform: the prompt; var: stage_d_datasets.lengths, the
  pinned var-2048's); pacing off, none waited;
- run outcome (run_outcome): invalid (the runner's verdict, the derived checks, (j), E), fail (valid, a judged view's
  maxR_run > 1), trivial (valid, no fail, V4's U_W < 0.9 x the budget or V2's N1 < 0.9 C), ok;
- point class (point_class), FAIL > TRIVIAL > NT > MARGINAL > PASS over the judged views (V-GW, V1 client; V2 the
  worse of upstream and client; V4 upstream): FAIL a fail run or a pooled R_long > 1; TRIVIAL a trivial run; NT fewer
  than n ok runs; PASS n ok runs with every judged term <= 1 - m; else MARGINAL. Descriptive: both views in W' and
  full, WINDOW_SENSITIVE, the first and second waves, the per-request decomposition (transport, gateway, vLLM, queue,
  prefill), in-vLLM averages, V1's output-length bins, V4's 8K count in W (stage_v_datasets.window_counts);
- the replay (replay_problems): blocks r = 1..n in capacity_plan.block_order, one boot and one V run per active point;
  once a run is not ok its point's skip group (stage_v_plan.skip_group: its product unit, the unit is dead) leaves
  the later boots, the same block's included; before every V boot the hour cap on the boot's recorded elapsed_s
  (stage_v_plan.cap_allows), a session's cap_stop {block, point, elapsed_s, forced} where the cap first stops it
  (the mini: forced, before its last planned V boot), every later planned run not run (cap);
- hypotheses (044 H-GW1-4, H-L1-2, H-P1-3, H-W0-2): PASS supported, FAIL and TRIVIAL not supported, MARGINAL
  inconclusive, NT NT; H-W0 every valid W run's U_W >= 0.9 x 32;
- units (the generator's only input, 044 產品對應): a unit holds when the session is valid (replay, sentinel gate,
  formal) and each of its points PASSes, PACE-2K unless vetoed (F-U2K-var PASS and P-U2K-var not); readings: V2's
  two 2 x 2 tables.

Usage (stage_v_report runs the refusals first and writes analysis.json):
    import stage_v_analysis as sva
    analysis = sva.analyze({"plan": ..., "session": ..., "rows": [...]}, requests)

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

import capacity_analysis as ca
import capacity_plan as cp
import core_b_analysis as ba
import core_b_plan as bp
import stage_d_analysis as sda
import stage_d_datasets as sdd
import stage_d_plan as sd
import stage_v_datasets as svd
import stage_v_plan as sv

SCHEMA_VERSION = 1
STAGE = sv.STAGE
WINDOWS = ca.WINDOWS
VIEWS = sv.VIEWS
COMPLETION_PATH = "/v1/completions"
EXEMPT_PATHS = ("/healthz", "/readyz", "/metrics")      # capserve.guard's (never imported): never joined
LOG_FIELDS = ("queue_wait_s", "ttft_s", "upstream_ttft_s", "duration_s", "t_arrival_s", "t_sent_s", "t_released_s")
CLIENT_KEYS = ("idx", "cls", "ttft_ms", "latency_ms", "out", "start_s")   # run_stage_d.derived_requests
GATEWAY_KEYS = ("out_planned", "units", "gw_queue_wait_ms", "gw_prefill_wait_ms", "gw_ttft_ms", "gw_upstream_ttft_ms",
                "gw_arrival_s", "gw_sent_s", "gw_released_s")
DERIVED_KEYS = CLIENT_KEYS + GATEWAY_KEYS
TIMES = ("gw_arrival_s", "gw_sent_s", "gw_released_s")   # the log's t_arrival_s, t_sent_s, t_released_s
OUT_BINS = ((1, 16), (17, 64), (65, 128), (129, 256), (257, 512))
FORCED_TIMEOUT_S = 5                                     # run_capacity.MINI_FORCED_TIMEOUT_S: the mini's forced timeout
HYPOTHESES = {"H-GW1": ("GW-U32K", "supported"), "H-GW2": ("GW-U2K-O512", "supported"),
              "H-GW3": ("GW-U8K-O512", "supported"), "H-GW4": ("GW-M8K-T", "not supported"),
              "H-L1": ("L-U2K", "supported"), "H-L2": ("L-U8K", "supported"), "H-P1": ("P-U2K", "supported"),
              "H-P2": ("P-U8K", "supported"), "H-P3": ("P-U2K-var", "not supported"),
              "H-W1": ("W-90", "not supported"), "H-W2": ("W-50", "not supported")}
H_W0 = ("W-90", "W-50")                                  # every valid run's U_W >= 0.9 x 32; predicted to hold
W_EFFECTIVE = {"v4-90": 24.6, "v4-50": 12.8}             # 043 7.3 A's effective concurrency (requests; descriptive)
VAR_PAIR = ("P-U2K-var", "F-U2K-var")
UNIFORM_PAIR = ("P-U2K", "F-U2K")
NOT_SUPPORTED = (
    "Stage V measures one machine, one model and the frozen request path of one gateway version, closed loop, "
    "/v1/completions with ignore_eos; V-GW covers C_PROD only and outputs of 128 or 512; V1 five pinned sequences of "
    "outputs uniform on [1, 512]; V2 the slow queue settings (120 s / 600 s), no first-wave claim, no fast 429, only "
    "D-C's var sequence; V4 mixes 0.9 and 0.5 of v-uniform-2k / v-uniform-8k with pacing off, judged from admission "
    "(upstream), not from arrival; none covers prompt lengths that vary within a class, open loop, 32K or T under a "
    "policy, or pacing and weighted-v1 together.",
)


class _Stop(Exception):
    """A structural mismatch: what follows cannot be replayed."""


def _finite(v) -> bool:
    return ca._num(v) and math.isfinite(v)


# ---------------------------------------------------------------- what a run plans (044 datasets)

def planned(plan: dict, point: str, block: int) -> list:
    """[(prompt tokens or None, output tokens)] per idx of one run: an L or W point's from its block's dataset
    (stage_v_datasets.spec, pure in the seed), any other point's from its client base (None: a var or mixed prompt)."""
    x = plan["points"][point]
    if x["out_pinned"] or x["admission"]:
        rows = svd.spec(x["datasets"][str(block)])
        if len(rows) != x["N"]:
            raise ca.AnalysisError(f"{point} block {block}: dataset {x['datasets'][str(block)]} has {len(rows)} rows, "
                                   f"N is {x['N']}")
        return rows
    base = sd.workload(x["client_base"])
    return [(base["prompt"] if base["mix"] == "uniform" else None, base["out"])] * x["N"]


def _class_of(x: dict) -> dict:
    """weighted-v1: the dataset prompt length -> its class ({name, units, max_concurrency}), in admission order."""
    return dict(zip((svd.PROMPT_2K, svd.PROMPT_8K), x["admission_params"]["classes"]))


def _units_of(x: dict, prompt) -> int:
    return _class_of(x)[prompt]["units"] if x["admission"] else 1


# ---------------------------------------------------------------- the access log (044 (h), (i))

def _entry_problems(e: dict, line: int, x: dict, main: bool) -> list:
    problems = []
    if e.get("status") != 200 or e.get("outcome") != "ok":
        problems.append(f"access-log line {line}: status {e.get('status')!r} outcome {e.get('outcome')!r} (044 (i): no "
                        "4xx, 429 or 5xx, main requests and warmups alike)")
    if not x["pacing"] and _finite(e.get("prefill_wait_s")) and e["prefill_wait_s"] > 0:
        problems.append(f"access-log line {line}: prefill_wait_s {e['prefill_wait_s']} with pacing off (044 (m) E)")
    if not main:
        if not _finite(e.get("t_arrival_s")):
            problems.append(f"access-log line {line}: t_arrival_s missing or not a number (044 (h))")
        return problems
    need = LOG_FIELDS + (("prefill_wait_s",) if x["pacing"] else ())
    missing = [k for k in need if not _finite(e.get(k))]
    if missing:
        return problems + [f"access-log line {line} lacks {missing} as numbers (044 (h))"]
    if not e["t_arrival_s"] <= e["t_sent_s"]:
        problems.append(f"access-log line {line}: t_sent_s {e['t_sent_s']} before t_arrival_s {e['t_arrival_s']}")
    if not e["t_sent_s"] <= e["t_released_s"]:
        problems.append(f"access-log line {line}: t_released_s {e['t_released_s']} before t_sent_s {e['t_sent_s']}")
    if any(e[k] < 0 for k in ("queue_wait_s", "prefill_wait_s") if k in need):
        problems.append(f"access-log line {line}: a negative wait")
    if x["admission"]:
        classes = {c["name"]: c["units"] for c in x["admission_params"]["classes"]}
        cls, units = e.get("admission_class"), e.get("admission_units")
        if cls not in classes:
            problems.append(f"access-log line {line}: admission_class {cls!r} is not one of {sorted(classes)} "
                            "(044 (h))")
        elif type(units) is not int or units != classes[cls]:
            problems.append(f"access-log line {line}: admission_units {units!r} != {cls}'s {classes[cls]} (044 (h))")
    return problems


def join_access_log(text: str, *, plan: dict, point: str, block: int) -> tuple:
    """(problems, {idx: access-log record}) of one run's gateway log (044 (h), (i)); [] problems when the log and the
    run's N requests are a bijection under the module docstring's rules. Lines that are not JSON objects (uvicorn,
    capserve's own prints, tracebacks) are not records; a line starting with "{" must be one."""
    x = plan["points"][point]
    n, C = x["N"], x["C"]
    prefix = sv.run_tag("V", point, C, block=block) + "-"
    problems, joined, lines, anon = [], {}, {}, []
    for k, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            e = None
        if not (isinstance(e, dict) and isinstance(e.get("path"), str) and "request_id" in e):
            problems.append(f"unreadable access-log line {k} (044 (h))")
            continue
        if e["path"] in EXEMPT_PATHS:
            continue
        if e["path"] != COMPLETION_PATH or e.get("route") != "completions" or e.get("method") != "POST":
            problems.append(f"access-log line {k}: {e.get('method')} {e['path']} is not the completion route of this "
                            "run (044 (h))")
            continue
        cid = e.get("client_request_id")
        if cid is None:
            problems += _entry_problems(e, k, x, main=False)
            anon.append((k, e))
            continue
        suffix = cid[len(prefix):] if isinstance(cid, str) and cid.startswith(prefix) else ""
        if not (suffix.isascii() and suffix.isdigit() and str(int(suffix)) == suffix and int(suffix) < n):
            problems.append(f"access-log line {k}: client_request_id {cid!r} is not {prefix}<idx> with idx < {n} "
                            "(044 (h): no other prefix)")
            continue
        idx = int(suffix)
        if idx in joined:
            problems.append(f"access-log lines {lines[idx]} and {k}: idx {idx} logged twice (044 (h))")
            continue
        problems += _entry_problems(e, k, x, main=True)
        joined[idx], lines[idx] = e, k
    missing = sorted(set(range(n)) - set(joined))
    if missing:
        problems.append(f"{len(missing)} of {n} requests have no access-log record: idx {missing} (044 (h))")
    want = bp.num_warmups(C) + 1
    if len(anon) < want:
        problems.append(f"{len(anon)} completion records without an id; the ready check and the C warmups make at "
                        f"least C + 1 = {want} (044 (h))")
    t0 = min((e["t_arrival_s"] for e in joined.values() if _finite(e.get("t_arrival_s"))), default=None)
    late = [k for k, e in anon if t0 is not None and _finite(e.get("t_arrival_s")) and e["t_arrival_s"] >= t0]
    if late:
        problems.append(f"access-log lines {late}: completion requests without the run's id arrived during the main "
                        "run (044 (h): only the ready check and the warmups come without one, before it)")
    return problems, joined


def _ms(v):
    return v * 1000 if _finite(v) else None


def gateway_requests(reqs: list, joined: dict, *, plan: dict, point: str, block: int) -> list:
    """derived/<tag>-requests.jsonl rows of a Stage V run (044 逐 request 的擷取): the client's rows
    (run_stage_d.derived_requests) with the access-log record of the same idx, as the module docstring lists them; a
    field the record lacks is None (derived_problems refuses it)."""
    x = plan["points"][point]
    want = planned(plan, point, block)
    t0 = min((e["t_arrival_s"] for e in joined.values() if _finite(e.get("t_arrival_s"))), default=None)
    out = []
    for q in reqs:
        i, e = q.get("idx"), joined.get(q.get("idx")) or {}
        row = {k: q.get(k) for k in CLIENT_KEYS}
        plan_i = want[i] if isinstance(i, int) and 0 <= i < len(want) else (None, None)
        if x["admission"]:
            row["cls"] = "long" if plan_i[0] == svd.PROMPT_8K else "short"
        row.update(out_planned=plan_i[1], units=e.get("admission_units") if x["admission"] else 1,
                   gw_queue_wait_ms=_ms(e.get("queue_wait_s")), gw_prefill_wait_ms=_ms(e.get("prefill_wait_s")),
                   gw_ttft_ms=_ms(e.get("ttft_s")), gw_upstream_ttft_ms=_ms(e.get("upstream_ttft_s")))
        for k in TIMES:
            v = e.get("t" + k[2:])
            row[k] = round(v - t0, 6) if _finite(v) and t0 is not None else None
        out.append(row)
    return out


# ---------------------------------------------------------------- derived checks (044 (e), (h), (k), (l))

def _pinned_problems(reqs, *, x: dict, want: list) -> list:
    """An L or W point's derived requests against its dataset: idx, numbers, out (k), out_planned and cls (l). The
    dataset itself puts short (and W's 8K) requests in W (stage_v_datasets.window_counts, pinned in the tests)."""
    if reqs is None:
        return ["no derived requests file"]
    if not all(isinstance(q, dict) and isinstance(q.get("idx"), int) for q in reqs):
        return ["derived requests without an integer idx"]
    n = len(want)
    if sorted(q["idx"] for q in reqs) != list(range(n)):
        return [f"idx is not 0..{n - 1} exactly once ({len(reqs)} requests)"]
    if not all(ca._num(q.get(k)) for q in reqs for k in ("ttft_ms", "latency_ms", "out")):
        return ["ttft_ms / latency_ms / out missing or not numbers"]
    problems = []
    bad = [q["idx"] for q in reqs if q["out"] != want[q["idx"]][1]]
    if bad:
        problems.append(f"(k) out is not the dataset's pinned output_tokens at idx {bad} (044)")
    bad = [q["idx"] for q in reqs if q.get("out_planned") != want[q["idx"]][1]]
    if bad:
        problems.append(f"out_planned is not the dataset's output_tokens at idx {bad}")
    cls = {q["idx"]: q.get("cls") for q in reqs}
    want_cls = {i: "long" if x["admission"] and p == svd.PROMPT_8K else "short" for i, (p, _) in enumerate(want)}
    bad = [i for i in range(n) if cls[i] != want_cls[i]]
    if bad:
        problems.append(f"cls is not the dataset's class at idx {bad} (044 (l): 8,064 tokens are long)")
    return problems


def _gateway_problems(reqs: list, *, x: dict, want: list, tolerance) -> list:
    nums = ("gw_queue_wait_ms", "gw_ttft_ms", "gw_upstream_ttft_ms", *TIMES) + (("gw_prefill_wait_ms",)
                                                                                if x["pacing"] else ())
    missing = sorted({k for q in reqs for k in nums if not _finite(q.get(k))})
    if missing:
        return [f"gateway fields {missing} missing or not numbers (044 (h))"]
    problems = []
    if any(q[k] < 0 for q in reqs for k in ("gw_queue_wait_ms", "gw_prefill_wait_ms") if k in nums):
        problems.append("a negative gateway wait")
    bad = [q["idx"] for q in reqs if not q["gw_arrival_s"] <= q["gw_sent_s"] <= q["gw_released_s"]]
    if bad:
        problems.append(f"gw_arrival_s <= gw_sent_s <= gw_released_s fails at idx {bad}")
    if min(q["gw_arrival_s"] for q in reqs) != 0:
        problems.append("gw_arrival_s is not counted from the run's earliest main arrival")
    bad = [q["idx"] for q in reqs if Fraction(q["gw_ttft_ms"]) > Fraction(q["ttft_ms"]) + tolerance]
    if bad:
        problems.append(f"the gateway's ttft is more than {tolerance} ms above the client's at idx {bad} (044 (h))")
    bad = [q["idx"] for q in reqs if q.get("units") != _units_of(x, want[q["idx"]][0])]
    if bad:
        problems.append(f"(l) units are not the dataset's class at idx {bad} (044 N2)")
    if not (x["out_pinned"] or x["admission"]):
        bad = [q["idx"] for q in reqs if q.get("out_planned") != want[q["idx"]][1]]
        if bad:
            problems.append(f"out_planned is not {want[0][1]} at idx {bad}")
    return problems


def derived_problems(reqs, *, plan: dict, point: str, block: int) -> list:
    """One run's derived requests (module docstring); [] when they are complete and consistent with its plan."""
    x = plan["points"][point]
    want = planned(plan, point, block)
    if x["out_pinned"] or x["admission"]:
        problems = _pinned_problems(reqs, x=x, want=want)
    else:
        problems = sda.derived_problems(reqs, x=sd.workload(x["client_base"]), c=x["C"], n=x["N"])
    if problems:
        return problems
    return _gateway_problems(reqs, x=x, want=want, tolerance=plan["join_tolerance_ms"])


# ---------------------------------------------------------------- views and run_eval_v1 (044 共通的定義)

def upstream_view(reqs: list) -> list:
    """044 upstream view: ttft := upstream_ttft (from the gateway's send to vLLM, C_PROD's definition), latency :=
    upstream_ttft + (latency - ttft), so every TPOT is the client's."""
    return [{**q, "ttft_ms": q["gw_upstream_ttft_ms"],
             "latency_ms": q["gw_upstream_ttft_ms"] + (q["latency_ms"] - q["ttft_ms"])} for q in reqs]


def slo_view_w(slo: dict) -> dict:
    """044 V4: run_eval at L 2048 and mix "mixed" judges 2K per run on the 2048 tier and the long class, 8K, pooled
    over the runs on the 32K slot, which this view sets to the 8192 tier (S2: 6 s); TPOT for every request."""
    t = slo["ttft_p95_ms"]
    return {**slo, "ttft_p95_ms": {**t, str(cp.LONG_L): t["8192"]}, "mixed_long": "budget"}


def run_eval_v1(reqs: list, *, L: int, mix: str, C: int, N: int, slo: dict, guard_band: float,
                window: str = cp.WINDOW) -> dict:
    """capacity_analysis.run_eval with one difference (044 差異 10): Y_TPOT is the P95 over the requests in W with out
    >= 2 (030's (latency - ttft) / (out - 1) has no value at out 1; vLLM 0.30's bench computes TPOT only for output
    lengths above 1); TTFT counts every request, and a one-token request's joint attainment reads its TTFT alone."""
    lo, hi = ca.window_bounds(window, C, N)
    W = [q for q in reqs if lo <= q["idx"] < hi]
    thr, band = ca.thresholds(slo, L, mix), ca._band(guard_band)
    short = [q["ttft_ms"] for q in W if q["cls"] == "short"]
    long_ = [q["ttft_ms"] for q in W if q["cls"] == "long"]
    tpot = [ca._tpot_ms(q) for q in W if q["out"] >= 2]
    Y = {"short": ca._p95(short), "tpot": ca._p95(tpot), "long": ca._p95(long_) if mix == "mixed" else None}
    R, over, over_band = {}, {}, {}
    for k in ca.TERMS:
        judged = thr[k] is not None and Y[k] is not None
        R[k] = Y[k] / thr[k] if judged else None
        over[k] = ca._over(Y[k], thr[k], Fraction(1)) if judged else None
        over_band[k] = ca._over(Y[k], thr[k], band) if judged else None
    fail = bool(over["short"] or over["tpot"])

    def met(q):
        tier = thr["long"] if q["cls"] == "long" else thr["short"]
        return (tier is None or q["ttft_ms"] <= tier) and (q["out"] < 2 or ca._tpot_ms(q) <= thr["tpot"])

    burst = [q["ttft_ms"] for q in reqs if q["idx"] < C]
    rs = [v for v in (R["short"], R["tpot"]) if v is not None]
    return {"window": window, "Y": Y, "R": R, "over": over, "over_band": over_band, "max_r": max(rs) if rs else None,
            "fail": fail, "search_pass": not fail and not over["long"],
            "within_band": not (over_band["short"] or over_band["tpot"]),
            "attainment": sum(map(met, W)) / len(W) if W else None, "burst_max_ttft_ms": max(burst) if burst else None,
            "n": {"short": len(short), "long": len(long_), "all": len(W)},
            "_short_ms": short, "_tpot_ms": tpot, "_long_ms": long_}


def judging(x: dict) -> tuple:
    """(L, mix) run_eval judges a point with: its client base's S2 tier; "mixed" for 007's 8K mix and for weighted-v1
    (its 8K requests are the long class, slo_view_w)."""
    base = sd.workload(x["client_base"])
    return base["tier"], "mixed" if base["mix"] == "mixed" or x["admission"] else "uniform"


def _slo_for(x: dict, slo: dict) -> dict:
    return slo_view_w(slo) if x["admission"] else slo


def _eval(reqs: list, plan: dict, x: dict, *, view: str, slo: dict, window: str = cp.WINDOW) -> dict:
    L, mix = judging(x)
    fn = run_eval_v1 if x["out_pinned"] else ca.run_eval
    return fn(upstream_view(reqs) if view == "upstream" else reqs, L=L, mix=mix, C=x["C"], N=x["N"],
              slo=_slo_for(x, slo), guard_band=plan["guard_band"], window=window)


# ---------------------------------------------------------------- in vLLM (044 (j), U_W, N1) and E

def in_vllm(reqs: list, *, C: int, N: int) -> dict:
    """Requests and units in vLLM on [gw_sent_s, gw_released_s): exact time averages over T_W ("_u_w", "_n_w"; None
    when T_W is empty) and peaks over the whole run (releases before sends at one instant: half-open intervals)."""
    by = {q["idx"]: q for q in reqs}
    a, b = Fraction(by[C]["gw_arrival_s"]), Fraction(by[N - C]["gw_arrival_s"])
    spans = [(Fraction(q["gw_sent_s"]), Fraction(q["gw_released_s"]), q["units"]) for q in reqs]
    events = sorted([(s, 1, u) for s, _, u in spans] + [(r, 0, u) for _, r, u in spans])
    now_n = now_u = peak_n = peak_u = 0
    per, peak_per = {}, {}
    for _, kind, u in events:
        d = 1 if kind else -1
        now_n, now_u, per[u] = now_n + d, now_u + d * u, per.get(u, 0) + d
        if kind:
            peak_n, peak_u, peak_per[u] = max(peak_n, now_n), max(peak_u, now_u), max(peak_per.get(u, 0), per[u])
    u_w = n_w = None
    if b > a:
        overlap = [(max(min(r, b) - max(s, a), 0), u) for s, r, u in spans]
        u_w = sum(t * u for t, u in overlap) / (b - a)
        n_w = sum(t for t, _ in overlap) / (b - a)
    return {"t_w_s": [float(a), float(b)], "u_w": None if u_w is None else float(u_w),
            "n_w": None if n_w is None else float(n_w), "peak_requests": peak_n, "peak_units": peak_u,
            "peak_by_units": {str(u): k for u, k in sorted(peak_per.items()) if k > 0}, "_u_w": u_w, "_n_w": n_w}


def inflight_problems(iv: dict, *, x: dict) -> list:
    """044 (j) at every instant of the run: requests <= C; weighted-v1 units <= budget and each class <= its cap."""
    problems = []
    if iv["peak_requests"] > x["C"]:
        problems.append(f"{iv['peak_requests']} requests in vLLM at once > C {x['C']} (044 (j))")
    if x["admission"]:
        p = x["admission_params"]
        if iv["peak_units"] > p["budget"]:
            problems.append(f"{iv['peak_units']} units in vLLM at once > the budget {p['budget']} (044 (j))")
        for c in p["classes"]:
            k = iv["peak_by_units"].get(str(c["units"]), 0)
            if k > c["max_concurrency"]:
                problems.append(f"{k} {c['name']} requests in vLLM at once > its cap {c['max_concurrency']} (044 (j))")
    return problems


def nontrivial(iv: dict, *, x: dict, plan: dict) -> dict | None:
    """044 non-trivial condition: V4 U_W >= u_w x the budget (28.8), V2 N1 >= n1 x C (28.8 / 7.2), exactly (an empty
    T_W shows neither: trivial); None for V-GW and V1."""
    if x["admission"]:
        name, key, value, base = "U_W", "u_w", iv["_u_w"], x["admission_params"]["budget"]
    elif x["study"] == "V2":
        name, key, value, base = "N1", "n1", iv["_n_w"], x["C"]
    else:
        return None
    threshold = Fraction(str(plan["nontrivial"][key])) * base
    return {"condition": name, "value": None if value is None else float(value), "threshold": float(threshold),
            "ok": value is not None and value >= threshold}


def burst_fit(lengths: list, burst: int) -> int:
    """The most prompts of lengths whose sum fits in burst, the shortest first (any arrival order)."""
    total = k = 0
    for n in sorted(lengths):
        if total + n > burst:
            break
        total, k = total + n, k + 1
    return k


def manipulation_e(reqs: list, *, x: dict) -> dict | None:
    """044 V2 E: pacing on, >= C - b - 1 first-wave requests waited for tokens; pacing off, none did (None: no V2)."""
    if x["study"] != "V2":
        return None
    if not x["pacing"]:
        waited = sum(1 for q in reqs if (q.get("gw_prefill_wait_ms") or 0) > 0)
        return {"pacing": False, "waited": waited, "b": None, "threshold": None, "ok": waited == 0}
    C, base = x["C"], sd.workload(x["client_base"])
    first = [base["prompt"]] * C if base["mix"] == "uniform" else sdd.lengths(base["tier"], C)
    b = burst_fit(first, x["pacing_params"]["burst_tokens"])
    waited = sum(1 for q in reqs if q["idx"] < C and q["gw_prefill_wait_ms"] > 0)
    return {"pacing": True, "waited": waited, "b": b, "threshold": C - b - 1, "ok": waited >= C - b - 1}


# ---------------------------------------------------------------- run outcome (044)

def run_valid(row: dict, reqs, plan: dict) -> tuple:
    """(valid, reasons) of a V run: the runner's verdict, then the derived checks, (j) and E (an invalid run is never
    evidence)."""
    if row.get("valid") is not True:
        return False, ["runner: " + "; ".join(row.get("reasons") or ["not valid"])]
    x = plan["points"][row["workload"]]
    problems = derived_problems(reqs, plan=plan, point=row["workload"], block=row["block"])
    if not problems:
        problems = inflight_problems(in_vllm(reqs, C=x["C"], N=x["N"]), x=x)
        e = manipulation_e(reqs, x=x)
        if e is not None and not e["ok"]:
            problems.append(f"E: {e['waited']} requests waited for tokens; pacing {'on' if e['pacing'] else 'off'} "
                            f"needs {'>= ' + str(e['threshold']) + ' in the first wave' if e['pacing'] else 'none'} "
                            "(044 V2 manipulation check)")
    return not problems, problems


def evaluate(row: dict, reqs, plan: dict, *, view: str, slo: dict | None = None, window: str = cp.WINDOW) -> tuple:
    """(valid, reasons, eval or None) of one V run in one view under slo (default: the plan's)."""
    ok, reasons = run_valid(row, reqs, plan)
    if not ok:
        return False, reasons, None
    x = plan["points"][row["workload"]]
    return True, [], _eval(reqs, plan, x, view=view, slo=slo or plan["slo"], window=window)


def _outcome(fail: bool, nt: dict | None) -> str:
    return "fail" if fail else "trivial" if nt is not None and not nt["ok"] else "ok"


def run_outcome(row: dict, reqs, plan: dict) -> str:
    """044 run outcome for the skip rule: "invalid", "fail" (a judged view's maxR_run > 1), "trivial" or "ok"."""
    ok, _ = run_valid(row, reqs, plan)
    if not ok:
        return "invalid"
    x = plan["points"][row["workload"]]
    fail = any(_eval(reqs, plan, x, view=v, slo=plan["slo"])["fail"] for v in x["judged_views"])
    return _outcome(fail, nontrivial(in_vllm(reqs, C=x["C"], N=x["N"]), x=x, plan=plan))


def point_class(judged: list, outcomes: list, *, n: int) -> str:
    """044 點的類別, the first that holds of FAIL > TRIVIAL > NT > MARGINAL > PASS: judged = capacity_analysis.classify
    of each judged view over the valid runs (FAIL: a fail run or the pooled R_long > 1), outcomes = every run's."""
    if "fail" in outcomes or any(s["class"] == "FAIL" for s in judged):
        return "FAIL"
    if "trivial" in outcomes:
        return "TRIVIAL"
    if outcomes.count("ok") < n or any(s["class"] == "NT" for s in judged):
        return "NT"
    return "PASS" if all(s["class"] == "PASS" for s in judged) else "MARGINAL"


# ---------------------------------------------------------------- descriptive (044 只作描述)

def _parts(reqs: list, *, C: int, N: int) -> dict:
    W = [q for q in reqs if C <= q["idx"] < N - C]
    return {"transport_ms": [q["ttft_ms"] - q["gw_ttft_ms"] for q in W],
            "gateway_ms": [q["gw_ttft_ms"] - q["gw_upstream_ttft_ms"] - q["gw_queue_wait_ms"]
                           - (q["gw_prefill_wait_ms"] or 0.0) for q in W],
            "vllm_ms": [q["gw_upstream_ttft_ms"] for q in W], "queue_wait_ms": [q["gw_queue_wait_ms"] for q in W],
            "prefill_wait_ms": [q["gw_prefill_wait_ms"] or 0.0 for q in W]}


def _p50_p95(values: list) -> dict:
    return {"p50": ba.percentile(values, 50), "p95": ba.percentile(values, 95)} if values else {"p50": None,
                                                                                                "p95": None}


def decompose(reqs: list, *, C: int, N: int) -> dict:
    """044's per-request split of the client TTFT in W (median and P95): transport (client - gateway ttft), the
    gateway's own body / parse / tokenize (gw ttft - upstream - queue wait - prefill wait), vLLM (upstream ttft), and
    the queue and prefill waits."""
    return {k: _p50_p95(v) for k, v in _parts(reqs, C=C, N=N).items()}


def waves(reqs: list, *, C: int, thresholds: dict) -> dict:
    """The first wave's (idx < C) largest TTFT; the second wave's (idx in [C, 2C)) largest and how many exceed their
    class's TTFT threshold (042: at 2K, C 32 it is 12.5 % of W)."""
    first = [q["ttft_ms"] for q in reqs if q["idx"] < C]
    second = [q for q in reqs if C <= q["idx"] < 2 * C]

    def over(q):
        t = thresholds["long"] if q["cls"] == "long" and thresholds["long"] is not None else thresholds["short"]
        return q["ttft_ms"] > t

    return {"first_max_ms": max(first) if first else None,
            "second_max_ms": max(q["ttft_ms"] for q in second) if second else None,
            "second_over": sum(map(over, second))}


def out_bins(runs: list, *, C: int, N: int, thresholds: dict) -> list:
    """V1 (descriptive): the requests in W of the valid runs by output length, with their TTFT / TPOT P95 (TPOT over
    out >= 2) and the share above each threshold."""
    out = []
    for lo, hi in OUT_BINS:
        qs = [q for reqs in runs for q in reqs if C <= q["idx"] < N - C and lo <= q["out"] <= hi]
        tp = [ca._tpot_ms(q) for q in qs if q["out"] >= 2]
        ttft = [q["ttft_ms"] for q in qs]
        out.append({"bin": [lo, hi], "n": len(qs), "ttft_p95_ms": ca._p95(ttft), "tpot_p95_ms": ca._p95(tp),
                    "ttft_over": sum(t > thresholds["short"] for t in ttft) / len(qs) if qs else None,
                    "tpot_over": sum(t > thresholds["tpot"] for t in tp) / len(tp) if tp else None})
    return out


# ---------------------------------------------------------------- the replay (044 replay, 停止規則與時數上限)

def v_order(plan: dict) -> list:
    """The V boots before skips and caps: [(block, point), ...], blocks r = 1..n in the plan's block orders."""
    return [(r, p) for r in range(1, plan["n"] + 1) for p in plan["block_orders"][str(r)]]


def _row_plan_problems(plan: dict, row: dict) -> list:
    tag, phase, w = row.get("tag"), row.get("phase"), row.get("workload")
    s = plan["sentinel"]
    try:
        if phase == "sentinel" and w == s["workload"]:
            want_tag = sv.run_tag(phase, w, s["C"], position=row.get("position"))
            want = {"C": s["C"], "N": s["N"], "timeout_s": s["timeout_s"]}
        elif phase == "V" and w in plan["points"]:
            x = plan["points"][w]
            want_tag = sv.run_tag(phase, w, x["C"], block=row.get("block"))
            forced = row.get("block") in x["forced_timeout_blocks"]
            want = {"C": x["C"], "N": x["N"], "timeout_s": FORCED_TIMEOUT_S if forced else x["timeout_s"]}
        else:
            return [f"row {tag}: phase {phase!r} / point {w!r} outside the plan"]
    except sv.PlanError as e:
        return [f"row {tag}: {e}"]
    problems = [] if tag == want_tag else [f"row {tag}: tag != {want_tag!r}"]
    problems += [f"row {tag}: {k} {row.get(k)!r} != planned {v!r}" for k, v in want.items() if row.get(k) != v]
    return problems


def _schedule(plan: dict, boots: list, rows: list, requests: dict, cap_stop) -> tuple:
    """(problems, {"skipped", "not_run_cap"}): replay_problems' work and the planned runs it found skipped (a dead
    unit) or not run (the hour cap)."""
    sched = {"skipped": [], "not_run_cap": []}
    for r in range(1, plan["n"] + 1):
        want = cp.block_order(plan["seed"], r, list(plan["points"]))
        if plan["block_orders"].get(str(r)) != want:
            return [f"block_orders {r} {plan['block_orders'].get(str(r))} != capacity_plan.block_order {want}"], sched
    if [b.get("boot") for b in boots] != list(range(len(boots))):
        return ["boots are not numbered 0 .. n-1 in session order"], sched
    for b, before in zip(boots, [None] + boots):
        t = b.get("elapsed_s")
        if not (_finite(t) and t >= 0 and (before is None or t >= before["elapsed_s"])):
            return [f"boot {b['boot']}: elapsed_s {t!r} is not a time >= 0 at or after the boot before it (044 時數上限: "
                    "the cap is recomputed from the recorded times)"], sched
    problems, by_boot, tags = [], {}, set()
    for i, row in enumerate(rows):
        b = row.get("boot")
        if not (isinstance(b, int) and 0 <= b < len(boots)):
            problems.append(f"row {i} ({row.get('tag')}) names no recorded boot")
            continue
        if row.get("tag") in tags:
            problems.append(f"duplicate run tag {row.get('tag')}")
        tags.add(row.get("tag"))
        by_boot.setdefault(b, []).append(row)
        problems += _row_plan_problems(plan, row)
    if problems:
        return problems, sched
    cursor = [0]

    def take(phase, w, *, block=None, position=None) -> dict:
        want = {"phase": phase, "workload": w, "block": block, "position": position}
        if cursor[0] >= len(boots):
            problems.append(f"session ends before boot {want} (044 order)")
            raise _Stop
        b = boots[cursor[0]]
        if {k: b.get(k) for k in want} != want:
            problems.append(f"boot {cursor[0]} is {({k: b.get(k) for k in want})}; the 044 order needs {want}")
            raise _Stop
        cursor[0] += 1
        return b

    def no_valid_on_failed_boot(b: dict, rows_b: list) -> None:
        if b.get("reached_first_bench") is not True and any(r.get("valid") is True for r in rows_b):
            problems.append(f"boot {b['boot']} failed before its first bench but holds a valid run")
            raise _Stop

    def sentinel(position: str) -> None:
        s = plan["sentinel"]
        b = take("sentinel", s["workload"], position=position)
        no_valid_on_failed_boot(b, by_boot.get(b["boot"], []))
        got = [(r.get("phase"), r.get("workload"), r.get("C"), r.get("position")) for r in by_boot.get(b["boot"], [])]
        if got != [("sentinel", s["workload"], s["C"], position)]:
            problems.append(f"{position} sentinel boot holds {got}, not one {s['workload']} C {s['C']} run")
            raise _Stop

    try:
        sentinel("start")
        order, dropped, stopped = v_order(plan), set(), False
        for k, (r, p) in enumerate(order):
            if p in dropped:
                sched["skipped"].append([r, p])
                continue
            x = plan["points"][p]
            last = not any(q not in dropped for _, q in order[k + 1:])
            here = isinstance(cap_stop, dict) and (cap_stop.get("block"), cap_stop.get("point")) == (r, p)
            t = cap_stop.get("elapsed_s") if here else (boots[cursor[0]].get("elapsed_s") if cursor[0] < len(boots)
                                                        else None)
            if _finite(t):
                allowed = sv.cap_allows(t, x["est_s"], plan["hour_cap_h"])
                must_stop = not allowed or (bool(plan["mini"]) and last)
            else:
                allowed, must_stop = True, False
            if here:
                forced = cap_stop.get("forced") is True
                prev, nxt = boots[cursor[0] - 1].get("elapsed_s"), (boots[cursor[0]].get("elapsed_s")
                                                                     if cursor[0] < len(boots) else None)
                if not (_finite(t) and t >= prev and (nxt is None or t <= nxt)):
                    problems.append(f"cap stop elapsed_s {t!r} is not between its neighbouring boots' {prev} and {nxt}")
                elif not must_stop:
                    problems.append(f"a cap stop (forced {forced}) before block {r} {p} at {t} s although the hour cap "
                                    f"allowed it (elapsed + {x['est_s']} s <= {plan['hour_cap_h']} h less the end "
                                    "reserve)")
                elif forced != allowed:
                    why = ("the hour cap allowed it, so only the mini's forced stop before its last planned V boot "
                           "stops here" if allowed else "the hour cap itself stops it (044)")
                    problems.append(f"cap stop before block {r} {p}: forced {forced}, but {why}")
                sched["not_run_cap"] = [[rr, q] for rr, q in order[k:] if q not in dropped]
                stopped = True
                break
            if must_stop:
                why = "the hour cap stops it" if not allowed else "the mini's forced stop comes before its last V boot"
                problems.append(f"block {r} {p} must not start at {t} s: {why}; the session holds no cap stop there "
                                "(044 時數上限)")
                raise _Stop
            b = take("V", p, block=r)
            rows_b = by_boot.get(b["boot"], [])
            no_valid_on_failed_boot(b, rows_b)
            got = [(row.get("phase"), row.get("workload"), row.get("block"), row.get("C")) for row in rows_b]
            if got != [("V", p, r, x["C"])]:
                problems.append(f"block {r} {p} ran {got}; 044 runs one V run at C {x['C']}")
                raise _Stop
            outcome = run_outcome(rows_b[0], requests.get(rows_b[0]["tag"]), plan)
            if rows_b[0].get("outcome") != outcome:
                problems.append(f"V {rows_b[0]['tag']}: recorded outcome {rows_b[0].get('outcome')!r}, recomputed "
                                f"{outcome!r}")
            if outcome != "ok":
                dropped |= set(x["skip_group"])
        if cap_stop is not None and not stopped:
            problems.append(f"the recorded cap stop {cap_stop!r} is not where this replay stops (044 時數上限)")
        sentinel("end")
        if cursor[0] != len(boots):
            problems.append(f"{len(boots) - cursor[0]} boot(s) after the end sentinel")
    except _Stop:
        pass
    return problems, sched


def replay_problems(plan: dict, boots: list, rows: list, requests: dict, *, cap_stop=None) -> list:
    """Every boot, outcome, skip (trivial runs and dead units included) and hour-cap decision recomputed from the rows
    and the recorded times; [] when the session ran exactly what 044 prescribes for its own data."""
    return _schedule(plan, boots, rows, requests, cap_stop)[0]


# ---------------------------------------------------------------- per point

def _public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _sampler(row: dict) -> dict:
    """The 1 Hz sampler's values of the whole bench process (descriptive; (j) is recomputed from the access log)."""
    m = row.get("metrics") or {}
    return {**ca.inflight(row), **{k: m.get(k) for k in ("kv_cache_usage_peak", "preemptions", "waiting_peak")}}


def _run_view(e: dict) -> dict:
    s = e["steady"]
    return {"Y": s["Y"], "R": s["R"], "max_r": s["max_r"], "fail": s["fail"], "attainment": s["attainment"],
            "burst_max_ttft_ms": s["burst_max_ttft_ms"], "n": s["n"], "Y_late": e["late"]["Y"],
            "Y_full": e["full"]["Y"]}


def point_analysis(plan: dict, point: str, rows: list, requests: dict, *, slo: dict,
                   schedule: dict | None = None) -> dict:
    """One point's runs, classes in both views and windows, its 044 class under slo and the descriptive values."""
    x = plan["points"][point]
    L, mix = judging(x)
    slo_x = _slo_for(x, slo)
    thr = ca.thresholds(slo_x, L, mix)
    evals = {v: {w: [] for w in WINDOWS} for v in VIEWS}
    runs, outcomes, parts, valid_reqs, ivs = [], [], [], [], []
    for row in sorted((r for r in rows if r.get("phase") == "V" and r.get("workload") == point),
                      key=lambda r: r.get("block") or 0):
        reqs = requests.get(row.get("tag"))
        ok, reasons = run_valid(row, reqs, plan)
        run = {"tag": row.get("tag"), "block": row.get("block"), "valid": ok, "reasons": reasons, **_sampler(row)}
        if not ok:
            run["outcome"] = "invalid"
            outcomes.append("invalid")
            runs.append(run)
            continue
        e = {v: {w: _eval(reqs, plan, x, view=v, slo=slo, window=w) for w in WINDOWS} for v in VIEWS}
        for v in VIEWS:
            for w in WINDOWS:
                evals[v][w].append(e[v][w])
        iv = in_vllm(reqs, C=x["C"], N=x["N"])
        nt = nontrivial(iv, x=x, plan=plan)
        outcome = _outcome(any(e[v]["steady"]["fail"] for v in x["judged_views"]), nt)
        view_reqs = {"client": reqs, "upstream": upstream_view(reqs)}
        run.update(outcome=outcome, views={v: _run_view(e[v]) for v in VIEWS}, in_vllm=_public(iv), nontrivial=nt,
                   e=manipulation_e(reqs, x=x),
                   waves={v: waves(view_reqs[v], C=x["C"], thresholds=thr) for v in VIEWS},
                   decomposition=decompose(reqs, C=x["C"], N=x["N"]))
        outcomes.append(outcome)
        runs.append(run)
        parts.append(_parts(reqs, C=x["C"], N=x["N"]))
        valid_reqs.append(reqs)
        ivs.append(iv)
    summary = {v: {w: ca.classify(evals[v][w], n=x["n"], L=L, mix=mix, slo=slo_x, guard_band=plan["guard_band"])
                   for w in WINDOWS} for v in VIEWS}
    judged = x["judged_views"]
    views = {v: {"class": summary[v]["steady"]["class"], "class_late": summary[v]["late"]["class"],
                 "class_full": summary[v]["full"]["class"], "worst_R": summary[v]["steady"]["worst_R"],
                 "pooled": summary[v]["steady"]["pooled"], "met": summary[v]["steady"]["met"],
                 "conclusions": summary[v]["steady"]["conclusions"]} for v in VIEWS}
    descriptive = {"decomposition": {k: _p50_p95([y for d in parts for y in d[k]]) for k in
                                     ("transport_ms", "gateway_ms", "vllm_ms", "queue_wait_ms", "prefill_wait_ms")},
                   "in_vllm": {"n_w": [iv["n_w"] for iv in ivs], "u_w": [iv["u_w"] for iv in ivs],
                               "peak_requests": max((iv["peak_requests"] for iv in ivs), default=None),
                               "peak_units": max((iv["peak_units"] for iv in ivs), default=None)}}
    if x["out_pinned"]:
        descriptive["out_bins"] = out_bins(valid_reqs, C=x["C"], N=x["N"], thresholds=thr)
    if x["admission"]:
        eff = W_EFFECTIVE.get(x["dataset"])
        descriptive["window_counts"] = {r: svd.window_counts(key, x["C"]) for r, key in x["datasets"].items()}
        descriptive["requests_vs_effective"] = {
            "effective_concurrency": eff, "n_w_at_least_0_9": None if eff is None else
            [Fraction(str(iv["n_w"])) >= Fraction(str(plan["nontrivial"]["u_w"])) * Fraction(str(eff))
             if iv["n_w"] is not None else None for iv in ivs]}
    sched = schedule or {"skipped": [], "not_run_cap": []}
    return {"study": x["study"], "unit": x["unit"], "C": x["C"], "N": x["N"], "n": x["n"], "classes": x["classes"],
            "view": x["view"], "judged_views": judged,
            "class": point_class([summary[v]["steady"] for v in judged], outcomes, n=x["n"]),
            "n_valid": sum(o != "invalid" for o in outcomes), "n_ok": outcomes.count("ok"), "views": views,
            "window_sensitive": any(views[v]["class_late"] != views[v]["class"] for v in judged),
            "skipped": [r for r, p in sched["skipped"] if p == point],
            "not_run_cap": [r for r, p in sched["not_run_cap"] if p == point], "runs": runs,
            "descriptive": descriptive}


# ---------------------------------------------------------------- hypotheses, units, readings (044)

def verdict(klass: str) -> str:
    """PASS supported; FAIL and TRIVIAL not supported; MARGINAL inconclusive; NT NT."""
    return "not supported" if klass == "TRIVIAL" else sda.h_verify(klass)


def hypotheses(points: dict) -> dict:
    out = {}
    for h, (p, prediction) in HYPOTHESES.items():
        if p in points:
            out[h] = {"point": p, "class": points[p]["class"], "verdict": verdict(points[p]["class"]),
                      "prediction": prediction}
    if all(p in points for p in H_W0):
        valid = {p: [r for r in points[p].get("runs") or [] if r.get("valid")] for p in H_W0}
        if any(not r["nontrivial"]["ok"] for rs in valid.values() for r in rs):
            v = "not supported"
        else:
            v = "supported" if all(valid.values()) else "NT"
        out["H-W0"] = {"points": list(H_W0), "verdict": v, "prediction": "supported"}
    return out


def _worst(point: dict, view: str):
    vals = [v for v in (((point.get("views") or {}).get(view) or {}).get("worst_R") or {}).values() if v is not None]
    return max(vals) if vals else None


def _validation(plan: dict, unit: str, pts: list, points: dict) -> dict:
    xs = [plan["points"][p] for p in pts]
    classes = []
    for x in xs:
        classes += [c for c in x["classes"] if c not in classes]
    out = {"study": xs[0]["study"], "unit": unit, "points": list(pts), "measured_classes": classes,
           "worst_r": {p: {v: _worst(points[p], v) for v in VIEWS} for p in pts}, "n": plan["n"],
           "queue": xs[0]["queue_settings"], "pacing": xs[0]["pacing_params"], "admission": xs[0]["admission_params"]}
    per_block = {p: [x["datasets"][str(r)] for r in range(1, plan["n"] + 1)] for p, x in zip(pts, xs) if x["per_block"]}
    if per_block:
        out["datasets"] = per_block
        out["datasets_sha256"] = {p: [plan["datasets"].get(k) for k in keys] for p, keys in per_block.items()}
    if xs[0]["admission"]:
        out["measured_mixes"] = [svd.MIXES[x["dataset"]] for x in xs]
    return out


def units(plan: dict, points: dict, *, session_ok: bool) -> dict:
    """044 產品對應: a unit holds when the session is valid and each of its points PASSes (PACE-2K: and not vetoed)."""
    out = {}
    for u, pts in plan["units"].items():
        classes = {p: points[p]["class"] for p in pts}
        veto = plan["vetoes"].get(u)
        vetoed = None if veto is None else (points[veto["control"]]["class"] == "PASS"
                                            and points[veto["paced"]]["class"] != "PASS")
        holds = bool(session_ok) and all(k == "PASS" for k in classes.values()) and not vetoed
        out[u] = {"points": list(pts), "classes": classes, "vetoed": vetoed, "holds": holds,
                  "validation": _validation(plan, u, pts, points) if holds else None}
    return out


def readings(points: dict) -> dict:
    """V2's 2 x 2 tables (044): the var pair and P-U2K / F-U2K."""
    out = {}
    if all(p in points for p in VAR_PAIR):
        paced, control = (points[p]["class"] for p in VAR_PAIR)
        reading = ("co_occur" if (paced, control) == ("PASS", "FAIL") else
                   "var_passes_without_pacing" if (paced, control) == ("PASS", "PASS") else
                   "not_fixed" if (paced, control) == ("FAIL", "FAIL") else
                   "veto" if control == "PASS" else "uninterpretable")
        out["var"] = {"paced": paced, "control": control, "reading": reading}
    if all(p in points for p in UNIFORM_PAIR):
        paced, control = (points[p]["class"] for p in UNIFORM_PAIR)
        reading = (("formal_class_holds" if paced == "PASS" else "attributed_to_pacing") if control == "PASS" else
                   "unattributable" if paced == "PASS" else "formal_class_lost")
        out["uniform"] = {"paced": paced, "control": control, "reading": reading}
    return out


# ---------------------------------------------------------------- the analysis

def _sentinel(plan: dict, rows: list, requests: dict, position: str) -> dict:
    s = plan["sentinel"]
    row = next((r for r in rows if r.get("phase") == "sentinel" and r.get("position") == position), None)
    return ca.sentinel_summary(row, requests.get(row.get("tag")) if row else None, C=s["C"], N=s["N"])


def _run_row(plan: dict, row: dict, requests: dict) -> dict:
    reqs = requests.get(row.get("tag"))
    if row.get("phase") == "sentinel" and isinstance(row.get("C"), int):
        ok, reasons = ca.run_valid(row, reqs, mix="uniform", C=row["C"], N=plan["sentinel"]["N"])
    elif row.get("phase") == "V" and row.get("workload") in plan["points"] and isinstance(row.get("block"), int):
        ok, reasons = run_valid(row, reqs, plan)
    else:
        ok, reasons = False, ["not a row of this plan"]
    return {k: row.get(k) for k in ("phase", "workload", "C", "block", "position", "tag", "outcome")} | {
        "valid": ok, "reasons": reasons}


def analyze(cap: dict, requests: dict, *, references: dict | None = None) -> dict:
    """analysis.json body (schema_version 1) of one Stage V session; the report adds provenance and platform."""
    plan, session, rows = cap["plan"], cap["session"], cap["rows"]
    replay, sched = _schedule(plan, session.get("boots") or [], rows, requests, session.get("cap_stop"))
    sent = {pos: _sentinel(plan, rows, requests, pos) for pos in ("start", "end")}
    gate = ca.sentinel_gate(sent["start"], sent["end"], tolerance=plan["sentinel"]["tolerance"])
    points = {p: point_analysis(plan, p, rows, requests, slo=plan["slo"], schedule=sched) for p in plan["points"]}
    secondary = {}
    for sid, slo in plan["secondary_slo"].items():
        res = {p: point_analysis(plan, p, rows, requests, slo=slo) for p in plan["points"]}
        secondary[sid] = {"slo": slo, "descriptive": True,
                          "points": {p: {"class": r["class"], "views": {v: r["views"][v]["class"] for v in VIEWS}}
                                     for p, r in res.items()}}
    session_ok = not replay and gate["ok"] and plan["mini"] is None
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "session": plan["session"], "kind": plan["kind"],
            "slo": plan["slo"],
            "protocol": {"seed": plan["seed"], "n": plan["n"], "window": plan["window"],
                         "guard_band": plan["guard_band"], "nontrivial": plan["nontrivial"],
                         "join_tolerance_ms": plan["join_tolerance_ms"], "hour_cap_h": plan["hour_cap_h"],
                         "end_reserve_h": plan["end_reserve_h"]},
            "replay": {"ok": not replay, "problems": replay},
            "schedule": {**sched, "cap_stop": session.get("cap_stop")},
            "sentinel": {**sent, "gate": gate,
                         "cross_session_shift": ca.cross_session_shift([sent["start"], sent["end"]], references)},
            "points": points, "secondary_slo": secondary, "hypotheses": hypotheses(points),
            "units": units(plan, points, session_ok=session_ok), "readings": readings(points),
            "not_supported": list(NOT_SUPPORTED), "runs": [_run_row(plan, row, requests) for row in rows]}
