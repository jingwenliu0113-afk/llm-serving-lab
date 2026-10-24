"""Tests for benchmarks/stage_v_analysis.py (decision 044): the access-log join and its (h) rules, run_eval_v1 (TPOT
only over out >= 2), the upstream view, the W SLO view and the pooled 8K class, the in-vLLM intervals (U_W, N1 and
(j)), the manipulation check E, run outcomes and point classes with TRIVIAL, the unit table with its veto, the replay
(order, skips, trivial runs, dead units, hour caps from recorded times) and the hypotheses. make_reqs() builds a run's
derived requests from a closed-loop FIFO gateway (closed_loop); simulate() plays a whole session as the runner does,
so every analysis below comes from a replayable session."""

import ast
import copy
import hashlib
import heapq
import json
import tomllib
from collections import deque
from fractions import Fraction
from pathlib import Path

import pytest

import capacity_analysis as ca
import capacity_plan as cp
import core_b_plan as bp
import run_capacity as rc
import run_stage_d as rd
import stage_d_plan as sd
import stage_d_datasets as sdd
import stage_v_analysis as sva
import stage_v_datasets as svd
import stage_v_plan as sv

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_v.toml").read_text())
SD_RAW = (ROOT / "configs" / "stage_d.toml").read_bytes()
SD_CFG = tomllib.loads(SD_RAW.decode())


def plan(session):
    return sv.build_plan(CFG, PINS, SD_CFG, session_name=session, pins_sha256="p" * 64, stage_v_sha256="v" * 64,
                         stage_d_sha256=hashlib.sha256(SD_RAW).hexdigest())


ADM, CLS, MINI = plan("V-ADM"), plan("V-CLS"), plan("mini")


def mini_plan():
    """The mini's session plan as the runner writes it: the dummy SLO (run_capacity.MINI_SLO)."""
    p = copy.deepcopy(MINI)
    p["slo"] = rc.MINI_SLO
    return p


def closed_loop(units, *, C, budget, caps, service):
    """044's gateway in front of vLLM, closed loop: C clients each send their next idx when their last one ends;
    strict FIFO admission (head-of-line blocking) on units within budget and per-units caps; a request stays
    service[units] s in vLLM. -> [(arrival, sent, released)] per idx."""
    n = len(units)
    arrival, sent, released = [None] * n, [None] * n, [None] * n
    queue, heap, used, per, nxt, now = deque(), [], 0, {}, min(C, n), 0.0
    for i in range(nxt):
        arrival[i] = 0.0
        queue.append(i)
    while queue or heap:
        while queue and used + units[queue[0]] <= budget and per.get(units[queue[0]], 0) < caps[units[queue[0]]]:
            i = queue.popleft()
            sent[i] = now
            used += units[i]
            per[units[i]] = per.get(units[i], 0) + 1
            heapq.heappush(heap, (now + service[units[i]], i))
        now, i = heapq.heappop(heap)
        released[i] = now
        used -= units[i]
        per[units[i]] -= 1
        if nxt < n:
            arrival[nxt] = now
            queue.append(nxt)
            nxt += 1
    return list(zip(arrival, sent, released))


def make_reqs(p, point, block=1, *, ttft=800.0, tpot=50.0, up=None, long_ttft=2500.0, long_up=None, service=None,
              prefill=None, slow_idx=(), override=None):
    """One valid run's derived requests (044 逐 request 的擷取): client ttft / latency, the gateway's fields from
    closed_loop, units, out_planned; a W point's 8K rows (and a mixed point's idx 5 mod 10) are the long class with
    long_ttft / long_up. P points: the first wave's idx >= b waited for tokens (E holds); prefill overrides it;
    override {idx: {field: value}} last."""
    x = p["points"][point]
    rows = sva.planned(p, point, block)
    W = bool(x["admission"])
    units = [(4 if pr == svd.PROMPT_8K else 1) if W else 1 for pr, _ in rows]
    budget, caps = (32, {1: 32, 4: 8}) if W else (x["C"], {1: x["C"]})
    times = closed_loop(units, C=x["C"], budget=budget, caps=caps, service=service or {1: 1.0, 4: 1.5})
    if x["pacing"] and prefill is None:
        b = 5 if x["client_base"] == "U2K-gw" else 4 if x["client_base"] == "U8K-gw" else 19
        prefill = {i: 100.0 for i in range(b, x["C"])}
    out = []
    for i, ((prompt, o), (a, s, r)) in enumerate(zip(rows, times)):
        long_ = (W and prompt == svd.PROMPT_8K) or (x["client_base"] == "M8K-T" and i % 10 == 5)
        t = (long_ttft if long_ else ttft) * (3 if i in slow_idx else 1)
        u = (long_up if long_ else up)
        u = t - 100.0 if u is None else u
        out.append({"idx": i, "cls": "long" if long_ else "short", "ttft_ms": float(t),
                    "latency_ms": float(t) + (o - 1) * tpot, "out": o, "start_s": a, "out_planned": o,
                    "units": units[i], "gw_queue_wait_ms": (s - a) * 1000,
                    "gw_prefill_wait_ms": (prefill or {}).get(i, 0.0) if x["pacing"] else None,
                    "gw_ttft_ms": float(t) - 2.0, "gw_upstream_ttft_ms": float(u), "gw_arrival_s": a,
                    "gw_sent_s": s, "gw_released_s": r})
    for i, change in (override or {}).items():
        out[i].update(change)
    return out


WARM_RELEASE = -20.0                    # the C warmups' release, seconds from the run's earliest main arrival


def warmups_of(x, *, release=WARM_RELEASE):
    """[[t_sent_s, t_released_s, units], ...] of a run's id-less completion records as log_text writes them and the
    runner records them (seconds from the run's earliest main arrival): exactly the C warmups of row 0 (2K: one
    unit); the frozen client sends no ready check (--ready-check-timeout-sec 0: 044 實作補記七)."""
    return [[round(-30.0 + k * 0.01, 6), release, 1] for k in range(x["C"])]


def row_for(p, point, block, *, boot=0, valid=True):
    x = p["points"][point]
    timeout = sva.FORCED_TIMEOUT_S if block in x["forced_timeout_blocks"] else x["timeout_s"]
    return {"phase": "V", "workload": point, "C": x["C"], "N": x["N"], "block": block, "position": None, "boot": boot,
            "tag": sv.run_tag("V", point, x["C"], block=block), "valid": valid,
            "reasons": [] if valid else ["simulated invalid run"], "timeout_s": timeout, "output_throughput": 300.0,
            "metrics": {"inflight_peak": x["C"], "kv_cache_usage_peak": 0.5, "preemptions": 0, "waiting_peak": 0},
            "warmups": warmups_of(x)}


def sentinel_reqs(s):
    return [{"idx": i, "cls": "short", "ttft_ms": 800.0, "latency_ms": 800.0 + 127 * 40.0, "out": 128,
             "start_s": float(i)} for i in range(s["N"])]


def simulate(p, kw=lambda point, block: {}, *, invalid=(), slow=None, drop=None):
    """Play p's session as the runner does (044): the start sentinel; blocks r = 1..n in the plan's block orders, one
    boot and one V run per active point; once a run is not ok its point (only it: 實作補記六) leaves the later boots;
    before every V boot the hour cap on the recorded elapsed_s (the mini: a forced stop before its last planned V
    boot); the end sentinel. kw(point, block) -> make_reqs keywords; (point, block) in invalid is an invalid run;
    slow(point, block) -> extra seconds that boot takes; drop(point) -> the points a run that is not ok drops (default:
    the point alone). Returns (cap, requests)."""
    boots, rows, requests, t = [], [], {}, [0.0]
    s = p["sentinel"]

    def boot(phase, w, *, block=None, position=None):
        b = {"boot": len(boots), "phase": phase, "workload": w, "block": block, "attempt": None,
             "position": position, "reached_first_bench": True, "problems": [], "elapsed_s": t[0]}
        boots.append(b)
        return b

    def sentinel(position):
        b = boot("sentinel", s["workload"], position=position)
        row = {"phase": "sentinel", "workload": s["workload"], "C": s["C"], "N": s["N"], "block": None,
               "position": position, "boot": b["boot"], "tag": sv.run_tag("sentinel", s["workload"], s["C"],
                                                                         position=position),
               "valid": True, "reasons": [], "timeout_s": s["timeout_s"], "output_throughput": 300.0, "metrics": {}}
        requests[row["tag"]] = sentinel_reqs(s)
        rows.append(row)
        t[0] += s["est_s"]

    sentinel("start")
    order = [(r, q) for r in range(1, p["n"] + 1) for q in p["block_orders"][str(r)]]
    dropped, cap_stop = set(), None
    for k, (r, q) in enumerate(order):
        if q in dropped:
            continue
        x = p["points"][q]
        last = not [z for _, z in order[k + 1:] if z not in dropped]
        if not sv.cap_allows(t[0], x["est_s"], p["hour_cap_h"]) or (p["mini"] and last):
            cap_stop = {"block": r, "point": q, "elapsed_s": t[0],
                        "forced": bool(p["mini"]) and sv.cap_allows(t[0], x["est_s"], p["hour_cap_h"])}
            break
        b = boot("V", q, block=r)
        row = row_for(p, q, r, boot=b["boot"], valid=(q, r) not in invalid)
        if row["valid"]:
            requests[row["tag"]] = make_reqs(p, q, r, **kw(q, r))
        row["outcome"] = sva.run_outcome(row, requests.get(row["tag"]), p)
        rows.append(row)
        if row["outcome"] != "ok":
            dropped |= set(drop(q) if drop else [q])
        t[0] += x["est_s"] + (slow(q, r) if slow else 0)
    sentinel("end")
    session = {"boots": boots, "cap_stop": cap_stop}
    return {"plan": p, "session": session, "rows": rows}, requests


def ok_kw(point, block):
    """A comfortable PASS everywhere (R ~ 0.4 - 0.5 against S2)."""
    return {"ttft": 800.0 if point != "GW-U32K" else 8000.0, "tpot": 45.0, "long_ttft": 2500.0, "long_up": 2400.0}


# ------------------------------------------------------------------ run_eval_v1, views (044 共通的定義) --

def reqs_for_eval(n, *, out=128, mix="uniform"):
    return [{"idx": i, "cls": "long" if mix == "mixed" and i % 10 == 5 else "short", "ttft_ms": 500.0 + (i * 37) % 900,
             "latency_ms": 500.0 + (i * 37) % 900 + (out - 1) * (40.0 + (i * 13) % 50), "out": out}
            for i in range(n)]


@pytest.mark.parametrize("mix", ["uniform", "mixed"])
@pytest.mark.parametrize("window", ca.WINDOWS)
def test_run_eval_v1_is_run_eval_when_every_out_is_at_least_2(mix, window):
    reqs = reqs_for_eval(320, mix=mix)
    for q in reqs:
        q["out"] = 2 + q["idx"] % 511
    for slo in (cp.SLO_S2, cp.SLO_S1, sva.slo_view_w(cp.SLO_S2)):
        kw = {"L": 2048, "mix": mix, "C": 32, "N": 320, "slo": slo, "guard_band": 0.05, "window": window}
        assert sva.run_eval_v1(reqs, **kw) == ca.run_eval(reqs, **kw)


def test_out_1_counts_for_ttft_and_never_for_tpot():
    reqs = reqs_for_eval(200, out=128)
    for q in reqs[8:192:10]:                                  # 19 requests in W with one output token
        q.update(out=1, ttft_ms=1900.0, latency_ms=1900.0)
    e = sva.run_eval_v1(reqs, L=8192, mix="uniform", C=8, N=200, slo=cp.SLO_S2, guard_band=0.05)
    W = [q for q in reqs if 8 <= q["idx"] < 192]
    assert e["Y"]["short"] == ba_p95([q["ttft_ms"] for q in W])                      # TTFT: every request in W
    assert e["Y"]["tpot"] == ba_p95([(q["latency_ms"] - q["ttft_ms"]) / (q["out"] - 1) for q in W if q["out"] >= 2])
    assert e["n"]["all"] == 184 and len(e["_tpot_ms"]) == 184 - 19
    assert e["attainment"] == 1.0                                                  # out 1: its TTFT alone
    with pytest.raises(ZeroDivisionError):                                         # why the copy: run_eval divides
        ca.run_eval(reqs, L=8192, mix="uniform", C=8, N=200, slo=cp.SLO_S2, guard_band=0.05)


def ba_p95(values):
    import core_b_analysis as ba
    return ba.percentile(values, 95)


def test_the_upstream_view_moves_ttft_and_keeps_tpot():
    reqs = make_reqs(CLS, "L-U2K", 3, ttft=1500.0, up=900.0, tpot=60.0)
    before = copy.deepcopy(reqs)
    up = sva.upstream_view(reqs)
    assert reqs == before                                                          # the derived rows stay as they are
    for q, u in zip(reqs, up):
        assert u["ttft_ms"] == q["gw_upstream_ttft_ms"] == 900.0
        assert u["latency_ms"] - u["ttft_ms"] == pytest.approx(q["latency_ms"] - q["ttft_ms"])
        assert {k: u[k] for k in ("idx", "cls", "out")} == {k: q[k] for k in ("idx", "cls", "out")}
    p = sva.run_eval_v1(up, L=2048, mix="uniform", C=32, N=320, slo=cp.SLO_S2, guard_band=0.05)
    c = sva.run_eval_v1(reqs, L=2048, mix="uniform", C=32, N=320, slo=cp.SLO_S2, guard_band=0.05)
    assert p["Y"]["tpot"] == pytest.approx(c["Y"]["tpot"]) and p["Y"]["short"] == 900.0 and c["Y"]["short"] == 1500.0


def test_the_w_slo_view_is_the_2k_tier_per_run_and_8k_on_6_s_as_the_pooled_long_class():
    assert sva.slo_view_w(cp.SLO_S2)["ttft_p95_ms"] == {"2048": 2000, "8192": 6000, "32768": 6000}
    assert sva.slo_view_w(cp.SLO_S2)["mixed_long"] == "budget" and cp.SLO_S2["ttft_p95_ms"]["32768"] == 20000
    assert sva.slo_view_w(cp.SLO_S1)["ttft_p95_ms"] == {"2048": 1000, "8192": 3000, "32768": 3000}
    assert ca.thresholds(sva.slo_view_w(cp.SLO_S2), 2048, "mixed") == {"short": 2000, "tpot": 100, "long": 6000}
    p = ADM
    reqs = make_reqs(p, "W-90", 1, ttft=900.0, up=800.0, long_ttft=9000.0, long_up=7000.0)
    row = row_for(p, "W-90", 1)
    ok, _, e = sva.evaluate(row, reqs, p, view="upstream")
    assert ok and e["Y"]["short"] == 800.0 and e["Y"]["long"] == 7000.0 and e["R"]["long"] == pytest.approx(7 / 6)
    assert not e["fail"] and sva.run_outcome(row, reqs, p) == "ok"     # one run's 8K above 6 s: pooled, no skip


def test_the_8k_class_is_pooled_over_the_runs_and_judged_once():
    p = ADM
    eights = [i for i, (pr, _) in enumerate(sva.planned(p, "W-90", 2)) if pr == svd.PROMPT_8K and 32 <= i < 288]
    slow = {i: {"gw_upstream_ttft_ms": 7000.0} for i in eights[:3]}          # 3 of run 2's 8K in W at 7 s

    def runs(kw):
        rows, requests = [], {}
        for r in range(1, 6):
            row = row_for(p, "W-90", r, boot=r)
            requests[row["tag"]] = make_reqs(p, "W-90", r, **{"ttft": 900.0, "up": 800.0, "long_up": 4000.0, **kw(r)})
            row["outcome"] = sva.run_outcome(row, requests[row["tag"]], p)
            rows.append(row)
        return rows, requests

    rows, requests = runs(lambda r: {"override": slow} if r == 2 else {})
    ok, _, e = sva.evaluate(rows[1], requests[rows[1]["tag"]], p, view="upstream")
    assert e["R"]["long"] > 1 and not e["fail"]                             # run 2's own 8K P95 is above 6 s ...
    out = sva.point_analysis(p, "W-90", rows, requests, slo=p["slo"])
    assert [r["outcome"] for r in out["runs"]] == ["ok"] * 5                # ... and never skips
    assert out["views"]["upstream"]["pooled"]["Y_long"] == 4000.0 and out["class"] == "PASS"   # pooled: 3 of ~125
    rows, requests = runs(lambda r: {"long_up": 7000.0} if r in (2, 4) else {})
    out = sva.point_analysis(p, "W-90", rows, requests, slo=p["slo"])
    assert out["views"]["upstream"]["pooled"]["R_long"] == pytest.approx(7 / 6) and out["class"] == "FAIL"
    assert [r["outcome"] for r in out["runs"]] == ["ok"] * 5 and out["n_ok"] == 5


# ------------------------------------------------------------------ the access-log join (044 (h), (i)) --

def entry(**kw):
    base = {"ts": 1.0, "request_id": "g" * 32, "method": "POST", "path": "/v1/completions", "route": "completions",
            "stream": True, "status": 200, "outcome": "ok", "queue_wait_s": 0.0, "ttft_s": 0.5,
            "upstream_ttft_s": 0.4, "duration_s": 3.0, "t_arrival_s": 100.0, "t_sent_s": 100.0,
            "t_released_s": 103.0}
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


def random_client(p, point):
    """Whether the point's frozen client is `vllm bench serve --dataset-name random` (read off its argv)."""
    x = p["points"][point]
    argv = sv.bench_argv(p, PINS, point, block=1, result_dir="r", result_filename="f",
                         datasets={k: f"{k}.jsonl" for k in (x["datasets"] or {}).values()})
    return argv[argv.index("--dataset-name") + 1] == "random"


def probe_entry(t_arrival_s, **kw):
    """vLLM 0.30 bench's prompt alignment (serve.py _align_prompts_to_server_tokenizer): one POST /tokenize, no
    x-request-id; the gateway has no /tokenize route, so 404."""
    return entry(**{"path": "/tokenize", "route": None, "stream": None, "status": 404, "outcome": None,
                    "queue_wait_s": None, "ttft_s": None, "upstream_ttft_s": None, "t_sent_s": None,
                    "t_released_s": None, "t_arrival_s": t_arrival_s, **kw})


def log_text(p, point, block, reqs, *, warmups=None, change=None, extra=(), t0=5000.0, probe=None,
             warm_release=WARM_RELEASE):
    """The gateway log of one run: uvicorn lines, a /readyz poll, the bench's POST /tokenize probe (probe None: exactly
    when its client is the random dataset's, as vLLM 0.30's; False: none; True or a dict of changes: one), the C
    warmups without an id (warmups: how many; no ready check, as the frozen client's --ready-check-timeout-sec 0: 044
    實作補記七), GET /metrics around the main run, one completion line per derived request (its gateway fields) and a
    print to stderr; change(entries) edits the main entries. The id-less records' intervals are warmups_of's
    (warm_release: when the C warmups leave vLLM); a weighted-v1 point's are row 0's class, 2K."""
    x = p["points"][point]
    tag = sv.run_tag("V", point, x["C"], block=block)
    if probe is None:
        probe = random_client(p, point)
    pol = {"admission_class": "v-uniform-2k", "admission_units": 1} if x["admission"] else {}
    lines = ["INFO:     Started server process [4242]", "INFO:     Waiting for application startup.",
             json.dumps(entry(method="GET", path="/readyz", route=None, stream=None, t_arrival_s=t0 - 50,
                              t_sent_s=None, t_released_s=None, queue_wait_s=None, ttft_s=None, upstream_ttft_s=None))]
    if probe is not False:
        lines.append(json.dumps(probe_entry(**{"t_arrival_s": t0 - 45, **(probe if isinstance(probe, dict) else {})})))
    for k in range(x["C"] if warmups is None else warmups):
        lines.append(json.dumps(entry(t_arrival_s=t0 - 30, t_sent_s=t0 - 30 + k * 0.01, t_released_s=t0 + warm_release,
                                      **pol)))
    lines.append(json.dumps(entry(method="GET", path="/metrics", route=None, stream=None, status=200,
                                  t_arrival_s=t0 - 1, t_sent_s=None, t_released_s=None, queue_wait_s=None,
                                  ttft_s=None, upstream_ttft_s=None)))
    main = []
    for q in reqs:
        e = entry(client_request_id=f"{tag}-{q['idx']}", queue_wait_s=q["gw_queue_wait_ms"] / 1000,
                  ttft_s=q["gw_ttft_ms"] / 1000, upstream_ttft_s=q["gw_upstream_ttft_ms"] / 1000,
                  t_arrival_s=t0 + q["gw_arrival_s"], t_sent_s=t0 + q["gw_sent_s"],
                  t_released_s=t0 + q["gw_released_s"])
        if q["gw_prefill_wait_ms"] is not None:
            e["prefill_wait_s"] = q["gw_prefill_wait_ms"] / 1000
        if x["admission"]:
            e["admission_class"], e["admission_units"] = ("v-uniform-8k", 4) if q["units"] == 4 else ("v-uniform-2k", 1)
        main.append(e)
    if change:
        change(main)
    lines += [json.dumps(e) for e in main] + [json.dumps(e) for e in extra]
    lines += [json.dumps(entry(method="GET", path="/metrics", route=None, stream=None, t_arrival_s=t0 + 999,
                               t_sent_s=None, t_released_s=None, queue_wait_s=None, ttft_s=None, upstream_ttft_s=None)),
              "capserve: serving PROVISIONAL profile provisional-stage-v (class v-uniform-2k)",
              "INFO:     Shutting down"]
    return "\n".join(lines) + "\n"


def client_part(p, point, block, reqs):
    """The client's rows as the runner makes them: run_stage_d.derived_requests of the bench result (a W point's are
    all short there: its client base is U2K-var)."""
    x = p["points"][point]
    base = sd.workload(x["client_base"])
    lens = [pr if pr is not None else bp.LONG_INPUT if q["cls"] == "long" else 1000
            for (pr, _), q in zip(sva.planned(p, point, block), reqs)]
    res = {"input_lens": lens, "output_lens": [q["out"] for q in reqs],
           "ttfts": [q["ttft_ms"] / 1000 for q in reqs], "latencies": [q["latency_ms"] / 1000 for q in reqs],
           "start_times": [100.0 + q["start_s"] for q in reqs]}
    return rd.derived_requests(res, x=base)


@pytest.mark.parametrize("point", ["L-U2K", "P-U2K", "F-U2K-var", "W-50", "GW-M8K-T", "GW-U8K-O512", "GW-U32K",
                                   "F-U2K"])
def test_a_clean_log_joins_every_request_once_and_rebuilds_the_derived_rows(point):
    p = ADM if point[0] in "PFW" else CLS
    reqs = make_reqs(p, point, 2)
    text = log_text(p, point, 2, reqs)
    assert ('"path": "/tokenize"' in text) is random_client(p, point)        # the random dataset's probe is in it
    problems, joined, warm = sva.join_access_log(text, plan=p, point=point, block=2)
    assert problems == [] and sorted(joined) == list(range(p["points"][point]["N"]))
    t0 = min(e["t_arrival_s"] for e in joined.values())                     # the C warmups, no ready check
    assert sva.warmup_spans(warm, joined, x=p["points"][point]) == warmups_of(p["points"][point]) and t0 == 5000.0
    client = client_part(p, point, 2, reqs)
    assert {q["cls"] for q in client} == ({"short", "long"} if point == "GW-M8K-T" else {"short"})
    got = sva.gateway_requests(client, joined, plan=p, point=point, block=2)
    assert [set(q) for q in got] == [set(sva.DERIVED_KEYS)] * len(reqs)
    for g, q in zip(got, reqs):
        assert g == pytest.approx(q)
    assert sva.derived_problems(got, plan=p, point=point, block=2) == []


def test_the_join_reads_only_its_tag_and_ignores_the_exempt_paths():
    p, point = ADM, "P-U8K"
    reqs = make_reqs(p, point, 4)
    noise = [entry(method="GET", path=path, route=None, status=status, outcome=None, t_arrival_s=5000.5,
                   t_sent_s=None, t_released_s=None) for path in sva.EXEMPT_PATHS for status in (200, 503)]
    problems, joined, _ = sva.join_access_log(log_text(p, point, 4, reqs, extra=noise), plan=p, point=point, block=4)
    assert problems == [] and len(joined) == 200
    assert sva.join_access_log(log_text(p, point, 4, reqs), plan=p, point=point, block=3)[0]      # block 3's tag


WARMUP_COUNT = "completion records without an id; the frozen client sends exactly its C"     # 044 實作補記七


def _joined_problems(p, point, block, reqs, **kw):
    return sva.join_access_log(log_text(p, point, block, reqs, **kw), plan=p, point=point, block=block)[0]


def test_every_join_failure_is_a_problem():
    p, point = CLS, "L-U8K"
    reqs = make_reqs(p, point, 1)
    tag = sv.run_tag("V", point, 8, block=1)
    cases = {
        "missing": (dict(change=lambda es: es.pop(7)), "7"),
        "duplicate": (dict(change=lambda es: es.append(dict(es[3]))), "3"),
        "extra idx": (dict(extra=[entry(client_request_id=f"{tag}-200")]), "200"),
        "not an idx": (dict(extra=[entry(client_request_id=f"{tag}-07")]), "07"),
        "other prefix": (dict(extra=[entry(client_request_id="bench-1a2b3c4d-5")]), "bench-1a2b3c4d-5"),
        "other run": (dict(extra=[entry(client_request_id=sv.run_tag("V", point, 8, block=2) + "-5")]), "r2-5"),
        "429 main": (dict(change=lambda es: es[9].update(status=429, outcome="rejected")), "429"),
        "500 main": (dict(change=lambda es: es[9].update(status=500)), "500"),
        "no status": (dict(change=lambda es: es[9].pop("status")), "status"),
        "not ok": (dict(change=lambda es: es[9].update(outcome="cancelled")), "cancelled"),
        "warmup 4xx": (dict(extra=[entry(status=400, outcome="client_error", t_arrival_s=4970.0)]), "400"),
        "too few without id": (dict(warmups=7), WARMUP_COUNT),                  # C - 1
        "too many without id": (dict(warmups=9), WARMUP_COUNT),                 # C + 1: an unknown extra request
        "id-less during the run": (dict(extra=[entry(t_arrival_s=5003.0)]), "during the main run"),
        "unexpected route": (dict(extra=[entry(path="/v1/chat/completions", route="chat")]), "/v1/chat/completions"),
        "unknown path": (dict(extra=[entry(method="GET", path="/v1/models", route=None)]), "/v1/models"),
        "missing field": (dict(change=lambda es: es[2].pop("t_released_s")), "t_released_s"),
        "upstream ttft": (dict(change=lambda es: es[2].pop("upstream_ttft_s")), "upstream_ttft_s"),
        "not a number": (dict(change=lambda es: es[2].update(queue_wait_s="0.1")), "queue_wait_s"),
        "time order": (dict(change=lambda es: es[2].update(t_sent_s=es[2]["t_arrival_s"] - 1)), "t_sent_s"),
        "release order": (dict(change=lambda es: es[2].update(t_released_s=es[2]["t_sent_s"] - 1)), "t_released_s"),
        "negative wait": (dict(change=lambda es: es[2].update(queue_wait_s=-0.001)), "negative"),
        "GET completions": (dict(extra=[entry(method="GET", status=405, t_arrival_s=4970.0)]), "GET /v1/completions"),
        "warmup without arrival": (dict(extra=[entry(t_arrival_s=None)]), "t_arrival_s"),
        "warmup without send": (dict(extra=[entry(t_arrival_s=4970.0, t_sent_s=None)]), "t_sent_s"),   # (j) reads
        "warmup without release": (dict(extra=[entry(t_arrival_s=4970.0, t_released_s=None)]), "t_released_s"),
        "warmup release order": (dict(extra=[entry(t_arrival_s=4970.0, t_sent_s=4971.0, t_released_s=4970.5)]),
                                 "t_released_s"),
    }
    assert _joined_problems(p, point, 1, reqs) == []
    assert _joined_problems(p, point, 1, reqs, warmups=8) == []                                 # exactly C
    for name, (kw, needle) in cases.items():
        got = _joined_problems(p, point, 1, reqs, **kw)
        assert got and any(needle in g for g in got), (name, got)


POINTS_ALL = {**ADM["points"], **CLS["points"], **MINI["points"]}


def test_the_random_dataset_rule_is_the_frozen_clients():
    """join_access_log expects the probe exactly from the clients whose frozen argv is --dataset-name random."""
    for p in (ADM, CLS, MINI):
        for point, x in p["points"].items():
            assert sva.random_dataset(x) is random_client(p, point), point
    assert {q for q, x in POINTS_ALL.items() if sva.random_dataset(x)} == {
        "GW-U32K", "GW-U2K-O512", "GW-U8K-O512", "P-U2K", "P-U8K", "F-U2K"}



def test_the_random_dataset_benchs_tokenize_probe_is_accepted_once_before_its_first_completion():
    """vLLM 0.30's random-dataset bench sends one POST /tokenize before its warmups; the gateway has no /tokenize (404)
    and the bench skips its alignment. Exactly that record is accepted: never counted as a warmup, never (i)."""
    p, point, t0 = CLS, "GW-U2K-O512", 5000.0
    x = p["points"][point]
    reqs = make_reqs(p, point, 1)
    tag = sv.run_tag("V", point, x["C"], block=1)
    assert _joined_problems(p, point, 1, reqs) == []
    cases = {
        "no probe": (dict(probe=False), "0 POST /tokenize"),
        "two probes": (dict(extra=[probe_entry(t0 - 44)]), "2 POST /tokenize"),
        "status 200": (dict(probe={"status": 200}), "status 200"),
        "with the first warmup": (dict(probe={"t_arrival_s": t0 - 30}), "before every completion"),
        "between the warmups": (dict(probe={"t_arrival_s": t0 - 25}), "before every completion"),
        "during the main run": (dict(probe={"t_arrival_s": t0 + 1}), "before every completion"),
        "no arrival": (dict(probe={"t_arrival_s": None}), "t_arrival_s"),
        "with an id": (dict(probe={"client_request_id": f"{tag}-0"}), "POST /tokenize is not the completion"),
        "GET": (dict(probe={"method": "GET", "status": 405}), "GET /tokenize is not the completion"),
        "detokenize": (dict(extra=[probe_entry(t0 - 44, path="/detokenize")]), "/detokenize"),
        "probe is no warmup": (dict(warmups=x["C"] - 1), WARMUP_COUNT),
    }
    for name, (kw, needle) in cases.items():
        got = _joined_problems(p, point, 1, reqs, **kw)
        assert got and any(needle in g for g in got), (name, got)
    for q, block in (("L-U2K", 1), ("GW-M8K-T", 1), ("W-90", 2), ("P-U2K-var", 1), ("F-U2K-var", 1)):
        pq = ADM if q[0] in "PFW" else CLS
        r = make_reqs(pq, q, block)
        assert _joined_problems(pq, q, block, r) == []
        got = _joined_problems(pq, q, block, r, probe=True)                 # a custom dataset's bench never aligns
        assert got and any("custom-dataset" in g for g in got), (q, got)


def test_an_unreadable_access_line_is_a_problem_and_other_output_is_not():
    p, point = CLS, "GW-U8K-O512"
    reqs = make_reqs(p, point, 1)
    text = log_text(p, point, 1, reqs)
    assert sva.join_access_log(text + "Traceback (most recent call last):\n  ...\n", plan=p, point=point,
                               block=1)[0] == []
    bad = sva.join_access_log(text + '{"ts": 1.0, "path": "/v1/compl\n', plan=p, point=point, block=1)[0]
    assert bad and "unreadable" in bad[0]
    bad = sva.join_access_log(text + '{"ts": 1.0, "path": "/v1/completions"}\n', plan=p, point=point, block=1)[0]
    assert bad and "unreadable" in bad[0]                                     # no request_id: not a guard record


def test_policy_fields_are_required_where_the_policy_runs():
    p = ADM
    reqs = make_reqs(p, "P-U2K", 1)
    assert any("prefill_wait_s" in x for x in _joined_problems(
        p, "P-U2K", 1, reqs, change=lambda es: es[40].pop("prefill_wait_s")))
    w = make_reqs(p, "W-90", 1)
    eight = next(i for i, q in enumerate(w) if q["units"] == 4)
    assert any("admission_units" in x for x in _joined_problems(
        p, "W-90", 1, w, change=lambda es: es[eight].pop("admission_units")))
    assert any("v-uniform-8k" in x for x in _joined_problems(
        p, "W-90", 1, w, change=lambda es: es[eight].update(admission_units=1)))
    assert any("admission_class" in x for x in _joined_problems(
        p, "W-90", 1, w, change=lambda es: es[eight].update(admission_class="uniform-8k")))
    f = make_reqs(p, "F-U2K", 1)                                 # pacing off: no completion record waited for tokens
    assert _joined_problems(p, "F-U2K", 1, f) == []
    assert any("E" in x for x in _joined_problems(p, "F-U2K", 1, f, extra=[entry(prefill_wait_s=0.2,
                                                                                    t_arrival_s=4970.0)]))


def test_the_gateway_ttft_may_exceed_the_clients_by_at_most_50_ms():
    p, point = CLS, "GW-U2K-O512"
    reqs = make_reqs(p, point, 1)
    reqs[100]["gw_ttft_ms"] = reqs[100]["ttft_ms"] + 50
    assert sva.derived_problems(reqs, plan=p, point=point, block=1) == []
    reqs[100]["gw_ttft_ms"] = reqs[100]["ttft_ms"] + 50.001
    assert any("50 ms" in x for x in sva.derived_problems(reqs, plan=p, point=point, block=1))


def test_derived_problems_hold_the_pinned_outputs_classes_and_units():
    p = CLS
    reqs = make_reqs(p, "L-U2K", 2)
    assert sva.derived_problems(reqs, plan=p, point="L-U2K", block=2) == []
    assert sva.derived_problems(reqs, plan=p, point="L-U2K", block=3)                 # block 3's pinned outputs
    bad = copy.deepcopy(reqs)
    bad[50]["out"] += 1                                                                 # (k): not the pinned length
    assert any("(k)" in x for x in sva.derived_problems(bad, plan=p, point="L-U2K", block=2))
    bad = copy.deepcopy(reqs)
    bad[50]["out_planned"] += 1
    assert any("out_planned" in x for x in sva.derived_problems(bad, plan=p, point="L-U2K", block=2))
    w = make_reqs(ADM, "W-50", 5)
    i = next(i for i, q in enumerate(w) if q["units"] == 4 and 32 <= i < 288)
    for change, needle in ((dict(units=1), "(l)"), (dict(cls="short"), "cls"), (dict(gw_sent_s=None), "gw_sent_s")):
        bad = copy.deepcopy(w)
        bad[i].update(change)
        assert any(needle in x for x in sva.derived_problems(bad, plan=ADM, point="W-50", block=5)), needle
    g = make_reqs(p, "GW-U2K-O512", 1)                                                  # fixed outputs: D-C's checks
    g[3]["out"] = 128
    assert any("out not all 512" in x for x in sva.derived_problems(g, plan=p, point="GW-U2K-O512", block=1))
    assert sva.derived_problems(None, plan=p, point="L-U2K", block=1) == ["no derived requests file"]
    for point, block, change, needle in (
            ("L-U2K", 2, lambda r: r[9].update(ttft_ms=None), "not numbers"),
            ("L-U2K", 2, lambda r: r[9].update(idx=10), "exactly once"),
            ("L-U2K", 2, lambda r: r[9].update(idx="9"), "integer idx"),
            ("GW-U2K-O512", 1, lambda r: r[9].update(gw_queue_wait_ms=-1.0), "negative"),
            ("GW-U2K-O512", 1, lambda r: r[9].update(gw_sent_s=r[9]["gw_released_s"] + 1), "gw_sent_s <="),
            ("GW-U2K-O512", 1, lambda r: [q.update(gw_arrival_s=q["gw_arrival_s"] + 1) for q in r], "earliest"),
            ("GW-U2K-O512", 1, lambda r: r[9].update(out_planned=128), "out_planned"),
            ("GW-U2K-O512", 1, lambda r: r[9].update(units=4), "(l)"),
            ("P-U2K", 1, lambda r: r[9].update(gw_prefill_wait_ms=None), "gw_prefill_wait_ms")):
        reqs = make_reqs(ADM if point[0] == "P" else p, point, block)
        change(reqs)
        got = sva.derived_problems(reqs, plan=ADM if point[0] == "P" else p, point=point, block=block)
        assert any(needle in x for x in got), (point, needle, got)


def test_a_dataset_must_have_the_points_n_rows():
    p = mini_plan()
    p["points"]["L-U8K"]["datasets"]["1"] = "v1-8192-r1"                    # 200 rows for the mini's N 20
    with pytest.raises(ca.AnalysisError, match="rows"):
        sva.planned(p, "L-U8K", 1)


def test_a_request_the_log_lacks_leaves_its_gateway_fields_empty():
    p, point = ADM, "W-50"
    reqs = make_reqs(p, point, 3)
    problems, joined, _ = sva.join_access_log(log_text(p, point, 3, reqs, change=lambda es: es.pop(40)), plan=p,
                                           point=point, block=3)
    assert problems and 40 not in joined
    got = sva.gateway_requests(client_part(p, point, 3, reqs), joined, plan=p, point=point, block=3)
    assert got[40]["gw_ttft_ms"] is None and got[40]["units"] is None and got[40]["out_planned"] == 128
    assert got[40]["cls"] == ("long" if sva.planned(p, point, 3)[40][0] == svd.PROMPT_8K else "short")
    assert any("gateway fields" in x for x in sva.derived_problems(got, plan=p, point=point, block=3))


# ------------------------------------------------------------------ in vLLM (U_W, N1, (j)) and E --

def interval_reqs(spans, *, n=320):
    """Requests whose arrivals put T_W = [0, 10) (idx 32 at 0 s, idx 288 at 10 s) and whose in-vLLM intervals are
    spans [(sent, released, units)], every other idx a zero-length visit before T_W."""
    reqs = []
    for i in range(n):
        a = 0.0 if i == 32 else 10.0 if i == 288 else -1.0 if i < 32 else 5.0
        reqs.append({"idx": i, "units": 1, "gw_arrival_s": a, "gw_sent_s": -1.0, "gw_released_s": -1.0})
    for k, (s, r, u) in enumerate(spans):
        reqs[40 + k].update(gw_sent_s=s, gw_released_s=r, units=u)
    return reqs


def test_u_w_is_the_exact_time_average_of_units_in_vllm_over_t_w():
    full = [(0.0, 10.0, 4)] * 8                                               # 32 units the whole of T_W
    iv = sva.in_vllm(interval_reqs(full), C=32, N=320)
    assert iv["_u_w"] == 32 and iv["_n_w"] == 8 and iv["t_w_s"] == [0.0, 10.0]
    hol = [(0.0, 10.0, 4)] * 7 + [(0.0, 3.0, 1), (3.5, 10.0, 4)]              # an 8K waits 0.5 s at the head
    iv = sva.in_vllm(interval_reqs(hol), C=32, N=320)
    assert iv["_u_w"] == 28 + Fraction(3, 10) + Fraction(4 * 65, 100)         # 28 + 0.3 + 2.6
    outside = [(-5.0, 0.5, 4), (9.5, 30.0, 4)]                                 # clipped to T_W
    assert sva.in_vllm(interval_reqs(outside), C=32, N=320)["_u_w"] == Fraction(4 * 1, 10)


def test_exactly_28_8_units_is_non_trivial_and_less_is_trivial():
    p, x = ADM, ADM["points"]["W-90"]
    edge = [(0.0, 10.0, 4)] * 7 + [(0.0, 2.0, 4)]                             # 28 + 4 x 0.2 = 28.8
    iv = sva.in_vllm(interval_reqs(edge), C=32, N=320)
    assert iv["_u_w"] == Fraction(144, 5)
    assert sva.nontrivial(iv, x=x, plan=p) == {"condition": "U_W", "value": 28.8, "threshold": 28.8, "ok": True}
    below = [(0.0, 10.0, 4)] * 7 + [(0.0, 1.999, 4)]
    assert sva.nontrivial(sva.in_vllm(interval_reqs(below), C=32, N=320), x=x, plan=p)["ok"] is False
    assert sva.nontrivial(iv, x=CLS["points"]["L-U2K"], plan=CLS) is None     # V-GW / V1: no condition
    m = mini_plan()                                                            # the mini's W at C 9: still the budget
    assert sva.nontrivial(iv, x=m["points"]["W-mini"], plan=m)["threshold"] == 28.8


def test_n1_is_requests_in_vllm_against_0_9_c():
    p = ADM
    x = p["points"]["P-U8K"]
    reqs = interval_reqs([(0.0, 10.0, 1)] * 7 + [(0.0, 2.0, 1)], n=200)
    for q in reqs:                                                            # T_W = [idx 8, idx 192)
        q["gw_arrival_s"] = 0.0 if q["idx"] == 8 else 10.0 if q["idx"] == 192 else q["gw_arrival_s"]
    iv = sva.in_vllm(reqs, C=8, N=200)
    assert iv["_n_w"] == Fraction(72, 10)
    assert sva.nontrivial(iv, x=x, plan=p) == {"condition": "N1", "value": 7.2, "threshold": 7.2, "ok": True}
    reqs[47]["gw_released_s"] = 1.9
    assert sva.nontrivial(sva.in_vllm(reqs, C=8, N=200), x=x, plan=p)["ok"] is False
    assert sva.nontrivial(sva.in_vllm(reqs, C=8, N=200), x=p["points"]["F-U2K"], plan=p)["condition"] == "N1"


def test_j_counts_the_warmups_too():
    """044 (j) over the whole bench process (實作補記六): the id-less completion records' intervals (row["warmups"],
    exactly the C warmups: 實作補記七) join the main requests' in the peaks; U_W and N1 stay on the main requests over
    T_W."""
    p = ADM
    reqs = make_reqs(p, "P-U8K", 1)
    row = row_for(p, "P-U8K", 1)
    assert sva.run_valid(row, reqs, p) == (True, [])
    late = dict(row, warmups=row["warmups"][:-1] + [[-30.0, 0.5, 1]])       # a warmup still in vLLM at the first wave
    ok, why = sva.run_valid(late, reqs, p)
    assert not ok and any("9 requests" in w and "(j)" in w for w in why) and sva.run_outcome(late, reqs, p) == "invalid"
    assert len(row["warmups"]) == 8                                          # C, no ready check (實作補記七)
    for bad, needle in ((dict(row, warmups=None), "warmup"), (dict(row, warmups=row["warmups"][:7]), "warmup"),
                        (dict(row, warmups=row["warmups"] + [[-30.0, -20.0, 1]]), "warmup"),       # C + 1
                        (dict(row, warmups=[[1.0, 0.5, 1]] + row["warmups"][1:]), "warmup"),
                        (dict(row, warmups=[[-30.0, -20.0, 4]] + row["warmups"][1:]), "warmup")):
        ok, why = sva.run_valid(bad, reqs, p)
        assert not ok and any(needle in w for w in why), why
    w = make_reqs(p, "W-90", 1)
    wrow = row_for(p, "W-90", 1)
    assert sva.run_valid(wrow, w, p) == (True, [])
    ok, why = sva.run_valid(dict(wrow, warmups=wrow["warmups"][:-1] + [[-0.5, 0.5, 1]]), w, p)
    assert not ok and any("units" in x and "(j)" in x for x in why)
    x = p["points"]["W-90"]
    span = [[-100.0, 1000.0, 1]]                                               # never in U_W / N1
    assert sva.in_vllm(w, C=32, N=320, warmups=span)["_u_w"] == sva.in_vllm(w, C=32, N=320)["_u_w"]
    assert sva.in_vllm(w, C=32, N=320, warmups=span)["peak_units"] == sva.in_vllm(w, C=32, N=320)["peak_units"] + 1
    assert sva.nontrivial(sva.in_vllm(w, C=32, N=320, warmups=span), x=x, plan=p)["ok"] is True


def test_j_holds_at_every_instant_on_half_open_intervals():
    x = ADM["points"]["W-90"]
    back_to_back = [(0.0, 10.0, 4)] * 7 + [(0.0, 3.0, 4), (3.0, 10.0, 4)]     # a release and a send at 3.0 s
    iv = sva.in_vllm(interval_reqs(back_to_back), C=32, N=320)
    assert iv["peak_units"] == 32 and iv["peak_by_units"] == {"4": 8} and sva.inflight_problems(iv, x=x) == []
    over = [(0.0, 10.0, 4)] * 8 + [(2.9, 3.1, 1)]
    got = sva.inflight_problems(sva.in_vllm(interval_reqs(over), C=32, N=320), x=x)
    assert any("33 units" in g for g in got)
    nine = [(0.0, 1.0, 4)] * 9
    assert any("9 v-uniform-8k" in g for g in sva.inflight_problems(sva.in_vllm(interval_reqs(nine), C=32, N=320),
                                                                     x=x))
    twos = [(0.0, 1.0, 1)] * 33
    got = sva.inflight_problems(sva.in_vllm(interval_reqs(twos), C=32, N=320), x=x)
    assert any("33 requests" in g for g in got) and any("33 v-uniform-2k" in g for g in got)
    p8 = ADM["points"]["P-U8K"]
    nine = interval_reqs([(0.0, 1.0, 1)] * 9, n=200)
    nine[8]["gw_arrival_s"], nine[192]["gw_arrival_s"] = 0.0, 10.0
    assert any("9 requests" in g and "C 8" in g for g in sva.inflight_problems(sva.in_vllm(nine, C=8, N=200), x=p8))
    empty = interval_reqs([(0.0, 10.0, 4)] * 8)                              # T_W = [0, 0): no average
    empty[288]["gw_arrival_s"] = 0.0
    iv = sva.in_vllm(empty, C=32, N=320)
    assert iv["u_w"] is None and sva.inflight_problems(iv, x=x) == []
    assert sva.nontrivial(iv, x=x, plan=ADM) == {"condition": "U_W", "value": None, "threshold": 28.8, "ok": False}


def test_e_needs_c_minus_b_minus_1_first_wave_waits():
    assert sva.burst_fit([1920] * 32, 9600) == 5 and sva.burst_fit([8064] * 8, 32256) == 4
    assert sva.burst_fit(sdd.lengths(2048, 32), 9600) == 19 and sva.burst_fit([9601], 9600) == 0
    assert sva.burst_fit([3, 1, 2], 3) == 2                                   # the shortest first, any arrival order
    want = {("V-ADM", "P-U2K"): (5, 26), ("V-ADM", "P-U8K"): (4, 3), ("V-ADM", "P-U2K-var"): (19, 12),
            ("mini", "P-U2K"): (5, 2)}
    for (s, point), (b, thr) in want.items():
        p = mini_plan() if s == "mini" else ADM
        reqs = make_reqs(p, point, 1, prefill={})
        x = p["points"][point]
        e = sva.manipulation_e(reqs, x=x)
        assert (e["b"], e["threshold"], e["waited"], e["ok"]) == (b, thr, 0, False), point
        for i in range(thr):
            reqs[x["C"] - 1 - i]["gw_prefill_wait_ms"] = 3.0
        assert sva.manipulation_e(reqs, x=x)["ok"] is True
        reqs[x["C"] - 1]["gw_prefill_wait_ms"] = 0.0
        assert sva.manipulation_e(reqs, x=x)["ok"] is False
        reqs[x["C"]]["gw_prefill_wait_ms"] = 3.0                              # after the first wave: never counted
        assert sva.manipulation_e(reqs, x=x)["ok"] is False
        row = row_for(p, point, 1)
        assert any("E" in r for r in sva.run_valid(row, reqs, p)[1]) and sva.run_outcome(row, reqs, p) == "invalid"
    off = make_reqs(ADM, "F-U2K-var", 1)
    assert sva.manipulation_e(off, x=ADM["points"]["F-U2K-var"]) == {"pacing": False, "waited": 0, "b": None,
                                                                     "threshold": None, "ok": True}
    off[200]["gw_prefill_wait_ms"] = 0.001
    assert sva.manipulation_e(off, x=ADM["points"]["F-U2K-var"])["ok"] is False
    assert sva.manipulation_e(make_reqs(CLS, "L-U2K", 1), x=CLS["points"]["L-U2K"]) is None


# ------------------------------------------------------------------ outcomes and classes --

def test_run_outcomes_invalid_fail_trivial_ok():
    p = ADM
    row = row_for(p, "P-U2K", 1)
    assert sva.run_outcome(row, make_reqs(p, "P-U2K", 1), p) == "ok"
    assert sva.run_outcome(row, make_reqs(p, "P-U2K", 1, ttft=2500.0, up=900.0), p) == "fail"   # client view
    assert sva.run_outcome(row, make_reqs(p, "P-U2K", 1, ttft=900.0, up=2500.0), p) == "fail"   # upstream view
    assert sva.run_outcome(row_for(p, "P-U2K", 1, valid=False), None, p) == "invalid"
    trivial = make_reqs(p, "P-U2K", 1, service={1: 1.0})
    for q in trivial:                                       # half the time in W only 16 requests are in vLLM
        if q["idx"] % 2 and 32 <= q["idx"] < 288:
            q["gw_released_s"] = q["gw_sent_s"] + 0.5
    assert sva.run_outcome(row, trivial, p) == "trivial"
    c = CLS
    assert sva.run_outcome(row_for(c, "L-U2K", 1), make_reqs(c, "L-U2K", 1, ttft=900.0, up=2500.0), c) == "ok"
    assert sva.run_outcome(row_for(c, "L-U2K", 1), make_reqs(c, "L-U2K", 1, ttft=2500.0, up=900.0), c) == "fail"
    w = make_reqs(p, "W-50", 1, ttft=2500.0, up=900.0)
    assert sva.run_outcome(row_for(p, "W-50", 1), w, p) in ("ok", "trivial")             # V4 judges upstream only


def test_the_runners_verdict_and_j_make_a_run_invalid():
    p = ADM
    reqs = make_reqs(p, "P-U8K", 2)
    assert sva.run_valid(row_for(p, "P-U8K", 2), reqs, p) == (True, [])
    ok, reasons = sva.run_valid(row_for(p, "P-U8K", 2, valid=False), reqs, p)
    assert not ok and reasons == ["runner: simulated invalid run"]
    assert sva.run_outcome(row_for(p, "P-U8K", 2, valid=False), reqs, p) == "invalid"
    reqs[100]["gw_released_s"] += 50.0                                          # a ninth request in vLLM at once
    ok, reasons = sva.run_valid(row_for(p, "P-U8K", 2), reqs, p)
    assert not ok and any("(j)" in r for r in reasons) and sva.run_outcome(row_for(p, "P-U8K", 2), reqs, p) == \
        "invalid"


def summary(klass):
    return {"class": klass}


@pytest.mark.parametrize("judged, outcomes, want", [
    (["FAIL"], ["ok", "trivial", "fail"], "FAIL"),
    (["PASS", "FAIL"], ["ok"] * 5, "FAIL"),
    (["PASS"], ["ok", "ok", "trivial"], "TRIVIAL"),
    (["MARGINAL"], ["ok", "ok", "ok", "ok", "trivial"], "TRIVIAL"),
    (["NT"], ["ok", "ok", "ok", "ok"], "NT"),
    (["PASS"], ["ok", "ok", "ok", "ok"], "NT"),
    (["NT"], ["ok"] * 5, "NT"),
    (["PASS"], ["ok"] * 5, "PASS"),
    (["PASS"], ["ok", "fail"], "FAIL"),
    (["PASS"], ["ok"] * 4 + ["invalid"], "NT"),
    (["PASS", "PASS"], ["ok"] * 5, "PASS"),
    (["PASS", "MARGINAL"], ["ok"] * 5, "MARGINAL"),
    (["MARGINAL"], ["ok"] * 5, "MARGINAL"),
])
def test_point_class_fail_trivial_nt_marginal_pass(judged, outcomes, want):
    assert sva.point_class([summary(k) for k in judged], outcomes, n=5) == want


def five_runs(p, point, kw):
    rows, requests = [], {}
    for r in range(1, 6):
        row = row_for(p, point, r, boot=r)
        requests[row["tag"]] = make_reqs(p, point, r, **kw(r))
        row["outcome"] = sva.run_outcome(row, requests[row["tag"]], p)
        rows.append(row)
    return rows, requests


def test_point_analysis_judges_the_worse_view_for_v2_and_the_guard_band():
    p = ADM
    rows, requests = five_runs(p, "P-U2K", lambda r: {"ttft": 1950.0, "up": 1000.0})
    out = sva.point_analysis(p, "P-U2K", rows, requests, slo=p["slo"])
    assert out["views"]["upstream"]["class"] == "PASS" and out["views"]["client"]["class"] == "MARGINAL"
    assert out["class"] == "MARGINAL" and out["judged_views"] == ["upstream", "client"]
    rows, requests = five_runs(p, "P-U2K", lambda r: {"ttft": 1950.0 if r == 4 else 1000.0, "up": 900.0})
    out = sva.point_analysis(p, "P-U2K", rows, requests, slo=p["slo"])
    assert out["class"] == "MARGINAL" and out["n_ok"] == 5
    rows, requests = five_runs(p, "P-U2K", lambda r: {"ttft": 1000.0, "up": 900.0})
    out = sva.point_analysis(p, "P-U2K", rows, requests, slo=p["slo"])
    assert out["class"] == "PASS" and out["window_sensitive"] is False
    rows, requests = five_runs(p, "P-U2K", lambda r: {"ttft": 1000.0, "up": 1950.0})
    assert sva.point_analysis(p, "P-U2K", rows, requests, slo=p["slo"])["class"] == "MARGINAL"   # upstream counts too
    c = CLS
    rows, requests = five_runs(c, "GW-U2K-O512", lambda r: {"ttft": 1000.0, "up": 1950.0})
    out = sva.point_analysis(c, "GW-U2K-O512", rows, requests, slo=c["slo"])
    assert out["class"] == "PASS" and out["views"]["upstream"]["class"] == "MARGINAL"    # V-GW: the client view
    rows, requests = five_runs(p, "W-90", lambda r: {"ttft": 2500.0, "up": 900.0, "long_up": 2400.0})
    out = sva.point_analysis(p, "W-90", rows, requests, slo=p["slo"])
    assert out["class"] == "PASS" and out["views"]["client"]["class"] == "FAIL"          # V4: the upstream view


def test_window_sensitive_compares_w_with_w_prime():
    c = CLS                                                    # a slow second wave: W's P95, never W' = [2C, N - C)
    rows, requests = five_runs(c, "GW-U2K-O512", lambda r: {"ttft": 1000.0, "slow_idx": range(32, 64)})
    out = sva.point_analysis(c, "GW-U2K-O512", rows, requests, slo=c["slo"])
    assert out["views"]["client"]["class"] == "FAIL" and out["views"]["client"]["class_late"] == "PASS"
    assert out["class"] == "FAIL" and out["window_sensitive"] is True


def test_a_trivial_run_makes_the_point_trivial():
    p = ADM
    rows, requests = five_runs(p, "F-U2K", lambda r: {"ttft": 1000.0, "up": 900.0})
    t = requests[rows[2]["tag"]]
    for q in t:
        if q["idx"] % 2 and 32 <= q["idx"] < 288:
            q["gw_released_s"] = q["gw_sent_s"] + 0.5
    rows[2]["outcome"] = sva.run_outcome(rows[2], t, p)
    out = sva.point_analysis(p, "F-U2K", rows, requests, slo=p["slo"])
    assert rows[2]["outcome"] == "trivial" and out["class"] == "TRIVIAL"
    assert out["runs"][2]["nontrivial"]["ok"] is False and out["runs"][2]["nontrivial"]["condition"] == "N1"


# ------------------------------------------------------------------ the replay (044 replay, 停止規則與時數上限) --

def test_a_simulated_session_replays_clean():
    for p in (ADM, CLS, mini_plan()):
        cap, requests = simulate(p, ok_kw)
        assert sva.replay_problems(p, cap["session"]["boots"], cap["rows"], requests,
                                   cap_stop=cap["session"]["cap_stop"]) == [], p["session"]


def _replay(cap, requests, **kw):
    s = cap["session"]
    return sva.replay_problems(cap["plan"], s["boots"], cap["rows"], requests, cap_stop=s.get("cap_stop"), **kw)


def test_the_replay_recomputes_the_block_order():
    cap, requests = simulate(ADM, ok_kw)
    boots = cap["session"]["boots"]
    boots[1]["workload"], boots[2]["workload"] = boots[2]["workload"], boots[1]["workload"]
    assert any("044 order" in x for x in _replay(cap, requests))
    p = copy.deepcopy(ADM)
    p["block_orders"]["1"] = sorted(p["block_orders"]["1"])
    assert any("block_order" in x for x in sva.replay_problems(p, [], [], {}))


def fail_kw(fail):
    def kw(point, block):
        base = ok_kw(point, block)
        return {**base, "ttft": 9000.0, "up": 9000.0, "long_up": 9000.0} if (point, block) in fail else base
    return kw


@pytest.mark.parametrize("first", ["L-U2K", "GW-U2K-O512"])
def test_a_failing_run_skips_only_its_own_point_in_the_later_blocks(first):
    """044 實作補記六: the inherited 039 / 030 rule, per point: its unit's other point keeps running in every block
    (its own hypothesis stays answerable), and the unit cannot hold."""
    partner = {"L-U2K": "GW-U2K-O512", "GW-U2K-O512": "L-U2K"}[first]
    cap, requests = simulate(CLS, fail_kw({(first, 2)}))
    booted = [(b["block"], b["workload"]) for b in cap["session"]["boots"] if b["phase"] == "V"]
    assert (2, first) in booted and not any(w == first for r, w in booted if r > 2)
    assert [r for r, w in booted if w == partner] == [1, 2, 3, 4, 5]          # same block or later: it still runs
    assert _replay(cap, requests) == []
    analysis = sva.analyze(cap, requests)
    assert analysis["points"][partner]["class"] == "PASS" and analysis["points"][first]["class"] == "FAIL"
    assert analysis["schedule"]["skipped"] == [[3, first], [4, first], [5, first]]
    assert analysis["points"][partner]["skipped"] == [] and analysis["points"][first]["skipped"] == [3, 4, 5]
    assert analysis["units"]["V1-2K"]["holds"] is False and analysis["units"]["V1-8K"]["holds"] is True
    cap2 = copy.deepcopy(cap)                                                  # a session that kept on booting it
    good, _ = simulate(CLS, ok_kw)
    assert _replay(good, requests) != []                                       # the good rows disagree with ...
    rows = cap2["rows"]
    tag = sv.run_tag("V", first, CLS["points"][first]["C"], block=2)
    row = next(r for r in rows if r["tag"] == tag)
    row["outcome"] = "ok"
    assert any("recorded outcome" in x for x in _replay(cap2, requests))
    unit = next(pts for pts in sv.UNITS.values() if first in pts)             # the old unit rule's session: refused
    dead, dead_requests = simulate(CLS, fail_kw({(first, 2)}), drop=lambda q: unit if q in unit else [q])
    assert any(partner in x for x in _replay(dead, dead_requests))


def test_every_preregistered_hypothesis_stays_answerable_when_its_unit_partner_fails():
    """Why per point (044 實作補記六): with the predicted W-90 FAIL in its first run W-50 still runs five times (H-W2,
    H-W0), and a V1 point's failure in block 1 leaves its partner's hypothesis (H-L1, H-GW3) to be answered."""
    cap, requests = simulate(ADM, fail_kw({("W-90", 1)}))
    a = sva.analyze(cap, requests)
    assert a["replay"]["ok"] and a["points"]["W-50"]["n_ok"] == 5 and a["points"]["W-90"]["class"] == "FAIL"
    assert a["hypotheses"]["H-W2"]["verdict"] == "supported" and a["hypotheses"]["H-W0"]["verdict"] == "supported"
    assert a["units"]["W1"]["holds"] is False
    cap, requests = simulate(CLS, fail_kw({("GW-U2K-O512", 1), ("L-U8K", 1)}))
    a = sva.analyze(cap, requests)
    assert a["replay"]["ok"] and a["hypotheses"]["H-L1"]["verdict"] == "supported"
    assert a["hypotheses"]["H-GW3"]["verdict"] == "supported"
    assert not a["units"]["V1-2K"]["holds"] and not a["units"]["V1-8K"]["holds"]


def test_the_044_hour_figures_follow_from_the_plan():
    """044 Sessions 與時數 and 待決事項第 7 項, from the plans' est_s under the per-point skip rule: V-ADM 2.75 h, 1.60 h
    as predicted (P-U2K-var, F-U2K-var, W-90 and W-50 each FAIL in their first run); V-CLS 5.26 h, 4.59 h when
    GW-M8K-T FAILs in its first run; the predicted W cost is one W-90 and one W-50 run, about 0.2 h."""
    def hours(p, fails=()):
        cap, _ = simulate(p, fail_kw(set(fails)))
        b = cap["session"]["boots"]
        return (b[-1]["elapsed_s"] + p["sentinel"]["est_s"]) / 3600

    first = {q: next(r for r in range(1, 6) if q in ADM["block_orders"][str(r)]) for q in ADM["points"]}
    assert round(hours(ADM), 2) == 2.75 and round(ADM["estimate_s"] / 3600, 2) == 2.75
    predicted = [(q, first[q]) for q in ("P-U2K-var", "F-U2K-var", "W-90", "W-50")]
    assert round(hours(ADM, predicted), 2) == 1.60
    w = ADM["points"]["W-90"]["est_s"] + ADM["points"]["W-50"]["est_s"]
    assert round(w / 3600, 2) == 0.19
    assert round(hours(CLS), 2) == 5.26 and round(hours(CLS, [("GW-M8K-T", 1)]), 2) == 4.59


def test_a_trivial_run_and_an_invalid_run_skip_too():
    order = [(r, q) for r in range(1, 6) for q in ADM["block_orders"][str(r)]]
    r0 = next(r for r, q in order if q == "W-90")
    cap, requests = simulate(ADM, ok_kw, invalid={("W-90", r0)})
    booted = [(b["block"], b["workload"]) for b in cap["session"]["boots"] if b["phase"] == "V"]
    assert not any(w == "W-90" for r, w in booted if r > r0) and _replay(cap, requests) == []
    assert [r for r, w in booted if w == "W-50"] == [1, 2, 3, 4, 5]          # its unit partner: not skipped

    cap, requests = simulate(ADM, ok_kw)
    assert len([b for b in cap["session"]["boots"] if b["workload"] == "F-U2K"]) == 5 and _replay(cap, requests) == []
    first = next(r for r, q in order if q == "F-U2K")
    tag = sv.run_tag("V", "F-U2K", 32, block=first)
    for q in requests[tag]:
        if q["idx"] % 2 and 32 <= q["idx"] < 288:
            q["gw_released_s"] = q["gw_sent_s"] + 0.5
    assert any("recomputed 'trivial'" in x for x in _replay(cap, requests))  # the run was trivial: no later boots


def test_the_hour_cap_is_recomputed_from_the_recorded_times():
    p = CLS
    cap, requests = simulate(p, ok_kw, slow=lambda q, r: 7200.0 if (q, r) == (p["block_orders"]["3"][0], 3) else 0)
    stop = cap["session"]["cap_stop"]
    assert stop is not None and stop["forced"] is False and _replay(cap, requests) == []
    analysis = sva.analyze(cap, requests)
    not_run = analysis["schedule"]["not_run_cap"]
    assert not_run[0] == [stop["block"], stop["point"]] and len(not_run) > 1
    assert all(analysis["points"][q]["class"] == "NT" for _, q in not_run)
    assert all(r in analysis["points"][q]["not_run_cap"] for r, q in not_run)
    late = copy.deepcopy(cap)                                                 # recorded later than the cap allows
    late["session"]["boots"][-2]["elapsed_s"] = late["session"]["cap_stop"]["elapsed_s"] = 7 * 3600.0
    late["session"]["boots"][-1]["elapsed_s"] = 7 * 3600.0
    assert any("hour cap" in x for x in _replay(late, requests))
    early = copy.deepcopy(cap)                                                # a stop the cap did not call for
    for b in early["session"]["boots"]:
        b["elapsed_s"] = min(b["elapsed_s"], 100.0)
    early["session"]["cap_stop"]["elapsed_s"] = 100.0
    assert any("cap allowed" in x for x in _replay(early, requests))
    nostop = copy.deepcopy(cap)
    nostop["session"]["cap_stop"] = None
    assert any("cap" in x for x in _replay(nostop, requests))
    back = copy.deepcopy(cap)
    back["session"]["boots"][3]["elapsed_s"] = back["session"]["boots"][2]["elapsed_s"] - 1
    assert any("elapsed_s" in x for x in _replay(back, requests))


def test_the_minis_forced_stop_comes_before_its_last_planned_v_boot():
    p = mini_plan()
    cap, requests = simulate(p, ok_kw)
    stop = cap["session"]["cap_stop"]
    assert stop["forced"] is True and (stop["block"], stop["point"]) == (2, p["block_orders"]["2"][-1])
    assert _replay(cap, requests) == []
    missing = copy.deepcopy(cap)
    missing["session"]["cap_stop"]["forced"] = False
    assert any("cap allowed" in x for x in _replay(missing, requests))
    formal = copy.deepcopy(cap)
    formal["plan"] = dict(formal["plan"], mini=None)
    assert any("forced" in x for x in _replay(formal, requests))


def test_the_minis_last_planned_v_boot_is_the_last_one_still_planned():
    p = mini_plan()
    cap, requests = simulate(p, ok_kw, invalid={("GW-M8K-T", 1)})         # VG-M8K-T is dead before block 2
    stop = cap["session"]["cap_stop"]
    last = [q for q in p["block_orders"]["2"] if q != "GW-M8K-T"][-1]
    assert (stop["block"], stop["point"], stop["forced"]) == (2, last, True) and _replay(cap, requests) == []


def test_a_cap_stop_must_sit_between_its_boots_and_be_reached():
    p = CLS
    cap, requests = simulate(p, ok_kw, slow=lambda q, r: 7200.0 if (q, r) == (p["block_orders"]["3"][0], 3) else 0)
    early = copy.deepcopy(cap)
    early["session"]["cap_stop"]["elapsed_s"] = early["session"]["boots"][-2]["elapsed_s"] - 1
    assert any("between" in x for x in _replay(early, requests))
    good, requests = simulate(ADM, ok_kw)
    good["session"]["cap_stop"] = {"block": 9, "point": "W-90", "elapsed_s": 1.0, "forced": False}
    assert any("not where this replay stops" in x for x in _replay(good, requests))


def test_the_minis_forced_timeout_is_its_only_other_timeout_and_walks_the_skip_rule():
    """044 實作補記六: the forced 5 s timeout in GW-U8K-O512's block 1, so the skip rule drops its block 2 on the
    hardware (n = 2: a block-2 timeout could skip nothing); L-U8K keeps both runs for the output_lens pins."""
    p = mini_plan()
    assert p["points"]["GW-U8K-O512"]["forced_timeout_blocks"] == [1] and sva.FORCED_TIMEOUT_S == \
        rc.MINI_FORCED_TIMEOUT_S
    cap, requests = simulate(p, ok_kw, invalid={("GW-U8K-O512", 1)})
    row = next(r for r in cap["rows"] if r.get("block") == 1 and r["workload"] == "GW-U8K-O512")
    assert row["timeout_s"] == 5 and row["outcome"] == "invalid" and _replay(cap, requests) == []
    problems, sched = sva.schedule(p, cap["session"]["boots"], cap["rows"], requests,
                                   cap_stop=cap["session"]["cap_stop"])
    assert problems == [] and [2, "GW-U8K-O512"] in sched["skipped"]
    assert [r["block"] for r in cap["rows"] if r["workload"] == "L-U8K"] == [1, 2]
    row["timeout_s"] = p["points"]["GW-U8K-O512"]["timeout_s"]
    assert any("timeout_s" in x for x in _replay(cap, requests))


def test_rows_and_boots_must_be_the_plans():
    cap, requests = simulate(ADM, ok_kw)
    for change, needle in ((lambda c: c["rows"][3].update(tag="V-x"), "tag"),
                           (lambda c: c["rows"][3].update(N=1), "N"),
                           (lambda c: c["rows"].append(dict(c["rows"][3])), "duplicate"),
                           (lambda c: c["session"]["boots"].pop(4), "boot"),
                           (lambda c: c["session"]["boots"][4].update(reached_first_bench=False), "first bench"),
                           (lambda c: c["session"]["boots"].append(dict(c["session"]["boots"][-1], boot=99)),
                            "numbered"),
                           (lambda c: c["rows"][5].update(boot=0), "sentinel boot holds"),
                           (lambda c: c["rows"].pop(5), "ran []"),
                           (lambda c: (c["rows"][5].update(boot=c["rows"][6]["boot"]),
                                       c["rows"][6].update(boot=c["rows"][5]["boot"] - 1)), "ran [("),
                           (lambda c: c["session"]["boots"].append(dict(c["session"]["boots"][-1],
                                                                        boot=len(c["session"]["boots"]))),
                            "after the end sentinel")):
        c = copy.deepcopy(cap)
        change(c)
        assert any(needle in x for x in _replay(c, requests)), needle


# ------------------------------------------------------------------ units, readings, hypotheses --

def classes(**k):
    return {p: {"class": c} for p, c in k.items()}


def test_each_unit_holds_only_when_every_point_passes():
    pts = {**classes(**{"GW-U32K": "PASS", "GW-M8K-T": "MARGINAL", "L-U2K": "PASS", "GW-U2K-O512": "PASS",
                        "L-U8K": "PASS", "GW-U8K-O512": "TRIVIAL"})}
    got = sva.units(CLS, pts, session_ok=True)
    assert {u: x["holds"] for u, x in got.items()} == {"VG-32K": True, "VG-M8K-T": False, "V1-2K": True,
                                                        "V1-8K": False}
    assert not any(x["holds"] for x in sva.units(CLS, pts, session_ok=False).values())
    for k in ("FAIL", "NT", "MARGINAL", "TRIVIAL"):
        assert sva.units(CLS, {**pts, "L-U2K": {"class": k}}, session_ok=True)["V1-2K"]["holds"] is False


@pytest.mark.parametrize("paced, control, pu2k, holds, vetoed", [
    ("FAIL", "FAIL", "PASS", True, False),
    ("PASS", "FAIL", "PASS", True, False),
    ("PASS", "PASS", "PASS", True, False),
    ("MARGINAL", "PASS", "PASS", False, True),
    ("FAIL", "PASS", "PASS", False, True),
    ("NT", "PASS", "PASS", False, True),
    ("MARGINAL", "MARGINAL", "PASS", True, False),
    ("FAIL", "FAIL", "MARGINAL", False, False),
])
def test_pace_2k_is_vetoed_when_the_control_var_passes_and_the_paced_var_does_not(paced, control, pu2k, holds, vetoed):
    pts = classes(**{"P-U2K": pu2k, "P-U8K": "PASS", "P-U2K-var": paced, "F-U2K-var": control, "F-U2K": "PASS",
                     "W-90": "PASS", "W-50": "PASS"})
    got = sva.units(ADM, pts, session_ok=True)
    assert (got["PACE-2K"]["holds"], got["PACE-2K"]["vetoed"]) == (holds, vetoed)
    assert got["PACE-8K"]["holds"] is True and got["W1"]["holds"] is True and got["W1"]["vetoed"] is None
    assert sva.units(ADM, {**pts, "W-50": {"class": "TRIVIAL"}}, session_ok=True)["W1"]["holds"] is False


@pytest.mark.parametrize("paced, control, want", [
    ("PASS", "FAIL", "co_occur"), ("PASS", "PASS", "var_passes_without_pacing"), ("FAIL", "FAIL", "not_fixed"),
    ("MARGINAL", "PASS", "veto"), ("NT", "PASS", "veto"), ("MARGINAL", "FAIL", "uninterpretable"),
    ("PASS", "MARGINAL", "uninterpretable"), ("TRIVIAL", "TRIVIAL", "uninterpretable")])
def test_the_var_pair_reading(paced, control, want):
    pts = classes(**{"P-U2K": "PASS", "F-U2K": "PASS", "P-U2K-var": paced, "F-U2K-var": control})
    assert sva.readings(pts)["var"]["reading"] == want


@pytest.mark.parametrize("paced, control, want", [
    ("PASS", "PASS", "formal_class_holds"), ("MARGINAL", "PASS", "attributed_to_pacing"),
    ("FAIL", "NT", "formal_class_lost"), ("PASS", "FAIL", "unattributable")])
def test_the_p_u2k_and_f_u2k_reading(paced, control, want):
    pts = classes(**{"P-U2K": paced, "F-U2K": control, "P-U2K-var": "FAIL", "F-U2K-var": "FAIL"})
    assert sva.readings(pts)["uniform"]["reading"] == want
    assert sva.readings(classes(**{"GW-U32K": "PASS"})) == {}


def test_the_unit_validation_section_names_what_the_generator_writes():
    cap, requests = simulate(CLS, ok_kw)
    a = sva.analyze(cap, requests)
    assert a["replay"]["ok"] and a["sentinel"]["gate"]["ok"]
    assert {u: x["holds"] for u, x in a["units"].items()} == {u: True for u in CLS["units"]}
    v = a["units"]["V1-8K"]["validation"]
    assert v["study"] == "V1" and v["unit"] == "V1-8K" and v["points"] == ["L-U8K", "GW-U8K-O512"]
    assert v["measured_classes"] == ["v-uniform-8k-o512-c8", "v-uniform-8k-o512"] and v["n"] == 5
    assert set(v["worst_r"]) == {"L-U8K", "GW-U8K-O512"} and set(v["worst_r"]["L-U8K"]) == {"client", "upstream"}
    assert v["worst_r"]["L-U8K"]["client"] == pytest.approx(
        max(x for x in a["points"]["L-U8K"]["views"]["client"]["worst_R"].values() if x is not None))
    assert v["datasets"] == {"L-U8K": [f"v1-8192-r{r}" for r in range(1, 6)]}
    m = sva.analyze(*simulate(mini_plan(), ok_kw))
    assert not any(x["holds"] for x in m["units"].values())                   # a mini never unlocks a unit


def test_hypotheses_map_classes_to_verdicts_with_044s_predictions():
    """044's predictions are sets of point classes (實作補記六): supported {PASS}, not supported {FAIL, TRIVIAL},
    H-GW4 {MARGINAL, FAIL} as 044 writes it; prediction_met is mechanical, None for an NT point."""
    pts = classes(**{"GW-U32K": "PASS", "GW-U2K-O512": "FAIL", "GW-U8K-O512": "TRIVIAL", "GW-M8K-T": "MARGINAL",
                     "L-U2K": "NT", "L-U8K": "PASS"})
    h = sva.hypotheses(pts)
    assert {k: (v["verdict"], v["predicted_classes"], v["prediction_met"]) for k, v in h.items()} == {
        "H-GW1": ("supported", ["PASS"], True), "H-GW2": ("not supported", ["PASS"], False),
        "H-GW3": ("not supported", ["PASS"], False), "H-GW4": ("inconclusive", ["FAIL", "MARGINAL"], True),
        "H-L1": ("NT", ["PASS"], None), "H-L2": ("supported", ["PASS"], True)}
    for klass, met in (("FAIL", True), ("PASS", False), ("TRIVIAL", False), ("NT", None)):
        assert sva.hypotheses({**pts, "GW-M8K-T": {"class": klass}})["H-GW4"]["prediction_met"] is met, klass
    adm = {**classes(**{"P-U2K": "PASS", "P-U8K": "MARGINAL", "P-U2K-var": "FAIL", "F-U2K-var": "FAIL",
                        "F-U2K": "PASS", "W-90": "FAIL", "W-50": "TRIVIAL"})}
    adm["W-90"]["runs"] = [{"valid": True, "nontrivial": {"ok": True}}, {"valid": False, "nontrivial": {"ok": False}}]
    adm["W-50"]["runs"] = [{"valid": True, "nontrivial": {"ok": False}}]
    h = sva.hypotheses(adm)
    assert {k: (v["verdict"], v.get("predicted_classes", v.get("prediction")), v["prediction_met"])
            for k, v in h.items()} == {
        "H-P1": ("supported", ["PASS"], True), "H-P2": ("inconclusive", ["PASS"], False),
        "H-P3": ("not supported", ["FAIL", "TRIVIAL"], True), "H-W0": ("not supported", "supported", False),
        "H-W1": ("not supported", ["FAIL", "TRIVIAL"], True), "H-W2": ("not supported", ["FAIL", "TRIVIAL"], True)}
    adm["W-50"]["runs"] = [{"valid": True, "nontrivial": {"ok": True}}]
    h = sva.hypotheses(adm)["H-W0"]                                            # an invalid run never counts
    assert (h["verdict"], h["prediction_met"]) == ("supported", True)
    adm["W-50"]["runs"] = [{"valid": False}]
    h = sva.hypotheses(adm)["H-W0"]
    assert (h["verdict"], h["prediction_met"]) == ("NT", None)
    assert {h: set(c) for h, (_, c) in sva.HYPOTHESES.items()} == {
        **{h: {"PASS"} for h in ("H-GW1", "H-GW2", "H-GW3", "H-L1", "H-L2", "H-P1", "H-P2")},
        "H-GW4": {"MARGINAL", "FAIL"}, **{h: {"FAIL", "TRIVIAL"} for h in ("H-P3", "H-W1", "H-W2")}}


# ------------------------------------------------------------------ the analysis --

def test_analyze_a_v_adm_session_end_to_end():
    cap, requests = simulate(ADM, ok_kw)
    a = sva.analyze(cap, requests)
    assert a["stage"] == "V" and a["session"] == "V-ADM" and a["replay"] == {"ok": True, "problems": []}
    assert {p: x["class"] for p, x in a["points"].items()} == {p: "PASS" for p in ADM["points"]}
    w = a["points"]["W-90"]
    assert w["judged_views"] == ["upstream"] and all(r["nontrivial"]["condition"] == "U_W" for r in w["runs"])
    assert w["descriptive"]["window_counts"] == {str(r): svd.window_counts(f"v4-90-r{r}", 32) for r in range(1, 6)}
    pu = a["points"]["P-U2K"]["runs"][0]
    assert pu["e"]["threshold"] == 26 and pu["e"]["ok"] and pu["nontrivial"]["condition"] == "N1"
    d = a["points"]["P-U2K"]["descriptive"]["decomposition"]
    assert set(d) == {"transport_ms", "gateway_ms", "vllm_ms", "queue_wait_ms", "prefill_wait_ms"}
    assert d["transport_ms"]["p50"] == pytest.approx(2.0) and d["vllm_ms"]["p95"] == pytest.approx(700.0)
    assert set(a["secondary_slo"]) == {"S1", "S3"} and a["units"]["PACE-2K"]["holds"] is True
    eff = w["descriptive"]["requests_vs_effective"]
    assert eff["effective_concurrency"] == 24.6 and len(eff["n_w_at_least_0_9"]) == 5
    assert eff["n_w_at_least_0_9"] == [n >= 0.9 * 24.6 for n in w["descriptive"]["in_vllm"]["n_w"]]
    assert pu["inflight_peak"] == 32 and pu["inflight_le_c"] is True
    assert a["hypotheses"]["H-W0"]["verdict"] == "supported" and a["readings"]["var"]["reading"] == \
        "var_passes_without_pacing"
    assert json.loads(json.dumps(a)) == a                                     # plain JSON: the report re-runs it


def test_analyze_reports_waves_bins_and_the_in_vllm_averages():
    cap, requests = simulate(CLS, lambda q, r: {**ok_kw(q, r), "slow_idx": [*range(32, 40), 70]})
    a = sva.analyze(cap, requests)
    run = a["points"]["L-U2K"]["runs"][0]
    assert run["waves"]["client"]["second_max_ms"] == pytest.approx(2400.0)
    assert run["waves"]["client"]["second_over"] == 8 and run["waves"]["client"]["first_max_ms"] == 800.0
    bins = a["points"]["L-U2K"]["descriptive"]["out_bins"]
    assert [b["bin"] for b in bins] == [[1, 16], [17, 64], [65, 128], [129, 256], [257, 512]]
    assert sum(b["n"] for b in bins) == 5 * 256
    assert a["points"]["L-U2K"]["runs"][0]["in_vllm"]["n_w"] == pytest.approx(32.0)
    assert "out_bins" not in a["points"]["GW-U32K"]["descriptive"]


def test_a_replay_problem_unlocks_no_unit():
    cap, requests = simulate(CLS, ok_kw)
    cap["rows"][4]["outcome"] = "fail"
    a = sva.analyze(cap, requests)
    assert a["replay"]["ok"] is False and not any(x["holds"] for x in a["units"].values())


def test_the_sentinel_gate_and_secondary_slos_are_stage_ds():
    cap, requests = simulate(ADM, ok_kw)
    end = next(r for r in cap["rows"] if r.get("position") == "end")
    requests[end["tag"]] = [dict(q, ttft_ms=q["ttft_ms"] * 1.05) for q in requests[end["tag"]]]
    a = sva.analyze(cap, requests)
    assert a["sentinel"]["gate"]["ok"] is False and not any(x["holds"] for x in a["units"].values())


# ------------------------------------------------------------------ isolation --

def test_the_analysis_imports_only_research_modules():
    tree = ast.parse((ROOT / "benchmarks" / "stage_v_analysis.py").read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert names <= {"__future__", "json", "math", "fractions", "capacity_analysis", "capacity_plan",
                     "core_b_analysis", "core_b_plan", "stage_d_analysis", "stage_d_datasets", "stage_d_plan",
                     "stage_v_datasets", "stage_v_plan"}
    tree = ast.parse((ROOT / "benchmarks" / "stage_v_datasets.py").read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert names <= {"__future__", "argparse", "hashlib", "json", "random", "pathlib", "typing", "core_b_plan",
                     "stage_d_plan", "stage_v_plan", "stage_d_datasets"}
