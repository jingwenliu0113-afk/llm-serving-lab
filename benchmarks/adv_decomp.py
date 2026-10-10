#!/usr/bin/env python3
"""Advanced (decision 024) per-run measurement and TTFT decomposition, fixed before any Advanced data.
Descriptive only: it feeds H-C4(ii) and H-C5 and never decides a removal category (A5).

Inputs of one boot (one bench), all on the same WSL wall clock (time.time()):
- server log lines of --enable-logging-iteration-details (vllm/v1/metrics/loggers.py:185-201), each
  prefixed by the runner's reader thread as "<epoch_ms> <original line>"; dummy lines are skipped and
  0-token steps never appear (core.py:515-560: the iteration index only advances on logged steps);
- OTLP `llm_request` spans (output_processor.py:782-842; names in vllm/tracing/utils.py): span start =
  frontend arrival, gen_ai.latency.time_in_queue Q = scheduled_ts - queued_ts (first schedule only,
  stats.py:520-524), time_in_model_prefill P = first_token_ts - scheduled_ts (includes re-queueing after
  a preemption), time_to_first_token = server TTFT from arrival;
- the bench --save-detailed lists; list index i is request id f"cmpl-{prefix}{i}-0" (bench serve sends
  x-request-id = prefix + i, datasets.py:665 / 2624; the completions endpoint uses
  f"cmpl-{x-request-id}-{prompt index}", completion/serving.py:143 / 179). Warmup and ready-check
  requests carry no id (serve.py:840-910): kept as "other" for the queue reconstruction only;
- Prometheus text from before and after the bench (deltas with counter-reset detection).

Rules (024 A8-A10, A13):
- per request: O = client TTFT - server TTFT (024 "量測"), F = server TTFT - Q - P (arrival -> queued,
  frontend + IPC), so O + F + Q + P = client TTFT; first token f = arrival + server TTFT, admission
  s = f - P, enqueue e = s - Q; a long request (32640) prefills over [s, f];
- step j covers I_j = (t_{j-1}, t_j]; its decision time is tau_j = t_{j-2} with
  free_est = round((1 - u_{j-2} / 100)(nb - 1)) (async scheduling, batch queue 2); the sensitivity
  analysis uses t_{j-1} / u_{j-1}. W_j = {i : e_i <= tau_j < s_i}, head = smallest e_i;
  N_j = ctx + gen tokens; categories: no-queue (W_j empty), budget-long / budget-other (N_j = 2048, a long
  request prefilling at tau_j or not), HOL-long / KV-short (N_j < 2048 and free_est < the head's demand:
  gate on ceil(prompt / 528) + 3, gate off the first chunk's), unexplained (otherwise); max-seqs
  (scheduled requests >= 128) is a flag only;
- each tail request's [e_i, s_i] is split over the I_j by overlap; time in no step, or in a step
  without tau, is "unclassified". Tail = short requests (L - 128) with client TTFT >= that run's short
  P90 (linear interpolation, ties included); the >= P95 subset is reported as well;
- validity (A10): (a) N main spans with exactly the client's prompt / completion tokens; (b) queue-time
  histogram delta count = spans received, |delta sum - sum Q| <= 1 ms; (c) without preemptions,
  sum ctx_tokens of the window's steps = delta vllm:prompt_tokens_total (exempt = None otherwise);
  (d) the window's iteration indices are consecutive.

summarize_run returns row["decomp"] (times in ms): "tail" {O, Q, P, F}; "q_shares" / "q_shares_sens"
are fractions of the tail's total Q, so the tail mean Q of category c is q_shares[c] x tail["Q"].
residual_bm_share = steps with >= 4 long requests prefilling at tau_j / steps; unexplained_break_share =
unexplained steps / queued steps (both over the main requests' [first arrival, last finish], 0.0 when
the denominator is empty).

Usage (from run_adv.py):
    import adv_decomp as ad
    steps = ad.parse_iteration_log(open(server_log).read().splitlines())
    spans = [json.loads(x) for x in open(spans_jsonl)]          # decode_otlp_traces(...) output
    row["decomp"] = ad.summarize_run(spans=spans, detailed=bench_json, steps=steps, prefix=prefix, L=L,
                                     cond=cond, num_gpu_blocks=plan["expected_blocks"][cond],
                                     metrics_before=text0, metrics_after=text1, window=(t0, t1))
    rows = ad.derived_requests(ad.align(ad.llm_request_records(spans), bench_json, prefix=prefix, L=L))

Stdlib only, Python 3.10 compatible (opentelemetry and run_core_b are imported lazily).
"""

from __future__ import annotations

import gzip as _gzip
import heapq
import math
import re

import core_b_analysis as ba
import core_b_plan as bp

BUDGET = bp.MAX_NUM_BATCHED_TOKENS  # 2048
BLOCK_SIZE = 528                     # Stage B block_size (hybrid alignment)
GDN_BLOCKS = 3                       # per-request GDN state blocks (C_KV, 008 / 009)
LONG_PREFILL_TOKEN_THRESHOLD = 512   # T (024 H-C1)
RESIDUAL_BM_LONGS = BUDGET // LONG_PREFILL_TOKEN_THRESHOLD  # 4 long chunks of 512 fill the budget again
QUEUE_TOL_S = 0.001
LAG_MAIN, LAG_SENS = 2, 1
LLM_REQUEST = "llm_request"
SPAN_ATTRS = {"id": "gen_ai.request.id", "prompt_tokens": "gen_ai.usage.prompt_tokens",
              "completion_tokens": "gen_ai.usage.completion_tokens", "Q": "gen_ai.latency.time_in_queue",
              "P": "gen_ai.latency.time_in_model_prefill", "server_ttft": "gen_ai.latency.time_to_first_token",
              "e2e": "gen_ai.latency.e2e", "decode": "gen_ai.latency.time_in_model_decode",
              "inference": "gen_ai.latency.time_in_model_inference"}
CATEGORIES = ("no-queue", "budget-long", "budget-other", "HOL-long", "KV-short", "unexplained")
SHARE_KEYS = CATEGORIES + ("unclassified",)
QUEUE_HIST = "vllm:request_queue_time_seconds"
PROMPT_TOKENS = "vllm:prompt_tokens_total"
PREEMPTIONS = "vllm:num_preemptions_total"
_KNOBS = {"B0": (True, False), "T": (True, True), "G": (False, False), "TG": (False, True), "K3": (True, False)}

_RE_STAMP = re.compile(r"^(\d+(?:\.\d+)?) ")
_RE_ITER = re.compile(r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, "
                      r"(\d+) generation tokens, iteration elapsed time: ([\d.]+) ms( \(dummy\))?, "
                      r"\w+ KV cache usage: ([\d.]+)%")


class DecompError(ValueError):
    pass


class CounterReset(DecompError):
    pass


# ---------------------------------------------------------------- server iteration log

def parse_timestamped_iteration_line(line: str) -> dict | None:
    """One "<epoch_ms> <vLLM log line>" -> step dict (t in epoch seconds, u in percent); None otherwise."""
    stamp = _RE_STAMP.match(line)
    m = _RE_ITER.search(line) if stamp else None
    if not m or m.group(7):
        return None
    return {"t": float(stamp.group(1)) / 1000, "index": int(m.group(1)), "ctx_req": int(m.group(2)),
            "ctx_tok": int(m.group(3)), "gen_req": int(m.group(4)), "gen_tok": int(m.group(5)),
            "elapsed_ms": float(m.group(6)), "u": float(m.group(8))}


def parse_iteration_log(lines) -> list:
    steps = [s for s in (parse_timestamped_iteration_line(x) for x in lines) if s]
    return sorted(steps, key=lambda s: s["index"])


def steps_in_window(steps: list, t0: float, t1: float) -> list:
    return [s for s in steps if t0 < s["t"] <= t1]


# ---------------------------------------------------------------- block arithmetic

def _free(frac: float, nb: int) -> int:
    return bp.round_half_up((1 - frac) * (nb - 1))


def free_blocks(u_pct: float, nb: int) -> int:
    """block_pool.py:869-880: usage = 1 - free / (nb - 1); u as the log prints it (percent, 0.1)."""
    return _free(u_pct / 100, nb)


def head_demand(tokens: int, *, gate: bool = True, threshold: int = 0, budget_left: int = BUDGET,
                block_size: int = BLOCK_SIZE) -> int:
    """Blocks the waiting head needs to be admitted: the full prompt with scheduler_reserve_full_isl
    (kv_cache_manager.py:508-524), else its first chunk (min of prompt, budget left, threshold)."""
    n = tokens if gate else min(tokens, budget_left, threshold or tokens)
    return math.ceil(n / block_size) + GDN_BLOCKS


def condition_knobs(cond: str, threshold: int = LONG_PREFILL_TOKEN_THRESHOLD) -> dict:
    if cond not in _KNOBS:
        raise ValueError(f"unknown condition {cond!r} (024: {sorted(_KNOBS)})")
    gate, chunked = _KNOBS[cond]
    return {"gate": gate, "threshold": threshold if chunked else 0}


# ---------------------------------------------------------------- OTLP spans

def _otlp_decoder():
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    def decode(body: bytes):
        msg = ExportTraceServiceRequest()
        msg.ParseFromString(body)
        return msg

    return decode


def _any_value(v):
    kind = v.WhichOneof("value")
    if kind is None:
        return None
    if kind == "array_value":
        return [_any_value(x) for x in v.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: _any_value(kv.value) for kv in v.kvlist_value.values}
    if kind == "bytes_value":
        return bytes(v.bytes_value).hex()
    return getattr(v, kind)


def decode_otlp_traces(body: bytes, gzip: bool, decoder=None) -> list:
    """An OTLP/HTTP protobuf ExportTraceServiceRequest body -> one flat dict per span.
    decoder(bytes) -> message; default: opentelemetry.proto (WSL venv only), imported here."""
    if gzip:
        body = _gzip.decompress(body)
    msg = (decoder or _otlp_decoder())(body)
    out = []
    for rs in msg.resource_spans:
        for ss in rs.scope_spans:
            for sp in ss.spans:
                out.append({"name": sp.name, "trace_id": bytes(sp.trace_id).hex(), "span_id": bytes(sp.span_id).hex(),
                            "start_time_unix_nano": int(sp.start_time_unix_nano),
                            "end_time_unix_nano": int(sp.end_time_unix_nano),
                            "attributes": {kv.key: _any_value(kv.value) for kv in sp.attributes}})
    return out


def span_record(span: dict) -> dict:
    """A decoded llm_request span -> per-request record (seconds; arrival = span start, epoch)."""
    attrs = span.get("attributes") or {}
    rec = {k: attrs.get(name) for k, name in SPAN_ATTRS.items()}
    rec["arrival"] = span["start_time_unix_nano"] / 1e9
    return rec


def llm_request_records(spans) -> list:
    """Decoded spans (only llm_request kept) or already mapped records (passed through)."""
    out = []
    for sp in spans:
        if "attributes" in sp:
            if sp.get("name") == LLM_REQUEST:
                out.append(span_record(sp))
        elif "id" in sp:
            out.append(sp)
    return out


# ---------------------------------------------------------------- alignment (A8)

def request_id(prefix: str, i: int) -> str:
    return f"cmpl-{prefix}{i}-0"


def request_class(tokens, L: int) -> str | None:
    if tokens == L - bp.N_GEN:
        return "short"
    if tokens == bp.LONG_INPUT:
        return "long"
    return None


def _times(rec: dict) -> dict:
    if any(rec.get(k) is None for k in ("arrival", "server_ttft", "P", "Q")):
        return {"f": None, "s": None, "e": None}
    f = rec["arrival"] + rec["server_ttft"]
    s = f - rec["P"]
    return {"f": f, "s": s, "e": s - rec["Q"]}


def align(records: list, detailed: dict, *, prefix: str, L: int) -> dict:
    """Spans <-> --save-detailed index by id; spans without the prefix (warmup, ready-check) -> "other"."""
    n = len(detailed["input_lens"])
    outs = detailed.get("output_lens") or [None] * n
    pat = re.compile("^cmpl-" + re.escape(prefix) + r"(\d+)-0$")
    by_idx, other, dup, unexpected = {}, [], [], []
    for rec in records:
        m = pat.match(rec.get("id") or "")
        if not m:
            other.append(rec)
        elif not 0 <= int(m.group(1)) < n:
            unexpected.append(rec["id"])
        elif int(m.group(1)) in by_idx:
            dup.append(rec["id"])
        else:
            by_idx[int(m.group(1))] = rec
    ttfts = detailed.get("ttfts") or []
    requests = []
    for i in range(n):
        sp = by_idx.get(i) or {}
        ttft = ttfts[i] if i < len(ttfts) else None
        r = {"idx": i, "id": request_id(prefix, i), "cls": request_class(detailed["input_lens"][i], L),
             "input_len": detailed["input_lens"][i], "output_len": outs[i], "client_ttft": ttft,
             "has_span": i in by_idx, **{k: sp.get(k) for k in ("arrival", "Q", "P", "server_ttft", "e2e", "prompt_tokens",
                                       "completion_tokens")}, **_times(sp)}
        has = r["s"] is not None and ttft is not None
        r["O"] = ttft - r["server_ttft"] if has else None
        r["F"] = r["server_ttft"] - r["Q"] - r["P"] if r["s"] is not None else None
        requests.append(r)
    return {"requests": requests, "other": other, "missing": [i for i in range(n) if i not in by_idx],
            "duplicates": dup, "unexpected": unexpected}


def _ms(x):
    return None if x is None else x * 1000


def derived_requests(aligned: dict) -> list:
    """derived/<tag>-requests.jsonl rows: numbers only (024 量測)."""
    return [{"idx": r["idx"], "cls": r["cls"], "client_ttft_ms": _ms(r["client_ttft"]), "arrival": r["arrival"],
             "Q_ms": _ms(r["Q"]), "P_ms": _ms(r["P"]), "server_ttft_ms": _ms(r["server_ttft"]), "O_ms": _ms(r["O"]),
             "F_ms": _ms(r["F"]), "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"]}
            for r in aligned["requests"]]


def request_timeline(records: list) -> list:
    """Every span (main, warmup, ready-check) with e / s / f, for W_j and the long prefill intervals."""
    out = []
    for rec in records:
        t = _times(rec)
        if t["s"] is not None and rec.get("prompt_tokens") is not None:
            out.append({"id": rec.get("id"), "tokens": rec["prompt_tokens"], **t})
    return out


# ---------------------------------------------------------------- A9 step classifier

def _heads(steps: list, waits: list, lag: int) -> dict:
    """j -> head of W_j (smallest e among e <= tau_j < s), swept in tau order."""
    heads, heap, p = {}, [], 0
    for j in sorted(range(lag, len(steps)), key=lambda j: steps[j - lag]["t"]):
        tau = steps[j - lag]["t"]
        while p < len(waits) and waits[p]["e"] <= tau:
            heapq.heappush(heap, (waits[p]["e"], p))
            p += 1
        while heap and waits[heap[0][1]]["s"] <= tau:
            heapq.heappop(heap)
        heads[j] = waits[heap[0][1]] if heap else None
    return heads


def classify_steps(steps: list, timeline: list, *, nb: int, gate: bool = True, threshold: int = 0,
                   lag: int = LAG_MAIN, budget: int = BUDGET, block_size: int = BLOCK_SIZE) -> list:
    """One record per step j >= 1 (I_j = (t_{j-1}, t_j]); cat None when tau_j = t_{j-lag} does not exist."""
    steps = sorted(steps, key=lambda s: s["index"])
    waits = sorted((r for r in timeline if r["s"] > r["e"]), key=lambda r: r["e"])
    longs = [r for r in timeline if r["tokens"] == bp.LONG_INPUT]
    heads = _heads(steps, waits, lag)
    out = []
    for j in range(1, len(steps)):
        st = steps[j]
        n_tok = st["ctx_tok"] + st["gen_tok"]
        rec = {"index": st["index"], "t0": steps[j - 1]["t"], "t1": st["t"], "tau": None, "cat": None,
               "n_tokens": n_tok, "free": None, "demand": None, "head_long": None, "n_long": None,
               "max_seqs": st["ctx_req"] + st["gen_req"] >= bp.MAX_NUM_SEQS}
        if j >= lag:
            tau = steps[j - lag]["t"]
            head = heads[j]
            rec.update(tau=tau, free=free_blocks(steps[j - lag]["u"], nb),
                       n_long=sum(1 for r in longs if r["s"] <= tau <= r["f"]))
            if head is None:
                rec["cat"] = "no-queue"
            elif n_tok >= budget:
                rec["cat"] = "budget-long" if rec["n_long"] else "budget-other"
            else:
                rec["head_long"] = head["tokens"] == bp.LONG_INPUT
                rec["demand"] = head_demand(head["tokens"], gate=gate, threshold=threshold,
                                            budget_left=budget - n_tok, block_size=block_size)
                if rec["free"] < rec["demand"]:
                    rec["cat"] = "HOL-long" if rec["head_long"] else "KV-short"
                else:
                    rec["cat"] = "unexplained"
        out.append(rec)
    return out


def attribute_queue(req: dict, classified: list) -> dict:
    """[e, s] of one request split over the step intervals by overlap (seconds); the rest is unclassified."""
    e, s = req["e"], req["s"]
    got = dict.fromkeys(SHARE_KEYS, 0.0)
    for c in classified:
        overlap = min(s, c["t1"]) - max(e, c["t0"])
        if overlap > 0 and c["cat"] is not None:
            got[c["cat"]] += overlap
    got["unclassified"] = (s - e) - sum(got[k] for k in CATEGORIES)
    return got


def q_shares(tail: list, classified: list) -> dict:
    """Fractions of the tail's total Q per category (all 0.0 when the tail never queued)."""
    total = dict.fromkeys(SHARE_KEYS, 0.0)
    for r in tail:
        for k, v in attribute_queue(r, classified).items():
            total[k] += v
    q = sum(r["s"] - r["e"] for r in tail)
    return {k: (v / q if q > 0 else 0.0) for k, v in total.items()}


def tail_set(requests: list, q: float = 90) -> list:
    """Short requests with client TTFT >= that run's short-class q-th percentile (linear, ties included)."""
    short = [r for r in requests if r["cls"] == "short" and r["client_ttft"] is not None]
    if not short:
        return []
    cut = ba.percentile([r["client_ttft"] for r in short], q)
    return [r for r in short if r["client_ttft"] >= cut]


def step_shares(classified: list, *, t_start: float, t_end: float) -> dict:
    sel = [c for c in classified if c["cat"] is not None and t_start <= c["t1"] <= t_end]
    queued = [c for c in sel if c["cat"] != "no-queue"]
    return {"residual_bm_share": sum(c["n_long"] >= RESIDUAL_BM_LONGS for c in sel) / len(sel) if sel else 0.0,
            "unexplained_break_share":
                sum(c["cat"] == "unexplained" for c in queued) / len(queued) if queued else 0.0,
            "max_seqs_share": sum(c["max_seqs"] for c in sel) / len(sel) if sel else 0.0,
            "n_steps": len(sel), "n_queued_steps": len(queued)}


# ---------------------------------------------------------------- Prometheus deltas

def _metrics(x) -> dict:
    if isinstance(x, dict):
        return x
    from run_core_b import parse_prometheus  # lazily: run_core_b imports tomllib (3.11+)

    return parse_prometheus(x or "")


def _total(m: dict, name: str) -> float:
    if not m.get(name):
        raise DecompError(f"{name} missing from /metrics")
    return sum(v for _, v in m[name])


def counter_delta(before, after, name: str) -> float:
    b, a = _total(_metrics(before), name), _total(_metrics(after), name)
    if a < b:
        raise CounterReset(f"{name} went from {b} to {a} (counter reset)")
    return a - b


def histogram_delta(before, after, name: str) -> dict:
    """{"buckets": {le: delta}, "sum", "count"}, label sets summed per le."""
    b, a = _metrics(before), _metrics(after)

    def buckets(m):
        rows = m.get(f"{name}_bucket")
        if not rows:
            raise DecompError(f"{name}_bucket missing from /metrics")
        out: dict = {}
        for labels, v in rows:
            out[labels.get("le")] = out.get(labels.get("le"), 0.0) + v
        return out

    bb, ab = buckets(b), buckets(a)
    if set(bb) != set(ab):
        raise DecompError(f"{name} bucket bounds differ between scrapes")
    delta = {"buckets": {le: ab[le] - bb[le] for le in ab},
             "sum": _total(a, f"{name}_sum") - _total(b, f"{name}_sum"),
             "count": _total(a, f"{name}_count") - _total(b, f"{name}_count")}
    if delta["count"] < 0 or any(v < 0 for v in delta["buckets"].values()):
        raise CounterReset(f"{name} decreased between scrapes (counter reset)")
    return delta


# ---------------------------------------------------------------- 1 Hz shares (A13)

def one_hz_shares(samples: list, *, nb: int, L: int, block_size: int = BLOCK_SIZE) -> dict:
    """KV-full-while-waiting (waiting >= 1, free < the short request's blocks: 7 at 2K, 19 at 8K) and
    gate-zone (waiting >= 1, free < 65) shares of the 1 Hz samples (kv_cache_usage_perc is 0..1)."""
    short_need = head_demand(L - bp.N_GEN, block_size=block_size)
    long_need = head_demand(bp.LONG_INPUT, block_size=block_size)
    rows = [(_free(s["vllm:kv_cache_usage_perc"], nb), s["vllm:num_requests_waiting"]) for s in samples
            if s.get("vllm:kv_cache_usage_perc") is not None and s.get("vllm:num_requests_waiting") is not None]
    if not rows:
        return {"kv_full_while_waiting": None, "gate_zone": None, "n_samples": 0}
    return {"kv_full_while_waiting": sum(w >= 1 and f < short_need for f, w in rows) / len(rows),
            "gate_zone": sum(w >= 1 and f < long_need for f, w in rows) / len(rows), "n_samples": len(rows)}


# ---------------------------------------------------------------- A10 validity and the run summary

def _check_a(al: dict, n: int) -> tuple:
    why = []
    if al["missing"] or al["duplicates"] or al["unexpected"]:
        why.append(f"(a) spans: {len(al['missing'])} missing, {len(al['duplicates'])} duplicate, "
                   f"{len(al['unexpected'])} out-of-range of N = {n}")
    bad = [r["idx"] for r in al["requests"] if r["has_span"]
           and (r["prompt_tokens"] != r["input_len"] or r["completion_tokens"] != r["output_len"])]
    if bad:
        why.append(f"(a) prompt / completion tokens differ from the client at {len(bad)} ids (first {bad[0]})")
    return not why, why


def _check_b(records: list, before, after) -> tuple:
    try:
        d = histogram_delta(before, after, QUEUE_HIST)
    except DecompError as exc:
        return False, [f"(b) {exc}"]
    qs = [r["Q"] for r in records if r.get("Q") is not None]
    why = []
    if d["count"] != len(records) or len(qs) != len(records):
        why.append(f"(b) queue-time histogram delta count {d['count']:g} != {len(records)} spans received")
    if abs(d["sum"] - sum(qs)) > QUEUE_TOL_S:
        why.append(f"(b) |delta sum {d['sum']:.6f} - sum Q {sum(qs):.6f}| > 1 ms")
    return not why, why


def _check_c(window_steps: list, before, after) -> tuple:
    try:
        pre = counter_delta(before, after, PREEMPTIONS)
        prompt = counter_delta(before, after, PROMPT_TOKENS)
    except DecompError as exc:
        return False, [f"(c) {exc}"]
    if pre > 0:
        return None, [f"(c) exempt: {pre:g} preemptions in the window"]
    ctx = sum(s["ctx_tok"] for s in window_steps)
    if ctx != prompt:
        return False, [f"(c) sum ctx_tokens {ctx} != delta prompt_tokens {prompt:g}"]
    return True, []


def _check_d(window_steps: list) -> tuple:
    idx = sorted(s["index"] for s in window_steps)
    if not idx:
        return False, ["(d) no iteration line in the window"]
    if any(b - a != 1 for a, b in zip(idx, idx[1:])):
        return False, [f"(d) iteration indices {idx[0]}..{idx[-1]} not consecutive ({len(idx)} lines)"]
    return True, []


def _means_ms(reqs: list) -> dict | None:
    have = [r for r in reqs if r["s"] is not None and r["O"] is not None]
    if not have:
        return None
    return {k: sum(r[k] for r in have) / len(have) * 1000 for k in ("O", "Q", "P", "F")}


def summarize_run(*, spans: list, detailed: dict | None, steps: list, prefix: str, L: int, cond: str,
                  num_gpu_blocks: int, metrics_before, metrics_after, window: tuple,
                  block_size: int = BLOCK_SIZE, budget: int = BUDGET,
                  long_prefill_token_threshold: int = LONG_PREFILL_TOKEN_THRESHOLD) -> dict | None:
    """row["decomp"] of one run (024 contract), or None when the bench produced no result.
    spans: everything the receiver got for this boot; window: (t0, t1) wall times of the two scrapes."""
    knobs = condition_knobs(cond, long_prefill_token_threshold)
    if not detailed or not detailed.get("ttfts") or not detailed.get("input_lens"):
        return None
    t0, t1 = window
    records = [r for r in llm_request_records(spans) if r.get("arrival") is not None and t0 <= r["arrival"] <= t1]
    al = align(records, detailed, prefix=prefix, L=L)
    win = steps_in_window(steps, t0, t1)
    (a, why_a), (b, why_b), (c, why_c), (d, why_d) = (_check_a(al, len(detailed["input_lens"])),
                                                      _check_b(records, metrics_before, metrics_after),
                                                      _check_c(win, metrics_before, metrics_after), _check_d(win))
    timeline = request_timeline(records)
    kw = dict(nb=num_gpu_blocks, gate=knobs["gate"], threshold=knobs["threshold"], budget=budget,
              block_size=block_size)
    main_cl = classify_steps(steps, timeline, lag=LAG_MAIN, **kw)
    sens_cl = classify_steps(steps, timeline, lag=LAG_SENS, **kw)
    tail, tail95 = tail_set(al["requests"], 90), tail_set(al["requests"], 95)
    timed = [r for r in tail if r["s"] is not None]
    done = [r for r in al["requests"] if r["arrival"] is not None]
    ends = [r["arrival"] + (r["e2e"] or r["server_ttft"] or 0.0) for r in done]
    shares = step_shares(main_cl, t_start=min(r["arrival"] for r in done), t_end=max(ends)) if done else \
        step_shares([], t_start=0.0, t_end=0.0)
    return {"valid": bool(a and b and d and c is not False), "checks": {"a": a, "b": b, "c": c, "d": d},
            "reasons": why_a + why_b + why_c + why_d,
            "tail": _means_ms(tail), "n_tail": len(tail), "tail95": _means_ms(tail95), "n_tail95": len(tail95),
            "q_shares": q_shares(timed, main_cl) if timed else None,
            "q_shares_sens": q_shares(timed, sens_cl) if timed else None,
            "residual_bm_share": shares["residual_bm_share"],
            "unexplained_break_share": shares["unexplained_break_share"],
            "max_seqs_share": shares["max_seqs_share"], "n_steps": shares["n_steps"],
            "n_queued_steps": shares["n_queued_steps"]}
