"""Tests for benchmarks/adv_decomp.py: decision 024 measurement / decomposition rules (A8-A10, A13),
written before any Advanced data. Every expected share below is computed by hand from a synthetic
timeline (steps one second apart so the overlaps are easy to read)."""

import gzip
import math
from types import SimpleNamespace

import pytest

import adv_decomp as ad

PREFIX = "adv-B0-2048-uniform-r1-"
LONG = 32640


# ---------------------------------------------------------------- iteration lines (loggers.py:185-201)

def iter_line(j, *, t, ctx_req=1, ctx_tok=2048, gen_req=0, gen_tok=0, u=50.1, dummy=False, extra=""):
    return (f"{t * 1000:.0f} (APIServer pid=3428) INFO 10-10 12:00:00 [loggers.py:185] Engine 000: Iteration({j}): "
            f"{ctx_req} context requests, {ctx_tok} context tokens, {gen_req} generation requests, "
            f"{gen_tok} generation tokens, iteration elapsed time: 12.34 ms{' (dummy)' if dummy else ''}, "
            f"GPU KV cache usage: {u:.1f}%{extra}")


def test_parse_timestamped_iteration_line():
    line = ("1760090000123 (APIServer pid=3428) INFO 10-10 12:00:00 [loggers.py:185] Engine 000: Iteration(41): "
            "2 context requests, 2048 context tokens, 7 generation requests, 7 generation tokens, "
            "iteration elapsed time: 45.12 ms, GPU KV cache usage: 56.3%")
    assert ad.parse_timestamped_iteration_line(line) == {
        "t": pytest.approx(1760090000.123, abs=1e-6), "index": 41, "ctx_req": 2, "ctx_tok": 2048, "gen_req": 7,
        "gen_tok": 7, "elapsed_ms": 45.12, "u": 56.3}
    frac = line.replace("1760090000123 ", "1760090000123.5 ", 1)
    assert ad.parse_timestamped_iteration_line(frac)["t"] == pytest.approx(1760090000.1235, abs=1e-6)
    enc = line + ", encoder inputs: 0, encoder output embeddings: 0"
    assert ad.parse_timestamped_iteration_line(enc)["u"] == 56.3


def test_iteration_parser_ignores_dummy_other_and_unstamped_lines():
    dummy = iter_line(3, t=1000.0, ctx_req=0, ctx_tok=0, dummy=True)
    assert ad.parse_timestamped_iteration_line(dummy) is None
    other = "1760090000123 (EngineCore pid=2) INFO [cuda.py:478] Using AttentionBackendEnum.FLASH_ATTN backend."
    assert ad.parse_timestamped_iteration_line(other) is None
    unstamped = iter_line(3, t=1000.0).split(" ", 1)[1]
    assert ad.parse_timestamped_iteration_line(unstamped) is None
    lines = [iter_line(2, t=1001.0), other, dummy, iter_line(1, t=1000.0), "", "garbage"]
    steps = ad.parse_iteration_log(lines)
    assert [s["index"] for s in steps] == [1, 2] and steps[0]["t"] == pytest.approx(1000.0)


def test_steps_in_window_is_left_open_right_closed():
    steps = ad.parse_iteration_log([iter_line(j, t=1000.0 + j) for j in range(5)])
    assert [s["index"] for s in ad.steps_in_window(steps, 1001.0, 1003.0)] == [2, 3]


# ---------------------------------------------------------------- block arithmetic (block_pool.py:869-880)

def test_free_blocks_inverts_usage():
    assert ad.free_blocks(56.3, 248) == 108  # (1 - 0.563) x 247 = 107.94
    assert ad.free_blocks(0.0, 248) == 247 and ad.free_blocks(100.0, 248) == 0
    assert ad.free_blocks(99.6, 248) == 1  # 0.988
    assert ad.free_blocks(80.0, 248) == 49  # 49.4
    assert ad.free_blocks(98.0, 186) == 4  # K3: 0.02 x 185 = 3.7


def test_head_demand_gate_on_and_off():
    assert ad.head_demand(LONG) == 65  # 62 attention + 3 GDN (kv_cache_manager.py:508-524)
    assert ad.head_demand(1920) == 7 and ad.head_demand(8064) == 19
    assert ad.head_demand(LONG, gate=False, budget_left=2048) == 7  # first chunk 2048 -> 4 + 3
    assert ad.head_demand(LONG, gate=False, budget_left=100) == 4
    assert ad.head_demand(LONG, gate=False, threshold=512, budget_left=2048) == 4  # T = 512 -> 1 + 3
    assert ad.head_demand(1920, gate=False, budget_left=2048) == 7
    assert ad.head_demand(LONG, threshold=512) == 65  # T alone keeps the full-ISL check


def test_condition_knobs():
    assert ad.condition_knobs("B0") == {"gate": True, "threshold": 0}
    assert ad.condition_knobs("T") == {"gate": True, "threshold": 512}
    assert ad.condition_knobs("G") == {"gate": False, "threshold": 0}
    assert ad.condition_knobs("TG") == {"gate": False, "threshold": 512}
    assert ad.condition_knobs("K3") == {"gate": True, "threshold": 0}
    with pytest.raises(ValueError):
        ad.condition_knobs("X")


# ---------------------------------------------------------------- OTLP spans (output_processor.py:782-842)

class FakeAny:
    def __init__(self, **kw):
        (self.kind, value), = kw.items()
        setattr(self, self.kind, value)

    def WhichOneof(self, group):
        assert group == "value"
        return self.kind


def kv(key, **value):
    return SimpleNamespace(key=key, value=FakeAny(**value))


def fake_message():
    llm = SimpleNamespace(name="llm_request", trace_id=b"\x01" * 16, span_id=b"\x02" * 8,
                          start_time_unix_nano=1_000_500_000_000, end_time_unix_nano=1_003_000_000_000,
                          attributes=[kv("gen_ai.request.id", string_value=f"cmpl-{PREFIX}0-0"),
                                      kv("gen_ai.usage.prompt_tokens", int_value=1920),
                                      kv("gen_ai.usage.completion_tokens", int_value=128),
                                      kv("gen_ai.latency.time_in_queue", double_value=0.25),
                                      kv("gen_ai.latency.time_in_model_prefill", double_value=0.5),
                                      kv("gen_ai.latency.time_to_first_token", double_value=0.76),
                                      kv("gen_ai.latency.e2e", double_value=2.5),
                                      kv("gen_ai.latency.time_in_model_decode", double_value=1.7),
                                      kv("gen_ai.latency.time_in_model_inference", double_value=2.2),
                                      kv("gen_ai.request.temperature", double_value=0.0),
                                      kv("flags", array_value=SimpleNamespace(values=[FakeAny(bool_value=True)]))])
    other = SimpleNamespace(name="load_model", trace_id=b"\x03" * 16, span_id=b"\x04" * 8,
                            start_time_unix_nano=5, end_time_unix_nano=6, attributes=[kv("x", int_value=1)])
    return SimpleNamespace(resource_spans=[SimpleNamespace(scope_spans=[SimpleNamespace(spans=[llm]),
                                                                        SimpleNamespace(spans=[other])])])


def test_decode_otlp_traces_with_injected_decoder_and_gzip():
    seen = []

    def decoder(body):
        seen.append(body)
        return fake_message()

    spans = ad.decode_otlp_traces(gzip.compress(b"payload"), True, decoder=decoder)
    assert seen == [b"payload"]
    assert [s["name"] for s in spans] == ["llm_request", "load_model"]
    s = spans[0]
    assert s["trace_id"] == "01" * 16 and s["span_id"] == "02" * 8
    assert s["start_time_unix_nano"] == 1_000_500_000_000
    assert s["attributes"]["gen_ai.usage.prompt_tokens"] == 1920 and s["attributes"]["flags"] == [True]
    ad.decode_otlp_traces(b"raw", False, decoder=decoder)
    assert seen[-1] == b"raw"


def test_llm_request_records_map_vllm_attribute_names():
    spans = ad.decode_otlp_traces(b"x", False, decoder=lambda b: fake_message())
    recs = ad.llm_request_records(spans)
    assert len(recs) == 1
    r = recs[0]
    assert r["id"] == f"cmpl-{PREFIX}0-0" and r["prompt_tokens"] == 1920 and r["completion_tokens"] == 128
    assert (r["Q"], r["P"], r["server_ttft"], r["e2e"]) == (0.25, 0.5, 0.76, 2.5)
    assert r["arrival"] == pytest.approx(1000.5)
    assert ad.llm_request_records(recs) == recs  # already-mapped records pass through unchanged


def test_decode_otlp_roundtrip_with_real_protobuf():
    pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    req = ExportTraceServiceRequest()
    span = req.resource_spans.add().scope_spans.add().spans.add()
    span.name, span.start_time_unix_nano = "llm_request", 1_000_000_000_000
    a = span.attributes.add()
    a.key, a.value.string_value = "gen_ai.request.id", f"cmpl-{PREFIX}3-0"
    b = span.attributes.add()
    b.key, b.value.double_value = "gen_ai.latency.time_in_queue", 0.125
    rec, = ad.llm_request_records(ad.decode_otlp_traces(gzip.compress(req.SerializeToString()), True))
    assert rec["id"] == f"cmpl-{PREFIX}3-0" and rec["Q"] == 0.125 and rec["arrival"] == 1000.0


# ---------------------------------------------------------------- alignment (A8)

def rec(i_or_id, *, arrival, Q, P, F=0.01, prompt=1920, completion=128, e2e=None):
    """A mapped llm_request record; server TTFT = F + Q + P (F = arrival -> queued, frontend + IPC)."""
    rid = i_or_id if isinstance(i_or_id, str) else ad.request_id(PREFIX, i_or_id)
    ttft = F + Q + P
    return {"id": rid, "arrival": arrival, "Q": Q, "P": P, "server_ttft": ttft,
            "e2e": e2e if e2e is not None else ttft + 1.0, "prompt_tokens": prompt, "completion_tokens": completion}


def test_request_id_format():
    assert ad.request_id(PREFIX, 7) == "cmpl-adv-B0-2048-uniform-r1-7-0"


def test_align_by_id_keeps_warmup_apart_and_reports_gaps():
    detailed = {"input_lens": [1920, LONG, 1920], "output_lens": [128] * 3, "ttfts": [0.5, 2.0, 0.9]}
    recs = [rec(2, arrival=1002.0, Q=0.3, P=0.4), rec("cmpl-0a1b2c3d-0", arrival=999.0, Q=0.0, P=0.1),
            rec(0, arrival=1000.0, Q=0.1, P=0.3, F=0.0), rec(1, arrival=1001.0, Q=0.5, P=1.2, prompt=LONG),
            rec("cmpl-9f8e7d6c-0", arrival=999.1, Q=0.0, P=0.1)]
    al = ad.align(recs, detailed, prefix=PREFIX, L=2048)
    assert [r["idx"] for r in al["requests"]] == [0, 1, 2]
    assert [r["cls"] for r in al["requests"]] == ["short", "long", "short"]
    assert [r["id"] for r in al["other"]] == ["cmpl-0a1b2c3d-0", "cmpl-9f8e7d6c-0"]
    assert al["missing"] == [] and al["duplicates"] == [] and al["unexpected"] == []
    r0, r2 = al["requests"][0], al["requests"][2]
    assert r0["O"] == pytest.approx(0.5 - 0.4) and r0["F"] == pytest.approx(0.0)
    assert r0["O"] + r0["Q"] + r0["P"] == pytest.approx(r0["client_ttft"])  # F = 0: O + Q + P = client TTFT
    assert r2["O"] == pytest.approx(0.9 - 0.71) and r2["F"] == pytest.approx(0.01)
    assert r2["O"] + r2["F"] + r2["Q"] + r2["P"] == pytest.approx(r2["client_ttft"])
    assert r2["f"] == pytest.approx(1002.71) and r2["s"] == pytest.approx(1002.31) and r2["e"] == pytest.approx(1002.01)

    gaps = ad.align([recs[0], recs[0], rec(5, arrival=1.0, Q=0, P=0)], detailed, prefix=PREFIX, L=2048)
    assert gaps["missing"] == [0, 1] and gaps["duplicates"] == [ad.request_id(PREFIX, 2)]
    assert gaps["unexpected"] == [ad.request_id(PREFIX, 5)]
    assert gaps["requests"][0]["Q"] is None and gaps["requests"][0]["client_ttft"] == 0.5
    other_run = ad.align([rec(0, arrival=1.0, Q=0, P=0)], detailed, prefix="adv-T-2048-uniform-r1-", L=2048)
    assert other_run["missing"] == [0, 1, 2] and len(other_run["other"]) == 1


def test_derived_requests_are_numbers_in_ms():
    detailed = {"input_lens": [1920, 1920], "output_lens": [128, 128], "ttfts": [0.5, 0.9]}
    al = ad.align([rec(0, arrival=1000.0, Q=0.1, P=0.3)], detailed, prefix=PREFIX, L=2048)
    rows = ad.derived_requests(al)
    assert rows[0] == {"idx": 0, "cls": "short", "client_ttft_ms": pytest.approx(500.0), "arrival": 1000.0,
                       "Q_ms": pytest.approx(100.0), "P_ms": pytest.approx(300.0),
                       "server_ttft_ms": pytest.approx(410.0), "O_ms": pytest.approx(90.0),
                       "F_ms": pytest.approx(10.0), "prompt_tokens": 1920, "completion_tokens": 128}
    assert rows[1]["Q_ms"] is None and rows[1]["client_ttft_ms"] == pytest.approx(900.0)


# ---------------------------------------------------------------- A9 step classifier

def steps_at(specs):
    """specs: per step j (t = 1000 + j) a dict of overrides for ctx_tok / gen_tok / u / ctx_req / gen_req."""
    return ad.parse_iteration_log([iter_line(j, t=1000.0 + j, **sp) for j, sp in enumerate(specs)])


def tl(e, s, tokens=1920, f=None, rid="r"):
    return {"id": rid, "e": e, "s": s, "f": s + 0.5 if f is None else f, "tokens": tokens}


def cats(classified):
    return [c["cat"] for c in classified]


def test_classifier_no_queue_and_unclassifiable_first_steps():
    cl = ad.classify_steps(steps_at([{}] * 4), [], nb=248)
    assert [c["index"] for c in cl] == [1, 2, 3]  # step 0 has no interval (t_{-1})
    assert cats(cl) == [None, "no-queue", "no-queue"]  # step 1 has no tau = t_{-1} at lag 2
    assert cl[1]["t0"] == pytest.approx(1001.0) and cl[1]["t1"] == pytest.approx(1002.0)
    assert cl[1]["tau"] == pytest.approx(1000.0)


def test_classifier_budget_long_and_budget_other():
    short = tl(999.5, 1010.0)
    long_prefilling = tl(999.0, 999.2, tokens=LONG, f=1010.0)
    cl = ad.classify_steps(steps_at([{}] * 4), [short, long_prefilling], nb=248)
    assert cats(cl) == [None, "budget-long", "budget-long"] and cl[1]["n_long"] == 1
    cl = ad.classify_steps(steps_at([{}] * 4), [short], nb=248)
    assert cats(cl) == [None, "budget-other", "budget-other"]
    cl = ad.classify_steps(steps_at([{"ctx_tok": 2000, "gen_tok": 48}] * 4), [short], nb=248)
    assert cats(cl)[1:] == ["budget-other", "budget-other"]  # N_j = ctx + gen = 2048


def test_classifier_hol_long_kv_short_unexplained():
    head_long, later_short = tl(999.0, 1010.0, tokens=LONG, rid="L"), tl(999.5, 1010.0, rid="S")
    hol = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 80.0}] * 4), [later_short, head_long], nb=248)
    assert cats(hol)[1:] == ["HOL-long", "HOL-long"]  # free 49 < 65
    assert hol[1]["free"] == 49 and hol[1]["demand"] == 65 and hol[1]["head_long"] is True
    head_short = tl(999.0, 1010.0)
    kv = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 98.0}] * 4), [head_short], nb=248)
    assert cats(kv)[1:] == ["KV-short", "KV-short"] and kv[1]["free"] == 5 and kv[1]["demand"] == 7  # 4.94
    un = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 50.1}] * 4), [head_short], nb=248)
    assert cats(un)[1:] == ["unexplained", "unexplained"] and un[1]["free"] == 123  # 123.25 >= 7


def test_classifier_8k_short_demand_19():
    head = tl(999.0, 1010.0, tokens=8064)
    blocked = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 93.0}] * 4), [head], nb=248)  # 0.07 x 247 = 17.29
    assert cats(blocked)[1:] == ["KV-short", "KV-short"] and blocked[1]["demand"] == 19
    fits = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 92.0}] * 4), [head], nb=248)  # 19.76 -> 20 >= 19
    assert cats(fits)[1:] == ["unexplained", "unexplained"]


def test_classifier_gate_off_uses_first_chunk_demand():
    head_long = tl(999.0, 1010.0, tokens=LONG)
    # N_j = 1000 -> budget left 1048 -> chunk 1048 -> ceil(1048 / 528) + 3 = 5
    fits = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 98.0}] * 4), [head_long], nb=248, gate=False)
    assert cats(fits)[1:] == ["unexplained", "unexplained"] and fits[1]["demand"] == 5  # free 5
    blocked = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 98.5}] * 4), [head_long], nb=248, gate=False)
    assert cats(blocked)[1:] == ["HOL-long", "HOL-long"]  # 3.705 -> 4 < 5
    tg = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 98.5}] * 4), [head_long], nb=248, gate=False,
                           threshold=512)
    assert tg[1]["demand"] == 4 and cats(tg)[1:] == ["unexplained", "unexplained"]


def test_classifier_head_is_earliest_enqueued_and_warmup_counts():
    head_short, later_long = tl(999.0, 1010.0, rid="S"), tl(999.5, 1010.0, tokens=LONG, rid="L")
    cl = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 98.0}] * 4), [later_long, head_short], nb=248)
    assert cats(cl)[1:] == ["KV-short", "KV-short"]  # head = smallest e
    # a request that is no longer waiting (s <= tau) is not in W_j
    gone = ad.classify_steps(steps_at([{"ctx_tok": 1000, "u": 98.0}] * 4), [tl(999.0, 1000.0)], nb=248)
    assert cats(gone)[1:] == ["no-queue", "no-queue"]
    # e == s never waits (Q = 0)
    assert cats(ad.classify_steps(steps_at([{}] * 4), [tl(1000.5, 1000.5)], nb=248))[1:] == ["no-queue"] * 2
    # warmup / ready-check spans are part of the timeline (A9 "含 warmup")
    recs = [rec("cmpl-warm-0", arrival=999.49, Q=10.0, P=0.5)]
    timeline = ad.request_timeline(recs)
    assert timeline[0]["e"] == pytest.approx(999.5) and timeline[0]["s"] == pytest.approx(1009.5)
    assert cats(ad.classify_steps(steps_at([{}] * 4), timeline, nb=248))[1:] == ["budget-other"] * 2


def test_classifier_max_seqs_is_only_a_flag():
    cl = ad.classify_steps(steps_at([{"ctx_req": 100, "gen_req": 28}, {"ctx_req": 100, "gen_req": 28},
                                     {"ctx_req": 100, "gen_req": 28}, {"ctx_req": 100, "gen_req": 27}]),
                           [], nb=248)
    assert [c["max_seqs"] for c in cl] == [True, True, False]
    assert cats(cl) == [None, "no-queue", "no-queue"]


def test_lag_switch_tau_and_usage():
    # a short request queued from 1001.5; steps 1 s apart, N_j = 2048
    cl2 = ad.classify_steps(steps_at([{}] * 5), [tl(1001.5, 1010.0)], nb=248, lag=2)
    cl1 = ad.classify_steps(steps_at([{}] * 5), [tl(1001.5, 1010.0)], nb=248, lag=1)
    assert cats(cl2) == [None, "no-queue", "no-queue", "budget-other"]  # tau_3 = t_1 = 1001 < 1001.5
    assert cats(cl1) == ["no-queue", "no-queue", "budget-other", "budget-other"]  # tau_3 = t_2 = 1002
    # usage is read at the same lagged step: u_1 = 98.0 (free 5 < 7), u_2 = 50.1 (free 123)
    specs = [{"ctx_tok": 1000, "u": 10.0}, {"ctx_tok": 1000, "u": 98.0}, {"ctx_tok": 1000, "u": 50.1},
             {"ctx_tok": 1000, "u": 10.0}]
    head = [tl(999.0, 1010.0)]
    assert ad.classify_steps(steps_at(specs), head, nb=248, lag=2)[2]["cat"] == "KV-short"  # step 3 uses u_1
    assert ad.classify_steps(steps_at(specs), head, nb=248, lag=1)[2]["cat"] == "unexplained"  # step 3 uses u_2


# ---------------------------------------------------------------- queue attribution, shares, tail

def test_attribution_by_overlap_sums_to_q_minus_unclassified():
    a = tl(1001.5, 1004.25)
    cl = ad.classify_steps(steps_at([{}] * 7), [a], nb=248)
    # tau_j = t_{j-2}: steps 2, 3 see no queue (a enqueues at 1001.5); steps 4-6 see a waiting
    assert cats(cl) == [None, "no-queue", "no-queue", "budget-other", "budget-other", "budget-other"]
    got = ad.attribute_queue(a, cl)
    assert got["no-queue"] == pytest.approx(0.5 + 1.0) and got["budget-other"] == pytest.approx(1.0 + 0.25)
    assert got["unclassified"] == pytest.approx(0.0)
    assert sum(v for k, v in got.items() if k != "unclassified") == pytest.approx(2.75 - got["unclassified"])

    b = tl(999.5, 1001.5)
    cl = ad.classify_steps(steps_at([{}] * 7), [b], nb=248)
    got = ad.attribute_queue(b, cl)
    # 0.5 s before t_0 (no step) + 1.0 s in step 1 (no tau at lag 2) -> unclassified; (1001, 1001.5] in step 2,
    # tau_2 = 1000 sees b waiting -> budget-other
    assert got["unclassified"] == pytest.approx(1.5) and got["budget-other"] == pytest.approx(0.5)
    assert sum(v for k, v in got.items() if k != "unclassified") == pytest.approx(2.0 - 1.5)
    assert set(got) == set(ad.SHARE_KEYS)


def test_q_shares_over_the_tail():
    a, b = tl(1001.5, 1004.25, rid="a"), tl(1002.5, 1003.5, rid="b")
    cl = ad.classify_steps(steps_at([{}] * 7), [a, b], nb=248)
    assert cats(cl) == [None, "no-queue", "no-queue", "budget-other", "budget-other", "budget-other"]
    shares = ad.q_shares([a, b], cl)
    # a: no-queue 1.5, budget-other 1.25; b: no-queue 0.5 (1002-1003), budget-other 0.5 -> total 3.75
    assert shares["no-queue"] == pytest.approx(2.0 / 3.75) and shares["budget-other"] == pytest.approx(1.75 / 3.75)
    assert sum(shares.values()) == pytest.approx(1.0) and set(shares) == set(ad.SHARE_KEYS)
    assert all(v == 0.0 for v in ad.q_shares([tl(1001.0, 1001.0)], cl).values())  # no queue time at all


def test_tail_set_p90_includes_ties():
    def reqs(ttfts, cls="short"):
        return [{"idx": i, "cls": cls, "client_ttft": t} for i, t in enumerate(ttfts)]

    assert [r["idx"] for r in ad.tail_set(reqs([0.1 * k for k in range(1, 11)]))] == [9]  # P90 = 0.91
    tied = reqs([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0])
    assert [r["idx"] for r in ad.tail_set(tied)] == [8, 9]  # P90 = 1.0
    mixed = tied + [{"idx": 10, "cls": "long", "client_ttft": 50.0}]
    assert [r["idx"] for r in ad.tail_set(mixed)] == [8, 9]  # long requests never enter the tail
    assert [r["idx"] for r in ad.tail_set(tied, q=95)] == [8, 9]


def test_step_shares_residual_bm_and_unexplained_break():
    # four long requests prefilling over [1000.5, 1003.5]; steps 0..9, t = 1000 + j; lag 2 -> tau_j = 998 + j
    longs = [tl(1000.0, 1000.5, tokens=LONG, f=1003.5, rid=f"L{k}") for k in range(4)]
    cl = ad.classify_steps(steps_at([{}] * 10), longs, nb=248)
    # classifiable steps 2..9 (tau 1000..1007); >= 4 long prefilling at tau 1001, 1002, 1003 -> 3 of 8
    sh = ad.step_shares(cl, t_start=1000.0, t_end=1009.0)
    assert sh["n_steps"] == 8 and sh["residual_bm_share"] == pytest.approx(3 / 8)
    three = ad.classify_steps(steps_at([{}] * 10), longs[:3], nb=248)
    assert ad.step_shares(three, t_start=1000.0, t_end=1009.0)["residual_bm_share"] == 0.0
    # period restricts the steps: t_1 in [1004, 1009] -> steps 4..9 (tau 1002..1007): 2 of 6
    assert ad.step_shares(cl, t_start=1004.0, t_end=1009.0)["residual_bm_share"] == pytest.approx(2 / 6)
    # unexplained-break share = unexplained / queued steps
    specs = [{"ctx_tok": 1000, "u": 50.1}] * 4 + [{"ctx_tok": 2048, "u": 50.1}] * 2 + [{"u": 50.1}] * 4
    q = ad.classify_steps(steps_at(specs), [tl(999.0, 1005.5)], nb=248)
    # steps 2..7 have tau 1000..1005 < 1005.5 -> queued; N < B at steps 2, 3 -> unexplained; 4, 5, 6, 7 -> budget
    sh = ad.step_shares(q, t_start=1000.0, t_end=1009.0)
    assert sh["n_queued_steps"] == 6 and sh["unexplained_break_share"] == pytest.approx(2 / 6)
    assert ad.step_shares([], t_start=0, t_end=1)["unexplained_break_share"] == 0.0


# ---------------------------------------------------------------- Prometheus deltas

def prom(*, q_count, q_sum, prompt_tokens, preemptions, b03=None):
    lbl = 'engine="0",model_name="m"'
    return "\n".join([
        "# HELP vllm:request_queue_time_seconds Histogram of time spent in WAITING phase for request.",
        "# TYPE vllm:request_queue_time_seconds histogram",
        f"vllm:request_queue_time_seconds_sum{{{lbl}}} {q_sum}",
        f'vllm:request_queue_time_seconds_bucket{{engine="0",le="0.3",model_name="m"}} '
        f"{q_count if b03 is None else b03}",
        f'vllm:request_queue_time_seconds_bucket{{engine="0",le="+Inf",model_name="m"}} {q_count}',
        f"vllm:request_queue_time_seconds_count{{{lbl}}} {q_count}",
        f"vllm:request_queue_time_seconds_created{{{lbl}}} 1.76e+09",
        f"vllm:prompt_tokens_total{{{lbl}}} {prompt_tokens}",
        f"vllm:num_preemptions_total{{{lbl}}} {preemptions}",
    ]) + "\n"


def test_histogram_delta_and_counter_delta():
    before = prom(q_count=3.0, q_sum=1.5, prompt_tokens=100.0, preemptions=0.0, b03=2.0)
    after = prom(q_count=9.0, q_sum=4.75, prompt_tokens=400.0, preemptions=2.0, b03=5.0)
    d = ad.histogram_delta(before, after, "vllm:request_queue_time_seconds")
    assert d == {"buckets": {"0.3": 3.0, "+Inf": 6.0}, "sum": pytest.approx(3.25), "count": 6.0}
    assert ad.counter_delta(before, after, "vllm:prompt_tokens_total") == 300.0
    assert ad.counter_delta(before, after, "vllm:num_preemptions_total") == 2.0
    two = after + 'vllm:prompt_tokens_total{engine="1",model_name="m"} 50.0\n'
    assert ad.counter_delta(before, two, "vllm:prompt_tokens_total") == 350.0  # label sets are summed


def test_counter_reset_and_missing_series_raise():
    before = prom(q_count=9.0, q_sum=4.75, prompt_tokens=400.0, preemptions=0.0)
    after = prom(q_count=3.0, q_sum=1.5, prompt_tokens=100.0, preemptions=0.0)
    with pytest.raises(ad.CounterReset):
        ad.histogram_delta(before, after, "vllm:request_queue_time_seconds")
    with pytest.raises(ad.CounterReset):
        ad.counter_delta(before, after, "vllm:prompt_tokens_total")
    with pytest.raises(ad.DecompError):
        ad.counter_delta(before, "", "vllm:prompt_tokens_total")
    with pytest.raises(ad.DecompError):
        ad.histogram_delta(before, "", "vllm:request_queue_time_seconds")


# ---------------------------------------------------------------- 1 Hz shares (A13, descriptive)

def test_one_hz_shares():
    def s(u, w):
        return {"vllm:kv_cache_usage_perc": u, "vllm:num_requests_waiting": w}

    samples = [s(0.99, 1), s(0.80, 2), s(0.99, 0), s(0.10, 3), s(None, 1)]
    got = ad.one_hz_shares(samples, nb=248, L=2048)
    # free: 2 (< 7, < 65), 49 (< 65), 2 but nobody waiting, 222; the None sample is skipped
    assert got == {"kv_full_while_waiting": 0.25, "gate_zone": 0.5, "n_samples": 4}
    assert ad.one_hz_shares([s(0.95, 1)], nb=248, L=8192)["kv_full_while_waiting"] == 1.0  # 12 < 19
    assert ad.one_hz_shares([s(0.95, 1)], nb=248, L=2048)["kv_full_while_waiting"] == 0.0  # 12 >= 7
    assert ad.one_hz_shares([], nb=248, L=2048)["kv_full_while_waiting"] is None


# ---------------------------------------------------------------- end to end: summarize_run (A8-A10)

def span_dict(r):
    """The decoded-span form (as decode_otlp_traces returns it) of a mapped record."""
    names = ad.SPAN_ATTRS
    return {"name": "llm_request", "start_time_unix_nano": int(round(r["arrival"] * 1e9)), "end_time_unix_nano": 0,
            "attributes": {names[k]: r[k] for k in ("id", "prompt_tokens", "completion_tokens", "Q", "P",
                                                     "server_ttft", "e2e")}}


def fixture(*, drop_steps=(), dup_step=None, completion3=128, after_over=None, preemptions_after=0.0,
            drop_span=None, no_prompt_tokens=None):
    main = [rec(i, arrival=1000.0 + 0.01 * i, Q=0.0, P=0.5, e2e=1.5) for i in range(8)]
    main.append(rec(8, arrival=1001.49, Q=2.75, P=0.5, e2e=5.01))   # e 1001.5, s 1004.25, end 1006.5
    main.append(rec(9, arrival=1002.49, Q=1.0, P=1.0, e2e=2.5))     # e 1002.5, s 1003.5
    main[3]["completion_tokens"] = completion3
    if no_prompt_tokens is not None:
        main[no_prompt_tokens]["prompt_tokens"] = None
    warm = [rec("cmpl-0a1b2c3d4e5f-0", arrival=999.5, Q=0.0, P=0.1),  # warmup / ready-check: no prefix
            rec("cmpl-9f8e7d6c-0", arrival=999.6, Q=0.0, P=0.1)]
    recs = [r for r in main if drop_span is None or r["id"] != ad.request_id(PREFIX, drop_span)] + warm
    spans = [span_dict(r) for r in recs] + [{"name": "other", "start_time_unix_nano": 0, "attributes": {}}]
    detailed = {"input_lens": [1920] * 10, "output_lens": [128] * 10, "completed": 10, "failed": 0,
                "ttfts": [0.6 + 0.01 * i for i in range(8)] + [3.3, 3.3]}
    lines = [iter_line(j, t=1000.0 + j) for j in range(7) if j not in drop_steps]
    if dup_step is not None:
        lines.append(iter_line(dup_step, t=1000.0 + dup_step + 0.5))
    steps = ad.parse_iteration_log(lines)
    ctx = sum(s["ctx_tok"] for s in steps)
    before = prom(q_count=3.0, q_sum=0.25, prompt_tokens=5000.0, preemptions=0.0)
    over = {"q_count": 0.0, "q_sum": 0.0, "prompt_tokens": 0.0, **(after_over or {})}
    after = prom(q_count=3.0 + 12 + over["q_count"], q_sum=0.25 + 3.75 + over["q_sum"],
                 prompt_tokens=5000.0 + ctx + over["prompt_tokens"], preemptions=preemptions_after)
    return dict(spans=spans, detailed=detailed, steps=steps, prefix=PREFIX, L=2048, cond="B0", num_gpu_blocks=248,
                metrics_before=before, metrics_after=after, window=(999.0, 1010.0))


def test_summarize_run_hand_computed():
    d = ad.summarize_run(**fixture())
    assert d["checks"] == {"a": True, "b": True, "c": True, "d": True} and d["valid"] is True
    assert d["n_tail"] == 2  # ties at the P90 (3.3 s)
    assert d["tail"]["O"] == pytest.approx(665.0) and d["tail"]["Q"] == pytest.approx(1875.0)
    assert d["tail"]["P"] == pytest.approx(750.0) and d["tail"]["F"] == pytest.approx(10.0)
    t = d["tail"]
    assert t["O"] + t["F"] + t["Q"] + t["P"] == pytest.approx(3300.0)  # = mean tail client TTFT
    assert set(d["q_shares"]) == set(ad.SHARE_KEYS) == set(d["q_shares_sens"])
    # main (tau = t_{j-2}): steps 2, 3 no-queue; 4, 5, 6 budget-other. 8: 1.5 / 1.25; 9: 0.5 / 0.5
    assert d["q_shares"]["no-queue"] == pytest.approx(2.0 / 3.75)
    assert d["q_shares"]["budget-other"] == pytest.approx(1.75 / 3.75)
    assert d["q_shares"]["unclassified"] == pytest.approx(0.0)
    # sensitivity (tau = t_{j-1}): steps 1, 2 no-queue; 3, 4, 5 budget-other; 6 no-queue.
    # 8: (1001, 1002] 0.5 no-queue + 2.25 budget-other; 9: 1.0 budget-other
    assert d["q_shares_sens"]["no-queue"] == pytest.approx(0.5 / 3.75)
    assert d["q_shares_sens"]["budget-other"] == pytest.approx(3.25 / 3.75)
    # period [1000.0, 1006.5]: classifiable steps 2..6 at lag 2; queued 4, 5, 6; none unexplained, no long
    assert d["n_steps"] == 5 and d["n_queued_steps"] == 3
    assert d["residual_bm_share"] == 0.0 and d["unexplained_break_share"] == 0.0
    assert d["tail95"]["Q"] == pytest.approx(1875.0)


def test_summarize_run_none_without_a_bench_result():
    assert ad.summarize_run(**{**fixture(), "detailed": None}) is None


def test_check_a_fails_on_token_mismatch_or_missing_span():
    d = ad.summarize_run(**fixture(completion3=127))
    assert d["checks"] == {"a": False, "b": True, "c": True, "d": True} and d["valid"] is False
    d = ad.summarize_run(**fixture(drop_span=2, after_over={"q_count": -1.0}))
    assert d["checks"]["a"] is False and d["checks"]["b"] is True and d["valid"] is False
    assert d["tail"]["Q"] == pytest.approx(1875.0)  # the tail itself is still described
    d = ad.summarize_run(**fixture(no_prompt_tokens=4))  # a span without the token attribute is a mismatch
    assert d["checks"] == {"a": False, "b": True, "c": True, "d": True}


def test_check_b_fails_on_count_or_sum():
    d = ad.summarize_run(**fixture(after_over={"q_count": 1.0}))
    assert d["checks"] == {"a": True, "b": False, "c": True, "d": True} and d["valid"] is False
    assert ad.summarize_run(**fixture(after_over={"q_sum": 0.002}))["checks"]["b"] is False
    assert ad.summarize_run(**fixture(after_over={"q_sum": 0.0005}))["checks"]["b"] is True  # within 1 ms
    reset = ad.summarize_run(**fixture(after_over={"q_count": -14.0}))
    assert reset["checks"]["b"] is False and any("reset" in r for r in reset["reasons"])


def test_check_c_fails_on_prompt_tokens_and_is_exempt_with_preemptions():
    d = ad.summarize_run(**fixture(after_over={"prompt_tokens": 1.0}))
    assert d["checks"] == {"a": True, "b": True, "c": False, "d": True} and d["valid"] is False
    d = ad.summarize_run(**fixture(after_over={"prompt_tokens": 1.0}, preemptions_after=2.0))
    assert d["checks"]["c"] is None and d["valid"] is True
    assert any("preemption" in r for r in d["reasons"])


def test_check_d_fails_on_gap_or_duplicate_index():
    d = ad.summarize_run(**fixture(drop_steps=(3,)))
    assert d["checks"] == {"a": True, "b": True, "c": True, "d": False} and d["valid"] is False
    d = ad.summarize_run(**fixture(dup_step=4))
    assert d["checks"]["d"] is False and d["checks"]["c"] is True
    empty = fixture()
    empty["steps"] = []
    assert ad.summarize_run(**empty)["checks"]["d"] is False


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        ad.summarize_run(**{**fixture(), "cond": "X"})


def test_otel_and_run_core_b_are_imported_lazily():
    """opentelemetry.proto exists only in the WSL venv; run_core_b imports tomllib (3.11+), so neither is a
    module-level import (stdlib only, Python 3.10 compatible)."""
    with open(ad.__file__) as fh:
        top = [ln for ln in fh.read().splitlines() if ln.startswith(("import ", "from "))]
    assert not [ln for ln in top if "opentelemetry" in ln or "run_core_b" in ln]
    assert math.isclose(ad.BUDGET, 2048) and ad.BLOCK_SIZE == 528
