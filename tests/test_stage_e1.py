"""Tests for benchmarks/stage_e1.py (decision 041 v3 §1): the slip-chain links L1-L5 on synthetic closed-loop benches,
continuous and g 24 slips, the send lag, the T / TG F-spike description, the chain summary, the manifest-checked
inputs, the clock sources and the freeze gate."""

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import stage_e1 as e1

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "configs" / "stage_e1.toml").read_text())
FROZEN = {**CFG, "freeze": {"e1": True, "e1_harness_sha256": e1.harness_sha256()}}
PERIOD = 0.416
EPOCH = 1.79e9                                   # server clock (spans, steps, derived arrival)
DELAY_MS = (0.2, 0.3, 0.45)                      # client send -> server arrival, by idx % 3
LAG_MS = 0.05                                    # client completion -> the send it releases
PERM = (3, 0, 1, 2, 7, 4, 5, 6)                  # g 0 completes in this idx order


def synth(slips=(5,), n=200, spike=True, extra_step=True, spikes=(), shifts=(0.326,) * 7):
    """A closed-loop 8K bench: one request every PERIOD s; a slip at group k moves 8k+j shifts[j - 1] s earlier
    (the k - 1 group finished early), adds 340 ms of TTFT and 348 ms of Q to 8k+1..8k+7, a 16 ms F spike to 8k and one
    more decode-only step before 8k's first prefill chunk. spikes: groups with only the F spike and the extra step."""
    arrival = [i * PERIOD for i in range(n)]
    for k in slips:
        for j in range(1, 8):
            arrival[8 * k + j] -= shifts[j - 1]
    ttft = [1000.0] * n
    q = [100.0] * n
    f = [26.0] * n
    for k in slips:
        for j in range(1, 8):
            ttft[8 * k + j] += 340.0
            q[8 * k + j] += 348.0
    for k in (tuple(slips) if spike else ()) + tuple(spikes):
        f[8 * k] = 42.0
    completion = [arrival[i + 8] if i + 8 < n else arrival[i] + 8 * PERIOD for i in range(n)]
    lat = [(c - a) * 1000 for c, a in zip(completion, arrival)]
    steps = []
    for g in range(n // 8):
        a = arrival[8 * g]
        extra = (g in slips and extra_step) or g in spikes
        offsets = (0.012, 0.027, 0.042, 0.087) if extra else (0.012, 0.027, 0.072)
        steps += [{"t": a + o, "ctx_tok": 0} for o in offsets] + [{"t": a + 0.3, "ctx_tok": 2048}]
    return {"ttft": ttft, "arrival": arrival, "lat": lat, "q": q, "f": f, "steps": steps, "completion": completion}


def run(d, **kw):
    args = dict(lat_ms=d["lat"], times_s=d["arrival"], cfg=CFG, F_ms=d["f"], Q_ms=d["q"], steps=d["steps"],
                arrivals_s=d["arrival"])
    args.update(kw)
    return e1.bench_episodes(d["ttft"], **args)


def client_side(d, clock):
    """The client's bench.json view of a synthetic bench on its own clock: each send DELAY_MS before the server
    arrival, g 0 completing out of order (PERM), every later send LAG_MS after the completion that releases it."""
    n = len(d["arrival"])
    start = [clock + a - DELAY_MS[i % 3] / 1000 for i, a in enumerate(d["arrival"])]
    done = [None] * n
    for m in range(n - 8):
        done[PERM[m] if m < 8 else m] = start[m + 8] - LAG_MS / 1000
    for i in range(n - 8, n):
        done[i] = clock + d["completion"][i]
    return start, [c - s for c, s in zip(done, start)]


def rows_of(d):
    """Advanced derived rows on the server clock; server_ttft - P = Q + F, so 8g's prefill starts at arrival + Q + F."""
    return [{"idx": i, "client_ttft_ms": d["ttft"][i], "arrival": EPOCH + d["arrival"][i], "Q_ms": d["q"][i],
             "F_ms": d["f"][i], "P_ms": 900.0, "server_ttft_ms": d["q"][i] + d["f"][i] + 900.0}
            for i in range(len(d["ttft"]))]


def test_every_link_holds_on_a_textbook_slip():
    out = run(synth())
    (ep,) = out["episodes"]
    assert (ep["start_idx"], ep["k"], ep["length"], ep["continuous"]) == (41, 5, 7, False)
    assert ep["links"] == {"L1": True, "L2": True, "L3": True, "L4": True, "L5": True}
    v = ep["values"]
    assert v["gap_ms"] == pytest.approx(90.0) and v["L3_dev_ms"] == pytest.approx(-326.0)
    assert (v["n_dec"], v["n_dec_median"]) == (4, 3) and v["dQ_ms"] == pytest.approx(348.0)
    assert out["l1_in_normal_groups"] == {"groups": 22, "spikes": 0} and ep["in_base_range"] is True


def test_a_missing_link_is_reported_not_hidden():
    (ep,) = run(synth(spike=False, extra_step=False))["episodes"]
    assert ep["links"]["L1"] is False and ep["links"]["L2"] is False and ep["links"]["L4"] is True
    s = e1.chain_summary([run(synth(spike=False))], CFG)
    assert s["with_a_failing_link"] == 1 and s["chain"] == "holds"                  # tolerance 1
    s = e1.chain_summary([run(synth(spike=False)), run(synth(spike=False))], CFG)
    assert s["chain"] == "incomplete"


def test_without_step_or_server_data_the_links_are_none():
    d = synth()
    (ep,) = run(d, F_ms=None, Q_ms=None, steps=None, arrivals_s=None)["episodes"]
    assert ep["links"] == {"L1": None, "L2": None, "L3": True, "L4": True, "L5": None}
    lat = d["lat"][:]
    lat[8 * 4 + 3] = None                                                          # a missing completion in k - 1
    (ep,) = run(d, lat_ms=lat)["episodes"]
    assert ep["links"]["L3"] is None and "L3_dev_ms" not in ep["values"]


def test_a_continuous_slip_counts_once_and_skips_l3():
    out = run(synth(slips=(8, 9)))
    eps = {e["k"]: e for e in out["episodes"]}
    assert eps[9]["continuous"] is True and eps[9]["links"]["L3"] is None and eps[8]["continuous"] is False
    s = e1.chain_summary([out], CFG)
    assert s["episodes"] == 1 and s["continuous"] == 1 and s["start_idx"] == {65: 1, 73: 1}


def test_a_g24_slip_is_listed_but_not_counted():
    out = run(synth(slips=(24,)))
    (ep,) = out["episodes"]
    assert (ep["k"], ep["in_base_range"]) == (24, False) and ep["links"]["L3"] is True
    s = e1.chain_summary([out], CFG)
    assert (s["episodes"], s["outside_base_range"], s["chain"]) == (0, 1, "not evaluable")
    s = e1.chain_summary([out, run(synth(slips=(24,), spike=False)), run(synth())], CFG)
    assert (s["episodes"], s["with_a_failing_link"], s["chain"]) == (1, 0, "holds")      # g 24's failing L1 ignored


def test_a_short_bench_has_no_g24():
    d = synth(slips=(22,), n=192)
    out = run(d)
    (ep,) = out["episodes"]
    assert ep["k"] == 22 and ep["links"] == {"L1": True, "L2": True, "L3": True, "L4": True, "L5": True}
    assert e1.baseline_groups(192, {22}) == [g for g in range(1, 23) if g not in (21, 22)]
    proxy = [(d["arrival"][i + 8] - d["arrival"][i]) * 1000 if i + 8 < 192 else None for i in range(192)]
    assert run(d, lat_ms=proxy)["episodes"][0]["values"]["L3_dev_ms"] == pytest.approx(-326.0)


def test_baselines_skip_g0_the_last_group_slips_and_their_precursors():
    assert e1.baseline_groups(200, {5}) == [g for g in range(1, 24) if g not in (4, 5)]
    assert e1.baseline_groups(200, set())[-1] == 23


def test_l1_spikes_in_normal_groups_are_counted():
    d = synth()
    d["f"][8 * 12] = 50.0
    s = e1.chain_summary([run(d)], CFG)
    assert s["l1_rate_in_normal_groups"] == pytest.approx(1 / 22) and s["chain"] == "incomplete"   # > 2 %


def test_send_lags_match_sends_and_completions_in_time_order():
    done = [1.7, 1.0, 1.2, 1.1, 1.6, 1.3, 1.5, 1.4]                                # g 0 finishes out of order
    sends = sorted(done)
    starts = [0.0] * 8 + [sends[1], sends[0]] + sends[2:]                          # idx 9 is sent before idx 8
    starts = [s + 0.0001 if i >= 8 else s for i, s in enumerate(starts)]
    lags = e1.send_lags_by_idx(starts, done + [9.0] * 8)
    assert sorted(lags) == list(range(8, 16)) and all(v == pytest.approx(0.1) for v in lags.values())
    d = synth()
    starts, lat = d["arrival"], d["lat"]
    lags = e1.send_lags_by_idx(starts, [s + v / 1000 for s, v in zip(starts, lat)])
    (ep,) = run(d, send_lags=lags)["episodes"]
    assert len(ep["values"]["send_lag_ms"]) == 7 and ep["values"]["send_lag_ms_max"] == pytest.approx(0.0, abs=1e-6)


def test_t_and_tg_get_only_the_anchored_f_spike_description():
    d = synth(slips=(), spikes=(6, 14))
    out = e1.f_spikes(rows_of(d), [{**s, "t": EPOCH + s["t"]} for s in d["steps"]], CFG)
    assert out["spike_groups"] == [6, 14] and out["rate_g1_23"] == pytest.approx(2 / 23)
    nd = out["n_dec_anchor"]
    assert nd["median_g1_23"] == 3 and nd["spikes_at_median_plus_1"] == 2
    assert (nd["spikes"][6]["n"], nd["spikes"][6]["decode_only"]) == (4, 4)
    assert nd["spikes"][6]["margin_ms"] == pytest.approx(12, abs=1e-3)              # the first stamp after arrival
    assert e1.f_spikes(rows_of(d), None, CFG)["n_dec_anchor"] is None
    steps = [{**s, "t": EPOCH + s["t"], "ctx_tok": 512 if s["t"] == d["arrival"][48] + 0.042 else s["ctx_tok"]}
             for s in d["steps"]]                                                   # a T chunk of another request
    six = e1.f_spikes(rows_of(d), steps, CFG)["n_dec_anchor"]["spikes"][6]
    assert (six["n"], six["decode_only"], six["ambiguous"]) == (4, 3, False)
    assert six["edges"] == pytest.approx({"arrival_minus_stamp_ms": 8 * 416 - 300, "stamp_minus_arrival_ms": 12,
                                          "s_minus_stamp_ms": 55, "stamp_minus_s_ms": 158}, abs=1e-3)
    assert six["margin_ms"] == pytest.approx(12, abs=1e-3)


def test_a_count_within_a_stamp_of_its_window_edge_is_ambiguous():
    d = synth(slips=(), spikes=(6, 14))
    s6 = d["arrival"][48] + (100 + 42) / 1000                                       # 8 * 6's prefill start
    steps = sorted([{"t": EPOCH + x["t"], "ctx_tok": x["ctx_tok"]} for x in d["steps"]] +
                   [{"t": EPOCH + s6 + 0.0005, "ctx_tok": 0}], key=lambda x: x["t"])  # 0.5 ms after s
    nd = e1.f_spikes(rows_of(d), steps, CFG)["n_dec_anchor"]
    assert nd["spikes"][6]["n"] == 4 and nd["spikes"][6]["ambiguous"] is True
    assert nd["spikes"][6]["edges"]["stamp_minus_s_ms"] == pytest.approx(0.5, abs=1e-3)
    assert (nd["spikes_at_median_plus_1"], nd["spikes_at_median_plus_1_unambiguous"]) == (2, 1)
    assert nd["ambiguous_groups_g1_23"] == 1


def test_the_config_is_041s_values_and_unfrozen():
    assert e1.config_problems(CFG) == [] and CFG["freeze"] == {"e1": False, "e1_harness_sha256": ""}
    for k, v in (("slip_ms", 150.0), ("f_spike_ms", 30.0), ("gap_ms", 250.0), ("chain_tolerance", 2),
                 ("l1_false_max", 0.05), ("chain_tolerance", 1.0)):
        assert e1.config_problems({**CFG, "e1": {**CFG["e1"], k: v}}), k
    assert e1.config_problems({**CFG, "freeze": {"e1": True, "e1_harness_sha256": ""}})
    assert e1.config_problems({**CFG, "freeze": {"e1": False, "e1_harness_sha256": "a" * 64}})
    assert e1.config_problems(FROZEN) == []
    assert {Path(f).name for f in e1.HARNESS_FILES} == {"stage_e1.py", "c0_analysis.py"}


def test_fetch_list_and_analyze_need_the_frozen_harness(tmp_path):
    assert "freeze.e1 must be true" in e1.gate_problems(CFG)[0]
    wrong = {**CFG, "freeze": {"e1": True, "e1_harness_sha256": "0" * 64}}
    assert e1.gate_problems(wrong) == ["the harness differs from freeze.e1_harness_sha256"]
    assert e1.gate_problems(FROZEN) == []
    root = synthetic_repo(tmp_path / "repo")
    with pytest.raises(e1.StageE1Error, match="freeze.e1"):
        e1.analyze(root, CFG, config_sha256="x")
    with pytest.raises(e1.StageE1Error, match="harness"):
        e1.analyze(root, wrong, config_sha256="x")


def test_the_output_is_a_new_file_under_results_stage_e(tmp_path):
    base = tmp_path / "results" / "stage_e"
    assert e1.out_problems(base / "e1-20261101" / "e1.json", root=tmp_path) == []
    assert e1.out_problems(base / "e1.json", root=tmp_path)
    assert e1.out_problems(tmp_path / "results" / "capacity" / "x" / "e1.json", root=tmp_path)
    assert e1.out_problems(base / "a" / ".." / ".." / "x" / "e1.json", root=tmp_path)
    (base / "e1-20261101").mkdir(parents=True)
    (base / "e1-20261101" / "e1.json").write_text("{}")
    assert e1.out_problems(base / "e1-20261101" / "e1.json", root=tmp_path) == [
        f"{(base / 'e1-20261101' / 'e1.json').resolve()} already exists"]


def test_the_fetch_list_is_what_041_names():
    (adv_sub, adv_session, adv), (c0_sub, c0_session, c0) = e1.FETCH
    assert (adv_sub, adv_session, c0_sub, c0_session) == ("advanced", "formal-20261011-adv", "capacity",
                                                          "explore-c0-20261016")
    assert len(adv) == 9 * 2 + 6 and "T-8192-uniform-r1.steps.jsonl" in adv and "B0-8192-uniform-r1.steps.jsonl" not in adv
    assert "TG-8192-uniform-r1.spans.jsonl" not in adv and "K3-8192-uniform-r3.bench.json" in adv
    assert len(c0) == 24


# ---------------------------------------------------------------- the whole analysis on a synthetic repo

def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _jsonl(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


def synthetic_repo(root, *, skip=(), clock=-1e5, adv_starts=None, e2e_extra_ms=None, shifts=(0.326,) * 7):
    """Committed derived files and manifests plus the raw copies under COPIES/<session>/ (fetch_raw.sh's DEST);
    skip: raw names not (yet) fetched; clock: the client perf_counter origin relative to the synthetic times (client
    files follow client_side); adv_starts: advanced bench.json start_times override; e2e_extra_ms: {idx: ms} added to
    the B0 spans' e2e; shifts: the advanced slips' per-position arrival shifts."""
    manifests = {e1.ADV: {}, e1.C0: {}, e1.CORE: {}}

    def raw(session, name, text):
        digest = hashlib.sha256(text.encode()).hexdigest()
        manifests[session][f"raw/{name}"] = {"sha256": digest}
        if name not in skip:
            _write(root, e1.COPIES / session.name / name, text)

    for tag in e1.ADV_TAGS:
        cond, r = tag.split("-")[0], int(tag[-1])
        d = synth(slips=(5,) if cond in e1.SLIP_CONDS else (), spikes=(6, 14) if cond in e1.AUX_CONDS else (),
                  shifts=shifts)
        _write(root, e1.ADV / "derived" / f"{tag}-requests.jsonl", _jsonl(rows_of(d)))
        raw(e1.ADV, f"{tag}.steps.jsonl", _jsonl([{**s, "t": EPOCH + s["t"]} for s in d["steps"]]))
        extra = (e2e_extra_ms or {}) if cond == "B0" else {}
        spans = [{"id": f"cmpl-adv-{cond}-8192-uniform-r{r}-{i}-0", "arrival": EPOCH + d["arrival"][i],
                  "e2e": d["completion"][i] - d["arrival"][i] + extra.get(i, 0.0) / 1000} for i in range(200)]
        raw(e1.ADV, f"{tag}.spans.jsonl", _jsonl(spans))
        starts, lats = client_side(d, clock)
        raw(e1.ADV, f"{tag}.bench.json", json.dumps({"start_times": adv_starts or starts, "latencies": lats}))
    for b in e1.C0_BENCHES:
        d = synth(slips=(11,) if b == "b1-n2" else ())
        starts, lats = client_side(d, clock)
        _write(root, e1.C0 / "derived" / f"{b}-requests.jsonl",
               _jsonl([{"idx": i, "ttft_ms": d["ttft"][i], "latency_ms": lats[i] * 1000} for i in range(200)]))
        raw(e1.C0, f"{b}.bench.json", json.dumps({"start_times": starts}))
    for r in (1, 2, 3):
        d = synth(slips=(8, 9) if r == 1 else (5,))
        starts, lats = client_side(d, clock)
        raw(e1.CORE, f"bench-uniform-8192-r{r}-c8.json",
            json.dumps({"ttfts": [x / 1000 for x in d["ttft"]], "latencies": lats, "start_times": starts}))
    for session, files in manifests.items():
        _write(root, session / "raw_manifest.json", json.dumps({"files": files}))
    return root


def analyze(root):
    return e1.analyze(root, FROZEN, config_sha256="c" * 64)


def rounded(x, nd=5):
    """Floats to nd decimals (ms values; a clock origin moves the last bits)."""
    if isinstance(x, float):
        return round(x, nd) + 0.0
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, nd) for v in x]
    return x


def links(out):
    return [(r["run"], [(x["k"], x["links"], {k: v for k, v in x["values"].items() if not k.startswith("send_lag")})
                        for x in r["episodes"]]) for ds in out["datasets"].values() for r in ds["runs"]]


def test_the_analysis_reports_each_dataset_from_manifest_checked_inputs(tmp_path):
    root = synthetic_repo(tmp_path / "repo")
    out = analyze(root)
    assert out["descriptive_only"] is True and out["harness_sha256"] == e1.harness_sha256()
    assert out["stage_e1_sha256"] == "c" * 64
    adv = out["datasets"]["advanced"]
    assert adv["summary"]["chain"] == "holds" and adv["summary"]["episodes"] == 9 and adv["blocked"] == []
    assert all(r["completion_source"] == "e2e" for r in adv["runs"])
    for r in adv["runs"]:                                    # the client clock only: never server arrival - e2e
        assert r["episodes"][0]["values"]["send_lag_ms"] == pytest.approx([LAG_MS] * 7, abs=1e-6)
    assert {r["run"].split("-")[0] for r in adv["runs"]} == set(e1.SLIP_CONDS)
    assert {r["run"].split("-")[0] for r in adv["auxiliary_T_TG"]} == set(e1.AUX_CONDS)
    assert all("episodes" not in r and r["f_spikes"]["spike_groups"] == [6, 14] for r in adv["auxiliary_T_TG"])
    assert adv["f_spike_rate_g1_23"] == pytest.approx({"B0_G_K3": 1 / 23, "T_TG": 2 / 23})
    c0 = out["datasets"]["c0"]
    assert c0["summary"]["start_idx"] == {89: 1} and c0["summary"]["chain"] == "holds"
    (ep,) = next(r for r in c0["runs"] if r["run"] == "b1-n2")["episodes"]
    gap = 416 - 326 - (DELAY_MS[89 % 3] - DELAY_MS[88 % 3])                          # client starts, not the grid
    assert ep["links"] == {"L1": None, "L2": None, "L3": True, "L4": True, "L5": None}
    assert ep["values"]["gap_ms"] == pytest.approx(gap, abs=1e-6)
    assert ep["values"]["send_lag_ms"] == pytest.approx([LAG_MS] * 7, abs=1e-6)
    assert all(r["send_lag_ms_max"] == pytest.approx(LAG_MS, abs=1e-6) for r in c0["runs"])  # g 0 out of order too
    core = out["datasets"]["core"]
    assert core["summary"]["continuous"] == 1
    assert all(r["send_lag_ms_max"] == pytest.approx(LAG_MS, abs=1e-6) for r in core["runs"])


def test_every_fetched_file_is_read_from_fetch_raws_destination(tmp_path):
    out = analyze(synthetic_repo(tmp_path / "repo"))
    fetched = {str(e1.COPIES / session / n) for _, session, names in e1.FETCH for n in names}
    assert fetched <= set(out["inputs_sha256"])
    local = {str(e1.COPIES / "formal-20261011-adv" / f"{t}.steps.jsonl") for t in e1.ADV_TAGS
             if t.split("-")[0] in e1.SLIP_CONDS}
    local |= {str(e1.COPIES / "formal-20261009-coreb-uniform" / f"bench-uniform-8192-r{r}-c8.json") for r in (1, 2, 3)}
    derived = {p for p in out["inputs_sha256"] if "/derived/" in p}
    assert set(out["inputs_sha256"]) == fetched | local | derived and len(derived) == 15 + 24


def test_the_clock_sources_never_mix(tmp_path):
    a = analyze(synthetic_repo(tmp_path / "a", clock=-1e5))
    b = analyze(synthetic_repo(tmp_path / "b", clock=7.5))
    assert rounded(a["datasets"]) == rounded(b["datasets"])
    c = analyze(synthetic_repo(tmp_path / "c", adv_starts=[1e9 + 13 * i for i in range(200)]))
    assert rounded(links(c)) == rounded(links(a))           # nonsense client times change only the client send lag
    lags = [x["values"]["send_lag_ms"] for r in c["datasets"]["advanced"]["runs"] for x in r["episodes"]]
    assert all(v > 1000 for v in sum(lags, []))


def test_l3_uses_the_e2e_completion_else_the_proxy_arrival_8_later(tmp_path):
    shifts = tuple(0.300 + 0.004 * j for j in range(1, 8))        # 8k+7 moves most: an i + 7 proxy would miss it
    extra = {8 * 4 + j: 3.0 for j in range(1, 8)}                  # the server's completions of g 4 come 3 ms later
    out = analyze(synthetic_repo(tmp_path / "e2e", e2e_extra_ms=extra, shifts=shifts))
    b0 = [r for r in out["datasets"]["advanced"]["runs"] if r["run"].startswith("B0")]
    assert all(r["episodes"][0]["values"]["L3_dev_ms"] == pytest.approx(-325.0, abs=1e-3) for r in b0)
    names = tuple(f"{t}.spans.jsonl" for t in e1.ADV_TAGS)
    out = analyze(synthetic_repo(tmp_path / "proxy", e2e_extra_ms=extra, skip=names, shifts=shifts))
    b0 = [r for r in out["datasets"]["advanced"]["runs"] if r["run"].startswith("B0")]
    assert all(r["completion_source"] == "proxy" for r in b0)
    assert all(r["episodes"][0]["values"]["L3_dev_ms"] == pytest.approx(-328.0, abs=1e-3) for r in b0)


def test_a_partial_fetch_blocks_only_what_depends_on_it(tmp_path):
    skip = ("G-8192-uniform-r2.spans.jsonl", "K3-8192-uniform-r1.bench.json", "T-8192-uniform-r3.steps.jsonl",
            "b1-n2.bench.json")
    out = analyze(synthetic_repo(tmp_path / "repo", skip=skip))
    adv = {r["run"]: r for r in out["datasets"]["advanced"]["runs"] + out["datasets"]["advanced"]["auxiliary_T_TG"]}
    assert adv["G-8192-uniform-r2"]["completion_source"] == "proxy" and adv["G-8192-uniform-r1"]["blocked"] == []
    assert adv["K3-8192-uniform-r1"]["blocked"] == ["bench.json (client send lag)"]
    assert "send_lag_ms" not in adv["K3-8192-uniform-r1"]["episodes"][0]["values"]
    assert adv["T-8192-uniform-r3"]["f_spikes"]["n_dec_anchor"] is None
    assert adv["T-8192-uniform-r2"]["f_spikes"]["n_dec_anchor"]["spikes_at_median_plus_1"] == 2
    c0 = {r["run"]: r for r in out["datasets"]["c0"]["runs"]}
    (ep,) = c0["b1-n2"]["episodes"]
    assert ep["links"]["L3"] is True and ep["links"]["L4"] is None and "send_lag_ms_max" not in c0["b1-n2"]
    assert out["datasets"]["c0"]["blocked"] == ["bench.json (L4 and send lag)"]
    assert out["datasets"]["advanced"]["summary"]["chain"] == "holds"


def test_a_file_that_fails_its_manifest_stops_the_analysis(tmp_path):
    root = synthetic_repo(tmp_path / "repo")
    copy = root / e1.COPIES / "formal-20261009-coreb-uniform" / "bench-uniform-8192-r2-c8.json"
    copy.write_text(copy.read_text().replace("0.", "1.", 1))
    with pytest.raises(e1.StageE1Error, match="raw_manifest"):
        analyze(root)
    root = synthetic_repo(tmp_path / "repo2")
    copy = root / e1.COPIES / "formal-20261011-adv" / "G-8192-uniform-r1.spans.jsonl"
    copy.write_text(copy.read_text() + "\n")
    with pytest.raises(e1.StageE1Error, match="raw_manifest"):
        analyze(root)


def test_a_file_fetch_raw_set_aside_as_a_mismatch_stops_the_analysis(tmp_path):
    root = synthetic_repo(tmp_path / "repo")
    copy = root / e1.COPIES / "formal-20261011-adv" / "K3-8192-uniform-r1.spans.jsonl"
    copy.rename(copy.with_name(copy.name + ".sha256-mismatch"))                    # what fetch_raw.sh's exit 5 leaves
    with pytest.raises(e1.StageE1Error, match="failed its raw_manifest.json sha256 when it was fetched"):
        analyze(root)
