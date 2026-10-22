"""Tests for benchmarks/stage_e1.py (decision 041 v3 §1): the slip-chain links L1-L5 on synthetic closed-loop benches,
continuous slips, the chain summary, the manifest-checked inputs and the freeze gate."""

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

import stage_e1 as e1

ROOT = Path(__file__).resolve().parent.parent
CFG = tomllib.loads((ROOT / "configs" / "stage_e1.toml").read_text())
FROZEN = {**CFG, "freeze": {"e1": True}}
PERIOD = 0.416


def synth(slips=(5,), n=200, spike=True, extra_step=True):
    """A closed-loop 8K bench: one request every PERIOD s; a slip at group k moves 8k+1..8k+7 326 ms earlier
    (the k - 1 group finished early), adds 340 ms of TTFT and 348 ms of Q to them, a 15 ms F spike to 8k and one more
    decode-only step before 8k's first prefill chunk."""
    arrival = [i * PERIOD for i in range(n)]
    for k in slips:
        for j in range(1, 8):
            arrival[8 * k + j] -= 0.326
    ttft = [1000.0] * n
    q = [100.0] * n
    f = [26.0] * n
    for k in slips:
        for j in range(1, 8):
            ttft[8 * k + j] += 340.0
            q[8 * k + j] += 348.0
        if spike:
            f[8 * k] = 42.0
    completion = [arrival[i + 8] if i + 8 < n else arrival[i] + 8 * PERIOD for i in range(n)]
    lat = [(c - a) * 1000 for c, a in zip(completion, arrival)]
    steps = []
    for g in range(n // 8):
        a = arrival[8 * g]
        offsets = (0.012, 0.027, 0.042, 0.087) if (g in slips and extra_step) else (0.012, 0.027, 0.072)
        steps += [{"t": a + o, "ctx_tok": 0} for o in offsets] + [{"t": a + 0.3, "ctx_tok": 2048}]
    return {"ttft": ttft, "arrival": arrival, "lat": lat, "q": q, "f": f, "steps": steps, "completion": completion}


def run(d, **kw):
    args = dict(lat_ms=d["lat"], times_s=d["arrival"], cfg=CFG, F_ms=d["f"], Q_ms=d["q"], steps=d["steps"],
                arrivals_s=d["arrival"])
    args.update(kw)
    return e1.bench_episodes(d["ttft"], **args)


def test_every_link_holds_on_a_textbook_slip():
    out = run(synth())
    (ep,) = out["episodes"]
    assert (ep["start_idx"], ep["k"], ep["length"], ep["continuous"]) == (41, 5, 7, False)
    assert ep["links"] == {"L1": True, "L2": True, "L3": True, "L4": True, "L5": True}
    v = ep["values"]
    assert v["gap_ms"] == pytest.approx(90.0) and v["L3_dev_ms"] == pytest.approx(-326.0)
    assert (v["n_dec"], v["n_dec_median"]) == (4, 3) and v["dQ_ms"] == pytest.approx(348.0)
    assert out["l1_in_normal_groups"] == {"groups": 22, "spikes": 0}


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


def test_a_continuous_slip_counts_once_and_skips_l3():
    out = run(synth(slips=(8, 9)))
    eps = {e["k"]: e for e in out["episodes"]}
    assert eps[9]["continuous"] is True and eps[9]["links"]["L3"] is None and eps[8]["continuous"] is False
    s = e1.chain_summary([out], CFG)
    assert s["episodes"] == 1 and s["continuous"] == 1 and s["start_idx"] == {65: 1, 73: 1}


def test_baselines_skip_g0_the_last_group_slips_and_their_precursors():
    assert e1.baseline_groups(200, {5}) == [g for g in range(1, 24) if g not in (4, 5)]
    assert e1.baseline_groups(200, set())[-1] == 23


def test_l1_spikes_in_normal_groups_are_counted():
    d = synth()
    d["f"][8 * 12] = 50.0
    s = e1.chain_summary([run(d)], CFG)
    assert s["l1_rate_in_normal_groups"] == pytest.approx(1 / 22) and s["chain"] == "incomplete"   # > 2 %


def test_send_lags_follow_the_closed_loop():
    starts = [0.0] * 8 + [1.0 + 0.0002 * i for i in range(8)]
    completions = [1.0 + 0.0002 * i - 0.0001 for i in range(8)] + [9.0] * 8
    assert max(e1.send_lags_ms(starts, completions)) == pytest.approx(0.1)


def test_the_config_is_028s_threshold_and_unfrozen():
    assert e1.config_problems(CFG) == [] and CFG["freeze"]["e1"] is False
    assert e1.config_problems({**CFG, "e1": {**CFG["e1"], "slip_ms": 150.0}})
    assert {Path(f).name for f in e1.HARNESS_FILES} == {"stage_e1.py", "c0_analysis.py"}


def test_the_fetch_list_is_what_041_names():
    (adv_sub, adv_session, adv), (c0_sub, c0_session, c0) = e1.FETCH
    assert (adv_sub, adv_session, c0_sub, c0_session) == ("advanced", "formal-20261011-adv", "capacity",
                                                          "explore-c0-20261016")
    assert len(adv) == 15 * 2 + 6 and "T-8192-uniform-r1.steps.jsonl" in adv and "B0-8192-uniform-r1.steps.jsonl" not in adv
    assert len(c0) == 24


# ---------------------------------------------------------------- the whole analysis on a synthetic repo

def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _jsonl(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


def synthetic_repo(root, *, fetched=True):
    manifests = {e1.ADV: {}, e1.C0: {}, e1.CORE: {}}

    def raw(session, rel, text, copy=None):
        p = _write(root, copy or (session / "raw" / rel), text)
        manifests[session][f"raw/{rel}"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    for tag in e1.ADV_TAGS:
        cond, r = tag.split("-")[0], int(tag[-1])
        d = synth(slips=(5,) if cond in e1.SLIP_CONDS else ())
        rows = [{"idx": i, "client_ttft_ms": d["ttft"][i], "arrival": d["arrival"][i], "Q_ms": d["q"][i],
                 "F_ms": d["f"][i]} for i in range(200)]
        _write(root, e1.ADV / "derived" / f"{tag}-requests.jsonl", _jsonl(rows))
        steps = _jsonl(d["steps"])
        if cond in e1.SLIP_CONDS:
            raw(e1.ADV, f"{tag}.steps.jsonl", steps, copy=e1.COPIES / "formal-20261011-adv" / f"{tag}.steps.jsonl")
        elif fetched:
            raw(e1.ADV, f"{tag}.steps.jsonl", steps)
        else:
            manifests[e1.ADV][f"raw/{tag}.steps.jsonl"] = {"sha256": "0" * 64}
        prefix = f"adv-{cond}-8192-uniform-r{r}-"
        spans = [{"id": f"cmpl-{prefix}{i}-0", "arrival": d["arrival"][i], "e2e": d["completion"][i] - d["arrival"][i]}
                 for i in range(200)]
        if fetched:
            raw(e1.ADV, f"{tag}.spans.jsonl", _jsonl(spans))
            raw(e1.ADV, f"{tag}.bench.json", "{}")
    for b in e1.C0_BENCHES:
        d = synth(slips=(11,) if b == "b1-n2" else ())
        _write(root, e1.C0 / "derived" / f"{b}-requests.jsonl",
               _jsonl([{"idx": i, "ttft_ms": d["ttft"][i], "latency_ms": d["lat"][i]} for i in range(200)]))
        if fetched:
            raw(e1.C0, f"{b}.bench.json", json.dumps({"start_times": d["arrival"]}))
    for r in (1, 2, 3):
        d = synth(slips=(8, 9) if r == 1 else (5,))
        name = f"bench-uniform-8192-r{r}-c8.json"
        raw(e1.CORE, name, json.dumps({"ttfts": [x / 1000 for x in d["ttft"]], "latencies": [x / 1000 for x in d["lat"]],
                                       "start_times": d["arrival"]}),
            copy=e1.COPIES / "formal-20261009-coreb-uniform" / name)
    for session, files in manifests.items():
        _write(root, session / "raw_manifest.json", json.dumps({"files": files}))
    return root


def test_the_analysis_reports_each_dataset_from_manifest_checked_inputs(tmp_path):
    root = synthetic_repo(tmp_path / "repo")
    out = e1.analyze(root, FROZEN, fetched=True)
    assert out["descriptive_only"] is True and out["harness_sha256"] == e1.harness_sha256()
    adv = out["datasets"]["advanced"]
    assert adv["summary"]["chain"] == "holds" and adv["summary"]["episodes"] == 9
    assert all(r["completion_source"] == "e2e" for r in adv["runs"])
    assert {r["run"].split("-")[0] for r in adv["runs"] if r["episodes"]} == set(e1.SLIP_CONDS)
    assert out["datasets"]["c0"]["summary"]["start_idx"] == {89: 1}
    core = out["datasets"]["core"]
    assert core["summary"]["continuous"] == 1 and all(r["send_lag_ms_max"] < 1 for r in core["runs"])


def test_without_the_fetched_raw_advanced_falls_back_to_the_proxy(tmp_path):
    root = synthetic_repo(tmp_path / "repo", fetched=False)
    out = e1.analyze(root, FROZEN, fetched=False)
    adv = out["datasets"]["advanced"]["runs"]
    assert all(r["completion_source"] == "proxy" for r in adv)
    assert out["datasets"]["c0"]["runs"][0]["episodes"] == [] and "send_lag_ms_max" not in out["datasets"]["c0"]["runs"][0]


def test_no_real_input_before_the_freeze_and_a_tampered_input_is_refused(tmp_path):
    root = synthetic_repo(tmp_path / "repo")
    with pytest.raises(e1.StageE1Error, match="freeze.e1"):
        e1.analyze(root, CFG, fetched=True)
    copy = root / e1.COPIES / "formal-20261009-coreb-uniform" / "bench-uniform-8192-r2-c8.json"
    copy.write_text(copy.read_text().replace("0.", "1.", 1))
    with pytest.raises(e1.StageE1Error, match="raw_manifest"):
        e1.analyze(root, FROZEN, fetched=True)
