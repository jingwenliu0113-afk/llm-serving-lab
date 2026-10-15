"""Tests for benchmarks/c0_analysis.py: the descriptive C0 rules of decision 028 (8K two-mode probe; exploratory,
never a hypothesis test). The classifier is checked on the committed Advanced derived files (B0 / G / K3 at 8192
uniform, read-only), where 028 counts 8 slip episodes, and on synthetic closed-loop cycles for every rule and
boundary."""

import json
import math
from pathlib import Path

import pytest

import c0_analysis as ca
import core_b_analysis as ba

ROOT = Path(__file__).resolve().parent.parent
ADV = ROOT / "results" / "advanced" / "formal-20261011-adv"
# 028 「已知線索」 4: (start group k, position, length) of every episode in the nine 8192-uniform runs
ADV_EPISODES = {
    "B0-8192-uniform-r1": [], "B0-8192-uniform-r2": [(16, 1, 7), (23, 1, 7)], "B0-8192-uniform-r3": [],
    "G-8192-uniform-r1": [(8, 1, 7)], "G-8192-uniform-r2": [], "G-8192-uniform-r3": [],
    "K3-8192-uniform-r1": [(16, 1, 7), (24, 1, 7)], "K3-8192-uniform-r2": [(16, 1, 7)],
    "K3-8192-uniform-r3": [(11, 1, 7), (24, 1, 7)],
}


def adv_requests(tag):
    recs = sorted((json.loads(x) for x in (ADV / "derived" / f"{tag}-requests.jsonl").read_text().splitlines()),
                  key=lambda r: r["idx"])
    assert [r["idx"] for r in recs] == list(range(200))
    return [r["client_ttft_ms"] for r in recs], [r["prompt_tokens"] for r in recs]


def adv_classified(tag):
    ttfts, lens = adv_requests(tag)
    return ttfts, ca.classify_bench(ttfts, C=8, L=8192, input_lens=lens)


def adv_rows():
    return {r["tag"]: r for r in (json.loads(x) for x in (ADV / "rows.jsonl").read_text().splitlines())}


def cycle(C=8, N=200, slips=(), shift=340.0):
    """Closed-loop TTFTs (ms) with period C: position 0 short, 1..C-2 long, C-1 a little shorter; idx < C is the
    first wave; each slipped group k shifts its positions 1..C-1 by `shift` (028 線索 2 / 3)."""
    base = [1485.0] + [2477.0] * (C - 2) + [2320.0]
    t = [base[i % C] for i in range(N)]
    for i in range(1, C):
        t[i] = 2800.0 + 1200.0 * i
    for k in slips:
        for p in range(1, C):
            t[C * k + p] += shift
    return t


# ------------------------------------------------------------------ the Advanced fixtures (028 線索 4 / 7) --

def test_advanced_fixtures_hold_exactly_the_8_episodes_of_028():
    total = 0
    for tag, want in ADV_EPISODES.items():
        _, got = adv_classified(tag)
        assert [(e["k"], e["position"], e["length"]) for e in got["episodes"]] == want, tag
        assert [e["start_idx"] for e in got["episodes"]] == [8 * k + 1 for k, _, _ in want]
        total += got["n_episodes"]
    assert total == 8


def test_slipped_count_equals_k_prom_of_the_queue_time_histogram():
    rows = adv_rows()
    for tag, want in ADV_EPISODES.items():
        _, got = adv_classified(tag)
        assert got["n_slipped"] == 7 * len(want)
        assert ca.k_prom(rows[tag]["scrape_delta"]) == got["n_slipped"], tag


def test_request_8k_plus_7_is_caught_by_its_position_baseline():
    """028: 8k+7 of a slipped group sits only 11-16 ms above a global m_A + 170 threshold, far above its own b_7."""
    for tag, want in ADV_EPISODES.items():
        ttfts, got = adv_classified(tag)
        for k, _, _ in want:
            i = 8 * k + 7
            assert i in got["slipped"]
            assert ttfts[i] - got["baselines_ms"][7] > 300
            assert 0 < ttfts[i] - (got["m_a_ms"] + ca.SLIP_MS) < 20


def test_advanced_modes_and_y_follow_the_episodes():
    rows = adv_rows()
    for tag, want in ADV_EPISODES.items():
        _, got = adv_classified(tag)
        assert got["y_ms"] == pytest.approx(rows[tag]["short_class_p95_ttft_ms"], abs=1e-6)
        assert got["mode"] == ("high" if want else "low"), tag
        assert (330 < got["y_ms"] - got["m_a_ms"] < 350) if want else (got["y_ms"] - got["m_a_ms"] < 10)


# ------------------------------------------------------------------ per-bench rules (synthetic) --

def test_baselines_are_per_position_medians_over_idx_at_least_c():
    for C in (4, 8, 9):
        t = cycle(C=C)
        got = ca.classify_bench(t, C=C, L=8192)
        assert got["baselines_ms"] == [1485.0] + [2477.0] * (C - 2) + [2320.0]  # the first wave is not in them
        assert got["m_a_ms"] == ba.percentile(t[C:], 50)
        assert got["slipped"] == [] and got["episodes"] == [] and got["mode"] == "low"


def test_slip_needs_strictly_more_than_170_ms_above_its_position_baseline():
    t = cycle()
    t[8 * 5 + 3] += 170.0
    t[8 * 6 + 3] += 170.001
    got = ca.classify_bench(t, C=8, L=8192)
    assert got["slipped"] == [8 * 6 + 3]
    assert got["episodes"] == [{"start_idx": 51, "k": 6, "position": 3, "length": 1}]


def test_the_first_c_dispatches_never_slip():
    t = cycle()
    t[0] += 5000.0  # the first wave is high by construction, idx 0 included
    got = ca.classify_bench(t, C=8, L=8192)
    assert got["slipped"] == [] and min(t[:8]) >= 1485.0


def test_episodes_are_maximal_consecutive_runs_across_group_boundaries():
    assert ca.episodes([17, 18, 19, 23, 24, 25, 40], C=8) == [
        {"start_idx": 17, "k": 2, "position": 1, "length": 3},
        {"start_idx": 23, "k": 2, "position": 7, "length": 3},
        {"start_idx": 40, "k": 5, "position": 0, "length": 1}]
    assert ca.episodes([], C=8) == []


def test_a_slipped_group_is_one_episode_of_length_c_minus_1():
    got = ca.classify_bench(cycle(slips=(8, 16)), C=8, L=8192)
    assert got["episodes"] == [{"start_idx": 65, "k": 8, "position": 1, "length": 7},
                               {"start_idx": 129, "k": 16, "position": 1, "length": 7}]
    assert got["n_slipped"] == 14 and got["n_episodes"] == 2


def test_mode_is_tested_in_the_fixed_order_other_high_low():
    assert ca.mode(3200.0, 2477.0) == "other"   # 723 > 600: "other" even though > 170
    assert ca.mode(3077.0, 2477.0) == "high"    # exactly 600 is not > 600
    assert ca.mode(3077.01, 2477.0) == "other"
    assert ca.mode(2647.0, 2477.0) == "low"     # exactly 170 is not > 170
    assert ca.mode(2647.01, 2477.0) == "high"
    assert ca.mode(2400.0, 2477.0) == "low"


def test_y_is_core_b_short_class_p95_and_p95_p50_exclude_the_first_c():
    t = cycle(slips=(8,))
    lens = [8064] * 200
    got = ca.classify_bench(t, C=8, L=8192, input_lens=lens)
    assert got["y_ms"] == pytest.approx(ba.short_class_p95_ttft_ms({"ttfts": [x / 1000 for x in t],
                                                                    "input_lens": lens}, L=8192))
    assert got["p95_excl_first_c_ms"] == ba.percentile(t[8:], 95)
    assert got["p50_excl_first_c_ms"] == ba.percentile(t[8:], 50)
    assert got["y_ms"] > got["p95_excl_first_c_ms"]  # the first wave sits in Y


def test_y_keeps_only_the_short_class():
    t = cycle()
    lens = [8064] * 200
    lens[50] = 32640
    t[50] = 99999.0
    got = ca.classify_bench(t, C=8, L=8192, input_lens=lens)
    assert got["y_ms"] == pytest.approx(ba.short_class_p95_ttft_ms({"ttfts": [x / 1000 for x in t],
                                                                    "input_lens": lens}, L=8192))


def test_tpot_excludes_the_first_c():
    reqs = [{"idx": i, "ttft_ms": 1000.0, "latency_ms": 1000.0 + 127 * (90.0 if i < 8 else 80.0), "output_len": 128}
            for i in range(200)]
    assert ca.tpot_excl_first_c_ms(reqs, C=8) == pytest.approx(80.0)
    reqs[20]["output_len"] = 1  # no TPOT for a single-token output
    assert ca.tpot_excl_first_c_ms(reqs, C=8) == pytest.approx(80.0)


def test_k_prom_is_the_0_3_to_0_5_s_bucket_delta_when_the_histogram_exists():
    delta = {"vllm:request_queue_time_seconds": {"buckets": {"0.3": 180.0, "0.5": 194.0, "+Inf": 208.0}}}
    assert ca.k_prom(delta) == 14
    assert ca.k_prom({"vllm:request_queue_time_seconds": None}) is None
    assert ca.k_prom({"vllm:request_queue_time_seconds": {"buckets": {"0.5": 1.0}}}) is None
    assert ca.k_prom(None) is None and ca.k_prom({}) is None


# ------------------------------------------------------------------ telemetry --

def smi(t, sm=2800.0, pstate="P1", temp=60.0, reasons=0x400):
    return {"t": t, "pstate": pstate, "sm_mhz": sm, "mem_mhz": 14001.0, "power_w": 250.0, "temp_c": temp,
            "util_pct": 100.0, "mem_used_mib": 14000.0, "event_reasons": reasons}


def test_bench_telemetry_reads_only_samples_inside_the_bench_window():
    samples = [smi(99.0, sm=100.0, pstate="P8", temp=90.0, reasons=0x40), smi(100.0, sm=2700.0),
               smi(101.0, sm=2900.0, pstate="P2", reasons=0x404), smi(102.0, sm=None, temp=None, reasons=None),
               smi(103.0, sm=2800.0, temp=71.0), smi(104.0, sm=100.0, temp=95.0, reasons=0x80)]
    got = ca.bench_telemetry(samples, 100.0, 103.0)
    assert got == {"n_samples": 4, "sm_min_mhz": 2700.0, "sm_median_mhz": 2800.0, "pstates": ["P1", "P2"],
                   "event_reasons_or": 0x404, "throttle_or": 0x4, "temp_max_c": 71.0}
    assert ca.bench_telemetry([], 0.0, 1.0)["n_samples"] == 0


def test_throttle_mask_is_the_nvml_slowdown_reasons():
    # SwPowerCap 0x4, HwSlowdown 0x8, SwThermal 0x20, HwThermal 0x40, HwPowerBrake 0x80 (nvml.h)
    assert ca.THROTTLE_MASK == 0x4 | 0x8 | 0x20 | 0x40 | 0x80
    assert not ca.THROTTLE_MASK & (0x1 | 0x2 | 0x10 | 0x100 | 0x400)


# ------------------------------------------------------------------ session rules --

def chi2_sf(x, df):
    """Upper tail of the chi-square distribution in closed form (integer df)."""
    if df % 2 == 0:
        return math.exp(-x / 2) * sum((x / 2) ** i / math.factorial(i) for i in range(df // 2))
    tail, term = math.erfc(math.sqrt(x / 2)), math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    for i in range(1, (df + 1) // 2):
        tail += term
        term *= x / (2 * i + 1)
    return tail


def test_hard_coded_chi_square_critical_values_are_the_0_95_quantiles():
    assert sorted(ca.CHI2_95) == [1, 2, 3, 4, 5]
    assert ca.CHI2_95[5] == 11.07 or ca.CHI2_95[5] == pytest.approx(11.0705, abs=1e-3)  # 028: 11.07 at 6 boots
    for df, x in ca.CHI2_95.items():
        assert chi2_sf(x, df) == pytest.approx(0.05, abs=2e-4), df


def test_binomial_tail_reproduces_the_028_clue_6_probability():
    assert ca.binomial_tail(8, 6, 1 / 8) == pytest.approx(8.5e-5, rel=0.01)
    assert ca.binomial_tail(8, 0, 1 / 8) == pytest.approx(1.0)
    assert ca.binomial_tail(0, 0, 1 / 8) == pytest.approx(1.0)


def boots_with(ns, valid=4):
    return [{"boot": b, "n_valid": valid, "n_b": n} for b, n in enumerate(ns, 1)]


def outcome(boots, stop_rule=None, complete=True):
    return ca.session_outcome(boots, stop_rule=stop_rule, complete=complete)


def test_r0_when_no_episode_at_all():
    got = outcome(boots_with([0] * 6))
    assert got["outcome"] == "R0" and got["branch"] == "A" and got["total_episodes"] == 0


def test_rboot_by_dispersion_above_the_chi_square_critical_value():
    got = outcome(boots_with([0, 0, 0, 0, 0, 12]))
    assert got["outcome"] == "Rboot" and got["branch"] == "B"
    assert got["n_bar"] == 2 and got["D"] == pytest.approx(60.0) and got["df"] == 5 and got["critical"] == ca.CHI2_95[5]


def test_rrun_when_dispersion_is_small_and_no_boot_is_empty():
    got = outcome(boots_with([3, 4, 3, 4, 4, 3]))
    assert got["outcome"] == "Rrun" and got["branch"] == "A"
    assert got["D"] == pytest.approx(6 * 0.25 / 3.5) and not got["zero_rule"]


def test_rboot_by_an_empty_boot_when_the_session_mean_is_at_least_one_per_bench():
    got = outcome(boots_with([0, 5, 5, 5, 5, 4]))  # 24 episodes / 24 benches = 1.0 (boundary, >=)
    assert got["D"] == pytest.approx(5.0) and got["D"] < got["critical"]
    assert got["zero_rule"] and got["outcome"] == "Rboot"
    got = outcome(boots_with([0, 5, 5, 5, 5, 3]))  # 23 / 24 < 1
    assert not got["zero_rule"] and got["outcome"] == "Rrun"


def test_dispersion_uses_only_boots_with_four_valid_benches():
    boots = boots_with([3] * 5) + [{"boot": 6, "n_valid": 3, "n_b": 30}]
    got = outcome(boots)
    assert got["qualifying_boots"] == [1, 2, 3, 4, 5] and got["df"] == 4 and got["D"] == 0.0
    assert got["total_episodes"] == 45 and got["outcome"] == "Rrun"


def test_empty_boot_rule_reads_every_boot_with_a_valid_bench_not_only_the_d_boots():
    """028: 「只算 4 個 bench 都有效的 boot」 qualifies D alone; an empty boot with 3 valid benches still counts."""
    boots = [{"boot": 1, "n_valid": 3, "n_b": 0}] + [{"boot": b, "n_valid": 4, "n_b": 6} for b in range(2, 7)]
    got = outcome(boots)  # 30 episodes / 23 valid benches = 1.30 per bench
    assert got["qualifying_boots"] == [2, 3, 4, 5, 6] and got["D"] == 0.0
    assert got["zero_rule"] and got["outcome"] == "Rboot" and got["branch"] == "B"
    # a boot without any valid bench (its server never started) observed nothing: never an empty boot
    boots[0] = {"boot": 1, "n_valid": 0, "n_b": 0}
    got = outcome(boots)
    assert not got["zero_rule"] and got["outcome"] == "Rrun"


def test_indeterminate_when_dispersion_cannot_be_computed():
    got = outcome([{"boot": 1, "n_valid": 4, "n_b": 2}] + [{"boot": b, "n_valid": 3, "n_b": 2} for b in range(2, 7)])
    assert got["D"] is None and got["outcome"] == "indeterminate" and got["branch"] == "C"


def test_stops_and_incomplete_sessions_map_to_branch_c_before_any_r_outcome():
    assert outcome(boots_with([0] * 5), stop_rule="cap", complete=False)["outcome"] == "cap"
    assert outcome(boots_with([0] * 4), stop_rule="failed_boots", complete=False)["outcome"] == "failed"
    assert outcome(boots_with([3] * 3), stop_rule="temperature", complete=False)["outcome"] == "indeterminate"
    assert outcome(boots_with([0] * 5), complete=False)["outcome"] == "indeterminate"
    for stop in ("cap", "failed_boots", "temperature"):
        assert outcome(boots_with([0] * 6), stop_rule=stop, complete=False)["branch"] == "C"


def test_branch_d_is_never_computed():
    assert ca.BRANCH == {"R0": "A", "Rrun": "A", "Rboot": "B", "cap": "C", "failed": "C", "indeterminate": "C"}
    assert "D" not in ca.BRANCH.values()


# ------------------------------------------------------------------ a whole session --

T0 = 1_800_000_000.0


def write_session(tmp_path, eps, *, stage="C0", stopped=None, invalid=(), startup_failed=(), hot=(), slow_clock=(),
                  p50_shift=None, boots_planned=6):
    """eps[b][j]: the slipped groups of bench j + 1 of boot b + 1. invalid / hot / slow_clock: (boot, bench) pairs;
    startup_failed: boots whose server never started (their benches are rows with ok false, not run)."""
    d = tmp_path / "explore-c0-20261016"
    (d / "derived").mkdir(parents=True)
    session = {"session_id": d.name, "formal": False, "stage": stage, "purpose": "exploratory", "decision": "028",
               "boots_planned": boots_planned, "stopped": stopped, "boots": {}}
    plan = {"stage": "C0", "L": 8192, "C": 8, "N": 200, "boots": boots_planned, "benches_per_boot": 4}
    rows, t = [], T0
    for b, benches in enumerate(eps, 1):
        smi_rows, up = [], b not in startup_failed
        for j, slips in enumerate(benches, 1):
            tag = f"b{b}-n{j}"
            row = {"boot": b, "bench": j, "tag": tag, "ok": up and (b, j) not in invalid, "reasons": [],
                   "t_start": t, "t_end": t + 320.0, "output_throughput": 80.4}
            if up:
                ttfts = [x + (p50_shift or {}).get(b, 0.0) for x in cycle(slips=slips)]
                reqs = [{"idx": i, "input_len": 8064, "output_len": 128, "ttft_ms": x, "latency_ms": x + 127 * 80.0}
                        for i, x in enumerate(ttfts)]
                (d / "derived" / f"{tag}-requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
                row.update(derived=f"derived/{tag}-requests.jsonl", scrape_delta={
                    "vllm:request_queue_time_seconds": {"buckets": {"0.3": 194.0 - 7 * len(slips), "0.5": 194.0}}})
                sm = 2500.0 if (b, j) in slow_clock else 2800.0
                reasons = 0x420 if (b, j) in hot else 0x400
                smi_rows += [smi(t + s, sm=sm, reasons=reasons) for s in range(0, 321, 10)]
            rows.append(row)
            t += 330.0
        session["boots"][f"b{b}"] = {"startup_ok": up, "smi": f"derived/b{b}-smi.jsonl" if up else None}
        if up:
            (d / "derived" / f"b{b}-smi.jsonl").write_text("".join(json.dumps(s) + "\n" for s in smi_rows))
        t += 60.0
    (d / "session.json").write_text(json.dumps(session))
    (d / "plan.json").write_text(json.dumps(plan))
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def even_eps():
    """3 episodes per boot, spread over the benches (consistent with run level)."""
    return [[[8], [16], [], [24]], [[16], [], [8, 24], []], [[], [8], [16], [11]],
            [[24], [8], [], [16]], [[16, 24], [], [8], []], [[], [], [16], [8, 24]]]


def test_analyse_session_end_to_end(tmp_path):
    a = ca.analyse_session(write_session(tmp_path, even_eps()))
    assert a["stage"] == "C0" and a["decision"] == "028" and a["formal"] is False
    assert len(a["benches"]) == 24 and all(x["valid"] for x in a["benches"])
    first = next(x for x in a["benches"] if x["tag"] == "b1-n1")
    assert first["episodes"] == [{"start_idx": 65, "k": 8, "position": 1, "length": 7}]
    assert first["k_prom"] == 7 and first["k_prom_matches"] is True and first["mode"] == "high"
    assert first["telemetry"]["n_samples"] == 33 and first["telemetry"]["sm_median_mhz"] == 2800.0
    assert [b["n_b"] for b in a["boots"]] == [3] * 6
    s = a["session"]
    assert s["outcome"] == "Rrun" and s["branch"] == "A" and s["total_episodes"] == 18 and s["D"] == 0.0
    rep = a["reported"]
    assert rep["both_modes_boots"] == 6
    assert rep["start_k"]["bench1"] == [8, 16, 16, 24, 24] and len(rep["start_k"]["bench2_4"]) == 13
    assert rep["k_8_16_24"]["episodes"] == 18 and rep["k_8_16_24"]["at"] == 17
    assert rep["k_8_16_24"]["chance"] == pytest.approx(1 / 8)
    assert rep["high_share"]["high"] == 15 and rep["high_share"]["valid"] == 24
    assert rep["high_share"]["phase1"] == [8, 12]
    assert a["flags"]["Fhw"]["flagged"] == [] and a["flags"]["Foff"]["flag"] is False
    assert "never computed" in a["branch_d"]


def test_rboot_session_and_failed_boot_accounting(tmp_path):
    eps = [[[8, 16], [16, 24], [8], [24]]] + [[[], [], [], []] for _ in range(5)]
    a = ca.analyse_session(write_session(tmp_path, eps, invalid={(6, 1)}))
    assert [b["n_b"] for b in a["boots"]] == [6, 0, 0, 0, 0, 0]
    assert a["boots"][5]["n_valid"] == 3 and a["boots"][5]["failed"] is False
    s = a["session"]
    assert s["qualifying_boots"] == [1, 2, 3, 4, 5] and s["outcome"] == "Rboot" and s["branch"] == "B"


def test_invalid_benches_are_excluded_and_two_invalid_fail_the_boot(tmp_path):
    eps = [[[8], [8], [8], [8]] for _ in range(6)]
    a = ca.analyse_session(write_session(tmp_path, eps, invalid={(2, 1), (2, 2)}, startup_failed={3}))
    b2, b3 = a["boots"][1], a["boots"][2]
    assert b2["n_valid"] == 2 and b2["n_b"] == 2 and b2["failed"] is True
    assert b3["startup_ok"] is False and b3["n_valid"] == 0 and b3["failed"] is True
    assert a["session"]["valid_benches"] == 18 and a["session"]["total_episodes"] == 18


def test_flags_fhw_and_foff(tmp_path):
    a = ca.analyse_session(write_session(tmp_path, even_eps(), hot={(1, 1)}, slow_clock={(2, 2)},
                                         p50_shift={3: 30.0}))
    fhw = a["flags"]["Fhw"]
    assert fhw["flagged"] == ["b1-n1", "b2-n2"]
    assert fhw["table"] == {"flag_with_episodes": 1, "flag_without_episodes": 1, "clean_with_episodes": 14,
                            "clean_without_episodes": 8}
    foff = a["flags"]["Foff"]
    assert foff["flag"] is True and foff["spread_pct"] == pytest.approx((2507.0 / 2477.0 - 1) * 100)


def test_stopped_session_is_branch_c(tmp_path):
    a = ca.analyse_session(write_session(tmp_path, even_eps()[:5], stopped={"rule": "cap", "reason": "x"}))
    assert a["session"]["outcome"] == "cap" and a["session"]["branch"] == "C"


@pytest.mark.parametrize("stage", ["C1", None, "c0"])
def test_refuses_any_session_whose_stage_is_not_c0(tmp_path, stage):
    d = write_session(tmp_path, even_eps(), stage=stage)
    with pytest.raises(ca.AnalysisError, match="stage"):
        ca.analyse_session(d)


def test_main_writes_analysis_json(tmp_path, capsys):
    d = write_session(tmp_path, even_eps())
    assert ca.main(["--session", str(d)]) == 0
    a = json.loads((d / "analysis.json").read_text())
    assert a["session"]["outcome"] == "Rrun"
    assert json.loads(capsys.readouterr().out)["branch"] == "A"
    s = write_session(tmp_path / "other", even_eps(), stage="C1")
    assert ca.main(["--session", str(s)]) == 2 and not (s / "analysis.json").exists()
