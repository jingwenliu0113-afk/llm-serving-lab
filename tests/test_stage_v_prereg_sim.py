"""Tests for benchmarks/stage_v_prereg_sim.py (decision 044 V4 「凍結前的設計檢查」): the closed-loop strict FIFO of
weighted-v1 (budget, class caps, head-of-line blocking, the next idx a turnaround after the (k - C + 1)-th release),
the duration pools from c1b's valid Phase C runs at C_PROD (the start-up wave and W), the client turnaround from D-C's
closed loop through the gateway, U_W / N_W / (j) from stage_v_analysis's own functions, the variants, the 0.93 x
budget design check, and the committed output, which must regenerate byte-identically from the committed files."""

import ast
import copy
import hashlib
import json
import random
import statistics
import tomllib
from fractions import Fraction
from pathlib import Path

import pytest

import stage_d_plan as sd
import stage_v_analysis as sva
import stage_v_datasets as svd
import stage_v_plan as sv
import stage_v_prereg_sim as vsim

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_v.toml").read_text())
SD_RAW = (ROOT / "configs" / "stage_d.toml").read_bytes()
SD_CFG = tomllib.loads(SD_RAW.decode())
ADM = sv.build_plan(CFG, PINS, SD_CFG, session_name="V-ADM", pins_sha256="p" * 64, stage_v_sha256="v" * 64,
                    stage_d_sha256=hashlib.sha256(SD_RAW).hexdigest())
COMMITTED = ROOT.joinpath(*vsim.OUT)
STUB = {"U2K": {"first": [5.0], "steady": [1.0]}, "U8K": {"first": [7.0], "steady": [2.0]}}


def sim(demands, *, C, budget, caps, durations, turnaround=0.0):
    """vsim.simulate with fixed durations ({class name or idx: s}) and a fixed turnaround."""
    def duration(i, sent):
        return durations[i] if i in durations else durations[demands[i][0]]
    return vsim.simulate(demands, C=C, budget=budget, caps=caps, duration=duration, turnaround=lambda: turnaround)


# ------------------------------------------------------------------ the closed-loop strict FIFO --

def test_a_head_that_does_not_fit_blocks_everyone_behind_it():
    """budget 4: idx 1 (4 units) waits for idx 0 and idx 2 (1 unit, fits) may not overtake it (044 HOL)."""
    d = [("a", 1), ("b", 4), ("a", 1), ("a", 1), ("a", 1)]
    out = sim(d, C=3, budget=4, caps={"a": 4, "b": 1}, durations={"a": 1.0, "b": 2.0}, turnaround=0.25)
    assert out["times"] == [(0.0, 0.0, 1.0), (0.0, 1.0, 3.0), (0.0, 3.0, 4.0), (1.25, 3.0, 4.0),
                            (3.25, 3.25, 4.25)]
    assert out["peak_queue"] == 2


def test_a_class_at_its_cap_waits_with_budget_to_spare():
    out = sim([("b", 4), ("b", 4)], C=2, budget=8, caps={"b": 1}, durations={"b": 2.0})
    assert out["times"] == [(0.0, 0.0, 2.0), (0.0, 2.0, 4.0)]
    assert out["peak_queue"] == 1


def test_idx_k_arrives_a_turnaround_after_the_k_minus_c_plus_1_th_release():
    """The bench's semaphore wakes its waiters in idx order: idx 2 follows the first release (idx 1's), idx 3 the
    second (idx 2's), whichever idx ended."""
    out = sim([("a", 1)] * 4, C=2, budget=2, caps={"a": 2}, durations={0: 3.0, 1: 1.0, 2: 1.0, 3: 1.0},
              turnaround=0.25)
    assert out["times"] == [(0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (1.25, 1.25, 2.25), (2.5, 2.5, 3.5)]
    assert out["peak_queue"] == 0


def test_the_first_wave_arrives_at_zero_in_idx_order():
    out = sim([("a", 1), ("b", 4), ("a", 1)], C=3, budget=4, caps={"a": 4, "b": 1}, durations={"a": 1.0, "b": 1.0})
    assert [t[0] for t in out["times"]] == [0.0, 0.0, 0.0] and out["times"][1][1] == 1.0


# ------------------------------------------------------------------ a W run on the plan --

def run(variant="committed", point="W-90", block=1, pools=STUB, gaps=(0.001,)):
    return vsim.simulate_run(ADM, point, block, rng=random.Random(0), pools=pools, gaps=list(gaps), variant=variant)


def test_a_w_run_follows_its_pinned_sequence_and_the_policy():
    for point in vsim.POINTS:
        x = ADM["points"][point]
        for block in (1, 5):
            reqs = run(point=point, block=block)["reqs"]
            rows = svd.spec(x["datasets"][str(block)])
            assert [q["idx"] for q in reqs] == list(range(x["N"])) == list(range(len(rows)))
            assert [q["units"] for q in reqs] == [4 if p == svd.PROMPT_8K else 1 for p, _ in rows]
            assert reqs[0]["gw_arrival_s"] == 0 and all(q["gw_arrival_s"] <= q["gw_sent_s"] <= q["gw_released_s"]
                                                        for q in reqs)
            assert all(round(q[k], 6) == q[k] for q in reqs for k in sva.TIMES)     # gateway_requests' rounding
            iv = sva.in_vllm(reqs, C=x["C"], N=x["N"])
            assert sva.inflight_problems(iv, x=x) == []


def test_the_run_takes_the_plan_s_budget_and_caps():
    """Plans with budget 16, or an 8K cap of 2 (never 044's: only the wiring): the simulation keeps to each, so
    stage_v_analysis finds (j) intact on that plan, and fills it (the budget, the cap)."""
    for key, value in (("budget", 16), ("cap", 2)):
        p = copy.deepcopy(ADM)
        adm = p["points"]["W-50"]["admission_params"]
        if key == "budget":
            adm["budget"] = value
        else:
            adm["classes"][1]["max_concurrency"] = value
        out = vsim.simulate_run(p, "W-50", 1, rng=random.Random(0), pools=STUB, gaps=[0.001], variant="committed")
        iv = sva.in_vllm(out["reqs"], C=32, N=320)
        assert (iv["peak_units"] if key == "budget" else iv["peak_by_units"]["4"]) == value
        assert vsim.run_stats(p, "W-50", out)["u_w"] <= adm["budget"]


@pytest.mark.parametrize("variant,first,scale", [("committed", True, 1), ("steady_only", False, 1),
                                                 ("8k_x0.8", True, 0.8), ("8k_x1.25", True, 1.25)])
def test_durations_come_from_the_class_pool_of_the_first_wave_or_w(variant, first, scale):
    """Sent at 0 (the start-up wave): idx < C's pool, else W's (steady_only: always W's); 8K x the variant's scale."""
    reqs = run(variant)["reqs"]
    seen = set()
    for q in reqs:
        pool = STUB["U8K" if q["units"] == 4 else "U2K"]
        want = (pool["first"] if first and q["gw_sent_s"] == 0 else pool["steady"])[0] * (scale if q["units"] == 4
                                                                                          else 1)
        assert q["gw_released_s"] - q["gw_sent_s"] == pytest.approx(want, abs=2e-6)
        seen.add((q["units"], q["gw_sent_s"] == 0))
    assert seen == {(1, True), (1, False), (4, True), (4, False)}


def test_the_turnaround_follows_each_release():
    reqs = run(gaps=(0.5,))["reqs"]
    released = sorted(q["gw_released_s"] for q in reqs)
    assert [q["gw_arrival_s"] for q in reqs[32:]] == pytest.approx([r + 0.5 for r in released[:288]], abs=2e-6)


def test_run_stats_are_stage_v_analysis_s_estimator():
    x = ADM["points"]["W-50"]
    out = run(point="W-50")
    stats = vsim.run_stats(ADM, "W-50", out)
    iv = sva.in_vllm(out["reqs"], C=x["C"], N=x["N"])
    assert stats["u_w"] == iv["_u_w"] and stats["n_w"] == iv["_n_w"]
    assert stats["trivial"] is (not sva.nontrivial(iv, x=x, plan=ADM)["ok"])
    assert stats["peak_queue"] == out["peak_queue"] and stats["bench_s"] == max(q["gw_released_s"] for q in out["reqs"])


def test_a_simulated_run_breaking_j_is_an_error():
    out = run()
    p = copy.deepcopy(ADM)
    p["points"]["W-90"]["admission_params"]["budget"] = 16
    with pytest.raises(vsim.SimError, match=r"\(j\)"):
        vsim.run_stats(p, "W-90", out)


def test_a_trivial_run_is_counted():
    """Clients that wait 5 s between 1 s requests leave the budget mostly idle: every run's U_W < 28.8."""
    pools = {"U2K": {"first": [1.0], "steady": [1.0]}, "U8K": {"first": [1.0], "steady": [1.0]}}
    out = vsim.simulate_point(ADM, "W-50", pools=pools, gaps=[5.0], variant="committed", reps=1)
    assert out["runs"] == out["trivial_runs"] == 5 and all(b["trivial_runs"] == 1 for b in out["blocks"].values())
    assert out["u_w"]["max"] < 28.8
    out = vsim.simulate_point(ADM, "W-50", pools=pools, gaps=[0.0], variant="committed", reps=1)
    assert out["trivial_runs"] == 0 and out["u_w"]["min"] >= 28.8


def test_each_variant_point_and_block_has_its_own_seeded_random(monkeypatch):
    pools = {"U2K": {"first": [5.0, 6.0], "steady": [1.0, 1.1]}, "U8K": {"first": [7.0, 7.5], "steady": [2.0, 2.2]}}
    a = vsim.simulate_point(ADM, "W-90", pools=pools, gaps=[0.001, 0.002], variant="8k_x0.8", reps=2)
    assert a == vsim.simulate_point(ADM, "W-90", pools=pools, gaps=[0.001, 0.002], variant="8k_x0.8", reps=2)
    assert vsim._rng("committed:W-90:1").random() == random.Random(
        int(hashlib.sha256(b"20261008:stage_v:prereg-sim:committed:W-90:1").hexdigest(), 16)).random()
    names, real = [], vsim._rng
    monkeypatch.setattr(vsim, "_rng", lambda name: names.append(name) or real(name))
    vsim.simulate_point(ADM, "W-50", pools=pools, gaps=[0.001], variant="steady_only", reps=2)
    assert names == [f"steady_only:W-50:{r}" for r in range(1, 6)]


def test_a_point_summary_over_its_blocks():
    pools = {"U2K": {"first": [5.0, 6.0], "steady": [1.0, 1.1]}, "U8K": {"first": [7.0, 7.5], "steady": [2.0, 2.2]}}
    out = vsim.simulate_point(ADM, "W-90", pools=pools, gaps=[0.001], variant="committed", reps=3)
    x = ADM["points"]["W-90"]
    assert out["runs"] == 15 and sorted(out["blocks"]) == ["1", "2", "3", "4", "5"]
    for r, b in out["blocks"].items():
        assert b["dataset"] == x["datasets"][r] and b["runs"] == 3
        assert b["long_in_w"] == svd.window_counts(x["datasets"][r], x["C"])["long"]
        assert b["u_w"]["min"] <= b["u_w"]["mean"] <= b["u_w"]["max"]
    assert out["u_w"]["min"] == min(b["u_w"]["min"] for b in out["blocks"].values())
    assert out["u_w"]["max"] == max(b["u_w"]["max"] for b in out["blocks"].values())
    assert isinstance(out["_mean_u_w"], Fraction) and out["u_w"]["mean"] == round(float(out["_mean_u_w"]), 4)
    assert abs(float(out["_mean_u_w"]) - statistics.fmean(b["u_w"]["mean"] for b in out["blocks"].values())) < 1e-4
    assert out["n_w"]["min"] <= out["n_w"]["mean"] <= out["n_w"]["max"]
    assert out["peak_queue"] == max(b["peak_queue"] for b in out["blocks"].values())


# ------------------------------------------------------------------ the pools --

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fake_c1b(tmp_path, *, invalid=(), c_prod=(2, 1)):
    """A c1b-shaped session: U2K / U8K at C_PROD c_prod, Phase C runs r1 / r2 (plus a run at another C and a Phase S
    run at C_PROD, never read), N 20, latency 1000 x (idx + 1) + 100 x block ms."""
    d = tmp_path / "c1b"
    d.mkdir(parents=True)
    (d / "analysis.json").write_text(json.dumps({"workloads": {
        "U2K": {"c_prod": {"status": "EXACT", "value": c_prod[0]}},
        "U8K": {"c_prod": {"status": "EXACT", "value": c_prod[1]}}}}))
    rows = []
    for w, C in zip(("U2K", "U8K"), c_prod):
        for phase, c, block in (("C", C, 1), ("C", C, 2), ("C", C + 1, 1), ("S", C, None)):
            tag = f"{phase}-{w}-c{c}-r{block}"
            rows.append({"phase": phase, "workload": w, "C": c, "N": 20, "block": block, "tag": tag,
                         "valid": tag not in invalid, "reasons": [], "derived": f"derived/{tag}-requests.jsonl"})
            write_jsonl(d / "derived" / f"{tag}-requests.jsonl",
                        [{"idx": i, "cls": "short", "ttft_ms": 100.0, "latency_ms": 1000.0 * (i + 1) + 100 * (block or 9),
                          "out": 128} for i in range(20)])
    write_jsonl(d / "rows.jsonl", rows)
    return d


def test_duration_pools_are_the_valid_phase_c_runs_at_c_prod(tmp_path):
    d = fake_c1b(tmp_path, invalid=("C-U2K-c2-r2",))
    pools = vsim.duration_pools(d)
    assert pools["U2K"]["c_prod"] == 2 and pools["U2K"]["runs"] == ["derived/C-U2K-c2-r1-requests.jsonl"]
    assert pools["U2K"]["first"] == [1.1, 2.1]                                       # idx < C
    assert pools["U2K"]["steady"] == [(1000.0 * (i + 1) + 100) / 1000 for i in range(2, 18)]   # W = [C, N - C)
    assert pools["U8K"]["runs"] == ["derived/C-U8K-c1-r1-requests.jsonl", "derived/C-U8K-c1-r2-requests.jsonl"]
    assert pools["U8K"]["first"] == [1.1, 1.2]
    assert len(pools["U8K"]["steady"]) == 36


def test_duration_pools_refuse_an_inexact_c_prod_or_no_valid_run(tmp_path):
    d = fake_c1b(tmp_path, invalid=("C-U8K-c1-r1", "C-U8K-c1-r2"))
    with pytest.raises(vsim.SimError, match="U8K"):
        vsim.duration_pools(d)
    a = json.loads((d / "analysis.json").read_text())
    a["workloads"]["U2K"]["c_prod"]["status"] = "BOUND"
    (d / "analysis.json").write_text(json.dumps(a))
    with pytest.raises(vsim.SimError, match="U2K"):
        vsim.duration_pools(d)


def test_the_committed_c1b_pools_are_lambda_ref_s_requests():
    """W's pools are exactly the requests stage_d_plan.lambda_ref averages (stage_d.toml pins their mean)."""
    pools = vsim.duration_pools(vsim.C1B_DIR)
    for w in ("U2K", "U8K"):
        ref = SD_CFG["stage_d"]["lambda_ref"][w]
        assert pools[w]["c_prod"] == ref["c_prod"] and len(pools[w]["steady"]) == ref["requests"]
        assert abs(statistics.fmean(pools[w]["steady"]) - ref["mean_latency_s"]) < 1e-6
        assert len(pools[w]["first"]) == len(pools[w]["runs"]) * ref["c_prod"]
    assert {c["max_concurrency"] for c in ADM["points"]["W-90"]["admission_params"]["classes"]} == {
        pools["U2K"]["c_prod"], pools["U8K"]["c_prod"]}


def fake_d_c(tmp_path, *, starts=None, invalid=()):
    """A D-C-shaped session: U2K-gw at C 2, N 20 (r1, r2) and a U2K-var run (never read); each idx k >= 2 starts
    0.001 x block s after the (k - 1)-th earliest finish."""
    d = tmp_path / "dc"
    d.mkdir(parents=True)
    rows = []
    for w, block in (("U2K-gw", 1), ("U2K-gw", 2), ("U2K-var", 1)):
        tag = f"V-{w}-c2-r{block}"
        rows.append({"phase": "V", "workload": w, "C": 2, "N": 20, "block": block, "tag": tag,
                     "valid": tag not in invalid, "reasons": [], "derived": f"derived/{tag}-requests.jsonl"})
        lat = [1.0 + 0.1 * (i % 3) for i in range(20)]
        start = [0.0, 0.0]
        for k in range(2, 20):
            start.append(sorted(s + la for s, la in zip(start, lat))[k - 2] + 0.001 * block)
        reqs = [{"idx": i, "cls": "short", "ttft_ms": 100.0, "latency_ms": lat[i] * 1000, "out": 128,
                 "start_s": (starts or start)[i]} for i in range(20)]
        write_jsonl(d / "derived" / f"{tag}-requests.jsonl", reqs)
    write_jsonl(d / "rows.jsonl", rows)
    return d


def test_the_turnaround_is_d_c_s_closed_loop_gap(tmp_path):
    d = fake_d_c(tmp_path, invalid=("V-U2K-gw-c2-r2",))
    t = vsim.turnaround_pool(d)
    assert t["C"] == 2 and t["runs"] == ["derived/V-U2K-gw-c2-r1-requests.jsonl"]
    assert t["gaps"] == pytest.approx([0.001] * 18, abs=1e-9)
    t = vsim.turnaround_pool(fake_d_c(tmp_path / "b"))
    assert len(t["runs"]) == 2 and t["gaps"][18:] == pytest.approx([0.002] * 18, abs=1e-9)


def test_the_turnaround_refuses_a_run_that_is_no_closed_loop(tmp_path):
    with pytest.raises(vsim.SimError, match="closed loop"):
        vsim.turnaround_pool(fake_d_c(tmp_path, starts=[0.0] * 20))
    with pytest.raises(vsim.SimError, match="U2K-gw"):
        vsim.turnaround_pool(fake_d_c(tmp_path / "b", invalid=("V-U2K-gw-c2-r1", "V-U2K-gw-c2-r2")))


def test_the_turnaround_must_be_the_w_points_client_concurrency(tmp_path):
    with pytest.raises(vsim.SimError, match="C 2"):
        vsim.prereg_sim(ADM, c1b_dir=vsim.C1B_DIR, d_c_dir=fake_d_c(tmp_path), reps=1)


# ------------------------------------------------------------------ the design check --

def test_the_design_check_is_the_mean_against_0_93_of_the_budget():
    at = Fraction(93, 100) * 32
    ok = vsim.design_check(ADM, {"W-90": at, "W-50": Fraction(31)})
    assert ok["threshold"] == 29.76 and ok["points"]["W-90"]["ok"] is True and ok["decision_needed"] is False
    low = vsim.design_check(ADM, {"W-90": Fraction(31), "W-50": at - Fraction(1, 10 ** 9)})
    assert low["points"]["W-50"]["ok"] is False and low["decision_needed"] is True
    assert low["points"]["W-90"]["ok"] is True


# ------------------------------------------------------------------ the committed output --

@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prereg")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(vsim, "ROOT", tmp)
        assert vsim.main([]) == 0
    return tmp.joinpath(*vsim.OUT).read_bytes()


def test_the_committed_output_regenerates_byte_identically(regenerated):
    assert COMMITTED.read_bytes() == regenerated


def test_the_committed_output_is_descriptive_only_with_its_design_check(regenerated):
    out = json.loads(regenerated)
    assert out["descriptive_only"] is True and out["decision"] == "044"
    assert list(out["points"]) == list(vsim.POINTS) and list(out["model"]["variants"]) == list(vsim.VARIANTS)
    check = out["design_check"]
    assert check["variant"] == vsim.DESIGN_VARIANT == "committed" and check["threshold"] == 29.76
    for point in vsim.POINTS:
        x = ADM["points"][point]
        p = out["points"][point]
        assert p["dataset"] == x["dataset"] and p["mix"] == svd.MIXES[x["dataset"]]
        assert p["trivial_threshold"] == 28.8 and p["budget"] == 32 and p["C"] == 32 and p["N"] == 320
        for v in p["variants"].values():
            assert v["runs"] == 5 * vsim.REPS and v["peak_queue"] < x["admission_params"]["max_queue"]
        mean = p["variants"]["committed"]["u_w"]["mean"]
        assert check["points"][point] == {"mean_u_w": mean, "ok": mean >= 29.76}
    assert check["decision_needed"] is not all(c["ok"] for c in check["points"].values())


def test_the_committed_output_names_its_inputs_by_sha256(regenerated):
    out = json.loads(regenerated)
    for key, d in (("c1b", vsim.C1B_DIR), ("d_c", vsim.D_C_DIR)):
        assert out["inputs"][key]["session"] == d.name
        files = out["inputs"][key]["files"]
        assert "rows.jsonl" in files
        for name, sha in files.items():
            assert hashlib.sha256((d / name).read_bytes()).hexdigest() == sha
    assert "analysis.json" in out["inputs"]["c1b"]["files"]
    assert sorted(out["inputs"]["c1b"]["files"]) == sorted(
        ["analysis.json", "rows.jsonl"] + out["pools"]["U2K"]["runs"] + out["pools"]["U8K"]["runs"])
    assert sorted(out["inputs"]["d_c"]["files"]) == sorted(["rows.jsonl"] + out["pools"]["turnaround"]["runs"])
    assert out["pools"]["turnaround"]["C"] == 32 and len(out["pools"]["turnaround"]["runs"]) == 5


def test_the_script_imports_only_research_modules():
    tree = ast.parse((ROOT / "benchmarks" / "stage_v_prereg_sim.py").read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert names <= {"__future__", "argparse", "collections", "fractions", "hashlib", "heapq", "itertools", "json",
                     "random", "statistics", "sys", "pathlib", "typing", "tomllib", "capacity_analysis",
                     "capacity_plan", "stage_d_plan", "stage_v_analysis", "stage_v_datasets", "stage_v_plan"}
    assert sd.C1B_SESSION == vsim.C1B_DIR.name and vsim.D_C_DIR.name == "formal-20261023-d-c"
