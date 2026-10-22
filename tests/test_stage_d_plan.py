"""Tests for benchmarks/stage_d_plan.py (decision 039): the stage_d.toml rules, the workloads' C_KV / C_cap / tau1, the
D-B lambda_ref from the committed c1b files, the server and client argv, run tags and the per-session plans."""

import copy
import json
import tomllib
from fractions import Fraction
from pathlib import Path

import pytest

import capacity_plan as cp
import core_b_plan as bp
import stage_d_plan as sd

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_d.toml").read_text())
C1B = ROOT / "results" / "capacity" / "formal-20261017-c1b"
SB = PINS["stage_b"]
UNIFORM_PLAN = {"num_gpu_blocks": SB["num_gpu_blocks"], "block_size": SB["block_size"]}
HEX = "a" * 64


def cfg(**stage_d):
    c = copy.deepcopy(CFG)
    c["stage_d"].update(stage_d)
    return c


def plan(session, c=None, formal=False):
    return sd.build_plan(c or CFG, PINS, session_name=session, uniform_plan=UNIFORM_PLAN, pins_sha256="p" * 64,
                         stage_d_sha256="s" * 64, formal=formal)


# ------------------------------------------------------------------ stage_d.toml --

def test_the_repo_config_is_the_039_draft():
    assert sd.stage_d_problems(CFG, PINS) == []
    assert all(len(v) == 64 for v in CFG["stage_d"]["var_dataset_sha256"].values())   # generated on WSL, pinned
    assert CFG["freeze"]["stage_d"] is False and CFG["slo"] == cp.SLO_S2
    assert CFG["stage_d"]["sessions"] == sd.SESSIONS and CFG["stage_d"]["sentinel"] == sd.SENTINEL


@pytest.mark.parametrize("key, value", [
    ("guard_band", 0.1), ("n_near", 3), ("n_low", 2), ("seed", 1), ("c1b_session", "formal-20261017-c1"),
    ("tau1_s", {"2048": 1.0, "8192": 2.95, "32768": 7.92}), ("t_tau_factor", 1),
    ("sessions", {**sd.SESSIONS, "D-B": ["U2K-P"]}),
    ("open_loop", {**sd.OPEN_LOOP, "grid": [0.5, 1.0]}), ("open_loop", {**sd.OPEN_LOOP, "trim": 0.05}),
    ("verify", {**sd.VERIFY, "points": {**sd.VERIFY["points"], "U2K-gw": 34}}),
    ("sentinel", {"workload": "U2K", "C": 32, "tolerance": 0.03}),
    ("mix_dataset_sha256", {"8192": HEX}), ("var_dataset_sha256", {"2048": "x"}),
])
def test_every_039_value_is_checked(key, value):
    assert any(key in p for p in sd.stage_d_problems(cfg(**{key: value}), PINS))


def test_the_slo_is_s2_and_unknown_keys_are_refused():
    c = copy.deepcopy(CFG)
    c["slo"]["tpot_p95_ms"] = 200
    assert any("[slo]" in p for p in sd.stage_d_problems(c, PINS))
    c = copy.deepcopy(CFG)
    c["stage_d"]["extra"] = 1
    assert any("[stage_d] keys" in p for p in sd.stage_d_problems(c, PINS))


def test_the_freeze_needs_three_mini_smokes_and_the_var_datasets(tmp_path):
    c = copy.deepcopy(CFG)
    c["freeze"]["stage_d"] = True
    c["stage_d"]["var_dataset_sha256"] = {"2048": "", "8192": ""}
    problems = sd.stage_d_problems(c, PINS)
    assert sum("freeze.mini_" in p for p in problems) == 3 and any("var_dataset_sha256 pinned" in p for p in problems)
    c["freeze"].update(mini_closed="smoke-d-a", mini_open="smoke-d-b", mini_verify="smoke-d-c")
    c["stage_d"]["var_dataset_sha256"] = {"2048": HEX, "8192": HEX}
    assert sd.stage_d_problems(c, PINS) == []
    assert len(sd.freeze_evidence_problems(c, tmp_path)) == 3            # no session.json yet
    for key, kind in sd.FREEZE_EVIDENCE.items():
        d = tmp_path.joinpath(*sd.RESULTS, c["freeze"][key])
        d.mkdir(parents=True)
        (d / "session.json").write_text(json.dumps({"mode": "mini", "kind": kind, "formal": False,
                                                    "smoke": {"ok": True}, "harness_sha256": sd.harness_sha256()}))
    assert sd.freeze_evidence_problems(c, tmp_path) == []
    (tmp_path.joinpath(*sd.RESULTS, "smoke-d-b", "session.json")).write_text(json.dumps(
        {"mode": "mini", "kind": "open", "formal": False, "smoke": {"ok": False}, "harness_sha256": sd.harness_sha256()}))
    assert len(sd.freeze_evidence_problems(c, tmp_path)) == 1
    (tmp_path.joinpath(*sd.RESULTS, "smoke-d-b", "session.json")).write_text(json.dumps(
        {"mode": "mini", "kind": "open", "formal": False, "smoke": {"ok": True}, "harness_sha256": "0" * 64}))
    assert any("another harness" in p for p in sd.freeze_evidence_problems(c, tmp_path))
    assert sd.freeze_evidence_problems(CFG, tmp_path) == []             # before the freeze


# ------------------------------------------------------------------ C_KV, cap, tau1 --

@pytest.mark.parametrize("w, kv_len, c_kv, c_cap", [
    ("U32K", 32768, 3, 6), ("U8K-T", 8192, 13, 26), ("M8K-T", 8192, 13, 26), ("U2K-O512", 2432, 30, 60),
    ("U8K-O512", 8576, 12, 24), ("U2K-P", 2048, 35, 70), ("U2K-var", 2048, 35, 70), ("U8K-gw", 8192, 13, 26),
])
def test_c_kv_and_c_cap_follow_030s_block_arithmetic(w, kv_len, c_kv, c_cap):
    """039 D-A's table: C_KV = core_b_plan.c_kv(248, 528, prompt + output), C_cap = min(128, 2 C_KV)."""
    assert (sd.kv_len(w), sd.c_kv_for(PINS, w), cp.c_cap_for(sd.c_kv_for(PINS, w))) == (kv_len, c_kv, c_cap)


@pytest.mark.parametrize("w, t1", [
    ("U2K", Fraction("1.845")), ("U32K", Fraction("7.92")), ("U8K-T", 2 * Fraction("2.95")),
    ("M8K-T", 2 * (Fraction(9, 10) * Fraction("2.95") + Fraction(1, 10) * Fraction("7.92"))),
    ("U2K-O512", 4 * Fraction("1.845")), ("U8K-O512", 4 * Fraction("2.95")), ("U2K-gw", Fraction("1.845")),
])
def test_tau1_is_030s_with_the_039_factors(w, t1):
    assert sd.tau1(CFG, w) == t1


def test_timeouts_are_030s_serial_bound_exactly():
    assert sd.timeout_s(Fraction("1.845"), 250) == cp.bench_timeout_s(2048, "uniform", 25, CFG["stage_d"]["tau1_s"])
    assert sd.timeout_s(Fraction(1), 30) == 120                         # 2 x 30 + 60 = 120: not rounded up again


# ------------------------------------------------------------------ D-B lambda_ref --

def test_lambda_ref_is_recomputed_from_the_committed_c1b_files():
    u2k, u8k = sd.lambda_ref(C1B, "U2K"), sd.lambda_ref(C1B, "U8K")
    assert (u2k["c_prod"], len(u2k["runs"]), u2k["requests"]) == (32, 3, 3 * (320 - 64))
    assert (u8k["c_prod"], len(u8k["runs"]), u8k["requests"]) == (8, 5, 5 * (200 - 16))
    assert u2k["lambda_ref"] == pytest.approx(2.5459, abs=1e-4) and u8k["lambda_ref"] == pytest.approx(0.6297, abs=1e-4)
    assert u2k["lambda_ref"] == round(32 / u2k["mean_latency_s"], 6)


def test_lambda_ref_needs_an_exact_c_prod(tmp_path):
    (tmp_path / "analysis.json").write_text(json.dumps({"workloads": {"U2K": {"c_prod": {"value": 32,
                                                                                          "status": "LOWER"}}}}))
    with pytest.raises(sd.PlanError, match="EXACT"):
        sd.lambda_ref(tmp_path, "U2K")


def test_open_points_follow_the_grid():
    ref = {"lambda_ref": 2.5}
    pts = sd.open_points(CFG, "U2K-P", ref)
    assert list(pts) == [str(i) for i in range(1, 10)]
    assert [pts[str(i)]["rate"] for i in (1, 6, 9)] == [1.0, 2.5, 3.75]
    assert [pts[str(i)]["N"] for i in (1, 6, 9)] == [300, 750, 1125]          # max(300, ceil(300 s x rate))
    assert pts["9"]["timeout_s"] == sd.timeout_s(Fraction("1.845"), 1125)
    assert sd.open_window(750, 0.1) == (75, 675) and sd.open_window(301, 0.1) == (31, 270)


# ------------------------------------------------------------------ argv --

def test_servers_are_030s_core_b_or_its_t_variant():
    b0 = sd.server_argv_for(PINS, "U32K", base_env={"PATH": "/usr/bin"})
    assert b0 == cp.server_argv_for(PINS, base_env={"PATH": "/usr/bin"})
    argv, env = sd.server_argv_for(PINS, "M8K-T", base_env={"PATH": "/usr/bin"})
    assert argv == b0[0] + ["--long-prefill-token-threshold", "512"] and env == b0[1]
    assert PINS["advanced"]["long_prefill_token_threshold"] == 512


def test_clients_are_cores_where_it_applies():
    kw = {"result_dir": "/r", "result_filename": "x.json"}
    assert sd.bench_argv_for(PINS, "U32K", C=2, **kw) == bp.bench_argv(PINS, L=32768, C=2, mix="uniform", port=8000,
                                                                       **kw)
    assert sd.bench_argv_for(PINS, "M8K-T", C=2, dataset="/m.jsonl", **kw) == bp.bench_argv(
        PINS, L=8192, C=2, mix="mixed", port=8000, mixed_dataset="/m.jsonl", **kw)
    assert sd.bench_argv_for(PINS, "U2K-gw", C=32, **kw) == bp.bench_argv(PINS, L=2048, C=32, mix="uniform",
                                                                          port=8080, **kw)


def test_a_512_token_client_differs_from_cores_only_in_its_output_length():
    kw = {"result_dir": "/r", "result_filename": "x.json"}
    core = bp.bench_argv(PINS, L=2048, C=4, mix="uniform", port=8000, **kw)
    i = core.index("--random-output-len")
    assert sd.bench_argv_for(PINS, "U2K-O512", C=4, **kw) == core[:i + 1] + ["512"] + core[i + 2:]


def test_a_var_client_uses_its_dataset_like_the_mixed_one():
    kw = {"result_dir": "/r", "result_filename": "x.json"}
    mixed = bp.bench_argv(PINS, L=2048, C=32, mix="mixed", port=8000, mixed_dataset="/v.jsonl", **kw)
    assert sd.bench_argv_for(PINS, "U2K-var", C=32, dataset="/v.jsonl", **kw) == mixed
    with pytest.raises(sd.PlanError, match="var dataset"):
        sd.bench_argv_for(PINS, "U2K-var", C=32, **kw)


def test_an_open_loop_client_is_poisson_without_a_concurrency_cap():
    kw = {"result_dir": "/r", "result_filename": "x.json"}
    argv = sd.bench_argv_for(PINS, "U8K-P", rate=0.6297, N=300, warmups=10, **kw)
    core = bp.bench_argv(PINS, L=8192, C=1, mix="uniform", port=8000, **kw)
    assert "--max-concurrency" not in argv
    assert argv[argv.index("--request-rate") + 1] == "0.6297" and argv[argv.index("--burstiness") + 1] == "1.0"
    assert argv[argv.index("--num-prompts") + 1] == "300" and argv[argv.index("--num-warmups") + 1] == "10"
    strip = {"--num-prompts", "--max-concurrency", "--num-warmups", "--request-rate", "--burstiness"}

    def rest(a):
        out, skip = [], False
        for x in a:
            if skip:
                skip = False
            elif x in strip:
                skip = True
            else:
                out.append(x)
        return out

    assert rest(argv) == rest(core)
    with pytest.raises(sd.PlanError):
        sd.bench_argv_for(PINS, "U8K-P", C=1, rate=1.0, **kw)


def test_run_tags():
    assert sd.run_tag("S", "U32K", 2) == "S-U32K-c2" and sd.run_tag("C", "M8K-T", 3, block=4) == "C-M8K-T-c3-r4"
    assert sd.run_tag("S", "U2K-P", 5, kind="open") == "S-U2K-P-i5"
    assert sd.run_tag("C", "U2K-P", 5, block=1, kind="open") == "C-U2K-P-i5-r1"
    assert sd.run_tag("V", "U2K-gw", 32, block=2, kind="verify") == "V-U2K-gw-c32-r2"
    assert sd.run_tag("sentinel", "U2K", 25, position="end", kind="open") == "sentinel-end-U2K-c25"
    with pytest.raises(sd.PlanError):
        sd.run_tag("C", "U32K", 2)
    with pytest.raises(sd.PlanError):
        sd.run_tag("S", "U9K", 2)


# ------------------------------------------------------------------ plans --

def test_closed_plans_carry_030s_search_inputs():
    p = plan("D-A1")
    assert (p["kind"], list(p["workloads"])) == ("closed", ["U32K", "U8K-T", "M8K-T"])
    u32 = p["workloads"]["U32K"]
    assert (u32["c_kv"], u32["c_cap"], u32["n_near"], u32["bracket"]) == (3, 6, 5, [1, 2, 4, 6])
    assert u32["timeouts_s"]["3"] == sd.timeout_s(Fraction("7.92"), 200)
    assert p["sentinel"] == {"workload": "U2K", "L": 2048, "mix": "uniform", "C": 25, "N": 250, "tolerance": 0.03,
                             "timeout_s": cp.bench_timeout_s(2048, "uniform", 25, CFG["stage_d"]["tau1_s"])}
    assert (p["n_low"], p["guard_band"], p["seed"], p["slo"]) == (3, 0.05, 20261008, cp.SLO_S2)
    assert list(plan("D-A2")["workloads"]) == ["U2K-O512", "U8K-O512"]


def test_capacity_plans_confirm_plan_builder_takes_a_stage_d_plan():
    """D-A / D-B run 030's own confirm set and Phase C order (capacity_plan.build_confirm_plan / phase_c_boots)."""
    p = plan("D-A1")
    searches = {"U32K": [{"C": 1, "pass": True, "within_band": True}, {"C": 2, "pass": True, "within_band": True},
                         {"C": 4, "pass": False, "within_band": False}, {"C": 3, "pass": False,
                                                                         "within_band": False}],
                "U8K-T": None, "M8K-T": [{"C": 1, "pass": False, "within_band": False}]}
    confirm = cp.build_confirm_plan(p, searches)
    assert [x["C"] for x in confirm["workloads"]["U32K"]["points"]] == [1, 2, 3]
    assert confirm["workloads"]["U8K-T"]["points"] == [] and confirm["workloads"]["M8K-T"]["points"][0]["C"] == 1
    assert cp.phase_c_boots(confirm, 1, {})


def test_the_open_plan_maps_the_grid_onto_030s_search():
    p = plan("D-B")
    x = p["workloads"]["U2K-P"]
    assert (x["c_cap"], x["scan"], x["lambda_ref"]["c_prod"]) == (9, list(range(1, 10)), 32) and "bracket" not in x
    assert x["points"]["6"]["rate"] == round(x["lambda_ref"]["lambda_ref"], 4)
    assert p["open_loop"] == sd.OPEN_LOOP


def test_the_verify_plan_runs_n_times_at_c_prod():
    p = plan("D-C")
    assert {w: (x["C"], x["n"]) for w, x in p["workloads"].items()} == {
        "U2K-var": (32, 5), "U8K-var": (8, 5), "U2K-gw": (32, 5), "U8K-gw": (8, 5)}
    assert p["verify"] == {"gateway_port": 8080, "gateway_profile": "service/profiles/verified-formal-20261017-c1b.json"}
    assert p["workloads"]["U8K-gw"]["gateway_class"] == "uniform-8k"


def test_plans_refuse_a_bad_config_an_unknown_session_and_an_unfrozen_formal():
    with pytest.raises(sd.PlanError, match="stage_d.toml"):
        plan("D-A1", c=cfg(n_near=3))
    with pytest.raises(sd.PlanError, match="unknown session"):
        plan("D-Z")
    with pytest.raises(sd.PlanError, match="freeze.stage_d"):
        plan("D-A1", formal=True)
    with pytest.raises(sd.PlanError, match="num_gpu_blocks"):
        sd.build_plan(CFG, PINS, session_name="D-A1", uniform_plan={"num_gpu_blocks": 186, "block_size": 528},
                      pins_sha256="p", stage_d_sha256="s")


def test_the_open_scan_runs_up_the_grid_to_the_first_fail():
    nxt = sd.open_search_next
    assert nxt(9, []) == 1 and nxt(9, [(1, True), (2, True)]) == 3
    assert nxt(9, [(1, True), (2, False)]) is None and nxt(9, [(i, True) for i in range(1, 10)]) is None
    for bad in ([(2, True)], [(1, False), (2, True)], [(1, True), (3, True)]):
        with pytest.raises(sd.PlanError):
            nxt(9, bad)
    assert sd.open_search_summary(9, [{"C": 1, "pass": True}, {"C": 2, "pass": False}])["c_hat"] == 1
    assert sd.open_search_summary(9, [{"C": 1, "pass": False}])["c_hat"] == 0


def test_the_open_confirm_set_is_the_neighbourhood_without_c_low():
    p = plan("D-B")
    hist = lambda k, cap=9: [{"C": i, "pass": i <= k, "within_band": i <= k} for i in range(1, min(k + 1, cap) + 1)]
    c = sd.build_open_confirm_plan(p, {"U2K-P": hist(5), "U8K-P": hist(0)})
    assert [(x["C"], x["role"], x["n"]) for x in c["workloads"]["U2K-P"]["points"]] == [
        (4, "near", 5), (5, "near", 5), (6, "near", 5)]
    assert [(x["C"], x["role"]) for x in c["workloads"]["U8K-P"]["points"]] == [(1, "one")]
    c9 = sd.build_open_confirm_plan(p, {"U2K-P": hist(9), "U8K-P": None})
    assert [x["C"] for x in c9["workloads"]["U2K-P"]["points"]] == [8, 9] and c9["workloads"]["U8K-P"]["points"] == []
    assert c["blocks"] == 5 and cp.phase_c_boots(c, 1, {})


def test_every_open_loop_run_has_its_own_seed():
    seeds = {sd.open_seed(w, i, phase=ph, block=b) for w in ("U2K-P", "U8K-P") for i in range(1, 10)
             for ph, b in (("S", None), *(("C", r) for r in range(1, 6)))}
    assert len(seeds) == 2 * 9 * 6 and all(0 <= x < 2 ** 31 for x in seeds)
    kw = {"result_dir": "/r", "result_filename": "x.json"}
    argv = sd.bench_argv_for(PINS, "U2K-P", rate=1.0, N=300, warmups=10, seed=123, **kw)
    assert argv[argv.index("--seed") + 1] == "123"
    with pytest.raises(sd.PlanError, match="Core B seed"):
        sd.bench_argv_for(PINS, "U2K", C=2, seed=123, **kw)


def test_lambda_ref_is_pinned_and_equals_the_committed_c1b_recomputation():
    assert sd.lambda_ref_problems(CFG, ROOT) == []
    assert CFG["stage_d"]["lambda_ref"]["U2K"] == {"c_prod": 32, "requests": 768, "mean_latency_s": 12.56945,
                                                   "lambda_ref": 2.545855}
    bad = cfg(lambda_ref={**CFG["stage_d"]["lambda_ref"], "U8K": {**CFG["stage_d"]["lambda_ref"]["U8K"],
                                                                  "lambda_ref": 0.7}})
    assert any("lambda_ref.U8K" in p for p in sd.stage_d_problems(bad, PINS))
    drift = cfg(lambda_ref={**CFG["stage_d"]["lambda_ref"], "U2K": {"c_prod": 32, "requests": 768,
                                                                    "mean_latency_s": 12.0, "lambda_ref": 2.666667}})
    assert sd.stage_d_problems(drift, PINS) == [] and any("U2K" in p for p in sd.lambda_ref_problems(drift, ROOT))


def test_the_harness_fingerprint_covers_every_stage_d_module():
    assert len(sd.harness_sha256()) == 64 and all((ROOT / f).is_file() for f in sd.HARNESS_FILES)
    assert {Path(f).name for f in sd.HARNESS_FILES} == {p.name for p in (ROOT / "benchmarks").glob("*stage_d*.py")}
