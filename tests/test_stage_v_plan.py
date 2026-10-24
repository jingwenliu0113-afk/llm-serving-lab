"""Tests for benchmarks/stage_v_plan.py (decision 044): configs/stage_v.toml against 044's tables, the points' server /
client / tau1 bases, sessions and units, timeouts, hour caps and run overheads, the client argv as the frozen
functions' output with only 044's three differences (asserted by diff), the gateway settings and the nested TOML
writer (tomllib round trip), the pacing parameters recomputed from the committed D5 file, the request path's AST
closure and the plans."""

import ast
import copy
import difflib
import hashlib
import json
import math
import sys
import tomllib
from fractions import Fraction
from pathlib import Path

import pytest

import capacity_plan as cp
import core_b_plan as bp
import stage_d_plan as sd
import stage_v_plan as svp

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG_RAW = (ROOT / "configs" / "stage_v.toml").read_bytes()
CFG = tomllib.loads(CFG_RAW.decode())
SD_RAW = (ROOT / "configs" / "stage_d.toml").read_bytes()
SD_CFG = tomllib.loads(SD_RAW.decode())
HEX = "a" * 64
KW = {"result_dir": "/r", "result_filename": "x.json"}

B0 = "service/profiles/verified-formal-20261023-d-a1b-b0.json"
T = "service/profiles/verified-formal-20261023-d-a1b-t.json"
STUDY = "service/profiles/provisional-stage-v.json"
# 044's point tables: study, server / client / tau1 base, C, profile, class(es), queue, admission, pacing, dataset
# (per block?), judging view, product unit, bench estimate (s)
P044 = {
    "GW-U32K": ("V-GW", "U32K", "U32K", "U32K", 2, B0, ["uniform-32k"], "product", False, "", "", False, "client",
                "VG-32K", 1462),
    "GW-U2K-O512": ("V-GW", "U2K-O512", "U2K-O512", "U2K-O512", 32, STUDY, ["v-uniform-2k-o512"], "product", False, "",
                    "", False, "client", "V1-2K", 216),
    "GW-U8K-O512": ("V-GW", "U8K-O512", "U8K-O512", "U8K-O512", 12, STUDY, ["v-uniform-8k-o512"], "product", False, "",
                    "", False, "client", "V1-8K", 426),
    "GW-M8K-T": ("V-GW", "M8K-T", "M8K-T", "M8K-T", 4, T, ["mixed-8k-t"], "product", False, "", "mixed-8192", False,
                 "client", "VG-M8K-T", 490),
    "L-U2K": ("V1", "U2K-O512", "U2K-var", "U2K-O512", 32, STUDY, ["v-uniform-2k-o512"], "product", False, "", "v1-2048",
              True, "client", "V1-2K", 156),
    "L-U8K": ("V1", "U8K-O512", "U8K-var", "U8K-O512", 8, STUDY, ["v-uniform-8k-o512-c8"], "product", False, "",
              "v1-8192", True, "client", "V1-8K", 370),
    "P-U2K": ("V2", "U2K-gw", "U2K-gw", "U2K-gw", 32, STUDY, ["v-uniform-2k"], "slow", False, "2048", "", False, "worse",
              "PACE-2K", 130),
    "P-U8K": ("V2", "U8K-gw", "U8K-gw", "U8K-gw", 8, STUDY, ["v-uniform-8k"], "slow", False, "8192", "", False, "worse",
              "PACE-8K", 325),
    "P-U2K-var": ("V2", "U2K-var", "U2K-var", "U2K-var", 32, STUDY, ["v-uniform-2k"], "slow", False, "2048", "var-2048",
                  False, "worse", "", 80),
    "F-U2K-var": ("V2", "U2K-var", "U2K-var", "U2K-var", 32, STUDY, ["v-uniform-2k"], "slow", False, "", "var-2048",
                  False, "worse", "", 78),
    "F-U2K": ("V2", "U2K-gw", "U2K-gw", "U2K-gw", 32, STUDY, ["v-uniform-2k"], "slow", False, "", "", False, "worse", "",
              125),
    "W-90": ("V4", "U2K-var", "U2K-var", "U8K-gw", 32, STUDY, ["v-uniform-2k", "v-uniform-8k"], "slow", True, "", "v4-90",
             True, "upstream", "W1", 170),
    "W-50": ("V4", "U2K-var", "U2K-var", "U8K-gw", 32, STUDY, ["v-uniform-2k", "v-uniform-8k"], "slow", True, "", "v4-50",
             True, "upstream", "W1", 330),
    "W-mini": ("V4", "U2K-var", "U2K-var", "U8K-gw", 9, STUDY, ["v-uniform-2k", "v-uniform-8k"], "slow", True, "",
               "v4-mini", False, "upstream", "", 30),
}
FIELDS = ("study", "server_base", "client_base", "tau_base", "C", "profile", "classes", "queue", "admission", "pacing",
          "dataset", "per_block", "view", "unit", "bench_est_s")
# 044 "共通的定義" / 逾時: ceil60(2 N tau1 + 60 s), tau1 from the frozen stage_d.toml of the tau1 base
TIMEOUTS = {"GW-U32K": 3240, "GW-U2K-O512": 4800, "GW-U8K-O512": 4800, "GW-M8K-T": 2820, "L-U2K": 4800, "L-U8K": 4800,
            "P-U2K": 1260, "P-U8K": 1260, "P-U2K-var": 1260, "F-U2K-var": 1260, "F-U2K": 1260, "W-90": 1980,
            "W-50": 1980}
# 044 "Sessions 與時數": 5 runs of (bench estimate + the run overhead of the point's server), in hours
HOURS = {"P-U2K": 0.31, "P-U8K": 0.58, "P-U2K-var": 0.24, "F-U2K-var": 0.24, "F-U2K": 0.31, "W-90": 0.37, "W-50": 0.59,
         "GW-U32K": 2.16, "GW-U2K-O512": 0.43, "GW-U8K-O512": 0.72, "GW-M8K-T": 0.83, "L-U2K": 0.35, "L-U8K": 0.65}
REQUEST_PATH = ["__init__", "__main__", "admission", "app", "config", "gateway_path", "guard", "keys", "metrics",
                "openai_api", "profile", "proxy", "retry", "server", "tokens", "upstream"]
VAR_PATH = "/d/var-2048.jsonl"


def stage_v(**values):
    c = copy.deepcopy(CFG)
    c["stage_v"].update(values)
    return c


def frozen_cfg():
    c = copy.deepcopy(CFG)
    c["freeze"] = {"stage_v": True, "mini": "smoke-v-20261101", "gateway_path_sha256": HEX}
    c["stage_v"]["datasets"] = {k: (v or HEX) for k, v in c["stage_v"]["datasets"].items()}
    return c


def plan(session, c=None, formal=False, sd_cfg=None, stage_d_sha256=None):
    return svp.build_plan(c or CFG, PINS, sd_cfg or SD_CFG, session_name=session, pins_sha256="p" * 64,
                          stage_v_sha256="v" * 64, stage_d_sha256=stage_d_sha256 or hashlib.sha256(SD_RAW).hexdigest(),
                          formal=formal)


def datasets_for(p):
    return {k: f"/d/{k}.jsonl" for x in p["points"].values() for k in (x["datasets"] or {}).values()}


def diff(a, b):
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    return [(op, a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in ops if op != "equal"]


# ------------------------------------------------------------------ stage_v.toml (044) --

def test_the_repo_config_is_044s_before_the_freeze():
    assert svp.stage_v_problems(CFG, PINS, SD_CFG) == []
    assert CFG["freeze"] == {"stage_v": False, "mini": "", "gateway_path_sha256": ""}
    v = CFG["stage_v"]
    assert (v["seed"], v["n"], v["guard_band"]) == (20261008, 5, 0.05)
    assert v["stage_d_sha256"] == hashlib.sha256(SD_RAW).hexdigest()               # the frozen stage_d.toml (039)
    assert v["stage_d_sha256"].startswith("be7909b1")
    assert v["hour_caps_h"] == {"mini": 1.0, "V-ADM": 3.5, "V-CLS": 7.0}
    assert (v["total_cap_h"], v["min_gap_h"]) == (12, 2)
    assert v["run_overhead_s"] == {"B0": 95, "T": 110}
    assert v["nontrivial"] == {"u_w": 0.9, "n1": 0.9}
    assert (v["join_tolerance_ms"], v["gateway_drain_s"]) == (50, 5)
    assert v["queue"] == {"product": {"max_queue_wait_s": 2.0, "request_deadline_s": 300.0},
                          "slow": {"max_queue_wait_s": 120.0, "request_deadline_s": 600.0}}
    assert v["weighted_v1"] == {"policy": "weighted-v1", "budget": 32, "max_queue": 32,
                                "classes": [{"name": "v-uniform-2k", "units": 1, "max_concurrency": 32},
                                            {"name": "v-uniform-8k", "units": 4, "max_concurrency": 8}]}
    assert {t: (v["pacing"][t]["burst_tokens"], v["pacing"][t]["rate_tokens_per_s"]) for t in ("2048", "8192")} == {
        "2048": (9600, 5920), "8192": (32256, 5800)}
    assert v["units"] == {"VG-32K": ["GW-U32K"], "VG-M8K-T": ["GW-M8K-T"], "V1-2K": ["L-U2K", "GW-U2K-O512"],
                          "V1-8K": ["L-U8K", "GW-U8K-O512"], "PACE-2K": ["P-U2K"], "PACE-8K": ["P-U8K"],
                          "W1": ["W-90", "W-50"]}
    assert v["sentinel"] == SD_CFG["stage_d"]["sentinel"] == {"workload": "U2K", "C": 25, "tolerance": 0.03}
    assert CFG["slo"] == SD_CFG["slo"] == cp.SLO_S2 and CFG["secondary_slo"] == SD_CFG["secondary_slo"]


def test_the_sessions_are_044s_with_the_f_u2k_control_in_v_adm():
    assert CFG["stage_v"]["sessions"] == svp.SESSIONS == {
        "mini": ["GW-U8K-O512", "L-U8K", "GW-M8K-T", "P-U2K", "F-U2K-var", "W-mini"],
        "V-ADM": ["P-U2K", "P-U8K", "P-U2K-var", "F-U2K-var", "F-U2K", "W-90", "W-50"],
        "V-CLS": ["GW-U32K", "GW-U2K-O512", "GW-U8K-O512", "GW-M8K-T", "L-U2K", "L-U8K"]}


def test_every_point_is_044s_with_separate_server_client_and_tau1_bases():
    assert set(CFG["stage_v"]["points"]) == set(svp.POINTS) == set(P044)
    for p, want in P044.items():
        x = CFG["stage_v"]["points"][p]
        assert tuple(x[k] for k in FIELDS) == want, p
        assert x["provisional"] is (x["profile"] == STUDY)
        assert x["out_pinned"] is (x["study"] == "V1")                         # only L points pin their outputs
        assert svp.POINTS[p] == {k: v for k, v in x.items() if k != "profile_sha256"}


def test_the_datasets_are_pinned_where_stage_d_and_pins_pinned_them():
    d = CFG["stage_v"]["datasets"]
    tiers = [f"v1-{t}-r{r}" for t in (2048, 8192) for r in range(1, 6)] + [f"v4-{m}-r{r}" for m in (90, 50)
                                                                          for r in range(1, 6)]
    assert set(d) == set(tiers) | {"v1-mini-8192", "v4-mini", "var-2048", "mixed-8192"}
    assert len(svp.DATASETS) == 22 and set(svp.DATASETS) == set(d) - {"var-2048", "mixed-8192"}
    assert all(d[k] == "" for k in svp.DATASETS)                    # generated on WSL and pinned before the mini
    assert d["var-2048"] == SD_CFG["stage_d"]["var_dataset_sha256"]["2048"]          # D-C's var-2048, the same file
    assert d["var-2048"].startswith("358ae8da")
    assert d["mixed-8192"] == PINS["core_b"]["mix_dataset_sha256"]["8192"] and d["mixed-8192"].startswith("ab548bed")


@pytest.mark.parametrize("key, value", [
    ("seed", 1), ("n", 3), ("guard_band", 0.1), ("stage_d_sha256", HEX), ("total_cap_h", 13), ("min_gap_h", 1),
    ("hour_caps_h", {"mini": 1.0, "V-ADM": 3.5, "V-CLS": 8.0}), ("run_overhead_s", {"B0": 95, "T": 139}),
    ("nontrivial", {"u_w": 0.8, "n1": 0.9}), ("join_tolerance_ms", 100), ("gateway_drain_s", 60),
    ("sessions", {**svp.SESSIONS, "V-ADM": ["P-U2K", "P-U8K", "P-U2K-var", "F-U2K-var", "W-90", "W-50"]}),
    ("queue", {"product": {"max_queue_wait_s": 2.0, "request_deadline_s": 300.0},
               "slow": {"max_queue_wait_s": 2.0, "request_deadline_s": 600.0}}),
    ("weighted_v1", {**svp.WEIGHTED_V1, "budget": 30}), ("units", {**svp.UNITS, "W1": ["W-90"]}),
    ("sentinel", {"workload": "U2K", "C": 32, "tolerance": 0.03}),
    ("pacing", {**svp.PACING, "2048": {**svp.PACING["2048"], "rate_tokens_per_s": 5922}}),
    ("gateway_path_files", ["service/capserve/app.py", "service/capserve/__init__.py"]),
])
def test_every_044_value_is_checked(key, value):
    assert any(key in p for p in svp.stage_v_problems(stage_v(**{key: value}), PINS, SD_CFG))


@pytest.mark.parametrize("p, key, value", [
    ("L-U2K", "client_base", "U2K-O512"), ("L-U2K", "server_base", "U2K-var"), ("W-90", "tau_base", "U2K-var"),
    ("GW-U8K-O512", "C", 8), ("L-U8K", "classes", ["v-uniform-8k-o512"]), ("P-U8K", "queue", "product"),
    ("P-U2K", "pacing", ""), ("F-U2K-var", "pacing", "2048"), ("W-50", "admission", False), ("W-90", "view", "client"),
    ("P-U2K-var", "unit", "PACE-2K"), ("GW-U32K", "provisional", True), ("L-U8K", "out_pinned", False),
    ("P-U8K", "limits", {"max_concurrency": 8, "max_queue": 32, "max_prompt_tokens": 8064, "max_output_tokens": 128}),
    ("GW-M8K-T", "profile", B0), ("GW-U32K", "profile_sha256", "x"), ("W-90", "dataset", "v4-50"),
])
def test_every_point_value_is_checked(p, key, value):
    c = copy.deepcopy(CFG)
    c["stage_v"]["points"][p][key] = value
    assert any(f"{p}" in x and key in x for x in svp.stage_v_problems(c, PINS, SD_CFG))


def test_the_slo_sentinel_and_var_dataset_must_be_stage_ds():
    s = copy.deepcopy(SD_CFG)
    s["slo"]["tpot_p95_ms"] = 150
    assert any("[slo]" in p for p in svp.stage_v_problems(CFG, PINS, s))
    s = copy.deepcopy(SD_CFG)
    s["stage_d"]["sentinel"]["C"] = 32
    assert any("sentinel" in p for p in svp.stage_v_problems(CFG, PINS, s))
    s = copy.deepcopy(SD_CFG)
    s["stage_d"]["var_dataset_sha256"]["2048"] = HEX
    assert any("var-2048" in p for p in svp.stage_v_problems(CFG, PINS, s))
    c = copy.deepcopy(CFG)
    c["stage_v"]["datasets"]["mixed-8192"] = HEX
    assert any("mixed-8192" in p for p in svp.stage_v_problems(c, PINS, SD_CFG))
    c = copy.deepcopy(CFG)
    c["secondary_slo"]["S1"]["tpot_p95_ms"] = 40
    assert any("[secondary_slo]" in p for p in svp.stage_v_problems(c, PINS, SD_CFG))
    s = copy.deepcopy(SD_CFG)
    s["secondary_slo"]["S3"]["tpot_p95_ms"] = 300
    assert any("[secondary_slo]" in p and "stage_d.toml" in p for p in svp.stage_v_problems(CFG, PINS, s))
    c, s = copy.deepcopy(CFG), copy.deepcopy(SD_CFG)        # both S3: still not the user's S2 (037 §3)
    c["slo"], s["slo"] = copy.deepcopy(cp.SLO_S3), copy.deepcopy(cp.SLO_S3)
    assert any("[slo]" in p and "S2" in p for p in svp.stage_v_problems(c, PINS, s))


def test_unknown_keys_and_missing_tables_are_refused():
    for edit, needle in ((lambda c: c["stage_v"].update(extra=1), "[stage_v] keys"),
                         (lambda c: c["stage_v"]["points"]["W-90"].update(extra=1), "[stage_v.points.W-90] keys"),
                         (lambda c: c["stage_v"]["points"].pop("F-U2K"), "[stage_v.points] keys"),
                         (lambda c: c["stage_v"]["datasets"].pop("v4-mini"), "[stage_v.datasets] keys"),
                         (lambda c: c["freeze"].pop("mini"), "[freeze] keys"),
                         (lambda c: c.update(extra=1), "stage_v.toml keys")):
        c = copy.deepcopy(CFG)
        edit(c)
        assert any(needle in p for p in svp.stage_v_problems(c, PINS, SD_CFG)), needle
    c = copy.deepcopy(CFG)
    c["freeze"]["stage_v"] = "false"
    assert any("freeze.stage_v" in p for p in svp.stage_v_problems(c, PINS, SD_CFG))
    for key, value in (("gateway_path_sha256", "abc"), ("gateway_path_sha256", None), ("mini", 1)):
        c = copy.deepcopy(CFG)                                  # before the freeze too
        c["freeze"][key] = value
        assert any(f"freeze.{key}" in p for p in svp.stage_v_problems(c, PINS, SD_CFG)), (key, value)


def test_the_freeze_needs_the_mini_the_request_path_sha_and_every_dataset():
    c = copy.deepcopy(CFG)
    c["freeze"]["stage_v"] = True
    problems = svp.stage_v_problems(c, PINS, SD_CFG)
    assert any("freeze.mini" in p for p in problems) and any("freeze.gateway_path_sha256" in p for p in problems)
    assert any("dataset" in p and "pinned" in p for p in problems)
    assert svp.stage_v_problems(frozen_cfg(), PINS, SD_CFG) == []
    for key, value in (("mini", "smoke-d-x"), ("gateway_path_sha256", "abc")):
        c = frozen_cfg()
        c["freeze"][key] = value
        assert any(f"freeze.{key}" in p for p in svp.stage_v_problems(c, PINS, SD_CFG))
    c = frozen_cfg()
    c["stage_v"]["datasets"]["v4-mini"] = ""
    assert any("v4-mini" in p for p in svp.stage_v_problems(c, PINS, SD_CFG))
    c = copy.deepcopy(CFG)
    c["stage_v"]["datasets"]["v1-2048-r1"] = "xyz"
    assert any("v1-2048-r1" in p for p in svp.stage_v_problems(c, PINS, SD_CFG))


# ------------------------------------------------------------------ profiles, limits --

def test_each_points_profile_is_pinned_by_its_sha256_and_its_limits_are_the_profiles():
    assert svp.profile_problems(CFG, ROOT) == []
    for p, x in CFG["stage_v"]["points"].items():
        doc = json.loads((ROOT / x["profile"]).read_text())
        assert x["profile_sha256"] == hashlib.sha256((ROOT / x["profile"]).read_bytes()).hexdigest()
        assert doc["status"] == ("provisional" if x["provisional"] else "verified")
        classes = {c["name"]: c for c in doc["classes"]}
        served = [classes[n] for n in x["classes"]]
        if x["admission"]:                       # weighted-v1's envelope (044 product note 6)
            adm = CFG["stage_v"]["weighted_v1"]
            want = {"max_concurrency": max(a["max_concurrency"] for a in adm["classes"]),
                    "max_queue": adm["max_queue"], "max_prompt_tokens": max(c["max_prompt_tokens"] for c in served),
                    "max_output_tokens": served[0]["max_output_tokens"]}
            assert [(a["name"], a["max_concurrency"]) for a in adm["classes"]] == [(c["name"], c["max_concurrency"])
                                                                                   for c in served]
        else:
            want = {k: served[0][k] for k in ("max_concurrency", "max_queue", "max_prompt_tokens",
                                              "max_output_tokens")}
        assert x["limits"] == want, p
    # 044 product note 10: P-U8K's queue limit is v-uniform-8k's 8; L-U8K serves the 8 V1-8K would unlock
    assert CFG["stage_v"]["points"]["P-U8K"]["limits"]["max_queue"] == 8
    assert CFG["stage_v"]["points"]["L-U8K"]["limits"]["max_concurrency"] == 8
    # weighted-v1's classes are 2K (1,920 tokens) and 8K (8,064 tokens), 128 output tokens (044 V4)
    study = {c["name"]: c for c in json.loads((ROOT / STUDY).read_text())["classes"]}
    assert [(study[n]["max_prompt_tokens"], study[n]["max_output_tokens"]) for n in ("v-uniform-2k", "v-uniform-8k")] \
        == [(1920, 128), (8064, 128)]


def test_a_changed_profile_is_a_problem(tmp_path):
    for x in CFG["stage_v"]["points"].values():
        (tmp_path / x["profile"]).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / x["profile"]).write_bytes((ROOT / x["profile"]).read_bytes())
    assert svp.profile_problems(CFG, tmp_path) == []
    (tmp_path / T).write_bytes((ROOT / T).read_bytes() + b"\n")
    assert [p for p in svp.profile_problems(CFG, tmp_path) if "GW-M8K-T" in p] != []
    assert all("GW-M8K-T" in p for p in svp.profile_problems(CFG, tmp_path))
    (tmp_path / STUDY).unlink()
    assert any("L-U2K" in p for p in svp.profile_problems(CFG, tmp_path))


# ------------------------------------------------------------------ units, timeouts, hours --

def test_units_live_in_one_formal_session_and_controls_skip_alone():
    for unit, points in svp.UNITS.items():
        assert any(set(points) <= set(svp.SESSIONS[s]) for s in ("V-ADM", "V-CLS")), unit
        for p in points:
            assert svp.POINTS[p]["unit"] == unit and svp.skip_group(p) == points
    assert {p for p, x in svp.POINTS.items() if not x["unit"]} == {"P-U2K-var", "F-U2K-var", "F-U2K", "W-mini"}
    for p in ("P-U2K-var", "F-U2K-var", "F-U2K", "W-mini"):
        assert svp.skip_group(p) == [p]          # the var pair and F-U2K never kill PACE-2K's point (044 V2)
    assert svp.VETOES == {"PACE-2K": {"control": "F-U2K-var", "paced": "P-U2K-var"}}


def test_the_timeouts_are_044s_table():
    for session in ("V-ADM", "V-CLS"):
        p = plan(session)
        for name, x in p["points"].items():
            assert x["timeout_s"] == TIMEOUTS[name], name
            assert x["timeout_s"] == sd.timeout_s(sd.tau1(SD_CFG, x["tau_base"]), bp.num_prompts(x["C"]))
            assert x["N"] == max(200, 10 * x["C"])
        assert p["sentinel"]["timeout_s"] == 1020 and p["sentinel"]["N"] == 250


def test_hour_caps_and_run_overheads_give_044s_estimates():
    for session, total in (("V-ADM", 2.75), ("V-CLS", 5.25)):
        p = plan(session)
        hours = {}
        for name, x in p["points"].items():
            assert x["run_overhead_s"] == (110 if x["server"] == "T" else 95)
            assert x["est_s"] == x["bench_est_s"] + x["run_overhead_s"]
            hours[name] = round(x["n"] * x["est_s"] / 3600, 2)
            assert hours[name] == HOURS[name], name
        sentinels = round(2 * p["sentinel"]["est_s"] / 3600, 2)
        assert p["sentinel"]["est_s"] == 103 + 95 and sentinels == 0.11
        assert round(sum(hours.values()) + sentinels, 2) == total
        assert p["estimate_s"] == sum(x["n"] * x["est_s"] for x in p["points"].values()) + 2 * p["sentinel"]["est_s"]
        assert 1.25 <= p["hour_cap_h"] / (p["estimate_s"] / 3600) <= 1.35        # caps about 1.3 x the estimate
    assert plan("V-CLS")["points"]["GW-M8K-T"]["server"] == "T"


def test_a_boot_starts_only_inside_the_cap_less_the_end_sentinel_reserve():
    cap = 3.5
    limit = (Fraction("3.5") - Fraction("0.06")) * 3600          # 12,384 s
    assert svp.cap_allows(float(limit - 225), 225, cap)
    assert not svp.cap_allows(float(limit - 225) + 0.001, 225, cap)
    assert svp.END_RESERVE_H == 0.06
    assert svp.total_cap_allows(12 - 3.5, 3.5) and not svp.total_cap_allows(8.6, 3.5)   # 12 h over every session
    assert svp.min_gap_ok("2026-11-01T10:00:00+00:00", "2026-11-01T12:00:00+00:00")
    assert not svp.min_gap_ok("2026-11-01T10:00:00+00:00", "2026-11-01T11:59:59+00:00")


# ------------------------------------------------------------------ run tags --

def test_run_tags():
    assert svp.run_tag("V", "GW-U2K-O512", 32, block=3) == "V-GW-U2K-O512-c32-r3"
    assert svp.run_tag("sentinel", "U2K", 25, position="start") == "sentinel-start-U2K-c25"
    for bad in (lambda: svp.run_tag("V", "U2K-gw", 32, block=1), lambda: svp.run_tag("V", "W-90", 32),
                lambda: svp.run_tag("C", "W-90", 32, block=1), lambda: svp.run_tag("sentinel", "W-90", 2,
                                                                                   position="end"),
                lambda: svp.run_tag("V", "W-90", 32, block=1, position="start")):
        with pytest.raises(svp.PlanError):
            bad()


# ------------------------------------------------------------------ client argv --

@pytest.mark.parametrize("session", ["V-ADM", "V-CLS", "mini"])
def test_the_client_is_the_frozen_functions_output_with_only_044s_three_differences(session):
    """044 client argv: (1) --base-url port 8000 -> 8080 where the frozen function gives 8000, (2) L points only:
    --custom-output-len 128 -> -1, (3) every gateway run adds --request-id-prefix <tag>-."""
    p = plan(session)
    datasets = datasets_for(p)
    for name, x in p["points"].items():
        base, tier_set = P044[name][2], False
        for r in range(1, p["n"] + 1):
            key = (x["datasets"] or {}).get(str(r))
            frozen = sd.bench_argv_for(PINS, base, C=x["C"], dataset=datasets.get(key), **KW)
            got = svp.bench_argv(p, PINS, name, block=r, datasets=datasets, **KW)
            tag = svp.run_tag("V", name, x["C"], block=r)
            want = []
            if frozen[frozen.index("--base-url") + 1] == "http://127.0.0.1:8000":
                want.append(("replace", ["http://127.0.0.1:8000"], ["http://127.0.0.1:8080"]))
            if name.startswith("L-"):
                want.append(("replace", ["128"], ["-1"]))
                assert got[got.index("--custom-output-len") + 1] == "-1"
                tier_set = True
            want.append(("insert", [], ["--request-id-prefix", f"{tag}-"]))
            assert diff(frozen, got) == want, (name, r)
            assert got[got.index("--base-url") + 1] == "http://127.0.0.1:8080"
        assert tier_set is name.startswith("L-")


def test_the_dataset_of_each_block():
    p = plan("V-CLS")
    assert p["points"]["L-U2K"]["datasets"] == {str(r): f"v1-2048-r{r}" for r in range(1, 6)}
    assert p["points"]["GW-M8K-T"]["datasets"] == {str(r): "mixed-8192" for r in range(1, 6)}
    assert p["points"]["GW-U32K"]["datasets"] is None
    assert plan("V-ADM")["points"]["P-U2K-var"]["datasets"] == {str(r): "var-2048" for r in range(1, 6)}
    assert plan("V-ADM")["points"]["W-50"]["datasets"]["4"] == "v4-50-r4"
    argv = svp.bench_argv(p, PINS, "L-U8K", block=4, datasets={"v1-8192-r4": "/d/4.jsonl"}, **KW)
    assert argv[argv.index("--dataset-path") + 1] == "/d/4.jsonl"
    with pytest.raises(svp.PlanError, match="v1-8192-r5"):
        svp.bench_argv(p, PINS, "L-U8K", block=5, datasets={"v1-8192-r4": "/d/4.jsonl"}, **KW)
    for bad in ({"block": 0}, {"block": 6}):
        with pytest.raises(svp.PlanError):
            svp.bench_argv(p, PINS, "GW-U32K", datasets={}, **bad, **KW)
    with pytest.raises(svp.PlanError):
        svp.bench_argv(p, PINS, "W-90", block=1, datasets={}, **KW)      # not a V-CLS point


# ------------------------------------------------------------------ gateway settings, TOML --

def test_the_gateway_settings_of_each_kind_of_point():
    snap = bp.snapshot_dir(PINS)
    common = {"backend_url": "http://127.0.0.1:8000", "host": "127.0.0.1", "port": 8080, "drain_timeout_s": 5,
              "tokenizer": "local", "tokenizer_path": snap}
    cls = plan("V-CLS")
    assert svp.gateway_settings(cls, PINS, "GW-U32K") == {"active_class": "uniform-32k", **common,
                                                          "max_queue_wait_s": 2.0, "request_deadline_s": 300.0}
    assert svp.gateway_settings(cls, PINS, "L-U8K")["active_class"] == "v-uniform-8k-o512-c8"
    adm = plan("V-ADM")
    assert svp.gateway_settings(adm, PINS, "P-U8K") == {
        "active_class": "v-uniform-8k", **common, "max_queue_wait_s": 120.0, "request_deadline_s": 600.0,
        "prefill_pacing": {"burst_tokens": 32256, "rate_tokens_per_s": 5800}}
    assert svp.gateway_settings(adm, PINS, "P-U2K-var")["prefill_pacing"] == {"burst_tokens": 9600,
                                                                               "rate_tokens_per_s": 5920}
    assert svp.gateway_settings(adm, PINS, "F-U2K") == {"active_class": "v-uniform-2k", **common,
                                                        "max_queue_wait_s": 120.0, "request_deadline_s": 600.0}
    assert svp.gateway_settings(adm, PINS, "W-90") == {
        **common, "max_queue_wait_s": 120.0, "request_deadline_s": 600.0,
        "admission": {"policy": "weighted-v1", "budget": 32, "max_queue": 32,
                      "classes": [{"name": "v-uniform-2k", "units": 1, "max_concurrency": 32},
                                  {"name": "v-uniform-8k", "units": 4, "max_concurrency": 8}]}}


@pytest.mark.parametrize("session", ["V-ADM", "V-CLS", "mini"])
def test_every_points_settings_round_trip_through_tomllib_and_are_pinned_in_the_plan(session):
    p = plan(session)
    for name, x in p["points"].items():
        settings = svp.gateway_settings(p, PINS, name)
        text = svp.toml_dumps(settings)
        assert tomllib.loads(text) == settings == x["gateway"]
        assert x["gateway_toml_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert ("active_class" in settings) is (not x["admission"])          # weighted-v1 has no active class
        assert ("prefill_pacing" in settings) is bool(x["pacing"]) and ("admission" in settings) is x["admission"]
        assert not ("prefill_pacing" in settings and "admission" in settings)  # never together (044 item 4)


def test_the_toml_writer_nests_tables_and_round_trips():
    doc = {"a": 1, "b": -2.5, "c": True, "d": "x \"q\" \\ \n\t\x7f\x01 é 😀", "e": [1, "2", [3.0], {"k": "v", "n": {"m": 1}}],
           "empty": [], "inline": [{}], "t": {"x": 0.1, "u": {"y": [False], "deeper": {"z": "w"}}, "after": 2},
           "bare-key_1": 3, "needs quote": 4, "": 5, "2048": {"burst_tokens": 9600}, "f": 1e+16, "g": -0.0,
           "big": 10 ** 18, "z": {}}
    text = svp.toml_dumps(doc)
    assert tomllib.loads(text) == doc
    assert repr(tomllib.loads(text)["g"]) == "-0.0" and isinstance(tomllib.loads(text)["f"], float)
    assert "[t.u.deeper]" in text and '["needs quote"]' not in text and '"needs quote" = 4' in text
    assert svp.toml_dumps({}) == "\n" and tomllib.loads(svp.toml_dumps({})) == {}
    assert svp.toml_dumps({"t": {"a": 1}}) == "[t]\na = 1\n"
    for bad in ({"x": None}, {"x": (1, 2)}, {"x": float("nan")}, {"x": float("inf")}, {"x": Fraction(1, 2)},
                {"x": {"y": [None]}}, {1: 2}):
        with pytest.raises(svp.PlanError):
            svp.toml_dumps(bad)


# ------------------------------------------------------------------ pacing (044 V2) --

def test_the_pacing_parameters_are_recomputed_from_the_committed_d5_file():
    d5 = json.loads((ROOT / "results" / "stage_d" / "d5-startup-wave" / "analysis.json").read_text())
    got = svp.pacing_params(d5, PINS)
    assert {t: (got[t]["burst_tokens"], got[t]["rate_tokens_per_s"]) for t in ("2048", "8192")} == {
        "2048": (9600, 5920), "8192": (32256, 5800)}
    assert round(got["2048"]["slope_ms"], 2) == 324.23 and round(got["8192"]["slope_ms"], 2) == 1389.15
    assert (got["2048"]["c_kv"], got["8192"]["c_kv"]) == (35, 13)
    assert (got["2048"]["burst_requests"], got["8192"]["burst_requests"]) == (5, 4)
    # the file's own fit uses every point (365.4 / 1,462.5 ms) and is never quoted (044 V2)
    assert round(d5["workloads"]["U2K"]["fit"]["slope_ms_per_request"], 1) == 365.4
    assert 10 * math.floor(1920 / 0.3653742919361527 / 10) != 5920
    assert svp.pacing_problems(CFG, ROOT, PINS) == []


def test_the_pacing_recomputation_reads_only_the_points_up_to_c_kv(tmp_path):
    d5 = json.loads((ROOT / "results" / "stage_d" / "d5-startup-wave" / "analysis.json").read_text())
    above = copy.deepcopy(d5)
    above["workloads"]["U2K"]["points"]["36"]["mean_ms"] = 99999.0           # above C_KV 35: not read
    assert svp.pacing_params(above, PINS) == svp.pacing_params(d5, PINS)
    inside = copy.deepcopy(d5)
    inside["workloads"]["U8K"]["points"]["12"]["mean_ms"] = 18000.0         # inside: changes the slope
    assert svp.pacing_params(inside, PINS)["8192"]["rate_tokens_per_s"] != 5800
    path = tmp_path.joinpath(*svp.D5_ANALYSIS.split("/"))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(inside))
    assert any("8192" in p for p in svp.pacing_problems(CFG, tmp_path, PINS))
    assert any("cannot recompute" in p for p in svp.pacing_problems(CFG, tmp_path / "nowhere", PINS))


def test_burst_is_the_requests_whose_serial_prefill_stays_inside_the_guard_band():
    """042 D5: k requests with k x slope <= (1 - m) x the S2 TTFT of the tier, k x the prompt."""
    for t, prompt, slo in (("2048", 1920, 2000), ("8192", 8064, 6000)):
        p = svp.PACING[t]
        d5 = json.loads((ROOT / svp.D5_ANALYSIS).read_text())
        slope = Fraction(svp.pacing_params(d5, PINS)[t]["slope_ms"])
        k = p["burst_requests"]
        assert k * slope <= Fraction(95, 100) * slo < (k + 1) * slope
        assert p["burst_tokens"] == k * prompt and p["prompt"] == prompt


# ------------------------------------------------------------------ the request path (AST closure) --

def test_the_request_path_is_main_s_import_closure_on_this_checkout():
    files = svp.gateway_path_files(ROOT)
    assert files == [f"service/capserve/{n}.py" for n in REQUEST_PATH] == CFG["stage_v"]["gateway_path_files"]
    assert "service/capserve/__init__.py" in files and "service/capserve/deploy.py" not in files
    assert not {f"service/capserve/{n}.py" for n in ("deploy", "itest", "loadtest", "runs", "procs",
                                                     "profile_gen")} & set(files)
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode() + b"\0" + (ROOT / rel).read_bytes() + b"\0")
    assert svp.gateway_path_sha256(ROOT) == svp.gateway_path_sha256() == h.hexdigest()
    assert svp.gateway_path_problems(CFG, ROOT) == []


def package(tmp_path, modules: dict) -> Path:
    pkg = tmp_path / "service" / "capserve"
    pkg.mkdir(parents=True)
    for name, text in modules.items():
        (pkg / f"{name}.py").write_text(text)
    return tmp_path


def test_every_import_form_counts_inside_functions_too_and_text_does_not(tmp_path):
    root = package(tmp_path, {
        "__init__": '"""doc"""\n',
        "__main__": ("import json\nfrom capserve.a import x\nimport capserve.b as bee\n\n"
                     "def main():\n    from capserve import c, NAME\n    from . import d\n    from .e import f\n"),
        "a": "x = 1\n", "b": "", "c": "import capserve\n", "d": "import capserve.g\n", "e": "f = 2\n",
        "g": "# import capserve.h\ns = 'from capserve import h'\n", "h": "",
        "i": "", "j": "import capserver\nimport other.capserve\n", "k": "from .. import far\nfrom ..m import n\n"})
    (root / "service" / "capserve" / "e.py").write_text("f = 2\nfrom .. import k\n")
    assert svp.gateway_path_files(root) == [f"service/capserve/{n}.py"
                                            for n in ("__init__", "__main__", "a", "b", "c", "d", "e", "g")]


def test_the_closure_is_transitive_and_follows_init(tmp_path):
    root = package(tmp_path, {"__init__": "from capserve import k\n", "__main__": "import capserve.a\n",
                              "a": "from capserve.b import y\n", "b": "y = 1\n", "k": "", "unused": ""})
    assert svp.gateway_path_files(root) == [f"service/capserve/{n}.py"
                                            for n in ("__init__", "__main__", "a", "b", "k")]


def test_a_changed_byte_in_the_path_changes_the_sha_and_one_outside_does_not(tmp_path):
    root = package(tmp_path, {"__init__": "", "__main__": "import capserve.a\n", "a": "x = 1\n", "deploy": "y = 1\n"})
    before = svp.gateway_path_sha256(root)
    (root / "service" / "capserve" / "deploy.py").write_text("y = 2\n")
    assert svp.gateway_path_sha256(root) == before
    (root / "service" / "capserve" / "a.py").write_text("x = 2\n")
    assert svp.gateway_path_sha256(root) != before


def test_the_listed_files_and_the_frozen_sha_must_be_the_checkouts(tmp_path):
    root = package(tmp_path, {"__init__": "", "__main__": "import capserve.a\n", "a": ""})
    c = copy.deepcopy(CFG)
    c["stage_v"]["gateway_path_files"] = svp.gateway_path_files(root)
    assert svp.gateway_path_problems(c, root) == []
    (root / "service" / "capserve" / "a.py").write_text("import capserve.b\n")
    (root / "service" / "capserve" / "b.py").write_text("")
    assert any("gateway_path_files" in p for p in svp.gateway_path_problems(c, root))
    c["stage_v"]["gateway_path_files"] = svp.gateway_path_files(root)
    c["freeze"]["gateway_path_sha256"] = svp.gateway_path_sha256(root)
    assert svp.gateway_path_problems(c, root) == []
    (root / "service" / "capserve" / "b.py").write_text("x = 1\n")
    assert any("gateway_path_sha256" in p for p in svp.gateway_path_problems(c, root))


# ------------------------------------------------------------------ harness, freeze evidence --

def test_harness_sha256_hashes_the_five_modules_path_and_bytes(tmp_path):
    assert svp.HARNESS_FILES == ("benchmarks/stage_v_plan.py", "benchmarks/stage_v_datasets.py",
                                 "benchmarks/stage_v_analysis.py", "benchmarks/run_stage_v.py",
                                 "benchmarks/stage_v_report.py")
    h = hashlib.sha256()
    for rel in svp.HARNESS_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"# {rel}\n")
        h.update(rel.encode() + b"\0" + f"# {rel}\n".encode() + b"\0")
    assert svp.harness_sha256(tmp_path) == h.hexdigest()
    (tmp_path / "benchmarks" / "stage_v_report.py").write_text("# changed\n")
    assert svp.harness_sha256(tmp_path) != h.hexdigest()


def test_the_freeze_evidence_is_a_passed_mini_of_this_harness_on_the_frozen_request_path(tmp_path, monkeypatch):
    monkeypatch.setattr(svp, "harness_sha256", lambda root=svp.ROOT: "b" * 64)
    assert svp.freeze_evidence_problems(CFG, tmp_path) == []                 # before the freeze
    c = frozen_cfg()
    assert any("no readable" in p for p in svp.freeze_evidence_problems(c, tmp_path))
    d = tmp_path.joinpath(*svp.RESULTS, "smoke-v-20261101")
    d.mkdir(parents=True)
    good = {"mode": "mini", "session": "mini", "formal": False, "smoke": {"ok": True}, "harness_sha256": "b" * 64,
            "gateway_path_sha256": HEX}
    (d / "session.json").write_text(json.dumps(good))
    assert svp.freeze_evidence_problems(c, tmp_path) == []
    for key, value, needle in (("smoke", {"ok": False}, "passed"), ("formal", True, "passed"),
                               ("session", "V-ADM", "passed"), ("harness_sha256", "0" * 64, "another harness"),
                               ("gateway_path_sha256", "0" * 64, "gateway_path_sha256")):
        (d / "session.json").write_text(json.dumps({**good, key: value}))
        assert any(needle in p for p in svp.freeze_evidence_problems(c, tmp_path)), key


# ------------------------------------------------------------------ plans --

def test_a_formal_plan_needs_the_freeze_and_the_mini_is_never_formal():
    with pytest.raises(svp.PlanError, match="freeze.stage_v"):
        plan("V-ADM", formal=True)
    assert plan("V-ADM", c=frozen_cfg(), formal=True)["session"] == "V-ADM"
    with pytest.raises(svp.PlanError, match="mini"):
        plan("mini", c=frozen_cfg(), formal=True)
    with pytest.raises(svp.PlanError, match="unknown session"):
        plan("D-C")
    with pytest.raises(svp.PlanError, match="stage_d_sha256"):
        plan("V-CLS", stage_d_sha256=HEX)
    bad = copy.deepcopy(CFG)
    bad["stage_v"]["n"] = 3
    with pytest.raises(svp.PlanError, match="stage_v.toml"):
        plan("V-CLS", c=bad)
    sd_bad = copy.deepcopy(SD_CFG)
    sd_bad["stage_d"]["tau1_s"]["2048"] = 1.0
    with pytest.raises(svp.PlanError, match="stage_d.toml"):
        plan("V-CLS", sd_cfg=sd_bad)


def test_the_formal_plans():
    for session in ("V-ADM", "V-CLS"):
        p = plan(session)
        assert list(p["points"]) == svp.SESSIONS[session] and p["n"] == 5 and p["mini"] is None
        assert (p["stage"], p["kind"]) == ("V", "verify")             # D-C's kind: run_stage_d's harness reads it
        assert p["block_orders"] == {str(r): cp.block_order(20261008, r, svp.SESSIONS[session]) for r in range(1, 6)}
        assert p["slo"] == cp.SLO_S2 and p["window"] == "steady" and p["guard_band"] == 0.05
        assert p["hour_cap_h"] == {"V-ADM": 3.5, "V-CLS": 7.0}[session] and p["total_cap_h"] == 12
        assert (p["min_gap_h"], p["end_reserve_h"]) == (2, 0.06)
        assert p["sentinel"] == {"workload": "U2K", "L": 2048, "mix": "uniform", "C": 25, "N": 250, "tolerance": 0.03,
                                 "timeout_s": 1020, "server": "B0", "bench_est_s": 103, "run_overhead_s": 95,
                                 "est_s": 198}
        assert (p["gateway_port"], p["gateway_drain_s"]) == (8080, 5)
        assert p["gateway_path_files"] == CFG["stage_v"]["gateway_path_files"] and p["gateway_path_sha256"] == ""
        assert (p["pins_sha256"], p["stage_v_sha256"]) == ("p" * 64, "v" * 64)
        assert p["stage_d_sha256"] == hashlib.sha256(SD_RAW).hexdigest()
        sb = PINS["stage_b"]
        assert (p["expected_blocks"], p["block_size"], p["kv_cache_memory_bytes"]) == (
            sb["num_gpu_blocks"], sb["block_size"], sb["kv_cache_memory_bytes"])
        assert p["long_prefill_token_threshold"] == 512 and p["server_ready_timeout_s"] == 1800
        used = {k for x in p["points"].values() for k in (x["datasets"] or {}).values()}
        assert p["datasets"] == {k: CFG["stage_v"]["datasets"][k] for k in used}
        for name, x in p["points"].items():
            assert x["skip_group"] == svp.skip_group(name) and x["forced_timeout_blocks"] == []
            assert x["queue_settings"] == CFG["stage_v"]["queue"][x["queue"]]
            assert x["judged_views"] == {"client": ["client"], "upstream": ["upstream"],
                                         "worse": ["upstream", "client"]}[x["view"]]
    assert plan("V-ADM")["units"] == {"PACE-2K": ["P-U2K"], "PACE-8K": ["P-U8K"], "W1": ["W-90", "W-50"]}
    assert plan("V-ADM")["vetoes"] == svp.VETOES and plan("V-CLS")["vetoes"] == {}
    assert plan("V-CLS")["units"] == {"VG-32K": ["GW-U32K"], "VG-M8K-T": ["GW-M8K-T"],
                                      "V1-2K": ["L-U2K", "GW-U2K-O512"], "V1-8K": ["L-U8K", "GW-U8K-O512"]}
    assert plan("V-ADM")["points"]["W-90"]["admission_params"] == svp.WEIGHTED_V1
    assert plan("V-ADM")["points"]["P-U8K"]["pacing_params"] == {"burst_tokens": 32256, "rate_tokens_per_s": 5800}
    assert plan("V-ADM", c=frozen_cfg())["gateway_path_sha256"] == HEX


def test_the_mini_plan():
    """044 mini: N 20, n 2, sentinel at C 2, the mini's C per point, v1-mini-8192 for L-U8K, W-mini's 33-unit wave,
    GW-U8K-O512's forced timeout in block 2; formal pacing and policy parameters."""
    p = plan("mini")
    assert p["mini"] == svp.MINI and p["n"] == 2 and p["hour_cap_h"] == 1.0
    assert {n: x["C"] for n, x in p["points"].items()} == {"GW-U8K-O512": 4, "L-U8K": 4, "GW-M8K-T": 2, "P-U2K": 8,
                                                          "F-U2K-var": 4, "W-mini": 9}
    assert all(x["N"] == 20 and x["n"] == 2 for x in p["points"].values())
    assert (p["sentinel"]["C"], p["sentinel"]["N"]) == (2, 20)
    assert p["points"]["L-U8K"]["datasets"] == {"1": "v1-mini-8192", "2": "v1-mini-8192"}
    assert p["points"]["W-mini"]["datasets"] == {"1": "v4-mini", "2": "v4-mini"}
    assert p["points"]["GW-U8K-O512"]["forced_timeout_blocks"] == [2]
    assert p["points"]["GW-U8K-O512"]["limits"]["max_concurrency"] == 12      # /readyz shows the class's 12
    assert p["points"]["L-U8K"]["skip_group"] == ["L-U8K", "GW-U8K-O512"]    # the unit dies with the forced timeout
    assert p["points"]["P-U2K"]["pacing_params"] == {"burst_tokens": 9600, "rate_tokens_per_s": 5920}
    assert p["points"]["W-mini"]["admission_params"] == svp.WEIGHTED_V1
    assert p["points"]["W-mini"]["timeout_s"] == sd.timeout_s(sd.tau1(SD_CFG, "U8K-gw"), 20)
    assert p["estimate_s"] / 3600 < 0.8 * p["hour_cap_h"]                     # about 0.6 h under a 1.0 h cap
    assert p["block_orders"] == {str(r): cp.block_order(20261008, r, svp.SESSIONS["mini"]) for r in (1, 2)}


def test_main_writes_the_plan(tmp_path, monkeypatch):
    out = tmp_path / "plan.json"
    assert svp.main(["--session", "V-CLS", "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    want = svp.build_plan(CFG, PINS, SD_CFG, session_name="V-CLS",
                          pins_sha256=hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest(),
                          stage_v_sha256=hashlib.sha256(CFG_RAW).hexdigest(),
                          stage_d_sha256=hashlib.sha256(SD_RAW).hexdigest())
    assert got == json.loads(json.dumps(want)) and svp.plan_sha256(got) == svp.plan_sha256(want)
    with pytest.raises(SystemExit):
        svp.main(["--session", "V-CLS", "--out", str(out), "--formal"])


def test_the_plan_module_imports_only_frozen_research_modules():
    """044 Harness: capacity_plan, core_b_plan, stage_d_plan and run_core_a.sha256_json, read-only; never capserve."""
    tree = ast.parse((ROOT / "benchmarks" / "stage_v_plan.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add(node.module.split(".")[0])
    assert names - set(sys.stdlib_module_names) == {"capacity_plan", "core_b_plan", "stage_d_plan", "run_core_a"}
    froms = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "run_core_a"]
    assert [[a.name for a in n.names] for n in froms] == [["sha256_json"]]
