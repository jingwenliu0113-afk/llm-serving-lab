"""Tests for benchmarks/capacity_plan.py and configs/capacity.toml: the Stage C (C1) settings and the deterministic
search / confirmation / Phase C order of decision 030 (PREREGISTRATION §8), fixed before any C1 data."""

import copy
import hashlib
import json
import math
import random
import statistics
import tomllib
from fractions import Fraction
from pathlib import Path

import pytest

import capacity_plan as cp
import core_b_plan as bp

ROOT = Path(__file__).resolve().parent.parent
UNIFORM_SESSION = ROOT / "results" / "core_b" / "formal-20261009-coreb-uniform"
MIXED2_SESSION = ROOT / "results" / "core_b" / "formal-20261010-coreb-mixed2" / "session.json"
S1 = {"id": "S1", "ttft_p95_ms": {"2048": 1000, "8192": 3000, "32768": 10000}, "tpot_p95_ms": 50,
      "mixed_long": "budget"}
S2 = {"id": "S2", "ttft_p95_ms": {"2048": 2000, "8192": 6000, "32768": 20000}, "tpot_p95_ms": 100,
      "mixed_long": "budget"}
S3 = {"id": "S3", "ttft_p95_ms": {"2048": 2000, "8192": 6000, "32768": 20000}, "tpot_p95_ms": 200,
      "mixed_long": "budget"}
OBS_FLAGS = ("--otlp-traces-endpoint", "--enable-logging-iteration-details", "--long-prefill-token-threshold",
             "--no-scheduler-reserve-full-isl", "--request-id-prefix")


@pytest.fixture(scope="module")
def cfg_bytes():
    return (ROOT / "configs" / "capacity.toml").read_bytes()


@pytest.fixture(scope="module")
def cfg(cfg_bytes):
    return tomllib.loads(cfg_bytes.decode())


@pytest.fixture(scope="module")
def draft(cfg):
    """The repo's capacity.toml as it was before the freeze (030 DRAFT): every value the same, freeze.capacity false."""
    return edited(cfg, "freeze.capacity", False)


@pytest.fixture(scope="module")
def draft_bytes(cfg_bytes):
    assert cfg_bytes.count(b"\ncapacity = true\n") == 1
    return cfg_bytes.replace(b"\ncapacity = true\n", b"\ncapacity = false\n")


@pytest.fixture(scope="module")
def pins():
    return tomllib.loads((ROOT / "configs" / "pins.toml").read_text())


@pytest.fixture(scope="module")
def core_session():
    return cp.load_uniform_session(UNIFORM_SESSION)


def edited(cfg, path, value):
    """A copy of cfg with the dotted key path set to value (None deletes it)."""
    c = copy.deepcopy(cfg)
    *parents, last = path.split(".")
    node = c
    for p in parents:
        node = node[p]
    if value is None:
        del node[last]
    else:
        node[last] = value
    return c


EVIDENCE = {"fs_order_smoke": "smoke-fs-order-20261016", "mini_smoke": "smoke-mini-20261016", "c0_decision": "031"}


def frozen(cfg, branch="A"):
    out = edited(edited(cfg, "freeze.capacity", True), "capacity.c0_branch", branch)
    for key, value in EVIDENCE.items():
        out = edited(out, f"freeze.{key}", value)
    return out


def build(cfg, pins, core_session, **kw):
    session, up = core_session
    return cp.build_plan(cfg, pins, session=session, uniform_plan=up, pins_sha256="ab" * 32,
                         capacity_sha256="cd" * 32, **kw)


# ------------------------------------------------------------------ the config file (030)

def test_repo_capacity_toml_follows_030(cfg, pins):
    assert cp.capacity_problems(cfg, pins) == []
    c = cfg["capacity"]
    assert cfg["schema_version"] == 1
    assert cfg["freeze"]["capacity"] is True           # frozen 2026-10-17 after the harness, smokes and C0 (030, 032)
    assert c["c0_branch"] == "A"                         # C0's result decision log 032: Rrun (028)
    assert (c["seed"], c["derived_from"]) == (20261008, "formal-20261009-coreb-uniform")
    assert c["workloads"] == ["U2K", "U8K", "M2K", "M8K"]
    assert (c["window"], c["guard_band"]) == ("steady", 0.05)
    assert (c["n_near"], c["n_near_bc"], c["n_low"]) == (5, 7, 3)
    assert c["n_near_bc_workloads"] == ["U8K"]
    assert c["c_cap"] == {"max": 128, "kv_factor": 2}
    assert c["tau1_s"] == {"2048": 1.845, "8192": 2.95, "32768": 7.92}
    assert c["server_ready_timeout_s"] == 1800
    assert c["sentinel"] == {"L": 2048, "mix": "uniform", "C": 25, "tolerance": 0.03}
    assert cfg["slo"] == S2                                # the user's choice (030 使用者決定)
    assert cfg["secondary_slo"] == {"S1": S1, "S3": S3}    # descriptive bounds only (030 8.2 / 8.5)


def test_capacity_toml_reuses_the_pinned_mixed_datasets(cfg, pins):
    """Stage C reads pins.toml read-only (030 implementation note); the key name is pins' own."""
    assert cfg["capacity"]["mix_dataset_sha256"] == pins["core_b"]["mix_dataset_sha256"]
    assert set(cfg["capacity"]["mix_dataset_sha256"]) == {"2048", "8192"}


def test_pins_toml_carries_no_stage_c_setting(pins):
    """030 implementation note: every Stage C value lives in capacity.toml, pins.toml stays the Advanced freeze."""
    assert "capacity" not in pins and "capacity" not in pins["freeze"] and "slo" not in pins


def test_code_constants_are_the_030_values():
    assert cp.SLO_S2 == S2 and cp.SECONDARY_SLO == {"S1": S1, "S3": S3}
    assert cp.WORKLOADS == ("U2K", "U8K", "M2K", "M8K")
    assert cp.WORKLOAD_SPEC == {"U2K": (2048, "uniform"), "U8K": (8192, "uniform"), "M2K": (2048, "mixed"),
                                "M8K": (8192, "mixed")}
    assert (cp.GUARD_BAND, cp.N_NEAR, cp.N_NEAR_BC, cp.N_LOW) == (0.05, 5, 7, 3)
    assert cp.C_CAP == {"max": 128, "kv_factor": 2}
    assert cp.SEED == bp.SEED == 20261008


@pytest.mark.parametrize("path,value", [
    ("schema_version", 2), ("capacity.seed", 1), ("capacity.derived_from", "formal-20261010-coreb-mixed2"),
    ("capacity.workloads", ["U2K", "U8K", "M2K"]), ("capacity.workloads", ["U8K", "U2K", "M2K", "M8K"]),
    ("capacity.workloads", ["U2K", "U8K", "M2K", "M8K", "U32K"]),
    ("capacity.window", "full"), ("capacity.guard_band", 0.1), ("capacity.guard_band", 0),
    ("capacity.n_near", 4), ("capacity.n_near", 5.0), ("capacity.n_near_bc", 6), ("capacity.n_low", 2),
    ("capacity.n_low", True), ("capacity.c_cap", {"max": 256, "kv_factor": 2}),
    ("capacity.c_cap", {"max": 128, "kv_factor": 1}), ("capacity.c_cap", {"max": 128}),
    ("capacity.tau1_s", {"2048": 1.845, "8192": 2.95}), ("capacity.tau1_s", {"2048": 1.9, "8192": 2.95, "32768": 7.92}),
    ("capacity.server_ready_timeout_s", 600), ("capacity.sentinel", {"L": 2048, "mix": "uniform", "C": 8,
                                                                   "tolerance": 0.03}),
    ("capacity.sentinel", {"L": 2048, "mix": "uniform", "C": 25, "tolerance": 0.05}),
    ("capacity.mix_dataset_sha256", {"2048": "00" * 32,
                                     "8192": "ab548bed27570845e58c92888939bc4cf9872e35f51b774ba34d63a6405829b2"}),
    ("capacity.mix_dataset_sha256", None), ("capacity.c0_branch", "E"), ("capacity.c0_branch", None),
    ("capacity.n_near_bc_workloads", []), ("capacity.n_near_bc_workloads", ["M8K"]),
    ("capacity.n_near_bc_workloads", ["U8K", "U32K"]), ("capacity.n_near_bc_workloads", ["U8K", "U8K"]),
    ("capacity.extra", 1), ("slo", S1), ("slo", S3), ("slo", {**S2, "tpot_p95_ms": 100.0}),
    ("slo", {**S2, "mixed_long": "best_effort"}), ("slo", {**S2, "ttft_p95_ms": {"2048": 2000, "8192": 6325,
                                                                                "32768": 20000}}),
    ("slo", {**S2, "p99": True}), ("secondary_slo", {"S1": S1}), ("secondary_slo", {"S1": S1, "S3": S2}),
    ("secondary_slo", {"S1": {**S1, "tpot_p95_ms": 60}, "S3": S3}), ("freeze.capacity", "yes"), ("freeze", None)])
def test_capacity_problems_catch_each_deviation_from_030(cfg, pins, path, value):
    bad = edited(cfg, path, value)
    assert len(cp.capacity_problems(bad, pins)) >= 1


def test_branch_c_limits_seven_restarts_to_uniform_8k(cfg, pins):
    """030 C0 branches: B = the affected workloads (at least U8K), C = U8K only."""
    b = edited(cfg, "capacity.c0_branch", "B")
    assert cp.capacity_problems(edited(b, "capacity.n_near_bc_workloads", ["U8K", "M8K"]), pins) == []
    c = edited(cfg, "capacity.c0_branch", "C")
    assert cp.capacity_problems(c, pins) == []
    assert len(cp.capacity_problems(edited(c, "capacity.n_near_bc_workloads", ["U8K", "M8K"]), pins)) == 1


@pytest.mark.parametrize("branch,ok", [("A", True), ("B", True), ("C", True), ("D", False), ("", False)])
def test_freeze_needs_a_c0_branch_a_b_or_c(cfg, draft, pins, branch, ok):
    """028 / 030: D (a harness defect) or a missing branch keeps §8 unfrozen."""
    assert (cp.capacity_problems(frozen(cfg, branch), pins) == []) is ok
    assert cp.capacity_problems(edited(draft, "capacity.c0_branch", branch), pins) == []  # any branch in DRAFT


def test_freeze_table_records_the_evidence(cfg):
    """030 implementation note: the freeze records the passed FS-order and mini smokes and C0's result decision log
    (032); the committed sessions and the log satisfy freeze_evidence_problems."""
    assert cfg["freeze"] == {"capacity": True, "fs_order_smoke": "smoke-fs-order-202610161531",
                             "mini_smoke": "smoke-mini-202610161531", "c0_decision": "032"}
    assert cp.freeze_evidence_problems(cfg, ROOT) == []


@pytest.mark.parametrize("key,value", [("fs_order_smoke", ""), ("mini_smoke", ""), ("c0_decision", ""),
                                       ("c0_decision", "31"), ("c0_decision", "abc"), ("fs_order_smoke", "../x"),
                                       ("mini_smoke", 3)])
def test_freeze_needs_the_smoke_and_c0_evidence(cfg, pins, key, value):
    assert cp.capacity_problems(frozen(cfg), pins) == []
    assert len(cp.capacity_problems(edited(frozen(cfg), f"freeze.{key}", value), pins)) >= 1


def _smoke_session(root, sid, mode, ok=True, formal=False):
    d = root / "results" / "capacity" / sid
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps({"session_id": sid, "mode": mode, "formal": formal,
                                                "smoke": {"ok": ok}}))


def test_freeze_evidence_reads_the_smoke_sessions_and_the_c0_log(cfg, draft, tmp_path):
    f = frozen(cfg)
    assert cp.freeze_evidence_problems(draft, tmp_path) == []        # nothing to check before the freeze
    assert len(cp.freeze_evidence_problems(f, tmp_path)) == 3        # two missing sessions, a missing decision log
    _smoke_session(tmp_path, EVIDENCE["fs_order_smoke"], "fs-order")
    _smoke_session(tmp_path, EVIDENCE["mini_smoke"], "mini")
    (tmp_path / "docs" / "decisions").mkdir(parents=True)
    (tmp_path / "docs" / "decisions" / "031-c0-result.md").write_text("# 031\n")
    assert cp.freeze_evidence_problems(f, tmp_path) == []


@pytest.mark.parametrize("swap", ["failed", "formal", "wrong_mode"])
def test_freeze_evidence_refuses_a_failed_formal_or_mislabelled_smoke(cfg, tmp_path, swap):
    _smoke_session(tmp_path, EVIDENCE["fs_order_smoke"], "mini" if swap == "wrong_mode" else "fs-order",
                   ok=swap != "failed", formal=swap == "formal")
    _smoke_session(tmp_path, EVIDENCE["mini_smoke"], "mini")
    (tmp_path / "docs" / "decisions").mkdir(parents=True)
    (tmp_path / "docs" / "decisions" / "031-c0-result.md").write_text("# 031\n")
    assert len(cp.freeze_evidence_problems(frozen(cfg), tmp_path)) == 1


def test_the_repo_freeze_evidence_holds_once_frozen(cfg):
    assert cp.freeze_evidence_problems(cfg, ROOT) == []


@pytest.mark.parametrize("stage", ["mac", "nvidia", "advanced"])
def test_freeze_needs_every_earlier_stage_frozen(cfg, draft, pins, stage):
    p = copy.deepcopy(pins)
    p["freeze"][stage] = False
    assert len(cp.capacity_problems(frozen(cfg), p)) == 1
    assert len(cp.capacity_problems(cfg, p)) == 1
    assert cp.capacity_problems(draft, p) == []


def test_n_near_per_workload_and_branch(cfg):
    for branch in ("", "A", "D"):
        c = edited(cfg, "capacity.c0_branch", branch)
        assert {w: cp.n_near_for(c, w) for w in cp.WORKLOADS} == dict.fromkeys(cp.WORKLOADS, 5)
    c = edited(edited(cfg, "capacity.c0_branch", "B"), "capacity.n_near_bc_workloads", ["U8K", "M8K"])
    assert {w: cp.n_near_for(c, w) for w in cp.WORKLOADS} == {"U2K": 5, "U8K": 7, "M2K": 5, "M8K": 7}
    c = edited(cfg, "capacity.c0_branch", "C")
    assert {w: cp.n_near_for(c, w) for w in cp.WORKLOADS} == {"U2K": 5, "U8K": 7, "M2K": 5, "M8K": 5}


# ------------------------------------------------------------------ C_KV, cap and timeouts

def test_c_kv_and_cap_come_from_block_arithmetic(pins):
    """030 8.4: C_cap(L) = min(128, 2 C_KV(L)), C_KV from core_b_plan.c_kv on the Stage B pool: 70 / 26."""
    assert (cp.c_kv_for(pins, 2048), cp.c_kv_for(pins, 8192)) == (35, 13)
    assert (cp.c_cap_for(35), cp.c_cap_for(13)) == (70, 26)
    assert cp.c_cap_for(64) == 128 and cp.c_cap_for(100) == 128 and cp.c_cap_for(1) == 2


def test_tau1_is_the_core_c1_time_per_request(pins):
    """030 8.4: tau1 = Core's C 1 bench time per request (N 200), checked against the committed rows."""
    rows = [json.loads(x) for x in (UNIFORM_SESSION / "rows.jsonl").read_text().splitlines() if x.strip()]
    for L, tau in cp.TAU1_S.items():
        per_req = statistics.mean(r["duration"] / r["num_prompts"] for r in rows
                                  if r["L"] == int(L) and r["C"] == 1 and r["ok"])
        assert abs(tau / per_req - 1) < 0.005, (L, tau, per_req)


def test_bench_timeout_is_ceil60_of_twice_the_serial_time_plus_60():
    tau = cp.TAU1_S
    assert cp.bench_timeout_s(2048, "uniform", 1, tau) == 840     # 2 x 200 x 1.845 + 60 = 798
    assert cp.bench_timeout_s(2048, "uniform", 25, tau) == 1020   # 2 x 250 x 1.845 + 60 = 982.5
    assert cp.bench_timeout_s(8192, "uniform", 26, tau) == 1620   # 2 x 260 x 2.95 + 60 = 1594
    assert cp.per_request_s(2048, "mixed", tau) == Fraction("2.4525")  # 0.9 x 1.845 + 0.1 x 7.92
    assert cp.bench_timeout_s(2048, "mixed", 1, tau) == 1080      # 2 x 200 x 2.4525 + 60 = 1041
    assert cp.per_request_s(8192, "mixed", tau) == Fraction("3.447")
    assert cp.bench_timeout_s(8192, "mixed", 8, tau) == 1440      # 1438.8
    # an exact multiple of 60 stays (no float rounding up): 2 x 200 x 0.15 + 60 = 120
    assert cp.bench_timeout_s(2048, "uniform", 1, {**tau, "2048": 0.15}) == 120
    assert cp.bench_timeout_s(2048, "uniform", 1, {**tau, "2048": 0.1500001}) == 180


# ------------------------------------------------------------------ Phase S (030 8.4)

def test_bracket_points():
    assert cp.bracket_points(70) == [1, 2, 4, 8, 16, 32, 64, 70]
    assert cp.bracket_points(26) == [1, 2, 4, 8, 16, 26]
    assert cp.bracket_points(64) == [1, 2, 4, 8, 16, 32, 64]
    assert cp.bracket_points(2) == [1, 2] and cp.bracket_points(1) == [1]


def run_search(c_cap, capacity):
    """Phase S against a monotone truth: a run passes iff C <= capacity."""
    hist = []
    while (c := cp.search_next(c_cap, hist)) is not None:
        hist.append((c, c <= capacity))
    return [c for c, _ in hist], cp.search_next(c_cap, hist)


@pytest.mark.parametrize("c_cap,capacity,want", [
    (70, 33, [1, 2, 4, 8, 16, 32, 64, 48, 40, 36, 34, 33]),
    (26, 8, [1, 2, 4, 8, 16, 12, 10, 9]),
    (70, 70, [1, 2, 4, 8, 16, 32, 64, 70]),                     # censored at the cap: no fail
    (70, 66, [1, 2, 4, 8, 16, 32, 64, 70, 67, 65, 66]),
    (70, 0, [1]),                                               # C 1 fails: C^ = 0
    (26, 1, [1, 2]), (26, 25, [1, 2, 4, 8, 16, 26, 21, 23, 24, 25])])
def test_search_brackets_then_bisects(c_cap, capacity, want):
    seq, after = run_search(c_cap, capacity)
    assert seq == want and after is None
    hist = [(c, c <= capacity) for c in seq]
    assert cp.search_summary(c_cap, [{"C": c, "pass": p, "within_band": p} for c, p in hist])["c_hat"] == \
        min(capacity, c_cap)


def test_search_is_exhaustive_and_recovers_every_monotone_capacity():
    for c_cap in (1, 2, 3, 26, 70, 128):
        for capacity in range(0, c_cap + 2):
            seq, _ = run_search(c_cap, capacity)
            assert len(seq) == len(set(seq))                      # never repeats a point
            summary = cp.search_summary(c_cap, [{"C": c, "pass": c <= capacity, "within_band": False} for c in seq])
            assert summary["c_hat"] == min(capacity, c_cap)


def test_search_next_refuses_a_history_the_rule_would_not_run():
    with pytest.raises(cp.PlanError):
        cp.search_next(70, [(2, True)])                           # must start at 1
    with pytest.raises(cp.PlanError):
        cp.search_next(70, [(1, True), (2, True), (8, True)])    # skipped 4
    with pytest.raises(cp.PlanError):
        cp.search_next(70, [(1, False), (2, False)])             # over after C 1 failed
    with pytest.raises(cp.PlanError):
        cp.search_summary(70, [{"C": 1, "pass": True, "within_band": True}])  # incomplete


def test_phase_s_retries_each_failed_boot_once_at_the_end_in_canonical_order():
    """030 8.4 step 4: a boot that fails before its first bench is retried once after every other Phase S boot."""
    assert cp.phase_s_retries(cp.WORKLOADS, set()) == []
    assert cp.phase_s_retries(cp.WORKLOADS, {"M8K", "U2K"}) == ["U2K", "M8K"]


def test_guard_band_point():
    """030 8.4 step 3: the largest tested c with every tested c' <= c inside the band."""
    h = [{"C": c, "pass": p, "within_band": b} for c, p, b in
         [(1, True, True), (2, True, True), (4, True, True), (8, True, True), (16, True, True), (32, True, False),
          (64, False, False), (48, False, False), (40, False, False), (36, False, False), (34, False, False),
          (33, True, False)]]
    assert cp.guard_point(h) == 16
    assert cp.search_summary(70, h) == {"c_hat": 33, "c_hat_g": 16, "f": 34}
    assert cp.guard_point([{"C": 1, "pass": True, "within_band": False},
                           {"C": 2, "pass": True, "within_band": True}]) == 0
    assert cp.guard_point([]) == 0


# ------------------------------------------------------------------ confirmation set (030 8.4)

def pts(points):
    return {p["C"]: (p["role"], p["n"]) for p in points}


@pytest.mark.parametrize("c_hat,c_hat_g,c_cap,want", [
    (0, 0, 70, {1: ("one", 5)}),
    (1, 1, 70, {1: ("near", 5), 2: ("near", 5)}),
    (33, 16, 70, {16: ("low", 3), 32: ("near", 5), 33: ("near", 5), 34: ("near", 5)}),
    (33, 32, 70, {17: ("low", 3), 32: ("near", 5), 33: ("near", 5), 34: ("near", 5)}),   # C^_g = C^ - 1: ceil(C^/2)
    (33, 0, 70, {17: ("low", 3), 32: ("near", 5), 33: ("near", 5), 34: ("near", 5)}),
    (3, 3, 26, {2: ("near", 5), 3: ("near", 5), 4: ("near", 5)}),                      # ceil(3/2) = 2 >= C^ - 1
    (4, 0, 26, {2: ("low", 3), 3: ("near", 5), 4: ("near", 5), 5: ("near", 5)}),
    (26, 8, 26, {8: ("low", 3), 25: ("near", 5), 26: ("near", 5)}),                    # C^ + 1 above the cap
    (2, 1, 26, {1: ("near", 5), 2: ("near", 5), 3: ("near", 5)})])
def test_confirm_points(c_hat, c_hat_g, c_cap, want):
    assert pts(cp.confirm_points(c_hat, c_hat_g, c_cap, n_near=5, n_low=3)) == want


def test_confirm_points_use_seven_restarts_near_the_boundary_under_b_or_c():
    assert pts(cp.confirm_points(8, 4, 26, n_near=7, n_low=3)) == {4: ("low", 3), 7: ("near", 7), 8: ("near", 7),
                                                                   9: ("near", 7)}
    assert pts(cp.confirm_points(0, 0, 26, n_near=7, n_low=3)) == {1: ("one", 7)}


def hist(seq_pass_band):
    return [{"C": c, "pass": p, "within_band": b} for c, p, b in seq_pass_band]


def searches_for(capacity, band, c_caps):
    out = {}
    for w, c_cap in c_caps.items():
        seq, _ = run_search(c_cap, capacity[w])
        out[w] = hist([(c, c <= capacity[w], c <= band[w]) for c in seq])
    return out


@pytest.fixture(scope="module")
def plan(cfg, pins, core_session):
    return build(cfg, pins, core_session)


@pytest.fixture(scope="module")
def confirm(plan):
    searches = searches_for({"U2K": 33, "U8K": 8, "M2K": 1, "M8K": 0}, {"U2K": 16, "U8K": 4, "M2K": 1, "M8K": 0},
                            {w: plan["workloads"][w]["c_cap"] for w in cp.WORKLOADS})
    return cp.build_confirm_plan(plan, searches)


def test_confirm_plan_is_a_pure_function_of_the_phase_s_rows(plan, confirm):
    wl = confirm["workloads"]
    assert {w: (wl[w]["c_hat"], wl[w]["c_hat_g"]) for w in cp.WORKLOADS} == \
        {"U2K": (33, 16), "U8K": (8, 4), "M2K": (1, 1), "M8K": (0, 0)}
    assert pts(wl["U2K"]["points"]) == {16: ("low", 3), 32: ("near", 5), 33: ("near", 5), 34: ("near", 5)}
    assert pts(wl["U8K"]["points"]) == {4: ("low", 3), 7: ("near", 5), 8: ("near", 5), 9: ("near", 5)}
    assert pts(wl["M2K"]["points"]) == {1: ("near", 5), 2: ("near", 5)}
    assert pts(wl["M8K"]["points"]) == {1: ("one", 5)}
    assert confirm["blocks"] == 5
    assert json.loads(json.dumps(confirm)) == confirm
    searches = {w: wl[w]["search"] for w in cp.WORKLOADS}
    assert cp.build_confirm_plan(plan, searches) == confirm
    assert cp.confirm_plan_sha256(confirm) == hashlib.sha256(
        json.dumps(confirm, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_confirm_plan_orders_are_seeded_exactly_as_030(confirm):
    for r in range(1, 6):
        want = list(cp.WORKLOADS)
        random.Random(int(hashlib.sha256(f"20261008:capacity:block:{r}".encode()).hexdigest(), 16)).shuffle(want)
        assert confirm["block_orders"][str(r)] == want == cp.block_order(20261008, r, cp.WORKLOADS)
        for w in cp.WORKLOADS:
            order = sorted(p["C"] for p in confirm["workloads"][w]["points"])
            random.Random(int(hashlib.sha256(f"20261008:capacity:{w}:{r}".encode()).hexdigest(), 16)).shuffle(order)
            assert confirm["workloads"][w]["orders"][str(r)] == order
    assert len({tuple(confirm["block_orders"][str(r)]) for r in range(1, 6)}) > 1


def test_a_workload_whose_search_never_ran_gets_no_points(plan):
    searches = searches_for({"U2K": 33, "U8K": 8, "M2K": 1, "M8K": 0}, dict.fromkeys(cp.WORKLOADS, 0),
                            {w: plan["workloads"][w]["c_cap"] for w in cp.WORKLOADS})
    searches["U8K"] = None   # both Phase S boots failed before their first bench
    conf = cp.build_confirm_plan(plan, searches)
    assert conf["workloads"]["U8K"]["c_hat"] is None and conf["workloads"]["U8K"]["points"] == []
    assert conf["workloads"]["U8K"]["orders"] == {}
    assert all(w != "U8K" for w, _ in cp.phase_c_boots(conf, 1, {}))


def test_confirm_plan_refuses_an_incomplete_search(plan):
    searches = {w: hist([(1, True, True)]) for w in cp.WORKLOADS}
    with pytest.raises(cp.PlanError):
        cp.build_confirm_plan(plan, searches)


# ------------------------------------------------------------------ Phase C (030 8.4)

def test_phase_c_block_1_runs_every_point_in_the_seeded_orders(confirm):
    boots = cp.phase_c_boots(confirm, 1, {})
    assert [w for w, _ in boots] == confirm["block_orders"]["1"]
    for w, points in boots:
        assert points == confirm["workloads"][w]["orders"]["1"]


def test_a_valid_fail_or_an_invalid_run_drops_the_point_and_every_higher_one(confirm):
    first = dict(cp.phase_c_boots(confirm, 1, {}))
    outcomes = {"U2K": [(1, c, "fail" if c == 33 else "ok") for c in first["U2K"]],
                "U8K": [(1, c, "invalid" if c == 8 else "ok") for c in first["U8K"]],
                "M2K": [(1, c, "fail" if c == 2 else "ok") for c in first["M2K"]],
                "M8K": [(1, 1, "fail")]}
    second = dict(cp.phase_c_boots(confirm, 2, outcomes))
    assert sorted(second["U2K"]) == [16, 32]
    assert sorted(second["U8K"]) == [4, 7]
    assert second["M2K"] == [1]
    assert "M8K" not in second                                     # a boot with no active point is skipped
    order = confirm["workloads"]["U2K"]["orders"]["2"]
    assert second["U2K"] == [c for c in order if c in (16, 32)]   # the seeded order, filtered


def test_low_point_stops_after_its_three_restarts_and_near_points_run_five(confirm):
    outcomes = {w: [] for w in cp.WORKLOADS}
    seen = {w: {} for w in cp.WORKLOADS}
    for r in range(1, confirm["blocks"] + 1):
        for w, points in cp.phase_c_boots(confirm, r, outcomes):
            for c in points:
                outcomes[w].append((r, c, "ok"))
                seen[w][c] = seen[w].get(c, 0) + 1
    assert seen["U2K"] == {16: 3, 32: 5, 33: 5, 34: 5}
    assert seen["M8K"] == {1: 5}
    assert cp.phase_c_boots(confirm, confirm["blocks"] + 1, outcomes) == []


def test_skips_take_effect_only_in_later_blocks(confirm):
    """A FAIL inside block r never removes a point from that block's own boots (the list is fixed at block start)."""
    assert "U2K" not in dict(cp.phase_c_boots(confirm, 2, {"U2K": [(1, 16, "fail")]}))  # 16 is U2K's lowest point
    with pytest.raises(cp.PlanError):
        cp.phase_c_boots(confirm, 1, {"U2K": [(1, 16, "fail")]})   # outcomes of the block being planned
    with pytest.raises(cp.PlanError):
        cp.phase_c_boots(confirm, 2, {"U2K": [(1, 16, "maybe")]})


# ------------------------------------------------------------------ argv (030 8.1)

SERVER = {"port": 8000, "vllm_bin": ".venv/bin/vllm", "base_env": {"PATH": "/usr/bin", "VLLM_X": "1"}}


def test_server_argv_is_the_core_b_argv_with_the_stage_b_values_and_no_observation_flag(pins):
    argv, env = cp.server_argv_for(pins, **SERVER)
    sb = pins["stage_b"]
    core = bp.server_argv(pins, gdn_prefill_backend=sb["gdn_prefill_backend"],
                          gpu_memory_utilization=sb["gpu_memory_utilization"], port=8000, vllm_bin=".venv/bin/vllm",
                          base_env=SERVER["base_env"], kv_cache_memory_bytes=sb["kv_cache_memory_bytes"])
    assert (argv, env) == core
    assert not any(f in argv for f in OBS_FLAGS)
    assert argv[argv.index("--kv-cache-memory-bytes") + 1] == "4294967296"
    assert "VLLM_X" not in env


def test_server_argv_is_what_core_b_mixed2_ran(pins):
    if not MIXED2_SESSION.is_file():
        pytest.skip("formal-20261010-coreb-mixed2/session.json not present in this checkout")
    runs = json.loads(MIXED2_SESSION.read_text())["runs"]
    assert runs and all(cp.server_argv_for(pins, **SERVER)[0] == run["argv"] for run in runs.values())


@pytest.mark.parametrize("w", cp.WORKLOADS)
def test_bench_argv_is_core_b_bench_argv(pins, w):
    L, mix = cp.WORKLOAD_SPEC[w]
    ds = f"/d/mixed-{L}.jsonl" if mix == "mixed" else None
    got = cp.bench_argv_for(pins, workload=w, C=7, result_dir="/r", result_filename="x.json", mixed_dataset=ds,
                            vllm_bin="v")
    assert got == bp.bench_argv(pins, L=L, C=7, mix=mix, result_dir="/r", result_filename="x.json", port=8000,
                                mixed_dataset=ds, vllm_bin="v")
    assert not any(f in got for f in OBS_FLAGS)
    assert got[got.index("--num-prompts") + 1] == "200" and got[got.index("--max-concurrency") + 1] == "7"


def test_run_tags():
    assert cp.run_tag("S", "U2K", 33) == "S-U2K-c33"
    assert cp.run_tag("C", "M8K", 1, block=4) == "C-M8K-c1-r4"
    assert cp.run_tag("sentinel", "U2K", 25, position="end") == "sentinel-end-U2K-c25"
    with pytest.raises(cp.PlanError):
        cp.run_tag("C", "U2K", 3)
    with pytest.raises(cp.PlanError):
        cp.run_tag("S", "X", 3)


# ------------------------------------------------------------------ the plan

def test_plan_records_the_workloads_caps_timeouts_and_identities(plan, pins):
    import run_core_b

    assert (plan["schema_version"], plan["stage"], plan["seed"]) == (1, "C1", 20261008)
    assert plan["derived_from"] == "formal-20261009-coreb-uniform"
    assert (plan["slo"], plan["secondary_slo"]) == (S2, {"S1": S1, "S3": S3})
    assert (plan["window"], plan["guard_band"], plan["n_low"], plan["c0_branch"]) == ("steady", 0.05, 3, "A")
    assert list(plan["workloads"]) == list(cp.WORKLOADS)
    for w, (L, mix) in cp.WORKLOAD_SPEC.items():
        wl = plan["workloads"][w]
        ckv = {2048: 35, 8192: 13}[L]
        assert (wl["L"], wl["mix"], wl["c_kv"], wl["c_cap"], wl["n_near"]) == (L, mix, ckv, min(128, 2 * ckv), 5)
        assert wl["bracket"] == cp.bracket_points(wl["c_cap"])
        assert wl["timeouts_s"] == {str(c): cp.bench_timeout_s(L, mix, c, cp.TAU1_S) for c in range(1, wl["c_cap"] + 1)}
    assert plan["sentinel"] == {"workload": "U2K", "L": 2048, "mix": "uniform", "C": 25, "N": 250, "tolerance": 0.03,
                                "timeout_s": 1020}
    assert plan["server_ready_timeout_s"] == 1800 and plan["expected_blocks"] == 248
    assert plan["mix_dataset_sha256"] == pins["core_b"]["mix_dataset_sha256"]
    assert (plan["pins_sha256"], plan["capacity_sha256"]) == ("ab" * 32, "cd" * 32)
    assert json.loads(json.dumps(plan)) == plan
    assert run_core_b.stage_b_problems(plan, pins, gdn_prefill_backend="triton", gpu_memory_utilization="0.91") == []


def test_plan_n_near_follows_the_c0_branch(cfg, pins, core_session):
    plan = build(edited(cfg, "capacity.c0_branch", "C"), pins, core_session)
    assert {w: plan["workloads"][w]["n_near"] for w in cp.WORKLOADS} == {"U2K": 5, "U8K": 7, "M2K": 5, "M8K": 5}
    assert {w: plan["workloads"][w]["c0_covariate"] for w in cp.WORKLOADS} == \
        {"U2K": False, "U8K": True, "M2K": False, "M8K": False}   # 030 8.1 B / C: C0's per-run slip count recorded
    assert not any(x["c0_covariate"] for x in build(cfg, pins, core_session)["workloads"].values())


def test_the_c0_covariate_is_recorded_for_uniform_workloads_only(cfg, pins, core_session):
    """028's slip marker (each position's own baseline over idx = j mod C, +170 ms) is defined on a uniform lattice; on
    a mixed run every long request would count as a slip. A mixed workload that C0's decision log names under branch B
    still gets n_near_bc (030 8.1: n never drops), without the covariate."""
    c = edited(edited(cfg, "capacity.c0_branch", "B"), "capacity.n_near_bc_workloads", ["U8K", "M8K"])
    assert cp.capacity_problems(c, pins) == []
    plan = build(c, pins, core_session)
    assert {w: (x["n_near"], x["c0_covariate"]) for w, x in plan["workloads"].items()} == {
        "U2K": (5, False), "U8K": (7, True), "M2K": (5, False), "M8K": (7, False)}


@pytest.mark.parametrize("meta", [{"formal": False}, {"env_valid": False}, {"session_id": "formal-x"}])
def test_plan_refuses_another_or_an_invalid_uniform_session(cfg, pins, core_session, meta):
    session, up = core_session
    with pytest.raises(cp.PlanError):
        cp.build_plan(cfg, pins, session={**session, **meta}, uniform_plan=up, pins_sha256="ab" * 32,
                      capacity_sha256="cd" * 32)


@pytest.mark.parametrize("change", [{"c_kv": {"2048": 36, "8192": 13, "32768": 3}}, {"num_gpu_blocks": 186},
                                    {"block_size": 512}])
def test_plan_recomputes_c_kv_and_refuses_a_mismatch(cfg, pins, core_session, change):
    """C_KV is recomputed with core_b_plan.c_kv from the derived_from session's pool, never typed by hand."""
    session, up = core_session
    with pytest.raises(cp.PlanError, match="C_KV|blocks|block_size"):
        cp.build_plan(cfg, pins, session=session, uniform_plan={**up, **change}, pins_sha256="ab" * 32,
                      capacity_sha256="cd" * 32)


def test_plan_refuses_a_config_that_breaks_030(cfg, pins, core_session):
    with pytest.raises(cp.PlanError, match="guard_band"):
        build(edited(cfg, "capacity.guard_band", 0.1), pins, core_session)


def test_formal_plan_needs_the_freeze(cfg, draft, pins, core_session):
    with pytest.raises(cp.PlanError, match="freeze"):
        build(draft, pins, core_session, formal=True)
    assert build(frozen(cfg), pins, core_session, formal=True)["c0_branch"] == "A"
    assert build(cfg, pins, core_session, formal=True)["c0_branch"] == "A"     # the repo's frozen capacity.toml


# ------------------------------------------------------------------ CLI

def test_cli_writes_the_plan_with_both_sha256(tmp_path, cfg_bytes, monkeypatch, pins):
    pins_bytes = (ROOT / "configs" / "pins.toml").read_bytes()
    monkeypatch.setattr(cp, "load_capacity_bytes", lambda: cfg_bytes)
    monkeypatch.setattr(cp, "load_pins_bytes", lambda: pins_bytes)
    out = tmp_path / "plan.json"
    assert cp.main(["--uniform-session", str(UNIFORM_SESSION), "--out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert plan["capacity_sha256"] == hashlib.sha256(cfg_bytes).hexdigest()
    assert plan["pins_sha256"] == hashlib.sha256(pins_bytes).hexdigest()
    assert plan["workloads"]["U2K"]["c_cap"] == 70


def test_cli_refuses_a_formal_plan_before_the_freeze_and_writes_nothing(tmp_path, draft_bytes, monkeypatch):
    monkeypatch.setattr(cp, "load_capacity_bytes", lambda: draft_bytes)
    with pytest.raises(SystemExit):
        cp.main(["--uniform-session", str(UNIFORM_SESSION), "--out", str(tmp_path / "plan.json"), "--formal"])
    assert not (tmp_path / "plan.json").exists()
