"""Tests for benchmarks/adv_plan.py: the decision 024 server conditions (B0 / T / G / TG / K3), the deterministic
request-id prefix, the T and K3 rules, pins [advanced], and the 60-boot Advanced plan (PREREGISTRATION §7),
fixed before any Advanced data."""

import copy
import hashlib
import itertools
import json
import random
import tomllib
from pathlib import Path

import pytest

import adv_plan as ap
import core_b_plan as bp

ROOT = Path(__file__).resolve().parent.parent
MIXED2_SESSION = ROOT / "results" / "core_b" / "formal-20261010-coreb-mixed2" / "session.json"
GIB = 2**30
STEP = 256 * 2**20
OBS = ["--otlp-traces-endpoint", "http://127.0.0.1:4318/v1/traces", "--enable-logging-iteration-details"]
T_FLAGS = ["--long-prefill-token-threshold", "512"]
G_FLAGS = ["--no-scheduler-reserve-full-isl"]
SERVER = {"gdn_prefill_backend": "triton", "gpu_memory_utilization": "0.91", "port": 8000,
          "vllm_bin": ".venv/bin/vllm"}
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u", "VLLM_LOGGING_LEVEL": "DEBUG",
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "grpc", "OTEL_SDK_DISABLED": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://elsewhere:4318"}
TIMEOUTS = {"uniform_2048": 300, "mixed_2048": 600, "uniform_8192": 720, "mixed_8192": 960}
CONDITIONS_TABLE = {"B0": [], "T": T_FLAGS, "G": G_FLAGS, "TG": T_FLAGS + G_FLAGS, "K3": []}  # 024 argv deltas
CELLS = list(itertools.product(("B0", "T", "G", "TG", "K3"), (2048, 8192), ("uniform", "mixed")))


@pytest.fixture(scope="module")
def pins_bytes():
    return (ROOT / "configs" / "pins.toml").read_bytes()


@pytest.fixture(scope="module")
def pins(pins_bytes):
    return tomllib.loads(pins_bytes.decode())


def server(cond, pins, base_env=BASE_ENV):
    return ap.server_argv_for(cond, pins, base_env=base_env, **SERVER)


def uniform_session(pins, knee=None, drop=None, **meta):
    """A formal, env-valid uniform session whose output throughput saturates at `knee` (C_knee = that C)."""
    knee = knee or {2048: 25, 8192: 8}
    up = bp.build_uniform_plan(pins, num_gpu_blocks=248, block_size=528)
    rows = [{"L": L, "mix": "uniform", "restart": r, "C": C, "ok": True, "output_throughput": 100.0 * min(C, knee[L])}
            for L in (2048, 8192) for r in (1, 2, 3) for C in up["grids"][str(L)]]
    rows = [x for x in rows if not (drop and drop(x))]
    session = {"session_id": "formal-20261009-coreb-uniform", "formal": True, "env_valid": True, **meta}
    return session, up, rows


def build(pins, **kw):
    session, up, rows = uniform_session(pins, **kw)
    return ap.build_plan(pins, session=session, uniform_plan=up, rows=rows, pins_sha256="ab" * 32)


# ------------------------------------------------------------------ conditions --

def test_the_five_conditions_two_levels_two_mixes_three_restarts():
    assert ap.CONDITIONS == ("B0", "T", "G", "TG", "K3")
    assert ap.LEVELS == (2048, 8192) and ap.MIXES == ("uniform", "mixed") and ap.RESTARTS == 3
    assert ap.C_KNEE == {2048: 25, 8192: 8}


def test_b0_is_the_core_server_argv_plus_the_two_observability_flags(pins):
    core, _ = bp.server_argv(pins, base_env=BASE_ENV, kv_cache_memory_bytes=pins["stage_b"]["kv_cache_memory_bytes"],
                             **SERVER)
    assert server("B0", pins)[0] == core + OBS


def test_b0_is_the_argv_core_b_mixed2_recorded_plus_observability(pins):
    """024 test list: B0 = the argv formal-20261010-coreb-mixed2 ran with, plus the two observability flags."""
    if not MIXED2_SESSION.is_file():
        pytest.skip("formal-20261010-coreb-mixed2/session.json not present in this checkout")
    runs = json.loads(MIXED2_SESSION.read_text())["runs"]
    assert runs and all(server("B0", pins)[0] == run["argv"] + OBS for run in runs.values())


def test_t_g_and_tg_only_append_their_flags(pins):
    b0 = server("B0", pins)[0]
    assert server("T", pins)[0] == b0 + T_FLAGS
    assert server("G", pins)[0] == b0 + G_FLAGS
    tg = server("TG", pins)[0]
    assert tg == b0 + T_FLAGS + G_FLAGS
    assert tg.count("--long-prefill-token-threshold") == 1 and tg.count("--no-scheduler-reserve-full-isl") == 1


def test_k3_only_replaces_the_kv_cache_bytes(pins):
    b0, k3 = server("B0", pins)[0], server("K3", pins)[0]
    assert len(k3) == len(b0)
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(b0, k3)) if a != b]
    assert diff == [(b0.index("--kv-cache-memory-bytes") + 1, "4294967296", "3221225472")]
    assert k3.count("--kv-cache-memory-bytes") == 1


def test_no_condition_carries_another_conditions_manipulation(pins):
    for cond in ap.CONDITIONS:
        argv = server(cond, pins)[0]
        assert ("--long-prefill-token-threshold" in argv) == (cond in ("T", "TG")), cond
        assert ("--no-scheduler-reserve-full-isl" in argv) == (cond in ("G", "TG")), cond
        assert "--scheduler-reserve-full-isl" not in argv, cond
        assert argv[argv.index("--kv-cache-memory-bytes") + 1] == ("3221225472" if cond == "K3" else "4294967296")
        assert argv[-len(OBS) - len(ap.condition_flags(cond, pins)):][:len(OBS)] == OBS, cond
        assert argv.count("--otlp-traces-endpoint") == 1 and argv.count("--enable-logging-iteration-details") == 1


def test_server_env_drops_inherited_otel_and_sets_only_the_protocol(pins):
    _, core_env = bp.server_argv(pins, base_env=BASE_ENV, **SERVER)
    want = {**{k: v for k, v in core_env.items() if not k.startswith("OTEL_")},
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"}
    for cond in ap.CONDITIONS:
        assert server(cond, pins)[1] == want, cond
    assert ap.server_env_for(BASE_ENV, pins) == want
    assert sorted(set(core_env) - set(want)) == ["OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_SDK_DISABLED"]
    assert "VLLM_LOGGING_LEVEL" not in want  # Core's rule still applies (§4)
    # without inherited OTEL_* the only difference from Core is the added protocol
    clean = {"PATH": "/usr/bin", "HOME": "/home/u"}
    _, core_clean = bp.server_argv(pins, base_env=clean, **SERVER)
    assert ap.server_env_for(clean, pins) == {**core_clean, "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"}
    assert ap.server_env_for(None, pins) == {**bp.server_argv(pins, **SERVER)[1],
                                             "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"}


def test_bench_argv_differs_from_core_only_by_the_request_id_prefix(pins):
    for L, mix in itertools.product(ap.LEVELS, ap.MIXES):
        C = ap.C_KNEE[L]
        common = {"L": L, "C": C, "mix": mix, "result_dir": "/r", "result_filename": "x.json", "port": 8000,
                  "mixed_dataset": f"/d/mixed-{L}.jsonl" if mix == "mixed" else None, "vllm_bin": ".venv/bin/vllm"}
        core = bp.bench_argv(pins, **common)
        assert "--request-id-prefix" not in core
        assert ap.bench_argv_for(pins, cond="TG", restart=2, **common) == \
            core + ["--request-id-prefix", f"adv-TG-{L}-{mix}-r2-"]


def test_request_id_prefix_format_and_uniqueness():
    assert ap.request_id_prefix("B0", 2048, "uniform", 1) == "adv-B0-2048-uniform-r1-"
    assert ap.request_id_prefix("K3", 8192, "mixed", 3) == "adv-K3-8192-mixed-r3-"
    prefixes = {ap.request_id_prefix(c, L, m, r) for (c, L, m) in CELLS for r in (1, 2, 3)}
    assert len(prefixes) == 60
    for bad in (("X", 2048, "uniform", 1), ("B0", 32768, "uniform", 1), ("B0", 2048, "burst", 1),
                ("B0", 2048, "uniform", 4)):
        with pytest.raises(ap.PlanError):
            ap.request_id_prefix(*bad)


def test_unknown_condition_is_refused(pins):
    with pytest.raises(ap.PlanError):
        server("T512", pins)
    with pytest.raises(ap.PlanError):
        ap.condition_flags("b0", pins)


# --------------------------------------------------------------------- rules --

def test_threshold_rule_t_is_mnbt_over_long_requests_plus_one():
    """024 H-C1: T = MNBT // (C_KV(32768) + 1) = 2048 // (3 + 1) = 512 (§6 frozen C_KV)."""
    assert bp.c_kv(248, 528, 32768) == 3
    assert ap.long_prefill_threshold(num_gpu_blocks=248, block_size=528) == 512


def test_k3_rule_186_blocks_pass_170_fail():
    """024 H-C4: C_KV(2K) >= 25, C_KV(8K) >= 8, >= 128 blocks (019), >= 67 blocks (one 32K request)."""
    assert (bp.c_kv(186, 528, 2048), bp.c_kv(186, 528, 8192)) == (26, 9)
    assert ap.k3_problems(186, block_size=528, c=ap.C_KNEE) == []
    p170 = ap.k3_problems(170, block_size=528, c=ap.C_KNEE)
    assert len(p170) == 1 and "2048" in p170[0] and "24" in p170[0]
    assert any("128" in p for p in ap.k3_problems(127, block_size=528, c={2048: 1, 8192: 1}))
    assert any("32768" in p for p in ap.k3_problems(66, block_size=528, c={2048: 1, 8192: 1}))
    assert ap.k3_problems(176, block_size=528, c=ap.C_KNEE) == [] and ap.k3_problems(175, block_size=528, c=ap.C_KNEE)


def test_block_bounds_bracket_the_pool_size_of_any_consistent_block_bytes():
    """vLLM 0.30.0: num_blocks = K // bytes_per_block (kv_cache_utils.py:1695 / 1750). 248 blocks at 4 GiB bracket
    bytes_per_block s in (4 GiB / 249, 4 GiB / 248], so floor(K 248 / 4 GiB) <= blocks(K) <= ceil(K 249 / 4 GiB) - 1."""
    assert ap.blocks_bounds(3 * GIB, ref_bytes=4 * GIB, ref_blocks=248) == (186, 186)
    assert ap.blocks_bounds(11 * STEP, ref_bytes=4 * GIB, ref_blocks=248) == (170, 171)  # 2.75 GiB
    # both ends fail the rule: 2.75 GiB is out whatever the exact block size is
    assert all(ap.k3_problems(n, block_size=528, c=ap.C_KNEE) for n in (170, 171))
    s_min, s_max = 4 * GIB // 249 + 1, 4 * GIB // 248
    for s in (s_min, (s_min + s_max) // 2, s_max):
        assert 4 * GIB // s == 248
        for m in range(1, 17):
            lo, hi = ap.blocks_bounds(m * STEP, ref_bytes=4 * GIB, ref_blocks=248)
            assert lo <= m * STEP // s <= hi, (s, m)


def test_kv_low_bytes_rule_derives_3_gib():
    assert ap.kv_low_bytes_rule(ref_bytes=4 * GIB, ref_blocks=248, block_size=528, c=ap.C_KNEE) == 3221225472
    # a reference whose bracket straddles the rule at 3 GiB (175 / 176 blocks) cannot decide: measure, do not guess
    assert ap.blocks_bounds(12 * STEP, ref_bytes=4 * GIB, ref_blocks=234) == (175, 176)
    with pytest.raises(ap.PlanError):
        ap.kv_low_bytes_rule(ref_bytes=4 * GIB, ref_blocks=234, block_size=528, c=ap.C_KNEE)
    with pytest.raises(ap.PlanError):  # no multiple below K4 fits the rule
        ap.kv_low_bytes_rule(ref_bytes=4 * GIB, ref_blocks=170, block_size=528, c=ap.C_KNEE)


# ---------------------------------------------------------------------- pins --

def test_repo_pins_advanced_follow_the_024_rules(pins):
    adv = pins["advanced"]
    assert ap.pins_problems(pins) == []
    assert isinstance(pins["freeze"]["advanced"], bool)
    assert (adv["seed"], adv["derived_from"]) == (20261008, "formal-20261009-coreb-uniform")
    assert adv["c"] == {"2048": 25, "8192": 8}
    assert (adv["long_prefill_token_threshold"], adv["kv_low_bytes"]) == (512, 3221225472)
    assert adv["num_gpu_blocks_k4"] == pins["stage_b"]["num_gpu_blocks"] == 248
    assert isinstance(adv["num_gpu_blocks_k3"], int) and adv["num_gpu_blocks_k3"] >= 0
    assert (adv["otlp_endpoint"], adv["otel_protocol"]) == ("http://127.0.0.1:4318/v1/traces", "http/protobuf")
    assert adv["timeouts_s"] == TIMEOUTS
    # 024 Harness: [advanced] carries the conditions table, each condition's argv delta as condition_flags appends it
    assert adv["conditions"] == CONDITIONS_TABLE == {c: ap.condition_flags(c, pins) for c in ap.CONDITIONS}
    assert list(adv["conditions"]) == list(ap.CONDITIONS)


def test_the_024_endpoint_protocol_and_timeouts_are_fixed_in_code():
    """pins_problems compares pins [advanced] with these 024 values, not with "any positive / any local" value."""
    assert ap.OTLP_ENDPOINT == "http://127.0.0.1:4318/v1/traces"
    assert ap.OTEL_PROTOCOL == "http/protobuf"
    assert ap.TIMEOUTS_S == TIMEOUTS


def edited(pins, freeze=None, **adv):
    p = copy.deepcopy(pins)
    p["advanced"].update(adv)
    if freeze is not None:
        p["freeze"]["advanced"] = freeze
    return p


@pytest.mark.parametrize("change", [
    {"seed": 1}, {"derived_from": "formal-20261010-coreb-mixed2"}, {"c": {"2048": 24, "8192": 8}},
    {"c": {"2048": 25}}, {"kv_low_bytes": 11 * STEP},
    {"kv_low_bytes": 3 * GIB + 1}, {"num_gpu_blocks_k4": 247}, {"num_gpu_blocks_k3": -1},
    {"num_gpu_blocks_k3": 170}, {"otel_protocol": "grpc"}, {"otel_protocol": "http/json"},
    {"otlp_endpoint": "http://example.com:4318/v1/traces"},
    # 024 fixes the receiver at :4318/v1/traces, not any local URL
    {"otlp_endpoint": "http://127.0.0.1:4317/v1/traces"}, {"otlp_endpoint": "http://127.0.0.1:4318/v1/logs"},
    {"otlp_endpoint": "http://127.0.0.1:4318/v1/traces/"}, {"otlp_endpoint": "http://localhost:4318/v1/traces"},
    {"timeouts_s": {k: v for k, v in TIMEOUTS.items() if k != "mixed_8192"}},
    {"timeouts_s": {**TIMEOUTS, "mixed_8192": 0}},
    # 024 fixes the timeouts at 300 / 600 / 720 / 960 s, not any positive int
    {"timeouts_s": {**TIMEOUTS, "mixed_8192": 1200}}, {"timeouts_s": {**TIMEOUTS, "uniform_2048": 301}},
    {"timeouts_s": {**TIMEOUTS, "mixed_2048": 600.0}}, {"timeouts_s": {**TIMEOUTS, "uniform_2048_extra": 300}},
    # the conditions table must be exactly condition_flags' argv deltas
    {"conditions": {k: v for k, v in CONDITIONS_TABLE.items() if k != "K3"}},
    {"conditions": {**CONDITIONS_TABLE, "K2": []}},
    {"conditions": {**CONDITIONS_TABLE, "T": ["--long-prefill-token-threshold", "1024"]}},
    {"conditions": {**CONDITIONS_TABLE, "TG": G_FLAGS + T_FLAGS}},
    {"conditions": {**CONDITIONS_TABLE, "G": ["--scheduler-reserve-full-isl"]}},
    {"conditions": {**CONDITIONS_TABLE, "K3": ["--kv-cache-memory-bytes", "3221225472"]}},
    {"conditions": {**CONDITIONS_TABLE, "T": "--long-prefill-token-threshold 512"}},
    {"conditions": {}}])
def test_pins_problems_catch_each_deviation_from_024(pins, change):
    p = edited(pins, **change)
    assert len(ap.pins_problems(p)) == 1, ap.pins_problems(p)
    with pytest.raises(ap.PlanError):
        build(p)


def test_the_conditions_table_cannot_drift_from_condition_flags(pins):
    """024: [advanced].conditions holds each condition's argv delta; it must equal what condition_flags appends."""
    # T changed alone: the T rule fails, and the table no longer matches the flags the code would append
    probs = ap.pins_problems(edited(pins, long_prefill_token_threshold=1024))
    assert len(probs) == 2, probs
    assert sum(x.startswith("long_prefill_token_threshold") for x in probs) == 1
    assert sum(x.startswith("conditions") for x in probs) == 1
    # T and the table changed together: only the T rule fails
    t1024 = ["--long-prefill-token-threshold", "1024"]
    p = edited(pins, long_prefill_token_threshold=1024, conditions={**CONDITIONS_TABLE, "T": t1024, "TG": t1024 + G_FLAGS})
    probs = ap.pins_problems(p)
    assert len(probs) == 1 and probs[0].startswith("long_prefill_token_threshold"), probs
    with pytest.raises(ap.PlanError):
        build(p)
    # no table, or no threshold to derive it from: refused, never a KeyError
    for key in ("conditions", "long_prefill_token_threshold"):
        p = copy.deepcopy(pins)
        del p["advanced"][key]
        probs = ap.pins_problems(p)
        assert any(x.startswith("conditions") for x in probs), (key, probs)
        with pytest.raises(ap.PlanError):
            build(p)


def test_pins_problems_name_the_024_endpoint_and_timeouts(pins):
    (endpoint,) = ap.pins_problems(edited(pins, otlp_endpoint="http://127.0.0.1:4317/v1/traces"))
    assert "http://127.0.0.1:4318/v1/traces" in endpoint and "024" in endpoint
    (timeouts,) = ap.pins_problems(edited(pins, timeouts_s={**TIMEOUTS, "mixed_8192": 1200}))
    assert "960" in timeouts and "024" in timeouts


def test_pins_without_an_advanced_table_are_refused(pins):
    p = copy.deepcopy(pins)
    del p["advanced"]
    assert len(ap.pins_problems(p)) == 1
    with pytest.raises(ap.PlanError):
        build(p)


def test_freeze_needs_the_measured_k3_block_count(pins):
    assert len(ap.pins_problems(edited(pins, freeze=True, num_gpu_blocks_k3=0))) == 1
    assert ap.pins_problems(edited(pins, freeze=True, num_gpu_blocks_k3=186)) == []
    assert ap.pins_problems(edited(pins, freeze=False, num_gpu_blocks_k3=186)) == []


def test_measured_k3_must_be_the_smallest_256_mib_multiple_that_passes(pins):
    # FS1 rule: step up by 256 MiB when the measured pool fails; the pinned K3 is then the smallest passing multiple
    assert ap.pins_problems(edited(pins, kv_low_bytes=13 * STEP, num_gpu_blocks_k3=177)) == []  # 3 GiB had failed
    assert len(ap.pins_problems(edited(pins, kv_low_bytes=13 * STEP, num_gpu_blocks_k3=201))) == 1  # 3 GiB would pass
    assert len(ap.pins_problems(edited(pins, kv_low_bytes=4 * GIB, num_gpu_blocks_k3=248))) == 1  # not below K4


# ---------------------------------------------------------------------- plan --

def test_plan_has_60_boots_in_3_blocks_each_a_permutation_of_the_20_cells(pins):
    boots = build(pins)["boots"]
    assert len(boots) == 60
    assert [b["block"] for b in boots] == [1] * 20 + [2] * 20 + [3] * 20
    for r in (1, 2, 3):
        cells = [(b["cond"], b["L"], b["mix"]) for b in boots if b["block"] == r]
        assert sorted(cells) == sorted(CELLS) and len(set(cells)) == 20
    assert all(b["restart"] == b["block"] for b in boots)
    per_cell = {}
    for b in boots:
        per_cell.setdefault((b["cond"], b["L"], b["mix"]), []).append(b["restart"])
    assert len(per_cell) == 20 and all(v == [1, 2, 3] for v in per_cell.values())


def test_block_order_is_the_seeded_sha256_shuffle_and_reproducible(pins):
    boots = build(pins)["boots"]
    orders = []
    for r in (1, 2, 3):
        want = list(CELLS)
        random.Random(int(hashlib.sha256(f"20261008:advanced:block:{r}".encode()).hexdigest(), 16)).shuffle(want)
        got = [(b["cond"], b["L"], b["mix"]) for b in boots if b["block"] == r]
        assert got == want, r
        assert ap.block_order(20261008, r) == want
        orders.append(tuple(got))
    assert len(set(orders)) == 3
    assert build(pins) == build(pins)


def test_boots_use_core_n_warmup_and_the_pinned_timeouts(pins):
    for b in build(pins)["boots"]:
        C = {2048: 25, 8192: 8}[b["L"]]
        assert set(b) == {"block", "cond", "L", "mix", "restart", "C", "N", "warmup", "timeout_s", "prefix"}
        assert (b["C"], b["N"], b["warmup"]) == (C, bp.num_prompts(C), bp.num_warmups(C))
        assert (b["N"], b["warmup"]) == {2048: (250, 25), 8192: (200, 8)}[b["L"]]
        assert b["timeout_s"] == TIMEOUTS[f"{b['mix']}_{b['L']}"]
        assert b["prefix"] == f"adv-{b['cond']}-{b['L']}-{b['mix']}-r{b['restart']}-"


def test_plan_header_records_c_conditions_blocks_timeouts_and_pins_sha(pins):
    import run_core_b

    plan = build(pins)
    assert (plan["schema_version"], plan["seed"], plan["derived_from"]) == (1, 20261008, "formal-20261009-coreb-uniform")
    assert plan["c"] == {"2048": 25, "8192": 8}
    assert plan["conditions"] == ["B0", "T", "G", "TG", "K3"]
    k3 = pins["advanced"]["num_gpu_blocks_k3"]
    assert plan["expected_blocks"] == {"B0": 248, "T": 248, "G": 248, "TG": 248, "K3": k3}
    assert plan["timeouts_s"] == TIMEOUTS and plan["pins_sha256"] == "ab" * 32
    assert json.loads(json.dumps(plan)) == plan
    # the Stage B server values ride along so the runner can reuse run_core_b.stage_b_problems unchanged
    assert (plan["gpu_memory_utilization"], plan["kv_cache_memory_bytes"], plan["block_size"]) == ("0.91", 4 * GIB, 528)
    assert run_core_b.stage_b_problems(plan, pins, gdn_prefill_backend="triton", gpu_memory_utilization="0.91") == []
    assert build(edited(pins, num_gpu_blocks_k3=186))["expected_blocks"]["K3"] == 186


@pytest.mark.parametrize("meta", [{"formal": False}, {"env_valid": False}, {"env_valid": None}, {"formal": "true"}])
def test_plan_refuses_a_non_formal_or_env_invalid_session(pins, meta):
    with pytest.raises(ap.PlanError, match="formal"):
        build(pins, **meta)


def test_plan_refuses_a_session_without_env_valid(pins):
    session, up, rows = uniform_session(pins)
    del session["env_valid"]
    with pytest.raises(ap.PlanError):
        ap.build_plan(pins, session=session, uniform_plan=up, rows=rows, pins_sha256="ab" * 32)


def test_plan_refuses_another_uniform_session(pins):
    with pytest.raises(ap.PlanError, match="derived_from"):
        build(pins, session_id="formal-20261010-coreb-mixed2")


@pytest.mark.parametrize("kw", [
    {"knee": {2048: 18, 8192: 8}}, {"knee": {2048: 25, 8192: 9}}, {"knee": {2048: 32, 8192: 7}},
    {"drop": lambda x: (x["L"], x["C"], x["restart"]) == (2048, 25, 3)},   # a missing restart: no C_knee (013)
    {"drop": lambda x: x["L"] == 8192}])
def test_plan_refuses_when_the_recomputed_knee_is_not_25_and_8(pins, kw):
    with pytest.raises(ap.PlanError, match="C_knee"):
        build(pins, **kw)


def test_plan_ignores_invalid_uniform_runs_as_the_core_knee_does(pins):
    session, up, rows = uniform_session(pins)
    rows.append({"L": 2048, "mix": "uniform", "restart": 1, "C": 16, "ok": False, "output_throughput": 1e9})
    rows.append({"L": 2048, "mix": "mixed", "restart": 1, "C": 4, "ok": True, "output_throughput": 1e9})
    plan = ap.build_plan(pins, session=session, uniform_plan=up, rows=rows, pins_sha256="ab" * 32)
    assert plan["c"] == {"2048": 25, "8192": 8}


# ----------------------------------------------------------------------- CLI --

def write_session(path, session, up, rows):
    path.mkdir()
    (path / "session.json").write_text(json.dumps(session))
    (path / "plan.json").write_text(json.dumps(up))
    (path / "rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


def test_cli_writes_the_plan_with_the_sha256_of_the_pins_bytes(tmp_path, pins, pins_bytes, monkeypatch):
    session, up, rows = uniform_session(pins)
    write_session(tmp_path / "u", session, up, rows)
    edited_bytes = pins_bytes.replace(b"\nnum_gpu_blocks_k3 = 0 ", b"\nnum_gpu_blocks_k3 = 186 ", 1)
    assert pins_bytes.count(b"\nnum_gpu_blocks_k3 = 0 ") == 1 and edited_bytes != pins_bytes
    for raw in (pins_bytes, edited_bytes):
        monkeypatch.setattr(ap, "load_pins_bytes", lambda raw=raw: raw)
        out = tmp_path / f"plan-{hashlib.sha256(raw).hexdigest()[:8]}.json"
        assert ap.main(["--uniform-session", str(tmp_path / "u"), "--out", str(out)]) == 0
        plan = json.loads(out.read_text())
        sha = hashlib.sha256(raw).hexdigest()
        assert plan["pins_sha256"] == sha
        assert plan == json.loads(json.dumps(ap.build_plan(tomllib.loads(raw.decode()), session=session,
                                                           uniform_plan=up, rows=rows, pins_sha256=sha)))
    assert plan["expected_blocks"]["K3"] == 186


@pytest.mark.parametrize("bad", [{"formal": False}, {"env_valid": False}, {"session_id": "formal-x"}])
def test_cli_refuses_and_writes_nothing(tmp_path, pins, pins_bytes, monkeypatch, bad):
    session, up, rows = uniform_session(pins, **bad)
    write_session(tmp_path / "u", session, up, rows)
    monkeypatch.setattr(ap, "load_pins_bytes", lambda: pins_bytes)
    with pytest.raises(SystemExit):
        ap.main(["--uniform-session", str(tmp_path / "u"), "--out", str(tmp_path / "plan.json")])
    assert not (tmp_path / "plan.json").exists()


def test_cli_refuses_a_knee_mismatch(tmp_path, pins, pins_bytes, monkeypatch):
    session, up, rows = uniform_session(pins, knee={2048: 18, 8192: 8})
    write_session(tmp_path / "u", session, up, rows)
    monkeypatch.setattr(ap, "load_pins_bytes", lambda: pins_bytes)
    with pytest.raises(SystemExit):
        ap.main(["--uniform-session", str(tmp_path / "u"), "--out", str(tmp_path / "plan.json")])
    assert not (tmp_path / "plan.json").exists()
