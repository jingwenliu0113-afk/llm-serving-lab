"""Tests for benchmarks/core_a_pilot.py: the Core A pilot of PREREGISTRATION §5
(-fa auto probe, --poll rule, memory cross-check, noise / MDE, duration calibration),
executed by run_core_a.py and summarised without exposing any hypothesis effect."""

import tomllib
from pathlib import Path

import pytest

import core_a_pilot as pl
import core_a_plan as cp

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def build(pins, sizes=("4B",), platform="macos-metal"):
    return pl.build_pilot_plan(pins, platform=platform, sizes=list(sizes), n_thr=8)


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_pilot_plan_is_a_runnable_pilot_plan(pins):
    plan = build(pins)
    assert plan["mode"] == "pilot" and plan["schema_version"] == cp.SCHEMA_VERSION
    assert plan["params"]["poll"] == 0
    assert len({j["id"] for j in plan["jobs"]}) == len(plan["jobs"])
    assert {j["kind"] for j in plan["jobs"]} <= {"fit", "ttft", "decode", "sentinel"}
    assert plan["jobs"][0]["position"] == "start" and plan["jobs"][-1]["position"] == "end"


def test_fa_probe_uses_auto_on_every_model_at_shortest_context(pins):
    jobs = [j for j in build(pins, sizes=("4B", "9B"))["jobs"] if j.get("block") == "fa_probe"]
    assert len(jobs) == 6
    for j in jobs:
        assert j["kind"] == "fit" and j["L"] == 2048 and arg(j["argv"], "-fa") == "auto"


def test_poll_block_alternates_and_overrides_expected_poll(pins):
    jobs = [j for j in build(pins)["jobs"] if j.get("block") == "poll"]
    assert [arg(j["argv"], "--poll") for j in jobs] == ["0", "50"] * pl.POLL_REPS
    size, quant, L = cp.sentinel_cell(pins)
    for j in jobs:
        assert (j["kind"], j["size"], j["quant"], j["L"]) == ("decode", size, quant, L)
        assert j["expect"]["poll"] == int(arg(j["argv"], "--poll"))


def test_memory_block_covers_every_quant_and_context_of_the_sizes(pins):
    jobs = [j for j in build(pins)["jobs"] if j.get("block") == "memory"]
    cells = {(j["quant"], j["L"]) for j in jobs}
    assert cells == {(q, L) for q in pins["core_a"]["quants"] for L in pins["core_a"]["context_lengths"]}
    assert all(arg(j["argv"], "-fa") == "on" and j["external_memory"] == "time-l" for j in jobs)


def test_calibration_uses_only_the_heaviest_context(pins):
    jobs = [j for j in build(pins)["jobs"] if j.get("block") == "calib"]
    assert {j["kind"] for j in jobs} == {"ttft", "decode"}
    assert {j["L"] for j in jobs} == {max(pins["core_a"]["context_lengths"])}  # no ratio across L


# ------------------------------------------------------------------ report --


def row(block, kind, avg_ts=None, poll=None, wall=10.0, **kw):
    r = {"block": block, "kind": kind, "outcome": "ok", "wall_s": wall, "size": "4B", "quant": "Q4_K_M",
         "L": 2048, "metrics": {"avg_ts": avg_ts} if avg_ts else None, "summary": None, "external_memory": None,
         "argv": ["x", "--poll", str(poll)] if poll is not None else ["x"]}
    r.update(kw)
    return r


def test_poll_rule_prefers_zero_unless_more_than_3pct_slower():
    rows = [row("poll", "decode", 100.0, 0), row("poll", "decode", 100.0, 50)] * 3
    assert pl.poll_decision(rows)["poll"] == 0
    rows = [row("poll", "decode", 96.0, 0), row("poll", "decode", 100.0, 50)] * 3
    d = pl.poll_decision(rows)
    assert d["poll"] == 50 and d["median_tok_s"] == {"0": 96.0, "50": 100.0}


def test_mde_from_process_level_cv():
    rows = [row("poll", "decode", v, 0) for v in (99.0, 100.0, 101.0)]
    m = pl.noise_and_mde(rows, poll=0)
    assert m["n"] == 3 and m["cv_pct"] == pytest.approx(1.0)
    # "beyond noise" = |x - y| > 2 * sqrt(SE_x^2 + SE_y^2), SE = s / sqrt(3), equal CV
    assert m["mde_pct"] == pytest.approx(2 * (2 / 3) ** 0.5 * 1.0)


def test_memory_cross_check_rows():
    summary = {"memory": {"unit": "MiB", "devices": [{"name": "MTL0", "self": 3000}], "host": {"self": 300},
                          "other": []}, "graph": {"nodes": "1800", "splits": "2"}, "fused_ops": []}
    r = row("memory", "fit", summary=summary,
            external_memory={"peak_footprint_bytes": 3500 * 2**20, "max_rss_bytes": 400 * 2**20})
    out = pl.memory_cross_check([r])
    assert out == [{"size": "4B", "quant": "Q4_K_M", "L": 2048, "outcome": "ok", "device_self_mib": 3000,
                    "host_self_mib": 300, "footprint_mib": 3500.0, "max_rss_mib": 400.0,
                    "footprint_over_self": pytest.approx(3500 / 3300), "graph_splits": "2"}]


def test_report_never_lists_cross_cell_speeds():
    rows = ([row("poll", "decode", 100.0, 0), row("poll", "decode", 100.0, 50)] * 3
            + [row("calib", "ttft", 900.0, L=32768, wall=60.0), row("calib", "decode", 40.0, L=32768, wall=30.0),
               row("memory", "fit", wall=20.0)])
    rep = pl.report(rows, n_formal_jobs={"fit": 54, "ttft": 54, "decode": 54, "sentinel": 2}, idle_s=30)
    text = repr(rep)
    assert "900" not in text and "40.0" not in text  # calibration speeds stay masked
    # 54 x (20 + 30) + 54 x (60 + 30) + 54 x (30 + 30) + 2 x (30 + 30) seconds
    assert rep["calibration"]["upper_bound_hours"] == round((54 * 50 + 54 * 90 + 54 * 60 + 2 * 60) / 3600, 2)


def test_graph_splits_collected_from_bench_and_fit_rows():
    rows = [row("poll", "decode", 100.0, 0, summary={"graph": {"nodes": "1879", "splits": "2"}}),
            row("memory", "fit", summary={"graph": {"nodes": "1900", "splits": "2"}, "memory": {}}),
            row("calib", "ttft", 900.0, L=32768, summary={"graph": {"nodes": "1879", "splits": "50"}})]
    g = pl.graph_splits(rows)
    assert g["values"] == {"2": 2, "50": 1} and g["max_splits"] == 50


def test_blocks_can_be_selected(pins):
    plan = pl.build_pilot_plan(pins, platform="macos-metal", sizes=["4B"], n_thr=8, blocks=["fa_probe"])
    assert {j["block"] for j in plan["jobs"]} == {"fa_probe", "sentinel"}
    assert all(arg(j["argv"], "-lv") == "4" for j in plan["jobs"] if j["kind"] == "fit")
