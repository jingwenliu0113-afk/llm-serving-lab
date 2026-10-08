"""Tests for benchmarks/core_a_plan.py: the Core A job list must implement
PREREGISTRATION §4 exactly (per-tool flags, one config per process, 3 reps,
seeded random order, sentinel at both ends) and refuse formal mode before the
matching pre-registration stage is frozen (decision 006)."""

import copy
import json
import tomllib
from collections import Counter
from pathlib import Path

import pytest

import core_a_plan as cp

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def plan(pins, **kw):
    args = dict(platform="macos-metal", mode="pilot", seed=7, n_thr=8, poll=0)
    args.update(kw)
    return cp.build_plan(pins, **args)


def jobs_of(p, kind):
    return [j for j in p["jobs"] if j["kind"] == kind]


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_job_counts(pins):
    p = plan(pins)
    counts = Counter(j["kind"] for j in p["jobs"])
    # 2 sizes x 3 quants x 3 contexts = 18 cells; x 3 reps per measurement
    assert counts == {"fit": 54, "ttft": 54, "decode": 54, "sentinel": 2}
    assert len({j["id"] for j in p["jobs"]}) == len(p["jobs"])


def test_every_cell_has_three_reps_of_each_kind(pins):
    p = plan(pins)
    cells = Counter((j["kind"], j["size"], j["quant"], j["L"]) for j in p["jobs"] if j["kind"] != "sentinel")
    assert set(cells.values()) == {3}
    assert len(cells) == 3 * 18


def test_sentinel_first_and_last(pins):
    jobs = plan(pins)["jobs"]
    for j in (jobs[0], jobs[-1]):
        assert j["kind"] == "sentinel"
        assert (j["size"], j["quant"], j["L"]) == cp.sentinel_cell(pins)
    assert [jobs[0]["position"], jobs[-1]["position"]] == ["start", "end"]


def test_sentinel_cell_is_smallest_size_lowest_bit_shortest_context(pins):
    # Derived from pins, not hard-coded, so a model change does not touch the harness.
    p = copy.deepcopy(pins)
    p["core_a"]["models"][0]["size"] = "4B"
    p["core_a"]["quants"] = ["Q8_0", "Q6_K", "Q4_K_M"]
    p["core_a"]["context_lengths"] = [4096, 16384, 65536]
    assert cp.sentinel_cell(p) == ("4B", "Q4_K_M", 4096)


def test_order_is_seeded(pins):
    a = [j["id"] for j in plan(pins, seed=1)["jobs"]]
    b = [j["id"] for j in plan(pins, seed=1)["jobs"]]
    c = [j["id"] for j in plan(pins, seed=2)["jobs"]]
    assert a == b
    assert a != c
    assert sorted(a) == sorted(c)


def test_ttft_argv_matches_prereg(pins):
    j = next(j for j in jobs_of(plan(pins), "ttft") if j["L"] == 8192)
    argv = j["argv"]
    assert argv[0] == "{bin}/llama-bench"
    assert arg(argv, "-m") == "{models}/" + j["model"]
    assert (arg(argv, "-p"), arg(argv, "-n")) == ("8192", "0")
    assert "-d" not in argv
    for flag, value in cp.BENCH_COMMON.items():
        assert arg(argv, flag) == value
    assert (arg(argv, "-t"), arg(argv, "--poll")) == ("8", "0")
    assert "-v" in argv
    # llama-bench rejects these (decision 002):
    for bad in ("-c", "-np", "--fit"):
        assert bad not in argv


def test_decode_argv_kv_equals_level(pins):
    for j in jobs_of(plan(pins), "decode") + jobs_of(plan(pins), "sentinel"):
        argv = j["argv"]
        assert (arg(argv, "-p"), arg(argv, "-n")) == ("0", "128")
        assert int(arg(argv, "-d")) + 128 == j["L"]


def test_fit_argv_matches_prereg(pins):
    j = next(j for j in jobs_of(plan(pins, n_thr=6, poll=50, platform="wsl2-cuda"), "fit") if j["L"] == 32768)
    argv = j["argv"]
    assert argv[0] == "{bin}/llama-completion"
    assert arg(argv, "-c") == "32768"
    assert (arg(argv, "-np"), arg(argv, "--fit"), arg(argv, "-n")) == ("1", "off", "128")
    for flag, value in cp.COMPLETION_COMMON.items():
        assert arg(argv, flag) == value
    assert (arg(argv, "-t"), arg(argv, "--poll")) == ("6", "50")
    assert {"--ignore-eos", "-no-cnv", "--log-jsonl"} <= set(argv)
    assert arg(argv, "-f") == "{prompt}"
    assert j["prompt"] == {"unit": cp.PROMPT_UNIT, "count": 32768 - 128}


def test_models_come_from_pins(pins):
    p = plan(pins)
    expected = {
        (m["size"], q): f["filename"] for m in pins["core_a"]["models"] for q, f in m["files"].items()
    }
    for j in p["jobs"]:
        assert j["model"] == expected[(j["size"], j["quant"])]


def test_plan_records_expectations(pins):
    p = plan(pins)
    assert p["expect"]["build_number"] == pins["llama_cpp"]["build_number"]
    assert p["expect"]["build_commit"] == pins["llama_cpp"]["commit"]
    assert p["params"] == {"n_thr": 8, "poll": 0, "reps": 3, "bench_r": 5, "delay_s": 5, "idle_s": 30}
    assert p["mode"] == "pilot" and p["platform"] == "macos-metal" and p["seed"] == 7
    assert json.loads(json.dumps(p)) == p


def test_formal_mode_requires_matching_freeze(pins):
    with pytest.raises(cp.PlanError, match="freeze.mac"):
        plan(pins, mode="formal")
    with pytest.raises(cp.PlanError, match="freeze.nvidia"):
        plan(pins, mode="formal", platform="wsl2-cuda")
    frozen = copy.deepcopy(pins)
    frozen["freeze"]["mac"] = True
    assert plan(frozen, mode="formal")["mode"] == "formal"


def test_unknown_platform_rejected(pins):
    with pytest.raises(cp.PlanError):
        plan(pins, platform="linux-rocm")


def test_main_writes_plan(tmp_path):
    out = tmp_path / "plan.json"
    rc = cp.main(["--platform", "macos-metal", "--mode", "pilot", "--seed", "3", "--n-thr", "8", "--poll", "0",
                  "--out", str(out)])
    assert rc == 0
    p = json.loads(out.read_text())
    assert p["seed"] == 3 and len(p["jobs"]) == 164
    assert len(p["pins_sha256"]) == 64
