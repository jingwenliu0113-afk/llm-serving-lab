"""Tests for the quality pilot (decision 005 'Pilot'): the extra pilot jobs in
run_quality.build_quality_plan and the masked report in quality/quality_pilot.py,
which prints throughput, determinism, sanity values, SEs and discordance (b + c)
but never an accuracy, a KLD of Q6_K / Q4_K_M or a sign of the H-A5 contrasts."""

import copy
import tomllib
from pathlib import Path

import pytest

import quality_pilot as qpl
import run_quality as rq

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_pilot_adds_fa_off_and_repeat_jobs_before_the_kld_file_is_deleted(pins):
    jobs = rq.build_quality_plan(pins, mode="pilot", n_thr=8, poll=0)["jobs"]
    for size in [m["size"] for m in pins["core_a"]["models"]]:
        ids = [j["id"] for j in jobs if j["size"] == size]
        fa_off = next(j for j in jobs if j["id"] == f"kld-{size}-Q8_0-fa_off")
        assert arg(fa_off["argv"], "-fa") == "off" and fa_off["variant"] == "fa_off"
        rep = next(j for j in jobs if j["id"] == f"kld-{size}-Q8_0-repeat")
        orig = next(j for j in jobs if j["id"] == f"kld-{size}-Q8_0")
        assert rep["argv"] == orig["argv"] and rep["repeat_of"] == orig["id"]
        delete = next(j for j in jobs if j["id"] == f"delete_kld-{size}")
        assert ids.index(fa_off["id"]) < ids.index(delete["id"]) and fa_off["id"] in delete["after"]
        mc_rep = next(j for j in jobs if j["id"] == f"mc-mmlu-{size}-Q8_0-repeat")
        assert mc_rep["repeat_of"] == f"mc-mmlu-{size}-Q8_0"


def test_formal_plan_has_no_pilot_extras(pins):
    frozen = copy.deepcopy(pins)
    frozen["freeze"]["mac"] = True
    jobs = rq.build_quality_plan(frozen, mode="formal", n_thr=8, poll=0)["jobs"]
    assert not any(j.get("variant") or j.get("repeat_of") for j in jobs)


# ------------------------------------------------------------------ report --

def kld_row(jid, point, mean, se, size="4B", wall=100.0, **kw):
    return {"job_id": jid, "kind": "kld", "size": size, "point": point, "outcome": "ok", "wall_s": wall,
            "metrics": {"kld_mean": mean, "kld_mean_se": se, "n_chunks": 20}, **kw}


def mc_row(jid, point, correct, task="mmlu", size="4B", wall=50.0, **kw):
    acc = 100.0 * sum(correct) / len(correct)
    return {"job_id": jid, "kind": "mc", "size": size, "point": point, "task": task, "outcome": "ok",
            "wall_s": wall, "metrics": {"n_tasks": len(correct), "accuracy_pct": acc, "correct": correct}, **kw}


def ppl_row(jid, size, point, nll):
    return {"job_id": jid, "kind": "ppl", "size": size, "point": point, "outcome": "ok", "wall_s": 30.0,
            "metrics": {"n_chunks": len(nll), "ppl": 9.87654, "chunk_nll": nll}}


ROWS = [
    {"job_id": "kld_base-4B", "kind": "kld_base", "size": "4B", "point": "BF16", "outcome": "ok", "wall_s": 80.0,
     "metrics": {"n_chunks": 20, "ppl": 8.123}},
    {"job_id": "kld_self-4B", "kind": "kld_self", "size": "4B", "point": "BF16", "outcome": "ok", "wall_s": 70.0,
     "metrics": {"kld_mean": 0.000001, "kld_mean_se": 0.0000001, "n_chunks": 20}},
    kld_row("kld-4B-Q8_0", "Q8_0", 0.00123, 0.00005),
    kld_row("kld-4B-Q6_K", "Q6_K", 0.00456, 0.00011),
    kld_row("kld-4B-Q4_K_M", "Q4_K_M", 0.01789, 0.00031),
    kld_row("kld-4B-Q8_0-fa_off", "Q8_0", 0.00124, 0.00005, variant="fa_off"),
    kld_row("kld-4B-Q8_0-repeat", "Q8_0", 0.00123, 0.00005, repeat_of="kld-4B-Q8_0"),
    mc_row("mc-mmlu-4B-BF16", "BF16", [1, 1, 0, 1, 0, 1, 1, 0]),
    mc_row("mc-mmlu-4B-Q8_0", "Q8_0", [1, 1, 0, 1, 0, 1, 0, 0]),
    mc_row("mc-mmlu-4B-Q6_K", "Q6_K", [1, 0, 0, 1, 1, 1, 0, 0]),
    mc_row("mc-mmlu-4B-Q4_K_M", "Q4_K_M", [0, 0, 0, 1, 1, 1, 0, 1]),
    mc_row("mc-mmlu-4B-Q8_0-repeat", "Q8_0", [1, 1, 0, 1, 0, 1, 0, 0], repeat_of="mc-mmlu-4B-Q8_0"),
    ppl_row("ppl-9B-Q4_K_M", "9B", "Q4_K_M", [2.0, 2.1, 1.9, 2.2]),
    ppl_row("ppl-4B-Q8_0", "4B", "Q8_0", [2.1, 2.0, 2.0, 2.4]),
]


def test_determinism_compares_repeats_with_their_originals():
    det = qpl.determinism(ROWS)
    assert det == [{"job": "kld-4B-Q8_0-repeat", "of": "kld-4B-Q8_0", "identical": True},
                   {"job": "mc-mmlu-4B-Q8_0-repeat", "of": "mc-mmlu-4B-Q8_0", "identical": True}]
    changed = [dict(r) for r in ROWS]
    changed[6] = kld_row("kld-4B-Q8_0-repeat", "Q8_0", 0.00125, 0.00005, repeat_of="kld-4B-Q8_0")
    assert qpl.determinism(changed)[0]["identical"] is False


def test_discordance_counts_without_accuracies():
    d = qpl.discordance(ROWS, n_formal={"mmlu": 3997})
    pair = next(x for x in d if x["task"] == "mmlu" and x["pair"] == ["Q8_0", "Q6_K"])
    assert pair["n"] == 8 and pair["b_plus_c"] == 2
    assert pair["ci_halfwidth_pp_formal"] == pytest.approx(196 * (2 / 8 / 3997) ** 0.5)
    assert set(pair) == {"task", "size", "pair", "n", "b_plus_c", "rate", "ci_halfwidth_pp_formal"}


def test_report_is_masked():
    rep = qpl.report(ROWS, pins_expected={"wikitext_chunks": 580, "mmlu_tasks": 3997, "hellaswag_tasks": 4000,
                                          "arc_tasks": 869, "winogrande_tasks": 1267}, pilot={"chunks": 20, "tasks": 200})
    text = repr(rep)
    for hidden in ("0.00456", "0.01789", "62.5", "50.0", "37.5", "9.87654"):  # Q6_K/Q4_K_M KLD, accuracies, PPL
        assert hidden not in text, hidden
    assert rep["sanity"]["kld_self"]["4B"] == 0.000001
    assert rep["sanity"]["q8_0_kld"]["4B"] == {"fa_on": 0.00123, "fa_off": 0.00124}
    # only the largest SE per size (power of the z tests): per-quant SEs track the KLD means and
    # would reveal the H-A5(b) ordering (013)
    assert rep["kld_se_formal_est"] == {"4B": {"max": pytest.approx(0.00031 * (20 / 580) ** 0.5)}}
    for hidden in ("0.00011", "0.00005"):
        assert hidden not in text, hidden
    ha5 = rep["ha5_paired_nll"]
    assert set(ha5) == {"n_chunks", "sd_chunk_diff", "ci_halfwidth_formal"}
    assert rep["throughput"]["formal_hours_est"] > 0
