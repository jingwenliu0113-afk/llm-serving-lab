"""Tests for benchmarks/stage_e2_plan.py (decision 041 v3 §2): conditions built from the frozen Advanced / Core B
builders, the manipulation checks, the clients, timeouts, the cell plan and the freeze."""

import json
import tomllib
from pathlib import Path

import pytest

import adv_plan as ap
import capacity_plan as cp
import stage_e2_plan as e2
import test_run_stage_e2 as tre2

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
REPO_CFG = tomllib.loads((ROOT / "configs" / "stage_e2.toml").read_text())   # frozen 2026-10-31 (041 §2.10)
CFG = {**REPO_CFG, "freeze": {"e2": False, "smoke_e2": ""}}                     # an explicit DRAFT config
SB = PINS["stage_b"]
KW = dict(gdn_prefill_backend=SB["gdn_prefill_backend"], gpu_memory_utilization=SB["gpu_memory_utilization"],
          vllm_bin="/v", base_env={"PATH": "/usr/bin", "HOME": "/h"})
B0_LOG = ROOT / "results" / "capacity" / "c0-step0" / "raw" / "formal-20261011-adv" / "B0-8192-uniform-r1.server.log"


def frozen(smoke="smoke-e2"):
    e = {**CFG["e2"], "shared_prefix": {**CFG["e2"]["shared_prefix"], "sha256": "s" * 64, "common_prefix": 1630}}
    return {**CFG, "freeze": {"e2": True, "smoke_e2": smoke}, "e2": e}


def test_the_repo_config_is_frozen_on_the_passed_smoke_of_this_harness():
    """041 v3 §2.10 (2026-10-31): the repo config names the passed smoke smoke-e2-20261031 (12 / 12 boots, every check
    true), which ran this harness and this config with the pinned datasets; any edit after the freeze fails here."""
    assert e2.config_problems(REPO_CFG, PINS) == [] and REPO_CFG["freeze"] == {"e2": True, "smoke_e2": "smoke-e2-20261031"}
    assert e2.harness_sha256() == "a7a5f9d377173c3535d225131ad26d700c80acc07ca5416ad24d6fd3eb238c3b"
    assert e2.smoke_problems(ROOT / "results" / "stage_e" / "smoke-e2-20261031", PINS, REPO_CFG) == []
    plan = e2.build_plan(PINS, REPO_CFG, mode="formal")
    assert len(plan["boots"]) == 48 and plan["cap_s"] == 7.5 * 3600


def test_the_draft_config_is_well_formed_and_unfrozen():
    assert e2.config_problems(CFG, PINS) == [] and CFG["freeze"] == {"e2": False, "smoke_e2": ""}
    assert CFG["e2"]["expected_blocks"] == {"B0": 248, "T": 248, "T256": 248, "K25": 155, "TK25": 155, "PC": 248}
    assert e2.config_problems(frozen(), PINS) == []
    for pc in (200, 248.0, True):
        bad = {**CFG, "e2": {**CFG["e2"], "expected_blocks": {**CFG["e2"]["expected_blocks"], "PC": pc}}}
        assert any("expected_blocks" in p for p in e2.config_problems(bad, PINS)), pc
    assert e2.config_problems({**CFG, "e2": {**CFG["e2"], "kv25_bytes": 2 * 2 ** 30}}, PINS)
    blank = {**CFG["e2"]["shared_prefix"], "sha256": "", "common_prefix": 0}
    unpinned = {**frozen(), "e2": {**frozen()["e2"], "shared_prefix": blank}}
    assert any("shared_prefix" in p for p in e2.config_problems(unpinned, PINS))
    with pytest.raises(e2.PlanError, match="pinned first"):
        e2.smoke_plan(PINS, {**CFG, "e2": {**CFG["e2"], "shared_prefix": blank}})   # the dataset comes before the smoke


def test_the_repo_config_pins_the_shared_prefix_dataset_made_on_wsl():
    """041 v3 §2.8 / §2.10: stage_e2_datasets.py on WSL (2026-10-31, deploy 88936bb) wrote 275 rows of 1,920 tokens whose
    common token-id prefix is 1,599 (>= 1,584 = 3 x 528); its sha256 and the measured prefix are pinned before the smoke."""
    sp = CFG["e2"]["shared_prefix"]
    assert sp["sha256"] == "ca148974b2b0045579e9573f9000fd5cb33ab9fa95c78d17961d00fa472e75f7"
    assert sp["common_prefix"] == 1599 >= sp["min_shared"] == 1584 and sp["rows"] == 275 and sp["total"] == 1920
    assert e2.smoke_plan(PINS, CFG)["datasets_sha256"]["shared:2048"] == sp["sha256"]


def test_a_plan_pins_the_configs_content_without_freeze():
    a, b = frozen(), {**frozen(), "freeze": {"e2": False, "smoke_e2": ""}}
    assert e2.config_sha256(a) == e2.config_sha256(b) != e2.config_sha256({**a, "e2": {**a["e2"], "t256": 128}})
    assert e2.smoke_plan(PINS, b)["stage_e2_sha256"] == e2.config_sha256(a)


def test_b0_and_t_are_exactly_advanceds_servers():
    for cond in ("B0", "T"):
        assert e2.server_argv_for(cond, PINS, CFG, port=8000, **KW) == ap.server_argv_for(cond, PINS, port=8000, **KW)


def test_the_new_conditions_change_only_their_knob():
    b0, _ = e2.server_argv_for("B0", PINS, CFG, **KW)
    t256, _ = e2.server_argv_for("T256", PINS, CFG, **KW)
    assert t256 == b0 + ["--long-prefill-token-threshold", "256"]
    k25, _ = e2.server_argv_for("K25", PINS, CFG, **KW)
    assert k25[k25.index("--kv-cache-memory-bytes") + 1] == "2684354560"
    assert [a for a in k25 if a != "2684354560"] == [a for a in b0 if a != str(SB["kv_cache_memory_bytes"])]
    tk25, _ = e2.server_argv_for("TK25", PINS, CFG, **KW)
    assert tk25 == k25 + ["--long-prefill-token-threshold", "512"]
    pc, env = e2.server_argv_for("PC", PINS, CFG, **KW)
    assert "--no-enable-prefix-caching" not in pc and pc.count("--enable-prefix-caching") == 1 and len(pc) == len(b0)
    with pytest.raises(e2.PlanError):
        e2.server_argv_for("A", PINS, CFG, **KW)


def test_inherited_pytorch_variables_never_reach_a_server():
    _, env = e2.server_argv_for("B0", PINS, CFG, **{**KW, "base_env": {"PATH": "/b", "PYTORCH_CUDA_ALLOC_CONF": "x"}})
    assert not any(k.startswith("PYTORCH_") for k in env)


def test_the_manipulation_check_on_a_real_b0_log_and_its_edits():
    if not B0_LOG.exists():   # raw/ is not in git: present on the lab Mac (fetch_raw.sh), absent in CI
        pytest.skip(f"{B0_LOG.name} (raw/) not present in this checkout")
    log = B0_LOG.read_text()
    facts, got, problems = e2.manipulation_problems(log, "B0", PINS, CFG)
    assert problems == [] and got["enable_prefix_caching"] is False and facts["attention_backend"] == "FLASH_ATTN"
    assert any("long_prefill_token_threshold" in p for p in e2.manipulation_problems(log, "T256", PINS, CFG)[2])
    assert any("kv_cache_memory_bytes" in p for p in e2.manipulation_problems(log, "K25", PINS, CFG)[2])
    pc = log.replace("'enable_prefix_caching': False", "'enable_prefix_caching': True")
    assert e2.manipulation_problems(pc, "PC", PINS, CFG)[2] == []
    assert e2.manipulation_problems(log.replace("'enable_prefix_caching': False, ", ""), "PC", PINS, CFG)[2] == []
    assert any("enable_prefix_caching" in p for p in e2.manipulation_problems(log, "PC", PINS, CFG)[2])
    assert any("enable_prefix_caching" in p for p in e2.manipulation_problems(pc, "B0", PINS, CFG)[2])


def test_the_cache_info_label_decides_prefix_caching():
    m = 'vllm:cache_config_info{block_size="528",enable_prefix_caching="True",num_gpu_blocks="248"} 1.0\n'
    assert e2.prefix_caching_label(m) is True
    assert e2.prefix_caching_label(m.replace('"True"', '"False"')) is False and e2.prefix_caching_label("") is None


def test_the_clients_are_core_bs_with_a_prefix_and_the_shared_dataset():
    ds = {"mixed:2048": "/d/mixed-2048.jsonl", "shared:2048": "/d/shared.jsonl"}
    u = e2.bench_argv_for(PINS, mix="uniform", C=8, prefix="e2-B0-c8-uniform-r1-", result_dir="/r",
                          result_filename="x.json", datasets=ds, vllm_bin="/v")
    assert u[-2:] == ["--request-id-prefix", "e2-B0-c8-uniform-r1-"] and u[u.index("--num-prompts") + 1] == "200"
    m = e2.bench_argv_for(PINS, mix="mixed", C=25, prefix="p", result_dir="/r", result_filename="x.json",
                          datasets=ds, vllm_bin="/v")
    assert m[m.index("--dataset-path") + 1] == "/d/mixed-2048.jsonl" and "--disable-shuffle" in m
    s = e2.bench_argv_for(PINS, mix="shared", C=25, prefix="p", result_dir="/r", result_filename="x.json",
                          datasets=ds, vllm_bin="/v")
    assert s[s.index("--dataset-path") + 1] == "/d/shared.jsonl" and "--random-input-len" not in s
    assert s[s.index("--custom-output-len") + 1] == "128" and s[s.index("--num-prompts") + 1] == "250"
    with pytest.raises(e2.PlanError):
        e2.bench_argv_for(PINS, mix="shared", C=25, prefix="p", result_dir="/r", result_filename="x", datasets={})


def test_timeouts_follow_030_and_double_for_chunked_conditions():
    assert e2.timeout_s("B0", "mixed", 25) == cp.bench_timeout_s(2048, "mixed", 25, cp.TAU1_S)
    assert e2.timeout_s("T256", "uniform", 25) == 2 * cp.bench_timeout_s(2048, "uniform", 25, cp.TAU1_S)
    assert e2.timeout_s("PC", "shared", 25) == cp.bench_timeout_s(2048, "uniform", 25, cp.TAU1_S)


def test_the_formal_plan_has_48_boots_in_three_shuffled_blocks(tmp_path):
    _, _, _, _, p = tre2.formal_setup(tmp_path)
    assert len(p["boots"]) == 48 and [b["block"] for b in p["boots"]] == [1] * 16 + [2] * 16 + [3] * 16
    cells = {(b["cell"], b["mix"]) for b in p["boots"] if b["block"] == 1}
    assert len(cells) == 16 and ("PC-on", "shared") in cells and ("PC-on", "uniform") not in cells
    orders = [[(b["cell"], b["mix"]) for b in p["boots"] if b["block"] == r] for r in (1, 2, 3)]
    assert orders[0] != orders[1] != orders[2]
    b8k = next(b for b in p["boots"] if b["cell"] == "B0@8k" and b["mix"] == "mixed")
    assert (b8k["cond"], b8k["C"], b8k["N"], b8k["warmup"], b8k["dataset"]) == ("K25", 8, 200, 8, "mixed:2048")
    assert len({b["prefix"] for b in p["boots"]}) == 48
    assert p["kv_cache_memory_bytes"]["K25"] == 2684354560 and p["thresholds"] == {"T": 512, "T256": 256, "TK25": 512}
    assert p["cap_s"] == 7.5 * 3600 and p["harness_sha256"] == e2.harness_sha256()


def test_formal_plans_need_the_freeze_and_a_passed_smoke_of_this_design(tmp_path):
    root, ds, pins, cfg, _ = tre2.formal_setup(tmp_path)
    with pytest.raises(e2.PlanError, match="freeze.e2"):
        e2.build_plan(pins, {**cfg, "freeze": {"e2": False, "smoke_e2": ""}}, mode="formal", root=root)
    with pytest.raises(e2.PlanError, match="smoke first"):
        e2.build_plan(pins, {**cfg, "freeze": {"e2": True, "smoke_e2": "none"}}, mode="formal", root=root)
    d = root / "results" / "stage_e" / "smoke-e2"
    s, plan = json.loads((d / "session.json").read_text()), json.loads((d / "plan.json").read_text())
    trimmed = {**plan, "boots": plan["boots"][-2:]}
    sm = s["smoke"]
    for over_s, over_plan, why in (
            ({"harness_sha256": "0" * 64}, None, "another harness"),
            ({"status": "running"}, None, "finished, passed"),
            ({"env_valid": False}, None, "env not valid"),
            ({"host": {"system": "Darwin", "wsl": False, "nvidia": False}}, None, "host"),
            ({"vllm_version": "0.29.0"}, None, "pinned"),
            ({"datasets_sha256": {**s["datasets_sha256"], "shared:2048": "0" * 64}}, None, "datasets"),
            ({"smoke": {**sm, "covered": sm["covered"][:-1]}}, None, "cover"),
            ({"smoke": {**sm, "measured_blocks": {**sm["measured_blocks"], "PC": 200}}}, None, "measured blocks"),
            ({"plan_sha256": e2.sha256_json(trimmed)}, trimmed, "smoke plan"),
            ({}, trimmed, "plan_sha256")):
        (d / "session.json").write_text(json.dumps({**s, **over_s}))
        (d / "plan.json").write_text(json.dumps(over_plan or plan))
        with pytest.raises(e2.PlanError, match=why):
            e2.build_plan(pins, cfg, mode="formal", root=root)
    (d / "session.json").write_text(json.dumps(s))
    (d / "plan.json").write_text(json.dumps(plan))
    assert e2.build_plan(pins, cfg, mode="formal", root=root)["boots"]


def test_the_smoke_plan_has_one_boot_per_condition_and_mix_of_the_design():
    p = e2.smoke_plan(PINS, frozen())
    design = {(e2._cell(c)[1], m) for c, _, _, mixes in e2.CELLS for m in mixes}
    assert [(b["cell"], b["mix"]) for b in p["boots"]] == list(e2.SMOKE) and {b["N"] for b in p["boots"]} == {20}
    assert {(b["cond"], b["mix"]) for b in p["boots"]} == design and len(design) == 12 and p["cap_s"] is None
