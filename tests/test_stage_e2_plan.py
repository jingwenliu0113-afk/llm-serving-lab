"""Tests for benchmarks/stage_e2_plan.py (decision 041 v3 §2): conditions built from the frozen Advanced / Core B
builders, the manipulation checks, the clients, timeouts, the cell plan and the freeze."""

import json
import tomllib
from pathlib import Path

import pytest

import adv_plan as ap
import capacity_plan as cp
import stage_e2_plan as e2

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_e2.toml").read_text())
SB = PINS["stage_b"]
KW = dict(gdn_prefill_backend=SB["gdn_prefill_backend"], gpu_memory_utilization=SB["gpu_memory_utilization"],
          vllm_bin="/v", base_env={"PATH": "/usr/bin", "HOME": "/h"})
B0_LOG = ROOT / "results" / "capacity" / "c0-step0" / "raw" / "formal-20261011-adv" / "B0-8192-uniform-r1.server.log"


def frozen(smoke="smoke-e2"):
    e = {**CFG["e2"], "expected_blocks": {**CFG["e2"]["expected_blocks"], "PC": 248},
         "shared_prefix": {**CFG["e2"]["shared_prefix"], "sha256": "s" * 64, "common_prefix": 1630}}
    return {**CFG, "freeze": {"e2": True, "smoke_e2": smoke}, "e2": e}


def test_the_repo_config_is_well_formed_and_unfrozen():
    assert e2.config_problems(CFG, PINS) == [] and CFG["freeze"] == {"e2": False, "smoke_e2": ""}
    assert e2.config_problems(frozen(), PINS) == []
    unmeasured = {**frozen(), "e2": {**frozen()["e2"], "expected_blocks": CFG["e2"]["expected_blocks"]}}
    assert any("PC must be measured" in p for p in e2.config_problems(unmeasured, PINS))
    assert e2.config_problems({**CFG, "e2": {**CFG["e2"], "kv25_bytes": 2 * 2 ** 30}}, PINS)


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


def smoke_session(root, sid="smoke-e2", **over):
    d = root / "results" / "stage_e" / sid
    d.mkdir(parents=True)
    s = {"stage": "E2", "mode": "smoke", "smoke": {"ok": True}, "harness_sha256": e2.harness_sha256(),
         "host": {"system": "Linux", "wsl": True, "nvidia": True}, **over}
    (d / "session.json").write_text(json.dumps(s))


def test_the_formal_plan_has_48_boots_in_three_shuffled_blocks(tmp_path):
    smoke_session(tmp_path)
    p = e2.build_plan(PINS, frozen(), mode="formal", root=tmp_path)
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


def test_formal_plans_need_the_freeze_and_a_passed_smoke(tmp_path):
    with pytest.raises(e2.PlanError, match="freeze.e2"):
        e2.build_plan(PINS, CFG, mode="formal", root=tmp_path)
    with pytest.raises(e2.PlanError, match="smoke first"):
        e2.build_plan(PINS, frozen(), mode="formal", root=tmp_path)
    smoke_session(tmp_path, harness_sha256="0" * 64)
    with pytest.raises(e2.PlanError, match="another harness"):
        e2.build_plan(PINS, frozen(), mode="formal", root=tmp_path)


def test_the_smoke_plan_has_one_boot_per_condition_and_mix_type():
    import run_capacity as rc
    with rc.mini_num_prompts(e2.SMOKE_N):
        p = e2.build_plan(PINS, CFG, mode="smoke")
    assert [(b["cell"], b["mix"]) for b in p["boots"]] == list(e2.SMOKE) and {b["N"] for b in p["boots"]} == {20}
    assert {b["cond"] for b in p["boots"]} == set(e2.CONDITIONS) and p["cap_s"] is None
