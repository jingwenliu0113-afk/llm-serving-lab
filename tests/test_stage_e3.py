"""Tests for benchmarks/stage_e3.py (decision 041 v3 §3): E3's config, plans (Core A's decode jobs for eight GGUFs at
L 2048), the per-platform freeze and pilot evidence, and the runner's rules (Mac lock and 140W adapter, GPU lock and
session B, GGUF sha256, sentinel drift, time cap)."""

import json
import shutil
import tomllib
from pathlib import Path

import pytest

import stage_e3 as e3

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_e3.toml").read_text())
NEW = {"4B-Q4_0": {"sha256": "a" * 64, "size_bytes": 2_500_000_000, "token_embd_bytes": 380_000_000,
                   "tensor_types": {"F32": 177, "Q4_0": 248, "Q6_K": 1}},
       "9B-Q4_0": {"sha256": "b" * 64, "size_bytes": 5_300_000_000, "token_embd_bytes": 560_000_000,
                   "tensor_types": {"F32": 177, "Q4_0": 249, "Q6_K": 1}}}


def pinned(freeze=None):
    """The repo config with the two new GGUFs pinned and, optionally, frozen platforms."""
    files = {k: ({**v, **NEW[k]} if k in NEW else v) for k, v in CFG["e3"]["files"].items()}
    f = {"e3_metal": False, "e3_cuda": False, "pilot_e3_metal": "", "pilot_e3_cuda": ""}
    for platform in freeze or ():
        key = e3.FREEZE_KEYS[platform]
        f.update({key: True, f"pilot_{key}": f"pilot-{key}"})
    return {**CFG, "freeze": f, "e3": {**CFG["e3"], "files": files}}


def file_hash(path):
    name = Path(path).name
    for key, f in e3.config_files(PINS, pinned()).items():
        if f["filename"] == name:
            return f["sha256"]
    raise OSError(f"no such file {path}")


def env(platform="macos-metal", adapter="140W USB-C Power Adapter", powermode=2):
    def capture(llama_bin, out_path, gpu_lock_held):
        rec = {"runtime": {"llama_cpp": {"build_number": PINS["llama_cpp"]["build_number"],
                                         "commit": PINS["llama_cpp"]["commit"]}},
               "repo": {"commit": "c", "dirty": False, "status": [], "source": "git"}, "probe_errors": []}
        if platform == "macos-metal":
            rec.update(os={"system": "Darwin", "wsl": False}, hardware={"iogpu_wired_limit_mb": 0},
                       power={"source": "AC", "battery_pct": 90, "thermal": {}, "powermode": powermode,
                              "ac_adapter": {"manufacturer": "Apple Inc.", "name": adapter, "watts": 94}})
        else:
            rec.update(os={"system": "Linux", "wsl": True},
                       gpu={"nvidia": [{"enforced_power_limit_w": 300.0, "power_default_limit_w": 300.0}]})
        Path(out_path).write_text(json.dumps(rec))
        return rec
    return capture


def bench_output(tps=50.0):
    """A llama-bench -o jsonl row that core_a_parse.classify_bench accepts for a decode job."""
    def run_proc(argv, out, err, timeout):
        n_thr, poll = argv[argv.index("-t") + 1], argv[argv.index("--poll") + 1]
        row = {"build_commit": PINS["llama_cpp"]["commit"], "n_threads": int(n_thr), "poll": int(poll),
               "n_batch": 2048, "n_ubatch": 512, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 999,
               "flash_attn": 1, "load_mode": "mmap",
               "n_prompt": int(argv[argv.index("-p") + 1]), "n_gen": int(argv[argv.index("-n") + 1]),
               "n_depth": int(argv[argv.index("-d") + 1]) if "-d" in argv else 0,
               "avg_ts": tps(argv) if callable(tps) else tps, "stddev_ts": 0.5, "avg_ns": 1e9, "stddev_ns": 1e6,
               "samples_ts": [tps(argv) if callable(tps) else tps] * 5, "samples_ns": [1e9] * 5,
               "model_filename": argv[argv.index("-m") + 1]}
        Path(out).write_text(json.dumps(row) + "\n")
        Path(err).write_text("")
        return 0, 2.0
    return run_proc


def run(root, plan, sid, **kw):
    platform = plan["platform"]
    args = dict(bin_dir="/b", models_dir="/m", out_dir=root / "results" / "stage_e" / sid, session_id=sid,
                capture_env=env(platform), lock_held=True, lock_check=lambda: True, run_proc=bench_output(),
                sleep=lambda s: None, read_power=lambda: {"source": "AC", "battery_pct": 90}, hash_file=file_hash,
                measure_b=lambda frozen: {"b_mib": frozen, "ok": True}, pins=PINS, cfg=pinned(),
                repo_root=root, clock=iter(range(0, 10 ** 9, 10)).__next__)
    args.update(kw)
    return e3.run_e3(plan, **args)


def pilots(root, platforms=e3.PLATFORMS):
    for platform in platforms:
        run(root, e3.build_plan(PINS, pinned(), platform=platform, mode="pilot"), f"pilot-{e3.FREEZE_KEYS[platform]}")
    return root


@pytest.fixture(scope="module")
def frozen_root(tmp_path_factory):
    return pilots(tmp_path_factory.mktemp("e3") / "repo")


# ---------------------------------------------------------------- config and plans

def test_the_repo_config_is_well_formed_unfrozen_and_waits_for_the_new_gguf_pins():
    assert e3.config_problems(CFG) == [] and CFG["freeze"] == {"e3_metal": False, "e3_cuda": False,
                                                               "pilot_e3_metal": "", "pilot_e3_cuda": ""}
    assert set(CFG["e3"]["files"]) == {f"{s}-{q}" for s in ("4B", "9B") for q in ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0")}
    assert e3.config_problems(pinned()) == []
    assert any("4B-Q4_0" in p for p in e3.config_problems({**CFG, "freeze": {**CFG["freeze"], "e3_cuda": True,
                                                                            "pilot_e3_cuda": "p"}}))


def test_the_old_files_come_from_pins_and_the_new_from_the_config():
    f = e3.config_files(PINS, pinned())
    assert f["9B-Q4_K_M"]["sha256"] == PINS["core_a"]["models"][1]["files"]["Q4_K_M"]["sha256"]
    assert f["9B-Q4_K_M"]["size_bytes"] == 5629109216 and f["9B-Q4_K_M"]["tied"] is False
    assert f["4B-Q4_0"]["sha256"] == "a" * 64 and f["4B-Q4_0"]["tied"] is True


def test_a_formal_plan_decodes_eight_configs_three_times_at_2048_between_two_sentinels(frozen_root):
    p = e3.build_plan(PINS, pinned(e3.PLATFORMS), platform="wsl2-cuda", mode="formal", root=frozen_root)
    jobs = p["jobs"]
    assert [j["id"] for j in (jobs[0], jobs[-1])] == ["sentinel-start", "sentinel-end"]
    body = jobs[1:-1]
    assert len(body) == 8 * 3 and {j["kind"] for j in body} == {"decode"} and {j["L"] for j in body} == {2048}
    assert {(j["size"], j["quant"]) for j in body} == {(s, q) for s in ("4B", "9B")
                                                       for q in ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0")}
    assert p["params"]["n_thr"] == PINS["stage_b"]["rtx_n_thr"] and p["expect"]["frozen_b_mib"] == PINS["stage_b"]["b_mib"]
    a = body[0]["argv"]
    assert a[a.index("-n") + 1] == "128" and a[a.index("-d") + 1] == "1920" and a[a.index("-r") + 1] == "5"
    mac = e3.build_plan(PINS, pinned(e3.PLATFORMS), platform="macos-metal", mode="formal", root=frozen_root)
    assert mac["params"]["n_thr"] == 8 and [j["id"] for j in mac["jobs"]] != [j["id"] for j in jobs]  # own order


def test_a_pilot_runs_the_two_q4_0s_and_the_sentinel_cell_once():
    p = e3.build_plan(PINS, pinned(), platform="macos-metal", mode="pilot")
    assert [(j["size"], j["quant"]) for j in p["jobs"][1:-1]] == [("4B", "Q4_0"), ("9B", "Q4_0"), ("4B", "Q4_K_M")]
    with pytest.raises(e3.StageE3Error, match="4B-Q4_0"):
        e3.build_plan(PINS, CFG, platform="macos-metal", mode="pilot")                  # new GGUFs not pinned yet


def test_formal_plans_need_their_platforms_freeze_and_pilot(frozen_root, tmp_path):
    with pytest.raises(e3.StageE3Error, match="freeze.e3_cuda"):
        e3.build_plan(PINS, pinned(("macos-metal",)), platform="wsl2-cuda", mode="formal", root=frozen_root)
    assert e3.build_plan(PINS, pinned(("wsl2-cuda",)), platform="wsl2-cuda", mode="formal", root=frozen_root)
    with pytest.raises(e3.StageE3Error, match="pilot first"):
        e3.build_plan(PINS, pinned(("wsl2-cuda",)), platform="wsl2-cuda", mode="formal", root=tmp_path)


def test_the_pilot_evidence_is_bound_to_this_harness_plan_and_host(frozen_root, tmp_path):
    cfg = pinned(e3.PLATFORMS)
    assert e3.freeze_evidence_problems(cfg, frozen_root, PINS) == []
    root = tmp_path / "copy"
    shutil.copytree(frozen_root, root)
    d = root / "results" / "stage_e" / "pilot-e3_cuda"
    s = json.loads((d / "session.json").read_text())
    for over, why in (({"harness_sha256": "0" * 64}, "another harness"), ({"mode": "formal"}, "not a"),
                      ({"host": {"system": "Darwin", "wsl": False, "nvidia": False}}, "host"),
                      ({"outcomes": {"ok": 4, "oom": 0, "error": 1, "skipped": 0}}, "every job ok")):
        (d / "session.json").write_text(json.dumps({**s, **over}))
        assert any(why in p for p in e3.freeze_evidence_problems(cfg, root, PINS)), over
    (d / "session.json").write_text(json.dumps(s))
    other = pinned(e3.PLATFORMS)
    other["e3"]["files"]["9B-Q4_0"] = {**other["e3"]["files"]["9B-Q4_0"], "sha256": "c" * 64}
    assert any("pilot's plan" in p for p in e3.freeze_evidence_problems(other, root, PINS))


def test_the_harness_covers_e3_and_its_unguarded_imports():
    assert {Path(f).name for f in e3.HARNESS_FILES} >= {"stage_e3.py", "stage_e3_analysis.py", "produce_gguf_e3.py",
                                                        "core_a_plan.py", "stage_b.py", "gguf_tensors.py"}
    assert all((ROOT / f).is_file() for f in e3.HARNESS_FILES) and len(e3.harness_sha256()) == 64


# ---------------------------------------------------------------- the runner

def test_a_mac_session_needs_the_lock_the_140w_adapter_and_one_powermode(frozen_root, tmp_path):
    plan = e3.build_plan(PINS, pinned(), platform="macos-metal", mode="pilot")
    with pytest.raises(e3.StageE3Error, match="Mac heavy lock"):
        run(tmp_path, plan, "a", lock_check=lambda: False)
    s = run(tmp_path, plan, "b")
    assert s["env_valid"] and s["host"]["system"] == "Darwin" and s["outcomes"]["ok"] == len(plan["jobs"])
    s = run(tmp_path, plan, "c", capture_env=env(adapter="70W USB-C Power Adapter"))
    assert not s["env_valid"] and any("140W" in f["reason"] for f in s["env_flags"])

    seq = iter([2] * 3 + [0] * 100)

    def drifting(llama_bin, out_path, gpu_lock_held):
        return env(powermode=next(seq))(llama_bin, out_path, gpu_lock_held)

    cfg = pinned(e3.PLATFORMS)
    formal = e3.build_plan(PINS, cfg, platform="macos-metal", mode="formal", root=frozen_root)
    with pytest.raises(e3.StageE3Error, match="exactly build_plan's"):
        run(tmp_path, {**formal, "jobs": formal["jobs"][:3]}, "d", expected_plan=formal, cfg=cfg)
    with pytest.raises(e3.StageE3Error, match="140W"):
        run(tmp_path, formal, "e", expected_plan=formal, cfg=cfg, capture_env=env(adapter="70W USB-C Power Adapter"))
    s = run(tmp_path, formal, "f", expected_plan=formal, cfg=cfg, capture_env=drifting)
    assert not s["env_valid"] and any("one powermode" in f["reason"] for f in s["env_flags"])


def test_every_gguf_is_checked_against_its_pin_before_the_session(tmp_path):
    plan = e3.build_plan(PINS, pinned(), platform="wsl2-cuda", mode="pilot")
    with pytest.raises(e3.StageE3Error, match="sha256"):
        run(tmp_path, plan, "a", hash_file=lambda path: "0" * 64)
    with pytest.raises(e3.StageE3Error, match="GPU lock"):
        run(tmp_path, plan, "b", lock_check=lambda: False)
    assert not (tmp_path / "results" / "stage_e" / "a").exists()


def test_a_formal_rtx_session_measures_b_and_records_the_sentinel_drift(frozen_root, tmp_path):
    cfg = pinned(e3.PLATFORMS)
    plan = e3.build_plan(PINS, cfg, platform="wsl2-cuda", mode="formal", root=frozen_root)
    measured = []
    s = run(tmp_path, plan, "f", expected_plan=plan, cfg=cfg,
            measure_b=lambda frozen: measured.append(frozen) or {"b_mib": frozen, "ok": True})
    assert measured == [PINS["stage_b"]["b_mib"]] and s["session_b"]["ok"] and s["sentinel_ok"] is True
    calls = iter(range(10 ** 6))
    s = run(tmp_path, plan, "g", expected_plan=plan, cfg=cfg,
            run_proc=bench_output(lambda argv: 50.0 if next(calls) == 0 else 60.0))  # end sentinel 20 % faster
    assert s["sentinel_ok"] is False and s["sentinel"]["drift_pct"] == pytest.approx(20.0)


def test_the_time_cap_skips_the_rest(frozen_root, tmp_path):
    cfg = pinned(e3.PLATFORMS)
    plan = e3.build_plan(PINS, cfg, platform="wsl2-cuda", mode="formal", root=frozen_root)
    ticks = iter(range(0, 10 ** 9, 600))                               # 10 min per clock read
    s = run(tmp_path, plan, "f", expected_plan=plan, cfg=cfg, clock=ticks.__next__)
    rows = [json.loads(x) for x in (tmp_path / "results" / "stage_e" / "f" / "rows.jsonl").read_text().splitlines()]
    assert s["outcomes"]["skipped"] > 0 and rows[-2]["reason"].startswith("time cap")
    assert rows[-1]["job_id"] == "sentinel-end" and rows[-1]["outcome"] == "ok" and s["cap_reached"] is True
