"""Tests for benchmarks/stage_e3.py (decision 041 v3 §3): E3's config, plans (Core A's decode jobs for eight GGUFs at
L 2048), the per-platform freeze and pilot evidence, and the runner's rules (Mac lock and 140W adapter, GPU lock and
session B, all eight GGUFs' sha256 and headers, each row's model identity, sentinel drift, the measurement-time cap,
the rerun limit)."""

import json
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

import stage_e3 as e3

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gguf_tensors  # noqa: E402
import produce_gguf_e3 as pe  # noqa: E402

PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
REPO_CFG = tomllib.loads((ROOT / "configs" / "stage_e3.toml").read_text())   # frozen 2026-10-31 (041 §3.7)
CFG = {**REPO_CFG, "freeze": {"e3_metal": False, "e3_cuda": False,               # an explicit DRAFT config
                              "pilot_e3_metal": "", "pilot_e3_cuda": ""}}
NEW = {"4B-Q4_0": {"sha256": "a" * 64, "size_bytes": 2_500_000_000, "token_embd_bytes": 380_000_000,
                   "token_embd_type": "Q6_K", "tied": True, "output_type": "",
                   "tensor_types": {"F32": 177, "Q4_0": 248, "Q6_K": 1}, "model_size": 2_490_000_000,
                   "model_type": "qwen35 4B Q4_0"},
       "9B-Q4_0": {"sha256": "b" * 64, "size_bytes": 5_300_000_000, "token_embd_bytes": 560_000_000,
                   "token_embd_type": "Q4_0", "tied": False, "output_type": "Q6_K",
                   "tensor_types": {"F32": 177, "Q4_0": 249, "Q6_K": 1}, "model_size": 5_290_000_000,
                   "model_type": "qwen35 9B Q4_0"}}


BLANK = {"sha256": "", "size_bytes": 0, "token_embd_bytes": 0, "token_embd_type": "", "output_type": "",
         "tensor_types": {}, "model_size": 0, "model_type": ""}
UNPINNED = {**CFG, "e3": {**CFG["e3"], "files": {k: ({**v, **BLANK} if k in e3.NEW else v)
                                                 for k, v in CFG["e3"]["files"].items()}}}


def pinned(freeze=None):
    """The repo config with the two new GGUFs pinned and, optionally, frozen platforms."""
    files = {k: ({**v, **NEW[k]} if k in NEW else v) for k, v in CFG["e3"]["files"].items()}
    f = {"e3_metal": False, "e3_cuda": False, "pilot_e3_metal": "", "pilot_e3_cuda": ""}
    for platform in freeze or ():
        key = e3.FREEZE_KEYS[platform]
        f.update({key: True, f"pilot_{key}": f"pilot-{key}"})
    return {**CFG, "freeze": f, "e3": {**CFG["e3"], "files": files}}


FILES = e3.config_files(PINS, pinned())
BY_NAME = {f["filename"]: f for f in FILES.values()}


def file_hash(path):
    name = Path(path).name
    if name not in BY_NAME:
        raise OSError(f"no such file {path}")
    return BY_NAME[name]["sha256"]


def fake_header(path, override=None):
    """A header whose header_facts are the pinned values (plus override)."""
    f = {**BY_NAME[Path(path).name], **(override or {})}
    counts = dict(f["tensor_types"])
    tensors = []

    def take(name, typ, nbytes):
        counts[typ] = counts.get(typ, 0) - 1
        tensors.append({"name": name, "type": typ, "bytes": nbytes, "dims": []})

    take("token_embd.weight", f["token_embd_type"], f["token_embd_bytes"])
    if not f["tied"]:
        take("output.weight", f["output_type"], 0)
    tensors += [{"name": f"blk.{t}.{i}", "type": t, "bytes": 0, "dims": []} for t, n in counts.items() for i in range(n)]
    tensors[-1]["bytes"] = f["model_size"] - f["token_embd_bytes"]
    return {"metadata": {}, "tensors": tensors}


def env(platform="macos-metal", adapter="140W USB-C Power Adapter", powermode=2, watts=94):
    def capture(llama_bin, out_path, gpu_lock_held):
        rec = {"runtime": {"llama_cpp": {"build_number": PINS["llama_cpp"]["build_number"],
                                         "commit": PINS["llama_cpp"]["commit"]}},
               "repo": {"commit": "c", "dirty": False, "status": [], "source": "git"}, "probe_errors": []}
        if platform == "macos-metal":
            rec.update(os={"system": "Darwin", "wsl": False}, hardware={"iogpu_wired_limit_mb": 0},
                       power={"source": "AC", "battery_pct": 90, "thermal": {}, "powermode": powermode,
                              "ac_adapter": {"manufacturer": "Apple Inc.", "name": adapter, "wattage_w": watts}})
        else:
            rec.update(os={"system": "Linux", "wsl": True},
                       gpu={"nvidia": [{"enforced_power_limit_w": 300.0, "power_default_limit_w": 300.0}]})
        Path(out_path).write_text(json.dumps(rec))
        return rec
    return capture


def bench_output(tps=50.0, model_type=None, timeouts=None):
    """A llama-bench -o jsonl row that core_a_parse.classify_bench accepts for a decode job, with the model identity
    llama-bench prints (model_type overridable)."""
    def run_proc(argv, out, err, timeout):
        if timeouts is not None:
            timeouts.append(timeout)
        n_thr, poll = argv[argv.index("-t") + 1], argv[argv.index("--poll") + 1]
        model = argv[argv.index("-m") + 1]
        f = BY_NAME[Path(model).name]
        row = {"build_commit": PINS["llama_cpp"]["commit"], "n_threads": int(n_thr), "poll": int(poll),
               "n_batch": 2048, "n_ubatch": 512, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 999,
               "flash_attn": 1, "load_mode": "mmap",
               "n_prompt": int(argv[argv.index("-p") + 1]), "n_gen": int(argv[argv.index("-n") + 1]),
               "n_depth": int(argv[argv.index("-d") + 1]) if "-d" in argv else 0,
               "avg_ts": tps(argv) if callable(tps) else tps, "stddev_ts": 0.5, "avg_ns": 1e9, "stddev_ns": 1e6,
               "samples_ts": [tps(argv) if callable(tps) else tps] * 5, "samples_ns": [1e9] * 5,
               "model_filename": model, "model_size": f["model_size"],
               "model_type": (model_type or {}).get(f["filename"], f["model_type"])}
        Path(out).write_text(json.dumps(row) + "\n")
        Path(err).write_text("")
        return 0, 2.0
    return run_proc


def run(root, plan, sid, **kw):
    platform = plan["platform"]
    args = dict(bin_dir="/b", models_dir="/m", out_dir=root / "results" / "stage_e" / sid, session_id=sid,
                capture_env=env(platform), lock_held=True, lock_check=lambda: True, run_proc=bench_output(),
                sleep=lambda s: None, read_power=lambda: {"source": "AC", "battery_pct": 90}, hash_file=file_hash,
                read_header=fake_header, measure_b=lambda frozen: {"b_mib": frozen, "ok": True, "drift": False},
                pins=PINS, cfg=pinned(), repo_root=root, clock=iter(range(0, 10 ** 9, 10)).__next__,
                expected_plan=plan)
    args.update(kw)
    return e3.run_e3(plan, **args)


def pilots(root, platforms=e3.PLATFORMS):
    for platform in platforms:
        run(root, e3.build_plan(PINS, pinned(), platform=platform, mode="pilot"), f"pilot-{e3.FREEZE_KEYS[platform]}")
    return root


@pytest.fixture(scope="module")
def frozen_root(tmp_path_factory):
    return pilots(tmp_path_factory.mktemp("e3") / "repo")


def formal(root, platform="wsl2-cuda"):
    cfg = pinned(e3.PLATFORMS)
    return e3.build_plan(PINS, cfg, platform=platform, mode="formal", root=root), cfg


# ---------------------------------------------------------------- config and plans

def test_the_repo_config_is_frozen_on_the_two_passed_pilots_of_this_harness():
    """041 §3.7 (2026-10-31): both platforms freeze together on their passed pilots, which ran this harness and this
    config with all eight GGUFs verified; any edit after the freeze fails here."""
    assert e3.config_problems(REPO_CFG) == [] and REPO_CFG["freeze"] == {
        "e3_metal": True, "e3_cuda": True, "pilot_e3_metal": "pilot-e3-metal-20261023",
        "pilot_e3_cuda": "pilot-e3-cuda-20261031"}
    assert e3.harness_sha256() == "85b9842c081355a157a2f7f1c24ffaa7585011546cb016a5cf061c87de957ac3"
    assert e3.freeze_evidence_problems(REPO_CFG, ROOT, PINS) == []
    for platform in e3.PLATFORMS:
        assert len(e3.build_plan(PINS, REPO_CFG, platform=platform, mode="formal", root=ROOT)["jobs"]) > 0


def test_the_draft_config_is_well_formed_unfrozen_with_the_new_ggufs_pinned():
    assert e3.config_problems(CFG) == [] and CFG["freeze"] == {"e3_metal": False, "e3_cuda": False,
                                                               "pilot_e3_metal": "", "pilot_e3_cuda": ""}
    assert set(CFG["e3"]["files"]) == {f"{s}-{q}" for s in ("4B", "9B") for q in ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0")}
    assert e3.new_file_problems(CFG) == [] and e3.config_problems(pinned()) == []
    assert any("4B-Q4_0" in p for p in e3.config_problems({**UNPINNED, "freeze": {**CFG["freeze"], "e3_cuda": True,
                                                                                 "pilot_e3_cuda": "p"}}))


@pytest.mark.parametrize("change, why", [
    ({"cap_hours": {"macos-metal": 2.0, "wsl2-cuda": 1.0}}, "cap_hours"),
    ({"n_params": {"4B": 4205751296}}, "n_params"),
    ({"files": {**CFG["e3"]["files"], "9B-Q6_K": {**CFG["e3"]["files"]["9B-Q6_K"], "model_type": ""}}}, "incomplete"),
    ({"files": {**CFG["e3"]["files"], "9B-Q6_K": {**CFG["e3"]["files"]["9B-Q6_K"], "output_type": ""}}}, "incomplete"),
    ({"files": {**CFG["e3"]["files"], "4B-Q6_K": {**CFG["e3"]["files"]["4B-Q6_K"], "sha256": "x"}}}, "keys"),
])
def test_the_config_pins_041s_caps_the_parameter_counts_and_every_header_value(change, why):
    assert any(why in p for p in e3.config_problems({**CFG, "e3": {**CFG["e3"], **change}}))


def test_the_pinned_header_values_are_the_local_ggufs():
    models = ROOT / "models" / "gguf"
    if not (models / "Qwen3.5-9B-BF16.gguf").is_file():
        pytest.skip("the Core A GGUFs are not on this machine")
    for key, c in CFG["e3"]["files"].items():
        if key in e3.NEW:
            if not (models / c["filename"]).is_file():
                continue
            assert (models / c["filename"]).stat().st_size == c["size_bytes"], key
        info = gguf_tensors.read_header(models / c["filename"])
        assert {**pe.header_facts(info), "model_type": pe.model_type(info, key.split("-")[0])} == \
            {k: c[k] for k in sorted(e3.FILE_KEYS - {"filename"})}, key
    for size in e3.SIZES:
        info = gguf_tensors.read_header(models / f"Qwen3.5-{size}-BF16.gguf")
        n = 0
        for t in info["tensors"]:
            k = 1
            for d in t["dims"]:
                k *= d
            n += k
        assert n == CFG["e3"]["n_params"][size]


def test_the_old_files_come_from_pins_and_the_new_from_the_config():
    f = e3.config_files(PINS, pinned())
    assert f["9B-Q4_K_M"]["sha256"] == PINS["core_a"]["models"][1]["files"]["Q4_K_M"]["sha256"]
    assert f["9B-Q4_K_M"]["size_bytes"] == 5629109216 and f["9B-Q4_K_M"]["tied"] is False
    assert f["9B-Q4_K_M"]["model_type"] == "qwen35 9B Q4_K - Medium" and f["9B-Q4_K_M"]["output_type"] == "Q6_K"
    assert f["4B-Q4_0"]["sha256"] == "a" * 64 and f["4B-Q4_0"]["tied"] is True
    assert all(set(v) == e3.NEW_FILE_KEYS for v in f.values())


def test_a_formal_plan_decodes_eight_configs_three_times_at_2048_between_two_sentinels(frozen_root):
    p = e3.build_plan(PINS, pinned(e3.PLATFORMS), platform="wsl2-cuda", mode="formal", root=frozen_root)
    jobs = p["jobs"]
    assert [j["id"] for j in (jobs[0], jobs[-1])] == ["sentinel-start", "sentinel-end"]
    body = jobs[1:-1]
    assert len(body) == 8 * 3 and {j["kind"] for j in body} == {"decode"} and {j["L"] for j in body} == {2048}
    assert {(j["size"], j["quant"]) for j in body} == {(s, q) for s in ("4B", "9B")
                                                       for q in ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0")}
    assert p["params"]["n_thr"] == PINS["stage_b"]["rtx_n_thr"] and p["expect"]["frozen_b_mib"] == PINS["stage_b"]["b_mib"]
    assert p["cap_s"] == 3600 and p["n_params"] == CFG["e3"]["n_params"]
    a = body[0]["argv"]
    assert a[a.index("-n") + 1] == "128" and a[a.index("-d") + 1] == "1920" and a[a.index("-r") + 1] == "5"
    mac = e3.build_plan(PINS, pinned(e3.PLATFORMS), platform="macos-metal", mode="formal", root=frozen_root)
    assert mac["params"]["n_thr"] == 8 and [j["id"] for j in mac["jobs"]] != [j["id"] for j in jobs]  # own order


def test_a_pilot_runs_the_two_q4_0s_and_the_sentinel_cell_once():
    p = e3.build_plan(PINS, pinned(), platform="macos-metal", mode="pilot")
    assert [(j["size"], j["quant"]) for j in p["jobs"][1:-1]] == [("4B", "Q4_0"), ("9B", "Q4_0"), ("4B", "Q4_K_M")]
    with pytest.raises(e3.StageE3Error, match="4B-Q4_0"):
        e3.build_plan(PINS, UNPINNED, platform="macos-metal", mode="pilot")             # new GGUFs not pinned


def test_formal_plans_need_both_platforms_frozen_and_both_pilots(frozen_root, tmp_path):
    with pytest.raises(e3.StageE3Error, match="freeze.e3_cuda"):
        e3.build_plan(PINS, pinned(("macos-metal",)), platform="macos-metal", mode="formal", root=frozen_root)
    with pytest.raises(e3.StageE3Error, match="freeze.e3_metal"):                  # the shared harness (041 §3.7)
        e3.build_plan(PINS, pinned(("wsl2-cuda",)), platform="wsl2-cuda", mode="formal", root=frozen_root)
    assert e3.build_plan(PINS, pinned(e3.PLATFORMS), platform="wsl2-cuda", mode="formal", root=frozen_root)
    root = tmp_path / "rtx-only"
    pilots(root, ("wsl2-cuda",))
    with pytest.raises(e3.StageE3Error, match="pilots first.*pilot_e3_metal"):
        e3.build_plan(PINS, pinned(e3.PLATFORMS), platform="wsl2-cuda", mode="formal", root=root)


def test_a_pilot_commits_no_speed(tmp_path):
    plan = e3.build_plan(PINS, pinned(), platform="wsl2-cuda", mode="pilot")
    s = run(tmp_path, plan, "p")
    out = tmp_path / "results" / "stage_e" / "p"
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert all(set(r) <= set(e3.PILOT_ROW_KEYS) for r in rows) and [r["outcome"] for r in rows] == ["ok"] * 5
    full = [json.loads(x) for x in (out / "raw" / "rows.full.jsonl").read_text().splitlines()]
    assert all(r["metrics"]["avg_ts"] for r in full)                                 # git-ignored raw/ only
    assert "start_tok_s" not in s["sentinel"] and s["sentinel"]["drift"] is False
    for f in (out / "rows.jsonl", out / "session.json"):
        text = f.read_text()
        assert "avg_ts" not in text and "wall_s" not in text and "tok_s" not in text, f
    assert [x.name for x in (out / "env").iterdir()] == ["preflight.json"]            # per-job captures: raw/env
    assert len(list((out / "raw" / "env").glob("*.json"))) == 2 * len(plan["jobs"])


def test_plans_are_never_written_into_results_stage_e(tmp_path):
    out = ROOT / "results" / "stage_e" / "plan-test-should-not-exist.json"
    assert e3.main(["plan", "--platform", "wsl2-cuda", "--mode", "pilot", "--out", str(out)]) == 2
    assert not out.exists()


def _tamper(root, sid, **over):
    d = root / "results" / "stage_e" / sid
    s = json.loads((d / "session.json").read_text())
    (d / "session.json").write_text(json.dumps({**s, **over}))
    return s


def test_the_pilot_evidence_is_bound_to_this_harness_plan_host_env_and_files(frozen_root, tmp_path):
    cfg = pinned(e3.PLATFORMS)
    assert e3.freeze_evidence_problems(cfg, frozen_root, PINS) == []
    root = tmp_path / "copy"
    shutil.copytree(frozen_root, root)
    s = json.loads((root / "results" / "stage_e" / "pilot-e3_cuda" / "session.json").read_text())
    m = json.loads((root / "results" / "stage_e" / "pilot-e3_metal" / "session.json").read_text())
    adapter = m["ac_adapter"]
    for sid, over, why in (
            ("pilot-e3_cuda", {"harness_sha256": "0" * 64}, "another harness"),
            ("pilot-e3_cuda", {"mode": "formal"}, "not a finished"),
            ("pilot-e3_cuda", {"status": "running"}, "not a finished"),
            ("pilot-e3_cuda", {"host": {"system": "Darwin", "wsl": False, "nvidia": False}}, "host"),
            ("pilot-e3_cuda", {"outcomes": {"ok": 4, "oom": 0, "error": 1, "skipped": 0}}, "every job ok"),
            ("pilot-e3_cuda", {"env_valid": False}, "env not valid"),
            ("pilot-e3_cuda", {"session_b": {"b_mib": 1, "drift": True}}, "undrifted session B"),
            ("pilot-e3_cuda", {"session_b": None}, "undrifted session B"),
            ("pilot-e3_cuda", {"ggufs_sha256": {k: v for k, v in list(s["ggufs_sha256"].items())[:3]}}, "all eight"),
            ("pilot-e3_cuda", {"header_verified": {**s["header_verified"], "9B-Q4_0": False}}, "headers"),
            ("pilot-e3_metal", {"ac_adapter": {**adapter, "name": "70W USB-C Power Adapter"}}, "140W"),
            ("pilot-e3_metal", {"ac_adapter": {**adapter, "wattage_w": None}}, "140W")):
        orig = _tamper(root, sid, **over)
        assert any(why in p for p in e3.freeze_evidence_problems(cfg, root, PINS)), over
        _tamper(root, sid, **orig)
    assert e3.freeze_evidence_problems(cfg, root, PINS) == []
    other = pinned(e3.PLATFORMS)
    other["e3"]["files"]["9B-Q4_0"] = {**other["e3"]["files"]["9B-Q4_0"], "sha256": "c" * 64}
    assert any("pilot's plan" in p for p in e3.freeze_evidence_problems(other, root, PINS))
    d = root / "results" / "stage_e" / "pilot-e3_cuda"
    plan = json.loads((d / "plan.json").read_text())
    (d / "plan.json").write_text(json.dumps({**plan, "cap_s": 7200}))
    _tamper(root, "pilot-e3_cuda", plan_sha256=e3.sha256_json({**plan, "cap_s": 7200}))
    assert any("pilot's plan" in p for p in e3.freeze_evidence_problems(cfg, root, PINS))


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
    assert s["ac_adapter"]["wattage_w"] == 94 and s["status"] == "finished"
    s = run(tmp_path, plan, "c", capture_env=env(adapter="70W USB-C Power Adapter"))
    assert not s["env_valid"] and any("140W" in f["reason"] for f in s["env_flags"])

    seq = iter([2] * 3 + [0] * 100)

    def drifting(llama_bin, out_path, gpu_lock_held):
        return env(powermode=next(seq))(llama_bin, out_path, gpu_lock_held)

    root = tmp_path / "f"
    shutil.copytree(frozen_root, root)
    plan, cfg = formal(root, "macos-metal")
    with pytest.raises(e3.StageE3Error, match="exactly build_plan's"):
        run(root, {**plan, "jobs": plan["jobs"][:3]}, "d", expected_plan=plan, cfg=cfg)
    with pytest.raises(e3.StageE3Error, match="140W"):
        run(root, plan, "e", expected_plan=plan, cfg=cfg, capture_env=env(adapter="70W USB-C Power Adapter"))
    assert json.loads((root / "results" / "stage_e" / "e" / "session.json").read_text())["status"] == "refused"
    s = run(root, plan, "f", expected_plan=plan, cfg=cfg, capture_env=drifting)
    assert not s["env_valid"] and any("one powermode" in f["reason"] for f in s["env_flags"])


def test_all_eight_ggufs_and_their_headers_are_checked_before_the_session(tmp_path):
    plan = e3.build_plan(PINS, pinned(), platform="wsl2-cuda", mode="pilot")
    hashed = []
    s = run(tmp_path, plan, "ok", hash_file=lambda p: hashed.append(Path(p).name) or file_hash(p))
    assert sorted(hashed) == sorted(BY_NAME) and s["ggufs_sha256"] == {n: f["sha256"] for n, f in BY_NAME.items()}
    assert s["header_verified"] == {k: True for k in FILES}
    with pytest.raises(e3.StageE3Error, match="sha256"):
        run(tmp_path, plan, "a", hash_file=lambda path: "0" * 64)
    with pytest.raises(e3.StageE3Error, match="GPU lock"):
        run(tmp_path, plan, "b", lock_check=lambda: False)
    for override, why in (({"token_embd_type": "Q8_0"}, "token_embd_type"), ({"tied": True}, "tied"),
                          ({"tensor_types": {"F32": 177, "Q4_0": 249, "Q6_K": 1}, "model_size": 1}, "model_size")):
        bad = lambda p, o=override: fake_header(p, o if Path(p).name == "Qwen3.5-9B-Q6_K.gguf" else None)  # noqa: E731
        with pytest.raises(e3.StageE3Error, match=why):
            run(tmp_path, plan, "h", read_header=bad)
    with pytest.raises(e3.StageE3Error, match="unreadable"):
        run(tmp_path, plan, "h", read_header=lambda p: (_ for _ in ()).throw(ValueError("not a GGUF")))
    for sid in ("a", "b", "h"):
        assert not (tmp_path / "results" / "stage_e" / sid).exists()


def test_a_row_with_another_models_identity_is_an_error(tmp_path):
    plan = e3.build_plan(PINS, pinned(), platform="wsl2-cuda", mode="pilot")
    s = run(tmp_path, plan, "x", run_proc=bench_output(model_type={"Qwen3.5-9B-Q4_0.gguf": "qwen35 9B Q4_K - Medium"}))
    rows = [json.loads(x) for x in (tmp_path / "results" / "stage_e" / "x" / "rows.jsonl").read_text().splitlines()]
    bad = [r for r in rows if r["outcome"] != "ok"]
    assert s["outcomes"]["error"] == 1 and bad[0]["quant"] == "Q4_0" and bad[0]["stage"] == "identity"
    assert "model_type" in bad[0]["reason"]


def test_session_b_is_measured_before_the_directory_and_its_drift_voids_the_session(tmp_path):
    plan = e3.build_plan(PINS, pinned(), platform="wsl2-cuda", mode="pilot")

    def fails(frozen):
        raise e3.stage_b.StageBError("another GPU process is running")

    with pytest.raises(e3.StageE3Error, match="session B not measured"):
        run(tmp_path, plan, "a", measure_b=fails)
    assert not (tmp_path / "results" / "stage_e" / "a").exists()
    s = run(tmp_path, plan, "b", measure_b=lambda frozen: {"b_mib": frozen + 900, "drift": True})
    assert s["env_valid"] is False and any("session B" in f["reason"] for f in s["env_flags"])


def test_a_formal_rtx_session_measures_b_and_records_the_sentinel_drift(frozen_root, tmp_path):
    root = tmp_path / "r"
    shutil.copytree(frozen_root, root)
    plan, cfg = formal(root)
    measured = []
    s = run(root, plan, "f", expected_plan=plan, cfg=cfg,
            measure_b=lambda frozen: measured.append(frozen) or {"b_mib": frozen, "ok": True, "drift": False})
    assert measured == [PINS["stage_b"]["b_mib"]] and s["session_b"]["ok"] and s["sentinel_ok"] is True
    root = tmp_path / "g"
    shutil.copytree(frozen_root, root)
    calls = iter(range(10 ** 6))
    s = run(root, plan, "g", expected_plan=plan, cfg=cfg,
            run_proc=bench_output(lambda argv: 50.0 if next(calls) == 0 else 60.0))  # end sentinel 20 % faster
    assert s["sentinel_ok"] is False and s["sentinel"]["drift_pct"] == pytest.approx(20.0)


def test_a_pilot_too_must_be_build_plans_and_only_pilot_or_formal_runs(frozen_root, tmp_path):
    pilot = e3.build_plan(PINS, pinned(), platform="wsl2-cuda", mode="pilot")
    full, _ = formal(frozen_root)
    with pytest.raises(e3.StageE3Error, match="exactly build_plan's"):        # the whole design labelled a pilot
        run(tmp_path, {**full, "mode": "pilot"}, "a", expected_plan=pilot)
    with pytest.raises(e3.StageE3Error, match="pilot or formal"):
        run(tmp_path, {**pilot, "mode": "smoke"}, "b")
    with pytest.raises(e3.StageE3Error, match="another harness"):
        run(tmp_path, {**pilot, "harness_sha256": "0" * 64}, "c")
    assert e3.formal_attempts(tmp_path, "wsl2-cuda") == [] and not (tmp_path / "results" / "stage_e" / "a").exists()


def test_a_plan_pins_the_configs_content_without_freeze():
    a = e3.build_plan(PINS, pinned(), platform="wsl2-cuda", mode="pilot")
    cfg = pinned()
    cfg["freeze"] = {**cfg["freeze"], "e3_metal": True, "pilot_e3_metal": "p"}
    assert e3.build_plan(PINS, cfg, platform="wsl2-cuda", mode="pilot") == a
    other = pinned()
    other["e3"] = {**other["e3"], "seed": 1}
    assert e3.config_sha256(other) != a["stage_e3_sha256"] == e3.config_sha256(pinned())


def test_a_session_directory_is_exactly_results_stage_e_id(tmp_path):
    plan = e3.build_plan(PINS, pinned(), platform="wsl2-cuda", mode="pilot")
    with pytest.raises(e3.StageE3Error, match="results/stage_e/<session id>"):
        run(tmp_path, plan, "x", out_dir=tmp_path / "results" / "stage_e" / "a" / "b")


def test_one_valid_formal_session_or_one_void_attempt_and_its_rerun(frozen_root, tmp_path):
    root = tmp_path / "valid"
    shutil.copytree(frozen_root, root)
    plan, cfg = formal(root)
    assert run(root, plan, "f1", expected_plan=plan, cfg=cfg)["env_valid"] is True
    with pytest.raises(e3.StageE3Error, match="already has formal E3 attempts"):
        run(root, plan, "f2", expected_plan=plan, cfg=cfg)

    root = tmp_path / "void"
    shutil.copytree(frozen_root, root)
    s = run(root, plan, "v1", expected_plan=plan, cfg=cfg, measure_b=lambda f: {"b_mib": f + 900, "drift": True})
    assert s["env_valid"] is False
    assert run(root, plan, "v2", expected_plan=plan, cfg=cfg, measure_b=lambda f: {"b_mib": f + 900, "drift": True})
    with pytest.raises(e3.StageE3Error, match="already has formal E3 attempts"):
        run(root, plan, "v3", expected_plan=plan, cfg=cfg)

    root = tmp_path / "interrupted"
    shutil.copytree(frozen_root, root)
    calls = iter(range(10 ** 6))

    def dies(argv, out, err, timeout):
        if next(calls) == 2:
            raise KeyboardInterrupt
        return bench_output()(argv, out, err, timeout)

    with pytest.raises(KeyboardInterrupt):
        run(root, plan, "i1", expected_plan=plan, cfg=cfg, run_proc=dies)
    stub = json.loads((root / "results" / "stage_e" / "i1" / "session.json").read_text())
    assert stub["status"] == "running" and [a["valid"] for a in e3.formal_attempts(root, "wsl2-cuda")] == [False]
    assert run(root, plan, "i2", expected_plan=plan, cfg=cfg)["env_valid"] is True
    with pytest.raises(e3.StageE3Error, match="already has formal E3 attempts"):
        run(root, plan, "i3", expected_plan=plan, cfg=cfg)

    root = tmp_path / "torn"                                       # an unreadable session.json still counts
    shutil.copytree(frozen_root, root)
    run(root, plan, "t1", expected_plan=plan, cfg=cfg, measure_b=lambda f: {"b_mib": f + 900, "drift": True})
    (root / "results" / "stage_e" / "t1" / "session.json").write_text('{"stage": "E3", "mo')
    assert not (root / "results" / "stage_e" / "t1" / "session.json.tmp").exists()
    attempts = e3.formal_attempts(root, "wsl2-cuda")
    assert [(a["dir"].name, a["valid"], a["started_at"]) for a in attempts] == [("t1", False, "")]
    run(root, plan, "t2", expected_plan=plan, cfg=cfg, measure_b=lambda f: {"b_mib": f + 900, "drift": True})
    with pytest.raises(e3.StageE3Error, match="already has formal E3 attempts"):
        run(root, plan, "t3", expected_plan=plan, cfg=cfg)


class FakeTime:
    """sleep() advances the clock; every clock() read also adds a job's 60 s."""
    def __init__(self):
        self.now = 0.0

    def sleep(self, s):
        self.now += s

    def clock(self):
        self.now += 60
        return self.now


class Battery:
    """AC at 90 %, except one charge-gate read at low_at that drops to 30 % for `polls` polls of 60 s."""
    def __init__(self, low_at, polls):
        self.calls, self.low_at, self.left = 0, low_at, polls

    def __call__(self):
        self.calls += 1
        if self.calls == self.low_at or (self.calls > self.low_at and self.left > 1):
            if self.calls > self.low_at:
                self.left -= 1
            return {"source": "AC", "battery_pct": 30}
        return {"source": "AC", "battery_pct": 90}


def test_the_time_cap_counts_measurement_only_and_skips_the_rest(frozen_root, tmp_path):
    root = tmp_path / "cap"
    shutil.copytree(frozen_root, root)
    plan, cfg = formal(root)
    ticks = iter(range(0, 10 ** 9, 600))                               # 10 min per clock read
    timeouts = []
    s = run(root, plan, "f", expected_plan=plan, cfg=cfg, clock=ticks.__next__, run_proc=bench_output(timeouts=timeouts))
    rows = [json.loads(x) for x in (root / "results" / "stage_e" / "f" / "rows.jsonl").read_text().splitlines()]
    assert s["outcomes"]["skipped"] > 0 and rows[-2]["reason"].startswith("time cap")
    assert rows[-1]["job_id"] == "sentinel-end" and rows[-1]["outcome"] == "ok" and s["cap_reached"] is True
    assert timeouts[0] == 3600 - 600 + e3.CAP_MARGIN_S and timeouts[-1] == 7200

    root = tmp_path / "charge"
    shutil.copytree(frozen_root, root)
    plan, cfg = formal(root, "macos-metal")
    t = FakeTime()
    s = run(root, plan, "m", expected_plan=plan, cfg=cfg, sleep=t.sleep, clock=t.clock,
            read_power=Battery(low_at=6, polls=70), capture_env=env("macos-metal"))
    assert [p["waited_s"] for p in s["charge_pauses"]] == [4200] and s["charge_pauses"][0]["job_id"]
    assert s["outcomes"]["skipped"] == 0 and t.now > plan["cap_s"] > s["measured_s"]      # the pause is not counted
    root = tmp_path / "pre"
    shutil.copytree(frozen_root, root)
    t = FakeTime()
    s = run(root, plan, "p", expected_plan=plan, cfg=cfg, sleep=t.sleep, clock=t.clock,
            read_power=Battery(low_at=1, polls=70), capture_env=env("macos-metal"))
    assert s["charge_pauses"][0]["job_id"] is None and s["outcomes"]["skipped"] == 0       # before the clock starts


def test_an_interrupt_before_the_first_job_is_refused_not_an_attempt(frozen_root, tmp_path):
    root = tmp_path / "r"
    shutil.copytree(frozen_root, root)
    plan, cfg = formal(root)

    def interrupted(**kw):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(root, plan, "f1", expected_plan=plan, cfg=cfg, capture_env=interrupted)
    s = json.loads((root / "results" / "stage_e" / "f1" / "session.json").read_text())
    assert s["status"] == "refused" and "KeyboardInterrupt" in s["refused"][0]
    assert e3.formal_attempts(root, "wsl2-cuda") == []


def test_the_refused_window_ends_when_the_first_job_process_starts(frozen_root, tmp_path):
    root = tmp_path / "r"
    shutil.copytree(frozen_root, root)
    plan, cfg = formal(root)
    captures = []

    def second_capture_exits(**kw):
        captures.append(kw)
        if len(captures) == 2:                                  # the first job's env_before, before its process
            raise SystemExit(143)
        return env("wsl2-cuda")(**kw)

    with pytest.raises(SystemExit):
        run(root, plan, "f1", expected_plan=plan, cfg=cfg, capture_env=second_capture_exits)
    s = json.loads((root / "results" / "stage_e" / "f1" / "session.json").read_text())
    assert s["status"] == "refused" and s["refused"][0].startswith("interrupted before the first job: SystemExit")
    assert e3.formal_attempts(root, "wsl2-cuda") == []
    calls = []

    def first_job_interrupted(*a):
        calls.append(a)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):                     # the first job's process started: an attempt
        run(root, plan, "f2", expected_plan=plan, cfg=cfg, run_proc=first_job_interrupted)
    assert [a["valid"] for a in e3.formal_attempts(root, "wsl2-cuda")] == [False] and len(calls) == 1
    with pytest.raises(SystemExit, match="129"):
        e3._exit_on_signal(1, None)


def test_an_interrupt_in_the_mac_first_jobs_charge_gate_is_refused(frozen_root, tmp_path):
    root = tmp_path / "r"
    shutil.copytree(frozen_root, root)
    plan, cfg = formal(root, "macos-metal")
    reads = []

    def power():                                                # the preflight gate passes, the first job's raises
        reads.append(1)
        if len(reads) > 1:
            raise KeyboardInterrupt
        return {"source": "AC", "battery_pct": 90}

    with pytest.raises(KeyboardInterrupt):
        run(root, plan, "m1", expected_plan=plan, cfg=cfg, read_power=power)
    s = json.loads((root / "results" / "stage_e" / "m1" / "session.json").read_text())
    assert s["status"] == "refused" and s["refused"][0].startswith("interrupted before the first job")
    assert e3.formal_attempts(root, "macos-metal") == []


@pytest.mark.parametrize("exc, how", [(SystemExit(143), "interrupted"), (OSError("x"), "failed")])
def test_refused_why_tells_an_interrupt_from_a_failure(exc, how):
    assert e3.refused_why(exc).startswith(f"{how} before the first job: {type(exc).__name__}")
