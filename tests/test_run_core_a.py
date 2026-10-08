"""Tests for benchmarks/run_core_a.py: executes a plan one process per job with
fake llama.cpp binaries, and checks the fixed results layout (docs/RESULTS_FORMAT.md),
env records around every process, sentinel drift and the formal-mode guards."""

import json
import stat
import sys
import tomllib
from pathlib import Path

import pytest

import core_a_plan as cp
import run_core_a as rc

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"

FAKE_BENCH = r'''
import json, os, sys
a = sys.argv
g = lambda f: a[a.index(f) + 1]
p, n = int(g("-p")), int(g("-n"))
d = int(g("-d")) if "-d" in a else 0
speed = float(os.environ.get("FAKE_TG_SPEED", "50.0"))
marker = os.environ.get("FAKE_SENTINEL_FILE")
if marker and p == 0 and d == int(os.environ["FAKE_SENTINEL_DEPTH"]) and g("-m").endswith(os.environ["FAKE_SENTINEL_MODEL"]):
    # sentinel: first call fast, second call slower by FAKE_DRIFT
    first = not os.path.exists(marker)
    open(marker, "a").write("x")
    if not first:
        speed *= 1.0 - float(os.environ.get("FAKE_DRIFT", "0"))
if g("-m").endswith(os.environ["FAKE_OOM_MODEL"]) and (p >= 8192 or d >= 8064):
    sys.stderr.write("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 15012.00 MiB on device 0: cudaMalloc failed: out of memory\n")
    sys.stderr.write("llama_bench: error: failed to create context with model '%s'\n" % g("-m"))
    sys.exit(1)
row = {"build_commit": "__COMMIT__", "build_number": 11146, "n_batch": 2048, "n_ubatch": 512,
       "n_threads": int(g("-t")), "poll": int(g("--poll")), "type_k": "f16", "type_v": "f16",
       "n_gpu_layers": 999, "flash_attn": 1, "load_mode": "mmap", "n_prompt": p, "n_gen": n, "n_depth": d,
       "avg_ns": 1000000, "stddev_ns": 1000, "avg_ts": speed, "stddev_ts": 0.1, "samples_ns": [1000000] * 5}
print(json.dumps(row))
'''.replace("__COMMIT__", COMMIT)

FAKE_COMPLETION = r'''
import json, os, sys
a = sys.argv
g = lambda f: a[a.index(f) + 1]
c = int(g("-c"))
prompt = open(g("-f")).read()
n_tok = prompt.count(" the")
log = lambda m, lv="info": print(json.dumps({"type": "log", "time": 1, "level": lv, "msg": m + "\n"}))
if g("-m").endswith(os.environ["FAKE_OOM_MODEL"]) and c >= 8192:
    log("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5120.00 MiB on device 0: cudaMalloc failed: out of memory", "error")
    log("llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache", "error")
    log("llama_completion: error: unable to create context", "error")
    sys.exit(1)
log("llama_context: n_ctx                 = %d" % c)
log("llama_context: flash_attn            = enabled")
log("common_perf_print: prompt eval time =     812.33 ms / %5d tokens (    0.42 ms per token,  2363.58 tokens per second)" % n_tok)
log("common_perf_print:        eval time =    2911.20 ms /   127 runs   (   22.92 ms per token,    43.62 tokens per second)")
print(json.dumps({"type": "fit_memory_breakdown", "data": {"unit": "MiB", "rows": [
    {"kind": "device", "name": "CUDA0", "description": "RTX 5070 Ti", "total": 16303, "free": 4000, "self": 11000,
     "model": 9000, "context": 1500, "compute": 500, "unaccounted": 1303},
    {"kind": "host", "name": "Host", "self": 400, "model": 390, "context": 0, "compute": 10}]}}))
'''


@pytest.fixture(autouse=True)
def fake_model_env(monkeypatch):
    # The fake binaries learn which files are "the sentinel" and "the one that OOMs"
    # from pins, so these tests do not depend on the chosen model family (decision 008).
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        pins = tomllib.load(fh)
    size, quant, L = cp.sentinel_cell(pins)
    files = {(m["size"], q): f["filename"] for m in pins["core_a"]["models"] for q, f in m["files"].items()}
    largest = pins["core_a"]["models"][-1]
    monkeypatch.setenv("FAKE_SENTINEL_MODEL", files[(size, quant)])
    monkeypatch.setenv("FAKE_SENTINEL_DEPTH", str(L - cp.N_GEN))
    monkeypatch.setenv("FAKE_OOM_MODEL", largest["files"][pins["core_a"]["quants"][0]]["filename"])
    return {"oom_size": largest["size"], "oom_quant": pins["core_a"]["quants"][0]}


def make_bin(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("llama-bench", FAKE_BENCH), ("llama-completion", FAKE_COMPLETION)):
        f = bin_dir / name
        f.write_text(f"#!{sys.executable}\n{body}")
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def small_plan(platform="wsl2-cuda", mode="pilot", contexts=(2048, 8192)):
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        pins = tomllib.load(fh)
    pins["core_a"]["context_lengths"] = list(contexts)  # keep the test fast
    if mode == "formal":
        pins["freeze"]["mac"] = pins["freeze"]["nvidia"] = True
    return cp.build_plan(pins, platform=platform, mode=mode, seed=1, n_thr=6, poll=0)


class EnvRecorder:
    def __init__(self, commit=COMMIT, build=11146):
        self.calls = []
        self.commit, self.build = commit, build

    def __call__(self, llama_bin, out_path, gpu_lock_held):
        self.calls.append((Path(llama_bin).name, Path(out_path).name, gpu_lock_held))
        rec = {"runtime": {"llama_cpp": {"build_number": self.build, "commit": self.commit}}, "probe_errors": []}
        Path(out_path).write_text(json.dumps(rec))
        return rec


def run(tmp_path, plan, monkeypatch=None, env=None, **kw):
    out = tmp_path / "session"
    summary = rc.run_session(
        plan, bin_dir=make_bin(tmp_path), models_dir=tmp_path / "models", out_dir=out,
        capture_env=env or EnvRecorder(), sleep=lambda s: None, session_id="s1", **kw,
    )
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    return out, summary, rows


def test_layout_and_one_row_per_job(tmp_path):
    plan = small_plan()
    out, summary, rows = run(tmp_path, plan)
    assert len(rows) == len(plan["jobs"])
    assert [r["job_id"] for r in rows] == [j["id"] for j in plan["jobs"]]
    assert json.loads((out / "plan.json").read_text()) == plan
    session = json.loads((out / "session.json").read_text())
    assert session == summary
    assert session["schema_version"] == rc.SCHEMA_VERSION
    assert session["plan_sha256"] == rc.sha256_json(plan)
    for r in rows:
        for key in ("stdout", "stderr", "env_before", "env_after"):
            assert (out / r[key]).is_file(), (r["job_id"], key)


def test_outcomes_ok_and_oom(tmp_path, fake_model_env):
    _, summary, rows = run(tmp_path, small_plan())
    by = {(r["kind"], r["size"], r["quant"], r["L"]) for r in rows if r["outcome"] == "oom"}
    big = (fake_model_env["oom_size"], fake_model_env["oom_quant"], 8192)
    assert ("fit", *big) in by
    assert ("ttft", *big) in by
    oom = [r for r in rows if r["outcome"] == "oom"]
    assert all(r["stage"] == "context" for r in oom)
    ok_fit = next(r for r in rows if r["kind"] == "fit" and r["outcome"] == "ok")
    assert ok_fit["summary"]["prompt_tokens"] == ok_fit["L"] - 128
    assert ok_fit["summary"]["memory"]["devices"][0]["self"] == 11000
    ok_bench = next(r for r in rows if r["kind"] == "decode" and r["outcome"] == "ok")
    assert ok_bench["metrics"]["avg_ts"] == 50.0
    assert summary["outcomes"]["error"] == 0
    assert summary["outcomes"]["oom"] == len(oom)


def test_env_captured_before_and_after_every_job(tmp_path):
    env = EnvRecorder()
    plan = small_plan()
    run(tmp_path, plan, env=env, gpu_lock_held=True)
    # preflight + before/after per job
    assert len(env.calls) == 1 + 2 * len(plan["jobs"])
    assert env.calls[0][1] == "preflight.json"
    assert all(held is True for _, _, held in env.calls)


def test_preflight_rejects_wrong_build(tmp_path):
    with pytest.raises(rc.RunError, match="build"):
        run(tmp_path, small_plan(), env=EnvRecorder(commit="0" * 40))


def test_sentinel_drift_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SENTINEL_FILE", str(tmp_path / "sentinel.marker"))
    monkeypatch.setenv("FAKE_DRIFT", "0.05")
    _, summary, _ = run(tmp_path, small_plan())
    assert summary["sentinel"]["drift"] is True
    assert summary["sentinel"]["drift_pct"] == pytest.approx(5.0, abs=0.01)


def test_sentinel_no_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SENTINEL_FILE", str(tmp_path / "sentinel.marker"))
    monkeypatch.setenv("FAKE_DRIFT", "0.01")
    _, summary, _ = run(tmp_path, small_plan())
    assert summary["sentinel"]["drift"] is False


def test_idle_between_processes(tmp_path):
    slept = []
    plan = small_plan()
    rc.run_session(plan, bin_dir=make_bin(tmp_path), models_dir=tmp_path / "m", out_dir=tmp_path / "o",
                   capture_env=EnvRecorder(), sleep=slept.append, session_id="s")
    assert slept == [plan["params"]["idle_s"]] * (len(plan["jobs"]) - 1)


def test_prompt_file_has_exact_unit_count(tmp_path):
    out, _, rows = run(tmp_path, small_plan())
    fit = next(r for r in rows if r["kind"] == "fit" and r["L"] == 2048)
    prompt = Path(fit["argv"][fit["argv"].index("-f") + 1]).read_text()
    assert prompt == cp.PROMPT_UNIT * 1920


def test_formal_nvidia_requires_gpu_lock(tmp_path):
    with pytest.raises(rc.RunError, match="GPU lock"):
        run(tmp_path, small_plan(mode="formal"), gpu_lock_held=False)


def test_formal_mac_requires_mac_lock(tmp_path):
    with pytest.raises(rc.RunError, match="with-mac-lock"):
        run(tmp_path, small_plan(platform="macos-metal", mode="formal"), path_exists=lambda p: False)


def test_mac_run_refused_while_another_job_holds_the_lock(tmp_path):
    held_by_other = lambda p: p == rc.MAC_HEAVY_LOCK  # noqa: E731
    with pytest.raises(rc.RunError, match="held by another job"):
        run(tmp_path, small_plan(platform="macos-metal"), path_exists=held_by_other)


def test_formal_mac_under_our_lock_runs(tmp_path):
    # Under tools/with-mac-lock the lock dir exists because *we* hold it.
    _, summary, _ = run(tmp_path, small_plan(platform="macos-metal", mode="formal", contexts=(2048,)),
                        mac_lock_held=True, path_exists=lambda p: True)
    assert summary["mac_lock_held"] is True and summary["mode"] == "formal"


def test_refuses_existing_output_dir(tmp_path):
    plan = small_plan()
    run(tmp_path, plan)
    with pytest.raises(rc.RunError, match="exists"):
        rc.run_session(plan, bin_dir=tmp_path / "bin", models_dir=tmp_path / "models", out_dir=tmp_path / "session",
                       capture_env=EnvRecorder(), sleep=lambda s: None, session_id="s1")


# -------------------------------------------- schema v2: external peak memory --


class FakeSampler:
    def __init__(self, path):
        self.path, self.started = path, False

    def start(self):
        self.started = True

    def stop(self):
        assert self.started
        self.path.write_text("1000\n9500\n9800\n1200\n")
        return {"gpu_mem_used_peak_mib": 9800, "gpu_mem_used_first_mib": 1000, "n_samples": 4}


def test_cuda_fit_rows_record_sampled_peak(tmp_path):
    made = []

    def factory(path):
        made.append(FakeSampler(path))
        return made[-1]

    out, _, rows = run(tmp_path, small_plan(), gpu_sampler=factory)
    fit = [r for r in rows if r["kind"] == "fit"]
    assert len(made) == len(fit)
    for r in fit:
        assert r["external_memory"]["gpu_mem_used_peak_mib"] == 9800
        assert (out / r["external_memory"]["samples"]).is_file()
    assert all(r["external_memory"] is None for r in rows if r["kind"] != "fit")


@pytest.mark.skipif(sys.platform != "darwin", reason="/usr/bin/time -l is the macOS (BSD) variant")
def test_mac_fit_rows_record_time_l_footprint(tmp_path):
    out, summary, rows = run(tmp_path, small_plan(platform="macos-metal", contexts=(2048,)),
                             path_exists=lambda p: False)
    fit = [r for r in rows if r["kind"] == "fit"]
    assert fit and all(r["outcome"] == "ok" for r in fit), [r["reason"] for r in fit]
    for r in fit:
        assert r["argv"][:2] == ["/usr/bin/time", "-l"]
        assert r["external_memory"]["peak_footprint_bytes"] > 0
        assert r["external_memory"]["max_rss_bytes"] > 0
    assert summary["schema_version"] == 2


def test_nvidia_smi_sampler_parses_peak(tmp_path, monkeypatch):
    fake = tmp_path / "fakebin" / "nvidia-smi"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\nfor v in 1000 9500 9800 1200; do echo $v; done\nexec sleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:/usr/bin:/bin")
    sampler = rc.NvidiaSmiSampler(tmp_path / "gpumem.txt")
    sampler.start()
    import time
    deadline = time.time() + 5
    while time.time() < deadline and len((tmp_path / "gpumem.txt").read_text().split()) < 4:
        time.sleep(0.05)
    assert sampler.stop() == {"gpu_mem_used_peak_mib": 9800, "gpu_mem_used_first_mib": 1000, "n_samples": 4}


def test_nvidia_smi_sampler_without_nvidia_smi(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    sampler = rc.NvidiaSmiSampler(tmp_path / "gpumem.txt")
    sampler.start()
    assert sampler.stop() is None


def test_pilot_plan_runs_and_rows_carry_the_block(tmp_path):
    import core_a_pilot as pl

    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        pins = tomllib.load(fh)
    pins["core_a"]["context_lengths"] = [2048]  # keep the test fast
    plan = pl.build_pilot_plan(pins, platform="wsl2-cuda", sizes=["4B"], n_thr=6)
    _, summary, rows = run(tmp_path, plan)
    assert summary["outcomes"]["error"] == 0, [(r["job_id"], r["reason"]) for r in rows if r["outcome"] != "ok"]
    assert [r["block"] for r in rows] == [j["block"] for j in plan["jobs"]]
    assert pl.poll_decision(rows)["poll"] == 0


def test_session_writes_a_raw_manifest(tmp_path):
    """raw/ stays in the local archive (not git); the committed manifest pins every raw file."""
    import hashlib

    out, _, rows = run(tmp_path, small_plan())
    manifest = json.loads((out / "raw_manifest.json").read_text())
    raw_files = sorted(p.relative_to(out).as_posix() for p in (out / "raw").iterdir())
    assert sorted(manifest["files"]) == raw_files
    for rel, meta in manifest["files"].items():
        data = (out / rel).read_bytes()
        assert meta == {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def test_session_flags_processes_with_an_invalid_power_state(tmp_path):
    """Decision 012: off AC, battery < 20% or a thermal / performance warning invalidates the session."""
    class PowerEnv(EnvRecorder):
        def __init__(self):
            super().__init__()
            self.n = 0

        def __call__(self, llama_bin, out_path, gpu_lock_held):
            rec = super().__call__(llama_bin, out_path, gpu_lock_held)
            self.n += 1
            rec["power"] = {"source": "AC", "battery_pct": 15 if self.n == 6 else 80,
                            "thermal": {"thermal_warning_recorded": False, "performance_warning_recorded": False}}
            Path(out_path).write_text(json.dumps(rec))
            return rec

    out, summary, rows = run(tmp_path, small_plan(), env=PowerEnv())
    assert summary["env_valid"] is False
    assert summary["env_flags"][0]["reason"] == "battery 15% < 20%"
    (tmp_path / "b").mkdir()
    _, ok_summary, _ = run(tmp_path / "b", small_plan())
    assert ok_summary["env_valid"] is True and ok_summary["env_flags"] == []
