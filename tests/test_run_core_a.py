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
        pins["stage_b"]["rtx_poll"] = 0
        return cp.build_plan(pins, platform=platform, mode=mode, seed=20261008,
                             n_thr=8 if platform == "macos-metal" else pins["stage_b"]["rtx_n_thr"], poll=0)
    return cp.build_plan(pins, platform=platform, mode=mode, seed=1, n_thr=6, poll=0)


class EnvRecorder:
    def __init__(self, commit=COMMIT, build=11146):
        self.calls = []
        self.commit, self.build = commit, build

    def __call__(self, llama_bin, out_path, gpu_lock_held):
        self.calls.append((Path(llama_bin).name, Path(out_path).name, gpu_lock_held))
        rec = {"runtime": {"llama_cpp": {"build_number": self.build, "commit": self.commit}}, "probe_errors": [],
               "power": dict(GOOD_POWER), "hardware": {"iogpu_wired_limit_mb": 0},
               "gpu": {"nvidia": [{"enforced_power_limit_w": 300.0, "power_default_limit_w": 300.0}]}}
        Path(out_path).write_text(json.dumps(rec))
        return rec


APPLE_70W = {"wattage_w": 68, "manufacturer": "Apple Inc.", "name": "70W USB-C Power Adapter"}
GOOD_POWER = {"source": "AC", "battery_pct": 80, "battery_state": "AC attached; not charging", "powermode": 0,
              "ac_adapter": APPLE_70W,
              "thermal": {"thermal_warning_recorded": False, "performance_warning_recorded": False}}


def run(tmp_path, plan, monkeypatch=None, env=None, **kw):
    out = tmp_path / "session"
    kw.setdefault("read_power", lambda: {"source": "AC", "battery_pct": 100})
    kw.setdefault("sleep", lambda s: None)
    summary = rc.run_session(
        plan, bin_dir=make_bin(tmp_path), models_dir=tmp_path / "models", out_dir=out,
        capture_env=env or EnvRecorder(), session_id="s1", **kw,
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
            rec["power"] = {**GOOD_POWER, "battery_pct": 15 if self.n == 6 else 80}
            Path(out_path).write_text(json.dumps(rec))
            return rec

    mac = dict(mac_lock_held=True, path_exists=lambda p: True)
    out, summary, rows = run(tmp_path, small_plan(platform="macos-metal"), env=PowerEnv(), **mac)
    assert summary["env_valid"] is False
    assert summary["env_flags"][0]["reason"] == "battery 15% < 20%"
    (tmp_path / "b").mkdir()
    _, ok_summary, _ = run(tmp_path / "b", small_plan(platform="macos-metal"), **mac)
    assert ok_summary["env_valid"] is True and ok_summary["env_flags"] == []


@pytest.mark.parametrize("rec, want", [
    ({"power": GOOD_POWER, "hardware": {"iogpu_wired_limit_mb": 0}}, []),
    ({"probe_errors": ["env_capture failed (exit 1): boom"]}, ["power source unknown (AC required)",
                                                               "thermal state unknown"]),
    ({"power": {**GOOD_POWER, "source": None}}, ["power source unknown (AC required)"]),
    ({"power": {**GOOD_POWER, "thermal": None}}, ["thermal state unknown"]),
    ({"power": {"source": None, "battery_pct": 15,
                "thermal": {"thermal_warning_recorded": True, "performance_warning_recorded": False}}},
     ["power source unknown (AC required)", "battery 15% < 20%", "thermal / performance warning recorded"]),
    ({"power": {**GOOD_POWER, "source": "Battery"}}, ["power source Battery (AC required)"]),
    ({"power": {**GOOD_POWER, "battery_pct": None}, "hardware": {"iogpu_wired_limit_mb": 0}}, []),  # no battery
])
def test_env_problems_on_the_mac_treat_unknown_as_invalid(rec, want):
    """012 / 013: a failed or partial probe cannot prove AC power or a clean thermal log (the §4 powermode /
    wired-limit rules are tested separately)."""
    config = rc.mac_config_problems(rec)
    assert [x for x in rc.env_problems(rec, "macos-metal") if x not in config] == want


def test_env_problems_on_the_rtx_check_the_power_limit_not_a_battery():
    """§4 RTX: enforced.power.limit must equal power.default_limit (unknown = invalid, 013); no battery rules."""
    gpu = lambda e, d: {"gpu": {"nvidia": [{"enforced_power_limit_w": e, "power_default_limit_w": d}]},  # noqa: E731
                        "power": {"source": None, "thermal": None}}
    assert rc.env_problems(gpu(300.0, 300.0), "wsl2-cuda") == []
    assert rc.env_problems(gpu(250.0, 300.0), "wsl2-cuda") == ["enforced power limit 250.0 W != default 300.0 W (§4)"]
    assert rc.env_problems(gpu(None, 300.0), "wsl2-cuda") == ["power limit unknown (§4)"]
    assert rc.env_problems({"probe_errors": ["x"]}, "wsl2-cuda") == ["nvidia-smi power limits unknown (§4)"]


def test_mac_session_with_a_failed_env_probe_is_env_invalid(tmp_path):
    class Broken(EnvRecorder):
        def __call__(self, llama_bin, out_path, gpu_lock_held):
            rec = super().__call__(llama_bin, out_path, gpu_lock_held)
            rec.update({"power": {"source": None, "battery_pct": None, "thermal": None},
                        "probe_errors": ["pmset batt: timed out", "pmset therm: timed out"]})
            Path(out_path).write_text(json.dumps(rec))
            return rec

    _, summary, _ = run(tmp_path, small_plan(platform="macos-metal", contexts=(2048,)), env=Broken(),
                        mac_lock_held=True, path_exists=lambda p: True)
    assert summary["env_valid"] is False


class Battery:
    """Fake pmset: returns the given battery percentages in order, then the last one forever."""

    def __init__(self, *pcts, source="AC"):
        self.pcts, self.source, self.calls = list(pcts), source, 0

    def __call__(self):
        self.calls += 1
        pct = self.pcts.pop(0) if len(self.pcts) > 1 else self.pcts[0]
        return {"source": self.source, "battery_pct": pct}


def test_charge_gate_idles_until_the_resume_level():
    """016: on AC below 40%, idle (charging) until >= 90% before the next process."""
    slept = []
    pause = rc.charge_gate(Battery(35, 50, 70, 91), slept.append)
    assert pause == {"from_pct": 35, "to_pct": 91, "waited_s": 180, "timed_out": False}
    assert slept == [rc.CHARGE_POLL_S] * 3


def test_charge_gate_does_nothing_when_charged_off_ac_or_unknown():
    for power in (Battery(40), Battery(95), Battery(10, source="Battery")):
        slept = []
        assert rc.charge_gate(power, slept.append) is None and slept == []
    assert rc.charge_gate(lambda: {"source": None, "battery_pct": None}, lambda s: None) is None


def test_charge_gate_gives_up_after_the_max_wait():
    pause = rc.charge_gate(Battery(30), lambda s: None)
    assert pause["timed_out"] is True and pause["waited_s"] == rc.CHARGE_MAX_WAIT_S and pause["to_pct"] == 30


def test_parse_pmset_batt_matches_env_capture():
    import env_capture as ec

    text = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t37%; charging; 1:02 remaining present: true\n"
    shared = {k: v for k, v in ec.parse_pmset_batt(text).items() if k in ("source", "battery_pct")}
    assert rc.parse_pmset_batt(text) == shared == {"source": "AC", "battery_pct": 37}


def test_mac_session_pauses_to_charge_and_records_it(tmp_path):
    power = Battery(100, 35, 60, 92, 92)
    _, summary, rows = run(tmp_path, small_plan(platform="macos-metal", contexts=(2048,)), read_power=power,
                           mac_lock_held=True, path_exists=lambda p: True)
    (pause,) = summary["charge_pauses"]
    assert pause["job_id"] == rows[1]["job_id"] and pause["from_pct"] == 35 and pause["to_pct"] == 92
    assert summary["env_valid"] is True


def test_rtx_session_never_reads_the_battery(tmp_path):
    power = Battery(10)
    _, summary, _ = run(tmp_path, small_plan(), read_power=power)
    assert power.calls == 0 and summary["charge_pauses"] == []



def test_mac_config_rules_powermode_adapter_and_wired_limit():
    """019 (§4 had powermode 2): every formal Mac run in powermode 0 (automatic) on the Apple 70 W adapter, with
    iogpu.wired_limit_mb = 0 (unknown = invalid, 013)."""
    ok = {"power": GOOD_POWER, "hardware": {"iogpu_wired_limit_mb": 0}}
    assert rc.mac_config_problems(ok) == []
    assert rc.mac_config_problems({**ok, "power": {**GOOD_POWER, "powermode": 2}}) == ["powermode 2 (019 requires 0)"]
    assert rc.mac_config_problems({**ok, "power": {**GOOD_POWER, "powermode": None}}) == \
        ["powermode None (019 requires 0)"]
    assert rc.mac_config_problems({**ok, "hardware": {"iogpu_wired_limit_mb": 8192}}) == \
        ["iogpu.wired_limit_mb 8192 (§4 requires 0)"]
    assert "powermode 2 (019 requires 0)" in rc.env_problems({**ok, "power": {**GOOD_POWER, "powermode": 2}},
                                                              "macos-metal")
    for adapter in (None, {**APPLE_70W, "name": "96W USB-C Power Adapter"}, {**APPLE_70W, "manufacturer": "Acme"},
                    {"wattage_w": None, "manufacturer": None, "name": None}):
        (why,) = rc.mac_config_problems({**ok, "power": {**GOOD_POWER, "ac_adapter": adapter}})
        assert why.startswith("AC adapter") and "019" in why
    # the negotiated wattage is recorded, not a condition
    assert rc.mac_config_problems({**ok, "power": {**GOOD_POWER, "ac_adapter": {**APPLE_70W, "wattage_w": 70}}}) == []


def test_formal_mac_run_refuses_a_preflight_in_the_wrong_powermode(tmp_path):
    class HighPower(EnvRecorder):
        def __call__(self, llama_bin, out_path, gpu_lock_held):
            rec = super().__call__(llama_bin, out_path, gpu_lock_held)
            rec["power"] = {**GOOD_POWER, "powermode": 2}
            Path(out_path).write_text(json.dumps(rec))
            return rec

    with pytest.raises(rc.RunError, match="powermode"):
        run(tmp_path, small_plan(platform="macos-metal", mode="formal", contexts=(2048,)), env=HighPower(),
            mac_lock_held=True, path_exists=lambda p: True)


# ----------------------------------------------------------- 019 battery drain --


def test_battery_drain_watch_needs_three_points_within_thirty_minutes():
    w = rc.BatteryDrainWatch()
    assert (rc.DRAIN_WINDOW_S, rc.DRAIN_STOP_PTS) == (1800, 3)
    assert [w.observe(t, p) for t, p in ((0, 100), (600, 99), (1200, 98))] == [None] * 3
    trip = w.observe(1500, 97, job_id="j", phase="after")
    assert trip == {"rule": "019", "job_id": "j", "phase": "after", "pct": 97, "window_max_pct": 100, "drop_pts": 3,
                    "window_s": 1800}


def test_battery_drain_watch_ignores_quantisation_slow_drains_and_unknown_readings():
    w = rc.BatteryDrainWatch()
    for t, pct in enumerate((80, 79, 80, 79, 78, 79, 80, None, 78)):  # +-1 point jitter, never 3 below the max
        assert w.observe(t * 60, pct) is None
    slow = rc.BatteryDrainWatch()  # 2 points per 30 min (4 %/h) never trips; 016 handles slow drains
    assert all(slow.observe(t * 900, 100 - t) is None for t in range(40))


def test_battery_drain_watch_window_is_inclusive_and_slides():
    w = rc.BatteryDrainWatch()
    assert w.observe(0, 100) is None and w.observe(1000, 98) is None
    assert w.observe(1801, 97) is None  # the 100 at t = 0 left the window; 98 - 97 = 1
    w2 = rc.BatteryDrainWatch()
    assert w2.observe(0, 100) is None and w2.observe(1800, 97)["drop_pts"] == 3  # exactly 30 min apart: inside


class DrainingEnv(EnvRecorder):
    """Battery readings in capture order (then the last forever); each capture advances the fake clock."""

    def __init__(self, pcts, clock, step_s=60):
        super().__init__()
        self.pcts, self.clock, self.step_s = list(pcts), clock, step_s

    def __call__(self, llama_bin, out_path, gpu_lock_held):
        rec = super().__call__(llama_bin, out_path, gpu_lock_held)
        self.clock.t += self.step_s
        pct = self.pcts.pop(0) if len(self.pcts) > 1 else self.pcts[0]
        rec["power"] = {**GOOD_POWER, "battery_pct": pct}
        Path(out_path).write_text(json.dumps(rec))
        return rec


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def formal_mac(tmp_path, pcts, **kw):
    clock = Clock()
    env = DrainingEnv(pcts, clock)
    plan = small_plan(platform="macos-metal", mode="formal", contexts=(2048,))
    out, summary, rows = run(tmp_path, plan, env=env, clock=clock, mac_lock_held=True, path_exists=lambda p: True, **kw)
    return plan, summary, rows


def test_formal_mac_session_stops_after_the_process_whose_after_reading_trips(tmp_path):
    # captures: preflight, then before / after per job; job 0 after = 97 vs 100 -> stop with job 0's row written
    plan, summary, rows = formal_mac(tmp_path, [100, 100, 97])
    assert [r["job_id"] for r in rows] == [plan["jobs"][0]["id"]]
    drain = summary["battery_drain"]
    assert drain["job_id"] == plan["jobs"][0]["id"] and drain["phase"] == "after" and drain["drop_pts"] == 3
    assert summary["env_valid"] is False
    assert any("battery drain" in f["reason"] and "019" in f["reason"] for f in summary["env_flags"])
    assert summary["sentinel"]["end_tok_s"] is None  # the end sentinel never ran


def test_formal_mac_session_skips_the_process_whose_before_reading_trips(tmp_path):
    plan, summary, rows = formal_mac(tmp_path, [100, 100, 99, 97])
    assert [r["job_id"] for r in rows] == [plan["jobs"][0]["id"]]  # job 1 never ran
    assert summary["battery_drain"]["job_id"] == plan["jobs"][1]["id"]
    assert summary["battery_drain"]["phase"] == "before" and summary["env_valid"] is False


def test_formal_mac_session_without_drain_runs_to_the_end(tmp_path):
    plan, summary, rows = formal_mac(tmp_path, [100, 99, 100, 98, 99])
    assert len(rows) == len(plan["jobs"]) and summary["battery_drain"] is None and summary["env_valid"] is True


def test_battery_drain_clock_excludes_charge_pauses(tmp_path):
    """019: the 30-minute window runs on session time minus 016 pauses. A pause that the fake sleep makes last
    3600 s would otherwise push the 50 % readings out of the window."""
    clock = Clock()

    def sleep(s):
        clock.t += s

    env = DrainingEnv([50, 50, 50, 47], clock, step_s=10)
    plan = small_plan(platform="macos-metal", mode="formal", contexts=(2048,))
    power = Battery(100, 35, *([36] * 59), 95)  # job 0: no pause; job 1: 60 polls x 60 s = 3600 s
    _, summary, rows = run(tmp_path, plan, env=env, clock=clock, sleep=sleep, read_power=power, mac_lock_held=True,
                           path_exists=lambda p: True)
    assert summary["charge_pauses"][0]["waited_s"] == 3600
    assert summary["battery_drain"]["phase"] == "before" and summary["battery_drain"]["window_max_pct"] == 50


def test_pilot_mac_sessions_record_but_never_stop_on_a_drain(tmp_path):
    clock = Clock()
    env = DrainingEnv([100, 90, 80], clock)
    plan = small_plan(platform="macos-metal", contexts=(2048,))
    _, summary, rows = run(tmp_path, plan, env=env, clock=clock, mac_lock_held=True, path_exists=lambda p: True)
    assert len(rows) == len(plan["jobs"]) and summary["battery_drain"] is None


def test_formal_rtx_session_remeasures_b_and_records_drift(tmp_path):
    """§4 step 4: B is re-measured at the start of every formal RTX session; > 0.25 GiB over the frozen B = drift."""
    calls = []

    def measure_b(frozen_b_mib):
        calls.append(frozen_b_mib)
        return {"b_mib": frozen_b_mib + 300, "frozen_b_mib": frozen_b_mib, "drift": True}

    plan = small_plan(mode="formal", contexts=(2048,))
    _, summary, _ = run(tmp_path, plan, gpu_lock_held=True, measure_b=measure_b)
    assert calls == [plan["expect"]["frozen_b_mib"]] and summary["session_b"]["drift"] is True
    (tmp_path / "x").mkdir()
    with pytest.raises(rc.RunError, match="session-start B"):
        run(tmp_path / "x", small_plan(mode="formal", contexts=(2048,)), gpu_lock_held=True)



def test_main_wires_the_session_start_b_for_formal_rtx_plans(tmp_path, monkeypatch):
    import stage_b

    seen = {}

    def fake_run_session(plan, **kw):
        seen.update(kw)
        raise rc.RunError("stop here")

    monkeypatch.setattr(rc, "run_session", fake_run_session)
    monkeypatch.setattr(stage_b, "measure_baseline", lambda **kw: {"b_mib": 1, "drift": False, **kw})
    plan = small_plan(mode="formal", contexts=(2048,))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    assert rc.main(["--plan", str(tmp_path / "plan.json"), "--bin", "b", "--models", "m", "--out", str(tmp_path / "o"),
                    "--session-id", "s", "--gpu-lock-held"]) == 2
    assert seen["measure_b"] is not None and seen["measure_b"](811)["frozen_b_mib"] == 811
