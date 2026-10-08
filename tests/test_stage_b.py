"""Tests for benchmarks/stage_b.py: the RTX Stage B measurements fixed in PREREGISTRATION §4 / §6
and decisions 003 / 004 / 009 (B, C_ctx, sysmem overflow), written before any NVIDIA data."""

import json

import pytest

import stage_b as sb


def test_parse_gpu_query():
    q = sb.parse_gpu_query("372, 16303, Enabled, Enabled\n")
    assert q == {"memory_used_mib": 372, "memory_total_mib": 16303, "display_active": "Enabled",
                 "display_mode": "Enabled"}
    with pytest.raises(sb.StageBError):
        sb.parse_gpu_query("[N/A], 16303, Enabled, Enabled\n")


class Gpu:
    def __init__(self, used, apps=(), apps_after=()):
        self.used, self.apps, self.apps_after, self.slept, self.calls = list(used), list(apps), list(apps_after), [], 0

    def query(self):
        self.calls += 1
        v = self.used.pop(0) if len(self.used) > 1 else self.used[0]
        return {"memory_used_mib": v, "memory_total_mib": 16303, "display_active": "Enabled", "display_mode": "Enabled"}

    def compute_apps(self):
        return self.apps if not self.slept else self.apps_after

    def sleep(self, s):
        self.slept.append(s)


def test_baseline_idles_five_minutes_then_takes_the_median_of_60_one_hertz_samples():
    """§4: idle 5 min after the last GPU process, 1 Hz x 60 s of memory.used, median = B."""
    gpu = Gpu([400] * 30 + [380] * 29 + [9000])
    res = sb.measure_baseline(query=gpu.query, compute_apps=gpu.compute_apps, sleep=gpu.sleep)
    assert gpu.slept[0] == 300 and gpu.slept[1:] == [1] * 59
    assert res["b_mib"] == 400 and len(res["samples_mib"]) == 60  # median is robust to one spike
    assert res["v_total_mib"] == 16303 and res["display_active"] == "Enabled" and res["idle_s"] == 300


def test_baseline_refuses_while_another_gpu_process_exists():
    gpu = Gpu([400], apps=[{"pid": 7944, "name": "python3", "used_mib": 9000}])
    with pytest.raises(sb.StageBError, match="GPU process"):
        sb.measure_baseline(query=gpu.query, compute_apps=gpu.compute_apps, sleep=gpu.sleep)
    assert gpu.slept == []
    gpu = Gpu([400], apps_after=[{"pid": 1, "name": "x", "used_mib": 100}])
    with pytest.raises(sb.StageBError, match="during"):
        sb.measure_baseline(query=gpu.query, compute_apps=gpu.compute_apps, sleep=gpu.sleep)


def test_parse_compute_apps():
    text = "7944, /home/u/.venv/bin/python3, 9000\n"
    assert sb.parse_compute_apps(text) == [{"pid": 7944, "name": "/home/u/.venv/bin/python3", "used_mib": 9000}]
    assert sb.parse_compute_apps("") == []
    assert sb.parse_compute_apps("7944, [Not Found], [N/A]\n") == [{"pid": 7944, "name": "[Not Found]",
                                                                    "used_mib": None}]


def summary(self_mib):
    return {"memory": {"unit": "MiB", "devices": [{"name": "CUDA0", "self": self_mib}], "host": {"self": 300},
                       "other": []}}


def test_c_ctx_is_peak_minus_b_minus_device_self():
    """004 / 009: C_ctx = peak memory.used during the 4B-Q4_K_M @ 2048 fit run - B - that run's device self."""
    assert sb.device_self_mib(summary(3000)) == 3000
    assert sb.device_self_mib({"memory": None}) is None
    assert sb.c_ctx_mib([1000, 3700, 3900, 3800], b_mib=400, self_mib=3000) == 500


def test_cctx_job_is_the_pinned_fit_command_for_the_anchor():
    job = sb.cctx_job(pins={"core_a": {"models": [
        {"size": "4B", "files": {"Q4_K_M": {"filename": "Qwen3.5-4B-Q4_K_M.gguf"}}},
        {"size": "9B", "files": {"Q8_0": {"filename": "Qwen3.5-9B-Q8_0.gguf"}}}]}}, n_thr=8, poll=0)
    a = dict(zip(job["argv"], job["argv"][1:]))
    assert a["-m"] == "{models}/Qwen3.5-4B-Q4_K_M.gguf" and a["-c"] == "2048" and a["--fit"] == "off"
    assert job["prompt_tokens"] == 2048 - 128


def test_overflow_job_uses_9b_q8_0_at_262144_with_a_short_prompt():
    """008: 9B-Q8_0 with -c 262144 (outside the 18 cells) exceeds 16 GB for any B; the KV is allocated at
    context creation, so a short prompt shows hard OOM vs sysmem spill."""
    job = sb.overflow_job(pins={"core_a": {"models": [
        {"size": "4B", "files": {"Q4_K_M": {"filename": "a.gguf"}}},
        {"size": "9B", "files": {"Q8_0": {"filename": "Qwen3.5-9B-Q8_0.gguf"}}}]}}, n_thr=8, poll=0)
    a = dict(zip(job["argv"], job["argv"][1:]))
    assert a["-m"] == "{models}/Qwen3.5-9B-Q8_0.gguf" and a["-c"] == "262144"
    assert job["prompt_tokens"] == sb.OVERFLOW_PROMPT_TOKENS == 1024


@pytest.mark.parametrize("result, self_mib, want", [
    ({"outcome": "oom"}, None, "hard-oom"),
    ({"outcome": "ok"}, 17000, "spill"),       # 17000 + 400 > 16303: the driver fell back to system memory
    ({"outcome": "ok"}, 15000, "fit"),         # would contradict N ~ 15.9 GiB; reported, not assumed
    ({"outcome": "error"}, None, "error"),
])
def test_classify_overflow(result, self_mib, want):
    assert sb.classify_overflow(result, self_mib=self_mib, v_total_mib=16303, b_mib=400) == want


def test_cli_refuses_without_the_gpu_lock(tmp_path, capsys):
    assert sb.main(["baseline", "--out", str(tmp_path / "b.json")]) == 2
    assert "gpu-lock-held" in capsys.readouterr().err
    assert not (tmp_path / "b.json").exists()


def test_cli_baseline_writes_json(tmp_path, monkeypatch):
    gpu = Gpu([400])
    monkeypatch.setattr(sb, "nvidia_query", gpu.query)
    monkeypatch.setattr(sb, "nvidia_compute_apps", gpu.compute_apps)
    monkeypatch.setattr(sb.time, "sleep", gpu.sleep)
    monkeypatch.setattr(sb, "windows_gpu_settings", lambda: {"hags": "2", "wslconfig": None})
    assert sb.main(["baseline", "--gpu-lock-held", "--out", str(tmp_path / "b.json")]) == 0
    rec = json.loads((tmp_path / "b.json").read_text())
    assert rec["b_mib"] == 400 and rec["windows"] == {"hags": "2", "wslconfig": None}
    assert sb.main(["baseline", "--gpu-lock-held", "--out", str(tmp_path / "b.json")]) == 2  # never overwrites


class FakeSampler:
    def __init__(self, path):
        self.path = path

    def start(self):
        self.path.write_text("380\n3600\n11700\n11650\n")

    def stop(self):
        return None


def test_run_fit_job_end_to_end_with_the_fake_llama_completion(tmp_path, monkeypatch):
    from test_run_core_a import EnvRecorder, make_bin

    monkeypatch.setenv("FAKE_OOM_MODEL", "Qwen3.5-9B-Q8_0.gguf")
    pins = {"core_a": {"models": [{"size": "4B", "files": {"Q4_K_M": {"filename": "Qwen3.5-4B-Q4_K_M.gguf"}}},
                                  {"size": "9B", "files": {"Q8_0": {"filename": "Qwen3.5-9B-Q8_0.gguf"}}}]}}
    bin_dir = str(make_bin(tmp_path))
    rec = sb.run_fit_job(sb.cctx_job(pins=pins, n_thr=8, poll=0), out=tmp_path / "cctx", bin_dir=bin_dir,
                         models_dir=str(tmp_path / "m"), capture_env=EnvRecorder(), sampler_cls=FakeSampler)
    assert rec["outcome"] == "ok" and rec["samples_mib"] == [380, 3600, 11700, 11650]
    assert sb.c_ctx_mib(rec["samples_mib"], b_mib=380, self_mib=sb.device_self_mib(rec["summary"])) == 320
    assert (tmp_path / "cctx" / "env.before.json").exists() and (tmp_path / "cctx" / "env.after.json").exists()
    rec = sb.run_fit_job(sb.overflow_job(pins=pins, n_thr=8, poll=0), out=tmp_path / "ovf", bin_dir=bin_dir,
                         models_dir=str(tmp_path / "m"), capture_env=EnvRecorder(), sampler_cls=FakeSampler)
    assert sb.classify_overflow(rec, self_mib=None, v_total_mib=16303, b_mib=380) == "hard-oom"
