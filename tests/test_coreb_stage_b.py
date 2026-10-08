"""Tests for benchmarks/coreb_stage_b.py: the vLLM Stage B steps of decision 003 with the details
fixed in decision 017 (cold / warm boots -> JIT timing and K; VLLM_WSL2_ENABLE_PIN_MEMORY A/B),
written before any NVIDIA data."""

import json
import tomllib
from pathlib import Path

import pytest

import core_b_plan as bp
import coreb_stage_b as sb
from test_run_core_b import METRICS, World, result_json

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_pin_memory_choice_rule():
    """003: 1 only if it is >= 5 % faster at both C = 1 and C = 16 and every group's run-to-run CV < 3 %."""
    base = {(0, 1): [100, 101, 99], (0, 16): [1000, 1010, 990]}
    fast = {**base, (1, 1): [106, 107, 105], (1, 16): [1060, 1070, 1050]}
    res = sb.pin_memory_choice(fast)
    assert res["choice"] == 1 and res["gain_pct"][1] == pytest.approx(6.0) and res["gain_pct"][16] == pytest.approx(6.0)
    assert sb.pin_memory_choice({**fast, (1, 16): [1040, 1040, 1040]})["choice"] == 0  # +4 % at C = 16
    noisy = {**fast, (0, 1): [90, 100, 110]}
    assert sb.pin_memory_choice(noisy)["choice"] == 0  # CV 10 %
    short = {**fast, (1, 1): [106, 107]}
    res = sb.pin_memory_choice(short)
    assert res["choice"] == 0 and "incomplete" in res["reason"]


class BootWorld(World):
    """A World whose server logs report the available KV memory, faster on the warm boot."""

    def __init__(self, tmp_path, kv_gib=("6.10", "6.37")):
        super().__init__(tmp_path)
        self.kv_gib, self.boots = list(kv_gib), 0

    def popen(self, argv, env, log_path):
        proc = super().popen(argv, env, log_path)
        kv = self.kv_gib[min(self.boots, len(self.kv_gib) - 1)]
        Path(log_path).write_text(f"INFO [gpu_worker.py:640] Available KV cache memory: {kv} GiB\n")
        self.boot_started_at = self.t
        self.boots += 1
        return proc

    def http_get(self, url):
        if url.endswith("/health"):
            cold = self.boots == 1
            ready = self.t - self.boot_started_at >= (300 if cold else 60)
            return (200 if ready else 0), ""
        return super().http_get(url)


def test_boots_measure_cold_and_warm_startup_and_take_k_from_the_warm_boot(tmp_path, pins):
    w = BootWorld(tmp_path)
    rec = sb.run_boots(pins, out=tmp_path / "boots", v_total_mib=16303, b_mib=380, gdn_prefill_backend="triton",
                       popen=w.popen, signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get,
                       now=w.now, sleep=w.sleep, toolchain_check=lambda env: [], compute_apps=lambda: [])
    assert rec["gpu_memory_utilization"] == "0.94"
    assert rec["boots"][0]["startup_s"] >= 300 and rec["boots"][1]["startup_s"] < rec["boots"][0]["startup_s"]
    assert rec["cache_effective"] is True
    assert rec["kv_cache_memory_bytes"] == 6400 * 2**20  # warm boot 6.37 GiB, floored to 256 MiB
    argv = w.started[0]
    assert "--kv-cache-memory-bytes" not in argv and argv[argv.index("--gpu-memory-utilization") + 1] == "0.94"
    caches = {k: v for k, v in w.server_envs[0].items() if k in sb.CACHE_VARS}
    assert set(caches) == set(sb.CACHE_VARS) and all(v.startswith(str(tmp_path / "boots")) for v in caches.values())
    assert w.server_envs[1] == w.server_envs[0]  # the warm boot reuses the cold boot's caches


def test_boots_refuse_while_another_gpu_process_runs(tmp_path, pins):
    w = BootWorld(tmp_path)
    with pytest.raises(sb.CoreBStageBError, match="GPU process"):
        sb.run_boots(pins, out=tmp_path / "boots", v_total_mib=16303, b_mib=380, gdn_prefill_backend="triton",
                     popen=w.popen, signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get,
                     now=w.now, sleep=w.sleep, toolchain_check=lambda env: [],
                     compute_apps=lambda: [{"pid": 1, "name": "x", "used_mib": 9000}])
    assert w.started == []


def test_pin_ab_runs_both_settings_at_c1_and_c16_three_times(tmp_path, pins):
    w = World(tmp_path)
    rec = sb.run_pin_ab(pins, out=tmp_path / "ab", gpu_memory_utilization="0.94", kv_cache_memory_bytes=6400 * 2**20,
                        gdn_prefill_backend="triton", popen=w.popen, signal_group=w.signal_group,
                        group_alive=w.group_alive, http_get=w.http_get, run_bench=w.run_bench, now=w.now,
                        sleep=w.sleep, toolchain_check=lambda env: [], compute_apps=lambda: [])
    assert [e["VLLM_WSL2_ENABLE_PIN_MEMORY"] for e in w.server_envs] == ["0", "1"]
    assert sorted(w.bench_calls) == sorted([(2048, 1)] * 6 + [(2048, 16)] * 6)
    assert rec["cache_config"] == {"0": {"num_gpu_blocks": 160, "block_size": 528},
                                   "1": {"num_gpu_blocks": 160, "block_size": 528}}
    assert rec["decision"]["choice"] == 0  # identical fake throughput: no gain
    for argv in w.started:
        assert argv[argv.index("--kv-cache-memory-bytes") + 1] == str(6400 * 2**20)
    assert json.loads((tmp_path / "ab" / "result.json").read_text())["decision"]["choice"] == 0


def test_cli_requires_the_gpu_lock(tmp_path, capsys):
    assert sb.main(["boots", "--b-mib", "380", "--v-total-mib", "16303", "--gdn-prefill-backend", "triton",
                    "--out", str(tmp_path / "b")]) == 2
    assert "gpu-lock-held" in capsys.readouterr().err


def test_bench_check_runs_one_uniform_bench_on_the_smoke_server_and_finds_the_metrics(tmp_path, pins):
    """003 gate step 5: `vllm bench serve` completes and writes its JSON; /metrics exposes KV usage, preemptions
    and cache_config_info. Uses the 008 smoke server (a tool check, not a measurement)."""
    w = World(tmp_path)
    rec = sb.run_bench_check(pins, out=tmp_path / "bc", gdn_prefill_backend="triton", popen=w.popen,
                             signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get,
                             run_bench=w.run_bench, now=w.now, sleep=w.sleep, toolchain_check=lambda env: [],
                             compute_apps=lambda: [])
    argv = w.started[0]
    assert argv[argv.index("--max-num-seqs") + 1] == "8" and argv[argv.index("--max-model-len") + 1] == "2048"
    assert argv[2] == pins["core_b"]["repo"]  # served model name == the bench client's --model
    assert w.bench_calls == [(2048, 4)]
    assert rec["bench"]["ok"] is True and rec["metrics_found"] == {
        "vllm:kv_cache_usage_perc": True, "vllm:num_preemptions_total": True, "vllm:cache_config_info": True}
    assert rec["gate_step5_ok"] is True and w.signals  # the server was torn down
    assert json.loads((tmp_path / "bc" / "result.json").read_text())["gate_step5_ok"] is True



@pytest.mark.parametrize("entry", ["boots", "pin_ab", "bench_check"])
def test_a_signal_during_the_boot_wait_tears_the_server_down(tmp_path, pins, entry):
    """A SIGTERM / SIGHUP (SystemExit) or Ctrl-C while waiting for /health must not leave a server holding the
    GPU after the lock is released."""
    import signal as sig

    w = BootWorld(tmp_path)

    def http_get(url):
        if url.endswith("/health"):
            raise SystemExit(143)
        return w.http_get(url)

    common = dict(gdn_prefill_backend="triton", popen=w.popen, signal_group=w.signal_group,
                  group_alive=w.group_alive, http_get=http_get, now=w.now, sleep=w.sleep,
                  toolchain_check=lambda env: [], compute_apps=lambda: [])
    with pytest.raises(SystemExit):
        if entry == "boots":
            sb.run_boots(pins, out=tmp_path / "o", v_total_mib=16303, b_mib=380, **common)
        elif entry == "pin_ab":
            sb.run_pin_ab(pins, out=tmp_path / "o", gpu_memory_utilization="0.91", kv_cache_memory_bytes=2**30,
                          run_bench=w.run_bench, **common)
        else:
            sb.run_bench_check(pins, out=tmp_path / "o", run_bench=w.run_bench, **common)
    assert sig.SIGINT in w.signals and not w.group_alive(w.proc)


def test_boots_disclose_that_the_driver_jit_cache_is_not_reset(tmp_path, pins):
    w = BootWorld(tmp_path)
    rec = sb.run_boots(pins, out=tmp_path / "boots", v_total_mib=16303, b_mib=380, gdn_prefill_backend="triton",
                       popen=w.popen, signal_group=w.signal_group, group_alive=w.group_alive, http_get=w.http_get,
                       now=w.now, sleep=w.sleep, toolchain_check=lambda env: [], compute_apps=lambda: [])
    assert "CUDA_CACHE_PATH" in rec["driver_jit_cache"] and "CUDA_CACHE_PATH" not in w.server_envs[0]
