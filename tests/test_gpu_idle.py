"""Tests for benchmarks/gpu_idle.py: decision 031's GPU idle criterion (a 30 s window at 1 Hz: utilization, no compute
process, a stable baseline, the VRAM headroom bound) and its spill guard (Windows shared usage of the NVIDIA adapter,
read from WSL through powershell.exe), with fake nvidia-smi / powershell output and a fake clock."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import gpu_idle as gi

ROOT = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        assert s >= 0, f"sleep({s})"
        self.sleeps.append(s)
        self.t += s


def sample(mem=1385, util=0, pstate="P8"):
    return {"mem_used_mib": mem, "util_pct": util, "pstate": pstate}


def samples(mems, util=0):
    return [sample(m, util) for m in mems]


# ------------------------------------------------------------------ constants (031) --

def test_constants_are_the_031_values():
    assert (gi.F_PEAK_MIB, gi.MARGIN_MIB, gi.UTIL_MAX, gi.RANGE_MAX_MIB, gi.SPILL_TOL_MIB, gi.WINDOWS) == (
        15157, 128, 5, 64, 64, 3)
    assert (gi.WINDOW_SAMPLES, gi.WINDOW_INTERVAL_S, gi.SHARED_USAGE_TIMEOUT_S) == (30, 1.0, 20)


def test_headroom_bound_is_computed_from_v_total():
    import tomllib

    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    assert pins["stage_b"]["v_total_mib"] == 16303
    assert gi.headroom_bound(pins["stage_b"]["v_total_mib"]) == 16303 - 15157 - 128 == 1018
    assert gi.headroom_bound(17000) == 17000 - 15157 - 128
    for bad in (None, "16303", 16303.0, True, 15000):
        with pytest.raises(ValueError):
            gi.headroom_bound(bad)


def test_criterion_records_every_number_of_the_rule():
    c = gi.criterion(16303)
    assert c == {"decision": "031", "v_total_mib": 16303, "f_peak_mib": 15157, "margin_mib": 128, "bound_mib": 1018,
                 "util_max_pct": 5, "range_max_mib": 64, "windows": 3, "window_samples": 30, "interval_s": 1.0,
                 "spill_tol_mib": 64}


# ------------------------------------------------------------------ nvidia-smi parsing --

def test_parse_gpu_sample_reads_memory_utilization_and_pstate():
    assert gi.parse_gpu_sample("1385, 0, P8\n") == {"mem_used_mib": 1385, "util_pct": 0, "pstate": "P8"}
    assert gi.parse_gpu_sample(" 15640 , 100 , P1 ") == {"mem_used_mib": 15640, "util_pct": 100, "pstate": "P1"}
    na = gi.parse_gpu_sample("1385, [N/A], [N/A]")
    assert na == {"mem_used_mib": 1385, "util_pct": None, "pstate": None}


@pytest.mark.parametrize("text", ["", "Failed to initialize NVML", "1385, 0", "1385, 0, P8\n1400, 0, P8",
                                  "[N/A], 0, P8"])
def test_parse_gpu_sample_refuses_anything_but_one_gpu_row_with_a_memory_value(text):
    with pytest.raises(ValueError):
        gi.parse_gpu_sample(text)


def test_parse_compute_apps():
    assert gi.parse_compute_apps("") == []
    assert gi.parse_compute_apps("No running processes found\n") == []
    assert gi.parse_compute_apps("123, python, 9000\n\n456, vllm, 100\n") == ["123, python, 9000", "456, vllm, 100"]


class Done:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def test_nvidia_sample_and_compute_apps_bound_nvidia_smi():
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return Done("1385, 3, P8\n" if "--query-gpu=memory.used,utilization.gpu,pstate" in argv else "1, x, 2\n")

    assert gi.nvidia_sample(run=run) == {"mem_used_mib": 1385, "util_pct": 3, "pstate": "P8"}
    assert gi.nvidia_compute_apps(run=run, timeout_s=3) == ["1, x, 2"]
    assert calls[0][0] == ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,pstate",
                           "--format=csv,noheader,nounits"]
    assert calls[1][0] == ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader,nounits"]
    assert calls[0][1]["timeout"] == gi.NVIDIA_SMI_TIMEOUT_S and calls[1][1]["timeout"] == 3
    assert all(kw["check"] is True for _, kw in calls)


# ------------------------------------------------------------------ the window (031) --

def test_idle_window_takes_n_samples_at_the_interval():
    clock, seen = Clock(), []

    def fn():
        seen.append(clock.t)
        return sample()

    got = gi.idle_window(fn, n=30, interval_s=1.0, now=clock.now, sleep=clock.sleep)
    assert len(got) == 30 and seen == [float(i) for i in range(30)]


def test_idle_window_does_not_drift_when_a_sample_is_slow():
    clock, seen = Clock(), []

    def slow():
        seen.append(clock.t)
        clock.t += 0.4                         # nvidia-smi takes 0.4 s
        return sample()

    gi.idle_window(slow, n=4, interval_s=1.0, now=clock.now, sleep=clock.sleep)
    assert seen == [0.0, 1.0, 2.0, 3.0]
    clock2, seen2 = Clock(), []

    def slower():
        seen2.append(clock2.t)
        clock2.t += 1.5                        # slower than the interval: the next sample starts at once
        return sample()

    gi.idle_window(slower, n=3, interval_s=1.0, now=clock2.now, sleep=clock2.sleep)
    assert seen2 == [0.0, 1.5, 3.0] and all(s == 0 for s in clock2.sleeps)


def test_an_idle_window_passes():
    assert gi.idle_problems(samples([1000] * 15 + [1018] * 15, util=5), [], bound_mib=1018) == []


@pytest.mark.parametrize("window, apps, match", [
    (samples([400] * 29) + [sample(400, 6)], [], "utilization.gpu"),            # > 5 % in one sample
    (samples([400] * 30), ["123, python, 9000"], "compute process"),
    (samples([400] * 29 + [465]), [], "range"),                                  # max - min 65 > 64
    (samples([1019] * 30), [], "headroom"),                                      # idle > bound
    (samples([1000] * 29 + [1019]), [], "headroom"),                             # the max sample is the idle value
    ([sample(400, None)] + samples([400] * 29), [], "utilization.gpu"),          # not reported: fails closed
    ([sample(None)] + samples([400] * 29), [], "memory.used"),
    ([], [], "no nvidia-smi sample"),
])
def test_each_031_condition_fails_the_window(window, apps, match):
    problems = gi.idle_problems(window, apps, bound_mib=1018)
    assert problems and any(match in p for p in problems), problems


def test_a_range_of_exactly_64_and_util_of_exactly_5_pass():
    assert gi.idle_problems(samples([900] * 29 + [964], util=5), [], bound_mib=1018) == []


def test_every_failing_condition_is_reported():
    window = samples([1100] * 29 + [1200], util=50)
    problems = gi.idle_problems(window, ["1, x, 2"], bound_mib=1018)
    assert len(problems) == 4


def test_window_report_summarises_the_samples():
    rep = gi.window_report(samples([1385] * 29 + [1393], util=1), [], bound_mib=1018, attempt=2)
    assert {k: rep[k] for k in ("attempt", "n", "bound_mib", "mem_min_mib", "mem_max_mib", "mem_range_mib",
                                "util_max_pct", "pstates", "compute_apps", "idle_mib")} == {
        "attempt": 2, "n": 30, "bound_mib": 1018, "mem_min_mib": 1385, "mem_max_mib": 1393, "mem_range_mib": 8,
        "util_max_pct": 1, "pstates": ["P8"], "compute_apps": [], "idle_mib": 1393}
    assert rep["mem_used_mib"] == [1385] * 29 + [1393] and rep["util_pct"] == [1] * 30
    assert rep["ok"] is False and any("headroom" in p for p in rep["problems"])


# ------------------------------------------------------------------ wait_idle: up to 3 windows --

def counting_apps(apps=()):
    calls = []

    def fn():
        calls.append(1)
        return list(apps)

    return fn, calls


def test_wait_idle_passes_on_the_first_idle_window():
    clock = Clock()
    apps, calls = counting_apps()
    ok, rep = gi.wait_idle(lambda: sample(600), apps, bound_mib=1018, now=clock.now, sleep=clock.sleep)
    assert ok is True and rep["attempt"] == 1 and rep["problems"] == [] and rep["earlier"] == []
    assert len(calls) == 1 and clock.t == pytest.approx(29.0)


def test_wait_idle_retries_and_passes_on_a_later_window():
    clock = Clock()
    mems = iter([3000] * 30 + [600] * 30)
    apps, calls = counting_apps()
    ok, rep = gi.wait_idle(lambda: sample(next(mems)), apps, bound_mib=1018, now=clock.now, sleep=clock.sleep)
    assert ok is True and rep["attempt"] == 2 and len(rep["earlier"]) == 1
    assert any("headroom" in p for p in rep["earlier"][0])


def test_wait_idle_gives_up_after_3_windows_with_the_last_report():
    clock = Clock()
    n = []

    def busy():
        n.append(1)
        return sample(1385)

    apps, calls = counting_apps()
    ok, rep = gi.wait_idle(busy, apps, bound_mib=1018, now=clock.now, sleep=clock.sleep)
    assert ok is False and rep["attempt"] == 3 and len(rep["earlier"]) == 2 and len(n) == 90 and len(calls) == 3
    assert any("1385" in p and "1018" in p for p in rep["problems"])
    assert clock.t < 3 * 30 + 3                     # about 90 s, never longer


def test_wait_idle_lets_nvidia_smi_failures_through():
    def broken():
        raise subprocess.TimeoutExpired("nvidia-smi", 15)

    with pytest.raises(subprocess.TimeoutExpired):
        gi.wait_idle(broken, lambda: [], bound_mib=1018, now=Clock().now, sleep=Clock().sleep)


# ------------------------------------------------------------------ shared usage (spill guard) --

PS_OUT = ("luid_0x00000000_0x000103ae_phys_0|dedicated usage|1396.2\r\n"
          "luid_0x00000000_0x000103ae_phys_0|shared usage|64.4\r\n"
          "luid_0x00000000_0x0000d6f1_phys_0|dedicated usage|12.0\r\n"
          "luid_0x00000000_0x0000d6f1_phys_0|shared usage|210.5\r\n")


def test_parse_shared_usage_picks_the_adapter_with_the_largest_dedicated_usage():
    assert gi.parse_shared_usage(PS_OUT) == {"luid": "luid_0x00000000_0x000103ae_phys_0", "dedicated_mib": 1396.2,
                                             "shared_mib": 64.4}
    # the LUIDs change across boots and the order of the counters is not fixed: every reading decides afresh
    swapped = PS_OUT.replace("0x000103ae", "0xAAAA").replace("1396.2", "5.0").replace("|dedicated usage|12.0",
                                                                                       "|Dedicated Usage|900.0")
    assert gi.parse_shared_usage(swapped)["luid"] == "luid_0x00000000_0x0000d6f1_phys_0"
    assert gi.parse_shared_usage(swapped)["shared_mib"] == 210.5


@pytest.mark.parametrize("text", ["", "Get-Counter : The specified object was not found\r\n",
                                  "luid_a_phys_0|dedicated usage|1396.2\n",              # no shared value
                                  "luid_a_phys_0|shared usage|64.4\n",                   # no dedicated value
                                  "luid_a_phys_0|dedicated usage|1,396.2\nluid_a_phys_0|shared usage|64.4\n"])
def test_parse_shared_usage_fails_closed(text):
    with pytest.raises(gi.SharedUsageError):
        gi.parse_shared_usage(text)


def test_read_shared_usage_runs_powershell_from_wsl_with_a_timeout():
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return Done(PS_OUT)

    assert gi.read_shared_usage(run=run)["shared_mib"] == 64.4
    argv, kw = calls[0]
    assert argv[:4] == [gi.POWERSHELL, "-NoProfile", "-NonInteractive", "-Command"] and len(argv) == 5
    assert gi.POWERSHELL == "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    # PowerShell single-quoted strings: one backslash each, exactly as the counter path is written
    assert "'\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Shared Usage'" in argv[4]
    assert "Get-Counter" in argv[4] and "InvariantCulture" in argv[4] and "-split '\\\\'" in argv[4]
    assert kw["timeout"] == 20 and kw["stdin"] == subprocess.DEVNULL and kw.get("check") is not True
    gi.read_shared_usage(run=run, timeout_s=5)
    assert calls[1][1]["timeout"] == 5


@pytest.mark.parametrize("effect", [subprocess.TimeoutExpired("powershell.exe", 20), FileNotFoundError("no exe"),
                                    PermissionError("interop off"), Done("", 1, "Get-Counter : error"),
                                    Done("garbage\r\n"),
                                    UnicodeDecodeError("utf-8", b"\xa5\xbf", 0, 1, "invalid start byte")])
def test_read_shared_usage_raises_on_any_failure(effect):
    def run(argv, **kw):
        if isinstance(effect, BaseException):
            raise effect
        return effect

    with pytest.raises(gi.SharedUsageError):
        gi.read_shared_usage(run=run)


def test_read_shared_usage_reads_through_non_utf8_stderr_of_a_successful_powershell():
    """powershell.exe writes its stderr (e.g. CLIXML progress records) in the Windows OEM code page, cp950 on a zh-TW
    Windows: with valid readings on stdout and exit 0 that is a reading, never a decode error (the real subprocess
    decoding, with read_shared_usage's own arguments)."""
    noise = b'#< CLIXML\r\n<Objs><S S="progress">' + "準備模組以供第一次使用".encode("cp950") + b"</S></Objs>\r\n"
    script = f"import sys; sys.stdout.buffer.write({PS_OUT.encode()!r}); sys.stderr.buffer.write({noise!r})"

    def run(argv, **kw):
        return subprocess.run([sys.executable, "-c", script], **kw)

    assert gi.read_shared_usage(run=run) == {"luid": "luid_0x00000000_0x000103ae_phys_0", "dedicated_mib": 1396.2,
                                             "shared_mib": 64.4}


def test_spill_is_64_mib_or_more_above_the_pre_boot_reading():
    """031 「多出 64 MiB 以上即為 spill」: 以上 includes 64 (readings have 0.1 MiB resolution, so + 64.0 happens)."""
    assert gi.spill(64.4, 128.3) is False                 # + 63.9
    assert gi.spill(64.4, 128.4) is True                  # + 64.0: the boundary is a spill
    assert gi.spill(32.7, 96.7) is True and gi.spill(0.1, 64.1) is True
    assert gi.spill(64.4, 128.5) is True
    assert gi.spill(64.4, 30.0) is False                  # a drop is never a spill
    assert gi.spill(64.4, 74.4, tol_mib=5) is True


# ------------------------------------------------------------------ SpillGuard: one boot's readings --

def reader(values):
    """A fake read_shared_usage: values in order (the last repeats); an exception instance is raised."""
    it = list(values)
    calls = []

    def read():
        v = it[min(len(calls), len(it) - 1)]
        calls.append(v)
        if isinstance(v, BaseException):
            raise v
        if isinstance(v, dict):
            return v
        return {"luid": "luid_a_phys_0", "dedicated_mib": 1396.0, "shared_mib": v}

    return read


def test_spill_guard_records_pre_ready_and_bench_readings():
    g = gi.SpillGuard(reader([64.4, 70.0, 128.3, 200.0]))          # + 63.9 at b1-n1: not a spill
    assert g.start()["shared_mib"] == 64.4
    assert g.check("ready")["spilled"] is False
    assert g.check("b1-n1")["spilled"] is False
    e = g.check("b1-n2")
    assert e["spilled"] is True and e["delta_mib"] == pytest.approx(135.6) and e["when"] == "b1-n2"
    rec = g.record()
    assert rec["tol_mib"] == 64 and rec["pre"]["shared_mib"] == 64.4 and rec["spilled"] == "b1-n2"
    assert [x["when"] for x in rec["readings"]] == ["ready", "b1-n1", "b1-n2"]
    assert g.row_record() == {"pre_mib": 64.4, "readings_mib": [70.0, 128.3, 200.0], "spilled": True, "tol_mib": 64}


def test_spill_guard_fails_closed_and_records_the_error():
    g = gi.SpillGuard(reader([subprocess.TimeoutExpired("powershell.exe", 20)]))
    with pytest.raises(gi.SharedUsageError, match="pre-boot"):
        g.start()
    assert g.pre is None and "TimeoutExpired" in g.record()["error"]
    g = gi.SpillGuard(reader([64.4, gi.SharedUsageError("no adapter")]))
    g.start()
    with pytest.raises(gi.SharedUsageError, match="ready"):
        g.check("ready")
    assert "no adapter" in g.record()["error"]


def test_spill_guard_refuses_a_reading_of_another_adapter():
    other = {"luid": "luid_b_phys_0", "dedicated_mib": 15000.0, "shared_mib": 64.4}
    g = gi.SpillGuard(reader([64.4, other]))
    g.start()
    with pytest.raises(gi.SharedUsageError, match="adapter"):
        g.check("ready")


def test_spill_guard_needs_the_pre_boot_reading_first():
    with pytest.raises(gi.SharedUsageError, match="pre-boot"):
        gi.SpillGuard(reader([64.4])).check("ready")


# ------------------------------------------------------------------ module boundary --

def test_gpu_idle_imports_only_the_standard_library():
    tree = ast.parse((ROOT / "benchmarks" / "gpu_idle.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module.split(".")[0])
    assert sorted(m for m in mods if m not in sys.stdlib_module_names and m != "__future__") == []
