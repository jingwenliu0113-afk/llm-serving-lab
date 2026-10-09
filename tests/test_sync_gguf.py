"""scripts/sync_gguf_to_wsl.py: only the 6 pinned quants go to WSL, verified by sha256 there."""

import tomllib
from pathlib import Path

import pytest

import sync_gguf_to_wsl as sg

ROOT = Path(__file__).resolve().parent.parent


def test_quant_files_are_the_six_pinned_quants():
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    files = sg.quant_files(pins)
    assert len(files) == 6 and not any("BF16" in n for n, _, _ in files)
    assert all(len(sha) == 64 and size > 0 for _, sha, size in files)


def test_parse_check_and_receive_command():
    sha = "a" * 64
    assert sg.parse_check(f"{sha}  Qwen3.5-4B-Q8_0.gguf\nnoise\n") == {"Qwen3.5-4B-Q8_0.gguf": sha}
    cmd = sg.receive_command("Qwen3.5-4B-Q8_0.gguf", sha, 123)
    assert cmd[0] == "ssh" and "gpu-server" in cmd and cmd[-4:] == [sg.RECEIVER, "Qwen3.5-4B-Q8_0.gguf", sha, "123"]
    assert not any("'" in a or '"' in a for a in cmd)  # nothing for cmd.exe to mangle
    assert sg.receive_command("Qwen3.5-4B-Q8_0.gguf", sha, "query")[-1] == "query"
    with pytest.raises(ValueError):
        sg.receive_command("x; rm -rf ~.gguf", sha, 1)
    with pytest.raises(ValueError):
        sg.receive_command("Qwen3.5-4B-Q8_0.gguf", "zz", 1)
    with pytest.raises(ValueError):
        sg.receive_command("Qwen3.5-4B-Q8_0.gguf", sha, "1; rm")


def test_receiver_appends_keeps_partials_and_verifies_before_renaming():
    s = sg.RECEIVER_SCRIPT
    assert 'cat >> ".$name.part"' in s and "INCOMPLETE" in s  # resumable; an interrupted copy is kept
    assert s.index("INCOMPLETE") < s.index("sha256sum") < s.index("mv ") and "MISMATCH" in s and sg.REMOTE_DIR in s


def test_copy_one_resumes_from_the_partial_size(tmp_path):
    import subprocess

    data = bytes(range(256)) * 40
    (tmp_path / "Qwen3.5-4B-Q8_0.gguf").write_bytes(data)
    sha = "b" * 64
    remote = bytearray()
    sends = []

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, f"{len(remote)}\n", "")

    def send(path, offset, cmd):
        sends.append(offset)
        chunk = path.read_bytes()[offset:]
        if len(sends) == 1:  # the link drops half way
            remote.extend(chunk[: len(chunk) // 2])
            return subprocess.CompletedProcess(cmd, 255, "", "Read from remote host: Operation timed out")
        remote.extend(chunk)
        return subprocess.CompletedProcess(cmd, 0, "VERIFIED Qwen3.5-4B-Q8_0.gguf", "")

    assert sg.copy_one("Qwen3.5-4B-Q8_0.gguf", sha, len(data), tmp_path, run=run, send=send,
                       sleep=lambda s: None) is True
    assert sends == [0, len(data) // 2] and bytes(remote) == data


def test_stage_b_files_go_first():
    files = [(n, "a" * 64, 1) for n in ("Qwen3.5-4B-Q8_0.gguf", "Qwen3.5-9B-Q8_0.gguf", "Qwen3.5-4B-Q4_K_M.gguf",
                                        "Qwen3.5-9B-Q6_K.gguf")]
    assert [f[0] for f in sg.copy_order(files)] == ["Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-9B-Q8_0.gguf",
                                                    "Qwen3.5-4B-Q8_0.gguf", "Qwen3.5-9B-Q6_K.gguf"]


def test_copy_one_backs_off_between_failed_attempts_and_reports_a_temporary_failure(tmp_path):
    import subprocess

    (tmp_path / "Qwen3.5-4B-Q8_0.gguf").write_bytes(b"x" * 10)
    slept = []

    def run(cmd, **kw):  # the link is down: ssh fails every time
        return subprocess.CompletedProcess(cmd, 255, "", "Could not resolve hostname")

    res = sg.copy_one("Qwen3.5-4B-Q8_0.gguf", "b" * 64, 10, tmp_path, run=run, send=None, attempts=5,
                      sleep=slept.append)
    assert res is None and len(slept) == 5 and slept == sorted(slept) and slept[0] > 0  # backoff, then give up


def test_success_needs_the_verified_line():
    import subprocess

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "0\n", "")

    def send(path, offset, cmd):  # rc 0 but no VERIFIED (e.g. an old receiver): not a success
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert sg.copy_one("Qwen3.5-4B-Q8_0.gguf", "b" * 64, 10, Path("/nonexistent"), run=run, send=send, attempts=2,
                       sleep=lambda s: None) is None


def test_receiver_stops_on_mkdir_cd_or_rename_failures():
    s = sg.RECEIVER_SCRIPT
    assert "set -eu" in s and "mv -T" in s and "NODIR" in s


def test_main_treats_a_failed_remote_check_as_temporary(monkeypatch, capsys):
    import subprocess

    monkeypatch.setattr(sg.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 255, "", "timed out"))
    assert sg.main(["--dry-run"]) == sg.EX_TEMPFAIL
    assert "remote check failed" in capsys.readouterr().err


def test_send_from_streams_the_tail_through_a_real_subprocess(tmp_path):
    """Regression: closing stdin and then calling communicate() raised 'flush of closed file'."""
    f = tmp_path / "x.gguf"
    f.write_bytes(b"abcdef")
    res = sg.send_from(f, 2, ["sh", "-c", "wc -c | tr -d ' '; echo VERIFIED x.gguf"])
    assert res.returncode == 0 and res.stdout.split() == ["4", "VERIFIED", "x.gguf"]


def test_reference_files_are_the_two_bf16_references_9b_first():
    """020: quality moved to the RTX, so the BF16 references go to WSL too (9B first, it is needed in any case)."""
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    files = sg.reference_files(pins)
    assert [n for n, _, _ in files] == ["Qwen3.5-9B-BF16.gguf", "Qwen3.5-4B-BF16.gguf"]
    assert all(len(sha) == 64 and size > 0 for _, sha, size in files)


class FakeClock:
    def __init__(self):
        self.t, self.slept = 0.0, 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.slept += s


def test_send_from_throttles_to_the_rate_limit(tmp_path):
    """--max-mib-s keeps the link usable for other traffic: 3 MiB at 1 MiB/s takes >= ~3 s on the fake clock."""
    f = tmp_path / "x.gguf"
    f.write_bytes(b"\0" * (3 * 2**20))
    clock = FakeClock()
    res = sg.send_from(f, 0, ["sh", "-c", "wc -c | tr -d ' '"], max_bytes_per_s=2**20, clock=clock.now,
                       sleep=clock.sleep)
    assert res.stdout.split() == [str(3 * 2**20)]
    assert 2.0 <= clock.slept <= 3.01
    free = FakeClock()
    sg.send_from(f, 0, ["sh", "-c", "cat > /dev/null"], clock=free.now, sleep=free.sleep)
    assert free.slept == 0.0  # no limit -> no sleeping
