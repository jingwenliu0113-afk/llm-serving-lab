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
        return subprocess.CompletedProcess(cmd, 0, "VERIFIED x", "")

    assert sg.copy_one("Qwen3.5-4B-Q8_0.gguf", sha, len(data), tmp_path, run=run, send=send) is True
    assert sends == [0, len(data) // 2] and bytes(remote) == data
