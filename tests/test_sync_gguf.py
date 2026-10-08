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
    cmd = sg.receive_command("Qwen3.5-4B-Q8_0.gguf", sha)
    assert cmd[0] == "ssh" and "gpu-server" in cmd and cmd[-3:] == [sg.RECEIVER, "Qwen3.5-4B-Q8_0.gguf", sha]
    assert not any("'" in a or '"' in a for a in cmd)  # nothing for cmd.exe to mangle
    with pytest.raises(ValueError):
        sg.receive_command("x; rm -rf ~.gguf", sha)
    with pytest.raises(ValueError):
        sg.receive_command("Qwen3.5-4B-Q8_0.gguf", "zz")


def test_receiver_verifies_before_renaming():
    s = sg.RECEIVER_SCRIPT
    assert s.index("sha256sum") < s.index("mv ") and "MISMATCH" in s and sg.REMOTE_DIR in s
