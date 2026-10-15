"""Tests for capserve/__main__.py: `python -m capserve serve --config F --profile P [--allow-provisional]` refuses a
provisional profile unless the flag or CAPSERVE_ALLOW_PROVISIONAL=1 allows it (027 §4; decision 029)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from capserve.__main__ import main

SERVICE = Path(__file__).resolve().parent.parent
ROOT = SERVICE.parent
DEV = str(SERVICE / "profiles" / "provisional-dev.json")


class Runner:
    def __init__(self):
        self.calls = []

    def __call__(self, settings, profile, *, api_key):
        self.calls.append((settings, profile, api_key))


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "gw.toml"
    path.write_text('active_class = "dev-2k"\n')
    return str(path)


def test_provisional_is_refused_without_the_flag(cfg, capsys):
    run = Runner()
    assert main(["serve", "--config", cfg, "--profile", DEV], env={}, run=run) == 2
    assert run.calls == [] and "--allow-provisional" in capsys.readouterr().err


@pytest.mark.parametrize("argv_extra, env", [(["--allow-provisional"], {}),
                                             ([], {"CAPSERVE_ALLOW_PROVISIONAL": "1"})])
def test_provisional_is_served_when_allowed_and_says_so(cfg, capsys, argv_extra, env):
    run = Runner()
    assert main(["serve", "--config", cfg, "--profile", DEV, *argv_extra], env=env, run=run) == 0
    settings, profile, api_key = run.calls[0]
    assert (settings.active_class, profile.profile_id, api_key) == ("dev-2k", "provisional-dev", None)
    assert "PROVISIONAL" in capsys.readouterr().err


def test_the_api_key_comes_from_the_environment_and_is_never_printed(cfg, capsys):
    run = Runner()
    env = {"CAPSERVE_API_KEY": "s3cr3t-key", "CAPSERVE_ALLOW_PROVISIONAL": "1"}
    assert main(["serve", "--config", cfg, "--profile", DEV], env=env, run=run) == 0
    assert run.calls[0][2] == b"s3cr3t-key"
    out = capsys.readouterr()
    assert "s3cr3t" not in out.out + out.err


@pytest.mark.parametrize("text", ['active_class = "nope"\n', 'active_class = "dev-2k"\nhost = "0.0.0.0"\n',
                                  'active_class = "dev-2k"\nworkers = 4\n'])
def test_bad_settings_exit_2(tmp_path, text):
    path = tmp_path / "gw.toml"
    path.write_text(text)
    run = Runner()
    assert main(["serve", "--config", str(path), "--profile", DEV, "--allow-provisional"], env={}, run=run) == 2
    assert run.calls == []


def test_a_public_bind_is_allowed_with_a_key(tmp_path):
    path = tmp_path / "gw.toml"
    path.write_text('active_class = "dev-2k"\nhost = "0.0.0.0"\n')
    run = Runner()
    assert main(["serve", "--config", str(path), "--profile", DEV, "--allow-provisional"],
                env={"CAPSERVE_API_KEY": "k"}, run=run) == 0


def test_module_entry_point_refuses_provisional(cfg):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSERVE_")}
    env["PYTHONPATH"] = f"{SERVICE}{os.pathsep}{ROOT / 'benchmarks'}"
    r = subprocess.run([sys.executable, "-m", "capserve", "serve", "--config", cfg, "--profile", DEV],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=60)
    assert r.returncode == 2 and "provisional" in r.stderr
    r = subprocess.run([sys.executable, "-m", "capserve"], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=60)
    assert r.returncode == 2
