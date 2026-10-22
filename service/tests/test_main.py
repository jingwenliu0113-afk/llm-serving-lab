"""Tests for capserve/__main__.py: `python -m capserve serve --config F --profile P [--allow-provisional]` refuses a
provisional profile unless the flag or CAPSERVE_ALLOW_PROVISIONAL=1 allows it (027 §4; decision 029), refuses an
unavailable active class of a verified profile whatever flag is given, and serves a verified 0 class as not
serviceable (decision 030 8.7)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from capserve.__main__ import main
from fakes.profiles import FROZEN, frozen_capacity, frozen_configs, rehash, verified_class, verified_doc

SERVICE = Path(__file__).resolve().parent.parent
ROOT = SERVICE.parent
DEV = str(SERVICE / "profiles" / "provisional-dev.json")


class Runner:
    def __init__(self):
        self.calls = []

    def __call__(self, settings, profile, *, api_key, local_tokenizer=None):
        self.calls.append((settings, profile, api_key))
        self.local_tokenizer = local_tokenizer


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


@pytest.fixture
def profiles(tmp_path):
    """The checkout's service/profiles: serve takes a verified profile only with the bytes of one of its files."""
    path = tmp_path / "profiles"
    path.mkdir()
    return path


def verified_files(tmp_path, profiles, cls=None, *, doc=None, name="verified.json"):
    """(gateway settings whose active class is the profile's, the verified profile in profiles/)."""
    doc = doc or verified_doc(classes=[verified_class(**(cls or {}))])
    settings = tmp_path / "gw-verified.toml"
    settings.write_text(f'active_class = "{doc["classes"][0]["name"]}"\n')
    path = profiles / name
    path.write_text(json.dumps(doc))
    return str(settings), str(path)


def serve(settings, path, profiles, *extra, env=None, run=None, frozen=FROZEN):
    return main(["serve", "--config", settings, "--profile", path, *extra], env=env or {}, run=run or Runner(),
                frozen=frozen, profiles_dir=profiles)


@pytest.mark.parametrize("status", ["UNSTABLE", "NT"])
@pytest.mark.parametrize("argv_extra, env", [([], {}), (["--allow-provisional"], {"CAPSERVE_ALLOW_PROVISIONAL": "1"})])
def test_an_unavailable_active_class_is_refused_whatever_flag(tmp_path, profiles, capsys, status, argv_extra, env):
    """030 8.7: UNSTABLE / NT -> unavailable: no verified number, never served, never a fallback."""
    settings, prof = verified_files(tmp_path, profiles, {"workload": "U8K", "status": status})
    run = Runner()
    assert serve(settings, prof, profiles, *argv_extra, env=env, run=run) == 2
    assert run.calls == [] and "unavailable" in capsys.readouterr().err


def test_a_verified_0_class_starts_and_says_it_refuses_every_request(tmp_path, profiles, capsys):
    settings, prof = verified_files(tmp_path, profiles, {"workload": "M8K", "status": "INFEASIBLE", "value": 0})
    run = Runner()
    assert serve(settings, prof, profiles, run=run) == 0
    settings, profile, _ = run.calls[0]
    assert profile.verified and profile.workload_class(settings.active_class).max_concurrency == 0
    err = capsys.readouterr().err
    assert "class_not_serviceable" in err and "PROVISIONAL" not in err


def test_a_verified_class_is_served_without_any_flag(tmp_path, profiles, capsys):
    run = Runner()
    assert serve(*verified_files(tmp_path, profiles), profiles, run=run) == 0
    assert run.calls[0][1].verified and capsys.readouterr().err == ""


def test_a_forged_verified_profile_is_refused(tmp_path, profiles, capsys):
    """Review finding: `capserve serve` started a hand-written verified profile (another SLO, a session that does not
    exist) without any flag. The loader now checks it against the session ids and the frozen configs."""
    doc = verified_doc(classes=[verified_class(workload="U8K", status="EXACT", value=13)],
                       slo={"id": "S3", "terms": {"ttft_p95_ms": {"2048": 99999}, "tpot_p95_ms": 999}})
    doc["provenance"]["source_session"] = "formal-does-not-exist"
    run = Runner()
    assert serve(*verified_files(tmp_path, profiles, doc=rehash(doc)), profiles, run=run) == 2
    err = capsys.readouterr().err
    assert run.calls == [] and "profile_id" in err and "slo" in err
    settings, prof = verified_files(tmp_path, profiles, name="v2.json")
    assert serve(settings, prof, profiles, run=run, frozen=frozen_configs(frozen_capacity(capacity=False))) == 2
    assert run.calls == [] and "[freeze] capacity" in capsys.readouterr().err


def test_a_verified_profile_is_served_only_as_a_service_profiles_file(tmp_path, profiles, capsys):
    """Like deploy (the release's service/profiles/<name>.json in DEPLOY_MANIFEST.json): a verified profile is served
    only with the bytes of one of the checkout's service/profiles/*.json files, which the Mac test regenerates."""
    settings, prof = verified_files(tmp_path, profiles)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes(Path(prof).read_bytes())
    run = Runner()
    assert serve(settings, str(elsewhere), profiles, run=run) == 0          # a copy (deploy's runs/<id>/profile.json)
    Path(prof).unlink()
    assert serve(settings, str(elsewhere), profiles, run=run) == 2
    assert len(run.calls) == 1 and "service/profiles" in capsys.readouterr().err
    (profiles / "verified.JSON").write_bytes(elsewhere.read_bytes())
    assert serve(settings, str(elsewhere), profiles, run=run) == 2


def test_the_default_profiles_dir_is_service_profiles():
    import capserve.__main__ as m
    assert m.PROFILES_DIR == SERVICE / "profiles"


def test_module_entry_point_refuses_provisional(cfg):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSERVE_")}
    env["PYTHONPATH"] = f"{SERVICE}{os.pathsep}{ROOT / 'benchmarks'}"
    r = subprocess.run([sys.executable, "-m", "capserve", "serve", "--config", cfg, "--profile", DEV],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=60)
    assert r.returncode == 2 and "provisional" in r.stderr
    r = subprocess.run([sys.executable, "-m", "capserve"], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=60)
    assert r.returncode == 2


def test_a_local_tokenizer_that_cannot_load_exits_2(tmp_path, capsys):
    path = tmp_path / "gw.toml"
    path.write_text(f'active_class = "dev-2k"\ntokenizer = "local"\ntokenizer_path = "{tmp_path / "none"}"\n')
    run = Runner()
    assert main(["serve", "--config", str(path), "--profile", DEV, "--allow-provisional"], env={}, run=run) == 2
    assert run.calls == [] and "tokenizer" in capsys.readouterr().err


def test_a_local_tokenizer_is_loaded_before_serving(tmp_path, monkeypatch):
    from capserve import tokens
    loaded = object()
    monkeypatch.setattr(tokens.LocalTokenizer, "load", classmethod(lambda cls, p: (p, loaded)))
    path = tmp_path / "gw.toml"
    path.write_text('active_class = "dev-2k"\ntokenizer = "local"\ntokenizer_path = "/snap"\n')
    run = Runner()
    assert main(["serve", "--config", str(path), "--profile", DEV, "--allow-provisional"], env={}, run=run) == 0
    assert run.local_tokenizer == ("/snap", loaded)
    run = Runner()
    assert main(["serve", "--config", str(cfg_path(tmp_path)), "--profile", DEV, "--allow-provisional"], env={},
                run=run) == 0
    assert run.local_tokenizer is None


def cfg_path(tmp_path):
    path = tmp_path / "plain.toml"
    path.write_text('active_class = "dev-2k"\n')
    return path
