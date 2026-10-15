"""Tests for capserve/profile_gen.py (M0 stub): no verified profile can be generated before Stage C is frozen
(configs/capacity.toml [freeze] capacity = true; a missing file or key is false; decision 029 / 030), and the
generator itself is M2 work, so M0 always refuses with exit 2 and never writes a profile. Every committed verified
profile must be what the generator writes (029), so in M0 no committed profile can be verified."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from capserve import profile_gen as pg

SERVICE = Path(__file__).resolve().parent.parent
ROOT = SERVICE.parent


def run(tmp_path, capacity_text):
    path = tmp_path / "capacity.toml"
    if capacity_text is not None:
        path.write_text(capacity_text)
    out = tmp_path / "profile.json"
    return pg.main(["--session", "results/capacity/formal-x", "--out", str(out)], capacity_toml=path), out


@pytest.mark.parametrize("text", [None, "", "[slo]\nid = 'S2'\n", "[freeze]\n", "[freeze]\ncapacity = false\n",
                                  "[freeze]\ncapacity = 'true'\n", "[freeze]\ncapacity = 1\n"])
def test_refused_until_capacity_is_frozen(tmp_path, capsys, text):
    code, out = run(tmp_path, text)
    err = capsys.readouterr().err
    assert code == 2 and not out.exists()
    assert "capacity.toml" in err and "[freeze] capacity" in err


def test_unreadable_capacity_toml_is_refused(tmp_path, capsys):
    code, out = run(tmp_path, "[freeze\n")
    assert code == 2 and not out.exists() and "cannot read" in capsys.readouterr().err


def test_even_after_the_freeze_m0_has_no_generator(tmp_path, capsys):
    code, out = run(tmp_path, "[freeze]\ncapacity = true\n")
    assert code == 2 and not out.exists() and "M2" in capsys.readouterr().err


def test_no_override_options():
    assert pg.option_names() == {"--session", "--out"}


def test_default_path_is_the_repo_capacity_toml():
    assert pg.CAPACITY_TOML == ROOT / "configs" / "capacity.toml"


def test_module_entry_point(tmp_path):
    env = {**os.environ, "PYTHONPATH": f"{SERVICE}{os.pathsep}{ROOT / 'benchmarks'}"}
    r = subprocess.run([sys.executable, "-m", "capserve.profile_gen", "--session", "s", "--out",
                        str(tmp_path / "p.json")], capture_output=True, text=True, env=env, cwd=ROOT, timeout=60)
    assert r.returncode == 2 and "refused" in r.stderr and not (tmp_path / "p.json").exists()   # frozen or not


def _regenerable(doc: dict) -> dict:
    """The profile without what a regeneration may change (029: generator_commit, and the hash over it)."""
    out = {k: v for k, v in doc.items() if k != "content_sha256"}
    out["provenance"] = {k: v for k, v in (doc.get("provenance") or {}).items() if k != "generator_commit"}
    return out


def test_every_committed_verified_profile_is_what_the_generator_writes(tmp_path):
    paths = sorted((SERVICE / "profiles").glob("*.json"))
    assert paths
    for path in paths:
        doc = json.loads(path.read_text())
        if doc.get("status") != "verified":
            continue
        session = ROOT / "results" / "capacity" / str((doc.get("provenance") or {}).get("source_session"))
        out = tmp_path / path.name
        code = pg.main(["--session", str(session), "--out", str(out)])
        assert code == 0 and out.exists(), f"{path.name} is verified, but the generator does not write it"
        assert _regenerable(json.loads(out.read_text())) == _regenerable(doc), path.name
