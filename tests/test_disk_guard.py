"""scripts/disk_guard.py: refuse to create a large artifact unless at least the
reserve stays free afterwards (docs/STORAGE_PLAN.md)."""

import json
from collections import namedtuple

import pytest

import disk_guard as dg

Usage = namedtuple("Usage", "total used free")
GiB = 1024**3


def usage(free_gib):
    return lambda path: Usage(460 * GiB, (460 - free_gib) * GiB, free_gib * GiB)


def test_ok_when_reserve_survives():
    r = dg.check("/x", need_bytes=20 * GiB, reserve_bytes=50 * GiB, disk_usage=usage(175))
    assert r["ok"] is True
    assert r["free_gib"] == 175.0 and r["after_gib"] == 155.0


def test_refuses_when_reserve_would_be_eaten():
    r = dg.check("/x", need_bytes=130 * GiB, reserve_bytes=50 * GiB, disk_usage=usage(175))
    assert r["ok"] is False
    assert r["after_gib"] == 45.0


def test_exact_boundary_is_ok():
    assert dg.check("/x", need_bytes=125 * GiB, reserve_bytes=50 * GiB, disk_usage=usage(175))["ok"] is True


def test_negative_need_rejected():
    with pytest.raises(ValueError):
        dg.check("/x", need_bytes=-1, reserve_bytes=0, disk_usage=usage(100))


def test_nonexistent_path_uses_nearest_existing_parent(tmp_path):
    target = tmp_path / "models" / "gguf" / "new"
    seen = []

    def fake(path):
        seen.append(path)
        return Usage(1, 0, 100 * GiB)

    dg.check(str(target), need_bytes=0, reserve_bytes=0, disk_usage=fake)
    assert seen == [str(tmp_path)]


def test_default_reserve_is_50_gib():
    assert dg.DEFAULT_RESERVE_GIB == 50


@pytest.mark.parametrize("need, code", [("10", 0), ("200", 3)])
def test_cli_exit_codes(tmp_path, monkeypatch, capsys, need, code):
    monkeypatch.setattr(dg.shutil, "disk_usage", usage(175))
    assert dg.main(["--path", str(tmp_path), "--need-gib", need]) == code
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is (code == 0)
    assert out["reserve_gib"] == 50.0
