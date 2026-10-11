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
    assert out["free_gib"] == 175.0  # the patched usage, never the machine's real free space


@pytest.mark.parametrize("free_gib, need, code", [(0, 0, 3), (10**7, 10**6, 0)])
def test_check_reads_disk_usage_at_call_time(tmp_path, monkeypatch, free_gib, need, code):
    """Regression: a default bound at import time ignored the patch and read the real disk, so the
    CLI outcome depended on the machine (passes only with 60-250 GiB free). The `seen` asserts are
    what make this fail against an import-time default on any machine; the exit codes alone could
    still match a real disk (e.g. < 50 GiB free for the first case)."""
    seen = []

    def fake(path):
        seen.append(path)
        return usage(free_gib)(path)

    monkeypatch.setattr(dg.shutil, "disk_usage", fake)
    assert dg.main(["--path", str(tmp_path), "--need-gib", str(need)]) == code
    assert seen == [str(tmp_path)]
    seen.clear()
    assert dg.check(str(tmp_path), need_bytes=need * GiB, reserve_bytes=50 * GiB)["ok"] is (code == 0)
    assert seen == [str(tmp_path)]
