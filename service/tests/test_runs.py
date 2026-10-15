"""Tests for capserve/runs.py (decisions 027 §5, 029 "產品負載測試"): product-test outputs live only under
<service_root>/runs, never under the repo's results/ or the research directory, never with a formal- id, and every
meta.json says purpose product-test, formal false, research_valid false, "not a capacity measurement"."""

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from capserve import runs

NOW = dt.datetime(2026, 10, 16, 3, 4, 5, tzinfo=dt.timezone.utc)
PROFILE = {"id": "provisional-rtx", "status": "provisional", "content_sha256": "ab" * 32}


@pytest.fixture
def roots(tmp_path):
    svc, research, repo = tmp_path / "svc", tmp_path / "research", tmp_path / "repo"
    for d in (svc, research, repo / "results"):
        d.mkdir(parents=True)
    return {"service_root": svc, "research_dir": research, "repo_root": repo}


def create(roots, run_id="pt-20261016T030405Z", **kw):
    args = {"kind": "deploy", "release": "c" * 40, "profile": PROFILE, "now_utc": NOW, **roots, **kw}
    return runs.create_run(run_id=run_id, **args)


def test_run_ids():
    assert runs.new_run_id(NOW) == "pt-20261016T030405Z"
    assert runs.new_run_id(NOW, "smoke") == "pt-20261016T030405Z-smoke"
    assert runs.new_run_id(NOW.astimezone(dt.timezone(dt.timedelta(hours=8)))) == "pt-20261016T030405Z"
    with pytest.raises(runs.RunError):
        runs.new_run_id(NOW, "Bad Name")


def test_create_run_writes_the_product_test_meta(roots):
    d = create(roots, extra={"scenario": "smoke"})
    assert d == roots["service_root"] / "runs" / "pt-20261016T030405Z"
    meta = json.loads((d / "meta.json").read_text())
    assert meta == {"schema_version": 1, "run_id": "pt-20261016T030405Z", "kind": "deploy",
                    "purpose": "product-test", "formal": False, "research_valid": False,
                    "note": "not a capacity measurement", "release": "c" * 40, "profile": PROFILE,
                    "created_utc": "2026-10-16T03:04:05Z", "scenario": "smoke"}
    assert runs.read_meta(d) == meta


@pytest.mark.parametrize("run_id", ["formal-20261016-x", "formal-pt-20261016T030405Z", "pt-20261016T030405Z/../x",
                                    "../pt-20261016T030405Z", "pilot-1", "pt-2026", "", "pt-20261016T030405Z-UP"])
def test_bad_ids_are_refused(roots, run_id):
    with pytest.raises(runs.RunError):
        create(roots, run_id=run_id)
    assert not (roots["service_root"] / "runs").exists() or not any((roots["service_root"] / "runs").iterdir())


def test_formal_ids_are_named_in_the_refusal(roots):
    with pytest.raises(runs.RunError, match="formal-"):
        create(roots, run_id="formal-20261016-capacity")


def test_a_service_root_inside_the_research_dir_is_refused(roots):
    inside = {**roots, "service_root": roots["research_dir"] / "svc"}
    with pytest.raises(runs.RunError, match="research"):
        create(inside)
    assert not (roots["research_dir"] / "svc").exists()


def test_a_service_root_under_results_is_refused(roots):
    under = {**roots, "service_root": roots["repo_root"] / "results" / "svc"}
    with pytest.raises(runs.RunError, match="results"):
        create(under)
    assert not (roots["repo_root"] / "results" / "svc").exists()


def test_a_symlinked_runs_dir_into_the_research_dir_is_refused(roots):
    os.symlink(roots["research_dir"], roots["service_root"] / "runs")
    with pytest.raises(runs.RunError, match="research"):
        create(roots)
    assert list(roots["research_dir"].iterdir()) == []


def test_an_existing_run_is_never_overwritten(roots):
    d = create(roots)
    before = (d / "meta.json").read_text()
    with pytest.raises(runs.RunError, match="exists"):
        create(roots, kind="loadtest")
    assert (d / "meta.json").read_text() == before


def test_path_problems_lists_everything(roots):
    problems = runs.run_path_problems("formal-x", service_root=roots["research_dir"],
                                      research_dir=roots["research_dir"], repo_root=roots["repo_root"])
    assert any("formal-" in p for p in problems) and any("research" in p for p in problems)


def test_the_default_repo_root_is_this_checkout():
    assert runs.REPO_ROOT == Path(__file__).resolve().parents[2]
    assert (runs.REPO_ROOT / "service" / "capserve" / "runs.py").is_file()


def test_update_meta_keeps_the_fixed_fields_and_the_start_mtime(roots):
    d = create(roots)
    meta = d / "meta.json"
    os.utime(meta, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    out = runs.update_meta(d, outcome={"ok": True, "problems": []})
    assert out["outcome"] == {"ok": True, "problems": []} and out["purpose"] == "product-test"
    assert json.loads(meta.read_text()) == out
    assert meta.stat().st_mtime_ns == 1_000_000_000_000_000_000   # IT-6: find -newer meta.json covers the run
    for fixed in ("purpose", "formal", "research_valid", "note", "run_id", "kind", "created_utc", "schema_version",
                  "release", "profile"):
        with pytest.raises(runs.RunError):
            runs.update_meta(d, **{fixed: "changed"})
    assert json.loads(meta.read_text()) == out


def test_the_provenance_of_a_run_cannot_be_relabelled_later(roots):
    """A provisional run never becomes a verified one, nor another release, after it started (027 §4)."""
    d = create(roots)
    with pytest.raises(runs.RunError, match="fixed"):
        runs.update_meta(d, release="0" * 40, profile={"id": "verified-x", "status": "verified"})
    meta = runs.read_meta(d)
    assert meta["release"] == "c" * 40 and meta["profile"] == PROFILE
    with pytest.raises(runs.RunError, match="fixed"):
        create(roots, run_id="pt-20261016T030406Z", extra={"release": "0" * 40})
