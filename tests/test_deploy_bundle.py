"""Tests for scripts/deploy_bundle.py: the remote copy is a bundle of a clean HEAD
plus DEPLOYED_COMMIT and DEPLOY_MANIFEST.json, which env_capture verifies (review F12)."""

import hashlib
import json
import subprocess
import tarfile

import pytest

import deploy_bundle as db


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def make_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "benchmarks").mkdir(parents=True)
    (root / "results" / "core_b").mkdir(parents=True)
    (root / "benchmarks" / "coreb_smoke.py").write_text("print('smoke')\n")
    (root / "results" / "core_b" / "S1.json").write_text("{}\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    return root


def test_bundle_contains_head_commit_and_manifest(tmp_path):
    root = make_repo(tmp_path)
    out = tmp_path / "bundle.tar"
    manifest = db.build_bundle(root, out)
    head = git(root, "rev-parse", "HEAD").strip()
    assert manifest["commit"] == head
    assert manifest["files"] == {
        "benchmarks/coreb_smoke.py": hashlib.sha256(b"print('smoke')\n").hexdigest(),
    }  # results/ is produced remotely and never overwritten by a deploy
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
        assert {"DEPLOYED_COMMIT", "DEPLOY_MANIFEST.json", "benchmarks/coreb_smoke.py"} <= names
        assert not any(n.startswith("results") for n in names)
        assert tar.extractfile("DEPLOYED_COMMIT").read().decode().strip() == head
        assert json.loads(tar.extractfile("DEPLOY_MANIFEST.json").read()) == manifest


def test_bundle_refuses_a_dirty_tree(tmp_path):
    root = make_repo(tmp_path)
    (root / "benchmarks" / "coreb_smoke.py").write_text("hot patch\n")
    with pytest.raises(db.DeployError, match="uncommitted"):
        db.build_bundle(root, tmp_path / "bundle.tar")
