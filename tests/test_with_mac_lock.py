"""tools/with-mac-lock: project-local wrapper for the shared Mac heavy-work lock
(MACHINE_POLICY "Mac Hard Lock"). Same semantics as the other portfolio projects:
mkdir lock + owner.txt, fail fast when held, never delete someone else's lock.
Tests point the lock at a temp dir via AI_PORTFOLIO_MAC_LOCK_DIR."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "with-mac-lock"


def run(lock, *cmd):
    env = {**os.environ, "AI_PORTFOLIO_MAC_LOCK_DIR": str(lock)}
    return subprocess.run([str(TOOL), *cmd], capture_output=True, text=True, env=env, timeout=30)


def test_default_lock_path_is_the_shared_one():
    assert "/tmp/ai-portfolio-mac-heavy.lock" in TOOL.read_text()


def test_runs_command_and_releases_lock(tmp_path):
    lock = tmp_path / "mac.lock"
    res = run(lock, "sh", "-c", "exit 0")
    assert res.returncode == 0
    assert not lock.exists()


def test_propagates_exit_code_and_releases_on_failure(tmp_path):
    lock = tmp_path / "mac.lock"
    res = run(lock, "sh", "-c", "exit 3")
    assert res.returncode == 3
    assert not lock.exists()


def test_owner_file_and_env_visible_while_running(tmp_path):
    lock = tmp_path / "mac.lock"
    probe = 'cat "$AI_PORTFOLIO_MAC_LOCK_DIR/owner.txt"; echo "held=$LLM_SERVING_LAB_MAC_LOCK_HELD"'
    res = run(lock, "sh", "-c", probe)
    assert res.returncode == 0
    for key in ("pid=", "started_at=", "cwd=", "command=", "project=llm-serving-lab"):
        assert key in res.stdout
    assert "held=1" in res.stdout


def test_held_lock_fails_fast_and_is_not_touched(tmp_path):
    lock = tmp_path / "mac.lock"
    lock.mkdir()
    (lock / "owner.txt").write_text("pid=4242\ncommand=cargo bench\n")
    marker = tmp_path / "ran"
    res = run(lock, "touch", str(marker))
    assert res.returncode == 75
    assert not marker.exists()
    assert "cargo bench" in res.stderr
    assert (lock / "owner.txt").read_text() == "pid=4242\ncommand=cargo bench\n"


def test_usage_without_command(tmp_path):
    res = run(tmp_path / "mac.lock")
    assert res.returncode == 2
    assert not (tmp_path / "mac.lock").exists()
