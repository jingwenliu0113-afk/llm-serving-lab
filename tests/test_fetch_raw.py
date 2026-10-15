"""tools/rtx/fetch_raw.sh (decision 028 step 0): read-only fetch of listed raw files with sha256 verification. Runs the
script in a copy of the repo layout against a fake `ssh` on PATH that executes the remote command in a fake WSL home
and records it."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "rtx" / "fetch_raw.sh"
SID = "formal-20261009-coreb-uniform"
FAKE_SSH = """#!/usr/bin/env bash
# fake ssh: record the remote command, run it locally with HOME = the fake WSL home (no login shell)
cmd="${@: -1}"
printf '%s\\n' "$cmd" >> "$FAKE_SSH_LOG"
cmd="${cmd/wsl.exe -d Ubuntu-22.04 -- bash -lc/bash -c}"
export HOME="$FAKE_WSL_HOME"
eval "$cmd"
"""


@pytest.fixture
def world(tmp_path):
    repo, home, bin_ = tmp_path / "repo", tmp_path / "home", tmp_path / "bin"
    (repo / "tools" / "rtx").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "tools" / "rtx" / "fetch_raw.sh")
    remote = home / "llm-serving-lab" / "results" / "core_b" / SID / "raw"
    remote.mkdir(parents=True)
    files = {"bench-uniform-8192-r1-c8.json": b'{"ttfts": [1.0]}', "server-uniform-8192-r1.log": b"line\n",
             "only-in-manifest.json": None, "tampered.json": b"original"}
    manifest = {}
    for name, data in files.items():
        blob = data if data is not None else b"x"
        manifest[f"raw/{name}"] = {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
        if data is not None:
            (remote / name).write_bytes(data)
    (remote / "tampered.json").write_bytes(b"changed on the remote")
    (remote / "unlisted.json").write_bytes(b"not in the manifest")
    session = repo / "results" / "core_b" / SID
    session.mkdir(parents=True)
    (session / "raw_manifest.json").write_text(json.dumps({"note": "n", "files": manifest}, indent=1) + "\n")
    bin_.mkdir()
    (bin_ / "ssh").write_text(FAKE_SSH)
    (bin_ / "ssh").chmod(0o755)
    log = tmp_path / "ssh.log"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_WSL_HOME": str(home), "FAKE_SSH_LOG": str(log)}

    def run(*args):
        return subprocess.run(["bash", str(repo / "tools" / "rtx" / "fetch_raw.sh"), *args], env=env,
                              capture_output=True, text=True)

    dest = repo / "results" / "capacity" / "c0-step0" / "raw" / SID
    return {"run": run, "dest": dest, "remote": remote, "log": log}


def remote_cmds(world):
    return world["log"].read_text().splitlines() if world["log"].exists() else []


def test_script_parses_and_never_mentions_the_lock():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    code = "\n".join(x.split("#", 1)[0] for x in SCRIPT.read_text().splitlines())
    assert "flock" not in code and "ai-portfolio-gpu.lock" not in code and "sudo" not in code


def test_fetches_listed_files_after_ls_and_verifies_sha256(world):
    names = ["bench-uniform-8192-r1-c8.json", "server-uniform-8192-r1.log"]
    res = world["run"]("core_b", SID, *names)
    assert res.returncode == 0, res.stdout + res.stderr
    for n in names:
        assert (world["dest"] / n).read_bytes() == (world["remote"] / n).read_bytes()
    assert sorted(p.name for p in world["dest"].iterdir()) == sorted(names)
    cmds = remote_cmds(world)
    assert len(cmds) == 2 and " ls -- " in cmds[0] and " tar -c -- " in cmds[1]
    for c in cmds:  # read-only: cd into the session's raw/, then ls or tar -c of exactly the listed names
        assert c.startswith(f'wsl.exe -d Ubuntu-22.04 -- bash -lc "cd ~/llm-serving-lab/results/core_b/{SID}/raw && ')
        assert c.endswith(" -- " + " ".join(names) + '"')
        assert "rm " not in c and ">" not in c and "lock" not in c


def test_refuses_to_overwrite_and_asks_nothing_remote(world):
    assert world["run"]("core_b", SID, "server-uniform-8192-r1.log").returncode == 0
    n = len(remote_cmds(world))
    res = world["run"]("core_b", SID, "server-uniform-8192-r1.log")
    assert res.returncode == 3 and "already exists" in res.stdout
    assert len(remote_cmds(world)) == n


def test_missing_remote_file_stops_before_any_copy(world):
    res = world["run"]("core_b", SID, "bench-uniform-8192-r1-c8.json", "only-in-manifest.json")
    assert res.returncode == 4
    assert not world["dest"].exists() or not any(world["dest"].iterdir())
    assert all(" tar " not in c for c in remote_cmds(world))


@pytest.mark.parametrize("args", [("core_b", SID, "unlisted.json"), ("core_b", SID, "../x.json"),
                                  ("core_b/..", SID, "a.json"), ("core_b", "../" + SID, "a.json"), ("core_b", SID)])
def test_refuses_unlisted_or_unsafe_names_without_contacting_the_remote(world, args):
    res = world["run"](*args)
    assert res.returncode == 2
    assert remote_cmds(world) == []


def test_sha256_mismatch_is_kept_aside_and_fails(world):
    res = world["run"]("core_b", SID, "tampered.json")
    assert res.returncode == 5 and "sha256 mismatch" in res.stdout
    assert not (world["dest"] / "tampered.json").exists()
    assert (world["dest"] / "tampered.json.sha256-mismatch").read_bytes() == b"changed on the remote"


def test_c0_raw_is_git_ignored_and_c0_outputs_are_not():
    ignored = [f"results/capacity/c0-step0/raw/{SID}/bench-uniform-8192-r1-c8.json",
               "results/capacity/explore-c0-20261016/raw/b1-n1.bench.json",
               "results/capacity/smoke-c0-20261016/raw/b1.server.log"]
    res = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", *ignored], capture_output=True,
                         text=True)
    assert sorted(res.stdout.split()) == sorted(ignored)
    kept = ["results/capacity/c0-step0/step0.json", "results/capacity/explore-c0-20261016/rows.jsonl",
            "results/capacity/explore-c0-20261016/derived/b1-n1-requests.jsonl",
            "results/capacity/explore-c0-20261016/raw_manifest.json"]
    res = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", *kept], capture_output=True, text=True)
    assert res.stdout.strip() == ""
