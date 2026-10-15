"""tools/rtx/deploy_service.sh (decision 029 "WSL 的部署目錄"): build a bundle with scripts/deploy_bundle.py and extract
it under `flock -n /tmp/ai-portfolio-gpu.lock` into ~/llm-serving-service/releases/<sha>, refusing an existing release;
never writing ~/llm-serving-lab. Runs the script in a copy of the repo layout with a fake bundle builder, a fake `ssh`
that runs the remote command in a fake WSL home, and a fake `flock` (macOS has none)."""

import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "rtx" / "deploy_service.sh"
SHA = "0123456789abcdef0123456789abcdef01234567"
FAKE_SSH = """#!/usr/bin/env bash
# fake ssh: record the remote command, run it locally with HOME = the fake WSL home
cmd="${@: -1}"
printf '%s\\n' "$cmd" >> "$FAKE_SSH_LOG"
cmd="${cmd/wsl.exe -d Ubuntu-22.04 -- bash -lc/bash -c}"
export HOME="$FAKE_WSL_HOME"
eval "$cmd"
"""
FAKE_FLOCK = """#!/usr/bin/env bash
# fake flock -n LOCK cmd...: record; refuse when FAKE_LOCK_BUSY=1
printf '%s\\n' "$*" >> "$FAKE_FLOCK_LOG"
[ "$1" = "-n" ] || exit 64
[ "${FAKE_LOCK_BUSY:-0}" = 1 ] && exit 1
shift 2
exec "$@"
"""
FAKE_PYTHON = """#!/usr/bin/env bash
# fake .venv/bin/python scripts/deploy_bundle.py --out X: copy a small bundle there, print deploy_bundle's JSON line
[ "${FAKE_BUNDLE_REFUSE:-0}" = 1 ] && { echo "deploy_bundle: uncommitted changes" >&2; exit 2; }
cp "$FAKE_TAR" "$3"
echo "{\\"commit\\": \\"$FAKE_SHA\\", \\"files\\": 2, \\"out\\": \\"$3\\"}"
"""


def make_tar(path: Path) -> None:
    with tarfile.open(path, "w") as tar:
        for name, data in {"DEPLOYED_COMMIT": (SHA + "\n").encode(), "service/capserve/x.py": b"x = 1\n",
                           "DEPLOY_MANIFEST.json": json.dumps({"commit": SHA, "files": {}}).encode()}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def world(tmp_path):
    repo, home, bin_ = tmp_path / "repo", tmp_path / "home", tmp_path / "bin"
    (repo / "tools" / "rtx").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "tools" / "rtx" / "deploy_service.sh")
    (repo / "scripts" / "deploy_bundle.py").write_text("# placeholder\n")
    research = home / "llm-serving-lab"
    research.mkdir(parents=True)
    (research / "DEPLOYED_COMMIT").write_text("research\n")
    bin_.mkdir()
    for name, text in (("ssh", FAKE_SSH), ("flock", FAKE_FLOCK)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    py = repo / ".venv" / "bin" / "python"
    py.write_text(FAKE_PYTHON)
    py.chmod(0o755)
    tar = tmp_path / "bundle.tar"
    make_tar(tar)
    logs = {"ssh": tmp_path / "ssh.log", "flock": tmp_path / "flock.log"}
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_WSL_HOME": str(home),
           "FAKE_SSH_LOG": str(logs["ssh"]), "FAKE_FLOCK_LOG": str(logs["flock"]), "FAKE_TAR": str(tar),
           "FAKE_SHA": SHA}

    def run(**extra):
        return subprocess.run(["bash", str(repo / "tools" / "rtx" / "deploy_service.sh")],
                              env={**env, **extra}, capture_output=True, text=True)

    def snapshot():
        return sorted((str(p.relative_to(research)), p.stat().st_mtime_ns) for p in research.rglob("*"))

    return {"run": run, "home": home, "research": research, "logs": logs, "snapshot": snapshot}


def test_script_parses_never_uses_sudo_and_takes_the_lock_without_waiting():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    code = "\n".join(x.split("#", 1)[0] for x in SCRIPT.read_text().splitlines() if not x.lstrip().startswith("echo"))
    assert "sudo" not in code and "flock -n $LOCK" in code and "LOCK=/tmp/ai-portfolio-gpu.lock" in code
    assert "llm-serving-lab" not in code              # the research checkout is never a target
    assert "rm -rf ~" not in code and "--overwrite" not in code


def test_deploys_a_new_release_under_the_lock(world):
    before = world["snapshot"]()
    r = world["run"]()
    assert r.returncode == 0, r.stdout + r.stderr
    rel = world["home"] / "llm-serving-service" / "releases" / SHA
    assert (rel / "DEPLOYED_COMMIT").read_text() == SHA + "\n" and (rel / "service" / "capserve" / "x.py").is_file()
    assert world["snapshot"]() == before                     # ~/llm-serving-lab untouched
    assert world["logs"]["flock"].read_text().startswith("-n /tmp/ai-portfolio-gpu.lock ")
    assert f"deployed {SHA}" in r.stdout


def test_an_existing_release_is_never_overwritten(world):
    assert world["run"]().returncode == 0
    rel = world["home"] / "llm-serving-service" / "releases" / SHA
    (rel / "service" / "capserve" / "x.py").write_text("hot patch kept as evidence\n")
    r = world["run"]()
    assert r.returncode == 1 and "already exists" in r.stdout
    assert (rel / "service" / "capserve" / "x.py").read_text() == "hot patch kept as evidence\n"


def test_a_held_lock_deploys_nothing(world):
    r = world["run"](FAKE_LOCK_BUSY="1")
    assert r.returncode == 1 and not (world["home"] / "llm-serving-service" / "releases" / SHA).exists()


def test_a_symlinked_service_root_is_refused(world):
    os.symlink(world["research"], world["home"] / "llm-serving-service")
    before = world["snapshot"]()
    r = world["run"]()
    assert r.returncode == 1 and world["snapshot"]() == before
    assert not world["logs"]["flock"].exists()


def test_a_symlinked_releases_dir_is_refused(world):
    """releases -> ~/llm-serving-lab would land the extraction inside the research checkout (027 §5)."""
    svc = world["home"] / "llm-serving-service"
    svc.mkdir()
    os.symlink(world["research"], svc / "releases")
    before = world["snapshot"]()
    r = world["run"]()
    assert r.returncode == 1 and world["snapshot"]() == before
    assert not (world["research"] / SHA).exists() and "symlink" in r.stdout
    assert not world["logs"]["flock"].exists()


def test_a_refused_bundle_contacts_nothing(world):
    r = world["run"](FAKE_BUNDLE_REFUSE="1")
    assert r.returncode == 2 and not world["logs"]["ssh"].exists()
