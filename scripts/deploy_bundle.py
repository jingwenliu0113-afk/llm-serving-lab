#!/usr/bin/env python3
"""Build the bundle that deploys this repo to the remote WSL copy (no .git there).

The bundle is `git archive HEAD` (minus results/, which is produced remotely)
plus DEPLOYED_COMMIT (full sha) and DEPLOY_MANIFEST.json (sha256 of every
deployed file). scripts/env_capture.py verifies the manifest on the remote, so
a hot-patched file shows up as dirty instead of being attributed to the commit.
Refuses when tracked files have uncommitted changes.

Usage (Mac):
    uv run python scripts/deploy_bundle.py --out <scratch>/bundle.tar
    ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- bash -lc "tar -x -C ~/llm-serving-lab"' < <scratch>/bundle.tar
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

EXCLUDE = ("results",)


class DeployError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def build_bundle(root: Path, out: Path) -> dict:
    root, out = Path(root), Path(out)
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise DeployError(f"uncommitted changes in tracked files; commit first:\n{dirty}")
    commit = _git(root, "rev-parse", "HEAD").strip()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "head.tar"
        pathspec = ["."] + [f":(exclude){p}" for p in EXCLUDE]
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar", "-o", str(archive), "HEAD", "--",
                        *pathspec], check=True)
        files = {}
        with tarfile.open(archive) as src, tarfile.open(out, "w") as dst:
            for member in src.getmembers():
                data = src.extractfile(member).read() if member.isfile() else None
                dst.addfile(member, io.BytesIO(data) if data is not None else None)
                if data is not None:
                    files[member.name] = hashlib.sha256(data).hexdigest()
            manifest = {"commit": commit, "files": dict(sorted(files.items()))}
            _add_bytes(dst, "DEPLOYED_COMMIT", (commit + "\n").encode())
            _add_bytes(dst, "DEPLOY_MANIFEST.json", (json.dumps(manifest, indent=1) + "\n").encode())
    return manifest


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_bundle(Path(__file__).resolve().parent.parent, Path(args.out))
    except DeployError as exc:
        print(f"deploy_bundle: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"commit": manifest["commit"], "files": len(manifest["files"]), "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
