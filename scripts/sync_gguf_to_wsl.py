#!/usr/bin/env python3
"""Copy the 6 pinned quantized GGUFs from the Mac to the WSL copy for Core A on the RTX (008).

Only `ssh gpu-server` is used (MACHINE_POLICY). Each file is streamed into a temporary name in
~/llm-serving-lab/models/gguf/ inside WSL, its sha256 is checked there against configs/pins.toml,
and only then renamed into place. Files already present with the pinned sha256 are skipped. The
BF16 references stay on the Mac (quality runs only there).

Usage (Mac):  uv run python scripts/sync_gguf_to_wsl.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

REMOTE_DIR = "/home/wenn0113/llm-serving-lab/models/gguf"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "gpu-server", "wsl.exe", "-d", "Ubuntu-22.04",
       "--", "bash", "-s"]


def quant_files(pins: dict) -> list:
    """(filename, sha256, size_bytes) of every pinned quantized GGUF (not the BF16 references)."""
    out = []
    for m in pins["core_a"]["models"]:
        for q, f in m["files"].items():
            if not f.get("sha256"):
                raise SystemExit(f"{f['filename']} has no pinned sha256 yet")
            out.append((f["filename"], f["sha256"], f["size_bytes"]))
    return out


def remote_script_check(files: list) -> str:
    lines = [f"cd {shlex.quote(REMOTE_DIR)} 2>/dev/null || exit 0"]
    for name, _, _ in files:
        q = shlex.quote(name)
        lines.append(f"[ -f {q} ] && sha256sum {q}")
    return "\n".join(lines) + "\nexit 0\n"


RECEIVER = "/tmp/llmsl-recv-gguf.sh"
RECEIVER_SCRIPT = f"""set -e
name=$1; sha=$2
mkdir -p {REMOTE_DIR}
cd {REMOTE_DIR}
cat > ".$name.part"
got=$(sha256sum ".$name.part" | cut -d' ' -f1)
if [ "$got" = "$sha" ]; then mv ".$name.part" "$name"; echo "VERIFIED $name"
else rm -f ".$name.part"; echo "MISMATCH $name $got"; exit 1; fi
"""
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.gguf$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def receive_command(name: str, sha256: str) -> list:
    """ssh argv that streams stdin into <name> on WSL (arguments carry no quotes: they cross cmd.exe)."""
    if not _SAFE_NAME.match(name) or not _SHA.match(sha256):
        raise ValueError(f"refusing unsafe name / sha: {name!r} {sha256!r}")
    return SSH[:-2] + ["bash", RECEIVER, name, sha256]


def parse_check(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            out[parts[1]] = parts[0]
    return out


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", default="models/gguf")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    files = quant_files(pins)
    have = parse_check(subprocess.run(SSH, input=remote_script_check(files), capture_output=True, text=True).stdout)
    todo = [f for f in files if have.get(f[0]) != f[1]]
    print(json.dumps({"present": sorted(n for n, s, _ in files if have.get(n) == s), "to_copy": [f[0] for f in todo],
                      "gib": round(sum(f[2] for f in todo) / 2**30, 2)}))
    if args.dry_run:
        return 0
    if todo:
        subprocess.run(SSH, input=f"cat > {RECEIVER} <<'EOF'\n{RECEIVER_SCRIPT}EOF\n", text=True, check=True)
    for name, sha, _ in todo:
        with open(Path(args.models) / name, "rb") as fh:
            res = subprocess.run(receive_command(name, sha), stdin=fh, capture_output=True, text=True)
        print(res.stdout.strip() or res.stderr.strip()[-300:])
        if res.returncode != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
