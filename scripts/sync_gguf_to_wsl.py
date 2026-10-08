#!/usr/bin/env python3
"""Copy the 6 pinned quantized GGUFs from the Mac to the WSL copy for Core A on the RTX (008).

Only `ssh gpu-server` is used (MACHINE_POLICY). Each file is streamed into a temporary name in
~/llm-serving-lab/models/gguf/ inside WSL, its sha256 is checked there against configs/pins.toml,
and only then renamed into place. Files already present with the pinned sha256 are skipped. The
BF16 references stay on the Mac (quality runs only there). The Mac <-> desktop link is slow and
drops; an interrupted copy keeps its partial file and the next attempt resumes from its size.

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
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
       "-o", "ServerAliveCountMax=6", "gpu-server", "wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", "-s"]
ATTEMPTS = 50
CHUNK = 4 * 2**20


def quant_files(pins: dict) -> list:
    """(filename, sha256, size_bytes) of every pinned quantized GGUF (not the BF16 references)."""
    out = []
    for m in pins["core_a"]["models"]:
        for q, f in m["files"].items():
            if not f.get("sha256"):
                raise SystemExit(f"{f['filename']} has no pinned sha256 yet")
            out.append((f["filename"], f["sha256"], f["size_bytes"]))
    return out


# Stage B needs these first: the C_ctx / sentinel anchor (009) and the sysmem-overflow model (008).
FIRST = ("Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-9B-Q8_0.gguf")


def copy_order(files: list) -> list:
    return sorted(files, key=lambda f: FIRST.index(f[0]) if f[0] in FIRST else len(FIRST))


def remote_script_check(files: list) -> str:
    lines = [f"cd {shlex.quote(REMOTE_DIR)} 2>/dev/null || exit 0"]
    for name, _, _ in files:
        q = shlex.quote(name)
        lines.append(f"[ -f {q} ] && sha256sum {q}")
    return "\n".join(lines) + "\nexit 0\n"


RECEIVER = "/tmp/llmsl-recv-gguf.sh"
RECEIVER_SCRIPT = f"""set -u
name=$1; sha=$2; size=$3
mkdir -p {REMOTE_DIR}
cd {REMOTE_DIR}
if [ "$size" = query ]; then stat -c %s ".$name.part" 2>/dev/null || echo 0; exit 0; fi
cat >> ".$name.part"
have=$(stat -c %s ".$name.part")
if [ "$have" -lt "$size" ]; then echo "INCOMPLETE $name $have/$size"; exit 2; fi
got=$(sha256sum ".$name.part" | cut -d' ' -f1)
if [ "$got" = "$sha" ]; then mv ".$name.part" "$name"; echo "VERIFIED $name"
else rm -f ".$name.part"; echo "MISMATCH $name $got"; exit 1; fi
"""
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.gguf$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def receive_command(name: str, sha256: str, size) -> list:
    """ssh argv for the receiver (arguments carry no quotes: they cross cmd.exe). size = expected bytes, or
    'query' to print the partial file's current size."""
    if not _SAFE_NAME.match(name) or not _SHA.match(sha256) or not (size == "query" or str(size).isdigit()):
        raise ValueError(f"refusing unsafe name / sha / size: {name!r} {sha256!r} {size!r}")
    return SSH[:-2] + ["bash", RECEIVER, name, sha256, str(size)]


def send_from(path: Path, offset: int, cmd: list) -> subprocess.CompletedProcess:
    """Stream path[offset:] into the receiver's stdin."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            while chunk := fh.read(CHUNK):
                proc.stdin.write(chunk)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    out, err = proc.communicate()
    return subprocess.CompletedProcess(cmd, proc.returncode, out.decode(errors="replace"), err.decode(errors="replace"))


def copy_one(name: str, sha: str, size: int, models: Path, *, run=subprocess.run, send=send_from,
             attempts: int = ATTEMPTS) -> bool:
    for _ in range(attempts):
        q = run(receive_command(name, sha, "query"), capture_output=True, text=True)
        have = int(q.stdout.strip() or 0) if q.returncode == 0 and q.stdout.strip().isdigit() else None
        if have is None:
            continue  # link down; try again
        res = send(models / name, have, receive_command(name, sha, size))
        print((res.stdout.strip() or res.stderr.strip()[-200:]), flush=True)
        if res.returncode == 0:
            return True
        if "MISMATCH" in res.stdout:
            return False
    return False


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
    todo = copy_order([f for f in files if have.get(f[0]) != f[1]])
    print(json.dumps({"present": sorted(n for n, s, _ in files if have.get(n) == s), "to_copy": [f[0] for f in todo],
                      "gib": round(sum(f[2] for f in todo) / 2**30, 2)}))
    if args.dry_run:
        return 0
    if todo:
        subprocess.run(SSH, input=f"cat > {RECEIVER} <<'EOF'\n{RECEIVER_SCRIPT}EOF\n", text=True, check=True)
    for name, sha, size in todo:
        if not copy_one(name, sha, size, Path(args.models)):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
