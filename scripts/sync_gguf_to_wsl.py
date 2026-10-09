#!/usr/bin/env python3
"""Copy the pinned GGUFs from the Mac to the WSL copy: the 6 quants for Core A on the RTX (008), and with
--set references the 2 BF16 references for quality, which moved to the RTX (020).

Only `ssh gpu-server` is used (MACHINE_POLICY). Each file is streamed into a temporary name in
~/llm-serving-lab/models/gguf/ inside WSL, its sha256 is checked there against configs/pins.toml,
and only then renamed into place. Files already present with the pinned sha256 are skipped. The
Mac <-> desktop link is slow and drops; an interrupted copy keeps its partial file and the next
attempt resumes from its size. --max-mib-s caps the send rate so the copy does not saturate the
uplink that other work (git push, other projects) shares.

Usage (Mac):  uv run python scripts/sync_gguf_to_wsl.py [--set quants|references] [--max-mib-s 1.2] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REMOTE_DIR = "/home/wenn0113/llm-serving-lab/models/gguf"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
       "-o", "ServerAliveCountMax=6", "gpu-server", "wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", "-s"]
ATTEMPTS = 50
EX_TEMPFAIL = 75   # tools/with-mac-lock convention: try again later
MAX_BACKOFF_S = 600
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


def reference_files(pins: dict) -> list:
    """(filename, sha256, size_bytes) of the BF16 references, 9B first (020: quality on the RTX)."""
    out = []
    for m in sorted(pins["core_a"]["models"], key=lambda m: m["size"] != "9B"):
        ref = m["reference"]
        if not ref.get("sha256"):
            raise SystemExit(f"{ref['filename']} has no pinned sha256 yet")
        out.append((ref["filename"], ref["sha256"], ref["size_bytes"]))
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
RECEIVER_SCRIPT = f"""set -eu
name=$1; sha=$2; size=$3
{{ mkdir -p {REMOTE_DIR} && cd {REMOTE_DIR}; }} || {{ echo "NODIR {REMOTE_DIR}"; exit 3; }}
if [ "$size" = query ]; then
  if [ -f ".$name.part" ]; then stat -c %s ".$name.part"
  elif [ -f "$name" ] && [ "$(sha256sum "$name" | cut -d' ' -f1)" = "$sha" ]; then echo "DONE"
  else echo 0; fi
  exit 0
fi
cat >> ".$name.part"
have=$(stat -c %s ".$name.part")
if [ "$have" -lt "$size" ]; then echo "INCOMPLETE $name $have/$size"; exit 2; fi
got=$(sha256sum ".$name.part" | cut -d' ' -f1)
if [ "$got" = "$sha" ]; then mv -T ".$name.part" "$name"; echo "VERIFIED $name"
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


def send_from(path: Path, offset: int, cmd: list, *, max_bytes_per_s: float | None = None,
              clock=time.monotonic, sleep=time.sleep) -> subprocess.CompletedProcess:
    """Stream path[offset:] into the receiver's stdin, at most max_bytes_per_s when set, with at most ~1 s of burst
    credit: after a stall the copy does not catch up at line rate."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    size = CHUNK if not max_bytes_per_s else 256 * 2**10
    t0, sent = clock(), 0
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            while chunk := fh.read(size):
                proc.stdin.write(chunk)
                sent += len(chunk)
                if max_bytes_per_s:
                    now = clock()
                    ahead = sent / max_bytes_per_s - (now - t0)
                    if ahead > 0:
                        sleep(ahead)
                    elif ahead < -1.0:  # stalled: forfeit credit beyond ~1 s
                        t0 = now - sent / max_bytes_per_s - 1.0
    except BrokenPipeError:
        pass
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    out, err = proc.stdout.read(), proc.stderr.read()  # tiny outputs; communicate() would flush the closed stdin
    proc.wait()
    return subprocess.CompletedProcess(cmd, proc.returncode, out.decode(errors="replace"), err.decode(errors="replace"))


def copy_one(name: str, sha: str, size: int, models: Path, *, run=subprocess.run, send=send_from,
             attempts: int = ATTEMPTS, sleep=time.sleep, max_bytes_per_s: float | None = None):
    """True = verified on WSL; False = sha mismatch or the remote directory is unusable (do not retry);
    None = still failing after `attempts` tries with exponential backoff (temporary: run again later)."""
    k, last = 0, None
    for _ in range(attempts):
        q = run(receive_command(name, sha, "query"), capture_output=True, text=True)
        if "NODIR" in q.stdout:
            print(q.stdout.strip(), flush=True)
            return False
        if q.returncode == 0 and q.stdout.strip() == "DONE":  # verified and renamed; only the reply was lost
            print(f"VERIFIED {name} (already in place)", flush=True)
            return True
        have = int(q.stdout.strip()) if q.returncode == 0 and q.stdout.strip().isdigit() else None
        if have is not None:
            if last is not None and have > last:
                k = 0  # the partial grew since the last try: the link works, it only dropped
            last = have
            res = (send(models / name, have, receive_command(name, sha, size), max_bytes_per_s=max_bytes_per_s)
                   if max_bytes_per_s else send(models / name, have, receive_command(name, sha, size)))
            print((res.stdout.strip() or res.stderr.strip()[-200:]), flush=True)
            if res.returncode == 0 and f"VERIFIED {name}" in res.stdout:
                return True
            if "MISMATCH" in res.stdout or "NODIR" in res.stdout:
                return False
        sleep(min(MAX_BACKOFF_S, 15 * 2 ** k))
        k += 1
    return None


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
    parser.add_argument("--set", choices=["quants", "references"], default="quants")
    parser.add_argument("--max-mib-s", type=float, default=None, help="average send-rate cap in MiB/s")
    args = parser.parse_args(argv)
    if args.max_mib_s is not None and args.max_mib_s <= 0:
        parser.error("--max-mib-s must be > 0")
    root = Path(__file__).resolve().parent.parent
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    files = quant_files(pins) if args.set == "quants" else reference_files(pins)
    rate = args.max_mib_s * 2**20 if args.max_mib_s else None
    chk = subprocess.run(SSH, input=remote_script_check(files), capture_output=True, text=True)
    if chk.returncode != 0:  # the remote script always exits 0, so this is ssh / wsl.exe failing
        print(f"remote check failed (rc {chk.returncode}): {chk.stderr.strip()[-300:]}", file=sys.stderr)
        return EX_TEMPFAIL
    have = parse_check(chk.stdout)
    todo = [f for f in (copy_order(files) if args.set == "quants" else files) if have.get(f[0]) != f[1]]
    print(json.dumps({"present": sorted(n for n, s, _ in files if have.get(n) == s), "to_copy": [f[0] for f in todo],
                      "gib": round(sum(f[2] for f in todo) / 2**30, 2)}))
    if args.dry_run:
        return 0
    if todo:
        inst = subprocess.run(SSH, input=f"cat > {RECEIVER} <<'EOF'\n{RECEIVER_SCRIPT}EOF\n", capture_output=True,
                              text=True)
        if inst.returncode != 0:  # ssh / wsl.exe failing: temporary, like the remote check
            print(f"receiver install failed (rc {inst.returncode}): {inst.stderr.strip()[-300:]}", file=sys.stderr)
            return EX_TEMPFAIL
    for name, sha, size in todo:
        res = copy_one(name, sha, size, Path(args.models), max_bytes_per_s=rate)
        if res is None:
            return EX_TEMPFAIL
        if res is False:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
