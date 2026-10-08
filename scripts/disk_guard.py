#!/usr/bin/env python3
"""Refuse to create a large artifact unless the reserve stays free afterwards.

docs/STORAGE_PLAN.md: run before every download, conversion, quantization or
KLD-base generation. Stdlib only (Mac and WSL).

Usage:
    python scripts/disk_guard.py --path models/gguf --need-gib 18 [--reserve-gib 50]
Exit 0 = ok, 3 = not enough space (JSON report on stdout either way).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

DEFAULT_RESERVE_GIB = 50
GiB = 1024**3


def _existing(path: str) -> str:
    p = Path(path).absolute()
    while not p.exists():
        p = p.parent
    return str(p)


def check(path: str, *, need_bytes: int, reserve_bytes: int,
          disk_usage: Callable = shutil.disk_usage) -> dict:
    if need_bytes < 0 or reserve_bytes < 0:
        raise ValueError("need and reserve must be non-negative")
    free = disk_usage(_existing(path)).free
    after = free - need_bytes
    return {
        "path": str(path),
        "free_gib": round(free / GiB, 2),
        "need_gib": round(need_bytes / GiB, 2),
        "reserve_gib": round(reserve_bytes / GiB, 2),
        "after_gib": round(after / GiB, 2),
        "ok": after >= reserve_bytes,
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True)
    parser.add_argument("--need-gib", required=True, type=float)
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    args = parser.parse_args(argv)
    report = check(args.path, need_bytes=int(args.need_gib * GiB), reserve_bytes=int(args.reserve_gib * GiB))
    print(json.dumps(report))
    if not report["ok"]:
        print("disk_guard: not enough free space; free scratch tiers first (docs/STORAGE_PLAN.md)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
