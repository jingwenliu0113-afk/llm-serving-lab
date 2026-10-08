#!/usr/bin/env python3
"""Produce the Core A GGUF artifacts for one size (decision 004 / 008).

    BF16 GGUF   = convert_hf_to_gguf.py <pinned snapshot> --outtype bf16 --no-mtp
    Q8_0/Q6_K/Q4_K_M = llama-quantize <BF16> <out> <Q>     (no imatrix)

Before anything runs: source shards must match pins.toml sha256, the Mac heavy
lock must not be held by another job, and disk_guard must pass. Every output
gets its sha256/size written back into pins.toml and a recipe
results/artifacts/<filename>.json (inputs, exact argv, toolchain, output hash),
so scratch tiers can be deleted and rebuilt bit-for-bit (docs/STORAGE_PLAN.md).
Deleting the source safetensors is a separate, explicit command.

Usage:
    uv run python scripts/produce_gguf.py produce --size 4B
    uv run python scripts/produce_gguf.py delete-source --size 4B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_guard  # noqa: E402
import env_capture  # noqa: E402

GiB = 1024**3
MAC_HEAVY_LOCK = "/tmp/ai-portfolio-mac-heavy.lock"
# BF16 GGUF ~ source shard bytes; Q8_0 + Q6_K + Q4_K_M ~ 1.24 x BF16 (decision 008 sizes).
NEED_FACTOR = 2.3


class ProduceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _model(pins: dict, size: str) -> dict:
    for m in pins["core_a"]["models"]:
        if m["size"] == size:
            return m
    raise ProduceError(f"size {size!r} not in pins.toml core_a.models")


def source_dir(model: dict) -> Path:
    org, name = model["source_repo"].split("/")
    return Path("models") / "hf" / f"{org}__{name}@{model['source_revision'][:8]}"


def plan_steps(pins: dict, size: str, root: Path) -> list:
    m = _model(pins, size)
    gguf = root / "models" / "gguf"
    bf16 = gguf / m["reference"]["filename"]
    steps = [{
        "kind": "convert", "quant": None, "output": m["reference"]["filename"],
        "argv": [str(root / "third_party" / "convert-venv" / "bin" / "python"),
                 str(root / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"),
                 str(root / source_dir(m)), "--outtype", "bf16", "--no-mtp", "--outfile", str(bf16)],
    }]
    quantize = root / "third_party" / "llama.cpp" / "build-metal" / "bin" / "llama-quantize"
    for q in pins["core_a"]["quants"]:
        out = gguf / m["files"][q]["filename"]
        steps.append({"kind": "quantize", "quant": q, "output": out.name,
                      "argv": [str(quantize), str(bf16), str(out), q]})
    return steps


def update_pins_text(text: str, filename: str, sha256: str, size: int) -> str:
    pattern = re.compile(
        r'(filename = "' + re.escape(filename) + r'"[^\n]*\n)sha256 = "([0-9a-f]*)"([^\n]*\n)size_bytes = (\d+)'
    )
    m = pattern.search(text)
    if m is None:
        raise ProduceError(f"{filename} not found in pins.toml")
    if m.group(2) and m.group(2) != sha256:
        raise ProduceError(f"{filename} already pinned with a different sha256 ({m.group(2)})")
    return text[:m.start()] + f'{m.group(1)}sha256 = "{sha256}"{m.group(3)}size_bytes = {size}' + text[m.end():]


def _default_lock_held_by_other() -> bool:
    return os.path.exists(MAC_HEAVY_LOCK) and os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") != "1"


def _default_disk_ok(path, need_gib: float) -> dict:
    return disk_guard.check(str(path), need_bytes=int(need_gib * GiB),
                            reserve_bytes=disk_guard.DEFAULT_RESERVE_GIB * GiB)


def _default_run(argv: list) -> int:
    return subprocess.run(argv).returncode


def _default_toolchain(root: Path) -> dict:
    res = env_capture.default_run([str(root / "third_party/llama.cpp/build-metal/bin/llama-completion"), "--version"])
    parsed = env_capture.parse_llama_version((res.stderr + res.stdout) if res else "") or {}
    return {"llama_cpp": f"{parsed.get('version')} (build {parsed.get('build_number')}, commit {parsed.get('commit')})"}


def produce(
    *,
    size: str,
    root: Path,
    run: Callable = _default_run,
    shard_sha256_override: dict | None = None,
    toolchain: dict | None = None,
    now: Callable = lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    lock_held_by_other: Callable = _default_lock_held_by_other,
    disk_ok: Callable = _default_disk_ok,
) -> dict:
    root = Path(root)
    pins_path = root / "configs" / "pins.toml"
    pins = tomllib.loads(pins_path.read_text())
    m = _model(pins, size)
    src = root / source_dir(m)

    expected = shard_sha256_override or m["source_shards"]
    for name, want in expected.items():
        path = src / name
        if not path.is_file() or _sha256(path) != want:
            raise ProduceError(f"source shard {name} missing or sha256 mismatch (pinned {want})")

    if lock_held_by_other():
        raise ProduceError(f"Mac heavy lock is held by another job ({MAC_HEAVY_LOCK}); retry later (MACHINE_POLICY)")

    need_gib = NEED_FACTOR * sum((src / n).stat().st_size for n in expected) / GiB
    gguf_dir = root / "models" / "gguf"
    guard = disk_ok(gguf_dir, need_gib)
    if not guard.get("ok"):
        raise ProduceError(f"not enough disk for {need_gib:.1f} GiB (docs/STORAGE_PLAN.md): {guard}")

    gguf_dir.mkdir(parents=True, exist_ok=True)
    recipes = root / "results" / "artifacts"
    recipes.mkdir(parents=True, exist_ok=True)
    tc = dict(toolchain or _default_toolchain(root))
    tc["converter_lock_sha256"] = _sha256(root / "configs" / "converter.lock.txt")
    tc["produce_python"] = platform.python_version()

    artifacts, bf16 = [], None
    for step in plan_steps(pins, size, root):
        started = now()
        code = run(step["argv"])
        out = gguf_dir / step["output"]
        if code != 0 or not out.is_file():
            raise ProduceError(f"{step['kind']} failed (exit {code}): {' '.join(step['argv'])}")
        digest, nbytes = _sha256(out), out.stat().st_size
        if step["kind"] == "convert":
            inputs = [{"repo": m["source_repo"], "revision": m["source_revision"], "shards": dict(expected)}]
            bf16 = {"filename": out.name, "sha256": digest}
        else:
            inputs = [bf16]
        recipe = {
            "schema_version": 1, "kind": step["kind"], "size": size, "quant": step["quant"],
            "inputs": inputs, "argv": step["argv"], "toolchain": tc,
            "output": {"filename": out.name, "sha256": digest, "size_bytes": nbytes},
            "started_at": started, "finished_at": now(), "disk_guard": guard,
        }
        (recipes / f"{out.name}.json").write_text(json.dumps(recipe, indent=1) + "\n")
        pins_path.write_text(update_pins_text(pins_path.read_text(), out.name, digest, nbytes))
        artifacts.append({"filename": out.name, "quant": step["quant"], "sha256": digest,
                          "size_bytes": nbytes, "argv": step["argv"]})
    return {"size": size, "artifacts": artifacts}


def delete_source(*, size: str, root: Path) -> list:
    """Delete the source safetensors once every derived artifact is pinned and has a recipe."""
    root = Path(root)
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    m = _model(pins, size)
    entries = [m["reference"], *m["files"].values()]
    for e in entries:
        if not e["sha256"] or not (root / "results" / "artifacts" / f"{e['filename']}.json").is_file():
            raise ProduceError(f"{e['filename']}: sha256 in pins and recipe are required before deleting the source")
    removed = []
    for name in m["source_shards"]:
        path = root / source_dir(m) / name
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=["produce", "delete-source"])
    parser.add_argument("--size", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    try:
        if args.cmd == "produce":
            print(json.dumps(produce(size=args.size, root=root), indent=1))
        else:
            print(json.dumps({"removed": delete_source(size=args.size, root=root)}, indent=1))
    except ProduceError as exc:
        print(f"produce_gguf: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
