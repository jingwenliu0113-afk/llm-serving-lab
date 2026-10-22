#!/usr/bin/env python3
"""Produce E3's two new GGUFs (decision 041 v3 §3.2): 4B-Q4_0 and 9B-Q4_0, quantized with llama.cpp b11146's
llama-quantize (default recipe, no --pure) from the pinned BF16 GGUFs (pins core_a.models.reference), on the Mac under
the Mac heavy lock.

- The BF16 input is checked against its pin (sha256 and size) and the toolchain against pins [llama_cpp]
  (produce_gguf.check_toolchain); the output is written to <name>.tmp and renamed only after llama-quantize exits 0.
- disk_guard keeps the 50 GiB reserve; an existing output is never overwritten.
- It prints the entry to pin in configs/stage_e3.toml (sha256, size, token_embd bytes, tie, tensor type counts read from
  the header with scripts/gguf_tensors.py). It never writes pins.toml or the config itself.

Usage:
    tools/with-mac-lock uv run python scripts/produce_gguf_e3.py --size 9B
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import disk_guard  # noqa: E402
import gguf_tensors  # noqa: E402
import produce_gguf as pg  # noqa: E402

QUANT = "Q4_0"
QUANT_FACTOR = 0.35                                    # Q4_0 output / BF16 input, generous (4.5 bits + Q6_K output)


class ProduceE3Error(RuntimeError):
    pass


def target(pins: dict, size: str) -> dict:
    m = next((m for m in pins["core_a"]["models"] if m["size"] == size), None)
    if m is None:
        raise ProduceE3Error(f"no {size} model in pins core_a")
    return {"size": size, "source": m["reference"]["filename"], "source_sha256": m["reference"]["sha256"],
            "source_bytes": m["reference"]["size_bytes"], "output": f"Qwen3.5-{size}-{QUANT}.gguf"}


def header_entry(path, sha256: str) -> dict:
    """The configs/stage_e3.toml entry of a produced GGUF, from its header."""
    info = gguf_tensors.read_header(path)
    names = {t["name"]: t for t in info["tensors"]}
    if "token_embd.weight" not in names:
        raise ProduceE3Error(f"{path}: no token_embd.weight")
    return {"filename": Path(path).name, "sha256": sha256, "size_bytes": Path(path).stat().st_size,
            "token_embd_bytes": names["token_embd.weight"]["bytes"], "tied": "output.weight" not in names,
            "tensor_types": dict(sorted(collections.Counter(t["type"] for t in info["tensors"]).items()))}


def toml_entry(key: str, e: dict) -> str:
    types = ", ".join(f'{k} = {v}' for k, v in e["tensor_types"].items())
    return "\n".join([f"[e3.files.{key}]", f'filename = "{e["filename"]}"', f'sha256 = "{e["sha256"]}"',
                      f'size_bytes = {e["size_bytes"]}', f'token_embd_bytes = {e["token_embd_bytes"]}',
                      f'tied = {"true" if e["tied"] else "false"}', f"tensor_types = {{ {types} }}"])


def produce(size: str, *, pins: dict, models: Path, lock_held: bool, quantize: str, toolchain: dict,
            run: Callable = pg._default_run, hash_file: Callable = pg._sha256,
            disk_ok: Callable = pg._default_disk_ok, size_of: Callable = lambda p: Path(p).stat().st_size) -> dict:
    if not lock_held:
        raise ProduceE3Error("GGUF production is a Mac heavy workload: run it under tools/with-mac-lock")
    pg.check_toolchain(toolchain, pins)
    t = target(pins, size)
    src, out = Path(models) / t["source"], Path(models) / t["output"]
    if out.exists():
        raise ProduceE3Error(f"{out} exists; never overwritten (delete it by hand only after checking why)")
    if not src.is_file() or size_of(src) != t["source_bytes"] or hash_file(src) != t["source_sha256"]:
        raise ProduceE3Error(f"{src} is not the pinned BF16 reference (size / sha256)")
    need_gib = t["source_bytes"] * QUANT_FACTOR / disk_guard.GiB
    guard = disk_ok(models, need_gib)
    if not guard.get("ok"):
        raise ProduceE3Error(f"disk guard failed: {guard}")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    code = run([quantize, str(src), str(tmp), QUANT])
    if code != 0 or not tmp.is_file():
        if tmp.exists():
            tmp.unlink()
        raise ProduceE3Error(f"llama-quantize exited {code}")
    os.replace(tmp, out)
    return header_entry(out, hash_file(out))


def main(argv: list | None = None) -> int:  # pragma: no cover - needs llama-quantize and the BF16 GGUFs
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", required=True, choices=["4B", "9B"])
    parser.add_argument("--models", default=str(ROOT / "models" / "gguf"))
    args = parser.parse_args(argv)
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    try:
        e = produce(args.size, pins=pins, models=Path(args.models),
                    lock_held=os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") == "1",
                    quantize=str(ROOT / "third_party" / "llama.cpp" / "build-metal" / "bin" / "llama-quantize"),
                    toolchain=pg._default_toolchain(ROOT))
    except (ProduceE3Error, pg.ProduceError) as exc:
        print(f"produce_gguf_e3: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(e))
    print(toml_entry(f"{args.size}-{QUANT}", e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
