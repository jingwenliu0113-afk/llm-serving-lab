#!/usr/bin/env python3
"""Produce E3's two new GGUFs (decision 041 v3 §3.2): 4B-Q4_0 and 9B-Q4_0, quantized with llama.cpp b11146's
llama-quantize (default recipe, no --pure) from the pinned BF16 GGUFs (pins core_a.models.reference), on the Mac under
the Mac heavy lock (owner verified, the env flag alone is not trusted); then copy them to WSL.

- The BF16 input is checked against its pin (sha256 and size) and the toolchain against pins [llama_cpp]
  (produce_gguf.check_toolchain); the output is written to <name>.tmp and renamed only after llama-quantize exits 0.
- disk_guard keeps the 50 GiB reserve; an existing output is never overwritten.
- It prints the entry to pin in configs/stage_e3.toml, read from the header with scripts/gguf_tensors.py (header_facts:
  token_embd bytes and type, tie, output.weight's type, tensor type counts, model_size = the tensors' bytes, which is
  llama-bench's model_size; model_type = llama-bench's "<arch> <size> <ftype>"). It never writes pins.toml or the
  config itself.
- sync copies the pinned Q4_0s to WSL with sync_gguf_to_wsl's receiver (sha256 and size from configs/stage_e3.toml,
  verified on WSL before the rename).

Usage:
    tools/with-mac-lock uv run python scripts/produce_gguf_e3.py produce --size 9B
    uv run python scripts/produce_gguf_e3.py sync                       # after both entries are pinned
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import disk_guard  # noqa: E402
import gguf_tensors  # noqa: E402
import produce_gguf as pg  # noqa: E402
import sync_gguf_to_wsl as sgw  # noqa: E402

QUANT = "Q4_0"
NEW = ("4B-Q4_0", "9B-Q4_0")
QUANT_FACTOR = 0.35                                    # Q4_0 output / BF16 input, generous (4.5 bits + Q6_K output)
# llama.cpp's llama_model_ftype_name for general.file_type (checked against the six Core A GGUFs' llama-bench rows)
FTYPE_NAMES = {2: "Q4_0", 7: "Q8_0", 15: "Q4_K - Medium", 18: "Q6_K"}


class ProduceE3Error(RuntimeError):
    pass


def target(pins: dict, size: str) -> dict:
    m = next((m for m in pins["core_a"]["models"] if m["size"] == size), None)
    if m is None:
        raise ProduceE3Error(f"no {size} model in pins core_a")
    return {"size": size, "source": m["reference"]["filename"], "source_sha256": m["reference"]["sha256"],
            "source_bytes": m["reference"]["size_bytes"], "output": f"Qwen3.5-{size}-{QUANT}.gguf"}


def header_facts(info: dict) -> dict:
    """What configs/stage_e3.toml pins of a GGUF and stage_e3.run_e3 checks before a session, from its header."""
    names = {t["name"]: t for t in info["tensors"]}
    embd, out = names.get("token_embd.weight") or {}, names.get("output.weight")
    return {"token_embd_bytes": embd.get("bytes"), "token_embd_type": embd.get("type"), "tied": out is None,
            "output_type": "" if out is None else out["type"],
            "tensor_types": dict(sorted(collections.Counter(t["type"] for t in info["tensors"]).items())),
            "model_size": sum(t["bytes"] or 0 for t in info["tensors"])}


def model_type(info: dict, size: str) -> str:
    ftype = info["metadata"].get("general.file_type")
    if ftype not in FTYPE_NAMES:
        raise ProduceE3Error(f"general.file_type {ftype!r} has no known llama.cpp name")
    return f"{info['metadata'].get('general.architecture')} {size} {FTYPE_NAMES[ftype]}"


def header_entry(path, sha256: str, size: str) -> dict:
    """The configs/stage_e3.toml entry of a produced GGUF, from its header."""
    info = gguf_tensors.read_header(path)
    facts = header_facts(info)
    if facts["token_embd_type"] is None:
        raise ProduceE3Error(f"{path}: no token_embd.weight")
    return {"filename": Path(path).name, "sha256": sha256, "size_bytes": Path(path).stat().st_size, **facts,
            "model_type": model_type(info, size)}


def toml_entry(key: str, e: dict) -> str:
    types = ", ".join(f'{k} = {v}' for k, v in e["tensor_types"].items())
    return "\n".join([f"[e3.files.{key}]", f'filename = "{e["filename"]}"', f'sha256 = "{e["sha256"]}"',
                      f'size_bytes = {e["size_bytes"]}', f'token_embd_bytes = {e["token_embd_bytes"]}',
                      f'token_embd_type = "{e["token_embd_type"]}"', f'tied = {"true" if e["tied"] else "false"}',
                      f'output_type = "{e["output_type"]}"', f"tensor_types = {{ {types} }}",
                      f'model_size = {e["model_size"]}', f'model_type = "{e["model_type"]}"'])


def produce(size: str, *, pins: dict, models: Path, lock_held: bool, lock_check: Callable, quantize: str,
            toolchain: dict, run: Callable = pg._default_run, hash_file: Callable = pg._sha256,
            disk_ok: Callable = pg._default_disk_ok, size_of: Callable = lambda p: Path(p).stat().st_size) -> dict:
    if not (lock_held and lock_check()):
        raise ProduceE3Error("GGUF production is a Mac heavy workload: run it under tools/with-mac-lock "
                             "(its owner is verified)")
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
    return header_entry(out, hash_file(out), size)


# ---------------------------------------------------------------- copy to WSL

def pinned_files(cfg: dict) -> list:
    """[(filename, sha256, size)] of the two Q4_0s from configs/stage_e3.toml; refuses before both are pinned."""
    out = []
    for key in NEW:
        f = cfg["e3"]["files"][key]
        if len(str(f.get("sha256") or "")) != 64 or not f.get("size_bytes"):
            raise ProduceE3Error(f"e3.files.{key} is not pinned yet")
        out.append((f["filename"], f["sha256"], f["size_bytes"]))
    return out


def sync(files: list, models: Path, *, run: Callable = subprocess.run, copy: Callable = sgw.copy_one) -> int:
    """sync_gguf_to_wsl's protocol for these files: check what WSL has, install the receiver, copy and verify each
    missing one. 0 = all verified; 1 = a mismatch (do not retry); EX_TEMPFAIL = try again later."""
    chk = run(sgw.SSH, input=sgw.remote_script_check(files), capture_output=True, text=True)
    if chk.returncode != 0:
        print(f"remote check failed (rc {chk.returncode}): {chk.stderr.strip()[-300:]}", file=sys.stderr)
        return sgw.EX_TEMPFAIL
    have = sgw.parse_check(chk.stdout)
    todo = [f for f in files if have.get(f[0]) != f[1]]
    print(json.dumps({"present": sorted(n for n, s, _ in files if have.get(n) == s), "to_copy": [f[0] for f in todo]}))
    if todo:
        inst = run(sgw.SSH, input=f"cat > {sgw.RECEIVER} <<'EOF'\n{sgw.RECEIVER_SCRIPT}EOF\n", capture_output=True,
                   text=True)
        if inst.returncode != 0:
            print(f"receiver install failed (rc {inst.returncode}): {inst.stderr.strip()[-300:]}", file=sys.stderr)
            return sgw.EX_TEMPFAIL
    for name, sha, size in todo:
        res = copy(name, sha, size, Path(models))
        if res is None:
            return sgw.EX_TEMPFAIL
        if res is False:
            return 1
    return 0


def main(argv: list | None = None) -> int:  # pragma: no cover - needs llama-quantize, the BF16 GGUFs or WSL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("produce")
    p.add_argument("--size", required=True, choices=["4B", "9B"])
    p.add_argument("--models", default=str(ROOT / "models" / "gguf"))
    p = sub.add_parser("sync")
    p.add_argument("--models", default=str(ROOT / "models" / "gguf"))
    args = parser.parse_args(argv)
    pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
    try:
        if args.cmd == "sync":
            cfg = tomllib.loads((ROOT / "configs" / "stage_e3.toml").read_text())
            return sync(pinned_files(cfg), Path(args.models))
        sys.path.insert(0, str(ROOT / "benchmarks"))
        import stage_e3
        e = produce(args.size, pins=pins, models=Path(args.models),
                    lock_held=os.environ.get("LLM_SERVING_LAB_MAC_LOCK_HELD") == "1",
                    lock_check=stage_e3.mac_lock_owned,
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
