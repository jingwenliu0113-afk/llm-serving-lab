#!/usr/bin/env python3
"""Read a GGUF header (metadata + tensor infos) with the stdlib, without loading weights.

Used to get the bytes of token_embd.weight per pinned GGUF (H-A1 / H-A2 subtract it for the
untied 9B; PREREGISTRATION §2, decisions 008 / 009). Block sizes follow gguf-py
GGML_QUANT_SIZES of the pinned llama.cpp.

Usage:
    uv run python scripts/gguf_tensors.py models/gguf/Qwen3.5-9B-Q4_K_M.gguf --tensor token_embd.weight
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

# ggml type id -> (name, block elements, block bytes)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 30: ("BF16", 1, 2),
}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _R:
    def __init__(self, fh):
        self.fh = fh

    def unpack(self, fmt):
        return struct.unpack(fmt, self.fh.read(struct.calcsize(fmt)))[0]

    def string(self):
        return self.fh.read(self.unpack("<Q")).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, n = self.unpack("<I"), self.unpack("<Q")
            if etype in _SCALAR:  # skip numeric arrays without materialising them
                self.fh.seek(n * struct.calcsize(_SCALAR[etype]), 1)
                return f"<array {n}>"
            items = [self.value(etype) for _ in range(n)]  # must be read to advance
            return items if n <= 16 else f"<array {n}>"
        return self.unpack(_SCALAR[vtype])


def read_header(path) -> dict:
    with open(path, "rb") as fh:
        r = _R(fh)
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = r.unpack("<I"), r.unpack("<Q"), r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.unpack("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            tid, offset = r.unpack("<I"), r.unpack("<Q")
            tname, blk, nbytes = GGML_TYPES.get(tid, (f"type{tid}", None, None))
            n = 1
            for d in dims:
                n *= d
            tensors.append({"name": name, "dims": dims, "type": tname, "offset": offset,
                            "bytes": n // blk * nbytes if blk else None})
    return {"version": version, "metadata": meta, "tensors": tensors}


def tensor_bytes(path, name: str) -> int | None:
    for t in read_header(path)["tensors"]:
        if t["name"] == name:
            return t["bytes"]
    return None


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gguf", nargs="+")
    parser.add_argument("--tensor", default="token_embd.weight")
    args = parser.parse_args(argv)
    for p in args.gguf:
        info = read_header(p)
        t = next((x for x in info["tensors"] if x["name"] == args.tensor), None)
        print(json.dumps({"file": Path(p).name, "tensor": args.tensor, "type": t and t["type"],
                          "bytes": t and t["bytes"], "has_output": any(x["name"] == "output.weight"
                                                                         for x in info["tensors"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
