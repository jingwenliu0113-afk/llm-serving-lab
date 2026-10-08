"""Tests for scripts/gguf_tensors.py: stdlib GGUF header reader for tensor byte sizes
(token_embd bytes feed H-A1 / H-A2, PREREGISTRATION §2)."""

import struct

import gguf_tensors as gt


def s(text):
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return s(key) + struct.pack("<I", 8) + s(val)


def kv_array_str(key, vals):
    return s(key) + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(vals)) + b"".join(s(v) for v in vals)


def kv_u32(key, v):
    return s(key) + struct.pack("<II", 4, v)


def tensor(name, dims, ggml_type, offset):
    return s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", ggml_type, offset)


def make_gguf(path):
    kvs = [kv_string("general.architecture", "qwen35"), kv_u32("qwen35.block_count", 32),
           kv_array_str("tokenizer.ggml.tokens", ["a", "b", "c"])]
    tensors = [tensor("token_embd.weight", [4096, 248320], 14, 0),       # Q6_K
               tensor("blk.0.attn_q.weight", [4096, 4096], 8, 999),     # Q8_0
               tensor("output_norm.weight", [4096], 0, 2000)]           # F32
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(tensors)
    path.write_bytes(data + b"\0" * 64)


def test_reads_tensor_bytes_and_metadata(tmp_path):
    p = tmp_path / "m.gguf"
    make_gguf(p)
    info = gt.read_header(p)
    assert info["metadata"]["general.architecture"] == "qwen35"
    assert info["metadata"]["qwen35.block_count"] == 32
    t = {x["name"]: x for x in info["tensors"]}
    assert t["token_embd.weight"]["type"] == "Q6_K"
    assert t["token_embd.weight"]["bytes"] == 4096 * 248320 // 256 * 210
    assert t["blk.0.attn_q.weight"]["bytes"] == 4096 * 4096 // 32 * 34
    assert t["output_norm.weight"]["bytes"] == 4096 * 4
    assert gt.tensor_bytes(p, "token_embd.weight") == 4096 * 248320 // 256 * 210
    assert gt.tensor_bytes(p, "output.weight") is None
