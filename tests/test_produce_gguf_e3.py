"""Tests for scripts/produce_gguf_e3.py (decision 041 v3 §3.2): E3's Q4_0 GGUFs from the pinned BF16 references, the
Mac lock, the toolchain and input checks, atomic output and the configs/stage_e3.toml entry read from the header."""

import struct
import tomllib
from pathlib import Path

import pytest

import produce_gguf_e3 as pe

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
TOOLCHAIN = {"llama_cpp": {"build_number": PINS["llama_cpp"]["build_number"], "commit": PINS["llama_cpp"]["commit"]},
             "llama_cpp_tree": {"head": PINS["llama_cpp"]["commit"], "dirty": False}}


def tiny_gguf(path, tensors):
    """A header-only GGUF: tensors = [(name, dims, ggml type id)]."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b
    data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", 0)
    for name, dims, tid in tensors:
        data += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        data += struct.pack("<I", tid) + struct.pack("<Q", 0)
    Path(path).write_bytes(data)


def fake_quantize(tensors):
    def run(argv):
        assert argv[-1] == "Q4_0" and argv[2].endswith(".tmp")
        tiny_gguf(argv[2], tensors)
        return 0
    return run


def setup(tmp_path, size="9B"):
    t = pe.target(PINS, size)
    models = tmp_path / "gguf"
    models.mkdir()
    (models / t["source"]).write_bytes(b"x" * 16)
    return models, t


def produce(models, t, **kw):
    args = dict(pins=PINS, models=models, lock_held=True, quantize="/q", toolchain=TOOLCHAIN,
                run=fake_quantize([("token_embd.weight", [256, 64], 2), ("output.weight", [256, 64], 14),
                                   ("blk.0.attn_q.weight", [256, 256], 2), ("blk.0.norm.weight", [256], 0)]),
                hash_file=lambda p: t["source_sha256"] if Path(p).name == t["source"] else "f" * 64,
                disk_ok=lambda path, need: {"ok": True},
                size_of=lambda p: t["source_bytes"] if Path(p).name == t["source"] else Path(p).stat().st_size)
    args.update(kw)
    return pe.produce(t["size"], **args)


def test_the_targets_are_the_pinned_bf16_references():
    t = pe.target(PINS, "9B")
    assert t["source"] == "Qwen3.5-9B-BF16.gguf" and t["output"] == "Qwen3.5-9B-Q4_0.gguf"
    assert t["source_bytes"] == 17920697312
    with pytest.raises(pe.ProduceE3Error):
        pe.target(PINS, "27B")


def test_a_q4_0_is_produced_atomically_and_its_entry_read_from_the_header(tmp_path):
    models, t = setup(tmp_path)
    e = produce(models, t)
    assert (models / "Qwen3.5-9B-Q4_0.gguf").is_file() and not (models / "Qwen3.5-9B-Q4_0.gguf.tmp").exists()
    assert e["tied"] is False and e["token_embd_bytes"] == 256 * 64 // 32 * 18
    assert e["tensor_types"] == {"F32": 1, "Q4_0": 2, "Q6_K": 1} and e["sha256"] == "f" * 64
    text = pe.toml_entry("9B-Q4_0", e)
    assert tomllib.loads(text)["e3"]["files"]["9B-Q4_0"]["tensor_types"]["Q6_K"] == 1
    with pytest.raises(pe.ProduceE3Error, match="exists"):
        produce(models, t)


def test_it_refuses_without_the_lock_a_bad_input_toolchain_or_disk(tmp_path):
    models, t = setup(tmp_path)
    with pytest.raises(pe.ProduceE3Error, match="with-mac-lock"):
        produce(models, t, lock_held=False)
    with pytest.raises(pe.ProduceE3Error, match="BF16 reference"):
        produce(models, t, size_of=lambda p: 16)                        # the stand-in file is not 17.9 GB
    bad_tc = {**TOOLCHAIN, "llama_cpp_tree": {"head": "0" * 40, "dirty": False}}
    with pytest.raises(pe.pg.ProduceError):
        produce(models, t, toolchain=bad_tc)


def test_a_failed_quantize_leaves_no_output(tmp_path):
    models, t = setup(tmp_path)
    src = models / t["source"]

    def failing(argv):
        Path(argv[2]).write_bytes(b"partial")
        return 1

    with pytest.raises(pe.ProduceE3Error, match="exited 1"):
        produce(models, t, run=failing)
    assert list(models.iterdir()) == [src]
