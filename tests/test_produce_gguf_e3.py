"""Tests for scripts/produce_gguf_e3.py (decision 041 v3 §3.2): E3's Q4_0 GGUFs from the pinned BF16 references, the
Mac lock (owner verified), the toolchain, input and disk checks, atomic output, the configs/stage_e3.toml entry read
from the header, and the copy to WSL."""

import json
import struct
import subprocess
import tomllib
from pathlib import Path

import pytest

import produce_gguf_e3 as pe

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
TOOLCHAIN = {"llama_cpp": {"build_number": PINS["llama_cpp"]["build_number"], "commit": PINS["llama_cpp"]["commit"]},
             "llama_cpp_tree": {"head": PINS["llama_cpp"]["commit"], "dirty": False}}


def tiny_gguf(path, tensors, file_type=2):
    """A header-only GGUF: tensors = [(name, dims, ggml type id)], with general.architecture and general.file_type."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b
    data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", 2)
    data += s("general.architecture") + struct.pack("<I", 8) + s("qwen35")
    data += s("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type)
    for name, dims, tid in tensors:
        data += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        data += struct.pack("<I", tid) + struct.pack("<Q", 0)
    Path(path).write_bytes(data)


def fake_quantize(tensors, file_type=2):
    def run(argv):
        assert argv[-1] == "Q4_0" and argv[2].endswith(".tmp")
        tiny_gguf(argv[2], tensors, file_type)
        return 0
    return run


def setup(tmp_path, size="9B"):
    t = pe.target(PINS, size)
    models = tmp_path / "gguf"
    models.mkdir()
    (models / t["source"]).write_bytes(b"x" * 16)
    return models, t


def produce(models, t, **kw):
    args = dict(pins=PINS, models=models, lock_held=True, lock_check=lambda: True, quantize="/q", toolchain=TOOLCHAIN,
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
    assert e["tied"] is False and e["token_embd_bytes"] == 256 * 64 // 32 * 18 and e["token_embd_type"] == "Q4_0"
    assert e["tensor_types"] == {"F32": 1, "Q4_0": 2, "Q6_K": 1} and e["sha256"] == "f" * 64
    assert e["output_type"] == "Q6_K" and e["model_type"] == "qwen35 9B Q4_0"
    assert e["model_size"] == 256 * 64 // 32 * 18 + 256 * 64 // 256 * 210 + 256 * 256 // 32 * 18 + 256 * 4
    text = pe.toml_entry("9B-Q4_0", e)
    entry = tomllib.loads(text)["e3"]["files"]["9B-Q4_0"]
    assert entry == {k: v for k, v in e.items()} and set(entry) == {
        "filename", "sha256", "size_bytes", "token_embd_bytes", "token_embd_type", "tied", "output_type",
        "tensor_types", "model_size", "model_type"}
    with pytest.raises(pe.ProduceE3Error, match="exists"):
        produce(models, t)


def test_a_tied_4b_has_no_output_type_and_an_unknown_file_type_is_refused(tmp_path):
    models, t = setup(tmp_path, "4B")
    tied = [("token_embd.weight", [256, 64], 14), ("blk.0.attn_q.weight", [256, 256], 2)]
    e = produce(models, t, run=fake_quantize(tied))
    assert e["tied"] is True and e["output_type"] == "" and e["model_type"] == "qwen35 4B Q4_0"
    (tmp_path / "x").mkdir()
    models, t = setup(tmp_path / "x", "4B")
    with pytest.raises(pe.ProduceE3Error, match="file_type"):
        produce(models, t, run=fake_quantize([("token_embd.weight", [256, 64], 14)], file_type=99))


def test_it_refuses_without_the_lock_a_bad_input_toolchain_or_disk(tmp_path):
    models, t = setup(tmp_path)
    with pytest.raises(pe.ProduceE3Error, match="with-mac-lock"):
        produce(models, t, lock_held=False)
    with pytest.raises(pe.ProduceE3Error, match="with-mac-lock"):
        produce(models, t, lock_check=lambda: False)                   # the env flag alone is never trusted
    with pytest.raises(pe.ProduceE3Error, match="BF16 reference"):
        produce(models, t, size_of=lambda p: 16)                        # the stand-in file is not 17.9 GB
    with pytest.raises(pe.ProduceE3Error, match="BF16 reference"):
        produce(models, t, hash_file=lambda p: "0" * 64)
    with pytest.raises(pe.ProduceE3Error, match="disk guard"):
        produce(models, t, disk_ok=lambda path, need: {"ok": False, "free_gib": 40})
    assert list(models.iterdir()) == [models / t["source"]]
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


# ---------------------------------------------------------------- copy to WSL

def test_only_pinned_q4_0s_are_synced():
    cfg = tomllib.loads((ROOT / "configs" / "stage_e3.toml").read_text())
    with pytest.raises(pe.ProduceE3Error, match="not pinned"):
        pe.pinned_files(cfg)
    files = {**cfg["e3"]["files"]}
    files["4B-Q4_0"] = {**files["4B-Q4_0"], "sha256": "a" * 64, "size_bytes": 10}
    files["9B-Q4_0"] = {**files["9B-Q4_0"], "sha256": "b" * 64, "size_bytes": 20}
    assert pe.pinned_files({**cfg, "e3": {**cfg["e3"], "files": files}}) == [
        ("Qwen3.5-4B-Q4_0.gguf", "a" * 64, 10), ("Qwen3.5-9B-Q4_0.gguf", "b" * 64, 20)]


def test_sync_copies_what_wsl_lacks_and_stops_on_a_mismatch(tmp_path, capsys):
    files = [("Qwen3.5-4B-Q4_0.gguf", "a" * 64, 10), ("Qwen3.5-9B-Q4_0.gguf", "b" * 64, 20)]
    calls = []

    def run(cmd, input, capture_output, text):
        calls.append(input)
        out = f"{'a' * 64}  Qwen3.5-4B-Q4_0.gguf\n" if "sha256sum" in input else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")

    copied = []
    assert pe.sync(files, tmp_path, run=run, copy=lambda *a: copied.append(a[:3]) or True) == 0
    assert copied == [files[1]] and "llmsl-recv-gguf.sh" in calls[1]
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == {"present": ["Qwen3.5-4B-Q4_0.gguf"],
                                                                   "to_copy": ["Qwen3.5-9B-Q4_0.gguf"]}
    assert pe.sync(files, tmp_path, run=run, copy=lambda *a: False) == 1
    assert pe.sync(files, tmp_path, run=run, copy=lambda *a: None) == pe.sgw.EX_TEMPFAIL
    down = lambda cmd, input, capture_output, text: subprocess.CompletedProcess(cmd, 255, "", "ssh: down")  # noqa: E731
    assert pe.sync(files, tmp_path, run=down, copy=lambda *a: True) == pe.sgw.EX_TEMPFAIL
