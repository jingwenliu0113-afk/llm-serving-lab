"""Tests for scripts/produce_gguf.py: BF16 GGUF + local quants for one Core A size,
with hashes back into pins.toml and a recipe per artifact (decision 004/008,
docs/STORAGE_PLAN.md)."""

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import pytest

import produce_gguf as pg

ROOT = Path(__file__).resolve().parent.parent

PINS_SNIPPET = '''[core_a.models.reference]
filename = "Qwen3.5-4B-BF16.gguf"
sha256 = ""
size_bytes = 0

[core_a.models.files.Q8_0]
filename = "Qwen3.5-4B-Q8_0.gguf"
sha256 = ""
size_bytes = 0
'''


def test_update_pins_text_fills_only_the_named_file():
    out = pg.update_pins_text(PINS_SNIPPET, "Qwen3.5-4B-Q8_0.gguf", "ab" * 32, 123)
    assert 'filename = "Qwen3.5-4B-Q8_0.gguf"\nsha256 = "' + "ab" * 32 + '"\nsize_bytes = 123' in out
    assert 'filename = "Qwen3.5-4B-BF16.gguf"\nsha256 = ""\nsize_bytes = 0' in out


def test_update_pins_text_refuses_to_overwrite_a_different_hash():
    once = pg.update_pins_text(PINS_SNIPPET, "Qwen3.5-4B-Q8_0.gguf", "ab" * 32, 123)
    assert pg.update_pins_text(once, "Qwen3.5-4B-Q8_0.gguf", "ab" * 32, 123) == once  # idempotent
    with pytest.raises(pg.ProduceError, match="different sha256"):
        pg.update_pins_text(once, "Qwen3.5-4B-Q8_0.gguf", "cd" * 32, 123)


def test_update_pins_text_unknown_file():
    with pytest.raises(pg.ProduceError):
        pg.update_pins_text(PINS_SNIPPET, "nope.gguf", "ab" * 32, 1)


def test_plan_commands_match_decision_008():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        pins = tomllib.load(fh)
    steps = pg.plan_steps(pins, "4B", root=Path("/r"))
    convert = steps[0]
    assert convert["output"] == "Qwen3.5-4B-BF16.gguf"
    assert convert["argv"][1].endswith("third_party/llama.cpp/convert_hf_to_gguf.py")
    assert "--no-mtp" in convert["argv"] and "--mmproj" not in convert["argv"]
    assert convert["argv"][convert["argv"].index("--outtype") + 1] == "bf16"
    quants = steps[1:]
    assert [s["quant"] for s in quants] == pins["core_a"]["quants"]
    for s in quants:
        assert s["argv"][0].endswith("build-metal/bin/llama-quantize")
        assert s["argv"][-1] == s["quant"]
        assert "--imatrix" not in s["argv"]  # decision 004: no imatrix


class FakeTools:
    """Pretends to be the converter and llama-quantize: writes deterministic bytes."""

    def __init__(self):
        self.calls = []

    def run(self, argv):
        self.calls.append(argv)
        out = Path(argv[argv.index("--outfile") + 1]) if "--outfile" in argv else Path(argv[-2])
        out.write_bytes(("artifact:" + out.name).encode())
        return 0


def setup_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    pins_text = (ROOT / "configs" / "pins.toml").read_text()
    (root / "configs" / "pins.toml").write_text(pins_text)
    (root / "configs" / "converter.lock.txt").write_text("torch==2.11.0\n")
    pins = tomllib.loads(pins_text)
    m = next(m for m in pins["core_a"]["models"] if m["size"] == "4B")
    src = root / pg.source_dir(m)
    src.mkdir(parents=True)
    shards = {}
    for name in m["source_shards"]:
        (src / name).write_bytes(name.encode())
        shards[name] = hashlib.sha256(name.encode()).hexdigest()
    return root, shards


def produce(root, shards, tools, **kw):
    kw.setdefault("lock_held_by_other", lambda: False)
    kw.setdefault("disk_ok", lambda path, need_gib: {"ok": True, "need_gib": need_gib})
    return pg.produce(size="4B", root=root, run=tools.run, shard_sha256_override=shards,
                      toolchain={"llama_cpp": "0.5.0 (build 11146, commit 7fe450e)"}, now=lambda: "T", **kw)


def test_produce_writes_artifacts_recipes_and_pins(tmp_path):
    root, shards = setup_repo(tmp_path)
    tools = FakeTools()
    summary = produce(root, shards, tools)
    names = [a["filename"] for a in summary["artifacts"]]
    assert names == ["Qwen3.5-4B-BF16.gguf", "Qwen3.5-4B-Q8_0.gguf", "Qwen3.5-4B-Q6_K.gguf", "Qwen3.5-4B-Q4_K_M.gguf"]
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    m = next(m for m in pins["core_a"]["models"] if m["size"] == "4B")
    for a in summary["artifacts"]:
        gguf = root / "models" / "gguf" / a["filename"]
        digest = hashlib.sha256(gguf.read_bytes()).hexdigest()
        assert a["sha256"] == digest
        entry = m["reference"] if a["filename"].endswith("BF16.gguf") else m["files"][a["quant"]]
        assert entry["sha256"] == digest and entry["size_bytes"] == gguf.stat().st_size
        recipe = json.loads((root / "results" / "artifacts" / f"{a['filename']}.json").read_text())
        assert recipe["output"] == {"filename": a["filename"], "sha256": digest, "size_bytes": gguf.stat().st_size}
        assert recipe["argv"] == a["argv"]
        assert recipe["toolchain"]["converter_lock_sha256"] == hashlib.sha256(b"torch==2.11.0\n").hexdigest()
        assert recipe["toolchain"]["llama_cpp"].startswith("0.5.0")
    q8 = json.loads((root / "results" / "artifacts" / "Qwen3.5-4B-Q8_0.gguf.json").read_text())
    assert q8["inputs"] == [{"filename": "Qwen3.5-4B-BF16.gguf", "sha256": summary["artifacts"][0]["sha256"]}]
    bf16 = json.loads((root / "results" / "artifacts" / "Qwen3.5-4B-BF16.gguf.json").read_text())
    assert bf16["inputs"][0]["repo"] == "Qwen/Qwen3.5-4B" and bf16["inputs"][0]["shards"] == shards


def test_produce_refuses_on_source_hash_mismatch(tmp_path):
    root, shards = setup_repo(tmp_path)
    bad = {k: "0" * 64 for k in shards}
    with pytest.raises(pg.ProduceError, match="sha256"):
        produce(root, bad, FakeTools())


def test_produce_refuses_when_mac_lock_held_by_other(tmp_path):
    root, shards = setup_repo(tmp_path)
    with pytest.raises(pg.ProduceError, match="Mac heavy lock"):
        produce(root, shards, FakeTools(), lock_held_by_other=lambda: True)


def test_produce_refuses_when_disk_guard_fails(tmp_path):
    root, shards = setup_repo(tmp_path)
    tools = FakeTools()
    with pytest.raises(pg.ProduceError, match="disk"):
        produce(root, shards, tools, disk_ok=lambda path, need_gib: {"ok": False, "need_gib": need_gib})
    assert tools.calls == []


def test_delete_source_requires_recipes_and_pins(tmp_path):
    root, shards = setup_repo(tmp_path)
    with pytest.raises(pg.ProduceError, match="recipe"):
        pg.delete_source(size="4B", root=root)
    produce(root, shards, FakeTools())
    removed = pg.delete_source(size="4B", root=root)
    assert removed and all(not Path(p).exists() for p in removed)
