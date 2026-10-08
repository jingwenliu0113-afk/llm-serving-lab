"""Tests for scripts/produce_gguf.py: BF16 GGUF + local quants for one Core A size,
with hashes back into pins.toml and a recipe per artifact (decision 004/008,
docs/STORAGE_PLAN.md)."""

import hashlib
import json
import subprocess
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


COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"
GOOD_TC = {
    "llama_cpp": {"version": "0.5.0", "build_number": 11146, "commit": COMMIT, "built_with": "AppleClang"},
    "llama_cpp_tree": {"head": COMMIT, "dirty": False},
    "converter_python": "3.12.13",
}


class FakeTools:
    """Pretends to be the converter and llama-quantize: writes deterministic bytes."""

    def __init__(self, salt="", salt_on=None, fail_on=None):
        self.calls, self.salt, self.salt_on, self.fail_on = [], salt, salt_on, fail_on

    def run(self, argv):
        self.calls.append(argv)
        out = Path(argv[argv.index("--outfile") + 1]) if "--outfile" in argv else Path(argv[-2])
        if self.fail_on and out.name.endswith(f"{self.fail_on}.gguf"):
            return 1
        salt = self.salt if self.salt_on is None or out.name.endswith(f"{self.salt_on}.gguf") else ""
        out.write_bytes(("artifact:" + salt + out.name).encode())
        return 0


SMALL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md")


def setup_repo(tmp_path, small_files=SMALL_FILES):
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
    for name in small_files:
        (src / name).write_text("small:" + name)
    return root, shards


def produce(root, shards, tools, **kw):
    kw.setdefault("lock_held_by_other", lambda: False)
    kw.setdefault("disk_ok", lambda path, need_gib: {"ok": True, "need_gib": need_gib})
    kw.setdefault("toolchain", GOOD_TC)
    return pg.produce(size="4B", root=root, run=tools.run, shard_sha256_override=shards, now=lambda: "T", **kw)


def gguf_names():
    return ["Qwen3.5-4B-BF16.gguf", "Qwen3.5-4B-Q8_0.gguf", "Qwen3.5-4B-Q6_K.gguf", "Qwen3.5-4B-Q4_K_M.gguf"]


def test_produce_writes_artifacts_recipes_and_pins(tmp_path):
    root, shards = setup_repo(tmp_path)
    tools = FakeTools()
    summary = produce(root, shards, tools)
    names = [a["filename"] for a in summary["artifacts"]]
    assert names == gguf_names()
    pins = tomllib.loads((root / "configs" / "pins.toml").read_text())
    m = next(m for m in pins["core_a"]["models"] if m["size"] == "4B")
    for a in summary["artifacts"]:
        gguf = root / "models" / "gguf" / a["filename"]
        digest = hashlib.sha256(gguf.read_bytes()).hexdigest()
        assert a["sha256"] == digest
        entry = m["reference"] if a["filename"].endswith("BF16.gguf") else m["files"][a["quant"]]
        assert entry["sha256"] == digest and entry["size_bytes"] == gguf.stat().st_size
        recipe = json.loads((root / "results" / "artifacts" / f"{a['filename']}.json").read_text())
        assert recipe["output"] == {"filename": a["filename"], "sha256": digest, "size_bytes": gguf.stat().st_size,
                                    "path": f"models/gguf/{a['filename']}"}
        assert recipe["argv"] == a["argv"]
        assert recipe["toolchain"]["converter_lock_sha256"] == hashlib.sha256(b"torch==2.11.0\n").hexdigest()
        assert recipe["toolchain"]["llama_cpp"]["build_number"] == 11146
        assert recipe["toolchain"]["converter_python"] == "3.12.13"
    q8 = json.loads((root / "results" / "artifacts" / "Qwen3.5-4B-Q8_0.gguf.json").read_text())
    assert q8["inputs"] == [{"filename": "Qwen3.5-4B-BF16.gguf", "sha256": summary["artifacts"][0]["sha256"]}]
    bf16 = json.loads((root / "results" / "artifacts" / "Qwen3.5-4B-BF16.gguf.json").read_text())
    assert bf16["inputs"][0]["repo"] == "Qwen/Qwen3.5-4B" and bf16["inputs"][0]["shards"] == shards
    assert not list((root / "models" / "gguf" / ".tmp").glob("*"))  # staging emptied


def test_recipe_hashes_every_non_shard_source_file(tmp_path):
    """README.md feeds general.* GGUF metadata, so the small-file set is part of the input (review F11)."""
    root, shards = setup_repo(tmp_path)
    produce(root, shards, FakeTools())
    bf16 = json.loads((root / "results" / "artifacts" / "Qwen3.5-4B-BF16.gguf.json").read_text())
    files = bf16["inputs"][0]["files"]
    assert set(files) == set(SMALL_FILES)
    assert files["README.md"] == hashlib.sha256(b"small:README.md").hexdigest()


def test_produce_refuses_without_required_small_files(tmp_path):
    root, shards = setup_repo(tmp_path, small_files=("config.json", "tokenizer.json"))
    tools = FakeTools()
    with pytest.raises(pg.ProduceError, match="README.md"):
        produce(root, shards, tools)
    assert tools.calls == []


def test_produce_refuses_on_source_hash_mismatch(tmp_path):
    root, shards = setup_repo(tmp_path)
    bad = {k: "0" * 64 for k in shards}
    with pytest.raises(pg.ProduceError, match="sha256"):
        produce(root, bad, FakeTools())


def test_produce_checks_the_lock_before_hashing_shards(tmp_path):
    root, shards = setup_repo(tmp_path)
    bad = {k: "0" * 64 for k in shards}  # would fail at hashing if hashing came first
    with pytest.raises(pg.ProduceError, match="Mac heavy lock"):
        produce(root, bad, FakeTools(), lock_held_by_other=lambda: True)


def test_produce_stops_between_steps_when_the_lock_is_taken(tmp_path):
    root, shards = setup_repo(tmp_path)
    tools = FakeTools()
    answers = iter([False, False, True])  # start, before convert, before Q8_0
    with pytest.raises(pg.ProduceError, match="Mac heavy lock"):
        produce(root, shards, tools, lock_held_by_other=lambda: next(answers))
    assert len(tools.calls) == 1


@pytest.mark.parametrize("bad", [
    {"llama_cpp": {**GOOD_TC["llama_cpp"], "build_number": 11000}},
    {"llama_cpp": {**GOOD_TC["llama_cpp"], "commit": "deadbeef"}},
    {"llama_cpp": None},
    {"llama_cpp_tree": {"head": COMMIT, "dirty": True}},
    {"llama_cpp_tree": {"head": "0" * 40, "dirty": False}},
])
def test_produce_refuses_on_toolchain_mismatch(tmp_path, bad):
    root, shards = setup_repo(tmp_path)
    tools = FakeTools()
    with pytest.raises(pg.ProduceError, match="toolchain"):
        produce(root, shards, tools, toolchain={**GOOD_TC, **bad})
    assert tools.calls == []


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


def test_rerun_skips_pinned_artifacts_and_resumes(tmp_path):
    root, shards = setup_repo(tmp_path)
    with pytest.raises(pg.ProduceError, match="quantize"):
        produce(root, shards, FakeTools(fail_on="Q6_K"))
    tools = FakeTools()
    summary = produce(root, shards, tools)
    ran = [Path(a[a.index("--outfile") + 1] if "--outfile" in a else a[-2]).name for a in tools.calls]
    assert ran == ["Qwen3.5-4B-Q6_K.gguf", "Qwen3.5-4B-Q4_K_M.gguf"]
    assert [a["reused"] for a in summary["artifacts"]] == [True, True, False, False]


def test_rerun_with_different_bytes_leaves_pinned_artifact_untouched(tmp_path):
    """Review F5: a mismatch must not clobber the pinned file or its recipe."""
    root, shards = setup_repo(tmp_path)
    produce(root, shards, FakeTools())
    q8 = root / "models" / "gguf" / "Qwen3.5-4B-Q8_0.gguf"
    recipe = root / "results" / "artifacts" / "Qwen3.5-4B-Q8_0.gguf.json"
    before = (q8.read_bytes(), recipe.read_text())
    q8.write_bytes(b"corrupted")  # forces Q8_0 to be rebuilt
    with pytest.raises(pg.ProduceError, match="differ from the pinned sha256"):
        produce(root, shards, FakeTools(salt="other-build:", salt_on="Q8_0"))
    assert recipe.read_text() == before[1]
    assert q8.read_bytes() == b"corrupted"  # not replaced by the mismatching rebuild
    assert not list((root / "models" / "gguf" / ".tmp").glob("*"))


# ----------------------------------------------------------- delete-source --


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


def commit_all(root):
    if not (root / ".git").exists():
        git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x", "--allow-empty")


def produced_and_committed(tmp_path):
    root, shards = setup_repo(tmp_path)
    (root / ".gitignore").write_text("models/\n")
    produce(root, shards, FakeTools())
    commit_all(root)
    return root


def test_delete_source_requires_recipes_and_pins(tmp_path):
    root, _ = setup_repo(tmp_path)
    with pytest.raises(pg.ProduceError, match="recipe"):
        pg.delete_source(size="4B", root=root)


def test_delete_source_after_commit_removes_shards_and_logs_it(tmp_path):
    root = produced_and_committed(tmp_path)
    result = pg.delete_source(size="4B", root=root, now=lambda: "T")
    assert result["removed"] and all(not (root / p).exists() for p in result["removed"])
    log = [json.loads(x) for x in (root / "results" / "artifacts" / "deletions.jsonl").read_text().splitlines()]
    assert log[-1]["size"] == "4B" and log[-1]["at"] == "T"
    assert sorted(log[-1]["verified_sha256"]) == sorted(gguf_names())
    assert sorted(log[-1]["removed"]) == sorted(result["removed"])


def test_delete_source_refuses_uncommitted_pins_or_recipes(tmp_path):
    """STORAGE_PLAN condition 2: the recipe must already be committed (review F1)."""
    root, shards = setup_repo(tmp_path)
    (root / ".gitignore").write_text("models/\n")
    commit_all(root)
    produce(root, shards, FakeTools())  # pins + recipes written but not committed
    with pytest.raises(pg.ProduceError, match="commit"):
        pg.delete_source(size="4B", root=root)
    assert all((root / pg.source_dir({"source_repo": "Qwen/Qwen3.5-4B",
                                      "source_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}) / n).exists()
               for n in shards)


def test_delete_source_refuses_when_a_gguf_is_missing_or_differs(tmp_path):
    root = produced_and_committed(tmp_path)
    q6 = root / "models" / "gguf" / "Qwen3.5-4B-Q6_K.gguf"
    original = q6.read_bytes()
    q6.write_bytes(b"x" * len(original))
    with pytest.raises(pg.ProduceError, match="Q6_K.*sha256"):
        pg.delete_source(size="4B", root=root)
    q6.unlink()
    with pytest.raises(pg.ProduceError, match="Q6_K.*missing"):
        pg.delete_source(size="4B", root=root)
