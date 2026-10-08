"""Contract tests for configs/pins.toml.

pins.toml is the machine-readable form of decisions 002-007. These tests
encode SPEC invariants so a later edit cannot silently break them:
  - Core A: one family, exactly 2 parameter sizes x 3 quantizations x 3 context lengths
  - Core B: exactly one fixed precision
  - every upstream artifact is pinned by immutable id (commit sha / sha256)
  - staged pre-registration (decision 006):
      freeze.mac    -> every derived GGUF hash + Mac toolchain + quality data pinned
      freeze.nvidia -> additionally CUDA toolchain, vLLM stack and Core B verified
"""

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PINS_PATH = ROOT / "configs" / "pins.toml"

SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXACT_VERSION = re.compile(r"^\d+\.\d+\.\d+(\.post\d+)?$")
PLATFORMS = {"macos-metal", "wsl2-cuda"}


@pytest.fixture(scope="module")
def pins():
    with PINS_PATH.open("rb") as fh:
        return tomllib.load(fh)


def derived_files(pins):
    for m in pins["core_a"]["models"]:
        yield m["reference"]
        yield from m["files"].values()


def test_schema_and_freeze_flags(pins):
    assert pins["schema_version"] == 1
    assert isinstance(pins["freeze"]["mac"], bool)
    assert isinstance(pins["freeze"]["nvidia"], bool)
    # Stage B is an addendum to stage A, never a replacement (decision 006).
    assert not (pins["freeze"]["nvidia"] and not pins["freeze"]["mac"])


def test_llama_cpp_pinned_to_release_tag_and_commit(pins):
    lc = pins["llama_cpp"]
    assert re.fullmatch(r"b\d+", lc["tag"])
    assert lc["build_number"] == int(lc["tag"][1:])
    assert SHA1.fullmatch(lc["commit"])
    assert set(lc["verified"]) | set(lc["pending"]) <= PLATFORMS
    assert not set(lc["verified"]) & set(lc["pending"])


def test_llama_cpp_toolchain_recorded_for_verified_platforms(pins):
    tc = pins["llama_cpp"]["toolchain"]
    assert EXACT_VERSION.fullmatch(tc["cmake"])
    for platform in pins["llama_cpp"]["verified"]:
        assert tc[platform.replace("-", "_")]["compiler"], platform


def test_vllm_pinned_to_exact_version_and_deps(pins):
    v = pins["vllm"]
    assert EXACT_VERSION.fullmatch(v["version"])
    assert SHA1.fullmatch(v["commit"])
    assert SHA256.fullmatch(v["wheel_sha256"])
    assert set(v["verified"]) | set(v["pending"]) <= {"wsl2-cuda"}
    for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "triton"):
        assert EXACT_VERSION.fullmatch(v["deps"][name]), name


def test_core_a_grid_is_2_sizes_x_3_quants_x_3_contexts(pins):
    core_a = pins["core_a"]
    assert len(core_a["quants"]) == 3 and len(set(core_a["quants"])) == 3
    ctx = core_a["context_lengths"]
    assert len(ctx) == 3
    assert all(isinstance(c, int) and c > 0 for c in ctx)
    assert ctx == sorted(set(ctx)), "context lengths must be strictly increasing"
    # llama.cpp pads n_ctx to a multiple of 256; off-grid values would make
    # llama-bench rows and -c memory runs describe different KV sizes.
    assert all(c % 256 == 0 for c in ctx)
    models = core_a["models"]
    assert len(models) == 2
    assert len({m["size"] for m in models}) == 2


def test_unfrozen_placeholders_present(pins):
    # Fields that must be filled before each freeze exist from day one (empty until then).
    assert "converter_lock" in pins["core_a"]
    assert "lock_file" in pins["vllm"]
    assert "mix_dataset_sha256" in pins["core_b"]
    for key in ("b_mib", "v_total_mib", "gpu_memory_utilization", "gdn_prefill_backend", "rtx_n_thr", "rtx_poll",
                "c_ctx_mib", "kv_cache_memory_bytes", "num_gpu_blocks", "block_size", "wsl2_pin_memory"):
        assert key in pins["stage_b"], key


def test_core_a_context_within_native_window(pins):
    core_a = pins["core_a"]
    assert max(core_a["context_lengths"]) <= core_a["native_context"]


def test_core_a_models_same_family_and_source_pinned(pins):
    core_a = pins["core_a"]
    for m in core_a["models"]:
        assert m["family"] == core_a["family"]
        assert m["license"]
        assert SHA1.fullmatch(m["source_revision"]), m["source_repo"]
        shards = m["source_shards"]
        assert shards, m["source_repo"]
        for name, digest in shards.items():
            assert name.endswith(".safetensors")
            assert SHA256.fullmatch(digest), name


def test_core_a_every_quant_has_a_derived_file(pins):
    core_a = pins["core_a"]
    for m in core_a["models"]:
        assert set(m["files"]) == set(core_a["quants"]), m["source_repo"]
        for quant, f in m["files"].items():
            assert f["filename"].endswith(".gguf")
            assert quant.lower() in f["filename"].lower()
        assert m["reference"]["filename"].endswith(".gguf")


def test_derived_hashes_well_formed_when_present(pins):
    for f in derived_files(pins):
        assert f["sha256"] == "" or SHA256.fullmatch(f["sha256"]), f["filename"]


def test_quality_sources_pinned(pins):
    files = pins["quality"]["files"]
    assert {"wikitext_zip", "wiki_test_raw", "hellaswag", "mmlu", "arc", "winogrande"} <= set(files)
    for key, f in files.items():
        assert SHA256.fullmatch(f["sha256"]), key
        assert isinstance(f["size_bytes"], int) and f["size_bytes"] > 0, key
        if "revision" in f:
            assert SHA1.fullmatch(f["revision"]), key


def test_stage_a_freeze_requires_derived_hashes_and_mac_toolchain(pins):
    if not pins["freeze"]["mac"]:
        pytest.skip("stage A (Mac) pre-registration not frozen yet")
    for f in derived_files(pins):
        assert SHA256.fullmatch(f["sha256"]), f["filename"]
        assert isinstance(f["size_bytes"], int) and f["size_bytes"] > 0
    assert "macos-metal" in pins["llama_cpp"]["verified"]
    assert (ROOT / pins["core_a"]["converter_lock"]).is_file()
    subsets = pins["quality"]["subsets"]
    for name in ("hellaswag", "mmlu"):
        assert SHA256.fullmatch(subsets[name]["sha256"]), name
        assert subsets[name]["index_file"], name


def test_stage_b_freeze_requires_verified_nvidia_stack(pins):
    if not pins["freeze"]["nvidia"]:
        pytest.skip("stage B (NVIDIA) addendum not frozen yet")
    assert "wsl2-cuda" in pins["llama_cpp"]["verified"]
    assert pins["llama_cpp"]["toolchain"]["wsl2_cuda"]["cuda_toolkit"]
    assert "wsl2-cuda" in pins["vllm"]["verified"]
    assert (ROOT / pins["vllm"]["lock_file"]).is_file()
    assert pins["core_b"]["status"] == "verified"
    mix = pins["core_b"]["mix_dataset_sha256"]
    assert set(mix) == {"2048", "8192"} and all(SHA256.fullmatch(v) for v in mix.values())
    sb = pins["stage_b"]
    assert sb["b_mib"] > 0 and sb["c_ctx_mib"] > 0 and sb["kv_cache_memory_bytes"] > 0 and sb["num_gpu_blocks"] > 0
    assert sb["wsl2_pin_memory"] in ("0", "1") and sb["gdn_prefill_backend"] and sb["rtx_poll"] >= 0


def test_core_b_single_fixed_precision(pins):
    core_b = pins["core_b"]
    assert isinstance(core_b["precision"], str) and core_b["precision"]
    assert core_b["dtype"]
    assert SHA1.fullmatch(core_b["revision"])
    assert core_b["family"] == pins["core_a"]["family"]
    assert core_b["status"] in {"proposed", "verified"}
    assert core_b["kv_cache_dtype"] == "auto"  # KV quantization is Advanced


def test_core_b_fixed_model_without_alternatives(pins):
    # Decision 008 (supersedes the Qwen3-8B-AWQ choice in 003): one fixed quantized
    # checkpoint, no fallback matrix; 16 GB must keep KV / concurrency headroom.
    core_b = pins["core_b"]
    assert core_b["repo"] == "RedHatAI/Qwen3.5-9B-quantized.w4a16"
    assert core_b["precision"] == "w4a16-gptq-int4-g128"
    assert core_b["dtype"] == "bfloat16"
    assert "fallback" not in core_b


def test_core_b_is_quantized_from_a_core_a_size(pins):
    # Lineage: the serving model is a quantization of one of the Core A sizes.
    core_a_sources = {m["source_repo"] for m in pins["core_a"]["models"]}
    assert pins["core_b"]["base_repo"] in core_a_sources


def test_core_a_ordering_conventions(pins):
    # benchmarks/core_a_plan.sentinel_cell relies on: models smallest first,
    # quants highest-bit first.
    core_a = pins["core_a"]
    sizes = [float(m["size"].rstrip("B")) for m in core_a["models"]]
    assert sizes == sorted(sizes)
    bpw = {"Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.5, "Q4_K_M": 4.85, "Q4_0": 4.5, "Q3_K_M": 3.9}
    assert all(q in bpw for q in core_a["quants"])
    assert [bpw[q] for q in core_a["quants"]] == sorted((bpw[q] for q in core_a["quants"]), reverse=True)


def test_quality_subset_index_files_exist(pins):
    for name, sub in pins["quality"]["subsets"].items():
        if sub["index_file"]:
            path = ROOT / sub["index_file"]
            assert path.is_file(), name
            record = json.loads(path.read_text())
            assert record["indices"] == sorted(set(record["indices"]))
            assert len(record["indices"]) == record["n"]


def test_quality_exclusions_are_a_subset_of_the_index_list(pins):
    for name, sub in pins["quality"]["subsets"].items():
        if sub.get("exclude_file"):
            excl = json.loads((ROOT / sub["exclude_file"]).read_text())
            idx = json.loads((ROOT / sub["index_file"]).read_text())
            assert set(excl["excluded"]) <= set(idx["indices"]), name
            assert excl["n_after"] == len(idx["indices"]) - len(excl["excluded"]), name


def test_gitignore_keeps_raw_outputs_and_secrets_out_of_git():
    import subprocess

    paths = ["results/core_a/macos-metal/s1/raw/fit-4B-Q8_0-2048-r1.stdout.txt",
             "results/quality/q1/raw/kld-4B-Q8_0.stdout.txt", "models/gguf/x.gguf", "data/kld/4B.kld",
             "third_party/convert-venv/bin/python", ".venv/bin/python", ".env", "bundle.tar",
             "results/stage_b/20261009/boots/cache/triton_cache_dir/x.so"]
    res = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", *paths], capture_output=True,
                         text=True)
    assert sorted(res.stdout.split()) == sorted(paths)
    kept = ["results/core_a/macos-metal/s1/rows.jsonl", "results/core_a/macos-metal/s1/raw_manifest.json",
            "results/artifacts/Qwen3.5-4B-Q8_0.gguf.json", "configs/pins.toml"]
    res = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", *kept], capture_output=True, text=True)
    assert res.stdout.strip() == ""


def test_mixed_dataset_sha256_is_one_per_mixed_level(pins):
    """007 defines `mixed` only at L = 2048 / 8192, one seeded file per level (013)."""
    mix = pins["core_b"]["mix_dataset_sha256"]
    assert isinstance(mix, dict) and set(mix) <= {"2048", "8192"}
    assert all(SHA256.fullmatch(v) for v in mix.values())
