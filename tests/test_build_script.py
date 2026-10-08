"""scripts/build_llama_cpp.sh must build exactly the commit pinned in configs/pins.toml,
with the same non-backend flags on both platforms (docs/decisions/002)."""

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_llama_cpp.sh"


def text():
    return SCRIPT.read_text()


def script_var(name):
    m = re.search(rf"^{name}=(\S+)", text(), re.MULTILINE)
    assert m, f"{name} not set in {SCRIPT.name}"
    return m.group(1).strip("\"'")


def pins():
    with (ROOT / "configs" / "pins.toml").open("rb") as fh:
        return tomllib.load(fh)


def branch_body(backend):
    m = re.search(rf"^\s*{backend}\)\s*\n(.*?)^\s*;;", text(), re.MULTILINE | re.DOTALL)
    assert m, f"no '{backend})' branch"
    lines = [ln.split("#", 1)[0].strip() for ln in m.group(1).splitlines()]
    return [ln for ln in lines if ln]


def test_script_commit_matches_pins():
    assert script_var("LLAMA_COMMIT") == pins()["llama_cpp"]["commit"]


def test_script_build_number_matches_pins():
    assert int(script_var("LLAMA_BUILD_NUMBER")) == pins()["llama_cpp"]["build_number"]


def test_script_cmake_matches_pins():
    assert script_var("CMAKE_PIN") == pins()["llama_cpp"]["toolchain"]["cmake"]


def test_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_backend_branches_only_set_backend_flags():
    prefixes = {"metal": ("-DGGML_METAL",), "cuda": ("-DGGML_CUDA", "-DCMAKE_CUDA_ARCHITECTURES")}
    for backend, allowed in prefixes.items():
        body = branch_body(backend)
        assert len(body) == 1, body
        m = re.fullmatch(r"BACKEND_FLAGS=\((.*)\)", body[0])
        assert m, body[0]
        for flag in m.group(1).split():
            assert flag.startswith(allowed), (backend, flag)


def test_common_flags_assigned_once_outside_branches():
    assert len(re.findall(r"^\s*COMMON_FLAGS\+?=", text(), re.MULTILINE)) == 1


def test_parallelism_is_bounded():
    # A bare `-j` lets make spawn every nvcc job at once inside the WSL2 VM.
    assert re.search(r'--build .*-j "\$JOBS"', text())
    assert not re.search(r"-j\s+--target", text())


def test_tree_guard_checks_untracked_files():
    # build-<backend>/ is already ignored by llama.cpp's own .gitignore (/build*).
    assert "--untracked-files=no" not in text()
    assert 'status --porcelain)" ]' in text()


def test_build_record_includes_compiler_versions():
    assert "COMPILER_VERSION" in text()


def test_targets_include_the_tools_core_needs():
    # llama-tokenize: cross-validates the Qwen3.5 tokenizer constants (WikiText chunks,
    # MMLU exclusions) against the pinned llama.cpp tokenizer (decision 005 pilot step 3).
    m = re.search(r"^TARGETS=\((.*)\)", text(), re.MULTILINE)
    assert m
    assert set(m.group(1).split()) == {"llama-bench", "llama-perplexity", "llama-completion", "llama-quantize",
                                       "llama-tokenize"}


def test_cuda_build_names_the_user_space_toolkit_explicitly():
    """015: CUDA 13.4.1 from NVIDIA's pip wheels laid out by scripts/wsl_cuda134_root.sh; named explicitly
    because WSL's PATH carries the Windows CUDA bin dirs. Metal builds get no toolkit flags."""
    t = text()
    assert 'CUDA_ROOT=${CUDA_ROOT:-$ROOT/.cuda-root-13.4}' in t
    assert 'export CUDACXX="$CUDA_ROOT/bin/nvcc"' in t
    assert 'TOOLKIT_FLAGS=(-G Ninja -DCMAKE_MAKE_PROGRAM="$CUDA_VENV/bin/ninja" -DCUDAToolkit_ROOT="$CUDA_ROOT")' in t
    assert 'CUDA_VENV=${CUDA_VENV:-$ROOT/.venv-cuda134}' in t
    assert '${TOOLKIT_FLAGS[@]+"${TOOLKIT_FLAGS[@]}"}' in t  # empty-array safe under set -u (bash 3.2)
