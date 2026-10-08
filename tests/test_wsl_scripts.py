"""The WSL helper scripts (decision 014) must match core_b_plan's paths and never use sudo
(MACHINE_POLICY: system packages are installed by the user)."""

import re
import subprocess
from pathlib import Path

import core_b_plan as bp

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "scripts" / "wsl_cuda_home_shim.sh"
CHECK = ROOT / "scripts" / "wsl_toolchain_check.sh"
CUDA_ROOT_SCRIPT = ROOT / "scripts" / "wsl_cuda134_root.sh"


def var(path, name):
    m = re.search(rf"^{name}=(\S+)", path.read_text(), re.MULTILINE)
    assert m, f"{name} not set in {path.name}"
    return m.group(1)


def test_paths_match_core_b_plan():
    assert var(SHIM, "PROJECT") == bp.PROJECT_DIR == var(CHECK, "PROJECT")
    assert var(SHIM, "SHIM").replace("$PROJECT", bp.PROJECT_DIR) == bp.FLASHINFER_CUDA_HOME
    assert var(CHECK, "SHIM").replace("$PROJECT", bp.PROJECT_DIR) == bp.FLASHINFER_CUDA_HOME


def test_scripts_never_use_sudo_and_parse():
    for path in (SHIM, CHECK, CUDA_ROOT_SCRIPT):
        code = "\n".join(ln.split("#", 1)[0] for ln in path.read_text().splitlines())
        assert not re.search(r"\bsudo\b", code), path.name
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_shim_provides_what_toolchain_problems_checks():
    text = SHIM.read_text()
    for rel in ("lib64/libcudart.so", "lib64/stubs/libcuda.so"):
        assert f'"$SHIM/{rel}"' in text
    assert 'for d in bin include nvvm lib' in text  # bin/nvcc, include/curand.h


def test_llama_cuda_root_matches_the_build_script_default():
    """015: scripts/wsl_cuda134_root.sh builds the toolkit root build_llama_cpp.sh uses by default."""
    assert var(CUDA_ROOT_SCRIPT, "PROJECT") == bp.PROJECT_DIR
    assert var(CUDA_ROOT_SCRIPT, "ROOT") == "$PROJECT/.cuda-root-13.4"
    build = (ROOT / "scripts" / "build_llama_cpp.sh").read_text()
    assert "$ROOT/.cuda-root-13.4" in build
    text = CUDA_ROOT_SCRIPT.read_text()
    for lib in ("libcudart", "libcublas", "libcublasLt"):
        assert lib in text
    assert '"$ROOT/lib64/stubs/libcuda.so"' in text and ".venv-cuda134" in text


def test_llama_cuda_lock_pins_one_cuda_13_4_1_set():
    lock = (ROOT / "configs" / "llama-cuda.lock.txt").read_text()
    pins = dict(re.findall(r"^([a-z0-9-]+)==(\S+) \\$", lock, re.MULTILINE))
    assert pins["nvidia-cuda-nvcc"] == pins["nvidia-cuda-runtime"] == pins["nvidia-cuda-crt"] == "13.4.92"
    assert pins["nvidia-nvvm"] == pins["nvidia-cuda-nvrtc"] == "13.4.92"
    assert pins["nvidia-cublas"] == "13.4.2.4" and pins["nvidia-cuda-cccl"] == "13.3.4.3.1"
    assert pins["ninja"] == "1.13.2"  # CMake generator; WSL has no make (015)
    assert "--hash=sha256:" in lock


def test_flashinfer_shim_uses_one_coherent_cuda_13_0_set():
    """014 addendum: nvcc major.minor must equal the cudart headers' (FlashInfer's CCCL #errors otherwise);
    the set matches the runtime vLLM / torch load (vllm.lock: cuda-runtime 13.0.96, curand 10.4.0.35)."""
    lock = (ROOT / "configs" / "flashinfer-cuda.lock.txt").read_text()
    pins = dict(re.findall(r"^([a-z0-9-]+)==(\S+) \\$", lock, re.MULTILINE))
    vllm = dict(re.findall(r"^([a-z0-9-]+)==(\S+) \\$", (ROOT / "configs" / "vllm.lock.txt").read_text(), re.MULTILINE))
    assert pins["nvidia-cuda-nvcc"].startswith("13.0.") and pins["nvidia-cuda-runtime"].startswith("13.0.")
    assert pins["nvidia-cuda-runtime"] == vllm["nvidia-cuda-runtime"] and pins["nvidia-curand"] == vllm["nvidia-curand"]
    assert ".venv-fi-cuda130" in SHIM.read_text() and "CUDART_VERSION" in SHIM.read_text()
