"""The WSL helper scripts (decision 014) must match core_b_plan's paths and never use sudo
(MACHINE_POLICY: system packages are installed by the user)."""

import re
import subprocess
from pathlib import Path

import core_b_plan as bp

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "scripts" / "wsl_cuda_home_shim.sh"
CHECK = ROOT / "scripts" / "wsl_toolchain_check.sh"


def var(path, name):
    m = re.search(rf"^{name}=(\S+)", path.read_text(), re.MULTILINE)
    assert m, f"{name} not set in {path.name}"
    return m.group(1)


def test_paths_match_core_b_plan():
    assert var(SHIM, "PROJECT") == bp.PROJECT_DIR == var(CHECK, "PROJECT")
    assert var(SHIM, "SHIM").replace("$PROJECT", bp.PROJECT_DIR) == bp.FLASHINFER_CUDA_HOME
    assert var(CHECK, "SHIM").replace("$PROJECT", bp.PROJECT_DIR) == bp.FLASHINFER_CUDA_HOME


def test_scripts_never_use_sudo_and_parse():
    for path in (SHIM, CHECK):
        code = "\n".join(ln.split("#", 1)[0] for ln in path.read_text().splitlines())
        assert not re.search(r"\bsudo\b", code), path.name
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_shim_provides_what_toolchain_problems_checks():
    text = SHIM.read_text()
    for rel in ("lib64/libcudart.so", "lib64/stubs/libcuda.so"):
        assert f'"$SHIM/{rel}"' in text
    assert 'for d in bin include nvvm lib' in text  # bin/nvcc, include/curand.h
