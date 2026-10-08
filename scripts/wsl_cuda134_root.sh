#!/usr/bin/env bash
# User-space CUDA 13.4.1 toolkit root for the llama.cpp CUDA build in WSL (decision 015). No sudo.
#
# NVIDIA's pip wheels (configs/llama-cuda.lock.txt, installed into .venv-cuda134) carry nvcc 13.4.92,
# cudart, cuBLAS and CCCL in one merged nvidia/cu13 tree. CMake's FindCUDAToolkit also needs unversioned
# libcudart.so / libcublas.so / libcublasLt.so and a libcuda.so stub (CUDA::cuda_driver); this links
# them into a toolkit root. WSL's real libcuda.so serves as the stub (soname libcuda.so.1).
#
# Usage (WSL): bash scripts/wsl_cuda134_root.sh   -> /home/wenn0113/llm-serving-lab/.cuda-root-13.4
set -euo pipefail

PROJECT=/home/wenn0113/llm-serving-lab
ROOT=$PROJECT/.cuda-root-13.4                   # = build_llama_cpp.sh CUDA_ROOT default
CU=$PROJECT/.venv-cuda134/lib/python3.12/site-packages/nvidia/cu13

for f in "$CU/bin/nvcc" "$CU/include/cuda_runtime.h" "$CU/include/cublas_v2.h" "$CU/include/cccl/cub/cub.cuh" \
         "$CU/lib/libcudart.so.13" "$CU/lib/libcublas.so.13" "$CU/lib/libcublasLt.so.13" /usr/lib/wsl/lib/libcuda.so; do
  [ -e "$f" ] || { echo "missing $f (uv pip sync .venv-cuda134 with configs/llama-cuda.lock.txt)" >&2; exit 1; }
done
mkdir -p "$ROOT/lib64/stubs"
for d in bin include nvvm lib; do ln -sfn "$CU/$d" "$ROOT/$d"; done
for lib in libcudart libcublas libcublasLt; do
  ln -sfn "$CU/lib/$lib.so.13" "$ROOT/lib64/$lib.so"
  ln -sfn "$CU/lib/$lib.so.13" "$ROOT/lib64/$lib.so.13"
done
for a in libcudart_static.a libcudadevrt.a; do ln -sfn "$CU/lib/$a" "$ROOT/lib64/$a"; done
ln -sfn /usr/lib/wsl/lib/libcuda.so "$ROOT/lib64/stubs/libcuda.so"
"$ROOT/bin/nvcc" --version | tail -2
echo "CUDA toolkit root: $ROOT"
