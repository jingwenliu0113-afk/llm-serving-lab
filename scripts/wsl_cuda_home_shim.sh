#!/usr/bin/env bash
# User-space CUDA_HOME for FlashInfer's JIT in the Core B server (decision 014). No sudo.
#
# vLLM 0.30.0's warmup always JIT-compiles FlashInfer's sampling module, which needs nvcc >= 12.9
# for sm_120, cuRAND headers and cudart. The project venv already carries them, hash-locked in
# configs/vllm.lock.txt (pip nvidia/cu13: nvcc 13.4.92, cudart 13, curand). This links them into
# the CUDA_HOME layout FlashInfer expects. nvcc is never put on PATH (014).
#
# Usage (WSL): bash scripts/wsl_cuda_home_shim.sh   -> /home/wenn0113/llm-serving-lab/.cuda-home-cu13
set -euo pipefail

PROJECT=/home/wenn0113/llm-serving-lab          # = core_b_plan.PROJECT_DIR
SHIM=$PROJECT/.cuda-home-cu13                   # = core_b_plan.FLASHINFER_CUDA_HOME
CU=$PROJECT/.venv/lib/python3.12/site-packages/nvidia/cu13

for f in "$CU/bin/nvcc" "$CU/include/curand.h" "$CU/lib/libcudart.so.13" /usr/lib/wsl/lib/libcuda.so; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }
done
mkdir -p "$SHIM/lib64/stubs"
for d in bin include nvvm lib; do ln -sfn "$CU/$d" "$SHIM/$d"; done
ln -sfn "$CU/lib/libcudart.so.13" "$SHIM/lib64/libcudart.so"       # FlashInfer: -L$CUDA_HOME/lib64 -lcudart
ln -sfn /usr/lib/wsl/lib/libcuda.so "$SHIM/lib64/stubs/libcuda.so"  # FlashInfer: -L$CUDA_HOME/lib64/stubs -lcuda
"$SHIM/bin/nvcc" --version | tail -2
echo "CUDA_HOME shim: $SHIM"
