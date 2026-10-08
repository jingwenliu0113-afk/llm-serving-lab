#!/usr/bin/env bash
# User-space CUDA_HOME for FlashInfer's JIT in the Core B server (decision 014). No sudo.
#
# vLLM 0.30.0's warmup always JIT-compiles FlashInfer's sampling module, which needs nvcc >= 12.9
# for sm_120, cuRAND headers and cudart, with nvcc major.minor equal to the toolkit headers' (its
# vendored CCCL #errors otherwise). The vLLM venv mixes nvcc 13.4 with cudart 13.0 headers, so the
# shim uses its own hash-locked CUDA 13.0 set (configs/flashinfer-cuda.lock.txt, .venv-fi-cuda130:
# nvcc 13.0.88, cudart 13.0.96 and curand 10.4.0.35 = the runtime torch / vLLM load). nvcc is never
# put on PATH (014).
#
# Usage (WSL): bash scripts/wsl_cuda_home_shim.sh   -> /home/wenn0113/llm-serving-lab/.cuda-home-cu13
set -euo pipefail

PROJECT=/home/wenn0113/llm-serving-lab          # = core_b_plan.PROJECT_DIR
SHIM=$PROJECT/.cuda-home-cu13                   # = core_b_plan.FLASHINFER_CUDA_HOME
CU=$PROJECT/.venv-fi-cuda130/lib/python3.12/site-packages/nvidia/cu13   # decision 014 addendum

for f in "$CU/bin/nvcc" "$CU/include/curand.h" "$CU/lib/libcudart.so.13" /usr/lib/wsl/lib/libcuda.so; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }
done
mkdir -p "$SHIM/lib64/stubs"
for d in bin include nvvm lib; do ln -sfn "$CU/$d" "$SHIM/$d"; done
ln -sfn "$CU/lib/libcudart.so.13" "$SHIM/lib64/libcudart.so"       # FlashInfer: -L$CUDA_HOME/lib64 -lcudart
ln -sfn /usr/lib/wsl/lib/libcuda.so "$SHIM/lib64/stubs/libcuda.so"  # FlashInfer: -L$CUDA_HOME/lib64/stubs -lcuda
"$SHIM/bin/nvcc" --version | tail -2
nv=$("$SHIM/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | awk '{print $1 * 1000 + $2 * 10}')
rt=$(sed -n 's/^#define CUDART_VERSION *\([0-9]*\)/\1/p' "$CU/include/cuda_runtime_api.h")
[ "$nv" = "$rt" ] || { echo "nvcc ${nv} and CUDART_VERSION ${rt} differ (CCCL would #error)" >&2; exit 1; }
echo "CUDA_HOME shim: $SHIM"
