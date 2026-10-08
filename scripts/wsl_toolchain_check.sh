#!/usr/bin/env bash
# Post-install check of the WSL toolchain for Core B (vLLM) and the llama.cpp CUDA build.
# No sudo, no GPU kernels, no vLLM start, no model load; writes only to a fresh /tmp dir.
#
# Usage (WSL):  bash scripts/wsl_toolchain_check.sh          # quick checks
#               JIT=1 bash scripts/wsl_toolchain_check.sh    # + compile FlashInfer's sampling module
#                 into a throwaway workspace (CPU-heavy for minutes; skip while another project
#                 runs formal GPU work; the real ~/.cache/flashinfer is not touched)
set -u
export LC_ALL=C
PROJECT=/home/wenn0113/llm-serving-lab
V=$PROJECT/.venv
SHIM=$PROJECT/.cuda-home-cu13
T=$(mktemp -d /tmp/llmsl-check.XXXXXX); trap 'rm -rf "$T"' EXIT
FAIL=0
ok() { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; FAIL=1; }
sec() { echo; echo "===== $* ====="; }

sec "1. host compilers (apt: gcc g++ libc6-dev)"
for b in cc gcc c++ g++; do p=$(command -v $b) && ok "$b -> $p" || bad "$b missing"; done
dpkg-query -W -f='${Package} ${Version}\n' gcc-11 g++-11 libc6 libc6-dev libstdc++-11-dev make 2>&1
printf '#include <stdio.h>\n#include <dlfcn.h>\nint main(void){puts("c-ok");return 0;}\n' > "$T/h.c"
gcc "$T/h.c" -o "$T/h" -ldl && "$T/h" >/dev/null && ok "gcc compile+link+run" || bad "gcc hello"
printf '#include <iostream>\n#include <string_view>\nint main(){std::string_view s{"ok"};std::cout<<s<<"\\n";}\n' > "$T/h.cpp"
c++ -std=c++17 "$T/h.cpp" -o "$T/hpp" && "$T/hpp" >/dev/null && ok "c++17 compile+link+run" || bad "c++ hello"

sec "2. Triton launcher stub (the S1 failure of 2026-10-08)"
TRITON_CACHE_DIR="$T/triton" "$V/bin/python" -I -c "
from triton.backends.nvidia.driver import CudaUtils
CudaUtils(); print('CudaUtils built')" 2>&1 | tail -2
ls "$T"/triton/*/cuda_utils*.so >/dev/null 2>&1 && ok "triton cuda_utils .so built" || bad "triton cuda_utils not built"

sec "3. CUDA 12.8 for llama.cpp (002)"
if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
  /usr/local/cuda-12.8/bin/nvcc --version | tail -1
  printf '__global__ void k(float*x){x[threadIdx.x]*=2.f;}\n' > "$T/k.cu"
  /usr/local/cuda-12.8/bin/nvcc -arch=sm_120a -c "$T/k.cu" -o "$T/k12.o" && ok "nvcc 12.8 sm_120a compile" || bad "nvcc 12.8 sm_120a"
  for f in libcudart.so libcublas.so libcublasLt.so stubs/libcuda.so; do
    [ -e "/usr/local/cuda-12.8/targets/x86_64-linux/lib/$f" ] && ok "cuda-12.8 $f" || bad "cuda-12.8 $f missing"
  done
  command -v make >/dev/null && ok "make $(make --version | head -1)" || bad "make missing"
else
  echo "SKIP  CUDA 12.8 not installed (llama.cpp step not done)"
fi

sec "4. Core B server invariants (014)"
command -v nvcc >/dev/null && bad "nvcc on PATH ($(command -v nvcc)): changes vLLM's has_flashinfer()" || ok "no nvcc on PATH"
/sbin/ldconfig -p | grep -q 'libcuda\.so\.1 .*/usr/lib/wsl/lib' && ok "ldconfig: WSL libcuda.so.1" || bad "WSL libcuda.so.1 not in ldconfig"
n12=$(/sbin/ldconfig -p | grep -cE 'libcudart\.so\.12 ')
echo "INFO  ldconfig libcudart.so.12 entries: $n12 (cuDNN frontend may warn 'Multiple libcudart'; see server log)"
"$V/bin/python" -I -c "import vllm, torch, triton, flashinfer; print('vllm', vllm.__version__, '| torch', torch.__version__, torch.version.cuda, '| triton', triton.__version__, '| flashinfer', flashinfer.__version__)" 2>&1 | tail -1

sec "5. FlashInfer JIT environment, as the launcher sets it (core_b_plan.server_env)"
if [ -d "$SHIM" ]; then
  env CUDA_HOME="$SHIM" PATH="$V/bin:$PATH" "$V/bin/python" -I -c "
import os, shutil
from flashinfer.jit.cpp_ext import get_cuda_path, get_cuda_version
h = get_cuda_path(); print('cuda_home', h, '| nvcc', get_cuda_version(), '| ninja', shutil.which('ninja'), '| nvcc on PATH', shutil.which('nvcc'))
for p in ('bin/nvcc', 'include/curand.h', 'lib64/libcudart.so', 'lib64/stubs/libcuda.so'):
    print(p, 'ok' if os.path.exists(os.path.join(h, p)) else 'MISSING')" 2>&1 | tail -6
  printf '__global__ void k(float*x){x[threadIdx.x]*=2.f;}\n' > "$T/k.cu"
  "$SHIM/bin/nvcc" -arch=sm_120f -std=c++17 -c "$T/k.cu" -o "$T/k13.o" && ok "shim nvcc sm_120f compile (host $(gcc -dumpversion))" || bad "shim nvcc compile"
else
  bad "CUDA_HOME shim $SHIM missing (bash scripts/wsl_cuda_home_shim.sh)"
fi

if [ "${JIT:-0}" = 1 ] && [ -d "$SHIM" ]; then
  sec "6. FlashInfer sampling module: real JIT build into a throwaway workspace"
  env CUDA_HOME="$SHIM" PATH="$V/bin:$PATH" FLASHINFER_WORKSPACE_BASE="$T/fi" "$V/bin/python" -I -c "
import time; t = time.time()
from flashinfer.jit.sampling import gen_sampling_module
gen_sampling_module().build_and_load(); print('sampling JIT OK in %.0f s' % (time.time() - t))" 2>&1 | tail -3
fi

echo; [ "$FAIL" = 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $FAIL
