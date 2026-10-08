#!/usr/bin/env bash
# Build llama.cpp at the pinned commit (docs/decisions/002-llama-cpp-pin.md).
#
# Usage: scripts/build_llama_cpp.sh metal|cuda
#
# Same source commit and same non-backend CMake flags on both platforms; only
# BACKEND_FLAGS differ. Output: third_party/llama.cpp/build-<backend>/bin/
# plus build-<backend>/build-record.txt (effective flags, compiler, cmake version).
set -euo pipefail

# Must match configs/pins.toml (enforced by tests/test_build_script.py).
LLAMA_COMMIT=7fe450e19305b828c199d602c23a8337aaa1f03b
LLAMA_BUILD_NUMBER=11146
LLAMA_REPO=https://github.com/ggml-org/llama.cpp
CMAKE_PIN=4.4.3

BACKEND=${1:?usage: $0 metal|cuda}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/third_party/llama.cpp
BUILD=$SRC/build-$BACKEND

case "$BACKEND" in
  metal)
    BACKEND_FLAGS=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
    ;;
  cuda)
    BACKEND_FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CUDA_NCCL=OFF)
    ;;
  *)
    echo "unknown backend: $BACKEND" >&2
    exit 2
    ;;
esac

COMMON_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_BUILD_IS_DEV=OFF
  -DLLAMA_BUILD_NUMBER=$LLAMA_BUILD_NUMBER
  -DLLAMA_BUILD_COMMIT=$LLAMA_COMMIT
  -DGGML_OPENMP=OFF
  -DLLAMA_OPENSSL=OFF
  -DLLAMA_USE_PREBUILT_UI=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
)
TARGETS=(llama-bench llama-perplexity llama-completion llama-quantize llama-tokenize)

# Bounded parallelism: a bare -j lets make start every nvcc job at once, which can
# OOM the WSL2 VM (and kill another project's process). ~2 GiB per nvcc job.
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN)}
if [ "$BACKEND" = cuda ] && [ "$JOBS" -gt 8 ]; then
  JOBS=8
fi

# GGML_NATIVE silently flips OFF when SOURCE_DATE_EPOCH is set; keep builds comparable.
unset SOURCE_DATE_EPOCH

cmake_() { uvx --from "cmake==$CMAKE_PIN" cmake "$@"; }

# 1. Fetch exactly the pinned commit (shallow; build number/commit are passed explicitly).
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" remote add origin "$LLAMA_REPO"
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$LLAMA_COMMIT" ]; then
  git -C "$SRC" fetch -q --depth 1 origin "$LLAMA_COMMIT"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
fi
HEAD=$(git -C "$SRC" rev-parse HEAD)
if [ "$HEAD" != "$LLAMA_COMMIT" ]; then
  echo "source at $HEAD, expected $LLAMA_COMMIT" >&2
  exit 1
fi
# build-<backend>/ is ignored by llama.cpp's own .gitignore (/build*), so any
# untracked file reported here is a stray source that would be compiled in.
if [ -n "$(git -C "$SRC" status --porcelain)" ]; then
  echo "source tree modified; refusing to build a non-pinned tree" >&2
  exit 1
fi

# 2. Configure + build.
cmake_ -S "$SRC" -B "$BUILD" "${COMMON_FLAGS[@]}" "${BACKEND_FLAGS[@]}"
cmake_ --build "$BUILD" --config Release -j "$JOBS" --target "${TARGETS[@]}"

# 3. Record what was actually built.
{
  echo "commit: $HEAD"
  echo "backend: $BACKEND"
  echo "cmake: $(cmake_ --version | head -1)"
  echo "common_flags: ${COMMON_FLAGS[*]}"
  echo "backend_flags: ${BACKEND_FLAGS[*]}"
  echo "jobs: $JOBS"
  echo "--- compiler versions (configure time) ---"
  grep -h 'COMPILER_VERSION "' "$BUILD"/CMakeFiles/*/CMake*Compiler.cmake 2>/dev/null || true
  if [ "$BACKEND" = cuda ]; then
    "$(sed -n 's/^CMAKE_CUDA_COMPILER:FILEPATH=//p' "$BUILD/CMakeCache.txt")" --version | tail -2
  fi
  echo "--- effective cache entries ---"
  grep -E '^(GGML_[A-Z0-9_]+|LLAMA_[A-Z0-9_]+|CMAKE_(BUILD_TYPE|C_COMPILER|CXX_COMPILER|CUDA_ARCHITECTURES|CUDA_COMPILER))(:[A-Z]+)?=' "$BUILD/CMakeCache.txt" | sort
  echo "--- version ---"
  "$BUILD/bin/llama-completion" --version 2>&1 || true
} > "$BUILD/build-record.txt"

cat "$BUILD/build-record.txt" | tail -4
echo "built: $BUILD/bin"
