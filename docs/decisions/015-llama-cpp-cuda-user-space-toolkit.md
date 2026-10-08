# 015 — llama.cpp CUDA build 改用 user-space 的 CUDA 13.4.1（NVIDIA pip wheels）

- 狀態：Accepted（2026-10-08）。build 與 `--version` 驗證，待 GPU lock 空出後進行，補記在 002 的驗證紀錄。
- 日期：2026-10-08
- 取代：002 的「toolkit = CUDA 12.8（apt `cuda-toolkit-12-8`）」，以及 002 addendum 的 apt 子集。llama.cpp 的 commit / build 與 CMake flags 不變。
- 時點：RTX 端沒有任何 Core A 數據，Stage B 尚未凍結。

## 背景

- 使用者在 WSL 只安裝了 Core B 需要的 `gcc g++ libc6-dev`，並指示：除非 user-space 的 CUDA 路線確實失敗，否則不要再要求安裝系統 CUDA toolkit。
- 試過 pip 的 CUDA 12.8：`nvidia-cuda-nvcc-cu12==12.8.93` 這個 wheel **只有 `ptxas`、nvvm 與 crt headers，沒有 nvcc 本體**，所以 user-space 拿不到 CUDA 12.8 的 nvcc。
  當時建好的 `.venv-cuda128` 已刪除。
- 002 選 12.8 的理由是：
  - 此 pin 的 llama.cpp CI 用 **12.8.2 與 13.4.1** 兩個版本 build。
  - 但 WSL-Ubuntu 的 apt repo 只提供到 13.3，可安裝且經 CI 測過的版本只剩 12.8。
- NVIDIA 的 CUDA 13 pip wheels（`nvidia/cu13` 合併目錄）**附有完整的 nvcc**（13.4.92，即 CUDA 13.4.1），不需要 sudo。

## 決策

1. llama.cpp 的 CUDA toolkit 改為 **CUDA 13.4.1**，來源是 NVIDIA 的 pip wheels，同一批釋出（2026-09-15/16）：
   - `nvidia-cuda-nvcc`、`-crt`、`-runtime`、`-nvrtc`、`nvidia-nvvm`：13.4.92。
   - `nvidia-cublas`：13.4.2.4。
   - `nvidia-cuda-cccl`：13.3.4.3.1。
   - 以 `configs/llama-cuda.requirements.in` 產生帶 hash 的 `configs/llama-cuda.lock.txt`，在 WSL 中產生。
   - 安裝在專用的 `~/llm-serving-lab/.venv-cuda134`，與 vLLM 的 venv 分開。
2. `scripts/wsl_cuda134_root.sh` 建立 toolkit root `~/llm-serving-lab/.cuda-root-13.4`：
   - `bin`、`include`、`nvvm`、`lib` 以 symlink 指向 `nvidia/cu13`。
   - `lib64/` 補上 CMake FindCUDAToolkit 需要的檔案：unversioned 與 `.so.13` 的 `libcudart`、`libcublas`、`libcublasLt`，以及 `libcudart_static.a`、`libcudadevrt.a`。
   - `lib64/stubs/libcuda.so` 指向 WSL 的 `libcuda.so`（soname `libcuda.so.1`），供 `CUDA::cuda_driver` 使用。
   - 已確認 nvcc 不解析 symlink：`_HERE_` 是 root 下的 `bin`，所以 `-I include` 與 `-isystem include/cccl` 都指向這個 root。
3. `scripts/build_llama_cpp.sh cuda` 的設定：
   - `CUDA_ROOT` 預設為 `<repo>/.cuda-root-13.4`。
   - 匯出 `CUDACXX=$CUDA_ROOT/bin/nvcc`，並傳入 `-DCUDAToolkit_ROOT=$CUDA_ROOT`。
     WSL 的 PATH 帶有 Windows 的 CUDA bin 目錄，所以一律明確指定。
   - 只在 cuda backend 生效；Metal 的指令不變。
   - build-record 記錄 toolkit flags 與 nvcc 版本。
   - `-DCMAKE_CUDA_ARCHITECTURES=120a-real` 等 backend flags 不變。
4. `pins.toml` `llama_cpp.toolchain.wsl2_cuda.cuda_toolkit = "13.4"`。
5. 不會用到 `/usr/local/cuda`，不需要 sudo，也不安裝任何 Linux driver。

## 為什麼

- 13.4.1 本來就是此 pin 的 CI 版本之一。002 不選它，只是因為 apt 裝不到，而 pip 可以。
- 元件版本以 lock 與 hash 固定，可重現，也與 vLLM 的 venv 隔離。
- driver 617.42 支援 CUDA 13.4，不需要動 Windows。

## 放棄的替代方案

- **請使用者以 sudo 安裝 apt 的 CUDA 12.8 子集**：違反使用者「不要再要求安裝系統 CUDA」的指示，而且 user-space 路線可行。
- **pip 的 CUDA 12.8**：wheel 不含 nvcc。
- **直接使用 vLLM venv 內的 CUDA 13**：版本混雜（nvcc 13.4.92、cudart 13.0.96、cuBLAS 13.1.1.3），不是 CI 測過的組合。
  而且 llama.cpp 會與 vLLM 的依賴綁在一起。
- **conda / micromamba 的 cuda-nvcc 12.8**：要多引入一整套套件管理器，管理面積比 pip wheels 大。

## 對實驗的影響

- Core A 的 RTX 結果是以 CUDA 13.4.1 build 的 llama.cpp 量得。報告與 build-record 要註明這點。
- Mac 端（Metal）不受影響。
- 因為只有一個 CUDA 版本，不做 CUDA 12.8 與 13.4 的比較，這也不在研究問題內。
