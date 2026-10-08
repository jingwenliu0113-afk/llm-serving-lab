# 014 — Core B 的 JIT 工具鏈：最小 apt 套件，以及給 FlashInfer 用的 user-space CUDA_HOME

- 狀態：Accepted（2026-10-08）。CUDA_HOME shim 能否實際編譯成功，要等使用者裝好 compiler 後，以 `scripts/wsl_toolchain_check.sh` 驗證；
  失敗時改用文末的 fallback，並在本文補記。
- 日期：2026-10-08
- 時點：Core B 沒有任何數據。S1 smoke 在啟動階段就失敗了（`results/core_b/smoke/20261008-s1-no-cc`），屬於 pass / fail 的啟動測試（006）。
- 依據：一個唯讀的 workflow，在 gpu-server WSL 盤點，並讀了 vLLM 0.30.0 / triton 3.7.1 / flashinfer 0.6.18.post1 的原始碼。
  同時在隔離環境做 apt 模擬（不需 sudo、沒有安裝任何東西），並由兩個對抗式驗證者複查必要性與充分性。

## 背景

- S1 的錯誤：`triton/runtime/build.py` 回報 "Failed to find C compiler"。
  Triton 在 `profile_run` 時要編譯 `CudaUtils` 的 C launcher stub，但 WSL 裡沒有任何 compiler。
- 裝了 compiler 之後還有下一關，已在不使用 GPU 的情況下重現：
  - vLLM 的 V2 warmup 用 `SamplingParams.for_sampler_warmup()`（temperature 0.9、top-p、top-k）。
  - `flashinfer_sampler_supported()` 只檢查 compute capability，所以 warmup **一定會** JIT 編譯 FlashInfer 的 sampling module。
  - 這個 JIT 需要：
    - nvcc ≥ 12.9（FlashInfer 對 SM 12.x 的要求；CUDA 12.8 會被拒絕）。
    - cuRAND headers（`sampling.cuh`）。
    - PATH 上的 ninja。
    - 一個 C++ compiler，用來編譯 host code 與連結。
  - 錯誤訊息：`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`。
- vLLM 的 `has_flashinfer()` 在沒有 flashinfer-cubin 的情況下，等於 `shutil.which("nvcc") is not None`。
  它控制 FlashInfer autotune warmup 等十幾條路徑，所以 nvcc 是否在 PATH 上，本身就會改變 server 的行為。

## 決策

1. **Core B 唯一需要 sudo 的 apt 套件：`gcc g++ libc6-dev`**，用 `--no-install-recommends` 安裝，下載約 58.5 MiB。由使用者自己安裝。
   - `g++` 是 FlashInfer JIT 的 host compiler 與 linker。
   - `libc6-dev` 提供 C headers 與 crt，在 `--no-install-recommends` 下不會被自動帶入。
   - 不需要：
     - `python3-dev`：uv 的 CPython 3.12.13 自帶 `Python.h`。
     - apt 的 cmake / ninja：cmake 用 uvx，ninja 用 venv 內的。
     - 任何給 vLLM 用的 apt CUDA。
2. **FlashInfer 的 nvcc 用 venv 裡已 hash-lock 的 pip CUDA 13**（`nvidia/cu13`：nvcc 13.4.92、cudart 13、curand headers；`configs/vllm.lock.txt`），
   不新增任何系統套件。
   - `scripts/wsl_cuda_home_shim.sh` 用 symlink 建出 FlashInfer 需要的 CUDA_HOME 結構，位置在 `/home/wenn0113/llm-serving-lab/.cuda-home-cu13`。
     其中 `lib64/libcudart.so` 指向 cudart 13，`lib64/stubs/libcuda.so` 指向 WSL 的 libcuda。
3. **Core B server env**（smoke 與 runner 共用 `core_b_plan.server_env`）在 §4 原有的變數之外，加上：
   - `CUDA_HOME=/home/wenn0113/llm-serving-lab/.cuda-home-cu13`。
   - PATH 最前面放 `/home/wenn0113/llm-serving-lab/.venv/bin`（ninja）。
   - **nvcc 不放進 PATH**，讓 `has_flashinfer()` 維持 False。這與系統上沒有 CUDA toolkit 時 vLLM 的行為相同，
     只有 sampler 這個無條件的 JIT 會用到 nvcc。
   - report 記錄 `CUDA_HOME` 與 `PATH_PREFIX`。
4. **啟動前檢查**：`core_b_plan.toolchain_problems` 在啟動 server 之前檢查下列項目，任何一項缺少就拒絕啟動，不必等模型載入後才失敗。
   - cc、c++、ninja。
   - CUDA_HOME 下的 `bin/nvcc`、`include/curand.h`、`lib64/libcudart.so`、`lib64/stubs/libcuda.so`。
   - PATH 上**不能**有 nvcc。
5. **env_capture** 在 Linux 上另外記錄 host C compiler（`cc --version`）、`CUDA_HOME`，以及 `$CUDA_HOME/bin/nvcc` 的版本。
6. **S1 從新的 session 目錄重跑**。舊的 S1 失敗紀錄照實保留。依 008，S1→S3 的 argv / env 必須相同。

## 為什麼

- vLLM 在這張 GPU 上的預設 sampler 就是 FlashInfer。補齊它 JIT 所需的工具，比關掉它更貼近預設配置。
- nvcc、cudart、curand 都已在 vLLM lock 中 pin 住版本，不會引入新的、沒有 pin 的系統元件。
- nvcc 不放進 PATH 是最小偏離：只滿足那一個無條件的 JIT，不打開其他受 `has_flashinfer()` 控制的路徑。
- 啟動前檢查能讓每次缺件在幾秒內就報出來。否則要等約 60 秒的模型載入，才在 warmup 失敗。

## 放棄的替代方案

- **`VLLM_USE_FLASHINFER_SAMPLER=0`**：
  - 不需要 nvcc，但偏離 vLLM 在 sm_120 上的預設 sampler。
  - greedy 請求本來就幾乎用不到這個 sampler，但這是設定變更，而且沒有必要。
- **nvcc 放進 PATH**：
  - 會讓 `has_flashinfer()` 變成 True，打開 FlashInfer autotune warmup 等路徑，帶來更多 JIT 與更大的配置差異。
- **apt 安裝 CUDA 13.0 子集**：需要 sudo，版本也與 vLLM lock 中的 pip CUDA 不同。保留作為 fallback。
- **完整的 apt CUDA toolkit**：需要 sudo、體積大，而且 vLLM 用不到其中大部分。

## Fallback（只在 shim 經 `JIT=1 bash scripts/wsl_toolchain_check.sh` 驗證失敗時使用）

- 由使用者以 sudo 安裝：
  ```
  cuda-nvcc-13-0=13.0.88-1 cuda-cudart-dev-13-0=13.0.96-1 libcurand-dev-13-0=10.4.0.35-1
  ```
  cuRAND 版本與 lock 中的 `nvidia-curand` 相同。需要先裝好 NVIDIA wsl-ubuntu keyring。
- `FLASHINFER_CUDA_HOME` 改為 `/usr/local/cuda-13.0`，其餘規則不變。補記在本文。

## 對實驗的影響

- PREREGISTRATION §4 的 Core B 環境變數加上 `CUDA_HOME` 與 PATH 前綴（014）。
- 只影響 server 的啟動工具鏈，不影響任何 hypothesis、metric 或 grid。Core B 目前沒有任何數據。
- 之後 003 的 JIT 計時要以全新的 cache 量測：Triton 用 `TRITON_CACHE_DIR`，FlashInfer 用 `FLASHINFER_WORKSPACE_BASE`。
  `wsl_toolchain_check.sh` 的 JIT 測試只編譯到暫存目錄，不會預熱正式 cache。
