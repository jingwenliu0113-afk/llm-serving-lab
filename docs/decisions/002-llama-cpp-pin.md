# 002 — llama.cpp 版本 pin 與 build 方式

- 狀態：Accepted（macOS Metal：見文末驗證紀錄；WSL2 CUDA：blocked，待遠端）
- 日期：2026-10-08
- 對應：SPEC Day 1 #2「固定 llama.cpp 版本」、SPEC Core A「鎖定 runtime commit / version、compiler、執行 flags」
- 機器可讀：`configs/pins.toml` `[llama_cpp]`；build：`scripts/build_llama_cpp.sh`

## 背景

Core A 兩個平台必須用同一 runtime、同一 commit。llama.cpp 有兩套 tag：

| 形式 | 性質 |
|---|---|
| `bNNNN` | 每個 CI build 一個 tag（NNNN = `git rev-list --count`），GitHub Release 標 prerelease，附預編 binary；一天十幾個 |
| `vX.Y.Z` | 2026 起的 semver release（`make-release.yml` 在 "bump version" commit 打 annotated tag），release 只附 `nightly-tag.txt` 指回對應 `bNNNN` |

對應關係（查核確認）：`v0.4.1 = b10964`（09-14）、`v0.5.0 = b11146`（09-23）、`v0.6.0 = b11429`（10-05）。

## 決策

- **Pin：`b11146` = `v0.5.0`，commit `7fe450e19305b828c199d602c23a8337aaa1f03b`**
  （release 發布 2026-09-23T18:36:37Z；ggml 0.25.1）。一律以完整 SHA checkout，不以 tag 名稱。
- **兩平台都從 source build**，用 `scripts/build_llama_cpp.sh`，非 backend 的 CMake flags 完全相同：
  `-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_IS_DEV=OFF -DLLAMA_BUILD_NUMBER=11146
  -DLLAMA_BUILD_COMMIT=<full sha> -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`
  - Metal：`-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`
  - CUDA：`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CUDA_NCCL=OFF`；
    **toolkit = CUDA 12.8**（`cuda-toolkit-12-8`，nvcc 12.8.93，來自 NVIDIA 的 WSL-Ubuntu apt repo；2026-10-08 決定）。理由：
    - Blackwell 需要 ≥ 12.8；遠端 driver 617.42 支援到 CUDA 13.4，12.8 與 13.x 都可用。
    - 此 pin 的 llama.cpp CI 實際用 12.8.2 與 13.4.1 build；但 WSL-Ubuntu repo 目前只提供到 13.3，沒有 13.4。
      在可安裝的版本中，只有 12.8 是 CI 測過的。
    - 與 vLLM 的 CUDA 13.0 runtime（torch wheel 內建）版本不同。這沒有衝突，001 已規定兩者各自 pin、各自記錄。
  - Targets：`llama-bench llama-perplexity llama-completion llama-quantize`。
- cmake 以 `uvx --from cmake==4.4.3` 執行（不做全域安裝）。
- 平行度有上限：`JOBS` 預設為 CPU 數，CUDA backend 最多 8，避免 nvcc job 同時啟動造成 WSL2 VM OOM，
  進而波及其他專案的 process。
- 每次 build 輸出 `build-<backend>/build-record.txt`，內容包括：實際生效的 cache entries（`GGML_NATIVE` 等）、
  configure 時的 compiler 版本（CUDA 另記 nvcc 版本）、`JOBS`、`--version`。
- source tree guard 用完整的 `git status --porcelain`。`build-*/` 已被 llama.cpp 自己的 `.gitignore` 排除，
  因此任何 untracked 檔都代表有雜散 source 會被編進去，直接拒絕 build。
- `GGML_OPENMP=OFF` 兩邊一致：AppleClang 沒有 OpenMP，Linux GCC 預設會開，不關掉就會造成 CPU 端不對稱。

## 為什麼是 v0.5.0，而不是更新的版本

1. 它是 semver release 點，不是隨手一個 nightly。
2. 已發布 15 天，有足夠時間暴露 regression。「≥ 2 週」是本專案的 heuristic，**不是** SPEC 規則。
3. `v0.6.0` 只有 3 天，而且 `llama-perplexity` 在 09-30 遷移到新的 batch API（quality 的主要工具）。
   v0.6.0 另外新增 Metal tensor-API 的 flash-attention kernel，但 tensor API 在 M5 之前的晶片預設關閉，
   M4 Pro 仍走 simdgroup FA kernel，所以這一點**不是**拒絕 v0.6.0 的理由（review 更正）。
   前提是 `GGML_METAL_TENSOR_ENABLE` / `GGML_METAL_TENSOR_DISABLE` 不設定；`env_capture` 會記錄它們。
4. 官方預編 binary 存在（`llama-b11146-bin-macos-arm64.tar.gz`，sha256 `1ad3f9ef…5711`；
   `…-ubuntu-cuda-12.8-x64.tar.gz`，sha256 `c2ab9e19…0e4e`），可作 cross-check。**但正式結果不用它們**，
   理由是 Linux 版用 `GGML_BACKEND_DL=ON`、`GGML_NATIVE=OFF` 加 CPU variants，macOS 版不是，兩平台 build parity 會被打破。

## 此 pin 已知問題與對應的 protocol 約束

| 問題 | 影響 | 約束（寫入 Week 1 harness / pre-registration） |
|---|---|---|
| #28849 auto-fit：未給 `-c` 時以 `n_ctx_train × slots` 配置 KV（09-26 才 revert，此 pin 內含） | 不給 `-c` 可能 OOM（#29521 在此 build 回報 Metal OOM） | `llama-completion` 一律明確給 `-c`、`-np 1`、`--fit off`、`-ngl 999`；`llama-perplexity` 一律明確給 `-c`、`--fit off`、`-ngl 999`（PPL / KLD 的 n_parallel 由工具設為 n_batch ÷ n_ctx，downstream 另加 `-np 16`）。`llama-bench` 不接受這些參數，它的 KV 由 `-p/-n/-d` 決定，且預設不做 fit |
| common tools 預設 `--fit on` 會改寫**未設定**的 `-c` / `-ngl` | 只在參數沒給時會遮蔽 OOM | 同上；`--fit off` 是保險，不是唯一防線 |
| `llama-bench` 不加 `-v` 時會丟掉 ggml 錯誤訊息（修正在此 pin 之後） | bench row 失敗時難以分類原因 | `llama-bench` 一律 `-v`，stderr 存檔；**一個 config 一個 process** |
| #29880 depth cache 只以 depth 為 key | 同一 process 掃 `-fa 0,1` 會出錯並默默重 prefill | 一個 process 只用一個 `-fa` 值 |
| #29858 `--poll 50` busy-wait（OpenMP OFF 時尤其明顯） | M4 Pro CPU/GPU 共用 SoC 功耗，spin 會干擾 GPU 速度 | `-t`（N_thr）與 `--poll`（P）依平台各自固定；選擇規則見 PREREGISTRATION §4（Mac 於 Stage A pilot、RTX 於 Stage B），預設 `--poll 0` |
| `n_ctx` 會 pad 到 256 的倍數 | `-d D -n 128` 的 KV 不一定等於 `-c` | context 長度一律取 256 的倍數（見 004） |
| #28877 sm_120 上 PDL 相關 misaligned address（open） | 只在 CUDA | 記錄 `GGML_CUDA_PDL`，出現時再處理 |
| `-fa on` 強制開啟時不會跑 device probe | 理論上可能默默退回 CPU attention | pilot 先以 `-fa auto` 確認 log 出現 "Flash Attention enabled"，再固定 `-fa on`（合法值為 `on/off/auto`） |

## 放棄的替代方案

- **`v0.6.0` / 最新 nightly**：太新，理由如上。
- **Homebrew llama.cpp**：版本會跟著 brew 移動，也拿不到相同的 flags。
- **兩邊用官方預編 binary**：build 設定不一致；Linux 版另需約 24.04 世代的 glibc（未驗證）。

## 對實驗的影響

- 每一筆結果都必須帶 `build_number = 11146` 與 commit；quality 數字綁定這個 build。
- 更換 pin = 新的 decision log + 受影響的量測全部重跑，不得混用不同 build 的數字。
- 量測能力摘要（給 Week 1 harness 用，細節於 pre-registration 凍結）：
  - engine TTFT(L) = `llama-bench -p L -n 0` 的 `avg_ns`（純 prefill，不含 tokenize / HTTP）
  - 生成速度(L) = `llama-bench -p 0 -n 128 -d (L-128)` 的 `avg_ts`
  - 記憶體：`llama-completion` / `llama-perplexity` 的 memory breakdown（`-lv 4` 或 `--log-jsonl`
    event `fit_memory_breakdown`）；Mac 另用 `/usr/bin/time -l`，WSL2 另用 `nvidia-smi memory.used` 輪詢再減去 idle baseline
    （WSL2 不支援 per-process 查詢）。

## 驗證紀錄

「verified」的定義：用 `scripts/build_llama_cpp.sh` 在 pin 上 build 成功、`--version` banner 與 pin 一致、
backend device 能初始化。帶模型的 smoke test 屬於 Week 1 #1，不在這裡宣稱。

- **macOS Metal：verified（2026-10-08）**
  - `version: 0.5.0 (build 11146, commit 7fe450e19305b828c199d602c23a8337aaa1f03b)`
  - `built with AppleClang 21.0.0.21000334 for Darwin arm64`；cmake 4.4.3；build 1 分 24 秒
  - `--list-devices`：`MTL0: Apple M4 Pro (38338 MiB, 38338 MiB free)`（= Metal recommendedMaxWorkingSetSize ≈ 37.4 GiB）
  - 生效 flags 摘要：`GGML_METAL=ON`、`GGML_METAL_EMBED_LIBRARY=ON`、`GGML_NATIVE=ON`（`-mcpu=native`）、
    `GGML_ACCELERATE=ON`、`GGML_BLAS_VENDOR=Apple`、`GGML_OPENMP=OFF`、`GGML_BACKEND_DL=OFF`。
    完整內容在 `third_party/llama.cpp/build-metal/build-record.txt`（不進 git，但每次 build 都可重產）。
- **WSL2 CUDA：blocked**。
  - 2026-10-08 先是 `ssh gpu-server` 無法解析；SSH 已於同日恢復。
  - 現在缺的是 WSL 內的 compiler 與 CUDA 12.8，待使用者以 sudo 安裝（見下方 addendum）。

## Addendum（2026-10-08）：CUDA 12.8 改裝最小子集，並明確指定 nvcc

- **不裝完整的 `cuda-toolkit-12-8`**，改裝 build 實際用到的子集（由 NVIDIA wsl-ubuntu repo 的 index 遞迴解析 Depends，並做 apt 模擬）。
  nvcc 與 cuBLAS 的版本與完整 toolkit **相同**：
  - 子集：`make cuda-nvcc-12-8=12.8.93-1 cuda-cudart-dev-12-8=12.8.90-1 cuda-driver-dev-12-8=12.8.90-1 cuda-cccl-12-8=12.8.90-1 libcublas-dev-12-8=12.8.5.5-1 libcublas-12-8=12.8.5.5-1`。
    `build-essential` 由 `cuda-nvcc-12-8` 依賴帶入。
  - 各套件的用途：
    - `cuda-nvcc-12-8`：nvcc 12.8.93，與本 pin 相同。
    - `cuda-cudart-dev-12-8`：`CUDA::cudart`。
    - `libcublas-dev-12-8`：`ggml-cuda` 連結 `CUDA::cublas`。
    - `cuda-driver-dev-12-8`：`CUDA::cuda_driver`（VMM）的 stub。WSL 的 libcuda 在 `/usr/lib/wsl/lib`，CMake 找不到。
    - `cuda-cccl-12-8`：`cub/cub.cuh`。
  - 大小：子集約下載 986 MiB、安裝 2.3 GiB。完整 toolkit 至少 7.76 GiB，還會從 Ubuntu 拉進 default-jre 與 X11。
  - 不包含任何 Linux driver 套件。WSL 的 driver 由 Windows 提供。
- **`scripts/build_llama_cpp.sh cuda` 明確設定 `CUDACXX=/usr/local/cuda-12.8/bin/nvcc`**。
  WSL 的 PATH 帶有 Windows 的 CUDA v13.3 / v12.8 bin 目錄，不能依賴 PATH 剛好正確。build-record 照舊記錄 nvcc 版本。
- 安裝會讓 `/usr/local/cuda` 指向 12.8，並加入 ldconfig 路徑。
  Core B 不使用它：server 一律明確設定 `CUDA_HOME`，nvcc 也不放進 PATH（014）。
  安裝後由 `scripts/wsl_toolchain_check.sh` 檢查兩件事：
  - ldconfig 中是否出現 `libcudart.so.12`。
  - server log 是否出現 cuDNN frontend 的 "Multiple libcudart" 警告。

