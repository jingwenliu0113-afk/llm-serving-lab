# NEXT

## 目前位置

**Core — SPEC「Day 1」（第一個 Core milestone）**：
- Mac 部分完成。
- 遠端 `gpu-server` 已可用；Day 1 NVIDIA 環境 smoke test 通過（driver / CUDA / PyTorch）。
- 剩下的 NVIDIA 工作（llama.cpp CUDA build、vLLM smoke test）等使用者安裝建置工具。

接著進行 Week 1 中 Mac 可做、且不需要大型下載的部分（benchmark harness）。Advanced / Optional 皆未觸碰。

## 上次完成（2026-10-08）

### Day 1 項目狀態

| # | SPEC Day 1 項目 | 狀態 | 產出 |
|---|---|---|---|
| 1 | 決定桌機 OS / 執行環境 | ✅ Windows + WSL2 `Ubuntu-22.04`（實測） | `docs/decisions/001`、`MACHINE_POLICY.md` |
| 2 | 固定 llama.cpp 版本 | ✅ pin `b11146` = `v0.5.0`（commit `7fe450e1…`）；Mac Metal 已 build 並驗證；RTX 用 CUDA 12.8 toolkit（已決定，未安裝） | `002`、`scripts/build_llama_cpp.sh` |
| 3 | 固定 vLLM 版本 | 🟡 `vllm==0.30.0`；**Core B 固定 `Qwen3-8B-AWQ`（使用者決定，無 fallback）**；smoke test 待建置工具 | `003` |
| 4 | 確認 NVIDIA driver / CUDA / framework | ✅ driver 617.42、CUDA UMD 13.4、sm_120、16303 MiB；PyTorch CUDA smoke 通過（bf16 matmul） | `001`「Day 1 NVIDIA 環境 smoke test」 |
| 5 | 選 model family | ✅ Qwen3 dense 8B / 14B；本地從 BF16 量化成 Q8_0 / Q6_K / Q4_K_M；context 2048 / 8192 / 32768 | `004` |
| 6 | 第一版 hypothesis、metrics、task set | ✅ DRAFT v1（未凍結；兩階段凍結） | `docs/PREREGISTRATION.md`、`005`–`007` |

### 已處理的使用者決定（2026-10-08）

1. **Mac 重負載 lock**：
   - 正式長時間 run（Core A Apple Silicon 正式 benchmark、約 19.5 h 的 quality）使用共享的 `/tmp/ai-portfolio-mac-heavy.lock`。
   - 已實作本 Project 自己的 `tools/with-mac-lock`（含測試），並寫入 `MACHINE_POLICY.md`「Mac Hard Lock」。
2. **大型下載**：
   - 不提前產生或下載約 45 GB 的 KLD 檔。KLD logits 是 Stage A 凍結後、正式 run 時才在本地產生。
   - 任何下載前都要先在 `pins.toml` 記錄來源、版本與 checksum（005）。
3. **Baseline commit**：`chore: establish core baseline`。
4. **Core B**：固定 `Qwen3-8B-AWQ`，移除 Qwen3-4B fallback；`tests/test_pins.py` 強制不得有 fallback。

### 其他

- `MACHINE_POLICY.md` 已整合遠端 GPU 資訊（`gpu-server` → WSL2 `Ubuntu-22.04` → `wenn0113` → RTX 5070 Ti），
  以及 GPU lock / Mac lock 規則。
- `env_capture.py` 已在 Mac 與 WSL 兩邊實機驗證；nvidia-smi 的測試 fixture 已換成真實輸出。

## 下一步

1. **Benchmark harness**（Week 1 #2–3，先寫測試）：plan / parse / runner 與 `docs/RESULTS_FORMAT.md`。
2. **Core A 模型**（需使用者同意下載）：
   - 依 `pins.toml` 已記錄的 revision 與 sha256，下載 `Qwen/Qwen3-8B` 與 `Qwen/Qwen3-14B` 的 safetensors（約 46 GB）。
   - 為 converter 產生 lock → 轉成 BF16 GGUF → `llama-quantize` 產生 3 個 quant → 把 sha256 / size 寫進 pins。
3. Mac 模型 smoke test → Mac pilot → Stage A 凍結 → 正式 run（用 `tools/with-mac-lock`）。

## Blocked

- ~~`ssh gpu-server` 無法連線~~ → **resolved**（2026-10-08，使用者完成設定並實測；MACHINE_POLICY「遠端環境」）。
- **遠端建置工具（需使用者以 sudo 安裝）**：WSL `Ubuntu-22.04` 內沒有 `gcc` / `g++` / `make` / `nvcc` / `cmake` / `uv`。

  ```bash
  sudo apt-get update && sudo apt-get install -y build-essential
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8     # 002：CUDA 12.8；只裝 toolkit，不要裝 Linux driver
  ```

  - `uv`：不需要 sudo，裝在使用者目錄；用來建立本 Project 的 Python 3.12 環境並執行 cmake。安裝方式由使用者決定。
    替代方案是系統 Python 3.10 的 `venv`，已確認可用，但與 Mac 端的 3.12 不一致。
  - 受影響的工作：llama.cpp CUDA build（002）、vLLM 安裝與 smoke test（003 gate）、RTX 正式實測、PREREGISTRATION Stage B。

## 待使用者確認

1. **Core A 模型下載**（約 46 GB safetensors，來源 / revision / sha256 已記錄於 `pins.toml`）：何時可以開始。
2. **遠端 `uv`**：要用官方安裝程式裝到使用者目錄，還是改用系統 Python 3.10 的 `venv`。
