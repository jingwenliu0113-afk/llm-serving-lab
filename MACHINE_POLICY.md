# MACHINE_POLICY

## 核心原則

**Claude Code 與專案控制中心都在 Mac。**

Mac 是主要開發機：所有一般開發、Git、Claude Code session 都留在 Mac。

桌機為 Windows + WSL2 + RTX 5070 Ti，只負責 NVIDIA-specific 執行。

## Mac 上負責

- benchmark harness
- Core A Apple Silicon benchmark
- quality evaluation〔2026-10-09 起依使用者決定改在 RTX 執行，見 decision 020〕
- configs / scripts
- 統一結果分析

## Mac Docker Desktop（Core 階段不使用）

- Core A 在 Mac 使用 native `llama.cpp`（Metal）；Docker 在 macOS 上無法使用 Metal（001）。
- Core B NVIDIA serving 在 `gpu-server` 的 WSL2 `Ubuntu-22.04` 執行。
- 不得為了方便而在 Mac 啟動 vLLM 或任何 model serving 的 Docker container。
- 不得執行任何會影響 Mac Docker Desktop 全域狀態的指令（例如啟動 / 停止 Docker Desktop、修改其設定、pull / prune image、建立或刪除 volume / network）。

## RTX 桌機負責

遠端可用時：

- Core A NVIDIA benchmark
- Core B vLLM benchmark
- CUDA / GPU profiling
- quality evaluation（2026-10-09 起，decision 020；GPU flock 下執行）

路徑：

```text
Mac
→ Tailscale / SSH
→ Windows
→ wsl.exe
→ WSL2 Ubuntu
→ RTX 5070 Ti
```

### 遠端環境（使用者已實測確認）

| 項目 | 值 |
|---|---|
| SSH alias | `gpu-server`（Mac 已可免密執行 `ssh gpu-server`） |
| 主機架構 | Windows + WSL2 |
| WSL distro | `Ubuntu-22.04` |
| WSL user | `wenn0113` |
| GPU | NVIDIA GeForce RTX 5070 Ti 16GB |
| GPU 可見性 | `ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- nvidia-smi'` 可正常看到 RTX 5070 Ti |
| flock | WSL 內已有 `/usr/bin/flock` |

遠端命令的基本形式：

```bash
ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- bash -lc "<command>"'
```

### 遠端 Python 環境

- `~/venvs/gpu-check` 只是 CUDA 驗證環境，**不是**本 Project 的正式 Python 環境，本 Project 不使用它。
- 需要 GPU Python 環境時，建立本 Project 自己的隔離環境：
  - `uv` 以 user scope 安裝到 `~/.local/bin/uv`（pinned 版本，不修改 shell rc）；不污染 system Python，不用 sudo。
  - 專案目錄與環境放在 WSL Linux filesystem：`/home/wenn0113/llm-serving-lab/`（`.venv` 在其中）。
    **不得**放在 `/mnt/c/...`。
- Hugging Face / model cache：`HF_HOME=/home/wenn0113/.cache/huggingface`（WSL Linux filesystem），
  遠端指令一律明確設定，不得放在 C 槽掛載路徑。
- 大型檔案的保留與刪除規則見 `docs/STORAGE_PLAN.md`。

## 遠端不可用

若桌機關機、遠端 GPU 不可用，或 GPU lock 被占用：

- 不得卡住整個 Project，改做其他可在 Mac 完成的 Core 工作。
- NVIDIA-specific 工作標記 `blocked`；若確實造成 blocker，再記錄到 `NEXT.md`。
- 繼續 Mac benchmark harness、quality、Apple Silicon benchmark、分析。
  〔020：正式 quality 改在 RTX 執行。遠端不可用時，正式 quality 只標記 blocked；Mac 只做 quality 的程式與分析，不跑正式 quality。〕

## Credential 安全

只使用：

```bash
ssh gpu-server
```

不得要求、索取、搜尋、讀取、保存或修改任何 credential，包括 SSH 密碼、SSH 私鑰、Tailscale key、sudo 密碼與 token。

## GPU Hard Lock

正式 NVIDIA / vLLM benchmark 必須取得 WSL 內：

```text
/tmp/ai-portfolio-gpu.lock
```

正式 Heavy 工作必須使用 `flock -n`，例如：

```bash
ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- bash -lc "flock -n /tmp/ai-portfolio-gpu.lock <heavy-command>"'
```

本 Project 中哪些工作需要 GPU lock：

- **需要**：正式 NVIDIA serving benchmark，以及 vLLM throughput / TTFT / TPOT / P95 等重工作業。
  〔020：quality 評測（`llama-perplexity`，wsl2-cuda），包括 pilot 與正式 run，一律需要。〕
- **不需要**：一般程式開發、設定、文件，以及小型啟動 smoke test。

若 Text-to-SQL 已取得 lock：

- 不收正式 NVIDIA performance numbers。
- 可繼續 Mac 端分析、code、tests、quality eval。
  〔020 之後，這裡的 quality eval 只指程式與分析。正式 quality 需要這把 GPU lock；lock 被占用時標記 blocked，不改到 Mac 跑。〕
- 不得搶鎖或終止對方 process。

## Mac Hard Lock

本 Project 的 Mac 正式長時間 run 屬於 Mac heavy workload，必須使用共享 lock：

```text
/tmp/ai-portfolio-mac-heavy.lock
```

- 適用：Core A Apple Silicon 正式 benchmark、正式 quality 評測（約 19.5 小時）。
  〔020：正式 quality 不再在 Mac 執行，改用 GPU Hard Lock。〕
- GGUF 產出（`scripts/produce_gguf.py produce`：BF16 轉換 + 量化，每個 size 約 20–40 分鐘的 CPU / IO 重負載）
  也經 `tools/with-mac-lock` 執行，避免在其他專案正式量測期間跑重負載（2026-10-08）。
- 使用本 Project 的 `tools/with-mac-lock <heavy-command>`。它與 Incident Investigator / Rust HNSW 共用同一個 lock path
  與相同語意（`mkdir` lock + `owner.txt`），是本 Project 自己的 script，不是跨專案共用 framework。
- 不得與 Incident Investigator / Rust HNSW 的正式重工作業同時執行。lock 被占用時直接放棄，改做其他 Mac 輕量工作。
- 不得自行刪除 lock，包括疑似殘留的 stale lock；需由使用者確認後手動清除。
- pilot / smoke 等非正式 run 不取 lock；但若 lock 被他人占用，也不執行，避免干擾對方的正式量測。

## Core A / Core B

- Core A 才 sweep quantization。
- Core B 固定一種 precision：固定使用單一 quantized checkpoint，不擴大模型選擇矩陣。
  原因是 RTX 5070 Ti 16GB 需要保留 KV cache / concurrency 空間。
  目前依 008 為 `RedHatAI/Qwen3.5-9B-quantized.w4a16`（取代原本的 `Qwen3-8B-AWQ`）。
- Mac / RTX 結果明確標示 hardware / backend。
- 不做「誰比較快」排行榜。

## 外部費用（2026-10-08 使用者指示）

- 任何會產生外部費用的操作（付費 API、雲端 GPU、付費資料服務等），**第一次執行前**必須先停下來向使用者說明：
  用途、預估請求數、token / 運算量、預估成本。**未經明確批准不得產生付費用量。**
- 不受此限：既有本地硬體運算（Mac、`gpu-server` 的 RTX 5070 Ti）、一般網路下載、Claude Code 本身的用量。
  大型下載仍需事先同意（見上文）。
- 目前 Core 的所有步驟都只用本地硬體與公開下載，沒有付費用量。
