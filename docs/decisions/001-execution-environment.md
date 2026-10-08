# 001 — 執行環境（桌機 OS、Mac 端、harness 工具鏈）

- 狀態：Accepted（RTX 端細節待遠端可用時驗證，見「待驗證」）
- 日期：2026-10-08
- 對應：SPEC Day 1 #1「決定桌機 OS / 執行環境」

## 背景

Core A 要在 M4 Pro 48GB 與 RTX 5070 Ti 16GB 上用同一 runtime 跑；Core B 只在 RTX 上用 vLLM。
`MACHINE_POLICY.md` 已規定桌機為 Windows + WSL2，控制中心在 Mac。這份 log 把它落成可執行、可記錄的具體環境，
並寫下這個選擇對結果解讀的影響。

## 決策

| 項目 | Mac 端 | RTX 端 |
|---|---|---|
| OS | macOS 27.0.1（build 26A434），原生執行 | Windows host + **WSL2 `Ubuntu-22.04`**（22.04.5 LTS，glibc 2.35，kernel 6.6.114.1-microsoft-standard-WSL2） |
| 硬體 | Apple M4 Pro，48 GB unified memory，12 logical CPU | RTX 5070 Ti 16 GB（sm_120）；host CPU AMD Ryzen 5 7600（6 核 12 緒）；WSL 內 RAM 30 GiB |
| GPU backend | llama.cpp **Metal** | llama.cpp **CUDA**、vLLM（皆在 WSL2 內） |
| 連線 | 本機 | Mac → Tailscale / SSH → `ssh gpu-server` → `wsl.exe` |
| Harness 語言 | Python 3.12（uv 管理） | 同左，WSL 內用 uv 安裝同版 Python |

另外：

1. 每次 benchmark / quality run 都要用 `scripts/env_capture.py`（schema v2）產生環境紀錄 JSON，
   和結果放在一起。欄位對應 EXPERIMENT_RULES「Benchmark Environment」：
   - git commit、dirty 判定與 status 清單（`results/` 下未 commit 的結果檔不算 dirty）；
   - OS、runtime version（llama.cpp 由 `--llama-bin`；vLLM 由 `--vllm-python` 指向 vLLM venv 的 interpreter，
     一併記錄 torch、torch CUDA runtime、transformers、triton）；
   - CUDA / driver（nvidia-smi 的身分資訊與功耗 / 溫度 / 時脈 / power limit 兩組查詢；PATH 上的 nvcc 另記）；
   - power state（Mac：電源、`powermode`、`pmset -g therm`；RTX：enforced vs default power limit）；
   - background load（loadavg、Mac lock 是否被占用及 owner）；
   - 影響 kernel / engine 的環境變數（`GGML_*`、`CUDA_*`、`VLLM_*` 等）；
   - Mac `iogpu.wired_limit_mb`（決定 Metal working set）、config sha256。
2. 任何 probe 失敗不得默默略過，一律寫進 `probe_errors`。
3. `env_capture.py` 只用 stdlib，同一份檔案在 Mac 與 WSL 都能跑，不另寫平台專屬版本。
4. 專案改為 git repo（`git init`，branch `main`），因為 EXPERIMENT_RULES 要求記錄 git commit，
   pre-registration 也必須先 commit。

## 為什麼

- **WSL2 而非原生 Windows**：vLLM 不支援原生 Windows。若 llama.cpp 用原生 Windows、vLLM 用 WSL2，
  RTX 端就有兩套 driver 路徑（WDDM 原生 vs WSL2 paravirtualized），等於引入額外變因。統一在 WSL2 後，
  兩個 Core 共用同一個 Windows driver 與同一條 WSL2 GPU 路徑。CUDA toolkit / runtime 則**各自 pin、各自記錄**：
  llama.cpp 用 WSL 內安裝的 toolkit build（002），vLLM 用 torch wheel 自帶的 CUDA runtime（003），兩者版本可能不同。
- **不 dual-boot 原生 Linux**：桌機是 Windows host、按需開機的遠端 worker（PORTFOLIO 規則），
  改動桌機開機設定不在本專案權限內。
- **Mac 原生而非容器**：macOS 上的 Docker 無法使用 Metal。

## 放棄的替代方案

- 原生 Windows 版 llama.cpp + WSL2 版 vLLM：兩套環境，Core A / Core B 的 RTX 結果難以對齊。
- RTX 端裝 dual-boot Ubuntu：最接近 bare-metal，但超出權限，也違反「桌機只當遠端 worker」。
- 在 Mac 上用 VM / container：拿不到 Metal。

## 對實驗的影響（解讀時必須保留的 caveat）

1. **RTX 結果一律標示為「RTX 5070 Ti / CUDA / WSL2」，不能宣稱是 bare-metal Linux 數字。**
   WSL2 透過 paravirtualized GPU driver（Windows driver + dxgkrnl）存取 GPU；
   compute-bound kernel 大致接近原生，但 host↔device 傳輸、pinned memory 與 page-locked 行為可能不同。
2. **有效 VRAM < 16 GB**：Windows 桌面合成器與其他程式會占用部分 VRAM。fit / OOM boundary 必須以
   run 前的 `nvidia-smi memory.used` 為 baseline 一起記錄（`env_capture` 已收）。
3. **WSL2 預設只拿到 host 約一半的 RAM**（`.wslconfig` 可調）。模型以 mmap 載入時會受影響；
   `env_capture` 記錄的 Linux `MemTotal` 是 WSL VM 內的數字，不是 host 實體 RAM。
4. Mac 端要記錄電源來源與 `pmset powermode`（目前觀察到 `powermode 2`；依 macOS 慣例
   0 = automatic、1 = low power、2 = high power，raw 值與標籤都存，避免誤讀）。
   正式 run 的電源條件寫進 pre-registration。

## 遠端實測（2026-10-08，SPEC Day 1 #4；唯讀檢查，不需 GPU lock）

使用者完成遠端設定後（MACHINE_POLICY「遠端環境」），從 Mac 執行 `ssh gpu-server 'wsl.exe -d Ubuntu-22.04 -- …'` 實測：

| 項目 | 結果 |
|---|---|
| Driver | KMD 617.42（WSL 內 NVIDIA-SMI 615.78.05），CUDA UMD 13.4；≥ R580，符合 vLLM 需求 ✅ |
| GPU | NVIDIA GeForce RTX 5070 Ti，`compute_cap = 12.0`，`memory.total = 16303 MiB` ✅（與 003 / 004 的假設相同） |
| 閒置狀態 | `memory.used` 1026 MiB、P8、`display_active = Disabled`。**這不是 PREREGISTRATION 定義的正式 B 量測**，只是參考 |
| 功耗 | `power.limit` = `enforced.power.limit` = `power.default_limit` = 300 W ✅ |
| flock | `/usr/bin/flock` ✅ |
| `env_capture.py` | 在 WSL 系統 Python 3.10.12 上執行成功，所有 GPU 欄位都解析正確 ✅ |
| 建置 / 執行工具 | **缺少** `nvcc`（沒有 CUDA toolkit）、`gcc`、`g++`、`make`、`cmake`、`uv` ⛔ |

### Day 1 NVIDIA 環境 smoke test（2026-10-08；小型 smoke，不取 GPU lock，執行前確認 GPU 閒置）

| 檢查 | 結果 |
|---|---|
| 執行前狀態 | `memory.used` 1013 MiB，`/tmp/ai-portfolio-gpu.lock` 無人持有 |
| PyTorch（`~/venvs/gpu-check`，僅作 CUDA 驗證用，未安裝任何東西） | torch 2.14.1+cu130、CUDA runtime 13.0、`cuda.is_available() = True` ✅ |
| 裝置 | RTX 5070 Ti、capability (12, 0)、arch list 含 `sm_120` ✅ |
| 計算 | 4096×4096 bf16 matmul 正常，結果全為 finite ✅ |
| 記憶體 | `torch.cuda.mem_get_info` total 16275 MiB（nvidia-smi 為 16303 MiB）、free 14767 MiB |
| 本 Project 環境的前提 | 系統 Python 3.10.12 有 `venv` 與 `ensurepip`（沒有系統 pip），可以不用 sudo 建立隔離環境 |

結論：driver / CUDA / PyTorch 正常。本 Project 自己的 llama.cpp CUDA build 與 vLLM 環境仍待建置工具（見下）。

- 建置工具與 CUDA toolkit 必須用 sudo 安裝。依 MACHINE_POLICY 不得取得 sudo credential，所以這一步需要使用者自行安裝（見 NEXT.md）。
- 仍待確認：Windows 的 HAGS 設定、`.wslconfig`，以及螢幕實際接在哪張卡上（`display_active = Disabled` 在 WSL 內可能不準）。
- 遠端副本不是 git repo，所以 `env_capture` 在遠端記不到 commit。harness 部署到遠端時必須一併帶上 Mac 端的 commit（Week 1 harness 的需求）。
