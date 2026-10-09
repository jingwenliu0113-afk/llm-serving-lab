# NEXT

## 目前位置

**Core — Stage A（§1–5）與 Stage B（§6）都已凍結**（2026-10-09）。Advanced / Optional 皆未觸碰。

- Stage A：harness commit `eb6eef1`，`freeze.mac = true`。
- Stage B：harness commit `19446d7`，凍結 commit `b42614a`，`freeze.nvidia = true`。
- 模型（008）：
  - Core A：Qwen3.5 4B + 9B GGUF，BF16 / Q8_0 / Q6_K / Q4_K_M。
  - Core B：RedHatAI/Qwen3.5-9B-quantized.w4a16。
  - context：2K / 8K / 32K。
- Decision log 001–020。細節以 decision log 為準，這裡只記現況與下一步。
- GitHub：`origin` 為 private repo。每個測試通過的 commit 都會自動 push；不 force push，不改寫 history。
  權重、GGUF、KLD、`results/**/raw/`、venv、credential 不進 git。

## 現況（2026-10-09 10:50）

### 使用者決定（2026-10-09 上午）

- **019**：
  - Mac 正式 run 用**自動電力模式**加上**原廠 70W adapter**，並記錄 power state。
  - 正式 Mac Core A 有電池下降停止規則：30 分鐘內下降 ≥ 3 點就停止，整個 session 無效。
  - Core B 的 `--max-num-seqs` 改為 **128**，grid 上限改為 **128**。
  - Stage B 不拆分。
- **020**：正式 quality 改在 **RTX** 上執行，flag set 與 hypothesis 不變。
  - 4B 一起搬是我的建議，待使用者確認，見 Blocked 1。
  9B BF16 會少量 spill 到系統記憶體，運算仍全部在 CUDA 上完成。

### RTX（gpu-server，GPU lock 下依序執行）

- **進行中**：正式 RTX Core A `formal-20261009-rtx-a1`。
  - 10:36 開始，session 開頭會重量 B；pilot 估計上限約 3.6 小時。
  - 跑完自動接正式 Core B uniform `formal-20261009-coreb-uniform`：3 個 L × 3 次 restart，粗估約 10 小時。
  - 背景串接腳本：`chain_rtx_formal.sh`。
- **Stage B 值**（§6）：

  | 項目 | 值 |
  |---|---|
  | B | 811 MiB |
  | V_total | 16,303 MiB |
  | C_ctx | −89 MiB |
  | U | 0.91 |
  | K | 4.0 GiB |
  | num_gpu_blocks / block_size | 248 / 528 |
  | pin memory | 0 |
  | N_thr / P | 6 / 0 |
  | HAGS | 開 |
  | 超量測試 | spill |
  | C_KV（2K / 8K / 32K） | 35 / 13 / 3 |
  | grid 2K | 1, 2, 4, 8, 16, 18, 25, 32, 35, 49, 64, 70, 128 |
  | grid 8K | 1, 2, 4, 7, 8, 9, 13, 16, 18, 26, 32, 64, 128 |
  | grid 32K | 1, 2, 3, 4, 6, 8, 16, 32 |

- **BF16 參考檔傳到 WSL**（020）：10:38 開始，限速 1.2 MiB/s，先 9B 後 4B，約 6 小時（`queue_sync_refs.sh`）。
- **注意**：正式 RTX session 進行中**不得重新部署**到 WSL。
  - 中途部署會悄悄換掉之後 process 使用的程式，而且不會留下標記（020 第 7 點）。
  - 審查後三個 runner 都加了 provenance 檢查，但進行中的 session 跑的是 `b42614a`，沒有這個檢查。
  - 等 GPU 上的正式 session 結束後，才部署 `run_quality` 的 RTX 支援。

### Mac

- 正式 Mac Core A 在等兩個條件，見 Blocked 0。
  - 背景 watcher `wait_power_then_formal_a.sh` 每 2 分鐘唯讀檢查一次。
  - 條件一符合，就在 Mac lock 下自動開始 `batch_formal_a.sh`（seed 20261008、`-t 8`、`--poll 0`、016、019）。
  - 開跑時以當下 HEAD 建立 git worktree（`.worktrees/<session>`），session 在 worktree 中執行，跑完再複製回主 repo 的 `results/`。
    這樣我在主目錄的修改與 commit 不會影響進行中的 session。
- 019 的 harness 已上線，並經過 review workflow：確認 10 項，均已修正。

## 下一步

1. **RTX**（GPU lock，依序）：
   1. 正式 Core A（進行中）。
   2. 正式 Core B uniform（排隊中）。
   3. 部署最新 commit。
   4. 用 `core_b_plan.py mixed --uniform-session …` 產生 mixed plan，接著跑正式 Core B mixed。
      執行時加上 `--mixed-dataset 2048=… --mixed-dataset 8192=…`，sha256 已 pin。
   5. quality：
      1. 把資料檔傳到 WSL，sha256 要與 Mac 及 pins 一致。
      2. 跑 RTX quality pilot（`run_quality.py plan --mode pilot --platform wsl2-cuda --n-thr 6 --poll 0`，看遮蔽版報告）。
         pilot 要確認：self-KLD 0、重跑完全相同，以及 9B BF16 的 spill 量。
      3. 跑正式 quality。
   6. 報告：
      - `core_a_report.py`（RTX 帶 `--v-total-mib 16303 --b-mib 811 --c-ctx-mib -89`）。
      - `core_b_report.py` 與 `core_b_figures.py`。
      - `quality_analysis.py`。
      - frontier 圖，要註明 quality 是在 wsl2-cuda 上量的。
2. **Mac**：條件符合後由 watcher 自動開始正式 Core A。之後跑 `core_a_report.py` 與 `core_a_figures.py`。
3. **全部正式數據完成後**：寫 SPEC 要求的 deployment decision table，以及結論。

## Blocked / 需要使用者

0. **正式 Mac Core A 在等兩件事**
   1. **AC 的能源模式目前是「高效能」**（powermode 2；09:30 時是 0，09:56 讀到 AC 設定為 2）。019 要求「自動」。
      - 請到「系統設定 > 電池 > 能源模式 > 接上電源轉接器時」選「自動」。這是系統設定，我不會修改。
   2. **Mac heavy lock** 從 09:29 起由另一個專案持有（incident-investigator 的 pilot：`collect run --set pilot`）。
      - 依 MACHINE_POLICY 不搶占。
      - AC 被切到 powermode 2 的時間點與它重疊，可能是那個專案的設定。兩個專案若對電源模式的要求不同，要由使用者協調。
   - 兩件事都符合時，watcher 會自動開跑，不必再通知我。

1. **4B 的 quality 是否也在 RTX 上量**（020；正式 quality 開始前需要確認）
   - 我的建議是一起搬，理由：
     - H-A5(a) 是 9B-Q4_K_M 與 4B-Q8_0 逐 chunk 的配對 ΔNLL，兩個 size 跨 Metal / CUDA 會混入 backend 差異；
     - 4B BF16 只有 7.85 GiB，放得進 VRAM。
   - 在你確認前我照此準備：BF16 會傳過去、pilot 也會跑，但不開始正式 quality。

## 建議（非 blocker）

- 正式 Mac run 期間盡量少開前景程式。
  - 前景負載不會讓 session 作廢，但會增加 noise。
  - 4B pilot 時前景 loadavg 為 6–9，CV 5.9%；9B pilot 在深夜，CV 1.0%。

## 已解決

- 遠端 SSH（2026-10-08）。
- WSL compiler（使用者已安裝）。
- llama.cpp 的 CUDA 12.8 系統套件：改走 user-space 路線，不再需要（015）。
- 2026-10-09：
  - Mac 正式 run 解除暫停（019）。
  - Core B 選 A（019）。
  - Stage B 不拆分（019）。
  - quality 的 Mac 磁碟不足：quality 改在 RTX 執行，問題消失（020）。清理 scratchpad 釋出約 2.2 GB。
  - 019 中「quality 是否套用電池下降停止規則」：quality 改在 RTX，不再適用（020）。
