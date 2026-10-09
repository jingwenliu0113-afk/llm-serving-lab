# NEXT

## 目前位置

**Core — Stage A（§1–5）與 Stage B（§6）都已凍結**（2026-10-09）。Advanced / Optional 皆未觸碰。

- Stage A：harness commit `eb6eef1`，`freeze.mac = true`。
- Stage B：harness commit `19446d7`，凍結 commit `b42614a`，`freeze.nvidia = true`。
- 模型（008）：
  - Core A：Qwen3.5 4B + 9B GGUF，BF16 / Q8_0 / Q6_K / Q4_K_M。
  - Core B：RedHatAI/Qwen3.5-9B-quantized.w4a16。
  - context：2K / 8K / 32K。
- Decision log 001–021。細節以 decision log 為準，這裡只記現況與下一步。
- GitHub：`origin` 為 private repo。每個測試通過的 commit 都會自動 push；不 force push，不改寫 history。
  權重、GGUF、KLD、`results/**/raw/`、venv、credential 不進 git。

## 現況（2026-10-09 23:50）

### 使用者決定（2026-10-09 上午）

- **019**：
  - Mac 正式 run 用**自動電力模式**加上**原廠 70W adapter**，並記錄 power state。
  - 正式 Mac Core A 的電池下降停止規則：13:xx 依使用者決定刪除。
  - 高效能模式（powermode 2）也可以，但同一 session 內要一致。
  - Core B 的 `--max-num-seqs` 改為 **128**，grid 上限改為 **128**。
  - Stage B 不拆分。
- **020**：正式 quality 改在 **RTX** 上執行，flag set 與 hypothesis 不變。
  - 4B 一起搬：使用者 2026-10-09 確認。
  9B BF16 會少量 spill 到系統記憶體，運算仍全部在 CUDA 上完成。

### RTX（gpu-server，GPU lock 下依序執行）

- **正式 RTX Core A 完成**（`formal-20261009-rtx-a1`，10:42–13:11）：
  - 164 / 164 ok，sentinel 漂移 0.06%，env 有效，session 開頭重量的 B = 462 MiB（沒有 drift）。
  - 判定（`analysis.json`）：H-A1、H-A3、H-A4a、H-A4b 為 not refuted；**H-A2 為 refuted**。
    - 推翻 H-A2 的是 4B-Q4_K_M 對 9B-Q8_0 這一組：觀測比 2.24，預測比 2.99，偏差 −25%，門檻是 ±20%。
    - 15 組配對的偏差全部是負的：bytes 少的配置都比純頻寬比例預測的慢。原因屬於推測，Core 不驗證。
  - 圖：`results/core_a/figures/{decode,prefill,memory}-rtx-formal.svg`。
- **正式 Core B uniform**（`formal-20261009-coreb-uniform`）：13:11 開始，23:30 時 80 / 102 列，全部 ok。
  - 剩下 22 個 32K 點，每點約 23 分鐘，預計 2026-10-10 早上 8 點左右結束。
- **uniform 之後的自動接續**（背景 `chain_rtx_after_uniform.sh`，每一步都在 GPU flock 下、ssh 前景執行，遇到問題就停）：
  1. 唯讀檢查 uniform session：formal、env 有效、102 列全部 ok。不符合就停止，不部署、不開跑任何東西。
  2. 確認 GPU 上沒有 compute process 後，部署 `2c2e764`（bundle 事先建好），並核對 `DEPLOYED_COMMIT`。
  3. RTX quality pilot（`pilot-<日期>-rtx-q`，005 pilot protocol，N_thr 6 / P 0），接著產生遮蔽版報告。
  4. 正式 Core B mixed（`formal-<日期>-coreb-mixed`），C = 各 L 的 uniform C_knee，使用 pin 過 sha256 的 mixed dataset。
  - `core_b_plan.py mixed` 現在會拒絕非 formal 或 env 無效的 uniform session（與 `core_b_report` 同一規則，018 補充）。

- （歷史）正式 RTX Core A `formal-20261009-rtx-a1`。
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

- **BF16 參考檔已傳到 WSL**（020）：9B、4B 都在 WSL 上以 sha256 驗證完成（20:09）。quality 資料檔 14 個也已核對。
- **注意**：正式 RTX session 進行中**不得重新部署**到 WSL。
  - 中途部署會悄悄換掉之後 process 使用的程式，而且不會留下標記（020 第 7 點）。
  - 審查後三個 runner 都加了 provenance 檢查，但進行中的 session 跑的是 `b42614a`，沒有這個檢查。
  - 等 GPU 上的正式 session 結束後，才部署 `run_quality` 的 RTX 支援。

### Mac

- **使用者 2026-10-10 01:45：暫時不要在 Mac 上開始新的重負載**（使用者自己有工作要跑，好了會通知）。
  - a2 已跑完且有效（見下）。RTX 照常進行（使用者只用 Mac）。
  - Mac 目前沒有排任何重負載；只做輕量的分析、圖與 commit。

- **2026-10-09 13:xx：使用者刪除 019 的電池下降停止規則，並要求重跑。**
  - `formal-20261009-a1` 依舊規則作廢，照實保留。
  - 重跑 `formal-20261009-a2` 在 Mac lock 下、固定 commit 的 worktree 中執行。
  - 適用的規則：012、016 充電閘門、70W adapter、同一 session 內 powermode 一致。

- **正式 Mac Core A 完成**（`formal-20261009-a2`，2026-10-09 20:15 至 2026-10-10 02:55，commit `8f8dc9a` 的 worktree，powermode 2，70W adapter）：
  - 164 / 164 ok，sentinel 漂移 1.78%（門檻 3%），env 有效，沒有充電暫停；每筆 env 的 commit 都相同且 dirty = False。
  - 判定（`analysis.json`）：H-A1、H-A3、H-A4a、H-A4b not refuted；**H-A2 refuted**。
    - 推翻 H-A2 的是與 RTX 相同的配對：4B-Q4_K_M 對 9B-Q8_0，觀測比 2.22，預測比 2.99，偏差 −25.8%。
    - 15 組配對中 13 組偏差為負。
  - 圖：`results/core_a/figures/{decode,prefill,memory}-mac-formal.svg`。worktree 已移除（內容與主 repo 的副本逐檔相同）。
- 019 的 harness 已上線，並經過 review workflow：確認 10 項，均已修正。

## 下一步

1. **RTX**（GPU lock，依序）：
   1. 正式 Core B uniform（進行中）。
   2. 自動接續：檢查 uniform → 部署 `2c2e764` → RTX quality pilot → 正式 Core B mixed（見上）。
   3. 把 uniform、mixed、quality pilot 的 session 取回 Mac 並 commit（`raw/` 不進 git）。
   4. 看 quality pilot 的遮蔽版報告：self-KLD 0、重跑完全相同、9B BF16 的 spill 量。通過才排正式 quality（兩個 size）。
   6. 報告：
      - `core_a_report.py`（RTX 帶 `--v-total-mib 16303 --b-mib 811 --c-ctx-mib -89`）。
      - `core_b_report.py` 與 `core_b_figures.py`。
      - `quality_analysis.py`。
      - frontier 圖，要註明 quality 是在 wsl2-cuda 上量的。
2. **Mac**：正式 Core A 已完成。依使用者要求，在使用者通知之前不在 Mac 上開始新的重負載（目前也沒有排任何 Mac 重負載）。
3. **deployment decision table**（SPEC）：
   - 規則 decision 021 已 commit（`41798d7`），generator `benchmarks/decision_table.py` 與 40 個測試已 commit（`278ff82`）。
     兩者都在 a2、正式 quality、正式 Core B 的分析出現之前（021 R0）。
   - 設計 agent 不讀 `results/`；我已看過 RTX Core A 的結果，021 已照實揭露。
   - review workflow 完成（確認約 13 項，駁回 4 項），修正已 commit（`31973d5`），全部在看到新數據之前。
     其中 R11 的淘汰語意有修訂（只由當下未被勝過的成員淘汰，避免 KLD 與 NLL 的隱含跨 metric 遞移），已記在 021。
   - generator 的第一次真實執行要在所有正式分析都完成之後，輸出照原樣 commit（R0）。
4. **全部正式數據完成後**：跑 `decision_table.py`，寫結論。

## Blocked / 需要使用者

目前沒有需要使用者決定的項目。

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
  - 4B 的 quality 也在 RTX 上量：使用者確認（020）。
