# 019 — 使用者決定（2026-10-09 上午）：Mac 正式 run 的電源條件、Core B `--max-num-seqs 128`、Stage B 不拆分、quality 磁碟

- 狀態：Accepted（使用者決定，2026-10-09）
- 日期：2026-10-09
- 時點：Stage A 已於 2026-10-09 01:15 凍結。**這是凍結後的變更**，但發生在兩個平台任何正式數據之前：
  - 正式 Mac Core A 依使用者指示暫停到今天上午。
  - 正式 quality 卡在磁碟。
  - 正式 RTX Core A 等 Stage B 凍結。
  - Core B 卡在 `--max-num-seqs`。

  pilot 只看過遮蔽版報告（012），沒有看過任何 hypothesis 的效果。
- 依 006，凍結後的任何變更都要記錄。本文改的是 protocol 參數：
  - §4 的 Mac powermode、AC adapter，以及新增的電池下降停止規則；
  - 007 的 `--max-num-seqs` 與 grid 上限。

  不改任何 hypothesis 或其門檻，也不改 metric、task set、資料切分。
- 來源：NEXT 的 Blocked 0–3，使用者 2026-10-09 上午的回覆。

## 1. Mac 正式 run：自動電力模式 + 原廠 70 W AC（取代 §4 的 powermode 2）

使用者決定：Mac 正式 benchmark 固定使用**自動電力模式 + 原廠 70W AC**，不是高效能模式；記錄 power state；
若執行中電池持續明顯下降，該 run 無效並停止。

### 規則

- **powermode = 0（自動）**：所有正式 Mac run，包括 Core A 與 quality。
  §4「所有正式 run 的 powermode 都必須相同」仍然成立，只是值從 2 改為 0。
- **AC adapter 必須是 Apple 原廠 70 W**：`pmset -g ac` 的 Manufacturer = `Apple Inc.`、Adapter Name = `70W USB-C Power Adapter`。
  - 2026-10-09 實際協商功率為 68 W，只記錄，不作條件。
  - 適用所有正式 Mac run。正式 run 開始前的 preflight 若不符，就拒絕開跑。
  - 每個 process 前後的 env 也要檢查，不符則該 session 為 env-invalid。
  - 讀不到 adapter 時同樣視為無效（013）。
- **記錄 power state**：每個 process 前後的 `env_capture` 記錄以下各項：
  - 電源來源與電量；
  - 電池狀態字串（`charging` / `discharging` / `charged` / `AC attached; not charging` 等）；
  - powermode；
  - adapter 的名稱、製造商、瓦數（不記 serial）；
  - thermal / performance warning。
- **電池下降停止規則**（只適用正式 Mac Core A，也就是 performance benchmark）：
  - 讀數：每個 process 前後 `env_capture` 記錄的電量（整數 %）。這些就是 012 有效性規則使用的同一組紀錄。
  - 時間軸：session 開始後經過的秒數，扣除 016 充電暫停的秒數。
  - **觸發條件**：任一讀數低於「它之前 30 分鐘內（含）所有讀數的最大值」**3 個百分點以上**。
  - 觸發後：
    - runner 立刻停止，不再開始任何 process，也不跑結尾的 sentinel。
    - 觸發點若是 process 開始前的讀數，該 process 不執行；若是結束後的讀數，該 process 的 row 照實寫入。
    - `session.json` 記錄 `battery_drain`：job、前後、降幅、視窗內的最大值與目前值。
    - env_flags 加一筆，所以 `env_valid = false`。
    - 整個 session 無效，與 012 的 env-invalid 相同：之後以同一個 seed、新的 session id 重跑。
      `core_a_report` 與圖表程式拒絕這種 session。
  - 若重跑在同樣條件下再次觸發，記為 blocker 回報使用者。我不會自行改電源條件或門檻。
  - **數字的理由**（在任何正式數據之前寫定）：
    - 3 個百分點：讀數只有 1% 的解析度，因此 3 點代表 30 分鐘內實際下降 ≥ 2%（≥ 4%/h）。
      單一的量化跳動不可能觸發，這是「明顯」的部分。
    - 30 分鐘視窗：必須是持續一段時間的淨下降才會觸發，這是「持續」的部分。
    - 更慢的下降交給 016 的充電閘門處理：電量 < 40% 時在 process 之間閒置充電到 90%。使用者指示 016 的門檻不變。
  - **參考**（pilot 的 env 紀錄；電量不是任何 outcome metric）：
    - powermode 2 時：4B pilot 在 30 分鐘內降 4 點，9B pilot 降 23 點，兩者都會觸發。
    - quality pilot 在第 36 分鐘從 powermode 2 切到 0 之後，電量從 56% 回升到 58%。
- **quality 不套用電池下降停止規則**：
  - quality 的 metric（KLD、ΔNLL、accuracy）是確定性的，012 已驗證重跑完全相同，不受電源狀態影響。
  - 使用者的停止規則針對的是 performance benchmark。
  - quality 仍適用 012 的有效性規則（AC、電量 ≥ 20%、沒有 warning）、016 的充電閘門，以及上面的 powermode 與 adapter 條件。

### 影響

- Mac 的 Core A 結果代表「自動電力模式、70 W adapter」這個平台條件，報告中要寫明。
- 012 的 pilot（4B / 9B Core A）是在 powermode 2 下跑的：
  - 時間校準（17.39 小時）與 noise（CV 1.0–5.9%）只作參考，正式 run 的時間與 noise 可能不同。
  - 這些值沒有被任何判定規則使用。
- §4 的「powermode 2」改為 0。這是凍結後的變更，§4 只加註記，指向本文。

## 2. Core B：`--max-num-seqs 128`，grid 上限 128（NEXT Blocked 2 的選項 A）

- 007：
  - `--max-num-seqs 264` 改為 **128**。
  - grid 規則「C_hi = 大於等於 max(8, 8 × C_KV) 的最小 2 的冪次，上限 256」的上限改為 **128**。
- 其餘不變：
  - C_KV 公式、grid 的構造（2 的冪次 + round(C_KV × 2^(k/2))）；
  - num-prompts、warmup、3 次 restart、C 的 seeded 順序；
  - `mixed` 的定義、H-B1 至 H-B4 的預測與推翻條件。
- 原因（Stage B 實測，`results/stage_b/20261009/boots`）：
  - vLLM 0.30.0 在 FULL CUDA graph 加上 Mamba 層時，要求 `max_num_seqs ≤ num_gpu_blocks`。
  - U = 0.91 時 num_gpu_blocks = 134。每條 decode 序列需要一個 GDN 狀態 block（約 50 MiB），264 條在 16 GB 卡上不可能達成。
- 007 原本的理由是「max-num-seqs 大於 grid 上限，knee 才不會是 scheduler 上限造成的」，現在變成兩者相等（128 = 128）：
  - C = 128 時，所有 request 仍可同時被排程；scheduler 上限不會低於任何被測的 C。
  - 由失敗那次 boot 估計的 C_KV（2K 約 19、8K 約 7、32K 約 1）都遠低於 128，knee 由 KV 容量決定，不是 scheduler 上限。
  - H-B1 要求 grid 涵蓋到 ≥ 4 × C_knee；C_knee ≤ 32 時仍然成立。
- **`mixed` dataset 不重建**：
  - 它在 2026-10-08 依原上限建立（`MIXED_ROWS` = 10 × 256 = 2560 列），sha256 已 pin。
  - plan 依檔案順序讀前 N = max(10C, 200) ≤ 1280 列；dataset 以十列為一組（9 短 + 1 長），所以前 N 列恰好有 N ÷ 10 個長 prompt。
  - `MIXED_ROWS` 因此固定為 2560，不再隨 grid 上限變動。
- **事先記下的風險**：
  - 017 規定 K 要向下取整到 256 MiB 的倍數。
  - max-num-seqs 264 時可用 KV 記憶體為 2.17 GiB（134 blocks）。若 128 時熱 boot 的值仍 < 2.25 GiB，K 會是 2.0 GiB，只剩約 123 blocks（< 128），
    vLLM 會拒絕以 K 啟動。
  - 改為 128 後，CUDA graph 的 capture size 會減少，預期可用記憶體會增加一些，但要實測才知道。
  - 若發生，記為 blocker 回報使用者，不自行調整 K、U 或 max-num-seqs。

## 3. Stage B 不拆分

- 不拆 Stage B。依 006，以 128 的新限制完成 Core B 的 Stage B 量測後，整段 §6 一次凍結（`freeze.nvidia = true`）。
- Stage B 量測依序為：boots（JIT、K）、pin-memory A/B、num_gpu_blocks / block_size、C_KV / grid。

## 4. quality：016 的充電門檻不變，先清理可重建的檔案

- 016 的充電門檻不改，也不等 ≥ 96 W 的 adapter。
- 只清理**可重建**的 scratch 與中間檔：manifest、結果、hash、recipe 一律不刪。
- 清理後若仍不足（KLD logits 約 70 GiB，加上 `disk_guard` 保留的 50 GiB），quality 維持 blocked，先做其他工作。

## 放棄的替代方案

- **維持 powermode 2**：使用者決定不用高效能模式。pilot 也顯示 powermode 2 下 70 W adapter 供電不足，電量持續下降。
- **電池下降時改為暫停充電而不是作廢**：使用者明確指示「該 run 無效並停止」。
- **把停止規則定成任何一次下降**：讀數只有 1% 解析度，會被量化跳動誤觸，不符合「持續明顯」。
- **Core B 的 PIECEWISE-only 或 eager（Blocked 2 的 B / C）**：偏離 vLLM 預設，使用者選 A。
- **依新上限重建 `mixed` dataset**：多一次重建與重新 pin，但結果完全相同（前 N 列不變）。

## 對實驗的影響

- PREREGISTRATION §4（Mac 電源）、§2（Core B），以及 007 加上凍結後註記，指向本文；原文不改寫。
- harness：
  - `run_core_a`：`POWERMODE_REQUIRED = 0`、adapter 檢查、電池下降停止規則（`BatteryDrainWatch`）。
  - `env_capture`：記錄 adapter 與電池狀態字串。
  - `core_a_report`：拒絕觸發停止規則的 session。
  - `core_b_plan`：`MAX_NUM_SEQS = 128`、`GRID_CAP = 128`，`MIXED_ROWS` 固定為 2560。
  - 以上都有測試。
