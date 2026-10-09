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
- **電池下降停止規則**（正式 Mac Core A；quality 是否套用，見本節最後，待使用者確認）：
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
    - 30 分鐘視窗：把 3 點變成一個最低速率。跌幅必須在 ≤ 30 分鐘內累積到 3 點才會觸發；更慢的下降不觸發，交給 016 處理。
      注意：視窗是時間跨度的**上限**，不是最短持續時間。規則沒有最短時間或後續確認條件，
      所以兩筆相隔幾分鐘的讀數之間跌 ≥ 3 點（包括電量計一次跳 3 點）也會立即觸發。
      「持續」只體現在：3 點一定代表 ≥ 2% 的實際電量流失，不可能來自單一的量化跳動。〔2026-10-09 審查後更正措辭；規則本身不變〕
    - 讀數間隔：正式 Core A 的讀數只來自每個 process 前後。pilot 中最長的 Core A process 約 13 分鐘（9B Q8_0 @ 32K 的 ttft，768 s），
      小於 30 分鐘，所以每個 process 內的下降都會被視窗涵蓋。
    - 更慢的下降交給 016 的充電閘門處理：電量 < 40% 時在 process 之間閒置充電到 90%。使用者指示 016 的門檻不變。
  - **參考**（pilot 的 env 紀錄；電量不是任何 outcome metric）：
    - powermode 2 時：4B pilot 在 30 分鐘內降 4 點，9B pilot 降 23 點，兩者都會觸發。
    - quality pilot 在第 36 分鐘從 powermode 2 切到 0 之後，電量從 56% 回升到 58%。
- **quality 是否套用電池下降停止規則：待使用者確認**（〔2026-10-09 審查後更正：原本寫成使用者決定，實際上是我的解讀〕）
  〔2026-10-09：正式 quality 改在 RTX 執行（020）。桌機沒有電池，這個問題不再存在。〕
  - 使用者的原話「Mac 正式 benchmark……若執行中電池持續明顯下降，該 run 無效並停止」沒有區分 Core A 與 quality。
    NEXT 原本的 Blocked 0 也把「Mac 正式 benchmark 的暫停」同時套用在正式 Core A 與正式 quality。
  - 我的提案是 quality 不套用，理由：
    - quality 的 metric（KLD、ΔNLL、accuracy）是確定性的，012 已驗證重跑完全相同，不受電源狀態影響；
    - 正式 quality 的單一 job 長 38–59 分鐘。只用 process 前後的讀數時，30 分鐘視窗涵蓋不到 job 內的下降；
      若要套用，必須在 job 進行中取樣電量。
  - **使用者確認之前，不跑正式 quality**。正式 quality 目前本來就因磁碟而 blocked（NEXT），所以不會造成額外延誤。
  - 不論結論為何，quality 都適用 012 的有效性規則（AC、電量 ≥ 20%、沒有 warning）、016 的充電閘門，
    以及上面的 powermode 與 adapter 條件（`run_quality` 的 preflight 與每個 process）。

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
  - U = 0.91 時 num_gpu_blocks = 134（2.17 GiB；這是冷 boot 的值，見下方補充）。
    vLLM 要求每條 decode 序列在每個 GDN group 各占一個 Mamba block，所以 max_num_seqs ≤ num_gpu_blocks。
    1 個 block 約 16.5 MiB；每條序列共 3 個 GDN block，約 49.5 MiB（PREREGISTRATION §2、008）。
    264 條至少需要 264 個 block（約 4.25 GiB），在 16 GB 卡上不可能達成。〔審查後更正：原本誤寫為每個 block 約 50 MiB〕
- 007 原本的理由是「max-num-seqs 大於 grid 上限，knee 才不會是 scheduler 上限造成的」，現在變成兩者相等（128 = 128）：
  - C = 128 時，所有 request 仍可同時被排程；scheduler 上限不會低於任何被測的 C。
  - 由失敗那次 boot 估計的 C_KV（2K 約 19、8K 約 7、32K 約 2，即 ⌊133 ÷ 66⌋；vLLM log 為 2.03x）都低於 128，
    knee 由 KV 容量決定，不是 scheduler 上限。〔審查後更正：32K 原誤寫為約 1〕
  - **事先記下的後果**（任何 Core B 數據之前；審查後補）：
    - block_size = 528 時 C_KV(2048) = ⌊(nb − 1) ÷ 7⌋。server 能以 128 啟動就表示 nb ≥ 128，所以 C_KV(2048) ≥ 18，
      8 × C_KV ≥ 144 > 128，L = 2048 的 grid 上限一定是 128。
    - 因此 007「8 × C_KV 等於 H-B2 區間上界的 4 倍」所保證的覆蓋，在 L = 2048 不再成立：
      knee 落在 2 × C_KV 附近或 64 時，4 × C_knee > 128，H-B1 在 L = 2048 只能是 supported 或 inconclusive，不可能 refuted。
      在 007 的上限 256 下，這些點原本可以檢驗。
    - H-B3 在 L = 2048 的 C_max 也由 256 變為 128。
    - L = 8192 與 32768 不受影響（8 × C_KV ≤ 128）。判定規則本身不變。
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
- **把停止規則定成任何一次 1 點的下降**：讀數只有 1% 解析度，會被量化跳動誤觸，不符合「持續明顯」。
- **Core B 的 PIECEWISE-only 或 eager（Blocked 2 的 B / C）**：偏離 vLLM 預設，使用者選 A。
- **依新上限重建 `mixed` dataset**：多一次重建與重新 pin，但結果完全相同（前 N 列不變）。

## 對實驗的影響

- PREREGISTRATION §4（Mac 電源）、§2 / §4（Core B），以及 003、007、018 加上凍結後註記，指向本文；原文不改寫。
- harness：
  - `run_core_a`：`POWERMODE_REQUIRED = 0`、adapter 檢查、電池下降停止規則（`BatteryDrainWatch`）。
  - `env_capture`：記錄 adapter 與電池狀態字串。
  - `core_a_report`：拒絕觸發停止規則的 session。
  - `core_b_plan`：`MAX_NUM_SEQS = 128`、`GRID_CAP = 128`，`MIXED_ROWS` 固定為 2560。
  - 以上都有測試。

## 審查（2026-10-09，review workflow：四個面向，每項發現另由對抗式驗證者複查；確認 10 項，駁回 10 項）

確認的項目都已處理，沒有改變任何規則：

- `parse_pmset_ac` 遇到空欄位時會吃到下一行，可能記到 serial。已改為只在同一行內比對，並加上測試。
- 003 的 Core B 參數表仍寫 264，而 §4 引用的正是這張表。已加註。
- 本文兩處數字錯誤（每個 GDN block 的大小、C_KV(32K)）已更正。
- 補記 cap 128 對 L = 2048 時 H-B1 可檢驗性的影響（見第 2 節）。
- 停止規則「30 分鐘視窗」的措辭會讓人誤以為規則有最短持續時間。已更正措辭；規則與程式不變。
- 「quality 不套用停止規則」原本記成使用者決定，實際上是我的解讀。已改為待使用者確認（見第 1 節）。
- 審查時 AC 的能源模式是 powermode 2：09:30 讀到 0，09:56 讀到 2。正式 Mac preflight 會因此拒絕開跑。
  系統設定由使用者處理，記在 NEXT 的 Blocked。
- `core_b_report` 原本不檢查 env_valid。這是凍結前就有的漏洞；core_a_report 與 quality_analysis 都會拒絕。
  已補上檢查與測試，記在 018 補充。

## 使用者補充（2026-10-09 午間）：高效能模式也可以

- 使用者：「允許電源高效能時也可以做」。
- 規則：
  - 正式 Mac run 的 powermode 可以是 **0（自動）或 2（高效能）**。
  - §4「所有正式 run 的 powermode 都必須相同」仍然成立：session 內每個 process 前後的 powermode 必須等於 preflight 的值，
    否則該 session 為 env-invalid。
  - `session.json` 記錄 `powermode`，報告中要註明。
- 其餘不變：原廠 70 W adapter、power state 紀錄、電池下降停止規則。
- **事先記下的預期**（pilot 的 env 紀錄）：powermode 2 加上 70 W 時，電量 30 分鐘內下降 4–23 點，很可能觸發停止規則。
  若觸發，照規則停止並作廢，記為 blocker 回報使用者，不自行放寬。
- harness：`POWERMODES_ALLOWED = (0, 2)`，並在正式 Mac session 中檢查每個 process 的 powermode 是否與 preflight 一致（有測試）。

