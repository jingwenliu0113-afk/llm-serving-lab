# 046 — Stage E1-P 的結果（formal-20261025-e1p）

- 狀態：**結果（描述，不是檢定）**。依 041 v3 §1.6–1.7 凍結的規則執行（`freeze.e1p = true`，凍結 commit `1de7895`，2026-10-25），凍結之後沒有改過任何規則、門檻或讀法。依 041 §1.1，結果只寫進本文與 Future Work，不改任何判定。
- 日期：2026-10-25
- 對應：041（§0、§1.6、§1.7、§1.8、§5 第 3 項、修訂紀錄「E1-P 凍結」）、045（E1-D，同一套描述量）、028（停止規則）、031（閒置與 spill）、024（觀測版 B0）
- 輸出（commit `178ba8b`）：`results/stage_e/e1p-analysis-20261025/e1p.json`，由 `benchmarks/stage_e1p_analysis.py` 在 Mac 上產生；`descriptive_only` 為 true，`decision` 為 "041 v3 §1.6"。session 的檔案在 `results/stage_e/formal-20261025-e1p/`（同一個 commit）。
- **讀法：無法判定（觸發太少）**（`reading.row` = few_spikes）。seed B 在 12 個 bench 中沒有任何尖峰。
- 本文的 JSON 路徑：沒有前綴的是 e1p.json；`session.`、`plan.`、`rows[i].` 是 session 目錄中的檔；`e1[tag].` 是 `derived/<tag>-e1.json`。

## 執行

1. **seed 檢查**（041 §1.7，commit `637d70f`，`seed_check.json` 的 `created_at` 為 2026-10-24 23:44:29Z）：
   - N 200 與 N 16 兩組，seed A（20261008）與 seed B（20261023）的每個 prompt 在 server 端都是 8,064 token（`checks[].seeds.X.server_tokens_ok` = n：200 / 200、16 / 16；`server_tokens_other` 都是空的）。
   - 兩個 seed 的 prompt 不同（`prompts_sha256`：N 200 為 `4b10c963…` 與 `c0683396…`）。
   - 正式 plan 記下的 `plan.seed_check_sha256`（`60cfd8e4…`）等於 commit 的檔。
2. **smoke 第一次嘗試 `smoke-e1p-20261025`**（2026-10-24 23:45:26–23:46:58Z，部署 `b517cdc`）：`status` 為 refused。
   - 第一次 boot 之前，031 的閒置判定在 3 個 30 秒的窗都不通過：閒置佔用 1,179 MiB，大於上限 1,018 MiB（`stopped.rule` = gpu_not_idle）。
   - 沒有啟動 server，沒有任何 bench（`outcomes` 全部為 0）；依 041 §1.7 記為 refused，不算嘗試。
   - 原因（041 §1.7）：Windows 的桌面程式佔用 GPU 記憶體。使用者關閉程式之後才做第二次嘗試。
3. **smoke 第二次嘗試 `smoke-e1p-20261025b`**（2026-10-25 10:15:15–10:18:20Z，部署 `ff66e6d`）：通過。
   - 閒置佔用 849 MiB；1 次 boot，2 個 bench（seed A、B 各一，N 16）都有效。
   - `smoke.json` 的 13 項檢查都通過。env 有效，`elapsed_s` 185。
   - `harness_sha256` 為 `414deba6…`。延遲已遮蔽（`masked` 列出 18 個檔），沒有看。
4. **凍結**：commit `1de7895`（2026-10-25 10:27:18Z）。`configs/stage_e1p.toml` 的 `freeze.e1p = true`、`freeze.smoke_e1p = "smoke-e1p-20261025b"`。041 §1.6 的設計、門檻與讀法都沒有改。
5. **部署**：正式 session 的 13 份 env 擷取（preflight 與 6 次 boot 各自的 before / after）都記錄 `DEPLOYED_COMMIT` 為 `1de7895`，dirty false。
   - 每份擷取的 power limit 都是 300 W。
   - `session.versions` 的 driver 617.42、vLLM 0.30.0、torch 2.13.0+cu130，都和 Phase 1 相同。
6. **正式 session `formal-20261025-e1p`**：
   - 時間：10:27:46Z 開始（`session.started_at`），12:55:11Z 結束（`finished_at`），`elapsed_s` 8,845（2.46 h），在 smoke 結束後約 9 分鐘開始。
   - 上限：`cap_s` 10,615 = 3.0 h − 185 s。smoke 加正式共 9,030 s，在 3.0 h（10,800 s）之內。
   - boot 規則（已用時間加 30 分鐘超過上限時，不開始新的 boot）沒有作用：第 6 次 boot 開始時已用 7,405 s，加 1,800 s 為 9,205 s。
   - 結果：`status` finished，`stopped` null，`env_valid` true，`env_flags` []。
   - `outcomes`：boots 6、benches 24、valid 24、invalid 0、not_run 0、failed_boots 0。

| boot | 順序（位置 1–4） | 時間（UTC） | 開始時已用 s | 啟動 s | 031 閒置 MiB | 最高溫 °C | 有效 bench |
|---|---|---|---|---|---|---|---|
| 1 | A B B A | 10:28:25–10:52:31 | 38.1 | 40 | 522 | 71 | 4 |
| 2 | B A A B | 10:53:05–11:17:04 | 1,518.4 | 32 | 522 | 71 | 4 |
| 3 | A B B A | 11:17:38–11:41:37 | 2,991.4 | 32 | 522 | 72 | 4 |
| 4 | B A A B | 11:42:11–12:06:10 | 4,464.3 | 32 | 522 | 72 | 4 |
| 5 | A B B A | 12:06:44–12:30:38 | 5,937.4 | 30 | 522 | 72 | 4 |
| 6 | B A A B | 12:31:12–12:55:11 | 7,405.2 | 32 | 522 | 72 | 4 |

- 欄位來源：`session.boots.bN.{t_start, t_end, elapsed_at_start_s, startup_s, idle.idle_mib, smi_summary.temp_max_c, n_valid}`；順序見 `plan.boots`。
- **閒置**：6 次 boot 的 031 閒置判定都在第 1 次通過，佔用都是 522 MiB（最小等於最大，util 0 %），上限 1,018 MiB。三個 session（被拒絕的 smoke、通過的 smoke、正式 session）的閒置佔用依序是 1,179、849、522 MiB。
- **spill**：Windows 端的 shared GPU 記憶體在每次 boot 就緒時比啟動前多 80 MiB（容許 160），每個 bench 之後比就緒時多 2 MiB（容許 64），都沒有 spill（`session.boots.bN.spill.readings`）。
- **溫度**：最高 72 °C，遠低於 88 °C 的停止門檻。
- **每個 bench**（`rows[i]`，24 列）：
  - 都是 completed 200、failed 0，沒有逾時；
  - `input_lens_8064` 都是 true，每個主 id 都有 span（`spans_main` 200），F 都算得出來，`e1_ok` 都是 true；
  - 時間窗內的 step 有 3,533–3,536 個，`num_gpu_blocks` 都是 248，沒有 spill，bench 後都健康。
  - OTLP receiver 每次 boot 收到 832 個 llm_request span（4 × 208，含 warmup），沒有 decode 錯誤（`session.boots.bN.receiver`）。
- **取回**：`results/stage_e/wsl-inventory.txt`（12:55:45Z）列出 WSL 上的 3 個目錄：本 session 與兩次 smoke。所以這是第一個也是唯一的正式嘗試，分析的 `first_valid_session` 選的就是它。raw/ 不進 git，`raw_manifest.json` 記錄 1,704 個 raw 檔的 sha256。
- **一處記錄上的差異**：commit `178ba8b` 的訊息寫「10:37-12:55Z」。但 `session.started_at` 是 10:27:46Z，preflight 的 env 擷取是 10:27:51Z（`env/preflight.json` 的 `captured_at_utc`），第 1 次 boot 的 before 擷取是 10:28:25Z，第一個 bench 在 10:29:07Z 開始（`rows[0].t_before`）。本文以 session.json 為準；數據與讀法都不受影響。

## 結果

### 每個 bench 的尖峰 idx（`seeds.X.per_bench[i].spike_idx`）

尖峰 idx 為 F(8k) > 35 ms 的 8k（k 1–24）。格中寫 seed 與尖峰 idx，「—」表示沒有尖峰。

| boot | 位置 1 | 位置 2 | 位置 3 | 位置 4 |
|---|---|---|---|---|
| 1 | A — | B — | B — | A — |
| 2 | B — | A — | A 128 | B — |
| 3 | A 16、88 | B — | B — | A — |
| 4 | B — | A — | A — | B — |
| 5 | A 88 | B — | B — | A — |
| 6 | B — | A 88 | A 88 | B — |

- **seed A**：6 個尖峰（`seeds.A.spikes`）。計次（`seeds.A.spike_idx_counts`）為 16 ×1、88 ×4、128 ×1。
- **seed B**：0 個尖峰（`seeds.B.spikes`），12 個 bench 的 288 組（12 × 24）都沒有。
- **重複位置**（至少 3 個 bench；`repeat_min` 3）：A_in = {88}（4 個 bench：b3-n1、b5-n1、b6-n2、b6-n3），B_in = {}。
- **A_in ∩ A_prior** = {88}（`reading.A_in_and_prior`；A_prior = {32, 64, 88, 128, 176, 192}）。
- **J** = |A∩B| / |A∪B| = 0 / 1 = 0.0（`reading.J`）。
- 第一次看到的 idx（只列出）：seed A 的 idx 16（滑移起始 17）不在 041 §1.2 的 21 段起始 idx 中，也不在 045 列出的 Advanced 15 個 run 的尖峰 idx 中。

### 讀法表（041 §1.6，依序取第一個符合的列）

1. **第 1 列：任一 seed 的尖峰總數 < 5，或 A_in、B_in 有一個是空集合 → 符合。** seed B 的尖峰總數 0 < 5，B_in 也是空集合。seed A 的 6 個不小於 5。讀法為「**無法判定（觸發太少）**」（`reading.text`）。
2. 依「取第一個符合的列」，第 2–5 列不再看。為了完整列出：
   - A_in ∩ A_prior 不是空集合，所以第 2 列本來也不會符合；
   - J = 0.0 是 B_in 為空集合時機械地算出的值（|A∪B| = 1），不代表第 3 列的「和 prompt 內容鎖定」。第 3 列只在兩個 seed 都有重複位置時才有意義。
3. **「是數據不足，不是觸發太少」的註記不適用**（041 §1.7）：兩個 seed 都有 12 個有效 bench（`data.valid`），不少於 `repeat_min` 3，所以 `data.insufficient_data` 為 false，`reading` 沒有 `note`。
   - 計畫的 bench 都跑完而且有效（`data.planned` = `data.valid` = A 12、B 12），沒有被排除的（`data.excluded` 為空），沒有缺 E1-D 描述量的（`data.without_e1` 為空），`data.incomplete_session` false。
   - 所以第 1 列在這裡的意思是 seed B 真的沒有觸發，不是數據不足。

### 另外描述的項目（041 §1.6）

1. **F 尖峰與滑移的一致性**（`seeds.X.spike_slip`）：
   - seed A：6 個尖峰都是滑移組，6 個滑移組都有尖峰；g 1–24 的 288 組中，其他 282 組兩者都沒有。
   - seed B：0 個尖峰、0 段滑移。
   - 重算：每個 bench 的滑移起始 idx 都等於尖峰 idx 加 1（`seeds.A.per_bench[i].episodes` 對 `spike_idx`）。
2. **滑移率**（段 / bench）：seed A 0.5（6 / 12，`seeds.A.slip_rate`）、seed B 0.0（0 / 12）、合計 0.25（6 / 24，`slip_rate_all`）。
   - 既有值（`earlier_slip_rates`，041 §1.6）：C0 0.21、Advanced 0.89、Core 2.67。
   - 這些值來自其他 session 與其他 server 設定，只並列，不比較（041 §0、§1.6）。
3. **第 1 個 bench 與其他位置**（`seeds.X.slips_by_position`，每個位置每個 seed 3 個 bench）：

| seed | 位置 1 | 位置 2 | 位置 3 | 位置 4 |
|---|---|---|---|---|
| A | 3 段 / 3 bench | 1 / 3 | 2 / 3 | 0 / 3 |
| B | 0 / 3 | 0 / 3 | 0 / 3 | 0 / 3 |

   - 兩個 seed 合計（重算）：位置 1 為 3 段 / 6 個 bench，位置 2–4 為 3 段 / 18 個 bench。
   - seed A 位置 1 的 3 段：b3-n1 2 段、b5-n1 1 段、b1-n1 0 段。

### seed A 的 6 段（E1-D 的量，`e1[tag].episodes[j]`）

| bench | k | 起始 idx | F(8k) ms | n_dec / 中位數 | L3 dev ms | gap ms | ΔQ ms | 送出延遲 ms | L1–L5 |
|---|---|---|---|---|---|---|---|---|---|
| b2-n3 | 16 | 129 | 42.35 | 4 / 3 | −338.5 | 89.70 | 348.56 | 0.117–0.199 | 全部成立 |
| b3-n1 | 2 | 17 | 42.86 | 4 / 3 | −331.2 | 90.91 | 347.76 | 0.052–0.196 | 全部成立 |
| b3-n1 | 11 | 89 | 41.78 | 4 / 3 | −336.3 | 93.84 | 348.43 | 0.056–0.300 | 全部成立 |
| b5-n1 | 11 | 89 | 41.41 | 4 / 3 | −337.3 | 91.43 | 348.92 | 0.117–0.193 | 全部成立 |
| b6-n2 | 11 | 89 | 40.80 | 4 / 3 | −337.2 | 91.12 | 348.99 | 0.062–0.278 | 全部成立 |
| b6-n3 | 11 | 89 | 41.47 | 4 / 3 | −334.9 | 91.07 | 349.02 | 0.133–0.241 | 全部成立 |

- 欄位來源：`values.{F0_ms, n_dec, n_dec_median, L3_dev_ms, gap_ms, dQ_ms, send_lag_ms}` 與 `links`。gap 是 server 的到達時間，送出延遲是 client 的時鐘（7 個值的最小–最大）。
- 6 段都是組內 `position` 1（從 8k+1 開始）、`length` 7、非連續、k 在 1–23 之內；6 段分布在 bench 位置 1、2、3（見上面的位置表）。
- **鏈**（`seeds.X.chain`）：
  - seed A：6 段都可評估，0 段有不成立的環；非滑移組（g 1–23）270 組中 L1 成立 0 組 → **holds**。
  - seed B：沒有段，`chain` 為 "not evaluable"；非滑移組 276 組中 L1 成立 0 組。
  - 起始 idx（`seeds.A.chain.start_idx`）：17 ×1、89 ×4、129 ×1。
- **F 的空隙**（從 derived 重算）：24 個 bench 的 576 組（k 1–24）中，非尖峰組的 F 最大 29.72 ms（b1-n1 的 idx 16），尖峰最小 40.80 ms。35 ms 的門檻落在空隙中間，沒有接近門檻的組。
- **錨定計數**（`e1[tag].f_spikes.n_dec_anchor`，窗為 (arrival, s + 5 ms]）：
  - 24 個 bench 的中位數都是 2，沒有 ambiguous；
  - 6 個尖峰的計數都是 3（中位數加 1），3 個都是 decode-only；margin 最小 2.75 ms（b3-n1 的 g 2）；
  - 最近時間戳減 s 的範圍（`nearest_stamp_minus_s_ms_range_g1_23`）：24 個 bench 合起來為 −2.24 至 +3.05 ms（下限在 b4-n3），仍在窗的終點之內。每個 bench 的上限依 seed 分開：seed A 的 12 個 bench 為 1.92–3.05 ms，seed B 的 12 個 bench 為 0.21–1.04 ms（見「只列出、不解讀的值」第 3 項）。
  - L2 的基準：24 個 bench 的 `n_dec_median` 都是 3。

### 只列出、不解讀的值

1. **seed B 的錨定計數 1**：seed B 有 6 個 bench（b1-n2、b3-n2、b3-n3、b5-n2、b6-n1、b6-n4）在 g 1–23 中各有 1 組的計數是 1（`n_counts_g1_23` = {"1": 1, "2": 22}）。
   - seed A 的 12 個 bench 只有 2 與 3。045 的 Advanced 15 個 run 也只有 2 與 3。
   - e1 檔沒有記錄是哪一組，要找出是哪一組需要 WSL 上的 raw（steps 與 spans）。
   - 041 沒有為這個量定義讀法，本文不解讀。
2. **錨定時間戳的下限 −2.24 ms**（b4-n3，seed A）：
   - 低於 041 §1.4 依 B0、G、K3 寫下的 −0.9 ms，也低於 045 的 Advanced 15 個 run 的每個下限（T、TG 為 −1.21 至 +3.43 ms，最低是 TG-r1 的 −1.21 ms）。
   - 其他 23 個 bench 的下限在 −0.97 至 −0.63 ms 之間。
   - 24 個 bench 的 `ambiguous_groups_g1_23` 都是 0。
   - 041 沒有為這個量定義讀法，本文不解讀。
3. **最近時間戳減 s 的上限，兩個 seed 分開**（每個 bench 的 `nearest_stamp_minus_s_ms_range_g1_23` 上限）：
   - seed A 的 12 個 bench 為 1.92–3.05 ms，seed B 的 12 個 bench 為 0.21–1.04 ms，兩組不重疊。
   - 045 的 Advanced 15 個 run 的上限為 1.77–3.43 ms。
   - 041 沒有為這個量定義讀法，本文不解讀。
4. **seed A 的 idx 16**：見上面「第一次看到的 idx」。

## 對 E1 研究問題的意義（只寫 041 允許的部分）

1. **E1-P 的問題（041 §1.6：F 的尖峰由 prompt 內容決定，還是由 bench 中的位置決定）沒有得到回答。** 讀法表的第 1 列符合，041 §1.8「能交付」中的「觸發和內容還是位置綁定（E1-P）」沒有交付，只寫進 Future Work。
2. **不重跑、不延長**：041 只允許環境無效的 session 以新 id 重跑一次（§0）；這個 session 有效，是唯一要分析的正式 session。041 §1.6 也規定「不依數據延長或提前停止」。任何後續的探針都要另寫 decision log；本文不提出。
3. **可以描述的事實**（同一 session，041 §0）：
   - seed A（20261008）就是 Advanced、C0 與 Core B 用的 seed（041 §1.2「三組數據都用 seed 20261008」；`configs/stage_e1p.toml` 的註解寫 A = Core B's seed），A_prior 就是從這組 prompt 的 Advanced 15 個 run 得來。
   - seed A 在 session 內重現 A_prior 中的位置 88（4 / 12 個 bench）。
   - seed B 在 12 個 bench 中沒有任何尖峰或滑移。
   - seed A 的 6 段，五環在觀測版 B0 上全部成立，和 E1-D（045）的描述一致。
   - 兩個 seed 在同一組 boot 中交錯執行（ABBA / BAAB），每個位置各 3 個 bench。
4. **seed 之間的尖峰數**：尖峰數 6 對 0 落在讀法表第 1 列（任一 seed 的尖峰總數 < 5），讀法為無法判定；本文不對這個差異另作解讀。
5. E1 不改任何判定（§1.1）。H-A2、H-B4、Advanced 與 C0 的判定都不受影響。

## 限制

1. **只做描述**（041 §1.1、§1.6）：沒有檢定，也沒有信賴區間。041 自己寫明「樣本小，讀法粗略」；J 的門檻是在沒有對應數據時定的（§5 第 3 項）。
2. **樣本小**：每個 seed 12 個 bench，全部只有 6 個尖峰、6 段滑移。只有一個 session、一台機器（RTX 5070 Ti，WSL2）、一個模型、兩個 seed，只有 8K uniform、C 8。
3. **server 不同**：E1-P 用的是 024 的觀測版 B0（OTLP 與 iteration log 開啟），C0 的 flag 是關的。結果不和 C0 直接比較（§1.6）。既有的滑移率來自其他 session 與 server，只並列，不比較。
4. **F > 35 ms 是看過舊數據之後才定的**（§5 第 3 項）。在本 session，非尖峰組最大 29.72 ms、尖峰最小 40.80 ms，所以尖峰集合在 29.72–40.80 ms 之間的任何門檻下都相同。
5. **raw/ 不進 git**：steps 與 spans 只在 WSL 上，以 `raw_manifest.json` 的 sha256 稽核。錨定計數與 n_dec 來自 runner 用 E1-D 凍結的函式寫出的 e1 檔；分析核對了每個檔的 sha256，也核對 runner 的尖峰等於由 derived F 重算的尖峰。
6. **時鐘**：gap 用 server 的到達時間，送出延遲只用 client 的時鐘，兩者不互相相減（041 §1.3）。
7. **三個 session（被拒絕的 smoke、通過的 smoke、正式 session）的閒置佔用不同**（1,179、849、522 MiB）：第一次是 Windows 桌面程式（041 §1.7）；849 與 522 MiB 沒有記錄原因（三個 session 的 `idle.compute_apps` 都是空的）。正式 session 的 6 次 boot 都是 522 MiB。

## 能支持與不能支持

- **能支持**：
  1. 在 `formal-20261025-e1p` 中，seed A 的尖峰位置 88 在 4 / 12 個 bench 重現；seed B 在 12 個 bench 中沒有尖峰。
  2. seed A 的 6 個尖峰和 6 段滑移一一對應，6 段的五環都成立；兩個 seed 的非滑移組（g 1–23）都沒有 L1 尖峰（270 + 276 組）。
  3. session 有效：24 / 24 個 bench 有效，沒有停止，env 有效，沒有 spill，最高 72 °C。
- **不能支持**：
  1. 觸發和內容綁定還是和位置綁定（讀法為無法判定）。
  2. seed 之間尖峰數差異的原因。
  3. 位置 1 的滑移較多是否有意義（3 / 6 對 3 / 18，事件太少，041 沒有定義讀法）。
  4. F 多約 15 ms 的根本原因（需要 profiling，§1.8）。
  5. 和 C0 或其他 session 的比較；推廣到其他 seed、模型、長度、C 或 session。
  6. 改變任何判定（§1.1）。

## 數字來源

- **e1p.json 直接讀取**：
  - `reading.{row, text, J, A_in, B_in, A_in_and_prior}`；
  - `data.{stopped, outcomes, planned, valid, excluded, without_e1, insufficient_data, incomplete_session}`；
  - `seeds.X.{spikes, spike_idx_counts, repeated, per_bench[i].{tag, position, spike_idx, episodes}, episodes, slip_rate, slips_by_position, spike_slip, chain.*}`；
  - `slip_rate_all`、`earlier_slip_rates`、`harness_sha256`、`stage_e1_harness_sha256`、`stage_e1p_sha256`、`inputs_sha256`。
- **session 目錄直接讀取**：
  - `session.json`：`started_at`、`finished_at`、`elapsed_s`、`cap_s`、`status`、`stopped`、`env_valid`、`env_flags`、`outcomes`、`versions`、`boots.bN.*`（時間、閒置、spill、receiver、溫度）；
  - `plan.json`：`cap_s`、`smoke`、`seed_check_sha256`、`boots`；
  - `rows.jsonl`：每個 bench 的有效性欄位；
  - `env/*.json`：`captured_at_utc`、`repo.commit`、`repo.dirty`、`gpu.nvidia[0].enforced_power_limit_w`；
  - `derived/<tag>-e1.json`：每段的值、錨定計數（`f_spikes.n_dec_anchor.{n_counts_g1_23, ambiguous_groups_g1_23, nearest_stamp_minus_s_ms_range_g1_23}`）。
- **其他檔**：`results/stage_e/smoke-e1p-20261025{,b}/session.json`（含 `idle.compute_apps`）、`smoke.json`、`env/preflight.json`；`results/stage_e/e1p-seed-check/seed_check.json`；`results/stage_e/wsl-inventory.txt`；`configs/stage_e1p.toml` 的 `seeds` 與註解；045 的錨定表（Advanced 15 個 run 的最近時間戳 − s 範圍）。
- **重算**（在 Mac 上，用 `inputs_sha256` 中已核對的檔）：
  - 尖峰 idx：從 `derived/<tag>-requests.jsonl` 的 `F_ms` 取 idx 8k（k 1–24）中 > 35 ms 的，24 個 bench 都等於 `per_bench[i].spike_idx` 與 e1 檔的 `spike_groups`。
  - A_in、B_in、J：照 041 §1.6 的定義計算，等於 `reading`。
  - 尖峰與滑移的一致性：比對每個 bench 的尖峰 idx 與滑移起始 idx 減 1。
  - F 的空隙（29.72 / 40.80 ms，576 組）、非滑移組的 L1（Σ `e1[tag].l1_in_normal_groups`：A 270 / 0、B 276 / 0）、錨定計數的分布與範圍、位置 1 對其他位置的合計：對相應欄位計次或取 min / max。
  - 時數：8,845 + 185 = 9,030 s；10,800 − 185 = 10,615 s；7,405.2 + 1,800 = 9,205.2 s。

## 獨立檢查

- **起草時（起草的 session）**：在 Mac 上直接呼叫 `stage_e1p_analysis.analyze()`（CLI 只接受新的 `results/stage_e/<session>/`，所以輸出寫到 scratchpad），輸出和 commit 的 e1p.json 逐位元組相同（sha256 `8827f8ff…`）；`first_valid_session` 選出本 session（唯一的正式嘗試），`session_problems` 沒有問題。
- **獨立核對（另一個核對者，沒有參與起草）**：
  - 照 CLI 的 `main` 走同一條路徑（`first_valid_session`、`session_problems`、`analyze`，加上 harness 與設定檔的 sha256）重跑，輸出和 commit 的 e1p.json 逐位元組相同（`cmp` 通過，sha256 `8827f8ff…`）。formal attempts 只有本 session。
  - 輸入：`inputs_sha256` 的 51 個檔（session.json、plan.json、rows.jsonl 與 24 個 bench 各 2 個 derived 檔）sha256 都相符，`rows_sha256` 等於 session.json 記錄的值。
  - harness：`harness_sha256()` 為 `414deba6…`，等於凍結與 smoke 的值；`stage_e1.harness_sha256()` 為 `87b937d1…`，等於 E1-D 的凍結值。`178ba8b` 之後的 commit 只改了 044。
  - 從 rows.jsonl 與 derived 獨立重算：尖峰、A_in、B_in、J、讀法表、尖峰與滑移的對應、滑移率、各位置的滑移數、非滑移組的 L1、F 的空隙、6 段的 F、gap、ΔQ，都和草稿相同；用 `c0_analysis.classify_bench` 從 derived 的 client TTFT 重新分類滑移，24 個 bench 都等於 e1 檔的 episodes。L3、送出延遲、n_dec 與錨定計數從 e1 檔讀（sha256 已核對）。
  - 執行事實（時間、上限、outcomes、閒置、spill、溫度、receiver、13 份 env、兩次 smoke、seed 檢查、inventory、1,704 個 raw 檔）都和檔案相同，只有下面第 1 項例外。
- **核對後的更正（修訂時逐項對 commit 的檔與 041 再核對過；重跑的輸出也再以 `cmp` 確認逐位元組相同）**：
  1. 第一份 env 擷取是 preflight 的 10:27:51Z（`env/preflight.json`），不是 10:28:25Z（那是 `env/b1.before.json`）。
  2. 刪除「041 的讀法表沒有一列描述一個 seed 有觸發、另一個沒有」與其後的內容解讀：6 對 0 正落在讀法表第 1 列，讀法為無法判定。
  3. 849 與 522 MiB 的閒置佔用沒有記錄原因（三個 session 的 `idle.compute_apps` 都是空的），刪除「都是 Windows 端的狀態」。
  4. 「三次嘗試」改為「三個 session」：refused 不算嘗試（041 §1.7）。
  5. 錨定時間戳下限 −2.24 ms 補上和 Advanced 15 個 run（最低 −1.21 ms）與其他 23 個 bench（−0.97 至 −0.63 ms）的比較；「不影響任何計數」改為可核對的 `ambiguous_groups_g1_23` 為 0。
  6. 補上最近時間戳減 s 的上限依 seed 分開（A 1.92–3.05 ms、B 0.21–1.04 ms），列入「只列出、不解讀的值」。
  7. 「`position` 1」改為組內位置（8k+1），並寫出 6 段在 bench 位置 1、2、3；補上 seed A 就是 Advanced、C0、Core B 用的 seed 20261008，A_prior 來自這組 prompt。
- 核對腳本放在 session 的 scratchpad，沒有 commit。
