# 048 — Stage Q（Q1、Q2）的結果

- 狀態：**結果**。依 040 凍結的規則執行（`freeze.stage_q = true`，harness `34256f12…`，2026-10-25 凍結），凍結之後沒有改過任何規則、界線或判定方式。執行上有一處偏離 040 實作補記 14（q1-metal 第 2 次嘗試從 worktree 執行，見「偏離」），另有一次違反操作規則（q2 期間在 Mac 上以 ssh 唯讀列出 WSL 目錄，見「執行」）。Q3 另外報告。
- 日期：2026-10-31
- 對應：040（Q1、Q2、揭露、實作補記 1–14、pilot 與嘗試紀錄）、005（quality 指標）、020（quality 改在 RTX）、037 §4（Phase 3 的 Mac 用 140W adapter）、PREREGISTRATION §10。
- 輸出（commit `c95fa70`）：`results/stage_q/analysis-20261031.json`，由 `quality/stage_q_analysis.py` 在 Mac 上產生。分析的 session：`formal-20261031-q1-metal`（Mac Metal）、`formal-20261031-q1-cuda` 與 `formal-20261031-q2`（RTX CUDA），以及 Phase 1 的 `results/quality/formal-20261010-quality`（CUDA 的 PPL 與 `-c 512` 的 KLD）。
- **判定**（事先的預測五個都是 supported；結果 2 個 supported、3 個 not supported，H-Q1b、H-Q1c、H-Q2b 的預測不成立）：

| id | 內容 | 判定 |
|---|---|---|
| H-Q1a | Metal 上兩個 size 都是 KLD(Q8_0) < KLD(Q6_K) < KLD(Q4_K_M)（4 個 z 檢定，Holm） | **supported** |
| H-Q1b | 每個 quant 與 size：KLD_Metal ÷ KLD_CUDA（同樣的前 200 chunks）在 [0.8, 1.25] | **not supported**（6 格中 3 格超出） |
| H-Q1c | 8 個設定：PPL_Metal ÷ PPL_CUDA（580 chunks）在 [0.995, 1.005] | **not supported**（9B Q4_K_M 超出） |
| H-Q2a | 兩個 context、兩個 size：KLD 的順序（8 個 z 檢定，Holm） | **supported** |
| H-Q2b | 每個 quant、size 與 context：KLD(c) ÷ KLD(512) 在 [0.5, 2.0] | **not supported**（12 格中 4 格超出） |

- 沒有任何 NT：三個部分都有有效的 session，每個比值與檢定都有值。

## 執行

| 部分 | session | 時間（UTC） | 平台與部署 | job | env |
|---|---|---|---|---|---|
| q1-metal | `formal-20261031-q1-metal`（第 2 次、最後一次嘗試） | 2026-10-30 21:35–23:57 | Mac，Mac lock，worktree 固定在 `09e2776`；powermode 2，Apple 140W adapter（協商 94 W） | 20 / 20 ok | 有效，沒有 flag，沒有充電暫停 |
| q1-cuda | `formal-20261031-q1-cuda` | 2026-10-30 23:21–23:36 | RTX，GPU lock，部署 `cd615d0` | 12 / 12 ok | 有效 |
| q2 | `formal-20261031-q2` | 2026-10-30 23:36–10-31 00:58 | RTX，GPU lock，部署 `cd615d0` | 24 / 24 ok | 有效 |

- **q1-metal 的嘗試**：第 1 次 `formal-20261025-q1-metal` 作廢，原因是 env 無效（session 期間主 checkout 有 commit）；另外 10 個 KLD job 被磁碟檢查跳過、2 個 delete 記為 error，8 個 PPL job ok 但不進入分析（040）。使用者釋放磁碟空間後做第 2 次。分析的 `parts.q1-metal.void_before` 為 `[formal-20261025-q1-metal]`，符合「之前最多一次作廢嘗試的第一個有效 session」。q1-cuda 與 q2 都是第一次嘗試即有效；三個部分都沒有 superseded 紀錄。
- **chunk 數**：Q1 的 KLD 在兩個平台都是前 200 chunks（`--chunks 200` 的 base）；Metal 的 8 個完整 PPL 都是 580 chunks；Q2 為 c2048 145 chunks、c8192 36 chunks（計分 token 145 × 1,023 = 148,335、36 × 4,095 = 147,420，和 Phase 1 c512 的 580 × 255 = 147,900 相近）。
- **env**：q1-metal 的 37 份擷取都記錄 commit `09e2776`、dirty false；q1-cuda 與 q2 的 62 份都記錄 `DEPLOYED_COMMIT` `cd615d0`、dirty false；llama.cpp 都是 build 11146（`7fe450e`），等於 pins。WSL 的擷取都有 `probe_errors`（部署的副本沒有 `.git`、沒有 `nvcc`），和 pilot 相同，不影響 repo 欄位。
- **取回**：`results/stage_q/wsl-inventory.txt`（2026-10-31 00:58:15Z）列出 WSL 上的 7 個目錄，都已在本機。
- **揭露（040 的 q2 紀錄）**：q2 執行中（00:29:55Z，`results/stage_e/wsl-inventory.txt` 的列出時間），我在 Mac 上執行了一次 `tools/rtx/fetch_all_sessions.sh stage_e`（以 ssh 唯讀列出 WSL 的目錄），違反「GPU session 期間不讀寫 WSL」的操作規則。q2 只量 PPL 與 KLD，不量時間，env 擷取沒有 flag。

### 偏離：q1-metal 第 2 次嘗試從 worktree 執行

- 040 實作補記 14 寫「正式 session 在主 checkout 執行（……worktree 的 `results/stage_q/` 看不到其他嘗試，所以 Stage Q 不這樣做）」。第 1 次嘗試作廢後，我在 040 的紀錄中寫了「或改從獨立的 clone 執行」（clone 同樣看不到主 checkout 中未 commit 的嘗試，也偏離補記 14），實際用的是固定在 `09e2776` 的 git worktree；執行前沒有修訂補記 14，也沒有另寫 decision log。符合規則的做法是留在主 checkout 執行、期間不 commit。事後在 040 的嘗試紀錄補上偏離說明（與本文一起 commit）。
- session 目錄在 worktree 中產生，結束後搬回主 checkout 的 `results/stage_q/`，在 `b9088a3` commit，之後移除 worktree。
- 紀錄：37 份 env 擷取的 `load.mac_heavy_lock_owner` 都記錄 runner 的 cwd 為 `.worktrees/q1-metal-20261031`、指令為 `quality/stage_q.py run`，所以從 worktree 執行有直接紀錄；`repo` 欄位都記錄 commit `09e2776`、dirty false（session 期間主 checkout 的 HEAD 移到 `2c5fddf`、`cd615d0`、`38a002e`、`ef524f3`、`534be5b`，擷取不受影響）；runner 記錄的 harness 等於凍結的 `34256f12…`；rows 的 argv 都指向主 checkout 的模型、資料與 llama.cpp。
- 可以證明的：`09e2776` 的 tree 包含第 1 次嘗試的 `session.json`，preflight 的 `repo.status` 只有本 session 的目錄，所以 runner 看得到第 1 次嘗試。不能事後證明的：worktree 建立當下，主 checkout 是否有未 commit 的 q1-metal 嘗試目錄（目前的主 checkout 沒有）。
- 對數值與判定沒有影響。

## 結果

### Q1：Metal 與 CUDA（`-c 512`）

**KLD（前 200 chunks；H-Q1a 用 Metal 一欄，H-Q1b 用比值）**

| size | quant | KLD Metal（mean ± SE） | KLD CUDA（mean ± SE） | Metal ÷ CUDA | 在 [0.8, 1.25] 內 |
|---|---|---|---|---|---|
| 4B | Q8_0 | 0.000680 ± 0.000110 | 0.001925 ± 0.000366 | 0.3532 | 否 |
| 4B | Q6_K | 0.004475 ± 0.000311 | 0.005256 ± 0.000422 | 0.8514 | 是 |
| 4B | Q4_K_M | 0.04220 ± 0.000862 | 0.04646 ± 0.000994 | 0.9084 | 是 |
| 9B | Q8_0 | 0.000506 ± 0.000047 | 0.001875 ± 0.000435 | 0.2699 | 否 |
| 9B | Q6_K | 0.005884 ± 0.000707 | 0.007075 ± 0.000784 | 0.8317 | 是 |
| 9B | Q4_K_M | 0.03895 ± 0.001004 | 0.05758 ± 0.001386 | 0.6764 | 否 |

- 兩個平台的 self-KLD（BF16 對自己的 base）都是 0（±0）。
- 每個 KLD 都是對自己平台的 BF16 base 算的。

**完整 PPL（580 chunks；H-Q1c）**

| size | 設定 | PPL Metal（`formal-20261031-q1-metal`） | PPL CUDA（Phase 1 `formal-20261010-quality`） | Metal ÷ CUDA | 在 [0.995, 1.005] 內 |
|---|---|---|---|---|---|
| 4B | BF16 | 9.9442 | 9.9454 | 0.9999 | 是 |
| 4B | Q8_0 | 9.9474 | 9.9573 | 0.9990 | 是 |
| 4B | Q6_K | 9.9857 | 9.9949 | 0.9991 | 是 |
| 4B | Q4_K_M | 10.3867 | 10.3916 | 0.9995 | 是 |
| 9B | BF16 | 8.1865 | 8.1862 | 1.0000 | 是 |
| 9B | Q8_0 | 8.1935 | 8.1950 | 0.9998 | 是 |
| 9B | Q6_K | 8.2113 | 8.2139 | 0.9997 | 是 |
| 9B | Q4_K_M | 8.3054 | 8.1983 | 1.0131 | 否 |

- CUDA 的 PPL 依 040 實作補記 4：BF16 取 Phase 1 self-KLD row 的 PPL(Q)，各 quant 取 KLD row 的 PPL(Q)。改用 Phase 1 的 ppl row 或 kld_base row 的 PPL，比值只在第 6 位小數不同；改用 KLD row 的 `ppl_base`（9.940257 / 8.182939），BF16 的比值為 1.00040 / 1.00044。三種對照都沒有任何格改變是否在界線內。

### Q2：長 context（RTX CUDA）

| size | quant | KLD c512（Phase 1，580 chunks） | KLD c2048（145 chunks） | ÷ c512 | KLD c8192（36 chunks） | ÷ c512 |
|---|---|---|---|---|---|---|
| 4B | Q8_0 | 0.002030 ± 0.000218 | 0.003366 ± 0.000294 | 1.6581 | 0.004338 ± 0.000421 | **2.1369** |
| 4B | Q6_K | 0.005460 ± 0.000212 | 0.008838 ± 0.000475 | 1.6187 | 0.01128 ± 0.000655 | **2.0652** |
| 4B | Q4_K_M | 0.04577 ± 0.000578 | 0.05616 ± 0.000984 | 1.2271 | 0.06237 ± 0.001293 | 1.3628 |
| 9B | Q8_0 | 0.002019 ± 0.000223 | 0.004225 ± 0.000423 | **2.0926** | 0.005012 ± 0.000500 | **2.4824** |
| 9B | Q6_K | 0.007259 ± 0.000427 | 0.01263 ± 0.000765 | 1.7396 | 0.01376 ± 0.000863 | 1.8963 |
| 9B | Q4_K_M | 0.05745 ± 0.000844 | 0.07536 ± 0.001675 | 1.3119 | 0.07539 ± 0.001791 | 1.3124 |

- 粗體為超出 [0.5, 2.0] 的 4 格。self-KLD 都是 0（±0）。
- c512 是 Phase 1 的 session，c2048 / c8192 是 Phase 3 的 `formal-20261031-q2`：同一個 RTX 平台、不同 session。

### 順序的 z 檢定（H-Q1a：Metal、前 200 chunks；H-Q2a：CUDA、c2048 與 c8192）

z = (KLD_hi − KLD_lo) ÷ √(SE_lo² + SE_hi²)，雙尾，各族內 Holm，α 0.05（005 的 `kld_test` 與 `holm`）。

| 族 | 4B Q6_K > Q8_0 | 4B Q4_K_M > Q6_K | 9B Q6_K > Q8_0 | 9B Q4_K_M > Q6_K |
|---|---|---|---|---|
| H-Q1a（c512） | 11.50 | 41.17 | 7.59 | 26.93 |
| H-Q2a c2048 | 9.80 | 43.31 | 9.61 | 34.07 |
| H-Q2a c8192 | 8.91 | 35.25 | 8.78 | 31.00 |

- 12 個檢定的方向都正確，Holm 之後都拒絕。最小的 z 為 7.59（Holm 第一步的臨界 z：Q1a 2.50、Q2a 2.73）。
- p 值：分析以 2·(1 − Φ(|z|)) 計算，12 個中有 11 個為 0.0（Φ(|z|) 在倍精度下捨入為 1，相減後為 0），不是真的 0；唯一非 0 的是 9B c512 Q6_K > Q8_0 的 3.2e-14。所以全部 p < 1e-13。
- SE 的兩個方向的偏差（005、040）：
  - 同一批 token 上的兩個 KLD 是正相關，用兩個邊際 SE 的平方和會高估差的 SE，屬保守；
  - 工具的 SE 假設 token 之間獨立，屬樂觀。
  - 兩者的淨效果無法從這些數據得知；最小 z 約為 Holm 第一步臨界值的 3 倍。

## 解讀（只寫 040 允許的部分）

- **順序（H-Q1a、H-Q2a，supported）**：Q8_0 < Q6_K < Q4_K_M 的 KLD 順序在 Metal（`-c 512`、前 200 chunks）與 CUDA 的 2K / 8K context 都成立，和 Core A 的 CUDA `-c 512` 結果一致。
- **兩個平台的數值差異（H-Q1b、H-Q1c，not supported）**：兩個平台（Metal 與 CUDA；Q1c 的 CUDA 一側來自 Phase 1）的數值差異超過事先寫定的實質界線。依 040 的揭露，這代表兩個平台的數值差異超過界線，**不代表任何一個平台有缺陷**；040 舉兩個 backend 的量化矩陣乘法不同（例如 CUDA 的 MMQ 會把 activation 量化成 q8_1）作為這樣解讀的理由。Stage Q 沒有做 backend 消融，不把結果歸因到任何 kernel。
  - Q1b：6 格的 Metal ÷ CUDA 都小於 1（0.270–0.908），超出的 3 格都低於 0.8。每個 KLD 都對自己平台的 BF16 算，沒有共同的參考，所以這不表示哪個平台比較接近「真正的」分布，也不是品質的優劣或平台排名。
  - Q1c：只有 9B Q4_K_M 超出（+1.31%）；其他 7 個設定的差距在 −0.10% 到 +0.003% 之間；9B BF16 的比值是 1.00003。
  - 9B Q4_K_M 在 Q1b 是 Metal 的 KLD 較低，在 Q1c 卻是 Metal 的 PPL 較高。同樣的前 200 chunks、同一對 Phase 3 session 的 KLD row 中，9B Q4_K_M 的 PPL(Q) 也是 Metal 較高（8.1218 對 8.0266，比值 1.0119；BF16 為 0.99993），所以這個對比不是 chunk 範圍或 session 不同造成的。KLD 與 PPL 量的東西不同，本文只並列，不加解釋。
  - **Phase 1 的 pilot 已經預示部分結果**（040 揭露，審查時於 2026-10-23 補記）：Phase 1 的 20-chunk pilot（Metal `pilot-20261008-both`、CUDA `pilot-20261010-rtx-q`）已經比較過兩個平台，顯示 H-Q1b 在 9B、H-Q1c 在 9B Q4_K_M 很可能不成立；界線與預測沒有因此修改。對照 pilot：9B Q8_0（pilot 0.32 → 0.270）與 9B Q4_K_M（0.59 → 0.676）兩次都超出；9B Q6_K 在 pilot 超出（0.68），這次在界線內（0.832）；4B Q8_0 在 pilot 在界線內（0.81），這次超出（0.353）。H-Q1c 的 9B Q4_K_M 在 pilot 是 1.025，這次是 1.013，兩次都超出。所以 H-Q1b 與 H-Q1c 的 not supported 都是 pilot 已經預示的方向；H-Q1c 超出的設定和 pilot 相同（只有 9B Q4_K_M）；H-Q1b 超出的格子和 pilot 不完全相同（4B Q8_0 新超出、9B Q6_K 回到界線內），所以 Q1b 的格子分布不能寫成「如 pilot 所預期」。
- **長 context（H-Q2b，not supported）**：4 / 12 格超出上界 2.0（最大 2.48），沒有任何一格低於 0.5。040 用「沒有數量級的變化」說明這條界線，但事先登記的操作定義是 [0.5, 2.0]，所以判定就是 not supported，不事後放寬。
  - 描述（040 沒有方向預測；c512 來自 Phase 1，跨 session）：12 個比值都大於 1（1.23–2.48）；Q8_0（1.66–2.48）與 Q6_K（1.62–2.07）的比值大於 Q4_K_M（1.23–1.36）。c2048 → c8192 的點估計沒有一個下降，但只有 4B Q6_K 與 4B Q4_K_M 的差距超出 2·√(SE² + SE²)，其他 4 個在雜訊內。這是事後的描述，只有一個 session，不寫成「誤差隨 context 累積」之類已確立的機制。
  - 超出的 4 格是 3 格 Q8_0 與 4B Q6_K c8192；絕對增加量以 Q4_K_M 最大（+0.010–0.018）。040 只定義比值的界線，本文不依絕對量解讀判定。
- **不改變的判定**：H-A5 與 Core A 的 quality 判定不變（040 揭露）。Stage Q 只是新的平台與 context 範圍的證據。

## 限制

- 每個部分只有一個正式 session，沒有估計 session 之間的變異。
- 只有 WikiText-2 test（pins 的 `wiki.test.raw`）；downstream 題庫沒有在新平台或新 context 下重測（040 的範圍）。
- llama.cpp b11146、005 的 flag set（Q2 只改 `-c`）；兩個 size（4B、9B）× BF16 / Q8_0 / Q6_K / Q4_K_M；只有一台 Mac（Metal）與一張 RTX 5070 Ti（WSL2 CUDA），不推廣到其他 GPU、backend 或 build。
- Q1 的 KLD 只有前 200 chunks（受 Mac 磁碟限制），不能直接和 Core A 的 580-chunk KLD 比。
- 長 context 只在 CUDA 上量。c8192 只有 36 chunks。
- Q1c 與 Q2b 是跨 session 的比較：CUDA 的 PPL 與 c512 的 KLD 來自 Phase 1（約三週前、不同部署）。Q1c 的 Metal 一側在 140W adapter（協商 94 W）、powermode 2 下取得，Phase 1 的 Mac 規則是 70W（037 §4）。
- 設計者在登記前看過 Core A 的 quality 結果，只用於 H-Q1b、H-Q1c、H-Q2b 的界線尺度與 KLD 的量級（040 揭露）。
- 偏離：q1-metal 第 2 次嘗試從 worktree 執行；q2 期間在 Mac 上以 ssh 唯讀列出 WSL 目錄一次（見「執行」）。
- Phase 1 的 20-chunk pilot 在登記前已預示 H-Q1b、H-Q1c 不成立的方向（040 揭露；見「解讀」）。

## 能支持與不能支持

- 能：
  - KLD 的順序在 Metal（`-c 512`、前 200 chunks）與 CUDA 的 2K / 8K 都成立。
  - 兩個平台（Metal 與 CUDA；Q1c 的 CUDA 一側來自 Phase 1）的 KLD 與 PPL 差異，在部分設定超過事先寫定的界線（Q1b 3 / 6 格、Q1c 1 / 8 個設定）。
  - （事後的描述，040 沒有方向預測）在 CUDA 上，2K / 8K 的 KLD 相對 Phase 1 的 512 都比較大（跨 session），其中 4 / 12 格超過 2 倍。
- 不能：
  - 說哪個平台的數值比較「正確」或品質比較好，或做平台排名。
  - 把差異歸因到 MMQ 或任何特定的 kernel。
  - 說 KLD 隨 context 增加是一般規律或已確立的機制。
  - 推廣到其他資料、build、硬體或下游任務。
  - 改變 H-A5 或 Core A 的判定。

## 數字來源

- **analysis-20261031.json 直接讀取**：`sessions`、`parts.*.{attempts, superseded, status, session, void_before}`、`phase1_session`、`harness_sha256`、`stage_q_sha256`、`pins_sha256`、`hypotheses.*.{verdict, prediction, bounds, ratios, inside, tests[].{name, z, p, direction_ok, p_adj, reject}}`。
- **session 的 rows 與檔案**：KLD 的 mean ± SE、PPL 與 chunk 數取自各 session 的 `rows.jsonl`（只取 ok、沒有 variant 或 repeat_of 的 row，與分析的選取規則相同）；`session.json`、`plan.json`、`env/*.json`；`results/stage_q/wsl-inventory.txt`；Phase 1 的 `results/quality/formal-20261010-quality/rows.jsonl`。
- **計算**：Holm 臨界 z、p 值為 0 的說明、計分 token 數、pilot 的比值（040 揭露）、c2048 → c8192 的差距與 2·√(SE² + SE²)、Q4_K_M 的絕對增加量。
- **其他欄位與 git**：env 擷取的 `load.mac_heavy_lock_owner`（cwd）、`repo.commit`、`repo.dirty`、`repo.status`；KLD row 的 `metrics.ppl_q` 與 `ppl_base`；commit `09e2776`、`b9088a3` 與 session 期間主 checkout 的 commit。

## 獨立檢查

- 三個獨立的核對者，都沒有參與起草：
  - **重算**：不 import `stage_q_analysis`，依 040 與 `quality_analysis` 的定義自己寫程式，從四個 session 的 rows 重算每個 KLD、PPL、z、p、Holm、比值、界線判定與五個判定；和 analysis-20261031.json 逐項比對 117 項，全部一致。另外核對 chunk 數、每個 job 都 ok、plan 與 rows 一致，並以 Metal 的 raw log 核對 row（CUDA 的 raw 沒有取回，只做 row 層級的核對）。本文的 KLD 與 PPL 表取自這次重算。
  - **重跑與來源**：以和 `main()` 相同的輸入呼叫 `analyze_sessions`，輸出和 commit 的 JSON 逐位元組相同；harness、設定檔與 pins 的 sha256 相符；每個部分的嘗試與選出的 session 符合規則；plan 等於凍結設定下的正式 plan；env 擷取的 commit 與 dirty 如「執行」所述；指出了 worktree 與補記 14 的差異。
  - **解讀**：依 040 確認五個判定，並列出本文必須寫與不能寫的項目（都已納入「解讀」「限制」與「能支持與不能支持」）。
- 核對腳本放在 session 的 scratchpad，沒有 commit。
- **核對後的更正**（兩位審查者，一位核對數字與事實、一位核對說法與 040 的規則；修訂時逐項再核對）：
  1. 從 worktree 執行有直接紀錄（env 擷取的 `load.mac_heavy_lock_owner` 記錄 cwd），不是推論；補上可以與不能事後證明的部分、符合規則的做法、clone 補救同樣偏離補記 14、搬回與 commit 的經過。
  2. 刪掉「差距不是 200-chunk 子集造成的」（只看 CUDA 一側、跨 session，Metal 沒有 580-chunk 的 KLD，不能支持因果說法）。
  3. 判定的標題寫明三個預測不成立；狀態列與「限制」列出兩處偏離（worktree、q2 期間的 WSL 列表）。
  4. 刪掉「所以這個差距不是來自 BF16 的基準」；MMQ 只作為 040「不代表缺陷」的理由，不寫成可能的原因；「backend」改為「兩個平台」。
  5. 9B Q4_K_M 的 KLD / PPL 對比改用同樣 200 chunks、同一對 session 的 PPL(Q)（1.0119）描述，不再以 chunk 或 session 不同作理由。
  6. pilot 的對照改寫：H-Q1b 與 H-Q1c 的 not supported 都是 pilot 預示的方向，Q1c 超出的設定和 pilot 相同，Q1b 的格子分布不同；「限制」加上這項登記前的知識。
  7. Q2b 的描述：刪掉依絕對量淡化判定的說法，改為完整的中性描述；c2048 → c8192 只有 2 個超出雜訊；「能支持」標明是事後描述、跨 session。
  8. 小處：作廢原因只是 env 無效；p 值為 0 的原因是捨入相消，不是下溢；PPL 對照的欄位名稱；WSL 列表的時間 00:29:55Z。
