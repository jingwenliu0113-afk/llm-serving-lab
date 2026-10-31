# 047 — Stage E3 的結果（formal-20261031-e3-metal、formal-20261031-e3-cuda）

- 狀態：**結果**。依 041 v3 §3 凍結的規則執行（`freeze.e3_metal = true`、`freeze.e3_cuda = true`，凍結 commit `cd615d0`，2026-10-30 22:50:55Z），凍結之後沒有改過任何規則、門檻或讀法；有一處執行上的偏離（Mac 正式 session 從 worktree 執行，見「偏離」）。兩個平台各自判定，不合併（041 §3.4）。E3 只描述 H-A2 被推翻的原因，不改變 Core A 的 H-A2 判定（refuted；PREREGISTRATION §11）。
- 日期：2026-10-31
- 對應：041（§0、§3.1–3.7、§5、修訂紀錄）、PREREGISTRATION §2（H-A2）與 §11、Core A 的 H-A2 被推翻（`formal-20261009-rtx-a1` 與 Mac Core A）、037 §4（Phase 3 的 Mac 用 140W adapter）。
- 輸出（commit `99f3327`）：`results/stage_e/e3-analysis-20261031/e3.json`（sha256 `742360ee…`），由 `benchmarks/stage_e3_analysis.py` 在 Mac 上產生。兩個 session 的檔在 `results/stage_e/formal-20261031-e3-metal/`（`7a0e694`）與 `results/stage_e/formal-20261031-e3-cuda/`（`38a002e`）。
- **讀法（041 §3.5，依序取第一個符合的列）**：
  - **Mac**：第 3 列「H-E3a 與 H-E3b 都 supported」→ **非比例成分與 4-bit k-quant 的格式成本並存**。
  - **RTX**：第 6 列「H-E3a supported，D 的方向或狀態在兩個 size 不同」→ **格式效應不一致，不解讀 H-E3b**。
  - **兩個平台**：落在不同的列，所以**原因依平台而異**，不寫一般結論（`cross_platform.statement`）。這是 041 §3.5 對「落在不同列」的固定寫法；RTX 的格式效應沒有解讀，所以它不表示 RTX 的原因已知而且和 Mac 不同。
- 本文的 JSON 路徑：沒有前綴的是 e3.json 的 `by_platform.<platform>.`；`session.`、`plan.`、`rows[i].` 是 session 目錄中的檔。

## 執行

1. **GGUF**：兩個 Q4_0 在 2026-10-23 於 Mac lock 下產生並 pin（4B `ee05b023…`、9B `2dfcbfd3…`），2026-10-25 傳到 WSL 並以 sha256 驗證。另外 6 個是 Core A 的 GGUF（pins）。
2. **pilot**（結果遮蔽，凍結前沒有看）：
   - Mac `pilot-e3-metal-20261023`（Mac lock，2026-10-23）：5 個 job ok，140W adapter（協商 94 W），env 有效。
   - RTX `pilot-e3-cuda-20261031`（GPU lock，2026-10-30 22:37–22:49Z，部署 `2c5fddf`）：5 個 job ok，sentinel 正常，session B 沒有 drift，env 有效。
3. **凍結**：commit `cd615d0`。兩個 pilot 的 `pilot_problems` 都沒有問題，harness `85b9842c…` 和兩個 pilot 記錄的相同。041 §3 的設計、門檻與讀法都沒有改。
4. **RTX 正式 `formal-20261031-e3-cuda`**（GPU lock，部署 `cd615d0`）：
   - 2026-10-30：指令約 22:51Z 開始（commit `38a002e` 的訊息）；session B 在建立 session 目錄之前量（041 §3.6），`session.started_at` 為 22:59:14Z，`finished_at` 為 23:20:29Z。
   - 量測時間 1,272.9 s，上限 3,600 s，沒有到上限。
   - 26 / 26 ok，`env_valid` true，`env_flags` []。
   - session B 245 MiB（凍結值 811 MiB），drift false。B 只用於記憶體判定，而記憶體判定需要 fit run，E3 不適用（041 §3.6）。spill 的速度判定比較每格的 decode 速度和錨點 4B-Q4_K_M 依 bytes 換算的預期：錨點以外的 7 格最低的比為 1.028（4B-Q4_0），門檻 0.5；e3.json 中 8 格的 spill 都是 false。
   - sentinel：184.76 → 187.02 tok/s，漂移 1.22%（門檻 3%）。
   - `results/stage_e/wsl-inventory.txt`（00:29:55Z）只列出這一個 E3 RTX 正式目錄，所以它是第一個也是唯一的嘗試。
5. **Mac 正式 `formal-20261031-e3-metal`**（Mac lock，從固定在 `b9088a3` 的 git worktree 執行；見下面的「偏離」）：
   - 00:05:47–00:29:31Z，量測時間 1,423.3 s，上限 5,400 s，沒有到上限。
   - 26 / 26 ok，`env_valid` true，`env_flags` []，沒有充電暫停。
   - Apple 140W USB-C Power Adapter，協商 94 W；powermode 2。
   - sentinel：59.524 → 59.387 tok/s，漂移 0.23%。
   - 53 份 env 擷取都記錄 repo commit `b9088a3`、dirty false。
6. **分析**（Mac，`99f3327`）：E3 的 harness、設定與程式在 `b9088a3` 之後都沒有改（`git diff b9088a3..99f3327` 在 benchmarks/、scripts/、configs/、tools/ 為空）。

### 偏離：Mac 正式 session 從 worktree 執行

- 041 §3.6 寫「正式 session 在主 checkout 執行（worktree 的 `results/stage_e/` 看不到其他嘗試）」。這次 Mac 的正式 session 從固定在 `b9088a3` 的 worktree 執行，`--models`、`--bin` 指向主 checkout 的絕對路徑。
- 原因：Stage Q 的 q1-metal 第 1 次嘗試因為 session 期間主 checkout 有 commit 而 env 無效（040）。從固定 commit 的 worktree 執行，可以避免同一個問題。
- 符合規則的替代做法（留在主 checkout 執行、期間不 commit，040 也列為補救之一）同樣可行；這次選了 worktree，計畫寫在 session 前的 NEXT.md（`b9088a3`），但沒有在執行前修訂 041 §3.6 或另寫 decision log。
- 這條規則的目的是讓 runner 看見每一次嘗試。就可查的證據而言，這個目的有滿足（不能事後證明的部分見下）：
  - RTX 的正式 session 在 `38a002e` commit，是 `b9088a3` 的祖先，所以 worktree 中有它；
  - session 結束後，目錄搬回主 checkout，在 `7a0e694` commit（父 commit 是 `b9088a3`，中間沒有其他 commit），worktree 已移除。
  - 執行者當時的觀察（沒有 commit 的紀錄）：開始前在 worktree 中執行 `stage_e3.formal_attempts`，Mac 沒有嘗試、RTX 只有 `formal-20261031-e3-cuda`（有效）；worktree 建立時主 checkout 的 git status 是乾淨的。
- 不能事後證明的部分：上一項的兩個觀察，以及 worktree 建立當下主 checkout 是否有 git 忽略、未 commit 的 Mac E3 嘗試目錄。目前主 checkout 的檔案系統中（含 git 忽略的內容）沒有其他 E3 正式目錄，這段期間也沒有執行過其他 Mac E3 session。
- 對數據與讀法沒有影響：同一個 harness（`85b9842c…`）與設定，plan 等於 `build_plan` 的正式 plan。

## 結果

### 每格的 decode 時間（L 2048，每格 3 個 ok 的 process）

t = 1000 / m（ms/token），SE(t) = 1000 × se / m²；x 是 041 §3.3 的 decode bytes（GB）。殘差 r = t − (a + b·x)，用 6 個舊設定的全域擬合；% 是 100 × r / t（觀測值）。041 沒有規定 % 的分母，這裡照實寫出。兩個平台各自分析，不比較快慢。

**Mac（`formal-20261031-e3-metal`）**

| 設定 | x (GB) | t | SE | 殘差 ms（%） |
|---|---|---|---|---|
| 4B-Q8_0 | 4.6528 | 22.2028 | 0.0004 | −1.183（−5.33%） |
| 4B-Q6_K | 3.6344 | 19.7200 | 0.0047 | +0.063（+0.32%） |
| 4B-Q4_K_M | 2.8792 | 16.8498 | 0.0075 | −0.042（−0.25%） |
| 4B-Q4_0 | 2.7143 | 15.3194 | 0.0027 | −0.968（−6.32%） |
| 9B-Q8_0 | 8.6172 | 37.1151 | 0.0039 | −0.787（−2.12%） |
| 9B-Q6_K | 6.6953 | 32.2195 | 0.0083 | +1.355（+4.21%） |
| 9B-Q4_K_M | 5.2274 | 26.0831 | 0.0089 | +0.593（+2.28%） |
| 9B-Q4_0 | 4.9116 | 23.9483 | 0.0047 | −0.385（−1.61%） |

全域擬合：a 6.3492 ms，b 3.6616 ms/GB，s 1.0262 ms（4 df），SE(a) 1.2295，SE(b) 0.2187。

**RTX（`formal-20261031-e3-cuda`）**

| 設定 | x (GB) | t | SE | 殘差 ms（%） |
|---|---|---|---|---|
| 4B-Q8_0 | 4.6528 | 7.2445 | 0.0189 | −0.113（−1.55%） |
| 4B-Q6_K | 3.6344 | 6.0662 | 0.0390 | −0.091（−1.51%） |
| 4B-Q4_K_M | 2.8792 | 5.3658 | 0.0268 | +0.098（+1.83%） |
| 4B-Q4_0 | 2.7143 | 4.9199 | 0.0343 | −0.154（−3.13%） |
| 9B-Q8_0 | 8.6172 | 11.9421 | 0.0499 | −0.085（−0.71%） |
| 9B-Q6_K | 6.6953 | 9.9722 | 0.0222 | +0.209（+2.10%） |
| 9B-Q4_K_M | 5.2274 | 8.0154 | 0.0051 | −0.018（−0.23%） |
| 9B-Q4_0 | 4.9116 | 7.5035 | 0.0072 | −0.158（−2.11%） |

全域擬合：a 1.8764 ms，b 1.1779 ms/GB，s 0.1431 ms（4 df），SE(a) 0.1714，SE(b) 0.0305。

### 前提與 hypotheses（041 §3.4）

| 項目 | Mac | RTX |
|---|---|---|
| 前提：H-A2 配對（9B-Q8_0 / 4B-Q4_K_M）的觀測比 | 2.2027（bytes 比 2.9929，SE 0.0010，差距 783 SE）→ 成立 | 2.2256（bytes 比 2.9929，SE 0.0145，差距 53 SE）→ 成立 |
| H-E3a：3 個舊格式的 a_f 都 > 2·SE（預測 supported） | Q8_0 4.701 ± 0.005、Q6_K 4.878 ± 0.014、Q4_K_M 5.528 ± 0.020（277–1,017 SE）→ **supported** | Q8_0 1.731 ± 0.072、Q6_K 1.428 ± 0.089、Q4_K_M 2.117 ± 0.060（16–35 SE）→ **supported** |
| κ（同 size 的 \|r(Q6_K) − r(Q4_K_M)\| 最大值） | 0.7614 ms（來自 9B） | 0.2277 ms（來自 9B） |
| D_4B = r(4B-Q4_0) − r(4B-Q4_K_M) | −0.9266（SE_D 0.0369，門檻 0.7614）→ 負且超出 | −0.2517（SE_D 0.0438，門檻 0.2277）→ 負且超出 |
| D_9B | −0.9787（SE_D 0.0698，門檻 0.7614）→ 負且超出 | −0.1400（SE_D 0.0131，門檻 0.2277）→ 在內 |
| H-E3b：兩個 D 都「負且超出」（不預測） | **supported** | **not supported** |
| 讀法表的列 | 第 3 列 | 第 6 列 |

- 兩個平台都沒有 NT：8 格都有 3 個 ok 的 process，RTX 沒有 spill（Mac 不適用 spill 判定）。
- 門檻 max(2·SE_D, κ) 在兩個平台、兩個 size 都由 κ 決定（2·SE_D 為 0.026–0.140 ms）。
- 邊際：
  - Mac 的 |D| 超過門檻 0.165 ms（4B）與 0.217 ms（9B）。
  - RTX 4B 只超過 0.024 ms（|D| / κ = 1.106）；9B 的 |D| 0.140 比門檻 0.228 小 0.088 ms（在內）。SE_D 只含同 session 的重複雜訊，不含 κ 那種 k-quant 之間的散布。

### 描述量（041 §3.5；沒有門檻，不決定任何列）

- **各格式的截距 a_f 與兩兩差距**（SE = √(SE₁² + SE₂²)，不同的格視為獨立；沒有多重比較校正）：

| 差距 | Mac | RTX |
|---|---|---|
| Q8_0 − Q6_K | −0.177 ± 0.015 | +0.303 ± 0.114 |
| Q8_0 − Q4_K_M | −0.827 ± 0.021 | −0.386 ± 0.093 |
| Q8_0 − Q4_0 | +0.041 ± 0.010 | +0.003 ± 0.105 |
| Q6_K − Q4_K_M | −0.650 ± 0.025 | −0.689 ± 0.108 |
| Q6_K − Q4_0 | +0.218 ± 0.017 | −0.300 ± 0.118 |
| Q4_K_M − Q4_0 | +0.868 ± 0.022 | +0.389 ± 0.098 |

  - a_f(Q4_0)：Mac 4.660 ± 0.008；RTX 1.728 ± 0.077。
  - Mac 的 a_f 以 Q4_K_M 最大（5.528）；RTX 的 a_f 以 Q4_K_M 最大（2.117）。
  - Q4_K_M − Q4_0 的截距差距用的是和 D_4B、D_9B 相同的 4 格，不是另一份獨立的證據。
- **3 個舊格式截距的同質性**：Mac χ² 1,708.0（2 df，加權平均 4.755 ms）；RTX χ² 44.96（2 df，加權平均 1.847 ms）。兩個平台的截距都明顯依格式而不同。Phase 1 的值（Mac 6.8、RTX 28.4；041 §5 第 6 項）只並列揭露，不解讀差異（跨 session；Mac 的 Phase 1 是 70W，041 §5 第 11 項）。
- **共用斜率擬合**（6 個舊設定，每個 size 一個截距、共同斜率，3 df；a0 與 c 由兩個 size 恰好解出，沒有自由度）：
  - Mac：b 3.169 ± 0.263 ms/GB，a_4B 7.797，a_9B 10.111，s 0.712，a0 5.746 ms，c 0.4875 ms / 十億參數。
  - RTX：b 1.133 ± 0.052 ms/GB，a_4B 2.008，a_9B 2.220，s 0.141，a0 1.821 ms，c 0.0445 ms / 十億參數。
- **φ**（H-A2 配對的偏差中 affine 模型能解釋的份額；兩種都報，定義和設計時的 tally 相同）：

| | in-sample（6 個設定） | leave-pair-out（4 個設定，2 df） | 設計時的 tally（Phase 1；041 §3.5） |
|---|---|---|---|
| Mac | 0.948 | 0.596 | 0.91 / 0.455 |
| RTX | 0.925 | 0.692 | 0.94 / 0.730 |

  in-sample 把配對本身放進擬合，偏樂觀。tally 只並列揭露、不解讀差異（跨 session；Mac 的 tally 是 Phase 1 在 70W 下的數字，041 §5 第 11 項）。

## 對 H-A2 研究問題的意義（只寫 041 允許的部分）

- **前提**：兩個平台都在各自的 session 內重現了 H-A2 的偏差（Mac 觀測比 2.203、RTX 2.226，bytes 比 2.993）。Phase 1 的觀測比為 Mac 2.221（70W）、RTX 2.243（041 §3.1 的 2.22–2.24 是這兩個值）；這是跨 session 的並列，不宣稱數值上重現，也不解讀差異。
- **Mac（第 3 列）**：在這個 session、L 2048、llama.cpp b11146 下，非比例成分與 4-bit k-quant 的格式成本並存：
  - 3 個舊格式的跨 size 截距都遠大於 0（H-E3a）。這原本就是預期會成立的結果（041 §5 第 4 項），這次是在同一 session 內重現。
  - 兩個 size 的 Q4_0 殘差都比 Q4_K_M 低（D_4B −0.927 ms、D_9B −0.979 ms：相對 6 個舊設定的全域擬合，Q4_0 比 Q4_K_M 快），差距超出同 session 兩個 k-quant 之間的一般散布 κ（H-E3b）。
  - H-E3b 事先沒有預測，不算預測成功。041 §5 第 5 項：設計前看過 Phase 1 的類比（Q8_0 − Q4_K_M 的殘差差，Mac 4B −1.47、9B −1.34 ms，超出門檻；RTX −0.11、−0.12 ms，小於 κ），方向和這次的平台差異相同。Q8_0 不是 Q4_0，所以 H-E3b 仍然是不預測，但這次的結果不是設計者沒見過的型態。
  - 這裡的「格式成本」是 Q4_K_M 與 Q4_0 的配方層級對比，不是單獨 Q4_K block 格式的每 byte 成本：兩者都用 llama-quantize 的預設配方（不是 `--pure`，沒有 imatrix），Q4_K_M 每個 size 比 Q4_0 多 32 個 Q6_K tensor；兩個 Q4_0 唯一的 Q6_K tensor 都是輸出投影（9B 的 `output.weight`；4B 是 tie 的 `token_embd`，521,472,000 B），和同 size 的 Q4_K_M 相同（041 §3.2，`configs/stage_e3.toml`）。
- **RTX（第 6 列）**：H-E3a supported；格式效應在兩個 size 不一致（4B 負且超出、9B 在內），所以不解讀 H-E3b。這**不是**「RTX 沒有格式成本」，也不是「只有 4B 有格式成本」，041 第 4 列（「有非比例成分、沒有格式成本的證據」）沒有成立。D_4B 與 D_9B 只作描述值；4B 的狀態只以 0.024 ms 越過門檻，很脆弱。
- **兩個平台**：原因依平台而異（041 §3.5 對不同列的固定寫法；RTX 的格式效應沒有解讀，所以不表示 RTX 的原因已知而且和 Mac 不同）。041 §3.4 規定兩個平台各自判定、不合併，所以 H-E3a 的判定只逐平台列出：Mac supported；RTX supported。
- **E3 分不開的事**（041 §3.1，重複）：非比例成分可能是固定開銷（kernel launch、CPU 端），也可能是頻寬效率隨張量大小改變，本設計分不開；截距依格式明顯不同（χ²），正是分不開的原因。所以不把加權平均截距或 a0 寫成「每 token 的固定開銷」，也不把 c 寫成量到的每參數成本。
- φ 的 leave-pair-out 值表示 affine 模型在不看配對時能解釋的配對偏差份額：Mac 0.596；RTX 0.692。剩下的部分（Mac 0.404；RTX 0.308）**不**量化歸因給格式成本（沒有這樣定義的量）。

## 限制

- 只量 L 2048，每格 3 次重複；8K、32K 與 4B-Q5_K_M 都不在設計內（041 §3.2）。
- 兩個平台的 SE 都只反映同一 session 內 3 個 process 的重複雜訊，不代表跨 session 的再現性（前提的 783 / 53 SE、H-E3a 的 SE 倍數都依賴它）；Mac 的 z 值特別大（277–1,017）。不和 Phase 1 的 SE 倍數比較。每個平台只有一個 session。
- κ 是看過 Phase 1 數據後定的門檻（041 §5 第 3 項）。兩個平台的門檻都由 κ 決定，而 κ 都來自 9B 的 Q6_K 與 Q4_K_M。揭露（不改變判定，041 §0）：Phase 1 Mac 的 κ 為 0.996（70W），比這次 Mac 的兩個 |D| 都大。
- Mac 結果在 140W adapter（協商 94 W）、powermode 2 下取得。Phase 1 Mac（70W）的數字（κ 0.996、χ² 6.8、φ tally、觀測比 2.221）只在本文並列揭露，不用於判定，也不解讀差異（041 §5 第 11 項）。
- RTX 只套用 spill 的速度判定（E3 不跑 fit run，記憶體判定不適用，041 §3.6）。
- commit 的 rows.jsonl 只有 `model` 檔名欄位；llama-bench 自己回報的 `model_size` 與 `model_type` 在 git 忽略的 raw/ 中。Mac 的 26 列已用本機 raw 核對；RTX 的 raw 沒有取回，只能間接確認：runner 對每列的身分不符會記為 error，而 26 列都是 ok。
- 設計者在設計前看過 Core、Advanced、C0、Stage C 的全部數據，以及 Core A 的 12 個殘差、a_f、κ、φ（041 §5 第 1、2 項；tally 在 `results/stage_e/prereg-tally/`）。
- 041 §3.6 的「在主 checkout 執行」有偏離（見「執行」）。

## 能支持與不能支持

- 能：
  - Mac：在同一 session 內重現 H-A2 配對的偏差；有跨 size 的非比例成分（H-E3a）；Q4_0 對 Q4_K_M 的配方層級格式成本，在兩個 size 都超出同 session 的 k-quant 散布（H-E3b）。
  - RTX：在同一 session 內重現 H-A2 配對的偏差；有跨 size 的非比例成分（H-E3a）。
- 不能：
  - 分開固定開銷與頻寬效率（041 §3.1）。
  - 說 RTX 有或沒有格式成本。
  - 把 Mac 的格式成本推廣到所有 k-quant、其他量化配方（`--pure`、imatrix）、其他模型或 size（只有 Qwen3.5 4B / 9B）、其他 context 長度、其他 llama.cpp build、其他 session 或電力條件，或其他平台。
  - 改變 Core A 的 H-A2 判定。
  - 量化格式成本解釋了 H-A2 偏差的多少。
  - 比較 Mac 與 RTX 的快慢（本專案不做平台排名）。

## 數字來源

- **e3.json 直接讀取**：`cells.<cfg>.{t, se, n, usable, spill, x_gb}`、`fit.{a, b, s, se_a, se_b, residuals_ms, residuals_pct}`、`premise.{observed_ratio, bytes_ratio, se, holds, phi}`、`intercepts.{by_format, weighted_mean, homogeneity_chi2, df, differences}`、`shared_slope.*`、`format_contrast.{kappa, by_size}`、`H-E3a`、`H-E3b`、`row`；`cross_platform.{rows, general, statement}`；`harness_sha256`、`inputs_sha256`。
- **session 目錄**：`session.json` 的 `started_at`、`finished_at`、`status`、`outcomes`、`env_valid`、`env_flags`、`sentinel`、`session_b`、`ac_adapter`、`powermode`、`charge_pauses`、`measured_s`、`cap_reached`；`plan.json` 的 `cap_s`；`env/*.json` 的 `repo.commit` 與 `repo.dirty`；`raw_manifest.json`；`results/stage_e/wsl-inventory.txt`。
- **由 e3.json 計算**：差距的 SE 倍數（例如前提的 783 SE、53 SE；a_f / SE）、|D| 與門檻的邊際、1 − φ；spill 的 m / 預期比 = t(4B-Q4_K_M)·x(格) / (t(格)·x(4B-Q4_K_M))（`core_a_analysis.spill_cells` 的速度判定）。
- **其他檔**：`results/stage_e/prereg-tally/output.txt`（Phase 1 的觀測比 Mac 2.221、RTX 2.243）；041 §5 第 5 項（Phase 1 的 Q8_0 − Q4_K_M 類比）。
- **git**：`b9088a3`、`38a002e`、`7a0e694`、`99f3327` 的內容與時間，`git diff b9088a3..99f3327`。

## 獨立檢查

- 三個獨立的核對者，都沒有參與起草：
  - **Mac 與 RTX 各一位重算**：不 import `stage_e3_analysis`，只依 041 §3.3–3.5 與凍結的 `core_a_analysis` 定義自己寫程式，從 rows.jsonl、設定檔與 pins 重算每格、擬合、前提與 φ、a_f 與差距、同質性、共用斜率、κ、D、判定與讀法表的列。e3.json 中每個平台的 128 個值全部相符（最大相對差 1.75e-13 與 1.34e-12）。另外核對：每格 3 個 ok 的 process、plan 與設定檔一致、sentinel、session 狀態；Mac 26 列的模型身分以本機 raw 核對，52 個 raw 檔的 sha256 與 `raw_manifest.json` 相符。
  - **重跑與來源**：直接呼叫分析函式（不寫進 repo），輸出和 commit 的 e3.json 完全相同；`inputs_sha256` 與 harness 相符；兩個 session 都是各自平台第一個也是唯一的正式嘗試，plan 等於目前設定下的正式 plan；確認了 worktree 的時間線與 env 擷取的 commit。
  - **讀法**：依 041 §3.4–3.5 確認兩個平台的列都是依序第一個符合的列，跨平台的敘述符合規則，並列出本文必須寫與不能寫的項目（都已納入「對 H-A2 研究問題的意義」「限制」與「能支持與不能支持」）。
- 核對腳本放在 session 的 scratchpad，沒有 commit。
- **核對後的更正**（兩位審查者，一位核對數字與事實、一位核對說法與 041 的規則；修訂時逐項對檔案與 041 再核對過）：
  1. 「Q4_0 比 Q4_K_M 更接近 bytes 的預測」改為「相對全域擬合，Q4_0 比 Q4_K_M 快」（D < 0 的意思；Mac 4B 的 Q4_0 殘差其實離擬合更遠）。
  2. spill 的速度判定不用 B：B 只用於需要 fit run 的記憶體判定；比值 1.028 是錨點以外 7 格的最低值，補上算法。
  3. 前提不再拿兩個平台合成的 Phase 1 範圍逐平台比較，改為各自並列 Phase 1 的值、不解讀。
  4. Phase 1 Mac（70W）的數字只並列揭露：刪掉 χ² 差異的解讀與「可比的是 leave-pair-out」。
  5. 補上 041 §5 第 5 項的揭露（Phase 1 的 Q8_0 − Q4_K_M 類比）、「不改變 H-A2 判定」的聲明、4B-Q4_0 的 tie token_embd（Q6_K）混淆、更完整的推廣限制。
  6. 偏離段落：worktree 中的 `formal_attempts` 與主 checkout 乾淨是執行者的觀察（沒有 commit 的紀錄），列入不能事後證明的部分；補上符合規則的替代做法與「沒有事前修訂 041 §3.6」；狀態列註明有一處偏離。
  7. 跨平台的共同點寫法（a_f 的排序、「六到七成」、合併的「能支持」）改為逐平台列出；「原因依平台而異」加上它只是固定寫法的說明；刪掉 RTX 9B 的「離 0 有 10.7 SE_D」；SE 的限制改為兩個平台都適用。
  8. 時間：凍結日期改為 UTC（2026-10-30 22:50:55Z）；RTX 指令開始時間改為 commit 訊息中的「約 22:51Z」；上限的來源改為 `plan.json` 的 `cap_s`。
