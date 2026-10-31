# 050 — Stage E2 的結果（formal-20261031-e2）

- 狀態：**結果**。依 041 v3 §2 凍結的規則執行（`freeze.e2 = true`，凍結 commit `3eafb5f`，2026-10-31 01:51:13Z），凍結之後沒有改過任何規則、門檻或讀法，執行上沒有偏離。E2 只回答「T 在 2K、C 25 為什麼無效」，不改變 Advanced 與 Core B 既有的判定（H-B4、025 的 T512 類別 N；PREREGISTRATION §11）。
- 日期：2026-10-31
- 對應：041（§0、§2.1–2.11、§5、§6 A2–A4、修訂紀錄「E2 的 WSL 原始碼核對」「E2 共享前綴 dataset 的 pin」「E2 凍結」）、024（A1、A5、A7(i)、A10、A13 與尾端分解）、025（2K、C 25 的 T512 為 N）、PREREGISTRATION §11。
- 輸出（commit `b792196`）：`results/stage_e/e2-analysis-20261031/analysis.json`（sha256 `97eb041c…`），由 `benchmarks/stage_e2_analysis.py` 在 Mac 上產生。session 的檔在 `results/stage_e/formal-20261031-e2/`（`31502a8`；raw/ 留在 WSL，只 commit `raw_manifest.json`）。
- **讀法（041 §2.7，依序取第一個符合的列）**：**R4**，不是 incomplete。
  - 041 的文字：「C 8 重建的 KV 結構不足以讓 T 失效，C 25 的失效需要和 C 有關的因素。β_ok 成立：符合 S，但也可能是負載與 KV 的交互作用，分不開。」β_ok 成立，所以採這個分支。
  - T256 附註（T256@25 為 I）：「沒有鑑別力（T256 自己的切段與 preemption 也有成本）」。
  - 依 041 §2.11 與 §6 A4，E2 以條件式結論結束：**C 25 的成因沒有確立**。
- 本文的 JSON 路徑：沒有前綴的是 analysis.json；`session.`、`plan.`、`rows[i].`、`derived/` 是 session 目錄中的檔。

## 名詞（041 §2.1–2.5、024 A5）

- **問題與候選解釋**（041 §2.1）：025 中 T512 在 2K、C 25 無效。三個候選：
  - **S（負載）**：C 25 的總 prefill 工作讓 step 一直滿載，T 只是重新分配；
  - **V（T × KV）**：KV 受限時，T 讓長類分段配置 block，准入只檢查、不預留，造成 preemption 與重算或 HOL 阻塞，抵消 T；
  - **Th（threshold 太大）**：T512 仍讓長類每個 step 占 512 token。
  - T@8k 用來分開 S 與 V，T256@25 用來分開 Th 與 {S, V}。
- **格**：全部 L 2048。B0 是 024 的觀測版基準（KV 4 GiB）；T 是 `--long-prefill-token-threshold 512`（即 T512），T256 是 256；@25、@8 是並行數 C 25、C 8；**@8k 是 C 8 再把 KV 降到 2.5 GiB**（condition K2.5；T@8k 是 TK2.5；k 指 KV，不是 8K context）。PC-off / PC-on 是共享前綴 workload 上 prefix caching 的關與開。
- **workload**：uniform 只有短類 request（1,920 + 128 token）；mixed 是 pinned dataset 的短類加長類（32,640 + 128 token）。
- **量**（041 §2.2）：
  - U、M 是 uniform、mixed 的短類 P95 TTFT（3 次 restart 平均，ms；括號為 SE = s/√3）。ρ = ln(M/U) 是 mixed 對短類的懲罰；H-B4 說 2K 時這個懲罰存在。
  - T 類格和同一 session、同一 C、同一 KV 大小的 B0 比（下標 0 為 B0、c 為該格）：f = 1 − ρ_c/ρ_0 是懲罰被移除的比例；m = ln(M_0/M_c) 是 mixed 短類本身的改善；u = ln(U_c/U_0) 是 uniform 變慢的程度；ρ_0 − ρ_c = m + u。
  - noise：兩值之差 > 2√(SE₁² + SE₂²)。類別（024 A5）：M（f ≥ 0.5）與 P（0.2 ≤ f < 0.5）都要求 ρ 與 M 的下降超出 noise；R 是 f ≥ 0.2、ρ 下降但 M 沒有；W 是 f ≤ −0.2 而且 ρ 上升；N 是其他。
  - **E（有效）**：類別 ∈ {M, P}，而且 m ≥ 0.20；**I（無效）**：類別 ∈ {N, R, W}，或類別 ∈ {M, P} 但 m < 0.20（另標「主要靠 uniform 變慢」）。

## 執行

1. **WSL 原始碼核對**（2026-10-30 23:2xZ 複製並核對，commit `534be5b` 於 23:54Z）：041 §2.10 的五項都已回答。PC-on 命中計數器的下界改為 (0.95 × N + max(warmup − 1, 0)) × 1,584，扣掉 warmup 的命中後仍要求 ≥ 95% 的主 request 命中（041 修訂紀錄「E2 的 WSL 原始碼核對」）。沒有任何 E2 數據。
2. **共享前綴 dataset**（2026-10-31 00:59Z 產生，commit `bf56610` 於 01:00Z）：275 列 × 1,920 token，共同前綴 1,599 ≥ 1,584，sha256 `ca148974…`。
3. **smoke `smoke-e2-20261031`**（GPU lock，部署 `173cb8f`，01:27:52–01:50:10Z）：12 / 12 次 boot 有效，12 項檢查都通過，env 有效；延遲遮蔽，沒有看。
4. **凍結**：commit `3eafb5f`（01:51:13Z），`smoke_problems` 沒有問題，harness `a7a5f9d3…` 等於 smoke 記錄的值。041 §2 的設計、門檻與讀法都沒有改。
5. **正式 `formal-20261031-e2`**（GPU lock，部署 `3eafb5f`）：
   - 在 WSL 主 checkout 的 `results/stage_e/` 下執行（041 §2.9）；97 個 env 檔的 `repo.commit` 都是 `3eafb5f`、`dirty` 都是 false。
   - `session.started_at` 01:51:41Z，`finished_at` 05:21:36Z（3.50 h；`plan.cap_s` 27,000 s = 7.5 h，沒有到上限）。
   - 48 / 48 次 boot 有效（`session.outcomes`），`env_valid` true，`env_flags` []，沒有 `stopped`；每列 `valid` 與 `ok` 都是 true，`invalid_reasons` 都是空的。
   - 3 個 restart block 的順序等於 041 §2.5 的種子洗牌（`random.Random(int(sha256("20261008:stage_e:E2:block:{r}"), 16))`），也等於 rows 的實際順序。
   - 操弄檢查每列都通過：
     - non-default args 與 attention backend 逐 boot 記錄在 `session.runs.*.{nondefault, log_facts}`：48 次都恰好是該 condition 的 key 加觀測 key，都是 FLASH_ATTN；
     - block 數（B0 / T / T256 / PC 248、K2.5 / TK2.5 155）、block size 528；
     - T、T256 與 TK2.5 的 chunk 檢查（`manip.chunk_ok`）；
     - `mamba_cache_mode` PC-off "none"、PC-on "align"；`mamba_block_size`（只記錄）PC-on "528"、其他 "32768"。
   - PC 的命中：每列的 `span_cached_key` 都是 None（vLLM 0.30.0 的 span 沒有 cached-token 欄位），所以依 041 §2.8 一律由計數器判定。PC-on 三次的 Δ`prefix_cache_hits` 都是 434,016 = 274 × 1,584，也就是 275 個 request（250 個主 request 加 25 個 warmup）中只有 1 個沒有命中；計數器只給總數，這和「fresh boot 的第一個 warmup 必然未命中」一致。這個值在 041 §2.8 的範圍 [414,216, 437,184]（= [(0.95 × 250 + 24) × 1,584, (250 + 25 + 1) × 1,584]）之內。PC-off 都是 0。
   - A10（024 的分解檢查）：PC-on 以外的 45 次都有效；其中 T@25、T256@25、T@8k 的 mixed（9 次）有 preemption，依 024 A10 (c) 豁免該項（`rows[i].decomp.reasons`：「(c) exempt: N preemptions in the window」），描述量表中這三格的份額依賴這個豁免。PC-on 見核對事實第 6 項。
   - session 期間（01:51:41–05:21:36Z）我沒有存取 WSL（沒有 ssh、部署或取回；執行者的陳述，沒有 commit 的紀錄）。可查的時間：部署在 `started_at` 之前；取回（05:21:49Z）與 Stage V 的 dataset（05:23Z 起，044 實作補記十二）都在 `finished_at` 之後。
6. **取回**：05:21:49Z `tools/rtx/fetch_all_sessions.sh stage_e`；`results/stage_e/wsl-inventory.txt` 中 E2 只有 `smoke-e2-20261031` 與 `formal-20261031-e2`，所以這是第一個也是唯一的正式嘗試。
7. **分析**：`stage_e2_analysis.py --session results/stage_e/formal-20261031-e2 --out results/stage_e/e2-analysis-20261031/analysis.json`。E2 的 harness 與設定在凍結之後沒有改：`git diff 3eafb5f..b792196` 在 benchmarks/、scripts/、tools/ 為空，configs/ 只有 `stage_v.toml`；analysis.json 的 `harness_sha256` 等於 session 的 `a7a5f9d3…`。`inputs_sha256` 記錄 session.json、plan.json、rows.jsonl 的雜湊。在另一個新目錄重跑一次，輸出逐位元相同。

## 結果

### 每格（每格 uniform 與 mixed 各 3 次有效 restart）

preemption 是 mixed 3 次的加總（uniform 每次都是 0）；gate-zone 與 β 取 mixed（見下一節）。

| 格 | U | M | ρ（SE） | f | 類別 | m | u | E / I | preemption | gate-zone | β |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0@25 | 4,242.1（18.6） | 23,001.4（58.0） | 1.6905（0.0051） | 參照 | | | | | 0 | 0.460 | 0.915 |
| T@25 | 4,553.7（5.3） | 21,603.2（87.8） | 1.5569（0.0042） | 0.079 | N | 0.063 | 0.071 | **I** | 266（97 / 82 / 87） | 0.695 | 0.872 |
| T256@25 | 5,412.8（25.5） | 23,442.3（194.1） | 1.4658（0.0095） | 0.133 | N | −0.019 | 0.244 | **I** | 371（109 / 136 / 126） | 0.727 | 0.860 |
| B0@8 | 971.8（4.5） | 6,999.6（7.0） | 1.9744（0.0048） | 參照 | | | | | 0 | 0 | 0.333 |
| T@8 | 1,931.6（2.2） | 1,682.5（34.2） | −0.1381（0.0204） | 1.070 | M | 1.426 | 0.687 | **E** | 0 | 0 | 0.013 |
| B0@8k | 972.6（3.6） | 7,607.8（3.6） | 2.0569（0.0038） | 參照 | | | | | 0 | 0.151 | 0.495 |
| T@8k | 1,931.9（2.5） | 2,163.0（33.6） | 0.1130（0.0156） | 0.945 | M | 1.258 | 0.686 | **E** | 44（16 / 14 / 14） | 0.078 | 0.101 |

- 來源：`cells.<格>.{U, U_se, M, M_se, rho, rho_se, gate_zone, beta, preemptions_total, n_valid}`、`versus.<格>.{f, category, m, u, verdict}`。`cells.preemptions_total` 由程式加總該格 6 次 run（uniform 與 mixed）；uniform 都是 0，所以等於 mixed 的加總。
- 每個判定都不在門檻邊緣：E 的兩格，f（1.070、0.945）遠高於 M 的 0.5，m（1.426、1.258）遠高於 0.20；I 的兩格，f（0.079、0.133）低於 P 的 0.20，即使類別是 P，m（0.063、−0.019）也低於 0.20，仍是 I。

### H-C0 閘門與兩個條件

- **H-C0 閘門**（`gates`，`core_b_analysis.h_b4_values` 的判定）：檢查 B0 在該格是否重現 H-B4，也就是 M 比 U 高 ≥ 20%，而且差超出 noise。閘門只是 T 對比的前提。三個 B0 參照都通過：
  - B0@25：+442.2%，|M − U| 18,759 ms 對 noise 門檻 2√(SE_M² + SE_U²) 122 ms；
  - B0@8：+620.3%，6,028 對 17 ms；
  - B0@8k：+682.2%，6,635 對 10 ms。
- **β_ok**（`beta_ok`，負載差是否夠大）：β 是 mixed bench 期間 1 Hz 取樣中 `num_requests_waiting > 0` 的份額（取樣窗是兩次 scrape 之間，含 warmup 與 bench 前後，024 的慣例，原始碼核對已記錄）。β(B0@25) 0.915 − β(B0@8) 0.333 = 0.582 ≥ 0.20，**成立**。同一 restart block 內配對的差為 0.577–0.587；任意配對的差最小也有 0.573。
- **KV_ok**（`kv_ok`，K2.5 的 KV 操弄是否有效）：gate-zone 是 1 Hz 取樣中「有 request 在等、free block < 65（放不下一個長類）」的份額（025 的定義）。B0@8k 的 mixed gate-zone 0.151（SE 0.0073）高於 B0@8 的 0（3 次都是 0），超出 noise（差 0.151、noise 門檻 0.0146）；uniform 特異性 |ln(U_8k/U_8)| = 0.0008 ≤ 0.10。**成立**。

### 解讀表（041 §2.7）

| 列 | 條件 | 這次 |
|---|---|---|
| R0 | B0@25 的 H-C0 失敗 | 通過，不符合 |
| R0b | T@25 為 E | I，不符合 |
| R0c | B0@25 或 T@25 為 NT | 都不是 NT，不符合 |
| R1 | B0@8 的 H-C0 失敗 | 通過，不符合 |
| R2 | T@8 為 I | E，不符合 |
| R3 | T@8 為 E、KV_ok、T@8k 為 I | T@8k 為 E，不符合 |
| **R4** | T@8 為 E、KV_ok、T@8k 為 E | **符合** |

- 解讀表用到的輸入（三個 B0 的 H-C0、T@25 / T@8 / T@8k 的 E / I、KV_ok、β_ok）都有確定值、沒有 NT，所以不是 incomplete（041 §2.7：只有在某個 NT 可能改變所在列時才標 incomplete）。T256@25 也不是 NT，所以 T256 附註也是確定的。R3 的子標籤不適用。
- 「機制解讀只適用於 T512」的加註只在 T256@25 為 E、而且主列是 R3 或 R4 時才有；這次 T256@25 為 I，不加。
- analysis.json 的 `row.row` 為 "R4"、`row.incomplete` 為 false；它的英文 `reading` 與 `t256_note` 是 041 文字的意譯，本文引用 041 的原文。

### Hypotheses（041 §2.6）

| id | 判定 | 事先的預測 | 預測對嗎 | 依據 |
|---|---|---|---|---|
| H-E2-0 | supported | supported | 對 | 三個參照都通過（+442%、+620%、+682%） |
| H-E2-R | supported | supported | 對 | T@25 為 I（N，f 0.079，m 0.063） |
| H-E2a | supported | supported | 對 | T@8 為 E（M，m 1.426） |
| H-E2b | not supported | 不預測（V 預測 supported；S、Th 預測 not supported） | 不適用 | T@8k 為 E（M，m 1.258） |
| H-E2c | not supported | 不預測（Th 預測 supported；S、V 預測 not supported） | 不適用 | T256@25 為 I（N，f 0.133，m −0.019） |
| H-E2k | not supported | supported，信心低 | 錯 | a = ρ_8k/ρ_8 − 1 = 0.042 < 0.20；方向條件都成立 |
| H-E2e | supported | supported（能力檢查，幾乎必然成立） | 對 | Y_on / Y_off = 0.244，超出 noise |

- H-E2-0、H-E2-R 與 H-E2e 都列在 041 §5 第 4 項「已知或幾乎必然的結果」中。H-E2-0 只有 C 25 是已知的（025 為 +445%）；C 8 事先沒有直接數據，只有 Stage C 另一個 session 的類比（mixed 2K 在 C 4 為 +587%）。
- H-E2k 單獨報告（041 §2.7），主表用的是 KV_ok。方向條件（ρ 上升超出 noise、M 上升超出 noise、uniform 不變：|ln(U_8k/U_8)| ≤ 0.10）都成立，只有劑量 a 0.042 沒有達到 024 A7(i) 的 0.20（025 在 C 25 的 K3 為 0.123，041 §5 第 9 項）。
- H-E2b 與 H-E2c 事先不預測，所以不算預測成功或失敗。

### E2e：prefix caching 的能力檢查（041 §2.8）

- Y 是共享前綴 workload 的短類 P95 TTFT（250 個 1,920-token request，C 25）：Y_off 4,262.6 ms（SE 2.3），Y_on 1,040.9 ms（SE 1.6），比 0.244 ≤ 0.8，差 3,222 ms 遠超出 noise（`hypotheses.H-E2e`；SE 由 rows 計算）。
- 在這個 hybrid（GDN）模型與 vLLM 0.30.0 上、這個 bench 的量法下（C 25、開頭同時送出 25 個 request），PC 這個處理（`--enable-prefix-caching`，加上 vLLM 因此改用的 `mamba_cache_mode` "align"，041 §2.8）讓共享前綴的短類 P95 TTFT 降為 0.244 倍。這是能力檢查，不是 H-B4 的機制檢驗：所有基準都關閉 PC（041 §2.8、§6 A3）。
- **效果量的來源**（核對時發現，必須揭露；是事後的描述，不改變 H-E2e 的判定，041 凍結的 Y 就是這個 P95）：
  - 每個 bench 一開始同時送出 C 個 request。C 25 時，PC-off 三次 run 中 P95 以上的 13 個 request 都在前 25 個之中（前 25 個最慢約 8.0 s）；去掉前 25 個，P95 為 997 / 992 / 993 ms。
  - PC-on 的前 25 個最慢約 1.48 s；去掉後 P95 為 1,042 / 1,039 / 1,038 ms，和不去掉時幾乎相同。
  - 所以 0.244 主要反映 PC 消除了開頭同時送出時的 prefill 排隊。去掉前 25 個之後，PC-on 的 P95 反而比 PC-off 高約 4.6%，P50 為 770 對 991 ms（約 0.78）。0.244 不能寫成穩態下 prefix caching 的延遲效果。

### 描述量（041 §2.2；沒有門檻，不決定任何列）

mixed 的 3 次平均；表中只有 R 附 SE（括號），其他欄的 SE 與 uniform、共享前綴的值在 `descriptive.<格>/<mix>`。uniform 的 preemption、R、gate-zone、kv_full_while_waiting 每格都是 0。

| 格（mixed） | total tok/s | 長類 P95（ms） | KV 峰值 | kv_full_while_waiting | R | Λ | L̄q | budget-long | HOL-long | KV-short | unexplained-break |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0@25 | 4,864 | 23,162 | 0.993 | 0.076 | 0 | 0.847 | 8.80 | 0.783 | 0.041 | 0.062 | 0.009 |
| T@25 | 4,053 | 39,991 | 1.000 | 0.254 | 143,529（8,579） | 0.419 | 5.16 | 0.446 | 0.188 | 0.244 | 0.093 |
| T256@25 | 3,722 | 52,311 | 1.000 | 0.299 | 169,506（12,487） | 0.292 | 4.62 | 0.232 | 0.630 | 0.065 | 0.074 |
| B0@8 | 4,454 | 8,708 | 0.700 | 0 | 0 | 0.588 | 0.69 | 0.974 | 0 | 0 | 0.003 |
| T@8 | 4,245 | 9,678 | 0.699 | 0 | 0 | 0.026 | 0.009 | 0 | 0 | 0 | 0.167 |
| B0@8k | 4,417 | 8,800 | 0.981 | 0.005 | 0 | 0.612 | 1.44 | 0.903 | 0.020 | 0.016 | 0.107 |
| T@8k | 4,168 | 14,841 | 1.000 | 0.042 | 8,014（340） | 0.066 | 0.077 | 0.033 | 0.207 | 0.421 | 0.024 |

- 欄的意思：
  - R = bench 時間窗內 Σ ctx_tok − Δ`prompt_tokens_total`，也就是被 preempt 後重新計算的 token；
  - KV 峰值是 1 Hz 取樣中 KV 使用率的最大值；kv_full_while_waiting 是「有 request 在等、free block < 7（放不下一個短類）」的份額；
  - Λ 是 step 的 token 數（ctx + gen）達到每 step 上限 2,048 的時間份額；L̄q 是平均排隊中的短類數（Little's law）；
  - budget-long、HOL-long、KV-short 是 024 的尾端分解：每次 run 中 TTFT ≥ 短類 P90 的短 request，其排隊時間 Q 依當時的 step 分類的份額。budget-long：step 已用滿 2,048 token，而且有長類在 prefill；HOL-long：step 沒有滿，但排頭的長類等不到 65 個 free block；KV-short：step 沒有滿，排頭的短類等不到 7 個 free block。其餘類別（no-queue、budget-other、unexplained）不列，所以三欄相加不是 1；例如 T@8 三欄都是 0，表示它的尾端 Q 都落在沒有列出的 no-queue 與 budget-other，不是沒有排隊；
  - unexplained-break 是排隊 step 中歸為 unexplained 的比例（step 沒有滿、free block 也夠，排頭卻沒有准入）。
- **T@8 的 preemption**（041 §2.2 必報，因為 C 8 時 3 個長類同時在場需要 233 / 247 個 block，可能 preempt）：0（6 次 run 都是 0）。
- **交互作用 I_KV**（`interaction_I_KV`，只作描述，041 §2.7）= (ρ_B0@8 − ρ_T@8) − (ρ_B0@8k − ρ_T@8k) = 0.169（SE 0.026），三個 block 各自為 0.203、0.189、0.114（由 rows 計算）。
- **allocator**（`allocator`）：未操弄，未檢驗（041 §6 A2）；block 粒度浪費上界 0.031（024 A13）。unexplained-break 份額：mixed 見上表最後一欄（SE：T@25 0.003、T256@25 0.013、T@8 0.167、B0@8k 0.054、T@8k 0.002，其餘 ≤ 0.002）；uniform 為 T@25 0.032、T256@25 0.047、T@8 0.083、B0@8k 0.067、T@8k 0.083，B0@25、B0@8 為 0；PC-off 0，PC-on 不報（核對事實第 6 項）。分母很小的格子見核對事實第 7 項。

### 核對時發現、必須一起讀的事實（都不改變任何判定）

1. **K2.5 重建的是 KV 結構，不是 C 25 的 KV 壓力**（041 §2.4 已說「不是頻率」）。KV_ok 成立，但壓力遠低於 C 25：B0@8k 的 gate-zone 0.151 對 B0@25 0.460、kv_full_while_waiting 0.005 對 0.076；T@8k 的 preemption 44 對 T@25 266（約 1/6）、gate-zone 0.078 對 0.695（約 1/9）、R 8,014 對 143,529。R4 的「KV 結構不足以讓 T 失效」只對這個程度的 KV 壓力成立。
2. **描述上，K2.5 下 T 的改善較小，但沒有變成 I**：I_KV（只作描述，041 §2.7）為 0.169（SE 0.026），三個 block 都為正；在 K2.5 下 T 的 f 從 1.070 降到 0.945，mixed 短類 P95 從 1,682 升到 2,163 ms（+28.6%），長類 P95 從 9,678 升到 14,841 ms（+53%），uniform 不變（1,931.6 對 1,931.9）。R4 不能讀成「KV 對 C 8 的 T 沒有影響」。
3. **C 8 的 T 讓 uniform 變慢約一倍；T@8 的 ρ 降幅約 1/3 來自 uniform 變慢，但 E 另外要求的 m ≥ 0.20 也遠遠成立**：1,920-token 的短類 prompt 超過 512 的 threshold，每個都被切成 4 段，U 從 971.8 升到 1,931.6 ms（u 0.687；K2.5 下 0.686）。T@8 的 M 低於 U，所以 ρ 為負、f 大於 1（1.070）。ρ 的降幅 ρ_0 − ρ_c = m + u = 2.113，其中 32.5% 來自 uniform 變慢（T@8k 為 35%），所以類別 M 有一部分來自 u；但 E 另外要求 m ≥ 0.20，T@8 的 m 本身是 1.426，`mostly_uniform_slower` 為 false，所以判定照規則成立。T 在 C 8 的代價（描述）：uniform 短類 P95 約兩倍、長類 P95 +11%（8,708 → 9,678 ms）、mixed throughput −4.7%（4,454 → 4,245 tok/s）。
4. **C 25 的 U 由開頭一起送出的 25 個 request 決定**：
   - B0@25、T@25、T256@25 的 uniform 與 PC-off，每次 run P95 以上的 13 個 request 都在前 25 個之中；去掉前 25 個，B0@25 的 P95 為 993–1,011 ms（U 4,242）。C 25 的 mixed M 不由前 25 個決定（P95 以上的 12 個都不在其中）。
   - C 8 的 uniform U 與 B0 的 M 是穩態值（去掉前 8 個幾乎不變）；T@8、T@8k 的 M 有少量開頭成分（P95 以上的 9 個 request 中有 3 個是前 8 個，去掉前 8 個後 M 低 6–7%，m 因而更大，不影響 E）。
   - 所以 C 25 的 ρ、f、u 含開頭排隊的成分；R4 比較的 C 25 與 C 8 的 f，是在性質不同的 U 上算的。
   - 判定不受影響，理由是：T@25 與 T256@25 的 m（0.063、−0.019）只用 M，而且都低於 0.20，不論類別為何都是 I（041 §2.2）；T@8 與 T@8k 的 U 是穩態值。
   - 但類別本身依賴這個量法：事後把 uniform 的前 25 個去掉重算（描述，不是判定），T@25 的 f 約 0.25，依 024 A5 會是 P（另標「主要靠 uniform 變慢」）；T256@25 的 f 約 0.39，會是 R（M 沒有下降）；兩格仍是 I。所以「類別 N」與 025 的 N 是同一種量法下的同一類別，不是和量法無關的性質。這個量法和 024 / 025 完全相同。
5. **PC-on 的 R 與 throughput 不能照字面讀**：PC-on 三次的 R 都是 −434,016，正好等於 −Δ`prefix_cache_hits`（step 的 ctx_tok 只算真正計算的 token，`prompt_tokens_total` 含 cached token），扣掉命中後為 0，也就是沒有重算。2.2 的 R 定義不適用於 PC 列（原始碼核對已記錄）。PC-on 的 total throughput 13,897 tok/s（PC-off 5,052）把 cached 的 1,584 token 也算進去，反映的是 run 變短（36.8 對約 101 s），不是計算速度。
6. **PC-on 的分解份額不報**：三次 PC-on 的分解都在 024 A10 (c)（Σ ctx_tok 93,984 ≠ Δprompt_tokens 528,000）不通過，這是 prefix caching 必然造成的。依 041 §2.2「分解的份額（tail、q_shares、unexplained-break）還要 3 次分解都通過 A10」，`descriptive.PC-on/shared` 的份額與 unexplained-break 為 null。smoke 的 A10 檢查對 PC 也豁免 (c)（`run_stage_e2.a10_ok`），但 041 §2.2 與 024 A10 只對 preemption 豁免，所以正式分析對 PC 不豁免；原始碼核對紀錄說 PC 列的份額「只是近似（描述用）」，這次是不報。
7. **unexplained-break 份額在幾個格子只是 1 至 3 個 step**：T@8 mixed 0.167（SE 0.167）是三次中只有一次有 3 / 6 個排隊 step 無法解釋；T@8 uniform、T@8k uniform（各 0.083）、B0@8k uniform（0.067）都是三次中只有一次有 1 個（分母 4 或 5）。這些值不代表 unexplained break 的程度。B0@8k mixed 的 0.107（SE 0.054）在三次之間是 92 / 561、0 / 557、88 / 557，看起來是同一段排隊時間在 HOL-long 與 unexplained 之間換類（推論，step 在 raw/ 中，沒有取回檢查）。
8. **025 的「budget-long 0.91」和這裡的 B0@25 budget-long 0.783 不是同一個量**：025 是 mixed 減 uniform 的 ΔQ 份額，這裡是 B0 mixed 本身的 tail Q 份額，所以不並列比較。

### 和 025 的並列（不同 session，只作描述）

- 2K、C 25 的 B0：E2 的 ρ 1.6905，025 為 1.695。T512：E2 的 f 0.079、類別 N、m 0.063，025 為 f 0.081、N、m 0.069；preemption 每次 82–97（共 266），025 為 86–117（共 296）。這是跨 session 的並列，不宣稱數值上重現；E2 自己的判定（H-E2-R supported）只用 E2 的數字。

## 對 E2 研究問題的意義（只寫 041 允許的部分）

- **問題**：為什麼 T512 在 2K、C 25 無效？候選為 S、V、Th（見「名詞」）。
- **在同一 session 內**：T@25 無效（I，類別 N），重現 025；三個 B0 參照都有 2K 的懲罰。
- **C 8**：T 在 4 GiB（T@8）與 2.5 GiB（T@8k）都有效（E）。K2.5 在 C 8 重建了 C 25、4 GiB 的 KV 結構（1 個長類放得下，2 個長類加其他放不下；041 §2.4），KV 操弄也確實造成 gate-zone 與 preemption（KV_ok 成立，T@8k 有 44 次 preemption），但 T 仍然有效。所以依 R4：**C 8 重建的 KV 結構不足以讓 T 失效，C 25 的失效需要和 C 有關的因素。**
- **S**：β_ok 成立（B0@25 的 β 0.915 對 B0@8 的 0.333），所以結果**符合 S，但也可能是負載與 KV 的交互作用，分不開**。041 §2.11 不允許把它寫成「飽和是成因」：S 沒有直接證據，在 C 25 也無法以「移除」確立成因（KV 不能調高，負載與 KV 不能同時分開）。負載差以 β 看很大，以 throughput 看很小（見「限制」中 041 §5 第 8 項）。
- **V**：不寫成「V 被推翻」。被否定的只是 V 對 T@8k 的預測（I），也就是 R3 的讀法「在 C 8 的負載下，KV 壓力就足以讓 T 失效」。C 25、4 GiB 的 KV 壓力在頻率與程度上都比 C 8、2.5 GiB 高（核對事實第 1 項），透過這些和 C 有關的因素，KV 仍可能是 C 25 失效的一部分，這和 R4 的「也可能是負載與 KV 的交互作用」一致。描述上，K2.5 下 C 8 的 T 改善也較小（I_KV，核對事實第 2 項）。
- **Th**：不寫成「Th 被排除」。T256@25 也無效（I，類別 N，m −0.019），preemption 更多（371）；依 041 的 T256 附註，這「沒有鑑別力（T256 自己的切段與 preemption 也有成本）」。
- **四格的型態**（T@25 I、T@8 E、T@8k E、T256@25 I）恰好是 041 §2.1 表中 S 那一欄的預測；但依上面兩點，這不是在 S、V、Th 中單獨選出 S 的證據，只能寫成「符合 S」。
- **KV 劑量（H-E2k）**：not supported。在 C 8，B0 的 KV 從 4 GiB 降到 2.5 GiB，方向條件都成立，但 ρ 只升高 4.2%（a 0.042 < 0.20）；事先預測 supported（信心低），預測錯。
- **prefix caching（H-E2e）**：在這個模型與 vLLM 版本上、這次的量法下，PC 處理讓共享前綴的短類 P95 TTFT 降為 0.244 倍（命中 1,584 token）；效果主要來自 C 25 開頭一起送出的 request，去掉開頭 25 個後 P95 沒有縮短（E2e 段）。

## 限制

- 只有一個 session；SE 只反映同一 session 內 3 次 restart 的雜訊，不代表跨 session 的再現性。只量 L 2048、C 8 與 C 25、4 GiB 與 2.5 GiB。
- 「高負載、KV 不受限」的格子做不出來（KV 無法調高，熱 boot 上限約 4.14 GiB，041 §2.1），所以 S 只能條件式解讀。
- 041 §5 第 8 項：設計前已知 C 8 的 mixed 可能接近飽和（Stage C 不同 session 的 M2K，即 mixed 2K，在 C 4 已是 B0@25 mixed throughput 的 0.79），這是 S 只能條件式解讀的原因之一。這次同一 session 內，B0@8 mixed 的 total throughput 4,454 tok/s 是 B0@25（4,864）的 0.92；以排隊與滿載 step 看，負載差很大（β 0.333 對 0.915，Λ 0.588 對 0.847），以 throughput 看則很小。β_ok 只用 β。§5 第 7 項：「uniform 意義下的飽和就讓 T 無效」已有反例（025 的 8K C 8）。
- K2.5 重建的是結構不是頻率（041 §2.4）：pinned dataset 在 C 8 時兩個長類同時在場比 C 25 少；這次實測的 KV 壓力也遠低於 C 25（核對事實第 1 項）。
- T 在所有 condition 都固定為 512。依 024 的規則 T = 2,048 ÷ (n + 1)（n 是 KV 能同時完整容納的長類數），K2.5 下 n = 2，會算出 682；這裡不跟著改，只讓 KV 變動（041 §2.3 的揭露）。
- C 25 的 U、Y_off 由開頭一起送出的 25 個 request 決定（核對事實第 4 項）；這是 024 / 025 沿用的量法，凍結的 Y 就是這個 P95。
- β 的 0.20 是在沒有對應數據時定的，m 的 0.20 是在看任何 E2 數據之前由審查建議的（041 §5 第 3 項）。設計者在設計前看過 Core、Advanced、C0、Stage C 的全部結果與 025 的尾端分解（041 §5 第 1、2 項）。
- allocator 未操弄，未檢驗（041 §6 A2）。
- per-run 的 β 與 gate-zone 由 runner 從 1 Hz 序列算出，序列在 raw/ 中、沒有取回 Mac；核對只確認了 3 次平均、SE，以及每個值都是該 run 取樣數上的整數比。T、T256 與 TK2.5 的 chunk 檢查也只能從 rows 的結果欄位（`manip.chunk_ok`）確認，step 在 raw/ 中。
- `session.boots` 是空的清單：E2 的每次 boot 記錄在 `session.runs`（48 筆，依 plan 順序）；`run_stage_e2` 不呼叫會寫入 `boots` 的函式。不影響任何判定。
- 一次 B0 boot（`b3-B0-c8-uniform-r3`）的 bench 後記憶體增量 15,163 MiB 比 F_peak 15,157 MiB（031 閒置判定用的峰值）多 6 MiB，原因是它的閒置基準較低（235 對 245 MiB）；bench 後的絕對值 15,398 MiB 和多數 B0 相同（B0 與 PC-off 為 15,388–15,400 MiB）。多出的 6 MiB 只會讓閒置上限從 1,018 變成 1,012 MiB，而各 boot 的閒置值是 235–245 MiB，所以不影響任何 run 的有效性，只是描述（041 §2.9）。T256 與 PC-on 的增量也都在 15,157 + 128 MiB 之內。

## 能支持與不能支持

- 能：
  - 在同一 session 內，T512 在 2K、C 25 無效（重現 025 的 N）；三個 B0 參照都有 2K 的懲罰。
  - T512 在 2K、C 8 有效，在 4 GiB 與 2.5 GiB 都是；C 8 重建的 KV 結構（以這次量到的壓力）不足以讓它失效。
  - C 25 的失效需要和 C 有關的因素，符合 S，但和負載 × KV 的交互作用分不開。
  - T256 在 C 25 也無效（不能據此判斷 threshold 大小）。
  - C 8 的 B0 KV 劑量效果 not supported（H-E2k：方向條件都成立，但 a 0.042 < 0.20；事先預測 supported、信心低，預測錯）。
  - 在這個 hybrid 模型與 vLLM 0.30.0 上、這次的量法下，PC 處理讓共享前綴的短類 P95 TTFT 降為 0.244 倍（主要來自開頭的同時送出；去掉開頭 25 個後 P95 沒有縮短）。
- 不能：
  - 確立 C 25 的成因，或說「飽和是成因」（041 §2.11）。
  - 說 V 被推翻、Th 被排除，或 KV 對 T 沒有影響。
  - 把 0.244 寫成穩態下 prefix caching 的延遲效果，或把 PC-on 的 throughput、R 當成計算量。
  - 推廣到其他 C、L、KV 大小、threshold、模型、vLLM 版本、session 或平台。
  - 改變 H-B4、025 的判定或任何產品設定。

## 數字來源

- **analysis.json 直接讀取**：`cells.*`、`gates.*`、`versus.*`、`beta_ok`、`kv_ok`、`hypotheses.*`、`interaction_I_KV`、`descriptive.*`、`allocator`、`row`、`session`、`harness_sha256`、`inputs_sha256`。
- **session 目錄**：
  - `session.json` 的 `started_at`、`finished_at`、`status`、`outcomes`、`env_valid`、`env_flags`、`harness_sha256`、`datasets_sha256`、`boots`、`runs.*.{startup_s, cache_config, idle, nondefault, log_facts}`；
  - `plan.json` 的 `cap_s` 與 boot 順序；
  - `rows.jsonl` 的 `valid`、`ok`、`invalid_reasons`、`manip`、`span_cached_key`、`prefix_cache_hits`、`preemptions`、`recompute_tokens`、`ctx_tokens_window`、`decomp`、`beta`、`one_hz`、`duration`、`short_class_p95_ttft_ms`、`mem_after_bench_mib`、`mem_increment_mib`；
  - `env/*.json` 的 `repo.commit` 與 `repo.dirty`；`raw_manifest.json`；`results/stage_e/wsl-inventory.txt`。
- **由 rows、derived 或 cells 計算**（不在 analysis.json 中）：Y_off / Y_on 的 SE；H-C0 與 KV_ok 的 noise 門檻（由 cells 的 SE 算出）；每個 block 的 I_KV（各 block 的 4 個 ρ）；β 的配對差；去掉前 C 個 request 的 P95、P50，以及據此重算的 f 與類別（`derived/*-requests.jsonl` 的 `client_ttft_ms`、`idx`、`cls`，線性內插）；I_KV 的 SE 倍數、u / (m + u)、各種比值與百分比。
- **其他檔**：`results/advanced/formal-20261011-adv/analysis.json`（025 的數字）；041 §2.1–2.4、§5。
- **git**：`534be5b`、`bf56610`、`173cb8f`、`3eafb5f`、`31502a8`、`b792196` 的內容與 commit 時間；原始碼核對與 dataset 的事件時間取自 041 修訂紀錄。`git diff 3eafb5f..b792196`。
- **執行者的陳述**：session 期間沒有存取 WSL。

## 獨立檢查

- 五個獨立的核對者，都沒有參與起草，分別：
  - **判定量重算**：不 import `stage_e2_analysis`，依 041 §2.2、024 / 025 的定義自己寫程式，從 rows.jsonl 重算每格、閘門、對比、β_ok、KV_ok、7 個 hypotheses 與 I_KV，206 個值和 analysis.json 全部相符（相對誤差 1e-9）；再用凍結的 `adv_analysis`、`core_b_analysis` 函式做第二次比對，192 個值也全部相符；48 次 run 的短類 P95 由 derived 重算，最大相對差 4.3e-15；`inputs_sha256` 相符。
  - **描述量重算**：16 個 (格, mix) 的描述量全部重現；解釋了 PC-on 的負 R、PC-on 分解為 null、mean 等於 SE 的格子，以及 T@8 的 M 低於 U 的機制（切段讓 uniform 的短類每波 8 個一起 prefill，mixed 的長類打散了這個同步）。
  - **有效性稽核**：session 與 48 列的每一項有效性條件、harness 與設定檔雜湊、dataset pin、種子洗牌、上限、唯一嘗試；PC 命中的實際數字。
  - **解讀表**：逐列套用 041 §2.7，確認 R4、不是 incomplete、β_ok 的分支與 T256 附註；嘗試主張其他列的三個論點（B0@8 gate-zone 為 0、T@8 的負 ρ、T@8k 有 preemption）都不成立；列出 041 §2.11 允許與不允許的說法。
  - **異常懷疑者**：逐 restart 的一致性、block 之間的漂移（block 1 的第一次 boot 讓 B0@25 的 U 低 1.3%，沒有單調漂移，不改任何判定或 noise 旗標）、P95 是否由少數 request 決定、β 的取樣窗、和 025 的並列；提出了「核對時發現」的大部分項目。
- 分析在新目錄重跑一次，輸出和 commit 的 analysis.json 逐位元相同。
- 核對腳本放在 session 的 scratchpad，沒有 commit。
- **核對後的更正**（三位審查者，一位核對數字與事實、一位核對說法與 041 的規則、一位核對完整性與可讀性；修訂時逐項對檔案與 041 再核對過，例如去掉開頭 request 後的 f 與類別、9 次 A10 (c) 豁免、T@8 的 M 的開頭成分都另外重算）：
  1. 補上 041 §2.2 與 §6 A2 要求每格附的 unexplained-break 份額（描述量表最後一欄與 allocator 段），原稿只列了分母很小的格子。
  2. 核對事實第 4 項原本把 E / I 寫成只由 m 決定；改為 041 §2.2 的規則（類別加 m），並補上類別依賴 U 的量法（去掉開頭 25 個後 T@25 會是 P、T256@25 會是 R，仍是 I），以及 T@8、T@8k 的 M 也有少量開頭成分。
  3. 第 3 項的標題原本寫「E 靠的是 m，不是 ρ」，改為 ρ 降幅的 1/3 來自 u、m 另外也遠超過門檻。
  4. I_KV 原本寫成「確實削弱」並強調 SE 倍數；改為「描述上」，並標明 041 §2.7 只作描述。
  5. 補上 9 次 mixed run 使用 024 A10 (c) 的 preemption 豁免；PC 的 (c) 豁免只在 smoke，正式分析照 024 不豁免。
  6. E2e 與「能支持」中的 prefix caching 效果改為限定這次的量法，並寫出處理包含 `mamba_cache_mode` "align"；長段落拆成子項。
  7. 補上 041 §5 第 7、8 項（C 8 的 mixed 接近飽和、同 session 的 throughput 比 0.92）；「能支持」中的 H-E2k 改為寫明 not supported 與預測錯。
  8. 補上「名詞」一節（格、workload、U / M、ρ、f、m、u、類別、E / I、noise、S / V / Th）與閘門、β、gate-zone、尾端分解各欄的說明；ρ 的 SE 移進表內。
  9. 數字：β 的最小差（同 block 配對 0.577–0.587，任意配對 0.573）；去掉前 25 個後的 P95 範圍 993–1,011 ms、P50 991 ms；B0 的 bench 後記憶體 15,388–15,400 MiB；non-default args 與 attention backend 其實逐 boot 記錄在 session.json；chunk 檢查補上 TK2.5；commit 時間與事件時間分開寫；PC 命中的範圍寫出數值。
  10. 其他：「session 期間沒有存取 WSL」改為執行者的陳述；V 那一段的引號改為 R3 的原文；T256 附註的條件補全；incomplete 的輸入只列 T@25 / T@8 / T@8k；補上凍結後 harness 未改的核對、`mamba_block_size` 與 `span_cached_key`、024 的 T 規則與 F_peak 的說明；「數字來源」補上欄位與 `173cb8f`。
