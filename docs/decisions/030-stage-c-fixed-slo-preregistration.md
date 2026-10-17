# 030 — Stage C（C1）事先登記：Fixed-SLO 下的最大並發容量（FROZEN）

- 狀態：**FROZEN（2026-10-17）**。使用者已於 2026-10-15 選定研究決策（見下方「使用者決定」）。
  harness、兩次 smoke 與 C0 都完成後凍結：C0 的結果為分支 A（decision 032）。
  凍結 commit 是設定 `configs/capacity.toml` 的 `[freeze] capacity = true` 的那個 commit（見文末「凍結紀錄」）。
- 日期：2026-10-15
- 對應：decision 027 §2、028（C0）、PREREGISTRATION §8（DRAFT）
- 時點：任何 C1 數據之前。設計經過三份獨立設計（validity、product、efficiency 三個角度）與一位評審綜合（評審重新核對了所有引用的數字）。
- 凍結的方式比照 011 / 024：
  - 一起 commit 並 push：§8、本文、`configs/capacity.toml`、分析程式與測試；
  - 再設定 `configs/capacity.toml` 的 `[freeze] capacity = true`（下文簡稱 `freeze.capacity`）；
  - 之後才收集任何 C1 數據。

## 使用者決定（2026-10-15，任何 C1 數據之前）

| 項目 | 選擇 |
|---|---|
| SLO | **S2 標準互動**：短類 P95 TTFT 2 / 6 / 20 s（2K / 8K / 32K），全部 request 的 P95 TPOT 100 ms |
| mixed 的長 request | **budget**：長 request 也要符合 32K 的 TTFT 門檻（20 s） |
| 時間窗 | **穩態** [C, N − C)；全部 request 的統計量並列報告 |
| GPU lock 的占用 | 可以一次連續占用約 7–9 h（最壞約 12 h），GPU 空著就跑 |

- 選項 O-32K、O-T 沒有被選，依預設不做；要做需另外的決定。
- 下列項目依預設值處理（文末），使用者沒有異議：guard band m = 5%、n_near = 5（C0 落在分支 B / C 時為 7）、C_cap = 2·C_KV、開場湧入只在 §8 揭露。
- 依 8.5 的規則，只凍結 S2 的預測；S1、S3 只作為描述性的界線報告。

## 實作補記：設定檔放在 `configs/capacity.toml`，不放進 `pins.toml`（2026-10-15，凍結前）

- 凍結的 `adv_report` 會拒絕 pins sha256 和 Advanced session 不同的情況（`adv_report.py:74-85`）。
  若把 Stage C 的設定加進 `configs/pins.toml`，Phase 1 的 Advanced 報告就再也無法用目前的 repo 重跑。
- 因此 Stage C 的所有設定（SLO、guard band、n、上限、逾時參數、sentinel、`c0_branch`、`[freeze] capacity`）放在獨立的 `configs/capacity.toml`。
  `configs/pins.toml` 維持 Advanced 凍結時的位元組（sha256 `232f0de9…`），由 `test_frozen_untouched` 保護。
- Stage C 與 capserve 仍以唯讀方式讀取 `pins.toml` 的 Core B 與 Stage B 的值。
- 這只是設定檔的位置，不改變任何規則。

## 實作補記二：harness 的細節（2026-10-16，凍結前；沒有任何 C1 數據）

harness 依本文以 TDD 實作：
- `capacity_plan` / `capacity_analysis` / `run_capacity` / `capacity_report` 與 `configs/capacity.toml`；
- 三個面向審查，共 21 項發現：17 項修正，2 項維持原行為並以測試鎖定，1 項只補測試，1 項由我在下面第 3 點處理。

本文沒有寫死、在實作時決定的細節如下，凍結時一併凍結。

1. **跳過規則的「有效的 FAIL」**：
   - 指單一 run 的 maxR_run > 1（只看短類與 TPOT）。
   - mixed 的長類合併計算，只在最後對該點的所有 run 判定一次，不觸發跳過；這樣不會因為反覆檢查合併值而產生偏差。
   - STRADDLE、「n 次中幾次符合」與每個 run 的結論，也都只看單一 run 的 FAIL。
2. **Ĉ_g 的 band 只看 R_short 與 R_TPOT**（8.4 第 3 步的原文）。mixed 時 Ĉ_g 可能大於 Ĉ。審查確認確認集合 K 不受影響。
3. **C_PROD 在所有狀態列都把 MARGINAL 當成 FAIL**：
   - 原文只在 EXACT 那一列寫明這個替換；INFEASIBLE 與 UNSTABLE 照字面只看 FAIL，所以 K 的最低點是 MARGINAL 時，C_PROD 會變成 NT。
   - 現在一致處理：C 1 只達到 MARGINAL 時，C_PROD 為 INFEASIBLE（值 0，profile 中代表拒絕這個 class）；最低點高於 1 時為 UNSTABLE。
   - 理由：NT 在 profile 中代表「無法提供」，而這種情況其實已經證明在 guard band 內無法安全服務，「驗證過的 0」比較保守，也比較一致。
4. **H-S1 照表判定**：mixed 為 CENSORED 時，不列為 refuted（表中只有 EXACT 或 LOWER），而是 inconclusive。
   H-S2 補上原文沒寫的兩格：EXACT 或 INFEASIBLE 且 ≤ C_KV 為 supported；LOWER 且 ≤ C_KV 為 inconclusive。
5. **C0 的共變量**：分支 B / C 的每 run 滑移數只對 uniform workload 計算。
   凍結的 Stage C 模組不能 import `c0_analysis`，所以在 `capacity_analysis` 中重寫了 028 的滑移判定，並以真實資料重現 028 的 B0 0/2/0、G 1/0/0、K3 2/1/2。
6. **階段 S 的順序**：
   - workload 依固定順序：U2K、U8K、M2K、M8K。
   - 第一個 bench 之前就失敗的 boot，在階段 S 結束時重試一次。
   - 兩次都失敗時，K 為空，該 workload 的估計量為 NT。
   - server 在 boot 檢查通過後、第一個 bench 之前就退出，也視為「第一個 bench 之前失敗」，不記成 C 1 失敗。
7. **同一 boot 內的順序**：先把完整的 sorted(K) 以 seed 洗牌，寫進確認計畫；之後被跳過的點只是從中濾掉。
8. **失敗的 boot**：階段 C 或 sentinel 的 boot 失敗時，計畫中的每個 run 都記為無效（觸發跳過規則）。開頭的 sentinel 無效時中止整個 session，因為 8.6 的閘門要求兩次 sentinel 都有效。
9. **旗標與描述的定義**：
   - WINDOW_SENSITIVE：C_SLO 或 C_PROD 的（值、狀態）在 W′ 下不同。
   - binding：K 中最小的 FAIL 點上，超過 1 的項目。
   - ABOVE_CKV：mixed 也用 C_KV(L)。
   - analysis.json 另外記錄每個 run 與每個點的 running + waiting 峰值及是否 ≤ C，以及 C_SLO 與 C_knee（25 / 8）的比較。
10. **比較一律精確**（Fraction）：sentinel 的 3% 含邊界；逾時剛好是 60 的倍數時不會因浮點誤差多加 60 s。
11. **CROSS_SESSION_SHIFT 的參考值**由 report 從已 commit 的資料重算（Core 314.94 tok/s、Advanced B0 穩態 P95 996.0 ms），不寫進設定檔。
12. **mini smoke**：
    - 內容：U2K 與 M2K、U2K 上限 6、n_near 2、n_low 1、sentinel 在 C 2、假的 SLO 門檻、U2K C 6 強制 5 s 逾時、N = 20。
    - 這樣可以走過夾擠、二分與 c_low（搜尋為 1, 2, 4, 6，再 5；K = {3, 4, 5, 6}），並觸發逾時與跳過規則。
    - N = 20 是在執行期間暫時替換 `core_b_plan.num_prompts` 的回傳值做到的（只在 `--mini`；檔案位元組不變，結束後一定還原）；plan 的檢查在替換之前用 Core 的 N 先做。
    - 跑完後丟棄所有延遲數值。
13. **FS-order 的判讀**：最晚到達的 C 個 request 中，序號不在 [N − C, N) 的不超過 3 個；使用 B0 的 argv，每個 bench 有自己的 request-id 前綴。
14. **formal session 在開始前就拒絕 report 會拒絕的 server env**（例如繼承了 OTEL_* 變數），避免白跑一整個 session；runner 與 report 共用同一份基準 env。
15. **report 只寫 `<session>/analysis.json`**，其他輸出路徑一律拒絕。
16. **`run_core_a.py` 也納入 `test_frozen_untouched`**，因為四個 Stage C 模組都 import 它（plan 與確認計畫的 sha256）。另有測試確認 Stage C 依賴的每個模組都有位元組保護。
17. **凍結需要證據**（`configs/capacity.toml` 的 `[freeze]`）：
    - 必填：`fs_order_smoke`、`mini_smoke`（兩次通過的 smoke 的 session id）、`c0_decision`（C0 結果的 decision log 編號）。
    - `capacity_problems` 檢查格式；`freeze_evidence_problems` 確認兩個 smoke session 存在、mode 正確、formal 為 false、smoke.ok 為 true，而且 C0 的 decision log 存在。
    - formal 的 runner 與 report 在證據不完整時都會拒絕。

**照實記錄的限制**
- WSL2 的 `nvidia-smi --query-compute-apps` 通常列不出 process，所以「沒有其他 compute process」這項檢查的實際防線是 memory.used ≤ 1,000 MiB（C0 已經因此停過一次）。
- 凍結的 `run_adv.start_server` 不會把 lock 的 fd 傳給 vLLM。runner 若被 SIGKILL，lock 會釋放，但 vLLM 可能還在執行。這是 Phase 1 helper 的既有行為，沿用不改。
- 正式 session 命名為 `formal-YYYYMMDD-c1`，因為 029 的產生器只接受 `formal-` 開頭的 session。

## 實作補記三：decision 031 的閒置判定與 spill 偵測（2026-10-16，凍結前；沒有任何 C1 數據）

- 每次 boot 前改用 031 的綜合判定（`benchmarks/gpu_idle.py`，最多 3 個 30 s 的窗），不通過時中止 session，和舊規則一樣。
- **新增 run 的有效條件 (g)「沒有 spill」**：
  - Windows 的 shared usage 在 boot 前、server 就緒後與每個 bench 之後各讀一次。任何讀數比 boot 前多出 64 MiB 以上（含 64）即為 spill。〔2026-10-16 依 031 修正案（使用者確認）改為兩段：就緒時比 boot 前多出超過 160 MiB 即 spill（正常 boot 固定多 80 MiB）；之後每個 bench 比就緒時的讀數多出 64 MiB 以上即 spill。〕
  - spill 的 run 無效。同一 boot 中之後計畫的 run 記為無效 row，不跑 bench，處理方式和 server 結束相同。
  - 讀不到時中止 session（fail closed）。
  - `capacity_report` 由 row 的讀數重算 (g)，不一致時拒絕產生報告。
- **階段 S 的 boot 在 server 就緒時就 spill**：
  - 視為 8.4 第 4 步的「第一個 bench 之前失敗」，在階段 S 結束時重試一次。
  - 理由：階段 S 只用來選點，它的數據不進入估計量；031 的「不補跑」針對的是正式量測的無效 run。階段 C 與 sentinel 的 run 仍一律不補跑。
- **階段 S 的 bench spill 之後**，同一 workload 剩下的搜尋步驟都算失敗，不跑 bench。這會讓 Ĉ 偏低，是保守的方向。
- 凍結的內容加入 `benchmarks/gpu_idle.py` 與 `tests/test_gpu_idle.py`；它也列入 Stage C 依賴模組的位元組保護。

## 8.0 揭露（只用已 commit 的資料，沒有新的 GPU run）

設計者看過所有 Core 與 Advanced 的結果。Core 的曲線只用在下列四處，SLO 的數值不取自 Core：逾時、預算、8.5 的事前預測、8.2 的「條件說明」。

- **D1. 開場湧入**：
  - `bench_argv` 使用 `--request-rate inf --max-concurrency C`，N = max(10C, 200)。開頭的 C 個 request 會一起送進閒置的 server，TTFT 呈階梯狀。
  - 2K uniform、C 25（Advanced B0 / G / K3 共 9 個 run）：
    - 全部 request 的短類 P95 為 4,258–4,262 ms；
    - 去掉 idx < C 與 idx ≥ N − C 後為 996–997 ms；
    - 開場最大值約 8.0 s。
  - mixed 幾乎不受影響：2K 為 23.2 對 23.3 s，8K 為 7.78 對 7.67 s。
  - 024 曾以「事後」為由，拒絕為 H-B4 使用穩態的 P95。Stage C 是新的估計量，在任何 C1 數據之前凍結，而且全部 request 的統計量一律並列報告。Phase 1 的判定都不變（027 §1）。
- **D2. 8K 的「雙模式」是每個 bench 中的滑移**（028 的線索）：
  - 全部 request 的 P95 偏高的 run，正好是有滑移的 run，因為開場已經貢獻了 7 個高值。
  - 穩態時間窗內的 P95 只有 B0 r2 偏高，因為只有它在時間窗內有兩次滑移。
  - Core r1 在 C 8 為 2,826 ms、在 C 9 為 2,502 ms，所以滑移發生在每個 bench，目前沒有 boot 層級的證據。
  - 這是探索性的觀察，由 C0 判定。
- **D3. request 序號等於送出順序**：60 個 Advanced run 中，前 C 個序號在 60 個 run 中都是最先到達 server 的 C 個；最後 C 個序號在 58 個 run 中如此。開場的到達時間差最多 10.3 ms。
- **D4. noise**：
  - Core 中所有 P95 TTFT 或 TPOT 的格子，restart 之間的最大差距為 2.06%（CV ≤ 1.06%）。
  - 唯一的例外是 8K C 9，差距 14.4%。

## 8.1 研究問題與估計量

**問題**：對 workload w 與使用者選定的 SLO S，在穩態下，已准入的 request 仍符合 S 的最大封閉迴路並發數 C 是多少？
- C 就是 `--max-concurrency`，也就是 gateway 的准入上限。
- TTFT 從 request 送出起算，不含 client 的 semaphore 等待（PREREGISTRATION §3），所以是准入之後的延遲。

**部署：Core B 的 server，原樣不變**
- `core_b_plan.server_argv` 加上 Stage B 的值：K 4 GiB（248 個 528 的 block）、max-num-seqs 128、MNBT 2048、U 0.91、triton GDN、prefix caching 關閉、pin memory 0。
- 不加觀測 flag。
- Client：原樣 import `core_b_plan.bench_argv`。

**每個 run（在某個 C 跑一次 bench）**
- **時間窗**（使用者選定）：穩態時間窗 W = 序號在 [C, N − C) 的 request（|W| = N − 2C ≥ 160）。
  Core 的做法（全部 request）只並列報告。
- Y_short：W 中短類的 P95 TTFT（uniform 時就是全部 request）。
- Y_TPOT：W 中全部 request 的 (latency − ttft) / (out − 1) 的 P95。這是新的統計量（run_adv 只算短類）。
- Y_long（只有 mixed）：W 中 32,640-token request 的 P95 TTFT，把該 C 所有有效 restart 合併計算，因為每個 run 只有約 18–20 個。
- 百分位數都用 `core_b_analysis.percentile`（線性內插）。
- R_j = Y_j / S_j；maxR_run = max(R_short, R_TPOT)。

**每個 C 的分類**（依該點要求的 n，見 8.4；FAIL 優先）

| 類別 | 規則 |
|---|---|
| FAIL | 任一有效 run 的 maxR_run > 1，或合併的 R_long > 1 |
| NT | 沒有有效的 FAIL，而且有效 run 少於 n |
| PASS | 有 n 個有效 run，每個 R ≤ 1 − m |
| MARGINAL | 其他 |

- guard band m = 0.05，約為 D4 差距的 2.4 倍，用來涵蓋無法以其他方式估計的日間漂移。

**估計量**（在確認集合 K(w) 上，由小到大）
- **C_SLO（研究）**：K 中最大的 c，使得 K 中每個 c′ ≤ c 都是 PASS 或 MARGINAL。
- **C_PROD（產品上限）**：同上，但只接受 PASS。只有 C_PROD 能寫進 verified profile。
- 向下封閉：准入上限必須在每個更低的負載下也成立，所以失敗點之上的通過一律不算。
- 未測的點之間假設單調。Core 有一個反例：32K 全部 request 的 P95 在 C 2 為 9.97–10.02 s，在 C 3 為 9.71 s。

**狀態**

| 狀態 | 意義 |
|---|---|
| EXACT | c + 1 在 K 中而且是 FAIL（C_PROD 時為 FAIL 或 MARGINAL） |
| LOWER | c + 1 沒有測、是 NT 或被跳過，所以只是下界 |
| CENSORED | c = C_cap |
| INFEASIBLE | C 1 就 FAIL，值為 0 |
| UNSTABLE | K 的最低點大於 1 而且 FAIL |
| NT | 其他無法檢定的情況 |

**旗標**

| 旗標 | 條件 |
|---|---|
| NONMONOTONE | 失敗點之上有通過的點 |
| STRADDLE | 同一個 C 的有效 run 結論不一致 |
| STRICT_COARSE | C_PROD < C_SLO − 1 |
| SEARCH_MISMATCH | \|C_SLO − Ĉ\| > 1 |
| WINDOW_SENSITIVE | 改用 W′ = [2C, N − C) 時判定不同 |
| ABOVE_CKV | C_PROD > C_KV。這時多出的 request 會在 vLLM 內排隊，gateway 無法讓它逾時 |

**只作描述，永遠不寫進 profile**
- C^full（Core 式、全部 request 的統計量）；
- 逐 request 的聯合達成率；
- 合併的 P95；
- 開場湧入的最大值；
- 是哪一項 SLO 先失守；
- KV 峰值、preemption、waiting 峰值。

**對雙模式的穩健性**：不依賴雙模式的成因，來自四條規則：
- 取最差的 run；
- 關鍵點用 n_near = 5 次 boot；
- 向下封閉；
- guard band。

不依模式排除任何 run，因為排除會讓容量偏高。若某個壞模式在單一 run 出現的機率為 q，n = 5 時 q ≥ 0.45 才能以 ≥ 95% 的機率抓到；n = 7 時為 q ≥ 0.35。較罕見的模式照實報告為「n 次中有幾次符合」。

**C0 的分支**（028）：

| 分支 | C0 的結果 | 協定的變更 |
|---|---|---|
| A | 只有每個 bench 的滑移 | 預設 |
| B | 有 boot 層級的成分 | 受影響的 workload（至少 uniform-8K）n_near = 7；C0 的每 run 標記記為共變量 |
| C | 無法判定，或達到上限而停止 | 同 B，只套用在 uniform-8K |
| D | 發現 harness 或 client 的缺陷 | 先寫 decision log 修正，§8 在那之前不凍結 |

任何分支都不改變 S、時間窗、統計量、m、上限、搜尋、確認規則、workload 或 hypothesis，也不減少 n。

## 8.2 SLO（使用者選定 S2；下表保留三個選項作為紀錄）

數值放在 `configs/capacity.toml` 的 `[slo]`：
- `id`；
- 2048 / 8192 / 32768 各自的 `ttft_p95_ms`；
- `tpot_p95_ms`；
- `mixed_long`：mixed 中長 request 的處理方式，"budget"（要符合 32K 的門檻）或 "best_effort"（不檢查）。

套用方式：uniform-L 用 L 的門檻；mixed-L 的短 request 用 L 的門檻、長 request 用 32K 的門檻；TPOT 套用在所有 request。

**依據**
- Nielsen：1 s 維持使用者的思緒；10 s 是注意力的上限，超過時要顯示進度。
- 2 s 是可容忍的等待（Nah 2004）。
- 默讀約 238 wpm（Brysbaert 2019），約每秒 5 個 token。所以 TPOT 200 ms 約等於閱讀速度，100 ms 約為 2 倍，50 ms 約為 4 倍。
- MLPerf（p99）：Llama-2-70B conversational 2 s / 200 ms，interactive 450 ms / 40 ms；Llama-3.1-405B long-context 6 s / 175 ms。
  這些來自二手來源，凍結前要以 MLCommons 的規則核對。
  〔2026-10-17 凍結前核對 MLCommons `inference_policies` 的 `inference_rules.adoc`：
  - Llama-2-70B：conversational 2,000 ms / 200 ms，interactive 450 ms / 40 ms，與上文相同。
  - Llama-3.1-405B：6,000 ms / 175 ms，與上文相同；規則中的類別名稱是 Server，不是上文寫的 long-context。另有 interactive 4,500 ms / 80 ms。
  - 百分位數：規則的情境表中 Server 的 tail latency 為 99%；LLM 的列本身沒有另寫百分位數。
  - 這些數字只是依據，不影響已選定的門檻。〕
- 輸入越長，TTFT 的門檻越寬，因為提交越多內容的使用者會預期等更久。8K 的門檻取 2K 與 32K 的幾何中點（2K 門檻的 √10 倍）。

| id | TTFT 2K / 8K / 32K | TPOT | 情境 |
|---|---|---|---|
| S1 嚴格互動 | 1 / 3 / 10 s | 50 ms | 反應快的 chat 與文件問答 |
| S2 標準互動 | 2 / 6 / 20 s | 100 ms | 可容忍等待的 chat；有進度提示的文件處理 |
| S3 寬鬆 | 2 / 6 / 20 s | 200 ms | 不即時閱讀輸出的 API、agent、內部呼叫 |

**不使用**
- E2E：它等於 TTFT + 127 × TPOT。
- P99：每個時間窗中高於 P99 的只有約 2–10 個 request。
- 8K 或 32K 的 TTFT ≤ 1 s：低於 C 1 的物理下限（1.40 s 與 6.29 s）。

**條件說明**（取自 Core，予以揭露；不是選擇的依據；選定後門檻永遠不再移動）
- S1：
  - 2K 的 TTFT 門檻落在 2K 的平台上（977–997 ms）。
  - C_SLO(U2K) 約 15，受 TPOT 限制（C 16 時 50.8–51.3 ms），但 C_PROD 只有約 2–3，預期會出現 STRICT_COARSE。
  - U8K 約 4，mixed 約 1。
- S2：U2K 約 32–34；U8K 為 8，C 9 恰在門檻邊緣（TPOT 100.8–101.7 ms）；mixed 約 1，因為短 request 會排在 6.3 s 的 32K prefill 之後。
- S3：TPOT 永遠不會先失守（Core 最大 147 ms）。U2K 約 36–38，U8K 約 14–15，mixed 約 1–2。
- 沒有選中的 SLO 會用同一份數據作為描述性的界線報告。

## 8.3 Workload

- **預設**：U2K、U8K、M2K、M8K。
  - Uniform：有 seed 的隨機 prompt，range-ratio 0。
  - Mixed：007 的 mix（90% 短、10% (32640, 128)），使用 pin 過 sha256 的檔案，`--disable-shuffle`。
- **選擇理由**：
  - 涵蓋 chat 與 RAG 兩類。
  - 部署若要接受 32K 的 prompt，mixed 是取得 verified 上限的唯一途徑。
  - mixed 也是 Phase 1 效果最大的地方（2.8–5.4 倍）。
- **Client**：每個 bench（搜尋與確認都一樣）都用 N = max(10C, 200)。
  - 因此在 Core 格點上，uniform 的 prompt 和 Core 完全相同。
  - `run_core_b.summarize_result` 原樣套用。
- **選項 O-32K**（不做，使用者沒有選）：加入 uniform-32K，上限 6，約多 3–8 h。

## 8.4 搜尋與確認（決定性，單一 session）

**上限**：C_cap(L) = min(128, 2·C_KV(L))，2K 為 70、8K 為 26，只由 block 算術推得。mixed 用同一個 L 的上限。

**階段 S（只用來選點）**：每個 workload 一次 boot，bench 由小到大。
- run 通過的條件：有效、maxR_run ≤ 1，而且（mixed 時）該 run 的 R_long ≤ 1。無效的 run 在導航時視為失敗，並標記。
- 程序：
  1. **夾擠**：依序跑 c = 1, 2, 4, …，最後跑 C_cap，直到第一個失敗點 F；P 是最後一個通過點。
  2. **二分**：當 F − P > 1，跑 mid = ⌊(P + F)/2⌋；通過則 P = mid，失敗則 F = mid。最後 Ĉ = P（C 1 失敗時為 0）。
  3. **guard band 點**：Ĉ_g 是最大的已測 c，使得每個已測的 c′ ≤ c 都有 maxR ≤ 1 − m。
  4. 在第一個 bench 之前就失敗的 boot，於階段 S 結束時重試一次。這時還沒有看到任何數據。

**確認集合**：由階段 S 的 rows 決定的純函數。進入階段 C 之前，先記錄它的 sha256。
- Ĉ = 0：K = {1}。
- 否則：K = {Ĉ − 1, Ĉ, Ĉ + 1} ∪ {c_low}，限制在 [1, C_cap] 之內。
  - 若 1 ≤ Ĉ_g < Ĉ − 1，c_low = Ĉ_g；否則 c_low = ⌈Ĉ/2⌉，若 c_low ≥ Ĉ − 1 則不加入。
- restart 數：Ĉ − 1、Ĉ、Ĉ + 1 以及 K = {1} 時，n = n_near（5）；c_low 為 n = 3。

**階段 C（估計量只使用這一階段的數據）**
- 第 r 個 block 中，每個 workload 一次新的 boot；workload 的順序以 `Random(int(sha256("20261008:capacity:block:r"), 16))` 決定。
- 同一次 boot 內，各點的順序以 "20261008:capacity:{w}:{r}" 為 seed。
- **跳過規則**：
  - 某點一旦出現有效的 FAIL，或出現無效的 run（不補跑，013），之後的 block 就不再測它和所有更高的點。
  - 沒有任何待測點的 boot 直接跳過。
- 搜尋階段的 run 永遠不進入估計量，避免 winner's curse。

**Sentinel**：session 開頭與結尾各一次 boot，跑 U2K、C 25（Core 315.0 tok/s；Advanced 穩態 996 ms）。

**Session**
- 順序：sentinel → 階段 S → 確認計畫 → 階段 C → sentinel。
- 全程在同一次 `flock -n /tmp/ai-portfolio-gpu.lock` 之下，以單一 ssh 前景指令執行，使用同一個 DEPLOYED_COMMIT。
- 期間不部署，也不跑任何產品測試。
- session 不接續；中止的 session 以新的 id 重跑。

**逾時**
- 每個 bench：ceil60(2·N·τ1 + 60 s)，τ1 為 Core 在 C 1 時每個 request 的時間：2K 1.845 s、8K 2.95 s、32K 7.92 s。
  mixed 用 0.9·τ1(L) + 0.1·τ1(32K)。這是全部串行時的上界。
- server 就緒：1,800 s。
- 逾時的 run 無效，並終止它的 process group。

**預算**（使用者接受一次連續占用，GPU 空著就跑）
- 估算方式：Core 的 bench 時間，加上每個 bench 55 s、每次 boot 90 s。

  | 情境 | n_near 5 | n_near 7 |
  |---|---|---|
  | S1 | 7.1 h | 8.7 h |
  | S2 | 7.5 h | 9.0 h |
  | S3 | 8.8 h | 10.7 h |
  | 高容量壓力情境 | 9.7 h | 12.1 h |

- smoke 另加 0.5 h。
- 比較：Core B uniform 用了 20.0 h，Advanced 用了 6.1 h。

## 8.5 Hypotheses（規則判定，不用 p 值；只凍結選定 SLO 的預測）

**H-S1（mixed 流量降低容量）**：L ∈ {2K, 8K} 時，C_SLO(M_L) < C_SLO(U_L)。

| 判定 | 條件 |
|---|---|
| supported | mixed 的值（EXACT，或 INFEASIBLE 時為 0）小於 uniform 的值（EXACT、LOWER 或 CENSORED） |
| refuted | mixed 的值（EXACT 或 LOWER）≥ uniform 的 EXACT 值 |
| inconclusive | 兩者的界線重疊 |
| NT | 任一邊為 NT 或 UNSTABLE，或 uniform 為 INFEASIBLE |

預測：S1、S2、S3 下都 supported（H-B4、025 R1、D1）。

**H-S2（KV 容量是 SLO 容量的上界）**：U2K 與 U8K 的 C_SLO ≤ C_KV（35 與 13）。
- refuted：EXACT 值或下界超過 C_KV，或結果為 CENSORED。
- 對產品的意義：只用 block 算術，是否就能作為 provisional 上限的安全天花板。
- 預測：S1、S2 下 supported，S3 下 refuted。

**只作描述**：SEARCH_MISMATCH（只做一次負載測試夠不夠）、C^full、達成率、哪一項先失守、C_PROD 時的開場湧入，以及 C_SLO 與 C_knee（25 / 8）的比較。

## 8.6 有效性、環境與檢查

**run 有效**的條件：
- (a) `summarize_result` ok；
- (b) server 就緒，log 中有 FLASH_ATTN；
- (c) `run_adv.parse_nondefault` 顯示 MNBT 2048 與 kv_cache_memory_bytes = K，沒有觀測、T 或 G 的 key；
- (d) /metrics 顯示 248 個 528 的 block；
- (e) ttfts、latencies、input_lens、output_lens 的長度都是 N；mixed 時 W 中至少有一個長 request；
- (f) 沒有逾時。

**session 有效**的條件：
- formal，而且 `freeze.capacity = true`；
- session 開始時與每次 boot 前，`held_gpu_lock` 都為真；
- 每次 boot 前 memory.used ≤ 1,000 MiB；〔2026-10-16 起由 decision 031 的綜合判定取代（lock、使用率 ≤ 5%、基線穩定、推導的 VRAM 餘裕），並新增 run 的有效性條件：boot 期間 Windows 的 shared usage 沒有比 boot 前多出 64 MiB 以上（沒有 spill）。使用者確認，凍結與任何 C1 數據之前。〕〔spill 的判定再依 031 修正案改為兩段（就緒時 > 160 MiB、之後每個 bench 相對就緒時 ≥ 64 MiB），使用者 2026-10-16 確認。〕
- 每次 boot 前後的 env 擷取都乾淨：power limit 為預設、commit 單一且乾淨並等於 DEPLOYED_COMMIT；
- vLLM 0.30.0；
- `stage_b_problems` 為空；
- pins、plan 與 dataset 的 sha256 都相符；
- 沒有繼承任何未列出的 engine 變數；
- **sentinel 閘門**：開頭與結尾的 output throughput 與 W 平均 TTFT 相差都 ≤ 3%（Core A 的規則）。
  不用 P95，因為 P95 會在滑移上翻轉。
- **replay**：報告程式從 rows 重算所有搜尋決策、K、n、跳過與順序；任何不一致都讓 session 無效。

**只作描述的檢查**
- CROSS_SESSION_SHIFT：sentinel 和 Core 或 Advanced 相差 > 5%。
- 1 Hz 的 MetricsSampler：running + waiting ≤ C、KV 峰值、preemption。

**凍結前的 smoke**（GPU lock 下，非正式，延遲數值丟棄）
- FS-order：一次 boot 加上 024 的觀測 flag，跑 U2K 的 C 2 與 C 70。前 C 個的集合必須相同，最後 C 個最多差 3 個，否則 §8 不凍結。
- 端到端的迷你 session，使用一個假的 SLO，涵蓋狀態機、逾時、跳過規則、replay 與 derived 檔的格式。

**凍結**：commit 並 push 下列內容，再設定 `freeze.capacity = true`：
- §8 與本文；
- `configs/capacity.toml`；
- 模組 `capacity_plan`、`run_capacity`、`capacity_analysis`、`capacity_report` 與它們的測試。
  這些模組只 import Core 與 Advanced 的程式，由 `test_core_untouched` 與 `test_frozen_untouched` 把關。

## 8.7 輸出與 verified profile（027 §4、029）

**檔案**：`results/capacity/<session>/`
- session.json、plan.json、confirm_plan.json、rows.jsonl、env/、raw_manifest.json；
- `derived/<tag>-requests.jsonl`：欄位 idx、cls、ttft_ms、latency_ms、out，不含 prompt 文字。
- 所以每個數字都能從 git 重算。

**analysis.json**（schema_version 1）
- provenance：commit、pins / plan / confirm_plan 的 sha256、c0_branch、n_near；
- platform：GPU、driver、vLLM、model@revision、server_argv_sha256（去掉 host / port）、env、K、U、block；
- slo；
- sentinel；
- 每個 workload：
  - c_kv、c_cap；
  - search、Ĉ、Ĉ_g；
  - K 與各點：類別、各 run（Y 值、全部 request 的百分位數、達成率、開場最大值、throughput、KV、preemption）、合併值；
  - c_slo 與 c_prod，各附值與狀態；
  - 先失守的項目、餘裕、旗標；
- secondary_slo；
- hypotheses；
- not_supported。

**報告拒絕執行**（exit 2）的情況：
- session 不是 formal、env 無效、已中止、有漂移，或 replay 不一致；
- 沒有凍結；
- 任何 sha256 不符；
- SLO 和 pins 不同。

**產生器**：029 的 `capserve.profile_gen`（M2 實作見 029 的 M2 補記：class 名稱、CENSORED 的 bound 為 `censored`、`--freeze-commit` 的 git 驗證），對每個 workload：

| 結果 | profile 的內容 |
|---|---|
| EXACT / LOWER / CENSORED | verified，max_concurrency = C_PROD，並標明是精確值或下界 |
| INFEASIBLE | verified，值為 0，代表「拒絕這個 class」 |
| UNSTABLE / NT | unavailable，附原因；永遠不給數字，也不退回其他數字 |

每個項目都附上：
- 適用範圍：輸入 L − 128 或 32,640 token、輸出 ≤ 128、封閉迴路穩態、准入之後的延遲；
- C_SLO、C_KV、ABOVE_CKV；
- 先失守的項目、餘裕、旗標、開場湧入；
- provenance。

## 8.8 Advanced R1（T = 512）

- **不在預設範圍內**：027 §2 固定使用 Core B 的 argv，T 會產生不同的 fingerprint，也就是另一份 profile。
- **選項 O-T**（不做，使用者沒有選）：
  - 對 U8K 與 M8K 用同一協定測 `[advanced.conditions].T`，約多 3–4 h。
  - 測出的 T profile 只能用於明確宣告為 M8K 的部署，而且要 C_PROD(M8K) 大於 B0 的值，同時要報告 T 下的 U8K。

## 8.9 風險與限制

- **合成的負載**：封閉迴路、固定長度、128 個 output token、ignore-eos。
  - 這會讓 request 鎖在 TTFT 的格狀結構上，使 P95 在格線附近很脆弱。
  - open-loop 或長度不一的流量會有不同的尾端，所以 profile 只適用於它寫明的範圍。
- **排除開場湧入**：穩態時間窗不含冷啟動時的湧入，只以附註報告。
- 未測的點之間假設單調。
- 機率 q < 0.45 的罕見模式可能漏抓。
- **時間上的混淆**：搜尋在 session 前段，確認在後段。
- guard band 沒有針對長時間的漂移校準。
- 只有一個平台：WSL2 / HAGS 上的一張 GPU、一個模型、一個 vLLM 版本，K 固定。
- **下列情況下無法解讀**：
  - FS-order 失敗、sentinel 漂移、replay 不一致；
  - argv、block 數或 vLLM 版本不符；
  - session 期間有其他 GPU process 或部署；
  - 凍結時沒有 c0_branch；
  - 第一個 C1 bench 之後任何規則改變（需要 post-hoc decision log 與新的 session）；
  - 所有接近邊界的點都是 NT。

## 凍結前的決定

使用者的選擇見上方「使用者決定」。時間窗的選擇是在看過開場湧入的數據後提出的，已揭露。

以下由我依預設值處理，使用者沒有異議：
- guard band m = 5%；
- n_near = 5，C0 落在分支 B 或 C 時改為 7；
- C_cap = 2·C_KV；
- 開場湧入的事實只在 §8 揭露，不另外為 Phase 1 寫說明，也不重新分析 Phase 1。

## 凍結紀錄（2026-10-17）

- **凍結證據**（`configs/capacity.toml` 的 `[freeze]`）：
  - FS-order smoke `smoke-fs-order-202610161531`；
  - mini smoke `smoke-mini-202610161531`；
  - C0 的結果 decision 032：分支 A。
  - 三者都已 commit（`f35fe85`），`freeze_evidence_problems` 為空。
- **凍結的程式**：`capacity_plan`、`capacity_analysis`、`run_capacity`、`capacity_report`、`gpu_idle`。
  它們和兩次 smoke 所部署的 `f0eee6a` 位元組相同。
- **新增 `tests/test_stage_c_frozen.py`**：以 sha256 固定這五個模組與凍結後的 `configs/capacity.toml`。
  - 用途：在花 7–9 h 跑正式 session 之前，就抓到凍結後的任何修改。
  - 凍結後要改任何規則，都需要新的 decision log 與新的 session（8.9）。
  - capserve 的 profile 產生器也會用 git 再核對一次。
- **測試的調整**：原本以「repo 的設定尚未凍結」為前提的測試，改為在測試中明確建立未凍結的設定（`draft`）。只改測試，沒有改任何規則或程式。
- 下一步：部署凍結 commit，確認 GPU 空閒與主機負載後，在 GPU lock 下執行正式 session `formal-YYYYMMDD-c1`。
- **結果**（凍結後註記）：`formal-20261017-c1` 已完成，見 decision 033。凍結的規則沒有改變。
- **最終結果**（凍結後註記，2026-10-18）：依 034 重跑的 `formal-20261017-c1b` 有效，取代 033，見 decision 035。凍結的規則沒有改變。
