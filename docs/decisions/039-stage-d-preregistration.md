# 039 — Stage D 事先登記（DRAFT）：容量的延伸

- 狀態：**DRAFT**。harness、smoke 與審查完成後凍結；凍結前不收任何 Stage D 的正式數據。
- 日期：2026-10-23
- 對應：037（Phase 3 範圍）、030（Stage C 的規則，本文沿用）、035（c1b 的結果）、036（適用條件中「不涵蓋」的項目）、038（capserve v1）
- 時點：任何 Stage D 數據之前。設計者看過 Core、Advanced、Stage C 的全部結果，用到的地方在「揭露」一節寫明。
- 凍結的方式比照 030：
  - 一起 commit 並 push：PREREGISTRATION §9、本文、`configs/stage_d.toml`、harness 與測試；
  - 再設定 `[freeze] stage_d = true`；
  - 之後才收集任何正式數據。

## 範圍

Stage D 回答 037 列出的容量問題，分成三部分。開場湧入另外以既有數據描述。

| 部分 | 問題 | 方法 | session |
|---|---|---|---|
| D-A | 其他 workload 與 server 設定下的容量：32K（O-32K）、T = 512（O-T）、輸出 512 token | 封閉迴路，沿用 030 的協定 | D-A1、D-A2 |
| D-B | 開放迴路（Poisson 到達）下的容量 | 依到達率搜尋 | D-B |
| D-C | 正式容量在兩個未驗證的條件下是否仍成立：class 內 prompt 長度不一、經過 gateway | 固定點驗證 | D-C |
| D5 | 開場湧入 | 只用 c1b 已 commit 的數據作描述，不用 GPU | — |

**沿用 030 的規則**（下文只寫差異）：
- 估計量 C_SLO 與 C_PROD，類別 FAIL / NT / PASS / MARGINAL；
- 狀態與旗標；
- guard band m = 0.05；n_near = 5，n_low = 3；
- 穩態時間窗 W = [C, N − C)，N = max(200, 10C)，warmup = C；
- 有效性：env、argv、block 數、replay、031 的閒置判定與兩段式 spill 偵測；
- 跳過規則；無效的 run 不補跑；
- sentinel 閘門（3%）與 CROSS_SESSION_SHIFT（5%）；
- 逾時 = ceil60(2 N τ1 + 60 s)；server 就緒逾時 1,800 s。

**SLO**：一律使用 S2（037 §3），短類 P95 TTFT 2 / 6 / 20 s（依 prompt 的長度等級），P95 TPOT 100 ms，mixed 的長 request 為 budget。
S1、S3 只作為描述性的界線報告，和 030 相同。

## D-A：封閉迴路的延伸（正式）

| workload | prompt | 輸出 | server | TTFT 門檻 | C_KV | C_cap |
|---|---|---|---|---|---|---|
| U32K | 32,640 | 128 | Core B | 20 s | 3 | 6 |
| U8K-T | 8,064 | 128 | Core B 加上 `--long-prefill-token-threshold 512` | 6 s | 13 | 26 |
| M8K-T | 007 的 8K mix | 128 | 同上 | 短類 6 s，長類 20 s | 13 | 26 |
| U2K-O512 | 1,920 | 512 | Core B | 2 s | 30 | 60 |
| U8K-O512 | 8,064 | 512 | Core B | 6 s | 12 | 24 |

- C_KV 以 `core_b_plan.c_kv(248, 528, prompt + 輸出)` 計算；C_cap = min(128, 2 C_KV)，和 030 相同。
- **T 的值**沿用 Advanced 的 condition T（pins `[advanced] long_prefill_token_threshold` = 512）。這會產生另一個 server fingerprint，所以它的容量只能寫進另一份 profile，只適用於宣告使用 T 的部署（030 8.8）。
- **輸出 512**：選 512，是因為 chat 回答常見的長度在幾百 token，比 128 更貼近產品。為了控制 GPU 時數，只選一個值。
  - TTFT 的門檻依 prompt 的長度等級（2K、8K），不依 prompt 加輸出的總長度。
  - TPOT 對所有 request 仍然是 100 ms。
- **τ1**（只用於逾時）：
  - U32K 為 7.92 s（030）；
  - T 的 workload 取 B0 的 2 倍：Advanced 中 T 讓 8K uniform 明顯變慢；
  - 輸出 512 取 τ1(L) × 4。
  - 這些都只會放寬逾時，不影響判定。
- **workload 的順序**：
  - D-A1：U32K、U8K-T、M8K-T；
  - D-A2：U2K-O512、U8K-O512。
  - 每個 session 開頭和結尾各有一次 sentinel（Core B 的 server、U2K、C 25，和 030 相同）。
- **預估時數**：D-A1 約 10 h（U32K 約 6 h），D-A2 約 10 h。兩者都在使用者 2026-10-15 允許的「7–9 h、最壞約 12 h」之內。

**Hypotheses**（只判定 S2；預測寫在數據之前）

| id | 內容 | 預測 | 依據 |
|---|---|---|---|
| H-D1 | C_SLO(U32K) ≤ C_KV(32K) = 3 | supported | 030 的 H-S2 延伸到 32K；Core 的 32K C_knee 為 2 |
| H-D2a | C_PROD(M8K-T) > C_PROD(M8K) = 2 | supported | Advanced 8K 為 R1：chunked prefill 壟斷 step budget，T 減輕短 request 的排隊 |
| H-D2b | C_PROD(U8K-T) < C_PROD(U8K) = 8 | supported | Advanced 中 T 讓 8K uniform 的短類 P95 TTFT 在 C 8 達到約 6.7 s，超過 6 s |
| H-D3 | 輸出 512 時 C_SLO ≤ C_KV(prompt + 512)（30 與 12） | supported | 030 的 H-S2：KV 容量是上限 |

- 判定的規則沿用 030 的 H-S2：EXACT 或 INFEASIBLE 而且不等式成立為 supported；LOWER 而且不等式成立為 inconclusive；不等式不成立為 not supported；NT 為 NT。
- H-D2a 與 H-D2b 比較的是 c1b 的 C_PROD（035），c1b 的值不重新估計。

## D-B：開放迴路（正式）

- **workload**：U2K-P 與 U8K-P。prompt 與輸出和 U2K、U8K 相同，server 為 Core B。
  mixed 不做，因為它的容量只有 2，Poisson 到達下的變異會被長 request 主導。這一點會寫進限制。
- **client**：
  - `vllm bench serve --request-rate λ --burstiness 1.0`（Poisson），不設 `--max-concurrency`；
  - N(λ) = max(300, ⌈300 s × λ⌉)，至少約 5 分鐘；warmup 10 個；
  - TTFT 從 request 送出起算。送出時刻就是排定的到達時刻，所以包含 vLLM 內部的排隊。
- **時間窗**：W = 序號在 [⌈0.1 N⌉, N − ⌈0.1 N⌉) 的 request。Y_short、Y_TPOT、R 與類別都和 030 相同。
- **參考到達率** λ_ref(w) = C_PROD(w) ÷ T̄(w)。
  - T̄ 是 c1b 中 C_PROD 點所有有效階段 C run 在 W 內 request 的平均 e2e 延遲。
  - 由 plan 從已 commit 的 c1b 逐 request 檔重算，不手寫。
  - 依 Little's law，這是平均並發等於 C_PROD 的到達率。
- **格點**：G(w) = λ_ref × {0.4, 0.55, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5}。
- **搜尋與確認**：
  - 階段 S 由小到大掃描 G，到第一個失敗點為止；λ̂ 是最後一個通過點。
  - 確認集合 K = {λ̂ 的前一點, λ̂, λ̂ 的後一點}（在 G 之內），n = 5；
  - 估計量 λ_SLO、λ_PROD 在 K 上依 030 的規則（向下封閉）；
  - 狀態：
    - EXACT：下一點為 FAIL（λ_PROD 時為 FAIL 或 MARGINAL）；
    - CENSORED：最高格點通過；
    - INFEASIBLE：最低格點失敗，代表 λ < 0.4 λ_ref；
    - 其他與 030 相同。
- **H-D4**：λ_PROD(w) < λ_ref(w)，預測 supported。
  - 理由：Poisson 的突發會讓瞬間並發超過平均值，而 C_PROD 在封閉迴路下已經緊貼 SLO。
  - 判定：EXACT 而且 λ_PROD ≤ 0.9 λ_ref 為 supported；λ_PROD ≥ λ_ref 為 not supported；其他為 inconclusive。
- **產品上的意義**：這是 vLLM 本身在 Poisson 到達下的容量。gateway 以 C_PROD 准入時，超過的 request 會在 gateway 排隊或回 429，不會進入 vLLM。所以 λ_PROD 不是 gateway 的上限，而是「不靠准入控制時」的到達率上限。
- **預估時數**：約 4–6 h。

## D-C：固定點驗證（正式）

在 C_PROD 跑 n = 5 次，分類規則和 030 的單一點相同。PASS 代表該條件下正式容量仍成立。

| 點 | 條件 | C |
|---|---|---|
| U2K-var | prompt 長度在 [64, 1,920] 均勻分布，輸出 128 | 32 |
| U8K-var | prompt 長度在 [64, 8,064] 均勻分布，輸出 128 | 8 |
| U2K-gw | U2K，經過 capserve gateway（正式設定，`tokenizer = "local"`） | 32 |
| U8K-gw | U8K，經過 gateway | 8 |

- **長度不一的 dataset**：
  - 和 007 的 mixed 相同，事先產生並 pin sha256；
  - 每個 prompt 的長度以 seed 決定，在範圍內均勻分布，並由 vLLM 自己的 RandomDataset 產生；
  - 每個 prompt 都必須重新 tokenize 成剛好目標長度，否則不寫出檔案（013）。
- **gateway**：
  - research runner 以子 process 啟動 `python -m capserve serve`，使用正式 profile 與 gateway 設定（port 8080 → vLLM 8000）；
  - research 的程式不 import capserve，維持 `test_research_isolation`；
  - client 的 TTFT 包含 gateway 的開銷。
  - vLLM 0.30.0 的 bench client 會送 `repetition_penalty: 1.0`。gateway 目前會拒絕這個欄位，所以要先把中性值 1.0 視為預設值並忽略。這屬於產品變更，在凍結前另外 commit 並測試。
- **H-D6**：U2K-var、U8K-var 為 PASS，預測 supported。理由：較短的 prompt 只會減少 prefill 與 KV。
- **H-D7**：U2K-gw、U8K-gw 為 PASS，預測 supported。理由：gateway 在 C_PROD 不會排隊，只多一次 loopback 往返。
- 判定：PASS 為 supported，FAIL 為 not supported，MARGINAL 為 inconclusive，NT 為 NT。
- **預估時數**：約 2.5 h。

## D5：開場湧入（描述，不用 GPU）

- 用 c1b 已 commit 的逐 request 檔，對每個 workload 與每個測過的 C，計算第一波（idx < C）的最大 TTFT。描述它和 C 的關係，例如以 C 的線性函數擬合，報告斜率與殘差。
- 這是事後的描述性分析，不是事先登記的檢定。
- 產品端的處理（准入時限制同時 prefill 的 token 數）屬於產品設計，等 D-A 的結果之後另寫 decision。

## 揭露

- **作為依據**：
  - Core 與 c1b 的結果用於 hypothesis 的預測、λ_ref 與格點，以及 τ1。
  - Advanced 的 T 結果用於 H-D2a / b 的預測。
- **不作為依據**：SLO 不是依數據挑選的，沿用使用者已選定的 S2。
- **c1b 的相關數字**：
  - C_PROD：U2K 32、U8K 8、M2K / M8K 2；
  - U8K 在 C 9 的最差 R_TPOT 為 1.006；
  - 開場湧入：uniform-2k 10.35 s、uniform-8k 11.04 s。

## 產品上的使用

- D-A 有效時，產生器會為每個 workload 產生 class，寫成新的 verified profile。T 有自己的一份 profile，因為 fingerprint 不同。
- **不改正式設定的釘選**：要不要服務新的 class，由使用者決定（037 §2）。
- D-B 與 D-C 不產生 profile，只寫進適用條件。

## 尚未決定（凍結前定案）

- `configs/stage_d.toml` 的格式。
- harness 的模組切分。凍結的 Stage C 模組只能 import，不能修改。
- smoke 的內容：mini（走過每一種 workload 與 gateway 路徑）與 FS-order（只對新的 client 設定）。

## 實作補記（2026-10-23，凍結前；沒有任何 Stage D 數據）

harness 依本文以 TDD 實作：`stage_d_plan`、`stage_d_analysis`、`run_stage_d`、`stage_d_report`、`stage_d_datasets` 與 `configs/stage_d.toml`。
凍結的 Stage C 模組只被 import 或繼承，沒有修改，`test_stage_c_frozen` 仍然通過。
本文沒有寫死、實作時決定的細節如下，凍結時一併凍結。

1. **重用的範圍**：
   - D-A 與 D-B 直接執行 `run_capacity` 的狀態機，以及 `capacity_plan` 的 `search_next`、`build_confirm_plan`、`phase_c_boots`；
   - 每個 run 的統計量、分類、估計量、旗標、sentinel 閘門與跨 session 偏移，都用 `capacity_analysis` 的凍結函式；
   - `StageDHarness` 繼承 `run_capacity._Harness`，只覆寫 server、client、結果檢查、逐 request 檔、tag 與 gateway。
2. **開放迴路的時間窗**：把 `run_eval` 的 C 設為 ⌈0.1 N⌉，它的穩態時間窗 [C, N − C) 就是本文的 [⌈0.1 N⌉, N − ⌈0.1 N⌉)。
   「開場湧入」欄位因此是前 10% 的 request，只作描述。
3. **H-D4 依格點的倍率判定**：
   - 格點的到達率是倍率 × λ_ref 再四捨五入到 4 位，所以 0.9 倍那一點的到達率可能略高於 0.9 λ_ref；
   - 改依格點的倍率判定：λ_PROD 的倍率 ≤ 0.9 而且 EXACT 或 INFEASIBLE 為 supported；倍率 ≥ 1 為 not supported；其他為 inconclusive。
   - INFEASIBLE（低於 0.4 λ_ref）也算 supported，因為它比 0.9 λ_ref 更低。
4. **H-D2a / b 的規則**（本文只寫了不等式）：
   - **大於**：值大於界線就是 supported，下界也算；EXACT 或 INFEASIBLE 而且不大於界線為 not supported；下界而且不大於界線為 inconclusive。
   - **小於**：EXACT 或 INFEASIBLE 而且小於界線為 supported；任何不小於界線的值為 not supported，下界也算；下界而且小於界線為 inconclusive。
   - NT 與 UNSTABLE 為 NT。
5. **H-D1 / H-D3** 使用 `capacity_analysis.h_s2`。它的 "refuted" 在 Stage D 寫成 "not supported"，和本文的用詞一致。
6. **D-C 不套用跳過規則**：每個點的 5 次都跑完。一次 FAIL 就足以判定，但跑完可以報告「5 次中幾次符合」。
   順序是每個 block 以 `capacity_plan.block_order` 洗牌，每個點各 boot 一次。
7. **D-C 的 gateway**：
   - research runner 以 `python -m capserve serve` 啟動 gateway（子 process，不 import capserve；有測試檢查原始碼）；
   - 設定為 127.0.0.1:8080 → vLLM 8000、`tokenizer = "local"`、`tokenizer_path` = 服務的 snapshot、該 workload 的 class；
   - profile 為 `verified-formal-20261017-c1b.json`；
   - gateway 的環境移除所有 `CAPSERVE_*`，並設 `HF_HUB_OFFLINE=1`。
   - `/readyz` 必須是 verified、class 正確，而且 max_concurrency 等於該點的 C，否則該 boot 在第一個 bench 之前就失敗。
   - gateway 一律先停（SIGTERM，30 s 後 SIGKILL），再進行 server 的 teardown。
8. **結果檢查**：
   - 每個 run 都要求 N 個完成、輸出都等於該 workload 的長度；
   - uniform 的 prompt 長度都等於設定值，mixed 符合 007 的數量，var 落在 [64, tier − 128]。
   - 逐 request 檔的 cls：mixed 的 32,640-token prompt 為 long，其他都是 short。
9. **var dataset**：
   - 長度以 sha256("<seed>:stage_d:var:<tier>") 為 seed，在 [64, tier − 128] 均勻抽樣；
   - 每一列以 RandomDataset(seed + i) 產生；
   - 列數 = N + warmup（2K 為 352，8K 為 208）。
   - 2026-10-23 在 WSL 產生，每個 prompt 都重新 tokenize 到剛好目標長度：
     - 2K 為 64–1,920，sha256 `358ae8da…`；
     - 8K 為 86–8,015，sha256 `d550acc8…`。
   - 兩者都 pin 在 `configs/stage_d.toml`，檔案不進 git（`data/` 依 005 不進 git）。
10. **mini smoke**（凍結的證據，三個）：
    - 和 Stage C 的 mini 相同：假的 SLO、N = 20、n_near 2、n_low 1、sentinel 在 C 2，跑完後丟棄所有延遲數值。
    - **closed**：U2K-O512 上限 3，在 C 3 強制 5 s 逾時，以走過跳過規則；M8K-T 上限 2，走過 T 的 server 檢查與 mixed。
    - **open**：U2K-P 只用前 3 個格點，每點 N = 20。
    - **verify**：U2K-var 與 U2K-gw 在 C 4 各兩個 block。gateway 仍服務 class 的正式上限（`class_limit`），所以 `/readyz` 比對的是正式上限。
11. **vLLM 的 bench client 每個 request 都送 `repetition_penalty: 1.0`**：gateway 把這個中性值視為預設值並忽略（038 的後續變更）。否則 D-C 經過 gateway 的每個 request 都會是 400。
