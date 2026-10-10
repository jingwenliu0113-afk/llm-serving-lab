# 025 — Advanced 結果：H-B4 的成因（formal-20261011-adv）

- 狀態：Accepted（結果報告；判定全部由凍結的 `adv_report.py` 產生）
- 日期：2026-10-11
- 對應：decision 023（Gate）、024（事先登記，凍結 commit `97a948c`）、PREREGISTRATION §7
- 時點：正式 session 跑完、第一次執行 `adv_report` 之後。本文**不改**任何 hypothesis、門檻、判讀矩陣或分析程式，只記錄結果、
  事先預期的對照與限制。標為「事後」的段落都是看到數據後才寫的描述、推測或敏感度計算，不是判定。

## Session

- `results/advanced/formal-20261011-adv`（raw/ 不進 git，sha256 見 `raw_manifest.json`）。
  - 部署 commit `97a948c`（`DEPLOYED_COMMIT` 核對過），GPU lock 下，2026-10-11 00:27:52–06:33:13（+08:00），約 6.1 h。
    024 的預估：正式 run 約 6.8 h，加上 smoke 共約 7.3 h。
  - pins sha256 `232f0de9…`（= 凍結的 pins），plan sha256 `0d601a1b…`，derived_from `formal-20261009-coreb-uniform`（C 為 25 / 8）。
- **有效性（A1）**：60 / 60 次 boot ok、有效，0 逾時、0 skipped，formal，env 有效，沒有 env flag，沒有未列出的繼承 env。
  - 操弄檢查：60 次都通過 log facts 與 block 數（K3 為 186，其餘 248）；T 與 TG 的 24 次 run 每個 step 都通過 chunk 檢查。
  - 分解（A10）：60 次全部有效。2K mixed 的 T、G、TG（9 次）有 preemption，依規則豁免檢查 (c)。
  - 每格都有 3 次有效 restart，沒有任何 NT。
- 報告指令（Mac，project venv）：
  `benchmarks/adv_report.py --session results/advanced/formal-20261011-adv --uniform-session results/core_b/formal-20261009-coreb-uniform --core-mixed-session results/core_b/formal-20261010-coreb-mixed2 --out results/advanced/formal-20261011-adv/analysis.json`
  - `--core-mixed-session` 只用於 A4 的描述性比值；`formal-20261010-coreb-mixed` 是失敗的 session（022），不可用。
- **獨立核對**：
  - 由 `rows.jsonl` 直接重算每個 (condition, L) 的 U、M、ρ、f、a、ln(U_c/U_0) 與 M 差的 noise 檢查，與 `analysis.json` 一致；每個類別也逐條對照 A5–A7 確認。
  - 另一個獨立的核對者以凍結程式重跑 `adv_report`，輸出與 commit 的 `analysis.json` 逐位元相同；並逐項核對本文與 README 的數字，提出的修正都已納入本文。

## 判定（`analysis.json`）

主要量：短類 P95 TTFT。U、M 為 3 次 restart 的平均（ms）；ρ = ln(M/U)。
f = 1 − ρ_c/ρ_0（log 比值的移除比例）；g = 1 − (M_c − U_c)/(M_0 − U_0)（絕對超額的移除比例，A13，只作描述）；a = ρ_K3/ρ_0 − 1。
所有 condition 的 g 類別都與 f 類別相同。preemption 是 3 次 restart 的總和。

### L = 2048（C 25）：**R6「KV 可能有關，但未確立」**

| condition | U | M | ρ | f / a | g | 類別 | mixed preemption（3 次合計） |
|---|---|---|---|---|---|---|---|
| B0 | 4,260 | 23,201 | 1.695 | — | — | — | 0 |
| T | 4,563 | 21,654 | 1.557 | f 0.081 | 0.098 | N | 296（每次 86–117） |
| G | 4,261 | 48,996 | 2.442 | f −0.441 | −1.362 | **W** | 263（每次 85–91） |
| TG | 4,561 | 24,168 | 1.667 | f 0.016 | −0.035 | N | 1,451（每次 470–494） |
| K3 | 4,260 | 28,585 | 1.904 | a 0.123 | −0.284 | (i) 不成立 | 0 |

- **H-C0 通過**：+445%，超出 noise（Core 為 +444%）。B0 對 Core 平均的比值（A4，描述）：uniform 1.000、mixed 1.001。
- H-C1 not supported（N：ρ 與 M 都下降並超出 noise，但 f 只有 0.08，未達 0.2）。
- H-C2 not supported：G 是 **W**。關閉 full-ISL 准入閘門後，mixed 的短類 P95 TTFT 變為 2.1 倍，並出現 preemption；
  uniform 不受影響（|ln(U_G/U_0)| = 0.0001）。依規則記為「閘門具保護作用」，作為 KV 證據 (iii)，不是 HOL 的證據。
- H-C3 not supported（cat(TG) = N）。
- H-C4 not supported（「KV 未顯示受限或敏感」；(i) 與 (ii) 都不成立）：
  - (i)：K3 使 ρ 與 M 都上升並超出 noise，uniform 特異性也成立，但 a = 0.123 < 0.20。
  - (ii)：B0-mixed 尾端 queue time 落在准入受阻 step 的份額為 0.103 < 0.5。
- e_KV = 1，唯一的一項是 (iii)。T、G、TG 都不是 M，且不符合 R5（需要 (i)），因此落在 **R6**。沒有 P 或 R 類的貢獻者。
- H-C5 **不一致**：B0 尾端 ΔQ（mixed − uniform）19.1 s 中，budget-long 占 0.91（HOL-long 0.05、KV-short 0.07；τ 敏感度分析下 c* 不變），
  但 cat(T) = N。依規則加註「機制歸因不一致，以操弄的結論為準」。
- 附件：ρ_TG/ρ_0 = 0.984；交互作用 I = −0.64 ± 0.05（只作描述，n = 3 的檢定力有限，事前已聲明）；
  KV 峰值（mixed）B0 99.6%，T / G / TG 100%，K3 99.5%。
- allocator：未操弄，未檢驗；浪費上界 3.1%，unexplained-break 份額 0.9%（只作描述）。

### L = 8192（C 8）：**R1「主因是 batching，即長 request 的 chunked prefill 壟斷每個 step 的 budget（H-C1 成立；007 的 conjecture 得到支持）」**

| condition | U | M | ρ | f / a | g | 類別 | mixed preemption |
|---|---|---|---|---|---|---|---|
| B0 | 2,593 | 7,783 | 1.099 | — | — | — | 0 |
| T | 6,758 | 6,151 | −0.094 | f 1.086 | 1.117 | **M** | 0 |
| G | 2,593 | 7,785 | 1.099 | f −0.0002 | −0.0004 | N | 0 |
| TG | 6,803 | 6,141 | −0.102 | f 1.093 | 1.128 | M | 0 |
| K3 | 2,818 | 10,819 | 1.345 | a 0.224 | −0.542 | (i) 成立 | 0 |

- **H-C0 通過**：+200%，超出 noise（Core 為 +175%；差距的來源見「限制」中的 uniform 雙模式）。B0 對 Core 平均的比值：uniform 0.915、mixed 0.998。
- **H-C1 supported**：cat(T) = M。ρ 從 1.099 降到 −0.094，M 也下降並超出 noise（7,783 → 6,151 ms），M guard 通過。
  - ρ 的下降（−1.193）中，約 80% 來自 uniform 變慢（ln(U_T/U_0) = +0.958），約 20% 來自 mixed 變快（ln(M_T/M_0) = −0.235）。
  - ρ_T < 0：mixed 的短類 P95 反而比 uniform 低約 9%。比值上的懲罰不只消失，而是反轉；而且主要是因為 uniform 變差。
- H-C2 not supported（N）。H-C3 不適用（單一移除已是 M）。
- H-C4 部分證據：(i) 成立（a = 0.224，ρ 與 M 都上升並超出 noise，|ln(U_K3/U_0)| = 0.083 ≤ 0.10）；
  (ii) 份額 0.0；(iii) 不成立。e_KV = 1，未達 R1「KV 也受限（次要）」附註所需的 2。
- H-C5 **一致**：B0 尾端 ΔQ 3.9 s 全部歸於 budget-long（份額 1.08；HOL-long 與 KV-short 為 0，其餘為 budget-other −0.42 s、no-queue +0.12 s；
  τ 敏感度分析下 c* 不變，份額 1.09），且 cat(T) = M。
- 規則要求附上的 M 類代價（T；TG 幾乎相同）：
  - output throughput：mixed −2.6 tok/s（60.4 → 57.7），uniform −0.8 tok/s（80.5 → 79.7）。
  - mixed 的長類 P95 TTFT：+2,416 ms（13,878 → 16,294，+17%）。
  - T 合法地改變 uniform（A6 不設檢查，只報告）：ln(U_T/U_0) = 0.958，即 uniform 的短類 P95 TTFT 變為 **2.6 倍**（2,593 → 6,758 ms）。
- 附件：ρ_TG/ρ_0 = −0.093；I = −0.009 ± 0.062；KV 峰值（mixed）97.3–99.2%；所有 condition 都沒有 preemption。
- allocator：未操弄，未檢驗；浪費上界 3.1%，unexplained-break 份額 0。

### 跨 L

兩個 L 落在不同列（R6 與 R1），依規則報告為**「成因依 L 而異」**，不合併成一般結論。

### 與事先預期的對照（024，只由 block 算術推得，不是判定依據）

- 8K：預期 R1 或 R4、budget-long 主導 → **符合**（R1、budget-long）。
- 2K：預期 R5 或 R2 / R4、HOL-long 主導 → **不符合**。實際是 R6；分解中 HOL-long 只占 ΔQ 的約 5%，budget-long 約 91%；
  G 不是 M 而是 W（關閉閘門使情況更糟）。

## 事後的描述（看到數據後才寫；不是判定，不改變任何一列）

- **8K 的 T 不是免費的修正**：它消除了 mixed 相對 uniform 的懲罰，但所有 8K prompt 也都被切成 512 token 的 chunk。
  - mixed 的短類 P95 降 21%。
  - uniform 的短類 P95 變為 2.6 倍。
  - 短類 P50：uniform 從 2.5 秒變 6.1 秒，mixed 從 2.5 秒變 4.1 秒。
  - mixed 的短類 P99 從 9.8 秒變 10.8 秒。
  - P95 TPOT：uniform 從 88 降到 59 ms，mixed 幾乎不變（130 → 125 ms）。
- **2K 的分解與操弄不一致**。一個未檢驗的可能解釋：在 C 25 下系統已經飽和，T 只是重新分配每個 step 的 token 預算，沒有減少 mixed 的總 prefill 工作量。可觀察到的現象：
  - T 下尾端的 ΔQ 從 19.1 秒降到 15.9 秒，ΔP 卻從約 0 增加到 2.0 秒。
    - mixed 的尾端 P 從 985 ms 變 3,841 ms；uniform 從 1,000 ms 變 1,843 ms。
    - P 包含 preempt 後重新排隊的時間（024「量測」），T-mixed 有 296 次 preemption。
  - mixed 的長類 P95 TTFT 從 23.3 秒升到 40.6 秒，mixed 的 output throughput 從 121 降到 100 tok/s。
  - 「budget-long」只表示「該 step 已用滿 2048 token，而且當時有長 request 在 prefill」，不量長 request 用掉多少 budget，
    也不等於「移除這個壟斷就能縮短排隊」。這個推測沒有被本實驗檢驗。
- **2K 關閉准入閘門（G）**：mixed 的 output throughput 從 121 降到 69 tok/s，total token throughput 從 4,852 降到 2,773 tok/s，短類 P99 從 28 秒變 53 秒。
  預設的 full-ISL 檢查在 2K mixed 下避免了 preemption 與重算。8K 下 G 沒有任何效果，也沒有 preemption，
  這與「8K 要 ≥ 3 個長 request 才會受 KV 限制」的 block 算術一致。
- **K3 在 2K mixed 對中位數的影響大於尾端**：mixed 的短類 P50 從 10.3 秒變 18.8 秒（+82%），P95 只 +23%；uniform 的 P50 不變（約 0.99 秒）。

## 限制

- 每格 n = 3 次 restart。noise 規則在 df = 2 時偏寬鬆，實際起作用的是實質門檻（0.2 / 0.5）。交互作用的檢定力有限。
- **8K uniform 的 boot 之間有兩個模式**（事後發現，成因未知）。每次 boot 只有一個 bench，所以這是 boot 層級的差異。
  - 短類 P95 落在約 2,481 ms 或約 2,817 ms：

    | | r1 | r2 | r3 |
    |---|---|---|---|
    | B0 | 2,481 | 2,817 | 2,480 |
    | G | 2,816 | 2,482 | 2,482 |
    | K3 | 2,819 | 2,815 | 2,819 |
    | Core uniform | 2,826 | 2,865 | 2,810 |

    T 與 TG 的 uniform 都約 6,660–6,810 ms，沒有這個現象；2K 也沒有（U 的 SE 為 1–2 ms）。
  - 影響：
    - SE(U_0) 為 4.3%，而 024 預期約 0.2–0.4%。
    - B0 比 Core 低 8.5%，以及 H-C0 的 +200%（Core +175%），大部分來自 B0 的 3 次中有 2 次落在低模式。
      B0 r2 也帶觀測 flag，卻落在 Core 的範圍內（2,810–2,865），所以這個差距不能單純歸因於觀測 flag。
    - |ln(U_K3/U_0)| = 0.083，接近 0.10 的上限。K3 的 3 次都在高模式。
  - **事後敏感度（不是判定）**：把 B0 與 K3 的 uniform 都換成同一個模式（低模式 2,481 或高模式 2,817）重算 8K：
    - a 為 0.29 或 0.32（仍 ≥ 0.20）；
    - T 的 f 為 1.08 或 1.09；
    - H-C0 為 +214% 或 +176%。
    - 8K 的類別與列都不變。
- KV 只能往下調：熱 boot 的可用量上限約 4.14 GiB，而 sm_120 走 FA2，不支援 fp8 KV。所以任何「KV 是主因」的結論都只能是推論級（R5）。
  本 session 沒有任何一個 L 落在 R5。
- 只測了一個 threshold（T = 512，由規則推得）與一個 KV 劑量（3 GiB）；G 只在 K = 4 GiB 下測。其他值的效果未知。
- allocator 未操弄。
- 所有 condition 都帶觀測 flag。2K 的 B0 與 Core 幾乎相同（比值 1.000 / 1.001）。H-C0 在兩個 L 都通過。
- 單一模型（Qwen3.5-9B W4A16）、單一 runtime 版本（vLLM 0.30.0）、單一 GPU（RTX 5070 Ti，WSL2）。

## 不變的部分

- Core 的所有判定、H-B4 的判定（not refuted）與 deployment decision table 都不變（023）。
  - 以目前的 repo 重跑 `decision_table.py`，輸出與 commit 的版本只差在 pins 的 sha256（pins 加了 `[advanced]`），表格內容完全相同。
    commit 的仍是第一次真實執行的輸出。
- 完整測試 1006 passed、5 skipped（結果 commit 前）。

## 下一步

- 023 的 Advanced 範圍（只隔離 H-B4 的成因）已完成。
- 下列項目都需要使用者另外決定，事先登記後才能收集數據：
  - 其他 Advanced 項目：fixed-SLO 最大並發、其他 threshold 或 KV 劑量、prefix caching 等。
  - 8K uniform 雙模式的成因。
  - Optional。
