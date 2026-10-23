# 041 — Stage E 事先登記（DRAFT，第 3 版）：三個機制問題

- 狀態：**DRAFT**，**E1-D 已凍結（2026-10-23，見 1.7）**；E1-P、E2、E3 仍為 DRAFT。寫在任何 Stage E 數據之前；各部分在 smoke / pilot 與審查後各自凍結。
  - 第 3 版有四個相對 037 範圍的決定（第 6 節 A1–A4）**等使用者決定**；本文先照建議的選項寫。
- 日期：2026-10-23。第 1 版 → 第 2 版 → 第 3 版都在同一天，三版之間沒有任何 Stage E 數據。修訂經過見文末。
- 對應：037（Phase 3 範圍）、024 / 025（Advanced）、028 / 032（C0）、030 / 039（逾時）、031（閒置與 spill）、PREREGISTRATION §2（H-A2）與 §11（本文摘要）。

## 0. 共通規則

| 旗標（每個部分自己的設定檔：`configs/stage_e1.toml`、`stage_e2.toml`、`stage_e3.toml`） | 範圍 | 凍結證據 |
|---|---|---|
| `freeze.e1` | E1-D（桌面分析） | 分析模組、合成 fixture 的測試與取回清單一起 commit；在取回任何 raw 之前完成 |
| `freeze.e1p` | E1-P（換 seed 探針） | 一次通過的 smoke，`harness_sha256` 相同 |
| `freeze.e2` | E2 | 一次通過的 smoke，加上 WSL 原始碼核對紀錄 |
| `freeze.e3_cuda`、`freeze.e3_metal` | E3 的兩個平台 | 各一次通過的 pilot；兩個平台共用 harness 與 `[e3]`，任一平台的正式 plan 都要兩個旗標與兩個 pilot（3.7） |

- 凍結前，正式 plan 一律拒絕（040 補記 13）。`pins.toml` 與 `capacity.toml` 不寫入。
- 每個部分一個設定檔（和 Stage Q3 相同）：一個部分改設定時，不會讓另一個部分的凍結證據失效。
- 凍結模組只 import，不修改。沒有 guard test 保護的匯入模組列入該部分的 `harness_sha256`：c0_analysis、run_c0_probe、core_a_plan、stage_b、produce_gguf、sync_gguf_to_wsl、gguf_tensors、disk_guard。
- 判定值只有 supported、not supported、NT。
  - **NT 不等於 not supported。** 解讀表中，某個 NT 若改成其他值就會換列時，結論標為 incomplete（024:429）。
  - 「H-C0 失敗」（現象不存在）與「NT」（數據不足）分開記錄。
- 有效性：無效的 run 不補跑；格子少於 3 次有效 restart 時為 NT（013）。env 無效或 sentinel 漂移的 session 整個作廢，可用新 id 重跑一次；只分析第一個有效的正式 session（040 補記 14）。
- 判定只用同一 session 的數字（024:75、:450；037 §4）。跨 session 的數字只用於設計、揭露與描述。
- 輸出在 `results/stage_e/<session>/`，不改任何既有 session：各部分的分析都寫到新的 session 目錄，並記錄輸入檔的 sha256；plan 檔不寫進 `results/stage_e/`。
- WSL 上跑的 session（E1-P、E2、E3 的 RTX）只有取回 Mac 之後，分析與嘗試次數的檢查才看得到：分析前先執行 `tools/rtx/fetch_all_sessions.sh stage_e`，取回 WSL 上 `results/stage_e` 的每個 session（不含 raw/），並 commit 它寫出的 `wsl-inventory.txt`（列出的時間與全部目錄）作為稽核紀錄。GPU 工作排在 Stage D 的正式 session 之後（037 §5），一律 `flock -n`，不搶占。

## 1. E1：8K 滑移的觸發點與雙模式（探索性）

### 1.1 問題

- 雙模式已由 C0 回答（032）：bench 的 P95 偏高等於該 bench 至少有 1 段滑移；每段長度 7，Y − m_A 為 337–338 ms。E1 只引用，不重新檢驗。
- 第 3 版分兩層：
  - **E1-D（桌面分析，0 GPU）**：滑移在 step 層級從哪個事件開始？在所有可得的數據上，用事先定義的量重現。
  - **E1-P（換 seed 探針，GPU；A1 待決定）**：這個起點和 prompt 內容綁定，還是和 bench 中的位置（時間）綁定？
- E1 只做描述，不是檢定；結果只寫進 E1 的結果文件與 Future Work，不改任何判定。

### 1.2 已知事實（設計前已看過，見第 5 節）

三組數據都用 seed 20261008，prompt 完全相同（8,064 token）。

| 數據集 | 滑移段 | 起始 idx | 直接量到的 |
|---|---|---|---|
| Advanced B0、G、K3 的 8K uniform（9 run，觀測 flag 開） | 8 | B0-r2：129、185；G-r1：65；K3-r1：129、193；K3-r2：129；K3-r3：89、193 | server 到達：8k → 8k+1 的間隔，滑移組 90.2–94.0 ms，正常組 414.9–419.8 ms；position 0 的 F，滑移組 41.5–43.8 ms，正常組 25.0–28.7 ms（T / TG 的正常組最大 29.1 ms） |
| C0（24 bench，flag 關） | 5 | 89 四次、129 一次 | 只有 TTFT 與 latency；前兆組 min_j(lat(8(k−1)+j) − lat(8(k−1))) 為 −335.4 至 −330.8 ms（35 個 offset 全部在 −335.4 至 −322.2 ms），其他 547 組 ≥ −17.3 ms |
| Core B C 8（3 run，flag 關） | 8（3 + 3 + 2） | r1：65、73、169；r2：73、89、169；r3：89、169 | client 送出延遲 0.054–0.409 ms；滑移組間隔 90.5–94.7 ms，正常組 410.6–426.1 ms；r1 的 k 8、9 為連續滑移（r2 的 k 9、11 不相鄰） |

- 21 段只落在 7 個起始 idx：89（7）、129（4）、169（3）、65（2）、73（2）、193（2）、185（1）。每段從 position 1 開始，長度 7。
- Advanced 的 step 紀錄：從 8k 到達到它的第一個 prefill chunk 之間，正常組都是 3 個 decode-only step，滑移組 6 / 6 都是 4 個（k ≤ 23；含 k 24 為 8 / 8）；8k 到達時與前一個 step 的時間差兩種組沒有分開（正常 1.3–3.8 ms，滑移 2.6–3.4 ms）。
- Advanced B0、G、K3 的 216 組中，「position 0 的 F > 35 ms」和滑移完全一致（8 / 8 對 0 / 208）。T 與 TG 的 TTFT 結構不同（長 prefill 切成較小的 chunk），028 的 `classify_bench` 在它們上面每個 run 標出 12–16 段長度 1–4 的「滑移」，這不是本節所說的滑移，所以 T、TG 不做滑移分類（見 1.4）；它們在 g 1–23 也有 20 次 F > 35 ms，集中在相近的 idx。
- 由已見數據可寫出一條五環的鏈（第 3 版的工作假設，不是結論）：
  1. 第 8k 個 request 的 F 多約 15 ms（約一個 decode-only step）；
  2. 它晚一次排程決策才進入 queue，第一個 prefill chunk 之前多一個 decode-only step；
  3. 還在 decode 的 8(k−1)+1…+7 少經過一個約 342 ms 的混合 step，提早約 330 ms 完成；
  4. 封閉迴路因此提早送出 8k+1…8k+7，間隔約 90 ms 而不是約 416 ms；
  5. 這 7 個 request 各多等一個 prefill step（每個 request 的 Q +347–351 ms，每段平均 ΔQ 348.8–349.6 ms），形成一段滑移。
- 第 2 版的「前一組的不同步」是第 3 環，是第 1–2 環的結果；第 2 版的 g*、往前追溯與三類事件全部不再使用。
- 「client 類」（晚送出 > 100 ms）只在 Core 的 8 段有直接量測（≤ 0.41 ms，不成立）；Advanced 的 8 段量到的是 server 端提早到達；C0 的 5 段是推論。028:136 與 032:67 的「位置或內容鎖定」仍然沒有答案。

### 1.3 數據可得性

| 數據 | 送出 | 完成 | TTFT | F、Q、P | step | 在 Mac | 要取回 |
|---|---|---|---|---|---|---|---|
| Advanced 8K uniform derived（B0、G、K3、T、TG，15 run） | server arrival | 無 | 有 | 有 | — | 已 commit | 否 |
| Advanced steps.jsonl（B0、G、K3） | — | — | — | — | 有 | gitignore 副本（c0-step0） | 否 |
| Advanced steps.jsonl（T、TG） | — | — | — | — | 有 | 否 | 是 |
| Advanced spans.jsonl、bench.json（B0、G、K3，9 run；T、TG 不用） | client start | e2e | 有 | 有 | — | 否 | 是 |
| C0 derived | 無 | 只有 latency | 有 | 無 | 無 | 已 commit | 否 |
| C0 raw bench.json（24） | start_times | start + latency | 有 | 無 | 無 | 否 | 是 |
| Core B C 8 raw bench.json（3） | start_times | start + latency | 有 | 無 | 無 | gitignore 副本 | 否 |

- 時鐘：bench 的 start_times 是 perf_counter 的相對值；span 與 step 用 WSL 的 epoch 時間。只在同一來源內比較，不跨時鐘對齊。
- 取回不可用時的備案：Advanced 的完成時間以封閉迴路的代理量 arrival(8(g+1)+j) 近似（Core 驗證兩者差 0.14–0.22 ms），其他需要取回的量標 blocked。

### 1.4 E1-D 的量

滑移段以 `c0_analysis.classify_bench` 判定（028），只用在 B0、G、K3、C0 與 Core。組 g 指 idx 8g…8g+7；基準只用 g 1–23；k 為 0 或 24 的段只列出，不計入鏈的判定（1.5）。k−1 本身也是滑移組時標「連續」，只計一次，不評估 L3。

- **L1**（Advanced）：F(8k) > 35 ms。
- **L2**（Advanced）：n_dec(k) = 8k 到達後、第一個 ctx_tok > 0 的 step 之前的 step 數；成立條件為等於同一 run 的 g 1–23 中位數加 1。
- **L3**（C0、Core；Advanced 需取回 e2e）：dev_j(k−1) = (lat(8(k−1)+j) − lat(8(k−1))) − 同一 bench、同一 j 的正常組中位數；成立條件 min_j dev < −170 ms（028 的 SLIP_MS）。
- **L4**：gap(k) = 到達(8k+1) − 到達(8k) < 200 ms。
- **L5**（Advanced）：ΔQ(k) = 8k+1…8k+7 的 Q 減同位置正常組 Q 的中位數，取平均；成立條件 > 170 ms。
- **送出延遲**（Core；C0 與 Advanced 需取回 bench.json）：依時間排序，第 C + m 個送出減第 m 個完成（封閉迴路的 semaphore，與 idx 無關）；每段列出 8k+1…8k+7 的值與最大值。只用 client 的時鐘（start_times 與 latency），不和 server 的時鐘相減。
- **F 尖峰的描述**（Advanced 全部 15 run；T、TG 只有這一項）：g 1–24 中 F(8g) > 35 ms 的組、F 的值、g 1–23 的尖峰率；有 step 紀錄時（T、TG 需取回），計時間戳落在 (arrival, s + 5 ms] 的 step 數與其中 decode-only 的數目，s = arrival + server_ttft − P 是 8g 被排進 engine 的時刻；對照同一 run 的 g 1–23 中位數，並列出「尖峰而且多 1 個 step」的組數。step 的時間戳是 runner 讀到該 iteration log 的 wall clock（取整到 ms，`run_adv.pump_log`），不是 step 結束的時刻：在已看過的 B0、G、K3 數據上，8g 被排進之前那個 step 的時間戳落在 s 的 −0.9 至 +3.1 ms，所以窗的終點是 s + 5 ms（decode step 約 11 ms 以上，不會多算下一個）。列出兩個窗邊（arrival 與 s + 5 ms）各自到兩側最近時間戳的距離與最近時間戳減 s，取最小距離為 margin；margin < 1 ms（時間戳的精度）時該組的計數標為 ambiguous，「尖峰而且多 1 個 step」另列不含 ambiguous 的組數。這個錨點不用「第一個 ctx_tok > 0 的 step」，因為 T 的小 chunk 可能屬於別的 request。

### 1.5 E1-D 的描述規則

- 逐段列出：數據集、run 或 bench、k、起始 idx、是否連續、可得的 L1–L5、送出延遲。
- 「鏈在該數據集成立」：可評估的段（g 1–23、非連續）中最多 1 段有某一環不成立，而且非滑移組中 L1 成立的比例 ≤ 2%；否則寫「鏈不完整」並列出例外。Advanced 的鏈只用 B0、G、K3；T、TG 的尖峰率另列在旁邊，結果文件明寫：若把 T、TG 的 20 次尖峰算進非滑移組，L1 的誤報率約 5.9%，超過 2%。
- 缺少某個 raw 檔時，只把依賴它的量標 blocked（例如沒有 spans 時 L3 用代理量、沒有 bench.json 時沒有送出延遲）；raw 檔的 sha256 和 raw_manifest 不符時停止分析，包括取回時不符、被 `fetch_raw.sh` 改名為 `<file>.sha256-mismatch` 的檔（`fetch-list` 也拒絕）。
- 三個數據集分開報告，不合併；起始 idx 的分布逐數據集列出。

### 1.6 E1-P：換 seed 的探針（A1 待決定）

- 問題：F 的尖峰（第 1 環）由 prompt 內容決定，還是由 bench 中的位置決定？封閉迴路的週期固定，位置等同時間。
- 設定：
  - server：024 的觀測版 B0（`adv_plan.server_argv_for("B0")`），F 要靠 OTLP 才量得到。**這和 C0（flag 關）不同，結果不和 C0 直接比較。**
  - client：`core_b_plan.bench_argv`，L 8192、uniform、C 8、N 200、warmup 8；request-id prefix `e1p-b{b}-n{j}-`。
  - seed A = 20261008，seed B = 20261023（現在固定）。
  - 6 次 boot，每次 4 個 bench；奇數 boot 為 A B B A，偶數 boot 為 B A A B。每個 seed 各 12 個 bench。
- 量：滑移段、每個 request 的 F、L1–L5、step 紀錄。尖峰 idx = position 0（8k，k 1–24）的 F > 35 ms。
- 重複位置：某個 seed 的有效 bench 中，至少 3 個 bench 出現尖峰的 idx；兩個 seed 的集合為 A_in、B_in（樣本小，讀法粗略）。
- 既有的重複位置：Advanced 15 個 run 中出現至少 2 次的 idx，A_prior = {32, 64, 88, 128, 176, 192}。
- 讀法（依序取第一個符合的；描述，不是檢定）：

| 條件 | 讀法 |
|---|---|
| 任一 seed 的尖峰總數 < 5，或 A_in、B_in 有一個是空集合 | 無法判定（觸發太少） |
| A_in ∩ A_prior 為空集合 | 既有的位置集中沒有在 session 內重現，不判定 |
| J(A_in, B_in) ≤ 0.2 | 和 prompt 內容鎖定：換 seed 之後，觸發位置改變 |
| J ≥ 0.5 | 和位置或時間鎖定，與內容無關 |
| 其他 | 無法判定 |

  - J = |A∩B| / |A∪B|；兩個 seed 各有 3 個重複位置且彼此獨立時，期望約 0.07。
- 另外描述：每個 seed 的 F 尖峰與滑移的一致性；滑移率（既有：C0 0.21、Advanced 0.89、Core 2.67 段 / bench）；第 1 個 bench 與其他位置的滑移數。
- 停止規則沿用 028 與 C0（run_c0_probe.py:815-829）：
  - 每次 boot 前做 031 的閒置判定與 spill 偵測；bench 逾時 720 s；
  - boot 沒有啟動，或有 2 個以上無效 bench，該 boot 失敗；第 2 次失敗就停止，不補跑；
  - 持有 lock 的上限 3.0 h（含 smoke），已用時間加 30 分鐘會超過時不開始新的 boot；GPU ≥ 88 °C 立即停止；不依數據延長或提前停止。

### 1.7 E1 的凍結證據

- **`freeze.e1`**：E1-D 的模組、合成 fixture 的測試（連續滑移、g 0 與 g 24 的排除、n = 192 的短 bench、時鐘不跨來源（client 與 server 的時間刻意不同，送出延遲與 C0 的 L4 都核對數值）、亂序完成的送出延遲、e2e 與代理量不同（代理量的 index 也核對）、部分取回、取回時 sha256 不符、窗邊的 ambiguous、s 落在時間戳 −0.9 至 +3.1 ms 的真實時序、receiver 格式的 spans 與 warmup span）與取回清單一起 commit，`freeze.e1_harness_sha256` 記下當時的 `harness_sha256()`；`[e1]` 的五個值固定為本節的值。`fetch-list` 與 `analyze` 在未凍結或 harness 不符時拒絕。之後才以 `tools/rtx/fetch_raw.sh` 唯讀取回（放進 `results/capacity/c0-step0/raw/<session>/`）並依 raw_manifest 驗證 sha256：
  - `advanced formal-20261011-adv`：{B0,G,K3}-8192-uniform-r{1,2,3} 的 spans.jsonl 與 bench.json，以及 {T,TG} 的 steps.jsonl；
  - `capacity explore-c0-20261016`：b*-n*.bench.json。
  - 取回會讀 WSL 的磁碟與網路，只在 `tools/rtx/gpu_status.sh` 印出 FREE 時進行（不拿、不等 lock），排在 GPU 正式 session 之外。
  - 輸出只能是 `results/stage_e/<session>/` 下的新檔，記錄每個輸入（derived 與 raw）的 sha256、harness 與 `configs/stage_e1.toml` 的 sha256。
- **E1-D 已凍結（2026-10-23）**：`configs/stage_e1.toml` 的 `freeze.e1 = true`，`e1_harness_sha256 = 87b937d12799853b50745bc4430de5c26f8b9db71cd6726aebb78466a656c00d`（`benchmarks/stage_e1.py` 與 `benchmarks/c0_analysis.py`）。凍結時清單中的檔一個都還沒取回。`tests/test_stage_e1.py` 檢查設定檔的 pin 等於目前的 harness，所以兩個檔都不再修改；E1-D 的規則之後只能以新的 decision log 修訂。輸出寫在 `results/stage_e/<新的 session>/`。
  - 取回：`python benchmarks/stage_e1.py fetch-list | while read -r args; do tools/rtx/fetch_raw.sh $args; done`，只在 `tools/rtx/gpu_status.sh` 印出 FREE 時執行。
- **`freeze.e1p`**：smoke 1 次 boot、2 個 bench（seed A、B 各一，N 16）。通過條件：每個主 id 都有 span、F 算得出來、時間窗內有 step 紀錄、seed B 的 input_lens 全部 8,064、derived schema 正確；延遲遮蔽，不看。

### 1.8 能交付與不能交付

- 能：滑移在 step 層級的完整鏈（大部分已看過，E1-D 讓它可以稽核，並補上完成時間與送出延遲）；觸發和內容還是位置綁定（E1-P）。
- 不能：frontend 多出約 15 ms 的根本原因（需要 profiling）；032 的「偏高比例與 Phase 1 不同」（需要多個 session，見 A1）。

## 2. E2：2K 的 H-B4 成因（正式，RTX）

### 2.1 問題與候選解釋

- 025：2K、C 25 時 B0 尾端 ΔQ 有 0.91 在 budget-long，但 T512 沒有移除 ρ（f 0.081，類別 N）；T 每個 run 造成 86–117 次 preemption。
- 第 3 版的問題：**為什麼 T 在 2K、C 25 無效？**
  - **S（負載）**：C 25 的總 prefill 工作讓 step 一直滿載，T 只是重新分配。
  - **V（T × KV）**：KV 受限時，T 讓長類分段配置 block，准入閘門只檢查、不預留，造成 over-commit（preemption 與重算）或 HOL 阻塞，抵消 T。
  - **Th（threshold 太大）**：T512 仍讓長類每個 step 占 512 token。

| cell | S | V | Th |
|---|---|---|---|
| T@25（4 GiB） | 無效 | 無效 | 無效 |
| T@8（4 GiB） | 有效（負載差足夠時） | 有效 | 有效 |
| T@8k（2.5 GiB） | 有效 | 無效 | 有效 |
| T256@25 | 無效 | 無效 | 有效 |

- T@8k 分開 S 與 V；T256@25 分開 Th 與 {S, V}。「高負載、KV 不受限」的格子做不出來（KV 無法調高，熱 boot 上限約 4.14 GiB），所以 S 只能以條件式讀法出現。

### 2.2 量的定義

- **U、M**：uniform 與 mixed 的短類 P95 TTFT（`core_b_analysis.short_class_p95_ttft_ms`），3 次 restart 平均；SE = s/√3。
- **ρ、f、類別**：ρ = ln(M/U)，f = 1 − ρ_c/ρ_0；類別 M、P、R、W、N、NT 依 024 A5，用凍結的 `adv_analysis` 函式（`cell_stats`、`rho_stats`、`removal_category`、`exceeds`、`beyond_noise`）。
- **參照**：同一 session、同一 C、同一 K 的 B0：T@25 與 T256@25 → B0@25；T@8 → B0@8；T@8k → B0@8k。
- **m、u**：m = ln(M_0/M_c)（mixed 短類本身的改善），u = ln(U_c/U_0)。
- **有效（E）與無效（I）**：
  - E：類別 ∈ {M, P}，而且 m ≥ 0.20（審查建議的值，在看任何 E2 數據之前定下）；
  - I：類別 ∈ {N, R, W}；或類別 ∈ {M, P} 但 m < 0.20（另標「主要靠 uniform 變慢」）；
  - 類別為 NT 時記為 NT。
- **H-C0 閘門**：每個 B0 參照各呼叫一次 `core_b_analysis.h_b4_values({2048: {C, uniform, mixed}})`；通過條件就是該函式的判定（增幅 ≥ 20%，且 |M − U| > 2√(SE_M² + SE_U²)）。閘門只用於 T 的對比，不影響 H-E2e。
- **負載 β**（只用於條件）：mixed bench 時間窗內，1 Hz 取樣中 `num_requests_waiting > 0` 的份額，3 次平均。β_ok：β(B0@25) − β(B0@8) ≥ 0.20。
- **KV 操弄有效（KV_ok）**：B0@8k 的 mixed gate-zone 份額（025 的 1 Hz 定義）高於 B0@8 並超出 noise（`exceeds`），而且 uniform 特異性 |ln(U_8k/U_8)| ≤ 0.10。
- **描述量**（每格都報告，不改判定）：f、m、u、類別；preemption 總數（**T@8 必報**：C 8 時 3 個長類同時在場為 233 / 247 個 block）；重算 token R = Σ ctx_tok − Δvllm:prompt_tokens_total（prompt_tokens 不計重算）；KV 峰值、kv_full_while_waiting、gate-zone 份額；B0 格的分解（budget-long、HOL-long、KV-short）；mixed 的 total token throughput；長類 P95；Λ（滿 2,048 token 的 step 時間份額）與 L̄q（平均排隊中的短類數）；交互作用 I_KV = (ρ_B0@8 − ρ_T@8) − (ρ_B0@8k − ρ_T@8k)，附 SE；allocator 的 block 粒度浪費上界與 unexplained-break 份額，寫成「未操弄，未檢驗」（024 A13）。
  - 計算方式：每格只在 3 次 restart 都有效時報（少於 3 次不報）；分解的份額（tail、q_shares、unexplained-break）還要 3 次分解都通過 A10（024:64）；R 的 Σ ctx_tok 取 bench 時間窗內的 step（和 Δprompt_tokens 一樣含 warmup）；Λ = 時間窗內相鄰 step 的區間 (t_{j−1}, t_j] 中，step j 的 ctx_tok + gen_tok ≥ 2,048 的區間長度份額；L̄q = 短類主 request 的 Q 加總 ÷ 時間窗長度（Little's law）；每個 (cell, mix) 報 3 次有效 restart 的平均與 SE（缺任一次時不報）；allocator 上界用 024 A13 的 0.031。

### 2.3 condition

所有 condition 都以 024 的觀測版 B0 為基礎。

| condition | 和 B0 的差異 | 預期 block / block_size | 操弄檢查 |
|---|---|---|---|
| B0 | 無 | 248 / 528 | 024 A1 |
| T | `--long-prefill-token-threshold 512` | 248 / 528 | 每個 step：ctx_tok ≤ 512 × ctx_req |
| T256 | `--long-prefill-token-threshold 256` | 248 / 528 | 每個 step：ctx_tok ≤ 256 × ctx_req |
| K2.5 | `--kv-cache-memory-bytes 2684354560` | 155 / 528（`blocks_bounds`） | /metrics 的 block 數 |
| TK2.5 | T 加 K2.5 | 155 / 528 | 兩項都檢查 |
| PC | 唯一的 `--no-enable-prefix-caching` 換成 `--enable-prefix-caching` | 248 / 528（smoke 確認） | 命中檢查（2.8） |

- non-default args 必須恰好是該 condition 的 key 加兩個觀測 key；`enable_prefix_caching` 只有 PC 為 True。
- T 在所有 condition 都固定為 512；依 024 的規則，K2.5 下 n = 2，算出的 T 是 682，這裡不跟著改，只讓 KV 變動（揭露）。

### 2.4 KV 算術

短類（1,920 + 128）占 7 個 block；長類（32,640 + 128）完成時 66 個，准入時 65 個。

| | 可用 block | 1 長 + 其他短 | 2 長 + 其他短 | 第 2 個長類何時能准入 |
|---|---|---|---|---|
| C 25，4 GiB | 247 | 234，放得下 | 293，放不下 | 系統中最多 16 個短類時 |
| C 8，4 GiB | 247 | 115 | 174（3 長 5 短 233 也放得下） | 一定可以，除非已有 3 個長類 |
| C 8，2.5 GiB | 154 | 115 | 174，放不下 | 系統中最多 3 個短類時 |

- K2.5 在 C 8 重建了 C 25、4 GiB 的**結構**（1 個長類放得下，2 個長類加其他放不下），**不是頻率**：pinned dataset 中，C 8 時兩個長類同時在場比 C 25 少。
- K2.5 符合 019 的 ≥ 128 block；C_KV(2K) = ⌊154/7⌋ = 22 ≥ 8。

### 2.5 格子

全部 L 2048，每格 3 次 restart；每個 restart 區塊內的 16 個 (cell, mix) 以 `random.Random(int(sha256(f"20261008:stage_e:E2:block:{r}"), 16))` 打亂；每次 boot 一個 bench（024）。

| cell | condition | C | mix | N / warmup | 用途 |
|---|---|---|---|---|---|
| B0@25 | B0 | 25 | u、m | 250 / 25 | 參照、H-C0 |
| T@25 | T | 25 | u、m | 250 / 25 | H-E2-R |
| T256@25 | T256 | 25 | u、m | 250 / 25 | H-E2c |
| B0@8 | B0 | 8 | u、m | 200 / 8 | 參照、H-C0 |
| T@8 | T | 8 | u、m | 200 / 8 | H-E2a |
| B0@8k | K2.5 | 8 | u、m | 200 / 8 | 參照、H-C0、H-E2k |
| T@8k | TK2.5 | 8 | u、m | 200 / 8 | H-E2b |
| PC-off | B0 | 25 | 共享前綴 | 250 / 25 | H-E2e |
| PC-on | PC | 25 | 共享前綴 | 250 / 25 | H-E2e |

- 共 48 次 boot。逾時 `capacity_plan.bench_timeout_s(2048, mix, C, TAU1_S)`，T 與 T256 乘以 2（039）。
- mixed 一律用 pinned 的 `data/core_b/mixed-2048.jsonl`（sha256 從 pins 讀，session 中核對），加 `--disable-shuffle`；C 8 讀前 200 列（180 短 + 20 長），是 C 25 前 200 列的前綴，不另產生 dataset。
- uniform 用 seed 20261008 的 random dataset，和 Core 相同。

### 2.6 Hypotheses

| id | 內容 | 預測 |
|---|---|---|
| H-E2-0 | 每個 B0 參照都重現 H-B4（`h_b4_values` 通過）；任一參照失敗為 not supported，否則任一為 NT 時為 NT；同時列出各參照的狀態 | supported（C 25 已知 +445%；C 8 只有 Stage C 不同 session 的 M2K C 4 +587%） |
| H-E2-R | T@25 為 I（在同一 session 重現 025） | supported（025：N，m 0.069） |
| H-E2a | T@8 為 E | supported（S、V、Th 都這樣預測） |
| H-E2b | T@8k 為 I | 不預測（V 預測 supported；S、Th 預測 not supported） |
| H-E2c | T256@25 為 E | 不預測（Th 預測 supported；S、V 預測 not supported） |
| H-E2k | B0 在 C 8 的 KV 劑量效果，完全照 024 A7(i)：a = ρ_8k/ρ_8 − 1 ≥ 0.20、uρ、m_up、\|ln(U_8k/U_8)\| ≤ 0.10；ρ_8 ≤ 0 時相對劑量沒有定義，記為 NT | supported，信心低（025 在 C 25 的 K3 只有 a 0.123） |
| H-E2e | 共享前綴 uniform：Y_on ≤ 0.8 × Y_off，且超出 noise | supported；只要有命中幾乎必然成立，這是能力檢查，不是機制檢驗 |

### 2.7 解讀表（依序取第一個符合的列）

| 列 | 條件 | 讀法 |
|---|---|---|
| R0 | B0@25 的 H-C0 失敗 | C 25 在 session 內沒有重現 H-B4，不做歸因 |
| R0b | T@25 為 E | 025 的 N 沒有在 session 內重現，E2 的問題不成立，照實報告 |
| R0c | B0@25 或 T@25 為 NT | incomplete，不解讀 |
| R1 | B0@8 的 H-C0 失敗 | C 8 沒有 2K 的懲罰，T 的負載與 KV 對比都無從檢驗 |
| R2 | T@8 為 I | β_ok 成立：負載較低、KV 不受限時 T 也無效，S 與 V 都不是 T 失效的必要條件。β_ok 不成立：只記「T 在 C 8 也無效，負載差不足」 |
| R3 | T@8 為 E、KV_ok 成立、T@8k 為 I | 在 C 8 的負載下，KV 壓力就足以讓 T 失效：支持 V；S 不是必要條件，S 在 C 25 是否另有貢獻無法辨識 |
| R4 | T@8 為 E、KV_ok 成立、T@8k 為 E | C 8 重建的 KV 結構不足以讓 T 失效，C 25 的失效需要和 C 有關的因素。β_ok 成立：符合 S，但也可能是負載與 KV 的交互作用，分不開。β_ok 不成立：不解讀 S |
| R5 | T@8 為 E，但 B0@8k 的 H-C0 失敗或 KV_ok 不成立 | KV 對比不成立，只記「T 在 C 8 有效」 |
| R6 | 其他（含 NT） | 照實報告，不解讀；有 NT 時標 incomplete |

- **R3 的子標籤**：T@8k 的 preemption > 0 而且 R > 0 → over-commit 路徑；preemption = 0 而且 B0@8k 的 HOL-long 加 KV-short 份額 ≥ 0.5 → HOL 路徑；其他 → 路徑未分辨。preemption 取 T@8k mixed 3 次有效 restart 的加總，R 取其平均；份額取 B0@8k mixed 3 次的平均，而且 3 次的分解都要通過 A10；任一缺值或分解無效時子標籤為 NT。
- **incomplete**：主列的某個輸入為 NT（參照的 H-C0、T 的判定、KV_ok）時，把它換成每一個可能的確定值重新套用解讀表，主列會改變才標 incomplete；R0c 一律 incomplete；R6 有任何 NT 時標 incomplete。H-C0 失敗是確定的結果，不是 NT。β_ok 為 NT 時，R2 與 R4 的讀法改為「β_ok NT，負載差無法判讀」並標 incomplete。
- **T256 附註**：T256@25 為 E → 較小的 threshold 在 C 25 有效，T512 的失效和 threshold 大小有關，與 S、V 的預測都不符；主列若是 R3 或 R4，加註「機制解讀只適用於 T512」。T256@25 為 I → 沒有鑑別力（T256 自己的切段與 preemption 也有成本）。NT → incomplete。
- **H-E2k** 單獨報告，回答 037 的「其他 KV 劑量」；主表用的是 KV_ok。
- I_KV、Λ、L̄q 與 025 尾端分解的觀測簽名只作描述，不決定任何列。

### 2.8 E2e：prefix caching 的能力檢查（A3 待決定）

- 問題：在這個 hybrid（GDN）模型與 vLLM 0.30.0 上，共享前綴時 prefix caching 能否縮短短類 TTFT？所有基準都關閉 PC，所以它不是已觀察到的 H-B4 的成因。
- 處理的定義：PC 的 flag，加上 vLLM 因此改變的 hybrid cache 模式（024:194）；smoke 記錄 mamba_cache_mode 與 attention block size。
- dataset（WSL、CPU 產生）：只呼叫一次 `RandomDataset.sample(num_requests = N + warmup, prefix_len, input_len, range_ratio = 0)`，seed 20261008；以 server 的 tokenizer 驗證每列恰好 1,920 token、所有列的共同 token-id 前綴 ≥ 3 × block_size（528 時 1,584），否則不寫檔；實測的共同前綴長度與 sha256 pin 在該部分的設定檔。
- 命中檢查（每次 boot）：span 有 cached-token 欄位時（名稱含 cached 與 token；smoke 記錄名稱），N 個主 request 都要有值，≥ 95% 命中 1,584 token；否則 Δprefix_cache_hits ≥ 0.95 × N × 1,584 且 ≤ (N + warmup + 1) × 1,584；PC-off 必須為 0（兩種來源都檢查）；不符合時該 run 無效。
- mamba_cache_mode 取自 /metrics 的 `vllm:cache_config_info` 標籤（WSL 原始碼核對確認標籤名稱）；smoke 中 PC-off 與 PC-on 都必須有值。
- block_size 若不是 528：另寫 decision log 並重做 dataset，不臨時改設計。

### 2.9 有效性與停止規則

- 有效性：024 A1 加 031（閒置判定、spill），再加 K2.5 的 155 個 block、T 與 T256 的 chunk 檢查、PC 的命中檢查、mixed dataset 的 sha256。同一格出現無效 run 後，不再跑該格剩下的 restart（024）。
- 閒置判定的上限：B0、T、K2.5 沿用 F_peak 15,157 MiB；T256 與 PC-on 由 smoke 量 bench 後的增量，超過 15,157 + 128 MiB 時，凍結前另寫 decision log 改用實測值。
- 停止規則：不提前停止，也不依結果決定任何事；連續 3 次 boot 沒有就緒時停止（session 記 stopped，剩下的 boot 記為 skipped，該 session 作廢）；持有 lock 的上限 7.5 h（約預估的 1.75 倍），到上限時在 boot 之間停止，沒跑完的格為 NT。
- 記憶體：每次 boot 在 bench 之後、teardown 之前量一次 nvidia-smi 的 memory.used，減去該 boot 閒置窗的 idle_mib，記為增量；量不到時記為 None 並記錄錯誤（描述量，不影響 Y 與 session；smoke 的記憶體條件因此不通過）。
- 重跑（§0）：session 目錄只能是 `results/stage_e/<id>`；第一個 boot 之前先寫 `session.json`（status running，之後以暫存檔加 rename 覆寫），中斷的 session 也是一次嘗試，讀不到時以 `plan.json` 判定為作廢的嘗試；在第一個 boot 之前被拒絕的（status refused）不算，沒有啟動任何 server 就中止的（例如第一個 boot 前的閒置判定失敗、vLLM 的路徑錯誤、OTLP 的 port 被占用）也記為 refused。正式 session 在主 checkout 的 `results/stage_e/` 下執行（worktree 看不到其他嘗試）。已有一個有效的正式 session（finished、env 有效、沒有 stopped），或已有兩次嘗試時，runner 拒絕開始，分析也只接受「之前最多一次作廢嘗試」的第一個有效 session。plan 記錄的 `stage_e2_sha256` 是設定檔去掉 `[freeze]` 後的內容雜湊。

### 2.10 E2 的凍結證據

- **WSL 原始碼核對**（CPU，不需要 GPU lock）：hybrid 模型的 prefix caching 路徑（mamba_cache_mode、block size）；custom dataset 的 warmup 來源（serve.py:840-868）；/metrics 中 prefix cache 計數器的名稱；span 是否帶 cached-token 欄位；1 Hz 取樣中 `num_requests_waiting` 的來源。
- **smoke**：設計中每個 (condition, mix) 一次 boot（12 次：B0、T、T256、K2.5、TK2.5 的 uniform 與 mixed，PC-off、PC-on 的共享前綴），`rc.mini_num_prompts(20)`。smoke 之前要先 pin 共享前綴 dataset 的 sha256；smoke 的 plan 必須恰好是 build_plan 的 smoke plan，datasets 都要符合 pin。通過條件：涵蓋全部 12 個 (cell, mix)；每次 boot 有效（non-default args 恰好符合、FLASH_ATTN、block 數 K2.5 155、其他含 PC 都是 248、block_size 528）；chunk 檢查；PC-on 命中符合、PC-off 為 0；PC-off 與 PC-on 的 mamba_cache_mode 有值；A10 (a)(b)(d) 通過，(c) 通過或豁免（有 preemption，或 PC）；T256 與 PC-on 的 bench 後記憶體增量 ≤ 15,157 + 128 MiB（否則依 2.9 另寫 decision log）；沒有 spill；teardown 乾淨；有 derived；env 有效。延遲遮蔽，不看：bench 結果、spans、OTLP、Prometheus 文字檔、steps、series 都刪除，session.json 不留 bench 時間窗與 startup 時間。
- 凍結證據（`smoke_problems`）：smoke 跑完而且通過、env 有效、WSL CUDA host、pins 的 vLLM（smoke 也在開始前核對版本）、相同 harness、plan 等於目前設定下的 smoke plan、datasets 等於 pin、涵蓋全部 (cell, mix)、量到的 block 數等於預期。

### 2.11 能交付與不能交付

- 能：T 在 2K 的失效是否依賴 KV 壓力（C 8，4 對 2.5 GiB）；較小的 threshold 是否有效；C 8 的 KV 劑量效果；prefix caching 的能力與效果量。
- 不能（A4 待確認）：在 C 25 以「移除」確立成因（KV 不能調高，負載與 KV 不能同時分開）；「飽和是成因」的直接證據（S 只在 R4 加 β_ok 時寫成「符合」）。

## 3. E3：H-A2 被推翻的原因（正式，Mac 與 RTX）

### 3.1 問題與候選機制

- H-A2 被推翻：4B-Q4_K_M 對 9B-Q8_0 的觀測時間比 2.22–2.24，bytes 比 2.99。
- 候選機制（可以並存）：
  - **非比例成分**：decode 時間中有一段和 bytes 不成比例的部分。可能是固定開銷（kernel launch、CPU 端），也可能是頻寬效率隨張量大小改變；本設計**分不開**這兩者。
  - **4-bit 格式成本**：k-quant（Q4_K_M）每個 byte 的成本比簡單格式（Q4_0）高。

| 量 | 非比例成分 | 格式成本 |
|---|---|---|
| 同一格式跨 size 的截距 a_f | > 0 | 不預測 |
| D_s = r(s, Q4_0) − r(s, Q4_K_M) | ≈ 0 | < 0，兩個 size 都是 |

### 3.2 設定

- 8 個設定：4B 與 9B 各自的 Q8_0、Q6_K、Q4_K_M、Q4_0。4B-Q5_K_M 刪除（兩個機制都預測它落在線上）。
- 只量 L 2048，3 次重複，前後各一個 sentinel（4B-Q4_K_M，漂移 ≤ 3%）。8K、32K 不服務任何 hypothesis，刪除。
- 其他沿用 Core A 的 decode 協定（llama.cpp b11146、005 的 flag set、`core_a_plan._bench_job`）；順序 `Random(sha256(f"20261008:stage_e:E3:{platform}"))`。
- **GGUF 產生**（Mac lock 下）：從 pins 的 BF16 GGUF 以 `llama-quantize` 產生 Q4_0，預設配方、不加 `--pure`；9B 沒有 tie，`output.weight` 留在 Q6_K（和 9B-Q4_K_M 相同），照實揭露；4B 有 tie，token_embd 以實際產物為準。
  sha256、大小、tensor 類型分布、token_embd 的 bytes 與類型、是否 tie、`output.weight` 的類型、model_size 與 model_type 都 pin 在該部分的設定檔（`produce_gguf_e3.header_facts` 從 header 讀出；6 個舊檔也一樣 pin，並以本機 header 測試）；磁碟保留 50 GiB；以 `produce_gguf_e3.py sync`（`sync_gguf_to_wsl` 的 receiver 與 `copy_one`，sha256 與大小取自設定檔）傳到 WSL 並驗證 sha256。
  - model_size 是 header 中所有 tensor 的 bytes 總和，等於 llama-bench 列中的 `model_size`（6 個舊檔都核對過）；檔案大小多出 metadata（tokenizer 等），所以列的核對不用 size_bytes。x_c 照 Core A 仍用檔案大小。model_type 是 llama-bench 的「架構 size ftype」字串（例如 `qwen35 9B Q4_0`），由 `general.file_type` 對應；對應錯誤時 pilot 會失敗，要在凍結前修正。
  - 參數數 P_4B = 4,205,751,296、P_9B = 8,953,803,264：BF16 header 的元素總數（等於 llama-bench 的 `model_n_params`），pin 在設定檔，用在 3.5 的共用斜率擬合。

### 3.3 量的定義（單位 ms/token）

- t_c = 1000 / m_c，m_c 是 `core_a_analysis.cell_stats` 在 L 2048 的 decode 平均 tok/s；SE(t_c) = 1000 × se / m²。
- x_c = `core_a_analysis.decode_bytes(file, embd, 2048)` = 檔案大小 − token_embd（只扣沒有 tie 的檔案）+ 32,768 × (2048 − 64) + 2·S；embd 由 `scripts/gguf_tensors.py` 讀出（更正第 2 版文字中相反的描述）。
- 全域擬合：同一 session 的 6 個舊設定，未加權 OLS t = a + b·x，得 a、b、殘差 SD s（4 df）、SE(a)、SE(b)。殘差 r_c = t_c − (a + b·x_c)，負值代表比擬合快。
- 跨 size 截距：a_f = (x_9B·t_4B − x_4B·t_9B) / (x_9B − x_4B)，SE(a_f) = √((x_9B·SE_4B)² + (x_4B·SE_9B)²) / (x_9B − x_4B)。
- 格式對比：D_s = r(s, Q4_0) − r(s, Q4_K_M)，SE_D = √(SE_t(Q4_0)² + SE_t(Q4_K_M)² + (Δx·SE_b)²)。
- 同 session 對照 κ = max over s 的 |r(s, Q6_K) − r(s, Q4_K_M)|（同 size 兩個 k-quant 之間的一般散布）。
- D_s 的狀態：「負且超出」= D_s < 0 且 |D_s| > max(2·SE_D, κ)；「正且超出」同一門檻；其他為「在內」。

### 3.4 Hypotheses

| id | 內容 | 預測 |
|---|---|---|
| 前提 | 同一 session 中，H-A2 配對的觀測比低於 bytes 比，差距 > 2·SE（delta method） | 成立 |
| H-E3a | 非比例成分的重現：3 個舊格式的 a_f 都 > 2·SE(a_f) | supported（Phase 1：Mac 4.6–11.6 SE，RTX 15–49 SE；這次只是在同一 session 重現） |
| H-E3b | 兩個 size 的 D_s 都是「負且超出」 | 不預測 |

兩個平台各自判定，不合併。

### 3.5 解讀表（每個平台，依序取第一個符合的列）

| 條件 | 讀法 |
|---|---|
| 前提不成立 | H-A2 的偏差沒有在 session 內重現，不做機制歸因 |
| 任一需要的格為 NT | incomplete，照實報告 |
| H-E3a 與 H-E3b 都 supported | 非比例成分與 4-bit k-quant 的格式成本並存 |
| H-E3a supported，兩個 size 的 D 都「在內」 | 有非比例成分；沒有格式成本的證據（不等於排除） |
| H-E3a supported，兩個 size 的 D 都「正且超出」 | Q4_0 比 bytes 預測的慢，和兩個候選都不符；照實報告並列出 output.weight 為 Q6_K 等混淆 |
| H-E3a supported，D 的方向或狀態在兩個 size 不同 | 格式效應不一致，不解讀 H-E3b |
| H-E3a not supported，H-E3b supported | 沒有可測的非比例成分；格式成本是唯一被支持的機制 |
| H-E3a not supported，其他 | 兩個候選都沒有被支持；H-A2 的偏差未被解釋 |

- 兩個平台落在同一列，才寫成一般結論；否則報告「原因依平台而異」。
- 兩個平台都有明確的列（含「前提不成立」）才比較；任一平台為 incomplete 或缺少時寫「incomplete」。
- 描述量：
  - a、b、s 與全部殘差（ms 與 %）；
  - 各格式 a_f（含可用時的 Q4_0）的兩兩差距，SE = √(SE_1² + SE_2²)（不同的格，獨立），以及 3 個舊格式的截距同質性 χ²；
  - 共用斜率擬合：6 個舊設定擬合 t = a_size + b·x（每個 size 一個截距、共同斜率，3 df）；再由 a_size = a0 + c·P_size / 10⁹ 解出 a0 與每十億參數的成本 c（沒有自由度），P 為 3.2 pin 的參數數；
  - φ = (R_bytes − R_affine) / (R_bytes − R_obs)，H-A2 配對的偏差中 affine 模型能解釋的份額；兩種都報：in-sample（6 個設定的全域擬合）與 leave-pair-out（只用配對以外的 4 個設定，2 df）。設計前的 tally 用 leave-pair-out（Phase 1：Mac 0.455、RTX 0.730；in-sample 為 0.91、0.94）。

### 3.6 有效性與停止規則

- Mac：Mac lock、Apple 140W adapter（記錄名稱與協商瓦數）、同一 session 一個電力模式、012 的電量與 thermal 規則與充電閘門；凍結的 `run_core_a` 要求 70W，所以 E3 有自己的 runner（重用 job 格式、解析與 env capture）。
- RTX：session 開始時量 B（`stage_b.measure_baseline`）；只套用 spill 的速度判定（E3 不跑 fit run，記憶體判定不適用，揭露）。
- 每次 session 前驗證 8 個 GGUF 的 sha256 與 header（token_embd 的 bytes 與類型、tie、`output.weight` 的類型、tensor 類型分布、model_size），任一不符就不開始；每列 llama-bench 結果的 model_filename、model_size、model_type 都要和 `configs/stage_e3.toml` 一致，不符的列記為 error。
- RTX 的 B 在建立 session 目錄之前量；量不到時什麼都不開始。
- sentinel 漂移 > 3% 或 env 無效：session 作廢，可重跑一次；只分析第一個有效的 session。具體規則：
  - 正式 session 在主 checkout 執行（worktree 的 `results/stage_e/` 看不到其他嘗試）。session 目錄只能是 `results/stage_e/<id>`（不可巢狀），第一個 job 之前先寫 `session.json`（status running；之後的覆寫是先寫暫存檔再 rename），所以中斷的 session 也是一次嘗試；`session.json` 讀不到時以 `plan.json` 判定，算一次作廢的嘗試（fail closed）。在第一個 job 之前就被拒絕的（status refused，例如 adapter 或 build 不符）不算。
  - pilot 也只能執行 build_plan 的 pilot plan（mode 只有 pilot 與 formal，plan 與 harness 都要相符），所以完整設計不能標成 pilot 執行。
  - 只有 `plan.json`、沒有其他檔案的目錄不算嘗試（沒有任何 job 執行過）；`results/stage_e/` 本身不是 session。
  - plan 記錄的 `stage_e3_sha256` 是設定檔去掉 `[freeze]` 後的內容雜湊，所以一個平台先跑完正式 session、另一個平台之後才凍結，前者仍可分析；分析比對 plan 時不比 `pins_sha256`（pins 帶進 plan 的內容逐項比對）。
  - 同一平台已有一個有效的正式 session，或已有兩次嘗試時，runner 拒絕再開始；分析也拒絕「之前有兩次以上嘗試，或之前已有有效 session」的 session。
- 任一格少於 3 個 ok 的 process 或有 spill：依賴該格的 hypothesis 為 NT（H-E3a 需要 6 個舊設定；H-E3b 需要 6 個舊設定（全域擬合與 κ）加兩個 size 的 Q4_0）。不改設定重試。
- 上限：Mac session 1.5 h，RTX 1.0 h，只計量測時間：從第一個 job 開始，扣掉 session 中的充電暫停；開始前的雜湊、header 檢查、B、preflight 與 Mac 在開始前的一次充電閘門都不計。到上限時跳過剩下的 job（未完成的格為 NT），但 end sentinel 一定執行（漂移要量）。每個 job 的 timeout 為 min(7,200 s, 剩餘上限 + 600 s)，end sentinel 為 7,200 s。

### 3.7 凍結證據

每個平台一次 pilot：兩個 Q4_0 加 4B-Q4_K_M，L 2048，各 1 次，加 sentinel；結果遮蔽（commit 的 rows.jsonl 只留 job 的身分與結果，完整的列在 git-ignored 的 `raw/rows.full.jsonl`，每個 job 前後的 env capture 也放在 `raw/env/`，因為它們的時間戳等於 job 的時間；session.json 的 sentinel 只留漂移）。RTX 的正式 plan 要在 WSL 上核對 Mac 的 pilot，所以凍結 commit 之後先以 `tools/rtx/push_session.sh stage_e <Mac pilot id>` 把它（不含 raw/）放到 WSL。兩個平台共用 harness 與 `[e3]`，所以任一平台的正式 plan 都要兩個 freeze 旗標都是 true、兩個 pilot 都通過：先跑完的平台不會因為另一個平台的 pilot 逼出的 harness 修正而無法分析、也無法重跑。通過條件（`stage_e3.pilot_problems`）：session 跑完；每個 job ok（含每列的模型身分）；8 個 GGUF 的 sha256 與 header 都相符；env 有效（Mac 為 Apple 140W 並記錄協商瓦數；RTX 已量 B 且沒有 drift）；plan 是 build_plan 在此設定下的 pilot plan（含上限與參數數）；記錄的 `harness_sha256` 等於目前的 harness。

## 4. 時數

| 部分 | GPU（RTX） | Mac |
|---|---|---|
| E1-D | 0（取回時桌機要開機，不需要 lock） | 桌面分析 |
| E1-P | smoke 0.2 h；正式約 2.5 h（上限 3.0 h，含 smoke） | — |
| E2 | smoke 約 0.5 h；正式約 4.2 h（上限 7.5 h） | — |
| E3 | pilot 0.1 h；正式約 0.4 h（上限 1.0 h） | GGUF 0.2–0.3 h；pilot 0.1 h；正式約 0.45 h（上限 1.5 h） |
| 合計 | 預期約 7.9 h；上限合計約 12.1 h | 約 1 h |

037 給 Stage E 的 GPU 預算是 10–15 h，預期值與上限合計都在內。

## 5. 揭露

1. 設計者看過 Core、Advanced、C0、Stage C 的全部結果、第 2 版的唯讀整理、第 2 版審查的 44 項 finding 與其中的數字，以及第 3 版設計 workflow（3 位設計者、3 位評審）的重算。
2. 設計前已看過的重算結果：21 段滑移與起始 idx；各位置的偏差；F 尖峰與滑移 8 / 8 對 0 / 208；正常 3 個、滑移 4 個 decode-only step；T 與 TG 的 F 尖峰；Core A 的 12 個殘差、各格式的 a_f、κ、φ；Stage C 的 M2K；uniform 的 throughput 比例；025 的尾端分解（`results/advanced/formal-20261011-adv/analysis.json`）。
   可以重跑的腳本與輸出 commit 在 `results/stage_e/prereg-tally/`（`descriptive_only`）：滑移與起始 idx、step 層級、Core A 的擬合、Stage C 的 M2K。
3. 看過數據之後才定的門檻：F > 35 ms（已見正常組與尖峰之間的空隙 29.1–39.2 ms）；κ（Phase 1 為 Mac 0.996、RTX 0.178 ms）。在看 E2 數據前、由審查建議的門檻：m ≥ 0.20。事先已存在的：170 ms（028）。沒有對應數據時定下的：β 的 0.20、E1-P 的 J 門檻。
4. 已知或幾乎必然的結果：H-E2-0 在 C 25（+445%）；H-E2-R（025 為 N）；H-E3a；H-E2e（只要有命中，每個 request 的 prefill 從 1,920 降到 336 token）；E1-D 的鏈大部分已在 derived 數據中看到。
5. H-E3b 的類比：Phase 1 的 Q8_0 − Q4_K_M 殘差差，Mac 為 4B −1.47、9B −1.34 ms（超過 max(2 SE, κ)），RTX 為 −0.11、−0.12 ms（小於 κ）。Q8_0 不是 Q4_0，只是類比，不改變「不預測」。
6. Phase 1 的截距同質性：各格式的 a_f 不同質（χ²：Mac 6.8、RTX 28.4，2 df）；這是 3.1 中「固定開銷或頻寬效率」分不開的理由。
7. 「uniform 意義下的飽和就讓 T 無效」已有反例：8K C 8 的 uniform throughput 為最大值的 0.963，和 2K C 25 的 0.967 幾乎相同，但 025 中 8K 的 T 是 M。
8. Stage C 的 M2K（`formal-20261017-c1`，不同 session）：total token throughput 在 C 1、2、3、4 為 2,093、2,950、3,474、3,826 tok/s，是 B0@25 mixed（4,852）的 0.43–0.79；短類 P95 為 327、638、6,614、6,656 ms；同 session 的 U2K 在 C 1、2、4、8 為 327、639、962、969 ms；M2K 在 C 4 的 KV 峰值 0.583。所以 C 8 的 mixed 可能接近飽和，這是 S 只能條件式解讀的原因。
9. 025 的 K3（C 25）：ρ 1.695 → 1.904，noise 門檻 0.0039；a 0.123，未達 A7(i) 的 0.20。第 3 版不重跑 K3 / K35@25。
10. 第 2 版的數字更正：C 4 → C 8 的 P95 為 969 → 977 ms（均值）；2 長 + 6 短為 174 / 247 個 block（不是 156），3 長的峰值 233。
11. Phase 3 在 Mac 上的數字是 140W adapter 的條件（實際協商 94 W），不和 Phase 1 的 70W 數字比較。
12. E1-D 與 E3 harness 的審查（凍結前）用 committed 的 derived 數據重算過：T、TG 上 `classify_bench` 標出的段與 20 次 F 尖峰的位置、C0 前兆組的 min_j、§1.2 的其他數字（全部重現）。這些都不是新的 raw 數據。
13. E1-D 凍結前的覆核用已看過的 B0、G、K3 steps（設計 tally 用過的本機副本）算了 F 尖峰的錨定計數，錨點因此改變：原本的窗 (arrival, s] 在 207 個正常組中有 189 個 ambiguous，8 個尖峰中沒有一個是「明確地多 1 個 step」；改為 (arrival, s + 5 ms] 之後，9 個 run 的中位數都是 2，8 個尖峰都是 3（中位數加 1），沒有 ambiguous，尖峰的 margin 最小 2.61 ms。T、TG 的 steps 尚未取回、沒有看過。

## 6. 相對 037 範圍的決定（等使用者決定）

- **A1. E1 的觸發點**：037 核准「找觸發點」（照 028 的方式做探索性 GPU 工作）。
  - (1) 桌面分析加換 seed 探針，約 2.5 h GPU（上限 3 h）。**建議。**
  - (2) 只做桌面分析；觸發的鎖定方式留在 Future Work（相對 037 的縮減）。
  - (3) 先做桌面分析，探針之後另寫 decision log 再決定。
  - 不論哪個：032 的「偏高比例與 Phase 1 不同」不做（需要多個 session），frontend 延遲的根本原因不在 Stage E。
- **A2. E2 的 allocator 操弄**：024 / 025 指的是 KV block allocator，在 vLLM 0.30.0 沒有旋鈕（block 528 由 hybrid alignment 固定），037 也不修改 vLLM。
  - (1) 記為「未操弄，未檢驗」，每格附浪費上界與 unexplained-break 份額。**建議。**
  - (2) 另加 PyTorch allocator（`expandable_segments`）的探索格（B0@25 的 uniform 與 mixed，6 次 boot，約 0.6 h），明寫它不是 024 / 025 的 allocator，也沒有機制說明。
- **A3. E2 的 prefix caching**：037 把它放在「H-B4 成因」下，但所有基準都關閉 PC，它不可能是已觀察到的 H-B4 的成因。
  - (1) 做成獨立的能力檢查（2.8），標明幾乎必然成立。**建議。**
  - (2) 另在標準 2K workload 上開 PC（6 次 boot），只能檢驗 align 切分，沒有對應的假說。
  - (3) 不做 PC。
- **A4. E2 能交付的範圍**：037 要「2K 的 H-B4 成因」；第 3 版只能檢驗 T 的失效是否依賴 KV、較小的 threshold 是否有效，在 C 25 無法以移除確立成因。
  - (1) 接受，E2 可能以條件式結論或「成因未確立」結束。**建議。**
  - (2) 加更多 C 或 K 的格子（每格約 0.5 h）；不建議，仍做不出「高負載、KV 不受限」的格子。
- 在 037 範圍內的變更（只告知）：E2 刪除 K3 / K35@25，「其他 KV 劑量」改由 K2.5@8 回答（H-E2k）；E3 以 4B-Q4_0 取代 4B-Q5_K_M、只量 L 2048、凍結拆成兩個平台。

## 修訂紀錄

### 第 1 版 → 第 2 版（2026-10-23）

一個唯讀的 workflow（4 個讀者整理凍結的 Advanced、C0、Core A 與 guard，1 個綜合者）指出第 1 版的 26 項問題，第 2 版逐項處理（E1 改為桌面分析；E2 改用 C 8、加 T256 與 K3、刪除 allocator 操弄；E3 在同一 session 擬合；GGUF 由 BF16 GGUF 量化等）。詳細的對照表見 git 歷史中的第 2 版。

### 第 2 版 → 第 3 版（2026-10-23）

- 一個審查 workflow（5 個審查者，每項由兩個 agent 嘗試反駁）找到 49 項，44 項確認。主要問題：
  - E2：解讀表缺 H-E2b 前提與 NT，第 2 列接受和飽和預測矛盾的結果；H-E2d 分不開 T × KV 的交互作用，而且結果從 025 已可預知；「飽和」沒有操作定義，mixed 在 C 3–4 就接近 C 25 的吞吐；PC 的處理也改變 GDN 的 cache 模式，命中檢查太寬；兩個數字錯誤。
  - E1：需要的欄位不在 Mac；g* 與「一個 step」沒有定義；三類事件有一類永遠成立、彼此不互斥也不窮盡；多數決規則沒有定義；起始 idx 的集中被拿掉。
  - E3：H-E3a 沒有定義而且不比 10% 嚴；H-E3b 的 noise 沒有定義；沒有解讀表；bytes 的描述正負號相反；沒有有效性與 NT 規則；8K / 32K 不服務任何 hypothesis。
  - 流程：刪除 allocator 操弄與縮小 E1 應由使用者決定；E3 的容許值是看過逐設定殘差後才定的；唯讀整理沒有 commit，無法稽核。
- 一個設計 workflow（3 位設計者分別以最小改動、機制優先、效度優先提出第 3 版，3 位評審重算數字後評分）一致建議以效度優先的版本為骨架，移植其他兩版的優點；第 3 版即依此寫成，並修正評審重算時找到的數字錯誤（A_prior 漏了 idx 64、正常組 F 的最大值、Core r2 的 k 9 與 11 不相鄰等）。
- 第 2 版的 E2（C 8 的 H-E2a、H-E2d 的 KV 劑量、解讀表）、E1 的 g* 與三類事件、E3 的 H-E3a / H-E3b 與 4B-Q5_K_M 全部被取代；相對 037 的範圍差異改列為第 6 節，等使用者決定。

### 第 3 版的 harness 審查（2026-10-23，凍結前）

- 一個審查 workflow 檢查 E1-D 與 E3 的 harness（審查者加反駁者），確認 34 項。E1-D 的處理：
  - 取回的檔案改從 `fetch_raw.sh` 實際寫入的 `results/capacity/c0-step0/raw/<session>/` 讀；取回清單只列讀者真的會開的檔（T、TG 的 spans 與 bench.json 不再取回）。
  - T、TG 不跑 `classify_bench`（它在 T、TG 上標出的段不是 §1.2 的滑移，而且會把 20 次 F 尖峰藏進「滑移組」）；改為所有 Advanced run 的 F 尖峰描述，n_dec 錨定在 8g 自己的 prefill 開始，並列 margin（1.2、1.4）。
  - 送出延遲依時間排序配對，每段列出；k 24 的段不計入鏈的判定；缺檔只標 blocked。
  - `freeze.e1_harness_sha256` 與 `[e1]` 的值固定；`fetch-list` 也受凍結限制；取回只在 GPU 空閒時；輸出限制在 `results/stage_e/<session>/` 的新檔並記錄輸入的 sha256（1.7）。
  - 1.2 的數字更正：C0 前兆組的 min_j 是 −335.4 至 −330.8 ms（原文的 −322.2 是全部 offset 的上界）。
  - 鏈的判定只用 B0、G、K3 是範圍上的選擇；T、TG 的尖峰率另列，並寫明把它們計入時 L1 的誤報率約 5.9%（1.5）。
- E3 的處理：
  - 8 個設定的 header 值（token_embd bytes 與類型、tie、`output.weight` 類型、tensor 類型分布、model_size）、model_type 與參數數都 pin 在設定檔；session 前核對 8 個檔的 sha256 與 header，每列核對模型身分（3.2、3.6）。
  - 上限只算量測時間，充電暫停扣除，每個 job 的 timeout 受剩餘上限限制；B 在建立目錄之前量（3.6）。
  - 重跑上限可以強制：先寫 session.json stub、目錄不可巢狀、runner 與分析都數嘗試次數（3.6）。
  - pilot 證據加上 env 有效、Mac adapter 與瓦數、RTX 的 B drift、8 個檔與 header、上限與參數數（3.7）。
  - 分析補上共用斜率擬合、a_f 兩兩差距、兩種 φ 的定義（3.5）；任一平台 incomplete 時不寫一般結論；H-E3b 的 NT 依賴改寫為 6 個舊設定加兩個 Q4_0（3.6）。
  - `produce_gguf_e3.py` 驗證 Mac lock 的擁有者，並加上把兩個 Q4_0 傳到 WSL 的 `sync`。
- 修正後的覆核（審查者加反駁者，確認 9 項，其中 1 項 must-fix）：
  - E3（must-fix）：plan 原本記錄設定檔 bytes 的雜湊，另一個平台之後凍結會讓已完成的正式 session 無法分析，而重跑上限又讓它無法重跑；改為不含 `[freeze]` 的內容雜湊，分析不比 `pins_sha256`（3.6）。
  - E3：pilot 也必須是 build_plan 的 pilot plan；嘗試次數在 `session.json` 讀不到時以 `plan.json` 判定；`session.json` 改為原子寫入（3.6）。
  - E1-D：F 尖峰計數的 margin 改為兩個窗邊、兩側的距離，低於 1 ms 標 ambiguous（1.4）；取回時 sha256 不符（`.sha256-mismatch`）也停止分析（1.5）；測試改為 client 與 server 的時間不同、g 0 亂序完成，核對送出延遲、C0 的 L4 與代理量的 index，覆核者的 10 個突變全部被抓到（1.7）。
- E1-D 凍結前覆核（逐條對照 1.3–1.7、在已看過的數據上重現 1.2、檢查取回後才有的檔案格式；確認 5 項，其中 2 項是同一個 must-fix）：
  - must-fix：F 尖峰錨定計數的窗終點原本是 s，但時間戳是 runner 讀到 log 的時刻，8g 被排進之前那個 step 的時間戳落在 s 的 −0.9 至 +3.1 ms，所以幾乎每組都是 ambiguous，而且少數「明確」的組計數偏少。改為 (arrival, s + 5 ms]，並在 §5 第 13 項揭露這是看過 B0、G、K3 的 steps 之後的改變（1.4）。
  - §1.2 的 Q 更正為每個 request +347–351 ms（每段平均 ΔQ 348.8–349.6 ms），和 028 一致。
  - T、TG 缺 steps 時的 blocked 標籤不再寫 L2；測試的 spans 改用 receiver 的實際格式並加入 warmup span。
  - 其餘：1.2 的 21 段、起始 idx、F、間隔、n_dec、送出延遲、C0 的 L3 全部重現；取回檔案的格式假設（span id、e2e 單位、warmup span、bench.json 的 start_times 與 latencies、manifest 鍵名、fetch_raw.sh 的路徑）都成立。
- E1-D 凍結前的最後一輪覆核（新的錨點在 T、TG 上是否穩健；以真實格式的替身檔跑完整的 analyze 與 CLI）：沒有阻擋凍結的問題；確認 2 項 minor，凍結前處理：
  - `--out` 只接受新的 session 目錄（原本可以把檔寫進既有的目錄）；
  - `fetch-list` 改為每個 session 一行、可直接當 `fetch_raw.sh` 的參數。
  - 另外加上兩個描述欄位：g 1–23 的計數分布與「最近時間戳減 s」的範圍，讓 T、TG 的時序是否和 B0、G、K3 相同可以直接看出。
  - 覆核確認：下一個時間戳至少在 s 之後 44 ms（不會多算下一個 step）；position 0 的 Q 在 15 個 run 都 ≤ 0.015 ms；T、TG 正常組的 F 為 24.8–29.1 ms，和 B0 相同。
- 之後凍結 E1-D（1.7）。

### E2 harness 審查（2026-10-23，smoke 與凍結之前）

- 一個審查 workflow（3 位審查者分別看 plan、runner、analysis，每項由一個 agent 嘗試反駁）確認 18 項，沒有被反駁的；全部處理：
  - smoke 沒有綁定設計：縮減的 smoke plan、改過的 expected_blocks、別的 dataset 都能通過凍結 → smoke plan 必須等於 build_plan 的，共享前綴 dataset 先 pin，凍結證據逐項核對（2.10）。smoke 改為 12 次 boot，涵蓋設計中每個 (condition, mix)（原文「約 9 次」）。
  - smoke 的通過條件缺 A10、teardown、mamba_cache_mode 與記憶體增量（原本記錄的其實是 KV 使用率）；PC 不跑 A10 → 全部補上，PC 用 B0 的 knob、(c) 豁免（2.10）。PC 的 block 數固定為 248（2.3、2.10）。
  - 「連續 3 次 boot 沒有就緒時停止」沒有實作；重跑上限沒有強制；session 目錄可以巢狀 → 實作，並比照 E3 寫 stub 與數嘗試次數（2.9）。
  - R、Λ、L̄q 沒有計算，R3 的子標籤只看 preemption；描述量大多沒有報 → 實作並定義計算方式（2.2、2.7）。
  - incomplete 把 H-C0 失敗當成 NT，也沒有照「NT 能否改變主列」判斷；β_ok 為 NT 時被讀成不成立；H-E2k 在 ρ_8 ≤ 0 時仍判定；H-E2-0 沒有三值判定；T256 的「只適用於 T512」附註沒有加 → 全部依 2.6、2.7 修正。
  - smoke 的延遲遮蔽不足 → 補上（2.10）。

### E3 與 E2 的修正後覆核（2026-10-23，pilot 之前）

- E3（確認 4 項，1 項 must-fix）：
  - must-fix：兩個平台各自凍結，但 harness 與 `[e3]` 是共用的；先跑完正式 session 的平台，會因另一個平台的 pilot 逼出的 harness 修正而無法分析、也無法重跑 → 任一平台的正式 plan 都要兩個旗標與兩個 pilot（3.7、§0 的表）。
  - pilot 把兩個 Q4_0 的 tok/s 寫進會 commit 的 rows.jsonl → pilot 的 rows.jsonl 只留身分與結果（3.7）。
  - 存在 `results/stage_e/` 的 plan 檔會被算成作廢的嘗試 → 只有 plan.json 的目錄不算，plan 指令拒絕寫進 `results/stage_e/`。
  - 分析的 `--out` 可以覆寫既有 session → 只接受新的 session 目錄，並記錄輸入的 sha256；E2 的分析同樣改為寫到新的 session 目錄（§0）。

### 第三輪覆核（2026-10-23，smoke 與 pilot 之前）

- E2（確認 4 項）：R3 的 HOL 路徑與描述量用了 A10 無效的分解份額、描述量在少於 3 次 restart 時仍報平均 → 依 2.2、2.7 修正；bench 後的 nvidia-smi 失敗會中止整個 session → 改為記錄錯誤；smoke 沒有核對 vLLM 版本 → smoke 也核對，凍結證據加上版本。
- E3（確認 3 項）：pilot 的逐 job env capture 帶有 job 的時間 → 移到 `raw/env/`；RTX 的正式 plan 需要 Mac 的 pilot 在 WSL 上，沒有支援的步驟 → `tools/rtx/push_session.sh`（3.7）；一個測試沒有真正改到 `[freeze]` → 修正。

### 第四輪覆核（2026-10-23）

- 確認 3 項（沒有 must-fix）：E2 判斷「沒有啟動 server」用的是有沒有 run 紀錄，server 啟動失敗時仍算一次嘗試 → 改為記錄 `server_started`（2.9）；`push_session.sh` 會把 macOS 的 AppleDouble 檔與 xattr 一起送到 WSL → 關掉；Stage E 在 WSL 上的 session 只有取回後才看得到 → `tools/rtx/fetch_all_sessions.sh` 與 commit 的清單（§0）。
