# 041 — Stage E 事先登記（DRAFT，第 2 版）：三個機制問題

- 狀態：**DRAFT**。harness、smoke / pilot 與審查完成後，三個部分各自凍結；凍結前不收任何 Stage E 的正式數據。
  - E1 是探索性的桌面分析（比照 028），E2、E3 是正式的。
- 日期：2026-10-23（第 1 版）；同日修訂為第 2 版，修訂內容見文末「修訂紀錄」。兩版之間沒有任何 Stage E 數據。
- 對應：
  - 037（Phase 3 範圍）、024 / 025（Advanced）、026（Future Work）、028 / 032（C0）、PREREGISTRATION §2（H-A2）；
  - 本文的 PREREGISTRATION 摘要是 §11（DRAFT）。
- 時點：任何 Stage E 數據之前。設計者看過 Core、Advanced、C0 的全部結果。第 2 版另外看過一次唯讀的整理（重算既有數據的描述統計），用到的地方在「揭露」寫明。

## 範圍

| 部分 | 問題 | 性質 | 平台 | 凍結 |
|---|---|---|---|---|
| E1 | 8K 延遲滑移的觸發點（032：滑移的機制已確立，觸發點未知） | 探索性，桌面分析（不用 GPU） | Mac | `freeze.e1` |
| E2 | 2K 的 H-B4 成因（025 為 R6）：飽和假說、chunk threshold、KV 劑量；另加 prefix caching | 正式 | RTX | `freeze.e2` |
| E3 | H-A2 被推翻的原因（Core A：bytes 少的設定都比頻寬比例預測的慢） | 正式 | Mac 與 RTX | `freeze.e3` |

- 三個部分各自凍結（和 Stage Q3 相同）：E3 需要 Mac lock，不應拖住 E2。
- 設定與 pin 都在 `configs/stage_e.toml`；`configs/pins.toml` 與 `configs/capacity.toml` 已凍結，不寫入。
- 凍結的 Core A、Core B、Advanced、C0、Stage C 模組只 import，不修改；沒有 guard test 鎖住的匯入模組列入各部分的 `harness_sha256`。

## E1：8K 滑移的觸發點（探索性桌面分析）

- **已知**（032；第 2 版的唯讀整理重算，見「揭露」）：
  - 每段滑移都是同一組（每 8 個 idx 一組）的第 1–7 個 request 多等一個 prefill step。Advanced 的 step 紀錄量到約 349 ms；C0 的 TTFT 差為 337–338 ms。
  - client 端沒有延遲：每個 request 都在前一個完成後 0.05–0.41 ms 送出（Core 的 C 8 原始數據）。
  - 滑移那一組到得**早**，不是晚：第 8k+1 個 request 在第 8k 個之後 90–95 ms 送出，平常是 410–426 ms。
  - 前兆在前一組（k − 1）：該組的第 1–7 個 request 比第 0 個早完成 322–335 ms；其他 547 組的這個差最多 17 ms。
- **第 1 版的問題（client 還是 server）已經由既有數據回答**：所有已知的滑移都屬於 server 類，client 類在結構上不可能出現。
  所以第 1 版的 GPU 探針（6 次 boot）不再有資訊價值，刪除。
- **第 2 版的問題**：前一組的不同步從哪裡開始？往前追溯，第一個讓同一組的完成時間散開超過一個 step 的事件是什麼？
- **方法**（只用已 commit 或本機 archive 中的數據，不用 GPU）：
  - 數據：
    - Advanced 的 8K uniform（B0、G、K3，每個 3 次 restart）：spans 與每個 step 的紀錄；
    - C0（`explore-c0-20261016`）與 Core B 的 C 8（`formal-20261009-coreb-uniform`）：逐 request 的 TTFT、延遲與送出時間。
  - 每段滑移（以 `c0_analysis.classify_bench` 判定，和 032 相同）計算：
    - 送出延遲 = 送出時間 − 釋放它的那個 request 的完成時間；
    - 第 k − 1、k − 2、… 組內的完成時間散布（第 j 個減第 0 個），往前追到散布 ≤ 一個 step 的那一組 g*；
    - 有 step 紀錄的（Advanced）：g* 到 k 之間每個 step 的 prefill / decode token 數與各 request 的狀態。
- **事先寫定的描述**（不是檢定）：
  - 每段滑移列出 g*、k − g*、g* 的散布開始於哪一種事件（以下三類，其他列為「其他」）：
    - **prefill 分段**：同一組的 prefill 被 step 的 token 預算（2,048）切到不同 step；
    - **完成時間**：同一組中某個 request 的 decode 多一個 step；
    - **到達**：同一組的 request 跨過 step 邊界才到達。
  - 有 5 段以上而且 ≥ 80% 屬於同一類時，報告該類為「主要的前兆」；否則報告「無法判定」。
- **限制**：Advanced 開了觀測 flag（024），028:66 刻意在 C0 關掉它；Advanced 的第一個 bench 也有滑移（9 次 boot 中 5 次），C0 沒有。兩組數據都報告，不合併。
- **結果**不改變任何既有判定，只寫進 Future Work 的說明。若要用新的探針驗證前兆（例如改變到達的偏移），需要新的 decision log。

## E2：2K 的 H-B4 成因（正式）

025 在 2K 的分解中，budget-long 占尾端 ΔQ 的 0.91，但 T = 512 沒有移除 ρ（f 0.08，類別 N）。
025 事後寫下一個未檢驗的可能解釋：**C 25 時系統已經飽和**，T 只是重新分配每個 step 的 token 預算，沒有減少 mixed 的總 prefill 工作量。E2 直接檢驗它。

- **024 的規則全部沿用**：
  - 主要量：短類 P95 TTFT；U、M 為 3 次 restart 的平均；ρ = ln(M ÷ U)；f = 1 − ρ_c ÷ ρ_0。
  - ρ_0 是**同一 session、同一 C** 的 B0（024:75、:450：因果對比一律以 in-session 的 B0 為參照）。
  - 類別 M / P / R / W / N / NT 與 noise 檢查用 `adv_analysis` 的凍結函式（`cell_stats`、`rho_stats`、`removal_category`、`exceeds`、`beyond_noise`）。
  - H-C0 的閘門：每個 C 先以 `core_b_analysis.h_b4_values` 確認該 C 的 B0 有 H-B4 現象（ρ_0 超出 noise）；否則該 C 的所有假說為 NT。
  - 有效性：024 的 A1 與 031（閒置判定、spill）；每個 cell 少於 3 個有效 run 時為 NT（013）。
- **格子**（全部 2K、uniform 與 mixed、各 3 次 restart；server 都是 024 的觀測版）：

| cell | condition | C | 用途 |
|---|---|---|---|
| B0@8、T@8 | B0、T（512） | 8 | H-E2a（未飽和時 T 是否移除 ρ） |
| B0@25、T@25 | B0、T（512） | 25 | H-E2b（在同一 session 重現 025 的 N） |
| T256@25 | T（256） | 25 | H-E2c（更小的 threshold） |
| K35@25、K3@25 | KV 3.5 GiB、3 GiB | 25 | H-E2d（KV 劑量） |
| PC-off、PC-on | prefix caching 關 / 開 | 25 | H-E2e（共享前綴，只用 uniform） |

  - 共 7 × 2 + 2 = 16 個 cell、48 次 boot；每個 restart 區塊內的順序以 seed 打亂（同 024）。預估約 4.3 h。
- **C 8 的理由**（取代第 1 版的 C 12）：
  - Core 的 2K uniform 在 C 4–8 的短類 P95 TTFT 幾乎不變（971 → 977 ms），throughput 為最大值的 0.745，是量過的未飽和點；C 12 從沒量過，而 C 16 的 P95 已加倍。
  - **混淆**：C 8 同時降低了 KV 壓力（2 個長類加 6 個短類約 156 個 block，遠低於 248）。所以 H-E2a 單獨無法分開「飽和」與「KV」，要和 H-E2d 一起解讀（見下表）。
  - mixed 在 C 8 的 N 為 200，長類的數量依 007（`mixed_counts`）。需要另外產生一份 2K、N 200 的 mixed dataset，sha256 pin 在 `stage_e.toml`。
- **Hypotheses**：

| id | 內容 | 預測 |
|---|---|---|
| H-E2a | C 8 時 T 的類別 ∈ {M, P}（024 A5；R 不算） | supported |
| H-E2b | C 25 時 T 的類別為 N（在同一 session 重現 025） | supported |
| H-E2c | C 25 時 T256 的類別 ∈ {M, P} | not supported（若飽和，任何 threshold 都只是重新分配） |
| H-E2d | C 25 時 ρ 隨 KV 減少而增加：ρ(K3) 與 ρ(K35) 依序超過 ρ(B0)（`exceeds`，兩兩比較） | 不預測方向，照實報告 |
| H-E2e | 共享前綴的 uniform：PC 開時短類 P95 TTFT ≤ 0.8 × PC 關，且超出 noise | supported |

- **事先寫定的解讀**：

| H-E2a | H-E2c | H-E2d | 解讀 |
|---|---|---|---|
| supported | not supported | not supported | 符合飽和假說：未飽和時 T 有效，飽和時任何 threshold 都無效，KV 劑量沒有關係 |
| supported | 任何 | supported | 飽和與 KV 無法分開，兩者都可能 |
| not supported | 任何 | 任何 | 不支持飽和假說：未飽和時 T 也無效 |
| 其他組合 | | | 照實報告，不解讀 |

- **condition 的定義**：
  - T、T256：`--long-prefill-token-threshold 512 / 256`，加上 024 的 chunk 檢查（每個 step 的 long prefill 不超過 threshold）。
  - K35：`--kv-cache-memory-bytes` 3.5 GiB（217 個 block，由 smoke 量測並 pin）；K3 為 024 的 K3（186 個 block）。
  - PC：B0 的 argv 中唯一的 `--no-enable-prefix-caching` 換成 `--enable-prefix-caching`；PC-off 與 PC-on 其他設定完全相同，dataset 也相同。
  - 每個 boot 從 server log 核對 non-default 參數（含 `enable_prefix_caching`）與 block 數；PC 另外以 `/metrics` 的 prefix cache 計數確認：PC-on 有命中，PC-off 沒有。
- **E2e 的 dataset**：每列 1,920 token，其中前 1,584 token（3 × 528，正好 3 個完整的 block）所有列共享，後 336 token 各自不同；N 250 + warmup 25 列，sha256 pin 在 `stage_e.toml`。
  - E2e 是另一個問題（prefix caching 對短類的效果），不是 H-B4 的成因；它沒有 mixed。
- **逾時**：沿用 030 / 039 的公式 `capacity_plan.bench_timeout_s(2048, mix, C, τ1)`，T 與 T256 乘以 2（039 的先例）。

## E3：H-A2 被推翻的原因（正式）

- **已知**（Core A）：推翻 H-A2 的配對是 4B-Q4_K_M 對 9B-Q8_0，觀測比 2.22–2.24，預測比 2.99，偏差約 −25%；兩個平台的 15 組配對中，Mac 13 組、RTX 15 組偏差為負。
- **兩個候選機制**：
  - **固定開銷**：每個 token 有一段和權重 bytes 無關的固定時間（kernel launch、CPU 端），t = a + b × bytes。
  - **反量化成本**：k-quant 每個 byte 的計算量較高，所以 Q4_K_M 等設定比 bytes 預測的慢。
- **方法**：
  - **新 GGUF**（在 Mac 上，Mac lock）：由 pins 的 BF16 GGUF 以 llama.cpp b11146 的 `llama-quantize` 產生：
    - 9B-Q4_0：bytes 和 9B-Q4_K_M 相近，但反量化簡單。預設的配方會把 `output.weight` 留在 Q6_K（約 16% 的 bytes），照實揭露並 pin 住 tensor 類型的分布；
    - 4B-Q5_K_M：bytes 介於 4B-Q4_K_M（2,708,804,544 B）與 4B-Q6_K（3,464,055,744 B）之間的 k-quant。
    - 兩個檔案的 sha256、大小與 tensor 類型 pin 在 `stage_e.toml`，再以 sha256 驗證後傳到 WSL。
  - **量測**：每個平台一個 session，量 8 個設定（4B 與 9B 各自的 Q8_0、Q6_K、Q4_K_M，加上兩個新的）。
    - 用 Core A 的 decode 協定（同一 runtime、flag、context 2048 / 8192 / 32768、3 次重複、sentinel），順序以 seed 打亂；
    - **所有比較都在同一 session 內**（037:95）：擬合只用該 session 的 6 個舊設定，不用 Phase 1 的數據。
  - **bytes 的定義**：`core_a_analysis.decode_bytes`（檔案大小，加上沒有 tie 時的 token embedding），和 Core A 相同。
  - **擬合**：每個平台以該 session 6 個舊設定在 L 2048 的 decode 時間擬合 t = a + b × bytes（OLS），預測兩個新設定。三個 L 的擬合只作描述。
- **Hypotheses**：

| id | 內容 | 預測 |
|---|---|---|
| H-E3a | 4B-Q5_K_M 的預測誤差 ≤ 6 個舊設定的最大樣本內殘差（絕對值）加 2 個 SE | supported |
| H-E3b | 9B-Q4_0 的殘差減 9B-Q4_K_M 的殘差 < 0，而且超出 noise（兩者 SE 的合成） | 不預測：固定開銷模型預測兩者相近，反量化成本預測 Q4_0 較負 |

  - 兩個平台各自判定，不合併。
- **Mac 的條件**：Mac lock、Apple 140W adapter、同一 session 一個電力模式（037 §4）、012 的電量與 thermal 規則。凍結的 `run_core_a` 要求 70W adapter，所以 E3 有自己的 runner（重用 `run_core_a` 的 job 格式、解析與 env capture）。
- **成本**：GGUF 產生約 0.2 h（Mac）；正式 session Mac 約 2 h、RTX 約 1.2 h；pilot 各約 0.2 h。

## 揭露

- **作為依據**：
  - Core A 的 decode 結果用於 E3 的配對、bytes 範圍與擬合方法；Advanced 的 2K 結果用於 E2 的格子；Core 的 2K uniform 曲線用於選 C 8。
  - 第 2 版之前，一次唯讀的整理（2026-10-23）重算了既有數據的描述統計：
    - E1 的送出延遲、到達間隔與前一組的完成散布（見 E1「已知」）；
    - Core A 的樣本內擬合：Mac a 5.944 ms、b 3.800 ms/GB，最大殘差 5.5%；RTX a 1.876 ms、b 1.183 ms/GB，最大殘差 2.0%（L 2048）。
      H-E3a 的容許值因此改為和該 session 的樣本內殘差掛鉤，不用第 1 版的固定 10%（兩個新設定都落在舊設定的 bytes 範圍內，10% 沒有鑑別力）。
- **E1 的答案**：第 1 版問的 client / server 已由既有數據回答（全部是 server）。這是重新分析既有數據，不是新的檢定。
- **不重新分析**：H-A2、H-B4 與 Advanced 的判定不變。

## 原本尚未決定的項目（第 2 版定案）

- harness 的切分：每個部分自己的模組；凍結的模組只 import 或繼承。
- E2e 的 dataset 與 C 8 的 mixed dataset：在 WSL 產生（vLLM 的 RandomDataset，每列重新 tokenize 到目標長度），sha256 pin 在 `stage_e.toml`。
- E3 的 GGUF：由 pins 的 BF16 GGUF 量化，不需要 safetensors（Mac 上的 safetensors 已在 2026-10-08 刪除，見 `results/artifacts/deletions.jsonl`）。
- **仍須在 smoke 確認的風險**（凍結前；不成立時另寫 decision log，不臨時改設計）：
  - vLLM 0.30.0 對 hybrid（GDN）模型的 prefix caching：是否只在 block 邊界重用、是否改變 chunk 的切分（024:194）；
  - K35 的實際 block 數（預期 217）與 PC 的 block 數；
  - C 8 的 mixed 與 E2e 的 dataset 在 vLLM bench 中的 warmup 取法。

## 修訂紀錄（第 1 版 → 第 2 版，2026-10-23，沒有任何 Stage E 數據）

一個唯讀的 workflow（4 個讀者整理凍結的 Advanced、C0、Core A 與 guard，1 個綜合者逐項核對）指出第 1 版有下列問題。全部在第 2 版處理：

| # | 第 1 版的問題 | 第 2 版 |
|---|---|---|
| 1 | E1 的 client / server 分類已由既有數據決定，client 類不可能出現 | E1 改為追溯前兆的桌面分析，刪除 GPU 探針 |
| 2 | 「約 349 ms」出自 Advanced 的 step 紀錄，不是 C0 | 改寫來源 |
| 3 | 「第一個 bench 不滑移」只在 C0 成立；開了觀測 flag 的 Advanced 有 5 / 9 | 寫入 E1 的限制 |
| 4 | 「和 C0 相同的格子」：flag 與 client 不同 | E1 不再跑新的 bench |
| 5 | 「任何 boot 失敗兩次就停止」語意不清 | E1 不再跑 GPU |
| 6 | 037 列的「8K 雙模式」與「其他 chunk threshold」被拿掉，沒有 decision log | E1 的問題就是雙模式的觸發點；E2 加入 T256 |
| 7 | C 12 沒有 Core 依據，「未飽和」不成立，且同時降低 KV 壓力；「025 在 C 25 為 N」是跨 session 的參照 | 改為 C 8（量過的未飽和點），揭露 KV 的混淆，T@25 在同一 session 重現 |
| 8 | H-E2a 的「M 或 P（f ≥ 0.2）」漏了 dρ 與 dM | 改為「024 A5 的類別 ∈ {M, P}」 |
| 9 | C 12 的逾時沒有定義 | 沿用 030 / 039 的公式 |
| 10 | H-E2b 的「<」沒有定義；K35 和 K3 在同一個區間；跨 session 用 025 的 K3 | 改為兩兩 `exceeds`；K3 在同一 session；不預測方向 |
| 11 | 「024 的 A4 比值」沒有 SE 與 noise 規則 | 不再需要跨 session 比較 |
| 12 | E2c 的「allocator」：024 / 025 指的是 KV block 粒度的浪費（沒有旋鈕，block 528 由 hybrid alignment 固定），不是 PyTorch 的 allocator；也沒有機制說明，且有 spill 風險 | 刪除 E2c；allocator 維持「未操弄」，理由寫在本表 |
| 13 | E2d：1,536 token 只有 2 個完整的 block；PC 關用 Core B 的設定；20% 的尺度不明；沒有 mixed | 改為 1,584（3 個 block）、兩邊同一組 flag、線性 0.8 倍加 noise，標明是另一個問題 |
| 14 | dataset 的 sha256 要寫進 pins，但 pins 已凍結 | 改寫進 `stage_e.toml` |
| 15 | E3 寫 safetensors 在 Mac | 改由 BF16 GGUF 量化 |
| 16 | `produce_gguf` 只能產生 pins 的 quant | E3 有自己的產生程式 |
| 17 | E3 和 Phase 1 的數據比較，違反 037 的「同一 session 內比較」 | 兩個平台都重量 6 個舊設定，在同一 session 內擬合 |
| 18 | 凍結的 `run_core_a` 要求 70W adapter | E3 有自己的 runner |
| 19 | H-E3a 的 10% 沒有鑑別力 | 改為樣本內最大殘差加 2 SE |
| 20 | 「bytes」沒有定義 | `core_a_analysis.decode_bytes` |
| 21 | Q4_0 的預設配方把 `output.weight` 留在 Q6_K | 揭露並 pin 住 tensor 類型 |
| 22 | 「介於兩者之間」沒有說是哪兩個 | 寫明 |
| 23 | 「取樣」不適用（llama-bench 不取樣） | 刪除 |
| 24 | H-E3b 的「bytes 比例」沒有說是比例還是 affine | 改為同一 session 的 affine 預測與殘差差 |
| 25 | 「凍結的 C0 / Core A 模組」其實沒有 guard test | 列入 `harness_sha256` |
| 26 | PREREGISTRATION 沒有 §11 | 新增 §11（DRAFT） |
