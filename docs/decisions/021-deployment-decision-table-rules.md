# 021 — deployment decision table 的推導規則（在其餘正式分析之前寫定）

- 狀態：Accepted（2026-10-10）
- 日期：2026-10-10
- 對應：SPEC「最後：deployment decision table；清楚說明什麼 workload 適合什麼配置」；PREREGISTRATION §2–§4；005、007、011、013、019、020
- 時點：
  - 已有的有效正式數據只有 RTX Core A（`formal-20261009-rtx-a1`，analysis 已看過）。
  - Mac Core A（`formal-20261009-a2`）、正式 Core B uniform 進行中；正式 quality、Core B mixed 尚未開始。
  - 本文與 generator、測試都要在這些 session 的分析被看過之前 commit（見 R0）。

## 背景

SPEC 要求最後產出 deployment decision table，但 PREREGISTRATION 沒有規定它的推導方式。
如果等全部數據出來才決定「什麼 workload 用哪一欄、平手怎麼處理、哪些 session 算數」，就有依結果挑規則的空間。
所以在其餘正式分析出來之前，把推導規則寫死成本文與程式。

本文不改任何 hypothesis、門檻、metric 或 protocol。verdict 只照抄，不影響任何推薦。

## 設計過程與揭露

- 規則由 review workflow 設計：三個獨立方案（Pareto 前緣、workload 優先、統計嚴謹）→ 三位評審（方法、實作、讀者價值）→ 綜合。
  - 設計 agent 被要求**不讀 `results/`**，只讀程式與文件。
  - 綜合者在 git context 中看過 commit 標題，包括 RTX Core A 的結果摘要（H-A2 refuted，觀測比 2.24 對預測比 2.99）
    以及 a2 / uniform 的進度。
  - 主 session（我）看過 RTX Core A 的 `analysis.json`，也產生過 RTX 的 decode / prefill / memory 圖。
- 規則中沒有任何一條讀取 verdict 或 H-A2 的數值，也沒有任何一條依賴「有沒有 spill」。
  統計嚴謹方案中「H-A2 refuted 時另外顯示 spill 速度條件」那一條，是在知道 RTX H-A2 結果之後寫的，所以刪除。
- 綜合者留了三個問題，由我決定（理由見各條）：
  1. 揭露與時機：照實揭露（本節），並遵守 R0。
  2. RTX 的記憶體軸用 VRAM（Σ devices.self），不用 device + host 合計（R6）。
  3. Core B 延遲敏感段的描述性 C 前緣（frontier_C）**刪除**，只保留完整曲線與三個點（R19）。

## 決策（推導規則）

### 流程

- **R0 時機**：
  - 本文、`benchmarks/decision_table.py` 與其測試先 commit，之後才第一次對真實 `results/` 執行。
  - Mac a2、正式 quality、正式 Core B 的 `analysis.json` 要在這個 commit 之後才產生或查看。
  - 第一次執行的輸出照原樣 commit。之後任何規則變更都要另寫 decision log，並標明 post-hoc。
- **R1 選 session（沒有選擇空間）**：
  - CLI 為 `python benchmarks/decision_table.py`，沒有任何指定、省略 session 或改變目錄的參數。
    讀 `results/`，寫 `results/decision_table/`。
  - 掃描 `results/core_a/{macos-metal,wsl2-cuda}/*/session.json`、`results/quality/*/session.json`、`results/core_b/*/session.json`，都只掃一層。
  - 閘門只用既有規則：
    - Core A：`core_a_report.session_problems` 為空，且 `platform` 等於目錄名稱。
    - quality：與 `quality_analysis.main` 相同（formal、env_valid、platform 為 wsl2-cuda）。
    - Core B uniform：formal、env_valid，plan 沒有 `derived_from` 而且所有 run 都是 uniform。
    - Core B mixed：plan 的 `derived_from` 等於選定的 uniform，所有 run 都是 mixed，`core_b_report.report(uniform, mixed)` 不拒絕。
  - 每一類輸入（Core A × 每個平台、quality、Core B uniform、Core B mixed）通過閘門的 session 數：
    - 恰好 1 個：使用。
    - 0 個：該段標 `pending: no valid formal session`。
    - ≥ 2 個：exit 2，不寫任何輸出，要求以 decision log 決定用哪一個。
  - 被排除的 session 只讀 `session.json`（Core B 另讀 `plan.json`），列入 `provenance.excluded` 並附閘門原文。
- **R2 分析檔**：選定 session 的 `analysis.json`（Core B 為 `results/core_b/<uniform>/analysis.json`）不存在時，該段標 `pending: analysis.json missing`。
  generator 不寫任何 `analysis.json`。
- **R3 完整性（fail closed）**：用凍結程式重算後逐項比對，不一致就 exit 2，不輸出部分結果。
  - Core A：以 `tied` 全部為 False、`embd_bytes` 取自存檔的 `subtracted_embd_bytes` 重跑 `core_a_report.report`。
    RTX 的 `platform_params` 用 `pins.stage_b`（因此同時核對存檔值等於 pins）。
    結果必須等於存檔的 platform、platform_params、subtracted_embd_bytes、spill、cells、hypotheses；存檔的 session、mode、env_valid、sentinel 必須等於 `session.json`。
  - quality：以預設的 N_BOOT、SEED 重跑 `quality_analysis.h_a5`，必須完全相等。
    downstream 只重算計算量小的欄位（(task, size, pair) 集合、n、delta_acc、flip_rate、n_boot、seed）；ci95 與 resolved 照抄，標 `ci95 from analysis.json (not re-verified)`。
  - Core B：`core_b_report.report(uniform, mixed 或 None)` 必須等於存檔的 `analysis.json`。
    mixed 出現之後，舊的 analysis.json 必須用凍結 CLI 重產。
  - 因 interpreter 差異而不相等時，唯一的處理是用凍結 CLI 重產 analysis.json，不放寬比對。

### Core A（每個平台、每個 L）

- **R4 候選**：pins 的 size × quant（4B-Q8_0、4B-Q6_K、4B-Q4_K_M、9B-Q8_0、9B-Q6_K、9B-Q4_K_M）。
  這個順序只用於迭代與顯示，從不參與選擇。BF16 只是 quality 參考，不是候選。
- **R5 cell 狀態**（依序判定）：
  1. 不在 cells 中 → not measured。
  2. outcome oom → oom。
  3. outcome error → error（附第一個非 ok 列的 reason 與 stage）。
  4. 在 spill 中 → spill（004：spill 不算 fit）。
  5. decode 或 ttft 缺值、n ≠ 3，或 fit/ok 且帶 memory breakdown 的列數 ≠ 3 → incomplete。
  6. 其餘 → candidate。
  - oom 與 spill 是真實的不可行結果；not measured、error、incomplete 屬於未知。
- **R6 各軸**（只算 candidate）：
  - decode：`cells[k].decode`（越高越好）；TTFT：`cells[k].ttft`（越低越好）；prefill 只顯示。
  - 記憶體：每個 fit/ok process 的值，以 `core_a_analysis._ms` 算 m、SE（越低越好）。
    - Mac：Σ devices.self + host.self（unified memory；等於 `core_a_figures.memory_series` 的合計）。
    - RTX：Σ devices.self（VRAM）。**理由**：RTX 上受限的是 VRAM，004 的 spill 記憶體條件與 `spill_cells` 用的正是這個量；
      host 端（CUDA 下 9B 的 token_embd）占用的是系統記憶體，另外顯示。
    - 兩平台的記憶體依 §3 各自定義，永不互比。
  - 只顯示：合計的最大值、host、external memory（Mac 為 footprint，標「不含 mmap 權重（012）」；RTX 為 `gpu_mem_used_peak_mib` − B），
    以及 RTX 的 spill_margin = V_total − B − C_ctx − max(Σ devices.self)。
- **R7 純量軸的關係**：直接呼叫凍結的 `core_a_analysis._beyond_noise`（§2：|x − y| > 2√(SE_x² + SE_y²)）。
  方向有利且超出 noise → better；反之 worse；其他（含 SE 缺值）→ tie。不另設最小差距。
- **R8 quality 關係**（兩平台共用，唯一來源是 H-A5 的 Holm 結果）：
  - h_a5 不是 not testable 時才可用；否則所有配對都是 unmeasured，並套用 R12。
  - 同 size 相鄰（Q8_0/Q6_K、Q6_K/Q4_K_M）：test 有 reject 且 z > 0 → 低 bit 那個較差；z < 0 → 反向並標 contrary；沒有 reject → tie（not resolved，不做等價宣稱）。
  - 同 size 的 Q8_0 對 Q4_K_M：兩個相鄰關係都 resolved 且同方向才有關係；否則 unmeasured，不把 tie 遞移出去。
  - 9B-Q4_K_M 對 4B-Q8_0（H-A5(a)）：reject 且 mean_delta < 0 → 9B-Q4_K_M 較好；> 0 → 反向並標 contrary；沒有 reject → tie。附 005 的 contamination 註記。
  - 其他跨 size 配對 → unmeasured。
  - 不使用：kld_mean 或 PPL 的直接比較、未校正的 p、ci95、downstream（005：CI 只作描述；跨 size 只能配對比較）。
- **R9 支配**：a 支配 b ⇔ 四軸（quality、decode、TTFT、記憶體）都不是 unmeasured、沒有一軸 a 較差、至少一軸 a 較好。
  靠 quality 的 tie 才成立的支配標 tie_dependent，措辭固定為 `no resolved quality advantage (no equivalence claimed)`。
- **R10 前緣**：支配關係在平手下不具遞移性，可能成環。前緣取支配圖中沒有外部入邊的強連通分量（source SCC）的聯集：保證非空；沒有環時就是一般的非支配集合。
  環的成員標 NOISE_CYCLE，不刪除。被支配的配置列出所有直接支配它的配置與逐軸關係。
- **R11 四個 workload**（主軸各對應 SPEC Core A 的一個 metric，之後先排 quality，其餘照 SPEC 的 metric 順序）：

  | workload | 讀者情境 | 軸順序 |
  |---|---|---|
  | A-Q 品質優先 | 一次一個 request，最在意輸出品質 | quality → TTFT → decode → 記憶體 |
  | A-D 生成速度優先 | 長回答、chat | decode → quality → TTFT → 記憶體 |
  | A-P 首 token 優先 | 長文件、RAG、摘要 | TTFT → quality → decode → 記憶體 |
  | A-M 記憶體優先 | 要為其他程式留空間 | 記憶體 → quality → TTFT → decode |

  - 從前緣開始逐軸篩選：第 k 步只淘汰「被同一集合內某成員在第 k 軸上超出 noise 地勝過，而且若 quality 排在前面，兩者的 quality 關係必須是 tie」的配置。
    unmeasured 永遠不當作平手，所以後續軸不能淘汰 quality 關係未量測的配置（避免「品質優先」因為速度選出品質未知的 size）。
  - 每一步必定非空，並記錄 kept 與 eliminated（附淘汰者）。
  - 結果：剩 1 個 → decided；> 1 且彼此 quality 都是 tie → tie；> 1 且有 unmeasured → incomparable。剩下的全部列出，不做 tie-break。
- **R12 推薦狀態**：
  - pending：沒有有效 Core A session 或缺 analysis.json。
  - not decidable：quality 關係不可用（沒有 quality session、缺 analysis.json，或 h_a5 為 not testable，附 missing 清單）。
    此時不計算前緣與推薦，也不退回只看效能；證據表、fit boundary 與 H-A5(a) 並列照常輸出。
  - no candidate：附每個配置的狀態。
  - provisional 旗標：同一 (平台, L) 有任何 not measured、error 或 incomplete 的配置時為 True，並列出名字。oom 與 spill 不觸發。
- **R13 替代方案與代價**：每個推薦逐軸列出 better / worse / tie / unmeasured，以及 decode、TTFT、記憶體的比值 ± SE（`core_a_analysis._ratio`，只顯示）。
  前緣中沒被選中的成員記錄它在哪一步、被誰淘汰。同 size 時附 downstream 的對應條目，標「描述性，未做多重比較校正（005）」；跨 size 標 not compared。
- **R14 H-A5(a) 並列**（§2 的報告要求）：每個平台、每個 L 並列 9B-Q4_K_M 與 4B-Q8_0 的狀態、記憶體、decode、TTFT、各軸關係，以及配對 test 的 mean_delta、ci95（標 descriptive only）、p_adj、reject、direction_ok。不另下結論。
- **R15 fit boundary**：每個 (平台, L) 列出非 candidate 的配置與 H-A1 的 predicted / observed。fit map 列每個配置在各 L 的狀態；
  「已量測的最大 fit L」只取已量測的 L，不用 004 模型外推。

### 判定、註記與隔離

- **R16 判定照抄**：H-A1..H-A4b、H-A5、H-B1..H-B4 的 verdict 與 units 原文照抄，附各 status 的計數。
  可檢定的 unit 數為 0 時加註 NO_TESTABLE_UNITS（例如沒有 mixed 時的 H-B4，凍結程式會回傳 not refuted）。
  not refuted 永遠不改寫成 confirmed。推薦與前緣的函式不接受 verdict 作為輸入。
- **R17 註記碼**：只用固定模板，依條件機械附加，例如：
  QUALITY_PLATFORM（Mac 段：quality 在 wsl2-cuda 量，作為 model × quant 屬性套用，Metal 上沒有量）、QUALITY_CTX_512（長 context 的 quality 沒有量）、
  CROSS_SIZE_UNMEASURED、TIE_NOT_EQUIVALENCE、CONTRARY_TO_PREDICTION、CONTAMINATION、NO_IMATRIX_Q4（004）、MAC_POWER_CONDITION（powermode、70 W、charge_pauses）、
  SPILL_EXCLUDED、TTFT_ENGINE_ONLY（§3）、N3_NOISE_RULE（n = 3 時 2·SE 規則偏寬鬆，照凍結規則套用）、PROVISIONAL、NOISE_CYCLE、
  KV_USAGE_INCLUDES_GDN、KNEE_POINT_ESTIMATE、KNEE_AT_GRID_TOP、H_B1_UNREFUTABLE、CORE_B_FIXED_DEPLOYMENT、MIXED_AT_KNEE_ONLY、
  NO_TESTABLE_UNITS、DOWNSTREAM_CI_NOT_REVERIFIED。
- **R18 平台隔離**（SPEC：不做 Mac / PC 排名）：每個平台段由同一個函式產生，只接收該平台的 Core A 輸入與共用的 quality 關係。
  沒有任何列、欄或句子同時含有兩個平台的 Core A 數值；段落順序固定為 macos-metal、wsl2-cuda（字母序）。
- **R20 Core A 與 Core B 分開**：兩者的數字永不放在同一列、同一張表或同一句話中（不同 engine、precision、format；不做 runtime 比較）。
  Scope 對照為固定文字：一次一個 request → 看使用者所在平台的 Core A 段；RTX 上並發 > 1 → 看 Core B 段；Mac 上並發 > 1 → not measured（SPEC：Core B 只在 NVIDIA）。

### Core B（只有 wsl2-cuda；每個 L）

- **R19**：
  - 曲線：每個 grid C 的 n_valid（ok 且 restart 不重複）；3 次都有效才算 complete，輸出 throughput、P50 / P95 TTFT 與 TPOT 的 m 與 SE（stdev / √3），
    以及 KV cache usage 峰值、preemption 次數；incomplete 的 C 只列 n_valid，不顯示部分平均。
  - H-B1 unit 為 not testable → no operating point，列出 missing_c，不推薦任何 C，不以其他 C 代替（013）。
  - **B-T 吞吐量導向**：推薦 C = 凍結的 C_knee（核對 H-B1、H-B2 unit 與 `core_b_analysis.knee()` 三者一致）。
    附 C_knee 的 throughput ± SE、grid 最大 throughput 及其 C、觀測增益，以及由凍結定義推得的上界 (1 ÷ 0.95 − 1) × 100 ≈ 5.26%。
    C_knee = C_max 時標 KNEE_AT_GRID_TOP：C_max 等於 GRID_CAP（128，019）時註明 grid 之外 throughput 可能仍會上升；
    否則註明 grid 上限來自 007 的規則（例如 L = 32768 為 32）。C_max < 4 × C_knee 時標 H_B1_UNREFUTABLE（由規則推導）。
  - **B-L 延遲敏感**：**不選 C**（依延遲目標選 C 是 Advanced 的 fixed-SLO 項目）。只列：5.26% 上界；C = 1、C_knee、C_max 三點的 P50 / P95 TTFT 與 TPOT；
    H-B1 的 status 與 first_c_at_3x；H-B3 的 g_tpot、g_ttft；固定句 `choosing C for a latency target is Advanced (fixed-SLO max concurrency); not done in Core`。
    （綜合方案原有的描述性 C 前緣 frontier_C 刪除：它不在 PREREGISTRATION 中，會擴大分析範圍；完整曲線表已回答 SPEC 的 latency–throughput 問題。）
  - **B-X 長短混合流量**（L ∈ 2048、8192）：照抄 H-B4 unit 與 mixed cell，附固定句「只在 C = uniform C_knee 量測；機制是 conjecture（007）；沒有評估任何緩解（Advanced）」。
    沒有 mixed → pending；L = 32768 → not defined（007）。
  - quality 欄固定為 `not measured (W4A16; 003/008) — Core A Q4_K_M quality does not apply`。
  - scope 列：repo@revision、precision、vLLM 版本、U、K、num_gpu_blocks、block_size、max_num_seqs、max_num_batched_tokens、uniform (L − 128, 128)、temperature 0、ignore-eos。

### 輸出

- **R21 not_supported 清單**（自動收集）：NO_TESTABLE_UNITS 的 hypothesis；not testable、inconclusive、descriptive 的 unit；unmeasured 的 quality 配對；
  tie_dependent 的支配；provisional、incomparable、not decidable、pending；NOISE_CYCLE；KNEE_AT_GRID_TOP；H_B1_UNREFUTABLE；未重新驗證的 downstream ci95；
  以及固定項：Metal 上沒有量 quality、長 context 沒有量 quality、Core B 沒有量 quality、Mac 沒有多使用者測量。
- **R22 確定性**：stdlib only（Python ≥ 3.11，讀 pins 需要 tomllib）。rows 讀入後排序，不依賴檔案順序；集合輸出為排序後的 list。
  JSON 以 `sort_keys=True, indent=1, ensure_ascii=False` 輸出；markdown 只由 JSON 產生，顯示精度寫死，所有判斷都以全精度值計算。
  輸出不含時鐘時間、hostname、絕對路徑或 git 狀態；provenance 只記 repo 相對路徑與 sha256（pins、generator 與所 import 的凍結模組、每個輸入檔）。
- **R23 檔案**：`results/decision_table/decision_table.json` 與 `decision_table.md`。不產生圖。
  exit 0 為成功（可以含 pending 段）；exit 2 為 R1 的 ≥ 2 或 R3 的不一致，此時不寫入、不覆寫任何檔案。
  `schema_version = 1`，改格式時要升版並寫 decision log。

## 為什麼

- 每個 workload 只有一個主軸、沒有權重，所以不需要任何新的數值門檻；取捨由「替代方案與代價」攤開給讀者。
- 「超出 noise」用 §2 已凍結的規則，quality 只用 H-A5 的 Holm 結果，與確認性分析一致，不在旁邊另開未校正的檢定。
- unmeasured 與 tie 分開：前者是「沒有證據」，後者是「有檢定但沒分出來」。把前者當平手會讓速度替使用者決定一個品質未知的選擇。
- session 選擇沒有任何參數，重算比對 fail closed，避免事後挑 session 或使用過期的分析檔。

## 放棄的替代方案

- 用配對 ΔNLL 的 bootstrap CI 決定 H-A5 配對的勝負：違反「CI 只作描述」。
- 跨 size 用未配對的 PPL ± SE 建立 quality 關係：違反 005「跨 size 只能配對比較」。
- 同 size 的 KLD 用未校正的 2·SE：等於在 Holm 檢定旁另跑未校正檢定。
- 加權綜合分數、合成 E2E 延遲：需要新的權重或 output 長度常數。
- 依延遲 SLO 選 C：Advanced。
- CLI 指定 session、「最早有效」規則：都留下事後挑選的空間。
- 前緣為空時退回「全部可行」：source SCC 保證非空，不需要。
- 記憶體用「三個 process 的區間是否重疊」判斷：§2 以外的第二套 noise 準則。
- RTX 記憶體軸用 device + host 合計：RTX 上受限的是 VRAM，合計會把系統記憶體混進 VRAM 預算。合計照樣顯示。
- Core B 描述性 C 前緣（frontier_C）：見 R19。

## 對實驗的影響

- 新增 `benchmarks/decision_table.py` 與 `tests/test_decision_table.py`，只 import 凍結模組，不修改它們。
- PREREGISTRATION 開頭加註本文；hypothesis、門檻、metric、protocol 都不變。
- 決策表是描述性的整理，不是新的檢定；它的任何內容都不回頭影響 verdict。

## 審查（2026-10-10，第一次真實執行之前）

review workflow：三個面向（規則一致性、真實輸入的穩健性、方法與誠實），每個面向的發現另由對抗式驗證者複查；
確認的問題約 13 項（去重後），駁回 4 項。審查者沒有讀 `results/`，也沒有對真實數據執行 generator。
以下修正都發生在 Mac a2、正式 quality、正式 Core B 的分析出現之前，也在 generator 第一次真實執行之前（R0）。

### 規則修訂：R11 的淘汰語意

- 原本：第 k 步一次淘汰「被同一集合內某成員勝過」的所有配置，即使勝過它的成員自己也在同一步被淘汰。
- 問題：這會形成隱含的跨 metric 遞移。例如 4B-Q8_0 被 9B-Q4_K_M 淘汰（配對 ΔNLL），9B-Q4_K_M 又被 9B-Q8_0 淘汰（KLD），
  結果 9B-Q8_0 與 4B-Q8_0 在沒有任何直接 quality 證據的情況下分出勝負。
  KLD（對同 size 的 BF16）與 NLL 是不同的量，「Q6_K 的 KLD 較小」不保證「NLL 較小」，所以這條鏈不成立。
  依同樣的機制，後續軸也可能經由已被淘汰的成員，把 quality 為 unmeasured 的配置淘汰掉，違反「unmeasured 永遠不當平手」。
  驗證者認為原寫法符合 R11 的字面，但我判斷它違反 R8 與 R11 的意圖，所以修訂。
- **修訂後**：每一步重複執行「只淘汰被**目前集合中未被任何成員勝過**的成員勝過的配置」，直到集合不再變動。
  結果在該步的規則下是 antichain，必定非空，與迭代順序無關。`eliminated[].by` 只列出淘汰當下未被勝過的成員。
- 影響：合成資料的預設情境中，A-Q 從「decided 9B-Q8_0」變為「incomparable 4B-Q8_0 ‖ 9B-Q8_0」，
  這才符合「跨 size 除 H-A5(a) 配對以外都是 unmeasured」。

### 其他修正（落實既有規則，不改規則）

- R1：判定 mixed session 是否採用時，不再讀它的 `rows.jsonl`（凍結 report 的拒絕條件只用 session 與 plan）。
  零個 run 的 mixed plan（兩個 L 都沒有 C_knee 時凍結程式會產生）視為 mixed，不再被當成無法辨識。
- R22：凍結 report 的拒絕訊息可能含有絕對路徑（server argv、env），寫入前遮蔽為 `<path>`。
  provenance 改記 `configs/pins.toml` 檔案本身的 sha256（可與 session 記錄的 `pins_sha256` 對照），並列出每個輸入 session 的 `pins_sha256`。
- R16：H-B1 的 inconclusive 依 §2「只報告、不計入」，不算可檢定的 unit；H-A1 只有 predicted 為 fit / oom、observed 為 fit / oom / spill 的 unit 才算可檢定。
- R8 / R14 / R17：H-A5(a) 配對的關係與每個 level 的並列一律附 CONTAMINATION（005），不論檢定結果。
  與預測相反的 Holm 關係只要出現在 candidate 之間，就在該 level、支配的邊與 not_supported 標 CONTRARY_TO_PREDICTION。
- R9：tie_dependent 改為逐邊記錄；只有所有支配邊都靠 quality 的 tie 時，整個配置才標 tie_dependent。
- R13：每個替代方案對推薦配置附上 downstream（同 size：兩者的配對與各自對 BF16；跨 size：not compared）；
  tie 或 incomparable 的推薦另列推薦配置彼此的逐軸關係。
- R19：B-X 依機械規則判定：該 L 不在 mixed plan 的 C_knee 中 → not measured；H-B4 unit 不是 tested → not testable；只有 tested 才是 ok。非 ok 都列入 not_supported。
- R21：只要輸出了 downstream，就列入「ci95 未重新驗證」，不論 quality 段是否可用。
- markdown：推薦表原本的「A-F not deployable」列把 not measured / error / incomplete 也稱為不可部署，與 R5 矛盾；
  改為兩列：「not feasible（oom / spill）」與「unknown †（not measured / error / incomplete）」。
  incomparable 的說明改為「quality 關係未量測（跨 size，或同 size 的鏈不嚴格）」。
- 測試由 40 個增為 52 個（每項修正先寫測試，確認失敗後才修改程式）。
