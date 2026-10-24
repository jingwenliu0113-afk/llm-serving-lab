# NEXT

## 目前位置

**Phase 3 進行中（使用者 2026-10-22 決定，decision 037）**：執行 036 列出的後續，SPEC 的 Optional 除外。
- **使用者的子決定（2026-10-23）**：
  - 服務常駐：先做功能，研究期間不常駐；研究的 GPU 工作結束後，等使用者同意才常駐。
  - thinking：gateway 預設關閉，client 可以開啟。
  - Mac 重負載：可以跑，取得 Mac lock，接 Apple 140W adapter（019 的 70W 規定只適用於 Phase 1 的 session）。
- **範圍**：
  - Research：Stage D（容量的延伸：D1 O-32K、D2 O-T、D3 輸出長度、D4 開放迴路、D5 開場湧入、D6 class 內 prompt 長度不一、D7 經過 gateway）；
    Stage E（E1 8K 滑移觸發點與雙模式、E2 2K 的 H-B4 成因、E3 H-A2）；Stage Q（Q1 Metal、Q2 長 context、Q3 W4A16）。
  - Engineering（capserve v1）：thinking、內建 tokenizer、adaptive Retry-After、quota、TLS、常駐功能、IT-3 的決定性證據；
    多 class 准入等 Stage D 的結果。
- **順序**：產品 v1 中不依賴研究的項目（Mac，TDD）與 Stage D 的事先登記並行 → Stage D 正式 session → Stage Q / E → 結案。
- **不變**：Phase 1、Phase 2 的結果與凍結；正式設定仍釘住 `verified-formal-20261017-c1b.json`，改變釘選需要使用者決定。
- **使用者端**：正式 session 之前，建議使用者停用 BingWallpaper 的自動啟動（Windows 系統設定，我不修改）。

### 下一步

- 產品 v1（decision 038）：
  - [x] thinking 預設關閉、client 可開啟（§1）；
  - [x] IT-3 改用 token 數判定，IT-4 測 thinking 開與關（§2）。RTX release `c9202c5`：verified 的 IT-3 決定性通過（8 / 128、49 / 128）。
  - [x] adaptive Retry-After（§3）。
  - [x] 內建 tokenizer（§4）。RTX 的 IT-4 中 23 項計數全部相等，兩份設定都已改為 `local`；local 模式的實際啟動等下一次 RTX 測試確認。
  - [x] quota（§5）。
  - [x] TLS（§6；Mac 上以真正的 uvicorn 與自簽測試憑證驗證；正式設定仍只綁 loopback）。
  - [x] 常駐功能（§7；研究期間不常駐）。
  - RTX 產品測試都已完成（release `c9202c5`、`f670c22`、`4711390`）：
    - verified 與 provisional 的 IT-0 至 IT-6 通過；
    - local tokenizer 下的正式設定通過；
    - 常駐模式的自動重啟與停止正常。
  - 多 class 准入與開場湧入的處理：等 Stage D 的結果。
- **Stage D**（decision 039，DRAFT）：
  - 範圍：D-A 封閉迴路延伸（U32K、T、輸出 512）、D-B 開放迴路、D-C 固定點驗證（prompt 長度不一、經過 gateway）、D5 開場湧入的描述。
  - [x] harness（`91e6e73`）：兩輪審查（13 + 8 項）全部處理；λ_ref pin 在設定檔、由 Mac 上的測試比對；凍結證據綁定 `harness_sha256`。
  - [x] 三個 mini smoke 全部通過（`smoke-d-{closed,open,verify}-20261023b`）。
  - [x] **凍結**（`fd95164`）：`freeze.stage_d = true`、`tests/test_stage_d_frozen.py`、039 與 §9 為 FROZEN。
  - [x] 正式 session：`formal-20261023-d-a1` 因 interop 失效在第一次 boot 前中止（039 執行紀錄一）；
        `-d-a1b`（10-22 20:11Z – 10-23 08:11Z）、`-d-a2`（– 14:18Z）、`-d-b`（– 18:15Z）、`-d-c`（– 19:48Z）全部 exit 0，共約 23.6 h。
  - [x] 取回（不含 raw/）、四份 report、D5（`62869cf`）。report 的判定：H-D1、H-D2a、H-D2b supported；
        H-D3 U2K-O512 not supported、U8K-O512 supported；H-D4 U2K-P NT、U8K-P supported；H-D6 U2K-var not supported、U8K-var supported；H-D7 兩點 supported。
  - [x] 判讀與結果文件 **042**：每個 session 與 D5 各由一個 agent 抽取、第二個 agent 反駁，判定全部和 report 一致；039 執行紀錄二與文字更正、PREREGISTRATION §9 註記。
  - [ ] D-B 無效 run（C-U2K-P-i3-r1）的原因（**等 GPU 空閒**）：`tools/rtx/fetch_raw.sh` 唯讀取回 `raw/C-U2K-P-i3-r1.bench.log`（sha256 `96a2d194…`），
        並查 WSL venv 的 aiohttp 版本，確認 042 推論的 keep-alive 競態；只更新 042 的說明，判定不變（U2K-P 仍為 NT）。
  - [x] 產品設計 **043**（DRAFT，`76115eb`）：三份設計經三位評審比較。待決事項 1–7 依建議定案：3 份不釘選的 verified profile、
        輸出上限 > 128 的 class 不能作為 active class、T 只服務 mixed-8k-t 而且現在不部署、正式釘選與 active class 不變、開場湧入維持現況。
  - [x] 043 實作（0 GPU，TDD）：loader 與 §9、產生器接受 D-A session、三份不釘選的 verified profile（`-d-a1b-b0`、`-d-a1b-t`、`-d-a2-b0`）、
        輸出 > 128 與 T 的拒絕、B0 拒絕 T workload、`/readyz` 揭露、gateway keep-alive 75 s、README；獨立審查後合併（`abca173`）。
  - [ ] RTX 產品測試（**等 GPU 空閒**，GPU lock 下）：`rtx-verified.toml` 的 IT-0 至 IT-6（keep-alive 改變了 `capserve_sha256`；輸出只進 `runs/`）。
  - [x] **使用者決定**（2026-10-24，043 待決事項 8、9）：V4、V-GW、V1、V2 都做，V3 延後；排在 E1-P、E1-D 取回與 Stage Q 的 RTX pilot 之後，session 之間留空檔。合計約 10–11 GPU h。
  - [x] **044**（DRAFT）：Stage V 的事先登記設計（三份設計經三位評審比較），PREREGISTRATION §12（DRAFT）。待決事項 1–8 依建議定案。
        共用一套 harness、一次 mini、兩個正式 session（V-ADM 約 2.75 h：V2 + V4 + F-U2K 對照；V-CLS 約 5.25 h：V-GW + V1），之間 ≥ 2 h；全期上限 12 h。
  - [x] 044 的實作（0 GPU，TDD；合併 `f12255f`）：產品程式（access log 欄位、weighted-v1、pacing、§12 loader、`provisional-stage-v.json`、AST 閉包雜湊）、
        harness 五個模組與 `configs/stage_v.toml`、U_W 的桌面模擬（W-90 31.55、W-50 30.80，門檻 28.8 可達）。正式路徑的渲染與 `/readyz` 舊欄位不變。
  - [x] 044 凍結前審查兩輪與修正後的獨立檢查（實作補記六、七、八）。第二輪的 must-fix：vLLM 0.30 bench 預設不送 ready check（`--ready-check-timeout-sec` 0），
        沒有 id 的 completion 紀錄改為剛好 C 筆（原本要求 ≥ C + 1，會讓每個 V run 無效）；另有：bench 之前 gateway 結束記為 fault、report 只看在本 session 之前開始的目錄、
        session 時間必須帶 offset 且不倒退。由我決定、凍結前使用者仍可改：跳過規則改為逐點（實作補記六第 6 項）。
  - [x] 044 產生器的 `--validation-session` 路徑（Mac、0 GPU，TDD，合成 session；`feat/044-generator`，實作補記九）：只由 analysis.json 的 units 寫出至多一份 §12 profile，
        沒有會寫檔的單位成立時 exit 0、不寫檔；凍結、DEPLOYED_COMMIT 的請求路徑、base profile、工作樹、mini 一律拒絕；`gateway_path_sha256` 不變；40 個突變全部被抓到。
  - [ ] 044 之後（**等 GPU 空閒**）：WSL 唯讀複核（CustomDataset output_tokens、/tokenize、x-request-id、aiohttp、ready check 預設略過）→
        WSL 上產生 dataset 並 pin → GPU 視窗 0（產品 IT-0–6、mini）→ 凍結 → V-ADM → ≥ 2 h → V-CLS。順序排在 E1-P、E1-D 取回與 Stage Q 的 RTX pilot 之後。
        凍結前（pin `configs/stage_v.toml` 之前）另改 `[stage_v.units]` 上方仍寫舊單位跳過規則的註解（Mac，實作補記九第 13 項）。
- **Stage Q**（decision 040，DRAFT；PREREGISTRATION §10）：
  - [x] Q1 / Q2 harness 與凍結檢查（`f65c081`、`49b88af`）：`configs/stage_q.toml`，三個 pilot 是凍結的證據，分析只接受凍結後的正式 session。
  - [x] Q3 harness（`quality/stage_q3.py`、`configs/stage_q3.toml`，獨立的凍結）。
  - [x] 審查（5 個審查者、每項由兩個 agent 嘗試反駁；33 項確認）全部處理（`0ec7b92`、`fd832e4`），見 040 的審查表。
        包括：q1-metal 原本因 70W／140W 兩條規則同時套用而永遠無法執行；Phase 1 的 20-chunk pilot 已揭露（界線與預測不改）。
  - [x] 修正後的覆核：Q1 / Q2 確認 5 項、Q3 確認 3 項（共 4 項 must-fix），全部處理（040「修正後的覆核」）：stub 與嘗試次數、分析自己找 session、`--superseded` 機制、精確的端點、pilot 記錄 base 大小、Q3 去掉結尾換行與 V1 的平手規則。
  - [x] q1-metal pilot（Mac lock 下，2026-10-23 06:16Z）：`results/stage_q/pilot-q1-metal-20261023`，20 個 job 全部 ok，`pilot_problems` 通過，harness `34256f12…` 等於目前的版本。
  - [ ] pilot：q1-cuda、q2（RTX，**等 GPU 空閒**）→ 三個 pilot 齊了才凍結（`freeze.stage_q` 與三個 pilot id）；凍結後 Stage Q 的 harness 不能再改。
  - [ ] Q3：下載 `Qwen/Qwen3.5-9B@c2022362` 的 BF16 safetensors（19.33 GB）到 WSL，**使用者 2026-10-23 同意**；下載會讀寫 WSL 的網路與磁碟，**等 Stage D 正式鏈結束、GPU 空閒**；
        之後在 WSL 產生 chunk 檔並 pin sha256 → pilot（含 `cpu_offload_gb` 與 WSL pinned memory 的可行性）→ 凍結。
  - 觀察：Mac 的 140W adapter 目前只協商到 94 W（20 V × 4.69 A；名稱檢查會通過）。可能是線材或連接埠的限制，重負載時電池可能下降（016 的充電閘門會處理）。
- **Stage E**（decision 041，DRAFT 第 3 版；PREREGISTRATION §11）：
  - [x] 第 2 版經審查（44 項確認）與設計 workflow（3 位設計者、3 位評審）後改寫為第 3 版（`6db1512`），設計前看過的重算在 `results/stage_e/prereg-tally/`。
  - [x] 041 §6 的 A1–A4：使用者 2026-10-23 決定都採建議的選項（A1 做換 seed 探針、A2 allocator 不操弄、A3 PC 為能力檢查、A4 接受條件式結論）。
  - [x] harness：E3（`320343a`）、E1-D（`1450a3c`）、E2（`a7a5e44`）；每個部分有自己的設定檔與凍結。
  - [x] E1-D 與 E3 的 harness 審查：確認 34 項，全部修正（E1-D `4f83e06`、E3 `c41c073`）。
  - [x] 修正後的覆核：確認 9 項（E3 的 plan 雜湊含 `[freeze]` 為 must-fix），全部修正（`13d7a74`）。
  - [x] E1-D 凍結前覆核兩輪（041 §1 逐條、在已看過的數據上重現 §1.2、取回檔案的格式、端到端）：F 尖峰的錨點改為 s + 5 ms（must-fix，§5 第 13 項揭露）等，全部處理（`937ed5d`）。
  - [x] **E1-D 凍結**（2026-10-23，`freeze.e1 = true`，`e1_harness_sha256 = 87b937d1…`）。
  - [ ] E1-D 取回（**等 GPU 空閒**：Stage D 的鏈結束、`tools/rtx/gpu_status.sh` 為 FREE）：
        `python benchmarks/stage_e1.py fetch-list | while read -r args; do tools/rtx/fetch_raw.sh $args; done`
        → `python benchmarks/stage_e1.py analyze --out results/stage_e/e1-<date>/e1.json` → 結果文件（描述，不是檢定）。
  - [x] E3 的兩個 Q4_0 已產生並 pin（2026-10-23，Mac lock 下；4B `ee05b023…` 2,543,899,584 B，9B `2dfcbfd3…` 5,313,356,768 B）。
  - [x] E3 Mac pilot（Mac lock 下，2026-10-23 06:2xZ）：`results/stage_e/pilot-e3-metal-20261023`，5 個 job ok、sentinel 正常、env 有效（140W adapter，協商 94 W），`pilot_problems` 通過，harness `85b9842c…` 等於目前的版本。結果已遮蔽。
  - [ ] E3：`produce_gguf_e3.py sync` 傳到 WSL（約 7.9 GB，**等 GPU 空閒**，不在正式 session 期間讀寫 WSL）→ RTX pilot → 兩個旗標一起凍結（Mac 先不凍：任一平台的正式 plan 都要兩個旗標與兩個 pilot，先凍沒有好處）→ `tools/rtx/push_session.sh stage_e pilot-e3-metal-20261023` → 正式（每個平台最多兩次嘗試）。
  - 注意：Stage Q 與 E3 的 Mac pilot 已綁定目前的 `stage_q.py` / `stage_e3.py`；若之後必須修改，pilot 要重跑。
  - [x] E2 harness 審查：確認 18 項，全部修正（smoke 綁定設計並改為 12 次 boot、通過條件補齊、停止規則、重跑上限、R / Λ / L̄q、incomplete 的代換規則等；041 修訂紀錄「E2 harness 審查」）。
  - [ ] 分析 WSL 上的 session 之前一律先 `tools/rtx/fetch_all_sessions.sh stage_e`（或 `stage_q`），commit `wsl-inventory.txt`。
  - [ ] E2（**等 GPU 空閒**，WSL）：原始碼核對（prefix caching 路徑、warmup 來源、`vllm:prefix_cache_hits_total`、`cache_config_info` 的 `mamba_cache_mode` 標籤、span 的 cached-token 欄位、`num_requests_waiting`）→ `stage_e2_datasets.py` 產生共享前綴 dataset 並 pin sha256 → smoke（12 次 boot）→ 凍結 → 正式（約 4.2 h）。
  - [x] E1-P harness 寫成並審查（2026-10-23，DRAFT）：`configs/stage_e1p.toml`、`benchmarks/stage_e1p_plan.py`、`run_stage_e1p.py`、`stage_e1p_analysis.py`、`stage_e1p_seed_check.py`、`tests/test_stage_e1p.py`；三輪審查（都沒有 must-fix）全部處理（041 修訂紀錄「E1-P harness 審查」「E1-P 修正後的覆核」「第三次覆核」；E2、E3、Stage Q、Q3 的中斷、SIGTERM 與 refused 範圍一併修正，040 #39–#49）。
  - GPU 佇列（2026-10-24）：Stage D 結束後，另一個專案的 `agent_gym_serve` 自 01:09Z 起持有 GPU lock；不搶占。
        lock 空出後依序：部署 HEAD（`deploy_bundle.py`）→ E1-P seed 檢查與 smoke → D-B 的 raw 取證 → E1-D 取回 → Q3 下載 → E3 / Stage Q 的 RTX pilot → E2。
  - [ ] E1-P（**等 GPU 空閒**，WSL）：
        1. seed 檢查（只用 CPU，`flock -n` 下）：`flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/stage_e1p_seed_check.py` → Mac 上 `tools/rtx/fetch_seed_check.sh`（核對 sha256）→ commit。不通過時（seed B 有 prompt 不是 8,064，或 A、B 相同）要另寫 decision log 換 seed，不能自行改。
        2. smoke：`python benchmarks/stage_e1p_plan.py --mode smoke --out ~/e1p-smoke-plan.json`，`flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/run_stage_e1p.py --plan ~/e1p-smoke-plan.json --out results/stage_e/smoke-e1p-<date> --gpu-lock-held --smoke`
        3. 取回、凍結（`freeze.e1p = true`、`smoke_e1p`）→ 正式（約 2.5 h，上限 3.0 h 減 smoke 的時間）→ `stage_e1p_analysis.py --out results/stage_e/e1p-analysis-<date>/e1p.json`。
## Phase 2 結案時的狀態（2026-10-19，decision 036）

- **正式成果**：`service/profiles/verified-formal-20261017-c1b.json`（decision 035）。
  - C_PROD：uniform 2K 32、uniform 8K 8、mixed 2K / 8K 2。
  - 適用條件、IT-3 的限制與正式 profile 的保護見 036。
- **結案檢查**（2026-10-19，036 的結案紀錄）：
  - 最終 release `5201760` 在 RTX 上通過正式設定的拒絕檢查；verified IT 的 IT-0、1、2、4、5、6 通過，IT-3 為無法判定；provisional-rtx 的 IT-3 通過。
  - 完整測試 2,310 個通過；工作區乾淨，已 push；RTX 上沒有殘留的程序或 lock。

（以下為 Phase 2 期間的紀錄）Research 與 Engineering 兩條線受控並行（使用者 2026-10-15 決定，decision 027）。
- Phase 1（Core + Advanced，026 的 milestone）結果保留，不重新分析；凍結程式由 `test_core_untouched`、`test_frozen_untouched` 保護。
- **Research Stage C**：
  - C0：8K 延遲雙模式的有限度排查（探索性）。
  - C1：Fixed-SLO 最大並發（正式）。PREREGISTRATION §8 已凍結（2026-10-17）；SLO 由使用者選定（S2）。
- **Engineering**：Capacity-Aware Serving Service MVP（`service/`），建在 vLLM 之上。只有 Stage C 驗證過的容量能成為正式限制。
- 任何在 RTX 上啟動 vLLM 的工作都要取得 GPU lock；產品測試的輸出不進 `results/`；正式研究 session 進行中不部署任何東西。

**Core 全部完成（2026-10-10）**：Mac / RTX Core A、Core B（uniform + mixed）、quality、deployment decision table。
**Gate 通過（使用者 2026-10-10 決定，decision 023）**：進入 Advanced，範圍只限隔離 H-B4 的成因。Optional 未觸碰。
**Advanced 完成（2026-10-11，decision 025）**：正式 session `formal-20261011-adv` 60 / 60 有效；
2K 落在 R6（KV 可能有關，但未確立），8K 落在 R1（主因是 chunked prefill 壟斷 step budget，H-C1 成立）；跨 L「成因依 L 而異」。

- Stage A：harness commit `eb6eef1`，`freeze.mac = true`。
- Stage B：harness commit `19446d7`，凍結 commit `b42614a`，`freeze.nvidia = true`。
- 模型（008）：
  - Core A：Qwen3.5 4B + 9B GGUF，BF16 / Q8_0 / Q6_K / Q4_K_M。
  - Core B：RedHatAI/Qwen3.5-9B-quantized.w4a16。
  - context：2K / 8K / 32K。
- Decision log 001–041（030 於 2026-10-17 凍結；033 是第一個 C1 session，已被取代；034 是重跑的取捨規則；035 是最終結果；036 是 Phase 2 結案；037 是 Phase 3 的範圍；038 是 capserve v1 的設計；039 / 040 / 041 是 Stage D / Q / E 的事先登記草稿）。細節以 decision log 為準，這裡只記現況與下一步。
- GitHub：`origin` 為 private repo。每個測試通過的 commit 都會自動 push；不 force push，不改寫 history。
  權重、GGUF、KLD、`results/**/raw/`、venv、credential 不進 git。

## 歷史紀錄：Core 執行期間的現況（2026-10-10 20:30 寫下，之後未更新；目前狀態以上方「目前位置」為準）

### 使用者決定（2026-10-09 上午）

- **019**：
  - Mac 正式 run 用**自動電力模式**加上**原廠 70W adapter**，並記錄 power state。
  - 正式 Mac Core A 的電池下降停止規則：13:xx 依使用者決定刪除。
  - 高效能模式（powermode 2）也可以，但同一 session 內要一致。
  - Core B 的 `--max-num-seqs` 改為 **128**，grid 上限改為 **128**。
  - Stage B 不拆分。
- **020**：正式 quality 改在 **RTX** 上執行，flag set 與 hypothesis 不變。
  - 4B 一起搬：使用者 2026-10-09 確認。
  9B BF16 會少量 spill 到系統記憶體，運算仍全部在 CUDA 上完成。

### RTX（gpu-server，GPU lock 下依序執行）

- **正式 RTX Core A 完成**（`formal-20261009-rtx-a1`，10:42–13:11）：
  - 164 / 164 ok，sentinel 漂移 0.06%，env 有效，session 開頭重量的 B = 462 MiB（沒有 drift）。
  - 判定（`analysis.json`）：H-A1、H-A3、H-A4a、H-A4b 為 not refuted；**H-A2 為 refuted**。
    - 推翻 H-A2 的是 4B-Q4_K_M 對 9B-Q8_0 這一組：觀測比 2.24，預測比 2.99，偏差 −25%，門檻是 ±20%。
    - 15 組配對的偏差全部是負的：bytes 少的配置都比純頻寬比例預測的慢。原因屬於推測，Core 不驗證。
  - 圖：`results/core_a/figures/{decode,prefill,memory}-rtx-formal.svg`。
- **正式 Core B uniform 完成**（`formal-20261009-coreb-uniform`，2026-10-09 13:11 至 2026-10-10 09:11，約 20 小時）：
  - 102 / 102 ok，formal，env 有效，沒有 teardown 問題。
  - uniform-only 分析：H-B1、H-B2、H-B3 not refuted。
    - C_knee：2K 為 25（C_KV 35）、8K 為 8（C_KV 13）、32K 為 2（C_KV 3），都落在 H-B2 的 [C_KV ÷ 2, 2 × C_KV] 內。
    - H-B1 三個 L 都是 supported（首次達到 3 倍 P95 TTFT 的 C：70 / 16 / 8）。
  - H-B4 要等 mixed，目前 0 個可檢定的 unit。
  - mixed 跑完後，`analysis.json` 要以 `--mixed` 重產（021 R3）。
  - 圖：`results/core_b/figures/{p95_ttft_ms,p95_tpot_ms,output_throughput}-uniform-formal.svg`。
- **uniform 之後的自動接續**（背景 `chain_rtx_after_uniform.sh`，每一步都在 GPU flock 下、ssh 前景執行，遇到問題就停）：
  1. 唯讀檢查 uniform session：formal、env 有效、102 列全部 ok。不符合就停止，不部署、不開跑任何東西。
  2. 確認 GPU 上沒有 compute process 後，部署 `2c2e764`（bundle 事先建好），並核對 `DEPLOYED_COMMIT`。
  3. RTX quality pilot（`pilot-<日期>-rtx-q`，005 pilot protocol，N_thr 6 / P 0），接著產生遮蔽版報告。
  4. 正式 Core B mixed（`formal-<日期>-coreb-mixed`），C = 各 L 的 uniform C_knee，使用 pin 過 sha256 的 mixed dataset。
  - `core_b_plan.py mixed` 現在會拒絕非 formal 或 env 無效的 uniform session（與 `core_b_report` 同一規則，018 補充）。

- （歷史）正式 RTX Core A `formal-20261009-rtx-a1`。
  - 10:36 開始，session 開頭會重量 B；pilot 估計上限約 3.6 小時。
  - 跑完自動接正式 Core B uniform `formal-20261009-coreb-uniform`：3 個 L × 3 次 restart，粗估約 10 小時。
  - 背景串接腳本：`chain_rtx_formal.sh`。
- **Stage B 值**（§6）：

  | 項目 | 值 |
  |---|---|
  | B | 811 MiB |
  | V_total | 16,303 MiB |
  | C_ctx | −89 MiB |
  | U | 0.91 |
  | K | 4.0 GiB |
  | num_gpu_blocks / block_size | 248 / 528 |
  | pin memory | 0 |
  | N_thr / P | 6 / 0 |
  | HAGS | 開 |
  | 超量測試 | spill |
  | C_KV（2K / 8K / 32K） | 35 / 13 / 3 |
  | grid 2K | 1, 2, 4, 8, 16, 18, 25, 32, 35, 49, 64, 70, 128 |
  | grid 8K | 1, 2, 4, 7, 8, 9, 13, 16, 18, 26, 32, 64, 128 |
  | grid 32K | 1, 2, 3, 4, 6, 8, 16, 32 |

- **BF16 參考檔已傳到 WSL**（020）：9B、4B 都在 WSL 上以 sha256 驗證完成（20:09）。quality 資料檔 14 個也已核對。
- **注意**：正式 RTX session 進行中**不得重新部署**到 WSL。
  - 中途部署會悄悄換掉之後 process 使用的程式，而且不會留下標記（020 第 7 點）。
  - 審查後三個 runner 都加了 provenance 檢查，但進行中的 session 跑的是 `b42614a`，沒有這個檢查。
  - 等 GPU 上的正式 session 結束後，才部署 `run_quality` 的 RTX 支援。

### Mac

- 使用者 2026-10-10 中午再次暫停 Mac 的高負載，**當天傍晚解除**（使用者親自通知）。
  - quality 跑完後，直接在 Mac 上跑 `quality_analysis.py`。downstream bootstrap 約 10 億次抽樣，會占一個 CPU 核心數分鐘。
    放在 Mac 上跑，是為了避免 021 R3 的跨平台浮點不一致。

- 使用者 2026-10-10 01:45 暫停 Mac 的新重負載；**03:55 解除**（由使用者的另一個 Claude session 轉達「mac 已經用好了」；使用者 2026-10-10 上午親自確認可以用）。
  解除後 Mac 也沒有待跑的重負載：Core A 已完成，quality 在 RTX，Core B 只在 RTX。
  - a2 已跑完且有效（見下）。RTX 照常進行（使用者只用 Mac）。
  - Mac 目前沒有排任何重負載；只做輕量的分析、圖與 commit。

- **2026-10-09 13:xx：使用者刪除 019 的電池下降停止規則，並要求重跑。**
  - `formal-20261009-a1` 依舊規則作廢，照實保留。
  - 重跑 `formal-20261009-a2` 在 Mac lock 下、固定 commit 的 worktree 中執行。
  - 適用的規則：012、016 充電閘門、70W adapter、同一 session 內 powermode 一致。

- **正式 Mac Core A 完成**（`formal-20261009-a2`，2026-10-09 20:15 至 2026-10-10 02:55，commit `8f8dc9a` 的 worktree，powermode 2，70W adapter）：
  - 164 / 164 ok，sentinel 漂移 1.78%（門檻 3%），env 有效，沒有充電暫停；每筆 env 的 commit 都相同且 dirty = False。
  - 判定（`analysis.json`）：H-A1、H-A3、H-A4a、H-A4b not refuted；**H-A2 refuted**。
    - 推翻 H-A2 的是與 RTX 相同的配對：4B-Q4_K_M 對 9B-Q8_0，觀測比 2.22，預測比 2.99，偏差 −25.8%。
    - 15 組配對中 13 組偏差為負。
  - 圖：`results/core_a/figures/{decode,prefill,memory}-mac-formal.svg`。worktree 已移除（內容與主 repo 的副本逐檔相同）。
- 019 的 harness 已上線，並經過 review workflow：確認 10 項，均已修正。

## Phase 2 的紀錄（已結案，decision 036）

### Research
1. **C0**（decision 028）：
   - harness 已 commit（`e4c23c9`），審查 6 項已處理。
   - **步驟 0 完成**（`ac29c54`）。事先寫定的重現檢查沒有重現；每次滑移都是錯過一個 2048-token prefill step（約 349 ms）；Core r2 的滑移較多，可能有 boot 層級的成分。
   - **閒置判定改為 decision 031 的綜合判定**（使用者 2026-10-16 確認），並依實測校準 spill 偵測（031 修正案，使用者確認）：
     就緒時比 boot 前多出 ≤ 160 MiB（正常 boot 固定 +80 MiB），之後每個 bench 相對就緒時 < 64 MiB。
   - 使用者結束了 ChatGPT、BingWallpaper；依使用者指示我結束了 Radeon Software。閒置 VRAM 降到 616–960 MiB。
   - WSL 的 interop binfmt 會在開機後消失，`read_shared_usage` 改為遇到 ENOEXEC 時經由 `/init` 呼叫（不改系統設定）。
   - **C0 完成**（部署 `f0eee6a`，2026-10-16 23:31 – 10-17 01:59，GPU lock 下）：`explore-c0-20261016` 24 / 24 個 bench 有效。
     滑移段 5 段，D = 3.40 < 11.07 → **Rrun → 分支 A**（decision 032）。起始組集中在 idx 89 等現象只報告，列入 Future Work。
   - 同一次 lock 的 Stage C smoke 都通過：mini `smoke-mini-202610161531`、FS-order `smoke-fs-order-202610161531`。
2. **§8**（decision 030，**FROZEN 2026-10-17**）：
   - 使用者已決定（2026-10-15）：SLO S2（TTFT 2 / 6 / 20 s，TPOT 100 ms）、mixed 長 request 為 budget、穩態時間窗、GPU 空著就跑。O-32K、O-T 不做。
   - 設定檔放在 `configs/capacity.toml`；`pins.toml` 維持 Advanced 凍結時的位元組（030 實作補記）。
3. **Stage C harness 完成**（`eb78a35`，仍為 DRAFT、未凍結）：審查 21 項已處理；030 實作補記二記錄了所有實作時的決定，
   包括 C_PROD 一律把 MARGINAL 當 FAIL，以及凍結必須附上的證據（兩次通過的 smoke 與 C0 的 decision log）。
   smoke 通過、C0 分支 A、凍結證據已填入 `configs/capacity.toml`（decision 032，`f35fe85`）。
   **已凍結**：`freeze.capacity = true` 單獨一個 commit（profile 產生器以它為凍結 commit）；`tests/test_stage_c_frozen.py` 以 sha256 固定五個 Stage C 模組與 capacity.toml。
   凍結 commit `b3c9109`（profile 產生器的 `freeze_commit_problems` 已確認它是合格的凍結 commit）。
   - **正式 session 完成**：`formal-20261017-c1`（部署 `b3c9109`，2026-10-17 02:51–10:18，7.44 h），69 / 74 個 run 有效，replay、sentinel、跨 session 偏移都通過。
     結果（decision 033）：U2K C_PROD 32 LOWER、M2K / M8K 2 EXACT、**U8K NT**。
     5 個無效的 run 都是 spill：BingWallpaper 在 08:29 自行重新啟動，在 NVIDIA 卡上用了約 190 MiB。spill 偵測擋下了它們，依規則不補跑。
   - **重跑（使用者決定，decision 034）完成，成為最終結果（decision 035）**：
     - session：`formal-20261017-c1b`，部署 `b3c9109`，10-17 18:20 – 10-18 02:40。
     - 80 / 80 個 run 有效，沒有 spill；replay、sentinel、跨 session 偏移都通過。
     - **C_PROD**：U2K 32 EXACT（C_SLO 34，STRICT_COARSE）、U8K 8 EXACT、M2K / M8K 2 EXACT。
     - **Hypotheses**：H-S1 與 H-S2 在 2K / 8K 都 supported，和預測相同。
     - 033 改標「已被取代」。兩個 session 能比較的部分都一致，依 034 不合併。

### Engineering
1. 架構：decision 029。
2. **M0 完成**（`3147165`）：
   - 審查 12 項，11 項修正，1 項延到 M2（見 029 實作補記）；service 測試 274 passed。
   - 刪除 uvicorn 的 `limit_concurrency`，因為它數的是連線，到上限時會讓所有 request 都拿到 503。
3. **M1 完成（Mac 端）**（`343155f`）：deploy、procs、runs、loadtest、itest（IT-0 至 IT-6）、`tools/rtx/deploy_service.sh`。
   審查 11 項（2 項重複）全部修正，見 029 的 M1 實作補記；完整測試 2006 passed。
   - **RTX 整合測試 IT-0 至 IT-6 全部通過**（2026-10-16，兩次：`pt-20261016T142720Z-itest`、`pt-20261016T143958Z`，release `635b954` / `f30d17c`），
     依 031 的產品規格（基線判定；> 2,048 MiB 拒絕）。只是產品測試，不是容量。
4. **M2 已實作**（`b91a9bf`）：產生器、載入時的 provenance 驗證、verified 0 與 unavailable class。
   **verified profile**：`service/profiles/verified-formal-20261017-c1b.json`（uniform-2k 32、uniform-8k 8、mixed-2k 2、mixed-8k 2，都是 exact）。
   第一份 `verified-formal-20261017-c1.json` 依 034 移除，git 歷史保留。
   - **以最終 profile 跑 RTX 整合測試**（release `7513d62`，2026-10-18，029 的最新補記）：
     - verified profile 的 IT-0、1、2、4、5、6 全部通過（`pt-20261017T184655Z-itest`）。
     - 同一 release、同一次 lock 中，provisional-rtx 的 IT-0 至 IT-6 全部通過，IT-3 是決定性的（`pt-20261017T184834Z-itest`）。
     - 128 token 的輸出上限讓 IT-3 無法區分 abort 和自然結束：verified profile 下的 IT-3 記為**無法判定，不算通過**（036）。
   - **正式 profile 的保護**（036，2026-10-19）：`rtx-verified.toml` 的 `production_profile` 釘住最終 profile。
     deploy 與 itest 會拒絕其他 profile 與任何允許 provisional 的執行；provisional profile 不得使用 Stage C 的 class 名稱。

### Phase 1 收尾（2026-10-11，已完成）

- **收尾工作（2026-10-11，已完成）**：
  - `test_disk_guard` 的環境相依已修正（026）：`disk_guard.check()` 改為呼叫時才取 `shutil.disk_usage`；先有紅燈測試，修正後完整測試通過。
  - README 與 NEXT 標記目前的 milestone；8K uniform 的雙模式記入 Future Work / limitations，不追查。

（以下為歷史紀錄）

0. **023 的 Advanced 範圍已完成**（decision 025，README「Advanced」一節）。
   - 正式 session `formal-20261011-adv`（部署 `97a948c`，GPU lock 下，2026-10-11 00:27–06:33）：60 / 60 次 boot 有效，0 逾時，env 有效。
   - `adv_report`（Mac，project venv）：2K → R6，8K → R1，跨 L 不合併。事先預期 8K 符合、2K 不符合，照實報告。
   - 以 `rows.jsonl` 獨立重算 U、M、ρ、f、a，與 `analysis.json` 一致；獨立核對者重跑 `adv_report` 逐位元相同，並核對 025 / README 的文字（14 項修正已納入）。
   - Core 判定與 decision table 不變（重跑只差 pins sha256，commit 的仍是第一次執行的輸出）。
   - 之後的任何 Advanced 項目或 Optional 都需要使用者決定；使用者 2026-10-11 決定收尾（026）。

1. （歷史）**Advanced（H-B4 成因隔離，decision 023、024）**：
   1. 事先登記草稿已 commit（`ae368e6`）：decision 024、PREREGISTRATION §7（DRAFT）。
      - 5 個 condition：B0、T（threshold 512）、G（關閉 full-ISL 准入閘門）、TG、K3（KV 3 GiB）。
      - 每個 condition × 2 個 L × 2 種 mix × 3 次 restart = 60 次 boot，預期約 7.3 h GPU。
   2. **harness 完成**：`adv_plan`、`adv_decomp`、`adv_analysis`、`run_adv`、`adv_report`、`adv_smoke`（TDD；Core 模組不改，有 sha256 回歸測試）。
      審查確認 13 項，全部修正並再驗證；完整測試 1006 passed。實作時的細節記在 024「實作補記」。
   3. **smoke 通過**（`results/advanced/smoke/20261011-fs`，FS0–FS5）：K3 實測 186 個 block，符合規則。
   4. **凍結**（2026-10-11）：`freeze.advanced = true`，PREREGISTRATION §7 為 FROZEN（024「凍結」一節）。
   5. 部署凍結 commit（`deploy_bundle.py` 需要 PATH 先找到 CommandLineTools 的 git）→ 正式 run（約 7–8 h，GPU lock 下）→ `adv_report` → 報告。
2. Core 的結果與 decision table 不因 Advanced 而改變。

## Blocked / 需要使用者

目前沒有。研究的閒置判定已滿足（使用者結束了背景程式）。若之後閒置 VRAM 又超過 1,018 MiB，研究會依 031 自動不開始。

## 建議（非 blocker）

- 正式 Mac run 期間盡量少開前景程式。
  - 前景負載不會讓 session 作廢，但會增加 noise。
  - 4B pilot 時前景 loadavg 為 6–9，CV 5.9%；9B pilot 在深夜，CV 1.0%。

## 已解決

- 2026-10-23：`/usr/bin/git`（Xcode shim）從 2026-10-10 起因 `libxcrun` 架構不相容而失敗；2026-10-23 檢查已恢復正常（`git version 2.54.0 (Apple Git-157)`），不再需要改用 CommandLineTools 的 git。

- 2026-10-11：
  - 下一步方向：使用者決定收尾，不新增 Advanced 項目；8K uniform 的雙模式記入 Future Work（026）。
  - `test_disk_guard` 的結果取決於機器的剩餘磁碟空間（只有 60–250 GiB 時通過）：已修正（026）。

- 遠端 SSH（2026-10-08）。
- WSL compiler（使用者已安裝）。
- llama.cpp 的 CUDA 12.8 系統套件：改走 user-space 路線，不再需要（015）。
- 2026-10-09：
  - Mac 正式 run 解除暫停（019）。
  - Core B 選 A（019）。
  - Stage B 不拆分（019）。
  - quality 的 Mac 磁碟不足：quality 改在 RTX 執行，問題消失（020）。清理 scratchpad 釋出約 2.2 GB。
  - 019 中「quality 是否套用電池下降停止規則」：quality 改在 RTX，不再適用（020）。
  - 4B 的 quality 也在 RTX 上量：使用者確認（020）。
