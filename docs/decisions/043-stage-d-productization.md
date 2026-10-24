# 043 — Stage D 的產品化：verified profile、輸出長度、T 的部署與准入的設計（DRAFT）

- 狀態：**DRAFT**（設計；第 1–7 項待決事項已依建議定案）。實作依文末「實作步驟」以 TDD 進行；驗證研究（待決事項第 8、9 項）等使用者決定。
- 日期：2026-10-24
- 對應：021（R1 不提供覆寫參數，R3 逐位元重現）、027 §4、029（產生器、loader、准入、M2 補記、036 的釘選補記）、030 8.7–8.9、035、036、037（§1 Engineering、§2、§5 第 2 步）、038、039（「產品上的使用」）、042（Stage D 的全部數字）
- 時點：042 之後，任何 Stage D 的產品程式之前。本文不產生數據，也不使用 GPU。
- 來源：本文由三份草稿（保守、產品優先、嚴謹）經三位評審比較後寫成。以保守版為骨架，併入另外兩份的做法，並改正評審找到的錯誤（見「設計審查改正的項目」）。

## 背景

- 042「產品上的使用」：
  - 產生器目前只接受 Stage C 的 session；
  - D-A 的 class 要擴充產生器之後才有 verified profile；
  - 不改正式設定的釘選；
  - D-B 與 D-C 只寫進適用條件；
  - 多 class 准入與開場湧入的處理另寫 decision，也就是本文。
- 037 §5 第 2 步（使用者已核准）：Stage D 之後產生新的 verified profile，只加入、不取代；據此設計多 class 准入，驗證後才上線。
- **任何設計都必須遵守的規則**：
  - 產品上限只能來自 verified profile，verified profile 只能由產生器寫出（027 §4、029、021 R1）；
  - 正式釘選 `verified-formal-20261017-c1b.json` 不變，要改由使用者決定（037 §2）；
  - 研究程式不 import capserve；凍結的研究模組只能 import，不能修改；
  - argv 一變，server fingerprint 就變，所以 profile 只適用於量測它的 server（T 有自己的 profile，030 8.8）；
  - 不宣稱超出量測範圍的事。量過的只有：封閉迴路、准入之後的延遲、實際量過的輸出長度。同一台 server 上多個 class 同時並發從未量過；
  - 029 規定：要先有驗證過的 mixed 容量，才做多 class 或 token 加權的准入；
  - 新的量測一律先事先登記。GPU 時間稀缺，目前 GPU lock 由另一個專案持有，不得搶占。

## 結論摘要

1. **產生器**接受 Stage D 的 D-A session。
   - 每個 (session, server) 產生一份 profile，共三份新的 verified profile；
   - 三份都只 commit，不釘選；
   - D-B、D-C 與中止的 `formal-20261023-d-a1` 都拒絕。
2. **輸出長度**：verified class 的 `max_output_tokens` 大於 128 時，不能作為 active class。
   - 理由：U8K 在輸出 512 時 C_PROD 為 12，在輸出 128 時 C 9 就已經 FAIL。
   - 要放寬，可以另寫「包絡」規則的 decision，或做驗證研究 V1。
3. **T profile**：只有 `mixed-8k-t` 可以服務（030 8.8）；`uniform-8k-t` 只作報告。
   - 本次 deploy 仍只啟動 B0，遇到 T profile 會明確拒絕；
   - T 的部署要檢查什麼，寫在第 6 節。
4. **ABOVE_CKV、WINDOW_SENSITIVE、C 32 的週期停頓**：只用 analysis.json 原有的欄位，加上一組封閉的 `not_covered` enum，帶進 profile，並在 `/readyz` 顯示。
   - **D-B 與 D-C** 只寫進 `service/README.md` 的適用條件，c1b profile 的位元組不變。
5. **多 class 准入與限制同時 prefill 的閘門**：本文只設計候選方案（`weighted-v1` 與 prefill pacing），都不上線。
   - 上線前要通過驗證研究，研究規模與 GPU 時數列在 7.4。
6. **本次可以上線、不需要新量測的**（0 GPU h）：
   - 產生器；
   - loader 與 schema；
   - 三份 profile（不釘選）；
   - 輸出長度與 T 的拒絕規則；
   - `/readyz` 揭露 flags 與 `not_covered`；
   - gateway 的 keep-alive 修正（不宣稱任何容量）；
   - README 的適用條件。

## 1. 產生器接受 Stage D（`service/capserve/profile_gen.py`）

### 1.1 介面

```text
python -m capserve.profile_gen --session results/stage_d/<sid> --freeze-commit <40 hex> --out-dir service/profiles
```

- **依 `--session` 的路徑分派**：
  - `results/capacity/` 走既有的 Stage C 路徑，用 `--out`，行為與輸出都不變；
  - `results/stage_d/` 走新的 `generate_stage_d`，用 `--out-dir`。
- `--out` 與 `--out-dir` 互斥，必須給其中一個；搭配錯的路徑時 exit 2。
- **檔名固定**為 `<out-dir>/<profile_id>.json`，沒有選擇空間。
  - 不加 `--server` 或 `--server-condition` 這類選擇器（021 R1）；
  - `test_no_override_options` 的集合只多一個 `--out-dir`。
- **一次寫出整個 session**：一個 session 的所有 server 的 profile 一起寫出。
  - 全部寫出或全部不寫：先讓每份文件都通過 loader，並確認每個目標檔都不存在，才開始寫；
  - 寫到一半失敗時，刪掉已寫出的檔案。
- 不建通用 framework，Stage D 的分支放在同一個檔案中。

### 1.2 凍結檢查

- **參數化 `freeze_commit_problems`**：新增關鍵字參數 `freeze_file`、`freeze_key`、`modules`，預設值是 Stage C 的值。
  - `_frozen_at`、`_earlier_freeze_problems`、`_frozen_bytes_problems` 一起參數化；
  - Stage C 的呼叫、行為與測試都不變。
- **Stage D 的呼叫**：`freeze_file = "configs/stage_d.toml"`、`freeze_key = "stage_d"`。
- **`--freeze-commit` 必須符合下列全部條件**：
  - 是 `fd95164f19cb261ccdf54b22ae93359f74b8f6be`：這個 commit 設定 `stage_d = true`，`fd95164^` 為 false，history 中沒有更早的 true；
  - 是 session 的 DEPLOYED_COMMIT `0ec7b9208ea3…` 的祖先；
  - 它的 `stage_d.toml` sha256 等於 session 記錄的值，也等於 checkout 中的檔案（`be7909b1…`）；
  - 下列檔案在凍結 commit、DEPLOYED_COMMIT 與產生器實際 import 的檔案三者之間逐位元相同：
    - `stage_d_plan.HARNESS_FILES` 的 5 個檔案。`stage_d_datasets` 即使沒有被 import，也和 checkout 中的檔案比對；
    - Stage C 的 5 個模組：capacity_plan、capacity_analysis、run_capacity、capacity_report、gpu_idle。
- **checkout 的狀態**：
  - `[freeze] stage_d` 為 true，而且 `stage_d_plan.freeze_evidence_problems(cfg, root)` 為空，也就是三個 mini 的 `harness_sha256` 都相符；
  - `configs/capacity.toml` 也已凍結，因為 Stage D 沿用 S2 與凍結的 Stage C 模組。
- **設定檔的讀取**：新增 `load_stage_d_configs(root)`，回傳 (stage_d.toml, pins, 兩者的 sha256)。
  - Stage C 的 `_configs` / `load_configs` 回傳的四元組不變，所以既有的呼叫與測試不受影響。

### 1.3 session 與 analysis

- **位置與狀態**：session 必須在 `results/stage_d/<session_id>`，id 以 `formal-` 開頭；
  `stage_d_report.refusal_problems(...)`（uniform 取 `plan.derived_from`）必須為空。
  中止的 `formal-20261023-d-a1` 因此被拒絕：它沒有 confirm_plan，也沒有 analysis.json。
- **種類**：`plan.kind` 與 `analysis.provenance.kind` 都必須是 `closed`。
  D-B（`open`）與 D-C（`verify`）一律拒絕，訊息為「039 / 042：D-B 與 D-C 不產生 profile」。
- **逐位元重現**（021 R3）：唯讀重跑 `stage_d_report.report(...)`，輸出以 `json.dumps(indent=1) + "\n"` 序列化後，必須和已 commit 的 analysis.json 逐位元相同。
  - `report.advanced_references` 為 true 時，帶入 `formal-20261011-adv` 的參考值；
  - 042 已在記憶體中確認過一次逐位元相同。
- **SLO**：analysis 的 `slo` = `stage_d.toml [slo]` = `capacity.toml [slo]`，都是 S2（037 §3）。

### 1.4 依 server 分組，逐 boot 計算 fingerprint

- **`server_groups(session, pins)`**：
  - 每次 boot 依它的 workload，以 `stage_d_plan.WORKLOADS[w]["server"]` 決定屬於 B0 或 T；
  - 每次 boot 都用記錄的 argv 與 env 計算 fingerprint，env 只取 `core_b_plan.recorded_env` 的 key；
  - sentinel（U2K）的 boot 屬於 B0 組，但不產生 class。
- **每一組的條件**：
  - 組內每次 boot 的 fingerprint 都相同；
  - B0 組等於 `profile.stage_b_fingerprint(pins)`（`98b9f41b…`）；
  - T 組等於新的 `profile.t_fingerprint(pins)`（`564d27ec…`）。
    它直接呼叫凍結的 `stage_d_plan.server_argv_for(pins, "U8K-T", port=PRODUCT_VLLM_PORT)`，不在 profile.py 中自己重組 argv。
- **交叉核對**：同一組內各 workload 的 `analysis.platform.server_argv_sha256` 必須相同（B0 為 `54614663…`，T 為 `56fcc296…`）。
- **不得沿用 Stage C 的 `session_fingerprint`**：它只看第一次 boot。
  - d-a1b 的 `boots[0]` 是 B0 的 `b000-sentinel-start-U2K`。沿用它，T 的 class 會被蓋上 B0 的 fingerprint，而且檢查會通過。
  - 已核對：B0 有 8 次 boot（U32K 6 次、sentinel 2 次），T 有 12 次。

### 1.5 狀態的對應與 evidence

**狀態的對應**沿用 030 8.7：

| C_PROD 的狀態 | profile 中的 class |
|---|---|
| EXACT / LOWER / CENSORED | verified，max_concurrency = max_queue = C_PROD |
| INFEASIBLE | verified 0 |
| UNSTABLE / NT | unavailable，不給數字，也不退回其他數字 |

- `source` 為 `stage_d:<workload>:c_prod:<status>`。這次五個 class 都是 EXACT。
- **margins**：Stage D 的 analysis 沒有 `margins` 欄位，也沒有 `L`，所以不能沿用 Stage C 的 `class_entry`（會 KeyError），改寫新的 `stage_d_class_entry`。
  - 新函式 `stage_d_margins(x)` = `{t: 1 − R for t, R in points[str(C_PROD)]["worst_R"].items() if R is not None}`。
    這和 `capacity_analysis` 寫入 `margins.c_prod` 的公式相同，所以不是新的統計量。
  - **測試**：同一個函式套在 c1b analysis 的四個 workload 上，結果必須等於各自的 `margins.c_prod`。
    例如 U2K 為 1 − 0.5892 = 0.4108。
- **evidence 只取 analysis.json 已有的欄位**，見第 3 節。產品程式不從 derived 檔另外計算任何數字。

### 1.6 拒絕的情況

產生器在下列任一情況下 exit 2，不寫任何檔案：

- 1.2、1.3、1.4 的任何一項不符；
- `kind` 不是 `closed`；
- 任何一組的 fingerprint 不一致，或不是產品 server 的 B0 / T；
- 工作樹有未 commit 的變更；
- `--out-dir` 位於 `results/` 之下；
- 任何一個目標檔已經存在；
- 產生的文件過不了 loader。這代表產生器有缺陷，同樣拒絕。

## 2. 產生的 profile 與 class

| profile_id | server | class | prompt ≤ / 輸出 | 上限（bound exact） | 在 C_PROD 的 margins（短類 / TPOT） | 重點 |
|---|---|---|---|---|---|---|
| `verified-formal-20261023-d-a1b-b0` | B0 | `uniform-32k` | 32,640 / 128 | 2 | 0.488 / 0.358 | C_KV 3；先失守的是 TPOT；第一波最大 TTFT 12.8 s（< 20 s） |
| `verified-formal-20261023-d-a1b-t` | T | `uniform-8k-t` | 8,064 / 128 | 4 | 0.060 / 0.827 | **只作報告**（030 8.8）；離 guard band 只差 0.010 |
| 同上 | T | `mixed-8k-t` | 32,640 / 128 | 4 | 0.114 / 0.212（long 0.420） | WINDOW_SENSITIVE（c_late 為 5）；H-D2a supported |
| `verified-formal-20261023-d-a2-b0` | B0 | `uniform-2k-o512` | 1,920 / 512 | 32 | 0.292 / 0.557 | ABOVE_CKV（C_KV 30）、WINDOW_SENSITIVE（c_full 為 UNSTABLE）；attainment 0.96875 |
| 同上 | B0 | `uniform-8k-o512` | 8,064 / 512 | 12 | 0.575 / 0.526 | C_KV 12；第一波最大 TTFT 16.95 s（6 s 的 2.8 倍） |

- `mixed-8k-t` 的 prompt 上限：`WORKLOADS["M8K-T"]["prompt"]` 是 None，所以取 `core_b_plan.LONG_INPUT`（32,640），和 c1b 的 mixed 相同。
- class 名稱刻意和 c1b 不同（加上 `-t`、`-o512`、`-32k`），所以寫給 B0 `uniform-8k` 的設定不會悄悄拿到 T 的 4，或 O512 的 12。
- **跨 session 的敏感度**（042 已更正）：在約 1.5% 的漂移內，U8K-T 的 C_PROD 可能是 3–4（只會往下），M8K-T 可能是 4–5（只會往上）。
  U8K-T 在 C 6 只以 0.27% 之差 FAIL，這影響的是 C_SLO，不是 C_PROD。

**和 039 / 042 / NEXT.md 字面的差異**（CLAUDE.md：先記錄再處理）
- 字面寫的是「一份 B0 profile（U32K、U2K-O512、U8K-O512），一份 T profile」。本文產生三份，B0 依 session 拆成兩份。
- 理由：
  - 一份 profile 只對應一個來源 session，provenance 與「重新產生」的測試都維持 1:1；
  - 合併要支援多來源的 provenance（schema 要改）；
  - gateway 一次只服務一個 active class，合併沒有產品上的好處。
- 依「待決事項的處理」第 1 項定案為 3 份。

## 3. Loader 與 schema（`service/capserve/profile.py`）

- **schema_version 仍為 1**：新欄位一律選用、預設 None。
  - c1b profile 的位元組、content_sha256 與載入結果都不變；
  - Stage C 的產生器不輸出這些新欄位，所以 c1b 重新產生的結果仍逐位元相同。
- **`ServerSpec.long_prefill_token_threshold: int | None`**：null 或缺少代表 B0；T 必須等於 pins `[advanced] long_prefill_token_threshold`（512）。
- **`Provenance.stage_d_sha256: str | None`**：§9 必填，§8 必須是 null 或缺少。
  §9 的 `capacity_sha256` 記錄產生時 checkout 的 capacity.toml，loader 照舊比對。
- **`ClassEvidence` 的新欄位**（都只作描述，永遠不是上限）：
  - `server`：`"B0"` 或 `"T"`。
  - `output_tokens`：量測時的輸出長度，512 或 128（ignore-eos）。
  - `descriptive`：`{"c_full": Estimand, "c_late": Estimand}`，取 analysis 的 `descriptive.*.c_prod`。
    例如 U2K-O512 的 c_full 為 UNSTABLE、c_late 為 32 LOWER；M8K-T 的 c_late 為 5 EXACT。
  - `at_c_prod`：`{attainment_min, preemptions_max, kv_cache_usage_peak_max}`，取 `points[C_PROD].runs` 中的有效 run。
    U2K-O512 為 0.96875、30、1.0。scope 中註明：preemption 與 KV 是整個 bench process 的值（042）。
  - `hypotheses`：`{id: verdict}`，從 analysis 的 `hypotheses[id][workload].verdict` 原樣複製。
    例如 U32K 為 H-D1 supported、M8K-T 為 H-D2a supported、U2K-O512 為 H-D3 not supported。
  - `not_covered`：封閉 enum 的清單（見第 5 節），由 `not_covered_for(workload, flags)` 決定。loader 要求它等於這個函式的結果，所以手改的清單無法載入。
- **`FrozenConfigs`（NamedTuple）**：
  - 新增 `stage_d: dict | None = None` 與 `stage_d_sha256: str | None = None`。有預設值，測試中既有的位置參數建構 `FrozenConfigs(cfg, CAP_SHA, PINS_SHA)` 不受影響；
  - `frozen_configs()` 在 `configs/stage_d.toml` 存在時讀取它。release 是 `git archive`，會包含這個檔案。
- **`PREREG_SECTIONS = ("§8", "§9")`**：`_verified_problems` 與 `provenance_problems` 依 `prereg_section` 分派。§8 的規則一字不改。
- **§9 的規則**：
  - `[freeze] stage_d` 與 `[freeze] capacity` 都為 true；
  - `stage_d_sha256`、`capacity_sha256`、`pins_sha256` 都等於這份 checkout 的檔案；
  - SLO 等於 `stage_d.toml [slo]`；
  - `profile_id` 為 `verified-<sid>-<b0|t>`，後綴由 `server.long_prefill_token_threshold` 決定；
  - `analysis_path` 為 `results/stage_d/<sid>/analysis.json`；
  - 只接受 D-A 的 5 個 workload，每個 workload 最多一個 class；
  - 名稱與 token 上限等於 `stage_d_class(w)`；
  - `WORKLOADS[w]["server"]` 與 `evidence.server` 都等於 profile 的 server；
  - `evidence.output_tokens` 等於 `WORKLOADS[w]["out"]`；
  - `fingerprint_sha256` 等於該 server 的期望值（B0 為 `stage_b_fingerprint`，T 為 `t_fingerprint`）。
  - U2K-P、U8K-P、`*-var`、`*-gw` 永遠不能成為 class。
- **source 規則一般化**：§8 為 `stage_c:…`，§9 為 `stage_d:…`。
- **保留名稱**：provisional profile 也不得使用 Stage D 的 class 名稱（延伸 036 的規則）。
- **`serving_class` 新增兩條拒絕**，都拋出 ProfileError，和 unavailable 走同一條路徑：
  `create_app` 拋出 ProfileError（不是 ConfigError），`capserve serve` exit 2，`deploy up` 在 preflight 拒絕，任何旗標都不能覆寫。
  1. verified class 的 `max_output_tokens` > `bp.N_GEN`（128）：見第 4 節。
  2. 在 T profile 上，只有 `evidence.workload == "M8K-T"`、而且 `evidence.hypotheses["H-D2a"] == "supported"` 的 class 可以服務（030 8.8：T profile 只用於明確宣告為 M8K 的部署，而且要 C_PROD 大於 B0 的值）。
     `uniform-8k-t` 一律拒絕，訊息為「030 8.8：只報告」。

## 4. 輸出長度 512 與 128

**表示**
- `max_output_tokens` 等於量測時的輸出長度；
- `evidence.output_tokens` 記錄同一個值；
- scope 寫明「exactly N output tokens (ignore-eos)」。

**gateway 的執行方式不變**（`openai_api.parse_request`）
- 沒有給 `max_tokens` 時，補上 class 的上限；超過時回 400 `max_tokens_exceeded`。
- 這只是上限。EOS、stop，以及預設關閉的 thinking，都會讓實際輸出更短，gateway 無法強制「不得更短」。
  用 `min_tokens` 或 `ignore_eos` 強制最短長度會改變產品語意，不採用。

**較短的輸出不一定更安全**（U8K 是反例）
- 輸出 128（c1b）：C_PROD 8；C 9 的 R_TPOT 1.006，FAIL；階段 S 在 C 12 的 R_TPOT 1.333，也是 FAIL。
- 輸出 512：C_PROD 12，先失守的項目從 TPOT 變成短類 TTFT。
- 推論（042）：穩態 TPOT 有一大部分來自新 request 的 prefill 插隊，輸出變短後攤到較少的 token。
- 所以 `uniform-8k-o512` 的 12 遇到較短的輸出時，會落在已經量到失守的並發。

**043 的規則**
- 輸出上限大於 128 的 verified class 不能作為 active class（見 3 的 `serving_class`）。
- 兩個 O512 class 都受影響。
  - `uniform-2k-o512` 的 32 在輸出 128（c1b，C 32 為 PASS）與 512 都量過；
  - 但 129–511 沒有量過，少於 128 的輸出對任何 class 都沒有量過，包括正式的 uniform-2k。

**放寬的兩條路**（目前採拒絕規則；放寬由使用者決定，見「待決事項的處理」第 3 項）
- **包絡規則**：另寫 decision。上限 = min(C_PROD(本身的輸出長度), C_PROD(同一 prompt、輸出 128、同一 server))，8K 為 min(12, 8) = 8，2K 為 32。
  - bound 標成新的 `envelope`，不是 `exact`；evidence 記錄兩個來源 session。
  - 這要支援多來源的 provenance，也只是兩個量測點之間的假設。
- **驗證研究 V1**（7.4）。

## 5. 旗標、C 32 的停頓、D-B 與 D-C 如何成為證據與適用條件

**`not_covered` 的封閉 enum**
- 所有 D-A class 都有：`prompt_length_variation`、`via_gateway`、`open_loop`、`output_length_other_than_measured`、`start_up_wave`、`concurrent_classes`。
- 有 ABOVE_CKV 的 class 另外加 `in_vllm_wait_beyond_c_kv`。
- profile 中只放 enum，不放任何 D-B 或 D-C 的數字。所以 profile 不會宣稱不是它的來源 session 量到的事實。
- `/readyz` 新增 `flags`、`not_covered` 與 `server_variant`；`capserve_profile_info` 新增 `flags` 與 `server_variant` 兩個 label。
  c1b 的 class 沒有 `not_covered`（null），它的適用條件以 README 為準。

**ABOVE_CKV 與 C 32 的週期停頓**（U2K-O512）
- 帶進 profile 的：
  - flags；
  - `c_kv` 30；
  - `at_c_prod`：attainment 0.96875、preemption 30、KV 峰值 1.0；
  - `hypotheses` 中 H-D3 為 not supported；
  - `not_covered` 中的 `in_vllm_wait_beyond_c_kv`。
- 停頓的幅度與位置不在 analysis.json 中，只寫在 043 與 README，並引用 042：
  - 每個 run 剛好 8 / 256 個 request 停頓，3.125%；
  - 時間 4.2–5.2 s，是 SLO 的 2.1–2.6 倍；
  - 位置在 idx 63 + 31k。
  - 這些 request 在 vLLM 內等待，gateway 無法讓它們逾時。
  - 32 → 33 是斷崖（R_short 0.708 → 1.031），guard band 沒有提供預警。

**WINDOW_SENSITIVE**
- 原樣複製旗標，並附 `descriptive`：
  - `mixed-8k-t` 的 c_late 為 5，所以 W 的 4 比較保守；
  - `uniform-2k-o512` 的 c_full 為 UNSTABLE：只要包含開場湧入，就沒有任何驗證過的上限。

**D-B**（不產生 profile，寫進 README）
- uniform-8k 不靠准入控制時，開放迴路的上限是 0.2519 req/s（0.4 λ_ref，EXACT）。這剛好是格點的最低點，差一格就是 INFEASIBLE。
- uniform-2k 為 NT。
- 產品部署中 vLLM 只綁 127.0.0.1:8100，不會在沒有准入的情況下直接對外。
- 產品實際的情況，也就是「gateway 的並發上限加上 Poisson 到達」，從未量過。
- **keep-alive 的說法改正**：gateway → vLLM 這一段，httpx 的 `keepalive_expiry` 2 s 低於 vLLM 的 5 s，所以受到保護。
  - client → gateway 這一段沒有保護：`server.build_config` 沒有設定 `timeout_keep_alive`，uvicorn 預設 5 s；
  - aiohttp（閒置保留 60 s、FIFO 重用）或 httpx（預設 5 s）的 client 連到 gateway 時，仍可能遇到 D-B 推論的同一類競態。
  - 修正方式見 7.2。

**D-C**（只涉及 c1b 的 class；c1b 的位元組不改，寫進 README）

| class | 長度不一（直連） | 經過 gateway |
|---|---|---|
| uniform-8k（8） | PASS：一條 pin 住的長度序列 86–8,015 | PASS：固定 8,064 token |
| uniform-2k（32） | FAIL：4 個 run 中 1 個超過 2 s（2,042 ms），r3 的 0.989 已在 guard band 內；超標集中在第二波，但 W′ 的 P95 仍比 uniform 高約 70% | PASS，但最差 R_TPOT 0.923，離 guard band 只剩 0.027 |

- uniform-8k 的兩種條件是分開驗證的，「長度不一加上 gateway」沒有量過。
- gateway 讓 W′ 的短類 P95 增加：U2K 約 +390 ms（一格），U8K 約 +710 ms（約兩格）。無法拆成 gateway 排隊與上游兩部分。
- D-C 量的是 `capserve_sha256 53488c6d…` 那一版的 gateway。之後 gateway 的變更對延遲的影響沒有量過。
- 新的 Stage D class 沒有任何長度不一、經過 gateway 或開放迴路的證據。

## 6. T profile 與 deploy（`service/capserve/deploy.py`）

**本次**
- preflight 在 `prof.server.long_prefill_token_threshold` 不是 null 時拒絕，訊息為「這個 deploy 只啟動 B0；T 的部署需要使用者決定（043）」。
- 目前 T profile 本來就會因為 fingerprint 不符而被拒絕，這條規則只是讓原因清楚。

**使用者決定服務 T 時，deploy 要檢查的事**（屆時另寫補記，以 TDD 實作）
1. **獨立的正式設定**，例如 `service/deploy/rtx-verified-t.toml`：
   - `production_profile` 釘住 T 的 profile，`active_class = "mixed-8k-t"`；
   - `rtx-verified.toml` 不變；
   - 一張 GPU 一次只能跑 B0 或 T 其中一個。
2. **argv**：`render_vllm` 從 profile 的 `server.long_prefill_token_threshold` 取值。profile 是唯一的來源，設定檔中沒有這個參數（021 R1）。
   - 參數加在 B0 argv 的最後面，順序和 `stage_d_plan.server_argv_for` 相同；
   - provisional profile 一律用 B0。
3. **fingerprint**：等於 profile 的值，也等於 `t_fingerprint(pins)`。
   - Mac 測試再和 d-a1b 中每次 T boot 記錄的 argv 與 env 比對；
   - B0 profile 遇上 T server，或 T profile 遇上 B0 server，都由 fingerprint 拒絕。
4. **vLLM 真的套用了這個參數**：`server_problems` 改用凍結的 `run_stage_d.nondefault_problems(log, kv_bytes=K, threshold=512 或 None)`，只 import。
   - T 的 log 必須有 512，B0 的 log 必須沒有；
   - 這會讓 B0 的檢查變嚴：多檢查 `kv_cache_memory_bytes`，並禁止 observation 與 G 的 key。所以合併前要先以 `rtx-verified.toml` 在 RTX 跑 IT-0 至 IT-6，確認既有的正式部署仍然通過。
   - block 248 × 528、FLASH_ATTN、GDN kernel、MNBT 2048 的檢查照舊。
5. **readiness**：`served_profile_problems` 也比對 `/readyz` 的 `server_variant`。
6. **產品測試 V5**：在 T 上跑 IT-0 至 IT-6，IT-1 確認 log 中有 T 的參數。約 0.25 GPU h，不是容量量測。

- `capserve serve` 是開發用的入口，不知道後端的 argv，無法擋住「profile 接到錯的 vLLM」。c1b 也一樣。README 寫明只有 `deploy up` 保證 server 正確。

## 7. 准入：多 class、token 加權、開場湧入

### 7.1 數據能支持什麼

**已驗證**
- 一次部署服務一個 class，上限是一個全域的並發數。
- 唯一驗證過的「一台 server 接受多種長度」，是單一 class 的固定 90 / 10 mix：
  - c1b 的 mixed-2k / mixed-8k 為 2；
  - T 的 mixed-8k-t 為 4。

**從未量過**：同一台 server 上兩個 class 同時並發。029 的前提還沒有達成。

**依 token 加權**
- 有反例的是含 32K 的組合：H-B4 中 2K 短類的 P95 從 4.26 s 變成 23.2 s；c1b 的 M2K 在 C 3 的 R_short 為 3.314。
- 只有 2K / 8K 的組合是從未量過，不是「被反駁」。
- **事後的觀察**（只作描述）：B0、輸出 128 時，三個 uniform 的 C_PROD × (prompt + 輸出) 都剛好是 65,536（32 × 2,048 = 8 × 8,192 = 2 × 32,768）。
  但這個關係在其他條件下不成立：T 為 4 × 8,192 = 32,768，O512 為 12 × 8,576 = 102,912 與 32 × 2,432 = 77,824，mixed 在 C 3 就失守。所以不能當作預算規則。
- KV 也不能預測容量：
  - U8K-T 的 KV 峰值在 C 4（PASS）為 0.308、C 5（MARGINAL）為 0.385、C 6（FAIL）為 0.462，TTFT 失守不是 KV 造成的；
  - U2K-O512 在 KV 峰值 1.0、超過 C_KV 時仍然通過。

**開場湧入（D5）**
- 只是事後的描述，而且是內插：2K 的 C 5–7、8K 的 C 5–6 都沒有量過。
- vLLM 在 MNBT 2048 下大致依序 prefill。同時到達的第 j 個 request 的首 token 時間約為 j × t_prefill：2K 為 324 ms，8K 為 1,389 ms。不論它在 gateway 還是在 vLLM 中等待，都是如此。
- 所以限制同時 prefill 的量，**不會降低 client 端第一波的 TTFT**。2K、C 32 時，32 × 324 ms ≈ 10.4 s 是串行 prefill 的物理下限。它只會：
  - 把等待從 vLLM 內（無法取消）移到 gateway 的佇列（可以取消，並依 `max_queue_wait_s` 2 s 快速回 429 帶 Retry-After）；
  - 讓「准入之後」的 TTFT 必然變好。這是構造造成的，只報告這個數字就是誤導。

### 7.2 不需要新量測、可以上線的（本次）

- 單一 active class 的 FIFO 准入不變，`admission.py` 不改。
- 第 1–6 節的產生器、loader、三份 profile（不釘選）、輸出長度與 T 的拒絕規則，以及 `/readyz` 的揭露。
- **gateway 的 keep-alive 修正**（產品修正，不宣稱任何容量）：
  - `Settings.keep_alive_s`，預設 75 s，交給 `uvicorn.Config(timeout_keep_alive=...)`。75 s 高於 aiohttp 的 60 s 與 httpx 的 5 s，所以 client 端的連線池會先讓連線過期。
  - 代價是閒置的 socket 保留得比較久。Guard 數的是 request，不是連線。
  - 這會改變 `capserve_sha256`，D-C 的 H-D7 只對應舊的那一版。修正之後在下一次 RTX 產品測試中跑 IT-0 至 IT-6。
- 文件寫明開場湧入的已知事實：
  - 第一波最大 TTFT：2K 為 10.35 s（經過 gateway 為 10.37–10.42 s），8K 為 11.04 s（11.00–11.10 s），uniform-32k 為 12.8 s，uniform-8k-o512 為 16.95 s；
  - 第二波已回到門檻內；
  - 產品不對第一波宣稱 SLO。
- 營運上，既有的 `capserve_queue_wait_seconds`、`capserve_ttft_seconds`（從到達起算）與 `capserve_upstream_ttft_seconds` 已經足以觀察湧入，不新增指標。

### 7.3 候選設計（只在驗證研究中啟用，驗證前不上線）

**A. 多 class 准入 `weighted-v1`**（對應 V4）

| 項目 | 設計 |
|---|---|
| 範圍 | B0；只用 c1b 的 uniform-2k（32）與 uniform-8k（8）；不含 32K（32K 的 prefill 會擋住所有人，而且已有反例） |
| 分類 | 每個 request 歸到能容納它的最小 class：≤ 1,920 token 為 2K，單位 1；≤ 8,064 token 為 8K，單位 4。計數用 local tokenizer（正式設定已是 `local`，在准入之前執行）；byte 上界只會多算，結果更保守 |
| 預算 | 共 32 單位（Σ 1 / C_PROD ≤ 1）；各 class 另有上限 32 / 8；嚴格 FIFO |
| 有效並發 | 全部 2K 等於 U2K 的 32，全部 8K 等於 U8K 的 8（兩者都是量過的狀態）；90 / 10 約 24.6，50 / 50 約 12.8 |
| 判定 | 2K request 對 2 s，8K request 對 6 s（8K 像 Y_long 一樣跨 run 合併）；所有 request 的 TPOT 對 100 ms |
| 啟用 | `[gateway] admission = "weighted-v1"`；`serve` 與 `deploy` 一律拒絕，除非 profile 帶有由 V4 session 產生的 verified `admission` 段，而且參數完全相同；沒有退回值。這個 schema 在 044 設計 |
| 驗證 | V4 時 prefill 閘門**關閉**：開著閘門時，兩端都不再是量過的狀態，准入之後的 TTFT 也會因構造而通過 |

**B. 開場湧入的 prefill pacing**（對應 V2）

| 項目 | 設計 |
|---|---|
| 機制 | `admission.py` 中，permit 之前多一道以 token 計的時間型 token bucket；嚴格 FIFO（隊首要同時拿到並發位置與 token，小 request 不能插隊） |
| 參數 | 2K：burst 9,600、rate 約 5,900 tok/s；8K：burst 32,256、rate 約 5,800 tok/s。來自 D5 的斜率與內插，是事後的描述；在 V2 的事先登記中凍結 |
| 穩態 | C_PROD 下的穩態 token 到達率為 2K 約 4,890 tok/s、8K 約 5,080 tok/s。2K 很少被限制；8K 已達 rate 的約 87%，穩態下偶爾會被限制，這要由 V2 量到 |
| 設定檢查 | burst ≥ class 的 max_prompt_tokens，否則 ConfigError（避免死結）。預設不設定，也就是關閉 |
| 為什麼用 token bucket | 和串流與否無關。另一個做法是在首 token 時釋放：非串流看不到首 token，只能等 response 結束（過度限制），或用 min(結束, 送出 + 門檻)（可能在 prefill 完成前就釋放，等於少限制，不是保守的做法） |
| metrics | `capserve_prefill_wait_seconds`；必須和 `capserve_ttft_seconds`（從到達起算）並列報告 |
| 能主張的 | vLLM 中的並發永遠 ≤ C_PROD，只會延後，不會多准入 |
| 不能主張的 | client 端第一波的 TTFT 變好；它修好了 U2K-var 的 FAIL；穩態的 P95 變好 |

**放棄的多 class 形式**
- **分區並發**：每個混合狀態都沒有量過，而且 029 寫明各 class 的容量不能相加。
- **模式切換**：一次只服務一類，切換前先排空。每次切換都會製造新的開場湧入；排空期間（2K 在 C 32 約 12 s），另一類的 request 會因 2 s 的排隊逾時而被 429。
- **在 class 內依 token 加權，准入超過 C_PROD 的短 prompt**：從未量過。

### 7.4 驗證研究（每項都要 PREREGISTRATION 新的一節與 decision 044 以後的事先登記、新的 harness 模組、mini smoke 與凍結）

**時數的估算方式**
- 每次 boot 的固定開銷約 1.6 分鐘（D-C：總時數 1.56 h 減去 bench 的 1.01 h，共 21 次 boot），加上 bench 的時間；
- bench 的時間取 042 的實績：

  | 點 | bench 約 |
  |---|---|
  | U2K-O512 @ 32 | 215 s |
  | U8K-O512 @ 12 | 425 s |
  | M8K-T @ 4 | 490 s |
  | U32K @ 2 | 1,460 s |
  | U2K-gw @ 32 | 125 s |
  | U8K-gw @ 8 | 318 s |

- 每個研究另加 2 次 sentinel（約 0.1 h）與 mini smoke（約 0.5 h）。

| id | 解鎖什麼 | 設計 | GPU h |
|---|---|---|---|
| V-GW | 某個 Stage D class 進入正式釘選 | 該 class 經過 gateway、在 C_PROD 的固定點驗證，比照 D-C 的協定，n = 5。各 class 可單獨做：U2K-O512 約 0.55；M8K-T 約 0.95（T 的 boot）；U8K-O512 約 0.8；U32K 約 2.3。U8K-T 只作報告，不做 | 0.55–4.2 |
| V1 | O512 的服務（取代包絡規則） | U2K @ 32 與 U8K @ **8**（不測 12：輸出 128 時已知會失守），輸出長度在 [1, 512] 均勻、事先 pin 住，n = 5。要先唯讀確認 vLLM 0.30 的 bench 能指定每個 request 的輸出長度 | 約 1.1–1.5 |
| V2 | prefill pacing | 經過 gateway：U2K @ 32 與 U8K @ 8 開啟 pacing；U2K-var @ 32 的 pacing 開與關各一點；n = 5。終點：W 與 c_full 的類別，准入之後的 TTFT 與從到達起算的 TTFT 並列。另外寫定「慢慢服務」或「快速 429」兩種佇列設定中測哪一種。第一波最大 TTFT 只作描述（依物理可以預期，不當終點） | 約 1.5 |
| V4 | `weighted-v1` | 2K / 8K 的 90 / 10 與 50 / 50 兩個點，pacing 關閉，n = 5，經過 gateway | 約 1.1–1.6 |
| V3 | 經過 gateway 的開放迴路 | U2K-P、U8K-P 經過 gateway，上限為 C_PROD；2–3 個到達率 × n = 5，每個 run 至少 5 分鐘。前提：先另寫 decision 修正 D-B 的 client 連線重用缺陷，並完成 gateway 的 keep-alive 修正 | 約 3–5 |
| V5 | T 的部署 | 在 T 上跑 IT-0 至 IT-6，產品測試，不需要事先登記 | 約 0.25 |

**V4（以及 V2）的事先登記必須寫定的內容**
1. **估計量**：
   - 主要的是准入之後（上游）的 TTFT，和 C_PROD 同一個定義；
   - 從到達起算的 TTFT 並列報告；
   - 另外寫定非平凡條件，排除「全部延後或拒絕也算通過」：429 比例的上限、goodput 的下限，或「W 中 vLLM 的在途數達到政策有效並發的一定比例」。
2. **負載**：雙峰的 2K / 8K 序列，事先產生並 pin sha256；封閉迴路 `--max-concurrency 32`。
   研究用的 gateway 佇列設定（max_queue、max_queue_wait_s）要記錄下來，而且和產品設定分開。
3. **規則**沿用 030 / 039：
   - guard band 0.05，n 5，取最差的 run；
   - 跳過規則；無效的 run 不補跑；
   - sentinel 閘門、CROSS_SESSION_SHIFT、031 的閒置判定與 spill 偵測。
4. **gateway 是被操弄的變因，所以要凍結**：capserve 的准入相關模組與政策設定以 sha256 凍結，這和 D-C 不同。
   研究 runner 以子 process 啟動 gateway，不 import capserve。
5. **逐 request 擷取** gateway 的 JSON access log：`queue_wait_s`、`upstream_ttft_s`、`ttft_s`（`guard.RequestRecord` 已有這些欄位）。對應到 bench request 的方式在事先登記中寫定。
6. **結果對應到產品的規則**：
   - 只有每一個點都 PASS，產生器（新的 validation-session 路徑）才寫出帶 `admission` 段的 verified profile；
   - FAIL、MARGINAL 或 NT 時，多 class 維持不可用；
   - 看到數據之後不調參數。
7. **事先寫定**：hypothesis 與預測、時數上限、停止規則。

- 研究依序執行，session 之間留出空檔（042 揭露：GPU lock 曾被連續占用 23.6 h）；不搶占 lock。

### 7.5 明確仍然不可用（寫進 043 與 README）

- 多 class 准入、token 加權准入、prefill pacing；
- O512 class 的服務；
- T 的部署，以及服務 `uniform-8k-t`；
- 經過 gateway 的開放迴路宣稱；
- uniform-2k 在 prompt 長度不一時的 SLO 宣稱。

## 8. 產品上可以得到什麼（依安全餘裕排序）

1. **不寫程式**：
   - 維持正式的 uniform-2k（32），並揭露 D-C 的 FAIL；
   - 或改用 uniform-8k（8），見「待決事項的處理」第 2 項（目前維持 uniform-2k）：長度不一與經過 gateway 分開驗證過、組合沒有量過；短 prompt 也以 6 s 判定；純 2K 流量從 32 降到 8。
2. **uniform-32k（2）**：
   - 適合全部是長文件的流量；
   - 輸出 128、B0，043 的規則允許服務；
   - 沒有經過 gateway 的證據；第一波 12.8 s，低於 20 s；
   - 對 90 / 10 的流量，既有的 mixed-* 也是 2。
3. **uniform-2k-o512（32）**：要先有輸出長度的決定與 V-GW；附 3.1% 的 4–5 s 停頓。
4. **mixed-8k-t（4）**：含 32K 文件的 mix 從 2 變 4；要先有 T 的部署、V-GW 與 V5。
   - V-GW 的理由（外推，只用來排序，不是預測）：在 C_PROD 的最差 R_short 0.886 加上 gateway 的約 +0.7 s，可能進入 guard band。
5. **uniform-8k-o512**：要 V1；12 對較短的輸出不安全。
6. **uniform-8k-t**：不服務（030 8.8，只作報告）。

## 9. 風險與限制

- **c1b 的重新產生**：改 loader 與 schema 可能讓釘選的 c1b profile 無法逐位元重新產生。
  對策：新欄位一律選用；「重新產生」的測試涵蓋 c1b；§8 的規則一字不改。
- **輸出長度的缺口**：gateway 只能強制上限。thinking 預設關閉時短回答很常見，正式的 uniform-2k 也沒有少於 128 個輸出 token 的證據。
- **8K-T 的敏感度**：兩個 C_PROD 在約 1.5% 的漂移內各有一個方向的敏感度。D 的環境本身比 c1b 慢約 1.5%，閒置 VRAM 也較高（651–872 MiB 對 567 MiB），原因沒有記錄。
- **KV 的斷崖**：U2K-O512 與 U8K-O512 在 KV 的邊界都是斷崖（0.708 → 1.031、0.425 → 1.104），guard band 沒有預警；環境稍慢就可能直接越過。
- **正式 class 的已知 FAIL**：釘選的 uniform-2k 在長度不一時已經量到 FAIL，但它的位元組不能改，只能寫進文件。
- **真實流量是開放迴路**：每段閒置之後都會有開場湧入，而經過 gateway 的 Poisson 到達從未量過。
- **gateway 的變更**：會改變 `capserve_sha256`。U2K-gw 的 TPOT 只剩 0.027 的餘裕，任何增加延遲的變更都可能吃掉它。
- **profile 變多**：誤設 active class 的機會增加。正式的釘選只保護 `rtx-verified.toml`；`capserve serve` 不驗證後端的 argv。
- **重新產生依賴 Mac 上的逐位元重現**：產生器在 Mac 上重跑 `stage_d_report.report()`，Python 或依賴的版本改變會讓重新產生的測試失敗。結果是 fail closed，但會擋住 commit。
- **B0 檢查變嚴**：deploy 改用 `nondefault_problems` 後，B0 的檢查會變嚴，必須先跑 RTX 的 IT。
- **D-B 的原因仍是推論**：無效 run 的原因待 raw 確認。不論原因為何，新的開放迴路研究都要先修正協定。
- **GPU 稀缺**：另一個專案目前持有 GPU lock。驗證研究要排隊，而且要和 Stage Q / E 一起排序。

## 設計審查改正的項目

- **keep-alive**：「gateway 對上游的 keepalive_expiry 2 s，所以 D-B 的競態不影響產品」只成立於 gateway → vLLM 這一段。
  `server.build_config` 沒有設定 `timeout_keep_alive`（uvicorn 預設 5 s），所以 client → gateway 這一段仍有同類的競態。已加入修正（7.2）。
- **錯誤類型**：active class 被拒絕時，`create_app` 拋出的是 ProfileError，不是 ConfigError。`Gateway.__init__` 直接呼叫 `serving_class`，`test_an_unavailable_active_class_never_builds_a_gateway` 斷言的也是 ProfileError。
- **uniform-8k 的宣稱**：「uniform-8k（8）已由 D-C 驗證，可直接服務 2K / 8K 混合流量」說過頭了。
  長度不一（一條固定的均勻序列、直連）與經過 gateway（固定 8,064）是分開驗證的；雙峰的 2K / 8K 與兩者的組合都沒有量過。
- **gateway 的影響**：經過 gateway 的增加量，U2K 約 +390 ms（一格），U8K 約 +710 ms，不是「都約一格」。
- **output-robust 的 min 規則**：它要把 c1b 寫進 Stage D 的 profile，需要多來源的 provenance，並改變 030 8.7 的 max_concurrency = C_PROD。
  本文改為：要用就另寫 decision，bound 標成 `envelope`。
- **新的統計量**：不在產品程式中從 derived 檔計算新的統計量（例如 tail）。停頓的幅度只寫在文件中。
- **KV 的數字**：「U8K-T 在 KV 0.31 時失守」是錯的。0.308 是 C 4（PASS）的值，C 5 為 0.385，C 6 為 0.462。
- **V1 的設計**：V1 不測 U8K @ 12。c1b 在輸出 128、C 12 時 R_TPOT 為 1.333，結果可以預期。
- **±1 的敏感度**：「C_PROD ±1」改依 042 的更正：只有一個方向，U8K-T 為 3–4，M8K-T 為 4–5。
- **030 8.8**：原文是「T profile 只用於宣告為 M8K 的部署，U8K-T 只作報告」。三份草稿都沒有執行機制，本文由 `serving_class` 執行。
- **M8K-T 的 prompt 上限**：`WORKLOADS["M8K-T"]["prompt"]` 是 None，32,640 取自 `core_b_plan.LONG_INPUT`。
- **多 class 的設計範圍**：「多 class 與開場湧入都不設計」和 037 §1、§5 第 2 步（使用者已核准）不一致。本文改為：設計候選方案，驗證之後才上線。
- **V4 的設計**：開著 pacing 並以准入之後的 TTFT 判定，PASS 幾乎是構造上的保證；開著 pacing 時，weighted-v1 的兩端也不再是量過的狀態。所以 V4 關閉 pacing，pacing 另由 V2 驗證。
- **pacing 的釋放時機**：非串流以 min(結束, 送出 + 門檻) 釋放並不保守，可能在 prefill 完成前就釋放。改用時間型的 token bucket。
- **V2 的終點**：「第一波最大 TTFT」依串行 prefill 的物理幾乎可以預期，改為只作描述。
- **token 加權**：「被反駁」過度概括。反例都含 32K；只有 2K / 8K 的組合是從未量過。
- **工程細節**：
  - `FrozenConfigs` 的新欄位要有預設值；
  - Stage C 的設定四元組不擴充，Stage D 另寫一個讀取函式；
  - T 的 fingerprint 直接呼叫凍結的 `server_argv_for`。

## 放棄的替代方案

- **把 B0 合成一份 profile**（042 的字面寫法）：要多來源的 provenance，重新產生也不再是 1:1，而且沒有產品上的好處（「待決事項的處理」第 1 項）。
- **在產生器加 `--server` 選擇器**：和 021 R1 有張力。`--out-dir` 一次寫出全部，就不需要選擇器。
- **schema_version 2，並依 prereg_section 把 Provenance 拆成 union**：改動量大。選用欄位加上依 section 分派，已經足夠。
- **強制 `ignore_eos` 或 `min_tokens = max_tokens`**：會改變產品語意與轉送的 body。
- **重新產生 c1b profile 來加入 D-C 的證據**：會改變釘選的位元組（037 §2）。
- **把 U2K-O512 的上限直接降到 31 或 C_KV 30**：這是事後的政策，不是估計量。要做也只能另寫 decision（見下）。

## 待決事項的處理

草稿列了 9 項。屬於工程預設或維持現狀、不改正式釘選也不用 GPU 的，依建議定案（2026-10-24），使用者之後可以改；
會花 GPU 時數或擴張研究範圍的，等使用者決定。

**依建議定案**
1. **profile 的包裝**：每個 (session, server) 一份，共 3 份。provenance 與重新產生都維持 1:1。和 039 / 042 字面（一份 B0）的差異記在第 2 節。
2. **正式設定的 active class**：維持 c1b 的 uniform-2k（32），README 揭露 D-C 長度不一的 FAIL。釘選與 `rtx-verified.toml` 都不變；任何 Stage D 的 class 都不進入正式釘選，直到它的 V-GW 通過。
3. **輸出上限超過 128 的 class**：採 043 的拒絕規則（(a)）。包絡規則與 V1 留待使用者要更長的回答時再決定。
4. **uniform-2k-o512 的 ABOVE_CKV 上限**：第 3 項放寬之前沒有意義，暫不定案。
5. **T 的部署**：現在不部署；deploy 明確拒絕 T profile。
6. **開場湧入的產品政策**：維持現況（慢慢服務，不開 pacing）。
7. **對外文件的 TTFT 定義**：容量以「准入之後」定義，對外文件兩者並列。

**等使用者決定**（新的 GPU 時數與事先登記）
8. **驗證研究**：V-GW（某個 Stage D class 進入正式釘選的前提）、V1（輸出長度）、V2（prefill pacing）、V4（`weighted-v1` 多 class 准入）各自是否執行；V3 延後，先另寫 decision 修正開放迴路的協定缺陷。
   - 建議：本次 0 GPU h。V4 是 037 §5 第 2 步「設計多 class 准入，驗證後才上線」的驗證，建議做，排在已經等 GPU 的 E1-P、E1-D 取回與 Stage Q pilot 之後；
     非平凡條件為「W 中 vLLM 的平均在途數達到政策有效並發的 90%」。其他項目等有產品需求再做。
9. **順序與空檔**：驗證研究和 Stage Q / E 的先後，以及 session 之間的空檔（042 揭露的 23.6 h 連續占用）。

## 實作步驟（TDD：每一步先寫失敗的測試，再實作；全部在 Mac 上，0 GPU h）

1. **Schema 與 loader**（`service/tests/test_profile.py` → `service/capserve/profile.py`）
   - **測試**：
     - c1b 原樣載入，content_sha256 不變；
     - §9 的規則：名稱、token 上限、server 的種類與 fingerprint（`t_fingerprint` 等於 `564d27ec…`）、`stage_d_sha256`、SLO、`profile_id`、`analysis_path`；
     - §9 profile 帶 B0 的 fingerprint 加上 T 的 threshold 時拒絕；
     - `not_covered` 是封閉 enum，而且等於 `not_covered_for` 的結果；
     - provisional profile 不得使用 Stage D 的名稱；
     - 以位置參數建構 `FrozenConfigs(cfg, CAP_SHA, PINS_SHA)` 仍然可用。
   - **實作**：
     - `ServerSpec.long_prefill_token_threshold`、`Provenance.stage_d_sha256`、`ClassEvidence` 的新欄位；
     - `FrozenConfigs` 的預設欄位；
     - `PREREG_SECTIONS`、`STAGE_D_CLASS_NAMES`、`stage_d_class`、`t_fingerprint`、`not_covered_for`；
     - 依 §8 / §9 分派。
2. **`serving_class` 的兩條拒絕**（`test_profile.py`、`test_app.py`、`test_main.py`、`test_deploy.py`）
   - **測試**：
     - 輸出上限 > 128 的 verified class：`create_app` 拋出 ProfileError、`capserve serve` exit 2、preflight 拒絕；
     - T profile 的 `uniform-8k-t` 一律拒絕；
     - `mixed-8k-t` 只有在 H-D2a 為 supported 時才可服務；
     - 正式設定仍釘住 c1b，而且 active class 可服務。
3. **deploy 拒絕 T profile**（`test_deploy.py`）：threshold 不是 null 時，preflight 給出 043 的訊息。
4. **產生器**（`service/tests/test_profile_gen.py`）
   - **以已 commit 的 session 測試**：
     - d-a1b 產生 `-b0`（只有 uniform-32k）與 `-t`（uniform-8k-t、mixed-8k-t），sentinel 不成為 class；
     - T class 的 fingerprint 不是 `boots[0]`（`b000-sentinel-start-U2K`）的值；
     - d-a2 產生一份 b0 profile、兩個 O512 class。數字、bound、flags、`attainment_min` 0.96875、`hypotheses` 都和第 2 節相符，margins = 1 − worst_R；
     - `stage_d_margins` 套在 c1b 上，等於四個 workload 的 `margins.c_prod`。
   - **以下全部 exit 2，而且不寫任何檔案**：
     - d-b、d-c、中止的 d-a1；
     - `--freeze-commit` 給 `fd95164^`、不是 DEPLOYED 的祖先、解凍後再凍結；
     - HARNESS_FILES 或 Stage C 模組的位元組不同；
     - analysis 不是逐位元相同；
     - 工作樹不乾淨、目標檔已存在、`--out-dir` 位於 `results/` 之下；
     - `--out` 配上 Stage D 的 session，或 `--out-dir` 配上 Stage C 的 session；
     - 合成的 session 中，某一組的 fingerprint 不一致。
   - **對應規則**（以合成的 session 測試）：UNSTABLE / NT 為 unavailable，INFEASIBLE 為 0。
   - **介面**：`test_no_override_options` 的集合為 `{"--session", "--out", "--out-dir", "--freeze-commit"}`。
   - **實作**：
     - 參數化 `freeze_commit_problems`；
     - `load_stage_d_configs`、`server_groups`、`stage_d_margins`、`stage_d_class_entry`、`generate_stage_d`；
     - 全部寫出或全部不寫。
5. **重新產生的測試**：`test_every_committed_verified_profile_is_what_the_generator_writes` 依 `prereg_section` 分派。
   - §8 用 `generate`；§9 用 `generate_stage_d`，並依 `profile_id` 取出對應的那一份；
   - c1b 除了 `generator_commit` 之外逐位元相同。
6. **commit 程式碼，然後產生 profile**：在乾淨的 HEAD 上執行
   `python -m capserve.profile_gen --session results/stage_d/formal-20261023-d-a1b --freeze-commit fd95164f19cb261ccdf54b22ae93359f74b8f6be --out-dir service/profiles`，
   d-a2 也一樣；再另外 commit 這三份 profile，不改 `rtx-verified.toml`。
7. **揭露**（`test_app.py`、`test_metrics.py`）：
   - `/readyz` 的 `flags`、`not_covered`、`server_variant`；
   - `capserve_profile_info` 的 label；
   - `served_profile_problems` 不受影響。
8. **keep-alive**（`test_server.py`、`test_config.py`）：
   - **測試**：
     - `build_config` 帶入 `timeout_keep_alive = keep_alive_s`，預設 75 s；
     - 整合測試：閒置 6 s 後重用連線，request 仍正常完成。
   - **之後**：在下一次 GPU 空閒、取得 lock 的產品測試中，以 `rtx-verified.toml` 跑 IT-0 至 IT-6（不是容量量測）。
9. **文件**：
   - `service/README.md`：第 5 節的適用條件，包括改正後的 keep-alive 說法；三份新 profile 都「不是正式設定」；第 7.5 節的不可用清單；
   - 042「產品上的使用」加上指向 043 的註記；
   - NEXT.md。
10. **完整測試與收尾**：
    - 完整測試通過，`test_research_isolation`、`test_stage_d_frozen`、`test_stage_c_frozen`、`test_core_untouched`、`test_frozen_untouched` 都不變而且通過；
    - 獨立審查；commit；push 到 private 的 origin（不 force）。
11. **之後，依使用者的決定**：
    - T 的部署（第 6 節的 1–6，加上 V5）；
    - 包絡規則或 V1；
    - V2 / V4 的事先登記（PREREGISTRATION 新的一節與 decision 044）：先寫失敗的測試與 harness，mini smoke，凍結之後才收正式數據；
    - `weighted-v1` 與 pacing 的程式，在它們各自的事先登記凍結之前以 TDD 完成，不提前寫。
## 使用者決定（2026-10-24）

- **待決事項第 8 項**：V4（`weighted-v1` 多 class 准入）、V-GW（Stage D class 經 gateway 的固定點驗證）、V1（輸出長度在 [1, 512] 不一）、V2（開場湧入的 prefill pacing）**都做**。V3 依建議延後。
- **待決事項第 9 項**：排在 E1-P、E1-D 取回與 Stage Q 的 RTX pilot 之後；session 之間留出空檔，不再連續占用 GPU。
- **估計的 GPU 時數**（7.4 的估算，不含重跑）：V4 約 1.1–1.6 h、V-GW 全做約 4.2 h、V1 約 1.1–1.5 h、V2 約 1.5 h，各加 mini smoke 約 0.5 h，合計約 10–11 h。
- **下一步**：另寫 decision 044 與 PREREGISTRATION 新的一節（事先登記、harness、mini smoke、凍結）；
  `weighted-v1` 與 pacing 的產品程式在 044 凍結之前以 TDD 完成。驗證通過之前，這些功能都不上線，正式釘選也不變。
