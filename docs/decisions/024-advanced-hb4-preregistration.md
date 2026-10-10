# 024 — Advanced 事先登記：隔離 H-B4 的成因（RTX，vLLM 0.30.0）

- 狀態：Accepted（草稿已 commit；K3 的實測 block 數由 smoke FS1 決定後，連同 `freeze.advanced = true` 凍結）
- 日期：2026-10-10
- 對應：decision 023（Gate 通過，範圍只限 H-B4）；SPEC「Advanced」；PREREGISTRATION §2 H-B4、§4、§6、新增的 §7；007、011、013、019、022
- 時點：任何 Advanced 數據之前。Core 的所有結果已 commit。

## 背景與設計過程

- Core 的 H-B4：在 C_knee（2K 為 25、8K 為 8）加入 10% 的 32K request，短 request 的 P95 TTFT 變為 uniform 的 5.4 倍（2K）與 2.8 倍（8K）。
  KV 使用率峰值 99.6% / 99.2%，沒有 preemption。007 把機制列為 conjecture。
- 設計由 workflow 完成：三個獨立方案（拆解優先、因子實驗、風險優先）→ 三位評審（因果識別、可行性、事先登記與範圍）→ 綜合。
  三位評審都選風險優先案為骨幹；綜合時修正所有確認的違規，並逐條對照已安裝的 vLLM 0.30.0 原始碼（檔名與行號寫在各條）。
- **揭露**：設計者看過 Core 的結果。用到的 Core 數字只有三類：C_knee、bench 時長（估時間與設逾時）、restart SE 的量級。
  T 與 K3 只由 Stage B 的 block 算術（248 / 528、C_KV）與原始碼推得；門檻沒有依任何 Core 延遲結果調整。
- 綜合者的摘要：

> 最終設計以三位 judge 一致選出的 risk 案為骨幹，修正了所有確認過的違規，並接上 decompose 的觀測層。觀測層只作描述性的機制證據，不決定任何 verdict。
> 
> **設計**
> - 共 5 個 server condition：
>   - B0：Core argv 加觀測。
>   - T：`--long-prefill-token-threshold 512`。
>   - G：`--no-scheduler-reserve-full-isl`。
>   - TG：T 與 G 同時套用。
>   - K3：`--kv-cache-memory-bytes` 3 GiB。
> - 每個 condition × L {2048 @ C 25, 8192 @ C 8} × mix {uniform, mixed} × 3 次 restart。
> - **每次 boot 只跑一個 bench**，與 Core mixed 相同，共 60 次 boot。3 個 block 內以 seed 隨機排列。
> - T×G 在 K = 4 GiB 構成完整 2×2，可以識別串聯瓶頸。T = 512 由規則推得：MNBT ÷ (C_KV(32K) + 1) = 2048 ÷ 4。這修正了 T = 1024 的錯誤前提：兩個長 request 同時 prefill 時，T = 1024 仍會占滿 budget。
> - 移除判定用 log 罰分 ρ = ln(M/U)，同時要求 M 本身下降並超出 noise（M guard）。新增 R 類「只有相對下降」，讓結果不會模稜兩可。
> - KV 只能往下調，所以「KV 主因」一律標為推論級（修正 R5 的過度宣稱）。逾時一律記 NT，不再計入任何證據。
> 
> **觀測**
> - OTLP 的 `llm_request` span：`time_in_queue` 與 `time_in_model_prefill`，以確定性的 `--request-id-prefix` 與 client 逐筆對齊。
> - `--enable-logging-iteration-details` 的逐 step 紀錄，由 runner 在讀取時打 wall-clock 時間戳。不新增任何 `VLLM_*` 變數。
> - 以上取代 risk 誤用的 `iteration_tokens_total`（在 first token 時一次計入整段 prompt）與 `waiting_by_reason{capacity}`（等於 `len(waiting)`）。
> 
> **Core 程式**
> - Core 模組一行都不改，全部用新模組 `adv_*`。
> 
> **GPU 時間**
> - 預期約 6.8 h，加 smoke 約 7.3 h。
> - 逾時設為 Core 時長 × 2 + 60 s，所以上限約 8.1 h。
> 
> **已唯讀核對 vLLM 0.30.0 原始碼**
> - `scheduler.py`：
>   - running 迴圈 612–807。
>   - threshold 截斷在 661–665 與 1066–1068。
>   - waiting 迴圈的條件 `token_budget > 0` 在 858。
>   - full-ISL 失敗後 break 在 1158–1180。
>   - preempt 後 recompute 並 prepend 回 waiting，在 1477–1519。
> - `kv_cache_manager.py` 508–524：只檢查目前的 free block，不預留。
> - `block_pool.py` 869–880：`usage = 1 − free/(nb − 1)`。
> - `stats.py`：`scheduled_ts` 只記第一次 scheduled。
> - `output_processor.py` 782–842：span 的各個屬性。
> - `otel.py` 94–101：需要 `http/protobuf`。
> - `core.py` 640–719、`config/vllm.py` 589–597：async scheduling 下 batch queue = 2。
> - `config/scheduler.py` 的 hash 只含 MNBT 與 `max_num_seqs`。
> - `fa_utils.py`：sm_120 走 FA2，不支援 fp8 KV。
> - `bench serve`：`--request-id-prefix` 預設為隨機值；warmup 與 ready-check 不帶 id。
> - WSL venv 已有 OpenTelemetry 1.44.0。

- **澄清**：摘要中「觀測層不決定任何 verdict」指的是不決定 T / G / TG 的移除類別（A5）。
  觀測層實際上用於 H-C4(ii)（准入受阻份額）與 H-C5（分解與操弄的一致性），並由 A10 的完全相等檢查把關；分解無效時這兩項記為 NT。

## 決策

### Hypotheses

#### H-C0

- **陳述**：重現閘門：在同一個 Advanced session 內，B0 在 L = 2048（C 25）與 L = 8192（C 8）仍重現 H-B4。B0 = Core B 的 server argv 原樣，只加兩個觀測 flag 與一個 OTEL env。
- **預測**：每個 L 直接呼叫 core_b_analysis.h_b4_values，輸入為 B0 的 uniform 與 mixed 各 3 次 restart 的短類 P95 TTFT。預測：增幅 ≥ 20%，且 |M−U| > 2√(SE_M²+SE_U²)。
- **推翻與判讀**：某個 L 的增幅 < 20%，或差距在 noise 內：該 L 的 H-C1 至 H-C5 全部記為「不可解讀」，不做歸因。另一個 L 照常分析。Core 的判定與 decision table 都不變（023）。
- **門檻依據**：逐字沿用已凍結的 H-B4 規則（20% 加 noise 規則），只用來確認現象在觀測與新 session 下仍然存在。所有因果對比都以 in-session 的 B0 為參照，不用 Core 的數字，因為不同 session 的日期、部署 commit、Windows idle VRAM 都可能不同。Core 的效果是 5.4 倍與 2.8 倍，觀測必須消掉 ≥ 95% 的效果才會讓閘門失敗。因此不另外跑無觀測的 B0（可省約 1.3 h），失敗時接受「無法區分觀測與 session 效應」這個限制。

#### H-C1

- **陳述**：Batching（chunked prefill 的 budget 壟斷，BM）是主要成因。V1 先排 running。T = 0 時，prefill 中的長 request 取 min(剩餘 prompt, 剩餘 budget)（scheduler.py:661-665）。budget 歸零後 waiting 迴圈不執行（scheduler.py:858 的 while … token_budget > 0），所以長 request 約 16 個 step 的 prefill 期間，短 request 都無法被准入。
- **預測**：cat(T) = M，條件是同時滿足：
- f_T = 1 − ρ_T/ρ_0 ≥ 0.5，其中 ρ = ln(M/U)；
- ρ_T < ρ_0 超出 noise；
- M_T < M_0 超出 noise。
- **推翻與判讀**：依 cat(T) 判讀：
- N 或 W：在該 L，BM 不是實質成因。
- P：BM 是貢獻者，不是主因。
- R（ρ 下降但 M 沒下降）：不算支持，標記「chunk 上限的副作用或 uniform 位移遮蔽」。
- NT：不可檢定。
- **門檻依據**：門檻：
- 0.5：移除過半，這是 SPEC「主要成因」的事前定義。
- 0.2：H-B4 的實質門檻，套用在 log 尺度。

用 log 比值，是因為 H-B4 以比值定義，而 T 會合理地改變 uniform。M guard 用來擋住「靠 uniform 變差造成的假移除」。

T 由規則推得：T = MNBT ÷ (n + 1) = 2048 ÷ 4 = 512。n = C_KV(32768) = ⌊247/66⌋ = 3，是 §6 的凍結值，也就是 KV 能同時完整容納的長 request 數。在 ≤ 3 個長 request 同時 prefill 的 step，至少保留 2048 − 3×512 − (C−1) ≥ 488 個 token 給其他 request。

不用 T = 1024：每個 request 各自截斷（scheduler.py:661-665），兩個長 request 同時 prefill 就會用掉整個 2048，移除不完整。

沒有用到任何延遲數字。

#### H-C2

- **陳述**：Scheduler 的准入閘門是主要成因。scheduler_reserve_full_isl = True（config/scheduler.py:172）時，waiting 中的長 request 要等目前的 free block ≥ 整段 prompt 的需求：65 = 62 個 attention block + 3 個 GDN block（kv_cache_manager.py:508-524；只檢查不預留，full_num_tokens = request.num_tokens）。allocate_slots 回傳 None 後，FCFS 會 break（scheduler.py:1158-1180），排在後面、原本放得下的短 request 也一起等。
- **預測**：cat(G) = M（定義同 H-C1），而且 uniform 特異性成立：|ln(U_G/U_0)| ≤ 0.10。
- **推翻與判讀**：- N：閘門不是實質成因。
- W，而且 G-mixed 的 preemption > 0（run 皆有效）：記為「閘門具保護作用」，作為 KV 證據 (iii)，不是 HOL 的證據。
- uniform 檢查失敗：判定標為 qualified，不能觸發 R2。
- 逾時或 crash：記為 NT，不當成任何證據。
- **門檻依據**：0.5 與 0.2 同 H-C1。

source 預測 G 不影響 uniform：在 C_knee，uniform 永遠放得下（2K 最多 175、8K 最多 152 個 block，都 ≤ 247），所以兩種准入規則下排程完全相同。0.10 約為實質門檻 20% 的一半（log 尺度），只用來抓意外的副作用。

G 是在 T = 0（Core 設定）下估計的主效果，T 下的效果由 TG 給出。這同時修正了 factorial 案「只在 P1 測 scheduler」的問題。

#### H-C3

- **陳述**：串聯瓶頸：BM 與准入閘門串聯，移除任一個都會暴露另一個。
- **預測**：cat(TG) = M，而且 cat(T) ≠ M、cat(G) ≠ M。
- **推翻與判讀**：- cat(TG) ≠ M：沒有串聯的證據。
- 若某個單一移除已是 M：成因是單一的（H-C1 或 H-C2），本假設不適用。
- **門檻依據**：沿用同一個「過半」規則。{T0, T512} × {gate on, gate off} 在 K = 4 GiB 構成完整的 2×2，可以直接讀出交互作用，不需模型假設。

交互作用 I = (ρ_0 − ρ_T) − (ρ_G − ρ_TG) 只作描述，SE = √ΣSE²。事前聲明：n = 3 時交互作用的檢定力有限。

#### H-C4

- **陳述**：KV 容量：在 Core 操作點，mixed 受 KV 限制，短 request（或排在前面的長 request）在等 free block。
- **預測**：(i) 劑量：K3（3 GiB，預期 186 個 block）下，同時滿足：
- a = ρ_K3/ρ_0 − 1 ≥ 0.20；
- ρ 與 M 都上升，且超出 noise；
- |ln(U_K3/U_0)| ≤ 0.10。

(ii) 觀測：B0-mixed 尾端短 request 的 server queue time 中，落在「准入受阻 step」的比例 ≥ 0.5（3 次 restart 平均；3 個 run 的分解都必須有效）。准入受阻 step 的條件：
- 有 request 在排隊；
- 該 step 排程的 token 數 < 2048；
- head 的需求 > 估計的 free block。
- **推翻與判讀**：- (i) 與 (ii) 都不成立：記為「KV 未顯示受限或敏感」。
- 只有一個成立：記為「部分證據」。
- (i) 的 uniform 檢查失敗：(i) 標為 qualified，視為不成立。
- (ii) 的分解無效：(ii) 記為 NT，視為不成立，結論標為 incomplete。
- **門檻依據**：KV 只能往下調，因此只能做敏感度檢驗加上觀測，任何「KV 主因」的結論都是推論級。往上調不可行：
- 熱 boot 的可用量上限是 4.14 GiB，K 已經取 4.0。
- WSL2 超量時會 spill，不會 OOM。
- sm_120 走 FA2，不支援 fp8 KV（fa_utils.py:96-108、282-307）。

0.20 是 H-B4 的實質門檻。

K3 的規則：滿足 C_KV(L;K) ≥ C_knee(L)（2K 與 8K）、num_gpu_blocks ≥ 128（019）、≥ 67 的最小 256 MiB 倍數。
- 3 GiB：預期 186 個 block，C_KV 為 26 / 9，符合。
- 2.75 GiB：預期 170 個 block，C_KV(2K) = 24 < 25，不符合。

K3 下 uniform 仍全部放得下；mixed 只要有 1 個長 request 就受限（2K：66+24×7 = 234 > 185；8K：66+7×19 = 199 > 185）。所以 K3 是只對 mixed 加壓的強劑量。

(ii) 的 0.5 也是「過半」。

#### H-C5

- **陳述**：一致性：分解得到的主導類別，與操弄的結果一致。
- **預測**：在 B0 中取尾端 ΔQ（mixed − uniform）的主導類別 c*，條件是 ΔQ_c*/ΔQ ≥ 0.5，候選為 budget-long、HOL-long、KV-short。
- c* = budget-long：cat(T) ∈ {M, P}。
- c* ∈ {HOL-long, KV-short}：H-C4 (i) 成立。
- **推翻與判讀**：對應不成立：記為「分解與操弄不一致」，以操弄的結論為準並標記。

下列情況記為 NT：
- 沒有主導類別；
- B0 任一 run 的分解無效；
- 改用 τ_j = t_{j−1} 與 u_{j−1} 重算後 c* 改變。
- **門檻依據**：本假設檢驗「分解作為機制歸因」是否可信，0.5 與其他「過半」規則一致。

HOL-long 對應到 KV，而不是 G。原因：在 FCFS 下，長 request 等的是 block；G 只會讓它提早進場，block 不夠時會 preempt 自己再重算（scheduler.py:745-795、1477-1519）。所以真正能檢驗的是 KV 敏感度。

### 操弄（server condition）

- B0（參照）：core_b_plan.server_argv 原樣，即 Core 的 K 4294967296（248 個 block）、MNBT 2048、max-num-seqs 128、FLASH_ATTN、async scheduling、prefix caching 關閉等。只在後面加上：
- --otlp-traces-endpoint http://127.0.0.1:4318/v1/traces
- --enable-logging-iteration-details
- server env OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf（otel.py 預設 grpc，stdlib receiver 收不了）

不加 VLLM_LOGGING_CONFIG_PATH，ms 時間戳改由 runner 讀 log 時打上。所有 condition 都帶相同的觀測。

已查證：ObservabilityConfig.compute_hash 的 factors 是空的，SchedulerConfig.compute_hash 只含 MNBT 與 max_num_seqs，所以觀測、T、G 都不改變 compile cache key。
- T（batching）：B0 加 --long-prefill-token-threshold 512，MNBT 固定 2048。
- running 與 waiting 兩個迴圈都會截斷（scheduler.py:661-662、1066-1068），所以短 request 也被切：2K 短 request（1920）切成 4 段，8K 短 request（8064）切成 16 段，長 request（32640）切成 64 段。
- step 大小、activation、CUDA graph、compile hash 都不變。
- G（scheduler 准入閘門）：B0 加 --no-scheduler-reserve-full-isl（BooleanOptionalAction，arg_utils.py:1632）。
- 新 request 只要第一個 chunk 的 block 放得下就准入：T = 0 時為 ⌈min(剩餘 budget, prompt)/528⌉ + 3，T = 512 時為 4 個。
- 已知機制：block 不夠時 preempt running[-1]，可能是長 request 自己，之後 recompute 並 prepend 回 waiting。preemption 全部計數。
- TG：同時加 T 與 G，完成 {T0, T512} × {gate on, gate off} 的 2×2。
- K3（KV 劑量）：B0 的 --kv-cache-memory-bytes 改為 3221225472（3 GiB，經 Core 的 server_argv 參數傳入，不改 Core 程式）。
- 實際 num_gpu_blocks 由 smoke 從 /metrics 讀出後凍結，預期 186。
- 若實測不符合 H-C4 的規則，以 256 MiB 為步長往上調。
- block_size（528）、kernel、numerics 都不變。
- 所有 condition 都固定不變：
- 模型與 revision、dtype、--language-model-only、max-model-len 32768、max-num-seqs 128、balanced、async scheduling、FLASH_ATTN、U 0.91、--no-enable-prefix-caching（mamba_cache_mode none，沒有 block-aligned split；scheduler.py:340-341）、gdn triton、generation-config vllm、§4 env、pin memory 0。
- uniform random seed 20261008，mixed 使用 pin 過 sha256 的檔案並加 --disable-shuffle。
- client flags 與 Core 相同，只多一個確定性的 --request-id-prefix adv-<cond>-<L>-<mix>-r<r>-。Core 原本就會送 x-request-id，只是 prefix 是隨機的（serve.py:1845-1850），所以只有字串不同。
- C 由 formal-20261009-coreb-uniform 經 core_b_analysis.knee 重算：2K 為 25，8K 為 8。
- N = max(10C, 200)，warmup = C。
- 不操弄的項目與理由：
- fp8 KV：sm_120 的 FLASH_ATTN 走 FA2，不支援；要用就得換 backend，同時改變 kernel 與 numerics。
- 調高 K：沒有 VRAM 餘量。
- MNBT 8192：會改 compile hash、step 大小與 activation；而且設了 kv_cache_memory_bytes 時 gpu_worker.py:542-561 會跳過 profiling，spill 偵測不到。
- priority policy：client 不送 priority。
- watermark：只會讓准入更嚴。
- 關閉 chunked prefill：需要 32K 的 budget。
- allocator：沒有可替換的旋鈕，hybrid alignment 也把 block size 固定為 528。
- C、dataset、max_num_seqs：都不改。

### 格子、run 數、順序與逾時

**格子與 run 數**
- 5 個 condition {B0, T, G, TG, K3} × 2 個 L {2048 @ C 25, 8192 @ C 8} × 2 個 mix {uniform, mixed} × 3 次 restart。
- 共 60 次 server boot，**每次 boot 只跑一個 bench**，與 Core mixed 的 protocol 相同。分析單位 = restart，每格 n = 3。
- 每次 bench：N 為 250 / 200，warmup 為 25 / 8。

**執行順序**
- 分成 3 個 block，r = 1..3。每個 block 包含 20 個 (condition, L, mix)，各出現一次，順序以 seed 決定：random.Random(int(sha256("20261008:advanced:block:r"), 16)).shuffle。
- B0 因此穿插在其他 condition 之間，condition 不會與時間混淆。
- 不做提前停止，也不依結果決定任何事。

**時間估計**（以 Core 實測為準）
- 單次 bench：uniform-2K 101.6 s、mixed-2K 265 s、uniform-8K 319 s、mixed-8K 425 s。一個 condition × 一次 restart 的 bench 合計 1110.6 s。
- 每次 boot 的額外開銷約 90 s：mixed2 實測熱 boot 約 65–75 s，再加 span flush 最多約 6 s，以及 env 擷取。
- 基準：15 × 1110.6 s + 60 × 90 s = 6.13 h。
- 假設 12 個操弄格的 mixed run 平均慢 30%，再加約 0.69 h，正式 run 預期約 6.8 h。
- smoke 約 0.5 h，**總預期約 7.3 h**。

**單一 run 的逾時**
- 取 Core 同格時長 × 2 + 60 s，再往上取整到 60 s 的倍數：uniform-2K 300 s、mixed-2K 600 s、uniform-8K 720 s、mixed-8K 960 s。
- 逾時的 run 記為 invalid（ok = false，reason = timeout）。
- 同一個 (condition, L, mix) 格一旦出現無效 run，就不再跑該格剩下的 restart。依 013，該格已經是 NT，這只省時間，不改變任何判定。跳過的 run 照實記錄。
- **上限約 8.1 h**，前提是 12 個操弄格的 mixed run 全部剛好跑到逾時，而且沒有省下時間。

**執行規範**
- 正式 run 全部在一個 formal session 內，在 flock -n /tmp/ai-portfolio-gpu.lock 下執行。lock 被占用時標記 blocked，不搶占，不刪除他人的 lock。
- 每次 boot 前檢查 GPU 閒置（memory.used ≤ 1000 MiB）。
- session 進行中不重新部署。
- env 無效的 session 整個重跑，不做部分補跑（013）。

### 量測

- 主要量：每個 run 的短類 P95 TTFT，即 input = L − 128 的 request，以線性內插計算。直接重用 run_core_b.summarize_result 與 core_b_analysis.short_class_p95_ttft_ms，同時得到 Core 的所有驗證與百分位欄位。
- Client 描述量（每個 run，由凍結的程式在執行時計算）：
- 短類 P50 / P99 TTFT。
- 長類（32640）P50 / P95 TTFT：長 request 承擔的代價。
- 短類與全部 request 的 P95 TTFT 與 TPOT。
- request / output / total throughput、duration、completed / failed。
用途：每個 M / P 判定都要並列報告 Δthroughput 與 Δ長類 P95，用來分辨「重新分配」與「移除」。
- Server 逐 request：OTLP 的 llm_request span（output_processor.py:782-842）。
- 收集方式：runner 內的 receiver 執行緒（127.0.0.1:4318，http/protobuf，用 venv 的 opentelemetry.proto 解碼）。原始 body 存在 raw/，sha256 寫入 manifest；只有 llm_request span 解成 JSONL。
- 欄位：
  - gen_ai.request.id（cmpl-<prefix><i>-0）；
  - usage.prompt_tokens / completion_tokens；
  - latency.time_in_queue：Q = scheduled_ts − queued_ts，scheduled_ts 只記第一次（stats.py:520-524）；
  - time_in_model_prefill：P = first_token_ts − scheduled_ts，包含 preempt 後重新排隊的時間；
  - time_to_first_token：server TTFT，由 frontend 從 arrival 起算；
  - e2e；
  - span start = arrival，frontend 的 wall clock。
- warmup 與 ready-check 的 request 沒有 prefix（serve.py:840-868），不進主結果，但保留，用來重建 queue。
- Server 逐 step：--enable-logging-iteration-details 的訊息（loggers.py:185-201），內容為 iteration index、context requests / tokens、generation requests / tokens、elapsed ms、KV cache usage %（精確到 0.1%）。
- server 的 stdout / stderr 經 pipe 進 runner 的 reader 執行緒，每行前面加上 time.time() 的 ms 時間戳後寫入 log。與 span 使用同一個 wall clock。
- dummy 行與 0-token step 不出現（core.py:515-559）。
- Prometheus 全文 /metrics，每個 bench 前後各抓一次，計算 delta：
- request_queue_time、request_prefill_time、time_to_first_token、request_decode_time、request_inference_time 的 bucket、sum、count；
- iteration_tokens_total 只作描述，並註明它是在 first token 時把整段 prompt 一次計入（loggers.py:1122-1125），不代表每 step 的大小；
- prompt / generation tokens counter；
- num_preemptions；
- cache_config_info。
這些量都包含 warmup。
- 1 Hz 時間序列（帶 wall 時間戳）：kv_cache_usage_perc、num_requests_waiting、num_requests_running、num_preemptions_total。
- free blocks 的換算為 round((1 − u)(nb − 1))（block_pool.py:869-880）。
- 衍生的描述量：KV-full-while-waiting 份額，條件為 waiting ≥ 1 且 free < 7（2K）或 < 19（8K）；gate-zone 份額，條件為 waiting ≥ 1 且 free < 65。
- waiting_by_reason 只照實記錄，不解讀為 KV，因為它的 capacity 值等於 len(waiting)（loggers.py:1034-1037）。
- 衍生量（run 時由凍結的 adv_decomp 計算）：
- 逐 request：idx、類別、client TTFT、arrival、Q、P、server TTFT、O = client TTFT − server TTFT、tokens。存在 derived/<tag>-requests.jsonl，只有數字，進 git。
- 逐 step：t、index、ctx req / tok、gen req / tok、u。存在 raw/，不進 git，sha256 寫入 manifest。
- 每個 run 的分解摘要寫進 rows.jsonl：
  - 尾端（短類 client TTFT ≥ 該 run 的 P90）的 O / Q / P 平均；
  - 各 step 類別的 Q 份額；
  - unexplained-break 份額；
  - ≥ 4 個長 request 同時 prefill 的 step 份額（T / TG 的殘餘 BM）；
  - 分解有效性 (a)–(d)。
- 操弄檢查與紀錄（每次 boot）：
- log 中出現 Using AttentionBackendEnum.FLASH_ATTN backend.。
- non-default args 包含 'max_num_batched_tokens': 2048，以及兩個觀測 key（'otlp_traces_endpoint'、'enable_logging_iteration_details': True）。
- 該 condition 的 key 必須恰好出現（'long_prefill_token_threshold': 512、'scheduler_reserve_full_isl': False、'kv_cache_memory_bytes': 3221225472），其他 condition 的 key 不得出現。這些值不會被 redact（api_utils.py:271）。
- /metrics 的 num_gpu_blocks 與 block_size 等於該 K 凍結的值（248 / 528，以及 K3 的實測值）。
- T 與 TG：每個 step 都必須滿足 ctx_tokens ≤ 512 × ctx_requests。
- 環境：
- 每次 boot 前後 env_capture：power limit、溫度、clock、版本、repo commit。
- 每次 boot 前檢查 GPU 閒置。
- 記錄 server argv 與 env、session 的 pins / plan sha256、mixed dataset sha256、是否持有 lock。
- memory.used：ready 後與每次 bench 後各記一次。只作描述；沒有任何操弄會增加 device 記憶體。

### 分析規則（A1–A14）

- A1 有效性：
- 只計入同時符合下列條件的 run：
  - ok：completed = N、failed = 0、輸入與輸出長度完全符合；
  - server 就緒；
  - log facts 與 condition 表完全一致；
  - num_gpu_blocks 等於凍結值；
  - T 與 TG 每個 step 都通過 chunk 檢查；
  - 沒有逾時；
  - session 為 formal 且 env 有效。
- 任一格不足 3 次有效 restart：需要該格的比較單位記為 NT，不以其他 run 替補（013）。
- 無效的 run 一律不作為任何證據。
- A2 每格統計：
- U_c、M_c = 3 次 restart 的短類 P95 平均，SE = s/√3。
- ρ_c = ln(M_c/U_c)，SE(ρ) = √((SE_M/M)² + (SE_U/U)²)（§2 的 delta method）。
- A3 noise：兩量 x、y 的差超出 noise 的條件是 |x − y| > 2√(SE_x² + SE_y²)，對 ρ 差與 M 差都適用。不以執行順序配對。
揭露：Core 的 restart SE 約為平均的 0.2–0.4%，預期真正起作用的是實質門檻，而不是 noise。
- A4 H-C0：直接呼叫 core_b_analysis.h_b4_values，輸入為 B0 每個 L 的 {uniform: {r: Y}, mixed: {r: Y}}。某個 L 失敗，該 L 的 H-C1 至 H-C5 全部記為「不可解讀」。另外描述性地報告 B0 與 Core 平均的比值，不作判定。
- A5 移除類別（c ∈ {T, G, TG}）：
- 定義：
  - f_c = 1 − ρ_c/ρ_0；
  - dρ：ρ_0 − ρ_c 超出 noise；
  - dM：M_0 − M_c 超出 noise；
  - uρ：ρ_c − ρ_0 超出 noise。
- 類別：
  - M：f ≥ 0.5 ∧ dρ ∧ dM。
  - P：0.2 ≤ f < 0.5 ∧ dρ ∧ dM。
  - R（只有相對下降）：f ≥ 0.2 ∧ dρ ∧ ¬dM。
  - W：f ≤ −0.2 ∧ uρ。
  - N：其他。
  - NT：缺有效格。
- 邊界值歸入較高的類別。
- f 的 SE 以 delta method 計算，只作描述。
- A6 uniform 特異性：
- G：|ln(U_G/U_0)| ≤ 0.10。
- TG：|ln(U_TG/U_T)| ≤ 0.10。
- K3：|ln(U_K3/U_0)| ≤ 0.10。
- 不符合時，該 condition 的判定標為 qualified，不能作為 M 或 KV (i) 的依據。
- T 合法地改變 uniform，只報 ln(U_T/U_0)，不設檢查。
- A7 KV 證據 e_KV（0–3）：
- (i) 劑量：a = ρ_K3/ρ_0 − 1 ≥ 0.20 ∧ uρ ∧ (M_K3 − M_0 超出 noise) ∧ K3 的 uniform 特異性成立。
- (ii) 觀測：B0-mixed 尾端 Q 中准入受阻（HOL-long 加 KV-short，限 explained）的份額，3 次 restart 平均 ≥ 0.5。3 個 run 的分解都必須有效，否則記為 NT 並視為不成立。
- (iii) 閘門具保護作用：cat(G) = W，3 格皆有效，且 G-mixed 的 preemption 總數 > 0。
- 逾時與無效 run 不計入任何一項。
- A8 尾端與對齊：
- 主結果 span 以 id 中的 prefix 與 client index 一一對應。
- 尾端 = 每個 run 中，client TTFT ≥ 該 run 短類 P90 的短 request（線性內插，含 ties）。
- 另外描述性地報告 ≥ P95 的子集。
- O、Q、P 先算每個 run 的尾端平均，再對 3 次 restart 取平均與 SE；Δ = mixed − uniform（同一 condition）。
- A9 step 分類（演算法事前寫定；只作描述，用於 KV (ii) 與 H-C5）：
- 每個 request（含 warmup）：
  - 准入時刻 s_i = arrival + server TTFT − P；
  - 入列時刻 e_i = s_i − Q；
  - 長 request 的 prefill 區間 = [s_i, arrival + server TTFT]。
- 每個 step j：
  - 區間 I_j = (t_{j−1}, t_j]；
  - 決策時刻 τ_j = t_{j−2}，free_est = round((1 − u_{j−2}/100)(nb − 1))。
  - 依據：async scheduling 下 batch queue = 2（config/vllm.py:589-597）。step j+1 的 schedule() 發生在 j−1 的輸出處理完之後、j 執行期間（core.py:640-719）。
  - 敏感度分析改用 t_{j−1} 與 u_{j−1}。
  - W_j = {i : e_i ≤ τ_j < s_i}，head = W_j 中 e_i 最小者。
  - N_j = ctx + gen，B = 2048（由 smoke 確認 2048-token 的 step 存在）。
- 類別：
  - no-queue：W_j 為空。
  - budget-long / budget-other：N_j = B，依 τ_j 時是否有長 request 在 prefill 區分。
  - HOL-long / KV-short：N_j < B，且 free_est < head 的需求（gate on 時，長 65、2K 短 7、8K 短 19；gate off 時為第一個 chunk 的需求）；依 head 是長或短區分。
  - unexplained：N_j < B，但 free_est ≥ head 的需求。
  - max-seqs：running ≥ 128，只作標記。
- 每個尾端短 request 的 [e_i, s_i] 依與各 I_j 的重疊時間分配到各類別；沒有任何 step 覆蓋的時間記為 unclassified。
- A10 分解有效性。每個 run 都要通過，否則該 run 的分解記為無效；這不影響 A1 的 Y。各項都由構造保證應該完全相等：
- (a) 主結果 span 數 = N，每個 id 的 prompt / completion tokens 與 client 完全一致。
- (b) bench 前後 queue-time histogram 的 Δcount = 期間收到的 span 總數（含 warmup 與 ready-check），且 |Δsum − ΣQ| ≤ 1 ms。兩者來自相同的 float。
- (c) 沒有 preemption 時，期間的 Σ ctx_tokens 等於 Δprompt_tokens；有 preemption 時豁免並報告。
- (d) 期間的 iteration index 連續，沒有缺號，代表 log 沒有遺失。
- A11 H-C5：
- 在 B0 中，ΔQ_c = 尾端各類別的 Q（mixed − uniform）。
- c* = argmax_c ΔQ_c，c ∈ {budget-long, HOL-long, KV-short}；主導的條件是 ΔQ_c*/ΔQ ≥ 0.5。
- 依 hypothesis 的對應判定。
- 敏感度分析（τ = t_{j−1}）下 c* 改變，記為 NT。
- A12 判定逐 L 進行，跨 L 不合併。兩個 L 落在 interpretation matrix 的同一列，才寫成 H-B4 的一般結論；否則報告為「成因依 L 而異」。所有 hypothesis、所有格，以及所有無效、NT、逾時、跳過的 run，都列在結果表中。
- A13 只作描述、永不改變判定的量：
- g_c = 1 − (M_c − U_c)/(M_0 − U_0)（絕對超額的移除比例）；
- 交互作用 I；
- 長類 TTFT、throughput、preemption、KV 峰值；
- histogram 的 queue / prefill 平均；
- 1 Hz 份額；
- 殘餘 BM 的 step 份額；
- unexplained-break 份額；
- allocator 的 block 粒度浪費上界（每個 request ≤ 3.1%）。
若 g 與 f 的類別不同，兩者並列，以 f 為準。
- A14 凍結與拒絕條件：
- 分析程式（adv_analysis.py、adv_report.py、adv_decomp.py，只用 stdlib，Python 3.10 相容）與測試，連同 decision 024、PREREGISTRATION §7、pins [advanced]（freeze.advanced = true），在任何 Advanced 正式數據之前 commit 並 push（011 式）。
- report 在下列情況拒絕執行：
  - session 不是 formal，或 env 無效；
  - freeze.advanced 不是 true；
  - plan 的 derived_from ≠ formal-20261009-coreb-uniform；
  - C 不等於重算的 knee；
  - argv 或 env 超出 condition 表；
  - mixed dataset 的 sha256 不符；
  - plan 或 pins 的 sha 不符。
- 凍結後的任何修改都要寫 post-hoc decision log。Core 的判定不受影響（023）。

### 判讀矩陣（逐 L，first match）

逐 L 判讀，first match。H-C0 在該 L 失敗時，直接套 R0。

**輸入**
- 移除類別：cat(T)、cat(G)、cat(TG)，取值為 M / P / R / N / W / NT。M 與 P 都要求 ρ 與 M 雙雙下降並超出 noise。
- KV 證據 e_KV（0–3）：
  - (i) K3 劑量敏感，且 uniform 特異性成立；
  - (ii) B0 的准入受阻份額 ≥ 0.5；
  - (iii) G 是 W，且有 preemption（run 皆有效）。
- qualified 的 condition 視為「非 M」。

**規則**
- **R0**：H-C0 失敗。結論：「該 L 在 session 內沒有重現 H-B4，不做歸因」。
- **R1**：cat(T) = M 且 cat(G) ≠ M。結論：主因是 batching，即長 request 的 chunked prefill 壟斷每個 step 的 budget（H-C1 成立；007 的 conjecture 得到支持）。附註：
  - cat(G) = P：加註「准入閘門也有貢獻」。
  - e_KV ≥ 2：加註「KV 也受限（次要）」。
- **R2**：cat(G) = M 且 cat(T) ≠ M，而且 G 通過 uniform 特異性。結論：主因是 scheduler 的准入閘門，即 full-ISL 檢查加上 FCFS break，由 KV 占用觸發（H-C2 成立）。附註：e_KV = 0 時標記「閘門有效，但缺少 KV 證據，不一致」，不重新解讀。
- **R3**：cat(T) = M 且 cat(G) = M。結論：兩個機制都是必要條件（conjunctive），移除任一個都消掉過半。
- **R4**：cat(TG) = M，而 cat(T) 與 cat(G) 都不是 M。結論：串聯瓶頸，batching 與准入閘門一起作用（H-C3 成立）。
- **R5**：T、G、TG 都不是 M，而且 (i) 成立，加上 (ii) 或 (iii) 至少一項。結論：「最可能是 KV 容量經 FCFS 准入造成（推論級：由排除、容量敏感、觀測三者推得；KV 無法往上調，無法用移除檢驗確認）」。附上分解的子標籤：HOL-long 主導，或 KV-short 主導。
- **R6**：T、G、TG 都不是 M，e_KV ≥ 1，但不符合 R5。結論：「KV 可能有關，但未確立」。列出 P 與 R 類的貢獻者。
- **R7**：T、G、TG 都不是 M，且 e_KV = 0。結論：「在 KV、batching、scheduler 之間沒有隔離出成因」。殘餘部分以 conjecture 歸給固定 C 下的封閉迴路運算負載（Little's law），並列出 P 與 R 類的貢獻者。

**所有列都要附上**
- ρ_TG/ρ_0 的殘餘比例；
- 每個 condition 的 f、g、a；
- R 類（只有相對下降）與 W 類的說明；
- 每個 M / P 的 Δthroughput 與 Δ長類 P95（分辨重新分配與移除）；
- preemption 與 KV 峰值。

**修飾**
- 若某個 NT 若改成 M 就會換到另一列，結論標為「incomplete」。
- H-C5 的結果附在後面：
  - 一致：直接附上。
  - 不一致：加註「機制歸因不一致，以操弄的結論為準」。
  - NT：照實記錄。
- allocator 一律寫成「未操弄，未檢驗」，並附上 unexplained-break 份額與浪費上界，只作描述。

**跨 L**
- 兩個 L 落在同一列：寫成 H-B4 的成因結論。
- 不同列：報告為「成因依 L 而異」，不合併。

**事前預期**（只由 block 算術推得，不是判定依據）
- 2K：系統中有 ≥ 2 個長 request 時，KV 受限，head 的長 request 要等 65 個 block。預期落在 R5 或 R2 / R4，HOL-long 主導。
- 8K：需要 ≥ 3 個長 request 才會受限（2 長 + 6 短 = 246 ≤ 247）。預期落在 R1 或 R4，budget-long 主導。

無論落在哪一列，Core 的 H-B4 判定與 decision table 都不改變。

### Confounds 與控制

- 觀測本身（OTLP 與逐 step log）
- 開銷：兩者都只在 frontend 執行（BatchSpanProcessor 由背景執行緒每 5 秒匯出；每個 step 一行 log），而且 5 個 condition 都相同，所以比較都在同一個觀測條件下進行。
- 參照：B0 與 Core 的 argv 不是逐位元相同。因此一律以 in-session 的 B0 為參照，並由 H-C0 確認現象仍在。
- 不新增 VLLM_* 變數：ms 時間戳改由 runner 讀 log 時打上，所以 server env 只多 OTEL_EXPORTER_OTLP_TRACES_PROTOCOL。
- 已查證：觀測不改變 scheduler 的程式路徑。log_stats 原本就開；iteration details 只是在每個 step 多呼叫一次 make_stats。
- T 會切短 request：2K 的短 request 切成 4 段，8K 的切成 16 段，uniform 也會跟著移動。
控制：
- 判定用 ρ（同一 condition 內的 mixed ÷ uniform）。
- M guard 要求 mixed 短類本身也下降。
- 只有相對下降時記為 R 類，不計為支持。
- 報告 ln(U_T/U_0)。
- T 會延長長 request 的 prefill（64 個 chunk），部分 KV 被占用的時間也跟著變長，與 KV 通道交織。
控制：
- 報告長類 TTFT 與 KV 時間序列。
- 由 2×2 讀出交互作用。
- 殘餘 BM 由「≥ 4 個長 request 同時 prefill」的 step 份額量化。在 K = 4 GiB 下，這種狀態必然伴隨 preemption，preemption 有計數。
- G 會造成 over-commit：preempt 後 recompute 並 prepend 回 waiting，可能 thrash。
控制：
- preemption 計數。
- 逾時記為 NT。
- 有效的 W 才記為「閘門具保護作用」。
- smoke FS3 先檢查這條路徑能完成、不會 hang。
- K3 只能往下調：它檢驗的是容量敏感度，不是移除檢驗。
控制：
- 規則保證 uniform 在 C_knee 仍全部放得下，並以 A6 的 uniform 檢查驗證。
- block_size、kernel、numerics 都不變。
- 任何 KV 主因的結論都標為推論級。
- 封閉迴路的運算負載（Little's law）：在固定 C 下，mixed 每個 request 平均的 prefill token 是 uniform 的 2.6 倍（2K）與 1.3 倍（8K），任何 work-conserving 的政策都會讓平均延遲上升。
控制：
- 每個 M / P 判定都並列 Δthroughput 與 Δ長類 P95，用來分辨重新分配與移除。
- interpretation matrix 的 fall-through 是「compute load（conjecture）」，不硬套到三個候選成因上。
- 順序與漂移：
- 每次 boot 只跑一個 bench，避免 boot 內的 carryover 與 uniform → mixed 的固定順序。
- 3 個 block 內以 seed 隨機排列，B0 穿插其中。
- 每次 boot 前檢查 GPU 閒置，前後各做一次 env_capture。
- 所有 condition 共用同一份 deploy commit，session 中途不重新部署。
- Async scheduling 的時序：step 的排程決策比它的 log 早兩格。
控制：
- 事先以 τ = t_{j−2} 與 u_{j−2} 為主設定，依據已在 source 中查證。
- 以 t_{j−1} 做敏感度分析；c* 改變時，H-C5 記為 NT。
- 所有分類都只作描述。
- 時鐘：
- span 的 arrival 與 runner 打的 log 時間戳都是同一台 WSL 的 time.time()。
- Q、P 是 engine 的 monotonic 時間差。
- client 的 start_times 是 perf_counter，只用 TTFT 的差值，不做絕對對齊。
- 對齊誤差只有 IPC 與讀 log 的毫秒級延遲，由 A10 的完全相等檢查把關。
- Prometheus histogram 包含 warmup 與 ready-check，1 Hz gauge 可能 alias。兩者都只用於描述，或用於 A10 的完全相等檢查。
- Spill：所有操弄都不增加 device 記憶體需求。
- MNBT 不變，所以 activation 不變。
- K ≤ 4 GiB。
- T 與 G 只改 scheduler。
memory.used 只描述性記錄。不用 V_total − 256 MiB 當門檻，因為 Stage B 實測 spill 時 memory.used 峰值只有 15,768 MiB，那個門檻永遠不會觸發。
- Numerics：沒有 fp8，也沒有 dtype 或 kernel 變更。T 只移動 chunk 邊界。長度由 --ignore-eos 固定，並逐 run 驗證。
- Workload 相同：
- 同一份 pin 過 sha256 的 mixed 檔，加 --disable-shuffle。
- 同一個 uniform seed、相同的 N 與 warmup。
- client 只差 request-id 的字串。
- C 對所有 condition 都固定為 Core 的 C_knee，不重算 knee。
- Allocator：0.30.0 沒有可替換的旋鈕，hybrid alignment 也把 block 固定為 528，所以只作描述：每個 request 的 block 粒度浪費 ≤ 3.1%，另報 unexplained-break 的份額。結論寫成「未操弄、未檢驗」。
- 揭露：
- T、K3 與事前預期，都只由 Stage B 的 block 算術（248 / 528、C_KV）與 source 推得。
- 用到的 Core 數字只有三類：C_knee、bench 時長（估時間與逾時）、restart SE 的量級。
- 門檻沒有依任何 Core 延遲結果調整。
- 多重比較：2 個 L × 6 個 hypothesis，全部事前指定、全部報告，不做事後挑選。跨 L 的一般結論需要兩個 L 一致。

### 可行性 smoke（只檢查能否執行，不看結果量）

- 共同規則（023）：
- 只檢查能否執行，在 GPU flock 下進行。
- 不在任何正式格（C_knee 加完整 N）跑 smoke。
- 輸出只有 pass / fail 與計數。bench JSON 讀完計數後立即刪除，衍生檔不含任何延遲欄位。
- smoke 同時為各 config 預熱 compile cache，所以正式 boot 都是熱 boot。
- smoke 的證據 commit 為非正式資料。
- FS0（CPU、Mac 或 WSL，不用 GPU）：
- adv_plan 由 formal-20261009-coreb-uniform 建出 60 次 boot，C = 25 / 8。順序可重現，每個 block 內每個 (condition, L, mix) 恰好一次。
- 每個 condition 的 argv 與 env 差異恰如 condition 表。
- 部署 commit 的 provenance：DEPLOYED_COMMIT 等於乾淨的 HEAD。Mac 端改用 CommandLineTools 的 git，因為 Xcode 的 shim 壞了。
- OTLP 解碼的往返測試在 WSL venv 中執行。
- FS1 啟動（B0、T、G、TG、K3 各 boot 一次）。每次都要確認：
- /health 回 200；
- log 出現 FLASH_ATTN，non-default args 恰好包含 MNBT 2048、兩個觀測 key、該 condition 的 key，其他 condition 的 key 不出現；
- /metrics 讀得到 num_gpu_blocks 與 block_size；
- receiver 收到 span；
- iteration 行能解析，而且帶 ms 時間戳。
K3 另依 H-C4 規則驗證實測 block 數：C_KV(2K) ≥ 25、C_KV(8K) ≥ 8、≥ 128、≥ 67；並以比例上界確認 2.75 GiB 不符合。不符合時以 256 MiB 為步長往上調，最後凍結 K3 與它的 block 數。
- FS2 旋鈕語意，取代 risk 誤用 iteration_tokens_total 的做法：
- 用 pinned mixed-2048 的第 5 列（index 4，32640 tokens）做成單列檔，在 B0 與 T 下各以 C = 1 跑一次。
- 由 iteration log 檢查：B0 有 ctx_requests = 1 且 ctx_tokens = 2048 的 step，同時確認 B = 2048；T 的每個 step 都是 ctx_tokens ≤ 512。
- 只印計數。
- FS3 閘門語意與 over-commit 穩定性：
- 用 4 個長列（index 4、10、…）做 C = 4 的 bench，需求 4 × 66 = 264 > 247。
- G 下：preemption 計數 > 0，4 筆全部完成且長度正確，沒有 crash 或 hang，上限 5 分鐘。
- B0 下（對照）：preemption = 0。
- 只印計數。
- FS4 harness 端到端（非正式）：在 B0 上跑 uniform-2K（C 2，N 20）與 mixed-2K（C 2，前 20 列）。
- rows、derived/、scrape delta、1 Hz 序列、操弄檢查欄位都必須齊全。
- A10 的 (a)–(d) 都通過。
- 分類器能跑完。
- CustomDataset 的 pandas 路徑（022）能載入。
- 數值遮蔽。
TG 也跑一次 mixed（C 2，N 20），只確認能完成。
- FS5 逾時路徑：在 B0 上以 5 秒逾時跑 uniform-2K（C 2，N 20）。
要求：
- client 的 process group 被殺掉；
- 該 row 為 ok = false，reason 為 timeout；
- 下一個 bench 前 server 仍然 healthy；
- teardown 乾淨，receiver 也關閉。
- 不做：fp8 KV 搭配 FLASH_ATTN。source 已足以判定 sm_120 走 FA2，不支援（fa_utils.py:96-108、282-307）。
合計約 30–40 分鐘 GPU。

### Harness 修改（Core 模組一行都不改）

- 原則：Core 模組一行都不改（core_b_plan、run_core_b、core_b_analysis、core_b_report、coreb_smoke），只 import 重用。這修正了 risk 案修改凍結 Core 程式的問題。所有新程式都先寫測試、確認失敗，再實作（TDD）。
- configs/pins.toml：
- 新增 [advanced]：seed 20261008、derived_from、C {2048: 25, 8192: 8}、conditions 表（每個 condition 的 argv 差異）、long_prefill_token_threshold 512、kv_low_bytes 3221225472、各 K 的 num_gpu_blocks（248 與 smoke 實測值）、otlp_endpoint、otel_protocol、各 (L, mix) 的逾時秒數。
- [freeze] 新增 advanced（凍結前為 false）。
- Core 的 pins 都不動。
- benchmarks/adv_plan.py：
- CONDITIONS 表。
- server_argv_for(cond)：呼叫 core_b_plan.server_argv（K3 經它原有的 kv_cache_memory_bytes 參數傳入），再加上該 condition 的 flags 與觀測 flags。
- server_env_for：先移除繼承的 OTEL_*，再加上 protocol。
- bench_argv_for：core_b_plan.bench_argv 加上 --request-id-prefix。
- build_plan(uniform session)：拒絕非 formal 或 env 無效的 session；以 core_b_analysis.knee 重算 C 並斷言為 25 / 8；產生 3 個 seeded block，每個 block 20 次 boot；記錄 expected_blocks、timeouts、derived_from。
- benchmarks/run_adv.py（runner）：
- 對每次 boot 依序：
  - 確認 lock 與 GPU 閒置；
  - 啟動 OTLP receiver 執行緒（http.server，以 venv 的 opentelemetry.proto 解碼 ExportTraceServiceRequest，支援 gzip，原始 body 存 raw/，llm_request 寫成 JSONL，回 200）；
  - 以 pipe 啟動 server，reader 執行緒為每行加上 time.time() 時間戳後寫檔；
  - 檢查 log facts、non-default args、blocks；
  - 抓 Prometheus 全文；
  - 啟動 1 Hz 時間序列 sampler（新 class，不改 MetricsSampler）；
  - 執行有逾時的 bench（start_new_session，逾時時 killpg）；
  - 再抓一次 Prometheus 全文；
  - 等 span flush，最多 30 s，直到主結果 span 數 = N；
  - 用 adv_decomp 產生 derived/ 與 rows 的摘要；
  - teardown（重用 coreb_smoke._shutdown），最後關 receiver。
- 任何例外都先 teardown。
- 跳過規則：同一格出現無效 run 後，該格剩下的 restart 不跑，並記錄。
- formal 的條件：freeze.advanced、--gpu-lock-held、Stage B pins（重用 run_core_b.stage_b_problems）、mixed sha256、vLLM 版本。
- session 目錄不覆寫、不續跑。
- benchmarks/adv_decomp.py（stdlib）：
- iteration 行的 parser，可解析帶時間戳前綴的行。
- span JSONL 與 client --save-detailed 依 id 對齊。
- 逐 request 的 O / Q / P。
- A9 的 step 分類器（主設定 τ = t_{j−2}，敏感度 t_{j−1}）。
- A10 的有效性檢查 (a)–(d)。
- 尾端集合與 run 摘要。
- Prometheus histogram delta，含 counter reset 偵測。
- 1 Hz 份額。
- benchmarks/adv_analysis.py（stdlib）：U、M、ρ、SE、noise 規則、A5 的 M / P / R / W / N / NT 類別、A6 的特異性檢查、e_KV、H-C0（直接重用 core_b_analysis.h_b4_values）、H-C1 至 H-C5、interpretation matrix（first match）、跨 L 規則、描述量。
- benchmarks/adv_report.py：CLI，執行 A14 的所有拒絕條件，輸出 analysis.json 與逐 (condition, L, mix) 的描述表。
- benchmarks/adv_smoke.py：FS1–FS5 的遮蔽版 smoke，只輸出計數，刪除延遲檔。
- 文件：
- decision 024：完整的事前登記，附 source 的檔案與行號依據、排除理由、預算、事前預期。
- PREREGISTRATION 新增 §7（Advanced，FROZEN）。
- 更新 NEXT。
以上全部在任何 Advanced 正式數據之前 commit 並 push。

### 測試（TDD，先寫測試）

- 回歸：既有的 Core 測試全部原樣通過，Core 模組沒有任何 diff（測試比對 git 的 blob hash）。
- test_adv_plan，argv：
- B0 的 argv 等於 formal-20261010-coreb-mixed2 的 session.json 中記錄的 argv，加上兩個觀測 flag。
- T 只多 ['--long-prefill-token-threshold', '512']。
- G 只多 '--no-scheduler-reserve-full-isl'。
- TG 兩個都有，各出現一次。
- K3 只把 K 值換成 3221225472，flag 不重複。
- env 只多 OTEL_EXPORTER_OTLP_TRACES_PROTOCOL，且繼承的 OTEL_* 被移除。
- bench argv 與 Core 只差 --request-id-prefix。
- test_adv_plan，plan：
- 共 60 次 boot，3 個 block，每個 block 是 20 格的排列。
- seed 可重現。
- C 由 uniform fixture 重算為 25 / 8；session 不是 formal、env 無效、或 C 不符時拒絕。
- N 與 warmup 與 Core 相同。
- 逾時值與 expected_blocks 寫入 plan。
- K3 規則的算術：3 GiB 為 186 個 block 時通過，170 個 block 時不通過；比例上界的推導。
- test_run_adv（使用 fake 的 popen、http、bench）：
- 沒有 lock 或 freeze.advanced 時拒絕 formal。
- log facts、non-default args、blocks 任一不符時，停止該次 boot，並寫出 ok = false 與原因。
- condition 的 key 多出或缺少時被偵測到。
- 逾時時 killpg，寫出 ok = false（reason timeout），session 繼續。
- 跳過規則：同一格有無效 run 後，剩下的 restart 不跑並記錄。
- span 不足時只把分解標為無效，不影響 Y。
- 任何例外都會 teardown 並關閉 receiver。
- 不覆寫既有的 session。
- reader 執行緒加時間戳後，coreb_smoke.parse_server_log 與 _RE_NONDEFAULT_BATCHED 仍能抽出 facts。
- test_otlp_receiver：
- 只保留 llm_request。
- attribute 正確對應到欄位。
- gzip body 能解。
- 回應 200。
- 寫檔失敗會拋出錯誤，不被吞掉。
- protobuf 往返測試只在有 opentelemetry.proto 時執行，否則 skip。
- test_adv_decomp：
- iteration 行的 parser：可解析帶時間戳的行，忽略 dummy 行與其他行。
- 依 id 對齊（含 cmpl-<prefix><i>-0 格式），warmup 被排除但保留。
- O + Q + P 等於 client TTFT。
- 在合成時間線上，各類別都能正確分類：no-queue、budget-long、budget-other、HOL-long、KV-short、unexplained、max-seqs。
- 歸因總和等於 Q 減去 unclassified。
- τ = t_{j−2} 與 t_{j−1} 的切換。
- 有效性檢查 (a)–(d) 各自都能觸發失敗。
- histogram delta 正確，counter reset 時報錯。
- free = round((1 − u)(nb − 1))。
- 尾端 P90 含 ties。
- 殘餘 BM 份額。
- test_adv_analysis：
- ρ 與 delta method 的 SE 與手算一致。
- noise 規則與 §2 一致。
- 類別邊界：f 恰為 0.2 歸 P，恰為 0.5 歸 M。
- R 類：ρ 下降但 M 沒下降。
- W 類。
- 只因 uniform 膨脹造成的大 f 不會成為 M。
- uniform 特異性的 qualified。
- e_KV：三項各自成立或不成立；逾時與無效 run 不計入。
- 缺 restart 時為 NT。
- H-C0 失敗時整個 L 不可解讀，並確認呼叫的是 h_b4_values。
- H-C5 的對應、無主導類別、τ 改變時為 NT。
- interpretation matrix 的 R0–R7 各有一個合成情境，產生指定的結論字串。
- NT 造成 incomplete 標記。
- 逐 L 判定，以及「成因依 L 而異」。
- test_adv_report：
- 下列情況拒絕執行：非 formal、env 無效、freeze.advanced 為 false、derived_from 不符、C 不符、argv 或 env 超出 condition 表、dataset 或 plan 的 sha 不符。
- 正常 fixture 端到端產出 analysis.json。
- test_pins（擴充）：
- freeze.advanced 為 true 時，必須已有 T、K3 的 bytes、各 K 的實測 block 數、逾時、conditions、seed、endpoint，沒有任何佔位值。
- 只在 Advanced 凍結時才強制 freeze.nvidia 與 freeze.mac 為 true。

## 為什麼

- 2×2（T × G）在 K = 4 GiB 下可以直接讀出 batching 與准入閘門各自的效果與交互作用，不需模型假設；K3 補上 KV 容量敏感度。
- 以同 condition 內的 ρ = ln(M/U) 加上 M guard 判定，避免 uniform 位移造成的假移除。
- 每次 boot 一個 bench、以 seed 隨機排列，與 Core mixed 的 protocol 一致，避免 carryover 與順序混淆。
- 觀測層（OTLP span、逐 step log）讓 KV 這個只能往下調的因子，也有獨立於操弄的機制證據；它只用於 H-C4(ii) 與 H-C5，並有完全相等的有效性檢查。

### 放棄的替代方案

- decompose 的 BUD（MNBT 8192 加 threshold 2048）：
- 同時改兩個 flag。
- MNBT 在 SchedulerConfig.compute_hash 中，會改變 compile、CUDA graph 形狀、activation 與 step 大小。
- 設了 kv_cache_memory_bytes 時，gpu_worker.py:542-561 會跳過 profiling。多出來的 activation 會 spill，而 V_total − 256 MiB 的門檻永遠不會觸發，所以偵測不到。
- T = 1024（risk 與 factorial）：threshold 是對每個 request 各自截斷（scheduler.py:661-665），兩個長 request 同時 prefill 就會用掉 2048，BM 移除不完整，「H-C1 被推翻」就不乾淨。改用 n = C_KV(32K) 推出的 T = 512。
- risk 的 FS2 用 iteration_tokens_total 驗證 chunk 大小：這個 histogram 在 first token 時一次加入整段 prompt（stats.py 的 update_from_output；loggers.py:1122-1125），在 T 與 B0 下都會誤判。改用 iteration-details log。
- 把 waiting_by_reason{capacity} 當成 KV 阻塞（factorial、risk）：它的值等於 len(waiting)（loggers.py:1034-1037），與 KV 無關。
- 每次 boot 跑 4 個 bench（factorial），或固定 uniform → mixed 的順序（risk）：會有 boot 內的 carryover 與順序混淆，也與 Core mixed 的 protocol 不同。改為每次 boot 一個 bench，所有 boot 以 seed 隨機排列。
- 第二個 KV 劑量 K3.5（risk）：為了把最壞情況壓在約 8 h 內而捨棄。保留由規則推得的 K_min（3 GiB），它是不影響 uniform 下最強的劑量。單調性檢查也因此放棄。
- 另跑一組無觀測、與 Core 逐位元相同的 B0：要多約 1.3 h，最壞情況會超過 8 h。H-C0 已足以確認現象（觀測必須消掉 ≥ 95% 的效果才會讓閘門失敗）。
- VLLM_LOGGING_CONFIG_PATH（decompose）：會新增一個 §4 沒有列出的 VLLM_* 變數。改由 runner 讀 log 時打 ms 時間戳，server env 不變。
- 修改凍結的 Core 模組（risk：server_argv、MetricsSampler、default_run_bench）：改為全部新增模組 adv_*，只 import 重用。
- 把 G / TG 的逾時算成 KV 證據（risk 的 e_KV）：無效的 run 不得作為任何判定的依據（013），一律記為 NT。
- R5 依「消去法加單邊劑量」直接寫成「KV 是主因」（risk）：改為推論級的措辭，而且要求 (i) 之外再有 (ii) 或 (iii)。
- 只看超額 E 判定移除（decompose）：缺少 M guard，uniform 移動時會出現假移除。改用 ρ 加 M guard，並新增 R 類。
- decompose 的 H-C1（Q 占比 ≥ 0.5）作為 hypothesis：三種機制與純運算負載下都會成立，沒有區辨力，改為只作描述。
- factorial 的 steady-state P95（排除前 C 個）：這個指標是看到 Core 結果後才提出的，不採用。
- fp8 KV：sm_120 走 FA2，flash_attn_supports_kv_cache_dtype 回傳 False；要用就得換 backend，同時改變 kernel 與 numerics。
- 調高 K 或 U：熱 boot 的可用量上限是 4.14 GiB，而且 WSL2 超量時會 spill，不會 OOM。
- scheduling_policy = priority：bench client 不送 priority，要改就得改 workload。
- T × K 或 G × K 的完整因子設計：超出 GPU 預算。KV 與 batching 的交互只作描述（長類 TTFT 與 KV 時間序列）。
- smoke 在正式 C 下以「多久內完成」當作 pass 條件（decompose F2）：等於從 smoke 讀出時長這個結果量。改為在非正式格只看計數。

## 對實驗的影響

- 新增 PREREGISTRATION §7（Advanced），凍結後與本文一起約束 Advanced 的所有分析。
- Core 的判定、decision table（021）與 README 的 Core 結論都不因 Advanced 而改變；Advanced 的結果另外報告。
- GPU 預算：預期約 7.3 h（含 smoke），上限約 8.1 h。只用 RTX；Mac 不參與。

## 實作補記（2026-10-10，凍結前；沒有任何 Advanced 數據）

harness 由 workflow 依本文實作（`adv_plan`、`adv_decomp`、`adv_analysis`、`run_adv`、`adv_report`、`adv_smoke`，嚴格 TDD），
再經三個面向的審查與對抗式驗證（確認 13 項，其中 1 項是既有、與 Advanced 無關的 `test_disk_guard` 環境相依），全部修正後再驗證（60 個 mutant 抓到 56 個，所有「撤銷修正」的 mutant 都被抓到）。
以下是本文原文沒有寫死、在實作時決定的細節，凍結時一併凍結：

- **TTFT 拆解恆等式**：本文同時寫「O = client TTFT − server TTFT」與「O + Q + P 等於 client TTFT」，兩者只在 arrival → queued 為 0 時一致。
  實作保留 O 的定義，另加 **F = server TTFT − Q − P**（arrival 到 queued：frontend 與 IPC），使 O + F + Q + P = client TTFT 精確成立。F 只作描述。
- **plan JSON** 另含 `gpu_memory_utilization`、`kv_cache_memory_bytes`（Stage B 的 K）、`block_size`，讓凍結的 `run_core_b.stage_b_problems` 可以核對。
- **pins `[advanced.conditions]`**：每個 condition 的 argv 差異（B0 []、T threshold 512、G `--no-scheduler-reserve-full-isl`、TG 兩者、K3 [] 而改 K），
  `adv_plan.pins_problems` 要求它與程式產生的 flags 完全相同；endpoint、protocol、各格逾時也要求等於本文的值。
- **Core 未改動的回歸測試**（`tests/test_core_untouched.py`）：比對 Core 6 個檔案在 `68f6a65` 的位元組 sha256（本文寫 git blob hash；效果相同，測試不呼叫 git）。
- **`test_pins`**：`freeze.advanced = true` 時，所有 `[advanced]` 值不得是 placeholder（`num_gpu_blocks_k3` > 0 且符合 K3 規則），且 `freeze.nvidia`、`freeze.mac` 都必須為 true。
- **分析的邊界情況**：
  - H-C0 不可檢定（B0 某格不足 3 次有效 restart）時，該 L 仍落在 R0（first match），但結論改為「H-C0 不可檢定，無法判定是否重現 H-B4，不做歸因」，並標 incomplete；與 H-C0 真的失敗（未重現）區分。
  - R4 只有在 H-C3 為 supported 時才寫「H-C3 成立」；cat(T) 或 cat(G) 為 NT 時改用「無法確認是串聯瓶頸」的變體，並標 incomplete。
  - 跨 L：任一 L 的 H-C0 不可檢定，就不寫一般結論。
  - A13 / A8 的描述量（histogram 的 queue / prefill 平均、1 Hz 份額、client 的短類 P50 / P99、長類 P50、短類 P95 TPOT、≥ P95 尾端的 O / Q / P 與其 mixed − uniform 差）
    逐 (condition, L, mix) 對有效 restart 取平均與 SE；任一有效 run 缺值則整項為 None，不算部分平均；只有 1 個 run 時不給 SE。判定一律不讀這些量。
- **runner 的安全措施**：
  - `--gpu-lock-held` 不再只信任旗標：session 開始與每次 boot 前，都由 `/proc/self/fd` 找到 lock 檔的 fd 並以 `flock(LOCK_EX|LOCK_NB)` 確認確實持有，否則拒絕。
  - teardown 期間收到的 SIGINT / SIGTERM / SIGHUP 先記錄，等 server 停止（含 SIGKILL 升級）、log 收完、receiver 關閉後才處理。
  - log reader 寫檔失敗、receiver 存檔失敗：中止 session，不寫成 timeout row。
  - boot 期間只作描述的 `nvidia-smi` 量測有 15 秒逾時，失敗只記錄，不影響有效性。
  - OTLP receiver 的連線與關閉都有逾時上限。
- **smoke 的遮蔽**：session.json 中的 bench 視窗時間（`t_before`、`t_after`、`startup_s`）也一併移除；FS4 另檢查 scrape delta、1 Hz 序列與 one_hz 份額都存在。
- **範圍外、照實記錄**：Core 的 runner（`run_core_b` 等）仍只信任 `--gpu-lock-held` 旗標；依「Core 模組不改」的原則不修改。
