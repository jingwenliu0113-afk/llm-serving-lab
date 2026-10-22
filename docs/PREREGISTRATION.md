# Pre-registration — Local LLM Serving Lab

> **2026-10-08：模型已依 decision 008 改為 Qwen3.5 4B + 9B（Core A）與 RedHatAI Qwen3.5-9B W4A16（Core B）。**
> 同日已依 008「對實驗的影響」重新推導：H-A1 至 H-A5 的預測、fit 規則中的 KV 與 recurrent 狀態、Core B 的 C_KV 公式、
> sentinel 與超量測試。仍待 pilot 才能凍結的 tokenizer 常數：WikiText chunk 數、MMLU 結構性排除清單。本文其餘結構不變。
> 同日依 review 做凍結前的一致性修訂（decision 009）：H-A4(b) 預測值、token_embd 扣除的適用範圍、
> fit run prompt、C_KV 的資料來源、H-A3 相鄰差距的寫法。hypothesis 的結構與門檻不變。
> 同日第二輪凍結前 review（decision 013）：Core B client flags、無效 run 與缺 C 的處理、H-A3 相鄰的定義、
> prefill 的計算順序、env 有效性（未知即無效）、分析 CLI 只處理有效的正式 session。hypothesis 的結構與門檻不變。

> 2026-10-08 至 10-09 凍結前另依 014–017 補齊：Core B 的 JIT 工具鏈與啟動 env（014）、Mac 充電閘門（016）、
> 003 的 Core B server 固定參數與 Stage B 的 U / K / JIT / A/B 規則（017）。hypothesis 的結構與門檻不變。

> 凍結後、任何正式數據之前的 harness 修正見 decision 018。018 只讓程式落實本文已凍結的規則，未修改本文的規則；
> 唯一的文字更正是 017 的 log 驗證句（0.30.0 不會印出），改以 non-default args 驗證同一件事。

> **凍結後、任何正式數據之前，使用者 2026-10-09 上午的決定見 decision 019**：
> - Mac 正式 run 改為 powermode 0（自動）加上原廠 70 W adapter，並新增電池下降停止規則〔同日 13:xx 使用者決定刪除此規則，見 019 的「使用者決定」〕
>   （正式 Mac Core A；quality 改在 RTX 執行後不適用，見 019、020）。
> - Core B 的 `--max-num-seqs` 從 264 改為 128，grid 上限從 256 改為 128。
> - Stage B 不拆分。
>
> **正式 quality 改在 RTX（wsl2-cuda）上執行，見 decision 020**（使用者同意，2026-10-09）。flag set 與所有 hypothesis 不變。
>
> hypothesis 與其門檻不變。下文相關處加了註記，原文不改寫。
>
> **SPEC 要求的 deployment decision table 的推導規則見 decision 021**（2026-10-10，在 Mac a2、正式 quality、正式 Core B 的分析之前寫定）。
> 決策表是描述性整理，不是新的檢定；本文的 hypothesis、門檻、metric、protocol 都不變。
>
> **正式 Core B mixed 第一次執行因 client 缺 pandas 全部失敗（沒有送出任何 request），見 decision 022**：
> 補裝 pandas（hash lock，server 不使用）後以新 session 重跑；007 的 mixed 定義與所有規則不變。
>
> **Gate 通過（使用者 2026-10-10 決定，decision 023）**：進入 Advanced，範圍只限隔離 H-B4 的成因。
> Advanced 的規則會在任何 Advanced 數據之前另寫一節並凍結；本文第 1–6 節不變。
>
> **Phase 2（使用者 2026-10-15 決定，decision 027）**：新增 Stage C（Fixed-SLO 容量）。它的規則會在任何 Stage C 正式數據之前
> 另寫為第 8 節並凍結（`configs/capacity.toml` 的 `freeze.capacity = true`；`pins.toml` 維持 Advanced 凍結時的位元組）；本文第 1–7 節不變。

> **狀態：第 1–5 節 FROZEN (Stage A)，2026-10-09**（harness commit `eb6eef1d43a15d70805cce83784646a038aed6d1`，
> `pins.toml` `freeze.mac = true`）。
> **第 6 節 FROZEN (Stage B)，2026-10-09**（harness commit `19446d7261ab0543ed5acd795b16947eaf18c075`，`pins.toml` `freeze.nvidia = true`；凍結 commit 只改 pins 與文件）。
> 依 decision 006 分兩階段凍結：
> - **Stage A**（本文第 1–5 節）：在任何 Mac 正式數據之前 commit，並把 `pins.toml` 設為 `freeze.mac = true`。
> - **Stage B**（第 6 節 addendum）：在任何 NVIDIA 正式數據之前 commit，並把 `freeze.nvidia = true`。
>   Stage B **只能代入 Stage A 規則所需的實測值與執行參數，不得修改任何 hypothesis 或規則**。
>
> 凍結後，hypothesis、metrics、task set、protocol 都不得依結果修改。任何變更都要先寫 decision log，並標明是 post-hoc。

決策依據：`docs/decisions/001`–`012`；機器可讀 pin：`configs/pins.toml`。

符號：

- **V_total** = 裝置記憶體總量；**B** = idle baseline；**C_ctx** = CUDA context 等 breakdown 以外的 overhead。
- **N_thr** = llama.cpp thread 數；**P** = `--poll` 值；**L** = context level；**C** = concurrency。

## 1. 研究問題

模型大小、量化、context、並發如何改變品質、延遲、吞吐量與記憶體 frontier？什麼條件下該用什麼配置？（SPEC）

## 2. Hypotheses（Stage A 逐字凍結，含 RTX 與 Core B）

### 通用規則

- **分析單位**：
  - Core A 的單位是 process-level mean：同一 cell 的每個獨立 process 內 `-r 5` 的平均。每個 cell 有 n = 3 個值，
    取平均 m，標準誤 SE = s ÷ √3。
  - Core B 的單位是 server restart：每次 restart 在每個 C 下的 output throughput 與 P95 值。n = 3，
    throughput 先對 3 次 restart 取平均，再計算 C_knee；P95 對 3 次 restart 取平均。
    只計入通過驗證的 run（completed、failed = 0、input / output 長度皆符合）；grid 中任一 C 沒有 3 次有效 restart，
    該 L 的比較單位記為 not testable，不以其他 C 代替（013）。
- **「超出 noise」**：
  - 兩個量 x、y 的差超出 noise，條件是 |x − y| > 2·√(SE_x² + SE_y²)。
  - 衍生的比值 r = m_a ÷ m_b，其 SE(r) = r·√((SE_a ÷ m_a)² + (SE_b ÷ m_b)²)（delta method）；D = 1 − r 的 SE 與 r 相同。
  - 不以執行順序把不同 cell 的 process 配對。
- **not testable 以「比較單位」為範圍**：某比較單位需要的 cell 若在該平台 OOM 或 spill，**只有該單位**記為 not testable，
  不算成立，也不算推翻。各 hypothesis 的比較單位：
  - H-A2：一個配對
  - H-A3：一組 (平台, size)
  - H-A4(a)：一組 (平台, size, L)
  - H-A4(b)：一個配置
  - Core B：一組 (L, mix)
  - H-A1 不適用本規則，因為 fit / OOM / spill 本身就是它的結果。
- **inconclusive**：grid 沒有涵蓋判定所需的範圍時，記為 inconclusive，同樣只報告、不計入。

### Core A — single-user（Qwen3.5 4B / 9B × Q8_0 / Q6_K / Q4_K_M × 2048 / 8192 / 32768 × {M4 Pro Metal, RTX 5070 Ti CUDA}）

共用量（008；兩個 size 的 attention 與 GDN 幾何相同）：

- KV(L) = 32,768 B × L：f16，只計 8 層 full attention。
- S = 每條序列的 recurrent 狀態約 50.25 MiB（24 層 GDN，f32）。
- token_embd 只對 untied 的 9B 扣除：tied 的 4B 在 decode 時會把整個 output（即 token_embd）讀一次，CUDA 上也有一份副本。
  適用範圍（009）：H-A2 的 bytes 在兩平台都扣（decode 只讀一列）；H-A1 的 N 只在 CUDA 扣，Mac 是 unified memory，不扣（004）。

**H-A1（fit boundary 由記憶體容量決定）**

- 預測依 004 的規則（008 修訂）：N = 檔案 − token_embd（僅 9B、僅 CUDA；Mac 不扣）+ KV(L) + S；
  predicted fit：N + 1.5 ≤ V_total − B；predicted OOM：N + 0.8 > V_total − B。
  - RTX 的 V_total、B 依第 4 節的程序在 Stage B 實測後代入，規則本身不得再改。
    以 V_total = 15.92 GiB 試算，最大的 N 是 9B-Q8_0 @ 32K 的 8.90 GiB，
    **所以只要 B ≤ 5.52 GiB，18 格全部預測 fit**（在 2K–32K 範圍內 16 GB 上沒有邊界；008）。
  - Mac：V_total = 37.44 GiB、B = 0，18 格全部預測 fit。
- *推翻條件*：任何 predicted-fit 的 cell OOM 或 spill；或任何 predicted-OOM 的 cell fit。boundary cell 只報告。

**H-A2（decode 受記憶體頻寬限制）**

- 定義 bytes = (檔案大小 − token_embd，僅 9B 扣除) + KV(L − 64) + 2·S。
  L − 64 是 128 個生成 token 的平均 depth；2·S 是每個 token 對 recurrent 狀態的讀與寫。
- 預測：同一平台上，L = 2048 時兩個都 fit 且都不是 spill 的配置 A、B，滿足 tok/s(A) ÷ tok/s(B) ≈ bytes(B) ÷ bytes(A)。
- 例：4B-Q8_0 與 9B-Q4_K_M 依上式，前者 decode 預測快約 1.125 倍。
- *推翻條件*：任一配對 |觀測比 ÷ 預測比 − 1| > 20%（以 process-level mean 計）。

**H-A3（context 的 decode 成本隨 KV 占比增加）**

- 定義降幅 D = 1 − tok/s(L = 32768) ÷ tok/s(L = 2048)。
- 預測：同一平台、同一 size 內，D(Q4_K_M) > D(Q6_K) > D(Q8_0)。依 H-A2 的頻寬模型，KV 在讀取 bytes 中的占比越高，D 越大；
  模型預測值（Q4_K_M / Q6_K / Q8_0）：4B 為 0.260 / 0.218 / 0.178，9B 為 0.162 / 0.131 / 0.105。
  相鄰差距約 0.026–0.042（最小為 9B 的 Q6_K vs Q8_0）；差距落在 noise 內時屬於「不推翻、也不確認」，照實報告。
- *推翻條件*：任一相鄰配對的順序相反，而且差距超出 noise。若某 (平台, size) 在 32K fit 的配置少於 2 個，記為 not testable。
  「相鄰」是在 32K fit 的配置中，依 Q8_0 → Q6_K → Q4_K_M 順序相鄰（例：Q6_K OOM 時比較 Q8_0 與 Q4_K_M；013）。

**H-A4（prefill 受算力限制）**

- (a) 對每個三個 quant 都 fit 的 (平台, size, L)：spread_prefill = 三個 quant 的 prefill tok/s（= L ÷ TTFT）最大值 ÷ 最小值；
  spread_decode 同理。預測 spread_prefill < spread_decode。
  *推翻條件*：任一符合條件的單位出現 spread_prefill ≥ spread_decode。
- (b) 對每個 32K 會 fit 的配置，預測 TTFT(32768) ÷ TTFT(8192) > 4（attention 為 O(L²)，呈超線性）。
  hybrid 架構只有 1/4 的層是 full attention，FLOP 模型的預測值為 9B 約 4.45、4B 約 4.84，離門檻 4 不遠（9B 約 11%）。
  FLOP 模型（009）：TTFT ∝ 2·P_nonembed·L + 65,536·L²，P_nonembed = 9B 6.92B / 4B 3.57B
  （`llama-bench` 只算最後一個 token 的 logits，token_embd 只查表）；65,536 = 2 × 8 層 full attention × 16 heads × 256，
  attention 為 causal（KV view 逐 ubatch 增長）。GDN 的 chunk 計算與 L 成線性，未計入，計入後比值更接近 4。
  *推翻條件*：任一配置的比值 ≤ 4。

**H-A5（quality–memory frontier）**

- (a) 較大的模型在 4-bit 下優於較小的模型在 8-bit 下：9B-Q4_K_M（約 5.62 GB，比 4B-Q8_0 多約 26%）
  在 WikiText-2 全部 chunk 上的配對 ΔNLL（9B-Q4_K_M − 4B-Q8_0）< 0。chunk 數依 Qwen3.5 tokenizer 於 pilot 確認後凍結。
  4B / 9B 之間沒有權重記憶體相近的配對（008），因此報告時要並列兩者的記憶體與 decode 速度（H-A2）。
  p 值以 chunk bootstrap（10,000 次，雙尾）計算（005）。
- (b) 同一 size 內，KLD 單調遞增：Q8_0 < Q6_K < Q4_K_M，共 4 個 z 檢定（005）。
- **判定只用 Holm**：(a) 與 (b) 共 5 個 p 值，α = 0.05 雙尾，做 Holm step-down；校正後顯著且方向符合預測才算成立。
  - *推翻條件*：任一檢定在 Holm 校正後不顯著，或方向相反。
  - ΔNLL 的 95% bootstrap CI 只作描述，不用於判定。
- Downstream 是 descriptive，不屬於本 hypothesis。事先寫下的預期：在這個 N 下，Q8_0 vs Q6_K 在四個 task 上都是 not resolved
  （配對 95% CI 包含 0）。

### Core B — multi-user（RedHatAI/Qwen3.5-9B-quantized.w4a16 / vLLM 0.30.0 / RTX 5070 Ti；以 008 的分段 smoke test 通過為前提）

- Core B 模型固定為 `RedHatAI/Qwen3.5-9B-quantized.w4a16`（008，`--language-model-only`），沒有 fallback。
  smoke test 失敗時 Core B 記為 blocked，並回報使用者。
- 定義（007）：
  - mix 分為 `uniform` 與 `mixed`。**H-B1 至 H-B3 只用 `uniform`；`mixed` 只用於 H-B4。**
  - 每個 `uniform` request 的 tokens = L，(input, output) = (L − 128, 128)。
  - C_KV(L) = ⌊(num_gpu_blocks − 1) ÷ (⌈L ÷ block_size⌉ + 3)⌋（hybrid 版，008）。
    num_gpu_blocks 與 block_size（預期約 528）取自 `/metrics` 的 `vllm:cache_config_info`（INFO log 不含 num_gpu_blocks）；
    交叉核對：num_gpu_blocks = ⌈GPU KV cache size × (⌈max_model_len ÷ block_size⌉ + 3) ÷ max_model_len⌉（009）。
    這是 Core B 唯一的 C_KV 規則；007 §3 已改為同一公式。
    每個 request 另需 3 個 GDN 狀態 block（每條序列約 49.5 MiB）。
  - 「KV cache usage %」包含 GDN 狀態 block，報告時要註明。
  - C_knee = grid 中「3 次 restart 平均」output throughput 達到 grid 最大值 95% 的最小 C。
  - C_max = grid 中最大的 C。
  - 〔凍結後註記，019：grid 上限由 007 的 256 改為 128，`--max-num-seqs` 由 264 改為 128，其餘 007 規則不變。〕

**H-B1（knee 之後延遲急升）**

- 預測：在每個 L 上，存在 C > C_knee，其 P95 TTFT ≥ 3 × P95 TTFT(C_knee)。
- *推翻條件*：grid 涵蓋到 ≥ 4 × C_knee，仍沒有任何 C 達到 3 倍。
- 若 grid 沒有涵蓋到 4 × C_knee（包括 C_knee = C_max 的情況），而且沒有 C 達到 3 倍，記為 inconclusive。

**H-B2（KV 容量決定 knee）**

- 預測：對每個 C_KV 落在 grid 範圍內的 L，C_knee ∈ [C_KV ÷ 2, 2 × C_KV]。
- *推翻條件*：任一這樣的 L 落在區間外。
- C_KV 超出 grid 的 L，只描述性報告 knee 位置；「在 KV 用盡前就先受算力限制」只是 conjecture。

**H-B3（TPOT 的成長慢於 TTFT）**

- G_TPOT = P95 TPOT(C_max) ÷ P95 TPOT(C = 1)；G_TTFT 同理。預測每個 L 上 G_TPOT < G_TTFT。
- *推翻條件*：任一 L 出現 G_TPOT ≥ G_TTFT。
- 機制（decode batching 攤提權重讀取）是 conjecture，Core 不檢驗（007）。

**H-B4（request mix 提高短 request 的尾延遲）**

- 在 L ∈ {2048, 8192}、C = 該 L 的 `uniform` C_knee 時，預測短類 request 的 P95 TTFT 在 `mixed` 中比 `uniform` 高 ≥ 20%，
  而且超出 noise。
- *推翻條件*：差距 < 20%，或差距在 noise 內。
- 機制（chunked prefill 互相干擾）是 conjecture，可能在 Advanced 隔離（007）。

## 3. Metrics 定義

### Core A

| Metric | 定義 | 產生的 process |
|---|---|---|
| fit / OOM / spill | 004「Fit 的操作定義」與 spill 定義（anchor 與 C_ctx 為 4B-Q4_K_M，bytes 同 H-A2；009） | `llama-completion`（同 cell 的 bench process 若失敗也記為 OOM） |
| engine TTFT(L) | `-p L -n 0` 的 `avg_ns`，同時保留 `samples_ns`；不含 tokenize / HTTP / sampling | `llama-bench` |
| prefill tok/s(L) | L ÷ TTFT(L)，每個 process 各自計算後再取 cell 平均（013） | 同上 |
| decode tok/s(L) | `-p 0 -n 128 -d (L-128)` 的 `avg_ts` | `llama-bench` |
| peak memory | **只取 fit-run process**：llama.cpp memory breakdown（model / context / compute）逐 device 與 host 報告，另列合計。Mac 另記 `/usr/bin/time -l` 的 peak memory footprint 與 maximum resident set size（pilot：footprint 不含 mmap 的權重、含 KV / compute；012）；RTX 另記 `nvidia-smi memory.used` 峰值減 B。`-lm mmap` 固定 | `llama-completion` |
| quality | 005 | `llama-perplexity`（只在 Mac）〔凍結後註記，020：改為只在 RTX（wsl2-cuda）量，兩個 size 都搬（4B 一起搬經使用者確認）；flag set 不變〕 |

兩平台的 peak memory 依平台分別定義，**不互相比大小**（不做排名）。pilot 要先交叉核對 breakdown 與外部量測。

### Core B

`vllm bench serve --save-detailed` 的 TTFT、TPOT、ITL、E2EL，固定
`--percentile-metrics ttft,tpot,itl,e2el,client_queue_time --metric-percentiles 50,95,99`；
request throughput（req/s）；output token throughput 與 total token throughput。

- TTFT 從 request 送出算起；client queue time 另報，不含在 TTFT 內。
- H-B4 的短類 P95 由 `--save-detailed` 的逐筆紀錄依 input 長度篩選後計算。
- 隨負載變化的記憶體訊號：Prometheus 的 KV cache usage % 與 preemption 次數（metric 名稱在 smoke test 確認）。
- GPU device memory 只作為 baseline 與配置量的紀錄（003）。

## 4. Protocol（Stage A）

### 各工具的 flag

| 工具 | flags |
|---|---|
| `llama-bench` | `-ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -lm mmap -t N_thr --poll P -r 5 --delay 5 -o jsonl -v`，context 只由 `-p/-n/-d` 決定（`llama-bench` 不接受 `-c/-np/--fit`）；**一個 config 一個 process** |
| `llama-completion`（fit / memory） | 004 的指令：`-c L -np 1 --fit off -ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -lm mmap -t N_thr --poll P -n 128 --ignore-eos -no-cnv --log-jsonl -lv 4`（`-lv 4` 只影響 log，讓 llama.cpp library 的 INFO 出現；012）；prompt 為 `" the"` 重複 L − 128 次（Qwen3.5 tokenizer 下每個單位恰為 1 token，id 279；harness 從 log 驗證 token 數；009） |
| `llama-perplexity`（quality） | 005 的 flags |

- **N_thr、P 依平台各自固定**：
  - Mac：N_thr = 8（M4 Pro 的 performance core 數，`hw.perflevel0.logicalcpu = 8`）。
  - P 的選擇規則（事先寫定）：pilot 在 sentinel cell 比較 `--poll 0` 與 `--poll 50`。只有在 `--poll 0` 的 decode tok/s 中位數低超過 3% 時才用 50，
    否則用 0（避免 CPU 空轉加熱 SoC）。
  - **Mac 結果：P = 0**（2026-10-08 pilot：poll 0 中位數 54.89 vs poll 50 53.54 tok/s；012）。
  - RTX：N_thr = WSL 內 host CPU 的 physical core 數，P 用同一規則，皆於 Stage B 決定。
- `-fa on` 之前，pilot 先以 `-fa auto` 確認兩個 backend 的 log 都出現 "Flash Attention enabled"。

### 重複次數與順序

- Core A：每個 cell 3 個獨立 process × `-r 5`；每個 session 內，cell 的執行順序以固定 seed 隨機排列，seed 記入結果。
  正式 session 的 seed 一律為 **20261008**（凍結時寫定；重跑的 session 也用同一個 seed）。
- Core B：每個 (L, mix) 3 次 server restart（007）。

### Core B server 與 Stage B 推導規則（003 的固定參數表，017 寫定細節）

- 所有 Core B server（smoke 除外，smoke 依 008）共用 `core_b_plan.server_argv`：
  - 007 / 008 已有的 flags。
  - `--max-num-batched-tokens 2048`、`--performance-mode balanced`、`--async-scheduling`、`--attention-backend FLASH_ATTN`。
  - 〔凍結後註記，019：`--max-num-seqs 128`（007 為 264）。〕
  - `--kv-cache-memory-bytes K`。
  - client 為 `--backend vllm`。
  - 以 log 驗證 attention backend 與 max_num_batched_tokens。
- **U**（`--gpu-memory-utilization`）：取 0.01 的倍數中，滿足 U × V_total ≤ V_total − B − 512 MiB 的最大值。
- **K**：在 U 下用全新的 cache 做兩次不帶 K 的 boot。第一次（冷）丟棄；第二次（熱）log 中的 `Available KV cache memory` 向下取整到 256 MiB。
  num_gpu_blocks / block_size 由帶 K 的 server 的 `/metrics` 讀取。
- **JIT 計時**：上述兩次 boot 從啟動到 `/health` 的秒數，只作描述。
- **`VLLM_WSL2_ENABLE_PIN_MEMORY` A/B**：依序以 0、1 各 boot 一次（帶 U、K），每次跑 `uniform` L = 2048，C = 1、16 各 3 次。
  - 1 在兩個 C 都快 ≥ 5%，而且四組的 CV 都 < 3%，才改為 1；否則維持 0。
  - 任一組少於 3 次有效 run 時，同樣維持 0。
- 以上都在 GPU lock 下執行（`benchmarks/coreb_stage_b.py`、`benchmarks/stage_b.py`）。

### RTX idle baseline B 的量測程序（Stage A 凍結）

1. 在 GPU lock 下進行，且沒有任何其他 GPU process。
2. 最後一個 GPU process 結束後，閒置 5 分鐘。
3. 以 1 Hz 取樣 `nvidia-smi memory.used` 共 60 秒，取中位數，即為 B。同時記錄螢幕是否接在這張卡上。
4. Stage B 量得的 B 是 H-A1 唯一使用的值。之後每個 session 開頭重量一次；若超過凍結的 B 0.25 GiB 以上，該 session 標記為 drift，重跑。

### 熱與背景負載控制

- **Mac**：
  - 接 AC 電源；所有正式 run 的 `powermode` 都必須相同（目前為 2 = high power）；`iogpu.wired_limit_mb` 必須為 0。
    〔凍結後註記，019：使用者決定所有正式 Mac run 改為 **powermode 0（自動）**（同日補充：2 = 高效能也可以，但一個 session 內必須一致），AC 必須是 Apple 原廠 `70W USB-C Power Adapter`；
    每個 process 前後記錄 power state。正式 Mac Core A 另有電池下降停止規則（quality 依 020 改在 RTX，不適用）：
    任一電量讀數比它之前 30 分鐘內（扣除 016 暫停）的最大值低 ≥ 3 個百分點，就立即停止，整個 session 無效。〕
    〔2026-10-09 13:xx：使用者決定**刪除電池下降停止規則**（019）。仍適用 012 的有效性規則、016 的充電閘門、
    019 的 adapter 條件，以及同一 session 內 powermode 一致（0 或 2）。〕
  - process 之間至少閒置 30 秒。
  - **sentinel cell**（最小 size × 最低 bit × 最短 context，由 pins 推導；目前為 4B-Q4_K_M decode @ L = 2048）在每個 session 開頭與結尾各量一次。兩次的 decode tok/s 相差 > 3% 時，
    整個 session 標記為 drift，所有 cell 在新的 session 重跑。
  - 每個 process 前後都跑一次 `env_capture`（thermal / loadavg / Mac lock owner / 電源）。
  - **電源有效性（012）**：任一 process 前後出現「電源不是 AC」、「電量 < 20%」或 thermal / performance warning，
    該 session 標為 env-invalid，整個 session 重跑（與 drift 相同）。quality session 同樣適用。
    env 紀錄無法判定電源來源或 thermal 狀態（probe 失敗）時，同樣視為 env-invalid（013）。
  - **充電閘門（016）**：Mac 上每個 process 開始前，若在 AC 上且電量 < 40%，閒置充電到 ≥ 90%（最多 4 小時）再開始。
    暫停記入 `session.json` 的 `charge_pauses`。有效性規則不變。
- **RTX**：
  - 在 GPU lock 下執行，`env_capture --gpu-lock-held`。
  - `enforced.power.limit` 必須等於 `power.default_limit`。
  - 每個 process 前後記錄溫度、SM clock、`clocks_event_reasons`。
  - sentinel 規則同上。
- **環境變數**：`GGML_*`、`CUDA_*`、`VLLM_*` 等一律不設定，除非本文件明列。`env_capture` 會記錄這些變數，名稱疑似 credential 的一律遮蔽值。
  - 明列（Core B）：`vllm serve` 與 `vllm bench serve` 都設 `HF_HOME`（gpu-server 的 HF cache）與 `HF_HUB_OFFLINE=1`；
    `vllm serve` 另設 `VLLM_GDN_DECODE_KERNEL=triton`（008、013），
    以及給 FlashInfer sampler JIT 用的 `CUDA_HOME`（user-space shim，指向 vLLM lock 中的 pip CUDA 13）與 PATH 前綴 `.venv/bin`（ninja）；
    nvcc 不放進 PATH（014）。
    〔凍結後註記，018 第 7 點：014 補充把 shim 改指向版本一致的 CUDA 13.0（`configs/flashinfer-cuda.lock.txt`），
    因為 vLLM lock 的 nvcc 13.4 與 cudart 13.0 headers 不一致；發生在任何 Core B 數據之前。〕

### Prompt 與 sampling

- Core A 效能：`llama-bench` 的合成 token（不經 chat template；token 內容由 libc `rand()` 產生，兩平台不同，對 dense 模型影響可忽略）。
- fit run：固定的重複 token 序列（`" the"` × (L − 128)；記憶體與 token 內容無關；009）。
- quality：log-likelihood scoring，沒有 sampling。
- Core B：`uniform` 用 random dataset，`mixed` 用自訂 dataset，seed 皆為 20261008（007）；
  `temperature 0`、`--ignore-eos`、`--generation-config vllm`。
  client 明確傳 `--temperature 0`（vLLM 0.30.0 的 `bench serve` 不再預設）、`--tokenizer` 指向 pinned snapshot
  （`bench serve` 沒有 `--revision`）；`mixed` 另傳 `--custom-output-len 128`（預設 256 會覆蓋每列的 output_tokens）。
  `mixed` 的短類與 `uniform` 分布相同（同產生器、seed、長度），token 內容不保證相同（013）。

### 失敗也要報告

OOM、spill、crash、drift、not testable、inconclusive 的單位一律進結果表，不刪除。
分析程式（`core_a_report`、`quality_analysis`）只對有效的正式 session 產生判定：pilot 維持遮蔽，
env-invalid 或 drift 的 session 整個重跑，不先看判定（013）。

## 5. Stage A 凍結 checklist（任何 Mac 正式數據之前）

- [x] 8 個衍生 GGUF 的 sha256 / size 都已填入 `pins.toml`（`freeze.mac = true` 由測試強制；4B 與 9B 各為 BF16 加 3 個 quant，附 recipe）
- [x] converter 的 Python lock 已 commit，`pins.toml` 的 `core_a.converter_lock` 指向它（004；`configs/converter.lock.txt`）
- [x] Quality：HellaSwag / MMLU 子集的題號清單、子集 `.bin` 的 sha256、MMLU 排除清單已 commit（005）
  - [x] 題號清單與 HellaSwag 子集 sha256（`pins.toml`）
  - [x] MMLU 結構性排除清單：用 Qwen3.5 tokenizer 計算，排除 3 題，剩 3997 題，sha256 已寫入 pins（005）
  - [x] 以 llama.cpp tokenizer 交叉驗證 MMLU 排除清單：16,000 個字串的 token 完全一致，排除清單相同；HellaSwag、ARC 走同一 scorer，皆 0 題（`quality/subsets/tokenizer-crosscheck.json`，2026-10-08）
  - [x] WikiText chunk 數：llama.cpp tokenizer 與 HF 完全一致（297,194 tokens，比照 `-f` 去掉一個結尾換行），580 chunks（同上檔案）
- [x] Mac pilot 完成：
  - [x] `-fa auto` probe：4B 與 9B 共 6 個 quant 在 Metal 皆為 "Flash Attention enabled"、graph splits = 2（`pilot-20261008-4b-fa`、`pilot-20261008-9b`；012）
  - [x] `--poll` 依規則選定：Mac P = 0（012）
  - [x] throughput 校準：正式 Core A 上限 17.39 h（9B）、正式 quality 兩個 size 約 14.2 h（012），皆不含 016 的充電暫停
  - [x] determinism：KLD 與 MMLU 重跑完全相同（4B、9B；012）
  - [x] memory metric 交叉核對：breakdown 的 context 增量 4B 990 MiB / 9B 958 MiB，footprint 約 1026 MiB，理論 960 MiB，一致（012）
  - [x] quality 數值 sanity check：self-KLD 0；Q8_0 KLD 4B 0.0014、9B 0.00047（fa on / off 相同；012）
  - [x] MDE（遮蔽方向）：Core A sentinel CV 1.0–5.9%（兩格相對 MDE 1.7–9.6%，依前景負載）；downstream CI 半寬 0–1.7 pp；H-A5 ΔNLL 的 MDE 約 0.009 nats/token（Holm 最嚴 α、80% power）；KLD SE 上界 4B 7.7e-4、9B 6.0e-4（012）
- [x] Benchmark harness 與輸出格式固定，記錄 harness 的 git commit：`eb6eef1d43a15d70805cce83784646a038aed6d1`
- [x] 判定規則的分析程式在任何數據前 commit（`benchmarks/core_a_analysis.py`、`quality/quality_analysis.py`、`benchmarks/core_b_plan.py`、`benchmarks/core_b_analysis.py`；011）；report CLI（`core_a_report.py`、`core_b_report.py`）亦同（013）
- [x] Mac 重負載 lock 的參與方式已由使用者決定，並寫入 MACHINE_POLICY 或 decision log（共享 `/tmp/ai-portfolio-mac-heavy.lock`，經 `tools/with-mac-lock`）
- [x] 本文件第 1–5 節狀態改為 FROZEN (Stage A) 並 commit（2026-10-09）

## 6. Stage B addendum（任何 NVIDIA 正式數據之前；只能代入實測值與執行參數）

**FROZEN (Stage B)，2026-10-09**（harness commit `19446d7261ab0543ed5acd795b16947eaf18c075`）。所有 GPU 步驟都在 GPU lock 下執行。

- [x] RTX 實測：V_total（`memory.total`）、B（依第 4 節程序）、C_ctx（004；4B-Q4_K_M @ 2048，009）、是否接螢幕、HAGS、`.wslconfig`
  - 2026-10-09（`results/stage_b/20261009/`，GPU lock 下）：
    - V_total = 16,303 MiB。
    - **B = 811 MiB**（02:09–02:15，5 分鐘閒置後 60 個 1 Hz 樣本的中位數，樣本介於 799–811）。
    - 螢幕不接在這張卡上（display_active = Disabled）。
    - **HAGS 開啟**（`HwSchMode` = 2）；`.wslconfig`：`[wsl2] memory=31GB`、`swap=8GB`
      （04:10，從 Windows 端以 `ssh gpu-server` 唯讀查詢，`windows_settings.json`）。
      WSL 內的 probe 呼叫 `reg.exe` / `cmd.exe` 會出現 `Exec format error`，代表 WSL interop 不可用。這是系統設定，不去修改。
    - **C_ctx = −89 MiB**（03:54，依規則 = 峰值 3,906 − B 811 − self 3,184）。
  - 觀察（照實記錄，不改規則）：這台桌機的 Windows idle VRAM 在當晚於 457–811 MiB 之間浮動，C_ctx 那次 run 開始時是 457。
    因此 C_ctx 的負值只是 B 與 C_ctx 分別量測造成的；spill 規則實際用到的是 B + C_ctx = 722 MiB，也就是 anchor 峰值減 self，不受 B 影響。
    H-A1 在 B ≤ 5.52 GiB 時 18 格都預測 fit，也不受影響。
    沒有為了得到正值而重量 B（避免挑選結果）。
- [x] sysmem 超量測試（9B-Q8_0、`-c 262144`，不屬於 18 格；N 約 15.90 GiB，任何 B 下都超量；008）：hard OOM 或 spill
  - **結果：spill，不是 hard OOM**（2026-10-09 04:56，GPU lock 下，`results/stage_b/20261009/overflow/`；raw 檔的 sha256 見 `raw_manifest.json`）。
    - process 正常結束（exit 0，13.9 s）。
    - llama.cpp breakdown 的 CUDA0：self 16,788 MiB（model 8,045 + context 8,242 + compute 501）、free 0。
      self 已超過裝置總量 16,275 MiB，也超過 V_total 16,303 − B 811。
    - `nvidia-smi memory.used` 峰值為 15,768 MiB，受實體 VRAM 限制。超出的部分由 WSL2 / WDDM 放到系統記憶體（sysmem fallback）。
  - 含意（只是套用既有規則，不改規則）：在這台 RTX 上，超出 VRAM 的 cell 不會 OOM，而是以 spill 完成。
    因此 RTX 的 fit 判定一律依 004 的兩條 spill 條件，也就是記憶體條件（self + C_ctx + B > V_total）與速度條件。
  - 照實記錄的觀察（依 004 只用來判斷行為，不作任何速度結論）：prompt eval 351.7 tok/s（1,024 tokens）、decode 39.4 tok/s（127 runs）。
- [x] llama.cpp CUDA build：toolkit 版本、`wsl2-cuda` verified、RTX 的 N_thr / P（依第 4 節規則）
  - toolkit：user-space CUDA 13.4.1 pip wheels 加上 Ninja（015，`configs/llama-cuda.lock.txt`），host compiler 為 GNU 11.4.0。
  - b11146 在 `wsl2-cuda` 上 verified（2026-10-09 02:07，`CUDA0 RTX 5070 Ti 16275 MiB`）。
  - N_thr = 6（Ryzen 5 7600，6 個 physical core）。
  - **P = 0**：依第 4 節規則，RTX pilot 中 poll 0 為 184.37 tok/s、poll 50 為 184.06 tok/s（2026-10-09，`pilot-20261009-rtx`；012 補充）。
  - pilot 34 / 34 ok，`-fa auto` 在 6 個 quant 上都是 enabled，sentinel CV 0.80%，正式 RTX Core A 上限約 3.62 小時。
- [x] vLLM：smoke test 通過（003 gate）、lock 檔已 commit（`vllm.lock_file`）、`VLLM_WSL2_ENABLE_PIN_MEMORY`（依 003 規則）、
      attention backend、performance mode / async scheduling 的實際生效值
  - smoke S1–S3 通過（2026-10-09 01:54–02:01，`results/core_b/smoke/20261009-s1s3-014b`）。
    003 gate 第 1、2、5 項通過（capability (12, 0)、arch list 含 sm_120、`vllm bench serve` 與 `/metrics` 名稱）。
  - lock 檔：`configs/vllm.lock.txt`（pins `vllm.lock_file`）。S1–S3 都在這份 lock 上執行。
    2026-10-09 核對 WSL 的 `.venv` 與 lock 完全一致（196 / 196 個套件）。JIT 透過 014 的 `CUDA_HOME` shim。
  - 017 的固定 flags（FLASH_ATTN、balanced、async scheduling、max_num_batched_tokens 2048）由 runner 以 log 驗證（018）。
  - boots、K、pin-memory A/B：先前卡在 `--max-num-seqs 264`（冷 boot 失敗保留在 `boots/`）。
    依 019 改為 128 後，在 GPU lock 下完成（09:36–10:30，`results/stage_b/20261009/boots-019`、`pin-ab-019`、`stage_b_019.log`）：
    - **boots**（U 0.91、全新 cache、不帶 K）：
      - 冷 boot 164.4 s、熱 boot 40.0 s，熱 boot 時 cache 有效。CUDA driver 的 PTX JIT cache 沒有重置（018 第 4 點）。
      - 冷 boot 的 `Available KV cache memory` 換算為 2.25 GiB（已取整），熱 boot 為 4.14 GiB，因此 **K = 4,294,967,296 B（4.0 GiB）**。
    - **pin-memory A/B**（帶 U、K；uniform L = 2048；C = 1、16 各 3 次，12 次全部有效）：
      - 1 相對 0：C = 1 快 9.13%，C = 16 只快 3.82%；四組 CV 都 < 0.3%。
      - 規則要求兩個 C 都快 ≥ 5%，所以 **維持 0**。
    - **實際生效值**：
      - non-default args 中有 `'max_num_seqs': 128`、`'max_num_batched_tokens': 2048`、`'async_scheduling': True`、`'attention_backend': 'FLASH_ATTN'`、`'gpu_memory_utilization': 0.91`。
      - log 中有 `Using AttentionBackendEnum.FLASH_ATTN backend.`。
      - `--performance-mode balanced` 等於 0.30.0 的預設值（`vllm/config/vllm.py`），所以不會出現在 non-default args 中。
    - 觀察（照實記錄）：264 那次失敗的 2.17 GiB 是**冷** boot 的值，熱 boot 的可用量高很多（128 時為 4.14 GiB）。
      但即使以熱 boot 計，約 248 個 block 仍小於 264，019 的必要性不變。
  - 003 gate 第 1–6 項全部通過。pins 中 `vllm.verified = ["wsl2-cuda"]`、`core_b.status = "verified"`。
- [x] Core B：
  - num_gpu_blocks 與 block_size（`/metrics` 的 `vllm:cache_config_info`；`GPU KV cache size` 只作紀錄與交叉核對）
  - 依 §2 的 C_KV 公式與 007 的 grid 規則，算出每個 L 的 C_KV 與 grid
  - `mixed` dataset 的 sha256（`core_b.mix_dataset_sha256`，每個 L 一個；2026-10-08 已在 WSL 建好並驗證可重現，見 013）
  - `--kv-cache-memory-bytes`（K）、`--gpu-memory-utilization`（U）：依 §4 的 017 規則由 B、V_total 與熱 boot 的 log 算出
  - JIT 冷 / 熱啟動秒數；pin-memory A/B 的結果與選定值
  - **結果**（2026-10-09，帶 K 的兩次 pin-ab boot，`/metrics` 相同）：
    - **num_gpu_blocks = 248**、**block_size = 528**。
    - 交叉核對：⌈GPU KV cache size 123,128 × (⌈32768 ÷ 528⌉ + 3) ÷ 32768⌉ = ⌈123,128 × 66 ÷ 32768⌉ = 248，一致。
    - C_KV（§2 公式 ⌊247 ÷ (⌈L ÷ 528⌉ + 3)⌋）：**L = 2048 → 35、8192 → 13、32768 → 3**。
    - grid（007 規則，上限依 019 為 128）：
      - 2048：1, 2, 4, 8, 16, 18, 25, 32, 35, 49, 64, 70, 128
      - 8192：1, 2, 4, 7, 8, 9, 13, 16, 18, 26, 32, 64, 128
      - 32768：1, 2, 3, 4, 6, 8, 16, 32
    - `mixed` dataset：sha256 已於 2026-10-08 pin（2048、8192）。依 019 維持 2560 列。
    - U = 0.91、K = 4.0 GiB、`VLLM_WSL2_ENABLE_PIN_MEMORY` = 0。JIT 冷 164.4 s、熱 40.0 s。
    - 依 019 事先記下的後果：C_KV(2048) = 35，2 × C_KV = 70。L = 2048 的 knee 若 ≥ 33，4 × C_knee > 128，H-B1 在這個 L 只能是 supported 或 inconclusive。
- [x] 本節狀態改為 FROZEN (Stage B) 並 commit（2026-10-09，harness commit `19446d7261ab0543ed5acd795b16947eaf18c075`，`freeze.nvidia = true`）

## 7. Advanced addendum：隔離 H-B4 的成因（decision 023、024）

**狀態：FROZEN (Advanced)，2026-10-11**。harness commit `6930d8f`；smoke FS1 實測 K3 = 186 個 block 後，`pins.toml` `freeze.advanced = true`，
連同分析程式與測試在同一個凍結 commit 中（凍結時的兩個小修正見 024「凍結」）。在這之前沒有任何 Advanced 正式數據。
本節只摘要；完整規則、門檻依據與原始碼依據見 decision 024。第 1–6 節不變，Core 的判定不受影響。

- **平台與固定項**：RTX 5070 Ti、vLLM 0.30.0、Core B 的模型、server argv、client flags、mixed dataset（pin 過 sha256）與 uniform seed。
  C 固定為 Core 的 C_knee（2K 為 25、8K 為 8，由 `formal-20261009-coreb-uniform` 重算）。
- **5 個 condition**（都加上相同的觀測：OTLP traces 與 `--enable-logging-iteration-details`）：
  - B0：Core argv。
  - T：`--long-prefill-token-threshold 512`（T = MNBT ÷ (C_KV(32K) + 1)）。
  - G：`--no-scheduler-reserve-full-isl`。
  - TG：T 與 G。
  - K3：`--kv-cache-memory-bytes` 3 GiB（規則：C_KV(L; K) ≥ C_knee、num_gpu_blocks ≥ 128、≥ 67 的最小 256 MiB 倍數）。
- **格子**：5 × L {2048, 8192} × mix {uniform, mixed} × 3 次 restart = 60 次 boot，每次 boot 一個 bench，3 個 block 內以 seed 20261008 隨機排列。
- **主要量**：每個 run 的短類 P95 TTFT（Core 的 `short_class_p95_ttft_ms`）。U、M = 3 次 restart 的平均；ρ = ln(M ÷ U)。
- **Hypotheses**：
  - H-C0：B0 在 session 內重現 H-B4（凍結的 `h_b4_values`）。失敗的 L 不做歸因。
  - H-C1：batching（chunked prefill 的 budget 壟斷）是主因：cat(T) = M。
  - H-C2：scheduler 的准入閘門是主因：cat(G) = M，且 |ln(U_G ÷ U_0)| ≤ 0.10。
  - H-C3：串聯瓶頸：cat(TG) = M，而 cat(T)、cat(G) 都不是 M。
  - H-C4：KV 容量：(i) K3 的劑量敏感度（a ≥ 0.20、uniform 特異性）；(ii) B0 尾端 queue time 中准入受阻的份額 ≥ 0.5。
  - H-C5：分解的主導類別與操弄的結果一致。
- **移除類別**（f = 1 − ρ_c ÷ ρ_0）：M（f ≥ 0.5，ρ 與 M 都下降並超出 noise）、P（0.2 ≤ f < 0.5）、R（只有相對下降）、W（f ≤ −0.2）、N、NT。
- **判讀矩陣**：逐 L，first match，R0–R7（見 024）。KV 只能往下調，任何「KV 主因」的結論都標為推論級。跨 L 一致才寫成一般結論。
- **有效性**：Core 的驗證規則（013）加上 log facts、block 數、chunk 檢查、逾時（Core 同格時長 × 2 + 60 s）。不足 3 次有效 restart 的格記為 NT。

## 8. Stage C addendum：Fixed-SLO 下的最大並發容量（decision 027、028、030）

**狀態：FROZEN（2026-10-17）**。使用者已選定 SLO S2、mixed 長 request 為 budget、穩態時間窗，並接受一次連續占用 GPU lock（030「使用者決定」）。
harness、smoke 與 C0 完成後，連同 `configs/capacity.toml` 一起凍結（`freeze.capacity = true`）。
〔2026-10-17：C0 的結果為 Rrun → 分支 A（decision 032）；兩次 smoke 都已通過；本節與 `configs/capacity.toml` 凍結（030「凍結紀錄」）。〕
〔凍結後註記，2026-10-17：正式 session `formal-20261017-c1` 的結果見 decision 033（U2K 32 LOWER、M2K / M8K 2 EXACT、U8K NT）。本節規則沒有改變。〕
〔凍結後註記，2026-10-18：使用者決定以相同的凍結協定重跑（034）。重跑的 `formal-20261017-c1b` 有效，成為最終結果（decision 035）：C_PROD 為 U2K 32、U8K 8、M2K / M8K 2，都是 EXACT。本節規則沒有改變。〕
在那之前沒有任何 C1 數據。本節只摘要；完整規則、揭露與依據見 decision 030。第 1–7 節不變，Phase 1 的判定不受影響。

- **C0**（028，探索性）：8K 雙模式的有限度排查（GPU lock 下最多 3.0 h），只決定本節的分支 A–D。
- **平台與固定項**：Core B 的 server argv 原樣（不加觀測 flag），Core B 的 client（`bench_argv`，N = max(10C, 200)），mixed dataset 使用 pin 過的 sha256。
- **估計量**：
  - 每個 workload 的 C_SLO（研究）與 C_PROD（產品上限，只接受 PASS，含 guard band m = 5%）；
  - 在確認集合上向下封閉，取最差的 run，並標明狀態（EXACT / LOWER / CENSORED / INFEASIBLE / UNSTABLE / NT）。
- **時間窗**：穩態 [C, N − C)（使用者選定）；全部 request 的統計量並列報告。
- **SLO**（使用者選定 S2）：短類 P95 TTFT 2 / 6 / 20 s（2K / 8K / 32K），全部 request 的 P95 TPOT 100 ms；mixed 的長 request 也要符合 20 s。
- **Workload**：U2K、U8K、M2K、M8K。O-32K、O-T 不做（使用者沒有選）。
- **搜尋**：
  - 階段 S：夾擠加二分，上限 2·C_KV，只用來選點。
  - 階段 C：在 {Ĉ − 1, Ĉ, Ĉ + 1, c_low} 上，以新的 boot 確認；關鍵點 n = 5，C0 落在分支 B / C 時為 7。
  - 有跳過規則；session 前後各一次 sentinel；replay 檢查。
- **Hypotheses**：H-S1（mixed 降低容量）、H-S2（C_SLO ≤ C_KV）；只凍結選定 SLO 的預測。
- **輸出**：`results/capacity/<session>/analysis.json`。verified profile 只能由產生器從這份 analysis 產生（029）。

## 9. Stage D addendum：容量的延伸（decision 037、039）

**狀態：FROZEN**（2026-10-23）。三個 mini smoke 與審查之後，連同 `configs/stage_d.toml` 一起凍結（`freeze.stage_d = true`，`tests/test_stage_d_frozen.py`）。凍結前沒有任何 Stage D 數據。
本節只摘要；完整規則、揭露與依據見 decision 039。第 1–8 節不變，Phase 1、Phase 2 的判定不受影響。

- **沿用 §8（030）的規則**：
  - 估計量 C_SLO 與 C_PROD，guard band m = 5%，n_near = 5、n_low = 3；
  - 穩態時間窗、搜尋與確認、跳過規則、有效性（含 031 的閒置判定與 spill 偵測）、sentinel 與 replay。
  - 程式上直接 import 凍結的 Stage C 模組，不修改它們。
- **SLO**：S2（037 §3），短類的 TTFT 門檻依 prompt 的長度等級。
- **D-A（封閉迴路，兩個 session）**：
  - U32K；
  - U8K-T、M8K-T：`--long-prefill-token-threshold 512`；
  - U2K-O512、U8K-O512：輸出 512 token。
  - Hypotheses：
    - H-D1：C_SLO(U32K) ≤ C_KV(32K)；
    - H-D2a：C_PROD(M8K-T) > 2；
    - H-D2b：C_PROD(U8K-T) < 8；
    - H-D3：輸出 512 時 C_SLO ≤ C_KV。
- **D-B（開放迴路）**：U2K-P、U8K-P，Poisson 到達，不限制並發。
  - 時間窗為 [⌈0.1 N⌉, N − ⌈0.1 N⌉)；
  - 格點為 λ_ref 的 0.4–1.5 倍，λ_ref 由 c1b 已 commit 的數據重算（數值 pin 在設定檔，測試比對；039 實作補記 13）；
  - 搜尋與確認在格點的索引上進行；
  - H-D4：λ_PROD ≤ 0.9 λ_ref（依格點的倍率判定）。
- **D-C（固定點驗證）**：每個點在 C_PROD 各 5 次 boot：
  - U2K-var、U8K-var：prompt 長度均勻分布，dataset 已 pin sha256；
  - U2K-gw、U8K-gw：經過 capserve gateway，使用正式設定的 local tokenizer。
  - Hypotheses：H-D6（長度不一）、H-D7（經過 gateway）為 PASS。
- **D5**：開場湧入，只用 c1b 的數據作事後描述，不是檢定。
- **輸出**：`results/stage_d/<session>/analysis.json`。新的 verified profile 不會自動取代正式設定的釘選（037 §2）。

## 10. Stage Q addendum：quality 的三個缺口（decision 037、040）

**狀態：DRAFT**。harness、pilot 與審查完成後，連同 `configs/stage_q.toml` 一起凍結（`freeze.stage_q = true`）。在那之前沒有任何 Stage Q 的正式數據。
本節只摘要；完整規則與揭露見 decision 040。第 1–9 節不變，H-A5 與 Core A 的判定不受影響。

- **沿用 §3（005）的規則**：WikiText-2 test、llama.cpp b11146、005 的 flag set、KLD 的流程（BF16 base → self-KLD → 3 個 quant → 刪除 base）、
  以正向標記與預期數量判定成功、KLD 的 z 檢定與 Holm 校正。程式上直接 import 凍結的 `run_quality` / `quality_parse` / `quality_analysis`。
- **Q1（Metal 與 CUDA）**：
  - Mac（Mac lock、Apple 140W adapter、同一 session 一個電力模式）：前 200 chunks 的 KLD，與 8 個設定的完整 PPL；
  - RTX：同樣前 200 chunks 的 KLD；完整 PPL 的對照是 Phase 1 的 `formal-20261010-quality`。
  - H-Q1a：Metal 上 KLD(Q8_0) < KLD(Q6_K) < KLD(Q4_K_M)（4 個檢定，Holm）；
  - H-Q1b：KLD_Metal ÷ KLD_CUDA 在 [0.8, 1.25]；
  - H-Q1c：PPL_Metal ÷ PPL_CUDA 在 [0.995, 1.005]。
- **Q2（長 context）**：RTX，`-c 2048` 與 `-c 8192`，完整的 test set。
  - H-Q2a：兩個 context、兩個 size 的 KLD 順序（8 個檢定，Holm）；
  - H-Q2b：KLD(c) ÷ KLD(512) 在 [0.5, 2.0]。
- **Q3（W4A16）**：vLLM 中 W4A16 對 BF16 的 PPL 與配對 ΔNLL；H-Q3：PPL 比 − 1 ≤ 2%。BF16 checkpoint 需要下載（約 19.3 GB），等使用者同意。
  - Q3 有自己的設定檔與凍結（`configs/stage_q3.toml`、`freeze.stage_q3`）；chunk 檔的 sha256 在凍結前 pin 住。
- **凍結的證據**：每個部分（q1-metal、q1-cuda、q2）一個通過的 pilot（每個 job 10 chunks、全部 ok），而且是同一個 harness（`harness_sha256`）。
- **輸出**：`results/stage_q/<session>/` 與 `results/stage_q/analysis.json`；尚未執行的部分，其 hypothesis 為 NT。
