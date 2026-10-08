# Pre-registration — Local LLM Serving Lab

> **2026-10-08：模型已依 decision 008 改為 Qwen3.5 4B + 9B（Core A）與 RedHatAI Qwen3.5-9B W4A16（Core B）。** 下列與模型綁定的內容在 Stage A 凍結前必須依 008「對實驗的影響」重新推導：H-A1 至 H-A5 的數值預測與範例、fit 規則中的 KV / recurrent 狀態、Core B 的 C_KV 公式、sentinel 與超量測試的配置。本文其餘結構不變。

> **狀態：DRAFT v1（2026-10-08，同日依兩輪 review 修訂）— 尚未凍結。**
> 依 decision 006 分兩階段凍結：
> - **Stage A**（本文第 1–5 節）：在任何 Mac 正式數據之前 commit，並把 `pins.toml` 設為 `freeze.mac = true`。
> - **Stage B**（第 6 節 addendum）：在任何 NVIDIA 正式數據之前 commit，並把 `freeze.nvidia = true`。
>   Stage B **只能代入 Stage A 規則所需的實測值與執行參數，不得修改任何 hypothesis 或規則**。
>
> 凍結後，hypothesis、metrics、task set、protocol 都不得依結果修改。任何變更都要先寫 decision log，並標明是 post-hoc。

決策依據：`docs/decisions/001`–`007`；機器可讀 pin：`configs/pins.toml`。

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

### Core A — single-user（Qwen3 8B / 14B × Q8_0 / Q6_K / Q4_K_M × 2048 / 8192 / 32768 × {M4 Pro Metal, RTX 5070 Ti CUDA}）

**H-A1（fit boundary 由記憶體容量決定）**

- 預測依 004「Fit 預測規則」：N = 檔案 − token_embd + KV；predicted fit：N + 1.5 ≤ V_total − B；predicted OOM：N + 0.8 > V_total − B。
  - RTX 的 V_total、B 依第 4 節的程序在 Stage B 實測後代入，規則本身不得再改。
  - Mac：V_total = 37.44 GiB、B = 0，18 格全部預測 fit。
- *推翻條件*：任何 predicted-fit 的 cell OOM 或 spill；或任何 predicted-OOM 的 cell fit。boundary cell 只報告。

**H-A2（decode 受記憶體頻寬限制）**

- 定義 bytes = (檔案大小 − token_embd) + KV bytes/token × (L − 64)。L − 64 是 128 個生成 token 的平均 depth。
- 預測：同一平台上，L = 2048 時兩個都 fit 且都不是 spill 的配置 A、B，滿足 tok/s(A) ÷ tok/s(B) ≈ bytes(B) ÷ bytes(A)。
- 例：14B-Q4_K_M 與 8B-Q8_0 的權重記憶體接近；依上式，前者 decode 預測約慢 6%。
- *推翻條件*：任一配對 |觀測比 ÷ 預測比 − 1| > 20%（以 process-level mean 計）。

**H-A3（context 的 decode 成本隨 KV 占比增加）**

- 定義降幅 D = 1 − tok/s(L = 32768) ÷ tok/s(L = 2048)。
- 預測：同一平台、同一 size 內，D(Q4_K_M) > D(Q6_K) > D(Q8_0)。依 H-A2 的頻寬模型，KV 在讀取 bytes 中的占比越高，D 越大；
  模型預測值為 8B：0.48 / 0.41 / 0.35，14B：0.36 / 0.30 / 0.25。
- *推翻條件*：任一相鄰配對的順序相反，而且差距超出 noise。若某 (平台, size) 在 32K fit 的配置少於 2 個，記為 not testable。

**H-A4（prefill 受算力限制）**

- (a) 對每個三個 quant 都 fit 的 (平台, size, L)：spread_prefill = 三個 quant 的 prefill tok/s（= L ÷ TTFT）最大值 ÷ 最小值；
  spread_decode 同理。預測 spread_prefill < spread_decode。
  *推翻條件*：任一符合條件的單位出現 spread_prefill ≥ spread_decode。
- (b) 對每個 32K 會 fit 的配置，預測 TTFT(32768) ÷ TTFT(8192) > 4（attention 為 O(L²)，呈超線性）。
  *推翻條件*：任一配置的比值 ≤ 4。

**H-A5（quality–memory frontier）**

- (a) 權重記憶體接近時（約 9 GB），14B-Q4_K_M 優於 8B-Q8_0：WikiText-2 上 584 個 chunk 的配對 ΔNLL（14B-Q4_K_M − 8B-Q8_0）< 0。
  p 值以 chunk bootstrap（10,000 次，雙尾）計算（005）。
- (b) 同一 size 內，KLD 單調遞增：Q8_0 < Q6_K < Q4_K_M，共 4 個 z 檢定（005）。
- **判定只用 Holm**：(a) 與 (b) 共 5 個 p 值，α = 0.05 雙尾，做 Holm step-down；校正後顯著且方向符合預測才算成立。
  - *推翻條件*：任一檢定在 Holm 校正後不顯著，或方向相反。
  - ΔNLL 的 95% bootstrap CI 只作描述，不用於判定。
- Downstream 是 descriptive，不屬於本 hypothesis。事先寫下的預期：在這個 N 下，Q8_0 vs Q6_K 在四個 task 上都是 not resolved
  （配對 95% CI 包含 0）。

### Core B — multi-user（Qwen3-8B-AWQ / vLLM 0.30.0 / RTX 5070 Ti；以 003 smoke test 通過為前提）

- Core B 模型固定為 `Qwen3-8B-AWQ`（使用者決定，003），沒有 fallback。AWQ 若在 smoke test 失敗，Core B 記為 blocked。
- 定義（007）：
  - mix 分為 `uniform` 與 `mixed`。**H-B1 至 H-B3 只用 `uniform`；`mixed` 只用於 H-B4。**
  - 每個 `uniform` request 的 tokens = L，(input, output) = (L − 128, 128)。
  - C_KV(L) = ⌊KV 池 tokens ÷ ⌈L ÷ 16⌉·16⌋。
  - C_knee = grid 中「3 次 restart 平均」output throughput 達到 grid 最大值 95% 的最小 C。
  - C_max = grid 中最大的 C。

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
| fit / OOM / spill | 004「Fit 的操作定義」與 spill 定義 | `llama-completion`（同 cell 的 bench process 若失敗也記為 OOM） |
| engine TTFT(L) | `-p L -n 0` 的 `avg_ns`，同時保留 `samples_ns`；不含 tokenize / HTTP / sampling | `llama-bench` |
| prefill tok/s(L) | L ÷ TTFT(L) | 同上 |
| decode tok/s(L) | `-p 0 -n 128 -d (L-128)` 的 `avg_ts` | `llama-bench` |
| peak memory | **只取 fit-run process**：llama.cpp memory breakdown（model / context / compute）逐 device 與 host 報告，另列合計。Mac 另記 `/usr/bin/time -l` 的 peak memory footprint；RTX 另記 `nvidia-smi memory.used` 峰值減 B。`-lm mmap` 固定 | `llama-completion` |
| quality | 005 | `llama-perplexity`（只在 Mac） |

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
| `llama-completion`（fit / memory） | 004 的指令：`-c L -np 1 --fit off -ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -lm mmap -t N_thr --poll P -n 128 --ignore-eos -no-cnv`；prompt 為 WikiText-2 test 的前 L − 128 個 token（harness 從 log 驗證 token 數） |
| `llama-perplexity`（quality） | 005 的 flags |

- **N_thr、P 依平台各自固定**：
  - Mac：N_thr = 8（M4 Pro 的 performance core 數，`hw.perflevel0.logicalcpu = 8`）。
  - P 的選擇規則（事先寫定）：pilot 在 sentinel cell 比較 `--poll 0` 與 `--poll 50`。只有在 `--poll 0` 的 decode tok/s 中位數低超過 3% 時才用 50，
    否則用 0（避免 CPU 空轉加熱 SoC）。
  - RTX：N_thr = WSL 內 host CPU 的 physical core 數，P 用同一規則，皆於 Stage B 決定。
- `-fa on` 之前，pilot 先以 `-fa auto` 確認兩個 backend 的 log 都出現 "Flash Attention enabled"。

### 重複次數與順序

- Core A：每個 cell 3 個獨立 process × `-r 5`；每個 session 內，cell 的執行順序以固定 seed 隨機排列，seed 記入結果。
- Core B：每個 (L, mix) 3 次 server restart（007）。

### RTX idle baseline B 的量測程序（Stage A 凍結）

1. 在 GPU lock 下進行，且沒有任何其他 GPU process。
2. 最後一個 GPU process 結束後，閒置 5 分鐘。
3. 以 1 Hz 取樣 `nvidia-smi memory.used` 共 60 秒，取中位數，即為 B。同時記錄螢幕是否接在這張卡上。
4. Stage B 量得的 B 是 H-A1 唯一使用的值。之後每個 session 開頭重量一次；若超過凍結的 B 0.25 GiB 以上，該 session 標記為 drift，重跑。

### 熱與背景負載控制

- **Mac**：
  - 接 AC 電源；所有正式 run 的 `powermode` 都必須相同（目前為 2 = high power）；`iogpu.wired_limit_mb` 必須為 0。
  - process 之間至少閒置 30 秒。
  - **sentinel cell**（8B-Q4_K_M decode @ L = 2048）在每個 session 開頭與結尾各量一次。兩次的 decode tok/s 相差 > 3% 時，
    整個 session 標記為 drift，所有 cell 在新的 session 重跑。
  - 每個 process 前後都跑一次 `env_capture`（thermal / loadavg / Mac lock owner）。
- **RTX**：
  - 在 GPU lock 下執行，`env_capture --gpu-lock-held`。
  - `enforced.power.limit` 必須等於 `power.default_limit`。
  - 每個 process 前後記錄溫度、SM clock、`clocks_event_reasons`。
  - sentinel 規則同上。
- **環境變數**：`GGML_*`、`CUDA_*`、`VLLM_*` 等一律不設定，除非本文件明列。`env_capture` 會記錄這些變數，名稱疑似 credential 的一律遮蔽值。

### Prompt 與 sampling

- Core A 效能：`llama-bench` 的合成 token（不經 chat template；token 內容由 libc `rand()` 產生，兩平台不同，對 dense 模型影響可忽略）。
- fit run：固定的 WikiText-2 token 序列。
- quality：log-likelihood scoring，沒有 sampling。
- Core B：`uniform` 用 random dataset，`mixed` 用自訂 dataset，seed 皆為 20261008（007）；
  `temperature 0`、`--ignore-eos`、`--generation-config vllm`。

### 失敗也要報告

OOM、spill、crash、drift、not testable、inconclusive 的單位一律進結果表，不刪除。

## 5. Stage A 凍結 checklist（任何 Mac 正式數據之前）

- [ ] 8 個衍生 GGUF 的 sha256 / size 都已填入 `pins.toml`（`freeze.mac = true` 由測試強制）
- [ ] converter 的 Python lock 已 commit，`pins.toml` 的 `core_a.converter_lock` 指向它（004）
- [ ] Quality：HellaSwag / MMLU 子集的題號清單、子集 `.bin` 的 sha256、MMLU 排除清單已 commit（005）
- [ ] Mac pilot 完成：
  - [ ] `-fa auto` probe
  - [ ] `--poll` 依規則選定
  - [ ] throughput 校準
  - [ ] determinism
  - [ ] memory metric 交叉核對
  - [ ] quality 數值 sanity check
  - [ ] MDE（遮蔽方向）
- [ ] Benchmark harness 與輸出格式固定，記錄 harness 的 git commit
- [ ] Mac 重負載 lock 的參與方式已由使用者決定，並寫入 MACHINE_POLICY 或 decision log
- [ ] 本文件第 1–5 節狀態改為 FROZEN (Stage A) 並 commit

## 6. Stage B addendum（任何 NVIDIA 正式數據之前；只能代入實測值與執行參數）

待填（遠端可用後；所有 GPU 步驟都在 GPU lock 下）：

- [ ] RTX 實測：V_total（`memory.total`）、B（依第 4 節程序）、C_ctx（004）、是否接螢幕、HAGS、`.wslconfig`
- [ ] sysmem 超量測試（14B-Q8_0、`-c 65536`，不屬於 18 格）：hard OOM 或 spill
- [ ] llama.cpp CUDA build：toolkit 版本、`wsl2-cuda` verified、RTX 的 N_thr / P（依第 4 節規則）
- [ ] vLLM：smoke test 通過（003 gate）、lock 檔已 commit（`vllm.lock_file`）、`VLLM_WSL2_ENABLE_PIN_MEMORY`（依 003 規則）、
      attention backend、performance mode / async scheduling 的實際生效值
- [ ] Core B：
  - KV 池 tokens（`GPU KV cache size`）
  - 依 007 規則算出每個 L 的 C_KV 與 grid
  - `mixed` dataset 的 sha256（`core_b.mix_dataset_sha256`）
  - `--kv-cache-memory-bytes`、`--gpu-memory-utilization`
- [ ] 本節狀態改為 FROZEN (Stage B) 並 commit
