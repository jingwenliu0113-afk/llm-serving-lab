# 008 — 模型重新選擇：Qwen3.5 4B + 9B（Core A）、RedHatAI Qwen3.5-9B W4A16（Core B）

- 狀態：**Accepted（agent 選定，2026-10-08）；下載前待使用者確認最終選擇與下載量**
- 日期：2026-10-08
- 取代：
  - 004 的 model family / size / 量化來源 repo 與 fit 預測表。量化方式（本地 Q8_0 / Q6_K / Q4_K_M、不用 imatrix）、
    context set（2048 / 8192 / 32768）、fit 規則的結構**保留**。
  - 003 的 Core B 模型（Qwen3-8B-AWQ）、precision 與 `--dtype`。vLLM 版本與其他 server protocol **保留**。
- 依據：使用者指示（2026-10-08）重新以目前世代的 Qwen 選模型；研究 workflow（4 位研究者 + 4 位查核者），
  以及我依 HF API 親自核對的 revision、sha256 與 config。
- 機器可讀：`configs/pins.toml` `[core_a]`、`[core_b]`

## 背景

使用者要求優先使用 2026 年目前世代的模型，不要因為舊的 pre-registration 而沿用 2025-04 的 Qwen3。條件如下：

- Core A 需要同一 family 的兩個 size，而且兩台機器都要能做合理的 benchmark。
- Core B 需要能在 16 GB 上留下 KV 空間的 quantized checkpoint。
- 選擇必須服務 TTFT / TPOT / throughput / P50 / P95 / KV pressure / concurrency 等研究問題，而不是追求最大的模型。

### 2026-10-08 的 Qwen 開源模型（HF API 查核）

| 世代 | 發布 | dense size | 其他 | 架構 |
|---|---|---|---|---|
| Qwen3.8 | 2026-08 | **只有 27B** | Flash-Next（約 180B，新架構 qwen4exp）、2.4T-A95B | 延續 Qwen3.5 的 architectural foundation |
| Qwen3.6 | 2026-04 | **只有 27B** | 35B-A3B MoE | 延續 Qwen3.5 的 architectural foundation |
| Qwen3.5 | 2026-02 | 0.8B / 2B / **4B / 9B** / 27B | 35B-A3B、122B-A10B、397B-A17B MoE | 3:1 hybrid（3 層 Gated-DeltaNet + 1 層 gated full attention）、VLM 封裝、原生 262,144 context |
| Qwen3 | 2025-04 | 0.6B–32B（含 8B / 14B） | 30B-A3B 等 | 傳統 full attention、純文字 |

- **Qwen3.6 / 3.8 延續 Qwen3.5 的 architectural foundation**（官方 README 的說法；本專案只比對過三者 27B 的
  config 維度與參數量，未逐項比對訓練與其他差異，所以不宣稱是「相同架構」）。
- 但目前 3.6 / 3.8 **沒有**適合本實驗的官方 dense 組合：沒有 4B / 9B 這兩個 size，唯一的 dense size 27B 又不符合
  RTX 16 GB 的條件。**因此 Core 選 Qwen3.5 4B + 9B。**
- 2026 年沒有任何約 14B 的 Qwen dense 模型。`Qwen3.6-9B`、`Qwen3.8-9B` 等 repo 不存在（回 401），或者只是社群 finetune。
- 4B 與 9B 的 attention / GDN 幾何完全相同：32 層 = 8 層 full attention（4 個 KV head × 256）+ 24 層 GDN。
  - f16 KV = **32,768 B/token（32 KiB）**，Qwen3-8B 是 144 KiB。
  - 每條序列另有固定的 recurrent 狀態，約 **50.25 MiB**（llama.cpp 以 f32 儲存）。

## 候選

### 候選 1：Qwen3.5 dense **4B + 9B**（← 選定）

1. **Size**：
   - `Qwen/Qwen3.5-4B` @ `851bf6e8…`：4.66B，含 vision 0.33B 與 MTP 0.12B；tied embedding。
   - `Qwen/Qwen3.5-9B` @ `c2022362…`：9.65B，含 vision 0.46B 與 MTP 0.24B；untied。
2. **Mac 48 GB**：18 格全部 fit。最大的是 9B-Q8_0 @ 32K，約 9.9 GiB，遠小於 37.44 GiB。
3. **RTX 5070 Ti 16 GB**：18 格全部 fit。最大的 GPU 需求是 9B-Q8_0 @ 32K 的 8.90 GiB，B = 1 GiB 時的 fit 上限約 13.42 GiB，
   **在現有 context 下沒有 fit 邊界**。
4. **Core A 的 quantization**：Q8_0 / Q6_K / Q4_K_M，本地量化。pinned `llama-quantize` 已支援 qwen35。
   預估 GGUF 大小（±3%，以實際產出為準）：
   - 9B：9.51 / 7.35 / 5.62 GB
   - 4B：4.47 / 3.45 / 2.70 GB
   - BF16 reference：9B 約 17.9 GB、4B 約 8.4 GB，Mac 都放得下，所以 005 的 KLD 設計可以保留。
5. **Core B**：`RedHatAI/Qwen3.5-9B-quantized.w4a16` @ `a398088c…`。
   - 格式：compressed-tensors GPTQ W4A16、group 128。GDN projection、embed、lm_head 維持 BF16。
   - 純文字 GPU 權重約 9.35 GiB，KV 池約 2.2–3.2 GiB。
   - C_KV(2K / 8K / 32K) 約 19–28 / 7–10 / 2（1 block = 16.5 MiB，池 2.2–3.2 GiB = 136–198 blocks；32K 要到 3 需 ≥ 3.21 GiB）。
     舊方案 Qwen3-8B-AWQ 約 22–24 / 5–6 / 1。池的高端在每個 L 都不比它差；低端的 2K 可能略少（009 更正，原寫 23–28 / 8–10 / 2–3）。
6. **最大風險**：
   - **H-A1 在 RTX 上退化**，沒有 fit 邊界。
   - KV/token 比舊方案小 4.5 倍，而且兩個 size 相同，context 對記憶體與 decode 的影響明顯變小。
   - hybrid 的 runtime 路徑較新。vLLM 在 sm_120 + WSL2 上的 GDN kernel 需要 JIT（FlashInfer CuTe-DSL / Triton），
     而且有 WSL2 上 Triton segfault 的回報（vLLM #39149、triton #9939 仍未解決）。
   - Core B 是第三方 checkpoint，沒有官方 9B 以下的 quant。

### 候選 2：Qwen3.5 / 3.6 / 3.8 dense **9B + 27B**（← 不選）

1. **Size**：9B（Qwen3.5）與 27B（3.5 / 3.6 / 3.8 的 27B 皆可；3.6 / 3.8 延續 Qwen3.5 的 architectural foundation）。
2. **Mac 48 GB**：可行。27B-Q8_0 @ 32K 約 30 GiB。但 27B 的 BF16 reference（約 50 GiB）**放不進 Mac**，
   005 的 KLD vs BF16 設計因此無法用在 27B。
3. **RTX 5070 Ti 16 GB**：**27B 在每個 quant、每個 context 都 OOM**。Q4_K_M @ 2K 的 GPU 需求 14.97 GiB，超過上限 13.42 GiB。
   這違反「兩個 size 都要能在兩台機器上做合理 benchmark」的條件。
4. **Core A 的 quantization**：9B 同候選 1。27B 在 RTX 上只有 Q3 以下才放得下，必須改 quant set。
5. **Core B**：27B 不可行。官方 27B GPTQ-Int4 為 28.16 GiB、FP8 為 28.75 GiB，社群 4-bit 也都 ≥ 17.7 GiB，
   所以 Core B 只能用 9B，結論與候選 1 相同。
6. **最大風險**：Core A 有一半的 cell 在 RTX 上直接失敗；Mac 端 27B 每個 run 都很慢（decode 估約 9.5 tok/s）；
   完整的 27B 磁碟流程約 233 GiB，超過可用空間。

### 參考：舊方案 Qwen3 8B + 14B（2025-04）

這是唯一能在 16 GB 上產生內部 fit 邊界的 Qwen 組合。Core B 有官方 AWQ checkpoint，KV 語意也最單純。
但它是前一代的純 full-attention 架構，不代表 2026 年部署的主流；使用者已明確要求優先目前世代，所以不選。

## 決策

- **Core A：Qwen3.5-4B + Qwen3.5-9B**。量化方式、quant set、context set 皆不變。
- **Core B：`RedHatAI/Qwen3.5-9B-quantized.w4a16` @ `a398088c4228b0ae0c8c78df88fd1e4bf445f068`**，`--dtype bfloat16`。
  只有這一個 checkpoint，沒有 fallback（延續使用者「不要擴大模型選擇矩陣」的原則）。
  smoke test 失敗時記為 blocker 並回報使用者，不自動換模型。
- **Context set 維持 2048 / 8192 / 32768**，不為了恢復 fit 邊界而延伸到 256K。
  理由：9B 要到 256K 才會在 RTX 出現邊界，但單次 256K prefill 在 M4 Pro 上粗估約 46 分鐘（FLOP 估計、未實測）。
  每個 cell 需要多次 prefill，Mac 端 Core A 會膨脹到數百小時。fit / OOM boundary 改以「各 cell 的 headroom」報告。
- **不需要更換 runtime pin**：
  - llama.cpp b11146 已註冊 `Qwen3_5ForConditionalGeneration` → `qwen35`，Metal 與 CUDA 都有 GDN / SSM_CONV / FA(256) kernel。
  - vLLM 0.30.0 已註冊同一架構，並支援 `--language-model-only`。

## 為什麼選候選 1

1. 3.6 / 3.8 延續 Qwen3.5 的 architectural foundation，但沒有適合本實驗的 4B / 9B 官方 dense 組合；
   Qwen3.5 4B + 9B 是目前世代中，**唯一**兩個 size 都能在兩台機器上完整跑完 Core A 的組合。
2. 兩個 size 的 KV / 狀態幾何相同，size 軸只改變權重，size 與 quant 對記憶體的效果可以乾淨分解。
3. hybrid attention 正是 2026 年部署的重點。「長 context 的 KV 成本大幅下降，換來每條序列固定的 recurrent 狀態」
   本身就是本研究問題（memory frontier、KV pressure、concurrency）值得報告的結果。
4. Core B 的 9B W4A16 與 Core A 的 9B 是同一個 base；在 16 GB 上，8K / 32K 的 KV 容量優於舊方案，2K 大致相當（009 更正）。

## 放棄的替代方案

- **候選 2（9B + 27B）**：27B 在 RTX 上全部 OOM，Core B 也不可行（見上）。
- **Qwen3 8B + 14B**：前一代架構（見上）。
- **context 延伸到 256K 以恢復邊界**：Mac 成本不可行（見上）。
- **把 BF16 當成一個 treatment 來製造邊界**（9B BF16 在 RTX 上全部 OOM）：會更動 quant set，而且與 005「BF16 是 reference」的角色衝突。不採用。
- **Core B 的其他選項**：
  - `Qwen/Qwen3.5-4B` BF16（官方，KV 最多）：沒有量化，不符合「quantized checkpoint」的要求。
  - `cyankiwi/Qwen3.5-9B-AWQ-4bit`：社群上傳，recipe 沒有文件。
  - `Intel/Qwen3.5-9B-int4-AutoRound`：repo 內有 93 個重複 tensor，沒有 license 欄位，GDN gate 也量化成 int4。
  - `RedHatAI/Qwen3.5-9B-FP8-dynamic`：KV 空間不到 0.8 GiB。
  - 官方 27B 的 FP8 / GPTQ：都超過 16 GB。

## 對實驗的影響

以下項目都與模型綁定，必須在 **Stage A 凍結前**依本 log 重新推導或修訂。
pre-registration 的其餘結構（兩階段凍結、metrics、harness、env capture）**不重做**。

1. **Fit 規則（004）**：
   - KV 只計 8 層 full attention（32 KiB/token），另外加上每條序列約 50.25 MiB 的 recurrent 狀態。
   - 4B 是 tied embedding，計算 GPU 需求時**不扣** token_embd（output 會在 GPU 上保留一份副本）。
   - 刻意超量測試改用 9B-Q8_0 搭配 `-c 262144` 以上（仍不屬於 18 格）。
2. **PREREGISTRATION 的 hypothesis**：
   - H-A1：在兩台機器上都預測全部 fit。
   - H-A2：bytes 公式要加入 recurrent 狀態的讀寫（約 2 × 50 MiB/token），並處理 tied 4B 的情況；
     原本「約 9 GB 同記憶體」的例子不再成立。
   - H-A3：預測的 D 降到 9B 約 0.11–0.17、4B 約 0.18–0.25，相鄰差距約 0.03，可能接近 noise，需要重寫。
   - H-A4(b)：TTFT(32K) ÷ TTFT(8K) 的 FLOP 預測值只剩 9B 約 4.45、4B 約 4.84，與門檻 4 的距離很小。
   - H-A5(a)：沒有接近同記憶體的配對（9B-Q4_K_M 5.62 GB vs 4B-Q8_0 4.47 GB），需要重新表述。
3. **Core B（003 / 007）**：
   - C_KV 公式要改成 hybrid 版：
     C_KV(L) = ⌊(num_gpu_blocks − 1) ÷ (⌈L ÷ block_size⌉ + 3)⌋，其中 block_size 在 log 中約為 528，
     每個 request 另需 3 個 GDN 狀態 block。
   - Prometheus 的「KV cache usage %」會把 GDN 狀態 block 也算進去。
   - 新增必須固定的 flag：`--language-model-only`、`--dtype bfloat16`、`--gdn-prefill-backend`（smoke test 依規則選定）、
     `VLLM_GDN_DECODE_KERNEL=triton`（避開 0.30.0 的 #59520 regression，必須記錄），並在 log 中確認 block size。
     `--no-enable-prefix-caching` 繼續保留。
   - 003「sm_120 上最不依賴 JIT」的理由不再成立。smoke test 必須驗證 GDN kernel 在 WSL2 + sm_120 上可用。
4. **Quality（005）**：
   - tokenizer 改變（vocab 248,320），WikiText 的 chunk 數與計分 token 數要依 Qwen3.5 tokenizer 重算。
   - MMLU 的結構性排除清單要依新 tokenizer 計算；題號清單本身與模型無關，不受影響。
   - KLD logits 檔約 74 GB / size（原為 45 GB），仍只在正式 run 時於本地產生。
5. **轉換**：使用 `convert_hf_to_gguf.py --outtype bf16 --no-mtp`，不加 `--mmproj`，只取文字部分。
   pinned converter 的 transformers 4.57.6 不認得 qwen3_5，會改讀 `config.json`。
   正式下載前，先用 `--vocab-only` 搭配 config 與 tokenizer 小檔驗證 converter 與 qwen35 pre-tokenizer。
6. **Pilot 必查**：fused GDN 沒有 device support probe，可能默默退回 CPU。
   要比對 `graph: nodes = …, splits = …`，或使用 `GGML_SCHED_DEBUG=2` 確認沒有多出的 split。

## Core B smoke test：分段進行（使用者指示，2026-10-08）

目標只有一個：確認 9B W4A16 能在 RTX 5070 Ti 16 GB 上**正常啟動、推論、釋放資源**。
不做 throughput 量測，不開高 concurrency。結果只用來判定 pass / fail（006）。

- **一律 text-only**：`--language-model-only`。本研究不使用 vision，所以不載入 vision tower；MTP 由 vLLM 自動略過。
- 共同設定：`HF_HOME=/home/wenn0113/.cache/huggingface`、`HF_HUB_OFFLINE=1`、`VLLM_GDN_DECODE_KERNEL=triton`、
  `--dtype bfloat16`、`--no-enable-prefix-caching`、`--gpu-memory-utilization 0.85`、`--max-num-seqs 8`、
  `--revision a398088c…`，以及 `--gdn-prefill-backend`（smoke 時記錄實際使用的 backend）。
- 每一段開始前都先看 `nvidia-smi`。若另有 GPU process 在用，就不啟動。
  smoke 屬於小型啟動測試，依 MACHINE_POLICY 不取 GPU lock。

| 段 | `--max-model-len` | 送出的 request | 通過條件 |
|---|---|---|---|
| S1 | 2048 | 1 個，prompt 約 1,800 tokens，`max_tokens` 128，`temperature` 0 | 啟動成功；log 出現 language-model-only、attention block size、`GPU KV cache size`、GDN kernel；回應正常；關閉後 60 秒內 `memory.used` 回到啟動前 ±256 MiB，沒有殘留 process |
| S2 | 8192 | 1 個，prompt 約 7,900 tokens | 同上 |
| S3 | 32768 | 1 個，prompt 約 32,500 tokens | 同上 |

- **前一段通過才進下一段。** 任何一段失敗就記為 blocker 並回報使用者，不自動換模型或調參數。
- S1–S3 都通過後，才依 003 的 gate 進行需要 GPU lock 的步驟：JIT 計時、`VLLM_WSL2_ENABLE_PIN_MEMORY` A/B、`vllm bench serve`、量 B。
- **前提**：WSL 目前沒有任何 C compiler（`cc` / `gcc`），而 vLLM 的 Triton / FlashInfer JIT 需要它，S1 很可能因此失敗。
  需要使用者先以 sudo 安裝 `build-essential`（NEXT.md）。

## 預估下載量（尚未下載）

| 檔案 | 位置 | 大小 |
|---|---|---|
| `Qwen/Qwen3.5-4B` safetensors（含 vision / MTP，轉換時只取文字部分） | Mac | 9.34 GB |
| `Qwen/Qwen3.5-9B` safetensors | Mac | 19.33 GB |
| `RedHatAI/Qwen3.5-9B-quantized.w4a16`（完整 snapshot，含 MTP 檔） | RTX WSL | 11.46 GB |
| **合計需下載** | | **約 40.1 GB** |

另外會在本地產生下列檔案（不是下載）：

- BF16 GGUF：約 26.3 GB。
- 6 個 quant：約 33.1 GB。這些檔案還要傳到 RTX 端；或在 WSL 重跑同一 pipeline，那樣 safetensors 要在遠端再下載一次，約 28.7 GB。
- KLD logits：約 74 GB / size，只在 Stage A 凍結後的正式 quality run 才產生，用完即刪。
