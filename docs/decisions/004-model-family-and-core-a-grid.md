# 004 — Model family、Core A 的 size × quant × context grid、量化來源、fit 規則

> **2026-10-08：model family / size / 來源 repo / fit 預測表已由 008 取代**（改為 Qwen3.5 4B + 9B，hybrid attention）。量化方式（本地 Q8_0 / Q6_K / Q4_K_M、不用 imatrix）、context set 與 fit 規則的結構仍有效；KV 與 recurrent 狀態的算法依 008 修訂。下方的 Qwen3 數字保留作為決策歷史。

- 狀態：Accepted（v1；在 Stage A pre-registration commit 前仍可修改，修改需留 log）
- 日期：2026-10-08（同日依 review 修訂：fit 規則、quant set 理由、轉換 pipeline）
- 對應：SPEC Day 1 #5「選 model family」、SPEC Core A「同一 model family：2 種 parameter size × 3 種 quantization × 3 種 context length」
- 機器可讀：`configs/pins.toml` `[core_a]`

## 背景

SPEC 要求 Core A 用同一個 model family、2 種 size、3 種 quant、3 種 context，在 48 GB unified memory 的 M4 Pro
與 16 GB VRAM 的 RTX 5070 Ti 上量 fit / TTFT / 生成速度 / peak memory / quality。
這個選擇決定了 fit boundary 會不會出現在 grid 裡、quantization 軸是否乾淨，以及 context 軸會不會被 RoPE scaling 汙染。

## 決策

| 項目 | 選擇 |
|---|---|
| Family | **Qwen3 dense**（Apache-2.0） |
| Sizes | **8B**（`Qwen/Qwen3-8B` @ `b968826d…`）、**14B**（`Qwen/Qwen3-14B` @ `40c06982…`） |
| Quantizations | **Q8_0 / Q6_K / Q4_K_M** |
| Context lengths | **2048 / 8192 / 32768** |
| 量化來源 | **本地從官方 BF16 safetensors 產生**，不用官方 GGUF；**不用 imatrix** |

來源 shard 的 sha256 全部寫在 `pins.toml`（取自 HF API 的 LFS oid）。衍生的 GGUF 在 Week 1 產生時記錄 sha256 與大小；
Stage A 凍結（`freeze.mac = true`）時必須全部填滿，由 `tests/test_pins.py` 強制。

### 轉換 pipeline（Week 1 執行前凍結 converter 環境）

```bash
# pinned llama.cpp source（002）內的 converter；Python 環境另以 lock 凍結（見下）
python third_party/llama.cpp/convert_hf_to_gguf.py <snapshot@revision> --outtype bf16 --outfile Qwen3-<S>-BF16.gguf
third_party/llama.cpp/build-metal/bin/llama-quantize Qwen3-<S>-BF16.gguf Qwen3-<S>-<Q>.gguf <Q>   # Q ∈ {Q8_0, Q6_K, Q4_K_M}
```

- converter 的依賴（pin 內 `requirements-convert_hf_to_gguf.txt`：`torch==2.11.0`、`transformers==4.57.6`，
  其餘為範圍）在轉換前以 `uv pip compile --generate-hashes` 凍結成 lock 檔，放在 `configs/`。
- 依 pin 的 `llama-quant.cpp`，Qwen3（`tie_word_embeddings = false`）的 `token_embd` 型別跟隨 ftype：
  Q8_0 → Q8_0、Q6_K → Q6_K、Q4_K_M → Q4_K；`output.weight` 在 Q6_K / Q4_K_M 為 Q6_K。

## 為什麼

### Qwen3

篩選條件：同一 family 有約 8B 與約 14B 兩個 dense size；llama.cpp 與 vLLM 都支援；license 允許公開 benchmark；
16 GB 卡上會出現真正的 fit boundary，48 GB Mac 則全部放得下。只有 Qwen3 全部符合（被淘汰的候選見下方）。
Qwen3（2025-04）距今約 1.5 年；選它是因為**唯一符合全部條件**，不是因為最新。README 要明講這點。

### 自己量化，不用官方 GGUF（關鍵）

官方 `Qwen/Qwen3-{8B,14B}-GGUF` 的 **Q4_K_M 是另一套 recipe**。GGUF header 寫著
`general.finetune = awq-compatible-Instruct`（`general.name = "Qwen3 8B Awq Compatible Instruct"`），
代表它先套用 AWQ scale 前處理，再做 k-quant；Q8_0 / Q6_K 則是 `Instruct`，屬一般量化（2026-10-08 讀 header range 確認）。
若直接使用，「quantization」這個變因會同時混入「bit 數」與「AWQ 前處理」，Q4_K_M 的品質會被高估。

本地統一 pipeline 的好處：

1. 三個 quant 出自同一個 BF16、同一支 quantizer、同一個 commit，quantization 軸乾淨。
2. BF16 GGUF 同時是 quality KLD 的 reference（005）。
3. 下載量反而較少：只需 BF16 約 46 GB；官方 6 個 quant 約 57 GB，另外還要再下載 reference。

**不用 imatrix**：imatrix 需要另選 calibration 資料，等於多一個變因；不用 imatrix 的 k-quant 是定義最清楚的 treatment。
代價是 Q4_K_M 的品質比 imatrix 版略差，這點必須寫進報告。

### Q8_0 / Q6_K / Q4_K_M

- 約 8.5 / 6.6 / 4.8 bpw，大致等距地涵蓋「接近無損 → 常用預設」的範圍。
- 三者都不需要 imatrix（低 bit 的 IQ 系列需要）。
- 這組組合讓 16 GB 的 fit boundary 正好落在 grid 內部（見下方預測表）；換成更低 bit，14B 就全部放得下，邊界會消失。

### 2048 / 8192 / 32768

- 以 ×4 等比間隔涵蓋短、中、長。
- 最大值等於 Qwen3 原生 window 32768，**不需要 YaRN / RoPE scaling**。config 的 `max_position_embeddings = 40960`
  依 Qwen 說明是保留 **32,768 輸出 + 8,192 prompt**（`Qwen/Qwen3-8B` README @ `b968826d`；GGUF README 沒有這段），
  官方建議 ≤ 32k 不要開 YaRN。
- 都是 256 的倍數。llama.cpp 會把 `n_ctx` pad 到 256，這樣 `llama-bench` 與 `-c` 的 KV 大小才會一致。

**Context level L 的操作定義（三種量測都讓 KV 配置 = L）：**

| 量測 | 工具與參數 |
|---|---|
| TTFT | `llama-bench -p L -n 0` |
| 生成速度 | `llama-bench -p 0 -n 128 -d (L-128)` |
| fit / peak memory | `llama-completion -c L`，prompt 為 L−128 tokens，`-n 128 --ignore-eos`（見下方 fit 定義） |

## Fit 的操作定義（Stage A 凍結）

一個 cell「fit」= 下面這個 process 成功完成 prefill (L−128) + decode 128，且沒有任何 allocation 錯誤：

```text
llama-completion -m <gguf> -c L -np 1 --fit off -ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 \
  -lm mmap -t N_thr --poll P -f <prompt of L-128 tokens> -n 128 --ignore-eos -no-cnv  (stderr 存檔)
```

- **不允許 partial offload**。
- 只要該 cell 的 fit、TTFT、decode 任一 process 出現 allocation 錯誤，這個 cell 就記為 OOM，並記錄失敗的工具與階段
  （load / context 建立 / prefill / decode）。
- Mac 的 working set 是 soft limit，Metal OOM 可能在 compute 時才出現（`kIOGPUCommandBufferCallbackErrorOutOfMemory`），
  同樣記為 OOM。**不得**修改 `iogpu.wired_limit_mb`（系統設定），`env_capture` 會記錄它，必須為 0。
- **RTX sysmem fallback 分支**：Stage B 前在 GPU lock 下刻意超量一次，確認 WSL2 是 hard OOM 還是溢出到系統記憶體。
  超量測試使用 **18 格以外**的配置（14B-Q8_0 搭配 `-c 65536`），結果只用來判斷 hard OOM 或 spill 的行為（006）。
- **spill（只適用 RTX）**：process 完成，但符合下列任一條件。spill **不算 fit**，在結果表中單獨標示；
  spill cell 不進入 H-A2 / H-A3 / H-A4 的比較單位（該單位記為 not testable）。
  1. 記憶體：fit run 的 llama.cpp breakdown 中 GPU device 的 self（model + context + compute）+ C_ctx + B > V_total。
     其中 C_ctx（CUDA context 等 breakdown 以外的 overhead）於 Stage B 在 GPU lock 下量一次：
     8B-Q4_K_M @ 2048 fit run 期間 `memory.used` 的峰值減 B，再減該 run 的 self。
  2. 速度：該 cell 的 `llama-bench` decode `avg_ts` < 0.5 × expected。
     expected = anchor 的實測 decode `avg_ts` × bytes(anchor) ÷ bytes(cell)。
     anchor 為同平台、同 L 的 8B-Q4_K_M（若 anchor 本身是 spill，改用 L = 2048 的 8B-Q4_K_M）；
     bytes 定義同 H-A2：(檔案 − token_embd) + KV bytes/token × (L − 64)。

## Fit 預測規則（H-A1 的依據；規則在 Stage A 凍結，輸入在 Stage B 實測）

- 每個 cell 的 GPU 需求：**N = 檔案大小 − token_embd bytes + KV(L)**。CUDA full offload 時 `token_embd` 留在 host；
  Metal 是 unified memory，Mac 端不扣。
- f16 KV bytes/token = 2 × layers × kv_heads × head_dim × 2：
  - 8B：2 × 36 × 8 × 128 × 2 = **147,456 B（144 KiB）**
  - 14B：2 × 40 × 8 × 128 × 2 = **163,840 B（160 KiB）**
- 非權重 / 非 KV 的 overhead O ∈ [0.8, 1.5] GiB（CUDA context 0.3–0.5 + compute buffer 0.5–1.0）。
- RTX：V_total = 實測 `memory.total`，B = 實測 idle VRAM baseline（量測程序見 PREREGISTRATION §4，於 Stage A 凍結）。
  分類規則：
  - **predicted fit**：N + 1.5 ≤ V_total − B
  - **predicted OOM**：N + 0.8 > V_total − B
  - 其餘為 **boundary**（只報告，不算 H-A1 的推翻依據）
- Mac：V_total = Metal working set 38338 MiB ≈ 37.44 GiB（2026-10-08 實測 `--list-devices`），B = 0，N 不扣 token_embd。

以 V_total = 15.92 GiB（16303 MiB，2026-10-08 已實測 `memory.total`）試算 N（GiB），並列出 B = 0 / B = 1.0 GiB 兩種 baseline 下的分類。
閒置時 `memory.used` 的參考值為 1026 MiB（約 1.0 GiB），但正式的 B 必須依 PREREGISTRATION §4 的程序量測。
檔案大小暫用同 tensor type 的官方檔案大小，本地檔案產生後再更新。

| cell | token_embd | N @ 2048 | N @ 8192 | N @ 32768 |
|---|---|---|---|---|
| 8B-Q8_0 | 0.62 | 7.78 fit / fit | 8.62 fit / fit | 12.00 fit / fit |
| 8B-Q6_K | 0.48 | 6.07 fit / fit | 6.91 fit / fit | 10.29 fit / fit |
| 8B-Q4_K_M | 0.33 | 4.64 fit / fit | 5.48 fit / fit | 8.86 fit / fit |
| 14B-Q8_0 | 0.77 | 14.16 **fit / OOM** | 15.10 **boundary / OOM** | 18.85 OOM / OOM |
| 14B-Q6_K | 0.59 | 11.01 fit / fit | 11.95 fit / fit | 15.70 OOM / OOM |
| 14B-Q4_K_M | 0.41 | 8.29 fit / fit | 9.23 fit / fit | 12.98 fit / fit |

- 分類取決於實測的 B（以 V_total = 15.92 GiB 試算各 cell 的門檻）：
  - 14B-Q8_0 @ 32K、14B-Q6_K @ 32K：任何 B ≥ 0 都預測 OOM。
  - 14B-Q8_0 @ 8K：B > 0.02 GiB 即預測 OOM，否則為 boundary。
  - 14B-Q8_0 @ 2K：B ≤ 0.26 為 fit，0.26–0.96 為 boundary，> 0.96 為 OOM。
  - 14B-Q4_K_M @ 32K：B ≤ 1.44 為 fit（此時若 OOM 就推翻 H-A1），1.44–2.14 為 boundary，> 2.14 為 OOM。
  - 8B-Q8_0 @ 32K：B ≤ 2.43 為 fit；14B-Q6_K @ 8K：B ≤ 2.48 為 fit；其餘 cell 在 B ≤ 3.41 時皆為 fit。
- **M4 Pro**：最大的 N（14B-Q8_0 @ 32K，不扣 token_embd）= 19.62 GiB，加上 O 仍遠小於 37.44 GiB，18 格全部預測 fit。

這個結果要寫成「memory architecture 的 finding」（48 GB unified vs 16 GB discrete 各自的可行配置），
**不是**排名（SPEC / MACHINE_POLICY）。

## 放棄的替代方案

| 候選 / 做法（2026-10 查核） | 不選的原因 |
|---|---|
| Qwen3.5 / 3.6 / 3.8 | 沒有約 14B 的 dense size；hybrid linear attention 只有 1/4 層有 KV，會讓「context → memory」的語意變得混亂；VLM |
| Gemma 4 | 研究時的淘汰理由「first-party GGUF 只有一種 QAT q4_0」在改採本地量化後已**不成立**。仍成立的疑慮：Gemma 系列（Gemma 2 / 3 已確認）採用 local sliding-window 與 global attention 交錯，KV 不隨 context 線性成長，會汙染 context → memory 的語意；Gemma 4 的架構與 size 組合**未逐一查核**。若要改選 Gemma 4，需要另開 decision 重查 |
| Ministral 3 8B / 14B | VLM；YaRN factor 16 永遠開著（原生 16k），context 軸會被 RoPE scaling 汙染 |
| Granite 4.2 8b / 30b | size 組合不對；30B 的 Q4_K_M 已有 17.7 GB，5070 Ti 上沒有邊界可看 |
| Llama 3.x | 「沒有 Meta 官方 GGUF」在本地量化下不再是理由。仍成立的理由：沒有約 14B 的 size（3.1 為 8B / 70B，3.2 為 1B / 3B），湊不出 8B / 14B 的組合；license 另有命名 / 標示條款 |
| Phi-4 | 只有 14B，沒有同 family 的約 8B sibling |
| 官方 Qwen GGUF | Q4_K_M 的 recipe 不同（AWQ-compatible），見上 |
| imatrix 量化 | 多一個 calibration 資料變因 |
| Q5_K_M / Q4_0 / Q3_K / IQ 系列 | Q5_K_M 與 Q6_K、Q4_K_M 太接近；Q4_0 是較舊的格式；Q3 以下與 IQ 系列需要 imatrix 才有合理品質；BF16 本身是 reference，不是 treatment |
| context 含 40960 或更長 | 超出原生 32k，需要 YaRN |

## 對實驗的影響

- **兩平台、quality、performance 用的必須是同一批 sha256 的檔案。** RTX 端可以拷貝後驗證 sha256，
  也可以在 WSL 重跑同一 pipeline 再比對 hash（順便檢查可重現性）。
- Qwen3 是 hybrid thinking 模型。Core A 的效能量測用 `llama-bench` 的 token 序列，quality 用 log-likelihood
  scoring，兩者都不經過 chat template，thinking mode 不影響。fit run 的 prompt 是固定 token 序列的文字檔，
  以 `-no-cnv` 送入，同樣不套 template。
- 磁碟（一次只處理一個 size）：14B 的 safetensors 29.5 GB + BF16 GGUF 29.5 GB + 3 個 quant 36.8 GB + KLD logits 約 45 GB，
  峰值約 141 GB；目前可用約 190 GiB。safetensors 轉換完、驗證 hash 後即可刪除（可依 revision 重新下載）。
- 換 family / size / quant / context 任一項，都要寫新的 decision log；pre-registration 之後再換，已收的數字全部作廢。
