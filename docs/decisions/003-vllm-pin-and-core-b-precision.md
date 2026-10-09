# 003 — vLLM 版本 pin 與 Core B 固定 precision

> **2026-10-08：Core B 的模型 / precision / dtype 已由 008 取代**（改為 `RedHatAI/Qwen3.5-9B-quantized.w4a16`、`--dtype bfloat16`）。vLLM 版本與其餘 server protocol 仍有效；hybrid 模型需要的新 flag 見 008。下方內容保留作為決策歷史。

> **2026-10-09：`--max-num-seqs` 由 264 改為 128，grid 上限由 256 改為 128（019，使用者決定）。**
> **2026-10-09：smoke test gate 第 1–6 項全部通過，vLLM 環境改為 Accepted（結果見 PREREGISTRATION §6）。**

- 狀態：**Core B 模型已由使用者固定為 `Qwen3-8B-AWQ`（2026-10-08）**；vLLM 環境待 RTX 5070 Ti smoke test 通過後改為 Accepted
- 日期：2026-10-08
- 對應：SPEC Day 1 #3「固定 vLLM 版本」、SPEC Core B「固定一種 precision」
- 機器可讀：`configs/pins.toml` `[vllm]`、`[vllm.deps]`、`[core_b]`

## 背景

Core B 只在 RTX 5070 Ti（sm_120、16 GB）的 WSL2 內跑 vLLM。PyPI 最近的 release 依序是
0.29.0（09-09）、0.30.0（09-22）、0.31.0（10-05）。

16 GB 放不下 Qwen3-8B BF16（權重 15.26 GiB），所以 Core B 的 precision 必須在「量化的 8B」與「較小的 BF16 sibling」之間做選擇。
目前所有判斷都來自 source / PyPI / HF metadata 與 wheel 內容，**沒有任何一項在實機跑過**。

## 決策

### vLLM 版本

- **`vllm==0.30.0`**：git tag commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；
  wheel `vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl`，sha256 `ef52ee58…6ac4`。
- 安裝在 WSL2 內獨立的 Python 3.12 venv。依 v0.30.0 CI lock 固定：
  `torch==2.13.0`（PyPI 版為 CUDA 13.0 build）、`transformers==5.16.1`、`tokenizers==0.23.1`、
  `huggingface-hub==1.31.0`、`triton==3.7.1`。
  - 只寫 `pip install vllm==0.30.0` 不可重現，因為 transitive deps（例如 nvcc / nvjitlink 13.x）會隨安裝日期浮動。
  - 做法：以 `uv pip compile --exclude-newer 2026-09-23` 解出完整 lock，smoke test 通過後連同 hash 凍結進
    `configs/`。
  - 不使用 `--torch-backend=auto`，它會依偵測到的 driver 自動挑 CUDA variant。
- Windows host driver 必須 ≥ R580（CUDA 13.x 的要求）。2026-10-08 實測為 617.42 ✅（001）。
- 為什麼是 0.30.0：它是「已發布 ≥ 2 週」條件下最新的 release（本專案 heuristic，不是 SPEC 規則），
  沒有 patch / .post 版，研究查核也沒有找到影響「單 GPU、dense 8B」路徑的 0.30.0 專屬回報。
  0.29.0 更舊，沒有任何對本專案有利、而 0.30.0 沒有的地方。
- 為什麼不選 0.31.0：它比 0.30.0 主要多的是 #57355 修正（`max_num_seqs` 非 8 的倍數時的 throughput cliff），
  以及 transformers 上限；兩者分別用「`max_num_seqs` 取 8 的倍數」和「lock」就能避開。
  等 smoke test 真正能跑時，若 0.31.x 已穩定，可以另開 decision 重新評估。

### Core B precision：`Qwen/Qwen3-8B-AWQ`（INT4 weight-only，W4A16，group 128）

- revision `4da05a8edb55c6046cce958586c33b61da07bb79`，權重 6,098,581,864 B（5.68 GiB），`--dtype float16`。
- 由 vLLM 的 Marlin kernel 執行。0.30.0 wheel 內 Marlin 有原生 `12.0f` 編譯，不依賴 DeepGEMM 或 FlashInfer JIT。
- 與 Core A 同一個 8B base：AWQ repo 的 `model-00002-of-00002.safetensors` 與 `Qwen/Qwen3-8B` 的
  `model-00005-of-00005.safetensors` sha256 完全相同（同一份 lm_head）。
- `--kv-cache-dtype auto`（f16 KV）。**KV quantization 屬 Advanced，Core 階段不開。**

| 候選 | 權重 | 估計 KV 容量\* | 主要風險 / 缺點 |
|---|---|---|---|
| **Qwen3-8B-AWQ (W4A16)** ← 選定 | 5.68 GiB | ~6.3–6.8 GiB ≈ 46–49k tokens | 4-bit 品質低於 FP8；prefill 為 dequant / compute-bound |
| Qwen3-8B-FP8（官方 block FP8 128×128） | 8.79 GiB | ~3.3–3.8 GiB ≈ 24–27k tokens | sm_120 上 DeepGEMM 會被優先選用，沒有 toolkit 時 crash（#60261，post-0.30 nightly 回報）；KV 空間最小 |
| RedHatAI/Qwen3-8B-FP8-dynamic（per-channel） | ~8.8 GiB | 同上 | 第三方 checkpoint；KV 空間小 |
| Qwen3-4B BF16（未量化） | 7.49 GiB | ~4.5–5.0 GiB ≈ 33–36k tokens | 4B 不在 Core A 的 size 集合內，無法對應 Core A 結果；原為 fallback，已依使用者決定移除 |
| Qwen3-8B BF16 | 15.26 GiB | — | 放不下 |

\* 假設 `gpu_memory_utilization` 0.85 × 15.92 GiB（16303 MiB，2026-10-08 已實測）、非 KV overhead 1–1.5 GiB、
f16 KV 144 KiB/token。**全部是估計值**，以實機 log 的 `GPU KV cache size: N tokens` 為準。

**選 AWQ 的理由：**

1. 與 Core A 的 8B 出自**同一個 base checkpoint**。注意這只是 lineage 相同：AWQ W4A16 與 Core A 的
   GGUF quant 是不同格式，decision table 裡 Core B 列的 quality 一律標「未量測（AWQ W4A16）」，
   **不得**沿用 Core A Q4_K_M 的 quality 數字。
2. KV 空間最大。Core B 的核心變因是 KV cache pressure，空間越大，可掃描的 concurrency × context 範圍越寬。
3. kernel 路徑在 sm_120 上最不依賴 JIT。

**使用者決定（2026-10-08）：Core B 固定使用 `Qwen3-8B-AWQ`，不保留 fallback，不再擴大模型選擇矩陣。**
理由是 RTX 5070 Ti 16 GB 需要保留 KV cache / concurrency 空間。原本列為 fallback 的 Qwen3-4B BF16 因此移除。
若 AWQ 在 smoke test 失敗，記為 blocker 並回報使用者，不自動改用其他模型。

## 為什麼

- **vLLM 0.30.0**：在「已發布 ≥ 2 週」的條件下最新、沒有已知影響 dense 單 GPU 路徑的回報，CI lock 也可直接用來凍結依賴
  （細節見上方「vLLM 版本」）。
- **AWQ W4A16**：在 16 GB 上留給 KV 的空間最大（Core B 的核心變因就是 KV pressure），kernel 路徑最不依賴 sm_120 上的 JIT，
  而且與 Core A 的 8B 出自同一個 base checkpoint。
- **把 server 的預設值全部寫明**：vLLM 預設開啟的 prefix caching、chunked prefill 與 batching 參數，都屬於 SPEC 的 Advanced 變因；
  Core 必須把它們固定並記錄，否則 frontier 會混入這些機制的效果。

## Core B server / client protocol 草案（smoke test 後於 Stage B addendum 凍結）

| 項目 | 設定 | 理由 |
|---|---|---|
| prefix caching | `--no-enable-prefix-caching` | vLLM 預設開啟，但 prefix caching 是 SPEC Advanced 變因；而且固定 seed 的 random dataset 重跑時會被 cache 灌水 |
| chunked prefill | 保持預設 on，但明確寫出 | batching policy 屬 Advanced，Core 固定一種 |
| `--max-num-batched-tokens` | `2048`（16 GB GPU 的預設值，明確寫出） | 同上 |
| `--max-num-seqs` | 264（8 的倍數，且大於 grid 上限 256；007）〔2026-10-09 起依 019 改為 128，與 grid 上限 128 相等；仍是 8 的倍數〕 | 避開 #57355；確保 knee 不是 scheduler 上限造成的〔019：vLLM 0.30.0 在 FULL CUDA graph 加上 Mamba 層時要求 max_num_seqs ≤ num_gpu_blocks〕 |
| performance mode / async scheduling | `--performance-mode balanced`（0.30.0 預設）與 async scheduling 的實際生效值，都明確寫入啟動指令，並從 engine config log 驗證 | 兩者都屬 batching / scheduler policy，Core 必須固定 |
| `--max-model-len` | **32768，所有 level 與 mix 共用同一個 server 設定**（007） | 讓 request 留在原生 32k window 內；level 之間只差在 request 長度，不會改變 scheduler / CUDA graph 的上限。預設 40960 只有在 KV 池小於 5.625 GiB（一條 40960-token 序列）時才會啟動失敗；以 AWQ 的估計 KV 池不會，但不能依賴預設值 |
| KV 池大小 | `--kv-cache-memory-bytes` 固定值；每次 run 記錄 log 的 `GPU KV cache size: N tokens` | 冷 / 熱 compile cache 會讓 KV 池相差約 5%（#54383）；先做一次丟棄用的 boot |
| `--gpu-memory-utilization` | 依實測 Windows idle VRAM baseline 設定（預設 0.92 很可能啟動失敗） | free VRAM < util × total 時 vLLM 拒絕啟動 |
| attention backend | `--attention-backend FLASH_ATTN` 明確指定，並記錄 "Using X attention backend" | 0.30.0 的 FA2 在 sm_120 走 compute_80 PTX JIT；不能讓它默默退回其他 backend |
| sampling | `--generation-config vllm`；client 端 `--temperature 0 --ignore-eos` | 不讓模型自帶的 generation_config 改變 sampling |
| WSL2 pinned memory | `VLLM_WSL2_ENABLE_PIN_MEMORY`：smoke test 時 A/B 一次後凍結（選擇規則見下） | 正反證據都有：#58849 回報開啟後約 +12%；envs.py 只說因 small performance regression 預設關閉，「RTX 5080 上 <2%」出自 PR #41496 的 review（review 更正出處） |
| 離線 | `HF_HUB_OFFLINE=1`，tokenizer 指向 pinned revision 的本地 snapshot | `vllm bench serve` 沒有 tokenizer revision 參數 |
| GPU lock | 正式 run，**以及**所有會產生 protocol 參數的步驟（A/B、idle baseline、KV 池大小設定）和刻意耗盡 VRAM 的步驟（004 的超量測試），一律 `flock -n /tmp/ai-portfolio-gpu.lock …`（WSL 內）；拿不到就中止，不搶鎖 | MACHINE_POLICY；否則 Text-to-SQL 同時在跑時，A/B 與 baseline 會被汙染，超量測試也可能害對方 OOM |
| client | `vllm bench serve --backend vllm --endpoint /v1/completions …`，`--save-result --save-detailed` | 同一個 install。request mix 為 007 的兩個離散 mix：`uniform` 用 `--dataset-name random --random-range-ratio 0`；`mixed` 用固定 seed 產生的自訂 dataset（記錄 sha256）。實際長度以 `--save-detailed` 的 input_lens 驗證 |

**GPU memory metric 的定義**：vLLM 依 utilization 或 kv-cache-memory-bytes 預先配置記憶體，`nvidia-smi` 的
memory.used 在設計上幾乎是常數。因此「隨負載變化的記憶體訊號」改用 Prometheus 的 KV cache usage 百分比與
preemption 次數；`nvidia-smi` device 記憶體只作為 baseline 紀錄。精確的 metric 名稱要在實機 `/metrics` 上確認。

**`VLLM_WSL2_ENABLE_PIN_MEMORY` 的選擇規則（事先寫定）**：預設 0（vLLM 預設）。只有在 A/B 中，`1` 在 C=1 與
C=16 兩點的 output throughput 都比 `0` 高 ≥ 5%，且 run-to-run CV < 3% 時才改為 1。結果寫入 Stage B addendum。

## Smoke test gate（遠端可用後，全部通過才把 003 改為 Accepted）

GPU lock 依 MACHINE_POLICY：步驟 1–2 是小型啟動 smoke test，**不需要** GPU lock。
但 vLLM 啟動時會預先占用約 `gpu_memory_utilization` 比例的 VRAM，所以啟動前要先看 `nvidia-smi`：
若另有 GPU process 在用，就不啟動，改做 Mac 端工作。
步驟 3–6 屬於 throughput 類重工作業，或會產生 protocol 數字，**必須**在 `flock -n /tmp/ai-portfolio-gpu.lock` 下執行。

1. `torch.cuda.get_device_capability() == (12, 0)`，且 `get_arch_list()` 含 sm_120。
2. 模型可以啟動並回應。2026-10-08 起依 008：模型改為 Qwen3.5-9B W4A16，`--language-model-only`，
   並依 **2K → 8K → 32K 分段**（008「Core B smoke test」），每段只送 1 個 request，前一段通過才進下一段。
3. 記錄 FA2 PTX JIT 的首次啟動時間；第二次 boot 時 JIT cache 有生效。
4. `VLLM_WSL2_ENABLE_PIN_MEMORY` 做 0/1 A/B 一次。
5. `vllm bench serve` 能跑完並產出 JSON；在 `/metrics` 確認 KV usage 與 preemption 的 metric 名稱。
6. 依 PREREGISTRATION §4 量 Windows idle VRAM baseline，並記錄螢幕是否接在這張卡上。

**Gate 進度（2026-10-09）**：
- 第 1 項通過：專案 venv 中 torch 2.13.0+cu130 的 capability 為 (12, 0)，arch list 含 sm_120。
- 第 2 項通過：008 的 S1 → S2 → S3 全部通過，見 008 補充。
- 第 5 項通過（2026-10-09，`results/stage_b/20261009/bench-check`）：
  - 在 008 的 S1 smoke server 上跑一次 `vllm bench serve`（uniform，L = 2048，C = 4，200 個 request），JSON 正常寫出。
    run_core_b 的驗證全部通過：completed 200、failed 0，input 全為 1920，output 全為 128。
  - `/metrics` 有 `vllm:kv_cache_usage_perc`、`vllm:num_preemptions_total`、`vllm:cache_config_info`。
  - 這是工具檢查，不是量測。
- 第 6 項的 idle baseline 已量（B = 811 MiB，見 Stage B 結果）。
- 第 3–4 項依 017 執行，但被 `--max-num-seqs 264` 的啟動問題擋住，見 NEXT 的 Blocked 2。
  〔2026-10-09：019 改為 `--max-num-seqs 128` 後解除；第 3–4 項以新限制繼續，結果見 PREREGISTRATION §6。〕

## 放棄的替代方案

見上表，以及：使用 vLLM 的 cu129 Docker image（#59157，torch 版本錯誤）、原生 Windows（vLLM 不支援）。

## 對實驗的影響

- Core B 的數字只代表「Qwen3-8B-AWQ / vLLM 0.30.0 / RTX 5070 Ti / WSL2」。
- Core B 不量 quality，依據是 SPEC Core B 的 metrics 清單裡沒有 quality。
- **不能**與 Core A 的 GGUF quant（Q8_0 / Q6_K / Q4_K_M）混為一談（EXPERIMENT_RULES）。
- Core B 的 env 紀錄一律用 `env_capture.py --vllm-python <vLLM venv 的 python>`，才會記到 vLLM / torch / CUDA runtime 版本。
- Core B 不設 fallback；AWQ 若無法在 RTX 上運作，視為 blocker，由使用者決定下一步。
