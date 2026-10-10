# Local LLM Serving Lab

模型大小、量化、context、並發如何改變**品質、延遲、吞吐量與記憶體**的 frontier？什麼條件下該用什麼配置？

這是一個事先登記（pre-registered）的實驗：hypothesis、門檻、metric 與 protocol 都在看到正式數據之前寫定並凍結，
之後的任何變更都有 decision log。結論只來自有效的正式 session，失敗、OOM、not testable 的單位照實報告。

> **狀態（2026-10-10）**：Core A（兩個平台）與 Core B 的正式數據已完成，結果見下方。
> 正式 quality 進行中；之後產生 deployment decision table。

## 研究範圍

| | Core A：single-user local inference | Core B：multi-user serving |
|---|---|---|
| Runtime | llama.cpp b11146（兩平台同一 commit） | vLLM 0.30.0 |
| 模型 | Qwen3.5 4B / 9B GGUF，Q8_0 / Q6_K / Q4_K_M（BF16 只當 quality 參考） | RedHatAI/Qwen3.5-9B-quantized.w4a16（固定一種 precision） |
| 變因 | size × quant × context（2K / 8K / 32K） | concurrency × context × request mix |
| 平台 | M4 Pro 48 GB（Metal）、RTX 5070 Ti 16 GB（WSL2 CUDA） | RTX 5070 Ti 16 GB |
| Metrics | fit / OOM / spill、TTFT、decode tok/s、peak memory、quality | TTFT、TPOT、P50 / P95、req/s、throughput、KV cache 使用率 |

- **不做**：Mac 與 PC 的排名、runtime 大亂鬥、沒有 hypothesis 的巨大 grid、自己寫 serving engine。
- **Quality**（decision 005、020）：
  - 量測項目：WikiText-2 的 PPL / KLD（對同 size 的 BF16），以及 HellaSwag / MMLU / ARC / WinoGrande 子集（描述性）。
  - 只在 wsl2-cuda 上量，視為 model × quant 的屬性，`-c 512`。

## Hypotheses（凍結於 [PREREGISTRATION](docs/PREREGISTRATION.md) §2）

- **H-A1** fit boundary 由記憶體容量決定。
- **H-A2** decode 受記憶體頻寬限制（±20%）。
- **H-A3** context 的 decode 成本隨 KV 占比增加。
- **H-A4** prefill 受算力限制（(a) quant 之間的差距、(b) 32K / 8K 的 TTFT 比值 > 4）。
- **H-A5** quality–memory frontier（9B-Q4_K_M 對 4B-Q8_0 的配對 ΔNLL；同 size 的 KLD 單調遞增；Holm 校正）。
- **H-B1** knee 之後延遲急升。
- **H-B2** KV 容量決定 knee。
- **H-B3** TPOT 的成長慢於 TTFT。
- **H-B4** request mix 提高短 request 的尾延遲。

## 文件

- [SPEC.md](SPEC.md)：研究問題與範圍。
- [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)：hypothesis、metric、protocol。
  分兩階段凍結：Stage A `eb6eef1`，Stage B `19446d7`。
- [docs/decisions/](docs/decisions/)：每個設計決策的 decision log（001–021）。
  - 021 是 deployment decision table 的推導規則，在其餘正式分析之前寫定。
- [docs/EXPERIMENT_RULES.md](docs/EXPERIMENT_RULES.md)、[docs/RESULTS_FORMAT.md](docs/RESULTS_FORMAT.md)、[docs/STORAGE_PLAN.md](docs/STORAGE_PLAN.md)。
- [MACHINE_POLICY.md](MACHINE_POLICY.md)：兩台機器的分工與 lock。
- [configs/pins.toml](configs/pins.toml)：所有 pin。
  - 內容：runtime commit、模型 sha256、Stage B 實測值、vLLM lock。
  - 凍結狀態由測試強制。

## 目錄

```text
benchmarks/   Core A / Core B 的 plan、runner、凍結的分析程式、報告、圖、決策表
quality/      quality 的 plan、runner、分析（H-A5）、資料子集
scripts/      llama.cpp build、GGUF 產生、env capture、WSL 部署與傳輸
configs/      pins.toml 與各 lock 檔
results/      每個 session 的 plan / rows / env / analysis（raw/ 不進 git）
tests/        pytest（stdlib only 的分析程式都有測試）
tools/        with-mac-lock（Mac 重負載 lock）
```

## 重現

前提：`uv`。Mac 上 llama.cpp 用 Metal build；RTX 在 WSL2 Ubuntu 22.04 上用 user-space CUDA 13.4.1（decision 015）。
正式 run 一律要拿到 lock：Mac 用 `tools/with-mac-lock`，RTX 用 `flock -n /tmp/ai-portfolio-gpu.lock`。

```bash
uv sync && uv run pytest -q
```

1. **llama.cpp 與 GGUF**：
   - `scripts/build_llama_cpp.sh metal|cuda`
   - `uv run python scripts/produce_gguf.py produce --size 4B|9B`（本地從官方 BF16 產生，不用 imatrix；sha256 寫入 pins）
2. **Core A**（每個平台一個正式 session，seed 20261008）：
   - `benchmarks/core_a_plan.py`
   - `benchmarks/run_core_a.py`
   - `benchmarks/core_a_report.py`：產生 `analysis.json`；RTX 要帶 Stage B 的 V_total / B / C_ctx。
   - `benchmarks/core_a_figures.py`
3. **Core B**（WSL，GPU lock）：
   - `benchmarks/core_b_plan.py uniform`（讀 pins 的 Stage B 值）
   - `benchmarks/run_core_b.py --formal`
   - `core_b_plan.py mixed --uniform-session …`
   - `run_core_b.py --mixed-dataset …`
   - `benchmarks/core_b_report.py`
   - `benchmarks/core_b_figures.py`
4. **Quality**（wsl2-cuda，GPU lock）：
   - `quality/run_quality.py plan --platform wsl2-cuda` 與 `run --gpu-lock-held`
   - `quality/quality_analysis.py`
5. **Deployment decision table**：`uv run python benchmarks/decision_table.py`
   - 沒有任何選擇 session 的參數。
   - 每類輸入要恰好一個有效的正式 session；每份 analysis 都會用凍結程式重算比對。
   - 輸出 `results/decision_table/`。

各指令的完整參數見檔案開頭的 docstring。

## 結果

> Core A（兩個平台）與 Core B 的正式數據已完成。quality（正式 run 進行中）與 deployment decision table 完成後補上。
> 判定都由事先凍結的分析程式產生（`analysis.json`）；"not refuted" 只代表沒有被推翻，不代表被證實。
> 依 SPEC，不做 Mac 與 PC 的排名：下面的數字只在同一平台內比較。

### Core A：single-user（llama.cpp b11146）

| session | 平台 | 狀態 |
|---|---|---|
| `formal-20261009-a2` | macos-metal（M4 Pro 48 GB） | 164 / 164 ok，sentinel 漂移 1.78%，env 有效 |
| `formal-20261009-rtx-a1` | wsl2-cuda（RTX 5070 Ti 16 GB） | 164 / 164 ok，sentinel 漂移 0.06%，env 有效 |

| hypothesis | macos-metal | wsl2-cuda |
|---|---|---|
| H-A1 fit boundary | not refuted：18 / 18 fit，沒有 spill | not refuted：18 / 18 fit，沒有 spill |
| H-A2 decode 受頻寬限制（±20%） | **refuted**：13 / 15 組偏差為負，最大 −25.8% | **refuted**：15 / 15 組偏差為負，最大 −25.1% |
| H-A3 context 的 decode 降幅 | not refuted：順序符合預測，相鄰差距在 noise 內 | not refuted：順序符合預測，差距超出 noise |
| H-A4(a) prefill 的 quant 差距小於 decode | not refuted | not refuted |
| H-A4(b) TTFT(32K) ÷ TTFT(8K) > 4 | not refuted（4.49–5.11） | not refuted（4.39–4.91） |

- **fit**：2K–32K 範圍內兩個平台都沒有碰到記憶體邊界，與 H-A1 的事前預測一致（16 GB 上 18 格都預測 fit）。
- **decode（H-A2）**：bytes 較少的配置確實較快，但都比純頻寬模型預測的慢。
  - 兩個平台上偏差最大的都是同一組：4B-Q4_K_M 對 9B-Q8_0，觀測比約 2.2，預測比 2.99。
  - 原因 Core 沒有檢驗。未檢驗的推測：有一部分成本與權重 bytes 無關，例如 GDN recurrent 狀態與 kernel 開銷。
- **context**：decode 從 2K 到 32K 的降幅，bit 越低越大。
  - macos-metal：4B 0.25–0.28，9B 0.16–0.23。
  - wsl2-cuda：4B 0.16–0.22，9B 0.12–0.15。
- **prefill**：
  - 同一 (size, L) 下三個 quant 的 prefill 差距（macos-metal 1.04–1.07，wsl2-cuda 1.16–1.31），都小於 decode 差距（1.26–1.48）。
  - TTFT 隨 context 超線性成長。
- 圖：`results/core_a/figures/{decode,prefill,memory}-{mac,rtx}-formal.svg`。

### Core B：multi-user serving（vLLM 0.30.0，Qwen3.5-9B W4A16，RTX 5070 Ti）

sessions：`formal-20261009-coreb-uniform`（102 / 102 ok）、`formal-20261010-coreb-mixed2`（6 / 6 ok）。
第一次 mixed 因 client 缺 pandas 全部失敗、沒有送出 request，保留為 `formal-20261010-coreb-mixed`（decision 022）。

| L | C_KV | C_knee | output tok/s：C_knee / grid 最大 | P95 TTFT（ms）：C = 1 / C_knee / C_max | G_TPOT / G_TTFT |
|---|---|---|---|---|---|
| 2K | 35 | 25 | 314.9 / 325.7（C 32） | 330 / 4,260 / 38,138（C_max 128） | 8.6 / 115.7 |
| 8K | 13 | 8 | 80.2 / 83.3（C 26） | 1,407 / 2,834 / 179,098（C_max 128） | 11.8 / 127.3 |
| 32K | 3 | 2 | 17.8 / 18.6（C 32） | 6,290 / 9,988 / 207,100（C_max 32） | 8.6 / 32.9 |

- **H-B1 not refuted**：三個 L 在 knee 之後 P95 TTFT 都達到 knee 的 3 倍（首次達到的 C 分別是 70 / 16 / 8）。
- **H-B2 not refuted**：C_knee 都落在 [C_KV ÷ 2, 2 × C_KV] 內，與「KV 容量（含 GDN 狀態 block）決定 knee」的預測一致；這不是因果檢驗。
- **H-B3 not refuted**：從 C = 1 到 C_max，P95 TPOT 只增加 8.6–11.8 倍，P95 TTFT 增加 33–127 倍。
- **H-B4 not refuted**：在 C = C_knee 時加入 10% 的長 request（32K），短 request 的 P95 TTFT：
  - 2K：4.26 → 23.2 秒（+444%）。
  - 8K：2.83 → 7.80 秒（+175%）。
  - 兩者都超出 noise。mixed 下 KV cache 使用率峰值 100%，沒有 preemption。機制（chunked prefill 干擾）是 conjecture。
- 圖：`results/core_b/figures/{p95_ttft_ms,p95_tpot_ms,output_throughput}-uniform-formal.svg`。

### Quality 與 deployment decision table

- 正式 quality 在 wsl2-cuda 上進行中（兩個 size，約 14 小時；decision 020）。
- 完成後由 `benchmarks/decision_table.py` 依 decision 021 產生 `results/decision_table/decision_table.md`。
  內容包括：什麼 workload 適合什麼配置，以及證據不支持的部分。
