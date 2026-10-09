# Local LLM Serving Lab

模型大小、量化、context、並發如何改變**品質、延遲、吞吐量與記憶體**的 frontier？什麼條件下該用什麼配置？

這是一個事先登記（pre-registered）的實驗：hypothesis、門檻、metric 與 protocol 都在看到正式數據之前寫定並凍結，
之後的任何變更都有 decision log。結論只來自有效的正式 session，失敗、OOM、not testable 的單位照實報告。

> **狀態（2026-10-10）**：正式數據收集中。
> - 已完成：RTX Core A（`formal-20261009-rtx-a1`）。
> - 進行中：Mac Core A、Core B uniform。
> - 排隊中：Core B mixed、quality。
>
> 結果段落會在全部正式分析完成後補上。

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

尚未產生。全部正式分析完成後，這裡會放：

- Core A：quality vs throughput、context vs memory、fit boundary、latency。
- Core B：concurrency vs TTFT / TPOT、throughput、P95。
- `results/decision_table/decision_table.md`：什麼 workload 適合什麼配置，以及證據不支持的部分。
