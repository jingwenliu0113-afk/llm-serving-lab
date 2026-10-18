# Local LLM Serving Lab SPEC

## 一句話問題

模型大小、量化、context、並發如何改變品質、延遲、吞吐量與記憶體 frontier？什麼條件下該用什麼配置？

---

# Core A：Single-user Local Inference

目標不是比較哪台電腦比較快。

目標是研究不同 memory / compute architecture 下的部署 frontier。

## Runtime

使用同一跨平台 runtime，例如 llama.cpp。

鎖定：

- runtime commit / version
- model hash
- compiler
- macOS / CUDA 環境
- 執行 flags

## 變因

同一 model family：

- 2 種 parameter size
- 3 種 quantization
- 3 種 context length

平台：

- M4 Pro 48GB
- RTX 5070 Ti 16GB

## Metrics

- model fit / OOM boundary
- TTFT
- generation speed
- peak memory
- quality

## Quality

Quality 是「model × quantization」屬性。

不需要在兩台硬體重複量。

至少：

- perplexity 或類似語言模型品質指標
- 一組固定 downstream task set

Quality task set 一旦正式實驗開始就凍結。

---

# Core B：Multi-user Serving

只在 NVIDIA 桌機。

使用真正為 online serving 設計的 engine，例如 vLLM。

Core B 固定一種 precision。

## 變因

- concurrency
- context length
- request mix
- KV cache pressure

## Metrics

- TTFT
- TPOT
- P50
- P95
- requests/sec
- throughput
- GPU memory

核心問題：

> 負載升高時，latency-throughput frontier 如何變化？

---

# Advanced

只有 Core 暴露出具體 bottleneck / trade-off 時才進。

不論結果是否符合原本預期。

可選：

- KV cache quantization
- batching policy
- request-length distribution
- prefix caching
- fixed SLO 下最大 concurrency

目標是隔離：

- KV cache
- batching
- allocator
- scheduler

哪個是主要成因。

最後整理 deployment decision table。

---

# Optional

- TensorRT-LLM
- speculative decoding
- larger models
- energy / power measurement
- serving cost model
- multi-GPU

---

# 不做什麼

- 不寫新的 vLLM。
- 不把 Mac / RTX 做排名。
- 不做 runtime 大亂鬥。
- 不跑沒有 hypothesis 的巨大 grid。

---

# GitHub 最終成果

Core A：

- quality vs throughput
- context vs memory
- fit boundary
- latency

Core B：

- concurrency vs TTFT / TPOT
- throughput
- P95
- VRAM

最後：

- deployment decision table
- 清楚說明什麼 workload 適合什麼配置

---

# Repo 建議結構

```text
llm-serving-lab/
├── CLAUDE.md
├── SPEC.md
├── NEXT.md
├── configs/
├── benchmarks/
├── quality/
├── results/
├── scripts/
├── tests/
└── docs/
    ├── EXPERIMENT_RULES.md
    └── decisions/
```

---

# Day 1

1. 決定桌機 OS / 執行環境。
2. 固定 llama.cpp 版本。
3. 固定 vLLM 版本。
4. 確認 NVIDIA driver / CUDA / framework 正常。
5. 選 model family。
6. 寫下第一版 hypothesis、metrics、task set。

---

# Week 1

Core A 先做：

1. Mac 與 RTX 兩邊都能跑同一 model / runtime。
2. 建 benchmark harness。
3. 固定輸出格式。
4. 跑小矩陣 smoke benchmark。
5. 固定 quality task set。
6. 產出第一張 memory / latency 圖。

Core B 只做到：

- vLLM 能啟動
- benchmark client 能打 request
- metrics 能被收集

先不要全量 sweep。

---

# Gate

## 進 Advanced

Core 暴露出：

- 明確 bottleneck
- 或明確 trade-off
- 而且可重現

即使符合預期也可進。

## 收尾

若 Core 結果非常乾淨、沒有值得隔離的機制：

- 寫 deployment decision table
- 收尾

---

# Phase 2（2026-10-15 起，使用者決定；decision 027；2026-10-19 結案，decision 036）

上文是 Phase 1（Core + Advanced）的 SPEC，原文不改。Phase 1 的結果與判定都保留，不重新分析。
Phase 2 分成兩條線，可以並行開發；正式 GPU 實驗必須獨占共享 GPU lock。

## Research：Stage C（Capacity）

- C0：8K 延遲雙模式的有限度排查（探索性；開始前另寫 decision log，事先寫定 boot 數、GPU 時數上限與停止規則）。
- C1：Fixed-SLO 下的最大並發容量（正式）。SLO、負載、指標、判定規則與容量的定義，都在任何正式數據之前寫進 PREREGISTRATION §8 並凍結。
  SLO 的數值由使用者選定。
- 平台、模型與 runtime 沿用 Core B（RTX 5070 Ti、vLLM 0.30.0、Qwen3.5-9B W4A16）。

## Engineering：Capacity-Aware LLM Serving Service（MVP）

- 沿用 vLLM，不開發新的推論引擎、不修改 vLLM。
- 功能：OpenAI-compatible API、部署管理、health check、metrics、超載保護，以及依研究結果制定的容量配置。
- 只有由 Stage C 正式、有效、凍結的分析產生的容量（verified）才能成為正式的產品限制；其他一律標為 provisional。
- 產品測試的輸出不進 `results/`，不得污染研究數據。

## 不做什麼（Phase 2）

- 不寫新的推論引擎或 scheduler。
- 不擴張研究範圍：Stage C 只做 C0 與 C1。其他 Future Work（026）仍要使用者另外決定。
- 不使用付費 API；不影響其他專案；GitHub 維持 private。

## 結案（2026-10-19，decision 036）

- **Research**：C0 為分支 A（032）。C1 的最終結果見 035：C_PROD 為 uniform 2K 32、uniform 8K 8、mixed 2K / 8K 2。
- **Engineering**：capserve MVP 完成。正式設定 `service/deploy/rtx-verified.toml` 只服務 verified profile `verified-formal-20261017-c1b.json`。
- 適用條件、IT-3 的限制與後續項目見 036。後續的產品化或研究由使用者另外決定。
