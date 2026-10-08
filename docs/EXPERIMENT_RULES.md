# Local LLM Serving Lab 實驗規範

## Pre-registration

正式 benchmark 前先 commit：

- hypothesis
- metrics
- model hash
- quantization set
- context set
- quality task set
- prompt
- sampling settings
- runtime version
- benchmark harness version

## Core A

- quantization 只在 Core A sweep。
- quality 每個 model × quantization 量一次。
- Mac / RTX 不重複算 quality。
- 正式 benchmark 時避免其他重負載。

## Core B

- 固定一種 precision。
- 不把 Core A 的 quantization format 與 Core B 混為一談。
- 記錄 concurrency、request-length distribution、KV cache 壓力。

## Benchmark Environment

每次記：

- git commit
- OS
- runtime version
- CUDA / driver
- power state
- background load
- benchmark config

## 報告

- 不只報 tokens/sec。
- 同時報 quality / latency / memory / throughput。
- 不只放圖，必須有 decision-oriented conclusion。
