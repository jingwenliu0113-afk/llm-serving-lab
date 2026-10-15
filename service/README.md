# capserve：Capacity-Aware Serving Service（MVP，M0）

capserve 是 vLLM 0.30.0 OpenAI server 前面的一層薄 gateway（decision 027 §3–§5、029）。
它不寫 engine 或 scheduler，也不修改 vLLM；推論一律由 vLLM 執行。

- OpenAI 相容介面：`/v1/chat/completions`、`/v1/completions`（串流與非串流）、`/v1/models`。
  只接受允許清單內的欄位，每個 request 只接受一個 prompt；其他欄位回 400 `unsupported_parameter`。
- 准入控制：FIFO、並發上限、有界佇列、排隊逾時；超載回 429、無法服務回 503，都帶 `Retry-After`。
- `/healthz`、`/readyz`、`/metrics`（Prometheus，只供營運，不是研究數據）。
- 上限來自 capacity profile（`profiles/`）中的 `active_class`。

## Provisional 的數值不是產品限制

- 目前**沒有任何驗證過的容量**。只有由 Stage C 正式、有效、凍結的分析經產生器產生的 `verified` profile，才能成為產品限制（027 §4）。
- `profiles/provisional-dev.json` 的上限（並發 4、佇列 4）是手動設定的開發值，只供開發與假後端的整合測試。
  它**不是容量**，不得當成產品限制引用，也不是 Core 的 C_knee。
- provisional profile 只有在 `--allow-provisional` 或 `CAPSERVE_ALLOW_PROVISIONAL=1` 時才能載入；
  這時啟動訊息、`/readyz`、`/metrics`（`capserve_profile_verified 0`）與每個 response 的 `x-capserve-profile` header 都標示 provisional。
- `python -m capserve.profile_gen` 在 `configs/capacity.toml` 的 `[freeze] capacity` 為 true 之前一律 exit 2；
  真正的產生器是 M2 的工作。

## 與 vLLM 預設不同之處

- 沒有給 `max_tokens` 時，補上 active class 的 `max_output_tokens`（vLLM 的預設是 context 剩下的長度）；超過時回 400。
- Qwen3.5 的 chat 預設會 thinking；output 上限小時，多數 chat 回答會停在 `<think>` 之中。
  是否開放 `chat_template_kwargs.enable_thinking` 是待決的產品決定，MVP 不開放。

## 執行

```bash
# 測試（Mac，只用假後端；不啟動 vLLM）
PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH .venv/bin/python -m pytest -q service/tests tests/test_research_isolation.py

# gateway（vLLM 已在 127.0.0.1:8100 執行時）
cat > /tmp/gw.toml <<'EOF'
active_class = "dev-2k"
backend_url = "http://127.0.0.1:8100"
EOF
PYTHONPATH=service:benchmarks .venv/bin/python -m capserve serve --config /tmp/gw.toml \
    --profile service/profiles/provisional-dev.json --allow-provisional
```

- 設定檔的欄位與預設值見 `capserve/config.py`。`CAPSERVE_API_KEY` 只從環境變數讀取；沒有設定時只能綁 loopback。
- 單一 process，uvicorn 固定 `http="h11"`、`loop="asyncio"`；SIGTERM 時先 drain，exit −15 為正常結束。
- 部署管理（`deploy`、`procs`、`runs`、`loadtest`）與 RTX 整合測試是 M1。
