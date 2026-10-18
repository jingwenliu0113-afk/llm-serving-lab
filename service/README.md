# capserve：Capacity-Aware Serving Service（MVP）

capserve 是 vLLM 0.30.0 OpenAI server 前面的一層薄 gateway（decision 027 §3–§5、029）。
它不寫 engine 或 scheduler，也不修改 vLLM；推論一律由 vLLM 執行。

- OpenAI 相容介面：`/v1/chat/completions`、`/v1/completions`（串流與非串流）、`/v1/models`。
  只接受允許清單內的欄位，每個 request 只接受一個 prompt；其他欄位回 400 `unsupported_parameter`。
- 准入控制：FIFO、並發上限、有界佇列、排隊逾時；超載回 429、無法服務回 503，都帶 `Retry-After`。
- `/healthz`、`/readyz`、`/metrics`（Prometheus，只供營運，不是研究數據）。
- 上限來自 capacity profile（`profiles/`）中的 `active_class`。

## 正式成果與適用條件（decision 035 / 036）

- 只有由 Stage C 正式、有效、凍結的分析經產生器產生的 `verified` profile，才能成為產品限制（027 §4）。
- 正式 profile：`profiles/verified-formal-20261017-c1b.json`（decision 035），四個 class 都是 verified、exact：

  | class | prompt 上限 | 輸出上限 | max_concurrency = max_queue |
  |---|---|---|---|
  | uniform-2k | 1,920 | 128 | 32 |
  | uniform-8k | 8,064 | 128 | 8 |
  | mixed-2k | 32,640 | 128 | 2 |
  | mixed-8k | 32,640 | 128 | 2 |

- **適用條件**（完整列表見 decision 036）：
  - RTX 5070 Ti 16 GB 上的 WSL2；vLLM 0.30.0；Qwen3.5-9B W4A16，Core B 的 server argv（每次 boot 以 fingerprint 核對）；
  - SLO S2：短類 P95 TTFT 2 / 6 s、P95 TPOT 100 ms、mixed 的 32K request ≤ 20 s；
  - 封閉迴路，輸出 128 token，TTFT 從准入後起算；每個關鍵點 5 次 boot，都在 5% guard band 內。
- **不涵蓋**：
  - 開放迴路或 Poisson 到達、其他輸出長度；
  - 准入前的排隊；
  - 開場湧入：C 個 request 同時到達時，uniform 第一波的 TTFT 最高 10.4 s（2K）與 11.0 s（8K），遠高於 SLO。

## 正式部署與保護

- 正式部署只用正式設定：

  ```bash
  python -m capserve.deploy up --config service/deploy/rtx-verified.toml \
      --profile service/profiles/verified-formal-20261017-c1b.json
  ```

  - 在 GPU lock 下執行，active class 為 uniform-2k。
- `rtx-verified.toml` 以 `[deploy] production_profile` 釘住這份 verified profile（036）。`capserve.deploy up`（包括 `--loadtest`）與 `capserve.itest` 會在啟動任何東西之前拒絕：
  - 其他任何 `--profile`，包括另一份 verified profile；
  - 任何允許 provisional 的執行：`--allow-provisional`，或 shell 中留著的 `CAPSERVE_ALLOW_PROVISIONAL=1`。
- provisional profile 不得使用 Stage C 的 class 名稱（uniform-2k、uniform-8k、mixed-2k、mixed-8k），id 也不得以 `verified-` 開頭。
- unavailable 或 verified 0 的 class 不能作為 active class。
- `python -m capserve serve` 是 gateway 的低階進入點，供開發使用，不讀部署設定；正式部署不直接用它。
- `rtx.toml` 是整合測試設定，使用 provisional-rtx，沒有釘選。

## 整合測試的結果與限制（029 最後的補記、036）

| 測試 | verified profile（`rtx-verified.toml`） | provisional-rtx（`rtx.toml`，同一 release） |
|---|---|---|
| IT-0、IT-1、IT-2、IT-4、IT-5、IT-6 | 通過 | 通過 |
| IT-3（client 斷線） | **無法判定，不算通過** | 通過（決定性） |

- verified profile 的輸出上限是 128 token。斷線時只剩約 1.0–1.4 s 的生成量，少於 IT-3 要求的 4 s，所以無法區分「被 abort」和「本來就跑完」。
  - 第一次執行記為不通過。斷線後生成確實停止、vLLM 的 running 與 waiting 都回到 0，但這不構成決定性的證據。
- 決定性的斷線驗證只來自 provisional-rtx（1,024 token 輸出上限）。斷線處理和 profile 無關，所以推論 verified 設定下也成立，但在 128 token 下沒有直接證據。
- 以 verified profile 跑 IT 時用：

  ```bash
  python -m capserve.itest IT-0 IT-1 IT-2 IT-4 IT-5 IT-6 --config service/deploy/rtx-verified.toml
  ```

  - 沒有 `--profile` 時，預設就是釘住的 profile。

## Provisional 的數值不是產品限制

- `profiles/provisional-dev.json` 的上限（並發 4、佇列 4）是手動設定的開發值，只供開發與假後端的整合測試。
  `profiles/provisional-rtx.json`（並發 4、佇列 4、輸出 1,024 token）只供 RTX 整合測試（`rtx.toml`）。
  它**不是容量**，不得當成產品限制引用，也不是 Core 的 C_knee。
- provisional profile 只有在 `--allow-provisional` 或 `CAPSERVE_ALLOW_PROVISIONAL=1` 時才能載入；
  這時啟動訊息、`/readyz`、`/metrics`（`capserve_profile_verified 0`）與每個 response 的 `x-capserve-profile` header 都標示 provisional。
- `python -m capserve.profile_gen` 在 `configs/capacity.toml` 的 `[freeze] capacity` 為 true 之前一律 exit 2（見下一節）。

## Verified profile（M2 的產生器）

```bash
PYTHONPATH=service:benchmarks .venv/bin/python -m capserve.profile_gen \
    --session results/capacity/formal-YYYYMMDD-c1 --freeze-commit <設定 freeze.capacity 的 commit，40 hex> \
    --out service/profiles/verified-formal-YYYYMMDD-c1.json
```

- 沒有任何覆寫參數。`--freeze-commit` 會以 git 驗證：
  - 該 commit 的 `capacity.toml` 已凍結，它的上一個 commit、以及它之前任何一個 commit 都沒有凍結過（先解凍、修改、再凍結不算 030 的凍結）；
  - 它是 session DEPLOYED_COMMIT 的祖先；
  - 凍結的位元組就是實際執行的位元組：該 commit 的 `capacity.toml` 的 sha256 等於 session 記錄、產生器讀到的值；
    Stage C 模組（`capacity_plan`、`capacity_analysis`、`run_capacity`、`capacity_report`、`gpu_idle`）在該 commit、在 session 的 DEPLOYED_COMMIT、
    以及產生器實際 import 的檔案三者相同。
- 拒絕（exit 2、不寫任何檔案）的情況：沒有凍結或凍結證據不全；session 不是 `results/capacity/formal-…`、不是 formal / env 有效 / C1，或 `capacity_report` 會拒絕它；
  以凍結的 Stage C 模組重跑 `capacity_report` 無法逐位元組重現 `analysis.json`；SLO 不是 `capacity.toml` 的；session 的 server 不是產品的 server；
  工作目錄有未 commit 的變更（generator_commit 必須是產生它的程式）；`--out` 已存在或位於 `results/` 之下。
- 輸出是決定性的（沒有時間戳記）；每個 commit 過的 verified profile 都由 Mac 測試重新產生並比對（忽略 generator_commit）。
  這個測試掃描所有 tracked 檔案（`results/` 之外），任何路徑、任何副檔名中 `status` 為 `verified` 的 JSON 都要能重新產生。
- **載入時的檢查**（`capserve serve` 與 `deploy up` 都一樣）：`configs/capacity.toml` 必須已凍結；provenance 的
  `capacity_sha256`、`pins_sha256` 必須等於這個 checkout（release）的 `configs/capacity.toml`、`configs/pins.toml`；
  `slo` 必須逐字等於 `capacity.toml` 的 `[slo]`；`profile_id` 必須是 `verified-<source_session>`，`analysis_path` 必須是
  `results/capacity/<source_session>/analysis.json`；每個 class 的名稱與 token 上限必須是它的 Stage C workload 的（下表），每個 workload 至多一個 class。
- **只服務 `service/profiles/<name>.json`**：`deploy up` 只接受 release 中 `service/profiles/` 最上層、小寫 `.json`、列在 `DEPLOY_MANIFEST.json` 的檔案；
  `capserve serve` 只接受位元組等於這個 checkout 的某個 `service/profiles/*.json` 的 verified profile（deploy 傳給它的 `runs/<id>/profile.json` 是這種複本）。
  數字本身不在執行期重算，而是由上面的 Mac 重新產生測試保證。
- 每個 Stage C workload 對應一個 class（輸出上限 128 token）：

  | class | workload | max_prompt_tokens |
  |---|---|---|
  | `uniform-2k` | U2K | 1,920 |
  | `uniform-8k` | U8K | 8,064 |
  | `mixed-2k` | M2K | 32,640（007 的 mix：90% 1,920、10% 32,640） |
  | `mixed-8k` | M8K | 32,640（007 的 mix：90% 8,064、10% 32,640） |

- class 的狀態依 C_PROD（030 8.7）：

  | C_PROD | class | gateway |
  |---|---|---|
  | EXACT / LOWER / CENSORED | `verified`，max_concurrency = max_queue = C_PROD，`bound` 為 exact / lower / censored | 正常服務 |
  | INFEASIBLE | `verified`，0（拒絕這個 class），bound exact | 啟動，但每個 completion request 都回 503 `class_not_serviceable`（不排隊、沒有 `Retry-After`、`x-should-retry: false`）；`/readyz` 為 503，`not_ready_reason` 為 `class_not_serviceable`；`deploy up` 在 preflight 拒絕（沒有可服務的東西，不啟動 vLLM） |
  | UNSTABLE / NT | `unavailable`，附原因，沒有任何數字 | 拒絕啟動（exit 2），任何旗標都不能覆寫；也不退回其他數字 |

- 每個 class 另附 Stage C 的證據（只作描述，不是上限）：適用範圍、C_PROD、C_SLO、C_KV、C_cap、旗標（例如 ABOVE_CKV）、先失守的項目、C_PROD 的餘裕與開場湧入。
  `/readyz` 另外列出 `class_status`、`bound`、`serviceable`。

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
- 部署管理與整合測試（M1，decision 029 的 M1 實作補記）：
  - `python -m capserve.deploy render|up|status|stop`：在 RTX 上以 GPU lock 啟動 vLLM 與 gateway（前景 supervisor），teardown 一定執行。
  - `python -m capserve.loadtest`：產品負載測試（smoke / steady / burst / overload），輸出到 `runs/`，**不是容量量測**。
  - `python -m capserve.itest IT-0 … IT-6`：RTX 整合測試；除了 IT-0 都需要 GPU lock。
  - `tools/rtx/deploy_service.sh`：把 release 部署到 WSL 的 `~/llm-serving-service/releases/<sha>`，不寫入研究目錄。
  - RTX 上的閒置判定依 decision 031 的產品規格：GPU lock、使用率 ≤ 5%、基線穩定；閒置用量高於研究的餘裕上限只發警告，高於 2,048 MiB 則拒絕（保護其他專案）。
