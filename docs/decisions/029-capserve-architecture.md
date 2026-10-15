# 029 — Engineering：capserve（Capacity-Aware Serving Service）MVP 架構

- 狀態：Accepted（架構；實作依本文以 TDD 進行）
- 日期：2026-10-15
- 對應：decision 027 §3–§5
- 時點：任何服務程式之前。設計經過一輪設計與一輪對抗式審查（審查改正了 20 項，見文末）。

## 定位

- capserve 是 vLLM 0.30.0 OpenAI server 前面的一層薄 gateway。它不寫 engine 或 scheduler，也不修改 vLLM。
- 相依套件只用 `configs/vllm.lock.txt` 中已 hash 鎖定的版本（`service` dependency group，`test_service_deps` 把關）。
  WSL 上直接以唯讀方式使用研究的 venv。
- 研究的 harness 與分析程式**不得** import capserve（`tests/test_research_isolation.py` 以 AST 檢查 `benchmarks/`、`scripts/`、`quality/`）。
  capserve 可以唯讀 import 研究模組，例如 `core_b_plan.server_argv`、`run_adv.held_gpu_lock`。

## 目錄

```text
service/
  README.md               # provisional 的數值不是產品限制（027 §4）
  capserve/
    __main__.py           # python -m capserve serve --config F --profile P [--allow-provisional]
    config.py             # 設定（TOML）；環境變數只讀 CAPSERVE_API_KEY、CAPSERVE_ALLOW_PROVISIONAL
    profile.py            # capacity profile 的 schema、canonical hash、server fingerprint、loader
    admission.py          # FIFO 准入、有界佇列、drain
    openai_api.py         # 允許清單式的 request model、OpenAI 格式的錯誤
    tokens.py             # prompt 長度上限：byte 快速路徑 + vLLM /tokenize
    upstream.py           # 資料面的 httpx client、BackendMonitor（自己的 client）
    proxy.py              # 非串流 proxy、SSEFramer、ProxyStream
    guard.py              # 最外層的純 ASGI middleware：request id、pending 上限、cleanup、access log
    metrics.py            # 獨立的 CollectorRegistry
    app.py                # create_app()：Starlette routes + lifespan（不用 FastAPI 的 docs 與預設錯誤）
    server.py             # uvicorn.Server 子類別；http="h11"、loop="asyncio"、單一 process
    procs.py              # start_group（傳遞 lock fd、PDEATHSIG）、vllm_env()
    deploy.py             # render | up | status | stop
    runs.py               # run 目錄只能在 <service_root>/runs 之下
    loadtest.py           # 產品的負載測試（不是研究量測）
    profile_gen.py        # verified profile 的產生器（freeze.capacity 之前一律拒絕）
  profiles/provisional-dev.json, provisional-rtx.json
  deploy/rtx.toml
  tests/fakes/{fake_vllm.py, servers.py}, tests/test_*.py
```

pyproject：testpaths 加入 `service/tests`，pythonpath 加入 `service`。

## Request 流程（`/v1/chat/completions`、`/v1/completions`）

1. **Guard**（純 ASGI，最外層）：
   - 指定 `x-request-id`。
   - pending ≥ max_pending 時，在讀 body 之前就回 429 `queue_full`。
     - pending 從到達算到 response 結束。
     - max_pending = max_concurrency + max_queue + pre_admission_slots（預設 16）。
   - health 與 metrics 不受限制。
2. **Auth**：選用的 bearer key，以 bytes 用 `hmac.compare_digest` 比較。
3. **Gate**：draining、backend 未就緒、profile 不允許時回 503。
4. **Body**：最多 1 MiB（否則 413），讀取逾時 10 s（否則 408）。
5. **驗證**：
   - 允許清單之外、或不是預設值的欄位，回 400 `unsupported_parameter`；
   - `model` 必須等於 served id，否則回 404；
   - 只接受一個 prompt；
   - 限制或補上 `max_tokens`。
6. **byte 快速路徑**：byte 數的上界已在 max_prompt_tokens 之內時不必 tokenize。byte-level BPE 每個 byte 最多一個 token。
7. **准入**：FIFO 取得 permit，同時監看斷線與 deadline。取得後立即在 ASGI scope 上登記 permit 的 cleanup。
8. **Tokenize**（只在需要時）：持有 permit 時才呼叫 `/tokenize`。超過上限回 400 `context_length_exceeded` 並釋放。
   這樣送進 vLLM 的 HTTP request 總數永遠 ≤ C，和研究的 `--max-concurrency` 語意一致。
9. **Upstream**：送出，非串流直接轉傳，串流則逐 event 轉傳。
10. **Cleanup**：在 shielded 的 `finally` 中釋放 permit（idempotent）、關閉 upstream、記錄 metrics，並寫一行 JSON access log。
    log 中不記錄 prompt 或 completion 的文字。

## OpenAI 相容介面

- **Routes**：`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/healthz`、`/readyz`、`/metrics`。
  - 沒有 catch-all passthrough，所以 vLLM 的 `/tokenize` 與管理端點都不對外。
  - `/v1/models` 只回 served id。
- **允許清單（MVP）**：
  - 共用欄位：model、stream、stream_options、max_tokens、temperature、top_p、stop、seed、presence_penalty、frequency_penalty、user，以及 n（只能是 1）。
  - Chat 另外接受：messages（只接受文字）、max_completion_tokens。
  - Completions 另外接受：prompt（字串、token id 列表，或只有一個元素的列表）、ignore_eos、min_tokens（≤ max_tokens）。
  - 其他一律回 400，包括 tools、response_format / structured outputs、logprobs、echo、best_of、beam search、chat_template、truncate_prompt_tokens。
    原因：這些都會改變送進 vLLM 的負載或序列數，而研究沒有量過。
- **`max_tokens`**：沒有給時，補上該 class 的 max_output_tokens（和 vLLM 的預設不同，會寫在文件中）；超過時回 400。
- **轉送**：
  - body 重新序列化後送給 vLLM；不轉送 client 的 Authorization。
  - vLLM 只綁 loopback，而且不設 `--api-key`。
- **錯誤**：
  - 格式為 `{"error":{"message","type","param","code"}}`。
  - 429 `rate_limit_error`（queue_full、queue_timeout）與 503 `server_error`（backend_unavailable、draining）都帶 `Retry-After`、`retry-after-ms` 與 `x-should-retry: true`。
    Retry-After 依原因取設定中的固定值，MVP 不做 adaptive。
  - upstream 連線失敗回 502，非串流超過 deadline 回 504。
  - upstream 的非 200 狀態碼保留。
  - 每個 response 都帶 `x-request-id` 與 `x-capserve-profile: <id>; status=<verified|provisional>`。
  - 不複製 hop-by-hop header。
- **非串流**：
  - 在 `fail_after(剩餘 deadline)` 內執行，另有每 250 ms 檢查一次斷線的 watcher。
  - client 斷線時取消 httpx request 並關閉連線，vLLM 就會 abort 該 request（RTX 整合測試 IT-3 驗證）。
- **串流**：
  - upstream 回 200 時，用 `ProxyStream` 逐 event 轉傳。
  - deadline 只套在每次 upstream 讀取上，不跨越 `yield`。
  - `SSEFramer` 只轉送完整的 event。
  - 第一個有內容的 event 記錄 gateway TTFT 與 upstream TTFT。
  - upstream 中斷或逾時時，送出一個 error event，再送 `[DONE]`。
  - cleanup 由 shielded 的 `finally` 依序執行：`body_iterator.aclose()`、`upstream.aclose()`、釋放 permit。
  - 慢的 client 會暫停寫入，進而暫停讀取 upstream（backpressure）。
- **Auth**：
  - `CAPSERVE_API_KEY` 只從環境變數讀取，不存檔、不寫 log，並從 vLLM 的環境中移除。
  - 沒有設 key 時，拒絕綁定非 loopback 的位址。

## 准入控制與超載保護

- `AdmissionController(max_concurrency, max_queue, max_queue_wait_s, retry_after_s, clock)`：
  - deque 加 future，嚴格 FIFO；只有在 `in_flight < max` 而且沒有人在等時才走快速路徑。
  - 等待時間上限為 min(max_queue_wait_s, deadline − now)。
  - 等待者被取消或逾時、但它的 future 已經拿到 permit 時，呼叫 `release()` 把位置交給下一位。permit 不會遺失。
- **429 與 503**：
  - 429：服務健康，但已達設定的容量（pending 上限、佇列已滿、排隊逾時）。
  - 503：目前無法服務（backend 未就緒、draining）。這類 request 不進佇列；backend 掉線時，正在排隊的 request 也一律回 503。
- **Deadline**：`request_deadline_s`（預設 300），從到達起算，涵蓋排隊、tokenize 與生成。
- **與 workload class 的對應**：
  - MVP 的設定只指定一個 `active_class`。它的 max_concurrency、max_queue 是全域值，它的 token 上限是硬性的 400。
  - 依 class 或依 token 加權的准入，要等有驗證過的 mixed 容量之後才做。原因：各 class 的 uniform 容量無法相加；
    2K 在 mixed 負載下，短 request 的 P95 TTFT 從 4.26 s 變 23.2 s（README H-B4）。
  - max_queue_wait_s（預設 2 s）與 max_queue 是產品政策，不是研究結果。
- **Drain**：
  - `server.py` 的 `handle_exit` 先設為 draining，再呼叫 `super().handle_exit`。
  - uvicorn 設定：`http="h11"`、`loop="asyncio"`、`timeout_graceful_shutdown=drain_timeout_s`、`limit_concurrency=2*max_pending+16`。
  - 永遠只有一個 process，不提供 workers 選項，否則 process 內的限制會被乘上 worker 數。
  - SIGTERM 時 listener 立即關閉；進行中的 request 完成或逾時取消；exit −15 視為正常。

## Capacity profile（JSON，pydantic `extra="forbid"`）

```json
{"schema_version": 1, "profile_id": "verified-<sid>", "status": "verified", "provisional_reason": null,
 "slo": {"id": "<§8 的 id>", "terms": {"<§8 原文>": 0}},
 "server": {"vllm_version": "0.30.0", "model_repo": "RedHatAI/Qwen3.5-9B-quantized.w4a16", "model_revision": "<pin>",
            "fingerprint_sha256": "<64 hex>", "num_gpu_blocks": 248, "block_size": 528},
 "classes": [{"name": "2k", "max_prompt_tokens": 1920, "max_output_tokens": 128,
              "max_concurrency": 1, "max_queue": 1, "source": "stage_c:L=2048:uniform"}],
 "provenance": {"source_session": "formal-...", "analysis_path": "results/capacity/<sid>/analysis.json",
                "analysis_sha256": "<64 hex>", "prereg_section": "§8", "freeze_commit": "<sha>", "generator_commit": "<sha>"},
 "content_sha256": "<去掉本欄位後 canonical JSON 的 sha256>"}
```

**Loader 的規則**
- content_sha256 必須相符，class 名稱不得重複。
- max_prompt_tokens + max_output_tokens ≤ 32768；1 ≤ max_concurrency ≤ 128；max_queue ≥ 0。
- **verified** 需要：完整的 provenance、64 位十六進位的 hash、`formal-` 開頭的 session，而且 provisional_reason 為 null。
- **provisional** 必須寫明原因。
- provisional 只有在 `--allow-provisional` 或 `CAPSERVE_ALLOW_PROVISIONAL=1` 時才能載入。
  這時 `/readyz`、metrics 與 response header 都標示 provisional。
- vLLM 版本與模型欄位必須等於 pins。

**Server fingerprint**
- 是下列兩者的 sha256：
  - argv[1:] 去掉 --host / --port；
  - `core_b_plan.recorded_env`。
- deploy 時，下列任一不符就拒絕使用 verified profile：
  - fingerprint；
  - vLLM `/metrics` 的 block 數或 block size；
  - log facts（FLASH_ATTN、GDN prefill kernel、MNBT 2048）。

**產生器**（`python -m capserve.profile_gen`）
- 沒有任何覆寫參數（021 R1）。
- 在 `configs/capacity.toml` 的 `[freeze] capacity` 為 true 之前一律 exit 2（檔案不存在也視為 false）。
- §8 凍結後：
  - 要求 session 為 formal 且 env 有效；
  - 用凍結的 Stage C 模組重算 analysis，並要求完全相同（021 R3）；
  - 依 §8 的規則把每個 L 對應到 class；
  - 不可檢定的 class 一律拒絕，不給數字，也不退回其他數字；
  - max_queue = max_concurrency（產品政策）。
- Mac 測試：每個 commit 過的 verified profile 都要能由產生器重新產生，而且相同（忽略 generator_commit）。

**provisional profile 的數值**
- provisional profile 中的上限只供開發與整合測試，不是產品限制。
- 目前沒有任何驗證過的容量，所以 `provisional-rtx.json` 刻意用保守的小值（例如並發 4）。
  文件中寫明「未驗證，不是容量」，也不引用 Core 的 C_knee。

## Health、readiness 與 metrics

- `/healthz`：event loop 活著就回 200。
- `/readyz`：profile 已載入且允許、backend monitor 就緒、不在 draining 時才回 200。
  body 列出 profile id、status、backend 狀態、in_flight、queued、pending 與各項上限。
- **BackendMonitor**：
  - 用自己的 2 條連線、2 s 逾時探測 `/health`；
  - 連續 3 次失敗才判定掉線；
  - `/v1/models` 只在啟動與恢復時檢查。
- **`/metrics`**：
  - 獨立的 registry，呼叫 `disable_created_metrics()`。
  - 這些數字只供營運使用，永遠不是研究證據。
  - 項目：

    | 種類 | 名稱 |
    |---|---|
    | counter | requests_total（route、stream、outcome）、rejections_total（reason）、client_errors_total（code）、upstream_errors_total（kind）、client_disconnects_total（phase） |
    | histogram | queue_wait_seconds、ttft_seconds（從到達起算，含排隊）、upstream_ttft_seconds（從送出 upstream 起算，和研究的 TTFT 定義相同）、e2e_seconds |
    | gauge | in_flight、queued、pending、concurrency_limit、queue_limit、backend_ready、profile_verified |
    | info | profile_info（profile_id、status、slo_id、source_session、analysis_sha256、active_class） |

## 部署管理（`python -m capserve.deploy`）

- **`render`**（Mac 也可執行）：印出 vLLM argv（由 `core_b_plan.server_argv` 以 Stage B 的值產生，port 8100）、recorded_env 與 fingerprint。
- **`up`**：前景的 supervisor，因為 WSL 會殺掉 detached 的 job。依序執行：
  1. 驗證 release 的 DEPLOY_MANIFEST，載入 pins 與 profile。
  2. **preflight**：
     - `run_adv.held_gpu_lock()` 為真（驗證 fd 確實持有 lock，不只信任旗標）；
     - GPU memory ≤ IDLE_MIB；
     - toolchain 沒有問題；
     - port 8100 與 8080 都空著；
     - run 目錄在 `service_root/runs` 之下、在研究目錄之外；
     - verified profile 的 fingerprint 相符。
  3. 建立 `runs/pt-<UTC>/meta.json`，內容為 `purpose: product-test`、`formal: false`、`research_valid: false`。
  4. 啟動 vLLM：
     - 用 `start_group`，把 lock 的 fd 傳給子 process（pass_fds），並設 PR_SET_PDEATHSIG。
       這樣 supervisor 即使被 SIGKILL，lock 也會一直持有到 vLLM 結束，避免 GPU 上還有 process 時 lock 卻已釋放。
     - vLLM 的環境由清理過的 base 產生（去掉 PYTHONPATH、PYTHONDONTWRITEBYTECODE、CAPSERVE_*），並設 `HOME=<service_root>/.home`、`TMPDIR=<service_root>/.tmp`。
       這樣產品的 JIT、autotune 快取不會寫進研究共用的 `~/.triton`、`~/.cache/{vllm,flashinfer}`，以免成為 C0 / C1 的干擾因素。
     - 等待 `/health`，再核對 cache config 與 log facts。
  5. 啟動 gateway，等待 `/readyz`，寫入 `state.json`。
  6. 執行 `capserve.loadtest <scenario>`（固定的 module，不接受任意指令），或持續運行，直到收到 signal、達到 `max_hold_s`（預設 3600 s，限制持有共享 lock 的時間），或任一子 process 結束。
  7. **teardown**（`finally`，在 `run_adv.DeferredSignals` 下）：
     - 先 SIGTERM gateway，再關閉 vLLM；
     - 確認 GPU memory 回到 IDLE_MIB 以下；
     - 問題寫進 meta.json。
- **WSL 的部署目錄**：`~/llm-serving-service/releases/<sha>/`，每個 release 一個新目錄，不覆寫。
  - 解壓也在 GPU lock 下進行，所以不會和正式 session 重疊。
  - 研究目錄 `~/llm-serving-lab`、它的 DEPLOYED_COMMIT 與 venv 都不寫入，也不安裝任何東西。
  - 產品的 vLLM 用 port 8100，和研究的 8000 分開。
- 任何產品 flag 的變更都會改變 fingerprint，需要另外的決定。

## 測試

**Mac：假的 vLLM 後端**
- 實作：Starlette app，可設定 TTFT、ITL、token 數、錯誤狀態、在第幾個 chunk 中斷、event 切割、hang、健康狀態。
- 記錄 `peak_in_flight` 與被取消的 id，並提供 asyncio Event 讓測試同步，不依賴 wall clock。
- test server 用 uvicorn，`http="h11"`、`loop="asyncio"`，和正式環境相同。

**單元測試**
- profile、admission（不超過 N、FIFO、取消時不遺失 permit、idempotent、drain）、guard、OpenAI 介面、SSEFramer、tokens、metrics 名稱、upstream 的 client 設定。

**整合測試（假後端）**
- 串流與非串流；
- 斷線時取消 upstream 並釋放位置；
- 超載：上限 4、佇列 4、40 個並發，peak_in_flight ≤ 4，每個 429 都帶 Retry-After；
- readiness；
- deploy 的 render、env 清理，以及 preflight 拒絕；
- 研究隔離（AST）。

**RTX 整合測試**（只是產品測試；GPU lock 下、ssh 前景；每次最多 1 h；輸出到 `~/llm-serving-service/runs/`）

| 測試 | 內容 |
|---|---|
| IT-0 | import 檢查；唯讀確認 vLLM 0.30.0 的 `/tokenize` 介面與 keep-alive 預設值。不需要 lock，不啟動 vLLM。 |
| IT-1 | 啟動、readiness、chat 與 completion 的串流與非串流、`/v1/models`。確認 vLLM 的 `/proc/<pid>/fd` 持有 lock。 |
| IT-2 | 3 倍超載：vLLM 的 running + waiting 永遠不超過上限；只有 429，沒有 5xx。 |
| IT-3 | 串流與非串流的斷線：vLLM 的 running + waiting 在 2 s 內下降。 |
| IT-4 | `/tokenize` 一致性：chat 的額外 token 數在設定的常數之內。 |
| IT-5 | 負載下停止：drain，GPU memory 回到 IDLE_MIB。 |
| IT-6 | 研究隔離：`find ~/llm-serving-lab ~/.cache/{vllm,flashinfer,huggingface} ~/.triton -newer runs/<id>/meta.json` 為空。 |

- **產品負載測試**（`capserve.loadtest`）：
  - 情境：smoke、steady、burst、overload。
  - 輸出到 `runs/pt-<UTC>-<scenario>/`，標示 `purpose: product-test`、`research_valid: false`，並註明「不是容量量測」。
  - `runs.py` 拒絕 `results/` 或研究目錄之下的路徑，也拒絕 `formal-` 開頭的 id；沒有 `--formal` 選項。

## 不做（MVP）

- 不寫新的 engine 或 scheduler，不加 Core argv 之外的 vLLM flag 或 patch。
- 不做多模型、LoRA、多 GPU、autoscaling、mixed-class 准入、優先順序、per-key quota。
- 不支援 tools、structured outputs、logprobs、embeddings、Responses API。
- 不做 TLS（之後交給 reverse proxy），不做常駐 daemon。常駐會無限期占用共享 lock，需要使用者決定。
- 不做 dashboard，不在 Mac 上跑 Docker 或 vLLM，不用付費 API。
- 不從產品測試推導任何容量結論。

## 里程碑

- **M0（Mac，TDD）**：profile、admission、guard、api、tokens、upstream、proxy、metrics、app、server、假後端與測試、provisional-dev profile。
- **M1**：deploy、procs、runs、loadtest；在研究 session 之間於 RTX 上跑 IT-0 至 IT-6。
- **M2**（§8 凍結並有有效的 C1 analysis 之後）：
  - 實作真正的產生器，commit verified profile；
  - production 模式不帶旗標也能啟動；
  - README 只引用 verified profile。
- **v1**：多 class 或 token 加權的准入（要有驗證過的 mixed 容量）、in-process tokenizer、adaptive Retry-After、quota、TLS proxy。

## 待決（產品，非研究）

- Qwen3.5 的 chat 預設會 thinking；MVP 預設 128 個 output token 時，多數 chat 回答會停在 `<think>` 之中。
  是否開放 `chat_template_kwargs.enable_thinking` 是產品決定，MVP 先不開放，並在 README 中說明。
- 未驗證的假設照實寫明：
  - 同一 class 內較短或長短不一的 prompt，在 C* 下也安全；
  - gateway 的 `/tokenize` 流量不在研究的負載之中。

## 設計審查改正的項目（摘要）

- 准入之前沒有上限：在讀 body 之前加上全域 pending 上限、byte 快速路徑，並只在持有 permit 時呼叫 `/tokenize`。
- 批次 prompt 與未量過的 vLLM 擴充欄位會突破並發上限：改為允許清單，只接受一個 prompt。
- Mac 與 WSL 的 HTTP stack 不同（WSL 會自動選 httptools / uvloop）：兩邊都明確指定 h11 / asyncio。
- httpx 的預設值（5 s 逾時、100 條連線、5 s keep-alive、trust_env）會破壞 proxy：改為明確設定。
- 產品的 vLLM 會繼承 PYTHONPATH，並把 JIT 快取寫進研究共用的目錄：改為清理過的環境，加上獨立的 HOME / TMPDIR，並加入 IT-6。
- supervisor 被殺時 lock 會提早釋放：把 lock 的 fd 傳給子 process，並設 PDEATHSIG。
- permit 遺失與 deadline 跨越 yield：在 scope 上登記 cleanup，deadline 只套在每次讀取上。
- SSE event 被切半：加入 SSEFramer。
- 其他：BackendMonitor 要連續 3 次失敗才判定掉線、IT-2 改看 running + waiting、drain 的語意、禁止多 worker、
  不用 FastAPI 的預設介面、metrics 的命名、`/tokenize` 一致性、release 目錄不覆寫、profile 的完整性檢查、
  bytes 比較 API key、hop-by-hop header、thinking 預設的待決項目。

## 實作補記（M0，2026-10-16）

M0 依本文以 TDD 實作，經兩個面向的審查（12 項，全部確認、11 項修正、1 項延到 M2）。以下是與本文不同、或本文沒有寫到的決定：

- **刪除 uvicorn 的 `limit_concurrency`**（本文原寫 `2*max_pending+16`）：
  - 它數的是連線，不是 request，閒置的 keep-alive socket 與從不送 request 的 socket 都算在內。
  - 達到上限時，uvicorn 會在 Guard 之前對所有 request 回自己的純文字 503：沒有 Retry-After，也沒有 x-request-id。
  - 審查實測：上限 4、佇列 4 時，64 個同時的 client 全部拿到 503。
  - 改為 `None`，只用 Guard 的 pending 上限。測試涵蓋超過舊上限的同時連線與閒置 socket。
- **串流的寫入也受 deadline 限制**：
  - 原本 deadline 只套在讀 upstream 上。client 連著但不讀時，寫入會無限期阻塞，permit 也一直被占住。
  - 現在每次寫入都受剩餘 deadline 限制。scope 只包住 send，不跨越 yield。
- **`/tokenize` 的 4xx 原樣轉傳**（和生成 request 的「upstream 非 200 保留」一致），只有 5xx 或回應沒有 count 時才回 502。
- **讀完並驗證 body 之後、准入之前，再檢查一次 draining 與 backend 狀態**：讀 body 最多 10 s，這段期間 backend 可能掉線。
- **串流中的錯誤事件**：串流開始後，vLLM 以 `data: {"error": {...}}` 回報錯誤。這類事件照樣轉給 client，但記為 upstream_errors 的 `stream_error`。
  vLLM 0.30 的格式要在 RTX 的 IT-1 確認。
- **會造成 500 的輸入一律改回 400**：NaN、±Infinity、1e999、lone surrogate、巢狀過深的 JSON。
- **cleanup 順序**：client 斷線時，不提前釋放 permit，而由 LIFO cleanup 依序先關 upstream、再釋放，所以 vLLM 中的 request 永遠 ≤ C。
- **`profile_gen` 讀 `configs/capacity.toml` 的 `[freeze] capacity`**（030 實作補記），不讀 pins。
- **verified profile 的治理**：
  - M0 用「每個 commit 過的 verified profile 都要能由產生器重新產生」的測試把關。產生器在凍結前一律 exit 2，所以 M0 不可能有 verified profile。
  - 「載入器在執行期驗證 provenance」延到 M2，和真正的產生器一起做。
- **其他**：
  - 測試用的 server 跑在測試自己的 event loop 中，但使用正式的 `server.build_config`（h11 / asyncio）。
  - `/v1/models` 只回 served id。
  - chat 的輸出上限一律以 `max_completion_tokens` 轉送。
  - uvicorn 固定 `ws="none"`。
  - outcome 新增 `server_error`；client 已斷線的 request 在 log 中記為 499。
  - 斷線 watcher 另看一個停止旗標，修正了 Starlette `is_disconnected` 可能吃掉 cancel 而卡住的問題（有回歸測試）。
  - 沒有加入 openai SDK 的測試群組，因為 M0 不安裝新套件。
- **IT-0 / IT-4 要確認**：
  - vLLM 0.30 的 `/tokenize` 是否接受 `add_generation_prompt` 與 `add_special_tokens`；
  - chat 是否接受 `max_completion_tokens`；
  - keep-alive 的實際值；
  - tokens 的額外常數（chat 32、per_message 16、completion 1）目前是暫定的寬鬆值。

