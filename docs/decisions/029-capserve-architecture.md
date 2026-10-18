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


## 實作補記（M1，2026-10-16；尚未在 RTX 上執行）

M1（`procs`、`runs`、`deploy`、`loadtest`、`itest`、`service/deploy/rtx.toml`、`profiles/provisional-rtx.json`、`tools/rtx/deploy_service.sh`）依本文以 TDD 實作，
經兩個面向的審查：11 項發現，其中 2 項重複，9 項全部修正，每項都有回歸測試。與本文不同、或本文沒有寫到的決定：

- **`state.json` 從開始就寫入**，依序為 starting → ready → stopping → stopped，所以開機期間也能 `stop`。
- **teardown 一定執行**：teardown 前後對 `state.json`、`meta.json` 的寫入都只是盡力而為。寫入失敗（例如磁碟滿）只記成問題、exit 1，不能跳過 teardown，也不能蓋掉正在傳遞的中斷。
- **伺服器檢查套用在所有 profile**：block 數與 block size、FLASH_ATTN、GDN prefill kernel、MNBT 2048，provisional 也一樣要檢查。
- **verified profile 在 M2 之前的規則**：必須位於 release 目錄之內、列在 `DEPLOY_MANIFEST.json` 中，而且位元組與 manifest 的 hash 相符，否則開機前就拒絕。
  也就是說，verified profile 只能是 commit 過的檔案，而既有的 Mac「重新產生」測試保證 M2 之前不可能有。
- **profile 只讀一次**：讀到的位元組經檢查、記錄後複製到 `runs/<id>/profile.json`，gateway 使用這份複本。
  就緒後，`/readyz` 的 profile id、status、active class 與四個上限都必須和 preflight 相同，否則 teardown 並 exit 2。
- **vLLM 的環境**：另外移除會改變快取位置的變數（`XDG_*_HOME`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`FLASHINFER_WORKSPACE_BASE`）。gateway 也使用服務自己的 HOME / TMPDIR。
- **部署設定的限制**：
  - `research_dir` 必須等於 `core_b_plan.PROJECT_DIR`，因為 server 的 PATH 與 CUDA_HOME 指向它；
  - vLLM 的 port 不能是 8000；
  - gateway 的 backend URL、host（固定 127.0.0.1）、port 與 drain 逾時都由 deploy 決定；
  - `max_hold_s` 必須有限且 ≤ 3600，從 supervisor 啟動起算，包含開機；vLLM 與 gateway 的就緒等待都截到剩餘時間；
  - `startup_timeout_s` 必須有限；`drain_timeout_s` ≤ 300（等於 gateway 預設的 request deadline）。
- **`provisional-rtx.json`**：並發 4、佇列 4，輸出上限 1024 token（研究是 128），讓 IT-3 能區分真正的 abort 和自然結束、IT-5 有長的 request 可以 drain。
  它的原因欄位寫明「不是容量，也不是 Core 的 C_knee」。
- **整合測試**（`python -m capserve.itest`）：
  - IT-1 至 IT-5 共用一次 boot：itest 持有 lock，以子 process 啟動 `capserve.deploy up`（hold 模式）並把 lock 傳下去；IT-5 走真正的 `capserve.deploy stop`。
  - 除了 IT-0 之外都需要 lock，包括 IT-6。
  - IT-3 要求斷線時至少還有 4 s 的生成量，而且斷線後 +1 s 到 +3 s 之間生成的 token 數不再增加。
  - IT-4 以 `/tokenize`（不加 special tokens）實測 chat template 的額外 token 數，並和設定的常數比較；另有空訊息與 8 則訊息兩種情況。
  - IT-0 要求 vLLM 的 keep-alive 長於 gateway 的 2 s 連線重用。
  - itest 安裝了 SIGTERM / SIGHUP 的處理；中斷時仍會寫出 `itest.json`，未完成的測試標為 failed，並記錄 `interrupted`。
- **`runs.py`**：`release` 與 `profile` 也是固定欄位，之後不能改寫。
- **`deploy_service.sh`**：`~/llm-serving-service` 或其中的 `releases/` 是 symlink 時一律拒絕；解壓後重新讀取 `DEPLOYED_COMMIT`；使用 `.venv/bin/python`。
  解壓不是原子操作，但不完整的 release 會被 `up` 的 manifest 檢查拒絕。
- **未驗證、要在 RTX 上確認的事**：
  - vLLM 另外啟動的 EngineCore process 是否也繼承了 lock 的 fd（IT-1 目前只檢查 API server）；
  - abort 之後 vLLM 的 running / waiting gauge 是否會更新（IT-3 另外看生成的 token 數）；
  - ssh / WSL 的引號處理與 vLLM 的 log 格式；
  - IT-6 沒有檢查 `/tmp/torchinductor_<user>`（本文的清單沒有列）。
- RTX 上的執行同樣受 1,000 MiB 的閒置上限限制，目前因 Windows 桌面應用程式占用 GPU 記憶體而 blocked。〔2026-10-16 起依 decision 031 的產品規格，見下方「031 的產品判定」。〕

## 031 的產品判定（2026-10-16，使用者確認）

本文 preflight、teardown 與 IT-5 中的「GPU memory ≤ IDLE_MIB（1,000 MiB）」由 decision 031 的產品規格取代，原文保留：

- **preflight**：本 process 確實持有 lock；取一個 30 個樣本的 1 Hz 窗，每個樣本 utilization.gpu ≤ 5%、沒有 compute process、memory.used 的範圍 ≤ 64 MiB。
  - 閒置基線取窗內的最大值。
  - 高於研究的餘裕上限（1,018 MiB）只發警告。
  - **高於 2,048 MiB 則拒絕**（031 的補充）：那已經不是桌面的用量，很可能是其他專案常駐的模型。
- **shared usage**：在 boot 前、vLLM 就緒後、工作結束後各記錄一次，只作描述，讀取失敗也不停止。
- **teardown 與 IT-5**：GPU 記憶體要在 60 s 內回到「boot 前基線 + 64 MiB」以內。
- **state.json 的 phase 順序**：preflight → starting → ready → stopping → stopped；preflight 被拒絕時為 refused。
  只有確實持有 lock 時才在 preflight 前寫入，所以 preflight 期間也能 `stop`。


## 實作補記（M2，2026-10-17；尚未有任何 verified profile）

M2（verified profile 的產生器與 class 狀態處理）依本文與 030 8.7 以 TDD 實作，經一輪審查：4 項發現全部屬實，3 項在程式中修正並附回歸測試，1 項是本補記。

**產生器**（`python -m capserve.profile_gen --session … --freeze-commit … --out …`）
- 拒絕時 exit 2，不寫任何檔案。
- 新增必填的 `--freeze-commit`：session.json 沒有記錄凍結 commit，所以由外部傳入，並以 git 驗證：
  - 它是第一次凍結（解凍後再凍結一律拒絕；日後若依 030 8.9 獲准，要另寫 decision 修改產生器）；
  - 該 commit 的 `configs/capacity.toml` 位元組等於 session 記錄的 sha256；
  - 5 個 Stage C 模組（capacity_plan、run_capacity、capacity_analysis、capacity_report、gpu_idle）在凍結 commit、DEPLOYED_COMMIT 與產生器實際 import 的檔案三者之間逐位元相同。
  本文原寫「沒有任何覆寫參數」；`--freeze-commit` 不是覆寫，而是被驗證的輸入。
- 其他拒絕條件：
  - 工作樹有未 commit 的變更；
  - `--out` 已存在或位於 `results/` 之下；
  - session 的 server fingerprint 不是產品 server 的；
  - session 或 analysis.json 格式損壞。
- class 名稱：`uniform-2k`、`uniform-8k`、`mixed-2k`、`mixed-8k`；`source` 為 `stage_c:<workload>:c_prod:<status>`。
- 對應：
  - EXACT / LOWER / CENSORED → verified，max_concurrency = max_queue = C_PROD，bound 為 exact / lower / censored；
  - INFEASIBLE → verified 0（bound exact，代表拒絕這個 class）；
  - UNSTABLE / NT → unavailable，附原因，數字欄位為 null。
  每個 class 另附 scope、C_PROD、C_SLO、C_KV、C_cap、flags、binding、margins、開場湧入。
- 輸出沒有時間戳記，可以重新產生並逐位元比較。

**Profile 的 schema**（schema_version 仍為 1，向下相容；兩個 provisional profile 不用改）
- class 新增 `status`、`bound`、`unavailable_reason`、`evidence`。
- verified 的 `max_concurrency` 可以是 0（INFEASIBLE）；unavailable 的數字欄位為 null。本文 loader 規則「1 ≤ max_concurrency」以此為例外。
- provenance 新增 `deployed_commit`、`capacity_sha256`、`pins_sha256`、`gpu`、`gpu_driver`；三個 commit 欄位必須是完整 40 hex。
- SLO 的 terms 允許一層巢狀，才能原樣複製 S2。

**載入時的 provenance 驗證**（M0 補記延到 M2 的項目）
- verified profile 必須同時符合：
  - `configs/capacity.toml` 已凍結；
  - `capacity_sha256` 與 `pins_sha256` 等於這份 checkout 的檔案；
  - SLO 等於 capacity.toml 的 `[slo]`；
  - `profile_id` 為 `verified-<source_session>`，`analysis_path` 為 `results/capacity/<sid>/analysis.json`；
  - class 的名稱與 token 上限符合它的 workload，每個 workload 最多一個 class。
- `capserve serve` 與 deploy 都只接受 `service/profiles/<name>.json` 中、位元組相符的 verified 檔案（deploy 傳給 gateway 的 `runs/<id>/profile.json` 複本仍可使用）。
- Mac 的「重新產生」測試掃描所有追蹤中的檔案（`git ls-files`），任何 status 為 verified 的 JSON 都要能由產生器重新產生。
- **沒有做**：執行期不重新計算數字，也不比對 RTX 上 analysis.json 的 sha256（該檔是否存在不確定，缺檔時就通過的檢查沒有意義）。數字的可信度依靠「只服務 commit 過的 `service/profiles` 檔案」加上重新產生測試。

**Gateway**
- active class 為 unavailable：`serve` 與 `deploy up` 都拒絕啟動，任何旗標都不能覆寫。
- active class 為 verified 0：
  - `capserve serve` 會啟動，但每個 completion request 在讀 body 之前就回 503 `class_not_serviceable`：不排隊、不帶 Retry-After、`x-should-retry: false`。本文「503 都帶 Retry-After」以此為例外。
  - `/readyz` 回 503，並附 `class_status`、`bound`、`serviceable`、`not_ready_reason`。
  - `deploy up` 在 preflight 就拒絕，因為 gateway 永遠不會就緒，啟動 vLLM 只是浪費。
- `AdmissionController` 新增關閉模式（0 / 0 代表一律拒絕）；M0 原本「0 會報錯」的測試已改寫。

## 以 verified profile 跑 RTX 整合測試（2026-10-18；release `7513d62`）

- 第一次以 `verified-formal-20261017-c1b.json`（`service/deploy/rtx-verified.toml`，active class uniform-2k）跑 IT-0 至 IT-6，結果 `pt-20261017T184411Z-itest`：只有 IT-3 不通過。
  - 斷線行為本身正確：stream / non-stream 斷線後 vLLM 的 running + waiting 在 0.06 s / 0.23 s 內回到 0，之後沒有再生成 token，gateway 的 in_flight 也回到 0。
  - 不通過的是「決定性」這一項：verified profile 的輸出上限是研究的 128 token，斷線時只剩約 1.4 s / 1.0 s 的生成量，少於 IT-3 要求的 4 s。
    所以這個測試無法區分真正的 abort 和自然結束。這正是 M1 補記中 `provisional-rtx.json` 把輸出上限設為 1,024 token 的原因。
- **規則**：以 verified profile 跑整合測試時，執行 IT-0、IT-1、IT-2、IT-4、IT-5、IT-6。
  IT-3 驗證的是 gateway 的程式，和 profile 無關，所以在同一個 release、同一次 lock 中，以 provisional-rtx 跑完整的 IT-0 至 IT-6。
  - 不修改 IT-3 的門檻：放寬它就無法分辨 abort 和自然結束。
- **同一個 release、同一次 lock 的結果**（2026-10-18 02:46–02:50）：
  - verified profile，IT-0、1、2、4、5、6 全部通過：`pt-20261017T184655Z-itest`；
  - provisional-rtx，IT-0 至 IT-6 全部通過，IT-3 是決定性的：`pt-20261017T184834Z-itest`。
- 這些都是產品測試，不是容量量測。輸出在 WSL 的 `~/llm-serving-service/runs/`，不在 `results/`。

## 正式設定的 profile 釘選（2026-10-19，Phase 2 結案，decision 036）

- **缺口**：正式部署設定 `rtx-verified.toml` 沒有指定要用哪一份 profile。
  shell 中留著的 `CAPSERVE_ALLOW_PROVISIONAL=1`，或誤傳一份 class 也叫 `uniform-2k` 的 provisional profile，都可能以正式設定的名義被部署。
- **修正**（TDD；結案 commit 新增 22 個測試案例，審查後再加 7 個，共 29 個）：
  - `DeployConfig` 新增 `production_profile`，必須是 `service/profiles/<name>.json`。`rtx-verified.toml` 釘住 `verified-formal-20261017-c1b.json`。
  - **`deploy.production_profile_problems`**：有釘選時，`deploy up`（包括 `--loadtest`）在任何東西啟動之前拒絕不是這份檔案的 `--profile`，以及任何允許 provisional 的執行（flag 或環境變數）。
    這時 profile 一律以「不允許 provisional」的方式解析。
  - **`capserve.itest`**：
    - `--profile` 的預設改為：正式設定時用釘住的 profile，其他設定時仍是 provisional-rtx；
    - 有釘選時，同樣在啟動前拒絕其他 profile 與任何允許 provisional 的執行。
      審查後改為和 `deploy.main` 用同一個有效值：flag 或會被子程序繼承的 `CAPSERVE_ALLOW_PROVISIONAL=1`。原本只看 flag，環境變數要到子程序 `deploy up` 才被拒絕。
  - **profile 載入器**：provisional profile 不得使用 Stage C 的 class 名稱（`CLASS_NAMES`），id 也不得以 `verified-` 開頭。
- **沒有改的部分**：
  - `python -m capserve serve` 不讀部署設定，仍可用 `--allow-provisional` 載入 provisional profile。這時處處標示 provisional（M0）。它是開發用的低階進入點，正式部署只用 `deploy up`。
  - IT-3 的門檻不變。
- **RTX 上的確認**（2026-10-19，最終 release `5201760`，036 的結案紀錄）：
  - 以 `rtx-verified.toml` 執行 itest 時，`--allow-provisional`、環境變數、另一份 profile 三種情況都在啟動前以 exit 2 拒絕；
  - 不指定 `--profile` 時載入釘選的 verified profile，IT-0、1、2、4、5、6 通過；
  - provisional-rtx 的 IT-3 通過。
