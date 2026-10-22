# 038 — capserve v1：不依賴研究結果的七個項目

- 狀態：Accepted（設計；實作依本文以 TDD 進行，實作時的細節記在文末的補記）
- 日期：2026-10-23
- 對應：decision 029（MVP 架構）、036（IT-3 的限制）、037（Phase 3 範圍與使用者的子決定）
- 時點：任何 v1 程式之前。多 class 或依 token 加權的准入要等 Stage D 的結果，不在本文。

## 共通原則（沿用 029）

- 不改 vLLM 的啟動參數。任何 argv 的變更都會改變 fingerprint，verified profile 就不再適用。
- 送進 vLLM 的負載不得超出研究量過的範圍：每個 request 的 prompt 不超過 class 的 max_prompt_tokens，輸出不超過 max_output_tokens，並發不超過 max_concurrency。
- 新功能都是產品政策，不是研究結果。產品測試的輸出不進 `results/`。
- 除非另外寫明，預設值維持 MVP 的行為；正式設定 `rtx-verified.toml` 的變更在 RTX 整合測試通過後才做。

## 1. Thinking：預設關閉，client 可以開啟（使用者決定，037）

- chat 接受 `chat_template_kwargs`，但其中只允許 `enable_thinking`（布林值）。其他 key 一律回 400 `unsupported_parameter`，因為 template 參數會改變 prompt。
- gateway 送給 vLLM 的 chat request 一律帶 `chat_template_kwargs: {"enable_thinking": X}`：client 有給就用 client 的值，沒有給就是 `false`。
- 計算 prompt 長度時，`/tokenize` 也帶同樣的 `chat_template_kwargs`，所以計數和實際送出的 prompt 相同。
  Qwen3.5 的 template 在 `enable_thinking` 為 false 時，生成提示會多出 `<think>\n\n</think>\n\n`。
  byte 快速路徑的 `chat_overhead` 必須涵蓋它，IT-4 會分別檢查 thinking 開與關的情況。
- completions 不受影響：raw prompt 不經過 chat template。
- **容量的適用性**：研究量的是 `/v1/completions`、固定長度的 prompt 與 128 個 output token。thinking 開或關都不改變這三個上限。
  thinking 關閉時，回答通常更早自然結束，只會減少負載。
- 沒有加 `--reasoning-parser`（會改變 argv），所以 thinking 開啟時，思考內容留在 `content` 中。文件會寫明。

## 2. IT-3 的決定性判定改用 token 數（036 揭露的限制）

- **原本的問題**：IT-3 要求斷線時至少還有 4 s 的生成時間，才算決定性。正式設定的輸出上限是 128 token，單一 request 約 1–2 s 就生成完，所以 IT-3 永遠無法判定。
- **新的判定**：
  - request 用 `/v1/completions`，帶 `ignore_eos: true`、`max_tokens` 等於 class 的上限 M。
  - 送出前 vLLM 必須閒置，並記下 `vllm:generation_tokens_total` 的值 g0。
  - 斷線並等 token 數不再增加之後，再讀一次，得到 g2。差值 Δ = g2 − g0 就是這個 request 實際生成的 token 數。
  - 沒有被 abort 時，Δ 一定等於 M（`ignore_eos`，沒有 stop，prompt 很短，不會碰到 max_model_len）。所以 **Δ < M 就證明生成被截斷**。
  - 這個判定不依賴生成需要多少時間，因此在 128 token 下也是決定性的。
- 原本的時間條件（2 s 內 running + waiting 回到 0、斷線後 token 不再增加、gateway in_flight 回到 0）保留。
  「剩餘生成時間 ≥ 4 s」改為只記錄的描述性數值。
- Δ = M 時判為不通過（不能證明 abort 有傳到 vLLM），不是無法判定。
- **為什麼不用 `vllm:request_success_total{finished_reason="abort"}`**：
  - vLLM 0.30.0 的這個 counter 只在 engine 的 iteration stats 中統計 finished request。
  - 唯讀檢查原始碼後，無法確定 client abort 一定會被計入。
  - 所以它只作為描述性的紀錄。

## 3. 依負載調整的 Retry-After

- **適用範圍**：只限 429 的 `queue_full` 與 `queue_timeout`。`backend_unavailable` 與 `draining` 維持設定中的固定值，因為恢復時間和負載無關。
- **估計方式**：
  - S̄ 是最近完成的 request 持有 permit 的時間，以 EWMA 計算（α = 0.2）。只計入正常完成的 request，不計斷線或錯誤。
  - 預估等待時間 = (queued + 1) × S̄ ÷ max_concurrency。
  - 還沒有任何完成的 request 時，退回固定值。
  - 結果夾在 [min_s, max_s] 之間（預設 1 與 60 s）。
  - `Retry-After` 取整數秒並無條件進位；`retry-after-ms` 取毫秒。
- **設定**：`[retry_after_s]` 加上 `mode = "fixed" | "adaptive"`，預設 `adaptive`，以及 `min_s`、`max_s`。原本各原因的固定值保留，作為退回值與 fixed 模式的值。
- 這只是給 client 的提示，不改變准入。

## 4. gateway 內建 tokenizer

- **目的**：
  - 經過 gateway 的 `/tokenize` 呼叫，不在研究量測的負載中（036 的「不涵蓋」；D7 會量它的影響）；
  - 每次呼叫還多一次 HTTP 往返。
- **設定**：`tokenizer = "vllm" | "local"`，預設 `vllm`，也就是 MVP 的行為。
  正式設定只有在 RTX 的 IT-4 證明 local 與 vLLM 的計數完全一致之後，才改成 `local`。
- **local 模式**：
  - 用 transformers 的 `AutoTokenizer`，從 vLLM 服務的同一份本地 HF snapshot 載入。設 `local_files_only`，不連網，不寫入快取。
  - chat 用 `apply_chat_template(..., add_generation_prompt=True, enable_thinking=X)`，completions 用 `add_special_tokens=True`。這和 vLLM 的做法相同。
  - text parts 的 message 先依 vLLM 0.30.0 的規則轉成字串，再交給 template。實作時以原始碼確認，IT-4 驗證。
  - tokenize 在 thread 中執行，不阻塞 event loop。
  - 它不碰 vLLM，所以在**准入之前**執行：過長的 prompt 不占 permit 就回 400。CPU 成本受 guard 的 pending 上限限制。
  - 載入失敗時拒絕啟動，不會悄悄退回 vllm 模式。
- **IT-4 擴充**：
  - 同一組 chat 與 completion 案例，比較 local 計數與 vLLM `/tokenize` 的計數，要求完全相等；
  - 加上 thinking 開與關的案例，以及 class 上限附近的長 prompt。

## 5. per-key quota

- **金鑰檔**：`CAPSERVE_API_KEYS_FILE` 指向一個 TOML 檔，放在 repo 與 release 目錄之外。
  - 每把金鑰只存名稱與 sha256，**不存明文**。
  - 選用的限制：`requests_per_minute` 與 `max_in_flight`。
  - 金鑰由使用者自己產生並放進檔案。我不產生、不讀取、不保存任何真實的金鑰；測試只用測試中產生的值。
- `CAPSERVE_API_KEY`（單一金鑰，沒有 quota）保留。兩個都設定時拒絕啟動。
- **檢查**：
  - 對 bearer 計算 sha256，再以 `hmac.compare_digest` 逐一比較所有項目（不提早結束）。
  - 驗證之後、讀 body 之前，檢查該金鑰的 quota：
    - `requests_per_minute`：token bucket；
    - `max_in_flight`：該金鑰在 gateway 內（排隊加執行）的 request 數。
  - 超過時回 429 `quota_exceeded`（rate_limit_error）並帶 Retry-After：bucket 的值是下一個 token 的時間，in_flight 的值沿用 3. 的估計。
- **紀錄**：access log 與 metrics 只記金鑰的名稱，不記金鑰。metrics 加上 `capserve_key_requests_total{key, outcome}`，label 的數量受金鑰檔限制。
- quota 只會讓 request 更少，不改變容量上限。

## 6. TLS

- gateway 直接以 uvicorn 的 `ssl_certfile` / `ssl_keyfile` 提供 HTTPS。這樣不必另外安裝 reverse proxy，也不必新增套件。
- **設定**：`[tls] certfile`、`keyfile`。
  - 路徑必須在 repo 與 release 目錄之外；
  - 憑證與私鑰由使用者提供，我不產生正式用的憑證或私鑰；
  - 測試用 `openssl` 在暫存目錄產生自簽憑證，測試結束即刪除。
- 綁定非 loopback 的位址仍然需要金鑰（`CAPSERVE_API_KEY` 或金鑰檔）。TLS 不取代驗證。
- supervisor 對自己的 gateway 做 readiness 檢查時，以 HTTPS 連到 loopback，並信任設定中的憑證。

## 7. 常駐需要的功能（研究期間不常駐，使用者決定，037）

- **`capserve.deploy up --resident`**：
  - 不套用 `max_hold_s`；
  - vLLM 或 gateway 意外結束時，由 supervisor 重新啟動：
    - backoff 為 5、10、20 … 秒，最多 300 秒；
    - 在 `restart_window_s`（預設 3600 s）內超過 `max_restarts`（預設 5）次就停止，並寫入 state.json；
  - supervisor 自己持有 GPU lock，所以重新啟動的期間 lock 不會被釋放。
  - 重新啟動前做同樣的 preflight。GPU memory 沒有回到閒置值以下時，等待而不啟動。
- **開機啟動**：
  - 我提供 WSL 端的腳本 `tools/rtx/capserve_resident.sh`：以 `flock -n` 取得 GPU lock，再執行 `deploy up --resident`；
  - 註冊 Windows 工作排程器的步驟寫成文件，由使用者自己做。這屬於系統設定，我不修改。
- **何時啟用**：
  - 研究的 GPU 工作結束、並取得使用者同意後，才真正常駐；
  - 在那之前只在 Mac 測試，以及做時間有界的 RTX 產品測試：啟動、殺掉 vLLM、確認重新啟動、停止，全部在一次 GPU lock 中完成。
- 常駐期間，研究要先執行 `capserve.deploy stop` 釋放 lock。runner 不會自己停掉服務：研究不得搶占，由使用者或我明確執行。

## 不變的部分

- 正式設定仍釘住 `verified-formal-20261017-c1b.json`。active class、上限、vLLM 的參數都不變。
- MVP 的所有拒絕規則與測試保留。v1 的每個項目都另外加上測試。

## 為什麼

- 每一項都只改 gateway 的行為，不改送進 vLLM 的負載上限。所以既有的 verified 容量仍然適用，不需要新的研究。
- local tokenizer 預設關閉，要先在 RTX 上證明計數一致：計數少算一個 token，就可能讓超過 class 上限的 prompt 進入 vLLM。

## 放棄的替代方案

- **TLS 用 Caddy 或 nginx 做 reverse proxy**：要在 WSL 上安裝新軟體，多一個 process 要管理；uvicorn 本身就支援 TLS。
- **金鑰檔存明文**：檔案外洩就等於金鑰外洩，sha256 不會。
- **用 `request_success_total{abort}` 判定 IT-3**：見 2.。
- **常駐時讓研究的 runner 自動停掉服務**：會變成由程式自動搶占，違反「不得搶占」。改由人明確執行。

## 實作補記

**1. thinking（2026-10-23）**
- `openai_api.ChatTemplateKwargs` 是 strict model，只有 `enable_thinking: bool | None`。
  - 其他 key 回 400 `unsupported_parameter`，param 為 `chat_template_kwargs.<key>`；
  - 非布林值（包括 0、1、"yes"）回 400 `invalid_value`；
  - `null`、`{}`、`{"enable_thinking": null}` 都視為沒有給，使用預設值。
- `parse_request` 對每個 chat request 重寫 `chat_template_kwargs`，只留下 `enable_thinking`，預設值為 `THINKING_DEFAULT = False`。completions 一律不帶。
- `PromptBudget.count` 對 chat 的 `/tokenize` 帶同樣的 `chat_template_kwargs`。
- byte 快速路徑的常數（`chat_overhead` 32）不變，由 IT-4 在 thinking 開與關兩種情況下驗證。
- MVP 測試中「`chat_template_kwargs` 是不支援的參數」那一項，改為 completions 帶它時拒絕。

**2. IT-3 與 IT-4（2026-10-23）**
- **IT-0** 加兩項：`/tokenize` 的 chat request 與 `ChatCompletionRequest` 都有 `chat_template_kwargs`。2026-10-23 唯讀查看 RTX 上的 vLLM 0.30.0，兩者都有。
- **IT-3** 的決定性條件改為 `the generation was cut short (decisive)`：在閒置的 vLLM 上記下 g0，token 不再增加後讀 g2，要求 0 ≤ g2 − g0 < M。
  - 舊的「剩餘時間 ≥ 4 s」不再是檢查，`remaining_s` 只寫入 data；
  - g0 或 g2 讀不到時判為不通過。
- **IT-4**：每個 chat 案例都在 thinking 關與開兩種情況下各測一次，檢查名稱加上 `, thinking off|on`。
- **假後端**：
  - 新增 `honor_max_tokens`：生成 min(n_tokens, request 的上限) 個 token，和 vLLM 帶 `ignore_eos` 時相同；
  - 串流的 role、finish 與 `[DONE]` event 不再計入 `generation_tokens_total`。
- **測試**：
  - 128 token、每個 token 20 ms 時，新判定是決定性的；舊判定在這種情況下永遠無法判定。
  - 一次生成全部 128 個 token 時，判為不通過。
  - 讀不到 token 數時，判為不通過。
- 新的 IT-3 要在下一次 RTX 產品測試中執行。在那之前，036 的結論（verified 設定下的 IT-3 無法判定）不變。

**3. adaptive Retry-After（2026-10-23）**
- 新模組 `capserve/retry.py` 的 `RetryAfterPolicy`：
  - guard 結束每個 request 時，以 `on_finish` 交給它；
  - 只計入 outcome 為 ok、而且有 queue_wait 的 request，服務時間 = duration − queue_wait。
- **guard 自己的 429**（pending 上限）也用同一個估計：`Guard` 的 `queue_full_retry_after_s` 改為 `retry_after_s(reason)`。
- `AdmissionController` 不變，仍只收四個原因的固定值（`RetryAfter.fixed_values()`）。
  它的 `Rejection.retry_after_s` 只供測試讀取，response 的值一律由 policy 決定。
- `openai_api.retry_headers` 接受小數：`Retry-After` 取 `ceil`，`retry-after-ms` 取 `round`。整數輸入的結果和 MVP 相同。
- **設定**：`[retry_after_s]` 加上 `mode`（預設 adaptive）、`min_s`（1）、`max_s`（60），`min_s ≤ max_s`。
- MVP 的超載測試在任何 request 完成之前就觸發 429，所以仍得到固定值 2，不需要修改。

**4. 內建 tokenizer（2026-10-23）**
- **RTX 上的事實**（唯讀檢查）：
  - vLLM 0.30.0 的 `/tokenize` 對 chat 走 `online_renderer.preprocess_chat`，也就是 chat template 加上 `chat_template_kwargs`；
  - 服務的 snapshot（`a398088c`）的 `chat_template.jinja` 和 Mac 上 Qwen3.5-9B 基礎模型的檔案位元組相同，但 `tokenizer.json` 不同；
  - 所以 Mac 上不能用基礎模型代替做一致性測試。計數是否一致只由 RTX 的 IT-4 判定。
  - research venv 有 transformers 5.16.1 與 tokenizers 0.23.1。Mac 的 venv 沒有，也不為此新增套件。
- **text parts**：gateway 一律把 text parts 接成字串再送出，兩種 tokenizer 模式都一樣。
  - 這樣 vLLM 不必判斷 content 的格式（string 或 openai），vLLM 與本地 tokenizer 看到的 messages 完全相同。
  - 接法是不加分隔字元，和 Qwen3.5 的 template 渲染列表的方式相同。
  - 這是 MVP 行為的小改變：原本列表原樣送給 vLLM，由 vLLM 決定怎麼接。
- **`LocalTokenizer`**：
  - chat：`apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs)`，再用 `encode(add_special_tokens=False)` 計數；
  - completions：`encode(add_special_tokens=True)`；token id 的 prompt 直接計數。
  - template 丟出錯誤時回 400 `invalid_value`（param `messages`）。vLLM 模式則是原樣轉傳 vLLM 的 400。
  - 載入時：snapshot 目錄不存在、沒有 transformers、或無法載入，都會讓 `capserve serve` exit 2（`create_app` 則是丟出 ConfigError）。
- **設定**：`tokenizer` 為 `vllm`（預設）或 `local`。`tokenizer_path` 只能和 `local` 一起出現。
  - deploy 擁有 `tokenizer_path`，值為 `core_b_plan.snapshot_dir(pins)`；`[gateway]` 不得自己設定。
  - gateway 的環境加上 `HF_HUB_OFFLINE=1`。
- **IT-0**：載入服務的 tokenizer，並計數一次（不需要 GPU）。
- **IT-4**：
  - 每個 completion 與 chat 案例（thinking 開與關）都要求本地計數等於 vLLM 的計數；
  - completion 加上 n = 1,920，chat 加上「' the' × 1,900」，都在 uniform-2k 的上限附近；
  - tokenizer 無法載入時 IT-4 不通過。
- **正式設定仍是 `tokenizer = "vllm"`**：RTX 的 IT-4 通過之後，另外 commit 切換。
