# capserve：Capacity-Aware Serving Service（MVP）

capserve 是 vLLM 0.30.0 OpenAI server 前面的一層薄 gateway（decision 027 §3–§5、029）。
它不寫 engine 或 scheduler，也不修改 vLLM；推論一律由 vLLM 執行。

- OpenAI 相容介面：`/v1/chat/completions`、`/v1/completions`（串流與非串流）、`/v1/models`。
  只接受允許清單內的欄位，每個 request 只接受一個 prompt；其他欄位回 400 `unsupported_parameter`。
- 准入控制：FIFO、並發上限、有界佇列、排隊逾時；超載回 429、無法服務回 503，都帶 `Retry-After`。
  - 429 的 `Retry-After` 依負載調整（038 §3）：(排隊數 + 1) × 最近完成的 request 的平均服務時間 ÷ 並發上限，夾在 1–60 s 之間。
    還沒有任何 request 完成時用固定值。`Retry-After` 取整數秒並無條件進位，`retry-after-ms` 保留估計值。
  - 503 維持固定值。設定 `[retry_after_s] mode = "fixed"` 可回到 MVP 的固定值。
- `/healthz`、`/readyz`、`/metrics`（Prometheus，只供營運，不是研究數據）。
- 上限來自 capacity profile（`profiles/`）中的 `active_class`。

## 正式成果與適用條件（decision 035 / 036）

- 只有由 Stage C（或 Stage D 的 D-A，043）正式、有效、凍結的分析經產生器產生的 `verified` profile，才能成為產品限制（027 §4）。
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
  - 封閉迴路，輸出 128 token，TTFT 從准入後起算。
  - 每個關鍵點 5 次 boot（c_low 為 3 次；uniform-2k 的 32 就是 c_low，3 / 3 PASS），每次的 R（指標 / SLO 門檻）都 ≤ 0.95，也就是留出 5% 的 guard band。
  - 落在 guard band 內（0.95 < R ≤ 1）的點是 MARGINAL，不算進 C_PROD。
- **不涵蓋**：
  - 開放迴路或 Poisson 到達、其他輸出長度；
  - 其他 prompt 長度分布：
    - uniform 只驗證了剛好 1,920 / 8,064 token；
    - mixed 只驗證了 90 / 10 的 mix，雖然 class 接受任何 ≤ 32,640 token 的 prompt；
  - 經過 capserve gateway 的流量：容量是直接對 vLLM 的 `/v1/completions` 量的，不含 gateway 的開銷、chat template 與 `/tokenize` 呼叫；
  - 准入前的排隊；
  - 開場湧入：C 個 request 同時到達時，uniform 第一波的 TTFT 最高 10.4 s（2K）與 11.0 s（8K），遠高於 SLO。
- Stage D（042）之後，uniform 兩類在長度不一、經過 gateway 時各有一個固定點的結果，開放迴路也有直連 vLLM 的數字，見下一節。
  c1b profile 的位元組與上表都不變。

## Stage D 之後的適用條件（decision 042 / 043）

**長度不一與經過 gateway**（D-C；只涉及 c1b 的 uniform class，在 C_PROD 的固定點；要求的 n = 5，U2K-var 4 / 5 有效，r5 依跳過規則沒有執行）

| class | 長度不一（直連 vLLM） | 經過 gateway |
|---|---|---|
| uniform-8k（8） | PASS：一條 pin 住的長度序列，86–8,015 token | PASS：固定 8,064 token |
| uniform-2k（32） | **FAIL**：4 個 run 中 1 個超過 2 s（2,042 ms），r3 的 R 0.989 已在 guard band 內；超標集中在第二波，但 W′ 的 P95 仍比 uniform 高約 70% | PASS，但最差 R_TPOT 0.923，離 guard band 只剩 0.027 |

- 正式的 active class 仍是 uniform-2k（32）。它在 prompt 長度不一時已經量到 FAIL；profile 的位元組不能改，只在這裡揭露（043「待決事項的處理」第 2 項）。
- uniform-8k 的兩種條件是分開驗證的。「長度不一加上 gateway」與雙峰的 2K / 8K 流量都沒有量過。
- gateway 讓 W′ 的短類 P95 增加：U2K 約 +390 ms（一格），U8K 約 +710 ms（約兩格）。無法拆成 gateway 排隊與上游兩部分。
- D-C 量的是 `capserve_sha256` `53488c6d…` 那一版的 gateway。之後 gateway 的變更（包括下面的 keep-alive 修正）對延遲的影響沒有量過。
- mixed-2k / mixed-8k 仍只到 c1b 的封閉迴路、直連、固定長度：長度不一、經過 gateway、開放迴路都沒有量過。

**開放迴路**（D-B；Poisson 到達、直連 vLLM、不靠准入控制）
- uniform-8k 的上限是 0.2519 req/s（0.4 λ_ref，EXACT）。這剛好是格點的最低點，差一格就是 INFEASIBLE。
- uniform-2k 為 NT：沒有任何驗證過的開放迴路到達率。
- 產品部署中 vLLM 只綁 127.0.0.1:8100，不會在沒有准入的情況下直接對外。
- 產品實際的情況，也就是「gateway 的並發上限加上 Poisson 到達」，從未量過。

**連線的 keep-alive**（043 對 042 說法的改正）
- gateway → vLLM：httpx 的 `keepalive_expiry` 2 s 低於 vLLM 的 5 s，所以 gateway 不會重用 vLLM 正要關閉的連線。
- client → gateway：之前 uvicorn 用預設的 5 s。vLLM bench 的 aiohttp 連線池（keepalive_timeout 60 s、FIFO 重用；aiohttp 預設 15 s）或 httpx（預設 5 s）的 client 連到 gateway 時，可能遇到 D-B 推論的同一類競態：重用一條 gateway 正要關閉的連線，request 在第一個 token 之前就失敗。
- **修正**：gateway 設定 `keep_alive_s`（預設 75 s）交給 uvicorn 的 `timeout_keep_alive`。閒置上限低於 75 s 的 client 連線池（aiohttp、httpx 的預設，以及 vLLM bench 的 60 s）會先讓閒置的連線過期；
  閒置更久或不設上限的 client（例如 Go 預設 90 s、urllib3）仍有同類競態。
  - 代價是閒置的 socket 保留得比較久。Guard 數的是 request，不是連線，所以不影響准入。
  - 這是產品修正，不宣稱任何容量。它會改變 `capserve_sha256`，所以 D-C 的 H-D7 只對應舊的那一版。
  - 下一次 GPU 空閒、取得 lock 時，以 `rtx-verified.toml` 跑 IT-0 至 IT-6（不是容量量測）。

**開場湧入**（D5 與 D-A；只作描述）
- 第一波（C 個 request 同時到達）的最大 TTFT：

  | class | 第一波最大 TTFT |
  |---|---|
  | uniform-2k（32） | 10.35 s（經過 gateway 10.37–10.42 s） |
  | uniform-8k（8） | 11.04 s（經過 gateway 11.00–11.10 s） |
  | uniform-32k（2，Stage D） | 12.8 s |
  | uniform-2k-o512（32，Stage D） | 10.50 s |
  | uniform-8k-o512（12，Stage D） | 16.95 s |

- 第二波（idx 在 [C, 2C)）的最大 TTFT 回到門檻內的有（各有效 run 最大值的範圍；以 derived 檔的 request 算出，只寫在文件中）：
  - c1b 固定長度：直連 uniform-2k 1.38 s、uniform-8k 2.47–2.48 s；經過 gateway（D-C）uniform-2k 1.71–1.73 s、uniform-8k 3.16–3.54 s（門檻 2 s / 6 s）；
  - uniform-8k 長度不一（D-C）：2.07–2.12 s（門檻 6 s）；
  - Stage D：uniform-32k 9.32–9.54 s（20 s）、uniform-8k-o512 2.52–2.55 s（6 s）、uniform-8k-t 5.38–5.61 s（6 s）、
    mixed-8k-t 短類 5.73–5.93 s（6 s）、長類 11.02–11.40 s（20 s）。
- 例外，第二波仍超過門檻：
  - uniform-2k-o512：每個 run 的 idx 63 停在 4.24–4.36 s，是門檻 2 s 的 2.1–2.2 倍（同一波其他 request 最高 1.72–1.76 s）。
    它在 vLLM 內等待，gateway 無法讓它逾時（見下一節「uniform-2k-o512 的 C 32 超過 C_KV 30」）。
  - uniform-2k 長度不一（D-C）：第二波最大 2.04–2.39 s，超過 2 s（上面 FAIL 的來源）。
- 產品不對第一波宣稱 SLO。
- vLLM 大致依序 prefill：同時到達的第 j 個 request 的首 token 時間約為 j × 單一 prompt 的 prefill 時間（2K 約 324 ms、8K 約 1,389 ms），不論它在 gateway 還是在 vLLM 中等待。
- 營運上以既有的 `capserve_queue_wait_seconds`、`capserve_ttft_seconds`（從到達起算）與 `capserve_upstream_ttft_seconds` 觀察湧入。
  容量以「准入之後」的 TTFT 定義，對外文件兩者並列。

## Stage D 的 verified profile（decision 043；都不是正式設定）

- 三份 profile，由產生器從 Stage D 的 D-A session 寫出（每個 session 與 server 一份），只 commit，**不釘選**。
  - 正式設定仍是 c1b，`rtx-verified.toml` 不變。
  - 任何 Stage D 的 class 都要先通過經過 gateway 的驗證（V-GW），才可能進入正式釘選，由使用者決定。

  | profile | server | class | prompt 上限 / 輸出 | 上限（exact） | 能否作為 active class（`serving_class`） |
  |---|---|---|---|---|---|
  | `verified-formal-20261023-d-a1b-b0` | B0 | `uniform-32k` | 32,640 / 128 | 2 | 可以（輸出 128、B0） |
  | `verified-formal-20261023-d-a1b-t` | T | `uniform-8k-t` | 8,064 / 128 | 4 | **不能**：030 8.8，只作報告 |
  | 同上 | T | `mixed-8k-t` | 32,640 / 128 | 4 | profile 允許（H-D2a supported），但 deploy 不啟動 T |
  | `verified-formal-20261023-d-a2-b0` | B0 | `uniform-2k-o512` | 1,920 / 512 | 32 | **不能**：輸出上限 > 128 |
  | 同上 | B0 | `uniform-8k-o512` | 8,064 / 512 | 12 | **不能**：輸出上限 > 128 |

  （T = `--long-prefill-token-threshold 512`，fingerprint 和 B0 不同；B0 是 Core B 的 server，和 c1b 相同。）
- **適用範圍**：封閉迴路、准入之後的延遲、量測時的輸出長度（剛好 128 或 512 個 token，ignore-eos）、各自的 server。
- **沒有量過的條件**：每個 class 的 `evidence.not_covered` 是一組封閉的 enum，profile 中不放任何 D-B 或 D-C 的數字。
  - 每個 class 都有：`prompt_length_variation`（prompt 長度不一）、`via_gateway`（經過 gateway）、`open_loop`（開放迴路）、
    `output_length_other_than_measured`（其他輸出長度）、`start_up_wave`（開場湧入）、`concurrent_classes`（同一台 server 上多個 class 同時並發）。
  - ABOVE_CKV 的 class（uniform-2k-o512）另外有 `in_vllm_wait_beyond_c_kv`。
- **uniform-2k-o512 的 C 32 超過 C_KV 30**（停頓只寫在文件中，引用 042）：
  - 每個 run 剛好 8 / 256 個 request（3.125%）停頓 4.2–5.2 s，是 SLO 的 2.1–2.6 倍，位置在 idx 63 + 31k；
  - 這些 request 在 vLLM 內等待，gateway 無法讓它們逾時；
  - 32 → 33 是斷崖（R_short 0.708 → 1.031），guard band 沒有提供預警。
- **WINDOW_SENSITIVE**：mixed-8k-t 的 c_late 為 5，所以 W 的 4 比較保守；uniform-2k-o512 的 c_full 為 UNSTABLE，只要包含開場湧入，就沒有任何驗證過的上限。
- **輸出長度**：較短的輸出不一定更安全。U8K 在輸出 512 時 C_PROD 為 12，在輸出 128（c1b）時 C 9 就已經 FAIL。
  gateway 只能強制輸出的上限（沒有給 `max_tokens` 時補上、超過回 400），EOS、stop 與預設關閉的 thinking 都會讓實際輸出更短。
- **跨 session 的敏感度**：在約 1.5% 的漂移內，U8K-T 的 C_PROD 可能是 3–4（只會往下），M8K-T 可能是 4–5（只會往上）。
- **服務規則**（`serving_class`；`create_app` 拋出 ProfileError、`capserve serve` exit 2、`deploy up` 在 preflight 拒絕，任何旗標都不能覆寫）：
  - 輸出上限大於 128 的 verified class 不能作為 active class（043 §4）。
  - T profile 只服務 `mixed-8k-t`，而且要 H-D2a 為 supported（030 8.8：T 只用於明確宣告為 M8K 的部署）；`uniform-8k-t` 只作報告。
  - B0 profile 上，T workload（U8K-T、M8K-T）的 class 一律拒絕（030 8.8：T 的上限不能用在 B0）。loader 已經拒絕這種 profile；`serving_class` 只讀 model，所以沒有經過 loader 的 profile 也擋得住。
  - `capserve.deploy` 只啟動 B0：profile 的 `server.long_prefill_token_threshold` 不是 null 時，preflight 拒絕。T 的部署要使用者決定（043 §6）。
  - `capserve serve` 是開發用的入口，不知道後端的 argv，無法擋住「profile 接到錯的 vLLM」（c1b 也一樣）。只有 `deploy up` 以 fingerprint 保證 server 正確。
- **揭露**：
  - `/readyz` 另外列出 `server_variant`（B0 / T），以及 active class 的 `flags` 與 `not_covered`。
    c1b 的 class 沒有 `not_covered`（null），它的適用條件以本 README 為準；provisional profile 的兩者都是 null。
  - `capserve_profile_info` 多兩個 label：`flags`（以逗號連接，沒有時為空字串）與 `server_variant`。
- **明確仍然不可用**（043 §7.5）：
  - 多 class 准入、token 加權准入、prefill pacing：只有候選設計，要先通過驗證研究（V4、V2），驗證之前不上線；
  - O512 class 的服務；
  - T 的部署，以及服務 `uniform-8k-t`；
  - 經過 gateway 的開放迴路宣稱；
  - uniform-2k 在 prompt 長度不一時的 SLO 宣稱。

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
- provisional profile 不得使用 Stage C 的 class 名稱（uniform-2k、uniform-8k、mixed-2k、mixed-8k）或 Stage D 的 class 名稱（uniform-32k、
  uniform-8k-t、mixed-8k-t、uniform-2k-o512、uniform-8k-o512；043），id 也不得以 `verified-` 開頭。
- unavailable 或 verified 0 的 class 不能作為 active class；輸出上限大於 128 的 verified class、T profile 中 `mixed-8k-t` 以外的 class 也不能（043）。
- `deploy up` 只啟動 B0：遇到 T 的 profile（`server.long_prefill_token_threshold` 不是 null）時，在 preflight 拒絕（043 §6）。
- `python -m capserve serve` 是 gateway 的低階進入點，供開發使用，不讀部署設定；正式部署不直接用它。
- `rtx.toml` 是整合測試設定，使用 provisional-rtx，沒有釘選。

## 整合測試的結果與限制（029 最後的補記、036、038）

**目前的結果**：release `c9202c5`（2026-10-23，038 的 IT），同一次 GPU lock 中執行

| 測試 | verified profile（`rtx-verified.toml`） | provisional-rtx（`rtx.toml`） |
|---|---|---|
| IT-0 至 IT-6 | 全部通過 | 全部通過 |
| IT-3（client 斷線，新判定） | **通過（決定性）**：串流生成 8 / 128 個 token，非串流 49 / 128 | 通過（決定性）：8 / 1,024、46 / 1,024 |
| IT-4（本地 tokenizer 與 vLLM 的計數） | 23 項全部相等 | 23 項全部相等 |

- 有 abort 時，生成的 token 數小於上限 M；沒有 abort 時，`ignore_eos` 會讓它剛好等於 M。所以 IT-3 在 128 token 下也是決定性的，036 的限制已解除。
- thinking 關閉時，chat template 比開啟時多 2 個 token（例如單一 user message：12 對 10），都在 `chat_overhead` 等常數之內。
- 這次執行時，正式設定仍是 `tokenizer = "vllm"`。計數一致後，兩份設定都改為 `local`（038 §4）。

**之前的結果**（036，舊判定）

| 測試（release `7513d62`；最終 `5201760` 重跑了 verified 的 IT-0、1、2、4、5、6 與 provisional 的 IT-0、3、6，結果相同） | verified profile（`rtx-verified.toml`） | provisional-rtx（`rtx.toml`，同一 release） |
|---|---|---|
| IT-0、IT-1、IT-2、IT-4、IT-5、IT-6 | 通過 | 通過 |
| IT-3（client 斷線） | **無法判定，不算通過** | 通過（決定性） |

- verified profile 的輸出上限是 128 token。斷線時只剩約 1.0–1.4 s 的生成量，少於 IT-3 要求的 4 s，所以無法區分「被 abort」和「本來就跑完」。
  - 第一次執行記為不通過。斷線後生成確實停止、vLLM 的 running 與 waiting 都回到 0，但這不構成決定性的證據。
- 決定性的斷線驗證只來自 provisional-rtx（1,024 token 輸出上限）。斷線處理和 profile 無關，所以推論 verified 設定下也成立，但在 128 token 下沒有直接證據。
- **038 §2 之後，IT-3 改用 token 數判定**：
  - 斷線的 request 用 `ignore_eos` 與 class 的輸出上限 M；
  - 比較 vLLM 的 `generation_tokens_total` 在 request 前後的差值。沒有 abort 時差值一定是 M，所以差值小於 M 就證明生成被截斷。
  - 這個判定不依賴剩餘的生成時間，在 128 token 下也是決定性的。
  - 新判定的結果見上方「目前的結果」。
- 以 verified profile 跑 IT 時用：

  ```bash
  python -m capserve.itest IT-0 IT-1 IT-2 IT-3 IT-4 IT-5 IT-6 --config service/deploy/rtx-verified.toml
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

# Stage D 的 D-A session（043）：一次寫出這個 session 每個 server 的 profile
PYTHONPATH=service:benchmarks .venv/bin/python -m capserve.profile_gen \
    --session results/stage_d/formal-YYYYMMDD-d-a1b --freeze-commit <設定 freeze.stage_d 的 commit，40 hex> \
    --out-dir service/profiles

# Stage V 的正式 session（044）：只為成立的單位寫出至多一份 §12 profile
PYTHONPATH=service:benchmarks .venv/bin/python -m capserve.profile_gen \
    --validation-session results/stage_v/formal-YYYYMMDD-v-adm --freeze-commit <設定 freeze.stage_v 的 commit，40 hex> \
    --out-dir service/profiles
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
- **Stage D**（043，PREREGISTRATION §9）：依 `--session` 的路徑分派。`results/stage_d/` 用 `--out-dir`，其他用 `--out`；兩者互斥，給錯的那一個時 exit 2。
  - 檔名固定為 `<out-dir>/<profile_id>.json`，profile_id 為 `verified-<session>-b0` 或 `-t`；沒有 `--server` 之類的選擇器。
  - 全部寫出或全部不寫：每份文件都先通過 loader、每個目標檔都不存在才開始寫；寫到一半失敗時，刪掉已寫出的檔案。
  - `--freeze-commit` 是第一次設定 `configs/stage_d.toml` `[freeze] stage_d = true` 的 commit（規則同上，對象是 `stage_d.toml`）；
    `stage_d_plan.HARNESS_FILES` 與上面 5 個 Stage C 模組在凍結 commit、DEPLOYED_COMMIT 與產生器實際執行的檔案三者相同；`capacity.toml` 也必須已凍結。
  - 只接受 D-A 的封閉迴路 session（plan 與 analysis 的 kind 都是 `closed`）。D-B、D-C 與中止的 `formal-20261023-d-a1` 一律拒絕。
  - 以凍結的模組唯讀重跑 `stage_d_report`，必須逐位元組重現 `analysis.json`。
  - 每次 boot 依它的 workload 分成 B0 與 T 兩組，各自以記錄的 argv 與 env 計算 fingerprint；每一組只能有一個值，而且必須是產品的 B0 或 T server。
  - `--out-dir` 必須是已存在、位於 `results/` 之外的目錄。
- **Stage V**（044，PREREGISTRATION §12）：只用 `--validation-session`（`--session` 指向 `results/stage_v/` 時 exit 2），只配 `--out-dir`。
  - 只由 `analysis.json` 的 `units` 決定寫什麼，而且只寫每一點都 PASS 的單位（PACE-2K 另需沒有被否決）；每個 session 至多一份 `verified-<session>-b0.json`：
    - PACE-2K / PACE-8K：c1b 的 `uniform-2k` / `uniform-8k` 原樣，加上量過的 `pacing`（含佇列設定）與 `validation`；
    - W1：`admission` 段（weighted-v1 的參數、佇列設定、量過的 mix 0.9 / 0.5、`gateway_path_sha256`、`validation`），成員是 c1b 的複本；
    - V1-2K / V1-8K：`uniform-2k-o1-512`（32）/ `uniform-8k-o1-512`（8），bound `validated`，evidence 為 d-a2-b0 的 O512 class 加上輸出分布、5 個 dataset 與 `via_gateway`；
      V1 class 的 evidence 中 scope、C_PROD、C_KV、flags、hypotheses 是 d-a2-b0 O512 class 在 D-A2 全部輸出 512 token 時的量測，
      它驗證的範圍是 `output_distribution` 加上 `validation`（C = `V1_C`，經過 gateway 的 client 視角）；
    - VG-32K / VG-M8K-T 不寫檔（只是正式釘選的候選）。
  - 沒有會寫檔的單位成立時，不寫任何檔、exit 0，摘要的 `written` 說明原因。
  - 另外的拒絕：`configs/stage_v.toml` 沒有凍結或凍結證據不全；`--freeze-commit` 不是第一次設定 `[freeze] stage_v = true` 的 commit
    （`stage_v_plan.HARNESS_FILES`、Stage D 與 Stage C 的凍結模組在凍結 commit、DEPLOYED_COMMIT 與產生器實際執行的檔案三者相同）；
    DEPLOYED_COMMIT 上請求路徑的 `gateway_path_sha256` 不是凍結值；c1b 或 d-a2-b0 不是凍結 commit 上的位元組；session 不是 `results/stage_v/formal-…`（mini 一律拒絕）；
    `stage_v_report` 會拒絕它，或唯讀重跑無法逐位元組重現 `analysis.json`；boot 的 server 不是該點的產品 server；
    同一個 plan（V-ADM / V-CLS）在同一次凍結下（記錄的 `stage_v_sha256`、`harness_sha256`、`gateway_path_sha256` 相同）已有一個較早開始、`stage_v_report` 接受的 session
    （044 停止規則「不重跑有效的 session 來換 PASS」，fail closed：較早的 session 有 analysis.json（report 只為它接受的 session 寫）就直接拒絕；
    否則只有在 session.json 記錄它中止或 `env_valid` 為 false，或 `refusal_problems` 為空而 report 自己判定無效（replay 不一致、sentinel 閘門失敗；分析本身失敗不算）時才不擋；
    一個目錄只有在 plan.json 等於它 session.json 記錄的 `plan_sha256`、兩者的 session 也相同時才算另一個 plan；
    目錄改過名、檔案和它的 session.json 記錄不符、`refusal_problems` 不為空、分析失敗、其他錯誤、這樣的 `formal-*` 目錄讀不到 session.json 或 started_at、`previous_formal` 鏈上的正式 session 不在 Mac 上、
    Mac 上不晚於本 session 開始的 Stage V 目錄扣掉它自己之後的時數加總不等於它開始時記錄的 `hours_used_h`（先取回每個 Stage V session），都拒絕；
    防的是誠實的錯誤，不防蓄意修改 session.json；每個 session 最多 2 次嘗試仍由操作者負責，044 實作補記十第 1、10、11 項）；
    停止規則的新凍結（解凍再凍結，或一直凍結、只更新 mini 與 `gateway_path_sha256`）之下的 session 一律拒絕，要等該凍結的 decision 一併修改產生器的「第一次凍結」規則（044 實作補記十第 2 項）。
  - 產生器不在 gateway 的請求路徑閉包內，修改它不會改變 `gateway_path_sha256`。
- 輸出是決定性的（沒有時間戳記）；每個 commit 過的 verified profile 都由 Mac 測試重新產生並比對（忽略 generator_commit）。
  這個測試掃描所有 tracked 檔案（`results/` 之外），任何路徑、任何副檔名中 `status` 為 `verified` 的 JSON 都要能重新產生。
- **載入時的檢查**（`capserve serve` 與 `deploy up` 都一樣）：`configs/capacity.toml` 必須已凍結；provenance 的
  `capacity_sha256`、`pins_sha256` 必須等於這個 checkout（release）的 `configs/capacity.toml`、`configs/pins.toml`；
  `slo` 必須逐字等於 `capacity.toml` 的 `[slo]`；`profile_id` 必須是 `verified-<source_session>`，`analysis_path` 必須是
  `results/capacity/<source_session>/analysis.json`；每個 class 的名稱與 token 上限必須是它的 Stage C workload 的（下表），每個 workload 至多一個 class。
  - §9（Stage D）的 profile 不套用上面 §8 的 `profile_id`、`analysis_path`、`slo` 與 class 表，改為要求：`configs/stage_d.toml` 與 `capacity.toml` 都已凍結；`stage_d_sha256`、`capacity_sha256`、`pins_sha256` 等於這個 checkout 的檔案；
    `slo` 逐字等於 `stage_d.toml` 的 `[slo]`；`profile_id` 為 `verified-<source_session>-<b0|t>`（依 `server.long_prefill_token_threshold`），
    `analysis_path` 為 `results/stage_d/<source_session>/analysis.json`；每個 class 是 D-A workload 的（下表），server、輸出長度與 `not_covered` 都要相符；
    `fingerprint_sha256` 是該 server（B0 或 T）的值。
  - §8 的 profile（例如 c1b）沒有任何 §9 的欄位，所以它的位元組、content_sha256 與載入結果都不變。
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

- 每個 D-A workload 對應一個 class（043 §2；sentinel、D-B、D-C 的 workload 不成為 class）：

  | class | workload | server | max_prompt_tokens | max_output_tokens |
  |---|---|---|---|---|
  | `uniform-32k` | U32K | B0 | 32,640 | 128 |
  | `uniform-8k-t` | U8K-T | T | 8,064 | 128 |
  | `mixed-8k-t` | M8K-T | T | 32,640（007 的 mix 的長 request） | 128 |
  | `uniform-2k-o512` | U2K-O512 | B0 | 1,920 | 512 |
  | `uniform-8k-o512` | U8K-O512 | B0 | 8,064 | 512 |

  - 名稱刻意和 c1b 的 class 不同，所以寫給 `uniform-8k` 的設定不會拿到 T 的 4 或 O512 的 12。

- class 的狀態依 C_PROD（030 8.7）：

  | C_PROD | class | gateway |
  |---|---|---|
  | EXACT / LOWER / CENSORED | `verified`，max_concurrency = max_queue = C_PROD，`bound` 為 exact / lower / censored | 正常服務 |
  | INFEASIBLE | `verified`，0（拒絕這個 class），bound exact | 啟動，但每個 completion request 都回 503 `class_not_serviceable`（不排隊、沒有 `Retry-After`、`x-should-retry: false`）；`/readyz` 為 503，`not_ready_reason` 為 `class_not_serviceable`；`deploy up` 在 preflight 拒絕（沒有可服務的東西，不啟動 vLLM） |
  | UNSTABLE / NT | `unavailable`，附原因，沒有任何數字 | 拒絕啟動（exit 2），任何旗標都不能覆寫；也不退回其他數字 |

- 每個 class 另附 Stage C 的證據（只作描述，不是上限）：適用範圍、C_PROD、C_SLO、C_KV、C_cap、旗標（例如 ABOVE_CKV）、先失守的項目、C_PROD 的餘裕與開場湧入。
  - Stage D 的 class 另外附上：server、量測時的輸出長度、其他時間窗的 C_PROD（c_full、c_late）、C_PROD 的有效 run 中最低的 attainment、
    最多的 preemption 與最高的 KV 峰值（後兩者是整個 bench process 的值）、hypothesis 的判定，以及 `not_covered`。
    數字都取自 analysis.json（margins 為 1 − worst_R，at_c_prod 為有效 run 的 min / max）；`not_covered` 由 `not_covered_for(workload, flags)` 決定；不從 derived 檔計算任何數字。
  - `/readyz` 另外列出 `class_status`、`bound`、`serviceable`、`server_variant`、`flags` 與 `not_covered`（043）。

## 與 vLLM 預設不同之處

- 沒有給 `max_tokens` 時，補上 active class 的 `max_output_tokens`（vLLM 的預設是 context 剩下的長度）；超過時回 400。
- **thinking 預設關閉**（使用者決定，037 / 038 §1）：
  - gateway 送給 vLLM 的每個 chat request 都帶 `chat_template_kwargs: {"enable_thinking": false}`。Qwen3.5 的 template 因此以空的 think 區塊開頭，直接回答。
  - client 可以用 `"chat_template_kwargs": {"enable_thinking": true}` 開啟。輸出仍受 class 的上限（128 token）限制，所以思考較長時，回答可能停在 `<think>` 之中。
  - `chat_template_kwargs` 只接受 `enable_thinking`，其他 key 回 400 `unsupported_parameter`。
  - 沒有啟用 vLLM 的 reasoning parser，因為那會改變啟動參數。thinking 開啟時，思考內容留在 `content` 中。
  - completions 不經過 chat template，不受影響。
- **text parts 合併成字串**（038 §4）：chat message 的 `content` 若是 text parts 的列表，gateway 會先接成一個字串，中間不加分隔字元，和 Qwen3.5 的 template 處理列表的方式相同，再送給 vLLM。
- **prompt 長度的計算**（038 §4）：
  - 預設 `tokenizer = "vllm"`：超過 byte 快速路徑時，持有 permit 呼叫 vLLM 的 `/tokenize`（MVP 的做法）。
  - `tokenizer = "local"`：
    - gateway 從 vLLM 服務的同一份 snapshot 載入 tokenizer，在准入之前自己計數，不呼叫 vLLM；
    - 部署時由 deploy 填入 snapshot 路徑；
    - 載入失敗時 gateway 拒絕啟動。
  - release `c9202c5` 的 IT-4 中，23 項計數全部相等，所以 `rtx.toml` 與 `rtx-verified.toml` 都已改為 `local`。

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
- **per-key quota**（038 §5）：
  - 設定 `CAPSERVE_API_KEYS_FILE` 指向一個 TOML 金鑰檔。檔案必須放在 repo 與 release 之外，和 `CAPSERVE_API_KEY` 二選一。
  - 檔案中每把金鑰只寫名稱與 sha256，**不寫金鑰本身**。可以加上 `requests_per_minute`、`max_in_flight` 兩種限制。

    ```toml
    [[keys]]
    name = "team-a"
    sha256 = "<金鑰 UTF-8 位元組的 sha256，64 個小寫十六進位字元>"
    requests_per_minute = 60
    max_in_flight = 8
    ```

  - 超過限制時回 429 `quota_exceeded`，並帶 Retry-After。
  - access log 與 metrics（`capserve_key_requests_total{key, outcome}`）只記名稱。
  - quota 只會擋掉 request，不會提高 profile 允許的並發。
- **TLS**（038 §6）：
  - gateway 設定加上 `[tls] certfile = "..."`、`keyfile = "..."` 就以 HTTPS 服務。
  - 憑證與私鑰由使用者提供，必須放在 repo 與 release 之外；啟動時會檢查可讀，而且兩者是一對，否則 exit 2。
  - TLS 不取代金鑰：綁定非 loopback 的位址仍然需要 `CAPSERVE_API_KEY` 或 `CAPSERVE_API_KEYS_FILE`。
  - 部署時，`[deploy] gateway_host` 可以是 `127.0.0.1`（預設）或 `0.0.0.0`；`0.0.0.0` 必須有 `[gateway.tls]`，而且環境中要有金鑰，否則在啟動任何東西之前就拒絕。
  - supervisor、loadtest 與 itest 一律經 loopback 連到自己的 gateway，只信任設定中的憑證，不檢查主機名稱。
  - 正式設定 `rtx-verified.toml` 仍然只綁 loopback，沒有 TLS。是否對外開放由使用者決定。
- **常駐**（038 §7；**研究期間不常駐**，使用者決定）：
  - `capserve.deploy up --resident`：
    - 不套用 `max_hold_s`；
    - vLLM 或 gateway 結束時，整套重新啟動（重新做 preflight），重啟前依序等待 5、10、20 … 秒，最長 300 秒；
    - 在 `restart_window_s`（預設 3,600 s）內失敗 `max_restarts`（預設 5）次就放棄，state.json 寫入 `gave_up`；
    - 期間 supervisor 一直持有 GPU lock。`capserve.deploy stop` 會結束它，不會再重啟。
  - `tools/rtx/capserve_resident.sh <sha>`：以 `flock -n` 取得 GPU lock 後，用正式設定啟動常駐模式。lock 被占用時立刻結束，不等待、不搶占。
  - 開機自動啟動要在 Windows 工作排程器註冊。這屬於系統設定，由使用者自己做；腳本開頭有指令範例。
  - 研究需要 GPU 時，要先明確執行 `capserve.deploy stop`。研究的 runner 不會自動停掉服務。
- 單一 process，uvicorn 固定 `http="h11"`、`loop="asyncio"`；SIGTERM 時先 drain，exit −15 為正常結束。
- client 連線的 keep-alive 為 `keep_alive_s`（預設 75 s，043），不是 uvicorn 的 5 s；原因見「Stage D 之後的適用條件」。
- 部署管理與整合測試（M1，decision 029 的 M1 實作補記）：
  - `python -m capserve.deploy render|up|status|stop`：在 RTX 上以 GPU lock 啟動 vLLM 與 gateway（前景 supervisor），teardown 一定執行。
  - `python -m capserve.loadtest`：產品負載測試（smoke / steady / burst / overload），輸出到 `runs/`，**不是容量量測**。
  - `python -m capserve.itest IT-0 … IT-6`：RTX 整合測試；除了 IT-0 都需要 GPU lock。
  - `tools/rtx/deploy_service.sh`：把 release 部署到 WSL 的 `~/llm-serving-service/releases/<sha>`，不寫入研究目錄。
  - RTX 上的閒置判定依 decision 031 的產品規格：GPU lock、使用率 ≤ 5%、基線穩定；閒置用量高於研究的餘裕上限只發警告，高於 2,048 MiB 則拒絕（保護其他專案）。
