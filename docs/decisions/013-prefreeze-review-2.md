# 013 — 凍結前 review（第二輪）：Core B client flags、無效 run、分析 CLI 的遮蔽與 session 有效性

- 狀態：Accepted（2026-10-08）
- 日期：2026-10-08
- 時點：Stage A 尚未凍結（`freeze.mac = false`）。沒有任何正式數據，NVIDIA / Core B 也還沒有任何數據。
- 來源：獨立 code review，範圍是 `b7863a9..a7ee04c`。
  兩個 HIGH 已對照 vLLM v0.30.0 原始碼確認：
  - `vllm/benchmarks/datasets/datasets.py`
  - `vllm/benchmarks/serve.py`

## 背景

review 找到兩類問題：

- 實作與既有文字不一致。
- 文字沒有寫明，實作默默選了一種做法。

全部在任何數據之前修正。每一項都註明是「讓實作符合文字」，還是「固定文字未寫明之處」。

## 決策

### 1. Core B client（實作不符 §4 / 007 → 修正實作）

- **`--temperature 0`**
  - §4 規定 temperature 0。
  - vLLM 0.30.0 的 `bench serve` 已不再預設送 0；沒設時由 server 端決定。
  - 補上。
- **`mixed` 加 `--custom-output-len 128`**
  - custom dataset 的 `--custom-output-len` 預設 256。除非設為 −1，它會覆蓋每列的 `output_tokens`。
  - 不補的話，每個 `mixed` request 都會生成 256 個 token（`--ignore-eos`），與 007 的 (·, 128) 不符。
    harness 會把所有 H-B4 run 判為 not ok。
- **client 的 tokenizer 固定到 pinned 版本**
  - `bench serve` 沒有 `--revision`，不指定時 client 端 tokenizer 會解析 repo 的 `main`。
  - 改為 `--tokenizer` 指向 pinned snapshot 目錄，並以 `HF_HUB_OFFLINE=1` 和與 server 相同的 `HF_HOME` 執行 bench。
- **`mixed` dataset 逐列檢查長度**
  - 產生時逐列檢查 token 數，算法與 vLLM custom dataset 計算 `prompt_len` 相同：`len(tokenizer(prompt).input_ids)`。
  - 每列必須恰為 L − 128 或 32640，任一列不符就不寫檔。

### 2. 更正 011 的一句描述（設計不變）

- 011 寫「短類就是 `uniform` 的 prompt」。查 `RandomDataset` 的實際行為：
  - 每個 request 的 offset 依序抽出，前 k 個與 num_requests 無關。
  - 但 decode → re-encode 後長度不符時，補 token 的抽樣所用的 RNG 狀態取決於 num_requests。
- 因此兩者**分布相同**（同產生器、同 seed、同長度），token 內容**不保證相同**。`uniform` 不同 C 之間也是如此。
- prefix caching 關閉、長度固定、`--ignore-eos`，token 內容對 TTFT / throughput 的影響可忽略。
  設計不改，只更正描述（`core_b_plan.py`、`core_b_mixed_dataset.py` 的說明一併更正）。

### 3. Core B 分析：只計入通過驗證的 run（固定文字未寫明之處）

- 下列情況，`run_core_b` 仍會記下 metrics，但標 `ok = false`：
  - failed > 0。
  - completed 不足。
  - input / output 長度不符。
- 分析只用 `ok` 的 run。
- §2 規定 throughput 與 P95 對 3 次 restart 取平均。若 grid 中任何 C 沒有 3 次有效 restart：
  - 該 L 的比較單位（H-B1–H-B3）記為 **not testable**，並列出缺的 C。
  - 不以較小的 C 代替 C_max。
- H-B4：每個 mix 也需要 3 次有效 restart。原程式只要求 2 次。
- plan JSON 的 C_KV / grid 以 `str(L)` 為 key，分析統一轉成 int。
  原本把 plan 的 C_KV 直接傳給 H-B2 時，每個 L 都會被判為 not testable。

### 4. Core A 分析

- **prefill（讓實作符合文字）**
  - §2 規定分析單位是 process-level mean；§3 定義 prefill tok/s = L ÷ TTFT。
  - 改為每個 process 先算 L ÷ TTFT（TTFT 為該 process `-r 5` 的 `avg_ns`），再取 cell 的平均與 SE。
  - 原本是 L ÷ mean(TTFT)，等於調和平均，差距約 CV²。
- **H-A3 的「相鄰」（固定文字未寫明之處）**
  - §2 規定「32K fit 的配置少於 2 個」才記為 not testable，所以只有 2 個時也可以檢驗。
  - 「相鄰」定義為：在 32K fit 的配置中，依 Q8_0 → Q6_K → Q4_K_M 的順序相鄰。
  - 例：Q6_K OOM 時，比較 Q8_0 與 Q4_K_M。原程式在這種情況下沒有任何配對，結果判為 not refuted。
- **不完整的資料**
  - cell 缺少比較所需的 metric（例如沒有 ttft）時，該比較單位記為 not testable，不再 crash。
  - RTX report 的 Stage B 值為 None 時直接報錯。

### 5. Session 有效性：狀態未知即無效（補充 012）

- Mac 上任一 env 紀錄若**電源來源未知**（例如 probe 失敗）或 **thermal 狀態未知**，該 session 標為 env-invalid。
- thermal 改為獨立檢查。原本電源來源缺失時，連 thermal 也一起略過。
- 沒有電池的 Mac 不檢查電量。RTX 桌機不檢查（012）。

### 6. 分析 CLI 只處理有效的正式 session

- `core_a_report` 與 `quality_analysis` 只接受同時符合下列條件的 session；否則拒絕執行，不寫 `analysis.json`：
  - `mode = formal`。
  - `env_valid = true`。
  - Core A 另需 sentinel `drift = false`。
- 理由：
  - pilot 要維持遮蔽（005、011）。
  - env-invalid 或 drift 的 session 依 §4 要整個重跑。若先看它的判定再重跑，就形成分岔路徑。
- 這些 session 的資料照樣保留，照實列入結果表。

### 7. 遮蔽的間接洩漏（照實記錄）

- 012 補充與已 commit 的 `results/quality/pilot-20261008-4b/pilot_report.json` 依 quant 列出 4B 的 KLD SE：
  Q8_0 約 1.3e-4、Q6_K 約 4.2e-4、Q4_K_M 約 7.7e-4。
- KLD 的 SE 與平均值同向變化，所以 4B 的 KLD 順序（H-A5(b)）在凍結前**可以推得**。
- 影響：
  - H-A5 的預測與門檻在 005 已寫定，早於 pilot，之後沒有改過。本項不改任何規則。
  - 已 commit 的 pilot 報告不改寫，history 不改。
- 修正：`quality_pilot` 之後只報每個 size 的**最大** SE。power 的估算只需要這個上界，所以 9B pilot 不會再洩漏順序。

### 8. §4 補列環境變數（讓文件符合實作）

- `VLLM_GDN_DECODE_KERNEL=triton` 已在 008 核准，但只寫在 008。§4 規定 `VLLM_*` 必須由本文件明列，因此補進 §4。
- 同時列出 server 與 bench 都設定的 `HF_HOME`、`HF_HUB_OFFLINE=1`。

## 為什麼

- 全部發生在任何正式數據之前。
- 1、4（prefill）、5、6、8：讓實作符合既有文字。
- 3、4（H-A3）：在文字未寫明之處，事先選定保守的做法。
- 7：照實記錄，不隱藏。

## 放棄的替代方案

- **`--custom-output-len -1`**（改用各列的 `output_tokens`）：結果同樣正確。
  不採用，因為 128 直接寫在指令中，不依賴檔案內容。
- **缺 restart 時用剩下的 restart 平均**：§2 寫的是 3 次平均。若不同 C 的 n 不一致，knee 的判定會受到失敗發生位置的影響。
- **分析 CLI 提供 `--force` 參數來處理 pilot**：沒有正當用途，不提供。

## 對實驗的影響

- PREREGISTRATION 依上修訂：
  - §2：Core B 無效 run 的處理、H-A3 相鄰的定義。
  - §3：prefill 的計算方式。
  - §4：環境變數、Core B client flags、電源有效性。
  - 「失敗也要報告」：分析 CLI 的限制。
- 分析程式與測試依上修改，仍在任何數據之前 commit，011 的條件維持。

## 補充（同日）：Core B 的 mixed plan 與 report 也在數據前寫定

- `core_b_plan.build_mixed_plan` 與 CLI（`uniform` / `mixed`）：
  - `mixed` 只在 L ∈ {2048, 8192} 跑，C 為該 L 的 `uniform` C_knee，3 次 restart（007）。
  - C_knee 由 `uniform` session 依 `core_b_analysis.knee` 算出。
  - 沒有 C_knee 的 L 不跑，並記錄原因。
  - plan 記錄 `derived_from`，即 `uniform` session 的 id。
- `benchmarks/core_b_report.py`：由正式的 `uniform` session 與由它衍生的正式 `mixed` session 產生 `analysis.json`。
  - 遇到下列任一情況就拒絕執行：
    - 任一 session 不是 formal。
    - `mixed` plan 衍生自其他 session。
    - `mixed` 的 C 不等於這裡重算的 C_knee。
  - H-B4 使用每個 run 的短類 P95 TTFT。這個值由 runner 從 `--save-detailed` 的逐筆資料算出，所以 report 不必讀 `raw/`。
  - 另附 (mix, L, C) 描述表：throughput、P95、KV cache 使用峰值、preemption 次數。
- `core_b_analysis.h_b4_values`：H-B4 的判定改為直接接受各 restart 的 P95 值。`h_b4` 先算出 P95 再呼叫它，規則不變。

## 補充（同日）：`mixed` dataset 每個 L 各 pin 一個 sha256

- 007 / 011 的設計是每個 L 一個 dataset 檔（2048、8192），但 `pins.toml` 的 `core_b.mix_dataset_sha256` 原本只有一個字串。
  改為以 L 為 key 的 table。
- `run_core_b` 新增兩項檢查：
  - plan 中有 `mixed` run 時，必須提供對應 L 的檔案。
  - 正式 run 在啟動 server 前，先比對檔案的 sha256 與 pin，不符就拒絕；session 記錄實際使用的 sha256。
- 兩個檔案已在 WSL 以 vLLM 0.30.0 的 `RandomDataset` 和 pinned tokenizer 建好（只用 CPU，沒有用 GPU）：
  - 每一列重新 tokenize 後都恰好等於目標長度。
  - 重建 `mixed-2048` 得到相同的 sha256。
  - 值已填入 pins，在 Stage B 隨 addendum 凍結。
