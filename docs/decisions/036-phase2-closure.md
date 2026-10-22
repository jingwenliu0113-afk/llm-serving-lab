# 036 — Phase 2 結案：以 verified profile 為正式成果

- 狀態：**Accepted**（使用者 2026-10-19 決定）。
- 日期：2026-10-19
- 對應：027（Phase 2 的兩條線）、029（capserve）、030（§8）、031、032–035（C0 與 C1 的結果）

## 使用者決定（2026-10-19）

- 接受 Phase 2 的正式研究結果，這一輪先封版，不再追加正式 GPU 實驗。
- 以目前的 verified profile 為正式成果，保留下列容量：
  - uniform 2K：32；
  - uniform 8K：8；
  - mixed 2K / 8K：2。
- 收尾範圍：
  - README、產品文件、正式報告與 decision log 彼此一致；
  - 寫清楚容量適用的條件；
  - 如實揭露 IT-3 在 128 token 下的限制；
  - 確認正式 profile 不會被 provisional profile 意外取代；
  - 確認測試、commit、push 與工作區都乾淨。
- 不修改已封存的結果、不放寬 SLO、不重跑正式 session。後續的產品化或研究由使用者另外決定。

## 正式成果

**研究**（decision 035，session `formal-20261017-c1b`；最終結果，取代 033）

| workload | C_PROD（產品上限） | C_SLO | 狀態 |
|---|---|---|---|
| uniform 2K | 32 | 34 | 都是 EXACT；33、34 為 MARGINAL（STRICT_COARSE） |
| uniform 8K | 8 | 8 | EXACT |
| mixed 2K | 2 | 2 | EXACT |
| mixed 8K | 2 | 2 | EXACT |

- H-S1（mixed 降低容量）與 H-S2（C_SLO ≤ C_KV）在 2K、8K 都 supported，和事先預測相同。
- C0（032）：8K 雙模式為 Rrun，採分支 A。
- 第一個 session（033）因環境干擾（spill）而被取代。重跑的規則在任何新數據之前寫定（034）。

**產品**：`service/profiles/verified-formal-20261017-c1b.json`，由產生器從上述 session 產生，四個 class 都是 verified、bound 都是 exact。

## 容量的適用條件

只有在下列條件下，這些數字才有根據。任何一項不同，就不是驗證過的容量。

**硬體與平台**
- NVIDIA GeForce RTX 5070 Ti，16 GB：
  - Windows 回報的 V_total 為 16,303 MiB；
  - driver 617.42，power limit 為預設的 300 W。
- 主機：AMD Ryzen 5 7600、32 GB RAM；Windows 主機上的 WSL2 Ubuntu-22.04。
- 這張卡同時驅動 4K 桌面。session 期間的閒置 VRAM 為 567–568 MiB。
- HAGS 開啟（`HwSchMode` = 2），`.wslconfig` 為 memory 31 GB、swap 8 GB。這是 Stage B 實測的平台設定（PREREGISTRATION §4），Stage C 的 session 沒有重新擷取。
- 每次 boot 前都以 031 的閒置判定與 spill 偵測檢查；GPU 上沒有其他工作。
- 單一 GPU，單一 vLLM 執行個體。

**軟體與模型**
- vLLM 0.30.0、torch 2.13.0+cu130。
- 模型：`RedHatAI/Qwen3.5-9B-quantized.w4a16`，revision `a398088c`（W4A16），`--language-model-only`，dtype bfloat16。
- Core B 的 server argv，原樣不變：
  - `--max-model-len 32768`、`--max-num-seqs 128`、`--max-num-batched-tokens 2048`；
  - `--attention-backend FLASH_ATTN`、`--no-enable-prefix-caching`、`--async-scheduling`、`--performance-mode balanced`；
  - `--gpu-memory-utilization 0.91`、`--kv-cache-memory-bytes 4294967296`（248 個 block，每個 528 token）；
  - GDN prefill 與 decode 都用 triton；pin memory 關閉（WSL2）。
- 產品端以 server fingerprint 核對，每次 boot 都檢查。

**SLO（S2，使用者選定）**
- 短類 request 的 P95 TTFT：2K 為 2 s、8K 為 6 s；全部 request 的 P95 TPOT 為 100 ms。
- mixed 中的 32K 長 request 也要符合 20 s 的 TTFT（budget）。
- 時間窗：穩態，序號 [C, N − C) 的 request。
- **PASS**：每個關鍵點要 5 次獨立 boot（c_low 為 3 次），每次的 R（指標 / 門檻）都 ≤ 0.95，也就是留出 5% 的 guard band。
  - 符合 SLO、但落在 guard band 內的點是 MARGINAL，不能算進 C_PROD。
- 估計量向下封閉：上限之下的每個並發數都要成立。

**Workload 與測試方式**
- **uniform 2K / 8K**：每個 request 的 prompt 為 1,920 / 8,064 token（random dataset，range ratio 0），輸出固定 128 token（`--ignore-eos`）。
- **mixed 2K / 8K**（007 的 mix）：90% 為 1,920 / 8,064 token、10% 為 32,640 token 的 prompt，輸出 128 token。
- **封閉迴路**：`vllm bench serve --max-concurrency C --request-rate inf`，temperature 0，seed 20261008，每個 run N = max(200, 10·C) 個 request，warmup 數等於 C。
- **TTFT 從 request 送出起算**：不含 client 端排隊（准入前）的等待，也就是准入之後的延遲。
- **產品對應**：每個 class 的 `max_concurrency` = `max_queue` = C_PROD，prompt 上限依 workload 而定，輸出上限 128 token。

**不在驗證範圍內**
- 開放迴路或 Poisson 到達、輸出長度不是 128。
- 其他 prompt 長度分布：
  - uniform 只驗證了剛好 1,920 / 8,064 token，class 接受較短的 prompt，但沒有驗證；
  - mixed 只驗證了 90 / 10 的 mix，雖然 class 接受任何 ≤ 32,640 token 的 prompt。
- 經過 capserve gateway 的流量：容量是直接對 vLLM 的 `/v1/completions` 量的，不含 gateway 的開銷、chat template 與 `/tokenize` 呼叫。這是 029「待決」中列出的未驗證假設。
- **開場湧入**：C 個 request 同時到達時，uniform 兩類第一波的 TTFT 最高 10.35 s（2K）與 11.04 s（8K），遠高於 SLO；mixed 兩類在 SLO 內。
- 准入之前的排隊時間。
- prefix caching、其他 GPU / driver / vLLM 版本 / 模型、多 GPU，以及 Mac 平台。
- 長時間的漂移：guard band 沒有針對它校準；session 內的 sentinel 漂移為 0.19%。

## 產品整合測試的結果與限制（029 最後的補記）

| 測試（release `7513d62`；最終 `5201760` 重跑了 verified 的 IT-0、1、2、4、5、6 與 provisional 的 IT-0、3、6，結果相同） | verified profile（`rtx-verified.toml`） | provisional-rtx（同一 release） |
|---|---|---|
| IT-0、IT-1、IT-2、IT-4、IT-5、IT-6 | 通過 | 通過 |
| **IT-3（client 斷線）** | **無法判定，不算通過** | 通過（決定性） |

- **verified profile 下的 IT-3**：
  - 第一次執行（`pt-20261017T184411Z-itest`）記為不通過。斷線後 vLLM 的 running 與 waiting 都回到 0，也沒有再生成 token。
  - 但 128 token 的輸出上限讓斷線時只剩約 1.0–1.4 s 的生成量，少於 IT-3 要求的 4 s，所以無法區分「被 abort」和「本來就跑完」。
  - 之後以 verified profile 執行時不跑 IT-3；結果記為無法判定，不是通過。
- **IT-3 的決定性驗證**：只來自 provisional-rtx（1,024 token 的輸出上限），在同一個 release、同一次 lock 中通過。
  - 斷線處理是 gateway 的程式，和 profile 無關，所以可以推論它在 verified 設定下行為相同。
  - 但這是推論，在 128 token 下沒有直接的決定性證據。
- IT 是產品功能的測試，不是容量量測，輸出不在 `results/`。

## 正式 profile 的保護（2026-10-19 新增，TDD）

既有的保護：
- provisional profile 要 `--allow-provisional` 或 `CAPSERVE_ALLOW_PROVISIONAL=1` 才能載入；
- verified profile 必須是 release 中 `service/profiles/<name>.json`、而且列在 manifest 中的原始位元組；
- 每份 commit 過的 verified profile 都要能由產生器重新產生，除了 generator_commit 與涵蓋它的 hash 之外逐位元組相同；
- unavailable 或 verified 0 的 class 不能作為 active class。

**收尾時發現的缺口**
- 正式部署設定沒有指定要用哪一份 profile。
- 如果 shell 留著 `CAPSERVE_ALLOW_PROVISIONAL=1`，或誤傳一份 class 名稱也叫 `uniform-2k` 的 provisional profile，它就可能以正式設定的名義被部署。

**新增**
1. **`service/deploy/rtx-verified.toml` 是正式設定**，以 `[deploy] production_profile` 釘住 `service/profiles/verified-formal-20261017-c1b.json`。
   有釘選時，`capserve.deploy up`（包括 `--loadtest`）與 `capserve.itest` 都會拒絕下列情況，而且在任何東西啟動之前拒絕：
   - 不是這份檔案的 `--profile`，包括另一份 verified profile；
   - 任何允許 provisional 的執行（`--allow-provisional` 或 `CAPSERVE_ALLOW_PROVISIONAL=1`）。
   `production_profile` 本身必須是 `service/profiles/<name>.json`。
2. **provisional profile 不得使用 Stage C 的 class 名稱**（uniform-2k、uniform-8k、mixed-2k、mixed-8k），id 也不得以 `verified-` 開頭。
   就算帶了 `--allow-provisional` 也一樣拒絕。
3. **`capserve.itest` 的預設 profile**：用正式設定時是釘住的 verified profile；用整合測試設定（`rtx.toml`，沒有釘選）時仍是 provisional-rtx。
4. **`python -m capserve serve` 是 gateway 的低階進入點**，供開發使用，不讀部署設定。正式部署只用 `capserve.deploy up --config service/deploy/rtx-verified.toml`。
   直接用 `serve` 載入 provisional profile 時，`/readyz`、`/metrics`（`capserve_profile_verified 0`）與每個 response 的 header 都會標示 provisional。

**測試**：先寫失敗的測試，再實作。在 deploy、itest、profile 新增 29 個測試案例（結案 commit 22 個，審查後再加 7 個），其中包括 repo 設定的檢查：
- 正式設定釘住最終 profile，而且 active class 可服務；
- 已 commit 的 provisional profile 沒有使用保留的名稱或 id。

## 結案時的狀態

- 研究程式、`configs/capacity.toml` 與所有正式 session 的結果都沒有修改。凍結的位元組由 `test_stage_c_frozen`、`test_frozen_untouched` 與 `test_core_untouched` 保護。
- 沒有任何正式 GPU 實驗在進行或排程。
- 釘選之後的最終 release 也在 RTX 上做了產品測試，見文末「結案紀錄」。

## Phase 3 的補記（2026-10-23，decision 037 / 038）

- 本文「verified profile 下的 IT-3 無法判定」的限制已由 038 §2 解除：IT-3 改以生成的 token 數判定。
- release `c9202c5` 以 `rtx-verified.toml` 執行 IT-0 至 IT-6，全部通過。IT-3 為決定性：串流生成 8 / 128 個 token，非串流 49 / 128（`pt-20261022T170336Z-itest`）。
- 本文其他內容，包括正式容量、適用條件與 profile 的釘選，都不變。

## 後續（由使用者另外決定）

- **產品化**：v1 項目（029「里程碑」），例如多 class 或 token 加權的准入、in-process tokenizer、adaptive Retry-After、quota、TLS proxy；常駐服務也需要使用者決定。
- **研究**：
  - 開放迴路或 Poisson 到達下的容量；
  - 其他輸出長度；
  - 開場湧入的處理；
  - O-32K、O-T；
  - 8K 滑移的觸發點；
  - 026 列出的 Phase 1 Future Work。
- **環境**：再跑正式 session 之前，建議使用者停用 BingWallpaper 的自動啟動（033 / 034）。這屬於系統設定，我不修改。

## 結案紀錄（2026-10-19）

**RTX 產品測試**：最終 release `5201760`，在同一次 GPU lock 中，01:16–01:19 執行。之後的 commit 只改文件。這是產品測試，不是研究實驗。
- **正式設定的拒絕路徑**：以 `rtx-verified.toml` 執行 `capserve.itest` 時，下列三種情況都以 exit 2 結束，沒有啟動任何東西：
  - 帶 `--allow-provisional`；
  - 只設定環境變數 `CAPSERVE_ALLOW_PROVISIONAL=1`；
  - 指定 `--profile service/profiles/provisional-rtx.json`。
- **verified profile**：沒有指定 `--profile`，預設就載入釘選的 `verified-formal-20261017-c1b.json`。IT-0、IT-1、IT-2、IT-4、IT-5、IT-6 全部通過（`pt-20261018T171607Z-itest`）。
- **provisional-rtx**：IT-0、IT-3、IT-6 通過，IT-3 是決定性的（`pt-20261018T171748Z-itest`）。
- **verified profile 下的 IT-3**：仍為無法判定，不算通過（見上文）。

**結案時的檢查**
- **RTX**：
  - 沒有殘留的 vLLM 或 capserve 程序，也沒有人持有 GPU lock；
  - 服務的 state.json 為 stopped；
  - 研究目錄仍是部署 `b3c9109`，216 個檔案和 manifest 一致。
- **Mac**：
  - 完整測試 2,310 個通過（5 個 skipped）；
  - 從 `7513d62` 到結案，`results/`、`configs/`、`benchmarks/`、`quality/`、`data/` 都沒有變更；從凍結 commit `b3c9109` 起，`benchmarks/` 與 `configs/` 也沒有變更；
  - 工作區乾淨，所有 commit 都已 push 到 private 的 origin/main。
