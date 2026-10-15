# 027 — Phase 2：Research（Fixed-SLO 容量）與 Engineering（Capacity-Aware Serving Service）受控並行

- 狀態：Accepted（使用者決定，2026-10-15）
- 日期：2026-10-15
- 對應：SPEC「Advanced：fixed SLO 下最大 concurrency」、「Phase 2」（本文新增）；decision 023–026
- 時點：026 暫停之後、任何 Phase 2 數據或程式之前。Core 與 Advanced 的結果都已 commit（`4cb751f`）。

## 背景

- 026 把 Core 加上已完成的 Advanced 定為 milestone，Project 暫時完成，Future Work 由使用者決定。
- 2026-10-15，使用者正式重新啟動，採 Research 與 Engineering 兩條線受控並行，不再只做 benchmark：
  - **Research**：優先研究 Fixed-SLO 下的最大並發容量。先有限度排查 8K 延遲雙模式。
    正式實驗前，先預註冊並凍結 SLO、負載、指標與判定規則。既有的 Core / Advanced 結果保留。
  - **Engineering**：建立可部署的 Capacity-Aware LLM Serving Service。
    - 沿用 vLLM，不開發新的推論引擎，先做最小可用版本。
    - 功能：OpenAI-compatible API、部署管理、health check、metrics、超載保護，以及依研究結果制定的容量配置。
  - 兩條線可以並行開發，但正式 GPU 實驗必須獨占共享的 GPU lock。
    產品測試不得污染研究數據；尚未驗證的容量不得當成正式的產品限制。
  - 不使用付費 API，不影響其他專案，不擅自擴張研究範圍，GitHub 維持 private。

## 決策

### 1. 既有結果與凍結狀態不變

- PREREGISTRATION §1–§7、decision 001–026、`results/` 下所有既有 session 與 analysis、deployment decision table，一律不改、不重新分析。
- Core 的 6 個模組維持 `test_core_untouched` 的保護。
- 另外新增測試，把 Advanced 的分析與執行程式（`adv_*.py`、`run_adv.py`）鎖定在凍結 commit `97a948c` 的位元組。
  Phase 2 只能 import 它們，不能修改。

### 2. Research：Stage C（Capacity）

- 新增 PREREGISTRATION §8（Stage C）。它和 §7 一樣，是獨立的一節，前面各節不變。
- 範圍只有兩項：
  - **C0：8K 延遲雙模式的有限度排查（探索性，非正式）**。
    - 開始前另寫 decision log，事先寫定 boot 數、GPU 時數上限、診斷項目與停止規則。
    - 目的只有一個：決定 §8 的協定是否需要額外的控制，例如 restart 數、模式記錄或排除規則。
    - 排查結果不改變任何既有判定。
  - **C1：Fixed-SLO 最大並發**（正式）。SLO、負載、指標、搜尋或格點、統計與判定規則、容量的定義，
    都要在任何 C1 正式數據之前寫進 §8 與一份 decision log，並把 `freeze.capacity = true` 一起 commit 凍結（011、024 的模式）。
- **SLO 的數值屬於重大研究決策**：我會提出有依據的選項，由使用者選定後再凍結。
  - SLO 要依產品需求與使用情境訂定，不得依 Core 曲線挑一個好看的值。
  - Core 曲線已經公開過。格點範圍與事前預期可以參考 Core，但要在 §8 中揭露。
- 平台、模型與 runtime 沿用 Core B：RTX 5070 Ti、vLLM 0.30.0、`RedHatAI/Qwen3.5-9B-quantized.w4a16`、Core B 的 server 設定與 client。
  若要改變，要另寫 decision log。

### 3. Engineering：Capacity-Aware Serving Service（MVP）

- 產品程式放在獨立的目錄（`service/`），有自己的測試；研究的 harness 與分析程式不 import 產品程式。
  產品可以 import 研究模組中**唯讀的**部分，例如沿用 Core B 的 server argv 產生方式。
- 推論一律由 vLLM 執行。產品是 vLLM 前面的一層，負責：
  - OpenAI-compatible 的 proxy；
  - 准入控制與超載保護：並發上限、有界佇列、佇列逾時、429 / 503 加上 Retry-After；
  - health 與 readiness；
  - Prometheus metrics；
  - vLLM 的部署管理：用 pin 過的 argv 啟動、停止、檢查狀態，並 drain。
- 不寫新的 scheduler 或 engine，也不修改 vLLM。
- **相依套件**：只使用 `configs/vllm.lock.txt` 中已經 hash 鎖定的版本（fastapi、uvicorn、httpx、prometheus-client、pydantic 等）。
  - WSL 上直接使用研究的 venv（唯讀），不另外安裝。
  - Mac 上以同版本裝進 dev 環境，只用來跑單元測試與假後端的測試。
  - Mac 上不啟動 vLLM（MACHINE_POLICY）。

### 4. 容量配置：只有驗證過的容量能成為正式限制

- 容量設定檔（capacity profile）都標有狀態：
  - **verified**：只能由產生器從 Stage C 正式、有效、凍結後的 analysis 產生，並記錄來源 session、analysis 的 sha256、SLO 與 commit。
  - **provisional**：其他所有來源，例如手動設定、Core 的 C_knee、pilot。
- 產品的 production 模式拒絕以 provisional 設定檔啟動，除非明確加上開發旗標；這時 health 與 metrics 都會標示 provisional。
- 文件與 README 不得把 provisional 的數值寫成產品的容量限制。

### 5. GPU 使用與防止污染

- 任何會在 RTX 上啟動 vLLM 的工作都要取得共享 GPU lock（`flock -n /tmp/ai-portfolio-gpu.lock`）。
  包括研究的正式 run、探索性排查、pilot，以及產品的整合測試與負載測試。拿不到就等待或改做 Mac 上的工作，不得搶占。
- 正式研究 session 之間不插入產品測試。正式 session 開始前，env 擷取要確認 GPU 上沒有其他 compute process（沿用既有規則）。
- **產品測試的輸出不進 `results/`**：
  - 產品的輸出寫到 gitignore 的 `runs/`，並標記 `purpose: product-test`。
  - 研究的分析程式只接受研究 runner 產生、`formal: true` 的 session。
- **產品的 WSL 部署和研究的部署分開**：
  - 產品程式部署到另一個目錄，不改動研究目錄 `~/llm-serving-lab` 與它的 `DEPLOYED_COMMIT`。
  - 正式研究 session 進行中，不部署任何東西，包括產品。

### 6. 自主程度

- 開發、測試、review、commit、push、更新 NEXT.md 都自主進行。
- 下列情況才詢問使用者：
  - 重大研究決策，例如 SLO 的數值、超出本文範圍的研究項目；
  - 權限問題；
  - 真正無法解決的問題。
- 付費 API、其他專案、GitHub 可見度都不變（MACHINE_POLICY、記憶中的規則）。

## 為什麼

- SPEC 的 Advanced 已經列出「fixed SLO 下最大 concurrency」。Core B 只提供完整曲線，decision table 的「延遲敏感」一行目前給不出建議（026 的 Future Work）。
- 產品需要一個在 SLO 下仍然成立的並發上限，研究則提供一個可驗證的數字。兩者之間以 profile 的狀態把關，研究結果才不會被產品測試或未驗證的值污染。
- 只沿用 vLLM，避免違反 SPEC「不寫新的 vLLM」與 CLAUDE.md「不自己寫 serving engine」。

## 放棄的替代方案

- 直接把 Core 的 C_knee 當產品上限：C_knee 是吞吐量的 knee，不是延遲 SLO 下的容量（Core 的 P95 TTFT 在 knee 已是 C = 1 的 1.6–13 倍），而且沒有經過驗證。
- 產品測試與研究共用 `results/`：無法保證研究的分析不讀到產品的數據。
- 另建一個安裝了新套件的 WSL venv：會多出一組沒有 pin 過的環境。直接重用已經 hash 鎖定的套件比較安全。

## 對實驗的影響

- 下一步：
  1. C0 排查的 decision log；
  2. §8 的草稿與 SLO 選項，SLO 由使用者選定；
  3. 產品 MVP 的架構與實作（TDD）。
- 研究與產品的進度分開記在 NEXT.md。
