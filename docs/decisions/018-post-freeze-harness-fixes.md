# 018 — Stage A 凍結後的 harness 修正（第三輪審查）：讓程式落實已凍結的規則

- 狀態：Accepted（2026-10-09）
- 日期：2026-10-09
- 時點：Stage A 已於 2026-10-09 凍結（harness commit `eb6eef1`）。目前**沒有任何正式數據**：
  - 正式 Mac Core A 依使用者指示暫停。
  - 正式 quality 卡在磁碟空間。
  - Core B 卡在 `--max-num-seqs`。
  - RTX Core A 尚未開始。
- 來源：對凍結前後新寫程式的審查 workflow。四個面向各一名 reviewer，每項發現另由一名對抗式驗證者複查。
  結果確認 18 項、駁回 11 項；其中 1 項被駁回的我仍照實處理，見下文第 7 點。
- **性質：所有修正都是讓實作符合已凍結的文字**（§4、003、007、012、016、017）。不改任何 hypothesis、metric、門檻、task set、
  資料切分或 protocol 數字。依 006，凍結後的任何變更都要記錄；本文就是這份紀錄。

## 修正（依區塊）

### 1. Mac 的正式 run（§4、004、012）

- §4 規定「所有正式 run 的 `powermode` 都必須相同（目前為 2 = high power）；`iogpu.wired_limit_mb` 必須為 0」。
  env_capture 一直有記錄這兩個值，但從未檢查。
  - 現在 `env_problems`（macos-metal）會在每個 process 前後檢查。
  - 正式 Core A 與正式 quality 的 preflight 若不符，在開跑前就拒絕。
  - 讀不到值時同樣視為無效（013）。
  - 目前這台 Mac 是 powermode 0（「自動」），與使用者今晚的說明一致，正式 run 本來就暫停中。
- §4 的「正式 session 的 seed 一律為 20261008」、Mac 的「N_thr = 8」與「P = 0」，原本沒有任何程式擋：
  - `core_a_plan` 現在只為 formal 產生 seed 20261008 的 plan。
  - Mac 的 N_thr / P 必須是 8 / 0。
  - RTX 的 N_thr / P 取自 pins `[stage_b]`；P 決定之前一律拒絕。
  - `core_a_report` 也會核對 seed。

### 2. RTX 的有效性（§4）

- 「`enforced.power.limit` 必須等於 `power.default_limit`」：`env_problems`（wsl2-cuda）現在會檢查，讀不到值同樣視為無效。
  RTX 沒有電池，仍然不檢查電量。
- 「之後每個 session 開頭重量一次 B；超過凍結的 B 0.25 GiB 以上 → drift」：
  - 正式 RTX Core A session 開頭必須用 `stage_b.measure_baseline` 重量 B，結果寫入 `session_b`。
  - `core_a_report` 拒絕 drift 的 session。
  - Core B 的 runner 原本就有「GPU memory.used > 1000 MiB 不啟動」的檢查（比 811 + 256 = 1067 更嚴）。

### 3. Core B 的 server 與 Stage B 值（§4、§6、003、007、017）

- 新增 pins `[stage_b]`，集中存放 §6 的實測值與執行參數：
  - B = 811 MiB、V_total = 16,303 MiB、U = 0.91、GDN prefill backend = triton（S1–S3 通過時用的）、RTX N_thr = 6。
  - P、C_ctx、K、num_gpu_blocks、block_size、pin-memory 選定值仍待量測。
- `server_env` 會移除從 shell 繼承來的 `VLLM_*` / `CUDA_*` / `GGML_*`（§4：除非本文件明列，一律不設定），
  並依 pins 設定 `VLLM_WSL2_ENABLE_PIN_MEMORY`（未 pin 時為 vLLM 預設的 0）。
  原本 A/B 的選擇完全沒有套用到正式 server。
- `core_b_plan uniform` 改為從 `[stage_b]` 產生 plan。原本的 CLI 無法帶入 K / U，產生的 plan 一定會被正式 run 拒絕。
- 正式 `run_core_b`：
  - U、K、GDN backend、pin-memory 必須等於 `[stage_b]` 的值。
  - 新增 env_capture（preflight 與每次 restart 前後），並核對 vLLM 版本與 power limit。
  - session 記錄每次 restart 的 server argv、repo commit、被移除的繼承變數。
- **log 驗證**（§4「以 log 驗證 attention backend 與 max_num_batched_tokens」）：
  - server 就緒後，核對 `Using AttentionBackendEnum.FLASH_ATTN backend.`，以及 non-default args 中的 `'max_num_batched_tokens': 2048`。
  - 不符就停止該次 restart。
  - **017 原本指定的 log 句 "Chunked prefill is enabled with max_num_batched_tokens=…" 在 0.30.0 的 server log 中不會出現**
    （`info_once`），因此改用上述兩句。這只是更正驗證來源的寫法，要驗證的事不變。
- `core_b_report` 會拒絕「`mixed` session 的 server 設定或 server env 與 `uniform` session 不同」的組合（007：所有 mix 共用同一個 server 設定）。
- `core_b_figures` 只畫三次 restart 都有效的點，並在註記中列出被排除的 cell 與無法檢驗的 L。

### 4. Stage B 工具

- 在等 `/health` 時收到 SIGTERM / SIGHUP / Ctrl-C，原本會跳過 teardown。
  vLLM server 於是在 lock 釋放後仍占著約 U × V_total 的 VRAM。現在 `_boot` 遇到任何例外都會先關掉自己啟動的 server。
- baseline：
  - 閒置期間每 5 秒檢查一次 GPU process。出現 process 就重新計時，符合 §4「最後一個 GPU process 結束後閒置 5 分鐘」。
  - 每次取樣前也檢查一次。
  - 可傳入凍結的 B，以判定 drift。
  - HAGS 與 `.wslconfig` 的讀取錯誤分開記錄，「沒有檔案」與「讀不到」不再混淆。
- boots 的「冷啟動」只重置 Triton / FlashInfer / vLLM 的 cache。CUDA driver 自己的 PTX JIT cache 沒有重置，
  因為 `CUDA_CACHE_PATH` 不在 §4 明列的變數中，所以冷啟動時間不一定包含 FA2 的 compute_80 PTX JIT。
  結果檔會照實記錄這一點；不改 protocol。

### 5. GGUF 傳輸腳本（只是工具，與實驗無關）

- 重試之間加入退避等待。連線中斷時回傳 75，交給排隊腳本稍後再試。
- 遠端的 sha 檢查失敗時，不再當成「檔案都不在」。
- 接收端改用 `set -eu` 與 `mv -T`：mkdir、cd、mv 任一失敗都會停下，不會誤報 VERIFIED。

### 6. 只參數化、不改數字：016 的充電門檻與正式 quality

- 正式 quality 中，9B 的 MC job 長 38–59 分鐘（以 pilot 換算）。
  以 016 自己估的每小時 25–40% 耗電，從 40% 開始跑可能在 job 結束時跌破 20%，使整個 session 作廢。
- 改 016 的 40% 屬於修改凍結的 §4，所以**不自行改動**。`charge_gate` 只做參數化，預設值不變；交由使用者決定，記在 NEXT 的 Blocked 1：
  - 換 ≥ 96 W adapter：最簡單，不需要改任何規則。
  - 或同意為 quality 補一則 016 addendum，例如暫停門檻 75%。

### 7. 照實標註：014 補充是在 Stage A 凍結之後才寫的

- §4 寫的是「`CUDA_HOME`（user-space shim，指向 vLLM lock 中的 pip CUDA 13）」。
- 014 補充（2026-10-09 01:45，凍結於 01:15）改為指向另一份 hash-locked 的 CUDA 13.0（`configs/flashinfer-cuda.lock.txt`）。
  原因是 vLLM venv 中 nvcc 13.4 與 cudart 13.0 headers 不一致，FlashInfer 內附 CCCL 的版本檢查會直接 `#error`。
- 這是**凍結後**的變更，但發生在任何 Core B 數據之前，也不涉及任何結果。審查驗證者認為它不構成缺陷而駁回了這一項，但依 006 仍應標明 post-freeze。
  §4 該句已加上註記。

## 對實驗的影響

- 規則本身都不變。正式 run 會被更嚴格地擋在規則之外，例如 powermode 0 時不能開始正式 Mac run。
- PREREGISTRATION §4 只加註記，指向本文與 014 補充；不改原文。

## 補充（2026-10-09 07:3x）：凍結後又發現的兩處程式與規則不一致

兩處都只修程式，規則不變。

- **Stage B 凍結測試假設 C_ctx > 0**
  - `tests/test_pins.py` 要求 `c_ctx_mib > 0`，原意是「已量測」（pins 用 0 代表還沒量測）。
  - 但 004 / 009 的定義是峰值 − B − self，本來就可能為負。實測值 −89 MiB 已照實記在 §6。
  - 不修的話，Stage B 凍結時測試會擋下一個合法的實測值；只有在「重量 C_ctx 直到得到正值」時才會通過，而那等於挑選結果。
  - 改成只把佔位值 0 視為未量測（`stage_b_unmeasured`）。順帶把 `block_size` 與 U 也納入凍結檢查。
  - 分析程式（`core_a_analysis` 的 spill 記憶體條件）原本就能處理負值。
- **pilot report 的正式 plan 計數**
  - 018 讓 `build_plan` 拒絕非 §4 的正式 seed，以及 P 未定時的 RTX plan。
  - pilot report 只為計算 job 數就建立了這樣的 plan，所以在 RTX pilot 之後失敗。
  - 已改用 `formal_job_counts`；job 集合不受 seed / N_thr / P 影響（見 012 的 RTX 補充）。

