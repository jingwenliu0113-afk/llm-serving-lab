# 049 — Stage Q3 的結果（formal-20261031-q3）

- 狀態：**結果**。依 040 凍結的 Q3 規則執行（`freeze.stage_q3 = true`、`pilot_q3 = "pilot-q3-20261025b"`、`pin_memory` "1"，harness `2bc3f2ee…`，2026-10-25 凍結），凍結之後沒有改過任何規則、門檻或判定方式。
- 日期：2026-10-31
- 對應：040（Q3、實作補記 7、8、14、15、Q3 的審查紀錄、pilot 紀錄與使用者對 pin_memory 的決定）、048（Q1 / Q2 的結果，GGUF 的 PPL）、PREREGISTRATION §10。
- 輸出（commit `173cb8f`）：`results/stage_q/q3-analysis-20261031.json`，由 `quality/stage_q3.py analyze` 在 Mac 上產生（raw/ 已由 `tools/rtx/fetch_raw.sh` 取回並以 sha256 核對）。session 的檔在 `results/stage_q/formal-20261031-q3/`（`3a6af53`）。
- **判定：H-Q3 not supported**。PPL(W4A16) ÷ PPL(BF16) − 1 = **+4.82%**，超過事先寫定的 2% 界線。040 的預測是 supported，**預測錯了**；凍結前的第二次 pilot（10 段）已經預示這個方向（見「限制」）。

## 執行

- **pilot 與使用者的決定**（040 pilot 紀錄）：
  - 第一次 pilot `pilot-q3-20261025`（pin_memory "0"）：BF16 job 在 vLLM 的 profile run 失敗（GDN 的 Triton kernel 收到 CPU 上的張量；原因是推論），屬於 040 預先列出的可行性風險。
  - 因為改用 "1" 會讓整台 Windows 主機（也是另一個專案的機器）承擔 WDDM pinned memory 的風險，由使用者決定。使用者（2026-10-25）決定試一次 "1"、兩個 job 都用，條件是只在 GPU lock 空著、主機可用記憶體 ≥ 20 GB 時跑 **pilot**；之後不論成敗都不再換設定，失敗就記為 NT（取代 040 第 77 行對 pilot 無法執行時的 NT 處理；依據是審查紀錄「pilot 若需要 "1"，再補記」）。
  - 第二次 pilot `pilot-q3-20261025b`（pin_memory "1"）兩個 job 都 ok，以此凍結；正式 session 沿用凍結的 "1"。正式前比照 pilot 的記憶體條件做了檢查，這是我自己沿用的操作規則（NEXT），不是使用者另外的決定。
- **正式 session `formal-20261031-q3`**（GPU lock，部署 `88936bb`）：
  - 2026-10-31 00:59:41–01:26:09Z（約 26.5 分鐘）。NEXT 依 pilot 的 wall time（10 段）線性外推約 5 小時（BF16 3.5 h、W4A16 1.6 h）；pilot 每個 job 的時間主要是固定的啟動開銷（載入、編譯、CUDA graph 等；推論，pilot 的 log 沒有取回），正式 log 中 W4A16 載入 36 s、init engine 129 s，BF16 載入 155 s。
  - 開始前（約 00:59Z，建立 plan 的同一個指令中）在 WSL 執行 `free -g`：available 29（GiB，WSL VM 內的值，不是 Windows 主機的讀數；指令輸出沒有 commit）。
  - WSL 存取與 session 的先後：部署 `88936bb` → E2 dataset 產生（vLLM 的 log 時間 08:59:15 +08:00，即 00:59:15Z）→ 建立 Q3 plan 與 `free -g` → 啟動 Q3（`started_at` 00:59:41Z），依序在同一個 shell 中執行；session 期間我沒有存取 WSL。NEXT 原本寫的「約 01:01Z 起」有誤，已更正。
  - outcomes：2 / 2 ok（W4A16、BF16），`env_valid` true，`env_flags` []；第一次也是唯一的正式嘗試（`results/stage_q/wsl-inventory.txt`，01:26:23Z）。
  - 5 份 env 擷取都記錄 `DEPLOYED_COMMIT` `88936bb`、dirty false；vLLM 0.30.0、torch 2.13.0+cu130、driver 617.42。
  - 設定（plan 與 server log）：兩個 job 都是 `VLLM_WSL2_ENABLE_PIN_MEMORY` "1"；`cpu_offload_gb` 只有 BF16 為 8（GiB；log：卸載 8.06 GiB 參數）；W4A16 為 0。兩個 job 都是 FLASH_ATTN、GDN prefill 用 Triton、prefix caching 關閉。
  - BF16 的 log 沒有第一次 pilot 那行「Pinned memory is not available」；vLLM 印出兩次 WDDM pinned memory 的警告。結束時 vLLM 在 SIGTERM 後 5 秒強制結束 BF16 的 EngineCore，exit code 仍為 0，分數檔與 manifest 相符。BF16 job 結束後的 env 擷取（01:26:08Z）GPU 佔用 235 MiB（session 開始時 245 MiB）。
  - 描述（沒有解釋）：這次的 KV cache 容量（W4A16 59,596、BF16 69,427 token）比 pilot（35,635、45,670）多約 24k token，設定相同；pilot 的 raw log 沒有取回，無法比較原因。兩次共同的 10 段（chunk 0–9）上，兩個 checkpoint 的逐段 NLL 和 pilot 逐位元相同，所以至少在這 10 段上容量的差異沒有改變分數；其他 570 段沒有直接對照。vLLM 在 WSL 內記錄的 `Available RAM`：W4A16 job 27.36 GiB（01:00:57Z）、BF16 job 16.56 GiB（都是執行中的讀數）。
- **輸入**：580 段 512-token 的 chunk（`chunks_sha256` `7174626e…`，pin 在 `configs/stage_q3.toml`；session、plan 與兩個 row 都記錄同一個值），每段計分後 255 個 token（位置 257–511），共 147,900 個 token，兩個 checkpoint 相同。
- **checkpoint**：W4A16 為 Core B 的 `RedHatAI/Qwen3.5-9B-quantized.w4a16@a398088c`（只核對 revision，沒有核對 sha256；040 審查 #21）；BF16 為 `Qwen/Qwen3.5-9B@c2022362`（每個 shard 的 sha256 都核對過）。W4A16 的 base 是從 recipe 推斷為 `Qwen/Qwen3.5-9B`，base 的 revision 沒有核對（pins 的註記）。

## 結果

| 項目 | 值 |
|---|---|
| PPL W4A16 | 8.5810 |
| PPL BF16（CPU 卸載 8 GB） | 8.1865 |
| PPL(W4A16) ÷ PPL(BF16) − 1 | +4.82%（界線 2%）→ **not supported** |
| 逐段配對 ΔNLL（W4A16 − BF16）平均 | 0.04706 nat / token |
| ΔNLL 的 95% CI（以 chunk 為單位 bootstrap 10,000 次，seed 20261008） | [0.04469, 0.04951]，換成 PPL 比為 +4.57% 到 +5.08% |
| top-1 一致率 | 0.9029（147,900 個位置中 133,537 個相同） |

- **ΔNLL CI**：整個區間都高於 ln(1.02) = 0.0198（下界約為它的 2.3 倍），chunk 間的抽樣變異不足以解釋超出界線。H-Q3 依點估計的 PPL 比對照實質門檻判定（040 實作補記 8），不是統計檢定；CI 只是附加資訊。CI 只反映這份 WikiText-2 test 的 chunk 間變異，不含其他資料或重跑之間的變異（每個 checkpoint 只跑一次；同設定的 pilot 與正式 session 在共同的 10 段上逐位元相同）；相鄰 chunk 是連續文字，CI 可能偏窄。
- **逐段分布**（描述）：580 段中 555 段（95.7%）ΔNLL > 0、25 段 < 0；中位數 0.0457，範圍 −0.0526 到 0.2688。
- **top-1 一致率**：兩個 checkpoint 的 rank-1 token 是否相同（約 9.7% 的位置不同），不是對實際 token 的準確率；沒有門檻，只作描述。`prompt_logprobs` 為 1 時，兩者都不是實際 token 的並列依 torch.topk 的順序決定，只影響這個描述量（040 實作補記 15）。

### 和 GGUF 的 PPL 並列（040 的報告項目；不同 runtime，只作描述）

同一份 WikiText-2 test、相同的 512-token chunk 與計分位置；GGUF 的值來自 048（llama.cpp b11146）。

| runtime | 設定 | PPL | 相對同 runtime 的 BF16 |
|---|---|---|---|
| vLLM 0.30.0（WSL2 CUDA） | BF16（CPU 卸載） | 8.1865 | — |
| vLLM 0.30.0（WSL2 CUDA） | W4A16 | 8.5810 | +4.82% |
| llama.cpp（Metal） | 9B BF16 / Q8_0 / Q6_K / Q4_K_M | 8.1865 / 8.1935 / 8.2113 / 8.3054 | — / +0.09% / +0.30% / +1.45% |
| llama.cpp（CUDA，Phase 1） | 9B BF16 / Q8_0 / Q6_K / Q4_K_M | 8.1862 / 8.1950 / 8.2139 / 8.1983 | — / +0.11% / +0.34% / +0.15% |

- vLLM 的 BF16 PPL（8.186505）和 llama.cpp Metal 的 9B BF16（8.1865）在印出的精度內相同，和 CUDA 的 8.1862 相對差約 3.7e-5。這是描述，不證明兩個 runtime 等價。
- 040 只登記並列描述，沒有登記兩者之間的優劣比較；兩個 runtime 的量化方式也不同，所以**不**寫成「W4A16 比某個 GGUF quant 好或差」。

## 解讀（只寫 040 允許的部分）

- 040 的預測理由是事前的一般性判斷（「4-bit 權重量化在 9B 規模，常見的 PPL 增幅低於 2%」），沒有附出處。這個 checkpoint 在這份資料上的增幅是 4.82%，預測不成立。
- 結果只描述這一組 checkpoint（RedHatAI 的 W4A16 對 Qwen 的 BF16）在這個設定下的 quality；不推論原因（例如校準資料、GDN 層或量化 recipe），也不說 W4A16「不可用」或「有缺陷」。
- **不改變其他判定**：040 規定「不重新分析：H-A5 與 Core A 的判定不變」；PREREGISTRATION §10 寫明第 1–9 節不變，所以 Core B 的判定也不變（Core B 用的就是這個 W4A16 checkpoint；Q3 只是它的新 quality 背景）。048 的 Q1 / Q2 判定也不受影響。

## 限制

- 只有一組 checkpoint；W4A16 只核對 revision，它的 base revision 沒有核對，不能確定是從 `c2022362` 量化的，所以結果是這兩個 checkpoint 的差，不只是量化的效果。
- 只有 WikiText-2 test、512-token 的 chunk（計分後 255 個 token）；沒有較長的 context，也沒有 downstream 題庫（040 的範圍）。
- vLLM 0.30.0、WSL2 上的 RTX 5070 Ti；BF16 以 `--cpu-offload-gb 8` 執行，沒有測試卸載或 pin_memory 是否改變 BF16 的數值（pin_memory "0" 的 pilot 失敗，沒有對照）。W4A16（沒有卸載）在兩次 pilot（"0" 與 "1"）的 10 段上，NLL 有 8 段不同，最大差 1.7e-8 nat，相對 ΔNLL 0.047 可以忽略。
- 每個 checkpoint 只跑一次。
- **登記前與執行前的資訊**：
  - 檢查第二次 pilot 時，兩個 checkpoint 在 10 段上的逐段 NLL 被印出並看過（040 pilot 紀錄）。那 10 段（chunk 0–9）上 W4A16 的 PPL 已比 BF16 高 4.24%，10 段的 ΔNLL 都大於 0，所以凍結時已經預示 H-Q3 不成立。門檻（≤ 2%）、預測與判定規則在 2026-10-23 就寫進 040 與實作補記 8，早於 pilot（10-25）、凍結（10-25）與正式 session（10-31），沒有因此修改。
  - 040 pilot 紀錄（與凍結 commit `8a02e68` 的訊息）寫「pilot 的 10 段不進入正式分析」並不準確：pilot 的 row 不進入分析，但同樣的 chunk 0–9 在正式 session 被重新計分，值和 pilot 逐位元相同，並包含在 580 段中。040 已補上更正。
- raw/ 的逐 token 分數含 rank-1 token id，可能部分還原原文，只放在本機；git 只有 sha256（040 實作補記 8）。

## 能支持與不能支持

- 能：在 vLLM 0.30.0、RTX 5070 Ti（WSL2）、WikiText-2 test 的 512-token chunk（計分後 255 token）上，`RedHatAI/Qwen3.5-9B-quantized.w4a16@a398088c` 的 PPL 比 `Qwen/Qwen3.5-9B@c2022362` 的 BF16（8 GiB CPU 卸載）高約 4.8%（chunk bootstrap 95% CI +4.6% 到 +5.1%，只含 chunk 間變異），每個 checkpoint 一次 run，超過事先寫定的 2% 界線。
- 不能：
  - 推廣到其他 W4A16 / 4-bit checkpoint、量化方式、資料集、context 長度或 runtime；也不說 9B 的 4-bit 量化「一般」都超過 2%。
  - 和 GGUF 的 quant 比較優劣。
  - 推論原因，或改變 Core A、Core B、H-A5、Q1 / Q2 的判定。

## 數字來源

- **q3-analysis-20261031.json**：`session`、`q3.{attempts, superseded, status, void_before}`、`outcomes`、`ppl`、`ppl_ratio_minus_1`、`tokens`、`delta_nll.{mean, ci95, n_chunks, n_boot, seed}`、`top1_agreement`、`H-Q3`、`harness_sha256`、`stage_q3_sha256`、`pins_sha256`。
- **session 目錄**：`session.json`（時間、outcomes、env、`chunks_sha256`、`vllm_version`）、`plan.json`（每個 job 的 env、`cpu_offload_gb`）、`rows.jsonl`（`chunk_nll`、`n_tokens`）、`env/*.json`、本機 raw/ 的 server log 與 `score-*.scores.jsonl`（逐段分布、top-1 的計數、KV 容量、可用 RAM、強制結束的訊息）；`results/stage_q/wsl-inventory.txt`。
- **其他**：GGUF 的 PPL 取自 048；pilot 的 KV 容量取自 040 的 pilot 紀錄；pilot 的逐段 NLL 取自 `results/stage_q/pilot-q3-20261025{,b}/rows.jsonl`；時間的推估取自 NEXT（更正前）；開始前的 `free -g` 與 E2 dataset 的 log 時間是當時指令的輸出（沒有 commit）。

## 獨立檢查

- 三個獨立的核對者，都沒有參與起草：
  - **重算**：不 import `stage_q3.py`，從 raw 的兩個分數檔（sha256 先和 manifest 核對）重算計分 token 數、兩個 PPL、比值、逐段 NLL 與配對 ΔNLL、bootstrap CI（用凍結的 `quality_analysis` bootstrap 函式，另以自己的 bootstrap 與常態近似作合理性檢查）、top-1 一致率與判定，和分析 JSON 全部相同；rows 的 `chunk_nll` 與 raw 逐位元相同；兩個 checkpoint 計分的是同樣的 580 段。chunk 檔本身不在 Mac 上，只核對各處記錄的 sha256 一致。
  - **重跑與來源**：以和 `main()` 相同的輸入呼叫 `analyze_q3`，輸出和 commit 的 JSON 逐位元組相同；harness、設定檔與 pins 的 sha256 在凍結、部署與目前的 commit 都相同；凍結證據、嘗試、plan 與 env 擷取如「執行」所述；指出了 KV 容量與 pilot 的差異和強制結束的訊息。
  - **解讀**：依 040 確認判定，並列出本文必須寫與不能寫的項目（都已納入「解讀」「限制」與「能支持與不能支持」）。
- 核對腳本放在 session 的 scratchpad，沒有 commit。
- **核對後的更正**（兩位審查者，一位核對數字與事實、一位核對說法與 040 的規則；修訂時逐項再核對）：
  1. 補上第二次 pilot 看到的內容（10 段的 PPL 比 +4.24%，凍結前已預示不成立）與正式 session 第 0–9 段和 pilot 逐位元相同；040 的「pilot 的 10 段不進入正式分析」更正為 pilot 的 row 不進入分析。
  2. 使用者決定的範圍是 pilot；正式前的記憶體檢查是我自己沿用的操作規則。29 GB 改為寫明是 WSL 內 `free -g` 的值與時間。
  3. 寫出 WSL 存取（部署、E2 dataset、plan、`free -g`）都在 session 開始之前，並更正 NEXT 的開始時間。
  4. 補上 W4A16 的 base revision 沒有核對；「能支持」加上範圍與 revision；pin_memory 的對照改為實際的 W4A16 pilot 差異；KV 容量的影響改用 10 段的直接證據。
  5. 小處：offload 的單位是 GiB；BF16 PPL 只和 Metal 在印出精度內相同；GPU 佔用的時間點；時間推估的出處；CI 的措辭；並列描述與優劣比較的區分；Core B 判定不變的依據改為 PREREGISTRATION §10。
