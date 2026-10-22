# 040 — Stage Q 事先登記（DRAFT）：quality 的三個缺口

- 狀態：**DRAFT**。harness、pilot 與審查完成後凍結；凍結前不收任何 Stage Q 的正式數據。
- 日期：2026-10-23
- 對應：037（Phase 3 範圍）、005（quality 指標與 task set）、020（quality 改在 RTX）、PREREGISTRATION §3 / H-A5
- 時點：任何 Stage Q 數據之前。設計者看過 Core A 的 quality 結果（README「Quality」），用到的地方在「揭露」寫明。

## 範圍

README「證據不支持的部分」列出三個缺口，本 stage 依序回答：

| 部分 | 問題 | 平台 | 正式數據 |
|---|---|---|---|
| Q1 | Core A 的 quality 是否不依賴平台（Metal 與 CUDA 是否相同） | Mac（macos-metal）與 RTX（wsl2-cuda） | KLD（前 200 chunks）與完整 PPL |
| Q2 | 長 context（2K、8K）下的 quality | RTX | KLD 與 PPL，-c 2048 與 8192 |
| Q3 | Core B 的 W4A16 checkpoint 的 quality | RTX（vLLM） | 同一 runtime 下 W4A16 對 BF16 的 PPL 與配對 ΔNLL |

**沿用 005 的規則**：
- WikiText-2 test（`wiki.test.raw`，pins 的 sha256）；llama.cpp b11146；005 的 flag set（只有 Q2 改 `-c`）；
- KLD 的流程：BF16 base → self-KLD → 3 個 quant → 刪除 base；
- 以正向標記加上預期數量判定成功（`quality_parse`）；
- KLD 的 z 檢定與 Holm 校正、報告 SE 的兩個方向的偏差。

**不做**：downstream 題庫在新平台或新 context 下的重測（005 中它們只是描述性的，9B 在 RTX 上就要 7 h）；寫在限制中。

## Q1：Metal 上的 quality

- **為什麼只用前 200 chunks 做 KLD**：Mac 剩約 97 GiB，完整的 KLD base 每個 size 約 73.5 GB，加上 50 GiB 保留量放不下（037 §4）。
  200 chunks 的 base 約 25 GB，一次只存在一個 size，算完即刪。
- **Mac 的工作**（每個 size，依序）：
  - Metal BF16 的 KLD base（前 200 chunks）→ self-KLD → Q8_0、Q6_K、Q4_K_M 的 KLD → 刪除 base；
  - 8 個設定（BF16 加 3 個 quant，兩個 size）的完整 PPL（580 chunks）。
- **RTX 的對照**：同樣的前 200 chunks，CUDA 的 KLD base、self-KLD、3 個 quant，兩個 size。
  這是新的 CUDA 數據，因為 Core A 的 KLD 是 580 chunks 的平均，不能直接和 200 chunks 比。
  完整 PPL 的對照是 Core A 已 commit 的 `formal-20261010-quality`。
- **條件**：Mac lock、Apple 140W adapter、同一 session 內電力模式一致（037 §4）、012 的電量與 thermal 規則；RTX 在 GPU lock 下。
- **Hypotheses**（預測寫在數據之前）：

| id | 內容 | 預測 |
|---|---|---|
| H-Q1a | Metal 上兩個 size 都是 KLD(Q8_0) < KLD(Q6_K) < KLD(Q4_K_M)（4 個 z 檢定，Holm） | supported |
| H-Q1b | 每個 quant 與 size：KLD_Metal ÷ KLD_CUDA（同樣的 200 chunks）在 [0.8, 1.25] | supported |
| H-Q1c | 8 個設定：PPL_Metal ÷ PPL_CUDA（580 chunks）在 [0.995, 1.005] | supported |

- H-Q1b 與 H-Q1c 的界線是事先寫定的實質門檻，不是統計檢定：兩個 backend 的數值差異，應遠小於量化本身的誤差。

## Q2：長 context 的 quality

- `-c 2048` 與 `-c 8192`（Core 的 context）。WikiText 的 token 數不變，所以計分的 token 數和 `-c 512` 相近：
  2K 為 145 chunks × 1,023，8K 為 36 chunks × 4,095（llama-perplexity 每個 chunk 計分 n_ctx − 1 − n_ctx ÷ 2 個 token；實作補記 2）。
  計分的是每個 chunk 的後半段，也就是較長的上下文之後的位置。
- 每個 size、每個 context：BF16 base → self-KLD → 3 個 quant 的 KLD → 刪除 base；另外記錄每個設定的 PPL。
- 9B BF16 在 16 GB 上會少量 spill 到系統記憶體（020），運算仍在 CUDA 上。任何 OOM 都記為 NT，不改設定重跑。
- **Hypotheses**：

| id | 內容 | 預測 |
|---|---|---|
| H-Q2a | 兩個 context、兩個 size：KLD(Q8_0) < KLD(Q6_K) < KLD(Q4_K_M)（8 個 z 檢定，Holm） | supported |
| H-Q2b | 每個 quant、size 與 context：KLD(c) ÷ KLD(512) 在 [0.5, 2.0] | supported |

- H-Q2b 的界線很寬：量化誤差可能隨上下文累積，也可能因為長上下文讓預測更確定而變小，目前沒有可靠的方向預測。
  所以只檢定「沒有數量級的變化」。

## Q3：W4A16 的 quality

- 同一個 runtime（vLLM 0.30.0）中比較：
  - W4A16：Core B 的 `RedHatAI/Qwen3.5-9B-quantized.w4a16`；
  - BF16：`Qwen/Qwen3.5-9B@c2022362`（Core A 的轉換來源，pins 有 sha256），以 `--cpu-offload-gb` 放進 16 GB。
- 以 HF tokenizer 切出和 llama-perplexity 相同的 512-token chunks。
  每個 chunk 用 `prompt_logprobs` 取得每個實際 token 的 logprob，計分後半段 255 個 token，和 005 相同。
- **報告**：
  - PPL；
  - 逐 chunk 的配對 ΔNLL（W4A16 − BF16），以 chunk 為單位 bootstrap 10,000 次；
  - top-1 一致率。
  - 和 GGUF 各 quant 的 PPL 並列，但那是不同的 runtime，只作描述。
- **H-Q3**：PPL(W4A16) ÷ PPL(BF16) − 1 ≤ 2%，預測 supported。4-bit 權重量化在 9B 規模，常見的 PPL 增幅低於 2%。
- **可行性風險**：vLLM 對 Qwen3.5 混合架構的 CPU offload 未經測試。pilot 時若無法執行，Q3 記為 NT，並另寫 decision log 說明替代方案，不臨時改用其他 runtime。

## 揭露

- **作為依據**：Core A 的 quality 結果（KLD 0.002–0.057、PPL 8.19–10.39）只用於 H-Q1b、H-Q1c、H-Q2b 的界線尺度，以及確認 KLD 大小的量級。
- **不重新分析**：H-A5 與 Core A 的判定不變。Stage Q 的結果只是新的、平台與 context 範圍的證據。
- **Phase 1 的 pilot 已經在前 20 chunks 比較過兩個平台**（審查時發現，2026-10-23 補記；本文原本沒有揭露）：
  - 005 的 pilot 以相同的 flag（只有 `-t` 不同）在 Metal（`pilot-20261008-both`，012）與 CUDA（`pilot-20261010-rtx-q`，020）跑過 `--chunks 20`。
    `--chunks N` 取的是前 N 個 chunk，所以這 20 個 chunk 正是 Q1 的 200 chunks（與 580 chunks 的 PPL）的開頭。
  - 已 commit 的數值（Metal ÷ CUDA）：
    - KLD：4B 的 Q8_0 0.81、Q6_K 0.93、Q4_K_M 0.95；9B 的 Q8_0 0.32、Q6_K 0.68、Q4_K_M 0.59；
    - PPL(Q)：多數在 0.999–1.004，但 9B Q4_K_M 為 1.025（8.596 對 8.385）。
  - 依這些數字，H-Q1b 在 9B 與 H-Q1c 在 9B Q4_K_M 很可能不成立。
  - **處理**：界線與預測不改，因為現在依這些數字修改就是看了數據後調整。結果照實報告；pilot 只有 20 chunks，本文的檢驗是 200 / 580 chunks。
  - **解讀**：兩個 backend 的量化矩陣乘法不同（例如 CUDA 的 MMQ 會把 activation 量化成 q8_1），所以「不成立」代表兩個平台的數值差異超過界線，不代表某一個平台有缺陷。

## 原本尚未決定的項目（已在實作補記中定案）

- harness 的切分：凍結的 `run_quality` / `quality_parse` / `quality_analysis` 只 import，不修改。
- pilot 的內容：見實作補記 5。
- Q3 的 BF16 checkpoint：見實作補記 7。

## 實作補記（2026-10-23，凍結前；沒有任何 Stage Q 數據）

harness 依本文以 TDD 實作：`quality/stage_q.py`（plan 與 runner）、`quality/stage_q_analysis.py`（hypothesis）與 `configs/stage_q.toml`。
本文沒有寫死、實作時決定的細節如下，凍結時一併凍結。

1. **重用的範圍**：
   - job 的格式、flag、成功判定與每個 job 前後的 env capture 都沿用 `run_quality`（import `_flags` 與 `quality_parse.classify`）；
   - z 檢定與 Holm 用 `quality_analysis` 的 `kld_test` 與 `holm`；
   - 三個凍結模組都沒有修改。
2. **chunk 數**（審查後修正計分數）：
   - Q1 的 KLD base 以 `--chunks 200` 產生，之後的 self-KLD 與 quant 的 KLD 讀同一個 base，chunk 數由 base 決定；
   - Q1 的完整 PPL 不加 `--chunks`，預期 580；
   - Q2 的 chunk 數 = WikiText 的 297,194 token ÷ context：2K 為 145，8K 為 36；每個 chunk 計分 1,023 / 4,095 個 token（512 為 255）。
3. **磁碟**：base 的大小依計分的 token 數換算（005 / 008：580 chunks、每個 255 token 的 base 為 73.5 GB），加 10%，由 `disk_guard` 在每個 base 之前檢查。
   每個 session 的 base 在自己的目錄（`data/kld/<session>/`）；base 失敗時，同組的 KLD run 記為 skipped，不會讀到別的 session 留下的 base。
4. **CUDA 的 BF16 PPL**（H-Q1c 的對照）：取 Phase 1 self-KLD row 的 PPL(Q)，和各 quant 取 KLD row 的 PPL(Q) 是同一種評估（README 的 9.945 / 8.186）。
5. **pilot**：
   - 每個部分（q1-metal、q1-cuda、q2）一個 pilot session，執行完整的 job 清單，但每個 job 只跑 10 chunks；
   - 用來確認標記、chunk 數、時間與磁碟用量，結果不分析（和 005 的 pilot 相同）。
6. **凍結的檢查**：
   - `configs/stage_q.toml` 的 `freeze.stage_q` 為 true 之前，`stage_q.py plan --mode formal` 拒絕；
   - 凍結的證據是三個通過的 pilot（每個 job 都 ok），而且 session 記錄的 `harness_sha256`（`stage_q.py` 與 `stage_q_analysis.py`）等於目前的 harness；
   - 正式 plan 記錄 `harness_sha256` 與 `stage_q_sha256`，runner 拒絕由其他 harness 建立的正式 plan；
   - `stage_q_analysis` 只接受正式、env 有效、部分正確、harness 與設定檔都相同的 session，並重新檢查凍結的證據；
   - 尚未執行的部分，其 hypothesis 為 NT（Q1 的 Mac 部分可能因為 Mac lock 較晚執行）。
7. **Q3 的 BF16 checkpoint**：
   - Mac 沒有保留 9B 的 safetensors（`models/hf` 只有 tokenizer 與設定），所以需要從 HF 下載 `Qwen/Qwen3.5-9B@c2022362`（19,329,393,661 B，pins 有每個 shard 的 sha256）。
   - 下載量大，先取得使用者同意。Q3 的 harness 可以先寫，pilot 與正式數據在下載之後。
8. **Q3 的 harness**（`quality/stage_q3.py`，`configs/stage_q3.toml`）：
   - **獨立的凍結**：Q3 用另一個 runtime，又要等下載，所以有自己的 `freeze.stage_q3` 與 pilot，不拖住 Q1 / Q2 的凍結。
   - **chunk 檔**：在 WSL 以 pin 住的 tokenizer 編碼 `wiki.test.raw`（不加 special token），切成 ⌊n ÷ 512⌋ 個 512-token chunk。
     檔案的 sha256 與 chunk 數在凍結前 pin 在設定檔，兩個 checkpoint 讀同一份 token id。
   - **計分**：每個 chunk 一個 prompt，`prompt_logprobs = 1`、`max_tokens = 1`、temperature 0。
     計分位置是 257–511（和 llama-perplexity 相同的 255 個 token），記錄實際 token 的 logprob 與 rank-1 的 token。
   - **engine**：沿用 Core B 和數值有關的設定（language model only、bfloat16、pins stage_b 的 GPU 記憶體比例與 GDN prefill backend、
     不用 prefix caching）；`max_model_len` 1024、每個 step 最多 512 token（限制 prompt logprob 的 fp32 張量大小）；
     BF16 另加 `cpu_offload_gb`（設定檔，pilot 確認）。
   - **runner**：每個 job 之前做 031 的閒置判定，前後 env capture，子 process 在 vLLM 的 venv 中計分；OOM 記為 oom（本文：NT，不改設定重跑）。
   - **資料的位置**：逐 token 的分數（含 rank-1 的 token id，可能部分還原原文）只放在 `raw/`，git 只有 sha256；
     每個 row commit 每個 chunk 的 NLL，分析時核對 `raw/` 與 manifest、與 row 的 NLL 一致。
   - **分析**：兩個 checkpoint 的 PPL、逐 chunk 配對的 ΔNLL（`quality_analysis` 的 bootstrap，10,000 次、Core B 的 seed）、top-1 一致率；
     H-Q3 依 PPL 比判定，任一 checkpoint 不是 ok 時為 NT。
9. **判定規則**（審查後寫明；凍結時一併凍結）：
   - 順序（H-Q1a、H-Q2a）沿用 005 的規則（`quality_analysis.h_a5`）：
     | 狀況 | 判定 |
     |---|---|
     | 任一 KLD 缺格（含 OOM、skipped） | NT |
     | 全族（4 / 8 個檢定）Holm 後全部顯著，而且方向都正確 | supported |
     | 其他 | not supported |
   - 界線（H-Q1b、H-Q1c、H-Q2b）是實質門檻，不是統計檢定：
     | 狀況 | 判定 |
     |---|---|
     | 任一已觀察到的比值超出界線（即使另有缺格） | not supported |
     | 沒有超出，但有缺格 | NT |
     | 全部在界線內（含端點） | supported |
   - 尚未執行的部分，其 hypothesis 為 NT。
10. **平台規則**（037 §4）：
    - Mac 上沿用 `run_core_a.env_problems` 的 AC、電量與 thermal 檢查，但它的 Phase 1 設定檢查（019 的 70W adapter）改為本 stage 的規則：powermode 0 或 2、Apple 140W adapter、`iogpu.wired_limit_mb` 0。
      原本兩條規則同時套用，任何 adapter 都會被拒絕，正式的 q1-metal 永遠無法執行（審查第 1 項）。
    - session 記錄執行的 host（OS、是否 WSL、是否有 nvidia GPU），pilot 與正式 session 都要符合該部分的平台。
    - lock 不只看旗標：GPU 以 `run_adv.held_gpu_lock` 確認這個 process 持有 flock；Mac 確認 `owner.txt` 的 pid 是這個 process 的祖先。
11. **正式 session 的輸入**：開始前核對 `wiki.test.raw` 與 plan 用到的每個 GGUF 的 sha256 等於 pins，兩個平台讀的是同一份輸入（H-Q1b / H-Q1c）。
12. **plan 的一致性**：正式 plan 由 `build_plan` 產生；runner 與分析都重建一次並要求完全相同，改過的 plan.json 不會被接受。
13. **凍結證據的完整性**：
    - pilot 的 plan.json 必須等於 `build_plan` 的 pilot plan，每個 job 都 ok，`plan_sha256` 相符，host 符合平台；
    - 建立正式 plan 時，該部分的 pilot 必須已在這台機器上通過（另外兩部分由 Mac 上的分析檢查），所以正式數據不會早於 pilot；
    - `harness_sha256` 也涵蓋 `run_quality`、`quality_parse` 與 `disk_guard`（沒有 guard test 鎖住它們）。
14. **只分析第一個正式 session**：同一部分若有多個正式 session，分析只接受最早開始的那一個；其他的需要另寫 decision log。這是為了不能事後挑選結果（例如 OOM 之後換設定重跑）。
15. **Q3 的修正**：
    - 子 process 的環境改用 Core B 的 `server_env`（CUDA_HOME shim、venv 的 bin 在 PATH 最前面、GDN decode kernel），啟動前跑 `toolchain_problems`；原本少了這些，vLLM 的 FlashInfer JIT 會失敗。
    - engine 加上 Core B 的 `attention_backend = FLASH_ATTN`，並從子 process 的 log 核對 attention backend、text-only 與 GDN prefill kernel（`coreb_smoke.parse_server_log`）。
    - 正式 session 要求 vLLM 為 pins 的 0.30.0，並核對 BF16 每個 shard 的 sha256（pins 的 `source_shards`）。
    - chunk 檔必須來自 pins 的 `wiki.test.raw` 與 005 交叉核對過的 tokenizer.json，而且 token 數等於 llama-perplexity 的 297,194；否則不寫檔，另寫 decision log。設定檔的 chunk 數只能是 0（尚未產生）或 580。
    - pilot 證據要符合目前設定的 job（engine、offload、env）、pin 住的 chunk 檔、host 與 vLLM 版本。
    - 逾時另外記錄（不再和 SIGKILL 混在一起），並結束整個 process group；OOM 只看 log 的最後 60 行，並加入 vLLM KV 不足的訊息。
    - rank-1 平手時取最小的 token id（兩個 checkpoint 用同一規則）。

## 凍結前的審查（2026-10-23）

一個 workflow（5 個審查者，每項發現由兩個獨立的 agent 嘗試反駁）審查 Q1 / Q2 與 Q3 的 harness。35 項中 33 項確認，去重後如下，全部在凍結前處理：

| # | 發現 | 處理 |
|---|---|---|
| 1 | q1-metal 的正式 run 永遠無法開始（70W 與 140W 兩條規則同時套用） | 實作補記 10 |
| 2 | 判定規則只在程式中，和 005 不同（多了 inconclusive） | 回到 005 的規則，寫成表（實作補記 9） |
| 3 | Phase 1 的 pilot 已在前 20 chunks 比較過兩個平台，本文沒有揭露 | 揭露，界線與預測不改（見「揭露」） |
| 4 | 凍結證據只在分析時檢查，正式數據可能早於 pilot | 建立正式 plan 時先檢查該部分的 pilot（實作補記 13） |
| 5 | 改過的正式 plan.json 會被接受 | runner 與分析都重建 plan 並比對（實作補記 12） |
| 6 | pilot 證據沒有綁定完整的 job 清單與平台 | 實作補記 13 |
| 7 | `harness_sha256` 沒有涵蓋 `run_quality`、`quality_parse`、`disk_guard` | 加入 |
| 8 | lock 只信任旗標 | 實作補記 10 |
| 9 | KLD base 的檔名不分 session | 實作補記 3 |
| 10 | 同一部分有多個正式 session 時可以挑選 | 實作補記 14 |
| 11 | 兩個平台的輸入檔沒有逐 session 核對 | 實作補記 11 |
| 12 | Q2 的計分 token 數差 1，base 大小的估計用了同樣的假設 | 實作補記 2、3 |
| 13 | MACHINE_POLICY 與 STORAGE_PLAN 沒有記錄 037 §4 / Q1 | 加上註記 |
| 14 | Q3 子 process 缺少 Core B 的 toolchain 環境 | 實作補記 15 |
| 15 | Q3 沒有指定 FLASH_ATTN，也沒有核對實際的 backend | 實作補記 15 |
| 16 | Q3 沒有檢查 vLLM 版本 | 實作補記 15 |
| 17 | Q3 的 chunk 檔沒有綁定 WikiText、tokenizer 與 580 chunks | 實作補記 15 |
| 18 | Q3 的 pilot 證據沒有綁定設定與 chunk 檔 | 實作補記 15 |
| 19 | Q3 的 rank-1 平手會讓整個 job 失敗 | 實作補記 15 |
| 20 | Q3 的逾時與 SIGKILL 無法區分；vLLM 的 KV 不足被記為 error | 實作補記 15 |
| 21 | Q3 沒有核對 checkpoint 的 sha256 | BF16 核對 pins 的 shard；W4A16 只有 revision |

被反駁的 2 項：rows.jsonl 沒有雜湊（分析與 runner 已以 plan、session 與 raw manifest 綁定）；輸入的「同一份」前提沒有證據（已由第 11 項處理）。

