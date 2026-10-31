# 040 — Stage Q 事先登記（DRAFT）：quality 的三個缺口

- 狀態：**Q1 / Q2 的正式結果見 048（H-Q1a、H-Q2a supported；H-Q1b、H-Q1c、H-Q2b not supported）；Q3 的正式結果見 049（H-Q3 not supported，+4.82%）**；**Q1 / Q2 已凍結（2026-10-25，`configs/stage_q.toml` 的 `freeze.stage_q = true` 與三個 pilot，harness `34256f12…`；見文末「pilot 紀錄」）**；**Q3 已凍結（2026-10-25，`freeze.stage_q3 = true`、`pilot_q3 = "pilot-q3-20261025b"`，harness `2bc3f2ee…`）**。凍結前沒有收任何 Stage Q 的正式數據。
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
   - 用來確認標記、chunk 數、時間與磁碟用量，結果不分析（和 005 的 pilot 相同）；每個 KLD base 跑完與刪除前都記錄實際大小（`base_bytes`），和該 job 的 `need_gib` 並列，用來核對補記 3 的估計。
6. **凍結的檢查**：
   - `configs/stage_q.toml` 的 `freeze.stage_q` 為 true 之前，`stage_q.py plan --mode formal` 拒絕；
   - 凍結的證據是三個通過的 pilot（每個 job 都 ok），而且 session 記錄的 `harness_sha256`（`stage_q.py` 與 `stage_q_analysis.py`）等於目前的 harness；
   - 正式 plan 記錄 `harness_sha256` 與 `stage_q_sha256`，runner 拒絕由其他 harness 建立的正式 plan；
   - `stage_q_analysis` 只接受正式、env 有效、部分正確、harness 與設定檔都相同的 session，並重新檢查凍結的證據；
   - 尚未執行的部分，其 hypothesis 為 NT（Q1 的 Mac 部分可能因為 Mac lock 較晚執行）。
7. **Q3 的 BF16 checkpoint**：
   - Mac 沒有保留 9B 的 safetensors（`models/hf` 只有 tokenizer 與設定），所以需要從 HF 下載 `Qwen/Qwen3.5-9B@c2022362`（19,329,393,661 B，pins 有每個 shard 的 sha256）。
   - 下載量大，先取得使用者同意（使用者 2026-10-23 同意）。Q3 的 harness 可以先寫，pilot 與正式數據在下載之後；下載排在 GPU 正式 session 之外。
8. **Q3 的 harness**（`quality/stage_q3.py`，`configs/stage_q3.toml`）：
   - **獨立的凍結**：Q3 用另一個 runtime，又要等下載，所以有自己的 `freeze.stage_q3` 與 pilot，不拖住 Q1 / Q2 的凍結。
   - **chunk 檔**：在 WSL 以 pin 住的 tokenizer 編碼 `wiki.test.raw`（不加 special token；和 llama-perplexity 的 `-f` 及 005 的交叉核對一樣先去掉一個結尾換行），切成 ⌊n ÷ 512⌋ 個 512-token chunk。
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
   - 比值以印出的十進位數值做精確的分數運算（端點本身算在界線內，不受浮點誤差影響）。
   - 尚未執行的部分（沒有任何正式嘗試），其 hypothesis 為 NT；每個嘗試都作廢的部分也是 NT，並列出那些嘗試。
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
    - 實作（修正後覆核，2026-10-23）：
      - 正式 session 在主 checkout 執行（Phase 1 的 Mac session 曾在 `.worktrees/` 的固定 commit 執行；worktree 的 `results/stage_q/` 看不到其他嘗試，所以 Stage Q 不這樣做）。session 目錄只能是 `results/stage_q/<id>`（不可巢狀）。第一個 job 之前先寫 `session.json`（status running，之後以暫存檔加 rename 覆寫），所以第一個 job 的程序啟動之後才中斷的 session 也是一次嘗試；讀不到 `session.json` 時以 `plan.json` 判定，算作廢的嘗試。只有 `plan.json` 的目錄不算；在第一個 job 的程序啟動之前結束的也不算：被拒絕、preflight 失敗或被中斷（含第一個 job 前的充電閘門與 env 擷取），一律記為 status refused（#39、#40）。Q3 的 031 閒置判定屬於它的 job（本條：閒置判定失敗時該 job 記為 error、session 作廢），所以第一個 job 的閒置判定被中斷或失敗時 session 照樣作廢。runner 收到 SIGHUP / SIGTERM 時轉成 SystemExit，上述紀錄與 Q3 scorer 的 process group 清理都會執行（#41）。
      - 有效 = 跑完而且 env 有效。env 無效（012）的 session 作廢，可用新 id 重跑一次；一個部分已有有效的正式 session，或已有兩次嘗試時，runner 拒絕開始。
      - 分析自己從 `results/stage_q` 找出每個部分的嘗試，取第一個有效的 session（之前最多一次作廢的嘗試），不再由 CLI 指定，所以不能省略已執行的部分。
      - 「另寫 decision log」的機制：取代者在**執行時**給 `--superseded <id>=docs/decisions/<log>.md`，runner 驗證後記進它自己的 `session.json`；log 必須已 commit、以 repo 內的相對路徑 `docs/decisions/<file>` 指定（兩台機器都要讀得到）、內文以完整的 token 同時寫出被取代與取代它的 session id（重跑的 id 常是原 id 加字，子字串不算）；被取代的必須是同一部分較早開始的正式嘗試，不能取代自己。分析沒有這個參數：哪些嘗試被取代，只從各 session 的紀錄推出，並再驗證一次 log，所以看過結果後不能再選；被取代的若是有效 session，之後必須有一個有效的正式 session 被分析（取代者本身作廢時，可依本條重跑一次），否則拒絕分析。這個決定放在所有被雜湊的檔案之外，不會讓 pilot 或正式 session 失效。
      - Q3 用同樣的規則（`stage_q3.formal_attempts`、`q3_session`）；Q3 的分析也自己找 session，沒有嘗試或全部作廢時 H-Q3 為 NT 並列出嘗試。Q3 某個 job 前 031 的閒置判定失敗時，該 job 記為 error，session 作廢（env 無效），可依本條重跑一次。
      - plan 指令拒絕把 plan 寫進 `results/stage_q/`；Q1 / Q2 與 Q3 的分析輸出都必須是 `results/stage_q/` 下的新 .json 檔；Q3 的分析在凍結前也一律拒絕。
      - WSL 上的部分（q1-cuda、q2、Q3）只有取回 Mac 之後，分析才看得到。分析前先執行 `tools/rtx/fetch_all_sessions.sh stage_q`：唯讀列出 WSL 上 `results/stage_q` 的每個 session 目錄、取回還沒有的（不含 raw/），並寫出 `results/stage_q/wsl-inventory.txt`（列出的時間與全部目錄），和 session 一起 commit。對 WSL 的部分，「不能省略已執行的部分」靠這一步與它的稽核紀錄。
15. **Q3 的修正**：
    - 子 process 的環境改用 Core B 的 `server_env`（CUDA_HOME shim、venv 的 bin 在 PATH 最前面、GDN decode kernel），啟動前跑 `toolchain_problems`；原本少了這些，vLLM 的 FlashInfer JIT 會失敗。
    - engine 加上 Core B 的 `attention_backend = FLASH_ATTN`，並從子 process 的 log 核對 attention backend、text-only 與 GDN prefill kernel（`coreb_smoke.parse_server_log`）。
    - 正式 session 要求 vLLM 為 pins 的 0.30.0，並核對 BF16 每個 shard 的 sha256（pins 的 `source_shards`）。
    - chunk 檔必須來自 pins 的 `wiki.test.raw` 與 005 交叉核對過的 tokenizer.json，而且 token 數等於 llama-perplexity 的 297,194；否則不寫檔，另寫 decision log。設定檔的 chunk 數只能是 0（尚未產生）或 580。
    - pilot 證據要符合目前設定的 job（engine、offload、env）、pin 住的 chunk 檔、host 與 vLLM 版本。
    - 逾時另外記錄（不再和 SIGKILL 混在一起），並結束整個 process group；OOM 只看 log 的最後 60 行，並加入 vLLM KV 不足的訊息。
    - rank-1 平手時取最小的 token id（兩個 checkpoint 用同一規則）。
      修正後覆核：vLLM V1 會把和 topk 並列最高的實際 token 排為 rank 2，所以 top-1 改依 logprob 的值決定：回傳的項目中 logprob 最大者取最小的 id。兩個都不是實際 token 的並列在 `prompt_logprobs = 1` 下看不到，依 torch.topk 的順序（只影響描述用的 top-1 一致率，H-Q3 只用實際 token 的 logprob）。審查表第 19 項的前提（rank-1 平手會讓 job 失敗）在 V1 下不成立。

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

### 修正後的覆核（2026-10-23，pilot 之前）

審查者加反駁者再看一次修正後的 harness，Q1 / Q2 確認 5 項（3 項 must-fix）、Q3 確認 3 項（1 項 must-fix），全部處理：

| # | 發現 | 處理 |
|---|---|---|
| 22 | 中止或巢狀的 session 讓「只分析第一個」可以被繞過 | 補記 14：stub、目錄不可巢狀、嘗試次數 |
| 23 | CLI 省略某一部分，已執行的部分會被當成尚未執行（NT） | 補記 14：分析自己找 session |
| 24 | 第一個 session 若 env 無效，該部分永遠無法分析；「另寫 decision log」沒有可用的機制 | 補記 14：一次作廢可重跑、`--superseded` |
| 25 | 界線的端點以浮點比值判定，正好在端點時會被判為超出 | 補記 9：精確分數 |
| 26 | pilot 沒有記錄 KLD base 的實際大小 | 補記 5：`base_bytes` |
| 27 | Q3 編碼前沒有去掉結尾換行，token 數差 1，chunk 檔寫不出來 | 補記 8 |
| 28 | Q3 只分析第一個 session 的規則和 Q1 / Q2 有同樣的問題 | 補記 14 |
| 29 | Q3 的 rank-1 平手規則在 vLLM V1 下從未生效，測試資料不是 V1 的形狀 | 補記 15 |

被反駁的 3 項：`pin_memory` 不影響數值，不屬於補記 8 要沿用 Core B 的設定（pilot 若需要 "1"，再補記）；GGUF 各 quant 的 PPL 並列只是結果文件中的描述，不需要改 harness；主機端的 OOM 會記為 error 並保留 exit code 與 log，判定同樣是 NT。

### 第三輪覆核（2026-10-23，pilot 之前）

確認 5 項（2 項 must-fix），全部處理：

| # | 發現 | 處理 |
|---|---|---|
| 30 | 分析時的 `--superseded` 可以不經任何 decision log 藏掉某部分唯一的有效 session | 補記 14：分析沒有這個參數，只讀各 session 的紀錄並驗證 |
| 31 | 看過兩個 session 的結果後，仍可用分析的參數二選一 | 同上 |
| 32 | Q3 的 gpu_not_idle 只記為 error，session 仍有效，唯一的重跑機會被用掉 | 補記 14：記為 env 問題，session 作廢 |
| 33 | log 是否寫出 session id 用子字串比對（-q2 會匹配 -q2b） | 補記 14：完整 token |
| 34 | Q3 的分析仍由參數指定 session，無法回報「尚未執行／全部作廢」，並覆寫 session 目錄的檔案 | 補記 14：自己找 session，輸出到新檔 |

被反駁的 1 項：「log 已 commit」只看 git index、內文讀工作目錄（runner 已用一份 commit 過的 log 放行取代者，分析再讀同一份；改傳別的 log 沒有可達的後果）。反駁時附帶的觀察已處理：WSL 的副本由 `deploy_bundle` 部署、沒有 `.git`，原本 RTX 上的 runner 一定判定 log「未 commit」；現在沒有 `.git` 時改以 `DEPLOY_MANIFEST.json` 判定（列在部署的 commit 中，而且內容沒有改過）。

### 第四輪覆核（2026-10-23，pilot 之前）

確認 4 項（沒有 must-fix），全部處理：

| # | 發現 | 處理 |
|---|---|---|
| 35 | Mac 上的分析只看得到已取回的 WSL session，可以靠不取回而省略已執行的部分或某個取代紀錄 | 補記 14：`tools/rtx/fetch_all_sessions.sh` 與 commit 的清單（不改 harness） |
| 36 | log 的路徑照輸入記錄，在 WSL 上用絕對路徑時，Mac 的分析永遠讀不到 | 補記 14：只接受 repo 內的相對路徑 |
| 37 | 有效 session 可以被作廢的 session 或它自己取代 | 補記 14：不能取代自己，被取代的必須是較早的嘗試；文字改為「之後必須有有效的 session 被分析」 |
| 38 | Q3 的分析在凍結前仍會寫出 NT | 補記 14：凍結前一律拒絕 |

### E1-P 審查順帶發現（2026-10-23，pilot 之前）

| # | 發現 | 處理 |
|---|---|---|
| 39 | runner 先寫 `running` 的 stub 再做 preflight（env 擷取、輸入核對）；這段期間被中斷，session 會留在 `running`，被算成一次作廢的嘗試，但其實沒有任何工作執行過 | 補記 14：從 stub 到第一個工作之前的任何結束都記為 `refused`（不算嘗試）。Q1 / Q2（`quality/stage_q.py`）與 Q3（`quality/stage_q3.py`）都改；E2、E3 同樣修正（041 修訂紀錄） |

#39 修正後的覆核（2026-10-23，pilot 之前）確認以下各項，全部處理：

| # | 發現 | 處理 |
|---|---|---|
| 40 | Q1 / Q2 的 refused 範圍在 job 迴圈之前就結束，第一個 job 的充電閘門（最多等 4 h）與 env 擷取被中斷仍算一次嘗試 | 補記 14：refused 的邊界改在第一個 job 的程序啟動時（`run_job`）；E3 同樣處理 |
| 41 | Stage Q、Q3（與 E3）沒有 SIGHUP / SIGTERM 的處理，被 `kill` 時 except / finally 都不執行；Q3 的 scorer 在另一個 session，runner 被殺後會繼續佔用 VRAM，而 GPU lock 已經釋放 | `run` 期間 SIGHUP / SIGTERM 轉成 SystemExit（同 coreb_smoke）；Q3 的 `default_run_proc` 在例外時殺掉整個 process group |
| 42 | Q3 第一個 job 的閒置判定被中斷時留在 `running` | 不改程式：補記 14 規定閒置判定屬於 job，失敗即作廢；寫進本文 |
| 43 | preflight 的任何例外都記成「interrupted」 | 中斷（KeyboardInterrupt / SystemExit）記為 interrupted，其他記為 failed，都附上例外 |
| 44 | 本文與 stub 的註解仍寫「中斷的 session 一律是一次嘗試」 | 補記 14 本文與四個 runner 的註解改寫 |

第三次覆核（2026-10-23，pilot 之前）確認以下各項，全部處理：

| # | 發現 | 處理 |
|---|---|---|
| 45 | Q1 / Q2 若每個 KLD base 的磁碟檢查都失敗，一個 job 都沒執行，session 卻記為 finished、有效，用掉該部分唯一的正式 session | 補記 14：沒有任何 job 程序啟動過的 session 記為 refused（附上原因） |
| 46 | SIGHUP 的處理會蓋掉繼承來的 SIG_IGN（nohup），原本能撐過斷線的 run 會中止 | `signals_exit`：繼承的 SIG_IGN 保持不變 |
| 47 | 兩段 try 之間還有幾行不在 refused 的範圍（E3 的計時與 pilot 檔、Stage Q 的設定、E2 的 pre-try） | 併入範圍；`started` 在 run_proc 之前設定（啟動前的 open / exec 失敗算一次嘗試，保守） |
| 48 | Stage Q、E3 的 module 說明仍寫中斷一律算一次嘗試 | 改寫 |
| 49 | 測試缺口：第一個 job 的 env 擷取與程序、磁碟全部失敗、訊號處理（含 Q3 scorer 的 process group 被殺）、`refused_why`、E3 Mac 第一個 job 的充電閘門、E2 harness 建構 | 補上測試 |

不修、只記錄（剩餘風險）：訊號剛好在 `Popen` 建立子程序、`proc` 尚未綁定的瞬間到達時，子程序可能留下（Q3 的 scorer、E2 的 vLLM server），E2 也可能把已啟動 server 的 session 記為 refused。這段時間是微秒到毫秒。若要封住，得在建立程序前封鎖訊號，但子程序會繼承被封鎖的訊號，vLLM 的關閉流程可能因此出問題，風險比要解決的問題大。發生時，`tools/rtx/gpu_status.sh` 與下一次的 031 閒置判定都會看到佔用 GPU 的程序。

## pilot 紀錄（2026-10-25，凍結前；pilot 不產生分析用的數據）

- **q1-metal**：`pilot-q1-metal-20261023`（Mac lock 下），20 個 job 都 ok，harness `34256f12…`。
- **q1-cuda**：`pilot-q1-cuda-20261025`（WSL，GPU lock 下，部署 `2e641fd`，13:29–13:34Z），12 個 job 都 ok，env 有效，harness `34256f12…`。
- **q2 第一次**：`pilot-q2-20261025`（13:51–14:16Z），24 個 job 中 22 個 ok、2 個 error，**不是通過的 pilot**：
  - `kld-9B-c2048-Q6_K` 以 -6（SIGABRT）結束：stderr 為 `CUDA error: unknown error`，發生在 `ggml_backend_cuda_synchronize` 的 `cudaStreamSynchronize`（KLD 計算開始後約 7 s）。不是 OOM；同一個 Q6_K 在 c8192、以及下一個 job（Q4_K_M c2048）都正常。job 前後的 env 擷取正常（43 °C、P3 / P8、佔用 142–152 MiB）；Windows 的 System 事件記錄在前後 3 小時內沒有 nvlddmkm、dxgkrnl、Display 或 WHEA 的事件。判讀為一次性的 CUDA 錯誤（推論，沒有更直接的證據）。
  - 因此 `delete_kld-9B-c2048` 依規則保留 KLD 基準檔供診斷，記為 error（`data/kld/pilot-q2-20261025/` 留在 WSL，不進 git）。
  - stderr / stdout 以 `fetch_raw.sh` 唯讀取回，sha256 和 raw_manifest 相符；session 已取回並 commit（不含 raw/）。
- 凍結證據要求每個 job 都 ok（實作補記 6），所以 q2 以新的 session id 重跑 pilot；設定、harness 與 plan 都不變。pilot 不是正式數據，重跑不影響任何判定。
- **q2 第二次**：`pilot-q2-20261025b`（14:33–14:58Z，部署 `2e641fd`），24 個 job 都 ok，env 有效，harness `34256f12…`。
- **凍結（2026-10-25）**：三個 pilot（`pilot-q1-metal-20261023`、`pilot-q1-cuda-20261025`、`pilot-q2-20261025b`）都是同一個 harness、每個 job 都 ok，`stage_q.freeze_evidence_problems` 沒有任何問題；`configs/stage_q.toml` 設 `freeze.stage_q = true` 與三個 pilot id。
  - `tests/test_stage_q.py` 檢查 repo 的設定檔指向這三個 pilot、harness 等於 `34256f12…` 而且凍結證據沒有問題，所以 Stage Q 的模組之後不再修改；拒絕條件的測試改用明確的 DRAFT 設定。`test_stage_q_analysis` 的 CLI 拒絕測試改用暫時的 DRAFT 設定檔（原本的測試在凍結後會把一份空的分析寫進 `results/stage_q/`；那份檔案沒有 commit，已刪除）。
  - 凍結沒有改動任何設計、門檻或判定規則；只改了 `[freeze]`。
- **Q3 第一次**：`pilot-q3-20261025`（15:09–15:14Z，部署 `5205915`，`pin_memory` "0"），**不是通過的 pilot**：
  - `score-w4a16` ok（10 段，97 s，KV 35,635 token）。
  - `score-bf16`（`cpu_offload_gb` 8）exit 1、沒有分數檔：vLLM 載入時卸載了 8.06 GB 參數（UVAOffloader），但印出「Pinned memory is not available on this platform; falling back to device memory for UVA buffers」；之後 profile run 時 Qwen3.5 GDN 的 Triton prefill kernel（`fused_post_conv_prep`）收到 CPU 上的張量：`ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`，engine 初始化失敗。
  - 推論：WSL 上關閉 pinned memory（`VLLM_WSL2_ENABLE_PIN_MEMORY=0`）時，卸載的權重是一般的 CPU 張量，GPU 上的 Triton kernel 讀不到；本文第 77 行預先列出的可行性風險。審查紀錄（第 231 行附近）記下「pilot 若需要 "1"，再補記」。
  - vLLM 同時警告：WSL2 上 CPU 卸載加 CUDA graph 需要 pinned memory，而 Windows（WDDM）對 pinned memory 有約實體記憶體 50% 的全系統上限，用量過多時整台主機可能失去回應、需要強制重開機。主機有 31.2 GB 實體記憶體（當時可用 26.7 GB），8 GB 約為上限的一半。
  - 因為改成 "1" 會讓整台 Windows 主機（也是另一個專案的機器）承擔風險，是否改用 "1" 重跑 pilot 交由使用者決定；不改的話，依本文第 77 行 Q3 記為 NT，另寫 decision log 說明替代方案。
- **使用者決定（2026-10-25）**：試一次 `pin_memory` "1"。`configs/stage_q3.toml` 的 `pin_memory` 由 "0" 改為 "1"（兩個 job 都用；依審查紀錄，pin_memory 不影響數值）。只在 GPU lock 空著、主機可用記憶體 ≥ 20 GB 時跑 pilot；這次之後不論成敗都不再換其他設定，失敗就依第 77 行把 Q3 記為 NT。
- **Q3 第二次**：`pilot-q3-20261025b`（15:18–15:25Z，部署 `f53f7d3`，`pin_memory` "1"）：兩個 job 都 ok（W4A16 97 s、KV 35,635 token；BF16 卸載 8 GB 217 s、KV 45,670 token），env 有效，harness `2bc3f2ee…`；過程中主機一直有回應（ssh 正常，GPU 回到 157 MiB）。
  - 揭露：pilot 的 row 照設計記錄每段的 NLL（本文「一個 row 記錄每段的 NLL」）；檢查 pilot 時兩個 checkpoint 在這 10 段的 NLL 被印出並看到。H-Q3 的門檻（PPL 比值 − 1 ≤ 2 %）與判定規則在凍結前早已寫定，沒有因此改動；pilot 的 10 段不進入正式分析。（2026-10-31 更正：pilot 的 row 不進入分析，但同樣的 chunk 0–9 在正式 session 被重新計分、值和 pilot 逐位元相同，包含在 580 段中；這 10 段上 W4A16 的 PPL 已比 BF16 高 4.24%，見 049。）
- **Q3 凍結（2026-10-25）**：`stage_q3.freeze_evidence_problems` 沒有任何問題；`configs/stage_q3.toml` 設 `freeze.stage_q3 = true` 與 `pilot_q3`。正式 plan 為兩個 job 各 580 段。`tests/test_stage_q3.py` 檢查 repo 的設定檔指向這個 pilot、`pin_memory` 為 "1"、harness 等於 `2bc3f2ee…` 而且凍結證據沒有問題；拒絕條件的測試改用明確的 DRAFT 設定。凍結沒有改動任何設計、門檻或判定規則。
- **q1-metal 正式第 1 次嘗試**：`formal-20261025-q1-metal`（Mac lock 下，15:09–16:44Z）：finished，但 **env 無效**，20 個 job 只有 8 個 ok，記為一次無效的嘗試（每個部分最多 2 次）。
  - 10 個 KLD job 被磁碟檢查跳過（4B、9B 都是）：`data/kld` 可用約 30 GB，需要 26 GB 加上保留 50 GB；`delete_kld-*` 因此記為 error。pilot 只跑少數 chunk，用量小，沒有遇到。
  - env 無效：session 執行期間，主 checkout 上有 commit（Q3 凍結 `f53f7d3`、`8a02e68`，暫停點 `c7ecdd4`），每個 job 前後的 env 擷取都記到「repo commit 與 preflight 不同（mid-session code change）」與 dirty。Mac 的 session 直接從主 checkout 執行，這是操作上的錯誤。
  - 8 個 PPL job（4B / 9B 的 BF16、Q8_0、Q6_K、Q4_K_M）都 ok，但屬於無效的 session，不進入分析。
  - 第 2 次嘗試之前：Mac 的可用空間要 ≥ 76 GB（26 + 50，由使用者決定釋放哪些檔案）；session 執行期間不得在主 checkout 上 commit 或改檔（或改從獨立的 clone 執行）。
- **q1-metal 正式第 2 次（最後一次）嘗試**：`formal-20261031-q1-metal`（Mac lock 下，2026-10-30 21:35–23:57Z）：**finished、20 / 20 ok、env 有效、沒有 env flag、沒有充電暫停**，powermode 2，140W adapter（協商 94 W），harness `34256f12…`。
  - 使用者先釋放空間（開始時 `data/kld` 可用 107.2 GiB；兩個 KLD base 各約 24–26 GiB，跑完即刪除）。
  - 從固定在 `09e2776` 的 git worktree（`.worktrees/`，git-ignored）執行，`--models`、`--data`、`--bin` 指向主 checkout 的絕對路徑；主 checkout 在這段期間的 commit 不影響 session 的 env capture。結束後把 session 目錄搬回主 checkout 的 `results/stage_q/`，再移除 worktree。
  - **偏離**：實作補記 14 寫「正式 session 在主 checkout 執行（……Stage Q 不這樣做）」，上面第 1 次嘗試後的補救只寫了「或改從獨立的 clone 執行」，實際用的是 worktree，沒有事先修訂補記 14。`09e2776` 的 tree 已含第 1 次嘗試的 `session.json`，worktree 中的 runner 看得到它（env 擷取的 lock owner 直接記錄 cwd 為 worktree）；建立 worktree 當下主 checkout 是否有未 commit 的 q1-metal 嘗試無法事後證明（目前沒有）。符合規則的做法是留在主 checkout、期間不 commit（「獨立的 clone」同樣偏離補記 14）。對數值與判定沒有影響；048 揭露。
  - 分析看到的 q1-metal 嘗試：`formal-20261025-q1-metal`（作廢）→ `formal-20261031-q1-metal`（有效），符合「之前最多一次作廢嘗試的第一個有效 session」。
- **q1-cuda 正式**：`formal-20261031-q1-cuda`（GPU lock 下，2026-10-30 23:21–23:36Z，部署 `cd615d0`）：12 / 12 ok、env 有效，第一次嘗試即有效。
- **q2 正式**：`formal-20261031-q2`（GPU lock 下，2026-10-30 23:36–10-31 00:58Z，部署 `cd615d0`）：24 / 24 ok、env 有效，第一次嘗試即有效。
  - 揭露：session 執行中（約 00:29Z），我在 Mac 上執行了一次 `tools/rtx/fetch_all_sessions.sh stage_e`（以 ssh 唯讀列出 WSL 上 `results/stage_e` 的目錄），違反「GPU session 期間不讀寫 WSL」的操作規則。q2 只量 PPL 與 KLD（數值結果，不量時間），env 擷取沒有 flag，判定不受影響。

