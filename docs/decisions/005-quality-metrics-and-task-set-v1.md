# 005 — Quality 指標與 task set v1

> **2026-10-08（008）**：模型改為 Qwen3.5，tokenizer 改變（vocab 248,320）。指標與 task set 的設計不變。依新 tokenizer 重算的結果如下：
> - WikiText：297,193 tokens → **580 chunks**，計分 147,900 個 token；KLD base 檔約 **73.5 GB / size**。
>   這是用 HF tokenizers 算的，pilot 時以 llama-perplexity 的 log 確認。
> - MMLU 結構性排除：4000 題中排除 **3 題**（source index 2151、2189、13184），剩 3997 題。
>   清單在 `quality/subsets/mmlu-test-4000.exclude.json`（`quality/mmlu_exclusions.py`），子集 sha256 `209ee07d…` 已寫入 pins。
>   4B 與 9B 的 `tokenizer.json` 相同，兩個 size 共用。與 llama.cpp tokenizer 的交叉驗證在 pilot 進行。
> - 下文中的 584 chunks、148,920 個 token、45 GB 等 Qwen3 數字保留作為歷史。
> - **H-A5 的跨 size 配對改為 9B-Q4_K_M − 4B-Q8_0**（PREREGISTRATION H-A5(a)；009）。下文所有 8B / 14B 字樣
>   （含 confirmatory family、per-chunk PPL 的執行、污染風險與成本估計）皆為歷史。

- 狀態：Accepted（v1 **草案**；pilot 校準後於 Stage A pre-registration commit 時凍結，凍結後不得依結果修改）
- 日期：2026-10-08（同日依 review 修訂：統計方法、章節、資料 pin）
- 對應：SPEC Core A Quality（「perplexity 或類似指標 + 一組固定 downstream task set」）、SPEC Day 1 #6、
  EXPERIMENT_RULES（quality 每個 model × quant 只量一次；Mac / RTX 不重複算）
- 機器可讀：`configs/pins.toml` `[quality]`（完整 revision、sha256、大小）

## 背景

SPEC 把 quality 定義為「model × quantization」的屬性：只量一次、不在兩台硬體重複量，task set 在正式實驗開始時凍結。
難處有三個：

1. 相鄰 quant（Q8_0 vs Q6_K）的品質差異很小，單純比 PPL 分不出來。
2. Qwen3 是 hybrid thinking 模型；生成式評測會把 sampling、thinking mode、chat template 全部拉進來。
3. 量測必須在 pinned runtime 上可重現，而且結果檔不能夾帶有授權疑慮的資料。

## 決策

### 指標與角色

| 角色 | 指標 | 資料 |
|---|---|---|
| **Confirmatory** | 同一 size 內：KL divergence vs 同 size 的 BF16 reference | WikiText-2 raw test，全部 584 chunks |
| **Confirmatory** | 跨 size（H-A5）：同一批 chunk 上的配對 ΔNLL（14B-Q4_K_M − 8B-Q8_0） | 同上 |
| **Descriptive** | Log-likelihood downstream accuracy 與 flip rate | HellaSwag、MMLU（cloze）、ARC、WinoGrande |

### 執行設定

- **平台**：Mac Metal、pinned build（002）、單一 flag set。MACHINE_POLICY 把 quality 分給 Mac，
  而且 14B BF16 reference（27.5 GiB）放不進 16 GB VRAM。
  〔2026-10-09 起依 020（使用者同意）改在 RTX 5070 Ti 的 wsl2-cuda 上執行，兩個 size 都搬（4B 一起搬是我的建議，待使用者確認）；flag set 不變，`-t` / `--poll` 改用 RTX 的 6 / 0。
  9B BF16 會少量 spill 到系統記憶體，所有運算仍在 CUDA 上完成。〕
- **flags**（`llama-perplexity`）：
  `-ngl 999 -fa on --fit off -ctk f16 -ctv f16 -b 2048 -ub 512 -c 512 -lm mmap -t N_thr --poll P`；
  downstream 另加 `-np 16`。PPL / KLD 模式下 n_parallel 由工具自動設為 n_batch ÷ n_ctx = 4，因此不另給 `-np`。
- **不用 chat template、不生成**：只做 log-likelihood scoring，sampling 與 thinking mode 都不介入。
- **量測點**：每個 size 4 個點（BF16 reference + 3 quants），共 8 個；另外每個 size 再跑一次 reference 對自己的 KLD
  （只算 KLD，用來量 floor）。BF16 也跑 downstream，三個 quant 才都能與真正的 reference 配對比較。

### WikiText-2（KLD / PPL）

- 語料：`wiki.test.raw`，為 `ggml-org/ci` @ `927b3642…` 中 `wikitext-2-raw-v1.zip` 的成員。完整 hash 與大小見 `pins.toml`。
- Qwen3 tokenizer 下約 299k tokens；`-c 512` 時為 584 chunks，每個 chunk 計分後半段 255 個 token，共 148,920 個。
- **KLD 流程**：先用 BF16 reference 跑 `--kl-divergence-base REF.kld`，再對每個 quant 跑 `--kl-divergence`。
  報告 mean / median / p99 KLD、"Same top p"、RMS Δp、paired ΔPPL。
- **跨 size 配對 ΔNLL**：14B-Q4_K_M 與 8B-Q8_0 各跑一次一般 perplexity 模式，加上 `--ppl-output-type 1`
  （逐 chunk 輸出 6 位小數的累積平均 NLL）。由累積值差分還原每個 chunk 的 NLL，誤差約 3e-4。
  兩個模型共用 tokenizer、語料與 `-c`，因此 token 序列完全相同，可以逐 chunk 配對。
- KLD logits 檔每個 size 約 45 GB，放在 `data/kld/`（git-ignored），3 個 quant 用完即刪，**絕不 commit**。
- **大型檔案政策（使用者決定，2026-10-08）**：
  - KLD logits 不是下載來的檔案，而是由本地 BF16 reference 執行 `--kl-divergence-base` **產生**的暫存檔。
    只有在 Stage A 凍結後、正式 quality run 確實需要時才產生，不提前產生。
  - 任何大型下載（模型 safetensors、資料集）之前，都要先在 `pins.toml` 記錄來源、版本（revision）與 checksum（sha256）。
    目前 safetensors 的 shard 與資料集都已記錄；未記錄的檔案不得下載。
  - 保留 / 刪除的分級、空間預算與 50 GiB 安全餘量，見 `docs/STORAGE_PLAN.md`。KLD logits 屬於可重建 scratch，一次只存在一個 size。

### Downstream（log-likelihood、cloze）

| Task | 模式 | 來源（`pins.toml` `[quality.files.*]`） | 題數 |
|---|---|---|---|
| HellaSwag | `--multiple-choice` | `hellaswag-validation.bin`（10042 題） | 固定 4000 題子集 |
| MMLU（cloze） | `--multiple-choice` | `mmlu-test.bin`（13943 題） | 固定 4000 題子集，再套用結構性排除規則 |
| ARC | `--multiple-choice` | `arc-combined-validation.bin`（前 570 題為 Easy、後 299 題為 Challenge） | 全部 869 題（需要 `-np ≥ 5`） |
| WinoGrande | `--winogrande` | `winogrande-debiased-eval.csv` | 全部 1267 題 |

- **子集凍結**：用一支很小的 deterministic script 產生子集，**commit 題號清單**與子集 `.bin` 的 sha256
  （`pins.toml` `[quality.subsets.*]`），不能只 commit seed（Python `random.sample` 不保證跨版本一致）。
  執行時 task 數設 0（全部、依檔案順序）。
  - 不用 llama.cpp 內建的 `--hellaswag-tasks` 抽樣：它用的 `std::uniform_int_distribution` 在 libc++ 與 libstdc++ 之間結果不同。
- **MMLU 結構性排除規則**（在任何 scoring 之前依結構判定，不看分數）：答案完全落在所有選項共同 token prefix 內的題目一律排除。
  這類題會在 `multiple_choice_score()` 觸發越界讀取；全集共 13 題，研究抽出的 4000 題子集裡有 3 題。
  題號清單要在 pinned tokenizer 下重算並 commit。

### 統計方法

- **Confirmatory family**：α = 0.05（雙尾），以 Holm 法校正下列 5 個檢定：
  - 每個 size 各 2 個：KLD(Q6_K) > KLD(Q8_0)、KLD(Q4_K_M) > KLD(Q6_K)。
    用工具輸出的 mean ± SE 做 z 檢定，分母取兩者 SE 的平方和開根號。
    同一批 token 上的配對資料為正相關，用邊際 SE 會高估差的 SE，屬保守做法；
    但工具的 SE 假設 token 之間獨立，屬樂觀做法。兩個方向的偏差都要寫進報告。
  - 跨 size 1 個：配對 ΔNLL 以 chunk 為單位 bootstrap 10,000 次，雙尾 p = 2·min(P\*(Δ ≥ 0), P\*(Δ ≤ 0))，下限 1/10,000。
  - **判定只用 Holm**：5 個 p 值一起做 Holm step-down，校正後顯著且方向符合預測才算成立。
    ΔNLL 的 95% bootstrap CI 只作描述，不用於判定。
- **KLD 不做逐 chunk bootstrap**：工具只輸出累積平均到 5 位小數，差分後的捨入誤差大於 Q8_0 的訊號。
  只報告工具的 mean ± SE，並註明 token 獨立的假設。
- **Downstream（descriptive，不做多重比較校正）**：
  - 每題對錯可由累積 accuracy 差分還原；task 數為 0 時題目與題號一一對應。
  - 每個 task 報告配對 Δacc 的 95% CI（paired bootstrap 或 Newcombe）：每個 quant vs BF16，以及相鄰 quant
    （Q8_0 vs Q6_K、Q6_K vs Q4_K_M）。同時報告 flip rate。
  - **CI 不含 0 才稱為 "resolved"**，否則一律寫「not resolved」。MDE 只用於解讀（說明這個 N 能分辨多大的差），
    不是判定規則；不做等價宣稱（沒有事先設定 TOST margin）。

### 已知限制（報告必須寫明）

- 分數**不能**與公開 leaderboard 比較：cloze 形式、沒有 few-shot、沒有 chat template、ARC 用 validation split、
  `mmlu-test.bin` 有 13943 題（原版 14042）。
- 汙染：WikiText / MMLU / HellaSwag / ARC 幾乎確定出現在 Qwen3 的預訓練資料中，絕對分數會偏高；
  同一 size 內比較 quant 不受影響，8B vs 14B 的比較（包括 H-A5）則可能部分反映記憶能力。
- quality 只在 `-c 512` 量，不代表長 context 下的品質。
- llama.cpp 的 WinoGrande scorer 有已知 FIXME：trailing context ≤ 3 tokens 時，第一個 logit 的位置不對。

### 執行可靠性（harness 必須做到）

- **`llama-perplexity` 在幾乎所有錯誤路徑都 exit 0。** 成功與否必須以「正向標記 + 預期數量」判定：
  例如 `Final estimate` / `====== KL divergence statistics ======` / `Final Winogrande score`、chunk 數 = 584、
  task 數 = 預期值。不能只靠 exit code，也不能只 grep 錯誤字串。
- KLD 模式遇到 vocab 不符時只印 `LOG_ERR` 然後繼續跑，harness 必須把它當成 hard fail。
  sanity check：Q8_0 的 KLD 應在 1e-3 量級，遠大於此代表來源不一致，不是量化造成的。
- v0.5.0 與 v0.6.0 之間 `tools/perplexity/perplexity.cpp` 的差異**只有 batch API 重構**，scoring 公式相同
  （2026-10-08 diff 確認）。研究在 v0.6.0 讀到的行為同樣適用於本 pin。

### Pilot（凍結前；不得用來依分數調整 task）

1. 用 `--chunks 20` 加上約 200 題子集，量測 throughput，並確認重跑結果 deterministic。
2. 數值 sanity check：`-fa on` vs `-fa off` 下 Q8_0 的 KLD，以及 reference 對自己的 KLD 約在 5e-4 量級。
3. 確認 llama.cpp 的 tokenization 與 HF tokenizer 的 token 數一致（584 chunks）。
4. 估計 discordance rate 以計算 MDE。**pilot script 只印出 b + c 與 N，不印任何一邊的 accuracy**，以免看到差異方向。
5. 若時間預算過長，只能在**看分數之前**用結構性方式縮小 N（例如改為 2000 題子集），並另寫 log。

### 資料授權與保存

- 授權：WikiText 為 CC-BY-SA-3.0 + GFDL；ARC 為 CC-BY-SA-4.0；MMLU 為 MIT；WinoGrande 的 data 為 CC-BY；
  HellaSwag 在 llama.cpp 原始碼註記為 MIT，但官方 repo 自 2026-09-14 起因 DMCA 被封鎖。以上不構成法律意見。
- **repo 內不放任何題目文字或資料檔**，只 commit 題號清單、sha256、aggregate 分數。原始資料存放在 `data/`
  （git-ignored），即本機私有、有 hash 的 archive，避免公開 URL 消失時無法重現。
- 只使用 revision-pinned 的 URL；llama.cpp 自帶的下載 script 抓的是 `main`，不用。

## 為什麼

- **KLD 是主要指標**：它對相鄰 quant 的分辨力遠高於 PPL。在 Llama-3-8B 上，PPL 的 SE（±0.038）大於 Q8_0→Q6_K 的差距（0.019），
  而 KLD（約 1.4e-3 vs 5.5e-3）分得很清楚。
- **log-likelihood downstream**：同一個 pinned 工具就能跑，沒有 sampling / template / thinking 的干擾，成本可控。
- **BF16 reference 由本地轉換產生**：與 004 的量化來源相同，8B 與 14B 的 reference 型別一致。
- **跨 size 只能配對比較**：兩個獨立 PPL 的差值會被 SE 淹沒；同一 token 序列的配對 ΔNLL 才有檢定力。

## 放棄的替代方案

- **只用 PPL**：Q8_0 與 Q6_K 分不開。
- **用官方 Q8_0 當 KLD reference**：Q8_0 自己的損失就量不到；而且官方檔案的來源與本地量化不同（004）。
- **用 ggml-org / unsloth 的 BF16 / F16 GGUF 當 reference**：8B 與 14B 的 dtype、converter 年代不一致；本地轉換最一致。
- **生成式任務（GSM8K 等）**：會把 sampling、thinking mode、chat template 全部拉進來，而且成本高。
- **完整 HellaSwag + MMLU**：約 31.5 h，MDE 只從約 1 pp 降到約 0.6 pp，不划算。
- **「觀測差 < MDE 就算 not resolved」**：MDE 是 power 的量，不是推論規則（review 更正），改為 CI 規則。

## 對實驗的影響

- quality 數字綁定 build 11146（002）、上述 flag set、`-c 512` 與 `pins.toml` 中的資料 hash。
  其中任何一項改變，都要寫新的 decision log 並**全部重跑** quality。
- quality 只在 Mac 量一次，RTX 不重量（EXPERIMENT_RULES）。RTX 上的效能結果引用同一批 sha256 檔案的 Mac quality 數字。
  〔020：改為只在 RTX 量一次，Mac 不量；兩個平台的效能結果都引用同一批 sha256 檔案的 RTX quality 數字。〕
- 成本：每個 (model, quant) 約 1.42M tokens。粗估 8B 每點約 1.6 h、14B 每點約 2.8 h，8 個點約 17.6 h；
  另有 2 次 self-KLD 與 2 次跨 size PPL pass（每次約 0.3M tokens，約 0.21 個點），約 1.9 h。
  合計約 **19.5 h** 的 Mac GPU 時間（由舊版 LLaMA-7B 數據換算，**以 pilot 校準為準**）。
- 需要 Mac 長時間的重負載 slot，且不得與正式效能 benchmark 重疊。Mac lock 如何參與見 `NEXT.md` 的「待使用者確認」。
