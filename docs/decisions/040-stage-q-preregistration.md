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
  2K 為 145 chunks × 1,024，8K 為 36 chunks × 4,096。
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

## 尚未決定（凍結前定案）

- harness 的切分：凍結的 `run_quality` / `quality_parse` / `quality_analysis` 只 import，不修改。
- pilot 的內容：每一種 job 各一個（少量 chunks），確認標記、chunk 數、時間與磁碟用量。
- Q3 的 BF16 checkpoint 是否需要傳到 WSL（約 18 GB；pins 已有 sha256，屬於已記錄的檔案）。
