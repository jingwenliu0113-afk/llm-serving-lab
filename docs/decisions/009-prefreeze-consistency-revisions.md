# 009 — Stage A 凍結前的一致性修訂（review 後；不改 hypothesis 結構與門檻）

- 狀態：Accepted（2026-10-08）
- 日期：2026-10-08
- 依據：commit `0d72a98` / `0c0e099` / `94190f5` 的 review workflow（code 與 numbers 兩個 lens，每項經 skeptic 驗證）。
  本 log 只收 numbers lens 中被判定為 valid / partially valid 的文件項目；code 項目已直接修正（`57c5b74`、`9f37785`）。
- 時點：`freeze.mac = false`，**沒有任何 Mac / NVIDIA 正式數據**。下列修訂都不是 post-hoc。

## 背景

008 把模型換成 Qwen3.5 後，PREREGISTRATION 與 004 / 005 / 007 有幾處沒有同步，或數字算錯。
這些文件在 Stage A 會逐字凍結，凍結後就不能再改，所以要在凍結前一次修正。

## 決策（逐項）

### 1. H-A4(b) 的預測值改為 9B 約 4.45、4B 約 4.84（PREREGISTRATION §2）

- 原文寫 9B 約 4.68、4B 約 5.36。這組數字只有在兩個條件下才算得出來：P 含 token_embd 與 output，且 attention 用非 causal 的完整 L²。
- pinned llama.cpp 的實際情形：
  - `llama-bench` 只計算最後一個 token 的 logits。
  - token_embd 只做查表，沒有 FLOPs。
  - KV view 逐 ubatch 增長，所以 attention 是 causal（約 L²/2）。
- 依此改用 TTFT ∝ 2·P_nonembed·L + 65,536·L²：
  - P_nonembed 為 9B 6.92B、4B 3.57B。
  - 65,536 = 2 × 8 層 full attention × 16 heads × 256。
  - 算出的預測值是 4.45 / 4.84，與 008 的數字相同。
- 門檻（≤ 4 即推翻）**不變**。9B 距離門檻只有約 11%。GDN 的 chunk 計算與 L 成線性，未計入；計入後比值會更接近 4。

### 2. 004 的 spill 定義依 008 改寫（004、PREREGISTRATION §3）

原定義仍指向已不存在的 8B-Q4_K_M。改為：

- **C_ctx** 在 Stage B 量一次：取 **4B-Q4_K_M @ 2048** 的 fit run，以該 run 期間 `memory.used` 的峰值減 B，再減該 run 的 self。
- **速度 spill 的 anchor** 改為同平台、同 L 的 **4B-Q4_K_M**；若 anchor 本身是 spill，改用 L = 2048 的 4B-Q4_K_M。
  它就是舊設計中「較小 size 的 Q4_K_M」的對應格，也與 sentinel 相同。
- **bytes 完全採用 PREREGISTRATION H-A2 的定義**：
  (檔案 − token_embd，僅 9B 扣除) + 32,768 × (L − 64) + 2·S。
  原文對每個 size 都扣 token_embd、沒有 2·S，卻寫成「同 H-A2」，兩者互相矛盾。

### 3. Core B 的 C_KV 只有一套規則（007 §3、PREREGISTRATION §2 / §6）

- 007 §3 的本文改為 hybrid 公式：C_KV(L) = ⌊(num_gpu_blocks − 1) ÷ (⌈L ÷ block_size⌉ + 3)⌋。舊的純 attention 公式標為歷史。
  - 以 160 個 block 為例：舊公式在 2K / 8K / 32K 得 38 / 9 / 2，hybrid 公式得 22 / 8 / 2，差距足以改變 H-B2 的判定。
- `num_gpu_blocks` 只在 vLLM 的 DEBUG log 出現，而 §4 禁止設定未列出的 `VLLM_*` 環境變數，因此改以下列方式取得：
  - 主要來源：server 啟動後讀 `/metrics` 的 `vllm:cache_config_info`。
  - 交叉核對：由 INFO log 的 `GPU KV cache size` 反推，num_gpu_blocks = ⌈tokens × (⌈max_model_len ÷ block_size⌉ + 3) ÷ max_model_len⌉。
    max_model_len 大於分母時，反推結果唯一。
- Stage B 的 checklist 改為記錄 num_gpu_blocks 與 block_size；`GPU KV cache size` 只作紀錄。

### 4. 005 banner 明寫 H-A5 的跨 size 配對

- 配對已改為 **9B-Q4_K_M − 4B-Q8_0**（PREREGISTRATION H-A5(a)）。
- 005 本文中所有 8B / 14B 字樣都是歷史。

### 5. fit run 的 prompt 對齊 harness（PREREGISTRATION §4、`core_a_plan.py`）

- 原 PREREGISTRATION 寫「WikiText-2 test 的前 L − 128 個 token」。004 只寫「固定 token 序列」，harness 實作為 `" the"` 重複 L − 128 次。
- 改為與 harness 一致：
  - `" the"` 在 Qwen3.5 tokenizer 下恰為 1 個 token（id 279）。2026-10-08 已實測 1 / 1,920 / 8,064 / 32,640 個單位都等於同數 token。
  - harness 會從 log 驗證 prompt 的 token 數 = L − 128。
- 理由：fit / OOM / peak memory 由 `-c L`、batch 與權重決定，與 token 內容無關。dense hybrid 模型的計算量也與內容無關。
- 新增一個測試，用 pinned tokenizer 驗證這個性質（tokenizer 不在時 skip）。

### 6. 008 的 C_KV 估計更正（008、NEXT.md）

- 以 008 自己的池大小推算：1 block = 8 層 × 528 × 4,096 B = 16.5 MiB，2.2–3.2 GiB = 136–198 blocks。
  C_KV(2K / 8K / 32K) 應為 **19–28 / 7–10 / 2**；32K 要到 3，池需 ≥ 3.21 GiB。
- 「新方案在每個 L 都不比 Qwen3-8B-AWQ 差」改為：「池的高端在每個 L 都不比舊方案差；低端的 2K 可能略少（19 vs 22–24）」。
- 這只是估計，不影響任何凍結規則。也不改變 008 的選擇理由：8K / 32K 仍然較好，而且舊方案是前一代架構。

### 7. 其他文字修正

- **PREREGISTRATION H-A3**：「相鄰差距約 0.03–0.04」改為「約 0.026–0.042（最小為 9B 的 Q6_K vs Q8_0）」。
- **004 banner**：超量測試改為 9B-Q8_0 搭配 `-c 262144`（006 / 008），本文的 14B 設定為歷史。
- **PREREGISTRATION 共用量**：token_embd 的扣除依用途分開寫：
  - H-A2 的 bytes：只對 untied 的 9B 扣除，兩平台都適用，因為 decode 只讀一列。
  - H-A1 的 N：只在 CUDA 對 9B 扣除；Mac 是 unified memory，不扣（004）。
  - 判定不受影響：Mac 最大的 N 約 9.9 GiB，遠小於 37.44 GiB。
- **PREREGISTRATION §5**：「Mac 重負載 lock 的參與方式」已由使用者決定（共享 `/tmp/ai-portfolio-mac-heavy.lock`，經 `tools/with-mac-lock`，MACHINE_POLICY），打勾。

## 為什麼

- 這些都是**凍結前**的筆誤、漏改或文件與 harness 不一致，修正後沒有任何 hypothesis 的結構、門檻、比較單位或資料切分改變。
- 不修的話，凍結後會出現兩套互相矛盾的規則（C_KV、spill、fit prompt），Stage B 就得在看到數據後選規則。這正是 006 要避免的情況。

## 放棄的替代方案

- **fit prompt 改成在 harness 實作 WikiText-2 前綴**：對記憶體結果沒有影響，卻要多一個 tokenizer 依賴與資料前處理步驟。
  不採用；改為修文件。
- **把 H-A4(b) 的門檻放寬到配合新預測**：門檻來自「attention 超線性」的機制，不是預測值，不改。
- **留到 Stage B 再處理 C_KV / spill**：違反 006「Stage B 只代入實測值」的規定。

## 對實驗的影響

- PREREGISTRATION、004、005、007、008 依上列各項修訂。各檔在修改處註明「009」。
- `benchmarks/core_a_plan.py` 只更新註解（Qwen3 → Qwen3.5），不改行為。新增 tokenizer 性質測試。
- 已收的數據：無，因此沒有任何數字作廢。
