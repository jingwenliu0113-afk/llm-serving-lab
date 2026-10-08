# 007 — Core B 的 request mix 與 grid 規則（與 Advanced「request-length distribution」劃界）

> **2026-10-08（008）**：Core B 改為 hybrid 模型。C_KV 公式要改為 ⌊(num_gpu_blocks − 1) ÷ (⌈L ÷ block_size⌉ + 3)⌋（block_size 約 528，另加 3 個 GDN 狀態 block），須在 Stage A 前於 PREREGISTRATION 修訂；mix 與 grid 的結構不變。

- 狀態：Accepted（**規則**於 Stage A 凍結；Stage B 只代入實測的 KV 容量）
- 日期：2026-10-08（同日依 re-review 修訂：長度、比例、grid 改為 Stage A 規則）
- 對應：SPEC Core B 變因「concurrency、context length、request mix、KV cache pressure」；
  SPEC Advanced「request-length distribution」；EXPERIMENT_RULES「Core B 記錄 request-length distribution」

## 背景

SPEC 把 **request mix** 列為 Core B 變因，又把 **request-length distribution** 列為 Advanced。
Day 1 的初稿沒有劃清兩者的界線，H-B3 / H-B4 也把 Core 無法隔離的機制寫進了預測。

re-review 另外指出：如果 mix 的長度、比例與 concurrency grid 留到 Stage B 才決定，而 smoke test 又已經在 RTX 上跑過
`vllm bench serve`，那就等於可以先看 NVIDIA 延遲數字，再挑能讓 H-B4 成立的參數，違反 006 的前提。

## 決策

### 1. Context level 與 server 設定

- Core B 的 context level L ∈ {2048, 8192, 32768}，與 Core A 相同。
- **所有 level 與所有 mix 共用同一個 server 設定**，`--max-model-len 32768`。level 之間只差在 request 長度，
  不會因為每個 level 各自設定而改變 scheduler / CUDA graph 的上限。

### 2. 兩個離散 mix（Core 只做這兩個水準）

| mix | 組成 | 用途 |
|---|---|---|
| `uniform` | 每個 request 都是 (input, output) = (L − 128, 128) | H-B1、H-B2、H-B3 |
| `mixed` | 90% 為 (L − 128, 128) 的短類，10% 為 (32640, 128) 的長類；只在 L ∈ {2048, 8192} 定義 | 只用於 H-B4 |

- 長度固定（`--random-range-ratio 0`），比例 *p* = 0.1 固定，dataset seed = **20261008**（皆於 Stage A 凍結）。
- `uniform` 用 `vllm bench serve --dataset-name random --random-range-ratio 0`。
- `mixed` 無法由單一 `--random-range-ratio` 產生（那是均勻分布），因此用固定 seed 產生自訂 dataset，
  記錄它的 sha256（`pins.toml` `core_b.mix_dataset_sha256`）。
- 每一次 run 都用 `--save-detailed` 的 input_lens / output_lens 驗證實際長度。

### 3. Concurrency grid（確定性規則；只代入實測的 C_KV）

- C_KV(L) = ⌊KV 池 tokens ÷ ⌈L ÷ 16⌉·16⌋，KV 池 tokens 取 server log 的 `GPU KV cache size`。
- grid = {1, 2, 4, …, C_hi}，其中 C_hi = 大於等於 max(8, 8 × C_KV) 的最小 2 的冪次，上限 256。
  8 × C_KV 等於 H-B2 預測區間上界（2 × C_KV）的 4 倍。
- 另外加入 round(C_KV × 2^(k/2))，k ∈ {−2, −1, 0, 1, 2}（≥ 1，去除重複），讓預測的 knee 附近比 2 倍更細。
- `--max-num-seqs 264`（8 的倍數，且大於 grid 上限 256），確保 knee 不會是 scheduler 上限造成的。
- 每個 C：num-prompts = max(10 × C, 200)，warmup = C 個 request。
- 每個 (L, mix) 做 3 次 server restart；每次 restart 內，C 的執行順序以固定 seed 隨機排列。

### 4. 長度分布本身不在 Core 掃描

range ratio、*p* 的多個取值、長尾分布等全部屬於 Advanced，只有通過 Gate 後才能做。Core 只**記錄**實際長度。

### 5. 機制只寫成 conjecture

H-B3 / H-B4 只預測**可觀測的效果**；括號內的機制標為「conjecture，Core 不檢驗；可能在 Advanced 隔離」。

## 為什麼

- 維持 SPEC 的 Core / Advanced 分界：Core 回答「負載與 mix 改變時 frontier 怎麼變」，Advanced 才回答「為什麼」。
- 所有會影響 hypothesis 判定的參數，都在看到任何 NVIDIA 數字之前用規則寫死；Stage B 只代入實測值。
- 共用 server 設定，讓 context level 只改變 request 長度這一個因素。
- 兩個離散 mix 足以檢驗 H-B4，grid 不會膨脹。

## 放棄的替代方案

- **用 `--random-range-ratio` 掃描多個值**：等於掃 request-length distribution，屬 Advanced。
- **用真實 trace（如 ShareGPT 長度分布）**：分布本身變成變因，另有資料來源與授權問題。
- **每個 level 各自設定 `--max-model-len`**：scheduler 與 CUDA graph 的上限會隨 level 改變，多一個混淆因素。
- **把長度、*p* 與 grid 留到 Stage B**：可能先看結果再挑參數（re-review R4）。
- **Core 不做 mix**：違反 SPEC，因為 request mix 是 Core B 列出的變因。

## 對實驗的影響

- Core B 的 grid = 3 個 level × `uniform`（全部 C）+ 2 個 level × `mixed`（只跑一個 C）× 3 次 restart。
- H-B4 在固定 C（= 該 level `uniform` 的 C_knee）下比較短類 request 的 P95 TTFT：`mixed` 對 `uniform`。
- 「為什麼會干擾」的結論不能從 Core 數據下；報告只能寫成 conjecture。
