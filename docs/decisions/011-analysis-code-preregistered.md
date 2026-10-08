# 011 — 分析程式在任何數據前寫定（Core A H-A1–H-A4、quality H-A5 與 downstream）

- 狀態：Accepted（2026-10-08）
- 日期：2026-10-08
- 對應：PREREGISTRATION §2（hypotheses 與推翻條件）、§5（Stage A checklist）；005「統計方法」
- 時點：沒有任何 Core A / quality 數據（pilot 也還沒跑）。

## 背景

PREREGISTRATION 以文字寫下判定規則。如果分析程式在看到數據之後才寫，實作細節（SE 的算法、哪些單位算 not testable、
CI 的方法）就可能被數據影響。所以判定規則在看到任何數據前就寫成程式並測試，與 Stage A 一起凍結。

## 決策

- `benchmarks/core_a_analysis.py` 實作：
  - 分析單位與「超出 noise」規則（§2 通用規則）。
  - H-A1 的 N 與 fit 預測：token_embd 只在 CUDA、只對 untied 扣除（009）。
  - H-A2 的 bytes 模型與 20% 容忍度；H-A3 的 D、相鄰配對，以及「反轉且超出 noise 才推翻」。
  - H-A4(a) spread、H-A4(b) 的比值 > 4。
  - 004 的 spill 規則（009 版：anchor 為 4B-Q4_K_M）。
- `quality/quality_analysis.py` 實作：
  - H-A5：KLD z 檢定、配對 ΔNLL chunk bootstrap、5 個 p 值的 Holm step-down。
  - downstream 的描述性比較。
- **實作上的選擇**（005 允許的選項中擇一，事先固定）：
  - downstream 的配對 Δacc 95% CI 用 **paired bootstrap**（005 寫「paired bootstrap 或 Newcombe」），
    與 ΔNLL 用同一套 bootstrap：10,000 次、percentile CI、seed 20261008。
  - bootstrap 用 Python `random.Random(seed)`（Mersenne Twister，跨 CPython 版本穩定），以 `floor(random() × n)` 取 index。
  - 雙尾 bootstrap p = 2·min(P\*(Δ ≥ 0), P\*(Δ ≤ 0))，下限 1 / n_boot，上限 1。
  - H-A4(a)、H-A4(b)、H-A2 依文字以點估計比較（文字沒有要求 noise 規則）；H-A3 依文字加上 noise 規則。
  - H-A5 缺任何一個所需 cell 時，整個 H-A5 記為 not testable（Holm family 不完整，不做部分判定）。

## 為什麼

- 先寫程式再看數據，才能確保判定不受結果影響。
- bootstrap 比 Newcombe 少一個需要另外驗證的公式，而且與 ΔNLL 的檢定一致。

## 放棄的替代方案

- **Newcombe（method 10）**：closed form、不需亂數，但我們沒有可對照的參考實作來驗證公式；不採用。
- **numpy / scipy**：專案維持 stdlib only（遠端 WSL 與 Mac 共用，不增加依賴）。

## 對實驗的影響

- 兩支分析程式與其測試在 Stage A 凍結時一併凍結。凍結後若要改，須寫 decision log 並標明 post-hoc。
- PREREGISTRATION §5 checklist 增加一項「分析程式已在數據前 commit」。
