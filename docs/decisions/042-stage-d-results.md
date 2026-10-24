# 042 — Stage D 的結果（formal-20261023-d-a1b、-d-a2、-d-b、-d-c）

- 狀態：**結果**。四個 session 都有效，判定全部依 039 凍結的規則（凍結 commit `fd95164`），沒有任何事後的修改。
  D-B 唯一一個無效 run 的原因：2026-10-25 已以 raw 確認和推論一致（見 D-B），判定不變。
- 日期：2026-10-24
- 對應：039（事先登記）、030（沿用的規則）、031（閒置與 spill）、035（c1b，H-D2 的比較對象與 λ_ref 的來源）、037、038
- 輸出（commit `62869cf`）：
  - `results/stage_d/formal-20261023-{d-a1b,d-a2,d-b,d-c}/analysis.json`：`stage_d_report`，在 Mac 上執行，含 Core B uniform 與 Advanced 的參考值；
  - `results/stage_d/d5-startup-wave/analysis.json`：`stage_d_analysis.py startup-wave`，只用 c1b 已 commit 的數據。

## 執行

- **部署**：四個 session 都是 `0ec7b92`。凍結 commit `fd95164` 之後只改了 4 個 Stage Q 檔案，Stage C / D 的凍結模組沒有變動。
  - `harness_sha256` `2b85155d…` 等於凍結證據（三個 mini smoke）；`stage_d.toml` `be7909b1…`、`pins.toml` `232f0de9…` 等於目前 repo 的檔案。
  - 每份 env 擷取都是 dirty false、`gpu_lock_held` true、vLLM 0.30.0、driver 617.42、power limit 300 W（預設）、`inherited_unlisted` []。
- **方式**：039 執行紀錄一的做法：Mac 端以 `caffeinate -i` 包住一個前景 ssh，WSL 上依序執行四個 session；每個 session 各自以 `flock -n` 取得 GPU lock，結束時釋放，任何一個回傳非 0 就停止後續。
- **第一次 D-A1**（`formal-20261023-d-a1`）在第一次 boot 前中止（039 執行紀錄一）：0 個 run、沒有用到任何階段 S 的重試，依 030 8.4 以新 id 重跑，不算進嘗試次數。

| session | 部分 | 時間（UTC） | 時數 | boot | run（有效 / 無效 / 逾時 / spill） | 031 閒置（MiB） | sentinel 漂移（吞吐 / TTFT） | 和 Core / Advanced 的偏移 |
|---|---|---|---|---|---|---|---|---|
| `-d-a1b` | D-A1 | 10-22 20:11:47 – 10-23 08:11:07 | 11.99 h | 20 | 57（57 / 0 / 0 / 0） | 651（b011 694） | 0.29% / 0.25% | −1.10%、−1.38% / +1.83%、+1.74% |
| `-d-a2` | D-A2 | 10-23 08:11:10 – 14:18:57 | 6.13 h | 14 | 49（49 / 0 / 0 / 0） | 651、655 | 1.34% / 1.45% | −1.25%、+0.07% / +1.46%、+0.57% |
| `-d-b` | D-B | 14:18:59 – 18:15:15 | 3.94 h | 10 | 20（19 / **1** / 0 / 0） | 655、872、707 | 0.18% / 0.21% | +0.03%、+0.21% / +0.65%、+0.06% |
| `-d-c` | D-C | 18:15:17 – 19:48:44 | 1.56 h | 21 | 21（21 / 0 / 0 / 0） | 707 | 0.09% / 0.12% | +0.19%、+0.11% / +0.13%、+0.51% |

- **有效性**（030 8.6，四個 session 都通過）：
  - replay 一致；sentinel 閘門通過（上限 3%）；
  - 每次 boot 的 031 閒置判定都在第 1 次嘗試通過（bound 1,018 MiB）；兩段式 spill 偵測都讀得到，沒有任何 spill；
  - rows.jsonl、每個 derived 檔、plan 與 confirm plan 的 sha256 都和 session.json 相符。
- 描述性的檢查 CROSS_SESSION_SHIFT（030 8.6，上限 5%）四個 session 都沒有觸發。
- **環境**：D-A1 與 D-A2 開頭 sentinel 的吞吐比 c1b 低約 1.5%（d-a1b 311.48、d-a2 310.99 對 316.07 tok/s）；D-A2 結尾已回到 315.16，D-B、D-C 的 sentinel 只比 c1b 低 0.2–0.3%。
  閒置 VRAM 則四個 session 都比 c1b 的 567–568 MiB 高（651–872 MiB；三個 mini smoke 已經是 651–719）。
  - 都在閘門與 031 的界線內，原因沒有記錄（Windows 端的狀態）。
  - D-B 期間 Windows 端的 GPU 記憶體變過兩次：shared 約 30 MiB 的配置在開頭 sentinel 出現、在 C-U8K-P-i1-r1 期間釋放；dedicated 從 655 變成 852、872，再到 707。都在界線內，相鄰 run 的 R 沒有可見的差異。
- **時數**：039 預估 D-A1 約 10 h、D-A2 約 10 h、D-B 4–6 h、D-C 約 2.5 h。
  - D-A1b 多約 2 h，仍在使用者「最壞約 12 h」之內：U32K 的 bench 為 6.30 h，和預估的約 6 h 相近，多出的主要是 U8K-T 與 M8K-T（bench 合計 4.92 h）和 20 次 boot 的開銷。
  - D-A2、D-B、D-C 都比預估短。D-B 是因為跳過規則：階段 C 計畫 30 個 run，只跑了 10 個。
    D-C 計畫的 20 個 run 跑了 19 個，跳過規則只省下 U2K-var 的 r5；它比預估短約 0.9 h，是因為 039 的預估高於實際需要的時間（bench 合計 1.01 h）。
  - **揭露**：四個 session 前後只隔約 3 秒，GPU lock 實際上被連續占用 23.6 h（10-22 20:11Z – 10-23 19:48Z），其他專案的 `flock -n` 在這段時間拿不到 lock。
    每個 session 單獨都在 12 h 之內，但連續占用超過 030 使用者決定時「一次約 7–9 h、最壞約 12 h」的字面範圍。之後的鏈式執行會在 session 之間留出空檔，或事先向使用者確認。

## 結果（S2：短類 P95 TTFT 2 / 6 / 20 s，P95 TPOT 100 ms；穩態時間窗）

### D-A：封閉迴路的延伸

| workload | server | C_KV / C_cap | 階段 S 的 Ĉ / Ĉ_g | 確認集合 K | C_SLO | **C_PROD** | 先失守的項目 | 旗標 |
|---|---|---|---|---|---|---|---|---|
| U32K | Core B | 3 / 6 | 2 / 2 | {1, 2, 3} | 2 EXACT | **2 EXACT** | TPOT | — |
| U8K-T | Core B + T | 13 / 26 | 5 / 4 | {3, 4, 5, 6} | 5 EXACT | **4 EXACT** | 短類 TTFT | — |
| M8K-T | Core B + T | 13 / 26 | 5 / 4 | {3, 4, 5, 6} | 5 EXACT | **4 EXACT** | TPOT | WINDOW_SENSITIVE |
| U2K-O512 | Core B | 30 / 60 | 32 / 32 | {16, 31, 32, 33} | 32 EXACT | **32 EXACT** | 短類 TTFT | WINDOW_SENSITIVE、ABOVE_CKV |
| U8K-O512 | Core B | 12 / 24 | 12 / 12 | {6, 11, 12, 13} | 12 EXACT | **12 EXACT** | 短類 TTFT | — |

（T = `--long-prefill-token-threshold 512`。C_KV = `core_b_plan.c_kv(248, 528, prompt + 輸出)`，重算與 plan 相同。）

**各點的判定**（n = 有效 run / 要求的 n；最差 R 是所有有效 run 中最大的 Y / S；mixed 另列合併的 R_long）

| workload | 點 | n | 類別 | 最差 R（短類 / TPOT） | 備註 |
|---|---|---|---|---|---|
| U32K | 1 | 5 / 5 | PASS | 0.324 / 0.135 | |
| U32K | 2 | 5 / 5 | PASS | 0.512 / 0.642 | |
| U32K | 3 | 1 / 5 | FAIL（跳過其餘 run） | 0.494 / 1.141 | Y_TPOT 114.08 ms |
| U8K-T | 3 | 3 / 3 | PASS | 0.708 / 0.158 | |
| U8K-T | 4 | 5 / 5 | PASS | 0.940 / 0.173 | 離 guard band 只差 0.010 |
| U8K-T | 5 | 5 / 5 | MARGINAL | 0.985 / 0.474 | 5 個 run 的 R_short 都在 0.968–0.985 |
| U8K-T | 6 | 1 / 5 | FAIL | 1.003 / 0.507 | Y_short 6,016 ms |
| M8K-T | 3 | 3 / 3 | PASS | 0.783 / 0.588 | R_long 0.505 |
| M8K-T | 4 | 5 / 5 | PASS | 0.886 / 0.788 | R_long 0.580 |
| M8K-T | 5 | 5 / 5 | MARGINAL（W′ 為 PASS） | 0.962 / 0.907 | R_long 0.633；只有 r2、r5 超過 0.95 |
| M8K-T | 6 | 1 / 5 | FAIL | 0.973 / 1.028 | R_long 0.796 |
| U2K-O512 | 16 | 3 / 3 | PASS | 0.501 / 0.245 | |
| U2K-O512 | 31 | 5 / 5 | PASS | 0.525 / 0.383 | |
| U2K-O512 | 32 | 5 / 5 | PASS | 0.708 / 0.443 | 見「超過 C_KV 的代價」 |
| U2K-O512 | 33 | 1 / 5 | FAIL | 1.031 / 0.476 | |
| U8K-O512 | 6 | 3 / 3 | PASS | 0.440 / 0.283 | |
| U8K-O512 | 11 | 5 / 5 | PASS | 0.423 / 0.445 | |
| U8K-O512 | 12 | 5 / 5 | PASS | 0.425 / 0.474 | |
| U8K-O512 | 13 | 1 / 5 | FAIL | 1.104 / 0.541 | |

### D-B：開放迴路（Poisson 到達）

檔案中的 "C" 是格點索引 i，到達率 = f × λ_ref，f = {0.4, 0.55, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5}[i − 1]。

| workload | λ_ref（req/s） | 階段 S（f：R 短類 / TPOT） | λ̂ | K | λ_SLO | **λ_PROD** | 先失守的項目 |
|---|---|---|---|---|---|---|---|
| U2K-P | 2.545855 | 0.4：0.463 / 0.335；0.55：0.417 / 0.459；0.7：0.517 / 0.752；0.8：0.737 / 0.854；0.9：1.236 / 1.026 ✗ | 0.8 | {0.7, 0.8, 0.9} | NT | **NT** | — |
| U8K-P | 0.62965 | 0.4：0.459 / 0.552；0.55：0.582 / 0.916；0.7：0.859 / 1.405 ✗ | 0.55 | {0.4, 0.55, 0.7} | 0.4 EXACT | **0.4 EXACT（0.2519 req/s）** | TPOT |

| workload | 點（f，req/s，N） | n | 類別 | 最差 R（短類 / TPOT） |
|---|---|---|---|---|
| U2K-P | 0.7（1.7821，535） | 0 / 5 | NT（唯一的 run 無效，其餘跳過） | — |
| U2K-P | 0.8（2.0367，612） | 1 / 5 | NT | 0.652 / 0.814 |
| U2K-P | 0.9（2.2913，688） | 1 / 5 | FAIL | 1.008 / 1.041 |
| U8K-P | 0.4（0.2519，300） | 5 / 5 | PASS | 0.622 / 0.668 |
| U8K-P | 0.55（0.3463，300） | 1 / 5 | FAIL | 0.693 / 1.073 |
| U8K-P | 0.7（0.4408，300） | 1 / 5 | FAIL | 0.771 / 1.225 |

- **U8K-P 的 0.4 是格點的最低點**。它不是 INFEASIBLE（最低點是 5 / 5 PASS），狀態是 EXACT，因為下一點 0.55 是 FAIL。真正的值在 [0.4, 0.55) λ_ref 之間，格點在這一段沒有更細的解析度。只差一格就會是 INFEASIBLE。
- **每個 run 的送出順序檢查**（039，50 ms）：18 個開放迴路 run（含無效的那個）都是 0 次倒序，全部通過。

### D-C：固定點驗證

| 點 | 條件 | C | n（有效 / 要求） | 類別 | 最差 R（短類 / TPOT） | 各 run 的 R_short |
|---|---|---|---|---|---|---|
| U2K-var | prompt 64–1,920 均勻，直連 vLLM | 32 | 4 / 5 | **FAIL** | 1.021 / 0.573 | 0.944、0.856、0.989、**1.021**（r5 依跳過規則沒有執行） |
| U8K-var | prompt 86–8,015，直連 | 8 | 5 / 5 | **PASS** | 0.409 / 0.609 | 0.389–0.409 |
| U2K-gw | U2K，經 capserve gateway | 32 | 5 / 5 | **PASS** | 0.696 / 0.923 | 0.695–0.696 |
| U8K-gw | U8K，經 gateway | 8 | 5 / 5 | **PASS** | 0.533 / 0.885 | 0.532–0.533 |

- D-C 沒有估計量，每個點只有 030 8.1 的單點類別。U2K-var 的 r5 不是無效，而是 r4 的有效 FAIL 之後依跳過規則不再測（039：「一次 FAIL 就是 FAIL」）。
- gateway 的版本：`capserve_sha256` `53488c6d…`，在部署 commit、凍結 commit 與 smoke 的部署上重算都相同。report 只檢查這個值的格式；gateway 的 Python 依賴不在任何雜湊內。

### Hypotheses（只判定 S2；預測都寫在數據之前，都是 supported）

| id | 內容 | 估計量 | 判定 |
|---|---|---|---|
| H-D1 | C_SLO(U32K) ≤ C_KV = 3 | 2 EXACT | **supported** |
| H-D2a | C_PROD(M8K-T) > C_PROD(M8K) = 2 | 4 EXACT | **supported** |
| H-D2b | C_PROD(U8K-T) < C_PROD(U8K) = 8 | 4 EXACT | **supported** |
| H-D3 | U2K-O512：C_SLO ≤ C_KV = 30 | 32 EXACT | **not supported** |
| H-D3 | U8K-O512：C_SLO ≤ C_KV = 12 | 12 EXACT | **supported** |
| H-D4 | U2K-P：λ_PROD < λ_ref | NT | **NT** |
| H-D4 | U8K-P：λ_PROD < λ_ref | 0.4 EXACT | **supported** |
| H-D6 | U2K-var 為 PASS | FAIL | **not supported** |
| H-D6 | U8K-var 為 PASS | PASS | **supported** |
| H-D7 | U2K-gw 為 PASS | PASS | **supported** |
| H-D7 | U8K-gw 為 PASS | PASS | **supported** |

- 每一項都由兩個獨立的 agent 依 039 的表重算，和 report 一致。
- H-D6 與 H-D7 依 report 和 030 H-S1 / H-S2 的寫法逐點判定。若把 H-D6 當成「兩點都 PASS」的合取命題，結果是 not supported。
- **判定的穩健度**（只作描述）：
  - H-D1 要翻轉，C_SLO 必須 ≥ 4，也就是 C 4（階段 S 的 R_TPOT 1.158）也要通過，TPOT 要快約 14%；C 3 單獨快 12% 只會讓 C_SLO = 3 = C_KV，仍然 supported。
  - H-D2a 要翻轉，M8K-T 的 C 3 要慢約 21%。
  - H-D2b 要失去 supported，最少要 U8K-T 的 C 5 快約 3.6%，而且 C 6 快約 5.3%（C 6 只有 1 個有效 run）：兩點都 PASS 時 C_PROD 為 6 LOWER，判定 inconclusive。
    要變成 not supported（C_PROD ≥ 8），C 8 在確認時必須 PASS，短類 P95 TTFT 要快約 17%（階段 S 的 R_short 1.143 降到 0.95 以下）。
  - 這些都遠大於約 1.5% 的跨 session 漂移。
  - 點估計本身比較敏感，但每一項只會往一個方向變：
    - U8K-T 的 C 6 只以 0.27% 之差 FAIL，這影響的是 C_SLO（可能是 6），不是 C_PROD，因為 C 5 本來就是 MARGINAL；
    - U8K-T 的 C 4 離 guard band 1.04%，所以它的 C_PROD 可能降為 3；
    - M8K-T 的 C 5 離 PASS 最多約 1.26%，所以它的 C_PROD 可能升為 5。
    - 反方向都超出漂移：U8K-T 升到 5 要 C 5 快約 3.6%；M8K-T 降到 3 要 C 4 慢約 7.3%。
    - 所以在約 1.5% 的漂移內，U8K-T 的 C_PROD 是 3–4，M8K-T 是 4–5。
  - U2K-var 的 r3 已經超出 guard band（0.989），所以即使 r4 沒有 FAIL，最好也只是 MARGINAL（inconclusive）。

## 判讀

### D-A1：32K 與 T

- **U32K**：C_SLO 2 等於 Core 32K 的 C_knee 2，先失守的是 TPOT。
  - C 3 = C_KV 時 KV 峰值 0.798（197 / 247 個可用 block；vLLM 的使用率為 1 − free / (248 − 1)，見 024。三個 request 同時到全長應為 198 / 247，取樣到的峰值少一個 block）；階段 S 的 C 4 時 running 峰值仍是 3、waiting 3，所以 C 4 的 TPOT（115.8 ms）和 C 3（115.4 ms）幾乎相同，多出來的只是排隊造成的 TTFT。
  - 短類 P95 TTFT 在 C 3 低於 C 2（9.87 s 對 10.11–10.24 s），和 030 揭露的 Core 反例相同。類別由 TPOT 決定，C 3 又是 K 的最高點，所以不觸發 NONMONOTONE。
- **T 的取捨**（B0 的數字來自 c1b，跨 session 的比較）：

  | workload | B0（c1b）C_SLO / C_PROD，先失守 | T（本次）C_SLO / C_PROD，先失守 |
  |---|---|---|
  | U8K | 8 / 8 EXACT，TPOT | 5 / 4 EXACT，短類 TTFT |
  | M8K | 2 / 2 EXACT，短類 TTFT | 5 / 4 EXACT，TPOT |

  - T 把 mixed 的 C_PROD 從 2 提高到 4，也把 uniform 從 8 降到 4，兩個方向都符合 039 的預測（Advanced 的 R1）。
  - 階段 S 的 U8K-T 在 C 8 的短類 Y 為 6,856 ms，和 039 依據中的「約 6.7 s」相符。
  - U8K-T 的 TTFT 失守不是 KV 造成的：C 5 / 6 的 KV 峰值只有 0.385 / 0.462。M8K-T 在階段 S 的 C 8 為 0.992。
  - T 會產生不同的 fingerprint，所以它的容量要寫在另一份 profile（030 8.8）。依 039，它只適用於宣告使用 T 的部署；
    030 8.8 的原文更窄：T 的 profile 只用於明確宣告為 M8K 的部署，並要同時報告 T 下的 U8K（本文的 U8K-T）。

### D-A2：輸出 512 與 H-D3

- **H-D3 在 U2K-O512 不成立是真的**：31、32 都是 5 / 5 PASS，而且 KV 確實被用滿（峰值 1.0，每個 run 分別有 3 與 30 次 preemption；33 為 43 次）。C_KV 的算法也正確。
- **機制**（依 024 對 vLLM 原始碼的唯讀查看，加上本次的計數）：
  - vLLM 准入時只檢查 prompt 需要的 block（⌈prompt / 528⌉ + 3），不預留輸出的 block。所以實際的准入上限是 ⌊247 / 7⌋ = 35（2K）與 ⌊247 / 19⌋ = 13（8K）：U2K-O512 在 C ≥ 35、U8K-O512 在 C ≥ 13 時，running 峰值都停在 35 與 13。
  - C_KV(prompt + 512) = 30 / 12 只是「全部 request 都到全長也不需要 preempt」的界線。
  - 兩個 workload 都滿足 C_KV(prompt + 輸出) ≤ C_SLO ≤ 准入上限：30 ≤ 32 ≤ 35，12 ≤ 12 ≤ 13。
  - C_SLO 落在這個區間的哪裡，和 preemption 造成的多秒 TTFT 停頓比例有關，但在 U2K 的實際邊界上不只由它決定：
    C 33 的 W 中有 18 個 request 超過 2 s，其中多秒停頓（2.5–7.3 s）只有 13 個（4.9%，低於 5%）。
    另外 5 個是第二波的 idx 34、51–54（2.04–2.06 s），比同波其他 request（約 1.7 s）晚一個 prefill 間隔，P95 2,062 ms 就是由這 5 個決定；階段 S 的 C 33 相同。
    它們和前一個 request 幾乎同時送出（第一波的 idx 0 / 1、19 / 20 幾乎同時結束），較可能來自開場湧入造成的送出重疊，而不是 preemption（推論，沒有量到）。
  - c1b（輸出 128）的 H-S2 成立，是因為 TPOT 先失守（U2K 在 35 的 R_TPOT 1.026、U8K 在 9 的 1.006），不是因為 KV 被量到是限制。所以 030 的 H-S2 不能延伸成「C_KV 一定是 SLO 容量的上界」。
- **超過 C_KV 的代價**（只作描述，不改估計量）：
  - C 32 的 6 個 run（含階段 S）都剛好有 8 / 256 個 W 中的 request TTFT 超過 2 s，位置固定在 idx 63 + 31k，相鄰停頓約隔 20 s；idx 63 為 4.24–4.36 s，其他 7 個為 4.97–5.22 s。
  - 3.125% < 5%，所以 P95（1.40 s）依規則通過。但在產品上，C_PROD 32 代表約每 31 個 request 有 1 個等到 SLO 的 2.1–2.6 倍。
  - idx 94 起的 7 個停頓，前一個 request 的 TPOT 約 47 ms（W 的中位數約 38 ms），符合被 preempt 後 recompute 的特徵（推論：sampler 只有計數，機制依 024 的 recompute 並 prepend 回 waiting）。
    idx 63 的前一個 request 為 38.1–38.8 ms，沒有這個特徵，idx 63 的停頓也較短。新的 request 則因為 block 不足而等待准入。這些 request 在 vLLM 內等待，gateway 無法讓它們逾時。
  - C 31 的 W 中沒有任何 request 超過 2 s（最大 1.35–1.41 s）。它的 3 次 preemption 至少有一次在第一波（第一波最後拿到首 token 的 request TPOT 約 49 ms，比同波晚結束約 6.6 s）。
  - C 33 的 W 中有 18 / 264 = 6.8% 超過 2 s，判為 FAIL；如上，其中多秒停頓 13 個，決定 P95 的是另外 5 個剛超過 2 s 的第二波 request。
    U8K-O512 的 C 13 為 36 / 174 超過 6 s，每 12 個 idx 出現一組三個停頓，是同一類現象。
  - 邊界是斷崖：U8K 的 R_short 從 12 的 0.425 跳到 13 的 1.104，U2K 從 32 的 0.708 跳到 33 的 1.031。K 中沒有任何 MARGINAL，guard band 在 KV 的斷崖前沒有提供預警。
  - 測過的點中，吞吐在 C 31（U2K）與 C 12（U8K）最高，之後下降（U8K 13 約 −11%）。U2K 的 17–30 沒有量過，所以真正的最高點可能更低。
- **和 c1b 的對照**：U2K 的 C_SLO 從 34 降到 32、C_PROD 都是 32（c1b 是 TPOT 的 guard band，本次是 33 的 TTFT FAIL）；U8K 的 C_SLO 與 C_PROD 都從 8 變成 12。
  先失守的項目從 TPOT 變成短類 TTFT。同一個 server 上，sentinel（輸出 128、C 25）的穩態 TPOT P95 約 75 ms，U2K-O512 在 C 31 只有約 38 ms。推論：穩態 TPOT 有一大部分來自新 request 的 prefill 插隊，輸出變長後攤到更多 token。
- **WINDOW_SENSITIVE**：在 W′ = [2C, N − C) 下 33 唯一的 run 的 R_short 為 0.880，該點變成 NT，估計量變成 32 LOWER。W′ 中仍有 4.76% 的 request 超過 2 s，停頓在穩態一直存在，只是剛好低於 5%。H-D3 在 W′ 下仍然是 not supported。
- **sampler 的範圍**：KV 峰值、preemption、running 與 waiting 峰值取樣自整個 `vllm bench` 子程序，包括 client 啟動與 C 個同時送出的 warmup。所以它們是整個 bench process 的值，不是量測的 N 個 request 的值。preemption 的確切時間只能從 WSL 上的 server log 確認。

### D-B：開放迴路與 H-D4

- **U8K-P 的開放迴路容量很低**：λ_PROD = 0.4 λ_ref = 0.2519 req/s。
  - 依 Little's law，各點的平均並發（λ × W 的平均 e2e）只有：0.4 為 1.21（S）/ 1.27–1.31（C），0.55 為 2.30（S）/ 2.51（C），0.7 為 3.88（S）/ 3.67（C）。封閉迴路的 C_PROD 是 8。
  - 在 0.7 的兩個 run 與 0.55 的階段 S run，server 的 running 峰值達到 13 = C_KV(8K)，KV 峰值 1.000；由 derived 重算的量測部分在途峰值為 15–19：請求群聚時 KV 瞬間滿載。
    但決定 0.55 為 FAIL 的 C-U8K-P-i2-r1 只有 running 12、KV 0.923，和 0.4 通過的 C-U8K-P-i1-r2 相同；KV 滿載的 S-U8K-P-i2 反而通過（R_TPOT 0.916）。所以 0.55 的 TPOT 失守不能歸因於 KV 滿載。
  - waiting 峰值在三個到達率都是 8–9，不能用來區分各點：sampler 涵蓋整個 bench process，包括不受 semaphore 限制、同時送出的 10 個 warmup。
  - 同一到達率下，不同 seed 的 run 之間 R_TPOT 從 0.916 到 1.073，變異很大。
  - TPOT 的尾端看起來來自停頓（多個 request 同時完成，例如 C-U8K-P-i1-r1 的 idx 131–137），和 8K 的滑移（E1 的研究對象）一致。只作事後的描述。
- **U2K-P 為 NT，原因是一個無效的 run**：
  - C-U2K-P-i3-r1（f 0.7）有 2 個 request 失敗（completed 533 / 535），依 030 8.6 (a) 無效。不是送出順序檢查、逾時、spill 或 server 結束：同一 boot 的下一個 run 完成 612 / 612。
  - block 1 的順序是 0.9 → 0.7 → 0.8。0.9 的 FAIL 只移除 0.9 以上；讓 0.7、0.8 原本計畫的 8 個 run 都沒有執行的，是 0.7 的那一個無效 run（030 的跳過規則，無效的 run 不補跑）。K 的最低點 0.7 是 NT（不是 PASS / MARGINAL，也不是 FAIL），所以依 030 8.1 的狀態表（039：其他與 030 相同），λ_SLO、λ_PROD 都沒有值，狀態為 NT；030 8.9 也把接近邊界的點為 NT 的情況列為無法解讀。
  - **只作描述，不是估計量**：階段 S 到 0.8 都通過（0.8 為 0.737 / 0.854），C-i4-r1（0.8）通過、C-i5-r1（0.9）FAIL。描述上的邊界在 0.8 與 0.9 之間，但依 030 8.4，階段 S 的 run 不進入估計量，不得用它補上 U2K 的數字。
  - 各點的平均並發：0.7 為 9.15（S）/ 10.64（C，無效），0.8 為 13.12 / 12.90，0.9 為 22.72 / 21.96；封閉迴路的 C_PROD 是 32。
- **兩個失敗 request 的原因（raw 已確認，2026-10-25；機制仍是推論，但所有可觀察的證據都和它一致）**：凍結的 bench client 的連線重用競態。
  - 失敗的 request ttft、latency、out 都是 0，也就是在收到第一個 chunk 之前就失敗。失敗當下在途只有 4 與 10 個；同一到達率的階段 S run 完成 535 / 535，更高的到達率都沒有失敗。所以和負載無關。
  - `vllm bench serve` 以 aiohttp 連線池重用連線（keepalive_timeout 60 s）。WSL 的 bench client 使用 aiohttp 3.14.3：`configs/vllm.lock.txt`（pins 的 `vllm.lock_file`）釘選這個版本，pins 註明 WSL .venv 與 lock 一致，022 只以 `--no-deps` 加裝 pandas，env 的 freeze_sha256 自 2026-10-10 到 D-B 都沒變。
    這個版本的連線池是 FIFO（取用 `popleft`、放回 `append`；3.13.2 與 3.14.5 的原始碼都是如此），總是重用閒置最久、仍連著的連線。
    vLLM 的 uvicorn 在連線閒置 5 s 時關閉（`VLLM_HTTP_TIMEOUT_KEEP_ALIVE` 預設 5，server 沒有覆寫）。client 剛好拿到一條 server 正要關閉的連線時，request 會在第一個 token 之前失敗。
  - 以 derived 的時序模擬 FIFO 重用與 5 s 關閉：兩個失敗分別重用了閒置 4.9987 s 與 4.9992 s 的連線，在 D-B 的 4,636 次重用中，離 5 s 邊緣最近的排名是第 6 與第 3。若失敗和重用無關，兩者都落在前 11 名的機率約 5 × 10⁻⁶。
  - 這主要出現在開放迴路：封閉迴路在量測期間一完成就立刻重用連線，不會閒置約 5 s。
    唯一的例外是 warmup 轉入第一波的那一刻：第一波會一次取用池中所有仍開著的連線，它們已閒置的時間等於各 warmup request 完成的時間差。
    C 超過 KV 容量或 warmup 中有大量 preemption 時（例如 U2K-O512 C 39 / 46 / 60、U8K-O512 C 14 / 16、U32K C 4），這個時間差會超過 5 s，所以這些 run 開始時有一次很小的暴露。
    已 commit 的 1,196 列 row 中，只有這一列 failed > 0，封閉迴路的 row 都是 0。產品的 gateway 正是為此把上游的 keepalive_expiry 設為 2 s（低於 5 s）。
  - b003 的 server log 大小沒有超出其他 boot 的線性模型，看不出 server 端的 traceback。
  - **raw 的確認**（2026-10-25，GPU 空閒時以 `tools/rtx/fetch_raw.sh` 唯讀取回，sha256 都和 raw_manifest 相符；raw/ 不進 git）：
    - `C-U2K-P-i3-r1.bench.log`（sha256 `96a2d194…`）：bench 跳過 ready check、10 個 warmup 完成之後開始主要的 run；兩個失敗分別是 `aiohttp.client_exceptions.ServerDisconnectedError: Server disconnected` 與 `ClientOSError: [Errno 104] Connection reset by peer`（底層 `ConnectionResetError`），兩者都發生在 `session.post` 之後、`resp.start` 讀取回應標頭時（`endpoint_request_func.py:204`），也就是 server 沒有回任何位元組就關閉了連線。
    - `b003-C-U2K-P-r1.server.log`（sha256 `3cfa05f0…`）：依 `POST /tokenize` 切成三段（這個 boot 的三個 run），第二段（C-i3-r1）的 `POST /v1/completions` 剛好 543 筆 = 成功的 533 + warmup 10，全部 200；兩個失敗的 request 沒有任何 access log。bench 期間 server 沒有任何 error 或 warning，唯一的 ERROR 是 23:56:16 關閉時的 `EngineDeadError`（teardown 的 abort，每個 boot 都有）。
    - WSL 的 `.venv` 中 aiohttp 為 3.14.3（`import aiohttp` 讀出），和 lock 一致。
    - 結論：失敗的兩筆在 server 端從未被處理，server 也沒有異常；client 在送出後、收到回應之前被 server 關閉連線。這正是 keep-alive 關閉與連線重用相撞的樣子，和上面的時序模擬一致。uvicorn 不記錄 keep-alive 的關閉，所以無法直接看到「server 在那一刻關閉了閒置連線」，機制仍標為推論；判定（U2K-P 為 NT）不變。
  - **待確認**：`raw/C-U2K-P-i3-r1.bench.log`（sha256 `96a2d194…`）中 vLLM bench 印出的錯誤。GPU 空閒時以 `fetch_raw.sh` 唯讀取回並核對 sha256，並以 `.venv/bin/python -m pip show aiohttp` 唯讀複核版本。
  - 不論原因為何，依 013 / 030 都不補跑，U2K-P 的判定維持 NT。

### D-C：長度不一與 gateway

- **U2K-var 的 FAIL 由第二波決定，但根本差異是 var 的尾端整體比 uniform 高約兩格**：
  - 尾端 TTFT 落在間距約 342 ms 的離散格點上（約 1,370、1,712、2,055、2,390 ms），2 s 在第 5、6 格之間。W 中 TTFT > 2 s 的個數為 r1 12、r2 3、r3 13、r4 16；n = 256 時要約 13–14 個以上，P95 才超過 2 s。r3（P95 1,978 ms）和 r4（2,042 ms）只差 3 個 request。
  - r4 的 16 個中有 15 個在 idx 32–63（第二波）。W′ 中 4 個 run 的 R_short 為 0.829–0.857，所以 class_late 是 NT（有效 run 不足 5 個），不是 PASS。
  - **第二波決定 W 的 P95，是 W 的定義在 2K、C 32 的結構性特徵**：第二波有 32 個 request，占 W 256 個的 12.5%，多於 P95 的 5% 尾端。c1b 的 uniform U2K 在 C 32 也是如此：W 的 P95 1,177–1,178 ms 由第二波決定，W′ 只有 1,003–1,004 ms。8K、C 8 時第二波只占 4.3%，不受影響。
  - **但 FAIL 和 uniform 的 PASS 之間的差別不在 W 的定義**：c1b uniform 第二波的最大 TTFT 只有約 1,385 ms，W 中沒有任何 request 超過 2 s；var 第二波的最大值為 2,041–2,389 ms（第 6、7 格），W 中有 3–16 個超過 2 s。
    即使排除第二波，W′ 中 var 的 P95 仍為 1,658–1,713 ms，比 uniform 高 65–71%（約兩格）。格點與計數只解釋 r3 與 r4 為何差 3 個 request 就翻轉；var 會 FAIL、uniform 不會，是因為長度不一讓尾端整體上移。
  - H-D6 的理由「較短的 prompt 只會減少 prefill 與 KV」在平均值上成立，在 P95 上不成立：var 每秒 prefill 的 token 比 c1b 少約 14%，平均 TTFT 幾乎相同（988–1,022 對 993–994 ms），但分布變寬（p5 約減半）。尾端由位置（排隊狀態）與格點決定，和該 request 自己的 prompt 長度相關很低（相關係數 0.29–0.34）。
  - 同一點的 5 個 run 重播同一條固定的長度序列（dataset 已 pin），所以 n 只量到時間上的變異。這條序列的第二波平均長度 1,094.8，高於整體的 989.3。U2K-var 的結果能否推廣到其他長度序列，沒有被量到。
- **gateway 的影響比「一次 loopback」大，但兩點都在 SLO 與 guard band 內**：
  - 和 c1b 直連比較（跨 session），用 W′ 才能看出穩態的影響：U2K 的短類 P95 從 1,003–1,004 ms 變成 1,389–1,392 ms（+38%），U8K 從 2,478–2,479 ms 變成 3,189–3,197 ms（+29%）。
  - U2K 的大部分 request 往上移了一格：W′ 中 c1b 有 202–203 / 224 個在 1,000–1,099 ms，U2K-gw 有 138–165 / 224 個在 1,300–1,399 ms。
    U8K 則分成兩群：c1b 有 126–132 / 176 個在 2,400–2,499 ms；U8K-gw 有 67–77 個上移約 350 ms、55–65 個上移約 700 ms，所以 P95 多了約 710 ms。
  - 兩者的平均 TTFT 都增加，平均 decode 時間減少相近的量，e2e 幾乎不變。U2K-gw 的最差 R_TPOT 0.923 比 c1b 的 0.932 低，是時間從 decode 移到了 TTFT，不是 gateway 帶來的好處。
  - runner 沒有擷取 gateway 的 `capserve_queue_wait_seconds` 與上游 TTFT，所以 039 的 H-D7 理由（gateway 在 C_PROD 不排隊）沒有被量到，TTFT 的增加也無法拆成 gateway 排隊與上游兩部分。H-D7 的 supported 只代表 client 端的 SLO 在 gateway 後仍然成立。
  - U2K-gw 在 C 32 的 n = 5 PASS 是「經過 gateway」的條件，不是 c1b 直連 C 32（c_low，n = 3）的重複，不能說成同一個估計量被補強。
- **開場湧入**：U2K-gw、U8K-gw 的第一波最大 TTFT（10.37–10.42 s、11.00–11.10 s）和 c1b 直連相近；U2K-var 為 4.61–4.66 s，U8K-var 為 5.44–5.52 s。

### D5：開場湧入（描述，事後）

來源是 c1b（035）已 commit 的逐 request 檔，取每個有效的階段 S / C run 的第一波（idx < C）最大 TTFT。

| workload | 測過的 C（run 數） | C_PROD 的第一波最大 TTFT（平均 / 最大） | ÷ S2 門檻 | C_PROD 時第一波超過門檻的個數 |
|---|---|---|---|---|
| U2K | 13（26） | C 32：10.35 / 10.35 s | 5.18 | 每個 run 27 / 32 |
| U8K | 9（22） | C 8：11.03 / 11.04 s | 1.84 | 每個 run 4 / 8 |
| M2K | 4（15） | C 2：0.65 / 0.70 s | 0.35 | 0 / 2 |
| M8K | 4（15） | C 2：2.74 / 2.75 s | 0.46 | 0 / 2 |

- **第一波的 TTFT 隨 C 線性增加，斜率就是每個 prompt 的 prefill 時間**：
  - 檔案中的全範圍擬合（U2K slope 365.4 ms、intercept −448.6 ms；U8K 1,462.5 / −431.4）跨過 C_KV 的不連續處，模型設定不對，intercept 沒有物理意義。
  - 只用 C ≤ C_KV 的點：U2K 324.2 ms / request（R² 0.99997），U8K 1,389.2 ms / request。換算成 prefill 速度約 5,920 與 5,800 token/s。兩條斜率的比 4.28，接近 prompt 長度的比 4.2。這一段的殘差仍有輕微的凸形，所以只能引用斜率。
  - 超過 C_KV 時有一段跳升：U2K 從 35 到 36 多了 2.38 s，而 C ≤ C_KV 時每多一個 request 只多約 0.31–0.35 s；U8K 從 12 到 16 共多了 7.14 s（平均每個 1.79 s，C ≤ C_KV 時約 1.37–1.43 s）。
    以 C ≤ C_KV 的直線為基準，C 36 高出 2.09 s，C 16 高出 1.64 s。多出來的 request 要等第一個 request 完整結束才能准入（c1b 所有 run 的 preemption 都是 0）。
- **超過門檻的只有第一波**：C_PROD 時第二波（idx 在 [C, 2C)）的最大 TTFT 已回到門檻內（U2K 1.38–1.39 s、U8K 2.47–2.48 s）。
  U8K 的第二波已經和穩態相同。U2K 的第二波仍高於穩態：32 個中有 15 個高於 W′ 的最大值（約 1.01 s），W 的 P95 就是被它們從 1.00 s 拉到 1.18 s（見 D-C）。
- **mixed 的數字不能推廣**：007 dataset 的第一個長 request 在 idx 4，c1b 的 mixed 只測到 C ≤ 4，所以第一波從來沒有長 request。mixed 的 0.70 / 2.75 s 只描述「全部是短 request 的湧入」。
  M2K 的 0.70 s 來自 C-M2K-c2-r5 一個整體變慢的 run（mean TTFT、P95、duration 都較高），典型值約 0.645 s。
- **和 039 字面的小差異**：程式排除了 sentinel 的 C 25（039 寫「每個測過的 C」），也只對 uniform 擬合（039 寫「例如以 C 的線性函數擬合」）。加入 C 25 後結論不變（C ≤ 35 的斜率 324.0 ms）。
- **量測情境**：每個 run 在第一波之前先送 C 個 warmup，所以 D5 描述的是「C 個 request 同時湧入閒置但已暖機的 server」，不是 boot 後的冷啟動。
- **產品上的換算**（內插，2K 的 C 5–7、8K 的 C 5–6 沒有量過）：要讓第一波的最大 TTFT 也在 S2 內，同時 prefill 的 request 數套上 030 的 guard band 後，2K 約 5 個（約 9,600 token，直線預測 1.60 s），8K 4 個（約 32,256 token，實測最大 5.49 s，R 0.915）。
  這是單一 request 的最大值和 P95 門檻的比較，產品設計要另寫 decision。

## 揭露與限制

- 只有一台機器（RTX 5070 Ti，WSL2）、一個模型、Core B 或 T 的 argv；輸出只有 128 與 512（512 只量了 uniform）；T 只量了 8K。
- **mixed 沒有開放迴路與 D-C 的數字**（039 事先寫明要列入限制）：D-B 不做 mixed，因為 mixed 的容量只有 2，Poisson 到達下的變異會被長 request 主導。
  D-C 也只驗證 uniform 兩類在 C_PROD 這一點，mixed 在長度不一或經過 gateway 時都沒有量過（report 的 not_supported 也這樣寫）。
- D-A 的結果是封閉迴路、准入之後的延遲；D-B 是不靠准入控制時 vLLM 本身的到達率上限，不是 gateway 的上限（039）。
- **開放迴路的協定缺陷**（raw 已確認與推論一致，見 D-B）：在凍結的協定下，任何開放迴路 run 都有一個和到達率無關的小機率，被 client 端的連線錯誤判為無效（D-B 的 18 個開放迴路 run 共 7,397 個 request，失敗 2 個）。
  封閉迴路在量測期間不受影響，只在 warmup 轉入第一波、而且 warmup 的完成時間相差超過 5 s 時有一次很小的暴露；已 commit 的封閉迴路 row 都沒有失敗。要修正（例如提高 server 的 keep-alive，或讓 client 不重用接近過期的連線），需要新的 decision log 與新的 session，不能補跑 D-B。
- 固定長度的封閉迴路排程幾乎是決定性的：同一點的 run 之間 preemption 次數完全相同（U2K-O512 C 32 每個 run 都是 30 次），n = 5 確認的是重現性，不是不同的狀態（030 8.9 的格狀結構限制）。
- raw/ 沒有取回 Mac（STORAGE_PLAN），所以 raw → derived 這一步只能信任 runner 的檢查與 sha256；bench 實際送出的 input_lens、gateway log 與 server log 都只在 WSL。
- D-C 的 verify 分析不輸出旗標；U2K-var 的時間窗敏感與 STRADDLE 型的情況只能從 class_late 與各 run 的結論看出來。
- D-C 的 `secondary_slo` 只輸出各點的類別；D-A / D-B 的只輸出估計量、先失守的項目與旗標。S1 / S3 下的 class_late、各點的類別與 R，是核對時另外算的，不在 analysis.json 中。

## 次要 SLO（只作界線，不判定）

- **S1**（1 / 3 / 10 s、50 ms）：U32K 1 EXACT；U8K-T、M8K-T UNSTABLE；U2K-O512 UNSTABLE（STRADDLE）；U8K-O512 12 EXACT；U2K-P NT；U8K-P INFEASIBLE（SEARCH_MISMATCH）；D-C 四點都 FAIL。
- **S3**（TPOT 200 ms）：U32K 2 LOWER；U8K-T 5 / 4 EXACT；M8K-T 5 LOWER / 4 EXACT（WINDOW_SENSITIVE）；U2K-O512 32 EXACT（WINDOW_SENSITIVE、ABOVE_CKV）；U8K-O512 12 EXACT；U2K-P NT；U8K-P 0.4 LOWER；D-C 中 U2K-var FAIL，其他三點 PASS。

## 產品上的使用（039）

- **profile 尚未產生**：`capserve.profile_gen` 目前只接受 Stage C 的 session（`results/capacity/`），要擴充到 Stage D 才能產生 D-A 的 verified profile。這是產品變更，另外 commit 並測試。
  - B0 的 class：U32K（2，exact）、U2K-O512（32，exact，附 ABOVE_CKV 與 WINDOW_SENSITIVE，並寫明 C 32 時 3.125% 的 request 停頓 4.2–5.2 s）、U8K-O512（12，exact）。
  - T 的另一份 profile：U8K-T（4）、M8K-T（4，附 WINDOW_SENSITIVE）。
  - 適用範圍：封閉迴路、准入之後的延遲、各自的輸出長度。
- **不改正式設定的釘選**：正式設定仍是 `verified-formal-20261017-c1b.json`；要不要服務新的 class 由使用者決定（037 §2）。
- **D-B 與 D-C 不產生 profile**，只寫進適用條件：
  - uniform-2k 沒有經過驗證的開放迴路到達率（NT）；uniform-8k 的開放迴路上限為 0.2519 req/s（0.4 λ_ref，EXACT），遠低於封閉迴路 C_PROD 對應的 λ_ref。不靠准入控制直接暴露 vLLM 時，要以這個數字為準。
  - uniform-8k 的上限 8 在長度不一與經過 gateway 時都成立。
  - uniform-2k 的上限 32 在長度不一時沒有通過驗證：1 / 4 個 run 超過 2 s，另有 r3 的 R 0.989 超出 guard band，所以即使 r4 沒有超標也只會是 MARGINAL。
    超標集中在開場後的第二波，但 W′ 中的 P95 仍為 1.66–1.71 s，比 uniform 的 1.00 s 高約 70%。
  - uniform-2k 經過 gateway 時上限 32 成立，但最差 R_TPOT 0.923，離 guard band 只剩 0.027。
  - mixed-2k / mixed-8k 的適用條件不變，仍只到 c1b 的封閉迴路、直連、固定長度：開放迴路、長度不一、經過 gateway 都沒有驗證。
- **多 class 准入與開場湧入的產品設計**（NEXT 中「等 Stage D 的結果」的兩項）：依本文 D5 與 D-A2 的數字另寫 decision，不在本文決定。
- **補記（2026-10-24）**：以上各項的產品設計與實作見 043。本文其他內容（數據與判定）都不變。
  - 產生器接受 D-A 的 session，寫出三份 verified profile，都只 commit、不釘選：B0 依 session 拆成 `-d-a1b-b0`（U32K）與 `-d-a2-b0`（U2K-O512、U8K-O512），T 一份 `-d-a1b-t`（U8K-T、M8K-T）。
    和上面把 B0 的三個 class 放在一份 profile 的寫法不同，理由見 043 §2。
  - 輸出上限大於 128 的 class（兩個 O512）與 T 的 `uniform-8k-t` 不能作為 active class；deploy 只啟動 B0。
  - D-B 與 D-C 寫進 `service/README.md` 的適用條件；多 class 准入與開場湧入的 pacing 只有候選設計，要先通過驗證研究才上線。

## 039 的文字更正（不影響數據與判定）

- 執行紀錄一的「2026-10-23 20:11Z（UTC）起」應為 **2026-10-22 20:11:47Z**（台灣時間 10-23 04:11）。session id 用的是當地日期。
- 「啟動後 44 秒」：session.json 記錄的是 38.7 s（19:59:40.21Z – 20:00:18.91Z）。44 s 較可能是 chain log 從 start 到 exit 的時間，包含 `stage_d_plan.py --formal`。
- 「中止的 session 已取回並 commit（只有 session.json）」：實際 commit 的是 5 個檔案（session.json、plan.json、env/preflight.json、raw_manifest.json、0 byte 的 rows.jsonl）。
- 先前 chain 的輸出中，D-A 三個 workload 的 `"C": null` 來自 `stage_d_plan.py` 印出的 plan 摘要：`C` 與 `n` 只有 D-C 的固定點才有，對判讀沒有任何意義。

## 獨立核對（2026-10-24）

- 每個 session 與 D5 各由一個 agent 抽取結果並依 039 的表重算判定，再由第二個 agent 嘗試反駁（共 10 個 agent，唯讀；重算時不 import 分析程式）。
- 兩者都從 rows 與 derived 檔重算 Y、R、類別、估計量、狀態、旗標、replay（階段 S 的搜尋：D-A 為夾擠與二分、D-B 為線性掃描；K、block 與 point 的洗牌、跳過規則），全部和 analysis.json 一致。
  另外在記憶體中重跑 `stage_d_report.report()` 與 `startup_wave()`（不寫檔），輸出和已 commit 的檔案逐位元組相同。
- 反駁找到的都是說明文字或機制解讀的錯誤（例如 D-A2 以線性 token 進度重建 KV 佔用不成立、D-B 的無效 run 原因原本寫成無法判定），判定與數據沒有任何不符。本文採用反駁後的版本。
- 本文的初稿再由 4 個 agent 逐段核對、每項指出的錯誤由第二個 agent 複核（17 個 agent）：確認 12 項說明或數字的錯誤（例如穩健度的算法、C 33 的 FAIL 由哪些 request 決定、U8K-P 的 KV 滿載不是 0.55 失守的原因、aiohttp 的版本、D5 的跳升數字、漏列 mixed 的限制），全部已修正；判定與估計量沒有任何錯誤。
