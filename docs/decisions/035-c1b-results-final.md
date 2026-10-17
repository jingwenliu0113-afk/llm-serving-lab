# 035 — Stage C 重跑的結果（formal-20261017-c1b）：有效，依 034 成為最終結果

- 狀態：**最終結果**。034 事先寫定的取捨規則判定，新 session 有效，所以四個 workload 全部以它為準。
- 日期：2026-10-18
- 對應：030（§8，凍結 commit `b3c9109`）、031、033（第一個 session，已被本 session 取代）、034（重跑與取捨規則）
- 輸出：
  - `results/capacity/formal-20261017-c1b/analysis.json`：`capacity_report`，在 Mac 上執行，含 Advanced 的參考值；
  - `service/profiles/verified-formal-20261017-c1b.json`：`capserve.profile_gen` 產生，沒有任何覆寫。

## 執行

- **開始條件**（034）：
  - 10-17 18:19 lock 空出；Windows CPU 0.6%，可用 RAM 23.9 GB。
  - 當時 BingWallpaper 沒有在執行，所以不需要結束它。
  - 18:20 在 WSL 取得 lock，持有 lock 時的 memory.used 為 567 MiB（≤ 950）。
- **部署與方式**：沿用研究目錄的部署 `b3c9109`，和第一個 session 位元組相同。GPU lock 下單一 ssh 前景指令。
- **時間**：2026-10-17 18:20 – 10-18 02:40，共 8.33 h。
- **run 統計**：
  - 26 次 boot、80 個 run，**全部有效**，0 個逾時、0 個 spill；
  - 階段 S 28 個、階段 C 50 個、sentinel 2 個。
- **環境**：
  - 每次 boot 的 031 閒置值都在 567–568 MiB。
  - shared usage 就緒時一律 +80 MiB（b021 為 +79.8 MiB）。
  - bench 後相對就緒時的增量最大為 +33.6 MiB（b004 的 S-M8K-c1，下一個 bench 回到 +2.0 MiB），在 64 MiB 以內。
- **有效性檢查**（8.6）：
  - replay 一致；
  - sentinel 閘門通過：throughput 漂移 0.19%、TTFT 漂移 0.18%，上限 3%；
  - 沒有 CROSS_SESSION_SHIFT：和 Core 的 throughput 差 +0.4% / +0.2%，和 Advanced 的 TTFT 差 −0.3% / +0.0%；
  - env 有效，argv、block 數（248）與版本都相符。
- **034 的取捨**：`capacity_report` 接受這個 session（formal、env 有效、沒有中止、sentinel 與 replay 都通過）。
  所以四個 workload 的估計量、旗標與 hypothesis 全部以本 session 為準，033 的結果改標為「已被取代」。

## 結果（S2：TTFT 2 / 6 / 20 s，TPOT 100 ms；穩態時間窗）

| workload | 階段 S 的 Ĉ / Ĉ_g | 確認集合 K | C_SLO | **C_PROD** | 先失守的項目 | 旗標 |
|---|---|---|---|---|---|---|
| U2K | 34 / 32 | {32, 33, 34, 35} | 34 EXACT | **32 EXACT** | TPOT | STRICT_COARSE |
| U8K | 8 / 8 | {4, 7, 8, 9} | 8 EXACT | **8 EXACT** | TPOT | — |
| M2K | 2 / 2 | {1, 2, 3} | 2 EXACT | **2 EXACT** | 短類 TTFT | — |
| M8K | 2 / 2 | {1, 2, 3} | 2 EXACT | **2 EXACT** | 短類 TTFT | — |

**各點的判定**（n = 有效 run / 要求的 n；最差 R 是所有有效 run 中最大的 Y / S）

| workload | 點 | n | 類別 | 最差 R（短類 / TPOT） |
|---|---|---|---|---|
| U2K | 32 | 3 / 3 | PASS | 0.589 / 0.932 |
| U2K | 33 | 5 / 5 | MARGINAL | 0.662 / 0.971 |
| U2K | 34 | 5 / 5 | MARGINAL | 0.684 / 0.999 |
| U2K | 35 | 1 / 5 | FAIL（跳過其餘 run） | 0.688 / 1.026 |
| U8K | 4 | 3 / 3 | PASS | 0.464 / 0.449 |
| U8K | 7 | 5 / 5 | PASS | 0.412 / 0.774 |
| U8K | 8 | 5 / 5 | PASS | 0.413 / 0.884 |
| U8K | 9 | 1 / 5 | FAIL | 0.414 / 1.006 |
| M2K | 1、2 | 5 / 5 | PASS | 2：0.347 / 0.642；R_long 0.339 |
| M2K | 3 | 1 / 5 | FAIL | 3.314 / 0.626 |
| M8K | 1、2 | 5 / 5 | PASS | 2：0.409 / 0.608；R_long 0.361 |
| M8K | 3 | 1 / 5 | FAIL | 1.276 / 0.713 |

- **U2K 的 C_PROD 與 C_SLO 不同**：
  - 33、34 的 5 次 run 都符合 S2，所以 C_SLO = 34；
  - 但最差的 R_TPOT（0.971、0.999）都超過 1 − m = 0.95，是 MARGINAL，所以 C_PROD 停在 32。
  - 這就是 STRICT_COARSE 旗標的意思：C_PROD < C_SLO − 1。
  - 34 的 TPOT 離門檻只有 0.1%，guard band 正是為這種情況而設。

**Hypotheses**（8.5，只判定 S2；兩項的事先預測都是 supported）
- **H-S1（mixed 降低容量）**：
  - 2K：**supported**，mixed 2 < uniform 34；
  - 8K：**supported**，mixed 2 < uniform 8。
- **H-S2（C_SLO ≤ C_KV）**：
  - U2K：**supported**，34 ≤ 35；
  - U8K：**supported**，8 ≤ 13。

**只作描述**（不進入估計量，也不寫進 profile 的上限）
- **全部 request 的統計量（C^full）**：
  - U2K 為 UNSTABLE：開場湧入使 K 的最低點 32 也 FAIL；
  - U8K、M2K、M8K 和穩態相同。
- **開場湧入**：在 C_PROD 時，第一波（idx < C）最大的 TTFT：
  - uniform-2k 10.35 s；
  - uniform-8k 11.04 s；
  - mixed-2k 0.70 s；
  - mixed-8k 2.75 s。
  - C 個 request 同時到達時，uniform 兩類第一波的 TTFT 遠高於穩態的 SLO：uniform-2k 10.35 s 對 2 s，uniform-8k 11.04 s 對 6 s。
  - mixed 兩類的第一波仍在 SLO 內：mixed-2k 0.70 s 對 2 s，mixed-8k 2.75 s 對 6 s。
  - profile 的 evidence 有記錄，產品文件也寫明。
- **和 Core 的 C_knee 比較**：U2K 34 對 25（+9），U8K 8 對 8（0）；M2K 2 對 25，M8K 2 對 8。
- **SEARCH_MISMATCH**：沒有出現，各 workload 的 |C_SLO − Ĉ| ≤ 1。
- **次要 SLO**（只作界線）：
  - S1（1 / 3 / 10 s、50 ms）：U2K UNSTABLE、U8K 4 LOWER、M2K / M8K 1；
  - S3（200 ms）：U2K 34 LOWER、U8K 8 LOWER、M2K / M8K 2。

## 和第一個 session（033）的比較（只作描述，不合併）

| workload | c1（033，已被取代） | c1b（本 session，最終） |
|---|---|---|
| U2K C_PROD / C_SLO | 32 LOWER / 32 LOWER | 32 EXACT / 34 EXACT |
| U8K C_PROD / C_SLO | NT / NT | 8 EXACT / 8 EXACT |
| M2K、M8K | 2 EXACT | 2 EXACT |
| 階段 S 的 Ĉ（U2K / U8K / M2K / M8K） | 34 / 8 / 2 / 2 | 34 / 8 / 2 / 2 |

- **兩個 session 能比較的部分都一致**：階段 S 的選點相同，U2K 的 C_PROD 與 mixed 的值相同，各點最差 R 的差異都在 0.04 以內。
- **差別只來自 033 的 spill**：沒有 spill 的這次，U2K 的 33、34 與 U8K 的 4、7、8 都湊足了 n 次有效 run。U8K 9 在兩個 session 都是 1 / 5 FAIL，沒有受到 spill 影響。
- 依 034 不合併兩個 session 的 run，也不逐 workload 挑選。

## Verified profile（029 / 8.7）

`service/profiles/verified-formal-20261017-c1b.json`，四個 class 都是 verified、bound 都是 exact：

| class | 接受的 prompt | max_concurrency = max_queue |
|---|---|---|
| uniform-2k | ≤ 1,920 token | 32 |
| uniform-8k | ≤ 8,064 token | 8 |
| mixed-2k | ≤ 32,640 token | 2 |
| mixed-8k | ≤ 32,640 token | 2 |

- 依 034，舊的 `verified-formal-20261017-c1.json` 從 `service/profiles/` 移除，git 歷史保留。
- `service/deploy/rtx-verified.toml` 的 active class 維持 uniform-2k。
- **適用範圍**（8.9）：封閉迴路、128 個 output token、准入之後的延遲；這台 RTX 5070 Ti（WSL2）、這個模型與 Core B 的 argv。
  開放迴路、其他長度的輸出與開場湧入都不在估計量內。
- **RTX 產品整合測試**（release `7513d62`，2026-10-18）：
  - verified profile 的 IT-0、1、2、4、5、6 全部通過；
  - IT-3 以 provisional-rtx 在同一 release 上通過；
  - 細節與理由見 029 最後的補記。
- **獨立核對**（2026-10-18，在 commit 之後、push 之前）：
  - 另一個 agent 不 import 分析程式，自己從 rows 與逐 request 檔重算：392 項比較，0 項不符。
  - `capacity_report` 能逐位元組重現 analysis.json；034 的各項條件都符合。
  - 文件中 5 處措辭或數字的錯誤已修正，都是說明文字，判定與數據不變。
