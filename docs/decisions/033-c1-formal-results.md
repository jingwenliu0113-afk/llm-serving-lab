# 033 — Stage C 正式結果（formal-20261017-c1）：U2K C_PROD 32（下界）、M2K / M8K 2、U8K 無法檢定

- 狀態：**已被取代**。重跑的 `formal-20261017-c1b` 有效，依 034 事先寫定的規則，四個 workload 全部以它為準（decision 035）。本文的數據與判定保留不改。
- 日期：2026-10-17
- 對應：030（§8，凍結 commit `b3c9109`）、031（閒置判定與 spill）、032（C0 → 分支 A）、029（verified profile）
- 輸出：
  - `results/capacity/formal-20261017-c1/analysis.json`：`capacity_report`，在 Mac 上執行，含 Advanced 的參考值；
  - `service/profiles/verified-formal-20261017-c1.json`：`capserve.profile_gen` 產生。依 034 已從 `service/profiles/` 移除，git 歷史保留。

## 執行

- **部署與方式**：部署 `b3c9109`，manifest 216 個檔案相符。在 GPU lock 下以單一 ssh 前景指令執行（030 8.4）。
- **時間**：2026-10-17 02:51:55 – 10:18:35，共 7.44 h，事先估計約 7.5 h。
- **開始前主機狀態**：Windows CPU 0.4%，可用 RAM 23.9 GB；031 的閒置 616 MiB。
- **run 統計**：
  - 23 次 boot、74 個 run，69 個有效、5 個無效、0 個逾時；
  - 階段 S 有 28 個 run，全部有效；階段 C 有 44 個 run，其中 39 個有效；
  - 兩次 sentinel 都有效。
- **有效性檢查**（8.6）：
  - replay 一致；
  - sentinel 的閘門通過：throughput 漂移 0.08%、TTFT 漂移 0.03%，上限 3%；
  - 沒有 CROSS_SESSION_SHIFT：和 Core / Advanced 相差 +0.4% / −0.2%，上限 5%；
  - env 有效，argv、block 數（248）與版本都相符。

## 結果（S2：TTFT 2 / 6 / 20 s，TPOT 100 ms；穩態時間窗）

| workload | 階段 S 的 Ĉ / Ĉ_g | 確認集合 K | C_SLO | **C_PROD** | 先失守的項目 | 旗標 |
|---|---|---|---|---|---|---|
| U2K | 34 / 32 | {32, 33, 34, 35} | 32 LOWER | **32 LOWER** | TPOT | SEARCH_MISMATCH |
| U8K | 8 / 8 | {4, 7, 8, 9} | NT | **NT** | TPOT | — |
| M2K | 2 / 2 | {1, 2, 3} | 2 EXACT | **2 EXACT** | 短類 TTFT | — |
| M8K | 2 / 2 | {1, 2, 3} | 2 EXACT | **2 EXACT** | 短類 TTFT | — |

**各點的判定**

| workload | 點 | 有效 / 要求的 n | 類別 |
|---|---|---|---|
| U2K | 32 | 3 / 3 | PASS |
| U2K | 33 | 3 / 5 | NT |
| U2K | 34 | 3 / 5 | NT |
| U2K | 35 | 1 / 5 | FAIL（R_TPOT 1.026，跳過其餘 run） |
| U8K | 4 | 2 / 3 | NT |
| U8K | 7 | 2 / 5 | NT |
| U8K | 8 | 2 / 5 | NT |
| U8K | 9 | 1 / 5 | FAIL（R_TPOT 1.006） |
| M2K | 1、2 | 5 / 5 | PASS |
| M2K | 3 | 1 / 5 | FAIL（短類 R 3.31） |
| M8K | 1、2 | 5 / 5 | PASS |
| M8K | 3 | 1 / 5 | FAIL（短類 R 1.28） |

**Hypotheses**（8.5，只判定 S2）
- **H-S1（mixed 降低容量）**：
  - 2K：**supported**，mixed 2 EXACT < uniform 32 LOWER，和預測相同；
  - 8K：**NT**，因為 uniform 為 NT。
- **H-S2（C_SLO ≤ C_KV）**：
  - U2K：**inconclusive**，32 LOWER ≤ 35；
  - U8K：**NT**。

**只作描述**（不進入估計量，也不寫進 profile）
- **全部 request 的統計量（C^full）**：U2K 為 UNSTABLE，因為開場湧入使 K 的最低點 32 也 FAIL。M2K、M8K 和穩態相同，都是 2。
- **和 Core 的 C_knee 比較**：U2K 為 32 對 25（+7）；M2K 為 2 對 25，M8K 為 2 對 8。
- **SEARCH_MISMATCH（U2K）**：只做一次負載測試會得到 Ĉ = 34，比確認後的 C_SLO 高 2。
- **U2K 33、34 已有的有效 run**：都符合 S2，但最差的 R_TPOT 分別為 0.971、0.999，都高於 1 − m = 0.95。
  就算 5 次都有效且都符合，它們最多只是 MARGINAL，C_PROD 仍會是 32，差別只在 C_SLO。這只是描述，狀態依規則仍為 NT。
- **U8K 已有的有效 run**：4、7、8 各 2 次，全部符合，最差 R_TPOT 依序為 0.449、0.773、0.883；9 為 FAIL。
  階段 S 的 Ĉ 是 8，和 Core 的 C_knee 8 相同。這些都不足以構成任何估計值。
- **次要 SLO**（只作界線）：
  - S1：M2K、M8K 都是 1；U2K 為 UNSTABLE；U8K 為 NT。
  - S3：U2K 為 32 LOWER；其餘見 analysis.json。

## 5 個無效的 run：spill（031 / 8.6 (g)），原因已查明

| boot | 時間 | 閒置 | shared usage：boot 前 → 就緒 → bench 後 | 結果 |
|---|---|---|---|---|
| b013 M8K r3 | 08:08 | 605 MiB | 34.2 → 114.2 → 117.8 → 86.8 | 有效（bench 後下降，沒有超過容許值） |
| b014 U8K r3 | 08:30 | **870 MiB** | 4.8 → **212.8（+208.0 > 160）** | 就緒時 spill：計畫中的 C-U8K-c8 / c7 / c4-r3 都無效，不跑 bench |
| b015 U2K r3 | 08:31 | 702 MiB | 4.8 → 148.8（+144，≤ 160）→ 86.8 | 有效 |
| b017 U2K r4 | 08:56 | 688 MiB | 4.8 → 84.8 → **183.9（+99.1 ≥ 64）** | C-U2K-c33-r4 spill，同一 boot 的 c34-r4 沒有執行 |

**原因**：之後唯讀查 Windows，發現 **BingWallpaper 在 2026-10-17 08:29 自行重新啟動**。
- 它的 WebView2 子程序在 NVIDIA 卡上用了約 190 MiB，時間和 b014 的閒置值跳升（605 → 870 MiB）完全吻合。
- 使用者在 10-16 22:10 結束過 BingWallpaper，它是自己重新啟動的。
- 另外，Radeon Software 在 10-16 22:12（我結束它之後）也已自行重新啟動。它整個 session 都在（約 38 MiB），已計入閒置值。
- 我沒有結束或修改任何 Windows 程序與設定。

**判讀**
- b014 就緒時，dedicated 為 15,220 MiB，比正常的 15,261 MiB 低，但閒置值多了 254 MiB；shared 同時多了 128 MiB。
  這代表 vLLM 的部分配置被移到系統記憶體，是真的 spill，偵測正確。b017 也是在 bench 期間 VRAM 接近滿載時發生。
- 依 030 的規則：無效的 run 不補跑（013）；之後的 block 不再測該點與更高的點（8.4 跳過規則）。
  所以 U8K 的 4、7、8 和 U2K 的 33、34 都停在有效 run 少於 n，判為 NT。
- **031 的閒置判定沒有擋住這次 spill**：
  - b014 的閒置 870 MiB 低於上限 1,018 MiB，所以通過了判定，但就緒時仍然 spill。
  - 原因是上限只是「推導的餘裕」，沒有考慮桌面程式在 boot 期間繼續增加用量，也沒有考慮 WDDM 不會把 VRAM 用到全滿。
  - 真正擋下 spill 的是 spill 偵測，這正是 031 設計兩道防線的理由。
  - 是否調整 031，要等使用者決定之後另寫 decision；本次結果不受影響。

## 依 8.9 的解讀

- **U8K**：所有接近邊界的點都是 NT，屬於 8.9 列出的「無法解讀」。不報告任何 U8K 容量。
- **U2K**：C_PROD = 32 是有效的下界（LOWER），可以寫進 profile。C_SLO 也只是下界。
- **M2K、M8K**：C_PROD = 2 EXACT。mixed 流量中有 32K 的長 request，並發 3 時短類的 TTFT 就超過 S2。
- 本 session 只代表這台機器、這個模型、Core B 的 argv 與 8.9 列出的範圍。

## Verified profile（029 / 8.7）

由產生器產生，沒有任何覆寫：`service/profiles/verified-formal-20261017-c1.json`。

| class | 狀態 | max_concurrency = max_queue | bound |
|---|---|---|---|
| uniform-2k（prompt ≤ 1,920） | verified | 32 | lower |
| uniform-8k（prompt ≤ 8,064） | **unavailable**（C_PROD 為 NT） | — | — |
| mixed-2k（prompt ≤ 32,640） | verified | 2 | exact |
| mixed-8k（prompt ≤ 32,640） | verified | 2 | exact |

- uniform-8k 為 unavailable 時，以它作為 active class 的 `serve` 與 `deploy up` 一律拒絕啟動，任何旗標都不能覆寫。
- 其他 class 是第一份正式的產品上限。它們只適用於 8.7 寫明的範圍：封閉迴路、128 個 output token、准入之後的延遲。

## 待使用者決定：U8K 是否補做

凍結的協定在 session 完成後沒有補跑的規定。任何補做都是研究決策，需要事先寫好的 decision log，而且要在看到新數據之前寫定。

可選的做法：
1. **接受現狀**：U8K 報告為 NT，產品不提供 uniform-8k。不再使用 GPU。
2. **以相同的凍結協定跑一個新的完整 session**（約 7.5 h，程式不改）。
   - 事先規定：新 session 有效時，**四個 workload 全部**以新 session 為準，profile 也改由新 session 產生；兩個 session 都報告。
   - 這樣不能逐 workload 挑選結果，避免 forking paths。
   - 前提是使用者先讓 BingWallpaper 不會自動啟動。這屬於系統設定，我不修改。
3. **只補 U8K**（約 2–2.5 h）：要修改凍結的程式與設定，等於重新凍結，程式改動與審查成本最高。不建議。

**使用者決定（2026-10-17）**：選擇第 2 個做法，並由我在開始前結束 BingWallpaper 的程序。事先寫定的取捨規則見 decision 034。
