# 012 — Core A Mac pilot（4B）的發現與凍結前調整

- 狀態：Accepted（2026-10-08）
- 日期：2026-10-08
- 依據：pilot session `results/core_a/macos-metal/pilot-20261008-4b`（22 / 22 ok）。
  plan 由 `benchmarks/core_a_pilot.py` 產生，報告為遮蔽版 `pilot_report.json`。
- 時點：Stage A 尚未凍結（`freeze.mac = false`），沒有任何正式數據。
  pilot 只看 noise、記憶體量測方式與執行參數，不看任何 hypothesis 對比（遮蔽，見 011）。

## 發現與決策

### 1. `--poll` → Mac 固定 P = 0（§4 規則的結果）

- sentinel cell（4B-Q4_K_M decode @ 2048）上 poll 0 / 50 交替各 3 個 process。
  decode tok/s 中位數：poll 0 為 54.89，poll 50 為 53.54。poll 0 沒有慢超過 3%，依規則用 **0**。
- 交替順序是每對都先跑 poll 0，而 session 內速度隨時間下降（見 4），所以順序對 poll 0 略為有利，估計約 1–2%。
  即使扣掉，poll 0 仍不比 poll 50 慢，結論不受影響。

### 2. fit run 加上 `-lv 4`（只影響 log）

- pinned common 把 llama.cpp library 的 INFO log 對應到 verbosity 4（trace），預設門檻是 3，所以 `llama-completion`
  的 log 裡沒有 `llama_context: n_ctx / flash_attn` 與 `graph: nodes = …, splits = …`。
  連帶 `-fa auto` probe 的 "Flash Attention enabled" 也看不到，pilot 的 probe 因此**無法判定**，需要重跑。
- 決策：fit 指令加上 `-lv 4`。它只改變 log 輸出，不影響計算。`llama-bench -v` 原本就會輸出這些 log。
- `-fa auto` probe 用新 flag 重跑（與 9B pilot 一起，pending）。
- 已確認：所有 bench process 的 `graph: … splits = 2`，也就是 token_embd 在 CPU_Mapped 加上 Metal，
  **沒有 fused GDN 默默退回 CPU 的跡象**（008 必查項；若退回，splits 會隨 24 層 GDN 增加）。bench row 現在也記錄 graph。

### 3. 記憶體量測的交叉核對：breakdown 得到驗證；Mac 的外部量測同時記 footprint 與 max RSS

| 4B-Q8_0 | 2048 | 8192 | 32768 | 2K→32K 增量 |
|---|---|---|---|---|
| breakdown device self（MiB） | 4875 | 5065 | 5865 | **+990** |
| `time -l` peak footprint（MiB） | 355 | 559 | 1381 | **+1026** |
| `time -l` max RSS（MiB） | 4582 | 4786 | 5608 | +1026 |
| 理論 KV 增量（32,768 B × 30,720） | | | | **+960** |

- footprint **不含** mmap 的權重（Metal no-copy buffer），但含 KV 與 compute。
  三種量測在 context 方向的增量一致（990 / 1026 / 960 MiB），所以 breakdown 作為主要的 peak memory 指標得到驗證。
- 決策：Mac 的外部量測同時報告 footprint 與 max RSS，兩者都已由 harness 記錄（schema v2）。報告時註明 footprint 不含權重。

### 4. noise：sentinel cell 的 process-level CV 5.9%，兩格相對 MDE 約 9.6%

- poll 0 的 3 個 process 是 60.6 / 54.8 / 54.9 tok/s。session 內有明顯的時間趨勢：
  開頭 59.4，中段約 53–55，結尾 57.7。
- 同時段 loadavg 為 6–9（12 核），來自使用者的前景程式（WindowServer、ChatGPT / Codex、Chrome 影片解碼）。
  `pmset` 沒有 thermal / performance warning。
- MDE 以 §2 的「超出 noise」規則換算：兩格相差約 9.6% 才算超出 noise（n = 3）。
  H-A3 的相鄰差距可能落在 noise 內，屬於 PREREGISTRATION 已寫明的「不推翻、也不確認」。不改規則。
- 正式 run 時前景負載越低，noise 越小；這是使用者的使用情境，列為建議，不是規則。

### 5. 電源：新增 session 有效性規則

- 電源是 70 W USB-C adapter。pilot 28 分鐘內，接著 AC 的電池仍從 80% 降到 76%，代表滿載時耗電超過供電。
- 決策（Stage A 規則）：任何 process 前後的 env 紀錄出現下列任一情況，該 session 標為 **env-invalid**，與 drift 相同處理：整個 session 重跑。
  - 電源不是 AC。
  - 電量 < 20%。
  - 記錄到 thermal / performance warning。
- harness 在 `session.json` 寫入 `env_flags` 與 `env_valid`；Core A 與 quality 都適用。
  RTX 桌機沒有電池，不檢查。

### 6. 其他

- sentinel drift 為 2.96%，低於 3%。維持 3% 規則；一個 pilot session 不足以作為修改門檻的依據。
- 正式 Mac Core A session 的時間上限估計為 **10.65 小時**，依 4B 的最慢 job 換算，9B pilot 後更新。

## 放棄的替代方案

- **改用 `-v` 讓 fit run 輸出全部 log**：會連 debug 也印出，log 體積大增；`-lv 4` 剛好夠。
- **以 footprint 取代 breakdown**：footprint 不含權重，不能代表 Metal working set 的使用量。
- **因為 noise 而增加重複次數**：要等 9B pilot 與實際的正式負載情況再評估；目前維持 n = 3。

## 對實驗的影響

- PREREGISTRATION §3（Mac 外部量測）、§4（fit 指令、Mac 的 P、電源有效性規則）、§5 checklist 依上更新。
- `-fa auto` probe 待重跑。9B pilot 會補上 9B 的 memory 與時間校準。

## 補充：quality pilot（4B，`results/quality/pilot-20261008-4b`，26 / 26 ok）

遮蔽版報告（`quality/quality_pilot.py`）只看下列各項，沒有任何 accuracy、Q6_K / Q4_K_M 的 KLD 或 PPL。

- **determinism**：Q8_0 的 KLD 與 MMLU 重跑，metrics 完全相同。
- **sanity**：
  - BF16 對自己的 KLD 為 −0.0，即 0。
  - Q8_0 的 KLD 在 `-fa on` 為 0.001398、`-fa off` 為 0.001399，符合 005 預期的 1e-3 量級，`-fa` 對數值幾乎沒有影響。
- **SE 換算到 580 chunks**：Q8_0 約 1.3e-4、Q6_K 約 4.2e-4、Q4_K_M 約 7.7e-4。只看 SE，不看平均。
- **downstream 的 MDE（只看 b + c）**：相鄰配對換算到正式 N 的 95% CI 半寬約 0.2–1.7 pp（ARC 最寬，因為 N = 869）。
- **時間**：4B 換算到正式 run 約 5.1 小時；9B pilot 後更新總時數（005 原估 19.5 小時）。
- H-A5 的配對 ΔNLL 需要兩個 size，等 9B pilot。
- 未發現需要修改 quality protocol 的問題。

## 補充：Core A 9B pilot（`results/core_a/macos-metal/pilot-20261008-9b`，22 / 22 ok，2026-10-08 23:28–00:06）

遮蔽版報告，與 4B 相同，只看下列各項。session 有效：env_valid，sentinel drift 0.53%。

- **`-fa auto` probe**：9B 的三個 quant 都是 "Flash Attention enabled"，graph splits = 2。所有 process 的 splits 都是 2，fused GDN 沒有退回 CPU。
- **`--poll`**：在同一個 sentinel cell 再比一次，poll 0 為 56.30 tok/s，poll 50 為 56.40 tok/s，poll 0 只慢 0.18%。P = 0 不變。
- **noise**：sentinel 的 process-level CV 為 **1.03%**，兩格相對 MDE 約 **1.7%**。
  4B pilot 是 5.9% / 9.6%，那時前景負載的 loadavg 為 6–9。這次在深夜執行，前景負載很低，與 012 第 4 點的判斷一致。
- **記憶體**：breakdown 的 device self 從 2K 到 32K，三個 quant 都增加 **+958 MiB**，理論 KV 增量為 960 MiB。
  footprint 增加約 1,025 MiB，與 4B 的結果一致。
- **時間校準**：正式 Core A 的上限估計改為 **17.39 小時**（由 9B 各類 job 最慢的 wall time 換算，取代 4B 換算的 10.65 小時）。
  不含 016 的充電暫停。
- **電源**：接著 70 W 的 AC，電量在 38 分鐘內從 100% 降到 73%。這是 decision 016（充電閘門）的依據。
- **smoke**（9B-Q4_K_M，`-c 2048`，greedy）：輸出 "The capital of France is Paris."，正常。

## 補充：quality pilot（4B + 9B，`results/quality/pilot-20261008-both`，52 / 52 ok，2026-10-09 00:06–01:05）

遮蔽版報告，只看 SD / SE / b + c / 時間，不看任何平均差或 accuracy。session 有效：env_valid。

- **determinism**：4B 與 9B 的 Q8_0 KLD、MMLU 重跑，結果完全相同。
- **sanity**：
  - 兩個 size 的 self-KLD 都是 0。
  - 9B Q8_0 的 KLD：`-fa on` 為 0.000471，`-fa off` 為 0.000471，相同。
  - 4B 同 012 前述，為 0.0014。
- **KLD SE**（換算到 580 chunks，只列每個 size 的最大值；013）：4B 為 7.7e-4，9B 為 6.0e-4。
- **H-A5(a) 的 power**：配對 ΔNLL 的 chunk 差值 SD 為 0.064（pilot 20 chunks）。
  - 換算到正式的 580 chunks，95% CI 半寬約 0.0052 nats/token。
  - 在 Holm 最嚴的 α = 0.01、80% power 下，MDE 約 **0.009 nats/token**。
- **downstream**：相鄰配對換算到正式 N，95% CI 半寬為 0–1.7 pp，與 4B 的結論相同。
- **時間**：正式 quality（兩個 size）約 **14.2 小時**，取代 005 原估的 19.5 小時。
  不含 016 的充電暫停，也不含 KLD 檔案的磁碟處理。
- 未發現需要修改 quality protocol 的問題。
