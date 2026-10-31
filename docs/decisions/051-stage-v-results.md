# 051 — Stage V 的結果（formal-20261031-v-adm、formal-20261031-v-cls）

- 狀態：**結果**。依 044 凍結的規則執行（`freeze.stage_v = true`，凍結 commit `c7ad0b4`，2026-10-31 06:27:26Z），凍結之後沒有改過任何規則、門檻、view、時間窗或讀法。執行依凍結的 harness，沒有偏離；凍結前就存在的一處文字和實作不一致（`aborted` 鍵）見核對事實 16，判定不受影響。兩個正式 session 都有效，都是第一次、也是唯一一次執行，沒有重跑。
- 日期：2026-10-31
- 對應：044（範圍、共通的定義、V-GW、V1、V2、V4、有效性、停止規則與時數上限、Sessions 與時數、產品對應、揭露、風險與限制、實作補記一至十三）、043（§4、§6、7.3、7.4）、042（D-A1b、D-A2、D-C、D5）、037 §2、PREREGISTRATION §12。
- 輸出：
  - V-ADM：session `results/stage_v/formal-20261031-v-adm/`（commit `db05c73`），`analysis.json`（`c37ff96`，sha256 `cd2aa428…`）。
  - V-CLS：session `results/stage_v/formal-20261031-v-cls/`（`f8e75b1`），`analysis.json`（`5b43df8`，sha256 `0a76ae7f…`）。
  - 兩份 analysis.json 都由 `benchmarks/stage_v_report.py` 在 Mac 上產生，重跑逐位元相同；raw/ 留在 WSL，只 commit `raw_manifest.json`。
- **結論（只寫 044 允許的部分）**：
  - **成立的單位**：PACE-8K（V-ADM）；VG-32K、VG-M8K-T、V1-2K、V1-8K（V-CLS）。
  - **不成立**：PACE-2K（P-U2K PASS，但被 var 那一對否決）；W1（W-90、W-50 都 FAIL）。
  - 12 個事先登記的 hypothesis 中 11 個如預測；**H-GW4 沒有如預測**（預測 MARGINAL 或 FAIL，實測 PASS）。不是 hypothesis 的對照組 F-U2K-var 也沒有如預測（預測 FAIL，實測 PASS），否決正是由它觸發。
  - 單位成立只決定產生器寫什麼，**正式釘選不會改變**（037 §2）：是否改釘選、是否開啟 PACE-8K 由使用者決定（見「使用者決定」）。
- 本文的 JSON 路徑：`A.` 是該 session 的 analysis.json；`session.`、`plan.`、`rows[i].`、`derived/` 是 session 目錄中的檔。

## 名詞（044）

- **view**：client 是 bench 量到的時間（經過 gateway）；upstream 是 gateway 送出到 vLLM 傳回 first token 的時間（`ttft := upstream_ttft`，`latency := upstream_ttft + (latency − ttft)`）。
- **判定 view**：V2 取 upstream 與 client 兩個類別中較差者；V4 用 upstream；V-GW 與 V1 用 client。
- **R**：Y / S。Y 是 W 中的 P95（TTFT 依 tier：2K 對 2,000 ms、8K 對 6,000 ms、32K 對 20,000 ms；TPOT 只算 out ≥ 2，對 100 ms），S 是 SLO S2 的門檻。
- **類別**（030 8.1，guard band 0.05）：FAIL > TRIVIAL > NT > MARGINAL > PASS，每個點取最差的 run。PASS 要求每個有效 run 的 R ≤ 0.95。
- **時間窗**：W = [C, N − C) 是判定窗；W′ = [2C, N − C) 與 full 只作描述（044 共通的定義、實作補記三第 12 項）；WINDOW_SENSITIVE 是任一判定 view 的 W′ 類別和 W 不同。
- **跳過規則**（實作補記六第 6 項，逐點）：一個點有一次 run 不是 ok 之後，只有該點不再測；同一單位的其他點照常測。
- **E 與 N1**（V2）：E 是 manipulation check（第一波中等 token 的 request 數 ≥ 門檻，否則 run 無效）；N1 是 vLLM 內平均 request 數 ≥ 0.9 C（否則 trivial）。**U_W**（V4）：T_W 中 vLLM 內的平均單位數，要求 ≥ 28.8。
- **單位**：產品對應的單位，只在它列出的每一點都 PASS（且沒有被否決）時成立：PACE-2K、PACE-8K、W1、VG-32K、VG-M8K-T、V1-2K、V1-8K。
- **ABOVE_CKV**：C 大於 C_KV（2K 為 30 時以全長 2,432 token 計，每個 request 8 個 block），vLLM 會 preempt；043 / 042 D-A2 已揭露。

## 執行

1. **凍結**：GPU 視窗 0（產品 IT-0–6，05:42–05:44Z；mini `smoke-v-20261031`，05:44:20–06:06:30Z；部署 `b792196`）通過，凍結 commit `c7ad0b4`（044 實作補記十三）。凍結後只改過 README 與 NEXT，請求路徑閉包、harness、設定與 dataset 的 pin 都沒有改。
2. **V-ADM `formal-20261031-v-adm`**（GPU lock，部署 `c7ad0b4`）：06:28:13–08:14:41Z（1.774 h，上限 3.5 h）。
3. **空檔**：V-ADM 結束到 V-CLS 開始 2 h 01 m 41.7 s（2.028260 h，≥ 2 h；只多 101.7 s）。開始前確認 GPU 為 FREE（記憶體 245 MiB、使用率 0%），在 `flock -n /tmp/ai-portfolio-gpu.lock` 下執行。
4. **V-CLS `formal-20261031-v-cls`**（GPU lock，部署 `c7ad0b4`）：10:16:22–15:34:22Z（5.299930 h，上限 7.0 h；plan 估計 5.256 h）。
5. **Mac**：每個 session 結束後以 `tools/rtx/fetch_all_sessions.sh stage_v` 取回（不含 raw/），`wsl-inventory.txt` 在 WSL 上只列出這三個目錄；以 `stage_v_plan.py --formal` 重建 plan，和 session 的 plan.json 逐位元相同；`stage_v_report.py` 產生 analysis.json 並重跑確認逐位元相同；再各跑一次獨立檢查（見文末）。
6. **時數**：全期累計 7.443647 h（mini 0.369418 h、V-ADM 1.774298 h、V-CLS 5.299930 h），12 h 總上限剩 4.556353 h。產品 IT-0–6 依 044 不算研究時數。

## 結果：V-ADM（`formal-20261031-v-adm`）

- 25 個 boot、25 個 run 全部有效（2 個 sentinel、23 個 V run）；`env_valid` true，`env_flags`、`gateway_faults`、`inherited_unlisted` 都是空的；沒有 `cap_stop`，`not_run_cap` 為空。
- replay 一致；sentinel 閘門通過（drift 0.059% / 0.025%，容許 3%）；CROSS_SESSION_SHIFT 為 false（四個值都在 ±0.39% 內，limit 5%，只作描述）。
- 跳過：P-U2K-var、W-90、W-50 的 r1 是有效的 fail，所以這三點的 r2–r5 不再測，共 12 個 boot（`A.schedule.skipped`）。其他四點各 5 個 ok 的 run。

### V2：prefill pacing（判定 view：upstream 與 client 取較差者）

| 點 | pacing | C | 類別 | 有效 / ok | client 最差 R（TTFT / TPOT） | upstream 最差 R（TTFT / TPOT） | E（等待數 ≥ 門檻） | N1（≥ 0.9C） |
|---|---|---|---|---|---|---|---|---|
| P-U2K | 開（9,600 / 5,920） | 32 | **PASS** | 5 / 5 | 0.695 / 0.922 | 0.692 / 0.922 | 27 ≥ 26 | 31.98 |
| P-U8K | 開（32,256 / 5,800） | 8 | **PASS** | 5 / 5 | 0.531 / 0.883 | 0.527 / 0.883 | 4 ≥ 3 | 7.99 |
| P-U2K-var | 開 | 32 | **FAIL** | 1 / 0（r2–r5 跳過） | **1.019** / 0.576 | 0.887 / 0.576（view 類別 NT） | 16 ≥ 12 | 31.90 |
| F-U2K-var（對照） | 關 | 32 | **PASS** | 5 / 5 | 0.920 / 0.571 | 0.917 / 0.571 | 0（關） | 31.98 |
| F-U2K（對照） | 關 | 32 | **PASS** | 5 / 5 | 0.696 / 0.922 | 0.692 / 0.922 | 0（關） | 31.98 |

- 佇列都是「慢慢服務」（120 s / 600 s）。P-U2K-var 是 WINDOW_SENSITIVE：client 的 W 為 FAIL，W′ 為 NT（只有 1 個 run；client W′ R 0.851）。
- **讀法（044 V2 的兩張 2×2 表）**：
  - var 那一對：P-U2K-var 非 PASS、F-U2K-var PASS → 「pacing 在 var 上有害，**否決 PACE-2K**」（`A.readings.var.reading = "veto"`）。
  - uniform 那一對：P-U2K PASS、F-U2K PASS → 「新版 gateway 下正式 class 仍成立；PACE-2K 依上面的規則」（`formal_class_holds`）。
  - 所以 PACE-2K 不上線的原因是否決，不是 P-U2K 失守。

### V4：`weighted-v1` 多 class 准入（判定 view：upstream）

| 點 | 類別 | 有效 / ok | upstream R：2K TTFT / TPOT / 8K long（合併） | U_W（≥ 28.8） | client R_short（只作描述） |
|---|---|---|---|---|---|
| W-90 | **FAIL** | 1 / 0（r2–r5 跳過） | **1.384** / 0.952 / 0.473 | 31.59 | 4.67 |
| W-50 | **FAIL** | 1 / 0（r2–r5 跳過） | **1.667** / 0.884 / 0.571 | 30.90 | 13.99 |

- 兩點都不是 TRIVIAL：U_W 都 ≥ 28.8，政策填滿了預算，失敗的是 2K 的 TTFT。依判定表，W1 不上線；不得事後改成限制 8K 比例、2K 優先或較小的預算。
- client view 依構造排隊（queue_wait 中位數 3.73 s / 20.13 s），只作描述，不判定。

## 結果：V-CLS（`formal-20261031-v-cls`）

- 32 個 boot、32 個 run 全部有效（2 個 sentinel、30 個 V run，每點 5 個，全部 ok）；`env_valid` true，`env_flags`、`gateway_faults`、`inherited_unlisted` 都是空的；`A.schedule` 的 `skipped`、`not_run_cap` 都是空的，`cap_stop` 為 null。
- replay 一致；sentinel 閘門通過（output throughput 316.23 → 315.94 tok/s，drift 0.091%；W 平均 TTFT 965.66 → 966.38 ms，drift 0.075%；容許 3%）；CROSS_SESSION_SHIFT 為 false（+0.41% / +0.32%、+0.75% / −0.34%，limit 5%，只作描述）。
- block 順序（`plan.block_orders`，由 `capacity_plan.block_order` 洗牌）和 rows 的實際順序相同。

### V-GW 與 V1（判定 view：client；佇列為產品預設 2 s / 300 s）

| 點 | server / class | C | N | 類別 | 有效 / ok | client 最差 R（TTFT / TPOT / long 合併） | upstream 最差 R TTFT | W′ / full（只作描述） |
|---|---|---|---|---|---|---|---|---|
| GW-U32K | B0 / uniform-32k（`…-d-a1b-b0`） | 2 | 200 | **PASS** | 5 / 5 | 0.621 / **0.625** / — | 0.618 | PASS / PASS |
| GW-U2K-O512 | B0 / `v-uniform-2k-o512`（provisional） | 32 | 320 | **PASS** | 5 / 5 | **0.702** / 0.436 / — | 0.698 | PASS / FAIL |
| GW-U8K-O512 | B0 / `v-uniform-8k-o512`（provisional） | 12 | 200 | **PASS** | 5 / 5 | **0.535** / 0.465 / — | 0.532 | PASS / PASS |
| GW-M8K-T | T / mixed-8k-t（`…-d-a1b-t`） | 4 | 200 | **PASS** | 5 / 5 | **0.796** / 0.776 / 0.565 | 0.794 | PASS / PASS |
| L-U2K | B0 / `v-uniform-2k-o512`，輸出 [1, 512] | 32 | 320 | **PASS** | 5 / 5 | 0.506 / **0.697** / — | 0.502 | PASS / FAIL |
| L-U8K | B0 / `v-uniform-8k-o512-c8`，輸出 [1, 512] | 8 | 200 | **PASS** | 5 / 5 | 0.478 / **0.731** / — | 0.474 | PASS / PASS |

- 粗體是決定該點最差 R 的項。upstream view 也全部 PASS，但只並列報告，不構成另一個判定。
- 六個點的 `window_sensitive` 都是 false。GW-U2K-O512 與 L-U2K 的 full 在兩種 view 都是 FAIL（開場第一波，見描述量），只作描述，不改變類別。
- 30 個 run 的最大 max_r 是 0.7961（GW-M8K-T r4），距 0.95 有 0.154；GW-M8K-T 的 run 間範圍 0.7889–0.7961。

## Hypotheses（兩個 session；預測以點的類別記錄，實作補記六第 11 項）

| id | 點 | 類別 | 判定 | 預測的類別 | 如預測 |
|---|---|---|---|---|---|
| H-P1 | P-U2K | PASS | supported | {PASS} | 是 |
| H-P2 | P-U8K | PASS | supported | {PASS} | 是 |
| H-P3 | P-U2K-var | FAIL | not supported | {FAIL, TRIVIAL} | 是（機制不同，見核對事實 3） |
| H-W1 | W-90 | FAIL | not supported | {FAIL, TRIVIAL} | 是 |
| H-W2 | W-50 | FAIL | not supported | {FAIL, TRIVIAL} | 是 |
| H-W0 | W-90、W-50 的 U_W | 31.59、30.90 | supported | supported | 是（每個 mix n = 1） |
| H-GW1 | GW-U32K | PASS | supported | {PASS} | 是 |
| H-GW2 | GW-U2K-O512 | PASS | supported | {PASS} | 是（數值預測 R 0.82–0.90 沒有命中，實測 0.702） |
| H-GW3 | GW-U8K-O512 | PASS | supported | {PASS} | 是 |
| **H-GW4** | GW-M8K-T | PASS | supported | {MARGINAL, FAIL} | **否** |
| H-L1 | L-U2K | PASS | supported | {PASS} | 是 |
| H-L2 | L-U8K | PASS | supported | {PASS} | 是 |

- **H-GW4 沒有如預測**（`A.hypotheses.H-GW4.prediction_met = false`）。044 本文寫「not supported（MARGINAL 或 FAIL）」；實作補記三第 16 項記錄了判定字串和括號中類別的不一致，六第 11 項照本文的字面，改以類別集合 {MARGINAL, FAIL} 記錄並機械判定。判定（verdict）是 supported。
- **不是 hypothesis 的對照組也沒有如預測**：044 V2 表預測 F-U2K-var 為 FAIL，實測 5 個 run 都 PASS，否決由這一格觸發。
- 所以兩個 session 都**不是**「每個預測都命中」。commit `c37ff96` 的訊息（every prediction met）與 NEXT 第 5d 項（所有預測都命中）都只對 V-ADM 的六個 H-* 成立，以本文為準；NEXT 在本 session 結束前更正。
- **描述性預測**：
  - V2（第一波最大 client TTFT 和 D-C 的 U2K-gw / U8K-gw 相差 ≤ 10%）：成立。P-U2K 10,292–10,356 ms 對 10,366–10,420 ms（最大差 1.23%）；P-U8K 10,976–11,072 ms 對 10,997–11,104 ms（1.15%）。跨 session，只作描述。
  - V-GW（W 中 queue_wait_s 的 P95 < 50 ms）：成立。六個點 W 中的 P95 都是 0.0 ms，V-CLS 所有 request（含第一波）的最大值也是 0 ms。這是 039 H-D7 的理由第一次被量到；但在 C_PROD，client 數等於 profile 的 max_concurrency，依構造不會排隊，所以它只確認了機制，沒有量到負載下 gateway 的排隊行為。

## 單位與產品對應（044 產品對應，機械式）

| 單位 | 需要 PASS 的點 | 結果 | 產物（產生器，本文之後；`profile_gen --validation-session`） | 產品上的效果與仍不涵蓋 |
|---|---|---|---|---|
| PACE-2K | P-U2K，且沒有被 var 否決 | **不成立**（被否決） | 不寫 | — |
| PACE-8K | P-U8K | **成立**（worst_r client = upstream = 0.883，n 5） | `verified-formal-20261031-v-adm-b0.json`：c1b uniform-8k（8）的複本，加上 `pacing` {32,256, 5,800} 與佇列 {120 s, 600 s} | 只在這份 profile、相同參數與佇列設定、相同 `gateway_path_sha256` 下可以開啟，預設關閉；不涵蓋快速 429、開放迴路；不得宣稱第一波 TTFT 改善 |
| W1 | W-90 與 W-50 | **不成立** | 不寫 `admission` 段 | `[admission] policy = "weighted-v1"` 不能上線 |
| VG-32K | GW-U32K | **成立**（worst_r 0.625） | 沒有新檔；`…-d-a1b-b0` 的位元組不變（`not_covered` 仍列 `via_gateway`，保守） | uniform-32k（2）成為正式釘選的候選；不涵蓋 prompt 長度不一、開放迴路；第一波約 12.6 s（揭露值 12.8 s） |
| VG-M8K-T | GW-M8K-T | **成立**（worst_r client 0.796、upstream 0.794） | 沒有新檔 | mixed-8k-t（4）成為候選；另需 043 §6 的 T 部署 1–6 與 V5；042 的 WINDOW_SENSITIVE 揭露保留；U8K-T 永遠只作報告 |
| V1-2K | L-U2K 與 GW-U2K-O512 | **成立**（worst_r 0.697 / 0.702） | `verified-formal-20261031-v-cls-b0.json` 的 `uniform-2k-o1-512`（1,920 / 512，C 32，bound `validated`） | 輸出 ≤ 512 的 2K class 可以服務，是釘選的候選；不涵蓋 prompt 長度不一、其他輸出分布；ABOVE_CKV（32 > 30，3.125% 停頓，見核對事實 11） |
| V1-8K | L-U8K 與 GW-U8K-O512 | **成立**（worst_r 0.731 / 0.535） | 同一份 profile 的 `uniform-8k-o1-512`（8,064 / 512，C 8，`validated`） | 8K 長輸出可以服務，取代 043 的包絡規則；不涵蓋 prompt 長度不一、其他輸出分布；沒有 C_KV 的問題（L-U8K 的 KV 峰值 0.628）；C 12 本身永遠不能服務（043 §4、044 實作補記六第 2 項），GW-U8K-O512 只是「全部 512」的最壞情況證據 |
| （描述） | var 那一對 | 否決 | 只改 README | 不得宣稱 pacing 修好 var |

- **產物**（本文 commit 之後，在乾淨的 `c1c55c8` 上執行 `profile_gen --validation-session … --freeze-commit c7ad0b4 --out-dir service/profiles`，兩個 session 都 exit 0，只寫出上表的兩個檔）：
  - `service/profiles/verified-formal-20261031-v-adm-b0.json`（檔案 sha256 `68051d9f…`，`content_sha256` `02afafe9…`；base c1b `1863d7e1…`）；
  - `service/profiles/verified-formal-20261031-v-cls-b0.json`（`0ea8752e…`，`content_sha256` `99521be7…`；base d-a2-b0 `d14caf40…`）；
  - 兩份的 `generator_commit` 都是 `c1c55c8`、`freeze_commit` 與 `deployed_commit` 都是 `c7ad0b4`、`gateway_path_sha256` `679a7c66…`；`test_profile_gen` 的重新產生測試涵蓋它們的位元組，`test_profile_gen_stage_v` 另檢查兩個 session 寫出的內容與「沒有釘選」。
- 「單位成立」不等於「上線」：正式釘選 `verified-formal-20261017-c1b.json`（`rtx-verified.toml`，active class uniform-2k，32）不會因任何結果自動改變。改釘選之後要跑 IT-0–6（037 §2）。
- V1 的兩個 class 從 D-A2 的 `…-d-a2-b0` 繼承 flags 與 `not_covered`（`uniform-2k-o1-512` 的 flags 為 [WINDOW_SENSITIVE, ABOVE_CKV]，含 `start_up_wave`、`in_vllm_wait_beyond_c_kv`），產生器只移除 `via_gateway` 與 `output_length_other_than_measured`，加入 `output_length_distribution`（044 §12 的 class）。

## 描述量（沒有門檻，不決定任何判定）

### V-ADM

- **pacing 只在第一波約束（uniform）**：P-U2K 每個 run 第一波有 27 個 request 等 token（每個約 325 ms = 1,920 / 5,920 s），P-U8K 有 4 個（約 1,391 ms）；idx ≥ C 的 request 在 10 個 run 中都沒有任何 prefill_wait 或 queue_wait。W 中的 token 到達率 P-U2K 約 4,939 tok/s、P-U8K 約 5,081 tok/s（rate 的 83% / 88%），和 H-P1 的理由一致；H-P2 預期穩態偶爾被限制，實測 W 中一次都沒有。所以在 W 中，P-U2K 實際上和 F-U2K 是同一件事的重複量測；PACE-8K 的 PASS 只表示 pacing 開啟時穩態仍符合 S2（W 中 pacing 沒有約束任何 request），不是改善。V-ADM 沒有 8K 不開 pacing 的對照，不能和關閉相比。
- **第一波沒有改善**：等待由 vLLM 內移到 gateway。P-U2K 的 upstream 第一波最大值 1.86–1.91 s，client 仍為 10.29–10.36 s，和 F-U2K 的 10.32–10.41 s 相同；P-U8K 的 upstream 5.55–5.64 s，client 10.98–11.07 s。
- **P-U2K-var r1 的 client view 失敗**（只描述這一個 run，不是因果）：
  - W 中 client TTFT 超過 2 s 的 18 個 request 全部在第二波（idx 42–62），W′ 中沒有；
  - 它們在 gateway 各等了 29–476 ms（queue_wait 加 prefill_wait，中位數 249 ms），upstream TTFT 只有 1 個超過 2 s（idx 42，2,000.9 ms）；
  - 以 seed 重建的長度（var-2048 的產生規則；dataset 不在 Mac 上，只做了 sha256 的三方比對）：第一波 27,598 token 中有 16 個 request 等 token，沒等的 16 個集中完成，封閉迴路的第二波在約 4.06 s 內送來 35,035 token（約 8,629 tok/s，高於 rate），idx 42–63 都在 bucket 幾乎為 0 時才被准入；
  - first token 大致落在約 340 ms 的格點上：幾十到幾百 ms 的送出延遲就把 first token 推到下一格（F r1 的第二波約 1,340 / 1,680 / 2,020 ms，P r1 約 2,037 / 2,380 ms），所以 client 越過 2 s，upstream 仍在 2 s 內；
  - 同一 idx 下，P r1 第二波的 upstream TTFT 平均反而比 F r1 高約 139 ms：gateway 的等待沒有換來較短的 vLLM 時間。
- **F-U2K-var 的 run 間變異接近門檻**：W 中超過 2 s 的個數為 6、6、12、4、11（幾乎都在第二波）。n = 256 時超過 ≥ 14 個 P95 必定 > 2 s，≤ 12 個不可能；r3 剛好 12 個（R 0.9197）。同一條序列在 D-C 直連為 0.944 / 0.856 / 0.989 / 1.021（跨 session，只作描述）。
- **W 點**：W-90 的 W 中 2K 有 103 / 230 個（44.8%）的 upstream TTFT 超過 2 s，W-50 為 83 / 128（64.8%），大多不在第二波。以粗略的重疊指標（8K 的 [t_sent, t_sent + upstream_ttft] 和 2K 的同一區間重疊）分組：W-90 沒有重疊的 2K 中位數 1,006 ms（35 / 134 超過 2 s），重疊 1 / 2 個 8K 的中位數 2,162 / 2,448 ms；W-50 重疊越多越高（398 → 1,984 → 2,159 → 2,818 ms）。方向和事先登記的理由（8K 的 prefill 擋住 2K）一致，但 W-90 沒有重疊的 2K 仍有 26% 超過 2 s，不是唯一的來源；只作描述，不能歸因。
- **U_W 對凍結前的模擬**（實作補記五，descriptive_only）：W-90 模擬 31.55（五個 block 平均；同一個 block 1 的 dataset 為 31.49）、實測 31.59；W-50 模擬 30.80（block 1 為 30.96）、實測 30.90。HOL 與往返少掉的單位 0.41 / 1.10（044 估約 1–1.5）。
- **secondary SLO**：S1 下七個點全部 FAIL；S3 下 P-U2K-var、W-90、W-50 為 FAIL，其他 PASS。full 視窗下，P-U2K 的 client、F-U2K-var 與 F-U2K 的兩種 view 都是 FAIL（第一波最大 client TTFT：P-U2K 與 F-U2K 約 10.3–10.4 s，F-U2K-var 約 4.6–4.7 s）。

### V-CLS

- **gateway 本身的開銷**（同一個 request 的 client − upstream TTFT，W，5 個 run 合併，P50 / P95）：GW-U32K 71.57 / 92.36 ms；GW-U2K-O512 6.97 / 8.91 ms；GW-U8K-O512 19.90 / 24.92 ms；GW-M8K-T 20.55 / 74.76 ms（P95 來自 32K 長類）；L-U2K 7.12 / 9.07 ms；L-U8K 20.11 / 25.45 ms。V-ADM 的 F-U2K、P-U2K 為 7.00 / 8.92、7.01 / 8.87 ms。開銷隨 prompt 長度增加（2K 約 7 ms、8K 約 20 ms、32K 約 72 ms），符合本機 tokenizer 的成本（推論）。
  - 這是含 bench 到 gateway 傳輸的 W 中的值；第一波同時到達時較大（2K 約 45 ms、32K 約 83 ms）。
  - 這和 044 預測依據中的 D-C 位移（2K 在 W +214 ms、W′ +390 ms，8K 約 +710 / +720 ms）**不是同一個估計量**：D-C 的位移是「經過 gateway 的 client TTFT」減「另一個 session 直連的 client TTFT」，這裡是同一個 request 的分解。用 D-C 那一種估計量（本 session 經過 gateway 的 client 合併 Y，減去 D-A 直連的合併 Y），各點方向不一：GW-U2K-O512 約 −18 ms、GW-U8K-O512 約 +658 ms（逐 run 約 +298 到 +690 ms，r4、r5 較低）、GW-U32K 約 +2.26 s、GW-M8K-T 約 −415 ms；差異落在 upstream（vLLM 端），不符合「每個 request 疊加一個固定開銷」。跨 session，只作描述，不能歸因。
- **GW-U2K-O512 的 ABOVE_CKV 停頓**：每個 run 都是 preemption 30、KV 峰值 1.0；W 中剛好 8 / 256（3.125%）個 request 停在 4.28–5.14 s，位置固定在 idx 63 + 31k（idx 63 為 4,282.0–4,299.0 ms，idx 94–280 為 4,979.2–5,140.6 ms），attainment 0.96875。停頓的 upstream TTFT 和 client 只差約 7 ms、queue_wait 為 0：停在 vLLM 內，gateway 沒有改變也沒有緩解它。結構和 042 D-A2 直連逐項相同（D-A2 在 C 32：worst_R short 0.708、preemption 30、attainment 0.96875）。waiting 峰值 28 不是 ABOVE_CKV 的訊號：同樣 C 32、沒有 preemption 的 L-U2K 也是 27–28，它反映開場時 chunked prefill 的佇列。
- **full 為 FAIL 的原因**（GW-U2K-O512、L-U2K）：第一波 32 個 request 同時送出，在 vLLM 中依序 prefill，每個約 335 ms（第一波 TTFT 由約 402 ms 起，到 10,350–10,625 ms），每個 run 有 27 / 32 個超過 2 s。full（N 320）中超過 2 s 的有 36 個（GW-U2K-O512，加上 9 個停頓）與 27–29 個（L-U2K），Y_full 短類 5,468–5,489 ms 與 5,485–5,836 ms。
- **L-U2K 的開場波漏進 W**（044 沒有預料到的機制）：第一波中輸出極短的 request 在開場的依序 prefill 還沒做完時就結束（例如 r5 的 idx 25 只有 4 個 token，約 4.12 s 結束），讓 idx 32 / 33 排在 9 到 20 個還沒 prefill 的 request 後面：r2 idx 32 為 3,163 ms（前面 9 個），r3 idx 32 為 3,507 ms（10 個），r5 idx 32 / 33 為 6,840 / 5,486 ms（20 / 15 個）；r1、r4 沒有。每個 run 最多 2 個，P95 不受影響（r5 的 W′ Y 728.8 ms 因此低於 W 的 964.9 ms）。這不是 KV 停頓（L-U2K 的 preemption 都是 0），不能和 ABOVE_CKV 混稱。
- **V1：TPOT 決定，而且比 O512 夥伴高**：L-U2K 0.669–0.697（GW-U2K-O512 0.435–0.436）、L-U8K 0.667–0.731（GW-U8K-O512 在 C 12 為 0.464–0.465）。輸出較短、完成得較快，W 中每秒到達的 request 較多（L-U2K 2.02–2.15 對 1.54；L-U8K 0.534–0.546 對 0.482），每個輸出 token 對應的 prefill token 約為兩倍（7.55 對 3.75；30.4 對 15.75），所以每個 decode step 有較多 prefill 插隊（042 D-A2 的機制，推論）。
- **V1 的輸出長度分箱**（044 V1 的描述，`A.points.L-*.descriptive.out_bins`，client，W，5 個 run 合併）：

  | 點 | 箱 | n | TTFT P95（ms） | TPOT P95（ms） | 超過 TTFT 門檻 | 超過 100 ms |
  |---|---|---|---|---|---|---|
  | L-U2K | 1–16 | 46 | 749.4 | 303.07 | 0 | 9.52% |
  | L-U2K | 17–64 | 122 | 747.1 | 81.87 | 0 | 1.64% |
  | L-U2K | 65–128 | 173 | 744.4 | 75.56 | 0 | 0 |
  | L-U2K | 129–256 | 308 | 973.4 | 67.10 | 0.32% | 0 |
  | L-U2K | 257–512 | 631 | 1,011.6 | 62.80 | 0.48% | 0 |
  | L-U8K | 1–16 | 22 | 1,719.2 | 15.18 | 0 | 4.76% |
  | L-U8K | 17–64 | 81 | 2,821.5 | 110.94 | 0 | 6.17% |
  | L-U8K | 65–128 | 109 | 2,847.6 | 81.38 | 0 | 2.75% |
  | L-U8K | 129–256 | 231 | 2,833.1 | 72.03 | 0 | 0 |
  | L-U8K | 257–512 | 477 | 2,840.4 | 63.03 | 0 | 0 |

  - L-U2K 的 1–16 箱正是 H-L1 寫的風險（極短輸出的 TPOT 只平均少數幾個 step），但只占 W 的 1,280 個中的 46 個（3.6%），不決定 P95。L-U8K 的 17–64 箱 TPOT P95 也超過 100 ms（110.9 ms，81 個中 6.17% 超過），一樣不決定合併的 P95。單位成立只看 W 中合併的 P95，不代表每一種輸出長度的 request 本身都符合 TPOT。
  - H-L2 的風險（out ≤ 20、和 prefill 重疊、TPOT > 100 ms，估計 2–3%）：實測每個 run 0、0、1、1、0 個，占 W 中 out ≥ 2 的 0–0.54%，合併 2 / 919 = 0.22%（由 derived 另算，044 沒有要求這個量）。W 中 out ≤ 20 的 request 占 1.6–6.5%（合併 3.15%，依據寫約 4%）。
- **H-L1 的依據低估了 KV**：依據寫「每個 request 4–5 個 block，32 個約 148 ≤ 247」，但依 C_KV 的公式（⌈L / 528⌉ + 3），實際需求接近 247，KV 峰值 0.976–1.0。preemption 為 0 的預測命中，餘裕遠比依據所寫的薄。
- **GW-M8K-T 為什麼沒有如預測**（只作描述，不能歸因）：
  - 預測的參考是 042 D-A1b 的 M8K-T C 4（W、直連 client view），最差 run 的 R_short 0.8856（Y 5,313.39 ms；044 寫 5,316 ms，是 0.886 × 6,000 的捨入），再疊加 D-C 在 U8K（uniform、B0、C 8、輸出 128）的 +350–720 ms，得 R 0.94–1.006。
  - 依跨 session 差（合併 Y），這一點約 −415 ms，預測時疊加的 +350–720 ms 沒有出現；upstream 的短類 W P95（4,715.7–4,761.3 ms）也比 D-A1b 直連的 5,051.6–5,313.4 ms 低約 290–600 ms。同一個 request 的 client − upstream（P50 20.55 ms、P95 74.76 ms）是另一個估計量，只作描述。
  - 比較的條件大致相同：server argv sha256 都是 `56fcc296…`，dataset 都是 `ab548bed…`，C 4、N 200、輸出 128、W 窗相同，W 中都是 172 個短類加 20 個長類。已知的不同至少有部署 commit（`0ec7b92` 對 `c7ad0b4`）、日期（10-23 對 10-31）、session 中的位置與前一個 boot，以及有沒有 gateway；GPU 溫度、時脈與 Windows 端的狀態沒有可比較的紀錄。
  - 分布上的差別在短類尾端：直連每個 run 有 10–12 個短類超過 5 s，經過 gateway 只有 6 個；直連那一點本身在 5 個 run 之間就差 262 ms，所以單一 session 的參考值離 guard band 的距離本來就不穩定。
- **GW-U32K**：由 TPOT 決定（0.623–0.625，D-A1b 直連為 0.642）。C 2 封閉迴路下 TTFT 是雙峰：約一半在 6.0–7.0 s（另一條在 decode），另一半在 10–13 s（和另一條的 32K prefill 重疊）。第一波最大 client TTFT 12,558–12,594 ms，沒有超過產品對應揭露的 12.8 s。和 D-A1b 直連相比，短類 P95 從 10,141 ms 變成 12,397 ms（合併；+2.26 s，落在 upstream），TPOT 的 P50 從 52.9–54.1 ms 降到 46.7–46.9 ms，和 042 D-C 描述的「時間從 decode 移到 TTFT」一致（跨 session，推論）。
- **GW-U8K-O512 的 r4、r5 較低**（max_r 0.510、0.474，其他 0.534–0.535）：W 中約 3.2 s 的 TTFT 群聚每群 10 個（idx 26–35、74–83、170–179）：r1 有兩群（74–83、170–179），r2 三群都有，r3 有兩群（26–35、74–83），r4 只有 74–83，r5 一群都沒有。W 有 176 個，P95 落在第 167 與 168 名之間，所以群數少於 2 時 P95 才會下降。GW 點是固定長度的 random dataset，不是 dataset 的差異；W 的平均 TPOT（44.10–44.22 ms）與 bench 時長（423.5–423.6 s）在 5 個 run 都相同；env 只有 boot 前後的閒置快照（boot 前 5 個 run 都是 44 °C、180 MHz；boot 後 48–49 °C、262–300 MHz，和 r4、r5 較低沒有對應關係）。r4、r5 在 block 中是第 3、第 2 個，前一個 boot 都是 L-U2K，n 只有 2，三者分不開，原因未歸因。對類別沒有影響。
- **secondary SLO**：S1 下六個點在兩種 view 都是 FAIL；S3 下六個點都是 PASS。

## 核對時發現、必須一起讀的事實（都不改變任何判定）

1. **V-ADM 的對照組預測沒有命中，否決由它觸發**：044 對 F-U2K-var 的理由是「D-C 直連 1.021，經過 gateway 再上移約一格」；本 session 經過 gateway 為 0.843–0.920，沒有上移。
2. **PACE-2K 的否決很脆弱，但依規則成立**：
   - P-U2K-var 只有 1 個 run（逐點跳過），FAIL 只在 client view，超過門檻約 37.5 ms（R 1.019），而且 WINDOW_SENSITIVE；
   - F-U2K-var 的 r3 在判定邊界上（再多 2 個 request 超過 2 s 就會 FAIL）。只要 F-U2K-var 有任一 run 的 R > 0.95，它就不是 PASS，否決規則不適用，PACE-2K 就不會被否決：R 在 (0.95, 1] 時為 MARGINAL，讀法是「無法判讀」；R > 1 時為 FAIL，讀法是「FAIL / FAIL：pacing 沒有修好 var」（044 原本的預測）；
   - 044 的規則是機械式的，看到數據之後不改 view、時間窗或門檻，也不重跑有效的 session 來換結果。這些是揭露，不是改判的理由。
3. **H-P3 如預測，但機制和事先寫的理由不同**：理由是「pacing 不改變 vLLM 內 prefill 的物理」，預期 vLLM 端失敗；實測 upstream 為 0.887（會通過），超出的是第二波在 gateway 的等待。pacing 在 var 的第二波會約束（idx ≥ C 有 22 個 request 等 token），在 uniform 點則完全不約束。044 的 H-P1 / H-P2 依據預期穩態低於 rate（8K「偶爾被限制」），沒有預期 var 第二波的集中到達會讓 pacing 約束。
4. **逐點跳過與 n = 1**：依 044 實作補記六第 6 項（取代本文原本的「單位已死」）。在舊規則下，W-90 r1 FAIL 之後 W-50 不會跑，H-W2 與 H-W0 會是 NT。P-U2K-var、W-90、W-50、H-W0 都只建立在 1 個 run 上；W 點的「合併」R_long 也只來自單一 run。
5. **接近 guard band 的 PASS**（門檻 0.95）：P-U2K 與 F-U2K 的 TPOT 0.922（044 已預告餘裕約 0.027）、F-U2K-var 的 TTFT 0.920。E 的餘裕 P-U2K 27 / 26、P-U8K 4 / 3，是 burst 個數在構造上決定的。
6. **W-90 的 TPOT 也超過 guard band**：upstream TPOT R 0.952 > 0.95，即使 2K TTFT 通過也至多 MARGINAL；044 只預期 2K TTFT 失敗。
7. **「FIFO」在毫秒尺度上不等於到達順序**：t_arrival 在讀 body 與 tokenize 之前記錄，入列在其後。P-U2K-var r1 中 idx 55 / 56 / 57 在 2 ms 內相繼到達，准入順序是 57、56、55。不違反規則。
8. **V-ADM 的時數與時刻**：實際 1.774 h，044 依預測估約 1.60 h；差異來自 F-U2K-var 沒有在 r1 FAIL，跑滿 5 次（約 0.19 h）。V-ADM 在 mini 結束約 22 分鐘後開始；2 h 空檔只約束正式 session 之間（實作補記四第 9 項），`previous_formal` 為 null。（NEXT 舊版與 commit `a094f5c` 的訊息寫「約 06:35Z 起」，實際為 06:28:13Z；NEXT 寫 V-CLS「10:16Z 後」，最早可開始的時刻其實是 10:14:41Z，10:16Z 是背景鏈保守設定的值。）
9. **Mac 上無法驗證的部分**（兩個 session）：
   - raw/ 與 gateway log 留在 WSL，所以 (h) 的完整雙射（沒有 id 的 completion 剛好 C 筆、/tokenize 例外、status 200）、(i) 的 4xx / 5xx、/readyz 的 tuple、gateway 的設定與 argv、bench 的 input_lens，只能信任 runner 寫進 rows 的 `access_log {problems [], joined = N}`、`valid`、`reasons` 與 session 的 `env_valid`（044 風險與限制）。Mac 上能從 derived 重算的（(h) 的 50 ms 容差、(j) 的在途峰值 = C、queue_wait、TTFT、TPOT）都成立。
   - dataset 都不在 Mac 上（V-CLS 的 11 個：`data/core_b/mixed-8192.jsonl`、`data/stage_v/v1-*.jsonl`；V-ADM 的 var 與 v4），sha256 只做了 session、plan、`configs/stage_v.toml` pin 的三方比對。V1 的 pin 值是唯一的內容證據：重算者由 seed 重建每列的 `out_planned`，和 derived 逐列相同。
   - 1 Hz sampler 的值（KV、preemption、waiting、running）只確認是從 rows 原樣抄出。env 快照都有 `probe_errors`（部署 bundle 沒有 .git、WSL 沒有 nvcc），和部署方式相符，`env_flags` 為空。
10. **V-CLS 也不是「每個預測都命中」**：H-GW4 的類別預測沒有命中；數值預測也有落空（H-GW2 預測 R 0.82–0.90，實測 0.702；H-GW4 預測 0.94–1.006，實測 0.796）。兩者的依據都是把 D-C 的跨 session 位移（+214 到 +720 ms）加到直連的參考值上。用同一種估計量（本 session 經過 gateway 的 client 合併 Y，減去 D-A 直連的合併 Y）看，各點方向不一：GW-U2K-O512 約 −18 ms、GW-M8K-T 約 −415 ms，沒有出現這個位移；GW-U8K-O512 約 +658 ms，和 +720 ms 相近；GW-U32K 約 +2.26 s，大於 H-GW1 依據的 0.35–0.72 s。同一個 request 的分解（2K 約 7 ms、8K 約 20 ms）是另一個估計量，不能用來檢驗這個前提。都是跨 session，只作描述。
11. **ABOVE_CKV 的 PASS 是「個數斷崖」型**：GW-U2K-O512 的 W P95 由停頓的「個數」決定，不是連續的。以 derived 加上 `core_b_analysis.percentile` 模擬（W = idx [32, 288)，把 W 中最高的非停頓 request 換成該 run 最小的停頓值）：停頓到 12 個時 P95 為 1,389.4–1,404.9 ms（PASS；若新增的停頓取代的是一般的 request，為 1,714.8–1,730.9 ms，仍 PASS）；13 個時為 2,113.2–2,122.9 ms（R 1.057–1.061，FAIL；另一種假設下為 2,361.7–2,368.9 ms）；14 個時為 4,282.0–4,299.0 ms。每個 run 實際 8 個，所以兩種假設下都是每個 run 最多再多 4 個停頓仍 PASS，第 13 個停頓才 FAIL。R 0.70 對 0.95 的差距看起來比實際安全。V 只量了 C 32，「32 → 33 是斷崖」沒有再次確認。
12. **W′ 與 full 只作描述**：出處是 044「共通的定義」的「只作描述」與實作補記三第 12 項（實作補記六、十也各有第 12 項，不是那兩項）。V-CLS 的 `window_sensitive` 都是 false，只代表 W′ 與 W 一致，不代表開場湧入成立，也不撤銷 042 / 043 的 WINDOW_SENSITIVE 旗標（包括 VG-M8K-T 的揭露與 `uniform-2k-o1-512` 繼承的旗標）。
13. **sentinel 看不到的漂移**：sentinel 是 B0 直連的 U2K、C 25，不經過 gateway，也不涵蓋 T server。GW-M8K-T 和 D-A1b 直連的 R_short 差 −0.05 到 −0.09，比 session 內 run 間範圍（0.007）大一個量級，session 內無法排除 T server 端的漂移；044 也預告約 1.5% 的跨 session 漂移就可能翻轉接近門檻的點。
14. **時程的餘裕**：V-CLS 和 V-ADM 的間隔只比 2 h 多 101.7 s（規則成立，沒有緩衝）。12 h 總上限剩 4.556353 h：再開一次 V-CLS（上限 7.0 h）會超過，依 044「停止規則與時數上限」必須先問使用者；本研究不打算再開任何 session。
15. **plan 的兩個雜湊**：`session.plan_sha256`（`e8d81160…`，V-CLS）是 `run_core_a.sha256_json(plan)` 的正規 JSON 雜湊，不是 plan.json 檔案位元組的 sha256（`21b8ccea…`）。commit `f8e75b1` 的訊息寫的是檔案位元組的值，兩者都正確。
16. **`aborted` 的寫法和實作補記不一致**（CLAUDE.md：先記錄差異）：實作補記六第 8 項寫「Python 看得到的每一種結束都寫出 `finished_at`、`in_progress: false` 與 `aborted`」，但 `benchmarks/run_stage_v.py:803` 只在例外時寫入 `session["aborted"]`。三個 session 的 session.json 都沒有 `aborted` 鍵，report 以 `s.get("aborted")` 為假判定沒有中止，判定不受影響。harness 是凍結的（`harness_sha256` `a1e4a564…`），不改程式；044 加註指向本文。
17. **GW 點的 n = 5 主要量到重現性**：GW 點是固定序列加上幾乎決定性的封閉迴路，V1 也只涵蓋 5 條抽樣的序列（044 風險與限制）。偵測力方面，單一 run 機率 q 的壞模式要 q ≥ 0.45 才有 ≥ 95% 的機率被抓到（030 8.1）。
18. **provisional 的 class**：GW-U2K-O512、GW-U8K-O512、L-U2K、L-U8K 服務的是 `service/profiles/provisional-stage-v.json`（以 `--allow-provisional` 載入）中的 `v-` 複本。`v-uniform-2k-o512`、`v-uniform-8k-o512` 的 limits 和 D-A2 的 class 逐欄相同，只有標籤與 verified 不同；L-U8K 的 `v-uniform-8k-o512-c8` 是 uniform-8k-o512 的 token 上限（8,064 / 512）加上 c1b uniform-8k 的 8 / 8，不是 D-A2 class 的逐欄複本（044 揭露、設計審查改正的項目「L-U8K 的上限」）。
19. **V1-8K 的 C 8 由 C 12 涵蓋**：「全部輸出 512」只在 C 12 量（GW-U8K-O512），依 030 的單調假設涵蓋 C 8；本 session 沒有在 C 8 量「全部 512」。

## 對 Stage V 研究問題的意義（只寫 044 允許的部分）

- **V-GW**（Stage D 的 class 經過 gateway，在 C_PROD 是否仍成立）：在 C_PROD、封閉迴路、`/v1/completions` 加 `ignore_eos`、這一版凍結的 gateway 請求路徑（`gateway_path_sha256` `679a7c66…`）下，四個點的 client 端 S2 都成立。uniform-32k（2）與 mixed-8k-t（4）因此成為正式釘選的候選；兩個 O512 點計入各自的 V1 單位。
- **V1**（輸出長度在 [1, 512] 不一）：在 5 條 pin 住的均勻序列、經過這一版 gateway、C 32 / 8 下，2K 與 8K 的 S2 都成立；「全部 512」由 GW-O512 涵蓋。V1-8K 取代 043 的包絡規則。
- **V2**（prefill pacing）：PACE-8K 成立：pacing 開啟時穩態的 S2 在兩種 view 下成立，不是第一波的改善（2K 的第一波 client TTFT 和同 session 的 F-U2K 相同；8K 和 D-C 相差 ≤ 10%，跨 session，只作描述）。PACE-2K 被 var 那一對否決（否決很脆弱，見核對事實 2）。uniform 那一對的讀法是「新版 gateway 下正式 class 仍成立」。不得宣稱 pacing 修好 var。
- **V4**（`weighted-v1`）：兩個 mix 都 FAIL（2K 的 TTFT），政策填滿了預算（U_W ≥ 28.8），不是 trivial。W1 不上線；看到數據之後不改成其他參數。

## 限制

- 一台機器（RTX 5070 Ti，driver 617.42）、一個模型（Qwen3.5-9B W4A16）、vLLM 0.30.0、一版凍結的 gateway；封閉迴路；只量 `/v1/completions` 加 `ignore_eos`；chat route、自然 EOS、thinking 都沒有量。
- 研究用的 gateway 是 `capserve serve`（port 8080，研究的 vLLM 在 8000），不是 `deploy up`；deploy 的監督與冷啟動不在量測範圍內（044 揭露）。改釘選後的 IT-0–6 是產品測試，不補這項量測。
- 依 044 揭露：設計者在寫預測之前看過 Core、Advanced、Stage C、Stage D 的全部結果，預測用到 042 的 R、D-C 的 gateway 位移（2K 在 W +214 ms、W′ +390 ms，8K 約 +720 / +710 ms）與 D5 的斜率；pacing 的參數是 D5 的事後內插，依取整規則凍結；weighted-v1 的參數照 043；沒有任何參數依 V 的數據選擇。
- V-GW 只在 C_PROD，輸出只有 128 或 512；V1 只有 5 條 [1, 512] 均勻序列；V2 只量「慢慢服務」的佇列設定（120 s / 600 s）與 D-C 的一條 var 序列，沒有快速 429；V4 只有 0.9 與 0.5 兩個 mix，pacing 關閉，以准入（upstream）判定，不是從到達起算。
- 都沒有涵蓋：class 內 prompt 長度不一、開放迴路、32K 或 T 在政策下、pacing 與 weighted-v1 同時開啟（`A.not_supported`）。
- V3（經過 gateway 的開放迴路）依使用者的決定延後。
- 所有和 042（D-A1b、D-A2、D-C）的比較都跨 session、跨部署 commit（D-C 還跨 gateway 版本，`capserve_sha256` `53488c6d…` 對 `1113302c…`），只作描述。
- 多個結論建立在單一 run 上（P-U2K-var、W-90、W-50、H-W0），這是逐點跳過規則的結果。

## 能支持與不能支持

**能支持**
- 在上述條件下：uniform-32k @ 2 與 mixed-8k-t @ 4（T）經過 gateway 的 client 端 S2 成立；2K（C 32）與 8K（C 8）在 [1, 512] 均勻輸出（5 條序列）下成立，「全部 512」由 GW-O512 涵蓋。
- PACE-8K：pacing {32,256, 5,800} 加上「慢慢服務」的佇列（120 s / 600 s）時，uniform-8k @ 8 穩態（W）的 S2 在 upstream 與 client 兩種 view 下都成立（n 5），vLLM 中的並發 ≤ 8。V-ADM 沒有 8K 不開 pacing 的對照，不能和關閉相比。
- 新版 gateway（`679a7c66…`）、「慢慢服務」的佇列下，正式釘選的 uniform-2k（32）仍符合 S2（F-U2K，n 5；uniform 那一對的讀法為 `formal_class_holds`）。
- 依 044 的機械讀法，PACE-2K 被 var 那一對否決（P-U2K-var 只有 1 個 run，FAIL 只在 client view；F-U2K-var PASS，n 5），只限 D-C 的這一條 var 序列。
- `weighted-v1`（043 的參數）在 0.9 與 0.5 兩個 mix 下（各 1 個 run，逐點跳過），2K 的 TTFT 不符合 S2。
- （只作描述）在 C_PROD（client 數等於 max_concurrency，依構造不會排隊）時，W 中的 queue_wait 都是 0；同一個 request 的 client − upstream TTFT（含 bench 到 gateway 的傳輸與 gateway 內部的處理）在 W 中的 P50 約 7 ms（2K）、20 ms（8K）、72 ms（32K），第一波同時到達時較大（2K 約 45 ms、32K 約 83 ms）。

**不能支持**
- 「經過 gateway 會讓 TTFT 變快 / 改善 SLO」（GW-M8K-T 的跨 session 差只能描述）；「H-GW4 沒有命中是因為 gateway 降低了延遲」或「D-A1b 的參考值是錯的」。
- 「D-C 的 +214 / +390 / +720 ms 被證明是（或不是）gateway 的開銷」：不是同一個估計量。
- 「gateway 的開銷只有 7 ms，所以 gateway 對 SLO 沒有影響」：upstream 在跨 session 時會移動，而且只量了 C_PROD 與封閉迴路。
- 任何高於這次 C 的容量（例如 mixed-8k-t @ 5、uniform-32k @ 3）：V-GW 只量 C_PROD，030 的單調假設只涵蓋 ≤ C（044「和 043 字面的差異」第 9 項）；8K 長輸出 @ 12 依 043 §4 與 044 實作補記六第 2 項永遠不能服務。
- 「短輸出的 request 本身符合 TPOT 100 ms」：分箱 P95 在 L-U2K 的 1–16 箱為 303 ms、L-U8K 的 17–64 箱為 110.9 ms；單位成立只看 W 中合併的 P95。
- 「已經改釘選」或「改釘選會自動發生」。
- 「V1-2K 沒有 KV 斷崖」或「ABOVE_CKV 不再適用」：GW-U2K-O512 仍有 preemption 30 與 3.125% 的 4–5 s 停頓。
- 「32 → 33 的斷崖再次確認」：V 只量了 C 32。
- 「第一波 / 開場湧入符合 SLO」或「pacing 改善第一波」。
- 其他輸出分布（例如大量 < 32 token 的輸出）、EOS 造成的自然長度、thinking、prompt 長度不一。
- 「pacing 修好 var」，或任何其他 weighted-v1 參數、mix 的結論。
- 「S1 成立」，或把 S3 的 PASS 當成判定。
- 「GW-U8K-O512 的 r4、r5 較低是因為溫度、時脈、dataset 或 block 位置」。

## 使用者決定（本文不改任何設定）

1. **是否改正式釘選**：目前 `rtx-verified.toml` 釘住 `verified-formal-20261017-c1b.json`，active class uniform-2k（32）。候選：
   - uniform-32k（2，`…-d-a1b-b0`，VG-32K）；
   - mixed-8k-t（4，`…-d-a1b-t`，VG-M8K-T；另需 043 §6 的 T 部署 1–6 與 V5）；
   - `uniform-2k-o1-512`（32，ABOVE_CKV，另繼承 WINDOW_SENSITIVE）與 `uniform-8k-o1-512`（8），由產生器在本文之後寫進 `verified-formal-20261031-v-cls-b0.json`（本文寫成時還不存在）。
   - 改了之後跑 IT-0–6（037 §2）。
2. **是否開啟 PACE-8K**：PACE-8K 只附在 `verified-formal-20261031-v-adm-b0.json`（由產生器在本文之後寫出）的 uniform-8k（8）上；gateway 的設定只接受 active class 自己驗證過的 pacing 與佇列。所以開啟它等於改正式釘選：production_profile 改成這份 profile，active class 改成 uniform-8k，佇列改成 120 s / 600 s，參數與 `gateway_path_sha256` 都要相同，改了之後跑 IT-0–6（037 §2）。目前的 uniform-2k（32）不能開啟 pacing（PACE-2K 被否決）。預設關閉。
3. V3、T 部署與 V5、常駐，維持既有的待決狀態（README Future Work）。

## 數字來源

- V-ADM：`results/stage_v/formal-20261031-v-adm/`（`db05c73`）與 `analysis.json`（`c37ff96`，sha256 `cd2aa42823b75a81af569286b7049de01845f47681c7263aac0d981aec46189a`）；rows sha256 `e429beee…`。
- V-CLS：`results/stage_v/formal-20261031-v-cls/`（`f8e75b1`）與 `analysis.json`（`5b43df8`，sha256 `0a76ae7f600583545a18da07562085fb2668611fba1ac45b7160b93d68e2cfd8`）；rows sha256 `dca2ea7b…`；plan.json 檔案 sha256 `21b8ccea…`。
- 類別、R、單位、hypothesis、out_bins、decomposition、sentinel、secondary SLO：`A.points`、`A.units`、`A.hypotheses`、`A.sentinel`、`A.secondary_slo`。
- 逐 request 的量（停頓的 idx 與 TTFT、開場波漏入、client − upstream、H-L2 的量、分組、重疊指標、個數斷崖的模擬）：由 `derived/<tag>-requests.jsonl` 另算，只作描述。
- 跨 session 的參考：`results/stage_d/formal-20261023-d-a1b/analysis.json`（`workloads.M8K-T.points.4`、`workloads.U32K.points.2`）、`results/stage_d/formal-20261023-d-a2/analysis.json`（`workloads.U2K-O512.points.32`、`workloads.U8K-O512.points.12`）、D-C 與 D5 依 042。
- 時數與空檔：三個 session 的 `session.started_at` / `finished_at`、`session.previous_formal`、`session.hours_used_h`。

## 獨立檢查（每個 session 一次 verify workflow，5 個 agent：重算、稽核、機制、反駁者、批評者）

- **重算**：只用 stdlib 自己寫的腳本（只複製 `core_b_analysis.percentile`），沒有 import 任何 Stage V 或 capacity 模組，由 plan、rows、session、derived 重算，和 analysis.json 逐葉比對：V-ADM 3,141 項、V-CLS 2,876 項，都是 0 項不符（浮點相對容差 1e-9，類別、判定、布林完全相同）。V-CLS 另做兩個擾動測試（W 上界改 N − C + 1、upstream 換成 gw_ttft_ms），分別出現 463 與 413 個不符，確認比對不是空轉。
- **稽核**：兩個 session 的 rows / derived / plan / profile / dataset pin / 設定檔的 sha256、harness `a1e4a564…`、gateway path `679a7c66…`、capserve `1113302c…`、repo_commit `c7ad0b4` 都相符；plan.json 和現在重新產生的逐位元相同；boot 順序、跳過、每個 session 的時數上限與結尾保留量、12 h 總上限、2 h 空檔、031 閒置判定（每個樣本 245 MiB、使用率 0%）、sentinel 都通過；`report()` 唯讀重跑 sha256 相同。
- **機制與反駁者**：見描述量與核對事實。V-ADM 七項主張都成立（六項附但書）；V-CLS 十項主張都成立，都附但書（已寫進核對事實 9–19）。反駁者也在 scratch 中試跑產生器：V-CLS 寫出一個檔，兩個 class 如上表，repo 的工作樹不變。
- **批評者**：四份報告的實質結論沒有矛盾。核對後的更正（都不影響結論）：
  - V-ADM：時數上限的估計量是 est_s 173 s，不是 78 s；V-CLS 的最早開始是 10:14:41Z；+214 ms 是 D-C 的值，V-ADM 的跨 session 差約 +202 ms。
  - V-CLS：GW-U2K-O512 的 transport 合併值是 P50 1.031 / P95 2.496 ms（不是 1.04 / 2.47）；「真正的餘裕是 5 個請求」應為「最多再多 4 個停頓，第 13 個才 FAIL」（核對事實 11）；spill 的判定量是 ready 相對開機前最大 80.0 MiB（容許 160）與 bench 相對 ready 最大 3.9 MiB（容許 64），83.9 MiB 不是判定量；D-A2 U8K-O512 @ 12 的 worst_R 是 short 0.4248 / TPOT 0.4736，不只 0.425；waiting 峰值 28 不是 ABOVE_CKV 的訊號（描述量）。
- **本文的審查**（四位審查者分別核對 V-CLS 數字、V-ADM 數字與執行經過、規則與產品對應、過度主張，再由批評者去重並抽查）：確認 30 項（1 項 must-fix、12 項 should-fix、17 項 minor），都已套用到本文、044 與 PREREGISTRATION 的註記，沒有一項改變判定、單位或 hypothesis。主要的更正：PACE-8K 的主張從「不損害穩態」改為「pacing 開啟時穩態的 S2 成立」（8K 沒有關閉的對照）；var 讀法在 F-U2K-var FAIL 時是「pacing 沒有修好 var」，不是「無法判讀」；跨 session 的位移改用同一種估計量（合併 Y）陳述；開啟 PACE-8K 等於改釘選；限制補上 044 的兩項揭露（`capserve serve` 不是 `deploy up`；預測與參數的來源）。
