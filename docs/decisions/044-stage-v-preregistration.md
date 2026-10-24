# 044 — Stage V 事先登記（DRAFT）：Stage D 產品化的驗證研究（V-GW、V1、V2、V4）

- 狀態：**DRAFT**（設計；待決事項已依建議定案，見文末）。harness、產品程式、一次 mini smoke 與兩輪審查都完成後凍結（見「凍結」）。凍結前不收任何 V 的正式數據。
  - 2026-10-24：產品程式與 harness 已實作並合併到 main（`f12255f`）；兩輪凍結前審查與修正後的獨立檢查都已處理（實作補記六、七、八）；產生器的 `--validation-session` 路徑已完成（`feat/044-generator`，實作補記九；合成 session，沒有 Stage V 數據）。尚待：WSL 唯讀複核、dataset pin、GPU 視窗 0（產品 IT 與 mini），之後才凍結。
- 日期：2026-10-24
- 對應：
  - 043：7.3 的候選設計、7.4 的驗證研究、「使用者決定」；
  - 030 §8：8.1、8.4、8.6、8.7、8.8，以及實作補記二；
  - 031；
  - 039：D-C，實作補記 7、8、15、16，兩輪審查；
  - 042：D-C 的 gateway 位移、W 的第二波結構、D-B 的 keep-alive 推論、D5；
  - 029、038：gateway、准入、access log、metrics；
  - 037：§2、§5 第 2 步；
  - 021：R1、R3；
  - 013、005。
- 時點：
  - 在 043 的使用者決定（2026-10-24）之後；
  - 在任何 V 數據之前，也在 044 的產品程式之前：產品程式依本文以 TDD 寫成，在凍結之前完成。
- 來源：
  - 三份設計草稿：精簡、嚴謹、產品保真。三位評審比較後，兩位選精簡版，所以本文以精簡版為骨架；
  - 併入嚴謹版的效度設計：V2 的 manipulation check、V4 的 TRIVIAL 類別、雙射式的 log 對應、var 的 2×2 讀法、能主張與不能主張、凍結前的 U_W 模擬、每個 block 換序列；
  - 併入產品版的機制：limits 相同的 study provisional profile、V2 的雙 view 判定、機械式的 2 h 空檔、只凍結請求路徑；
  - 改正評審找到的錯誤，見「設計審查改正的項目」。

## 範圍

| 研究 | 問題 | 點（C） | 解鎖什麼（只在單位內每一點都 PASS 時） |
|---|---|---|---|
| V-GW | Stage D 的 class 經過 capserve gateway，在 C_PROD 是否仍成立（D-C 的協定） | GW-U32K（2）、GW-U2K-O512（32）、GW-U8K-O512（12）、GW-M8K-T（4，T server）；不做 U8K-T（030 8.8：只作報告） | uniform-32k、mixed-8k-t 可以列入正式釘選的候選；兩個 O512 點是 V1 單位的一部分 |
| V1 | 輸出長度在 [1, 512] 均勻分布、事先 pin 住時，SLO 是否成立 | L-U2K（32）、L-U8K（8），經過 gateway | 新的 class `uniform-2k-o1-512`（32）、`uniform-8k-o1-512`（8） |
| V2 | 開場湧入的 prefill pacing（開 / 關），經過 gateway | P-U2K（32）、P-U8K（8）、P-U2K-var（32）；對照組（關）F-U2K（32）、F-U2K-var（32） | uniform-2k、uniform-8k 各自的 `pacing` 段 |
| V4 | `weighted-v1` 多 class 准入，pacing 關閉 | W-90（90/10）、W-50（50/50），client 32 | `admission` 段 |

- V3 依使用者的決定延後。V5（T 上的 IT-0–6）是產品測試，不需要事先登記。
- 結果目錄為 `results/stage_v/<sid>/`，記在 PREREGISTRATION §12（見文末附錄）。

### 和 043 字面的差異（CLAUDE.md：先記錄再處理）

1. **共用 harness**：四個研究共用一套 harness、一份 `configs/stage_v.toml`、一次 mini 與一次凍結。043 字面是每項各自一套。
   - 判定規則不變。
   - 依 039 實作補記 15，任何 harness 修正都會讓所有 mini 失效，分開跑沒有好處。
2. **V1 經過 gateway**：043 沒有寫明。
   - 產品一定經過 gateway；
   - 「輸出長度不一加上 gateway」正是 043 指出缺少的組合；
   - GPU 時數相同。
3. **O512 與 V1 的 class 以 study provisional profile 服務**：用 limits 逐欄相等的 `v-` 複本，在 `--allow-provisional` 下服務。
   - 043 分支 `616e066` 的 `serving_class` 只在 `self.verified` 而且輸出上限 > 128 時拒絕，所以 043 的這條規則不需要修改。
4. **候選政策（pacing、weighted-v1）的研究用途**：可以在 provisional profile 上以 `capserve serve --allow-provisional` 啟用。
   - 043 7.3 寫的是「serve 與 deploy 一律拒絕，除非有 verified 的段」；
   - verified profile 仍一律要 §12 段；
   - `deploy` 遇到沒有 §12 段的候選政策一律拒絕，provisional 的設定也一樣。（待決事項的處理第 1 項）
5. **V2 的判定**：取 upstream（准入之後）與 client（從到達起算）兩個類別中較差者。
   - 043 字面是「主要的是 upstream，兩者並列」，本文更嚴；
   - c_full 只作描述，因為第一波依物理會讓 client 端兩臂都 FAIL，upstream 端則依構造通過，c_full 無法區分兩臂。
6. **V2 的 var 點不作為 pacing 的閘門**：043 7.4 第 6 項寫「每一個點都 PASS」，本文改為否決規則：若 F-U2K-var PASS，而 P-U2K-var 不是 PASS，就否決 PACE-2K。
   - 原因：依預測兩點都會 FAIL。照字面，pacing 永遠不會上線；而 var 的 FAIL 在 D-C 已經存在，和 pacing 無關。
7. **V2 也有非平凡條件**（N1）與 manipulation check（E）：043 只要求 V4「（以及 V2）」寫定非平凡條件。
8. **V4 的非平凡條件以准入單位計**：043 的寫法是「在途數 ≥ 有效並發 24.6 / 12.8 的 90%」，這裡改成「vLLM 內平均單位數 ≥ 0.9 × 32」。
   - 原因：8K 在 vLLM 內停留較久，在途組成會偏向 8K，以 request 數計會因組成而失敗，而不是因為政策；
   - request 數的版本並列報告。
9. **GW-U8K-O512（12）有產品用途**：它是 V1-8K 單位中「全部 512」的最壞情況。依 030 的單調假設（C_PROD 本來就涵蓋 ≤ C 的每個並發），12 的 PASS 涵蓋 8。
   - 043 §4 規定 12 本身永遠不能服務，這一點不變。
10. **V1 的 TPOT 只算 out ≥ 2 的 request**：030 的公式在 out = 1 時沒有定義，vLLM 0.30 的 bench 也只對 output_len > 1 計算 TPOT。TTFT 照常包含 out = 1。
11. **產生器的時間點**：依 043，產生器的 validation-session 路徑在第一個正式 V session 之前完成。
    - 它不在凍結的請求路徑閉包內，所以可以在凍結之後才寫；
    - 對應規則在本文固定。

## 沿用的規則（只寫差異）

- **030 8.1**：
  - S2：短類 P95 TTFT 2 / 6 / 20 s 依 prompt 的長度等級，P95 TPOT 100 ms，mixed 的長類為 budget；
  - guard band m = 0.05，取最差的 run，比較用 Fraction 精確計算；
  - 單點類別 FAIL / NT / PASS / MARGINAL，FAIL 優先；
  - S1 / S3 只作描述性的界線。
- **時間窗**：W = [C, N − C)，W′ = [2C, N − C)，full；N = max(200, 10C)，warmup = C。
- **039 D-C**：
  - 固定點，n = 5，每個 run 一次 boot；
  - block r = 1..5 的順序為 `capacity_plan.block_order(20261008, r, 該 session 的點)`；
  - 跳過規則：一次有效的 FAIL 或一次無效的 run 之後，不再測該點；無效的 run 不補跑（013）。
- **030 補記二第 1 點**：合併的長類（GW-M8K-T 的 32K、W 點的 8K 槽）只在最後判定一次，不觸發跳過。
- **有效性**：030 8.6 (a)–(g)、031 的閒置判定與兩段式 spill 偵測、replay。
- **sentinel**：B0 直連、U2K、C 25、N 250，開頭與結尾各一次，閘門 3%；CROSS_SESSION_SHIFT 5% 只作描述，參考值為 Core B uniform 與 Advanced。
- **逾時**：ceil60(2 N τ1 + 60 s)，τ1 取凍結的 `stage_d_plan.tau1(stage_d.toml, <τ1 base>)`；server 就緒 1,800 s，gateway 就緒 120 s。
- **039 實作補記 7、16 的 gateway 規則**：
  - 以子 process 執行 `python -m capserve serve`，研究程式不 import capserve；
  - port 8080 被佔用時中止 session（`gateway_port_busy`）；
  - `/readyz` 不符時，該 boot 在第一個 bench 之前失敗；
  - bench 前後都檢查 gateway 是否還在；
  - 先停 gateway：SIGTERM，30 s 後 SIGKILL，在 `run_adv.DeferredSignals` 之下；
  - gateway 的環境移除所有 `CAPSERVE_*`，並設 `HF_HUB_OFFLINE=1`；`drain_timeout_s` 為 5 s。
- **偵測力**（030 8.1）：單一 run 機率為 q 的壞模式，n = 5 時 q ≥ 0.45 才能以 ≥ 95% 的機率抓到。

## 共通的定義

**兩種 TTFT**：每個 request 都記錄，由 access log 以 `client_request_id` 對應。
- **client**：bench 的 TTFT，從送出起算，包含 gateway 的 body 處理、tokenize、排隊與 pacing。
- **upstream**：access log 的 `upstream_ttft_s`，從 gateway 送往 vLLM 起算，和 C_PROD 是同一個定義（029）。
  - 「第一個 token」的時間點和 vLLM 0.30 bench 相同：第一個 JSON 帶非空 `choices` 的 `data:` event，text 可以是空字串；gateway 的 `ttft_s` 也記在這一刻（實作補記六第 1 項）。
  - upstream view 的換算：ttft := upstream_ttft，latency := upstream_ttft + (latency − ttft)；所以 TPOT 不變。

**每個研究的判定 view**：

| 研究 | 判定 view | 並列報告 |
|---|---|---|
| V-GW、V1 | client（同 D-C 的 H-D7） | upstream |
| V2 | upstream 與 client 的類別取較差者 | — |
| V4 | upstream（gateway 對 vLLM 扮演 bench semaphore 的角色） | client（32 個 client 多於政策的有效並發，依構造會排隊） |

**每個 run 的統計量**：
- Y_short：W 中短類的 P95 TTFT（`core_b_analysis.percentile`）；
- Y_TPOT：W 中 request 的 P95 (latency − ttft) / (out − 1)，L 點只算 out ≥ 2；
- Y_long：跨 run 合併；
- R = Y / S，maxR_run = max(R_short, R_TPOT)。

**在 vLLM 內的區間**：access log 新增 gateway 的 monotonic 時間 `t_arrival_s`、`t_sent_s`、`t_released_s`。
- 區間為 [t_sent, t_released)；
- `t_released_s` 在 `Permit.release` 中、把 slot 交給下一個 waiter 之前記錄，所以相鄰 permit 的區間不會重疊；
- T_W = [t_arrival(idx C), t_arrival(idx N − C))。

**run outcome**：
- `invalid`：任一有效條件不成立；
- `fail`：有效，任一判定 view 的 maxR_run > 1；
- `trivial`：有效，沒有 fail，但非平凡條件不成立（V4 的 U_W、V2 的 N1）；
- `ok`：其他。
- 跳過：outcome 不是 `ok` 時，只有該點之後的 boot 不再測，同一產品單位的其他點照常測（沿用 039 / 030 的逐點規則；實作補記六第 6 項取代原本的「單位已死」）。單位仍要每一點都 PASS 才成立（產品對應）。

**點的類別**：FAIL > TRIVIAL > NT > MARGINAL > PASS，依序判定，第一個成立的就是結果。

| 類別 | 規則 |
|---|---|
| FAIL | 任一有效 run 為 fail，或合併的 R_long > 1 |
| TRIVIAL | 任一有效 run 為 trivial |
| NT | 有效而且 `ok` 的 run 少於 5（包括被跳過、被時數上限停止） |
| PASS | 5 個 `ok` 的 run，每個判定項目與 view 都 ≤ 0.95，合併的 R_long ≤ 0.95 |
| MARGINAL | 其他 |

- 判定：PASS 為 supported；FAIL、TRIVIAL 為 not supported；MARGINAL 為 inconclusive；NT 為 NT。

**只作描述**（每個點都報告）：
- W′ 與 full 在兩種 view 下的類別、WINDOW_SENSITIVE；
- 第一波（idx < C）的最大 TTFT；第二波（idx 在 [C, 2C)）的最大值與超過門檻的個數。2K、C 32 時第二波占 W 的 12.5%，決定 W 的 P95（042）。
- 逐 request 的分解：
  - client − gw ttft_s：傳輸；
  - gw ttft_s − upstream_ttft_s − queue_wait_s − prefill_wait_s：gateway 內部的 body、parse、tokenize；
  - upstream_ttft_s：vLLM 端。
  - 報告 W 中的中位數與 P95。這是第一次能拆開 042 的 +390 / +710 ms，但和 D-C 的比較跨版本、跨 session，不能歸因。
- 在 vLLM 內的平均與最大值：request 數與單位數；
- KV、preemption：整個 bench process 的值（042）。

## V-GW：Stage D 的 class 經過 gateway（在 C_PROD）

| 點 | server | profile / class | client（凍結的函式） | τ1 base | C | N | 逾時 |
|---|---|---|---|---|---|---|---|
| GW-U32K | B0 | 已 commit 的 `verified-formal-20261023-d-a1b-b0` / uniform-32k（verified，不加旗標） | `bp.bench_argv(L=32768, uniform, port=8080)` | U32K | 2 | 200 | 3,240 s |
| GW-U2K-O512 | B0 | `provisional-stage-v` / `v-uniform-2k-o512`（1,920 / 512，32 / 32） | `sd.bench_argv_for("U2K-O512")`，port 改 8080 | U2K-O512 | 32 | 320 | 4,800 s |
| GW-U8K-O512 | B0 | `v-uniform-8k-o512`（8,064 / 512，12 / 12） | `sd.bench_argv_for("U8K-O512")`，port 改 8080 | U8K-O512 | 12 | 200 | 4,800 s |
| GW-M8K-T | T（`sd.server_argv_for(pins, "M8K-T")`，log 必須有 512） | 已 commit 的 `…-d-a1b-t` / mixed-8k-t（verified；H-D2a supported） | `bp.bench_argv(L=8192, mixed, port=8080)`，用 007 的 8K mix（`ab548bed…`） | M8K-T | 4 | 200 | 2,820 s |

- 佇列設定為產品預設（max_queue_wait_s 2 s、request_deadline_s 300 s），和 D-C 相同。在 C_PROD 時不會排隊。
- 兩個 O512 複本的 limits 逐欄等於 `-d-a2-b0` 的 class（測試鎖定），只有標籤和 verified 不同（揭露）。

| id | 點：PASS | 預測 | 依據（跨 session、跨 gateway 版本的外推，只用來預測） |
|---|---|---|---|
| H-GW1 | GW-U32K | supported | D-A1b 在 C 2 的最差 R 為 0.512 / 0.642；gateway 讓 TTFT 增加約 0.35–0.72 s，只占 20 s 門檻的 2–4%；D-C 中 TPOT 沒有變差（時間從 decode 移到 TTFT） |
| H-GW2 | GW-U2K-O512 | supported（低信心） | 直連 0.708（1,416 ms），加上 D-C 2K 在 W 的 +214 ms（W′ 為 +390 ms），約 1.63–1.81 s，R 0.82–0.90；3.125% 的 4–5 s 停頓 < 5%；32 → 33 是斷崖 |
| H-GW3 | GW-U8K-O512 | supported | 直連 0.425（2,550 ms），加上 8K 在 W 的約 +720 ms，R 約 0.55 |
| H-GW4 | GW-M8K-T | not supported（MARGINAL 或 FAIL） | 直連短類 0.886（5,316 ms），加上 8K 的 +350–720 ms，R 0.94–1.006。1.006 是本文自己的計算，043 §8 只寫「可能進入 guard band」 |

- **描述性預測**：W 中 queue_wait_s 的 P95 < 50 ms。這是 039 H-D7 的理由第一次被量到。

**判定表**

| 類別 | 處理 |
|---|---|
| PASS | GW-U32K、GW-M8K-T 成為正式釘選的候選；GW-O512 計入它的 V1 單位 |
| MARGINAL / FAIL / NT / TRIVIAL | 不成為候選，寫進 README 的適用條件；不改測較低的 C（那是新的估計量） |

**能主張與不能主張**
- 能：在 C_PROD、封閉迴路、這一版凍結的 gateway 請求路徑下，client 端的 SLO 成立。
- 不能：
  - prompt 長度不一、開放迴路、開場湧入；
  - 其他 gateway 版本；
  - 其他輸出長度：GW 點只有 128 或 512。

## V1：輸出長度在 [1, 512] 不一（經過 gateway）

| 點 | server | class | client | τ1 base | C | N | 逾時 |
|---|---|---|---|---|---|---|---|
| L-U2K | B0 | `v-uniform-2k-o512`（1,920 / 512，32 / 32） | `sd.bench_argv_for("U2K-var", dataset=v1-2048-r<b>)`：port 改 8080，`--custom-output-len 128` 改為 `-1` | U2K-O512 | 32 | 320 | 4,800 s |
| L-U8K | B0 | `v-uniform-8k-o512-c8`（8,064 / 512，8 / 8，就是要解鎖的上限） | 同上，由 `"U8K-var"` 產生 | U8K-O512 | 8 | 200 | 4,800 s |

**dataset**：每點 5 條序列，block r 用第 r 條，所以 n 也涵蓋序列的變異。
- 每列 `{"prompt", "output_tokens"}`，prompt 固定 1,920 / 8,064 token；
- row 0 的輸出為 128，所以 warmup 等於 U2K / U8K 的 warmup；
- 其餘每列的輸出由 `Random(sha256("20261008:stage_v:v1:<tier>:<r>")).randint(1, 512)` 產生。
- **vLLM 0.30 原始碼的唯讀核對**（`CustomDataset.sample`、`serve.py`、`endpoint_request_func.py`）：
  - 只有 `--custom-output-len -1` 時，才逐列讀取 `output_tokens` 作為 `max_tokens`，預設的 256 會覆寫每列的值；
  - 加上 `--ignore-eos`，輸出就剛好等於它；
  - 結果的 `output_lens` 來自 usage 的 completion_tokens。

**估計量**：client view；TTFT 用 W 中全部 request；TPOT 只算 out ≥ 2。
- 描述：依輸出長度分箱（1–16、17–64、65–128、129–256、257–512），報告 TTFT / TPOT 的 P95 與超過門檻的比例。

| id | 點：PASS | 預測 | 依據 |
|---|---|---|---|
| H-L1 | L-U2K | supported | U2K-gw 為 0.696 / 0.923。平均輸出 256.5，完成率約為輸出 128 時的一半，prefill 插隊的步數比例約減半（042 D-A2 的機制）；每個 request 需要 4–5 個 block，32 個約 148 ≤ 247，預期沒有 preemption。風險：極短輸出的 TPOT 只平均少數幾個 step |
| H-L2 | L-U8K | supported（低信心） | U8K-gw 為 0.533 / 0.885；8K 的 prefill 占 4 個 chunk step；out ≤ 20 的 request 約 4%，其中和 prefill 重疊、TPOT 超過 100 ms 的估計約 2–3%，低於 5% |

**判定表**
- **V1-2K 單位**：L-U2K 與 GW-U2K-O512 都 PASS。
- **V1-8K 單位**：L-U8K 與 GW-U8K-O512 都 PASS。
- 任一點不 PASS：該單位不上線。
- L 點 FAIL 也否定 043 的包絡規則在該 tier 的值，因為 V1 量的正是 min(32, 32) = 32 與 min(12, 8) = 8。

**能主張與不能主張**
- 能：在 5 條 [1, 512] 均勻分布的序列、經過這一版 gateway、C 32 / 8 下，SLO 成立；「全部 512」的情況由 GW-O512 涵蓋。
- 不能：
  - 其他輸出分布，例如大量短於 32 token 的輸出；
  - EOS 造成的自然長度、thinking 開啟；
  - prompt 長度不一。

## V2：prefill pacing（經過 gateway）

**參數**：依規則凍結，不是依數據選擇。
- rate = ⌊prompt ÷ 斜率，取到 10 tok/s⌋。斜率是 `results/stage_d/d5-startup-wave/analysis.json` 中，C ≤ C_KV（2K 35、8K 13）各點第一波最大 TTFT 平均的最小平方斜率：2K 為 324.23 ms，8K 為 1,389.15 ms。
  - 檔案中的 `fit` 用的是全部的點（365.4 / 1,462.5 ms），不能直接引用。
  - 結果：2K 5,920 tok/s，8K 5,800 tok/s；測試從 committed 的檔案重算並比對。
- burst：2K 9,600（5 × 1,920），8K 32,256（4 × 8,064），沿用 042 D5 的換算。
- var 點用 2K 的參數。
- 機制（043 7.3 B）：
  - 單一嚴格 FIFO，隊首要同時拿到並發位置與 token；
  - `prefill_wait_s` = 隊首有空的並發位置、但 bucket 的 token 不足時等待的時間。

| 點 | class | pacing | dataset / client | C | N | 逾時 |
|---|---|---|---|---|---|---|
| P-U2K | `v-uniform-2k`（1,920 / 128，32 / 32） | 開 | `sd.bench_argv_for("U2K-gw")`，原樣 | 32 | 320 | 1,260 s |
| P-U8K | `v-uniform-8k`（8,064 / 128，8 / 8） | 開 | `sd.bench_argv_for("U8K-gw")`，原樣 | 8 | 200 | 1,260 s |
| P-U2K-var | `v-uniform-2k` | 開 | D-C 的 var-2048（`358ae8da…`，同一條序列），port 改 8080 | 32 | 320 | 1,260 s |
| F-U2K-var | `v-uniform-2k` | 關（對照組） | 同上 | 32 | 320 | 1,260 s |
| F-U2K | `v-uniform-2k` | 關（對照組，待決事項第 6 項） | `sd.bench_argv_for("U2K-gw")`，原樣 | 32 | 320 | 1,260 s |

- **佇列設定為「慢慢服務」**：max_queue_wait_s 120 s、request_deadline_s 600 s，max_queue 32，所有點相同。
  - 不測「快速 429」：凍結的 bench 把非 200 都算成 failed，第一波一定會出現 429，run 依 8.6 (a) 無效，在封閉迴路下量不到。
- **E（manipulation check，有效條件）**：每個有效 run，第一波中 prefill_wait_s > 0 的個數 ≥ C − b − 1。
  - b 是第一波中最短的幾個 prompt 能放進 burst 的最大個數，和到達順序無關：uniform 2K 時門檻為 26，8K 為 3，var 由 pin 住的長度計算；
  - 減 1 是容許第一波到達期間的補充；
  - 關的點：沒有任何 completion 紀錄的 prefill_wait_s > 0。
  - E 不成立時 run 無效。
- **N1（非平凡條件）**：T_W 中 vLLM 內 request 數的時間平均 ≥ 0.9 C（28.8 / 7.2），否則該 run 為 trivial。

| id | 點：PASS（兩種 view 取較差者） | 預測 | 依據 |
|---|---|---|---|
| H-P1 | P-U2K | supported | 穩態的 token 到達約 4,890 tok/s，低於 rate；限制發生時的延後約等於 vLLM 本來就有的串行 prefill（043 7.1）；風險：U2K-gw 的 TPOT 餘裕只有 0.027 |
| H-P2 | P-U8K | supported | 穩態約 5,080 tok/s，為 rate 的 88%，偶爾被限制；U8K-gw 為 0.533 / 0.885 |
| H-P3 | P-U2K-var | not supported | pacing 不改變 vLLM 內 prefill 的物理；D-C 直連的 var 已是 1.021，經過 gateway 再上移約一格 |
| — | F-U2K-var（對照組，不是 hypothesis） | FAIL | 同上 |
| — | F-U2K（對照組，不是 hypothesis；新版 gateway 下正式 class 的重新驗證） | PASS | U2K-gw 為 0.696 / 0.923；TPOT 餘裕只有 0.027 |

- **描述性預測**：第一波的最大 client TTFT 和 D-C 的 U2K-gw / U8K-gw 相差 ≤ 10%。這是跨 session 的比較，只作描述。

**判定表**
- **PACE-2K**：P-U2K PASS（每個有效 run 都通過 E 與 N1），而且沒有被否決。
- **PACE-8K**：P-U8K PASS。
- 否則該 class 的 pacing 不上線。

| P-U2K-var | F-U2K-var | 讀法 |
|---|---|---|
| PASS | FAIL | 只限這條序列、n = 5：pacing 與通過同時出現（不寫「pacing 修好 var」） |
| PASS | PASS | 新版 gateway 下 var 已經通過，pacing 的效果沒有被顯示 |
| FAIL | FAIL | 預測：pacing 沒有修好 var |
| 非 PASS | PASS | pacing 在 var 上有害，**否決 PACE-2K** |
| 其他 | — | 無法判讀 |

| P-U2K | F-U2K | 讀法 |
|---|---|---|
| PASS | PASS | 新版 gateway 下正式 class 仍成立；PACE-2K 依上面的規則 |
| 非 PASS | PASS | 失守歸因於 pacing，PACE-2K 不上線 |
| 非 PASS | 非 PASS | 新版 gateway 下正式的 uniform-2k（32）已不成立；寫進結果 decision，由使用者決定（不自動改釘選） |
| PASS | 非 PASS | 無法歸因（同一 session 內的偶發），PACE-2K 依規則判定，並揭露 |

**能主張與不能主張**
- 能：pacing 開啟時，穩態的 SLO 在兩種 view 下都成立，而且 vLLM 中的並發 ≤ C_PROD。
- 不能：
  - 第一波的 TTFT 變好；
  - 快速 429 時的行為；
  - 開放迴路；
  - 這條 var 序列以外的情況。

## V4：`weighted-v1` 多 class 准入（pacing 關閉）

- **政策**（043 7.3 A）：
  - 依 local tokenizer 的計數分類：≤ 1,920 token 為 2K，1 單位；≤ 8,064 token 為 8K，4 單位；更長的回 400；
  - 一律計數，不走 `fast_fits` 的 byte 快速路徑；
  - 預算 32 單位，各 class 上限 32 / 8；嚴格 FIFO，包括隊首阻塞（HOL）；
  - max_queue 32，佇列設定為「慢慢服務」（120 s / 600 s）；
  - class 為 `v-uniform-2k` 與 `v-uniform-8k`。
- **client**：`sd.bench_argv_for("U2K-var", C=32, dataset=v4-<mix>-r<b>)`，port 改 8080，輸出 128，N 320，W [32, 288)，τ1 base 為 U8K-gw（2.95 s），逾時 1,980 s。
- **dataset**：每個 mix 5 條序列，320 列。
  - row 0 為 2K，所以 warmup 是 32 個 2K，和 U2K 相同；
  - 其餘 319 列依確切個數，以 `sha256("20261008:stage_v:v4:<mix>:<r>")` 為 seed 洗牌：90/10 為 287 × 2K 加 32 × 8K（合計 288 / 32），50/50 為 159 + 160（合計 160 / 160）。
  - W 中 8K 的個數由 dataset 決定，寫進 plan。
- **估計量**：upstream view。
  - 2K request：每個 run 對 2 s；
  - 8K request：放進「long 槽」，跨 run 合併對 6 s，和 Y_long 相同，不觸發跳過；
  - 所有 request 的 TPOT 對 100 ms。
  - 實作：`ca.run_eval` / `ca.classify` 以 L 2048、mix "mixed"、SLO 視圖 `ttft_p95_ms = {2048: 2000, 8192: 6000, 32768: 6000}` 計算，由測試鎖定。
- **非平凡條件**（每個有效 run）：
  - U_W = (1 / |T_W|) ∫_{T_W} Σ_j units_j · 1[t_sent_j ≤ t < t_released_j] dt；
  - 要求 U_W ≥ 0.9 × 32 = 28.8（Fraction），否則該 run 為 trivial。
  - 這排除「全部延後」；「全部拒絕」會出現 429，讓 run 無效。
  - 「vLLM 內 request 數 ≥ 0.9 × 24.6 / 12.8」只作描述。
- **凍結前的設計檢查**（嚴嚴謹版）：
  - 在 Mac 上，用已 commit 的 c1b 與 D 的逐 request 時間，做 weighted-v1 / 嚴格 FIFO 的離散事件模擬，估計兩個 mix 的 U_W；
  - 結果 commit 到 `results/stage_v/prereg-sim/`（descriptive_only），不是數據；
  - 任一 mix 的模擬平均 < 0.93 × 32 時，凍結前在本文修改門檻或設計，並揭露。

| id | 點：PASS 且每個 run 的 U_W ≥ 28.8 | 預測 | 依據 |
|---|---|---|---|
| H-W1 | W-90 | not supported（2K 的 TTFT） | 8K 約占 prefill token 的 32%（258,048 / 811,008）；一個 8K 要 4 個 MNBT-2048 step（約 1.39 s），期間到達的 2K 要等；W 中 2K 的基準約 1.0 s，加上被擋住的時間 > 2 s，遠超過 5% 的 2K 會遇到；D-C 的 var 在更小的異質性下已讓 W′ 上移 65–71% |
| H-W2 | W-50 | not supported | 8K 約占 prefill token 的 81% |
| H-W0 | 兩個 mix 每個有效 run 的 U_W ≥ 28.8 | 成立（由模擬確認） | HOL 預估只浪費約 1–1.5 單位 |

**判定表**

| W-90 | W-50 | 結果 |
|---|---|---|
| PASS | PASS | **W1 單位**解鎖，可以寫出 `admission` 段 |
| 任一不是 PASS | — | 不上線。不得事後改成限制 8K 比例、2K 優先或較小的預算（那是新設計，需要新的 decision） |
| 任一為 TRIVIAL | — | 不上線，在結果 decision 記為政策缺陷（無法填滿預算）；不改 0.9 |

- 結果如預測時，044 之內不再做多 class 的量測。

**能主張與不能主張**
- 能：在 mix 0.9 / 0.5（各 5 條序列）、預算飽和、pacing 關閉時，准入之後的 SLO 成立。兩端的 1.0 / 0.0 是 c1b 的 U2K @ 32 與 U8K @ 8。
- 不能：
  - 從到達起算的 TTFT；
  - 0.5–0.9 以外的 mix（中間的 mix 只是內插）；
  - 含 32K、T、開放迴路；
  - pacing 與 weighted-v1 同時開啟；
  - 產品預設的 2 s / 429 佇列設定。

## 有效性（030 8.6 (a)–(g)、031、039 D-C，再加上下列各項）

- **(h) access log 對應是雙射**：每個 boot 只有一個 run，所以在 `gateway_down` 之後解析 log；這時 log reader 已經 join，結果是決定性的。
  - completion route 上 `client_request_id = <tag>-<idx>` 的紀錄，idx 0..N−1 各剛好一筆，status 200，outcome ok；
  - 每筆都帶 queue_wait_s、ttft_s、upstream_ttft_s、duration_s、t_arrival_s、t_sent_s、t_released_s；P 點另有 prefill_wait_s，W 點另有 admission_class / admission_units。
  - completion route 上沒有 id 的紀錄剛好 C 筆，也就是 warmup：凍結的 client 不送 ready check（vLLM 0.30 的 `--ready-check-timeout-sec` 預設為 0，沒有任何 argv 設定它），request 也不重試，所以少一筆或多一筆都是問題；每筆都帶 t_arrival_s、t_sent_s、t_released_s，(j) 也讀它們的區間（實作補記六第 9 項；實作補記七第 1 項）；
  - `/metrics`、`/readyz`、`/healthz` 的紀錄一律排除。bench 在主要 run 前後會對 base_url 發 GET /metrics，runner 也會輪詢 /readyz。
  - random dataset 的 client（`--dataset-name random`）在 warmup 之前送一次沒有 id 的 `POST /tokenize`（vLLM 0.30 的 prompt 對齊）；gateway 沒有這個 route，回 404，bench 因此略過對齊。剛好一筆、狀態 404、在所有 completion 紀錄之前時接受，不計入沒有 id 的筆數，也不適用 (i)；其他 `/tokenize` 紀錄（多一筆、少一筆、別的狀態、較晚、custom dataset 的 client）都是問題（實作補記六第 5 項；「在 warmup 之前」原寫「在 ready check 之前」，實作補記七第 1 項）。
  - 沒有其他 prefix；
  - 每筆 gw ttft_s ≤ client ttft + 50 ms。
- **(i)** 主要 request 與 warmup 都沒有任何 4xx、429 或 5xx。
- **(j)** 1 Hz sampler 的 running + waiting 峰值 ≤ C（V4 為 32），由 access log 的區間重算，主要 request 之外也包括 C 個 warmup 的區間（實作補記六第 9 項；凍結的 client 不送 ready check，實作補記七第 1 項）：
  - 在途數 ≤ C；
  - V4 的單位 ≤ 32，2K ≤ 32 個，8K ≤ 8 個。
  - 整個 bench process 都經過 gateway，所以任何時刻都必須成立；D-C 的 21 個 run 都符合。
- **(k)** L 點：output_lens 逐一等於 pin 住的 output_tokens，input_lens 全部等於 prompt。
- **(l)** W 點：input_lens 逐一等於 dataset，每個 idx 的單位數等於它的 class（N2）。
- **(m)** P 點：E。
- **boot**：`/readyz` 必須等於該點凍結的 tuple，否則該 boot 在第一個 bench 之前失敗。tuple 包括：
  - profile id 與 status；
  - class 或 classes；
  - limits；
  - 佇列設定；
  - admission 與 pacing 的政策與參數；
  - `gateway_path_sha256`；
  - serving status（verified 或 provisional）。
- **session**：
  - formal，`freeze.stage_v` 為 true 而且附 mini 證據；
  - session 開始時 `gateway_path_sha256` 等於凍結值；每個點的 profile、每個 dataset、渲染出的 gateway settings 的 sha256 都等於計畫值；
  - DEPLOYED_COMMIT 單一而且乾淨，vLLM 0.30.0，持有 lock，沒有繼承任何 engine 變數；
  - 兩次 sentinel 都有效而且通過閘門；
  - replay 一致：順序、逐點的跳過（含 trivial 與無效的 run）、時數上限的停止；
  - 和上一個正式 V session 的 `finished_at` 相隔 ≥ 2 h。
  - session 另記錄完整的 `capserve_sha256`（`rd.capserve_sha256`），只作 provenance。

## 停止規則與時數上限

- **session 的時數上限**：mini 1.0 h、V-ADM 3.5 h、V-CLS 7.0 h，約為估計的 1.3 倍。
  - 每個 V boot 開始之前：若已經過的時間 + 該點事先寫定的估計 > 上限 − 0.06 h（預留給結尾 sentinel），就不開始；
  - 之後所有計畫的 run 都記為 not run（cap），相關的點為 NT；
  - 結尾 sentinel 照跑，session 仍然有效；
  - report 用記錄的 boot 起始時間重算這個決定。
- **全期總上限 12 h**（mini、正式、重跑、中止的 session 都算）：開始任何 V session 之前，若已用的時數 + 該 session 的上限 > 12 h，就不開始，先問使用者。
- **中止**（031 gpu_not_idle、開頭 sentinel 無效、`gateway_port_busy`、spill 讀不到、lock、任何 sha 不符）：
  - 以新 id 重跑，不接續；
  - 在第一個 V boot 之前中止、而且 0 個 V run 的不算嘗試（030 8.4）；
  - 每個 session 最多 2 次嘗試，第 3 次要先寫 decision log 由使用者決定。
- **sentinel 閘門失敗**：該 session 不解鎖任何單位；可以用新 id 重跑一次，算第 2 次嘗試。
- **gateway 造成的無效 run**（gateway 結束、5xx、對應不完整、(j) 被違反、`/readyz` 不符）：
  - 不開始下一個 V session，先寫 decision log；
  - 修正 capserve 之後，要新的凍結與新的 mini；
  - 兩個 session 都在新的凍結下重跑。舊結果只對舊的 sha 有效，依服務規則不能上線。
- **看到數據之後不改**：指標、view、門檻、時間窗、0.9、序列、政策參數、佇列設定、跳過規則。
  - 不重跑有效的 session 來換 PASS；新的參數就是新的研究。
- **GPU**：
  - 拿不到 lock 就等，標記 blocked，繼續 Mac 上的工作，不搶占，不刪別人的 lock；
  - 每個 session 一個前景 ssh；
  - 不和其他正式 session 串接。

## Sessions 與時數

### 前置與時序（排在 E1-P、E1-D 取回與 Stage Q 的 RTX pilot 之後）

1. **043 合併到 main**：`feat/043-stage-d-profiles` 的 `616e066` 已完成 043 第 1–3 步，共 11 個檔案：
   - schema 與 loader；
   - `serving_class` 的兩條拒絕；
   - deploy 拒絕 T。
   - 還沒做的是第 4–10 步：產生器、重新產生的測試、三份 profile 的 commit、揭露、keep-alive 75 s、文件、審查。
   - 為什麼是前置：GW-U32K / GW-M8K-T 要服務已 commit 的 profile 位元組；keep-alive 是 V2 / V4 有效的前提，因為 pacing 與佇列會讓 warmup 的完成時間相差超過 5 s。
2. **044 的產品程式**（Mac，TDD）合併。
3. **V harness、stage_v.toml、測試**；U_W 的桌面模擬。
4. **兩輪獨立審查**（harness 與產品程式），都在任何 GPU 使用之前，以免 mini 要重跑。
5. **datasets**：WSL、只用 CPU、`flock -n` 下產生，和 E1-P 的 seed 檢查相同；sha256 pin 進 stage_v.toml 並 commit。
6. **GPU 視窗 0**（一個 lock）：
   - 部署 HEAD；
   - 產品 IT-0–6（`rtx-verified.toml`，043 第 8 步，約 0.3 h，輸出只進 `runs/`）；
   - mini `smoke-v-<date>`；
   - 釋放 lock。
7. commit mini → 凍結 commit → push → 部署凍結 commit。
8. **產生器的 validation-session 路徑**（Mac，TDD，合成 session），在第一個正式 V session 之前完成。
9. **GPU 視窗 1**：`formal-<date>-v-adm`。
10. **空檔**：≥ 2 h，由 runner 強制，並釋放 lock。V-ADM 有 gateway 造成的無效 run 時停在這裡。
11. **GPU 視窗 2**：`formal-<date>-v-cls`。
12. **Mac**：
    - 取回（不含 raw/）、report；
    - 結果 decision；
    - 產生器只為通過的單位寫 profile；
    - README；
    - 使用者決定是否改釘選。

- 每個 session 以一個前景 ssh 執行：Mac 端以 `caffeinate -i` 包住，並加上 `ServerAliveInterval`；各自 `flock -n /tmp/ai-portfolio-gpu.lock`。

**每個 session 的結構**：開頭 sentinel → block r = 1..5，每個點一次 boot、一個 run → 結尾 sentinel。
- **boot 內的順序**：
  1. 時數上限檢查；
  2. port 8080 檢查；
  3. 031 的閒置判定；
  4. server 啟動與檢查、spill；
  5. gateway 與 `/readyz`；
  6. bench；
  7. `gateway_down`，log reader join；
  8. 解析 access log，寫 derived；
  9. decide；
  10. server teardown。
- **時數的估算**：
  - 每個 run 的固定開銷：B0 95 s，取 D-C 的 (1.557 − 1.009) h ÷ 21 = 94 s；
  - T 取 110 s：d-a1b 的 T boot `startup_s` 和 B0 同為 32 s；d-a1b 每次 boot 平均 129 s，但每次 boot 平均有 2.85 個 run；加 15 s 餘裕給 M8K 的 warmup；
  - bench 的時間取 042 的實測；V1 / V4 依實測內插。

### V-ADM（先跑：被操弄的准入程式風險最高；若要修正 capserve，V-CLS 直接在修正後的版本上跑）

| 點 | bench 估 | 5 個 run（h） |
|---|---|---|
| P-U2K | 130 s（U2K-gw 實測 125 s） | 0.31 |
| P-U8K | 325 s（318 s） | 0.58 |
| P-U2K-var | 80 s（D-C 同一份 var 實測 75 s） | 0.24 |
| F-U2K-var | 78 s | 0.24 |
| F-U2K | 125 s（U2K-gw 實測） | 0.31 |
| W-90 | 170 s（288 × 0.394 s + 32 × 1.59 s） | 0.37 |
| W-50 | 330 s | 0.59 |
| sentinel × 2 | 103 s | 0.11 |
| **合計** | | **2.75**（依預測，var 與 W 各在第一個 run 就 FAIL：約 1.60；逐點跳過，實作補記六第 6 項）；上限 3.5 |

### V-CLS（後跑；FIFO gateway：V-GW 與 V1）

| 點 | bench 估 | 5 個 run（h） |
|---|---|---|
| GW-U32K | 1,462 s | 2.16 |
| GW-U2K-O512 | 216 s | 0.43 |
| GW-U8K-O512 | 426 s | 0.72 |
| GW-M8K-T（T） | 490 s | 0.83 |
| L-U2K | 156 s（126 s 與 216 s 依平均輸出內插） | 0.35 |
| L-U8K | 370 s（318 s 與 468 s 內插；D-A2 階段 S 實測 U8K-O512 @ 8 為 468 s） | 0.65 |
| sentinel × 2 | 103 s | 0.11 |
| **合計** | | **5.25**（若 GW-M8K-T 在第一個 run 就 FAIL：約 4.6）；上限 7.0 |

### 合計

| 項目 | GPU h（估） | 依預測 | 上限 |
|---|---|---|---|
| 產品 IT-0–6（不是研究） | 0.3 | — | — |
| mini | 0.6 | — | 1.0 |
| V-ADM | 2.75 | 1.60 | 3.5 |
| V-CLS | 5.25 | 4.6–5.25 | 7.0 |
| **研究合計** | **8.6** | **6.8–7.5** | 全期 12 |

- 043 估計約 10–11 h。省下的是：
  - 三次 mini，約 1.5 h；
  - 兩對 sentinel，約 0.2 h；
  - var 的 bench 依實測修正。
- 不合併成一個約 7.7 h 的 session：只省 0.11 h，卻違反「session 之間留空檔」，而且中止時損失更多。

## Harness（`benchmarks/`；凍結模組只 import，一個位元組都不改；不 import capserve）

| 模組 | import（只 import） | 內容 |
|---|---|---|
| `stage_v_plan.py` | `capacity_plan`（block_order、_keys、_expect、SLO_S2、SECONDARY_SLO、SERVER_READY_TIMEOUT_S）、`core_b_plan`（num_prompts、num_warmups、bench_argv、snapshot_dir、SEED、N_GEN）、`stage_d_plan`（WORKLOADS 作為 base spec、server_argv_for、bench_argv_for、tau1、timeout_s、GATEWAY_PORT）、`run_core_a.sha256_json` | POINTS（server / client / τ1 三個 base 分開寫定）、SESSIONS、UNITS；`stage_v_problems`、`build_plan`（formal 需要凍結）、`run_tag`（`V-<point>-c<C>-r<b>`）、`bench_argv`、`gateway_settings`、巢狀 TOML writer（`run_stage_d._toml` 只寫平面鍵）、`harness_sha256`、`gateway_path_files` / `gateway_path_sha256`、`freeze_evidence_problems`、pacing 參數的重算 |
| `stage_v_datasets.py` | `stage_d_datasets.vllm_sampler`、`core_b_plan`、`stage_v_plan` | v1-{2048,8192}-r1..5、v4-{90,50}-r1..5、兩份 mini dataset |
| `stage_v_analysis.py` | `capacity_analysis`（run_eval、classify、window_bounds、sentinel_summary、sentinel_gate、cross_session_shift）、`stage_d_analysis`（derived_problems：用於固定輸出的點；h_verify）、`core_b_analysis.percentile`、`capacity_plan`、`stage_v_plan` | 見下方 |
| `run_stage_v.py` | `run_stage_d`（StageDHarness、server_checks、nondefault_problems、summarize、derived_requests、default_port_free、capserve_sha256、mini_num_prompts）、`run_capacity`、`run_adv`、`gpu_idle`、`coreb_smoke`、`core_b_plan`、`stage_d_plan`、`stage_v_plan`、`stage_v_analysis` | 見下方 |
| `stage_v_report.py` | `capacity_report`（spill_problems、load_uniform、load_advanced_references）、`capacity_analysis`、`run_adv`、`run_stage_d`、`stage_d_plan`、`stage_v_*` | 拒絕條件比照 `stage_d_report`；另檢查 gateway 的 settings 與 argv、`/readyz` 紀錄、對應問題、gateway path sha、2 h 空檔；只寫 `<sid>/analysis.json`，在 Mac 上重跑必須逐位元相同（021 R3） |

**`stage_v_analysis` 的函式**
- `join_access_log`：(h) 的所有規則；
- `run_eval_v1`：`ca.run_eval` 的複本，唯一的差別是 TPOT 只取 out ≥ 2；
- `upstream_view`、`slo_view_w`；
- `in_vllm`（U_W、N1、(j) 的上限）、`manipulation_e`；
- `run_outcome`、`point_class`（含 TRIVIAL）、`decompose`；
- `replay_problems`：順序、逐點的跳過、時數上限；
- `hypotheses`；
- `units`：產生器唯一讀取的段；
- `analyze`。

**`StageVHarness(rd.StageDHarness)` 覆寫的方法**
- 覆寫 `boot`、`gateway_up`、`_row`、`client_argv`、`bench`、`run_point`、`decide`。原因：凍結的 `sd.workload`、`sd.run_tag`、`sda.replay_problems` 遇到 GW-*、L-*、P-*、W-* 會拋出 PlanError。
  - `boot`：server 用 `sd.server_argv_for(pins, base)`，檢查用 `rd.server_checks(…, w=base)`（T 為 512），並記錄 boot 的起始時間；
  - `gateway_up`：只有 provisional 的點才加 `--allow-provisional`，並檢查 `/readyz` 的 tuple；
  - `run_point`：bench → `gateway_down` → 解析 log → derived。
- 狀態機 `_v_machine`：sentinel → block → 逐點的跳過、時數上限 → sentinel。
- 開始前的拒絕：
  - 輸出不在 `results/stage_v/<sid>`；
  - 沒有凍結；
  - gateway path、profile、dataset 的 sha 不符；
  - 和上一個正式 V session 的 `finished_at` 相隔未滿 2 h（讀 `results/stage_v/formal-*/session.json`）；任何 `formal-*` / `smoke-v-*` 目錄沒有已結束的 session.json 時也拒絕（實作補記六第 8 項）。

**client argv**：等於凍結函式的輸出，只允許三種差異，由測試以 diff 斷言。
1. 凍結函式給的是 8000 時，`--base-url` 的 port 改為 8080；
2. 只有 L 點：`--custom-output-len 128` 改為 `-1`；
3. 每個 gateway run 都加上 `--request-id-prefix <tag>-`。vLLM 0.30 的預設值是 `bench-<uuid8>-`，主要 request 本來就帶 `x-request-id`，所以負載不變；只有 C 個 warmup 重送 row 0，而且不帶 id（凍結的 client 不送 ready check：`--ready-check-timeout-sec` 預設 0，實作補記七第 1 項）。

**`configs/stage_v.toml`**
- `schema_version`。
- `[freeze]`：`stage_v`、`mini`、`gateway_path_sha256`。
- `[stage_v]`：
  - seed 20261008、n 5、guard_band 0.05；
  - `stage_d_sha256` = `be7909b1…`，也就是讀取 τ1 與 var pin 的那份凍結 stage_d.toml；
  - sessions；`hour_caps_h = {mini 1.0, V-ADM 3.5, V-CLS 7.0}`、`total_cap_h 12`、`min_gap_h 2`；
  - `run_overhead_s = {B0 95, T 110}`；
  - `nontrivial = {u_w 0.9, n1 0.9}`；
  - `join_tolerance_ms 50`、`gateway_drain_s 5`；
  - `gateway_path_files`：AST 閉包的結果。
- `[stage_v.points.<id>]`：study、server_base、client_base、tau_base、C、profile 與它的 sha256、class(es)、provisional、queue、admission、pacing、dataset、view、unit、bench_est_s。
- `[stage_v.queue]`：product = {2.0, 300}，slow = {120.0, 600}。
- `[stage_v.weighted_v1]`：units {2K 1, 8K 4}、budget 32、caps {32, 8}、上限 1,920 / 8,064、order fifo、max_queue 32。
- `[stage_v.pacing]`：2K {9600, 5920}、8K {32256, 5800}，以及重算規則。
- `[stage_v.datasets]`：22 個 sha256，加上 var-2048 = `358ae8da…`、mixed-8192 = `ab548bed…`。
- `[stage_v.units]`：V1-2K、V1-8K、W1 的點組合。
- `[stage_v.sentinel]`：等於 stage_d 的值。
- `[slo]`、`[secondary_slo]`：等於 stage_d.toml。
- 不寫 pins.toml、capacity.toml、stage_d.toml。

**datasets**（WSL、只用 CPU、`flock -n`；依 005 不進 git；sha256 在 mini 之前 pin）
- 第 i 列的 prompt 由 `RandomDataset(random_seed = 20261008 + i)` 產生，長度為該列的目標長度；
- 重新 tokenize 後，任何一列不是剛好的長度，就不寫出檔案（013）；
- 列數 = N，bench 以 `--disable-shuffle` 讀取；
- row 0 的規則見 V1 / V4；
- mini 的兩份 dataset：
  - `v1-mini-8192`，20 列：row 0 輸出 128，row 5 / 6 / 7 的輸出為 1 / 2 / 512；
  - `v4-mini`，20 列：row 0 為 2K，row 1–8 為 8K，其餘 6 × 2K 加 5 × 8K 以 seed 洗牌。

**逐 request 的擷取**：寫進 `derived/<tag>-requests.jsonl`，所以 mini 的 `rc.mask_smoke` 會一併只保留 idx / cls / out。
- 欄位：
  - `gw_queue_wait_ms`、`gw_prefill_wait_ms`、`gw_ttft_ms`、`gw_upstream_ttft_ms`；
  - `gw_arrival_s`、`gw_sent_s`、`gw_released_s`：以 run 中最早的主要到達為 0；
  - `units`、`out_planned`；
  - W 點的 cls：8,064 為 long。
- gateway log 留在 `raw/<tag>.gateway.log`，它的 sha256 記在 raw_manifest。mini 在 mini_checks 之後刪除這些 log，sha256 仍在 raw_manifest（實作補記六第 10 項）。

**`gateway_path_files`**
- 由 AST 計算 `service/capserve/__main__.py` 的 import 閉包，只讀檔案、不 import，函式內的 import 也算。
- 現況是 15 個檔案：`__init__`、`__main__`、config、keys、profile、tokens、server、app、admission、guard、metrics、openai_api、proxy、retry、upstream，加上新的模組；不含 deploy、itest、loadtest、runs、procs、profile_gen。
- profile.py import 的 capacity_plan、core_b_plan、stage_d_plan 由既有的 frozen 測試鎖住；gateway 的 Python 依賴只靠 env 擷取的 freeze_sha256。

### 測試（Mac）

- **`tests/test_stage_v_plan.py`**：
  - 設定檔等於 044，SLO 等於 stage_d.toml；
  - argv 只差上述三處（diff）；
  - 逾時表：3,240、4,800、4,800、2,820、4,800、4,800、1,260 × 4、1,980 × 2、sentinel 1,020；
  - 每個點的 gateway settings，以及巢狀 TOML 經 tomllib 往返不變；
  - pacing 參數從 committed 的 d5 重算為 5,920 / 5,800；
  - 單位組合；
  - AST 閉包包含 `__init__.py`、不含 deploy；
  - 未凍結時拒絕 formal plan。
- **`test_stage_v_datasets.py`**（stub sampler）：
  - 決定性；
  - 個數剛好：288 / 32、160 / 160；
  - row 0 的規則；
  - 輸出在 [1, 512]；
  - mini dataset 的前 9 列；
  - tokenize 不精確時不寫檔。
- **`test_stage_v_analysis.py`**：
  - 輸入全部 out ≥ 2 時，`run_eval_v1` 等於 `ca.run_eval`；out = 1 只從 TPOT 排除；
  - upstream view 的 TPOT 不變；
  - W 的 SLO 視圖，以及 8K 的合併；
  - 對應失敗的情況：缺行、重複、多行、別的 prefix、非 200、沒有 id 的不是剛好 C 筆（少一筆或多一筆，實作補記七第 1 項）；豁免路徑被忽略；50 ms 的容差；
  - U_W 以合成的區間測試，包括 HOL 造成的空檔與剛好 28.8 的邊界；N1、E、(j)；
  - 類別含 TRIVIAL；
  - 每個單位的判定表，包括否決規則；
  - replay：順序、逐點的跳過、trivial、時數上限依記錄時間重算；
  - hypothesis 的判定對應。
- **`test_run_stage_v.py`**（假的 popen / http / bench）：
  - `--allow-provisional` 只出現在 provisional 的點；
  - `/readyz` 不符時 boot 失敗；
  - port 被佔用時中止；
  - gateway 結束時 run 無效；
  - log 在 `gateway_down` 之後解析；
  - 時數上限、2 h 空檔、sha 不符時拒絕；
  - mini plan。
- **`test_stage_v_report.py`**：
  - 各種拒絕；
  - argv 與 settings 的檢查；
  - 由 committed 的檔案逐位元重現。
- **`test_stage_v_frozen.py`**（凍結時加入）：
  - 5 個模組、stage_v.toml、study profile 的 sha256；
  - 在最後一個 V session commit 之前，目前的 `gateway_path_sha256` 必須等於凍結值。
- **既有測試不改而且通過**：`test_research_isolation`（AST 掃描已涵蓋 `benchmarks/`）、`test_stage_d_frozen`、`test_stage_c_frozen`、`test_core_untouched`、`test_frozen_untouched`。

## 產品程式（凍結前以 TDD 完成，預設全部關閉；在請求路徑閉包內的，由 `gateway_path_sha256` 凍結）

1. **access log 與 `RequestRecord` 的新欄位**：
   - `client_request_id`：client 送來的 `x-request-id`，限 `[A-Za-z0-9._:-]{1,128}`，否則為 null；不回傳給 client，response 的 x-request-id 仍是 gateway 自己的；
   - `t_arrival_s`、`t_sent_s`、`t_released_s`：monotonic；
   - `prefill_wait_s`、`admission_class`、`admission_units`；
   - metric `capserve_prefill_wait_seconds`。
2. **`weighted-v1`**（`admission.py`）：
   - 嚴格 FIFO、隊首阻塞、預算與各 class 上限；取消時不漏 slot；
   - 一律以 local tokenizer 計數；大於 8,064 回 400；
   - `[admission]` 啟用時：
     - Settings 不設 `active_class`；
     - `max_pending` = 預算 + max_queue + pre_admission_slots；
     - `data_client` 的連線池 = 預算 + 4；
     - metrics 的 label 與 `/readyz` 改以政策表示。
3. **pacing**：token bucket 和並發位置在同一個 FIFO 中；burst < class 的 max_prompt 時為 ConfigError；`[prefill_pacing]`。
4. **同時開啟 pacing 與 weighted-v1**：一律 ConfigError（從未量過）。
5. **服務規則**（任何旗標都不能覆寫）：
   - 候選政策只能在兩種情況下啟用：
     - provisional profile 加上 `--allow-provisional`，這是研究用途；
     - verified profile 帶有完全相同的 §12 段，包括佇列設定與 `gateway_path_sha256`。
   - `deploy` 遇到沒有 §12 段的候選政策一律拒絕。
6. **§12 的 loader**：
   - `PREREG_SECTIONS` 加入 `§12`；
   - 新的 bound `validated`；
   - 保留名稱加入 `uniform-2k-o1-512`、`uniform-8k-o1-512`；
   - 只讀 `configs/stage_v.toml`，不 import Stage V 的模組；
   - `not_covered` 的 enum 加入 `output_length_distribution`。
7. **自我雜湊**：以和研究端相同的 AST 閉包演算法計算 `gateway_path_sha256`，並在 `/readyz` 揭露；`service/tests` 斷言兩端在這份 checkout 上的值相同。
8. **`/readyz` 新增**：admission、prefill_pacing、佇列設定、serving status。
9. **`service/profiles/provisional-stage-v.json`**，B0 fingerprint：

   | class | prompt / 輸出 | 並發 / 佇列 |
   |---|---|---|
   | v-uniform-2k | 1,920 / 128 | 32 / 32 |
   | v-uniform-8k | 8,064 / 128 | 8 / 8 |
   | v-uniform-2k-o512 | 1,920 / 512 | 32 / 32 |
   | v-uniform-8k-o512 | 8,064 / 512 | 12 / 12 |
   | v-uniform-8k-o512-c8 | 8,064 / 512 | 8 / 8 |

   - 測試：limits 逐欄等於 c1b 與 `-d-a2-b0` 的 base class；c8 等於要解鎖的值。
- **產生器**（`profile_gen --validation-session`，在閉包之外）：在第一個正式 V session 之前，以合成的 session 用 TDD 完成，見產品對應。

## mini smoke（一次：`smoke-v-<date>`，lock 下，約 0.6 h，上限 1.0 h）

- **設定**：
  - 假的 SLO（`rc.MINI_SLO`）；
  - N = 20（`mini_num_prompts`）；
  - n = 2；
  - sentinel 在 C 2；
  - pacing 與政策都用**正式參數**。

| 點 | C | 驗證什麼 |
|---|---|---|
| GW-U8K-O512 | 4 | provisional 複本（`/readyz` 顯示 class_limit 12，r1 的 boot 照樣在 bench 之前檢查）；r1 強制 5 s 逾時，r2 依跳過規則不再測（走過跳過規則，實作補記六第 7 項） |
| L-U8K | 4 | 逐列的輸出長度（含 1 / 2 / 512）、out = 1 從 TPOT 排除、`v-uniform-8k-o512-c8`；兩個 run 都有效，所以輸出 512 與 8,064 token 的邊界也由它驗證 |
| GW-M8K-T | 2 | T server（log 中有 512）、不加旗標的 verified T profile、32,640 token 的 local 計數 |
| P-U2K | 8 | 正式的 burst 9,600：第一波 8 個中有 3 個必須等待（E 的門檻 ≥ 2）；N1 |
| F-U2K-var | 4 | pacing 關閉、var-2048、沒有 prefill_wait |
| W-mini | 9 | 正式的預算 32、上限 32 / 8；第一波為 1 個 2K 加 8 個 8K，共 33 單位 > 32，必定有一個排隊；單位、U_W、(j) |

- 在最後一次計畫的 V boot 之前強制觸發時數上限的停止；結尾 sentinel 照跑。
- 跳過是逐點的：同一單位的夥伴點照常測，由假的 session 測試涵蓋（實作補記六第 6 項）。
- **通過條件**：
  - replay 乾淨；
  - 跳過規則走過：強制逾時的點在之後的 block 被跳過（`skip_walked`）；
  - 除了刻意的逾時與上限停止，所有 run 有效；
  - 對應是雙射，`/readyz` 的 tuple 正確；
  - output_lens 等於 pin 的值；
  - E 有作用，至少一個 queue_wait > 0；
  - U_W 與 N1 算得出來；
  - teardown 乾淨；
  - session.json 記錄 `harness_sha256`、`gateway_path_sha256`，以及每個點的 profile 與每個 dataset 的 sha256（凍結時必須等於 pin 值，實作補記六第 12 項）；
  - 延遲數值依 `rc.mask_smoke` 丟棄，`raw/*.gateway.log` 也刪除（sha256 留在 raw_manifest）。

## 凍結（比照 039）

1. 一起 commit 並 push：
   - PREREGISTRATION §12、044；
   - `configs/stage_v.toml`、5 個模組與測試；
   - 已合併的產品程式與 study profile；
   - `results/stage_v/prereg-sim/`。
2. 兩輪獨立審查都在 GPU 之前完成；datasets 產生並 pin。
3. GPU 視窗 0：產品 IT 與 mini，commit mini。
4. 設定 `[freeze] stage_v = true`、mini id、`gateway_path_sha256`，加入 `test_stage_v_frozen`，push，部署凍結 commit。
5. 之後才收集正式數據。

- 凍結之後，任何 harness 或請求路徑的修改，都要新的 decision、新的 mini 與新的 session。
- 從凍結到 V-CLS 結束，請求路徑閉包內的檔案在 main 上不能改。deploy、itest、loadtest、profile_gen 不受影響。

## 產品對應（機械式；正式釘選 c1b 與 `rtx-verified.toml` 不會因任何結果自動改變，037 §2）

- 每個單位只在它列出的每一點都 PASS 時成立。FAIL、MARGINAL、NT、TRIVIAL 都代表不可用：沒有退回值，看到數據之後也不調參數。

| 單位 | 需要 PASS 的點 | 產物 | 產品上的效果 | 仍不涵蓋 |
|---|---|---|---|---|
| VG-32K | GW-U32K | 沒有新檔；`…-d-a1b-b0` 的位元組不變（`not_covered` 仍列 `via_gateway`，保守） | uniform-32k（2）成為正式釘選的候選；由使用者決定是否改釘選，改了之後跑 IT-0–6 | prompt 長度不一、開放迴路；第一波 12.8 s |
| VG-M8K-T | GW-M8K-T | 沒有新檔 | mixed-8k-t（4）成為候選；另需 043 §6 的 T 部署 1–6 與 V5 | WINDOW_SENSITIVE；U8K-T 永遠只作報告 |
| V1-2K | L-U2K 與 GW-U2K-O512 | `verified-<v-cls sid>-b0.json` 的 `uniform-2k-o1-512`（1,920 / 512，32，bound `validated`） | 輸出 ≤ 512 的 2K class 可以服務，是釘選的候選 | prompt 長度不一、其他輸出分布、ABOVE_CKV（32 > 30，附 3.125% 停頓的揭露） |
| V1-8K | L-U8K 與 GW-U8K-O512 | 同一份 profile 的 `uniform-8k-o1-512`（8,064 / 512，8，`validated`） | 8K 長輸出可以服務；取代 043 的包絡規則 | 同上，沒有 C_KV 的問題 |
| PACE-2K | P-U2K（含 E、N1），而且沒有被 var 否決 | `verified-<v-adm sid>-b0.json`：c1b uniform-2k（32）的複本，加上 `pacing` {9,600, 5,920} 與佇列設定 {120 s, 600 s} | 只在這份 profile、相同參數與相同佇列設定、相同 `gateway_path_sha256` 下可以開啟；預設關閉 | 快速 429、開放迴路；不得宣稱第一波 TTFT 改善 |
| PACE-8K | P-U8K | 同一份 profile：uniform-8k（8）的複本，加上 `pacing` {32,256, 5,800} | 同上 | 同上 |
| W1 | W-90 與 W-50（每個 run 的 U_W ≥ 28.8，單位正確） | 同一份 profile 的 `admission` 段：weighted-v1 的全部參數、佇列設定、`measured_mixes` [0.9, 0.5]、`gateway_path_sha256` | `[admission] policy = "weighted-v1"` 只接受參數完全相同的段；pacing 必須關閉 | 其他 mix（只是內插）、32K、T、開放迴路、從到達起算的 TTFT |
| （描述） | var 那一對 | 只改 README | 依 V2 的 2×2 讀法；不得宣稱 pacing 修好 var | — |

- **產生器**：`python -m capserve.profile_gen --validation-session results/stage_v/<sid> --freeze-commit <stage_v 的凍結 commit> --out-dir service/profiles`。
  - 只讀 analysis.json 的 `units`；
  - 以 `stage_v_report.report()` 唯讀重跑，結果必須逐位元相同；
  - 每個 session 最多一份 `verified-<sid>-b0.json`，只含成立的單位；
  - 下列情況拒絕：凍結檢查不符、部署 commit 的 `gateway_path_sha256` 和凍結值不同、base profile 不是 committed 的位元組、工作樹不乾淨。
- **§12 的 provenance**：
  - validation_session、analysis 的 path 與 sha；
  - stage_v_sha256、gateway_path_sha256、capserve_sha256；
  - base_profiles [{id, content_sha256}]；
  - freeze / generator / deployed commit、pins、gpu、driver。
- **§12 的 class**：
  - 複本：逐欄等於 base class，再加上 `validation` {study, points, class, worst_R client / upstream, n}；
  - V1 的 class：evidence 記錄 {dist "uniform[1,512]"、5 個 dataset 的 sha、via_gateway true、兩點的 worst_R}；
  - `not_covered` 依 enum 重算：移除 `via_gateway` 與 `output_length_other_than_measured`，加入 `output_length_distribution`。

## 揭露

- 設計者看過 Core、Advanced、Stage C、Stage D 的全部結果。預測用到：
  - 042 的 R；
  - D-C 的 gateway 位移：2K 在 W 為 +214 ms、W′ 為 +390 ms，8K 約 +720 / +710 ms；
  - D5 的斜率。
- pacing 的參數是 D5 的事後內插（043 已揭露），本文依取整規則凍結。weighted-v1 的參數照 043。沒有任何參數依 V 的數據選擇。
- U_W 的模擬只用已 commit 的數據，只用來檢查門檻是否達得到，不預測 SLO。
- O512、V1、V2、V4 的點量的是 provisional 的 `v-` 複本：准入的程式路徑與 limits 相同，只有標籤不同，`/readyz` 與 header 都標示 provisional。
- 研究用的 gateway 是 `capserve serve`（port 8080，研究的 vLLM 在 8000），不是 `deploy up`。deploy 的監督與冷啟動不在量測範圍內。
- 只量 `/v1/completions`、`ignore_eos`。chat route、自然 EOS 都沒有量。
- 直連的 sentinel 仍有 042 推論的極小 keep-alive 競態，只在 warmup 轉入第一波時暴露。發生時 run 無效，不補跑。
- 本機的 vLLM 原始碼是 scratch 中的副本（`ready_checker.py` 不在其中）。凍結前在 WSL 的 venv 唯讀複核：
  - `CustomDataset.sample` 的 `output_tokens`；
  - warmup 重送 row 0；
  - ready check 預設略過（`--ready-check-timeout-sec` 預設 0，`benchmark()` 印出 `Skipping endpoint ready check.`；實作補記七第 1 項）；
  - `x-request-id`；
  - 前後各兩次 GET /metrics（spec decode 與 diffusion 的 metrics）；
  - aiohttp 的 limit = C 與 keepalive 60 s；
  - `main_async` 只在 dataset 為 random / prefix_repetition 時呼叫 `_align_prompts_to_server_tokenizer`，它對 `{base_url}/tokenize` 只送一次 POST，失敗（404）就略過對齊（實作補記六第 5 項；scratch 副本 `serve.py` 的 sha256 為 `ec255c3b…eec142`）。
  - 並在本文記錄這些檔案的 sha256。

## 風險與限制

- 凍結前的程式量大：043 的第 4–10 步加上 044 的產品程式，任何一項延誤都會讓 V 延後。
- 預測接近 guard band 的點：GW-U2K-O512（0.82–0.90，KV 斷崖）、GW-M8K-T（約 1.0）、P-U2K（TPOT 餘裕 0.027）。約 1.5% 的跨 session 漂移就可能翻轉判定，而且不能重跑來換 PASS。
- V1 極短輸出的 TPOT 只由少數幾個 step 決定，可能決定 P95。這是事先登記的定義，事後不能改。
- 固定序列加上幾乎決定性的封閉迴路：V-GW 與 var 的 n 主要量到重現性；V1 / V4 也只涵蓋 5 條抽樣的序列。
- 點越多，偶發 FAIL 的機會越大。這是保守的方向，不做校正。
- 和 D-C、D-A 的所有比較都跨 session、跨 gateway 版本，只作描述。
- raw/ 與 gateway log 不取回 Mac。derived 中的 gateway 欄位只能信任 runner 的解析與 sha256。
- GPU 是共用的。V 排在 E1-P、E1-D、Stage Q pilot 之後，加上 2 h 的空檔，總時程可能拉長到數天。
- 042 揭露的連續 23.6 h，是本專案 Stage D 的四個 session 讓其他專案拿不到 lock；所以本研究分成獨立的 lock 視窗。

## 設計審查改正的項目

- **043 分支的進度**：`616e066` 已含 043 第 1–3 步（profile.py +287 行、app、`__main__`、deploy），不是「只改了 loader 的測試」。
- **不需要修改 043 的拒絕規則**：limits 相同的 provisional 複本可以用既有的 `--allow-provisional` 服務 O512 與 V1 的 class。
- **L-U8K 的上限**：原設計經 d-a2-b0 服務時，gateway 的上限是 12，不是要解鎖的 8。改用 c8 複本。
- **政策 mini 一定會受限**：
  - 「client 12 一定受限」不成立：90/10 約 16 單位，50/50 約 30 單位，都低於 32；
  - 「client 9、75% 8K」的期望約 29 單位，也不保證。
  - 改為前 9 列決定性的 33 單位。
- **(h) 的計數**：
  - 「沒有 id 的行數 = 1 + warmup」沒有限定 route，會被 GET /metrics 與 /readyz 誤判，改為只計 completion route（個數見下一項；實作補記七第 1 項：沒有 ready check，剛好 C）；
  - 「剛好 C + 1」依賴 ready check 只送一次，改為 ≥ C + 1。（實作補記七第 1 項更正：凍結的 client 根本不送 ready check，`--ready-check-timeout-sec` 預設 0，改為剛好 C。）
- **在途區間的終點**：用 guard 的 ts 當終點，會因 permit 早於 `_finish` 釋放而產生 C + 1 的重疊，改用 `t_released_s`。ts 也只是 1 ms 精度的 wall time。
- **時數估算**：
  - 「每次 boot 1.6 min，gateway 點另加 0.5 min」重複計算，D-C 的 94 s 已含 10 次 gateway boot；
  - T 的開銷不是每個 run 139 s：每次 boot 平均有 2.85 個 run，T 的 server 啟動與 B0 同為 32 s；
  - U8K-O512 @ 8 有實測值 468 s；var 經過 gateway 實測 75 s。
- **pacing 的 rate**：依「⌊prompt ÷ slope⌋」，2K 是 5,921，不是 5,922。改為「取到 10 tok/s」，結果為 5,920 / 5,800，由測試從 d5 重算。
- **V2 的判定 view**：以 client 判定和 043 7.4 第 1 項的字面不同，原設計沒有列為差異。本文改為兩種 view 取較差者，並記錄（差異第 5 項）。
- **argv 的 base**：L 點若從 U2K-O512 產生，凍結函式不會有 custom 的旗標。本文把 server、client、τ1 的 base 分開寫定。
- **mini 的延遲遮蔽**：獨立的 `derived/<tag>-gateway.jsonl` 不在 `rc.mask_smoke` 的範圍內，改為併入 `*-requests.jsonl`。
- **gateway 設定**：drain 60 s 和 D-C 的 5 s 不同，改用 5 s。
- **狀態機**：`run_stage_d._verify_machine` 沒有時數上限，跳過也只看 `rows[0]["outcome"]`，所以另寫 `_v_machine`。
- **凍結範圍**：
  - 整個 `service/capserve/*.py` 會擋住 deploy、itest、profile_gen 的開發；
  - 手選的子集漏了 `__init__.py`；
  - 改為 AST 閉包。
- **產品 IT**：原設計漏列 043 第 8 步要求的 RTX IT-0–6，已排進 GPU 視窗 0。
- **23.6 h 的方向**：「另一個專案曾連續持有 lock 23.6 h」說反了；那是本專案 Stage D 連續占用了 lock。
- **8K 的判定**：50/50 的 8K 逐 run 判定偏離 030 補記二第 1 點與 043，維持跨 run 合併。

## 放棄的替代方案

- **四個研究各自一套 harness、mini 與凍結**（043 字面）：多約 1.5 h，而且任何修正都讓所有 mini 失效。
- **五個以上的正式 session**：鎖視窗越多，日曆時間越長，每個 session 還要多一對 sentinel。
- **V1 直連**：O512 的單位會由「gateway 下剛好 512」與「直連下輸出不一」拼成，組合沒有量過。
- **新的 `serve --validation` 模式，或讓 verified 檔以 provisional 身分服務**：比複本更大的產品表面，也要改 043 的規則。
- **V2 只看 upstream**：依構造通過。
- **V2 的 var 點當閘門**：依預測 pacing 永遠不會上線，而且 var 的 FAIL 和 pacing 無關。
- **V4 以 request 數計的非平凡條件**：會被在途組成的偏移誤判。
- **由既有欄位推算在途區間**（ts − duration_s + ttft_s − upstream_ttft_s）：終點晚於釋放，精度只有 1 ms。
- **超過時數上限就中止並作廢整個 session**：連已有效的點也一起丟掉。
- **從 `rtx-verified.toml` 渲染研究用的 gateway 設定**：和產品檔耦合，drain 也不同。
- **刪去 GW-U8K-O512 @ 12**：使用者已核准；本文讓它成為 V1-8K 的最壞情況證據。
- **V1 / V4 每點只用一條序列**：n 只量到時間上的變異。
- **快速 429 的佇列設定，或 pacing 以產品的 2 s 上線**：前者在封閉迴路下 run 無效，後者宣稱超出量測範圍。
- **數據之後才寫產生器**：改為在第一個正式 session 之前完成。

## 待決事項的處理（2026-10-24）

草稿列了 9 項。第 9 項（和 Stage Q / E 的先後）已由使用者在 043 決定。其餘 8 項都在使用者核准的 V2 / V4 範圍內，屬於協定或工程的細節，
不增加研究問題、不改正式釘選，GPU 時數的增加在 0.31 h 以內，所以依建議定案；使用者之後可以改，但必須在凍結之前。

1. **候選政策的研究用途**：pacing 與 weighted-v1 可以在 provisional profile 上以 `capserve serve --allow-provisional` 啟用；
   verified profile 一律要 §12 段；`deploy` 一律拒絕沒有 §12 段的候選政策。（043 7.3 的字面因此修改；不這樣做 V2 / V4 無法執行。）
2. **access log 新增欄位**：`client_request_id`（client 提供、經過清理、不回傳）與 monotonic 的到達、送出、釋放時間。
3. **pacing 只在「慢慢服務」的佇列設定下驗證與上線**（120 s / 600 s）；上線時必須用相同的佇列設定。快速 429 在封閉迴路下量不到。
4. **gateway 請求路徑的 sha 不同時**：pacing / admission 段拒絕服務（它們就是被操弄的變因）；V1 的 class 與 V-GW 的候選資格只在 `/readyz` 揭露 `gateway_validated: false`。
5. **凍結期間的產品開發**：從凍結到 V-CLS 結束，請求路徑閉包內的變更改在 branch 上進行，V-CLS 之後再合併；閉包外的 deploy、itest、profile_gen 照常開發。
6. **V-ADM 加入 F-U2K @ 32**（新版 gateway、FIFO、固定長度，約 +0.31 h）：P-U2K 在同一 session 內的對照，也重新驗證正式 class 在要上線的 gateway 版本下仍成立。V-ADM 約 2.75 h，上限 3.5 h 不變。
7. **V4 照 043 的 weighted-v1 執行**，雖然預測兩點都不支持：依跳過規則代價約 0.2 h；凍結前改設計是沒有證據的新設計。
8. **session 的切分**：兩個正式 session（V-ADM 約 2.75 h、V-CLS 約 5.25 h），都在使用者「一次約 7–9 h」之內；之間 ≥ 2 h，各自取得與釋放 lock。

---

## 附錄：PREREGISTRATION.md §12 的文字

```markdown
## 12. Stage V addendum：Stage D 產品化的驗證研究（decision 043、044）

**狀態：DRAFT**。harness、044 的產品程式、一次 mini smoke 與兩輪審查完成後，連同 `configs/stage_v.toml` 一起凍結
（`freeze.stage_v = true`、`tests/test_stage_v_frozen.py`）。凍結前沒有任何 Stage V 數據。
本節只摘要；完整規則、揭露與依據見 decision 044。第 1–11 節不變，Phase 1–3 既有的判定不受影響。

- **沿用 §8 / §9（030、039 D-C）**：
  - 固定點，n = 5，每個 run 一次 boot，block 以 `capacity_plan.block_order` 洗牌；
  - 030 8.1 的單點類別，guard band 0.05，取最差的 run；
  - W = [C, N − C)、N = max(200, 10C)；跳過規則；無效的 run 不補跑；
  - 有效性含 031，sentinel 閘門 3%、replay；
  - gateway 以子 process 啟動，研究程式不 import capserve；凍結的 Stage C / D 模組只 import。
- **SLO**：S2。
- **V-GW**：經過 gateway 在 C_PROD。點為 U32K @ 2、U2K-O512 @ 32、U8K-O512 @ 12、M8K-T @ 4（T）；判定用 client 端的 TTFT。
- **V1**：輸出在 [1, 512] 均勻（每點 5 條 pin 住的序列），U2K @ 32、U8K @ 8，經過 gateway；TPOT 只算 out ≥ 2。
- **V2**：prefill pacing（2K 9,600 / 5,920 tok/s，8K 32,256 / 5,800 tok/s），經過 gateway。
  - 點為 U2K @ 32、U8K @ 8、var-2048 @ 32 的開，以及 U2K @ 32 與 var-2048 @ 32 的關（對照組）；
  - 判定取 upstream 與 client 兩個類別中較差者；
  - manipulation check（第一波的 prefill_wait）不成立時 run 無效；
  - 非平凡條件 N1：vLLM 內平均 ≥ 0.9 C；
  - var 那一對不作閘門，只作否決（pacing 讓通過的 var 不再通過時否決）。
- **V4**：`weighted-v1`（單位 1 / 4、預算 32、上限 32 / 8、嚴格 FIFO），pacing 關閉，client 32。
  - 2K / 8K 的 90/10 與 50/50，每個 mix 5 條序列；
  - 判定用 upstream 的 TTFT：2K 每個 run 對 2 s，8K 跨 run 合併對 6 s，TPOT 對 100 ms；
  - 非平凡條件：T_W 中 vLLM 內平均單位數 U_W ≥ 28.8，否則為 TRIVIAL（not supported）。
- **逐 request 擷取**：gateway 的 access log 以 `client_request_id` 雙射對應 bench 的每個 request，記錄 queue_wait、
  prefill_wait、ttft、upstream_ttft，以及 monotonic 的到達、送出、釋放時間。
- **gateway 是被操弄的變因**：`capserve serve` 的 import 閉包（AST 計算）以 `gateway_path_sha256` 凍結；
  研究用的佇列設定（V2 / V4 為 120 s / 600 s）記錄在設定檔，和產品設定分開。
- **Hypotheses**（預測寫在數據之前）：
  - H-GW1–4：GW-M8K-T 預測 MARGINAL 或 FAIL（not supported），其他 supported；
  - H-L1–2：supported；
  - H-P1–3：var-pace 預測 not supported；
  - H-W0–2：W-90 / W-50 預測 not supported。
  - 預測以點的類別記錄（supported 為 PASS，not supported 為 FAIL 或 TRIVIAL），是否如預測由類別機械判定，NT 不判定。
- **session**：mini 一次；`formal-*-v-adm`（V2 + V4，約 2.75 h，上限 3.5 h）、`formal-*-v-cls`（V-GW + V1，約 5.3 h，
  上限 7.0 h），之間 ≥ 2 h；全期上限 12 h。
  - 超過 session 的上限時不再開始新的 boot，剩下的點為 NT；
  - 跳過規則逐點：一個點有一次 run 不是 ok（無效、FAIL 或 trivial）之後，只有該點不再測；同一產品單位的其他點照常測，
    單位仍要每一點都 PASS 才成立。
- **產品對應**：每個單位只在它列出的每一點都 PASS 時成立。
  - 產生器（`profile_gen --validation-session`）只為成立的單位寫出 §12 verified profile：V1 的 `validated` class、
    `pacing` 段、`admission` 段；
  - V-GW 的通過只讓 uniform-32k / mixed-8k-t 成為正式釘選的候選；
  - 正式釘選不會自動改變（037 §2）。
- **輸出**：`results/stage_v/<session>/analysis.json`。
```

## 實作補記（2026-10-24，產品程式第 1–9 項；沒有任何 Stage V 數據）

產品程式第 1–9 項已在 `feat/044-stage-v` 以 TDD 實作（`service/capserve/`、`service/tests/`）。下列是本文沒有寫死、實作時決定的細節，以及本文和程式不一致、依程式處理的地方（依 CLAUDE.md 先記錄）。候選政策在設定沒有 `[admission]` / `[prefill_pacing]` 時完全不啟用：准入仍是原本的 `AdmissionController`，正式部署的行為不變；c1b、三份 Stage D profile、`rtx-verified.toml` 的位元組都不變，c1b 與 Stage D 的 profile 仍逐位元重新產生。

1. **access log 的欄位名**：`RequestRecord` 既有的 `arrival`、`upstream_sent` 就是 monotonic 的到達與送出時間，access log 以 `t_arrival_s`、`t_sent_s` 寫出；記錄內的名稱不變，以免改動 proxy 與既有測試。`client_request_id`、`t_released_s`、`prefill_wait_s`、`admission_class`、`admission_units` 是新欄位，未設定時不寫出：預設設定下只多 `client_request_id`（client 有送時）與三個時間；`prefill_wait_s` 只在 pacing 開啟時、`admission_*` 只在 weighted-v1 時出現。`capserve_prefill_wait_seconds` 一律註冊，只有 pacing 開啟時才有觀測值。`ttft_s`、`upstream_ttft_s` 仍只在串流的 request 記錄（既有行為），(h) 依此讀取。
2. **`t_released_s` 的記錄點**：`Permit.release` 先讀 controller 的時鐘、呼叫 `on_release`，再把 slot 交給下一個 waiter；測試在回呼中確認 waiter 當時仍在排隊。
3. **queue_wait_s 與 prefill_wait_s 互斥**：pacing 開啟時 `queue_wait_s` = 總等待 − `prefill_wait_s`，所以「逐 request 的分解」中 gw ttft_s − upstream_ttft_s − queue_wait_s − prefill_wait_s 不會重複扣除。`prefill_wait_s` 是在隊首、有空的並發位置、只缺 token 的時間（0.0 代表沒有等 token）；token 在准入時扣除，之後即使取消也不退還（保守的方向）。
4. **pacing 也一律計數**（本文只對 weighted-v1 寫「一律計數，不走 `fast_fits`」）：bucket 以 prompt 的 token 數扣除，byte 上界會多算，2K 的 RandomDataset prompt 的上界可能超過 burst 9,600 而永遠進不去，所以 pacing 也一律以 local tokenizer 計數。任一候選政策都要求 `tokenizer = "local"`，否則 ConfigError。
5. **設定的格式**：
   - `[admission]` = {policy "weighted-v1", budget, max_queue, classes [{name, units, max_concurrency}]}。設定時不設 `active_class`；兩者必須恰好設一個。
   - `[prefill_pacing]` = {burst_tokens, rate_tokens_per_s}，都是整數（本文的 rate 取到 10 tok/s）。
   - 參數必須符合所命名的 class，否則 ConfigError：cap 等於 class 的 max_concurrency；units × cap = budget（043 7.3 A：單一 class 恰好在它的 C_PROD 填滿預算）；max_prompt_tokens 嚴格遞增（prompt 分到第一個容得下的 class）；max_output_tokens 相同；burst ≥ class 的 max_prompt_tokens。
   - 本文 `[stage_v.weighted_v1]` 的「order fifo」與「上限 1,920 / 8,064」不是設定值：順序固定是嚴格 FIFO，上限取自 profile 的 class。
6. **weighted-v1 的揭露**：沒有 active class，所以 `/readyz` 的 active_class、class_status、bound、flags、not_covered 為 null。
   - `limits` 是政策的包絡：max_concurrency 為最大的 cap（units × cap = budget 時，在途的 request 數不會超過它）、max_queue 為政策的 max_queue、max_prompt_tokens 為最大的 class、max_output_tokens 為共同值；max_pending = budget + max_queue + pre_admission_slots，上游連線池 = budget + 4。
   - `admission` 列出參數與各 class 的 token 上限，`admission_state` 是即時的單位數與各 class 的在途數。
   - `capserve_profile_info` 的 active_class label 為 `weighted-v1:<class>+<class>`，flags 為空字串；concurrency / queue limit 的 gauge 為包絡值；Retry-After 的估計也用包絡的 max_concurrency。
7. **`/readyz` 的新欄位**：`serving_status`（profile 的 verified / provisional）、`queue` {max_queue_wait_s, request_deadline_s}、`admission`、`admission_state`、`prefill_pacing`、`gateway_path_sha256`、`gateway_validated`。
   - `gateway_validated` 只對 §12 profile 有值（執行中的 gateway_path_sha256 是否等於 provenance 的值），其他 profile 為 null。
   - 和待決事項第 4 項的差異：V-GW 通過的 class 不產生新檔（`…-d-a1b-b0` 的位元組不變），profile 中沒有可比對的 sha，所以「V-GW 的候選資格揭露 `gateway_validated: false`」目前無法由 `/readyz` 表示。使用者決定改釘選時另行處理（README 或新的 decision）。
8. **服務規則只有一個入口**：`capserve.config.served_classes`，由 `capserve serve`（exit 2）、`create_app`（ProfileError / ConfigError）與 `deploy up`（preflight 拒絕）共用。
   - provisional profile 只有在 `--allow-provisional`（或 `CAPSERVE_ALLOW_PROVISIONAL=1`）下才會載入，所以「provisional 加上旗標」在這裡就是「provisional」。deploy 以 `study=False` 呼叫，provisional 一律拒絕。
   - §12 profile：pacing 時，active class 的 `pacing`（參數與 queue）必須完全相同；weighted-v1 時，profile 的 `admission` 段（policy、budget、max_queue、classes 與 queue）必須完全相同；兩者都要求執行中的 gateway_path_sha256 等於 provenance 的值（weighted-v1 另比對段內的值）。
   - deploy 遇到相同的 §12 段時照常啟動：`gateway.toml` 寫出這些表（`toml_dumps` 支援 inline table 的陣列），`/readyz` 必須回報相同的政策、參數與 queue 設定。deploy 以 gateway 實際執行的 release 計算 gateway_path_sha256。
9. **§12 profile 的 schema**（loader 依此檢查，第 8 步的產生器必須照此寫出）：
   - **provenance**：沿用既有欄位，`source_session` 就是本文的 validation_session，analysis 的 path 與 sha 也是既有欄位；新增 `stage_v_sha256`、`gateway_path_sha256`、`capserve_sha256`、`base_profiles` [{id, content_sha256}]；`stage_d_sha256` 必填。
   - **複本**：c1b 的 uniform-2k / uniform-8k，§8 的 class 與 evidence 原樣。可帶 `pacing` {burst_tokens, rate_tokens_per_s, queue}，此時必須帶 PACE-2K / PACE-8K 的 `validation`；沒有 pacing 的複本只能作為 admission 段的成員，不帶 validation。
   - **V1 class**：名稱 uniform-2k-o1-512 / uniform-8k-o1-512，token 上限取 D-A 的 U2K-O512 / U8K-O512，bound `validated`，max_concurrency = max_queue = V1 的點的 C，2K 為 32、8K 為 8（`V1_C`，loader 逐值檢查，不是 O512 的 C_PROD；實作補記六第 2 項），source `stage_v:V1-2K` / `stage_v:V1-8K`。evidence 是 O512 class 的 evidence（server B0、output_tokens 512），not_covered 為 `v1_not_covered`，另加 `output_distribution` "uniform[1,512]"、`datasets_sha256`（5 個不同的 sha）、`via_gateway` true。`serving_class` 的 043 §4 拒絕不適用於 §12 的 validated class；同一個 bound 出現在 §8 / §9 時，loader 拒絕，只讀 model 時 043 §4 仍拒絕。
   - **validation** = {study, unit, points, measured_classes, worst_r {point: {client, upstream}}, n}。V1 class 的 points 必須等於該單位的點，依 `[stage_v.units]` 的順序（`V1_POINTS`；實作補記六第 2 項）。本文寫的「class」改名為 `measured_classes`：`class` 是 Python 的保留字，而且 V1-8K 的兩點服務不同的 class。本文放在 V1 evidence 的「兩點的 worst_R」也放在 validation 的 worst_r。
   - **admission 段**：weighted-v1 的參數、queue、measured_mixes（每個都在 (0, 1) 內）、gateway_path_sha256（必須等於 provenance 的值）與 V4 的 validation。
   - **只讀 stage_v.toml 的三個鍵**：`[freeze] stage_v`、`[freeze] gateway_path_sha256`、`[slo]`，另比對整個檔案的 sha256。pacing 與 admission 的數值，和 §8 / §9 的上限一樣，信任來源是「committed 的檔案加上 Mac 上的重新產生測試」，loader 不和 `[stage_v.pacing]` / `[stage_v.weighted_v1]` 比對，避免和還沒寫的 harness 鍵名耦合。重新產生的測試在產生器那一步加入。
   - 這些欄位在 §8、§9 與 provisional profile 中一律為 null，所以候選政策不能手動加到 c1b 或 Stage D 的 profile 上。
   - 沒有 `configs/stage_v.toml`（目前的 checkout）時，每一份 §12 profile 都被拒絕（fail closed）。
10. **P-U8K 的 max_queue**：V2 寫「max_queue 32，所有點相同」，但 max_queue 是 profile 的 class 上限，`v-uniform-8k` 依第 9 項是 8 / 8。依程式，P-U8K 的佇列上限是 8；封閉迴路 C 8 時，在途加上排隊的 request 不會超過 8，不影響量測。harness 的 `stage_v.toml` 應記錄 8。
11. **gateway_path_sha256**：新模組 `service/capserve/gateway_path.py` 本身也在閉包內，所以閉包現在是 16 個檔案（本文列的 15 個加上 gateway_path）。
    - 雜湊的算法同 `run_stage_d.capserve_sha256`：依路徑排序，每個檔案依序為 path、NUL、bytes、NUL。
    - 研究端（`stage_v_plan`）必須以同一算法自行實作，不 import capserve。`service/tests/test_gateway_path.py` 中比對兩端的測試，在 `stage_v_plan` 存在之前會 skip。
    - 另有測試鎖定 capserve 是平坦的 package（沒有子 package，閉包只追一層）。
12. **study profile**：`service/profiles/provisional-stage-v.json`，B0 的 fingerprint 與 KV 幾何。五個 v- class 的 limits 由已 commit 的 c1b / `-d-a2-b0` class 逐欄複製；c8 為 uniform-8k-o512 的 token 上限，加上 c1b uniform-8k 的 8 / 8。測試從 committed 的檔案逐欄比對。

## 實作補記二（2026-10-24，harness：`stage_v_plan` 與 `configs/stage_v.toml`；沒有任何 Stage V 數據）

`benchmarks/stage_v_plan.py`、`configs/stage_v.toml`（`[freeze] stage_v = false`、`mini = ""`、`gateway_path_sha256 = ""`）與 `tests/test_stage_v_plan.py` 已依本文以 TDD 實作。凍結模組只 import（`capacity_plan`、`core_b_plan`、`stage_d_plan`、`run_core_a.sha256_json`），一個位元組都沒有改，也不 import capserve。產品端另有 `service/tests/test_stage_v_settings.py`：研究端渲染的每個點的 gateway 設定，capserve 都能載入，並在該點的 profile 上服務，`/readyz` 的 limits、佇列設定與候選政策都等於計畫值；deploy（`study=False`）拒絕其中的候選政策。`service/tests/test_gateway_path.py` 中比對兩端 AST 閉包的測試不再 skip，兩端在這份 checkout 上相同（16 個檔案）。下列是本文沒有寫死、實作時決定的細節，以及和本文字面不同、依程式處理的地方。

1. **`configs/stage_v.toml` 在凍結前就存在**：本文「凍結」第 1 步本來就要求它和 harness 一起 commit，只是 `stage_v = false`。所以產品補記第 9 項的「沒有 `configs/stage_v.toml`（目前的 checkout）」不再成立。§12 的 loader 仍然 fail closed：`[freeze] stage_v` 不是 true，每一份 §12 profile 都被拒絕，正式部署的行為不變。產品測試 `test_the_frozen_configs_default_to_the_repo_files` 改為讀取這份檔案，並斷言 `stage_v = false`。
2. **點的欄位**：
   - 本文列出的欄位之外，另加三個：`per_block`（block r 讀 `<dataset>-r<r>`）、`out_pinned`（只有 L 點，`--custom-output-len -1`）、`limits`（`/readyz` 必須回報的 max_concurrency、max_queue、max_prompt_tokens、max_output_tokens；weighted-v1 為產品補記第 6 項的包絡）。
   - 「class(es)」寫成清單 `classes`，weighted-v1 依准入順序列出兩個 class。
   - `view` 的值為 `client`、`upstream`、`worse`（V2：兩種 view 取較差者）。
   - profile 的 sha256 寫在設定檔，由 `profile_problems` 對 checkout 比對，不寫死在模組中。
3. **P-U8K 的 max_queue 記為 8**（產品補記第 10 項）：`limits` 記錄 class 的 8；W 點記錄政策的 32。
4. **單位與跳過**：
   - `[stage_v.units]` 列出產品對應的全部 7 個單位，單點的單位也列出，供 analysis 的 `units` 段使用。本文設定檔一節只寫了 V1-2K、V1-8K、W1 三個多點單位。
   - 跳過逐點：每個點只跳過自己（實作補記六第 6 項；這裡原本寫「跳過的群組是該點所屬單位的所有點」）。P-U2K-var、F-U2K-var、F-U2K 與 mini 的 W-mini 不屬於任何單位。
   - var 那一對不放進 PACE-2K：單位要每一點都 PASS，依預測 var 會 FAIL，PACE-2K 就永遠不會成立；否決規則也無法成立。
   - 否決規則寫在模組（`VETOES`），不在設定檔。
5. **`[stage_v.weighted_v1]` 就是 gateway 的 `[admission]` 表**（policy、budget、max_queue、classes [{name, units, max_concurrency}]）。
   - 依產品補記第 5 項，不寫「order fifo」與「上限 1,920 / 8,064」。
   - 測試改從 study profile 確認兩個 class 為 1,920 / 8,064 token、輸出 128。
6. **pacing 的重算規則寫進設定檔**：`source`、`round_to = 10`，以及每個 tier 的 workload、prompt、c_kv、burst_requests、burst_tokens、rate_tokens_per_s。
   - burst 的個數也依規則重算：`burst_requests = ⌊(1 − m) × 該 tier 的 S2 TTFT ÷ 斜率⌋`。2K 為 ⌊1,900 ÷ 324.23⌋ = 5，8K 為 ⌊5,700 ÷ 1,389.15⌋ = 4，就是 042 D5「套上 guard band 後」的個數；本文只寫「沿用 042 D5 的換算」。數值不變（9,600 / 32,256）。
   - 和 λ_ref 一樣，數值 pin 在設定檔，因為 WSL 的部署沒有 `results/`。`pacing_problems` 在 Mac 上從 committed 的 D5 檔重算並比對。
7. **逾時表**：V2 有 5 個點（待決事項第 6 項加入 F-U2K），所以 1,260 s 有 5 個，不是測試清單寫的「1,260 × 4」。其他值都和本文相同。
8. **mini 的參數放在 `stage_v_plan.MINI`**：n 2、N 20、sentinel 在 C 2、各點的 C、L-U8K 改讀 `v1-mini-8192`、GW-U8K-O512 在 r1 強制逾時（實作補記六第 7 項；原本是 r2）。
   - W-mini 是 `POINTS` 中只屬於 mini 的點：V4、C 9、`v4-mini`，不屬於任何單位。
   - 本文沒有寫 mini 各點的 bench 估計。為了時數上限的檢查，模組寫入 N = 20 的粗估（合計約 0.54 h，本文估 0.6 h），只用於 mini 自己的上限檢查。
   - 假的 SLO、N = 20 的 patch、5 s 的強制逾時，以及「在最後一次計畫的 V boot 之前強制停止」由 runner 實作，和 `run_stage_d` 相同。
9. **時數規則的常數**：上限檢查的保留量 0.06 h（`END_RESERVE_H`）與 sentinel 的 bench 估計 103 s 是模組常數，不在設定檔。`cap_allows`、`total_cap_allows`、`min_gap_ok` 以 Fraction 精確計算，由 runner 與 report 共用。
10. **請求路徑與 profile 的檢查不在 `build_plan` 內**：
    - `gateway_path_problems`（列出的檔案等於 AST 閉包；有凍結值時 sha 相等）與 `profile_problems` 由 runner 在 session 開始時呼叫。
    - 原因：V-CLS 之後，請求路徑可以在 main 上繼續修改（見「凍結」），report 仍必須能重建計畫。
    - `build_plan` 只檢查 stage_v.toml、stage_d.toml 與兩者的一致性。
11. **`stage_v_problems` 也讀 stage_d.toml**：`[slo]`、`[secondary_slo]`、sentinel 與 var-2048 必須等於凍結的 stage_d.toml。`build_plan` 另要求傳入的 stage_d.toml 的 sha256 等於 `stage_d_sha256`（`be7909b1…`），並執行 `stage_d_plan.stage_d_problems`。
12. **gateway 設定一律寫出佇列設定**：產品的 2.0 / 300.0 也寫（等於預設值，行為不變），所以 `/readyz` 的 queue 可以直接和計畫比對。計畫記錄每個點渲染出的 TOML 位元組的 sha256（`gateway_toml_sha256`）。
13. **`--request-id-prefix <tag>-` 加在 argv 的最後**；diff 測試確認凍結函式的輸出只有三處差異。
14. **巢狀 TOML writer 自行跳脫字串**（`"`、`\`、控制字元與 U+007F）：JSON 的字串跳脫不處理 U+007F，而 `ensure_ascii` 會產生 TOML 不接受的 surrogate 跳脫。非有限的浮點數、None 與其他型別一律拒絕。
15. **計畫帶 `kind = "verify"`**：每個 session 都是 D-C 的形式（固定點、n 個 run、每個 run 一次 boot），`run_stage_d` 的 harness 會讀這個鍵。

## 實作補記三（2026-10-24，harness：`stage_v_datasets` 與 `stage_v_analysis`；沒有任何 Stage V 數據）

`benchmarks/stage_v_datasets.py`、`benchmarks/stage_v_analysis.py` 與 `tests/test_stage_v_datasets.py`、`tests/test_stage_v_analysis.py` 已依本文以 TDD 實作。凍結模組只 import（`capacity_analysis`、`capacity_plan`、`core_b_analysis`、`core_b_plan`、`stage_d_analysis`、`stage_d_datasets`、`stage_d_plan`），一個位元組都沒有改，也不 import capserve。dataset 的測試用假的 sampler，vLLM 只在 WSL 的 CLI 載入（`stage_d_datasets.vllm_sampler`）。analysis 的測試以封閉迴路、嚴格 FIFO 的假 gateway 產生逐 request 的資料，並以和 runner 相同的規則模擬整個 session，再由 replay 檢查。下列是本文沒有寫死、實作時決定的細節，以及和本文字面不同、依程式處理的地方。

1. **vLLM 0.30 原始碼的唯讀核對**（scratch 中的副本）：`CustomDataset.sample` 只有在 output_len 為 None 或 −1 時才讀每列的 `output_tokens`；`prompt_len` 是 `len(tokenizer(prompt).input_ids)`，和 `stage_d_datasets.vllm_sampler` 的計數相同；request_id 為 `request_id_prefix + str(列號)`，列數等於 N 時不會 oversample；warmup 送 row 0 的 `test_input`，不帶 request_id，也就不帶 `x-request-id`。和本文一致。（這次核對漏了 random dataset 在 warmup 之前送的一次 `POST /tokenize`，見實作補記六第 5 項；也誤以為 ready check 會送出，其實 `--ready-check-timeout-sec` 預設 0，凍結的 client 不送，見實作補記七第 1 項。）
2. **mini dataset 的 seed**：本文沒有寫。用 `"<seed>:stage_v:<key>"`（`v1-mini-8192`、`v4-mini`）。`v1-mini-8192` 先依 V1 的規則抽 row 1–19，再把 row 5 / 6 / 7 換成 1 / 2 / 512。
3. **W 中 8K 的個數不寫進 plan**（本文 V4 寫「寫進 plan」）：
   - 這個個數只由 seed 決定，不需要 tokenizer。`stage_v_datasets.window_counts` 計算，測試把 10 條序列的值 pin 住（90/10 為 24–27，50/50 為 126–129），analysis 在每個 W 點的 descriptive 中逐 block 報告。
   - 不寫進 plan 的原因：`stage_v_datasets` import `stage_v_plan`，反方向會循環 import；plan 的形狀也已由前一步的測試鎖定。
4. **import 比本文的表多**：`stage_v_datasets` 另 import `stage_d_plan`（V1 / V4 的 prompt 長度取自 base workload）。`stage_v_analysis` 另 import：
   - `stage_v_datasets`：在 Mac 上從 seed 重算 L 點每列的 pin 值與 W 點每列的 class，不信任 runner；
   - `stage_d_datasets`：E 的 b，取 var-2048 的長度；
   - `stage_d_plan`、`core_b_plan`。
   全部都是只 import 的研究模組，測試以 AST 鎖住 import 清單。
5. **(h) 的細節**（`join_access_log`）：
   - 只讀 JSON 物件的行。uvicorn、capserve 的 stderr、traceback 都不是紀錄；以 `{` 開頭卻不是 guard 紀錄的行算問題。
   - 「沒有 id 的剛好 C 筆」用 `core_b_plan.num_warmups(C)`（原為 ≥ `num_warmups(C) + 1`，實作補記七第 1 項）。另外要求這些紀錄都在第一個主要 request 到達之前，因為只有 warmup 不帶 id。
   - completion route 指 `POST /v1/completions`。除了 `/healthz`、`/readyz`、`/metrics`，以及 random dataset 的 client 剛好一次、404 的 `POST /tokenize`（實作補記六第 5 項），其他 route 或 method 的紀錄都算問題（本文只寫「沒有其他 prefix」）。
   - id 的後綴必須是 0..N−1 的標準十進位寫法（`07` 不算）。
   - 50 ms 的容差在 derived 檢查（`gw_ttft_ms ≤ ttft_ms + 50`），因為 log 本身看不到 client 的 TTFT。
   - 主要紀錄另外檢查 t_arrival_s ≤ t_sent_s ≤ t_released_s，以及等待時間不為負。
6. **(i) 也套用到沒有 id 的紀錄**：任何非 200 或 outcome 不是 ok 的 completion 紀錄（主要 request 或 warmup）都讓 run 無效。（原寫「log 中無法分辨 warmup 和 ready check 的重試」「ready check 可能重試只影響個數的下限」；凍結的 client 不送 ready check，沒有 id 的紀錄就是 C 個 warmup：實作補記七第 1 項。）
7. **(j) 由 access log 的區間判定**：
   - [t_sent, t_released) 為半開區間，同一時刻先釋放再送出，以 Fraction 精確計算整個 bench process 的峰值（主要 request 之外，C 個 warmup 的區間也算：實作補記六第 9 項、實作補記七第 1 項）：request 數 ≤ C；weighted-v1 的單位數 ≤ 預算，各 class ≤ 上限。
   - 1 Hz sampler 的峰值和 030 8.6 一樣只作描述（`inflight_le_c`），不是有效條件；它可能取樣到 vLLM 收尾的瞬間。
8. **T_W 為空**（只可能出現在 mini，W 只有 2–4 個 request）：U_W 或 N1 無法計算，視為非平凡條件不成立，也就是 trivial，不是 invalid。V-GW 與 V1 沒有非平凡條件，不受影響。
9. **E**：pacing 開的點，b 由第一波的 prompt 長度計算，和到達順序無關（uniform 用 prompt，var 用 `stage_d_datasets.lengths(2048, C)`，就是 var-2048 的前 C 列）。門檻 2K 為 26、8K 為 3、var 為 12（b 19）、mini 的 P-U2K（C 8）為 2。pacing 關的點，derived 與 log 中任何 completion 紀錄（含 warmup）的 prefill_wait 都不能 > 0。
10. **derived 的欄位**（`gateway_requests`）：
    - `run_stage_d.derived_requests` 的欄位，加上 `out_planned`、`units`、`gw_queue_wait_ms`、`gw_prefill_wait_ms`、`gw_ttft_ms`、`gw_upstream_ttft_ms`、`gw_arrival_s`、`gw_sent_s`、`gw_released_s`。
    - 時間以 run 中最早的主要到達為 0，四捨五入到 µs（單調，不會產生新的重疊）。
    - W 點的 cls 由 dataset 決定（8,064 為 long），因為 client base U2K-var 會把每個 request 標成 short。units 取 log 的 `admission_units`，其他點為 1。
    - derived 不帶 prompt 長度（本文的欄位表也沒有），所以 (l) 的「input_lens 逐一等於 dataset」仍由 runner 以 bench 的 input_lens 檢查；Mac 上檢查的是每列的 units 與 cls。
    - 固定輸出的點照 D-C 用 `stage_d_analysis.derived_problems`。L / W 點另寫檢查：(k) out 等於 pin 值，`out_planned` 等於 dataset，cls 等於 dataset 的 class。W 中一定有 short 與 8K，由 dataset 保證（第 3 項）。
11. **run_eval_v1 的第二個差異**：本文說「唯一的差別是 TPOT 只取 out ≥ 2」。逐 request 的聯合達成率（只作描述）也必須跳過 out = 1 的 TPOT，否則會除以 0。這不影響任何判定，測試確認 out 全部 ≥ 2 時兩者完全相同。
12. **點的類別**：除了判定 view 的 `ca.classify` 為 FAIL，任何 run 的 outcome 為 fail 也是 FAIL（本文「任一有效 run 為 fail」的字面）。W′ 與 full 的類別、WINDOW_SENSITIVE（任一判定 view 的 W′ 類別和 W 不同）只作描述。
13. **replay 與時數上限的紀錄格式**（runner 必須照此寫，report 照此重算）：
    - 每個 boot 記錄 `elapsed_s`，也就是時數上限檢查當下距 session 開始的秒數，必須單調不減。
    - 停止時，`session["cap_stop"] = {block, point, elapsed_s, forced}`。被停止的 run 不寫成 row，analysis 從 replay 算出 `schedule.not_run_cap`。
    - mini 的強制停止發生在「當下仍計畫中的最後一個 V boot」之前，也就是依已有的跳過計算，`forced = true`。真正的上限停止 `forced = false`。
    - mini 的強制逾時 row 的 `timeout_s` 為 5 s（`run_capacity.MINI_FORCED_TIMEOUT_S`，由測試鎖定）。
    - boot 與 row 沿用 `run_capacity` 的形狀：V boot 的 `workload` 就是點的 id，sentinel 為 U2K。
14. **units 段**（產生器唯一讀取的段）：
    - `holds` 要求 session 有效：replay 乾淨、sentinel 閘門通過、不是 mini。env、凍結與 sha 的拒絕由 report 負責。
    - `validation` = {study（單位第一個點的研究）、unit、points、measured_classes、worst_r {點: {client, upstream}}（各項 worst_R 的最大值）、n、queue、pacing、admission，有逐 block dataset 的點另加 datasets 與 datasets_sha256，W1 另加 measured_mixes [0.9, 0.5]}。
    - `vetoed` 只有 PACE-2K 有值，其他單位為 null。
15. **V2 的兩張 2×2 表**寫成 `readings`：
    - var 那一對：co_occur、var_passes_without_pacing、not_fixed、veto、uninterpretable；
    - P-U2K / F-U2K：formal_class_holds、attributed_to_pacing、formal_class_lost、unattributable。
16. **hypothesis 的判定**：
    - H-W0：任一有效 W run 的 U_W < 28.8 為 not supported；兩個 mix 都至少有一個有效 run 而且全部 ≥ 28.8 為 supported；其他為 NT。
    - H-GW4 的預測寫成 not supported。依對應規則，MARGINAL 的判定是 inconclusive，FAIL 才是 not supported。實作補記六第 11 項改為以點的類別記錄預測（H-GW4 為 MARGINAL 或 FAIL），並機械判定是否如預測。
17. **另加的描述**：W 點的 vLLM 內 request 數是否 ≥ 0.9 × 043 的有效並發（24.6 / 12.8），以及 sampler 的 KV、preemption 與 waiting 峰值。

## 實作補記四（2026-10-24，harness：`run_stage_v` 與 `stage_v_report`；沒有任何 Stage V 數據）

`benchmarks/run_stage_v.py`、`benchmarks/stage_v_report.py` 與 `tests/test_run_stage_v.py`、`tests/test_stage_v_report.py` 已依本文以 TDD 實作。凍結模組只 import，一個位元組都沒有改，也不 import capserve（AST 測試鎖住 import 清單）。runner 的測試在假的 WSL 上跑完整的 session：vLLM 沿用 `test_run_stage_d` 的假 server；假的 `capserve serve` 在 provisional profile 沒有 `--allow-provisional` 時以 exit 2 結束，`/readyz` 由它收到的 settings 與 profile 算出；每個 bench 的 request 與 access log 由 `test_stage_v_analysis` 的封閉迴路嚴格 FIFO gateway 產生，主要 request 的 access log 行要等 gateway 停止、log reader 讀到 EOF 才寫進檔案，所以在 `gateway_down` 之前解析 log 的實作會失敗。每個 session 都再由 `stage_v_analysis` replay。report 的測試由這些 session 產生 analysis.json，並逐項觸發拒絕條件。下列是本文沒有寫死、實作時決定的細節，以及和本文字面不同、依程式處理的地方。

1. **gateway log 的時間戳**：`run_adv.start_server` 的 log reader 在每一行前面加上 wall clock（ms）與一個空格（`run_adv.pump_log`），所以 `raw/<boot tag>.gateway.log` 中的 access log 行不是以 `{` 開頭。runner 先移除每行開頭的這個戳記（`access_log_text`），再交給 `stage_v_analysis.join_access_log`。raw 檔原樣保留，sha256 記在 raw_manifest（mini 例外：刪除檔案、保留 sha256，實作補記六第 10 項）。
2. **derived 的檢查、(j) 與 E 併入 runner 的判定**：`stage_v_analysis.run_valid` 在 runner 的判定（bench、spill、gateway 與 access log 的對應）之外，另查 derived 的檢查、(j) 與 E。runner 在 decide 時先執行它；不成立時，把原因寫進 row 並把 row 設為無效，再以 `run_outcome` 決定 outcome。所以 row 的 `valid` 和 replay 重算的結果一致。V run 的 derived 檢查因此只在這一處（sentinel 仍在 run_point 用 `capacity_analysis.derived_problems`）。
3. **log reader 沒有 join 時 run 無效**：(h) 的前提是「`gateway_down` 之後 log reader 已經 join，log 是完整的」。reader 沒有 join 時，access log 可能不完整，該 run 記為無效，並列為 gateway 造成的無效 run。
4. **`session["gateway_faults"]`**：停止規則要求「gateway 造成的無效 run 之後，不開始下一個 V session，先寫 decision log」。runner 把下列情況逐一記下 boot、run 與原因：
   - boot 時 gateway 未就緒、`/readyz` 不符、就緒後立刻結束；
   - bench 之前或期間 gateway 結束（bench 之前的情況原本沒有記錄，實作補記七第 3 項補上）；
   - log reader 沒有 join；
   - access log 的對應有問題；
   - (j) 被違反。
   runner 本身不據此拒絕下一個 session，「每個 session 最多 2 次嘗試」也沒有機械化：兩者都要判斷哪些 session 屬於同一次凍結、哪些中止的 session 算嘗試，仍由操作者依本文先寫 decision log。
5. **輸出目錄的名稱**：formal 必須是 `results/stage_v/formal-<id>`，mini 必須是 `results/stage_v/smoke-v-<id>`，而且直接位於 `results/stage_v` 之下（本文只寫「輸出不在 `results/stage_v/<sid>`」）。原因：2 h 空檔讀的是 `formal-*`，凍結證據要求 `smoke-v-*`（`stage_v_plan` 的規則）。這個規則也排除了 `runs/` 之下的路徑，所以不另做 `runs/` 檢查。
6. **mode 與 plan**：
   - `--mini` 只執行 session "mini" 的 plan；`--formal` 只執行 V-ADM / V-CLS（`build_plan` 本來就拒絕 formal 的 mini）。
   - mini 的 session plan 是 `stage_v_plan` 的 mini plan，SLO 換成 `run_capacity.MINI_SLO`，stage 為 "V-mini"。N = 20 由 `run_stage_d.mini_num_prompts` 在 session 期間 patch。
7. **mini 也比對 dataset 的 sha256**：`run_stage_d` 只在 formal 比對。依本文的時序，datasets 在 mini 之前就 pin，所以 mini 讀到的每個 dataset 也必須等於 pin 值；還是 "" 時一律拒絕。
8. **全期 12 h**：開始任何 V session（含 mini）之前，runner 加總 `results/stage_v/*/session.json` 的 finished_at − started_at（精確到 µs，以 Fraction 比較）。加上本 session 的上限超過 12 h 就拒絕，訊息要求先問使用者。
   - 任何 session.json 缺少其中一個時間就拒絕（fail closed）。
   - `formal-*` / `smoke-v-*` 目錄沒有 session.json、讀不了、或仍標示進行中（`in_progress`、finished_at 為 null）時也拒絕（實作補記六第 8 項）。
   - 在建立目錄之前就被拒絕的 session 不算；建立目錄之後被拒絕的 session 有已結束的 session.json，時數照算。
   - runner 只能看到該 checkout 的 `results/stage_v`。
   - session.json 記錄 `hours_used_h`。
9. **2 h 空檔**（只對 formal）：在 `formal-*` 之中，取 started_at 不晚於本 session 開始時間、finished_at 最晚的一個，它必須比本 session 的 started_at 早至少 2 h；任何 `formal-*` 缺少時間、沒有 session.json 或仍在進行中就拒絕（實作補記六第 8 項）。
   - session.json 記錄 `previous_formal`。
   - report 用同一個函式（`run_stage_v.gap_problems`）在 Mac 上檢查已 commit 的 `formal-*`，排除自己，而且只看 started_at 不晚於自己開始時間的目錄；讀不到 started_at 的目錄仍然檢查（fail closed），本 session 沒有可讀的 started_at 時檢查全部（實作補記七第 4 項）。
10. **elapsed_s 與時數上限**：session 開始時（取 started_at 的同一刻）記下 monotonic 的 t0。每個 boot 的 `elapsed_s` = round(now − t0, 3)，時數上限的判定與記錄用同一個值，所以 report 的 replay 由記錄值重算，結果必然相同。sentinel 的 boot 也記錄 `elapsed_s`，因為 replay 要求每個 boot 都有。
11. **渲染出的 gateway settings**：plan 的 `gateway_toml_sha256` 由 `build_plan` 以同一個 writer 算出，runner 又驗證 plan 等於 `build_plan` 的結果，所以 session 開始時再比對一次是同義的，不另寫檢查。每個 boot 寫出 `toml_dumps(x["gateway"])`，並記錄其 sha256（`toml_sha256`），report 比對它等於 plan 的值。
12. **`/readyz` 的 tuple**（`READYZ_KEYS`）：
    - 比對的欄位：
      - profile {id, status}、serving_status；
      - active_class（weighted-v1 為 null）；
      - limits：max_concurrency、max_queue、max_prompt_tokens、max_output_tokens，weighted-v1 為產品補記第 6 項的包絡；
      - queue；
      - admission：policy、budget、max_queue，以及各 class 的 name、units、max_concurrency；
      - prefill_pacing；
      - gateway_path_sha256：等於 session 開始時算出的值，formal 時就是凍結值。
    - 不比對的欄位：
      - admission 中各 class 的 token 上限，由 profile 的 sha256 鎖住；
      - `gateway_validated`：study profile 與 Stage D profile 一律為 null（產品補記第 7 項）。
    - profile id 讀自 checkout 中、sha256 已經比對過的 profile 檔。
13. **gateway 的 argv**：和 D-C 相同，為 `python -m capserve serve --config raw/<boot tag>.gateway.toml --profile <checkout>/<profile>`，只有 provisional 的點加上 `--allow-provisional`。session 記錄 `root`（checkout 的絕對路徑）與 gateway_python，report 以它們重建 argv 並逐字比對。
14. **bench 結果的檢查**：
    - 固定輸出的 client（uniform、007 的 mix、var）沿用 `run_stage_d.summarize`。
    - L / W 點要求 input_lens、output_lens 逐列等於該 block 的 dataset（`stage_v_analysis.planned`，由 seed 重算），也就是 (k)、(l) 在 runner 這一側的檢查。實作補記三第 10 項留給 runner 的「input_lens 逐一等於 dataset」在這裡完成。
    - sentinel 的 derived 檢查用 `capacity_analysis.derived_problems`，和 analysis 判定 sentinel 的函式相同。
15. **import 比本文的表多或少**：
    - `run_stage_v` 另外 import：
      - `capacity_analysis`：sentinel 的 derived 檢查；
      - `capacity_plan`：CLI 讀 pins；
      - `run_core_a`：env 與 provenance 的檢查、raw manifest，和 `run_stage_d` 相同。
    - `run_stage_v` 不 import `stage_d_analysis`。
    - `stage_v_report` import `run_stage_v`：輸出位置、gateway argv、`/readyz` tuple、2 h 空檔的規則和 runner 共用。它另 import `run_core_a.sha256_json`，不直接 import `run_stage_d`。
    - 全部都是只 import 的研究模組，測試以 AST 鎖住清單。
16. **report 的拒絕條件**：除了比照 `stage_d_report` 的部分，還有下列各項，都是本文的規則，只是 report 一列沒有逐項寫出：
    - session 的 `harness_sha256` 等於目前（凍結）的 harness；
    - `gateway_path_sha256` 等於凍結值；
    - `profile_sha256` 等於 plan 的 pin；
    - pacing 由 committed 的 D5 重算（`pacing_problems`）；
    - committed 的 profile 等於 pin 住的位元組（`profile_problems`）；
    - 有效的 V run 的 access log 對應沒有問題，而且剛好 N 筆；
    - plan 的 session 是 V-ADM 或 V-CLS。
    - analysis.json 已經存在、而且和重跑的位元組不同時，report 拒絕，不覆寫（021 R3）。
17. **boot 的 tag**：V boot 的 tag 為 `b<NNN>-V-<point>-r<block>`。`run_capacity.boot_tag` 會把 V phase 寫成 `fs-<w>`，D-C 沿用了這個標籤，本文沒有規定。sentinel 的 tag 不變。
18. **mini 的通過條件**（`mini_checks`）：
    - 「E 有作用，至少一個 queue_wait > 0」寫成兩項：某個有效的 pacing run 中有 request 的 prefill_wait > 0；某個有效的 weighted-v1 run 中有 request 的 queue_wait > 0。
    - 「U_W 與 N1 算得出來」：至少一個有效 run 有非平凡條件，而且每個有非平凡條件的有效 run，T_W 都不是空的。
    - 其他各項照本文：replay 乾淨；除了刻意的逾時，所有 run 有效；刻意的逾時發生；derived 的欄位；每個 boot 都通過檢查；gateway 回報 tuple 並乾淨停止；強制的上限停止；session.json 記錄兩個 sha；teardown 乾淨。
    - 實作補記六第 7 項另加 `skip_walked`：強制逾時的點在之後每個 block 的 run 都被跳過，而且至少有一個。

## 實作補記五（2026-10-24，V4 凍結前的設計檢查：U_W 的桌面模擬；沒有任何 Stage V 數據）

`benchmarks/stage_v_prereg_sim.py` 與 `tests/test_stage_v_prereg_sim.py` 依本文 V4「凍結前的設計檢查」以 TDD 實作，輸出 `results/stage_v/prereg-sim/sim.json`（`descriptive_only: true`）。腳本只讀已 commit 的 c1b 與 D-C 檔，不 import capserve；凍結模組只 import（`capacity_analysis`、`capacity_plan`、`stage_d_plan`，以及 `stage_v_analysis`、`stage_v_datasets`、`stage_v_plan`），AST 測試鎖住 import 清單。U_W、N_W、(j) 與非平凡條件直接用 `stage_v_analysis` 的 `in_vllm`、`inflight_problems`、`nontrivial`，也就是 V4 判定時用的同一個估計量。測試要求 committed 的輸出由 committed 的檔案逐位元重現。下列是本文沒有寫死、實作時決定的細節，以及結果。

1. **模擬的範圍**：V-ADM plan（`stage_v_plan.build_plan`，非 formal）的 W-90、W-50：C 32、N 320、weighted-v1 的預算 32、單位 1 / 4、上限 32 / 8。block r 用該 block pin 住的序列（`stage_v_analysis.planned`，只由 seed 決定）；每條序列模擬 20 個 run，每個 mix 共 100 個。
   - 主要 run 從空的狀態開始，因為 bench 會等 warmup 完成。idx 0..C−1 在時間 0 依 idx 順序到達；idx k ≥ C 在第 k − C + 1 個釋放之後、再隔一個 client 往返時間到達（vLLM bench 的 semaphore 依 idx 順序喚醒等待者）。
   - gateway：單一嚴格 FIFO，每個事件之後重新 pump，和 capserve 的 `PolicyController` 相同（研究端不 import）。隊首的單位放得進剩餘預算、而且它的 class 未達上限才送出；隊首放不進時，後面的都不能插隊（HOL）。
   - gateway 自己的 parse 與 tokenize 時間不在任何 committed 檔中，不模擬。
2. **vLLM 內的時間**：2K 取 c1b U2K @ C_PROD 32、8K 取 c1b U8K @ C_PROD 8 的 e2e latency，只用有效的 Phase C run（`run_valid`，和 `stage_d_plan.lambda_ref` 相同）。c1b 是直連，所以 latency 就是在 vLLM 內的時間。
   - 在時間 0 送出的 request（開場湧入）從 idx < C 抽，其他從 W = [C, N − C) 抽。W 的 pool 正是 λ_ref 平均的那些 request：U2K 768 個、U8K 920 個，測試對 stage_d.toml 的 pin 比對。
   - 本文寫「c1b 與 D 的逐 request 時間」。D 的檔案只用在第 3 項的往返時間：D-C 經過 gateway 的 latency 含 gateway 的開銷，不是 vLLM 內的時間；c1b 則沒有 start_s。
   - 主要假設：每個 request 在 vLLM 內的時間和混合的組成無關。這是 043 7.3 A 的前提：每個 class 單獨在它的 C_PROD 填滿預算是量過的狀態，混合的 batch 從未量過。
3. **client 往返時間**：取自 D-C 的 U2K-gw，也就是經過 gateway、C 32（W 點的 client 並發數）的 5 個有效 run。
   - 計算方式：idx k ≥ C 的 start_s，減去第 k − C + 1 早的 client 完成時間（start_s + latency）。
   - 共 1,440 個，平均 1.79 ms（0.28–4.22 ms）。
   - 任何一個為負，就不是 C 個 client 的封閉迴路，腳本拒絕。往返 run 的 C 必須等於 W 點的 C。
4. **變體**（只作描述）：
   - `committed`：上述模型，設計檢查用這個；
   - `steady_only`：每個 request 都從 W 抽；
   - `8k_x0.8`、`8k_x1.25`：8K 在 vLLM 內的時間乘上 0.8 / 1.25。這是對第 2 項主要假設的假設情境，committed 的檔案無法涵蓋。
   - 每個（變體、點、block）有自己的 seed：`sha256("20261008:stage_v:prereg-sim:<變體>:<點>:<block>")`。
5. **設計檢查的規則**：每個 mix 的 `committed` 模擬 run，U_W 平均（以 Fraction 精確計算）≥ 0.93 × 32 = 29.76。
   - 模擬的 run 違反 (j) 是程式錯誤（`SimError`），不是結果。
   - 腳本只報告檢查結果，不修改任何門檻或設計。
6. **結果**（descriptive_only；不是 Stage V 數據，不預測 SLO）：
   - **兩個 mix 都 ≥ 29.76，不需要在凍結前修改門檻或設計。**
     - W-90：U_W 平均 31.55（各 run 31.38–31.80）。
     - W-50：U_W 平均 30.80（30.50–31.15）。
   - 四個變體的平均在 30.80–31.59 之間。單一 run 的最小值為 30.45（W-50、steady_only）。兩個 mix、四個變體共 800 個 run，沒有任何一個 < 28.8（trivial）。
   - H-W0 的預測（兩個 mix 每個有效 run 的 U_W ≥ 28.8）在模擬中成立。
     - HOL 加上往返時間浪費的單位：90/10 約 0.45，50/50 約 1.2。
     - 和本文「約 1–1.5 單位」的估計同一量級。
   - 描述：
     - vLLM 內的平均 request 數 N_W 為 24.55 / 12.29，043 的有效並發為 24.6 / 12.8。
     - gateway 佇列的峰值為 17 / 24，低於 max_queue 32，所以封閉迴路下不會出現 429。
     - 模擬的 bench 時間平均為 178 s / 339 s，本文的估計為 170 s / 330 s；W-50 的 `8k_x1.25` 為 406 s。
       本文的估計沒有修改：兩點各 5 個 run，合計多約 86 s（0.024 h），V-ADM 的上限 3.5 h 仍有餘裕。
7. **限制**：
   - 每個 request 的時間和混合的組成無關（第 2 項）；
   - gateway 的 parse 與 tokenize 不計；
   - 模擬只是對同一份 c1b 的重抽樣。
   - 這些只影響「門檻是否達得到」的估計，不涉及任何判定。
8. **輸出與凍結**：
   - 輸出不記錄 stage_v.toml 的 sha256，只記錄用到的 plan 值（C、N、預算、class、序列）與每個輸入檔的 sha256。所以 datasets 的 sha256 pin 進設定檔、`freeze.stage_v` 改為 true 之後，輸出仍逐位元相同：序列只由 seed 決定。
   - 腳本不在 `HARNESS_FILES` 中：它不是 harness，runner 與 report 都不讀它。
   - `test_stage_v_frozen` 是否鎖它的 sha256，由凍結那一步決定。重新產生的測試已確保 committed 的輸出就是這份腳本在 committed 檔上的結果。

## 實作補記六（2026-10-24，第一輪凍結前審查）

第一輪凍結前審查確認的問題，依 TDD 修正：先寫會失敗的測試，再改程式，並以 in-process 的突變（只在記憶體中改模組原始碼）確認每條新規則都有測試抓得到。凍結模組一個位元組都沒有改，研究程式仍不 import capserve；c1b、三份 Stage D profile 與 `rtx-verified.toml` 的位元組不變，仍逐位元重新產生；新的規則不需要旗標，也沒有任何覆寫選項。沒有任何 Stage V 數據。

1. **gateway 的「第一個 token」改為 vLLM 0.30 bench 的定義**（`proxy.is_token_event`）：
   - 舊規則：`ttft_s`、`upstream_ttft_s` 記在第一個 text（chat 為 delta.content）非空的 event（`has_content`）。
   - vLLM 0.30 bench（`endpoint_request_func.async_request_openai_completions`，chat 的函式也一樣）在第一個 JSON 帶非空 `choices` 的 chunk 記 TTFT，不管 text 是否為空；原始碼註解寫明特殊 token 的 text 可能是空的。`--ignore-eos` 加上亂碼或自訂 prompt 時，第一個生成的 token 可能是不完整的 UTF-8 byte-BPE token 或被略過的特殊 token，串流中的 text 為 ""。
   - 所以在舊規則下：gateway 的時間點至少比 client 晚一個 decode step；out = 1 而且唯一的 token 解碼為 "" 時，`ttft_s` 根本不會記錄，而 (h) 要求每筆都帶 `ttft_s`，run 會無效。「共通的定義」中「upstream 和 C_PROD 是同一個定義」、029 metrics 表的「和研究的 TTFT 定義相同」與 (h) 的 50 ms 比對，都不成立。
   - 新規則：一個 event 的第一個 `data:` 行是 JSON 物件，而且 `choices` 是非空的 list，就是 token event。`[DONE]`、只帶 usage 的 chunk（`"choices": []`）、in-band 的 error event、SSE 註解與無法解析的行都不是。`ttft_s`、`upstream_ttft_s` 記在第一個 token event。上述三處從此成立，(h) 的 50 ms 比對的是同一個事件。「共通的定義」已補上這個時間點。
   - `has_content` 不變，仍用於 itest 的內容檢查與 loadtest，這些檢查不因此變寬。loadtest 的 client 端 ttft 仍是第一個有內容的 event，只作營運描述，不是研究數據。
   - 營運 metrics：`capserve_ttft_seconds`、`capserve_upstream_ttft_seconds` 的觀測點隨之改變，說明文字改為「到第一個 token event」。completions 每個 decode step 一個 chunk，所以只有第一個 token 的 text 為空時才不同，差一個 decode step；以前沒有觀測值的「out = 1、text 為空」的 request 現在有觀測值。chat route 的第一個 chunk 是只帶 role 的 delta，現在由它記錄，和 vLLM bench 的 chat 規則相同；chat 不在 Stage V 的量測範圍（揭露）。
   - 測試（`test_proxy`）：述詞對各種 event 的判斷；第一個 chunk 為空 text、之後有 text 時記在第一個 chunk；out = 1 唯一的 chunk 為空 text 時兩者都有值；每個 chunk 都是空 text、再接 usage 與 [DONE] 時兩者都有值；usage-only chunk 不記錄。
   - `proxy.py` 與 `metrics.py` 在請求路徑閉包內，所以 `gateway_path_sha256` 改變。目前沒有凍結值（`[freeze] gateway_path_sha256 = ""`）；凍結值在本輪修正與第二輪凍結前審查之後才計算。
2. **V1 class 的 C 由 loader 釘死**（`profile.V1_C`、`profile.V1_POINTS`）：
   - 舊規則：loader 只檢查 max_queue = max_concurrency、值在 1..128，沒有檢查值本身。V1 class 帶的是 D-A O512 的 evidence，8K 的 c_prod 是 12 而不是 8，所以沒有任何證據值可以比對。寫錯的 C（例如 8K 寫成 O512 的 C_PROD 12）會通過 loader，`serving_class` 又對 validated class 解除 043 §4 的 128 token 規則。實作補記一第 9 項的「loader 依此檢查」因此不成立。
   - 新規則：`V1_C = {"U2K-O512": 32, "U8K-O512": 8}`，就是 L-U2K / L-U8K 的 C。V1 class 必須 max_concurrency = max_queue = V1_C[workload]，否則拒絕。GW-U8K-O512 在 12 是「全部 512」的最壞情況，涵蓋 8（「和 043 字面的差異」第 9 項），12 本身仍永遠不能服務。
   - 另外，V1 class 的 `validation.points` 必須等於該單位的點，依 `[stage_v.units]` 的順序（V1-2K：L-U2K、GW-U2K-O512；V1-8K：L-U8K、GW-U8K-O512），也就是 `stage_v_analysis` 的 `units` 段寫出的值。
   - loader 仍只讀 stage_v.toml 的三個鍵（實作補記一第 9 項）。兩個常數改由產品測試比對 committed 的 `configs/stage_v.toml`：單位的點等於 V1_POINTS；L 點的 C 與 limits 的 max_concurrency 等於 V1_C；GW 點的 C ≥ V1_C。
   - 拒絕的測試：8K 在 12（O512 的 C_PROD）與 9、2K 在 64 與 31；points 順序顛倒、少一點、換成另一個單位的點。
   - `profile.py` 在閉包內，同第 1 項。實作補記一第 9 項已依此改寫。
3. **說明文字的更正**（行為不變）：`profile.py` 的模組說明、`frozen_configs` 的註解、`_stage_v_provenance_problems` 的說明，以及 `service/tests/fakes/profiles.py`、`test_profile.py` 的說明，原本寫「凍結前沒有任何 checkout 有 `configs/stage_v.toml`」。實作補記二第 1 項之後這不成立：檔案在凍結前就 commit，`[freeze] stage_v = false`。fail closed 仍成立，理由是 `stage_v = false`，不是檔案不存在。fakes 的 `FROZEN` 模擬沒有這個檔案的 checkout，`cp.frozen_configs()` 是 repo 的現況。
4. **`/readyz` 的 tuple 和真的 gateway 交叉比對**（`service/tests/test_stage_v_settings.py`）：
   - 實作補記四第 12 項的 tuple 原本只和 `test_run_stage_v` 中手寫的假 `/readyz` 比對過。
   - 新測試對 mini、V-ADM、V-CLS 的每一個點（共 19 個），用該點渲染出的 settings 與 profile 建立真的 `create_app`，注入 local tokenizer 的替身（Mac 上沒有 snapshot），斷言 `run_stage_v.readyz_tuple(gateway.readiness())` 等於 `run_stage_v.readyz_want(...)`。gateway_path_sha256 用研究端算出的值，並先斷言它等於產品端的值。
   - service 的測試可以 import 研究 harness，研究程式仍不 import capserve。
   - 目前 19 個點全部一致，所以這是回歸保護，不是錯誤修正。

第 5–12 項是 harness 端（`benchmarks/stage_v_plan.py`、`stage_v_analysis.py`、`run_stage_v.py`、`stage_v_report.py`）與 mini 的修正。它們只改 harness、測試與文件，不改 `service/capserve/`，所以 `gateway_path_sha256` 不因此改變；`harness_sha256` 改變，mini 還沒跑，沒有任何證據失效。第 6 項改變實驗設計（跳過規則），由主 session 在本輪審查中決定，本項就是它的 decision log。

5. **random dataset 的 `POST /tokenize`**（`stage_v_analysis.join_access_log`、`random_dataset`）：
   - vLLM 0.30 的 `vllm bench serve` 在 `--dataset-name random`（和 prefix_repetition）時，於 warmup 之前呼叫 `_align_prompts_to_server_tokenizer`（原寫「於 ready check 之前」；ready check 預設略過，實作補記七第 1 項），對 `{base_url}/tokenize` 送一次沒有 `x-request-id` 的 POST；任何錯誤都讓它印出警告、略過對齊（scratch 副本 `serve.py` 的 `main_async`，sha256 `ec255c3b…eec142`）。Stage V 的 base_url 是 gateway，capserve 沒有 `/tokenize` route（app.py 刻意不暴露），所以回 404，guard 仍寫一行 access log（method POST、path `/tokenize`、status 404，沒有 outcome）。
   - 舊規則把它當成「不是 completion route」的 (h) 問題：GW-U32K、GW-U2K-O512、GW-U8K-O512、P-U2K、P-U8K、F-U2K 的每個 run 都會無效（mini 的 GW-U8K-O512 與 P-U2K 也是），VG-32K、V1-2K、V1-8K、PACE-2K、PACE-8K 都不可能成立。兩個假的 log 都沒有這一行，實作補記三第 1 項的原始碼核對也漏了它。
   - 新規則：client base 為 uniform（`stage_d_plan.bench_argv_for` 給 `--dataset-name random` 的那些；測試逐點比對凍結的 argv）時，接受剛好一筆 method POST、path `/tokenize`、status 404、沒有 `client_request_id`、t_arrival_s 為數字而且早於每一筆 completion 紀錄（有沒有 id 都算）的紀錄。它不計入沒有 id 的筆數（剛好 C，原寫「≥ C + 1」，實作補記七第 1 項），也不適用 (i)。下列都是 (h) 的問題：random dataset 的 client 沒有或多於一筆；狀態不是 404；在任何 completion 紀錄之後才到；custom dataset 的 client（L、W、var、007 的 mix）出現 `POST /tokenize`；帶 id 的 `POST /tokenize`、其他 method 或 `/detokenize`（後者照舊是「不是 completion route」）。
   - 404 讓 bench 略過對齊，在這裡是 no-op：bench 本來就用 served snapshot 的 tokenizer 產生 prompt。直連時 vLLM 回 200，Core B 的 server log 只有 `POST /tokenize 200`、沒有 `/detokenize`（本機 archive 的 `results/capacity/c0-step0/raw/formal-20261009-coreb-uniform/server-uniform-8192-r1.log` 第 110 行），也就是長度本來就一致、不需要重新對齊。依程式推斷，D-C 的 U2K-gw / U8K-gw 也走過同一個 404（同一版 bench，gateway 同樣沒有 `/tokenize` route），所以負載和 D-C 相同；D-C 的 raw gateway log 不在 Mac 上，無法直接確認。
   - 假的 log（`tests/test_stage_v_analysis.log_text`，`test_run_stage_v` 的 World 也用它）對 random dataset 的點寫出這一行；clean join 的參數化測試加入 GW-U8K-O512、GW-U32K、F-U2K；反例逐項測試；runner 的測試確認每個 V boot 的 raw gateway log 中這一行出現的次數等於凍結 argv 是否為 random。
   - **和真的 gateway 交叉比對**（`service/tests/test_stage_v_settings.py`）：對 mini 的 P-U2K、GW-U8K-O512、L-U8K、W-mini、F-U2K-var，用該點的設定、政策與 profile 建立真的 gateway stack（uvicorn、guard，後面是假的 vLLM），送 vLLM 0.30 形狀的序列：runner 的 /readyz、random dataset 的 `POST /tokenize`、C 個 warmup（不送 ready check）、兩次 GET /metrics（spec decode 與 diffusion）、N 個帶 `x-request-id <tag>-<idx>`、並發 C 的 request、兩次 GET /metrics；payload 是 bench 的（含 `repetition_penalty` 1.0、`logprobs` null）。真的 access log 交給 `join_access_log` 必須沒有問題、剛好 N 筆，id-less 的紀錄給出 C 個 (j) 的區間。研究端的假 log 從此不能悄悄偏離真的格式。（原本的序列手寫了一個 bench 從來不送的 ready check，期望 C + 1 個區間，所以抓不到假 log 與真 bench 之間的這個偏離：實作補記七第 1 項。）
   - 本文 (h) 一節與「揭露」的 WSL 複核清單已補上；實作補記三第 1、5 項加註。
6. **跳過規則改回逐點（所有單位）**：
   - 舊規則（本文「run outcome」與 §12、實作補記二第 4 項）：一個點的 run 不是 ok 時，它所屬產品單位的其他點也不再測（「單位已死」）。
   - 新規則：沿用 039 D-C 與 030 8.4 的逐點規則（本文「沿用的規則」本來就這樣寫）：一個點有一次無效的 run，或一次有效的 fail / trivial run 之後，只有該點在之後的 block 不再測；同一單位的其他點照常測。單位仍要每一點都 PASS 才成立（`stage_v_analysis.units`），產品決策不變；否決規則與 var 那一對不變。
   - 原因：
     - 解決本文的內部矛盾：V-ADM 表中「依預測約 1.60 h」與待決事項第 7 項的「約 0.2 h」都假設 W-50 至少跑一次；但 block 1 的順序是 W-90 在 W-50 之前，依 H-W1 的預測 W-90 在第一個 run 就 FAIL，單位規則下 W-50 一次都不跑（實際 1.48 h、0.07 h）。
     - 每個事先登記的 hypothesis 都仍可回答：單位規則下，W-90 如預測 FAIL 時 H-W2 與 H-W0 必為 NT（實作補記五的 U_W 模擬就是為 H-W0 做的）；GW-U2K-O512 在 block 1 不是 ok 時 H-L1 為 NT；L-U8K 在 block 1 不是 ok 時 H-GW3 為 NT。逐點規則下，每個點都跑到它自己的第一次失敗為止。
     - 回到沿用的規則，不需要另外解釋「單位已死」。
     - GPU 代價有上界（見下）。
   - 時數，依 plan 的 est_s 重算（`test_the_044_hour_figures_follow_from_the_plan` 鎖住）：V-ADM 全部 run 2.753 h，依預測（P-U2K-var、F-U2K-var、W-90、W-50 各在第一個 run 就 FAIL）1.600 h；V-CLS 全部 run 5.256 h（表中的 5.25 是各列四捨五入後的和），GW-M8K-T 在第一個 run 就 FAIL 時 4.589 h；待決事項第 7 項的 W 代價是一個 W-90 加一個 W-50 run，0.191 h。本文的時數表、待決事項第 7 項與 §12 的「約 2.75 h」「約 5.3 h」在逐點規則下都成立，數字不改。
   - 額外 GPU 時數的上界：逐點規則下每個 session 最多就是全部 run 的估計（2.75 / 5.26 h），本來就是上限 3.5 / 7.0 h 的依據，時數上限的停止規則不變；全期上限 12 h（1.0 + 3.5 + 7.0 = 11.5 h）不變。和單位規則相比，多出的只是死單位原本會跳過的 run：依預測多 0.118 h（W-50 一個 run）；最壞情況 V-ADM 多 0.590 h（W-90 在 block 1 不是 ok，W-50 仍跑 5 次），V-CLS 多 1.073 h（block 1 中 GW-U2K-O512 在 L-U2K 之前、L-U8K 在 GW-U8K-O512 之前：L-U2K 的 5 次 0.349 h 加 GW-U8K-O512 的 5 次 0.724 h）。
   - 程式：`stage_v_plan.skip_group` 與 plan 中每點的 `skip_group` 欄位刪除；runner 的 `_v_machine` 與 replay 都只把該點移出之後的 block。本文「run outcome」、(session) replay、Harness、測試清單與 §12（本文附錄與 PREREGISTRATION.md）已改寫；實作補記二第 4 項加註。
   - 測試：一個點 FAIL 之後只有它被跳過，夥伴點 5 個 block 都跑而且 PASS，單位不成立；用舊的單位規則模擬出的 session，replay 拒絕；W-90 依預測 FAIL 時 W-50 跑 5 次，H-W2、H-W0 有判定；V1 兩點在 block 1 失敗時 H-L1、H-GW3 有判定；runner 的假 session 也是如此。
7. **mini 的強制逾時移到 GW-U8K-O512 的 block 1**（`stage_v_plan.MINI`、`run_stage_v.mini_checks`）：
   - 舊：GW-U8K-O512 r2 強制 5 s 逾時。seed 的 block 2 順序是 P-U2K、F-U2K-var、W-mini、L-U8K、GW-U8K-O512、GW-M8K-T：夥伴 L-U8K 已經先跑，n = 2 沒有 block 3，所以這個逾時什麼都不會跳過，mini 在硬體上從未走過跳過規則；本文 mini 表「走過跳過規則與單位已死」不成立，`mini_checks` 也沒有檢查任何跳過。
   - 逐點規則下，要讓同一點之後的 run 被跳過，n = 2 時逾時只能在 block 1。新：`MINI["forced_timeout"] = {"GW-U8K-O512": [1]}`。GW-U8K-O512 r1 逾時、無效，r2 依跳過規則不再測；強制的上限停止仍在 GW-M8K-T r2 之前；L-U8K 兩個 run 都有效，output_lens 的 pin 檢查（含 1 / 2 / 512）仍成立。
   - 代價與替代：GW-U8K-O512 在 mini 中沒有有效的 run。它原本驗證的「輸出 512、8,064 token 的邊界」改由 L-U8K 的有效 run 驗證（`v-uniform-8k-o512-c8` 為 8,064 / 512，每列 prompt 8,064 token，row 7 輸出 512）；provisional 複本的 `/readyz`（class_limit 12）在 r1 的 boot、bench 之前照樣檢查；只有固定輸出 512 的 client 與 D-C 的 derived 檢查在 mini 中沒有有效 run（同一套凍結函式已在 Stage D 的硬體上跑過）。換成其他點在 block 1 逾時，失去的是唯一的驗證：P-U2K（E、N1）、W-mini（佇列、U_W）、F-U2K-var（pacing 關的 E）、GW-M8K-T（T server 的 bench）、L-U8K（output_lens）。
   - `mini_checks` 新增 `skip_walked`：每個強制逾時的 (點, block)，該點在之後每個 block 的 run 都在 replay 的 `schedule.skipped` 中，而且至少有一個這樣的 run（不能空洞地成立）。輸出另加 `skipped`。`stage_v_analysis.schedule` 公開 replay 的 skipped / not_run_cap（`replay_problems` 不變）。
   - 本文 mini 表已改寫；實作補記二第 8 項、實作補記四第 18 項加註。測試：mini 的 timeouts 為 [[GW-U8K-O512, 1]]、skipped 為 [[2, GW-U8K-O512]]、L-U8K 兩個 run；每一項 mini 檢查都能失敗，含 skip_walked 的空洞與「強制的點沒有被跳過」兩種情況。
8. **session.json 一律寫出，缺少就拒絕**（`run_stage_v.run_session`、`hours_used`、`gap_problems`、`stage_v_report`）：
   - 舊：`hours_used`、`gap_problems` 只讀存在的 `results/stage_v/*/session.json`，session.json 只在 run_session 結尾的 finally 寫出。建立目錄之後、try 之前的拒絕（preflight 的 vLLM 版本、DEPLOYED_COMMIT）沒有 session.json；硬中止（SIGKILL、OOM、`wsl --shutdown`、Windows 重開機）也沒有。這種目錄被靜默略過：12 h 少算、2 h 空檔不擋。實作補記四第 8、9 項的 fail closed 只涵蓋存在但缺時間的 session.json。
   - 新：
     - 建立目錄之後立刻寫暫時的 session.json（`in_progress: true`、`finished_at: null`）；之後的每一步（plan.json、env 擷取、preflight 的拒絕、harness、整個 session）都在寫 session.json 的 try/finally 之內，所以 Python 看得到的每一種結束都寫出 `finished_at`、`in_progress: false` 與 `aborted`。沒有 harness 時（preflight 被拒絕）`env_valid` 為 false、0 個 run、`rows_sha256` 為 null。
     - `hours_used` 讀 `results/stage_v` 下每一個 `formal-*` / `smoke-v-*` 目錄（以及其他有 session.json 的目錄），`gap_problems` 讀每一個 `formal-*` 目錄：沒有 session.json、讀不了、仍標示進行中、或缺時間，都是問題。runner 因此拒絕開始；`stage_v_report` 另呼叫 `hours_used` 與 `gap_problems`（後者只看 `formal-*`），同樣拒絕，但只看 started_at 不晚於該 session 開始時間的目錄（讀不到 started_at 的仍算，fail closed），所以之後或無關的未結束目錄不會擋住較早 session 的 report 與產生器的重跑（實作補記七第 4 項）。`prereg-sim` 這類不是 session 的目錄不讀。
   - 操作：遇到這種目錄時，先寫 decision log 記錄它的時數與結束時間（最後一個 raw 檔的時間，或 started_at 加它的時數上限），把這個 finished_at 記入它的 session.json（帶時區 offset，而且不早於 started_at，實作補記八第 1 項），才開始下一個 session。沒有任何旗標可以跳過這個檢查。
   - 本文 Harness 一節的開始前拒絕、實作補記四第 8、9 項已加註。測試：只有 plan.json 與 rows.jsonl 的 formal 目錄、讀不了的 session.json、`in_progress` 的 session.json（有或沒有 finished_at）、只有 plan.json 的 mini 目錄，都讓 hours_used / gap_problems 回報問題、runner 拒絕、report 拒絕（report 只在該目錄的 started_at 不晚於本 session 時拒絕，實作補記七第 4 項；這些測試的目錄與本 session 同時開始）；env 擷取時 session.json 已經存在而且是進行中；preflight 因 vLLM 版本被拒絕的 session 留下已結束、可計時的 session.json。
9. **(j) 也算 warmup 的區間**（`join_access_log`、`warmup_spans`、`warmup_problems`、`in_vllm`；標題原寫「ready check 與 warmup」，實作補記七第 1 項）：
   - 舊：(j) 的峰值只由 N 筆主要 request 的區間計算；本文 (j) 要求整個 bench process 的任何時刻。warmup 的紀錄也帶 t_sent_s、t_released_s，卻沒有算進去，warmup 轉入第一波時殘留的 permit（042 與本文標出的轉折點）看不到。
   - 新：`join_access_log` 回傳第三個值，即沒有 id 的 completion 紀錄（C 個 warmup）；這些紀錄也必須帶 t_arrival_s ≤ t_sent_s ≤ t_released_s（數字），weighted-v1 的點另要求正確的 admission_class / units（(h)）。runner 以 `warmup_spans` 把它們的 [t_sent, t_released, units]（以 run 中最早的主要到達為 0，取到 µs，和 derived 的時間相同）記入 row 的 `warmups`。`run_valid` 要求 row 有剛好 C 個格式正確的區間（`warmup_problems`，units 必須是該點的 class 的單位；原為至少 C + 1，實作補記七第 1 項），(j) 的峰值把它們和主要 request 一起算；U_W 與 N1 仍只算主要 request 在 T_W 中的區間。
   - 正式 session 中每個點的 C 都等於 gateway 的上限，所以 (j) 依構造成立，除非 gateway 的釋放順序有錯，這正是 (j) 要抓的。mini 中幾個點的 C 小於 class 的上限；bench 在 warmup 結束之後、主要 run 之前還會發兩次 GET /metrics，warmup 的 permit 在 response 結束時就釋放。
   - mini 的 rows 經 `rc.mask_smoke` 只留 `MASK_ROW_KEYS`，warmups 不保留。prereg-sim 不傳 warmups，輸出逐位元相同。
   - 本文 (h)、(j) 與實作補記三第 7 項已加註。測試：warmup 在第一波時仍在 vLLM 內（9 > C 8；W 點 33 單位）時 run 無效；row 沒有 warmups、不是剛好 C 個（實作補記七第 1 項）、區間倒置、單位不對時無效；U_W 不受 warmup 影響；runner 的假 session 中每個 V row 記錄 C 個區間，warmup 延到主要 run 開始之後的 run 無效並列為 gateway fault。
10. **mini 刪除 gateway log**（`run_stage_v.run_session`）：本文 mini 寫「延遲數值依 `rc.mask_smoke` 丟棄」，但 mask_smoke 留下 `raw/<boot>.gateway.log`，其中有每個 mini request 的 ttft_s、upstream_ttft_s、queue_wait_s、duration_s。新：mini 在寫出 raw manifest 之後刪除 `raw/*.gateway.log`，sha256 仍在 raw_manifest，session.json 列出刪除的檔名（`gateway_logs_discarded`）。formal session 不受影響。本文「逐 request 的擷取」、mini 的通過條件與實作補記四第 1 項已加註。
11. **預測以點的類別記錄**（`stage_v_analysis.HYPOTHESES`、`hypotheses`）：本文寫 H-GW4 的預測為「not supported（MARGINAL 或 FAIL）」，程式卻存成判定字串 "not supported"，依對應規則只有 FAIL 會得到它（實作補記三第 16 項記錄了這個不一致但沒有解決），MARGINAL 時「是否如預測」可以兩邊解讀。新：每個預測是點的類別集合：supported 為 {PASS}；not supported 為 {FAIL, TRIVIAL}，也就是對應到 not supported 的類別；H-GW4 為 {MARGINAL, FAIL}，照本文的字面。analysis.json 的每個點 hypothesis 寫出 `predicted_classes` 與機械判定的 `prediction_met`（NT 為 null），不再寫 `prediction` 字串；H-W0 仍寫 `prediction: "supported"`，另加 `prediction_met`（NT 為 null）。判定（verdict）不變。§12（本文附錄與 PREREGISTRATION.md）已補上這一點。
12. **凍結證據也綁住 profile 與 dataset**（`stage_v_plan.freeze_evidence_problems`、`mini_dataset_keys`）：study profile 不在請求路徑閉包內，舊規則只比對 mini 的 harness_sha256 與 gateway_path_sha256，所以 mini 之後改了 profile（並更新 pin）或 dataset，證據仍然有效。新：mini 的 session.json 中，6 個點的 `profile_sha256` 必須等於 stage_v.toml 目前的 pin，`dataset_sha256` 必須剛好是 mini 讀的 4 個 dataset（mixed-8192、v1-mini-8192、v4-mini、var-2048）而且等於 pin。測試：任一點或任一 dataset 不同、缺少、多出，以及 mini 之後 pin 改變，都讓凍結證據失效。
- **突變檢查**（只在記憶體中改模組原始碼；腳本留在本機 scratchpad，未納入 repo，突變清單如下）：第 5–12 項共 40 個突變全部被測試抓到，包括：任何 client 都接受 /tokenize、接受零筆或兩筆、任何狀態、任何時間或剛好同時、任何 method、帶 id、算成 warmup、`random_dataset` 恆真或含 var；replay 或 runner 改回單位規則或從不跳過；mini 的逾時改回 r2 或換到 L-U8K、skip_walked 空洞、不看 skipped、算到同一個 block；只讀有 session.json 的目錄、忽略 in_progress、不寫暫時的 session.json、finally 不清 in_progress、拒絕時不記 aborted、report 不查；凍結證據不比 profile 或 dataset、dataset 清單忽略 MINI；mini 保留 gateway log；峰值不含 warmup、U_W 含 warmup、run_valid 不查 warmups、不查個數或單位、id-less 紀錄不要區間、runner 不記 warmups；H-GW4 改回 not supported、NT 判為如預測或未如預測。

## 實作補記七（2026-10-24，凍結前第二輪審查；沒有任何 Stage V 數據）

第二輪凍結前審查確認一個必須在 mini 之前修正的問題（第 1 項）與三個小問題（第 2–4 項），依 TDD 修正：先寫會失敗的測試，再改程式，並以 in-process 的突變確認新規則有測試抓得到。只改 harness、測試、設定檔的註解與文件，不改 `service/capserve/`，所以 `gateway_path_sha256` 不變；`harness_sha256` 改變（`stage_v_analysis.py`、`stage_v_datasets.py`、`run_stage_v.py`、`stage_v_report.py`），mini 還沒跑，沒有任何證據失效。凍結模組一個位元組都沒有改，研究程式仍不 import capserve。

1. **凍結的 client 不送 ready check**（`stage_v_analysis.join_access_log`、`warmup_problems`）：
   - 原始碼（scratch 副本 `serve.py`，sha256 `ec255c3b…eec142`）：`--ready-check-timeout-sec` 在第 1954–1959 行預設為 0，說明文字寫明 ready check 預設略過；`main_async` 把 argparse 的值傳給 `benchmark()`（第 2262 行）；`benchmark()` 只在 > 0 時呼叫 `wait_for_endpoint`，否則印出 `Skipping endpoint ready check.`（第 870–886 行）。之後只有 C 個 warmup 送 row 0 的 `test_input`（第 888–910 行）。`benchmark()` 關鍵字參數的 600 s 預設值不會用到，因為 CLI 一律傳入 argparse 的值。
   - argv：mini、V-ADM、V-CLS 每個點與 block 的 `stage_v_plan.bench_argv` 都沒有這個旗標，而且都是 `--num-warmups C`（`core_b_plan.bench_argv` 也是如此）。
   - 實測（本機 archive，vLLM 0.30.0）：`results/capacity/c0-step0/raw/formal-20261011-adv/*.bench.log` 九個檔案的 Namespace 都是 `ready_check_timeout_sec=0`，都印出 `Skipping endpoint ready check.`，接著是 `Warming up with 8 requests...`；adv 的 server log 為 208 = 200 + 8 筆 completion。Core B 的 `formal-20261009-coreb-uniform/server-uniform-8192-r1.log` 中，從每一行 `POST /tokenize` 到下一行（最後一段到檔尾）的 13 段，剛好都是 N + C 筆 `POST /v1/completions`（201、202、204、207、208、209、213、216、218、286、352、704、1408，即 max(10C, 200) + C），從來不是 N + C + 1；第一行之前沒有 completion。
   - 舊規則（本文 (h)、實作補記三第 5 項、實作補記六第 9 項）要求沒有 id 的 completion 紀錄 ≥ C + 1、row 的 warmup 區間 ≥ C + 1。真的 bench 只送 C 筆，所以每個有 bench 結果的 V run 都會無效，並記為 gateway fault；逐點跳過之後，每個點在 block 1 之後就不再測，mini 不會通過（不會有凍結證據），正式 session 的每個點都會是 NT，停止規則也會在 V-ADM 之後停下。測試全部通過，是因為三個假的序列（`tests/test_stage_v_analysis.log_text`，`test_run_stage_v` 的 World 也用它；`service/tests/test_stage_v_settings.py` 的真 gateway 交叉比對）都手寫了一個 bench 從來不送的 ready check。第三個測試本來就是為了抓這種偏離，卻建立在同一個錯誤前提上。
   - 新規則：沒有 id 的 completion 紀錄剛好 `core_b_plan.num_warmups(C)` = C 筆；少一筆或多一筆（不明的額外 request）都是 (h) 的問題。用「剛好」是安全的：vLLM 0.30 的 request 函式不重試，沒有 ready check 時也沒有別的東西會送沒有 id 的 completion。其他規則不變：status 200 / outcome ok、t_arrival_s ≤ t_sent_s ≤ t_released_s（數字）、都在第一個主要 request 到達之前。`warmup_problems` 要求 row 剛好有 C 個區間。random dataset 的 `POST /tokenize` 說明改為「在 warmup 之前」。
   - 不改 client argv：加上這個旗標會違反「只允許三種差異」。新的 plan 測試（`test_the_frozen_client_sends_its_c_warmups_and_no_ready_check`）把個數綁在 argv 上：mini、V-ADM、V-CLS 的每個點與 block，凍結的 argv 不帶 `--ready-check-timeout-sec`（或設為 0），而且 `--num-warmups` 為 C。
   - 三個假的序列都刪除 ready check：`log_text` 與 `warmups_of`（World 隨之改變），真 gateway 交叉比對改為期望 C 個區間。對應失敗的測試：C − 1 與 C + 1 都是問題；刪除「ready check 重試」的接受；「/tokenize 不算 warmup」改用新的訊息。
   - 已在原處更正並加註：本文 (h) 的兩處、(j)、client argv 第 3 項、測試清單、設計審查改正的項目、「揭露」的 WSL 複核清單（加上 ready check 預設略過）；實作補記三第 1、5、6、7 項；實作補記六第 5、9 項；`stage_v_datasets.py` 的說明（只有 warmup 重送 row 0）。PREREGISTRATION.md §12 與 NEXT.md 沒有這個前提。024 的文件、`run_adv` 與 `adv_decomp` 把沒有 prefix 的 span 稱為「warmup 與 ready-check」，預設有 ready check；`formal-20261011-adv` 同樣沒有送（見上）。分類與 024 (b) 的計數只看實際收到的 span，不受影響。這些都是凍結的檔案（`test_frozen_untouched`），不屬於 Stage V，不改。
2. **`configs/stage_v.toml` 的註解**：`[stage_v.points.GW-U32K]` 上方對 unit 的說明仍是舊的單位跳過規則（「"" 表示單獨跳過的點」）。改為：點所屬的 `[stage_v.units]` 產品單位，沒有則為 ""；每個點都單獨跳過（實作補記六第 6 項）。只改註解，值不變（以 tomllib 確認解析結果相同）。stage_v.toml 不在 `HARNESS_FILES` 中，它的位元組由凍結那一步 pin。
3. **bench 之前 gateway 結束也記為 gateway fault**（`run_stage_v.run_point`）：實作補記四第 4 項把「bench 之前或期間 gateway 結束」列入 `session["gateway_faults"]`，但只有「期間」有記錄（`_after_bench`）。run_point 的「bench 之前 gateway 已結束」分支現在在 decide 之前呼叫 `_fault(<boot tag>, 原因, run=<run tag>)`。測試：假 session 中 F-U2K r1 的 gateway 在 bench 之前結束時，gateway_faults 剛好有一筆 {boot, run, reasons}；沒有這個呼叫時測試失敗。實作補記四第 4 項已加註。
4. **report 只看在本 session 開始時或之前開始的目錄**（`stage_v_report.refusal_problems`；`run_stage_v.hours_used` / `gap_problems` 新增的 `not_after`）：
   - 舊：report 的 `hours_used` 與 `gap_problems` 讀 `results/stage_v` 下的每一個目錄，所以一個之後的、或無關的未結束 `formal-*` / `smoke-v-*` 目錄，會讓較早 session 的 report 拒絕；產生器唯讀重跑 report 並要求逐位元相同（產品對應），也會因此失敗。
   - 新：report 只檢查 session.json 的 started_at 不晚於本 session started_at 的目錄。沒有 session.json、讀不了、沒有 started_at、或 started_at 不是可比較的 ISO 時間的目錄仍然檢查（fail closed）；本 session 沒有可讀的 started_at 時，照舊檢查全部並拒絕。runner 不傳 `not_after`，照舊檢查全部目錄，開始前的拒絕不變。
   - 測試：之後開始、未結束的目錄（formal 與 smoke-v）不擋 report，但 runner 的 `hours_used`（formal 另加 `gap_problems`）仍回報；較早或同時開始、未結束的目錄仍讓 report 拒絕；沒有 started_at、started_at 為 null、不是時間、不是字串、session.json 不是 JSON、沒有 session.json，都仍讓 report 拒絕。
   - 實作補記四第 9 項、實作補記六第 8 項與 `stage_v_report` 的說明已加註。
- **突變檢查**（只在記憶體中改模組原始碼；腳本留在本機 scratchpad，未納入 repo，突變清單如下）：第 1–4 項共 20 個突變全部被測試抓到：沒有 id 的筆數改回 C + 1、只要求至少 C、只要求至多 C、不檢查個數、把 /tokenize 算成 warmup；row 的區間改回 C + 1、至少 C、至多 C、任何個數；client 加上 `--ready-check-timeout-sec 600` 或 `=5`；bench 之前 gateway 結束時不記 fault、記錄時不帶 run；report 的 hours_used 或 gap_problems 改回檢查全部目錄、report 不查 hours_used、讀不到 started_at 的目錄被略過、同時開始的目錄被略過、可讀 started_at 的目錄全部被略過、`not_after` 被忽略。

## 實作補記八（2026-10-24，第二輪修正後的獨立檢查；沒有任何 Stage V 數據）

第二輪修正（實作補記七）提交後，再由兩個獨立的檢查分別做程式的突變與真 gateway 序列、文件的逐段比對。沒有新的 must-fix；以下兩項依 TDD 修正，其餘是文字更正（已在原處改好：實作補記四第 9 項、實作補記六第 8 項、設計審查改正的項目、實作補記七第 1 項的 Core B 分段與 024 的說明、兩處突變腳本的引用）。只改 harness、測試與文件，`gateway_path_sha256` 不變；`harness_sha256` 再次改變（`run_stage_v.py`），mini 還沒跑，沒有任何證據失效。

1. **session 時間必須可信**（`run_stage_v._times`，`hours_used` 與 `gap_problems` 都經過它）：實作補記六第 8 項要求操作者為被強制結束的 session 手寫 finished_at。時區寫錯時（例如 10:00Z 開始、17:00Z 結束，寫成 `17:00+08:00`，即 09:00Z），該 session 的時數是負的，12 h 全期上限少算，2 h 間隔也會在實際只隔 1 h 時通過，兩條規則都沒有任何提示。現在 started_at 或 finished_at 沒有時區 offset，或 finished_at 早於 started_at，都和未結束的目錄一樣是問題（runner 拒絕開始，report 拒絕）。runner 自己寫的時間一律帶 offset，不受影響。測試：錯誤時區、任一個或兩個沒有 offset，都讓 hours_used 與 gap_problems 各回報一個問題、runner 拒絕；同一時刻的不同 offset（0 h）可以。原本「沒有時區時時數照算、只有間隔無法判斷」的測試改為兩者都回報問題。突變：拿掉 offset 檢查或順序檢查，測試都失敗。
2. **client argv 的 ready check 測試也看縮寫**（`tests/test_stage_v_plan.py`）：vLLM 的 `FlexibleArgumentParser` 保留 argparse 的前綴比對，並把 `_` 換成 `-`，所以 `--ready 600`、`--ready_check_timeout_sec=5` 也會打開 ready check。測試改為把每個 `--` 參數正規化後，凡是 `--ready-check-timeout-sec` 的前綴都必須取值 0（或不出現）。突變：在 argv 加上 `--ready 600` 或 `--ready_check_timeout_sec=5`，測試失敗；加上 `--ready-check-timeout-sec 0` 則通過。

## 實作補記九（2026-10-24，產生器的 validation-session 路徑；沒有任何 Stage V 數據）

`service/capserve/profile_gen.py` 的 `--validation-session` 路徑與 `service/tests/test_profile_gen_stage_v.py` 已依「產品對應」以 TDD 實作（時序第 8 步，在第一個正式 V session 之前；Mac 工作，不用 GPU）。產生器在請求路徑閉包之外（`__main__` 的 AST 閉包不含 profile_gen），`gateway_path_sha256` 仍為 `679a7c66…`；harness 一個位元組都沒有改，`harness_sha256` 仍為 `a1e4a564…`；研究程式仍不 import capserve。測試的 session 全部是合成的：由真的 `run_stage_v.run_session` 在 `test_run_stage_v` 的假 WSL 中寫出（`test_stage_v_analysis.fail_kw` 讓指定的 run FAIL），analysis.json 由真的 `stage_v_report` CLI 產生，所以 units、provenance、platform 的形狀都是真的。git 以假的回答代替（凍結 commit、它的 stage_v.toml、凍結模組、DEPLOYED_COMMIT 的請求路徑、base profile），git 的 helper 另以真的 repository 測試。下列是本文沒有寫死、實作時決定的細節，以及本文和程式不一致、依程式（§12 schema，實作補記一第 9 項）處理的地方。

1. **CLI**：`--validation-session` 和 `--session` 互斥、擇一必填，只能配 `--out-dir`（給 `--out` 時 exit 2）。`--session` 指向 `results/stage_v/` 時 exit 2，訊息指向 `--validation-session`。選項只多這一個，沒有單位或 class 的選擇器（021 R1；`test_no_override_options` 已更新）。
2. **沒有單位成立時 exit 0**：session 重現成功、只是沒有會寫檔的單位成立（全部不成立，或只有 VG-32K / VG-M8K-T）時，這是結果，不是拒絕：不寫任何檔、exit 0，stdout 的摘要 `{"holds": [...], "profiles": [], "written": "no profile: …"}` 說明原因。拒絕一律 exit 2、不寫檔，和 §8 / §9 相同。
3. **只由 units 決定寫什麼**（`holding_units`）：單位只有在 `holds` 為 true、每一點的類別都是 PASS、而且（PACE-2K）沒有被否決時才寫。三個條件在可重現的正式 session 中等價，產生器仍逐一檢查。本文「只讀 analysis.json 的 units」解讀為「寫什麼只由 units 決定」：provenance 與 server 的值取自同一份 analysis.json 的 `provenance`、`platform`（report 為產生器寫的段），fingerprint 取自 session.json 的 boot 紀錄，和 §9 相同。
4. **單位 → 內容**（依 §12 schema）：
   - PACE-2K / PACE-8K：c1b（`COPY_BASE`）的 uniform-2k / uniform-8k 逐欄原樣，加上 `pacing`（單位 validation 的 pacing 與 queue）與 validation。
   - W1：`admission` 段 = 單位 validation 的 weighted-v1 參數，其中研究 class `v-uniform-2k` / `v-uniform-8k` 改為 c1b 的複本 `uniform-2k` / `uniform-8k`（`STUDY_COPIES`），加上 queue、`measured_mixes`、provenance 的 `gateway_path_sha256` 與 V4 的 validation。PACE 單位不成立的成員是 c1b 的 class 原樣（不帶 pacing、validation）。PACE-2K 與 W1 都成立時，uniform-2k 同時是帶 pacing 的複本與 admission 的成員：loader 接受，pacing 與 weighted-v1 永遠不同時開啟，`served_classes` 依設定只比對其中一段。
   - V1-2K / V1-8K：d-a2-b0（`V1_BASE`）的 uniform-2k-o512 / uniform-8k-o512，改名為 uniform-2k-o1-512 / uniform-8k-o1-512，max_concurrency = max_queue = `V1_C`（32 / 8），bound `validated`，source `stage_v:<單位>`。evidence 是 O512 class 的 evidence 原樣（scope、C_PROD 12 等描述的是 D-A2 的量測），`not_covered` 以 `v1_not_covered` 重算，另加 `output_distribution`、L 點的 5 個 dataset sha256（單位 validation 的 `datasets_sha256[L 點]`；GW 點輸出固定，沒有逐 block 的 dataset）與 `via_gateway` true。
   - VG-32K / VG-M8K-T：不寫檔（`NO_FILE_UNITS`）。
   - validation 取單位 validation 的 {study, unit, points, measured_classes, worst_r, n}（`capserve.profile.Validation` 的欄位）；queue、pacing、admission、datasets、mixes 放在 §12 規定的位置。
   - class 的順序：複本（U2K、U8K），再 V1（2K、8K）。`admission` 一律寫出（不成立時為 null），`server.long_prefill_token_threshold` 寫 null。`base_profiles` 只列實際用到的 base（V-ADM 為 c1b，V-CLS 為 d-a2-b0）。本文的 validation_session 就是既有的 `source_session`（實作補記一第 9 項）。
5. **凍結檢查**（比照 §8 / §9 的 `freeze_commit_problems`）：
   - `--freeze-commit` 是第一次設定 `configs/stage_v.toml` `[freeze] stage_v = true` 的 commit（它的上一個 commit 與更早的歷史都沒有凍結過），是 session DEPLOYED_COMMIT 的祖先，該 commit 的 stage_v.toml 的 sha256 等於 session 記錄、產生器讀到的值。
   - `STAGE_V_MODULES` 在凍結 commit、DEPLOYED_COMMIT 與產生器實際執行的檔案三者相同：`stage_v_plan.HARNESS_FILES` 的 5 個模組，加上 §9 的 Stage D 凍結檔與 5 個 Stage C 模組。本文沒有列出範圍；Stage V 的 harness 會 import 這些 Stage D / C 模組。
   - `stage_v_plan.freeze_evidence_problems` 為空，stage_d.toml 與 capacity.toml 都已凍結（loader 的 §12 規則）。session 的 `harness_sha256`、記錄的 `gateway_path_sha256`、mini 證據與每個 pin 由重跑的 `stage_v_report.refusal_problems` 檢查。
6. **DEPLOYED_COMMIT 的請求路徑**（`deployed_gateway_path_sha256`）：以 git（`ls-tree`、`show`）取出 DEPLOYED_COMMIT 追蹤的 `service/capserve/*`，寫到暫存目錄，由 `capserve.gateway_path` 自己計算 AST 閉包的 sha256，必須等於 stage_v.toml 的凍結值。閉包外的檔案（profile_gen、deploy）不同不影響；閉包內任一檔案不同、或 `__main__` 多 import 一個模組時拒絕。這和 report 檢查的「session 記錄的值」互相獨立，後者是 runner 在 WSL 上自己算的。
7. **base profile 的「committed 位元組」**：解讀為 Stage V 凍結 commit 上的位元組。c1b 與 d-a2-b0 必須逐位元等於 `git show <凍結 commit>:<path>`，而且以 loader 載入為這個 checkout 的 verified profile；study profile 的 v- class 是凍結前由測試對這些位元組鎖定的。工作樹乾淨的檢查另外保證它們等於 HEAD。
8. **重跑 report 的輸入**：uniform 一律讀 stage_d.toml `[stage_d] derived_from` 的 Core B uniform session（Stage V 與 Stage D 共用的凍結值；`stage_v_report` 的 CLI 不限定是哪一個），Advanced 只在 analysis.json 的 `advanced_references` 為 true 時讀。用了別的 uniform session 的 analysis.json 無法逐位元重現，因而被拒絕。
9. **server**：V boot 的 workload 是點的 id，先換成該點的 `server_base`，再以 §9 的 `server_groups` 分組。B0 只能有一個 fingerprint，而且是產品的 B0（`stage_b_fingerprint`）。T 組（GW-M8K-T）也照 §9 檢查，雖然 §12 profile 只寫 B0。
10. **只接受 formal-**：session_id 以 `formal-` 開頭、位於 `results/stage_v/<session_id>`。mini（`smoke-v-*`）由產生器本身、report 與 analysis（`holds` 要求不是 mini）三處拒絕。
11. **重新產生的測試**：`test_profile_gen.regenerate` 依 prereg_section 分派到 `generate_stage_v`，並依 profile_id 選取（實作補記一第 9 項：「重新產生的測試在產生器那一步加入」）。目前沒有 commit 任何 §12 profile；分派與選取以合成 session 的 §12 profile 測試。
12. **服務規則的端到端測試**：寫出的 profile 經 loader 載入後，`capserve.config.served_classes` 只在設定的 pacing（含 queue）或 admission 段與 profile 完全相同、而且 gateway_path_sha256 相同時才服務（serve 與 deploy 都一樣）。不同的 rate、另一個 tier 的參數、產品的 2 s / 300 s 佇列、不同的 max_queue、不同的請求路徑一律 ProfileError。V1 的 class 在沒有候選政策時以 512 輸出服務，帶 pacing 時拒絕。
13. **發現但沒有處理的不一致**：`configs/stage_v.toml` `[stage_v.units]` 上方的註解仍寫舊的單位跳過規則（「once a run of one is not ok, all of them leave the later boots」），和實作補記六第 6 項的逐點規則不符；實作補記七第 2 項只改了 `[stage_v.points.GW-U32K]` 上方那一處。只是註解，值不變；不在本步範圍，凍結前（pin stage_v.toml 的位元組之前）修改。
- **突變檢查**（只在記憶體中改模組原始碼：pytest plugin 在收集之前把改過的原始碼 exec 進 `capserve.profile_gen`；腳本留在本機 scratchpad，未納入 repo）：40 個突變全部被 `test_profile_gen_stage_v.py` 抓到，未突變的基準全部通過。包括：
  - 凍結：產生器與 CLI 兩處的 stage_v、capacity、stage_d；凍結證據；freeze commit 的檢查整段拿掉、只比對 Stage D 模組、模組清單少了 Stage D / C；
  - 請求路徑：拿掉比較、改算這個 checkout、讀不到時不報；
  - base profile：拿掉位元組比較、不存在不報、拿掉 loader、問題不計入、`base_profiles` 一律列兩個；
  - session 與重現：拿掉 report 的拒絕、逐位元重現、寫檔前的 loader 檢查、formal- 檢查、路徑檢查、SLO、server 分組、point 不換成 server_base；
  - CLI：工作樹、out-dir、覆寫、接受 `--out`、`--session` 指向 Stage V、沒有單位時的說明與 exit 0；
  - 單位篩選：不看 holds、不看否決、不看 PASS、全部單位；未知單位；成員一律不帶 pacing；PACE 複本不帶 validation；V1 用 O512 的 C_PROD；不重算 not_covered。
