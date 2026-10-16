# 028 — C0：8K 延遲雙模式的有限度排查（探索性，非正式）

- 狀態：Accepted（計畫；任何 C0 數據之前寫定）
- 日期：2026-10-15
- 對應：decision 027 §2（C0）、025「限制」、026 Future Work
- 時點：Phase 2 開始後，任何 C0 的 GPU run 或 raw 檔取回之前。設計經過一輪設計與一輪對抗式審查（審查改正了 20 項，見文末）。
- 性質：**探索性、描述性**。不是 hypothesis 檢定，不改變任何 Phase 1 的判定，也不修改 025。
  唯一的用途是決定 PREREGISTRATION §8 要採用哪個分支（下文「對 §8 的對應」）。

## 已知線索（只讀已 commit 的資料，沒有新的 GPU run）

來源：
- `results/advanced/formal-20261011-adv/rows.jsonl`：B0 / G / K3 在 8192 uniform 的 9 個 run；
- 這些 run 的 `derived/<tag>-requests.jsonl`；
- `results/core_b/formal-20261009-coreb-uniform/rows.jsonl` 第 40–78 行。

`idx` 是 bench 清單中的序號。在這 9 個 run 中，idx ≥ 8 時 arrival 單調遞增，所以 idx 等於送出順序。

1. **不是速度差異**。9 個 run 的 P50 為 2,476–2,478 ms，TPOT 80.30–80.54 ms，duration 318.0–318.3 s，output 80.44–80.50 tok/s。
2. **封閉迴路的週期**：C = 8 時，每 8 個 request 為一組，週期 12.72 s。
   - 第 8k 個 request 的 TTFT 約 1,485 ms。
   - 第 8k+1 至 8k+6 個約 2,477 ms。
   - 第 8k+7 個約 2,320 ms。
3. **滑移（slip）**：偶爾整組往後滑一格。
   - 這時第 8k+1 個 request 在第 8k 個之後 90–94 ms 就到達。正常的 208 組都是 415–420 ms，滑移的 8 組都是 90–94 ms。
   - 第 8k+1 至 8k+7 個各多排隊 347–351 ms，TTFT 變為約 2,816 ms。
   - 下一組恢復正常，前面的組也沒有任何前兆。
4. **滑移的次數**：共 8 次。
   - G r1：1 次；K3 r1：2 次；K3 r2：1 次；K3 r3：2 次；B0 r2：2 次。
   - 其餘 4 個 run 都是 0 次。
5. **P95 為什麼會翻轉**：
   - `--request-rate inf` 讓開頭的 7 個 request 一起湧入，每個 run 都有 7 個 TTFT 落在 2,814–11,043 ms。
   - N = 200 時，線性內插的 P95 約為第 11 大的值，所以只要有一次滑移，這個 run 的 P95 就會變高。
6. **位置集中**（事後發現，屬推測）：8 次中有 6 次從第 k = 8、16、24 組開始。若位置隨機，機率約 8.5 × 10⁻⁵。
7. **Advanced 的 queue-time histogram 完全對應滑移**：(0.3, 0.5] s 區間的計數一定是 7 × 滑移次數。這個 histogram 只有在開啟觀測 flag 時才有記錄。
8. **與 C 有關**：Core 中 C = 4 與 C = 7 的 6 個 run，P95 − P50 都是 330–364 ms；C = 9 只有 r2 偏高。
9. **目前的資料無法區分 boot 層級與 run 層級**：Advanced 每次 boot 只跑一個 bench；Core 在每次 boot 中，C = 8 也只跑一次。
10. env 的擷取都是閒置時的快照，看不出和溫度或 pstate 有關。`clocks_event_reasons` 在 WSL 下的意義尚未驗證。

## 步驟 0：用既有 raw 檔做桌上檢查（不用 GPU）

- **先 commit，再取回**：`benchmarks/c0_step0.py`、`tools/rtx/fetch_raw.sh` 與下列檢查都先 commit，之後才從 WSL 唯讀取回 raw 檔。
  這樣 S1 才能算是對線索 6 的重現，而不是看過資料後再定規則。
- `fetch_raw.sh`：
  - 先用唯讀的 `ls` 確認 WSL 上的 raw 封存存在；
  - 只取回列出的檔案，放到 gitignore 的區域；
  - 依 `raw_manifest.json` 驗證 sha256；
  - 不碰 GPU lock。
- **檢查**：
  - **S1（Core）**：對 `bench-uniform-8192-r{1,2,3}-c{4,7,8,9,13}.json` 套用下文的分類器，記錄每次滑移的起始組、長度與位置。
    事先寫定的重現檢查：C = 8 與 C = 9 的滑移中，從 idx ≡ 1（mod 64）開始的比例，與隨機的 1/8 相比。
  - **S2**：Core 中 C = 4 與 C = 7 偏高的 run 是否也是滑移造成。
  - **S3（Advanced 的 `steps.jsonl`）**：滑移組的 step 時間和鄰近組比較；多出來的排隊時間是否等於一個 step。
  - **S4（Advanced 的 `server.log`、`bench.log`）**：滑移時刻附近是否有週期性的 log；高低兩種 boot 的啟動訊息（compile cache、CUDA graph 大小）是否不同。
  - **S5（Core 的 `server-uniform-8192-r{1,2,3}.log`）**：Core r2 在所有 C 都慢約 1%，啟動訊息是否不同。
- **輸出**：
  - 描述性的結果寫到 `results/capacity/c0-step0/`。
  - 不寫入任何 Phase 1 的目錄，不重算任何 `analysis.json` 或判定（027 §1）。
- 若 WSL 上已經沒有 raw 封存：照實記錄，直接進入 GPU 探針。C0 的目的不需要找出機制。
- 步驟 0 可以在 GPU 探針之前補一項診斷，但探針開始之後不能再改。

## GPU 探針

**設定（只有一組，與 C1 相同）**
- Server：`core_b_plan.server_argv`，K = 4 GiB（248 個 block）、max-num-seqs 128、MNBT 2048、U 0.91、triton GDN prefill。
  - 不加觀測 flag。C1 沿用 Core 的 argv（027 §2），而機制資料已經在 Advanced 的 raw 檔中。
  - prefix caching 維持關閉，所以同一次 boot 重複跑相同的 prompt 仍然有效。
- Client：`core_b_plan.bench_argv`，L = 8192、uniform、C = 8、N = 200、8 個 warmup、seed 20261008。

**規模**
- 6 次 boot，每次 boot 連續跑 4 個 bench，共 24 個。只有一組，所以不需要隨機排列；記錄每個 bench 在 boot 中的位置。

**每次 boot 都相同的量測**
- Core 的 `run_core_b.MetricsSampler`（1 Hz）。
- 每個 bench 前後各一次完整的 `/metrics` 擷取。
- 每次 boot 一個 `nvidia-smi` 取樣 process（1 Hz）。
  - 欄位：pstate、clocks.sm、clocks.mem、power.draw、temperature.gpu、utilization.gpu、memory.used、clocks_event_reasons.active。
  - 限制：Phase 1 沒有這個取樣，它本身可能影響正在觀察的毫秒級競爭，所以要和 Phase 1 的偏高比例（12 個 run 中 8 個）比較。
- server log 每行加上毫秒時間戳（`run_adv.LogReader`），擷取 non-default args、backend、compile cache、CUDA graph、KV 記憶體、啟動時間、WARNING / ERROR。
- `--save-detailed` 的逐 request TTFT 與 ITL。
- 每次 boot 前後各做一次 env 擷取，並記錄每個 bench 的開始、結束時間與位置。

**smoke**：在同一段 lock 時間內先跑 1 次 boot、1 個 bench（N = 16、C = 8）。輸出寫到 `results/capacity/smoke-c0-<date>/`，不分析。

**預算**
- 每次 boot 約 24 分鐘：啟動約 32 s，4 個 bench 各約 330 s，env 擷取與 teardown 約 60 s。
- 預期總計約 2.5 h。**硬上限：持有 lock 3.0 h，含 smoke。**

**停止規則**
1. `flock -n` 失敗就不開始，標記 blocked。不搶占，不刪除 lock。
2. 每次 boot 前，下列條件都要成立，否則停止：
   - `run_adv.held_gpu_lock()` 為真；
   - GPU 上沒有其他 compute process；
   - 60 s 內 memory.used ≤ 1,000 MiB。〔2026-10-16 起由 decision 031 的綜合判定取代：lock、使用率 ≤ 5%、基線穩定、由實測 vLLM 峰值推導的 VRAM 餘裕（目前 1,018 MiB），並在 boot 期間以 Windows 的 shared usage 偵測 spill。使用者確認，任何 C0 數據之前。〕
3. 已用時間加 30 分鐘會超過上限時，不再開始新的 boot。
4. bench 達到 720 s 逾時（pins `[advanced].timeouts_s.uniform_8192`）就終止，關閉 server，該 bench 無效。
5. **無效的 bench**：exit code ≠ 0 或逾時、completed ≠ 200 或 failed > 0、bench 後 server 不健康、block 數 ≠ 248、non-default args ≠ {MNBT 2048, K}。
   - 無效的 bench 一律排除，不補跑。
   - 啟動失敗，或有 2 個以上無效 bench 的 boot，算失敗。失敗 2 次就停止。
6. GPU 溫度 ≥ 88 °C 時立即關閉並停止。
7. 不依數據提前停止或延長。任何追加都要新的 decision log。
8. session 期間 GPU 上不跑其他東西，也不部署任何東西。
9. driver、vLLM 或 torch 的版本和 Phase 1 不同時，照實記錄並繼續（Phase 1 的 driver 為 617.42）。

**禁止**：鎖定時脈、修改 power limit 或 persistence mode、Nsight，以及任何 argv 的變更。

## 描述性分析（事先寫定）

**每個 bench**
- Y = `core_b_analysis.short_class_p95_ttft_ms`。
- m_A = idx ≥ C 的 TTFT 中位數。
- b_j = idx ≥ C 且 idx ≡ j（mod C）的 TTFT 中位數，也就是每個位置自己的基準。
  用每個位置自己的基準，是因為第 8k+7 個 request 的滑移值（2,659–2,662 ms）只比整體門檻高 11–16 ms。
- **滑移**：idx ≥ C，且 TTFT − b_(idx mod C) > 170 ms。
- **滑移段**：連續的滑移 request。記錄起始組 k、長度與位置。
- **k_prom**：bench 前後 queue-time histogram 的 (0.3, 0.5] s 區間的計數差。檢查它是否等於滑移的 request 數。
- **模式**，依序判定：
  1. Y − m_A > 600 ms：other；
  2. Y − m_A > 170 ms：high；
  3. 其他：low。
- 另外記錄：
  - 去掉前 C 個 request 後的 P95、P50、TPOT、throughput；
  - 每個 bench 的遙測：SM 時脈的最小值與中位數、出現過的 pstate、throttle 位元的 OR、最高溫度。

**每次 boot**：n_b = 有效 bench 的滑移段總數；P50 的中位數。

**整個 session**（依序取第一個符合的）
- **R0（沒有重現）**：滑移段總數為 0。
- **Rboot（boot 層級的差異）**，下列任一成立：
  - 分散度 D = Σ(n_b − n̄)² / n̄（只算 4 個 bench 都有效的 boot）> χ²₀.₉₅(boot 數 − 1)。6 次 boot 時門檻為 11.07。
  - 某次 boot 的滑移段為 0，而 session 平均每個 bench ≥ 1 段。
- **Rrun**：其他情況，符合 run 層級。

**只報告、不作判定**
- 同時出現兩種模式的 boot 數；
- 第 1 個 bench 與第 2–4 個 bench 的起始組分布：起始組相同，暗示由 client 或時間鎖定的觸發；起始組不同，暗示 server 狀態；
- k ∈ {8, 16, 24} 的集中現象是否重現；
- 偏高比例與 Phase 1 的 8/12 比較，用來看取樣是否改變了現象。

**旗標**
- **Fhw**：有 throttle 位元，或 SM 時脈低於該 boot 中位數 × 0.95 的 bench，與滑移交叉列表。
- **Foff**：各 boot 的 P50 中位數相差 > 1%。

**檢定力**：若各 boot 相同（約每次 boot 3.6 段），D 大約只能偵測 ≥ 5 倍的 boot 間差異。更小的差異照實報告為「無法偵測」。

## 對 §8 的對應（C0 只決定這一件事）

結果寫進 `configs/capacity.toml` 的 `c0_branch`，並在 §8 凍結之前另寫一份結果的 decision log。

| C0 結果 | §8 分支 | §8 的變更 |
|---|---|---|
| R0 或 Rrun | A | 預設協定 |
| Rboot | B | 受影響的 workload（至少 uniform-8K）每個關鍵點的 restart 從 5 增為 7；每個 run 的滑移數記為共變量 |
| 超過上限、失敗的 boot 過多，或無法判定 | C | 同 B，只套用在 uniform-8K |
| 發現 harness 或 client 的缺陷 | D | 先寫 decision log 修正，§8 在那之前不凍結 |

- 任何分支都不改變 SLO、時間窗、統計量、guard band、上限、搜尋與確認規則、workload 或 hypothesis，也不減少 restart。
- 不依模式排除任何 run，因為排除會讓容量偏高。

## 實作

- **`benchmarks/run_c0_probe.py`**（子指令 `plan`、`smoke`、`run`）。只 import 凍結的模組：
  - `core_b_plan`：server_argv、bench_argv、server_env、toolchain_problems；
  - `run_adv`：start_server、LogReader、held_gpu_lock、nvidia_mem_used_bounded、scrape_deltas、default_run_bench、DeferredSignals、shutdown_group、parse_nondefault、blocks_problems；
  - `run_core_b`：MetricsSampler、summarize_result、make_vllm_env_capture、stage_b_problems；
  - `run_core_a`：env_problems、sha256_json、write_raw_manifest。

  它只新增：
  - 每次 boot 跑多個 bench 的迴圈；
  - nvidia-smi 取樣；
  - 啟動訊息擷取；
  - non-default args 檢查（恰好是 MNBT 與 K）；
  - 上限與停止邏輯。
- **`benchmarks/c0_analysis.py`**（只用 stdlib）實作上文的描述性規則，拒絕 stage 不是 "C0" 的 session。
- **`benchmarks/c0_step0.py`** 與 **`tools/rtx/fetch_raw.sh`**：步驟 0。
- **測試**（Mac，用假物件）：
  - plan 的決定性；
  - 上限與停止規則，沒有 lock 時拒絕；
  - argv 等於 Core 的 argv，沒有觀測 key；
  - 分類器以 Advanced 的 derived 檔為 fixture：要找到 8 段、起始組與上文相同、滑移數等於 k_prom、第 8k+7 個由位置基準抓到。
  - `test_core_untouched` 與 `test_frozen_untouched` 維持通過。
- **輸出**：`results/capacity/explore-c0-<YYYYMMDD>/`。
  - `session.json` 含 `{formal: false, stage: "C0", purpose: "exploratory", decision: "028"}`。
  - 另有 plan.json、rows.jsonl、derived/、env/、raw_manifest.json、analysis.json；raw/ 不進 git。
  - C1 的分析程式拒絕 stage ≠ "C1" 或 formal ≠ true 的 session。
- **執行順序**：
  1. commit 本文與實作；
  2. 在 session 開始前部署並核對 `DEPLOYED_COMMIT`；
  3. 確認 `tools/rtx/gpu_status.sh` 回報 FREE；
  4. 在 ssh 前景、`flock -n /tmp/ai-portfolio-gpu.lock` 下執行 `run_c0_probe.py run --smoke-first --cap-hours 3.0`。

## 設計審查改正的項目（摘要）

- 滑移數是 8 次，不是 9 次；機率改為 P(3/3 偏高) = 0.20。
- 「多出一個 scheduler step」尚未證實，因為 step 的時間只在 raw 檔中；改列為 S3 的檢查。
- 「沒有觀測 flag 時也看得到模式」尚未觀察過，要由探針驗證。
- 分類器改用每個位置自己的基準；模式的判定順序寫死（other 先判定）。
- 加入分散度 D，才能區分 boot 層級與 run 層級；只看「同一 boot 出現兩種模式」無法區分。
- 刪除觀測 flag 那一組：它是假性重複（3 對 3 個 boot），而且 C1 已經固定使用 Core 的 argv。
- 新增：nvidia-smi 取樣的限制說明、smoke 步驟、driver 版本檢查、無效 bench 的規則、改用 Core 的 MetricsSampler、
  步驟 0 只作描述並寫到 Phase 1 目錄之外、raw 檔的唯讀取回。

## 實作補記（2026-10-16；任何 C0 數據與 raw 檔取回之前）

harness 依本文以 TDD 實作，經一輪審查（6 項，4 項修正、1 項依下文改寫本文、1 項接受）後確定以下細節，與本文一起生效：

- **S1 的重現檢查只用 C = 8**：
  - 原文「C = 8 與 C = 9 的滑移中，從 idx ≡ 1（mod 64）開始的比例，與 1/8 相比」寫錯了。
  - C = 9 時，N = 200 以內的滑移起點 9k + 1 沒有任何一個 ≡ 1（mod 64），所以 1/8 不適用。
  - 改為只看 C = 8：第 1–24 組中，從第 8、16、24 組開始的比例，與 1/8 相比，附 binomial 上尾機率。
  - C = 9 的滑移起點只作描述（`c9_start_idx`）。
  - 這項修改在 `c0_step0.py` commit 與 raw 檔取回之前完成。
- **Rboot 的「空 boot」規則**：
  - 「只算 4 個 bench 都有效的 boot」只修飾分散度 D。
  - 「某次 boot 的滑移段為 0」看的是每個至少有 1 個有效 bench 的 boot。
  - 完全沒有有效 bench 的 boot 沒有觀測值，不算空 boot。
- **新增停止規則 `telemetry`**：88 °C 的停止規則完全依賴 nvidia-smi 取樣。
  - 取樣沒有啟動、process 結束，或連續 30 s 沒有溫度讀數時，處理方式和超溫相同：中斷 bench、關閉 server、停止 session。
  - smoke 要求取樣有溫度讀數、沒有取樣錯誤，否則不開始探針。
  - 因 telemetry 停止，屬於「無法判定」，也就是分支 C。
- **teardown 期間觸發的停止旗標**也會停止 session。
- **3.0 h 上限也約束 server 啟動的等待**：等待時間取啟動逾時與上限剩餘時間的較小值；被上限截斷時以 `cap` 停止。
  - teardown 一律執行，不因上限而跳過，所以 lock 的持有時間最多可能超過上限約 80 s（最壞情況的關閉時間）。
- **輸出路徑**：`run_c0_probe` 與 `c0_step0` 共用同一套規則，拒絕寫進任何 Phase 1 目錄。

## 步驟 0 的結果（2026-10-16；描述性，不改變任何 Phase 1 的判定）

raw 檔已從 WSL 唯讀取回（Core 18 個、Advanced 27 個，sha256 全部和 `raw_manifest.json` 相符；放在 gitignore 的 `results/capacity/c0-step0/raw/`）。
輸出：`results/capacity/c0-step0/step0.json`。

- **S3 的修正（取回 raw 檔之後、GPU 探針之前）**：
  - vLLM 0.30 log 中的「iteration elapsed time」約 0.01–0.02 ms，量的不是 step 的實際耗時。
  - 改用相鄰兩個 step 的 log 時間戳之差作為 step 的實際時間；原本的值另外保留為 `log_elapsed_ms`。
  - 這是量測方式的錯誤，不是規則的改變；有回歸測試。
- **S1（事先寫定的重現檢查）：沒有重現**。
  - Core 在 C = 8 的滑移共 8 次，只有 1 次從第 8 的倍數組開始，比例 0.125，等於隨機的 1/8（binomial 上尾 p = 0.66）。
  - Advanced 中「集中在第 8、16、24 組」的現象（線索 6）應視為巧合。
- **S1（描述）**：Core 每個 run 的滑移段數：

  | C | r1 | r2 | r3 |
  |---|---|---|---|
  | 4 | 8 | 15 | 14 |
  | 7 | 1 | 9 | 1 |
  | 8 | 3 | 3 | 2 |
  | 9 | 0 | 2 | 0 |
  | 13 | 0 | 1 | 0 |

  - r2 在 C = 7 與 C = 9 明顯較多。r2 也是全面慢約 1% 的那次 boot（線索 8）。這暗示可能有 boot 層級的成分，但只有 3 次 boot，要由 GPU 探針判斷。
  - C = 9 的滑移起點 idx 為 19、73。
- **S2**：Core 在 C = 4 與 C = 7 的 6 個偏高的 run，全部是滑移造成。去掉滑移的 request 後，P95 − m_A 只剩 11–31 ms。
- **S3（機制）**：
  - 8 次滑移中，多出的排隊時間約 349 ms，等於該組 prefill step 實際時間的中位數，比值 0.990–0.997。
  - 滑移的組比鄰近組多一個 step（135 對 134）。
  - 也就是說，第 8k+1 個 request 提早到達（相隔 90–94 ms，不是 415–420 ms），錯過了正在進行的 2048-token prefill step，於是整組多等一個 step。
    這是 request 到達時機與 step 邊界之間的競爭。
- **S4**：滑移前後 1 s 內的 log，只有每秒一次的 `/metrics` 擷取。非滑移的組也都有，沒有找到其他週期性的事件。
- **S5**：Core r2 的啟動訊息和 r1、r3 只差在 CUDA graph 的擷取時間（7 s 對 6 s）。non-default args、backend、compile cache、KV 記憶體都相同。

這些結果只是探索性的描述，不改變 GPU 探針的設計（6 次 boot × 4 個 bench），也不改變 §8 的任何規則。

## 執行紀錄

- **2026-10-16 01:10，第一次嘗試**（部署 `ac29c54`，GPU lock 下）：smoke 在第一次 boot 之前，依停止規則 2 停止：
  memory.used 1,362 MiB，60 s 後仍 > 1,000 MiB。沒有啟動任何 vLLM，探針沒有開始。
  - Phase 1 的閒置用量為 688 MiB（Advanced 的 preflight）。多出的部分來自 Windows 端的桌面應用程式（`nvidia-smi` 列出 Edge WebView2、ChatGPT、設定、BingWallpaper、codex-computer-use、Radeon Software 等 C+G process）。
  - **不放寬門檻**：Phase 1 中 vLLM 就緒時整張卡用到約 14.2–15.3 GB。閒置用量增加後剩下的空間只有幾百 MiB，WDDM 不夠時會 spill 到系統記憶體而不是報錯，會悄悄改變延遲。
  - 遠端保留 `results/capacity/smoke-c0-20261016`（stage "C0-smoke"，記錄了停止原因），不分析。
  - 之後改為自動等待：每 5 分鐘唯讀檢查一次，lock 空著而且閒置用量 ≤ 950 MiB 時才開始（新的 smoke 目錄名稱），最多等 24 h。
