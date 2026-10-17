# NEXT

## 目前位置

**Phase 2 進行中（使用者 2026-10-15 決定，decision 027）**：Research 與 Engineering 兩條線受控並行。
- Phase 1（Core + Advanced，026 的 milestone）結果保留，不重新分析；凍結程式由 `test_core_untouched`、`test_frozen_untouched` 保護。
- **Research Stage C**：
  - C0：8K 延遲雙模式的有限度排查（探索性）。
  - C1：Fixed-SLO 最大並發（正式）。PREREGISTRATION §8 已凍結（2026-10-17）；SLO 由使用者選定（S2）。
- **Engineering**：Capacity-Aware Serving Service MVP（`service/`），建在 vLLM 之上。只有 Stage C 驗證過的容量能成為正式限制。
- 任何在 RTX 上啟動 vLLM 的工作都要取得 GPU lock；產品測試的輸出不進 `results/`；正式研究 session 進行中不部署任何東西。

**Core 全部完成（2026-10-10）**：Mac / RTX Core A、Core B（uniform + mixed）、quality、deployment decision table。
**Gate 通過（使用者 2026-10-10 決定，decision 023）**：進入 Advanced，範圍只限隔離 H-B4 的成因。Optional 未觸碰。
**Advanced 完成（2026-10-11，decision 025）**：正式 session `formal-20261011-adv` 60 / 60 有效；
2K 落在 R6（KV 可能有關，但未確立），8K 落在 R1（主因是 chunked prefill 壟斷 step budget，H-C1 成立）；跨 L「成因依 L 而異」。

- Stage A：harness commit `eb6eef1`，`freeze.mac = true`。
- Stage B：harness commit `19446d7`，凍結 commit `b42614a`，`freeze.nvidia = true`。
- 模型（008）：
  - Core A：Qwen3.5 4B + 9B GGUF，BF16 / Q8_0 / Q6_K / Q4_K_M。
  - Core B：RedHatAI/Qwen3.5-9B-quantized.w4a16。
  - context：2K / 8K / 32K。
- Decision log 001–033（030 於 2026-10-17 凍結；033 是 C1 的結果）。細節以 decision log 為準，這裡只記現況與下一步。
- GitHub：`origin` 為 private repo。每個測試通過的 commit 都會自動 push；不 force push，不改寫 history。
  權重、GGUF、KLD、`results/**/raw/`、venv、credential 不進 git。

## 歷史紀錄：Core 執行期間的現況（2026-10-10 20:30 寫下，之後未更新；目前狀態以上方「目前位置」為準）

### 使用者決定（2026-10-09 上午）

- **019**：
  - Mac 正式 run 用**自動電力模式**加上**原廠 70W adapter**，並記錄 power state。
  - 正式 Mac Core A 的電池下降停止規則：13:xx 依使用者決定刪除。
  - 高效能模式（powermode 2）也可以，但同一 session 內要一致。
  - Core B 的 `--max-num-seqs` 改為 **128**，grid 上限改為 **128**。
  - Stage B 不拆分。
- **020**：正式 quality 改在 **RTX** 上執行，flag set 與 hypothesis 不變。
  - 4B 一起搬：使用者 2026-10-09 確認。
  9B BF16 會少量 spill 到系統記憶體，運算仍全部在 CUDA 上完成。

### RTX（gpu-server，GPU lock 下依序執行）

- **正式 RTX Core A 完成**（`formal-20261009-rtx-a1`，10:42–13:11）：
  - 164 / 164 ok，sentinel 漂移 0.06%，env 有效，session 開頭重量的 B = 462 MiB（沒有 drift）。
  - 判定（`analysis.json`）：H-A1、H-A3、H-A4a、H-A4b 為 not refuted；**H-A2 為 refuted**。
    - 推翻 H-A2 的是 4B-Q4_K_M 對 9B-Q8_0 這一組：觀測比 2.24，預測比 2.99，偏差 −25%，門檻是 ±20%。
    - 15 組配對的偏差全部是負的：bytes 少的配置都比純頻寬比例預測的慢。原因屬於推測，Core 不驗證。
  - 圖：`results/core_a/figures/{decode,prefill,memory}-rtx-formal.svg`。
- **正式 Core B uniform 完成**（`formal-20261009-coreb-uniform`，2026-10-09 13:11 至 2026-10-10 09:11，約 20 小時）：
  - 102 / 102 ok，formal，env 有效，沒有 teardown 問題。
  - uniform-only 分析：H-B1、H-B2、H-B3 not refuted。
    - C_knee：2K 為 25（C_KV 35）、8K 為 8（C_KV 13）、32K 為 2（C_KV 3），都落在 H-B2 的 [C_KV ÷ 2, 2 × C_KV] 內。
    - H-B1 三個 L 都是 supported（首次達到 3 倍 P95 TTFT 的 C：70 / 16 / 8）。
  - H-B4 要等 mixed，目前 0 個可檢定的 unit。
  - mixed 跑完後，`analysis.json` 要以 `--mixed` 重產（021 R3）。
  - 圖：`results/core_b/figures/{p95_ttft_ms,p95_tpot_ms,output_throughput}-uniform-formal.svg`。
- **uniform 之後的自動接續**（背景 `chain_rtx_after_uniform.sh`，每一步都在 GPU flock 下、ssh 前景執行，遇到問題就停）：
  1. 唯讀檢查 uniform session：formal、env 有效、102 列全部 ok。不符合就停止，不部署、不開跑任何東西。
  2. 確認 GPU 上沒有 compute process 後，部署 `2c2e764`（bundle 事先建好），並核對 `DEPLOYED_COMMIT`。
  3. RTX quality pilot（`pilot-<日期>-rtx-q`，005 pilot protocol，N_thr 6 / P 0），接著產生遮蔽版報告。
  4. 正式 Core B mixed（`formal-<日期>-coreb-mixed`），C = 各 L 的 uniform C_knee，使用 pin 過 sha256 的 mixed dataset。
  - `core_b_plan.py mixed` 現在會拒絕非 formal 或 env 無效的 uniform session（與 `core_b_report` 同一規則，018 補充）。

- （歷史）正式 RTX Core A `formal-20261009-rtx-a1`。
  - 10:36 開始，session 開頭會重量 B；pilot 估計上限約 3.6 小時。
  - 跑完自動接正式 Core B uniform `formal-20261009-coreb-uniform`：3 個 L × 3 次 restart，粗估約 10 小時。
  - 背景串接腳本：`chain_rtx_formal.sh`。
- **Stage B 值**（§6）：

  | 項目 | 值 |
  |---|---|
  | B | 811 MiB |
  | V_total | 16,303 MiB |
  | C_ctx | −89 MiB |
  | U | 0.91 |
  | K | 4.0 GiB |
  | num_gpu_blocks / block_size | 248 / 528 |
  | pin memory | 0 |
  | N_thr / P | 6 / 0 |
  | HAGS | 開 |
  | 超量測試 | spill |
  | C_KV（2K / 8K / 32K） | 35 / 13 / 3 |
  | grid 2K | 1, 2, 4, 8, 16, 18, 25, 32, 35, 49, 64, 70, 128 |
  | grid 8K | 1, 2, 4, 7, 8, 9, 13, 16, 18, 26, 32, 64, 128 |
  | grid 32K | 1, 2, 3, 4, 6, 8, 16, 32 |

- **BF16 參考檔已傳到 WSL**（020）：9B、4B 都在 WSL 上以 sha256 驗證完成（20:09）。quality 資料檔 14 個也已核對。
- **注意**：正式 RTX session 進行中**不得重新部署**到 WSL。
  - 中途部署會悄悄換掉之後 process 使用的程式，而且不會留下標記（020 第 7 點）。
  - 審查後三個 runner 都加了 provenance 檢查，但進行中的 session 跑的是 `b42614a`，沒有這個檢查。
  - 等 GPU 上的正式 session 結束後，才部署 `run_quality` 的 RTX 支援。

### Mac

- 使用者 2026-10-10 中午再次暫停 Mac 的高負載，**當天傍晚解除**（使用者親自通知）。
  - quality 跑完後，直接在 Mac 上跑 `quality_analysis.py`。downstream bootstrap 約 10 億次抽樣，會占一個 CPU 核心數分鐘。
    放在 Mac 上跑，是為了避免 021 R3 的跨平台浮點不一致。

- 使用者 2026-10-10 01:45 暫停 Mac 的新重負載；**03:55 解除**（由使用者的另一個 Claude session 轉達「mac 已經用好了」；使用者 2026-10-10 上午親自確認可以用）。
  解除後 Mac 也沒有待跑的重負載：Core A 已完成，quality 在 RTX，Core B 只在 RTX。
  - a2 已跑完且有效（見下）。RTX 照常進行（使用者只用 Mac）。
  - Mac 目前沒有排任何重負載；只做輕量的分析、圖與 commit。

- **2026-10-09 13:xx：使用者刪除 019 的電池下降停止規則，並要求重跑。**
  - `formal-20261009-a1` 依舊規則作廢，照實保留。
  - 重跑 `formal-20261009-a2` 在 Mac lock 下、固定 commit 的 worktree 中執行。
  - 適用的規則：012、016 充電閘門、70W adapter、同一 session 內 powermode 一致。

- **正式 Mac Core A 完成**（`formal-20261009-a2`，2026-10-09 20:15 至 2026-10-10 02:55，commit `8f8dc9a` 的 worktree，powermode 2，70W adapter）：
  - 164 / 164 ok，sentinel 漂移 1.78%（門檻 3%），env 有效，沒有充電暫停；每筆 env 的 commit 都相同且 dirty = False。
  - 判定（`analysis.json`）：H-A1、H-A3、H-A4a、H-A4b not refuted；**H-A2 refuted**。
    - 推翻 H-A2 的是與 RTX 相同的配對：4B-Q4_K_M 對 9B-Q8_0，觀測比 2.22，預測比 2.99，偏差 −25.8%。
    - 15 組配對中 13 組偏差為負。
  - 圖：`results/core_a/figures/{decode,prefill,memory}-mac-formal.svg`。worktree 已移除（內容與主 repo 的副本逐檔相同）。
- 019 的 harness 已上線，並經過 review workflow：確認 10 項，均已修正。

## 下一步（Phase 2）

### Research
1. **C0**（decision 028）：
   - harness 已 commit（`e4c23c9`），審查 6 項已處理。
   - **步驟 0 完成**（`ac29c54`）。事先寫定的重現檢查沒有重現；每次滑移都是錯過一個 2048-token prefill step（約 349 ms）；Core r2 的滑移較多，可能有 boot 層級的成分。
   - **閒置判定改為 decision 031 的綜合判定**（使用者 2026-10-16 確認），並依實測校準 spill 偵測（031 修正案，使用者確認）：
     就緒時比 boot 前多出 ≤ 160 MiB（正常 boot 固定 +80 MiB），之後每個 bench 相對就緒時 < 64 MiB。
   - 使用者結束了 ChatGPT、BingWallpaper；依使用者指示我結束了 Radeon Software。閒置 VRAM 降到 616–960 MiB。
   - WSL 的 interop binfmt 會在開機後消失，`read_shared_usage` 改為遇到 ENOEXEC 時經由 `/init` 呼叫（不改系統設定）。
   - **C0 完成**（部署 `f0eee6a`，2026-10-16 23:31 – 10-17 01:59，GPU lock 下）：`explore-c0-20261016` 24 / 24 個 bench 有效。
     滑移段 5 段，D = 3.40 < 11.07 → **Rrun → 分支 A**（decision 032）。起始組集中在 idx 89 等現象只報告，列入 Future Work。
   - 同一次 lock 的 Stage C smoke 都通過：mini `smoke-mini-202610161531`、FS-order `smoke-fs-order-202610161531`。
2. **§8**（decision 030，**FROZEN 2026-10-17**）：
   - 使用者已決定（2026-10-15）：SLO S2（TTFT 2 / 6 / 20 s，TPOT 100 ms）、mixed 長 request 為 budget、穩態時間窗、GPU 空著就跑。O-32K、O-T 不做。
   - 設定檔放在 `configs/capacity.toml`；`pins.toml` 維持 Advanced 凍結時的位元組（030 實作補記）。
3. **Stage C harness 完成**（`eb78a35`，仍為 DRAFT、未凍結）：審查 21 項已處理；030 實作補記二記錄了所有實作時的決定，
   包括 C_PROD 一律把 MARGINAL 當 FAIL，以及凍結必須附上的證據（兩次通過的 smoke 與 C0 的 decision log）。
   smoke 通過、C0 分支 A、凍結證據已填入 `configs/capacity.toml`（decision 032，`f35fe85`）。
   **已凍結**：`freeze.capacity = true` 單獨一個 commit（profile 產生器以它為凍結 commit）；`tests/test_stage_c_frozen.py` 以 sha256 固定五個 Stage C 模組與 capacity.toml。
   凍結 commit `b3c9109`（profile 產生器的 `freeze_commit_problems` 已確認它是合格的凍結 commit）。
   - **正式 session 完成**：`formal-20261017-c1`（部署 `b3c9109`，2026-10-17 02:51–10:18，7.45 h），69 / 74 個 run 有效，replay、sentinel、跨 session 偏移都通過。
     結果（decision 033）：U2K C_PROD 32 LOWER、M2K / M8K 2 EXACT、**U8K NT**。
     5 個無效的 run 都是 spill：BingWallpaper 在 08:29 自行重新啟動，在 NVIDIA 卡上用了約 190 MiB。spill 偵測擋下了它們，依規則不補跑。
   - **待使用者決定**：U8K 是否以相同的凍結協定重跑一個完整 session（033 的選項 1–3）。
     若要重跑，使用者要先讓 BingWallpaper 不會自動啟動（屬於系統設定，我不修改），我再事先寫好兩個 session 的取捨規則。

### Engineering
1. 架構：decision 029。
2. **M0 完成**（`3147165`）：
   - 審查 12 項，11 項修正，1 項延到 M2（見 029 實作補記）；service 測試 274 passed。
   - 刪除 uvicorn 的 `limit_concurrency`，因為它數的是連線，到上限時會讓所有 request 都拿到 503。
3. **M1 完成（Mac 端）**（`343155f`）：deploy、procs、runs、loadtest、itest（IT-0 至 IT-6）、`tools/rtx/deploy_service.sh`。
   審查 11 項（2 項重複）全部修正，見 029 的 M1 實作補記；完整測試 2006 passed。
   - **RTX 整合測試 IT-0 至 IT-6 全部通過**（2026-10-16，兩次：`pt-20261016T142720Z-itest`、`pt-20261016T143958Z`，release `635b954` / `f30d17c`），
     依 031 的產品規格（基線判定；> 2,048 MiB 拒絕）。只是產品測試，不是容量。
4. **M2 已實作**（`b91a9bf`）：產生器、載入時的 provenance 驗證、verified 0 與 unavailable class。
   **第一份 verified profile**：`service/profiles/verified-formal-20261017-c1.json`（uniform-2k 32、mixed-2k 2、mixed-8k 2、uniform-8k unavailable）。
   下一步：以 verified profile（`service/deploy/rtx-verified.toml`，active class uniform-2k）在 RTX 上跑 IT-0 至 IT-6，不能和研究 session 重疊。
   - 10:26 起 GPU lock 由另一個專案（agent-gym 的 ollama）持有，不搶占。
   - 背景每 5 分鐘唯讀檢查一次，lock 空著時才部署 release 並執行 IT，最多等 12 h。遠端 log：`/tmp/llmsl-itest-verified.log`。

### Phase 1 收尾（2026-10-11，已完成）

- **收尾工作（2026-10-11，已完成）**：
  - `test_disk_guard` 的環境相依已修正（026）：`disk_guard.check()` 改為呼叫時才取 `shutil.disk_usage`；先有紅燈測試，修正後完整測試通過。
  - README 與 NEXT 標記目前的 milestone；8K uniform 的雙模式記入 Future Work / limitations，不追查。

（以下為歷史紀錄）

0. **023 的 Advanced 範圍已完成**（decision 025，README「Advanced」一節）。
   - 正式 session `formal-20261011-adv`（部署 `97a948c`，GPU lock 下，2026-10-11 00:27–06:33）：60 / 60 次 boot 有效，0 逾時，env 有效。
   - `adv_report`（Mac，project venv）：2K → R6，8K → R1，跨 L 不合併。事先預期 8K 符合、2K 不符合，照實報告。
   - 以 `rows.jsonl` 獨立重算 U、M、ρ、f、a，與 `analysis.json` 一致；獨立核對者重跑 `adv_report` 逐位元相同，並核對 025 / README 的文字（14 項修正已納入）。
   - Core 判定與 decision table 不變（重跑只差 pins sha256，commit 的仍是第一次執行的輸出）。
   - 之後的任何 Advanced 項目或 Optional 都需要使用者決定；使用者 2026-10-11 決定收尾（026）。

1. （歷史）**Advanced（H-B4 成因隔離，decision 023、024）**：
   1. 事先登記草稿已 commit（`ae368e6`）：decision 024、PREREGISTRATION §7（DRAFT）。
      - 5 個 condition：B0、T（threshold 512）、G（關閉 full-ISL 准入閘門）、TG、K3（KV 3 GiB）。
      - 每個 condition × 2 個 L × 2 種 mix × 3 次 restart = 60 次 boot，預期約 7.3 h GPU。
   2. **harness 完成**：`adv_plan`、`adv_decomp`、`adv_analysis`、`run_adv`、`adv_report`、`adv_smoke`（TDD；Core 模組不改，有 sha256 回歸測試）。
      審查確認 13 項，全部修正並再驗證；完整測試 1006 passed。實作時的細節記在 024「實作補記」。
   3. **smoke 通過**（`results/advanced/smoke/20261011-fs`，FS0–FS5）：K3 實測 186 個 block，符合規則。
   4. **凍結**（2026-10-11）：`freeze.advanced = true`，PREREGISTRATION §7 為 FROZEN（024「凍結」一節）。
   5. 部署凍結 commit（`deploy_bundle.py` 需要 PATH 先找到 CommandLineTools 的 git）→ 正式 run（約 7–8 h，GPU lock 下）→ `adv_report` → 報告。
2. Core 的結果與 decision table 不因 Advanced 而改變。

## Blocked / 需要使用者

目前沒有。研究的閒置判定已滿足（使用者結束了背景程式）。若之後閒置 VRAM 又超過 1,018 MiB，研究會依 031 自動不開始。

## 建議（非 blocker）

- 2026-10-10 20:14 起，`/usr/bin/git`（Xcode shim）因 `libxcrun` 架構不相容而失敗，可能是 Xcode 更新所致。
  目前改用 `/Library/Developer/CommandLineTools/usr/bin/git`，不修改 xcode-select 等系統設定。
  `scripts/deploy_bundle.py` 內部呼叫 `git`；下次部署前要先確認 PATH，或由使用者修復 Xcode。

- 正式 Mac run 期間盡量少開前景程式。
  - 前景負載不會讓 session 作廢，但會增加 noise。
  - 4B pilot 時前景 loadavg 為 6–9，CV 5.9%；9B pilot 在深夜，CV 1.0%。

## 已解決

- 2026-10-11：
  - 下一步方向：使用者決定收尾，不新增 Advanced 項目；8K uniform 的雙模式記入 Future Work（026）。
  - `test_disk_guard` 的結果取決於機器的剩餘磁碟空間（只有 60–250 GiB 時通過）：已修正（026）。

- 遠端 SSH（2026-10-08）。
- WSL compiler（使用者已安裝）。
- llama.cpp 的 CUDA 12.8 系統套件：改走 user-space 路線，不再需要（015）。
- 2026-10-09：
  - Mac 正式 run 解除暫停（019）。
  - Core B 選 A（019）。
  - Stage B 不拆分（019）。
  - quality 的 Mac 磁碟不足：quality 改在 RTX 執行，問題消失（020）。清理 scratchpad 釋出約 2.2 GB。
  - 019 中「quality 是否套用電池下降停止規則」：quality 改在 RTX，不再適用（020）。
  - 4B 的 quality 也在 RTX 上量：使用者確認（020）。
