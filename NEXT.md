# NEXT

## 目前位置

**Core — Stage A（第 1–5 節）已於 2026-10-09 凍結**（harness commit `eb6eef1`，`freeze.mac = true`）。
NVIDIA 端（Stage B）平行推進中。Advanced / Optional 皆未觸碰。

- 模型（008，使用者已確認）：Core A 為 Qwen3.5 4B + 9B；Core B 為 RedHatAI/Qwen3.5-9B-quantized.w4a16；context 2K / 8K / 32K。
- Decision log 001–018。詳細內容以 decision log 為準，這裡只記現況與下一步。
- GitHub：`origin` 為 private repo。每個測試通過的 commit 都會自動 push；不 force push，不改寫 history。
  權重、GGUF、KLD、`results/**/raw/`、venv、credential 不進 git。

## 現況（2026-10-09 07:40）

### Mac（Core A / quality）

- GGUF ×8 全部 pin 並附 recipe。兩個 size 的 safetensors 都已依規則刪除（`results/artifacts/deletions.jsonl`）。
- Pilot 全部完成（遮蔽版報告，012 及其補充）：
  - 4B Core A 22 / 22、9B Core A 22 / 22，`-fa` probe 正常。
  - quality 4B 26 / 26、4B + 9B 52 / 52。
  - 估計值：
    - 正式 Core A ≤ 17.39 h，正式 quality 約 14.2 h，兩者都不含充電暫停。
    - H-A5 ΔNLL 的 MDE 約 0.009 nats/token。
    - Core A 兩格 MDE 為 1.7–9.6%，依前景負載而定。
- 016 充電閘門已上線：電量 < 40% 時閒置充電到 90%。
- **正式 Core A：DEFERRED**，依使用者指示，見 Blocked 0。
- **正式 quality：Blocked**，磁碟不足，見 Blocked 1。
- 6 個 quant 正在背景傳到 WSL（`scripts/sync_gguf_to_wsl.py`，在 Mac lock 下，傳完在遠端以 sha256 驗證）。
- **注意**：2026-10-09 01:05 起系統正在更新 Xcode，`/usr/bin/git`（xcrun shim）暫時無法使用（libxcrun 架構不符）。
  - 目前改直接用 `/Library/Developer/CommandLineTools/usr/bin/git`，跑測試時把它放在 PATH 最前面。
  - 不修改 xcode-select 等系統設定。
  - Metal build 的 llama.cpp 執行檔不受影響。

### NVIDIA（Core B / Core A RTX）

- 使用者已裝 `gcc g++ libc6-dev`。`wsl_toolchain_check.sh`（`JIT=0`）全部通過。
- Core B：
  - JIT 經由 user-space `CUDA_HOME` shim（014）。
  - server 已補齊 003 的固定 flags，U / K / JIT / A/B 的規則已寫定（017）。
  - `mixed` dataset 已 pin。
- llama.cpp CUDA：user-space CUDA 13.4.1 + Ninja（015）。configure 已通過。RTX 的 N_thr = 6（Ryzen 5 7600，6 核）。
- Stage B 工具已備妥並有測試：`benchmarks/stage_b.py`（B、C_ctx、超量）、`benchmarks/coreb_stage_b.py`（boots、pin-ab）。
- **Core B smoke S1 → S2 → S3 全部通過**（2026-10-09 01:54–02:01，`results/core_b/smoke/20261009-s1s3-014b`；008 補充）。
  先前兩次失敗照實保留：沒有 C compiler；FlashInfer JIT 的 CCCL 版本檢查。後者由 014 補充解決：shim 改用版本一致的 CUDA 13.0（`.venv-fi-cuda130`）。
- 003 gate 第 1–2 項通過：專案 venv 的 capability 為 (12, 0)，arch list 含 sm_120。
- **llama.cpp b11146 在 wsl2-cuda 上 verified**（02:07；CUDA 13.4.1 user space，GNU 11.4.0，`CUDA0 RTX 5070 Ti 16275 MiB`）。
- **Stage B（2026-10-09 02:09–02:17，GPU lock 下，`results/stage_b/20261009/`）**：
  - **B = 811 MiB**（60 個 1 Hz 樣本介於 799–811），**V_total = 16,303 MiB**，display_active = Disabled（螢幕不在這張卡上）。
    HAGS 讀不到（`reg.exe` 沒有回傳值）。依 017，U = **0.91**。
  - boots 的冷啟動在 vLLM 初始化時失敗：
    `max_num_seqs (264) exceeds available Mamba cache blocks (134)`（`vllm/config/compilation.py`：FULL CUDA graph 加上 Mamba 層時，
    要求 max_num_seqs ≤ num_gpu_blocks）。可用 KV 記憶體 2.17 GiB，66,528 tokens，num_gpu_blocks = 134。
    **見 Blocked 2，需要使用者決定**；Core B 的 Stage B（K、JIT 計時、pin-memory A/B）與正式 Core B 都暫停。
- **第三輪審查（018）**：凍結後新寫程式的 18 項確認問題全部修正，都是讓程式落實已凍結的規則。
  主要內容：Stage B 值集中到 pins `[stage_b]`、Core B 正式 run 的防護、log 驗證、powermode / seed / power-limit 檢查，
  以及 boot 期間被中斷時會關掉 server。
- **003 gate 第 5 項通過**（02:34，`results/stage_b/20261009/bench-check`）：在 S1 smoke server 上 `vllm bench serve` 完成並通過完整驗證，
  `/metrics` 的三個 metric 名稱都存在。
- WSL 會在 wsl.exe 結束時終止背景 process，所以遠端長時間工作都在 ssh 前景執行（`ServerAliveInterval`）。
- Mac ↔ 桌機約 0.7–0.9 MB/s，而且會斷線。GGUF 傳輸已改成可續傳（`sync_gguf_to_wsl.py`）。
  02:37 起在 Mac lock 下傳輸（可續傳）。已在遠端驗證的有：**4B-Q8_0、4B-Q4_K_M、9B-Q8_0（04:5x）**，約 1.6–2.4 MB/s。
  其餘依序為 4B-Q6_K → 9B-Q6_K → 9B-Q4_K_M。
- **C_ctx = −89 MiB**（03:54，依 009 規則；照實記錄，說明見 §6）。spill 規則用到的 B + C_ctx = 722 MiB。
- **sysmem 超量測試：spill，不是 hard OOM**（04:56）。9B-Q8_0 搭配 `-c 262144` 時，self 16,788 MiB，超過 16,275 MiB 的卡，
  process 正常結束。RTX 的 fit 判定照 004 的 spill 規則；規則不變。
- **HAGS 開啟**；`.wslconfig` 為 memory=31GB、swap=8GB。
  - 從 Windows 端唯讀查詢。WSL interop 執行 `.exe` 會出現 `Exec format error`，屬於系統設定，不修改。
- 6 個 quant 已全部在 WSL 驗證完畢（07:02）。
- **RTX Core A pilot 完成**（07:03–07:27，`results/core_a/wsl2-cuda/pilot-20261009-rtx`，34 / 34 ok，遮蔽版報告；012 補充）：
  - **P = 0**（poll 0 為 184.37 tok/s，poll 50 為 184.06 tok/s），已寫入 pins `[stage_b] rtx_poll = 0`。
  - `-fa auto` 在 6 個 quant 上都是 enabled，graph splits 都是 2。
  - sentinel CV 0.80%，記憶體增量與 Mac 一致；正式 RTX Core A 上限約 3.62 小時。
  - 報告程式的 bug（018 之後用 seed 0 建正式 plan）已修正；凍結測試原本把 C_ctx > 0 當成「已量測」，也已修正。兩者都記在 018 補充。
- **Core A 的 RTX Stage B 項目已全部量完**：B、V_total、C_ctx、HAGS / `.wslconfig`、超量測試、N_thr、P。
  剩下的 §6 項目全是 Core B 的，都卡在 Blocked 2。
- GGUF 傳輸會占滿上行頻寬，GitHub 的 TLS 連線因此逾時。現在 push 時會先短暫暫停 sender（SIGSTOP / SIGCONT，傳輸不中斷）。

## 下一步（依序；Mac 重負載經 `tools/with-mac-lock`，GPU 重負載經 `flock -n`，不搶 lock）

1. NVIDIA：
   1. ~~S1 → S2 → S3~~ ✅；~~llama.cpp CUDA build~~ ✅。
   3. Stage B 量測（進行中），全部在 GPU lock 下：
      - `stage_b.py baseline` 量 B，依 017 算出 U。
      - `coreb_stage_b.py boots` 取得 JIT 冷 / 熱時間與 K。
      - ~~`stage_b.py cctx` 與 `overflow`~~ ✅（C_ctx −89 MiB；spill）。
      - `coreb_stage_b.py pin-ab`，同時取得 num_gpu_blocks / block_size。
   4. ~~RTX Core A pilot~~ ✅（P = 0）。
   5. 填 §6 → `freeze.nvidia = true` → 正式 Core A（RTX）與 Core B（uniform → 由 `core_b_plan.py mixed` 產生 mixed plan → report / 圖）。
2. 正式 Core A（Mac）：等使用者解除暫停（見 Blocked 0）。
   指令在 scratchpad 的 `batch_formal_a.sh`（seed 20261008、`-t 8`、`--poll 0`、含 016）。
   跑完後用 `core_a_report.py` 出 analysis，用 `core_a_figures.py` 出 decode / prefill / frontier 圖。
3. 正式 quality（Mac）：等磁碟（見 Blocked 1）。之後用 `quality_analysis.py` 出 analysis，並畫 frontier 圖。

## Blocked / 需要使用者

0. **AC 的能源模式目前是「高效能」（powermode 2），正式 Mac Core A 無法開始**
   - 019 依使用者決定，要求正式 Mac run 用自動模式（powermode 0）。
   - 09:30 時讀到 0，09:56 讀到 AC 設定為 2（`pmset -g custom`：AC Power powermode 2、Battery Power powermode 0）。
   - 請到「系統設定 > 電池 > 能源模式 > 接上電源轉接器時」選「自動」。這是系統設定，我不會修改。
   - 另外，Mac heavy lock 從 09:29 起由另一個專案持有（incident-investigator 的 pilot：`collect run --set pilot`）。
     依 MACHINE_POLICY 不搶占，等它釋放。
     AC 被切到 powermode 2 的時間點和它重疊，可能是那個專案的設定。兩個專案若對電源模式要求不同，要由使用者協調。
   - 背景已有唯讀 watcher（`wait_power_then_formal_a.sh`）每 2 分鐘檢查一次。
     條件是 AC、powermode 0、Apple 70W adapter，而且 Mac lock 空著；一符合就在 Mac lock 下自動開始正式 Mac Core A，不必再通知我。

1. **正式 quality：磁碟空間**（Mac；不影響 Core A）
   - 使用者決定（2026-10-09，019 §4）：
     - 016 的門檻不改，也不等 ≥ 96 W adapter。
     - 先清理可重建的 scratch；manifest、結果、hash、recipe 不刪。
     - 仍不足時，quality 維持 blocked。
   - 已清理：本 session scratchpad 中研究用的原始碼與 wheel 解壓檔、舊 tar，約 2.2 GB。其餘可重建的 scratch 很少：
     - `third_party/convert-venv`（0.5 GB）：重建 BF16 時要用，保留。
     - `~/.cache/huggingface` 的 1.4 GB 不屬於本專案，不動。
   - 現在可用 **94 GiB**。quality 需要 KLD 約 70 GiB 加上 50 GiB 保留量，共 120 GiB，**差約 26 GiB，所以 quality 維持 blocked**。
   - Mac Core A 正式 run 完成後，依 STORAGE_PLAN 既定做法再實測一次（以 `disk_guard` 為準）。
     既定做法：跑某個 size 的 quality 時，暫時移除另一個 size 的 3 個 quant，之後以 `produce_gguf.py produce` 從 BF16 依 sha256 重建。
     - 4B quality 時移除 9B 的 quant（約 21 GiB），可用約 115 GiB，仍差約 5 GiB。
     - 9B quality 時移除 4B 的 quant 與已用完的 4B BF16（約 18 GiB），可用約 112 GiB，仍差約 8 GiB。
   - 需要使用者擇一：
     - 釋出約 10 GiB 以上的空間；
     - 同意 KLD 放到其他磁碟；
     - 同意在 4B quality 期間暫時刪除 9B BF16（S 級）。
       之後要重新下載 pinned safetensors（19.3 GB），轉回 BF16 後 sha256 必須與 pins 相同。這需要下載，所以要使用者同意。
   - 我不會降低 50 GiB 的保留量。
   - **另一個待確認問題：quality 是否套用 019 的電池下降停止規則**（019 第 1 節）。
     你的原話沒有區分 Core A 與 quality。我的提案是 quality 不套用，理由：
     - quality 的結果是確定性的；
     - 單一 job 長 38–59 分鐘，30 分鐘視窗只看前後讀數會涵蓋不到。

     確認之前不跑正式 quality。若要套用，我會加入 job 進行中的電量取樣。

## 建議（非 blocker）

- 正式 Mac run 期間盡量少開前景程式。
  - 前景負載不會讓 session 作廢，但會增加 noise。
  - 4B pilot 時前景 loadavg 為 6–9，CV 5.9%；9B pilot 在深夜，CV 1.0%。

已解決：
- 遠端 SSH（2026-10-08）。
- WSL compiler（使用者已安裝）。
- llama.cpp 的 CUDA 12.8 系統套件：改走 user-space 路線，不再需要（015）。
- 2026-10-09 上午，使用者決定（decision 019）：
  - Mac 正式 run 解除暫停，改用自動電力模式加上原廠 70 W adapter，並加上電池下降停止規則；
  - Core B 選 A（`--max-num-seqs 128`，grid 上限 128）；
  - Stage B 不拆分。
