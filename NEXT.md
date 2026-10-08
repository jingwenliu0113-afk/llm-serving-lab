# NEXT

## 目前位置

**Core — Stage A（第 1–5 節）已於 2026-10-09 凍結**（harness commit `eb6eef1`，`freeze.mac = true`）。
NVIDIA 端（Stage B）平行推進中。Advanced / Optional 皆未觸碰。

- 模型（008，使用者已確認）：Core A 為 Qwen3.5 4B + 9B；Core B 為 RedHatAI/Qwen3.5-9B-quantized.w4a16；context 2K / 8K / 32K。
- Decision log 001–017。詳細內容以 decision log 為準，這裡只記現況與下一步。
- GitHub：`origin` 為 private repo。每個測試通過的 commit 都會自動 push；不 force push，不改寫 history。
  權重、GGUF、KLD、`results/**/raw/`、venv、credential 不進 git。

## 現況（2026-10-09 01:15）

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
  02:37 起在 Mac lock 下傳輸（可續傳）。**4B-Q8_0 已於 03:25 傳完並在遠端驗證**（約 1.6 MB/s）。
  之後先傳 Stage B 需要的 4B-Q4_K_M 與 9B-Q8_0，再傳其餘 3 個。
  傳完後依序：`stage_b.py cctx` → `overflow` → RTX Core A pilot（`--n-thr 6`）。

## 下一步（依序；Mac 重負載經 `tools/with-mac-lock`，GPU 重負載經 `flock -n`，不搶 lock）

1. NVIDIA：
   1. ~~S1 → S2 → S3~~ ✅；~~llama.cpp CUDA build~~ ✅。
   3. Stage B 量測（進行中），全部在 GPU lock 下：
      - `stage_b.py baseline` 量 B，依 017 算出 U。
      - `coreb_stage_b.py boots` 取得 JIT 冷 / 熱時間與 K。
      - `stage_b.py cctx` 與 `overflow`。
      - `coreb_stage_b.py pin-ab`，同時取得 num_gpu_blocks / block_size。
   4. RTX Core A pilot：`core_a_pilot.py plan --platform wsl2-cuda --sizes 4B 9B --n-thr 6`，在 lock 下跑，決定 P。
   5. 填 §6 → `freeze.nvidia = true` → 正式 Core A（RTX）與 Core B（uniform → 由 `core_b_plan.py mixed` 產生 mixed plan → report / 圖）。
2. 正式 Core A（Mac）：等使用者解除暫停（見 Blocked 0）。
   指令在 scratchpad 的 `batch_formal_a.sh`（seed 20261008、`-t 8`、`--poll 0`、含 016）。
   跑完後用 `core_a_report.py` 出 analysis，用 `core_a_figures.py` 出 decode / prefill / frontier 圖。
3. 正式 quality（Mac）：等磁碟（見 Blocked 1）。之後用 `quality_analysis.py` 出 analysis，並畫 frontier 圖。

## Blocked / 需要使用者

0. **解除 Mac 正式 benchmark 的暫停**（使用者 2026-10-09 指示）：正式 Core A、正式 quality、任何 Mac heavy run 都等使用者通知。
   - §4 規定正式 run 的 **powermode 必須為 2（高效能）**、`iogpu.wired_limit_mb` 必須為 0。
     018 起 harness 會強制檢查；目前是 powermode 0（「自動」），正式 run 會在開跑前被拒絕。
     請使用者在正式 run 前切到「高效能」；這是系統設定，我不會修改。
   - 建議接上 ≥ 96 W adapter（見 1 與 3）。

1. **正式 quality：磁碟空間，以及 016 的充電門檻**（Mac；不影響 Core A）
   - 充電門檻（018 第 6 點）：正式 quality 中 9B 的 MC job 長 38–59 分鐘，從 40% 開始跑可能跌破 20%，使整個 session 作廢。
     改門檻屬於修改凍結的 §4，不自行處理。選項：
     - 換 ≥ 96 W adapter：不必改任何規則。
     - 或同意為 quality 補一則 016 addendum（例如暫停門檻 75%）；程式已參數化。
   - 目前可用 94 GiB。每個 size 的 KLD logits 約需 70 GiB，加上 `disk_guard` 的 50 GiB 保留量，共 120 GiB。
   - 即使依 STORAGE_PLAN 暫時刪掉另一個 size 的 quant（4B 約 11 GB，9B 約 23 GB；之後可從 BF16 重建），仍不夠。
     BF16 不能刪：safetensors 已刪，BF16 無法重建。
   - 需要使用者擇一：
     - 釋出約 30 GiB 以上的空間；
     - 或同意把 KLD 檔放到外接 / 其他磁碟。
   - 降低 50 GiB 保留量屬於規則變更，我不會自行修改。
2. **Core B：`--max-num-seqs 264` 在這張卡上無法啟動（凍結的 §4 / 007，要由使用者決定）**
   - 原因：Qwen3.5 是 hybrid 模型，每條 decode 序列需要一個 GDN（Mamba）狀態 block，約 50 MiB。
     vLLM 0.30.0 在 FULL CUDA graph 下要求 max_num_seqs ≤ num_gpu_blocks。
     U = 0.91 時 num_gpu_blocks = 134；264 條序列光是 GDN 狀態就要 13 GiB 以上，16 GB 卡上不論 U 為何都不可能達成。
   - 007 選 264 的目的，是讓 knee 不會被 scheduler 上限截斷（grid 上限 256）。
     實測 num_gpu_blocks = 134 時，依 §2 公式 C_KV(2K) = ⌊133 ÷ 7⌋ = 19、C_KV(8K) = 7、C_KV(32K) = 1，knee 遠低於 128。
   - 選項（都要寫 decision log，並標明是凍結後、但在任何 Core B 數據之前的變更）：
     - **(建議) A**：`--max-num-seqs 128`（8 的倍數，且 ≤ num_gpu_blocks），grid 上限從 256 改為 128，其餘規則不變。
       H-B1 需要 grid 涵蓋 4 × C_knee：以 C_KV 估計約 76 ≤ 128，仍可檢驗。
     - B：保留 264，改用 PIECEWISE-only 的 CUDA graph（`--compilation-config '{"cudagraph_mode": "PIECEWISE"}'`）。
       decode 沒有 FULL graph，偏離 vLLM 預設，throughput 會偏低。
     - C：保留 264，加 `--enforce-eager`。完全沒有 CUDA graph，偏離最大，不建議。
   - 在決定之前不會跑任何 Core B 的 Stage B 或正式量測；baseline B 已量好，可以沿用。
3. **建議（非 blocker）**：Mac 換 ≥ 96 W 的 adapter。
   - 016 已讓長 session 在 70 W 下仍有效，但充電暫停會讓正式 Core A 從約 17 h 拉長到約 1.5–2 倍。
4. **建議（非 blocker）**：正式 Mac run 期間盡量少開前景程式。
   - 4B pilot 時前景負載的 loadavg 為 6–9，CV 5.9%；9B pilot 在深夜，CV 1.0%。
   - 前景負載不會讓 session 作廢，但會增加 noise。

已解決：遠端 SSH（2026-10-08）、WSL compiler（使用者已安裝）。llama.cpp 的 CUDA 12.8 系統套件已改用 user-space 路線，不再需要（015）。
