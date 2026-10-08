# NEXT

## 目前位置

**Core — Week 1 收尾，Stage A（Mac）凍結前最後一步；NVIDIA 端平行推進。** Advanced / Optional 皆未觸碰。

- 模型（008，使用者已確認）：Core A 為 Qwen3.5 4B + 9B；Core B 為 RedHatAI/Qwen3.5-9B-quantized.w4a16；context 2K / 8K / 32K。
- Decision log 001–016。詳細內容以 decision log 為準，這裡只記現況與下一步。
- GitHub：`origin` 為 private repo。每個測試通過的 commit 都會自動 push；不 force push，不改寫 history。
  權重、GGUF、KLD、`results/**/raw/`、venv、credential 不進 git。

## 現況（2026-10-09 00:45）

### Mac（Core A / quality）

- GGUF ×8 全部 pin 並附 recipe（4B、9B 各為 BF16 加 Q8_0 / Q6_K / Q4_K_M）。
  兩個 size 的 safetensors 都已依規則刪除，紀錄在 `results/artifacts/deletions.jsonl`。
- Pilot（遮蔽版報告，012 及其補充）：
  - 4B Core A：22 / 22 ok。
  - 4B `-fa` probe：ok。
  - 4B quality：26 / 26 ok。
  - **9B Core A：22 / 22 ok**。FA 已啟用，graph splits = 2，CV 1.03%，KV 增量 +958 MiB，正式 Core A 上限 **17.39 h**。
  - 9B smoke：正常。
- **quality pilot（4B + 9B）**：00:06 起在 Mac lock 下執行（`results/quality/pilot-20261008-both`），完成後算 H-A5 ΔNLL 的 SD / MDE 與總時數。
- **電源**：70 W adapter 滿載時電量仍會下降，9B pilot 38 分鐘內從 100% 掉到 73%。
  因此加上 **016 充電閘門**：電量 < 40% 時閒置充電到 90% 再繼續，讓長 session 不會因為電量 < 20% 作廢。

### NVIDIA（Core B / Core A RTX）

- 使用者已裝 `gcc g++ libc6-dev`（2026-10-08 23:38）。`wsl_toolchain_check.sh`（`JIT=0`）全部通過。
- Core B：
  - FlashInfer 的 JIT 透過 user-space `CUDA_HOME` shim，使用 vLLM lock 中的 pip CUDA 13.4.92（014）。
  - 啟動前會先檢查工具鏈。
  - `mixed` dataset ×2 已建好並 pin（013 補充）。
- llama.cpp CUDA：改用 user-space CUDA 13.4.1（pip wheels，`configs/llama-cuda.lock.txt`）加 Ninja（015）。configure 已通過。
- **GPU lock 從 23:38 起由 Text-to-SQL 持有**（`inference.generate`）。
  背景輪詢只用唯讀的 `lslocks`，lock 釋放且 GPU 閒置後，自動依序跑 `JIT=1` 檢查與 S1。

## 下一步（依序；Mac 重負載都經 `tools/with-mac-lock`，不搶 lock）

1. quality pilot 完成後：
   - 用遮蔽版報告算 H-A5 MDE 與 quality 總時數，寫進 012 補充。
   - 補完 PREREGISTRATION §5 checklist：GGUF ×8、9B probe / 校準 / 記憶體、MDE。
   - 記錄 harness commit，設 `freeze.mac = true`，第 1–5 節標為 **FROZEN (Stage A)**，然後 commit。
2. **正式 Core A（Mac）：DEFERRED**。使用者 2026-10-09 指示：Mac 今晚是「自動」電力模式，供電只有約 68 W，
   暫停所有新的正式 Mac performance benchmark 與 Mac heavy run，等使用者通知解除再跑。
   - 已備妥：凍結後的指令在 scratchpad 的 `batch_formal_a.sh`：plan seed 20261008、`-t 8`、`--poll 0`，經 `tools/with-mac-lock`，含 016 充電閘門。
   - 跑完後用 `core_a_report.py` 出 analysis，並出圖。
3. NVIDIA：
   - `JIT=1` 檢查 → S1 → S2 → S3（008；每段通過才進下一段；smoke 不取 lock，lock 被占用時不啟動）。
   - 之後在 `flock -n` 下跑 llama.cpp CUDA build，並把 6 個 quant 傳到 WSL（約 33 GB，在 Mac lock 下進行）。
   - 然後是 Stage B 的量測（B / C_ctx、超量測試、vLLM 的 JIT 計時與 pin-memory A/B），以及 §6 addendum。
4. 正式 quality（Mac）：磁碟不足，見下方 Blocked。

## Blocked / 需要使用者

0. **解除 Mac 正式 benchmark 的暫停**（使用者 2026-10-09 指示）：正式 Core A、正式 quality、任何 Mac heavy run 都等使用者通知。
   - 正式 run 時建議：電力模式設為「高效能」或至少不要用「自動」（這屬於系統設定，我不會修改），並接上 ≥ 96 W adapter。

1. **正式 quality 的磁碟空間**（Mac；不影響 Core A）
   - 目前可用 94 GiB。每個 size 的 KLD logits 約需 70 GiB，加上 `disk_guard` 的 50 GiB 保留量，共 120 GiB。
   - 即使依 STORAGE_PLAN 暫時刪掉另一個 size 的 quant（4B 約 11 GB，9B 約 23 GB；之後可從 BF16 重建），仍不夠。
     BF16 不能刪：safetensors 已刪，BF16 無法重建。
   - 需要使用者擇一：
     - 釋出約 30 GiB 以上的空間；
     - 或同意把 KLD 檔放到外接 / 其他磁碟。
   - 降低 50 GiB 保留量屬於規則變更，我不會自行修改。
2. **建議（非 blocker）**：Mac 換 ≥ 96 W 的 adapter。
   - 016 已讓長 session 在 70 W 下仍有效，但充電暫停會讓正式 Core A 從約 17 h 拉長到約 1.5–2 倍。
3. **建議（非 blocker）**：正式 Mac run 期間盡量少開前景程式。
   - 4B pilot 時前景負載的 loadavg 為 6–9，CV 5.9%；9B pilot 在深夜，CV 1.0%。
   - 前景負載不會讓 session 作廢，但會增加 noise。

已解決：遠端 SSH（2026-10-08）、WSL compiler（使用者已安裝）。llama.cpp 的 CUDA 12.8 系統套件已改用 user-space 路線，不再需要（015）。
