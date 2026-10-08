# 016 — Mac session 的充電暫停：電量不足時閒置充電，不讓長 session 因電量作廢

- 狀態：Accepted（2026-10-09）
- 日期：2026-10-09
- 時點：Stage A 尚未凍結，沒有任何 Mac 正式數據。只影響 harness 的排程，不改任何 hypothesis、metric、門檻或 grid。

## 背景

- 012 規定：任何 process 前後的 env 紀錄出現電量 < 20%，整個 session 就是 env-invalid，必須重跑。
- 這台 Mac 接的是 70 W adapter，滿載時供電不足，接著 AC 電池仍會下降：
  - 4B Core A pilot：28 分鐘內 80% → 76%。
  - 9B Core A pilot（2026-10-08 23:28–00:06）：38 分鐘內 **100% → 73%**。
  - 接著跑的 quality pilot：約 30 分鐘內 73% → 60%。
- 正式 Core A 的時間上限估計為 17.39 小時（9B pilot 校準），quality 也要好幾個小時。
  用這個 adapter 一路跑下去，2–3 小時內就會低於 20%，session 一定作廢。

## 決策

1. **充電閘門**：Mac 上（Core A 的 `macos-metal`，以及 quality），每個 process 開始前讀一次 `pmset -g batt`。
   - 在 AC 上且電量 < **40%** 時，閒置充電，每 60 秒檢查一次，直到 ≥ **90%** 才開始這個 process。
   - 最多等 **4 小時**。超時就照常繼續，由 012 的規則判定有效性。
   - 不在 AC 上，或讀不到電量時，不暫停；這兩種情況由 012 / 013 的規則判為 env-invalid。
2. 每次暫停記入 `session.json` 的 `charge_pauses`，內容包括：job、起始與結束電量、等待秒數、是否超時。
3. RTX 桌機沒有電池，不讀電量。
4. 012 的有效性規則不變：仍然是 AC、電量 ≥ 20%、沒有 thermal / performance warning。
   閘門只是讓長 session 能在規則內完成，不放寬任何條件。

## 為什麼

- 暫停只發生在 process 之間，不影響任何 process 內的量測。
- cell 的執行順序仍是預先以 seed 隨機排列，暫停不會改變順序，也不會系統性地偏向某些 cell。
- 40% 的門檻留了足夠餘量：pilot 中單一 process 最長只有幾分鐘，以每小時 40% 的耗電估算，跑完仍遠高於 20%。
- 90% 讓每次暫停之後可以連續跑一段時間，減少暫停次數。

## 放棄的替代方案

- **降低 012 的電量門檻**：等於放寬有效性規則，不可行。
- **把正式 session 拆成多個短 session**：會改變 sentinel drift 的判定單位，也是 protocol 變更，比本方案大。
- **等使用者換更大的 adapter**：仍是建議（≥ 96 W），但不應讓 Core 停在這裡。換了 adapter 之後，閘門自然不會觸發。

## 對實驗的影響

- 正式 Mac session 的實際時間會變長，因為多了充電暫停。估計 Core A 從約 17 小時增加到約 1.5–2 倍。
- 暫停之後 SoC 會比較涼，可能讓暫停後第一個 process 稍快。process 本身有 warmup，`-r 5` 也取平均；
  這部分的影響由 process-level 的 SE 與 sentinel drift 吸收。報告時要列出暫停次數與位置。
- PREREGISTRATION §4 的電源有效性段落加註本閘門。
