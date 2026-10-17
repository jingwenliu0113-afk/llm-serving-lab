# 034 — Stage C 重跑一個完整的正式 session（事先寫定的取捨規則）

- 狀態：**Accepted**（使用者 2026-10-17 決定）。本文在新 session 的任何數據之前 commit。
- 日期：2026-10-17
- 對應：030（§8，凍結 commit `b3c9109`）、031、033（第一個正式 session 的結果）

## 背景

- 第一個正式 session `formal-20261017-c1` 有效，但 U8K 為 NT（033）。
- 原因是環境：BingWallpaper 在 session 中途自行啟動，在 NVIDIA 卡上多用了約 190 MiB VRAM，造成 spill。
  - spill 偵測擋下了受影響的 run，依規則不補跑；
  - 跳過規則因此讓 U8K 的確認點都不足 n 次有效 run。
- 凍結的協定在 session 完成後沒有補跑的規定。這次重跑是研究決策，由使用者決定。

## 使用者決定（2026-10-17，看過 033 之後、任何新數據之前）

1. **以相同的凍結協定重跑一個完整的正式 session**：不改程式，也不只補 U8K。
2. **BingWallpaper**：由我在 session 開始前結束它的程序，不修改任何設定。
   它仍可能在 session 中途自行啟動；萬一發生，spill 偵測會照規則處理。

## 決策（事先寫定，不看新數據就生效）

**session**
- id：`formal-<開始日期>-c1b`。
- 部署：沿用第一個 session 的研究目錄部署 `b3c9109`，開始前核對 `DEPLOYED_COMMIT`。
- 執行方式和第一個 session 相同：GPU lock 下單一 ssh 前景指令，先跑 `capacity_plan --formal`，再跑 `run_capacity --formal`。
- 凍結的 Stage C 模組與 `configs/capacity.toml` 一個位元組都不改：
  - `tests/test_stage_c_frozen.py` 會檢查；
  - profile 產生器會用 git 再核對一次。
- C0 的分支仍是 A。SLO、時間窗、guard band、n、上限、搜尋與確認規則、workload 與 hypothesis 都不變。

**開始前的環境**
- lock 空著才開始，不搶占。
- 結束 BingWallpaper 的程序（使用者同意），之後 GPU 的 memory.used 必須 ≤ 950 MiB 才開始。這和 028 第一次自動等待用的值相同，比 031 的上限 1,018 MiB 更嚴。
- Windows CPU ≤ 20%，可用 RAM ≥ 8 GiB。
- 每次 boot 仍由 runner 執行 031 的閒置判定與 spill 偵測。

**取捨規則**（避免逐 workload 挑選結果，也就是 forking paths）
- **新 session 有效時**，四個 workload 的估計量、旗標與 hypothesis 判定**全部以新 session 為準**。
  - 有效指 `capacity_report` 接受它：formal、env 有效、沒有中止、sentinel 閘門通過、replay 一致。
  - 不論新 session 中任何 workload 的結果比第一個好或壞，都一樣以新 session 為準。
  - verified profile 改由新 session 產生；部署設定改指向新 profile。
  - 舊的 profile 從 `service/profiles/` 移除，git 歷史保留。
- **新 session 無效或中止時**，033 的結果維持為最終結果，verified profile 不變。
- **永遠不做的事**：
  - 不合併兩個 session 的 run；
  - 不逐 workload 從兩個 session 中挑選；
  - 不把兩個 session 的點或估計量合併計算。
- **兩個 session 都報告**：README 與結果的 decision log 並列兩者，第一個 session 標明「已由 c1b 取代」或「最終」。
- **只重跑這一次**：新 session 的結果無論如何，都不再有第三個 session，除非另寫 decision 並由使用者決定。
  新 session 中又出現 spill 或 NT 時，同樣照實報告。

**產品整合測試**
- 以 verified profile 跑的 IT-0 至 IT-6 延到取捨確定之後，並使用最終的 profile。
- 期間不部署、不跑任何產品測試。

## 為什麼

- **重跑的理由和結果的數值無關**：
  - 原因是已查明的環境干擾，也就是外部程式在 session 中途啟動，而且 spill 偵測也記錄到了；
  - 決定重跑時沒有看任何新數據。
- **事先規定「全部以新 session 為準」**：如果可以逐 workload 選比較好的結果，容量會偏高。
  所以把唯一的選擇變成「整個 session 是否有效」，這個條件和容量的數值無關。
- **沿用同一個部署 commit**：兩個 session 執行的位元組完全相同，差別只有環境。

## 放棄的替代方案

- **只補 U8K**：要修改凍結的 runner 與設定，等於重新凍結。不採用（使用者決定）。
- **把兩個 session 合併**：協定沒有跨 session 合併的規則，而且 boot 層級的條件不同。不採用。
- **為了這次重跑收緊 031 的上限或加新的檢查**：要修改凍結的 `gpu_idle`。改以開始前的環境控制處理，見上文。

## 對實驗的影響

- 第一個 session 的數據與 033 都不改。
- 新 session 的結果另寫一份 decision log（下一個編號），並依上面的規則決定最終結果與 profile。

## 執行紀錄

- **2026-10-17 10:50 起等待**：lock 由另一個專案（agent-gym）持有，不搶占。
- **15:31 與 17:22，lock 兩度空出**：
  - 兩次都依本文結束了 BingWallpaper，它在兩次之間又自行啟動過。
  - 15:31 之後 30 s 內，lock 又被其他專案取得，沒有開始。
  - 17:23 時 memory.used 為 567 MiB、Windows CPU 2.4%。但從我確認 lock 空著到實際取得 lock 之間約 45 s，lock 又被取得，`flock -n` 回傳 75。
  - 兩次都沒有啟動任何 vLLM，也沒有產生任何數據。
- **改進等待方式**（不涉及任何研究規則）：
  - Windows 的檢查與結束 BingWallpaper 先做，接著立即在 WSL 取得 lock；
  - memory.used ≤ 950 MiB 改在持有 lock 時檢查，不符合就以 76 結束並釋放 lock；
  - 搶不到 lock（75）或不符合（76）時繼續等待，最多 24 h。
