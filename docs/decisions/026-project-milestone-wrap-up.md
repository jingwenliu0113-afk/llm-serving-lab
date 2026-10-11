# 026 — 收尾：Core 加已完成的 Advanced 為目前的 milestone，Project 暫時完成

- 狀態：Accepted（使用者決定，2026-10-11）
- 日期：2026-10-11
- 對應：SPEC「Gate／進 Advanced」；decision 023（Advanced 範圍）、025（Advanced 結果）
- 時點：Advanced 正式結果與報告都已 commit（`41cfc15`、`6ad5221`）之後。

## 背景

- 025 完成後，我列出三個方向：收尾、另一個 Advanced 項目、追查 8K uniform 的雙模式。依 023，這些都要由使用者決定。
- 另有一個既有的測試可靠性問題：`tests/test_disk_guard.py` 的結果取決於執行機器的剩餘磁碟空間。
  這是在 Advanced 實作審查時發現的，與 Advanced 無關（024「實作補記」）。

## 決策（使用者）

1. **實驗收尾，不新增 Advanced 項目**。目前的 defensible milestone 是：
   - Core：A（兩個平台）、B（uniform 與 mixed）、quality、deployment decision table；
   - 已完成的 Advanced：H-B4 的成因隔離（025）。
   - Project 暫時完成。
2. **8K uniform P95 的雙群現象記為 Future Work / limitations，目前不追查**。
   理由：它不影響既有的部署結論。025 的事後敏感度也顯示，把兩個模式拉齊後，8K 的類別與判讀列都不變。
3. **修正 `test_disk_guard`**。這是既有測試的可靠性問題，不算擴張研究範圍。修正後跑完整測試，再 commit 並 push。

## test_disk_guard 的修正

- 原因：`scripts/disk_guard.py` 的 `check()` 把 `shutil.disk_usage` 當成預設參數，在 import 時就綁定了。
  CLI 測試 patch 的是 `disk_guard.shutil.disk_usage`，但 `main()` 呼叫 `check()` 時仍用到原本的函式，所以讀到的是真實磁碟。
  - 測試只有在剩餘空間介於 60 與 250 GiB 之間時才會通過。這台 Mac 當時剩 157 GiB，所以沒有被發現；磁碟更大的 WSL 或空間不足的機器會失敗。
- 修正（先有紅燈測試）：
  - `check()` 的 `disk_usage` 預設改為 `None`，呼叫時才取 `shutil.disk_usage`。正式執行時的行為不變：
    `run_quality`、`produce_gguf` 都沒有傳這個參數，仍讀真實磁碟。
  - CLI 測試另外檢查回報的 `free_gib` 等於 patch 的值。
  - 新增回歸測試：用兩個任何真實磁碟都不會產生的情況（剩 0 GiB、剩 10⁷ GiB），確認 `main()` 與 `check()` 都只讀 patch 的函式。
- 範圍：只改 `scripts/disk_guard.py` 與它的測試。Core 受保護的 6 個檔案（`test_core_untouched`）、
  任何 hypothesis、門檻、分析程式與已 commit 的結果都不變。

## 不變的部分

- Core 與 Advanced 的所有判定、deployment decision table、凍結的 pre-registration（§1–§7）都不變。
- 之後若要繼續（Future Work 中的任何一項，或 Optional），都要由使用者重新決定，並且先事先登記、凍結，再收集數據。

## Future Work（目前不做）

- 8K uniform 短類 P95 在 boot 之間的雙模式（約 2.48 / 2.82 秒，成因未知；025「限制」）。
- 其他 Advanced 項目：fixed-SLO 最大並發、其他 chunk threshold 或 KV 劑量、prefix caching、allocator 的操弄。
- 2K 的 H-B4 成因（025：R6，KV 可能有關但未確立；分解與操弄不一致）。
- H-A2 被推翻的原因（Core 沒有檢驗）。
- Metal 上的 quality、長 context（> 512）的 quality、W4A16 的 quality。
