# Stage E prereg tally（041 第 3 版的揭露，descriptive_only）

041 第 3 版寫成之前，設計者已看過下列由既有數據重算的描述統計。這裡保存可以重跑的腳本與輸出，讓「已知」與「揭露」可以稽核。
這些都不是 Stage E 的數據，也不是檢定。

| 腳本 | 內容 | 數據 |
|---|---|---|
| `e1_slips.py` | 三組 8K uniform 數據的滑移段、起始 idx、送出／到達間隔 | Advanced derived、C0 derived、Core B C 8 的 raw 副本（gitignore） |
| `e1_steps.py` | Advanced 的 decode-only step 數、到達相位、position 0 的 F | Advanced derived 與 steps.jsonl 的副本（gitignore） |
| `e3_core_a_fit.py` | Core A 在 L 2048 的擬合、殘差、各格式的跨 size 截距、H-A2 配對的 φ | Core A analysis.json、pins |
| `stage_c_m2k.py` | Stage C 的 M2K 與 U2K：throughput、短類 P95、KV 峰值 | `formal-20261017-c1` |

- 從 repo 根目錄執行：`python3 results/stage_e/prereg-tally/<腳本>`；`output.txt` 是 2026-10-23 的輸出。
- 需要 gitignore 的 raw 副本的腳本，在沒有副本的機器上會失敗；輸出仍以 `output.txt` 為準。
