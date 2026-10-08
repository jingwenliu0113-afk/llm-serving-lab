# 010 — Core A results schema v2：外部 peak memory 與 graph splits

- 狀態：Accepted（2026-10-08）
- 日期：2026-10-08
- 對應：PREREGISTRATION §3「peak memory」列；008「Pilot 必查」第 6 項（fused GDN 可能默默退回 CPU）；
  `docs/RESULTS_FORMAT.md` 規定改格式要升 schema 並寫 log。
- 時點：沒有任何 Core A 數據（pilot 也還沒跑），所以 v1 沒有需要轉換的結果。

## 背景

PREREGISTRATION §3 的 peak memory 定義除了 llama.cpp 的 memory breakdown，還要求外部量測：

- Mac：`/usr/bin/time -l` 的 peak memory footprint。
- RTX：`nvidia-smi memory.used` 峰值減 B。

schema v1 的 harness 沒有收集這兩項，pilot 也就無法做「memory metric 交叉核對」。
另外，008 指出 pinned llama.cpp 的 fused GDN 沒有 device probe（`auto_fgdn = false`）。
退回 CPU 的唯一可見訊號是 `graph: nodes = …, splits = …` 這一行，v1 沒有解析它。

## 決策

- **plan（schema 2）**：
  - fit job 加上 `external_memory` 欄位：`time-l`（Mac）或 `nvidia-smi`（RTX）。
  - Mac 的 fit argv 前面加上 `/usr/bin/time -l`。
  - `llama-completion` 本身的 flag 不變。
- **runner / rows（schema 2）**：
  - Mac：判定前先從 stderr 移除 time 區塊。peak footprint 與 max RSS 寫入 `external_memory`；raw stderr 原檔保留。
  - RTX：fit process 執行期間以 `nvidia-smi --query-gpu=memory.used -lms 200` 取樣，原始取樣存成 `raw/<job>.gpumem.txt`。
    `external_memory` 記峰值、第一筆與樣本數。B 依 PREREGISTRATION §4 量測，在分析階段扣除。
  - `summary.graph` 記 nodes / splits；`summary.fused_ops` 記 fused-op probe 的結果行（只在 `auto` 時出現，例如 pilot 的 `-fa auto` probe）。
- rows 增加 `block`：pilot plan（`benchmarks/core_a_pilot.py`）的區塊名稱；正式 plan 為 null。
- 外部量測缺失時，fit 的 outcome **不變**（fit / OOM 的判定仍依 004），只是該欄位為 null，分析時列為缺值。

## 為什麼

- 這是在實作已預先登記的 metric，沒有新增或修改 metric 定義。
- `time -l` 只多一個 wrapper process，不改變 llama.cpp 的行為；`nvidia-smi` 取樣是另一個 process，不碰 GPU 計算。
- graph splits 是唯一能在 log 中看出 GDN 退回 CPU 的訊號，必須在 pilot 前就記錄下來。

## 放棄的替代方案

- **在 Mac 用 `footprint` / `vmmap` 取樣**：需要另外的取樣 process 與權限，`time -l` 是 process 結束時 kernel 給出的峰值，最簡單也最準。
- **RTX 改用 NVML（pynvml）**：runner 必須只用 stdlib（WSL 與 Mac 共用同一支檔案）。
- **把 `GGML_SCHED_DEBUG=2` 加進正式 run**：§4 禁止未列出的 `GGML_*` 環境變數；它只在 pilot 有疑問時手動使用。

## 對實驗的影響

- 結果格式升為 schema 2（`docs/RESULTS_FORMAT.md`）。v1 沒有任何既有結果。
- pilot 的 memory 交叉核對可以直接比較：breakdown 的 device self 與外部量測。
  在 Mac 上，`-lm mmap` 的權重頁可能不計入 footprint，這正是 pilot 要確認的事。
