# 022 — 正式 Core B mixed 因缺 pandas 全部失敗：補裝 client 依賴後重跑

- 狀態：Accepted（2026-10-10）
- 日期：2026-10-10
- 時點：
  - 正式 Core B uniform 已完成並分析（`formal-20261009-coreb-uniform`）。
  - 正式 mixed 第一次執行全部失敗，**沒有送出任何 request**。
  - 正式 quality 尚未開始。
- 不改任何 hypothesis、門檻、metric、grid 或 007 的 mixed 定義；server 設定與 uniform 相同。

## 背景

- `formal-20261010-coreb-mixed`（10:04–10:10，部署 commit `2c2e764`）的 6 個 run 都是 `vllm bench serve exit 1`。
  - 原因：`--dataset-name custom` 走 vLLM 的 `CustomDataset`，它用 `pandas.read_json` 讀檔。
  - pandas 屬於 `vllm[bench]` extra，`configs/vllm.lock.txt` 沒有安裝。錯誤訊息是 `ImportError: Please install vllm[bench] for bench support`。
  - 失敗發生在讀 dataset 時，還沒有對 server 送出任何 request。
- uniform 用 `RandomDataset`，不需要 pandas，所以沒有受影響。
- **這是我的疏失**：
  - Stage B 的 smoke 與 003 gate 只測過 random dataset。
  - 013 在 WSL 上建好並驗證了 mixed dataset 檔案本身，但從來沒有用 `CustomDataset` 載入它。
  - harness 的測試只檢查 `bench serve` 的參數，沒有檢查 client 的依賴。

## 決策

1. **失敗的 session 照原樣保留**並 commit（`raw/` 除外），列在結果中，不刪除。
2. **在 WSL 的 `.venv` 補裝 pandas**：
   - 新增兩個套件：`pandas==3.0.6`、`python-dateutil==2.9.0.post0`。
   - 版本與 hash 鎖在 `configs/vllm-bench-extra.lock.txt`，pins 記為 `vllm.bench_extra_lock_file`。
   - 安裝方式：`--require-hashes --no-deps`。解析時以現有 venv 為約束，numpy 2.3.5、six 1.17.0 維持原版本。
   - 安裝前後的 `uv pip freeze` 比對：只多這兩行，其他 196 個套件不變。
3. **server 不受影響**：
   - vLLM 0.30.0 的 CUDA serving 路徑不 import pandas。只有 ROCm 的 `_aiter_ops` 與 `profiler/layerwise_profile` 以 placeholder 引用。
   - server argv 與 server env 不變，`core_b_report` 會核對它們與 uniform 相同。
   - env_capture 記錄的 venv `freeze_sha256` 在這之後的 session 會不同。這個值照實記錄，不是有效性條件。
4. **安裝後的 CPU 檢查**（不用 GPU）：
   - 用凍結程式產生的同一組 `bench serve` 參數，指向沒有 server 的 port。
   - 結果：mixed-2048 抽樣 250 個 request、mixed-8192 抽樣 200 個，等於 num_prompts = max(10C, 200)。唯一的錯誤是預期中的連線失敗。
5. **以新的 session 重跑正式 mixed**（`formal-20261010-coreb-mixed2`）：
   - 同一個 plan 規則，C 等於 uniform 的 C_knee（2K 為 25，8K 為 8）。
   - 部署不變（`2c2e764`）。
   - GPU lock 目前由另一個專案持有（10:11 起），不搶占，等 lock 釋放後才開始。
6. **021 補充（generator 第一次真實執行之前）**：
   - 規則：所有 run 都失敗的 session（`outcomes.ok == 0`）沒有任何數據，不採用。它會列在 `provenance.excluded`，原因寫「no valid run」。
   - 適用：Core A、quality、Core B uniform、Core B mixed。
   - 否則這個失敗的 mixed 會與重跑的 session 同時通過 R1 閘門，讓 generator 依 R1 拒絕執行。
   - 已加測試（先寫測試，確認失敗後才修改程式）。
7. 正式 quality 與 9B BF16 記憶體探測排在 mixed2 之後。兩者用 llama.cpp，不受 pandas 影響。

## 為什麼

- 這是讓已凍結的 007 mixed protocol 能夠執行的最小改動：不改 dataset、參數、server 或任何規則。
- 失敗的 run 沒有產生任何量測，所以重跑不涉及挑選結果。失敗的 session 仍然保留並列出。

## 放棄的替代方案

- **另建 client 專用的 overlay（PYTHONPATH）**：可讓 server 的 venv 與 lock 完全一致。
  但要修改凍結的 harness（`bench_env`）、新增未列在 §4 的環境變數，並重新部署。改動比較大。
- **改用不需要 pandas 的 dataset 類型**：會改變 007 的 mixed 定義與已 pin 的 dataset 用法，不允許。
- **重跑 uniform**：server 端沒有任何改變，沒有必要。
- **刪掉失敗的 session，或把它移出掃描路徑**：等於隱藏失敗。改用 021 的機械排除規則，讓它出現在 excluded 清單中。

## 對實驗的影響

- Core B mixed 的數據全部來自 `formal-20261010-coreb-mixed2`。
- 決策表的 excluded 清單會列出 `formal-20261010-coreb-mixed`，原因為 no valid run。
- PREREGISTRATION 開頭加註本文。
