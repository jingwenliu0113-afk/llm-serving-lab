# 006 — 兩階段 pre-registration（Stage A：Mac / Stage B：NVIDIA addendum）

- 狀態：Accepted
- 日期：2026-10-08
- 對應：EXPERIMENT_RULES「正式 benchmark 前先 commit pre-registration」、MACHINE_POLICY「遠端不可用時繼續 Mac 工作」
- 機器可讀：`configs/pins.toml` `[freeze] mac / nvidia`（`tests/test_pins.py` 強制各階段的必要欄位）

## 背景

原本的 `docs/PREREGISTRATION.md` 只有一次全有或全無的凍結，而凍結 checklist 包含只能在 RTX 上完成的項目
（CUDA build 驗證、WSL2 超量測試、Core B smoke test）。遠端目前不可用（001），結果會變成二選一：

- 先跑 Mac 正式數據再凍結 → 違反 EXPERIMENT_RULES；
- Mac 閒置等遠端 → 違反 MACHINE_POLICY「遠端不可用時繼續 Apple Silicon benchmark、quality」。

## 決策

Pre-registration 分兩階段 commit：

| 階段 | 何時 commit | 凍結內容 | `pins.toml` |
|---|---|---|---|
| **Stage A** | 任何 Mac 正式數據（Core A Apple Silicon、quality）之前 | **全部 hypotheses（含 RTX 與 Core B 的預測，逐字凍結）**、所有 metric 定義、Core A protocol（兩平台共用部分）、quality task set、Mac 端 pilot 決定的參數 | `freeze.mac = true` |
| **Stage B** | 任何 NVIDIA 正式數據之前 | **只能代入** Stage A 已寫定規則所需的實測值，以及純執行性的參數：RTX 的 N_thr / P、CUDA toolkit、vLLM lock、實測的 V_total / B / C_ctx / KV 池、依 007 規則算出的 Core B grid。**不得新增或修改任何 hypothesis 或規則** | `freeze.nvidia = true` |

- Stage B 是 Stage A 的 dated addendum，不是取代。`freeze.nvidia = true` 而 `freeze.mac = false` 是非法狀態（測試強制）。
- Stage B 的參數若會影響 Stage A 已凍結 hypothesis 的判定方式（例如 fit 規則的輸入），只能以 Stage A 已寫好的規則
  計算，不能改規則本身。
- RTX 的 fit 預測在 Stage A 凍結為「規則」（004：以實測 VRAM 總量與 idle baseline 為輸入），而不是寫死的估計值；
  baseline B 的**量測程序**也在 Stage A 凍結（PREREGISTRATION §4）。Core B 的長度、比例、grid 同樣是 Stage A 規則（007）。
  因此 Stage B 代入實測值之後，預測完全由規則決定，沒有事後調整空間。
- **Stage B commit 之前在 RTX 上得到的任何數字**（smoke test、`VLLM_WSL2_ENABLE_PIN_MEMORY` A/B、sysmem 超量測試、
  `--poll` 選擇）只能用於 pass/fail 與事先寫定的選擇規則，不得用來檢視任何 hypothesis 的結果。
  超量測試必須用 **18 格以外**的配置，避免提前看到 H-A1 的格子。原為 14B-Q8_0 搭配 `-c 65536`；
  依 008 改為 9B-Q8_0 搭配 `-c 262144`。

## 為什麼

- 讓 Mac 在遠端不可用期間能合法地開始正式工作，同時保持「先寫預測再看數據」。
- 所有 hypothesis 在第一筆正式數據之前就逐字凍結，因此 Stage B 晚 commit 不會帶來看結果再改預測的空間；
  Stage B 只補「Mac 上無法知道」的執行參數。

## 放棄的替代方案

- **單一凍結、等遠端**：Mac 閒置，違反 MACHINE_POLICY。
- **Mac 與 RTX 各自一份獨立 pre-registration**：RTX 端可能在看到 Mac 結果後才寫 hypothesis，等於事後預測。
- **先跑 Mac，事後補 pre-registration**：直接違反 EXPERIMENT_RULES。

## 對實驗的影響

- Stage A 之後收到的 Mac 數據，在 Stage B commit 前**不得**用來修改任何 RTX / Core B 預測。
- 報告中每一筆結果都要標明它屬於哪個 stage，以及對應 stage commit 的 git hash。
- `docs/PREREGISTRATION.md` 依此拆成 Stage A 本文與 Stage B addendum 區段。
