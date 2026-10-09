# 020 — 正式 quality 改在 RTX（wsl2-cuda）上執行

- 狀態：Accepted（quality 搬到 RTX：使用者同意，2026-10-09；4B 一起搬：我的建議，待使用者確認，確認前照此準備）
- 日期：2026-10-09
- 時點：Stage A 與 Stage B 都已凍結，任何正式 quality 數據之前。quality 只有 Mac 上的 pilot（012，遮蔽版報告）。
- 依 006，凍結後的任何變更都要記錄。本文改的是正式 quality 的**執行平台**，以及 MACHINE_POLICY 對 quality 的分工。
  不改任何 hypothesis 或其門檻，也不改 metric、task set、資料、flag set（005）。

## 背景

- 正式 quality 原本在 Mac 上跑（MACHINE_POLICY「Mac 上負責：quality evaluation」、005、PREREGISTRATION §3）。在 Mac 上卡住三件事：
  1. 磁碟：KLD logits 約 70 GiB，加上 50 GiB 保留量，共需 120 GiB；清理可重建的 scratch 後只有 94 GiB（NEXT）。
  2. Mac lock：另一個專案的 run 從 09:29 起持有 lock，正式 Mac Core A 也要排同一個 lock。
  3. 019 的電池下降停止規則是否適用 quality，尚待確認。
- 使用者問是否改用 RTX 5070 Ti。我說明了代價（下文），**使用者同意 quality 搬到 RTX**（2026-10-09）。
- 使用者另外請我對 4B 給建議。我的建議是 4B 一起搬，理由見下文；使用者沒有反對之前，照此執行。

## 決策

1. **兩個 size 的正式 quality 都在 wsl2-cuda 上執行**：RTX 5070 Ti，llama.cpp b11146（同一 commit）CUDA build，user-space CUDA 13.4.1（015）。
   - **4B 也要搬**〔我的建議，待使用者確認；正式 quality 開始前需要確認〕：H-A5(a) 是 9B-Q4_K_M 與 4B-Q8_0 在每個 WikiText chunk 上的配對 ΔNLL。
     兩個 size 若在不同平台量（Metal 與 CUDA），ΔNLL 會混進兩個 backend 的數值差異。所以兩個 size 必須在同一平台。
   - 4B BF16（7.85 GiB）放得進 VRAM，不會 spill。
2. **flag set 不變**（005：`-ngl 999 -fa on --fit off -ctk f16 -ctv f16 -b 2048 -ub 512 -c 512 -lm mmap`）。
   只有 `-t N_thr --poll P` 改用 RTX 的值：6 / 0（§4、§6）。
3. **9B BF16 放不進 16 GB VRAM：允許 spill，不降低 `-ngl`**。
   - 估算：9B BF16 檔案 16.69 GiB，CUDA 下 token_embd（1.895 GiB）留在 host，GPU 上的權重約 14.8 GiB。
     加上計算 buffer（光是 248,320 詞表的 logits 就約 0.5 GiB）與 CUDA context，約需 16 GiB；扣掉 Windows 的 idle 用量後，可用約 15.2–15.5 GiB。
     所以約有 0.5–1 GiB 會由 WSL2 / WDDM 放到系統記憶體。Stage B 的超量測試已驗證這種情況會 spill，不會 OOM。
   - spill 時所有運算仍由同一組 CUDA kernel 完成，只是部分資料改經 PCIe 存取，所以數值不受影響，只是變慢。quality 不量速度。
   - **不**用 `-ngl` 少於全部層來避免 spill：那樣會讓 BF16 參考的部分層改用 CPU kernel 計算，
     而 quant 是純 CUDA，KLD 會混入 CPU 與 GPU 的數值差異。
   - 實際的 device self / free 會從 llama.cpp 的 breakdown 照實記錄。
4. **有效性與 lock 改用 RTX 規則**：
   - GPU flock（MACHINE_POLICY）。
   - `enforced.power.limit` 必須等於 `power.default_limit`，讀不到即無效（§4、013、018）。
   - 桌機沒有電池，所以 016 的充電閘門、019 的 powermode、adapter 與電池下降規則都不適用。
     019 中「quality 是否套用電池下降停止規則」的問題因此不再存在。
5. **資料與參考檔**：
   - 兩個 BF16 GGUF 依 sha256 傳到 WSL，傳完在遠端驗證（`sync_gguf_to_wsl.py --set references`，限速以免占滿上行頻寬）。
   - quality 資料檔（WikiText、HellaSwag / MMLU / ARC 子集、WinoGrande）傳到 WSL，sha256 與 Mac 的檔案及 pins 核對。
   - KLD logits 放在 WSL（可用約 900 GB）；同一 size 算完即刪的規則不變（STORAGE_PLAN 的 S 級 scratch）。
6. **先跑 RTX 的 quality pilot**：沿用 005 的 pilot protocol（20 chunks、200 tasks、`-fa off` 檢查、重跑檢查），只看遮蔽版報告。
   pilot 要確認：
   - self-KLD = 0；
   - KLD 與 MMLU 重跑完全相同；
   - 9B BF16 實際 spill 的量與時間。

   CUDA 若出現不可重現的結果，記為 blocker 回報使用者，不改 metric。
7. **執行時機**：等 GPU 上的正式 RTX Core A / Core B session 結束後，才部署新的 `run_quality`。
   原本寫的理由（「env_capture 比對部署 manifest，中途換檔會變成 dirty」）**不正確**，2026-10-09 審查後更正：
   - 部署 bundle 會同時換掉 `DEPLOYED_COMMIT` 與 manifest，所以部署完成後 env_capture 記錄的是新 commit，而且 dirty = False。
   - 中途部署會悄悄改變之後每個 process 使用的程式（含每次重新執行的 env_capture.py），**不會留下任何標記**。
   - 因此審查後在三個 runner（`run_core_a`、`run_core_b`、`run_quality`）的正式 session 加入 provenance 檢查：
     每次 env 紀錄的 commit 必須等於 preflight 的 commit，而且 dirty 必須是 False，否則該 session 為 env-invalid；
     preflight 本身 dirty 或讀不到時，拒絕開跑。
   - 目前進行中的 RTX Core A / Core B uniform 跑的是 `b42614a`，沒有這個檢查，所以仍要遵守「中途不部署」的操作規則。

## 為什麼

- quality 的 metric 是模型輸出的確定性性質。平台只影響 kernel 的數值細節；在同一平台內比較 quant 與 BF16 的結構不變。
- 一次解決磁碟、Mac lock 與 019 待確認問題這三個卡點，不必降低任何保留量或門檻。
- 兩個 size 在同一平台，H-A5(a) 的配對比較才乾淨。

## 放棄的替代方案

- **留在 Mac**：要等使用者釋出約 10–26 GiB 空間，而且要和其他專案排 Mac lock。
- **4B 在 Mac、9B 在 RTX**：H-A5(a) 的配對 ΔNLL 會混入平台差異。
- **9B BF16 部分 offload**：BF16 參考會混入 CPU kernel 的數值。
- **改用 Q8_0 當參考**：改變 005 的 metric 定義，不允許。

## 對實驗的影響

- quality 的數字代表「CUDA backend、b11146、005 flag set」。同一平台內 quant 之間可以比較，不與任何 Mac 數字混用。
- 012 的 Mac quality pilot（時間校準、determinism、MDE）只作參考。RTX pilot 會重新確認 determinism 與時間。
- frontier 圖（Core A 速度對 quality）要註明 quality 是在 wsl2-cuda 上量的。
- 加註、不改寫原文的文件：
  - PREREGISTRATION §3 的 quality 列；
  - 005；
  - MACHINE_POLICY 的分工；
  - STORAGE_PLAN 的 KLD 段落；
  - 019 的 quality 問題。
- harness：
  - `run_quality` 加入 `platform`。wsl2-cuda 走 GPU lock、RTX 的 env 規則，沒有充電閘門；
    正式 run 需要 `freeze.nvidia` 與 §4 的 RTX N_thr / P。
  - `sync_gguf_to_wsl.py` 加入 `--set references` 與 `--max-mib-s`。
  - 以上都有測試。

## 審查（2026-10-09，review workflow：三個面向，每項發現另由對抗式驗證者複查；確認 12 項，駁回 9 項）

確認的項目都已處理，沒有改變任何規則：

- RTX 上的 quality pilot 原本不檢查 GPU lock。現在 RTX 上的 quality（pilot 與正式）一律要求 `--gpu-lock-held`。
- harness 原本仍接受 Mac 的正式 quality。現在：
  - `build_quality_plan` 與 `run_quality` 都拒絕非 wsl2-cuda 的正式 quality（沒有 platform 欄位的舊 plan 視為 Mac）。
  - `quality_analysis` 與 frontier 圖只接受 wsl2-cuda 的 quality session，analysis.json 會記錄 platform，
    frontier 圖的註記寫明 KLD 在 wsl2-cuda 上量。
  - `run_quality plan` 的 `--platform` 預設仍是 macos-metal，只供 Mac pilot 使用。
- WSL 上沒有 `data/kld/`（git-ignored，也不在部署 bundle 中）。現在 `kld_base` 會自動建立。
- WSL「可用約 900 GB」的說法經實測確認無誤：VHDX 所在的 Windows C: 剩餘 1.3 TB。已記入 STORAGE_PLAN。
- 傳輸腳本：
  - 限速器停頓後最多只保留約 1 秒的額度；
  - 驗證完成但回覆遺失時，query 會回 DONE，不再從 0 重傳；
  - 安裝 receiver 失敗時回傳 75（暫時性錯誤）；
  - `--max-mib-s` 必須 > 0。
- STORAGE_PLAN 與 MACHINE_POLICY 補上註記（Mac 上正式 quality 的段落、GPU lock 需求、「Text-to-SQL 持有 lock 時可做 quality eval」）。
- 「4B 一起搬」原本記成使用者同意，實際上是我的建議。已改為待使用者確認（正式 quality 之前要確認）。
- 「中途部署會變 dirty」的理由不正確（見第 7 點）。已更正，並在三個 runner 加入 provenance 檢查。
  Mac 的正式 Core A 另外改在固定 commit 的 git worktree 中執行（`.worktrees/<session>`，已加入 .gitignore；
  由 scratchpad 的 `batch_formal_a.sh` 建立）。這樣我在主目錄的修改與 commit 不會碰到進行中的 session。

