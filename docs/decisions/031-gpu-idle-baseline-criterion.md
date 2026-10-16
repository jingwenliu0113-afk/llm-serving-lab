# 031 — RTX 閒置基線的調查，與新的 GPU 閒置判定方式（DRAFT，待使用者確認）

- 狀態：**Accepted**（使用者 2026-10-16 確認；見文末「使用者決定與最終規格」）。原本的草稿內容保留如下。
- 日期：2026-10-16
- 對應：MACHINE_POLICY「GPU Hard Lock」、decision 027 §5、028 停止規則 2、030 8.6、029 的 deploy preflight
- 時點：C0 第一次嘗試（10-16 01:10）依停止規則 2 在任何 boot 之前停止之後。使用者要求確認閒置基線，並在必要時提出新的判定方式。
- 範圍：只影響 Phase 2 尚未產生數據的工作（C0、C1、產品測試）。Phase 1 的程式、結果與門檻都不改（`run_core_b.IDLE_MIB` 與 `run_adv.IDLE_MIB` 是凍結的程式，維持 1,000）。

## 調查（2026-10-16 16:30 左右，全部唯讀）

**WSL**
- WSL 剛被 ssh 喚醒，沒有任何其他專案的程序（沒有 ollama、vLLM、torch 或其他 python 工作），load 0.00，RAM 空閒 30 GB。
- GPU lock 沒有人持有。
- `nvidia-smi` 1 Hz 取樣 30 s：memory.used 固定 1,385 MiB，utilization.gpu 0%，P8，4.8–8.9 W，38 °C；看不到任何 compute process。

**Windows 主機**（效能計數器）
- GPU 引擎：5 次取樣中，沒有任何程序的使用率超過 0.5%。
- CPU 1.6–2.1%，RAM 空閒 21.8 GB（共 31.9 GB）。Windows 已運行 1 天 7 小時。
- **NVIDIA RTX 5070 Ti 在驅動 4K（3840×2160）螢幕**；AMD 內顯沒有接螢幕。所以桌面合成與所有桌面程式都在 NVIDIA 卡上。
- NVIDIA 卡的 dedicated usage 為 1,396 MiB，**shared usage 32.7 MiB**。
- 依程序分（NVIDIA 卡，MiB）：

  | 程序 | MiB | 類型 |
  |---|---|---|
  | dwm | 598.6 | Windows 系統 |
  | explorer | 314.9 | Windows 系統 |
  | ChatGPT | 222.0 | 使用者應用程式 |
  | msedgewebview2（兩個） | 205.8 | 應用程式內嵌的 Edge |
  | BingWallpaper | 134.0 | 使用者應用程式 |
  | RadeonSoftware | 105.1 | AMD 驅動的控制介面 |
  | 其他（csrss、TextInputHost、StartMenu、LockApp、Settings、codex-computer-use 等） | 每個 11–38 | 多為系統 |

**Phase 1 的閒置基線**（已 commit 的 env 擷取）：
- 各 session 的閒置 memory.used 在 154–810 MiB 之間，中位數 186–617 MiB。
- 現在的 1,385–1,396 MiB 比 Phase 1 任何一個 session 都高。差別來自桌面上開著的應用程式，不是計算負載。

**結論：約 1.39 GB 是穩定的閒置占用**：沒有計算負載，也沒有其他專案的程序。它不是異常負載，但比 Phase 1 多出約 700–1,200 MiB 的 VRAM。

## 為什麼不能只看基線偏移與使用率

vLLM 在 Core B 設定（K = 4 GiB、U 0.91）下的實測記憶體（Advanced 的 K4 boot，memory.used 減去該 boot 之前的閒置值）：

- 就緒時：+14,635 MiB；
- bench 跑完後：**最多 +15,157 MiB**（K3 為 +14,182）。

在現在的閒置值下：1,385 + 15,157 = **16,542 MiB，超過整張卡的 16,303 MiB 約 240 MiB**。
- WDDM 不夠時不會報錯，而是把某些配置移到系統記憶體（Stage B 的超量測試已確認是 spill，不是 OOM）。
- 這會悄悄改變延遲，而且和 Phase 1 不可比。
- 要讓峰值放得下，閒置用量最多只能是 16,303 − 15,157 = 1,146 MiB，而且這是沒有任何餘裕的情況。
- 所以 Phase 1 的 1,000 MiB 實際上就是「VRAM 餘裕」的限制，不是隨意的數字。只看使用率與基線偏移（例如「比基線多不超過 X」）會放行會 spill 的情況。

## 提議的判定方式

### 研究（C0、C1）：綜合判定，以實測的 vLLM 峰值推導餘裕

每次 boot 前，以 1 Hz 取樣 30 s，下列全部成立才開始：

1. **共享 GPU lock**：本 process 確實持有（驗證 fd），和現在一樣。
2. **沒有計算負載**：每個樣本的 `utilization.gpu` ≤ 5%，WSL 看不到 compute process。
3. **基線穩定**：30 s 內 memory.used 的最大值減最小值 ≤ 64 MiB，代表沒有程式正在載入或釋放 VRAM。
4. **VRAM 餘裕**（取代固定的 1,000 MiB）：閒置 memory.used ≤ V_total − F_peak − margin。
   - V_total = 16,303 MiB；
   - F_peak = 15,157 MiB（Phase 1 實測，Core B argv、K 4 GiB，bench 後的最大增量）；
   - margin = 128 MiB。
   - 目前推得的上限為 **1,018 MiB**。數值和舊門檻幾乎相同，但改由實測推導，日後 argv 或 K 改變時要重新推導。
5. **新增：直接偵測 spill**（boot 期間）：
   - 在 boot 前與每個 bench 後，讀 Windows 的 NVIDIA adapter shared usage（目前閒置為 32.7 MiB）。
   - 比 boot 前增加超過 64 MiB 時，該 bench 與之後的 bench 都無效（研究），並停止該 boot。
   - 這是門檻 4 的直接驗證：門檻 4 預測不會 spill，這一步確認實際上沒有 spill。

另外，執行前先看主機負載（只決定何時開始，不是有效性規則）：Windows 的 CPU 10 s 平均 ≤ 20%，RAM 空閒 ≥ 8 GiB。
Mac 只負責 ssh，不影響 RTX 的量測；但我在 Mac 上跑完整測試等較重的工作之前，也會先看 Mac 的負載（使用者 2026-10-16 提醒：Mac 上有其他工作在跑，但不使用 lock）。

### 產品測試（capserve 的 IT 與 loadtest）：可以採用基線偏移

產品的整合測試驗證的是功能（串流、斷線、超載時回 429、drain），不是延遲；它的輸出也不是容量。因此：

- 判定改為：lock、沒有計算負載（同上 2）、基線穩定（同上 3）。
- 不要求餘裕 4；若餘裕不足，記錄為警告，並描述性地記錄 shared usage 的變化。
- IT-5 的「GPU 記憶體回到閒置」改為「回到 boot 前的基線 + 64 MiB 以內」，不再用固定的 1,000 MiB。

## 解除研究阻塞的選項（需要使用者決定）

| 選項 | 內容 | 評估 |
|---|---|---|
| A（建議） | 執行研究的期間，由使用者關閉不需要的**使用者應用程式**：ChatGPT、使用內嵌 Edge 的 app、BingWallpaper、Radeon Software、設定、codex-computer-use | 合計約 700 MiB，預期閒置約 690 MiB，和 Phase 1 相當。不碰 Windows 系統程序（dwm、explorer、csrss 等） |
| B | 把螢幕改接主機板（AMD 內顯），讓 NVIDIA 卡不負責桌面 | 最穩定，閒置可能接近 Phase 1 的低值；但要改變硬體接線 |
| C（不建議） | 在現在的閒置值下先跑一次約 20 分鐘的診斷，同時用 spill 偵測，看 WDDM 會不會改把桌面的配置移出 | 依上面的算術預期一定會有東西被移出；即使只移出桌面的配置，adapter 的 shared usage 也會上升，研究的 run 仍會判無效 |

## 不變的部分

- Phase 1 的所有程式、結果與門檻不變。
- 028 停止規則 2 與 030 8.6 的「≤ 1,000 MiB」只有在使用者確認後才改寫為上面的綜合判定；改寫時保留舊文字並標註。
- 不關閉任何 Windows 系統程序，不搶占其他專案的 GPU。

## 使用者決定與最終規格（2026-10-16）

**使用者決定**
1. 研究（C0、Stage C）採用綜合判定。
2. 解除方式：使用者表示桌機上已經沒有開任何應用程式。唯讀查詢顯示，占用 NVIDIA VRAM 的程序都沒有視窗，是登入時自動啟動的背景或系統匣程式：
   - ChatGPT（10-15 09:18 起，共 15 個程序，含子程序 codex-computer-use）；
   - BingWallpaper 與它的內嵌 Edge；
   - Radeon Software；
   - 背景暫停的「設定」。
   另一組內嵌 Edge 屬於 Windows 搜尋（SearchHost），算系統元件，不動。我不會關閉任何程序；是否從系統匣結束這些常駐程式，由使用者決定。
   在閒置值降到餘裕上限以下之前，研究維持 blocked。
3. 產品的整合測試改用基線判定，現在就在目前的閒置值下執行。

**研究的最終規格**（取代 028 停止規則 2 與 030 8.6 的「memory.used ≤ 1,000 MiB」）

每次 boot 前取 30 個 1 Hz 樣本，下列全部成立才開始；最多試 3 次（約 90 s），都不成立就以 `gpu_not_idle` 停止，語意和舊的 `gpu_busy` 相同：
- 本 process 確實持有共享 GPU lock；
- 每個樣本的 utilization.gpu ≤ 5%，WSL 看不到 compute process；
- 30 個樣本的 memory.used 最大值減最小值 ≤ 64 MiB；
- 閒置 memory.used ≤ V_total − F_peak − margin。
  - V_total 讀 pins `[stage_b].v_total_mib`（16,303）；
  - F_peak = 15,157 MiB（Phase 1 實測，有出處）；
  - margin = 128 MiB；
  - 上限由程式計算，目前是 1,018 MiB。

**spill 偵測（boot 期間）**
- 讀取方式：從 WSL 呼叫 `powershell.exe`，讀 `\GPU Adapter Memory(*)\Shared Usage`，取 dedicated usage 最大的 adapter（NVIDIA 卡；LUID 每次開機可能不同，所以每次都重新判斷）。
- 讀取時機：boot 前（閒置判定通過之後）、server 就緒後，以及每個 bench 之後。
- 任何一次讀數比 boot 前多出 64 MiB 以上即為 spill。這時該 bench 無效，停止該 boot，之後計畫中的 bench 也記為無效（不補跑）。
- 讀取失敗或逾時（20 s）時，比照遙測中斷，fail closed：停止。
- WSL 執行中時，閒置的 shared usage 約 64 MiB（WSL 關閉時約 33 MiB），所以一律以同一次 boot 開始前的讀數為基準。

**主機負載**：只決定何時開始，不是有效性規則，在 runner 之外檢查。Windows 的 CPU 10 s 平均 ≤ 20%、RAM 空閒 ≥ 8 GiB 時才開始；Mac 的負載只供參考。

**產品的最終規格**（capserve 的 deploy、itest、loadtest）
- preflight：本 process 確實持有 lock；30 s 內 utilization.gpu 每個樣本都 ≤ 5%；memory.used 的最大值減最小值 ≤ 64 MiB。
  - 記錄閒置基線。
  - 餘裕不足（閒置 > 推得的上限）時記為警告，不拒絕。
- shared usage 在 boot 前、就緒後與測試後都記錄，只作描述。
- teardown 與 IT-5：GPU 記憶體在 60 s 內回到「boot 前基線 + 64 MiB」以內，取代固定的 IDLE_MIB。
- 產品的輸出仍只寫到 `runs/`，不是容量量測。


## 執行紀錄

- **2026-10-16 22:10**：使用者結束了 ChatGPT 與 BingWallpaper，閒置值降為 1,043 MiB（穩定、0%），仍高於上限 1,018 MiB。
  其餘非系統程式：Radeon Software 105 MiB、工作管理員 19 MiB、Chrome 遠端桌面 44 MiB。
- **2026-10-16 22:1x**：依使用者指示（「你去關」）結束系統匣的 Radeon Software。
  - 它拒絕一般的結束要求（taskkill 回報「只能強制終止」），所以改以 `taskkill /F` 強制結束；工作管理員已經不在執行。
  - Chrome 遠端桌面不關，因為使用者可能正透過它連線；AMD 的服務與 Windows 系統程序都不碰。
  - Radeon Software 下次登入時會自動啟動；開機自動啟動的設定沒有改。
- 之後的閒置值：**901 MiB**，30 s 內最大減最小為 0 MiB，使用率 0%，沒有 compute process，符合研究的綜合判定（≤ 1,018 MiB）。

## 實作補記（2026-10-16）

依「使用者決定與最終規格」以 TDD 實作：`benchmarks/gpu_idle.py`、C0 與 Stage C 的 runner、`capacity_report`、capserve 的 deploy 與 itest。
審查有 8 項發現，6 項修正，2 項依下文決定。

- **spill 的邊界包含 64 MiB**：「多出 64 MiB 以上」中的「以上」包含本數，程式為 reading ≥ pre + 64。草稿中「增加超過 64 MiB」的寫法以此為準。
- **研究**：
  - 樣本缺少 utilization 或 memory 時，該窗不通過；
  - 同一次 boot 中 adapter 的 LUID 改變時，視為讀不到；
  - C0 與 Stage C 的細節見 028 與 030 的對應補記。
- **Stage C 階段 S 的 boot 在就緒時就 spill**：沿用 030 8.4 的重試（理由見 030 的實作補記三）。這是凍結前、沒有任何數據時的實作對應，由我決定並記錄。
- **產品**：
  - 一次 30 s 的窗；
  - 也拒絕有 compute process 的情況（草稿中的「沒有計算負載（同上 2）」）；
  - 缺少 `v_total_mib` 的 pins 拒絕；
  - `max_hold_s` 包含 preflight 的窗；
  - shared usage 的三次讀數記在 meta.json，只作描述。
- **補充：產品的閒置上限 2,048 MiB**（審查發現，我依「不影響其他專案」加上，可由使用者撤回）：
  - 原本的產品規格只有警告，沒有上限。這樣若其他專案的模型以 0% 使用率常駐在 VRAM（WSL 不一定列得出它的程序），產品測試仍會啟動 vLLM 並擠到對方。
  - 閒置用量 > 2,048 MiB 時拒絕。這約比今天最重的桌面基線（1,396 MiB）高 650 MiB，任何常駐的模型都會超過它。
  - 在研究的餘裕上限與 2,048 MiB 之間仍只發警告。
- **執行時間**：每次 boot 至少多約 30 s（最多約 90 s）的閒置窗；Stage C 約 40 次 boot，合計多約 20 分鐘。
- **尚未在 RTX 上驗證**：從 WSL 呼叫 `powershell.exe` 的 `read_shared_usage`。執行前會先在 WSL 上唯讀跑一次。

## 執行紀錄二（2026-10-16 22:25–22:32）

- 第一次批次（部署 `635b954`）：
  - C0、mini smoke 與 FS-order smoke 都在第一次 boot 之前，以 `spill_unreadable` 停止（fail closed），沒有啟動任何 vLLM。
  - 原因：WSL 執行 Windows 程式的 binfmt 註冊（`/proc/sys/fs/binfmt_misc/WSLInterop`）在 WSL 開機後幾分鐘內消失，呼叫 `powershell.exe` 得到 `Exec format error`。22:24 手動驗證時（開機後約 1 分鐘）還能讀。這是啟用 systemd 的 WSL 的已知問題。
  - 永久修好需要用 sudo 改 binfmt 的設定，屬於系統設定，不由我修改。
- **修正（不碰系統設定）**：binfmt 的註冊本來就只是執行 `/init <exe> <args>`。`read_shared_usage` 遇到 ENOEXEC 時，改由 `/init` 再呼叫一次；其他錯誤仍 fail closed。已在 WSL 上實際驗證 `/init powershell.exe` 能讀到計數器，並有回歸測試。
- 同一批次的**產品整合測試 IT-0 至 IT-6 在 RTX 上全部通過**。輸出在 WSL 的 `~/llm-serving-service/runs/pt-20261016T142720Z-itest`，只是產品測試，不是容量量測。
  產品端的 shared usage 讀數因同樣的原因失敗，但只作描述，不影響結果。


## 執行紀錄三與 spill 規則的校準（2026-10-16 22:35–22:42；修正案為 DRAFT，待使用者確認）

第二次批次（部署 `f30d17c`，閒置 616 MiB，Windows CPU 0.7%）：
- 透過 `/init` 已能讀到 shared usage。
- C0 smoke、mini smoke 與 FS-order smoke 的 boot 在 server 就緒時，都被 spill 偵測擋下，沒有跑任何 bench：
  shared usage 從 35.2 MiB 升到 115.2 MiB，**+80.0 MiB**，超過容許的 64 MiB。
- **產品整合測試 IT-0 至 IT-6 再次全部通過**（`pt-20261016T143958Z`）。它記錄了完整的三次讀數：

  | 時間點 | shared usage | 比 boot 前多 | NVIDIA dedicated |
  |---|---|---|---|
  | boot 前 | 35.2 MiB | — | 625.6 MiB |
  | vLLM 就緒 | 115.2 MiB | +80.0 MiB | 15,261.5 MiB |
  | 工作負載之後 | 117.2 MiB | +82.0 MiB | 15,413.5 MiB |

**判讀**
- vLLM 就緒時 dedicated 為 15,262 MiB，離 V_total 16,303 MiB 還有約 1,040 MiB；工作負載之後還有約 890 MiB。VRAM 沒有超額，不可能是 spill。
- +80.0 MiB 在四次 boot（三次研究、一次產品）都完全相同，應是 vLLM／CUDA 在 WSL 中正常 boot 時固定的共用記憶體配置（推測是 host 端可見的配置）。工作負載之後只再多 2 MiB。
- 所以「以 boot 前為基準、容許 64 MiB」的規則，會把每一次正常 boot 判為 spill。64 MiB 是我在沒有實測的情況下提出的，校準錯了。
- 依 031，讀數失敗或 spill 時一律停止，所以這兩次批次都沒有產生任何研究數據。

**修正案（DRAFT，使用者確認之前研究不執行）**：spill 偵測改為兩段，以實測校準
1. **就緒時**：shared(就緒) − shared(boot 前) ≤ 160 MiB（實測的正常 boot 配置 80 MiB 的 2 倍）。超過即為 spill，該 boot 不跑 bench。
2. **每個 bench 之後**：shared(bench 後) − shared(就緒) < 64 MiB。達到 64 MiB 即為 spill，該 bench 無效並停止該 boot；容許值仍為 64 MiB，但基準改為就緒時的讀數。
3. 每次讀數都同時記錄 Windows 的 NVIDIA dedicated usage，只作描述，用來判讀 spill 是否和 VRAM 接近滿載同時發生。

其他規則（閒置判定、讀數失敗時 fail closed、產品只作描述）都不變。
