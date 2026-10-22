# 037 — Phase 3：036 列出的後續全部執行（Optional 除外）

- 狀態：Accepted（使用者決定，2026-10-22；三個產品與資源的子決定，2026-10-23）
- 日期：2026-10-23
- 對應：SPEC「Phase 3」（本文新增）；decision 026（Phase 1 Future Work）、027（Phase 2 治理規則）、029（capserve 架構）、030（Stage C）、036（Phase 2 結案與後續）
- 時點：Phase 2 結案（`10761a5`）之後，任何 Phase 3 的程式或數據之前。

## 背景

- 036 結案時列出後續項目，交由使用者決定。2026-10-22 我把尚未做的事分成六類：
  1. 產品化（029 的 v1）；
  2. 研究（036 的後續）；
  3. Phase 1 的 Future Work（026）；
  4. SPEC 的 Optional；
  5. 使用者端的環境（BingWallpaper 的自動啟動）；
  6. NEXT.md 中已過時的一條建議（`/usr/bin/git` 失敗）。
- **使用者決定（2026-10-22）**：「除了 4 以外都做」。
- 開始前我詢問了三件屬於使用者的事，**使用者的回答（2026-10-23）**：
  - **服務常駐**：先做常駐需要的功能，研究期間不常駐。常駐的 vLLM 會無限期占用共享 GPU lock 與約 15 GB VRAM，研究與其他專案都會用不到 GPU。
  - **Thinking**：gateway 預設關閉 thinking，client 可以明確開啟。
  - **Mac 重負載**：可以在 Mac 上跑。使用者表示「電源已經改 140w 了」：Mac 現在接 Apple 140W USB-C Power Adapter（`pmset -g ac`，2026-10-23 協商為 94 W）。

## 決策

### 1. 範圍

**Research**（GPU 為主；每個 study 都先事先登記並凍結，才收正式數據）
- **Stage D：容量的延伸**（036 的研究後續，以及產品文件中寫明「未驗證」的兩個假設）
  - D1：O-32K，uniform-32K 的容量（030 8.3 定義的選項）。
  - D2：O-T，U8K 與 M8K 在 `--long-prefill-token-threshold 512` 下的容量（030 8.8 定義的選項）。
  - D3：128 以外的輸出長度。
  - D4：開放迴路（Poisson 到達）下的容量。
  - D5：開場湧入。研究端量它和並發、prompt 長度的關係；產品端的處理見 Engineering。
  - D6：同一 class 內 prompt 長度不一時，正式容量是否仍然成立（029「未驗證的假設」）。
  - D7：經過 gateway（含 `/tokenize` 流量）時，正式容量是否仍然成立（036「不涵蓋的部分」）。
- **Stage E：機制**（026 的 Future Work 與 C0 留下的問題）
  - E1：8K 延遲滑移的觸發點，以及 8K uniform 的雙模式（C0 只確立了機制，沒有找到觸發點）。
  - E2：2K 的 H-B4 成因（025 為 R6）。包括其他 chunk threshold、其他 KV 劑量、allocator 的操弄，以及 prefix caching。
  - E3：H-A2 被推翻的原因（Core A，兩個平台）。
- **Stage Q：quality 的缺口**（README「證據不支持的部分」）
  - Q1：Metal 上的 quality（Mac）。
  - Q2：長 context（> 512）的 quality。
  - Q3：Core B 的 W4A16 checkpoint 的 quality。

**Engineering：capserve v1**（029 的 v1 與待決事項）
- Thinking：預設關閉，client 可以開啟（使用者決定）。
- gateway 內建 tokenizer（in-process），取代對 vLLM `/tokenize` 的呼叫；必須和 vLLM 的 tokenize 結果完全一致。
- 依負載調整的 Retry-After。
- per-key quota。
- TLS。
- 常駐需要的功能：自動重啟、開機啟動的腳本、status / stop。**研究期間不常駐**（使用者決定）；研究的 GPU 工作全部結束後，等使用者同意才常駐。
- 多 class 或依 token 加權的准入：029 規定要有驗證過的 mixed 容量才做。所以設計依 Stage D 的結果，上線前要經過驗證。
- IT-3 在正式設定下的決定性證據（036 揭露的限制）：改用不依賴 128 token 輸出時間的判定方式，屬於產品測試。

**環境與文件**
- BingWallpaper 的自動啟動屬於 Windows 系統設定，由使用者處理，我不修改（033、034）。
  沒有處理時，031 的閒置判定與 spill 偵測仍會擋下受影響的 run，但那些 run 會作廢。
- NEXT.md 中「`/usr/bin/git` 會失敗」的建議已過時：2026-10-23 檢查，`/usr/bin/git --version` 正常回傳 `git version 2.54.0 (Apple Git-157)`。

**不做**
- SPEC 的 Optional：TensorRT-LLM、speculative decoding、更大的模型、功耗量測、serving 成本模型、多 GPU（使用者決定）。
- 寫新的推論引擎或 scheduler、修改 vLLM（SPEC、CLAUDE.md）。

### 2. 既有結果與凍結狀態不變

- PREREGISTRATION §1–§8、decision 001–036、`results/` 下所有既有 session、deployment decision table，一律不改、不重新分析。
- 凍結的程式維持現有測試的保護：`test_core_untouched`、`test_frozen_untouched`、`test_stage_c_frozen`。
  Phase 3 只能 import 它們，不能修改。需要不同行為時，寫新的模組。
- **正式產品 profile 不變**：`rtx-verified.toml` 仍釘住 `verified-formal-20261017-c1b.json`。
  Phase 3 的 study 可以產生新的 verified profile，例如 uniform-32k 或 T = 512 的 profile；但要改正式設定的釘選，需要使用者決定，並另寫 decision log。

### 3. 研究的規則（沿用 027 §2、§5 與 030）

- **事先登記**：每個 study 都在 PREREGISTRATION 新增獨立的一節，並寫一份 decision log。
  hypothesis、metric、負載、判定規則與停止規則都在正式數據之前寫定，和程式、測試一起 commit 後凍結。
  探索性的工作（例如 E1）照 028 的方式：事先寫定 boot 數、GPU 時數上限與停止規則，結果不當作正式判定。
- **SLO**：容量類的 study 沿用使用者為 Stage C 選定的 S2（TTFT 2 / 6 / 20 s，TPOT 100 ms，P95）。
  S2 也是正式產品 profile 的 SLO，沿用它，新的容量才能和既有的容量並列。若某個 study 需要不同的 SLO，屬於重大研究決策，會先問使用者。
- **平台**：沿用 Core B 的 RTX 5070 Ti、vLLM 0.30.0、Qwen3.5-9B W4A16 與 server 參數。只有被操弄的變因可以不同，例如 D2 的 T、E2 的 KV 劑量。
  Core A 與 quality 沿用 llama.cpp b11146 與 005 的 flag set。
- **GPU**：
  - 一律取得共享 GPU lock，不搶占，不刪除別人的 lock；GPU 空著就跑（沿用 030 的使用者決定）。
  - 開始前檢查 Mac 與 Windows 的負載；031 的閒置判定與 spill 偵測適用於所有啟動 vLLM 的 study。
  - 正式 session 進行中，不部署任何東西。
- **結果**：無效的 run 不補跑，不得只報告成功的案例，看到數據之後不改規則（013、CLAUDE.md）。

### 4. Mac 重負載（使用者決定，2026-10-23）

- 可以在 Mac 上跑重負載，條件如下：
  - 用 `tools/with-mac-lock` 取得 Mac lock；lock 被占用時就等，不搶占。
  - 接 Apple 140W USB-C Power Adapter；同一個 session 內的電力模式一致（沿用 019）。
- **019 的 70W 規定只適用於 Phase 1 的 session**：凍結的 `run_core_a.py` 要求 70W adapter，這個規定不改，它是那些 session 的紀錄條件。
  Phase 3 在 Mac 上的新 runner 改為要求 Apple 140W adapter，並記錄協商的瓦數。
  - 所以 Phase 3 的 Mac 數字和 Phase 1 的 Mac 數字不在同一電源條件下，報告時寫明。
  - E3 若在 Mac 上重新量 decode，只和同一 session 內的數字比較。
- Mac 磁碟剩約 97 GiB（2026-10-23），容不下 020 估計的完整 KLD 參考檔加上保留量（約 120 GiB）。
  所以 Q1 改用縮小的設計，細節寫在 Q1 的事先登記中。

### 5. 執行順序

GPU 只有一張，所以研究 study 依序執行；產品的 Mac 開發和它們並行。
1. **現在**：本文、SPEC、NEXT。
   - 產品中不依賴研究結果的 v1 項目（Mac，TDD）：thinking、內建 tokenizer、Retry-After、quota、TLS、常駐功能、IT-3。
   - 同時寫 Stage D 的事先登記與 harness。
2. **Stage D 的正式 session**。完成後產生新的 verified profile（只加入，不取代正式釘選），並據此設計多 class 的准入，驗證後才上線。
3. **Stage Q 與 Stage E**。Q1 在 Mac 上跑，可以和 GPU 上的 study 並行；Q2、Q3、E1–E3 依準備完成的順序使用 GPU。
4. **結案**：README、產品文件、decision log 的一致性檢查與獨立審查，和 036 相同。

預估的資源用量（粗估，各 study 的事先登記會重新估計）：
- **GPU**：約 45–65 h，分成多個 session。
  - Stage D 約 25–35 h；
  - Stage E 約 10–15 h；
  - Q2、Q3 約 10–15 h。
- **Mac**：Q1 約 10–20 h；E3 若在 Mac 上量則另計。
- 沒有付費用量；模型都已在本地。若某個 study 需要新的下載，會先寫明檔案與大小。

### 6. 自主程度（沿用 027 §6）

- 自主進行：開發、測試、審查、commit、push、更新 NEXT.md。
- 會詢問使用者的情況：
  - 重大研究決策，例如新的 SLO，或超出本文範圍的研究項目；
  - 權限；
  - 真正的阻擋問題；
  - 改變正式產品 profile 的釘選，或讓服務常駐。

## 為什麼

- 使用者決定執行 036 列出的全部後續，Optional 除外。本文把範圍、順序與共通規則寫清楚，之後每個 study 只需要寫它自己的設計。
- Stage D 排在最前面：它延用凍結的 Stage C 協定，成本最低；產品的多 class 准入也要等它的結果。
- 正式 profile 的釘選維持不變。Phase 3 產生的新容量都還沒有在產品中驗證，不應該在研究完成時就自動成為產品的上限。

## 放棄的替代方案

- **把所有研究項目放進一個大 session**：不同的 study 需要不同的 server 參數（T、KV 劑量、allocator）與不同的 runtime（llama.cpp、vLLM）。
  合併會讓一個失敗拖垮全部，也違反 SPEC「不跑沒有 hypothesis 的巨大 grid」。
- **讓服務在研究空檔常駐**：使用者沒有選。常駐期間其他專案用不到 GPU。
- **修改凍結的 `run_core_a.py` 改接受 140W**：會改變已凍結的位元組，也讓 Phase 1 的紀錄條件變得模糊。

## 對實驗的影響

- 本文只定範圍與規則，不產生任何數據。
- 下一步：
  - 產品 v1 中不依賴研究的項目，以 TDD 實作；
  - Stage D 的事先登記草稿：D1–D7 的 hypothesis、負載與判定規則。
