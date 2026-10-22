# 041 — Stage E 事先登記（DRAFT）：三個機制問題

- 狀態：**DRAFT**。harness、smoke 與審查完成後凍結；凍結前不收任何 Stage E 的正式數據。E1 是探索性的（比照 028），E2、E3 是正式的。
- 日期：2026-10-23
- 對應：037（Phase 3 範圍）、024 / 025（Advanced）、026（Future Work）、028 / 032（C0）、PREREGISTRATION §2（H-A2）
- 時點：任何 Stage E 數據之前。設計者看過 Core、Advanced、C0 的全部結果；用到的地方在各節寫明。

## 範圍

| 部分 | 問題 | 性質 | 平台 |
|---|---|---|---|
| E1 | 8K 延遲滑移的觸發點（C0：5 段中 4 段從 idx 89 開始，每次 boot 的第 1 個 bench 都沒有） | 探索性 | RTX |
| E2 | 2K 的 H-B4 成因（025 為 R6）：飽和假說、KV 劑量、allocator；另加 prefix caching | 正式 | RTX |
| E3 | H-A2 被推翻的原因（Core A：bytes 少的設定都比頻寬比例預測的慢） | 正式 | Mac 與 RTX |

## E1：8K 滑移的觸發點（探索性，比照 028）

- **已知**（032）：
  - 每段滑移都是同一組 7 個 request 多等一個約 349 ms 的 2048-token prefill step；
  - 第 2–4 個 bench 的起始組集中在 idx 89；每次 boot 的第 1 個 bench 都沒有滑移。
- **問題**：觸發在 client 端（request 送出晚了）還是 server 端（request 準時到達，但排程晚了一個 step）？
- **方法**：
  - 和 C0 相同的格子：U8K、C 8、N 200，每次 boot 4 個 bench，6 次 boot；
  - server 加上 Advanced 的觀測 flag（OTLP span 與每個 step 的紀錄，024 的 B0 觀測版）。
  - 每段滑移比較兩個時間：
    - 該組 request 在 server 端的到達時間，相對於前一組完成的時間；
    - 第一個 prefill step 的開始時間。
- **事先寫定的分類**（每段滑移）：
  - **client**：該組的到達比前一組完成晚 > 100 ms；
  - **server**：到達準時（≤ 100 ms），但第一個 prefill step 晚了 ≥ 1 個 step；
  - **其他**：都不是。
  - 5 段以上而且 ≥ 80% 屬於同一類時，報告該類為主要觸發；否則報告「無法判定」。
- **上限**：GPU lock 下最多 3.0 h；任何 boot 失敗兩次就停止。結果不改變任何既有判定，只寫進 Future Work 的說明。

## E2：2K 的 H-B4 成因（正式）

025 在 2K 的分解中，budget-long 占尾端 ΔQ 的 0.91，但 T = 512 沒有移除 ρ（f 0.08，類別 N）。
025 事後寫下一個未檢驗的可能解釋：**C 25 時系統已經飽和**，T 只是重新分配每個 step 的 token 預算，沒有減少 mixed 的總 prefill 工作量。E2 直接檢驗它。

| 部分 | condition | 格子 | boot 數 |
|---|---|---|---|
| E2a 飽和假說 | B0、T（512） | 2K，C 12，uniform 與 mixed，各 3 次 restart | 12 |
| E2b KV 劑量 | B0、K35（KV 3.5 GiB） | 2K，C 25，uniform 與 mixed，各 3 次 | 12 |
| E2c allocator | B0、A（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`） | 同上，B0 與 E2b 共用 | 6 |
| E2d prefix caching | PC 關（Core B）、PC 開 | 2K，C 25，共享前綴的 uniform（1,536 token 共享，加 384 token 各自不同），各 3 次 | 6 |

- **C 12 的理由**：Core 的 2K uniform 在 C 12 遠低於 knee（25），系統未飽和；這是 Core 已公開的曲線，在此揭露。
- **主要量**沿用 024：短類 P95 TTFT；U、M 為 3 次 restart 的平均；ρ = ln(M ÷ U)；f = 1 − ρ_c ÷ ρ_0（ρ_0 為同一 C 的 B0）。
  類別 M / P / R / W / N / NT 沿用 024 的規則與 noise 檢查。
- **Hypotheses**：

| id | 內容 | 預測 |
|---|---|---|
| H-E2a | C 12 時 T 的類別為 M 或 P（f ≥ 0.2），而 025 在 C 25 為 N | supported |
| H-E2b | ρ 隨 KV 單調：ρ(4.0) < ρ(3.5) < ρ(3.0)；ρ(3.0) 取 025 的 K3 | supported |
| H-E2c | A 的類別為 N（\|f\| < 0.2）：allocator 不是成因 | supported |
| H-E2d | PC 開時，共享前綴 uniform 的短類 P95 TTFT 下降 ≥ 20%，而且超出 noise | supported |

- H-E2b 用到 025 的 K3 數值。這是跨 session 的比較，只要 E2 的 B0 對 025 的 B0 在 noise 之內（024 的 A4 比值）才判定，否則為 NT。
- **預估**：36 次 boot，約 3.5–4 h。

## E3：H-A2 被推翻的原因（正式）

- **已知**（Core A）：推翻 H-A2 的配對是 4B-Q4_K_M 對 9B-Q8_0，觀測比 2.22–2.24，預測比 2.99，偏差約 −25%；
  兩個平台的 15 組配對中，13–15 組偏差為負。
- **兩個候選機制**：
  - **固定開銷**：每個 token 有一段和權重 bytes 無關的固定時間（kernel launch、取樣、CPU 端），t = a + bytes ÷ BW。
    這會讓越小、越快的設定越不符合頻寬比例。
  - **反量化成本**：k-quant 每個 byte 的計算量較高，所以 Q4_K_M 等設定比 bytes 預測的慢。
- **方法**：
  - **探索（用既有數據）**：每個平台以 Core A 的 decode 數據擬合 t = a + b × bytes，報告 a、b 與各 quant 的殘差。這是事後分析，只用來產生預測。
  - **確認（新數據）**：新增兩個 GGUF：
    - 9B-Q4_0：bytes 和 Q4_K_M 相近，但反量化簡單；
    - 4B-Q5_K_M：bytes 介於兩者之間的 k-quant。
    兩個平台都以 Core A 的協定量 decode（同一 runtime、flag 與 context）。
- **Hypotheses**（預測在新數據之前寫定，數值由探索步驟算出後、在凍結時填入）：

| id | 內容 | 預測 |
|---|---|---|
| H-E3a | 固定開銷模型對新設定的 decode 時間預測誤差 ≤ 10% | supported |
| H-E3b | 9B-Q4_0 的 decode 比 9B-Q4_K_M 快，而且差距超過兩者 bytes 比例所能解釋的部分（超出 noise） | 不預測方向；照實報告 |

- **成本**：在 Mac 上產生 2 個 GGUF（重負載，Mac lock，`produce_gguf`），傳到 WSL；兩個平台各約 1–2 h。

## 尚未決定（凍結前定案）

- harness 的切分：凍結的 Core A、Advanced、C0 模組只 import，不修改。
- E2d 的共享前綴 dataset：產生方式與 sha256。
- E3 的 GGUF 產生：需要 `models/hf` 的 safetensors（已在 Mac，pins 有 sha256），以及 llama.cpp b11146 的量化工具。
