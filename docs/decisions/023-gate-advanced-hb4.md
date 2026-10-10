# 023 — Gate 通過：進入 Advanced，範圍限定為隔離 H-B4 的成因

- 狀態：Accepted（使用者決定，2026-10-10）
- 日期：2026-10-10
- 對應：SPEC「Gate／進 Advanced」「Advanced」；PREREGISTRATION §2 H-B4；007
- 時點：Core 全部完成。Core 的 hypothesis 判定、decision table（021，第一次真實執行 `f6048d9`）與 README 結論都已 commit。

## 背景

- SPEC 的 Gate：Core 暴露出明確、可重現的 bottleneck 或 trade-off 時，可以進 Advanced，不論結果是否符合原本的預期。
  Advanced 的目標是隔離 KV cache、batching、allocator、scheduler 中哪一個是主要成因。
- Core 量到最明確、可重現的 trade-off 是 H-B4：
  - 在 uniform 的 C_knee 加入 10% 的 32K request，短 request 的 P95 TTFT 變為 uniform 的 5.4 倍（2K，C 25）與 2.8 倍（8K，C 8）。
  - 3 次 server restart 都重現，差距超出 noise。
  - mixed 下 KV cache 使用率峰值 99.6% / 99.2%（含 GDN 狀態 block），沒有 preemption。
  - 007 把機制（chunked prefill 互相干擾）列為 conjecture，Core 不檢驗。
- 我列出三個選項：隔離 H-B4 成因、fixed-SLO 最大並發、收尾。**使用者選擇「隔離 H-B4 成因」**。

## 決策

1. Gate 通過，進入 Advanced。範圍只限 H-B4 的成因隔離；其他 Advanced 項目（fixed-SLO 最大並發、prefix caching 等）不做，除非使用者另外決定。
2. 平台：只用 RTX 5070 Ti（Core B 的 vLLM 0.30.0 環境、同一模型、同一 dataset 產生器）。Mac 不參與。
3. **事先登記**：
   - Advanced 的 hypothesis、操弄的變因、grid、判定規則與分析程式，都在任何 Advanced 數據之前寫成 decision log 與 PREREGISTRATION 的新一節，並 commit 凍結。
   - 可行性 smoke（例如某個 server 參數能否啟動）只檢查能否執行，不量 H-B4 的結果量；量測值即使出現也不看。
4. Core 的所有結果、判定與 decision table 都不因 Advanced 而改變。Advanced 的結果另外報告。
5. 已知的候選成因（只是設計的起點，不是結論）：
   - **chunked prefill 干擾**：長 prompt 分塊後占用每一步的 token 預算（`max_num_batched_tokens` 2048），短 request 的 prefill 被延後。
   - **KV 容量壓力**：長 request 每個約需 66 個 block（共 247 個可用），短 request 可能要排隊等 block 釋放。
   - 兩者的區分工具：vLLM 的 `request_queue_time_seconds` 與 `request_prefill_time_seconds`，可以把 TTFT 拆成排隊與 prefill 兩段；
     另外可以操弄 token 預算與 KV 容量（例如 fp8 KV cache，可行性要先 smoke）。

## 為什麼

- H-B4 是 Core 中效果最大、最可重現的現象，而且 Core 只量到結果、沒有量到機制。SPEC 的 Advanced 正是為了隔離這類機制。
- 只做一個項目，避免擴張研究問題（CLAUDE.md、EXPERIMENT_RULES）。

## 放棄的替代方案

- fixed-SLO 最大並發：使用者沒有選。
- 收尾：使用者沒有選。

## 對實驗的影響

- 下一步：設計 Advanced 的事先登記（decision 024 與 PREREGISTRATION 新一節）、需要的 harness 修改與測試，然後凍結，再收集數據。
