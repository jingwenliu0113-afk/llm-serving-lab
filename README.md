# Local LLM Serving Lab

[![CI](https://github.com/jingwenliu0113-afk/llm-serving-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/jingwenliu0113-afk/llm-serving-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![vLLM 0.30.0](https://img.shields.io/badge/vLLM-0.30.0-orange.svg)
![llama.cpp b11146](https://img.shields.io/badge/llama.cpp-b11146-lightgrey.svg)

在一台 Apple M4 Pro 與一張 RTX 5070 Ti 上，用事先登記的實驗回答：模型大小、量化、context 長度與並發，如何改變本地 LLM 的**品質、延遲、吞吐量與記憶體**？
再把量到的容量做成一個會強制執行上限的 serving gateway。

A pre-registered benchmark study of local LLM inference and serving on an Apple M4 Pro and an RTX 5070 Ti: how model size, quantization,
context length and concurrency change **quality, latency, throughput and memory** — and a serving gateway that enforces the capacity the study measured.

## 一頁摘要 / At a glance

### 做了什麼 / What was built

- **Benchmark harness**（Python）：Core A（llama.cpp，單一使用者）、Core B（vLLM，多使用者）、quality（PPL / KLD / downstream）與後續各 stage 的
  plan、runner、凍結的分析程式、報告與圖。每個 session 都記錄環境與 sha256；延遲類的 session 另記錄 GPU 閒置判定與（適用時的）sentinel 漂移。
- **capserve**：架在 vLLM 前面的 OpenAI 相容 gateway（Starlette / uvicorn），依「SLO 驗證過的容量 profile」做准入控制
  （並發上限、有界佇列、逾時、429 / 503 與 adaptive Retry-After），另有 local tokenizer、per-key quota、TLS、Prometheus metrics、
  部署工具，以及在 RTX 上執行的整合測試 IT-0–6。
- **Profile 產生器**：把凍結的分析結果轉成 verified capacity profile；測試會從來源 session 重新產生每一份已 commit 的 verified profile，
  除 generator commit 與雜湊外逐欄比對。
- 約 3,650 個自動化測試；52 份 decision log。

<!-- -->

- **Benchmark harness** (Python): plans, runners, frozen analysis code, reports and figures for Core A (llama.cpp, single user),
  Core B (vLLM, multi-user), quality (PPL / KLD / downstream tasks) and the follow-up stages. Every session records its environment and
  sha256 checksums; latency sessions also record a GPU-idle check and, where defined, sentinel drift.
- **capserve**: an OpenAI-compatible gateway in front of vLLM (Starlette / uvicorn) that admits requests by an SLO-verified capacity profile
  (concurrency limit, bounded queue, timeouts, 429 / 503 with adaptive Retry-After), with a local tokenizer, per-key quotas, TLS,
  Prometheus metrics, deployment tooling and integration tests (IT-0–6) run on the RTX.
- **Profile generator**: turns frozen analysis outputs into verified capacity profiles; a test regenerates every committed verified profile
  from its source session and compares every field except the generator commit and content hash.
- About 3,650 automated tests; 52 decision records.

### 重要發現 / Key findings

1. **單一使用者的 decode 速度不和權重 bytes 成比例**：bytes 較少的配置較快，但加速比多半小於頻寬模型的預測。
   「±20% 內符合頻寬模型」在兩個平台都被推翻：每個平台 15 組中有 1 組超出（−25.8% / −25.1%），其餘 14 組在 ±20% 內；
   原因（固定開銷或頻寬效率）沒有分開。
2. **品質與記憶體的取捨**：9B-Q4_K_M 的 WikiText NLL 比 4B-Q8_0 低（配對 ΔNLL −0.194 nats / token，Holm 校正後顯著），
   代價是權重檔多約 26%、decode 慢約 9–15%。WikiText 可能在預訓練資料中，跨 size 的差距可能部分反映記憶能力（005）。
3. **多使用者 serving**（vLLM）：2K / 8K / 32K 的吞吐量 knee 在並發 25 / 8 / 2，落在事先預測的 KV cache 容量 C_KV（35 / 13 / 3）的
   ½ 到 2 倍之間（H-B2；不是因果檢驗）。在 knee 的並發下混入 10% 的 32K request，短 request 的 P95 TTFT 增加 444%（2K）與 175%（8K）。
   8K 的主因是長 request 的 chunked prefill 占滿每個 step 的 token budget；2K 的成因沒有確立。
4. **SLO 下的最大並發寫成產品上限**：SLO 為短 request 的 P95 TTFT 2 s（2K）/ 6 s（8K）、mixed 中 32K request ≤ 20 s、P95 TPOT 100 ms，
   以穩態時間窗判定並留 5% 的 guard band。在封閉迴路、輸出 128 token 下，uniform 2K 為 32、8K 為 8、混入長 request 時為 2（不含開場湧入）；
   產生器把它寫成 profile，正式設定以 uniform-2k（32）准入。
5. **經過 gateway 的驗證**：uniform-32k（並發 2）、mixed-8k-t（4），以及輸出長度在 1–512 token 不一的 2K（32）/ 8K（8）class，經過 gateway 都符合 SLO
   （封閉迴路、只在該並發、穩態時間窗；不對開場第一波宣稱 SLO；輸出長度只用了 5 條序列；mixed-8k-t 需要尚未部署的 T server）。
   在不排隊的穩態時間窗中，同一個 request 的 client TTFT 比 gateway 量到的 upstream TTFT 多約 7 / 20 / 72 ms
   （2K / 8K / 32K，中位數，含傳輸；第一波較大），這不代表 gateway 對 SLO 沒有影響。
   8K 開啟 prefill pacing 時穩態仍符合 SLO（不是改善：穩態中 pacing 沒有約束任何 request，也沒有關閉的對照），2K 被否決；多 class 加權准入沒有通過。
6. **沒有如預測的結果也照實報告**：例如 H-A2 被推翻；W4A16 checkpoint 的 PPL 比 BF16 高 4.82%（超過事先寫定的 2%）；
   Metal 與 CUDA 的 KLD / PPL 差異在部分設定超出事先界線；Stage V 的 12 個 hypothesis 中 1 個（H-GW4）沒有如預測，
   對照組 F-U2K-var 也沒有如預測（2K 的 pacing 因此被否決），部分數值預測也落空。

<!-- -->

1. **Single-user decode speed does not scale in proportion to weight bytes**: configurations with fewer bytes are faster, but most speedups
   fall short of the bandwidth model. The "within ±20% of the bandwidth model" hypothesis is refuted on both platforms: 1 of 15 pairs per
   platform falls outside (−25.8% / −25.1%), the other 14 are within ±20%; fixed overhead vs. bandwidth efficiency was not separated.
2. **Quality vs. memory**: 9B-Q4_K_M has lower WikiText NLL than 4B-Q8_0 (paired ΔNLL −0.194 nats/token, significant after Holm correction),
   at the cost of a ~26% larger weight file and ~9–15% slower decode. WikiText is likely in the pretraining data, so the cross-size gap
   may partly reflect memorization (005).
3. **Multi-user serving** (vLLM): the throughput knee sits at concurrency 25 / 8 / 2 for 2K / 8K / 32K contexts, within the preregistered
   ½× to 2× of KV-cache capacity C_KV (35 / 13 / 3) (H-B2; not a causal test). At the knee concurrency, adding 10% 32K requests raises
   short-request P95 TTFT by 444% (2K) and 175% (8K). At 8K the cause is long requests' chunked prefill filling each step's token budget;
   at 2K the cause was not established.
4. **SLO-bound concurrency as a product limit**: the SLO is short-request P95 TTFT 2 s (2K) / 6 s (8K), ≤ 20 s for 32K requests in the
   mixed workload, and P95 TPOT 100 ms, judged on the steady-state window with a 5% guard band. With a closed loop and 128-token outputs,
   the maximum concurrency is 32 for uniform 2K, 8 for uniform 8K and 2 with long requests mixed in (start-up burst excluded);
   the generator writes them into a profile, and the production config admits by uniform-2k (32).
5. **Validated through the gateway**: uniform-32k (concurrency 2), mixed-8k-t (4), and 2K (32) / 8K (8) classes with outputs varying from
   1 to 512 tokens all meet the SLO through the gateway (closed loop, at that concurrency, steady-state window; no SLO claim for the
   start-up wave; output lengths from 5 sequences only; mixed-8k-t needs the T server, not deployed). In the steady-state window with no
   queuing, client TTFT exceeds the gateway-measured upstream TTFT of the same request by a median ~7 / 20 / 72 ms (2K / 8K / 32K,
   including transport; larger in the start-up wave); this does not show the gateway has no SLO impact. With prefill pacing on, 8K still
   meets the SLO in steady state (not an improvement: pacing never throttled a steady-state request, and there is no pacing-off control);
   2K was vetoed; weighted multi-class admission failed.
6. **Results that went against predictions are reported too**: e.g. H-A2 refuted; the W4A16 checkpoint's PPL is 4.82% above BF16
   (beyond the preregistered 2%); Metal-vs-CUDA KLD / PPL differences exceed the preregistered bounds in some settings; 1 of 12 Stage V
   hypotheses (H-GW4) and the F-U2K-var control did not come out as predicted (the latter vetoed 2K pacing), and some numeric
   predictions missed.

### 方法 / Method

- 每個階段都在看到該階段自己的正式數據之前，先寫定 hypothesis、門檻、metric、protocol 與分析程式並凍結（設定檔與程式的 sha256 由測試鎖定）；
  後面的階段是在看過前面的結果之後設計的（已揭露）。之後的任何變更都有 decision log。判定由凍結的程式機械產生；
  有效的 session 不因結果重跑（唯一的例外是 Stage C 的 c1：U8K 因環境干擾無法檢定，由使用者決定、在新數據之前寫定取捨規則後重跑為 c1b，
  兩個 session 都保留並報告，034）；失敗、OOM、無效的 run 照實報告；重要結果另做獨立重算。
  不做 Mac 與 PC 的排名：效能數字只在同一平台內比較；Stage Q 的 Metal ÷ CUDA 比值是事先登記的一致性檢查，不是排名。
- Each stage freezes its hypotheses, thresholds, metrics, protocol and analysis code before that stage's own formal data (sha256 of configs
  and code locked by tests); later stages were designed after earlier results (disclosed). Every later change has a decision record.
  Verdicts come mechanically from the frozen code; valid sessions are not rerun because of their results (the one exception, Stage C's c1,
  where U8K was not testable after environmental interference, was rerun as c1b by user decision under a selection rule written before any
  new data; both sessions are kept and reported, 034); failed, OOM and invalid runs are reported; key results are independently recomputed.
  No Mac-vs-PC ranking: performance numbers are compared only within a platform; Stage Q's Metal ÷ CUDA ratios are a preregistered
  consistency check, not a ranking.

### 技術 / Stack

Python 3.12（requires-python ≥ 3.11）、uv、pytest；llama.cpp b11146（Metal、CUDA）；vLLM 0.30.0；Starlette、uvicorn、pydantic、httpx、Prometheus client；
WSL2 Ubuntu 22.04、CUDA 13；Qwen3.5 4B / 9B（GGUF Q8_0 / Q6_K / Q4_K_M、W4A16）。

### 怎麼看這個 repo / How to navigate

- 下方「結果」是完整的結果與限制（繁體中文）；[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) 是事先登記；
  [docs/decisions/](docs/decisions/) 是 decision log；[service/](service/) 是 capserve；「重現」一節是重現步驟。
- The full results and limitations are below (in Traditional Chinese); [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) is the
  preregistration, [docs/decisions/](docs/decisions/) the decision records, [service/](service/) capserve, and "重現" the reproduction steps.

### 授權 / License

- 程式碼以 MIT 授權（[LICENSE](LICENSE)）。Code is MIT-licensed ([LICENSE](LICENSE)).
- 模型與資料集不在 repo 內；repo 只有題號清單、sha256、彙總分數，以及每題對錯、每段 NLL 等不含原文的衍生數值，沒有任何題目文字。
  各自的授權如下（不構成法律意見）：
  - Qwen/Qwen3.5-4B、Qwen/Qwen3.5-9B：Apache-2.0；GGUF 由官方 BF16 在本機轉出（不散布）。
  - RedHatAI/Qwen3.5-9B-quantized.w4a16：Apache-2.0（模型卡）。
  - WikiText-2：CC BY-SA 3.0 與 GFDL。
  - HellaSwag、MMLU、ARC、WinoGrande：取自 llama.cpp 評測用的 `ikawrakow/validation-datasets-for-llama.cpp` 與
    `ikawrakow/winogrande-eval-for-llama.cpp`（revision 見 [configs/pins.toml](configs/pins.toml)）。
    MMLU 為 MIT，ARC 為 CC BY-SA 4.0，WinoGrande 的資料為 CC BY；HellaSwag 在 llama.cpp 原始碼註記為 MIT
    （decision 005 記錄官方 repo 自 2026-09-14 起因 DMCA 被封鎖）。
- Models and datasets are not included; the repo holds only item-index lists, checksums, aggregate scores and derived per-item / per-chunk
  values (0/1 correctness, NLL), never item text. Their licenses (not legal advice):
  - Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B: Apache-2.0; GGUFs converted locally from the official BF16 (not redistributed).
  - RedHatAI/Qwen3.5-9B-quantized.w4a16: Apache-2.0 (model card).
  - WikiText-2: CC BY-SA 3.0 and GFDL.
  - HellaSwag, MMLU, ARC, WinoGrande: from `ikawrakow/validation-datasets-for-llama.cpp` and `ikawrakow/winogrande-eval-for-llama.cpp`,
    used for llama.cpp evaluation (revisions in [configs/pins.toml](configs/pins.toml)). MMLU is MIT, ARC CC BY-SA 4.0, WinoGrande data
    CC BY; HellaSwag is marked MIT in the llama.cpp source (decision 005: the official repo has been DMCA-blocked since 2026-09-14).

## 狀態 / Status

> **狀態（2026-11-01）：Phase 3 已完成**（decision 052）：036 列出的後續研究與產品 v1（SPEC 的 Optional 除外）都已完成；
> V3、T 的部署與 V5、常駐依 052 留在 Future Work，本專題不執行。
> 下文 Phase 1、Phase 2 的結果都已定案，不因 Phase 3 而改變；正式產品 profile 的釘選也不變。
> - **Phase 1 已完成**，結果保留、不重新分析：
>   - Core：A（macos-metal、wsl2-cuda）、B（uniform、mixed）、quality、deployment decision table。
>   - Advanced：隔離 H-B4 的成因（事先登記與凍結見 024，結果見 025）。
> - **Phase 2 已完成**（decision 027 的兩條線）：
>   - Research Stage C：Fixed-SLO 下的最大並發容量。事先登記已凍結（PREREGISTRATION §8、decision 030，2026-10-17）；
>     C0 的有限度排查為分支 A（decision 032）。最終結果（decision 035，重跑 `formal-20261017-c1b`）：C_PROD 為 uniform 2K 32、uniform 8K 8、mixed 2K / 8K 2。
>   - Engineering：建在 vLLM 之上的 Capacity-Aware Serving Service（`service/`），以正式設定 `service/deploy/rtx-verified.toml` 部署。
>   - 只有 Stage C 驗證過的容量（`service/profiles/verified-formal-20261017-c1b.json`）是產品限制；Phase 2 結案時其他都標為 provisional。
>     Phase 3 另產生三份 Stage D verified profile（043），只 commit、不釘選，不是正式設定。
> - **Phase 3**（decision 037–052；結果見下方「Phase 3 結果」）：
>   - 研究已完成的部分（判定或讀法都依各自凍結的規則）：
>     - Stage D（042，四個 session 都有效；RTX、直連 vLLM、封閉迴路，除非另外註明）：U32K 的 C_SLO 2 ≤ C_KV 3（H-D1 supported）；
>       T（chunk 上限 512）讓 mixed 8K 的 C_PROD 2 → 4、uniform 8K 8 → 4（H-D2a、H-D2b supported；B0 來自 c1b，跨 session）；
>       輸出 512 時 uniform 2K 的 C_SLO 32 > C_KV 30（H-D3 not supported）；
>       正式 class uniform-2k（32）在 prompt 長度不一時 FAIL（H-D6 not supported；只重播一條 pin 住的序列）；
>       經過 gateway 兩個 uniform class 都 PASS（H-D7，只對應 `capserve_sha256` `53488c6d…` 那一版）；
>       開放迴路 uniform 8K 為 0.2519 req/s（H-D4 supported），uniform 2K 為 NT。
>     - Stage E1-D（045，描述，不是檢定）：8K 雙模式的「滑移」在三個既有數據集都符合事先定義的計數規則
>       （holds，不是統計結論；C0 與 Core 只量得到五環中的兩環）。
>     - Stage E1-P（046，描述）：滑移的觸發（F 尖峰）和 prompt 內容還是 bench 中的位置綁定：無法判定（觸發太少；seed B 在 12 個 bench 中沒有尖峰）。
>     - Stage E2（050）：T 在 2K、C 8 有效（KV 4 與 2.5 GiB），在 C 25 無效；結果符合 S（負載），
>       但和負載 × KV 的交互作用分不開，C 25 的成因沒有確立（讀法 R4）。
>     - Stage E3（047，兩個平台各自判定，不比較快慢）：H-E3a（跨 size 的非比例成分）Mac、RTX 各自 supported；
>       H-E3b（Q4_K_M 對 Q4_0 的配方層級格式成本）Mac supported，RTX not supported（格式效應在兩個 size 不一致，不解讀，也不表示 RTX 沒有格式成本）；
>       依固定寫法，原因依平台而異。H-A2 仍是 refuted。
>     - Stage Q（048）：KLD 的順序在 Metal（`-c 512`）與 CUDA 2K / 8K 都成立（H-Q1a、H-Q2a supported）；
>       Metal 與 CUDA 的 KLD / PPL 差異在部分設定超出事先界線（H-Q1b 3 / 6 格、H-Q1c 1 / 8 個設定，not supported；不代表任何平台有缺陷，也不是平台排名）；
>       2K / 8K 對 512 的 KLD 比值有 4 / 12 格超出 [0.5, 2.0]（H-Q2b not supported，跨 session）。
>     - Stage Q3（049）：這一組 W4A16 checkpoint 的 PPL 比 BF16 高 4.82%，超過事先寫定的 2%（H-Q3 not supported；點估計對照門檻，不是統計檢定）。
>   - 產品：capserve v1（038）的七項已實作；v1 的 release 在 RTX 上通過 IT-0 至 IT-6 與時間有界的常駐檢查，TLS 沒有在 RTX 上測；
>     Stage D 的產品化（043）實作完成，三份 Stage D verified profile 只 commit、不釘選；
>     043 要求的 RTX IT-0 至 IT-6 於 2026-10-31 以 release `b792196` 通過。
>   - Stage V（044 / 051，Stage D 產品化的驗證研究；經過 capserve gateway、封閉迴路）：兩個正式 session 都有效。
>     uniform-32k（2）與 mixed-8k-t（4，T）經過 gateway 在 C_PROD 成立（V-GW）；輸出在 [1, 512] 均勻時 2K（32）與 8K（8）成立（V1，5 條序列）；
>     prefill pacing 只有 8K 成立（PACE-8K；2K 被 var 那一對否決），`weighted-v1` 多 class 准入不成立（2K 的 TTFT）；
>     H-GW4 與對照組 F-U2K-var 沒有如預測。新的兩份 verified profile 只 commit、不釘選。
>   - 正式釘選不變：`rtx-verified.toml` 仍釘住 `verified-formal-20261017-c1b.json`，active class 為 uniform-2k（32）。

## 研究範圍

| | Core A：single-user local inference | Core B：multi-user serving |
|---|---|---|
| Runtime | llama.cpp b11146（兩平台同一 commit） | vLLM 0.30.0 |
| 模型 | Qwen3.5 4B / 9B GGUF，Q8_0 / Q6_K / Q4_K_M（BF16 只當 quality 參考） | RedHatAI/Qwen3.5-9B-quantized.w4a16（固定一種 precision） |
| 變因 | size × quant × context（2K / 8K / 32K） | concurrency × context × request mix |
| 平台 | M4 Pro 48 GB（Metal）、RTX 5070 Ti 16 GB（WSL2 CUDA） | RTX 5070 Ti 16 GB |
| Metrics | fit / OOM / spill、TTFT、decode tok/s、peak memory、quality | TTFT、TPOT、P50 / P95、req/s、throughput、KV cache 使用率 |

- **不做**：Mac 與 PC 的排名、runtime 大亂鬥、沒有 hypothesis 的巨大 grid、自己寫 serving engine。
- **Quality**（decision 005、020）：
  - 量測項目：WikiText-2 的 PPL / KLD（對同 size 的 BF16），以及 HellaSwag / MMLU / ARC / WinoGrande 子集（描述性）。
  - 只在 wsl2-cuda 上量，視為 model × quant 的屬性，`-c 512`。

## Hypotheses（凍結於 [PREREGISTRATION](docs/PREREGISTRATION.md) §2）

- **H-A1** fit boundary 由記憶體容量決定。
- **H-A2** decode 受記憶體頻寬限制（±20%）。
- **H-A3** context 的 decode 成本隨 KV 占比增加。
- **H-A4** prefill 受算力限制（(a) quant 之間的差距、(b) 32K / 8K 的 TTFT 比值 > 4）。
- **H-A5** quality–memory frontier（9B-Q4_K_M 對 4B-Q8_0 的配對 ΔNLL；同 size 的 KLD 單調遞增；Holm 校正）。
- **H-B1** knee 之後延遲急升。
- **H-B2** KV 容量決定 knee。
- **H-B3** TPOT 的成長慢於 TTFT。
- **H-B4** request mix 提高短 request 的尾延遲。
- **Advanced**（§7，decision 024）：H-C0 在 session 內重現 H-B4；H-C1 batching（chunked prefill 壟斷 step budget）；H-C2 scheduler 的准入閘門；
  H-C3 兩者串聯；H-C4 KV 容量（推論級）；H-C5 分解與操弄一致。

## 文件

- [SPEC.md](SPEC.md)：研究問題與範圍。
- [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)：hypothesis、metric、protocol。
  分兩階段凍結：Stage A `eb6eef1`，Stage B `19446d7`。
  之後的 addendum（§7 Advanced、§8–§12 Stage C / D / Q / E / V）各自在正式數據之前凍結；Stage E 的各部分分別凍結。
- [docs/decisions/](docs/decisions/)：每個設計決策的 decision log（001–052）。
  - 021 是 deployment decision table 的推導規則，在其餘正式分析之前寫定。
  - 023 是 Gate 的決定，024 是 Advanced 的事先登記（凍結 commit `97a948c`），025 是 Advanced 的結果，026 是 Phase 1 收尾的決定，027 是 Phase 2 的兩條線與治理規則。
  - Phase 2：
    - 028 / 032 是 C0；029 是 capserve 的架構與實作補記；030 是 Stage C 的事先登記（凍結 commit `b3c9109`）；031 是 GPU 閒置判定；
    - 033 是第一個 C1 session（已被取代），034 是重跑的取捨規則，035 是最終結果，036 是 Phase 2 結案（含適用條件、IT 的限制與正式 profile 的保護）。
  - Phase 3：
    - 037 是範圍、順序與共通規則；038 是 capserve v1 中不依賴研究結果的設計（已實作；RTX 產品測試通過 IT-0 至 IT-6 與常駐檢查，TLS 沒有在 RTX 上測）。
    - 039 / 040 / 041 是 Stage D / Q / E 的事先登記，都已在正式數據之前凍結：039 於 2026-10-23（`fd95164`）；
      040 的 Q1 / Q2 與 Q3 於 2026-10-25；041 各部分分別凍結：E1-D 2026-10-23、E1-P 2026-10-25、E3 與 E2 2026-10-31
      （041 用當地日期；047 以 UTC 記 E3 的凍結為 2026-10-30 22:50:55Z）。
    - 042 是 Stage D 的結果；043 是 Stage D 的產品化（實作完成：產生器、三份不釘選的 verified profile、拒絕規則、keep-alive 修正；驗證研究轉到 044）。
    - 044 是 Stage V（Stage D 產品化的驗證研究：V-GW、V1、V2、V4）的事先登記，2026-10-31 凍結（mini `smoke-v-20261031`）；051 是 Stage V 的結果。
    - 052 是 Phase 3 的結案（正式設定不變、公開方式、授權）。
    - 045 / 046 是 Stage E1-D / E1-P 的結果（描述，不是檢定）；047 是 Stage E3 的結果；048 是 Stage Q（Q1、Q2）的結果；049 是 Stage Q3 的結果；050 是 Stage E2 的結果。
- [docs/EXPERIMENT_RULES.md](docs/EXPERIMENT_RULES.md)、[docs/RESULTS_FORMAT.md](docs/RESULTS_FORMAT.md)、[docs/STORAGE_PLAN.md](docs/STORAGE_PLAN.md)。
- [MACHINE_POLICY.md](MACHINE_POLICY.md)：兩台機器的分工與 lock。
- [configs/pins.toml](configs/pins.toml)：所有 pin。
  - 內容：runtime commit、模型 sha256、Stage B 實測值、vLLM lock。
  - 凍結狀態由測試強制。

## 目錄

```text
benchmarks/   Core A / Core B 的 plan、runner、凍結的分析程式、報告、圖、決策表
quality/      quality 的 plan、runner、分析（H-A5）、資料子集
scripts/      llama.cpp build、GGUF 產生、env capture、WSL 部署與傳輸
configs/      pins.toml 與各 lock 檔
results/      每個 session 的 plan / rows / env / analysis（raw/ 不進 git）
tests/        pytest（stdlib only 的分析程式都有測試）
tools/        with-mac-lock（Mac 重負載 lock）、RTX 的部署與取回腳本
service/      capserve：Capacity-Aware Serving Service（Phase 2 Engineering；見 service/README.md）
```

## 重現

前提：`uv`。Mac 上 llama.cpp 用 Metal build；RTX 在 WSL2 Ubuntu 22.04 上用 user-space CUDA 13.4.1（decision 015）。
正式 run 一律要拿到 lock：Mac 用 `tools/with-mac-lock`，RTX 用 `flock -n /tmp/ai-portfolio-gpu.lock`。

```bash
uv sync && uv run pytest -q
```

1. **llama.cpp 與 GGUF**：
   - `scripts/build_llama_cpp.sh metal|cuda`
   - `uv run python scripts/produce_gguf.py produce --size 4B|9B`（本地從官方 BF16 產生，不用 imatrix；sha256 寫入 pins）
2. **Core A**（每個平台一個正式 session，seed 20261008）：
   - `benchmarks/core_a_plan.py`
   - `benchmarks/run_core_a.py`
   - `benchmarks/core_a_report.py`：產生 `analysis.json`；RTX 要帶 Stage B 的 V_total / B / C_ctx。
   - `benchmarks/core_a_figures.py`
3. **Core B**（WSL，GPU lock）：
   - `benchmarks/core_b_plan.py uniform`（讀 pins 的 Stage B 值）
   - `benchmarks/run_core_b.py --formal`
   - `core_b_plan.py mixed --uniform-session …`
   - `run_core_b.py --mixed-dataset …`
   - `benchmarks/core_b_report.py`
   - `benchmarks/core_b_figures.py`
4. **Quality**（wsl2-cuda，GPU lock）：
   - `quality/run_quality.py plan --platform wsl2-cuda` 與 `run --gpu-lock-held`
   - `quality/quality_analysis.py`
5. **Deployment decision table**：`uv run python benchmarks/decision_table.py`
   - 沒有任何選擇 session 的參數。
   - 每類輸入要恰好一個有效的正式 session；每份 analysis 都會用凍結程式重算比對。
   - 輸出 `results/decision_table/`。
6. **Advanced**（WSL，GPU lock；decision 024）：
   - `benchmarks/adv_plan.py --uniform-session results/core_b/formal-20261009-coreb-uniform --out …`
   - `benchmarks/run_adv.py --plan … --out … --gdn-prefill-backend triton --gpu-memory-utilization 0.91 --formal --gpu-lock-held --mixed-dataset 2048=… --mixed-dataset 8192=…`
   - `benchmarks/adv_report.py --session … --uniform-session … --core-mixed-session … --out …/analysis.json`（在 Mac 上執行）
   - `benchmarks/adv_smoke.py`：凍結前的可行性 smoke（只輸出 pass / fail 與計數）。

各指令的完整參數見檔案開頭的 docstring。

## 結果

> 判定都由事先凍結的分析程式產生（`analysis.json`）；"not refuted" 只代表沒有被推翻，不代表被證實。
> 依 SPEC，不做 Mac 與 PC 的排名：下面的數字只在同一平台內比較。

### Core A：single-user（llama.cpp b11146）

| session | 平台 | 狀態 |
|---|---|---|
| `formal-20261009-a2` | macos-metal（M4 Pro 48 GB） | 164 / 164 ok，sentinel 漂移 1.78%，env 有效；powermode 2（高效能）、Apple 70 W adapter |
| `formal-20261009-a1` | macos-metal | 作廢：17 個 job 後觸發 019 的電池下降停止規則（100% → 85%）。之後依使用者決定刪除該規則，重跑為 a2 |
| `formal-20261009-rtx-a1` | wsl2-cuda（RTX 5070 Ti 16 GB） | 164 / 164 ok，sentinel 漂移 0.06%，env 有效 |

| hypothesis | macos-metal | wsl2-cuda |
|---|---|---|
| H-A1 fit boundary | not refuted：18 / 18 fit，沒有 spill | not refuted：18 / 18 fit，沒有 spill |
| H-A2 decode 受頻寬限制（±20%） | **refuted**：13 / 15 組偏差為負，最大 −25.8% | **refuted**：15 / 15 組偏差為負，最大 −25.1% |
| H-A3 context 的 decode 降幅 | not refuted：順序符合預測，相鄰差距在 noise 內 | not refuted：順序符合預測，差距超出 noise |
| H-A4(a) prefill 的 quant 差距小於 decode | not refuted | not refuted |
| H-A4(b) TTFT(32K) ÷ TTFT(8K) > 4 | not refuted（4.49–5.11） | not refuted（4.39–4.91） |

- **fit**：2K–32K 範圍內兩個平台都沒有碰到記憶體邊界，與 H-A1 的事前預測一致（16 GB 上 18 格都預測 fit）。
- **decode（H-A2）**：bytes 較少的配置較快，但多數配對的加速比小於純頻寬模型的預測（macos-metal 13 / 15 組、wsl2-cuda 15 / 15 組）。
  - 每個平台只有 4B-Q4_K_M 對 9B-Q8_0 這 1 組超出 ±20%（觀測比約 2.2，預測比 2.99），因此 refuted；其餘 14 組都在 ±20% 內。
  - macos-metal 上 4B-Q8_0 對 9B-Q4_K_M、9B-Q6_K 這 2 組略快於預測（+5.0%、+2.4%）。
  - 原因 Core 沒有檢驗。未檢驗的推測：有一部分成本與權重 bytes 無關，例如 GDN recurrent 狀態與 kernel 開銷。
    - Phase 3 Stage E3（047，見下方）在每個平台各自的一個 session 內重現了跨 size 的非比例成分（Mac、RTX 各自 supported；Mac 另有 Q4_K_M 對 Q4_0 的配方層級格式成本），
      但分不開固定開銷與頻寬效率，也沒有檢驗 GDN recurrent 狀態這個具體來源，所以這個推測仍未檢驗。H-A2 的判定不變。
- **context**：decode 從 2K 到 32K 的降幅，點估計在兩個平台上都是 bit 越低越大。
  - macos-metal：4B 0.25–0.28，9B 0.16–0.23；相鄰差距都在 noise 內，屬於不推翻、也不確認。
  - wsl2-cuda：4B 0.16–0.22，9B 0.12–0.15；相鄰差距都超出 noise。
- **prefill**：
  - 同一 (size, L) 下三個 quant 的 prefill 差距（macos-metal 1.04–1.07，wsl2-cuda 1.16–1.31），都小於 decode 差距（1.26–1.48）。
  - TTFT 隨 context 超線性成長。TTFT 是 llama-bench 的 engine 時間，不含 tokenize / HTTP / sampling（§3）。
- 「超出 noise」指 n = 3 個 process 時 |x − y| > 2√(SE_x² + SE_y²)（§2）；df = 2 時這個規則偏寬鬆，照凍結規則套用。
- 圖：`results/core_a/figures/{decode,prefill,memory}-{mac,rtx}-formal.svg`。

### Core B：multi-user serving（vLLM 0.30.0，Qwen3.5-9B W4A16，RTX 5070 Ti）

sessions：`formal-20261009-coreb-uniform`（102 / 102 ok）、`formal-20261010-coreb-mixed2`（6 / 6 ok）。
第一次 mixed 因 client 缺 pandas 全部失敗、沒有送出 request，保留為 `formal-20261010-coreb-mixed`（decision 022）。

| L | C_KV | C_knee | output tok/s：C_knee / grid 最大 | P95 TTFT（ms）：C = 1 / C_knee / C_max | G_TPOT / G_TTFT |
|---|---|---|---|---|---|
| 2K | 35 | 25 | 314.9 / 325.7（C 32） | 330 / 4,260 / 38,138（C_max 128） | 8.6 / 115.7 |
| 8K | 13 | 8 | 80.2 / 83.3（C 26） | 1,407 / 2,834 / 179,098（C_max 128） | 11.8 / 127.3 |
| 32K | 3 | 2 | 17.8 / 18.6（C 32） | 6,290 / 9,988 / 207,100（C_max 32） | 8.6 / 32.9 |

- **H-B1 not refuted**：三個 L 在 knee 之後 P95 TTFT 都達到 knee 的 3 倍（首次達到的 C 分別是 70 / 16 / 8）。
- **H-B2 not refuted**：C_knee 都落在 [C_KV ÷ 2, 2 × C_KV] 內，與「KV 容量（含 GDN 狀態 block）決定 knee」的預測一致；這不是因果檢驗。
- **H-B3 not refuted**：從 C = 1 到 C_max，P95 TPOT 變為 8.6–11.8 倍，P95 TTFT 變為 33–127 倍。
- **H-B4 not refuted**：在 C = C_knee 時加入 10% 的長 request（32K），短 request 的 P95 TTFT：
  - 2K：4.26 → 23.2 秒（+444%）。
  - 8K：2.83 → 7.80 秒（+175%）。
  - 兩者都超出 noise。mixed 下 KV cache 使用率峰值為 99.6%（2K）與 99.2%（8K），這個使用率包含 GDN 狀態 block；沒有 preemption。
  - 機制（chunked prefill 干擾）在 Core 是 conjecture；Advanced 的檢驗結果見下方。
  - 描述性（不屬於 H-B4）：同一個 C 下，mixed 全部 request 的 P95 TPOT 是 uniform 的 2.0 倍（2K：150 對 74.5 ms）與 1.5 倍（8K：130 對 88.8 ms）。
    output tok/s 也較低（120.8 對 314.9、60.2 對 80.2），但 mixed 含 32K prompt，不是同一負載（total token throughput 只低約 4% 與 2%）。
- 圖：`results/core_b/figures/{p95_ttft_ms,p95_tpot_ms,output_throughput}-uniform-formal.svg`。

### Quality（wsl2-cuda，`-c 512`；decision 005、020）

session：`formal-20261010-quality`（46 / 46 ok，env 有效）。quality 視為 model × quant 的屬性，兩個平台共用（Phase 1 在 Metal 上沒有量）。
Phase 3 的 Stage Q（048，見下方）在 Metal 上量了 `-c 512`：「兩個平台共用」只在 KLD 的順序上有證據（H-Q1a）；數值在部分設定超出事先界線
（H-Q1b 3 / 6 格、H-Q1c 1 / 8 個設定；不代表哪個平台有缺陷）。本表與 H-A5 都不變。

| config | KLD vs 同 size BF16（± SE） | same top-1 % | WikiText PPL（± SE，描述性） |
|---|---|---|---|
| 4B-Q8_0 | 0.00203 ± 0.00022 | 98.3 | 9.957 ± 0.072 |
| 4B-Q6_K | 0.00546 ± 0.00021 | 96.7 | 9.995 ± 0.072 |
| 4B-Q4_K_M | 0.0458 ± 0.0006 | 90.6 | 10.392 ± 0.075 |
| 9B-Q8_0 | 0.00202 ± 0.00022 | 98.4 | 8.195 ± 0.056 |
| 9B-Q6_K | 0.00726 ± 0.00043 | 96.8 | 8.214 ± 0.056 |
| 9B-Q4_K_M | 0.0574 ± 0.0008 | 90.0 | 8.198 ± 0.055 |

（BF16 參考：4B PPL 9.945、9B PPL 8.186；self-KLD 為 0。）

- KLD 是對同 size 的 BF16 算的，4B 與 9B 的 KLD 不可互比；PPL 跨 size 也只作描述（005：跨 size 只能配對比較）。
- 005 規定報告必須寫明的限制：
  - KLD ± SE 是 llama-perplexity 的輸出，假設 token 之間獨立，屬樂觀做法。
  - H-A5(b) 的 z 檢定用邊際 SE，忽略同一批 token 上的正相關，屬保守做法。
  - llama.cpp 的 WinoGrande scorer 有已知 FIXME：trailing context ≤ 3 tokens 時，第一個 logit 的位置不對。
  - downstream 分數不能與公開 leaderboard 比較：cloze 形式、沒有 few-shot、沒有 chat template。
- Q4_K_M 是不用 imatrix 的 k-quant（004），品質比常見的 imatrix 版略差；這裡的數字不代表 imatrix 量化的 Q4_K_M。

- **H-A5 not refuted**（Holm，5 個檢定都顯著且方向符合預測）：
  - (b) 兩個 size 都是 KLD(Q8_0) < KLD(Q6_K) < KLD(Q4_K_M)。
  - (a) 9B-Q4_K_M 對 4B-Q8_0 的配對 ΔNLL 為 −0.194 nats/token。
    - 代價（各自同平台比較）：9B-Q4_K_M 的權重檔大約多 26%，實測記憶體軸多 15–18%（macos-metal）與 9–11%（wsl2-cuda）。
      decode 比 4B-Q8_0 慢約 9–15%，TTFT 約為 1.5–1.9 倍。
    - WikiText、MMLU、HellaSwag、ARC 幾乎確定出現在預訓練資料中。同 size 內比較 quant 不受影響，
      但跨 size 的比較（H-A5(a)）可能部分反映記憶能力（005）。
- **downstream**（描述性，沒有多重比較校正）：40 組配對中 5 組 resolved。
  - Q8_0 對 BF16 在 4 個 task、2 個 size 上都沒有 resolved 的差異。
  - 4B-Q4_K_M 在 ARC 比 BF16 低 3.0 pp、比 Q6_K 低 3.7 pp，在 MMLU 比 BF16 低 0.7 pp。
  - 9B-Q6_K 在 ARC 比 BF16 低 0.9 pp、比 Q8_0 低 1.15 pp，都 resolved。
    - 後者與事先寫下的預期（Q8_0 對 Q6_K 在四個 task 上都不會 resolved）不符，照實報告。
    - 9B-Q4_K_M 對 BF16 在 ARC 沒有 resolved（+0.1 pp），所以這個下降不是單調的。
- 圖：`results/core_a/figures/frontier-{mac,rtx}-formal.svg`（KLD 對 decode，註明 KLD 在 wsl2-cuda 上量；4B 與 9B 的點在 quality 軸上不可互比）。

### Deployment decision table（[`results/decision_table/decision_table.md`](results/decision_table/decision_table.md)）

- 規則在看到 Mac Core A、quality、Core B 的分析之前寫定（decision 021，含審查修正）。第一次真實執行的輸出照原樣 commit。
- 「超出 noise」用 §2 的規則；quality 關係只用 H-A5 的 Holm 結果；沒有檢定過的 quality 配對永遠不當作平手。

**single-user（Core A；兩個平台的推薦相同，數字見各自的段落）**

| workload | 2K / 8K / 32K 的推薦 | 依據 |
|---|---|---|
| A-D 生成速度優先（長回答、chat） | 4B-Q4_K_M | decode 最快；代價是 4B 內 KLD 最高（0.046，對 4B BF16），4B ARC / MMLU 有 resolved 的下降；不用 imatrix |
| A-M 記憶體優先 | 4B-Q4_K_M | fit run 記憶體最小 |
| A-P 首 token 優先（長文件、RAG） | 4B-Q8_0 | TTFT 的點估計最短，多數格子超出 noise。macos-metal 8K（對 4B-Q6_K、4B-Q4_K_M）與 wsl2-cuda 2K（對 4B-Q4_K_M）的 TTFT 差距在 noise 內，這些格子由 quality 選出 4B-Q8_0 |
| A-Q 品質優先 | 4B-Q8_0 ‖ 9B-Q8_0（無法比較） | 兩者都是各自 size 內 KLD 最低；跨 size 只量了 9B-Q4_K_M 對 4B-Q8_0，所以 9B-Q8_0 對 4B-Q8_0 沒有直接證據 |

- 2K–32K 範圍內，每個平台、每個 L 的 6 個配置都 fit，而且都在前緣上（每個配置都在某個軸上有取捨）。
- 一個有證據的跨 size 結論：9B-Q4_K_M 的 WikiText NLL 比 4B-Q8_0 低（H-A5(a)；可能部分反映記憶能力，005）。
  代價（各自同平台比較）是權重檔多約 26%（實測記憶體軸多 9–18%）、decode 慢約 9–15%、TTFT 約為 1.5–1.9 倍。

**multi-user（Core B；RTX 5070 Ti，vLLM，W4A16）**

- 吞吐量導向：並發設在 C_knee（2K：25、8K：8、32K：2）。
  - C_knee 是 3 次 restart 平均上的點估計（凍結定義，不套 noise 規則）。8K 的 C = 7 只比 95% 門檻低 0.34 tok/s（約 1 個 SE），knee 落在 7 或 8 並不確定。
  - 超過 C_knee，grid 內的 output throughput 最多再增加 3.4–4.2%（依定義上限 5.3%）。
  - P95 TTFT 會急升：C_max 時為 C_knee 的 9–63 倍（2K 約 9 倍、8K 約 63 倍、32K 約 21 倍）。
- 延遲敏感：依延遲目標選並發屬於 Advanced（fixed-SLO），Core 只提供完整曲線。
- 長短 request 混合：在 C_knee 加入 10% 的 32K request，短 request 的 P95 TTFT 變為 2.8–5.4 倍（+175% 至 +444%；H-B4），全部 request 的 P95 TPOT 變為 1.5–2.0 倍。
  需要短 request 低延遲時，這是 Core 量到的主要風險；緩解方法屬於 Advanced（成因的結果見下一節，決策表本身不因此改變）。

### Advanced：H-B4 的成因（[decision 025](docs/decisions/025-advanced-hb4-results.md)）

session：`formal-20261011-adv`（60 / 60 次 boot 有效，0 逾時；凍結 commit `97a948c`）。規則在任何 Advanced 數據之前凍結（024、PREREGISTRATION §7）。

- 同一 session 內比較 5 個 server condition：
  - B0：Core 設定，加上觀測（OTLP、逐 step log）。
  - T：每個 request 的 prefill chunk 上限 512 token。
  - G：關閉 full-ISL 准入閘門。
  - TG：T 加 G。
  - K3：KV cache 從 4 GiB 降到 3 GiB。
- 格子：每個 condition × L {2K @ C 25, 8K @ C 8} × {uniform, mixed} × 3 次 restart。
- 主要量：短類 P95 TTFT 的 ρ = ln(mixed ÷ uniform)。移除比例 f = 1 − ρ_c ÷ ρ_B0；M = 移除過半，而且 ρ 與 mixed 的絕對值都下降並超出 noise；W = ρ 反而上升至少 20% 並超出 noise；N = 其他（本 session 沒有出現部分移除的 P 或 R 類）。

| | 2K（C 25） | 8K（C 8） |
|---|---|---|
| H-C0：B0 重現 H-B4 | 通過（+445%；Core +444%） | 通過（+200%；Core +175%） |
| T：chunk 上限 512 | N（f 0.08） | **M（f 1.09）** |
| G：關閉准入閘門 | **W**（f −0.44：變差；mixed 3 次共 263 次 preemption） | N（f 0.00） |
| TG | N（f 0.02） | M（f 1.09） |
| K3：KV 3 GiB | ρ +12%，未達 +20% 門檻 | ρ +22%，劑量敏感成立 |
| 判讀列 | **R6：KV 可能有關，但未確立** | **R1：主因是 batching，即長 request 的 chunked prefill 壟斷每個 step 的 budget（H-C1 成立）** |
| 分解與操弄（H-C5） | 不一致：分解指向 budget-long，但 T 的移除未達門檻（f 0.08，N）；以操弄為準 | 一致（budget-long） |

- **跨 L：成因依 L 而異**（兩個 L 落在不同列，依規則不合併）。事先預期 8K 為 R1 或 R4（符合）、2K 為 R5 或 R2 / R4（不符合）。
- **8K**：限制 chunk 到 512 後，ρ 從 1.10 降到 −0.09：mixed 的短類 P95 TTFT 從 7.8 秒降到 6.2 秒，但 uniform 也從 2.6 秒變 6.8 秒。
  - ρ 的下降中約 80% 來自 uniform 變慢，約 20% 來自 mixed 變快。依凍結規則仍是 M，因為 mixed 的絕對值也下降並超出 noise。
  - 其他代價：mixed 中長 request 的 P95 TTFT 多 17%，output tok/s 少約 4%。
  - 事後的描述：T 不是免費的修正。它主要把 uniform 拉慢，而不是讓 mixed 變快。
- **2K**：三種移除都沒有消掉過半。關閉預設的准入閘門反而使 mixed 的短類 P95 TTFT 變為 2.1 倍、output tok/s 少 43%，代表預設閘門在這裡有保護作用。
  KV 的角色只到「可能有關」：唯一的 KV 證據是關閉閘門後出現 preemption 而且變差。
  降低 KV 讓 ρ 上升 12%，未達事先定的 20%；准入受阻的觀測份額 0.10，未達 0.5。
- allocator 沒有操弄。KV 只能往下調（熱 boot 可用量上限約 4.14 GiB，sm_120 不支援 fp8 KV），所以「KV 是主因」最多只能是推論級（R5）；兩個 L 都沒有落在 R5。
- 8K uniform 的 boot 之間有兩個模式（約 2.48 與 2.82 秒，成因未知），使 8K 的 noise 偏大。事後把模式拉齊重算（敏感度計算，不是判定），8K 的類別與列都不變（025「限制」）。
- Core 的判定與 deployment decision table 都不因 Advanced 改變。
- Phase 3 的後續（025 的判定、類別與判讀列都不變）：8K 雙模式的滑移見 E1-D / E1-P（045、046，描述）；
  2K 的 T 失效見 E2（050，R4，C 25 的成因沒有確立）；T 在 fixed-SLO 容量上看到同方向的取捨（042：uniform 8K 的 C_PROD 8 → 4、mixed 8K 2 → 4，跨 session）。

### Phase 2 Stage C：Fixed-SLO 下的最大並發（decision 030 / 033–036）

**最終結果：`formal-20261017-c1b`**（035）。
- 凍結 commit `b3c9109`，C0 分支 A；8.33 h，80 / 80 個 run 有效，沒有 spill。
- SLO 為 S2：TTFT 2 / 6 / 20 s，TPOT 100 ms；穩態時間窗。
- 關鍵點要 5 次有效的 boot（c_low 為 3 次；uniform 2K 的 C_PROD 32 就是 c_low，3 / 3 PASS），每次的 R（指標 / SLO 門檻）都 ≤ 0.95，才算 PASS，也就是留出 5% 的 guard band。
  符合 SLO 但落在 guard band 內（0.95 < R ≤ 1）的點是 MARGINAL，不能算進 C_PROD。
- 有效性檢查都通過：replay、sentinel（漂移 0.19%）、跨 session 的偏移。

| workload | C_PROD（產品上限） | C_SLO | 備註 |
|---|---|---|---|
| uniform 2K | **32** | 34 | 先失守的是 TPOT。33、34 符合 SLO，但 TPOT 已在 guard band 內（STRICT_COARSE） |
| uniform 8K | **8** | 8 | 先失守的是 TPOT（C 9 為 100.6 ms） |
| mixed 2K（10% 32K 長 request） | **2** | 2 | 並發 3 時，短類 TTFT 超過門檻 |
| mixed 8K（10% 32K 長 request） | **2** | 2 | 同上 |

- **H-S1（mixed 降低容量）**：2K、8K 都 supported。
- **H-S2（C_SLO ≤ C_KV）**：U2K（34 ≤ 35）、U8K（8 ≤ 13）都 supported。
- 兩項都和事先預測相同。
  - Stage D 的補充（042 D-A2；判定不變）：c1b 的 H-S2 成立，是因為 TPOT 先失守，不是因為量到 KV 是限制。
    輸出 512 時 uniform 2K 的 C_SLO 為 32 > C_KV 30，所以 H-S2 不能延伸成「C_KV 一定是 SLO 容量的上界」。
- **開場湧入不在估計量內**：C 個 request 同時到達時，第一波的 TTFT 最高可到 10.4 s（uniform-2k）、11.0 s（uniform-8k）。
  - Stage D 的 D5（042，事後描述，來源是 c1b）：只有第一波超過門檻，C_PROD 時第二波已回到門檻內；
    只用 C ≤ C_KV 的點，第一波最大 TTFT 對 C 的斜率為 324.2 ms / request（U2K）與 1,389.2 ms / request（U8K）。量的是已暖機的 server，不是冷啟動。
  - 產品不對第一波宣稱 SLO（043）。
- **verified profile**：`service/profiles/verified-formal-20261017-c1b.json`，由產生器產生，是正式的產品上限。
  - uniform-2k 32、uniform-8k 8、mixed-2k 2、mixed-8k 2，四個都是 exact。
  - 範圍只限封閉迴路、128 個 output token、准入之後的延遲（030 8.9）。
  - Stage D 另有三份 verified profile，只 commit、不釘選，都不是正式設定（043，見「Phase 3 結果」的產品段）。
- **第一個 session `formal-20261017-c1`（033）已被取代**：
  - 原因：BingWallpaper 在 session 中途自行啟動，造成 2 次 spill，共 5 個 run 無效。結果是 U8K 無法檢定，U2K 只有下界。
  - 使用者決定以相同的凍結協定重跑，並事先寫定「新 session 有效時全部以它為準」（034）。
  - 兩個 session 能比較的部分都一致：階段 S 的選點相同，U2K 的 C_PROD 都是 32，mixed 都是 2。

**容量的適用條件**（完整列表見 decision 036）
- **硬體**：RTX 5070 Ti 16 GB（driver 617.42，同時驅動桌面），Windows 上的 WSL2；單一 GPU、單一 vLLM，GPU 上沒有其他工作。
- **軟體**：vLLM 0.30.0；`RedHatAI/Qwen3.5-9B-quantized.w4a16`（W4A16）。
  - Core B 的 server argv：max-model-len 32768、MNBT 2048、KV 4 GiB（248 個 block）、prefix caching 關閉、FLASH_ATTN。
  - 產品端每次 boot 都以 fingerprint 核對。
- **SLO**：S2，短類 P95 TTFT 2 / 6 s（2K / 8K）、全部 request 的 P95 TPOT 100 ms；mixed 的 32K request 也要 ≤ 20 s；穩態時間窗。
- **Workload**：
  - uniform：1,920 / 8,064 token 的 prompt；
  - mixed：90% 短 prompt、10% 為 32,640 token；
  - 輸出固定 128 token，temperature 0。
- **測試方式**：
  - 封閉迴路，`--max-concurrency C`，每個 run 至少 200 個 request；
  - TTFT 是准入之後的延遲，不含 client 端排隊；
  - 關鍵點各 5 次獨立 boot，c_low 為 3 次（uniform 2K 的 32 屬此）。
- **不涵蓋**：
  - 開放迴路或 Poisson 到達、其他輸出長度；
  - 其他 prompt 長度分布：uniform 只驗證了剛好 1,920 / 8,064 token，mixed 只驗證了 90 / 10 的 mix；
  - 經過 capserve gateway 的流量：容量是直接對 vLLM 的 `/v1/completions` 量的，不含 gateway 的開銷、chat template 與 `/tokenize` 呼叫；
  - 開場湧入：C 個 request 同時到達時，uniform 第一波的 TTFT 最高 10–11 s；
  - 准入前的排隊、prefix caching，以及其他 GPU、版本或模型。
- **Stage D 的補充量測**（042；正式 c1b profile 的範圍不因此擴大，上面的「不涵蓋」仍適用）：
  - uniform-2k（32）在 prompt 長度不一時 **FAIL**（直連、只重播一條 pin 住的長度序列），uniform-8k（8）PASS；
  - 經過 gateway 兩者都 PASS，但只對應 `capserve_sha256` `53488c6d…` 那一版，之後的 gateway 變更沒有量；
  - 開放迴路、輸出長度與 mixed 見「Phase 3 結果」的 Stage D 段。

**產品整合測試**（RTX；029、036、038、043）

| 測試（release `7513d62`；最終 `5201760` 重跑了 verified 的 IT-0、1、2、4、5、6 與 provisional 的 IT-0、3、6，結果相同） | verified profile | provisional-rtx（同一 release） |
|---|---|---|
| IT-0、1、2、4、5、6 | 通過 | 通過 |
| IT-3（client 斷線） | **無法判定，不算通過** | 通過（決定性） |

- **verified profile 下的 IT-3**：輸出上限 128 token，斷線時只剩約 1–1.4 s 的生成量，無法區分「被 abort」和「本來就跑完」。
  - 第一次執行記為不通過；斷線後生成確實停止，但這不構成決定性的證據。
  - 決定性的斷線驗證只來自 provisional-rtx（1,024 token）。斷線處理和 profile 無關，所以推論 verified 設定下也成立，但沒有直接證據。
  - **Phase 3 已解除這個限制**（decision 038 §2）：IT-3 改為以生成的 token 數判定。
    release `c9202c5`（2026-10-23）以 verified profile 執行時是決定性的通過：串流生成 8 / 128 個 token，非串流 49 / 128。
- **正式 profile 的保護**：`rtx-verified.toml` 以 `production_profile` 釘住 verified profile。
  - 部署與 IT 都會拒絕其他 profile，也拒絕任何允許 provisional 的執行（`--allow-provisional`、`CAPSERVE_ALLOW_PROVISIONAL=1`）。
  - provisional profile 不得使用 Stage C 的 class 名稱。
  - Phase 3（043）另加：provisional profile 也不得使用 Stage D 的 class 名稱；輸出上限大於 128 的 verified class、T profile 上 mixed-8k-t 以外的 class
    都不能作為 active class（ProfileError，任何旗標都不能覆寫）；deploy 只啟動 B0，所以 T 目前不部署。

Phase 3 的產品測試（038、043；只證明功能正常，不是容量量測：輸出只在 WSL 的 `~/llm-serving-service/runs/`，`research_valid` false）：

| release | 設定 | 測試 | 結果 |
|---|---|---|---|
| `c9202c5`（038） | verified profile（`rtx-verified.toml`） | IT-0 至 IT-6 | 全部通過。IT-3 改以生成的 token 數判定後是決定性的通過（串流 8 / 128、非串流 49 / 128）；IT-4 的 23 項計數全部相等 |
| `c9202c5`（038） | provisional-rtx（`rtx.toml`） | IT-0 至 IT-6 | 全部通過；IT-3 為 8 / 1,024 與 46 / 1,024 |
| `f670c22`（038） | verified profile，`tokenizer = "local"` | IT-0 至 IT-6 | 全部通過（gateway 在 local 模式下實際啟動） |
| `4711390`（038） | 釘選的 profile，常駐模式 | 時間有界的常駐檢查 | SIGKILL vLLM 後自動重啟並再度就緒；stop 後沒有再重啟 |
| `b792196`（043 第 8 步，2026-10-31） | verified profile（`rtx-verified.toml`） | IT-0 至 IT-6 | 全部通過（keep-alive 修正改變了 `capserve_sha256`，所以重跑） |

- IT-4 的計數一致之後，兩份 RTX 設定才改為 `tokenizer = "local"`（038 §4）；計數一致只限 RTX 上測過的 23 個案例。
- keep-alive 修正與之後的 gateway 變更對延遲的影響沒有量過；`b792196` 的 IT 只證明功能正常。
- TLS 沒有在 RTX 上測（正式設定只綁 loopback，沒有 TLS）。

### Phase 3 結果（decision 037–052）

- 每個 study 都在正式數據之前事先登記並凍結（037 §3）；判定或讀法由凍結的分析程式產生，執行上的偏離寫在各段。
- supported / not supported 是套用事先寫定規則的結果：多數是逐點類別或點估計對照門檻，Stage Q 的順序是 Holm 校正的 z 檢定；NT 是 not testable。
  E1 是探索性的描述（037 §3），結果不當作正式判定。
- Phase 1、Phase 2 的判定、deployment decision table 與正式產品的釘選都不因 Phase 3 而改變。
- 依 SPEC 仍不做 Mac 與 PC 的排名：Stage Q 的 Metal ÷ CUDA 比值是事先登記的平台一致性檢查；E3 兩個平台各自判定，不比較快慢。

#### Stage D：容量的延伸（decision 039 / [042](docs/decisions/042-stage-d-results.md)）

sessions：`formal-20261023-d-a1b`、`-d-a2`、`-d-b`、`-d-c`，四個都有效；判定全部依 039 凍結的規則（凍結 commit `fd95164`），沒有事後修改。
只依 S2 判定（S1、S3 只作描述性的界線）。RTX 5070 Ti、vLLM 0.30.0、W4A16；server 只有 Core B（B0）或 T（`--long-prefill-token-threshold 512`）。

| D-A（封閉迴路，准入之後；都是 EXACT） | C_KV | C_SLO | C_PROD | 先失守 | 旗標 |
|---|---|---|---|---|---|
| U32K | 3 | 2 | **2** | TPOT | — |
| U8K-T | 13 | 5 | **4** | 短類 TTFT | — |
| M8K-T（10% 32K） | 13 | 5 | **4** | TPOT | WINDOW_SENSITIVE |
| U2K-O512（輸出 512） | 30 | 32 | **32** | 短類 TTFT | WINDOW_SENSITIVE、ABOVE_CKV |
| U8K-O512（輸出 512） | 12 | 12 | **12** | 短類 TTFT | — |

| hypothesis（事先預測都是 supported） | 判定 |
|---|---|
| H-D1 C_SLO(U32K) ≤ C_KV = 3 | supported（2） |
| H-D2a C_PROD(M8K-T) > C_PROD(M8K) = 2 | supported（4） |
| H-D2b C_PROD(U8K-T) < C_PROD(U8K) = 8 | supported（4） |
| H-D3 輸出 512 時 C_SLO ≤ C_KV | U2K-O512 **not supported**（32 > 30）；U8K-O512 supported（12） |
| H-D4 開放迴路 λ_PROD < λ_ref | U2K-P **NT**；U8K-P supported（0.4 λ_ref EXACT = 0.2519 req/s） |
| H-D6 prompt 長度不一時 C_PROD 仍 PASS（直連） | U2K-var **not supported**（FAIL）；U8K-var supported |
| H-D7 經過 gateway 時 C_PROD 仍 PASS | U2K-gw、U8K-gw 都 supported |

- 圖例：NT = not testable；λ_ref = C_PROD ÷ c1b 在 C_PROD 的平均 e2e 延遲（Little's law 下平均並發等於 C_PROD 的到達率）；
  W′ = [2C, N − C)，比穩態時間窗 [C, N − C) 再去掉開場的第二波；WINDOW_SENSITIVE = 改用 W′ 時判定不同；
  ABOVE_CKV = C_PROD > C_KV（多出的 request 在 vLLM 內排隊，gateway 無法讓它逾時）。
- H-D6、H-D7 是逐點判定；若把 H-D6 當成「兩點都 PASS」的合取命題，結果是 not supported。
- **T 的取捨**（B0 的數字來自 c1b，跨 session）：U8K 的 C_PROD 8 → 4，M8K 2 → 4，所以 T 不是全面提高容量；T 只量了 8K、只有 512 一個值。
- **超過 C_KV**（U2K-O512）：准入只檢查 prompt 需要的 block、不預留輸出的 block（024 的原始碼唯讀查看加上本次計數），所以 C_SLO 可以超過 C_KV。
  代價（描述）：C 32 的每個 run 都有 8 / 256 個 request 停頓 4.24–5.22 s，P95 依規則仍通過；32 → 33 是斷崖（R_short 0.708 → 1.031），guard band 沒有預警。
- **D-B**（開放迴路，直連 vLLM、不靠准入控制，不是 gateway 的上限）：U8K-P 的 0.4 λ_ref 是格點最低點，真正的值在 [0.4, 0.55) λ_ref 之間，只差一格就是 INFEASIBLE。
  U2K-P 為 NT：一個 run 只完成 533 / 535 個 request 而無效（和 client 連線重用的競態一致，機制仍是推論）；依規則不補跑。
- **D-C**（固定點）：
  - U2K-var 各 run 的 R_short 為 0.944、0.856、0.989、1.021，只重播一條 pin 住的長度序列；去掉第二波（W′）後 P95 仍比 c1b 的 uniform 高 65–71%（跨 session）。
  - 經過 gateway：W′ 的短類 P95 比 c1b 直連多 38%（U2K）與 29%（U8K）（跨 session，無法拆成 gateway 排隊與上游）；uniform-2k 的最差 R_TPOT 0.923，離 guard band 只剩 0.027。
    兩點只對應 `capserve_sha256` `53488c6d…` 那一版，之後的 gateway 變更（含 043 的 keep-alive 修正）沒有量。
  - uniform-8k 的長度不一與經過 gateway 是分開驗證的，組合沒有量過。
- **D5 開場湧入**：用 c1b 既有數據的事後描述，不是事先登記的檢定（見 Stage C 的「開場湧入」）。
- 揭露：
  - 四個 session 連續占用 GPU lock 23.6 h，超過使用者「一次約 7–9 h、最壞約 12 h」的字面範圍（每個 session 單獨都在 12 h 之內）。
  - 跨 session 漂移約 1.5% 時，U8K-T 的 C_PROD 是 3–4，M8K-T 是 4–5。
  - 範圍只有一台機器、一個模型；輸出只有 128 與 512（512 只量了 uniform，O512 class 不能服務），129–511 與少於 128 的輸出對任何 class（含正式的 uniform-2k）都沒有量過；
    mixed 沒有開放迴路、長度不一與經過 gateway 的數字。

#### Stage E1-D / E1-P：8K 滑移的觸發（decision [045](docs/decisions/045-stage-e1d-results.md)、[046](docs/decisions/046-stage-e1p-results.md)；描述，不是檢定）

依 041 §1.1，E1 只做描述（沒有檢定，也沒有信賴區間），結果只寫進 decision 與 Future Work，不改任何判定。

- **E1-D**（045，`e1-20261025`，只在 Mac 上分析既有數據，沒有用到 GPU）：Advanced（B0、G、K3）、C0、Core B C 8 三個數據集的滑移鏈都 **holds**。
  - holds 是 041 事先定義的計數規則，不是統計結論；三個數據集中，計入且可評估的段裡有一環不成立的都是 0 段（容許 1 段）。
  - 只有 Advanced 量得到五環（L1–L5）；C0 與 Core 沒有 F、Q 與 step 紀錄，只量得到 L3、L4，L1、L2、L5 是 null（沒有評估），不是成立。
  - Advanced 中每組第一個 request 的 F 尖峰（F > 35 ms；F 是 server 端從 arrival 到 queued 的時間）和滑移一一對應（8 / 8 對 0 / 208），
    尖峰組在被排進 engine 之前多經過一個 decode-only step。
  - 限制：F > 35 ms 是看過數據後才定的，B0、G、K3 上的重現不是獨立驗證；若把 T、TG 的 20 次尖峰算進非滑移組，L1 誤報率 5.90%，超過 2%；
    樣本 21 段，一台機器、一個模型、一組 prompt，只有 8K uniform、C 8；三個數據集來自三個 session，不合併。
- **E1-P**（046，`formal-20261025-e1p`，觀測版 B0）：問 F 尖峰和 prompt 內容還是 bench 中的位置綁定。讀法為 **「無法判定（觸發太少）」**。
  - seed A（20261008）有 6 個尖峰，位置 88 在 4 / 12 個 bench 重現，6 段滑移的五環都成立；seed B（20261023）在 12 個有效 bench 中沒有任何尖峰。
  - 只有一個 session、兩個 seed；server 和 C0 不同，不和 C0 比較。依 041 不重跑、不延長，任何後續探針都要另寫 decision log。

#### Stage E2：2K 的 T 為什麼無效（decision [050](docs/decisions/050-stage-e2-results.md)）

session：`formal-20261031-e2`（48 / 48 次 boot 有效；凍結 commit `3eafb5f`，執行上沒有偏離）。全部 L 2048。
025 中 T512 在 2K、C 25 無效；三個候選解釋是 S（負載）、V（T 與 KV 受限的交互作用）、Th（threshold 太大）。
f 是 mixed 對短類的懲罰被移除的比例；m 是 mixed 短類 P95 本身的改善（對數）；T 有效（E）要求類別 M 或 P，而且 m ≥ 0.20。

| 格 | f | 類別 | T 有效（E）/ 無效（I） | preemption（mixed 3 次） |
|---|---|---|---|---|
| T@25（C 25，KV 4 GiB） | 0.079 | N | **I** | 266 |
| T256@25（threshold 256） | 0.133 | N | **I** | 371 |
| T@8（C 8，KV 4 GiB） | 1.070 | M | **E** | 0 |
| T@8k（C 8，KV 2.5 GiB） | 0.945 | M | **E** | 44 |

- 讀法 **R4**（041 §2.7），不是 incomplete：「C 8 重建的 KV 結構不足以讓 T 失效，C 25 的失效需要和 C 有關的因素。
  β_ok 成立：符合 S，但也可能是負載與 KV 的交互作用，分不開。」依 041 §2.11 與 §6 A4，**C 25 的成因沒有確立**。
- 前提都成立：三個 B0 參照都重現 2K 的懲罰（H-C0：+442.2%、+620.3%、+682.2%）；
  C 25 與 C 8 的負載差夠大（β_ok：mixed 期間取樣中有 request 在排隊的份額 0.915 對 0.333，差 0.582 ≥ 0.20）；
  KV 降到 2.5 GiB 確實造成 KV 受限（KV_ok：gate-zone，即有 request 在等、free block 放不下一個長類的份額，0.151 對 0，超出 noise）。
- hypotheses：H-E2-0（B0 重現懲罰）、H-E2-R（T@25 無效）、H-E2a（T@8 有效）supported，預測對；
  H-E2b（T@8k 無效，V 的預測）、H-E2c（T256@25 有效，Th 的預測）not supported，事先不預測；
  H-E2k（C 8 時 B0 的 KV 降到 2.5 GiB 讓 ρ 至少升高 20%）not supported，預測錯（a 0.042）；H-E2e（prefix caching 的能力檢查）supported（Y_on ÷ Y_off 0.244）。
  H-E2-0（只有 C 25 事先已知）、H-E2-R 與 H-E2e 列在 041 §5 第 4 項「已知或幾乎必然的結果」。
- 限定：
  - K2.5 重建的是 KV 結構，不是 C 25 的 KV 壓力（gate-zone 0.151 對 0.460、preemption 44 對 266）；「KV 結構不足以讓 T 失效」只對這個程度成立。
    描述上，K2.5 下 T 的改善較小（f 1.070 → 0.945；I_KV 0.169，只作描述），所以 R4 不能讀成「KV 對 T 沒有影響」。
  - 四格的型態恰好符合 S 的預測，但不能單獨選出 S：V 沒有被推翻，Th 沒有被排除（T256 依 041 的附註「沒有鑑別力」）。
  - C 25 的 U（uniform 的短類 P95）由開頭同時送出的 25 個 request 決定（和 024 / 025 相同的量法）；類別 N 依賴這個量法，
    但 T@25 與 T256@25 的 m 都低於 0.20，判定 I 不受影響。
  - T 在 C 8 有效但有代價（描述）：uniform 短類 P95 約兩倍、長類 P95 +11%、mixed throughput −4.7%。
  - H-E2e 只是能力檢查，不是 H-B4 的機制檢驗；0.244 主要來自開頭同時送出的 25 個 request，去掉這 25 個之後 P95 沒有縮短，不是穩態效果。
  - 只有一個 session，SE 只含同 session 內 3 次 restart 的雜訊；和 025 的並列是跨 session 的描述。allocator 仍未操弄、未檢驗。
  - 不改變 H-B4、025 的判定（2K 仍是 R6、T512 仍是 N）或任何產品設定。

#### Stage E3：H-A2 被推翻的原因（decision [047](docs/decisions/047-stage-e3-results.md)）

sessions：`formal-20261031-e3-metal`、`formal-20261031-e3-cuda`（各 26 / 26 ok，env 有效）。只量 L 2048，每格 3 次重複，每個平台一個 session；兩個平台各自判定，不合併，不比較快慢。

| | macos-metal | wsl2-cuda |
|---|---|---|
| 前提：H-A2 配對（9B-Q8_0 / 4B-Q4_K_M）的觀測比，bytes 比 2.9929 | 2.2027，成立 | 2.2256，成立 |
| H-E3a 非比例成分：a_f（Q8_0 / Q6_K / Q4_K_M，ms；事先預測 supported） | **supported**（4.701 / 4.878 / 5.528） | **supported**（1.731 / 1.428 / 2.117） |
| H-E3b 4-bit k-quant 的格式成本：D_4B、D_9B（門檻 κ；事先不預測） | **supported**（−0.9266、−0.9787，都超出 κ 0.7614） | **not supported**（−0.2517 超出、−0.1400 在 κ 0.2277 之內） |
| 讀法表（041 §3.5） | 第 3 列：非比例成分與 4-bit k-quant 的格式成本並存 | 第 6 列：格式效應不一致，不解讀 H-E3b |

- 定義（041 §3.3）：a_f 是同一格式跨 size 的截距（ms / token），> 0 代表 decode 時間中有一段和 bytes 不成比例；D_s = r(s, Q4_0) − r(s, Q4_K_M) 是對 6 個舊設定全域擬合的殘差差
  （D < 0 指相對擬合 Q4_0 比 Q4_K_M 快），負而且 |D_s| 超出 max(2·SE_D, κ) 才算格式成本；κ 是兩個 size 中 |r(s, Q6_K) − r(s, Q4_K_M)| 的最大值。
- 兩個平台落在不同的列，依 041 §3.5 的固定寫法是「原因依平台而異」；RTX 的格式效應沒有解讀，所以這句話不表示 RTX 的原因已知而且和 Mac 不同，也不表示 RTX 沒有格式成本。
- 「格式成本」是 Q4_K_M 對 Q4_0 的配方層級對比，不是 Q4_K block 格式本身的每 byte 成本。非比例成分是固定開銷還是頻寬效率，本設計分不開。
- RTX 的 D_4B 只以 0.024 ms 越過門檻，很脆弱。κ 是看過 Phase 1 數據後才定的；H-E3b 事先沒有預測，不算預測成功。
- SE 只反映同一 session 內 3 個 process 的雜訊；Mac 在 140W adapter（協商 94 W）下取得，和 Phase 1（70W）的數字只並列、不解讀。
- 偏離（已揭露）：Mac 正式 session 從固定在 `b9088a3` 的 git worktree 執行，偏離 041 §3.6；harness 與設定相同，對數據與讀法沒有影響。
- **H-A2 的判定不變（refuted）**；E3 沒有檢驗 GDN recurrent 狀態這個具體來源。

#### Stage Q：Metal 與長 context 的 quality（decision [048](docs/decisions/048-stage-q-results.md)）

sessions：`formal-20261031-q1-metal`（Metal `-c 512`；第 1 次嘗試 env 無效作廢，這是第 2 次）、`formal-20261031-q1-cuda`、`formal-20261031-q2`（CUDA `-c 2048` / `-c 8192`）；
CUDA 的 PPL 與 `-c 512` 的 KLD 來自 Phase 1 的 `formal-20261010-quality`。事先預測五個都是 supported；結果 2 個 supported、3 個 not supported。

| hypothesis | 判定 | 關鍵數字 |
|---|---|---|
| H-Q1a Metal 上 KLD(Q8_0) < KLD(Q6_K) < KLD(Q4_K_M) | supported | 4 個 z 檢定都通過 Holm |
| H-Q1b KLD_Metal ÷ KLD_CUDA 在 [0.8, 1.25]（前 200 chunks） | **not supported**（3 / 6 格超出） | 超出：4B Q8_0 0.3532、9B Q8_0 0.2699、9B Q4_K_M 0.6764；6 格都小於 1（0.270–0.908） |
| H-Q1c PPL_Metal ÷ PPL_CUDA 在 [0.995, 1.005] | **not supported**（1 / 8 個設定超出） | 9B Q4_K_M 1.0131；其他 7 個的差距在 −0.10% 到 +0.003% |
| H-Q2a CUDA 2K / 8K 上 KLD 的順序 | supported | 8 個 z 檢定都通過 Holm |
| H-Q2b KLD(c) ÷ KLD(512) 在 [0.5, 2.0] | **not supported**（4 / 12 格超出上界） | 4B Q8_0 c8192 2.1369、4B Q6_K c8192 2.0652、9B Q8_0 c2048 2.0926、9B Q8_0 c8192 2.4824 |

- H-Q1b、H-Q1c 的 not supported 只代表兩個平台的數值差異超過事先寫定的界線，不代表任何一個平台有缺陷，也不是品質優劣或平台排名。
  每個 KLD 都對自己平台的 BF16 算，沒有共同的參考；沒有做 backend 消融，不歸因到任何 kernel。
- 9B Q4_K_M 在 Q1b 是 Metal 的 KLD 較低，在 Q1c 卻是 Metal 的 PPL 較高；只並列，不解釋。
- H-Q2b 的 12 個比值都大於 1（1.23–2.48）是事後描述（040 沒有方向預測；c512 來自 Phase 1，跨 session），不能說 KLD 隨 context 增加是一般規律；界線不事後放寬。
- 限定：Q1 的 KLD 只有前 200 chunks，不能直接和 Core A 的 580-chunk KLD 比；Q1c 與 Q2b 是跨 session 的比較；長 context 只在 CUDA 量，c8192 只有 36 chunks；
  每部分一個 session；只有 WikiText-2，downstream 沒有重測。
- Phase 1 的 pilot 已預示 H-Q1b（9B）與 H-Q1c（9B Q4_K_M）不成立的方向，界線與預測沒有因此修改；Q1b 超出的格子和 pilot 不完全相同（4B Q8_0 新超出、9B Q6_K 回到界線內）。
- 偏離與違規（已揭露）：q1-metal 第 2 次嘗試從 worktree 執行（對數值與判定沒有影響）；q2 期間在 Mac 上以 ssh 唯讀列出 WSL 目錄一次（q2 只量 PPL 與 KLD、不量時間，env 擷取沒有 flag）。
- H-A5 與 Core A 的 quality 判定不變。

#### Stage Q3：Core B 的 W4A16 checkpoint 的 quality（decision [049](docs/decisions/049-stage-q3-results.md)）

session：`formal-20261031-q3`（2 / 2 ok；vLLM 0.30.0，RTX 5070 Ti，WSL2）。WikiText-2 test 的 580 段 512-token chunk，每段計分後 255 個 token。

- **H-Q3 not supported**：PPL W4A16 8.5810 對 BF16（CPU 卸載 8 GiB）8.1865，**+4.82%**，超過事先寫定的 2% 界線。
  040 預測 supported，預測錯了；凍結前的第二次 pilot（10 段，+4.24%）已預示這個方向，門檻與判定規則沒有因此修改。
- 判定依點估計的 PPL 比對照實質門檻，不是統計檢定。附加資訊：逐段配對 ΔNLL 0.04706 nat / token，95% CI [0.04469, 0.04951]
  （只含 chunk 間的變異，相鄰 chunk 是連續文字，可能偏窄）；top-1 一致率 0.9029（兩個 checkpoint 的 rank-1 token 是否相同，不是準確率）。
- 只描述這一組 checkpoint：`RedHatAI/Qwen3.5-9B-quantized.w4a16@a398088c` 對 `Qwen/Qwen3.5-9B@c2022362`。
  W4A16 的 base revision 沒有核對，所以 +4.82% 是兩個 checkpoint 的差，不只是量化的效果。每個 checkpoint 只跑一次；BF16 的 CPU 卸載是否改變數值沒有測試。
- 和 GGUF 的 PPL 並列只作描述（不同 runtime 與量化方式），不比優劣；不推論原因，也不說 W4A16「不可用」或「有缺陷」，不推廣成 9B 的 4-bit 量化一般都超過 2%。
- Core A、Core B、H-A5 與 Q1 / Q2 的判定都不變（Core B 用的就是這個 checkpoint，Q3 只是它的 quality 背景）。

#### 產品：capserve v1 與 Stage D 的產品化（decision [038](docs/decisions/038-capserve-v1-design.md)、[043](docs/decisions/043-stage-d-productization.md)）

- **v1（038）**：thinking 預設關閉（client 可開啟）、IT-3 以 token 數判定、adaptive Retry-After、gateway 內建 tokenizer、per-key quota、TLS、常駐功能。
  - 七項都只改 gateway 的行為，不改 vLLM 的啟動參數與送進 vLLM 的負載上限，所以送進 vLLM 的負載仍在 verified 範圍內
    （容量本身是直連 vLLM 量的；經過 gateway 的 client 端延遲只有 D-C 對舊版 gateway `53488c6d…` 的兩點）；經過 gateway 的 chat 流量沒有量過。
  - TLS 只在自動化測試中以測試產生的自簽憑證驗證（038 補記 6），沒有在 RTX 上測；正式設定只綁 loopback。常駐只做過時間有界的 RTX 測試，研究期間不常駐（037）。
- **Stage D 的產品化（043，實作完成）**：產生器寫出三份 verified profile，都**只 commit、不釘選**，不是正式設定：
  - `verified-formal-20261023-d-a1b-b0`：uniform-32k（2）；
  - `verified-formal-20261023-d-a1b-t`：uniform-8k-t（4，只作報告）、mixed-8k-t（4）；
  - `verified-formal-20261023-d-a2-b0`：uniform-2k-o512（32，ABOVE_CKV）、uniform-8k-o512（12）。
  - verified 不等於可服務：uniform-8k-t 與兩個 O512 class 不能作為 active class；mixed-8k-t 的 profile 允許，但 deploy 不啟動 T（拒絕規則見上方「正式 profile 的保護」）。
  - keep-alive 修正：client → gateway 這段的閒置上限改為 75 s（原本是 uvicorn 預設的 5 s）。只保護閒置上限低於 75 s 的 client，
    閒置更久或不設上限的 client 仍有同類競態；它改變了 `capserve_sha256`，所以 D-C 的 H-D7 只對應舊版。
- **正式設定不變**：active class 維持 c1b 的 uniform-2k（32），並揭露 D-C 長度不一的 FAIL（見 Stage D 段的 D-C）。
  uniform-32k 與 mixed-8k-t 已通過 V-GW（051），成為正式釘選的候選；是否改釘選由使用者決定，改了之後跑 IT-0–6；T 現在不部署。
- **仍然不可用**（043 §7.5，Stage V 之後）：多 class 准入與 token 加權准入（`weighted-v1` 不成立）；2K 的 prefill pacing（被否決）；
  兩個 O512 class 的服務（由 V1 的 `uniform-*-o1-512` 取代，見 Stage V 段）；T 的部署與 uniform-8k-t 的服務；
  經過 gateway 的開放迴路宣稱；uniform-2k 在 prompt 長度不一時的 SLO 宣稱。PACE-8K 有 verified profile，但預設關閉、不釘選。
- IT 的結果見上方「產品整合測試」。

#### Stage V：Stage D 產品化的驗證研究（decision [044](docs/decisions/044-stage-v-preregistration.md)）

- 範圍（044）：V-GW（Stage D 的 class 經過 capserve gateway，在 C_PROD 是否仍成立）、V1（輸出長度在 [1, 512] 均勻、事先 pin 住時 SLO 是否成立）、
  V2（開場湧入的 prefill pacing，開 / 關）、V4（`weighted-v1` 多 class 准入，pacing 關閉）；V3（經過 gateway 的開放迴路）依使用者的決定延後。
- 事先登記已於 2026-10-31 凍結（mini `smoke-v-20261031` 通過，延遲已遮蔽）。結果見 decision [051](docs/decisions/051-stage-v-results.md)。

sessions：`formal-20261031-v-adm`（V2、V4；25 / 25 有效，1.77 h）、`formal-20261031-v-cls`（V-GW、V1；32 / 32 有效，5.30 h）；部署凍結 commit `c7ad0b4`，GPU lock 下執行，
之間空檔 2.03 h；都是第一次、也是唯一一次執行。判定 view：V-GW、V1 用 client，V2 取 upstream 與 client 較差者，V4 用 upstream；判定只看 W，guard band 0.95。

| 點 | 判定 | 最差 R（判定 view） | 預測 |
|---|---|---|---|
| GW-U32K（uniform-32k @ 2） | PASS | 0.625（TPOT） | PASS |
| GW-U2K-O512（輸出 512 @ 32） | PASS | 0.702（TTFT；ABOVE_CKV，每個 run 8 / 256 個 4–5 s 停頓） | PASS |
| GW-U8K-O512（輸出 512 @ 12） | PASS | 0.535（TTFT） | PASS |
| GW-M8K-T（mixed-8k-t @ 4，T） | PASS | 0.796（TTFT） | **MARGINAL 或 FAIL（沒有如預測）** |
| L-U2K（輸出 [1, 512] @ 32） | PASS | 0.697（TPOT） | PASS |
| L-U8K（輸出 [1, 512] @ 8） | PASS | 0.731（TPOT） | PASS |
| P-U2K / P-U8K（pacing 開） | PASS / PASS | 0.922 / 0.883（TPOT） | PASS / PASS |
| P-U2K-var（pacing 開） | FAIL（1 個 run） | 1.019（client TTFT） | FAIL |
| F-U2K-var / F-U2K（對照，pacing 關） | PASS / PASS | 0.920 / 0.922 | **FAIL（沒有如預測）** / PASS |
| W-90 / W-50（`weighted-v1`） | FAIL / FAIL（各 1 個 run） | 1.384 / 1.667（2K TTFT） | FAIL / FAIL |

- **成立的單位**：VG-32K、VG-M8K-T（只成為正式釘選的候選，沒有新檔）；V1-2K、V1-8K（`verified-formal-20261031-v-cls-b0`：`uniform-2k-o1-512` 32、`uniform-8k-o1-512` 8，bound `validated`）；
  PACE-8K（`verified-formal-20261031-v-adm-b0`：c1b uniform-8k（8）加上 pacing {32,256, 5,800} 與佇列 120 s / 600 s）。兩份 profile 都只 commit、不釘選。
- **不成立**：PACE-2K（P-U2K PASS，但 var 那一對讀法為「pacing 在 var 上有害」，否決）；W1（W-90、W-50 的 2K TTFT FAIL，U_W ≥ 28.8，不是 trivial）。
- 12 個 hypothesis 中 11 個如預測；H-GW4 與不是 hypothesis 的對照組 F-U2K-var 沒有如預測，PACE-2K 的否決正是由後者觸發。
- 讀法的限定：
  - PACE-8K 只表示 pacing 開啟時穩態的 S2 成立（W 中 pacing 沒有約束任何 request），不是改善；V-ADM 沒有 8K 不開 pacing 的對照。
    第一波沒有改善：2K 和同 session 的 F-U2K 相同；8K 只能和 D-C 跨 session 比較（相差 ≤ 10%，只作描述）。
  - PACE-2K 的否決很脆弱：P-U2K-var 只有 1 個 run、只在 client view 超過（R 1.019），F-U2K-var 的 r3 在判定邊界上；規則是機械式的，不重跑、不改判。
  - GW-U2K-O512 的 PASS 是「個數斷崖」型：每個 run 8 個停頓，到 13 個 P95 就 FAIL；V 只量了 C 32。
  - GW-U2K-O512 與 L-U2K 的 full 視窗為 FAIL（開場第一波），只作描述；不對第一波宣稱 SLO。
  - V1 只涵蓋 5 條 [1, 512] 均勻序列；極短輸出的 TPOT（L-U2K 1–16 箱 P95 303 ms）不決定合併的 P95，不代表每一種輸出長度都符合。
  - 在 C_PROD 時 gateway 不排隊（queue_wait 0，依構造）；同一個 request 的 client − upstream TTFT 約 7 ms（2K）到 72 ms（32K）。
    和 042 直連的跨 session 差各點方向不一（−415 ms 到 +2.26 s），只作描述，不能歸因。
- 範圍：一台機器、一個模型、一版凍結的 gateway（`capserve serve`，不是 `deploy up`）、封閉迴路、`/v1/completions` 加 `ignore_eos`；
  沒有量 prompt 長度不一、開放迴路、32K 或 T 在政策下、pacing 與 weighted-v1 同時開啟、快速 429。

### 證據不支持的部分

- Metal 上的 quality、長 context（> 512）的 quality、Core B（W4A16）的 quality：Phase 1、Phase 2 都沒有量；Phase 3 的 Stage Q / Q3 量了一部分（048、049）。仍不支持的是：
  - Metal 與 CUDA 數值差異的原因（沒有 backend 消融）；2K / 8K 的 KLD 比 512 大（事後描述、跨 session、每個 context 一個 session）是否為一般規律，以及原因；
  - Metal 上的長 context 與 32K 的 quality；新平台或新 context 下的 downstream；session 之間的變異；
  - W4A16 的長 context 與 downstream quality、其他 4-bit checkpoint，以及 +4.82% 的原因（Q3 只有 vLLM 0.30.0、WikiText-2 512-token chunk、這一組 checkpoint、各一次 run）。
- Mac 的多使用者 serving：沒有量（SPEC：Core B 只在 NVIDIA）。
- 跨 size 的 quality 只有 H-A5(a) 這一組配對；其他 4B 對 9B 的 quality 關係都是 unmeasured。
- H-A2 被推翻的原因：Core 沒有檢驗；Phase 3 Stage E3（047）只量 L 2048、每個平台一個 session。仍不支持的是：
  非比例成分是固定開銷還是頻寬效率；RTX 有沒有格式成本；格式成本解釋了偏差的多少；推廣到其他 k-quant、量化配方、模型、context 或 build。
- H-B4 的機制：只在 8K 隔離出主因（Advanced R1）。2K 落在 R6：T、G、TG 都沒有移除過半，KV 可能有關，但未確立（未達 R5）。allocator 沒有操弄。
  - Phase 3 Stage E2（050）：T512 在 2K、C 25 仍無效，在 C 8（KV 4 GiB 與 2.5 GiB）有效；讀法 R4，C 25 的成因仍沒有確立
    （符合 S，但和負載 × KV 的交互作用分不開）；allocator 仍未操弄、未檢驗。
- Advanced 只測了一個 chunk threshold（512）與一個 KV 劑量（3 GiB）；其他值、其他模型與 runtime 版本的效果未知。
  - Phase 3 另外只測了 threshold 256（只在 2K、C 25，沒有鑑別力）與 KV 2.5 GiB（只在 2K、C 8；H-E2k not supported，a 0.042 < 0.20），
    以及 Stage D 中 T = 512 在 8K 的容量（042）。其他 threshold、KV 劑量、C、L、模型與 runtime 版本仍未知。
- 8K 滑移鏈（E1-D、E1-P）：只做描述，鏈的因果沒有檢驗；F 多約 15 ms 的根本原因沒有量（需要 profiling）；觸發和 prompt 內容還是 bench 位置綁定沒有回答，
  seed 之間尖峰數 6 對 0 的原因也未確立；C0 與 Core 的 L1、L2、L5 沒有量；T、TG 的尖峰是否造成滑移沒有描述。
- 容量（Stage D 之後）：mixed 的開放迴路與長度不一都沒有量（經過 gateway 只有 GW-M8K-T 在 C_PROD，051）；uniform-2k 的開放迴路為 NT；「gateway 的並發上限加上 Poisson 到達」從未量過；
  H-D7 的理由（gateway 在 C_PROD 不排隊）在 Stage V 量到 queue_wait 0，但那是依構造（client 數等於上限），負載下的排隊沒有量；D-B 無效 run 的 keep-alive 機制仍是推論；
  長度不一只重播了一條序列；新的 Stage D class 只有經過 gateway（C_PROD）與輸出長度不一（V1）的證據（051），沒有 prompt 長度不一與開放迴路的證據。
- 產品（043 §7.5、051）：`weighted-v1`（043 的參數）在 0.9 / 0.5 兩個 mix 下 2K 的 TTFT 不符合 S2（各 1 個 run），其他參數、mix 與政策都沒有量；
  prefill pacing 只有 8K 成立（預設關閉、不釘選），2K 被否決，第一波沒有改善；pacing 與 weighted-v1 同時開啟、32K 或 T 在政策下都沒有量；
  keep-alive 修正與之後的 gateway 變更對延遲的影響沒有量過；TLS 沒有在 RTX 上測。

## Future Work（decision 026；Phase 2 的處理見 027）

以下都沒有在 Phase 1 中回答。Phase 2（027）只處理其中的 fixed-SLO 最大並發，以及 8K 雙模式的有限度排查；其餘仍要使用者另外決定。
Phase 2 結案後的後續項目（產品化 v1、開放迴路容量、其他輸出長度、開場湧入、O-32K / O-T 等）見 decision 036。
使用者 2026-10-22 決定在 Phase 3 執行這些後續與下列各項，SPEC 的 Optional 除外（decision 037）。
Phase 3 到 2026-10-31 的處理：產品化 v1 由 038 完成；O-32K、O-T、輸出 512 與直連的開放迴路由 Stage D 量測（042；uniform-2k 的開放迴路為 NT），
開場湧入由 042 的 D5 以 c1b 的既有數據做事後描述。043 把 D-A 的結果寫成三份只 commit、不釘選的 verified profile 與拒絕規則，D-B、D-C 只寫進適用條件；
O512 與 T 的服務、prefill pacing、經過 gateway 的開放迴路宣稱仍然不可用（043 §7.5），產品也不對第一波宣稱 SLO。
Stage V（051）之後：V1 的兩個 `validated` class 與 PACE-8K 有 verified profile（不釘選）；`weighted-v1` 與 PACE-2K 不成立；T 的服務與經過 gateway 的開放迴路宣稱仍不可用。
下列各項的 Phase 3 結果標在各項之下；仍開放的部分照實列出。

- **8K uniform 的雙模式**：短類 P95 在 boot 之間落在約 2.48 秒或約 2.82 秒，成因未知（025「限制」）。
  它使 8K 的 noise 偏大；事後把模式拉齊重算，8K 的類別與判讀列都不變，也不影響既有的部署結論，所以目前不追查。
  - Phase 2 的有限度排查（C0，decision 028 / 032）：機制是一組 request 錯過一個 2048-token prefill step（約 349 ms）。
    6 次 boot 中沒有偵測到 boot 層級的差異（Rrun，檢定力有限），Stage C 採預設協定。
  - C0 中觸發的時點仍未知，Phase 2 結案時決定不再追查：5 段滑移中有 4 段從同一個 request（idx 89）開始；每次 boot 的第 1 個 bench 都沒有滑移；
    偏高比例 21%，低於 Phase 1 的 67%。
  - Phase 3 的 Stage E1 接著追查（描述，不是檢定，不改任何判定；結果見「Phase 3 結果」）：E1-D（045）的滑移鏈依事先定義的計數規則在三個數據集都 holds
    （Advanced 量得到五環，C0 與 Core 只量得到 L3、L4）；E1-P（046）無法判定（觸發太少）。
    - 仍未回答：鏈的因果、F 多約 15 ms 的根本原因（需要 profiling）、觸發和內容還是位置綁定、seed 之間尖峰數 6 對 0 的原因、
      032 的偏高比例和 Phase 1 不同（需要多個 session）。
- **2K 的 H-B4 成因**：落在 R6（KV 可能有關，但未確立），分解與操弄不一致。
  - Phase 3 Stage E2（050）：讀法 R4，依 041 以條件式結論結束，C 25 的成因沒有確立；「高負載、KV 不受限」的格子做不出來，仍是未解的部分。
- **其他 Advanced 項目**：fixed-SLO 最大並發（已由 Phase 2 的 Stage C 回答，見 035）、其他 chunk threshold 或 KV 劑量、prefix caching、allocator 的操弄。
  - Phase 3：E2 量了 threshold 256（只在 2K、C 25；也無效，但依 041 的附註沒有鑑別力，不能據此判斷 threshold 大小）、KV 2.5 GiB（只在 2K、C 8；H-E2k not supported），
    以及 prefix caching 的能力檢查（H-E2e supported；只是能力檢查，不是 H-B4 的機制檢驗；限這個模型、vLLM 0.30.0 與開頭同時送出的量法，不是穩態效果）；
    T = 512 在 8K 的容量取捨由 042 量到。其他 chunk threshold 仍未回答；allocator 的操弄仍沒有做。
- **H-A2 被推翻的原因**：Core 沒有檢驗。
  - Phase 3 Stage E3（047）：Mac 與 RTX 各自重現跨 size 的非比例成分（H-E3a）；Mac 另有 Q4_K_M 對 Q4_0 的配方層級格式成本（H-E3b），RTX 的格式效應不一致、不解讀。
    仍開放：非比例成分是固定開銷還是頻寬效率；RTX 有沒有格式成本；格式成本解釋了偏差的多少；
    8K / 32K、imatrix、`--pure`、其他模型與 build 都沒有量；SE 只含同 session 的雜訊，跨 session 的再現性未知。
- **quality 的缺口**：Metal 上的 quality、長 context（> 512）的 quality、Core B（W4A16）的 quality。
  - Phase 3：Stage Q（048）量了 Metal（`-c 512`）與 CUDA 的 2K / 8K，Q3（049）量了 W4A16，都只有 WikiText-2 的 PPL / KLD、每部分一個 session
    （Q3 只有這一組 checkpoint、各一次 run）；判定見「Phase 3 結果」。
    留下的問題：兩平台數值差異的成因、9B Q4_K_M 的 KLD / PPL 對比、2K / 8K 的 KLD 比 512 大是否為一般規律及其原因、downstream 重測；
    W4A16 的長 context / downstream、其他 checkpoint，以及超出 2% 的原因。
- **產品與容量的後續**（043、044）：
  - V-GW、V1、V2、V4（Stage V，044）：使用者 2026-10-24 決定都做；已完成，結果見 051 與「Phase 3 結果」的 Stage V 段。
    是否改正式釘選（uniform-32k、mixed-8k-t、`uniform-*-o1-512`）、是否開啟 PACE-8K（等於改釘選）由使用者決定；改了之後跑 IT-0–6。
  - V3（經過 gateway 的開放迴路）延後：要先另寫 decision 修正 D-B 的 client 連線重用缺陷，不能補跑 D-B。
  - T 的部署與 V5 要使用者決定；較長輸出：V1 的兩個 class（輸出 ≤ 512，5 條均勻序列）有 verified profile，V1-8K 取代 043 的包絡規則，超過 512 的輸出沒有量；mixed 的各項延伸都沒有量。
  - 常駐要等研究的 GPU 工作結束，並取得使用者同意。
