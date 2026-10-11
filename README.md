# Local LLM Serving Lab

模型大小、量化、context、並發如何改變**品質、延遲、吞吐量與記憶體**的 frontier？什麼條件下該用什麼配置？

這是一個事先登記（pre-registered）的實驗：hypothesis、門檻、metric 與 protocol 都在看到正式數據之前寫定並凍結，
之後的任何變更都有 decision log。結論只來自有效的正式 session，失敗、OOM、not testable 的單位照實報告。

> **狀態（2026-10-11）：Project 暫時完成**（使用者決定，decision 026）。目前的 defensible milestone：
> - **Core**：A（macos-metal、wsl2-cuda）、B（uniform、mixed）、quality、deployment decision table。
> - **已完成的 Advanced**：隔離 H-B4 的成因（decision 023 的範圍；事先登記與凍結見 024，結果見 025）。
>
> 不新增 Advanced 項目；尚未回答的問題列在下方的 Future Work。

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
- [docs/decisions/](docs/decisions/)：每個設計決策的 decision log（001–026）。
  - 021 是 deployment decision table 的推導規則，在其餘正式分析之前寫定。
  - 023 是 Gate 的決定，024 是 Advanced 的事先登記（凍結 commit `97a948c`），025 是 Advanced 的結果，026 是收尾的決定。
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
tools/        with-mac-lock（Mac 重負載 lock）
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

session：`formal-20261010-quality`（46 / 46 ok，env 有效）。quality 視為 model × quant 的屬性，兩個平台共用（Metal 上沒有量）。

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

### 證據不支持的部分

- Metal 上的 quality、長 context（> 512）的 quality、Core B（W4A16）的 quality：都沒有量。
- Mac 的多使用者 serving：沒有量（SPEC：Core B 只在 NVIDIA）。
- 跨 size 的 quality 只有 H-A5(a) 這一組配對；其他 4B 對 9B 的 quality 關係都是 unmeasured。
- H-A2 被推翻的原因：沒有檢驗。
- H-B4 的機制：只在 8K 隔離出主因（Advanced R1）。2K 落在 R6：T、G、TG 都沒有移除過半，KV 可能有關，但未確立（未達 R5）。allocator 沒有操弄。
- Advanced 只測了一個 chunk threshold（512）與一個 KV 劑量（3 GiB）；其他值、其他模型與 runtime 版本的效果未知。

## Future Work（目前不做；decision 026）

以下都沒有在本 milestone 中回答。若要繼續，要由使用者重新決定，並且先事先登記、凍結，再收集數據。

- **8K uniform 的雙模式**：短類 P95 在 boot 之間落在約 2.48 秒或約 2.82 秒，成因未知（025「限制」）。
  它使 8K 的 noise 偏大；事後把模式拉齊重算，8K 的類別與判讀列都不變，也不影響既有的部署結論，所以目前不追查。
- **2K 的 H-B4 成因**：落在 R6（KV 可能有關，但未確立），分解與操弄不一致。
- **其他 Advanced 項目**：fixed-SLO 最大並發、其他 chunk threshold 或 KV 劑量、prefix caching、allocator 的操弄。
- **H-A2 被推翻的原因**：Core 沒有檢驗。
- **quality 的缺口**：Metal 上的 quality、長 context（> 512）的 quality、Core B（W4A16）的 quality。
