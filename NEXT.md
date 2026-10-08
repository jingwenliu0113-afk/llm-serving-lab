# NEXT

## 目前位置

**Core — SPEC Day 1 已完成（NVIDIA 環境 smoke test 通過）；Week 1 進行中。**

- 模型已依使用者指示重新選擇（**decision 008**，使用者已確認）：Core A 為 Qwen3.5 4B + 9B，Core B 為 RedHatAI Qwen3.5-9B W4A16。
- 凍結前的一致性修訂（**decision 009**）已完成；hypothesis 的結構與門檻不變。
- Week 1 中 Mac 可做、不需大型下載的部分已完成：harness、輸出格式、quality 子集工具。
- Advanced / Optional 皆未觸碰。

## 上次完成（2026-10-08）

### Commit 狀態

- 所有已驗證的工作都已 commit 並 push 到 private `origin/main`（最新：decision 013 的三個 commit）。
- 目前 `uv run pytest -q`：**365 passed、2 skipped**（skip 的是兩個 freeze 階段的檢查，屬預期）。

### 模型重新選擇（decision 008）

- Qwen3.6 / 3.8 延續 Qwen3.5 的 architectural foundation（hybrid：3 層 Gated-DeltaNet + 1 層 full attention，VLM 封裝）。
  但 3.6 / 3.8 只有 27B dense，沒有適合本實驗 4B / 9B 與 RTX 16 GB 條件的組合，因此 Core 選 **Qwen3.5** 4B + 9B。
- 兩個候選：
  - **候選 1：4B + 9B（選定）**。兩台機器 18 格全部 fit。
  - **候選 2：9B + 27B（不選）**。27B 在 RTX 上每個 quant 都 OOM，Core B 也不可行。
- Core B：`RedHatAI/Qwen3.5-9B-quantized.w4a16`（GPTQ W4A16 g128），C_KV(2K / 8K / 32K) 約 19–28 / 7–10 / 2（009 更正）。
  沒有官方 9B 以下的 quant。
- 主要取捨：
  - 現有 context 下 RTX 沒有 fit 邊界，改以 headroom 報告；不延伸到 256K，因為 Mac 端成本不可行。
  - KV 每 token 只有 32 KiB，每條序列另有約 50 MiB 的 recurrent 狀態。
  - vLLM 的 GDN kernel 需要 JIT，WSL2 + sm_120 上的可用性還沒驗證。
- 不需要更換 runtime pin（llama.cpp b11146、vLLM 0.30.0 都已支援 `Qwen3_5ForConditionalGeneration`）。
- 已處理：
  - `pins.toml` 的 Core A / B 已換成新模型（revision 與 shard sha256 由 HF API 核對）；`test_pins` 強制 Core B 沒有 fallback，而且 base 必須是 Core A 的其中一個 size。
  - 003 / 004 / 005 / 007 / PREREGISTRATION 開頭都加了「已由 008 取代 / 需重新推導」的說明，原內容保留作為歷史。
  - MACHINE_POLICY 的 Core B 模型行已更新。

### Week 1（Mac，不需下載）

- **Benchmark harness**（先寫測試）：
  - `benchmarks/core_a_plan.py`：依 pins 產生 job；sentinel 由 pins 推導。
  - `benchmarks/core_a_parse.py`：OOM 字串取自 pinned source。
  - `benchmarks/run_core_a.py`：一個 job 一個 process；每個 process 前後擷取環境；sentinel drift 檢查；檢查 GPU / Mac lock；開跑前驗證 build。
  - parse 與 runner 只用 stdlib，相容 Python 3.10，遠端 WSL 也能直接跑。
- **輸出格式**：`docs/RESULTS_FORMAT.md`（Week 1 #3）。
- **Quality 子集**：
  - `quality/mc_subset.py`。
  - HellaSwag / MMLU 的 4000 題清單（seed 20261008，與模型無關）。這些清單與研究階段的獨立實作 byte 相同。
  - HellaSwag 子集的 sha256 已寫入 pins。
- **MACHINE_POLICY**：
  - Mac Hard Lock：已實作 `tools/with-mac-lock`。
  - Docker：Core 階段不使用 Mac Docker Desktop，不得在 Mac 啟動 model serving container。
  - 遠端環境資訊。
- **Day 1 NVIDIA 環境 smoke test**：PyTorch 2.14.1+cu130、sm_120、bf16 matmul 都正常（001）。

## 進行中（2026-10-08 晚）

### GitHub

- `origin` = `https://github.com/jingwenliu0113-afk/llm-serving-lab.git`（**Private**），`main` 已設 upstream。
- 每個 coherent、測試通過的 commit 自動 push；禁止 force push、改寫 history。
- 不進 git：權重 / HF cache、GGUF、KLD、`results/**/raw/`（以 `raw_manifest.json` 的 sha256 驗證）、venv、credential（`.gitignore` 與測試）。

### 本輪重點（2026-10-08 晚，皆已 push）

- **4B 全流程完成**：GGUF ×4 已產出並 pin → tokenizer 交叉驗證 → `delete-source`（`results/artifacts/deletions.jsonl`）
  → Mac smoke 正常 → Core A pilot（22 / 22 ok）→ quality pilot（26 / 26 ok）。
- **tokenizer 交叉驗證**（`quality/subsets/tokenizer-crosscheck.json`）：llama.cpp 與 HF 完全一致。
  WikiText 580 chunks；MMLU 排除清單相同（3 題），HellaSwag 與 ARC 也走同一 scorer，皆 0 題。
- **Core A pilot（decision 012）**：
  - Mac P = 0。
  - sentinel CV 5.9%，兩格 MDE 約 9.6%。
  - graph splits = 2，GDN 沒有退回 CPU。
  - 記憶體三種量測一致：breakdown 得到驗證，footprint 不含權重。
  - fit run 加 `-lv 4`。
  - 新增電源有效性規則。
  - 第一張圖：`results/core_a/figures/memory-vs-context-macos-4b-pilot.svg`。
- **quality pilot**：determinism ✓；self-KLD 0；Q8_0 KLD 0.0014（fa on / off 相同）；downstream CI 半寬 0.2–1.7 pp；
  4B 正式約 5.1 h。
- **分析程式先於數據寫定**（decision 011）：Core A H-A1–H-A4、quality H-A5、Core B H-B1–H-B4、Core B grid / runner / mixed dataset。
- 其他：
  - raw outputs 不進 git，改以 `raw_manifest.json` 驗證。
  - `produce_gguf` 可以只從 BF16 重新量化。
  - GGUF header reader。
  - Core A report CLI。
- **凍結前第二輪 review（decision 013）**，皆在任何數據前修正。vLLM 的兩項已對照 v0.30.0 原始碼確認：
  - Core B client：
    - 補 `--temperature 0`（0.30.0 不再預設）。
    - `mixed` 補 `--custom-output-len 128`（預設 256 會覆蓋每列）。
    - `--tokenizer` 改用 pinned snapshot，bench 加 `HF_HUB_OFFLINE=1`。
    - `mixed` dataset 逐列檢查長度。
  - Core B 分析：
    - 只計入 `ok` 的 run。
    - 任一 grid C 缺 3 次有效 restart，該 L 記為 not testable。
    - plan 的 `str(L)` key 可直接使用。
  - Core A：
    - prefill 改為 process-level（先算 L ÷ TTFT 再平均）。
    - H-A3 的「相鄰」以 32K fit 的配置為準。
    - 缺 metric 時記為 not testable，不再 crash。
  - Mac env 紀錄若電源或 thermal 狀態未知，視為 env-invalid。
  - 分析 CLI 只處理有效的正式 session（pilot / env-invalid / drift 一律拒絕）。
  - 照實記錄：4B pilot 依 quant 列出的 KLD SE 使 H-A5(b) 的順序在凍結前可推得（規則未改）。
    之後 pilot 只報每個 size 的最大 SE。
  - 更正 011 的描述：`mixed` 短類與 `uniform` 分布相同，token 內容不保證相同。
- 4B `-fa auto` probe 以 `-lv 4` 重跑：三個 quant 都是 "Flash Attention enabled"、graph splits = 2（`pilot-20261008-4b-fa`）。

### 下載與 artifact

| 項目 | 位置 | 狀態 |
|---|---|---|
| 4B GGUF ×4 | Mac `models/gguf/` | ✅ pin + recipe；safetensors 已依規則刪除 |
| 9B safetensors（19.33 GB） | Mac `models/hf/Qwen__Qwen3.5-9B@c2022362/` | ⏳ shard 1–3 已驗證 sha256；shard 4（約 3.35 GB）下載中，22:07 時約 2.03 GB。網路不穩（curl 自動續傳 / 重試），完成後 log 會出現 `ALL_SHARDS_VERIFIED`（scratchpad `fetch_9b.log`） |
| Core B checkpoint | WSL HF cache | ✅ sha256 已驗證 |

## 下一步（依序；Mac 重負載都經 `tools/with-mac-lock`）

1. ~~4B 的 `-fa auto` probe 以 `-lv 4` 重跑~~ → ✅ 已完成並 push。
2. 9B 下載完成後，依序執行：
   - scratchpad 的 `batch_9b.sh`：`produce --size 9B`。**已排隊**（`queue_9b.sh`）：等 `ALL_SHARDS_VERIFIED` 之後，每 2 分鐘試一次
     `tools/with-mac-lock`；lock 被占用就放棄該次（22:07 時由 incident-investigator 的 `env soak` 持有），不等待、不碰 lock。
   - commit pins 與 recipe。
   - `batch_9b_pilot.sh`：
     - delete-source；
     - 9B smoke；
     - 9B Core A pilot（含 fa probe / memory / calib）；
     - 兩個 size 的 quality pilot，用來估 H-A5 ΔNLL 的 SD 與總時數。
3. 補完 PREREGISTRATION §5：
   - 8 個 GGUF hash；
   - `-fa auto` probe；
   - throughput 校準（9B）；
   - H-A5 MDE；
   - harness commit。
   完成後：
   - `freeze.mac = true`；
   - PREREGISTRATION 第 1–5 節標為 FROZEN (Stage A)；
   - commit。
4. 正式 run 前，**向使用者報告並建議**（不是 blocker，但會影響數據品質與筆電使用）：
   - 正式 Core A（Mac）約 ≤ 11 h（9B 後更新），quality 兩個 size 合計約 10–15 h，期間持有 Mac heavy lock。
   - pilot 期間前景程式（ChatGPT / Codex、Chrome 影片、WindowServer）讓 loadavg 達 6–9，是 noise 的主因。
   - 70 W adapter 滿載時電池仍會下降（28 分鐘內 80% → 76%）；電量 < 20% 會使 session 作廢。建議用 ≥ 96 W adapter。
   - 磁碟：正式 quality 的 KLD（68.4 GiB / size）會比 50 GiB 餘量少約 15 GiB，依 STORAGE_PLAN 暫時刪另一個 size 的
     quant，之後從 BF16 重建。
5. Core B：使用者裝好 `build-essential` 後：
   - 用 `scripts/deploy_bundle.py` 部署；
   - 在新的 session 重跑 S1 → S2 → S3；
   - 通過後進入 Stage B（量 B / C_ctx、超量測試、vLLM lock、`mixed` dataset sha256）。

## Blocked

- ~~`ssh gpu-server`~~ → **resolved**（2026-10-08）。
- **遠端 C compiler / CUDA toolkit（需使用者以 sudo 安裝）**：2026-10-08 22:05 唯讀檢查，gcc / cc / g++ / make / nvcc 仍不存在。
  - Core B 的 S1 smoke 已確認卡在缺 C compiler。
  - llama.cpp CUDA build 需要 `cuda-toolkit-12-8`。

  ```bash
  sudo apt-get update && sudo apt-get install -y build-essential
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8     # 002：只裝 toolkit，不要裝 Linux driver
  ```

## 待使用者處理

1. 執行上面的 sudo 安裝。`build-essential` 是 Core B 的直接 blocker；CUDA toolkit 是 RTX 端 llama.cpp 的 blocker。
