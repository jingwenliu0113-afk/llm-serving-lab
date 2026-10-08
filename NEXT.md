# NEXT

## 目前位置

**Core — SPEC Day 1 已完成（NVIDIA 環境 smoke test 通過）；Week 1 進行中。**

- 模型已依使用者指示重新選擇（**decision 008**，使用者已確認）：Core A 為 Qwen3.5 4B + 9B，Core B 為 RedHatAI Qwen3.5-9B W4A16。
- 凍結前的一致性修訂（**decision 009**）已完成；hypothesis 的結構與門檻不變。
- Week 1 中 Mac 可做、不需大型下載的部分已完成：harness、輸出格式、quality 子集工具。
- Advanced / Optional 皆未觸碰。

## 上次完成（2026-10-08）

### Commit 狀態

- `b5f7414 chore: establish core baseline`（Day 1 的成果，26 個檔案；用 `git archive` 匯出的乾淨副本也測試通過）。
- baseline 之後的修改**尚未 commit**：
  - harness：`benchmarks/`、`tests/test_core_a_*`、`tests/test_run_core_a.py`、`tests/fixtures/`
  - quality 子集工具：`quality/`、`tests/test_mc_subset.py`
  - `docs/RESULTS_FORMAT.md`、decision 008，以及 pins / docs / MACHINE_POLICY 的更新
- 目前 `uv run pytest -q`：**150 passed、2 skipped**（skip 的是兩個 freeze 階段的檢查，屬預期）。

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

### Commit（最新在上）

- `a1a2200` raw outputs 不進 git，改 commit `raw_manifest.json`；`.gitignore` 加強
- `f98ac5d` 分析程式先於數據寫定（decision 011）：`core_a_analysis.py`（H-A1–H-A4、spill）、`quality_analysis.py`（H-A5 + Holm、downstream CI）
- `d60b3c8` quality pilot：`-fa off` / 重跑 determinism job，遮蔽式 pilot 報告（`quality/quality_pilot.py`）
- `033f4b8` Core A pilot plan 與遮蔽式報告（`benchmarks/core_a_pilot.py`）
- `3ee94ca` tokenizer 交叉驗證工具（`quality/tokenizer_crosscheck.py`）；build 加 `llama-tokenize`
- `0d678ba` results schema v2（decision 010）：Mac `/usr/bin/time -l`、RTX `memory.used` 取樣、graph splits
- `0353d7c` **4B 的 BF16 / Q8_0 / Q6_K / Q4_K_M 已產出並 pin**（4.48 / 3.46 / 2.71 GB，符合 008）
- `b409bcd` decision 009：凍結前一致性修訂（review 後）
- `9f37785`、`57c5b74` review 的 code 修正（Core B smoke teardown、produce_gguf 安全性）
- 目前 `uv run pytest -q`：**297 passed、2 skipped**

### 下載與 artifact

| 項目 | 位置 | 狀態 |
|---|---|---|
| 4B safetensors | Mac `models/hf/Qwen__Qwen3.5-4B@851bf6e8/` | ✅ 已驗證；GGUF 已產出。`delete-source` 待 Mac lock 空出（要 hash 約 19 GB） |
| 4B GGUF ×4 | Mac `models/gguf/` | ✅ sha256 已寫入 pins，recipe 在 `results/artifacts/` |
| 9B safetensors（19.33 GB） | Mac `models/hf/Qwen__Qwen3.5-9B@c2022362/` | ⏳ curl 續傳中（約 2 MB/s，預計 21:30 前後完成；scratchpad `fetch_9b.log`）；小檔集合已與 4B 相同 |
| Core B checkpoint | WSL HF cache `…/models--RedHatAI--Qwen3.5-9B-quantized.w4a16/snapshots/a398088c…/` | ✅ sha256 已驗證 |
| quality 資料集 | Mac `data/datasets/`（git-ignored） | ✅ 依 pins 驗證；pilot 子集在 `data/datasets/pilot/` |

### Tokenizer 相關（Qwen3.5）

- HF tokenizer：MMLU 排除 3 題（2151、2189、13184）；HellaSwag、ARC 也走同一個 `--multiple-choice` scorer，檢查結果為 0 題。
- WikiText 依 HF tokenizer 為 580 chunks。
- 以上要用 `quality/tokenizer_crosscheck.py` 對 pinned llama.cpp tokenizer 驗證（需先 rebuild 出 `llama-tokenize`）。

## 下一步（依序；Mac 重負載都要等 Mac heavy lock 沒被他人占用）

1. Mac lock 空出時（incident-investigator 的 soak 每輪約 35 分鐘，持續在取 lock）：
   - 重新 build metal（`scripts/build_llama_cpp.sh`，多了 `llama-tokenize`），跑 `tokenizer_crosscheck.py`，
     結果 commit 到 `quality/subsets/tokenizer-crosscheck.json`；若與 pins 不同，以 llama.cpp 為準修正並寫 log。
   - `produce_gguf.py delete-source --size 4B` → commit `deletions.jsonl`。
   - Mac 模型 smoke（4B-Q4_K_M 一次 `llama-completion`）。
   - Core A pilot（4B）：`core_a_pilot.py plan --sizes 4B` → `run_core_a.py` → `core_a_pilot.py report`；
     決定 `--poll`、確認 `-fa auto` 與 graph splits、memory 交叉核對。
   - quality pilot（4B）：`run_quality.py plan --mode pilot` → run → `quality_pilot.py`（只看遮蔽量）。
2. 9B 下載完成後：`tools/with-mac-lock uv run python scripts/produce_gguf.py produce --size 9B` → commit pins / recipe
   → `delete-source --size 9B` → commit；之後補 9B 的 pilot。
3. 完成 PREREGISTRATION §5 Stage A checklist → `freeze.mac = true` → commit → 用 `tools/with-mac-lock` 跑正式 Core A（Mac）與 quality。
4. Core B：使用者裝好 `build-essential` 後，用 `scripts/deploy_bundle.py` 部署，再在新的 session 目錄重跑 S1 → S2 → S3。

## Blocked

- ~~`ssh gpu-server`~~ → **resolved**（2026-10-08）。
- **遠端 C compiler / CUDA toolkit（需使用者以 sudo 安裝）**：
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
