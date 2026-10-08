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

### Commit（最新在上；皆未 push）

- （本次）decision 009：凍結前一致性修訂（H-A4(b) 4.45 / 4.84、spill anchor 改 4B-Q4_K_M、C_KV 單一規則、fit prompt 對齊 harness）
- `9f37785` Core B smoke：process group teardown、真實 log fact、GPU lock / idle 1000 MiB、不覆寫證據；deploy bundle + manifest
- `57c5b74` produce_gguf：刪 source 前驗證 GGUF 與 commit、`.tmp` staging 不覆寫 pinned 檔、toolchain 檢查、每步重查 lock
- `b7863a9` NEXT 更新
- `61a5984` quality runner（`quality/run_quality.py`、`quality_parse.py`）與 `[quality.expected]`
- `a2785df` MMLU 結構性排除（3 題）與子集 hash
- `df9c720` Core B S1 失敗證據（WSL 缺 C compiler）
- `0c0e099` PREREG 依 008 重新推導
- `0d72a98` Core B 分段 smoke runner、GGUF 產出工具、env `DEPLOYED_COMMIT` fallback
- `94190f5` converter / vLLM 的 hashed lock
- 目前 `uv run pytest -q`：244 passed、2 skipped

### 下載與 artifact

| 項目 | 位置 | 狀態 |
|---|---|---|
| 9B 的小檔（config / tokenizer / README / LICENSE 等） | Mac `models/hf/Qwen__Qwen3.5-9B@c2022362/` | ✅ 與 4B 的小檔集合相同（README 會寫入 GGUF metadata） |
| 4B 權重 | Mac `models/hf/Qwen__Qwen3.5-4B@851bf6e8/` | ✅ 2 個 shard sha256 已驗證（18:24） |
| 9B 權重（19.33 GB） | 同上 9B 目錄 | ⏳ curl 續傳下載中（scratchpad `fetch_9b.log`） |
| Core B checkpoint | WSL HF cache `…/models--RedHatAI--Qwen3.5-9B-quantized.w4a16/snapshots/a398088c…/` | ✅ sha256 已驗證 |
| quality 資料集 | Mac `data/datasets/`（git-ignored） | ✅ 全部依 pins 驗證；pilot 子集在 `data/datasets/pilot/` |
| converter / vLLM 環境 | Mac `third_party/convert-venv`、WSL `~/llm-serving-lab/.venv` | ✅ |

### Tokenizer 相關（Qwen3.5）

- MMLU 排除 3 題（2151、2189、13184），4B 與 9B 的 tokenizer 相同。
- WikiText 依 HF tokenizer 為 580 chunks。這兩項都需要以 llama.cpp tokenizer 交叉驗證（要先 build `llama-tokenize`）。

## 下一步（依序；Mac 重負載都要等 Mac heavy lock 沒被他人占用）

1. Mac lock 空出時：
   - 把 `llama-tokenize` 加進 build targets，重新 build metal。
   - 以 llama.cpp tokenizer 驗證 WikiText 的 580 chunks 與 MMLU 排除清單；若不同就修正 pins，以 llama.cpp 為準。
2. 4B（已下載）：`produce --size 4B` → commit `configs/pins.toml` 與 `results/artifacts/` → `delete-source --size 4B`
   → commit `results/artifacts/deletions.jsonl`。順序由 `delete-source` 強制（STORAGE_PLAN）。
3. 9B 下載完成後同上。
4. Mac 模型 smoke test（4B-Q4_K_M），以及 Core A 與 quality 的 pilot：
   - Core A pilot：`benchmarks/core_a_plan.py --mode pilot`。
   - quality pilot：`quality/run_quality.py plan --mode pilot`。
   - 決定 `--poll`。
   - 檢查 GDN 的 graph splits。
   - 交叉核對 memory metric。
5. 完成 PREREGISTRATION §5 Stage A checklist → `freeze.mac = true` → commit → 用 `tools/with-mac-lock` 跑正式 Core A（Mac）與 quality。
6. Core B：使用者裝好 `build-essential` 後，在新的 session 目錄重跑 S1 → S2 → S3。

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
