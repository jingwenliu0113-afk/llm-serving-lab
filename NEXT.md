# NEXT

## 目前位置

**Core — SPEC Day 1 已完成（NVIDIA 環境 smoke test 通過）；Week 1 進行中。**

- 模型已依使用者指示重新選擇（**decision 008**）：Core A 改為 Qwen3.5 4B + 9B，Core B 改為 RedHatAI Qwen3.5-9B W4A16。
  **正式下載前等使用者確認。**
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
- Core B：`RedHatAI/Qwen3.5-9B-quantized.w4a16`（GPTQ W4A16 g128），C_KV(2K / 8K / 32K) 約 23–28 / 8–10 / 2–3。
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

## 下一步

使用者已同意 008（2026-10-08），並要求下載前先完成 7 項修正：
- 已完成：008 的架構敘述、storage plan、遠端 uv / 環境規範、HF cache 位置、smoke 分段程序、language-model-only。
- 待完成：獨立 commit。

1. **Staged download**（每一步先跑 `scripts/disk_guard.py`，依 `docs/STORAGE_PLAN.md` 處理）：
   - D0：下載 9B 的 config / tokenizer 小檔，執行 converter `--vocab-only` 驗證。
   - D1：4B → BF16 GGUF → 3 個 quant → 寫入 hash 與 recipe → 刪除 safetensors。
   - D2：9B，步驟同 D1。
   - 遠端：以 user scope 安裝 `uv`，建立 `/home/wenn0113/llm-serving-lab/.venv`，
     把 RedHatAI 9B W4A16 下載到 `HF_HOME=/home/wenn0113/.cache/huggingface`。
2. 依 008「對實驗的影響」修訂 PREREGISTRATION 中與模型綁定的部分。
3. Mac 模型 smoke test → pilot（檢查 GDN graph splits）→ MMLU 排除清單 → Stage A 凍結 → 正式 run。
4. Core B：S1（2K）→ S2（8K）→ S3（32K）smoke。前提是遠端有 C compiler。

## Blocked

- ~~`ssh gpu-server`~~ → **resolved**（2026-10-08）。
- **遠端建置工具（需使用者以 sudo 安裝）**：WSL `Ubuntu-22.04` 內沒有 `gcc` / `g++` / `make` / `nvcc` / `cmake` / `uv`。

  ```bash
  sudo apt-get update && sudo apt-get install -y build-essential
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8     # 002：只裝 toolkit，不要裝 Linux driver
  ```

  - 受影響：llama.cpp CUDA build、vLLM 安裝與 smoke test（008 的 GDN kernel 驗證）、RTX 正式實測、Stage B。

## 待使用者確認

1. 遠端 `build-essential`（C compiler）：vLLM 的 Triton / FlashInfer JIT 需要它，Core B 的 S1 smoke 很可能因此卡住。
2. 遠端 `cuda-toolkit-12-8`：llama.cpp CUDA build 需要。
