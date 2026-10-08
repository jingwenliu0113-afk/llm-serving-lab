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

## 進行中（2026-10-08 傍晚）

### Commit

- `1d952c2`：harness、quality 子集、storage plan、decision 008，以及 7 項修正。
- **之後尚未 commit 的修改**：
  - converter 依賴：`configs/converter.requirements.in`、`configs/converter.lock.txt`（hashed，Mac arm64），以及 pins 中的 `converter_lock` 欄位
  - vLLM 依賴：`configs/vllm.requirements.in`、`configs/vllm.lock.txt`（sha256 `4a424aa8…`，WSL 解析）
  - `benchmarks/coreb_smoke.py` 與其測試
  - `env_capture` 的 `DEPLOYED_COMMIT` fallback
  - PREREGISTRATION 依 008 重新推導、006 的超量測試配置
  - `results/core_b/smoke/20261008-s1-no-cc/`（S1 的失敗證據）
  - `scripts/produce_gguf.py` 與其測試：驗證來源 sha256、檢查 Mac lock、跑 disk_guard，再依序轉 BF16 → 3 個 quant。
    每個產物都會回填 pins 並寫 recipe 到 `results/artifacts/`。刪除來源檔需另外執行 `delete-source`，
    而且必須在 recipe 與 hash 都存好之後。

  目前 `uv run pytest -q`：172 passed、2 skipped。

### 下載與 artifact

| 項目 | 位置 | 狀態 |
|---|---|---|
| D0：9B 的 config 與 tokenizer | Mac `models/hf/Qwen__Qwen3.5-9B@c2022362/` | ✅ converter `--vocab-only` 驗證通過（qwen35，vocab 248,320） |
| D1：4B 權重（9.34 GB） | Mac `models/hf/Qwen__Qwen3.5-4B@851bf6e8/` | ⏳ 下載中，約 2.6 MB/s。hf-xet 卡住、hf 的 HTTP 讀取逾時又丟失進度，已改用 curl：pinned revision URL、斷點續傳、每個檔案下載完立即驗 sha256 |
| D2：9B 權重（19.33 GB） | Mac，與 D0 同目錄 | 待 D1 處理完 |
| Core B checkpoint（11.46 GB） | WSL `/home/wenn0113/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-quantized.w4a16/snapshots/a398088c…/` | ✅ sha256 驗證通過 |
| converter 環境 | Mac `third_party/convert-venv`（git-ignored） | ✅ torch 2.11.0、transformers 4.57.6 |
| vLLM 環境 | WSL `/home/wenn0113/llm-serving-lab/.venv`（uv 0.11.8 user scope，Python 3.12.13，7.7 GB） | ✅ vllm 0.30.0、torch 2.13.0+cu130 |

### Core B smoke

- **S1（2K）❌**：`RuntimeError: Failed to find C compiler`。Triton 的 JIT 需要系統 C compiler。
- 失敗之前都正常：模型載入（9.48 GiB，Marlin W4A16）、FA2、block size 528、GDN 走 Triton/FLA。
  失敗後 GPU 記憶體回到 379 MiB，沒有殘留 process。
- 依使用者指示，S1 沒通過就不進 S2 / S3。沒有用使用者空間的 compiler 繞過，因為那會在 pinned 環境中加入未記錄的元件。

## 下一步

1. D1 下載完成後，執行 `uv run python scripts/produce_gguf.py produce --size 4B`（若 Mac heavy lock 被他人占用，工具會拒絕執行），
   再執行 `delete-source --size 4B`。
2. D2（9B），步驟相同。
3. Mac 模型 smoke test → pilot（GDN graph splits、`--poll` 規則、MMLU 排除清單、WikiText chunk 數）→ Stage A 凍結 → 正式 run（用 `tools/with-mac-lock`）。
4. 使用者裝好 `build-essential` 後，在新的 session 目錄重跑 S1 → S2 → S3。

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

## 待使用者確認

1. 執行上面的 sudo 安裝（`build-essential` 是 Core B 的直接 blocker）。
2. 是否要把 `1d952c2` 之後的修改另做一個 commit。
