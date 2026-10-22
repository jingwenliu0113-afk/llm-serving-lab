# Storage plan（大型 artifact 的保留、刪除與空間預算）

- 日期：2026-10-08（使用者要求：正式下載前先規劃，不讓所有大型中間產物同時常駐 Mac）
- 工具：`scripts/disk_guard.py`（測試：`tests/test_disk_guard.py`）
- 原則：**可以刪除的只有能重建的 scratch；實驗結果、manifest、hash、config、重現所需的 recipe 一律不刪。**

## 分級

| 等級 | 內容 | 位置（Mac） | 規則 |
|---|---|---|---|
| **K：必須保留（進 git）** | `results/`（rows、session、env、`raw_manifest.json`；raw log 本身只留本機 archive，見下）、`configs/pins.toml`（所有 hash）、converter / vLLM lock、`docs/`、artifact recipe（`results/artifacts/*.json`，見下）、子集題號清單 | repo | **永不刪除**。不得為了省空間刪除 |
| **W：工作集（實驗期間保留；可重建）** | 6 個 quantized GGUF（Core A 兩台機器都要用，sha256 已 pin） | `models/gguf/` | 保留到 Core A 與 quality 正式 run 都完成。之後可刪，因為可依 recipe 重建且有 hash 驗證 |
| **S：可重建 scratch** | 原始 HF safetensors（可依 pinned revision 重新下載，shard sha256 已 pin） | `models/hf/<org>__<name>@<rev>/` | BF16 GGUF 產出、sha256 寫入 pins、recipe 存檔之後即可刪 |
| **S** | BF16 GGUF（可由 safetensors 重建） | `models/gguf/` | 量化完成後仍保留，因為它是 quality 的 BF16 reference；該 size 的 quality 正式 run 完成後即可刪 |
| **S** | KLD logits（約 74 GB / size，可由 BF16 GGUF 重建） | `data/kld/`〔020 起在 WSL：`~/llm-serving-lab/data/kld/`〕 | 只在 Stage A 凍結後的正式 quality run 時產生；同一 size 的 3 個 quant 算完即刪；**一次只存在一個 size**。〔040：Stage Q 的 base 在 `data/kld/<session>/`。Q1 的 Mac base 只有前 200 chunks，約 25 GB / size；Q2 在 WSL，2K / 8K 的 base 和 512 的大小相近。〕 |
| — | quality 資料集（數 MB，sha256 已 pin） | `data/datasets/` | 保留（很小）；不進 git |
| **K（本機 archive，不進 git）** | 各 session 的 `raw/`（process 的完整 stdout / stderr、GPU 記憶體取樣） | `results/**/raw/` | **永不刪除**；不進 git（使用者 2026-10-08 指示），以該 session 已 commit 的 `raw_manifest.json` sha256 驗證 |

刪除 S 級檔案前，必須同時滿足下列條件：

1. 由它產生的下游檔案已驗證，sha256 已寫入 `pins.toml`。
2. 產生它與由它衍生的完整指令與 toolchain 已寫入 artifact recipe，並且已 commit。
3. 刪除這件事記錄在 `NEXT.md`，或寫在該 recipe 中。

source safetensors 的刪除由 `scripts/produce_gguf.py delete-source` 強制執行這三個條件：

- 刪除前會重新驗證磁碟上每個衍生 GGUF 的大小與 sha256，必須同時符合 pins 與 recipe。
- `pins.toml` 與 recipe 必須已 commit。
- 刪除紀錄 append 到 `results/artifacts/deletions.jsonl`（K 級，刪除後 commit）。

順序固定為：`produce` → commit pins 與 recipe → `delete-source` → commit `deletions.jsonl`。

## Artifact recipe（K 級，重現用）

每產生一個大型 artifact，就寫一份 `results/artifacts/<filename>.json`，內容包括：

- 輸入：repo@revision 與 sha256，或上游 GGUF 的 sha256。
- 完整指令。
- toolchain：llama.cpp commit / build number、converter lock 的 sha256、Python 版本。
- 輸出的 sha256 與大小。
- 產生時間，以及 `disk_guard` 的檢查結果。

有了 recipe，任何 S 或 W 級檔案刪除後都能逐位元重建，並用 hash 驗證。

## 安全餘量與檢查

- 預設保留 **50 GiB** 的剩餘空間（`DEFAULT_RESERVE_GIB`），約為 Mac 資料磁碟 460 GiB 的 11%。
- **每一次**下載、轉換、量化、產生 KLD 之前，都要先跑：

  ```bash
  uv run python scripts/disk_guard.py --path <target dir> --need-gib <預估大小>
  ```

  回傳碼不是 0 就停止，先依上表刪除可刪的 S 級檔案，或記為 blocker。
- 可用空間會隨其他專案變動（2026-10-08 實測 175 GiB），所以以執行當下的檢查為準，不靠預算表。

## 處理順序與空間預算（Mac；可用 175 GiB − 保留 50 GiB = 125 GiB）

| 階段 | 動作 | 新增 | 峰值（GiB） |
|---|---|---|---|
| D0 | 下載 9B 的 config / tokenizer 小檔，執行 converter `--vocab-only` 驗證 | < 0.1 | ~0 |
| D1 | 下載 4B safetensors（9.34 GB）→ BF16 GGUF → 3 個 quant → 寫入 hash 與 recipe → **刪 4B safetensors** | 8.7 + 7.8 + 9.9 | 26.4（刪除前） |
| D2 | 下載 9B safetensors（19.33 GB）→ BF16 GGUF → 3 個 quant → 寫入 hash 與 recipe → **刪 9B safetensors** | 18.0 + 16.7 + 20.9 | 4B 的 17.7 + 55.6 = 73.3（刪除前） |
| 常駐 | 6 個 quant（30.8）+ 2 個 BF16 reference（24.5） | | 55.3 |
| Q4B | 正式 quality（4B）：產生 4B KLD（約 69 GiB）→ 算完即刪 → 刪 4B BF16 | +69 | 124.3 ⚠️ |
| Q9B | 正式 quality（9B）：產生 9B KLD（約 69 GiB）→ 算完即刪 → 刪 9B BF16 | +69 | 116.5 |

- Q4B 的峰值貼近上限。如果當時 `disk_guard` 不通過，就**先跑 9B quality**，或先把 4B 的 quant 暫時移出 Mac，
  不得刪除 K 級檔案。
- **2026-10-08 依實際可用空間重估**（19:18 實測可用 134.0 GiB，9B 已下載 7.0 GiB）：
  - 9B 下載完、產出 9B GGUF、刪掉 9B safetensors 之後，預估可用約 103 GiB。
  - KLD base 每個 size 68.4 GiB（580 × 255 × 248,320 × 2 B），扣掉後只剩約 35 GiB，**低於 50 GiB 餘量約 15 GiB**。
  - 處理方式（不動 K 級、不重新下載）：
    1. Core A 正式 run 需要的 6 個 quant 先用完，RTX 端也先拷貝並驗證 sha256。
    2. 跑某個 size 的 quality 時，**暫時刪除另一個 size 的 3 個 quant**（W 級，約 19.5 GiB）。
    3. 那個 size 的 quality 做完、KLD 刪掉之後，用 `produce_gguf.py produce` 從仍保留的 BF16 重新量化。
       `produce` 只在 sha256 與 pin 相同時才放回原位，不同就中止並保留現狀。
  - 若屆時其他專案使空間再減少，先記為 blocker，不降低 50 GiB 餘量。
- quality 正式 run 屬於 Mac heavy workload，必須用 `tools/with-mac-lock`（MACHINE_POLICY）。
  〔020 起不適用：正式 quality 與 KLD 都改在 RTX（WSL）上，使用 GPU flock。
  本節 Q4B / Q9B 的 Mac 磁碟規劃，以及「暫時刪除另一個 size 的 quant」的步驟都不再需要。Mac 上的 BF16 依 S 級規則處理。〕

## RTX（WSL2 `Ubuntu-22.04`，家目錄 ext4，可用約 928 GB）

〔2026-10-09 實測（020 審查）：WSL ext4 為 1007G，可用 873G。這是 VHDX 的上限；VHDX 所在的 Windows C: 實際剩餘 1.3 TB
（唯一的磁碟，`/mnt/c` 與 PowerShell `Get-PSDrive` 一致），所以 873G 確實可用。〕

| 內容 | 位置 | 等級 |
|---|---|---|
| 專案副本與 Python 環境 | `/home/wenn0113/llm-serving-lab/`（`.venv` 由 user scope 的 `uv` 建立） | 程式由 Mac 以 `scripts/deploy_bundle.py` 部署（乾淨 HEAD + `DEPLOYED_COMMIT` + `DEPLOY_MANIFEST.json`，env_capture 會驗證），Git 只在 Mac |
| HF cache（Core B checkpoint） | `HF_HOME=/home/wenn0113/.cache/huggingface` | W（Core B 期間保留） |
| Core A 的 quantized GGUF（由 Mac 傳過去，以 sha256 驗證） | `/home/wenn0113/llm-serving-lab/models/gguf/` | W |
| uv 本體 / Python / cache | `~/.local/bin/uv`、`~/.local/share/uv/`、`~/.cache/uv/` | 工具 |
| BF16 reference GGUF（2 個，約 24.5 GiB；Mac 以 `sync_gguf_to_wsl.py --set references` 傳送，sha256 驗證；020） | `/home/wenn0113/llm-serving-lab/models/gguf/` | S（該 size 的正式 quality 完成後即可刪） |
| quality 資料集（14 個檔案，sha256 與 Mac 及 pins 一致；020） | `/home/wenn0113/llm-serving-lab/data/datasets/` | 保留 |
| KLD logits（約 68 GiB / size） | `/home/wenn0113/llm-serving-lab/data/kld/`（`run_quality` 自動建立） | S（同一 size 的 3 個 quant 算完即刪，一次只存在一個 size） |

- 一律不放在 `/mnt/c/...`，也不使用 `~/venvs/gpu-check` 或系統 Python 安裝套件。
- 遠端空間充足，但每次大型下載前仍跑 `disk_guard.py`（同一支 script，可直接在 WSL 執行）。
