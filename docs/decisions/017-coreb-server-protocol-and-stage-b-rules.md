# 017 — 補齊 003 的 Core B server 固定參數，並事先寫定 Stage B 的 U、K、JIT 計時與 pin-memory A/B

- 狀態：Accepted（2026-10-09）
- 日期：2026-10-09
- 時點：Stage A 尚未凍結，NVIDIA 沒有任何 Core B 數據（S1 尚未通過）。
- 對應：003 的「Core B 固定參數」表與 smoke gate 3–4、PREREGISTRATION §4 / §6、006（Stage B 只能代入實測值）。

## 背景：程式與 003 不一致（先記錄差異，再處理）

003 開頭寫明：「vLLM 版本與其餘 server protocol 仍有效」（008 只換了模型）。但對照 003 的固定參數表，
`core_b_plan.server_argv` 少了以下幾項：

| 003 要求 | 程式現況 |
|---|---|
| `--max-num-batched-tokens 2048`（明確寫出） | 沒有 |
| performance mode / async scheduling 的實際生效值明確寫入指令 | 沒有 |
| `--attention-backend FLASH_ATTN` 明確指定 | 沒有，靠自動選擇（S1 log 自動選中的也是 FLASH_ATTN） |
| `--kv-cache-memory-bytes` 固定值（先做一次丟棄用的 boot） | 沒有 |
| client `--backend vllm` | `openai`（0.30.0 兩者對應同一個 request function，行為相同） |

另外，006 規定 Stage B **只能代入實測值與執行參數**，但下列推導規則都還沒寫定：
- `--gpu-memory-utilization`：003 只寫「依實測 Windows idle VRAM baseline 設定」。
- K 的取法。
- JIT 計時的程序。
- A/B 的次數、L 與順序。

這些都必須在 Stage A 凍結前寫定。

## 決策

1. **server 固定 flags**（`core_b_plan.server_argv`，smoke 以外的所有 Core B server 共用）：
   - `--max-num-batched-tokens 2048`
   - `--performance-mode balanced`
   - `--async-scheduling`：在本設定下，0.30.0 的預設會解析為開啟（generation、無 speculative decoding、uni executor）。
     明確開啟時若不相容會直接報錯，不會默默關閉。
   - `--attention-backend FLASH_ATTN`
   - `--kv-cache-memory-bytes K`
   - client 改為 `--backend vllm`。
   - 驗證方式：log 的 "Using FLASH_ATTN"、"Chunked prefill is enabled with max_num_batched_tokens=2048"，以及 non-default args。
2. **U（`--gpu-memory-utilization`）**：取 0.01 的倍數中，滿足 U × V_total ≤ V_total − B − 512 MiB 的最大值。
   - vLLM 啟動時要求可用 VRAM ≥ U × total；512 MiB 是給 Windows idle VRAM 波動的餘量。
   - 例：B = 380 MiB、V_total = 16303 MiB 時，U = 0.94。
3. **K（`--kv-cache-memory-bytes`）**：
   - 在 U 下做兩次 boot，不帶 K，使用全新的 Triton / FlashInfer / vLLM cache。
   - 第一次是冷 boot，丟棄（003：冷熱 compile cache 會讓 KV 池相差約 5%）。
   - 第二次是熱 boot，同一個 cache。K 為熱 boot log 中的 `Available KV cache memory`，向下取整到 256 MiB 的倍數。
   - 之後所有 Core B server 都帶 `--kv-cache-memory-bytes K`。
   - num_gpu_blocks / block_size 由帶 K 啟動的 server 的 `/metrics` 讀取（009），再依 §2 算 C_KV 與 grid。
4. **JIT 計時（003 #3）**：上面那兩次 boot 從啟動到 `/health` 200 的秒數，就是冷 / 熱啟動時間。
   熱 boot 較快，代表 cache 有生效。只作描述，不進任何判定。
5. **`VLLM_WSL2_ENABLE_PIN_MEMORY` A/B（003 #4）**：
   - 依序以 0、1 各 boot 一次，都帶 U 與 K。
   - 每次跑 `uniform`、L = 2048，C = 1 與 C = 16 各 3 次 `vllm bench serve`。
     num-prompts 與 warmup 依 007；只計入通過驗證的 run。
   - 判定照 003：1 在兩個 C 的平均 output throughput 都比 0 高 ≥ 5%，**而且四組的 run-to-run CV 都 < 3%**，才改為 1；否則維持 0。
   - 任一組少於 3 次有效 run 時，同樣維持 0。
6. 2–5 都在 GPU lock 下執行，有其他 GPU process 時拒絕執行，結果不得覆寫。
   實作在 `benchmarks/coreb_stage_b.py`（`boots`、`pin-ab`）；B、C_ctx、超量測試在 `benchmarks/stage_b.py`。
7. 正式 Core B run：plan 必須帶 K（`build_uniform_plan(..., kv_cache_memory_bytes=K, gpu_memory_utilization=U)`），否則 runner 拒絕。

## 為什麼

- 讓實作符合仍然有效的 003。
- 把「依實測設定」改寫成確定的公式，讓 Stage B 只代入 B、V_total 與 log 的實測值，符合 006。
- A/B 每點 3 次，才算得出 003 要求的 run-to-run CV。

## 放棄的替代方案

- **U 沿用 smoke 的 0.85**：會白白少掉約 1.4 GiB 的 KV 池，與 003「依實測 baseline 設定」不符。
- **不固定 K，只用 U**：003 已指出冷熱 cache 會讓 KV 池浮動約 5%，C_KV 與 grid 會隨 boot 改變。
- **A/B 交錯順序（0、1、0、1）**：要多 boot 兩次。5% 的門檻遠大於 session 內的漂移，維持 0 → 1 的單一順序，並如實記錄。

## 對實驗的影響

- PREREGISTRATION §4 補上 Core B server flags 與 U / K / JIT / A/B 的規則；§6 只代入實測值。
- smoke（S1–S3）的指令依 008 原樣，不受影響：它只判定能否啟動，flags 已在 008 寫定。
- 以上都發生在任何 Core B 數據之前。
