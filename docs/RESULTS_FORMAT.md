# Results format（Core A，schema v2）

SPEC Week 1 #3「固定輸出格式」。由 `benchmarks/run_core_a.py` 產生；欄位由 `tests/test_run_core_a.py` 鎖定。
修改格式要提升 `SCHEMA_VERSION` 並寫 decision log。

- v2（decision 010）：fit row 增加 `external_memory`（PREREGISTRATION §3 的外部 peak memory 量測）；`summary` 增加 `graph` 與 `fused_ops`；Mac 的 fit run 以 `/usr/bin/time -l` 包裝。

## 目錄結構

```text
results/core_a/<platform>/<session_id>/
├── plan.json          # 執行的 plan 全文（core_a_plan.py 產生；job 順序以此為準，不靠 seed 重算）
├── session.json       # session 摘要（見下）
├── rows.jsonl         # 每個 job 一列，依執行順序
├── env/
│   ├── preflight.json         # 開跑前的 env_capture；用來驗證 build 與 pin 一致
│   ├── <job_id>.before.json   # 每個 process 前
│   └── <job_id>.after.json    # 每個 process 後
├── raw/
│   ├── <job_id>.stdout.txt
│   ├── <job_id>.stderr.txt        # Mac fit：末尾的 /usr/bin/time -l 區塊保留在原檔
│   └── <job_id>.gpumem.txt        # RTX fit：nvidia-smi memory.used 每 200 ms 的取樣（MiB）
└── prompts/
    └── unit_x<count>.txt      # fit run 的 prompt（PROMPT_UNIT × count）
```

- `platform` ∈ {`macos-metal`, `wsl2-cuda`}。
- session 目錄已存在時，runner 拒絕執行。session 不續跑也不覆寫；中斷或 drift 時整個 session 重跑。
- `raw/` 與 `env/` 保留所有失敗的 process（OOM / crash 也要報告）。

## `session.json`

| 欄位 | 說明 |
|---|---|
| `schema_version` | 2 |
| `session_id`、`platform`、`mode` | `mode` ∈ {`pilot`, `formal`}；formal 需對應 stage 已凍結（006）與 lock（MACHINE_POLICY） |
| `seed`、`plan_sha256`、`pins_sha256` | plan 與 pins 的 hash（`plan_sha256` 是 canonical JSON 的 sha256） |
| `gpu_lock_held`、`mac_lock_held` | 是否在對應 lock 下執行 |
| `bin_dir`、`models_dir` | 執行時的絕對路徑 |
| `started_at`、`finished_at` | UTC ISO 8601 |
| `outcomes` | `{"ok": n, "oom": n, "error": n}` |
| `sentinel` | `start_tok_s`、`end_tok_s`、`drift_pct`、`drift`（> 3% 為 true，整個 session 需重跑）、`threshold_pct` |

## `rows.jsonl`（每列一個 job）

| 欄位 | 說明 |
|---|---|
| `job_id` | 例：`decode-8B-Q4_K_M-2048-r2`、`sentinel-start` |
| `kind` | `fit` / `ttft` / `decode` / `sentinel` |
| `position` | sentinel 為 `start` / `end`，其他為 null |
| `block` | pilot plan 的區塊（`fa_probe` / `poll` / `memory` / `calib` / `sentinel`，`benchmarks/core_a_pilot.py`）；正式 plan 為 null |
| `size`、`quant`、`L`、`rep`、`model` | cell 與 GGUF 檔名 |
| `argv` | 實際執行的完整指令（路徑已展開） |
| `exit_code` | 整數；timeout 為 null |
| `wall_s` | process 的 wall time |
| `outcome` | `ok` / `oom` / `error`（`core_a_parse.py`） |
| `stage` | 失敗階段：`load` / `context` / `eval` / `unknown`；ok 為 null |
| `reason`、`evidence` | 判定理由與觸發判定的 log 行 |
| `metrics` | bench 類：`avg_ns`、`stddev_ns`、`avg_ts`、`stddev_ts`、`samples_ns` |
| `summary` | fit 類：`n_ctx`、`flash_attn`、`prompt_tokens`、`eval_runs`、`memory`（`unit`、`devices[]`、`host`、`other[]`，單位 MiB）、`graph`（`nodes`、`splits`；pp / tg 不同時為 `"a / b"`）、`fused_ops`（fused op probe 的結果行，只在 `auto` 設定時出現） |
| `external_memory` | fit 類（v2）：Mac 為 `{"peak_footprint_bytes", "max_rss_bytes"}`（`/usr/bin/time -l`）；RTX 為 `{"gpu_mem_used_peak_mib", "gpu_mem_used_first_mib", "n_samples", "samples"}`（B 在分析時扣除）；量不到或非 fit 為 null |
| `stdout`、`stderr`、`env_before`、`env_after`、`gpu_mem_samples` | session 目錄內的相對路徑（`gpu_mem_samples` 只在 RTX fit） |

## 判定規則摘要

- **bench（ttft / decode / sentinel）**：
  - exit 0，而且 stdout 恰好有 1 列 jsonl，各項設定都符合 PREREGISTRATION §4，才算 `ok`。
  - 必須符合的設定：`n_batch`、`n_ubatch`、`type_k/v`、`n_gpu_layers`、`flash_attn = 1`、`load_mode`、`n_prompt/n_gen/n_depth`、`n_threads`、`poll`、`build_commit`，以及 5 個 samples。
  - 任一項不符就是 `error`，原因寫在 `reason`。
- **fit**：`llama-completion --log-jsonl`（Mac 外層為 `/usr/bin/time -l`，判定前先移除 stderr 末尾的 time 區塊）。以下都滿足才算 `ok`：
  - exit 0；
  - prompt token 數等於 L − 128；
  - 有 eval timing；
  - 有 `fit_memory_breakdown` event。
- **OOM**：出現 pinned source 中的配置失敗字串，而且 process 失敗或不完整，就判為 `oom`。完成的 run 若含配置失敗字串，記為 `error`，交人工檢查。
- spill、H-A1 的 predicted fit / OOM、各 hypothesis 的比較，都屬於分析階段，不在 runner 判定。
