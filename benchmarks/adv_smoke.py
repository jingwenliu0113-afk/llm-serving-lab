#!/usr/bin/env python3
"""Advanced feasibility smokes FS1-FS5 (decision 024 「可行性 smoke」; 023): can the Advanced harness run on the RTX?
Never a measurement: no smoke runs at a formal cell (C_knee with the full N), and everything the smoke keeps is masked
(pass / fail and counts only). Runs in WSL with the project venv, under the shared GPU lock.

Every boot goes through the runner (run_adv.run_session) as its own one-boot, non-formal session
<out>/<probe>/, so the smoke exercises the formal code path: plan check, receiver, timestamping reader,
manipulation check, /metrics, 1 Hz series, bench with timeout, adv_decomp, teardown. run_adv.plan_problems
requires Core's N / warmup at a boot's C and the plan's c / timeouts_s, so a smoke plan declares c = the smoke's C
and timeouts_s = its timeout, keeps num_prompts(C) / num_warmups(C) in the boot and records the bench's real
counts as smoke_counts. The runner's row therefore says N = num_prompts(C) and ok = false (completed != that N);
the smoke judges completion from its own count of the bench JSON (N_bench), and run_adv waits its full 30 s for
the spans it will never get. The smoke wraps two of the runner's injected callables:
- run_bench: the bench runs at the smoke's own N / warmup (--num-prompts and --num-warmups are rewritten,
  --max-concurrency is set to the smoke's C; nothing else in the argv changes; the timeout comes from the plan);
  right after the bench the smoke
  counts its --save-detailed JSON (completed, failed, lengths), probes /health and looks for survivors of the
  client's process group (/proc cmdlines carrying the request-id prefix);
- http_get: every /metrics text is reduced to cache_config_info (num_gpu_blocks, block_size) and
  num_preemptions_total.
After the session: server / receiver ports closed and the GPU back to idle (teardown), then masking.

Probes (024):
- FS1: B0, T, G, TG, K3 boot once each (uniform-2K, C 1, N 1): /health 200, log facts and non-default args exactly
  as the condition table (the runner's manip.facts_ok), /metrics num_gpu_blocks = the expected count and
  block_size = 528, the receiver got the main spans (A10 a), every iteration line parses with the runner's
  epoch-ms stamp; T / TG pass the chunk check. K3: the measured num_gpu_blocks against the 024 rule (C_KV(2048) >= 25,
  C_KV(8192) >= 8, >= 128, >= 67 via core_b_plan.c_kv / adv_plan.k3_problems), and the next lower 256 MiB multiple
  provably fails (adv_plan.blocks_bounds); otherwise the next step is K + 256 MiB (--k3-bytes).
- FS2: row index 4 of the pinned mixed-2048 file (32640 tokens) as a one-row dataset, C 1 under B0 and T: B0 has a
  step with 1 context request and 2048 context tokens and no step above 2048 (B = 2048); T has no step above 512.
- FS3: the first four long rows (indices 4, 10, ...) at C 4 (4 x 66 > 247 blocks), cap 300 s: G preempts (> 0)
  and all four complete with the right lengths; B0 (control) has no preemption.
- FS4: B0 uniform-2K and mixed-2K (the pinned file's first 20 rows, CustomDataset / pandas, 022) at C 2, N 20:
  every contract row field present, derived/ written, the scrape delta complete (every histogram / counter of
  run_adv, no delta error), the 1 Hz shares computed and the 1 Hz series longer than its start / stop samples,
  A10 (a) (b) (d) pass and (c) does not fail, the classifier ran; TG mixed-2K at C 2, N 20 only has to complete.
- FS5: B0 uniform-2K, C 2, N 20 with a 5 s timeout: row ok = false with reason timeout, no client process left,
  the server still healthy after the kill, teardown clean (server and receiver ports closed, GPU idle, no
  teardown problem).

Masking (023 / 024): the bench JSON is counted, then deleted when the boot is over (the runner reads it first);
every raw file carrying per-request or histogram latency (spans, OTLP bodies, Prometheus text) is deleted; bench
logs lose their result section and progress bars; rows.jsonl keeps counts and flags only and derived/*.jsonl lose
every latency column; session.json loses the runner's startup_s and bench window (t_before / t_after) and is scanned
like the rewritten files. Server logs stay in raw/ (git-ignored, local); the smoke reads token counts from them,
never times. The smoke's own report (smoke.json) holds pass / fail, counts and the K3 block count.

--gpu-lock-held is verified (run_adv.held_gpu_lock: this process holds the lock through flock's descriptor), not
trusted; run_adv checks it again at each probe's session start and before its boot.

Usage (WSL, ~/llm-serving-lab):
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python benchmarks/adv_smoke.py --gpu-lock-held \\
        --out results/advanced/smoke/<session> --mixed-dataset data/core_b/mixed-2048.jsonl [--only FS1,FS2]
        [--k3-bytes 3489660928]

Stdlib only, Python 3.10 compatible (tomllib, run_core_b and run_adv are imported lazily).
"""

from __future__ import annotations

import copy
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import adv_decomp as ad
import adv_plan as ap
import core_b_plan as bp
import coreb_smoke as smoke

SCHEMA_VERSION = 1
FS_NAMES = ("FS1", "FS2", "FS3", "FS4", "FS5")
L_SMOKE = 2048
IDLE_MIB = 1000
RELEASE_TIMEOUT_S = 60
SERVER_PORT = 8000                  # run_adv.PORT (Core B's port)
FS2_ROW = 4                         # 024: the 5th row of the pinned mixed-2048 file, 32640 tokens
FS3_LONG_ROWS = 4                   # 024: indices 4, 10, ...; 4 x 66 = 264 > 247 blocks
FS3_CAP_S = 300                     # 024: "上限 5 分鐘"
FS5_TIMEOUT_S = 5
SMALL_TIMEOUT_S = 300               # FS1 / FS2: one or two requests
BLOCK = 10                          # the mixed file comes in blocks of ten (9 short + 1 long, core_b_plan.mixed_rows)
LONG_RATIO = 3                      # a long row (32640 tokens) has >= 6x the characters of any short one (1920)
BANNER = "Serving Benchmark Result"
_TQDM = re.compile(r"[^\r\n]*\|[^\r\n|]*\|\s*\d+/\d+ \[[^\]\r\n]*\][^\r\n]*")
LATENCY_KEY = re.compile(r"(?i)(ttft|tpot|itl|e2e|latency|_ms$|duration|throughput|elapsed|arrival|queue_time|"
                         r"^[QPOF]$|^t_before$|^t_after$|^startup_s$)")  # the last three: run_adv's bench window
LATENCY_MARKERS = (b"gen_ai.latency", b'"ttfts"', b"time_to_first_token_seconds", b"request_queue_time_seconds",
                   b"request_prefill_time_seconds")
ROW_KEEP = ("cond", "L", "mix", "restart", "block", "C", "N", "ok", "reasons", "timeout", "skipped", "num_prompts",
            "completed", "failed", "preemptions", "manip", "healthy_after_bench", "n_steps_window", "decomp_error")
ROW_FIELDS = ("cond", "L", "mix", "restart", "block", "C", "N", "ok", "reasons", "timeout", "skipped",
              "num_prompts", "completed", "failed", "duration", "request_throughput", "output_throughput",
              "total_token_throughput", "short_class_p95_ttft_ms", "long_class_p95_ttft_ms", "p95_tpot_ms",
              "preemptions", "kv_peak", "manip", "decomp")
NOT_NULL = ("preemptions", "kv_peak", "manip", "decomp")
MANIP_FIELDS = ("facts_ok", "blocks_ok", "chunk_ok", "num_gpu_blocks")
DECOMP_FIELDS = ("valid", "checks", "tail", "q_shares", "q_shares_sens", "residual_bm_share", "unexplained_break_share")


class SmokeError(RuntimeError):
    pass


# ------------------------------------------------------------------ datasets --

def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def long_row_indices(lines: list, n: int) -> list:
    """The long row of each of the first n blocks of ten: its longest prompt, which must be >= LONG_RATIO x the
    block's second longest (no tokenizer needed; the bench's input_lens confirm 32640 afterwards)."""
    out = []
    for b in range(n):
        block = lines[b * BLOCK:(b + 1) * BLOCK]
        if len(block) < BLOCK:
            raise SmokeError(f"mixed dataset has no complete block {b}")
        size = sorted(((len(json.loads(x)["prompt"]), b * BLOCK + i) for i, x in enumerate(block)), reverse=True)
        if size[0][0] < LONG_RATIO * size[1][0]:
            raise SmokeError(f"block {b}: no single long row (longest {size[0][0]} vs {size[1][0]} characters)")
        out.append(size[0][1])
    return out


def prepare_datasets(src, out_dir, pins: dict) -> dict:
    """FS2 / FS3 subsets of the pinned mixed-2048 file, rows copied verbatim under <out>/raw/datasets/ (git-ignored)."""
    src = Path(src)
    pinned = (pins["core_b"].get("mix_dataset_sha256") or {}).get(str(L_SMOKE))
    got = _sha256_file(src)
    if got != pinned:
        raise SmokeError(f"{src}: sha256 {got} != pinned core_b.mix_dataset_sha256[{L_SMOKE}] {pinned!r}")
    with open(src) as fh:
        lines = [next(fh).rstrip("\n") for _ in range(FS3_LONG_ROWS * BLOCK)]
    longs = long_row_indices(lines, FS3_LONG_ROWS)
    if longs[0] != FS2_ROW:
        raise SmokeError(f"the first long row is index {longs[0]}, not index {FS2_ROW} as 024 FS2 expects")
    d = Path(out_dir) / "raw" / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    row4, long4 = d / f"mixed-{L_SMOKE}-row{FS2_ROW}.jsonl", d / f"mixed-{L_SMOKE}-long{FS3_LONG_ROWS}.jsonl"
    row4.write_text(lines[FS2_ROW] + "\n")
    long4.write_text("".join(lines[i] + "\n" for i in longs))
    return {"row4": str(row4), "long4": str(long4), "pinned": str(src), "long_rows": longs}


# --------------------------------------------------------------------- specs --

def _spec(fs, name, cond, mix, C, N, warmup, timeout_s, dataset=None, expect_lens=None) -> dict:
    short = L_SMOKE - bp.N_GEN
    if expect_lens is None:
        n_short, n_long = (N, 0) if mix == "uniform" else bp.mixed_counts(N)
        expect_lens = [short] * n_short + [bp.LONG_INPUT] * n_long
    return {"fs": fs, "name": name, "cond": cond, "L": L_SMOKE, "mix": mix, "C": C, "N": N, "warmup": warmup,
            "timeout_s": timeout_s, "dataset": dataset, "expect_lens": expect_lens}


def probe_specs(fs: str, pins: dict, datasets: dict) -> list:
    """The boots of one smoke (024), in run order."""
    to = pins["advanced"]["timeouts_s"]
    if fs == "FS1":
        return [_spec(fs, f"fs1-{c}", c, "uniform", 1, 1, 0, SMALL_TIMEOUT_S) for c in ap.CONDITIONS]
    if fs == "FS2":
        return [_spec(fs, f"fs2-{c}", c, "mixed", 1, 1, 0, SMALL_TIMEOUT_S, datasets["row4"], [bp.LONG_INPUT])
                for c in ("B0", "T")]
    if fs == "FS3":
        return [_spec(fs, f"fs3-{c}", c, "mixed", 4, 4, 0, FS3_CAP_S, datasets["long4"], [bp.LONG_INPUT] * 4)
                for c in ("G", "B0")]
    if fs == "FS4":
        return [_spec(fs, "fs4-B0-uniform", "B0", "uniform", 2, 20, 2, to[f"uniform_{L_SMOKE}"]),
                _spec(fs, "fs4-B0-mixed", "B0", "mixed", 2, 20, 2, to[f"mixed_{L_SMOKE}"], datasets["pinned"]),
                _spec(fs, "fs4-TG-mixed", "TG", "mixed", 2, 20, 2, to[f"mixed_{L_SMOKE}"], datasets["pinned"])]
    if fs == "FS5":
        return [_spec(fs, "fs5-B0-uniform", "B0", "uniform", 2, 20, 2, FS5_TIMEOUT_S)]
    raise SmokeError(f"unknown smoke {fs!r}; 024 defines {FS_NAMES}")


# ------------------------------------------------------------- plan and argv --

def pins_for_smoke(pins: dict, k3_bytes: int | None = None) -> dict:
    """A copy of the pins, K3 = k3_bytes when the smoke steps it up (024: 256 MiB steps)."""
    p = copy.deepcopy(pins)
    if k3_bytes is not None:
        if not (isinstance(k3_bytes, int) and k3_bytes > 0 and k3_bytes % ap.KV_STEP == 0):
            raise SmokeError(f"--k3-bytes {k3_bytes!r} is not a positive multiple of 256 MiB (024)")
        p["advanced"]["kv_low_bytes"] = k3_bytes
    return p


def expected_blocks(pins: dict, k3_bytes: int | None = None) -> dict:
    """num_gpu_blocks per condition: the pinned counts, or for an unmeasured / stepped K3 the lower end of the
    bracket from 248 blocks at the Stage B K (adv_plan.blocks_bounds; exactly 186 at 3 GiB)."""
    adv, sb = pins["advanced"], pins["stage_b"]
    kv = adv["kv_low_bytes"] if k3_bytes is None else k3_bytes
    k3 = adv.get("num_gpu_blocks_k3") or 0
    if not k3 or kv != adv["kv_low_bytes"]:
        k3 = ap.blocks_bounds(kv, ref_bytes=sb["kv_cache_memory_bytes"], ref_blocks=sb["num_gpu_blocks"])[0]
    return {c: k3 if c == "K3" else adv["num_gpu_blocks_k4"] for c in ap.CONDITIONS}


def smoke_plan(spec: dict, pins: dict, *, pins_sha256: str, k3_bytes: int | None = None) -> dict:
    """A one-boot, non-formal plan that run_adv.plan_problems accepts: the plan's c and timeouts_s carry the smoke's
    C and timeout, the boot keeps Core's N / warmup at that C (run_adv requires them) and smoke_counts holds the
    N / warmup the bench really runs with (the run_bench wrapper rewrites them, with_counts)."""
    adv, sb = pins["advanced"], pins["stage_b"]
    C, L, mix = spec["C"], spec["L"], spec["mix"]
    boot = {"block": 1, "cond": spec["cond"], "L": L, "mix": mix, "restart": 1, "C": C, "N": bp.num_prompts(C),
            "warmup": bp.num_warmups(C), "timeout_s": spec["timeout_s"],
            "prefix": ap.request_id_prefix(spec["cond"], L, mix, 1)}
    return {"schema_version": ap.SCHEMA_VERSION, "smoke": spec["fs"], "seed": adv["seed"],
            "derived_from": adv["derived_from"], "c": {str(L): C}, "conditions": list(ap.CONDITIONS),
            "boots": [boot], "smoke_counts": {"N": spec["N"], "warmup": spec["warmup"]},
            "expected_blocks": expected_blocks(pins, k3_bytes),
            "timeouts_s": {**adv["timeouts_s"], f"{mix}_{L}": spec["timeout_s"]}, "pins_sha256": pins_sha256,
            "gpu_memory_utilization": sb["gpu_memory_utilization"],
            "kv_cache_memory_bytes": sb["kv_cache_memory_bytes"], "block_size": sb["block_size"]}


def flag_value(argv: list, flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def result_path(argv: list) -> Path | None:
    d, f = flag_value(argv, "--result-dir"), flag_value(argv, "--result-filename")
    return Path(d) / f if d and f else None


def with_counts(argv: list, *, C: int, N: int, warmup: int) -> list:
    """The bench argv at the smoke's C / N / warmup; every other argument unchanged."""
    out = list(argv)
    for flag, value in (("--max-concurrency", C), ("--num-prompts", N), ("--num-warmups", warmup)):
        if out.count(flag) != 1:
            raise SmokeError(f"bench argv has {out.count(flag)} x {flag} (expected one)")
        out[out.index(flag) + 1] = str(value)
    return out


# --------------------------------------------------------------------- K3 rule --

def k3_report(num_gpu_blocks: int | None, *, kv_bytes: int, pins: dict) -> dict:
    """024 H-C4 K3 rule on the measured pool, and whether the next lower 256 MiB multiple provably fails."""
    bs = pins["stage_b"]["block_size"]
    c = {str(L): C for L, C in pins["advanced"]["c"].items()}
    rep = {"kv_low_bytes": kv_bytes, "num_gpu_blocks": num_gpu_blocks, "block_size": bs}
    if not isinstance(num_gpu_blocks, int) or num_gpu_blocks <= 0:
        return {**rep, "c_kv": None, "rule_ok": False, "problems": ["num_gpu_blocks not read from /metrics"],
                "lower_bytes": kv_bytes - ap.KV_STEP, "lower_blocks_max": None, "lower_fails": None,
                "next_kv_low_bytes": None, "freeze_ready": False}
    problems = ap.k3_problems(num_gpu_blocks, block_size=bs, c=c)
    lower = kv_bytes - ap.KV_STEP
    upper = ap.blocks_bounds(lower, ref_bytes=kv_bytes, ref_blocks=num_gpu_blocks)[1] if lower > 0 else 0
    lower_fails = bool(ap.k3_problems(upper, block_size=bs, c=c)) if lower > 0 else True
    return {**rep, "c_kv": {str(L): bp.c_kv(num_gpu_blocks, bs, L) for L in (*ap.LEVELS, bp.MAX_MODEL_LEN)},
            "rule_ok": not problems, "problems": problems, "lower_bytes": lower, "lower_blocks_max": upper,
            "lower_fails": lower_fails, "next_kv_low_bytes": kv_bytes + ap.KV_STEP if problems else None,
            "freeze_ready": not problems and lower_fails}


# ------------------------------------------------------------- iteration log --

def _is_iteration_line(line: str) -> bool:
    return "Iteration(" in line and "context requests" in line and "(dummy)" not in line


def step_counts(lines, *, threshold: int = ad.LONG_PREFILL_TOKEN_THRESHOLD, budget: int = ad.BUDGET) -> dict:
    """Token counts of the logged steps (never their times): which lines parse with the runner's epoch-ms stamp, the
    full-budget single-request steps (B0) and the steps above the T threshold."""
    lines = list(lines)
    steps = ad.parse_iteration_log(lines)
    return {"n_iteration_lines": sum(_is_iteration_line(x) for x in lines), "n_steps": len(steps),
            "epoch_ms_stamps": bool(steps) and all(s["t"] > 1e9 for s in steps),
            "n_ctx_steps": sum(s["ctx_req"] > 0 for s in steps),
            "n_ctx_budget_single": sum(s["ctx_req"] == 1 and s["ctx_tok"] == budget for s in steps),
            "max_step_tokens": max((s["ctx_tok"] + s["gen_tok"] for s in steps), default=None),
            "n_ctx_over_threshold": sum(s["ctx_tok"] > threshold for s in steps),
            "n_chunk_violations": sum(s["ctx_tok"] > threshold * s["ctx_req"] for s in steps)}


# ------------------------------------------------------------- bench and mask --

def count_bench_result(path, expect_lens: list) -> dict:
    """Counts from a bench --save-detailed JSON; no latency value leaves this function."""
    if path is None or not Path(path).is_file():
        return {"exists": False}
    res = json.loads(Path(path).read_text())
    lens, outs = res.get("input_lens") or [], res.get("output_lens") or []
    n_in = sum((Counter(lens) & Counter(expect_lens)).values())
    return {"exists": True, "n": len(lens), "completed": res.get("completed"), "failed": res.get("failed"),
            "n_input_ok": n_in, "inputs_ok": sorted(lens) == sorted(expect_lens),
            "n_output_ok": sum(o == bp.N_GEN for o in outs), "outputs_ok": outs == [bp.N_GEN] * len(expect_lens)}


def latency_keys(obj, path: str = "") -> list:
    """Every key (dotted path) that names a latency / throughput quantity."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if LATENCY_KEY.search(str(k)):
                out.append(p)
            out += latency_keys(v, p)
    elif isinstance(obj, list):
        for v in obj:
            out += latency_keys(v, path)
    return out


def mask_row(row: dict | None) -> dict | None:
    """A runner row reduced to counts and flags (024: 衍生檔不含任何延遲欄位)."""
    if row is None:
        return None
    out = {k: row[k] for k in ROW_KEEP if k in row}
    dec = row.get("decomp")
    out["decomp"] = {"valid": dec.get("valid"), "checks": dec.get("checks")} if isinstance(dec, dict) else None
    out["kv_peak_present"] = row.get("kv_peak") is not None
    if isinstance(row.get("spans"), dict):
        out["spans_main"] = row["spans"].get("main")
    return out


def _drop_latency(obj):
    if isinstance(obj, dict):
        return {k: _drop_latency(v) for k, v in obj.items() if not LATENCY_KEY.search(str(k))}
    if isinstance(obj, list):
        return [_drop_latency(v) for v in obj]
    return obj


def redact_bench_log(text: str) -> str:
    """A bench log without its result section and progress bars (errors before them stay readable)."""
    i = text.find(BANNER)
    if i >= 0:
        text = text[:text.rfind("\n", 0, i) + 1] + "[result section removed]\n"
    return _TQDM.sub("[progress removed]", text)


def row_field_problems(row: dict | None) -> list:
    """Contract fields of a runner row that are missing (or null where 024 needs a value)."""
    if row is None:
        return ["row"]
    out = [k for k in ROW_FIELDS if k not in row]
    out += [f"{k} is null" for k in NOT_NULL if k in row and row[k] is None]
    if isinstance(row.get("manip"), dict):
        out += [f"manip.{k}" for k in MANIP_FIELDS if k not in row["manip"]]
    if isinstance(row.get("decomp"), dict):
        out += [f"decomp.{k}" for k in DECOMP_FIELDS if k not in row["decomp"]]
        out += [f"decomp.checks.{k}" for k in "abcd" if k not in (row["decomp"].get("checks") or {})]
    return out


def _payload(path: Path) -> bytes:
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError):
            pass
    return data


def mask_session(d: Path, result_paths: list, row_extra: dict | None = None) -> dict:
    """Delete / redact / rewrite everything in a probe's session dir that carries a latency value."""
    deleted, truncated, rewritten = [], [], []
    for p in result_paths:
        if p is not None and Path(p).is_file():
            Path(p).unlink()
            deleted.append(str(Path(p).relative_to(d)) if Path(p).is_relative_to(d) else str(p))
    keep = {d / "rows.jsonl", d / "session.json", d / "plan.json", d / "raw_manifest.json"}
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        rel = p.relative_to(d)
        if p in keep or rel.parts[0] in ("derived", "env") or "server" in p.name:
            continue
        if any(m in _payload(p) for m in LATENCY_MARKERS):
            p.unlink()
            deleted.append(str(rel))
            continue
        if p.suffix in (".log", ".txt"):
            text = p.read_text(errors="replace")
            red = redact_bench_log(text)
            if red != text:
                p.write_text(red)
                truncated.append(str(rel))
    rows = d / "rows.jsonl"
    if rows.is_file():
        rows.write_text("".join(json.dumps({**mask_row(json.loads(x)), **(row_extra or {})}) + "\n"
                                for x in rows.read_text().splitlines() if x.strip()))
        rewritten.append("rows.jsonl")
    for p in sorted((d / "derived").rglob("*.jsonl")) if (d / "derived").is_dir() else []:
        p.write_text("".join(json.dumps(_drop_latency(json.loads(x))) + "\n" for x in p.read_text().splitlines()
                             if x.strip()))
        rewritten.append(str(p.relative_to(d)))
    meta = d / "session.json"  # committed: its runs lose startup_s and the bench window t_before / t_after
    if meta.is_file():
        meta.write_text(json.dumps(_drop_latency(json.loads(meta.read_text())), indent=1) + "\n")
        rewritten.append("session.json")
    left = []
    for name in rewritten:
        text = (d / name).read_text()
        docs = [json.loads(text)] if name == "session.json" else [json.loads(x) for x in text.splitlines()
                                                                   if x.strip()]
        for doc in docs:
            left += [f"{name}:{k}" for k in latency_keys(doc)]
    return {"deleted": deleted, "truncated": truncated, "rewritten": rewritten, "unmasked_fields": sorted(set(left))}


# ------------------------------------------------------------------ one probe --

def _metrics_summary(res) -> dict | None:
    from run_core_b import cache_config, gauge, parse_prometheus  # lazily: run_core_b imports tomllib (3.11+)

    if not (isinstance(res, tuple) and len(res) == 2 and res[0] == 200 and isinstance(res[1], str)):
        return None
    m = parse_prometheus(res[1])
    return {"cache_config": cache_config(m), "preemptions_total": gauge(m, "vllm:num_preemptions_total")}


def observability_problems(row: dict | None, series_samples: int) -> list:
    """024 FS4 "scrape delta、1 Hz 序列 … 都必須齊全": every histogram / counter of run_adv's scrape delta present
    with no delta error, the 1 Hz shares computed from >= 1 sample, and a series of more than 2 samples (start and
    stop alone mean the 1 Hz loop never ran). Names and counts only, never a value."""
    import run_adv as ra  # lazily, as run_adv_session_runner

    out = []
    sd = (row or {}).get("scrape_delta")
    if not isinstance(sd, dict) or not sd:
        out.append("scrape delta missing from the row")
    else:
        out += [f"scrape delta: {k} missing" for k in (*ra.HISTOGRAMS, *ra.COUNTERS) if sd.get(k) is None]
        if sd.get("errors"):
            out.append(f"scrape delta: {len(sd['errors'])} error(s) (counter reset or bucket change)")
    oh = (row or {}).get("one_hz")
    if not (isinstance(oh, dict) and (oh.get("n_samples") or 0) >= 1 and oh.get("kv_full_while_waiting") is not None
            and oh.get("gate_zone") is not None):
        out.append("1 Hz shares not computed (no sample with KV usage and the waiting count)")
    if series_samples <= 2:
        out.append(f"1 Hz series has {series_samples} sample(s): the sampler loop never ran (start / stop only)")
    return out


def _count_lines(paths) -> int:
    return sum(len([x for x in p.read_text().splitlines() if x.strip()]) for p in paths)


def _gather(d: Path) -> dict:
    rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()] \
        if (d / "rows.jsonl").is_file() else []
    row = rows[0] if len(rows) == 1 else None
    derived = sorted((d / "derived").rglob("*.jsonl")) if (d / "derived").is_dir() else []
    lines = []
    for p in sorted(d.rglob("*server*.log")):
        lines += p.read_text(errors="replace").splitlines()
    meta = json.loads((d / "session.json").read_text()) if (d / "session.json").is_file() else {}
    dec = row.get("decomp") if row else None
    series = _count_lines(sorted((d / "raw").glob("*.series.jsonl"))) if (d / "raw").is_dir() else 0
    return {"row": row, "n_rows": len(rows), "field_problems": row_field_problems(row),
            "derived": {"files": len(derived), "rows": _count_lines(derived)},
            "steps": step_counts(lines), "teardown_problems": meta.get("teardown_problems") or [],
            "aborted": meta.get("aborted"),
            "decomp_ran": isinstance(dec, dict) and dec.get("q_shares") is not None,
            "series_samples": series, "observability": observability_problems(row, series)}


def run_probe(spec: dict, *, pins: dict, out, session_runner: Callable, http_get: Callable, gpu_mem_used: Callable,
              procs_with: Callable, port_open: Callable, now: Callable = time.monotonic, sleep: Callable = time.sleep,
              pins_sha256: str, k3_bytes: int | None = None, server_port: int = SERVER_PORT,
              idle_mib: int = IDLE_MIB, release_timeout_s: float = RELEASE_TIMEOUT_S) -> dict:
    """One boot through the runner as a one-boot, non-formal session <out>/<name>/; returns the masked probe report."""
    t0, mem = now(), gpu_mem_used()
    while mem > idle_mib and now() - t0 < release_timeout_s:  # a just-torn-down server releases within seconds
        sleep(2)
        mem = gpu_mem_used()
    if mem > idle_mib:
        raise SmokeError(f"GPU busy before {spec['name']}: memory.used {mem} MiB > {idle_mib} MiB after "
                         f"{release_timeout_s} s; stopping (MACHINE_POLICY: never preempt)")
    run_pins = pins_for_smoke(pins, k3_bytes)
    plan = smoke_plan(spec, run_pins, pins_sha256=pins_sha256, k3_bytes=k3_bytes)
    d = Path(out) / spec["name"]
    health_url = f"http://127.0.0.1:{server_port}/health"
    rec = {"bench": None, "health_after_bench": None, "residual_client_procs": None, "result_paths": [],
           "metrics": []}

    def wrap_http_get(fn):
        def get(url, *a, **kw):
            res = fn(url, *a, **kw)
            if str(url).endswith("/metrics"):
                s = _metrics_summary(res)
                if s:
                    rec["metrics"].append(s)
            return res
        return get

    def wrap_run_bench(fn):
        def run(*a, **kw):
            a = list(a)
            if "argv" in kw:
                kw["argv"] = argv = with_counts(kw["argv"], C=spec["C"], N=spec["N"], warmup=spec["warmup"])
            else:
                a[0] = argv = with_counts(a[0], C=spec["C"], N=spec["N"], warmup=spec["warmup"])
            path, prefix = result_path(argv), flag_value(argv, "--request-id-prefix")
            rec["result_paths"].append(path)
            try:
                return fn(*a, **kw)
            finally:
                rec["bench"] = count_bench_result(path, spec["expect_lens"])
                try:
                    rec["health_after_bench"] = http_get(health_url)[0] == 200
                except Exception:  # noqa: BLE001 - a dead server is a finding, not a crash
                    rec["health_after_bench"] = False
                left = procs_with(prefix) if prefix else None
                rec["residual_client_procs"] = len(left) if left is not None else None
        return run

    reasons = []
    try:
        session_runner(plan, pins=run_pins, out_dir=d, session_id=f"{Path(out).name}-{spec['name']}",
                       mixed_datasets={spec["L"]: spec["dataset"]} if spec["dataset"] else {},
                       wrap_http_get=wrap_http_get, wrap_run_bench=wrap_run_bench)
    except Exception as exc:  # noqa: BLE001 - recorded; masking and the teardown checks still run
        reasons.append(f"harness error: {type(exc).__name__}: {str(exc)[:300]}")
    g = _gather(d) if d.is_dir() else {"row": None, "n_rows": 0, "field_problems": ["row"],
                                       "derived": {"files": 0, "rows": 0}, "steps": step_counts([]),
                                       "teardown_problems": [], "aborted": None, "decomp_ran": False,
                                       "series_samples": 0, "observability": observability_problems(None, 0)}
    if g["n_rows"] != 1:
        reasons.append(f"the runner wrote {g['n_rows']} rows (expected 1)")
    t0 = now()
    idle = gpu_mem_used() <= idle_mib
    while not idle and now() - t0 < release_timeout_s:
        sleep(2)
        idle = gpu_mem_used() <= idle_mib
    receiver = urlparse(run_pins["advanced"]["otlp_endpoint"])
    post = {"server_port_open": port_open("127.0.0.1", server_port),
            "receiver_port_open": port_open(receiver.hostname, receiver.port), "gpu_idle_after": idle}
    counts = {"N_bench": spec["N"], "warmup_bench": spec["warmup"]}
    masked = mask_session(d, rec["result_paths"], counts) if d.is_dir() else \
        {"deleted": [], "truncated": [], "rewritten": [], "unmasked_fields": []}
    cfgs = [m["cache_config"] for m in rec["metrics"] if m["cache_config"]]
    pre = [m["preemptions_total"] for m in rec["metrics"] if m["preemptions_total"] is not None]
    scraped = pre[-1] - pre[0] if len(pre) >= 2 else None
    row_pre = (g["row"] or {}).get("preemptions")
    preemptions = row_pre if row_pre is not None else scraped
    return {"name": spec["name"], "fs": spec["fs"], "cond": spec["cond"], "L": spec["L"], "mix": spec["mix"],
            "C": spec["C"], "N": spec["N"], "warmup": spec["warmup"], "timeout_s": spec["timeout_s"],
            "kv_cache_memory_bytes": ap.kv_cache_bytes_for(spec["cond"], run_pins),
            "expected_blocks": plan["expected_blocks"][spec["cond"]], "session_dir": spec["name"],
            "row": {**mask_row(g["row"]), **counts} if g["row"] else None, "field_problems": g["field_problems"],
            "derived": g["derived"], "decomp_ran": g["decomp_ran"], "steps": g["steps"],
            "series_samples": g["series_samples"], "observability": g["observability"],
            "bench": rec["bench"] or {"exists": False},
            "health_after_bench": rec["health_after_bench"], "residual_client_procs": rec["residual_client_procs"],
            "cache_config": cfgs[-1] if cfgs else None,
            "preemptions": int(preemptions) if preemptions is not None else None,
            "preemptions_scraped": int(scraped) if scraped is not None else None,
            "teardown_problems": g["teardown_problems"], "runner_aborted": g["aborted"], "post": post,
            "masked": masked, "reasons": reasons}


# ------------------------------------------------------------------ FS verdicts --

def _boot_problems(p: dict, tag: str) -> list:
    """Checks every probe needs: a row, the runner's own facts check, a completed bench, a live server after it."""
    out = [f"{tag}: {r}" for r in p["reasons"]]
    row = p["row"]
    if row is None:
        return out + [f"{tag}: no row"]
    if (row.get("manip") or {}).get("facts_ok") is not True:
        out.append(f"{tag}: log facts / non-default args do not match the condition table (manip.facts_ok)")
    return out


def _completion_problems(p: dict, tag: str) -> list:
    b, row = p["bench"], p["row"] or {}
    out = []
    if row.get("timeout"):
        out.append(f"{tag}: the bench hit its {p['timeout_s']} s timeout")
    if not b.get("exists"):
        return out + [f"{tag}: no bench result"]
    if not (b["completed"] == p["N"] and not b["failed"] and b["inputs_ok"] and b["outputs_ok"]):
        out.append(f"{tag}: {b['completed']} of {p['N']} requests complete ({b['failed']} failed; "
                   f"{b['n_input_ok']} / {b['n_output_ok']} with the right input / output length)")
    if p["health_after_bench"] is not True:
        out.append(f"{tag}: /health not 200 after the bench")
    return out


def _fs1(reps: list, pins: dict) -> dict:
    bs = pins["stage_b"]["block_size"]
    reasons, k3 = [], None
    for p in reps:
        c = p["cond"]
        reasons += _boot_problems(p, c) + _completion_problems(p, c)
        cfg, manip = p["cache_config"], ((p["row"] or {}).get("manip") or {})
        if cfg is None:
            reasons.append(f"{c}: /metrics vllm:cache_config_info not read")
        else:
            if cfg["block_size"] != bs:
                reasons.append(f"{c}: /metrics block_size {cfg['block_size']} != {bs}")
            if cfg["num_gpu_blocks"] != p["expected_blocks"]:
                reasons.append(f"{c}: /metrics num_gpu_blocks {cfg['num_gpu_blocks']} != expected "
                               f"{p['expected_blocks']}")
        if p["row"] is not None and manip.get("blocks_ok") is not True:
            reasons.append(f"{c}: the runner's blocks check failed (manip.blocks_ok)")
        if (((p["row"] or {}).get("decomp") or {}).get("checks") or {}).get("a") is not True:
            reasons.append(f"{c}: receiver: the main-result spans are missing or misaligned (A10 a)")
        st = p["steps"]
        if st["n_iteration_lines"] == 0:
            reasons.append(f"{c}: no iteration line in the server log")
        elif st["n_steps"] < st["n_iteration_lines"] or not st["epoch_ms_stamps"]:
            reasons.append(f"{c}: {st['n_iteration_lines'] - st['n_steps']} of {st['n_iteration_lines']} iteration "
                           "lines lack the runner's epoch-ms timestamp")
        if c in ("T", "TG") and manip.get("chunk_ok") is not True:
            reasons.append(f"{c}: chunk check (ctx_tokens <= {ad.LONG_PREFILL_TOKEN_THRESHOLD} x ctx_requests) "
                           "did not pass")
        if c == "K3":
            measured = (cfg or {}).get("num_gpu_blocks", manip.get("num_gpu_blocks"))
            k3 = k3_report(measured, kv_bytes=p["kv_cache_memory_bytes"], pins=pins)
            if not k3["rule_ok"]:
                reasons.append(f"K3: {'; '.join(k3['problems'])}; next step kv_low_bytes = {k3['next_kv_low_bytes']}")
            elif not k3["lower_fails"]:
                reasons.append(f"K3: {k3['lower_bytes']} B may already pass the rule (<= {k3['lower_blocks_max']} "
                               "blocks): not the smallest 256 MiB multiple")
    return {"ok": not reasons, "reasons": reasons, "k3": k3}


def _fs2(reps: list, pins: dict) -> dict:
    t, budget = pins["advanced"]["long_prefill_token_threshold"], bp.MAX_NUM_BATCHED_TOKENS
    reasons = []
    for p in reps:
        c, st = p["cond"], p["steps"]
        reasons += _boot_problems(p, c) + _completion_problems(p, c)
        if c == "B0":
            if st["n_ctx_budget_single"] < 1:
                reasons.append(f"B0: no step with 1 context request and {budget} context tokens")
            if st["max_step_tokens"] != budget:
                reasons.append(f"B0: largest step {st['max_step_tokens']} tokens != the {budget} budget (B)")
        else:
            if st["n_ctx_steps"] < 1:
                reasons.append(f"{c}: no context step in the server log")
            if st["n_ctx_over_threshold"]:
                reasons.append(f"{c}: {st['n_ctx_over_threshold']} steps with more than {t} context tokens")
            if ((p["row"] or {}).get("manip") or {}).get("chunk_ok") is not True:
                reasons.append(f"{c}: the runner's chunk check did not pass")
    return {"ok": not reasons, "reasons": reasons}


def _fs3(reps: list, pins: dict) -> dict:
    reasons = []
    for p in reps:
        c, n = p["cond"], p["preemptions"]
        reasons += _boot_problems(p, c) + _completion_problems(p, c)
        if c == "G" and not (n or 0) > 0:
            reasons.append(f"G: {'no preemption' if n is not None else 'preemptions not measured'} "
                           "(024 expects over-commit: 4 x 66 > 247 blocks)")
        if c == "B0" and n != 0:
            reasons.append(f"B0: {n} preemption(s); the full-ISL gate should prevent over-commit")
    return {"ok": not reasons, "reasons": reasons}


def _fs4(reps: list, pins: dict) -> dict:
    reasons = []
    for p in reps:
        tag = p["name"]
        reasons += _boot_problems(p, tag) + _completion_problems(p, tag)
        if p["cond"] != "B0":
            continue  # TG mixed only has to complete (024)
        if p["field_problems"]:
            reasons.append(f"{tag}: row fields missing or null: {', '.join(p['field_problems'])}")
        reasons += [f"{tag}: {x}" for x in p["observability"]]
        if p["derived"]["rows"] != p["N"]:
            reasons.append(f"{tag}: derived/ holds {p['derived']['rows']} request rows (expected N = {p['N']})")
        checks = ((p["row"] or {}).get("decomp") or {}).get("checks") or {}
        reasons += [f"{tag}: A10 ({k}) failed" for k in "abd" if checks.get(k) is not True]
        if checks.get("c") is False:
            reasons.append(f"{tag}: A10 (c) failed")
        if not p["decomp_ran"]:
            reasons.append(f"{tag}: the step classifier produced no tail shares")
        if p["masked"]["unmasked_fields"]:
            reasons.append(f"{tag}: latency fields left after masking: {p['masked']['unmasked_fields']}")
    return {"ok": not reasons, "reasons": reasons}


def _fs5(reps: list, pins: dict) -> dict:
    (p,) = reps
    row, post = p["row"], p["post"]
    reasons = [f"harness: {r}" for r in p["reasons"]]
    if row is None:
        reasons.append("no row")
    elif not (row.get("ok") is False and row.get("timeout") is True and
              any("timeout" in str(r).lower() for r in row.get("reasons") or [])):
        reasons.append(f"row is not a timeout row (ok={row.get('ok')!r}, timeout={row.get('timeout')!r})")
    if p["residual_client_procs"] != 0:
        reasons.append(f"client process group not killed: {p['residual_client_procs']!r} process(es) left")
    if p["health_after_bench"] is not True:
        reasons.append("server not healthy after the killed bench")
    if post["server_port_open"]:
        reasons.append(f"server port {SERVER_PORT} still open after teardown")
    if post["receiver_port_open"]:
        reasons.append("receiver port still open after teardown")
    if not post["gpu_idle_after"]:
        reasons.append("GPU memory not released after teardown")
    if p["teardown_problems"]:
        reasons.append(f"teardown: {p['teardown_problems']}")
    return {"ok": not reasons, "reasons": reasons}


_EVAL = {"FS1": _fs1, "FS2": _fs2, "FS3": _fs3, "FS4": _fs4, "FS5": _fs5}


def evaluate(fs: str, reps: list, *, pins: dict) -> dict:
    return _EVAL[fs](reps, pins)


# ------------------------------------------------------------------ the smoke --

def run_smoke(out, *, fs, pins: dict, pins_sha256: str, mixed_dataset, session_runner: Callable, http_get: Callable,
              gpu_mem_used: Callable, procs_with: Callable, port_open: Callable, now: Callable = time.monotonic,
              sleep: Callable = time.sleep, k3_bytes: int | None = None, server_port: int = SERVER_PORT,
              clock: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat()) -> dict:
    """FS in 024 order; smoke.json is rewritten after each FS, also when the smoke stops (GPU busy)."""
    out = Path(out)
    if out.exists():
        raise SmokeError(f"{out} exists; smoke evidence is never overwritten (use a new session dir)")
    fs = [f for f in FS_NAMES if f in fs]
    out.mkdir(parents=True)
    report = {"schema_version": SCHEMA_VERSION, "session_id": out.name, "formal": False, "gpu_lock_held": True,
              "pins_sha256": pins_sha256, "kv_low_bytes": k3_bytes or pins["advanced"]["kv_low_bytes"],
              "started_at": clock(), "datasets": None, "fs": {}, "aborted": None, "ok": False}

    def save():
        report["ok"] = report["aborted"] is None and bool(report["fs"]) and all(v["ok"] for v in report["fs"].values())
        (out / "smoke.json").write_text(json.dumps(report, indent=1) + "\n")

    try:
        datasets = {}
        if any(f in fs for f in ("FS2", "FS3", "FS4")):
            datasets = prepare_datasets(mixed_dataset, out, pins)
            report["datasets"] = {"pinned_sha256": pins["core_b"]["mix_dataset_sha256"][str(L_SMOKE)],
                                  "long_rows": datasets["long_rows"]}
        for name in fs:
            reps = [run_probe(s, pins=pins, out=out, session_runner=session_runner, http_get=http_get,
                              gpu_mem_used=gpu_mem_used, procs_with=procs_with, port_open=port_open, now=now,
                              sleep=sleep, pins_sha256=pins_sha256, k3_bytes=k3_bytes, server_port=server_port)
                    for s in probe_specs(name, pins, datasets)]
            report["fs"][name] = {**evaluate(name, reps, pins=pins), "probes": reps}
            save()
    except SmokeError as exc:
        report["aborted"] = str(exc)
        raise
    finally:
        report["finished_at"] = clock()
        save()
    return report


# ----------------------------------------------------------- real backends --

def procs_with(token: str) -> list | None:
    """PIDs (other than this one) whose /proc cmdline contains token (WSL / Linux); None without /proc, so the FS5
    process-group check fails instead of passing unchecked."""
    if not Path("/proc").is_dir():
        return None
    out = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if token in cmd and int(p.name) != os.getpid():
            out.append(int(p.name))
    return out


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def run_adv_session_runner(*, gdn_prefill_backend: str, gpu_memory_utilization: str, vllm_bin: str,
                           root: Path | None = None, **backends) -> Callable:
    """The runner the smoke drives: run_adv.run_session, non-formal, under the held lock (verified by run_adv at the
    session start and before the boot), wired as run_adv's own CLI wires it (start_server, default_run_bench,
    run_core_b.default_http_get, nvidia-smi with a timeout, the lock probe, env_capture); backends replace any of
    those (tests). The smoke's wrappers go around http_get and run_bench."""
    import run_adv as ra

    def runner(plan, *, pins, out_dir, session_id, mixed_datasets, wrap_http_get, wrap_run_bench):
        kw = backends
        if "popen" not in backends:
            import run_core_b as rb

            kw = {"popen": ra.start_server, "signal_group": smoke.signal_group, "group_alive": smoke.group_alive,
                  "http_get": rb.default_http_get, "run_bench": ra.default_run_bench,
                  "gpu_mem_used": ra.nvidia_mem_used_bounded, "gpu_lock_free": smoke.gpu_lock_free,
                  "lock_held_check": ra.held_gpu_lock,
                  "capture_env": rb.make_vllm_env_capture(Path(root) / "scripts" / "env_capture.py"), **backends}
        kw = {**kw, "http_get": wrap_http_get(kw["http_get"]), "run_bench": wrap_run_bench(kw["run_bench"])}
        return ra.run_session(plan, pins=pins, pins_sha256=plan["pins_sha256"], out_dir=out_dir,
                              gdn_prefill_backend=gdn_prefill_backend, gpu_memory_utilization=gpu_memory_utilization,
                              session_id=session_id, gpu_lock_held=True, formal=False, vllm_bin=vllm_bin,
                              mixed_datasets=mixed_datasets, **kw)

    return runner


def main(argv: list | None = None, *, lock_held_check: Callable | None = None) -> int:
    """lock_held_check (tests inject it): run_adv.held_gpu_lock, the flag alone is never trusted."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="new smoke session dir, e.g. results/advanced/smoke/<session>")
    parser.add_argument("--mixed-dataset", default="data/core_b/mixed-2048.jsonl",
                        help="the pinned mixed-2048 file (FS2-FS4)")
    parser.add_argument("--only", default=",".join(FS_NAMES), help="comma-separated subset of FS1-FS5")
    parser.add_argument("--k3-bytes", type=int, default=None,
                        help="K3 for FS1 when the 3 GiB pool fails the 024 rule (256 MiB steps)")
    parser.add_argument("--vllm-bin", default=".venv/bin/vllm")
    parser.add_argument("--gpu-lock-held", action="store_true",
                        help="required: run under flock -n /tmp/ai-portfolio-gpu.lock")
    args = parser.parse_args(argv)
    fs = [x.strip() for x in args.only.split(",") if x.strip()]
    if not fs or any(f not in FS_NAMES for f in fs):
        parser.error(f"--only takes a subset of {FS_NAMES}")
    if not args.gpu_lock_held:
        print(f"adv_smoke: smokes run on the GPU only under the shared lock: flock -n {smoke.GPU_LOCK} ... "
              "--gpu-lock-held (MACHINE_POLICY; 023)", file=sys.stderr)
        return 2
    import run_adv as ra

    if not (lock_held_check or ra.held_gpu_lock)():
        print(f"adv_smoke: --gpu-lock-held, but this process does not hold {smoke.GPU_LOCK} (no locked descriptor "
              f"on it): run it as flock -n {smoke.GPU_LOCK} .venv/bin/python benchmarks/adv_smoke.py ... "
              "(MACHINE_POLICY; 023)", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists():
        print(f"adv_smoke: {out} exists; smoke evidence is never overwritten", file=sys.stderr)
        return 2
    import tomllib

    root = Path(__file__).resolve().parent.parent
    raw = (root / "configs" / "pins.toml").read_bytes()
    pins = tomllib.loads(raw.decode())
    problems = ap.pins_problems(pins)
    if problems:
        print("adv_smoke: pins [advanced]: " + "; ".join(problems), file=sys.stderr)
        return 2
    sb = pins["stage_b"]
    runner = run_adv_session_runner(gdn_prefill_backend=sb["gdn_prefill_backend"],
                                    gpu_memory_utilization=sb["gpu_memory_utilization"], vllm_bin=args.vllm_bin,
                                    root=root)
    import run_core_b as rb

    for sig in (smoke.signal.SIGHUP, smoke.signal.SIGTERM):
        smoke.signal.signal(sig, smoke._exit_on_signal)
    try:
        report = run_smoke(out, fs=fs, pins=pins, pins_sha256=hashlib.sha256(raw).hexdigest(),
                           mixed_dataset=args.mixed_dataset, session_runner=runner, http_get=rb.default_http_get,
                           gpu_mem_used=ra.nvidia_mem_used_bounded, procs_with=procs_with, port_open=port_open,
                           k3_bytes=args.k3_bytes, server_port=ra.PORT)
    except SmokeError as exc:
        print(f"adv_smoke: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": report["ok"], **{f: {"ok": v["ok"], "reasons": v["reasons"]}
                                             for f, v in report["fs"].items()},
                      "k3": (report["fs"].get("FS1") or {}).get("k3")}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
