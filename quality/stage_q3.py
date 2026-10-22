#!/usr/bin/env python3
"""Stage Q3 (decision 040): the quality of Core B's W4A16 checkpoint against its BF16 base in one runtime (vLLM
0.30.0), on WikiText-2 test cut and scored as llama-perplexity cuts and scores it (005).

- chunks (WSL, project venv, offline HF cache): the pinned tokenizer encodes wiki.test.raw without special tokens;
  floor(n / 512) chunks of 512 ids in order (llama-perplexity: tokens // n_ctx; Qwen3.5 adds no BOS) are written as
  {"i", "ids"} JSONL. Its sha256 and chunk count are pinned in configs/stage_q3.toml before the freeze.
- Scoring: each chunk is one prompt (TokensPrompt) with SamplingParams(max_tokens 1, temperature 0, prompt_logprobs
  1). The scored tokens are positions FIRST + 1 .. N_CTX - 1 (llama-perplexity scores logits [n_ctx / 2, n_ctx - 1),
  255 tokens per chunk); for each, the logprob of the actual token and the id of the rank-1 token.
- plan: two jobs, W4A16 (pins core_b) and BF16 (pins core_a 9B source, cpu_offload_gb from the config), with Core B's
  engine settings (language model only, bfloat16, pins stage_b GPU memory utilization and GDN prefill backend, no
  prefix caching); a pilot scores PILOT_CHUNKS chunks, a formal plan every chunk and needs freeze.stage_q3.
- run (WSL, GPU lock): per job, the 031 idle window, env capture, one child process (`score`) in the vLLM venv, env
  capture, and the output checked against the chunks. Token-level scores stay in raw/ (the top-1 ids can echo the
  text, 005: no item text in git); each row commits the per-chunk NLL. An OOM is outcome oom (040: NT, no rerun with
  other settings).
- analyze: PPL of each checkpoint, the paired per-chunk delta NLL (W4A16 - BF16) with a chunk bootstrap
  (quality_analysis's frozen helpers), top-1 agreement; H-Q3: PPL(W4A16) / PPL(BF16) - 1 <= 2 %.

Usage:
    .venv/bin/python quality/stage_q3.py chunks --tokenizer <snapshot dir> --text data/datasets/wiki.test.raw \\
        --out data/stage_q/q3-chunks-512.jsonl                                           # WSL
    python quality/stage_q3.py plan --mode pilot --out plan.json
    flock -n /tmp/ai-portfolio-gpu.lock .venv/bin/python quality/stage_q3.py run --plan plan.json --data data \\
        --out results/stage_q/<session> --gpu-lock-held                                  # WSL
    python quality/stage_q3.py analyze --session results/stage_q/<session>             # Mac, with raw/ fetched
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_idle as gi  # noqa: E402
import quality_analysis as qa  # noqa: E402
from run_core_a import env_problems, provenance_problems, sha256_json, write_raw_manifest  # noqa: E402

SCHEMA_VERSION = 1
STAGE, PART, PLATFORM = "Q", "q3", "wsl2-cuda"
N_CTX = 512
FIRST = N_CTX // 2                                       # llama-perplexity: first = n_ctx / 2
SCORED = N_CTX - FIRST - 1                               # 255 tokens per chunk
PILOT_CHUNKS = 10
MAX_MODEL_LEN = 1024                                     # a 512-token chunk and the one generated token
MAX_NUM_SEQS = 4
H_Q3_MAX = 0.02                                          # 040: PPL(W4A16) / PPL(BF16) - 1 <= 2 %
CHUNKS_PATH = "{data}/stage_q/q3-chunks-512.jsonl"
CONFIG = ROOT / "configs" / "stage_q3.toml"
HARNESS_FILES = ("quality/stage_q3.py",)
RESULTS = ("results", "stage_q")
JOBS = ("w4a16", "bf16")
CONFIG_KEYS = {"schema_version": None, "freeze": {"stage_q3", "pilot_q3"},
               "stage_q3": {"chunks_sha256", "chunks", "bf16_cpu_offload_gb", "max_num_batched_tokens", "pin_memory",
                            "n_boot", "seed"}}
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")
CHILD_ENV_DROP = ("VLLM_", "CAPSERVE_", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES")


class StageQ3Error(RuntimeError):
    pass


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harness_sha256(root=ROOT) -> str:
    """sha256 over HARNESS_FILES (path and bytes): the harness a pilot or a formal session ran."""
    h = hashlib.sha256()
    for rel in HARNESS_FILES:
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------- config and freeze

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def config_problems(cfg: dict) -> list:
    """configs/stage_q3.toml: exactly its keys and types; once frozen, the pilot and the chunk file pinned."""
    if set(cfg) != set(CONFIG_KEYS):
        return [f"top-level keys {sorted(cfg)} are not {sorted(CONFIG_KEYS)}"]
    problems = []
    if cfg["schema_version"] != 1:
        problems.append(f"schema_version {cfg['schema_version']!r} is not 1")
    for table, keys in CONFIG_KEYS.items():
        if keys is not None and set(cfg[table]) != keys:
            problems.append(f"[{table}] keys {sorted(cfg[table])} are not {sorted(keys)}")
    if problems:
        return problems
    f, q = cfg["freeze"], cfg["stage_q3"]
    if not isinstance(f["stage_q3"], bool) or not isinstance(f["pilot_q3"], str):
        problems.append("freeze.stage_q3 must be a bool and freeze.pilot_q3 a string")
    if not isinstance(q["chunks_sha256"], str) or not _is_int(q["chunks"]) or q["chunks"] < 0:
        problems.append("stage_q3.chunks_sha256 must be a string and stage_q3.chunks a count")
    if f["stage_q3"] is True:
        if not f["pilot_q3"]:
            problems.append("freeze.pilot_q3 must name the passed pilot once freeze.stage_q3 is true")
        if len(str(q["chunks_sha256"])) != 64 or not q["chunks"]:
            problems.append("stage_q3.chunks_sha256 / chunks must be pinned once freeze.stage_q3 is true")
    if not isinstance(q["bf16_cpu_offload_gb"], (int, float)) or isinstance(q["bf16_cpu_offload_gb"], bool) \
            or q["bf16_cpu_offload_gb"] < 0:
        problems.append("stage_q3.bf16_cpu_offload_gb must be a number >= 0")
    if not _is_int(q["max_num_batched_tokens"]) or q["max_num_batched_tokens"] < N_CTX:
        problems.append(f"stage_q3.max_num_batched_tokens must be an int >= {N_CTX}")
    if q["pin_memory"] not in ("0", "1"):
        problems.append("stage_q3.pin_memory must be \"0\" or \"1\"")
    if not _is_int(q["n_boot"]) or q["n_boot"] < 1 or not _is_int(q["seed"]):
        problems.append("stage_q3.n_boot must be a positive int and stage_q3.seed an int")
    return problems


def freeze_evidence_problems(cfg: dict, root) -> list:
    """Once freeze.stage_q3 is true: pilot_q3 is a q3 pilot under root/results/stage_q with both jobs ok and the
    current harness_sha256. [] before the freeze."""
    freeze = cfg.get("freeze") or {}
    if freeze.get("stage_q3") is not True:
        return []
    sid = freeze.get("pilot_q3")
    path = Path(root).joinpath(*RESULTS, str(sid), "session.json")
    try:
        s = json.loads(path.read_text())
    except (OSError, ValueError):
        return [f"freeze.pilot_q3: no readable {path}"]
    out = s.get("outcomes") or {}
    if (s.get("mode"), s.get("part")) != ("pilot", PART) or out.get("ok") != len(JOBS) or out.get("oom") or \
            out.get("error"):
        return [f"freeze.pilot_q3: {sid} is not a passed q3 pilot (both jobs ok)"]
    if s.get("harness_sha256") != harness_sha256():
        return [f"freeze.pilot_q3: {sid} ran another harness (harness_sha256 {s.get('harness_sha256')!r})"]
    return []


# ---------------------------------------------------------------- chunks and scores

def cut_chunks(ids: list, n_ctx: int = N_CTX) -> list:
    """llama-perplexity's chunks: floor(len / n_ctx) consecutive windows; the remainder is not scored."""
    return [list(ids[i * n_ctx:(i + 1) * n_ctx]) for i in range(len(ids) // n_ctx)]


def build_chunks(text: str, encode: Callable, out) -> dict:
    ids = list(encode(text))
    chunks = cut_chunks(ids)
    data = "".join(json.dumps({"i": i, "ids": c}) + "\n" for i, c in enumerate(chunks)).encode()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    return {"tokens": len(ids), "chunks": len(chunks), "sha256": hashlib.sha256(data).hexdigest(), "path": str(out)}


def load_jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def scored_positions() -> range:
    return range(FIRST + 1, N_CTX)


def _field(entry, name):
    return entry[name] if isinstance(entry, dict) else getattr(entry, name)


def scored(ids: list, prompt_logprobs: list) -> dict:
    """vLLM prompt logprobs (per position: {token id (int or str): Logprob or {"logprob", "rank"}}) -> the actual
    token's logprob and the rank-1 id at every scored position."""
    logprobs, top1 = [], []
    for p in scored_positions():
        d = prompt_logprobs[p] or {}
        entry = d.get(ids[p], d.get(str(ids[p])))
        if entry is None:
            raise StageQ3Error(f"position {p}: no logprob for the prompt token {ids[p]}")
        best = [int(k) for k, v in d.items() if _field(v, "rank") == 1]
        if len(best) != 1:
            raise StageQ3Error(f"position {p}: {len(best)} rank-1 tokens")
        logprobs.append(float(_field(entry, "logprob")))
        top1.append(best[0])
    return {"logprobs": logprobs, "top1": top1}


def output_problems(rows: list, chunks: list, n: int) -> list:
    """A score file: n rows, row k for chunk k, SCORED finite logprobs <= 0 and SCORED integer ids each."""
    problems = [] if len(rows) == n else [f"{len(rows)} scored chunks, {n} expected"]
    for k, (r, c) in enumerate(zip(rows, chunks[:n])):
        lp, top = r.get("logprobs"), r.get("top1")
        if r.get("i") != c["i"] or c["i"] != k:
            problems.append(f"row {k}: chunk {r.get('i')!r}, not {k}")
        if not isinstance(lp, list) or len(lp) != SCORED or not all(
                isinstance(x, (int, float)) and math.isfinite(x) and x <= 0 for x in lp):
            problems.append(f"row {k}: logprobs are not {SCORED} finite values <= 0")
        if not isinstance(top, list) or len(top) != SCORED or not all(_is_int(x) for x in top):
            problems.append(f"row {k}: top1 is not {SCORED} token ids")
        if len(problems) >= 10:
            break
    return problems


def chunk_nll(rows: list) -> list:
    return [-sum(r["logprobs"]) / len(r["logprobs"]) for r in rows]


# ---------------------------------------------------------------- plans

def model_spec(pins: dict, which: str) -> dict:
    if which == "w4a16":
        return {"repo": pins["core_b"]["repo"], "revision": pins["core_b"]["revision"]}
    m = next(m for m in pins["core_a"]["models"] if m["size"] == "9B")
    return {"repo": m["source_repo"], "revision": m["source_revision"]}


def engine(pins: dict, cfg: dict, which: str) -> dict:
    """Core B's engine (core_b_plan.server_argv) where it bears on the numbers; the sizes fit a 512-token chunk."""
    sb, cb, q = pins["stage_b"], pins["core_b"], cfg["stage_q3"]
    return {"language_model_only": True, "dtype": cb["dtype"], "kv_cache_dtype": cb["kv_cache_dtype"],
            "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": q["max_num_batched_tokens"],
            "gpu_memory_utilization": float(sb["gpu_memory_utilization"]), "enable_prefix_caching": False,
            "gdn_prefill_backend": sb["gdn_prefill_backend"], "seed": 0,
            "cpu_offload_gb": q["bf16_cpu_offload_gb"] if which == "bf16" else 0}


def build_plan(pins: dict, cfg: dict, *, mode: str, pins_sha256: str | None = None,
               stage_q3_sha256: str | None = None) -> dict:
    if mode not in ("pilot", "formal"):
        raise StageQ3Error(f"unknown mode {mode!r}")
    problems = config_problems(cfg)
    if mode == "formal" and (problems or cfg["freeze"]["stage_q3"] is not True):
        raise StageQ3Error("formal Q3 plans need freeze.stage_q3 = true in configs/stage_q3.toml (040 凍結)"
                           + (f": {'; '.join(problems)}" if problems else ""))
    if problems:
        raise StageQ3Error("configs/stage_q3.toml: " + "; ".join(problems))
    formal = mode == "formal"
    n = cfg["stage_q3"]["chunks"] if formal else PILOT_CHUNKS
    env = {"HF_HUB_OFFLINE": "1", "VLLM_WSL2_ENABLE_PIN_MEMORY": cfg["stage_q3"]["pin_memory"]}
    jobs = [{"id": f"score-{w}", "which": w, "model": model_spec(pins, w), "engine": engine(pins, cfg, w),
             "n_chunks": n, "env": env} for w in JOBS]
    return {"schema_version": SCHEMA_VERSION, "stage": STAGE, "part": PART, "mode": mode, "platform": PLATFORM,
            "pins_sha256": pins_sha256, "stage_q3_sha256": stage_q3_sha256, "harness_sha256": harness_sha256(),
            "chunks": {"path": CHUNKS_PATH, "sha256": cfg["stage_q3"]["chunks_sha256"] if formal else None,
                       "chunks": cfg["stage_q3"]["chunks"] if formal else None},
            "n_ctx": N_CTX, "first": FIRST, "scored": SCORED, "jobs": jobs}


# ---------------------------------------------------------------- the runner (WSL)

def default_run_proc(argv: list, out, err, timeout_s: float, env: dict) -> tuple:
    t0 = time.monotonic()
    with open(out, "w") as fo, open(err, "w") as fe:
        try:
            code = subprocess.run(argv, stdout=fo, stderr=fe, env=env, timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            code = -9
    return code, round(time.monotonic() - t0, 3)


def child_env(job: dict, base: dict | None = None) -> dict:
    base = dict(os.environ if base is None else base)
    env = {k: v for k, v in base.items() if not k.startswith(CHILD_ENV_DROP)}
    return {**env, **job["env"]}


def default_wait_idle(pins: dict) -> Callable:
    bound = gi.headroom_bound(pins["stage_b"]["v_total_mib"])
    return lambda: gi.wait_idle(gi.nvidia_sample, gi.nvidia_compute_apps, bound_mib=bound)


def output_dir_problems(out_dir, *, repo_root=ROOT) -> list:
    root = Path(repo_root).resolve().joinpath(*RESULTS)
    out = Path(out_dir).resolve()
    if out == root or root not in out.parents:
        return [f"{out_dir} is not a session directory under results/stage_q/ (040)"]
    return []


def run_q3(plan: dict, *, data_dir, out_dir, capture_env: Callable, session_id: str, gpu_lock_held: bool,
           wait_idle: Callable, repo_root=ROOT, run_proc: Callable = default_run_proc,
           vllm_python: str = ".venv/bin/python", timeout_s: float = 4 * 3600,
           now: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat()) -> dict:
    """One Q3 session: both score jobs in order under the GPU lock."""
    if (plan.get("stage"), plan.get("part"), plan.get("platform")) != (STAGE, PART, PLATFORM):
        raise StageQ3Error(f"not a Q3 plan ({plan.get('stage')!r}, {plan.get('part')!r}, {plan.get('platform')!r})")
    problems = output_dir_problems(out_dir, repo_root=repo_root)
    if problems:
        raise StageQ3Error("; ".join(problems))
    if not gpu_lock_held:
        raise StageQ3Error("Q3 runs vLLM: hold the GPU lock (flock -n /tmp/ai-portfolio-gpu.lock ... --gpu-lock-held)")
    if plan["mode"] == "formal" and plan.get("harness_sha256") != harness_sha256():
        raise StageQ3Error("the formal plan was built by another harness (harness_sha256); rebuild it with this one")
    out = Path(out_dir)
    if out.exists():
        raise StageQ3Error(f"{out} exists; sessions are never resumed or overwritten")
    chunks_path = Path(plan["chunks"]["path"].replace("{data}", str(data_dir)))
    try:
        chunks_sha = file_sha256(chunks_path)
        chunks = load_jsonl(chunks_path)
    except (OSError, ValueError) as exc:
        raise StageQ3Error(f"no readable chunks file {chunks_path}: {exc}") from exc
    if plan["chunks"]["sha256"] is not None and chunks_sha != plan["chunks"]["sha256"]:
        raise StageQ3Error(f"the chunks file {chunks_path} is not the pinned one (sha256 {chunks_sha})")
    if any(len(c["ids"]) != N_CTX for c in chunks) or len(chunks) < max(j["n_chunks"] for j in plan["jobs"]):
        raise StageQ3Error(f"the chunks file {chunks_path} does not hold the planned {N_CTX}-token chunks")
    for sub in ("raw", "env"):
        (out / sub).mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=gpu_lock_held)
    formal = plan["mode"] == "formal"
    if formal:
        bad = env_problems(pre, PLATFORM) + provenance_problems(pre, pre)
        if bad:
            raise StageQ3Error("formal Q3 run refused before it starts: " + "; ".join(bad))
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": STAGE, "part": PART,
               "mode": plan["mode"], "platform": PLATFORM, "plan_sha256": sha256_json(plan),
               "pins_sha256": plan.get("pins_sha256"), "gpu_lock_held": gpu_lock_held,
               "harness_sha256": harness_sha256(), "chunks_sha256": chunks_sha, "started_at": now()}
    rows, env_flags = [], []
    with open(out / "rows.jsonl", "w") as fh:
        for job in plan["jobs"]:
            jid = job["id"]
            rel = {"job": f"raw/{jid}.job.json", "scores": f"raw/{jid}.scores.jsonl", "stdout": f"raw/{jid}.stdout.txt",
                   "stderr": f"raw/{jid}.stderr.txt", "env_before": f"env/{jid}.before.json",
                   "env_after": f"env/{jid}.after.json"}
            base = {"job_id": jid, "kind": "score", "which": job["which"], "model": job["model"],
                    "n_chunks": job["n_chunks"], "chunks_sha256": chunks_sha}
            idle, report = wait_idle()
            if not idle:
                row = {**base, "outcome": "error", "reason": f"gpu_not_idle (031): {report}"}
            else:
                (out / rel["job"]).write_text(json.dumps(job, indent=1) + "\n")
                env_before = capture_env(out_path=out / rel["env_before"], gpu_lock_held=gpu_lock_held)
                argv = [vllm_python, str(Path(repo_root) / "quality" / "stage_q3.py"), "score", "--job",
                        str(out / rel["job"]), "--chunks", str(chunks_path), "--out", str(out / rel["scores"])]
                code, wall = run_proc(argv, out / rel["stdout"], out / rel["stderr"], timeout_s, child_env(job))
                env_after = capture_env(out_path=out / rel["env_after"], gpu_lock_held=gpu_lock_held)
                if formal:
                    for phase, rec in (("before", env_before), ("after", env_after)):
                        env_flags.extend({"job_id": jid, "phase": phase, "reason": why}
                                         for why in env_problems(rec, PLATFORM) + provenance_problems(rec, pre))
                text = (out / rel["stdout"]).read_text(errors="replace") + (out / rel["stderr"]).read_text(
                    errors="replace")
                scores = load_jsonl(out / rel["scores"]) if code == 0 and (out / rel["scores"]).is_file() else None
                bad = output_problems(scores, chunks, job["n_chunks"]) if scores is not None else None
                if scores is not None and not bad:
                    row = {**base, "outcome": "ok", "reason": None, "chunk_nll": chunk_nll(scores),
                           "n_tokens": SCORED * len(scores)}
                elif any(m in text for m in OOM_MARKERS):
                    row = {**base, "outcome": "oom", "reason": "out of memory (040: NT, no rerun with other settings)"}
                else:
                    row = {**base, "outcome": "error",
                           "reason": "; ".join(bad) if bad else f"exit {code} without a score file"}
                row.update(argv=argv, exit_code=code, wall_s=wall, **rel)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    outcomes = {"ok": 0, "oom": 0, "error": 0}
    for r in rows:
        outcomes[r["outcome"]] += 1
    session.update({"finished_at": now(), "outcomes": outcomes, "env_flags": env_flags, "env_valid": not env_flags})
    write_raw_manifest(out)
    (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    return session


def score(job: dict, chunks_path, out_path) -> None:  # pragma: no cover - needs vLLM and the GPU (WSL)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    chunks = load_jsonl(chunks_path)[: job["n_chunks"]]
    llm = LLM(model=job["model"]["repo"], revision=job["model"]["revision"], **job["engine"])
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outs = llm.generate([TokensPrompt(prompt_token_ids=c["ids"]) for c in chunks], params, use_tqdm=False)
    with open(out_path, "w") as fh:
        for c, o in zip(chunks, outs):
            fh.write(json.dumps({"i": c["i"], **scored(c["ids"], o.prompt_logprobs)}) + "\n")


# ---------------------------------------------------------------- analysis

def analyze(w4a16: list | None, bf16: list | None, *, n_boot: int = qa.N_BOOT, seed: int = qa.SEED) -> dict:
    """Score rows of both checkpoints over the same chunks -> PPL, paired delta NLL, top-1 agreement and H-Q3."""
    if w4a16 is None or bf16 is None:
        return {"H-Q3": {"verdict": "NT", "reason": "a checkpoint has no ok score (oom or error)",
                         "prediction": "supported"}}
    if [r["i"] for r in w4a16] != [r["i"] for r in bf16]:
        raise StageQ3Error("the two checkpoints were not scored on the same chunks")
    nw, nb = chunk_nll(w4a16), chunk_nll(bf16)
    tokens = sum(len(r["logprobs"]) for r in bf16)
    ppl = {"w4a16": math.exp(-sum(sum(r["logprobs"]) for r in w4a16) / tokens),
           "bf16": math.exp(-sum(sum(r["logprobs"]) for r in bf16) / tokens)}
    delta = [a - b for a, b in zip(nw, nb)]
    boots = sorted(qa._bootstrap_means(delta, n_boot, seed))
    agree = sum(x == y for a, b in zip(w4a16, bf16) for x, y in zip(a["top1"], b["top1"]))
    ratio = ppl["w4a16"] / ppl["bf16"] - 1
    return {"ppl": ppl, "ppl_ratio_minus_1": ratio, "tokens": tokens,
            "delta_nll": {"mean": sum(delta) / len(delta), "ci95": [qa._percentile(boots, 0.025),
                                                                     qa._percentile(boots, 0.975)],
                          "n_chunks": len(delta), "n_boot": n_boot, "seed": seed},
            "top1_agreement": agree / tokens,
            "H-Q3": {"verdict": "supported" if ratio <= H_Q3_MAX else "not supported", "bound": H_Q3_MAX,
                     "prediction": "supported"}}


def session_problems(path, *, cfg: dict, stage_q3_sha256: str, pins_sha256: str) -> list:
    """A formal, env-valid q3 session of the frozen harness, configs/stage_q3.toml, pins.toml and pinned chunks."""
    path = Path(path)
    try:
        meta = json.loads((path / "session.json").read_text())
        plan = json.loads((path / "plan.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"{path}: no readable session.json / plan.json ({exc})"]
    harness, pinned = harness_sha256(), cfg["stage_q3"]["chunks_sha256"]
    problems = []
    if meta.get("mode") != "formal" or plan.get("mode") != "formal":
        problems.append(f"{path.name}: mode {meta.get('mode')!r} is not formal (pilots stay masked, 040)")
    if meta.get("env_valid") is not True:
        problems.append(f"{path.name}: the session is env-invalid (rerun it under a new id, 012)")
    if (meta.get("stage"), meta.get("part"), plan.get("part"), meta.get("platform")) != (STAGE, PART, PART, PLATFORM):
        problems.append(f"{path.name}: not a q3 session (stage / part / platform)")
    if meta.get("plan_sha256") != sha256_json(plan):
        problems.append(f"{path.name}: plan.json is not the plan the session ran (plan_sha256)")
    if meta.get("harness_sha256") != harness or plan.get("harness_sha256") != harness:
        problems.append(f"{path.name}: ran another harness (harness_sha256; 040 凍結 freezes it)")
    if plan.get("stage_q3_sha256") != stage_q3_sha256:
        problems.append(f"{path.name}: planned with another configs/stage_q3.toml (stage_q3_sha256)")
    if plan.get("pins_sha256") != pins_sha256:
        problems.append(f"{path.name}: planned with another configs/pins.toml (pins_sha256)")
    if meta.get("chunks_sha256") != pinned or (plan.get("chunks") or {}).get("sha256") != pinned:
        problems.append(f"{path.name}: scored another chunk file (chunks_sha256)")
    return problems


def _scores(path: Path, row: dict, manifest: dict) -> list:
    f = path / row["scores"]
    if not f.is_file():
        raise StageQ3Error(f"{f} is missing: fetch raw/ from the local archive first (tools/rtx/fetch_raw.sh)")
    if file_sha256(f) != (manifest.get("files", {}).get(row["scores"]) or {}).get("sha256"):
        raise StageQ3Error(f"{f} does not match raw_manifest.json")
    rows = load_jsonl(f)
    if [round(x, 12) for x in chunk_nll(rows)] != [round(x, 12) for x in row["chunk_nll"]]:
        raise StageQ3Error(f"{f} does not give the row's per-chunk NLL")
    return rows


def analyze_session(path, *, cfg: dict, stage_q3_sha256: str, pins_sha256: str, root=ROOT,
                    n_boot: int | None = None) -> dict:
    """H-Q3 of a formal session (040 凍結): the frozen config and its pilot, the session's provenance, and the raw
    scores checked against raw_manifest.json and the committed per-chunk NLL. An oom or error leaves H-Q3 NT."""
    problems = config_problems(cfg)
    if not problems and cfg["freeze"]["stage_q3"] is not True:
        problems = ["freeze.stage_q3 is not true in configs/stage_q3.toml (040 凍結: no Q3 analysis before it)"]
    if problems:
        raise StageQ3Error("; ".join(problems))
    problems = freeze_evidence_problems(cfg, root) + session_problems(path, cfg=cfg, stage_q3_sha256=stage_q3_sha256,
                                                                      pins_sha256=pins_sha256)
    if problems:
        raise StageQ3Error("; ".join(problems))
    path = Path(path)
    rows = {r["which"]: r for r in load_jsonl(path / "rows.jsonl")}
    if set(rows) != set(JOBS):
        raise StageQ3Error(f"{path.name}: rows are {sorted(rows)}, not {sorted(JOBS)}")
    manifest = json.loads((path / "raw_manifest.json").read_text())
    scores = {w: _scores(path, r, manifest) if r["outcome"] == "ok" else None for w, r in rows.items()}
    q = cfg["stage_q3"]
    out = analyze(scores["w4a16"], scores["bf16"], n_boot=n_boot or q["n_boot"], seed=q["seed"])
    return {"session": path.name, "harness_sha256": harness_sha256(), "stage_q3_sha256": stage_q3_sha256,
            "pins_sha256": pins_sha256, "outcomes": {w: r["outcome"] for w, r in rows.items()}, **out}


# ---------------------------------------------------------------- CLI

def _hf_encoder(snapshot: str) -> Callable:  # pragma: no cover - needs transformers (WSL)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    return lambda text: tok(text, add_special_tokens=False)["input_ids"]


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chunks")
    for a in ("--tokenizer", "--text", "--out"):
        p.add_argument(a, required=True)
    p = sub.add_parser("plan")
    p.add_argument("--mode", required=True, choices=["pilot", "formal"])
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    for a in ("--plan", "--data", "--out"):
        p.add_argument(a, required=True)
    p.add_argument("--gpu-lock-held", action="store_true")
    p = sub.add_parser("score")
    for a in ("--job", "--chunks", "--out"):
        p.add_argument(a, required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--session", required=True, help="results/stage_q/<formal q3 session>, raw/ fetched")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "chunks":  # pragma: no cover - WSL
            print(json.dumps(build_chunks(Path(args.text).read_text(), _hf_encoder(args.tokenizer), args.out)))
        elif args.cmd == "plan":
            raw, cfg_raw = (ROOT / "configs" / "pins.toml").read_bytes(), CONFIG.read_bytes()
            plan = build_plan(tomllib.loads(raw.decode()), tomllib.loads(cfg_raw.decode()), mode=args.mode,
                              pins_sha256=hashlib.sha256(raw).hexdigest(),
                              stage_q3_sha256=hashlib.sha256(cfg_raw).hexdigest())
            Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
        elif args.cmd == "analyze":
            raw, cfg_raw = (ROOT / "configs" / "pins.toml").read_bytes(), CONFIG.read_bytes()
            session = Path(args.session)
            if output_dir_problems(session):
                raise StageQ3Error("; ".join(output_dir_problems(session)))
            result = analyze_session(session, cfg=tomllib.loads(cfg_raw.decode()),
                                     stage_q3_sha256=hashlib.sha256(cfg_raw).hexdigest(),
                                     pins_sha256=hashlib.sha256(raw).hexdigest())
            (session / "analysis.json").write_text(json.dumps(result, indent=1) + "\n")
            print(json.dumps({"H-Q3": result["H-Q3"]["verdict"]}))
        elif args.cmd == "run":  # pragma: no cover - WSL
            import run_core_b as rb
            pins = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
            s = run_q3(json.loads(Path(args.plan).read_text()), data_dir=Path(args.data).resolve(), out_dir=args.out,
                       capture_env=rb.make_vllm_env_capture(ROOT / "scripts" / "env_capture.py",
                                                            str(ROOT / ".venv" / "bin" / "python")),
                       session_id=Path(args.out).name, gpu_lock_held=args.gpu_lock_held,
                       wait_idle=default_wait_idle(pins), vllm_python=str(ROOT / ".venv" / "bin" / "python"))
            print(json.dumps(s["outcomes"]))
        else:  # pragma: no cover - the child process in the vLLM venv
            score(json.loads(Path(args.job).read_text()), args.chunks, args.out)
    except StageQ3Error as exc:
        print(f"stage_q3: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
