#!/usr/bin/env python3
"""Stage Q3 (decision 040): the quality of Core B's W4A16 checkpoint against its BF16 base in one runtime (vLLM
0.30.0), on WikiText-2 test cut and scored as llama-perplexity cuts and scores it (005).

- chunks (WSL, project venv, offline HF cache): the tokenizer whose tokenizer.json is the cross-checked one (005's
  quality/subsets/tokenizer-crosscheck.json) encodes the pinned wiki.test.raw without special tokens; it must give
  llama-perplexity's 297,194 tokens, cut into floor(n / 512) = 580 chunks of 512 ids in order (Qwen3.5 adds no BOS),
  written as {"i", "ids"} JSONL with a .meta.json of its sources. The file's sha256 is pinned in configs/stage_q3.toml
  before the pilot.
- Scoring: each chunk is one prompt (TokensPrompt) with SamplingParams(max_tokens 1, temperature 0, prompt_logprobs
  1). The scored tokens are positions FIRST + 1 .. N_CTX - 1 (llama-perplexity scores logits [n_ctx / 2, n_ctx - 1),
  255 tokens per chunk); for each, the logprob of the actual token and the rank-1 id (ties: the smallest id).
- plan: two jobs, W4A16 (pins core_b) and BF16 (pins core_a 9B source, cpu_offload_gb from the config), with Core B's
  engine where it bears on the numbers (language model only, bfloat16, FLASH_ATTN, pins stage_b GPU memory
  utilization and GDN prefill backend, no prefix caching); a pilot scores PILOT_CHUNKS chunks, a formal plan all of
  them and needs freeze.stage_q3 and the passed pilot.
- run (WSL, GPU lock verified): Core B's server env (core_b_plan.server_env) and toolchain check, then per job the 031
  idle window, env capture, one child process (`score`) in its own process group, env capture, and the output checked
  against the chunks and the engine facts in its log (coreb_smoke.parse_server_log). Token-level scores stay in raw/
  (the top-1 ids can echo the text, 005: no item text in git); each row commits the per-chunk NLL. An out-of-memory
  failure is outcome oom (040: NT, no rerun with other settings); a timeout is an error.
- analyze: PPL of each checkpoint, the paired per-chunk delta NLL (W4A16 - BF16) with a chunk bootstrap
  (quality_analysis's frozen helpers), top-1 agreement; H-Q3: PPL(W4A16) / PPL(BF16) - 1 <= 2 %. Only the first
  formal Q3 session is analyzed.

Usage:
    .venv/bin/python quality/stage_q3.py chunks --tokenizer <dir with tokenizer.json> \\
        --text data/datasets/wiki.test.raw --out data/stage_q/q3-chunks-512.jsonl        # WSL
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
import re
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_b_plan as bp  # noqa: E402
import coreb_smoke as cs  # noqa: E402
import gpu_idle as gi  # noqa: E402
import quality_analysis as qa  # noqa: E402
from run_core_a import env_problems, provenance_problems, sha256_json, write_raw_manifest  # noqa: E402

SCHEMA_VERSION = 1
STAGE, PART, PLATFORM = "Q", "q3", "wsl2-cuda"
N_CTX = 512
FIRST = N_CTX // 2                                       # llama-perplexity: first = n_ctx / 2
SCORED = N_CTX - FIRST - 1                               # 255 tokens per chunk
WIKITEXT_TOKENS = 297_194                                # llama-perplexity's count (pins: 580 chunks of 512)
CHUNKS = WIKITEXT_TOKENS // N_CTX
PILOT_CHUNKS = 10
MAX_MODEL_LEN = 1024                                     # a 512-token chunk and the one generated token
MAX_NUM_SEQS = 4
ATTENTION_BACKEND = "FLASH_ATTN"                         # core_b_plan.server_argv (003: pinned, never auto)
H_Q3_MAX = 0.02                                          # 040: PPL(W4A16) / PPL(BF16) - 1 <= 2 %
CHUNKS_PATH = "{data}/stage_q/q3-chunks-512.jsonl"
CONFIG = ROOT / "configs" / "stage_q3.toml"
CROSSCHECK = ROOT / "quality" / "subsets" / "tokenizer-crosscheck.json"
HARNESS_FILES = ("quality/stage_q3.py",)                 # every module it imports is pinned by a guard test
RESULTS = ("results", "stage_q")
JOBS = ("w4a16", "bf16")
CONFIG_KEYS = {"schema_version": None, "freeze": {"stage_q3", "pilot_q3"},
               "stage_q3": {"chunks_sha256", "chunks", "bf16_cpu_offload_gb", "max_num_batched_tokens", "pin_memory",
                            "n_boot", "seed"}}
# torch's allocator and vLLM's KV sizing (kv_cache_utils.py); matched only in the log's last OOM_TAIL lines
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "No available memory for the cache blocks",
               "larger than the available KV cache memory")
OOM_TAIL = 60


class StageQ3Error(RuntimeError):
    pass


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def harness_sha256(root=ROOT) -> str:
    """sha256 over HARNESS_FILES (path and bytes): the harness a pilot or a formal session ran."""
    h = hashlib.sha256()
    for rel in HARNESS_FILES:
        h.update(rel.encode() + b"\0" + (Path(root) / rel).read_bytes() + b"\0")
    return h.hexdigest()


def host_of(rec: dict) -> dict:
    os_info = (rec or {}).get("os") or {}
    return {"system": os_info.get("system"), "wsl": os_info.get("wsl"),
            "nvidia": bool(((rec or {}).get("gpu") or {}).get("nvidia"))}


def host_problems(host: dict | None) -> list:
    want = {"system": "Linux", "wsl": True, "nvidia": True}
    return [] if host == want else [f"ran on {host!r}, not a {PLATFORM} host {want}"]


def vllm_version(rec: dict):
    return (((rec or {}).get("runtime") or {}).get("vllm") or {}).get("vllm")


# ---------------------------------------------------------------- config and freeze

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def config_problems(cfg: dict) -> list:
    """configs/stage_q3.toml: exactly its keys and types; once frozen, the pilot and the 580-chunk file pinned."""
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
    if not isinstance(q["chunks_sha256"], str) or not _is_int(q["chunks"]) or q["chunks"] not in (0, CHUNKS):
        problems.append(f"stage_q3.chunks_sha256 must be a string and stage_q3.chunks 0 (not yet) or {CHUNKS}")
    if f["stage_q3"] is True:
        if not f["pilot_q3"]:
            problems.append("freeze.pilot_q3 must name the passed pilot once freeze.stage_q3 is true")
        if len(str(q["chunks_sha256"])) != 64 or q["chunks"] != CHUNKS:
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


def pilot_problems(path, pins: dict, cfg: dict) -> list:
    """A passed Q3 pilot (040 凍結): mode pilot on a WSL CUDA host with the pinned vLLM, plan.json the pilot ran and equal
    to build_plan's pilot plan under this config (engine, offload, env), both jobs ok, the pinned chunk file, and the
    current harness."""
    path = Path(path)
    try:
        s = json.loads((path / "session.json").read_text())
        plan = json.loads((path / "plan.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"{path.name}: no readable session.json / plan.json ({exc})"]
    out, problems = s.get("outcomes") or {}, []
    if (s.get("mode"), s.get("part"), s.get("platform")) != ("pilot", PART, PLATFORM):
        problems.append(f"{path.name}: not a q3 pilot (mode / part / platform)")
    problems += [f"{path.name}: {p}" for p in host_problems(s.get("host"))]
    if s.get("vllm_version") != pins["vllm"]["version"]:
        problems.append(f"{path.name}: vLLM {s.get('vllm_version')!r}, not the pinned {pins['vllm']['version']}")
    if s.get("plan_sha256") != sha256_json(plan):
        problems.append(f"{path.name}: plan.json is not the plan the pilot ran (plan_sha256)")
    if plan.get("jobs") != build_plan(pins, cfg, mode="pilot")["jobs"]:
        problems.append(f"{path.name}: the pilot's jobs are not build_plan's pilot jobs under this config")
    if out.get("ok") != len(JOBS) or out.get("oom") or out.get("error"):
        problems.append(f"{path.name}: not a passed q3 pilot (both jobs ok; outcomes {out})")
    if s.get("chunks_sha256") != cfg["stage_q3"]["chunks_sha256"]:
        problems.append(f"{path.name}: scored another chunk file than the pinned one (chunks_sha256)")
    if s.get("harness_sha256") != harness_sha256():
        problems.append(f"{path.name}: ran another harness (harness_sha256 {s.get('harness_sha256')!r})")
    return problems


def freeze_evidence_problems(cfg: dict, root, pins: dict) -> list:
    """Once freeze.stage_q3 is true: pilot_q3 under root/results/stage_q passes pilot_problems. [] before the freeze."""
    freeze = cfg.get("freeze") or {}
    if freeze.get("stage_q3") is not True:
        return []
    path = Path(root).joinpath(*RESULTS, str(freeze.get("pilot_q3")))
    return [f"freeze.pilot_q3: {p}" for p in pilot_problems(path, pins, cfg)]


# ---------------------------------------------------------------- chunks and scores

def cut_chunks(ids: list, n_ctx: int = N_CTX) -> list:
    """llama-perplexity's chunks: floor(len / n_ctx) consecutive windows; the remainder is not scored."""
    return [list(ids[i * n_ctx:(i + 1) * n_ctx]) for i in range(len(ids) // n_ctx)]


def chunk_source_problems(*, text_sha256: str, tokenizer_sha256: str, tokens: int, pins: dict,
                          crosscheck: dict) -> list:
    """The chunk file's sources: the pinned wiki.test.raw, the cross-checked HF tokenizer (005) and llama-perplexity's
    token count, so both checkpoints are scored on the chunks 005's KLD / PPL use."""
    problems = []
    want_text = pins["quality"]["files"]["wiki_test_raw"]["sha256"]
    if text_sha256 != want_text:
        problems.append(f"wiki.test.raw sha256 {text_sha256} is not the pinned {want_text}")
    if tokenizer_sha256 != crosscheck["hf_tokenizer_sha256"]:
        problems.append(f"tokenizer.json sha256 {tokenizer_sha256} is not the cross-checked "
                        f"{crosscheck['hf_tokenizer_sha256']} (005)")
    if WIKITEXT_TOKENS // N_CTX != pins["quality"]["expected"]["wikitext_chunks"]:
        problems.append(f"{WIKITEXT_TOKENS // N_CTX} chunks disagree with pins quality.expected.wikitext_chunks")
    if tokens != WIKITEXT_TOKENS:
        problems.append(f"the tokenizer gives {tokens} tokens, not llama-perplexity's {WIKITEXT_TOKENS} "
                        "(the chunks would not be 005's; decide in a decision log)")
    return problems


def build_chunks(ids: list, out, *, sources: dict) -> dict:
    """Write the chunk file and its .meta.json (sources: text and tokenizer sha256, token count)."""
    chunks = cut_chunks(ids)
    data = "".join(json.dumps({"i": i, "ids": c}) + "\n" for i, c in enumerate(chunks)).encode()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    info = {**sources, "tokens": len(ids), "chunks": len(chunks), "sha256": hashlib.sha256(data).hexdigest(),
            "path": str(out)}
    out.with_suffix(".meta.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def load_jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def scored_positions() -> range:
    return range(FIRST + 1, N_CTX)


def _field(entry, name):
    return entry[name] if isinstance(entry, dict) else getattr(entry, name)


def scored(ids: list, prompt_logprobs: list) -> dict:
    """vLLM prompt logprobs (per position: {token id (int or str): Logprob or {"logprob", "rank"}}) -> the actual
    token's logprob and the top-1 id at every scored position: the smallest id among the returned entries with the
    highest logprob (vLLM V1 ranks a tied actual token 2 and the topk entry 1, so the rule reads the values, for both
    checkpoints; a tie between two other tokens is not visible with prompt_logprobs = 1)."""
    logprobs, top1 = [], []
    for p in scored_positions():
        d = prompt_logprobs[p] or {}
        entry = d.get(ids[p], d.get(str(ids[p])))
        if entry is None:
            raise StageQ3Error(f"position {p}: no logprob for the prompt token {ids[p]}")
        if not any(_field(v, "rank") == 1 for v in d.values()):
            raise StageQ3Error(f"position {p}: no rank-1 token")
        top = max(float(_field(v, "logprob")) for v in d.values())
        logprobs.append(float(_field(entry, "logprob")))
        top1.append(min(int(k) for k, v in d.items() if float(_field(v, "logprob")) == top))
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


def engine_problems(log_text: str, pins: dict) -> tuple:
    """(facts, problems): the child's log must show Core B's attention backend, text-only mode and GDN prefill kernel."""
    facts = cs.parse_server_log(log_text)
    want_gdn = cs.GDN_KERNEL_NAMES[pins["stage_b"]["gdn_prefill_backend"]]
    problems = []
    if facts.get("attention_backend") != ATTENTION_BACKEND:
        problems.append(f"attention backend {facts.get('attention_backend')!r}, not {ATTENTION_BACKEND}")
    if not facts.get("text_only"):
        problems.append("no text-only line (language_model_only)")
    if facts.get("gdn_prefill_kernel") != want_gdn:
        problems.append(f"GDN prefill kernel {facts.get('gdn_prefill_kernel')!r}, not {want_gdn}")
    keep = ("attention_backend", "text_only", "gdn_prefill_kernel", "gdn_decode_kernel", "kv_cache_tokens")
    return {k: facts.get(k) for k in keep}, problems


def oom_in(text: str) -> bool:
    return any(m in line for line in text.splitlines()[-OOM_TAIL:] for m in OOM_MARKERS)


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
            "attention_backend": ATTENTION_BACKEND, "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": q["max_num_batched_tokens"],
            "gpu_memory_utilization": float(sb["gpu_memory_utilization"]), "enable_prefix_caching": False,
            "gdn_prefill_backend": sb["gdn_prefill_backend"], "seed": 0,
            "cpu_offload_gb": q["bf16_cpu_offload_gb"] if which == "bf16" else 0}


def build_plan(pins: dict, cfg: dict, *, mode: str, pins_sha256: str | None = None,
               stage_q3_sha256: str | None = None, root=ROOT) -> dict:
    """Pilot plans need a well-formed config; formal plans need freeze.stage_q3 and the passed pilot under root."""
    if mode not in ("pilot", "formal"):
        raise StageQ3Error(f"unknown mode {mode!r}")
    problems = config_problems(cfg)
    if mode == "formal" and (problems or cfg["freeze"]["stage_q3"] is not True):
        raise StageQ3Error("formal Q3 plans need freeze.stage_q3 = true in configs/stage_q3.toml (040 凍結)"
                           + (f": {'; '.join(problems)}" if problems else ""))
    if problems:
        raise StageQ3Error("configs/stage_q3.toml: " + "; ".join(problems))
    formal = mode == "formal"
    if formal:
        evidence = freeze_evidence_problems(cfg, root, pins)
        if evidence:
            raise StageQ3Error("formal Q3 plans need the passed pilot first (040 凍結): " + "; ".join(evidence))
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
    """(exit code, wall s); None on a timeout. The child gets its own process group, killed whole on a timeout or an
    interrupt, so vLLM's EngineCore never outlives it (as run_adv / run_c0_probe)."""
    t0 = time.monotonic()
    with open(out, "w") as fo, open(err, "w") as fe:
        proc = subprocess.Popen(argv, stdout=fo, stderr=fe, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            code = None
        except BaseException:
            _kill_group(proc)
            raise
        if code is None:
            _kill_group(proc)
    return code, round(time.monotonic() - t0, 3)


def _kill_group(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def child_env(job: dict, base: dict | None = None) -> dict:
    """Core B's server env (drops inherited VLLM_* / CUDA_* / GGML_*; CUDA_HOME shim, venv bin first on PATH, the GDN
    decode kernel, 014) without CAPSERVE_*, plus the job's own variables."""
    env = bp.server_env(dict(os.environ if base is None else base), pin_memory=job["env"]["VLLM_WSL2_ENABLE_PIN_MEMORY"])
    return {**{k: v for k, v in env.items() if not k.startswith("CAPSERVE_")}, **job["env"]}


def default_wait_idle(pins: dict) -> Callable:  # pragma: no cover - nvidia-smi (WSL)
    bound = gi.headroom_bound(pins["stage_b"]["v_total_mib"])
    return lambda: gi.wait_idle(gi.nvidia_sample, gi.nvidia_compute_apps, bound_mib=bound)


def default_lock_check() -> bool:  # pragma: no cover - /proc (WSL)
    import run_adv
    return run_adv.held_gpu_lock()


def shard_problems(snapshot: Path, want: dict, hash_file: Callable = file_sha256) -> list:
    problems = []
    for name, sha in sorted(want.items()):
        try:
            got = hash_file(snapshot / name)
        except OSError as exc:
            problems.append(f"{snapshot / name}: unreadable ({exc})")
            continue
        if got != sha:
            problems.append(f"{snapshot / name}: sha256 {got} is not the pinned {sha}")
    return problems


def default_checkpoint_problems(plan: dict, pins: dict) -> list:  # pragma: no cover - the HF cache (WSL)
    """The BF16 shards against pins core_a 9B source_shards (the W4A16 repo is pinned by revision only)."""
    from huggingface_hub import snapshot_download

    job = next(j for j in plan["jobs"] if j["which"] == "bf16")
    snap = Path(snapshot_download(job["model"]["repo"], revision=job["model"]["revision"], local_files_only=True))
    shards = next(m for m in pins["core_a"]["models"] if m["size"] == "9B")["source_shards"]
    return shard_problems(snap, shards)


def encode_text(raw: bytes, encode: Callable) -> list:
    """The test set as llama-perplexity -f reads it (common/arg.cpp pops one trailing newline; 005's
    tokenizer_crosscheck.perplexity_prompt), encoded."""
    text = raw.decode()
    return list(encode(text[:-1] if text.endswith("\n") else text))


def output_dir_problems(out_dir, *, repo_root=ROOT) -> list:
    """A session directory is exactly results/stage_q/<id> (no nesting: every attempt is visible)."""
    if Path(out_dir).resolve().parent != Path(repo_root).resolve().joinpath(*RESULTS):
        return [f"{out_dir} is not results/stage_q/<session id> (040)"]
    return []


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def formal_attempts(root) -> list:
    """Every formal q3 attempt under results/stage_q, oldest first (040 補記 14): a directory whose session.json (final
    or the running stub) or, when that is unreadable, plan.json is a formal q3 one; a plan saved alone and runs refused
    before their first job are left out. valid = finished and env valid; an unreadable session.json is a void
    attempt, oldest (fail closed)."""
    base = Path(root).joinpath(*RESULTS)
    out = []
    for d in sorted({f.parent for name in ("session.json", "plan.json") for f in base.rglob(name)} - {base}):
        s = _read_json(d / "session.json")
        if not isinstance(s, dict) and not any(x.name != "plan.json" for x in d.iterdir()):
            continue
        head = s if isinstance(s, dict) else _read_json(d / "plan.json")
        if not isinstance(head, dict) or (head.get("stage"), head.get("mode"), head.get("part")) != \
                (STAGE, "formal", PART):
            continue
        s = s if isinstance(s, dict) else {}
        if s.get("status") == "refused":
            continue
        out.append({"dir": d, "id": d.name, "started_at": s.get("started_at") or "", "status": s.get("status"),
                    "valid": s.get("status") == "finished" and s.get("env_valid") is True,
                    "superseded": dict(s.get("superseded") or {})})
    return sorted(out, key=lambda a: (a["started_at"], str(a["dir"])))


def git_tracked(path, root=ROOT) -> bool:
    """Committed: tracked by git here; on the WSL copy (deploy_bundle: no .git), listed in DEPLOY_MANIFEST.json with
    exactly these bytes."""
    root, path = Path(root).resolve(), Path(path).resolve()
    if (root / ".git").exists():  # pragma: no cover - git
        return subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(path)],
                              capture_output=True).returncode == 0
    manifest = _read_json(root / "DEPLOY_MANIFEST.json")
    if not isinstance(manifest, dict) or root not in path.parents:
        return False
    return (manifest.get("files") or {}).get(path.relative_to(root).as_posix()) == file_sha256(path)


def superseded_problems(superseded: dict, new_id: str, *, root=ROOT, tracked: Callable = git_tracked) -> list:
    """040 補記 14's exception: each superseded session id names a git-tracked decision log under docs/decisions/ whose
    text names both that id and new_id (the decision lives outside every hashed file)."""
    problems = []
    for sid, log in sorted((superseded or {}).items()):
        path = (Path(root) / log).resolve()
        if sid == new_id:
            problems.append(f"--superseded {sid}: a session cannot supersede itself")
        elif not str(log).startswith("docs/decisions/") or Path(root).resolve().joinpath("docs", "decisions") \
                not in path.parents or path.relative_to(Path(root).resolve()).as_posix() != log or not path.is_file():
            problems.append(f"--superseded {sid}: {log} is not a repo-relative decision log docs/decisions/<file> "
                            "(recorded as typed, read on both machines)")
        elif not tracked(path):
            problems.append(f"--superseded {sid}: {log} is not committed (git ls-files)")
        elif not (_names(text := path.read_text(), sid) and _names(text, new_id)):
            problems.append(f"--superseded {sid}: {log} does not name both {sid} and {new_id}")
    return problems


def _names(text: str, sid: str) -> bool:
    """sid as a whole token (a rerun's id often extends the first one's, e.g. -c1 -> -c1b)."""
    return re.search(rf"(?<![\w-]){re.escape(sid)}(?![\w-])", text) is not None


def recorded_supersedes(attempts: list) -> dict:
    """{superseded id: (decision log, the attempt that recorded it)} from the attempts' session.json (each run records
    its --superseded), so the analysis cannot choose them afterwards."""
    return {sid: (log, a["id"]) for a in attempts for sid, log in a["superseded"].items()}


def recorded_supersede_problems(attempts: list, *, root=ROOT, tracked: Callable = git_tracked) -> list:
    """Every recorded supersede checks out: its log (superseded_problems) and an earlier attempt it names."""
    started = {a["id"]: a["started_at"] for a in attempts}
    out = []
    for sid, (log, by) in sorted(recorded_supersedes(attempts).items()):
        out += superseded_problems({sid: log}, by, root=root, tracked=tracked)
        if sid != by and not (sid in started and started[sid] < started[by]):
            out.append(f"{by} supersedes {sid}, which is not an earlier formal attempt of this part")
    return out


def run_q3(plan: dict, *, data_dir, out_dir, capture_env: Callable, session_id: str, gpu_lock_held: bool,
           wait_idle: Callable, pins: dict, repo_root=ROOT, run_proc: Callable = default_run_proc,
           vllm_python: str = ".venv/bin/python", timeout_s: float = 4 * 3600,
           now: Callable = lambda: dt.datetime.now(dt.timezone.utc).isoformat(), expected_plan: dict | None = None,
           lock_check: Callable | None = None, base_env: dict | None = None, toolchain_check: Callable | None = None,
           checkpoint_problems: Callable | None = None, superseded: dict | None = None,
           tracked: Callable = git_tracked) -> dict:
    """One Q3 session: both score jobs in order under the GPU lock. A formal plan must equal build_plan's
    (expected_plan); a formal session also needs the pinned vLLM, a clean env and the pinned BF16 shards."""
    if (plan.get("stage"), plan.get("part"), plan.get("platform")) != (STAGE, PART, PLATFORM):
        raise StageQ3Error(f"not a Q3 plan ({plan.get('stage')!r}, {plan.get('part')!r}, {plan.get('platform')!r})")
    problems = output_dir_problems(out_dir, repo_root=repo_root)
    if problems:
        raise StageQ3Error("; ".join(problems))
    if not (gpu_lock_held and (lock_check or default_lock_check)()):
        raise StageQ3Error("Q3 runs vLLM: hold the GPU lock (flock -n /tmp/ai-portfolio-gpu.lock ... --gpu-lock-held; "
                           "the flag alone is never trusted)")
    formal = plan["mode"] == "formal"
    if formal and plan.get("harness_sha256") != harness_sha256():
        raise StageQ3Error("the formal plan was built by another harness (harness_sha256); rebuild it with this one")
    if formal and (expected_plan is None or plan != expected_plan):
        raise StageQ3Error("a formal plan must be exactly build_plan's for these pins and configs/stage_q3.toml "
                           "(040 凍結); rebuild it with stage_q3.py plan --mode formal")
    out = Path(out_dir)
    if out.exists():
        raise StageQ3Error(f"{out} exists; sessions are never resumed or overwritten")
    if formal:
        bad = superseded_problems(superseded or {}, session_id, root=repo_root, tracked=tracked)
        attempts = formal_attempts(repo_root)
        gone = set(superseded or {}) | set(recorded_supersedes(attempts))
        prior = [a for a in attempts if a["id"] not in gone]
        if len(prior) >= 2 or any(a["valid"] for a in prior):
            bad.append(f"q3 already has formal attempts {[a['id'] for a in prior]}: one valid session, or one void "
                       "attempt and its rerun, at most (040 補記 14; --superseded needs a decision log)")
        if bad:
            raise StageQ3Error("; ".join(bad))
    chunks_path = Path(plan["chunks"]["path"].replace("{data}", str(data_dir)))
    try:
        chunks_sha = file_sha256(chunks_path)
        chunks = load_jsonl(chunks_path)
    except (OSError, ValueError) as exc:
        raise StageQ3Error(f"no readable chunks file {chunks_path}: {exc}") from exc
    if plan["chunks"]["sha256"] is not None and chunks_sha != plan["chunks"]["sha256"]:
        raise StageQ3Error(f"the chunks file {chunks_path} is not the pinned one (sha256 {chunks_sha})")
    if any(len(c["ids"]) != N_CTX for c in chunks) or len(chunks) < max(j["n_chunks"] for j in plan["jobs"]) or \
            (formal and {len(chunks), plan["chunks"]["chunks"]} != {j["n_chunks"] for j in plan["jobs"]}):
        raise StageQ3Error(f"the chunks file {chunks_path} does not hold the planned {N_CTX}-token chunks")
    envs = {j["id"]: child_env(j, base_env) for j in plan["jobs"]}
    tool = (toolchain_check or bp.toolchain_problems)(envs[plan["jobs"][0]["id"]])
    if tool:
        raise StageQ3Error("the vLLM toolchain is incomplete (014): " + "; ".join(tool))
    for sub in ("raw", "env"):
        (out / sub).mkdir(parents=True)
    session = {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": STAGE, "part": PART,
               "mode": plan["mode"], "platform": PLATFORM, "status": "running", "plan_sha256": sha256_json(plan),
               "pins_sha256": plan.get("pins_sha256"), "gpu_lock_held": gpu_lock_held,
               "harness_sha256": harness_sha256(), "chunks_sha256": chunks_sha, "inputs_verified": formal,
               "superseded": dict(superseded or {}), "started_at": now()}

    def write_session():                                    # atomic: a torn write never hides the attempt
        tmp = out / "session.json.tmp"
        tmp.write_text(json.dumps(session, indent=1) + "\n")
        os.replace(tmp, out / "session.json")

    write_session()                                         # the stub: an interrupted session is still an attempt
    try:                                    # before the first job nothing ran: any end here is refused
        (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
        pre = capture_env(out_path=out / "env" / "preflight.json", gpu_lock_held=gpu_lock_held)
        session.update(host=host_of(pre), vllm_version=vllm_version(pre))
        if formal:
            bad = env_problems(pre, PLATFORM) + provenance_problems(pre, pre) + host_problems(host_of(pre))
            if vllm_version(pre) != pins["vllm"]["version"]:
                bad.append(f"vLLM {vllm_version(pre)!r}, not the pinned {pins['vllm']['version']} (040: one runtime)")
            if not bad:
                bad = (checkpoint_problems or default_checkpoint_problems)(plan, pins)
            if bad:
                session.update(status="refused", refused=bad)
                write_session()
                raise StageQ3Error("formal Q3 run refused before it starts: " + "; ".join(bad))
    except BaseException as exc:
        if session["status"] != "refused":
            session.update(status="refused", refused=[f"interrupted before the first job: {type(exc).__name__}: {exc}"])
            write_session()
        raise
    rows, env_flags = [], []
    with open(out / "rows.jsonl", "w") as fh:
        for job in plan["jobs"]:
            jid = job["id"]
            rel = {"job": f"raw/{jid}.job.json", "scores": f"raw/{jid}.scores.jsonl", "stdout": f"raw/{jid}.stdout.txt",
                   "stderr": f"raw/{jid}.stderr.txt", "env_before": f"env/{jid}.before.json",
                   "env_after": f"env/{jid}.after.json"}
            base = {"job_id": jid, "kind": "score", "which": job["which"], "model": job["model"],
                    "n_chunks": job["n_chunks"], "chunks_sha256": chunks_sha, "env": bp.recorded_env(envs[jid])}
            idle, report = wait_idle()
            if not idle:
                row = {**base, "outcome": "error", "reason": f"gpu_not_idle (031): {report}"}
                env_flags.append({"job_id": jid, "phase": "before", "reason": "gpu_not_idle (031): the job never "
                                                                            "started, the session is void"})
            else:
                (out / rel["job"]).write_text(json.dumps(job, indent=1) + "\n")
                env_before = capture_env(out_path=out / rel["env_before"], gpu_lock_held=gpu_lock_held)
                argv = [vllm_python, str(Path(repo_root) / "quality" / "stage_q3.py"), "score", "--job",
                        str(out / rel["job"]), "--chunks", str(chunks_path), "--out", str(out / rel["scores"])]
                code, wall = run_proc(argv, out / rel["stdout"], out / rel["stderr"], timeout_s, envs[jid])
                env_after = capture_env(out_path=out / rel["env_after"], gpu_lock_held=gpu_lock_held)
                if formal:
                    for phase, rec in (("before", env_before), ("after", env_after)):
                        env_flags.extend({"job_id": jid, "phase": phase, "reason": why}
                                         for why in env_problems(rec, PLATFORM) + provenance_problems(rec, pre))
                text = (out / rel["stdout"]).read_text(errors="replace") + "\n" + (out / rel["stderr"]).read_text(
                    errors="replace")
                facts, engine_bad = engine_problems(text, pins)
                scores = load_jsonl(out / rel["scores"]) if code == 0 and (out / rel["scores"]).is_file() else None
                bad = output_problems(scores, chunks, job["n_chunks"]) if scores is not None else None
                if code is None:
                    row = {**base, "outcome": "error", "reason": f"timeout after {timeout_s} s (process group killed)"}
                elif scores is not None and not bad and not engine_bad:
                    row = {**base, "outcome": "ok", "reason": None, "chunk_nll": chunk_nll(scores),
                           "n_tokens": SCORED * len(scores)}
                elif scores is None and oom_in(text):
                    row = {**base, "outcome": "oom", "reason": "out of memory (040: NT, no rerun with other settings)"}
                else:
                    why = (bad or []) + engine_bad if scores is not None else [f"exit {code} without a score file"]
                    row = {**base, "outcome": "error", "reason": "; ".join(why)}
                row.update(argv=argv, exit_code=code, wall_s=wall, engine_facts=facts, **rel)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    outcomes = {"ok": 0, "oom": 0, "error": 0}
    for r in rows:
        outcomes[r["outcome"]] += 1
    session.update({"status": "finished", "finished_at": now(), "outcomes": outcomes, "env_flags": env_flags,
                    "env_valid": not env_flags})
    write_raw_manifest(out)
    write_session()
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


def session_problems(path, *, cfg: dict, stage_q3_sha256: str, pins_sha256: str, pins: dict, root=ROOT,
                     tracked: Callable = git_tracked) -> list:
    """A finished, env-valid formal q3 session directly under results/stage_q, on a WSL CUDA host with the pinned vLLM,
    its plan.json exactly build_plan's, the pinned chunks, and the first valid formal q3 session after at most one
    void attempt (attempts a later run recorded as superseded are left out, each checked against its committed
    decision log naming both ids; 040 補記 14)."""
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
    if meta.get("status") != "finished":
        problems.append(f"{path.name}: status {meta.get('status')!r}, not a finished session")
    if path.resolve().parent != Path(root).resolve().joinpath(*RESULTS):
        problems.append(f"{path}: not results/stage_q/<session id>")
    if meta.get("env_valid") is not True:
        problems.append(f"{path.name}: the session is env-invalid (void: one rerun under a new id, 012)")
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
    problems += [f"{path.name}: {p}" for p in host_problems(meta.get("host"))]
    if meta.get("vllm_version") != pins["vllm"]["version"] or meta.get("inputs_verified") is not True:
        problems.append(f"{path.name}: not the pinned vLLM with verified checkpoints (vllm_version / inputs_verified)")
    if not problems:
        try:
            want = build_plan(pins, cfg, mode="formal", pins_sha256=pins_sha256, stage_q3_sha256=stage_q3_sha256,
                              root=root)
        except StageQ3Error as exc:
            want = None
            problems.append(f"{path.name}: build_plan refuses its formal plan ({exc})")
        if want is not None and plan != want:
            problems.append(f"{path.name}: plan.json is not build_plan's formal q3 plan")
        attempts = formal_attempts(root)
        problems += [f"{path.name}: {p}" for p in recorded_supersede_problems(attempts, root=root, tracked=tracked)]
        gone = recorded_supersedes(attempts)
        earlier = [a for a in attempts if a["dir"].resolve() != path.resolve()
                   and a["started_at"] < (meta.get("started_at") or "") and a["id"] not in gone]
        if len(earlier) > 1 or any(a["valid"] for a in earlier):
            problems.append(f"{path.name}: earlier formal q3 attempts {[a['id'] for a in earlier]} (040: the first "
                            "valid session after at most one void attempt; another needs --superseded and its decision "
                            "log when it runs)")
    return problems


def q3_session(*, root=ROOT, tracked: Callable = git_tracked) -> tuple:
    """(the session dir to analyze or None, record), as stage_q_analysis.part_session: the first valid formal q3
    attempt after at most one void one, recorded supersedes left out once their decision logs check out."""
    attempts = formal_attempts(root)
    bad = recorded_supersede_problems(attempts, root=root, tracked=tracked)
    if bad:
        raise StageQ3Error("; ".join(bad))
    sup = recorded_supersedes(attempts)
    kept = [a for a in attempts if a["id"] not in sup]
    record = {"attempts": [a["id"] for a in kept], "superseded": {sid: by for sid, (_, by) in sorted(sup.items())}}
    valid = [i for i, a in enumerate(kept) if a["valid"]]
    if not valid and any(a["valid"] for a in attempts if a["id"] in sup):
        raise StageQ3Error("q3: a valid session was superseded, but no valid formal session replaces it")
    if not kept:
        return None, {**record, "status": "not run"}
    if not valid:
        return None, {**record, "status": "no valid formal session (every attempt is void)"}
    if valid[0] > 1:
        raise StageQ3Error(f"q3: {valid[0]} void attempts before {kept[valid[0]]['id']}; 040 allows one rerun")
    return kept[valid[0]]["dir"], {**record, "status": "analyzed", "session": kept[valid[0]]["id"],
                                   "void_before": [a["id"] for a in kept[:valid[0]]]}


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


def analyze_session(path, *, cfg: dict, stage_q3_sha256: str, pins_sha256: str, pins: dict, root=ROOT,
                    n_boot: int | None = None, tracked: Callable = git_tracked) -> dict:
    """H-Q3 of a formal session (040 凍結): the frozen config and its pilot, the session's provenance, and the raw
    scores checked against raw_manifest.json and the committed per-chunk NLL. An oom or error leaves H-Q3 NT."""
    problems = config_problems(cfg)
    if not problems and cfg["freeze"]["stage_q3"] is not True:
        problems = ["freeze.stage_q3 is not true in configs/stage_q3.toml (040 凍結: no Q3 analysis before it)"]
    if problems:
        raise StageQ3Error("; ".join(problems))
    problems = freeze_evidence_problems(cfg, root, pins) + session_problems(
        path, cfg=cfg, stage_q3_sha256=stage_q3_sha256, pins_sha256=pins_sha256, pins=pins, root=root,
        tracked=tracked)
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


def analyze_q3(*, cfg: dict, stage_q3_sha256: str, pins_sha256: str, pins: dict, root=ROOT,
               n_boot: int | None = None, tracked: Callable = git_tracked) -> dict:
    """The analysis finds its own session (040 補記 14, as Q1 / Q2): H-Q3 of it, or NT ("not run", every attempt
    void) with the attempts listed; nothing before the freeze and its evidence (040 凍結)."""
    problems = config_problems(cfg)
    if not problems and cfg["freeze"]["stage_q3"] is not True:
        problems = ["freeze.stage_q3 is not true in configs/stage_q3.toml (040 凍結: no Q3 analysis before it)"]
    problems = problems or freeze_evidence_problems(cfg, root, pins)
    if problems:
        raise StageQ3Error("; ".join(problems))
    path, record = q3_session(root=root, tracked=tracked)
    if path is None:
        return {"session": None, "q3": record, "harness_sha256": harness_sha256(), "stage_q3_sha256": stage_q3_sha256,
                "pins_sha256": pins_sha256, "H-Q3": {"verdict": "NT", "why": record["status"]}}
    return {**analyze_session(path, cfg=cfg, stage_q3_sha256=stage_q3_sha256, pins_sha256=pins_sha256, pins=pins,
                              root=root, n_boot=n_boot, tracked=tracked), "q3": record}


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
    p.add_argument("--superseded", action="append", default=[], help="<session id>=docs/decisions/<log>.md")
    p = sub.add_parser("score")
    for a in ("--job", "--chunks", "--out"):
        p.add_argument(a, required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--out", required=True, help="results/stage_q/<name>.json (a new file); the session's raw/ fetched")
    args = parser.parse_args(argv)
    raw, cfg_raw = (ROOT / "configs" / "pins.toml").read_bytes(), CONFIG.read_bytes()
    pins, cfg = tomllib.loads(raw.decode()), tomllib.loads(cfg_raw.decode())
    pins_sha, cfg_sha = hashlib.sha256(raw).hexdigest(), hashlib.sha256(cfg_raw).hexdigest()
    try:
        if args.cmd == "chunks":  # pragma: no cover - WSL
            text = Path(args.text).read_bytes()
            tok_sha = file_sha256(Path(args.tokenizer) / "tokenizer.json")
            ids = encode_text(text, _hf_encoder(args.tokenizer))
            sources = {"text_sha256": hashlib.sha256(text).hexdigest(), "tokenizer_json_sha256": tok_sha,
                       "tokenizer": str(args.tokenizer)}
            bad = chunk_source_problems(text_sha256=sources["text_sha256"], tokenizer_sha256=tok_sha, tokens=len(ids),
                                        pins=pins, crosscheck=json.loads(CROSSCHECK.read_text()))
            if bad:
                raise StageQ3Error("no chunk file written: " + "; ".join(bad))
            print(json.dumps(build_chunks(ids, args.out, sources=sources)))
        elif args.cmd == "plan":
            plan = build_plan(pins, cfg, mode=args.mode, pins_sha256=pins_sha, stage_q3_sha256=cfg_sha)
            out = Path(args.out).resolve()
            if ROOT.joinpath(*RESULTS).resolve() in (out, *out.parents):
                raise StageQ3Error("write plans outside results/stage_q (a session directory is the runner's)")
            out.write_text(json.dumps(plan, indent=1) + "\n")
        elif args.cmd == "analyze":
            out = Path(args.out).resolve()
            if out.parent != ROOT.joinpath(*RESULTS).resolve() or out.suffix != ".json" or out.exists():
                raise StageQ3Error(f"--out {out} is not a new .json file directly under results/stage_q/")
            result = analyze_q3(cfg=cfg, stage_q3_sha256=cfg_sha, pins_sha256=pins_sha, pins=pins)
            out.write_text(json.dumps(result, indent=1) + "\n")
            print(json.dumps({"H-Q3": result["H-Q3"]["verdict"]}))
        elif args.cmd == "run":  # pragma: no cover - WSL
            import run_core_b as rb
            plan = json.loads(Path(args.plan).read_text())
            expected = build_plan(pins, cfg, mode="formal", pins_sha256=pins_sha, stage_q3_sha256=cfg_sha) \
                if plan.get("mode") == "formal" else None
            python = str(ROOT / ".venv" / "bin" / "python")
            s = run_q3(plan, data_dir=Path(args.data).resolve(), out_dir=args.out,
                       capture_env=rb.make_vllm_env_capture(ROOT / "scripts" / "env_capture.py", python),
                       session_id=Path(args.out).name, gpu_lock_held=args.gpu_lock_held,
                       wait_idle=default_wait_idle(pins), pins=pins, vllm_python=python, expected_plan=expected,
                       superseded=dict(x.split("=", 1) for x in args.superseded))
            print(json.dumps(s["outcomes"]))
        else:  # pragma: no cover - the child process in the vLLM venv
            score(json.loads(Path(args.job).read_text()), args.chunks, args.out)
    except StageQ3Error as exc:
        print(f"stage_q3: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
