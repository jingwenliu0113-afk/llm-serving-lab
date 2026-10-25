"""Tests for quality/stage_q3.py (decision 040 Q3): W4A16 against BF16 in vLLM on WikiText-2 512-token chunks scored
as llama-perplexity scores them; the chunk file, plans, the prompt-logprob parser, the runner, the freeze evidence and
the analysis."""

import hashlib
import json
import math
import shutil
import sys
import time
import tomllib
from pathlib import Path

import pytest

import stage_q3 as q3

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
PINS_SHA = hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()
REPO_CFG = tomllib.loads((ROOT / "configs" / "stage_q3.toml").read_text())   # frozen 2026-10-25 (040 凍結, Q3)
CFG = {**REPO_CFG, "freeze": {"stage_q3": False, "pilot_q3": ""}}               # an explicit DRAFT config
CROSSCHECK = json.loads((ROOT / "quality" / "subsets" / "tokenizer-crosscheck.json").read_text())
ENGINE_LOG = ("INFO Using FLASH_ATTN backend.\nINFO running in text-only mode\n"
              "INFO Using Triton/FLA GDN prefill kernel\nINFO GDN decode kernel: triton\n")


# ---------------------------------------------------------------- chunks and the parser

def test_chunks_are_cut_as_llama_perplexity_cuts_them(tmp_path):
    assert q3.cut_chunks(list(range(1030))) == [list(range(512)), list(range(512, 1024))]
    info = q3.build_chunks(list(range(1100)), tmp_path / "c.jsonl", sources={"text_sha256": "t"})
    rows = [json.loads(x) for x in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert [r["i"] for r in rows] == [0, 1] and rows[1]["ids"][0] == 512
    assert info["tokens"] == 1100 and info["chunks"] == 2 and len(info["sha256"]) == 64
    assert json.loads((tmp_path / "c.meta.json").read_text())["text_sha256"] == "t"
    assert q3.build_chunks(list(range(1100)), tmp_path / "d.jsonl", sources={})["sha256"] == info["sha256"]


def test_the_chunk_file_comes_from_the_pinned_text_the_crosschecked_tokenizer_and_005s_count():
    good = dict(text_sha256=PINS["quality"]["files"]["wiki_test_raw"]["sha256"],
                tokenizer_sha256=CROSSCHECK["hf_tokenizer_sha256"], tokens=297_194, pins=PINS, crosscheck=CROSSCHECK)
    assert q3.chunk_source_problems(**good) == [] and q3.WIKITEXT_TOKENS // q3.N_CTX == 580
    for over, why in (({"text_sha256": "0" * 64}, "wiki.test.raw"), ({"tokenizer_sha256": "0" * 64}, "cross-checked"),
                      ({"tokens": 297_190}, "297190 tokens")):
        assert any(why in p for p in q3.chunk_source_problems(**{**good, **over}))


def test_the_scored_positions_are_the_second_half_after_the_first_token():
    assert (q3.N_CTX, q3.FIRST, q3.SCORED) == (512, 256, 255)
    assert list(q3.scored_positions()) == list(range(257, 512))


def logprobs_for(ids, actual=-1.5, top=None):
    out = [None]
    for p in range(1, len(ids)):
        d = {str(ids[p]): {"logprob": actual, "rank": 2 if top is not None else 1}}
        if top is not None:
            d[str(top)] = {"logprob": -0.1, "rank": 1}
        out.append(d)
    return out


def test_prompt_logprobs_give_the_actual_token_and_the_top_one():
    ids = [7] * 512
    s = q3.scored(ids, logprobs_for(ids))
    assert len(s["logprobs"]) == 255 and s["logprobs"][0] == -1.5 and s["top1"] == [7] * 255
    s = q3.scored(ids, logprobs_for(ids, top=9))
    assert s["top1"] == [9] * 255 and s["logprobs"][-1] == -1.5

    class Lp:                                                  # vLLM's Logprob objects, keyed by int ids
        def __init__(self, logprob, rank):
            self.logprob, self.rank = logprob, rank

    objs = [None] + [{7: Lp(-2.0, 1)} for _ in range(511)]
    assert q3.scored(ids, objs)["logprobs"][0] == -2.0
    tied = [None] + [{7: Lp(-0.69, 2), 3: Lp(-0.69, 1)} for _ in range(511)]   # vLLM V1: the actual token ranks 2
    assert q3.scored(ids, tied)["top1"] == [3] * 255                    # tied top logprob: the smallest id
    tied = [None] + [{7: Lp(-0.69, 2), 9: Lp(-0.69, 1)} for _ in range(511)]
    assert q3.scored(ids, tied)["top1"] == [7] * 255                    # whichever id torch.topk returned
    assert q3.scored(ids, [None] + [{7: Lp(-0.5, 1)} for _ in range(511)])["top1"] == [7] * 255
    with pytest.raises(q3.StageQ3Error, match="position 257"):
        q3.scored(ids, [None] + [{"8": {"logprob": -1.0, "rank": 1}} for _ in range(511)])


def test_the_engine_facts_must_be_core_bs():
    facts, bad = q3.engine_problems(ENGINE_LOG, PINS)
    assert bad == [] and facts["attention_backend"] == "FLASH_ATTN"
    _, bad = q3.engine_problems(ENGINE_LOG.replace("FLASH_ATTN", "TRITON_ATTN"), PINS)
    assert any("FLASH_ATTN" in p for p in bad)
    assert q3.engine_problems("", PINS)[1]


def test_oom_is_read_from_the_end_of_the_log():
    assert q3.oom_in("x\n" * 5 + "ValueError: No available memory for the cache blocks. Try increasing\n")
    assert not q3.oom_in("OutOfMemoryError (recovered)\n" + "line\n" * 100 + "RuntimeError: other\n")


# ---------------------------------------------------------------- config, plans and the freeze evidence

def test_the_repo_config_is_frozen_on_the_passed_pilot_of_this_harness(monkeypatch):
    """040 凍結 (Q3, 2026-10-25): the repo config names the passed pilot (pin_memory "1", user decision 2026-10-25),
    which ran this harness, this config and the pinned chunk file; any edit after the freeze fails here."""
    monkeypatch.setattr(q3, "CHUNKS", q3.WIKITEXT_TOKENS // q3.N_CTX)
    assert q3.config_problems(REPO_CFG) == [] and REPO_CFG["freeze"] == {"stage_q3": True, "pilot_q3": "pilot-q3-20261025b"}
    assert REPO_CFG["stage_q3"]["pin_memory"] == "1"
    assert q3.harness_sha256() == "2bc3f2ee99128bf394c0e03029632e1e216d43c184616d0464c48ccbb6690480"
    assert q3.freeze_evidence_problems(REPO_CFG, ROOT, PINS) == []
    assert [(j["id"], j["n_chunks"]) for j in q3.build_plan(PINS, REPO_CFG, mode="formal", root=ROOT)["jobs"]] == [
        ("score-w4a16", 580), ("score-bf16", 580)]


def test_the_draft_config_is_well_formed_and_not_yet_frozen(monkeypatch):
    monkeypatch.setattr(q3, "CHUNKS", q3.WIKITEXT_TOKENS // q3.N_CTX)       # the repo pins the real file (580)
    assert q3.config_problems(CFG) == [] and CFG["freeze"]["stage_q3"] is False
    assert CFG["stage_q3"]["chunks"] == 580 and len(CFG["stage_q3"]["chunks_sha256"]) == 64   # pinned 2026-10-25
    assert q3.config_problems({**CFG, "freeze": {"stage_q3": True, "pilot_q3": ""}})
    assert q3.config_problems({**CFG, "stage_q3": {**CFG["stage_q3"], "bf16_cpu_offload_gb": -1}})
    assert q3.config_problems({**CFG, "stage_q3": {**CFG["stage_q3"], "chunks": 58}})       # 0 or 580 only


def chunk_file(path, n=12):
    rows = [{"i": i, "ids": [(i * 7 + k) % 50 for k in range(512)]} for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def pinned_cfg(path, n=12, frozen=False):
    """The repo config with the chunk file pinned (count n: tests use small files) and, if frozen, the pilot named."""
    return {**CFG, "freeze": {"stage_q3": frozen, "pilot_q3": "pilot-q3" if frozen else ""},
            "stage_q3": {**CFG["stage_q3"], "chunks_sha256": q3.file_sha256(path), "chunks": n}}


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    """The tests' chunk files hold 12 chunks, so a pinned count of 12 stands for the whole file (580 in the repo)."""
    monkeypatch.setattr(q3, "CHUNKS", 12)


def capture(out_path, gpu_lock_held, *, vllm="0.30.0"):
    rec = {"repo": {"commit": "c", "dirty": False}, "probe_errors": [], "runtime": {"vllm": {"vllm": vllm}},
           "os": {"system": "Linux", "wsl": True},
           "gpu": {"nvidia": [{"enforced_power_limit_w": 300.0, "power_default_limit_w": 300.0}]}}
    Path(out_path).write_text(json.dumps(rec))
    return rec


def score_rows(chunks, n, lp=-1.0):
    return [{"i": c["i"], "logprobs": [lp] * 255, "top1": [c["ids"][p] for p in range(257, 512)]} for c in chunks[:n]]


def fake_score(lp=-1.0, oom_for=None, log=ENGINE_LOG):
    def run_proc(argv, out, err, timeout, env):
        job = json.loads(Path(argv[argv.index("--job") + 1]).read_text())
        target = Path(argv[argv.index("--out") + 1])
        Path(out).write_text(log)
        if job["id"] == oom_for:
            Path(err).write_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
            return 1, 3.0
        chunks = [json.loads(x) for x in Path(argv[argv.index("--chunks") + 1]).read_text().splitlines()]
        value = lp[job["id"]] if isinstance(lp, dict) else lp
        target.write_text("".join(json.dumps(r) + "\n" for r in score_rows(chunks, job["n_chunks"], value)))
        Path(err).write_text("")
        return 0, 5.0
    return run_proc


def run(root, plan, sid, run_proc=None, **kw):
    args = dict(data_dir=root / "data", out_dir=root / "results" / "stage_q" / sid, capture_env=capture,
                session_id=sid, gpu_lock_held=True, repo_root=root, run_proc=run_proc or fake_score(),
                wait_idle=lambda: (True, {"ok": True}), pins=PINS, lock_check=lambda: True,
                toolchain_check=lambda env: [], base_env={"PATH": "/usr/bin", "HOME": "/h"},
                checkpoint_problems=lambda plan, pins: [])
    args.update(kw)
    return q3.run_q3(plan, **args)


def repo(tmp_path):
    root = tmp_path / "repo"
    chunk_file(root / "data" / "stage_q" / "q3-chunks-512.jsonl")
    return root, root / "data" / "stage_q" / "q3-chunks-512.jsonl"


def with_pilot(tmp_path):
    """A repo with the pinned 12-chunk file and a real passed pilot (fake vLLM) named by the frozen config."""
    root, chunks = repo(tmp_path)
    run(root, q3.build_plan(PINS, pinned_cfg(chunks), mode="pilot"), "pilot-q3")
    return root, pinned_cfg(chunks, frozen=True)


def test_plans_score_both_checkpoints_with_the_core_b_engine(tmp_path, monkeypatch):
    root, cfg = with_pilot(tmp_path)
    p = q3.build_plan(PINS, cfg, mode="formal", stage_q3_sha256="q" * 64, pins_sha256="p" * 64, root=root)
    w, b = p["jobs"]
    assert (w["id"], b["id"]) == ("score-w4a16", "score-bf16")
    assert w["model"] == {"repo": "RedHatAI/Qwen3.5-9B-quantized.w4a16",
                          "revision": "a398088c4228b0ae0c8c78df88fd1e4bf445f068"}
    assert b["model"] == {"repo": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
    for j in (w, b):
        e = j["engine"]
        assert e["language_model_only"] is True and e["dtype"] == "bfloat16" and e["enable_prefix_caching"] is False
        assert e["attention_backend"] == "FLASH_ATTN" and e["gdn_prefill_backend"] == "triton"
        assert e["gpu_memory_utilization"] == 0.91 and e["max_num_batched_tokens"] == 512
        assert j["n_chunks"] == 12 and j["env"]["HF_HUB_OFFLINE"] == "1"
    assert (w["engine"]["cpu_offload_gb"], b["engine"]["cpu_offload_gb"]) == (0, CFG["stage_q3"]["bf16_cpu_offload_gb"])
    assert p["chunks"]["chunks"] == 12 and p["harness_sha256"] == q3.harness_sha256()
    monkeypatch.setattr(q3, "CHUNKS", q3.WIKITEXT_TOKENS // q3.N_CTX)       # the repo config pins the real 580
    pilot = q3.build_plan(PINS, CFG, mode="pilot")
    assert all(j["n_chunks"] == q3.PILOT_CHUNKS for j in pilot["jobs"]) and pilot["chunks"]["sha256"] is None


def test_formal_plans_need_the_q3_freeze_the_pinned_chunks_and_the_pilot(tmp_path):
    with pytest.raises(q3.StageQ3Error, match="freeze.stage_q3"):
        q3.build_plan(PINS, CFG, mode="formal")
    root, cfg = with_pilot(tmp_path)
    unpinned = {**cfg, "stage_q3": {**cfg["stage_q3"], "chunks_sha256": ""}}
    with pytest.raises(q3.StageQ3Error, match="chunks_sha256"):
        q3.build_plan(PINS, unpinned, mode="formal", root=root)
    with pytest.raises(q3.StageQ3Error, match="pilot first"):
        q3.build_plan(PINS, cfg, mode="formal", root=tmp_path / "empty")


def test_the_freeze_needs_a_passed_pilot_of_this_harness_config_and_chunk_file(tmp_path):
    root, cfg = with_pilot(tmp_path)
    assert q3.freeze_evidence_problems(CFG, root, PINS) == []
    assert q3.freeze_evidence_problems(cfg, root, PINS) == []
    d = root / "results" / "stage_q" / "pilot-q3"
    s = json.loads((d / "session.json").read_text())
    for over, why in (({"outcomes": {"ok": 1, "oom": 1, "error": 0}}, "not a passed q3 pilot"),
                      ({"harness_sha256": "0" * 64}, "another harness"), ({"vllm_version": "0.29.0"}, "vLLM"),
                      ({"chunks_sha256": "0" * 64}, "another chunk file"),
                      ({"host": {"system": "Darwin", "wsl": False, "nvidia": False}}, "not a wsl2-cuda host")):
        (d / "session.json").write_text(json.dumps({**s, **over}))
        assert any(why in p for p in q3.freeze_evidence_problems(cfg, root, PINS)), over
    (d / "session.json").write_text(json.dumps(s))
    offload = {**cfg, "stage_q3": {**cfg["stage_q3"], "bf16_cpu_offload_gb": 6}}       # not what the pilot ran
    assert any("pilot jobs" in p for p in q3.freeze_evidence_problems(offload, root, PINS))


# ---------------------------------------------------------------- the runner

def test_score_outputs_are_checked_against_the_chunks(tmp_path):
    chunks = chunk_file(tmp_path / "c.jsonl")
    good = score_rows(chunks, 10)
    assert q3.output_problems(good, chunks, 10) == []
    assert q3.output_problems(good[:9], chunks, 10)
    assert q3.output_problems([{**good[0], "i": 3}] + good[1:], chunks, 10)
    assert q3.output_problems([{**good[0], "logprobs": [0.5] * 255}] + good[1:], chunks, 10)
    assert q3.output_problems([{**good[0], "logprobs": [float("nan")] * 255}] + good[1:], chunks, 10)
    assert q3.output_problems([{**good[0], "top1": [1] * 254}] + good[1:], chunks, 10)


def test_a_pilot_session_scores_both_checkpoints_in_core_bs_env(tmp_path):
    root, chunks = repo(tmp_path)
    plan = q3.build_plan(PINS, pinned_cfg(chunks), mode="pilot")
    envs = []

    def record(argv, out, err, timeout, env):
        envs.append(env)
        return fake_score()(argv, out, err, timeout, env)

    s = run(root, plan, "s", run_proc=record)
    out = root / "results" / "stage_q" / "s"
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in rows] == ["ok", "ok"] and s["outcomes"] == {"ok": 2, "oom": 0, "error": 0}
    assert s["part"] == "q3" and s["harness_sha256"] == q3.harness_sha256() and s["vllm_version"] == "0.30.0"
    assert rows[0]["scores"].startswith("raw/") and (out / rows[0]["scores"]).is_file()      # token-level: raw only
    assert rows[0]["chunks_sha256"] == q3.file_sha256(chunks)
    assert rows[0]["chunk_nll"] == [1.0] * 10 and rows[0]["n_tokens"] == 10 * 255
    assert rows[0]["engine_facts"]["attention_backend"] == "FLASH_ATTN" and (out / "raw_manifest.json").is_file()
    assert all(e["CUDA_HOME"] and e["PATH"].startswith(q3.bp.VENV_BIN) and e["HF_HUB_OFFLINE"] == "1" for e in envs)
    assert rows[0]["env"]["VLLM_GDN_DECODE_KERNEL"] == "triton"


def test_a_job_whose_engine_is_not_core_bs_is_an_error(tmp_path):
    root, chunks = repo(tmp_path)
    plan = q3.build_plan(PINS, pinned_cfg(chunks), mode="pilot")
    s = run(root, plan, "s", run_proc=fake_score(log=ENGINE_LOG.replace("FLASH_ATTN", "TRITON_ATTN")))
    assert s["outcomes"] == {"ok": 0, "oom": 0, "error": 2}


def test_an_oom_is_recorded_as_oom_and_a_timeout_as_an_error(tmp_path):
    root, chunks = repo(tmp_path)
    plan = q3.build_plan(PINS, pinned_cfg(chunks), mode="pilot")
    assert run(root, plan, "s", run_proc=fake_score(oom_for="score-bf16"))["outcomes"] == {"ok": 1, "oom": 1, "error": 0}

    def hung(argv, out, err, timeout, env):
        Path(out).write_text("")
        Path(err).write_text("")
        return None, timeout

    s = run(root, plan, "t", run_proc=hung)
    rows = [json.loads(x) for x in (root / "results" / "stage_q" / "t" / "rows.jsonl").read_text().splitlines()]
    assert s["outcomes"]["error"] == 2 and all("timeout" in r["reason"] for r in rows)


def test_the_default_runner_kills_the_whole_group_on_a_timeout(tmp_path):
    t0 = time.monotonic()
    code, wall = q3.default_run_proc([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "o",
                                     tmp_path / "e", 0.5, {"PATH": "/usr/bin:/bin"})
    assert code is None and time.monotonic() - t0 < 10


def test_the_runner_refuses_without_the_lock_outside_results_with_other_chunks_or_a_bad_toolchain(tmp_path):
    root, chunks = repo(tmp_path)
    plan = q3.build_plan(PINS, pinned_cfg(chunks), mode="pilot")
    for kw in ({"gpu_lock_held": False}, {"lock_check": lambda: False}):
        with pytest.raises(q3.StageQ3Error, match="GPU lock"):
            run(root, plan, "s", **kw)
    with pytest.raises(q3.StageQ3Error, match="results/stage_q"):
        run(root, plan, "s", out_dir=tmp_path / "elsewhere")
    with pytest.raises(q3.StageQ3Error, match="toolchain"):
        run(root, plan, "s", toolchain_check=lambda env: ["no ninja"])
    assert not (root / "results" / "stage_q" / "s").exists()
    root2, cfg2 = with_pilot(tmp_path / "b")
    formal = q3.build_plan(PINS, cfg2, mode="formal", root=root2)
    chunk_file(root2 / "data" / "stage_q" / "q3-chunks-512.jsonl", n=13)           # not the pinned file any more
    with pytest.raises(q3.StageQ3Error, match="pinned one"):
        run(root2, formal, "f", expected_plan=formal)


def test_a_formal_run_needs_build_plans_plan_the_pinned_vllm_and_the_pinned_shards(tmp_path):
    root, cfg = with_pilot(tmp_path)
    plan = q3.build_plan(PINS, cfg, mode="formal", root=root)
    with pytest.raises(q3.StageQ3Error, match="exactly build_plan's"):
        run(root, {**plan, "jobs": plan["jobs"][:1]}, "f", expected_plan=plan)
    with pytest.raises(q3.StageQ3Error, match="not the pinned 0.30.0"):
        run(root, plan, "g", expected_plan=plan, capture_env=lambda out_path, gpu_lock_held: capture(
            out_path, gpu_lock_held, vllm="0.29.0"))
    with pytest.raises(q3.StageQ3Error, match="sha256"):
        run(root, plan, "h", expected_plan=plan, checkpoint_problems=lambda plan, pins: ["x: sha256 bad"])
    s = run(root, plan, "i", expected_plan=plan)
    assert s["inputs_verified"] is True and s["env_valid"] is True


def test_the_bf16_shards_are_checked_against_pins(tmp_path):
    want = next(m for m in PINS["core_a"]["models"] if m["size"] == "9B")["source_shards"]
    for name in want:
        (tmp_path / name).write_text("x")
    assert q3.shard_problems(tmp_path, want, hash_file=lambda p: want[Path(p).name]) == []
    assert len(q3.shard_problems(tmp_path, want)) == 4                          # the files hold "x"


def test_a_job_waits_for_an_idle_gpu(tmp_path):
    root, chunks = repo(tmp_path)
    plan = q3.build_plan(PINS, pinned_cfg(chunks), mode="pilot")
    s = run(root, plan, "s", wait_idle=lambda: (False, {"problems": ["busy"]}))
    out = root / "results" / "stage_q" / "s"
    assert s["outcomes"] == {"ok": 0, "oom": 0, "error": 2}
    assert all("gpu_not_idle" in json.loads(x)["reason"] for x in (out / "rows.jsonl").read_text().splitlines())


# ---------------------------------------------------------------- the analysis

def test_the_analysis_gives_ppl_paired_delta_nll_and_top1_agreement():
    b = [{"i": i, "logprobs": [-2.0] * 255, "top1": [1] * 255} for i in range(20)]
    w = [{"i": i, "logprobs": [-2.01] * 255, "top1": [1] * 254 + [2]} for i in range(20)]
    a = q3.analyze(w, b, n_boot=200, seed=1)
    assert a["ppl"]["bf16"] == pytest.approx(math.exp(2.0)) and a["ppl"]["w4a16"] == pytest.approx(math.exp(2.01))
    assert a["ppl_ratio_minus_1"] == pytest.approx(math.exp(0.01) - 1)
    assert a["delta_nll"]["mean"] == pytest.approx(0.01) and a["delta_nll"]["n_chunks"] == 20
    assert a["delta_nll"]["ci95"][0] == pytest.approx(0.01) and a["top1_agreement"] == pytest.approx(254 / 255)
    assert a["H-Q3"]["verdict"] == "supported"
    w2 = [{**r, "logprobs": [-2.03] * 255} for r in w]
    assert q3.analyze(w2, b, n_boot=200, seed=1)["H-Q3"]["verdict"] == "not supported"
    assert q3.analyze(None, b)["H-Q3"]["verdict"] == "NT"
    with pytest.raises(q3.StageQ3Error, match="same chunks"):
        q3.analyze(w[:19], b)


def formal_session(tmp_path, lp=None, oom_for=None, sid="formal-q3"):
    root, cfg = with_pilot(tmp_path)
    plan = q3.build_plan(PINS, cfg, mode="formal", stage_q3_sha256="q" * 64, pins_sha256=PINS_SHA, root=root)
    out = root / "results" / "stage_q" / sid
    run(root, plan, sid, expected_plan=plan,
        run_proc=fake_score(lp or {"score-w4a16": -2.01, "score-bf16": -2.0}, oom_for=oom_for))
    return root, cfg, out


def analyze_session(root, cfg, out, **kw):
    return q3.analyze_session(out, cfg=cfg, stage_q3_sha256="q" * 64, pins_sha256=PINS_SHA, pins=PINS, root=root, **kw)


def test_a_formal_session_is_analyzed_from_its_rows_and_raw_scores(tmp_path):
    root, cfg, out = formal_session(tmp_path)
    a = analyze_session(root, cfg, out, n_boot=100)
    assert a["H-Q3"]["verdict"] == "supported" and a["ppl"]["bf16"] == pytest.approx(math.exp(2.0))
    assert a["session"] == "formal-q3" and a["harness_sha256"] == q3.harness_sha256()
    assert a["delta_nll"]["n_chunks"] == 12 and a["top1_agreement"] == 1.0


def test_an_oom_leaves_h_q3_nt(tmp_path):
    root, cfg, out = formal_session(tmp_path, oom_for="score-bf16")
    assert analyze_session(root, cfg, out)["H-Q3"]["verdict"] == "NT"


def test_only_the_first_valid_formal_q3_session_after_one_void_attempt_is_analyzed(tmp_path):
    root, cfg, out = formal_session(tmp_path)
    later = root / "results" / "stage_q" / "formal-q3-b"
    shutil.copytree(out, later)
    meta = json.loads((later / "session.json").read_text())
    (later / "session.json").write_text(json.dumps({**meta, "started_at": "9999"}))
    with pytest.raises(q3.StageQ3Error, match="earlier formal q3 attempts"):
        analyze_session(root, cfg, later)
    (out / "session.json").write_text(json.dumps({**json.loads((out / "session.json").read_text()), "env_valid": False}))
    assert analyze_session(root, cfg, later, n_boot=50)["session"] == "formal-q3-b"     # after one void attempt
    log = root / "docs" / "decisions" / "999-q3.md"
    log.parent.mkdir(parents=True)
    log.write_text("formal-q3 is superseded by formal-q3-b.\n")
    (out / "session.json").write_text(json.dumps({**json.loads((out / "session.json").read_text()), "env_valid": True}))
    meta = json.loads((later / "session.json").read_text())                          # what formal-q3-b's run recorded
    (later / "session.json").write_text(json.dumps({**meta, "superseded": {"formal-q3": "docs/decisions/999-q3.md"}}))
    a = analyze_session(root, cfg, later, n_boot=50, tracked=lambda p: True)
    assert a["session"] == "formal-q3-b"
    q = q3.analyze_q3(cfg=cfg, stage_q3_sha256="q" * 64, pins_sha256=PINS_SHA, pins=PINS, root=root, n_boot=50,
                      tracked=lambda p: True)
    assert q["session"] == "formal-q3-b" and q["q3"]["superseded"] == {"formal-q3": "formal-q3-b"}
    with pytest.raises(q3.StageQ3Error, match="not committed"):
        analyze_session(root, cfg, later, n_boot=50, tracked=lambda p: False)


def test_the_q3_analysis_finds_its_session_or_reports_nt(tmp_path):
    root, cfg = with_pilot(tmp_path)
    kw = dict(cfg=cfg, stage_q3_sha256="q" * 64, pins_sha256=PINS_SHA, pins=PINS, root=root, n_boot=50)
    with pytest.raises(q3.StageQ3Error, match="freeze.stage_q3"):                    # no NT before the freeze either
        q3.analyze_q3(**{**kw, "cfg": {**cfg, "freeze": {**cfg["freeze"], "stage_q3": False}}})
    assert q3.analyze_q3(**kw)["H-Q3"] == {"verdict": "NT", "why": "not run"}
    plan = q3.build_plan(PINS, cfg, mode="formal", stage_q3_sha256="q" * 64, pins_sha256=PINS_SHA, root=root)
    busy = iter([(True, {}), (False, {"problems": ["not idle"]})])
    s = run(root, plan, "v1", expected_plan=plan, wait_idle=lambda: next(busy),
            run_proc=fake_score({"score-w4a16": -2.01, "score-bf16": -2.0}))
    assert s["env_valid"] is False                                                    # gpu_not_idle voids it (031)
    out = q3.analyze_q3(**kw)
    assert out["H-Q3"]["verdict"] == "NT" and out["q3"]["attempts"] == ["v1"]
    run(root, plan, "v2", expected_plan=plan, run_proc=fake_score({"score-w4a16": -2.01, "score-bf16": -2.0}))
    out = q3.analyze_q3(**kw)
    assert out["session"] == "v2" and out["q3"]["void_before"] == ["v1"] and out["H-Q3"]["verdict"] == "supported"


def test_a_formal_q3_run_counts_its_attempts(tmp_path):
    root, cfg = with_pilot(tmp_path)
    plan = q3.build_plan(PINS, cfg, mode="formal", stage_q3_sha256="q" * 64, pins_sha256=PINS_SHA, root=root)
    run(root, plan, "a", expected_plan=plan, run_proc=fake_score({"score-w4a16": -2.01, "score-bf16": -2.0}))
    assert json.loads((root / "results" / "stage_q" / "a" / "session.json").read_text())["status"] == "finished"
    with pytest.raises(q3.StageQ3Error, match="already has formal attempts"):
        run(root, plan, "b", expected_plan=plan, run_proc=fake_score({"score-w4a16": -2.01, "score-bf16": -2.0}))
    with pytest.raises(q3.StageQ3Error, match="results/stage_q/<session id>"):
        run(root, plan, "x", out_dir=root / "results" / "stage_q" / "n" / "x", expected_plan=plan)


def test_the_test_set_is_encoded_without_its_last_newline():
    seen = []
    assert q3.encode_text(b"a b \n \n", lambda t: seen.append(t) or [1, 2]) == [1, 2] and seen == ["a b \n "]
    assert q3.encode_text(b"abc", lambda t: seen.append(t) or [3]) == [3] and seen[-1] == "abc"


def test_the_analysis_refuses_what_040_does_not_allow(tmp_path):
    root, cfg, out = formal_session(tmp_path)
    with pytest.raises(q3.StageQ3Error, match="freeze.stage_q3"):
        analyze_session(root, {**cfg, "freeze": {**cfg["freeze"], "stage_q3": False}}, out)
    with pytest.raises(q3.StageQ3Error, match="configs/pins.toml"):
        q3.analyze_session(out, cfg=cfg, stage_q3_sha256="q" * 64, pins_sha256="0" * 64, pins=PINS, root=root)
    meta = json.loads((out / "session.json").read_text())
    for over, why in (({"mode": "pilot"}, "not formal"), ({"vllm_version": "0.29.0"}, "pinned vLLM"),
                      ({"host": {"system": "Darwin", "wsl": False, "nvidia": False}}, "not a wsl2-cuda host")):
        (out / "session.json").write_text(json.dumps({**meta, **over}))
        with pytest.raises(q3.StageQ3Error, match=why):
            analyze_session(root, cfg, out)
    (out / "session.json").write_text(json.dumps(meta))
    scores = out / "raw" / "score-bf16.scores.jsonl"
    scores.write_text(scores.read_text().replace("-2.0", "-1.0"))
    with pytest.raises(q3.StageQ3Error, match="raw_manifest"):
        analyze_session(root, cfg, out)
    scores.unlink()
    with pytest.raises(q3.StageQ3Error, match="fetch"):
        analyze_session(root, cfg, out)


def test_a_log_counts_as_committed_on_the_wsl_copy_through_its_deploy_manifest(tmp_path):
    log = tmp_path / "docs" / "decisions" / "999-x.md"
    log.parent.mkdir(parents=True)
    log.write_text("a supersedes b\n")
    assert q3.git_tracked(log, root=tmp_path) is False                                # no .git, no manifest
    sha = hashlib.sha256(log.read_bytes()).hexdigest()
    (tmp_path / "DEPLOY_MANIFEST.json").write_text(json.dumps({"files": {"docs/decisions/999-x.md": sha}}))
    assert q3.git_tracked(log, root=tmp_path) is True
    log.write_text("edited after the deploy\n")
    assert q3.git_tracked(log, root=tmp_path) is False



def test_an_interrupt_before_the_first_job_is_refused_not_an_attempt(tmp_path):
    root, cfg = with_pilot(tmp_path)
    plan = q3.build_plan(PINS, cfg, mode="formal", root=root)

    def interrupted(**kw):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(root, plan, "f", expected_plan=plan, capture_env=interrupted)
    s = json.loads((root / "results" / "stage_q" / "f" / "session.json").read_text())
    assert s["status"] == "refused" and "KeyboardInterrupt" in s["refused"][0] and q3.formal_attempts(root) == []


def test_a_sigterm_or_sighup_becomes_systemexit():
    with pytest.raises(SystemExit, match="143"):
        q3._exit_on_signal(15, None)


@pytest.mark.parametrize("exc, how", [(SystemExit(143), "interrupted"), (OSError("x"), "failed")])
def test_refused_why_tells_an_interrupt_from_a_failure(exc, how):
    assert q3.refused_why(exc).startswith(f"{how} before the first job: {type(exc).__name__}")


def test_a_runner_stopped_by_sigterm_kills_the_scorers_whole_group(tmp_path):
    import os
    import signal
    import time
    pidfile = tmp_path / "bg.pid"
    argv = ["sh", "-c", f"sleep 60 & echo $! > {pidfile}; wait"]       # a scorer with a child of its own
    previous = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(SystemExit(143)))
    try:
        with q3.signals_exit():
            signal.setitimer(signal.ITIMER_REAL, 1.0)                 # stands in for the SIGTERM
            with pytest.raises(SystemExit):
                q3.default_run_proc(argv, tmp_path / "o", tmp_path / "e", 30, dict(os.environ))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    pid = int(pidfile.read_text())
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"the scorer's child {pid} outlived the runner")
