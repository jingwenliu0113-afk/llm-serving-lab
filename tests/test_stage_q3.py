"""Tests for quality/stage_q3.py (decision 040 Q3): W4A16 against BF16 in vLLM on WikiText-2 512-token chunks scored
as llama-perplexity scores them; the chunk file, plans, the prompt-logprob parser, the runner and the analysis."""

import json
import math
import tomllib
from pathlib import Path

import pytest

import stage_q3 as q3

ROOT = Path(__file__).resolve().parent.parent
PINS = tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
CFG = tomllib.loads((ROOT / "configs" / "stage_q3.toml").read_text())
FROZEN = {**CFG, "freeze": {"stage_q3": True, "pilot_q3": "pilot-q3"},
          "stage_q3": {**CFG["stage_q3"], "chunks_sha256": "c" * 64, "chunks": 580}}


def test_chunks_are_cut_as_llama_perplexity_cuts_them(tmp_path):
    assert q3.cut_chunks(list(range(1030))) == [list(range(512)), list(range(512, 1024))]
    info = q3.build_chunks("a b c", lambda text: list(range(1100)), tmp_path / "c.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert [r["i"] for r in rows] == [0, 1] and rows[1]["ids"][0] == 512
    assert info["tokens"] == 1100 and info["chunks"] == 2 and len(info["sha256"]) == 64
    assert q3.build_chunks("a b c", lambda text: list(range(1100)), tmp_path / "d.jsonl")["sha256"] == info["sha256"]


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
    with pytest.raises(q3.StageQ3Error, match="position 257"):
        q3.scored(ids, [None] + [{"8": {"logprob": -1.0, "rank": 1}} for _ in range(511)])


def test_plans_score_both_checkpoints_with_the_core_b_engine():
    p = q3.build_plan(PINS, FROZEN, mode="formal", stage_q3_sha256="q" * 64, pins_sha256="p" * 64)
    w, b = p["jobs"]
    assert (w["id"], b["id"]) == ("score-w4a16", "score-bf16")
    assert w["model"] == {"repo": "RedHatAI/Qwen3.5-9B-quantized.w4a16",
                          "revision": "a398088c4228b0ae0c8c78df88fd1e4bf445f068"}
    assert b["model"] == {"repo": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
    for j in (w, b):
        e = j["engine"]
        assert e["language_model_only"] is True and e["dtype"] == "bfloat16" and e["enable_prefix_caching"] is False
        assert e["gpu_memory_utilization"] == 0.91 and e["max_num_batched_tokens"] == 512
        assert j["n_chunks"] == 580 and j["env"]["HF_HUB_OFFLINE"] == "1"
    assert (w["engine"]["cpu_offload_gb"], b["engine"]["cpu_offload_gb"]) == (0, CFG["stage_q3"]["bf16_cpu_offload_gb"])
    assert p["chunks"] == {"path": "{data}/stage_q/q3-chunks-512.jsonl", "sha256": "c" * 64, "chunks": 580}
    assert p["harness_sha256"] == q3.harness_sha256() and p["stage_q3_sha256"] == "q" * 64
    pilot = q3.build_plan(PINS, CFG, mode="pilot")
    assert all(j["n_chunks"] == q3.PILOT_CHUNKS for j in pilot["jobs"]) and pilot["chunks"]["sha256"] is None


def test_formal_plans_need_the_q3_freeze_and_the_pinned_chunks():
    with pytest.raises(q3.StageQ3Error, match="freeze.stage_q3"):
        q3.build_plan(PINS, CFG, mode="formal")
    unpinned = {**FROZEN, "stage_q3": {**FROZEN["stage_q3"], "chunks_sha256": ""}}
    with pytest.raises(q3.StageQ3Error, match="chunks_sha256"):
        q3.build_plan(PINS, unpinned, mode="formal")


def test_the_repo_config_is_well_formed_and_not_yet_frozen():
    assert q3.config_problems(CFG) == [] and CFG["freeze"]["stage_q3"] is False
    assert q3.config_problems({**CFG, "freeze": {"stage_q3": True, "pilot_q3": ""}})
    assert q3.config_problems({**CFG, "stage_q3": {**CFG["stage_q3"], "bf16_cpu_offload_gb": -1}})


def test_the_freeze_needs_a_passed_pilot_of_this_harness(tmp_path):
    assert q3.freeze_evidence_problems(CFG, tmp_path) == []
    assert q3.freeze_evidence_problems(FROZEN, tmp_path)
    d = tmp_path / "results" / "stage_q" / "pilot-q3"
    d.mkdir(parents=True)
    s = {"mode": "pilot", "part": "q3", "outcomes": {"ok": 2, "oom": 0, "error": 0},
         "harness_sha256": q3.harness_sha256()}
    (d / "session.json").write_text(json.dumps(s))
    assert q3.freeze_evidence_problems(FROZEN, tmp_path) == []
    (d / "session.json").write_text(json.dumps({**s, "outcomes": {"ok": 1, "oom": 1, "error": 0}}))
    assert q3.freeze_evidence_problems(FROZEN, tmp_path)
    (d / "session.json").write_text(json.dumps({**s, "harness_sha256": "0" * 64}))
    assert any("another harness" in p for p in q3.freeze_evidence_problems(FROZEN, tmp_path))


def chunk_file(path, n=12):
    rows = [{"i": i, "ids": [(i * 7 + k) % 50 for k in range(512)]} for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def score_rows(chunks, n, lp=-1.0, top_same=True):
    return [{"i": c["i"], "logprobs": [lp] * 255, "top1": [c["ids"][p] if top_same else -1 for p in range(257, 512)]}
            for c in chunks[:n]]


def test_score_outputs_are_checked_against_the_chunks(tmp_path):
    chunks = chunk_file(tmp_path / "c.jsonl")
    good = score_rows(chunks, 10)
    assert q3.output_problems(good, chunks, 10) == []
    assert q3.output_problems(good[:9], chunks, 10)
    assert q3.output_problems([{**good[0], "i": 3}] + good[1:], chunks, 10)
    assert q3.output_problems([{**good[0], "logprobs": [0.5] * 255}] + good[1:], chunks, 10)
    assert q3.output_problems([{**good[0], "logprobs": [float("nan")] * 255}] + good[1:], chunks, 10)
    assert q3.output_problems([{**good[0], "top1": [1] * 254}] + good[1:], chunks, 10)


def capture(out_path, gpu_lock_held):
    rec = {"repo": {"commit": "c", "dirty": False}, "probe_errors": [], "runtime": {"vllm": {"version": "0.30.0"}},
           "gpu": {"nvidia": [{"enforced_power_limit_w": 300.0, "power_default_limit_w": 300.0}]}}
    Path(out_path).write_text(json.dumps(rec))
    return rec


def run(tmp_path, plan, run_proc, **kw):
    data = tmp_path / "data"
    chunks = chunk_file(data / "stage_q" / "q3-chunks-512.jsonl")
    out = tmp_path / "results" / "stage_q" / "s"
    args = dict(data_dir=data, out_dir=out, capture_env=capture, session_id="s", gpu_lock_held=True,
                repo_root=tmp_path, run_proc=run_proc, wait_idle=lambda: (True, {"ok": True}))
    args.update(kw)
    return q3.run_q3(plan, **args), out, chunks


def fake_score(chunks_lp=-1.0, oom_for=None):
    def run_proc(argv, out, err, timeout, env):
        job = json.loads(Path(argv[argv.index("--job") + 1]).read_text())
        target = Path(argv[argv.index("--out") + 1])
        if job["id"] == oom_for:
            Path(out).write_text("")
            Path(err).write_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
            return 1, 3.0
        chunks = [json.loads(x) for x in Path(argv[argv.index("--chunks") + 1]).read_text().splitlines()]
        lp = chunks_lp[job["id"]] if isinstance(chunks_lp, dict) else chunks_lp
        target.write_text("".join(json.dumps(r) + "\n" for r in score_rows(chunks, job["n_chunks"], lp)))
        Path(out).write_text("")
        Path(err).write_text("")
        return 0, 5.0
    return run_proc


def test_a_pilot_session_scores_both_checkpoints(tmp_path):
    plan = q3.build_plan(PINS, CFG, mode="pilot")
    s, out, chunks = run(tmp_path, plan, fake_score())
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in rows] == ["ok", "ok"] and s["outcomes"] == {"ok": 2, "oom": 0, "error": 0}
    assert s["part"] == "q3" and s["harness_sha256"] == q3.harness_sha256()
    assert rows[0]["scores"].startswith("raw/") and (out / rows[0]["scores"]).is_file()      # token-level: raw only
    assert rows[0]["chunks_sha256"] == q3.file_sha256(tmp_path / "data" / "stage_q" / "q3-chunks-512.jsonl")
    assert rows[0]["chunk_nll"] == [1.0] * 10 and rows[0]["n_tokens"] == 10 * 255
    assert (out / "raw_manifest.json").is_file()


def test_an_oom_is_recorded_as_oom_and_the_other_job_still_runs(tmp_path):
    plan = q3.build_plan(PINS, CFG, mode="pilot")
    s, out, _ = run(tmp_path, plan, fake_score(oom_for="score-bf16"))
    assert s["outcomes"] == {"ok": 1, "oom": 1, "error": 0}


def test_the_runner_refuses_without_the_lock_outside_results_or_with_other_chunks(tmp_path):
    plan = q3.build_plan(PINS, CFG, mode="pilot")
    with pytest.raises(q3.StageQ3Error, match="GPU lock"):
        run(tmp_path, plan, fake_score(), gpu_lock_held=False)
    with pytest.raises(q3.StageQ3Error, match="results/stage_q"):
        run(tmp_path / "x", plan, fake_score(), out_dir=tmp_path / "elsewhere")
    formal = q3.build_plan(PINS, FROZEN, mode="formal")
    with pytest.raises(q3.StageQ3Error, match="chunks"):
        run(tmp_path / "y", formal, fake_score())                     # the file is not the pinned one


def test_a_job_waits_for_an_idle_gpu(tmp_path):
    plan = q3.build_plan(PINS, CFG, mode="pilot")
    s, out, _ = run(tmp_path, plan, fake_score(), wait_idle=lambda: (False, {"problems": ["busy"]}))
    assert s["outcomes"] == {"ok": 0, "oom": 0, "error": 2}
    assert all("gpu_not_idle" in json.loads(x)["reason"] for x in (out / "rows.jsonl").read_text().splitlines())


def test_the_analysis_gives_ppl_paired_delta_nll_and_top1_agreement():
    chunks = [{"i": i, "ids": [1] * 512} for i in range(20)]
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


# ---------------------------------------------------------------- the analysis of a formal session (040 凍結)

import hashlib  # noqa: E402

PINS_SHA = hashlib.sha256((ROOT / "configs" / "pins.toml").read_bytes()).hexdigest()


def formal_session(tmp_path, lp=None, oom_for=None):
    """A frozen config pinned to a 12-chunk file, its passed pilot and a formal session run with fake scores."""
    root = tmp_path / "repo"
    chunks = chunk_file(root / "data" / "stage_q" / "q3-chunks-512.jsonl")
    cfg = {**CFG, "freeze": {"stage_q3": True, "pilot_q3": "pilot-q3"},
           "stage_q3": {**CFG["stage_q3"], "chunks_sha256": q3.file_sha256(
               root / "data" / "stage_q" / "q3-chunks-512.jsonl"), "chunks": len(chunks)}}
    pilot = root / "results" / "stage_q" / "pilot-q3"
    pilot.mkdir(parents=True)
    (pilot / "session.json").write_text(json.dumps({"mode": "pilot", "part": "q3", "harness_sha256": q3.harness_sha256(),
                                                    "outcomes": {"ok": 2, "oom": 0, "error": 0}}))
    plan = q3.build_plan(PINS, cfg, mode="formal", stage_q3_sha256="q" * 64, pins_sha256=PINS_SHA)
    out = root / "results" / "stage_q" / "formal-q3"
    q3.run_q3(plan, data_dir=root / "data", out_dir=out, capture_env=capture, session_id="formal-q3",
              gpu_lock_held=True, repo_root=root, wait_idle=lambda: (True, {}),
              run_proc=fake_score(lp or {"score-w4a16": -2.01, "score-bf16": -2.0}, oom_for=oom_for))
    return root, cfg, out


def analyze_session(root, cfg, out, **kw):
    return q3.analyze_session(out, cfg=cfg, stage_q3_sha256="q" * 64, pins_sha256=PINS_SHA, root=root, **kw)


def test_a_formal_session_is_analyzed_from_its_rows_and_raw_scores(tmp_path):
    root, cfg, out = formal_session(tmp_path)
    a = analyze_session(root, cfg, out, n_boot=100)
    assert a["H-Q3"]["verdict"] == "supported" and a["ppl"]["bf16"] == pytest.approx(math.exp(2.0))
    assert a["session"] == "formal-q3" and a["harness_sha256"] == q3.harness_sha256()
    assert a["delta_nll"]["n_chunks"] == 12 and a["top1_agreement"] == 1.0


def test_an_oom_leaves_h_q3_nt(tmp_path):
    root, cfg, out = formal_session(tmp_path, oom_for="score-bf16")
    assert analyze_session(root, cfg, out)["H-Q3"]["verdict"] == "NT"


def test_the_analysis_refuses_what_040_does_not_allow(tmp_path):
    root, cfg, out = formal_session(tmp_path)
    with pytest.raises(q3.StageQ3Error, match="freeze.stage_q3"):
        analyze_session(root, {**cfg, "freeze": {**cfg["freeze"], "stage_q3": False}}, out)
    with pytest.raises(q3.StageQ3Error, match="configs/pins.toml"):
        q3.analyze_session(out, cfg=cfg, stage_q3_sha256="q" * 64, pins_sha256="0" * 64, root=root)
    meta = json.loads((out / "session.json").read_text())
    (out / "session.json").write_text(json.dumps({**meta, "mode": "pilot"}))
    with pytest.raises(q3.StageQ3Error, match="not formal"):
        analyze_session(root, cfg, out)
    (out / "session.json").write_text(json.dumps(meta))
    scores = out / "raw" / "score-bf16.scores.jsonl"
    scores.write_text(scores.read_text().replace("-2.0", "-1.0"))
    with pytest.raises(q3.StageQ3Error, match="raw_manifest"):
        analyze_session(root, cfg, out)
    scores.unlink()
    with pytest.raises(q3.StageQ3Error, match="fetch"):
        analyze_session(root, cfg, out)
