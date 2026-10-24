"""Decision 044: the gateway settings the Stage V harness renders (stage_v_plan.gateway_settings through its nested TOML
writer) are settings capserve loads and serves on each point's profile, with the /readyz limits, queue settings and
candidate policy the plan expects, and the real gateway's /readyz body gives, through the runner's readyz_tuple, the
tuple the runner's readyz_want freezes for that point (實作補記六). Research never imports capserve, so this
cross-check lives on the product side (a service test may import the research harness)."""

import asyncio
import hashlib
import json
import logging
import tomllib
from pathlib import Path

import pytest

from capserve import config as cfgmod
from capserve import gateway_path as gp
from capserve import profile as cp
from capserve.app import create_app
from capserve.tokens import LocalTokenizer
from fakes.fake_tokenizer import WordsHF
from fakes.fake_vllm import MODEL, FakeConfig
from fakes.servers import stack

svp = pytest.importorskip("stage_v_plan")
rsv = pytest.importorskip("run_stage_v")
sva = pytest.importorskip("stage_v_analysis")

ROOT = Path(__file__).resolve().parents[2]


def build(session: str) -> dict:
    raw = {n: (ROOT / "configs" / f"{n}.toml").read_bytes() for n in ("stage_v", "stage_d", "pins")}
    cfg, sd_cfg, pins = (tomllib.loads(raw[n].decode()) for n in ("stage_v", "stage_d", "pins"))
    return svp.build_plan(cfg, pins, sd_cfg, session_name=session, pins_sha256=hashlib.sha256(raw["pins"]).hexdigest(),
                          stage_v_sha256=hashlib.sha256(raw["stage_v"]).hexdigest(),
                          stage_d_sha256=hashlib.sha256(raw["stage_d"]).hexdigest())


@pytest.mark.parametrize("session", ["mini", "V-ADM", "V-CLS"])
def test_every_points_settings_load_and_serve_on_its_profile(tmp_path, session):
    plan = build(session)
    for name, x in plan["points"].items():
        path = tmp_path / f"{name}.toml"
        path.write_text(svp.toml_dumps(x["gateway"]))
        settings = cfgmod.load_settings(path)
        prof = cp.load_profile(ROOT / x["profile"], allow_provisional=True)
        assert prof.verified is (not x["provisional"]), name
        classes = cfgmod.served_classes(settings, prof, gateway_path_sha256=gp.gateway_path_sha256(), study=True)
        assert [c.name for c in classes] == x["classes"], name
        limits = cfgmod.served_limits(settings, classes)
        assert {k: limits[k] for k in x["limits"]} == x["limits"], name
        assert cfgmod.queue_settings(settings) == x["queue_settings"], name
        want = (["admission"] if x["admission"] else []) + (["prefill_pacing"] if x["pacing"] else [])
        assert cfgmod.candidate_policies(settings) == want, name
        assert (settings.host, settings.port, settings.tokenizer) == ("127.0.0.1", 8080, "local")
        assert settings.backend_url == "http://127.0.0.1:8000" and settings.drain_timeout_s == 5


def test_no_candidate_policy_of_the_study_deploys():
    """044 item 5: the research settings serve a candidate policy only as a study (capserve serve
    --allow-provisional); deploy (study=False) refuses each of them."""
    plan = build("V-ADM")
    for name in ("P-U2K", "W-90"):
        x = plan["points"][name]
        settings = cfgmod.Settings.model_validate(x["gateway"])
        prof = cp.load_profile(ROOT / x["profile"], allow_provisional=True)
        with pytest.raises(cp.ProfileError, match="deploy never serves a candidate policy"):
            cfgmod.served_classes(settings, prof, gateway_path_sha256=gp.gateway_path_sha256(), study=False)


@pytest.mark.parametrize("session", ["mini", "V-ADM", "V-CLS"])
def test_the_real_gateways_readyz_gives_each_points_frozen_tuple(tmp_path, session):
    """044 實作補記六: for every point of every session, the gateway capserve builds from the point's rendered settings
    and profile (a local tokenizer injected: no snapshot on the Mac) reports, through run_stage_v.readyz_tuple, exactly
    run_stage_v.readyz_want; the runner's fakes (tests/test_run_stage_v) only model that body."""
    plan = build(session)
    sha = svp.gateway_path_sha256()
    assert sha == gp.gateway_path_sha256()
    for name, x in plan["points"].items():
        path = tmp_path / f"{name}.toml"
        path.write_text(svp.toml_dumps(x["gateway"]))
        prof = cp.load_profile(ROOT / x["profile"], allow_provisional=True)
        gateway = create_app(cfgmod.load_settings(path), prof, local_tokenizer=LocalTokenizer(WordsHF()))
        profile_id = json.loads((ROOT / x["profile"]).read_text())["profile_id"]
        want = rsv.readyz_want(plan, name, profile_id=profile_id, gateway_path_sha256=sha)
        assert rsv.readyz_tuple(gateway.readiness()) == want, name
        assert None not in (want["profile"]["id"], want["serving_status"], want["limits"], want["queue"]), name


def _bench_body(prompt: str, out: int) -> dict:
    """vLLM 0.30 bench's completions payload (endpoint_request_func.async_request_openai_completions with
    --ignore-eos --temperature 0): repetition_penalty 1.0 and logprobs null are dropped by the gateway as defaults."""
    return {"model": MODEL, "prompt": prompt, "repetition_penalty": 1.0, "max_tokens": out, "logprobs": None,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True, "temperature": 0}


@pytest.mark.anyio
@pytest.mark.parametrize("point", ["P-U2K", "GW-U8K-O512", "L-U8K", "W-mini", "F-U2K-var"])
async def test_a_vllm_shaped_bench_through_the_real_gateway_joins_its_access_log(caplog, point):
    """044 實作補記六: the request sequence of vLLM 0.30's `vllm bench serve` through the real gateway stack (uvicorn,
    the guard, the point's settings, policy and profile; a fake vLLM behind it): the runner's /readyz poll, the random
    dataset's POST /tokenize (serve.py _align_prompts_to_server_tokenizer; 404 here), the ready check and C warmups
    without an id, GET /metrics twice, N requests with x-request-id <tag>-<idx> at concurrency C, GET /metrics twice
    (serve.py benchmark: the spec-decode and diffusion metrics before and after). The access log the gateway really
    writes joins with stage_v_analysis.join_access_log without a problem, and its id-less records give (j) the
    warmups' intervals, so the research side's fakes (tests/test_stage_v_analysis.log_text) cannot drift from it."""
    caplog.set_level(logging.INFO, logger="capserve.access")
    plan = build("mini")
    x = plan["points"][point]
    C, N, out = x["C"], x["N"], x["limits"]["max_output_tokens"]
    argv = svp.bench_argv(plan, _pins(), point, block=1, result_dir="r", result_filename="f",
                          datasets={k: "d.jsonl" for k in (x["datasets"] or {}).values()})
    random_dataset = argv[argv.index("--dataset-name") + 1] == "random"
    over = {k: v for k, v in x["gateway"].items() if k not in ("backend_url", "host", "port", "drain_timeout_s")}
    if x["admission"]:
        over["active_class"] = None
    prof = cp.load_profile(ROOT / x["profile"], allow_provisional=True)
    tag = svp.run_tag("V", point, C, block=1)
    row0 = " ".join(["w"] * 12)
    async with stack(FakeConfig(), prof, local_tokenizer=LocalTokenizer(WordsHF()), **over) as st, st.client() as c:
        assert (await c.get("/readyz")).status_code == 200
        if random_dataset:
            r = await c.post("/tokenize", json={"model": MODEL, "prompt": row0, "add_special_tokens": False})
            assert r.status_code == 404
        assert (await c.post("/v1/completions", json=_bench_body(row0, out))).status_code == 200      # ready check
        warm = await asyncio.gather(*(c.post("/v1/completions", json=_bench_body(row0, out)) for _ in range(C)))
        assert [r.status_code for r in warm] == [200] * C
        for _ in range(2):                              # fetch_spec_decode_metrics, fetch_diffusion_metrics
            assert (await c.get("/metrics")).status_code == 200
        sem = asyncio.Semaphore(C)

        async def one(i):
            async with sem:
                return await c.post("/v1/completions", json=_bench_body(" ".join(["w"] * (8 + i)), out),
                                    headers={"x-request-id": f"{tag}-{i}"})

        assert [r.status_code for r in await asyncio.gather(*(one(i) for i in range(N)))] == [200] * N
        for _ in range(2):
            assert (await c.get("/metrics")).status_code == 200
    lines = [r.getMessage() for r in caplog.records if r.name == "capserve.access"]
    assert len(lines) == 1 + random_dataset + 1 + C + 2 + N + 2
    problems, joined, warm = sva.join_access_log("\n".join(lines) + "\n", plan=plan, point=point, block=1)
    assert problems == [] and sorted(joined) == list(range(N)) and len(warm) == C + 1
    spans = sva.warmup_spans(warm, joined, x=x)                              # (j) reads them with the main requests
    assert len(spans) == C + 1 and all(s <= r < 0 for s, r, _ in spans)
    row = {"valid": True, "workload": point, "block": 1, "warmups": spans}
    assert sva.warmup_problems(row, x) == []
    assert any(json.loads(e)["path"] == "/tokenize" for e in lines) is random_dataset


def _pins() -> dict:
    return tomllib.loads((ROOT / "configs" / "pins.toml").read_text())
