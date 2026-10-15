"""Tests for capserve/loadtest.py (decision 029 "產品負載測試"): the smoke / steady / burst / overload scenarios against
the gateway's /v1/completions (prompts " the" x n inside the active class, ignore_eos), client-side TTFT / e2e, the
status histogram, Retry-After on every rejection, vLLM running + waiting sampled against the limit (IT-2), and the
outputs (requests.jsonl, summary.json) of a product test that never claims a capacity. Runs against the fake vLLM
behind the real gateway (fakes.servers.stack)."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

import run_core_b
from capserve import loadtest as lt
from capserve import runs
from fakes.fake_vllm import MODEL, FakeConfig
from fakes.servers import make_profile, stack

LIMITS = lt.Limits(max_concurrency=4, max_queue=4, max_prompt_tokens=1920, max_output_tokens=128, max_pending=24)


def rows(run_dir: Path) -> list:
    return [json.loads(x) for x in (run_dir / "requests.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------ scenarios --

def test_the_four_scenarios_and_their_client_counts():
    assert set(lt.SCENARIOS) == {"smoke", "steady", "burst", "overload"}
    assert lt.SCENARIOS["smoke"].clients(LIMITS) == 1
    assert lt.SCENARIOS["steady"].clients(LIMITS) == 4
    assert lt.SCENARIOS["burst"].clients(LIMITS) == 16
    assert lt.SCENARIOS["overload"].clients(LIMITS) == 24          # 3 x (limit + queue): more than limit + queue
    assert lt.SCENARIOS["overload"].clients(LIMITS) > LIMITS.max_concurrency + LIMITS.max_queue
    assert lt.SCENARIOS["smoke"].allowed == {200} and lt.SCENARIOS["steady"].allowed == {200}
    assert lt.SCENARIOS["overload"].allowed == {200, 429} and lt.SCENARIOS["overload"].require_rejection


def test_requests_stay_inside_the_active_class():
    small = lt.Limits(max_concurrency=1, max_queue=0, max_prompt_tokens=100, max_output_tokens=8, max_pending=17)
    for sc in lt.SCENARIOS.values():
        body = lt.request_body(sc, small, model=MODEL, index=1)
        n = body["prompt"].count(" the")
        assert body["prompt"] == " the" * n and 1 <= n < small.max_prompt_tokens
        assert len(body["prompt"].encode()) + 1 <= small.max_prompt_tokens        # the gateway's byte fast path
        assert body["max_tokens"] <= small.max_output_tokens and body["ignore_eos"] is True
        assert body["model"] == MODEL and body["stream"] is True                   # odd index: stream
    assert lt.request_body(lt.SCENARIOS["smoke"], LIMITS, model=MODEL, index=2)["stream"] is False


def test_limits_come_from_readyz():
    body = {"limits": {"max_concurrency": 4, "max_queue": 4, "max_pending": 24, "max_prompt_tokens": 1920,
                       "max_output_tokens": 128}}
    assert lt.Limits.from_readyz(body) == LIMITS


# ------------------------------------------------------------------ evaluation --

def rec(status=200, *, retry_after=None, error=None, stream=False, done=True, stream_error=None):
    return {"status": status, "retry_after": retry_after, "error": error, "stream": stream, "done": done,
            "stream_error": stream_error, "ttft_s": 0.1 if stream else None, "e2e_s": 0.2}


def evaluate(name, records, samples=None, *, sampled=True, timed_out=False):
    return lt.evaluate(lt.SCENARIOS[name], LIMITS, records, samples if samples is not None else
                       [{"running": 4.0, "waiting": 0.0}], vllm_sampled=sampled, timed_out=timed_out)


def test_a_clean_overload_passes():
    checks, problems = evaluate("overload", [rec(), rec(429, retry_after="2"), rec(stream=True)])
    assert problems == [] and all(v is not False for v in checks.values())


@pytest.mark.parametrize("records, check", [
    ([rec(), rec(502)], "no_5xx"),
    ([rec(), rec(503, retry_after="5")], "statuses_allowed"),
    ([rec(), rec(429)], "retry_after_on_every_rejection"),
    ([rec(), rec(None, error="ReadError")], "no_transport_errors"),
    ([rec(), rec(stream=True, done=False)], "streams_complete"),
    ([rec(), rec(stream=True, stream_error="stream_broken")], "streams_complete"),
    ([rec(429, retry_after="2")], "some_success"),
    ([rec(), rec()], "rejection_seen"),
])
def test_overload_failures(records, check):
    checks, problems = evaluate("overload", records)
    assert checks[check] is False and any(check in p for p in problems)


def test_a_429_is_a_failure_in_smoke():
    checks, _ = evaluate("smoke", [rec(), rec(429, retry_after="2")])
    assert checks["statuses_allowed"] is False and checks["rejection_seen"] is None


def test_vllm_beyond_the_limit_fails_and_missing_samples_are_not_a_pass():
    checks, problems = evaluate("overload", [rec(), rec(429, retry_after="2")],
                                [{"running": 4.0, "waiting": 1.0}])
    assert checks["vllm_within_limit"] is False and any("5" in p for p in problems)
    checks, _ = evaluate("overload", [rec(), rec(429, retry_after="2")], [{"error": "refused"}])
    assert checks["vllm_within_limit"] is False
    checks, _ = evaluate("overload", [rec(), rec(429, retry_after="2")], [], sampled=False)
    assert checks["vllm_within_limit"] is None


def test_the_time_limit_is_a_failure():
    checks, problems = evaluate("smoke", [rec()], timed_out=True)
    assert checks["within_time_limit"] is False and problems


# ------------------------------------------------------------------ against the fake stack --

@pytest.mark.anyio
async def test_fake_vllm_metrics_count_running_generations():
    async with stack(FakeConfig(hang=True)) as st, st.client() as c, httpx.AsyncClient(
            base_url=st.gateway.settings.backend_url, trust_env=False) as v:
        task = asyncio.create_task(c.post("/v1/completions", json={"model": MODEL, "prompt": "hi"}))
        await asyncio.wait_for(st.fake.started.wait(), 5)
        m = run_core_b.parse_prometheus((await v.get("/metrics")).text)
        assert run_core_b.gauge(m, "vllm:num_requests_running") == 1
        assert run_core_b.gauge(m, "vllm:num_requests_waiting") == 0
        assert run_core_b.cache_config(m) == {"num_gpu_blocks": 248, "block_size": 528}
        st.fake.release.set()
        assert (await task).status_code == 200


@pytest.mark.anyio
async def test_smoke_against_the_gateway(tmp_path):
    async with stack(FakeConfig(n_tokens=3)) as st:
        summary = await lt.run_async("smoke", gateway_url=st.url, vllm_url=st.gateway.settings.backend_url,
                                     run_dir=tmp_path, sample_interval_s=0.01)
    assert summary["ok"] is True and summary["problems"] == [] and summary["statuses"] == {"200": 4}
    assert summary["purpose"] == "product-test" and summary["note"] == "not a capacity measurement"
    assert summary["research_valid"] is False and summary["limits"]["max_concurrency"] == 4
    assert summary["profile"]["status"] == "provisional"
    rs = rows(tmp_path)
    assert len(rs) == 4 and [r["stream"] for r in rs] == [False, True, False, True]
    assert all(r["status"] == 200 and r["e2e_s"] > 0 and r["request_id"] for r in rs)
    assert all(r["ttft_s"] is not None and r["done"] for r in rs if r["stream"])
    assert " the" not in (tmp_path / "requests.jsonl").read_text()     # no prompt text is ever recorded
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    for name in ("gateway-metrics-before.txt", "gateway-metrics-after.txt", "vllm-metrics-before.txt",
                 "vllm-metrics-after.txt"):
        assert (tmp_path / name).read_text()
    assert "capserve_profile_info" in (tmp_path / "gateway-metrics-after.txt").read_text()
    sent = [s.body for s in st.fake.generations]
    assert all(b["ignore_eos"] is True and b["prompt"] == " the" * 16 for b in sent)


@pytest.mark.anyio
async def test_overload_gives_only_200_and_429_with_retry_after_and_never_exceeds_the_limit(tmp_path):
    prof = make_profile(max_concurrency=2, max_queue=1)
    async with stack(FakeConfig(ttft_s=0.02, itl_s=0.005, n_tokens=8), prof, max_queue_wait_s=0.05) as st:
        summary = await lt.run_async("overload", gateway_url=st.url, vllm_url=st.gateway.settings.backend_url,
                                     run_dir=tmp_path, sample_interval_s=0.005)
        assert st.fake.peak_in_flight <= 2
    assert summary["clients"] == 9 and summary["n_requests"] == 36
    assert set(summary["statuses"]) == {"200", "429"} and summary["ok"] is True, summary["problems"]
    assert summary["checks"]["rejection_seen"] and summary["checks"]["retry_after_on_every_rejection"]
    assert summary["vllm"]["max_running_plus_waiting"] <= 2 and summary["vllm"]["samples"] > 0
    assert all(r["retry_after"] for r in rows(tmp_path) if r["status"] == 429)


@pytest.mark.anyio
async def test_upstream_5xx_fails_the_scenario(tmp_path):
    async with stack(FakeConfig(fail_status=500)) as st:
        summary = await lt.run_async("smoke", gateway_url=st.url, vllm_url=None, run_dir=tmp_path)
    assert summary["ok"] is False and summary["checks"]["no_5xx"] is False
    assert summary["vllm"] is None and summary["checks"]["vllm_within_limit"] is None


@pytest.mark.anyio
async def test_the_time_limit_cancels_and_fails(tmp_path):
    async with stack(FakeConfig(hang=True)) as st:
        summary = await lt.run_async("smoke", gateway_url=st.url, vllm_url=None, run_dir=tmp_path,
                                     time_limit_s=0.3)
    assert summary["ok"] is False and summary["checks"]["within_time_limit"] is False
    assert any(r["error"] == "cancelled" for r in rows(tmp_path))


@pytest.mark.anyio
async def test_the_api_key_is_sent_and_never_recorded(tmp_path):
    async with stack(FakeConfig(n_tokens=2), api_key=b"s3cr3t") as st:
        summary = await lt.run_async("smoke", gateway_url=st.url, vllm_url=None, run_dir=tmp_path, api_key="s3cr3t")
    assert summary["ok"] is True
    for f in tmp_path.iterdir():
        assert "s3cr3t" not in f.read_text(), f.name


@pytest.mark.anyio
async def test_an_unreachable_gateway_is_a_failed_run_not_a_crash(tmp_path):
    summary = await lt.run_async("smoke", gateway_url="http://127.0.0.1:9", vllm_url=None, run_dir=tmp_path)
    assert summary["ok"] is False and any("readyz" in p for p in summary["problems"])


# ------------------------------------------------------------------ the CLI --

@pytest.mark.anyio
async def test_cli_writes_a_product_run_under_the_service_root(tmp_path):
    cfg = tmp_path / "deploy.toml"
    svc = tmp_path / "svc"
    cfg.write_text(f'[deploy]\nservice_root = "{svc}"\n[gateway]\nactive_class = "test"\n')
    async with stack(FakeConfig(n_tokens=2)) as st:
        code = await asyncio.to_thread(lt.main, ["smoke", "--config", str(cfg), "--gateway", st.url, "--vllm",
                                                 st.gateway.settings.backend_url])
    assert code == 0
    (run_dir,) = (svc / "runs").iterdir()
    assert run_dir.name.startswith("pt-") and run_dir.name.endswith("-smoke")
    meta = runs.read_meta(run_dir)
    assert meta["purpose"] == "product-test" and meta["formal"] is False and meta["research_valid"] is False
    assert meta["kind"] == "loadtest" and meta["scenario"] == "smoke" and meta["note"] == "not a capacity measurement"
    assert meta["profile"]["status"] == "provisional" and meta["outcome"]["ok"] is True
    assert json.loads((run_dir / "summary.json").read_text())["ok"] is True


def test_cli_has_no_formal_option_and_refuses_unknown_scenarios(capsys):
    for action in lt.parser()._actions:
        assert not any("formal" in s for s in action.option_strings)
    with pytest.raises(SystemExit):
        lt.main(["capacity"])
