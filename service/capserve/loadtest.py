"""python -m capserve.loadtest <scenario> [--config F] [--gateway URL] [--vllm URL]   (decision 029 "產品負載測試")

A product load test of the gateway. It is never a research measurement and never a capacity claim: research TTFT
comes only from the research client in formal sessions, and the limits under test come from the profile (a
provisional one is not a capacity at all, 027 §4). Every output says purpose product-test and "not a capacity
measurement" and lives under <service_root>/runs (runs.py).

Requests: POST /v1/completions with prompt " the" x n (coreb_smoke.PROMPT_UNIT, one Qwen3.5 token per unit; n kept
inside the gateway's byte fast path, so inside the active class), max_tokens within the class, ignore_eos,
temperature 0; odd-numbered requests stream. The limits come from the gateway's /readyz.

Scenarios (L = max_concurrency, Q = max_queue; closed-loop clients, all released at once):
- smoke:    1 client x 4 requests; only 200;
- steady:   L clients x 5 requests (never more than the limit in flight); only 200;
- burst:    2 (L + Q) clients x 1 request; 200 and 429 (with Retry-After), nothing else;
- overload: 3 (L + Q) clients x 4 requests (more than limit + queue); 200 and 429 with Retry-After only, at least one
            429, no 5xx (RTX IT-2).

Recorded: per request (requests.jsonl; never the prompt or completion text) status, Retry-After, x-request-id,
client-side TTFT (first content event) and e2e, stream completeness; the status histogram; gateway and vLLM /metrics
snapshots before and after; vLLM num_requests_running + num_requests_waiting sampled during the run, which must never
exceed max_concurrency (the gateway sends at most L requests to vLLM; RTX IT-2). summary.json holds the checks; the
scenario is ok when every applicable check holds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import httpx

import coreb_smoke as smoke
import run_core_b as rb
from capserve import runs
from capserve.config import API_KEY_ENV, TokenOverheads
from capserve.proxy import SSEFramer, has_content, is_error

PROMPT_UNIT = smoke.PROMPT_UNIT
COMPLETION_OVERHEAD = TokenOverheads().completion_overhead   # the gateway's byte fast-path allowance
SAMPLE_INTERVAL_S = 0.1
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=330.0, write=10.0, pool=None)   # beyond the 300 s deadline
PROBE_TIMEOUT = httpx.Timeout(10.0)
REJECTIONS = (429, 503)


@dataclass(frozen=True)
class Limits:
    max_concurrency: int
    max_queue: int
    max_prompt_tokens: int
    max_output_tokens: int
    max_pending: int

    @classmethod
    def from_readyz(cls, body: dict) -> "Limits":
        lim = body["limits"]
        return cls(**{k: int(lim[k]) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Scenario:
    name: str
    clients: Callable[[Limits], int]
    requests_per_client: int
    prompt_tokens: int
    max_tokens: int
    allowed: frozenset
    require_rejection: bool
    description: str


SCENARIOS = {s.name: s for s in (
    Scenario("smoke", lambda lim: 1, 4, 16, 16, frozenset({200}), False,
             "1 client x 4 requests, stream and non-stream"),
    Scenario("steady", lambda lim: lim.max_concurrency, 5, 256, 64, frozenset({200}), False,
             "max_concurrency closed-loop clients x 5 requests"),
    Scenario("burst", lambda lim: 2 * (lim.max_concurrency + lim.max_queue), 1, 256, 64, frozenset({200, 429}),
             False, "2 x (limit + queue) requests at once"),
    Scenario("overload", lambda lim: 3 * (lim.max_concurrency + lim.max_queue), 4, 256, 64, frozenset({200, 429}),
             True, "3 x (limit + queue) closed-loop clients x 4 requests"),
)}


def request_body(sc: Scenario, limits: Limits, *, model: str, index: int) -> dict:
    fast_path = (limits.max_prompt_tokens - COMPLETION_OVERHEAD) // len(PROMPT_UNIT.encode())
    n = max(1, min(sc.prompt_tokens, fast_path))
    return {"model": model, "prompt": PROMPT_UNIT * n, "max_tokens": max(1, min(sc.max_tokens,
                                                                                  limits.max_output_tokens)),
            "ignore_eos": True, "temperature": 0.0, "stream": index % 2 == 1}


# ---------------------------------------------------------------- one request

def _error_code(event: bytes) -> str:
    for line in event.splitlines():
        if line.startswith(b"data:"):
            try:
                return str(json.loads(line[5:].strip())["error"].get("code") or "error")
            except (ValueError, KeyError, AttributeError, TypeError):
                return "error"
    return "error"


def _is_done(event: bytes) -> bool:
    return any(line.startswith(b"data:") and line[5:].strip() == b"[DONE]" for line in event.splitlines())


async def _one(client: httpx.AsyncClient, body: dict, r: dict, clock: Callable[[], float]) -> None:
    """Send one request and fill r in place (so a cancelled request is still recorded)."""
    t0 = clock()
    try:
        async with client.stream("POST", "/v1/completions", json=body) as resp:
            r["status"] = resp.status_code
            r["retry_after"] = resp.headers.get("retry-after")
            r["request_id"] = resp.headers.get("x-request-id")
            if resp.status_code == 200 and body["stream"]:
                r["done"] = False
                framer = SSEFramer()
                async for chunk in resp.aiter_bytes():
                    for event in framer.feed(chunk):
                        if is_error(event):
                            r["stream_error"] = _error_code(event)
                        elif has_content(event):
                            r["events"] += 1
                            if r["ttft_s"] is None:
                                r["ttft_s"] = clock() - t0
                        elif _is_done(event):
                            r["done"] = True
            else:
                content = await resp.aread()
                try:
                    doc = json.loads(content)
                except ValueError:
                    doc = None
                doc = doc if isinstance(doc, dict) else {}
                if resp.status_code == 200:
                    r["completion_tokens"] = (doc.get("usage") or {}).get("completion_tokens")
                else:
                    r["error_code"] = (doc.get("error") or {}).get("code")
    except httpx.HTTPError as exc:
        r["error"] = type(exc).__name__
    except asyncio.CancelledError:
        r["error"] = "cancelled"
        raise
    finally:
        r["e2e_s"] = clock() - t0


def _new_record(c: int, k: int, index: int, stream: bool) -> dict:
    return {"client": c, "seq": k, "index": index, "stream": stream, "status": None, "retry_after": None,
            "request_id": None, "ttft_s": None, "e2e_s": None, "events": 0, "done": None, "stream_error": None,
            "error_code": None, "completion_tokens": None, "error": None}


async def _client(c: int, client, sc: Scenario, limits: Limits, model: str, records: list, go: asyncio.Event,
                  clock) -> None:
    await go.wait()
    for k in range(sc.requests_per_client):
        index = c * sc.requests_per_client + k
        body = request_body(sc, limits, model=model, index=index)
        r = _new_record(c, k, index, body["stream"])
        records.append(r)
        await _one(client, body, r, clock)


async def _sample(client: httpx.AsyncClient, samples: list, stop: asyncio.Event, interval_s: float, clock,
                  t0: float) -> None:
    while True:
        t = round(clock() - t0, 4)
        try:
            resp = await client.get("/metrics")
            if resp.status_code == 200:
                m = rb.parse_prometheus(resp.text)
                samples.append({"t_s": t, "running": rb.gauge(m, "vllm:num_requests_running"),
                                "waiting": rb.gauge(m, "vllm:num_requests_waiting")})
            else:
                samples.append({"t_s": t, "error": f"status {resp.status_code}"})
        except httpx.HTTPError as exc:
            samples.append({"t_s": t, "error": type(exc).__name__})
        try:
            await asyncio.wait_for(stop.wait(), interval_s)
            return
        except TimeoutError:
            pass


# ---------------------------------------------------------------- evaluation

def _retry_after_ok(value) -> bool:
    return isinstance(value, str) and value.strip().isdigit()


def valid_samples(samples: list) -> list:
    return [s for s in samples if s.get("running") is not None and s.get("waiting") is not None]


def evaluate(sc: Scenario, limits: Limits, records: list, samples: list, *, vllm_sampled: bool,
             timed_out: bool) -> tuple:
    """(checks, problems): None = not applicable; ok = no check is False."""
    statuses = [r["status"] for r in records if r["status"] is not None]
    rejections = [r for r in records if r["status"] in REJECTIONS]
    transport = [r["error"] for r in records if r["error"] and r["error"] != "cancelled"]
    streams = [r for r in records if r["stream"] and r["status"] == 200 and r["error"] is None]
    broken = [r for r in streams if not r["done"] or r["stream_error"]]
    good = valid_samples(samples)
    peak = max((s["running"] + s["waiting"] for s in good), default=None)
    checks = {
        "statuses_allowed": all(s in sc.allowed for s in statuses),
        "no_5xx": not any(s >= 500 for s in statuses),
        "no_transport_errors": not transport,
        "retry_after_on_every_rejection": all(_retry_after_ok(r["retry_after"]) for r in rejections),
        "streams_complete": not broken,
        "some_success": 200 in statuses,
        "rejection_seen": (429 in statuses) if sc.require_rejection else None,
        "vllm_within_limit": None if not vllm_sampled else (peak is not None and peak <= limits.max_concurrency),
        "within_time_limit": not timed_out,
    }
    details = {
        "statuses_allowed": f"saw {sorted(set(statuses) - sc.allowed)}; {sc.name} allows only {sorted(sc.allowed)}",
        "no_5xx": f"{sum(s >= 500 for s in statuses)} response(s) with status >= 500",
        "no_transport_errors": f"{len(transport)} transport error(s): {dict(Counter(transport))}",
        "retry_after_on_every_rejection": f"{sum(not _retry_after_ok(r['retry_after']) for r in rejections)} "
                                          "429 / 503 response(s) without a Retry-After",
        "streams_complete": f"{len(broken)} stream(s) without [DONE] or with an error event",
        "some_success": "no request succeeded",
        "rejection_seen": "no 429: the scenario never overloaded the gateway",
        "vllm_within_limit": ("no vLLM /metrics sample" if peak is None else
                              f"vLLM running + waiting reached {peak:g} > max_concurrency {limits.max_concurrency}"),
        "within_time_limit": "the time limit cancelled the remaining requests",
    }
    problems = [f"{name}: {details[name]}" for name, value in checks.items() if value is False]
    return checks, problems


def _describe(values: list) -> dict | None:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None

    def q(p: float) -> float:
        return round(xs[max(0, math.ceil(p * len(xs)) - 1)], 4)

    return {"n": len(xs), "p50": q(0.5), "p95": q(0.95), "max": round(xs[-1], 4)}


# ---------------------------------------------------------------- the run

def _write_jsonl(path: Path, items) -> None:
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in items))


async def _snapshot(client: httpx.AsyncClient | None, path: Path) -> None:
    if client is None:
        return
    try:
        resp = await client.get("/metrics")
        path.write_text(resp.text if resp.status_code == 200 else f"# status {resp.status_code}\n")
    except httpx.HTTPError as exc:
        path.write_text(f"# {type(exc).__name__}: {exc}\n")


async def run_async(scenario: str, *, gateway_url: str, vllm_url: str | None, run_dir, time_limit_s: float | None
                    = None, api_key: str | None = None, sample_interval_s: float = SAMPLE_INTERVAL_S,
                    clock: Callable[[], float] = time.monotonic, verify=True) -> dict:
    sc = SCENARIOS[scenario]
    run_dir = Path(run_dir)
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    summary = {"schema_version": 1, "scenario": sc.name, "description": sc.description, "purpose": runs.PURPOSE,
               "note": runs.NOTE, "research_valid": False, "gateway_url": gateway_url, "vllm_url": vllm_url,
               "time_limit_s": time_limit_s}
    async with httpx.AsyncClient(base_url=gateway_url, trust_env=False, timeout=PROBE_TIMEOUT,
                                 headers=headers, verify=verify) as probe:
        try:
            ready = await probe.get("/readyz")
            readyz = ready.json() if ready.status_code == 200 else None
            models = await probe.get("/v1/models") if readyz else None
            model = models.json()["data"][0]["id"] if models is not None and models.status_code == 200 else None
            why = None if model else f"gateway /readyz {ready.status_code} or /v1/models unusable"
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            readyz, why = None, f"gateway /readyz: {type(exc).__name__}: {exc}"
    if why:
        summary.update(ok=False, problems=[why if "readyz" in why else f"readyz: {why}"])
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        return summary

    limits = Limits.from_readyz(readyz)
    n_clients = sc.clients(limits)
    sample_body = request_body(sc, limits, model=model, index=0)
    summary.update(profile=readyz.get("profile"), active_class=readyz.get("active_class"), limits=asdict(limits),
                   clients=n_clients, requests_per_client=sc.requests_per_client,
                   prompt_tokens=sample_body["prompt"].count(PROMPT_UNIT), max_tokens=sample_body["max_tokens"])
    records, samples = [], []
    timed_out = False
    pool = httpx.Limits(max_connections=n_clients + 4, max_keepalive_connections=n_clients + 4)
    async with httpx.AsyncClient(base_url=gateway_url, trust_env=False, timeout=REQUEST_TIMEOUT, headers=headers,
                                 limits=pool, verify=verify) as gw, \
            httpx.AsyncClient(base_url=vllm_url or "http://invalid", trust_env=False, timeout=PROBE_TIMEOUT) as vc:
        vllm = vc if vllm_url else None
        await _snapshot(gw, run_dir / "gateway-metrics-before.txt")
        await _snapshot(vllm, run_dir / "vllm-metrics-before.txt")
        go, stop = asyncio.Event(), asyncio.Event()
        t0 = clock()
        sampler = asyncio.create_task(_sample(vllm, samples, stop, sample_interval_s, clock, t0)) if vllm else None
        tasks = [asyncio.create_task(_client(c, gw, sc, limits, model, records, go, clock)) for c in range(n_clients)]
        go.set()
        try:
            async with asyncio.timeout(time_limit_s):
                await asyncio.gather(*tasks)
        except TimeoutError:
            timed_out = True
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            stop.set()
            if sampler is not None:
                await sampler
        duration = clock() - t0
        await _snapshot(gw, run_dir / "gateway-metrics-after.txt")
        await _snapshot(vllm, run_dir / "vllm-metrics-after.txt")

    records.sort(key=lambda r: r["index"])
    checks, problems = evaluate(sc, limits, records, samples, vllm_sampled=vllm_url is not None,
                                timed_out=timed_out)
    good = valid_samples(samples)
    statuses = Counter(r["status"] for r in records if r["status"] is not None)
    summary.update(
        n_requests=len(records), not_sent=n_clients * sc.requests_per_client - len(records),
        statuses={str(k): statuses[k] for k in sorted(statuses)},
        errors=dict(sorted(Counter(r["error"] for r in records if r["error"]).items())),
        retry_after_missing=sum(not _retry_after_ok(r["retry_after"]) for r in records
                                if r["status"] in REJECTIONS),
        descriptive={"note": "client-side, product test: not research evidence and not a capacity",
                     "ttft_s": _describe([r["ttft_s"] for r in records if r["status"] == 200]),
                     "e2e_s_ok": _describe([r["e2e_s"] for r in records if r["status"] == 200])},
        vllm=None if vllm_url is None else {
            "samples": len(good), "sample_errors": len(samples) - len(good),
            "max_running": max((s["running"] for s in good), default=None),
            "max_waiting": max((s["waiting"] for s in good), default=None),
            "max_running_plus_waiting": max((s["running"] + s["waiting"] for s in good), default=None)},
        duration_s=round(duration, 3), checks=checks, ok=not problems, problems=problems)
    _write_jsonl(run_dir / "requests.jsonl", records)
    if vllm_url is not None:
        _write_jsonl(run_dir / "vllm-samples.jsonl", samples)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def run_scenario(scenario: str, **kw) -> dict:
    """The synchronous entry (deploy's loadtest workload, the CLI)."""
    return asyncio.run(run_async(scenario, **kw))


# ---------------------------------------------------------------- CLI

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m capserve.loadtest",
                                description="product load test of the gateway (never a capacity measurement)")
    p.add_argument("scenario", choices=sorted(SCENARIOS))
    p.add_argument("--config", help="deploy config (default service/deploy/rtx.toml): service_root and ports")
    p.add_argument("--gateway", help="gateway base URL (default: the config's gateway port on 127.0.0.1)")
    p.add_argument("--vllm", help="vLLM base URL for /metrics (default: the config's; 'none' to skip sampling)")
    p.add_argument("--time-limit-s", type=float, default=None, help="cancel the remaining requests after this")
    return p


def release_sha(root: Path = runs.REPO_ROOT) -> str | None:
    try:
        return (root / "DEPLOYED_COMMIT").read_text().split()[0]
    except (OSError, IndexError):
        return None


def main(argv: list | None = None, *, env=None) -> int:
    args = parser().parse_args(argv)
    from capserve import deploy   # deploy imports this module

    env = os.environ if env is None else env
    try:
        cfg = deploy.load_config(args.config or deploy.DEFAULT_CONFIG)
    except deploy.DeployError as exc:
        print(f"capserve.loadtest: {exc}", file=sys.stderr)
        return 2
    gateway = args.gateway or cfg.gateway_url
    verify = True if args.gateway else cfg.gateway_verify   # 038 §6: the config's own gateway: its certificate
    vllm = None if args.vllm == "none" else (args.vllm or cfg.vllm_url)
    status, text = deploy.default_http_get(f"{gateway}/readyz", verify=verify)
    try:
        readyz = json.loads(text) if status == 200 else None
    except ValueError:
        readyz = None
    if not isinstance(readyz, dict):
        print(f"capserve.loadtest: {gateway}/readyz answered {status}; nothing was sent", file=sys.stderr)
        return 2
    now = deploy.utcnow()
    try:
        run_dir = runs.create_run(service_root=cfg.service_root, research_dir=cfg.research_dir,
                                  run_id=runs.new_run_id(now, args.scenario), kind="loadtest", release=release_sha(),
                                  profile=readyz.get("profile"), now_utc=now,
                                  extra={"scenario": args.scenario, "gateway_url": gateway, "vllm_url": vllm})
    except runs.RunError as exc:
        print(f"capserve.loadtest: {exc}", file=sys.stderr)
        return 2
    summary = run_scenario(args.scenario, gateway_url=gateway, vllm_url=vllm, run_dir=run_dir,
                           time_limit_s=args.time_limit_s, api_key=env.get(API_KEY_ENV) or None, verify=verify)
    runs.update_meta(run_dir, outcome={"ok": summary["ok"], "problems": summary["problems"]})
    print(json.dumps({"run_dir": str(run_dir), "ok": summary["ok"], "statuses": summary.get("statuses"),
                      "problems": summary["problems"]}, indent=1))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
