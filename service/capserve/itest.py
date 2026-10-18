"""python -m capserve.itest IT-0 [IT-1 ... IT-6] [--config F] [--profile P] [--allow-provisional] [--since <run id>]

The RTX integration tests of decision 029 ("RTX 整合測試"): product tests only, never research, never a capacity.
Each test is a deterministic pass / fail check; the results go to <service_root>/runs/pt-<UTC>-itest/itest.json
(meta.json: purpose product-test, formal false, research_valid false).

- IT-0 (no lock, no boot): imports, and a read-only look at the installed vLLM 0.30.0 sources (located with
  importlib, never imported): the /tokenize route and its request fields (add_generation_prompt, add_special_tokens),
  max_completion_tokens on chat requests, and the HTTP keep-alive default, which must outlive the gateway pool's
  keepalive_expiry (upstream.KEEPALIVE_EXPIRY_S) so a pooled connection is never reused just as vLLM closes it.
- IT-1..IT-5 (under the GPU lock): ONE boot for all of them. This process starts `python -m capserve.deploy up`
  (hold mode) as a child with the lock descriptor passed on (procs.start_group, PDEATHSIG), waits for state.json
  phase ready, runs the checks in order, then stops it with `capserve.deploy stop` (IT-5 does the stop itself):
  IT-1 readiness, /v1/models, chat and completion in stream and non-stream mode, the gateway /metrics, and that
       /proc/<vLLM pid>/fd and /proc/<gateway pid>/fd hold the lock file;
  IT-2 3x overload (capserve.loadtest overload): vLLM running + waiting never above max_concurrency; only 200 and
       429 with Retry-After; no 5xx;
  IT-3 a stream and a non-stream client disconnect: vLLM running + waiting back to 0 within 2 s, no tokens generated
       after the disconnect, and the request would otherwise have run well past the window (decisive);
  IT-4 /tokenize parity, with the gateway's own PromptBudget against vLLM: " the" x n is n tokens, every measured
       chat / completion count is within the byte fast-path bound, and every chat case's template overhead (its
       count minus the tokens of each message's text alone, add_special_tokens false) is within the constants in
       settings [tokens]: chat_overhead + sum(per_message + role bytes). The byte bound alone cannot catch a constant
       that is too small (" the" leaves 4 bytes per token of slack); an empty message and a many-message case
       expose chat_overhead and per_message;
  IT-5 stop under load: limit + queue clients, `capserve.deploy stop` once the gateway is saturated; the running
       requests drain with 200, queued ones get 503 with Retry-After, nothing is cut, the supervisor exits 143 with
       a clean teardown and GPU memory back <= the supervisor run's pre-boot baseline + 64 MiB within 60 s
       (decision 031; the supervisor's preflight recorded the baseline from its 30 s idle window).
- IT-6 (no boot, under the GPU lock: 029 exempts only IT-0, and 027 §5 puts no product test inside a formal research
  session): `find ~/llm-serving-lab ~/.cache/{vllm,flashinfer,huggingface} ~/.triton -newer runs/<id>/meta.json` is
  empty, by default against this itest run's own meta.json (created before IT-0 and the boot), or --since <an
  earlier product run>. Run it in the same locked invocation, so no research run can interleave.

SIGTERM and SIGHUP (an ssh hang-up) raise SystemExit(128 + signum) as in `capserve.deploy up`, so the supervisor is
stopped and itest.json / meta.json still record the finished tests, the rest as not run, and `interrupted`.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import dataclasses
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

import core_b_plan as bp
import coreb_smoke as smoke
import run_adv
import run_core_b as rb
from capserve import deploy, loadtest, procs, runs
from capserve.config import API_KEY_ENV, Settings, load_settings
from capserve.proxy import SSEFramer, has_content, is_error
from capserve.tokens import PromptBudget, TokenizeError
from capserve.upstream import KEEPALIVE_EXPIRY_S

TESTS = ("IT-0", "IT-1", "IT-2", "IT-3", "IT-4", "IT-5", "IT-6")
BOOTED = ("IT-1", "IT-2", "IT-3", "IT-4", "IT-5")
DEFAULT_PROFILE = runs.REPO_ROOT / "service" / "profiles" / "provisional-rtx.json"
PROMPT_UNIT = smoke.PROMPT_UNIT
DISCONNECT_WINDOW_S = 2.0       # 029 IT-3: running + waiting drops within 2 s
DECISIVE_FACTOR = 2.0           # the request must have had >= 2 x the window of generation left
READ_EVENTS = 8                 # content events read before the stream client disconnects
IT2_TIME_LIMIT_S = 900.0
SATURATION_WAIT_S = 120.0
SUPERVISOR_EXTRA_S = 120.0      # readiness wait beyond the supervisor's own (gateway start, first state write)
IMPORTS = ("capserve.app", "capserve.server", "capserve.deploy", "capserve.loadtest", "capserve.procs",
           "capserve.runs", "core_b_plan", "coreb_smoke", "run_adv", "run_core_b")
CHAT_CASES = (
    ("one user message (' the' x 16)", [{"role": "user", "content": PROMPT_UNIT * 16}]),
    ("system + user", [{"role": "system", "content": "You are terse."},
                       {"role": "user", "content": PROMPT_UNIT * 16}]),
    ("multi-turn", [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
                    {"role": "user", "content": PROMPT_UNIT * 16}]),
    ("long user message (' the' x 1000)", [{"role": "user", "content": PROMPT_UNIT * 1000}]),
    ("non-ASCII (CJK)", [{"role": "user", "content": "測試" * 50}]),
    ("text parts", [{"role": "user", "content": [{"type": "text", "text": "hello"},
                                                 {"type": "text", "text": " world"}]}]),
    ("empty user message", [{"role": "user", "content": ""}]),             # the template alone: chat_overhead
    ("many messages (8)", [{"role": "system", "content": "Be brief."}]      # per_message, 8 times over
     + [{"role": r, "content": "Hi"} for r in ("user", "assistant") * 3] + [{"role": "user", "content": "Hi"}]),
)
COMPLETION_NS = (1, 16, 256, 1000)


class Checks:
    def __init__(self):
        self.items: list = []

    def add(self, name: str, ok, detail=None) -> bool:
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def result(self, test: str, **data) -> dict:
        return {"test": test, "ok": bool(self.items) and all(i["ok"] for i in self.items), "checks": self.items,
                "data": data, "error": None}


def _failed(test: str, error: str) -> dict:
    return {"test": test, "ok": False, "checks": [], "data": {}, "error": error}


# ---------------------------------------------------------------- IT-0: imports, vLLM sources (read-only)

def locate_vllm() -> tuple:
    """(package dir, installed version) without importing vllm (importing it would touch caches)."""
    try:
        spec = importlib.util.find_spec("vllm")
        version = importlib.metadata.version("vllm")
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        return None, None
    if spec is None or not spec.submodule_search_locations:
        return None, version
    return Path(list(spec.submodule_search_locations)[0]), version


def class_index(root: Path) -> dict:
    """name -> ClassDef for every class under root (first definition wins)."""
    index = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
        except (SyntaxError, ValueError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                index.setdefault(node.name, node)
    return index


def class_fields(index: dict, name: str, _seen=None) -> set:
    """Annotated fields of a class and of its bases found in the index (pydantic request models)."""
    seen = _seen or set()
    node = index.get(name)
    if node is None or name in seen:
        return set()
    seen.add(name)
    out = {s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    for base in node.bases:
        base_name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None
        if base_name:
            out |= class_fields(index, base_name, seen)
    return out


def keep_alive_default(pkg: Path) -> tuple:
    """(seconds, source): vLLM's uvicorn timeout_keep_alive default, or uvicorn's when vLLM never sets it."""
    envs = pkg / "envs.py"
    text = envs.read_text(errors="replace") if envs.is_file() else ""
    m = re.search(r'"VLLM_HTTP_TIMEOUT_KEEP_ALIVE"\s*,\s*"?(\d+)', text)
    used = any("timeout_keep_alive" in p.read_text(errors="replace") for p in (pkg / "entrypoints").rglob("*.py"))
    if m and used:
        return int(m.group(1)), "VLLM_HTTP_TIMEOUT_KEEP_ALIVE default (vllm/envs.py)"
    if not used:
        import uvicorn
        return inspect.signature(uvicorn.Config).parameters["timeout_keep_alive"].default, "uvicorn default"
    return None, "timeout_keep_alive is set but its default could not be read"


def it0(*, locate: Callable = locate_vllm, pins: dict | None = None) -> dict:
    pins = bp.load_pins() if pins is None else pins
    c = Checks()
    for mod in IMPORTS:
        try:
            importlib.import_module(mod)
            c.add(f"import {mod}", True)
        except Exception as exc:  # noqa: BLE001 - an import failure is the finding
            c.add(f"import {mod}", False, f"{type(exc).__name__}: {exc}")
    pkg, version = locate()
    want = pins["vllm"]["version"]
    c.add(f"vLLM version is the pinned {want}", version == want, version)
    if not c.add("vLLM sources found", pkg is not None and (pkg / "entrypoints").is_dir(), str(pkg)):
        return c.result("IT-0", vllm_version=version)
    entry = pkg / "entrypoints"
    index = class_index(entry)
    route = any('"/tokenize"' in p.read_text(errors="replace") for p in entry.rglob("*.py"))
    c.add("/tokenize route exists", route)
    chat_fields = class_fields(index, "TokenizeChatRequest")
    comp_fields = class_fields(index, "TokenizeCompletionRequest")
    c.add("/tokenize chat request accepts add_generation_prompt and add_special_tokens",
          {"messages", "add_generation_prompt", "add_special_tokens"} <= chat_fields, sorted(chat_fields))
    c.add("/tokenize completion request accepts add_special_tokens", {"prompt", "add_special_tokens"} <= comp_fields,
          sorted(comp_fields))
    c.add("/tokenize response has count", "count" in class_fields(index, "TokenizeResponse"))
    c.add("chat completions accept max_completion_tokens",
          "max_completion_tokens" in class_fields(index, "ChatCompletionRequest"))
    seconds, source = keep_alive_default(pkg)
    c.add("vLLM keep-alive outlives the gateway pool's keepalive_expiry",
          seconds is not None and seconds > KEEPALIVE_EXPIRY_S,
          f"{seconds} s ({source}) vs keepalive_expiry {KEEPALIVE_EXPIRY_S} s")
    return c.result("IT-0", vllm_version=version, vllm_root=str(pkg), keep_alive_s=seconds,
                    keep_alive_source=source, tokenize_chat_fields=sorted(chat_fields),
                    tokenize_completion_fields=sorted(comp_fields))


# ---------------------------------------------------------------- booted checks (IT-1..IT-5)

@dataclass(frozen=True)
class Ctx:
    cfg: deploy.DeployConfig
    settings: Settings            # the gateway's settings (token overheads)
    run_dir: Path                 # this itest run's directory
    vllm_pid: int | None
    gateway_pid: int | None
    lock_path: str = smoke.GPU_LOCK
    proc_root: str = "/proc"
    api_key: str | None = None
    poll_s: float = 0.05
    idle_wait_s: float = 30.0
    token_watch: tuple = (1.0, 3.0)   # seconds after the disconnect: generation tokens must not grow between them
    hold_before_close_s: float = 0.5

    @property
    def gateway_url(self) -> str:
        return self.cfg.gateway_url

    @property
    def vllm_url(self) -> str:
        return self.cfg.vllm_url

    def gateway_client(self, **kw) -> httpx.AsyncClient:
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.AsyncClient(base_url=self.gateway_url, trust_env=False, headers=headers,
                                 timeout=kw.pop("timeout", loadtest.REQUEST_TIMEOUT), **kw)

    def vllm_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.vllm_url, trust_env=False, timeout=httpx.Timeout(30.0))


async def _served(gw: httpx.AsyncClient) -> tuple:
    """(readyz body, Limits, served model id)."""
    ready = (await gw.get("/readyz")).json()
    model = (await gw.get("/v1/models")).json()["data"][0]["id"]
    return ready, loadtest.Limits.from_readyz(ready), model


def _sse(text: str) -> list:
    return [e.encode() + b"\n\n" for e in text.split("\n\n") if e.strip()]


async def it1(ctx: Ctx) -> dict:
    c = Checks()
    async with ctx.gateway_client() as gw:
        r = await gw.get("/readyz")
        ready = r.json() if r.status_code == 200 else {}
        c.add("gateway /readyz is 200", r.status_code == 200, r.text[:300])
        profile = ready.get("profile") or {}
        c.add("readyz names the profile and its status", profile.get("status") in ("verified", "provisional"),
              profile)
        r = await gw.get("/v1/models")
        ids = [m.get("id") for m in r.json().get("data", [])] if r.status_code == 200 else []
        c.add("/v1/models lists only the served model", len(ids) == 1, ids)
        model = ids[0] if ids else bp.load_pins()["core_b"]["repo"]
        header = f"{profile.get('id')}; status={profile.get('status')}"

        chat = {"model": model, "messages": [{"role": "user", "content": "Reply with one word."}],
                "max_completion_tokens": 16}
        r = await gw.post("/v1/chat/completions", json=chat)
        msg = (r.json().get("choices") or [{}])[0].get("message") if r.status_code == 200 else None
        c.add("chat non-stream", r.status_code == 200 and isinstance(msg, dict), f"{r.status_code} {r.text[:200]}")
        c.add("responses carry x-capserve-profile", r.headers.get("x-capserve-profile") == header,
              r.headers.get("x-capserve-profile"))
        r = await gw.post("/v1/chat/completions", json={**chat, "stream": True})
        events = _sse(r.text) if r.status_code == 200 else []
        c.add("chat stream ends with [DONE]", r.status_code == 200 and events and loadtest._is_done(events[-1])
              and not any(is_error(e) for e in events), f"{r.status_code}, {len(events)} events")

        comp = {"model": model, "prompt": PROMPT_UNIT * 16, "max_tokens": 8, "ignore_eos": True, "temperature": 0.0}
        r = await gw.post("/v1/completions", json=comp)
        doc = r.json() if r.status_code == 200 else {}
        c.add("completion non-stream", r.status_code == 200 and bool(doc.get("choices")), f"{r.status_code}")
        usage = doc.get("usage")
        if usage is not None:   # the real vLLM reports usage; ignore_eos must give exactly max_tokens
            c.add("ignore_eos yields exactly max_tokens", usage.get("completion_tokens") == 8, usage)
        r = await gw.post("/v1/completions", json={**comp, "stream": True})
        events = _sse(r.text) if r.status_code == 200 else []
        c.add("completion stream ends with [DONE]", r.status_code == 200 and events and loadtest._is_done(events[-1])
              and any(has_content(e) for e in events) and not any(is_error(e) for e in events),
              f"{r.status_code}, {len(events)} events")
        r = await gw.get("/metrics")
        c.add("gateway /metrics has capserve_profile_info", r.status_code == 200 and "capserve_profile_info" in r.text)
    for name, pid in (("vLLM", ctx.vllm_pid), ("gateway", ctx.gateway_pid)):
        fds = procs.fds_on(pid, ctx.lock_path, proc_root=ctx.proc_root) if pid else []
        c.add(f"{name} holds the GPU lock descriptor", bool(fds), f"pid {pid}: fds {fds} -> {ctx.lock_path}")
    return c.result("IT-1", profile=profile)


async def it2(ctx: Ctx) -> dict:
    sub = ctx.run_dir / "it2"
    sub.mkdir()
    s = await loadtest.run_async("overload", gateway_url=ctx.gateway_url, vllm_url=ctx.vllm_url, run_dir=sub,
                                 time_limit_s=IT2_TIME_LIMIT_S, api_key=ctx.api_key, sample_interval_s=ctx.poll_s)
    checks, lim = s.get("checks") or {}, s.get("limits") or {}
    c = Checks()
    c.add("overload sends 3 x (limit + queue) clients",
          bool(lim) and s.get("clients", 0) >= 3 * (lim["max_concurrency"] + lim["max_queue"]),
          {"clients": s.get("clients"), "limits": lim, "problems": s.get("problems")})
    c.add("vLLM running + waiting never above max_concurrency", checks.get("vllm_within_limit") is True, s.get("vllm"))
    c.add("only 200 and 429", checks.get("statuses_allowed") is True, s.get("statuses"))
    c.add("no 5xx", checks.get("no_5xx") is True, s.get("statuses"))
    c.add("every 429 carries Retry-After", checks.get("retry_after_on_every_rejection") is True,
          s.get("retry_after_missing"))
    c.add("at least one 429 and one 200", checks.get("rejection_seen") is True and checks.get("some_success") is True,
          s.get("statuses"))
    c.add("no transport errors", checks.get("no_transport_errors") is True, s.get("errors"))
    c.add("streams complete", checks.get("streams_complete") is True)
    c.add("within the time limit", checks.get("within_time_limit") is True)
    return c.result("IT-2", summary=str(sub / "summary.json"), statuses=s.get("statuses"), vllm=s.get("vllm"))


async def _vllm_load(vc: httpx.AsyncClient) -> tuple:
    """(running + waiting, generation_tokens_total) from vLLM /metrics; None where unreadable."""
    try:
        r = await vc.get("/metrics")
    except httpx.HTTPError:
        return None, None
    if r.status_code != 200:
        return None, None
    m = rb.parse_prometheus(r.text)
    running, waiting = rb.gauge(m, "vllm:num_requests_running"), rb.gauge(m, "vllm:num_requests_waiting")
    load = None if running is None or waiting is None else running + waiting
    return load, rb.gauge(m, "vllm:generation_tokens_total")


async def _until_load(vc, pred, limit_s: float, poll_s: float) -> float | None:
    """Seconds until pred(running + waiting) held, or None within limit_s."""
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    while True:
        load, _ = await _vllm_load(vc)
        if load is not None and pred(load):
            return loop.time() - t0
        if loop.time() - t0 >= limit_s:
            return None
        await asyncio.sleep(poll_s)


async def _until_gateway_idle(gw, limit_s: float, poll_s: float) -> bool:
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    while loop.time() - t0 < limit_s:
        r = await gw.get("/readyz")
        if r.json().get("in_flight") == 0:
            return True
        await asyncio.sleep(poll_s)
    return False


async def _disconnect(ctx: Ctx, c: Checks, vc, gw, case: str, model: str, limits, itl: float | None) -> dict:
    loop = asyncio.get_running_loop()
    clock = loop.time
    idle = await _until_load(vc, lambda x: x == 0, ctx.idle_wait_s, ctx.poll_s)
    c.add(f"{case}: vLLM idle before the request", idle is not None)
    body = {"model": model, "prompt": PROMPT_UNIT * 16, "max_tokens": limits.max_output_tokens, "ignore_eos": True,
            "temperature": 0.0, "stream": case == "stream"}
    out = {"max_tokens": body["max_tokens"]}
    async with ctx.gateway_client() as one:
        if case == "stream":
            times = []
            async with one.stream("POST", "/v1/completions", json=body) as resp:
                out["status"] = resp.status_code
                framer = SSEFramer()
                async for chunk in resp.aiter_bytes():
                    times += [clock() for e in framer.feed(chunk) if has_content(e)]
                    if len(times) >= READ_EVENTS:
                        break
            t_close = clock()   # leaving the block closed the connection mid-stream
            itl = (times[-1] - times[0]) / (len(times) - 1) if len(times) >= 2 else None
            remaining = None if itl is None else (body["max_tokens"] - len(times)) * itl
            out.update(events_read=len(times), itl_s=itl)
        else:
            task = asyncio.create_task(one.post("/v1/completions", json=body))
            t0 = clock()
            started = await _until_load(vc, lambda x: x >= 1, 30.0, ctx.poll_s)
            c.add(f"{case}: the request reached vLLM", started is not None)
            await asyncio.sleep(ctx.hold_before_close_s)
            elapsed = clock() - t0
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
                await task
            t_close = clock()
            remaining = None if itl is None else body["max_tokens"] * itl - elapsed
    drop = await _until_load(vc, lambda x: x == 0, DISCONNECT_WINDOW_S, ctx.poll_s)
    c.add(f"{case}: vLLM running + waiting back to 0 within 2 s", drop is not None,
          None if drop is None else f"{drop:.2f} s")
    c.add(f"{case}: the request would have run past the window (decisive)",
          remaining is not None and remaining >= DECISIVE_FACTOR * DISCONNECT_WINDOW_S,
          None if remaining is None else f"~{remaining:.1f} s of generation left at the disconnect")
    first, second = ctx.token_watch
    await asyncio.sleep(max(0.0, t_close + first - clock()))
    _, g1 = await _vllm_load(vc)
    await asyncio.sleep(max(0.0, t_close + second - clock()))
    _, g2 = await _vllm_load(vc)
    c.add(f"{case}: no tokens generated after the disconnect", g1 is not None and g1 == g2,
          f"generation_tokens_total {g1} at +{first:g} s, {g2} at +{second:g} s")
    c.add(f"{case}: gateway in_flight back to 0", await _until_gateway_idle(gw, DISCONNECT_WINDOW_S, ctx.poll_s))
    out.update(drop_s=drop, remaining_s=remaining)
    return out


async def it3(ctx: Ctx) -> dict:
    c = Checks()
    async with ctx.gateway_client() as gw, ctx.vllm_client() as vc:
        _, limits, model = await _served(gw)
        stream = await _disconnect(ctx, c, vc, gw, "stream", model, limits, None)
        plain = await _disconnect(ctx, c, vc, gw, "non-stream", model, limits, stream.get("itl_s"))
    return c.result("IT-3", stream=stream, non_stream=plain)


def _message_text(content) -> str:
    """A message's text exactly as the byte bound counts it (tokens.PromptBudget.upper_bound)."""
    return content if isinstance(content, str) else "".join(part["text"] for part in content)


async def _text_tokens(vc: httpx.AsyncClient, model: str, text: str) -> int:
    """The tokens of a message's text alone (/tokenize, add_special_tokens false); an empty text has none."""
    if not text:
        return 0
    r = await vc.post("/tokenize", json={"model": model, "prompt": text, "add_special_tokens": False})
    count = r.json().get("count") if r.status_code == 200 else None
    if not isinstance(count, int):
        raise TokenizeError(r.status_code, f"/tokenize of a message text returned {r.status_code} without a count", r)
    return count


async def it4(ctx: Ctx) -> dict:
    c = Checks()
    tokens = ctx.settings.tokens.model_dump()
    model = bp.load_pins()["core_b"]["repo"]
    measured_completion, measured_chat = [], []
    async with ctx.vllm_client() as vc:
        budget = PromptBudget(vc, **tokens)
        for n in COMPLETION_NS:
            body = {"model": model, "prompt": PROMPT_UNIT * n}
            try:
                count = await budget.count("completions", body)
            except (TokenizeError, httpx.HTTPError) as exc:
                c.add(f"' the' x {n} is {n} tokens", False, f"/tokenize failed: {exc}")
                continue
            bound = budget.upper_bound("completions", body)
            c.add(f"' the' x {n} is {n} tokens", count == n, count)
            c.add(f"completion bound holds at n={n}", count <= bound, f"{count} <= {bound}")
            measured_completion.append({"n": n, "count": count, "bound": bound})
        for name, messages in CHAT_CASES:
            body = {"model": model, "messages": messages}
            try:
                count = await budget.count("chat", body)
                content = sum([await _text_tokens(vc, model, _message_text(m["content"])) for m in messages])
            except (TokenizeError, httpx.HTTPError) as exc:
                c.add(f"chat bound holds: {name}", False, f"/tokenize failed: {exc}")
                continue
            bound = budget.upper_bound("chat", body)
            c.add(f"chat bound holds: {name}", count <= bound, f"{count} <= {bound}")
            # 029 IT-4: the template's own tokens within the constants, whatever the text's bytes per token
            template = count - content
            allowed = budget.chat_overhead + sum(budget.per_message + len(m["role"].encode("utf-8")) for m in messages)
            c.add(f"chat template within the overhead constants: {name}", template <= allowed,
                  f"{template} <= {allowed} (chat_overhead + per_message and role bytes per message)")
            measured_chat.append({"name": name, "messages": len(messages), "count": count, "bound": bound,
                                  "content_tokens": content, "template_tokens": template, "allowed": allowed})
    return c.result("IT-4", settings_tokens=tokens, completion=measured_completion, chat=measured_chat)


def evaluate_it5(records: list, stopped: dict, *, saturated: bool, in_flight_at_stop: int) -> list:
    """The IT-5 checks from the client records (all sent before the stop) and the supervisor's stop result."""
    c = Checks()
    teardown = (stopped.get("outcome") or {}).get("teardown") or {}
    c.add("the gateway was saturated before the stop", saturated, in_flight_at_stop)
    c.add("deploy stop returned 0", stopped.get("stop_exit") == 0, stopped.get("stop_exit"))
    c.add("the supervisor exited 0 or 143 (SIGTERM)", stopped.get("supervisor_exit") in (0, 128 + signal.SIGTERM),
          stopped.get("supervisor_exit"))
    c.add("teardown had no problems", teardown.get("problems") == [], teardown.get("problems"))
    gpu, baseline = teardown.get("gpu_mem_after_mib"), teardown.get("gpu_baseline_mib")
    limit = None if baseline is None else baseline + deploy.IDLE_RETURN_TOL_MIB   # 031, not the fixed 1,000 MiB
    c.add(f"GPU memory back <= the pre-boot baseline + {deploy.IDLE_RETURN_TOL_MIB} MiB (031)",
          gpu is not None and limit is not None and gpu <= limit,
          {"after_mib": gpu, "baseline_mib": baseline, "limit_mib": limit})
    drained = [r for r in records if r["ended_after_stop"] and r["status"] == 200]
    c.add("requests running at the stop completed with 200 (drain)",
          in_flight_at_stop >= 1 and len(drained) >= in_flight_at_stop,
          f"{len(drained)} completed after the stop; {in_flight_at_stop} were running")
    cut = [r["error"] for r in records if r["error"]]
    c.add("no request was cut (no transport error)", not cut, cut[:10])
    statuses = sorted({r["status"] for r in records if r["status"] is not None})
    c.add("only 200, 429 and 503", set(statuses) <= {200, 429, 503}, statuses)
    rejected = [r for r in records if r["status"] in (429, 503)]
    c.add("every 429 / 503 carries Retry-After", all(loadtest._retry_after_ok(r["retry_after"]) for r in rejected),
          sum(not loadtest._retry_after_ok(r["retry_after"]) for r in rejected))
    return c.items


async def it5(ctx: Ctx, *, stop: Callable[[], dict]) -> dict:
    """stop() blocks until the supervisor is gone: {"stop_exit", "supervisor_exit", "outcome"}."""
    loop = asyncio.get_running_loop()
    async with ctx.gateway_client() as probe:
        ready, limits, model = await _served(probe)
    n_clients = limits.max_concurrency + limits.max_queue
    sc = dataclasses.replace(loadtest.SCENARIOS["overload"], max_tokens=limits.max_output_tokens)
    stopping = asyncio.Event()
    records = []

    async def client(i: int, gw) -> None:
        k = 0
        while not stopping.is_set():
            body = loadtest.request_body(sc, limits, model=model, index=i + k)
            r = loadtest._new_record(i, k, i * 1000 + k, body["stream"])
            records.append(r)
            await loadtest._one(gw, body, r, loop.time)
            r["t_end"] = loop.time()
            k += 1

    pool = httpx.Limits(max_connections=n_clients + 4, max_keepalive_connections=n_clients + 4)
    async with ctx.gateway_client(limits=pool) as gw, ctx.gateway_client() as probe:
        tasks = [asyncio.create_task(client(i, gw)) for i in range(n_clients)]
        t0, in_flight = loop.time(), 0
        while loop.time() - t0 < SATURATION_WAIT_S:
            in_flight = (await probe.get("/readyz")).json().get("in_flight", 0)
            if in_flight >= limits.max_concurrency:
                break
            await asyncio.sleep(ctx.poll_s)
        saturated = in_flight >= limits.max_concurrency
        stopping.set()
        t_stop = loop.time()
        stopped = await asyncio.to_thread(stop)
        done, pending = await asyncio.wait(tasks, timeout=ctx.cfg.drain_timeout_s + 60)
        for t in pending:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for r in records:
        r["ended_after_stop"] = r.get("t_end", float("inf")) > t_stop
    c = Checks()
    c.items = evaluate_it5(records, stopped, saturated=saturated, in_flight_at_stop=in_flight)
    return c.result("IT-5", in_flight_at_stop=in_flight, clients=n_clients, statuses=sorted(
        {str(r["status"]) for r in records}), stop=stopped)


# ---------------------------------------------------------------- IT-6: research isolation

def research_paths(cfg: deploy.DeployConfig, *, home: Path) -> list:
    """029 IT-6: the research checkout and the JIT / model caches research boots use."""
    return [Path(cfg.research_dir), home / ".cache" / "vllm", home / ".cache" / "flashinfer",
            home / ".cache" / "huggingface", home / ".triton"]


def it6(*, reference: Path, paths: list, run: Callable = subprocess.run) -> dict:
    c = Checks()
    existing = [p for p in paths if Path(p).exists()]
    missing = [str(p) for p in paths if not Path(p).exists()]
    newer: list = []
    if existing:
        r = run(["find", *map(str, existing), "-newer", str(reference)], capture_output=True, text=True,
                timeout=1800)
        newer = [line for line in r.stdout.splitlines() if line]
        c.add("find ran cleanly", r.returncode == 0, (r.stderr or "")[-500:])
    c.add("nothing in the research checkout or its caches is newer than the reference", not newer,
          f"{len(newer)} path(s)")
    return c.result("IT-6", reference=str(reference), reference_mtime_ns=Path(reference).stat().st_mtime_ns,
                    paths=[str(p) for p in paths], missing=missing, newer=newer[:200], n_newer=len(newer))


# ---------------------------------------------------------------- orchestration

CHECKS = {"IT-1": it1, "IT-2": it2, "IT-3": it3, "IT-4": it4, "IT-5": it5}


def _home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def install_exit_handlers(set_handler: Callable = signal.signal) -> None:
    """SIGTERM / SIGHUP -> SystemExit(128 + signum) (coreb_smoke._exit_on_signal, as capserve.deploy main): an ssh
    hang-up runs the finally blocks, which stop the supervisor and write itest.json."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        set_handler(sig, smoke._exit_on_signal)


@dataclass
class Effects:
    base_env: dict = field(default_factory=lambda: dict(os.environ))
    start_group: Callable = procs.start_group
    held_gpu_lock: Callable = run_adv.held_gpu_lock
    lock_fd: Callable = procs.lock_fd
    python: str = sys.executable
    wall: Callable = deploy.utcnow
    now: Callable = time.monotonic
    sleep: Callable = time.sleep
    stop_supervisor: Callable = deploy.stop
    checks: dict = field(default_factory=lambda: dict(CHECKS))
    it0: Callable = it0
    it6: Callable = it6
    home: Path = field(default_factory=_home)
    proc_root: str = "/proc"
    lock_path: str = smoke.GPU_LOCK
    install_signals: Callable = install_exit_handlers


def _guarded(test: str, fn: Callable[[], dict]) -> dict:
    try:
        return {"test": test, **fn()}
    except Exception as exc:  # noqa: BLE001 - a crashing check is a failed test, never a crashed session
        return _failed(test, f"{type(exc).__name__}: {exc}")


def _supervisor_outcome(state: dict) -> dict | None:
    try:
        return runs.read_meta(state["run_dir"]).get("outcome")
    except (KeyError, TypeError, OSError, ValueError):
        return None


def _wait_ready(cfg, sup, fx: Effects) -> tuple:
    """(state, None) once state.json says this supervisor is ready; (None, why) otherwise."""
    limit = cfg.startup_timeout_s + deploy.GATEWAY_READY_TIMEOUT_S + deploy.READY_OVERHEAD_S + SUPERVISOR_EXTRA_S
    t = fx.now()
    while True:
        state = deploy.read_state(cfg)
        if state and state.get("supervisor_pid") == sup.pid and state.get("phase") == "ready":
            return state, None
        code = sup.poll()
        if code is not None:
            return None, f"the supervisor exited (code {code}) before it was ready"
        if fx.now() - t >= limit:
            return None, f"the supervisor was not ready within {limit:g} s"
        fx.sleep(2)


def _wait_exit(sup, fx: Effects, limit_s: float):
    t = fx.now()
    while sup.poll() is None and fx.now() - t < limit_s:
        fx.sleep(1)
    return sup.poll()


def run_booted(tests: list, cfg, args, run_dir: Path, lock_fd: int, fx: Effects, results: dict,
               supervisor: dict) -> None:
    """One `capserve.deploy up` (hold) for every booted test. results and the supervisor summary are filled in place,
    so an interrupt (SIGTERM / SIGHUP) keeps what finished; the supervisor is stopped in every case."""
    env = {**fx.base_env, "PYTHONPATH": os.pathsep.join([str(deploy.RELEASE_ROOT / "service"),
                                                          str(deploy.RELEASE_ROOT / "benchmarks")]),
           "PYTHONDONTWRITEBYTECODE": "1"}
    argv = [fx.python, "-m", "capserve.deploy", "up", "--config", str(Path(args.config).resolve()), "--profile",
            str(Path(args.profile).resolve())] + (["--allow-provisional"] if args.allow_provisional else [])
    sup = fx.start_group(argv, env, run_dir / "supervisor.log", cwd=deploy.RELEASE_ROOT, keep_fds=(lock_fd,))
    stopped: dict = {}
    state = None

    def stop_and_wait() -> dict:
        if "result" not in stopped:
            code = fx.stop_supervisor(cfg)
            exit_ = _wait_exit(sup, fx, deploy.default_stop_wait_s(cfg) + 30)
            stopped["result"] = {"stop_exit": code, "supervisor_exit": exit_,
                                 "outcome": _supervisor_outcome(deploy.read_state(cfg) or {})}
        return stopped["result"]

    try:
        state, why = _wait_ready(cfg, sup, fx)
        if state is None:
            for t in tests:
                results[t] = _failed(t, f"{why}; see {run_dir / 'supervisor.log'} and the supervisor run's meta.json")
        else:
            ctx = Ctx(cfg=cfg, settings=load_settings(Path(state["run_dir"]) / "gateway.toml"), run_dir=run_dir,
                      vllm_pid=state.get("vllm_pid"), gateway_pid=state.get("gateway_pid"), lock_path=fx.lock_path,
                      proc_root=fx.proc_root, api_key=fx.base_env.get(API_KEY_ENV) or None)
            for t in tests:
                kw = {"stop": stop_and_wait} if t == "IT-5" else {}
                results[t] = _guarded(t, lambda: asyncio.run(fx.checks[t](ctx, **kw)))
    finally:
        if sup.poll() is None and "result" not in stopped:
            stop_and_wait()
        final = stopped.get("result") or {"stop_exit": None, "supervisor_exit": sup.poll(),
                                          "outcome": _supervisor_outcome(state or {})}
        teardown = (final["outcome"] or {}).get("teardown")
        clean = final["supervisor_exit"] in (0, 128 + signal.SIGTERM) and (teardown or {}).get("problems") == []
        supervisor.update({"exit": final["supervisor_exit"], "stop_exit": final["stop_exit"], "teardown": teardown,
                           "run_dir": (state or {}).get("run_dir"), "ok": clean})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m capserve.itest",
                                description="RTX integration tests IT-0..IT-6 of decision 029 (product tests only)")
    p.add_argument("tests", nargs="+", choices=TESTS, help="tests to run (always in IT-0..IT-6 order)")
    p.add_argument("--config", default=str(deploy.DEFAULT_CONFIG), help="deploy config (TOML)")
    p.add_argument("--profile", help="capacity profile for the boot (JSON; default: the config's production_profile, "
                                     "else service/profiles/provisional-rtx.json)")
    p.add_argument("--allow-provisional", action="store_true", help="allow a provisional profile (not product limits)")
    p.add_argument("--since", help="IT-6 reference: an earlier product run id (default: this itest run)")
    return p


def main(argv: list | None = None, *, fx: Effects | None = None) -> int:
    fx = fx or Effects()
    args = parser().parse_args(argv)
    fx.install_signals()   # before any run dir or boot: a hang-up still stops the supervisor and writes itest.json
    tests = [t for t in TESTS if t in set(args.tests)]
    booted = [t for t in tests if t in BOOTED]
    locked = [t for t in tests if t != "IT-0"]   # 029: only IT-0 runs without the GPU lock

    def say(msg: str) -> None:
        print(f"capserve.itest: {msg}", file=sys.stderr, flush=True)

    try:
        cfg = deploy.load_config(args.config)
    except deploy.DeployError as exc:
        say(str(exc))
        return 2
    if args.profile is None:   # 036: a production config tests the verified profile it pins
        args.profile = str(deploy.RELEASE_ROOT / cfg.production_profile) if cfg.production_profile else \
            str(DEFAULT_PROFILE)
    problems = deploy.production_profile_problems(cfg, args.profile, allow_provisional=args.allow_provisional,
                                                  release_root=deploy.RELEASE_ROOT)
    if problems:
        say("; ".join(problems) + "; nothing started")
        return 2
    lock_fd = None
    if locked:
        lock_fd = fx.lock_fd() if fx.held_gpu_lock() else None
        if lock_fd is None:
            say(f"{', '.join(locked)} need the GPU lock (029: only IT-0 runs without it; 027 §5: never inside a "
                f"formal research session): run under `flock -n {smoke.GPU_LOCK} ...`; nothing started")
            return 2
    reference = None
    if args.since is not None:
        problems = runs.run_path_problems(args.since, service_root=cfg.service_root, research_dir=cfg.research_dir)
        reference = runs.runs_root(cfg.service_root) / args.since / "meta.json"
        problems = [p for p in problems if "exists" not in p]   # it must exist here
        if problems or not reference.is_file():
            say(f"--since {args.since!r}: not a product run with a meta.json ({'; '.join(problems) or reference})")
            return 2
    now = fx.wall()
    try:
        run_dir = runs.create_run(service_root=cfg.service_root, research_dir=cfg.research_dir,
                                  run_id=runs.new_run_id(now, "itest"), kind="itest", release=loadtest.release_sha(),
                                  profile={"path": str(Path(args.profile).resolve())}, now_utc=now,
                                  extra={"tests": tests, "since": args.since})
    except runs.RunError as exc:
        say(str(exc))
        return 2
    reference = reference or run_dir / "meta.json"
    results, supervisor, interrupted = {}, ({} if booted else None), None
    try:
        if "IT-0" in tests:
            results["IT-0"] = _guarded("IT-0", fx.it0)
        if booted:
            run_booted(booted, cfg, args, run_dir, lock_fd, fx, results, supervisor)
        if "IT-6" in tests:
            results["IT-6"] = _guarded("IT-6", lambda: fx.it6(reference=reference,
                                                               paths=research_paths(cfg, home=fx.home)))
    except BaseException as exc:   # SIGTERM / SIGHUP (SystemExit), Ctrl-C: record what finished, then go on
        interrupted = f"{type(exc).__name__}({exc})"
        raise
    finally:
        for t in tests:
            results.setdefault(t, _failed(t, f"interrupted before it finished ({interrupted})"))
        ok = (interrupted is None and all(r["ok"] for r in results.values())
              and (supervisor is None or bool(supervisor.get("ok"))))
        doc = {"schema_version": 1, "purpose": runs.PURPOSE, "note": runs.NOTE, "research_valid": False,
               "tests": {t: results[t] for t in tests}, "supervisor": supervisor, "interrupted": interrupted,
               "ok": ok}
        (run_dir / "itest.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        runs.update_meta(run_dir, outcome={"ok": ok, "tests": {t: results[t]["ok"] for t in tests},
                                           "supervisor": None if supervisor is None else supervisor.get("ok"),
                                           "interrupted": interrupted})
        print(json.dumps({"run_dir": str(run_dir), "ok": ok, "tests": {t: results[t]["ok"] for t in tests},
                          "interrupted": interrupted, "supervisor": supervisor and {
                              "exit": supervisor.get("exit"), "ok": supervisor.get("ok")}}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
