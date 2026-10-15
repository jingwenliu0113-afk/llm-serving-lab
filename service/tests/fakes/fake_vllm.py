"""A fake vLLM 0.30.0 OpenAI server for the capserve tests (decision 029 "Mac：假的 vLLM 後端"): no GPU, no model.

Endpoints: /health, /v1/models, /tokenize (count = whitespace-separated words), /v1/completions and
/v1/chat/completions in stream and non-stream mode. FakeConfig scripts the behaviour: ttft_s / itl_s / n_tokens,
fail_status (+ fail_body), drop_after_chunks (the connection breaks inside the next event), split_events (each event
in two writes), pause_after_chunks (the stream waits for `release` after that many events), hang (a generation
waits for `release` before its first token), healthy / health_delay_s, model_id, tokenize_fail_status (/tokenize
answers with vLLM's OpenAI-shaped error, as it does when the chat template raises) and tokenize_delay_s.

/metrics exposes vllm:num_requests_running (generations in flight; /tokenize does not create an engine request),
vllm:num_requests_waiting (always 0: the fake has no scheduler queue), vllm:generation_tokens_total (one per generated
step) and vllm:cache_config_info at the Stage B values.

It records peak_in_flight (generations and /tokenize calls together: every HTTP request that does model work),
cancelled_ids (requests whose client disconnected before the end), the request bodies and headers, and exposes
asyncio Events (started, cancelled, release) so tests synchronize on events, not on wall-clock sleeps. Runs in the
test's event loop (servers.serve).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import anyio
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import core_b_plan as bp

_PINS = bp.load_pins()
MODEL = _PINS["core_b"]["repo"]
NUM_GPU_BLOCKS, BLOCK_SIZE = _PINS["stage_b"]["num_gpu_blocks"], _PINS["stage_b"]["block_size"]


class _Drop(Exception):
    """Raised inside a streaming response: uvicorn closes the connection mid-body (an upstream crash)."""


@dataclass
class FakeConfig:
    model_id: str = MODEL
    ttft_s: float = 0.0
    itl_s: float = 0.0
    n_tokens: int = 4
    fail_status: int | None = None
    fail_body: bytes | None = None
    drop_after_chunks: int | None = None
    split_events: bool = False
    pause_after_chunks: int | None = None
    hang: bool = False
    healthy: bool = True
    health_delay_s: float = 0.0
    tokenize_fail_status: int | None = None
    tokenize_delay_s: float = 0.0


@dataclass
class Seen:
    path: str
    body: dict
    headers: dict = field(default_factory=dict)


class FakeVLLM:
    def __init__(self, cfg: FakeConfig | None = None):
        self.cfg = cfg or FakeConfig()
        self.in_flight = 0
        self.running = 0          # generations only (vllm:num_requests_running)
        self.tokens_generated = 0  # vllm:generation_tokens_total
        self.peak_in_flight = 0
        self.seen: list[Seen] = []
        self.cancelled_ids: list[int] = []
        self.completed_ids: list[int] = []
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self._next_id = 0
        self.app = Starlette(routes=[
            Route("/health", self.health, methods=["GET"]),
            Route("/v1/models", self.models, methods=["GET"]),
            Route("/tokenize", self.tokenize, methods=["POST"]),
            Route("/metrics", self.metrics, methods=["GET"]),
            Route("/v1/completions", self.generate, methods=["POST"]),
            Route("/v1/chat/completions", self.generate, methods=["POST"]),
        ])

    # ------------------------------------------------------------ bookkeeping

    def _enter(self, generation: bool = True) -> int:
        self._next_id += 1
        self.in_flight += 1
        self.running += int(generation)
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        self.started.set()
        return self._next_id

    def _leave(self, generation: bool = True) -> None:
        self.in_flight -= 1
        self.running -= int(generation)

    def _cancel(self, rid: int) -> None:
        self.cancelled_ids.append(rid)
        self.cancelled.set()

    @property
    def generations(self) -> list:
        return [s for s in self.seen if s.path != "/tokenize"]

    # ------------------------------------------------------------ endpoints

    async def health(self, request: Request) -> Response:
        if self.cfg.health_delay_s:
            await asyncio.sleep(self.cfg.health_delay_s)
        return Response(status_code=200 if self.cfg.healthy else 503)

    async def metrics(self, request: Request) -> Response:
        labels = f'model_name="{self.cfg.model_id}",engine="0"'
        text = (f"vllm:num_requests_running{{{labels}}} {self.running}\n"
                f"vllm:num_requests_waiting{{{labels}}} 0\n"
                f"vllm:generation_tokens_total{{{labels}}} {self.tokens_generated}\n"
                f'vllm:cache_config_info{{block_size="{BLOCK_SIZE}",num_gpu_blocks="{NUM_GPU_BLOCKS}"}} 1.0\n')
        return Response(text, media_type="text/plain; version=0.0.4")

    async def models(self, request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": self.cfg.model_id, "object": "model"}]})

    async def tokenize(self, request: Request) -> Response:
        body = await request.json()
        self.seen.append(Seen("/tokenize", body, dict(request.headers)))
        self._enter(generation=False)
        try:
            if self.cfg.tokenize_delay_s:
                await asyncio.sleep(self.cfg.tokenize_delay_s)
            if self.cfg.tokenize_fail_status is not None:
                return JSONResponse({"error": {"message": "fake tokenize failure", "type": "BadRequestError",
                                               "param": None, "code": self.cfg.tokenize_fail_status}},
                                    status_code=self.cfg.tokenize_fail_status)
            if "messages" in body:
                text = " ".join(m["content"] if isinstance(m["content"], str) else
                                " ".join(p["text"] for p in m["content"]) for m in body["messages"])
            else:
                text = body["prompt"]
            n = len(text.split())
            return JSONResponse({"count": n, "max_model_len": 32768, "tokens": list(range(n))})
        finally:
            self._leave(generation=False)

    async def generate(self, request: Request):
        body = await request.json()
        self.seen.append(Seen(request.url.path, body, dict(request.headers)))
        chat = request.url.path.endswith("/chat/completions")
        if self.cfg.fail_status is not None:
            content = self.cfg.fail_body or json.dumps(
                {"error": {"message": "fake failure", "type": "BadRequestError", "param": None,
                           "code": self.cfg.fail_status}}).encode()
            return Response(content, status_code=self.cfg.fail_status, media_type="application/json")
        return (_Stream if body.get("stream") else _NonStream)(self, chat)

    async def first_token_wait(self) -> None:
        if self.cfg.hang:
            await self.release.wait()
        if self.cfg.ttft_s:
            await asyncio.sleep(self.cfg.ttft_s)


def _completion(chat: bool, rid: int, text: str) -> dict:
    if chat:
        return {"id": f"chatcmpl-{rid}", "object": "chat.completion", "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}]}
    return {"id": f"cmpl-{rid}", "object": "text_completion", "model": MODEL,
            "choices": [{"index": 0, "text": text, "finish_reason": "length"}]}


def _events(chat: bool, rid: int, n: int) -> list:
    def ev(payload):
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    if chat:
        out = [ev({"id": f"chatcmpl-{rid}", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})]
        out += [ev({"id": f"chatcmpl-{rid}", "choices": [{"index": 0, "delta": {"content": f" t{i}"}}]})
                for i in range(n)]
        out.append(ev({"id": f"chatcmpl-{rid}", "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}))
    else:
        out = [ev({"id": f"cmpl-{rid}", "choices": [{"index": 0, "text": f" t{i}", "finish_reason": None}]})
               for i in range(n)]
        out.append(ev({"id": f"cmpl-{rid}", "choices": [{"index": 0, "text": "", "finish_reason": "length"}]}))
    return out + [b"data: [DONE]\n\n"]


class _NonStream:
    def __init__(self, fake: FakeVLLM, chat: bool):
        self.fake, self.chat = fake, chat

    async def __call__(self, scope, receive, send):
        fake = self.fake
        rid = fake._enter()
        gone = False
        try:
            async with anyio.create_task_group() as tg:
                async def watch():
                    nonlocal gone
                    while (await receive())["type"] != "http.disconnect":
                        pass
                    gone = True
                    tg.cancel_scope.cancel()

                tg.start_soon(watch)
                await fake.first_token_wait()
                for _ in range(fake.cfg.n_tokens):
                    if fake.cfg.itl_s:
                        await asyncio.sleep(fake.cfg.itl_s)
                    fake.tokens_generated += 1
                tg.cancel_scope.cancel()
            if gone:
                fake._cancel(rid)
                return
            text = "".join(f" t{i}" for i in range(fake.cfg.n_tokens))
            await JSONResponse(_completion(self.chat, rid, text))(scope, receive, send)
            fake.completed_ids.append(rid)
        finally:
            fake._leave()


class _Stream:
    def __init__(self, fake: FakeVLLM, chat: bool):
        self.fake, self.chat = fake, chat

    async def __call__(self, scope, receive, send):
        fake = self.fake
        cfg = fake.cfg
        rid = fake._enter()
        gone = finished = False
        try:
            async with anyio.create_task_group() as tg:
                async def watch():
                    nonlocal gone
                    while (await receive())["type"] != "http.disconnect":
                        pass
                    gone = not finished   # h11 also reports disconnect once the response is complete
                    tg.cancel_scope.cancel()

                tg.start_soon(watch)
                await send({"type": "http.response.start", "status": 200,
                            "headers": [(b"content-type", b"text/event-stream; charset=utf-8")]})
                await fake.first_token_wait()
                for i, event in enumerate(_events(self.chat, rid, cfg.n_tokens)):
                    if cfg.drop_after_chunks is not None and i == cfg.drop_after_chunks:
                        await send({"type": "http.response.body", "body": event[:len(event) // 2], "more_body": True})
                        raise _Drop("fake upstream crash")
                    if cfg.pause_after_chunks is not None and i == cfg.pause_after_chunks:
                        await fake.release.wait()
                    if i and cfg.itl_s:
                        await asyncio.sleep(cfg.itl_s)
                    fake.tokens_generated += 1
                    parts = [event[:len(event) // 2], event[len(event) // 2:]] if cfg.split_events else [event]
                    for part in parts:
                        await send({"type": "http.response.body", "body": part, "more_body": True})
                        if cfg.split_events:
                            await asyncio.sleep(0.001)
                finished = True
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                tg.cancel_scope.cancel()
            if gone:
                fake._cancel(rid)
            else:
                fake.completed_ids.append(rid)
        except BaseExceptionGroup as group:
            raise group.exceptions[0] if len(group.exceptions) == 1 else group
        finally:
            fake._leave()
