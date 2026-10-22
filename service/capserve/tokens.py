"""Prompt-length bound: a byte fast path before admission, vLLM /tokenize only under a permit (decision 029 steps 6, 8).

- Fast path: Qwen3.5's byte-level BPE produces at most one token per UTF-8 byte, so a prompt whose byte count plus
  the template overheads (settings [tokens]) is within max_prompt_tokens cannot exceed it; no tokenization needed.
  A token-id prompt is counted exactly. The chat overheads are deliberately generous constants until RTX IT-4
  measures them; an overhead that is too large only costs a /tokenize call.
- Otherwise, with settings tokenizer "vllm" (the default), the gateway asks vLLM's /tokenize, with the same messages,
  add_generation_prompt and chat_template_kwargs (chat; 038 §1: enable_thinking changes the generation prompt) or the
  same prompt with add_special_tokens (completions) that the generation request uses, while holding a permit, so
  /tokenize traffic counts against max_concurrency and never reaches vLLM for a request that is later rejected.
- With tokenizer "local" (038 §4) the gateway counts itself (LocalTokenizer), from the snapshot vLLM serves, as
  vLLM 0.30.0 does: chat = the tokenizer's chat template (tokenize false, add_generation_prompt true, the request's
  chat_template_kwargs) encoded without special tokens (vLLM's chat default), completions = the prompt encoded with
  them. It never reaches vLLM, so it runs before admission, in a worker thread. The gateway sends string content
  only (openai_api joins text parts), so no content-format detection can make vLLM render other messages. Whether
  the counts equal vLLM's is RTX IT-4's question; the production config switches to local only after it passes.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import httpx


class TokenizeError(RuntimeError):
    """/tokenize failed; response is vLLM's answer (a 4xx is the client's fault and is passed through)."""

    def __init__(self, status: int, message: str, response: httpx.Response):
        super().__init__(message)
        self.status = status
        self.response = response


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(part["text"] for part in content)


class TokenizerLoadError(RuntimeError):
    """The local tokenizer cannot be loaded (no snapshot, no transformers): the gateway refuses to start."""


class TemplateRejected(ValueError):
    """The chat template raised on these messages (as vLLM's /tokenize answers 400)."""


class LocalTokenizer:
    def __init__(self, hf):
        self.hf = hf   # a transformers tokenizer (or a test stand-in with apply_chat_template / encode)

    @classmethod
    def load(cls, path: str) -> "LocalTokenizer":
        if not Path(path).is_dir():
            raise TokenizerLoadError(f"{path}: no tokenizer snapshot directory (settings tokenizer_path)")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenizerLoadError(f"the local tokenizer needs transformers: {exc}") from exc
        try:
            return cls(AutoTokenizer.from_pretrained(path, local_files_only=True))
        except Exception as exc:   # noqa: BLE001 - whatever the snapshot's problem, the gateway must not start
            raise TokenizerLoadError(f"{path}: cannot load the tokenizer: {exc}") from exc

    def count(self, route: str, body: dict) -> int:
        if route == "chat":
            try:
                text = self.hf.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True,
                                                   **(body.get("chat_template_kwargs") or {}))
            except Exception as exc:   # noqa: BLE001 - the template's raise_exception, on the client's messages
                raise TemplateRejected(str(exc)) from exc
            return len(self.hf.encode(text, add_special_tokens=False))
        prompt = body["prompt"]
        if isinstance(prompt, list):
            return len(prompt)
        return len(self.hf.encode(prompt, add_special_tokens=True))


class PromptBudget:
    def __init__(self, client: httpx.AsyncClient, *, chat_overhead: int, per_message: int, completion_overhead: int,
                 local: LocalTokenizer | None = None):
        self.client = client
        self.chat_overhead = chat_overhead
        self.per_message = per_message
        self.completion_overhead = completion_overhead
        self.local = local

    def upper_bound(self, route: str, body: dict) -> int:
        """An upper bound on the prompt tokens of a validated body (exact for token-id prompts)."""
        if route == "chat":
            return self.chat_overhead + sum(self.per_message + _nbytes(m["role"]) + _nbytes(_text(m["content"]))
                                            for m in body["messages"])
        prompt = body["prompt"]
        if isinstance(prompt, list):
            return len(prompt)
        return _nbytes(prompt) + self.completion_overhead

    def fast_fits(self, route: str, body: dict, max_prompt_tokens: int) -> bool:
        return self.upper_bound(route, body) <= max_prompt_tokens

    async def count(self, route: str, body: dict) -> int:
        """The prompt's token count: the local tokenizer's (in a worker thread), else vLLM /tokenize's (token-id
        prompts need no request)."""
        if self.local is not None:
            return await anyio.to_thread.run_sync(self.local.count, route, body)
        if route == "chat":
            payload = {"model": body["model"], "messages": body["messages"], "add_generation_prompt": True}
            if "chat_template_kwargs" in body:   # 038 §1: the same template arguments as the generation request
                payload["chat_template_kwargs"] = body["chat_template_kwargs"]
        elif isinstance(body["prompt"], list):
            return len(body["prompt"])
        else:
            payload = {"model": body["model"], "prompt": body["prompt"], "add_special_tokens": True}
        resp = await self.client.post("/tokenize", json=payload)
        if resp.status_code != 200:
            raise TokenizeError(resp.status_code, f"/tokenize returned {resp.status_code}", resp)
        try:
            count = resp.json()["count"]
        except (ValueError, KeyError, TypeError) as exc:
            raise TokenizeError(resp.status_code, "/tokenize returned no count", resp) from exc
        if not isinstance(count, int):
            raise TokenizeError(resp.status_code, "/tokenize returned no count", resp)
        return count
