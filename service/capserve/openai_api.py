"""The OpenAI-compatible request surface and error objects (decision 029 "OpenAI 相容介面").

Allowlist (MVP); every other field is 400 unsupported_parameter, because it changes the load or the number of
sequences vLLM serves and the research never measured it (tools, response_format / structured outputs, logprobs,
echo, best_of, beam search, chat_template, truncate_prompt_tokens, vLLM sampling extensions, ...):
- common: model, stream, stream_options (include_usage), max_tokens, temperature, top_p, stop, seed,
  presence_penalty, frequency_penalty, user, n (1 only);
- chat: messages (system / user / assistant, text only; text parts are joined into one string, without a separator,
  as Qwen3.5's template renders them: 038 §4), max_completion_tokens, and chat_template_kwargs with
  enable_thinking only (decision 038 §1; any other template argument changes the prompt the research never measured);
- completions: prompt (exactly one: a string, a token-id list, or a one-element list of either), ignore_eos,
  min_tokens (<= max_tokens).
A field sent as null is ignored, as are logprobs / echo / use_beam_search sent as false (their defaults).
A body vLLM could not be sent is 400 too: non-finite numbers, lone surrogates, JSON nested deeper than the parser.

Thinking (038 §1, user decision 037): every chat request reaches vLLM with chat_template_kwargs.enable_thinking set,
to the client's value or to THINKING_DEFAULT (false); Qwen3.5's template then opens the answer with an empty
think block instead of a thinking one. Completions never carry it (a raw prompt has no template).

max_tokens: missing -> the active class's max_output_tokens is injected (vLLM's own default would be the rest of the
context; documented in service/README.md); above it -> 400 max_tokens_exceeded. Chat forwards the bound as
max_completion_tokens only (vLLM reads max_completion_tokens or max_tokens; max_tokens is deprecated there).
The body sent to vLLM is re-serialized from the validated model, never the client's bytes.

Errors are {"error": {"message", "type", "param", "code"}}; rejections (429 / 503) add Retry-After (whole seconds,
rounded up), retry-after-ms and x-should-retry: true. Messages never echo request content.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import JSONResponse

ROUTES = ("chat", "completions")
THINKING_DEFAULT = False          # 038 §1: thinking off unless the client turns it on
_OFF_DEFAULTS = {"logprobs": False, "echo": False, "use_beam_search": False}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, *, param: str | None = None,
                 type_: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.code, self.message, self.param, self.type = status, code, message, param, type_

    def response(self) -> JSONResponse:
        return openai_error(self.status, self.type, self.code, self.message, param=self.param)


def error_body(type_: str, code: str, message: str, *, param: str | None = None) -> dict:
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


def retry_headers(retry_after_s: float) -> dict:
    """Retry-After in whole seconds, rounded up so a client never retries early (an adaptive estimate, 038 §3, is not
    a whole number); retry-after-ms keeps the estimate."""
    return {"Retry-After": str(math.ceil(retry_after_s)), "retry-after-ms": str(round(retry_after_s * 1000)),
            "x-should-retry": "true"}


def openai_error(status: int, type_: str, code: str, message: str, *, param: str | None = None,
                 retry_after_s: float | None = None) -> JSONResponse:
    headers = retry_headers(retry_after_s) if retry_after_s is not None else None
    return JSONResponse(error_body(type_, code, message, param=param), status_code=status, headers=headers)


# ---------------------------------------------------------------- request models

class _Strict(BaseModel):
    # allow_inf_nan=False: json.loads reads NaN / Infinity (and 1e999 as inf), but the body sent upstream must be JSON
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class StreamOptions(_Strict):
    include_usage: bool | None = None


class TextPart(_Strict):
    type: Literal["text"]
    text: str


class ChatMessage(_Strict):
    role: Literal["system", "user", "assistant"]
    content: str | list[TextPart]


class ChatTemplateKwargs(_Strict):
    enable_thinking: bool | None = None


class _Common(_Strict):
    model: str
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(None, ge=1)
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    n: int | None = None


class ChatBody(_Common):
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    chat_template_kwargs: ChatTemplateKwargs | None = None


class CompletionBody(_Common):
    prompt: str | list[int] | list[str] | list[list[int]]
    ignore_eos: bool | None = None
    min_tokens: int | None = Field(None, ge=0)


_MODELS = {"chat": ChatBody, "completions": CompletionBody}
_FIELD_NAMES = {name for m in (StreamOptions, TextPart, ChatMessage, ChatTemplateKwargs, ChatBody, CompletionBody)
                for name in m.model_fields}


@dataclass(frozen=True)
class Forward:
    body: dict          # what is sent to vLLM
    stream: bool
    max_tokens: int     # the effective output bound


def _param(loc: tuple, *, keep_last: bool) -> str:
    """OpenAI-style param path ('messages[0].content') from a pydantic loc, without union-member tags."""
    parts = [x for x in (loc[:-1] if keep_last else loc) if isinstance(x, int) or x in _FIELD_NAMES]
    if keep_last:
        parts.append(loc[-1])
    out = ""
    for x in parts:
        out += f"[{x}]" if isinstance(x, int) else (f".{x}" if out else str(x))
    return out


def _translate(err: ValidationError) -> ApiError:
    errors = err.errors(include_input=False, include_url=False)
    extra = [e for e in errors if e["type"] == "extra_forbidden"]
    if extra:
        param = _param(extra[0]["loc"], keep_last=True)
        return ApiError(400, "unsupported_parameter", f"unsupported parameter {param!r} (capserve allowlist, 029)",
                        param=param)
    first = errors[0]
    param = _param(first["loc"], keep_last=False)
    return ApiError(400, "invalid_value", f"invalid value for {param!r}: {first['msg']}", param=param)


def _drop_defaults(doc: dict) -> dict:
    return {k: v for k, v in doc.items()
            if v is not None and not (k in _OFF_DEFAULTS and type(v) is bool and v is _OFF_DEFAULTS[k])}


def _check_text_only(messages) -> None:
    if not isinstance(messages, list):
        return
    for i, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for j, part in enumerate(content):
                if not (isinstance(part, dict) and part.get("type") == "text"):
                    raise ApiError(400, "unsupported_parameter", "only text content parts are supported",
                                   param=f"messages[{i}].content[{j}].type")


def _joined(content) -> str:
    return content if isinstance(content, str) else "".join(part["text"] for part in content)


def _one_prompt(prompt):
    """A string or a non-empty token list; a one-element list is unwrapped; anything else is a batch (400)."""
    if isinstance(prompt, list) and prompt and not isinstance(prompt[0], int):
        if len(prompt) != 1:
            raise ApiError(400, "unsupported_parameter", "exactly one prompt per request is supported", param="prompt")
        prompt = prompt[0]
    if isinstance(prompt, list) and not prompt:
        raise ApiError(400, "invalid_value", "the prompt is empty", param="prompt")
    return prompt


def parse_request(route: str, raw: bytes, *, served_model: str, max_output_tokens: int) -> Forward:
    """Validate a /v1/chat/completions ('chat') or /v1/completions ('completions') body; ApiError otherwise."""
    if route not in _MODELS:
        raise ValueError(f"unknown route {route!r}")
    try:
        doc = json.loads(raw)
    except (ValueError, RecursionError):   # RecursionError: nested deeper than the parser's stack
        doc = None
    if not isinstance(doc, dict):
        raise ApiError(400, "invalid_json", "the request body must be a JSON object")
    doc = _drop_defaults(doc)
    if route == "chat":
        _check_text_only(doc.get("messages"))
        if isinstance(doc.get("messages"), list):
            doc["messages"] = [_drop_defaults(m) if isinstance(m, dict) else m for m in doc["messages"]]
    try:
        req = _MODELS[route].model_validate(doc)
    except ValidationError as exc:
        raise _translate(exc) from None
    if req.model != served_model:
        raise ApiError(404, "model_not_found", "the model does not exist on this server", param="model")
    if req.n not in (None, 1):
        raise ApiError(400, "unsupported_parameter", "only n = 1 is supported", param="n")
    body = req.model_dump(exclude_unset=True)
    # the field(s) the client used for the output bound, and the one vLLM gets
    out_key = "max_completion_tokens" if route == "chat" else "max_tokens"
    given = {k: body.pop(k) for k in ("max_tokens", "max_completion_tokens") if k in body}
    if len(set(given.values())) > 1:
        raise ApiError(400, "invalid_value", "max_tokens and max_completion_tokens differ",
                       param="max_completion_tokens")
    if route == "completions":
        body["prompt"] = _one_prompt(body["prompt"])
    else:
        body["messages"] = [{**m, "content": _joined(m["content"])} for m in body["messages"]]
        thinking = (body.pop("chat_template_kwargs", None) or {}).get("enable_thinking")
        body["chat_template_kwargs"] = {"enable_thinking": THINKING_DEFAULT if thinking is None else thinking}
    key, max_tokens = next(iter(given.items())) if given else (out_key, max_output_tokens)
    if max_tokens > max_output_tokens:
        raise ApiError(400, "max_tokens_exceeded",
                       f"{key} {max_tokens} exceeds this server's output bound {max_output_tokens}", param=key)
    body[out_key] = max_tokens
    if body.get("min_tokens") is not None and body["min_tokens"] > max_tokens:
        raise ApiError(400, "invalid_value", "min_tokens exceeds max_tokens", param="min_tokens")
    try:   # a \ud800 escape decodes to a lone surrogate, which neither the byte count nor the upstream body can encode
        json.dumps(body, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        raise ApiError(400, "invalid_value", "the request contains text that is not valid Unicode") from None
    return Forward(body=body, stream=req.stream, max_tokens=max_tokens)
