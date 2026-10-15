"""Tests for capserve/tokens.py: the byte fast path and the /tokenize count that mirrors the generation request
(decision 029 steps 6 and 8)."""

import json

import httpx
import pytest

from capserve.tokens import PromptBudget, TokenizeError

M = "m"


def budget(handler=None, **kw):
    seen = []

    def default(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"count": 7, "max_model_len": 32768, "tokens": [1] * 7})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler or default), base_url="http://vllm")
    args = {"chat_overhead": 10, "per_message": 3, "completion_overhead": 1, **kw}
    return PromptBudget(client, **args), seen


def test_completion_bound_is_one_token_per_utf8_byte_plus_overhead():
    b, _ = budget()
    body = {"model": M, "prompt": "é" * 4}             # 8 bytes + 1
    assert b.upper_bound("completions", body) == 9
    assert b.fast_fits("completions", body, 9) and not b.fast_fits("completions", body, 8)


def test_token_id_prompts_are_counted_exactly():
    b, _ = budget()
    body = {"model": M, "prompt": [5] * 20}
    assert b.upper_bound("completions", body) == 20
    assert b.fast_fits("completions", body, 20) and not b.fast_fits("completions", body, 19)


def test_chat_bound_counts_roles_texts_and_overheads():
    b, _ = budget()
    body = {"model": M, "messages": [{"role": "system", "content": "ab"},
                                     {"role": "user", "content": [{"type": "text", "text": "c"},
                                                                  {"type": "text", "text": "dé"}]}]}
    # 10 + (3 + 6 "system" + 2) + (3 + 4 "user" + 1 + 3)
    assert b.upper_bound("chat", body) == 10 + 11 + 11
    assert b.fast_fits("chat", body, 32) and not b.fast_fits("chat", body, 31)


@pytest.mark.anyio
async def test_count_mirrors_the_chat_messages_with_the_generation_prompt():
    b, seen = budget()
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert await b.count("chat", {"model": M, "messages": messages, "max_completion_tokens": 5, "stream": True}) == 7
    assert seen == [{"model": M, "messages": messages, "add_generation_prompt": True}]


@pytest.mark.anyio
async def test_count_of_a_text_prompt_adds_special_tokens_like_completions():
    b, seen = budget()
    assert await b.count("completions", {"model": M, "prompt": "a b", "max_tokens": 5}) == 7
    assert seen == [{"model": M, "prompt": "a b", "add_special_tokens": True}]


@pytest.mark.anyio
async def test_count_of_token_ids_needs_no_request():
    b, seen = budget()
    assert await b.count("completions", {"model": M, "prompt": [1, 2, 3]}) == 3
    assert seen == []


@pytest.mark.anyio
async def test_a_failed_tokenize_is_a_tokenize_error():
    b, _ = budget(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(TokenizeError) as exc:
        await b.count("completions", {"model": M, "prompt": "x"})
    assert exc.value.status == 500
    b, _ = budget(lambda request: httpx.Response(200, json={"tokens": []}))
    with pytest.raises(TokenizeError):
        await b.count("completions", {"model": M, "prompt": "x"})


@pytest.mark.anyio
async def test_a_tokenize_error_keeps_the_upstream_response():
    shaped = {"error": {"message": "System message must be at the beginning.", "type": "BadRequestError",
                        "param": None, "code": 400}}
    b, _ = budget(lambda request: httpx.Response(400, json=shaped))
    with pytest.raises(TokenizeError) as exc:
        await b.count("chat", {"model": M, "messages": [{"role": "user", "content": "x"}]})
    assert exc.value.status == 400 and exc.value.response.json() == shaped
