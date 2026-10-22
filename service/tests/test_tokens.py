"""Tests for capserve/tokens.py: the byte fast path and the /tokenize count that mirrors the generation request
(decision 029 steps 6 and 8)."""

import json

import httpx
import pytest

from capserve.tokens import LocalTokenizer, PromptBudget, TemplateRejected, TokenizeError, TokenizerLoadError

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
    kwargs = {"enable_thinking": False}     # 038 §1: /tokenize renders the same template as the generation request
    assert await b.count("chat", {"model": M, "messages": messages, "max_completion_tokens": 5, "stream": True,
                                  "chat_template_kwargs": kwargs}) == 7
    assert seen == [{"model": M, "messages": messages, "add_generation_prompt": True, "chat_template_kwargs": kwargs}]


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


# ------------------------------------------------------------------ local tokenizer (038 §4) --

class FakeHF:
    """An HF tokenizer stand-in: the template is '<' + contents joined by '|' + a thinking marker; one token per
    character, plus one special token when asked."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append(("template", messages, kw))
        if self.fail:
            raise ValueError("System message must be at the beginning.")
        return "<" + "|".join(m["content"] for m in messages) + ("T" if kw.get("enable_thinking", True) else "N")

    def encode(self, text, *, add_special_tokens):
        self.calls.append(("encode", text, add_special_tokens))
        return [0] * (len(text) + int(add_special_tokens))


def test_local_chat_count_renders_the_template_like_vllm():
    hf = FakeHF()
    body = {"model": M, "messages": [{"role": "user", "content": "abc"}], "chat_template_kwargs":
            {"enable_thinking": False}, "max_completion_tokens": 5}
    assert LocalTokenizer(hf).count("chat", body) == len("<abcN")
    assert hf.calls == [("template", body["messages"], {"tokenize": False, "add_generation_prompt": True,
                                                        "enable_thinking": False}),
                        ("encode", "<abcN", False)]   # vLLM's chat requests: add_special_tokens false


def test_local_completion_count_adds_special_tokens_like_vllm():
    hf = FakeHF()
    assert LocalTokenizer(hf).count("completions", {"model": M, "prompt": "abcd"}) == 5
    assert hf.calls == [("encode", "abcd", True)]
    assert LocalTokenizer(hf).count("completions", {"model": M, "prompt": [7, 8, 9]}) == 3


def test_a_template_that_raises_is_a_rejected_template():
    with pytest.raises(TemplateRejected, match="System message"):
        LocalTokenizer(FakeHF(fail=True)).count("chat", {"model": M, "messages": [{"role": "user", "content": "x"}]})


def test_loading_needs_transformers_and_the_snapshot(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_transformers(name, *args, **kw):
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_transformers)
    with pytest.raises(TokenizerLoadError, match="transformers"):
        LocalTokenizer.load(str(tmp_path))
    monkeypatch.setattr(builtins, "__import__", real_import)
    with pytest.raises(TokenizerLoadError, match="no tokenizer"):
        LocalTokenizer.load(str(tmp_path / "missing"))


@pytest.mark.anyio
async def test_a_local_budget_never_calls_vllm():
    b, seen = budget(local=LocalTokenizer(FakeHF()))
    assert b.local is not None
    assert await b.count("chat", {"model": M, "messages": [{"role": "user", "content": "ab"}]}) == len("<abT")
    assert await b.count("completions", {"model": M, "prompt": "xyz"}) == 4
    assert seen == []
