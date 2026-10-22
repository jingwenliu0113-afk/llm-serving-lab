"""Tests for capserve/openai_api.py: the allowlisted request surface (exactly one prompt, max_tokens injection and
bound) and OpenAI-shaped errors with Retry-After (decision 029 "OpenAI 相容介面")."""

import json

import pytest

from capserve import openai_api as api

M = "RedHatAI/Qwen3.5-9B-quantized.w4a16"
MAX_OUT = 128
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}   # 038 §1: the gateway's default


def parse(route, body):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return api.parse_request(route, raw, served_model=M, max_output_tokens=MAX_OUT)


def chat(**kw):
    return {"model": M, "messages": [{"role": "user", "content": "hi"}], **kw}


def comp(**kw):
    return {"model": M, "prompt": "hi", **kw}


def error_of(route, body):
    with pytest.raises(api.ApiError) as exc:
        parse(route, body)
    return exc.value


# ------------------------------------------------------------------ max_tokens --

def test_chat_injects_the_class_output_bound():
    fwd = parse("chat", chat())
    assert fwd.body == {"model": M, "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": MAX_OUT,
                        **THINK_OFF}
    assert (fwd.stream, fwd.max_tokens) == (False, MAX_OUT)


def test_completions_inject_max_tokens():
    fwd = parse("completions", comp())
    assert fwd.body == {"model": M, "prompt": "hi", "max_tokens": MAX_OUT} and fwd.max_tokens == MAX_OUT


@pytest.mark.parametrize("kw, want", [({"max_tokens": 64}, 64), ({"max_completion_tokens": 64}, 64),
                                      ({"max_tokens": 64, "max_completion_tokens": 64}, 64),
                                      ({"max_tokens": MAX_OUT}, MAX_OUT)])
def test_chat_max_tokens_is_normalised_to_max_completion_tokens(kw, want):
    fwd = parse("chat", chat(**kw))
    assert fwd.body["max_completion_tokens"] == want and "max_tokens" not in fwd.body and fwd.max_tokens == want


def test_chat_conflicting_token_fields_are_400():
    e = error_of("chat", chat(max_tokens=64, max_completion_tokens=32))
    assert (e.status, e.param) == (400, "max_completion_tokens")


@pytest.mark.parametrize("route, body, param", [
    ("chat", chat(max_tokens=MAX_OUT + 1), "max_tokens"),
    ("chat", chat(max_completion_tokens=MAX_OUT + 1), "max_completion_tokens"),
    ("completions", comp(max_tokens=MAX_OUT + 1), "max_tokens"),
])
def test_above_the_class_bound_is_400(route, body, param):
    e = error_of(route, body)
    assert (e.status, e.code, e.param) == (400, "max_tokens_exceeded", param)


def test_non_positive_max_tokens_is_400():
    e = error_of("completions", comp(max_tokens=0))
    assert (e.status, e.code, e.param) == (400, "invalid_value", "max_tokens")


def test_min_tokens_must_not_exceed_max_tokens():
    assert parse("completions", comp(min_tokens=10, max_tokens=10)).body["min_tokens"] == 10
    assert parse("completions", comp(min_tokens=MAX_OUT)).body["max_tokens"] == MAX_OUT
    e = error_of("completions", comp(min_tokens=11, max_tokens=10))
    assert (e.status, e.param) == (400, "min_tokens")
    assert error_of("completions", comp(min_tokens=MAX_OUT + 1)).param == "min_tokens"


# ------------------------------------------------------------------ allowlist --

def test_allowlisted_fields_pass_through():
    body = comp(stream=True, stream_options={"include_usage": True}, temperature=0, top_p=0.9, stop=["\n"], seed=1,
                presence_penalty=0.1, frequency_penalty=0.2, user="u", n=1, ignore_eos=True, max_tokens=8)
    fwd = parse("completions", body)
    assert fwd.body == body and fwd.stream is True
    body = chat(stream=False, temperature=0.5, stop="x", messages=[
        {"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "assistant", "content": "b"}])
    assert parse("chat", body).body == {**body, "max_completion_tokens": MAX_OUT, **THINK_OFF}


@pytest.mark.parametrize("route, extra", [
    ("chat", {"tools": [{"type": "function", "function": {"name": "f"}}]}),
    ("chat", {"tool_choice": "auto"}),
    ("chat", {"response_format": {"type": "json_object"}}),
    ("chat", {"logprobs": True}),
    ("chat", {"top_logprobs": 2}),
    ("chat", {"chat_template": "{{ x }}"}),
    ("completions", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("chat", {"ignore_eos": True}),
    ("chat", {"min_tokens": 1}),
    ("completions", {"echo": True}),
    ("completions", {"logprobs": 0}),
    ("completions", {"best_of": 2}),
    ("completions", {"use_beam_search": True}),
    ("completions", {"truncate_prompt_tokens": 5}),
    ("completions", {"prompt_logprobs": 1}),
    ("completions", {"structured_outputs": {"json": {}}}),
    ("completions", {"top_k": 5}),
    ("completions", {"max_completion_tokens": 5}),
])
def test_everything_else_is_unsupported_parameter(route, extra):
    body = (chat if route == "chat" else comp)(**extra)
    e = error_of(route, body)
    assert (e.status, e.type, e.code, e.param) == (400, "invalid_request_error", "unsupported_parameter",
                                                  next(iter(extra)))


def test_nulls_and_off_defaults_are_ignored():
    fwd = parse("chat", chat(tools=None, logprobs=False, max_tokens=None, temperature=None))
    assert fwd.body == {"model": M, "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": MAX_OUT,
                        **THINK_OFF}
    fwd = parse("completions", comp(echo=False, use_beam_search=False, logprobs=None, suffix=None))
    assert fwd.body == {"model": M, "prompt": "hi", "max_tokens": MAX_OUT}


# ------------------------------------------------------------------ thinking (038 §1) --

def test_chat_thinking_is_off_unless_the_client_turns_it_on():
    assert parse("chat", chat()).body["chat_template_kwargs"] == {"enable_thinking": False}
    on = parse("chat", chat(chat_template_kwargs={"enable_thinking": True}))
    assert on.body["chat_template_kwargs"] == {"enable_thinking": True}
    off = parse("chat", chat(chat_template_kwargs={"enable_thinking": False}))
    assert off.body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("kwargs", [None, {}, {"enable_thinking": None}])
def test_chat_template_kwargs_without_a_value_keep_the_default(kwargs):
    assert parse("chat", chat(chat_template_kwargs=kwargs)).body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("kwargs, key", [({"foo": 1}, "foo"), ({"enable_thinking": True, "add_vision_id": True},
                                                                "add_vision_id")])
def test_other_chat_template_kwargs_are_unsupported(kwargs, key):
    e = error_of("chat", chat(chat_template_kwargs=kwargs))
    assert (e.status, e.code, e.param) == (400, "unsupported_parameter", f"chat_template_kwargs.{key}")


@pytest.mark.parametrize("value", ["yes", 1, 0, [True]])
def test_enable_thinking_must_be_a_boolean(value):
    e = error_of("chat", chat(chat_template_kwargs={"enable_thinking": value}))
    assert (e.status, e.code, e.param) == (400, "invalid_value", "chat_template_kwargs.enable_thinking")


def test_completions_never_get_chat_template_kwargs():
    assert "chat_template_kwargs" not in parse("completions", comp()).body


def test_n_other_than_one_is_unsupported():
    e = error_of("chat", chat(n=2))
    assert (e.code, e.param) == ("unsupported_parameter", "n")


def test_nested_unknown_fields_are_unsupported():
    e = error_of("chat", chat(messages=[{"role": "user", "content": "hi", "name": "bob"}]))
    assert (e.code, e.param) == ("unsupported_parameter", "messages[0].name")
    e = error_of("completions", comp(stream=True, stream_options={"continuous_usage_stats": True}))
    assert (e.code, e.param) == ("unsupported_parameter", "stream_options.continuous_usage_stats")


# ------------------------------------------------------------------ one prompt, text only --

@pytest.mark.parametrize("prompt, want", [("s", "s"), ([1, 2, 3], [1, 2, 3]), (["s"], "s"), ([[1, 2]], [1, 2])])
def test_exactly_one_prompt_is_normalised(prompt, want):
    assert parse("completions", comp(prompt=prompt)).body["prompt"] == want


@pytest.mark.parametrize("prompt", [["a", "b"], [[1], [2]], [], [[]], [["a"]], 5, None, [True]])
def test_batches_and_non_prompts_are_400(prompt):
    e = error_of("completions", comp(prompt=prompt))
    assert e.status == 400 and e.param == "prompt"


def test_non_text_parts_and_roles_are_400():
    image = [{"type": "image_url", "image_url": {"url": "data:,"}}]
    e = error_of("chat", chat(messages=[{"role": "user", "content": image}]))
    assert (e.status, e.code, e.param) == (400, "unsupported_parameter", "messages[0].content[0].type")
    e = error_of("chat", chat(messages=[{"role": "tool", "content": "x"}]))
    assert (e.status, e.param) == (400, "messages[0].role")
    assert error_of("chat", chat(messages=[])).param == "messages"
    assert error_of("chat", chat(messages=[{"role": "user", "content": None}])).param == "messages[0].content"


# ------------------------------------------------------------------ model, json, types --

def test_wrong_model_is_404():
    e = error_of("chat", chat(model="other"))
    assert (e.status, e.type, e.code, e.param) == (404, "invalid_request_error", "model_not_found", "model")


@pytest.mark.parametrize("raw", [b"{not json", b"[1]", b"", "é".encode("latin-1")])
def test_bodies_that_are_not_a_json_object_are_400(raw):
    e = error_of("chat", raw)
    assert (e.status, e.code) == (400, "invalid_json")


@pytest.mark.parametrize("kw, param", [({"temperature": "hot"}, "temperature"), ({"stream": "yes"}, "stream"),
                                       ({"max_tokens": True}, "max_tokens"), ({"stop": [1]}, "stop"),
                                       ({"model": None}, "model")])
def test_wrong_types_are_invalid_value(kw, param):
    e = error_of("completions", comp(**kw))
    assert (e.status, e.code, e.param) == (400, "invalid_value", param)


@pytest.mark.parametrize("field, literal", [("temperature", "NaN"), ("top_p", "Infinity"),
                                            ("presence_penalty", "-Infinity"), ("frequency_penalty", "1e999")])
def test_non_finite_numbers_are_invalid_value(field, literal):
    # json.loads reads them (1e999 as inf), but the body sent upstream must be valid JSON (httpx: allow_nan=False)
    e = error_of("completions", ('{"model": "%s", "prompt": "hi", "%s": %s}' % (M, field, literal)).encode())
    assert (e.status, e.code, e.param) == (400, "invalid_value", field)


@pytest.mark.parametrize("route, body", [
    ("completions", '{"model": "%s", "prompt": "a\\ud800b"}' % M),
    ("completions", '{"model": "%s", "prompt": "hi", "stop": ["\\udfff"]}' % M),
    ("chat", '{"model": "%s", "messages": [{"role": "user", "content": [{"type": "text", "text": "\\ud800"}]}]}' % M),
])
def test_lone_surrogates_are_invalid_value(route, body):
    # a \\ud800 escape decodes to a str that cannot be encoded as UTF-8 (the byte count, the upstream body)
    e = error_of(route, body.encode())
    assert (e.status, e.code) == (400, "invalid_value")


def test_json_nested_too_deep_for_the_parser_is_invalid_json():
    e = error_of("completions", b"[" * 100_000 + b"]" * 100_000)
    assert (e.status, e.code) == (400, "invalid_json")


def test_unknown_route_is_a_programming_error():
    with pytest.raises(ValueError):
        parse("embeddings", comp())


# ------------------------------------------------------------------ errors --

def test_error_shape_without_retry():
    r = api.openai_error(400, "invalid_request_error", "unsupported_parameter", "no", param="tools")
    assert r.status_code == 400 and r.media_type == "application/json"
    assert json.loads(r.body) == {"error": {"message": "no", "type": "invalid_request_error", "param": "tools",
                                            "code": "unsupported_parameter"}}
    assert "retry-after" not in r.headers and "x-should-retry" not in r.headers


def test_rejections_carry_retry_after_headers():
    r = api.openai_error(429, "rate_limit_error", "queue_full", "busy", retry_after_s=2)
    assert (r.headers["retry-after"], r.headers["retry-after-ms"], r.headers["x-should-retry"]) == ("2", "2000", "true")
    assert json.loads(r.body)["error"]["param"] is None


def test_api_error_to_response():
    e = api.ApiError(404, "model_not_found", "unknown model", param="model")
    r = e.response()
    assert r.status_code == 404 and json.loads(r.body)["error"]["code"] == "model_not_found"
