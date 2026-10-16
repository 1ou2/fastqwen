"""Tests of `qwenbench.client` — network simulated by httpx.MockTransport.

No server or GPU is needed: we replay frozen SSE streams.
"""

import httpx
import pytest

from qwenbench.client import ChatClient, SamplingSpec, build_request_body

SSE_NOMINAL = (
    'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
    'data: {"choices":[{"delta":{"reasoning_content":"I "}}]}\n\n'
    'data: {"choices":[{"delta":{"reasoning_content":"am thinking."}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Paris"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"."}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5,'
    '"total_tokens":17}}\n\n'
    "data: [DONE]\n\n"
)

SSE_TOOL = (
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
    '"function":{"name":"get_weather","arguments":"{\\"loc"}}]}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"ation\\": \\"Lyon\\"}"}}]}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    "data: [DONE]\n\n"
)

SSE_MALFORMED = (
    'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    "data: {oops\n\n"
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)

SSE_WITH_REASONING_TOKENS = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"reasoning":"We"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"reasoning":" need"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"Canberra"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":59,"completion_tokens":6,'
    '"total_tokens":65,"completion_tokens_details":{"reasoning_tokens":4}}}\n\n'
    "data: [DONE]\n\n"
)

SSE_TWO_CHOICES = (
    'data: {"choices":[{"index":0,"delta":{"content":"A"}},'
    '{"index":1,"delta":{"content":"B"}}]}\n\n'
    "data: [DONE]\n\n"
)


def make_client(handler) -> ChatClient:
    return ChatClient(
        base_url="http://testserver", transport=httpx.MockTransport(handler)
    )


def sse_handler(payload: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=payload.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    return handler


# -- request body (pure) -----------------------------------------------------

def test_body_thinking_disabled():
    body = build_request_body("m", "p", SamplingSpec(thinking=False, max_tokens=64))
    assert body["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": True,
    }
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.0
    assert "reasoning_effort" not in body
    assert "top_k" not in body
    assert body["seed"] == 0


def test_body_effort_top_level_or_template():
    top = build_request_body("m", "p", SamplingSpec(reasoning_effort="xhigh"))
    assert top["reasoning_effort"] == "xhigh"
    assert "reasoning_effort" not in top["chat_template_kwargs"]

    tpl = build_request_body(
        "m", "p", SamplingSpec(reasoning_effort="xhigh", effort_in_template=True)
    )
    assert tpl["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    assert "reasoning_effort" not in tpl


def test_body_tools_and_penalty():
    tools = ({"type": "function", "function": {"name": "get_weather"}},)
    body = build_request_body(
        "m", "p", SamplingSpec(tools=tools, presence_penalty=1.5, top_k=20)
    )
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["presence_penalty"] == 1.5
    assert body["top_k"] == 20


def test_body_without_stream():
    body = build_request_body("m", "p", SamplingSpec(), stream=False)
    assert "stream" not in body
    assert "stream_options" not in body


# -- measurement of a stream -------------------------------------------------

def test_stream_once_nominal():
    client = make_client(sse_handler(SSE_NOMINAL))
    sample = client.stream_once("p", SamplingSpec(max_tokens=16), config="t")

    assert sample.http_status == 200
    assert sample.config == "t"
    assert sample.reasoning_text == "I am thinking."
    assert sample.answer_text == "Paris."
    assert sample.tool_calls == []
    m = sample.metrics
    assert m.n_total_tok == 5
    assert m.n_prompt_tok == 12
    assert m.usage_authoritative is True
    assert m.n_reasoning_events == 2
    assert m.n_answer_events == 2
    assert m.ttft_s >= 0.0
    assert m.finish_reason == "stop"
    assert m.warnings == ()
    assert sample.request_body["model"] == "Qwen/Qwen3.8-27B"


def test_stream_once_flags_unreadable_lines():
    client = make_client(sse_handler(SSE_MALFORMED))
    sample = client.stream_once("p", SamplingSpec())
    assert sample.answer_text == "ok"
    assert any("unreadable" in w for w in sample.metrics.warnings)


def test_stream_once_tool():
    client = make_client(sse_handler(SSE_TOOL))
    sample = client.stream_once("p", SamplingSpec(thinking=False))
    assert len(sample.tool_calls) == 1
    assert sample.tool_calls[0]["name"] == "get_weather"
    assert sample.tool_calls[0]["arguments"] == '{"location": "Lyon"}'
    # 2 chunks carry tool-call fragments -> 2 "tool" events (the n_*_events
    # fields count deltas, not tokens), but only 1 assembled call.
    assert sample.metrics.n_tool_events == 2
    assert sample.metrics.finish_reason == "tool_calls"


def test_stream_once_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "max_tokens invalid"}})

    client = make_client(handler)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        client.stream_once("p", SamplingSpec())


def test_to_dict_serializable():
    client = make_client(sse_handler(SSE_NOMINAL))
    payload = client.stream_once("p", SamplingSpec(), config="t").to_dict()
    assert payload["metrics"]["n_total_tok"] == 5
    assert payload["config"] == "t"


# -- startup wait and fingerprint --------------------------------------------

def test_wait_until_ready_then_models():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": "loading"})
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.8-27B"}]})

    client = make_client(handler)
    waited = client.wait_until_ready(timeout_s=5.0, poll_s=0.01)
    assert waited >= 0.0
    assert calls["n"] == 3
    assert client.models() == ["Qwen/Qwen3.8-27B"]


def test_wait_until_ready_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "chargement"})

    client = make_client(handler)
    with pytest.raises(TimeoutError, match="not ready"):
        client.wait_until_ready(timeout_s=0.05, poll_s=0.01)


def test_fingerprint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.8-27B"}]})

    fingerprint = make_client(handler).fingerprint()
    assert fingerprint["model_ids"] == ["Qwen/Qwen3.8-27B"]
    assert len(fingerprint["v1_models_sha256"]) == 64
    assert fingerprint["base_url"] == "http://testserver"


# -- exact reasoning/answer token split --------------------------------------

def test_stream_once_reads_the_exact_reasoning_token_split():
    """Fixture shaped like the real capture: reasoning + content + breakdown."""
    client = make_client(sse_handler(SSE_WITH_REASONING_TOKENS))
    sample = client.stream_once("p", SamplingSpec(thinking=True), config="t")
    m = sample.metrics
    assert sample.reasoning_text == "We need"
    assert sample.answer_text == "Canberra"
    assert m.n_total_tok == 6
    assert (m.n_reasoning_tok, m.n_non_reasoning_tok) == (4, 2)
    assert m.reasoning_share_basis == "tokens"
    assert m.reasoning_share == pytest.approx(4 / 6)


def test_stream_once_rejects_multiple_choices():
    client = make_client(sse_handler(SSE_TWO_CHOICES))
    with pytest.raises(ValueError, match="n > 1"):
        client.stream_once("p", SamplingSpec())
