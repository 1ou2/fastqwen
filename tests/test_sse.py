"""Tests of `qwenbench.sse` — pure parsing, without server."""

import pytest

from qwenbench.sse import (
    assemble_tool_calls,
    chunk_finish_reason,
    extract_usage,
    iter_deltas,
    parse_sse_data,
    sse_payload,
)


def test_parse_sse_data_nominal():
    assert parse_sse_data('data: {"a": 1}') == {"a": 1}
    assert parse_sse_data("data:{\"a\":1}") == {"a": 1}


def test_parse_sse_data_ignores_noise():
    assert parse_sse_data("") is None
    assert parse_sse_data("\n") is None
    assert parse_sse_data("data: [DONE]") is None
    assert parse_sse_data("event: ping") is None
    assert parse_sse_data(": keep-alive") is None
    assert parse_sse_data("data:") is None


def test_parse_sse_data_rejects_invalid_json_and_non_dict():
    assert parse_sse_data("data: {oops") is None
    assert parse_sse_data("data: [1, 2]") is None
    assert parse_sse_data('data: "chaine"') is None


def test_sse_payload_distinguishes_malformed_from_noise():
    """The client must be able to count unreadable lines: hence the distinction."""
    assert sse_payload("data: {oops") == "{oops"
    assert sse_payload("event: ping") is None
    assert sse_payload("data: [DONE]") is None


def test_iter_deltas_reasoning_then_answer():
    assert iter_deltas({"choices": [{"delta": {"reasoning_content": "Je"}}]}) == [
        ("reasoning", "Je")
    ]
    assert iter_deltas({"choices": [{"delta": {"content": "Paris"}}]}) == [
        ("answer", "Paris")
    ]
    assert iter_deltas({"choices": [{"delta": {"role": "assistant"}}]}) == []
    assert iter_deltas({"choices": []}) == []


def test_iter_deltas_both_in_same_chunk():
    chunk = {"choices": [{"delta": {"reasoning_content": "fin", "content": "Aa"}}]}
    assert iter_deltas(chunk) == [("reasoning", "fin"), ("answer", "Aa")]


def test_iter_deltas_tool_calls():
    chunk = {
        "choices": [
            {"delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]}}
        ]
    }
    assert iter_deltas(chunk) == [("tool", "")]


def test_extract_usage():
    assert extract_usage({"choices": []}) is None
    usage = extract_usage(
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4,
                                  "total_tokens": 7}}
    )
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 4, 7)
    assert usage.reasoning_tokens is None


def test_extract_usage_reads_the_reasoning_breakdown():
    """Real vLLM shape: completion_tokens_details.reasoning_tokens."""
    usage = extract_usage(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 59,
                "completion_tokens": 41,
                "total_tokens": 100,
                "prompt_tokens_details": None,
                "completion_tokens_details": {"reasoning_tokens": 30},
            },
        }
    )
    assert usage is not None
    assert usage.reasoning_tokens == 30
    assert usage.total_tokens == 100


def test_extract_usage_tolerates_null_details():
    usage = extract_usage(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "completion_tokens_details": None,
            },
        }
    )
    assert usage is not None
    assert usage.reasoning_tokens is None


def test_chunk_finish_reason():
    assert chunk_finish_reason({"choices": [{"finish_reason": "stop"}]}) == "stop"
    assert chunk_finish_reason({"choices": [{"finish_reason": None}]}) is None
    assert chunk_finish_reason({"choices": []}) is None


def test_assemble_tool_calls_fragments():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "get_weather",
                                                      "arguments": '{"loc'}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'ation": "Lyon"}'}}
        ]}}]},
    ]
    calls = assemble_tool_calls(chunks)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["id"] == "call_1"
    assert calls[0]["arguments"] == '{"location": "Lyon"}'


def test_assemble_tool_calls_empty():
    assert assemble_tool_calls([{"choices": [{"delta": {"content": "x"}}]}]) == []


def test_multiple_choices_are_rejected():
    """`n > 1` would silently merge two completions: refuse instead."""
    chunk = {
        "choices": [
            {"index": 0, "delta": {"content": "A"}},
            {"index": 1, "delta": {"content": "B"}},
        ]
    }
    with pytest.raises(ValueError, match="n > 1"):
        iter_deltas(chunk)
