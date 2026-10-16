"""Tests of `qwenbench.metrics` — pure, without network or server."""

import json

import pytest

from qwenbench.metrics import (
    StreamEvent,
    Usage,
    percentile,
    summarize_stream,
)


def ev(index: int, t: float, kind: str, text: str = "") -> StreamEvent:
    return StreamEvent(index=index, t_rel_s=t, kind=kind, text=text)


def test_nominal_stream():
    events = [
        ev(0, 0.50, "reasoning", "The"),
        ev(1, 0.75, "reasoning", " reasoning"),
        ev(2, 1.00, "answer", "Canberra"),
        ev(3, 1.25, "answer", "."),
    ]
    m = summarize_stream(
        events,
        usage=Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        t_end_s=1.30,
        finish_reason="stop",
    )
    assert m.ttft_s == pytest.approx(0.50)
    assert m.e2e_s == pytest.approx(1.30)
    assert m.decode_s == pytest.approx(0.75)
    assert m.decode_tok_s == pytest.approx(4.0)  # (4 - 1) / 0.75
    assert m.inter_chunk_p50_s == pytest.approx(0.25)
    assert m.inter_chunk_p95_s == pytest.approx(0.25)
    assert m.n_total_tok == 4
    assert m.n_prompt_tok == 10
    assert m.n_reasoning_events == 2
    assert m.n_answer_events == 2
    assert m.n_tool_events == 0
    assert m.reasoning_share == pytest.approx(0.5)
    assert m.reasoning_share_basis == "events"  # no breakdown in this `usage`
    assert m.n_reasoning_tok is None
    assert m.n_non_reasoning_tok is None
    assert m.usage_authoritative is True
    assert m.warnings == ()


def test_exact_reasoning_token_split_from_usage():
    """The server's breakdown gives the exact split: no chunk arithmetic."""
    events = [
        ev(0, 0.5, "reasoning", "We"),
        ev(1, 0.9, "reasoning", " need"),
        ev(2, 1.3, "answer", "Canberra"),
    ]
    m = summarize_stream(
        events,
        usage=Usage(
            prompt_tokens=59,
            completion_tokens=6,
            total_tokens=65,
            reasoning_tokens=4,
        ),
        t_end_s=1.5,
        finish_reason="stop",
    )
    assert (m.n_reasoning_tok, m.n_non_reasoning_tok) == (4, 2)
    assert m.reasoning_share_basis == "tokens"
    assert m.reasoning_share == pytest.approx(4 / 6)


def test_without_breakdown_the_share_falls_back_to_events():
    events = [
        ev(0, 0.5, "reasoning", "The"),
        ev(1, 0.9, "answer", "Canberra"),
    ]
    m = summarize_stream(
        events,
        usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        t_end_s=1.0,
    )
    assert m.n_reasoning_tok is None
    assert m.n_non_reasoning_tok is None
    assert m.reasoning_share_basis == "events"
    assert m.reasoning_share == pytest.approx(0.5)


def test_inconsistent_reasoning_tokens_are_flagged():
    m = summarize_stream(
        [ev(0, 0.5, "answer", "x")],
        usage=Usage(
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            reasoning_tokens=9,
        ),
        t_end_s=0.6,
    )
    assert m.n_reasoning_tok is None
    assert m.n_non_reasoning_tok is None
    assert any(
        "reasoning_tokens (9) > completion_tokens (2)" in w for w in m.warnings
    )


def test_opening_chunk_without_content_is_ignored():
    """A "role-only" chunk must not be counted as the first token."""
    events = [
        ev(0, 0.30, "answer", ""),  # assistant role, empty content
        ev(1, 0.90, "answer", "Canberra"),
        ev(2, 1.10, "answer", "."),
    ]
    m = summarize_stream(events, t_end_s=1.2)
    assert m.ttft_s == pytest.approx(0.90)


def test_empty_stream_raises():
    with pytest.raises(ValueError, match="empty stream"):
        summarize_stream([])


def test_stream_without_output_raises():
    with pytest.raises(ValueError, match="without output"):
        summarize_stream([ev(0, 0.1, "answer", "")])


def test_non_monotonic_timestamp_raises():
    events = [ev(0, 1.0, "answer", "a"), ev(1, 0.5, "answer", "b")]
    with pytest.raises(ValueError, match="non-monotonic"):
        summarize_stream(events)


def test_end_before_first_token_raises():
    with pytest.raises(ValueError, match="before the first token"):
        summarize_stream([ev(0, 1.0, "answer", "a")], t_end_s=0.5)


def test_without_usage_falls_back_to_chunks():
    events = [ev(i, 0.1 * i, "answer", "x") for i in range(4)]
    m = summarize_stream(events, t_end_s=0.5, finish_reason="stop")
    assert m.usage_authoritative is False
    assert m.n_total_tok == 4
    assert m.n_prompt_tok is None
    assert any("usage absent" in w for w in m.warnings)


def test_single_token_zero_rate():
    m = summarize_stream(
        [ev(0, 0.4, "answer", "Canberra")],
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        t_end_s=0.5,
    )
    assert m.decode_s == 0.0
    assert m.decode_tok_s == 0.0
    assert any("fewer than 2 tokens" in w for w in m.warnings)


def test_tool_only_stream():
    m = summarize_stream(
        [ev(0, 0.4, "tool", "")],
        usage=Usage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        t_end_s=0.9,
        finish_reason="tool_calls",
    )
    assert m.ttft_s == pytest.approx(0.4)
    assert m.n_tool_events == 1
    assert m.decode_tok_s == 0.0
    assert any("no answer token" in w for w in m.warnings)


def test_inconsistent_more_chunks_than_tokens():
    events = [ev(i, 0.1 * i, "answer", "x") for i in range(4)]
    m = summarize_stream(
        events,
        usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        t_end_s=0.5,
    )
    assert any("inconsistency" in w for w in m.warnings)


def test_truncation_flagged():
    events = [ev(0, 0.2, "reasoning", "a"), ev(1, 0.4, "reasoning", "b")]
    m = summarize_stream(
        events,
        usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        t_end_s=0.5,
        finish_reason="length",
    )
    assert any("truncated" in w for w in m.warnings)


def test_to_dict_is_serializable_and_carries_the_new_fields():
    m = summarize_stream(
        [ev(0, 0.2, "reasoning", "a"), ev(1, 0.4, "answer", "b")],
        usage=Usage(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            reasoning_tokens=1,
        ),
        t_end_s=0.5,
        finish_reason="stop",
    )
    payload = m.to_dict()
    assert json.loads(json.dumps(payload))["n_total_tok"] == 2
    assert isinstance(payload["warnings"], list)
    assert payload["n_reasoning_tok"] == 1
    assert payload["n_non_reasoning_tok"] == 1
    assert payload["reasoning_share_basis"] == "tokens"
    assert "inter_chunk_p50_s" in payload
    assert "itl_p50_s" not in payload


def test_percentile():
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.9) == 5.0
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0.0) == 1.0
    assert percentile([1, 2, 3, 4], 1.0) == 4.0
