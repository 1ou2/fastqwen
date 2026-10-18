"""Offline tests for the teacher-generation helpers (Stage B1).

No server, no GPU: what is tested here is the set of decisions that, if wrong,
silently waste a booked H100 slot — contaminated prompts, an unstable shard
partition, or a resume that replays or drops records.
"""

from __future__ import annotations

import json

import pytest

from qwenbench import dataset as ds


def _row(row_id: str, prompt: str, slice_name: str = "S1") -> dict:
    return {"id": row_id, "slice": slice_name, "prompt": prompt}


def _payload(content: str = "answer", reasoning: str = "thought", tokens: int = 7) -> dict:
    return {
        "choices": [{"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": tokens,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }


# --- prompt validation -----------------------------------------------------


def test_clean_prompts_have_no_problem():
    rows = [_row("a", "one"), _row("b", "two", "S2")]

    assert ds.check_prompts(rows, set()) == []


def test_excluded_prompt_is_refused():
    rows = [_row("a", "one")]

    problems = ds.check_prompts(rows, {ds.prompt_sha256("one")})

    assert problems == ["a: prompt is in the frozen evaluation set (contamination)"]


def test_missing_field_duplicate_id_and_duplicate_prompt_are_reported():
    rows = [_row("a", "same"), _row("a", "same"), {"id": "c", "slice": "S1"}]

    problems = ds.check_prompts(rows, set())

    assert "a: duplicate id" in problems
    assert "a: duplicate prompt of a" in problems
    assert "c: missing 'prompt'" in problems


def test_unknown_slice_is_reported():
    problems = ds.check_prompts([_row("a", "one", "S9")], set())

    assert problems == ["a: unknown slice 'S9'"]


# --- deterministic sharding ------------------------------------------------


def test_partitions_are_a_deterministic_covering():
    rows = [_row(str(i), f"prompt {i}") for i in range(20)]

    shards = [ds.partition(rows, index, 4) for index in range(4)]
    ids = [row["id"] for shard in shards for row in shard]

    assert sorted(ids) == sorted(row["id"] for row in rows)
    assert len(ids) == len(set(ids))
    assert [ds.partition(rows, 2, 4)] == [shards[2]]  # stable across calls


def test_single_shard_returns_everything():
    rows = [_row("a", "one")]

    assert ds.partition(rows, 0, 1) == rows


def test_invalid_shard_arguments_raise():
    with pytest.raises(ValueError):
        ds.partition([], 0, 0)
    with pytest.raises(ValueError):
        ds.partition([], 4, 4)


# --- resumption ------------------------------------------------------------


def test_done_ids_ignores_a_truncated_final_line(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps({"id": "a"}) + "\n"
        + json.dumps({"id": "b"}) + "\n"
        + '{"id": "c", "content": "trunc'  # crash mid-write
    )

    assert ds.done_ids(path) == {"a", "b"}


def test_done_ids_of_a_missing_file_is_empty(tmp_path):
    assert ds.done_ids(tmp_path / "nope.jsonl") == set()


# --- completion records ----------------------------------------------------


def test_record_keeps_reasoning_and_answer_apart():
    record = ds.build_record(
        _row("q1", "question"),
        _payload("the answer", "the thought"),
        teacher={"model": "Qwen/Qwen3.8-27B", "revision": "abc"},
        spec={"temperature": 0.0, "reasoning_effort": "low"},
        template_sha256="deadbeef",
        finished_at="2026-10-19T01:00:00+02:00",
    )

    assert record["reasoning"] == "the thought"
    assert record["content"] == "the answer"
    assert record["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "reasoning_tokens": 5}
    assert record["teacher"]["revision"] == "abc"
    assert record["template_sha256"] == "deadbeef"
    assert record["prompt_sha256"] == ds.prompt_sha256("question")


def test_record_tolerates_a_minimal_payload():
    record = ds.build_record(
        _row("q1", "question"),
        {"choices": []},
        teacher={},
        spec={},
        template_sha256=None,
        finished_at="now",
    )

    assert record["content"] == ""
    assert record["reasoning"] == ""
    assert record["usage"]["completion_tokens"] is None


def test_token_totals_count_empty_outputs():
    good = ds.build_record(_row("a", "p"), _payload("ok"), teacher={}, spec={}, template_sha256=None, finished_at="")
    empty = ds.build_record(_row("b", "p"), _payload("   ", ""), teacher={}, spec={}, template_sha256=None, finished_at="")

    totals = ds.token_totals([good, empty])

    assert totals["records"] == 2
    assert totals["completion_tokens"] == 14
    assert totals["reasoning_tokens"] == 10
    assert totals["empty"] == 1


def test_slice_counts_are_sorted():
    rows = [_row("a", "1", "S2"), _row("b", "2", "S1"), _row("c", "3", "S2")]

    assert ds.slice_counts(rows) == {"S1": 1, "S2": 2}
