"""Offline tests for the evaluation-set helpers (§6.8, §7 contamination rule).

Pure logic only: no network, no datasets on disk. The point is that the
deduplication and exclusion rules are not trusted to the builder script.
"""

from __future__ import annotations

import pytest

from qwenbench import eval_sets as es


def _row(row_id: str, prompt: str, **extra) -> dict:
    return {"id": row_id, "prompt": prompt, **extra}


# --- normalization and hashing --------------------------------------------


def test_normalization_ignores_case_and_whitespace():
    assert es.normalize("  Combien\tfont  17 × 23 ?\n") == "combien font 17 × 23 ?"


def test_prompt_hash_ignores_cosmetic_differences():
    a = es.prompt_sha256("Write a test for clamp().")
    b = es.prompt_sha256("  write a TEST   for clamp(). \n")

    assert a == b
    assert es.prompt_sha256("Write a test for clamp!") != a


# --- duplicates ------------------------------------------------------------


def test_duplicates_within_a_dataset_are_reported():
    rows = [_row("a", "Same prompt"), _row("b", "same  PROMPT "), _row("c", "other")]

    duplicates = es.duplicates_within(rows)

    assert duplicates == {es.prompt_sha256("Same prompt"): ["a", "b"]}


def test_duplicates_across_datasets_are_contamination():
    datasets = {
        "validation": [_row("v1", "Quelle heure est-il ?")],
        "calibration": [_row("c1", "quelle heure est-il ? ")],
    }

    duplicates = es.duplicates_across(datasets)

    assert len(duplicates) == 1
    assert duplicates[0]["refs"] == ["validation:v1", "calibration:c1"]


def test_disjoint_datasets_have_no_contamination():
    datasets = {"a": [_row("a1", "one")], "b": [_row("b1", "two")]}

    assert es.duplicates_across(datasets) == []


def test_duplicate_ids_are_reported():
    rows = [_row("dup", "one"), _row("dup", "two"), _row("unique", "three")]

    assert es.duplicate_ids(rows) == ["dup"]


# --- schema checks ---------------------------------------------------------


def test_missing_fields_are_reported_per_row():
    rows = [_row("a", "ok", slice="S1"), {"id": "b", "prompt": "", "slice": "S2"}]

    problems = es.missing_fields(rows, ("id", "prompt", "slice"), "validation")

    assert problems == [
        "validation:b: missing 'prompt'",
    ]


def test_count_by_includes_absent_values():
    rows = [_row("a", "x", slice="S1"), _row("b", "y"), _row("c", "z", slice="S1")]

    assert es.count_by(rows, "slice") == {"S1": 2, "absent": 1}


# --- tool resolution -------------------------------------------------------


CATALOG = {"weather": [{"type": "function", "function": {"name": "get_weather"}}]}


def test_tools_are_resolved_by_reference():
    resolved = es.inline_tools({"id": "t1", "tools_ref": "weather"}, CATALOG)

    assert resolved["tools"] == CATALOG["weather"]


def test_no_tool_reference_means_no_tools():
    resolved = es.inline_tools({"id": "t1", "tools_ref": None}, CATALOG)

    assert resolved["tools"] is None


def test_dangling_tool_reference_raises():
    with pytest.raises(KeyError, match="weather_v2"):
        es.inline_tools({"id": "t1", "tools_ref": "weather_v2"}, CATALOG)


# --- exclusion index -------------------------------------------------------


def test_exclusion_index_flattens_every_dataset():
    datasets = {
        "validation": [_row("v1", "one")],
        "qualitative": [_row("q1", "two")],
    }

    index = es.exclusion_index(datasets)

    assert [entry["dataset"] for entry in index] == ["validation", "qualitative"]
    assert index[0]["sha256"] == es.prompt_sha256("one")
    assert all(len(entry["sha256"]) == 64 for entry in index)
