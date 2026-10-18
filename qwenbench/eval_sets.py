"""Evaluation datasets (§6.8, §6.5) — pure helpers.

The qualitative 30 and the acceptance validation set are *frozen* artifacts: they
must not change between the baseline and the optimized models, and no prompt of
them may ever appear in the teacher's training data (§7, "Calibration contaminated
by eval prompts"). This module holds the pure logic — normalization, hashing,
duplicate and contamination detection — so the builder script stays a thin CLI
and the rules themselves are testable offline.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical form of a prompt, used for deduplication and exclusion.

    Collapsing whitespace and case matters: the same question re-imported with a
    different trailing space must be recognized as the same prompt, otherwise a
    contaminated evaluation set goes unnoticed.
    """
    return _WHITESPACE.sub(" ", text).strip().casefold()


def prompt_sha256(text: str) -> str:
    """Hash of the normalized prompt: the exclusion key (§6.4, §7)."""
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, reporting the offending line on a syntax error."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def missing_fields(rows: Sequence[Mapping[str, Any]], required: Sequence[str], name: str) -> list[str]:
    """Rows lacking a required field, as ``name:id:field`` problems."""
    problems: list[str] = []
    for row in rows:
        row_id = row.get("id", "?")
        for field in required:
            if row.get(field) in (None, ""):
                problems.append(f"{name}:{row_id}: missing {field!r}")
    return problems


def duplicate_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """IDs used more than once."""
    seen: dict[str, int] = {}
    for row in rows:
        seen[str(row.get("id"))] = seen.get(str(row.get("id")), 0) + 1
    return sorted(row_id for row_id, count in seen.items() if count > 1)


def duplicates_within(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Normalized prompt hashes appearing more than once in the same dataset."""
    seen: dict[str, list[str]] = {}
    for row in rows:
        seen.setdefault(prompt_sha256(str(row["prompt"])), []).append(str(row["id"]))
    return {key: ids for key, ids in seen.items() if len(ids) > 1}


def duplicates_across(datasets: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Prompts shared by two datasets — the contamination the plan forbids.

    ``datasets`` maps a display name to its rows, e.g. ``{"validation": [...]}``.
    """
    index: dict[str, list[str]] = {}
    for name, rows in datasets.items():
        for row in rows:
            index.setdefault(prompt_sha256(str(row["prompt"])), []).append(f"{name}:{row['id']}")
    return [
        {"sha256": key, "refs": refs}
        for key, refs in sorted(index.items())
        if len(refs) > 1
    ]


def count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    """Distribution of a field, ``"absent"`` included so nothing is silent."""
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) if row.get(key) is not None else "absent")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def inline_tools(row: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve ``tools_ref`` against the shared catalog.

    A dangling reference is an error, not an empty tool list: a tool-calling item
    silently rendered without tools measures nothing.
    """
    ref = row.get("tools_ref")
    resolved = {**row, "tools": None}
    if ref is None:
        return resolved
    if ref not in catalog:
        raise KeyError(f"unknown tools_ref {ref!r} in item {row.get('id')!r}")
    resolved["tools"] = catalog[ref]
    return resolved


def exclusion_index(datasets: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, str]]:
    """Flattened ``id / dataset / sha256`` list, written into the run manifest."""
    return [
        {"dataset": name, "id": str(row["id"]), "sha256": prompt_sha256(str(row["prompt"]))}
        for name, rows in datasets.items()
        for row in rows
    ]
