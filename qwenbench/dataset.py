"""Teacher-generation dataset helpers (Stage B1) — pure functions.

The generator is an I/O loop; everything that decides *what* is generated and
*whether the result is usable* lives here, so it is testable without a server:

- the prompt shards are validated against the frozen exclusion list
  (``benchmarks/validation/manifest.json``): a contaminated prompt is refused
  before a single GPU-second is spent;
- sharding is deterministic, so four workers or four nodes partition the same
  set without coordination;
- resumption reads the ids already present in an output file and ignores a
  truncated final line, so a 20-hour job that dies at hour 19 loses one request;
- a completion record keeps the reasoning and the answer **separate**, the
  token usage, and the exact teacher/template/image versions it was produced
  with (§6.4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .eval_sets import prompt_sha256

#: Fields every prompt shard must provide.
PROMPT_FIELDS: tuple[str, ...] = ("id", "slice", "prompt")

#: Slice names of the plan (§Stage B1). S4 is optional.
SLICES: tuple[str, ...] = ("S1", "S2", "S3", "S4")


def load_prompt_files(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read and concatenate prompt shards, keeping their source in the row."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
                rows.append({**row, "source_file": str(path)})
    return rows


def check_prompts(
    rows: Sequence[Mapping[str, Any]],
    excluded: set[str],
) -> list[str]:
    """Problems that must stop the run before it starts.

    An excluded prompt is not a warning: if it reaches the teacher, the gate G2
    measurement no longer means anything.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}

    for row in rows:
        row_id = str(row.get("id", "?"))
        for field in PROMPT_FIELDS:
            if row.get(field) in (None, ""):
                problems.append(f"{row_id}: missing {field!r}")
        if row_id in seen_ids:
            problems.append(f"{row_id}: duplicate id")
        seen_ids.add(row_id)

        slice_name = row.get("slice")
        if slice_name not in SLICES:
            problems.append(f"{row_id}: unknown slice {slice_name!r}")

        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        digest = prompt_sha256(prompt)
        if digest in excluded:
            problems.append(f"{row_id}: prompt is in the frozen evaluation set (contamination)")
        if digest in seen_hashes:
            problems.append(f"{row_id}: duplicate prompt of {seen_hashes[digest]}")
        seen_hashes[digest] = row_id

    return problems


def shard_key(row: Mapping[str, Any]) -> int:
    """Stable integer for a prompt, independent of file order."""
    raw = f"{row.get('id')}\x00{row.get('prompt')}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def partition(
    rows: Sequence[Mapping[str, Any]],
    shard_index: int,
    shard_count: int,
) -> list[Mapping[str, Any]]:
    """Deterministic partition: every row lands in exactly one shard."""
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count})")
    if shard_count == 1:
        return list(rows)
    return [row for row in rows if shard_key(row) % shard_count == shard_index]


def done_ids(path: Path) -> set[str]:
    """IDs already written, tolerating a truncated last line.

    A crash mid-write leaves an incomplete JSON object; that is expected, and
    the corresponding request is simply replayed.
    """
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            if record.get("id") is not None:
                ids.add(str(record["id"]))
    return ids


def _first_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = payload.get("choices") or []
    return choices[0] if choices else {}


def build_record(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    teacher: Mapping[str, Any],
    spec: Mapping[str, Any],
    template_sha256: str | None,
    finished_at: str,
) -> dict[str, Any]:
    """Normalize one completion into the record the SFT stage consumes.

    The reasoning is kept apart from the answer: training on the concatenation
    would teach the draft to continue a thought the target had already closed.
    """
    choice = _first_choice(payload)
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = payload.get("usage") or {}

    return {
        "id": row["id"],
        "slice": row["slice"],
        "source_file": row.get("source_file"),
        "prompt": row["prompt"],
        "tools": row.get("tools"),
        "prompt_sha256": prompt_sha256(str(row["prompt"])),
        "tool_calls": message.get("tool_calls") or [],
        "reasoning": reasoning,
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        },
        "teacher": dict(teacher),
        "spec": dict(spec),
        "template_sha256": template_sha256,
        "finished_at": finished_at,
    }


def token_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Aggregate token counts, plus the empty-output count worth watching."""
    totals = {"records": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "empty": 0}
    for record in records:
        totals["records"] += 1
        usage = record.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            totals[key] += int(usage.get(key) or 0)
        if not (record.get("content") or "").strip():
            totals["empty"] += 1
    return totals


def slice_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("slice"))] = counts.get(str(row.get("slice")), 0) + 1
    return dict(sorted(counts.items()))
