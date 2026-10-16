"""Parsing of the SSE chunks of an OpenAI-compatible server — pure functions.

No state, no network access: we transform lines and dicts. This is what makes
the semantics of the stream (reasoning / answer / tool calls)
testable without a server.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .metrics import Kind, Usage

__all__ = [
    "assemble_tool_calls",
    "chunk_finish_reason",
    "extract_usage",
    "iter_deltas",
    "parse_sse_data",
    "single_choice",
    "sse_payload",
]


def sse_payload(line: str) -> str | None:
    """Content of a `data: ...` line, or None if the line brings nothing.

    None covers: empty line, comment/keep-alive, other SSE fields
    (`event:`, `id:`, `:`) and the `[DONE]` sentinel.
    """
    if not line:
        return None
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    return payload


def parse_sse_data(line: str) -> dict[str, Any] | None:
    """Like `sse_payload`, then deserializes the JSON.

    Returns None if the line is not usable *or* if the JSON is
    unreadable. The client, for its part, uses `sse_payload` to distinguish the two
    cases and count unreadable lines.
    """
    payload = sse_payload(line)
    if payload is None:
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def single_choice(chunk: dict[str, Any]) -> dict[str, Any]:
    """The chunk's only choice, or an empty dict (chunk of `usage`: choices == []).

    Raises `ValueError` if the chunk carries several choices, i.e. if the request
    asked for `n > 1` completions. We do not demultiplex them: silently keeping
    `choices[0]` would merge several different completions into one stream, which
    is exactly the kind of quiet corruption a measurement harness must refuse.
    """
    choices = chunk.get("choices") or []
    if not choices:
        return {}
    if len(choices) > 1:
        raise ValueError(
            f"chunk carries {len(choices)} choices: `n > 1` is not supported "
            "(their deltas would be merged into a single stream)"
        )
    if not isinstance(choices[0], dict):
        return {}
    return choices[0]


def iter_deltas(chunk: dict[str, Any]) -> list[tuple[Kind, str]]:
    """Deltas carried by a chunk, in order: reasoning, answer, tool.

    A chunk can carry reasoning *and* answer (end of the
    reasoning): we return both rather than lose one.
    """
    delta = single_choice(chunk).get("delta") or {}
    out: list[tuple[Kind, str]] = []
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
    if isinstance(reasoning, str) and reasoning:
        out.append(("reasoning", reasoning))
    text = delta.get("content") or ""
    if isinstance(text, str) and text:
        out.append(("answer", text))
    if delta.get("tool_calls"):
        out.append(("tool", ""))
    return out


def extract_usage(chunk: dict[str, Any]) -> Usage | None:
    """`usage` of the chunk, if present (last chunk in streaming).

    Also reads `completion_tokens_details.reasoning_tokens`, which is the only
    *exact* source for the reasoning/answer token split.
    """
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    reasoning_tokens = (
        details.get("reasoning_tokens") if isinstance(details, dict) else None
    )
    return Usage(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        reasoning_tokens=reasoning_tokens,
    )


def chunk_finish_reason(chunk: dict[str, Any]) -> str | None:
    reason = single_choice(chunk).get("finish_reason")
    return reason if isinstance(reason, str) else None


def assemble_tool_calls(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merges `delta.tool_calls` fragments into complete calls.

    In streaming, the arguments arrive in pieces: `{"loc` then
    `ation": "Lyon"}`. We reconstitute by `index`.
    """
    acc: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        delta = single_choice(chunk).get("delta") or {}
        for fragment in delta.get("tool_calls") or []:
            if not isinstance(fragment, dict):
                continue
            index = fragment.get("index", 0)
            slot = acc.setdefault(index, {"id": None, "name": "", "arguments": ""})
            if fragment.get("id"):
                slot["id"] = fragment["id"]
            function = fragment.get("function") or {}
            if function.get("name"):
                slot["name"] += str(function["name"])
            if function.get("arguments"):
                slot["arguments"] += str(function["arguments"])
    return [acc[index] for index in sorted(acc)]
