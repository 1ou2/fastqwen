"""Metrics of a generation stream — pure functions.

Input: the sequence of deltas received from the server, each timestamped in *relative
time* since the request was sent. Output: a `CompletionMetrics`.

No network access, no clock read here: everything is testable offline.

Conventions (to know before reading a number):

- **TTFT**: time until the first sign of output (text *or* tool-call
  fragment). An opening chunk without content (role alone) does not count.
- **decode_s**: from the first to the last *token*. The prefill is therefore excluded,
  which is the point: we measure decoding, not reading the prompt.
- **decode_tok_s**: `(completion_tokens - 1) / decode_s`, with the token count
  *of the server* (`usage`) when it is available — it is the only authoritative
  source. Chunks are not tokens: the detokenizer groups several tokens per chunk
  (measured on this project: 148 chunks for 154 tokens on one run, and 2 chunks
  for a 40-token tool call). Without `usage`, we fall back on the number of
  chunks and emit a warning (`usage_authoritative=False`).
- **Reasoning** tokens **count** in the decode throughput (they are decoded
  tokens) and they are **charged to `max_tokens`**, because the cap applies to
  `completion_tokens`, which includes them. Their share is reported separately
  because it drives the total time until the answer, not the throughput.
- **`n_reasoning_tok` / `n_non_reasoning_tok`**: the *exact* split, read from
  `usage.completion_tokens_details.reasoning_tokens`; the second is then
  `completion_tokens - reasoning_tokens`, which covers the answer text **and** any
  tool-call tokens (a tool call is generated text, charged to `max_tokens` too).
  They stay `None` when the server does not report the breakdown: we never present
  a chunk count as a token count.
- The `n_*_events` fields count the **deltas (chunks)** received, not the tokens:
  their use is *ordinal* (comparing two configurations with each other), never
  absolute. A tool call arrives in several fragments, hence in several `tool`
  events, whereas `Sample.tool_calls` gives the real count after assembly.
- **`inter_chunk_*`**, not "inter-token": the interval is measured between
  *chunks*. When a chunk carries several tokens, it overstates the true
  per-token interval by the batching factor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

Kind = Literal["reasoning", "answer", "tool"]

#: Event kinds that correspond to decoded tokens.
TOKEN_KINDS: frozenset[str] = frozenset({"reasoning", "answer"})

#: Event kinds that signal an output (hence a TTFT).
OUTPUT_KINDS: frozenset[str] = frozenset({"reasoning", "answer", "tool"})


@dataclass(frozen=True)
class StreamEvent:
    """A delta received from the server.

    index   : arrival order, 0-based (ours, not the server's `choices[].index`)
    t_rel_s : seconds elapsed since the request was sent
    kind    : "reasoning" | "answer" | "tool"
    text    : text delta ("" for a tool-call fragment)
    """

    index: int
    t_rel_s: float
    kind: Kind
    text: str = ""


@dataclass(frozen=True)
class Usage:
    """Token count reported by the server (last chunk).

    `reasoning_tokens` comes from `completion_tokens_details.reasoning_tokens`
    and is `None` when the server omits the breakdown (or sends `null`).
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class CompletionMetrics:
    ttft_s: float
    e2e_s: float
    decode_s: float
    decode_tok_s: float
    inter_chunk_p50_s: float
    inter_chunk_p95_s: float
    n_total_tok: int
    n_prompt_tok: int | None
    n_reasoning_events: int
    n_answer_events: int
    n_tool_events: int
    reasoning_share: float
    usage_authoritative: bool
    finish_reason: str | None
    #: Exact split (server breakdown) — None when unavailable.
    #: `n_non_reasoning_tok` = answer text tokens + tool-call tokens.
    n_reasoning_tok: int | None = None
    n_non_reasoning_tok: int | None = None
    #: What `reasoning_share` is computed from: exact tokens, or chunk counts.
    reasoning_share_basis: Literal["tokens", "events"] = "events"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in (
            "ttft_s",
            "e2e_s",
            "decode_s",
            "inter_chunk_p50_s",
            "inter_chunk_p95_s",
        ):
            d[key] = round(float(d[key]), 6)
        d["decode_tok_s"] = round(float(self.decode_tok_s), 3)
        d["reasoning_share"] = round(float(self.reasoning_share), 4)
        d["warnings"] = list(self.warnings)
        return d


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile by linear interpolation, `q` in [0, 1]."""
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize_stream(
    events: Sequence[StreamEvent],
    *,
    usage: Usage | None = None,
    t_end_s: float | None = None,
    finish_reason: str | None = None,
) -> CompletionMetrics:
    """Computes the metrics of a generation from its timestamped deltas.

    Raises `ValueError` if the stream is empty, if it carries no output, or if the
    timestamps are not monotonic: in a measurement harness, an unusable stream
    must be noisy, not silently rounded to zero.
    """
    if not events:
        raise ValueError("empty stream: no event received from the server")

    for prev, cur in zip(events, events[1:]):
        if cur.t_rel_s < prev.t_rel_s:
            raise ValueError(
                f"non-monotonic timestamp: event {cur.index} "
                f"({cur.t_rel_s:.6f}s) before event {prev.index} "
                f"({prev.t_rel_s:.6f}s)"
            )

    output = [e for e in events if e.kind == "tool" or e.text]
    if not output:
        raise ValueError("stream without output: no text delta and no tool call")

    token_events = [e for e in output if e.kind in TOKEN_KINDS]
    reasoning_events = [e for e in token_events if e.kind == "reasoning"]
    answer_events = [e for e in token_events if e.kind == "answer"]
    tool_events = [e for e in output if e.kind == "tool"]

    warnings: list[str] = []
    ttft_s = output[0].t_rel_s
    e2e_s = t_end_s if t_end_s is not None else events[-1].t_rel_s
    if e2e_s < ttft_s:
        raise ValueError(
            f"end of stream ({e2e_s:.6f}s) before the first token ({ttft_s:.6f}s)"
        )

    if len(token_events) >= 2:
        decode_s = token_events[-1].t_rel_s - token_events[0].t_rel_s
        inter_chunks = [
            b.t_rel_s - a.t_rel_s
            for a, b in zip(token_events, token_events[1:])
        ]
    else:
        decode_s = 0.0
        inter_chunks = []
        warnings.append("fewer than 2 tokens: decode throughput not measurable")

    if usage is not None and usage.completion_tokens is not None:
        n_total_tok = int(usage.completion_tokens)
        usage_authoritative = True
    else:
        n_total_tok = len(token_events)
        usage_authoritative = False
        warnings.append("usage absent: tokens counted by chunks (approximate)")

    if usage_authoritative and len(token_events) > n_total_tok:
        warnings.append(
            f"inconsistency: {len(token_events)} chunks for "
            f"{n_total_tok} tokens reported by the server"
        )

    # Exact reasoning/answer split, when the server reports the breakdown.
    n_reasoning_tok: int | None = None
    n_non_reasoning_tok: int | None = None
    if usage is not None and usage.reasoning_tokens is not None:
        if (
            usage.completion_tokens is not None
            and usage.reasoning_tokens > usage.completion_tokens
        ):
            warnings.append(
                f"inconsistency: reasoning_tokens ({usage.reasoning_tokens}) > "
                f"completion_tokens ({usage.completion_tokens})"
            )
        else:
            n_reasoning_tok = int(usage.reasoning_tokens)
            if usage.completion_tokens is not None:
                n_non_reasoning_tok = int(usage.completion_tokens) - n_reasoning_tok

    if (
        n_reasoning_tok is not None
        and usage_authoritative
        and n_total_tok > 0
    ):
        reasoning_share = n_reasoning_tok / n_total_tok
        reasoning_share_basis: Literal["tokens", "events"] = "tokens"
    else:
        n_token_events = len(token_events)
        reasoning_share = (
            len(reasoning_events) / n_token_events if n_token_events else 0.0
        )
        reasoning_share_basis = "events"

    if decode_s > 0.0 and n_total_tok >= 2:
        decode_tok_s = (n_total_tok - 1) / decode_s
    else:
        decode_tok_s = 0.0

    if not answer_events:
        warnings.append(
            "no answer token (reasoning alone or truncated output)"
        )
    if finish_reason == "length":
        warnings.append("truncated output (finish_reason=length)")

    return CompletionMetrics(
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        decode_s=decode_s,
        decode_tok_s=decode_tok_s,
        inter_chunk_p50_s=percentile(inter_chunks, 0.50),
        inter_chunk_p95_s=percentile(inter_chunks, 0.95),
        n_total_tok=n_total_tok,
        n_prompt_tok=usage.prompt_tokens if usage is not None else None,
        n_reasoning_events=len(reasoning_events),
        n_answer_events=len(answer_events),
        n_tool_events=len(tool_events),
        reasoning_share=reasoning_share,
        usage_authoritative=usage_authoritative,
        finish_reason=finish_reason,
        n_reasoning_tok=n_reasoning_tok,
        n_non_reasoning_tok=n_non_reasoning_tok,
        reasoning_share_basis=reasoning_share_basis,
        warnings=tuple(warnings),
    )
