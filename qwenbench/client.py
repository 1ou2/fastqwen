"""Minimal HTTP client to an OpenAI-compatible vLLM server.

Why raw `httpx` rather than the `openai` SDK: we need **per-chunk** timestamping
(TTFT, inter-token latency) and to see what really goes over the wire
(`reasoning_content`, `usage`). The SDK hides exactly these two things.

Only module of the package that touches the network and the clock.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx

from .metrics import CompletionMetrics, StreamEvent, Usage, summarize_stream
from .sse import (
    assemble_tool_calls,
    chunk_finish_reason,
    extract_usage,
    iter_deltas,
    sse_payload,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"


@dataclass(frozen=True)
class SamplingSpec:
    """Parameters of a request. Temperature 0 by default: reproducibility."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 160
    seed: int | None = 0
    thinking: bool = True
    reasoning_effort: str | None = None  # "low" | "medium" | "xhigh"
    #: Diagnostic: push `reasoning_effort` into `chat_template_kwargs` instead
    #: of the top level (the server may honor only one of the two).
    effort_in_template: bool = False
    presence_penalty: float | None = None
    tools: tuple[dict[str, Any], ...] = ()


def build_request_body(
    model: str, prompt: str, spec: SamplingSpec, *, stream: bool = True
) -> dict[str, Any]:
    """Body of the `/v1/chat/completions` request — pure function.

    Made testable: what we send to the server is data, not a side
    effect. `enable_thinking` and `preserve_thinking` follow the model card
    of Qwen3.8; `reasoning_effort` is the official control parameter.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": spec.max_tokens,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "chat_template_kwargs": {
            "enable_thinking": spec.thinking,
            "preserve_thinking": True,
        },
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if spec.top_k is not None:
        body["top_k"] = spec.top_k
    if spec.seed is not None:
        body["seed"] = spec.seed
    if spec.presence_penalty is not None:
        body["presence_penalty"] = spec.presence_penalty
    if spec.reasoning_effort:
        if spec.effort_in_template:
            body["chat_template_kwargs"]["reasoning_effort"] = spec.reasoning_effort
        else:
            body["reasoning_effort"] = spec.reasoning_effort
    if spec.tools:
        body["tools"] = list(spec.tools)
        body["tool_choice"] = "auto"
    return body


@dataclass
class Sample:
    """A measured request: the number, the text, and what was sent."""

    config: str
    model: str
    prompt: str
    request_body: dict[str, Any]
    metrics: CompletionMetrics
    reasoning_text: str = ""
    answer_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    http_status: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "model": self.model,
            "prompt": self.prompt,
            "request_body": self.request_body,
            "started_at": self.started_at,
            "http_status": self.http_status,
            "metrics": self.metrics.to_dict(),
            "reasoning_text": self.reasoning_text,
            "answer_text": self.answer_text,
            "tool_calls": self.tool_calls,
        }


class ChatClient:
    """Streaming chat-completions client, with per-chunk timestamping."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str = "EMPTY",
        timeout_s: float = 900.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            transport=transport,
        )

    # -- lifecycle -------------------------------------------------------
    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # -- introspection ---------------------------------------------------
    def models(self) -> list[str]:
        """Identifiers served by the server (`GET /v1/models`)."""
        response = self._http.get("/v1/models")
        response.raise_for_status()
        return [item.get("id", "") for item in response.json().get("data", [])]

    def fingerprint(self) -> dict[str, Any]:
        """Identity of the server, for the run manifest (§6.4 of the plan)."""
        payload = self._http.get("/v1/models").json()
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return {
            "base_url": self.base_url,
            "model_ids": [item.get("id", "") for item in payload.get("data", [])],
            "v1_models_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    def wait_until_ready(self, timeout_s: float = 1800.0, poll_s: float = 5.0) -> float:
        """Waits until `/v1/models` responds. Returns the wait in seconds.

        Useful during the loading of the 54 GB of BF16 weights: the probe is
        started right away and it waits instead of failing.
        """
        t0 = time.perf_counter()
        last_error = "no attempt"
        while True:
            waited = time.perf_counter() - t0
            if waited > timeout_s:
                raise TimeoutError(
                    f"server not ready after {waited:.0f}s on {self.base_url} "
                    f"(last error: {last_error})"
                )
            try:
                self._http.get("/v1/models").raise_for_status()
                return time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001 - we want the error as is
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(poll_s)

    # -- measurement -----------------------------------------------------
    def stream_once(
        self, prompt: str, spec: SamplingSpec, *, config: str = ""
    ) -> Sample:
        """A streaming request, measured end to end."""
        body = build_request_body(self.model, prompt, spec)
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        events: list[StreamEvent] = []
        chunks: list[dict[str, Any]] = []
        usage: Usage | None = None
        finish_reason: str | None = None
        malformed = 0
        status = 0

        with self._http.stream("POST", "/v1/chat/completions", json=body) as response:
            status = response.status_code
            if status >= 400:
                response.read()
                raise RuntimeError(
                    f"HTTP {status} on /v1/chat/completions: {response.text[:800]}"
                )
            for line in response.iter_lines():
                t_rel = time.perf_counter() - t0
                payload = sse_payload(line)
                if payload is None:
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(chunk, dict):
                    malformed += 1
                    continue
                chunks.append(chunk)
                chunk_usage = extract_usage(chunk)
                if chunk_usage is not None:
                    usage = chunk_usage
                for kind, text in iter_deltas(chunk):
                    events.append(
                        StreamEvent(
                            index=len(events), t_rel_s=t_rel, kind=kind, text=text
                        )
                    )
                reason = chunk_finish_reason(chunk)
                if reason is not None:
                    finish_reason = reason

        t_end = time.perf_counter() - t0
        metrics = summarize_stream(
            events, usage=usage, t_end_s=t_end, finish_reason=finish_reason
        )
        if malformed:
            metrics = replace(
                metrics,
                warnings=metrics.warnings + (f"{malformed} unreadable SSE line(s)",),
            )
        return Sample(
            config=config,
            model=self.model,
            prompt=prompt,
            request_body=body,
            metrics=metrics,
            reasoning_text="".join(
                e.text for e in events if e.kind == "reasoning"
            ),
            answer_text="".join(e.text for e in events if e.kind == "answer"),
            tool_calls=assemble_tool_calls(chunks),
            started_at=started_at,
            http_status=status,
        )


def run_config(
    client: ChatClient,
    prompt: str,
    spec: SamplingSpec,
    *,
    config: str,
    runs: int = 1,
) -> tuple[list[Sample], list[dict[str, Any]]]:
    """Runs the same configuration `runs` times. Returns (samples, errors).

    An error does not interrupt the probe: it is recorded, because the
    fact that a configuration fails is a result in itself.
    """
    samples: list[Sample] = []
    errors: list[dict[str, Any]] = []
    for run_index in range(runs):
        try:
            samples.append(client.stream_once(prompt, spec, config=config))
        except Exception as exc:  # noqa: BLE001 - we log and continue
            errors.append(
                {
                    "config": config,
                    "run": run_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return samples, errors


def median(values: Sequence[float]) -> float:
    """Median (mean of the two central values if the count is even)."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0
