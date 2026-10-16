"""Measurement harness of the fastqwen project (Qwen3.8-27B on DGX Spark).

Modules:
  metrics : *pure* computation of the metrics of a generation stream (TTFT, decode
            throughput, inter-token latency, reasoning/answer split)
  sse     : *pure* parsing of the SSE chunks of an OpenAI-compatible server
  client  : minimal HTTP client (httpx) with per-chunk timestamping
  report  : aggregation and evaluation of the acceptance criteria (§6.5 of the plan)

Design rule: everything that produces a number is pure (explicit inputs/outputs, no
clock, no network) and testable offline. The network and the clock are confined to
`client`.
"""

from .client import ChatClient, Sample, SamplingSpec, build_request_body
from .metrics import CompletionMetrics, StreamEvent, Usage, summarize_stream

__all__ = [
    "ChatClient",
    "CompletionMetrics",
    "Sample",
    "SamplingSpec",
    "StreamEvent",
    "Usage",
    "build_request_body",
    "summarize_stream",
]

__version__ = "0.1.0"
