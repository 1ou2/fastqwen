"""Aggregation of the samples and evaluation of the acceptance criteria.

Pure functions: we take `Sample` objects as input, we return serializable
dicts. The criteria are those of §6.5 of the plan (blocks R0-R5 and probe).

Each criterion returns `{"ok": True | False | None, "detail": "..."}`. `None` means
"not testable with these data" (configuration absent, a single run…):
this is different from a failure, and the display distinguishes them.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from .client import Sample, median
from .metrics import CompletionMetrics

#: Configuration names produced by `scripts/probe.py`.
OFF_CONFIG = "thinking_off"
ON_CONFIG = "thinking_on"
LOW_CONFIG = "effort_low"
HIGH_CONFIG = "effort_xhigh"
HIGH_TEMPLATE_CONFIG = "effort_xhigh_tpl"
REF_CONFIG = "decode_ref"
TOOL_CONFIG = "tool_call"

LOW_HARD_CONFIG = "effort_low_hard"
HIGH_HARD_CONFIG = "effort_xhigh_hard"

#: Plausible throughput range for 27B BF16 (54 GB) on GB10 (~273 GB/s).
THROUGHPUT_RANGE_TOK_S = (2.0, 6.5)

#: Threshold of the reasoning-event ratio (xhigh / low) above which we
#: consider that `reasoning_effort` has a measurable effect. Below it, the gap is
#: too small to conclude: verdict *indeterminate*, not failure.
EFFORT_RATIO_THRESHOLD = 1.20


def _by_config(samples: Sequence[Sample], config: str) -> list[Sample]:
    return [s for s in samples if s.config == config]


def _median_of(samples: Iterable[Sample], getter) -> float:
    values = [getter(s) for s in samples]
    return median(values) if values else 0.0


def _median_or_none(samples: Iterable[Sample], getter) -> float | None:
    """Median over the samples that have the value; None if any of them lacks it."""
    values = [getter(s) for s in samples]
    if not values or any(v is None for v in values):
        return None
    return median(values)


def _median_int_or_none(samples: Iterable[Sample], getter) -> int | None:
    value = _median_or_none(samples, getter)
    return None if value is None else int(value)


def _reasoning_measure(samples: Sequence[Sample]) -> tuple[Any, str]:
    """Which quantity to compare for `reasoning_effort`, and what to call it.

    Preferred: the *exact* reasoning token count, available when the server
    reports `completion_tokens_details.reasoning_tokens` for every sample.
    Fallback: the number of reasoning deltas (chunks) — an ordinal proxy only,
    since one chunk may carry several tokens.
    """
    if samples and all(s.metrics.n_reasoning_tok is not None for s in samples):
        return (lambda s: float(s.metrics.n_reasoning_tok)), "reasoning tokens"
    return (lambda s: float(s.metrics.n_reasoning_events)), "reasoning chunks"


def summarize_configs(samples: Sequence[Sample]) -> dict[str, dict[str, Any]]:
    """Medians per configuration (JSON-serializable)."""
    configs: dict[str, list[Sample]] = {}
    for sample in samples:
        configs.setdefault(sample.config, []).append(sample)

    summary: dict[str, dict[str, Any]] = {}
    for name, group in configs.items():
        summary[name] = {
            "runs": len(group),
            "ttft_s_median": round(
                _median_of(group, lambda s: s.metrics.ttft_s), 4
            ),
            "decode_tok_s_median": round(
                _median_of(group, lambda s: s.metrics.decode_tok_s), 3
            ),
            "e2e_s_median": round(_median_of(group, lambda s: s.metrics.e2e_s), 4),
            "n_total_tok_median": _median_of(group, lambda s: s.metrics.n_total_tok),
            "n_reasoning_events_median": _median_of(
                group, lambda s: s.metrics.n_reasoning_events
            ),
            "n_answer_events_median": _median_of(
                group, lambda s: s.metrics.n_answer_events
            ),
            "n_reasoning_tok_median": _median_int_or_none(
                group, lambda s: s.metrics.n_reasoning_tok
            ),
            "n_non_reasoning_tok_median": _median_int_or_none(
                group, lambda s: s.metrics.n_non_reasoning_tok
            ),
            "finish_reasons": sorted(
                {str(s.metrics.finish_reason) for s in group}
            ),
        }
    return summary


def evaluate_findings(
    samples: Sequence[Sample],
    *,
    off_config: str = OFF_CONFIG,
    on_config: str = ON_CONFIG,
    low_config: str = LOW_CONFIG,
    high_config: str = HIGH_CONFIG,
    high_template_config: str = HIGH_TEMPLATE_CONFIG,
    low_hard_config: str = LOW_HARD_CONFIG,
    high_hard_config: str = HIGH_HARD_CONFIG,
    ref_config: str = REF_CONFIG,
    tool_config: str = TOOL_CONFIG,
) -> dict[str, dict[str, Any]]:
    """Evaluates the acceptance criteria of unit 1 on the samples."""
    findings: dict[str, dict[str, Any]] = {}

    # Criterion 2 — is the reasoning/answer split wired?
    off = _by_config(samples, off_config)
    on = _by_config(samples, on_config)
    if not off or not on:
        findings["c2_reasoning_split"] = {
            "ok": None,
            "detail": f"missing configurations ({off_config} / {on_config})",
        }
    else:
        getter, noun = _reasoning_measure(off + on)
        off_reasoning = _median_of(off, getter)
        on_reasoning = _median_of(on, getter)
        findings["c2_reasoning_split"] = {
            "ok": bool(off_reasoning == 0 and on_reasoning > 0),
            "detail": (
                f"{noun}: thinking=off {off_reasoning:.0f}, "
                f"thinking=on {on_reasoning:.0f}"
            ),
        }

    # Criterion 3 — does `reasoning_effort` have a **measurable** effect?
    # Three states: True (effect measured), None (indeterminate: gap within the
    # noise, or a truncated run), False (no reasoning at all). A ratio close to 1
    # does not prove that the parameter is ignored: it may lack room to act, hence
    # the test on a hard prompt (`c3c`), which is the only decisive one.
    def effort_finding(low_name: str, high_name: str, label: str) -> dict[str, Any]:
        low = _by_config(samples, low_name)
        high = _by_config(samples, high_name)
        if not low or not high:
            return {
                "ok": None,
                "verdict": "absent",
                "detail": f"missing configurations ({label})",
            }

        # A run that hit `max_tokens` measures the cap we set, not the model's
        # effort: its token count *is* the cap, so no comparison is possible.
        truncated = sorted(
            {s.config for s in low + high if s.metrics.finish_reason == "length"}
        )
        if truncated:
            return {
                "ok": None,
                "verdict": "inconclusive_truncated",
                "detail": (
                    f"{label}: run(s) hit the max_tokens cap "
                    f"({', '.join(truncated)}) — the count measures the cap, not "
                    "the effort setting; raise the cap or use an easier prompt"
                ),
            }

        getter, noun = _reasoning_measure(low + high)
        exact = noun == "reasoning tokens"
        low_reasoning = _median_of(low, getter)
        high_reasoning = _median_of(high, getter)
        base = (
            f"{label}: xhigh={high_reasoning:.0f} vs low={low_reasoning:.0f} {noun}"
            + ("" if exact else " [exact token split unavailable]")
        )
        if low_reasoning == 0 and high_reasoning == 0:
            return {
                "ok": False,
                "verdict": "no_reasoning",
                "detail": f"{base} — no reasoning observed",
            }
        ratio = (high_reasoning / low_reasoning) if low_reasoning else float("inf")
        threshold = f"{EFFORT_RATIO_THRESHOLD:.2f}"
        if ratio >= EFFORT_RATIO_THRESHOLD:
            ok, verdict = True, "effect_measured"
            suffix = f"effect measured (ratio ≥ threshold {threshold})"
        elif ratio < 0.90:
            ok, verdict = None, "contrary_effect"
            suffix = (
                "unexpected contrary effect (xhigh reasons less than low, "
                f"ratio < 0.90 < threshold {threshold}) — to investigate"
            )
        else:
            ok, verdict = None, "inconclusive_gap"
            suffix = (
                "inconclusive gap: the parameter is accepted but with no effect "
                f"measurable here (ratio below the threshold {threshold}; prompt too easy, "
                "or parameter really ignored)"
            )
        return {
            "ok": ok,
            "verdict": verdict,
            "detail": f"{base} (ratio {ratio:.2f}) — {suffix}",
        }

    findings["c3_effort_top_level"] = effort_finding(
        low_config, high_config, "reasoning_effort (top level)"
    )
    findings["c3b_effort_via_template"] = effort_finding(
        low_config, high_template_config, "reasoning_effort (chat_template_kwargs)"
    )
    findings["c3c_effort_hard_prompt"] = effort_finding(
        low_hard_config, high_hard_config, "reasoning_effort (hard prompt)"
    )

    # Criterion 4 — does the server report a consistent `usage`?
    if not samples:
        findings["c4_usage_coherent"] = {"ok": None, "detail": "no sample"}
    else:
        incoherent = [
            s.config
            for s in samples
            if not s.metrics.usage_authoritative
            or s.metrics.n_prompt_tok in (None, 0)
            or any("inconsistency" in w for w in s.metrics.warnings)
        ]
        prompt_median = _median_of(
            samples, lambda s: float(s.metrics.n_prompt_tok or 0)
        )
        completion_median = _median_of(samples, lambda s: s.metrics.n_total_tok)
        findings["c4_usage_coherent"] = {
            "ok": not incoherent,
            "detail": (
                f"{len(samples) - len(incoherent)}/{len(samples)} samples "
                f"with authoritative usage; prompt≈{prompt_median:.0f} tokens, "
                f"completion≈{completion_median:.0f}"
                + (f"; suspects: {sorted(set(incoherent))}" if incoherent else "")
            ),
        }

    # Criterion 5 — throughput dispersion over repeated runs
    ref = _by_config(samples, ref_config)
    rates = [s.metrics.decode_tok_s for s in ref if s.metrics.decode_tok_s > 0]
    if len(rates) < 2:
        findings["c5_throughput_dispersion"] = {
            "ok": None,
            "detail": f"{len(rates)} usable run(s) on {ref_config}: "
            "dispersion not measurable",
        }
    else:
        ratio = max(rates) / min(rates)
        findings["c5_throughput_dispersion"] = {
            "ok": bool(ratio <= 1.10),
            "detail": (
                f"min {min(rates):.2f} / max {max(rates):.2f} tok/s "
                f"(ratio {ratio:.3f}, threshold 1.10 over {len(rates)} runs)"
            ),
        }

    # Criterion 6 — plausible physical range for 54 GB BF16 on GB10
    if not rates:
        findings["c6_physical_range"] = {
            "ok": None,
            "detail": f"no throughput measured on {ref_config}",
        }
    else:
        median_rate = median(rates)
        low, high = THROUGHPUT_RANGE_TOK_S
        findings["c6_physical_range"] = {
            "ok": bool(low <= median_rate <= high),
            "detail": (
                f"{median_rate:.2f} tok/s (expected {low}-{high} tok/s for "
                "54 GB BF16 at ~273 GB/s)"
            ),
        }

    # Bonus — are the tool-calling flags wired?
    tool_samples = _by_config(samples, tool_config)
    if not tool_samples:
        findings["c_tool_calls"] = {
            "ok": None,
            "detail": f"missing configuration ({tool_config})",
        }
    else:
        calls = [c for s in tool_samples for c in s.tool_calls]
        named = [c for c in calls if c.get("name")]
        valid_json = 0
        for call in named:
            try:
                json.loads(call.get("arguments") or "")
                valid_json += 1
            except Exception:  # noqa: BLE001 - unreadable argument = result
                pass
        findings["c_tool_calls"] = {
            "ok": bool(named) and valid_json == len(named),
            "detail": (
                f"{len(calls)} assembled tool call(s), "
                f"{len(named)} named, {valid_json} with valid JSON arguments"
            ),
        }
        # Observed behaviour: vLLM only streams the `tool_calls` fragments once
        # the complete block is received, so TTFT ≈ total duration and `decode_s = 0`.
        # Informative (ok=None): this is not a defect, but the time metrics
        # are not usable there — these cases are measured in non-streaming.
        buffered = [
            s
            for s in tool_samples
            if s.metrics.e2e_s > 0 and s.metrics.ttft_s >= 0.90 * s.metrics.e2e_s
        ]
        if buffered:
            buffered_detail = (
                f"{len(buffered)}/{len(tool_samples)} tool request(s) with TTFT "
                "≥ 90% of the total duration: tool_calls fragments buffered by the "
                "server → measure these cases in non-streaming"
            )
        else:
            buffered_detail = "incremental tool stream: TTFT usable"
        findings["c_tool_buffered_stream"] = {
            "ok": None,
            "verdict": "buffered" if buffered else "incremental",
            "detail": buffered_detail,
        }

    return findings


def median_metrics(samples: Sequence[Sample]) -> CompletionMetrics:
    """Median of the metrics over several runs of the same configuration.

    Used for the summary row of the probe. An optional field becomes `None` as
    soon as one run lacks it; `warnings` is the union of all runs, because the
    interesting warnings are precisely the exceptional ones.
    """
    if not samples:
        raise ValueError("median_metrics() needs at least one sample")
    metrics = [s.metrics for s in samples]

    def med(getter: Any) -> float:
        return median([getter(m) for m in metrics])

    warnings: list[str] = []
    for m in metrics:
        for warning in m.warnings:
            if warning not in warnings:
                warnings.append(warning)

    finish_reasons = {m.finish_reason for m in metrics}
    exact_reasoning = all(m.n_reasoning_tok is not None for m in metrics)
    exact_non_reasoning = all(m.n_non_reasoning_tok is not None for m in metrics)
    prompt_known = all(m.n_prompt_tok is not None for m in metrics)
    return CompletionMetrics(
        ttft_s=med(lambda m: m.ttft_s),
        e2e_s=med(lambda m: m.e2e_s),
        decode_s=med(lambda m: m.decode_s),
        decode_tok_s=med(lambda m: m.decode_tok_s),
        inter_chunk_p50_s=med(lambda m: m.inter_chunk_p50_s),
        inter_chunk_p95_s=med(lambda m: m.inter_chunk_p95_s),
        n_total_tok=int(med(lambda m: m.n_total_tok)),
        n_prompt_tok=int(med(lambda m: m.n_prompt_tok)) if prompt_known else None,
        n_reasoning_events=int(med(lambda m: m.n_reasoning_events)),
        n_answer_events=int(med(lambda m: m.n_answer_events)),
        n_tool_events=int(med(lambda m: m.n_tool_events)),
        reasoning_share=med(lambda m: m.reasoning_share),
        usage_authoritative=all(m.usage_authoritative for m in metrics),
        finish_reason=(
            next(iter(finish_reasons)) if len(finish_reasons) == 1 else None
        ),
        n_reasoning_tok=(
            int(med(lambda m: m.n_reasoning_tok)) if exact_reasoning else None
        ),
        n_non_reasoning_tok=int(med(lambda m: m.n_non_reasoning_tok)) if exact_non_reasoning else None,
        reasoning_share_basis=(
            "tokens"
            if all(m.reasoning_share_basis == "tokens" for m in metrics)
            else "events"
        ),
        warnings=tuple(warnings),
    )


def exit_code(findings: dict[str, dict[str, Any]], n_samples: int) -> int:
    """0 if everything is green or not testable, 1 if a criterion fails, 2 if nothing measured."""
    if n_samples == 0:
        return 2
    return 1 if any(f.get("ok") is False for f in findings.values()) else 0
