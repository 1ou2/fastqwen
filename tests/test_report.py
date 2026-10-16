"""Tests of `qwenbench.report`: are the acceptance criteria reliable?

We build synthetic samples to verify that each criterion
detects what it claims to detect — including failure cases and
*indeterminate* cases (which must not be confused with failures).
"""

import pytest

from qwenbench.client import Sample
from qwenbench.metrics import CompletionMetrics
from qwenbench.report import (
    EFFORT_RATIO_THRESHOLD,
    OFF_CONFIG,
    ON_CONFIG,
    REF_CONFIG,
    evaluate_findings,
    exit_code,
    median_metrics,
    summarize_configs,
)


def make_sample(
    config: str,
    *,
    reasoning_events: int = 0,
    answer_events: int = 5,
    total_tok: int = 5,
    prompt_tok: int = 10,
    decode_tok_s: float = 4.0,
    ttft_s: float = 0.5,
    e2e_s: float = 2.0,
    authoritative: bool = True,
    warnings: tuple = (),
    tool_calls: list | None = None,
    reasoning_tok: int | None = None,
    finish_reason: str = "stop",
) -> Sample:
    non_reasoning_tok = None if reasoning_tok is None else total_tok - reasoning_tok
    metrics = CompletionMetrics(
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        decode_s=1.0,
        decode_tok_s=decode_tok_s,
        inter_chunk_p50_s=0.2,
        inter_chunk_p95_s=0.3,
        n_total_tok=total_tok,
        n_prompt_tok=prompt_tok,
        n_reasoning_events=reasoning_events,
        n_answer_events=answer_events,
        n_tool_events=0,
        reasoning_share=0.0,
        usage_authoritative=authoritative,
        finish_reason=finish_reason,
        n_reasoning_tok=reasoning_tok,
        n_non_reasoning_tok=non_reasoning_tok,
        reasoning_share_basis="tokens" if reasoning_tok is not None else "events",
        warnings=warnings,
    )
    return Sample(
        config=config,
        model="m",
        prompt="p",
        request_body={},
        metrics=metrics,
        tool_calls=tool_calls or [],
        started_at="2026-10-16T22:00:00+02:00",
        http_status=200,
    )


def test_c2_split_detects_wiring():
    samples = [
        make_sample(OFF_CONFIG, reasoning_events=0, answer_events=4),
        make_sample(ON_CONFIG, reasoning_events=120, answer_events=4, total_tok=124),
    ]
    finding = evaluate_findings(samples)["c2_reasoning_split"]
    assert finding["ok"] is True
    assert "reasoning chunks" in finding["detail"]  # no breakdown -> chunk basis


def test_c2_split_fails_if_reasoning_never_arrives():
    samples = [
        make_sample(OFF_CONFIG, reasoning_events=0),
        make_sample(ON_CONFIG, reasoning_events=0),  # parser not wired
    ]
    findings = evaluate_findings(samples)
    assert findings["c2_reasoning_split"]["ok"] is False


def test_c2_uses_exact_tokens_when_available():
    """Real capture: 59 prompt tokens, 41 completion, of which 30 reasoning."""
    samples = [
        make_sample(
            OFF_CONFIG, reasoning_events=0, reasoning_tok=0, total_tok=4,
            answer_events=4,
        ),
        make_sample(
            ON_CONFIG, reasoning_events=30, reasoning_tok=30, total_tok=41,
            answer_events=7,
        ),
    ]
    finding = evaluate_findings(samples)["c2_reasoning_split"]
    assert finding["ok"] is True
    assert "reasoning tokens" in finding["detail"]


def test_c3_effect_measured_on_chunks():
    samples = [
        make_sample("effort_low", reasoning_events=100, total_tok=105),
        make_sample("effort_xhigh", reasoning_events=300, total_tok=305),
    ]
    finding = evaluate_findings(samples)["c3_effort_top_level"]
    assert finding["ok"] is True
    assert finding["verdict"] == "effect_measured"
    assert "reasoning chunks" in finding["detail"]


def test_c3_uses_exact_tokens_when_available():
    samples = [
        make_sample("effort_low", reasoning_tok=100, total_tok=110),
        make_sample("effort_xhigh", reasoning_tok=300, total_tok=310),
    ]
    finding = evaluate_findings(samples)["c3_effort_top_level"]
    assert finding["ok"] is True
    assert finding["verdict"] == "effect_measured"
    assert "reasoning tokens" in finding["detail"]
    assert str(EFFORT_RATIO_THRESHOLD) in finding["detail"]


def test_c3_inconclusive_gap_is_not_a_failure():
    """This is the real case observed on 16/10: ratio ~1 on a trivial prompt."""
    samples = [
        make_sample("effort_low", reasoning_events=200, total_tok=205),
        make_sample("effort_xhigh", reasoning_events=210, total_tok=215),
    ]
    finding = evaluate_findings(samples)["c3_effort_top_level"]
    assert finding["ok"] is None
    assert finding["verdict"] == "inconclusive_gap"
    assert exit_code({"c3": finding}, 2) == 0  # indeterminate != failure


def test_c3_refuses_to_conclude_when_truncated():
    """The real 16/10 hard-prompt case: 504/512 tokens of reasoning at low."""
    samples = [
        make_sample(
            "effort_low_hard", reasoning_tok=504, total_tok=512,
            answer_events=0, finish_reason="length",
        ),
        make_sample(
            "effort_xhigh_hard", reasoning_tok=356, total_tok=512,
            answer_events=153, finish_reason="length",
        ),
    ]
    finding = evaluate_findings(samples)["c3c_effort_hard_prompt"]
    assert finding["ok"] is None
    assert finding["verdict"] == "inconclusive_truncated"
    assert "effort_low_hard" in finding["detail"]
    assert exit_code({"c3c": finding}, 2) == 0


def test_c3_contrary_effect():
    samples = [
        make_sample("effort_low", reasoning_events=300, total_tok=305),
        make_sample("effort_xhigh", reasoning_events=100, total_tok=105),
    ]
    finding = evaluate_findings(samples)["c3_effort_top_level"]
    assert finding["ok"] is None
    assert finding["verdict"] == "contrary_effect"


def test_c3_no_reasoning_is_a_failure():
    samples = [
        make_sample("effort_low", reasoning_events=0),
        make_sample("effort_xhigh", reasoning_events=0),
    ]
    finding = evaluate_findings(samples)["c3_effort_top_level"]
    assert finding["ok"] is False
    assert finding["verdict"] == "no_reasoning"


def test_c3_missing_configuration():
    findings = evaluate_findings([make_sample("other")])
    assert findings["c3_effort_top_level"]["ok"] is None
    assert findings["c3_effort_top_level"]["verdict"] == "absent"
    assert findings["c3c_effort_hard_prompt"]["verdict"] == "absent"


def test_c3c_hard_prompt_is_the_decisive_test():
    samples = [
        make_sample("effort_low_hard", reasoning_events=80, total_tok=85),
        make_sample("effort_xhigh_hard", reasoning_events=400, total_tok=405),
    ]
    finding = evaluate_findings(samples)["c3c_effort_hard_prompt"]
    assert finding["ok"] is True
    assert finding["verdict"] == "effect_measured"


def test_c4_inconsistent_usage_detected():
    good = [make_sample(REF_CONFIG, authoritative=True, prompt_tok=12)]
    assert evaluate_findings(good)["c4_usage_coherent"]["ok"] is True

    bad = [
        make_sample(REF_CONFIG, authoritative=False),
        make_sample(REF_CONFIG, prompt_tok=0),
        make_sample(REF_CONFIG, warnings=("inconsistency: 9 chunks for 2 tokens",)),
    ]
    assert evaluate_findings(bad)["c4_usage_coherent"]["ok"] is False


def test_c5_rate_dispersion():
    stable = [make_sample(REF_CONFIG, decode_tok_s=v) for v in (4.0, 4.1, 3.9)]
    assert evaluate_findings(stable)["c5_throughput_dispersion"]["ok"] is True

    unstable = [make_sample(REF_CONFIG, decode_tok_s=v) for v in (2.0, 4.0)]
    assert evaluate_findings(unstable)["c5_throughput_dispersion"]["ok"] is False

    single = [make_sample(REF_CONFIG, decode_tok_s=4.0)]
    assert evaluate_findings(single)["c5_throughput_dispersion"]["ok"] is None


def test_c6_physical_range():
    plausible = [make_sample(REF_CONFIG, decode_tok_s=4.2)]
    assert evaluate_findings(plausible)["c6_physical_range"]["ok"] is True

    too_slow = [make_sample(REF_CONFIG, decode_tok_s=0.5)]
    assert evaluate_findings(too_slow)["c6_physical_range"]["ok"] is False


def test_c_tool_calls():
    valid = [
        make_sample(
            "tool_call",
            answer_events=0,
            tool_calls=[{"name": "get_weather", "arguments": '{"location": "Lyon"}'}],
        )
    ]
    assert evaluate_findings(valid)["c_tool_calls"]["ok"] is True

    unnamed = [make_sample("tool_call", tool_calls=[{"name": "", "arguments": "{}"}])]
    assert evaluate_findings(unnamed)["c_tool_calls"]["ok"] is False

    broken_json = [
        make_sample("tool_call", tool_calls=[{"name": "f", "arguments": "{oops"}])
    ]
    assert evaluate_findings(broken_json)["c_tool_calls"]["ok"] is False


def test_c_tool_buffered_stream_detected():
    """Real case: TTFT 9.36 s for a total duration of 9.58 s."""
    buffered = [
        make_sample("tool_call", ttft_s=9.36, e2e_s=9.58, decode_tok_s=0.0,
                    tool_calls=[{"name": "get_weather", "arguments": "{}"}])
    ]
    finding = evaluate_findings(buffered)["c_tool_buffered_stream"]
    assert finding["verdict"] == "buffered"
    assert finding["ok"] is None  # informative: never makes the probe fail

    incremental = [
        make_sample("tool_call", ttft_s=0.4, e2e_s=9.0,
                    tool_calls=[{"name": "get_weather", "arguments": "{}"}])
    ]
    assert (
        evaluate_findings(incremental)["c_tool_buffered_stream"]["verdict"]
        == "incremental"
    )


def test_summarize_configs_medians():
    samples = [
        make_sample(REF_CONFIG, decode_tok_s=4.0, total_tok=10, reasoning_tok=3),
        make_sample(REF_CONFIG, decode_tok_s=5.0, total_tok=20, reasoning_tok=5),
        make_sample(ON_CONFIG, decode_tok_s=1.0, reasoning_events=7),
    ]
    summary = summarize_configs(samples)
    assert summary[REF_CONFIG]["runs"] == 2
    assert summary[REF_CONFIG]["decode_tok_s_median"] == 4.5
    assert summary[REF_CONFIG]["n_total_tok_median"] == 15
    assert summary[REF_CONFIG]["n_reasoning_tok_median"] == 4  # median(3, 5)
    assert summary[REF_CONFIG]["n_non_reasoning_tok_median"] == 11  # median(7, 15)
    assert summary[ON_CONFIG]["n_reasoning_events_median"] == 7
    # No breakdown on this configuration -> reported as unknown, never guessed.
    assert summary[ON_CONFIG]["n_reasoning_tok_median"] is None


def test_median_metrics_over_runs():
    samples = [
        make_sample(
            REF_CONFIG, ttft_s=0.4, decode_tok_s=4.0, total_tok=10,
            reasoning_tok=2,
            warnings=("truncated output (finish_reason=length)",),
        ),
        make_sample(REF_CONFIG, ttft_s=0.6, decode_tok_s=4.2, total_tok=20,
                    reasoning_tok=6),
        make_sample(REF_CONFIG, ttft_s=0.5, decode_tok_s=4.1, total_tok=30,
                    reasoning_tok=10),
    ]
    m = median_metrics(samples)
    assert m.ttft_s == pytest.approx(0.5)
    assert m.decode_tok_s == pytest.approx(4.1)
    assert m.n_total_tok == 20
    assert m.n_reasoning_tok == 6  # median(2, 6, 10)
    assert m.n_non_reasoning_tok == 14  # median(8, 14, 20)
    assert m.reasoning_share_basis == "tokens"
    assert m.finish_reason == "stop"
    # The union of the runs' warnings: the exceptional one is the interesting one.
    assert any("truncated" in w for w in m.warnings)


def test_median_metrics_mixed_finish_reason_is_unknown():
    samples = [
        make_sample(REF_CONFIG, finish_reason="length"),
        make_sample(REF_CONFIG, finish_reason="stop"),
    ]
    assert median_metrics(samples).finish_reason is None


def test_median_metrics_needs_a_sample():
    with pytest.raises(ValueError, match="at least one sample"):
        median_metrics([])


def test_exit_code():
    assert exit_code({}, 0) == 2
    assert exit_code({"a": {"ok": True}}, 3) == 0
    assert exit_code({"a": {"ok": None}}, 3) == 0
    assert exit_code({"a": {"ok": False}}, 3) == 1
