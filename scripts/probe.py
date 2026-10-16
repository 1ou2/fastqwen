#!/usr/bin/env python3
"""Startup probe (unit 1): validates the server wiring and lays down the
first reference numbers of the project.

Four parts, enabled with `--parts`:

  a  wiring     : thinking on/off, `reasoning_effort` at the top level *and*
                  in `chat_template_kwargs`
  b  throughput : repeated runs of a frozen configuration -> TTFT, tok/s, dispersion
  c  tools      : a request with a tool schema -> `--tool-call-parser`
  d  effort     : low vs xhigh on a **hard** prompt — the only decisive test
                  for `reasoning_effort` (on a trivial prompt, both settings
                  give the same reasoning length)

Writes `benchmarks/runs/<run_id>/probe.json`: raw samples, summary per
configuration, and verdict of the acceptance criteria.

Usage:
    python -m scripts.probe --dry-run          # prints the plan, calls nothing
    python -m scripts.probe --parts d          # effort/hard prompt (~3 min)
    python -m scripts.probe                    # everything (~10 min at 4 tok/s)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qwenbench.client import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ChatClient,
    Sample,
    SamplingSpec,
    run_config,
)
from qwenbench.client import CompletionMetrics  # noqa: E402
from qwenbench.report import (  # noqa: E402
    evaluate_findings,
    exit_code,
    median_metrics,
    summarize_configs,
)

# The prompt constants below are deliberately kept in French: they are *data*
# sent to the model (this project targets French usage, see the plan's scope),
# not prose. Do not translate them: the recorded probe results were produced
# with exactly these strings, and changing them would change model behaviour.
EASY_PROMPT = (
    "Quelle est la capitale de l'Australie ? Réponds en une seule phrase."
)

DEDUCT_PROMPT = (
    "Un train part à 9h15 et roule pendant 2h40. À quelle heure arrive-t-il ? "
    "Termine ta réponse par « Réponse : HH:MM »."
)

TOOL_PROMPT = "Quel temps fait-il à Lyon ? Utilise l'outil get_weather."

#: *Hard* prompt: this is where `reasoning_effort` has room to act.
#: Verifiable answer (9^3 = 729 numbers without digit 7 between 1 and 1000).
HARD_PROMPT = (
    "Combien d'entiers entre 1 et 1000 (inclus) ne contiennent aucun chiffre 7 ? "
    "Vérifie ton raisonnement, puis termine par « Réponse : N »."
)

TOOLS: tuple[dict, ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gives the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, e.g. Lyon"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
)


@dataclass(frozen=True)
class ProbeConfig:
    name: str
    part: str
    prompt: str
    spec: SamplingSpec
    runs: int
    description: str


def build_configs(args: argparse.Namespace) -> list[ProbeConfig]:
    """Probe matrix. The names are those expected by `qwenbench.report`."""
    parts = set(args.parts)
    configs: list[ProbeConfig] = []

    if "a" in parts:
        base = {"thinking": True, "max_tokens": args.cap_a}
        configs += [
            ProbeConfig(
                "thinking_off", "a", EASY_PROMPT,
                SamplingSpec(thinking=False, max_tokens=args.cap_a), 1,
                "thinking disabled",
            ),
            ProbeConfig(
                "thinking_on", "a", EASY_PROMPT,
                SamplingSpec(**base), 1,
                "thinking enabled, default effort",
            ),
            ProbeConfig(
                "effort_low", "a", EASY_PROMPT,
                SamplingSpec(**base, reasoning_effort="low"), 1,
                "effort=low (top-level parameter)",
            ),
            ProbeConfig(
                "effort_xhigh", "a", EASY_PROMPT,
                SamplingSpec(**base, reasoning_effort="xhigh"), 1,
                "effort=xhigh (top-level parameter)",
            ),
            ProbeConfig(
                "effort_xhigh_tpl", "a", EASY_PROMPT,
                SamplingSpec(**base, reasoning_effort="xhigh", effort_in_template=True),
                1,
                "effort=xhigh (in chat_template_kwargs)",
            ),
        ]

    if "b" in parts:
        configs.append(
            ProbeConfig(
                "decode_ref", "b", DEDUCT_PROMPT,
                SamplingSpec(thinking=True, reasoning_effort="low",
                             max_tokens=args.cap_b),
                args.runs,
                f"reference throughput ({args.runs} identical runs)",
            )
        )

    if "c" in parts:
        configs.append(
            ProbeConfig(
                "tool_call", "c", TOOL_PROMPT,
                SamplingSpec(thinking=False, max_tokens=args.cap_c, tools=TOOLS),
                1,
                "tool call (validates --tool-call-parser / --enable-auto-tool-choice)",
            )
        )

    if "d" in parts:
        configs += [
            ProbeConfig(
                "effort_low_hard", "d", HARD_PROMPT,
                SamplingSpec(thinking=True, reasoning_effort="low",
                             max_tokens=args.cap_d),
                1,
                "effort=low on hard prompt",
            ),
            ProbeConfig(
                "effort_xhigh_hard", "d", HARD_PROMPT,
                SamplingSpec(thinking=True, reasoning_effort="xhigh",
                             max_tokens=args.cap_d),
                1,
                "effort=xhigh on hard prompt",
            ),
        ]

    return configs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Startup probe: server wiring + first numbers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--parts", default="abcd", help="subset of 'abcd'")
    parser.add_argument("--runs", type=int, default=3, help="runs of part b")
    parser.add_argument("--cap-a", type=int, default=384, help="max_tokens of part a")
    parser.add_argument("--cap-b", type=int, default=160, help="max_tokens of part b")
    parser.add_argument("--cap-c", type=int, default=160, help="max_tokens of part c")
    parser.add_argument("--cap-d", type=int, default=512, help="max_tokens of part d")
    parser.add_argument("--wait-seconds", type=float, default=1800.0,
                        help="maximum wait for server startup")
    parser.add_argument("--no-wait", action="store_true",
                        help="fail immediately if the server is not ready")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="HTTP timeout per request (s)")
    parser.add_argument("--out-dir", default="benchmarks/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without calling the server")
    args = parser.parse_args(argv)
    if not set(args.parts) <= {"a", "b", "c", "d"} or not args.parts:
        parser.error("--parts must be a non-empty subset of 'abcd'")
    return args


def format_row(label: str, m: CompletionMetrics) -> str:
    warnings = f"  ⚠ {'; '.join(m.warnings)}" if m.warnings else ""
    if m.n_reasoning_tok is not None and m.n_non_reasoning_tok is not None:
        split = (
            f"  [{m.n_reasoning_tok} reasoning + "
            f"{m.n_non_reasoning_tok} other tok]"
        )
    else:
        split = ""
    return (
        f"  {label:<18} ttft {m.ttft_s:6.2f}s  rate {m.decode_tok_s:6.2f} tok/s  "
        f"tok {m.n_total_tok:>4}  reason {m.n_reasoning_events:>4}  "
        f"answer {m.n_answer_events:>4}{split}  "
        f"{m.finish_reason or '-':<11}{warnings}"
    )


def print_plan(configs: list[ProbeConfig], args: argparse.Namespace) -> None:
    print(f"Server      : {args.base_url}  (model {args.model})")
    print(f"Sampling    : temperature 0, top_p 1, seed 0 (reproducibility §6.5.4)")
    total = sum(c.runs for c in configs)
    print(f"Matrix      : {len(configs)} configuration(s), {total} request(s)\n")
    print(f"  {'config':<18} {'part':<7} {'runs':>4}  {'cap':>4}  description")
    for cfg in configs:
        print(
            f"  {cfg.name:<18} {cfg.part:<7} {cfg.runs:>4}  "
            f"{cfg.spec.max_tokens:>4}  {cfg.description}"
        )
    print(
        "\n  Estimate    : ~4 tok/s in BF16 on GB10, hence a theoretical ceiling of "
        f"~{total * 200 / 4 / 60:.0f} min if everything maxes out.\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = build_configs(args)
    run_id = args.run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path(args.out_dir) / run_id

    print_plan(configs, args)
    if args.dry_run:
        print("--dry-run: no call performed.")
        return 0

    started_at = datetime.now().astimezone().isoformat()
    samples: list[Sample] = []
    errors: list[dict] = []
    fingerprint: dict = {}

    try:
        with ChatClient(args.base_url, args.model, timeout_s=args.timeout) as client:
            if not args.no_wait:
                print(f"Waiting for the server (max {args.wait_seconds:.0f}s)…", flush=True)
                waited = client.wait_until_ready(args.wait_seconds)
                print(f"  server ready after {waited:.0f}s\n")
            fingerprint = client.fingerprint()
            print(f"Server       : {fingerprint['model_ids']}")
            print(f"Fingerprint  : {fingerprint['v1_models_sha256'][:16]}…\n")

            for cfg in configs:
                print(
                    f"[{cfg.part}] {cfg.name} — {cfg.description}", flush=True
                )
                got, errs = run_config(
                    client, cfg.prompt, cfg.spec, config=cfg.name, runs=cfg.runs
                )
                samples += got
                errors += errs
                for sample in got:
                    print(format_row(f"run {sample.started_at[11:19]}", sample.metrics))
                if len(got) > 1:
                    print(format_row("median", median_metrics(got)))
                for err in errs:
                    print(f"  ✗ run {err['run']}: {err['error']}")
                print()
    except TimeoutError as exc:
        print(f"✗ {exc}")
        return 2

    findings = evaluate_findings(samples)
    print("Acceptance criteria (§6.5)")
    for name, finding in findings.items():
        mark = {True: "✅", False: "❌", None: "⚠️ "}[finding.get("ok")]
        verdict = finding.get("verdict")
        suffix = f" [{verdict}]" if verdict else ""
        print(f"  {mark} {name}{suffix}: {finding.get('detail')}")

    payload = {
        "probe": "unit-1",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "note": "temperature 0: reproducibility (§6.5.4 of the plan)",
        },
        "fingerprint": fingerprint,
        "configs": [
            {
                "name": cfg.name,
                "part": cfg.part,
                "description": cfg.description,
                "runs": cfg.runs,
                "spec": asdict(cfg.spec),
                "prompt": cfg.prompt,
            }
            for cfg in configs
        ],
        "samples": [s.to_dict() for s in samples],
        "errors": errors,
        "summary": summarize_configs(samples),
        "findings": findings,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe.json"
    out_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWritten: {out_file}")

    code = exit_code(findings, len(samples))
    print(f"Exit code: {code} (0 = green, 1 = failing criterion, 2 = nothing measured)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
