#!/usr/bin/env python3
"""Sample generation prompts from local parquet corpora into JSONL shards."""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.eval_sets import prompt_sha256  # noqa: E402

_EXPLICIT_WORD_COUNT_RE = re.compile(r"\b(\d[\d,]*)\s*[- ]?\s*words?\b", re.IGNORECASE)
_S1_SOFTWARE_TASK_RE = re.compile(
    r"\b(?:build|create|develop|design|implement|program|write)\b"
    r"[\s\S]{0,180}\b(?:application|apps?|websites?|sites?|webpages?|platforms?|"
    r"dashboards?|chrome extensions?|software|gui|cms|bots?|games?)\b",
    re.IGNORECASE,
)
_S1_CODE_TASK_RE = re.compile(
    r"\b(?:build|create|develop|design|implement|program|write)\b"
    r"[\s\S]{0,180}\b(?:program|code|script|function|algorithm|programming language)\b",
    re.IGNORECASE,
)
_PROGRAMMING_LANGUAGE_RE = re.compile(
    r"\b(?:python|javascript|typescript|java|ruby|php|swift|kotlin|rust|golang|go)\b"
    r"|c\+\+|c#",
    re.IGNORECASE,
)


def _s1_filter_reason(prompt: str) -> str | None:
    """Reject obvious S2 software-build tasks and extreme-length requests."""
    for match in _EXPLICIT_WORD_COUNT_RE.finditer(prompt):
        requested_words = int(match.group(1).replace(",", ""))
        if requested_words >= 5_000:
            return "s1_extreme_output"
    if _S1_SOFTWARE_TASK_RE.search(prompt):
        return "s1_software_task"
    if _S1_CODE_TASK_RE.search(prompt) and _PROGRAMMING_LANGUAGE_RE.search(prompt):
        return "s1_software_task"
    return None


def _load_exclusions(path: pathlib.Path) -> set[str]:
    if not path.is_file():
        raise SystemExit(f"Exclusion manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = (manifest.get("exclusion") or {}).get("prompts") or []
    return {entry["sha256"] for entry in entries if entry.get("sha256")}


def _sample(
    parquet_files: list[pathlib.Path],
    prompt_field: str,
    slice_name: str,
    count: int,
    seed: int,
    exclusions: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the sampler dependency with: uv pip install datasets") from exc

    rows = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in parquet_files]},
        split="train",
        streaming=True,
    )
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    stats = {
        "scanned": 0,
        "empty": 0,
        "excluded": 0,
        "duplicate": 0,
        "s1_software_task": 0,
        "s1_extreme_output": 0,
        "eligible": 0,
    }

    for source_row in rows:
        stats["scanned"] += 1
        prompt = source_row.get(prompt_field)
        if not isinstance(prompt, str) or not prompt.strip():
            stats["empty"] += 1
            continue

        prompt = prompt.strip()
        digest = prompt_sha256(prompt)
        if digest in exclusions:
            stats["excluded"] += 1
            continue
        if digest in seen:
            stats["duplicate"] += 1
            continue

        if slice_name == "S1":
            rejection = _s1_filter_reason(prompt)
            if rejection is not None:
                stats[rejection] += 1
                continue

        seen.add(digest)
        stats["eligible"] += 1
        record = {
            "id": f"{slice_name.lower()}-ultrachat-{digest[:20]}",
            "slice": slice_name,
            "prompt": prompt,
        }
        if len(selected) < count:
            selected.append(record)
        else:
            replacement = rng.randrange(stats["eligible"])
            if replacement < count:
                selected[replacement] = record

    return selected, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-glob", required=True, help="local parquet path or glob")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--slice", default="S1", choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument(
        "--exclusion-manifest",
        default=str(ROOT / "benchmarks" / "validation" / "manifest.json"),
    )
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error("--count must be at least 1")

    pattern = pathlib.Path(args.parquet_glob)
    if not pattern.is_absolute():
        pattern = ROOT / pattern
    parquet_files = [pathlib.Path(path) for path in sorted(glob.glob(str(pattern)))]
    if not parquet_files:
        raise SystemExit(f"No parquet files matched: {pattern}")

    output = pathlib.Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {output}")

    exclusions = _load_exclusions(pathlib.Path(args.exclusion_manifest))
    selected, stats = _sample(
        parquet_files,
        args.prompt_field,
        args.slice,
        args.count,
        args.seed,
        exclusions,
    )
    if len(selected) != args.count:
        raise SystemExit(
            f"Only found {len(selected)} eligible prompts; requested {args.count}. "
            "No output written."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} prompts to {output}")
    print(f"Source files: {len(parquet_files)}; exclusions: {len(exclusions)} hashes")
    print(f"Rows scanned: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())