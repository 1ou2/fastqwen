#!/usr/bin/env python3
"""Prepare pinned CFP and Future Code sources for teacher generation."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import pathlib
import random
import re
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.dataset import check_prompts  # noqa: E402
from qwenbench.eval_sets import load_jsonl, prompt_sha256, write_jsonl  # noqa: E402
from scripts.sample_corpora import _load_exclusions  # noqa: E402

SOURCES = {
    "cfp": {
        "repo": "CATIE-AQ/CFP",
        "revision": "8cc06909e4bd09255b3b379d9135f87fabeabc17",
        "slice": "S1",
    },
    "future-code": {
        "repo": "future-architect/Llama-3.3-Future-Code-Instructions",
        "revision": "148be2ee007fcd511356e0a2a8212e7b6e0fda66",
        "slice": "S2",
    },
}
CFP_EXCLUDED_SOURCES = {
    "xglue/qam": "noncommercial_or_unclear_terms",
    "facebook/belebele": "benchmark_source",
    "Mathoctopus/GSM8KInstruct_Parallel": "benchmark_source",
    "qanastek/frenchmedmcqa_validation": "upstream_validation",
}


def _adapt(source: str, row: dict, revision: str, index: int) -> dict | None:
    required = ("inputs", "dataset", "task", "subject") if source == "cfp" else (
        "input", "id", "description_language",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{source} row {index}: missing columns {missing}")
    if source == "future-code" and row["description_language"] != "en":
        return None
    prompt = row["inputs" if source == "cfp" else "input"]
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    prompt = prompt.strip()
    record = {
        "id": f"{SOURCES[source]['slice'].lower()}-{source}-{prompt_sha256(prompt)[:20]}",
        "slice": SOURCES[source]["slice"],
        "prompt": prompt,
        "language": "fr" if source == "cfp" else "en",
        "source_dataset": SOURCES[source]["repo"],
        "source_revision": revision,
        "source_row": index,
    }
    if source == "cfp":
        record.update(upstream_dataset=row["dataset"], task=row["task"], subject=row["subject"])
    else:
        record.update(source_id=row["id"], task_language=row.get("task_language"))
    return record


def _select(rows, source, revision, count, holdout, scan_limit, max_chars, seed, excluded):
    rng = random.Random(seed)
    selected = []
    seen = set()
    stats = collections.Counter()
    candidates = itertools.islice(rows, scan_limit) if scan_limit else rows
    capacity = count + holdout if count else None
    for index, row in enumerate(candidates):
        stats["scanned"] += 1
        record = _adapt(source, row, revision, index)
        if record is None:
            stats["empty_or_non_english"] += 1
            continue
        if source == "cfp" and record["upstream_dataset"] in CFP_EXCLUDED_SOURCES:
            stats[CFP_EXCLUDED_SOURCES[record["upstream_dataset"]]] += 1
            continue
        digest = prompt_sha256(record["prompt"])
        if digest in excluded:
            stats["excluded_prompt"] += 1
            continue
        if digest in seen:
            stats["duplicate"] += 1
            continue
        seen.add(digest)
        if len(record["prompt"]) > max_chars:
            stats["over_character_limit"] += 1
            continue
        stats["eligible"] += 1
        if capacity is None or len(selected) < capacity:
            selected.append(record)
        else:
            replacement = rng.randrange(stats["eligible"])
            if replacement < capacity:
                selected[replacement] = record
    if len(selected) < holdout + (count or 1):
        raise ValueError(
            f"Only {len(selected)} eligible prompts for count={count}, holdout={holdout}; "
            f"increase --scan-limit or reduce --count/--holdout. Stats: {dict(stats)}"
        )
    rng.shuffle(selected)
    return selected[holdout:], selected[:holdout], dict(stats)


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _source_rows(args, revision):
    from huggingface_hub import hf_hub_download

    repo = SOURCES[args.source]["repo"]
    card = pathlib.Path(hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision))
    if args.source == "cfp":
        path = pathlib.Path(args.csv) if args.csv else pathlib.Path(
            hf_hub_download(repo, "train.csv", repo_type="dataset", revision=revision)
        )

        def csv_rows():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                yield from csv.DictReader(handle, delimiter=";")

        return csv_rows(), {"path": str(path), "sha256": _sha256(path)}, card

    from datasets import load_dataset

    files = [f"train-{index:05d}-of-00012.parquet" for index in range(12)]
    random.Random(args.seed).shuffle(files)
    urls = [f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}" for name in files]
    rows = load_dataset(
        "parquet", data_files={"train": urls}, split="train", streaming=True,
        columns=["id", "input", "description_language"],
        filters=[("description_language", "=", "en")],
    )
    return rows, {"urls": urls, "columns": list(rows.column_names), "filter": "description_language=en"}, card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=list(SOURCES))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--revision", help="full source commit SHA; defaults to the pinned revision")
    parser.add_argument("--csv", help="local CFP CSV instead of downloading it")
    parser.add_argument("--count", type=int, help="generation prompts; 0 keeps all eligible (CFP default)")
    parser.add_argument("--holdout", type=int, default=500)
    parser.add_argument("--scan-limit", type=int, help="0 scans everything; Future Code default is 250000")
    parser.add_argument("--max-prompt-chars", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--exclude-prompts", action="append", default=[], help="additional prompt JSONL; repeatable")
    parser.add_argument("--exclusion-manifest", default=str(ROOT / "benchmarks/validation/manifest.json"))
    args = parser.parse_args(argv)
    revision = args.revision or SOURCES[args.source]["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("--revision must be a full 40-character commit SHA")
    args.count = args.count if args.count is not None else (0 if args.source == "cfp" else 50000)
    args.scan_limit = args.scan_limit if args.scan_limit is not None else (0 if args.source == "cfp" else 250000)
    if min(args.count, args.holdout, args.scan_limit) < 0 or args.max_prompt_chars < 1:
        parser.error("counts must be nonnegative and --max-prompt-chars positive")
    if args.csv and args.source != "cfp":
        parser.error("--csv is only valid for CFP")
    output_dir = pathlib.Path(args.out_dir)
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing directory: {output_dir}")
    exclusion_path = pathlib.Path(args.exclusion_manifest)
    excluded = _load_exclusions(exclusion_path)
    extra_inputs = []
    for filename in args.exclude_prompts:
        path = pathlib.Path(filename)
        for row in load_jsonl(path):
            excluded.add(prompt_sha256(row["prompt"]))
        extra_inputs.append({"path": str(path), "sha256": _sha256(path)})
    rows, source_input, card = _source_rows(args, revision)
    prompts, heldout, stats = _select(
        rows, args.source, revision, args.count, args.holdout, args.scan_limit,
        args.max_prompt_chars, args.seed, excluded,
    )
    problems = check_prompts(prompts + heldout, excluded)
    if problems:
        raise ValueError(f"Prepared prompts failed validation: {problems[:10]}")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "prompts.jsonl", prompts)
    write_jsonl(output_dir / "heldout.jsonl", heldout)
    (output_dir / "source-card.md").write_bytes(card.read_bytes())
    distributions = {
        key: dict(sorted(collections.Counter(str(row.get(key)) for row in prompts).items()))
        for key in ("upstream_dataset", "task", "subject", "task_language")
        if any(key in row for row in prompts)
    }
    manifest = {
        "script": "scripts/prepare_generation_prompts.py",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": {**SOURCES[args.source], "revision": revision, "input": source_input},
        "selection": {"count": args.count, "holdout": args.holdout, "scan_limit": args.scan_limit,
                      "max_prompt_chars": args.max_prompt_chars, "seed": args.seed},
        "exclusion_manifest": {"path": str(exclusion_path), "sha256": _sha256(exclusion_path)},
        "extra_exclusions": extra_inputs,
        "cfp_excluded_sources": CFP_EXCLUDED_SOURCES if args.source == "cfp" else {},
        "stats": stats,
        "generation_records": len(prompts), "heldout_records": len(heldout),
        "distributions": distributions,
        "files": {name: {"sha256": _sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
                  for name in ("prompts.jsonl", "heldout.jsonl", "source-card.md")},
        "notes": [
            "Source assistant answers are never exported.",
            "Keep heldout.jsonl out of generation and training inputs.",
            "Exact normalized hashes do not detect translated or paraphrased benchmark overlap.",
            "CFP has per-upstream terms; Future Code declares the Llama 3.3 license. Review source-card.md before use.",
            "Future Code is a reservoir sample of a bounded English prefix in seeded file order, not of the full corpus.",
            "Future Code task_language is not projected: it is absent from some pinned Parquet shards; no language is inferred.",
            "The character limit rejects oversized inputs without truncation; no output-suitability classifier is applied.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(prompts)} generation prompts and {len(heldout)} held-out prompts in {output_dir}")
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())