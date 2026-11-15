#!/usr/bin/env python3
"""Freeze a deterministic, evaluation-separated calibration mixture."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.eval_sets import load_jsonl, prompt_sha256
from qwenbench.quantization import (
    CALIBRATION_COUNTS, QA_GROUPS, calibration_conversation, file_hash,
    pinned_target, write_manifest,
)

DEFAULT_EXCLUSIONS = [
    ROOT / "benchmarks/prompts/eval_prompts.jsonl",
    ROOT / "benchmarks/quality/quality-200.jsonl",
    ROOT / "benchmarks/validation/acceptance.jsonl",
    ROOT / "data/runs/cfp-prep-20261103/heldout.jsonl",
    ROOT / "data/runs/future-code-prep-20261103/heldout.jsonl",
    ROOT / "data/runs/draft-compare-20261105/prompts.jsonl",
    ROOT / "data/runs/draft-compare-long-20261105/prompts.jsonl",
    ROOT / "data/runs/draft-compare-long-20261105/diagnostic.jsonl",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="GROUP=JSONL")
    parser.add_argument("--samples", nargs="+", metavar="GROUP=COUNT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--exclude-prompts", type=Path, nargs="+", default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--contract", type=Path, default=ROOT / "benchmarks/compat/tokenizer_contract.json")
    parser.add_argument("--smoke", action="store_true", help="Allow a small, incomplete mixture; not a production calibration")
    args = parser.parse_args(argv)
    counts = dict(CALIBRATION_COUNTS)
    if args.smoke and args.samples is None:
        parser.error("--smoke requires explicit --samples GROUP=COUNT")
    if args.samples is not None:
        counts = {}
        for argument in args.samples:
            group, separator, count = argument.partition("=")
            if not separator or group not in CALIBRATION_COUNTS or group in counts or not count.isdigit() or int(count) < 1:
                parser.error("Sample counts must be unique GROUP=COUNT with positive counts")
            counts[group] = int(count)
    if not args.smoke and set(counts) != set(CALIBRATION_COUNTS):
        parser.error("Production calibration requires French, code, tools, and reasoning")
    if not args.smoke and (sum(counts.values()) < 512 or args.max_length < 2048):
        parser.error("Production calibration requires at least 512 sequences and a 2048-token cap")
    if args.max_length < 2 or args.output_dir.exists():
        parser.error("Use a length of at least two and a fresh output directory")
    inputs = []
    for argument in args.input:
        group, separator, filename = argument.partition("=")
        if not separator or group not in counts:
            parser.error("Inputs must be GROUP=JSONL for a requested calibration group")
        path = Path(filename).resolve()
        if any(item["group"] == group and item["path"] == str(path) for item in inputs):
            parser.error("Duplicate calibration input")
        inputs.append({"group": group, "path": str(path), "sha256": file_hash(path)})
    if {item["group"] for item in inputs} != set(counts):
        parser.error("Every requested calibration group needs an input")
    excluded = set()
    exclusions = []
    for path in args.exclude_prompts:
        records = load_jsonl(path)
        for record in records:
            if isinstance(record.get("prompt"), str) and record["prompt"].strip():
                digest = prompt_sha256(record["prompt"])
                if record.get("prompt_sha256", digest) != digest:
                    raise ValueError(f"Excluded prompt hash mismatch: {path}")
            elif isinstance(record.get("prompt_sha256"), str) and len(record["prompt_sha256"]) == 64:
                digest = record["prompt_sha256"]
            else:
                raise ValueError(f"Invalid excluded prompt: {path}")
            excluded.add(digest)
        exclusions.append({"path": str(path.resolve()), "sha256": file_hash(path), "records": len(records)})
    if not excluded:
        raise ValueError("Calibration exclusions cannot be empty")
    contract, snapshot = pinned_target(args.contract)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    template = (snapshot / "chat_template.jinja").read_text()
    stats = Counter()

    def candidates(group):
        seen = set()
        for item in inputs:
            if item["group"] != group:
                continue
            with Path(item["path"]).open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.endswith("\n"):
                        raise ValueError(f"Incomplete calibration JSONL: {item['path']}:{line_number}")
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
                        raise ValueError(f"Invalid calibration record: {item['path']}:{line_number}")
                    if record.get("qa_group") and record["qa_group"] != QA_GROUPS.get(group):
                        continue
                    if "text" in record:
                        text = record["text"]
                        if group in ("tools", "reasoning") or not isinstance(text, str) or not text.strip():
                            raise ValueError("Raw text is supported only for French/code calibration")
                        prompts = [record.get("prompt", text)]
                        if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
                            raise ValueError("Raw calibration text needs a valid prompt when one is supplied")
                        messages = None
                    else:
                        messages, prompts = calibration_conversation(record, group)
                        text = None
                    hashes = [prompt_sha256(prompt) for prompt in prompts]
                    if excluded.intersection(hashes):
                        raise ValueError(f"Calibration contamination: {record['id']}")
                    key = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
                    if key in seen:
                        stats[f"{group}_duplicates"] += 1
                        continue
                    seen.add(key)
                    yield {
                        "id": record["id"], "group": group, "prompt_hashes": hashes,
                        "messages": messages, "text": text, "tools": record.get("tools"),
                        "source": item["path"], "source_line": line_number,
                        "selection_key": hashlib.sha256(f"{args.seed}:{key}".encode()).hexdigest(),
                    }

    selected = []
    for group, count in counts.items():
        group_rows = heapq.nsmallest(count, candidates(group), key=lambda row: row["selection_key"])
        if len(group_rows) != count:
            raise ValueError(f"Insufficient {group} records: {len(group_rows)} < {count}")
        selected.extend(group_rows)
    all_hashes = [digest for row in selected for digest in row["prompt_hashes"]]
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("Selected calibration prompts overlap across records/groups")
    samples = []
    for row in selected:
        text = row["text"]
        if text is None:
            text = tokenizer.apply_chat_template(
                row["messages"], tools=row["tools"], chat_template=template, tokenize=False,
                add_generation_prompt=False, enable_thinking=row["group"] == "reasoning",
                reasoning_effort="low",
            )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            raise ValueError(f"Calibration text has fewer than two tokens: {row['id']}")
        stats["truncated_sequences"] += int(len(ids) > args.max_length)
        metadata = {key: row[key] for key in ("id", "group", "prompt_hashes", "source", "source_line")}
        samples.append({**metadata, "input_ids": ids[:args.max_length], "original_tokens": len(ids)})
    for item in inputs + exclusions:
        if file_hash(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Calibration input changed during preparation: {item['path']}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "calibration.jsonl"
    with output.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    manifest = {
        "script": "scripts/build_calib_set.py", "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(), "smoke": args.smoke,
        "target": contract["models"]["target"], "template_sha256": contract["chat_template"]["sha256"],
        "contract_sha256": file_hash(args.contract), "inputs": inputs, "exclusions": exclusions,
        "counts": counts, "max_length": args.max_length, "seed": args.seed, "stats": dict(stats),
        "tokens": sum(len(sample["input_ids"]) for sample in samples),
        "min_tokens": min(len(sample["input_ids"]) for sample in samples),
        "max_tokens": max(len(sample["input_ids"]) for sample in samples),
        "files": {"calibration.jsonl": file_hash(output)},
        "notes": [
            "Exact normalized user-prompt exclusions; not semantic deduplication.",
            "Conversations use the pinned target template; French/code raw text does not.",
            "Each sequence is explicitly capped, not padded to imply full-length coverage.",
            "Source labels and tool/reasoning structure are checked, not factual correctness or licensing.",
        ],
    }
    write_manifest(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "counts": counts, "tokens": manifest["tokens"],
                      "smoke": args.smoke, "stats": manifest["stats"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
