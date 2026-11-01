#!/usr/bin/env python3
"""Classify prompt JSONL files through the local d1 service and export S1/S2."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.dataset import check_prompts, load_prompt_files
from qwenbench.decision import DecisionClient, POLICY_SHA256, QUESTIONS, route_decision
from qwenbench.eval_sets import prompt_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-margin", type=float, default=0.2)
    parser.add_argument(
        "--exclusion-manifest", type=pathlib.Path,
        default=ROOT / "benchmarks" / "validation" / "manifest.json",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.timeout <= 0:
        parser.error("--batch-size and --timeout must be positive")
    if not 0 <= args.min_confidence <= 1 or not 0 <= args.min_margin <= 1:
        parser.error("Thresholds must be in [0, 1]")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a fresh run directory")

    rows = load_prompt_files(args.input)
    if not rows or any(
        not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
        or not isinstance(row.get("id"), str) or not row["id"].strip()
        for row in rows
    ):
        parser.error("Inputs must contain nonempty string ids and prompts")
    exclusion_bytes = args.exclusion_manifest.read_bytes()
    exclusions = json.loads(exclusion_bytes)["exclusion"]["prompts"]
    problems = check_prompts(rows, {entry["sha256"] for entry in exclusions})
    if problems:
        parser.error("\n".join(problems))

    def file_hash(path: pathlib.Path) -> str:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    with DecisionClient(args.base_url, args.timeout) as client:
        identity = client.identity()
        if args.batch_size > identity["max_batch_size"]:
            parser.error("--batch-size exceeds the server limit")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest: dict[str, Any] = {
            "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "server": identity, "base_url": args.base_url,
            "policy_sha256": POLICY_SHA256, "questions": QUESTIONS,
            "batch_size": args.batch_size, "min_confidence": args.min_confidence,
            "min_margin": args.min_margin,
            "inputs": [{"path": str(path), "sha256": file_hash(path)} for path in args.input],
            "exclusion_sha256": hashlib.sha256(exclusion_bytes).hexdigest(),
            "input_records": len(rows),
        }
        manifest_path = args.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        counts: Counter[str] = Counter()
        try:
            with ExitStack() as stack:
                handles = {
                    name: stack.enter_context((args.output_dir / f"{name}.jsonl").open("x", encoding="utf-8"))
                    for name in ("decisions", "S1", "S2", "review")
                }
                for offset in range(0, len(rows), args.batch_size):
                    batch = rows[offset:offset + args.batch_size]
                    results = client.classify([row["prompt"] for row in batch])
                    for row, result in zip(batch, results, strict=True):
                        routing = route_decision(result, args.min_confidence, args.min_margin)
                        audit = {
                            **row, "prompt_sha256": prompt_sha256(row["prompt"]),
                            "decision": routing, "raw_decision": result,
                            "policy_sha256": POLICY_SHA256,
                        }
                        handles["decisions"].write(json.dumps(audit, ensure_ascii=False) + "\n")
                        destination = routing["label"] if routing["export"] else "review"
                        exported = {**row, "slice": destination} if routing["export"] else audit
                        handles[destination].write(json.dumps(exported, ensure_ascii=False) + "\n")
                        counts[destination] += 1
                    for handle in handles.values():
                        handle.flush()
                    print(f"Classified {sum(counts.values())}/{len(rows)}: {dict(counts)}", flush=True)
            manifest["status"] = "completed"
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            manifest["counts"] = dict(counts)
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())