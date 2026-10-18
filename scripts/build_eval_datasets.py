#!/usr/bin/env python3
"""Build the frozen evaluation sets: the qualitative 30 and the acceptance set.

Inputs (authored data, committed):
    data/prompts/tools.json             shared tool schemas, referenced by name
    data/prompts/qualitative-30.jsonl   the 30 prompts demoed in the video (§6.8)
    data/prompts/validation-seed.jsonl  held-out prompts for gate G2 (S1-S3)

Outputs (derived, committed for reproducibility):
    benchmarks/prompts/eval_prompts.jsonl
    benchmarks/validation/acceptance.jsonl
    benchmarks/validation/manifest.json

The manifest carries the **exclusion hashes**: `generate_dataset.py` must refuse
any teacher prompt whose normalized hash appears there (§7, contamination rule).
The validation set is a *seed*: it is extended before the H100 run by sampling
the S1-S3 corpora, and the manifest is rebuilt — never edited by hand, so the
hash recorded with a run always matches the prompts it was measured on.

Usage:
    python -m scripts.build_eval_datasets
    python -m scripts.build_eval_datasets --check    # validate only, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qwenbench.eval_sets import (  # noqa: E402
    count_by,
    duplicate_ids,
    duplicates_across,
    duplicates_within,
    exclusion_index,
    inline_tools,
    load_jsonl,
    missing_fields,
    prompt_sha256,
    write_jsonl,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Where the authored prompts live (§9: `data/prompts/`).
DATA_DIR = ROOT / "data" / "prompts"
QUALITATIVE_SRC = DATA_DIR / "qualitative-30.jsonl"
VALIDATION_SRC = DATA_DIR / "validation-seed.jsonl"
TOOLS_SRC = DATA_DIR / "tools.json"

#: Where the built artifacts live (§9: `benchmarks/`).
QUALITATIVE_OUT = ROOT / "benchmarks" / "prompts" / "eval_prompts.jsonl"
VALIDATION_OUT = ROOT / "benchmarks" / "validation" / "acceptance.jsonl"
MANIFEST_OUT = ROOT / "benchmarks" / "validation" / "manifest.json"

#: Existing evaluation sets that must also stay out of the training data. Each is
#: checked for overlap, never merged into the validation set.
OTHER_EVAL_SETS = {"quality200": ROOT / "benchmarks" / "quality" / "quality-200.jsonl"}

QUALITATIVE_REQUIRED = ("id", "family", "category", "language", "prompt", "grade")
VALIDATION_REQUIRED = ("id", "slice", "language", "prompt")

#: Per-dataset schema. Datasets that are only *checked* for contamination (the
#: existing quality set) carry their own schema, so only the dedup keys are required.
REQUIRED_FIELDS = {
    "qualitative": QUALITATIVE_REQUIRED,
    "validation": VALIDATION_REQUIRED,
}
FALLBACK_REQUIRED = ("id", "prompt")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualitative_row(row: dict, catalog: dict) -> dict:
    resolved = inline_tools(row, catalog)
    return {
        "set": "qualitative",
        "id": row["id"],
        "category": row["category"],
        "language": row["language"],
        "prompt": row["prompt"],
        "reference": row.get("reference"),
        "grade": row["grade"],
        "tools": resolved["tools"],
        "prompt_sha256": prompt_sha256(row["prompt"]),
    }


def _validation_row(row: dict, catalog: dict) -> dict:
    resolved = inline_tools(row, catalog)
    return {
        "set": "acceptance",
        "id": row["id"],
        "slice": row["slice"],
        "language": row["language"],
        "prompt": row["prompt"],
        "tools": resolved["tools"],
        "prompt_sha256": prompt_sha256(row["prompt"]),
    }


def _collect_problems(datasets: dict[str, list[dict]]) -> list[str]:
    problems: list[str] = []
    for name, rows in datasets.items():
        required = REQUIRED_FIELDS.get(name, FALLBACK_REQUIRED)
        problems += missing_fields(rows, required, name)
        for row_id in duplicate_ids(rows):
            problems.append(f"{name}: duplicate id {row_id!r}")
        for sha, ids in duplicates_within(rows).items():
            problems.append(f"{name}: duplicate prompt {sha[:12]}… in {ids}")
    for duplicate in duplicates_across(datasets):
        problems.append(f"contamination: {duplicate['sha256'][:12]}… shared by {duplicate['refs']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen evaluation sets.")
    parser.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = parser.parse_args(argv)

    catalog = json.loads(TOOLS_SRC.read_text())
    qualitative_src = load_jsonl(QUALITATIVE_SRC)
    validation_src = load_jsonl(VALIDATION_SRC)

    datasets: dict[str, list[dict]] = {"qualitative": qualitative_src, "validation": validation_src}
    for name, path in OTHER_EVAL_SETS.items():
        if path.is_file():
            datasets[name] = load_jsonl(path)

    problems = _collect_problems(datasets)
    for name, path in OTHER_EVAL_SETS.items():
        if not path.is_file():
            print(f"  [skip] {name}: {path.relative_to(ROOT)} not found")

    print(f"  tools catalog : {len(catalog)} set(s) — {', '.join(catalog)}")
    print(f"  qualitative   : {len(qualitative_src)} prompts from {QUALITATIVE_SRC.relative_to(ROOT)}")
    print(f"  validation    : {len(validation_src)} prompts from {VALIDATION_SRC.relative_to(ROOT)}")
    print(f"  by slice      : {count_by(validation_src, 'slice')}")

    if problems:
        print("\n❌ the evaluation sets are not publishable:")
        for problem in problems:
            print(f"   - {problem}")
        return 1

    print("\n✅ no missing field, no duplicate id, no duplicate prompt, no contamination")

    if args.check:
        print("   (--check: nothing written)")
        return 0

    qualitative_rows = [_qualitative_row(row, catalog) for row in qualitative_src]
    validation_rows = [_validation_row(row, catalog) for row in validation_src]
    write_jsonl(QUALITATIVE_OUT, qualitative_rows)
    write_jsonl(VALIDATION_OUT, validation_rows)

    manifest = {
        "script": "scripts/build_eval_datasets.py",
        "built_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "sources": {
            "qualitative": {
                "path": str(QUALITATIVE_SRC.relative_to(ROOT)),
                "sha256": _sha256(QUALITATIVE_SRC),
                "rows": len(qualitative_src),
            },
            "validation": {
                "path": str(VALIDATION_SRC.relative_to(ROOT)),
                "sha256": _sha256(VALIDATION_SRC),
                "rows": len(validation_src),
            },
            "tools": {
                "path": str(TOOLS_SRC.relative_to(ROOT)),
                "sha256": _sha256(TOOLS_SRC),
                "sets": sorted(catalog),
            },
        },
        "datasets": {
            "qualitative": {
                "path": str(QUALITATIVE_OUT.relative_to(ROOT)),
                "sha256": _sha256(QUALITATIVE_OUT),
                "rows": len(qualitative_rows),
                "by_category": count_by(qualitative_rows, "category"),
                "by_language": count_by(qualitative_rows, "language"),
                "by_grade": count_by(qualitative_rows, "grade"),
            },
            "validation": {
                "path": str(VALIDATION_OUT.relative_to(ROOT)),
                "sha256": _sha256(VALIDATION_OUT),
                "rows": len(validation_rows),
                "by_slice": count_by(validation_rows, "slice"),
                "by_language": count_by(validation_rows, "language"),
                "with_tools": sum(1 for row in validation_rows if row["tools"]),
            },
        },
        "contamination": {
            "checked": sorted(datasets),
            "duplicates": duplicates_across(datasets),
        },
        "exclusion": {
            "note": "normalized prompt hashes the teacher must never train on (§7)",
            "count": len(exclusion_index({k: v for k, v in datasets.items()})),
            "prompts": exclusion_index({k: v for k, v in datasets.items()}),
        },
        "notes": [
            "validation is a seed (S1-S3): extend it from the training corpora before the "
            "H100 run, then rebuild this manifest so the exclusion list stays exact",
            "the 30 qualitative prompts are also measured in the acceptance sweep "
            "(gate G2 covers S2/S3 + the demo prompts)",
        ],
    }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"\n  wrote {QUALITATIVE_OUT.relative_to(ROOT)} ({len(qualitative_rows)} rows)")
    print(f"  wrote {VALIDATION_OUT.relative_to(ROOT)} ({len(validation_rows)} rows)")
    print(f"  wrote {MANIFEST_OUT.relative_to(ROOT)} "
          f"({manifest['exclusion']['count']} exclusion hashes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
