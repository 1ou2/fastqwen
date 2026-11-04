#!/usr/bin/env python3
"""Audit completed teacher shards and export conservative text-only SFT splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenbench.eval_sets import load_jsonl, prompt_sha256  # noqa: E402
from qwenbench.qa import POLICY, POLICY_SHA256, assess, validation_member  # noqa: E402


def _file_hash(path: pathlib.Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _cohort(manifest: dict) -> dict:
    teacher = manifest["teacher"]
    identity = {key: teacher[key] for key in ("model", "revision", "image_digest", "template_sha256")}
    if not all(identity.values()) or manifest["spec"].get("thinking") is not False:
        raise ValueError("Text-only export requires a pinned, non-thinking generation cohort")
    return {"teacher": identity, "spec": manifest["spec"]}


def _input_info(argument: str) -> dict:
    group, separator, filename = argument.partition("=")
    if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", group):
        raise ValueError("Each --input must be GROUP=SHARD.jsonl with a lowercase group name")
    path = pathlib.Path(filename).resolve()
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["failures"]["count"] != 0:
        raise ValueError(f"Generation has failures: {manifest_path}")
    recorded_path = pathlib.Path(manifest["output"]["path"])
    if not recorded_path.is_absolute():
        recorded_path = ROOT / recorded_path
    if recorded_path.resolve() != path:
        raise ValueError(f"Manifest output path does not match shard: {path}")
    digest = _file_hash(path)
    if digest != manifest["output"]["sha256"] or path.stat().st_size != manifest["output"]["bytes"]:
        raise ValueError(f"Generation output hash or size mismatch: {path}")
    return {
        "group": group, "path": str(path), "sha256": digest,
        "manifest_path": str(manifest_path), "manifest_sha256": _file_hash(manifest_path),
        "records": manifest["totals"]["records"],
        "completion_tokens": manifest["totals"]["completion_tokens"],
        "cohort": _cohort(manifest),
        "generation_inputs": manifest["inputs"],
    }


def _write_manifest(path: pathlib.Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="GROUP=SHARD.jsonl")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--heldout-prompts", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--validation-fraction", type=float, default=.02)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--exclusion-manifest", type=pathlib.Path, default=ROOT / "benchmarks/validation/manifest.json")
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be strictly between 0 and 1")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a fresh directory")

    inputs = [_input_info(argument) for argument in args.input]
    if len({item["path"] for item in inputs}) != len(inputs):
        parser.error("The same input file was specified more than once")
    cohort = inputs[0]["cohort"]
    if any(item["cohort"] != cohort for item in inputs):
        parser.error("Input manifests describe different generation cohorts/settings")
    exclusion_bytes = args.exclusion_manifest.read_bytes()
    entries = json.loads(exclusion_bytes)["exclusion"]["prompts"]
    excluded = {entry["sha256"] for entry in entries}
    if not excluded or any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in excluded):
        parser.error("Frozen exclusion manifest must contain valid prompt hashes")
    heldout_hashes = set()
    heldout_inputs = []
    for path in args.heldout_prompts:
        rows = load_jsonl(path)
        for row in rows:
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                parser.error(f"Invalid held-out prompt in {path}")
            heldout_hashes.add(prompt_sha256(row["prompt"]))
        heldout_inputs.append({"path": str(path.resolve()), "sha256": _file_hash(path), "records": len(rows)})

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "script": "scripts/export_training_data.py", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "policy": POLICY, "policy_sha256": POLICY_SHA256, "cohort": cohort,
        "inputs": inputs,
        "exclusion_manifest": {"path": str(args.exclusion_manifest.resolve()),
                               "sha256": hashlib.sha256(exclusion_bytes).hexdigest(), "hashes": len(excluded)},
        "heldout_inputs": heldout_inputs, "heldout_hashes": len(heldout_hashes),
        "split": {"seed": args.seed, "validation_fraction": args.validation_fraction,
                  "method": "SHA256(seed:normalized_prompt_sha256) threshold; independent of input order"},
        "notes": [
            "Accepted means passed automated text-only QA, not verified factual/code correctness or license clearance.",
            "Review records are excluded from both training and validation; no automatic override is applied.",
            "Raw target text, reasoning, teacher metadata, and source references are preserved without rewriting.",
            "Duplicate ids or normalized prompts are rejected after the first occurrence in the recorded input order.",
            "The hash split is approximately the requested fraction per group; it is not an exact-size stratified split.",
            "No resampling or balancing is applied. Use qa_group and token counts when selecting training mixture weights.",
            "Upstream dataset terms still apply, including CFP per-source terms and Future Code Llama 3.3 conditions.",
        ],
    }
    _write_manifest(manifest_path, manifest)
    counts = Counter()
    tokens = Counter()
    groups = defaultdict(Counter)
    group_tokens = defaultdict(Counter)
    reason_counts = Counter()
    seen_ids = set()
    seen_prompts = set()
    names = ("train", "validation", "review", "rejected", "decisions")
    try:
        with ExitStack() as stack:
            handles = {name: stack.enter_context((args.output_dir / f"{name}.jsonl").open("x", encoding="utf-8"))
                       for name in names}
            for item in inputs:
                records = 0
                completion_tokens = 0
                stream_hash = hashlib.sha256()
                with pathlib.Path(item["path"]).open("rb") as handle:
                    for line_number, raw in enumerate(handle, 1):
                        stream_hash.update(raw)
                        if not raw.endswith(b"\n"):
                            raise ValueError(f"Incomplete JSONL tail: {item['path']}:{line_number}")
                        if not raw.strip():
                            continue
                        record = json.loads(raw)
                        if not isinstance(record, dict):
                            raise ValueError(f"Non-object record: {item['path']}:{line_number}")
                        teacher = record.get("teacher") or {}
                        if any(teacher.get(key) != cohort["teacher"][key] for key in ("model", "revision", "image_digest")):
                            raise ValueError(f"Record teacher differs from manifest: {item['path']}:{line_number}")
                        if record.get("template_sha256") != cohort["teacher"]["template_sha256"] or record.get("spec") != cohort["spec"]:
                            raise ValueError(f"Record template/spec differs from manifest: {item['path']}:{line_number}")
                        usage = (record.get("usage") or {}).get("completion_tokens")
                        if isinstance(usage, bool) or not isinstance(usage, int) or usage < 0:
                            raise ValueError(f"Invalid completion token count: {item['path']}:{line_number}")
                        records += 1
                        completion_tokens += usage
                        destination, reasons = assess(record)
                        prompt = record.get("prompt")
                        digest = prompt_sha256(prompt) if isinstance(prompt, str) and prompt.strip() else None
                        record_id = record.get("id")
                        if digest is not None:
                            if record.get("prompt_sha256") != digest:
                                destination, reasons = "rejected", ["prompt_hash_mismatch"]
                            elif digest in excluded:
                                destination, reasons = "rejected", ["frozen_evaluation_prompt"]
                            elif digest in heldout_hashes:
                                destination, reasons = "rejected", ["external_heldout_prompt"]
                            elif digest in seen_prompts or (isinstance(record_id, str) and record_id in seen_ids):
                                destination, reasons = "rejected", ["duplicate_id_or_prompt"]
                            seen_prompts.add(digest)
                        if isinstance(record_id, str):
                            seen_ids.add(record_id)
                        if destination == "accepted":
                            destination = "validation" if validation_member(digest, args.seed, args.validation_fraction) else "train"
                        exported = {**record, "qa_group": item["group"], "qa_input_file": item["path"], "qa_input_line": line_number}
                        payload = exported if destination in ("train", "validation") else {"record": exported, "qa_reasons": reasons}
                        handles[destination].write(json.dumps(payload, ensure_ascii=False) + "\n")
                        handles["decisions"].write(json.dumps({
                            "id": record_id, "prompt_sha256": digest, "qa_group": item["group"],
                            "input_file": item["path"], "input_line": line_number,
                            "destination": destination, "reasons": reasons, "completion_tokens": usage,
                        }, ensure_ascii=False) + "\n")
                        counts[destination] += 1
                        tokens[destination] += usage
                        groups[item["group"]][destination] += 1
                        group_tokens[item["group"]][destination] += usage
                        reason_counts.update(reasons)
                if stream_hash.hexdigest() != item["sha256"] or records != item["records"] or completion_tokens != item["completion_tokens"]:
                    raise ValueError(f"Input changed or totals differ from manifest: {item['path']}")
                print(f"Audited {records} records from {item['group']}: {pathlib.Path(item['path']).name}", flush=True)
        if _file_hash(args.exclusion_manifest) != manifest["exclusion_manifest"]["sha256"]:
            raise ValueError("Frozen exclusion manifest changed during export")
        for item in heldout_inputs:
            if _file_hash(pathlib.Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Held-out file changed during export: {item['path']}")
        for item in inputs:
            if _file_hash(pathlib.Path(item["manifest_path"])) != item["manifest_sha256"]:
                raise ValueError(f"Generation manifest changed during export: {item['manifest_path']}")
        manifest["files"] = {f"{name}.jsonl": {"sha256": _file_hash(args.output_dir / f"{name}.jsonl"),
                                               "bytes": (args.output_dir / f"{name}.jsonl").stat().st_size}
                             for name in names}
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            input_records=sum(counts.values()), counts=dict(counts), completion_tokens=dict(tokens),
            groups={group: {"records": dict(values), "completion_tokens": dict(group_tokens[group])}
                    for group, values in groups.items()}, reason_counts=dict(reason_counts),
        )
        _write_manifest(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "counts": dict(counts), "reasons": dict(reason_counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())