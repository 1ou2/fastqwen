#!/usr/bin/env python3
"""B1 — dataset generation by the teacher (Stage B, on the H100s).

The teacher is the Qwen3.8-27B itself; the student is trained on its answers.
This script only *generates*: it never trains, and it refuses to start on a
contaminated prompt set.

Design points that matter for a booked cluster:

- **resumable**: the output is a JSONL append log; a rerun skips ids already
  present and replays the one truncated by a crash (``qwenbench.dataset.done_ids``);
- **shardable**: ``--shard-index/--shard-count`` splits the prompt set
  deterministically, so several processes/nodes can share the run without
  coordination;
- **reproducible**: temperature 0, fixed seed, pinned teacher revision, image
  digest and chat-template hash recorded in the run manifest (§6.4);
- **batched**: ``--concurrency`` in-flight requests, because a 27B teacher on
  4×H100 is idle behind a single synchronous client.

The body of each request is built by ``qwenbench.client.build_request_body``, so
generation uses exactly the semantics the measurements use.

Usage:
    python -m scripts.generate_dataset --prompts shards/*.jsonl --dry-run
    python -m scripts.generate_dataset --prompts shards/*.jsonl \\
        --image-digest sha256:... --concurrency 128

Exit codes: 0 complete, 1 refused/bad input, 2 finished with failed requests
(resumable: rerun the same command).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qwenbench.client import DEFAULT_BASE_URL, DEFAULT_MODEL, SamplingSpec, build_request_body  # noqa: E402
from qwenbench.dataset import (  # noqa: E402
    build_record,
    check_prompts,
    done_ids,
    load_prompt_files,
    partition,
    slice_counts,
)
from qwenbench.eval_sets import prompt_sha256  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_EXCLUSION_MANIFEST = ROOT / "benchmarks" / "validation" / "manifest.json"
DEFAULT_OUT_DIR = ROOT / "benchmarks" / "datasets"
TEMPLATE_COPY = ROOT / "cache" / "tokenizers" / "target" / "tokenizer_config.json"

#: Chat template the 3.8 template is expected to be; recorded so a silent template
#: change is visible in the manifest even when the server keeps the same URL.
PROGRESS_EVERY = 50


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _teacher_revision(model: str) -> str | None:
    """Pinned commit recorded by the local HF cache, if the model is cached."""
    if "/" not in model:
        return None
    repo = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model.replace('/', '--')}"
    ref = repo / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else None


def _template_sha256() -> str | None:
    """Hash of the frozen 3.8 chat template, as pinned by the tokenizer contract."""
    if not TEMPLATE_COPY.is_file():
        return None
    config = json.loads(TEMPLATE_COPY.read_text())
    template = config.get("chat_template")
    if isinstance(template, list):
        template = template[0].get("template") if template else None
    return hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None


def _load_exclusions(path: pathlib.Path) -> set[str]:
    if not path.is_file():
        raise SystemExit(
            f"❌ {path.relative_to(ROOT)} is missing: refusing to generate without the "
            "frozen exclusion list (rebuild it with scripts/build_eval_datasets.py)"
        )
    manifest = json.loads(path.read_text())
    entries = (manifest.get("exclusion") or {}).get("prompts") or []
    return {entry["sha256"] for entry in entries if entry.get("sha256")}


def _scan_totals(path: pathlib.Path) -> dict[str, int]:
    """Stream the output back for the manifest; never keeps it in memory."""
    totals = {"records": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "empty": 0}
    if not path.is_file():
        return totals
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            totals["records"] += 1
            usage = record.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                totals[key] += int(usage.get(key) or 0)
            if not (record.get("content") or "").strip():
                totals["empty"] += 1
    return totals


def _print_plan(rows, excluded_count: int, args, output: pathlib.Path, resumed: int) -> None:
    print("\n--- plan ---")
    print(f"  prompts      : {len(rows)} to generate ({resumed} already done, skipped)")
    print(f"  slices       : {slice_counts(rows)}")
    print(f"  exclusions   : {excluded_count} frozen hashes enforced")
    print(f"  shard        : {args.shard_index}/{args.shard_count}")
    print(f"  output       : {output.relative_to(ROOT)}")
    print(f"  teacher      : {args.model} @ {args.base_url}")
    print(f"  pinned       : revision={_teacher_revision(args.model) or 'unknown'} "
          f"image={args.image_digest or 'UNPINNED'}")
    print(f"  template sha : {_template_sha256() or 'unknown'}")
    print(f"  spec         : temp={args.temperature} max_tokens={args.max_tokens} "
          f"thinking={args.thinking} effort={args.effort} seed={args.seed}")
    print(f"  concurrency  : {args.concurrency}")
    print(f"  worst case   : {len(rows) * args.max_tokens:,} completion tokens")


async def _one(
    client: httpx.AsyncClient,
    args,
    spec: SamplingSpec,
    row: dict,
    teacher: dict,
    template_sha256: str | None,
) -> dict:
    body = build_request_body(args.model, row["prompt"], spec, stream=False)
    if row.get("tools"):
        body["tools"] = row["tools"]
        body["tool_choice"] = "auto"

    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            response = await client.post("/v1/chat/completions", json=body)
            if response.status_code >= 500 or response.status_code == 429:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
            response.raise_for_status()
            payload = response.json()
            return build_record(
                row,
                payload,
                teacher=teacher,
                spec={
                    "temperature": spec.temperature,
                    "top_p": spec.top_p,
                    "max_tokens": spec.max_tokens,
                    "seed": spec.seed,
                    "thinking": spec.thinking,
                    "reasoning_effort": spec.reasoning_effort,
                },
                template_sha256=template_sha256,
                finished_at=_now(),
            )
        except Exception as exc:  # network, 5xx, malformed JSON
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                await asyncio.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))
    raise RuntimeError(last_error)


async def _run(args, rows, spec, output: pathlib.Path, teacher: dict, template_sha256: str | None) -> list[dict]:
    errors: list[dict] = []
    write_lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)

    done = 0
    started = time.monotonic()

    async def worker(handle, error_handle) -> None:
        nonlocal done
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                record = await _one(client, args, spec, row, teacher, template_sha256)
            except Exception as exc:
                failure = {"id": row.get("id"), "error": str(exc), "at": _now()}
                async with write_lock:
                    error_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    error_handle.flush()
                errors.append(failure)
                continue
            async with write_lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                done += 1
                if done % PROGRESS_EVERY == 0:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(f"  {done}/{len(rows)} generated — {done / elapsed:.1f} req/s", flush=True)

    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=httpx.Timeout(args.timeout),
        limits=limits,
    ) as client:
        output.parent.mkdir(parents=True, exist_ok=True)
        error_path = output.with_suffix(".errors.jsonl")
        with output.open("a", encoding="utf-8") as handle, error_path.open("a", encoding="utf-8") as error_handle:
            await asyncio.gather(
                *(worker(handle, error_handle) for _ in range(args.concurrency))
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B1 — dataset generation by the teacher.")
    parser.add_argument("--prompts", nargs="+", required=True, help="prompt shards (JSONL)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--exclusion-manifest", default=str(DEFAULT_EXCLUSION_MANIFEST))
    parser.add_argument("--image-digest", default=None)
    parser.add_argument("--effort", default="low", choices=["low", "medium", "xhigh"])
    parser.add_argument("--thinking", action="store_true", help="enable thinking (S4); off by default")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=0, help="generate only the first N (smoke test)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan, call nothing")
    args = parser.parse_args(argv)

    prompt_paths = [pathlib.Path(path).resolve() for path in args.prompts]
    missing = [str(path) for path in prompt_paths if not path.is_file()]
    if missing:
        print(f"❌ missing prompt shard(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    excluded = _load_exclusions(pathlib.Path(args.exclusion_manifest))
    rows = load_prompt_files(prompt_paths)
    problems = check_prompts(rows, excluded)
    if problems:
        print("❌ refusing to generate (fix the input first):", file=sys.stderr)
        for problem in problems[:20]:
            print(f"   - {problem}", file=sys.stderr)
        if len(problems) > 20:
            print(f"   … and {len(problems) - 20} more", file=sys.stderr)
        return 1

    rows = partition(rows, args.shard_index, args.shard_count)
    if args.limit:
        rows = rows[: args.limit]

    spec = SamplingSpec(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        thinking=args.thinking,
        reasoning_effort=args.effort,
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path(args.out_dir)
    output = out_dir / run_id / f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}.jsonl"
    already = done_ids(output)
    rows = [row for row in rows if str(row["id"]) not in already]

    teacher = {
        "model": args.model,
        "revision": _teacher_revision(args.model),
        "image_digest": args.image_digest,
    }
    template_sha256 = _template_sha256()
    _print_plan(rows, len(excluded), args, output, len(already))

    if args.dry_run:
        print("\n  (--dry-run: nothing was called, nothing was written)")
        return 0

    if not args.image_digest:
        print("\n❌ --image-digest is required for a real run (§6.4 manifest)", file=sys.stderr)
        return 1
    if not rows:
        print("\n✅ nothing left to do for this shard")
        return 0

    started_at = _now()
    errors = asyncio.run(_run(args, rows, spec, output, teacher, template_sha256))
    totals = _scan_totals(output)

    manifest = {
        "script": "scripts/generate_dataset.py",
        "stage": "B1",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": _now(),
        "git_commit": _git_commit(),
        "base_url": args.base_url,
        "teacher": {**teacher, "template_sha256": template_sha256},
        "spec": {
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": spec.max_tokens,
            "seed": spec.seed,
            "thinking": spec.thinking,
            "reasoning_effort": spec.reasoning_effort,
        },
        "inputs": {
            "prompts": [{"path": str(path), "sha256": _sha256_file(path)} for path in prompt_paths],
            "exclusion_manifest": {
                "path": str(pathlib.Path(args.exclusion_manifest)),
                "sha256": _sha256_file(pathlib.Path(args.exclusion_manifest)),
                "hashes": len(excluded),
            },
            "shard": {"index": args.shard_index, "count": args.shard_count},
            "limit": args.limit or None,
        },
        "output": {
            "path": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
        },
        "totals": totals,
        "failures": {"count": len(errors), "sample": errors[:10]},
        "notes": [
            "S1-S3 are generated without thinking (predictable token budget); S4 with --thinking",
            "rerun the same command to resume: completed ids are skipped",
        ],
    }
    manifest_path = output.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"\n  records  : {totals['records']} ({totals['empty']} empty answers)")
    print(f"  tokens   : {totals['completion_tokens']:,} completion, "
          f"{totals['reasoning_tokens']:,} reasoning")
    print(f"  failures : {len(errors)}")
    print(f"  manifest : {manifest_path.relative_to(ROOT)}")

    if errors:
        print("\n⚠️  some requests failed after retries — rerun the same command to resume")
        return 2
    print("\n✅ shard complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
