#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  printf 'Usage: bash scripts/run_french_code_bulk.sh SMOKE_DIR NEW_BULK_DIR\n' >&2
  exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python="$root/.venv/bin/python"
smoke_dir=$(realpath -- "$1")
bulk_dir=$(realpath -m -- "$2")
if [[ -e "$bulk_dir" ]]; then
  printf 'Refusing existing bulk directory: %s\n' "$bulk_dir" >&2
  exit 2
fi

teacher_rev='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
image_digest='sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f'
fr_prompts="$root/data/runs/cfp-prep-20261103/prompts.jsonl"
code_prompts="$root/data/runs/future-code-prep-20261103/prompts.jsonl"

"$python" - "$smoke_dir" "$fr_prompts" "$code_prompts" "$teacher_rev" "$image_digest" <<'PY'
import hashlib
import json
import pathlib
import sys
import httpx

smoke_dir, fr, code, revision, digest = sys.argv[1:]
expected_spec = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8192,
                 "seed": 0, "thinking": False, "reasoning_effort": "low"}
for source, filename in (("fr", fr), ("code", code)):
    path = pathlib.Path(filename)
    with path.open("rb") as handle:
        input_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    preparation = json.loads((path.parent / "manifest.json").read_text())
    assert input_sha == preparation["files"]["prompts.jsonl"]["sha256"], path
    for index in (0, 1):
        manifest_path = pathlib.Path(smoke_dir) / f"{source}-replica-{index}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["failures"]["count"] == 0, manifest_path
        assert manifest["totals"]["records"] == 100 and manifest["totals"]["empty"] == 0, manifest_path
        assert manifest["inputs"]["prompts"][0]["sha256"] == input_sha, manifest_path
        assert manifest["spec"] == expected_spec, manifest_path
        assert manifest["teacher"]["revision"] == revision, manifest_path
        assert manifest["teacher"]["image_digest"] == digest, manifest_path
        assert manifest["base_url"] == f"http://127.0.0.1:{8000 + index}", manifest_path
for port in (8000, 8001):
    response = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=30)
    response.raise_for_status()
    assert any(row["id"] == "Qwen/Qwen3.8-27B" for row in response.json()["data"]), port
print("PASS: frozen input hashes, all smoke manifests, sampling settings, and live model endpoints")
PY

for prompts in "$fr_prompts" "$code_prompts"; do
  "$python" -m scripts.generate_dataset \
    --prompts "$prompts" --teacher-revision "$teacher_rev" --image-digest "$image_digest" \
    --max-tokens 8192 --effort low --temperature 0 --seed 0 --dry-run
done

mkdir -- "$bulk_dir"
cp -- "$smoke_dir/deployment.txt" "$bulk_dir/teacher-deployment.txt"
printf 'Smoke: %s\nBulk: %s\nStarted: %s\n' \
  "$smoke_dir" "$bulk_dir" "$(date --iso-8601=seconds)" | tee "$bulk_dir/launch.txt"

run_replica() {
  local index="$1" source prompts
  for source in fr code; do
    if [[ "$source" == fr ]]; then prompts="$fr_prompts"; else prompts="$code_prompts"; fi
    printf 'Starting %s on replica %s\n' "$source" "$index"
    "$python" -u -m scripts.generate_dataset \
      --prompts "$prompts" --shard-index "$index" --shard-count 2 \
      --base-url "http://127.0.0.1:$((8000 + index))" \
      --teacher-revision "$teacher_rev" --image-digest "$image_digest" \
      --out-dir "$bulk_dir" --run-id "$source-replica-$index" \
      --max-tokens 8192 --effort low --temperature 0 --seed 0 \
      --concurrency 64 --timeout 900 \
      > "$bulk_dir/$source-$index.log" 2>&1 || return "$?"
    printf 'Completed %s on replica %s\n' "$source" "$index"
  done
}

run_replica 0 &
pid0=$!
run_replica 1 &
pid1=$!
status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
printf 'Replica exit codes: %s %s\n' "$status0" "$status1" | tee "$bulk_dir/exit-status.txt"
if [[ "$status0" != 0 || "$status1" != 0 ]]; then
  printf 'A worker failed. Inspect logs; do not relaunch this script into the existing directory.\n' >&2
  exit 1
fi
printf 'Bulk generation complete. Teachers remain running; perform output QA before training.\n'