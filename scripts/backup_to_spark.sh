#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash backup_to_spark.sh [DESTINATION [SSH_HOST [REMOTE_ROOT]]]' \
    'Run on the Spark, not on the temporary GPU server.' \
    'Defaults: ./fastqwen-backup gpu-server /opt/home/gabriel/dev/fastqwen' \
    'Copies data/runs, benchmarks, data/prompts, docs, and root Markdown files.' \
    'Requires rsync on both machines and SSH access to SSH_HOST.' \
    'Reruns resume transfers; local files are never deleted.'
}

if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  usage
  exit 0
fi
if (( $# > 3 )); then
  usage >&2
  exit 2
fi

destination=${1:-./fastqwen-backup}
host=${2:-gpu-server}
remote_root=${3:-/opt/home/gabriel/dev/fastqwen}

if [[ "$host" == -* || "$host" == *:* || "$host" == *[[:space:]]* || "$remote_root" != /* ]]; then
  printf 'Use an SSH alias or user@hostname and an absolute remote root.\n' >&2
  exit 2
fi
for dependency in rsync ssh realpath mktemp; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    printf 'Missing local dependency: %s\n' "$dependency" >&2
    exit 1
  fi
done

mkdir -p -- "$destination"
destination=$(realpath -- "$destination")
source_path="$host:${remote_root%/}/"
filters=(
  --include='/data/'
  --include='/data/runs/***'
  --include='/data/prompts/***'
  --include='/benchmarks/***'
  --include='/docs/***'
  --include='/*.md'
  --exclude='*'
)

printf 'Source: %s\nDestination: %s\n' "$source_path" "$destination"
printf 'Stop writers before the final backup; active files can change during verification.\n'
rsync --archive --checksum --partial --info=progress2 --protect-args \
  "${filters[@]}" -- "$source_path" "$destination/"

verification=$(mktemp)
trap 'rm -f -- "$verification"' EXIT
printf '\nVerifying remote and local file contents with checksums...\n'
rsync --archive --checksum --dry-run --itemize-changes --protect-args \
  "${filters[@]}" -- "$source_path" "$destination/" > "$verification"

if [[ -s "$verification" ]]; then
  printf 'Backup verification found differences; keep the server and rerun:\n' >&2
  cat "$verification" >&2
  exit 1
fi

printf 'Verified backup: %s\n' "$destination"
printf 'Keep this directory until copied into durable storage. No local files were deleted.\n'