#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
OUTPUT="${1:-data/runs/sft-pilot-20261104}"
GPU="${GPU_INDEX:-2}"
STEPS="${MAX_STEPS:-2000}"
LOG="${OUTPUT}.log"
STATUS="${OUTPUT}.exit-status.txt"

if [[ ! "$GPU" =~ ^[0-9]+$ || ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'GPU_INDEX and MAX_STEPS must be nonnegative/positive integers\n' >&2
    exit 1
fi
if [[ -e "$OUTPUT" || -e "$LOG" || -e "$STATUS" ]]; then
    printf 'Choose a fresh output path; refusing to overwrite %s\n' "$OUTPUT" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$OUTPUT")" || exit 1
FREE="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)" || exit 1
if [[ ! "$FREE" =~ ^[0-9]+$ || "$FREE" -lt 60000 ]]; then
    printf 'GPU %s needs at least 60000 MiB free; teacher containers are not stopped automatically\n' "$GPU" >&2
    exit 1
fi
SPACE="$(df -Pk -- "$(dirname -- "$OUTPUT")" | awk 'NR == 2 {print $4}')"
if [[ ! "$SPACE" =~ ^[0-9]+$ || "$SPACE" -lt 104857600 ]]; then
    printf 'At least 100 GiB free disk is required for tokenized data and optimizer checkpoints\n' >&2
    exit 1
fi

printf 'Single-GPU mixed-source exploratory pilot: GPU=%s steps=%s output=%s\n' "$GPU" "$STEPS" "$OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/.venv/bin/python" -u scripts/train_draft.py \
    --qa-dir data/runs/qa-export-20261104-v1 \
    --output-dir "$OUTPUT" \
    --group s1-en --group s1-fr --group s2-code \
    --max-length 2048 --gradient-accumulation 4 --max-steps "$STEPS" \
    --eval-per-group 64 --save-steps 100 --eval-steps 200 \
    2>&1 | tee "$LOG"
RESULTS=("${PIPESTATUS[@]}")
CODE="${RESULTS[0]}"
if [[ "${RESULTS[1]}" -ne 0 ]]; then
    CODE=1
fi
printf '%s\n' "$CODE" > "$STATUS"
exit "$CODE"