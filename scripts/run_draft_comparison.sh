#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
OUTPUT="${1:-data/runs/draft-compare-20261105}"
RAW_GPU="${RAW_GPU_INDEX:-2}"
TRAINED_GPU="${TRAINED_GPU_INDEX:-3}"
LOG="${OUTPUT}.log"
STATUS="${OUTPUT}.exit-status.txt"

if [[ ! "$RAW_GPU" =~ ^[0-9]+$ || ! "$TRAINED_GPU" =~ ^[0-9]+$ || "$RAW_GPU" == "$TRAINED_GPU" ]]; then
    printf 'Choose two distinct integer GPU indices\n' >&2
    exit 1
fi
if [[ -e "$OUTPUT" || -e "$LOG" || -e "$STATUS" ]]; then
    printf 'Refusing existing comparison output/log/status: %s\n' "$OUTPUT" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$OUTPUT")" || exit 1
for GPU in "$RAW_GPU" "$TRAINED_GPU"; do
    FREE="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)" || exit 1
    if [[ ! "$FREE" =~ ^[0-9]+$ || "$FREE" -lt 16000 ]]; then
        printf 'GPU %s needs at least 16000 MiB free; no containers will be stopped\n' "$GPU" >&2
        exit 1
    fi
done
SPACE="$(df -Pk -- "$(dirname -- "$OUTPUT")" | awk 'NR == 2 {print $4}')"
if [[ ! "$SPACE" =~ ^[0-9]+$ || "$SPACE" -lt 10485760 ]]; then
    printf 'At least 10 GiB free disk is required\n' >&2
    exit 1
fi

run_comparison() {
    CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" -u scripts/compare_drafts.py prepare \
        --run-dir "$OUTPUT" --per-group 100 --diagnostic-per-group 64 \
        --max-new-tokens 128 --gammas 1 2 4 8 || return 1
    cp -- scripts/run_draft_comparison.sh "$OUTPUT/source/scripts/run_draft_comparison.sh" || return 1
    CUDA_VISIBLE_DEVICES="$RAW_GPU" "$ROOT/.venv/bin/python" -u scripts/compare_drafts.py evaluate \
        --run-dir "$OUTPUT" --arm raw > "$OUTPUT/raw.log" 2>&1 &
    RAW_PID=$!
    CUDA_VISIBLE_DEVICES="$TRAINED_GPU" "$ROOT/.venv/bin/python" -u scripts/compare_drafts.py evaluate \
        --run-dir "$OUTPUT" --arm trained > "$OUTPUT/trained.log" 2>&1 &
    TRAINED_PID=$!
    printf 'Students started: raw GPU=%s PID=%s; trained GPU=%s PID=%s\n' "$RAW_GPU" "$RAW_PID" "$TRAINED_GPU" "$TRAINED_PID"
    wait "$RAW_PID"
    RAW_CODE=$?
    wait "$TRAINED_PID"
    TRAINED_CODE=$?
    printf 'raw=%s trained=%s\n' "$RAW_CODE" "$TRAINED_CODE" | tee "$OUTPUT/workers.exit-status.txt"
    if [[ "$RAW_CODE" -ne 0 || "$TRAINED_CODE" -ne 0 ]]; then
        return 1
    fi
    "$ROOT/.venv/bin/python" -u scripts/compare_drafts.py summarize \
        --run-dir "$OUTPUT" --bootstrap-samples 2000
}

run_comparison 2>&1 | tee "$LOG"
RESULTS=("${PIPESTATUS[@]}")
CODE="${RESULTS[0]}"
if [[ "${RESULTS[1]}" -ne 0 ]]; then
    CODE=1
fi
printf '%s\n' "$CODE" > "$STATUS"
exit "$CODE"