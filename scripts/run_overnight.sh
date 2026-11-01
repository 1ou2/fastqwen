if ! python -m scripts.sample_corpora \
  --parquet-glob 'data/corpora/ultrachat_200k/data/train_sft-*.parquet' \
  --count 180000 --slice S1 --seed 20261101 \
  --output "$NIGHT_DIR/candidates.jsonl" \
  > "$NIGHT_DIR/sampling.log" 2>&1; then
  cat "$NIGHT_DIR/sampling.log"
  exit 1
fi

jq -c 'select((.prompt | length) <= 8000)' \
  "$NIGHT_DIR/candidates.jsonl" > "$NIGHT_DIR/bounded-candidates.jsonl"
jq -c 'select((.prompt | length) > 8000)' \
  "$NIGHT_DIR/candidates.jsonl" > "$NIGHT_DIR/long-candidates-review.jsonl"
wc -l "$NIGHT_DIR/"*candidates*.jsonl

python -u -m scripts.classify_prompts \
  --input "$NIGHT_DIR/bounded-candidates.jsonl" \
  --output-dir "$CLASS_DIR" \
  --base-url http://127.0.0.1:8010 \
  --batch-size 16 --timeout 900 --min-confidence 0.8 --min-margin 0.2 \
  2>&1 | tee "$NIGHT_DIR/classification.log"

jq -e '.status == "completed" and .counts.S1 >= 10000' \
  "$CLASS_DIR/manifest.json"
wc -l "$CLASS_DIR/S1.jsonl" "$CLASS_DIR/S2.jsonl" "$CLASS_DIR/review.jsonl"

[[ "$(ps -p "$D1_PID" -o args=)" == *scripts.serve_decision_model* ]]
kill -TERM "$D1_PID"
docker stop dazzling_snyder

docker run -d --name "${NIGHT}-teacher-0" \
  --gpus '"device=0,1"' --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8000:8000 --entrypoint /bin/bash "$IMAGE_REF" \
  -lc "python3 -m pip uninstall -y torchcodec && exec vllm serve Qwen/Qwen3.8-27B --revision $TEACHER_REV --tensor-parallel-size 2 --port 8000"

docker run -d --name "${NIGHT}-teacher-1" \
  --gpus '"device=2,3"' --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 127.0.0.1:8001:8000 --entrypoint /bin/bash "$IMAGE_REF" \
  -lc "python3 -m pip uninstall -y torchcodec && exec vllm serve Qwen/Qwen3.8-27B --revision $TEACHER_REV --tensor-parallel-size 2 --port 8000"

python -u - <<'PY'
from qwenbench.client import ChatClient
for port in (8000, 8001):
    with ChatClient(base_url=f"http://127.0.0.1:{port}") as client:
        print(f"Waiting for teacher on port {port}", flush=True)
        client.wait_until_ready(timeout_s=1800)
        print(client.models(), flush=True)
PY

for index in 0 1; do
  python -m scripts.generate_dataset \
    --prompts "$CLASS_DIR/S1.jsonl" \
    --shard-index "$index" --shard-count 2 \
    --teacher-revision "$TEACHER_REV" --image-digest "$IMAGE_DIGEST" \
    --out-dir "$NIGHT_DIR" --run-id "replica-$index" \
    --max-tokens 8192 --effort low --temperature 0 --seed 0 \
    --concurrency 64 --timeout 900 --dry-run

  python -u -m scripts.generate_dataset \
    --prompts "$CLASS_DIR/S1.jsonl" --limit 20 \
    --shard-index "$index" --shard-count 2 \
    --base-url "http://127.0.0.1:$((8000 + index))" \
    --teacher-revision "$TEACHER_REV" --image-digest "$IMAGE_DIGEST" \
    --out-dir "$NIGHT_DIR" --run-id "smoke-replica-$index" \
    --max-tokens 8192 --effort low --temperature 0 --seed 0 \
    --concurrency 16 --timeout 900
  jq -e '.failures.count == 0 and .totals.empty == 0 and .totals.records == 20' \
    "$NIGHT_DIR/smoke-replica-$index/manifest.json"
done

for index in 0 1; do
  python -u -m scripts.generate_dataset \
    --prompts "$CLASS_DIR/S1.jsonl" \
    --shard-index "$index" --shard-count 2 \
    --base-url "http://127.0.0.1:$((8000 + index))" \
    --teacher-revision "$TEACHER_REV" --image-digest "$IMAGE_DIGEST" \
    --out-dir "$NIGHT_DIR" --run-id "replica-$index" \
    --max-tokens 8192 --effort low --temperature 0 --seed 0 \
    --concurrency 64 --timeout 900 \
    > "$NIGHT_DIR/generation-$index.log" 2>&1 &
  if [[ "$index" == 0 ]]; then PID0=$!; else PID1=$!; fi
done

STATUS0=0
STATUS1=0
wait "$PID0" || STATUS0=$?
wait "$PID1" || STATUS1=$?
printf 'Replica exit codes: %s %s\n' "$STATUS0" "$STATUS1" \
  | tee "$NIGHT_DIR/exit-status.txt"
jq '{run_id, totals, failures: .failures.count}' \
  "$NIGHT_DIR/replica-0/manifest.json" "$NIGHT_DIR/replica-1/manifest.json"