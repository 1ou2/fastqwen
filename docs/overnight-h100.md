# Overnight S1 generation on four H100s

## Goal and current host state

Generate an exploratory English S1 wave with the existing classification policy.
Do not change thresholds or instructions tonight. Do not invent S2/S3/S4 shards
to keep GPUs busy: those sources are not ready for a production mixture.

At preparation time, GPUs 0-1 host the teacher container `dazzling_snyder`, GPU 2
hosts d1 on port 8010, and GPU 3 is free. Output storage has approximately 3.2 TB
available. Recheck these facts before using the commands below.

The unattended sequence is:

1. Sample up to 180,000 candidate S1 prompts using the current regex sampler.
2. Exclude candidates longer than 8,000 characters for this wave, recording them
   separately. This also guarantees a batch of 16 fits d1's 128,000-character
   request limit. This is a conservative preparation limit, not a token limit.
3. Classify the bounded candidates using the unchanged d1 policy and thresholds.
4. Only after successful classification, stop the explicitly identified d1
   process and the existing teacher container.
5. Start two pinned teacher containers: GPUs 0-1 on port 8000, GPUs 2-3 on 8001.
6. Wait for both servers, validate both partitions, run 20 inference requests
  on each replica, and generate disjoint halves of the accepted S1 prompts
  with 64 concurrent requests per server.

The first phase does **not** use all GPUs. After it finishes, both TP=2 replicas
can use all four. Classification at the measured 48 prompts/s takes roughly an
hour for 180,000 prompts; longer inputs may make it slower. If 100,000 prompts
are accepted and average 814 completion tokens, generation is approximately
81 million completion tokens. At an assumed aggregate 3,000-6,000 tokens/s,
that is approximately 3.8-7.5 hours, plus preparation and startup. These are
planning estimates, not a promise that the job lasts all night or saturates GPUs.

The regex sampling, character limit, and conservative suitability gate bias
this wave toward shorter answers. Keep it separate from a final EN/FR and
multi-slice training mixture. The held-out validation expansion and formal bulk
acceptance gates are still unfinished; this is an exploratory generation wave.

## 1. Identify the decision process

In a normal host terminal, inspect the listener:

```bash
ss -ltnp '( sport = :8010 )'
```

Find the Python PID in the output, then inspect it, substituting the real PID:

```bash
ps -p DECISION_PID -o user,pid,args
```

It must be your `scripts.serve_decision_model` process. The earlier server log's
PID may be stale after a restart. The queued sequence will send SIGTERM to this
specific PID only after classification succeeds. Do not use `killall python`.

Check the current teacher container and classifier:

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
curl --fail http://127.0.0.1:8010/health
nvidia-smi
```

Confirm that `dazzling_snyder` is the teacher to replace, that the reported d1
policy matches the current repository, and that no unrelated job uses GPUs 2-3.

## 2. Create a persistent terminal

```bash
tmux new -s qwen-night
```

If tmux is unavailable, use the cluster's approved persistent-session or batch
scheduler mechanism; do not rely on an SSH foreground process surviving logout.
The following commands assume tmux. Do not stop the d1 server yet: classification
needs it, and it must remain alive until that phase completes.

## 3. Configure the run inside tmux

Run this from the repository root. Replace `DECISION_PID` with the PID verified
above. No package installation or model download is performed by this block.

```bash
cd /opt/home/gabriel/dev/fastqwen
source .venv/bin/activate
set -euo pipefail

export D1_PID=DECISION_PID
export NIGHT="s1-night-$(date +%Y%m%d-%H%M%S)"
export NIGHT_DIR="data/runs/$NIGHT"
export CLASS_DIR="$NIGHT_DIR/classification"
export IMAGE_DIGEST='sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f'
export IMAGE_REF="vllm/vllm-openai@$IMAGE_DIGEST"
export TEACHER_REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'

[[ "$D1_PID" =~ ^[0-9]+$ ]]
[[ "$(ps -p "$D1_PID" -o args=)" == *scripts.serve_decision_model* ]]
mkdir -p "$NIGHT_DIR"
printf 'Night directory: %s\nDecision PID: %s\n' "$NIGHT_DIR" "$D1_PID"
df -h "$NIGHT_DIR"
```

The teacher revision and image are those already used for the smoke runs.
Confirm that the pinned teacher snapshot and image are still cached before
leaving the job unattended:

```bash
test -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/$TEACHER_REV"
docker image inspect "$IMAGE_REF" --format '{{json .RepoDigests}}'
```

If your HF cache location differs, update both the snapshot check and Docker
volume mount below consistently. The commands here use your observed default
cache. If the checks fail, resolve that before continuing.

## 4. Paste the unattended work sequence

Paste the **entire** following block inside the same tmux shell. It queues the
later phases while classification runs. Do not run the generation commands
separately against the old server. Stop-on-error behavior prevents generation
from using incomplete classification exports.

```bash
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
```

If sampling reports fewer than 180,000 eligible prompts, it writes no candidate
file. Read the printed error, use a new NIGHT value, and rerun with `--count
150000`. Do not hide unrelated sampling failures by automatically reducing count.

The shared prompt file plus `--shard-index 0/1 --shard-count 2` prevents duplicate
work between replicas. **Separate run IDs** are necessary because the current
generator writes one `manifest.json` per run directory. Thinking stays disabled
because `--thinking` is deliberately absent. Do not change the cap or policy
mid-run. Containers retain the known TorchCodec-uninstall workaround; the base
image digest alone does not describe that runtime modification. Keep this launch
recipe with the generation manifests.

Smoke outputs deliberately overlap the large wave's inputs. Keep them for
diagnostics, not as additional training records; use only `replica-0` and
`replica-1` as the large-wave outputs.

## 5. Detach and monitor

Detach tmux with **Ctrl+B, then D**. This is not Ctrl+C. Then, from another
terminal, inspect the newest run (or substitute the NIGHT_DIR printed above):

```bash
tmux attach -t qwen-night
```

During classification, progress is visible in the tmux terminal and in
`classification.log`. Once generation starts, monitor from another terminal:

```bash
watch -n 5 nvidia-smi
tail -f data/runs/YOUR_NIGHT_DIRECTORY/generation-0.log \
  data/runs/YOUR_NIGHT_DIRECTORY/generation-1.log
```

Replace YOUR_NIGHT_DIRECTORY with the actual run name. Generation logs should
advance on both replicas. GPU memory allocation alone is not utilization: check
GPU-Util and the request progress. A single instant of low utilization can be
normal during startup or between batches. Sustained low utilization plus stalled
logs needs inspection; increasing concurrency blindly will not fix HTTP errors.

To investigate startup failures:

```bash
docker logs --tail 100 YOUR_NIGHT_DIRECTORY-teacher-0
docker logs --tail 100 YOUR_NIGHT_DIRECTORY-teacher-1
```

The names here use the NIGHT value, not a full filesystem path. Leave the input,
exclusion manifest, tokenizer contract, and code unchanged until both workers
finish. Do not log out of the separate d1 terminal before classification finishes
unless that server is itself in a persistent session.

## 6. In the morning

Check both exit codes and manifests. Exit 0 means complete; exit 2 means some
requests failed. Inspect `.errors.jsonl` and the generation logs. Rerun only the
failed replica with the **same input, shard parameters, run ID, and generation
settings** to retry unfinished requests; append to its existing generation log
rather than overwriting it. Do not resume classification into an existing directory.

Do not assume abrupt mid-write recovery is fully safe: the current generator
skips an incomplete final JSONL line when scanning but does not repair that tail
before appending. Inspect/repair an incomplete final line before resuming an
abruptly interrupted worker. Ordinary failed requests between flushed records
do not have this partial-write problem.

Inspect `totals.finish_reason_length`, `totals.empty`, sample content, repetition,
and prompt diversity before using outputs for training. Exclude incomplete or
degenerate teacher answers according to a documented QA policy. Zero reported
reasoning tokens alone is not proof about reasoning without parser support;
the request's thinking-disabled flag is the intended generation setting.

The Docker containers remain running after generation finishes for inspection.
They do not automatically start more work, train the student, or shut down.
Stop only these named containers once they are no longer needed. Keep the
classification audit, preparation inputs/logs, both generation directories, and
this recipe together; do not commit generated datasets.