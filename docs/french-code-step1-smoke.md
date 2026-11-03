# Step 1: restart teachers and smoke-test French/code

Run everything below **on the H100 server**, from the repository root. This
step does not start bulk generation or training. Budget 20-45 minutes for
startup, 400 smoke answers, and inspection; startup or long answers can take more.

Layout: GPUs 0-1 serve host port 8000; GPUs 2-3 serve host port 8001. Both are
independent TP=2 Qwen teacher replicas. Each generates 100 French and 100 code
answers. Together that is 200 answers per source, without duplicates within a
source. The source prompts were already shuffled deterministically in preparation.

Run blocks in order. **If a command fails, stop and investigate before continuing.**
These are interactive instructions; they do not set `bash -e`, which could close
your tmux shell on a failed command. Do not execute all blocks as an unattended
script: source-term review and answer inspection are intentional manual gates.

## 0. Source terms

Read [French/code preparation](french-code-preparation.md#terms-to-review-before-training-or-publishing).
CFP retains sources with individual attribution/share-alike obligations. Future
Code declares the Llama 3.3 license, including a naming condition for distributed
models improved using its generated outputs. Decide whether these terms fit the
project before spending GPU time on them. Preparation is not legal clearance.

## 1. Enter a persistent terminal

On the H100 server:

```bash
tmux new -s fr-code-smoke
```

If that session already exists, attach instead with `tmux attach -t fr-code-smoke`.
Inside it:

```bash
cd /opt/home/gabriel/dev/fastqwen
source .venv/bin/activate
set -uo pipefail

export RUN="fr-code-smoke-$(date +%Y%m%d-%H%M%S)"
export SMOKE_DIR="$PWD/data/runs/$RUN"
export IMAGE_DIGEST='sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f'
export IMAGE_REF="vllm/vllm-openai@$IMAGE_DIGEST"
export TEACHER_REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
export HF_CACHE_DIR="$HOME/.cache/huggingface"
export FR_PROMPTS="$PWD/data/runs/cfp-prep-20261103/prompts.jsonl"
export CODE_PROMPTS="$PWD/data/runs/future-code-prep-20261103/prompts.jsonl"

mkdir -p "$SMOKE_DIR"
printf 'Run directory: %s\n' "$SMOKE_DIR"
printf 'RUN=%s\nIMAGE_REF=%s\nTEACHER_REV=%s\nHF_CACHE_DIR=%s\n' \
  "$RUN" "$IMAGE_REF" "$TEACHER_REV" "$HF_CACHE_DIR" \
  | tee "$SMOKE_DIR/deployment.txt"
```

Do **not** source the old `scripts/qwen-night.sh`: it expects an old d1 PID and
sets paths for the earlier English classification workflow. No d1 server is
needed for this smoke.

## 2. Preflight

```bash
nvidia-smi
docker ps --format 'table {{.Names}}\t{{.Ports}}'
ss -ltnp '( sport = :8000 or sport = :8001 )'
df -h "$SMOKE_DIR"

test -f "$FR_PROMPTS"
test -f "$CODE_PROMPTS"
test -d "$HF_CACHE_DIR/hub/models--Qwen--Qwen3.8-27B/snapshots/$TEACHER_REV"
docker image inspect "$IMAGE_REF" --format '{{json .RepoDigests}}'
```

Confirm all four H100s are available and ports 8000/8001 are free. If another
process owns a GPU or port, identify it before taking action; do not blindly kill
Python processes or stop unrelated containers. Stopped old containers may remain
in `docker ps -a`; they do not occupy GPU memory or these ports.

If the snapshot check fails because your cache is elsewhere, change
`HF_CACHE_DIR` consistently for both the check and Docker mounts. If the image
check fails, resolve the image/cache issue before continuing. These instructions
reuse cached artifacts rather than downloading models or changing packages.

Validate the selected generation inputs, not the held-out files:

```bash
python -m scripts.generate_dataset \
  --prompts "$FR_PROMPTS" \
  --teacher-revision "$TEACHER_REV" --image-digest "$IMAGE_DIGEST" \
  --max-tokens 8192 --dry-run

python -m scripts.generate_dataset \
  --prompts "$CODE_PROMPTS" \
  --teacher-revision "$TEACHER_REV" --image-digest "$IMAGE_DIGEST" \
  --max-tokens 8192 --dry-run
```

Both must exit 0. Expect 41,197 S1 prompts, 50,000 S2 prompts, 290 exclusions,
and a non-null template hash beginning `c3cf9e34`. Preserve all preparation
inputs and the frozen exclusion manifest unchanged during the smoke.

## 3. Start the pinned teacher replicas

```bash
docker run -d --name "${RUN}-teacher-0" \
  --gpus '"device=0,1"' --ipc=host \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -p 127.0.0.1:8000:8000 --entrypoint /bin/bash "$IMAGE_REF" \
  -lc "python3 -m pip uninstall -y torchcodec && exec vllm serve Qwen/Qwen3.8-27B --revision $TEACHER_REV --tensor-parallel-size 2 --port 8000"

docker run -d --name "${RUN}-teacher-1" \
  --gpus '"device=2,3"' --ipc=host \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -p 127.0.0.1:8001:8000 --entrypoint /bin/bash "$IMAGE_REF" \
  -lc "python3 -m pip uninstall -y torchcodec && exec vllm serve Qwen/Qwen3.8-27B --revision $TEACHER_REV --tensor-parallel-size 2 --port 8000"
```

Each command must return a container ID. Unique timestamped names avoid clashes
with stopped old containers. Both containers listen internally on 8000; the
second is exposed as 8001 on the host. Host bindings are loopback-only.

The `torchcodec` uninstall is the workaround that worked on this host. Keep this
recipe with the manifests: the base image digest alone does not capture that
runtime modification. This is not a new vLLM environment or a quantization run.

Wait for model readiness and prove each server can execute an inference request:

```bash
python -u - <<'PY'
import httpx
from qwenbench.client import ChatClient, SamplingSpec, build_request_body

model = "Qwen/Qwen3.8-27B"
for port in (8000, 8001):
    base_url = f"http://127.0.0.1:{port}"
    print(f"Waiting for {base_url}", flush=True)
    with ChatClient(base_url=base_url) as client:
        client.wait_until_ready(timeout_s=1800)
        models = client.models()
        assert model in models, models
    body = build_request_body(
        model, "Reply with a short greeting.",
        SamplingSpec(max_tokens=64, temperature=0, seed=0, thinking=False, reasoning_effort="low"),
        stream=False,
    )
    with httpx.Client(timeout=900) as client:
        response = client.post(f"{base_url}/v1/chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
    content = payload["choices"][0]["message"].get("content") or ""
    assert content.strip(), payload
    print(f"Inference OK on {port}: {content[:120]}", flush=True)
PY
```

Continue only if the block exits 0 and reports inference success for both ports.
`/v1/models` alone is insufficient: a server can respond there before its first
inference succeeds. If startup fails, use the troubleshooting section below.

## 4. Generate the source smokes

French first, then code. Two workers run concurrently within each source, one
per replica. Each generates 100 disjoint prompts using the same source file and
`--shard-index 0/1 --shard-count 2`. Separate run IDs avoid manifest overwrites.

```bash
run_source_smoke() {
  local source="$1" prompts="$2"
  local index pid0 pid1 status0=0 status1=0
  for index in 0 1; do
    python -u -m scripts.generate_dataset \
      --prompts "$prompts" --limit 100 \
      --shard-index "$index" --shard-count 2 \
      --base-url "http://127.0.0.1:$((8000 + index))" \
      --teacher-revision "$TEACHER_REV" --image-digest "$IMAGE_DIGEST" \
      --out-dir "$SMOKE_DIR" --run-id "$source-replica-$index" \
      --max-tokens 8192 --effort low --temperature 0 --seed 0 \
      --concurrency 64 --timeout 900 \
      > "$SMOKE_DIR/$source-$index.log" 2>&1 &
    if [[ "$index" == 0 ]]; then pid0=$!; else pid1=$!; fi
  done
  wait "$pid0" || status0=$?
  wait "$pid1" || status1=$?
  printf '%s worker exit codes: %s %s\n' "$source" "$status0" "$status1" \
    | tee "$SMOKE_DIR/$source-exit-status.txt"
  [[ "$status0" == 0 && "$status1" == 0 ]]
}

run_source_smoke fr "$FR_PROMPTS" && run_source_smoke code "$CODE_PROMPTS"
```

Both sources must report exit codes `0 0`. If French fails, code does not start.
Check the per-worker logs; do not hide failures by starting bulk generation.
These settings match the previous non-thinking generation wave. Do not add
`--thinking`, adjust sampling, or silently change the cap during the smoke.

## 5. Check counts and estimate workload

The check below asserts complete, nonempty, error-free smokes and reports cap
counts, actual mean answer length, and observed wall-clock rates per source.

```bash
python - <<'PY'
import datetime
import json
import os
import pathlib

root = pathlib.Path(os.environ["SMOKE_DIR"])
for source, total_prompts in (("fr", 41197), ("code", 50000)):
    manifests = []
    ids = set()
    for index in (0, 1):
        directory = root / f"{source}-replica-{index}"
        manifest = json.loads((directory / "manifest.json").read_text())
        totals = manifest["totals"]
        assert manifest["failures"]["count"] == 0, manifest["failures"]
        assert totals["records"] == 100 and totals["empty"] == 0, totals
        for line in (directory / f"shard-{index:02d}-of-02.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["id"] not in ids, row["id"]
            ids.add(row["id"])
        manifests.append(manifest)
    tokens = sum(item["totals"]["completion_tokens"] for item in manifests)
    clipped = sum(item["totals"]["finish_reason_length"] for item in manifests)
    start = min(datetime.datetime.fromisoformat(item["started_at"]) for item in manifests)
    finish = max(datetime.datetime.fromisoformat(item["finished_at"]) for item in manifests)
    seconds = (finish - start).total_seconds()
    mean = tokens / 200
    rate = tokens / seconds
    hours = total_prompts * mean / rate / 3600
    print(f"{source}: 200 records; caps={clipped}; mean={mean:.1f} completion tokens")
    print(f"  smoke wall time={seconds:.1f}s; combined rate={rate:.0f} completion tokens/s")
    print(f"  provisional full-source time on both replicas={hours:.2f}h, excluding startup/retries")
    if clipped:
        print("  REVIEW capped answers before deciding on the bulk settings")
print("PASS: all four smoke manifests and disjoint record IDs")
PY
```

Zero caps is desirable, but does not prove quality. Any cap requires inspecting
the affected answer. More than 2 caps in a source's 200 answers (>1%) is a
conservative stop signal for this smoke, not an established statistical gate.
Do not blindly raise the cap: long answers may be repetition rather than useful
content. If a setting changes, create a new smoke run and record the decision.

A 100-request worker is short relative to a bulk run and includes its startup,
prefill, and drain phases. Treat estimated hours as provisional, not a promise;
the source sample can also miss rare very long answers. Keep a startup/retry
reserve when deciding tonight's schedule.

## 6. Inspect actual answers

Print a few prompt/answer pairs per source, plus capped answers. This command
does not run code from the model or modify generated files.

```bash
python - <<'PY'
import json
import os
import pathlib

root = pathlib.Path(os.environ["SMOKE_DIR"])
for source in ("fr", "code"):
    rows = []
    for index in (0, 1):
        path = root / f"{source}-replica-{index}" / f"shard-{index:02d}-of-02.jsonl"
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    examples = sorted(rows, key=lambda row: row["id"])
    selected = examples[::40] + [row for row in examples if row["finish_reason"] == "length"]
    printed = set()
    for row in selected:
        if row["id"] in printed:
            continue
        printed.add(row["id"])
        print(f"\n=== {source}: {row['id']} [{row['finish_reason']}] ===")
        print("PROMPT:", row["prompt"])
        print("ANSWER:\n", row["content"])
PY
```

Inspect more than the five displayed examples per source: aim for at least
10-20 varied answers, using the JSONL files in the editor. Check:

- French: natural French, preserved context, correct requested task, no invented
  missing context, no repeated phrases or endless self-corrections.
- Code: relevant solution, sensible requested language, complete code/explanation,
  no repetition. Visual inspection is not proof that code passes tests.
- Both: no source answers copied into input records, no reasoning spill, no
  systematic refusals, incomplete responses, or formatting/template problems.

Do not execute generated code directly on this server. Any later correctness
grading must use a disposable, network-disabled sandbox with timeouts.

## 7. Stop here and decide on bulk generation

Step 1 passes when both servers answered real requests, all four worker exit
codes are 0, manifests show 100 records each with no failures or empty answers,
caps have been reviewed, and varied answer inspection is acceptable.

Report the run directory and the two source summaries from section 5. Keep the
containers running if bulk generation is the next action. Smoke outputs overlap
the future bulk inputs; archive them for diagnostics but do not add them as extra
training records. Bulk workers need their own run IDs and manifests.

This step does **not** implement or launch the SFT harness. Its smoke is a later
gate once a QA-cleaned training export and trainer exist.

## Monitoring and troubleshooting

From a second H100 terminal, replace the example run name with yours:

```bash
watch -n 5 nvidia-smi
```

In a terminal where the exported variables from section 1 are still present:

```bash
tail -f "$SMOKE_DIR/fr-0.log" "$SMOKE_DIR/fr-1.log"
docker logs --tail 100 "${RUN}-teacher-0"
docker logs --tail 100 "${RUN}-teacher-1"
```

Use the analogous `code-0.log`/`code-1.log` once code starts. `tail -f` before
log creation may report missing files; it is not a server failure. Interrupting
`tail` or `watch` with Ctrl+C stops the monitor, not the workers in another shell.

GPU memory allocation is not utilization. Look for GPU-Util during requests and
advancing worker logs. Startup and gaps between sources can legitimately be idle.
The workers log progress every 50 answers, so a short smoke may be quiet until
the first 50 finish. Inspect Docker logs and HTTP errors rather than increasing concurrency
blindly. A readiness timeout or HTTP failure is a stop signal for this step.

Detach tmux with Ctrl+B, then D. Reattach using `tmux attach -t fr-code-smoke`.
The run-directory files are included in the Spark backup. To save container logs
after the smoke:

```bash
docker logs "${RUN}-teacher-0" > "$SMOKE_DIR/teacher-0.log" 2>&1
docker logs "${RUN}-teacher-1" > "$SMOKE_DIR/teacher-1.log" 2>&1
```

If abandoning the smoke, wait for workers to finish or stop them deliberately
before stopping their servers. Stop only these newly named containers:

```bash
docker stop "${RUN}-teacher-0" "${RUN}-teacher-1"
```

Do not delete the smoke files or old runs. An ordinary failed-request retry can
use the same worker settings and IDs; an abrupt interruption may leave a partial
JSONL tail that must be inspected before resuming, as noted in the overnight
runbook.