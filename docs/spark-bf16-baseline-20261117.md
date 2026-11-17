# Spark BF16 fixed-length baseline: November 17, 2026

This is the first batch-1 serving measurement for R3 in the
[project plan](../project-plan.md). It establishes a fixed-length speed
reference, not a full R3 run or a quality baseline.

## Results

Both workloads completed three measured requests with zero failures. The raw
JSON confirms the exact input and output lengths for every request.

| Input tokens | Output tokens | TTFT median / p95 (ms) | TPOT median / p95 (ms) | Decode tok/s | End-to-end median / p95 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 128 | 520.77 / 561.86 | 228.72 / 239.68 | **4.37** | 29.568 / 31.001 |
| 4,096 | 128 | 3,375.03 / 3,417.74 | 238.11 / 240.77 | **4.20** | 33.615 / 33.996 |

Decode speed is `1000 / median TPOT in ms`; it excludes time to the first
token. The benchmark's aggregate output throughput (4.26 and 3.80 tok/s,
respectively) includes prefill and is not the decode metric. With just three
requests per workload, p95 is descriptive, not a stable tail estimate.

At 4,096 input tokens, median TTFT was about 2.85 seconds higher, while
median decode speed was about 4% lower. The 512-token result was close to the
October [4.36 tok/s probe](../benchmarks/runs/20261016-224112/probe.json),
although the workloads differ.

## Configuration

- DGX Spark (NVIDIA GB10), driver `580.178.04`, local vLLM `0.31.0` with the
  Rust frontend. Repository commit: `ac5b23227b1f206157c263fca8205798f3f4585a`.
- `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
  BF16, text-only, no speculative decoding. Server flags: `--max-model-len 8192`,
  `--max-num-seqs 1`, `--gpu-memory-utilization 0.8`,
  `--no-enable-prefix-caching`, and `--generation-config vllm`.
- Native `vllm bench serve` against local `/v1/completions` with backend
  `openai`, synthetic random prompts (seed `42`, length range ratio `0`),
  temperature `0`, `--ignore-eos`, and max concurrency `1`. Each case had a
  startup probe and one warmup request before the three measured requests.
  No chat template or reasoning-effort arm was used.

## Local Artifacts

- [512-token detailed JSON](../input-512-output-128.json)
- [4,096-token detailed JSON](../input-4096-output-128.json)
- [Benchmark log](../benchmarks/runs/20261117T212754Z-spark-bf16/benchmark.log)
  and [server log](../benchmarks/runs/20261117T212754Z-spark-bf16/server.log)
  (local, Git-ignored logs).

The initial attempt wrote no benchmark log because `RUN` was unset in the
client shell, so `tee` tried to write `/benchmark.log`. That attempt completed
the 512-token case and was interrupted during the 4,096-token case. The second
attempt completed and logged both cases, but only the `tee` path had been fixed:
vLLM still received `--result-dir ''` and saved its JSON in the repository root.
The table uses only the two completed cases from that second attempt. Before
the next run, set `RUN` in the client shell and use the same directory for
`--result-dir "$RUN"` and `tee "$RUN/benchmark.log"`.

## Remaining R3 Work

This run did not measure time to a naturally completed chat answer, reasoning
token counts, peak unified memory, or power. The 512-output-token cases, prefill
ramp, concurrency sweep, and formal run manifest are also outstanding. Reuse
these two workloads unchanged to compare mixed NVFP4 decode speed; use a separate
frozen chat workload with normal EOS and explicit reasoning settings for time
to the complete answer.

## Methodology
In terminal one
cd ~/dev/fastqwen
source .venv/bin/activate
MODEL=Qwen/Qwen3.8-27B
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
RUN="benchmarks/runs/$(date -u +%Y%m%dT%H%M%SZ)-spark-bf16"
mkdir -p "$RUN"
printf 'Use this in terminal 2: RUN=%q\n' "$RUN"

(
  set -x
  HF_HUB_OFFLINE=1 VLLM_USE_RUST_FRONTEND=1 VLLM_ENGINE_READY_TIMEOUT_S=1800 \
    vllm serve "$MODEL" --revision "$REV" --dtype bfloat16 \
    --host 127.0.0.1 --port 8000 --language-model-only \
    --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching --generation-config vllm
) 2>&1 | tee "$RUN/server.log"

In terminal two, export RUN
curl -fsS http://127.0.0.1:8000/health && echo "Ready"
TOKENIZER="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

(
  set -ex
  date -u
  git rev-parse HEAD
  vllm --version
  nvidia-smi --query-gpu=name,driver_version --format=csv
  for INPUT in 512 4096; do
    vllm bench serve --backend openai \
      --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
      --model Qwen/Qwen3.8-27B --tokenizer "$TOKENIZER" \
      --dataset-name random --random-input-len "$INPUT" \
      --random-output-len 128 --random-range-ratio 0 --seed 42 \
      --max-concurrency 1 --num-warmups 1 --num-prompts 3 \
      --temperature 0 --ignore-eos \
      --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95 \
      --save-result --save-detailed --result-dir "$RUN" \
      --result-filename "input-${INPUT}-output-128.json"
  done
) 2>&1 | tee "$RUN/benchmark.log"