# Step 2: French and code bulk generation

The successful smoke is `data/runs/fr-code-smoke-20261103-214906`.
All four workers produced 100 records with zero failures and zero empty answers.
French had zero capped answers; code had one out of 200. Capped and degenerate
bulk answers still need QA filtering before training.

The existing teachers use GPUs 0-1 and 2-3, exposed on host ports 8000 and 8001.
Do not restart or stop them while their generation workers are active.

## Launch

Run on the H100 server from the repository root:

```bash
cd /opt/home/gabriel/dev/fastqwen
export BULK="fr-code-bulk-$(date +%Y%m%d-%H%M%S)"
export BULK_DIR="$PWD/data/runs/$BULK"
tmux new-session -d -s "$BULK" \
  "bash scripts/run_french_code_bulk.sh data/runs/fr-code-smoke-20261103-214906 '$BULK_DIR' > '$BULK_DIR-launch.log' 2>&1"
printf 'Session: %s\nOutputs: %s\nLauncher log: %s-launch.log\n' "$BULK" "$BULK_DIR" "$BULK_DIR"
```

Use this only for a new run; do not launch a second copy while bulk workers are
already active. The launcher validates source hashes, smoke configuration, and
live model endpoints before spending GPU time. It refuses an existing output
directory. It then launches two workers: each generates its French partition,
then its code partition immediately, without waiting for the other replica.
An error stops that replica's queue; the other replica can finish its own work.

Each source/replica has a separate directory and manifest. Settings match the
smoke: 8,192-token cap, temperature 0, seed 0, thinking disabled, effort low,
concurrency 64, timeout 900 seconds. Held-out and smoke files are not inputs.
Never merge the overlapping smoke answers into these bulk training outputs.

## Expected workload

Measured mean smoke completion lengths were 404.6 French tokens and 1,062.8
code tokens. This implies approximately 16.7M French and 53.1M code completion
tokens for the prepared inputs, or 69.8M total.

The short smokes achieved about 2,182 and 2,064 combined completion tokens/s.
Direct wall-time extrapolation is 2.12 h French plus 7.15 h code, approximately
9.3 h total. Those small runs include prefill and drain effects, including a
long capped code answer. Bulk throughput may be higher; last night's English
wave achieved 6,740 tokens/s. Do not promise a nine-hour duration or constant
GPU saturation. Use progress logs for the live estimate.

## Monitor

In the shell where the variables above are defined:

```bash
tail -n 30 "$BULK_DIR-launch.log"
tail -f "$BULK_DIR/fr-0.log" "$BULK_DIR/fr-1.log"
```

Once code starts, use the corresponding `code-0.log` and `code-1.log`.
Logs print progress every 50 completions. Watch the GPU activity separately:

```bash
watch -n 5 nvidia-smi
```

SSH disconnection does not stop the tmux job or detached Docker containers.
The tmux session exits when the script completes; output logs remain available.
Completion is recorded in `$BULK_DIR/exit-status.txt`; success is `0 0`, followed
by output integrity and QA checks. Containers are not automatically stopped.

If preflight fails before the directory exists, read the launcher log and fix
the reported issue. If a generation worker fails, preserve its partial outputs
and rerun that individual generator command with identical input, settings,
shard index, and run ID. Do not rerun the whole launcher against an existing
directory. Inspect a partial JSONL tail before resuming an abruptly killed worker.

Store the QA exporter and trainer code in Git separately. While GPUs generate,
the next development task is the QA-cleaned English export and SFT harness;
do not start training on these GPUs until generation workers have released them.