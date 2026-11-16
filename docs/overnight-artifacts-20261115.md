# Overnight H100 artifact runs: November 15, 2026

## Verified launch

With user approval, both jobs were launched in separate detached tmux sessions
at **21:58:10 CET / 20:58:10 UTC**. They survive SSH disconnection; no teacher
container was stopped and neither Python environment was modified.

| Job | GPU | tmux session | Verified initial state |
|---|---|---|---|
| Mixed NVFP4 target export | 2 | `nvfp4-night-20261115` | Export manifest `running` |
| Subsequent `mlp-only` target export | 2, sequentially | Same session | Queued; starts only if mixed exits successfully |
| Fresh 10,000-update draft SFT | 3 | `sft-long-20261115` | Manifest `preparing`; CPU data preparation progressing |

Both sessions were independently checked alive at 21:58 CET. The initial SFT
check showed more than 3,000 processed rows and a growing log, not GPU training
or a saved checkpoint yet. The existing teacher remains on GPUs 0-1. Assignment
to GPU 3 is not proof of utilization while CPU preparation is running.

A follow-up startup check observed **calibration 157/512** on GPU 2, approximately
54,457 MiB allocated and 67% GPU utilization. SFT preparation had advanced to
**25,070 processed rows** with its worker still alive; GPU 3 had not begun training
at that check. The existing teacher endpoint remained responsive. These are
startup observations, not completed model results.

The [launch record](../data/runs/nvfp4-night-20261115/launch-record.json) records
session creation times, GPU assignments, output paths, and launch settings.
Estimates remain 1-2 hours per quantization recipe and roughly 6-8 hours plus
preparation for SFT; these are not measured completion promises.

## Quantization

The [archived launcher](../data/runs/nvfp4-night-20261115/launch.sh) exposes only
GPU 2, runs mixed first, and starts `mlp-only` only after a successful mixed exit.
It refuses existing output/log/status paths. A logging failure is treated as a
failed stage, and each stage records its exit code.

Both recipes passed CPU/input preflight before launch. They use the same frozen
[calibration candidate](../data/runs/calibration-nvfp4-20261115/manifest.json):
512 mixed-domain sequences, 421,921 tokens, capped at 2,048 per sequence.
This is approximately 40% of the full-length token volume in the project plan.
These are exploratory candidate exports, not quality-qualified deployment models.

| Recipe | Output directory | Sibling log and exit status |
|---|---|---|
| Mixed | `data/runs/nvfp4-mixed-20261115/` | `nvfp4-mixed-20261115.log`, `nvfp4-mixed-20261115.exit-status.txt` |
| MLP-only | `data/runs/nvfp4-mlp-only-20261115/` | `nvfp4-mlp-only-20261115.log`, `nvfp4-mlp-only-20261115.exit-status.txt` |

Paths in the last column are under `data/runs/`. The overall chain exit status is
`data/runs/nvfp4-night-20261115/exit-status.txt`. Queued paths/status files do not
exist until their stage starts/finishes.

The usable `checkpoint/` is written during final export, not incrementally during
calibration. Successful completion requires exit zero and a manifest with
`status: completed`; a live session or partially written directory is not success.
The exporter verifies original weight coverage, exact MTP/vision/Gated DeltaNet
preservation, packed NVFP4 shapes/scales, and source hashes. Failed directories
are preserved and require a new output path to retry.

## Longer SFT

The existing [SFT launcher](../scripts/run_sft_night.sh) was used unchanged:

```bash
tmux new-session -d -s sft-long-20261115 \
  -c /opt/home/gabriel/dev/fastqwen \
  'env GPU_INDEX=3 MAX_STEPS=10000 WORLD_SIZE=1 bash scripts/run_sft_night.sh data/runs/sft-long-20261115'
```

This is a **fresh run from the pinned original Qwen3.5-2B draft**, not a resume of
the completed pilot. The pilot and its final model are preserved.

- Input: existing QA export, English/French/code groups, all length-eligible rows.
- Maximum length 2,048; overlength targets are skipped, not truncated.
- Batch size 1, accumulation 4, learning rate `1e-5`, seed `20261104`.
- Budget: 10,000 optimizer updates, roughly 40,000 example presentations.
- Checkpoints every 100 updates, retaining up to three; validation every 200
  updates with up to 64 eligible examples per group.
- No distributed training, tool/reasoning SFT, or sequence packing.
- Main `.venv`; quantization independently uses `.venv-quant`.

Outputs are under `data/runs/sft-long-20261115/`, with
`training-manifest.json`, periodic `checkpoint-*`, and `final/` on success.
The sibling log is `data/runs/sft-long-20261115.log`; the launcher writes
`data/runs/sft-long-20261115.exit-status.txt` when it exits.

CPU preparation automatically precedes training. The initial `preparing` status
means the job started, not that GPU updates have already occurred. A complete
checkpoint can resume an interrupted run using the identical configuration and
output directory; do not rerun the fresh-run launcher against that directory.
More training does not guarantee improved draft acceptance or Spark speed.

## Monitor and detach

```bash
tmux attach -t nvfp4-night-20261115
tmux attach -t sft-long-20261115

tail -n 30 data/runs/nvfp4-mixed-20261115.log
tail -n 30 data/runs/sft-long-20261115.log
```

Detach with **Ctrl+B, then D**, not Ctrl+C. Finished tmux sessions may disappear;
use the logs, exit-status files, and manifests to distinguish completion from
failure. Do not launch duplicates while these sessions are active.

## Backup and remaining work

All run outputs, logs, status files, launch metadata, and archived quantization
launcher are in the existing Spark backup scope. Application code and dependency
files still require Git separately.

An interim backup while these jobs write is not a final snapshot. After the jobs
finish and documentation is updated, rerun the Spark backup and require
`Verified backup` before releasing the H100 host. Keep an additional durable copy.

Spark native NVFP4/vLLM/MTP compatibility, quantized quality, speculative
acceptance, and end-to-end speed/memory remain unverified. See the
[NVFP4 runbook](nvfp4-quantization.md) and
[backup runbook](backup-to-spark.md) for the existing safeguards and limitations.
