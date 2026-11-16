# Overnight H100 artifact runs: November 15, 2026

## Completed results and conclusions: November 16, 2026

All three jobs finished successfully: both quantization stages, their overall
chain, and the SFT launcher recorded exit code **0**. The two export manifests
and the training manifest have `status: completed`. The launch observations
below are historical, not the current job state.

### Target quantization

| Recipe | Verified NVFP4 modules | Checkpoint disk size | Finished (CET) |
|---|---:|---:|---|
| [Mixed](../data/runs/nvfp4-mixed-20261115/manifest.json) | 256 | 28,574,978,139 bytes (28.57 GB / 26.61 GiB) | November 15, 22:03:32 |
| [MLP-only](../data/runs/nvfp4-mlp-only-20261115/manifest.json) | 192 | 30,986,660,543 bytes (30.99 GB / 28.86 GiB) | November 15, 22:08:49 |

Both used the same 512-sequence, 421,921-token mixed-domain calibration candidate.
The exporter passed original weight coverage, exact preservation of MTP/vision/
Gated DeltaNet weights, normalized configuration, NVFP4 layout/metadata, and
original source-hash checks. All 15 native MTP tensors were preserved.

Mixed saves approximately **2.41 GB** on disk versus MLP-only. These are partial
quantization recipes; disk size is not a measurement of serving memory or speed.
Neither export has yet been quality-qualified or validated for native NVFP4
serving and MTP operation on Spark/GB10. The calibration candidate still contains
only about 40% of the token volume described in the project plan.

### Longer draft SFT

The [training manifest](../data/runs/sft-long-20261115/training-manifest.json)
records **10,000 optimizer updates**, approximately 40,000 example presentations,
and **0.2182 epochs**. This was a fresh run, not a pilot resume.
The [final model](../data/runs/sft-long-20261115/final/) is saved; its weight file
contains 3,763,692,048 bytes (approximately 3.76 GB).

| Metric | [2,000-step pilot](../data/runs/sft-pilot-20261104/training-manifest.json) | Overnight 10,000-step run |
|---|---:|---:|
| Final validation loss (lower is better) | 0.664089 | **0.637361** |
| Average training loss | 0.777031 | 0.757809 |
| Trainer-reported training runtime | 4,155.6 seconds (1h 09m 15.6s) | 19,583.5 seconds (5h 26m 23.5s) |

Final validation loss decreased **4.02%** relative to the pilot on the same
192 validation examples: 64 English, 64 French, and 64 code. This aggregate
teacher-answer loss is not a factual-quality score, a per-domain improvement
result, or a speculative acceptance measurement.

The [validation history](../data/runs/sft-long-20261115/trainer_state.json)
shows diminishing returns:

| Update | Validation loss |
|---|---:|
| 2,000 | 0.662570 |
| 4,000 | 0.641222 |
| 6,000 | 0.637086 |
| 6,600 (best observed) | **0.636217** |
| 10,000 (final) | 0.637361 |

Improvement largely plateaued around 6,000 updates. The small rise from the best
observed loss to the final loss is not strong evidence of overfitting on this
small monitored set. Best-checkpoint selection was not enabled; step 6,600 was
not retained. The remaining resumable checkpoints are 9,800, 9,900, and 10,000,
alongside the final model. The cosine learning-rate schedule also changes with
the update budget, so this is not simply a continuation of the pilot trajectory.

### Measured time spent by phase

Times below use manifest timestamps and recorded runtimes, not the original
launch estimates. All clock times are **CET (UTC+1)**; durations are rounded to
one decimal second. Quantization and SFT ran concurrently on different GPUs,
so their durations must not be added to obtain elapsed overnight wall time.

| Phase | Time window (CET) | Duration | Measurement |
|---|---|---:|---|
| Mixed target loading, calibration, export, and verification | November 15, 21:58:15.8 to 22:03:32.2 | **5m 16.4s** | Export manifest `runtime_seconds` |
| Gap between quantization stages | November 15, 22:03:32.2 to 22:03:39.2 | 7.0s | Difference between stage timestamps |
| MLP-only target loading, calibration, export, and verification | November 15, 22:03:39.2 to 22:08:49.0 | **5m 09.8s** | Export manifest `runtime_seconds` |
| Both quantization stages, first stage start to last stage finish | November 15, 21:58:15.8 to 22:08:49.0 | **10m 33.2s** | Manifest wall-clock interval, including stage gap |
| SFT preparation and initialization before training | November 15, 21:58:11.4 to 22:09:12.2 | **11m 00.8s** | Training manifest `started_at` to `last_launch_at` |
| SFT training loop, including periodic validation and checkpointing | Starts November 15, 22:09:12.2 | **5h 26m 23.5s** | Trainer `train_runtime` |
| Final post-training validation | After training | **13.6s** | Final `eval_runtime`; additional to the training loop |
| Remaining post-launch overhead outside training and final validation | Not separately timestamped | Approximately 2.3s | Residual, not an independently measured save duration |
| Complete SFT run, preparation through final artifacts | November 15, 21:58:11.4 to November 16, 03:35:51.6 | **5h 37m 40.2s** | Training manifest `started_at` to `updated_at` |

SFT preparation includes tokenization, data checks, model loading, and trainer
initialization; CPU tokenization alone was not separately timed. Likewise, the
quantization manifests do not separate loading, calibration, export, hashing,
and verification durations. Periodic SFT validation/checkpoint costs are included
in the training-loop runtime and have no separate aggregate timing here.
The approximately 2.3-second residual includes final saving/state bookkeeping
and timing-boundary overhead; it should not be interpreted as an exact export
benchmark. Both quantization runs took about five minutes rather than the
estimated one to two hours per recipe.

### Overall conclusion and next gates

**Successful artifact production, a modest draft validation-loss improvement,
but no demonstrated deployment speedup yet.** Additional training should not be
the automatic next step given the observed plateau.

1. Compare the new draft against the original draft and pilot on frozen
   per-domain diagnostics and teacher-trace/speculative acceptance evaluations.
   Earlier comparison results evaluated the pilot, not this new final model.
2. Validate both quantized targets on Spark/GB10: native loading, MTP engagement,
   quality versus the unquantized target, serving memory, and end-to-end speed.
3. Rerun the Spark backup after completion and require `Verified backup`, plus
   an additional durable copy, before releasing the H100 host. A completed final
   backup was not verified as part of this results analysis.

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
