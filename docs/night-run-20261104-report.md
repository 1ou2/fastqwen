# November 4 report: QA, SFT implementation, and overnight launch

Status observations began at **2026-11-04 22:00:31 CET** on the H100 cluster.
The log, manifest, session, and GPU readings were collected separately around
that time, not as an atomic snapshot. This is an evening progress report, not a
claim that the overnight pilot has completed.

## Summary

- English, French, and code teacher generation is complete: 201,694 raw answers.
- Automated QA and deterministic training/validation exports are complete.
- A single-GPU, text-only SFT trainer is implemented and real checkpoint
  save/resume smokes passed after repairing checkpoint key remapping.
- All 178 repository tests passed before launch.
- A bounded EN/FR/code SFT pilot was launched with approval at **21:58 CET** in
  detached tmux. At observation time it was preparing data on CPU, not yet
  performing GPU optimizer updates.
- No full-epoch draft, measured acceptance result, or final Spark speedup exists
  from this work yet.

## Completed data work

The two completed teacher waves supplied 110,497 English S1, 41,197 French S1,
and 50,000 code S2 answers. The QA exporter consumed only the six bulk shards;
overlapping smoke outputs were not added as extra training targets.

| Group | Training rows | Validation rows | Quarantined | Rejected |
| --- | ---: | ---: | ---: | ---: |
| English S1 | 107,896 | 2,271 | 297 | 33 |
| French S1 | 40,052 | 892 | 245 | 8 |
| Code S2 | 48,043 | 1,018 | 652 | 287 |
| Total | **195,991** | **4,181** | **1,194** | **328** |

The training export contains **159,586,897 teacher completion tokens**.
All rejected records hit the generation cap. Quarantine covers conservative
repetition, technical S1/code boundary, and template-marker flags; it is not a
declaration that every flagged answer is bad.

Raw records were preserved. Independent full-data checks verified output/input
hashes, record accounting, target preservation, unique accepted IDs/prompts,
zero train/validation normalized-prompt overlap, and zero accepted overlap with
the 290 frozen exclusions or 1,000 new source hold-outs. The new hash-based
training validation split did not rewrite those evaluation sets.

Passing QA does not establish factual/code correctness, semantic near-duplicate
exclusion, or licensing clearance. Details and input provenance are in the
[QA report](training-data-qa.md) and
[export manifest](../data/runs/qa-export-20261104-v1/manifest.json).

## Training implementation

The main [requirements file](../requirements.txt) now includes PyTorch,
Transformers, and Accelerate. The user installed these in `.venv`;
`.venv-decision` remains dedicated to the classifier.

The verified training runtime is PyTorch **2.15.0+cu132**, Transformers
**5.22.1**, Accelerate **1.15.0**, datasets **5.1.0**, and CUDA **13.2**.
CUDA and BF16 forward/backward/optimizer operations worked on an H100 NVL.

Implemented files:

- [Serialization helpers](../qwenbench/training.py): extract token IDs from the
  installed tokenizer's structured result; assert exact generation-prefix
  matching; mask user/template-prefix labels; preserve assistant/end-token
  targets; mask padding; skip overlength examples without truncating targets.
- [Trainer](../scripts/train_draft.py): verify QA hashes and pinned tokenizer/
  template files; cache and hash tokenized datasets; train the text decoder;
  checkpoint optimizer/scheduler/RNG state; require matching configuration and
  source hashes for resume; reject unvalidated distributed launches; stop on
  logged non-finite loss/gradient metrics; export a final model/tokenizer.
- [Night launcher](../scripts/run_sft_night.sh): refuse existing output paths,
  check free GPU/disk capacity, write a persistent log and exit-status file, and
  leave container management explicit.
- [Tests](../tests/test_training.py): label boundaries, padding, overlength and
  invalid records, strict weight restoration, CLI parsing, and distributed
  refusal. SFT plus QA tests numbered 41; the complete repository suite numbered
  178, with the classifier test's optional FastAPI dependency supplied separately.

Trainer, helper, and QA policy source snapshots are archived inside the pilot
run and included in its hashed configuration. This preserves the implementation
with the run artifacts through the existing Spark backup.

## GPU validation and fixes

Both English-only smokes used 64 training and 8 validation examples, first at a
512-token cap and then at a 2,048-token cap with accumulation 4. Each trained to
step 2, saved a checkpoint, and resumed to the original planned step 4. The
larger smoke included training sequences up to **1,911 tokens**; it does not prove
every possible maximum-length or mixed-source input has been exercised.

Compatibility issues found and repaired before accepting the smoke gate:

- The tokenizer returns a structured encoding, not the plain list assumed by
  older examples.
- This Transformers version requires explicit `warmup_steps`; `warmup_ratio`
  caused a pre-training failure.
- Loader diagnostics contain sets and needed deterministic JSON conversion.
- Saved text checkpoints use `model.language_model.*` keys, while the active
  text model uses `model.*`. Trainer's direct resume load produced widespread
  key warnings. That apparent step-4 completion was **not** accepted as proof of
  correct resume. The repaired path loads through `from_pretrained` for key
  remapping, checks loader diagnostics, and restores weights strictly in place.

The final repaired smokes had finite losses. The larger smoke's final validation
loss was approximately **0.9600** over eight examples. These are plumbing checks,
not evidence of improved draft acceptance or general quality.

Smoke artifacts are under `data/runs/sft-smoke-20261104/` and
`data/runs/sft-smoke-2048-20261104/`. They are distinct from the overnight pilot.

## What was launched

| Setting | Approved pilot |
| --- | --- |
| Session | `sft-pilot-20261104` |
| Launch | November 4, 21:58 CET / 20:58 UTC |
| Student | Qwen/Qwen3.5-2B, revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Teacher provenance | Qwen/Qwen3.8-27B, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Training device | GPU 2, one H100 NVL |
| Sources | QA-accepted English S1, French S1, code S2 |
| Mixture | Natural row proportions among length-eligible records; no balancing |
| Sequence cap | 2,048 tokens; overlength records skipped, not truncated |
| Optimizer-update cap | 2,000 |
| Microbatch / accumulation | 1 / 4; roughly 8,000 training examples processed |
| Learning rate / seed | `1e-5` / `20261104` |
| Save / retention | Every 100 updates; up to three checkpoints |
| Validation | Every 200 updates; up to 64 retained examples per source group |
| Output | `data/runs/sft-pilot-20261104/` |
| Log | `data/runs/sft-pilot-20261104.log` |
| Process result | `data/runs/sft-pilot-20261104.exit-status.txt`, written on exit |

This is a **bounded exploratory pilot**, not a full epoch over the QA corpus.
All requested source groups are prepared before the training sampler runs; the
2,000-update cap does not limit preprocessing to 8,000 records. Length filtering
can change the mixture relative to the original QA export.

Launch command, already executed; do not launch a duplicate:

```bash
tmux new-session -d -s sft-pilot-20261104 \
  bash scripts/run_sft_night.sh data/runs/sft-pilot-20261104
```

## Current status

The tmux session was verified alive. The
[pilot manifest](../data/runs/sft-pilot-20261104/training-manifest.json) reported
`preparing`. The latest sampled progress line was:

```text
Generating train split: 37156 examples [02:23, 237.46 examples/s]
```

This is the dataset library constructing tokenized training examples, **not
teacher generation, optimizer steps, or completed epochs**. Final retained counts
and token budgets remain pending until preparation completes. No error was
observed in the inspected progress output. No process exit-status file, pilot
checkpoint directory, or pilot final model export existed at the status check.

| GPU | Memory used, MiB | Utilization at sampled instant | Role |
| --- | ---: | ---: | --- |
| 0 | 89,081 | 0% | Teacher replica 0 remains loaded |
| 1 | 88,999 | 0% | Teacher replica 0 remains loaded |
| 2 | 0 | 0% | Reserved for pilot after CPU preparation |
| 3 | 0 | 0% | Free; not used by this single-GPU pilot |

Teacher replica 1 was stopped with approval only after worker and live request
checks showed no active/queued generation. Replica 0 and monitoring services were
left untouched. GPU inactivity during CPU preparation is expected; this snapshot
does not establish a hang or continuous inactivity.

## Overnight handoff and remaining gates

Leave the tmux run and frozen inputs/source files unchanged. It will move from
CPU preparation into GPU training automatically. Monitor with:

```bash
tail -n 40 data/runs/sft-pilot-20261104.log
```

In the morning, check the process exit status, manifest status, final retained
source counts, finite training/validation metrics, and checkpoint/final export
availability. A successful bounded run should reach step 2,000; a session
disappearing or GPU utilization falling is not sufficient evidence of success.
Use the [resume instructions](sft-night-20261104.md) for an interrupted run; do
not restart the fresh-output launcher against existing artifacts.

No reliable pilot finish time has been measured. The tiny smokes included startup,
evaluation, and checkpoint IO on every update. Re-estimate GPU runtime from the
first 50-100 pilot updates, separately from CPU preparation. Hybrid-attention
training currently uses correct but slower PyTorch fallbacks because
`causal_conv1d` and `flash-linear-attention` are absent. Do not change native
kernels/packages mid-run; optimized and distributed stacks need separate smokes.

Remaining work includes pilot assessment, explicit mixture/length-policy review,
larger training if warranted, validated tool data, offline raw/trained draft
acceptance and gamma sweeps, quantization/calibration, and final Spark quality,
speed, memory, and speculative-exactness comparisons. No acceptance or headline
speed claim follows from a saved checkpoint alone.

The [Spark backup script](../scripts/backup_to_spark.sh) covers run artifacts,
source snapshots, logs, checkpoints, and this report. Actual durable transfer
remains unverified here; verify the final backup before releasing the cluster.