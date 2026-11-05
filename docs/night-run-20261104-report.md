# November 4-5 report: completed SFT pilot

## Completion update: November 5

The first mixed English/French/code SFT pilot **completed successfully at
23:18:43 CET on November 4**, with process exit code **0**, manifest status
`completed`, and **2,000 optimizer updates**. It did not run through the night:
the bounded job finished about 81 minutes after launch. No further training was
automatically queued.

The November 5 verification, beginning at **20:19 CET**, confirmed the completed
state, final export, and retained checkpoints. All four GPUs reported 0%
utilization at the sampled instant. Teacher replica 0 remained loaded on GPUs
0-1; GPUs 2-3 reported no allocated memory. The pilot tmux session had exited.
No containers or jobs were changed during that check.

### Work completed

- The prior teacher generation and QA work supplied 195,991 accepted training
  rows and 4,181 validation rows, with EN/FR/code source labels and preserved raw
  targets. The full QA accounting is retained in the historical section below.
- The main `.venv` gained the training dependencies. A pinned-template,
  assistant-only, text-decoder SFT harness and guarded detached launcher were
  implemented, with no silent target truncation.
- Real GPU save/resume smokes passed after fixing checkpoint key remapping;
  all 178 repository tests passed before launch.
- The approved single-H100 pilot trained on the prepared mixed-source corpus,
  saved periodic optimizer checkpoints, and exported its first trained draft.

### Measured timing

| Milestone or duration | Recorded result |
| --- | --- |
| Pilot launch | November 4, 21:58:00 CET |
| Training launch recorded | November 4, 22:09:10 CET |
| Final manifest update | November 4, 23:18:43 CET |
| Preparation/startup before training | 11 min 10 sec |
| Trainer-reported runtime | 1 h 9 min 16 sec |
| Total launch-to-final-update time | 1 h 20 min 43 sec |
| Mean optimizer-update time | Approximately 2.08 sec |

The preparation/startup interval includes work before `Trainer.train`, not an
isolated CPU-tokenizer benchmark. Trainer runtime includes periodic validation
and checkpoint writes. Final evaluation/export adds time beyond that runtime.
These are **single-H100 training measurements**, not Spark inference results.

### Retained training data

| Source | Length-eligible rows | Overlength skips | Prepared assistant-label tokens |
| --- | ---: | ---: | ---: |
| English S1 | 101,957 | 5,939 | 78,310,157 |
| French S1 | 39,579 | 473 | 15,601,992 |
| Code S2 | 41,761 | 6,282 | 32,581,763 |
| Total | **183,297** | **12,694** | **126,493,912** |

The prepared corpus contains 156,877,034 input tokens, including masked prompt
tokens. The 2,048-token limit excluded long sequences without truncating answers.
Code lost proportionally more rows than French, so length filtering changed the
source mixture. No balancing, packing, tool trajectories, or reasoning training
was added.

The pilot used microbatch 1, gradient accumulation 4, and 2,000 updates:
approximately **8,000 examples**, or **4.36% of one epoch** over retained rows.
The prepared token totals above describe the available corpus, **not tokens all
consumed during this pilot**. Validation used 192 examples, 64 per source,
selected in source order among length-eligible validation records.

### Results and interpretation

| Optimizer step | Aggregate validation loss |
| --- | ---: |
| 200 | 0.713012 |
| 400 | 0.694058 |
| 800 | 0.673295 |
| 1,200 | 0.664993 |
| 1,400 | 0.664083 |
| 1,600 | 0.663842 |
| 1,800 | 0.663906 |
| 2,000 | **0.664089** |

Mean reported training loss was **0.777031**. Validation loss decreased during
the pilot and largely plateaued after step 1,400; the final checkpoint was not
the lowest-loss logged point. The cosine learning-rate schedule also decayed
toward zero, so the plateau alone does not establish that further training is
useless or that overfitting occurred.

There is **no step-zero/untrained baseline evaluation**, no per-source loss
breakdown, no scored code/factual quality comparison, and no raw-versus-trained
speculative acceptance measurement yet. Step 200 is already a trained model.
The loss change therefore cannot be presented as improvement over the original
student, a passed acceptance gate, or a Spark speedup.

### Artifact verification

The November 5 checks verified process exit code 0, manifest completion, and
global step 2,000; rehashed all recorded tokenized-data and archived-source
files; checked checkpoint state-file presence; and opened the final safetensors
header successfully. This was an integrity check, not an inference-quality test
or a full numerical scan of every saved tensor.

- [Pilot manifest](../data/runs/sft-pilot-20261104/training-manifest.json):
  configuration, pinned revisions, preparation counts, runtime, and final metrics.
- [Trainer state](../data/runs/sft-pilot-20261104/trainer_state.json): optimizer
  progress and logged validation history.
- [Final export](../data/runs/sft-pilot-20261104/final): model, configuration,
  tokenizer, and teacher chat template. The weight file is 3,763,692,048 bytes.
- [Checkpoint 1800](../data/runs/sft-pilot-20261104/checkpoint-1800),
  [checkpoint 1900](../data/runs/sft-pilot-20261104/checkpoint-1900), and
  [checkpoint 2000](../data/runs/sft-pilot-20261104/checkpoint-2000): each has
  nonempty model, optimizer, scheduler, RNG, and trainer-state files.
- [Pilot log](../data/runs/sft-pilot-20261104.log) and
  [process exit status](../data/runs/sft-pilot-20261104.exit-status.txt).

### Next priorities

1. **Verify the durable Spark backup.** Preserve the final model, optimizer
   checkpoints, tokenized data, manifests, logs, and archived implementation
   before releasing the cluster. Backup tooling exists; the actual transfer is
   still unverified here.
2. **Evaluate raw versus trained draft on H100.** Add a shared held-out baseline,
   per-source loss/quality checks, and teacher-token acceptance with a gamma
   sweep. The loaded teacher can use GPUs 0-1 while the student uses GPU 2.
3. **Choose further training from evidence.** Review mixture and length policy
   before scaling. A straight extrapolation from this pilot suggests roughly
   26 hours for one retained-data epoch on one H100 at unchanged settings;
   this is a planning estimate, not a measured full-epoch duration or a reason
   to assume linear four-GPU speedup. Extending the completed job also needs an
   explicit new schedule/configuration, not an unnoticed resume-contract change.
4. **Use remaining cluster access for calibration/export where appropriate.**
   Optimized hybrid-attention kernels, distributed SFT, and quantization/export
   harnesses need separate validation; do not assume they are already ready.
5. **Keep final claims on Spark.** Baseline, FP8/NVFP4, native MTP versus raw and
   trained draft, speculative exactness, quality, memory, and serving comparisons
   remain separate project gates.

The next useful GPU experiment is a **raw-versus-trained evaluation**, not an
unmeasured larger training run. See the [SFT runbook](sft-night-20261104.md) for
the implementation and original launch/recovery procedure.

## Original evening report: historical snapshot

The remainder preserves the launch-time observations from November 4. References
below to preparation, absent pilot checkpoints, or pending completion describe
that earlier snapshot; the completion update above is the current outcome.

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