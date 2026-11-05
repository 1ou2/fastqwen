# Overnight long-horizon comparison: November 5-6

## Purpose and authorization

The user authorized this bounded overnight comparison on November 5, after the
completed 128-token experiment. This document is written before launch. It
specifies the intended job; actual progress and completion are recorded in the
run artifacts described below.

Launch was confirmed at 22:12 CET on November 5 in tmux session
`draft-compare-long-20261105`. The initial live process and log confirmed 512
teacher tokens, 300 prompts, a 2,048-token context cap, and all four gamma values.
At that check, preparation was performing provenance and candidate checks; this
launch record does not claim that reference generation or student evaluation has
completed.

The [earlier comparison](draft-comparison-20261105.md) found higher trained-draft
acceptance across English, French, and code. However, 257 of 300 teacher traces
reached the 128-token cap. Tonight's job tests whether the trained draft retains
its advantage over a longer answer horizon, without changing its weights.

## Fixed configuration

| Setting | Value |
| --- | --- |
| Run directory | `data/runs/draft-compare-long-20261105` |
| tmux session | `draft-compare-long-20261105` |
| Teacher | Pinned Qwen3.8-27B, existing replica on GPUs 0-1 |
| Raw student | Pinned original Qwen3.5-2B, GPU 2 |
| Trained student | Completed 2,000-update pilot final export, GPU 3 |
| Fresh acceptance prompts | 100 English, 100 French, 100 code |
| Maximum teacher trace | 512 tokens, or earlier EOS |
| Maximum prompt plus trace | 2,048 tokens |
| Gamma values | 1, 2, 4, 8 |
| Monitored diagnostic | Same 64 examples/source, 192 total |
| Selection seed | 20261105 |
| Uncertainty | 2,000 paired, source-stratified prompt bootstrap replicates |
| Student numerics | BF16, SDPA, greedy generation |
| Environment | Main `.venv`, not classifier-only `.venv-decision` |

The teacher reference stage runs first. The raw and trained evaluations then run
concurrently on separate GPUs. The paired summary runs automatically only if both
workers succeed. No training, quantization, package installation, or additional
queued experiment is part of this job.

## Method and interpretation

The existing [comparison harness](../scripts/compare_drafts.py) verifies model,
tokenizer, template, data, and implementation provenance. It freezes exact teacher
prompt/completion token IDs, scores teacher-trace agreement and NLL, and replays
greedy draft blocks against those references. It separately counts accepted draft
tokens, teacher corrections, teacher bonuses, and verification cycles.

QA-training prompts and all 192 monitored-validation prompts remain excluded from
acceptance selection using normalized prompt hashes. Candidate sources and
selection rules are unchanged from the earlier experiment. The 512-token horizon
leaves at most 1,536 prompt tokens, so the selected cohort may differ from the
128-token experiment. Prompts can also overlap the earlier comparison: this is
not a wholly independent blind replication. Interpret raw versus trained within
this run; do not attribute cross-run metric differences solely to horizon length.

This remains offline finite-horizon replay, not native speculative serving.
Report capped versus EOS-ended traces again. Replay output identity is by
construction, not a native-engine exactness gate. Acceptance, NLL, and verification
counts do not establish wall-clock speedup, full-answer quality, tool-use quality,
or Spark deployment performance. No gamma is declared optimal from replay alone.

Estimated duration is provisionally 4-8 hours based on the shorter completed run.
Longer prefixes and reference PyTorch kernels make this uncertain; the job has no
automatic eight-hour cutoff. It stops after this finite workload and summary.
The existing teacher container stays loaded. The backup process, earlier results,
and trained checkpoint are not stopped or overwritten. Failed partial runs are
preserved and require a fresh directory rather than automatic restart.

## Launch and monitoring

From the repository root, launch once using the
[guarded launcher](../scripts/run_draft_comparison.sh):

```bash
tmux new-session -d -s draft-compare-long-20261105 \
  bash scripts/run_draft_comparison.sh data/runs/draft-compare-long-20261105 512
```

The launcher refuses existing output/log/status paths and requires free student
GPU memory and disk space. Its optional second argument is the teacher-token
horizon; omitting it preserves the original 128-token default.

```bash
tail -n 10 data/runs/draft-compare-long-20261105.log
tail -n 5 data/runs/draft-compare-long-20261105/raw.log
tail -n 5 data/runs/draft-compare-long-20261105/trained.log
cat data/runs/draft-compare-long-20261105.exit-status.txt
cat data/runs/draft-compare-long-20261105/workers.exit-status.txt
```

Worker logs appear after reference preparation. Successful whole-run completion
requires launcher exit status `0`, worker statuses `raw=0 trained=0`, completed
arm manifests, and `summary.json` with `status: completed`. The shared preparation
manifest alone is not a completion signal for the whole experiment. Missing final
status files alone do not prove that a process is still alive; inspect logs and
the tmux session if progress stops.

Tomorrow, compare paired per-source acceptance changes and confidence intervals,
teacher-token agreement, NLL, cycle counts, and cap frequency. Retain the current
checkpoint regardless of the outcome. Native integration and independent Spark
correctness, quality, and speed measurements remain the next deployment gates.