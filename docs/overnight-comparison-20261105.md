# Overnight long-horizon comparison: November 5-6

## Completed results: November 6

The comparison **completed successfully at 02:20:41 CET on November 6**
(01:20:41 UTC). The launcher exited `0`, worker statuses were
`raw=0 trained=0`, and both arm manifests and the paired summary report
`status: completed`. These records were inspected on November 15.

Authoritative artifacts:

- [Paired summary](../data/runs/draft-compare-long-20261105/summary.json).
- [Shared preparation manifest](../data/runs/draft-compare-long-20261105/manifest.json).
- [Raw arm manifest](../data/runs/draft-compare-long-20261105/raw/manifest.json)
  and [trained arm manifest](../data/runs/draft-compare-long-20261105/trained/manifest.json).
- [Launcher exit status](../data/runs/draft-compare-long-20261105.exit-status.txt)
  and [worker exit statuses](../data/runs/draft-compare-long-20261105/workers.exit-status.txt).

**Conclusion: the unchanged 2,000-update pilot retains its acceptance advantage
over longer teacher traces, across English, French, and code.** This supports
keeping it as a trained-draft candidate. It does not establish native speculative
correctness, deployment speedup, or improved standalone answer quality.

### Pooled acceptance

The completed run evaluated 300 prompts, 100 per source, with a 512-token teacher
horizon and a 2,048-token prompt-plus-trace cap. Both arms used identical frozen
references, BF16, SDPA, and greedy generation with thinking disabled.
Gains and confidence intervals are percentage points, not relative percentages.
Intervals use 2,000 paired, source-stratified prompt bootstrap replicates.

| Gamma | Raw acceptance | Trained acceptance | Gain (pp) | 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 74.95% | 77.57% | +2.61 | +2.30 to +2.95 |
| 2 | 65.53% | 68.68% | +3.15 | +2.82 to +3.48 |
| 4 | 50.97% | 54.65% | +3.68 | +3.32 to +4.04 |
| 8 | 33.89% | 37.33% | +3.44 | +3.08 to +3.81 |

All 12 per-source/gamma acceptance-difference intervals are also above zero.
These are individual intervals, not a simultaneous multiple-comparison guarantee.
Per-source gains are:

| Gamma | English gain (pp) | French gain (pp) | Code gain (pp) |
| --- | ---: | ---: | ---: |
| 1 | +2.40 | +2.70 | +2.74 |
| 2 | +2.76 | +3.17 | +3.54 |
| 4 | +2.96 | +4.04 | +4.17 |
| 8 | +2.61 | +3.28 | +4.78 |

Code has the largest acceptance gain at every tested gamma. The summary contains
the underlying source-level acceptance rates and confidence intervals.

### Verification work and agreement

For the same 111,958 reference output tokens:

| Gamma | Raw cycles | Trained cycles | Cycle reduction | Trained output tokens/cycle |
| --- | ---: | ---: | ---: | ---: |
| 1 | 64,064 | 63,127 | 1.46% | 1.77 |
| 2 | 48,558 | 47,281 | 2.63% | 2.37 |
| 4 | 37,016 | 35,314 | 4.60% | 3.17 |
| 8 | 30,441 | 28,365 | 6.82% | 3.95 |

Gamma 8 saves more verification cycles but also proposes longer blocks that can
be rejected. This accounting does not identify the fastest native-serving gamma.

On the selected fresh teacher traces, both NLL and top-1 agreement improved
in every source:

| Source | Raw token-weighted NLL | Trained token-weighted NLL | Raw top-1 agreement | Trained top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| English | 1.059 | 0.953 | 70.99% | 73.19% |
| French | 1.054 | 0.926 | 72.16% | 74.92% |
| Code | 0.547 | 0.458 | 83.51% | 85.73% |
| Pooled | 0.858 | 0.753 | 76.18% | 78.54% |

Pooled token-weighted NLL fell by approximately 12.3%. The separate 192-example
monitored-validation diagnostic again yielded NLL 0.765 versus 0.667 and target
top-1 accuracy 78.25% versus 80.59%. It reuses the SFT-monitored subset and is not
an independent blind evaluation.

The raw worker reported 13,642.4 seconds of runtime and finished at 02:20:40 CET;
the trained worker reported 13,162.7 seconds and finished at 02:12:40 CET.
The whole job finished approximately four hours nine minutes after the recorded
22:12 CET launch. Worker runtimes include offline scoring and repeated-prefix
generation; they are not native speculative-serving speed measurements.

### Limits and follow-up

Of the 300 traces, **155 (51.7%) reached the 512-token cap** and 145 ended normally.
The earlier 128-token run capped 257 traces (85.7%), but the longer-context
eligibility rule can change the selected cohort. Do not attribute cross-run
metric differences solely to horizon length or treat this as a wholly independent
replication.

Training and all monitored-validation prompts were excluded by normalized exact
hashes, not semantic deduplication. Thinking and tools were excluded. Replay
identity is guaranteed by teacher correction in the offline oracle, not by a
native-engine exactness test. No full-answer, reasoning, tool-use, Spark speed,
or native MTP comparison conclusion follows from these results.

On November 15, the user prioritized further productive H100 work over immediate
native-serving verification. Additional training is therefore a possible
follow-up experiment, not a demonstrated requirement or improvement. Preserve
the current pilot and both comparison runs as controls. Native correctness and
Spark quality/performance checks remain required before deployment claims.
This documentation update launches no job.

## Original purpose and authorization (historical)

The user authorized this bounded overnight comparison on November 5, after the
completed 128-token experiment. The remaining sections preserve the original
launch plan and estimates; the completion update above supersedes their pending
progress descriptions.

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

The command below records the completed launch. Do not rerun it against the
existing output directory; a new experiment requires a fresh directory.

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