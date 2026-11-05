# Raw versus trained draft comparison: November 5

The [comparison harness](../scripts/compare_drafts.py) and
[detached launcher](../scripts/run_draft_comparison.sh) compare the pinned original
Qwen3.5-2B with the completed 2,000-update pilot. The live Qwen3.8-27B teacher
stays loaded on GPUs 0-1. No teacher container, training checkpoint, source
dataset, or backup process is stopped or rewritten by this comparison.

## Completed results and conclusion

The full comparison completed on November 5 at 22:04:25 CET. Both student arms
completed, both worker exit codes were zero, and the launcher exited zero.
The authoritative results are in the
[completed summary](../data/runs/draft-compare-20261105/summary.json).

**Conclusion: the 2,000-update pilot is a better draft on this evaluation.**
Acceptance improved across English, French, and code, including fresh prompts
excluded from QA training and the monitored validation subset. Retain the pilot
checkpoint as the trained-draft candidate for native integration. This is evidence
of improved short-horizon teacher alignment, not a measured serving speedup or a
general answer-quality improvement.

### Pooled acceptance

The comparison used 300 prompts, 100 per source, with up to 128 teacher tokens
per prompt. Gains and intervals below are in percentage points, not relative
percentages. Intervals use 2,000 paired, source-stratified prompt bootstrap
replicates.

| Gamma | Raw acceptance | Trained acceptance | Gain (pp) | 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 73.32% | 77.18% | +3.86 | +3.34 to +4.43 |
| 2 | 63.92% | 67.99% | +4.07 | +3.51 to +4.60 |
| 4 | 49.15% | 54.26% | +5.11 | +4.46 to +5.73 |
| 8 | 32.53% | 37.12% | +4.59 | +3.98 to +5.16 |

All 12 reported per-source/gamma acceptance-difference intervals are also above
zero. These are individual intervals, not a simultaneous multiple-comparison
guarantee. At gamma 4, the gains are +3.69 points for English, +5.14 for French,
and +6.77 for code. Code benefits most at every tested gamma.

Acceptance decreases with larger gamma because an early mismatch rejects the
remaining proposed suffix. This does not mean larger gamma necessarily performs
worse: it can still yield more output tokens per verification cycle.

### Verification work and supporting diagnostics

For the same 36,008 reference output tokens, the trained draft reduces the number
of offline verification cycles relative to the raw draft:

| Gamma | Raw cycles | Trained cycles | Cycle reduction | Trained output tokens/cycle |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20,844 | 20,394 | 2.16% | 1.77 |
| 2 | 15,912 | 15,371 | 3.40% | 2.34 |
| 4 | 12,314 | 11,533 | 6.34% | 3.12 |
| 8 | 10,270 | 9,348 | 8.98% | 3.85 |

Gamma 8 saves more verification cycles, but drafts more tokens that may be
rejected. Neither this table nor the replay runtime establishes which gamma has
the best wall-clock performance in a native serving engine.

On fresh teacher traces, pooled token-weighted NLL fell from 0.920 to 0.760,
a 17.4% reduction. Teacher-token top-1 agreement rose from 74.77% to 78.06%.
Both metrics improved in every source. On the separate 192-example monitored
validation diagnostic, token-weighted NLL fell from 0.765 to 0.667 and target
top-1 accuracy rose from 78.25% to 80.59%. The fresh-trace results provide the
stronger evidence that the gain extends beyond the monitored subset.

### Limits and next decision

Of the 300 teacher traces, 257 (85.7%) reached the 128-token cap and 43 ended
normally. The result therefore does not establish full-answer quality, tool-use
performance, or general reasoning quality. Prompt exclusion is based on normalized
exact hashes, not semantic deduplication. Bootstrap uncertainty describes this
selected sample, not all domains. Replay identity is by construction and does
not replace a native speculative-engine exactness test.

**Next priority: native integration and Spark measurements, initially testing
gamma 4 and 8 against the raw draft and the appropriate serving baselines.**
Verify native correctness, end-to-end latency/throughput, and independent quality
before claiming a deployment win. Further SFT is not the immediate priority:
first determine whether the demonstrated acceptance gain outweighs drafting
overhead. The current checkpoint is a supported candidate, not a proven optimal
training endpoint. No additional job is authorized or launched by this conclusion.

## Method and scope

This is **offline greedy teacher-trace replay**, not native vLLM speculative
serving, stochastic rejection sampling, or a Spark speed benchmark. Exact teacher
completion IDs, including terminal EOS when present, are requested through the
live `/v1/completions` endpoint using already-templated prompt IDs. No completion
text is decoded and re-tokenized as a substitute for the original token trace.

Both students use identical token IDs, BF16, SDPA, greedy generation, stopping
rules, horizon, prompt template, and source selections. Thinking is disabled
through the verified template prefix. No tools, reasoning trajectories, packing,
or sample-dependent truncation enter this first comparison.

### Monitored-validation diagnostics

The harness reconstructs the exact 192 cached training-validation examples,
asserting input and label equality, and evaluates both models on those same
targets. The full run uses 64 examples/source. It reports both assistant-token
weighted NLL and mean per-example NLL, plus accuracy against the saved target
tokens, separately by source and pooled.

This subset was monitored during SFT; it is **not blind final evaluation**.
Its target-token accuracy is not presented as an independently obtained teacher
argmax on arbitrary off-trajectory contexts. The comparison supplies the missing
untrained baseline under one explicit scoring implementation; it need not exactly
reproduce Trainer's loss reduction or numerical path.

### Fresh acceptance prompts

The full run freezes 100 prompts/source using seed 20261105 and deterministic
normalized-prompt hash ordering. Candidate pools are:

- English S1: frozen acceptance prompts plus previously unmonitored English
  QA-validation prompts. The frozen English pool alone contains only ten prompts.
- French S1: frozen acceptance prompts plus CFP preparation hold-outs.
- Code S2: frozen acceptance prompts plus Future Code preparation hold-outs.

All QA-training prompt hashes and all 192 monitored-validation prompt hashes
are excluded. Tools, duplicates, and prompts that cannot fit the entire requested
trace within the 2,048-token context cap are excluded before outcome inspection.
Every selected row records its source, normalized prompt hash, and exact prefix.
This is exact normalized-prompt separation, not semantic near-duplicate removal.
Frozen evaluation/preparation files are not rebuilt or modified.

The teacher supplies up to 128 tokens per prompt for the full run. EOS-ended and
length-capped references are counted separately. This measures **finite-horizon
acceptance**, not full-answer quality. Reference generation is repeated once for
the first selected prompt to check deterministic token identity.

### Greedy replay and uncertainty

For each student and gamma 1, 2, 4, and 8, proposals begin from the same corrected
teacher prefix. Accept only the consecutive matching prefix of a proposed block.
At the first mismatch, commit the teacher correction; after a fully accepted
block, count the teacher bonus token separately when the horizon permits it.
Traces terminate at the recorded teacher EOS or horizon.

Report proposed and accepted counts, corrections, bonuses, accepted-prefix length,
and output tokens per verification cycle. Pooled acceptance is the ratio of total
accepted to total proposed tokens, not the mean of per-prompt percentages. A
separate equal-source mean is labelled explicitly.

Raw/trained results are paired by prompt and gamma. The full report uses 2,000
paired bootstrap replicates, resampling prompts within each source group and
reporting percentile 95% intervals for the pooled acceptance-rate difference.
These intervals describe the selected prompt sample, not all possible domains.

Replay output equality with the frozen reference is checked for every prompt,
arm, and gamma. This is **identity by construction of the offline oracle**, not
an independently validated native speculative-engine exactness gate. Student
generation starts from fresh prefixes for each block, so runtime includes
repeated prefill. Neither that time nor output tokens per cycle establishes a
deployment speedup. Native MTP/raw/trained exactness and performance remain Spark
experiments.

## Validation completed before larger launch

The pure accounting/statistics tests cover mismatch boundaries, teacher bonuses,
partial blocks, EOS traces, invalid proposals, pooled ratios, paired bootstrap
reproducibility, and mismatched pairs. CLI parsing and invalid counts are tested.

The real GPU smoke is under `data/runs/draft-compare-smoke-20261105/`: one fresh
prompt/source, one monitored diagnostic/source, 16-token reference horizon, and
all four gamma values. Preparation verified cached-validation reconstruction,
exact teacher prompt/completion IDs, live image/revision, tokenizer/template
compatibility, hold-out hashes, and independent reference regeneration. Original
and trained arms completed on GPUs 2 and 3 with no missing/unexpected weights,
finite losses, and matching replay outputs. The paired summary completed.

All three smoke references were length-capped. One prompt per source cannot
support a quality conclusion; within-source bootstrap intervals are necessarily
degenerate at that smoke size. Mixed signs across gamma values are not interpreted
as evidence for or against the trained draft.

## Full bounded run

The launcher first freezes the shared 300 teacher traces, then evaluates raw and
trained students concurrently on GPUs 2 and 3. It never frees GPUs by stopping
containers. It requires two distinct GPUs with at least 16,000 MiB free each,
10 GiB free disk, and fresh output/log/status paths.

Launch once from the repository root:

```bash
tmux new-session -d -s draft-compare-20261105 \
  bash scripts/run_draft_comparison.sh data/runs/draft-compare-20261105
```

Monitor the preparation/overall log and, once started, both model logs:

```bash
tail -n 25 data/runs/draft-compare-20261105.log
tail -n 20 data/runs/draft-compare-20261105/raw.log
tail -n 20 data/runs/draft-compare-20261105/trained.log
```

The shared `manifest.json` has `stage: prepare`; its completed status means frozen
reference preparation succeeded, **not that the whole comparison finished**.
Each arm has its own manifest. A successful whole run requires both arm manifests
completed, zero worker/launcher exit codes, and `summary.json` present with
completed status. Failed directories may contain useful partial outputs; do not
consume them as complete results or reuse them as a fresh run.

Shared inputs, source snapshots, model-file hashes, teacher deployment details,
per-prompt losses and traces, replay accounting, and summary results live under
`data/runs/`. The existing Spark backup covers these artifacts; the launcher also
archives itself with the other implementation snapshots.

No completion ETA is promised before observing longer-run progress. Reference
PyTorch hybrid-attention kernels remain in use; no package/kernel changes are
made. No SFT extension, quantization export, or tool-use generation is queued.
After the result, decide whether more training is justified from per-source loss
and paired acceptance changes; preserve final quality and speed claims for Spark.