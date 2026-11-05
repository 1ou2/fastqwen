# Raw versus trained draft comparison: November 5

The [comparison harness](../scripts/compare_drafts.py) and
[detached launcher](../scripts/run_draft_comparison.sh) compare the pinned original
Qwen3.5-2B with the completed 2,000-update pilot. The live Qwen3.8-27B teacher
stays loaded on GPUs 0-1. No teacher container, training checkpoint, source
dataset, or backup process is stopped or rewritten by this comparison.

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