# Overnight S1 run analysis: November 1-2, 2026

Run directory: `data/runs/s1-night-20261101-222544`.
Analysis date: November 3, 2026.

## Summary

Generation completed successfully on two TP=2 teacher replicas using all four
H100 GPUs. It produced 110,497 answers and 94,227,430 completion tokens. The
outputs are intact, but require quality filtering before training. This is an
exploratory English S1 wave, not a verified, balanced training mixture.

## Generation results

| Metric | Replica 0 | Replica 1 | Combined |
| --- | ---: | ---: | ---: |
| Answers | 55,257 | 55,240 | 110,497 |
| Prompt tokens | 11,526,310 | 11,491,533 | 23,017,843 |
| Completion tokens | 47,052,016 | 47,175,414 | 94,227,430 |
| Completion tokens/s | 3,365 | 3,407 | Approximately 6,740 |
| Answers reaching the cap | 10 | 23 | 33 |
| Failed requests | 0 | 0 | 0 |
| Empty answers | 0 | 0 | 0 |

Combined throughput uses the wall time from the first worker's start to the
last worker's finish, rather than summing rates measured over different intervals.
Generation lasted approximately 3 hours 53 minutes, finishing around 03:15 CET
on November 2. Both replicas performed similarly. There is no GPU utilization
time series to establish continuous saturation or utilization after completion.

Completion lengths: mean 852.76 tokens, median 816, p90 1,712, p95 1,986, and
p99 2,543. Only 33 answers reached the 8,192-token cap: approximately 0.030%.

## Integrity and exit status

Both complete output files were checked:

- Every accepted S1 prompt has exactly one generated answer.
- No missing, unexpected, or duplicate IDs; no changed prompts.
- Both JSONL files parse completely and end with a newline.
- Output SHA-256 hashes and recomputed token totals match the manifests.

| Output | SHA-256 |
| --- | --- |
| `replica-0/shard-00-of-02.jsonl` | `19da04c24b3928a468ce7966cb5d0e7ef725c003d23e51ae820b1d1d6f801965` |
| `replica-1/shard-01-of-02.jsonl` | `1f8ca3bd9274ab1df268d461c16ce8f17c2813fc5993ae483a100265f71677ac` |

The recorded worker exit statuses are `1 1`, but neither generation failed.
Both workers saved their outputs and manifests, then raised `ValueError` in
`scripts/generate_dataset.py` while printing
`manifest_path.relative_to(ROOT)`: the supplied output path was relative while
`ROOT` was absolute. No regeneration is needed. Fix the final path-printing
operation before the next run.

## Preparation and classification

The sampler scanned 207,865 source records and selected 180,000 candidates after
regex filtering and deduplication. The 8,000-character limit removed 184, leaving
179,816 candidates for classification.

| Export | Prompts |
| --- | ---: |
| Accepted S1 | 110,497 (61.5%) |
| Accepted S2 | 3,387 |
| Review | 65,932 |

Classification took 2,934.77 seconds, approximately 49 minutes or 61.27 prompts/s.
Before suitability and confidence export gates, the route counts were 143,552
S1, 4,334 S2, 31,866 UNCERTAIN, and 64 OTHER. Suitability labels were 162,812
suitable, 15,482 oversized, and 1,522 incomplete. These are classifier decisions,
not measured ground-truth accuracy. The S2 and review exports were not generated
in this wave.

## Frozen configuration

- Teacher: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- vLLM image digest: `sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f`.
- Runtime workaround: uninstall `torchcodec` inside the container before serving;
  this modification is not captured by the base image digest alone.
- Two TP=2 replicas, concurrency 64 per replica, timeout 900 seconds.
- Temperature 0, top-p 1, seed 0, thinking disabled, effort low, max tokens 8,192.
- Chat template SHA-256: `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
- Exclusion manifest: 290 normalized hashes; SHA-256
  `8c6780c2632021af018d567017b711e19a31ca8f61c03660c86484b14db4b677`.
- Classifier: `LiquidAI/d1-2B`, revision `10e7671d44e4b37700020b1eed3f594c5f2af0b4`.
- Classifier policy SHA-256: `14b7b5498d29c6a084ac67159fc2f0d630777a123e57776a8a98d0666bc33da1`.
- Minimum confidence 0.8 and margin 0.2, unchanged for this run.

## Quality risks

Repetition is the main observed defect. A basic scan flagged 188 answers using
repeated lines of at least 30 characters occurring at least 10 times, or a
non-whitespace character repeated at least 80 times. Some flags are harmless
repeated list labels.

A stronger scan combined long character runs with 1-12-word sequences repeated
at least 20 times in the final 6,000 characters. It flagged 71 answers, including
56 that stopped normally. These heuristic sets overlap and must not be added
together or treated as confirmed defect counts.

Inspected examples include repeated names, repeated self-corrections, and
phrase loops in answers about sports, music, cafes, and scientific institutions.
Filtering only `finish_reason == "length"` would miss defective training targets.
Some legitimate recipes or structured lists also match repetition heuristics.

There is residual S1/S2 boundary leakage: 84 answers contain fences labelled with
technical languages. Examples concern Z3, database tuning, responsive layouts,
and the Canvas API. This is a review signal, not a complete leakage measurement
or an automatic rejection rule; general explanations can legitimately contain
code. Factual accuracy has not been systematically evaluated.

## Recommended disposition

1. Preserve the raw outputs, candidates, classifier audit, logs, and manifests.
2. Make separate QA-cleaned exports; exclude the 33 capped answers, review
   repetition flags, and inspect technical boundary cases.
3. Review a broader random sample for factual accuracy and prompt/answer fit.
4. Do not merge overlapping smoke outputs into the large-wave training export.
5. Complete held-out acceptance gates and EN/FR mixture balancing before treating
   this as a finalized training dataset. The regex, character, and suitability
   filters bias this wave toward shorter answers.

Related procedures: [overnight runbook](overnight-h100.md),
[decision model](decision-model.md), and [backup procedure](backup-to-spark.md).