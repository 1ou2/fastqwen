# Overnight French/code run analysis: November 3-4, 2026

Run directory: `data/runs/fr-code-bulk-20261103-220905`.
Analysis date: November 4, 2026. All timestamps below are CET (UTC+01:00).

## Conclusion

The run completed successfully: **91,197 answers and 75,017,801 completion
tokens in 3 hours 5 minutes 35 seconds**, with zero request failures and zero
empty answers. Both replica queues exited with code 0. Output integrity and
complete prompt coverage were independently verified.

The measured aggregate rate of **6,737 completion tokens/s** closely matches the
previous English wave's approximately 6,740 tokens/s. The four-H100, two-TP=2
layout therefore delivered consistent bulk throughput across the two nights.

The outputs are not yet a QA-cleaned training set. In particular, 287 code
answers reached the cap, and repetition, factual accuracy, code correctness,
and the final language/slice mixture still need review. No SFT was queued or run.

## Timing and hardware

- Hardware: four NVIDIA H100 NVL GPUs, two independent TP=2 teacher replicas.
- Launcher started: November 3 at 22:09:07.
- First generation worker started: November 3 at 22:09:08.
- Last French worker finished: November 3 at 22:52:43.
- Last code worker finished: November 4 at 01:14:43.
- End-to-end generation wall time: 11,134.87 seconds, approximately 3.093 hours.
- Four GPUs allocated for that interval correspond to approximately 12.37
  GPU-hours; this excludes preparation, teacher startup, and post-run idle time
  and is not an energy measurement or proof of constant utilization.

Each replica generated its French partition, then immediately its code partition.
There was no global barrier between the two sources. Source durations below use
the earliest start and latest finish within that source; they overlap slightly
because one replica began code before the other finished French. Do not sum
these durations as if they were disjoint phases.

## Results by source

| Metric | French S1 | English code S2 | Combined |
| --- | ---: | ---: | ---: |
| Answers | 41,197 | 50,000 | 91,197 |
| Prompt tokens | 3,705,039 | 6,508,803 | 10,213,842 |
| Completion tokens | 17,531,941 | 57,485,860 | 75,017,801 |
| Mean completion tokens/answer | 425.56 | 1,149.72 | 822.59 |
| Source span / overall wall time | 43m 35s | 2h 22m 28s | 3h 5m 35s |
| Completion tokens/s | 6,703 | 6,725 | 6,737 |
| Failed requests | 0 | 0 | 0 |
| Empty answers | 0 | 0 | 0 |
| Answers reaching the 8,192-token cap | 8 | 287 | 295 |
| Cap rate | 0.0194% | 0.5740% | 0.3235% |

The four JSONL answer files occupy 411,046,530 bytes, approximately 392 MiB,
excluding manifests, logs, preparation artifacts, and smoke outputs. Reported
reasoning-token totals are zero; thinking was explicitly disabled in requests.
Zero reported reasoning tokens alone is not independent proof of no reasoning
spill when no reasoning parser is configured.

## Results by replica

| Worker | Answers | Completion tokens | Worker duration | Completion tokens/s | Capped |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fr-replica-0` | 20,607 | 8,770,565 | 43m 35s | 3,353 | 6 |
| `fr-replica-1` | 20,590 | 8,761,376 | 43m 6s | 3,388 | 2 |
| `code-replica-0` | 24,929 | 28,733,209 | 2h 21m 58s | 3,373 | 139 |
| `code-replica-1` | 25,071 | 28,752,651 | 2h 20m 4s | 3,421 | 148 |

Both queues completed without the manifest-path printing crash that affected
the older English run; the regression fix was in place for this wave.

## Integrity and provenance

The morning checks independently verified:

- Every prepared generation prompt has exactly one answer, with no missing,
  unexpected, or duplicate IDs and no changed prompts.
- All four output files parse as complete, newline-terminated JSONL.
- SHA-256 hashes, byte sizes, record counts, and completion-token totals match
  their respective manifests.
- All four manifests report zero failures and zero empty answers; the launcher
  records `Replica exit codes: 0 0`.

| Output under the run directory | SHA-256 |
| --- | --- |
| `fr-replica-0/shard-00-of-02.jsonl` | `0cf0167cc13182ed8006f335e54e88ba592f391ecde14ccc707bbcdbcb8abdb2` |
| `fr-replica-1/shard-01-of-02.jsonl` | `29978b9ad9caa23f78f6bb7d988a473b97b5a35924f89e9a23be095fab19f4ab` |
| `code-replica-0/shard-00-of-02.jsonl` | `b3b70e1284c23f20afc8dfb4a3f3163dcd964d9c6708b24afdd17d4c290950b6` |
| `code-replica-1/shard-01-of-02.jsonl` | `8e756e1db971427e933ae7b175603473ae63e30adee0d9f21968ee23b51d565c` |

Preparation directories are `data/runs/cfp-prep-20261103` and
`data/runs/future-code-prep-20261103`. Each has a pinned source card, manifest,
generation prompts, and 500 separate held-out prompts. Those held-out files
were not generation inputs. Source assistant answers were discarded.

- French: `CATIE-AQ/CFP`, revision `8cc06909e4bd09255b3b379d9135f87fabeabc17`.
  Excludes selected benchmark-derived sources, the labelled medical validation
  split, and XGLUE/QAM; retained inputs still have per-source license obligations.
- Code: `future-architect/Llama-3.3-Future-Code-Instructions`, revision
  `148be2ee007fcd511356e0a2a8212e7b6e0fda66`. A deterministic 50,500-row reservoir
  sample of a bounded 250,000-English-row subset yielded 50,000 generation and
  500 held-out prompts. It is not a uniform sample of the entire corpus.
- The code source declares the Llama 3.3 license, including a naming condition
  for distributed models improved using Llama outputs. Regenerating answers
  does not remove obligations associated with its generated prompts.
- Preparation checked frozen benchmark exclusions, normalized duplicates,
  previous accepted English prompts, and cross-source/held-out separation.
  Exact hashes do not establish absence of paraphrased benchmark contamination.
- No d1 routing/suitability pass was applied to these two prepared sources.

## Frozen generation configuration

- Teacher: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- vLLM image: `sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f`.
- Runtime workaround: uninstall `torchcodec` inside each container before
  serving. The base image digest alone does not capture this modification.
- Template SHA-256: `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
- Temperature 0, top-p 1, seed 0, thinking disabled, effort low, max tokens 8,192.
- Concurrency 64 and timeout 900 seconds per replica.
- Disjoint shard indices 0/1 with shard count 2; separate source/replica run IDs.
- Successful smoke: `data/runs/fr-code-smoke-20261103-214906`, 200 answers/source.
  Smoke outputs overlap the bulk inputs and must not be added as extra training
  records when assembling the bulk export.

## Estimate versus actual runtime

The short smokes suggested approximately 16.7M French and 53.1M code completion
tokens, 69.8M combined, and about 9.3 hours of generation. Actual output was
75.02M tokens, but wall time was only 3.09 hours: roughly three times faster
than that duration extrapolation.

The short-smoke rates were approximately 2,182 French and 2,064 code tokens/s,
whereas sustained bulk rates were around 6,700 tokens/s. A 100-request worker
includes a disproportionate prefill/drain interval, and a long capped answer can
dominate its finish time. This is consistent with the estimate error, although
no controlled experiment isolated the contribution of each effect.

For future scheduling, keep source-specific mean answer lengths, use a longer
steady-state throughput window, compare against measured prior bulk rates, and
retain a retry reserve. Smoke tests establish basic correctness; their elapsed
times are not reliable throughput benchmarks for an overnight queue.

At 08:14 CET, the teachers were still loaded but all GPUs reported 0% utilization,
and the bulk tmux session had exited normally. No downstream training was queued,
as agreed at handoff. Generation therefore did not fill the whole night; no
automatic teacher shutdown or SFT handoff was configured. Continuous GPU activity
was not recorded by this analysis, so isolated utilization readings are not a
complete utilization history.

## Quality and next actions

1. Preserve raw outputs, preparation artifacts, logs, manifests, and overlapping
   smoke outputs separately; refresh the verified backup to the Spark.
2. Produce QA-cleaned exports. Exclude the 295 capped answers from complete-target
   training exports, then inspect repetition and other degeneration in normally
   stopped answers as well. Cap counts alone do not measure total defect rate.
3. Review French task/subject balance and factual quality. For code, inspect
   completeness and later test correctness in a network-disabled, disposable
   sandbox; no full correctness evaluation or repetition scan has been completed
   for this wave at the time of this report.
4. Build the SFT harness and validate assistant-only loss, the teacher template,
   checkpoint saving, and resume before launching training. Freeze the exact
   QA-cleaned inputs; do not train on unreviewed raw output by default.
5. Add tool-use coverage and complete acceptance/evaluation infrastructure before
   treating the eventual draft as the final FR/EN/code/tool mixture.

Together with the previous English wave, raw bulk generation now totals
**201,694 answers and 169,245,231 completion tokens**, excluding smoke outputs.
That is useful material for distillation, not a count of approved training tokens
or evidence that draft acceptance has improved. No new student checkpoint exists
from this night's job.

Related documents: [previous English night](night-run-20261101-analysis.md),
[French/code preparation](french-code-preparation.md),
[smoke procedure](french-code-step1-smoke.md),
[bulk launcher and monitoring](french-code-step2-bulk.md),
[project status and handoff](project-status-20261103.md), and
[Spark backup](backup-to-spark.md).