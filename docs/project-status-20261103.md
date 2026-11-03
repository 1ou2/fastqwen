# Project recap and overnight decision

Status: November 3, 2026. The four H100s remain available for more than 12 hours;
the Qwen teacher containers are stopped. This is a status snapshot, not a change
to the reference project plan.

Times below are rough wall-clock planning budgets, not GPU-hours. Only the
English classification and generation durations are measured. Other estimates
include explicit GPU counts where known; missing implementation work is separate
from runtime. Spark baseline completion has not been rechecked in this session.

## Whole project

| Step | Machine | Current state | Rough time |
| --- | --- | --- | --- |
| Model/cache, container, hardware setup | Spark and H100, separately | H100 serving validated; teacher/student revisions pinned | 1-3 h setup per environment; downloads depend on bandwidth |
| Tokenizer and teacher-template contract | CPU on either host | Pinned contract exists | 10-30 min for a verification run |
| Frozen evaluation sets and grading infrastructure | CPU on either; Spark for scored inference | 290 frozen exclusions; new 500 FR + 500 code held-out prompts; broader grading/sandbox work unfinished | 2-6 h implementation/review; small CPU checks take minutes |
| Step 0: Spark baseline, public anchor, BF16 reference outputs/logits | Spark | Probe exists; formal G0 completion not established here | 1-4+ h runtime once harnesses work; large suites can take much longer |
| A: FP8 arm | Spark for load/inference and quality comparisons | Not implemented/verified in this session | 1-2 h initial load/smoke; broader scoring extra |
| A: mixed calibration set and NVFP4 export | H100 | Preparation/export harnesses missing | 2-4+ h implementation; plan budgets 1-2 h GPU runtime |
| A: NVFP4 loading and quality gate G1 | Spark | Pending compatible exported checkpoint | 1-3 h checks once measurement tools work |
| A: optional aggressive quantization and quality comparison | H100 for export; Spark for measurements | Pending; not an overnight priority | 1-3+ h export/recipe work, then 1-2+ h measurement |
| B1: English S1 classification | H100 host; d1 on one GPU | Complete, 179,816 candidates classified | Measured 49 min |
| B1: English S1 teacher generation | H100, four GPUs as two TP=2 replicas | Complete: 110,497 answers, 94.23M completion tokens | Measured 3 h 53 min |
| B1: French and code prompt preparation | CPU/network on H100 host | Complete: 41,197 FR + 50,000 code prompts, separate hold-outs and manifests | Preparation runs take minutes; adapter is implemented |
| B1: French and code teacher smoke | H100 | Ready to launch after source-term review and teacher restart | Budget 20-45 min including startup, 200 requests/source, and inspection |
| B1: French and code bulk generation | H100, four GPUs as two TP=2 replicas | Ready after smoke; no answers generated yet | Provisional 3-8 h; re-estimate from smoke |
| B1: S3 tool-use sources, formatting, parser, and validation | CPU preparation on H100 host; teacher on H100 | No bulk source/shard ready; single-turn tools are supported by generator | Budget 2-4+ h preparation, then 1-3 h for a small validated wave; highly source-dependent |
| B1/B2: QA-cleaned training exports and held-out splits | CPU on H100 or Spark | Raw answers checked; exporter/manual disposition missing | 1-2+ h implementation/review, then minutes per export |
| B2: SFT harness and checkpoint/resume smoke | H100 | Trainer missing | 2-4+ h implementation; 15-45 min GPU smoke/debug once runnable |
| B2: first 2B SFT checkpoint | H100 | Requires clean data and successful smoke | Reference plan: 6-10 h on one H100; distributed runtime is unmeasured |
| C preparation: raw/trained draft acceptance and gamma sweep | H100, teacher plus student | Offline acceptance harness missing | 2-4+ h implementation; plan budgets 1-2 h evaluation runtime |
| B2: optional logit KD | H100 | Only if measured acceptance misses G2 | Plan budgets another 5 h; implementation and actual throughput unverified |
| C: native MTP, raw draft, trained draft, optional MTP+draft | Spark | Pending serving compatibility and trained checkpoint | Budget 2-4+ h comparisons once harnesses work |
| C: greedy token-identity verification | Spark | Required for final speculative arms | 30-90 min for short frozen demos; compatibility/debugging extra |
| D: KV dtype, prefill, prefix caching, graphs, batching | Spark | Pending winning serving configuration | 2-4+ h controlled A/B measurements |
| E: final race and qualitative captures | Spark | Last step, after quality/exactness gates | 1-3 h capture/measurement; video editing excluded |
| Backup of temporary-server artifacts | Spark pulls from H100 server | Script implemented; actual final transfer not verified here | Budget 10-30+ min depending on network and checkpoint size |

H100 throughput is useful for generation/training planning, not for claiming
Spark inference speedups. Spark remains the machine for all headline speed,
memory, quality, and speculative-exactness results. Formal G0/G1/G2/G3 gates
remain applicable; an exploratory cluster checkpoint is not a final success claim.

## Generation time assumptions

The measured English rate was approximately 6,740 completion tokens/s for both
TP=2 replicas, or 3,400 for one replica. French and code rates are unmeasured.
Using the observed combined rate only as a provisional assumption:

| Source | Assumed mean answer length | Approximate completion volume | Pure generation time on four GPUs |
| --- | ---: | ---: | ---: |
| 41,197 French prompts | 250-1,000 tokens | 10-41M tokens | 0.4-1.7 h |
| 50,000 code prompts | 1,000-3,000 tokens | 50-150M tokens | 2.1-6.2 h |

These answer lengths are scenarios, not measurements. Queue planning should use
the representative mean completion length and actual wall-clock throughput from
the source smoke, plus startup/retry reserve. Do not use 8,192 tokens per prompt
as the expected workload: that is the safety cap. One TP=2 replica roughly
doubles these durations only if it maintains its measured rate.

## What to launch tonight

### Reliable first launch

1. Review the source terms documented in the preparation guide.
2. Restart the two pinned teacher replicas and run 200-request smokes for both
   French and code. Inspect caps, repetition, French quality, and code fit.
3. If the smokes pass, launch the French and code generation queues on all four
   GPUs with disjoint partitions and separate worker run IDs.
4. While generation runs, implement the QA exporter and SFT trainer on CPU.

The advantage is that validated generation can make progress while the missing
training infrastructure is built. It need not occupy all 12 hours to be useful.

### Second launch, gated on a real SFT smoke

Once the trainer is ready, free a teacher replica's two GPUs after its worker
finishes, or stop that worker cleanly before stopping its container. Do not kill
a server that still has an active generation worker. Smoke-test SFT on roughly
2,000 clean English records for 50-100 steps, including checkpoint save/resume,
assistant-only loss masking, and held-out loss. Tool and reasoning records need
their own serialization tests before entering that trainer.

If smoke passes, launch a checkpointed exploratory English SFT job on those
two GPUs while the other teacher replica continues its own generation partition.
This is a temporary S1 checkpoint, not the final FR/EN/code/tool mixture.
Do not add partially generated or unreviewed new shards to an active training job.
Freeze training inputs and their hashes when it starts.

If all generation finishes before the trainer is ready, stop both teachers and
use the available GPUs for the smoke and validated training job. Four-GPU
training requires its own distributed smoke; do not assume linear speedup or
change the training world size mid-run without a verified resume procedure.

### Do not make S3 the critical path tonight

Tool-use examples are important for the final draft, but a correct S3 pipeline
needs actual schemas, correct tool names/arguments, no-tool cases, and validated
serialization. The current generator sends `tools` and saves `tool_calls` for a
single user turn; it does not execute tools or construct multi-turn trajectories.
A tool-only answer can have empty text: the current generic empty-answer count
would flag it, so text-only QA must not discard those records automatically.
Server parser configuration and structured tool-call extraction need a real smoke.

Prepare a small S3 pilot only after the generation queue and SFT smoke are
secured. Do not label ordinary instructions asking for JSON as real tool-use data.
Native MTP comparisons, optional KD, reasoning S4, and quantization are not the
critical path for this night's draft-training objective.

## Expected morning deliverables

- Audited French and code teacher outputs, or valid partial outputs with progress
  and failure logs if time expires.
- A QA-cleaned English training export.
- A tested SFT trainer and, if its gate passes, a resumable first checkpoint.
- Source/preparation/generation/training manifests and logs saved under
  `data/runs/`, including training checkpoints and optimizer state, so the existing
  Spark backup includes them. Files in a separate top-level checkpoint folder
  would not be copied by that script.

Before releasing the server, stop writers and perform the final verified backup.

Related documents: [reference plan](../project-plan.md),
[night analysis](night-run-20261101-analysis.md),
[French/code preparation and smoke commands](french-code-preparation.md),
[teacher deployment](overnight-h100.md), and [Spark backup](backup-to-spark.md).