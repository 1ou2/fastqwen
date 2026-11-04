# Project recap and overnight decision

## QA update: November 4

Step 1 is complete. The CPU-only exporter audited all 201,694 bulk records and
produced 195,991 training rows (159.59M completion tokens), 4,181 validation rows,
1,194 quarantined rows, and 328 rejected capped answers. EN, FR, and code source
groups remain explicit, with no automatic mixture balancing. Raw records and
the frozen/new evaluation hold-outs are unchanged; independent checks found no
accepted hold-out contamination or train/validation overlap. Automated QA is not
a correctness or licensing clearance. See [policy, results, and reproducible
command](training-data-qa.md). No SFT was launched or teacher container changed.
Next: implement the SFT harness and validate serialization/checkpoint/resume on
a small smoke before committing to a training run.

## Morning update: November 4

The French/code run **completed successfully at 01:14:43 CET** with worker exit
codes `0 0`. It produced 91,197 answers and 75,017,801 completion tokens in
3h 5m 35s, averaging 6,737 completion tokens/s. French contributed 41,197 answers
and 17.53M tokens; code contributed 50,000 answers and 57.49M tokens. No request
failures or empty answers were reported. There were 8 capped French answers and
287 capped code answers. The later QA update above supersedes this morning
snapshot's pending-export status; human review remains pending.

Morning checks verified output hashes, sizes, counts, complete prompt coverage,
unchanged prompts, no duplicate IDs, and valid newline-terminated JSONL. At 08:14,
the teachers remained loaded, all GPUs reported idle, and the tmux job had exited
normally. No SFT was queued or run. At that point the QA exporter and SFT harness
were next; the exporter has since completed as described above.
The backup script now also includes checkpoint and calibration directories.

See [the full November 3-4 analysis](night-run-20261103-analysis.md) for timings,
token totals, replica results, estimate error, provenance, and conclusions.
The sections below preserve the November 3 end-of-day snapshot and its estimates;
their running-state descriptions are historical, not the current job status.

## November 3 snapshot

Status: November 3, 2026, end-of-day handoff. French/code bulk generation is
running on all four H100s. Interactive work is paused for the night; generation
will continue unattended. This is a status snapshot, not a change to the reference
project plan.

Times below are rough wall-clock planning budgets, not GPU-hours. Only the
English classification/generation and French/code smoke durations are measured.
Other estimates include explicit GPU counts where known; missing implementation
work is separate from runtime. Spark baseline completion has not been rechecked
in this session.

## Completed today

- Analyzed the completed English wave: 110,497 answers, 94.23M completion tokens,
  complete prompt coverage, valid JSONL, and matching output hashes/manifests.
  Identified capped answers, repetition, and residual S1/S2 boundary cases for QA.
- Fixed the generator's end-of-run manifest-path printing crash. Regression
  tests cover relative, external absolute, and repository-local output paths.
- Added the resumable Spark-side backup script and instructions. It copies runs,
  reference inputs, and documentation, checks contents with checksums, and never
  deletes local files. Local transfer/recovery tests passed; the real Spark
  transfer has not been verified here.
- Prepared pinned CFP and Future Code prompts: 41,197 French S1 and 50,000 English
  S2 generation prompts, plus 500 held-out prompts per source. Source cards,
  provenance, exclusions, distributions, and hashes are saved with each export.
  Original assistant answers were discarded; licensing caveats are documented.
- Verified hashes, counts, cross-source and held-out separation, frozen benchmark
  exclusions, and exclusion of the previous accepted English prompts. Thirty
  focused preparation/dataset tests passed, as did both generator dry-runs.
- Wrote restart/smoke instructions, restarted the pinned two TP=2 teachers, and
  completed all four 100-answer smoke workers. No failures or empty answers;
  French had zero capped answers and code had one out of 200.
- Added and validated a guarded bulk launcher, then started the French/code
  generation queues in detached tmux at 22:09:07 CET.

## Overnight handoff

- Session: `fr-code-bulk-20261103-220905`.
- Bulk outputs: `data/runs/fr-code-bulk-20261103-220905/`.
- Launcher log: `data/runs/fr-code-bulk-20261103-220905-launch.log`.
- Smoke outputs: `data/runs/fr-code-smoke-20261103-214906/`.
- Teachers: `fr-code-smoke-20261103-214906-teacher-0` on GPUs 0-1/port 8000 and
  `fr-code-smoke-20261103-214906-teacher-1` on GPUs 2-3/port 8001.

At the last read-only handoff check, replica 0 had logged 2,650/20,607 French
answers and replica 1 had logged 2,750/20,590. These are logged progress counters,
not final totals. All four GPUs reported 100% utilization at that instant; this
does not establish continuous saturation.

Each replica finishes its French partition, then automatically starts its code
partition. Separate source/replica directories prevent manifest overwrites.
Inputs and generation settings remain frozen. SSH disconnection does not stop
the detached tmux workers or Docker containers.

**Only generation is queued.** No QA exporter, SFT trainer, acceptance evaluation,
or tool-use job will start automatically tonight. When generation completes,
the launcher records worker exit codes and exits; the teacher containers stay
running until explicitly stopped. Do not stop them while workers are active.

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
| B1: French and code teacher smoke | H100 | Complete: 200 answers/source, no failures or empty answers; one code cap | Measured 37.1 s French and 103.0 s code, excluding teacher startup and review |
| B1: French and code bulk generation | H100, four GPUs as two TP=2 replicas | Running since 22:09:07 CET; French then code per replica | Short-smoke extrapolation approximately 9.3 h; bulk runtime not measured yet |
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

The English bulk wave measured approximately 6,740 completion tokens/s across
both TP=2 replicas. Today's short French/code smokes measured:

| Source | Measured mean smoke answer length | Projected full-source completion volume | Time extrapolated from short smoke on four GPUs |
| --- | ---: | ---: | ---: |
| 41,197 French prompts | 404.6 tokens | 16.7M tokens | 2.12 h |
| 50,000 code prompts | 1,062.8 tokens | 53.1M tokens | 7.15 h |

The implied total is approximately 69.8M completion tokens and 9.3 h. This is
not a promised duration: combined smoke rates were only about 2,182 French and
2,064 code tokens/s and include short-run prefill/drain effects and a capped code
answer. Steady bulk throughput may be higher; source variation and retries may
also change the result. Use live progress for updated estimates. The 8,192-token
cap is a safety limit, not an expected answer length.

## Decisions and next work

### Tonight: generation only

The restart, smoke, and bulk launch are complete. Leave the queues running;
do not launch duplicate workers, change input files, or stop their teachers.
Development of QA and training is deferred to the next working session.

In the morning, read `exit-status.txt` and the four manifests, verify coverage
and hashes, and inspect failures, caps, and repetition. Success is worker exit
codes `0 0`, followed by integrity and quality checks; an absent exit-status file
does not prove failure if the workers are still running. Preserve any partial
outputs and diagnose failed workers before resuming.

### Next session: QA exporter and SFT smoke

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
  and failure logs if the job has not finished. Final auditing remains a morning
  task; raw completion alone is not a QA-cleaned training set.
- Preserved source/preparation/smoke/generation manifests and logs under
  `data/runs/`, covered by the Spark backup.
- No new SFT checkpoint is expected tonight: the QA exporter and trainer still
  need implementation and smoke validation. Future training checkpoints and
  optimizer state should also live under `data/runs/`; a separate top-level
  checkpoint folder would not be copied by the existing backup script.

Before releasing the server, stop writers and perform the final verified backup.

Related documents: [reference plan](../project-plan.md),
[night analysis](night-run-20261101-analysis.md),
[French/code preparation and smoke commands](french-code-preparation.md),
[step 1 smoke procedure](french-code-step1-smoke.md),
[step 2 bulk launch and monitoring](french-code-step2-bulk.md),
[teacher deployment](overnight-h100.md), and [Spark backup](backup-to-spark.md).