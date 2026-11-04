# Training data QA: November 4, 2026

Step 1 is complete: CPU-only automated QA and a deterministic training/validation
export of the six completed English, French, and code bulk shards. No smoke
answers were included. No SFT was launched, and teacher containers were untouched.

Implementation: [exporter](../scripts/export_training_data.py),
[policy](../qwenbench/qa.py), and [tests](../tests/test_qa.py).
The completed [export manifest](../data/runs/qa-export-20261104-v1/manifest.json)
records input/output hashes, provenance, policy, split settings, counts, token
budgets, and completion status. Outputs live under
`data/runs/qa-export-20261104-v1/`, already covered by the Spark backup script.
The actual Spark transfer still needs verification.

## Results

| Source group | Train | Validation | Review | Rejected | Train completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s1-en` | 107,896 | 2,271 | 297 | 33 | 91,339,179 |
| `s1-fr` | 40,052 | 892 | 245 | 8 | 16,661,161 |
| `s2-code` | 48,043 | 1,018 | 652 | 287 | 51,586,557 |
| Total | **195,991** | **4,181** | **1,194** | **328** | **159,586,897** |

All 201,694 raw records have a disposition. Accepted train plus validation:
200,172 records and 162,973,146 completion tokens. Validation alone has 3,386,249
completion tokens. Review retains 3,585,109 tokens; rejected records contain
2,686,976 tokens. Token budgets are the teacher's recorded completion-token
counts, not student-tokenized sequence lengths or measured training cost.

All 328 rejected records hit the generation cap. Review reasons below overlap;
their sum is not the number of reviewed records:

| Review reason | Flags |
| --- | ---: |
| Repeated long line | 933 |
| Repeated word sequence | 131 |
| S1 technical code fence | 128 |
| Long identical-character run | 46 |
| Template/reasoning marker | 2 |

There were no duplicate, prompt-hash mismatch, frozen-evaluation, or external
hold-out hits. No empty, unsuccessful, tool, or separate-reasoning records were
encountered. These cases are covered by policy/tests, not inferred absent solely
from an empty-content aggregate.

## Policy v1

Accepted means **passed automated text-only QA**, not factually correct, passing
code tests, safe for every application, or cleared for redistribution.

- Reject missing/invalid IDs, prompts, or text content; capped, unsuccessful, or
  empty text answers. Reject normalized-prompt hash mismatches and excluded or
  duplicate prompts/IDs. The first occurrence wins in recorded input order;
  rejected or quarantined records are not silently replaced by later duplicates.
- Quarantine tool-bearing records before the empty-text check. Tool-only answers
  need a tool-aware serialization/validation pipeline, not this text export.
- Quarantine unsupported slices, separate reasoning, thinking-enabled records,
  and literal `<think>`, `</think>`, `<|im_start|>`, or `<|im_end|>` markers.
- Quarantine runs of at least 80 identical non-whitespace characters, stripped
  lines of at least 30 characters repeated at least 10 times, or a repeated
  whitespace-delimited 1-12-word sequence appearing at least 20 times within the
  final 6,000 answer characters. Detection is case-sensitive and conservative;
  it does not detect every form of degeneration.
- Quarantine S1 answers with recognized technical code-fence languages. S2 code
  fences are allowed. The policy manifest records the language list.

These are triage rules. Lists, ASCII separators, repeated code, legitimate marker
examples, and useful mathematical answers with code can trigger review. No
reviewed record is treated as confirmed bad or automatically restored. Human
review can inform a separately versioned export; do not edit raw targets or the
completed export in place.

## Split and provenance

The frozen 290-prompt exclusion manifest and both 500-prompt preparation
hold-outs remain unchanged. The new validation split is separate from those
evaluation sets. Membership uses a SHA-256 threshold of
`SHA256("20261104:" + normalized_prompt_sha256)` with fraction `0.02`.
This gives an approximately 2% split, independent of input order for unique
prompts; it is not an exact-size stratified sample. Deduplication prevents the
same ID or normalized prompt from appearing in both accepted splits. This is
exact normalized-prompt exclusion, not semantic near-duplicate detection.

Each input must match its generation manifest's SHA-256, byte count, record count,
completion-token total, and teacher cohort. Every record's teacher identity,
template hash, and sampling settings must match. Incomplete tails, malformed JSON,
inconsistent provenance, and changed inputs fail closed. Only a manifest with
`status: completed` is consumable; failed exports can contain partial files and
must never be used for training. Existing output directories are refused.

Train/validation rows preserve original generation fields and add `qa_group`,
`qa_input_file`, and `qa_input_line`. They are not yet serialized student chat
examples. Review/rejected rows wrap the preserved record plus `qa_reasons`;
`decisions.jsonl` maps every raw position to its disposition. The original bulk
files are never rewritten. File hashes cover all five JSONL outputs.

Group labels are explicit source assignments, not automatic language detection
or a new classifier pass. Original `source_file`, prompt IDs, and preparation
artifacts remain available for provenance joins. CFP's per-upstream obligations
and Future Code's Llama 3.3 conditions still apply. This QA pass does not resolve
license clearance or correct source misclassification.

No balancing or resampling was applied. Source row fractions and token fractions
differ substantially; keep the groups separate when choosing SFT mixture weights.
Tool-use trajectories, student-template serialization, length/packing checks,
checkpoint/resume smoke, and held-out acceptance evaluation remain future work.

## Reproduce

Run from the repository root, selecting a fresh output directory:

```bash
uv run python scripts/export_training_data.py \
  --input s1-en=data/runs/s1-night-20261101-222544/replica-0/shard-00-of-02.jsonl \
  --input s1-en=data/runs/s1-night-20261101-222544/replica-1/shard-01-of-02.jsonl \
  --input s1-fr=data/runs/fr-code-bulk-20261103-220905/fr-replica-0/shard-00-of-02.jsonl \
  --input s1-fr=data/runs/fr-code-bulk-20261103-220905/fr-replica-1/shard-01-of-02.jsonl \
  --input s2-code=data/runs/fr-code-bulk-20261103-220905/code-replica-0/shard-00-of-02.jsonl \
  --input s2-code=data/runs/fr-code-bulk-20261103-220905/code-replica-1/shard-01-of-02.jsonl \
  --heldout-prompts data/runs/cfp-prep-20261103/heldout.jsonl \
  --heldout-prompts data/runs/future-code-prep-20261103/heldout.jsonl \
  --seed 20261104 --validation-fraction 0.02 \
  --output-dir data/runs/qa-export-20261104-v1-repeat
```

All six input manifests, the frozen exclusion file, and both hold-out files are
hashed in the export manifest. The default frozen exclusion file is
`benchmarks/validation/manifest.json`; use `--exclusion-manifest` for another
explicitly chosen contract. Do not glob every JSONL in a run tree: that can
accidentally include smoke answers, errors, or overlapping exports.

## Verification

The 26 focused QA tests passed, including tools, caps, repetition, C++ fences,
duplicates, prompt-hash mismatches, exclusions, split order independence,
fresh-directory refusal, modified files, malformed/truncated JSONL, inconsistent
totals, and failed-state manifests. Editor diagnostics and Pylance syntax checks
were clean for the three new Python files.

An independent full-data audit rehashed inputs, generation manifests, exclusions,
hold-outs, and all output files; compared every exported record's original fields
with its raw source position; checked all records were accounted for exactly
once; reconciled decisions, reason counts, record counts, and token totals; and
checked accepted IDs/prompts, stop/nonempty/text-only targets, split separation,
and exclusions. Result: zero changed raw records, zero train/validation prompt
overlap, and zero accepted protected-prompt overlap.

This audit is evidence of integrity and policy compliance, not a scored quality
or speculative-decoding acceptance result. The next implementation step is the
SFT harness and a small checkpoint/resume smoke, with mixture and serialization
decisions made explicitly before a long training run.