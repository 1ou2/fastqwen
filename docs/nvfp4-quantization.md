# Offline NVFP4 quantization on H100

This workflow quantizes the pinned **Qwen3.8-27B target**, not the trained 2B
draft. Calibration/export run on one H100; native NVFP4 loading, quality, speed,
memory, and MTP engagement are separate Spark/GB10 checks.

## Environment

Use a separate environment: ModelOpt 0.48.0's HF integration requires
Transformers below 5.19, while the training environment uses 5.22.1.
The minimal dependency set avoids unnecessary TensorRT-LLM/DeepSpeed installs.

```bash
uv venv --python .venv/bin/python .venv-quant
uv pip install --python .venv-quant/bin/python -r requirements-quantization.txt
```

The pinned target must already be downloaded into its HF cache snapshot.
No model download, remote model code, existing teacher container stop, or training
environment modification is performed by the scripts.

## Calibration inputs

[The builder](../scripts/build_calib_set.py) accepts `--input GROUP=JSONL`.
Default counts are French 256, code 128, tools 64, reasoning 64. Production
requires all four groups, at least 512 sequences, and a cap of at least 2,048
tokens. Actual lengths and capped-sequence counts are recorded; short inputs
are not padded to pretend they contain 2,048 real tokens.

Each JSONL row has an `id` and one of:

- French/code raw text: `{"id": "fr-1", "text": "..."}`.
- A teacher record with `prompt`, `content`, optional `reasoning`,
  `tools`, and `tool_calls`.
- A text-only `messages` conversation using user/assistant/system/tool roles.
  Tools rows require real `tools` schemas, including for refusal/no-tool cases.
  OpenAI JSON-string tool arguments are normalized to objects for the target template.
  Reasoning rows require an assistant's nonempty `reasoning_content`.

The existing QA training export can supply French and code: `qa_group` is
filtered to `s1-fr` and `s2-code` respectively. Tools/reasoning must be supplied
separately; they are not manufactured from the non-thinking QA corpus.
Source structure checks are not factual, tool-execution, or licensing clearance.

All default exclusion files must exist: frozen qualitative/quality/acceptance
sets, French/code preparation hold-outs, both comparison prompt sets, and the
monitored diagnostic hashes. Extra/new evaluations must be included in an
explicit complete `--exclude-prompts FILE ...` list; this option replaces the
defaults. Exclusions use normalized exact user-prompt hashes, not semantic
deduplication. Inputs are hashed and selections are deterministic.

```bash
.venv-quant/bin/python scripts/build_calib_set.py \
  --input fr=data/runs/qa-export-20261104-v1/train.jsonl \
  --input code=data/runs/qa-export-20261104-v1/train.jsonl \
  --input tools=data/calib/nvfp4-inputs-20261115/tools-conversations.jsonl \
  --input reasoning=data/calib/nvfp4-inputs-20261115/reasoning-conversations.jsonl \
  --output-dir data/runs/calibration-nvfp4-20261115
```

The tools/reasoning paths above were prepared in the full calibration run below.
The builder writes `calibration.jsonl` and a completed `manifest.json`.

## Export

[The exporter](../scripts/quantize_nvfp4.py) defaults to `mixed`: text MLP and
full-attention Linear projections become NVFP4. Gated DeltaNet projections,
vision, embeddings, LM head, and MTP stay in their original precision.
`--recipe mlp-only` is the more conservative alternative. Neither recipe promises
the size/speed of a fully quantized model.

The checkpoint contains 15 `mtp.*` tensors absent from the Transformers model.
The official ModelOpt exporter carries checkpoint-only weights through; this
workflow checks complete original-key coverage and exact MTP tensor identity.
Vision and Gated DeltaNet tensors are also compared exactly. Packed weight
dimensions and all NVFP4 scale tensor dimensions/dtypes are checked against the
original Linear shapes. Configuration checks use normalized HF configuration
objects so a harmless canonical vision-type name is not treated as a model
change. Loss of original weights or semantic configuration changes fails the run.

First run CPU/input preflight, then expose a free GPU explicitly:

```bash
.venv-quant/bin/python scripts/quantize_nvfp4.py \
  --calibration-dir data/runs/calibration-nvfp4-20261115 \
  --output-dir data/runs/nvfp4-mixed-20261115 --dry-run

CUDA_VISIBLE_DEVICES=2 .venv-quant/bin/python -u scripts/quantize_nvfp4.py \
  --calibration-dir data/runs/calibration-nvfp4-20261115 \
  --output-dir data/runs/nvfp4-mixed-20261115
```

GPU 2 is an example, not a reservation. Check current GPU users before launch.
The exporter requires one visible GPU and enough free GPU/disk bytes for the
original checkpoint plus an 8 GiB reserve. OOM/runtime/export errors are explicit;
failed output directories are preserved and cannot be reused as fresh runs.

Successful runs have `manifest.json` with `status: completed` and a
`checkpoint/` directory containing packed NVFP4 safetensors, scales, ModelOpt
quantization metadata, original tokenizer/template, and preserved MTP weights.
Source weights are not rewritten. Manifests record hashes, versions, recipe,
command, GPU, timings, and implementation snapshots. Existing Spark backup
scope includes these runs; rerun backup after completion.

## Bounded implementation smoke

An incomplete mixture must be explicitly labelled at both stages:

```bash
.venv-quant/bin/python scripts/build_calib_set.py \
  --input fr=data/runs/qa-export-20261104-v1/train.jsonl \
  --input code=data/runs/qa-export-20261104-v1/train.jsonl \
  --samples fr=2 code=2 --max-length 128 --smoke \
  --output-dir data/runs/calibration-nvfp4-smoke-20261115

CUDA_VISIBLE_DEVICES=2 .venv-quant/bin/python -u scripts/quantize_nvfp4.py \
  --calibration-dir data/runs/calibration-nvfp4-smoke-20261115 \
  --output-dir data/runs/nvfp4-smoke-20261115 --smoke
```

This checks real target calibration/export mechanics, not production quality.
Do not deploy it as the final calibrated candidate.

## Verified implementation result: November 15, 2026

The real [H100 smoke run](../data/runs/nvfp4-smoke-20261115-v3/manifest.json)
completed at 21:12:08 CET (20:12:08 UTC), with process exit code zero and
`status: completed`. It used four capped 128-token sequences (two French,
two code), 512 tokens total, and the pinned full target on GPU 2.

- 256 text Linear modules were quantized with the mixed recipe.
- The [exported checkpoint](../data/runs/nvfp4-smoke-20261115-v3/checkpoint)
  contains 28,574,978,139 bytes (about 28.57 GB / 26.61 GiB).
- All 15 MTP, 333 vision, and 432 Gated DeltaNet tensors were checked unchanged.
- Original checkpoint coverage, normalized model configuration, NVFP4 metadata,
  calibration integrity, and original source-file hashes passed.
- Recorded run time was 130.3 seconds; peak allocated GPU memory was
  55,616,692,224 bytes. This tiny run does not predict a full calibration duration.
- A subsequent layout check verified packed weights and all three scale tensors
  for every quantized module against the original dimensions.

Two earlier attempts are preserved as failed runs: the first exposed
Transformers' set-valued loading diagnostics during manifest serialization;
the second rejected the harmless vision-type alias before normalized checks were
introduced. Neither is a completed export result.

The teachers on GPUs 0-1 were not stopped; GPU 2 was released after the bounded
run. At that point no full production calibration, Spark serving, quality campaign,
or performance comparison had been launched; the tools/reasoning inputs were
still missing.

## Full calibration prepared: November 15, 2026

The [full calibration manifest](../data/runs/calibration-nvfp4-20261115/manifest.json)
is completed with `smoke: false`. The token file contains **512 sequences,
421,921 actual tokens**, with a 2,048-token cap and no padding:

| Group | Sequences | Tokens | Median length | Capped sequences |
|---|---:|---:|---:|---:|
| French | 256 | 125,752 | 333 | 1 |
| Code | 128 | 139,501 | 993.5 | 19 |
| Tools | 64 | 35,843 | 391 | 1 |
| Reasoning | 64 | 120,825 | 2,048 | 36 |
| Total | 512 | 421,921 | - | 57 |

French/code come from the existing QA training export. The missing groups were
prepared from **original synthetic prompts**, not copied evaluation questions or
reasoning fabricated from non-thinking QA answers. Their
[preparation manifest](../data/calib/nvfp4-inputs-20261115/manifest.json) archives
the authoring script, source tool schemas, family counts, and exclusion hashes.
Tool prompts cover weather, documentation/code search, simulated tickets, and
fictional currency conversions. Reasoning uses eight parameterized families:
cache capacity, dependency scheduling, release alignment, replica quorums,
timestamp correction, SQL cardinality, batch allocation, and Boolean deployment
constraints. This is deliberately limited coverage, not a broad reasoning corpus.

The existing teacher did not have a tool parser enabled and was left untouched.
A task-owned temporary teacher on GPU 2 used the pinned target/image with
`--tool-call-parser qwen3_xml`, `--enable-auto-tool-choice`, and
`--reasoning-parser qwen3`; its
[deployment and stop record](../data/calib/nvfp4-inputs-20261115/teacher-deployment.json)
and log are preserved. It has been stopped and GPU 2 released.

The [frozen conversation manifest](../data/calib/nvfp4-inputs-20261115/conversation-manifest.json)
links every generation run and records:

- 48 structured tool-call records and 16 no-tool answers, with actual argument
  names/types/required fields checked against the schemas. No tools were executed.
- Ten initially capped no-tool answers were regenerated with a larger budget.
  One final tool-group answer still reached its 2,048-token generation cap.
- All 64 reasoning records have explicit teacher-generated reasoning. Ten reached
  the 3,072-token generation cap; one has no final answer after its reasoning.
  These prefixes are retained for activation calibration, not accepted as graded
  or complete training answers. Calibration truncation is reported separately.

The [integrity report](../data/runs/calibration-nvfp4-20261115/validation.json)
verifies 512 unique user prompts, zero exact overlap with **1,581 excluded prompt
hashes**, input/token hashes, token bounds, and exact re-tokenization of all 128
tool/reasoning sequences with the pinned template. Real tool arguments exposed
and fixed a JSON-string/object template incompatibility; 24 focused tests passed.

Production-mode export preflight passed. **The full NVFP4 export has not been
started**; use the export command above as the next step. This calibration meets
the preparation contract, but does not establish quantized quality, native Spark
compatibility, MTP engagement, or speedup. Both the calibration run and its
source conversations are in the Spark backup scope; rerun backup to include them.

References: [ModelOpt PTQ](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq)
and [unified HF export](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html).
