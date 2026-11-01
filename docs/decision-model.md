# Prompt classification with LiquidAI/d1-2B

## Purpose and model interface

Use [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) to route candidate
prompts before generating expensive teacher completions. Unlike a chat model,
d1 returns typed decisions and probabilities from a forward pass. There is no
temperature, thinking flag, completion cap, or generated explanation to parse.

The implementation follows the model card's Transformers interface:
`AutoModel.from_pretrained(..., trust_remote_code=True)` and
`model.system_one_batch([(state, questions), ...])`. Each prompt is a separate
state; prompts are not concatenated into one conversation. The service is our
own HTTP adapter, **not** Liquid's hosted API or an OpenAI/vLLM endpoint.

Implementation:

- [Policy and HTTP client](../qwenbench/decision.py)
- [JSONL classification runner](../scripts/classify_prompts.py)
- [Transformers HTTP service](../scripts/serve_decision_model.py)
- [Optional service dependencies](../requirements-decision.txt)
- [Mocked tests](../tests/test_decision.py)

## Client algorithm

1. Read one or more prompt JSONL files with `id`, `slice`, and `prompt` fields.
   Preserve the original prompt and any extra fields. Reject empty ids/prompts,
   duplicate ids, normalized duplicate prompts, invalid slices, and prompts in
   the frozen evaluation exclusion manifest before sending any classification.
2. Read `/health`. Require `LiquidAI/d1-2B`, a full pinned model commit SHA, and
   the same classification-policy SHA as the client. Check the server batch limit.
3. Send batches, in input order, to `POST /classify`. Default batch size is 16,
   with one HTTP request in flight. Each state is the original prompt string.
   Two named `choice` questions assess task routing and suitability in the same
   model call. Every candidate is model-classified; the sampler's regexes are
   not applied again here.
4. Validate the response count and policy hash. Require all expected probability
   labels, finite probabilities in `[0, 1]`, a distribution summing approximately
   to one, and a selected choice matching a maximum probability. Malformed
   responses and HTTP errors stop the run; they never become accepted S1 rows.
5. Compute confidence from the selected label's probability and margin from the
   two highest probabilities. Route confidence below `--min-confidence` or
   margin below `--min-margin` to `UNCERTAIN`. Apply both gates independently to
   suitability before exporting. Defaults are 0.8 and 0.2; these are initial
   operating settings, **not calibrated guarantees**.
6. Append an audit entry and one destination entry per prompt, flushing files
   after each batch. Record input hashes, exclusion hash, exact questions,
   thresholds, model revision, and server library versions in the run manifest.

### Routing policy

| Label | Requested task |
| --- | --- |
| S1 | General Q&A, advice, summaries, translation, non-code explanations, creative writing |
| S2 | Code generation/explanation/debugging, SQL, shell commands, software architecture, configuration or deployment |
| OTHER | Explicit external tool execution or tasks outside the two slices; not automatically exported as S3 |
| UNCERTAIN | Ambiguous intent or insufficient context |

Classify the requested deliverable, not isolated keywords. WordPress theme
implementation, S3 caching configuration, Kotlin marketplaces, D3 visualization
implementation, and Ember deployment belong to S2. A beginner explanation of
cloud computing, film script, or onboarding program can remain S1. English and
French use the same policy. Instructions embedded in candidate prompts are data,
not instructions for the classifier; this policy instruction is not a security
guarantee against adversarial inputs.

Suitability is separate: `suitable`, `oversized`, or `incomplete`. The policy
assumes an **8192-token teacher completion budget**. A ten-page screenplay can
be S1 and oversized; a complete production software project can be S2 and
oversized. Change the policy if the teacher budget changes, then use a new run.

### Output files

Each run requires a new output directory:

- `decisions.jsonl`: all input rows, normalized prompt hashes, raw model answers
  and usage, computed routing, suitability, export decision, and policy hash.
- `S1.jsonl` and `S2.jsonl`: only confidently routed, confidently suitable rows.
  These retain ids and extra fields, with `slice` updated to the destination.
  Ids are deliberately not renamed, even if an old id starts with `s1-`.
- `review.jsonl`: all non-exported rows, including OTHER, uncertain decisions,
  incomplete context, and oversized requests. Useful S2 candidates remain in
  the audit/review output even when unsuitable for the current budget.
- `manifest.json`: provenance, status, timestamps, and destination counts.

Paths supplied to the runner are relative to the current working directory.
There is no automatic retry, resume, or cross-run cache. On a failure, inspect
the partial audit and manifest, fix the cause, and rerun into a **fresh** directory.
Do not treat a partial run as a completed classification dataset. Ordinary
exceptions mark the manifest failed; abrupt termination can leave status running.
Keep the audit and manifest together when moving or reusing decisions.

## Deploy on the H100 host

### Isolated environment and model download

Use a free GPU. If both TP=2 teacher replicas occupy all four H100s, stop one
replica first or classify in a separate phase. Do not assume GPU 2 is available
and do not change the existing teacher environment.

For a system Python 3.12 on Debian/Ubuntu, install the matching development
headers and C build tools (run yourself, or ask the cluster administrator):

```bash
sudo apt-get install python3.12-dev build-essential
```

Triton compiles its CUDA driver helper on the first inference call, even when
the service does not explicitly use `model.compile()`. A Python virtual
environment and pip-installed packages do not supply missing system headers.
For other Python installations, use headers matching that interpreter.

From the repository root:

```bash
uv venv --python 3.12 .venv-decision
uv pip install --python .venv-decision/bin/python -r requirements-decision.txt
.venv-decision/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The model card currently requires `transformers>=5.19`, plus Torch, torchvision,
and Pillow. Choose compatible Torch/torchvision CUDA wheels for the actual host
driver if the default installation is unsuitable. The service uses BF16 on CUDA
and FP32 with `--device cpu`; CPU is a functional fallback, not the H100 path.

Review the model's [license](https://huggingface.co/LiquidAI/d1-2B/blob/main/LICENSE)
and custom Python code before loading: `trust_remote_code=True` executes code
downloaded from the repository. Select a reviewed full 40-character commit SHA
from the model repository's Files/history view and set it explicitly:

```bash
export D1_REVISION='<reviewed 40-character commit SHA>'
hf download LiquidAI/d1-2B --revision "$D1_REVISION"
```

Otherwise download without the revision and retreive it later 
```bash
hf download LiquidAI/d1-2B
export D1_REVISION="$(
  .venv-decision/bin/python -c '
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
print((Path(HF_HUB_CACHE) / "models--LiquidAI--d1-2B" / "refs" / "main").read_text().strip())
'
)"

printf 'Decision model revision: %s\n' "$D1_REVISION"

```

Replace the placeholder before running. `hf download` uses the standard HF cache;
no `--local-dir` is needed. Download and serving must use the same cache settings
and user. `HF_HOME`/`HF_HUB_CACHE` work through the Hugging Face libraries; this
service does not hardcode a cache directory. For the previously observed Xet
authentication problem, retry with `HF_HUB_DISABLE_XET=1` rather than deleting
the cache. Record the reviewed SHA and freeze the installed environment for
production; dependency lower bounds alone are not a reproducibility lock.

### Start the service

Example using GPU 2 **only if it is free**, and port 8010:

```bash
CUDA_VISIBLE_DEVICES=2 .venv-decision/bin/python -m scripts.serve_decision_model \
  --revision "$D1_REVISION" --device cuda --port 8010 --max-batch-size 32
```

The visible GPU is addressed as `cuda` inside the process. The model loads once
before the server starts. One Uvicorn worker owns the model; a lock serializes GPU
calls. Do not add multiple workers, which would load multiple copies. This is
baseline eager inference without compilation or cross-request dynamic batching;
published compiled benchmark rates are not promised by this adapter.

Check that the service has loaded in another terminal:

```bash
curl --fail http://127.0.0.1:8010/health
```

`/health` does not execute inference or verify Triton's runtime build dependencies.
If the first `/classify` request fails with `fatal error: Python.h: No such file
or directory`, inspect the server interpreter's header directory:

```bash
.venv-decision/bin/python -c 'import sysconfig; from pathlib import Path; header = Path(sysconfig.get_path("include")) / "Python.h"; print(header, header.is_file())'
```

Install the matching development headers as above and check that this prints
`True`. Restart the decision server and retry classification using a fresh
output directory, such as `data/runs/d1-classification-smoke-500-v2`; the failed
run directory is retained for inspection and cannot be overwritten. No model
redownload or environment recreation is needed for this missing-header fix.

Health reports the model SHA, policy SHA, dtype, device, versions, and batch limit.
The server refuses mismatched policies (409), invalid batches (422), and prompts
over 32,000 characters or batches over 128,000 characters (413). These are service
memory safeguards, not the model's token context limit. No prompt is silently
truncated. Reduce batch size for aggregate-limit failures; handle individually
oversized inputs separately. Character limits do not guarantee a batch fits GPU
memory, so start small and measure with representative lengths.

The default bind address is loopback. This service has no authentication; do not
expose it publicly. For a remote client, prefer an SSH tunnel:

```bash
ssh -L 8010:127.0.0.1:8010 gabriel@dlxxxx
```

### Classify a candidate shard

The client only needs the repository's HTTPX dependency, not Torch/Transformers.
In the existing project environment:

```bash
python -m scripts.classify_prompts \
  --input data/prompts/shards/s1-ultrachat-filtered-500-v2.jsonl \
  --output-dir data/runs/d1-classification-smoke-500 \
  --base-url http://127.0.0.1:8010 \
  --batch-size 16 --min-confidence 0.8 --min-margin 0.2
```

Repeat `--input` for multiple shards. For production sampling, classify an
oversized candidate pool and then select the required number of accepted S1
rows. Existing regex-filtered inputs cannot recover valid S1 prompts already
discarded by the sampler; an unfiltered candidate pool is preferable when
evaluating this classifier as a replacement.

Only completed `S1.jsonl`/`S2.jsonl` exports should feed teacher generation.
The teacher generator still performs its own exclusion check. Classification
does not replace output QA for truncation, empty outputs, or repetition loops.

## Validation and calibration

```bash
uv run --with pytest --with fastapi python -m pytest tests/test_decision.py
```

These tests verify the HTTP adapter, probability gates, request limits, and JSONL
exports using mocks; they do not certify the checkpoint's semantic accuracy or
GPU compatibility. Before bulk classification, manually label 200-300 examples,
including EN/FR, known technical misses, benign technology discussion, fragments,
and oversized creative requests. Measure S2 leakage into accepted S1, valid S1
removed, suitability errors, and review rate. Tune thresholds on a development
subset and report performance on a separate held-out subset. Use a new run
directory for each policy or threshold change.