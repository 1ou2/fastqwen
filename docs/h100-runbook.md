# H100 runbook — B1 dataset generation

Everything needed for this stage is in git. The DGX Spark is **not reachable**
from the cluster: model weights, the vLLM image and the generation output are
the only things fetched or produced here that do not come from the repository.

## 1. Clone and environment

```bash
git clone <repo> fastqwen && cd fastqwen
python3 --version                 # >= 3.10; 3.12 is what the Spark runs
uv pip install httpx pytest       # httpx is already in the vLLM image
```

No package install is needed: the code is imported from the repository root.

## 2. What the repository already pins

| Artefact | Why it is committed |
|---|---|
| `benchmarks/validation/manifest.json` | **the exclusion list** — the generator refuses to start without it |
| `benchmarks/compat/tokenizer_contract.json` | tokenizer/template hashes, used for the run manifest off-machine |
| `data/prompts/*.jsonl` | authored sources of the 30 demo prompts and the validation seed |
| `benchmarks/quality/quality-200.jsonl` | checked for contamination, and the future quality suite |

`cache/` is gitignored on purpose. The generator falls back to the committed
contract artifact for the chat-template hash, so nothing is silently null.

## 3. One-time downloads (start these first — 54 GB)

```bash
hf download Qwen/Qwen3.8-27B --local-dir "$HF_HOME/Qwen3.8-27B"
hf download Qwen/Qwen3.5-2B  --local-dir "$HF_HOME/Qwen3.5-2B"
```

Record the commit SHA of each revision (`hf models info … --format json`).

## 4. Serve the teacher

Two replicas of TP=2 beat one TP=4 here: 27B BF16 is 54 GB, so a 27 GB half
per replica leaves room for KV, and the two servers scale independently.

```bash
vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 2 --port 8000 &
vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 2 --port 8001 &
```

Freeze the power/clock settings before measuring anything, and record the
**image digest** — it is a required argument below.

## 5. Validate before spending GPU time

```bash
python -m scripts.generate_dataset \
    --prompts data/prompts/shards/s1.jsonl \
    --dry-run
```

Check: `exclusions : 290`, non-null `template sha` (`c3cf9e34…`), non-null
revision, slice counts. It exits non-zero if any prompt is contaminated.

## 6. Smoke test (500 prompts, one small shard)

```bash
python -m scripts.generate_dataset \
    --prompts data/prompts/shards/s1.jsonl --limit 500 \
    --base-url http://127.0.0.1:8000 --concurrency 128 \
    --image-digest sha256:... --teacher-revision <sha> \
    --run-id smoke --out-dir /data/runs
```

Then verify, before the bulk run:

- `manifest.json` has image digest, revision, template hash, token totals;
- `.errors.jsonl` is absent or tiny;
- empty answers < 1 %;
- measure tokens/s, then size the wave: `tokens ÷ tok/s = hours`.

Resume check: kill it and rerun the **same command** — completed ids are skipped.

## 7. Launch (one shard per replica)

```bash
# replica 1
python -m scripts.generate_dataset --prompts data/prompts/shards/{s1,s2,s3}.jsonl \
    --shard-index 0 --shard-count 2 --base-url http://127.0.0.1:8000 ...
# replica 2
python -m scripts.generate_dataset --prompts data/prompts/shards/{s1,s2,s3}.jsonl \
    --shard-index 1 --shard-count 2 --base-url http://127.0.0.1:8001 ...
```

`--thinking` stays **off** for S1–S3 (predictable token budget); use it only for
an S4 reasoning slice. `reasoning_effort=low` is frozen.

## 8. What to bring back

The run directory (`/data/runs/<run-id>/`): `shard-*.jsonl`, `manifest.json`,
`*.errors.jsonl`. Archive it with its manifest; the dataset is content, the
manifest is its provenance. Never commit the output — `benchmarks/datasets/`
is ignored.

Do not run SFT, acceptance or quantization here unless the generation finished
early; those belong to the next session.
