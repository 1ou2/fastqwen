# Plan for tomorrow — Tuesday 19/10

**Objective of the day:** leave with a launched (or finished) B1 generation run plus its manifest, and have the SFT harness ready to use the output the moment it lands. Nothing else.

**Already frozen (no work needed):** tokenizer contract + artifacts, the 30 demo prompts, 60 validation seeds, the exclusion manifest (290 hashes), and `generate_dataset.py` with its guards.

---

## 08:45–09:15 · Phase 0 — access inventory

Do this before touching any data; it is where cluster days die.

- [ ] 4× H100 allocated, visible (`nvidia-smi`), driver/CUDA versions noted
- [ ] Docker + `nvidia-container-toolkit` working; **pull the inference image and record its digest**
- [ ] Check the image serves a 27B Qwen3.8 on Hopper (arch support is not guaranteed by version alone)
- [ ] Disk ≥ 300 GB free on the generation volume; HF token valid
- [ ] **Start the teacher download immediately** (54 GB) and the student (≈5 GB) in the background — it runs while you do Phase 1

**Gate P0:** image starts and answers one prompt. If not, the day's plan shifts to image work only.

---

## 09:15–10:30 · Phase 1 — freeze the inputs (CPU/network only, runs beside Phase 0)

Order matters here: **validation first, generation shards second**, because the generator enforces the manifest.

1. **Write `scripts/sample_corpora.py`** (~1 h) — samples N rows per source, normalizes to `{id, slice, prompt}`, dedupes, hashes, writes shards. Must refuse prompts already in the exclusion manifest.
2. **Extend validation to ≥400** — sample ~340 extra prompts from the S2/S3 sources, keep the 60 authored seeds, rebuild with `python -m scripts.build_eval_datasets`, confirm `contamination: []` and the new hash count.
3. **Build the S1–S3 generation shards**, then run the generator in `--dry-run` to confirm: no contamination, slice counts, worst-case token estimate.

**Source shortlist (verified today, 10 min to confirm each on the cluster):**

| Slice | Source | Notes |
|---|---|---|
| S1 EN | `HuggingFaceH4/ultrachat_200k` | parquet, 1.6 GB, 200k dialogues — **preferred over OpenHermes** (67 GB single JSON) |
| S2 | `ise-uiuc/Magicoder-OSS-Instruct-75K` | 605 MB JSONL, 75k code instructions |
| S3 | `Salesforce/xlam-function-calling-60k` | 472 MB, **gated**: accept the terms before the day starts, or fall back to `glaiveai/glaive-function-calling-v2` |
| S1 FR | *unresolved* | no clean general FR instruction set found. Decide: a French instruction dataset you trust, or FR prompts authored from a template bank. **This is the only genuinely open question.** |

**Gate P1:** `--dry-run` passes with 0 contamination and a shard count you trust.

---

## 10:30–11:15 · Phase 2 — smoke test (the most important 45 min)

- [ ] Start the teacher: **2 replicas × TP=2** (not 1 × TP=4) — 27B BF16 is 54 GB, so two 27 GB halves per replica leave room for KV and double aggregate throughput
- [ ] Run `generate_dataset --limit 500 --shard-index 0` end-to-end
- [ ] Verify: manifest fields complete (image digest, revision, template SHA `c3cf9e34…`), error file empty, empty-answer rate < 1 %, and **measure real tokens/s**
- [ ] Kill it mid-flight and rerun: the resume must skip exactly the completed ids

**Gate P2:** tokens/s measured → recompute the wave size before launching.

---

## 11:15–11:45 · Phase 3 — size the wave and launch

Arithmetic to finalize on the spot: `tokens_target ÷ measured_tok/s = hours`. Budget ~6 h of generation today, keep the rest for overflow.

Starting proposal: **S1 ≈ 30 M, S2 ≈ 8 M, S3 ≈ 4 M output tokens** (~42 M total, 55–75 k prompts), thinking **off**, `reasoning_effort=low`, `max_tokens=1024`, concurrency 128 per replica. Extend tomorrow if throughput is better than expected — the log is resumable, so there is no penalty for starting modest.

Launch both shards (`--shard-count 2`), one per replica.

---

## 11:45–19:00 · Phase 4 — generation runs; write the next two harnesses

Nothing GPU-bound here, so use the window:

- [ ] `scripts/train_draft.py` — SFT on (prompt, teacher content), reasoning kept separate, template-hashed, checkpointing every N steps, resume from checkpoint
- [ ] `scripts/offline_acceptance.py` — acceptance rate + γ sweep on the extended validation set, reasoning/answer split, H100-only (§Stage C)
- [ ] Smoke `train_draft.py` on 50 steps with the first ~2k generated records: loss decreasing, checkpoint written, resume works
- [ ] **Update `benchmarks/results.md`** with the two known deviations: audio tokens are effective-identical (§3.4 refinement), and the validation set size vs the planned 2,000

Monitor every ~45 min: error file growth, empty answers, throughput drift, `finish_reason=length` rate (if high, `max_tokens` is clipping answers — record it, don't silently raise it mid-run).

---

## 19:00–19:30 · Phase 5 — decide the night

| Condition | Action |
|---|---|
| Generation done **and** SFT smoke passed | Launch SFT overnight (6–10 h on 1 H100) → G2 ready tomorrow morning |
| Generation still running | Let it run; SFT tomorrow. Do not start SFT on a partial set |
| Generation done, SFT not ready | Stop at the manifest; finish SFT tomorrow morning |

---

## Contingencies

| Symptom | Response |
|---|---|
| Image won't serve the model | Spend the day on the image only; the data phases are already frozen and reusable |
| xlam gated / terms not accepted | glaive-function-calling as S3 substitute; record the swap in results.md |
| Tokens/s far below estimate | Cut S1 first (it is the volume, not the signal), keep S2/S3 whole |
| High `finish_reason=length` | Fix `max_tokens` **before** the bulk run, not during |
| Empty answers > 1 % | Assume template/tool-schema mismatch; stop and inspect before burning GPUs |
| Preemption | Rerun the same command; shards resume idempotently |

---

## Do **not** do tomorrow

NVFP4 calibration, S4 reasoning traces, any Spark-side measurement, and any tuning of the draft. The trip's deliverable is the dataset + a usable draft, not a final number.

## What "done" looks like

Shards with hashes · extended validation + rebuilt manifest · generation output + `manifest.json` (image digest, revision, template SHA, token totals) · `train_draft.py` and `offline_acceptance.py` smoke-tested · `results.md` updated · optionally one draft checkpoint.