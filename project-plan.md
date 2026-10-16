# Project Plan — "I made my model speak 10x faster"

> YouTube video: the inference race. Making Qwen3.8-27B comfortable locally
> on DGX Spark via quantization, distillation of a home-made draft model and
> speculative decoding — all measured, step by step, in vLLM.
>
> Project directory: `/home/gabriel/dev/fastqwen`
> Start date: 22/08/2026
> Revision: 16/10/2026 — addition of §6.5 (baseline measurement suite); unit 1 implemented (`qwenbench/`, `scripts/probe.py`, `tests/`); repository switched to English (official language)
>
> **This document is the reference plan for the implementation.** Any difference
> found along the way is documented in `benchmarks/results.md`, not here.

---

## 1. Project objective

Original YouTube video on LLM inference optimization techniques.

- **Narration**: Qwen3.8-27B is excellent but too slow to run
  comfortably locally. We optimize it step by step, with a visual benchmark
  at each chapter.
- **Title metric**: two distinct figures, never conflated —
  1. **decode tokens/s** (batch=1, fixed generation length,
     `reasoning_effort` frozen): the pure speed measurement;
  2. **total time to the complete final answer** at identical
     `reasoning_effort`: this is what the audience feels, and this is the figure
     that will carry the title. Any variation in the number of reasoning tokens
     is explicitly mentioned on screen, otherwise the gain is fake.
- **Constraints**: evaluations systematically on DGX Spark; H100 (cluster
  of 4) only for training; max 3 consecutive days of GPU.
- **Scope**: text only. The 27B's vision encoder is neither quantized
  nor evaluated.

### 1.1 Learning objectives (personal success criteria)

The project is as much a teaching aid as a video. **vLLM is the
single engine of the project**, precisely to build up skill on it: every
mechanism below must be understood, manipulated, and explained in the video.

| Mechanism | Where it is worked on |
|---|---|
| PagedAttention, KV cache management, unified memory | Step 0, Stage D |
| Continuous batching, `--max-num-seqs`, saturation | Step 0, Stage D |
| Chunked prefill, TTFT, prefix caching | Stage D |
| CUDA graphs, `--enforce-eager`, determinism | Protocol (§6) |
| Offline quantization (NVFP4, FP8) and kernels | Stage A |
| Quantized KV cache | Stage D |
| Speculative decoding: verification, acceptance rate, γ | Stage C |
| Multi-token prediction (MTP) native | Stage C |
| SFT and logit KD distillation | Stage B |
| Profiling: Prometheus metrics, `vllm bench` | Protocol (§6) |

---

## 2. Hardware

| Machine | Use |
|---|---|
| NVIDIA DGX Spark (GB10, sm_121, unified memory 128 GB, ~273 GB/s) | All benchmarks and inferences (this is the audience's machine) |
| 4× H100 | Dataset generation, calibration, training of the draft model |

---

## 3. Models selected

| Role | Model | Status |
|---|---|---|
| Target (video hero) | **Qwen3.8-27B** | ✅ day-0 support (vLLM, SGLang, llama.cpp), already installed |
| Built-in "free" draft | **Native MTP heads of the 3.8-27B** | ✅ embedded multi-token prediction (cf. §3.3) |
| Student / draft model | **Qwen3.5-2B initialized then distilled** | ✅ tokenizer compatibility verified |

### 3.1 Characteristics of the target that constrain everything else

- **27B dense** (no MoE): every parameter is read at every token → decoding is
  purely **memory-bound** (§5).
- **Hybrid attention stack**: 64 layers alternating 3 *Gated DeltaNet* blocks
  for 1 *gated-attention* block. Direct consequence: the KV cache is much
  smaller than a classic transformer (only one layer in four has one),
  which reinforces the thesis "the bottleneck is reading the weights".
- **Native context 262,144 tokens** (~1M with YaRN): memory pressure comes
  mainly from the weights and the prefill, not from the cache.
- **Hybrid thinking, enabled by default**, driven by `reasoning_effort`
  (`low` / `medium` / `xhigh`). There is no longer an "Instruct" variant.
- **Vision**: multimodal encoder present, out of scope.
- **Apache 2.0 license**.
- **The 3.8 family has no model below 27B**: there is no native little brother
  to use as a draft. This is the justification for Stage B.

### 3.2 Why Qwen3.5-2B as student

The 3.8 only exists in 27B (and above). The smallest model from the same
tokenizer lineage is in the previous generation: **Qwen3.5-2B**. It serves
as the initialization point, then it is distilled from the 27B itself.

### 3.3 The MTP heads: the built-in draft

The 27B embeds **multi-token prediction (MTP)** heads: the model
natively predicts several tokens ahead, which gives "free" speculative
decoding (no training, no H100 GPU consumed). Community stacks on GB10
even run as `NVFP4 + MTP3`.

MTP is therefore a **mandatory arm of Stage C**, not a bonus. It provides
the honest comparison line: if the home-made distilled draft does not beat the
native MTP, that is the video's result, and it is interesting to explain.

### 3.4 Tokenizer compatibility (verified on 22/08/2026)

Script: `check_tokenizer_compat.py` (repo root). This script served to
**validate the feasibility** of the student choice; it must be hardened (§3.5)
before serving as a reproducible guardrail in the pipeline.

Results:

- `vocab.json` and `merges.txt` bit-for-bit identical; tokenizer core identical
- Base vocabulary: 248,044 tokens on both sides
- Functional encoding strictly identical (FR tests, quotes, emoji,
  multilingual, chat tokens)
- Only differences: 7 audio/TTS special tokens on the 3.8 side (IDs 248070-248076,
  **beyond** the base vocabulary, therefore appended with no ID shift) and the
  `chat_template`

**Decision**: compatible. Initialize the student from Qwen3.5-2B as is, without
vocabulary extension. Use the **3.8 chat template everywhere** where a
template is required (dataset generation, final inference, conversational
evaluations) — and **never** for the perplexity computation (§6, point 4).

### 3.5 Compatibility contract to lock down

Speculative decoding requires an **exact** token ↔ ID match. Before
Stage B, `check_tokenizer_compat.py` must assert, on **pinned HF revisions**,
the following hard requirements:

1. Equality of the set of (token → ID) pairs over the entire shared vocabulary.
2. Identical IDs for `bos`, `eos`, `pad` and the chat tokens
   (`im_start`, `im_end`, and any special used by the 3.8 template).
3. Any additional token of the 3.8 is strictly beyond the size of the
   base vocabulary: no shared ID is shifted.
4. `apply_chat_template(...)` with the 3.8 template yields identical IDs
   on both tokenizers.
5. A missing tokenizer file is a **failure**, never a warning.

**Tolerated** differences, listed explicitly: `chat_template`,
`clean_up_tokenization_spaces`, audio/TTS tokens. The script's verdict must not
depend on these cosmetic differences.

To be confirmed under real conditions: vLLM validates tokenizer compatibility
when loading a `draft_model`; note the exact message in case of failure.

---

## 4. Step-by-step flow (input → output)

```
Qwen3.8-27B (BF16, hybrid thinking, native MTP heads)
  │
  ├──> A: Quantization ─────────────────────────┐
  │     FP8 → NVFP4 4-bit → aggressive tier     │
  │                                             ├──> C: Speculative decoding
  ├──> B: Draft distillation ───────────────────┤     native MTP · raw draft
  │     (teacher = 3.8-27B itself, on H100)     │     distilled draft · MTP+draft
  │                                             │
  └──> D: KV cache & serving ───────────────────┘
        (prefill, cache, CUDA graphs)
                    │
                    └──> E: Video climax
```

### Step 0 — Prerequisites & baseline

- **Input**: Qwen3.8-27B BF16 on DGX Spark
- **Prerequisites to validate before any measurement** (blocking):
  1. **Working vLLM image on GB10 (sm_121)**: standard vLLM does not
     install as is on Spark; use a community image
     **pinned by digest** (reference: `eugr/spark-vllm-b12x`) or build
     from `eugr/spark-vllm-docker`. Note digest + exact vLLM version: this is
     the measurement core of all reproducibility.
  2. **FP8 support** on GB10 confirmed by a short run.
  3. **NVFP4 support** (loading a 4-bit checkpoint + generation) confirmed.
  4. **MTP mode**: identify the exact `--speculative-config` key supported by
     this vLLM version and verify with a short run. (`qwen3_5_mtp` exists in
     vLLM 0.31.0 — §6.5.6 — to be reconfirmed on the pinned image.)
- **Action**: full benchmark with the uniform protocol (§6), with
  `reasoning_effort` **frozen and documented**.
- **Output**: reference figures — everything will be compared to this.

**Choice of inference runtime: vLLM everywhere, for everything.** A single engine for
the baseline, the quantizations, the benchmarks and the speculative decoding.
Two reasons: (1) otherwise we measure the runtime and not the optimization;
(2) **vLLM is the engine I want to master** — it is a learning objective
of the project, not a default choice. Concretely, we use
`vllm bench latency|throughput|serve` and the Prometheus endpoint for measurements,
with a thin wrapper (`scripts/bench_spark.py`) rather than a home-made harness.

### Stage A — Quantization

Three tiers, from safest to most aggressive.

1. **FP8**: the simplest. Qwen publishes `-FP8` repos for all its open
   models: we start from the official checkpoint, without conversion. Almost no loss.
2. **NVFP4 4-bit**: the main tier, and the **native format of Blackwell**
   (FP4 with FP8 scale per micro-block). This is the GB10 hardware path:
   so this is where the real speed gain is. Produced offline with
   NVIDIA ModelOpt / `llm-compressor`, then loaded by vLLM (modelopt format
   detected from the checkpoint config).
3. **The limit**: a genuinely more aggressive method (3-bit / 2-bit HQQ-style,
   or degraded mixed recipe) to show the breakage. If the degradation is not
   spectacular, that is also a result to tell ("modern methods
   are robust down to 4-bit").

- **Output**: the same compressed 27B, measured at the 3 tiers
- **Expected gain**: ~3x decode speed, ~3.4x memory (56 GB → ~16 GB)
- **Insight**: quantization optimizes memory reads, not compute — the
  model remains a dense one generating token by token
- **Watch point**: the *Gated DeltaNet* blocks and the vision encoder are
  not standard `Linear`. If a tier does not load or breaks quality,
  the fallback recipe is a **mixed recipe**: keep these layers
  (and the ViT) in BF16/FP8, quantize only the attention projections and the
  MLP. This fallback is a pedagogical result, not a failure.

**NVFP4 calibration** (and any tier that requires one):

- **Mandatory mixed dataset**, ~512 sequences of 2048 tokens, breakdown:
  - general French text (Wikipedia FR, press, prose) — the bulk of the volume
  - **code** (Python, shell, JSON, diffs) — protects code generation
    abilities
  - **tool-calling / agentic**: tool-call trajectories, JSON
    schemas, refusals when no tool fits — protects exactly the
    abilities the protocol measures (§6, point 5)
  - a share of conversations with reasoning traces, since the model
    reasons by default
- **Contamination rule**: no prompt from the evaluation set
  (`benchmarks/`) appears in the calibration. Automatic verification.
- **Traceability**: hash of the calibration set recorded in the run manifest.

### Stage B — Distillation: creating the draft model *(on H100)*

The Qwen3.8 only exists in 27B: there is no compatible small native model.
So we build the missing draft model ourselves.

- **B1. Dataset generation** (the teacher is the Qwen3.8-27B itself)

  The dataset is built in **slices**, because the acceptance rate
  depends on the distribution: the draft must be good where the video
  demonstrates.

  | Slice | Content | Target volume |
  |---|---|---|
  | S1 | General conversational FR + EN (OpenHermes-2.5, UltraChat) | ~120M tokens |
  | S2 | **Code** (generation, correction, explanation instructions) | ~25M tokens |
  | S3 | **Tool-calling / agentic** (tool schemas, chained calls, refusals) | ~15M tokens |
  | S4 | Reasoning traces (optional, depending on budget) | +30-60M tokens |

  - Generation by vLLM on 4×H100, with the 3.8 chat template
  - `reasoning_effort` **frozen** for generation too: this is what makes the
    token budget predictable (thinking active by default would multiply the
    volume by 2 to 4 and blow up the H100 slot)
  - Duration: ~14-24 h on the 4×H100
  - **Offline validation set**: ~2,000 prompts representative of
    slices S2/S3 + prompts from the final demo, held out and **never seen in
    training** — they serve to predict the acceptance rate on H100
    before consuming Spark time (§6, point 7)

- **B2. SFT of the student**
  - Input: Qwen3.5-2B initialized as is + B1 dataset
  - Output: **draft model statistically aligned with the 27B**
  - Duration: ~6-10 h on 1×H100
  - **Logit KD option**: +5 h, to be triggered only if gate G2
    (§7) is not passed

### Stage C — Speculative decoding: the convergence

Four arms, measured with the same protocol:

- **C1. Native MTP alone** — the 27B's MTP heads, γ ∈ {1,3} (reference: community
  stacks use `MTP3`). This is the "free" arm.
- **C2. Undistilled Qwen3.5-2B draft** — result deliberately shown:
  disappointing gains, low acceptance rate. This is the control that **proves**
  the usefulness of Stage B.
- **C3. Distilled draft** (from Stage B) — the heart of the video.
- **C4. MTP + distilled draft** — if the vLLM version allows it, otherwise
  documented as a tool limitation.

**γ protocol**: γ is not guessed on Spark. The theoretical gain of
speculative decoding is

```
       1 - α^(γ+1)
S  =  ─────────────────        α = acceptance rate, c = draft cost
       (1 - α) · (γ·c + 1)         expressed as a fraction of the target cost
```

`α` and `c` are measured **offline on H100** (`scripts/offline_acceptance.py`,
sweep γ ∈ {1,3,5,7}), and only the 2 best candidates are validated on
Spark. The acceptance rate is measured **separately on the reasoning segments
and on the final answer**: this is where a 2B student falls off, and
this is a novel measurement that is interesting to show.

**Exactness verification (mandatory, and very visual)**: in greedy, the
speculative decoding with verification is **exact**. We therefore validate C1-C4 by
a **token-by-token identity test** against the target alone, on the 30 qualitative
prompts. This is the proof that the speedup is free in quality — and
it replaces a full quality campaign, which would be wasted time.

- **Output**: quality of the 27B, multiplied speed, optimal γ, draft cost
- **Key insight**: speculative decoding only speeds up if the draft predicts
  like the target — hence the need for distillation
- **Spark bonus**: batch=1 decoding there is memory-bound, ideal conditions:
  verifying γ+1 tokens costs ≈ a single read of the target weights

### Stage D — KV cache & serving *(the step missing from throughput)*

Speculative decoding attacks decoding. The other half of the problem is
prefill and cache management. It is also the project's vLLM playground.

Optimizations to test one by one, in **single-variable A/B**, on the
winning configuration from Stage C:

1. **FP8 KV cache** (`--kv-cache-dtype fp8`) — note: with hybrid
   attention, only one layer in four has a KV cache; measuring the real gain
   is a result in itself.
2. **Chunked prefill** (`--max-num-batched-tokens`) — direct effect on TTFT
   in long context.
3. **Prefix caching** — decisive for repeated system / tool prompts.
4. **CUDA graphs** and `--enforce-eager` — speed / determinism trade-off.
5. **Continuous batching**: `--max-num-seqs` increasing, to plot the
   throughput vs latency curve (the Spark is single-user, but the curve is
   pedagogical).

**Rule**: any variant that does not hold within the noise (§6.1) is rejected and
documented as such. A rejected optimization is a publishable result.

- **Output**: the final serving configuration, and the breakdown
  "what comes from quantization / draft / serving"

### Stage E — Video climax

Final race on Spark: BF16 baseline vs full pipeline (27B NVFP4 + distilled
draft + serving config from Stage D), same question, stopwatch on
screen. Cumulative table of gains stage by stage.

- Same `reasoning_effort` on both sides, and **number of reasoning tokens
  displayed** next to the stopwatch
- **"Why not a smaller model?" control**: a reference line
  with a smaller model served at comparable throughput (3.5/3.6 family, MoE with
  a few billion active). This is the audience's first objection and
  it is closed for ~1 h of Spark.

---

## 5. The set: the physics of the problem

"Physics" segment of the video, and a guardrail against exaggerated
promises. The Spark has ~273 GB/s of unified memory bandwidth:

| Configuration | Weights read / token | Theoretical ceiling batch=1 |
|---|---|---|
| BF16 | ~56 GB | **≈ 5 tokens/s** |
| 4-bit (NVFP4) | ~16 GB | **≈ 17 tokens/s** |
| 4-bit + speculative decoding | ~16 GB, γ+1 tokens verified | **≈ 17 × tokens accepted per pass** |

Speculative decoding does not exceed this ceiling: it increases the number of
tokens produced **per read of the weights**. With α = 0.7 and γ = 3, we accept on
average ~2.5 tokens per target pass → ceiling ≈ 40 tokens/s. The community
figures on GB10 (19-40 tokens/s depending on content type) are consistent
with this calculation.

**Narrative consequence**: the "10x" of the title cannot come from
tokens/s alone. It comes from total time to the answer, and we must be explicit
about the share that comes from speed and the share that comes from the number of tokens.

---

## 6. Evaluation protocol (uniform at each stage)

On DGX Spark, same prompts and parameters at each measurement, everything goes through
vLLM. Measurements via `vllm bench` + `scripts/bench_spark.py` wrapper + Prometheus
metrics from vLLM.

1. **Speed**: decode tokens/s (batch=1), time-to-first-token (TTFT),
   and — for the video — total time to the complete answer.
   **3 runs per measurement** (thermal variance of the Spark), median and p95.
2. **Memory**: peak unified memory, sampled **outside vLLM**
   (`tegrastats`), not just the figure reported by the engine.
3. **Energy**: average power (W) during the run. Free to collect,
   very telling on video.
4. **Quality — perplexity** ⭐: held-out Wikipedia FR (~200k tokens). Metric
   most sensitive to fine quantization degradations:
   FP8 ≈ +0-2% PPL · 4-bit ≈ +2-5% · aggressive methods ≈ +20%+.
   - **Computation on raw text**: no chat template, no reasoning. Sliding
     windows of 512 tokens, to avoid context and cache effects.
   - Procedure documented in the script, not in a notebook.
5. **Quality — general abilities**: `benchmarks/quality/quality-200.jsonl`,
   **in addition to** the other levels:
   - 180 self-graded items, deterministic grading: gsm8k (80, numeric
     exact match), humaneval (40, code execution + tests), ifeval (40, strict
     instruction following), agentic_coding (20, execution)
   - ⚠️ The 20 `hard_reasoning` items (grading `manual`) are **excluded from the
     score** (not reproducible); reserved for possible qualitative
     demonstrations
   - **Prerequisite before use**: (a) the 20 `agentic_coding` items have no
     tests — their `reference` field is a placeholder (`"exec+asserts"`) that
     lets the model write its own asserts; they must be **rewritten
     with real external tests** before counting in the score, otherwise we
     measure self-grading; (b) any execution of generated code goes through a
     **sandbox** (network-less container, timeouts, disposable FS)
   - **Answer extraction**: reasoning is removed before grading
     (thinking is active by default); the extractor is tested on real outputs,
     not assumed
   - Statistical resolution: with 180 items, the standard error is ±3.7 pts and
     an **unpaired** comparison detects nothing below ~7-8 pts. The expected
     FP8/BF16 differences are within this noise → **the comparison is done item by
     item, paired (McNemar)**, not on the totals. Perplexity remains the fine
     metric.
   - English content: normal, degradations are correlated across languages;
     FR remains covered by wiki-FR perplexity and the qualitative prompts
6. **Quality — agentic** ⭐ (Qwen3.8 excels in this area, we must be able to
   verify that the optimization does not break its flagship abilities):
   - Frozen subset of **BFCL** (~300 cases, single/multi-turn tool calling,
     deterministic scoring)
   - **Home-made mini tool-calling bench**: 30-50 graduated tasks (simple call,
     choice among close tools, chained calls, refusal when no tool
     fits), fake tools, automatic scoring (valid JSON? right tool?
     right arguments?) — reproducible, free and very visual on video
   - Excluded (too heavy/noisy to detect fine degradation):
     SWE-bench, WebArena, OSWorld… cited as on-screen references only
7. **Speculative decoding only**: draft acceptance rate (vLLM
   metrics), **separately on the reasoning and answer segments**, plus
   the token-by-token identity test (§ Stage C).
8. **Qualitative**: 30 frozen diversified prompts, greedy answers
   (temperature 0, comparable between checkpoints), archived per stage —
   raw material for the video.

### 6.1 Determinism and comparability

- vLLM's greedy is not bit-for-bit deterministic depending on batch
  composition. Fix the composition, and if necessary `--enforce-eager` or
  batch-invariant kernels.
- Measure and **report the variance**: it is part of the result.
- If two configurations differ by less than 2 standard deviations, they are
  declared equivalent.

### 6.2 Spark measurement hygiene

Systematic warm-up run, frozen power mode, timestamp and
ambient temperature recorded, no other GPU process, a measurement point
< 30 min.

### 6.3 The chat template is not universal

The 3.8 template is used for: dataset generation, conversational
evaluations, qualitative prompts, final inference.
It is **not** used for perplexity (raw text, §6, point 4).

### 6.4 Run manifest (mandatory)

Each run writes `benchmarks/runs/<run>/manifest.json`: vLLM image digest,
vLLM version, HF revision of the models, quantization recipe and hash of the
calibration set, `reasoning_effort`, γ, MTP mode, training dataset
hash, git commit, exact command. `benchmarks/results.md` aggregates
everything else and serves as the basis for the video's final table.

### 6.5 Baseline — the reference measurement suite

The baseline is not a score: it is a **reference artifact** (BF16 logits,
perplexity, speed figures) against which each variant is diffed. It is
frozen **before** any optimization, and it is gate G0 (§7).

#### 6.5.1 We do not follow the model card's benchmark list

The figures on the HF page are **capability claims**, not
degradation measurement instruments: they are produced in BF16, with
non-reproducible agentic harnesses (Claude Code, 256K context, `avg@3`,
8 h timeouts) and paid judges. Three of them are moreover
**in-house and unpublished**.

| Card benchmark | Verdict |
|---|---|
| GPQA Diamond | ✅ **Kept** — public, 198 MCQ, deterministic, cheap. Ceiling effect (89.2) → insensitive used alone |
| IFBench | ✅ **Kept** — string verification; instruction following is the first ability quantization breaks |
| LiveCodeBench v6 | ⚠️ Public but **drifting** (contest data) + execution sandbox → freeze a snapshot, otherwise HumanEval+ / BigCodeBench |
| HLE | ⚠️ Judged by an LLM → not reproducible. Set aside (or frozen subset without a judge) |
| Terminal Bench 2.1 · SWE-bench Pro · DeepSWE 1.1 · NL2Repo-Bench · Agents' Last Exam | ❌ Agentic harnesses + API judges + 256K. **Cited on screen as reference only** (already §6, point 6) |
| QwenSWEBench · CoWorkBench · JobBench | ❌ In-house, unpublished → not runnable |
| VL table (MathVision, RealWorldQA, OmniDocBench…) | ❌ Out of scope (§1) |

**Corollary**: "matching the card figures" is not an objective. They
serve as a narrative marker, never as a target.

#### 6.5.2 The three criteria for selecting a benchmark

| Criterion | Practical consequence |
|---|---|
| **Sensitivity** | Must see a 1-3 % weight perturbation. At n=198, GPQA-D has a standard error of ±2.2 pts → **only** paired tests and PPL/KL detect 4-bit breakage |
| **Determinism** | No LLM judge, no agentic harness, no drifting data — otherwise the delta is not attributable to our change |
| **BF16 cost** | 27B BF16 = 54 GB ÷ ~273 GB/s ≈ **~5 tok/s ceiling** in decoding. A generative benchmark costs ~4× more on the baseline than on the 4-bit model → favor **prefill-only** measurements |

#### 6.5.3 The baseline suite (blocks R0-R5)

| Block | Content | Tool | GB10 cost |
|---|---|---|---|
| **R0** | **Public anchor**: GPQA-D + IFEval, temp 0, thinking off. Role: prove that the chat template, the thinking removal and the grader are correct — not match the card | `lm-eval` | ~1 h |
| **R1** | **Weight quality** ⭐: (a) PPL wiki-FR 200k, 512 windows (§6, point 4); (b) **KL + top-1 agreement vs frozen BF16 logits** (§6.5.5) | vLLM `prompt_logprobs` | minutes |
| **R2** | **Task quality**: quality-200 + GPQA-D + BFCL subset, **item-by-item (McNemar)** comparison, never on the totals | home-made grader + `vllm bench serve` (built-in **BFCL** dataset: `--bfcl-categories simple,live_simple,multiple`) | one night |
| **R3** | **Speed** (title metric): `vllm bench latency` batch=1 (in 512/4096, out 128/512); prefill ramp (1k/8k/32k/128k); `vllm bench serve` `--max-num-seqs` sweep 1/4/8/16/32; peak unified memory (`tegrastats`) + average power → tokens/joule. 3 runs, median + p95 (§6, points 1-3) | `vllm bench` + `bench_spark.py` | ~1 h |
| **R4** | **Speculation**: acceptance rate and length by position γ, sweep γ ∈ {1,2,3,5}, token-by-token identity test — **replayed on the quantized stack** (the MTP heads are quantized too, and the acceptance rate there is more fragile than accuracy) | `vllm bench serve` (built-in *spec bench* dataset: `--spec-bench-category`, `--spec-bench-output-len`) + Prometheus | ~1 h |
| **R5** | **Long context**: NIAH or RULER-lite at 32k/128k. **Absent from §6 today**; to be run as soon as the KV cache is quantized (Stage D) | home-made script | ~1 h |

**Mandatory arms** to cross with each block: `reasoning_effort`
(low/medium/xhigh) and thinking on/off are **explicit arms**, not implicit
settings, and the **number of reasoning tokens is reported with the
stopwatch** (§1, §6, point 1).

#### 6.5.4 Baseline time budget — the rule that makes measurement possible

Arithmetic: 54 GB of BF16 weights → ~5 tok/s theoretical, 3.5-4.5 tok/s real.
A generative evaluation in thinking is therefore over budget:

| Run | Estimated cost | Verdict |
|---|---|---|
| GPQA-D (198 items) at `xhigh` (~7.5k reasoning tokens/item) | **~100 h** | ❌ infeasible |
| The same in non-thinking (~600 tokens/item) | ~8 h | ✅ one night |
| quality-200 (180 self-graded items, ~400 tokens) in non-thinking | ~5 h | ✅ one night |
| PPL + KL (prefill only, 200k tokens) | **minutes** | ✅ at will |

**Rule**: the **differential** quality suites run in **non-thinking
(or `reasoning_effort=low`), temperature 0, `max_tokens` fixed**; thinking
mode is reserved for a frozen demonstration set of 20-50 items. A single
setting for all compared variants — internal consistency is what
matters, not resemblance to the card. Execution order: prototype the
harness on the 4-bit model (fast), then run the BF16 baseline only
**once**, at night.

#### 6.5.5 Three mandatory additions to protocol §6

1. **KL vs reference logits** (the most important). This is the standard
   quantization instrument: more sensitive than PPL and almost free. On a
   frozen input set (512-2048 tokens), we freeze the BF16 logits (top-k), then
   compute **KL / JS + top-1 agreement** for each variant. Decisive advantage: it
   **localizes** the breakage (by layer, by position), where an aggregate score
   drowns it. → `scripts/dump_ref_logits.py`, `scripts/kl_vs_ref.py`, artifacts in
   `benchmarks/ref_logits/`.
2. **A public anchor** (R0). A home-made set of 200 items with home-made numeric/code
   extractors can be silently broken. A GPQA-D + IFEval run
   in `lm-eval` recalibrates the harness — and verifies along the way that the
   hybrid architecture (Gated DeltaNet) does not force a slow or incompatible path in
   transformers.
3. **A long-context probe** (R5), as soon as the KV cache enters the
   scope.

#### 6.5.6 Facts verified on the machine (16/10/2026)

- vLLM **0.31.0** and transformers **5.20.1** in `.venv`; subcommands
  `bench latency|serve|throughput|sweep|startup|mm-processor`.
- **MTP**: the `qwen3_5_mtp` method **exists** in this version (as well as
  generic `mtp`) → form to test:
  `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`.
  To be reconfirmed on the pinned image (§5, prerequisite 4).
- Speculation Prometheus metrics already available:
  `vllm:spec_decode_num_accepted_tokens_total`, `..._num_draft_tokens_total`,
  `..._num_drafts_total`, `..._num_accepted_tokens_per_pos_total` → acceptance rate and
  length by position, without home-made instrumentation.
- Datasets **built into** `vllm bench serve`: BFCL (`--bfcl-categories`) and
  *spec bench* (`--spec-bench-category`, `--spec-bench-output-len`) → R2 and R4
  without a home-made harness (exact `--dataset-name` names to be confirmed by
  `vllm bench serve --help=all`).
- `lm-eval` **absent** from the venv → `uv pip install -U "lm-eval[vllm]"`.
- **BF16 weights present**: 18 shards ≈ 54 GB under
  `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B` → the baseline can
  start without downloading.
- Third-party comparison points already cached (**not** baselines):
  `ggml-org/Qwen3.8-27B-GGUF` (20 GB),
  `starkweatherdigital/qwen3.8-flash-next-nvfp4` (131 GB).
- **First reference measurement (16/10/2026, BF16, batch=1, temp 0)**:
  **4.36 tok/s** in decoding (median of 3 runs, dispersion 2.3 %), TTFT 0.51 s,
  ITL p50 0.24 s. Ceiling reminder: 54 GB ÷ ~273 GB/s ≈ 5.05 tok/s → **86 % of the
  memory ceiling**. Raw artifact: `benchmarks/runs/20261016-224112/probe.json`.
- **`reasoning_effort` is a real template parameter**
  (`resolved_reasoning_effort = reasoning_effort|default('xhigh')`, injected as a
  system message). But on a trivial prompt, `low` and `xhigh` give the same
  reasoning length: the test is only decisive on a hard prompt
  (§6.5.3, part `d` of the probe). Reasoning is produced **in English**
  even when the answer is in French.
- **The tool-call stream is not incremental**: vLLM only streams the
  `tool_calls` fragments once the complete block is received → TTFT ≈ total duration and
  `decode_s = 0`. Consequence: for the mini tool-calling bench (§6, point 6), the
  timing metrics must come from **non-streamed** requests.

---

## 7. H100 GPU budget

| Task | Estimated duration |
|---|---|
| NVFP4 calibration (mixed recipe) | ~1-2 h on 1×H100 |
| Dataset generation by the 27B (B1, slices S1-S3) | 14-24 h on 4×H100 |
| Slice S4 (reasoning traces), optional | +4-8 h on 4×H100 |
| 2B student SFT (B2) | 6-10 h on 1×H100 |
| Offline sweep of acceptance rate and γ | ~1-2 h on 1×H100 |
| Logit KD option | +5 h on 1×H100 |

**Consumption**: ~75-120 GPU-hours, i.e. **~20-30 h wall-clock** sequentially
on the cluster. Over a 3-consecutive-day window, we set a **50 % reserve
rule**: we plan nothing beyond ~36 h wall-clock, the rest
absorbing iterations, resumptions and dataset regenerations.

### Go/no-go gates (pre-registered)

| Gate | Moment | Criterion | If not passed |
|---|---|---|---|
| **G0** | End of Step 0 | The public anchor (R0) runs, the BF16 logit reference is frozen (§6.5.5) and the full baseline is written to `benchmarks/results.md` | No optimization is launched: this is the foundation of all comparisons |
| **G1** | End of Stage A | The NVFP4 checkpoint loads and generates in vLLM on GB10, PPL within the expected band | Mixed recipe (DeltaNet + ViT in BF16/FP8), then re-test |
| **G2** | After B2, on H100 | Acceptance rate ≥ 55-60 % on the offline validation set | Logit KD (+5 h) then more data; if < 40 % after iteration → native MTP becomes the video's final arm, and the distillation failure becomes the chapter |
| **G3** | Stage D | The optimization comes out of the noise (> 2 standard deviations) | Rejected and documented |

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Incompatible tokenizer between 3.5 and 3.8 | ✅ Cleared: verified compatible (§3.4); contract hardened and pinned before Stage B (§3.5) |
| The distilled draft does not beat native MTP | MTP is an arm in its own right (§ Stage C): the result is publishable as is, and it is an honest and rare conclusion |
| vLLM unusable on GB10, or MTP mode missing from the image | Blocking Step 0: image pinned by digest, MTP mode identified before any measurement. The "vLLM only" choice is an accepted learning objective, so we invest in the image, not in a second engine |
| NVFP4 recipe incompatible with the Gated DeltaNet blocks / the ViT | Mixed recipe, non-standard layers in BF16/FP8; measure the real impact rather than assume |
| Default thinking blurs all measurements and blows up the dataset budget | `reasoning_effort` frozen and documented at each stage, including for B1 generation; number of reasoning tokens displayed with the stopwatch |
| FP8 not supported on the Spark's GB10 | Verified at Step 0; otherwise we skip to the NVFP4 tier, no impact on the rest |
| Degradation barely visible with modern quantization | Perplexity remains sensitive; otherwise it is a narrative result ("robust down to 4-bit") |
| The model card benchmarks become the target (agentic harnesses, API judges, 3 unpublished in-house) | §6.5.1: list mapped into verdicts; they remain narrative on-screen markers, never targets |
| Generative baseline in thinking infeasible in wall-clock time (~100 h for GPQA-D) | §6.5.4: non-thinking or `effort=low`, temp 0, `max_tokens` fixed for the differential suites; thinking reserved for demos |
| Quantization breakage invisible on aggregate scores (GPQA-D ±2.2 pts at n=198) | §6.5.5: KL vs reference logits as fine metric + paired comparison (McNemar) |
| Low acceptance rate of the distilled draft | Measured offline on H100 before any Spark run; gate G2: logit KD, more data, or switch to MTP |
| Invalid `agentic_coding` items (self-grading) | Rewriting with external tests + sandbox **before** producing any score (§6, point 5) |
| Unexpected blocking blocker on the 3.8 | Global fallback: Qwen3.5-27B (identical pipeline, native little brothers available) |
| Calibration contaminated by eval prompts | Automatic intersection check, set hash in the manifest |
| Reproducibility drift (image, revisions, dataset) | Mandatory run manifest (§6.4); repo under git as of now |

---

## 9. Repo structure (planned)

```
fastqwen/
├── .gitignore                   # .env, cache/, checkpoints/, HF outputs
├── .env                         # config variables (paths, ports, HF_TOKEN)
├── project-plan.md              # this file
├── pyproject.toml               # metadata + pytest config (package not installed)
├── qwenbench/                   # reusable measurement core (unit 1)
│   ├── metrics.py               # pure: TTFT, throughput, reasoning/answer split
│   ├── sse.py                   # pure: SSE chunk parsing
│   ├── client.py                # httpx: per-chunk timestamping, waiting, fingerprint
│   └── report.py                # pure: medians + acceptance criteria (§6.5)
├── tests/                       # offline tests (pytest, no server, no GPU)
├── check_tokenizer_compat.py    # compatibility contract (hard requirements §3.5)
├── data/
│   ├── prompts/                 # evaluation prompts + generated datasets
│   └── calib/                   # mixed calibration set (FR + code + tools)
├── scripts/
│   ├── probe.py                 # unit 1: startup probe (wiring + throughput)
│   ├── generate_dataset.py      # B1: generation by the teacher (vLLM, H100),
│   │                            #      slices S1-S4 + validation set
│   ├── train_draft.py           # B2: student SFT (+ logit KD option)
│   ├── offline_acceptance.py    # C: acceptance rate + γ sweep (H100)
│   ├── quantize_nvfp4.py        # A: FP8 / NVFP4 / aggressive tier (ModelOpt,
│   │                            #     llm-compressor, mixed fallback recipe)
│   ├── build_calib_set.py       # A: mixed calibration set + anti-contamination
│   ├── build_eval_datasets.py   # PPL wiki-FR corpus, BFCL subset,
│   │                            # mini tool-calling bench, 30 qualitative prompts
│   ├── build_agentic_coding.py  # external tests for the 20 agentic_coding items
│   ├── grade_quality200.py      # quality-200 grader (numeric_exact, sandboxed exec,
│   │                            # ifeval_strict — 180 items, manual excluded)
│   ├── verify_identity.py       # C: greedy token-by-token identity vs target
│   ├── bench_spark.py           # uniform protocol (vllm bench + Prometheus wrapper)
│   ├── dump_ref_logits.py       # 6.5.5: freezes the reference BF16 logits (KL)
│   ├── kl_vs_ref.py             # 6.5.5: KL/JS + top-1 agreement vs reference
│   └── run_anchor.py            # 6.5.3 R0: public anchor GPQA-D + IFEval (lm-eval)
└── benchmarks/
    ├── quality/quality-200.jsonl # general abilities (180 self-graded items,
    │                             # 20 manual excluded)
    ├── ppl_corpus/wiki-fr-heldout.txt
    ├── mcq/mmlu_subset.jsonl     # optional, recognized on-screen reference
    ├── ref_logits/base-bf16/     # 6.5.5: frozen BF16 logits (KL metric)
    ├── agentic/bfcl_subset.jsonl + toolcalling_mini.jsonl
    ├── prompts/eval_prompts.jsonl
    ├── longcontext/niah_*.jsonl  # 6.5.3 R5: long-context probe (32k/128k)
    ├── runs/<run>/manifest.json  # manifest of each run (§6.4)
    ├── samples/<run>/            # greedy outputs per stage
    └── results.md                # cumulative results table
```

---

## 10. Technical references

External landmarks for the implementation and for situating our figures
(these are not our stack choices: vLLM is the project's single engine).

- Qwen 3.8 — family composition, licenses, hybrid thinking, context,
  hybrid dense architecture:
  https://codersera.com/blog/qwen-3-8-model-lineup-2026/
- Qwen3.5-2B (student): https://huggingface.co/Qwen/Qwen3.5-2B
- Serving Qwen3.8-27B on DGX Spark, benchmarking protocol and documented A/B:
  https://github.com/amasu/dgx-spark-qwen38
- vLLM + NVFP4 + MTP3 on DGX Spark:
  https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark
- NVFP4 checkpoints with MTP heads for sm_121:
  https://huggingface.co/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121
- AWQ W4A16 checkpoint (format comparison reference):
  https://huggingface.co/Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM
- MTP in speculation, throughput measurement:
  https://github.com/julianmb/q38rocm
- vLLM model documentation (Qwen3.8-27B):
  https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html
