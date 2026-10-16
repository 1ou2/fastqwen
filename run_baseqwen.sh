#!/usr/bin/env bash
export VLLM_USE_RUST_FRONTEND=1

# Engine startup timeout. Default = 600 s (vllm/envs.py).
# There is NO CLI flag for this setting: the Rust frontend receives it via
# --args-json from VLLM_ENGINE_READY_TIMEOUT_S (vllm/v1/utils.py:400).
# 600 s is not enough here: ~410 s of HF download + ~360 s to load the
# 18 shards (52 GB) + TorchInductor compilation/warmup.
export VLLM_ENGINE_READY_TIMEOUT_S=1800

vllm serve Qwen/Qwen3.8-27B \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data