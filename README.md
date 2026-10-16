# fastqwen
Accelerate qwen 3.8

# Base qwen version
Use official recipe https://recipes.vllm.ai/Qwen/Qwen3.8-27B?hardware=dgx_spark_gb10

## install
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
uv pip install -U "transformers>=5.8.0"

## Run
Run script is `run_baseqwen.sh`
