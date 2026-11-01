IMAGE_TAG=vllm/vllm-openai:v0.32.0-cu129

docker pull "$IMAGE_TAG"

IMAGE_REF=$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE_TAG")
printf 'Pinned image: %s\n' "$IMAGE_REF"

docker run --rm --gpus '"device=0,1"' --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 \
  "$IMAGE_REF" \
  --model Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --port 8000

# for uninstalling torchcodec and running the server
# with v0.32.0-cu129 there is an error related to torchcodec, so we uninstall it before running the server
docker run --rm --gpus '"device=0,1"' --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  "$IMAGE_REF" \
  -lc 'python3 -m pip uninstall -y torchcodec && exec vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 2 --port 8000'