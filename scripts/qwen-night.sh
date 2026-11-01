cd /opt/home/gabriel/dev/fastqwen
source .venv/bin/activate
set -euo pipefail

export D1_PID=3058752
export NIGHT="s1-night-$(date +%Y%m%d-%H%M%S)"
export NIGHT_DIR="data/runs/$NIGHT"
export CLASS_DIR="$NIGHT_DIR/classification"
export IMAGE_DIGEST='sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f'
export IMAGE_REF="vllm/vllm-openai@$IMAGE_DIGEST"
export TEACHER_REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'

[[ "$(ps -p "$D1_PID" -o args=)" == *scripts.serve_decision_model* ]]
mkdir -p "$NIGHT_DIR"
printf 'Run directory: %s\n' "$NIGHT_DIR"

test -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/$TEACHER_REV"
docker image inspect "$IMAGE_REF" --format '{{json .RepoDigests}}'