repo="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B"
rev=$(cat "$repo/refs/main")

ls -lh "$repo/snapshots/$rev"
du -sh "$repo/blobs"
du -shL "$repo/snapshots/$rev"
hf cache verify Qwen/Qwen3.5-2B --revision "$rev"

repo="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B"
rev=$(cat "$repo/refs/main")

ls -lh "$repo/snapshots/$rev"
du -sh "$repo/blobs"
du -shL "$repo/snapshots/$rev"
hf cache verify Qwen/Qwen3.8-27B --revision "$rev"