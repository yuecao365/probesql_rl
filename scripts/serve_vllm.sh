#!/usr/bin/env bash
# Serve a local model with native tool calling for scripts/rollout.py.
#   bash scripts/serve_vllm.sh models/Qwen2.5-Coder-7B-Instruct qwen7b
set -euo pipefail
MODEL=${1:?model path}
NAME=${2:?served model name}
vllm serve "$MODEL" --served-model-name "$NAME" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 --gpu-memory-utilization 0.9 --port 8000
