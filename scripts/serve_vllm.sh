#!/usr/bin/env bash
# Serve the student locally with native tool calling, for the pilot and for RL rollouts.
#   bash scripts/serve_vllm.sh models/Qwen3-8B qwen8b
#
# Qwen3 emits <think> blocks by default. MUA-RL's published numbers are "Non-thinking",
# and thinking would eat the reply budget in a 10-turn episode, so callers disable it per
# request with chat_template_kwargs={"enable_thinking": false}. --reasoning-parser is set
# anyway so that anything the model does emit inside <think> is separated from the tool
# call rather than corrupting it.
#
# 32k, not 16k: measured on this model, one telecom episode ran 56 assistant turns and
# 16,132 tokens against a 16,384 limit. An episode that hits the wall comes back as an
# agent error and scores zero, so the number would be measuring the context budget rather
# than the policy. Qwen3-8B is trained to 40,960 and the KV cache has room.
set -euo pipefail

# Activate the env rather than relying on the caller: LD_PRELOAD is stored as a conda env
# var, so it only takes effect on `conda activate`. Launched from a bare shell the old
# system libstdc++ wins and vLLM dies on a CXXABI_1.3.15 ImportError.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl

MODEL=${1:?model path}
NAME=${2:?served model name}
vllm serve "$MODEL" --served-model-name "$NAME" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 --port 8000
