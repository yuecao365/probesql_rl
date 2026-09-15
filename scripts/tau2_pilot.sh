#!/usr/bin/env bash
# Pilot: is tau2-bench telecom trainable at 8B? Settings and thresholds are frozen
# in docs/pilot_tau2.md -- do not tune them after seeing a result.
#
#   bash scripts/serve_vllm.sh models/Qwen3-8B qwen8b     # GPU box, first
#   bash scripts/tau2_pilot.sh
#   python scripts/tau2_buckets.py outputs/tau2_pilot_telecom.json
#
# The agent is the local vLLM student; the user simulator runs on the DeepSeek API so
# the GPU serves one model only. litellm forwards unknown kwargs to the provider, which
# is how api_base reaches the OpenAI-compatible vLLM endpoint.
#
# enable_thinking=false is not optional. Measured against this endpoint: with thinking
# on, Qwen3-8B answers the same prompt with a clarifying question and no tool call at
# all; with it off it calls the tool. It is the setting MUA-RL's numbers refer to too.
set -euo pipefail

REPO=/root/probesql
TAU2=/root/autodl-tmp/tau2-bench
G=${G:-8}
N=${N:-60}
SEED=${SEED:-0}
MODEL=${MODEL:-qwen8b}
BASE_URL=${BASE_URL:-http://localhost:8000/v1}
USER_LLM=${USER_LLM:-deepseek/deepseek-chat}
OUT=$REPO/outputs/tau2_pilot_telecom.json

set -a; source "$REPO/.env"; set +a

# Sample from the `base` split: that is telecom's standard 114-task benchmark, the set
# MUA-RL's 19.1 refers to, and the only pool comparable to anything published. tasks.json
# holds the 2285-task combinatorial expansion, which is the RL *training* pool, not the
# benchmark. Seeded rather than --num-tasks, which takes a prefix and so would only ever
# cover the first few issue templates.
mapfile -t TASK_IDS < <("$TAU2/.venv/bin/python" - "$SEED" "$N" <<'PY'
import json, random, sys
seed, n = int(sys.argv[1]), int(sys.argv[2])
splits = json.load(open("/root/autodl-tmp/tau2-bench/data/tau2/domains/telecom/split_tasks.json"))
ids = splits["base"]
for t in random.Random(seed).sample(ids, min(n, len(ids))):
    print(t)
PY
)
echo "sampled ${#TASK_IDS[@]} telecom tasks (seed $SEED), G=$G"

cd "$TAU2"
env -u HTTPS_PROXY -u HTTP_PROXY DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  .venv/bin/tau2 run \
    --domain telecom \
    --agent-llm "openai/$MODEL" \
    --agent-llm-args "{\"temperature\":1.0,\"api_base\":\"$BASE_URL\",\"api_key\":\"dummy\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}" \
    --user-llm "$USER_LLM" \
    --user-llm-args '{"temperature":0.0}' \
    --task-ids "${TASK_IDS[@]}" \
    --num-trials "$G" \
    --max-concurrency "${CONCURRENCY:-8}" \
    --save-to "$OUT"

echo
"$REPO/.venv/bin/python" "$REPO/scripts/tau2_buckets.py" "$OUT" 2>/dev/null \
  || python "$REPO/scripts/tau2_buckets.py" "$OUT"
