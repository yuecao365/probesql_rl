#!/usr/bin/env bash
# Evaluate one arm under the frozen protocol in docs/eval_protocol.md.
#
#   bash scripts/serve_vllm.sh models/Qwen3-8B qwen8b          # or a merged SFT model
#   ARM=arm0 bash scripts/tau2_eval.sh
#   python scripts/tau2_buckets.py outputs/eval_arm0.json
#
# All 114 base tasks, 4 rollouts each. Not a sample: the pilot's 60-task draw was for a
# GO/NO-GO decision, and reusing it as a baseline would mean later arms are measured on a
# different task set. Four rollouts rather than eight because the standard error is
# dominated by task sampling (2.33% vs 2.11%), so coverage is the better buy.
set -euo pipefail

REPO=/root/probesql
TAU2=/root/autodl-tmp/tau2-bench
ARM=${ARM:?set ARM, e.g. arm0 or arm1_ep2}
MODEL=${MODEL:-qwen8b}
G=${G:-4}
BASE_URL=${BASE_URL:-http://localhost:8000/v1}
USER_LLM=${USER_LLM:-deepseek/deepseek-chat}
OUT=$REPO/outputs/eval_${ARM}.json

set -a; source "$REPO/.env"; set +a

RESUME=""
if [ -e "$OUT" ]; then
  if [ "${FORCE:-0}" = 1 ]; then mv "$OUT" "$OUT.$(date +%s).bak"
  else RESUME=--auto-resume; fi
fi

cd "$TAU2"
env -u HTTPS_PROXY -u HTTP_PROXY DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  .venv/bin/tau2 run \
    --domain telecom \
    --agent-llm "openai/$MODEL" \
    --agent-llm-args "{\"temperature\":1.0,\"api_base\":\"$BASE_URL\",\"api_key\":\"dummy\",\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}" \
    --user-llm "$USER_LLM" \
    --user-llm-args '{"temperature":0.0}' \
    --num-trials "$G" ${RESUME:-} \
    --max-concurrency "${CONCURRENCY:-8}" \
    --save-to "$OUT"

echo
python "$REPO/scripts/tau2_buckets.py" "$OUT"
