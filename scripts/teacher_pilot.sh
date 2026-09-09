#!/usr/bin/env bash
# Pilot two DeepSeek teachers on the same SFT questions and measure real cost via account balance.
#   bash scripts/teacher_pilot.sh 10 4 nothink
set -euo pipefail
set -a; source .env; set +a
N=${1:-10}; G=${2:-4}; TAG=${3:-run}
head -$N splits/sft_ids.txt > outputs/pilot_ids.txt
balance() { curl -s -m 30 --noproxy '*' https://api.deepseek.com/user/balance -H "Authorization: Bearer $DEEPSEEK_API_KEY" | python -c "import sys,json; print(json.load(sys.stdin)['balance_infos'][0]['total_balance'])"; }
for MODEL in deepseek-v4-flash deepseek-v4-pro; do
  B0=$(balance)
  env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy python scripts/rollout.py \
    --json data/bird/train/train.json --db-dir data/bird/train/train_databases --ids outputs/pilot_ids.txt \
    --g $G --temperature 1.0 --max-tokens 2048 --workers 8 --base-url https://api.deepseek.com --api-key "$DEEPSEEK_API_KEY" \
    --extra-body '{"thinking": {"type": "disabled"}}' \
    --model $MODEL --out outputs/pilot_${TAG}_${MODEL}.jsonl
  B1=$(balance)
  echo "COST $MODEL: $B0 -> $B1 CNY for $((N*G)) rollouts"
  python scripts/metrics.py outputs/pilot_${TAG}_${MODEL}.jsonl
  python scripts/build_sft.py outputs/pilot_${TAG}_${MODEL}.jsonl --per-question 2 --out outputs/pilot_${TAG}_${MODEL}_sft.jsonl
done
