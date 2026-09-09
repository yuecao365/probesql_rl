#!/usr/bin/env bash
# M1 baselines against a running vLLM server (scripts/serve_vllm.sh).
set -euo pipefail
DEV=data/bird/dev_20240627/dev.json; DEVDB=data/bird/dev_20240627/dev_databases
TRAIN=data/bird/train/train.json; TRAINDB=data/bird/train/train_databases
URL=http://localhost:8000/v1; MODEL=${1:-qwen7b}; TAG=${2:-coder7b}
[ -f outputs/m1_${TAG}_dev300_single.jsonl ] || python scripts/single_turn.py --json $DEV --db-dir $DEVDB --n 300 --workers 48 \
    --base-url $URL --model $MODEL --out outputs/m1_${TAG}_dev300_single.jsonl
python scripts/rollout.py --json $DEV --db-dir $DEVDB --n 300 --g 1 --temperature 0.6 --workers 48 --lenient-tool-parse \
    --base-url $URL --model $MODEL --out outputs/m1_${TAG}_dev300_agent.jsonl
python scripts/rollout.py --json $TRAIN --db-dir $TRAINDB --n 200 --g 8 --temperature 1.0 --workers 64 --lenient-tool-parse \
    --base-url $URL --model $MODEL --out outputs/m1_${TAG}_train200_g8.jsonl
python scripts/metrics.py outputs/m1_${TAG}_dev300_single.jsonl outputs/m1_${TAG}_dev300_agent.jsonl outputs/m1_${TAG}_train200_g8.jsonl
