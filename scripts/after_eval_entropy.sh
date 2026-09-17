#!/usr/bin/env bash
# Queued chain: wait for the arm1 evaluation to end, free the GPU, then measure entropy.
#
# Entropy is the last outstanding SFT diagnostic. It needs the whole GPU, and the arm1
# vLLM server is holding 74 GB of it, so the two cannot overlap -- hence the wait.
#
# No shutdown logic anywhere in here, deliberately: a previous EXIT trap powered the box
# off mid-training. This script stops one vLLM server it identifies by model path and
# nothing else.
set -uo pipefail
cd /root/probesql
PY=/root/miniconda3/envs/rl/bin/python
LOG=logs/entropy.log
: > "$LOG"

echo "[$(date '+%F %T')] waiting for tau2_eval to finish" | tee -a "$LOG"
while pgrep -f "bin/tau2 run" >/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] eval done" | tee -a "$LOG"

# Stop only our own arm1 server, matched on its model path rather than on 'vllm'.
PID=$(pgrep -f "vllm serve /root/probesql/models/sft_v2_ep3" | head -1)
if [ -n "${PID:-}" ]; then
  echo "[$(date '+%F %T')] stopping vLLM pid $PID" | tee -a "$LOG"
  kill "$PID"
  for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
fi
sleep 20
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"

for CK in "" ckpt/sft_v2/checkpoint-88 ckpt/sft_v2/checkpoint-176 ckpt/sft_v2/checkpoint-264; do
  NAME=${CK:-base}
  echo "=== $NAME ===" | tee -a "$LOG"
  if [ -z "$CK" ]; then
    $PY scripts/entropy.py --base models/Qwen3-8B --data data/sft/teacher_v2.jsonl 2>&1 | tee -a "$LOG"
  else
    $PY scripts/entropy.py --base models/Qwen3-8B --adapter "$CK" \
        --data data/sft/teacher_v2.jsonl 2>&1 | tee -a "$LOG"
  fi
done
echo "[$(date '+%F %T')] entropy chain finished" | tee -a "$LOG"
