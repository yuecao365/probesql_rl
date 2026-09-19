#!/usr/bin/env bash
# Runs once arm 3's last training step lands: merge, serve, evaluate under the frozen protocol.
#
# Checks disk before it starts. Last night the same chain failed at the merge because the
# system disk was full, and the training run before it died at step 17 for the same reason --
# a 16 GB model write is not something to discover halfway through.
set -uo pipefail
cd /root/probesql
RUN=arm3_0918_0344
VENV=/root/autodl-tmp/envs/verl
LOG=logs/arm3_finish.log
: > "$LOG"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for training to finish"
while ps -eo cmd | grep -q "[m]ain_ppo"; do sleep 60; done
say "training ended"

DATA_FREE=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
SYS_FREE=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
say "disk: data ${DATA_FREE}G, system ${SYS_FREE}G"
[ "$DATA_FREE" -ge 20 ] || { say "need 20G on the data disk for the merged model, have ${DATA_FREE}G"; exit 1; }
[ "$SYS_FREE" -ge 4 ] || { say "need 4G on the system disk for vllm and ray, have ${SYS_FREE}G"; exit 1; }

STEP=$(ls -d ckpt/rl/$RUN/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
[ -n "$STEP" ] || { say "no checkpoint under ckpt/rl/$RUN"; exit 1; }
say "using step $STEP"

OUT=models/arm3_s$STEP
ARM=arm3_s$STEP
say "merging"
"$VENV/bin/python" scripts/rl_to_hf.py "ckpt/rl/$RUN/global_step_$STEP/actor" \
    --base models/sft_v2_ep3 --out "$OUT" >> "$LOG" 2>&1 || { say "merge failed"; exit 1; }
grep -m1 "merge verified" "$LOG" | tee -a "$LOG"

say "serving"
nohup bash scripts/serve_vllm.sh "$OUT" qwen8b > logs/vllm_$ARM.log 2>&1 &
for _ in $(seq 1 40); do
  grep -q "Application startup complete" logs/vllm_$ARM.log 2>/dev/null && break
  sleep 15
done
grep -q "Application startup complete" logs/vllm_$ARM.log || { say "vllm did not come up"; exit 1; }

say "evaluating 114 x 4 under the frozen protocol"
ARM=$ARM bash scripts/tau2_eval.sh > logs/eval_$ARM.log 2>&1
say "evaluation done"
grep -E "OVERALL pass@1|fix-it|escalate|all_pass|all_fail|protocol errors|mean turns|termination" \
    logs/eval_$ARM.log | tail -9 | tee -a "$LOG"
say "balance errors: $(grep -c 'Insufficient Balance' logs/eval_$ARM.log)"

for p in $(pgrep -f "vllm serve"); do kill "$p" 2>/dev/null; done
say "chain finished"
