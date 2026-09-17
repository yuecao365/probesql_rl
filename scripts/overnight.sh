#!/usr/bin/env bash
# Runs after arm 2b's last training step, unattended.
#
# Evaluates two checkpoints under the frozen protocol -- step 10 and step 20 -- so the
# morning starts with a curve on the 114 base tasks rather than with the training reward,
# which this project has already shown is dominated by which eight tasks a step drew.
#
# No shutdown logic. The GPU keeps running when this finishes; stopping the instance is a
# decision for a person.
set -uo pipefail
cd /root/probesql
RUN=arm2_grpo_0917_1758
VENV=/root/autodl-tmp/envs/verl
LOG=logs/overnight.log
: > "$LOG"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for training to finish"
while ps -eo cmd | grep -q "[m]ain_ppo"; do sleep 60; done
say "training ended"
ls -d ckpt/rl/$RUN/global_step_* 2>/dev/null | tee -a "$LOG"

for STEP in 10 20; do
  CK=ckpt/rl/$RUN/global_step_$STEP/actor
  [ -d "$CK" ] || { say "no checkpoint at step $STEP, skipping"; continue; }
  OUT=models/arm2b_s$STEP
  ARM=arm2b_s$STEP

  say "step $STEP: merging"
  "$VENV/bin/python" scripts/rl_to_hf.py "$CK" --base models/sft_v2_ep3 --out "$OUT" \
      >> "$LOG" 2>&1 || { say "step $STEP: merge failed"; continue; }

  say "step $STEP: serving"
  nohup bash scripts/serve_vllm.sh "$OUT" qwen8b > logs/vllm_$ARM.log 2>&1 &
  for _ in $(seq 1 40); do
    grep -q "Application startup complete" logs/vllm_$ARM.log 2>/dev/null && break
    sleep 15
  done
  grep -q "Application startup complete" logs/vllm_$ARM.log || { say "step $STEP: vllm did not come up"; continue; }

  say "step $STEP: evaluating 114 x 4"
  ARM=$ARM bash scripts/tau2_eval.sh > logs/eval_$ARM.log 2>&1
  say "step $STEP: done"
  grep -E "OVERALL pass@1|fix-it|escalate|all_pass|protocol errors|mean turns|termination" \
      logs/eval_$ARM.log | tail -8 | tee -a "$LOG"
  grep -c "Insufficient Balance" logs/eval_$ARM.log | xargs -I{} say "step $STEP: balance errors {}"

  for p in $(pgrep -f "vllm serve"); do kill "$p" 2>/dev/null; done
  sleep 20
done
say "overnight chain finished"
