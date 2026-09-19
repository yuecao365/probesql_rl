#!/usr/bin/env bash
# How much does the answer move between two checkpoints of the same run?
#
# Every arm comparison so far rests on a single snapshot -- step 15 of each run -- and the
# two that landed inside the noise band cannot be read without knowing how wide that band is.
# This evaluates step 10 of arm 2b and of arm 3 under the same frozen protocol, which gives
# two things at once: the spread between step 10 and step 15 within a run, and a second,
# independent arm3-vs-arm2b comparison at a different point in training.
#
# The two merged models share one directory because the data disk holds 29 GB and each is 16.
# Overwriting avoids needing to delete anything.
set -uo pipefail
cd /root/probesql
VENV=/root/autodl-tmp/envs/verl
OUT=models/_ckpt_probe
LOG=logs/ckpt_noise.log
: > "$LOG"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

for spec in "arm2b_s10:ckpt/rl/arm2_grpo_0917_1758/global_step_10/actor" \
            "arm3_s10:ckpt/rl/arm3_0918_0344/global_step_10/actor"; do
  ARM=${spec%%:*}; CK=${spec#*:}
  [ -d "$CK" ] || { say "$ARM: no checkpoint at $CK"; continue; }

  FREE=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
  [ "$FREE" -ge 18 ] || { say "$ARM: need 18G, have ${FREE}G"; exit 1; }

  say "$ARM: merging"
  "$VENV/bin/python" scripts/rl_to_hf.py "$CK" --base models/sft_v2_ep3 --out "$OUT" \
      >> "$LOG" 2>&1 || { say "$ARM: merge failed"; continue; }
  grep "merge verified" "$LOG" | tail -1 | sed "s/^/  $ARM /" | tee -a "$LOG"

  say "$ARM: serving"
  nohup bash scripts/serve_vllm.sh "$OUT" qwen8b > logs/vllm_$ARM.log 2>&1 &
  for _ in $(seq 1 40); do
    grep -q "Application startup complete" logs/vllm_$ARM.log 2>/dev/null && break
    sleep 15
  done
  grep -q "Application startup complete" logs/vllm_$ARM.log || { say "$ARM: vllm did not start"; continue; }

  say "$ARM: evaluating"
  ARM=$ARM bash scripts/tau2_eval.sh > logs/eval_$ARM.log 2>&1
  grep -E "OVERALL pass@1|all_pass|mean turns|termination" logs/eval_$ARM.log | tail -4 | sed "s/^/  $ARM /" | tee -a "$LOG"
  say "$ARM: balance errors $(grep -c 'Insufficient Balance' logs/eval_$ARM.log)"

  for p in $(pgrep -f "vllm serve"); do kill "$p" 2>/dev/null; done
  sleep 20
done
say "finished"
