#!/usr/bin/env bash
# Evaluate arm 2b and arm 3 at step 25 under the frozen protocol.
#
# The two arms differ in one thing, the advantage estimator, and everything else -- learning
# rate 2e-5, four updates per rollout batch, batch of eight tasks, twenty-five steps -- is
# identical, so the comparison is clean. At fifteen steps the shaped reward was already
# significant on efficiency (successful trajectories 5.0% shorter, repeated tool calls down
# 61.8%) and indistinguishable on pass@1; ten more steps says whether the correctness channel,
# which only touches the 16% of rollouts that fail, has started to accumulate.
#
# Models share one directory because the data disk cannot hold two sixteen-gigabyte merges.
# The previous file is truncated first: safetensors needs the full space alongside, it does
# not overwrite in place, and discovering that mid-write cost a run earlier today.
set -uo pipefail
cd /root/probesql
VENV=/root/autodl-tmp/envs/verl
OUT=models/_eval25
LOG=logs/eval25.log
: > "$LOG"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for training to finish"
while ps -eo cmd | grep -qE "[m]ain_ppo|[r]esume25.sh"; do sleep 60; done
say "training ended"

for spec in "arm2b_s25:ckpt/rl/arm2_grpo_0917_1758/global_step_25/actor" \
            "arm3_s25:ckpt/rl/arm3_0918_0344/global_step_25/actor"; do
  ARM=${spec%%:*}; CK=${spec#*:}
  [ -d "$CK" ] || { say "$ARM: no checkpoint at $CK"; continue; }

  [ -f "$OUT/model.safetensors" ] && { truncate -s 0 "$OUT/model.safetensors"; say "$ARM: reclaimed the previous merge"; }
  FREE=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
  say "$ARM: ${FREE}G free"
  [ "$FREE" -ge 18 ] || { say "$ARM: need 18G, stopping"; exit 1; }

  say "$ARM: merging"
  "$VENV/bin/python" scripts/rl_to_hf.py "$CK" --base models/sft_v2_ep3 --out "$OUT" >> "$LOG" 2>&1 \
      || { say "$ARM: merge failed"; continue; }
  grep "merge verified" "$LOG" | tail -1 | sed "s/^/  $ARM /" | tee -a "$LOG"

  say "$ARM: serving"
  nohup bash scripts/serve_vllm.sh "$OUT" qwen8b > logs/vllm_$ARM.log 2>&1 &
  for _ in $(seq 1 40); do grep -q "Application startup complete" logs/vllm_$ARM.log 2>/dev/null && break; sleep 15; done
  grep -q "Application startup complete" logs/vllm_$ARM.log || { say "$ARM: vllm did not start"; continue; }

  say "$ARM: evaluating 114 x 4"
  ARM=$ARM bash scripts/tau2_eval.sh > logs/eval_$ARM.log 2>&1
  grep -E "OVERALL pass@1|all_pass|all_fail|mean turns|mean tool calls|termination" logs/eval_$ARM.log \
      | tail -6 | sed "s/^/  $ARM /" | tee -a "$LOG"
  say "$ARM: balance errors $(grep -c 'Insufficient Balance' logs/eval_$ARM.log)"

  for p in $(pgrep -f "vllm serve"); do kill "$p" 2>/dev/null; done
  sleep 20
done
say "finished"
