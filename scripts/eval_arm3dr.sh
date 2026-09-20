#!/usr/bin/env bash
# Evaluate arm 3dr at steps 15 and 25, step 15 first under the frozen protocol.
#
# Arm 4 differs from the already-measured arm 3 in exactly one thing: beta, which decides
# whether a trajectory's reward sits on its last token or on the turns that earned it. The
# reward function is byte-identical (sum over tokens == arm 3's score, pinned by a test), the
# advantage normalisation is the same, dynamic sampling is off in both, and every other
# hyper-parameter matches. Step 15 is included because that is where arm 3's efficiency effect
# was significant and arm 4 is aimed at the same channel.
#
# Models share one directory because the data disk cannot hold two sixteen-gigabyte merges.
# The previous file is truncated first: safetensors needs the full space alongside, it does
# not overwrite in place, and discovering that mid-write cost a run earlier.
set -uo pipefail
cd /root/probesql
VENV=/root/autodl-tmp/envs/verl
OUT=models/_eval25
LOG=logs/eval_arm3dr_chain.log
: > "$LOG"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait on *python* processes only. A command line is a string, and any shell that merely
# mentions the trainer -- a monitor, a launcher's own echo -- matches a `ps | grep`. That is
# how this chain sat idle for twenty minutes after training had already finished, and it is
# the second time the same self-match has cost a wait loop in this project.
trainer_running() {
  local p exe
  for p in /proc/[0-9]*; do
    [ "$p" = "/proc/$$" ] && continue
    exe=$(readlink "$p/exe" 2>/dev/null) || continue
    case "$exe" in *python*) ;; *) continue ;; esac
    tr '\0' ' ' < "$p/cmdline" 2>/dev/null | grep -q "verl\.trainer\.main_ppo" && return 0
  done
  return 1
}

say "waiting for training to finish"
while trainer_running; do sleep 60; done
say "training ended"

for spec in "arm3dr_s15:ckpt/rl/arm3dr_0919_1341/global_step_15/actor" \
            "arm3dr_s25:ckpt/rl/arm3dr_0919_1341/global_step_25/actor"; do
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
