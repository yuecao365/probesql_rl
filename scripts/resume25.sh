#!/usr/bin/env bash
# Continue arm 2b and arm 3 from step 15 to step 25, then evaluate both at 25.
#
# The question is the one the 15-step numbers could not answer: arm 3's reward has two
# channels, an efficiency term that only ranks successes and a partial-credit term that only
# ranks failures. At 15 steps the first is already significant (successful trajectories 5.0%
# shorter, repeated tool calls down 61.8%) and the second is not visible at all, which is what
# the mechanism predicts -- failures are 16% of rollouts, so that channel carries far less
# signal. Ten more steps says whether it starts to show in pass@1.
#
# Both continue at batch 8 with the same reward and learning rate, so the curve stays one
# curve and the 15-step points remain part of it.
set -uo pipefail
cd /root/probesql
LOG=logs/resume25.log
: > "$LOG"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

for spec in "arm2:arm2_grpo_0917_1758" "arm3:arm3_0918_0344"; do
  ARM=${spec%%:*}; EXPN=${spec#*:}
  LAST=$(ls -d ckpt/rl/$EXPN/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
  say "$ARM ($EXPN): resuming from step ${LAST:-none} to 25"
  [ -n "$LAST" ] || { say "$ARM: no checkpoint, skipping"; continue; }
  [ "$LAST" -ge 25 ] && { say "$ARM: already at $LAST"; continue; }

  FREE=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
  [ "$FREE" -ge 5 ] || { say "$ARM: only ${FREE}G free, stopping"; exit 1; }

  EXP=$EXPN STEPS=25 bash scripts/rl_arm.sh "$ARM" >> logs/${ARM}_resume.log 2>&1
  NEW=$(ls -d ckpt/rl/$EXPN/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
  say "$ARM: now at step ${NEW:-?}"
done
say "training finished"
