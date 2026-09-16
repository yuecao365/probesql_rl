#!/usr/bin/env bash
# Finish the D4 epoch scan unattended, then power the box off.
#
#   nohup bash scripts/overnight.sh > logs/overnight.log 2>&1 &
#
# Waits for the running SFT to finish, then for each epoch checkpoint: merge the adapter,
# serve it, evaluate all 114 base tasks at 4 rollouts, tear the server down and delete the
# merged weights before the next one. Shuts the instance down at the end.
#
# Deliberately not `set -e`. A failure in one epoch must not skip the shutdown -- an
# unattended box that stays up all night costs more than the evaluation it was running. The
# trap fires on every exit path; every stage has a timeout so nothing can hang past it.
set -uo pipefail

REPO=/root/probesql
BASE=$REPO/models/Qwen3-8B
RUN=${RUN:-ckpt/sft_v1}
SHUTDOWN=${SHUTDOWN:-1}

cd "$REPO"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup() {
  log "cleanup: stopping any vLLM"
  pkill -f "bin/vllm serve" 2>/dev/null
  sleep 5
  git add -A && git commit -q -m "D4 epoch scan: arm 1 evaluations

Unattended run of scripts/overnight.sh." 2>/dev/null && log "committed results"
  if [ "$SHUTDOWN" = 1 ]; then
    log "shutting down"
    shutdown -h now 2>/dev/null || poweroff 2>/dev/null || halt 2>/dev/null
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------- wait for training
log "waiting for SFT to finish"
while pgrep -f "sft/train.py" >/dev/null; do sleep 60; done
log "SFT done; checkpoints: $(ls "$RUN" 2>/dev/null | tr '\n' ' ')"

python scripts/sft_curve.py "$RUN" --png docs/assets/sft_v1_loss.png 2>&1 | tail -20

# ---------------------------------------------------------------- one epoch at a time
for CKPT in $(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  STEP=$(basename "$CKPT" | cut -d- -f2)
  EP=$(python - "$CKPT" <<'PY'
import json, sys
print(round(json.load(open(sys.argv[1] + "/trainer_state.json"))["epoch"]))
PY
)
  ARM="arm1_ep${EP}"
  MERGED="$REPO/models/sft_v1_ep${EP}"
  log "=== epoch $EP (step $STEP) ==="

  if [ -f "$REPO/outputs/eval_${ARM}.json/results.json" ]; then
    log "already evaluated, skipping"
    continue
  fi

  log "merging -> $MERGED"
  timeout 1800 python sft/merge.py --base "$BASE" --adapter "$CKPT" --out "$MERGED" \
    || { log "merge failed for epoch $EP"; rm -rf "$MERGED"; continue; }

  log "serving"
  nohup bash scripts/serve_vllm.sh "$MERGED" qwen8b > "logs/vllm_${ARM}.log" 2>&1 &
  for _ in $(seq 1 90); do
    curl -s http://localhost:8000/v1/models >/dev/null 2>&1 && break
    grep -qE "Traceback|out of memory" "logs/vllm_${ARM}.log" && break
    sleep 10
  done
  if ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    log "server never came up for epoch $EP"; pkill -f "bin/vllm serve"; rm -rf "$MERGED"; continue
  fi

  log "evaluating $ARM"
  timeout 7200 env ARM="$ARM" MODEL=qwen8b CONCURRENCY=8 bash scripts/tau2_eval.sh \
    > "logs/eval_${ARM}.log" 2>&1
  log "$(tail -20 "logs/eval_${ARM}.log")"

  pkill -f "bin/vllm serve"; sleep 10
  rm -rf "$MERGED"                      # 16 GB each; keep only the adapters
  log "epoch $EP finished"
done

# ---------------------------------------------------------------- summary
log "=== D4 summary ==="
for f in outputs/eval_arm0.json outputs/eval_arm1_ep*.json; do
  [ -e "$f" ] || continue
  echo "--- $f"
  python scripts/tau2_buckets.py "$f" 2>/dev/null | head -12
done
log "all done"
