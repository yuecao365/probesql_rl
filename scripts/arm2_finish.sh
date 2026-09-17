#!/usr/bin/env bash
# Everything that happens after arm 2's last training step.
#
#   bash scripts/arm2_finish.sh ckpt/rl/arm2_grpo_0917_1410
#
# Three things, in the order they unblock each other: turn verl's sharded checkpoint into
# something vLLM can serve, draw the training curves, and then measure the policy on the
# frozen evaluation set -- 114 base tasks x 4 rollouts, the same protocol arm 0 and arm 1
# were measured under. Only that last number is comparable across arms; the training reward
# is sampled at temperature 1.0 on a different pool and says nothing about where the policy
# now stands.
#
# No shutdown logic, deliberately.
set -euo pipefail
cd /root/probesql
CKPT=${1:?checkpoint dir, e.g. ckpt/rl/arm2_grpo_0917_1410}
STEP=$(ls -d "$CKPT"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
[ -n "$STEP" ] || { echo "no global_step_* under $CKPT"; exit 1; }
echo "using $STEP"

VENV=/root/autodl-tmp/envs/verl
OUT=models/arm2_rl

# 1. verl's shards -> a Hugging Face model directory
"$VENV/bin/python" -m verl.model_merger merge \
  --backend fsdp --local_dir "$STEP/actor" --target_dir "$OUT"

# 2. curves, from the tensorboard scalars the run wrote as it went
"$VENV/bin/python" scripts/rl_report.py \
  "tensorboard_log/turncredit-rl/$(basename "$CKPT")" --out docs/assets --tag arm2

echo
echo "merged into $OUT. Next, under the frozen protocol:"
echo "  bash scripts/serve_vllm.sh $OUT qwen8b        # in its own shell"
echo "  bash scripts/tau2_eval.sh arm2"
echo "  python scripts/tau2_buckets.py outputs/eval_arm2.json"
