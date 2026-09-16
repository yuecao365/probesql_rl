#!/usr/bin/env bash
# arm 2 -- vanilla GRPO on top of the SFT checkpoint, outcome reward only, CA-0.
#
#   bash scripts/rl_arm2.sh smoke     two tasks, two rollouts, one step: plumbing only
#   bash scripts/rl_arm2.sh           the real run
#
# What is deliberate here, since every one of these was measured rather than assumed:
#
#   enable_thinking=False       with thinking on, Qwen3-8B makes zero tool calls. The SFT
#                               data was built this way; a rollout that differs is a
#                               different model.
#   filter_groups.enable=True   43.9% of groups at G=4 were all-pass or all-fail and carry
#                               no gradient. Sampling past them costs ~1.78x in user-
#                               simulator API calls and is still worth it.
#   norm_adv_by_std_in_grpo     off: dividing by the group std upweights tasks that are
#                               nearly always solved or nearly never solved.
#   clip_ratio_high > low       clip-higher, the entropy-collapse guard. Entropy is 0.357
#                               nats at the SFT checkpoint and is logged every step.
#   use_kl_loss / kl_in_reward  both off. There is a verifiable reward; a KL leash to the
#                               SFT policy would only slow the thing being measured.
#   loss_agg_mode=token-mean    token-level loss, so a 40-turn episode is not weighted the
#                               same as a 6-turn one.
#
# No shutdown logic. A previous EXIT trap powered the box off mid-training.
set -euo pipefail
cd /root/probesql
set -a; source .env; set +a

MODE=${1:-full}
VENV=/root/autodl-tmp/envs/verl
EXP="arm2_grpo_$(date +%m%d_%H%M)"

if [ "$MODE" = "smoke" ]; then
  BATCH=2; N=2; STEPS=1; EXP="smoke_$(date +%m%d_%H%M)"
else
  BATCH=8; N=8; STEPS=${STEPS:-50}
fi

mkdir -p logs ckpt/rl
# litellm otherwise spends 40s per process failing to fetch a price list from GitHub.
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONPATH=/root/probesql:${PYTHONPATH:-}
export TAU2_DOMAIN=telecom TAU2_USER_LLM=deepseek/deepseek-chat TAU2_USER_TEMP=0.0
export TAU2_W_DELTA=0.0          # outcome-only: this is what makes it arm 2 and not arm 3
export VLLM_USE_V1=1
# verl hardcodes flash_attention_2; it is not installed and only ships as an sdist that
# takes hours to compile. See the local patch in verl/utils/model.py.
export VERL_ATTN_IMPLEMENTATION=sdpa

env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
$VENV/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=score \
  algorithm.filter_groups.max_num_gen_batches=8 \
  data.train_files=data/rl/train.parquet \
  data.val_files=data/rl/val.parquet \
  data.train_batch_size=$BATCH \
  data.max_prompt_length=8192 \
  data.max_response_length=24576 \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=/root/probesql/models/sft_v2_ep3 \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.max_model_len=32768 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=40 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tau2 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=rl/agent_loop.yaml \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  trainer.use_v1=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=$STEPS \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.default_local_dir=ckpt/rl/$EXP \
  trainer.project_name=turncredit-rl \
  trainer.experiment_name=$EXP \
  trainer.logger='["console"]' \
  2>&1 | tee logs/$EXP.log
