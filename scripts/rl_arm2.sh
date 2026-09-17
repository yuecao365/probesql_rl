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
#   agent.num_workers=32       sized against the KV cache, and it must divide
#                               train_batch_size * rollout.n = 64, which rules out 12. Qwen3-8B costs 144 KB
#                               of KV per token, so vLLM's 39.6 GB holds 169k tokens once the
#                               16.4 GB of weights are out; a telecom episode reaches twenty to
#                               thirty thousand tokens, which is six to eight episodes' worth.
#                               Thirty-two in flight oversubscribes it by four times and vLLM
#                               spends the difference evicting and recomputing prefixes --
#                               measured as agent turns per minute falling 163 -> 32 as the
#                               contexts grew. Evaluation reached 11.8 episodes a minute at
#                               concurrency 8 precisely because vLLM had the whole card.
#   dataloader_num_workers=2   eight forked workers exhausted the container memory and
#                               the run was killed at the end of the first step.
#   num_workers raised to 32   the earlier sizing assumed 23k-token episodes, which was
#                               the duplicated-schema bug; at 2.6k tokens the KV cache
#                               has room for many more in flight.
#   val_before_train=False     the pre-training validation pass costs 64 episodes and the
#                               held-out set cannot resolve anything anyway (+/-10.5%).
#   gpu_memory_utilization=0.5 the actor is bf16 and needs about 20 GB, so 0.35 was starving
#                               the KV cache: with 32 concurrent episodes whose contexts keep
#                               growing, agent turns per minute fell from 29 to 15 over ten
#                               minutes as sequences were preempted.
#   use_remove_padding=True    on, and it has to be: the trainer calls
#                               left_right_2_no_padding unconditionally, so the unpad
#                               code runs either way (verl/utils/attention_utils.py is
#                               patched locally to fall back to verl's own pure-torch
#                               copy when flash-attn is absent). Turning it off only
#                               made the actor carry padding into the forward pass,
#                               where logits for a padded 32k sequence over a 151k
#                               vocabulary came to 17.4 GB and went out of memory.
#   entropy chunking           logits are the largest tensor in this run by far; chunked
#                               entropy and entropy checkpointing keep that peak down.
#   use_dynamic_bsz=False      the dynamic path packs sequences through the same
#                               unpad_input, and flash-attn is not installed here -- it only
#                               ships as an sdist that takes hours to build, which is why the
#                               actor runs on sdpa. Fixed micro-batches of one avoid the
#                               packing path; at 30 turns a sequence is long enough that one
#                               per micro-batch is what fits anyway.
#   max_assistant_turns=30     the teacher's successful trajectories run to 15 turns at the
#                               median and 25 at the 95th percentile, so a cap of 30 truncates
#                               0.6% of them. What it does cut is the pathological tail, which
#                               is mostly episodes that were going to score zero anyway.
#   target_modules explicit    not 'all-linear'. vLLM packs q/k/v into one qkv_proj slice
#                               and gate/up into gate_up_proj, and decides what to pack from
#                               peft_config.target_modules. The string 'all-linear' leaves it
#                               unable to group them, so set_lora receives one tensor where it
#                               iterates a list of three and indexes a 1-D row. These are also
#                               the seven modules the SFT checkpoint was trained with, which
#                               they have to match anyway.
#
# No shutdown logic. A previous EXIT trap powered the box off mid-training.
set -euo pipefail
cd /root/probesql
set -a; source .env; set +a

MODE=${1:-full}
VENV=/root/autodl-tmp/envs/verl
EXP="arm2_grpo_$(date +%m%d_%H%M)"

if [ "$MODE" = "smoke" ]; then
  # 8 x 2 = 16 trajectories: the dispatcher chunks by worker count, so a smoke batch
  # smaller than rollout.agent.num_workers fails on an uneven chunk before training starts.
  BATCH=8; N=2; STEPS=1; EXP="smoke_$(date +%m%d_%H%M)"
  # Dynamic sampling off for the smoke: at G=2 almost every group is uniform and the
  # sampler would keep drawing batches instead of reaching the training step, which is
  # the whole point of the smoke. Sixteen episodes take about eight minutes, against
  # twenty-seven for a real step -- four separate defects in the training path were each
  # found by paying that twenty-seven minutes first.
  EXTRA="algorithm.filter_groups.enable=False"
else
  BATCH=8; N=8; STEPS=${STEPS:-20}
  EXTRA=""
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
  data.dataloader_num_workers=2 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=/root/probesql/models/sft_v2_ep3 \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=False \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.multi_stage_wake_up=True \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
  actor_rollout_ref.rollout.max_model_len=32768 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tau2 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=rl/agent_loop.yaml \
  actor_rollout_ref.rollout.agent.num_workers=32 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  trainer.use_v1=False \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=$STEPS \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.default_local_dir=ckpt/rl/$EXP \
  trainer.project_name=turncredit-rl \
  trainer.experiment_name=$EXP \
  trainer.logger='["console","tensorboard"]' \
  ${EXTRA:-} \
  2>&1 | tee logs/$EXP.log
