# Wiring τ²-bench into veRL

Read off verl 0.9.0's wheel before installing anything, so the RL stage starts from a known
shape rather than a guess. Nothing here is running yet.

## The environment split, and why

| env | holds | used for |
|---|---|---|
| `rl` | torch 2.13, vllm 0.28, **transformers 5.16.1** | SFT training, evaluation serving |
| `verl` (new) | verl 0.9.0, **transformers <5.11**, ray | RL training |
| tau2's `.venv` | python 3.12, litellm | the benchmark itself |

verl pins `transformers>=5.5.3,<5.11`. `sft/data.py`'s loss mask was written against 5.16's
chat-template behaviour — the empty `<think>` block Qwen3 prefills, and `warmup_steps`
accepting a fraction — so installing verl into `rl` would downgrade transformers underneath
the one module that must not move. Three envs, handing off through model files on disk.

`bash scripts/setup_box.sh verl` creates it.

## Three interfaces that matter

**`verl/tools/base_tool.py`** maps onto tau2 almost one-for-one:

| verl | tau2 |
|---|---|
| `create(instance_id)` | `AgentGymEnv.reset()` — one env per rollout |
| `execute(instance_id, parameters) -> (ToolResponse, reward, metrics)` | `env.step(action)` |
| `calc_reward(instance_id)` | `Π(env_assertions)` |
| `release(instance_id)` | tear the env down |

`execute` returning a reward and a metrics dict is where the per-turn process signal goes:
`Δ_k = progress_k − progress_{k−1}` is computed there and handed back, with no change to
verl's loop.

**`verl/experimental/agent_loop/tool_agent_loop.py`** is the multi-turn driver. It already
does what the pilot's runner does by hand: generate, parse a tool call, execute, append the
observation, repeat.

**`verl/trainer/ppo/core_algos.py`** holds the advantage estimators, and they are a plugin
registry — fourteen of them, each declared with `@register_adv_est("name")`. Two consequences:

- **CA-1 and CA-2 are two new registered functions plus a config string**, not a fork of
  verl. This is the project's main deliverable and it turns out to need no surgery.
- `norm_adv_by_std_in_grpo=False` in the existing GRPO estimator *is* Dr.GRPO's de-biasing,
  so one of the planned ablation switches already ships.

The GRPO signature carries everything the turn-level schemes need:

```python
compute_grpo_outcome_advantage(
    token_level_rewards,   # (bs, response_length)
    response_mask,         # (bs, response_length) — its segment structure *is* the turn boundaries
    index,                 # groups the G rollouts of one task
    norm_adv_by_std_in_grpo,
)
```

Turn boundaries do not need to be passed separately: they are recoverable from
`response_mask`, exactly as they are in `sft/data.py`'s mask.

## Async rollout

`verl/experimental/fully_async_policy/fully_async_rollouter.py` ships. It matters here more
than it would elsewhere: an episode runs 20–50 turns and every turn waits on a DeepSeek
round-trip for the user simulator, so a synchronous rollout leaves the GPU idle for most of
the wall clock. Worth measuring the throughput difference once the synchronous path works —
correctness first, then speed.

## Order of work

1. `setup_box.sh verl`, confirm it imports and does not touch the `rl` env
2. Wrap tau2 as a `BaseTool`; get one episode through `tool_agent_loop` end to end
3. Terminal reward only, CA-0, short run — that is arm 2
4. Process reward in `execute` — arm 3
5. Register `ca1_discounted_turn` and `ca2_position_normalized` — arms 4a/4b
