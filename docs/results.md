# Results

Every number here is measured under the frozen protocol in `docs/eval_protocol.md`:
telecom `base`, all 114 tasks, 4 rollouts, DeepSeek user simulator at temperature 0,
Qwen3-8B served at 32k with `enable_thinking=false`.

## arm 0 — Qwen3-8B zero-shot (2026-09-16)

```
  fix-it tasks        9.3% +/- 1.6%   (94 tasks)
  escalate tasks     20.0% +/- 3.4%   (20 tasks)
  OVERALL pass@1     11.2% +/- 1.5%    resolves ~4.2% differences

  all_fail / mixed / all_pass     62.3% / 37.7% / 0.0%

  protocol errors    62.3%   (3439/5520 agent tool results)
  killed by errors   64.1%
  mean turns         22.9
  mean tool calls    12.3
  termination        too_many_errors 288 · user_stop 153 · max_steps 8
```

**The dominant failure is protocol, not capability.** 62.3% of the agent's tool results are
`Tool 'X' not found` — it calls the thirty tools that belong to the user's phone, having only
thirteen backend APIs of its own. A representative episode: ten calls, all the same error,
repeating the first five verbatim from the sixth onward, without once speaking to the user.

**Escalate tasks pass at twice the rate of fix-it tasks** (20.0% vs 9.3%) while the RL training
pool contains almost none of them (0.6% against the benchmark's 17.5%). A policy trained to
persist will lose them, so a real gain on the other 82% can read as a drop in the headline.

**The mixed bucket is 37.7%**, so more than a third of tasks carry a non-zero GRPO advantage.

### Why this differs from the pilot and from MUA-RL

The pilot measured 17.5% on a random 60-task draw; the full 114 give 11.2%. The 6.3-point gap
is task sampling — **a 60-task result cannot serve as a baseline**, which is why arm 0 was rerun.

MUA-RL reports 19.1 for Qwen3-8B zero-shot on telecom, 7.9 points above this. Candidate
explanations, unresolved: they may evaluate at temperature 0 where this protocol uses 1.0, and
telecom is out-of-distribution for them in a way that does not change the zero-shot number but
might change how they prompt. Worth one controlled check before the number is used for anything.

## Teacher sampling for SFT (2026-09-16)

DeepSeek, chosen over Qwen3-14B on pre-registered gates measured at 15 tasks:

| | Qwen3-8B (student) | Qwen3-14B | DeepSeek |
|---|---|---|---|
| protocol error rate | 62.3% | 65.8% | **0.0%** |
| acceptance | — | 3.3% | **71.8%** |
| median turns | 22.9 | 16 | **15** |

Scale within the Qwen3 family does not fix the protocol confusion: the 14B calls the user's
tools exactly as the 8B does, and 60 rollouts yielded two usable trajectories.

Full run: 600 rollouts over 150 stratified tasks, 431 accepted (71.8%), **269 examples covering
141 tasks**. Strict and lenient acceptance differ by one rollout. Encoded: zero dropped at 32k,
median 9,611 tokens, 11.1% of tokens in the loss, 297,767 learned tokens total, and no tool
response, scaffolding, think block or policy document in any learned span.

Stratification cannot fully repair the pool: it drains both scarce strata dry and still reaches
only 8.5% escalate tasks and 12% `service_issue`, against the benchmark's 17.5% and 25%. There
are twelve and eighteen such tasks in the entire pool.

## D7 — the user simulator is part of the environment (2026-09-16)

Same Qwen3-8B agent, same tasks, different user simulator:

| | DeepSeek user | local Qwen3-4B user |
|---|---|---|
| user-side tool error rate | 0.1% | **0.0%** |
| pass@1 | 17.5% (60-task pilot) | **0.0%** (68 episodes) |
| protocol errors (agent side) | 54% | 49% |
| **success among normally-terminating episodes** | **38.7%** | **0.0%** |
| `###OUT-OF-SCOPE###` | 0.2% | 4.4% |

The 4B operates its thirty device tools flawlessly and still makes every task unsolvable. It
cannot improvise inside a persona: asked for a date of birth its instructions do not mention,
it ends the conversation, where DeepSeek offers an account PIN instead. Agent-side protocol
errors are unchanged, so the two failures are additive, not the same one.

The consequence is a constraint, not a curiosity: the user simulator is frozen as DeepSeek for
SFT, RL and evaluation alike. Swapping it to save API spend would invalidate the SFT data and
every number above, and risks an empty mixed bucket — no gradient, no project.

## arm 1 — SFT cold start, full protocol (2026-09-17)

Qwen3-8B + LoRA(r=16) SFT v2, epoch 3, merged. 114 base tasks x 4 rollouts, DeepSeek user at
T=0, thinking off, 32k. Same frozen protocol as arm 0.

| | arm 0 (base) | arm 1 (SFT v2 ep3) |
|---|---|---|
| pass@1 | 11.2% | **75.9% +/- 2.5%** |
| fix-it / escalate | - | 75.5% / 77.5% |
| protocol error rate | 62.4% | **0.3%** |
| episodes killed by errors | - | 0.0% |
| termination | - | 456/456 normal (user stop) |
| mean turns / tool calls | - | 18.5 / 5.7 |

The headline number is not the interesting one. What SFT bought is the protocol: 62.4% of
baseline episodes died on tool-call format rather than on reasoning, and that is now 0.3%.

**The number that sets up arm 2 is pass^4 = 41.2%** against pass^1 = 75.9%. The policy knows
how to solve these tasks and cannot do it four times running. Consistency, not capability, is
what is left on the table, and that is what a group-relative method is for.

Group composition at G=4, which decides how much of an RL batch carries gradient:

| bucket | share | consequence |
|---|---|---|
| all_pass | 41.2% | zero advantage; wasted rollouts |
| **mixed** | **56.1%** | the only groups that train |
| all_fail | 2.6% | zero advantage under a binary reward |

Dynamic sampling therefore discards 43.9% of sampled groups, a **1.78x cost multiplier** on the
user-simulator API. At G=8 the mixed share rises, so 1.78x is an upper bound.

The 41.2% all_pass share is a statement about the training pool, not about the reward: after SFT
the pool is too easy. Reward shaping cannot fix it; difficulty-filtering the pool can.

## Entropy — SFT did not collapse the policy (2026-09-17)

Mean token entropy over assistant spans on held-out trajectories, same mask as training:

| model | entropy (nats) |
|---|---|
| Qwen3-8B base | 0.281 |
| SFT epoch 1 | 0.414 |
| SFT epoch 2 | 0.376 |
| **SFT epoch 3** | **0.357** |

Entropy falls monotonically across epochs but stays above the base model at epoch 3. A collapsed
policy has nothing for GRPO to explore; this one still has spread. This is the measure that
answers "is epoch 3 overtrained", and the loss curve is not -- training loss keeps falling in a
model whose output distribution has already degenerated.

## Training pool — the process reward is coarser than assumed (2026-09-17)

Over the 2,171 tasks of `full\base`, `env_assertions` per task is 1 (985 tasks), 2 (1,078) or
3 (108). So `progress_k` takes two values on 45% of the pool, three on 50%.

**On the 45% with a single assertion, the process reward is identical to the binary terminal
reward** and carries no extra information. The dense signal in this domain is
`evaluation_criteria.actions` (median 6 per task, max 11), not the assertions. Arm 3 has to be
built on `action_hit_k`, with `delta progress` as the secondary term, not the reverse.

## arm 2 — GRPO, ten steps (2026-09-17)

Ten steps of GRPO on top of the SFT checkpoint: eight tasks per step, eight rollouts each,
outcome reward only (`w_delta=0`, CA-0), advantage not normalised by group std, clip-higher at
0.28, token-level loss, dynamic sampling on. Two hours four minutes, no errors after the run
started.

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| reward | .562 | .453 | .469 | .453 | .766 | .641 | .625 | .672 | **.266** | .469 |
| entropy | .602 | .657 | .632 | .626 | .586 | .652 | .618 | .589 | .656 | .651 |
| grad norm | .011 | .011 | .010 | .011 | .010 | .010 | .010 | .010 | .010 | .010 |
| response len | 3194 | 3082 | 3244 | 3209 | 2589 | 2869 | 2771 | 3298 | 3736 | 3353 |

**The training reward has no trend.** It ranges over fifty points and ends where it started.
Steps five through eight looked like a rise; step nine at 0.266 and step ten at 0.469 ended
that reading. With eight tasks per step, which tasks get drawn dominates the number -- the
batch, not the policy, is what moves between steps.

**The policy barely moved.** Gradient norm is 0.010 or 0.011 at every step, clip fraction and
`ppo_kl` are zero throughout, and merging the trained adapter back onto the base changes a
weight by at most 1.9e-06. At lr=1e-6 on a LoRA adapter, ten steps do not reach the policy.
Two things follow for the next run, both with numbers behind them rather than taste: raise the
learning rate, since LoRA carries the whole update in a rank-16 subspace, and raise the tasks
per step, since AReaL's telecom runs used 8x64 against our 8x8.

**Entropy held at 0.59-0.66**, above the 0.357 the SFT checkpoint started from, so there is no
collapse and nothing was lost by exploring.

`rollout_corr/kl` sat at 0.0002-0.0011: the rollout engine and the trainer agree on the
log-probabilities of the same sequence almost exactly. In this configuration the
rollout-training mismatch a TIS correction targets is small, which is itself worth reporting.

### arm 2 under the frozen protocol

A first attempt reported 18.6% and was void: the user simulator's account ran out of balance
mid-run, 346 of 456 episodes ended in `infrastructure_error`, and mean turns fell to 4.2
against arm 1's 18.5. It is kept as `outputs/eval_arm2.json.balance_failed.bak` because it is
a clean example of an environment failure wearing the costume of a capability regression --
the headline number alone looks like a catastrophic drop. The re-run below was checked for
health before its score was read: zero balance errors, mean turns 18.6, and 455 of 456
episodes ending in `user_stop`.

| | arm 0 | arm 1 | arm 2 |
|---|---|---|---|
| pass@1 | 11.2% | 75.9% +/- 2.5% | **79.2% +/- 2.5%** |
| fix-it (94 tasks) | - | 75.5% | 77.1% |
| escalate (20 tasks) | - | 77.5% | 88.8% |
| all_pass | - | 41.2% | 50.9% |
| all_fail | - | 2.6% | 3.5% |
| protocol errors | 62.4% | 0.3% | 0.2% |

**The +3.3 points are not distinguishable from zero.** Both arms ran the same 114 tasks, so
the comparison can be paired, which takes task difficulty out of the variance:

```
paired bootstrap over tasks, 10,000 resamples
  difference  +3.3%   95% CI [-1.6%, +8.2%]
  per task    arm 2 better on 31, worse on 23, tied on 60
  sign test   p = 0.34
```

That agrees with everything the training run said: gradient norm fixed at 0.010 for ten steps,
`ppo_kl` and clip fraction at zero throughout, and a largest weight change of 1.9e-06. Three
independent measurements, one conclusion -- at lr=1e-6 with eight tasks a step, ten steps of
GRPO do not move the policy. The escalate split (77.5% -> 88.8%) and the all_pass share
(41.2% -> 50.9%) both point the right way and neither survives alone: escalate is twenty
tasks, and all_pass comes from the same rollouts as the headline.

Arm 2's value is not a score. It is a working single-GPU agentic RL loop -- rollout against a
live user simulator, LoRA weight sync into vLLM, GRPO update, checkpoint, merge, evaluation --
and two parameters to change next, each with a measurement behind it rather than a preference.
