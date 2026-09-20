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

## Three metrics that were zero by construction (2026-09-18)

Arm 2's write-up leans on `ppo_kl` and `pg_clipfrac` being zero for ten steps. Two of those
three readings were worthless, and the reason is worth keeping.

With `ppo_epochs=1` and `ppo_mini_batch_size == train_batch_size`, the only forward pass of a
step runs on the same parameters that produced `old_log_prob`. The ratio is `exp(0) = 1` for
every token, so `ppo_kl` is zero and the clip fraction is zero **whatever the policy does**.
They measure the configuration, not the update. Only the gradient norm and the merged weight
delta survive as evidence, and those two still say what arm 2 concluded.

Setting `ppo_mini_batch_size=2` against `train_batch_size=8` gives four gradient updates per
batch and makes both metrics live again. It also restores verl's own 4:1 default -- the
earlier `ppo_mini_batch_size=$BATCH` was the non-standard setting, not this.

## The learning rate was four thousand times too small (2026-09-18)

`optim.lr=1e-6` was inherited from verl's full-parameter PPO example and never adapted to a
rank-16 LoRA adapter, although our own SFT trains the same adapter at 1e-4. The check is one
line: merge the trained adapter onto the base and take the largest weight change.

| | SFT, lr=1e-4 | arm 2, lr=1e-6 |
|---|---|---|
| max weight delta | 3.5e-03 | 9.3e-07 |

Arm 2b onward runs at `lr=2e-5`. This is the single change that turned arm 2's +3.3 points
(CI crossing zero) into arm 2b's +8.1.

## arm 2b and arm 3 — GRPO and shaped reward, 15 and 25 steps (2026-09-18)

Same protocol as arm 1 throughout: the frozen `base` split, 114 tasks, four rollouts, the
DeepSeek user simulator. Training draws only from `full \ base`. Every number below comes out
of `scripts/rl_stats.py`, which reads the saved runs and recomputes the table and the tests.

| | p^1 | p^2 | p^3 | p^4 | all_fail | turns | calls | dup | errors |
|---|---|---|---|---|---|---|---|---|---|
| arm 1 SFT | 75.9% | 61.0% | 50.2% | 41.2% | 2.6% | 17.7 | 5.7 | 0.09 | 0.0% |
| arm 2b s15 | 84.0% | 75.6% | 70.0% | 65.8% | 3.5% | 17.9 | 6.2 | 0.23 | 0.0% |
| arm 2b s25 | 83.6% | 74.0% | 67.3% | 62.3% | 2.6% | 16.8 | 5.8 | 0.11 | 0.2% |
| arm 3 s15 | 80.7% | 68.4% | 59.6% | 52.6% | 1.8% | 16.8 | 5.8 | 0.10 | 0.4% |
| **arm 3 s25** | **86.4%** | **77.8%** | **71.5%** | **66.7%** | 1.8% | 16.6 | 6.3 | 0.14 | 0.4% |

`turns`, `calls` and `dup` are over successful rollouts only, so a fast failure cannot flatter
them. `p^k` is tau2's own pass^k, the chance that all k of k drawn rollouts pass.

Paired bootstrap over tasks, 10,000 resamples:

```
  arm 2b s15 - SFT      +8.1%   CI [ +3.1%, +13.2%]  p=0.0012   43 better / 18 worse / 53 tied
  arm 2b s25 - SFT      +7.7%   CI [ +3.1%, +12.3%]  p=0.0014   43 / 19 / 52
  arm 3  s15 - SFT      +4.8%   CI [ +0.0%,  +9.6%]  p=0.058    37 / 23 / 54
  arm 3  s25 - SFT     +10.5%   CI [ +5.5%, +15.6%]  p<0.0001   49 / 21 / 44
  arm 3  s25 - arm 2b s25  +2.9%   CI [ -1.5%,  +7.2%]  p=0.22   27 / 23 / 64
  arm 3  s15 - arm 2b s15  -3.3%   CI [ -7.7%,  +1.1%]  p=0.15   18 / 31 / 65
```

**RL over the SFT checkpoint is real and arm 3 over arm 2b is not shown.** The +2.9 points at
step 25 and the -3.3 at step 15 have overlapping intervals and opposite signs; one seed and a
measured noise floor of +/-2.5 points between two snapshots of the same run do not separate
them. The honest claim is that the shaped reward did not cost anything and that its 25-step
result is the best single number in the table.

What did move under the shaped reward is the thing it was designed to move, and judging it by
pass^1 was the wrong yardstick from the start: arm 3's efficiency term multiplies `R`, so it
only ranks trajectories that already succeeded. Paired over the tasks both arms solved, over
successful rollouts only:

```
  arm 3 s15 vs arm 2b s15   (110 tasks)
      turns             18.30 -> 17.39    -5.0%   p=0.0040
      tool calls         6.35 ->  6.01    -5.3%   p=0.0004
      duplicate calls    0.26 ->  0.10   -61.8%   p=0.0002
  arm 3 s25 vs arm 2b s25   (109 tasks)
      turns             16.97 -> 16.55    -2.5%   p=0.15
      tool calls         5.88 ->  6.42    +9.1%   p<0.0001
      duplicate calls    0.13 ->  0.19   +45.1%   p=0.19
```

**At step 15 the shaped reward buys a 5.0% shorter solution and 61.8% fewer repeated calls at
no cost in pass^1.** That is the claim the design supports, and it is a continuous paired
measurement rather than a 114-task binomial, which is why it clears significance where the
pass^1 comparison cannot.

By step 25 the efficiency channel has stopped paying and the correctness channel has taken
over -- arm 3 is making *more* tool calls than arm 2b while scoring higher. Arm 2b shows the
mirror image on its own: from s15 to s25 it loses 0.4 points of pass^1 and gains a 5.7%
shorter solution. Both arms end up spending their budget on whichever channel still has room,
and 25 steps is where the two cross. Arm 3 is the only arm that improves with the extra ten
steps (+5.7 points, CI [+1.8%, +9.9%], p=0.005); arm 2b is flat (-0.4 points, p=0.87).

Neither `all_fail` moves out of the 1.8-3.5% band and protocol errors stay under 0.5%, so
none of this is a health artefact.

## Can a per-turn credit signal be recovered at all? (2026-09-19)

Arm 4 assigns credit to the turn where an expected action completed, which needs a per-turn
hit signal. The first design read it off the `reward_info` the environment returns each step.
That does not work: `AgentGymEnv._get_reward` returns `{}` while `_simulation_run is None`, and
`_simulation_run` is assigned only in the orchestrator thread's `finally` -- that is, when the
episode ends. `reward_info` is empty on every non-terminal step.

The replacement needs no environment support. `ActionEvaluator` matches gold actions by set
membership over the tool calls in the trajectory, so it is a pure function of the message
prefix and monotone by construction; the agent loop already holds the history it is building
and can replay the comparison incrementally for nothing. `scripts/turn_hit_probe.py` runs that
replay over the 456 saved episodes of arm 3 s25:

```
  expected actions per task           4.65 mean, 12 max
  distinct turns carrying a hit       3.99 mean, 4 median, 11 max
  first-to-last hit spread            7 turns median, 18 at p90
  hits landing on the final turn      4.4%
  relative position of a hit          0.54 median, 0.25 at p10, 0.93 at p90
  decile histogram                    [1, 91, 217, 264, 281, 250, 207, 153, 197, 292]
```

**Only 4.4% of hits land on the last turn, so this signal cannot be recovered from the
terminal reward** -- which is the whole case for arm 4. Arm 2b's episodes give 3.91 / 7 / 3.1%
/ 0.52, so the spread is a property of the task and not of arm 3's reward. The first decile
holds 1 hit out of 1953, confirming on a larger sample that discounting a terminal reward
backwards is the wrong direction here.

Two findings change the design:

**77% of hits happen in a user message** (1513 user against 440 assistant), because three
quarters of telecom's expected actions are performed by the customer on the handset. Those
tokens are zeros in `response_mask` and can carry no gradient, so credit has to be attributed
backwards to the assistant turn that asked for the action.

**The degenerate cases are exactly the single-action tasks** -- 100% of the m=1 tasks put all
hits in one turn, against 4% at m=2 and 0% at m>=4. It is structural, not a signal defect.

Finally, the claim that arm 4 revives groups that dynamic sampling discards is weaker than it
looked. 94 of 114 tasks have all four rollouts succeeding, but only **one** of those groups
also has identical turn counts, which is what it takes for arm 3's advantage to be exactly
zero -- the group-relative efficiency term is doing more work than expected. Inside all-pass
groups the local term's pooled standard deviation is 0.0323 against arm 3's typical 0.105,
so arm 4 adds about 31% on top of an existing signal rather than rescuing a dead one.

## Correction: arm 3 was not Dr.GRPO (2026-09-19)

`algorithm.norm_adv_by_std_in_grpo=False` is on every run's command line, and arm 2 obeyed it.
Arm 3 did not. `compute_advantage` passes the flag into verl's own GRPO branch but builds
`adv_kwargs` for a *registered* estimator from the batch alone, so `grpo_efficiency` took its
own signature default of `True` and arm 3 ran as standard GRPO.

Nothing raised an error. The only trace was in a metric nobody had a reason to read:

| | max advantage | un-normalised bound |
|---|---|---|
| arm 2b, built-in `grpo` | 0.875 | 0.875 = 1 - 1/8, exactly one success in a group of eight |
| arm 3, registered `grpo_efficiency` | 2.44 | 1.3 |

Arm 2b sits exactly on the bound; arm 3 sits at 1.9x it. Arm 4's first smoke read 5.95, which
is what finally sent me looking.

Two consequences, and only the second changes anything already written:

- Arm 3's numbers stand. Standard GRPO is a legitimate configuration, and the run is
  internally consistent with itself.
- **The arm 2b vs arm 3 comparison is two variables, not one**: shaped reward *and* advantage
  normalisation. The +2.9% at step 25 and -3.3% at step 15 were already inside the noise band,
  so no conclusion moves, but neither difference can be attributed to reward shaping alone.

The estimators now read the flag off the config they are handed, and the runner pins each arm
to what it actually did rather than to what reads well. Arm 4 keeps `True` deliberately: its
whole design is to sit one variable from the arm 3 run that is already measured, and switching
normalisation at the same time would cost that.

The same investigation found that `logger.warning` from `rl/` never reached the run logs at
all, which is why the agent loop's per-episode marker had been invisible for a whole run.
Diagnostics now print.

## arm 4 — the reward, redistributed (2026-09-19)

Arm 4 changes where a trajectory's reward sits, not how much it is. Two identities hold by
construction and are what let it be compared against the existing arm 3 run with no matched
control:

```
  sum over tokens of the reward   == arm 3's score for that trajectory
  mean over turns of the advantage == arm 3's advantage for that trajectory
```

The per-turn weight is built from *where the expected actions completed*, which is an
observation rather than a payment -- a successful episode earns the same 1.0 whichever turn it
did the work on. Reading positions instead of rewarding them is what gives arm 4 a signal on
the 86% of rollouts that succeed, where arm 3's `(1 - R)` partial-credit term is identically
zero. `beta` is the only knob; `beta=0` reproduces arm 3 bit for bit, which is a test rather
than an argument.

```
  r_k     = actions first completed at turn k / total hits
  L_k     = r_k + 0.3 * L_{k+1}                     lam=0.3
  L_k    += 0.5 * mean(L)                           a turn after the last hit is not worthless
  shape_k = water_fill(L_k / mean(L), cap=3.0)      mean(shape) == 1
  G_k     = S_i * ((1 - beta) + beta * shape_k)     beta=0.35
```

`water_fill` replaces the obvious clip-then-renormalise, which does not work and looks as
though it does: clipping lowers the mean, so dividing by it scales every entry back up
*including the one just clipped*, and a cap of 3.0 was letting 12.0 through as 4.9. Moving the
excess to the entries still under the cap keeps the mean at one and makes the cap real. It
binds only on the single-action tasks, which are 14% of episodes and the only ones that put
every hit in one turn. With it, `sd(shape)` over the 430 replayed episodes is 0.827, and
`beta = 0.31 / 0.827` rounds to 0.35.

Verified on 32 live trajectories before the run started:

```
  CA3 check: turns/traj=19.12 turn_adv_var=1.4671 max|adv|=5.716
             norm_by_std=True sum_drift=1.19e-07
  critic/score/max 1.0000001   actor/ppo_kl 2.1e-05   actor/entropy 0.611
```

`sum_drift` is the check that matters. `G_k` is recovered downstream by multiplying a turn's
share by the turn count, so if the agent loop divided by a different count than `turn_spans`
finds in the trainer, every advantage would be scaled by the ratio and nothing would say so.
At 1.19e-07 it is float32 noise: the two agree. `turn_adv_var` being non-zero is the other
half -- turns are actually differentiated rather than quietly collapsing back to arm 3.

One honest caveat. `turn_adv_var=1.47` is stronger than the 0.31x the design aimed at, because
std normalisation amplifies everything inside a low-variance group and all-pass groups are
exactly those. The balance *within* a group is unchanged, and arm 3 carries the same
normalisation, so the comparison stays one variable -- but beta is not doing quite what its
derivation says, and the three diagnostics are what will show whether that matters.

## arm 4 — turn-level credit, measured (2026-09-19)

Twenty-five steps, `beta=0.35`, everything else identical to arm 3. This is the only
single-variable comparison in the project: the reward function is byte-identical (the row sum
is arm 3's score, checked every step at `sum_drift` 1.0-1.3e-07), the advantage normalisation
is the same, dynamic sampling is off in both, and `beta=0` reproduces arm 3 bit for bit in the
tests. What changed is where a trajectory's reward sits.

| | p^1 | p^2 | p^3 | p^4 | all_fail | turns | calls |
|---|---|---|---|---|---|---|---|
| arm 1 SFT | 75.9% | 61.0% | 50.2% | 41.2% | 2.6% | 17.7 | 5.7 |
| arm 2b s25 | 83.6% | 74.0% | 67.3% | 62.3% | 2.6% | 16.8 | 5.8 |
| **arm 3 s25** | **86.4%** | **77.8%** | **71.5%** | **66.7%** | 1.8% | 16.6 | 6.3 |
| arm 4 s15 | 81.4% | 69.4% | 61.0% | 54.4% | 1.8% | 17.6 | 5.7 |
| arm 4 s25 | 82.0% | 71.3% | 64.0% | 58.8% | 2.6% | 17.6 | 5.8 |

```
  arm 4 s15 - arm 3 s15   +0.7%   CI [-3.5%, +5.0%]   p=0.80    29 / 29 / 56
  arm 4 s25 - arm 3 s25   -4.4%   CI [-9.2%, +0.2%]   p=0.075   18 / 34 / 62
  arm 4 s25 - arm 1 SFT   +6.1%   CI [+1.3%, +11.0%]  p=0.017
```

**Turn-level credit did not improve pass^1**, flat at step 15 and negative but not significant
at step 25. It did something else, in the same direction at both step counts and growing with
training:

```
                       total turns        tool turns          message turns
  arm 4 s15 vs arm 3   +4.5%  p=0.0038    +1.6%  p=0.33    ->  +5.7%   p=0.0036
  arm 4 s25 vs arm 3   +8.2%  p<0.0001    +1.0%  p=0.54    ->  +11.3%  p<0.0001
```

**Every extra turn is a turn spent talking to the customer. Backend tool turns do not move at
all**, at either step count. Successful arm 4 trajectories run 1.36 turns longer than arm 3's
and make no more backend calls for it.

The cause is the attribution rule, not the mechanism. Telecom is dual control and 76% of its
expected actions are performed by the customer on the handset, inside a user message whose
tokens are zeros in `response_mask` and can carry no gradient. Credit for them is therefore
attributed backwards, to the assistant turn that asked for the action -- which is a message
turn. Arm 4 puts its largest advantages there, and the policy generalised the wrong invariant:
not *say this, here*, but *produce more customer-facing instruction turns*.

Three things say the mechanism itself worked and the result is a property of the design:

- `sum_drift` stayed at 1.0-1.3e-07 for all 25 steps, so the reward function never drifted
  from arm 3's.
- `turn_advantage_var` ran between 0.11 and 4.70 and never trended to zero, so turns stayed
  differentiated rather than quietly collapsing back to a single scalar.
- The effect grows monotonically from s15 to s25 (+5.7% -> +11.3%), which noise does not do.

**What arm 4 is worth reporting for is the negative result and its mechanism**: on a
dual-control task, turn-level credit shifts the policy toward whichever action type the
attribution rule points at. The obvious next design is to stop crediting the asking turn and
instead credit the turn by what the environment state did, which does not privilege one side
of the dual control -- but that is a different experiment, not a tweak.

`turn_advantage_var` also recorded the difficulty bias that Dr.GRPO describes, as a by-product.
It tracks `max|adv|` exactly and splits into two regimes: easy batches (reward ~0.75, 18-21
turns) give 1.0-2.4, hard batches (reward 0.41-0.57, 21-26 turns) give 0.11-0.29. The same
credit signal receives a gradient weight that differs 5-10x depending on how hard the batch
happened to be, which is what dividing by the group standard deviation does.

## arm 3dr — the shaped reward without the normalisation (auto-generated 2026-09-19 20:35)

Same reward as arm 3, `norm_adv_by_std_in_grpo=False`, everything else identical. This
separates "the shaped reward did this" from "dividing by the group std did this".

```
114 tasks shared by all arms, 4 rollouts each

                p^1     p^2     p^3     p^4  all_fail   turns   calls   dups  errors
arm1 SFT     75.9%  61.0%  50.2%  41.2%      2.6%    17.7     5.7   0.09    0.0%
arm2b s15    84.0%  75.6%  70.0%  65.8%      3.5%    17.9     6.2   0.23    0.0%
arm2b s25    83.6%  74.0%  67.3%  62.3%      2.6%    16.8     5.8   0.11    0.2%
arm3  s15    80.7%  68.4%  59.6%  52.6%      1.8%    16.8     5.8   0.10    0.4%
arm3  s25    86.4%  77.8%  71.5%  66.7%      1.8%    16.6     6.3   0.14    0.4%
arm4  s15    81.4%  69.4%  61.0%  54.4%      1.8%    17.6     5.7   0.17    0.2%
arm4  s25    82.0%  71.3%  64.0%  58.8%      2.6%    17.6     5.8   0.15    0.0%
arm3dr s15   78.3%  65.8%  57.2%  50.9%      3.5%    17.8     6.0   0.10    0.9%
arm3dr s25   86.2%  76.2%  68.2%  61.4%      0.9%    17.3     6.1   0.11    0.0%

paired bootstrap on per-task pass^1, 10,000 resamples over tasks
  arm2b s25 - arm1 SFT      +7.7%  95% CI [ +3.1%,+12.3%]  p=0.0014 * better/worse/tied 43/19/52
  arm3  s25 - arm1 SFT     +10.5%  95% CI [ +5.5%,+15.6%]  p=0.0000 * better/worse/tied 49/21/44
  arm4  s25 - arm1 SFT      +6.1%  95% CI [ +1.3%,+11.0%]  p=0.0168 * better/worse/tied 43/21/50
  arm3  s15 - arm2b s15     -3.3%  95% CI [ -7.7%, +1.1%]  p=0.1548   better/worse/tied 18/31/65
  arm3  s25 - arm2b s25     +2.9%  95% CI [ -1.5%, +7.2%]  p=0.2160   better/worse/tied 27/23/64
  arm4  s15 - arm3  s15     +0.7%  95% CI [ -3.5%, +5.0%]  p=0.8046   better/worse/tied 29/29/56
  arm4  s25 - arm3  s25     -4.4%  95% CI [ -9.2%, +0.2%]  p=0.0750   better/worse/tied 18/34/62
  arm4  s15 - arm2b s15     -2.6%  95% CI [ -6.4%, +1.1%]  p=0.1892   better/worse/tied 21/29/64
  arm4  s25 - arm2b s25     -1.5%  95% CI [ -5.9%, +2.9%]  p=0.5238   better/worse/tied 20/29/65
  arm4  s25 - arm4  s15     +0.7%  95% CI [ -3.5%, +4.8%]  p=0.8026   better/worse/tied 26/24/64
  arm3dr s15 - arm2b s15     -5.7%  95% CI [-10.3%, -1.3%]  p=0.0118 * better/worse/tied 19/38/57
  arm3dr s25 - arm2b s25     +2.6%  95% CI [ -0.9%, +6.1%]  p=0.1744   better/worse/tied 25/19/70
  arm3  s15 - arm3dr s15    +2.4%  95% CI [ -1.8%, +6.8%]  p=0.2718   better/worse/tied 30/25/59
  arm3  s25 - arm3dr s25    +0.2%  95% CI [ -3.7%, +4.2%]  p=0.9538   better/worse/tied 28/25/61
  arm3dr s25 - arm1 SFT     +10.3%  95% CI [ +5.9%,+14.9%]  p=0.0000 * better/worse/tied 42/12/60

efficiency, paired over tasks both arms solved at least once, successful rollouts only
  arm3dr s15 vs arm2b s15   (109 tasks)
      turns             18.18 ->  18.03   -0.8%  95% CI [ -0.76, +0.45]  p=0.6280 
      tool calls         6.33 ->   6.11   -3.4%  95% CI [ -0.42, -0.02]  p=0.0320*
      duplicate calls    0.26 ->   0.16  -39.1%  95% CI [ -0.20, -0.01]  p=0.0340*
  arm3dr s25 vs arm2b s25   (110 tasks)
      turns             17.06 ->  17.53   +2.7%  95% CI [ -0.10, +1.03]  p=0.1026 
      tool calls         5.89 ->   6.16   +4.7%  95% CI [ +0.10, +0.45]  p=0.0020*
      duplicate calls    0.13 ->   0.12   -8.1%  95% CI [ -0.07, +0.05]  p=0.7372 
  arm3  s15 vs arm2b s15   (110 tasks)
      turns             18.30 ->  17.39   -5.0%  95% CI [ -1.57, -0.27]  p=0.0040*
      tool calls         6.35 ->   6.01   -5.3%  95% CI [ -0.53, -0.14]  p=0.0004*
      duplicate calls    0.26 ->   0.10  -61.8%  95% CI [ -0.25, -0.08]  p=0.0002*
  arm3  s25 vs arm2b s25   (109 tasks)
      turns             16.97 ->  16.55   -2.5%  95% CI [ -1.00, +0.14]  p=0.1468 
      tool calls         5.88 ->   6.42   +9.1%  95% CI [ +0.35, +0.73]  p=0.0000*
      duplicate calls    0.13 ->   0.19  +45.1%  95% CI [ -0.03, +0.16]  p=0.1930 
  arm4  s15 vs arm3  s15   (111 tasks)
      turns             17.34 ->  18.12   +4.5%  95% CI [ +0.23, +1.33]  p=0.0038*
      tool calls         6.04 ->   5.91   -2.2%  95% CI [ -0.31, +0.05]  p=0.1460 
      duplicate calls    0.11 ->   0.19  +76.2%  95% CI [ -0.00, +0.18]  p=0.0570 
  arm4  s25 vs arm3  s25   (109 tasks)
      turns             16.63 ->  17.99   +8.2%  95% CI [ +0.75, +2.02]  p=0.0000*
      tool calls         6.40 ->   5.99   -6.3%  95% CI [ -0.58, -0.23]  p=0.0000*
      duplicate calls    0.17 ->   0.18   +6.0%  95% CI [ -0.08, +0.10]  p=0.8268 
```
