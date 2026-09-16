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
