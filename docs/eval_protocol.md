# Evaluation protocol (frozen 2026-09-16)

Every arm is measured this way. Changing any line below makes earlier numbers incomparable,
so it does not change mid-project — that is the whole point of writing it down before arm 0.

## Setting

| | value | why |
|---|---|---|
| task set | telecom **`base`, all 114 tasks** | the standard split; the one τ²'s leaderboard verifier loads and the one MUA-RL's 19.1 / 21.8 refer to. Never trained on: the RL pool is `full ∖ base`. |
| rollouts | **4 per task** (456 episodes) | bootstrapped from the pilot, the standard error is 2.33% at 4 rollouts and 2.11% at 8. Noise is dominated by *task* sampling, not rollout sampling, so doubling rollouts buys 0.2 points of precision at twice the cost. Spend the budget on covering all 114 tasks instead. |
| agent temperature | 1.0 | matches rollout conditions; we report empirical pass@1, not peak accuracy |
| user simulator | **DeepSeek**, temperature 0.0 | frozen. It is part of the environment: a local Qwen3-4B was measured and, among episodes that terminated normally, solved 0.0% against DeepSeek's 38.7%. Swapping it invalidates both the SFT data and every number here. |
| thinking | `enable_thinking=false` | with thinking on, Qwen3-8B answers the same prompt with a clarifying question and no tool call at all |
| context | 32k | one pilot episode reached 16,132 tokens against a 16,384 limit; overruns return as agent errors scoring zero |
| turn budget | tau2 default `max_steps=200`, `max_errors=10` | MUA-RL's setting, left alone |

## What gets reported, every time

The headline number alone is misleading here, because the benchmark and the training pool
disagree about what kind of task matters.

```
pass@1 overall
  ├─ on "fix it" tasks          (94 of 114)
  └─ on "escalate" tasks        (20 of 114, 17.5%) -- expected action is transfer_to_human_agents
pass^4
protocol error rate             -- share of agent tool results that are "Tool 'X' not found"
too_many_errors share           -- episodes killed by hitting max_errors
mean turns / mean tool calls
termination mix
```

The split matters because the RL training pool is 0.6% escalate tasks against the benchmark's
17.5%. A policy trained to persist will lose the escalate tasks the base model currently gets
partly by accident (30.7% vs 14.5% at baseline), so a real gain on 82% of the benchmark can
show up as a *drop* in the headline number. Always read the two rows before the total.

## Resolution — what this protocol can and cannot detect

Bootstrapped on the pilot's measured per-task distribution:

| configuration | SE | smallest detectable difference (2.8 SE) |
|---|---|---|
| 114 x 4 | 2.33% | **6.5 points** |
| 114 x 8 | 2.11% | 5.9 points |
| 114 x 16 | 1.96% | 5.5 points |

**The plan's original gate — "arm 2 beats arm 1 by >= 2 points" — is not statistically
reachable at any affordable rollout count.** MUA-RL's own 8B gain on telecom is 2.7 points,
which their evaluation could not have resolved either; they report no interval and no seeds.

So pass@1 is the *comparable* number, not the *decisive* one. Arm-to-arm decisions use the
continuous metrics above, which have far lower variance than a binary outcome: protocol error
rate, mean turns, and (from M4) the per-turn assertion-satisfaction rate that the process
reward is built on. Report pass@1 with its interval and say plainly when a difference is
inside the noise.

## Cost

456 episodes per evaluation. Agent runs locally on vLLM; the user simulator is the only paid
component, at roughly 0.037-0.082 CNY per episode measured on the pilot — 17 to 37 CNY per arm.
