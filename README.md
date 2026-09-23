# Post-training a dual-control tool agent

Qwen3-8B on **τ²-bench telecom**, one A800. The agent diagnoses a phone's connectivity fault by
reading a policy manual and calling thirteen backend APIs — but **76% of the actions the task
expects must be performed by the customer**, on a handset the agent cannot touch, so it has to
talk a simulated user through them. τ² calls this dual control, and it is what makes turn-level
credit assignment a real question here: turns do genuinely different things.

The reward is fully programmatic — `R = Π(env_assertions)`, a product of deterministic Python
assertions over device state, with no LLM judge anywhere.

![what each stage bought](docs/assets/ladder.png)

| | pass^1 | pass^4 | protocol errors | mean turns |
|---|---|---|---|---|
| Qwen3-8B base | 11.2% | 0.0% | 62.4% | 22.9 |
| + SFT | 75.9% | 41.2% | 0.3% | 17.7 |
| + GRPO | 83.6% | 62.3% | 0.2% | 16.8 |
| **+ shaped reward** | **86.4%** | **66.7%** | 0.4% | 16.6 |
| + turn-level credit | 82.0% | 58.8% | 0.0% | 17.6 |

114 tasks × 4 rollouts under a protocol frozen before the first run
([docs/eval_protocol.md](docs/eval_protocol.md)). `pass^k` is τ²'s own metric: the chance that
all k of k drawn rollouts pass. Differences are paired bootstraps over tasks, against a measured
noise floor of ±2.5 points.

## The four stages

**SFT — the action space.** 62.4% of the base model's tool results are `Tool 'X' not found`: it
calls the thirty tools that belong to the *customer's phone*, having only thirteen of its own.
784 rejection-filtered teacher trajectories over 249 tasks (698 training, 86 held out by task)
fix that — protocol errors **62.4% → 0.3%**. Published as
[`cy-330/tau2-telecom-agent-sft`](https://huggingface.co/datasets/cy-330/tau2-telecom-agent-sft),
with zero overlap against the evaluation set.

**GRPO — +7.7 points** over the SFT checkpoint (p=0.0014), `pass^4` 41.2% → 62.3%: what RL buys
here is consistency, not single-shot capability. The change that made it work was the learning
rate — veRL's full-parameter PPO default of 1e-6 moves a rank-16 LoRA's merged weights by
9.3e-07, four thousand times less than the same adapter under SFT at 1e-4.

**Shaped reward — +10.5 points**, the best arm at **86.4% pass^1**, and at equal pass^1 it also
gives 5.0% shorter successful trajectories with 61.8% fewer repeated tool calls. Two terms, made
disjoint by construction: `0.2·(1−R)·hit` separates failures from each other, `0.3·R·eff` ranks
successes by speed within their own group. The split follows from the data — every successful
trajectory hits 100% of its expected actions, so action hits carry no information among
successes.

**Turn-level credit — the project's research question, and a negative result.** See below.

The gain is not uniform: it concentrates on `mms_issue` (**+15.8 points**), which is 89.1% of
the RL training pool but only 43.0% of the benchmark.

## Why turn-level credit failed

The same trajectory reward, redistributed across turns instead of landing on the last one. The
design holds two identities — the row sum is byte-identical to the baseline's, and `β=0`
reproduces the baseline bit for bit — so it is the one genuinely single-variable comparison in
the project.

It did not improve pass^1, and it lengthened trajectories by 8.2% (p<0.0001). **Every extra turn
is a turn spent talking to the customer** (+11.3%, p<0.0001); backend tool turns do not move at
all (+1.0%, p=0.54). Dual control puts 76% of expected actions inside a user message, whose
tokens carry no gradient, so credit is attributed backwards to the assistant turn that *asked* —
a message turn. The policy generalised the wrong invariant: not "say this, here", but "produce
more customer-facing turns".

Training reward did not rise (0.620 against the baseline's 0.624), so the objective was never
exploited. The **attribution rule** put the gradient in the wrong place.

## Layout

```
rl/tau2_agent_loop.py   drives a tau2 episode as a veRL agent loop
rl/credit.py            four advantage estimators behind algorithm.adv_estimator
rl/shape.py             where a reward sits inside a trajectory (pure, unit-tested)
rl/test_credit.py       17 property tests, including "beta=0 reproduces the baseline"
scripts/rl_arm.sh       one runner, one arm per argument
scripts/rl_stats.py     recomputes the table above and every paired test from the saved runs
scripts/build_sft.py    teacher sampling, rejection, stratification, loss-mask verification
docs/results.md         the full lab record, run by run, including corrections
docs/eval_protocol.md   the frozen protocol
```

More: [docs/results.md](docs/results.md) for the complete record · the teacher trajectories on
[Hugging Face](https://huggingface.co/datasets/cy-330/tau2-telecom-agent-sft) · results, lab log
and Agentic RL theory notes (in Chinese) on the project page
[yuecao365.github.io/tau2telecom_RL](https://yuecao365.github.io/tau2telecom_RL/).

## Scope

On-policy RL with a verifiable reward, one GPU, single seed, 25 steps × 64 trajectories. No
reward model, no preference optimisation, no critic. The user simulator is frozen as DeepSeek
and is part of the environment: a published ablation on this benchmark moves the score by twenty
points when it is swapped, so any τ² telecom number is only meaningful together with it.
