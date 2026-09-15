# Pilot: is τ²-bench trainable at 7B?

Written **before** any result was seen (2026-09-15). The criteria below are frozen; if the
run comes back ambiguous the answer is NO-GO, not a new threshold.

## Why this pilot exists

The project is considering moving off the hidden-schema BIRD environment (see
`docs/results.md` and the scenario diagnosis) onto τ²-bench. Everything else about that
move is favourable — official Gymnasium RL interface, a few hundred MB on disk, user
simulator can run on an API so the GPU stays free, and genuinely heterogeneous turn roles.
One thing is unverified: **whether Qwen2.5-7B-Instruct produces a usable gradient signal
there.** MUA-RL reports 28.3 on τ² telecom with a *32B* model, so a 7B could plausibly
score near zero, and a near-zero score means an empty mixed bucket and no GRPO gradient.

This is the same check M1 already ran on BIRD (`0 < pass@G < 1` share, measured 47.9%).

## Setting

| | value | why |
|---|---|---|
| domain | **telecom** | 2285 tasks with 2218 distinct expected action sequences, and reward is `ENV_ASSERTION` only — a product of Python assertions on environment state, no LLM judge. retail (112/114) and airline (50/50) need NL/COMMUNICATE judging, so they are evaluation domains, not training domains. |
| `solo_mode` | **False** | 32 telecom tasks expect actions with `requestor: user`; in solo mode those are unreachable. False is also the setting MUA-RL's number refers to. |
| agent | Qwen2.5-7B-Instruct via local vLLM, litellm `openai/...` + `api_base` | the student |
| user simulator | DeepSeek via API | keeps the GPU for vLLM only; ~0.01 CNY/rollout at M2 rates |
| tasks | 60, sampled with seed 0 from `data/tau2/domains/telecom/tasks.json` | |
| rollouts | G = 8, temperature 1.0 | same as the M1 pass@8 histogram |
| agent temperature | 1.0 | we are measuring spread, not peak accuracy |
| user temperature | 0.0 | the user simulator is environment, not policy; keep it deterministic |

## Decision criteria (frozen)

Primary, identical to M1's GO gate:

- **GO** — `mixed` bucket (`0 < correct < 8`) **> 20%** of tasks.
- **NO-GO** — `all_fail` bucket **> 70%** of tasks.
- Anything between the two is a **weak GO**: proceed only if the format-legality number
  below is healthy, otherwise treat as NO-GO.

Secondary, recorded but not gating:

| metric | why it matters |
|---|---|
| format legality (valid tool call per agent reply) | M1 found Qwen2.5-Coder emitted 0/24 parseable `<tool_call>` blocks zero-shot; if the 7B cannot speak τ²'s action protocol, that is an SFT problem, not a NO-GO |
| pass@1, pass^8 | comparability with MUA-RL's 28.3 (32B) |
| turns and tool calls per episode | sequence-length budget for RL |
| termination status mix | τ² truncates at `max_steps`; a high truncation rate means the turn budget, not ability, is binding |
| user-simulator token cost | budget check before scaling to full training |

## What each outcome means

- **GO** → migrate. Archive the SQL environment on a branch (do not delete), then redo
  M1/M2 on telecom: zero-shot baselines, teacher rejection sampling, LoRA SFT.
- **NO-GO** → stay on BIRD and instead drop the hidden-schema constraint: show the full
  schema, keep the tools for value grounding (`sample_rows`) and the execute-inspect-revise
  loop. That version answers the "unnatural setting" objection without a migration, and
  M1 already has its baseline (full-schema single-turn 40.7).

Either way the SQL code is kept.

## Cost

60 tasks x 8 rollouts = 480 episodes. Agent calls run locally; the user simulator is the
only paid component, estimated at 10-20 CNY based on M2's 0.0125 CNY/rollout. GPU time is
one vLLM session, roughly 1-2 hours at 5.59 CNY/hour.
