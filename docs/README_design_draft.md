> **This is the design draft written before any of it ran, kept for the record.**
> It is superseded by the top-level README and by `docs/results.md`. Two things in it did not
> survive contact with the environment: the per-turn process reward built on `env_assertions`
> (measured too coarse — 45% of the pool has a single assertion, where it is identical to the
> terminal reward), and the framing of MUA-RL's numbers as the comparison point (the published
> telecom band for this benchmark is much higher and depends heavily on the user simulator).

# Turn-Level Credit Assignment for Multi-Turn Tool Agents

Post-training a customer-service agent on **τ²-bench telecom**, where an agent diagnoses a
phone's connectivity fault by reading a policy manual, calling backend tools, and *talking a
simulated user through steps it cannot perform itself*. Training compares outcome-only GRPO
against verifiable per-turn process rewards and three turn-level credit-assignment schemes.

Status: **starting on this environment.** The project previously ran a BIRD text-to-SQL setting
with a hidden schema; that setting turned out to make exploration mechanical rather than a real
decision, and it is retired on the `sql-env` tag. First step here is one pilot —
see [docs/pilot_tau2.md](docs/pilot_tau2.md).

## Why this environment

The research question is whether credit can be assigned to individual *turns* rather than to a
whole trajectory. That question only has teeth when turns do genuinely different things. In
telecom they do: read status, call a backend tool, **instruct the user to act**, confirm the
result. τ² calls this dual control — 32 of the tasks expect actions whose `requestor` is the
user, so the agent cannot simply do them itself.

telecom is also the only τ² domain whose reward is fully programmatic. `ENV_ASSERTION` is a
product of deterministic Python assertions over device state (`assert_mobile_data_status`,
`assert_internet_speed`), with no LLM judge anywhere. retail needs `NL_ASSERTION` on 112 of 114
tasks and airline needs `COMMUNICATE` on all 50, so both are evaluation domains only.

Scale: 2285 tasks with 2218 distinct expected action sequences, a median of 6 expected actions
and 17 tools. Task ids encode difficulty as `[issue]fault|fault|...[PERSONA:None|Easy|Hard]`;
the fault count peaks at 5. Note that `--num-tasks N` takes a *prefix*, and the prefix is the
one-fault tasks — always sample with a seed.

## Reference point

MUA-RL (arXiv 2508.18669) trained Qwen3 with GRPO and a binary reward in this environment, and
published every scale. The 8B row is the one that matters here:

| | τ2 Retail | τ2 Airline | τ2 Telecom |
|---|---|---|---|
| Qwen3-8B zero-shot | 41.0 | 12.5 | **19.1** |
| Qwen3-8B cold-start (SFT) | 31.4 | 16.0 | **9.0** |
| MUA-RL-8B (SFT + RL) | 49.8 | 19.0 | **21.8** |
| MUA-RL-32B | 67.3 | 45.4 | 28.3 |

Two things to take from it. Zero-shot telecom is **not** zero, so a group of rollouts will
contain both successes and failures and GRPO will have a gradient. And their SFT stage *halved*
the 8B telecom score — a published negative result with no mechanism given, which is what the
SFT-scale diagnostic in this project is aimed at.

## Design

**Terminal reward** is the official rule unchanged: `R = Π(env_assertions) ∈ {0, 1}`.

**Process reward**, computed per turn, all programmatic:

```
progress_k = fraction of env_assertions satisfied after turn k     (assertions are
Δ_k        = progress_k − progress_{k−1},  progress_0 = 0           deterministic, side-effect
r_k        = w1·Δ_k + w2·action_hit_k − w3·dup_k − w4·idle_k        free, callable mid-episode)
```

`Δ_k` is signed, so it telescopes to the final progress: no sequence of actions can farm more
total process reward than the state it ends on. `action_hit_k` is a second signal from
`evaluation_criteria.actions`, which every task carries.

**Credit assignment** — the point of the project, three schemes behind one switch:

```
CA-0  trajectory-level   A_{i,k} = A_i                                          veRL default
CA-1  discounted turn    G_{i,k} = Σ_{j≥k} γ^{j−k} r_{i,j} + γ^{T−k} R_i,       pooled normalization
CA-2  position-normalized  same G, normalized among trajectories that reached turn k
```

## Budget

Measured on the smoke run and the domain files: policy documents are ~5.2k tokens, an easy
episode's conversation is ~2.4k, and a hard one should stay under ~8k, so a **16k** context
budget holds. Qwen3-8B with LoRA co-located with vLLM comes to roughly 61 GB of the A800's 80 GB
(16.4 weights + 0.8 LoRA/optimizer + ~8 activations + 16.4 rollout copy + ~19 KV cache).

## Layout

```
sft/        data.py (chat-template encoding with a per-token loss mask) · train.py (LoRA) · merge.py
scripts/    tau2_pilot.sh · tau2_buckets.py (GO/NO-GO) · dump_mask.py · serve_vllm.sh
docs/       pilot_tau2.md (frozen protocol) · theory.html (Agentic RL notes)
tests/      pytest, edge cases only
```

τ²-bench itself lives outside the repo at `/root/autodl-tmp/tau2-bench` with its own Python 3.12
venv, so its litellm/fastapi stack never meets the training environment's torch/vLLM.

```
pytest -q

bash scripts/serve_vllm.sh models/Qwen3-8B qwen8b     # GPU box
bash scripts/tau2_pilot.sh                            # prints the GO/NO-GO table
```

## Notes

📐 [Agentic RL theory notes](https://yuecao365.github.io/probesql_rl/theory.html) — policy
gradient → GRPO/DAPO/Dr.GRPO → multi-turn masking and credit assignment → RLVR reward design →
training infrastructure.
