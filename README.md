# ProbeSQL-RL

Post-training a **schema-probing Text-to-SQL agent** with multi-turn RL. The policy is
told only the database name and its table names; it has to discover columns, value
formats and join keys through tools before it can write SQL. Training compares
outcome-only GRPO against verifiable per-turn process rewards and several turn-level
credit-assignment schemes, on BIRD with Spider held out for out-of-distribution evaluation.

Status: **M1 done, M2 (teacher sampling + SFT cold start) in progress**. Zero-shot baselines are in [docs/results.md](docs/results.md).

## Environment

| Tool | Returns |
|---|---|
| `list_tables()` | table names |
| `describe_table(table)` | columns, types, primary key, foreign keys |
| `sample_rows(table, n<=5)` | real rows, so value formats are learned from data |
| `search_column(keyword)` | `table.column type` for every column whose name contains the keyword |
| `run_sql(query)` | first 20 rows of a read-only SELECT, or the error message |
| `submit(query)` | ends the episode |

Every query runs on a read-only connection behind an allow-list authorizer, a 5 s
progress-handler timeout and row/cell truncation. A verifier flags degenerate queries
(`WHERE 1=0`, `LIMIT 0`, constant SELECTs), formatting-only duplicates and idle streaks;
those flags are both safety limits and process-reward inputs.

Messages use the OpenAI chat shape with Qwen-style `<tool_call>` parsing, so the API
teacher, the chat-template renderer and the RL framework consume one protocol
(`env/prompt.py`). Teacher sampling and RL rollouts share `env/rollout.py`.

## Locked decisions

These are fixed at the end of M0 and will not change mid-project, so numbers stay comparable.

- **Correctness = BIRD's official rule, verbatim**: `set(pred_rows) == set(gold_rows)` on raw
  values, column order sensitive, row order and duplicates ignored, no normalization, no row
  cap, 30 s budget. `scripts/check_consistency.py` checks this against the official judge on
  real model predictions (862/862 agree on BIRD dev after removing an earlier float rounding).
- **Process signal**: `overlap = |P ∩ G| / |P ∪ G|` over the same row sets;
  `Δ_k = overlap_k − overlap_{k−1}` (signed, `overlap_0 = 0`), computed on `run_sql` / `submit`
  calls and aggregated per turn. Gold results are used in training only, never at evaluation.
- **One turn = one reply**, up to 4 tool calls executed in order; probes get 5 s, `submit` 30 s.
- **Hidden schema**: BIRD's `database_description` files are never shown to the policy.
- **Turn budget** 10; evaluation is 4 rollouts per question, empirical pass@1.

## Layout

```
env/        db.py sandbox · compare.py judge + overlap · verifier.py · tools.py · prompt.py · rollout.py · tasks.py · policy.py
eval/       bird_official.py (verbatim leaderboard judge) · consistency.py · metrics.py
sft/        data.py (rejection filter + chat-template encoding with loss mask) · train.py (LoRA) · merge.py
scripts/    serve_vllm.sh · rollout.py · single_turn.py · metrics.py · m1.sh · build_sft.py · dump_mask.py · audit_sample.py · check_consistency.py
docs/       results.md (every number reported so far) · theory notes
tests/      pytest, edge cases only
data/ models/ ckpt/   symlinks to the data disk (git-ignored)
docs/       notes, incl. the Agentic RL theory write-up
```

```
pytest -q
python scripts/check_consistency.py --json data/bird/dev_20240627/dev.json \
    --db-dir data/bird/dev_20240627/dev_databases --gold 20

bash scripts/serve_vllm.sh models/Qwen2.5-Coder-7B-Instruct qwen7b       # on the GPU box
python scripts/rollout.py --json data/bird/dev_20240627/dev.json --db-dir data/bird/dev_20240627/dev_databases \
    --n 300 --g 1 --base-url http://localhost:8000/v1 --model qwen7b --out outputs/dev300_agent.jsonl
python scripts/metrics.py outputs/dev300_agent.jsonl
```

Any OpenAI-compatible endpoint works as the policy, so the same rollout code
drives the local student and the API teacher. SFT data is the teacher's accepted
trajectories rendered through the student's own chat template:

```
python scripts/build_sft.py outputs/teacher_train.jsonl --per-question 2 --out data/sft/teacher.jsonl
python scripts/dump_mask.py data/sft/teacher.jsonl --model models/Qwen2.5-7B-Instruct --index 0   # eyeball the loss mask
python sft/train.py --model models/Qwen2.5-7B-Instruct --data data/sft/teacher.jsonl --out ckpt/sft_r32
python sft/merge.py --base models/Qwen2.5-7B-Instruct --adapter ckpt/sft_r32 --out models/qwen7b-sft
```

## Notes

📐 [Agentic RL theory notes](https://yuecao365.github.io/probesql_rl/theory.html) — policy
gradient → GRPO/DAPO/Dr.GRPO → multi-turn masking and credit assignment → RLVR reward design →
training infrastructure.
