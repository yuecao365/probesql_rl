# Results log

All BIRD numbers are on the dev split unless noted. "agent" = hidden-schema multi-turn
environment (`scripts/rollout.py`), "single" = full DDL, one shot (`scripts/single_turn.py`).
Questions whose gold SQL fails to execute are excluded (1 of 300 dev, 8 of 200 train samples).

## M1 · zero-shot baselines (2026-09-09)

Qwen2.5-Coder-7B-Instruct, no training. Agent runs use the lenient tool-call parser
(`--lenient-tool-parse`): the model never produced a `<tool_call>` block on its own
(strict parse 0/24), so `lenient_call_ratio = 1.0` here is the format-legality baseline
that SFT has to fix.

| run | n | pass@1 | notes |
|---|---|---|---|
| single, full schema, T=0 | 300 | **44.0** | simple 51.7 / moderate 34.0 / challenging 30.0 |
| agent, hidden schema, T=0.6 | 299 | **30.8** | simple 39.2 / moderate 20.2 / challenging 13.8 |

Hidden schema costs 13 points at zero shot. Agent diagnostics (dev300):

| metric | value |
|---|---|
| status | submitted 248 · parse_error 23 · max_turns 25 · truncated 3 |
| avg turns / completion tokens | 5.7 / 851 |
| probe-turn ratio | 61% |
| first-SQL accuracy / error-recovery rate | 28.4% / 6.7% |
| hallucinated column or table | 31.8% of trajectories |
| duplicate call / idle streak ≥ 2 / degenerate query | 17.4% / 17.4% / 3.7% |

### pass@8 histogram (BIRD train, 192 questions × 8, T=1.0)

| bucket | share |
|---|---|
| all 8 fail | 31.8% |
| mixed | **61.5%** |
| all 8 pass | 6.8% |

pass@1 33.5%, pass^8 6.8%. Only the mixed bucket yields a non-zero GRPO advantage, so
61.5% of prompts carry gradient before any cold start. M1 GO criterion (> 20%) met;
NO-GO trigger (all-fail > 70%) far away.

### Base-model comparison: Qwen2.5-7B-Instruct vs Qwen2.5-Coder-7B-Instruct

Same three runs, same seeds and prompts. The general Instruct model emits native
`<tool_call>` blocks (strict parse 100%), so its agent rows do not need the lenient parser.

| | Coder-7B | Instruct-7B |
|---|---|---|
| single, full schema (dev300) | **44.0** | 40.7 |
| agent, hidden schema (dev300) | 30.8 | **35.5** |
| hidden-schema cost | −13.2 | −5.2 |
| strict tool-call format | 0% | **100%** |
| error-recovery rate | 6.7% | **18.2%** |
| hallucinated column/table | 31.8% | 31.1% |
| avg turns / completion tokens | 5.7 / 851 | 6.6 / 304 |
| parse_error / max_turns / truncated (of 299) | 23 / 25 / 3 | 6 / 31 / 0 |
| train pass@8: all-fail / mixed / all-pass | 31.8 / **61.5** / 6.8 | 36.5 / 47.9 / 15.6 |
| train pass@1 | 33.5 | **37.9** |

Coder has the stronger SQL prior in the one-shot setting; Instruct is the stronger
*agent*: it follows the tool protocol without help, recovers from errors 2.7× more often,
and uses a third of the tokens. Both keep the mixed bucket well above the 20% GO line.
