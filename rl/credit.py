"""Turn-level credit assignment for veRL, as two advantage estimators.

Import this module before the trainer builds its advantage function and the names become
available to `algorithm.adv_estimator`:

    ca1_discounted_turn     discounted return-to-go per turn, normalised over all turns
    ca2_position_normalized same returns, normalised among trajectories that reached that turn

Why this exists. veRL's GRPO gives one scalar advantage to a whole trajectory and broadcasts
it to every action token. A telecom episode here runs about eighteen assistant turns, of
which perhaps three decide the outcome -- diagnose the fault, call the right backend API,
tell the customer which switch to flip. Broadcasting reinforces the other fifteen just as
hard. These two estimators give each turn its own advantage instead, and differ only in what
they normalise against, which is the one variable arm 4a and arm 4b separate.

Turn boundaries come from `response_mask`: a run of ones is tokens the policy generated, a
run of zeros is a tool result or the customer speaking. Runs of ones are therefore turns, and
no extra bookkeeping has to survive the trip from the agent loop to the trainer.

CA-2's warning. Turn fifteen exists only in long trajectories, and long trajectories are the
harder tasks, so pooling turn 2 of an easy task with turn 15 of a hard one compares different
things. CA-2 normalises within a turn position to remove that, and pays for it in sample
size: the deepest positions have very few trajectories. `turn_position_counts` is returned in
the metrics so a curve is never read without knowing how many trajectories stand behind its
tail.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__file__)


def _norm_by_std(config, fallback: bool) -> bool:
    """Whether to divide the advantage by the group's standard deviation.

    verl passes `norm_adv_by_std_in_grpo` to its own GRPO branch but not to a registered
    estimator: `compute_advantage` builds `adv_kwargs` from the batch alone, so a custom
    estimator silently takes its own default. Ours defaulted to True, which is why arm 3 ran
    as standard GRPO although `algorithm.norm_adv_by_std_in_grpo=False` was on the command
    line -- visible only as a max advantage of 2.44 where the un-normalised bound is 1.3.
    Reading it off `config` makes the flag mean what it says.
    """
    if config is None:
        return fallback
    value = getattr(config, "norm_adv_by_std_in_grpo", None)
    if value is None and hasattr(config, "get"):
        value = config.get("norm_adv_by_std_in_grpo", None)
    return fallback if value is None else bool(value)


def turn_spans(mask_row: torch.Tensor) -> list[tuple[int, int]]:
    """Contiguous runs of ones in a response mask, as [start, end) pairs."""
    spans, start = [], None
    for i, v in enumerate(mask_row.tolist()):
        if v > 0 and start is None:
            start = i
        elif v <= 0 and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask_row)))
    return spans


def _returns_per_turn(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
) -> tuple[list[list[tuple[int, int]]], list[list[float]]]:
    """Discounted return-to-go for every turn of every trajectory.

    A turn's reward is whatever reward landed on its tokens -- the terminal reward sits on
    the last token, and a per-turn process reward sits on that turn's tokens -- so summing
    over the span is agnostic to which of the two the run is using.
    """
    all_spans, all_returns = [], []
    for row in range(response_mask.shape[0]):
        spans = turn_spans(response_mask[row])
        per_turn = [float(token_level_rewards[row, s:e].sum()) for s, e in spans]
        running, returns = 0.0, []
        for r in reversed(per_turn):
            running = r + gamma * running
            returns.append(running)
        returns.reverse()
        all_spans.append(spans)
        all_returns.append(returns)
    return all_spans, all_returns


def _scatter(advantages: torch.Tensor, spans, values) -> torch.Tensor:
    for row, (row_spans, row_vals) in enumerate(zip(spans, values)):
        for (s, e), v in zip(row_spans, row_vals):
            advantages[row, s:e] = v
    return advantages


def compute_ca1_discounted_turn(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    gamma: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CA-1: every turn carries its own discounted return, normalised across all turns.

    The baseline is the group's mean over every (trajectory, turn) pair, which keeps GRPO's
    critic-free structure: a turn is good if it beats the other turns this group produced on
    the same task.
    """
    norm_adv_by_std_in_grpo = _norm_by_std(config, norm_adv_by_std_in_grpo)
    spans, returns = _returns_per_turn(token_level_rewards, response_mask, gamma)

    pooled = defaultdict(list)
    for row, row_returns in enumerate(returns):
        pooled[index[row]].extend(row_returns)
    stats = {}
    for idx, vals in pooled.items():
        t = torch.tensor(vals, dtype=torch.float32)
        stats[idx] = (t.mean(), t.std() if t.numel() > 1 else torch.tensor(1.0))

    centred = []
    for row, row_returns in enumerate(returns):
        mean, std = stats[index[row]]
        if norm_adv_by_std_in_grpo:
            centred.append([(v - float(mean)) / (float(std) + epsilon) for v in row_returns])
        else:
            centred.append([v - float(mean) for v in row_returns])

    advantages = torch.zeros_like(token_level_rewards)
    advantages = _scatter(advantages, spans, centred) * response_mask
    return advantages, advantages


def compute_ca2_position_normalized(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    gamma: float = 0.95,
    min_count: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CA-2: the same returns, normalised within (group, turn position).

    Turn k is compared only against turn k of the trajectories that also reached it. Positions
    with fewer than `min_count` trajectories fall back to the pooled baseline, because a
    standard deviation over one sample is not a baseline.
    """
    norm_adv_by_std_in_grpo = _norm_by_std(config, norm_adv_by_std_in_grpo)
    spans, returns = _returns_per_turn(token_level_rewards, response_mask, gamma)

    by_pos = defaultdict(list)
    pooled = defaultdict(list)
    for row, row_returns in enumerate(returns):
        pooled[index[row]].extend(row_returns)
        for k, v in enumerate(row_returns):
            by_pos[(index[row], k)].append(v)

    pooled_stats = {}
    for idx, vals in pooled.items():
        t = torch.tensor(vals, dtype=torch.float32)
        pooled_stats[idx] = (float(t.mean()), float(t.std()) if t.numel() > 1 else 1.0)

    pos_stats, counts = {}, {}
    for key, vals in by_pos.items():
        counts[key[1]] = counts.get(key[1], 0) + len(vals)
        if len(vals) >= min_count:
            t = torch.tensor(vals, dtype=torch.float32)
            pos_stats[key] = (float(t.mean()), float(t.std()))

    centred = []
    for row, row_returns in enumerate(returns):
        row_out = []
        for k, v in enumerate(row_returns):
            mean, std = pos_stats.get((index[row], k), pooled_stats[index[row]])
            row_out.append((v - mean) / (std + epsilon) if norm_adv_by_std_in_grpo else v - mean)
        centred.append(row_out)

    advantages = torch.zeros_like(token_level_rewards)
    advantages = _scatter(advantages, spans, centred) * response_mask
    # Deep positions rest on very few trajectories; a reader of the curve has to see that.
    compute_ca2_position_normalized.turn_position_counts = dict(sorted(counts.items()))
    return advantages, advantages


def register() -> list[str]:
    """Make both estimators visible to veRL's `algorithm.adv_estimator`."""
    from verl.trainer.ppo.core_algos import register_adv_est

    register_adv_est("ca1_discounted_turn")(compute_ca1_discounted_turn)
    register_adv_est("ca2_position_normalized")(compute_ca2_position_normalized)
    register_adv_est("grpo_efficiency")(compute_grpo_efficiency)
    register_adv_est("ca3_shaped_turn")(compute_ca3_shaped_turn)
    return [
        "ca1_discounted_turn",
        "ca2_position_normalized",
        "grpo_efficiency",
        "ca3_shaped_turn",
    ]


def compute_grpo_efficiency(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    w_eff: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO, with a group-relative efficiency bonus paid only to trajectories that solved.

    This is the half of arm 3's reward that the agent loop cannot compute, because it depends
    on the other rollouts of the same task. The loop supplies R plus failure-side partial
    credit; this adds `w_eff * eff` on top, where `eff` is 1 for the fastest success in the
    group and 0 for the slowest.

    Why group-relative rather than an absolute target. An absolute form, min(1, 15/T), maps
    the observed 14-22 turn range onto 0.68-1.0, and after weighting it produced a mean
    absolute advantage of 0.012 inside all-pass groups against 0.25-0.50 in mixed ones -- it
    would have been drowned. Normalising inside the group restores the full range whatever the
    task's natural length, and at w_eff=0.3 gives 0.105 against 0.480, about a quarter as
    strong, which is the intended ordering: solving still dominates, speed breaks the tie.

    Why it is worth having at all: on arm 2b, 75 of 114 tasks had all four rollouts succeed.
    Their advantages are identically zero and those rollouts teach nothing, while the turn
    counts inside them differ by a median of six. The signal was there and unused.

    R is recovered as `score >= 1.0`: the agent loop scores a solved episode exactly 1.0 and a
    failed one at most w_hit, which is well below 1.
    """
    norm_adv_by_std_in_grpo = _norm_by_std(config, norm_adv_by_std_in_grpo)
    scores = token_level_rewards.sum(dim=-1)
    solved = (scores >= 1.0 - 1e-6).float()

    turns = torch.tensor(
        [max(1, len(turn_spans(response_mask[i]))) for i in range(response_mask.shape[0])],
        dtype=torch.float32, device=scores.device,
    )

    by_group = defaultdict(list)
    for i in range(scores.shape[0]):
        if solved[i] > 0:
            by_group[index[i]].append(float(turns[i]))
    bounds = {g: (min(v), max(v)) for g, v in by_group.items()}

    eff = torch.zeros_like(scores)
    for i in range(scores.shape[0]):
        if solved[i] > 0 and index[i] in bounds:
            lo, hi = bounds[index[i]]
            eff[i] = 1.0 if hi == lo else (hi - float(turns[i])) / (hi - lo)
    shaped = scores + w_eff * solved * eff

    id2 = defaultdict(list)
    for i in range(shaped.shape[0]):
        id2[index[i]].append(shaped[i])
    stats = {}
    for g, vals in id2.items():
        t = torch.stack(vals)
        stats[g] = (t.mean(), t.std() if t.numel() > 1 else torch.tensor(1.0, device=t.device))

    out = shaped.clone()
    for i in range(out.shape[0]):
        mean, std = stats[index[i]]
        out[i] = (out[i] - mean) / (std + epsilon) if norm_adv_by_std_in_grpo else out[i] - mean
    advantages = out.unsqueeze(-1) * response_mask
    return advantages, advantages



def compute_ca3_shaped_turn(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    w_eff: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arm 4: arm 3's advantage, redistributed across the turns of each trajectory.

    The agent loop has already placed `G_k / T` on the last token of turn k, where
    `G_k = S_i * ((1 - beta) + beta * shape_k)` and `mean_k G_k == S_i`. So this reads the
    same `S_i` off the row sum that `compute_grpo_efficiency` does, computes the same
    group-relative efficiency bonus against the same baseline, and differs from it in one
    respect: a turn is credited with `G_k` rather than with the trajectory's `S_i`.

    Two identities hold for every beta and are what make arm 4 comparable against the arm 3
    run that already exists, with no matched control:

        sum_over_tokens(rewards)  == S_i           the reward function is untouched
        mean_over_turns(advantage) == arm 3's advantage for that trajectory

    At beta=0 the first identity forces `G_k == S_i` for every k and this function returns
    `compute_grpo_efficiency`'s output exactly. `test_credit.py` pins that.
    """
    norm_adv_by_std_in_grpo = _norm_by_std(config, norm_adv_by_std_in_grpo)
    scores = token_level_rewards.sum(dim=-1)
    solved = (scores >= 1.0 - 1e-6).float()

    spans = [turn_spans(response_mask[i]) for i in range(response_mask.shape[0])]
    turns = torch.tensor(
        [max(1, len(s)) for s in spans], dtype=torch.float32, device=scores.device
    )

    by_group = defaultdict(list)
    for i in range(scores.shape[0]):
        if solved[i] > 0:
            by_group[index[i]].append(float(turns[i]))
    bounds = {g: (min(v), max(v)) for g, v in by_group.items()}

    eff = torch.zeros_like(scores)
    for i in range(scores.shape[0]):
        if solved[i] > 0 and index[i] in bounds:
            lo, hi = bounds[index[i]]
            eff[i] = 1.0 if hi == lo else (hi - float(turns[i])) / (hi - lo)
    bonus = w_eff * solved * eff
    shaped = scores + bonus

    # The baseline is over trajectories, not over turns. Pooling turns would weight a long
    # trajectory more heavily and smuggle in a length penalty that duplicates `eff`.
    id2 = defaultdict(list)
    for i in range(shaped.shape[0]):
        id2[index[i]].append(shaped[i])
    stats = {}
    for g, vals in id2.items():
        t = torch.stack(vals)
        stats[g] = (t.mean(), t.std() if t.numel() > 1 else torch.tensor(1.0, device=t.device))

    advantages = torch.zeros_like(token_level_rewards)
    turn_var = []
    for i in range(advantages.shape[0]):
        mean, std = stats[index[i]]
        row_spans = spans[i] or [(0, response_mask.shape[1])]
        n = len(row_spans)
        vals = []
        for s, e in row_spans:
            g_k = float(token_level_rewards[i, s:e].sum()) * n
            a = g_k + float(bonus[i]) - float(mean)
            if norm_adv_by_std_in_grpo:
                a = a / (float(std) + epsilon)
            advantages[i, s:e] = a
            vals.append(a)
        if len(vals) > 1:
            turn_var.append(float(torch.tensor(vals).var()))

    advantages = advantages * response_mask
    # The one diagnostic that says whether credit assignment happened at all: zero variance
    # within a trajectory means this degenerated back into arm 3.
    compute_ca3_shaped_turn.turn_advantage_var = (
        sum(turn_var) / len(turn_var) if turn_var else 0.0
    )
    # And the one that says whether the agent loop and the trainer agree about the turns.
    # `G_k` is recovered by multiplying a turn's share by the turn count, so if the loop
    # divided by a different count than `turn_spans` finds here, every advantage is scaled by
    # the ratio and nothing downstream would say so. Checking it is two lines: the recovered
    # returns must average back to the row sum.
    drift = 0.0
    for i in range(advantages.shape[0]):
        row_spans = spans[i]
        if not row_spans:
            continue
        n = len(row_spans)
        recovered = sum(float(token_level_rewards[i, s:e].sum()) * n for s, e in row_spans) / n
        drift = max(drift, abs(recovered - float(scores[i])))
    compute_ca3_shaped_turn.turn_sum_drift = drift
    compute_ca3_shaped_turn.turn_count_mean = (
        sum(len(sp) for sp in spans) / max(1, len(spans))
    )
    compute_ca3_shaped_turn.max_abs_advantage = float(advantages.abs().max())
    # print, not logger: warnings from this module never reached the run logs, which is why
    # the loop's own TAU2_EPISODE_DONE line was invisible for a whole run too.
    print(
        f"CA3 check: turns/traj={compute_ca3_shaped_turn.turn_count_mean:.2f} "
        f"turn_adv_var={compute_ca3_shaped_turn.turn_advantage_var:.4f} "
        f"max|adv|={compute_ca3_shaped_turn.max_abs_advantage:.3f} "
        f"norm_by_std={norm_adv_by_std_in_grpo} sum_drift={drift:.2e}",
        flush=True,
    )
    return advantages, advantages

# Registering at import time, not only through register(). verl builds the advantage function
# inside the TaskRunner Ray actor, a different process from the one that loads the agent loop,
# so a call made there never reaches here. `actor_rollout_ref.model.external_lib=rl.credit`
# imports this module in the process that needs it, and the import alone is enough.
try:
    register()
except Exception:  # pragma: no cover - verl may not be importable in a bare test process
    pass
