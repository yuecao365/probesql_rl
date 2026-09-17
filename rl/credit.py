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

from collections import defaultdict

import numpy as np
import torch


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
    return ["ca1_discounted_turn", "ca2_position_normalized"]
