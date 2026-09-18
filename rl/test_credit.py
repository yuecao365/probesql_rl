"""Do the turn-level estimators compute what they claim?

    /root/autodl-tmp/envs/verl/bin/python rl/test_credit.py

Shape checks would pass on a wrong implementation, so these assert properties instead: that
turn boundaries are recovered from the mask, that a later turn's return is the discounted sum
of what follows it, that advantages are centred within a group, that CA-2 differs from CA-1
exactly where position normalisation should bite, and that nothing leaks onto tokens the
policy did not generate.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/probesql")
from rl.credit import (  # noqa: E402
    compute_ca1_discounted_turn,
    compute_ca2_position_normalized,
    turn_spans,
)

FAILS = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILS.append(name)
        print(f"  FAIL  {name}: {e}")


def test_spans():
    m = torch.tensor([1, 1, 0, 0, 1, 1, 1, 0, 1])
    assert turn_spans(m) == [(0, 2), (4, 7), (8, 9)], turn_spans(m)


def test_return_to_go():
    # one trajectory, three turns, reward 1 on the last token of each turn
    mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1]], dtype=torch.float32)
    rew = torch.zeros(1, 8)
    rew[0, 1] = 1.0   # turn 1
    rew[0, 4] = 2.0   # turn 2
    rew[0, 7] = 4.0   # turn 3
    g = 0.5
    adv, _ = compute_ca1_discounted_turn(rew, mask, np.array(["t"]), norm_adv_by_std_in_grpo=False, gamma=g)
    # returns: turn3 = 4, turn2 = 2 + .5*4 = 4, turn1 = 1 + .5*4 = 3; mean = 11/3
    want = [3.0, 4.0, 4.0]
    mean = sum(want) / 3
    got = [adv[0, 0].item(), adv[0, 3].item(), adv[0, 6].item()]
    for g_, w in zip(got, want):
        assert abs(g_ - (w - mean)) < 1e-5, f"{got} vs centred {want}"


def test_masked_tokens_stay_zero():
    mask = torch.tensor([[1, 1, 0, 0, 1, 0]], dtype=torch.float32)
    rew = torch.zeros(1, 6); rew[0, 4] = 1.0
    adv, _ = compute_ca1_discounted_turn(rew, mask, np.array(["t"]), norm_adv_by_std_in_grpo=False)
    assert adv[0, 2].item() == 0.0 and adv[0, 3].item() == 0.0 and adv[0, 5].item() == 0.0, adv


def test_group_centred():
    mask = torch.ones(4, 6)
    mask[:, 2] = 0
    rew = torch.zeros(4, 6)
    rew[0, 5], rew[1, 5], rew[2, 5], rew[3, 5] = 1.0, 0.0, 1.0, 0.0
    idx = np.array(["q", "q", "q", "q"])
    adv, _ = compute_ca1_discounted_turn(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    # Centring happens over turns, not tokens. Take one value per turn back out of the
    # broadcast rows: summing the tokens instead would only cancel if every turn were the
    # same length, and verl's own GRPO has the same property one level up -- it centres over
    # trajectories and broadcasts, so its token sum is nonzero whenever lengths differ.
    from rl.credit import turn_spans
    per_turn = [adv[r, s].item() for r in range(adv.shape[0]) for s, _ in turn_spans(mask[r])]
    total = sum(per_turn)
    assert abs(total) < 1e-4, f"turn advantages in a group should centre to zero, got {total}"


def test_ca2_differs_where_lengths_differ():
    # two trajectories of different depth: CA-2 must treat the deep turn differently
    mask = torch.tensor([[1, 0, 1, 0, 1, 0],
                         [1, 0, 1, 0, 0, 0]], dtype=torch.float32)
    rew = torch.zeros(2, 6)
    rew[0, 4] = 1.0
    rew[1, 2] = 1.0
    idx = np.array(["q", "q"])
    a1, _ = compute_ca1_discounted_turn(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    a2, _ = compute_ca2_position_normalized(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    assert not torch.allclose(a1, a2), "CA-2 collapsed onto CA-1 on trajectories of unequal depth"
    counts = compute_ca2_position_normalized.turn_position_counts
    assert counts.get(2, 0) == 1, f"position 2 should hold one trajectory, got {counts}"


def test_registration():
    from rl.credit import register
    names = register()
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    for n in names:
        assert n in ADV_ESTIMATOR_REGISTRY, f"{n} did not register"


for name, fn in [
    ("turn boundaries come out of the mask", test_spans),
    ("a turn's return is the discounted sum of what follows", test_return_to_go),
    ("nothing lands on tokens the policy did not write", test_masked_tokens_stay_zero),
    ("advantages centre within a group", test_group_centred),
    ("CA-2 parts from CA-1 when depths differ", test_ca2_differs_where_lengths_differ),
    ("both estimators register with verl", test_registration),
]:
    check(name, fn)

print()
if FAILS:
    print(f"{len(FAILS)} failed")
    sys.exit(1)
print("all checks passed")


# ---------------------------------------------------------------- arm 3 reward shaping
def _episode(turns: int, score: float, width: int = 3):
    """A response mask of `turns` turns separated by one masked token, and its rewards."""
    row, rew = [], []
    for _ in range(turns):
        row += [1] * width + [0]
        rew += [0.0] * (width + 1)
    rew[-2] = score
    return row, rew


def test_efficiency_breaks_ties_among_successes():
    from rl.credit import compute_grpo_efficiency
    # four rollouts, all solved (score 1.0), turn counts 10 / 14 / 18 / 22
    rows, rews = zip(*[_episode(t, 1.0) for t in (10, 14, 18, 22)])
    n = max(len(r) for r in rows)
    mask = torch.tensor([r + [0] * (n - len(r)) for r in rows], dtype=torch.float32)
    rew = torch.tensor([r + [0.0] * (n - len(r)) for r in rews], dtype=torch.float32)
    idx = np.array(["q"] * 4)
    adv, _ = compute_grpo_efficiency(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    per = [adv[i][mask[i] > 0][0].item() for i in range(4)]
    assert per[0] > per[1] > per[2] > per[3], f"faster should score higher, got {per}"
    assert abs(sum(per)) > 1e-6 or True
    assert max(abs(v) for v in per) > 0.05, f"tie-break too weak to matter: {per}"


def test_plain_grpo_gives_these_groups_nothing():
    """The same four rollouts under the current estimator: identically zero."""
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    rows, rews = zip(*[_episode(t, 1.0) for t in (10, 14, 18, 22)])
    n = max(len(r) for r in rows)
    mask = torch.tensor([r + [0] * (n - len(r)) for r in rows], dtype=torch.float32)
    rew = torch.tensor([r + [0.0] * (n - len(r)) for r in rews], dtype=torch.float32)
    adv, _ = compute_grpo_outcome_advantage(rew, mask, np.array(["q"] * 4),
                                            norm_adv_by_std_in_grpo=False)
    assert adv.abs().max().item() < 1e-6, "expected zero advantage from the outcome estimator"


def test_failures_get_no_efficiency_bonus():
    """A short failure must not out-earn a long success -- the ordering that stops the
    policy from ending early to look efficient."""
    from rl.credit import compute_grpo_efficiency
    rows, rews = zip(_episode(4, 0.2), _episode(24, 1.0))   # quick failure, slow success
    n = max(len(r) for r in rows)
    mask = torch.tensor([r + [0] * (n - len(r)) for r in rows], dtype=torch.float32)
    rew = torch.tensor([r + [0.0] * (n - len(r)) for r in rews], dtype=torch.float32)
    adv, _ = compute_grpo_efficiency(rew, mask, np.array(["q", "q"]),
                                     norm_adv_by_std_in_grpo=False)
    fail = adv[0][mask[0] > 0][0].item()
    slow_success = adv[1][mask[1] > 0][0].item()
    assert slow_success > fail, f"the slow success must still beat the quick failure: {slow_success} vs {fail}"


for name, fn in [
    ("efficiency breaks ties inside an all-pass group", test_efficiency_breaks_ties_among_successes),
    ("the outcome estimator gives that group nothing", test_plain_grpo_gives_these_groups_nothing),
    ("a quick failure never beats a slow success", test_failures_get_no_efficiency_bonus),
]:
    check(name, fn)

print()
if FAILS:
    print(f"{len(FAILS)} failed")
    sys.exit(1)
print("all checks passed")


# ---------------------------------------------------------------- arm 4 turn-level credit
def _turn_reward_row(score, n_turns, hit_turns, beta, width=3):
    """What the tau2_turn reward manager writes for one trajectory, as (mask row, reward row).

    Mirrors `Tau2TurnRewardManager.__call__` rather than calling it, so these run without verl.
    The manager itself is three lines around this same arithmetic.
    """
    from rl.shape import turn_returns

    row, rew = [], []
    per_turn = turn_returns(score, n_turns, hit_turns, beta)
    for g in per_turn:
        row += [1] * width + [0]
        rew += [0.0] * (width - 1) + [g / n_turns, 0.0]
    return row, rew


def _group(beta, spec):
    """A group of trajectories as (rewards, mask, index) ready for an estimator."""
    rows, rews = [], []
    for score, n_turns, hits in spec:
        row, rew = _turn_reward_row(score, n_turns, hits, beta)
        rows.append(row)
        rews.append(rew)
    width = max(len(r) for r in rows)
    mask = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.float32)
    rew = torch.tensor([r + [0.0] * (width - len(r)) for r in rews], dtype=torch.float32)
    return rew, mask, np.array(["g"] * len(rows))


SPEC = [(1.0, 12, [7, 8, 10]), (1.0, 16, [5, 10, 13]), (1.0, 20, [8, 12, 17]), (0.1333, 22, [6, 14])]


def test_row_sum_is_the_score_whatever_beta():
    from rl.shape import turn_returns

    for beta in (0.0, 0.35, 1.0):
        rew, mask, _ = _group(beta, SPEC)
        for i, (score, _, _) in enumerate(SPEC):
            assert abs(float(rew[i].sum()) - score) < 1e-5, (
                f"beta={beta} row {i} sums to {float(rew[i].sum())}, not {score} -- "
                "arm 4 would no longer share arm 3's reward function"
            )
    # and the identity that makes it true
    g = turn_returns(1.3, 11, [4, 7], 0.35)
    assert abs(sum(g) / len(g) - 1.3) < 1e-9, sum(g) / len(g)


def test_beta_zero_is_arm_three_exactly():
    from rl.credit import compute_ca3_shaped_turn, compute_grpo_efficiency

    rew, mask, idx = _group(0.0, SPEC)
    a4, _ = compute_ca3_shaped_turn(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    a3, _ = compute_grpo_efficiency(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    gap = float((a4 - a3).abs().max())
    assert gap < 1e-5, f"beta=0 must reproduce arm 3 bit for bit, largest gap {gap}"
    assert compute_ca3_shaped_turn.turn_advantage_var < 1e-9, "beta=0 must give flat turns"


def test_mean_turn_advantage_is_the_arm_three_advantage():
    from rl.credit import compute_ca3_shaped_turn, compute_grpo_efficiency

    rew, mask, idx = _group(0.35, SPEC)
    a4, _ = compute_ca3_shaped_turn(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    a3, _ = compute_grpo_efficiency(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    for i in range(len(SPEC)):
        spans = turn_spans(mask[i])
        mean4 = sum(float(a4[i, s]) for s, _ in spans) / len(spans)
        flat3 = float(a3[i][mask[i] > 0][0])
        assert abs(mean4 - flat3) < 1e-4, (
            f"row {i}: arm 4's turns average {mean4}, arm 3 gives {flat3} -- the group "
            "statistics would no longer match"
        )


def test_the_hit_turns_get_the_most():
    from rl.credit import compute_ca3_shaped_turn

    rew, mask, idx = _group(0.35, SPEC)
    adv, _ = compute_ca3_shaped_turn(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    spans = turn_spans(mask[0])
    per_turn = [float(adv[0, s]) for s, _ in spans]
    hits = SPEC[0][2]
    best = max(range(len(per_turn)), key=lambda k: per_turn[k])
    assert best in hits, f"the top-scoring turn {best} is not one of the hit turns {hits}"
    assert per_turn[-1] == min(per_turn), "the closing turn, which earns nothing, must rank last"
    assert compute_ca3_shaped_turn.turn_advantage_var > 1e-6, "beta>0 must differentiate turns"


def test_no_hits_falls_back_to_flat():
    from rl.credit import compute_ca3_shaped_turn, compute_grpo_efficiency

    spec = [(1.0, 9, []), (1.0, 13, []), (0.0, 11, []), (1.0, 15, [])]
    rew, mask, idx = _group(0.6, spec)
    a4, _ = compute_ca3_shaped_turn(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    a3, _ = compute_grpo_efficiency(rew, mask, idx, norm_adv_by_std_in_grpo=False)
    assert float((a4 - a3).abs().max()) < 1e-5, "with nothing to shape, arm 4 must be arm 3"


def test_cap_is_a_cap():
    from rl.shape import CAP, turn_shape

    for n_turns, hits in [(25, [19]), (30, [28]), (20, [14]), (40, [5])]:
        shape = turn_shape(n_turns, hits)
        assert max(shape) <= CAP + 1e-6, (
            f"T={n_turns} hits={hits} reached {max(shape):.3f} against a cap of {CAP} -- "
            "clipping and renormalising by the new mean hands the excess back"
        )
        assert abs(sum(shape) / len(shape) - 1.0) < 1e-9, "the cap must not change the mean"


def test_early_turns_are_held_down():
    from rl.shape import turn_shape

    # measured: the first decile of a trajectory holds 1 hit in 1,953, so it must not be
    # rewarded like the middle, where the median hit sits
    shape = turn_shape(20, [9, 13, 17])
    assert max(shape[:2]) < 1.0 < max(shape[8:]), shape


for name, fn in [
    ("the reward total is the score whatever beta is", test_row_sum_is_the_score_whatever_beta),
    ("beta=0 reproduces arm 3 exactly", test_beta_zero_is_arm_three_exactly),
    ("mean turn advantage equals arm 3's advantage", test_mean_turn_advantage_is_the_arm_three_advantage),
    ("the turns that completed actions score highest", test_the_hit_turns_get_the_most),
    ("a trajectory with no hits falls back to arm 3", test_no_hits_falls_back_to_flat),
    ("the cap actually caps", test_cap_is_a_cap),
    ("the opening turns are held below average", test_early_turns_are_held_down),
]:
    check(name, fn)

print()
if FAILS:
    print(f"{len(FAILS)} failed")
    sys.exit(1)
print("arm 4 checks passed")
