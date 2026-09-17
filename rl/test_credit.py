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
