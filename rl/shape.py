"""Where inside a trajectory a reward was earned, as a per-turn weight of mean one.

Arm 4 changes *where* a trajectory's reward sits, not how much it is. That separation is what
makes it comparable against arm 3 without re-running a control: the reward function is
untouched, and the only new object is `shape`, a vector over turns constrained to

    mean(shape) == 1

so that redistributing by it cannot change any trajectory's total. Everything here is pure
arithmetic over turn indices; nothing imports verl or tau2, so it can be tested on its own.

The shape is built from *when the expected actions completed*, which is an observation, not a
reward: a successful episode earns the same 1.0 whatever turn it did the work on. Reading
positions rather than paying for them is why arm 4 still has a signal on the 86% of rollouts
that succeed, where arm 3's `(1 - R)` partial-credit term is identically zero.

Three constants, each set from the 430 replayed episodes in `scripts/turn_hit_probe.py`:

LAM=0.3   A turn that set up the next one should share its credit, but not much: hits sit at
          relative position 0.25 at p10 and the first decile of every trajectory holds 1 hit
          out of 1,953. Discounting a terminal reward backwards is the wrong direction here,
          and 0.3 decays to 9% two turns back and 2.7% three turns back.
FLOOR=0.5  Without it, every turn after the last hit gets weight zero and drops out of the
          gradient entirely. Closing the conversation is part of the task.
CAP=3.0   Single-action tasks are 14% of episodes and 100% of them put every hit in one turn,
          so a 25-turn trajectory would hand that turn twelve times the average share.
"""

from __future__ import annotations

LAM = 0.3
FLOOR = 0.5
CAP = 3.0


def water_fill(x: list[float], cap: float = CAP) -> list[float]:
    """Cap each entry at `cap` while keeping the mean, by moving the excess to the others.

    The obvious alternative -- clip, then divide by the new mean -- does not work, and the
    reason is worth stating because it looks correct. Clipping lowers the mean, so dividing by
    it scales *every* entry back up, including the one just clipped: a cap of 3.0 lets 12.0
    through as 4.9. Handing the excess to the entries still under the cap keeps the mean at 1
    without giving any of it back to the entry that overflowed.
    """
    x = list(x)
    for _ in range(50):
        excess = sum(v - cap for v in x if v > cap)
        if excess <= 1e-12:
            break
        x = [min(cap, v) for v in x]
        free = [i for i, v in enumerate(x) if v < cap - 1e-12]
        if not free:
            return [1.0] * len(x)  # every turn at the cap means the cap is not binding
        share = excess / len(free)
        for i in free:
            x[i] += share
    return x


def turn_shape(n_turns: int, hit_turns: list[int]) -> list[float]:
    """Per-turn weights of mean one, peaked on the turns that completed expected actions.

    `hit_turns` holds one entry per expected action, giving the zero-based assistant turn that
    first completed it; repeats are allowed when one turn completed several. An empty list --
    a trajectory that reached nothing -- gives a flat shape, which makes arm 4 fall back to
    arm 3 exactly for that trajectory rather than inventing a preference.
    """
    if n_turns <= 0:
        return []
    if not hit_turns or n_turns == 1:
        return [1.0] * n_turns

    per_turn = [0.0] * n_turns
    for h in hit_turns:
        if 0 <= h < n_turns:
            per_turn[h] += 1.0 / len(hit_turns)

    # local return: a turn keeps its own hits and LAM of whatever the rest of the episode won
    local = [0.0] * n_turns
    running = 0.0
    for k in range(n_turns - 1, -1, -1):
        running = per_turn[k] + LAM * running
        local[k] = running

    mean = sum(local) / n_turns
    if mean <= 0:
        return [1.0] * n_turns
    local = [v + FLOOR * mean for v in local]
    mean = sum(local) / n_turns
    return water_fill([v / mean for v in local])


def turn_returns(score: float, n_turns: int, hit_turns: list[int], beta: float) -> list[float]:
    """The trajectory's score spread over its turns: G_k = S * ((1 - beta) + beta * shape_k).

    `mean(G) == score` for every beta, so a group's statistics are whatever they were under
    arm 3. beta=0 returns a flat vector and is the null the tests pin: arm 4 at beta=0 has to
    reproduce arm 3 bit for bit, not approximately.
    """
    if n_turns <= 0:
        return []
    if beta <= 0.0:
        return [score] * n_turns
    shape = turn_shape(n_turns, hit_turns)
    return [score * ((1.0 - beta) + beta * s) for s in shape]
