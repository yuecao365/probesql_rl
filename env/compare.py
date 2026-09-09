"""Result-set comparison: the terminal reward and the process-reward signal.

Equality is exactly BIRD's official execution-accuracy rule: the set of raw
row tuples, so row order and duplicates are ignored, column order matters, and
values are compared as SQLite returns them (Python already treats 1 and 1.0 as
equal; '1' and 1 stay different). No normalization is applied on purpose: an
earlier version rounded floats and coerced numeric strings, and on real model
predictions that accepted answers the official judge rejects. The training
reward must never be looser than the metric it is trained towards.

`overlap` is plain set arithmetic on the same rows, so it equals 1 exactly
when `equal` holds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def rows_set(rows: Iterable[Sequence]) -> set[tuple]:
    return {tuple(r) for r in rows}


def equal(pred: Iterable[Sequence], gold: Iterable[Sequence]) -> bool:
    return rows_set(pred) == rows_set(gold)


def overlap(pred: Iterable[Sequence], gold: Iterable[Sequence]) -> float:
    """Jaccard similarity of the two row sets, in [0, 1]; 1 for two empty sets."""
    p, g = rows_set(pred), rows_set(gold)
    union = len(p | g)
    return 1.0 if union == 0 else len(p & g) / union


def deltas(overlaps: Sequence[float]) -> list[float]:
    """Δ_k = overlap_k − overlap_{k−1}, with overlap_0 = 0.

    Signed on purpose: a turn that regresses is penalized, which is exactly the
    per-turn distinction turn-level credit assignment needs. The sum telescopes
    to the last overlap, so no sequence of queries can farm more total credit
    than the result it ends on.
    """
    prev, out = 0.0, []
    for o in overlaps:
        out.append(o - prev)
        prev = o
    return out
