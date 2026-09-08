"""Result-set comparison: the terminal reward and the process-reward signal.

Two result sets are compared as multisets of rows after normalizing every cell,
so `SELECT a, b` vs `SELECT b, a`, `1` vs `'1'` vs `1.0`, and unordered rows all
count as the same answer, while duplicates (SELECT vs SELECT DISTINCT) do not.
Floats are canonicalized by rounding to 6 significant digits instead of pairwise
isclose(); that loses exactness at rounding boundaries but keeps rows hashable,
which is what lets `overlap` be plain multiset arithmetic and guarantees
`overlap == 1` exactly when the unordered contents match.

Row order is checked only in `equal(..., ordered=True)`, which the caller sets
when the gold SQL has ORDER BY. `overlap` never checks order: it is a partial
credit signal about content, and the terminal reward catches ordering mistakes.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

_INT_LITERAL = re.compile(r"-?(0|[1-9][0-9]*)")
_FLOAT_LITERAL = re.compile(r"-?(0|[1-9][0-9]*)\.[0-9]+")

# Ranks give a total order across types so rows with mixed cells can be sorted.
_RANK = {type(None): 0, int: 1, float: 1, str: 2, bytes: 3}


def _canon_cell(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        v = float(f"{v:.6g}")
        return int(v) if v.is_integer() else v
    if isinstance(v, str):
        # Only literals that round-trip exactly; '007' or '1e3' stay strings so
        # codes and identifiers are not silently turned into numbers.
        if _INT_LITERAL.fullmatch(v):
            return int(v)
        if _FLOAT_LITERAL.fullmatch(v):
            return _canon_cell(float(v))
    return v


def _sort_key(v):
    return (_RANK[type(v)], v)


def canonical(rows: Iterable[Sequence]) -> list[tuple]:
    """Normalize cells and sort them within each row, so column order is irrelevant."""
    return [tuple(sorted((_canon_cell(c) for c in row), key=_sort_key)) for row in rows]


def equal(pred: Iterable[Sequence], gold: Iterable[Sequence], *, ordered: bool = False) -> bool:
    p, g = canonical(pred), canonical(gold)
    return p == g if ordered else Counter(p) == Counter(g)


def overlap(pred: Iterable[Sequence], gold: Iterable[Sequence]) -> float:
    """Multiset Jaccard similarity of the two row sets, in [0, 1]."""
    p, g = Counter(canonical(pred)), Counter(canonical(gold))
    union = sum((p | g).values())
    if union == 0:
        return 1.0
    return sum((p & g).values()) / union


def deltas(overlaps: Sequence[float]) -> list[float]:
    """Per-turn process reward from a trajectory's overlap scores.

    Potential-based shaping with the potential being the best overlap reached so
    far, so a turn is credited only for exceeding every previous turn. The sum
    telescopes to the best overlap, and oscillating between two queries earns
    nothing after the first visit.
    """
    best, out = 0.0, []
    for o in overlaps:
        gain = max(o - best, 0.0)
        out.append(gain)
        best = max(best, o)
    return out
