"""Result-set comparison: the terminal reward and the process-reward signal.

Equality deliberately mirrors BIRD's official execution accuracy: a set of row
tuples, so row order and duplicates are ignored but column order matters. On top
of that, cells are normalized so SQLite's weak typing and float noise cannot
flip the verdict (`1` / `'1'` / `1.0` agree, floats round to 6 significant
digits). Staying at least as strict as the official metric is what keeps the
training reward from teaching answers the evaluation would reject.

Floats are rounded rather than compared with isclose() so rows stay hashable:
`overlap` is then plain set arithmetic and equals 1 exactly when `equal` holds.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_INT_LITERAL = re.compile(r"-?(0|[1-9][0-9]*)")
_FLOAT_LITERAL = re.compile(r"-?(0|[1-9][0-9]*)\.[0-9]+")


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


def canonical(rows: Iterable[Sequence]) -> set[tuple]:
    return {tuple(_canon_cell(c) for c in row) for row in rows}


def equal(pred: Iterable[Sequence], gold: Iterable[Sequence]) -> bool:
    return canonical(pred) == canonical(gold)


def overlap(pred: Iterable[Sequence], gold: Iterable[Sequence]) -> float:
    """Jaccard similarity of the two row sets, in [0, 1]; 1 for two empty sets."""
    p, g = canonical(pred), canonical(gold)
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
