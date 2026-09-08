import math

import pytest

from env.compare import canonical, deltas, equal, overlap


# --- cell normalization -------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        (1, 1.0),
        (1, "1"),
        (1.5, "1.50"),
        (0, "-0"),
        (1.0000001, 1.0),
        (0.1 + 0.2, 0.3),
        (True, 1),
    ],
)
def test_weak_types_and_float_tolerance_unify(a, b):
    assert equal([(a,)], [(b,)])


@pytest.mark.parametrize(
    "a, b",
    [
        (1.001, 1.0),
        ("007", 7),
        ("1e3", 1000),
        (" 1", 1),
        (None, 0),
        (None, ""),
        (None, "None"),
        ("a", b"a"),
    ],
)
def test_distinct_values_stay_distinct(a, b):
    assert not equal([(a,)], [(b,)])


def test_null_equals_null():
    assert equal([(None, 1)], [(1, None)])


def test_mixed_type_rows_are_sortable():
    assert equal([(None, 1, "a", b"b", 2.5)], [(2.5, b"b", "a", 1, None)])


# --- row / column order and duplicates ----------------------------------------


def test_row_order_ignored_unless_ordered():
    pred, gold = [(1,), (2,)], [(2,), (1,)]
    assert equal(pred, gold)
    assert not equal(pred, gold, ordered=True)
    assert equal(pred, [(1,), (2,)], ordered=True)


def test_column_order_ignored():
    assert equal([("x", 1)], [(1, "x")])


def test_duplicates_matter():
    assert not equal([(1,), (1,)], [(1,)])
    assert not equal([(1,)], [(1,), (1,)])


def test_different_arity_never_equal():
    assert not equal([(1, 2)], [(1,)])
    assert overlap([(1, 2)], [(1,)]) == 0.0


def test_canonical_does_not_mutate_input():
    rows = [[2, 1]]
    canonical(rows)
    assert rows == [[2, 1]]


# --- overlap ------------------------------------------------------------------


def test_overlap_of_two_empty_sets_is_one():
    assert overlap([], []) == 1.0
    assert equal([], [])


def test_overlap_with_one_empty_set_is_zero():
    assert overlap([], [(1,)]) == 0.0
    assert overlap([(1,)], []) == 0.0


def test_overlap_is_one_iff_unordered_equal():
    assert overlap([(1,), (2,)], [(2,), (1,)]) == 1.0
    assert overlap([(1,), (1,)], [(1,)]) < 1.0


def test_overlap_partial_and_symmetric():
    pred, gold = [(1,), (2,), (3,)], [(2,), (3,), (4,)]
    assert overlap(pred, gold) == pytest.approx(2 / 4)
    assert overlap(pred, gold) == overlap(gold, pred)


def test_overlap_counts_duplicate_matches_by_multiplicity():
    assert overlap([(1,), (1,), (1,)], [(1,), (1,)]) == pytest.approx(2 / 3)


def test_overlap_ignores_order_even_when_it_would_matter():
    assert overlap([(1,), (2,)], [(2,), (1,)]) == 1.0
    assert not equal([(1,), (2,)], [(2,), (1,)], ordered=True)


# --- deltas -------------------------------------------------------------------


def test_deltas_empty():
    assert deltas([]) == []


def test_deltas_regression_and_oscillation_earn_nothing():
    assert deltas([0.5, 0.2, 0.5, 0.2, 0.5]) == [0.5, 0.0, 0.0, 0.0, 0.0]


def test_deltas_never_negative_and_sum_to_best():
    scores = [0.1, 0.4, 0.3, 0.9, 0.9, 0.7]
    d = deltas(scores)
    assert all(x >= 0 for x in d)
    assert math.isclose(sum(d), max(scores))


def test_deltas_repeated_query_credited_once():
    assert deltas([1.0, 1.0, 1.0]) == [1.0, 0.0, 0.0]
