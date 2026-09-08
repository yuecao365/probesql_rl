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


def test_null_equals_null_in_place():
    assert equal([(None, 1)], [(None, 1)])
    assert not equal([(None, 1)], [(1, None)])


# --- row / column order and duplicates (BIRD semantics) -----------------------


def test_row_order_ignored():
    assert equal([(1,), (2,)], [(2,), (1,)])


def test_column_order_matters():
    assert not equal([("x", 1)], [(1, "x")])


def test_duplicates_ignored():
    assert equal([(1,), (1,)], [(1,)])


def test_different_arity_never_equal():
    assert not equal([(1, 2)], [(1,)])
    assert overlap([(1, 2)], [(1,)]) == 0.0


def test_canonical_does_not_mutate_input():
    rows = [[2.0, "1"]]
    canonical(rows)
    assert rows == [[2.0, "1"]]


# --- overlap ------------------------------------------------------------------


def test_overlap_of_two_empty_sets_is_one():
    assert overlap([], []) == 1.0
    assert equal([], [])


def test_overlap_with_one_empty_set_is_zero():
    assert overlap([], [(1,)]) == 0.0
    assert overlap([(1,)], []) == 0.0


def test_overlap_is_one_iff_equal():
    assert overlap([(1,), (2,), (2,)], [(2,), (1,)]) == 1.0
    assert overlap([(1,), (2,)], [(2, 1)]) == 0.0


def test_overlap_partial_and_symmetric():
    pred, gold = [(1,), (2,), (3,)], [(2,), (3,), (4,)]
    assert overlap(pred, gold) == pytest.approx(2 / 4)
    assert overlap(pred, gold) == overlap(gold, pred)


def test_overlap_superset_is_penalized():
    assert overlap([(1,), (2,), (3,), (4,)], [(1,)]) == pytest.approx(1 / 4)


# --- deltas -------------------------------------------------------------------


def test_deltas_empty():
    assert deltas([]) == []


def test_deltas_first_turn_is_measured_from_zero():
    assert deltas([0.4]) == [0.4]


def test_deltas_regression_is_negative_and_sum_is_last():
    scores = [0.1, 0.9, 0.3, 0.6]
    d = deltas(scores)
    assert d == pytest.approx([0.1, 0.8, -0.6, 0.3])
    assert math.isclose(sum(d), scores[-1])


def test_deltas_repeated_query_earns_nothing():
    assert deltas([1.0, 1.0, 1.0]) == [1.0, 0.0, 0.0]


def test_deltas_oscillation_nets_only_the_final_state():
    assert math.isclose(sum(deltas([0.9, 0.0, 0.9, 0.0])), 0.0)
