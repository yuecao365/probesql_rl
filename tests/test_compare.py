import math

import pytest

from env.compare import deltas, equal, overlap


# --- exactly the official rule ------------------------------------------------


@pytest.mark.parametrize("a, b", [(1, 1.0), (0, -0.0), (True, 1)])
def test_python_equality_is_the_only_normalization(a, b):
    assert equal([(a,)], [(b,)])


@pytest.mark.parametrize(
    "a, b",
    [
        (1, "1"),
        (1.5, "1.50"),
        (1.0000001, 1.0),
        (39.752034, 39.75203),  # a ROUND()ed gold vs an unrounded prediction: the official judge rejects it
        ("007", 7),
        (None, 0),
        (None, ""),
        ("a", b"a"),
    ],
)
def test_no_leniency_beyond_the_official_judge(a, b):
    assert not equal([(a,)], [(b,)])


def test_null_matches_null_in_place():
    assert equal([(None, 1)], [(None, 1)])
    assert not equal([(None, 1)], [(1, None)])


def test_row_order_ignored():
    assert equal([(1,), (2,)], [(2,), (1,)])


def test_column_order_matters():
    assert not equal([("x", 1)], [(1, "x")])


def test_duplicates_ignored():
    assert equal([(1,), (1,)], [(1,)])


def test_different_arity_never_equal():
    assert not equal([(1, 2)], [(1,)])
    assert overlap([(1, 2)], [(1,)]) == 0.0


def test_accepts_lists_as_rows():
    assert equal([[1, "a"]], [(1, "a")])


# --- overlap ------------------------------------------------------------------


def test_overlap_of_two_empty_sets_is_one():
    assert overlap([], []) == 1.0 and equal([], [])


def test_overlap_with_one_empty_set_is_zero():
    assert overlap([], [(1,)]) == 0.0 and overlap([(1,)], []) == 0.0


def test_overlap_is_one_iff_equal():
    assert overlap([(1,), (2,), (2,)], [(2,), (1,)]) == 1.0
    assert overlap([(1,), (2,)], [(2, 1)]) == 0.0


def test_overlap_partial_and_symmetric():
    pred, gold = [(1,), (2,), (3,)], [(2,), (3,), (4,)]
    assert overlap(pred, gold) == pytest.approx(2 / 4) == overlap(gold, pred)


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
    assert d == pytest.approx([0.1, 0.8, -0.6, 0.3]) and math.isclose(sum(d), scores[-1])


def test_deltas_repeated_query_earns_nothing():
    assert deltas([1.0, 1.0, 1.0]) == [1.0, 0.0, 0.0]


def test_deltas_oscillation_nets_only_the_final_state():
    assert math.isclose(sum(deltas([0.9, 0.0, 0.9, 0.0])), 0.0)
