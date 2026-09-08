import pytest

from env.verifier import Tracker, is_degenerate, is_read_query, normalize


@pytest.mark.parametrize(
    "a, b",
    [
        ("SELECT a FROM t", "select   a\nfrom t;"),
        ("SELECT a FROM t WHERE x = 1", "SELECT a FROM t WHERE x=1"),
        ("SELECT a FROM t -- note", "SELECT a FROM t"),
        ("SELECT /* c */ a FROM t", "SELECT a FROM t"),
        ("SELECT count( * ) FROM t", "SELECT count(*) FROM t"),
    ],
)
def test_normalize_equivalences(a, b):
    assert normalize(a) == normalize(b)


def test_normalize_keeps_semantic_differences():
    assert normalize("SELECT a FROM t") != normalize("SELECT b FROM t")
    assert normalize("SELECT a FROM t WHERE x = 1") != normalize("SELECT a FROM t WHERE x = 10")


@pytest.mark.parametrize("sql", ["SELECT 1", "  with c as (select 1) select * from c", "-- hi\nSELECT 1", "Select 1;"])
def test_read_queries(sql):
    assert is_read_query(sql)


@pytest.mark.parametrize("sql", ["PRAGMA table_info(t)", "EXPLAIN SELECT 1", "INSERT INTO t VALUES (1)", "", "selection"])
def test_non_read_queries(sql):
    assert not is_read_query(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t WHERE 1=0",
        "SELECT a FROM t WHERE 1 = 0",
        "SELECT a FROM t WHERE false",
        "SELECT a FROM t WHERE 1<>1",
        "SELECT a FROM t LIMIT 0",
        "SELECT 1",
        "SELECT 'x', 2",
        "SELECT fromage",
    ],
)
def test_degenerate(sql):
    assert is_degenerate(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t WHERE 10=0",
        "SELECT a FROM t WHERE x = 0",
        "SELECT a FROM t LIMIT 10",
        "SELECT a FROM t WHERE flag = false",
        "SELECT count(*) FROM t",
    ],
)
def test_not_degenerate(sql):
    assert not is_degenerate(sql)


def test_tracker_duplicate_ignores_formatting():
    t = Tracker()
    assert not t.record("run_sql", {"query": "SELECT a FROM t"}, ok=True)
    assert t.record("run_sql", {"query": "select a\nfrom t;"}, ok=True)


def test_tracker_same_query_different_tool_is_not_duplicate():
    t = Tracker()
    t.record("run_sql", {"query": "SELECT a FROM t"}, ok=True)
    assert not t.record("submit", {"query": "SELECT a FROM t"}, ok=True)


def test_tracker_non_sql_args_order_independent():
    t = Tracker()
    t.record("sample_rows", {"table": "t", "n": 3}, ok=True)
    assert t.record("sample_rows", {"n": 3, "table": "t"}, ok=True)
    assert not t.record("sample_rows", {"table": "t", "n": 2}, ok=True)


def test_idle_streak_counts_failures_and_repeats_and_resets():
    t = Tracker()
    t.record("run_sql", {"query": "SELECT a FROM t"}, ok=False)
    t.record("run_sql", {"query": "SELECT a FROM t"}, ok=False)
    assert t.idle_streak == 2
    t.record("run_sql", {"query": "SELECT b FROM t"}, ok=True)
    assert t.idle_streak == 0
    t.record("run_sql", {"query": "SELECT b FROM t"}, ok=True)
    assert t.idle_streak == 1
