import hashlib
import sqlite3

import pytest

from env.db import DbError, QueryTimeout, Result, connect, execute


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        c.executemany("INSERT INTO t VALUES (?, ?)", [(i, f"n{i}") for i in range(10)])
        # Invalid UTF-8 bytes stored as TEXT, as found in some benchmark DBs.
        c.execute("INSERT INTO t VALUES (99, CAST(x'ff41' AS TEXT))")
    return str(path)


@pytest.fixture
def conn(db_path):
    return connect(db_path)


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_missing_file_raises_db_error(tmp_path):
    with pytest.raises(DbError):
        connect(str(tmp_path / "nope.sqlite"))


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1, 'x')",
        "UPDATE t SET name = 'x'",
        "DELETE FROM t",
        "DROP TABLE t",
        "CREATE TABLE u (a)",
        "CREATE TEMP TABLE u (a)",
        "ATTACH DATABASE ':memory:' AS m",
        "PRAGMA user_version = 3",
        "BEGIN",
    ],
)
def test_writes_are_rejected_and_file_untouched(db_path, conn, sql):
    before = _digest(db_path)
    with pytest.raises(DbError):
        execute(conn, sql)
    assert _digest(db_path) == before
    assert execute(conn, "SELECT count(*) FROM t").rows == [(11,)]


def test_multiple_statements_rejected(conn):
    with pytest.raises(DbError):
        execute(conn, "SELECT 1; SELECT 2")


def test_syntax_error_is_db_error(conn):
    with pytest.raises(DbError, match="syntax"):
        execute(conn, "SELEC 1")


def test_empty_statement_is_db_error(conn):
    with pytest.raises(DbError):
        execute(conn, "-- nothing here")


def test_pragma_read_allowed(conn):
    res = execute(conn, "PRAGMA table_info(t)")
    assert [r[1] for r in res.rows] == ["id", "name"]


def test_truncation_boundary(conn):
    exact = execute(conn, "SELECT id FROM t WHERE id < 5", max_rows=5)
    assert len(exact.rows) == 5 and not exact.truncated
    over = execute(conn, "SELECT id FROM t WHERE id < 6", max_rows=5)
    assert len(over.rows) == 5 and over.truncated


def test_unlimited_rows_never_truncate(conn):
    res = execute(conn, "SELECT id FROM t", max_rows=None)
    assert len(res.rows) == 11 and not res.truncated


def test_columns_and_empty_result(conn):
    res = execute(conn, "SELECT id AS k, name FROM t WHERE 1 = 0")
    assert res == Result(("k", "name"), [], False)


def test_invalid_utf8_does_not_crash(conn):
    (name,) = execute(conn, "SELECT name FROM t WHERE id = 99").rows[0]
    assert name.endswith("A")


def test_timeout_then_connection_still_usable(conn):
    infinite = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    with pytest.raises(QueryTimeout):
        execute(conn, infinite, timeout_s=0.2)
    assert execute(conn, "SELECT 1").rows == [(1,)]
