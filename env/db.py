"""Read-only SQLite execution sandbox.

The policy runs arbitrary SQL against benchmark databases during rollouts, so
every query must be unable to mutate the file, unable to run forever, and unable
to flood the context with rows. Safety is layered: the connection is opened in
read-only mode, and an allow-list authorizer rejects anything that is not a read
before it executes, so a write attempt fails with a clear DbError rather than a
cryptic SQLite message. Timeouts use SQLite's progress handler instead of threads,
which keeps execution single-threaded and lets the connection be reused after an
interrupt. Row truncation lives here; cell rendering does not (see tools.py).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass


class DbError(Exception):
    """Any failure to execute a query: syntax, write attempt, missing table..."""


class QueryTimeout(DbError):
    pass


@dataclass(frozen=True)
class Result:
    columns: tuple[str, ...]
    rows: list[tuple]
    truncated: bool  # more rows existed beyond max_rows


# Everything a SELECT (with CTEs, subqueries, functions) or a PRAGMA read needs.
# PRAGMA is allowed because schema probing relies on `PRAGMA table_info`; writing
# pragmas still fail because the connection itself is read-only.
_READ_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_RECURSIVE,
    }
)


def _authorize(action: int, *_args) -> int:
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def connect(path: str) -> sqlite3.Connection:
    """Open `path` read-only. Raises DbError if the file cannot be opened."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        raise DbError(str(e)) from e
    conn.execute("PRAGMA query_only = 1")
    conn.set_authorizer(_authorize)
    # Some benchmark databases contain non-UTF-8 text; replacing bad bytes beats
    # crashing the whole rollout on one cell.
    conn.text_factory = lambda b: b.decode("utf-8", "replace")
    return conn


def execute(
    conn: sqlite3.Connection, sql: str, *, timeout_s: float = 5.0, max_rows: int | None = 1000
) -> Result:
    """Run one read-only statement and return at most `max_rows` rows (all rows if None).

    Raises QueryTimeout if the statement does not finish within `timeout_s`,
    DbError for every other failure.
    """
    deadline = time.monotonic() + timeout_s
    # Checked every 1000 VM steps; returning True aborts the statement.
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
    try:
        cur = conn.execute(sql)
        if cur.description is None:
            raise DbError("statement returned no result set")
        rows = cur.fetchall() if max_rows is None else cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise QueryTimeout(f"query exceeded {timeout_s}s") from e
        raise DbError(str(e)) from e
    # Python 3.10 reports multi-statement input as sqlite3.Warning, not Error.
    except (sqlite3.Error, sqlite3.Warning) as e:
        raise DbError(str(e)) from e
    finally:
        conn.set_progress_handler(None, 0)
    columns = tuple(d[0] for d in cur.description)
    if max_rows is None:
        return Result(columns, rows, truncated=False)
    return Result(columns, rows[:max_rows], truncated=len(rows) > max_rows)
