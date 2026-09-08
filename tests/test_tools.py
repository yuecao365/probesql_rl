import sqlite3

import pytest

from env import db
from env.tools import MAX_CELL_CHARS, MAX_ROWS, MAX_SAMPLE_ROWS, SPECS, Toolbox, render


@pytest.fixture
def box(tmp_path):
    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE "my table" (id INTEGER PRIMARY KEY, "Full Name" TEXT, blob BLOB)')
        c.execute("CREATE TABLE orders (oid INTEGER PRIMARY KEY, uid INTEGER, note, FOREIGN KEY(uid) REFERENCES \"my table\"(id))")
        c.executemany('INSERT INTO "my table" VALUES (?, ?, ?)', [(i, "x" * 100 if i == 0 else f"name{i}", b"\x00" * 3) for i in range(30)])
        c.execute("INSERT INTO orders VALUES (1, 1, NULL)")
    return Toolbox(db.connect(str(path)))


def test_specs_cover_all_six_tools_and_only_submit_is_not_callable(box):
    names = [s["name"] for s in SPECS]
    assert names == ["list_tables", "describe_table", "sample_rows", "search_column", "run_sql", "submit"]
    assert box.call("submit", {"query": "SELECT 1"}).startswith("Error")


def test_unknown_tool_and_private_methods_are_errors(box):
    assert box.call("drop_everything", {}).startswith("Error: unknown tool")
    assert box.call("_tables", {}).startswith("Error: unknown tool")
    assert box.call("call", {"name": "x", "args": {}}).startswith("Error: unknown tool")


def test_bad_arguments_are_errors_not_exceptions(box):
    assert box.call("describe_table", {}).startswith("Error: bad arguments")
    assert box.call("list_tables", {"extra": 1}).startswith("Error: bad arguments")


def test_describe_missing_table(box):
    assert box.call("describe_table", {"table": "nope"}) == "Error: no such table 'nope'"


def test_describe_shows_pk_fk_and_untyped_columns(box):
    out = box.call("describe_table", {"table": "orders"})
    assert "oid INTEGER PRIMARY KEY" in out
    assert "note ANY" in out
    assert 'FOREIGN KEY uid -> my table.id' in out


def test_table_names_with_spaces_are_quoted(box):
    out = box.call("sample_rows", {"table": "my table", "n": 1})
    assert out.splitlines()[0] == "id | Full Name | blob"


def test_sample_rows_clamps_n(box):
    assert len(box.call("sample_rows", {"table": "my table", "n": 100}).splitlines()) == MAX_SAMPLE_ROWS + 1
    assert len(box.call("sample_rows", {"table": "my table", "n": -5}).splitlines()) == 2


def test_cell_truncation_null_and_blob(box):
    out = box.call("sample_rows", {"table": "my table", "n": 1})
    cells = out.splitlines()[1].split(" | ")
    assert len(cells[1]) == MAX_CELL_CHARS and cells[1].endswith("…")
    assert cells[2] == "<blob 3B>"
    assert box.call("run_sql", {"query": "SELECT note FROM orders"}).splitlines()[1] == "NULL"


def test_search_column_case_insensitive_and_no_hits(box):
    assert box.call("search_column", {"keyword": "NAME"}) == "my table.Full Name TEXT"
    assert box.call("search_column", {"keyword": "zzz"}).startswith("No column")


def test_run_sql_truncates_at_max_rows(box):
    lines = box.call("run_sql", {"query": 'SELECT id FROM "my table"'}).splitlines()
    assert len(lines) == MAX_ROWS + 2 and lines[-1].startswith("... (showing first 20")


def test_run_sql_errors_become_text(box):
    assert box.call("run_sql", {"query": "SELECT nope FROM orders"}) == "Error: no such column: nope"
    assert box.call("run_sql", {"query": "DELETE FROM orders"}) == "Error: only SELECT queries are allowed"
    assert box.call("run_sql", {"query": "PRAGMA table_info(orders)"}) == "Error: only SELECT queries are allowed"
    assert box.call("run_sql", {"query": "SELECT oid FROM orders WHERE 1=0"}) == "(empty result)"


def test_run_sql_timeout_becomes_text(box, monkeypatch):
    monkeypatch.setattr("env.tools.TIMEOUT_S", 0.1)
    out = box.call("run_sql", {"query": "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"})
    assert out.startswith("Error: query exceeded")


def test_render_empty_columns_only():
    assert render(db.Result(("a",), [], False)) == "(empty result)"
