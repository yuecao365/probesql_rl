import sqlite3

import pytest

from env import db
from env.tools import MAX_CELL_CHARS, MAX_OBS_CHARS, MAX_ROWS, MAX_SAMPLE_ROWS, SPECS, Toolbox, clip, render


@pytest.fixture
def box(tmp_path):
    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE "my table" (id INTEGER PRIMARY KEY, "Full Name" TEXT, blob BLOB)')
        c.execute("CREATE TABLE orders (oid INTEGER PRIMARY KEY, uid INTEGER, note, FOREIGN KEY(uid) REFERENCES \"my table\"(id))")
        c.executemany('INSERT INTO "my table" VALUES (?, ?, ?)', [(i, "x" * 100 if i == 0 else f"name{i}", b"\x00" * 3) for i in range(30)])
        c.execute("INSERT INTO orders VALUES (1, 1, NULL)")
    return Toolbox(db.connect(str(path)))


def test_specs_cover_all_six_tools_but_sql_tools_are_not_dispatched(box):
    names = [s["name"] for s in SPECS]
    assert names == ["list_tables", "describe_table", "sample_rows", "search_column", "run_sql", "submit"]
    assert box.call("submit", {"query": "SELECT 1"}).startswith("Error: unknown tool")
    assert box.call("run_sql", {"query": "SELECT 1"}).startswith("Error: unknown tool")
    assert box.call("execute", {"query": "SELECT 1"}).startswith("Error: unknown tool")


def test_unknown_tool_and_private_methods_are_errors(box):
    assert box.call("drop_everything", {}).startswith("Error: unknown tool")
    assert box.call("tables", {}).startswith("Error: unknown tool")
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
    assert render(box.execute("SELECT note FROM orders")).splitlines()[1] == "NULL"


def test_search_column_case_insensitive_and_no_hits(box):
    assert box.call("search_column", {"keyword": "NAME"}) == "my table.Full Name TEXT"
    assert box.call("search_column", {"keyword": "zzz"}).startswith("No column")


def test_execute_keeps_full_result_but_render_shows_max_rows(box):
    res = box.execute('SELECT id FROM "my table"')
    assert len(res.rows) == 30 and not res.truncated
    lines = render(res).splitlines()
    assert len(lines) == MAX_ROWS + 2 and lines[-1].startswith("... (showing first 20")


def test_render_flags_db_level_truncation(box):
    res = db.execute(box.conn, 'SELECT id FROM "my table"', max_rows=3)
    assert render(res, max_rows=10).splitlines()[-1] == "... (showing first 3 rows, more exist)"


@pytest.mark.parametrize(
    "query, message",
    [
        ("SELECT nope FROM orders", "no such column: nope"),
        ("DELETE FROM orders", "only SELECT queries are allowed"),
        ("PRAGMA table_info(orders)", "only SELECT queries are allowed"),
    ],
)
def test_execute_rejections(box, query, message):
    with pytest.raises(db.DbError, match=message):
        box.execute(query)


def test_execute_timeout(box, monkeypatch):
    monkeypatch.setattr("env.tools.TIMEOUT_S", 0.1)
    with pytest.raises(db.QueryTimeout):
        box.execute("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c")


def test_render_empty_columns_only():
    assert render(db.Result(("a",), [], False)) == "(empty result)"


def test_clip_boundary():
    assert clip("x" * MAX_OBS_CHARS) == "x" * MAX_OBS_CHARS
    out = clip("x" * (MAX_OBS_CHARS + 1))
    assert out.startswith("x" * MAX_OBS_CHARS) and out.endswith("(observation truncated)")


def test_wide_observations_are_clipped(tmp_path):
    path = tmp_path / "w.sqlite"
    cols = ", ".join(f"c{i} TEXT" for i in range(200))
    with sqlite3.connect(path) as c:
        c.execute(f"CREATE TABLE wide ({cols})")
        c.execute("INSERT INTO wide VALUES (" + ", ".join(["'v'"] * 200) + ")")
    box = Toolbox(db.connect(str(path)))
    assert len(box.call("describe_table", {"table": "wide"})) <= MAX_OBS_CHARS + 30
    assert render(box.execute("SELECT * FROM wide")).endswith("(observation truncated)")
