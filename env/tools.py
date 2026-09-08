"""The six tools a policy may call, and the text each one returns.

The schema is hidden from the prompt, so the tools are the only way to learn
it: list tables, describe one, look at real rows, search column names, run a
query, submit. Every probing tool returns a string and never raises; failures
become observations the model must read and react to, which is the behavior
being trained. Observation formatting lives here and nowhere else, since the
exact text is part of what SFT data and RL rollouts must share.

`run_sql` and `submit` are declared here but executed by the rollout loop via
`execute()`, because their full result feeds the reward while the model only
sees the first MAX_ROWS rows.
"""

from __future__ import annotations

import sqlite3

from env import db, verifier

MAX_ROWS = 20  # rows shown to the model
RESULT_ROWS = 10_000  # rows kept for reward computation
MAX_SAMPLE_ROWS = 5
MAX_CELL_CHARS = 64
TIMEOUT_S = 5.0

PROBES = ("list_tables", "describe_table", "sample_rows", "search_column")

SPECS = [
    {"name": "list_tables", "description": "List all tables in the database.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "describe_table", "description": "Columns, types, primary key and foreign keys of one table.", "parameters": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]}},
    {"name": "sample_rows", "description": f"Show up to n real rows of a table (n <= {MAX_SAMPLE_ROWS}). Use it to learn value formats.", "parameters": {"type": "object", "properties": {"table": {"type": "string"}, "n": {"type": "integer", "default": 3}}, "required": ["table"]}},
    {"name": "search_column", "description": "Find columns across all tables whose name contains the keyword (case-insensitive).", "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}},
    {"name": "run_sql", "description": f"Execute a read-only SELECT. Returns at most {MAX_ROWS} rows or the error message.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "submit", "description": "Submit the final SQL and end the episode.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _cell(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bytes):
        return f"<blob {len(v)}B>"
    s = str(v)
    return s if len(s) <= MAX_CELL_CHARS else s[: MAX_CELL_CHARS - 1] + "…"


def render(result: db.Result, max_rows: int = MAX_ROWS) -> str:
    if not result.rows:
        return "(empty result)"
    shown = result.rows[:max_rows]
    lines = [" | ".join(result.columns)]
    lines += [" | ".join(_cell(c) for c in row) for row in shown]
    if result.truncated or len(result.rows) > max_rows:
        lines.append(f"... (showing first {len(shown)} rows, more exist)")
    return "\n".join(lines)


class Toolbox:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def call(self, name: str, args: dict) -> str:
        """Dispatch a probing tool; run_sql and submit are the rollout's job."""
        if name not in PROBES:
            return f"Error: unknown tool '{name}'"
        try:
            return getattr(self, name)(**args)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"

    def execute(self, query: str) -> db.Result:
        """Full result of a policy-written SELECT. Raises db.DbError."""
        if not verifier.is_read_query(query):
            raise db.DbError("only SELECT queries are allowed")
        return db.execute(self.conn, query, timeout_s=TIMEOUT_S, max_rows=RESULT_ROWS)

    def tables(self) -> list[str]:
        rows = db.execute(self.conn, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").rows
        return [r[0] for r in rows]

    def list_tables(self) -> str:
        return "\n".join(self.tables())

    def describe_table(self, table: str) -> str:
        if table not in self.tables():
            return f"Error: no such table '{table}'"
        cols = db.execute(self.conn, f"PRAGMA table_info({_quote(table)})").rows
        fks = db.execute(self.conn, f"PRAGMA foreign_key_list({_quote(table)})").rows
        # table_info: (cid, name, type, notnull, default, pk); foreign_key_list: (id, seq, table, from, to, ...)
        lines = [f"{c[1]} {c[2] or 'ANY'}{' PRIMARY KEY' if c[5] else ''}" for c in cols]
        lines += [f"FOREIGN KEY {fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks]
        return "\n".join(lines)

    def sample_rows(self, table: str, n: int = 3) -> str:
        if table not in self.tables():
            return f"Error: no such table '{table}'"
        n = max(1, min(int(n), MAX_SAMPLE_ROWS))
        return render(db.execute(self.conn, f"SELECT * FROM {_quote(table)} LIMIT {n}"))

    def search_column(self, keyword: str) -> str:
        kw = keyword.lower()
        hits = []
        for table in self.tables():
            for c in db.execute(self.conn, f"PRAGMA table_info({_quote(table)})").rows:
                if kw in c[1].lower():
                    hits.append(f"{table}.{c[1]} {c[2] or 'ANY'}")
        return "\n".join(hits) if hits else f"No column name contains '{keyword}'"
