import pytest

from env.prompt import Task, build_messages, parse_tool_call


def test_initial_state_has_tables_but_no_columns():
    msgs = build_messages(Task("db1", "How many?", gold_sql="SELECT count(*) FROM t"), ["t", "u"], max_turns=10)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Tables: t, u" in msgs[1]["content"]
    assert "Hint" not in msgs[1]["content"]
    assert "count(*)" not in msgs[0]["content"] + msgs[1]["content"]
    assert "at most 10 tool calls" in msgs[0]["content"]


def test_evidence_included_when_present():
    msgs = build_messages(Task("db1", "q", evidence="year means 2020"), ["t"], max_turns=5)
    assert msgs[1]["content"].endswith("Hint: year means 2020")


@pytest.mark.parametrize(
    "text, expected",
    [
        ('<tool_call>{"name": "list_tables", "arguments": {}}</tool_call>', ("list_tables", {})),
        ('<tool_call>{"name": "list_tables"}</tool_call>', ("list_tables", {})),
        ('think...\n<tool_call>\n{"name": "run_sql", "arguments": {"query": "SELECT 1"}}\n</tool_call>\n', ("run_sql", {"query": "SELECT 1"})),
        ('<tool_call>{bad json}</tool_call><tool_call>{"name": "submit", "arguments": {"query": "x"}}</tool_call>', ("submit", {"query": "x"})),
        ('<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>', ("a", {})),
    ],
)
def test_parse_tool_call(text, expected):
    assert parse_tool_call(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "SELECT 1",
        "<tool_call></tool_call>",
        '<tool_call>{"arguments": {}}</tool_call>',
        '<tool_call>{"name": 3}</tool_call>',
        '<tool_call>{"name": "x", "arguments": "not a dict"}</tool_call>',
        '<tool_call>{"name": "x", "arguments": {}}',
    ],
)
def test_parse_tool_call_rejects(text):
    assert parse_tool_call(text) is None
