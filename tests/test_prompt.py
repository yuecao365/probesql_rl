from env.prompt import Task, build_messages


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

