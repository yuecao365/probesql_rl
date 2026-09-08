import sqlite3

import pytest

from env import db
from env.prompt import Task
from env.rollout import run


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE t (id INTEGER, v INTEGER)")
        c.executemany("INSERT INTO t VALUES (?, ?)", [(i, i * 10) for i in range(4)])
    return db.connect(str(path))


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c-{name}", "function": {"name": name, "arguments": args}}]}


def scripted(*replies):
    it = iter(replies)
    return lambda messages, tools: next(it)


TASK = Task("db", "ids with v >= 20", gold_sql="SELECT id FROM t WHERE v >= 20")


def test_submit_ends_episode_and_scores(conn):
    traj = run(TASK, conn, scripted(call("submit", query="SELECT id FROM t WHERE v > 15")), max_turns=5)
    assert traj.status == "submitted" and traj.correct is True
    assert traj.final_sql == "SELECT id FROM t WHERE v > 15"
    assert len(traj.steps) == 1 and traj.steps[0].overlap == 1.0
    assert traj.messages[-1]["role"] == "assistant"  # no observation appended after submit


def test_parse_error_when_no_tool_call(conn):
    traj = run(TASK, conn, scripted({"role": "assistant", "content": "SELECT 1"}), max_turns=5)
    assert traj.status == "parse_error" and traj.steps == [] and traj.correct is None


def test_truncated_reply(conn):
    traj = run(TASK, conn, scripted({"role": "assistant", "content": "...", "truncated": True}), max_turns=5)
    assert traj.status == "truncated"


def test_turn_budget_exhausted(conn):
    traj = run(TASK, conn, scripted(*[call("list_tables")] * 3), max_turns=3)
    assert traj.status == "max_turns" and len(traj.steps) == 3 and traj.correct is None
    assert [s.duplicate for s in traj.steps] == [False, True, True]
    assert traj.steps[-1].idle_streak == 2


def test_observation_appended_with_tool_call_id(conn):
    traj = run(TASK, conn, scripted(call("describe_table", table="t"), call("submit", query="SELECT 1")), max_turns=5)
    tool_msg = traj.messages[3]
    assert tool_msg == {"role": "tool", "tool_call_id": "c-describe_table", "content": "id INTEGER\nv INTEGER"}


def test_run_sql_overlaps_and_failures(conn):
    traj = run(
        TASK, conn,
        scripted(
            call("run_sql", query="SELECT id FROM t"),  # superset: 2/4
            call("run_sql", query="SELECT nope FROM t"),  # error
            call("run_sql", query="SELECT id FROM t WHERE 1=0"),  # degenerate, empty
            call("submit", query="SELECT id FROM t WHERE v = 20"),  # subset, wrong
        ),
        max_turns=5,
    )
    ok = [s.ok for s in traj.steps]
    assert ok == [True, False, True, True]
    assert [s.overlap for s in traj.steps] == pytest.approx([0.5, 0.0, 0.0, 0.5])
    assert [s.degenerate for s in traj.steps] == [False, False, True, False]
    assert traj.steps[1].observation == "Error: no such column: nope"
    assert traj.correct is False


def test_failed_submit_is_incorrect_not_none(conn):
    traj = run(TASK, conn, scripted(call("submit", query="DROP TABLE t")), max_turns=5)
    assert traj.status == "submitted" and traj.correct is False and traj.steps[0].ok is False


def test_without_gold_nothing_is_scored(conn):
    task = Task("db", "q")
    traj = run(task, conn, scripted(call("run_sql", query="SELECT id FROM t"), call("submit", query="SELECT id FROM t")), max_turns=5)
    assert [s.overlap for s in traj.steps] == [None, None] and traj.correct is None


def test_unknown_tool_is_an_error_observation_not_a_crash(conn):
    traj = run(TASK, conn, scripted(call("drop_db"), call("submit", query="SELECT 1")), max_turns=5)
    assert traj.steps[0].ok is False and traj.steps[0].observation.startswith("Error: unknown tool")


def test_only_first_tool_call_is_used(conn):
    reply = call("list_tables")
    reply["tool_calls"].append(call("submit", query="SELECT 1")["tool_calls"][0])
    traj = run(TASK, conn, scripted(reply, call("submit", query="SELECT 1")), max_turns=5)
    assert [s.tool for s in traj.steps] == ["list_tables", "submit"]


def test_broken_gold_raises_before_any_turn(conn):
    with pytest.raises(db.DbError):
        run(Task("db", "q", gold_sql="SELECT nope FROM t"), conn, scripted(), max_turns=5)
