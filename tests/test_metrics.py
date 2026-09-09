import pytest

from eval.metrics import buckets, summarize


def step(tool, overlap=None, observation="", duplicate=False, idle=0, degenerate=False):
    return {"tool": tool, "args": {}, "observation": observation, "ok": not observation.startswith("Error"),
            "duplicate": duplicate, "idle_streak": idle, "degenerate": degenerate, "overlap": overlap}


def rec(qid, correct, steps, status="submitted", difficulty="simple", tokens=0):
    return {"id": qid, "k": 0, "difficulty": difficulty, "status": status, "correct": correct, "steps": steps,
            "messages": [{"role": "assistant", "usage": {"prompt_tokens": 1, "completion_tokens": tokens}}]}


def test_empty():
    assert summarize([]) == {"trajectories": 0}


def test_buckets_by_question_across_rollouts():
    records = [rec("a", True, []), rec("a", False, []), rec("b", False, []), rec("c", True, []), rec("c", True, [])]
    assert buckets(records) == {"all_fail": ["b"], "mixed": ["a"], "all_pass": ["c"]}
    s = summarize(records)
    assert s["questions"] == 3 and s["pass^G"] == pytest.approx(1 / 3)
    assert s["buckets"] == pytest.approx({"all_fail": 1 / 3, "mixed": 1 / 3, "all_pass": 1 / 3})


def test_first_sql_and_recovery_ignore_probe_turns_and_trajectories_without_sql():
    records = [
        rec("a", True, [step("list_tables"), step("run_sql", 0.5), step("submit", 1.0)]),  # wrong first, recovered
        rec("b", False, [step("run_sql", 0.0), step("submit", 0.0)]),  # wrong first, not recovered
        rec("c", True, [step("submit", 1.0)]),  # right first
        rec("d", False, [step("list_tables")], status="max_turns"),  # no SQL turn at all
    ]
    s = summarize(records)
    assert s["first_sql_acc"] == pytest.approx(1 / 3)
    assert s["recovery_rate"] == pytest.approx(1 / 2)
    assert s["probe_turn_ratio"] == pytest.approx(2 / 7)
    assert s["avg_turns"] == pytest.approx(7 / 4)


def test_hallucination_duplicate_degenerate_idle_are_per_trajectory():
    records = [
        rec("a", False, [step("run_sql", 0.0, "Error: no such column: x"), step("run_sql", 0.0, "Error: no such column: x", duplicate=True, idle=2)]),
        rec("b", True, [step("submit", 1.0, degenerate=True)]),
    ]
    s = summarize(records)
    assert s["hallucination_rate"] == 0.5 and s["duplicate_rate"] == 0.5
    assert s["degenerate_rate"] == 0.5 and s["idle_rate"] == 0.5


def test_tokens_status_and_difficulty():
    records = [rec("a", True, [], tokens=100, difficulty="simple"), rec("b", False, [], status="parse_error", tokens=300, difficulty="challenging")]
    s = summarize(records)
    assert s["avg_completion_tokens"] == 200
    assert s["status"] == {"submitted": 1, "parse_error": 1}
    assert s["pass@1_by_difficulty"] == {"challenging": 0.0, "simple": 1.0}


def test_missing_difficulty_is_grouped_as_all():
    assert summarize([rec("a", True, [], difficulty=None)])["pass@1_by_difficulty"] == {"all": 1.0}


def test_lenient_call_ratio_counts_only_messages_with_calls():
    r = rec("a", True, [])
    r["messages"] = [{"role": "assistant", "tool_calls": [1], "lenient": True}, {"role": "assistant", "tool_calls": [1]},
                     {"role": "assistant", "content": "no call"}, {"role": "tool"}]
    assert summarize([r])["lenient_call_ratio"] == 0.5
