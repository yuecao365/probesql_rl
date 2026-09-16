from sft.filter import agent_calls, length, reject_reason, to_messages

SYS = "system prompt"


def call(name="get_customer_by_phone", args=None, requestor="assistant", cid="c1"):
    return {"id": cid, "name": name, "arguments": args or {"phone_number": "1"}, "requestor": requestor}


def sim(reward=1.0, stop="user_stop", calls=(), tool_msgs=(), content="ok"):
    msgs = [{"role": "user", "content": "help"}]
    for c in calls:
        msgs.append({"role": "assistant", "content": content, "tool_calls": [c]})
    msgs += list(tool_msgs)
    if not calls:
        msgs.append({"role": "assistant", "content": content})
    return {"task_id": "t", "termination_reason": stop, "reward_info": {"reward": reward}, "messages": msgs}


def test_accepts_a_clean_trajectory():
    assert reject_reason(sim(calls=[call(cid="a"), call(args={"phone_number": "2"}, cid="b")])) is None


def test_rejects_unsolved():
    assert reject_reason(sim(reward=0.0)) == "wrong"


def test_rejects_abnormal_termination():
    assert reject_reason(sim(stop="max_steps")).startswith("abnormal_stop")
    assert reject_reason(sim(stop="agent_error")).startswith("abnormal_stop")


def test_tool_error_is_a_flag_not_a_verdict():
    s = sim(calls=[call()], tool_msgs=[{"role": "tool", "content": "Error: nope", "error": True}])
    assert reject_reason(s) == "tool_error"
    assert reject_reason(s, allow_tool_errors=True) is None


def test_rejects_a_repeated_call_even_with_reordered_arguments():
    a = call(args={"x": 1, "y": 2}, cid="a")
    b = call(args={"y": 2, "x": 1}, cid="b")
    assert reject_reason(sim(calls=[a, b])) == "duplicate_call"


def test_user_side_calls_are_not_the_agents():
    """In dual control the user acts too; those calls are neither the agent's work nor
    its duplicates."""
    s = sim(calls=[call(cid="a"), call(cid="b", requestor="user")])
    assert len(agent_calls(s)) == 1
    assert reject_reason(s) is None


def test_to_messages_nests_tool_calls_and_prepends_the_system_prompt():
    s = sim(calls=[call(cid="a")])
    out = to_messages(s, SYS)
    assert out[0] == {"role": "system", "content": SYS}
    assistant = next(m for m in out if m["role"] == "assistant" and m.get("tool_calls"))
    fn = assistant["tool_calls"][0]["function"]
    assert fn["name"] == "get_customer_by_phone" and isinstance(fn["arguments"], dict)


def test_to_messages_drops_observations_the_agent_never_saw():
    s = sim(calls=[call(cid="a")], tool_msgs=[
        {"role": "tool", "id": "a", "content": "agent sees this", "requestor": "assistant"},
        {"role": "tool", "id": "u", "content": "user-side only", "requestor": "user"},
    ])
    contents = [m.get("content") for m in to_messages(s, SYS)]
    assert "agent sees this" in contents and "user-side only" not in contents


def test_length_ranks_shorter_trajectories_first():
    short = sim(calls=[call(cid="a")])
    long = sim(calls=[call(cid="a"), call(args={"phone_number": "2"}, cid="b")])
    assert length(short) < length(long)
