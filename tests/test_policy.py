import json
from types import SimpleNamespace as NS

import pytest

from env.policy import ChatPolicy, _extract_call, _to_wire


class FakeClient:
    def __init__(self, response):
        self.response, self.calls = response, []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self.response


def _resp(content="", tool_calls=None, finish="stop"):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg, finish_reason=finish)], usage=NS(prompt_tokens=10, completion_tokens=5))


def _call(name, arguments):
    return NS(id="call_1", function=NS(name=name, arguments=arguments))


def test_arguments_string_becomes_dict_and_usage_recorded():
    client = FakeClient(_resp("thinking", [_call("run_sql", '{"query": "SELECT 1"}')]))
    reply = ChatPolicy(client, "m", 0.5, 100)([{"role": "user", "content": "q"}], [{"name": "run_sql"}])
    assert reply["tool_calls"] == [{"id": "call_1", "function": {"name": "run_sql", "arguments": {"query": "SELECT 1"}}}]
    assert reply["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert "truncated" not in reply
    assert client.calls[0]["tools"] == [{"type": "function", "function": {"name": "run_sql"}}]


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]"])
def test_bad_arguments_mean_no_tool_call(arguments):
    reply = ChatPolicy(FakeClient(_resp("", [_call("run_sql", arguments)])), "m", 0, 1)([], [])
    assert "tool_calls" not in reply


def test_missing_arguments_is_a_no_arg_call():
    reply = ChatPolicy(FakeClient(_resp("", [_call("list_tables", None)])), "m", 0, 1)([], [])
    assert reply["tool_calls"][0]["function"]["arguments"] == {}


def test_length_finish_marks_truncated():
    reply = ChatPolicy(FakeClient(_resp("partial", None, finish="length")), "m", 0, 1)([], [])
    assert reply["truncated"] is True and "tool_calls" not in reply


def test_none_content_becomes_empty_string():
    reply = ChatPolicy(FakeClient(_resp(None, [_call("list_tables", "{}")])), "m", 0, 1)([], [])
    assert reply["content"] == ""


def test_to_wire_serializes_arguments_and_drops_internal_fields():
    internal = {"role": "assistant", "content": None, "usage": {"x": 1}, "truncated": False,
                "tool_calls": [{"id": "c1", "function": {"name": "submit", "arguments": {"query": "SELECT 1"}}}]}
    wire = _to_wire(internal)
    assert wire == {"role": "assistant", "content": "",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "submit", "arguments": '{"query": "SELECT 1"}'}}]}
    assert json.loads(wire["tool_calls"][0]["function"]["arguments"]) == {"query": "SELECT 1"}


def test_to_wire_leaves_tool_and_user_messages_alone():
    tool = {"role": "tool", "tool_call_id": "c1", "content": "x"}
    assert _to_wire(tool) is tool


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"name": "list_tables", "arguments": {}}', ("list_tables", {})),
        ('{"name": "list_tables"}', ("list_tables", {})),
        ('Let me look.\n```json\n{"name": "describe_table", "arguments": {"table": "a"}}\n```\nthen more', ("describe_table", {"table": "a"})),
        ('{"name": "a", "arguments": {}} {"name": "b", "arguments": {}}', ("a", {})),
        ('{"x": 1} {"name": "run_sql", "arguments": {"query": "SELECT \'{\' FROM t"}}', ("run_sql", {"query": "SELECT '{' FROM t"})),
        ('{broken {"name": "ok", "arguments": {}}', ("ok", {})),
    ],
)
def test_extract_call(text, expected):
    call = _extract_call(text)
    assert (call["function"]["name"], call["function"]["arguments"]) == expected and call["id"] == "lenient"


@pytest.mark.parametrize("text", ["", "SELECT 1", '{"arguments": {}}', '{"name": 3}', '{"name": "x", "arguments": "s"}', "{"])
def test_extract_call_rejects(text):
    assert _extract_call(text) is None


def test_lenient_only_when_server_found_nothing():
    strict = ChatPolicy(FakeClient(_resp('{"name": "b"}', [_call("a", "{}")])), "m", 0, 1, lenient=True)([], [])
    assert strict["tool_calls"][0]["function"]["name"] == "a" and "lenient" not in strict
    fallback = ChatPolicy(FakeClient(_resp('{"name": "b"}', None)), "m", 0, 1, lenient=True)([], [])
    assert fallback["tool_calls"][0]["function"]["name"] == "b" and fallback["lenient"] is True
    off = ChatPolicy(FakeClient(_resp('{"name": "b"}', None)), "m", 0, 1)([], [])
    assert "tool_calls" not in off
