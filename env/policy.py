"""Policy adapter over any OpenAI-compatible chat endpoint.

A vLLM server started with `--tool-call-parser hermes` and the DeepSeek API
expose the same interface, so one adapter serves both the local student and the
API teacher, and the server does chat-template rendering and tool-call parsing
in the model's native format. The adapter only translates between the wire
format and the message shape rollout.py expects: tool arguments are a JSON
string on the wire and a dict internally, and the internal `usage` / `truncated`
/ `lenient` fields never leave the process.

Untrained models often write the call as bare JSON without the `<tool_call>`
tags the server's parser needs. With `lenient=True` the first JSON object that
carries a `name` is accepted instead and the reply is flagged, so zero-shot
baselines can act while the flag still measures format legality. Trained
policies are run strict, matching what the RL framework's parser accepts.
"""

from __future__ import annotations

import json

_DECODER = json.JSONDecoder()


def _to_wire(message: dict) -> dict:
    if message.get("role") != "assistant":
        return message
    out = {"role": "assistant", "content": message.get("content") or ""}  # drops usage / truncated / lenient
    if message.get("tool_calls"):
        out["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])}}
            for c in message["tool_calls"]
        ]
    return out


def _extract_call(text: str) -> dict | None:
    """First JSON object in `text` shaped like {"name": ..., "arguments": {...}}."""
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments", {}), dict):
            return {"id": "lenient", "function": {"name": obj["name"], "arguments": obj.get("arguments", {})}}
        start = text.find("{", start + 1)
    return None


def _from_wire(choice, usage, lenient: bool) -> dict:
    msg = choice.message
    reply = {
        "role": "assistant",
        "content": msg.content or "",
        "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
    }
    if choice.finish_reason == "length":
        reply["truncated"] = True
    calls = []
    for c in msg.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            continue  # unparseable arguments count as no call, i.e. a parse_error turn
        if isinstance(args, dict):
            calls.append({"id": c.id, "function": {"name": c.function.name, "arguments": args}})
    if not calls and lenient and (found := _extract_call(reply["content"])):
        calls, reply["lenient"] = [found], True
    if calls:
        reply["tool_calls"] = calls
    return reply


class ChatPolicy:
    def __init__(self, client, model: str, temperature: float, max_tokens: int, lenient: bool = False):
        self.client, self.model, self.temperature, self.max_tokens, self.lenient = client, model, temperature, max_tokens, lenient

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[_to_wire(m) for m in messages],
            tools=[{"type": "function", "function": t} for t in tools],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return _from_wire(resp.choices[0], resp.usage, self.lenient)
