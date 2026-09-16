"""Which tau2 trajectories are fit for SFT, and how to reshape them for the chat template.

Pure functions over the dicts in a tau2 `results.json`, so they are testable without
installing tau2 (which lives in its own Python 3.12 venv). `scripts/build_sft.py` is the
caller that supplies the pieces only tau2 knows: the tool schemas and the agent's system
prompt.

Acceptance is rejection sampling: the episode must have solved the task, ended normally,
and never repeated a call. Whether a tool error also disqualifies it is a flag, not a
decision, because the trade-off is real and belongs to measurement. Strict data is clean
but teaches nothing about recovery, and this student errors in most episodes -- the first
two correct trajectories ever sampled here were both rejected for one hallucinated phone
number early on, before the agent recovered and solved the task. Report both rates on the
teacher pilot and pick with the numbers in hand.

Two things are deliberately *not* rejected here. Long trajectories
pass the filter and are ranked by length afterwards, because "shortest clean trajectory
per task" is a better selector than any cutoff this module could justify. And trajectories
where the user did most of the work pass too -- in telecom 83% of expected actions have
`requestor: user`, so an agent that talks the user through a fix is succeeding, not idling.
"""

from __future__ import annotations

import json

NORMAL_STOPS = ("user_stop", "agent_stop")


def reward(sim: dict) -> float:
    return (sim.get("reward_info") or {}).get("reward") or 0.0


def agent_messages(sim: dict) -> list[dict]:
    return [m for m in sim.get("messages", []) if m.get("role") == "assistant"]


def agent_calls(sim: dict) -> list[dict]:
    """Every tool call the agent made, in order. User-side actions are not the agent's."""
    return [
        c
        for m in agent_messages(sim)
        for c in (m.get("tool_calls") or [])
        if c.get("requestor", "assistant") == "assistant"
    ]


def _call_key(call: dict) -> str:
    return f"{call.get('name')}:{json.dumps(call.get('arguments') or {}, sort_keys=True)}"


def reject_reason(sim: dict, *, allow_tool_errors: bool = False) -> str | None:
    """Why this trajectory is unfit for SFT, or None if it passes every check."""
    if reward(sim) < 1.0:
        return "wrong"
    if sim.get("termination_reason") not in NORMAL_STOPS:
        return f"abnormal_stop:{sim.get('termination_reason')}"
    if not allow_tool_errors and any(
        m.get("error") for m in sim.get("messages", []) if m.get("role") == "tool"
    ):
        return "tool_error"
    keys = [_call_key(c) for c in agent_calls(sim)]
    if len(keys) != len(set(keys)):
        return "duplicate_call"
    if not agent_messages(sim):
        return "empty"
    return None


def to_messages(sim: dict, system_prompt: str) -> list[dict]:
    """tau2's transcript in the OpenAI chat shape `sft.data.encode` expects.

    Three shape changes. The system prompt is prepended, since tau2 keeps it outside the
    transcript. Tool calls move from tau2's flat `{id, name, arguments}` into the nested
    `{"function": {...}}` form the chat template renders. Everything the *user* did --
    their own tool calls, in this dual-control domain -- is dropped from the agent's view,
    because the agent never saw those calls, only the user's words about them.
    """
    out = [{"role": "system", "content": system_prompt}]
    for m in sim.get("messages", []):
        role = m.get("role")
        if role == "assistant":
            calls = [c for c in (m.get("tool_calls") or []) if c.get("requestor", "assistant") == "assistant"]
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if calls:
                msg["tool_calls"] = [
                    {"id": c.get("id"), "function": {"name": c["name"], "arguments": c.get("arguments") or {}}}
                    for c in calls
                ]
            out.append(msg)
        elif role == "tool":
            if m.get("requestor", "assistant") != "assistant":
                continue
            out.append({"role": "tool", "tool_call_id": m.get("id"), "content": m.get("content") or ""})
        elif role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
    return out


def length(sim: dict) -> tuple[int, int]:
    """Sort key for picking the shortest clean trajectory: agent turns, then tool calls."""
    return len(agent_messages(sim)), len(agent_calls(sim))
