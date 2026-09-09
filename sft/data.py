"""Turn rollout trajectories into SFT examples with a per-token loss mask.

Acceptance is the plan's four-way rejection sampling: correct final answer,
normal termination, no failed tool call, no repeated call (and no call that
needed the lenient parser). Encoding renders the very same wire messages the
policy adapter sends, through the model's own chat template, so what the student
trains on is byte-identical to what it will see at rollout time. The mask is
found by rendering each prefix: tokens between "prefix + generation prompt" and
"prefix + assistant turn" are the model's own, everything else (system, user,
tool results, template scaffolding) is masked out of the loss.
"""

from __future__ import annotations

from env.policy import _to_wire
from env.tools import SPECS

TOOLS = [{"type": "function", "function": t} for t in SPECS]
IGNORE = -100


def reject_reason(record: dict) -> str | None:
    """Why a trajectory is unfit for SFT, or None if it passes all four checks."""
    if record["status"] != "submitted":
        return "not_submitted"
    if not record["correct"]:
        return "wrong"
    if any(not s["ok"] for s in record["steps"]):
        return "tool_error"
    if any(s["duplicate"] for s in record["steps"]):
        return "duplicate_call"
    if any(m.get("lenient") for m in record["messages"]):
        return "lenient_format"
    return None


def _tokens(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tokenizer.encode(text, add_special_tokens=False)


def encode(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """input_ids + labels (IGNORE outside assistant turns), or None if too long."""
    wire = [_to_wire(m) for m in messages]
    ids = _tokens(tokenizer, wire, add_generation_prompt=False)
    if len(ids) > max_len:
        return None
    labels = [IGNORE] * len(ids)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    for i, m in enumerate(wire):
        if m["role"] != "assistant":
            continue
        start = len(_tokens(tokenizer, wire[:i], add_generation_prompt=True))
        end = len(_tokens(tokenizer, wire[: i + 1], add_generation_prompt=False))
        # The template closes a turn with <|im_end|>\n; the model chose <|im_end|>, not the newline.
        if ids[end - len(newline) : end] == newline:
            end -= len(newline)
        labels[start:end] = ids[start:end]
    return {"input_ids": ids, "labels": labels}
