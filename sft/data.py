"""Turn rollout trajectories into SFT examples with a per-token loss mask.

Encoding runs a trajectory through the model's own chat template with tool
arguments as JSON objects, which is what the model emits natively and what the
serving side renders back into its context, so the student trains on exactly
what it will see at rollout time. The mask is found by rendering each prefix:
tokens between "prefix + generation prompt" and "prefix + assistant turn" are
the model's own, everything else (system, user, tool results, template
scaffolding) is masked out of the loss.

Tool schemas are a parameter rather than an import: the environment owns them,
and this module has to encode whatever environment is in play. The acceptance
filter that decides which trajectories are worth encoding is likewise the
environment's business and lives next to it.
"""

from __future__ import annotations

IGNORE = -100


def as_tools(specs: list[dict]) -> list[dict]:
    """Wrap bare function schemas in the `{"type": "function"}` envelope templates expect."""
    return [{"type": "function", "function": s} for s in specs]


def _template_message(m: dict) -> dict:
    """The chat-template view of a message: roles, text and object-valued tool calls only."""
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = [{"type": "function", "function": c["function"]} for c in m["tool_calls"]]
    return out


def _tokens(tokenizer, messages, tools, add_generation_prompt: bool) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tokenizer.encode(text, add_special_tokens=False)


def encode(tokenizer, messages: list[dict], tools: list[dict], max_len: int) -> dict | None:
    """input_ids + labels (IGNORE outside assistant turns), or None if too long."""
    wire = [_template_message(m) for m in messages]
    ids = _tokens(tokenizer, wire, tools, add_generation_prompt=False)
    if len(ids) > max_len:
        return None
    labels = [IGNORE] * len(ids)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    for i, m in enumerate(wire):
        if m["role"] != "assistant":
            continue
        start = len(_tokens(tokenizer, wire[:i], tools, add_generation_prompt=True))
        end = len(_tokens(tokenizer, wire[: i + 1], tools, add_generation_prompt=False))
        # The template closes a turn with <|im_end|>\n; the model chose <|im_end|>, not the newline.
        if ids[end - len(newline) : end] == newline:
            end -= len(newline)
        labels[start:end] = ids[start:end]
    return {"input_ids": ids, "labels": labels}
