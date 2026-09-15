"""Turn rollout trajectories into SFT examples with a per-token loss mask.

Encoding runs a trajectory through the model's own chat template with tool
arguments as JSON objects, which is what the model emits natively and what the
serving side renders back into its context, so the student trains on exactly
what it will see at rollout time.

The mask is found by scanning the rendered token stream for ChatML turn
boundaries: everything between a `<|im_start|>assistant\n` marker and the
`<|im_end|>` that closes it is the model's own, and everything else (system,
user, tool results, template scaffolding) is masked out of the loss. An earlier
version derived the spans from prefix lengths instead -- render `messages[:i]`
with a generation prompt, render `messages[:i+1]` without, take the difference.
That assumed the template is prefix-stable, and Qwen3's is not: it injects an
empty `<think></think>` block into the *last* assistant turn only, so a prefix
render is not a prefix of the full render and every span came out misaligned,
silently leaking tool output into the loss. Scanning the final stream cannot
drift from it.

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


def _render(tokenizer, messages, tools) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    return tokenizer.encode(text, add_special_tokens=False)


def _find(haystack: list[int], needle: list[int], start: int) -> int:
    """Index of the next occurrence of `needle` at or after `start`, or -1."""
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def assistant_spans(tokenizer, ids: list[int]) -> list[tuple[int, int]]:
    """[start, end) of every assistant turn, `<|im_end|>` included: the model emitted it."""
    opener = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    closer = tokenizer.convert_tokens_to_ids("<|im_end|>")
    spans, at = [], 0
    while (i := _find(ids, opener, at)) != -1:
        start = i + len(opener)
        end = next((j for j in range(start, len(ids)) if ids[j] == closer), len(ids) - 1)
        spans.append((start, end + 1))
        at = end + 1
    return spans


def encode(tokenizer, messages: list[dict], tools: list[dict], max_len: int) -> dict | None:
    """input_ids + labels (IGNORE outside assistant turns), or None if too long."""
    ids = _render(tokenizer, [_template_message(m) for m in messages], tools)
    if len(ids) > max_len:
        return None
    labels = [IGNORE] * len(ids)
    for start, end in assistant_spans(tokenizer, ids):
        labels[start:end] = ids[start:end]
    return {"input_ids": ids, "labels": labels}
