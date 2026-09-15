"""Does the served model emit what the chat template renders? Run before trusting SFT data.

    python scripts/check_think_block.py --base-url http://localhost:8000/v1 --model qwen8b

Qwen3's template gives the *last* assistant turn an empty `<think></think>` block and
earlier turns none, so a trajectory is encoded in two different formats. Training on that
is only correct if it matches what the model actually produces at rollout time. This asks
the live endpoint, with thinking on and off, and prints the raw reply next to the template's
rendering of that same reply so the two can be compared by eye.

The decision this informs: strip the block from SFT data, normalize every turn to carry it,
or leave it. Nothing about masking is trustworthy until it is made.
"""

from __future__ import annotations

import argparse
import json

import openai

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Read the device status.",
            "parameters": {"type": "object", "properties": {"device": {"type": "string"}}, "required": ["device"]},
        },
    }
]
MESSAGES = [
    {"role": "system", "content": "You are a telecom support agent. Use the tools."},
    {"role": "user", "content": "My mobile data is not working."},
]


def ask(client, model: str, enable_thinking: bool | None):
    kwargs = {}
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    r = client.chat.completions.create(
        model=model, messages=MESSAGES, tools=TOOLS, temperature=0.0, max_tokens=512, **kwargs
    )
    return r.choices[0].message


def show(label: str, m) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"content          : {m.content!r}")
    print(f"reasoning_content: {getattr(m, 'reasoning_content', None)!r}")
    calls = [{"name": c.function.name, "arguments": c.function.arguments} for c in (m.tool_calls or [])]
    print(f"tool_calls       : {json.dumps(calls, ensure_ascii=False)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen8b")
    ap.add_argument("--tokenizer", default="models/Qwen3-8B")
    args = ap.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key="dummy")
    replies = {}
    for label, flag in [("default (no chat_template_kwargs)", None), ("enable_thinking=False", False), ("enable_thinking=True", True)]:
        try:
            m = ask(client, args.model, flag)
            replies[label] = m
            show(label, m)
        except Exception as e:  # a server that rejects the kwarg is itself the answer
            print(f"\n{label}: FAILED -- {e}")

    # Now the other half: what the template produces when that reply is fed back as history.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    m = replies.get("enable_thinking=False") or next(iter(replies.values()), None)
    if m is None:
        return
    assistant = {"role": "assistant", "content": m.content or ""}
    if m.tool_calls:
        assistant["tool_calls"] = [
            {"type": "function", "function": {"name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")}}
            for c in m.tool_calls
        ]
    print(f"\n{'=' * 70}\ntemplate rendering of that same reply\n{'=' * 70}")
    for n, msgs in [("as the last turn", MESSAGES + [assistant]),
                    ("as a historical turn", MESSAGES + [assistant, {"role": "tool", "content": "ok"}, {"role": "assistant", "content": "done"}])]:
        text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=False)
        tail = text[text.index("<|im_start|>assistant") :]
        print(f"\n--- {n} ---\n{tail[:400]}")

    print(f"\n{'=' * 70}")
    print("Compare: does the raw reply above contain the <think></think> block that the")
    print("template inserts? If they differ, SFT data and rollout disagree on every turn.")


if __name__ == "__main__":
    main()
