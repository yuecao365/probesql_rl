import os

import pytest

from sft.data import IGNORE, as_tools, encode

MODELS = ["models/Qwen3-8B", "models/Qwen2.5-7B-Instruct"]
MODEL = next((m for m in MODELS if os.path.exists(f"{m}/tokenizer.json")), MODELS[0])
pytestmark = pytest.mark.skipif(not os.path.exists(f"{MODEL}/tokenizer.json"), reason="tokenizer not on disk")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


SPECS = [
    {"name": "get_status", "description": "Read device status.", "parameters": {"type": "object", "properties": {"device": {"type": "string"}}, "required": ["device"]}},
    {"name": "toggle_data", "description": "Turn mobile data on or off.", "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}},
]
TOOLS = as_tools(SPECS)

MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "My data is not working."},
    {"role": "assistant", "content": "Let me look.", "tool_calls": [{"id": "c1", "function": {"name": "get_status", "arguments": {"device": "phone"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "data_enabled: false\nSECRET_OBSERVATION"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {"name": "toggle_data", "arguments": {"on": True}}}]},
]


def test_mask_covers_only_assistant_text(tok):
    ex = encode(tok, MESSAGES, TOOLS, max_len=4096)
    ids, labels = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labels)
    on = tok.decode([t for t, l in zip(ids, labels) if l != IGNORE])
    off = tok.decode([t for t, l in zip(ids, labels) if l == IGNORE])
    assert "Let me look." in on and "<tool_call>" in on
    assert '"arguments": {"on": true}' in on  # an object, never an escaped string
    assert on.count("<|im_end|>") == 2 and "<|im_start|>" not in on
    assert "SECRET_OBSERVATION" not in on and "SECRET_OBSERVATION" in off
    assert "sys" in off and "not working" in off and "get_status" in off  # tool spec + user turn are context


def test_mask_boundaries(tok):
    ex = encode(tok, MESSAGES, TOOLS, max_len=4096)
    ids, labels = ex["input_ids"], ex["labels"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    # every <|im_end|> that closes an assistant turn is learned, the newline after it is not
    ends = [i for i, t in enumerate(ids) if t == im_end and labels[i] != IGNORE]
    assert len(ends) == 2 and all(labels[i + 1] == IGNORE for i in ends)
    # the generation prompt "<|im_start|>assistant\n" before each turn is context
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    assert all(labels[i] == IGNORE for i, t in enumerate(ids) if t == im_start)


def test_tools_reach_the_template(tok):
    """A tool absent from `tools` must not appear in the rendered context."""
    ex = encode(tok, MESSAGES, as_tools(SPECS[:1]), max_len=4096)
    text = tok.decode(ex["input_ids"])
    assert "get_status" in text and "Turn mobile data on or off" not in text


def test_too_long_is_dropped(tok):
    assert encode(tok, MESSAGES, TOOLS, max_len=50) is None
