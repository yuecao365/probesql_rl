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


def test_no_observation_leaks_at_any_trajectory_length(tok):
    """The bug this guards: Qwen3's template is not prefix-stable, so span arithmetic
    derived from prefix renders drifts and pulls tool output into the loss."""
    for n in range(3, len(MESSAGES) + 1):
        ex = encode(tok, MESSAGES[:n], TOOLS, max_len=4096)
        on = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE])
        assert "SECRET_OBSERVATION" not in on, f"observation leaked at length {n}"
        assert "<|im_start|>" not in on, f"scaffolding leaked at length {n}"


def test_every_learned_span_is_one_assistant_turn(tok):
    ex = encode(tok, MESSAGES, TOOLS, max_len=4096)
    ids, labels = ex["input_ids"], ex["labels"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    spans, i = [], 0
    while i < len(ids):
        if labels[i] != IGNORE:
            j = i
            while j < len(ids) and labels[j] != IGNORE:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    assert len(spans) == 2                                  # two assistant turns
    assert all(ids[e - 1] == im_end for _, e in spans)      # each closes on <|im_end|>


def test_think_block_is_prompt_not_generation(tok):
    """Served with enable_thinking=False, `<think>\n\n</think>` is prefilled into the
    prompt, so it must never be learned -- and every turn must carry it, or turns are
    conditioned on a prefix the model never saw at rollout time."""
    ex = encode(tok, MESSAGES, TOOLS, max_len=4096)
    ids, labels = ex["input_ids"], ex["labels"]
    text = tok.decode(ids)
    if "<think>" not in text:
        pytest.skip("not a thinking-style template")
    on = tok.decode([t for t, l in zip(ids, labels) if l != IGNORE])
    assert "<think>" not in on and "</think>" not in on
    # one think block per assistant turn, none left bare
    assert text.count("<think>") == text.count("<|im_start|>assistant")
    assert "<|im_start|>assistant\n<tool_call>" not in text
