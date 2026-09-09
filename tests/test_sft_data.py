import os

import pytest

from sft.data import IGNORE, encode, reject_reason

MODEL = "models/Qwen2.5-7B-Instruct"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{MODEL}/tokenizer.json"), reason="tokenizer not on disk")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def step(ok=True, duplicate=False):
    return {"tool": "run_sql", "ok": ok, "duplicate": duplicate}


def record(correct=True, status="submitted", steps=(), lenient=False):
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "submit", "arguments": {"query": "SELECT 1"}}}]}]
    if lenient:
        msgs[0]["lenient"] = True
    return {"correct": correct, "status": status, "steps": list(steps), "messages": msgs}


@pytest.mark.parametrize(
    "rec, reason",
    [
        (record(correct=False), "wrong"),
        (record(status="max_turns"), "not_submitted"),
        (record(steps=[step(ok=False)]), "tool_error"),
        (record(steps=[step(duplicate=True)]), "duplicate_call"),
        (record(lenient=True), "lenient_format"),
    ],
)
def test_reject(rec, reason):
    assert reject_reason(rec) == reason


def test_accept_clean_trajectory():
    assert reject_reason(record(steps=[step(), step()])) is None


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "Tables: t\n\nQuestion: q"},
    {"role": "assistant", "content": "Let me look.", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
     "tool_calls": [{"id": "c1", "function": {"name": "describe_table", "arguments": {"table": "t"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "id INTEGER\nSECRET_OBSERVATION"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {"name": "submit", "arguments": {"query": "SELECT id FROM t"}}}]},
]


def test_mask_covers_only_assistant_text(tok):
    ex = encode(tok, MESSAGES, max_len=4096)
    ids, labels = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labels)
    on = tok.decode([t for t, l in zip(ids, labels) if l != IGNORE])
    off = tok.decode([t for t, l in zip(ids, labels) if l == IGNORE])
    assert "Let me look." in on and "SELECT id FROM t" in on and "<tool_call>" in on
    assert on.count("<|im_end|>") == 2 and "<|im_start|>" not in on
    assert "SECRET_OBSERVATION" not in on and "SECRET_OBSERVATION" in off
    assert "sys" in off and "Question: q" in off and "describe_table" in off  # tool spec + user turn are context


def test_mask_boundaries(tok):
    ex = encode(tok, MESSAGES, max_len=4096)
    ids, labels = ex["input_ids"], ex["labels"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    # every <|im_end|> that closes an assistant turn is learned, the newline after it is not
    ends = [i for i, t in enumerate(ids) if t == im_end and labels[i] != IGNORE]
    assert len(ends) == 2 and all(labels[i + 1] == IGNORE for i in ends)
    # the generation prompt "<|im_start|>assistant\n" before each turn is context
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    assert all(labels[i] == IGNORE for i, t in enumerate(ids) if t == im_start)


def test_too_long_is_dropped(tok):
    assert encode(tok, MESSAGES, max_len=50) is None
