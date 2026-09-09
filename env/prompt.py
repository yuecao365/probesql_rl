"""Prompt construction: the one protocol shared by teacher sampling, SFT data
and RL rollouts.

Messages use the OpenAI chat shape (system / user / assistant with tool_calls /
tool) so that an API teacher, a chat-template renderer and an RL framework all
consume the same object. Tool schemas are passed alongside the messages rather
than pasted into the system prompt, because chat templates render them in the
model's own native format. The initial state intentionally omits columns: the
policy must earn the schema through tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM = """You are an expert SQL analyst working on an unfamiliar SQLite database.
You know only the table names. Before writing SQL, explore the schema with the
tools: describe the tables you need, sample real rows to learn value formats,
and run candidate queries to check their results. Every reply must call at
least one tool; independent probes may be batched in one reply. When you are
confident, call `submit` with the final SQL; that ends the task. You have at
most {max_turns} replies in total."""


@dataclass(frozen=True)
class Task:
    db_id: str
    question: str
    evidence: str = ""  # BIRD's external knowledge hint; empty for Spider
    gold_sql: str | None = None  # training only; never shown to the policy


def build_messages(task: Task, tables: list[str], max_turns: int) -> list[dict]:
    user = f"Database: {task.db_id}\nTables: {', '.join(tables)}\n\nQuestion: {task.question}"
    if task.evidence:
        user += f"\nHint: {task.evidence}"
    return [
        {"role": "system", "content": SYSTEM.format(max_turns=max_turns)},
        {"role": "user", "content": user},
    ]

