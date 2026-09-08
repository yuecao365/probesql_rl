"""Cheap syntactic guards against the ways a policy can game the environment.

Three things are detected: degenerate queries that return nothing on purpose
(`WHERE 1=0`, `LIMIT 0`, constant SELECTs), repeated calls that differ only in
formatting, and idle streaks where turns stop producing new information. They
double as safety limits and as process-reward inputs, so every verdict is a
plain value the reward code can weight.

Detection is regex-level by design. A real SQL parser would catch more shapes,
but the goal is to make the obvious hacks unprofitable, not to prove properties
of SQL; anything subtler shows up in the result-set overlap instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_PUNCT_SPACE = re.compile(r"\s*([=<>!,()])\s*")
_EMPTY_FILTER = re.compile(r"\bwhere\s+(0|false|1\s*=\s*0|1\s*<>\s*1|1\s*!=\s*1)\b")
_LIMIT_ZERO = re.compile(r"\blimit\s+0\b")
_FROM = re.compile(r"\bfrom\b")


def normalize(sql: str) -> str:
    """Canonical form for duplicate detection: case, whitespace, comments, `;`."""
    sql = _COMMENT.sub(" ", sql).lower()
    sql = " ".join(sql.split())
    sql = _PUNCT_SPACE.sub(r"\1", sql)
    return sql.rstrip("; ")


def is_read_query(sql: str) -> bool:
    """Only SELECT/WITH may reach run_sql; PRAGMA, EXPLAIN etc. are tool-level errors."""
    return normalize(sql).split(" ", 1)[0] in ("select", "with")


def is_degenerate(sql: str) -> bool:
    """A query written to succeed without answering anything."""
    n = normalize(sql)
    return bool(_EMPTY_FILTER.search(n) or _LIMIT_ZERO.search(n)) or not _FROM.search(n)


def _call_key(tool: str, args: dict) -> str:
    if "query" in args:
        return f"{tool}:{normalize(str(args['query']))}"
    return f"{tool}:{json.dumps(args, sort_keys=True)}"


@dataclass
class Tracker:
    """Per-episode memory of tool calls for duplicate and idle detection."""

    seen: set[str] = field(default_factory=set)
    idle_streak: int = 0

    def record(self, tool: str, args: dict, ok: bool) -> bool:
        """Log one call; returns whether it duplicated an earlier one.

        A turn is informative only if it is new and succeeded, so failures and
        repeats both extend the idle streak.
        """
        key = _call_key(tool, args)
        duplicate = key in self.seen
        self.seen.add(key)
        self.idle_streak = 0 if ok and not duplicate else self.idle_streak + 1
        return duplicate
