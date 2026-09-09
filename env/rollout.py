"""Multi-turn episode loop with a pluggable policy.

The policy is any callable mapping (messages, tool specs) to an assistant
message; the loop owns everything else: dispatching tools, appending
observations, tracking duplicates and idle turns, and scoring each run_sql
against the gold result when one is available. The loop records facts, not
rewards: per-turn overlap, exec success, degeneracy and duplicate flags are
stored on each Step so reward weighting can change without re-running anything.

An episode ends on `submit`, on a reply without a tool call (parse_error), on a
truncated generation, or when the turn budget runs out. Those four statuses
are the anomaly counters the training dashboard needs. A reply may carry
several tool calls; they run in order, each gets its own tool message, and a
`submit` ends the episode as soon as it is reached. A turn is one reply, so
turn-level credit is unaffected by how many probes a reply batches.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from env import compare, db, prompt, verifier
from env.tools import GOLD_TIMEOUT_S, RESULT_ROWS, SPECS, Toolbox, render

# (messages, tool specs) -> assistant message {"role", "content", "tool_calls": [{"id", "function": {"name", "arguments": dict}}]}
# plus "truncated": True when generation hit the length limit. Adapters for an
# API or a local engine convert their native output into this shape.
Policy = Callable[[list[dict], list[dict]], dict]


@dataclass
class Step:
    turn: int  # index of the assistant reply this call came from
    tool: str
    args: dict
    observation: str
    ok: bool
    duplicate: bool
    idle_streak: int
    degenerate: bool = False  # SQL tools only
    overlap: float | None = None  # run_sql/submit with gold only; 0.0 when execution failed


@dataclass
class Trajectory:
    messages: list[dict]
    steps: list[Step] = field(default_factory=list)
    status: str = "max_turns"  # submitted | parse_error | truncated | max_turns
    final_sql: str | None = None
    correct: bool | None = None  # None without gold or without a submission


def run(task: prompt.Task, conn: sqlite3.Connection, policy: Policy, max_turns: int = 10) -> Trajectory:
    box = Toolbox(conn)
    gold_rows = db.execute(conn, task.gold_sql, timeout_s=GOLD_TIMEOUT_S, max_rows=RESULT_ROWS).rows if task.gold_sql else None
    traj = Trajectory(prompt.build_messages(task, box.tables(), max_turns))
    tracker = verifier.Tracker()

    for turn in range(max_turns):
        reply = policy(traj.messages, SPECS)
        traj.messages.append(reply)
        if reply.get("truncated"):
            traj.status = "truncated"
            return traj
        calls = reply.get("tool_calls") or []
        if not calls:
            traj.status = "parse_error"
            return traj

        for call in calls:
            name, args = call["function"]["name"], call["function"]["arguments"]
            if name in ("run_sql", "submit"):
                query = str(args.get("query", ""))
                try:
                    result = box.execute(query)
                    observation, ok, rows = render(result), True, result.rows
                except db.DbError as e:
                    observation, ok, rows = f"Error: {e}", False, []
                step = Step(turn, name, args, observation, ok, tracker.record(name, args, ok), tracker.idle_streak,
                            degenerate=verifier.is_degenerate(query),
                            overlap=compare.overlap(rows, gold_rows) if gold_rows is not None else None)
            else:
                observation = box.call(name, args)
                ok = not observation.startswith("Error")
                step = Step(turn, name, args, observation, ok, tracker.record(name, args, ok), tracker.idle_streak)
            traj.steps.append(step)

            if name == "submit":
                traj.status = "submitted"
                traj.final_sql = query
                if gold_rows is not None:
                    traj.correct = ok and compare.equal(rows, gold_rows)
                return traj
            traj.messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": observation})
    return traj
