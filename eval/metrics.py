"""Trajectory-level metrics for baselines, pass@k histograms and ablation tables.

Records are the JSON lines written by scripts/rollout.py: one trajectory per
line with `id`, `k`, `status`, `correct`, `difficulty`, `steps` and `messages`.
Everything here is a pure function of those records, so any run can be
re-summarized without touching a model. Question-level buckets (all-fail,
mixed, all-pass) are the GRPO gradient diagnostic: only the mixed bucket
produces a non-zero advantage.
"""

from __future__ import annotations

from collections import Counter, defaultdict

SQL_TOOLS = ("run_sql", "submit")


def _first_sql_correct(rec) -> bool | None:
    for s in rec["steps"]:
        if s["tool"] in SQL_TOOLS:
            return s["overlap"] == 1.0
    return None


def _tokens(rec) -> int:
    return sum(m.get("usage", {}).get("completion_tokens", 0) for m in rec["messages"])


def buckets(records) -> dict[str, list[str]]:
    """Question ids grouped by how many of their rollouts were correct."""
    per_q = defaultdict(list)
    for r in records:
        per_q[r["id"]].append(bool(r["correct"]))
    out = {"all_fail": [], "mixed": [], "all_pass": []}
    for qid, cs in per_q.items():
        out["all_pass" if all(cs) else "all_fail" if not any(cs) else "mixed"].append(qid)
    return out


def summarize(records) -> dict:
    n = len(records)
    if n == 0:
        return {"trajectories": 0}
    b = buckets(records)
    n_q = sum(len(v) for v in b.values())
    steps = [s for r in records for s in r["steps"]]
    calls = [m for r in records for m in r["messages"] if m.get("tool_calls")]
    first = [(r, _first_sql_correct(r)) for r in records]
    wrong_first = [r for r, f in first if f is False]
    by_diff = defaultdict(list)
    for r in records:
        by_diff[r.get("difficulty") or "all"].append(bool(r["correct"]))
    return {
        "trajectories": n,
        "questions": n_q,
        "pass@1": sum(bool(r["correct"]) for r in records) / n,
        "pass^G": len(b["all_pass"]) / n_q,
        "buckets": {k: len(v) / n_q for k, v in b.items()},
        "status": dict(Counter(r["status"] for r in records)),
        "avg_turns": len(steps) / n,
        "lenient_call_ratio": sum(bool(m.get("lenient")) for m in calls) / max(len(calls), 1),
        "avg_completion_tokens": sum(_tokens(r) for r in records) / n,
        "probe_turn_ratio": sum(s["tool"] not in SQL_TOOLS for s in steps) / max(len(steps), 1),
        "first_sql_acc": sum(f is True for _, f in first) / max(sum(f is not None for _, f in first), 1),
        "recovery_rate": sum(bool(r["correct"]) for r in wrong_first) / max(len(wrong_first), 1),
        "hallucination_rate": sum(any("no such" in s["observation"] for s in r["steps"]) for r in records) / n,
        "duplicate_rate": sum(any(s["duplicate"] for s in r["steps"]) for r in records) / n,
        "degenerate_rate": sum(any(s["degenerate"] for s in r["steps"]) for r in records) / n,
        "idle_rate": sum(any(s["idle_streak"] >= 2 for s in r["steps"]) for r in records) / n,
        "pass@1_by_difficulty": {d: sum(v) / len(v) for d, v in sorted(by_diff.items())},
    }
