"""Report one arm under the frozen protocol in docs/eval_protocol.md.

    python scripts/tau2_buckets.py outputs/eval_arm0.json

The headline pass@1 is the comparable number, not the decisive one: bootstrapped on the
pilot, 114 tasks x 4 rollouts resolves differences of about 6.5 points, and the gains in
play are smaller than that. So this prints the low-variance signals alongside it -- the
protocol error rate above all, since 54% of baseline episodes die calling the user's tools.

It also splits by task type. The benchmark is 17.5% escalate tasks while the RL training
pool is 0.6%, and the two types reward opposite behaviour ("keep trying" vs "recognise this
cannot be fixed"). A policy that improves on the 82% can still lose the headline, so the
split is printed before the total, not after.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict

SPLITS = "/root/autodl-tmp/tau2-bench/data/tau2/domains/telecom/split_tasks.json"
TASKS = "/root/autodl-tmp/tau2-bench/data/tau2/domains/telecom/tasks.json"
GO_MIXED, NOGO_ALL_FAIL = 0.20, 0.70
FORMAT_FAILURES = ("agent_error", "too_many_errors")


def load(path: str) -> list[dict]:
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    with open(path) as f:
        return json.load(f)["simulations"]


def escalate_tasks() -> set[str]:
    """Tasks whose only expected action is handing off to a human."""
    with open(TASKS) as f:
        tasks = json.load(f)
    return {
        t["id"]
        for t in tasks
        if [a["name"] for a in (t["evaluation_criteria"].get("actions") or [])] == ["transfer_to_human_agents"]
    }


def solved(sim: dict) -> bool:
    return ((sim.get("reward_info") or {}).get("reward") or 0.0) >= 1.0


def buckets(sims: list[dict]) -> dict[str, list[str]]:
    per_task = defaultdict(list)
    for s in sims:
        per_task[s["task_id"]].append(solved(s))
    out = {"all_fail": [], "mixed": [], "all_pass": []}
    for tid, rs in per_task.items():
        out["all_pass" if all(rs) else "all_fail" if not any(rs) else "mixed"].append(tid)
    return out


def pass_at_1(sims: list[dict]) -> tuple[float, float, int]:
    """Rate, plus a task-level standard error -- rollouts of one task are not independent."""
    per_task = defaultdict(list)
    for s in sims:
        per_task[s["task_id"]].append(solved(s))
    rates = [sum(v) / len(v) for v in per_task.values()]
    if not rates:
        return 0.0, 0.0, 0
    se = statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else 0.0
    return statistics.mean(rates), se, len(rates)


def protocol_errors(sims: list[dict]) -> tuple[int, int]:
    """Agent tool results that failed because the tool belongs to the user, not the agent."""
    bad = tot = 0
    for s in sims:
        for m in s.get("messages", []):
            if m.get("role") != "tool" or m.get("requestor", "assistant") == "user":
                continue
            tot += 1
            c = m.get("content") or ""
            if m.get("error") and "Tool" in c and "not found" in c:
                bad += 1
    return bad, tot


def report(sims: list[dict]) -> None:
    esc = escalate_tasks()
    rate, se, n_tasks = pass_at_1(sims)
    print(f"{n_tasks} tasks x {len(sims) / max(n_tasks, 1):.0f} rollouts = {len(sims)} episodes\n")

    for label, sel in (("fix-it tasks", lambda s: s["task_id"] not in esc), ("escalate tasks", lambda s: s["task_id"] in esc)):
        group = [s for s in sims if sel(s)]
        if not group:
            continue
        r, e, n = pass_at_1(group)
        print(f"  {label:<18}{r:6.1%} +/- {e:.1%}   ({n} tasks)")
    print(f"  {'OVERALL pass@1':<18}{rate:6.1%} +/- {se:.1%}")
    print(f"  {'':18}{'':6}      resolves ~{2.8 * se:.1%} differences\n")

    b = buckets(sims)
    for k in ("all_fail", "mixed", "all_pass"):
        note = f"   (GO above {GO_MIXED:.0%})" if k == "mixed" else f"   (NO-GO above {NOGO_ALL_FAIL:.0%})" if k == "all_fail" else ""
        print(f"  {k:<18}{len(b[k]) / max(n_tasks, 1):6.1%}{note}")

    bad, tot = protocol_errors(sims)
    term = Counter(s.get("termination_reason") for s in sims)
    turns = [sum(1 for m in s.get("messages", []) if m.get("role") == "assistant") for s in sims]
    calls = [sum(len(m.get("tool_calls") or []) for m in s.get("messages", []) if m.get("role") == "assistant") for s in sims]
    print(f"\n  {'protocol errors':<18}{bad / max(tot, 1):6.1%}   ({bad}/{tot} agent tool results)")
    print(f"  {'killed by errors':<18}{sum(term[k] for k in FORMAT_FAILURES) / len(sims):6.1%}")
    print(f"  {'mean turns':<18}{statistics.mean(turns):6.1f}")
    print(f"  {'mean tool calls':<18}{statistics.mean(calls):6.1f}")
    print(f"  {'termination':<18}{dict(term)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    report(load(args.results))


if __name__ == "__main__":
    main()
