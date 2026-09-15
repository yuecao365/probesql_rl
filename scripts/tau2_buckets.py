"""GO/NO-GO table for the tau2 pilot: does a 7B produce a usable GRPO gradient here?

    python scripts/tau2_buckets.py outputs/tau2_pilot_telecom.json

Mirrors eval/metrics.py: the question-level buckets are the gradient diagnostic,
since only the mixed bucket yields a non-zero group advantage. Everything else
is context for reading a borderline result -- in particular the termination mix,
because `agent_error` / `too_many_errors` mean the policy cannot speak the action
protocol, which is an SFT problem rather than a reason to abandon the domain.

Thresholds come from docs/pilot_tau2.md and were frozen before any run.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

GO_MIXED = 0.20
NOGO_ALL_FAIL = 0.70
FORMAT_FAILURES = ("agent_error", "too_many_errors")


def load(path: str) -> list[dict]:
    """Accept either the monolithic json or the directory format tau2 may write."""
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    with open(path) as f:
        return json.load(f)["simulations"]


def buckets(sims: list[dict]) -> dict[str, list[str]]:
    per_task = defaultdict(list)
    for s in sims:
        reward = (s.get("reward_info") or {}).get("reward")
        per_task[s["task_id"]].append(bool(reward and reward >= 1.0))
    out = {"all_fail": [], "mixed": [], "all_pass": []}
    for tid, rs in per_task.items():
        out["all_pass" if all(rs) else "all_fail" if not any(rs) else "mixed"].append(tid)
    return out


def summarize(sims: list[dict]) -> dict:
    b = buckets(sims)
    n_task = sum(len(v) for v in b.values())
    rewards = [(s.get("reward_info") or {}).get("reward") or 0.0 for s in sims]
    turns, calls = [], []
    for s in sims:
        msgs = s.get("messages") or []
        turns.append(sum(1 for m in msgs if m.get("role") == "assistant"))
        calls.append(sum(len(m.get("tool_calls") or []) for m in msgs))
    term = Counter(s.get("termination_reason") for s in sims)
    return {
        "tasks": n_task,
        "rollouts": len(sims),
        "pass@1": sum(r >= 1.0 for r in rewards) / max(len(sims), 1),
        "pass^G": len(b["all_pass"]) / max(n_task, 1),
        "all_fail": len(b["all_fail"]) / max(n_task, 1),
        "mixed": len(b["mixed"]) / max(n_task, 1),
        "all_pass": len(b["all_pass"]) / max(n_task, 1),
        "avg_turns": sum(turns) / max(len(turns), 1),
        "avg_calls": sum(calls) / max(len(calls), 1),
        "format_failure": sum(term[k] for k in FORMAT_FAILURES) / max(len(sims), 1),
        "termination": dict(term),
    }


def verdict(s: dict) -> str:
    if s["mixed"] > GO_MIXED:
        return "GO"
    if s["all_fail"] > NOGO_ALL_FAIL:
        return "NO-GO"
    return "WEAK GO (proceed only if format_failure is low; otherwise treat as NO-GO)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    s = summarize(load(args.results))

    print(f"{s['tasks']} tasks x {s['rollouts'] / max(s['tasks'], 1):.0f} rollouts = {s['rollouts']}\n")
    print(f"{'pass@1':<16}{s['pass@1']:.1%}")
    print(f"{'pass^G':<16}{s['pass^G']:.1%}\n")
    print(f"{'all_fail':<16}{s['all_fail']:.1%}   (NO-GO above {NOGO_ALL_FAIL:.0%})")
    print(f"{'mixed':<16}{s['mixed']:.1%}   (GO above {GO_MIXED:.0%})")
    print(f"{'all_pass':<16}{s['all_pass']:.1%}\n")
    print(f"{'avg turns':<16}{s['avg_turns']:.1f}")
    print(f"{'avg calls':<16}{s['avg_calls']:.1f}")
    print(f"{'format failure':<16}{s['format_failure']:.1%}")
    print(f"{'termination':<16}{s['termination']}\n")
    print(f"VERDICT: {verdict(s)}")


if __name__ == "__main__":
    main()
