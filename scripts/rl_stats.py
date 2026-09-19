"""Recompute the frozen-protocol table and the paired tests from the saved eval runs.

Everything in docs/results.md comes out of this file, so a number in the write-up can be
re-derived rather than trusted. Resampling is over *tasks*, never over rollouts: the four
rollouts of one telecom fault share a solution and treating them as four samples would
understate every interval.

    python scripts/rl_stats.py
"""
from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from math import comb

ARMS = [
    ("arm1 SFT",  "outputs/eval_arm1.json"),
    ("arm2b s15", "outputs/eval_arm2b_s15.json"),
    ("arm2b s25", "outputs/eval_arm2b_s25.json"),
    ("arm3  s15", "outputs/eval_arm3_s15.json"),
    ("arm3  s25", "outputs/eval_arm3_s25.json"),
    ("arm4  s15", "outputs/eval_arm4_s15.json"),
    ("arm4  s25", "outputs/eval_arm4_s25.json"),
]


def load(path: str):
    d = json.load(open(f"{path}/results.json"))
    per_task = defaultdict(list)
    for s in d["simulations"]:
        reward = (s.get("reward_info") or {}).get("reward", 0.0)
        calls = [
            (tc["name"], json.dumps(tc.get("arguments"), sort_keys=True))
            for m in s["messages"]
            if m["role"] == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        ]
        per_task[s["task_id"]].append({
            "solved": reward >= 1.0,
            "turns": sum(1 for m in s["messages"] if m["role"] == "assistant"),
            "calls": len(calls),
            "dups": len(calls) - len(set(calls)),
            "term": s.get("termination_reason"),
        })
    return per_task


def pass_hat(runs, k):
    """tau2's pass^k: the chance that all k of k drawn rollouts pass."""
    n, c = len(runs), sum(r["solved"] for r in runs)
    return comb(c, k) / comb(n, k) if n >= k else float("nan")


def boot(a: list[float], b: list[float], n=10000, seed=0):
    """Paired bootstrap over tasks on mean(b) - mean(a)."""
    rng = random.Random(seed)
    idx = range(len(a))
    diffs = []
    for _ in range(n):
        s = [rng.randrange(len(a)) for _ in idx]
        diffs.append(statistics.mean(b[i] for i in s) - statistics.mean(a[i] for i in s))
    diffs.sort()
    point = statistics.mean(b) - statistics.mean(a)
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n)]
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / n
    return point, lo, hi, min(1.0, p)


data = {name: load(path) for name, path in ARMS}
tasks = sorted(set.intersection(*(set(d) for d in data.values())))
print(f"{len(tasks)} tasks shared by all arms, {len(data['arm1 SFT'][tasks[0]])} rollouts each\n")

print(f"{'':<11} {'p^1':>7} {'p^2':>7} {'p^3':>7} {'p^4':>7} {'all_fail':>9} "
      f"{'turns':>7} {'calls':>7} {'dups':>6} {'errors':>7}")
for name, _ in ARMS:
    d = data[name]
    row = [statistics.mean(pass_hat(d[t], k) for t in tasks) for k in (1, 2, 3, 4)]
    allfail = statistics.mean(not any(r["solved"] for r in d[t]) for t in tasks)
    runs = [r for t in tasks for r in d[t]]
    ok = [r for r in runs if r["solved"]]
    err = statistics.mean(r["term"] not in ("user_stop", "agent_stop") for r in runs)
    print(f"{name:<11} " + " ".join(f"{v:>6.1%}" for v in row) +
          f" {allfail:>9.1%} {statistics.mean(r['turns'] for r in ok):>7.1f} "
          f"{statistics.mean(r['calls'] for r in ok):>7.1f} "
          f"{statistics.mean(r['dups'] for r in ok):>6.2f} {err:>7.1%}")

print("\npaired bootstrap on per-task pass^1, 10,000 resamples over tasks")
pairs = [("arm1 SFT", "arm2b s25"), ("arm1 SFT", "arm3  s25"), ("arm1 SFT", "arm4  s25"),
         ("arm2b s15", "arm3  s15"), ("arm2b s25", "arm3  s25"),
         ("arm3  s15", "arm4  s15"), ("arm3  s25", "arm4  s25"),
         ("arm2b s15", "arm4  s15"), ("arm2b s25", "arm4  s25"),
         ("arm4  s15", "arm4  s25")]
for a, b in pairs:
    va = [pass_hat(data[a][t], 1) for t in tasks]
    vb = [pass_hat(data[b][t], 1) for t in tasks]
    pt, lo, hi, p = boot(va, vb)
    win = sum(x < y for x, y in zip(va, vb))
    loss = sum(x > y for x, y in zip(va, vb))
    star = "*" if p < 0.05 else " "
    print(f"  {b} - {a:<11} {pt:>+7.1%}  95% CI [{lo:>+6.1%},{hi:>+6.1%}]  "
          f"p={p:<7.4f}{star} better/worse/tied {win}/{loss}/{len(tasks)-win-loss}")

print("\nefficiency, paired over tasks both arms solved at least once, successful rollouts only")
for a, b in [("arm2b s15", "arm3  s15"), ("arm2b s25", "arm3  s25"),
             ("arm3  s15", "arm4  s15"), ("arm3  s25", "arm4  s25")]:
    both = [t for t in tasks
            if any(r["solved"] for r in data[a][t]) and any(r["solved"] for r in data[b][t])]
    print(f"  {b} vs {a}   ({len(both)} tasks)")
    for key, label in (("turns", "turns"), ("calls", "tool calls"), ("dups", "duplicate calls")):
        va = [statistics.mean(r[key] for r in data[a][t] if r["solved"]) for t in both]
        vb = [statistics.mean(r[key] for r in data[b][t] if r["solved"]) for t in both]
        pt, lo, hi, p = boot(va, vb)
        rel = pt / statistics.mean(va) if statistics.mean(va) else float("nan")
        star = "*" if p < 0.05 else " "
        print(f"      {label:<16} {statistics.mean(va):>6.2f} -> {statistics.mean(vb):>6.2f}  "
              f"{rel:>+6.1%}  95% CI [{lo:>+6.2f},{hi:>+6.2f}]  p={p:.4f}{star}")
