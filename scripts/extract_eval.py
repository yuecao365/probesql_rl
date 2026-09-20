"""Distil each frozen-protocol evaluation into a per-rollout CSV.

`outputs/` is 1.1 GB of full transcripts and cannot go in the repository, but every number in
docs/results.md is computed from four columns per rollout. Exporting those makes the paired
tests reproducible from the repo alone, without shipping the conversations.

    python scripts/extract_eval.py
"""
from __future__ import annotations

import csv
import json
import pathlib

out_dir = pathlib.Path("docs/assets/eval")
out_dir.mkdir(parents=True, exist_ok=True)
total = 0
for d in sorted(pathlib.Path("outputs").glob("eval_*.json")):
    src = d / "results.json"
    if not src.exists() or ".bak" in d.name:
        continue
    name = d.name[len("eval_"):-len(".json")]
    data = json.loads(src.read_text())
    rows = []
    for s in data["simulations"]:
        msgs = [m for m in s["messages"] if m["role"] == "assistant"]
        tool = [m for m in msgs if m.get("tool_calls")]
        calls = [(tc["name"], json.dumps(tc.get("arguments"), sort_keys=True))
                 for m in tool for tc in m["tool_calls"]]
        rows.append({
            "task_id": s["task_id"],
            "trial": s.get("trial"),
            "solved": int((s.get("reward_info") or {}).get("reward", 0.0) >= 1.0),
            "turns": len(msgs),
            "tool_turns": len(tool),
            "msg_turns": len(msgs) - len(tool),
            "calls": len(calls),
            "distinct_calls": len(set(calls)),
            "termination": s.get("termination_reason"),
        })
    dest = out_dir / f"{name}.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    total += len(rows)
    print(f"  {name}: {len(rows)} rollouts -> {dest}")
print(f"{total} rollouts total")
