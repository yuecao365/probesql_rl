"""Distil each run's training log into a small CSV.

The raw logs are 7.6 GB, mostly Ray's interleaved worker output, and cannot go in the repo.
What a reader actually needs is the per-step scalar row verl prints at the end of every step,
which is a few kilobytes per run. This pulls those out so the training curves stay
reproducible from the repository alone.

    python scripts/extract_metrics.py
"""
from __future__ import annotations

import csv
import pathlib
import re

RUNS = {
    "arm2_lr1e-6": "logs/arm2_grpo_0917_1758.log",
    "arm2b": "logs/arm2b_launch.log",
    "arm3": "logs/arm3_launch.log",
    "arm3_resume": "logs/arm3_0918_0344.log",
    "arm3dr": "logs/arm3dr_launch.log",
    "arm4": "logs/arm4_launch.log",
}
KEEP = [
    "training/global_step", "critic/rewards/mean", "critic/score/mean",
    "critic/advantages/max", "critic/advantages/min", "actor/entropy",
    "actor/grad_norm", "actor/ppo_kl", "actor/pg_clipfrac", "actor/lr",
    "response_length/mean", "num_turns/mean", "rollout_corr/kl", "timing_s/step",
]
NUM = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

out_dir = pathlib.Path("docs/assets/metrics")
out_dir.mkdir(parents=True, exist_ok=True)
for name, path in RUNS.items():
    src = pathlib.Path(path)
    if not src.exists():
        print(f"  skip {name}: no {path}")
        continue
    rows = []
    with src.open(errors="ignore") as fh:
        for line in fh:
            if "training/global_step:" not in line:
                continue
            row = {}
            for key in KEEP:
                m = re.search(re.escape(key) + r":(?:np\.\w+\()?" + NUM.pattern, line)
                if m:
                    row[key] = m.group(1)
            if row.get("training/global_step"):
                rows.append(row)
    rows.sort(key=lambda r: int(float(r["training/global_step"])))
    seen, uniq = set(), []
    for r in rows:
        if r["training/global_step"] not in seen:
            seen.add(r["training/global_step"])
            uniq.append(r)
    dest = out_dir / f"{name}.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEEP)
        w.writeheader()
        w.writerows(uniq)
    print(f"  {name}: {len(uniq)} steps -> {dest}")
