"""Is the held-out curve a property of the model, or of the 25 tasks that happened to be held out?

    python scripts/val_bootstrap.py --base models/Qwen3-8B --run ckpt/sft_v2 \
        --data data/sft/teacher_v2.jsonl

Scores every saved checkpoint on the held-out set one example at a time, then resamples
*tasks* (not examples) to put an interval on each epoch's loss and on the epoch-to-epoch
change. The split was one arbitrary draw — a tenth of tasks at seed 0, unstratified — so a
conclusion like "it overfits after epoch 2" is only worth anything if it survives redrawing
that split.

Resampling is by task because trajectories of one task are not independent: four rollouts
of the same telecom fault share a solution, and treating them as four samples would
understate the interval by about half.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
from collections import defaultdict

import torch

sys.path.insert(0, "/root/probesql")

from sft.data import IGNORE, as_tools, encode  # noqa: E402


def held_out(raw: list[dict], val_frac: float, seed: int = 0) -> set[str]:
    tasks = sorted({r["task_id"] for r in raw})
    random.Random(seed).shuffle(tasks)
    return set(tasks[: max(1, round(len(tasks) * val_frac))])


def per_example_loss(model, tok, rows: list[dict], max_len: int) -> list[tuple[str, float]]:
    out = []
    with torch.no_grad():
        for r in rows:
            ex = encode(tok, r["messages"], as_tools(r["tools"]), max_len)
            if ex is None:
                continue
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            labels = torch.tensor([ex["labels"]], device="cuda")
            loss = model(ids, labels=labels).loss.item()
            out.append((r["task_id"], loss))
    return out


def boot(pairs: list[tuple[str, float]], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Mean and its standard error, resampling tasks rather than examples."""
    by_task = defaultdict(list)
    for tid, loss in pairs:
        by_task[tid].append(loss)
    tasks = list(by_task)
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        drawn = [rng.choice(tasks) for _ in tasks]
        vals = [v for t in drawn for v in by_task[t]]
        means.append(statistics.mean(vals))
    return statistics.mean(means), statistics.stdev(means)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=32768)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    with open(args.data) as f:
        raw = [json.loads(line) for line in f]
    val = held_out(raw, args.val_frac)
    rows = [r for r in raw if r["task_id"] in val]
    print(f"held-out: {len(rows)} examples over {len(val)} tasks\n")

    ckpts = sorted(glob.glob(os.path.join(args.run, "checkpoint-*")),
                   key=lambda p: int(p.split("checkpoint-")[1]))
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    base.eval()

    results = {}
    for ck in ckpts:
        step = int(ck.split("checkpoint-")[1])
        model = PeftModel.from_pretrained(base, ck)
        model.eval()
        pairs = per_example_loss(model, tok, rows, args.max_len)
        mean, se = boot(pairs)
        results[step] = (pairs, mean, se)
        print(f"  step {step:>4}: held-out loss {mean:.4f} +/- {se:.4f}  (n={len(pairs)})")
        model.unload()

    steps = sorted(results)
    print("\nepoch-to-epoch change, bootstrapped over the same task draws")
    print(f"  {'from':>6}{'to':>6}{'Δ loss':>10}{'95% CI':>22}{'':>6}")
    rng = random.Random(1)
    for a, b in zip(steps, steps[1:]):
        pa, pb = results[a][0], results[b][0]
        by_a, by_b = defaultdict(list), defaultdict(list)
        for t, v in pa:
            by_a[t].append(v)
        for t, v in pb:
            by_b[t].append(v)
        tasks = [t for t in by_a if t in by_b]
        deltas = []
        for _ in range(4000):
            drawn = [rng.choice(tasks) for _ in tasks]
            ma = statistics.mean(v for t in drawn for v in by_a[t])
            mb = statistics.mean(v for t in drawn for v in by_b[t])
            deltas.append(mb - ma)
        d, sd = statistics.mean(deltas), statistics.stdev(deltas)
        lo, hi = d - 1.96 * sd, d + 1.96 * sd
        verdict = "improves" if hi < 0 else "worsens" if lo > 0 else "indistinguishable"
        print(f"  {a:>6}{b:>6}{d:>10.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]   {verdict}")


if __name__ == "__main__":
    main()
