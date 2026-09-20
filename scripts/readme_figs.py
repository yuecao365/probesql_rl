"""The three figures a reader of the README needs, from the frozen evaluation runs.

    /root/autodl-tmp/tau2-bench/.venv/bin/python scripts/readme_figs.py

  ladder.png      what each stage bought, pass^1 and pass^4 with bootstrap intervals
  efficiency.png  the paired trajectory-efficiency comparisons, which is where the shaped
                  reward and the credit assignment actually differ
  coverage.png    gain against the share of the training pool each issue type occupies,
                  which is the evidence that the remaining headroom is data, not algorithm

Intervals are over tasks, never over rollouts: four rollouts of one telecom fault share a
solution and treating them as four samples would understate every interval by about half.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import defaultdict
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/assets"
TAU2 = "/root/autodl-tmp/tau2-bench/data/tau2/domains/telecom/"
INK, ACC, WARN = "#22303f", "#2f6f9f", "#b5533c"


def load(name):
    d = json.load(open(f"outputs/eval_{name}.json/results.json"))
    per = defaultdict(list)
    for s in d["simulations"]:
        msgs = [m for m in s["messages"] if m["role"] == "assistant"]
        calls = [(tc["name"], json.dumps(tc.get("arguments"), sort_keys=True))
                 for m in msgs if m.get("tool_calls") for tc in m["tool_calls"]]
        per[s["task_id"]].append({
            "solved": (s.get("reward_info") or {}).get("reward", 0.0) >= 1.0,
            "turns": len(msgs), "calls": len(calls), "dups": len(calls) - len(set(calls))})
    return per


def phat(runs, k):
    n, c = len(runs), sum(r["solved"] for r in runs)
    return comb(c, k) / comb(n, k) if n >= k else float("nan")


def ci(vals, n=10000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(vals[rng.randrange(len(vals))] for _ in vals) for _ in range(n))
    return statistics.mean(vals), means[int(0.025 * n)], means[int(0.975 * n)]


def paired(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        s = [rng.randrange(len(a)) for _ in a]
        ds.append(statistics.mean(b[i] for i in s) - statistics.mean(a[i] for i in s))
    ds.sort()
    p = 2 * min(sum(d <= 0 for d in ds), sum(d >= 0 for d in ds)) / n
    return statistics.mean(b) - statistics.mean(a), ds[int(0.025 * n)], ds[int(0.975 * n)], min(1.0, p)


ARMS = ["arm0", "arm1", "arm2b_s25", "arm3_s25", "arm4_s25"]
LABEL = {"arm0": "Qwen3-8B\nbase", "arm1": "+ SFT", "arm2b_s25": "+ GRPO",
         "arm3_s25": "+ shaped\nreward", "arm4_s25": "+ turn-level\ncredit"}
D = {a: load(a) for a in ARMS}
T = sorted(set.intersection(*(set(d) for d in D.values())))

# ---------------------------------------------------------------- 1. the ladder
fig, ax = plt.subplots(figsize=(7.6, 4.2))
for k, colour, mark in ((1, ACC, "o"), (4, INK, "s")):
    ys, los, his = [], [], []
    for a in ARMS:
        m, lo, hi = ci([phat(D[a][t], k) for t in T])
        ys.append(m * 100); los.append((m - lo) * 100); his.append((hi - m) * 100)
    ax.errorbar(range(len(ARMS)), ys, yerr=[los, his], marker=mark, color=colour,
                capsize=4, lw=2, ms=7, label=f"pass^{k}")
    for i, y in enumerate(ys):
        ax.annotate(f"{y:.1f}", (i, y), textcoords="offset points", xytext=(0, 11 if k == 1 else -18),
                    ha="center", fontsize=9, color=colour, weight="bold")
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=9)
ax.set_ylabel("%  (114 tasks x 4 rollouts)")
ax.set_title("What each stage bought", loc="left", weight="bold")
ax.legend(frameon=False); ax.grid(axis="y", alpha=.25); ax.set_ylim(-4, 100)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/ladder.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 2. efficiency
import os
ALL_COMPS = [("arm2b_s15", "arm3dr_s15", "shaped reward\n(single variable, 15 steps)"),
             ("arm2b_s15", "arm3_s15", "shaped reward + std norm\n(vs GRPO, 15 steps)"),
             ("arm3_s25", "arm4_s25", "turn-level credit\n(vs shaped, 25 steps)")]
COMPS = [c for c in ALL_COMPS
         if all(os.path.exists(f"outputs/eval_{n}.json/results.json") for n in c[:2])]
for a, b, _ in COMPS:
    for n in (a, b):
        D.setdefault(n, load(n))
fig, axes = plt.subplots(1, len(COMPS), figsize=(4.7 * len(COMPS), 4.0), sharey=True)
axes = [axes] if len(COMPS) == 1 else list(axes)
for ax, (a, b, title) in zip(axes, COMPS):
    both = [t for t in T if any(r["solved"] for r in D[a][t]) and any(r["solved"] for r in D[b][t])]
    names, rels, errs, ps = [], [], [], []
    for key, lab in (("turns", "turns"), ("calls", "tool calls"), ("dups", "duplicate calls")):
        va = [statistics.mean(r[key] for r in D[a][t] if r["solved"]) for t in both]
        vb = [statistics.mean(r[key] for r in D[b][t] if r["solved"]) for t in both]
        pt, lo, hi, p = paired(va, vb)
        base = statistics.mean(va)
        names.append(lab); rels.append(pt / base * 100)
        errs.append([(pt - lo) / base * 100, (hi - pt) / base * 100]); ps.append(p)
    y = range(len(names))
    cols = [ACC if r < 0 else WARN for r in rels]
    ax.barh(list(y), rels, color=cols, height=.55,
            xerr=list(zip(*errs)), error_kw=dict(ecolor=INK, lw=1.2, capsize=4))
    for i, (r, p) in enumerate(zip(rels, ps)):
        txt = f"{r:+.1f}%  p={p:.3f}" + ("*" if p < .05 else "")
        if abs(r) > 25:   # a long bar carries its label inside, or it collides with the axis
            ax.text(r / 2, i, txt, va="center", ha="center", fontsize=9, color="white", weight="bold")
        else:
            ax.text(r + (3 if r > 0 else -3), i, txt,
                    va="center", ha="left" if r > 0 else "right", fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(names)
    ax.axvline(0, color=INK, lw=1)
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    ax.set_xlim(-78, 46); ax.grid(axis="x", alpha=.25)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
axes[0].set_xlabel("change in successful trajectories, paired over tasks (%)")
fig.suptitle("Trajectory efficiency: the axis each design was aimed at",
             x=.01, ha="left", weight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/efficiency.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 3. coverage
sp = json.load(open(TAU2 + "split_tasks.json"))
base = set(sp["base"])
pool = {t["id"] for t in json.load(open(TAU2 + "tasks.json"))} - base
issue = lambda i: re.match(r"\[([^\]]+)\]", i).group(1)
share = {k: sum(issue(i) == k for i in pool) / len(pool) for k in {issue(i) for i in base}}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for k in sorted(share, key=lambda k: -share[k]):
    ts = [t for t in T if issue(t) == k]
    va = [phat(D["arm1"][t], 1) for t in ts]
    vb = [phat(D["arm3_s25"][t], 1) for t in ts]
    pt, lo, hi, p = paired(va, vb)
    ax.errorbar(share[k] * 100, pt * 100, yerr=[[(pt - lo) * 100], [(hi - pt) * 100]],
                marker="o", ms=10, color=ACC if p < .05 else WARN, capsize=5, lw=2)
    ax.annotate(f"{k}\nn={len(ts)}, p={p:.3f}", (share[k] * 100, pt * 100),
                textcoords="offset points", xytext=(12, -6), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("share of the RL training pool (%, log)")
ax.set_ylabel("pass^1 gain of RL over SFT (points)")
ax.set_title("Where the RL gain landed, against training-pool coverage", loc="left", weight="bold")
ax.axhline(0, color=INK, lw=1); ax.grid(alpha=.25); ax.set_xlim(.4, 200)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/coverage.png", dpi=160); plt.close(fig)
print("wrote", OUT + "/{ladder,efficiency,coverage}.png")
