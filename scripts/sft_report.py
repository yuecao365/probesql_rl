"""Publication-quality figures for an SFT run, for a README or a report.

    python scripts/sft_report.py ckpt/sft_v2 --out docs/assets --tag sft_v2

Writes three figures and a markdown block that embeds them:

  <tag>_loss.png      training vs held-out loss, with epoch boundaries
  <tag>_health.png    gradient norm and learning-rate schedule
  <tag>_data.png      what the training set is made of

Deliberately not a wandb screenshot. A reader of the repo should be able to see what was
trained on and how it behaved without an account, and the figures should carry their own
caveats -- the held-out curve here is drawn with its noise band, because at 86 examples a
gap under ~0.05 is not distinguishable from zero and a figure that hides that invites the
wrong conclusion.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

PALETTE = {"train": "#2563eb", "eval": "#dc2626", "grad": "#ea580c", "lr": "#7c3aed",
           "bar": "#0891b2", "grid": "#e5e7eb", "text": "#374151"}


def load_history(run_dir: str) -> tuple[list[dict], list[dict]]:
    states = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    if not states:
        raise SystemExit(f"no trainer_state.json under {run_dir}")
    newest = max(states, key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
    with open(newest) as f:
        hist = json.load(f)["log_history"]
    return [r for r in hist if "loss" in r], [r for r in hist if "eval_loss" in r]


def noise_floor(train: list[dict], n_val: int, batch: int = 8) -> float:
    """Standard error of an eval_loss, estimated from plateau batch-to-batch spread.

    Each logged point averages `batch * logging_steps` examples, so the per-example spread
    is the logged spread scaled back up; the eval mean over n_val examples then has that
    spread over sqrt(n_val).
    """
    plateau = [r["loss"] for r in train if r["step"] >= train[-1]["step"] * 0.5]
    if len(plateau) < 3:
        return 0.0
    per_example = statistics.stdev(plateau) * (batch * 5) ** 0.5
    return per_example / n_val**0.5


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=PALETTE["text"], pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["text"])
    ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["text"])
    ax.grid(True, color=PALETTE["grid"], lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", default="data/sft/teacher_v2.jsonl")
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-val", type=int, default=86)
    args = ap.parse_args()
    tag = args.tag or os.path.basename(args.run_dir.rstrip("/"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, ev = load_history(args.run_dir)
    os.makedirs(args.out, exist_ok=True)
    se = noise_floor(train, args.n_val)
    epochs = sorted({int(r["epoch"]) for r in train if r["epoch"] >= 1})
    bounds = [next(r["step"] for r in train if r["epoch"] >= e) for e in epochs]

    # ---------------------------------------------------------------- loss
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["step"] for r in train], [r["loss"] for r in train],
            color=PALETTE["train"], lw=1.8, label="training")
    if ev:
        xs = [r["step"] for r in ev]
        ys = [r["eval_loss"] for r in ev]
        if se:
            ax.fill_between(xs, [y - 1.96 * se for y in ys], [y + 1.96 * se for y in ys],
                            color=PALETTE["eval"], alpha=0.15, lw=0)
        ax.plot(xs, ys, "s--", color=PALETTE["eval"], ms=7, lw=1.6, label="held-out (25 tasks)")
    for b in bounds:
        ax.axvline(b, color=PALETTE["grid"], ls="--", lw=1)
    for e, b in zip(epochs, bounds):
        ax.text(b, ax.get_ylim()[1], f" epoch {e}", va="top", fontsize=8, color=PALETTE["text"])
    style(ax, f"{tag} — loss", "optimizer step", "loss")
    ax.legend(frameon=False, fontsize=9)
    if se:
        ax.text(0.98, 0.95, f"shaded: ±1.96 SE ≈ ±{1.96 * se:.3f}\ngaps below that are not resolvable",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#6b7280")
    fig.tight_layout()
    fig.savefig(f"{args.out}/{tag}_loss.png", dpi=150)
    print(f"wrote {args.out}/{tag}_loss.png")

    # ---------------------------------------------------------------- health
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([r["step"] for r in train], [r["grad_norm"] for r in train], color=PALETTE["grad"], lw=1.6)
    style(ax[0], "gradient norm", "optimizer step", "‖g‖")
    ax[1].plot([r["step"] for r in train], [r["learning_rate"] for r in train], color=PALETTE["lr"], lw=1.6)
    style(ax[1], "learning rate (cosine)", "optimizer step", "lr")
    for a in ax:
        for b in bounds:
            a.axvline(b, color=PALETTE["grid"], ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(f"{args.out}/{tag}_health.png", dpi=150)
    print(f"wrote {args.out}/{tag}_health.png")

    # ---------------------------------------------------------------- data
    if os.path.exists(args.data):
        with open(args.data) as f:
            rows = [json.loads(line) for line in f]
        turns = [r["agent_turns"] for r in rows]
        issue: dict[str, int] = {}
        for r in rows:
            k = r["task_id"].split("]")[0].strip("[")
            issue[k] = issue.get(k, 0) + 1

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].hist(turns, bins=range(min(turns), max(turns) + 2), color=PALETTE["bar"], alpha=0.85)
        ax[0].axvline(statistics.median(turns), color=PALETTE["eval"], ls="--", lw=1.5,
                      label=f"median {statistics.median(turns):.0f}")
        style(ax[0], "teacher trajectory length", "assistant turns", "trajectories")
        ax[0].legend(frameon=False, fontsize=9)

        names = sorted(issue, key=issue.get, reverse=True)
        ax[1].barh([n.replace("_", " ") for n in names], [issue[n] for n in names],
                   color=PALETTE["bar"], alpha=0.85)
        style(ax[1], "issue type", "trajectories", "")
        ax[1].invert_yaxis()
        fig.suptitle(f"{len(rows)} trajectories over {len({r['task_id'] for r in rows})} tasks",
                     fontsize=10, color=PALETTE["text"])
        fig.tight_layout()
        fig.savefig(f"{args.out}/{tag}_data.png", dpi=150)
        print(f"wrote {args.out}/{tag}_data.png")

    # ---------------------------------------------------------------- markdown
    md = [f"### SFT ({tag})", "",
          f"![loss](assets/{tag}_loss.png)", "",
          f"![health](assets/{tag}_health.png)", "",
          f"![data](assets/{tag}_data.png)", ""]
    if ev:
        md += ["| epoch | step | training loss | held-out loss | gap |", "|---|---|---|---|---|"]
        for r in ev:
            near = min(train, key=lambda t: abs(t["step"] - r["step"]))
            md.append(f"| {r['epoch']:.0f} | {r['step']} | {near['loss']:.4f} | "
                      f"{r['eval_loss']:.4f} | {r['eval_loss'] - near['loss']:+.4f} |")
        md += ["", f"Held-out loss is measured on {args.n_val} examples from tasks absent from "
                   f"training; its standard error is about {se:.3f}, so a gap below "
                   f"{2.8 * se:.3f} is not distinguishable from zero."]
    path = f"{args.out}/{tag}_report.md"
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
