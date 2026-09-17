"""Publication-quality figures for a veRL run, for a README or a report.

    python scripts/rl_report.py tensorboard_log/turncredit-rl/<run> --out docs/assets --tag arm2

Reads the run's tensorboard scalars and writes three figures plus a markdown block:

  <tag>_reward.png   the headline: mean reward per step, with its within-step spread
  <tag>_health.png   entropy, clip fraction, gradient norm, KL -- the four ways a GRPO run
                     goes wrong before the reward curve shows it
  <tag>_rollout.png  what the episodes looked like: response length, turns, and the share of
                     groups that carried any gradient at all

The health panel is not decoration. Entropy collapse is the standard failure mode of GRPO on
long-horizon tasks, and it shows up in the entropy trace well before the reward stalls; the
SFT checkpoint starts at 0.357 nats, which is the line to watch it against. The zero-advantage
share matters for a different reason -- it is the fraction of sampled episodes that cost money
and taught the policy nothing.
"""

from __future__ import annotations

import argparse
import glob
import os

PALETTE = {"a": "#2563eb", "b": "#dc2626", "c": "#ea580c", "d": "#7c3aed",
           "e": "#0891b2", "grid": "#e5e7eb", "text": "#374151"}

# Metric names differ a little across verl versions; take the first that exists.
WANTED = {
    "reward": ["critic/score/mean", "critic/rewards/mean", "reward/mean"],
    "reward_std": ["critic/score/std", "critic/rewards/std"],
    "entropy": ["actor/entropy", "actor/entropy_loss"],
    "clip": ["actor/pg_clipfrac", "actor/clipfrac"],
    "grad": ["actor/grad_norm"],
    "kl": ["actor/kl", "actor/kl_loss", "critic/kl"],
    "length": ["response_length/mean", "tokens/response_length/mean"],
    "turns": ["env/tau2_turns", "tau2_turns", "agent_loop/tau2_turns"],
    "pg_loss": ["actor/pg_loss"],
}


def read_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not files:
        raise SystemExit(f"no tensorboard events under {run_dir}")
    out: dict[str, list[tuple[int, float]]] = {}
    for f in sorted(files):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            out.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    for tag in out:
        out[tag] = sorted(set(out[tag]))
    return out


def pick(scalars, names):
    for n in names:
        if n in scalars and scalars[n]:
            return n, scalars[n]
    return None, None


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=PALETTE["text"], pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["text"])
    ax.set_ylabel(ylabel, fontsize=9, color=PALETTE["text"])
    ax.grid(True, color=PALETTE["grid"], lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["text"], labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--tag", default="arm2")
    ap.add_argument("--sft-entropy", type=float, default=0.357,
                    help="entropy of the SFT checkpoint, drawn as the line to fall from")
    ap.add_argument("--sft-pass1", type=float, default=0.759,
                    help="arm 1 pass@1 on the frozen evaluation set, for context only")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = read_scalars(args.run_dir)
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(s)} scalars: {sorted(s)[:8]}{' ...' if len(s) > 8 else ''}")

    # ------------------------------------------------------------------ reward
    name, series = pick(s, WANTED["reward"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if series:
        xs, ys = zip(*series)
        _, std = pick(s, WANTED["reward_std"])
        if std and len(std) == len(series):
            sd = [v for _, v in std]
            ax.fill_between(xs, [y - v for y, v in zip(ys, sd)], [y + v for y, v in zip(ys, sd)],
                            color=PALETTE["a"], alpha=0.13, lw=0, label="+/- 1 sd within step")
        ax.plot(xs, ys, "o-", color=PALETTE["a"], ms=4, lw=1.8, label=name)
    ax.axhline(args.sft_pass1, color=PALETTE["b"], ls="--", lw=1.2,
               label=f"arm 1 on the frozen eval set ({args.sft_pass1:.3f})")
    style(ax, f"{args.tag} — training reward", "optimizer step", "mean reward")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.99, 0.02,
            "training reward is measured at temperature 1.0 on the training pool;\n"
            "it is not comparable to the frozen evaluation number above",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#6b7280")
    fig.tight_layout()
    fig.savefig(f"{args.out}/{args.tag}_reward.png", dpi=150)
    print(f"wrote {args.out}/{args.tag}_reward.png")

    # ------------------------------------------------------------------ health
    panels = [("entropy", "entropy", "nats", PALETTE["c"]),
              ("clip", "clip fraction", "share", PALETTE["d"]),
              ("grad", "gradient norm", "||g||", PALETTE["e"]),
              ("pg_loss", "policy-gradient loss", "loss", PALETTE["a"])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, (key, title, unit, colour) in zip(axes, panels):
        nm, ser = pick(s, WANTED[key])
        if ser:
            xs, ys = zip(*ser)
            ax.plot(xs, ys, color=colour, lw=1.6)
        else:
            ax.text(0.5, 0.5, "not logged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#9ca3af")
        if key == "entropy" and args.sft_entropy:
            ax.axhline(args.sft_entropy, color=PALETTE["b"], ls="--", lw=1.1)
            ax.text(0.02, args.sft_entropy, " SFT start", va="bottom", fontsize=7,
                    color=PALETTE["b"], transform=ax.get_yaxis_transform())
        style(ax, title, "step", unit)
    fig.tight_layout()
    fig.savefig(f"{args.out}/{args.tag}_health.png", dpi=150)
    print(f"wrote {args.out}/{args.tag}_health.png")

    # ----------------------------------------------------------------- rollout
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title, unit in ((axes[0], "length", "response length", "tokens"),
                                 (axes[1], "turns", "assistant turns per episode", "turns")):
        nm, ser = pick(s, WANTED[key])
        if ser:
            xs, ys = zip(*ser)
            ax.plot(xs, ys, color=PALETTE["e"], lw=1.6)
        else:
            ax.text(0.5, 0.5, "not logged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#9ca3af")
        style(ax, title, "step", unit)
    fig.tight_layout()
    fig.savefig(f"{args.out}/{args.tag}_rollout.png", dpi=150)
    print(f"wrote {args.out}/{args.tag}_rollout.png")

    # ---------------------------------------------------------------- markdown
    md = [f"### RL ({args.tag})", "",
          f"![reward](assets/{args.tag}_reward.png)", "",
          f"![health](assets/{args.tag}_health.png)", "",
          f"![rollout](assets/{args.tag}_rollout.png)", ""]
    if series:
        md += ["| step | mean reward |", "|---|---|"]
        md += [f"| {x} | {y:.4f} |" for x, y in series]
    path = f"{args.out}/{args.tag}_report.md"
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
