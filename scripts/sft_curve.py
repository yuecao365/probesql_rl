"""Plot the SFT loss curve from a run's checkpoints, for the D4 epoch scan.

    python scripts/sft_curve.py ckpt/sft_v1                    # terminal
    python scripts/sft_curve.py ckpt/sft_v1 --png docs/sft_v1_loss.png

Reads `log_history` out of the newest `trainer_state.json` under the run directory. The
Trainer's own stdout is unusable for this: transformers 5 overwrites the line with a tqdm
bar, so the logged losses never survive in the log file even at logging_steps=5.

Both curves are drawn. Training loss falls whether the model is learning the behaviour or
memorising the tasks, so on its own it says nothing about when to stop; the point where the
held-out loss turns up while the training loss keeps falling is the answer. The gap between
them is printed per epoch for the same reason.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def eval_history(run_dir: str) -> list[dict]:
    """Held-out loss, one point per epoch. The training curve alone cannot say when to stop."""
    import glob as _glob
    states = _glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    if not states:
        return []
    newest = max(states, key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
    with open(newest) as f:
        return [r for r in json.load(f)["log_history"] if "eval_loss" in r]


def history(run_dir: str) -> list[dict]:
    states = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    if not states:
        raise SystemExit(f"no trainer_state.json under {run_dir} yet")
    newest = max(states, key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
    with open(newest) as f:
        return [r for r in json.load(f)["log_history"] if "loss" in r]


def sparkline(rows: list[dict], key: str, height: int = 12, width: int = 68) -> str:
    """A terminal plot; matplotlib is not worth a GPU box's dependencies for this."""
    vals = [r[key] for r in rows]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    # resample to the target width
    cols = [vals[round(i * (len(vals) - 1) / (width - 1))] for i in range(min(width, len(vals)))] if len(vals) > 1 else vals
    grid = [[" "] * len(cols) for _ in range(height)]
    for x, v in enumerate(cols):
        y = height - 1 - round((v - lo) / span * (height - 1))
        grid[y][x] = "*"
    out = []
    for i, row in enumerate(grid):
        label = f"{hi - i * span / (height - 1):7.3f}"
        out.append(f"{label} |{''.join(row)}")
    out.append(f"{'':7} +{'-' * len(cols)}")
    out.append(f"{'':8}step {rows[0]['step']}{' ' * max(0, len(cols) - 18)}step {rows[-1]['step']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--png", help="also write a matplotlib figure here")
    args = ap.parse_args()

    rows = history(args.run_dir)
    print(f"{len(rows)} logged points, steps {rows[0]['step']}-{rows[-1]['step']}, "
          f"epochs {rows[0]['epoch']:.2f}-{rows[-1]['epoch']:.2f}\n")
    print("loss")
    print(sparkline(rows, "loss"))
    print("\ngrad_norm")
    print(sparkline(rows, "grad_norm", height=8))

    ev = eval_history(args.run_dir)
    if ev:
        print("\nheld-out loss per epoch  (the turn here is where to stop)")
        print(f"  {'epoch':>6}{'step':>7}{'eval_loss':>12}{'train_loss':>12}{'gap':>9}")
        for r in ev:
            near = min(rows, key=lambda t: abs(t["step"] - r["step"]))
            gap = r["eval_loss"] - near["loss"]
            print(f"  {r['epoch']:>6.0f}{r['step']:>7}{r['eval_loss']:>12.4f}{near['loss']:>12.4f}{gap:>9.4f}")
    else:
        print("\n(no held-out loss: this run had no eval_dataset)")

    print(f"\n{'step':>6}{'epoch':>7}{'loss':>9}{'grad_norm':>11}{'lr':>11}")
    for r in rows:
        print(f"{r['step']:>6}{r['epoch']:>7.2f}{r['loss']:>9.4f}{r['grad_norm']:>11.3f}{r['learning_rate']:>11.2e}")

    # per-epoch means: the number D4 compares against each checkpoint's protocol error rate
    by_epoch: dict[int, list[float]] = {}
    for r in rows:
        by_epoch.setdefault(int(r["epoch"]) + 1, []).append(r["loss"])
    print("\nmean loss per epoch")
    for ep, losses in sorted(by_epoch.items()):
        print(f"  epoch {ep}: {sum(losses) / len(losses):.4f}  ({len(losses)} points)")

    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot([r["step"] for r in rows], [r["loss"] for r in rows], marker="o", ms=3, label="train")
        if ev:
            ax[0].plot([r["step"] for r in ev], [r["eval_loss"] for r in ev],
                       marker="s", ms=6, color="tab:red", label="held-out")
            ax[0].legend()
        ax[0].set_ylabel("loss")
        ax[1].plot([r["step"] for r in rows], [r["grad_norm"] for r in rows], marker="o", ms=3, color="tab:orange")
        ax[1].set_ylabel("grad norm")
        ax[1].set_xlabel("optimizer step")
        for a in ax:
            a.grid(alpha=0.3)
            for ep in sorted({int(r["epoch"]) for r in rows if r["epoch"] >= 1}):
                boundary = next((r["step"] for r in rows if r["epoch"] >= ep), None)
                if boundary:
                    a.axvline(boundary, color="grey", ls="--", lw=0.8)
        fig.suptitle(os.path.basename(args.run_dir.rstrip("/")))
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
        fig.savefig(args.png, dpi=120)
        print(f"\nwrote {args.png}")


if __name__ == "__main__":
    main()
