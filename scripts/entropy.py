"""Token-level entropy of a checkpoint on held-out trajectories.

    python scripts/entropy.py --base models/Qwen3-8B --adapter ckpt/sft_v2/checkpoint-98 \
        --data data/sft/teacher_v2.jsonl

Reports mean entropy over the tokens the model is trained to produce -- the assistant
spans, found by the same mask the trainer uses -- so the number is about the policy's own
output distribution and not about how confident it is when copying a policy document.

Why this and not just the loss: an SFT run that has started memorising its tasks keeps
driving training loss down while its output distribution collapses, and a collapsed policy
has nothing left for GRPO to explore. Entropy is the signal that shows it, and it is the
one number an epoch scan cannot be read without.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

import torch

sys.path.insert(0, "/root/probesql")

from sft.data import IGNORE, as_tools, encode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="omit to measure the base model")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=24, help="trajectories to sample")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=32768)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    with open(args.data) as f:
        raw = [json.loads(line) for line in f]

    # Same split the trainer makes, so this measures held-out behaviour.
    tasks = sorted({r["task_id"] for r in raw})
    random.Random(0).shuffle(tasks)
    val_tasks = set(tasks[: max(1, round(len(tasks) * args.val_frac))])
    held = [r for r in raw if r["task_id"] in val_tasks]
    rows = random.Random(0).sample(held, min(args.n, len(held)))

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    per_traj = []
    with torch.no_grad():
        for r in rows:
            ex = encode(tok, r["messages"], as_tools(r["tools"]), args.max_len)
            if ex is None:
                continue
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            labels = torch.tensor(ex["labels"])
            logits = model(ids).logits[0].float()
            # position t predicts token t+1, so score the positions whose *next* token is learned
            keep = (labels[1:] != IGNORE).nonzero().squeeze(-1)
            if keep.numel() == 0:
                continue
            logp = torch.log_softmax(logits[keep], dim=-1)
            ent = -(logp.exp() * logp).sum(-1)
            per_traj.append(ent.mean().item())

    label = args.adapter or f"{args.base} (base)"
    print(f"{label}")
    print(f"  {len(per_traj)} held-out trajectories")
    print(f"  mean token entropy {statistics.mean(per_traj):.4f} nats")
    print(f"  per-trajectory sd  {statistics.stdev(per_traj):.4f}" if len(per_traj) > 1 else "")
    print(f"  range              {min(per_traj):.4f} - {max(per_traj):.4f}")


if __name__ == "__main__":
    main()
