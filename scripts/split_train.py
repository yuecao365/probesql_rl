"""Split BIRD train questions into disjoint SFT and RL pools, by question, with a fixed seed.

    python scripts/split_train.py --json data/bird/train/train.json --sft 600 --out splits

SFT questions must never reach RL: a question the student was fine-tuned on
tends to come back all-correct in its GRPO group, which is zero advantage and
wasted rollouts. Splitting by question (not database) is enough because dev
databases are disjoint from train anyway.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, ".")

from env.tasks import load_bird  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--sft", type=int, required=True, help="number of SFT questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = [e.id for e in load_bird(args.json, "")]
    random.Random(args.seed).shuffle(ids)
    os.makedirs(args.out, exist_ok=True)
    for name, chunk in [("sft_ids.txt", ids[: args.sft]), ("rl_ids.txt", ids[args.sft :])]:
        with open(os.path.join(args.out, name), "w") as f:
            f.write("\n".join(chunk) + "\n")
        print(f"{name}: {len(chunk)} questions")


if __name__ == "__main__":
    main()
