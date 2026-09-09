"""Rejection-sample rollout trajectories into an SFT file.

    python scripts/build_sft.py outputs/teacher_train.jsonl --per-question 2 --out data/sft/teacher.jsonl

Keeps trajectories that pass sft.data.reject_reason, at most --per-question
distinct ones per question, preferring the shortest: among equally correct and
clean trajectories the one with fewer turns shows the most purposeful probing,
and that is the behaviour worth imitating. Prints the rejection breakdown,
which doubles as the teacher's pass rate on this environment.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")

from sft.data import reject_reason  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--per-question", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reasons, kept = Counter(), defaultdict(list)
    total = 0
    for path in args.files:
        with open(path) as f:
            for r in map(json.loads, f):
                total += 1
                reason = reject_reason(r) if r["status"] != "gold_error" else "gold_error"
                reasons[reason or "accepted"] += 1
                if reason is None:
                    kept[r["id"]].append(r)

    out_rows = []
    for qid, rows in kept.items():
        shortest_first = sorted(rows, key=lambda r: (len(r["steps"]), r["k"]))
        distinct = list({r["final_sql"]: r for r in reversed(shortest_first)}.values())  # keep the shortest per final SQL
        distinct.sort(key=lambda r: (len(r["steps"]), r["k"]))
        out_rows += [{"id": qid, "k": r["k"], "messages": r["messages"]} for r in distinct[: args.per_question]]
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{total} trajectories: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    print(f"pass rate {reasons['accepted'] / max(total, 1):.3f}; {len(kept)} questions with an accepted trajectory; wrote {len(out_rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
