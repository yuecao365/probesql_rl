"""Dump questions from one pass@k bucket for manual label auditing (plan D3).

    python scripts/audit_sample.py outputs/m1_coder7b_train200_g8.jsonl --json data/bird/train/train.json \
        --db-dir data/bird/train/train_databases --bucket all_fail --n 100 --out outputs/d3_audit.md

Each entry shows the question, evidence, gold SQL, the gold result's first rows
and the distinct SQL the model submitted, so a reviewer can label it as
"model failure" or "gold / question / evidence problem".
"""

import argparse
import json
import random
import sys

sys.path.insert(0, ".")

from env import db  # noqa: E402
from env.tasks import load_bird, load_spider  # noqa: E402
from env.tools import GOLD_TIMEOUT_S, render  # noqa: E402
from eval.metrics import buckets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--json", required=True)
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--spider", action="store_true")
    ap.add_argument("--bucket", default="all_fail", choices=["all_fail", "mixed", "all_pass"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.records) as f:
        records = [r for r in map(json.loads, f) if r["status"] != "gold_error"]
    ids = buckets(records)[args.bucket]
    ids = random.Random(args.seed).sample(ids, min(args.n, len(ids)))
    examples = {e.id: e for e in (load_spider if args.spider else load_bird)(args.json, args.db_dir)}
    by_id = {}
    for r in records:
        by_id.setdefault(r["id"], []).append(r)

    with open(args.out, "w", encoding="utf-8") as out:
        out.write(f"# Audit sample: bucket={args.bucket}, {len(ids)} of {len(buckets(records)[args.bucket])} questions\n\n")
        out.write("Label each entry: `MODEL` (model could not solve it) or `DATA` (gold / question / evidence is wrong or ambiguous).\n\n")
        for qid in ids:
            ex = examples[qid]
            conn = db.connect(ex.db_path)
            gold = render(db.execute(conn, ex.task.gold_sql, timeout_s=GOLD_TIMEOUT_S, max_rows=5), max_rows=5)
            conn.close()
            submitted = sorted({r["final_sql"] for r in by_id[qid] if r["final_sql"]})
            out.write(f"## {qid} · {ex.task.db_id}\n\n**Label:** \n\n**Q:** {ex.task.question}\n\n")
            if ex.task.evidence:
                out.write(f"**Evidence:** {ex.task.evidence}\n\n")
            out.write(f"**Gold:**\n```sql\n{ex.task.gold_sql}\n```\n```\n{gold}\n```\n\n")
            out.write(f"**Model submitted ({len(submitted)} distinct of {len(by_id[qid])} rollouts):**\n")
            for s in submitted[:4]:
                out.write(f"```sql\n{s}\n```\n")
            out.write("\n---\n\n")
    print(f"wrote {len(ids)} entries to {args.out}")


if __name__ == "__main__":
    main()
