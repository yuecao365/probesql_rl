"""Summarize one or more trajectory files and, optionally, list bucket ids.

    python scripts/metrics.py outputs/dev300_zeroshot.jsonl
    python scripts/metrics.py outputs/train200_g8.jsonl --bucket all_fail > all_fail_ids.txt
"""

import argparse
import json
import sys

sys.path.insert(0, ".")

from eval.metrics import buckets, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bucket", choices=["all_fail", "mixed", "all_pass"])
    args = ap.parse_args()
    for path in args.files:
        with open(path) as f:
            records = [r for r in map(json.loads, f) if r["status"] != "gold_error"]
        if args.bucket:
            print("\n".join(buckets(records)[args.bucket]))
            continue
        print(f"== {path}")
        for k, v in summarize(records).items():
            print(f"{k:24} {json.dumps(v) if isinstance(v, dict) else f'{v:.4f}' if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()
