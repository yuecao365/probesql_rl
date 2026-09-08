"""Compare the training reward's verdict with the official BIRD judge.

    python scripts/check_consistency.py --json data/bird/dev/dev.json --db-dir data/bird/dev/dev_databases --gold 20
    python scripts/check_consistency.py --json ... --db-dir ... --predictions preds.jsonl

`--gold N` judges the first N gold queries against themselves (every verdict
must be accept/accept); `--predictions` takes JSON lines of {"id", "sql"}.
"""

import argparse
import json
import sys

sys.path.insert(0, ".")

from env.tasks import load_bird  # noqa: E402
from eval.consistency import check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--db-dir", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--gold", type=int, metavar="N")
    group.add_argument("--predictions")
    args = ap.parse_args()

    examples = load_bird(args.json, args.db_dir)
    if args.gold is not None:
        examples = examples[: args.gold]
        preds = {ex.id: ex.task.gold_sql for ex in examples}
    else:
        with open(args.predictions) as f:
            preds = {str(r["id"]): r["sql"] for r in map(json.loads, f)}

    verdicts = check(examples, preds)
    for v in verdicts:
        if not v.agree:
            print(f"DISAGREE id={v.id} ours={v.ours} official={v.official}")
    agree = sum(v.agree for v in verdicts)
    accepted = sum(v.official for v in verdicts)
    print(f"{len(verdicts)} judged, {agree} agree, official accepts {accepted}")


if __name__ == "__main__":
    main()
