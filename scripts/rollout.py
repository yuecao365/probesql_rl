"""Run the multi-turn agent on a benchmark sample and dump trajectories.

    python scripts/rollout.py --json data/bird/dev_20240627/dev.json --db-dir data/bird/dev_20240627/dev_databases \
        --n 300 --g 1 --base-url http://localhost:8000/v1 --model qwen7b --out outputs/dev300_zeroshot.jsonl

One JSON line per (question, rollout index). Re-running with the same --out
skips pairs already on disk, so an interrupted run on a rented GPU resumes
instead of restarting.
"""

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

sys.path.insert(0, ".")

from openai import OpenAI  # noqa: E402

from env import db, rollout  # noqa: E402
from env.policy import ChatPolicy  # noqa: E402
from env.tasks import load_bird, load_spider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--spider", action="store_true")
    ap.add_argument("--n", type=int, help="questions to sample; default all")
    ap.add_argument("--g", type=int, default=1, help="rollouts per question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    examples = (load_spider if args.spider else load_bird)(args.json, args.db_dir)
    if args.n:
        examples = random.Random(args.seed).sample(examples, args.n)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["id"], r["k"]) for r in map(json.loads, f)}
    jobs = [(ex, k) for ex in examples for k in range(args.g) if (ex.id, k) not in done]
    print(f"{len(jobs)} rollouts to run, {len(done)} already done")

    policy = ChatPolicy(OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=5),
                        args.model, args.temperature, args.max_tokens)
    lock = threading.Lock()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def one(ex, k):
        conn = db.connect(ex.db_path)
        try:
            traj = rollout.run(ex.task, conn, policy, max_turns=args.max_turns)
            rec = asdict(traj)
        except db.DbError as e:  # the gold itself failed; nothing to learn from
            rec = {"messages": [], "steps": [], "status": "gold_error", "final_sql": None, "correct": None, "error": str(e)}
        finally:
            conn.close()
        rec.update(id=ex.id, k=k, db_id=ex.task.db_id, difficulty=ex.difficulty)
        with lock, open(args.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec["correct"]

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, ex, k) for ex, k in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
