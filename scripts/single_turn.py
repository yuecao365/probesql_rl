"""Single-turn, full-schema baseline: the number the hidden-schema agent is measured against.

    python scripts/single_turn.py --json ... --db-dir ... --n 300 --base-url ... --model qwen7b --out outputs/dev300_single.jsonl

The model sees every CREATE TABLE statement plus the evidence and answers once
with a ```sql block. Records use the same shape as scripts/rollout.py (with no
steps) so scripts/metrics.py reads them unchanged.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

from openai import OpenAI  # noqa: E402

from env import db  # noqa: E402
from env.tasks import load_bird, load_spider  # noqa: E402
from eval.consistency import ours  # noqa: E402

SYSTEM = "You are an expert SQL analyst. Given a SQLite schema and a question, answer with one SQL query in a ```sql block."
_SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.S)


def schema_ddl(db_path: str) -> str:
    conn = db.connect(db_path)
    rows = db.execute(conn, "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name").rows
    conn.close()
    return "\n".join(r[0] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--spider", action="store_true")
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    examples = (load_spider if args.spider else load_bird)(args.json, args.db_dir)
    if args.n:
        examples = random.Random(args.seed).sample(examples, args.n)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=5)
    lock = threading.Lock()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def one(ex):
        user = f"Schema:\n{schema_ddl(ex.db_path)}\n\nQuestion: {ex.task.question}"
        if ex.task.evidence:
            user += f"\nHint: {ex.task.evidence}"
        resp = client.chat.completions.create(
            model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        text = resp.choices[0].message.content or ""
        blocks = _SQL_BLOCK.findall(text)
        sql = blocks[-1].strip() if blocks else None
        rec = {
            "id": ex.id, "k": 0, "db_id": ex.task.db_id, "difficulty": ex.difficulty,
            "status": "submitted" if sql else "parse_error", "final_sql": sql,
            "correct": ours(sql, ex.task.gold_sql, ex.db_path) if sql else False,
            "steps": [], "messages": [{"role": "assistant", "content": text,
                                       "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}}],
        }
        with lock, open(args.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(one, ex) for ex in examples]), 1):
            fut.result()
            if i % 50 == 0:
                print(f"{i}/{len(examples)}", flush=True)


if __name__ == "__main__":
    main()
