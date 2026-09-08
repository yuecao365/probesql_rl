"""The official BIRD execution-accuracy judgement, kept verbatim.

`execute_sql` is copied unchanged from DAMO-ConvAI/bird/llm/src/evaluation.py
so that every comparison against it is against the real leaderboard rule:
`set(pred_rows) == set(gold_rows)`, any exception or a 30s timeout counts as 0.
Everything around it (prediction files, multiprocessing, difficulty buckets)
is the official CLI's concern and is not reproduced here.
"""

import sqlite3

from func_timeout import FunctionTimedOut, func_timeout

META_TIME_OUT = 30.0


def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def judge(predicted_sql: str, ground_truth: str, db_path: str) -> int:
    """1 if the official rule accepts the prediction, else 0."""
    try:
        return func_timeout(META_TIME_OUT, execute_sql, args=(predicted_sql, ground_truth, db_path))
    except (FunctionTimedOut, Exception):
        return 0
