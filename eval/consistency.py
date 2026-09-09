"""Agreement between the training reward's verdict and the official BIRD judge.

The training reward is only trustworthy if it never accepts what the leaderboard
rejects. This runs both judges on the same (prediction, gold, database) triples
and reports every disagreement with its direction, so a lenient training reward
is caught before any RL step is spent on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from env import compare, db
from env.tasks import Example
from env.tools import GOLD_TIMEOUT_S
from eval.bird_official import judge


@dataclass(frozen=True)
class Verdict:
    id: str
    ours: bool
    official: bool

    @property
    def agree(self) -> bool:
        return self.ours == self.official


def ours(predicted_sql: str, gold_sql: str, db_path: str) -> bool:
    """The verdict the rollout's terminal reward would give."""
    conn = db.connect(db_path)
    try:
        gold = db.execute(conn, gold_sql, timeout_s=GOLD_TIMEOUT_S, max_rows=None).rows
        pred = db.execute(conn, predicted_sql, timeout_s=GOLD_TIMEOUT_S, max_rows=None).rows
    except db.DbError:
        return False  # a broken gold can never be matched, same as the official rule
    return compare.equal(pred, gold)


def check(examples: list[Example], predictions: dict[str, str]) -> list[Verdict]:
    """Judge each example whose id has a prediction; missing ids are skipped."""
    out = []
    for ex in examples:
        if ex.id not in predictions:
            continue
        sql = predictions[ex.id]
        out.append(Verdict(ex.id, ours(sql, ex.task.gold_sql, ex.db_path), bool(judge(sql, ex.task.gold_sql, ex.db_path))))
    return out
