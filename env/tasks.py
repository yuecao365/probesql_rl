"""Benchmark loading: BIRD and Spider examples as Task objects plus a database path.

Both benchmarks ship a JSON list of questions and one SQLite file per database,
differing only in field names, so one loader with a field map covers both.
BIRD's `database_description` CSVs are deliberately not loaded: under the
hidden-schema setting the policy must learn the schema by probing, and giving
it column descriptions would collapse that into ordinary schema-prefill.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from env.prompt import Task


@dataclass(frozen=True)
class Example:
    id: str
    task: Task
    db_path: str
    difficulty: str | None  # BIRD only: simple / moderate / challenging


def _load(json_path: str, db_dir: str, sql_field: str, evidence_field: str | None) -> list[Example]:
    with open(json_path, encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for i, it in enumerate(items):
        db = it["db_id"]
        task = Task(
            db_id=db,
            question=it["question"].strip(),
            evidence=(it.get(evidence_field) or "").strip() if evidence_field else "",
            gold_sql=it[sql_field].strip(),
        )
        # BIRD's train split has no question_id; positional ids are stable per file.
        ident = str(it.get("question_id", i))
        out.append(Example(ident, task, os.path.join(db_dir, db, f"{db}.sqlite"), it.get("difficulty")))
    return out


def load_bird(json_path: str, db_dir: str) -> list[Example]:
    return _load(json_path, db_dir, sql_field="SQL", evidence_field="evidence")


def load_spider(json_path: str, db_dir: str) -> list[Example]:
    return _load(json_path, db_dir, sql_field="query", evidence_field=None)
