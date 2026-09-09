import json
import sqlite3

import pytest

from env.tasks import load_bird
from eval import bird_official
from eval.consistency import check, ours


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "d" / "d.sqlite"
    path.parent.mkdir()
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE t (id INTEGER, name TEXT, code TEXT)")
        c.executemany("INSERT INTO t VALUES (?, ?, ?)", [(1, "a", "1"), (2, "b", "2"), (2, "b", "2")])
    return str(path)


# --- the official rule, verbatim ---------------------------------------------


def test_official_ignores_row_order_and_duplicates(db_path):
    assert bird_official.judge("SELECT id FROM t ORDER BY id DESC", "SELECT DISTINCT id FROM t", db_path) == 1


def test_official_is_column_order_sensitive(db_path):
    assert bird_official.judge("SELECT name, id FROM t", "SELECT id, name FROM t", db_path) == 0


def test_official_is_type_strict(db_path):
    assert bird_official.judge("SELECT code FROM t", "SELECT id FROM t", db_path) == 0


def test_official_error_is_zero(db_path):
    assert bird_official.judge("SELECT nope FROM t", "SELECT id FROM t", db_path) == 0
    assert bird_official.judge("SELECT id FROM t", "SELECT nope FROM t", db_path) == 0


def test_official_timeout_is_zero(db_path, monkeypatch):
    monkeypatch.setattr(bird_official, "META_TIME_OUT", 0.2)
    infinite = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    assert bird_official.judge(infinite, "SELECT 1", db_path) == 0


# --- consistency --------------------------------------------------------------


def test_ours_broken_gold_is_false(db_path):
    assert ours("SELECT id FROM t", "SELECT nope FROM t", db_path) is False


def test_check_matches_official_and_skips_missing(tmp_path, db_path):
    (tmp_path / "dev.json").write_text(json.dumps([
        {"question_id": i, "db_id": "d", "question": "q", "evidence": "", "SQL": "SELECT id FROM t"} for i in range(4)
    ]))
    examples = load_bird(str(tmp_path / "dev.json"), str(tmp_path))
    preds = {
        "0": "SELECT DISTINCT id FROM t",  # both accept
        "1": "SELECT name FROM t",  # both reject
        "2": "SELECT code FROM t",  # '1' vs 1: both reject, no normalization on our side either
        # "3" has no prediction and must be skipped
    }
    verdicts = {v.id: v for v in check(examples, preds)}
    assert set(verdicts) == {"0", "1", "2"}
    assert verdicts["0"].agree and verdicts["0"].ours
    assert verdicts["1"].agree and not verdicts["1"].ours
    assert verdicts["2"].agree and not verdicts["2"].ours
