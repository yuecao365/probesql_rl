import json

import pytest

from env.tasks import load_bird, load_spider


@pytest.fixture
def root(tmp_path):
    (tmp_path / "bird_dev.json").write_text(json.dumps([
        {"question_id": 7, "db_id": "a", "question": " How many? ", "evidence": " hint ", "SQL": " SELECT 1 ", "difficulty": "simple"},
        {"question_id": 8, "db_id": "b", "question": "q", "evidence": None, "SQL": "SELECT 2", "difficulty": "challenging"},
    ]), encoding="utf-8")
    (tmp_path / "bird_train.json").write_text(json.dumps([
        {"db_id": "a", "question": "q", "evidence": "", "SQL": "SELECT 1"},
        {"db_id": "a", "question": "q2", "SQL": "SELECT 2"},
    ]), encoding="utf-8")
    (tmp_path / "spider_dev.json").write_text(json.dumps([
        {"db_id": "c", "question": "q", "query": "SELECT 3"},
    ]), encoding="utf-8")
    return tmp_path


def test_bird_fields_are_stripped_and_evidence_optional(root):
    ex = load_bird(str(root / "bird_dev.json"), "/dbs")
    assert ex[0].id == "7" and ex[0].difficulty == "simple"
    assert ex[0].task.question == "How many?" and ex[0].task.evidence == "hint" and ex[0].task.gold_sql == "SELECT 1"
    assert ex[0].db_path == "/dbs/a/a.sqlite"
    assert ex[1].task.evidence == ""


def test_bird_train_without_question_id_uses_position(root):
    ex = load_bird(str(root / "bird_train.json"), "/dbs")
    assert [e.id for e in ex] == ["0", "1"]
    assert ex[1].task.evidence == "" and ex[1].difficulty is None


def test_spider_has_no_evidence_or_difficulty(root):
    ex = load_spider(str(root / "spider_dev.json"), "/dbs")
    assert ex[0].task.gold_sql == "SELECT 3" and ex[0].task.evidence == "" and ex[0].difficulty is None
    assert ex[0].db_path == "/dbs/c/c.sqlite"
