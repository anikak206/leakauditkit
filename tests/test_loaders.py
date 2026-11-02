from pathlib import Path

import pytest

from leakauditkit.loaders import load, load_csv, load_jsonl


def test_jsonl_load(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"id":"a","text":"hello"}\n{"id":"b","text":"world"}\n', encoding="utf-8")
    rows = load_jsonl(p)
    assert len(rows) == 2
    assert rows[0]["id"] == "a"


def test_jsonl_auto_id(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"text":"hello"}\n', encoding="utf-8")
    rows = load_jsonl(p)
    assert "id" in rows[0]


def test_jsonl_invalid(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(p)


def test_csv_load(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("id,text\n1,hello\n2,world\n", encoding="utf-8")
    rows = load_csv(p)
    assert len(rows) == 2
    assert rows[0]["text"] == "hello"


def test_dispatch(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"id":"a","text":"hello"}\n', encoding="utf-8")
    assert load(p)[0]["id"] == "a"


def test_dispatch_unsupported(tmp_path: Path):
    p = tmp_path / "t.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load(p)