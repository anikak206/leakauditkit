import json
from pathlib import Path

from click.testing import CliRunner

from leakauditkit.cli import main


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_audit_command(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    evalp = tmp_path / "eval.jsonl"
    _write_jsonl(train, [{"id": "t1", "text": "duplicate text alpha beta gamma delta"}])
    _write_jsonl(evalp, [
        {"id": "e1", "text": "duplicate text alpha beta gamma delta"},
        {"id": "e2", "text": "unrelated zebra elephant"},
    ])
    out = tmp_path / "out"
    runner = CliRunner()
    res = runner.invoke(main, ["audit", "--train", str(train), "--eval", str(evalp),
                               "--detector", "ngram", "--output", str(out), "--format", "json"])
    assert res.exit_code == 0, res.output
    assert (out / "ngram_report.json").exists()


def test_audit_all_formats(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    evalp = tmp_path / "eval.jsonl"
    _write_jsonl(train, [{"id": "t1", "text": "hello world foo bar baz"}])
    _write_jsonl(evalp, [{"id": "e1", "text": "hello world foo bar baz"}])
    out = tmp_path / "out"
    runner = CliRunner()
    res = runner.invoke(main, ["audit", "--train", str(train), "--eval", str(evalp),
                               "--detector", "ngram", "--output", str(out), "--format", "all"])
    assert res.exit_code == 0, res.output
    for ext in ("json", "md", "html"):
        assert (out / f"ngram_report.{ext}").exists()