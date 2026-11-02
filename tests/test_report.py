from pathlib import Path

from leakauditkit.detectors.ngram import NgramDetector
from leakauditkit.report import write_html, write_json, write_markdown


def test_report_files(tmp_path: Path):
    train = [{"id": "t1", "text": "duplicate text alpha beta gamma delta"}]
    eval_ = [{"id": "e1", "text": "duplicate text alpha beta gamma delta"}]
    res = NgramDetector(n=3, threshold=0.3, min_shared=2).detect(train, eval_)
    write_json(res, tmp_path / "r.json")
    write_markdown(res, tmp_path / "r.md")
    write_html(res, tmp_path / "r.html")
    assert (tmp_path / "r.json").exists()
    assert (tmp_path / "r.md").exists()
    assert (tmp_path / "r.html").exists()
    assert "contamination_rate" in (tmp_path / "r.json").read_text()