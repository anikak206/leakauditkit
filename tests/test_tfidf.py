from leakauditkit.detectors.tfidf import TfidfDetector


def test_tfidf_verbatim():
    train = [{"id": "t1", "text": "the quick brown fox jumps over the lazy dog"}]
    eval_ = [{"id": "e1", "text": "the quick brown fox jumps over the lazy dog"}]
    det = TfidfDetector(threshold=0.5)
    res = det.detect(train, eval_)
    assert len(res.matches) == 1


def test_tfidf_clean():
    train = [{"id": "t1", "text": "completely unrelated training sentence about cats"}]
    eval_ = [{"id": "e1", "text": "totally different evaluation sentence about dogs"}]
    det = TfidfDetector(threshold=0.5)
    res = det.detect(train, eval_)
    assert len(res.matches) == 0


def test_tfidf_threshold_validation():
    import pytest
    with pytest.raises(ValueError):
        TfidfDetector(ngram_range=(2, 1))