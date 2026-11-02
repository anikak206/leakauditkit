import pytest

from leakauditkit.detectors.ngram import NgramDetector, ngrams, tokenize


def test_tokenize_basic():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_ngrams_length():
    toks = ["a", "b", "c", "d"]
    assert ngrams(toks, 2) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert ngrams(toks, 5) == []


def test_ngrams_invalid():
    with pytest.raises(ValueError):
        ngrams(["a"], 0)


def test_exact_duplicate_flagged():
    train = [{"id": "t1", "text": "the quick brown fox jumps over the lazy dog today"}]
    eval_ = [{"id": "e1", "text": "the quick brown fox jumps over the lazy dog today"}]
    det = NgramDetector(n=4, threshold=0.3, min_shared=2)
    res = det.detect(train, eval_)
    assert len(res.matches) == 1
    assert res.matches[0].eval_id == "e1"
    assert res.matches[0].train_id == "t1"


def test_clean_eval_not_flagged():
    train = [{"id": "t1", "text": "completely unrelated training sentence about cats"}]
    eval_ = [{"id": "e1", "text": "totally different evaluation sentence about dogs"}]
    det = NgramDetector(n=4, threshold=0.3, min_shared=2)
    res = det.detect(train, eval_)
    assert len(res.matches) == 0
    assert res.contamination_rate == 0.0


def test_threshold_filters():
    train = [{"id": "t1", "text": "a b c d e f g h i j"}]
    eval_ = [{"id": "e1", "text": "a b c d e x y z"}]
    det = NgramDetector(n=3, threshold=0.9, min_shared=1)
    res = det.detect(train, eval_)
    assert len(res.matches) == 0


def test_min_shared():
    det = NgramDetector(n=3, threshold=0.1, min_shared=10)
    train = [{"id": "t1", "text": "a b c d e"}]
    eval_ = [{"id": "e1", "text": "a b c d e"}]
    res = det.detect(train, eval_)
    assert len(res.matches) == 0


def test_empty_inputs():
    det = NgramDetector()
    res = det.detect([], [])
    assert res.n_train == 0 and res.n_eval == 0
    assert len(res.matches) == 0


def test_empty_text_skipped():
    train = [{"id": "t1", "text": ""}]
    eval_ = [{"id": "e1", "text": "hello world foo bar"}]
    det = NgramDetector(n=2, threshold=0.1, min_shared=1)
    res = det.detect(train, eval_)
    assert len(res.matches) == 0


def test_best_match_per_eval():
    train = [
        {"id": "t1", "text": "shared phrase one two three four five"},
        {"id": "t2", "text": "shared phrase one two three four five extra"},
    ]
    eval_ = [{"id": "e1", "text": "shared phrase one two three four five"}]
    det = NgramDetector(n=3, threshold=0.2, min_shared=2)
    res = det.detect(train, eval_)
    # only one match per eval item
    assert len([m for m in res.matches if m.eval_id == "e1"]) == 1


def test_idf_mode():
    train = [{"id": "t1", "text": "common common common rare_unique_token here"}]
    eval_ = [{"id": "e1", "text": "common common common rare_unique_token here"}]
    det = NgramDetector(n=2, threshold=0.1, min_shared=1, use_idf=True)
    res = det.detect(train, eval_)
    assert len(res.matches) == 1


def test_contamination_rate():
    train = [{"id": "t1", "text": "duplicate text alpha beta gamma delta"}]
    eval_ = [
        {"id": "e1", "text": "duplicate text alpha beta gamma delta"},
        {"id": "e2", "text": "unrelated zebra elephant"},
    ]
    det = NgramDetector(n=3, threshold=0.3, min_shared=2)
    res = det.detect(train, eval_)
    assert res.contamination_rate == pytest.approx(0.5)


def test_threshold_validation():
    with pytest.raises(ValueError):
        NgramDetector(threshold=1.5)
    with pytest.raises(ValueError):
        NgramDetector(n=0)