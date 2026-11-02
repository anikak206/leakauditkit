import numpy as np
import pytest

pytest.importorskip("numpy")

from leakauditkit.detectors import semantic as sem_mod
from leakauditkit.detectors.semantic import SemanticDetector


class FakeModel:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        out = []
        for t in texts:
            v = self.mapping.get(t, np.zeros(4))
            v = np.asarray(v, dtype=float)
            if normalize_embeddings:
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
            out.append(v)
        return np.asarray(out)


def test_semantic_contaminated(monkeypatch):
    # map text -> embedding vector
    mapping = {
        "duplicate sentence here": [1.0, 0.0, 0.0, 0.0],
        "totally different content": [0.0, 1.0, 0.0, 0.0],
    }
    fake = FakeModel(mapping)
    monkeypatch.setattr(sem_mod, "_load_model", lambda name: fake)

    train = [{"id": "t1", "text": "duplicate sentence here"}]
    eval_ = [
        {"id": "e1", "text": "duplicate sentence here"},
        {"id": "e2", "text": "totally different content"},
    ]
    det = SemanticDetector(threshold=0.85)
    res = det.detect(train, eval_)
    assert len(res.matches) == 1
    assert res.matches[0].eval_id == "e1"


def test_semantic_clean(monkeypatch):
    mapping = {
        "train sentence": [1.0, 0.0, 0.0, 0.0],
        "eval sentence": [0.0, 1.0, 0.0, 0.0],
    }
    fake = FakeModel(mapping)
    monkeypatch.setattr(sem_mod, "_load_model", lambda name: fake)
    train = [{"id": "t1", "text": "train sentence"}]
    eval_ = [{"id": "e1", "text": "eval sentence"}]
    det = SemanticDetector(threshold=0.85)
    res = det.detect(train, eval_)
    assert len(res.matches) == 0


def test_semantic_threshold_validation():
    with pytest.raises(ValueError):
        SemanticDetector(batch_size=0)