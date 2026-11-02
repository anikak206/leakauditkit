"""TF-IDF cosine contamination detector (zero heavy dependencies)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseDetector, ContaminationMatch, DetectionResult


class TfidfDetector(BaseDetector):
    """Detects eval items whose TF-IDF vector is close to a train item.

    Uses scikit-learn only, so it works out of the box without
    sentence-transformers. Useful as a strong, dependency-light baseline
    between lexical n-grams and dense embeddings.
    """

    name = "tfidf"

    def __init__(self, threshold: float = 0.5, ngram_range: tuple[int, int] = (1, 2),
                 min_df: int = 1) -> None:
        super().__init__(threshold=threshold)
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError("invalid ngram_range")
        self.ngram_range = ngram_range
        self.min_df = min_df

    def _text_of(self, item: dict[str, Any]) -> str:
        for key in ("text", "content", "passage", "document"):
            if key in item and item[key] is not None:
                return str(item[key])
        return ""

    def detect(self, train: Iterable[dict[str, Any]], eval_: Iterable[dict[str, Any]]) -> DetectionResult:
        train_list = list(train)
        eval_list = list(eval_)

        train_texts = [self._text_of(t) for t in train_list]
        eval_texts = [self._text_of(e) for e in eval_list]
        all_texts = train_texts + eval_texts

        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=self.min_df)
        try:
            X = vectorizer.fit_transform(all_texts)
        except ValueError:
            # empty vocabulary
            X = None

        if X is None or X.shape[1] == 0:
            return DetectionResult(detector=self.name, n_train=len(train_list),
                                  n_eval=len(eval_list), threshold=self.threshold,
                                  matches=[], stats={"error": "empty vocabulary"})

        train_X = X[: len(train_list)]
        eval_X = X[len(train_list):]
        sim = cosine_similarity(eval_X, train_X)

        matches: list[ContaminationMatch] = []
        for ei in range(len(eval_list)):
            row = sim[ei]
            # flag every train above threshold; report best per eval
            flagged = [ti for ti in range(len(train_list)) if float(row[ti]) >= self.threshold]
            if not flagged:
                continue
            ti = max(flagged, key=lambda t: float(row[t]))
            matches.append(ContaminationMatch(
                train_id=str(train_list[ti].get("id", f"train_{ti}")),
                eval_id=str(eval_list[ei].get("id", f"eval_{ei}")),
                score=round(float(row[ti]), 4),
                method="tfidf",
                evidence=f"cosine={float(row[ti]):.4f}",
                meta={"n_flagged_trains": len(flagged)},
            ))

        return DetectionResult(
            detector=self.name,
            n_train=len(train_list),
            n_eval=len(eval_list),
            threshold=self.threshold,
            matches=matches,
            stats={
                "vocab_size": int(X.shape[1]),
                "max_sim_mean": round(float(sim.max(axis=1).mean()) if len(sim) else 0.0, 4),
                "max_sim_max": round(float(sim.max()) if len(sim) else 0.0, 4),
            },
        )