"""Semantic (embedding cosine) contamination detector.

sentence-transformers is an optional dependency. Import is deferred so the
package stays importable on machines without it (CI, minimal installs).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .base import BaseDetector, ContaminationMatch, DetectionResult

_MODEL_CACHE: dict[str, Any] = {}


def _load_model(model_name: str) -> Any:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError(
            "sentence-transformers is required for the semantic detector. "
            "Install with: pip install 'leakauditkit[semantic]'"
        ) from exc
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


class SemanticDetector(BaseDetector):
    """Detects eval items semantically close to any train item.

    Embeds train and eval texts, computes cosine similarity, flags pairs
    above threshold.
    """

    name = "semantic"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.85,
        batch_size: int = 32,
        lowercase: bool = False,
    ) -> None:
        super().__init__(threshold=threshold)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self.batch_size = batch_size
        self.lowercase = lowercase

    def _text_of(self, item: dict[str, Any]) -> str:
        for key in ("text", "content", "passage", "document"):
            if key in item and item[key] is not None:
                t = str(item[key])
                return t.lower() if self.lowercase else t
        return ""

    def detect(
        self,
        train: Iterable[dict[str, Any]],
        eval_: Iterable[dict[str, Any]],
    ) -> DetectionResult:
        train_list = list(train)
        eval_list = list(eval_)

        train_texts = [self._text_of(t) for t in train_list]
        eval_texts = [self._text_of(e) for e in eval_list]

        model = _load_model(self.model_name)
        train_emb = np.asarray(
            model.encode(train_texts, batch_size=self.batch_size, show_progress_bar=False,
                         normalize_embeddings=True)
        )
        eval_emb = np.asarray(
            model.encode(eval_texts, batch_size=self.batch_size, show_progress_bar=False,
                         normalize_embeddings=True)
        )

        # cosine similarity matrix (embeddings already normalized)
        sim = eval_emb @ train_emb.T

        matches: list[ContaminationMatch] = []
        for ei in range(len(eval_list)):
            row = sim[ei]
            best_ti = int(np.argmax(row)) if len(row) else 0
            best_score = float(row[best_ti]) if len(row) else 0.0
            # flag every train item above threshold, but report best per eval
            flagged_train_indices = [ti for ti in range(len(train_list)) if float(row[ti]) >= self.threshold]
            if not flagged_train_indices:
                continue
            # choose highest-scoring train as evidence
            ti = max(flagged_train_indices, key=lambda t: float(row[t]))
            matches.append(
                ContaminationMatch(
                    train_id=str(train_list[ti].get("id", f"train_{ti}")),
                    eval_id=str(eval_list[ei].get("id", f"eval_{ei}")),
                    score=round(float(row[ti]), 4),
                    method="semantic",
                    evidence=f"cosine={float(row[ti]):.4f} (best_train={train_list[ti].get('id', ti)})",
                    meta={
                        "best_score": round(best_score, 4),
                        "n_flagged_trains": len(flagged_train_indices),
                    },
                )
            )

        return DetectionResult(
            detector=self.name,
            n_train=len(train_list),
            n_eval=len(eval_list),
            threshold=self.threshold,
            matches=matches,
            stats={
                "model": self.model_name,
                "max_sim_mean": round(float(sim.max(axis=1).mean()) if len(sim) else 0.0, 4),
                "max_sim_max": round(float(sim.max()) if len(sim) else 0.0, 4),
            },
        )