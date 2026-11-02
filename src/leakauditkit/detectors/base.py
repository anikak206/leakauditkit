"""Base abstractions for leakage detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContaminationMatch:
    """A single detected overlap between a train item and an eval item."""

    train_id: str
    eval_id: str
    score: float
    method: str
    evidence: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_id": self.train_id,
            "eval_id": self.eval_id,
            "score": self.score,
            "method": self.method,
            "evidence": self.evidence,
            "meta": self.meta,
        }


@dataclass
class DetectionResult:
    """Aggregated output of a detector run."""

    detector: str
    n_train: int
    n_eval: int
    threshold: float
    matches: list[ContaminationMatch] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def contaminated_eval_ids(self) -> set[str]:
        return {m.eval_id for m in self.matches}

    @property
    def contamination_rate(self) -> float:
        if self.n_eval == 0:
            return 0.0
        return len(self.contaminated_eval_ids) / self.n_eval

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "threshold": self.threshold,
            "n_matches": len(self.matches),
            "contamination_rate": round(self.contamination_rate, 4),
            "matches": [m.to_dict() for m in self.matches],
            "stats": self.stats,
        }


class BaseDetector(ABC):
    """Abstract base for all leakage detectors."""

    name: str = "base"

    def __init__(self, threshold: float = 0.8) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.threshold = threshold

    @abstractmethod
    def detect(
        self,
        train: Iterable[dict[str, Any]],
        eval_: Iterable[dict[str, Any]],
    ) -> DetectionResult:
        """Return contamination matches above threshold."""

    @staticmethod
    def _ids(items: Iterable[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for i, it in enumerate(items):
            out.append(str(it.get("id", f"item_{i}")))
        return out