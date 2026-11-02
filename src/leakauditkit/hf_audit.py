"""Audit HuggingFace datasets for benchmark contamination."""

from __future__ import annotations

from typing import Any

from .detectors.ngram import NgramDetector
from .loaders import load_hf


def audit_hf_dataset(
    train_dataset: str,
    eval_dataset: str,
    train_split: str = "train",
    eval_split: str = "test",
    text_col: str = "text",
    limit: int | None = 200,
    n: int = 5,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Load two HF datasets and run n-gram contamination detection.

    Requires the optional `datasets` dependency.
    """
    train = load_hf(train_dataset, split=train_split, text_col=text_col, limit=limit)
    eval_ = load_hf(eval_dataset, split=eval_split, text_col=text_col, limit=limit)
    det = NgramDetector(n=n, threshold=threshold, min_shared=3)
    result = det.detect(train, eval_)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "n_train": len(train),
        "n_eval": len(eval_),
        "result": result.to_dict(),
    }