"""Configuration models for LeakAuditKit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DetectorConfig(BaseModel):
    name: Literal["ngram", "tfidf", "semantic", "all"] = "all"
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    n: int = Field(default=5, ge=1)
    min_shared: int = Field(default=3, ge=1)
    use_idf: bool = False
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(default=32, ge=1)


class AuditConfig(BaseModel):
    train_path: Path
    eval_path: Path
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    output_dir: Path = Path("./audit_report")
    format: Literal["json", "html", "markdown", "all"] = "all"

    @field_validator("train_path", "eval_path", "output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v)