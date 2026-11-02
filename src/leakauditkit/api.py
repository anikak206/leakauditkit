"""FastAPI service exposing LeakAuditKit detectors."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .detectors.ngram import NgramDetector

app = FastAPI(title="LeakAuditKit API", version="0.2.0")


class AuditRequest(BaseModel):
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]]
    detector: str = "ngram"
    threshold: float = 0.3
    n: int = 5


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/audit")
def audit(req: AuditRequest) -> dict[str, Any]:
    if req.detector == "ngram":
        det: Any = NgramDetector(n=req.n, threshold=req.threshold, min_shared=3)
    elif req.detector == "tfidf":
        from .detectors.tfidf import TfidfDetector

        det = TfidfDetector(threshold=req.threshold)
    else:
        raise HTTPException(status_code=400, detail=f"unknown detector: {req.detector}")
    result = det.detect(req.train, req.eval)
    from typing import cast

    return cast(dict[str, Any], result.to_dict())