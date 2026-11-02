"""Data loaders for train/eval corpora."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} of {path} is not a JSON object")
            if "id" not in obj:
                obj["id"] = f"{path.stem}_{line_no}"
            if "text" not in obj and "content" not in obj:
                # allow any text-bearing key; otherwise coerce whole object
                obj["text"] = json.dumps(obj, ensure_ascii=False)
            out.append(obj)
    return out


def load_csv(path: str | Path, text_col: str = "text", id_col: str = "id") -> list[dict[str, Any]]:
    path = Path(path)
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            item: dict[str, Any] = dict(row)
            if id_col in item and item[id_col]:
                item["id"] = item[id_col]
            else:
                item["id"] = f"{path.stem}_{i}"
            if text_col in item:
                item["text"] = item[text_col]
            out.append(item)
    return out


def load_hf(dataset_name: str, split: str = "train", text_col: str = "text",
            id_col: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset split. Requires `datasets` (optional dep)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for HF loading. Install with: pip install 'leakauditkit[hf]'"
        ) from exc
    ds = load_dataset(dataset_name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        item = dict(row)
        if id_col and id_col in item:
            item["id"] = item[id_col]
        else:
            item["id"] = f"{dataset_name}:{split}:{i}"
        if text_col in item:
            item["text"] = item[text_col]
        out.append(item)
    return out


def load(path: str | Path | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Dispatch loader by file extension."""
    if path is None:
        raise ValueError("path is required")
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return load_jsonl(p)
    if p.suffix.lower() == ".csv":
        return load_csv(p, **{k: v for k, v in kwargs.items() if k in ("text_col", "id_col")})
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError("JSON file must contain a list of objects")
    raise ValueError(f"Unsupported file type: {p.suffix}")