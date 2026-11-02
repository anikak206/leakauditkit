# LeakAuditKit

> Audit ML/LLM evaluation pipelines for data leakage and shortcut learning.

[![CI](https://github.com/<user>/leakauditkit/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/leakauditkit/actions)

## Why

LLM benchmark scores are inflated by train/test contamination. Tabular ML models
cheat via feature leakage. Most teams discover this *after* shipping. LeakAuditKit
is a small, dependency-light CLI that runs two complementary detectors:

- **N-gram detector** — verbatim / near-verbatim overlap (Jaccard over n-grams,
  optional IDF weighting, doc-frequency filtering).
- **Semantic detector** — embedding cosine similarity (sentence-transformers,
  optional dependency).

It emits JSON / Markdown / self-contained HTML scorecards with per-pair evidence.

## Install

```bash
pip install -e .            # core (n-gram detector, CLI)
pip install -e .[semantic]  # + embedding detector
pip install -e .[dev]       # + tests/lint
```

## One-command demo

```bash
leakaudit audit --train examples/train.jsonl --eval examples/eval.jsonl --detector all --format all --output ./audit_report
```

Docker:

```bash
docker build -t leakauditkit .
docker run --rm -v "$PWD:/data" leakauditkit audit --train /data/examples/train.jsonl --eval /data/examples/eval.jsonl --detector ngram
```

## Architecture

```
src/leakauditkit/
  cli.py            Click CLI
  config.py         Pydantic config models
  loaders.py        jsonl / csv / json / HF loaders
  detectors/
    base.py         BaseDetector + result models
    ngram.py        N-gram overlap (IDF-weighted Jaccard)
    semantic.py     Embedding cosine (lazy import)
  report.py         JSON / Markdown / HTML scorecards
```

## Metrics & evaluation

This is a tooling project; its quality is measured by detector precision/recall on
a labelled contamination benchmark. **Numbers are reported only once measured on
real data** — see `docs/eval.md` (planned). Current status: unit tests cover
detection logic; a labelled benchmark is in progress.

## Limitations

- Semantic detector requires `sentence-transformers` (heavy, optional).
- N-gram detector is lexical: paraphrased leakage (semantic, non-lexical) is
  only caught by the semantic detector.
- No cross-lingual support yet.
- Thresholds are heuristic; calibrate on your domain.

## License

MIT