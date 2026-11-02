# Shortcut learning audit

`leakaudit shortcuts` audits a prediction file for spurious correlations.

## Checks
- Class imbalance (majority class ratio >= 0.8).
- Model vs majority baseline (model must beat baseline).
- Single-feature shortcuts: a feature whose value alone predicts the label with
  >= 90% accuracy.
- Subgroup performance gaps (max-min accuracy >= 0.2).
- Length bias (|corr(text length, correctness)| >= 0.3).
- Feature-label mutual information (ranking).

## Input format
JSONL with at least `label` and `prediction` fields, plus feature columns.
```json
{"text": "...", "label": "pos", "prediction": "pos", "source": "A"}
```