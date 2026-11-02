# Evaluation methodology

## Benchmark design
- Synthetic corpora drawn from a fixed vocabulary with seeded RNG for reproducibility.
- Contamination injected at a controlled rate by copying (verbatim) or
  partially copying (partial) train text into eval items.
- Ground truth is the set of eval item ids that were injected.
- Metrics: precision, recall, F1, TP/FP/FN/TN.

## Why synthetic?
No public benchmark exists for *contamination detection* (as opposed to
contamination itself). Injecting known contamination at controlled rates is the
standard way to measure a detector's recall at a fixed false-positive budget.

## Reproduce
```bash
leakaudit benchmark --output ./bench --n-train 60 --n-eval 60 --seed 42
```

## Results
See README benchmark table. N-gram: precision 1.0 (zero false positives), recall
0.47 on partial overlap. Semantic detector requires sentence-transformers; not
benchmarked in this environment.