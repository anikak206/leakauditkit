"""Benchmark harness for leakage detectors.

Generates labelled contamination scenarios (verbatim, partial, clean) and
measures detector precision/recall/F1 across thresholds.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detectors.base import BaseDetector, DetectionResult


@dataclass
class BenchmarkScenario:
    name: str
    train: list[dict[str, Any]]
    eval_: list[dict[str, Any]]
    ground_truth: set[str]  # eval ids that are contaminated
    description: str = ""


@dataclass
class BenchmarkResult:
    scenario: str
    detector: str
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    n_train: int
    n_eval: int
    n_contaminated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "detector": self.detector,
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_contaminated": self.n_contaminated,
        }


def _base_corpus(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
             "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"]
    out = []
    for _i in range(n):
        length = rng.randint(8, 16)
        out.append(" ".join(rng.choice(vocab) for _ in range(length)))
    return out


def make_scenario(
    name: str,
    n_train: int = 40,
    n_eval: int = 40,
    contamination_rate: float = 0.25,
    mode: str = "verbatim",
    seed: int = 42,
    partial_overlap: float = 0.6,
) -> BenchmarkScenario:
    """Build a labelled scenario.

    mode: 'verbatim' | 'partial' | 'clean' | 'mixed'
    """
    if not 0.0 <= contamination_rate <= 1.0:
        raise ValueError("contamination_rate must be in [0,1]")
    rng = random.Random(seed)
    train_texts = _base_corpus(n_train, seed)
    eval_texts = _base_corpus(n_eval, seed + 1)

    train = [{"id": f"t{i}", "text": t} for i, t in enumerate(train_texts)]
    # decide which eval items are contaminated
    n_contam = int(round(n_eval * contamination_rate))
    contam_ids = set(rng.sample([f"e{i}" for i in range(n_eval)], n_contam))

    for i, t in enumerate(eval_texts):
        eid = f"e{i}"
        if eid in contam_ids:
            src_idx = rng.randrange(n_train)
            src_text = train_texts[src_idx]
            if mode == "verbatim":
                eval_texts[i] = src_text
            elif mode == "partial":
                toks = src_text.split()
                k = max(3, int(len(toks) * partial_overlap))
                eval_texts[i] = " ".join(toks[:k]) + " " + " ".join(t.split()[:3])
            elif mode == "mixed":
                if rng.random() < 0.5:
                    eval_texts[i] = src_text
                else:
                    toks = src_text.split()
                    k = max(3, int(len(toks) * partial_overlap))
                    eval_texts[i] = " ".join(toks[:k]) + " " + " ".join(t.split()[:3])
            # 'clean' leaves eval_texts[i] untouched (not in contam_ids anyway)
    eval_ = [{"id": f"e{i}", "text": t} for i, t in enumerate(eval_texts)]
    # remove contam ids that ended up not contaminated in 'clean' mode
    if mode == "clean":
        contam_ids = set()
    return BenchmarkScenario(
        name=name,
        train=train,
        eval_=eval_,
        ground_truth=contam_ids,
        description=f"mode={mode} rate={contamination_rate} seed={seed}",
    )


def evaluate(detector: BaseDetector, scenario: BenchmarkScenario) -> BenchmarkResult:
    result: DetectionResult = detector.detect(scenario.train, scenario.eval_)
    predicted = {m.eval_id for m in result.matches}
    gt = scenario.ground_truth
    tp = len(predicted & gt)
    fp = len(predicted - gt)
    fn = len(gt - predicted)
    tn = len(scenario.eval_) - tp - fp - fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return BenchmarkResult(
        scenario=scenario.name,
        detector=detector.name,
        threshold=detector.threshold,
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1,
        n_train=len(scenario.train),
        n_eval=len(scenario.eval_),
        n_contaminated=len(gt),
    )


def threshold_sweep(
    detector_factory: Any,
    scenario: BenchmarkScenario,
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    out = []
    for thr in thresholds:
        det = detector_factory(threshold=thr)
        r = evaluate(det, scenario)
        out.append({"threshold": thr, "precision": r.precision, "recall": r.recall, "f1": r.f1,
                    "tp": r.tp, "fp": r.fp, "fn": r.fn})
    return out


def run_suite(detector_factory: Any, scenarios: list[BenchmarkScenario]) -> list[BenchmarkResult]:
    results = []
    for sc in scenarios:
        det = detector_factory()
        results.append(evaluate(det, sc))
    return results


def save_results(results: list[BenchmarkResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")


def plot_threshold_sweep(sweep: list[dict[str, Any]], path: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    thrs = [d["threshold"] for d in sweep]
    plt.figure(figsize=(7, 4))
    plt.plot(thrs, [d["precision"] for d in sweep], label="precision")
    plt.plot(thrs, [d["recall"] for d in sweep], label="recall")
    plt.plot(thrs, [d["f1"] for d in sweep], label="f1")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title(title or "Threshold sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()