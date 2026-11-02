"""Audit ML/LLM predictions for shortcut learning and spurious correlations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score


@dataclass
class SubgroupStat:
    feature: str
    value: str
    n: int
    accuracy: float


@dataclass
class ShortcutReport:
    n: int
    model_accuracy: float
    majority_baseline_accuracy: float
    class_imbalance_ratio: float
    majority_class: str
    feature_mi: list[tuple[str, float]]  # (feature, mutual information)
    subgroup_gaps: list[tuple[str, float]]  # (feature, max-min accuracy gap)
    single_feature_shortcuts: list[tuple[str, float]]  # feature whose value predicts label with >= threshold acc
    length_label_corr: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "model_accuracy": round(self.model_accuracy, 4),
            "majority_baseline_accuracy": round(self.majority_baseline_accuracy, 4),
            "class_imbalance_ratio": round(self.class_imbalance_ratio, 4),
            "majority_class": self.majority_class,
            "feature_mi": [(f, round(v, 4)) for f, v in self.feature_mi],
            "subgroup_gaps": [(f, round(v, 4)) for f, v in self.subgroup_gaps],
            "single_feature_shortcuts": [(f, round(v, 4)) for f, v in self.single_feature_shortcuts],
            "length_label_corr": round(self.length_label_corr, 4),
            "flags": self.flags,
        }


def _encode_feature(values: list[Any]) -> np.ndarray:
    # factorize to integers; for numeric, keep as-is if all numeric
    try:
        arr = np.asarray(values, dtype=float)
        if not np.isnan(arr).any():
            return arr.reshape(-1, 1)
    except (ValueError, TypeError):
        pass
    codes, _ = np.unique(values, return_inverse=True)
    return codes.reshape(-1, 1)


def audit_shortcuts(
    records: list[dict[str, Any]],
    label_key: str = "label",
    pred_key: str = "prediction",
    feature_keys: list[str] | None = None,
    text_key: str | None = None,
    shortcut_acc_threshold: float = 0.9,
    subgroup_gap_threshold: float = 0.2,
) -> ShortcutReport:
    if not records:
        raise ValueError("records must be non-empty")
    y = np.asarray([r[label_key] for r in records])
    preds = np.asarray([r[pred_key] for r in records])
    n = len(records)

    # majority baseline
    classes, counts = np.unique(y, return_counts=True)
    maj_idx = int(np.argmax(counts))
    majority_class = str(classes[maj_idx])
    majority_acc = float(counts[maj_idx] / n)
    model_acc = float(accuracy_score(y, preds))

    imbalance = float(counts[maj_idx] / n)

    if feature_keys is None:
        feature_keys = [k for k in records[0] if k not in (label_key, pred_key, text_key or "")]

    # mutual information
    feature_mi: list[tuple[str, float]] = []
    X_cols = []
    for fk in feature_keys:
        col = _encode_feature([r.get(fk) for r in records])
        X_cols.append(col)
        try:
            mi = mutual_info_classif(col, y, discrete_features=True, random_state=0)
            feature_mi.append((fk, float(mi[0])))
        except Exception:
            feature_mi.append((fk, 0.0))
    feature_mi.sort(key=lambda x: x[1], reverse=True)

    # subgroup performance gaps
    subgroup_gaps: list[tuple[str, float]] = []
    for fk in feature_keys:
        groups: dict[Any, list[int]] = {}
        for i, r in enumerate(records):
            groups.setdefault(r.get(fk), []).append(i)
        accs = []
        for _v, idxs in groups.items():
            if len(idxs) < 2:
                continue
            a = accuracy_score(y[idxs], preds[idxs])
            accs.append(a)
        gap = (max(accs) - min(accs)) if len(accs) >= 2 else 0.0
        subgroup_gaps.append((fk, float(gap)))
    subgroup_gaps.sort(key=lambda x: x[1], reverse=True)

    # single-feature shortcut: does a feature value map deterministically to a label?
    single_shortcuts: list[tuple[str, float]] = []
    for fk in feature_keys:
        g2: dict[Any, list[Any]] = {}
        for r in records:
            g2.setdefault(r.get(fk), []).append(r[label_key])
        # max purity
        purities = []
        for _v, labels in g2.items():
            if not labels:
                continue
            from collections import Counter

            c = Counter(labels)
            purities.append(c.most_common(1)[0][1] / len(labels))
        if purities:
            # weighted purity
            total = sum(len(g2[v]) for v in g2)
            sum(
                (len(g2[v]) / total) * (Counter(g2[v]).most_common(1)[0][1] / len(g2[v]))
                for v in g2
            )
            # feature-only accuracy if we predict majority label per feature value
            feat_acc = 0.0
            for _v, labels in g2.items():
                maj = Counter(labels).most_common(1)[0][0]
                feat_acc += sum(1 for lab in labels if lab == maj)
            feat_acc /= n
            if feat_acc >= shortcut_acc_threshold:
                single_shortcuts.append((fk, float(feat_acc)))

    # length-label correlation
    length_corr = 0.0
    if text_key is not None:
        lengths = np.asarray([len(str(r.get(text_key, ""))) for r in records], dtype=float)
        # pearson corr between length and correctness
        correct = (preds == y).astype(float)
        if lengths.std() > 0 and correct.std() > 0:
            length_corr = float(np.corrcoef(lengths, correct)[0, 1])
            if math.isnan(length_corr):
                length_corr = 0.0

    flags: list[str] = []
    if imbalance >= 0.8:
        flags.append(f"severe_class_imbalance: majority class '{majority_class}' = {imbalance:.1%}")
    if model_acc - majority_acc < 0.05 and model_acc < 0.9:
        flags.append("model_not_better_than_majority_baseline")
    if single_shortcuts:
        for fk, acc in single_shortcuts:
            flags.append(f"shortcut_feature: '{fk}' alone predicts label with {acc:.1%} accuracy")
    for fk, gap in subgroup_gaps:
        if gap >= subgroup_gap_threshold:
            flags.append(f"subgroup_performance_gap: '{fk}' gap={gap:.2f}")
    if abs(length_corr) >= 0.3:
        flags.append(f"length_bias: corr(length, correctness)={length_corr:.2f}")

    return ShortcutReport(
        n=n,
        model_accuracy=model_acc,
        majority_baseline_accuracy=majority_acc,
        class_imbalance_ratio=imbalance,
        majority_class=majority_class,
        feature_mi=feature_mi,
        subgroup_gaps=subgroup_gaps,
        single_feature_shortcuts=single_shortcuts,
        length_label_corr=length_corr,
        flags=flags,
    )