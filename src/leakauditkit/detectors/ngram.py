"""N-gram overlap detector (verbatim / near-verbatim contamination)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .base import BaseDetector, ContaminationMatch, DetectionResult

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return _WORD_RE.findall(text)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class NgramDetector(BaseDetector):
    """Detects eval items that share n-grams with train items.

    Scoring: Jaccard similarity over the set of n-grams of the two texts,
    optionally weighted by document frequency (rare n-grams count more).
    """

    name = "ngram"

    def __init__(
        self,
        n: int = 5,
        threshold: float = 0.3,
        min_shared: int = 3,
        use_idf: bool = False,
        lowercase: bool = True,
    ) -> None:
        super().__init__(threshold=threshold)
        if n < 1:
            raise ValueError("n must be >= 1")
        if min_shared < 1:
            raise ValueError("min_shared must be >= 1")
        self.n = n
        self.min_shared = min_shared
        self.use_idf = use_idf
        self.lowercase = lowercase

    def _text_of(self, item: dict[str, Any]) -> str:
        for key in ("text", "content", "passage", "document"):
            if key in item and item[key] is not None:
                return str(item[key])
        return ""

    def _grams_of(self, item: dict[str, Any]) -> set[tuple[str, ...]]:
        return set(ngrams(tokenize(self._text_of(item), self.lowercase), self.n))

    def detect(
        self,
        train: Iterable[dict[str, Any]],
        eval_: Iterable[dict[str, Any]],
    ) -> DetectionResult:
        train_list = list(train)
        eval_list = list(eval_)

        train_grams: list[set[tuple[str, ...]]] = [self._grams_of(t) for t in train_list]
        eval_grams: list[set[tuple[str, ...]]] = [self._grams_of(e) for e in eval_list]

        # document frequency over train corpus for IDF weighting
        df: Counter[tuple[str, ...]] = Counter()
        for gs in train_grams:
            for g in gs:
                df[g] += 1
        n_train = len(train_list)

        def idf(g: tuple[str, ...]) -> float:
            if not self.use_idf:
                return 1.0
            return 1.0 / (1.0 + df.get(g, 0))

        # inverted index: n-gram -> list of train indices (performance)
        inv: dict[tuple[str, ...], list[int]] = {}
        for ti, tg in enumerate(train_grams):
            for g in tg:
                inv.setdefault(g, []).append(ti)

        matches: list[ContaminationMatch] = []
        for ei, eg in enumerate(eval_grams):
            if not eg:
                continue
            # candidate train indices sharing at least one n-gram
            cand_counter: Counter[int] = Counter()
            for g in eg:
                for ti in inv.get(g, ()):
                    cand_counter[ti] += 1
            for ti in cand_counter:
                if cand_counter[ti] < self.min_shared:
                    continue
                tg = train_grams[ti]
                if not tg:
                    continue
                shared = eg & tg
                if len(shared) < self.min_shared:
                    continue
                # weighted jaccard
                inter_w = sum(idf(g) for g in shared)
                union_w = sum(idf(g) for g in (eg | tg))
                score = inter_w / union_w if union_w > 0 else 0.0
                if score >= self.threshold:
                    evidence = " | ".join(" ".join(g) for g in list(shared)[:3])
                    matches.append(
                        ContaminationMatch(
                            train_id=str(train_list[ti].get("id", f"train_{ti}")),
                            eval_id=str(eval_list[ei].get("id", f"eval_{ei}")),
                            score=round(float(score), 4),
                            method=f"ngram-{self.n}",
                            evidence=evidence,
                            meta={"shared_count": len(shared)},
                        )
                    )

        # keep best (highest score) match per eval item for reporting clarity
        best_by_eval: dict[str, ContaminationMatch] = {}
        for m in matches:
            prev = best_by_eval.get(m.eval_id)
            if prev is None or m.score > prev.score:
                best_by_eval[m.eval_id] = m
        deduped = list(best_by_eval.values())

        return DetectionResult(
            detector=self.name,
            n_train=n_train,
            n_eval=len(eval_list),
            threshold=self.threshold,
            matches=deduped,
            stats={
                "n": self.n,
                "min_shared": self.min_shared,
                "use_idf": self.use_idf,
                "raw_matches": len(matches),
                "unique_eval_contaminated": len(deduped),
            },
        )