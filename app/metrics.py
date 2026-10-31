"""Standard IR evaluation metrics for binary relevance.

`ranked` is a list of document ids in rank order; `relevant` is the set of relevant document ids.
"""
from __future__ import annotations

import math
from typing import Sequence


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def f1_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    p, r = precision_at_k(ranked, relevant, k), recall_at_k(ranked, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def average_precision(ranked: Sequence[int], relevant: set[int]) -> float:
    """AP = (sum of precision at each relevant rank) / (number of relevant documents)."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Binary-gain NDCG: DCG = sum rel_i / log2(i + 1), normalised by the ideal DCG."""
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
