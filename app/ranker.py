"""Ranking models: TF-IDF with cosine similarity, and Okapi BM25.

Both rankers score documents term-at-a-time over the postings lists, so only
documents that contain at least one query term are ever touched.
"""
from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

from .index import InvertedIndex


class Ranker(ABC):
    name = "base"

    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    @abstractmethod
    def score(self, terms: list[str]) -> dict[int, float]:
        """Return {doc_id: score} for every document matching at least one query term."""

    def rank(
        self,
        terms: list[str],
        k: int = 10,
        offset: int = 0,
        allowed: set[int] | None = None,
    ) -> tuple[int, list[tuple[int, float]]]:
        """Return (total matches, ranked slice [offset, offset + k)).

        Ties are broken by document id so results are deterministic.
        `allowed` optionally restricts results to a set of documents (e.g. one category).
        """
        scores = self.score(terms)
        if allowed is not None:
            scores = {d: s for d, s in scores.items() if d in allowed}
        top = heapq.nlargest(offset + k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return len(scores), top[offset: offset + k]


class TfidfRanker(Ranker):
    """Vector space model.

    Weight of term t in document d:   w(t, d) = (1 + log tf) * log(N / df)
    The query is weighted the same way, and documents are ranked by cosine similarity.
    """

    name = "tfidf"

    def __init__(self, index: InvertedIndex) -> None:
        super().__init__(index)
        n = index.n_docs
        self.idf = {term: math.log(n / len(plist)) for term, plist in index.postings.items()}
        squared = defaultdict(float)
        for term, plist in index.postings.items():
            idf = self.idf[term]
            for doc_id, tf in plist.items():
                weight = (1.0 + math.log(tf)) * idf
                squared[doc_id] += weight * weight
        self.doc_norm = [math.sqrt(squared.get(d, 0.0)) for d in range(n)]

    def score(self, terms: list[str]) -> dict[int, float]:
        query_weights: dict[str, float] = {}
        for term, count in Counter(terms).items():
            if term in self.index.postings:
                query_weights[term] = (1.0 + math.log(count)) * self.idf[term]
        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
        if query_norm == 0.0:
            return {}

        dot: dict[int, float] = defaultdict(float)
        for term, q_weight in query_weights.items():
            idf = self.idf[term]
            for doc_id, tf in self.index.postings[term].items():
                dot[doc_id] += q_weight * (1.0 + math.log(tf)) * idf
        return {
            doc_id: value / (self.doc_norm[doc_id] * query_norm)
            for doc_id, value in dot.items()
            if self.doc_norm[doc_id] > 0.0
        }


class Bm25Ranker(Ranker):
    """Okapi BM25 with the non-negative IDF used by Lucene:

        idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
        score(d, q) = sum over query terms of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    """

    name = "bm25"

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(index)
        self.k1 = k1
        self.b = b
        n = index.n_docs
        self.idf = {
            term: math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in index.postings.items()
        }

    def score(self, terms: list[str]) -> dict[int, float]:
        avgdl = self.index.avg_doc_len or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            plist = self.index.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_id, tf in plist.items():
                length_norm = 1.0 - self.b + self.b * self.index.doc_len[doc_id] / avgdl
                scores[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * length_norm)
        return dict(scores)
