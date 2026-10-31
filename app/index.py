"""Inverted index: term -> {doc_id: term frequency}, plus document lengths."""
from __future__ import annotations

from collections import Counter
from typing import Iterable


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, dict[int, int]] = {}
        self.doc_len: list[int] = []

    @classmethod
    def build(cls, docs_tokens: Iterable[list[str]]) -> "InvertedIndex":
        """Build the index from one token list per document (doc_id = position in the iterable)."""
        index = cls()
        for doc_id, tokens in enumerate(docs_tokens):
            index.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                index.postings.setdefault(term, {})[doc_id] = tf
        return index

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def vocab_size(self) -> int:
        return len(self.postings)

    @property
    def avg_doc_len(self) -> float:
        return sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

    def df(self, term: str) -> int:
        """Document frequency: number of documents containing the term."""
        return len(self.postings.get(term, ()))
