"""SearchEngine: ties together the documents, the inverted index, the rankers and the snippets."""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path

from . import config
from .index import InvertedIndex
from .loader import Document, load_documents
from .preprocess import preprocess
from .ranker import Bm25Ranker, TfidfRanker
from .snippets import make_snippet

log = logging.getLogger(__name__)

CACHE_VERSION = 1
MODELS = ("bm25", "tfidf")


def _fingerprint(docs: list[Document]) -> str:
    """Cheap content signature: if the dataset changes, the cached index is rebuilt."""
    h = hashlib.sha1()
    for d in docs:
        h.update(f"{d.name}:{len(d.title)}:{len(d.body)}\n".encode())
    return h.hexdigest()


def _load_or_build_index(docs: list[Document], cache_file: Path | None) -> InvertedIndex:
    fingerprint = _fingerprint(docs)
    if cache_file and Path(cache_file).exists():
        try:
            with open(cache_file, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") == CACHE_VERSION and payload.get("fingerprint") == fingerprint:
                log.info("Loaded index from cache %s", cache_file)
                return payload["index"]
        except Exception:  # corrupt or incompatible cache: just rebuild
            log.warning("Index cache unreadable, rebuilding", exc_info=True)

    log.info("Building index for %d documents...", len(docs))
    started = time.perf_counter()
    index = InvertedIndex.build(preprocess(f"{d.title}\n{d.body}") for d in docs)
    log.info("Index built in %.1fs (%d terms)", time.perf_counter() - started, index.vocab_size)

    if cache_file:
        try:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as fh:
                pickle.dump({"version": CACHE_VERSION, "fingerprint": fingerprint, "index": index}, fh)
        except OSError:
            log.warning("Could not write index cache", exc_info=True)
    return index


class SearchEngine:
    def __init__(self, docs: list[Document], index: InvertedIndex) -> None:
        self.docs = docs
        self.index = index
        self.rankers = {
            "bm25": Bm25Ranker(index, config.BM25_K1, config.BM25_B),
            "tfidf": TfidfRanker(index),
        }
        self.name_to_id = {d.name: d.doc_id for d in docs}
        self.category_ids: dict[str, set[int]] = {}
        for d in docs:
            self.category_ids.setdefault(d.category, set()).add(d.doc_id)

    @classmethod
    def from_directory(cls, root: Path, cache_file: Path | None = None) -> "SearchEngine":
        docs = load_documents(Path(root))
        return cls(docs, _load_or_build_index(docs, cache_file))

    def get_document(self, doc_id: int) -> Document | None:
        return self.docs[doc_id] if 0 <= doc_id < len(self.docs) else None

    def stats(self) -> dict:
        return {
            "documents": self.index.n_docs,
            "vocabulary": self.index.vocab_size,
            "avg_doc_length": round(self.index.avg_doc_len, 1),
            "categories": {c: len(ids) for c, ids in sorted(self.category_ids.items())},
        }

    def search(
        self,
        query: str,
        model: str = "bm25",
        k: int = 10,
        offset: int = 0,
        category: str | None = None,
    ) -> dict:
        if model not in self.rankers:
            raise ValueError(f"Unknown model '{model}'. Choose one of: {', '.join(MODELS)}.")
        if category and category not in self.category_ids:
            raise ValueError(f"Unknown category '{category}'.")

        started = time.perf_counter()
        terms = preprocess(query)
        allowed = self.category_ids[category] if category else None
        total, ranked = self.rankers[model].rank(terms, k, offset, allowed)

        stems = set(terms)
        results = []
        for position, (doc_id, score) in enumerate(ranked, start=offset + 1):
            doc = self.docs[doc_id]
            results.append(
                {
                    "rank": position,
                    "doc_id": doc_id,
                    "title": doc.title,
                    "category": doc.category,
                    "name": doc.name,
                    "score": round(score, 4),
                    "snippet": make_snippet(doc.body or doc.title, stems),
                }
            )
        return {
            "query": query,
            "model": model,
            "terms": sorted(stems),
            "total": total,
            "offset": offset,
            "took_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": results,
        }
