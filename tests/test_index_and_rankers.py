import math
from pathlib import Path

from app.engine import SearchEngine
from app.index import InvertedIndex
from app.snippets import make_snippet
from tests.helpers import make_corpus


def build_engine(tmp_path: Path) -> SearchEngine:
    return SearchEngine.from_directory(make_corpus(tmp_path), cache_file=None)


def test_inverted_index_statistics():
    index = InvertedIndex.build([["a", "b", "a"], ["b", "c"]])
    assert index.n_docs == 2
    assert index.postings["a"] == {0: 2}
    assert index.df("b") == 2
    assert index.doc_len == [3, 2]
    assert math.isclose(index.avg_doc_len, 2.5)
    assert index.df("missing") == 0


def test_both_models_rank_the_right_category_first(tmp_path):
    engine = build_engine(tmp_path)
    queries = {
        "stock market shares": "business",
        "football match goal": "sport",
        "mobile phone internet": "tech",
    }
    for model in ("bm25", "tfidf"):
        for query, category in queries.items():
            top = engine.search(query, model=model, k=1)["results"][0]
            assert top["category"] == category, (model, query)


def test_scores_are_sorted_and_positive(tmp_path):
    engine = build_engine(tmp_path)
    for model in ("bm25", "tfidf"):
        scores = [r["score"] for r in engine.search("market shares rates", model=model, k=10)["results"]]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0 for s in scores)


def test_unknown_words_and_stop_word_only_queries_return_nothing(tmp_path):
    engine = build_engine(tmp_path)
    assert engine.search("zzzzqqq", k=5)["total"] == 0
    assert engine.search("the and of", k=5)["total"] == 0


def test_pagination_is_consistent(tmp_path):
    engine = build_engine(tmp_path)
    query = "market music match virus growth"
    full = engine.search(query, k=9)["results"]
    assert len(full) > 4
    page2 = engine.search(query, k=3, offset=3)["results"]
    assert [r["doc_id"] for r in page2] == [r["doc_id"] for r in full[3:6]]
    assert [r["rank"] for r in page2] == list(range(4, 4 + len(page2)))


def test_category_filter(tmp_path):
    engine = build_engine(tmp_path)
    results = engine.search("match", category="sport", k=10)["results"]
    assert results and all(r["category"] == "sport" for r in results)


def test_invalid_model_and_category_raise(tmp_path):
    engine = build_engine(tmp_path)
    for kwargs in ({"model": "nope"}, {"category": "nope"}):
        try:
            engine.search("market", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_index_cache_round_trip(tmp_path):
    bbc = make_corpus(tmp_path)
    cache = tmp_path / "index.pkl"
    first = SearchEngine.from_directory(bbc, cache_file=cache)
    assert cache.exists()
    second = SearchEngine.from_directory(bbc, cache_file=cache)
    assert first.search("market")["results"] == second.search("market")["results"]


def test_snippet_marks_query_words_by_stem():
    segments = make_snippet("Markets fell as the market closed.", {"market"})
    marked = [s["text"] for s in segments if s["match"]]
    assert marked == ["Markets", "market"]
    # the snippet is the original text from the first to the last word, so the final "." is not included
    assert "".join(s["text"] for s in segments) == "Markets fell as the market closed"
