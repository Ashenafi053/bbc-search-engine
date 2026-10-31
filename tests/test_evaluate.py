from app.engine import SearchEngine
from app.evaluate import run_evaluation, to_markdown
from tests.helpers import make_corpus

QUERIES = [
    {"query": "stock market shares", "category": "business"},
    {"query": "football match goal", "category": "sport"},
    {"query": "computer virus security", "category": "tech"},
]


def test_evaluation_report_shape_and_sanity(tmp_path):
    engine = SearchEngine.from_directory(make_corpus(tmp_path), cache_file=None)
    report = run_evaluation(engine, QUERIES, depth=50)
    assert report["n_queries"] == 3
    for model in ("bm25", "tfidf"):
        mean = report["models"][model]["mean"]
        assert set(mean) == set(report["metrics"])
        assert all(0.0 <= v <= 1.0 for v in mean.values())
        assert mean["MRR"] == 1.0          # the right category is always ranked first on this corpus
        assert 0.0 < mean["MAP"] <= 1.0
    assert "| MAP |" in to_markdown(report)


def test_explicit_relevance_judgments_override_category(tmp_path):
    engine = SearchEngine.from_directory(make_corpus(tmp_path), cache_file=None)
    queries = [{"query": "interest rates inflation", "relevant": ["business/002.txt"]}]
    report = run_evaluation(engine, queries, depth=50)
    assert report["relevance"] == "explicit judgments"
    assert report["models"]["bm25"]["mean"]["MRR"] == 1.0
