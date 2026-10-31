import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from tests.helpers import make_corpus  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([
        {"query": "stock market shares", "category": "business"},
        {"query": "football match goal", "category": "sport"},
    ]))
    app = create_app(data_dir=make_corpus(tmp_path), queries_file=queries, cache_file=tmp_path / "index.pkl")
    with TestClient(app) as test_client:   # the context manager runs the startup (index loading)
        yield test_client


def test_health_and_stats(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    stats = client.get("/api/stats").json()
    assert stats["documents"] == 9
    assert stats["categories"] == {"business": 3, "sport": 3, "tech": 3}


def test_search_returns_ranked_results_with_snippets(client):
    data = client.get("/api/search", params={"q": "stock market shares"}).json()
    assert data["total"] >= 1
    first = data["results"][0]
    assert first["category"] == "business"
    assert any(segment["match"] for segment in first["snippet"])


def test_search_validation(client):
    assert client.get("/api/search").status_code == 422
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "model": "bogus"}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "k": 500}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "category": "nope"}).status_code == 400


def test_models_and_category_filter(client):
    for model in ("bm25", "tfidf"):
        data = client.get("/api/search", params={"q": "match", "model": model, "category": "sport"}).json()
        assert data["model"] == model
        assert all(r["category"] == "sport" for r in data["results"])


def test_document_endpoint(client):
    assert client.get("/api/doc/0").json()["title"]
    assert client.get("/api/doc/9999").status_code == 404


def test_evaluate_endpoint(client):
    report = client.get("/api/evaluate").json()
    assert report["n_queries"] == 2
    assert set(report["models"]) == {"bm25", "tfidf"}


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "BBC News Search" in page.text
    assert client.get("/static/app.js").status_code == 200
