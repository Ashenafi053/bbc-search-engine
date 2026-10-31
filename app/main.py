"""FastAPI application: JSON API under /api plus the vanilla HTML/CSS/JS frontend from /static."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .engine import SearchEngine
from .evaluate import load_queries, run_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def create_app(
    data_dir: Path | None = None,
    queries_file: Path | None = None,
    cache_file: Path | None = None,
) -> FastAPI:
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    queries_file = Path(queries_file) if queries_file else config.QUERIES_FILE
    cache_file = Path(cache_file) if cache_file else config.CACHE_FILE

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = SearchEngine.from_directory(data_dir, cache_file)
        app.state.eval_report = None
        yield

    app = FastAPI(
        title="BBC News Search Engine",
        description="TF-IDF and BM25 search over the BBC News dataset.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def engine_of(request: Request) -> SearchEngine:
        return request.app.state.engine

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/stats")
    def stats(request: Request):
        return engine_of(request).stats()

    @app.get("/api/search")
    def search(
        request: Request,
        q: str = Query(..., min_length=1, max_length=200, description="Free-text query"),
        model: Literal["bm25", "tfidf"] = "bm25",
        k: int = Query(config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K),
        offset: int = Query(0, ge=0, le=1000),
        category: str | None = None,
    ):
        try:
            return engine_of(request).search(q, model=model, k=k, offset=offset, category=category or None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/doc/{doc_id}")
    def document(request: Request, doc_id: int):
        doc = engine_of(request).get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"doc_id": doc.doc_id, "name": doc.name, "category": doc.category,
                "title": doc.title, "body": doc.body}

    @app.get("/api/evaluate")
    def evaluate(request: Request):
        if request.app.state.eval_report is None:
            if not queries_file.exists():
                raise HTTPException(status_code=404, detail=f"Queries file not found: {queries_file}")
            request.app.state.eval_report = run_evaluation(engine_of(request), load_queries(queries_file))
        return request.app.state.eval_report

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(config.STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
    return app


app = create_app()
