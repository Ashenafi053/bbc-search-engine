"""Offline evaluation of both ranking models.

Queries live in data/queries.json. Each query has a text and either
  * "category": every document in that BBC category is treated as relevant, or
  * "relevant": an explicit list of document names (e.g. ["business/001.txt"]) for hand-made judgments.

Run from the project root:

    python -m app.evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import config
from .engine import MODELS, SearchEngine
from .metrics import (average_precision, f1_at_k, mean, ndcg_at_k, precision_at_k,
                      recall_at_k, reciprocal_rank)
from .preprocess import preprocess

METRICS = ["P@5", "P@10", "R@10", "F1@10", "MAP", "MRR", "NDCG@10"]


def load_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def relevant_set(engine: SearchEngine, query: dict) -> set[int]:
    if "relevant" in query:
        return {engine.name_to_id[n] for n in query["relevant"] if n in engine.name_to_id}
    return set(engine.category_ids.get(query.get("category", ""), set()))


def score_ranking(ranked: list[int], relevant: set[int]) -> dict[str, float]:
    return {
        "P@5": precision_at_k(ranked, relevant, 5),
        "P@10": precision_at_k(ranked, relevant, 10),
        "R@10": recall_at_k(ranked, relevant, 10),
        "F1@10": f1_at_k(ranked, relevant, 10),
        "AP": average_precision(ranked, relevant),
        "RR": reciprocal_rank(ranked, relevant),
        "NDCG@10": ndcg_at_k(ranked, relevant, 10),
    }


def evaluate_model(engine: SearchEngine, model: str, queries: list[dict], depth: int) -> dict:
    per_query = []
    for q in queries:
        relevant = relevant_set(engine, q)
        if not relevant:
            logging.getLogger(__name__).warning("Skipping query with no relevant documents: %r", q["query"])
            continue
        _, ranked = engine.rankers[model].rank(preprocess(q["query"]), k=depth)
        scores = score_ranking([doc_id for doc_id, _ in ranked], relevant)
        per_query.append({"query": q["query"], "relevant": len(relevant), **{k: round(v, 4) for k, v in scores.items()}})

    summary = {m: mean([r[m] for r in per_query]) for m in ("P@5", "P@10", "R@10", "F1@10", "NDCG@10")}
    summary["MAP"] = mean([r["AP"] for r in per_query])
    summary["MRR"] = mean([r["RR"] for r in per_query])
    return {"mean": {m: round(summary[m], 4) for m in METRICS}, "per_query": per_query}


def run_evaluation(engine: SearchEngine, queries: list[dict], depth: int = config.EVAL_DEPTH) -> dict:
    models = {m: evaluate_model(engine, m, queries, depth) for m in MODELS}
    n_queries = len(next(iter(models.values()))["per_query"])
    uses_categories = all("relevant" not in q for q in queries)
    return {
        "n_queries": n_queries,
        "depth": depth,
        "relevance": "category" if uses_categories else "explicit judgments",
        "metrics": METRICS,
        "models": models,
    }


def to_markdown(report: dict) -> str:
    names = {"bm25": "BM25", "tfidf": "TF-IDF + Cosine"}
    models = list(report["models"])
    lines = [
        f"Evaluation on {report['n_queries']} queries (ranking depth {report['depth']}, relevance: {report['relevance']}).",
        "",
        "| Metric | " + " | ".join(names[m] for m in models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for metric in report["metrics"]:
        lines.append(f"| {metric} | " + " | ".join(f"{report['models'][m]['mean'][metric]:.4f}" for m in models) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF and BM25 on the query set.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--queries", type=Path, default=config.QUERIES_FILE)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = SearchEngine.from_directory(args.data_dir, config.CACHE_FILE)
    report = run_evaluation(engine, load_queries(args.queries))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.out / "evaluation.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nSaved to {args.out / 'evaluation.json'} and {args.out / 'evaluation.md'}")


if __name__ == "__main__":
    main()
