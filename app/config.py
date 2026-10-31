"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("BBC_DATA_DIR", BASE_DIR / "data" / "bbc"))
QUERIES_FILE = Path(os.getenv("QUERIES_FILE", BASE_DIR / "data" / "queries.json"))
CACHE_FILE = Path(os.getenv("INDEX_CACHE", BASE_DIR / "data" / "index.pkl"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
STATIC_DIR = BASE_DIR / "static"

# BM25 parameters (Robertson and Zaragoza, 2009): k1 in [1.2, 2.0], b = 0.75 are common defaults.
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

DEFAULT_TOP_K = 10
MAX_TOP_K = 50
EVAL_DEPTH = 1000  # how deep each ranking is evaluated (needed for a meaningful AP/MAP)
