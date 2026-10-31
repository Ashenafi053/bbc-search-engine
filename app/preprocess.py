"""Text preprocessing shared by indexing and querying.

Using one pipeline for both documents and queries guarantees that query terms
can match index terms: lowercase -> tokenize -> remove stop words -> stem.
"""
from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem import PorterStemmer

STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are aren as at be because been before being
    below between both but by can cannot could couldn d did didn do does doesn doing don down during
    each few for from further had hadn has hasn have haven having he her here hers herself him himself
    his how i if in into is isn it its itself just ll m me mightn more most mustn my myself needn no
    nor not now o of off on once only or other our ours ourselves out over own re s same shan she
    should shouldn so some such t than that the their theirs them themselves then there these they
    this those through to too under until up ve very was wasn we were weren what when where which
    while who whom why will with won would wouldn y you your yours yourself yourselves
    said says say also mr mrs ms
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """Porter-stem a single lowercase word (cached, because vocabularies repeat a lot)."""
    return _stemmer.stem(word)


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def preprocess(text: str) -> list[str]:
    """Full pipeline: tokenize, drop stop words and 1-character tokens, then stem."""
    return [stem(t) for t in tokenize(text) if len(t) > 1 and t not in STOP_WORDS]
