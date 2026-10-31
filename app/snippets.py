"""Build result snippets with the query words marked.

The snippet is returned as a list of {"text": ..., "match": bool} segments instead of HTML,
so the frontend can build DOM nodes with textContent and never needs innerHTML (no XSS risk).
"""
from __future__ import annotations

import re

from .preprocess import stem

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def make_snippet(text: str, query_stems: set[str], width: int = 40) -> list[dict]:
    text = " ".join(text.split())
    words = list(_WORD_RE.finditer(text))
    if not words:
        return []

    hits = {i for i, m in enumerate(words) if stem(m.group().lower()) in query_stems}
    if hits:
        # Choose the window of `width` words that contains the most query words.
        best = max(hits, key=lambda i: (sum(1 for j in range(i, min(i + width, len(words))) if j in hits), -i))
        start = max(0, best - 5)
    else:
        start = 0
    end = min(len(words), start + width)

    seg_start, seg_end = words[start].start(), words[end - 1].end()
    segments: list[dict] = []
    if start > 0:
        segments.append({"text": "\u2026 ", "match": False})
    pos = seg_start
    for i in range(start, end):
        if i in hits:
            m = words[i]
            if m.start() > pos:
                segments.append({"text": text[pos:m.start()], "match": False})
            segments.append({"text": m.group(), "match": True})
            pos = m.end()
    if pos < seg_end:
        segments.append({"text": text[pos:seg_end], "match": False})
    if end < len(words):
        segments.append({"text": " \u2026", "match": False})
    return segments
