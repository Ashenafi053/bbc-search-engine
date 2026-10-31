"""Load the BBC News dataset (or any folder of .txt files) into Document objects.

Expected layout (Greene and Cunningham, 2006):

    data/bbc/business/001.txt
    data/bbc/entertainment/001.txt
    data/bbc/politics/001.txt
    data/bbc/sport/001.txt
    data/bbc/tech/001.txt

The first line of each file is the article title; the rest is the body.
The folder name is used as the category. A flat folder of .txt files also works
(all documents then get the category "general"), so you can index your own materials later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IGNORED_NAMES = {"readme.txt"}


@dataclass(frozen=True)
class Document:
    doc_id: int
    name: str      # path relative to the dataset root, e.g. "business/001.txt"
    category: str
    title: str
    body: str


def _read_text(path: Path) -> str:
    """The BBC files are not always valid UTF-8, so fall back to Windows-1252 / Latin-1."""
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def load_documents(root: Path) -> list[Document]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}. Download the BBC News dataset and place it "
            "there (see the Dataset section of the README), or set BBC_DATA_DIR."
        )
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".txt" and p.name.lower() not in IGNORED_NAMES
    )
    if not paths:
        raise FileNotFoundError(f"No .txt files found under {root}.")

    documents: list[Document] = []
    for doc_id, path in enumerate(paths):
        category = path.parent.name if path.parent != root else "general"
        text = _read_text(path).replace("\r\n", "\n").strip()
        title, _, body = text.partition("\n")
        documents.append(
            Document(
                doc_id=doc_id,
                name=path.relative_to(root).as_posix(),
                category=category,
                title=title.strip() or path.stem,
                body=body.strip(),
            )
        )
    return documents
