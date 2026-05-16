from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 retrieval over a corpus of documents.

    Serializes to/from JSON (not binary formats) so index files are
    human-inspectable and safe to load from untrusted sources.
    Each document dict must have 'id' and 'abstract' keys; 'title' is optional.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def index(self, corpus: list[dict[str, Any]]) -> None:
        """Build BM25 index from corpus documents."""
        self._doc_ids = [doc["id"] for doc in corpus]
        self._tokenized_corpus = [
            self._tokenize(doc.get("title", "") + " " + doc.get("abstract", ""))
            for doc in corpus
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Return top_k doc_ids ranked by BM25 score."""
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve().")
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._doc_ids[i] for i in ranked[:top_k]]

    def save(self, path: Path) -> None:
        """Persist index state to JSON. BM25Okapi is rebuilt on load from the
        tokenized corpus stored in the JSON file — no binary serialization used."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self._doc_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }
        path.write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        """Load from JSON and rebuild the BM25 index in memory."""
        state = json.loads(Path(path).read_text())
        obj = cls(k1=state["k1"], b=state["b"])
        obj._doc_ids = state["doc_ids"]
        obj._tokenized_corpus = state["tokenized_corpus"]
        obj._bm25 = BM25Okapi(obj._tokenized_corpus, k1=obj.k1, b=obj.b)
        return obj
