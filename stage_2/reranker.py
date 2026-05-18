from __future__ import annotations


class BGEReranker:
    """BGE-reranker-v2-m3 cross-encoder. Lazy-loads FlagReranker on first use.

    Takes query + dict of {doc_id: text}, returns doc_ids sorted by
    cross-encoder score (highest first). Expensive but highly accurate.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "mps",
        top_k: int = 5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self.model_name, use_fp16=True)

    def rerank(
        self, query: str, passages: dict[str, str], top_k: int | None = None
    ) -> list[str]:
        """Score each (query, passage) pair; return doc_ids sorted by score."""
        self._load()
        k = top_k if top_k is not None else self.top_k
        doc_ids = list(passages.keys())
        pairs = [[query, text] for text in passages.values()]
        scores = self._model.compute_score(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [doc_ids[i] for i in ranked[: min(k, len(doc_ids))]]
