from __future__ import annotations

from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from stage_1.embedder import BGEEmbedder


def rrf_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_k: int | None = None,
) -> list[str]:
    """Reciprocal Rank Fusion over multiple ranked doc_id lists.

    score(d) = sum over lists: 1 / (k + rank(d, list))
    Docs absent from a list contribute 0 for that list.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(scores, key=lambda d: scores[d], reverse=True)
    return fused[:top_k] if top_k is not None else fused


class HybridRetriever:
    """BM25 + BGE-M3 dense retrieval fused via RRF.

    rank_passages(): closed-corpus use (RAGBench — no Qdrant needed).
    retrieve():       open-corpus use (Qdrant must be running).
    """

    def __init__(
        self,
        embedder: BGEEmbedder,
        rrf_k: int = 60,
        qdrant_client=None,
        collection: str = "",
    ) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.qdrant_client = qdrant_client
        self.collection = collection

    def _bm25_rank(self, query: str, passages: dict[str, str]) -> list[str]:
        doc_ids = list(passages.keys())
        tokenized = [text.lower().split() for text in passages.values()]
        bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        scores = bm25.get_scores(query.lower().split())
        return [doc_ids[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]

    # Truncate passages to ~6000 chars (~4096 tokens) to avoid OOM on large passages
    _MAX_PASSAGE_CHARS: int = 6000

    def _dense_rank(self, query: str, passages: dict[str, str]) -> list[str]:
        doc_ids = list(passages.keys())
        self.embedder._load()
        q_vec = np.array(self.embedder.embed_query(query))
        truncated = [v[: self._MAX_PASSAGE_CHARS] for v in passages.values()]
        p_vecs = self.embedder._model.encode(
            truncated, normalize_embeddings=True, batch_size=4, show_progress_bar=False
        )
        # Free MPS cache after encoding to prevent accumulation across repeated calls
        try:
            import torch
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        sims = p_vecs @ q_vec
        return [doc_ids[i] for i in sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)]

    def rank_passages(
        self, query: str, passages: dict[str, str], top_k: int = 10
    ) -> list[str]:
        """Hybrid rank a provided passages dict (closed-corpus)."""
        bm25_ranked = self._bm25_rank(query, passages)
        dense_ranked = self._dense_rank(query, passages)
        return rrf_fusion([bm25_ranked, dense_ranked], k=self.rrf_k, top_k=top_k)

    def retrieve(self, query: str, top_k: int = 50) -> list[str]:
        """Hybrid retrieve from Qdrant (open-corpus). Requires qdrant_client."""
        if self.qdrant_client is None:
            raise RuntimeError("qdrant_client required for open-corpus retrieve()")
        q_vec = self.embedder.embed_query(query)
        hits = self.qdrant_client.search(
            collection_name=self.collection, query_vector=q_vec, limit=top_k
        )
        seen: set[str] = set()
        doc_ids: list[str] = []
        for hit in hits:
            doc_id = hit.payload["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
        return doc_ids[:top_k]
