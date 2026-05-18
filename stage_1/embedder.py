from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 0) -> list[str]:
    """Split text into fixed-size word chunks. No sentence awareness — deliberately naive."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


class BGEEmbedder:
    """BAAI/bge-m3 dense embedder. Lazy-loads model on first use."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "mps") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        self._load()
        vecs = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        )
        return vecs.tolist()

    def embed_query(self, query: str) -> list[float]:
        self._load()
        return self._model.encode([query], normalize_embeddings=True)[0].tolist()

    def embed_and_rank_passages(
        self, query: str, passages: dict[str, str], top_k: int = 10, batch_size: int = 4
    ) -> list[str]:
        """Rank passages by cosine similarity to query. For closed-corpus benchmarks.

        passages: dict mapping doc_id -> text
        batch_size: kept small to avoid MPS OOM on repeated calls.
        Returns: list of doc_ids ranked by cosine similarity (highest first)
        """
        import numpy as np

        self._load()
        doc_ids = list(passages.keys())
        texts = list(passages.values())

        query_vec = self._model.encode(
            [query], normalize_embeddings=True, batch_size=1, show_progress_bar=False
        )[0]
        passage_vecs = self._model.encode(
            texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
        )

        # Free MPS cache after encoding to prevent accumulation across many queries
        try:
            import torch
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

        # Cosine similarity (dot product since vectors are normalized)
        scores = passage_vecs @ query_vec
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [doc_ids[i] for i in ranked_indices[:top_k]]
