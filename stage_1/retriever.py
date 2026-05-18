from __future__ import annotations

from qdrant_client import QdrantClient

from stage_1.embedder import BGEEmbedder


class QdrantRetriever:
    """Dense retriever using Qdrant HNSW + BGE-M3 embeddings."""

    def __init__(
        self, collection: str, host: str, port: int, embedder: BGEEmbedder
    ) -> None:
        self.collection = collection
        self.client = QdrantClient(host=host, port=port)
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Return top_k unique doc_ids ranked by cosine similarity."""
        query_vec = self.embedder.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k * 3,  # over-fetch since multiple chunks per doc
        )
        seen: set[str] = set()
        doc_ids: list[str] = []
        for hit in hits:
            doc_id = hit.payload["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
            if len(doc_ids) >= top_k:
                break
        return doc_ids
