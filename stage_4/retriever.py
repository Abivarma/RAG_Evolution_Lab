from __future__ import annotations

from collections import defaultdict
from typing import Any


def fuse_graph_and_vector(
    graph_ids: list[str],
    vector_ids: list[str],
    k: int = 60,
    top_k: int = 10,
) -> list[str]:
    """RRF fusion of graph-retrieved and vector-retrieved document IDs."""
    scores: dict[str, float] = defaultdict(float)
    for rank, doc_id in enumerate(graph_ids, start=1):
        scores[doc_id] += 1.0 / (k + rank)
    for rank, doc_id in enumerate(vector_ids, start=1):
        scores[doc_id] += 1.0 / (k + rank)
    fused = sorted(scores, key=lambda d: scores[d], reverse=True)
    return fused[:top_k]


def _extract_urls_from_text(text: str, passages: dict[str, str]) -> list[str]:
    """Extract passage IDs (URLs) that appear in LightRAG's response text."""
    found: list[str] = []
    seen: set[str] = set()
    for url in passages:
        if url in text and url not in seen:
            seen.add(url)
            found.append(url)
    return found


class GraphRetriever:
    """Stage 4: LightRAG graph search fused with Stage 2 vector retrieval via RRF.

    For closed-corpus benchmarks (MultiHopRAG passages dict):
    - LightRAG.query() finds entity/community paths relevant to the query
    - We parse which passage URLs appear in the LightRAG answer
    - Stage 2 pipeline ranks the same passages via dense+reranker
    - RRF fuses both ranked lists

    stage2_pipeline: Stage2Pipeline instance or None (graph-only mode).
    """

    def __init__(
        self,
        rag: Any,
        stage2_pipeline: Any | None,
        top_k_graph: int = 10,
        top_k_vector: int = 10,
        rrf_k: int = 60,
        query_mode: str = "hybrid",
    ) -> None:
        self.rag = rag
        self.stage2_pipeline = stage2_pipeline
        self.top_k_graph = top_k_graph
        self.top_k_vector = top_k_vector
        self.rrf_k = rrf_k
        self.query_mode = query_mode

    def _graph_search(self, query: str, passages: dict[str, str], top_k: int) -> list[str]:
        """Query LightRAG and extract passage IDs from its response."""
        try:
            from lightrag import QueryParam
            response = self.rag.query(query, param=QueryParam(mode=self.query_mode))
        except Exception:
            try:
                from lightrag import QueryParam
                response = self.rag.query(query, param=QueryParam(mode="naive"))
            except Exception:
                response = ""

        found = _extract_urls_from_text(response, passages)

        # Fallback: if LightRAG cited no passages from our set, use BM25
        if not found:
            from stage_0.retriever import BM25Retriever
            bm25 = BM25Retriever()
            found = bm25.retrieve_from_passages(query, passages, top_k=top_k)

        return found[:top_k]

    def retrieve_from_passages(
        self, query: str, passages: dict[str, str], top_k: int = 10
    ) -> list[str]:
        """Fuse graph search + vector retrieval over a closed-corpus passages dict."""
        graph_ids = self._graph_search(query, passages, top_k=self.top_k_graph)

        if self.stage2_pipeline is not None:
            vector_ids = self.stage2_pipeline.retrieve_from_passages(
                query, passages, top_k=self.top_k_vector
            )
        else:
            vector_ids = []

        if not vector_ids:
            return graph_ids[:top_k]
        if not graph_ids:
            return vector_ids[:top_k]

        return fuse_graph_and_vector(graph_ids, vector_ids, k=self.rrf_k, top_k=top_k)
