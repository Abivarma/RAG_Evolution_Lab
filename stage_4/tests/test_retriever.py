from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from stage_4.retriever import GraphRetriever, fuse_graph_and_vector


def test_fuse_deduplicates() -> None:
    graph_ids = ["doc_a", "doc_b", "doc_c"]
    vector_ids = ["doc_c", "doc_d", "doc_e"]
    fused = fuse_graph_and_vector(graph_ids, vector_ids, k=60, top_k=5)
    assert fused.count("doc_c") == 1
    assert len(fused) == 5


def test_fuse_top_ranked_in_both_wins() -> None:
    graph_ids = ["doc_winner", "doc_b", "doc_c"]
    vector_ids = ["doc_winner", "doc_d", "doc_e"]
    fused = fuse_graph_and_vector(graph_ids, vector_ids, k=60, top_k=3)
    assert fused[0] == "doc_winner"


def test_fuse_top_k_respected() -> None:
    graph_ids = [f"g{i}" for i in range(10)]
    vector_ids = [f"v{i}" for i in range(10)]
    fused = fuse_graph_and_vector(graph_ids, vector_ids, k=60, top_k=5)
    assert len(fused) == 5


def test_retriever_parses_urls_from_response() -> None:
    mock_rag = MagicMock()
    mock_rag.query.return_value = (
        "Based on https://example.com/article1 and https://example.com/article2, "
        "the answer involves the topic in question."
    )
    passages = {
        "https://example.com/article1": "Article 1 text.",
        "https://example.com/article2": "Article 2 text.",
        "https://example.com/article3": "Unrelated article.",
    }
    retriever = GraphRetriever(rag=mock_rag, stage2_pipeline=None, top_k_graph=5, top_k_vector=5, rrf_k=60)
    results = retriever.retrieve_from_passages("some multi-hop query", passages, top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    # Both cited URLs should be in results
    assert "https://example.com/article1" in results
    assert "https://example.com/article2" in results


def test_retriever_fallback_to_bm25_when_no_urls_cited() -> None:
    mock_rag = MagicMock()
    mock_rag.query.return_value = "I cannot find relevant information."
    passages = {"url_a": "biomaterials tissue engineering", "url_b": "machine learning neural networks"}
    retriever = GraphRetriever(rag=mock_rag, stage2_pipeline=None, top_k_graph=5, top_k_vector=5, rrf_k=60)
    results = retriever.retrieve_from_passages("biomaterials inductive properties", passages, top_k=2)
    assert len(results) <= 2
    assert all(r in passages for r in results)
