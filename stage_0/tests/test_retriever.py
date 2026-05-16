from __future__ import annotations

from pathlib import Path

import pytest

from stage_0.retriever import BM25Retriever

SMALL_CORPUS = [
    {"id": "doc1", "title": "neural network transformers attention", "abstract": "We study self-attention mechanisms in transformers."},
    {"id": "doc2", "title": "retrieval augmented generation RAG", "abstract": "RAG combines retrieval with LLM generation."},
    {"id": "doc3", "title": "BM25 keyword search baseline", "abstract": "BM25 is a classical keyword retrieval algorithm."},
]


@pytest.fixture
def retriever() -> BM25Retriever:
    r = BM25Retriever(k1=1.5, b=0.75)
    r.index(SMALL_CORPUS)
    return r


def test_retriever_returns_ids(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("transformers attention", top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(r, str) for r in results)


def test_retriever_ranks_by_relevance(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("BM25 keyword retrieval", top_k=3)
    assert results[0] == "doc3", f"Expected doc3 first, got {results}"


def test_retriever_handles_unknown_terms(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("xyzzy frobnicator", top_k=3)
    assert isinstance(results, list)


def test_retriever_save_load(retriever: BM25Retriever, tmp_path: Path) -> None:
    index_path = tmp_path / "bm25_index.json"
    retriever.save(index_path)
    loaded = BM25Retriever.load(index_path)
    assert retriever.retrieve("RAG generation", top_k=2) == loaded.retrieve("RAG generation", top_k=2)


def test_retriever_top_k_respected(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("neural network", top_k=1)
    assert len(results) == 1


def test_retrieve_from_passages() -> None:
    r = BM25Retriever(k1=1.5, b=0.75)
    passages = {
        "0": "neural network attention transformers deep learning",
        "1": "BM25 keyword retrieval classical algorithm",
        "2": "RAG combines retrieval with language models",
    }
    results = r.retrieve_from_passages("keyword search BM25", passages, top_k=3)
    assert results[0] == "1", f"Expected '1' first, got {results}"
