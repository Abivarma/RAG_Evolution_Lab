from __future__ import annotations

import pytest
from stage_2.hybrid_retriever import rrf_fusion, HybridRetriever


def test_rrf_fusion_single_list() -> None:
    ranked = ["doc1", "doc2", "doc3"]
    fused = rrf_fusion([ranked], k=60)
    assert fused[0] == "doc1"
    assert fused[1] == "doc2"


def test_rrf_fusion_two_lists_agree() -> None:
    list1 = ["doc_a", "doc_b", "doc_c"]
    list2 = ["doc_a", "doc_c", "doc_b"]
    fused = rrf_fusion([list1, list2], k=60)
    assert fused[0] == "doc_a"


def test_rrf_fusion_disagreement_resolved_by_sum() -> None:
    # doc_z is rank 2 in both lists, doc_a only in list1, doc_b only in list2
    # doc_z should win: both lists contribute, while others have only one contribution
    list1 = ["doc_a", "doc_z", "doc_c"]
    list2 = ["doc_b", "doc_z", "doc_d"]
    fused = rrf_fusion([list1, list2], k=60)
    assert fused[0] == "doc_z"


def test_rrf_fusion_top_k_limits_output() -> None:
    list1 = [f"doc{i}" for i in range(20)]
    fused = rrf_fusion([list1], k=60, top_k=5)
    assert len(fused) == 5


def test_rrf_fusion_merges_unique_ids() -> None:
    fused = rrf_fusion([["doc_a", "doc_b"], ["doc_c", "doc_d"]], k=60)
    assert set(fused) == {"doc_a", "doc_b", "doc_c", "doc_d"}


def test_hybrid_retriever_closed_corpus() -> None:
    from unittest.mock import MagicMock
    import numpy as np

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
    mock_embedder._model = MagicMock()
    mock_embedder._model.encode.return_value = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])

    retriever = HybridRetriever(embedder=mock_embedder, rrf_k=60)
    passages = {
        "0": "BM25 keyword retrieval exact match query",
        "1": "unrelated content about something else entirely",
        "2": "partial keyword match here",
    }
    ranked = retriever.rank_passages("BM25 keyword retrieval", passages, top_k=3)
    assert ranked[0] == "0"
    assert len(ranked) == 3
