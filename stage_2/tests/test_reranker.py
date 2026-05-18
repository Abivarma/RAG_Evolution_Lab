from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from stage_2.reranker import BGEReranker


def _make_reranker(scores: list[float]) -> BGEReranker:
    r = BGEReranker.__new__(BGEReranker)
    r.model_name = "BAAI/bge-reranker-v2-m3"
    r.device = "mps"
    r.top_k = 3
    r.batch_size = 32
    r.max_length = 512
    r.use_fp16 = False
    r._model = MagicMock()
    r._tokenizer = MagicMock()
    # Patch _score_pairs to return the provided scores directly
    r._score_pairs = MagicMock(return_value=scores)
    return r


def test_reranker_returns_top_k() -> None:
    reranker = _make_reranker([0.9, 0.1, 0.7, 0.3, 0.5])
    passages = {
        "doc_a": "Highly relevant.", "doc_b": "Unrelated.",
        "doc_c": "Somewhat relevant.", "doc_d": "Marginal.", "doc_e": "Moderate.",
    }
    ranked = reranker.rerank("query", passages, top_k=3)
    assert len(ranked) == 3
    assert ranked[0] == "doc_a"   # score 0.9
    assert ranked[1] == "doc_c"   # score 0.7


def test_reranker_top_k_larger_than_passages() -> None:
    reranker = _make_reranker([0.8, 0.6])
    passages = {"x": "text x", "y": "text y"}
    ranked = reranker.rerank("query", passages, top_k=10)
    assert len(ranked) == 2


def test_reranker_preserves_all_ids() -> None:
    reranker = _make_reranker([0.5, 0.3, 0.8])
    passages = {"a": "t", "b": "t", "c": "t"}
    ranked = reranker.rerank("q", passages, top_k=3)
    assert set(ranked) == {"a", "b", "c"}
