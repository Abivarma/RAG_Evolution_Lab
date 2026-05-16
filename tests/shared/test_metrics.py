from __future__ import annotations

import pytest
from shared.eval.metrics import recall_at_k, mrr, ndcg_at_k, hit_rate


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b", "c"}, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_zero():
    assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == pytest.approx(0.0)


def test_recall_at_k_truncates_to_k():
    # Only top-2 retrieved: [a, b]. Relevant = {a,b,c,d}. Hits=2, |relevant|=4
    assert recall_at_k(["a", "b", "c", "d"], {"a", "b", "c", "d"}, k=2) == pytest.approx(2 / 4)


def test_mrr_first_hit():
    assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_hit():
    assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)


def test_mrr_no_hit():
    assert mrr(["x", "y", "z"], {"a"}) == pytest.approx(0.0)


def test_ndcg_perfect():
    retrieved = ["a", "b", "c"]
    relevance = {"a": 2, "b": 1, "c": 1}
    assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)


def test_hit_rate_found():
    assert hit_rate(["x", "a", "y"], {"a"}, k=3) is True


def test_hit_rate_not_found():
    assert hit_rate(["x", "y", "z"], {"a"}, k=3) is False
