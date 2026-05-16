from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit. 0 if no hit."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalized DCG at k. relevance maps doc_id to integer grade (0, 1, or 2)."""

    def dcg(ranked: list[str], grades: dict[str, int], cutoff: int) -> float:
        return sum(
            (2 ** grades.get(doc_id, 0) - 1) / math.log2(i + 2)
            for i, doc_id in enumerate(ranked[:cutoff])
        )

    actual = dcg(retrieved, relevance, k)
    ideal = dcg(sorted(relevance, key=relevance.get, reverse=True), relevance, k)
    return actual / ideal if ideal > 0 else 0.0


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> bool:
    """True if any relevant doc appears in top-k."""
    return any(doc_id in relevant for doc_id in retrieved[:k])


def compute_retrieval_metrics(
    retrieved: list[str],
    relevant: set[str],
    relevance_grades: dict[str, int] | None = None,
) -> dict[str, float]:
    """Full retrieval metrics bundle for one query."""
    grades = relevance_grades or {doc_id: 1 for doc_id in relevant}
    return {
        "recall_at_5": recall_at_k(retrieved, relevant, k=5),
        "recall_at_10": recall_at_k(retrieved, relevant, k=10),
        "recall_at_20": recall_at_k(retrieved, relevant, k=20),
        "mrr": mrr(retrieved, relevant),
        "ndcg_at_10": ndcg_at_k(retrieved, grades, k=10),
        "hit_rate_10": float(hit_rate(retrieved, relevant, k=10)),
    }
