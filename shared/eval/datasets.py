from __future__ import annotations

from typing import Any


def load_ragbench(split: str = "test", n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load RAGBench queries. Returns list of {id, query, relevant_ids, answer}."""
    from datasets import load_dataset

    ds = load_dataset("rungalileo/ragbench", "techqa", split=split, trust_remote_code=True)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    return [
        {
            "id": f"ragbench_{i}",
            "query": row["question"],
            "relevant_ids": row.get("documents", []),
            "answer": row.get("response", ""),
        }
        for i, row in enumerate(ds)
    ]


def load_multihop_rag(split: str = "test", n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load MultiHop-RAG queries."""
    from datasets import load_dataset

    ds = load_dataset("yixuantt/MultiHopRAG", split=split, trust_remote_code=True)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    return [
        {
            "id": f"multihop_{i}",
            "query": row.get("query", row.get("question", "")),
            "relevant_ids": row.get("evidence_list", []),
            "answer": row.get("answer", ""),
        }
        for i, row in enumerate(ds)
    ]


def load_finder(split: str = "test", n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load FinDER (SEC 10-K QA) queries."""
    from datasets import load_dataset

    ds = load_dataset("Linq-AI-Research/FinDER", split=split, trust_remote_code=True)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    return [
        {
            "id": f"finder_{i}",
            "query": row.get("question", ""),
            "relevant_ids": row.get("context_ids", []),
            "answer": row.get("answer", ""),
        }
        for i, row in enumerate(ds)
    ]


BENCHMARK_LOADERS = {
    "ragbench": load_ragbench,
    "multihop": load_multihop_rag,
    "finder": load_finder,
}
