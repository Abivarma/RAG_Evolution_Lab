from __future__ import annotations

from typing import Any


def load_ragbench(split: str = "test", n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load RAGBench as a closed-corpus retrieval benchmark.

    Each item includes the 5 candidate passages and the set of relevant document indices.
    Relevant doc indices are derived from all_relevant_sentence_keys (format: "0b" = doc 0, sentence b).
    """
    from datasets import load_dataset

    ds = load_dataset("rungalileo/ragbench", "techqa", split=split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    results = []
    for i, row in enumerate(ds):
        # Build per-question corpus: doc index -> text
        docs = row.get("documents", [])
        passages = {str(j): text for j, text in enumerate(docs)}

        # Ground truth: which doc indices contain relevant sentences?
        relevant_keys = row.get("all_relevant_sentence_keys", [])
        relevant_doc_indices = {key.rstrip("abcdefghijklmnopqrstuvwxyz") for key in relevant_keys}
        # relevant_doc_indices is now a set of strings like {"0", "2"}

        results.append({
            "id": row.get("id", f"ragbench_{i}"),
            "query": row["question"],
            "passages": passages,          # dict: doc_index_str -> text
            "relevant_ids": relevant_doc_indices,  # set of doc index strings
            "answer": row.get("response", ""),
        })
    return results


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
