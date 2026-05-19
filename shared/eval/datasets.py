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


def load_multihop_rag(split: str = "train", n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load MultiHop-RAG queries with their evidence passage texts.

    Dataset: yixuantt/MultiHopRAG (config: MultiHopRAG, only split: train)
    Each row has a query, a list of evidence dicts (each with title, url, fact text),
    and an answer. We build passages from the shared 609-article corpus, with
    relevant_ids = the URL keys of articles that appear in evidence_list.

    NOTE: This is an open-corpus benchmark (609 news articles). For retrieval
    evaluation, we use the corpus articles as passages and match by URL.
    For a quick closed-corpus eval (no Qdrant), we include evidence passages directly.
    """
    from datasets import load_dataset

    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    # Build corpus: url -> text (title + body)
    corpus_ds = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    corpus: dict[str, str] = {
        row["url"]: f"{row['title']}. {row['body']}"
        for row in corpus_ds
        if row.get("url")
    }

    results = []
    for i, row in enumerate(ds):
        evidence_list = row.get("evidence_list", [])
        # relevant_ids = URLs of articles needed to answer the question
        relevant_urls: set[str] = {e["url"] for e in evidence_list if e.get("url")}
        # Build per-query passages from corpus (all 609 articles)
        # For tractability we include only the relevant + a sample of negatives
        passages: dict[str, str] = {}
        for url in relevant_urls:
            if url in corpus:
                passages[url] = corpus[url]
        # Add up to 10 negatives from corpus for non-trivial retrieval
        neg_count = 0
        for url, text in corpus.items():
            if url not in relevant_urls and neg_count < 10:
                passages[url] = text
                neg_count += 1

        results.append({
            "id": f"multihop_{i}",
            "query": row.get("query", ""),
            "passages": passages,
            "relevant_ids": relevant_urls,
            "answer": row.get("answer", ""),
        })
    return results


def load_scifact(n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load BEIR SciFact as an open-corpus retrieval benchmark.

    SciFact is the correct benchmark for measuring Recall@K differences between
    BM25, dense, and hybrid retrieval. Properties:
    - 300 queries (scientific claims to verify)
    - 5,183 document corpus (scientific paper abstracts)
    - ~1.13 relevant docs per query — Recall@K genuinely varies by method
    - Small enough to run BM25 in-memory (no Qdrant required for closed eval)

    We load the full 5,183-doc corpus as passages per query so that all stages
    use the same closed-corpus protocol — this ensures fair comparison.
    Relevant docs are the papers cited as evidence for each claim.
    """
    from collections import defaultdict

    from datasets import load_dataset

    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")

    # Build full corpus dict once (5,183 docs — fits in memory)
    corpus: dict[str, str] = {
        row["_id"]: f"{row['title']}. {row['text']}"
        for row in corpus_ds
    }

    qrels: dict[str, dict[str, str]] = defaultdict(dict)
    for row in qrels_ds:
        # Normalise both IDs to strings for consistent matching with corpus dict keys
        qrels[str(row["query-id"])][str(row["corpus-id"])] = row["score"]

    # Only include queries that have qrels (300 of the 1,109 total queries)
    queries_with_qrels: list[dict] = [
        {"_id": row["_id"], "text": row["text"] or row["title"]}
        for row in queries_ds
        if row["_id"] in qrels
    ]
    if n_samples:
        queries_with_qrels = queries_with_qrels[:n_samples]

    results = []
    for q in queries_with_qrels:
        qid = q["_id"]
        query_qrels = qrels[qid]
        relevant_ids: set[str] = {cid for cid, score in query_qrels.items() if score > 0}

        results.append({
            "id": f"scifact_{qid}",
            "query": q["text"],
            # Pass full corpus — harnesses use BM25 pre-filtering internally
            # (retrieve_from_passages builds a fresh BM25 per query, which is fast)
            "passages": corpus,
            "relevant_ids": relevant_ids,
            "answer": "",
        })
    return results


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
    "ragbench": load_ragbench,       # closed-corpus ranking (MRR/nDCG) — S0-S3
    "scifact": load_scifact,         # open-corpus retrieval (Recall@K) — S0-S3
    "multihop": load_multihop_rag,   # multi-hop retrieval — S3, S4, S8
    "finder": load_finder,           # financial QA on 10-Ks — S5, S6
}
