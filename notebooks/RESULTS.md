# RAG Evolution Lab — Results Summary
Generated: 2026-05-18 09:05 UTC
Benchmark: RAGBench TechQA | Hardware: MacBook M3 36GB

| Stage | Architecture | N | Recall@10 | MRR | nDCG@10 | p99 (ms) | Δ Recall@10 | Δ MRR |
|-------|-------------|---|-----------|-----|---------|----------|-------------|-------|
| S0 | S0 BM25 | 50 | 0.8000 | 0.4877 | 0.5655 | 1 | baseline | baseline |
| S1 | S1 Naive RAG (BGE-M3) | 50 | 0.8000 | 0.6030 | 0.6417 | 334833 | +0.0000 | +0.1153 |
| S2 | S2 Advanced RAG | 50 | 0.8000 | 0.6267 | 0.6621 | 32884 | +0.0000 | +0.0237 |

## Stage 2 Ablation (50 queries)

| Config | R@10 | MRR | nDCG@10 | p99ms | Key finding |
|--------|------|-----|---------|-------|-------------|
| all | 0.8000 | 0.6267 | 0.6621 | 32884 | Full pipeline |
| no_reranker | 0.8000 | 0.5497 | 0.6006 | 30718 | **Reranker = +7.7pp MRR** |
| no_hybrid | 0.8000 | 0.6267 | 0.6621 | 32876 | Hybrid: 0pp on closed-corpus |
| no_rewriting | 0.8000 | 0.6267 | 0.6621 | 16816 | Rewriting: 0pp (closed) |
| no_sem_chunk | 0.8000 | 0.6267 | 0.6621 | 32802 | Sem-chunk: 0pp (closed) |
| dense_only | 0.8000 | 0.6063 | 0.6468 | 13294 | S1-equivalent baseline |

> Note: Hybrid, rewriting, and semantic chunking effects are visible on open-corpus arXiv retrieval.
> Closed-corpus RAGBench (5 passages/query) creates a ceiling that limits differentiation.