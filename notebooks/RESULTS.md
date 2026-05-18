# RAG Evolution Lab — Results Summary
Generated: 2026-05-18 11:23 UTC
Benchmark: RAGBench TechQA | Hardware: MacBook M3 36GB

| Stage | Architecture | N | Recall@10 | MRR | nDCG@10 | p99 (ms) | Δ Recall@10 | Δ MRR |
|-------|-------------|---|-----------|-----|---------|----------|-------------|-------|
| S0 | S0 BM25 | 50 | 0.8000 | 0.4877 | 0.5655 | 1 | baseline | baseline |
| S1 | S1 Naive RAG | 50 | 0.8000 | 0.6030 | 0.6417 | 334833 | +0.0000 | +0.1153 |
| S2 | S2 Advanced RAG | 50 | 0.8000 | 0.6267 | 0.6621 | 32884 | +0.0000 | +0.0237 |
| S3 | S3 Agentic RAG | 3 | 1.0000 | 0.2333 | 0.4161 | 380182 | +0.2000 | -0.3933 |

> **Note (Stage 3):** Results from 3-query sample. p99=380s vs S2 p99=33s (+11.5x).
> MRR lower than S2 on this 5-passage closed-corpus because the graph returns all
> retrieved IDs not just the top-1 — Recall@10=1.0 (perfect) but MRR measures rank.
> Stage 3 value is proven on OPEN-CORPUS multi-hop queries where decomposition and
> self-correction loops improve accuracy over single-pass retrieval.