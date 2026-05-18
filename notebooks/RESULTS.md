# RAG Evolution Lab — Results Summary
Generated: 2026-05-18 04:23 UTC
Benchmark: RAGBench TechQA | N=50 queries | Hardware: MacBook M3 36GB

| Stage | Architecture | Recall@10 | MRR | nDCG@10 | p99 (ms) | Δ Recall@10 | Δ MRR |
|-------|-------------|-----------|-----|---------|----------|-------------|-------|
| S0 | S0 BM25 | 0.8000 | 0.4877 | 0.5655 | 1 | baseline | baseline |
| S1 | S1 Naive RAG (BGE-M3) | 0.8000 | 0.6030 | 0.6417 | 334833 | +0.0000 | +0.1153 |