# RAG Evolution Lab — Results Summary
Generated: 2026-05-19 08:17 UTC
Hardware: MacBook M3 36GB

## Open-Corpus Retrieval — SciFact (50 queries, 5,183 docs, ~1 relevant/query)

| Stage | Architecture | R@5 | R@10 | R@20 | MRR | nDCG@10 | p99 | Δ R@10 | Δ MRR |
|-------|-------------|-----|------|------|-----|---------|-----|--------|-------|
| S0 | BM25 Baseline | 0.7800 | 0.7970 | 0.7970 | 0.6586 | 0.6917 | 256ms | baseline | baseline |
| S1 | Naive RAG (BGE-M3) | 0.7530 | 0.8520 | 0.8520 | 0.6977 | 0.7248 | 27s | +0.0550 | +0.0392 |
| S2 | Advanced RAG (hybrid+rerank) | 0.8390 | 0.8880 | 0.8880 | 0.8217 | 0.8342 | 92s | +0.0360 | +0.1240 |

## Closed-Corpus Ranking — RAGBench TechQA (50 queries, 5 passages/query)
*(Only MRR/nDCG meaningful here — Recall@10 is locked at 0.80 by corpus design)*

| Stage | Architecture | MRR | nDCG@10 | p99 |
|-------|-------------|-----|---------|-----|
| S0 | BM25 | 0.4877 | 0.5655 | 1ms |
| S1 | Naive RAG | 0.6030 | 0.6417 | 335s |
| S2 | Advanced RAG | 0.6267 | 0.6621 | 33s |

## Multi-Hop Retrieval — MultiHopRAG (S3 Agentic, 5 queries)
*(Correct benchmark for agentic/iterative stages per spec §2.4, §2.9)*

| Stage | Architecture | N | R@10 | MRR | nDCG@10 | p99 |
|-------|-------------|---|------|-----|---------|-----|
| S3 | Agentic RAG (LangGraph) | 5 | 1.0000 | 1.0000 | 0.9362 | 616s |
| S4 | GraphRAG | — | TBD | TBD | TBD | — |
| S8 | CoRAG Iterative | — | TBD | TBD | TBD | — |

## Key Findings

### Open-Corpus Retrieval (SciFact, where Recall@K varies by method):
- BM25 → Dense (S0→S1): Recall@10 +5.5pp, MRR +3.9pp, latency +105x
- Dense → Advanced (S1→S2): Recall@10 +3.6pp, MRR +12.4pp, latency +3.4x
- Total (S0→S2): Recall@10 +9.1pp, MRR +16.3pp at 358x latency cost

### Why RAGBench alone is insufficient:
- RAGBench provides exactly 5 pre-selected passages per question (closed-corpus reranking)
- Recall@10 is locked at 0.80 regardless of retrieval method (40/50 queries have ≥1 relevant doc)
- Only MRR and nDCG@10 vary — both show the expected BM25 < Dense < Advanced progression

### Correct benchmark per stage (spec §3.2):
- S0-S2: SciFact (open-corpus Recall@K) + RAGBench (ranking quality MRR/nDCG)
- S3, S4, S8: MultiHopRAG (multi-hop queries where decomposition/iteration matter)
- S5, S6: FinDER + FinanceBench (SEC 10-K financial QA)