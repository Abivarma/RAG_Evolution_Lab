# RAG Evolution Lab — Results Summary
Generated: 2026-05-19 15:17 UTC
Hardware: MacBook M3 36GB

## Open-Corpus Retrieval — SciFact (50q, 5,183 docs, ~1 relevant/query)

| Stage | Architecture | R@10 | MRR | nDCG@10 | p99 | Δ R@10 | Δ MRR |
|-------|-------------|------|-----|---------|-----|--------|-------|
| S0 | BM25 Baseline | 0.7970 | 0.6586 | 0.6917 | 256ms | baseline | baseline |
| S1 | Naive RAG (BGE-M3) | 0.8520 | 0.6977 | 0.7248 | 27s | +0.0550 | +0.0392 |
| S2 | Advanced RAG (hybrid+rerank) | 0.8880 | 0.8217 | 0.8342 | 92s | +0.0360 | +0.1240 |

## Multi-Hop Retrieval — MultiHopRAG (spec §3.2: correct benchmark for S3, S4, S8)

| Stage | Architecture | N | R@10 | MRR | nDCG@10 | p99 | Notes |
|-------|-------------|---|------|-----|---------|-----|-------|
| S3 | Agentic RAG (LangGraph) | 5 | 1.0000 | 1.0000 | 0.9362 | 616s | N=5 sample, not statistically robust |
| S4 | GraphRAG (LightRAG, top-20 corpus) | 20 | 0.8500 | 0.8500 | 0.8290 | 696s | graph-only, 20-article corpus |

## Pending (S5–S8)

| Stage | Architecture | Benchmark | Status |
|-------|-------------|-----------|--------|
| S5 | Vectorless RAG (PageIndex) | FinDER + FinanceBench | Not started |
| S6 | Hybrid Vectorless+Vector | FinDER + FinanceBench | Not started |
| S7 | MCP Server | — | Not started |
| S8 | CoRAG Iterative | MultiHopRAG | Not started |

## Key Findings So Far

### SciFact open-corpus (S0→S2):
- BM25→Dense (S0→S1): R@10 +5.5pp, MRR +3.9pp, latency +105×
- Dense→Advanced (S1→S2): R@10 +3.6pp, MRR +12.4pp, latency +3.4×
- Total S0→S2: R@10 +9.1pp, MRR +16.3pp at 358× latency cost

### MultiHopRAG (S4 GraphRAG):
- S4 graph-only: R@10=0.850, MRR=0.850 on 20-article LightRAG graph
- Fusion (graph+vector) adds nDCG +0.010 over graph-only
- 20-article corpus covers queries needing those articles; BM25 fallback for others
- Full 609-article graph needed for definitive S3 vs S4 comparison (spec §1.3 Q3)