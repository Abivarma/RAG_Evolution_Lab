# RAG Evolution Lab — Technical Specification

> A 9-stage architectural learning project that benchmarks the entire RAG retrieval landscape — from BM25 baseline through iterative agentic retrieval — on a MacBook M3 36GB, with industry-standard labeled benchmarks at every stage.

**Author:** [Your Name]
**Spec version:** 1.0
**Target build duration:** 12 weekends
**Primary hardware:** MacBook M3, 36GB RAM, 1TB SSD + external/box warm storage
**Final deliverable:** Open-source repo + 9-post blog series + reusable decision matrix

---

## Table of Contents

1. Project Framing & Success Criteria
2. The 9 Stages — System Architecture Diagrams
3. Dataset Plan & Storage Tier Strategy
4. The Metrics Matrix (Industry-Standard Formulas)
5. Claude Code Mastery Plan (Threaded Through All Stages)
6. Blog Series Outlines with LinkedIn Hooks
7. Week-by-Week Build Roadmap
8. Appendix — Library Versions, Configs, Failure Modes

---

# 1. Project Framing & Success Criteria

## 1.1 The thesis

> **The "best" RAG architecture doesn't exist. There are only patterns that win on specific corpora, latency budgets, and reasoning shapes. This project builds and measures all the major patterns on the same data so the trade-offs become concrete.**

This is not a "build a chatbot" project. It is a **learning architecture comparison** that produces a defensible decision matrix you can use in interviews and architectural reviews.

## 1.2 What "done" looks like

You ship a public GitHub repo containing:

- **9 runnable stage implementations**, each in its own subdirectory with a clean README
- **A shared eval harness** that runs the same labeled benchmark across all stages and emits a unified metrics CSV
- **A decision matrix document** (the synthesis blog post) that maps `corpus shape × latency budget × reasoning depth → recommended pattern` with measured numbers
- **An MCP server wrapper** around your best-performing stage so any MCP client can use your knowledge base
- **A `CLAUDE.md`** that documents the project for Claude Code, with slash commands and subagent definitions

## 1.3 Success criteria (measurable)

You can answer all of the following with numbers from your own runs, not vibes:

1. How much does hybrid retrieval (BM25 + dense + rerank) improve Recall@10 over naive dense-only?
2. How much latency does an agentic supervisor pattern add per query at p99?
3. On which query types does GraphRAG outperform vector RAG, and by how much?
4. What is the accuracy gap between vectorless RAG and the best vector pipeline on FinanceBench?
5. What is the cost-per-correct-answer for each stage (local + API tokens combined)?
6. Where do CoRAG's iterative loops earn their extra compute and where do they not?

## 1.4 Non-goals

- We are NOT building a production RAG product. We're building a **comparison framework**.
- We are NOT fine-tuning models. We're benchmarking off-the-shelf components.
- We are NOT optimizing every hyperparameter exhaustively. We're picking sensible defaults and measuring what they buy.
- We are NOT building a frontend. The eval harness CLI is the interface.

---

# 2. The 9 Stages — System Architecture Diagrams

Each stage adds one architectural shift. The point is **measurable delta** between consecutive stages.

| # | Stage | The architectural shift | Library winner (2026) | Corpus |
|---|-------|--------------------------|----------------------|--------|
| 0 | BM25 Baseline | Pre-AI keyword retrieval | `rank_bm25` or Elasticsearch | arXiv |
| 1 | Naive RAG | Add dense embeddings | `sentence-transformers` + Qdrant | arXiv |
| 2 | Advanced RAG | Add hybrid + rerank + query rewriting | Qdrant hybrid + BGE-reranker-v2 | arXiv |
| 3 | Agentic RAG | Add supervisor multi-agent loop | LangGraph | arXiv |
| 4 | GraphRAG | Add knowledge graph traversal | Microsoft GraphRAG or LightRAG | arXiv |
| 5 | Vectorless RAG | Replace vectors with LLM tree reasoning | PageIndex | SEC 10-Ks |
| 6 | Hybrid Vectorless + Vector | Combine both philosophies | Custom orchestrator | SEC 10-Ks |
| 7 | MCP-Wrapped | Expose as portable protocol | Anthropic MCP SDK | both |
| 8 | CoRAG Iterative | Retrieve-read-decide loop | Custom LangGraph extension | arXiv |

## 2.1 Stage 0 — BM25 Baseline

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  User query  │──────▶│  BM25 Index  │──────▶│   Top-K      │
│  (raw text)  │       │  (1.7M docs) │       │   passages   │
└──────────────┘       └──────────────┘       └──────────────┘
                                                      │
                                                      ▼
                                               No LLM. Return
                                               passages as-is.
```

**Why this exists:** Without a non-AI baseline, every later stage looks impressive. Real architects always show "what does the dumbest thing actually do?" first. This number anchors every other number.

**Components:**
- `rank_bm25` Python lib for the index (sklearn-style API, fits in RAM for 1.7M abstracts)
- No LLM, no embeddings, no reranker
- Scoring: classical BM25 with default k1=1.5, b=0.75

**What we measure:** Recall@5/10/20, MRR, nDCG@10, latency p50/p95/p99, RAM footprint.

**Expected outcome:** Surprisingly strong on keyword-heavy queries. Falls apart on paraphrased, conceptual, or implicit queries — which is exactly the gap dense retrieval closes.

## 2.2 Stage 1 — Naive RAG

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────┐
│  User query  │───▶│  Embed query  │───▶│  Qdrant ANN  │───▶│  Top-K   │
└──────────────┘    │   (BGE-M3)    │    │   (HNSW)     │    │ passages │
                    └───────────────┘    └──────────────┘    └────┬─────┘
                                                                  │
                                                                  ▼
                                                          ┌───────────────┐
                                                          │  Qwen 14B Q4  │
                                                          │  "Answer this │
                                                          │  using these" │
                                                          └───────┬───────┘
                                                                  │
                                                                  ▼
                                                            Final answer
```

**Why this exists:** The "textbook 2023 RAG diagram" everyone copies. Establishes the floor for AI-based retrieval.

**Components:**
- **Embedding model:** BGE-M3 (multilingual, dense+sparse capable but only dense used here)
- **Vector DB:** Qdrant single-node, HNSW index, default params (m=16, ef_construct=100)
- **Chunking:** Fixed 512-token chunks, no overlap (deliberately naive)
- **Generator:** Qwen 2.5 14B Instruct, Q4_K_M quantization, via Ollama
- **Top-K:** 5 passages stuffed into context

**The naive sins we deliberately commit:**
- No reranking
- No query rewriting
- Fixed chunk size (the worst chunking strategy)
- No metadata filtering
- Top-K of 5 with no thought to recall

These are the sins Stage 2 fixes. Showing the *bad* version is what makes the comparison meaningful.

## 2.3 Stage 2 — Advanced RAG (Hybrid + Rerank)

```
                          ┌─────────────────┐
                          │   User query    │
                          └────────┬────────┘
                                   │
                  ┌────────────────┼────────────────┐
                  │                │                │
                  ▼                ▼                ▼
          ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
          │ Query rewrite│  │ Dense embed │  │ BM25 sparse  │
          │   (Qwen)     │  │  (BGE-M3)   │  │   (Qdrant)   │
          └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
                 │                │                 │
                 │      ┌─────────┴────────┐        │
                 │      │ Hybrid retrieval │◀───────┘
                 │      │  RRF (top 50)    │
                 │      └─────────┬────────┘
                 │                │
                 │                ▼
                 │      ┌──────────────────┐
                 │      │  BGE-reranker-v2 │
                 │      │   Cross-encoder  │
                 │      │   (top 50 → 5)   │
                 │      └─────────┬────────┘
                 │                │
                 └────────────────┤
                                  ▼
                         ┌──────────────────┐
                         │   Qwen 14B Q4    │
                         │   Final answer   │
                         └──────────────────┘
```

**What changed vs Stage 1:**
- **Hybrid retrieval:** BM25 + dense scores fused with Reciprocal Rank Fusion (RRF). Catches both lexical and semantic matches.
- **Reranking:** Two-stage retrieval. ANN gets you 50 candidates fast; cross-encoder reranker (BGE-reranker-v2) does precise pairwise scoring on those 50 to pick top 5.
- **Query rewriting:** Qwen rewrites the query into 2-3 paraphrases, all three are retrieved, fused.
- **Semantic chunking:** Recursive splitter respecting paragraph and section boundaries, ~512 tokens with 50-token overlap.
- **Metadata filtering:** arXiv category, year, author available as Qdrant payload filters.

**This is what most production RAG looks like in 2026.** Stages 3+ are increasingly less common in production but increasingly more interesting architecturally.

## 2.4 Stage 3 — Agentic RAG (LangGraph Supervisor)

```
                    ┌───────────────────────┐
                    │   Supervisor Agent    │
                    │  (Qwen / Claude API)  │
                    └───────────┬───────────┘
                                │  routes
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐    ┌──────────────────┐    ┌───────────────┐
│  Retriever    │    │   Validator      │    │ Synthesizer   │
│   Agent       │    │   Agent          │    │   Agent       │
│               │    │                  │    │               │
│ Picks strategy│    │ "Do these chunks │    │ Composes final│
│ Hybrid/dense  │    │ actually answer  │    │ answer with   │
│ Query rewrite │    │ the question?"   │    │ citations     │
└───────┬───────┘    └──────────┬───────┘    └───────┬───────┘
        │                       │                    │
        │ If validator says NO──┘                    │
        │ supervisor re-routes to retriever          │
        │ with new strategy                          │
        ▼                                            ▼
   Retrieval                                   Final answer
   (Stage 2 stack)                             with reasoning trace
```

**What changed vs Stage 2:**
- A **stateful graph** instead of a one-shot pipeline. LangGraph manages the state machine.
- **Self-correction loop:** validator can reject retrieved context and request re-retrieval with a different strategy.
- **Query decomposition:** complex queries get broken into sub-queries by the supervisor.
- **Per-query strategy selection:** the supervisor decides whether this query needs dense, hybrid, or filtered retrieval.

**The honest trade-off:** Stage 3 will be **slower per query** than Stage 2 (more LLM calls). It should be **more accurate on hard queries** but might be equal or worse on easy queries (over-engineering simple retrievals). The metrics matrix will show this directly.

**Library choice rationale:** LangGraph won the 2025-2026 orchestration race because it gives you proper state machines with cycles, conditional edges, and persistence — which CrewAI and AutoGen don't do as cleanly. CrewAI is great for sequential role-play workflows; LangGraph is what you want when the graph has loops and conditional branches.

## 2.5 Stage 4 — GraphRAG

```
PHASE 1: ONE-TIME GRAPH BUILD (offline, ~6 hours)
                                                          ┌─────────────┐
arXiv corpus ──▶ Entity extraction ──▶ Relation extraction ──▶ Knowledge │
                  (LLM-based)             (LLM-based)         │  Graph    │
                                                              │  (Neo4j)  │
                                                              └─────┬─────┘
                                                                    │
                                                                    ▼
                                                            Community detection
                                                            (Leiden algorithm)
                                                                    │
                                                                    ▼
                                                            Community summaries
                                                            (LLM-generated)

PHASE 2: QUERY TIME
                  ┌──────────────┐
                  │  User query  │
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      ┌──────────────┐     ┌─────────────────┐
      │ Vector search│     │  Graph traversal│
      │ (Stage 2)    │     │  Entity → neigh │
      └──────┬───────┘     │  Community → all│
             │             └────────┬────────┘
             │                      │
             └──────────┬───────────┘
                        ▼
                ┌───────────────┐
                │  Context fuse │
                │  (rerank both)│
                └───────┬───────┘
                        ▼
                ┌───────────────┐
                │ Qwen / Claude │
                │ Final answer  │
                └───────────────┘
```

**What changed vs Stage 3:**
- A **knowledge graph** is built from the corpus by extracting entities and relationships using an LLM. Stored in Neo4j (or LightRAG if you want lighter weight).
- At query time, you can do **graph traversal** alongside vector retrieval. For "what papers cite ReAct and discuss tool use" — the graph wins outright.
- **Community detection** (Leiden) clusters related entities; pre-computed community summaries let you answer global questions like "what are the main themes in agent papers from 2024?".

**The honest trade-off:** Graph build is **expensive** — Microsoft GraphRAG can spend ~$50-200 in API costs on a corpus this size if you use frontier models. Use **local Qwen 14B for entity extraction** with batching to keep this cheap (~$0 + 6 hours of M3 time). LightRAG is the lighter-weight alternative if Microsoft GraphRAG is too heavy.

**When GraphRAG wins (per MultiHop-RAG benchmarks):** multi-hop questions across documents, queries about relationships between entities, global summary questions. **When it loses:** single-hop factual lookups, where the extra graph traversal just adds latency.

## 2.6 Stage 5 — Vectorless RAG (PageIndex)

```
PHASE 1: TREE BUILD (offline, ~30 min per filing × 10K filings = parallelized)
                                                          ┌──────────────┐
SEC 10-K PDFs ──▶ Section detection ──▶ Tree construction ──▶ Hierarchical│
                   (LLM-based)            (recursive)        │ Tree Index │
                                                             │  (JSON)    │
                                                             └──────────────┘
                                                              Per-filing tree
                                                              ~137 LLM calls

PHASE 2: QUERY TIME (per query)
                  ┌──────────────┐
                  │  User query  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌─────────────────────────┐
                  │ Filing router (which    │
                  │ filing(s) to search?)   │
                  │ — needed since 10K docs │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Tree navigation LLM call│
                  │ "Given this TOC, which  │
                  │  nodes likely answer?"  │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Fetch raw text of those │
                  │ specific nodes only     │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Synthesis LLM call      │
                  │ Final answer + citations│
                  └─────────────────────────┘
```

**The fundamental philosophy shift:**
- **No embeddings. No vectors. No similarity search.**
- The LLM **reads a table of contents and reasons** about which sections to retrieve.
- Every retrieval trace is **human-auditable** — you can see exactly which tree nodes the LLM chose and why.

**Why SEC 10-Ks specifically:** Vectorless RAG needs structurally rich documents. arXiv papers are too short/flat. 10-K filings have deep hierarchy (Items 1-15, sub-items, exhibits) and are 100-300 pages each — exactly what tree navigation is built for.

**The killer caveat to measure honestly:** Latency. Multiple sequential LLM calls per query. Expect 5-30 seconds per query depending on tree depth. This is **not for interactive chat use** — it's for high-stakes analytical queries where accuracy matters more than speed.

**The expected wow number:** Vectorless RAG hit ~98.7% on FinanceBench in published benchmarks. Reproducing even 90%+ on your local setup is interview gold.

## 2.7 Stage 6 — Hybrid Vectorless + Vector

```
                  ┌──────────────┐
                  │  User query  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌─────────────────────────┐
                  │ Query classifier (LLM)  │
                  │ Is this:                │
                  │ A) factual lookup       │ ──┐
                  │ B) analytical deep-dive │ ──┼──┐
                  │ C) cross-doc multi-hop  │ ──┘  │
                  └─────────────────────────┘      │
                         │                          │
            ┌────────────┼────────────┐             │
            ▼            ▼            ▼             │
       ┌────────┐  ┌──────────┐  ┌──────────┐      │
       │ Stage 2│  │ Stage 5  │  │ Stage 4  │      │
       │ Hybrid │  │Vectorless│  │ GraphRAG │      │
       └────┬───┘  └────┬─────┘  └────┬─────┘      │
            │           │             │             │
            └───────────┼─────────────┘             │
                        ▼                           │
                ┌───────────────┐                   │
                │  Final answer │ ◀─────────────────┘
                │ + which path  │
                └───────────────┘
```

**The architectural insight:** You don't have to pick one. A **query router** sends each query to the retrieval strategy best suited for it. This is what real enterprises end up building.

**The honest cost:** You're now maintaining 3 retrieval backends. Operationally heavier. But the metrics matrix will show whether the accuracy gains justify the complexity.

**What we measure here:** End-to-end accuracy + average latency, vs each pure stage. Plus router accuracy (did it pick the right strategy?).

## 2.8 Stage 7 — MCP-Wrapped Service

```
                ┌──────────────────────────────────────────┐
                │           MCP CLIENT LAYER               │
                │  (Claude Code, Cursor, custom agents)    │
                └──────────────────┬───────────────────────┘
                                   │ MCP protocol (stdio or HTTP)
                                   │ tool: search_knowledge_base
                                   │ tool: explain_with_evidence
                                   ▼
                ┌──────────────────────────────────────────┐
                │       YOUR MCP SERVER (Python)           │
                │   anthropic-mcp-sdk                      │
                │                                          │
                │   • Exposes search/explain as tools      │
                │   • Routes to Stage 6 hybrid backend     │
                │   • Returns structured results + traces  │
                └──────────────────┬───────────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────────┐
                │          Stage 6 Backend                 │
                │     (Hybrid Vector + Vectorless)         │
                └──────────────────────────────────────────┘
```

**The philosophical shift:** Up to Stage 6, your system is an island. With MCP, **any MCP client** — Claude Code, Cursor, custom agents — can use your RAG as a tool with zero integration code.

**This is NOT a speed optimization.** MCP doesn't make queries faster. It makes your system **composable**. That's the 2026 enterprise win — your team's knowledge becomes a reusable tool, not a siloed app.

**What we measure:** Not latency (it's the same as Stage 6 plus protocol overhead, ~5-20ms). We measure **integration friction** — how many lines of code does a new client need to use your system? With MCP it's roughly 3 lines vs ~50-100 for a custom REST integration.

**Resume-grade story:** "I built an MCP server exposing a multi-stage RAG system, callable from Claude Code, Cursor, or any MCP client." That sentence opens doors in 2026.

## 2.9 Stage 8 — CoRAG Iterative Retrieval

```
                  ┌──────────────┐
                  │  User query  │
                  └──────┬───────┘
                         │
                         ▼
              ┌────────────────────────┐
              │  Step 1: Initial       │
              │  retrieval (Stage 2)   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Step 2: LLM reads,    │
              │  decides:              │
              │  "Is this enough?"     │
              └─────────┬──────────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
            YES                  NO
              │                   │
              │                   ▼
              │      ┌────────────────────────┐
              │      │ Step 3: Generate sub-  │
              │      │ query from gaps        │
              │      └────────────┬───────────┘
              │                   │
              │                   ▼
              │      ┌────────────────────────┐
              │      │ Step 4: Retrieve more  │
              │      └────────────┬───────────┘
              │                   │
              │                   └──── loop back to Step 2
              │                          (max 5 iterations)
              ▼
        ┌──────────────┐
        │ Final answer │
        │ with full    │
        │ retrieval    │
        │ trace        │
        └──────────────┘
```

**The architectural shift vs Stage 3:** Stage 3's agents *decide upfront* what to retrieve. CoRAG retrieves, **reads what it got**, then **decides what to retrieve next** based on what was missing. It's iterative depth-first vs Stage 3's breadth-first.

**Why it matters:** State-of-the-art on multi-hop. CoRAG-8B posts state-of-the-art performance on multi-hop QA including HotpotQA subsets, beating systems built on LLMs three to five times larger because single-pass RAG struggles with multi-hop by design.

**The honest trade-off:** Up to 5x the LLM calls of a single-pass RAG. Latency 5-15 seconds per query. **This is the most expensive stage by far** and only earns its cost on hard multi-hop questions.

**Critical eval:** This stage MUST be benchmarked specifically on MultiHop-RAG. Running it on RAGBench single-hop will make it look terrible because the iterations add no value. Picking the right benchmark for each stage is itself a measured skill in this project.

---

# 3. Dataset Plan & Storage Tier Strategy

## 3.1 The two corpora

### Corpus A: arXiv (Stages 0-4, 7, 8)

| Field | Value |
|---|---|
| Source | HuggingFace `arxiv-community/arxiv_dataset` |
| Size | 1.7M papers (metadata + abstracts) |
| Format | JSON, ~3GB raw |
| License | CC0 metadata |
| Storage after embedding | ~12GB (BGE-M3 dim 1024, fp16) |
| Storage after quantization | ~3GB (binary quantization) |

We use **abstracts only**, not full text. Reasoning: full-text PDFs total 1.1TB and don't fit your drive. Abstracts give you the enterprise-scale retrieval story without the storage tax.

### Corpus B: SEC 10-K Filings (Stages 5, 6, 7)

| Field | Value |
|---|---|
| Source | HuggingFace `PleIAs/SEC` |
| Size used | 10,000 filings (subset of 245K available) |
| Format | Parquet, ~6GB after subset |
| Avg filing length | ~34K words |
| License | Public domain (SEC filings) |
| Tree index storage | ~3GB |

We subset to 10K filings because: (a) we have warm storage to archive intermediate artifacts so we can be more generous, (b) 10K filings is enough to make vectorless meaningfully challenged (filing router becomes non-trivial), (c) less than 245K means tree build completes in days not weeks.

## 3.2 The three eval sets

| Eval set | Source | Size | Use case |
|---|---|---|---|
| **RAGBench** | `galileo-ai/ragbench` | 100K labeled examples, 12 sub-datasets | Stages 0, 1, 2, 3 — single-hop, multi-domain |
| **MultiHop-RAG** | `yixuantt/MultiHop-RAG` | Multi-hop queries with evidence chains | Stages 4, 8 — graph traversal & iterative |
| **FinDER** | `Linq-AI-Research/FinDER` | 5,703 expert-labeled 10-K QA triplets | Stages 5, 6 — vectorless on SEC |
| FinanceBench (supplementary) | `PatronusAI/financebench` | 150 sample QA, full set on GitHub | Stages 5, 6 — comparable to PageIndex's 98.7% benchmark |

## 3.3 Storage tier strategy

You mentioned external/box storage. Here's how it integrates:

```
                          ┌────────────────────────────┐
                          │  HOT TIER (M3 1TB SSD)     │
                          │  Currently-active stage    │
                          │  ~80-120 GB                │
                          │  Sub-ms access             │
                          └────────────┬───────────────┘
                                       │ when stage complete
                                       ▼
                          ┌────────────────────────────┐
                          │  WARM TIER (external/box)  │
                          │  Last 2-3 completed stages │
                          │  ~200-500 GB                │
                          │  Re-mountable in minutes   │
                          └────────────┬───────────────┘
                                       │ when 3+ stages old
                                       ▼
                          ┌────────────────────────────┐
                          │  COLD TIER (compressed)    │
                          │  Reference snapshots only  │
                          │  ~50-100 GB compressed     │
                          │  Re-runnable on demand     │
                          └────────────────────────────┘
```

**What lives where:**

- **Hot (M3 SSD):** Active stage's vector index, current LLM weights, running Qdrant/Neo4j services, current eval cache
- **Warm (external):** Past stage vector indexes (you might re-run for comparison), processed corpus snapshots, eval result CSVs, all blog post draft material
- **Cold (compressed):** Original raw datasets (re-downloadable from HF if lost), one-time intermediate artifacts (parsed PDF markdown), historical experiment logs

**Promotion/demotion script:** A `/promote-stage` slash command (defined in section 5) tar-gzips the previous stage's artifacts and moves them to warm storage automatically when you start a new stage.

---

# 4. The Metrics Matrix (Industry-Standard Formulas)

This is the heart of the project. Every stage produces the same metrics so they're directly comparable.

## 4.1 Retrieval quality metrics

### Recall@K

> Of the documents that *should* have been retrieved (ground truth), what fraction did we actually retrieve in our top-K results?

$$\text{Recall@K} = \frac{|\text{retrieved}_K \cap \text{relevant}|}{|\text{relevant}|}$$

We measure Recall@5, Recall@10, Recall@20. K=10 is the most-cited number.

### Mean Reciprocal Rank (MRR)

> Average of `1/rank` of the first relevant document. Punishes systems that bury the right answer deep.

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

### nDCG@10 (Normalized Discounted Cumulative Gain)

> Accounts for graded relevance (some hits are more relevant than others) and rewards putting most-relevant first.

$$\text{DCG}_p = \sum_{i=1}^{p} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$
$$\text{nDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}$$

### Hit Rate

> Binary: did *any* relevant document appear in top-K? Useful as a coarse floor metric.

## 4.2 Generation quality metrics (the RAG triad + TRACe)

### Faithfulness

> Is every claim in the generated answer supported by the retrieved context? Hallucination = unfaithful.

Computed via RAGAS: decompose answer into claims, check each claim against context using an LLM judge.

### Answer Relevance

> Does the answer actually address what was asked?

Computed via RAGAS: generate questions that the answer would answer, embed and compare to original question.

### Context Precision

> Of the chunks we retrieved, what fraction were actually useful for the answer?

### Context Recall

> Of the information needed for the answer, what fraction did we retrieve?

### TRACe metrics (from RAGBench paper)

| Metric | What it measures |
|---|---|
| **T**rust | Faithfulness of response to context |
| **R**elevance | Did retrieved context match the query? |
| **A**ccuracy | Did the answer match ground truth? |
| **C**ompleteness | Did we retrieve all needed info? |
| **e** = explainable | Are these scores interpretable? |

RAGBench publishes ground truth labels for these, so you don't need an LLM judge — you compare directly.

## 4.3 Performance metrics

| Metric | How measured |
|---|---|
| Latency p50, p95, p99 | Wall-clock time over 1000 queries, percentiles |
| Throughput (queries/sec) | Concurrent request handling with 10 workers |
| Index build time | Wall-clock for full corpus indexing |
| Index storage size | `du -sh` of index directory |
| Peak RAM during query | `psutil` polling at 100ms during query |
| Peak RAM during indexing | Same, captured during build |

## 4.4 Cost metrics

| Metric | How measured |
|---|---|
| Input tokens per query | Sum across all LLM calls in the pipeline |
| Output tokens per query | Same |
| $/1000 queries | Tokens × current Claude API pricing (if using API) |
| Local GPU-hours equivalent | Estimated from Ollama's reported eval times |

## 4.5 Context engineering metrics

These are extras that go deep on the *context engineering* angle.

| Metric | How measured |
|---|---|
| Context window utilization % | (tokens sent / model max) per query |
| Chunk size vs recall curve | Sweep 256, 512, 1024, 2048 token chunks; plot Recall@10 |
| Reranker top-k vs latency | Sweep rerank top-k from 10 to 100; plot p95 latency |
| Position bias (lost-in-the-middle) | Inject correct chunk at positions 1, 5, 10, plot accuracy |

## 4.6 The decision matrix (final output)

The synthesis blog produces this. Each cell is a stage's normalized score (0-1) on that dimension.

|  | Recall@10 | Faithfulness | p99 latency | Cost/query | Explainability | Multi-hop acc |
|---|---|---|---|---|---|---|
| S0 BM25 | x.xx | n/a | x.xx | x.xx | high | x.xx |
| S1 Naive | x.xx | x.xx | x.xx | x.xx | low | x.xx |
| S2 Advanced | x.xx | x.xx | x.xx | x.xx | low | x.xx |
| S3 Agentic | x.xx | x.xx | x.xx | x.xx | med | x.xx |
| S4 Graph | x.xx | x.xx | x.xx | x.xx | med | x.xx |
| S5 Vectorless | x.xx | x.xx | x.xx | x.xx | **high** | x.xx |
| S6 Hybrid | x.xx | x.xx | x.xx | x.xx | med | x.xx |
| S7 MCP | (same as S6) | | | | | |
| S8 CoRAG | x.xx | x.xx | x.xx | x.xx | high | **x.xx** |

The decision matrix maps query characteristics to recommended stages:

| Query type | Latency budget | Recommended pattern |
|---|---|---|
| Single-hop factual | <100ms | S0 or S2 |
| Single-hop semantic | <500ms | S2 |
| Multi-hop reasoning | <5s | S4 or S8 |
| Cross-doc synthesis | <10s | S4 |
| Regulated/auditable | any | S5 |
| Interactive agent tool | <500ms | S2 + MCP |
| Mixed workload | <2s | S6 + MCP |

---

# 5. Claude Code Mastery Plan (Threaded Through All Stages)

I moved this section ahead of blogs/roadmap because it's the **operational backbone** for building the rest. Doing it right makes every later stage faster.

## 5.1 The CLAUDE.md project file

This file goes at the repo root. Claude Code reads it as project-level context every session.

```markdown
# RAG Evolution Lab

## Project context
9-stage RAG architecture comparison on a MacBook M3 36GB.
See SPEC.md for the full design. We are currently working on Stage {N}.

## Hardware constraints (HARD LIMITS)
- 36GB RAM total. One heavy service (Qdrant/Neo4j) + Qwen 14B + workspace must fit.
- 1TB SSD with warm tier on external storage.
- No GPU beyond Apple M3 Metal. Don't suggest CUDA solutions.

## Stack
- Python 3.11, uv for env management
- Qdrant for vector store (single binary, port 6333)
- Neo4j for graph (port 7687, only when Stage 4 active)
- Ollama for local LLM serving (Qwen 2.5 14B Q4_K_M default)
- LangGraph for agentic stages
- BGE-M3 embeddings, BGE-reranker-v2 cross-encoder
- RAGAS for eval, plus custom harness for TRACe metrics

## Coding conventions
- Type hints on all functions. Use `from __future__ import annotations`.
- All configs in `config.toml` per stage, never hardcoded.
- Eval results saved to `results/stage_{N}/{timestamp}.json`.
- Every stage has identical eval harness signature: `python -m stage_{N}.eval --benchmark {name}`.

## Anti-patterns we explicitly avoid
- Don't suggest fine-tuning. We're benchmarking off-the-shelf.
- Don't suggest cloud-only solutions (Pinecone, OpenAI embeddings).
  We're measuring local-first patterns.
- Don't conflate "agentic" with "more LLM calls is better".
  Latency/cost trade-offs must be measured, not assumed.

## What I want from Claude Code in this project
- Lead with the architectural shift each change embodies
- Always reference the metric this change affects
- Suggest the minimal change before any refactor
- If I'm about to commit a stage-coupling sin, flag it
```

## 5.2 Custom slash commands

Slash commands live in `.claude/commands/`. They are markdown files that get expanded into prompts.

### `/eval-stage <N> [--benchmark NAME]`

Runs the eval harness for stage N and prints a summary. File: `.claude/commands/eval-stage.md`

```markdown
Run the eval harness for stage $1 against the benchmark $2 (default: ragbench).

Steps:
1. Verify stage $1 services are running (Qdrant/Neo4j/Ollama as needed)
2. cd into stage_$1/
3. Run: `python -m eval --benchmark $2 --output results/$(date +%Y%m%d_%H%M%S).json`
4. Parse results JSON and print summary table:
   - Recall@5/10/20, MRR, nDCG@10
   - Faithfulness, Answer Relevance, Context Precision/Recall
   - p50/p95/p99 latency
   - Total tokens, estimated cost
5. Compare to last run's results. Flag any metric that regressed >5%.

If services aren't running, suggest the exact docker/ollama commands to start them.
```

### `/compare-stages <N> <M>`

Diffs two stages' eval results side by side. File: `.claude/commands/compare-stages.md`

```markdown
Compare the most recent eval results from stage $1 and stage $2.

Print a side-by-side table for every metric. Highlight:
- Where stage $2 beat stage $1 (green/+)
- Where stage $2 regressed vs stage $1 (red/-)
- Where the gap is within noise (<2%)

End with a one-paragraph summary: "Stage $2 buys X at the cost of Y."
This summary is the candidate sentence for the corresponding blog post.
```

### `/ingest <corpus>`

Runs corpus ingestion with appropriate chunking for the current stage.

### `/promote-stage`

The storage tier mover. When you start working on stage N+1, this archives stage N to warm storage.

```markdown
Promote the current completed stage to warm tier.

1. Identify current stage from .claude/current_stage.txt
2. tar -czf {warm_tier_path}/stage_{N}_$(date +%Y%m%d).tar.gz stage_{N}/data/
3. Verify the archive (-tzf | wc -l)
4. Remove stage_{N}/data/ contents (NOT the code, just data/index)
5. Update .claude/current_stage.txt to {N+1}
6. Print disk space freed
```

### `/blog-draft <N>`

Generates the first draft of the blog post for stage N using the eval results as backbone.

## 5.3 Subagents

Subagents are scoped contexts Claude Code can spawn for parallel work. Define them in `.claude/agents/`.

### `chunking-experimenter`

Sweeps chunk sizes on a fixed query set, reports the Recall@10 curve. Use when picking chunking strategy for a new stage.

```yaml
name: chunking-experimenter
description: Runs chunk-size sweep experiments on a corpus subset
tools: [bash, python, str_replace, create_file]
system_prompt: |
  You are a focused experimenter. Your only job is to run the chunking
  sweep experiment defined in scripts/chunking_sweep.py with the
  parameters provided. Run it, capture results, plot if asked, and
  return concise findings. Do not refactor unrelated code.
```

### `eval-runner`

Long-running eval batches. Spawn this when you want to run a 5000-query benchmark and not block the main session.

### `metrics-aggregator`

Walks `results/` and produces the cross-stage comparison table. Use before writing the synthesis blog.

## 5.4 Hooks

Hooks fire on specific Claude Code events. Define in `.claude/settings.json`.

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "scripts/verify_no_secrets.sh $CLAUDE_TOOL_OUTPUT_FILE",
        "description": "Make sure no API keys are in committed files"
      }
    ],
    "Stop": [
      {
        "command": "scripts/log_latency.py --session $CLAUDE_SESSION_ID",
        "description": "Log session-end metric so we can profile token usage over time"
      }
    ]
  }
}
```

## 5.5 MCP integration (the meta moment)

Stage 7 builds an MCP server. After that stage, you add it to Claude Code's MCP config:

```json
{
  "mcpServers": {
    "rag-evolution-lab": {
      "command": "python",
      "args": ["-m", "stage_7.mcp_server"],
      "env": {
        "RAG_BACKEND": "stage_6_hybrid"
      }
    }
  }
}
```

Now Claude Code itself can query your 1.7M arXiv index and 10K SEC filings. **You can ask Claude Code questions about your own corpus while developing the project.** That's a beautiful demo.

## 5.6 Claude Code workflow per stage

Standard loop:

1. Read `SPEC.md` section for the upcoming stage with Claude Code
2. `/promote-stage` to archive previous stage
3. Ask Claude Code: "Implement stage N per spec. Start with the eval harness so we have a test." (TDD-style)
4. Build out the stage in incremental commits
5. `/eval-stage N` to run the benchmark
6. `/compare-stages N-1 N` to see the architectural delta
7. `/blog-draft N` to capture findings while fresh
8. Move on to N+1

---

# 6. Blog Series Outlines with LinkedIn Hooks

9 stage posts + 1 synthesis post = 10-part series. Each follows the same skeleton: hook → problem → architecture → numbers → trade-off lesson.

## 6.1 Post 1 — The baseline nobody publishes: BM25 over 1.7M papers

**LinkedIn hook (first 3 lines, before the "see more" cutoff):**

> Every AI engineer skips the baseline.
>
> I indexed 1.7M arXiv papers using BM25 — the algorithm from 1994 — and ran it against modern dense retrieval.
>
> The result reframed how I think about RAG.

**Structure:**
- The problem: AI-first thinking skips the question "is this even better than keyword search?"
- The architecture diagram (Stage 0)
- Setup walkthrough (rank_bm25, the indexing time, the surprise of how cheap it was)
- The numbers: Recall@10, MRR, latency
- The honest finding: where BM25 actually wins (acronym-heavy, exact-phrase queries)
- The teaser: dense retrieval should beat this, right? Next post finds out.

## 6.2 Post 2 — Naive RAG: the 2023 architecture and why it's not enough

**Hook:**

> The "textbook RAG diagram" you've seen 1000 times has a recall problem.
>
> I built it as-drawn over 1.7M arXiv papers.
>
> Recall@10 was 𝟯𝟴%. That number is the start of every RAG conversation that matters.

**Structure:**
- Show the canonical 2023 diagram everyone draws
- Walk through the build: BGE-M3, Qdrant, Qwen 14B
- The numbers vs BM25 (sometimes worse, surprisingly)
- The four sins of naive RAG: fixed chunks, no rerank, no rewrite, no filtering
- Teaser: each of those sins gets fixed in the next post and we measure how much it buys.

## 6.3 Post 3 — Advanced RAG: hybrid + reranking + the metrics that move

**Hook:**

> One change in my RAG pipeline lifted Recall@10 by 𝟮𝟴 points.
>
> It wasn't a bigger model. It wasn't a better embedding. It was 2 lines of code for hybrid retrieval.
>
> Here's the full ablation.

**Structure:**
- The 4 changes from Stage 1: hybrid (RRF), rerank, query rewriting, semantic chunking
- The architecture diagram (Stage 2)
- Ablation table: each change's individual contribution
- Latency cost: rerank adds ~150ms — when is it worth it?
- "This is what most 2026 production RAG looks like" — make this claim and back it.

## 6.4 Post 4 — Agentic RAG with LangGraph: when more LLM calls earn their cost

**Hook:**

> I made my RAG system 𝟱𝘅 slower on purpose.
>
> Then measured exactly where that extra latency bought real accuracy and where it bought nothing.
>
> The result: agentic RAG is the right answer 𝟯𝟬% of the time. Here's how to know which 30%.

**Structure:**
- The supervisor-worker pattern, why it's the 2026 enterprise winner
- Why LangGraph beat CrewAI and AutoGen for this use case
- The validator agent loop in code (snippet)
- The numbers: latency went up, but on hard queries (multi-step) accuracy went up more
- The honest finding: on easy queries it's net-negative
- Production lesson: classify queries first, route to agentic only when needed

## 6.5 Post 5 — GraphRAG: when vectors aren't enough

**Hook:**

> Vector search can't answer "who works on agent papers that cite ReAct?"
>
> Not "won't" — 𝗰𝗮𝗻'𝘁. The information isn't in any single chunk.
>
> Building the knowledge graph took 6 hours. The first multi-hop query justified all of it.

**Structure:**
- The fundamental limit of vector retrieval: it's chunk-shaped, the world isn't
- Entity + relation extraction, community detection
- The MultiHop-RAG benchmark — and why it's the ONLY fair test
- Cost comparison: Microsoft GraphRAG with frontier LLMs vs LightRAG with local Qwen
- Where graph loses: single-hop factual queries (the irony)
- When you need this and when you don't

## 6.6 Post 6 — Vectorless RAG: the most interesting RAG paper of 2025

**Hook:**

> PageIndex hit 98.7% on FinanceBench without a single vector.
>
> No embeddings. No similarity search. Just an LLM reading a table of contents.
>
> I reproduced it locally on 10K SEC filings. Here's what it actually does.

**Structure:**
- The philosophy shift: reasoning over similarity
- How the tree gets built (the ~137 LLM calls/doc cost)
- How retrieval becomes navigation
- The latency tax: seconds, not ms
- When this is the right call: regulated, audited, complex-structure docs
- When it absolutely isn't: chat, scale, latency-sensitive

## 6.7 Post 7 — Hybrid vectorless + vector: the router pattern enterprises actually ship

**Hook:**

> Real enterprise RAG isn't one pattern. It's a router that picks per query.
>
> I built one. It outperforms every single pattern in isolation on mixed workloads.
>
> The architectural lesson is bigger than RAG.

**Structure:**
- The router pattern explained
- Query classifier: what features predict which retrieval strategy wins
- The orchestration code (LangGraph again)
- Numbers: combined accuracy vs each pure stage
- Cost of complexity: 3 backends to maintain
- The meta-lesson: hybrid > pure when the workload is heterogeneous

## 6.8 Post 8 — MCP: why protocol is the 2026 unlock

**Hook:**

> Your knowledge base is an island until you wrap it in MCP.
>
> 3 lines of integration code instead of 100. Any MCP client can use it. Including Claude Code.
>
> Here's the full pattern.

**Structure:**
- What MCP is (and isn't — it's not a speed boost)
- The wrap-around code (anthropic-mcp-sdk minimal example)
- Demo: Claude Code querying your own RAG system
- The composability win: same backend, multiple clients
- Why this is the 2026 enterprise pattern

## 6.9 Post 9 — CoRAG: when retrieval becomes a loop

**Hook:**

> CoRAG-8B beats systems built on LLMs 5x its size on multi-hop QA.
>
> The trick isn't a bigger model. It's letting the model retrieve, read, then decide what to retrieve next.
>
> I implemented it locally. Here's what changed.

**Structure:**
- The single-pass limit on multi-hop questions (with examples)
- The iterative retrieve-read-decide loop, in pseudocode
- The honest cost: up to 5x the LLM calls
- The benchmark: only run this on multi-hop, or you'll embarrass yourself
- Where iterative wins, where it's overkill

## 6.10 Post 10 (synthesis) — The architect's decision matrix

**Hook:**

> I built 9 RAG architectures on the same data and measured them with the same benchmarks.
>
> The answer to "which RAG should I use?" is a 2D table, not a single recommendation.
>
> Here's the matrix.

**Structure:**
- Recap the 9 stages in one sentence each
- The big table (Section 4.6 of this spec)
- 5 worked examples: "given these constraints, here's which stage"
- The meta-lesson: pattern selection is itself the skill
- Repo link, open invitation for forks/PRs

## 6.11 LinkedIn algorithm notes (per your preferences)

Each post should:
- Open with the **hook in first 3 lines** (before the "see more" cutoff)
- Use **unicode bold** (𝗯𝗼𝗹𝗱) sparingly for numbers and key claims — bold rendering survives LinkedIn
- Use **short paragraphs** (1-2 sentences each) — long blocks kill mobile reading
- Include **one image/diagram per post** — algorithm rewards media
- End with a **question or invitation** — comments are the highest-value signal
- Post the **code link in the first comment**, not the body — links in body suppress reach
- Tag **2-3 specific concepts** as hashtags (#RAG #LangGraph #MCPProtocol), not generic ones (#AI #ML)
- Cadence: post 1/week for 10 weeks. Consistency beats burst.

---

# 7. Week-by-Week Build Roadmap

12 weekends total. Buffer weeks 11-12 for blog writing and synthesis. Assumes ~8 hours/weekend.

## Week 1 — Foundations

- Set up the repo, `CLAUDE.md`, `.claude/` directory, `uv` env
- Install Ollama + Qwen 2.5 14B Q4_K_M, verify inference speed
- Install Qdrant single-binary, verify it starts
- Download arXiv dataset, extract metadata + abstracts JSON (3GB)
- Download RAGBench, MultiHop-RAG, FinDER
- Write the eval harness skeleton (the file every stage imports)
- **Deliverable:** working "hello world" — eval harness runs against a fake retriever and produces JSON output

## Week 2 — Stage 0 (BM25) + Stage 1 (Naive RAG)

- Build BM25 index over 1.7M abstracts
- Run RAGBench eval against BM25 → first real numbers
- Build Stage 1 naive RAG: embed everything (~6 hours), Qdrant index, Qwen generation
- Run RAGBench eval against Stage 1
- **Deliverable:** Posts 1 & 2 draft, two stages benchmarked

## Week 3 — Stage 2 (Advanced RAG)

- Add hybrid (BM25 sparse + dense fused with RRF) to Qdrant
- Add BGE-reranker-v2 cross-encoder
- Add query rewriting via Qwen
- Add semantic chunking
- Ablation runs (each component on/off)
- **Deliverable:** Post 3 draft, ablation table

## Week 4 — Stage 3 (Agentic RAG)

- Install LangGraph
- Build supervisor + retriever + validator + synthesizer graph
- Wire it to Stage 2's retrieval backend
- Eval on RAGBench
- **Deliverable:** Post 4 draft, LangGraph code

## Week 5 — Stage 4 prep (GraphRAG entity extraction)

- Install Neo4j (Docker)
- Decide: Microsoft GraphRAG vs LightRAG (recommend LightRAG for cost)
- Run entity + relation extraction over arXiv subset (start with 100K papers, scale)
- ~6-12 hour batch job — use a subagent
- **Deliverable:** Populated graph, community summaries

## Week 6 — Stage 4 (GraphRAG query side)

- Build graph traversal queries
- Build the vector + graph fusion logic
- Eval on **MultiHop-RAG specifically** (not RAGBench)
- **Deliverable:** Post 5 draft, multi-hop comparison numbers

## Week 7 — Stage 5 (Vectorless RAG)

- Promote stages 0-3 artifacts to warm storage
- Download SEC 10-K subset (10K filings)
- Convert PDFs to markdown (this is the one job to consider offloading to Colab)
- Install PageIndex, build trees for all 10K filings
- Run FinDER eval
- **Deliverable:** Post 6 draft, FinanceBench reproduction numbers

## Week 8 — Stage 6 (Hybrid Vectorless + Vector)

- Build query classifier (start with rule-based, then LLM-based)
- Build router → Stage 2 / Stage 4 / Stage 5 logic
- End-to-end eval on mixed workload (RAGBench + FinDER mixed)
- **Deliverable:** Post 7 draft, router accuracy + downstream numbers

## Week 9 — Stage 7 (MCP server)

- Install anthropic-mcp-sdk
- Wrap Stage 6 as MCP server (Python)
- Register with Claude Code (`mcp.json` config)
- Demo: query your own RAG from Claude Code chat
- **Deliverable:** Post 8 draft, working MCP integration

## Week 10 — Stage 8 (CoRAG)

- Extend LangGraph with iterative retrieve-read-decide loop
- Implement gap-detection prompt
- Eval on MultiHop-RAG (the right benchmark for this stage)
- **Deliverable:** Post 9 draft, iterative vs single-pass numbers

## Week 11 — Synthesis & decision matrix

- Run the `metrics-aggregator` subagent across all stages
- Build the full decision matrix (Section 4.6)
- Write the synthesis blog post
- **Deliverable:** Post 10 draft, master metrics CSV

## Week 12 — Polish & ship

- Repo README + setup docs
- Reproducibility: one-script-installs-everything
- Final blog edits, scheduled posting cadence
- LinkedIn: post 1 goes live
- **Deliverable:** Repo public, post 1 live, weekly cadence scheduled

## Risk and slip strategy

If you slip (and you will somewhere — usually GraphRAG):

- **Week 5-6 GraphRAG slip:** acceptable, the most likely point. Buffer 1 week. Use LightRAG over MS GraphRAG to compress.
- **Week 7 SEC PDF parsing slip:** this is the one place to use Colab for batch parsing. Acceptable.
- **Week 10 CoRAG slip:** this is the stretch stage. If short on time, ship it as "v2 stretch" and synthesize the existing 8 stages.

The series IS valuable at 8 stages. Don't sacrifice quality on stages 0-3 to rush stage 8.

---

# 8. Appendix

## A. Library versions (May 2026)

```
python = "^3.11"
qdrant-client = "^1.13"
neo4j = "^5.27"
sentence-transformers = "^3.5"  # for BGE-M3
FlagEmbedding = "^1.4"           # for BGE-reranker-v2
langgraph = "^0.3"
langchain = "^0.3"
ragas = "^0.2"
ollama = "^0.4"
rank_bm25 = "^0.2"
pageindex = "^0.5"
mcp = "^1.0"                     # anthropic-mcp-sdk
neo4j-graphrag = "^1.0"          # if using MS GraphRAG
lightrag = "^0.2"                # alternative to MS GraphRAG
```

## B. Likely failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| Embedding 1.7M abstracts OOMs the M3 | Batch at 64, not 256. ~6 hours total. |
| Qdrant index too big to fit RAM | Enable scalar quantization (4-bit) — Qdrant native feature |
| Neo4j fights for RAM with Qwen | Shut Qwen down during graph build; use Claude API for that one job |
| PageIndex tree build too slow on local Qwen | Use Claude Haiku via API for tree generation; cheaper than wall-clock time |
| RAGAS LLM judge is too slow | Use the finetuned DeBERTa judge from RAGBench paper (way faster) |
| Eval set is too small to be statistically meaningful | Run with 3 random seeds; report mean ± std |

## C. Estimated total resource use

- **M3 active hours:** ~150-200 hours across 12 weeks
- **Claude API costs (heavy jobs):** ~$50-150 estimated
- **External storage usage:** ~300GB peak
- **Repo final size (with sample data):** ~5GB

## D. What goes in your resume after this ships

> **RAG Evolution Lab** (Open Source, github.com/yourhandle/rag-evolution-lab)
>
> Built a 9-stage architectural comparison of retrieval-augmented generation patterns — from BM25 baseline through agentic, GraphRAG, vectorless (PageIndex), MCP-wrapped, and CoRAG iterative retrieval — benchmarked end-to-end on 1.7M arXiv papers and 10K SEC 10-K filings using RAGBench, MultiHop-RAG, and FinDER. Produced a measurable decision matrix mapping query characteristics to optimal retrieval pattern. All stages run on a MacBook M3 36GB. Published 10-part blog series documenting architectural trade-offs with unified industry-standard metrics (Recall@K, MRR, nDCG, RAGAS triad, TRACe). 1 of stack: LangGraph, Qdrant, Neo4j, PageIndex, Ollama (Qwen 2.5 14B), BGE-M3 + BGE-reranker-v2, Anthropic MCP SDK.

---

**End of spec v1.0.**

Next steps once you've reviewed:
1. Push back on anything that doesn't fit
2. We start with Week 1 setup
3. I'll write Post 1 with you when Stage 0 numbers come in
