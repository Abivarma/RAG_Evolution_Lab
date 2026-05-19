from __future__ import annotations

import tomllib
from dataclasses import dataclass

from stage_1.embedder import BGEEmbedder
from stage_1.generator import OllamaGenerator
from stage_2.chunker import SemanticChunker
from stage_2.hybrid_retriever import HybridRetriever, rrf_fusion
from stage_2.query_rewriter import QueryRewriter
from stage_2.reranker import BGEReranker


@dataclass
class AblationFlags:
    use_query_rewriting: bool = True
    use_hybrid: bool = True
    use_reranker: bool = True
    use_semantic_chunking: bool = True


class Stage2Pipeline:
    """Wires all Stage 2 components; each is independently togglable via AblationFlags."""

    def __init__(
        self,
        embedder: BGEEmbedder,
        rewriter: QueryRewriter,
        hybrid_retriever: HybridRetriever,
        reranker: BGEReranker,
        generator: OllamaGenerator,
        flags: AblationFlags,
        top_k_candidates: int = 50,
        top_k_final: int = 5,
    ) -> None:
        self.embedder = embedder
        self.rewriter = rewriter
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.flags = flags
        self.top_k_candidates = top_k_candidates
        self.top_k_final = top_k_final

    def retrieve_from_passages(
        self, query: str, passages: dict[str, str], top_k: int = 10
    ) -> list[str]:
        """Full Stage 2 pipeline over a closed-corpus passages dict.

        1. BM25 pre-filter to top-200 for corpora > 500 docs (e.g. SciFact 5,183 docs)
        2. Query rewriting (if enabled): expand to N+1 queries
        3. Hybrid BM25+dense RRF per query, fuse all ranked lists
        4. Cross-encoder reranker top-50 -> top-k
        """
        # Pre-filter large corpora so dense encoding stays tractable
        if len(passages) > 500:
            bm25_ids = self.hybrid._bm25_rank(query, passages)[:200]
            passages = {cid: passages[cid] for cid in bm25_ids}

        queries = self.rewriter.rewrite(query) if self.flags.use_query_rewriting else [query]

        all_ranked: list[list[str]] = []
        for q in queries:
            if self.flags.use_hybrid:
                ranked = self.hybrid.rank_passages(q, passages, top_k=len(passages))
            else:
                ranked = self.embedder.embed_and_rank_passages(q, passages, top_k=len(passages))
            all_ranked.append(ranked)

        candidates = rrf_fusion(all_ranked, k=60, top_k=self.top_k_candidates)
        candidates = [c for c in candidates if c in passages]

        if self.flags.use_reranker and candidates:
            return self.reranker.rerank(query, {c: passages[c] for c in candidates}, top_k=top_k)
        return candidates[:top_k]


def load_pipeline(
    config_path: str = "stage_2/config.toml",
    flags: AblationFlags | None = None,
) -> Stage2Pipeline:
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    if flags is None:
        ab = cfg.get("ablation", {})
        flags = AblationFlags(
            use_query_rewriting=ab.get("use_query_rewriting", True),
            use_hybrid=ab.get("use_hybrid", True),
            use_reranker=ab.get("use_reranker", True),
            use_semantic_chunking=ab.get("use_semantic_chunking", True),
        )

    embedder = BGEEmbedder(model_name=cfg["embedding"]["model"], device=cfg["embedding"]["device"])
    rewriter = QueryRewriter(
        model=cfg["query_rewriter"]["model"],
        n_paraphrases=cfg["query_rewriter"]["n_paraphrases"],
        temperature=cfg["query_rewriter"]["temperature"],
    )
    hybrid_retriever = HybridRetriever(embedder=embedder, rrf_k=cfg["hybrid"]["rrf_k"])
    reranker = BGEReranker(
        model_name=cfg["reranker"]["model"],
        device=cfg["reranker"]["device"],
        top_k=cfg["reranker"]["top_k_final"],
    )
    generator = OllamaGenerator(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    return Stage2Pipeline(
        embedder=embedder,
        rewriter=rewriter,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        flags=flags,
        top_k_candidates=cfg["reranker"]["top_k_candidates"],
        top_k_final=cfg["reranker"]["top_k_final"],
    )
