from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from typing import Any


def prepare_corpus_json(
    n_articles: int | None = None,
    output_path: Path | None = None,
    top_n_by_query_coverage: int | None = None,
) -> dict[str, str]:
    """Build {url: title + body} dict from MultiHopRAG corpus.

    Args:
        n_articles: Take first N articles from corpus (simple truncation).
        top_n_by_query_coverage: Take the N articles most frequently referenced
            in evidence chains — maximises query coverage per article.
            At top_50: covers 38.7% of all 2,556 queries.
            At top_100: covers 52.2% of all queries.
        output_path: Save as JSON if provided.

    Returns dict of unique article URL -> concatenated title + body text.
    """
    from collections import Counter

    from datasets import load_dataset

    if top_n_by_query_coverage is not None:
        # Count how many queries reference each article URL
        queries_ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
        url_counts: Counter = Counter()
        for row in queries_ds:
            for ev in row.get("evidence_list", []):
                if ev.get("url"):
                    url_counts[ev["url"]] += 1
        top_urls = {url for url, _ in url_counts.most_common(top_n_by_query_coverage)}

        corpus_ds = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
        articles: dict[str, str] = {}
        for row in corpus_ds:
            url = row.get("url", "")
            if url and url in top_urls:
                title = row.get("title", "")
                body = row.get("body", "")
                articles[url] = f"{title}\n\n{body}".strip()
    else:
        corpus_ds = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
        if n_articles:
            corpus_ds = corpus_ds.select(range(min(n_articles, len(corpus_ds))))

        articles = {}
        for row in corpus_ds:
            url = row.get("url", "")
            if not url:
                continue
            title = row.get("title", "")
            body = row.get("body", "")
            articles[url] = f"{title}\n\n{body}".strip()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(articles, indent=2))

    return articles


def _make_lightrag(working_dir: str, llm_model: str, embed_dim: int) -> Any:
    """Construct LightRAG instance using Qwen 14B (LLM) + BGE-M3 (embed via sentence-transformers)."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    # BGEEmbedder from stage_1 — lazy-loaded on first call
    _embedder_cache: list = []

    def _get_embedder():
        if not _embedder_cache:
            from stage_1.embedder import BGEEmbedder
            _embedder_cache.append(BGEEmbedder(device="mps"))
        return _embedder_cache[0]

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        hashing_kv: Any = None,
        **kwargs,
    ) -> str:
        import ollama as _ollama
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        resp = _ollama.generate(
            model=llm_model,
            prompt=full_prompt,
            options={"temperature": 0.1, "num_predict": 512},
        )
        return resp["response"]

    async def embed_func(texts: list[str]) -> list[list[float]]:
        import numpy as np
        loop = asyncio.get_event_loop()
        embedder = _get_embedder()
        # Run CPU-bound embedding in executor to not block event loop
        result = await loop.run_in_executor(None, lambda: embedder.embed(texts, batch_size=4))
        # LightRAG expects a numpy array (checks .size attribute)
        return np.array(result, dtype=np.float32)

    embedding_func = EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=8192,
        func=embed_func,
    )

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embedding_func,
    )


async def build_graph_async(
    corpus: dict[str, str],
    working_dir: str,
    llm_model: str,
    embed_dim: int,
    batch_size: int = 4,
) -> Any:
    """Insert all corpus articles into LightRAG and build the knowledge graph."""
    rag = _make_lightrag(working_dir, llm_model, embed_dim)
    await rag.initialize_storages()

    texts = list(corpus.values())
    total = len(texts)
    print(f"Building graph from {total} articles (batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        await rag.ainsert(batch)
        done = min(i + batch_size, total)
        print(f"  Inserted {done}/{total} articles")

    print("Graph build complete.")
    return rag


def build_graph(
    config_path: str = "stage_4/config.toml",
    top_n: int = 50,
) -> Any:
    """Synchronous wrapper: build LightRAG graph from MultiHopRAG corpus.

    Default: top_n=50 most-referenced articles (~15 min build on M3).
    These 50 articles cover 38.7% of all 2,556 multi-hop queries.
    Full 609-article build takes ~33 hours — only viable overnight.
    """
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    corpus_path = Path(cfg["corpus"]["corpus_path"])
    working_dir = cfg["lightrag"]["working_dir"]

    if corpus_path.exists():
        print(f"Loading corpus from {corpus_path}...")
        corpus = json.loads(corpus_path.read_text())
    else:
        print(f"Building top-{top_n} coverage-optimised corpus from MultiHopRAG...")
        corpus = prepare_corpus_json(
            top_n_by_query_coverage=top_n,
            output_path=corpus_path,
        )
        print(f"Corpus: {len(corpus)} articles covering ~{top_n*0.77:.0f}% of queries")

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    return asyncio.run(
        build_graph_async(
            corpus=corpus,
            working_dir=working_dir,
            llm_model=cfg["llm"]["model"],
            embed_dim=cfg["embedding"]["dim"],
            batch_size=cfg["embedding"]["batch_size"],
        )
    )


def load_existing_graph(config_path: str = "stage_4/config.toml") -> Any:
    """Load an already-built LightRAG graph from disk (no rebuild)."""
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    working_dir = cfg["lightrag"]["working_dir"]
    if not Path(working_dir).exists():
        raise RuntimeError(
            f"Graph not found at {working_dir}. "
            "Run: uv run python -m stage_4.graph_builder"
        )

    return _make_lightrag(
        working_dir=working_dir,
        llm_model=cfg["llm"]["model"],
        embed_dim=cfg["embedding"]["dim"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Stage 4 LightRAG graph")
    parser.add_argument("--config", default="stage_4/config.toml")
    parser.add_argument("--n-articles", type=int, default=None,
                        help="Take first N articles (for quick smoke tests)")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Top-N most query-covering articles (default: 50, ~15 min build)")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = tomllib.load(fh)

    if args.n_articles:
        # Quick test mode: first N articles from corpus
        corpus = prepare_corpus_json(n_articles=args.n_articles)
    else:
        # Production mode: top-N by query coverage
        corpus = prepare_corpus_json(top_n_by_query_coverage=args.top_n)
        # Save for reuse
        import json as _json
        Path(cfg["corpus"]["corpus_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg["corpus"]["corpus_path"]).write_text(_json.dumps(corpus, indent=2))
        print(f"Corpus saved: {len(corpus)} articles")

    Path(cfg["lightrag"]["working_dir"]).mkdir(parents=True, exist_ok=True)
    asyncio.run(build_graph_async(
        corpus=corpus,
        working_dir=cfg["lightrag"]["working_dir"],
        llm_model=cfg["llm"]["model"],
        embed_dim=cfg["embedding"]["dim"],
        batch_size=cfg["embedding"]["batch_size"],
    ))
    print("Graph build complete.")
