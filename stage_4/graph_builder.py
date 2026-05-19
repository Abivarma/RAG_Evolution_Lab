from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from typing import Any


def prepare_corpus_json(
    n_articles: int | None = None,
    output_path: Path | None = None,
) -> dict[str, str]:
    """Build {url: title + body} dict from MultiHopRAG corpus.

    Returns dict of unique article URL -> concatenated title + body text.
    Saves to output_path as JSON if provided.
    """
    from datasets import load_dataset

    corpus_ds = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    if n_articles:
        corpus_ds = corpus_ds.select(range(min(n_articles, len(corpus_ds))))

    articles: dict[str, str] = {}
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


def build_graph(config_path: str = "stage_4/config.toml") -> Any:
    """Synchronous wrapper: build full LightRAG graph from MultiHopRAG corpus."""
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    corpus_path = Path(cfg["corpus"]["corpus_path"])
    working_dir = cfg["lightrag"]["working_dir"]

    if corpus_path.exists():
        print(f"Loading corpus from {corpus_path}...")
        corpus = json.loads(corpus_path.read_text())
    else:
        print("Downloading MultiHopRAG corpus...")
        corpus = prepare_corpus_json(output_path=corpus_path)

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
                        help="Limit corpus for quick tests (omit for full 609)")
    args = parser.parse_args()

    if args.n_articles:
        with open(args.config, "rb") as fh:
            cfg = tomllib.load(fh)
        corpus = prepare_corpus_json(n_articles=args.n_articles)
        asyncio.run(build_graph_async(
            corpus=corpus,
            working_dir=cfg["lightrag"]["working_dir"],
            llm_model=cfg["llm"]["model"],
            embed_dim=cfg["embedding"]["dim"],
            batch_size=cfg["embedding"]["batch_size"],
        ))
    else:
        build_graph(args.config)
        print("Full graph build complete.")
