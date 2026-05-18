from __future__ import annotations

import tomllib

from stage_1.embedder import BGEEmbedder
from stage_1.generator import OllamaGenerator
from stage_1.retriever import QdrantRetriever


def load_pipeline(
    config_path: str = "stage_1/config.toml",
) -> tuple[QdrantRetriever, OllamaGenerator]:
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    embedder = BGEEmbedder(
        model_name=cfg["embedding"]["model"],
        device=cfg["embedding"]["device"],
    )
    retriever = QdrantRetriever(
        collection=cfg["qdrant"]["collection"],
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        embedder=embedder,
    )
    generator = OllamaGenerator(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    return retriever, generator
