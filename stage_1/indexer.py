from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from stage_1.embedder import BGEEmbedder, chunk_text


def build_qdrant_index(
    abstracts_path: Path,
    collection: str,
    host: str,
    port: int,
    embedding_dim: int,
    chunk_size: int,
    batch_size: int,
    device: str,
) -> None:
    print(f"Loading abstracts from {abstracts_path}...")
    corpus = json.loads(abstracts_path.read_text())
    print(f"Loaded {len(corpus):,} documents.")

    client = QdrantClient(host=host, port=port)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    embedder = BGEEmbedder(device=device)
    all_chunks: list[dict] = []
    for doc in corpus:
        text = doc.get("title", "") + " " + doc.get("abstract", "")
        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=0)):
            all_chunks.append({"text": chunk, "doc_id": doc["id"], "chunk_idx": idx})

    print(f"Total chunks: {len(all_chunks):,}. Embedding in batches of {batch_size}...")
    t0 = time.perf_counter()

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors = embedder.embed([c["text"] for c in batch], batch_size=batch_size)
        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=str(uuid4()), vector=vectors[j], payload=batch[j])
                for j in range(len(batch))
            ],
        )
        done = min(i + batch_size, len(all_chunks))
        print(f"  Indexed {done:,}/{len(all_chunks):,} chunks...")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=None)
    args = parser.parse_args()

    with open("stage_1/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    build_qdrant_index(
        abstracts_path=args.corpus or Path(cfg["data"]["abstracts_path"]),
        collection=cfg["qdrant"]["collection"],
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        embedding_dim=cfg["embedding"]["dimension"],
        chunk_size=cfg["chunking"]["chunk_size"],
        batch_size=cfg["embedding"]["batch_size"],
        device=cfg["embedding"]["device"],
    )
