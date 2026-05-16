from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

from stage_0.retriever import BM25Retriever


def build_index(abstracts_path: Path, index_path: Path, k1: float, b: float) -> BM25Retriever:
    print(f"Loading abstracts from {abstracts_path}...")
    corpus = json.loads(abstracts_path.read_text())
    print(f"Loaded {len(corpus):,} documents.")

    retriever = BM25Retriever(k1=k1, b=b)
    t0 = time.perf_counter()
    retriever.index(corpus)
    elapsed = time.perf_counter() - t0
    print(f"Index built in {elapsed:.1f}s")

    retriever.save(index_path)
    size_mb = index_path.stat().st_size / 1e6
    print(f"Index saved to {index_path} ({size_mb:.1f} MB)")
    return retriever


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open("stage_0/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    build_index(
        abstracts_path=args.corpus or Path(cfg["data"]["abstracts_path"]),
        index_path=args.output or Path(cfg["data"]["index_path"]),
        k1=cfg["bm25"]["k1"],
        b=cfg["bm25"]["b"],
    )
