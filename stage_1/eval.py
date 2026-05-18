from __future__ import annotations
"""Stage 1 eval harness entry point.

Usage: uv run python -m stage_1.eval --benchmark ragbench [--n-samples 1000] [--no-generation]
"""

import argparse
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.eval.datasets import BENCHMARK_LOADERS
from shared.eval.harness import EvalHarness
from shared.eval.reporter import aggregate, check_regression, print_summary, save
from stage_1.embedder import BGEEmbedder
from stage_1.generator import OllamaGenerator
from stage_1.retriever import QdrantRetriever


class Stage1Harness(EvalHarness):
    stage = 1

    def __init__(
        self,
        retriever: QdrantRetriever,
        generator: OllamaGenerator,
        embedder: BGEEmbedder,
        top_k_context: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.embedder = embedder
        self.top_k_context = top_k_context

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self.retriever.retrieve(query, top_k=top_k)

    def retrieve_item(self, item: dict[str, Any], top_k: int = 10) -> list[str]:
        """Use dense embedding ranking for closed-corpus items (e.g. RAGBench)."""
        if "passages" in item:
            return self.embedder.embed_and_rank_passages(
                item["query"], item["passages"], top_k=top_k
            )
        return self.retrieve(item["query"], top_k=top_k)

    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        return self.generator.generate(query, retrieved_ids[: self.top_k_context])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 — Naive RAG eval")
    parser.add_argument("--benchmark", default="ragbench", choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-generation", action="store_true",
                        help="Skip LLM generation (retrieval-only eval)")
    args = parser.parse_args()

    with open("stage_1/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    embedder = BGEEmbedder(
        model_name=cfg["embedding"]["model"],
        device=cfg["embedding"]["device"],
    )

    # For closed-corpus benchmarks (ragbench), we don't need Qdrant.
    # For open-corpus (future stages), load_pipeline() connects to Qdrant.
    if args.benchmark == "ragbench":
        # Closed-corpus: retrieval via dense embedding over per-query passages
        generator = OllamaGenerator(
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"]["max_tokens"],
        )
        # Dummy QdrantRetriever — won't be called for ragbench
        retriever = QdrantRetriever(
            collection=cfg["qdrant"]["collection"],
            host=cfg["qdrant"]["host"],
            port=cfg["qdrant"]["port"],
            embedder=embedder,
        )
    else:
        from stage_1.pipeline import load_pipeline
        retriever, generator = load_pipeline()

    harness = Stage1Harness(
        retriever=retriever,
        generator=generator,
        embedder=embedder,
        top_k_context=cfg["llm"]["top_k_context"],
    )

    queries = BENCHMARK_LOADERS[args.benchmark](n_samples=args.n_samples)
    print(f"Running {len(queries)} queries against Stage 1 Naive RAG...")

    results = harness.run(
        queries, top_k=cfg["eval"]["top_k"], include_generation=not args.no_generation
    )
    metrics = aggregate(results, stage=1, benchmark=args.benchmark)

    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or output_dir / f"{ts}.json"

    prev_runs = sorted(output_dir.glob("*.json"))
    if prev_runs:
        check_regression(metrics, prev_runs[-1])

    save(metrics, output_path)
    print_summary(metrics)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
