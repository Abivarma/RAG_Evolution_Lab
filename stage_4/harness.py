from __future__ import annotations
"""Stage 4 GraphRAG harness entry point.

Usage:
  # Build full graph first (~30-90 min):
  uv run python -m stage_4.graph_builder

  # Quick test with 5 articles:
  uv run python -m stage_4.graph_builder --n-articles 5

  # Run eval (graph must be built):
  uv run python -m stage_4.harness --benchmark multihop --n-samples 20 --no-generation
  uv run python -m stage_4.harness --benchmark multihop --n-samples 3 --graph-only
"""

import argparse
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.eval.datasets import BENCHMARK_LOADERS
from shared.eval.harness import EvalHarness
from shared.eval.reporter import aggregate, check_regression, print_summary, save
from stage_4.graph_builder import load_existing_graph
from stage_4.retriever import GraphRetriever


class Stage4Harness(EvalHarness):
    """Stage 4: GraphRAG — LightRAG graph search fused with Stage 2 vector retrieval."""

    stage = 4

    def __init__(self, retriever: GraphRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        raise NotImplementedError("Use retrieve_item() — passages required for MultiHopRAG")

    def retrieve_item(self, item: dict[str, Any], top_k: int = 10) -> list[str]:
        passages = item.get("passages", {})
        return self.retriever.retrieve_from_passages(item["query"], passages, top_k=top_k)

    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        return "", 0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 — GraphRAG harness")
    parser.add_argument("--benchmark", default="multihop", choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--graph-only", action="store_true",
                        help="Disable Stage 2 vector backend; graph search only")
    args = parser.parse_args()

    with open("stage_4/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    print("Loading LightRAG graph...")
    rag = load_existing_graph()
    # initialize_storages must be called before any query
    import asyncio as _asyncio
    _asyncio.run(rag.initialize_storages())

    stage2_pipeline = None
    if not args.graph_only:
        try:
            from stage_2.pipeline import AblationFlags, load_pipeline as load_s2
            flags = AblationFlags(
                use_query_rewriting=False,
                use_hybrid=True,
                use_reranker=True,
                use_semantic_chunking=True,
            )
            print("Loading Stage 2 pipeline (vector backend)...")
            stage2_pipeline = load_s2(flags=flags)
            print("Stage 2 loaded.")
        except Exception as e:
            print(f"Warning: Stage 2 pipeline failed to load ({e}), running graph-only.")

    retriever = GraphRetriever(
        rag=rag,
        stage2_pipeline=stage2_pipeline,
        top_k_graph=cfg["retrieval"]["top_k_graph"],
        top_k_vector=cfg["retrieval"]["top_k_vector"],
        rrf_k=cfg["retrieval"]["rrf_k"],
        query_mode=cfg["lightrag"]["query_mode"],
    )

    harness = Stage4Harness(retriever=retriever)
    queries = BENCHMARK_LOADERS[args.benchmark](n_samples=args.n_samples)
    mode = "graph-only" if (args.graph_only or stage2_pipeline is None) else "graph+vector"
    print(f"Running {len(queries)} queries — Stage 4 GraphRAG [{mode}]...")

    results = harness.run(
        queries,
        top_k=cfg["run"]["top_k"],
        include_generation=not args.no_generation,
    )
    metrics = aggregate(results, stage=4, benchmark=args.benchmark)

    output_dir = Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "_graph_only" if (args.graph_only or stage2_pipeline is None) else ""
    output_path = args.output or output_dir / f"{ts}{suffix}.json"

    prev_runs = sorted(output_dir.glob("*.json"))
    if prev_runs:
        check_regression(metrics, prev_runs[-1])

    save(metrics, output_path)
    print_summary(metrics)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
