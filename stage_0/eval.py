from __future__ import annotations
"""Stage 0 eval harness entry point.

Usage: uv run python -m stage_0.eval --benchmark ragbench [--n-samples 1000]
"""

import argparse
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from shared.eval.datasets import BENCHMARK_LOADERS
from shared.eval.harness import EvalHarness
from shared.eval.reporter import aggregate, check_regression, print_summary, save
from stage_0.retriever import BM25Retriever


class Stage0Harness(EvalHarness):
    stage = 0

    def __init__(self, retriever: BM25Retriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self.retriever.retrieve(query, top_k=top_k)

    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        # Stage 0 has no LLM — return passage IDs as the "answer"
        return " | ".join(retrieved_ids[:5]), 0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 — BM25 Baseline eval")
    parser.add_argument("--benchmark", default="ragbench", choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open("stage_0/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    index_path = Path(cfg["data"]["index_path"])
    if not index_path.exists():
        print(f"Index not found at {index_path}.")
        print("Run: uv run python -m stage_0.indexer")
        raise SystemExit(1)

    print(f"Loading BM25 index from {index_path}...")
    retriever = BM25Retriever.load(index_path)

    harness = Stage0Harness(retriever)
    queries = BENCHMARK_LOADERS[args.benchmark](n_samples=args.n_samples)
    print(f"Running {len(queries)} queries...")

    results = harness.run(queries, top_k=cfg["bm25"]["top_k"])
    metrics = aggregate(results, stage=0, benchmark=args.benchmark)

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
