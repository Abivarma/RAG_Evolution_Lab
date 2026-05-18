from __future__ import annotations
"""Stage 3 Agentic RAG harness entry point.

Usage:
  uv run python -m stage_3.harness --benchmark ragbench --n-samples 100
  uv run python -m stage_3.harness --benchmark ragbench --n-samples 50 --no-generation
"""

import argparse
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.eval.datasets import BENCHMARK_LOADERS
from shared.eval.harness import EvalHarness
from shared.eval.reporter import aggregate, check_regression, print_summary, save
from stage_2.pipeline import load_pipeline as load_stage2_pipeline
from stage_3.graph import build_graph
from stage_3.state import make_initial_state


class Stage3Harness(EvalHarness):
    """Stage 3: Agentic RAG via LangGraph supervisor-validator-synthesizer loop."""

    stage = 3

    def __init__(self, graph: Any, top_k: int = 10) -> None:
        self.graph = graph
        self.top_k = top_k
        self._last_state: dict | None = None

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        raise NotImplementedError("Use retrieve_item() — graph needs full item context")

    def retrieve_item(self, item: dict[str, Any], top_k: int = 10) -> list[str]:
        """Run the full agentic graph; cache state for generate() call."""
        passages = item.get("passages")
        initial = make_initial_state(item["query"], passages=passages)
        final_state = self.graph.invoke(initial)
        self._last_state = final_state
        return final_state["retrieved_ids"]

    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        """Return the answer already produced by the synthesizer node."""
        if self._last_state and self._last_state.get("answer"):
            return self._last_state["answer"], 0, 0
        return "", 0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 — Agentic RAG harness")
    parser.add_argument("--benchmark", default="ragbench", choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args()

    with open("stage_3/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    stage2_pipeline = load_stage2_pipeline()

    graph = build_graph(
        pipeline=stage2_pipeline,
        max_iterations=cfg["graph"]["max_iterations"],
        top_k=cfg["retrieval"]["top_k"],
        top_k_context=5,
    )

    harness = Stage3Harness(graph=graph, top_k=cfg["run"]["top_k"])
    queries = BENCHMARK_LOADERS[args.benchmark](n_samples=args.n_samples)
    print(f"Running {len(queries)} queries through Stage 3 Agentic RAG...")
    print(f"Config: max_iterations={cfg['graph']['max_iterations']}, model=qwen2.5:14b")

    results = harness.run(
        queries,
        top_k=cfg["run"]["top_k"],
        include_generation=not args.no_generation,
    )
    metrics = aggregate(results, stage=3, benchmark=args.benchmark)

    output_dir = Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or output_dir / f"{ts}.json"

    prev_runs = sorted(output_dir.glob("*.json"))
    if prev_runs:
        check_regression(metrics, prev_runs[-1])

    save(metrics, output_path)
    print_summary(metrics)
    print(f"Results saved to {output_path}")

    if results and harness._last_state:
        print("\n=== Reasoning trace (last query) ===")
        for step in harness._last_state.get("reasoning_trace", []):
            print(f"  {step}")


if __name__ == "__main__":
    main()
