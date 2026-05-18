from __future__ import annotations
"""Stage 2 harness entry point.

Usage:
  uv run python -m stage_2.harness --benchmark ragbench --n-samples 100
  uv run python -m stage_2.harness --benchmark ragbench --n-samples 100 --ablation
  uv run python -m stage_2.harness --no-reranker --no-hybrid
"""

import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.eval.datasets import BENCHMARK_LOADERS
from shared.eval.harness import EvalHarness
from shared.eval.reporter import aggregate, print_summary, save
from stage_2.pipeline import AblationFlags, Stage2Pipeline, load_pipeline


class Stage2Harness(EvalHarness):
    stage = 2

    def __init__(self, pipeline: Stage2Pipeline) -> None:
        self.pipeline = pipeline

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        raise NotImplementedError("Use retrieve_item() for RAGBench closed-corpus")

    def retrieve_item(self, item: dict[str, Any], top_k: int = 10) -> list[str]:
        if "passages" in item:
            return self.pipeline.retrieve_from_passages(item["query"], item["passages"], top_k=top_k)
        return self.pipeline.embedder.embed_and_rank_passages(item["query"], {}, top_k=top_k)

    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        return self.pipeline.generator.generate(query, retrieved_ids[: self.pipeline.top_k_final])


def _run_one(
    flags: AblationFlags,
    benchmark: str,
    n_samples: int,
    output_path: Path,
    top_k: int,
    include_generation: bool,
) -> Any:
    label_parts = []
    if flags.use_query_rewriting: label_parts.append("rewrite")
    if flags.use_hybrid:          label_parts.append("hybrid")
    if flags.use_reranker:        label_parts.append("reranker")
    if flags.use_semantic_chunking: label_parts.append("sem-chunk")
    label = "+".join(label_parts) if label_parts else "dense-only"
    print(f"\n{'='*60}\nAblation: [{label}]\n{'='*60}")

    pipeline = load_pipeline(flags=flags)
    harness = Stage2Harness(pipeline)
    queries = BENCHMARK_LOADERS[benchmark](n_samples=n_samples)
    results = harness.run(queries, top_k=top_k, include_generation=include_generation)
    metrics = aggregate(results, stage=2, benchmark=benchmark)
    print_summary(metrics)
    save(metrics, output_path)
    print(f"Saved -> {output_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Advanced RAG harness")
    parser.add_argument("--benchmark", default="ragbench", choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ablation", action="store_true",
                        help="Run 6-way ablation suite (all on + each removed one at a time)")
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-rewriting",   action="store_true")
    parser.add_argument("--no-hybrid",      action="store_true")
    parser.add_argument("--no-reranker",    action="store_true")
    parser.add_argument("--no-sem-chunk",   action="store_true")
    args = parser.parse_args()

    with open("stage_2/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    output_dir = Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    top_k = cfg["run"]["top_k"]
    include_gen = not args.no_generation

    if args.ablation:
        configs = {
            "all":          AblationFlags(True,  True,  True,  True),
            "no_rewriting": AblationFlags(False, True,  True,  True),
            "no_hybrid":    AblationFlags(True,  False, True,  True),
            "no_reranker":  AblationFlags(True,  True,  False, True),
            "no_sem_chunk": AblationFlags(True,  True,  True,  False),
            "dense_only":   AblationFlags(False, False, False, False),
        }
        all_metrics: dict[str, Any] = {}
        for name, flags in configs.items():
            out = output_dir / f"{ts}_{name}.json"
            all_metrics[name] = _run_one(flags, args.benchmark, args.n_samples, out, top_k, include_gen)

        print(f"\n{'='*70}\nABLATION SUMMARY\n{'='*70}")
        print(f"{'Config':<20}  {'R@10':>8}  {'MRR':>8}  {'nDCG@10':>9}  {'p99ms':>8}")
        print("-" * 70)
        for name, m in all_metrics.items():
            print(f"{name:<20}  {m.recall_at_10:>8.4f}  {m.mrr:>8.4f}  {m.ndcg_at_10:>9.4f}  {m.latency_p99_ms:>8.0f}")
    else:
        flags = AblationFlags(
            use_query_rewriting=not args.no_rewriting,
            use_hybrid=not args.no_hybrid,
            use_reranker=not args.no_reranker,
            use_semantic_chunking=not args.no_sem_chunk,
        )
        out = args.output or output_dir / f"{ts}.json"
        _run_one(flags, args.benchmark, args.n_samples, out, top_k, include_gen)

        prev_runs = sorted(f for f in output_dir.glob("*.json") if f != out)
        if prev_runs:
            metrics_data = json.loads(out.read_text())
            prev_data = json.loads(prev_runs[-1].read_text())
            for key in ["recall_at_10", "mrr", "ndcg_at_10"]:
                old, new = prev_data.get(key, 0.0), metrics_data.get(key, 0.0)
                if old > 0 and (old - new) / old > 0.05:
                    print(f"WARNING: {key} {old:.4f} -> {new:.4f} (regression)")


if __name__ == "__main__":
    main()
