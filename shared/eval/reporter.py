from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from shared.eval.harness import QueryResult, StageMetrics
from shared.eval.metrics import compute_retrieval_metrics


def aggregate(results: list[QueryResult], stage: int, benchmark: str) -> StageMetrics:
    """Aggregate per-query results into StageMetrics."""
    all_metrics: list[dict[str, float]] = []
    latencies: list[float] = []
    total_in = total_out = 0
    peak_ram = 0.0

    for r in results:
        m = compute_retrieval_metrics(r.retrieved_ids, r.relevant_ids, r.relevance_grades or None)
        all_metrics.append(m)
        latencies.append(r.latency_ms)
        total_in += r.input_tokens
        total_out += r.output_tokens
        peak_ram = max(peak_ram, r.ram_peak_mb)

    def mean(key: str) -> float:
        return statistics.mean(row[key] for row in all_metrics)

    latencies.sort()
    n = len(latencies)

    def pct(p: float) -> float:
        return latencies[min(int(p * n), n - 1)]

    return StageMetrics(
        stage=stage,
        benchmark=benchmark,
        n_queries=n,
        recall_at_5=mean("recall_at_5"),
        recall_at_10=mean("recall_at_10"),
        recall_at_20=mean("recall_at_20"),
        mrr=mean("mrr"),
        ndcg_at_10=mean("ndcg_at_10"),
        hit_rate_10=mean("hit_rate_10"),
        latency_p50_ms=pct(0.50),
        latency_p95_ms=pct(0.95),
        latency_p99_ms=pct(0.99),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        peak_ram_mb=peak_ram,
    )


def save(metrics: StageMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {**metrics.__dict__, "saved_at": datetime.now(timezone.utc).isoformat()}
    output_path.write_text(json.dumps(data, indent=2))


def print_summary(metrics: StageMetrics) -> None:
    print(f"\n{'='*60}")
    print(f"Stage {metrics.stage} | Benchmark: {metrics.benchmark} | N={metrics.n_queries}")
    print(f"{'='*60}")
    print(f"Recall@5:    {metrics.recall_at_5:.4f}")
    print(f"Recall@10:   {metrics.recall_at_10:.4f}  <- primary metric")
    print(f"Recall@20:   {metrics.recall_at_20:.4f}")
    print(f"MRR:         {metrics.mrr:.4f}")
    print(f"nDCG@10:     {metrics.ndcg_at_10:.4f}")
    print(f"Hit Rate@10: {metrics.hit_rate_10:.4f}")
    print(f"---")
    print(f"p50 latency: {metrics.latency_p50_ms:.1f}ms")
    print(f"p95 latency: {metrics.latency_p95_ms:.1f}ms")
    print(f"p99 latency: {metrics.latency_p99_ms:.1f}ms")
    print(f"---")
    print(f"Input tokens:  {metrics.total_input_tokens:,}")
    print(f"Output tokens: {metrics.total_output_tokens:,}")
    print(f"Peak RAM:      {metrics.peak_ram_mb:.1f} MB")
    print(f"{'='*60}\n")


def check_regression(new: StageMetrics, prev_path: Path, threshold: float = 0.05) -> None:
    """Warn on metrics that regressed >threshold vs the previous run."""
    if not prev_path.exists():
        return
    prev = json.loads(prev_path.read_text())
    for key in ["recall_at_10", "mrr", "ndcg_at_10"]:
        old_val = prev.get(key, 0.0)
        new_val = getattr(new, key, 0.0)
        if old_val > 0 and (old_val - new_val) / old_val > threshold:
            pct_drop = (old_val - new_val) / old_val * 100
            print(f"WARNING: {key} dropped {old_val:.4f} -> {new_val:.4f} ({pct_drop:.1f}%)")
