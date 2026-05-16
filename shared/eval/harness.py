from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import psutil

from shared.eval.metrics import compute_retrieval_metrics


@dataclass
class QueryResult:
    query_id: str
    query: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    answer: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    ram_peak_mb: float = 0.0
    relevance_grades: dict[str, int] = field(default_factory=dict)


@dataclass
class StageMetrics:
    stage: int
    benchmark: str
    n_queries: int
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float
    hit_rate_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    total_input_tokens: int
    total_output_tokens: int
    peak_ram_mb: float
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class EvalHarness(ABC):
    """Base class every stage's eval.py inherits from."""

    stage: int

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Return a ranked list of doc_ids for the query."""
        ...

    def retrieve_item(self, item: dict[str, Any], top_k: int = 10) -> list[str]:
        """Retrieve for a full item dict. Override in stages that need passage-level routing."""
        return self.retrieve(item["query"], top_k=top_k)

    @abstractmethod
    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        """Return (answer_text, input_tokens, output_tokens)."""
        ...

    def run(
        self,
        queries: list[dict[str, Any]],
        top_k: int = 10,
        include_generation: bool = False,
    ) -> list[QueryResult]:
        results: list[QueryResult] = []
        proc = psutil.Process()

        for item in queries:
            t0 = time.perf_counter()
            ram_before = proc.memory_info().rss / 1024 / 1024

            retrieved = self.retrieve_item(item, top_k=top_k)

            answer = None
            in_toks = out_toks = 0
            if include_generation:
                answer, in_toks, out_toks = self.generate(item["query"], retrieved)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            ram_after = proc.memory_info().rss / 1024 / 1024

            results.append(QueryResult(
                query_id=str(item.get("id", item["query"][:16])),
                query=item["query"],
                retrieved_ids=retrieved,
                relevant_ids=set(item.get("relevant_ids", [])),
                answer=answer,
                latency_ms=elapsed_ms,
                input_tokens=in_toks,
                output_tokens=out_toks,
                ram_peak_mb=max(ram_before, ram_after),
                relevance_grades=item.get("relevance_grades", {}),
            ))

        return results
