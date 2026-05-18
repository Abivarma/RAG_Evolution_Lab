from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict


class RetrievalStrategy(str, Enum):
    HYBRID = "hybrid"
    DENSE = "dense"
    BM25 = "bm25"


class AgentState(TypedDict):
    """Shared state flowing through all LangGraph nodes.

    Fields:
      query:           The user's original question (never mutated).
      passages:        Dict of {doc_id: text} for closed-corpus benchmarks (RAGBench).
                       None for open-corpus runs where Qdrant is used.
      retrieved_ids:   Ranked list of doc_ids returned by the last retrieval.
      iteration:       How many retrieval+validation cycles have completed.
      validated:       True once the validator accepts the retrieved context.
      answer:          Final generated answer (set by synthesizer).
      reasoning_trace: List of human-readable steps for auditability.
      strategy:        Retrieval strategy the supervisor selected.
      sub_queries:     Optional list of decomposed sub-queries from supervisor.
      validator_score: Float 0-1 confidence score from last validator run.
    """

    query: str
    passages: dict[str, str] | None
    retrieved_ids: list[str]
    iteration: int
    validated: bool
    answer: str | None
    reasoning_trace: list[str]
    strategy: RetrievalStrategy
    sub_queries: list[str]
    validator_score: float


def make_initial_state(
    query: str,
    passages: dict[str, str] | None = None,
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
) -> AgentState:
    """Construct a clean initial state for a new query."""
    return AgentState(
        query=query,
        passages=passages,
        retrieved_ids=[],
        iteration=0,
        validated=False,
        answer=None,
        reasoning_trace=[],
        strategy=strategy,
        sub_queries=[],
        validator_score=0.0,
    )
