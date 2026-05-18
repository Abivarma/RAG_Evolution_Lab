from __future__ import annotations

from stage_3.state import AgentState, RetrievalStrategy, make_initial_state


def test_initial_state_has_required_fields() -> None:
    state = make_initial_state("What is RAG?", {"0": "RAG passage text"})
    assert state["query"] == "What is RAG?"
    assert state["passages"] == {"0": "RAG passage text"}
    assert state["retrieved_ids"] == []
    assert state["iteration"] == 0
    assert state["validated"] is False
    assert state["answer"] is None
    assert state["reasoning_trace"] == []
    assert state["strategy"] == RetrievalStrategy.HYBRID


def test_initial_state_without_passages() -> None:
    state = make_initial_state("query")
    assert state["passages"] is None


def test_retrieval_strategy_values() -> None:
    assert RetrievalStrategy.HYBRID == "hybrid"
    assert RetrievalStrategy.DENSE == "dense"
    assert RetrievalStrategy.BM25 == "bm25"


def test_state_reasoning_trace_is_list() -> None:
    state = make_initial_state("q")
    state["reasoning_trace"].append("step 1")
    assert state["reasoning_trace"] == ["step 1"]
