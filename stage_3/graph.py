from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from stage_3.agents import (
    retriever_node,
    supervisor_node,
    synthesizer_node,
    validator_node,
)
from stage_3.state import AgentState


def _should_continue(state: AgentState, max_iterations: int) -> Literal["supervisor", "synthesizer"]:
    """Conditional edge: loop back to supervisor if not validated and under limit."""
    if state["validated"]:
        return "synthesizer"
    if state["iteration"] >= max_iterations:
        return "synthesizer"
    return "supervisor"


def build_graph(
    pipeline: Any,
    max_iterations: int = 3,
    top_k: int = 10,
    top_k_context: int = 5,
) -> Any:
    """Build and compile the LangGraph StateGraph for Stage 3.

    Graph flow:
      supervisor -> retriever -> validator --(accept)--> synthesizer -> END
                        ^              |
                        +-- (reject) --+
    """
    graph = StateGraph(AgentState)

    def _supervisor(state: AgentState) -> dict:
        return supervisor_node(state, pipeline, max_iterations=max_iterations)

    def _retriever(state: AgentState) -> dict:
        return retriever_node(state, pipeline, top_k=top_k)

    def _validator(state: AgentState) -> dict:
        return validator_node(state)

    def _synthesizer(state: AgentState) -> dict:
        return synthesizer_node(state, top_k_context=top_k_context)

    def _should_continue_bound(state: AgentState) -> Literal["supervisor", "synthesizer"]:
        return _should_continue(state, max_iterations=max_iterations)

    graph.add_node("supervisor", _supervisor)
    graph.add_node("retriever", _retriever)
    graph.add_node("validator", _validator)
    graph.add_node("synthesizer", _synthesizer)

    graph.set_entry_point("supervisor")

    graph.add_edge("supervisor", "retriever")
    graph.add_edge("retriever", "validator")
    graph.add_edge("synthesizer", END)

    graph.add_conditional_edges(
        "validator",
        _should_continue_bound,
        {"supervisor": "supervisor", "synthesizer": "synthesizer"},
    )

    return graph.compile()
