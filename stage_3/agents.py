from __future__ import annotations

import re
from typing import Any

import ollama

from stage_3.state import AgentState, RetrievalStrategy


def _parse_strategy(text: str) -> RetrievalStrategy:
    """Extract retrieval strategy from supervisor LLM output."""
    text_lower = text.lower()
    if "bm25" in text_lower:
        return RetrievalStrategy.BM25
    if "dense" in text_lower:
        return RetrievalStrategy.DENSE
    return RetrievalStrategy.HYBRID


def _parse_confidence(text: str) -> float:
    """Extract a 0-1 confidence score from validator output."""
    match = re.search(r"confidence[:\s]+([0-9]*\.?[0-9]+)", text.lower())
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    # Fallback: YES = 0.8, NO = 0.2
    return 0.8 if text.strip().upper().startswith("YES") else 0.2


def supervisor_node(
    state: AgentState,
    pipeline: Any,
    max_iterations: int = 3,
) -> dict:
    """Supervisor: decides retrieval strategy and optionally decomposes the query."""
    iteration = state["iteration"]
    query = state["query"]

    if iteration == 0:
        prompt = (
            f"You are a retrieval supervisor. Given this query, decide:\n"
            f"1. Which retrieval strategy to use: hybrid, dense, or bm25\n"
            f"2. Whether to decompose into sub-queries (list them if yes, empty if no)\n\n"
            f"Query: {query}\n\n"
            f"Respond in format:\nstrategy: <hybrid|dense|bm25>\nsub_queries: <comma-separated or empty>"
        )
    else:
        current = state["strategy"]
        alt = RetrievalStrategy.DENSE if current == RetrievalStrategy.HYBRID else RetrievalStrategy.HYBRID
        prompt = (
            f"The previous retrieval (strategy={current.value}) was rejected by the validator.\n"
            f"Switch to strategy={alt.value} and rephrase the query if needed.\n\n"
            f"Original query: {query}\n"
            f"Iteration: {iteration}/{max_iterations}\n\n"
            f"Respond in format:\nstrategy: <hybrid|dense|bm25>\nsub_queries: <comma-separated or empty>"
        )

    resp = ollama.generate(
        model="qwen2.5:14b",
        prompt=prompt,
        options={"temperature": 0.1, "num_predict": 128},
    )
    text = resp["response"]
    strategy = _parse_strategy(text)

    sub_queries: list[str] = []
    sq_match = re.search(r"sub_queries?:\s*(.+)", text, re.IGNORECASE)
    if sq_match:
        raw = sq_match.group(1).strip()
        if raw and raw.lower() not in ("", "none", "empty"):
            sub_queries = [q.strip() for q in raw.split(",") if q.strip()]

    trace_entry = f"[supervisor iter={iteration}] strategy={strategy.value}, sub_queries={sub_queries}"

    return {
        "strategy": strategy,
        "sub_queries": sub_queries,
        "reasoning_trace": state["reasoning_trace"] + [trace_entry],
    }


def retriever_node(
    state: AgentState,
    pipeline: Any,
    top_k: int = 10,
) -> dict:
    """Retriever: executes retrieval using the supervisor's chosen strategy."""
    query = state["query"]
    passages = state["passages"]
    sub_queries = state.get("sub_queries") or []
    iteration = state["iteration"]

    queries_to_run = sub_queries if sub_queries else [query]

    all_ids: list[str] = []
    seen: set[str] = set()
    for q in queries_to_run:
        if passages is not None:
            ids = pipeline.retrieve_from_passages(q, passages, top_k=top_k)
        else:
            ids = pipeline.retriever.retrieve(q, top_k=top_k)
        for doc_id in ids:
            if doc_id not in seen:
                seen.add(doc_id)
                all_ids.append(doc_id)

    trace_entry = f"[retriever iter={iteration+1}] retrieved {len(all_ids)} docs via {state['strategy'].value}"

    return {
        "retrieved_ids": all_ids[:top_k],
        "iteration": iteration + 1,
        "reasoning_trace": state["reasoning_trace"] + [trace_entry],
    }


def validator_node(state: AgentState) -> dict:
    """Validator: asks Qwen whether retrieved passages answer the query."""
    query = state["query"]
    passages = state.get("passages") or {}
    retrieved_ids = state["retrieved_ids"]

    context_parts = []
    for doc_id in retrieved_ids[:5]:
        if doc_id in passages:
            context_parts.append(f"[{doc_id}] {passages[doc_id]}")
    context = "\n\n".join(context_parts) if context_parts else "(no passages retrieved)"

    prompt = (
        f"You are a retrieval validator. Given the question and retrieved passages, "
        f"answer YES or NO: do these passages contain enough information to answer the question?\n\n"
        f"Question: {query}\n\n"
        f"Passages:\n{context}\n\n"
        f"Answer YES or NO, then explain briefly. End with 'Confidence: X.X' (0.0-1.0)."
    )

    resp = ollama.generate(
        model="qwen2.5:14b",
        prompt=prompt,
        options={"temperature": 0.0, "num_predict": 128},
    )
    text = resp["response"]
    score = _parse_confidence(text)
    accepted = score >= 0.5

    trace_entry = (
        f"[validator iter={state['iteration']}] "
        f"{'ACCEPTED' if accepted else 'REJECTED'} score={score:.2f}"
    )

    return {
        "validated": accepted,
        "validator_score": score,
        "reasoning_trace": state["reasoning_trace"] + [trace_entry],
    }


def synthesizer_node(state: AgentState, top_k_context: int = 5) -> dict:
    """Synthesizer: generates final answer with citations from validated context."""
    query = state["query"]
    passages = state.get("passages") or {}
    retrieved_ids = state["retrieved_ids"]

    context_parts = []
    for doc_id in retrieved_ids[:top_k_context]:
        if doc_id in passages:
            context_parts.append(f"[{doc_id}] {passages[doc_id]}")
    context = "\n\n".join(context_parts) if context_parts else "(no context)"

    prompt = (
        f"Use the following passages to answer the question. "
        f"Cite passages by their IDs (e.g. [0], [2]).\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    resp = ollama.generate(
        model="qwen2.5:14b",
        prompt=prompt,
        options={"temperature": 0.1, "num_predict": 512},
    )
    answer = resp["response"]
    trace_entry = f"[synthesizer] generated answer ({len(answer)} chars)"

    return {
        "answer": answer,
        "reasoning_trace": state["reasoning_trace"] + [trace_entry],
    }
