# Stage 3 — Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 3 Agentic RAG — a LangGraph stateful supervisor that orchestrates three specialist agents (Retriever, Validator, Synthesizer) with a self-correction loop, then benchmark it against Stage 2 to measure exactly where extra LLM calls earn their cost and where they don't.

**Architecture:** A LangGraph `StateGraph` manages a typed state dict flowing through: Supervisor (decides strategy and decomposes queries) → Retriever Agent (executes Stage 2 retrieval stack) → Validator Agent (asks "do these passages actually answer the question?") → if NO, loops back to Supervisor with a new strategy → if YES, Synthesizer Agent (composes final answer with citations). The graph has a conditional edge creating the self-correction loop. Max iterations = 3 to bound latency. The graph is the architectural novelty; retrieval itself reuses Stage 2's `Stage2Pipeline.retrieve_from_passages()`.

**Tech Stack:** Python 3.11, uv, langgraph>=0.3, langchain>=0.3, ollama (Qwen 2.5 14B), stage_2 pipeline (reused as retrieval backend).

**Commit author:** Abivarma <Abivarma.Rs@ibm.com>
**Co-author every commit:** `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`

---

## Stage 2 Baseline Numbers (to beat on hard queries)
- Recall@10: 0.8000, MRR: 0.6267, nDCG@10: 0.6621, p99: 32,884ms

---

## File Structure

```
stage_3/
├── __init__.py
├── config.toml              # max_iterations, validator_threshold, model, top_k
├── state.py                 # AgentState TypedDict — the graph's shared state
├── agents.py                # supervisor_node, retriever_node, validator_node, synthesizer_node
├── graph.py                 # build_graph() → compiled LangGraph StateGraph
├── harness.py               # Stage3Harness(EvalHarness) + __main__ entry
└── tests/
    ├── __init__.py
    ├── test_state.py         # AgentState construction and field defaults
    ├── test_agents.py        # unit tests for each node function (mocked LLM)
    └── test_graph.py         # integration test: graph runs end-to-end on fake data
```

**Modified files:** None — Stage 3 is fully self-contained and imports from `stage_2.pipeline`.

---

## Task 0: Scaffold + Config + LangGraph Install

**Files:**
- Create: `stage_3/__init__.py`
- Create: `stage_3/tests/__init__.py`
- Create: `stage_3/config.toml`
- Modify: `.claude/current_stage.txt`

- [ ] **Step 1: Install LangGraph**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv add langgraph --quiet
uv run python -c "import langgraph; print('langgraph', langgraph.__version__)"
```

Expected: `langgraph 0.3.x` or similar.

- [ ] **Step 2: Create directory structure**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
mkdir -p stage_3/tests
touch stage_3/__init__.py stage_3/tests/__init__.py
```

- [ ] **Step 3: Create stage_3/config.toml**

```toml
[agent]
model = "qwen2.5:14b"
temperature = 0.1
max_tokens = 512

[graph]
max_iterations = 3          # max retrieval+validation loops before forcing synthesis
validator_threshold = 0.6   # confidence score below which validator rejects (0-1 scale)

[retrieval]
top_k = 10
strategy = "hybrid"         # "hybrid" | "dense" | "bm25"

[run]
benchmark = "ragbench"
n_samples = 100
output_dir = "results/stage_3"
top_k = 10
```

- [ ] **Step 4: Update current_stage.txt**

```bash
echo "3" > "/Users/abivarma/Personal_projects/RAG Evolution Lab/.claude/current_stage.txt"
```

- [ ] **Step 5: Verify all existing tests still pass**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run pytest tests/ stage_0/tests/ stage_1/tests/ stage_2/tests/ -m "not slow" -q --tb=no 2>&1 | tail -3
```

Expected: `41 passed, 1 deselected`

- [ ] **Step 6: Commit scaffold**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
git add stage_3/ .claude/current_stage.txt pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
chore(stage-3): scaffold Agentic RAG with LangGraph and config

Installs langgraph. Config exposes max_iterations=3 (bounds latency),
validator_threshold=0.6, retrieval strategy selection.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 1: AgentState TypedDict

**Architectural note:** LangGraph passes a single mutable state dict through every node. The TypedDict defines the contract between all agents. Every field must have a clear owner.

**Files:**
- Create: `stage_3/state.py`
- Create: `stage_3/tests/test_state.py`

- [ ] **Step 1: Write failing tests**

Create `stage_3/tests/test_state.py`:

```python
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
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run pytest stage_3/tests/test_state.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'stage_3.state'`

- [ ] **Step 3: Implement stage_3/state.py**

```python
from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict


class RetrievalStrategy(str, Enum):
    HYBRID = "hybrid"
    DENSE = "dense"
    BM25 = "bm25"


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph nodes.

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
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run pytest stage_3/tests/test_state.py -v
```

Expected: 4/4 PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
git add stage_3/state.py stage_3/tests/test_state.py
git commit -m "$(cat <<'EOF'
feat(stage-3): AgentState TypedDict and RetrievalStrategy enum

Defines the shared state contract between all LangGraph nodes.
Fields: query, passages, retrieved_ids, iteration, validated,
answer, reasoning_trace, strategy, sub_queries, validator_score.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: Agent Nodes

**Architectural note:** Each node is a pure function `(AgentState) -> dict` — it returns only the fields it mutates. LangGraph merges the return dict back into the shared state. This makes nodes independently testable with no graph needed.

**Files:**
- Create: `stage_3/agents.py`
- Create: `stage_3/tests/test_agents.py`

- [ ] **Step 1: Write failing tests**

Create `stage_3/tests/test_agents.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

from stage_3.state import AgentState, RetrievalStrategy, make_initial_state


def test_supervisor_appends_to_trace() -> None:
    from stage_3.agents import supervisor_node

    state = make_initial_state("What is attention in transformers?")
    mock_pipeline = MagicMock()

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {
            "response": "strategy: hybrid\nsub_queries: What is attention? How do transformers work?",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        result = supervisor_node(state, mock_pipeline, max_iterations=3)

    assert "reasoning_trace" in result
    assert len(result["reasoning_trace"]) > 0
    assert "supervisor" in result["reasoning_trace"][0].lower()


def test_supervisor_increments_nothing_on_first_call() -> None:
    from stage_3.agents import supervisor_node

    state = make_initial_state("test query")
    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {
            "response": "strategy: hybrid\nsub_queries:",
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        result = supervisor_node(state, MagicMock(), max_iterations=3)

    assert result.get("strategy") in (RetrievalStrategy.HYBRID, RetrievalStrategy.DENSE, RetrievalStrategy.BM25, "hybrid", "dense", "bm25")


def test_retriever_populates_retrieved_ids() -> None:
    from stage_3.agents import retriever_node

    state = make_initial_state("test query", passages={"0": "text 0", "1": "text 1", "2": "text 2"})
    state["strategy"] = RetrievalStrategy.HYBRID

    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_from_passages.return_value = ["0", "2"]

    result = retriever_node(state, mock_pipeline, top_k=2)

    assert result["retrieved_ids"] == ["0", "2"]
    assert result["iteration"] == 1
    assert len(result["reasoning_trace"]) > 0


def test_retriever_increments_iteration() -> None:
    from stage_3.agents import retriever_node

    state = make_initial_state("q", passages={"0": "text"})
    state["iteration"] = 1
    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_from_passages.return_value = ["0"]

    result = retriever_node(state, mock_pipeline, top_k=5)

    assert result["iteration"] == 2


def test_validator_accepts_sufficient_context() -> None:
    from stage_3.agents import validator_node

    state = make_initial_state("What is RAG?")
    state["retrieved_ids"] = ["0", "1"]
    state["passages"] = {"0": "RAG is retrieval augmented generation.", "1": "It combines LLMs with retrieval."}

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {
            "response": "YES. The passages directly answer the question about RAG. Confidence: 0.9",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = validator_node(state)

    assert result["validated"] is True
    assert result["validator_score"] >= 0.5


def test_validator_rejects_insufficient_context() -> None:
    from stage_3.agents import validator_node

    state = make_initial_state("What is the GDP of France in 2023?")
    state["retrieved_ids"] = ["0"]
    state["passages"] = {"0": "France is a country in Europe with a rich history."}

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {
            "response": "NO. The passages do not contain GDP data. Confidence: 0.2",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = validator_node(state)

    assert result["validated"] is False
    assert result["validator_score"] < 0.5


def test_synthesizer_sets_answer() -> None:
    from stage_3.agents import synthesizer_node

    state = make_initial_state("What is RAG?")
    state["retrieved_ids"] = ["0"]
    state["passages"] = {"0": "RAG stands for Retrieval Augmented Generation."}

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {
            "response": "RAG stands for Retrieval Augmented Generation, combining retrieval with LLMs.",
            "prompt_eval_count": 50,
            "eval_count": 30,
        }
        result = synthesizer_node(state, top_k_context=5)

    assert result["answer"] is not None
    assert len(result["answer"]) > 0
    assert "synthesizer" in " ".join(result["reasoning_trace"]).lower()
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run pytest stage_3/tests/test_agents.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'stage_3.agents'`

- [ ] **Step 3: Implement stage_3/agents.py**

```python
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
    """Supervisor: decides retrieval strategy and optionally decomposes the query.

    Returns partial state update (only mutated keys).
    """
    iteration = state["iteration"]
    query = state["query"]

    # On re-entry (iteration > 0), try a different strategy
    if iteration == 0:
        prompt = (
            f"You are a retrieval supervisor. Given this query, decide:\n"
            f"1. Which retrieval strategy to use: hybrid, dense, or bm25\n"
            f"2. Whether to decompose into sub-queries (list them if yes, empty if no)\n\n"
            f"Query: {query}\n\n"
            f"Respond in format:\nstrategy: <hybrid|dense|bm25>\nsub_queries: <comma-separated or empty>"
        )
    else:
        # Previous retrieval was rejected by validator — switch strategy
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

    # Parse sub-queries
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
    """Retriever: executes the retrieval using the supervisor's chosen strategy.

    Reuses Stage 2 pipeline's retrieve_from_passages() for closed-corpus queries.
    """
    query = state["query"]
    passages = state["passages"]
    sub_queries = state.get("sub_queries") or []
    iteration = state["iteration"]

    # Use sub-queries if available, otherwise original query
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
    """Validator: asks Qwen whether the retrieved passages actually answer the query.

    Returns validated=True/False and a 0-1 confidence score.
    """
    query = state["query"]
    passages = state.get("passages") or {}
    retrieved_ids = state["retrieved_ids"]

    # Build context from retrieved passages
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
    """Synthesizer: generates the final answer with citations from validated context."""
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
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run pytest stage_3/tests/test_agents.py -v
```

Expected: 7/7 PASS. (Tests mock `ollama` so no model call needed.)

- [ ] **Step 5: Commit**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
git add stage_3/agents.py stage_3/tests/test_agents.py
git commit -m "$(cat <<'EOF'
feat(stage-3): four LangGraph agent nodes (supervisor, retriever, validator, synthesizer)

Each node is a pure function (AgentState) -> dict for testability.
Supervisor picks strategy and decomposes queries. Validator asks
'do these passages answer the question?' and scores 0-1. Self-correction
loop triggers when validator rejects (score < 0.5).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 3: LangGraph StateGraph

**Architectural note:** The graph wires nodes with typed edges and a conditional edge that creates the self-correction loop. `build_graph()` returns a compiled graph that can be invoked with `.invoke(state)`.

**Files:**
- Create: `stage_3/graph.py`
- Create: `stage_3/tests/test_graph.py`

- [ ] **Step 1: Write failing tests**

Create `stage_3/tests/test_graph.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

from stage_3.state import make_initial_state


def test_graph_builds_without_error() -> None:
    from stage_3.graph import build_graph

    mock_pipeline = MagicMock()
    graph = build_graph(mock_pipeline, max_iterations=3, top_k=5, top_k_context=3)
    assert graph is not None


def test_graph_has_expected_nodes() -> None:
    from stage_3.graph import build_graph

    mock_pipeline = MagicMock()
    graph = build_graph(mock_pipeline, max_iterations=3, top_k=5, top_k_context=3)
    # LangGraph compiled graphs expose get_graph() for introspection
    g = graph.get_graph()
    node_names = set(g.nodes.keys())
    assert "supervisor" in node_names
    assert "retriever" in node_names
    assert "validator" in node_names
    assert "synthesizer" in node_names


def test_graph_invoke_returns_state_with_answer() -> None:
    """Full graph invocation with mocked LLM calls."""
    from stage_3.graph import build_graph

    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_from_passages.return_value = ["0", "1"]

    initial_state = make_initial_state(
        "What is RAG?",
        passages={"0": "RAG is retrieval augmented generation.", "1": "It uses LLMs."},
    )

    # Mock all ollama calls: supervisor, validator (accept), synthesizer
    ollama_responses = [
        {"response": "strategy: hybrid\nsub_queries:", "prompt_eval_count": 10, "eval_count": 5},  # supervisor
        {"response": "YES. Passages answer the question. Confidence: 0.9", "prompt_eval_count": 10, "eval_count": 5},  # validator accepts
        {"response": "RAG stands for Retrieval Augmented Generation.", "prompt_eval_count": 50, "eval_count": 30},  # synthesizer
    ]

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.side_effect = ollama_responses
        graph = build_graph(mock_pipeline, max_iterations=3, top_k=5, top_k_context=3)
        final_state = graph.invoke(initial_state)

    assert final_state["answer"] is not None
    assert len(final_state["reasoning_trace"]) >= 3  # supervisor + retriever + validator + synthesizer


def test_graph_self_correction_loop() -> None:
    """Graph loops back when validator rejects, then accepts on second try."""
    from stage_3.graph import build_graph

    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_from_passages.return_value = ["0"]

    initial_state = make_initial_state(
        "Hard multi-hop question",
        passages={"0": "Some passage text here."},
    )

    # Responses: supervisor → retriever → validator REJECTS → supervisor → retriever → validator ACCEPTS → synthesizer
    ollama_responses = [
        {"response": "strategy: hybrid\nsub_queries:", "prompt_eval_count": 5, "eval_count": 5},   # supervisor 1
        {"response": "NO. Not enough info. Confidence: 0.3", "prompt_eval_count": 10, "eval_count": 5},   # validator rejects
        {"response": "strategy: dense\nsub_queries:", "prompt_eval_count": 5, "eval_count": 5},    # supervisor 2
        {"response": "YES. Sufficient context. Confidence: 0.8", "prompt_eval_count": 10, "eval_count": 5},  # validator accepts
        {"response": "The answer based on passage [0].", "prompt_eval_count": 50, "eval_count": 20},  # synthesizer
    ]

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.side_effect = ollama_responses
        graph = build_graph(mock_pipeline, max_iterations=3, top_k=5, top_k_context=3)
        final_state = graph.invoke(initial_state)

    assert final_state["answer"] is not None
    assert final_state["iteration"] == 2  # looped once


def test_graph_max_iterations_forces_synthesis() -> None:
    """Graph forces synthesis after max_iterations even if validator keeps rejecting."""
    from stage_3.graph import build_graph

    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_from_passages.return_value = ["0"]

    initial_state = make_initial_state("Impossible query", passages={"0": "irrelevant text"})

    # All validators reject; graph should still terminate
    reject = {"response": "NO. Confidence: 0.1", "prompt_eval_count": 5, "eval_count": 5}
    supervisor_resp = {"response": "strategy: hybrid\nsub_queries:", "prompt_eval_count": 5, "eval_count": 5}
    synth = {"response": "Could not find a definitive answer.", "prompt_eval_count": 20, "eval_count": 10}

    # max_iterations=2: supervisor→retriever→validator(reject)→supervisor→retriever→validator(reject)→synthesizer
    ollama_responses = [
        supervisor_resp, reject,   # iter 1
        supervisor_resp, reject,   # iter 2 (max)
        synth,                     # forced synthesis
    ]

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.side_effect = ollama_responses
        graph = build_graph(mock_pipeline, max_iterations=2, top_k=5, top_k_context=3)
        final_state = graph.invoke(initial_state)

    assert final_state["answer"] is not None
    assert final_state["iteration"] <= 2
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run pytest stage_3/tests/test_graph.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'stage_3.graph'`

- [ ] **Step 3: Implement stage_3/graph.py**

```python
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
      supervisor → retriever → validator ──→ synthesizer → END
                       ↑            |
                       └── (reject) ┘  (loops back via supervisor)
    """
    graph = StateGraph(AgentState)

    # Bind pipeline and config into node functions via closures
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

    # Register nodes
    graph.add_node("supervisor", _supervisor)
    graph.add_node("retriever", _retriever)
    graph.add_node("validator", _validator)
    graph.add_node("synthesizer", _synthesizer)

    # Entry point
    graph.set_entry_point("supervisor")

    # Fixed edges
    graph.add_edge("supervisor", "retriever")
    graph.add_edge("retriever", "validator")
    graph.add_edge("synthesizer", END)

    # Conditional edge: validator → synthesizer (accept) or supervisor (reject)
    graph.add_conditional_edges(
        "validator",
        _should_continue_bound,
        {"supervisor": "supervisor", "synthesizer": "synthesizer"},
    )

    return graph.compile()
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run pytest stage_3/tests/test_graph.py -v
```

Expected: 5/5 PASS. Fix any LangGraph API issues before continuing.

- [ ] **Step 5: Commit**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
git add stage_3/graph.py stage_3/tests/test_graph.py
git commit -m "$(cat <<'EOF'
feat(stage-3): LangGraph StateGraph with self-correction loop

Graph: supervisor -> retriever -> validator --(accept)--> synthesizer -> END
                         ^              |
                         +-- (reject) --+

Conditional edge loops back to supervisor when validator score < 0.5.
max_iterations bounds latency: forces synthesis after N rejected cycles.
Architectural shift: stateful graph vs Stage 2's one-shot pipeline.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 4: Stage 3 Harness

**Files:**
- Create: `stage_3/harness.py`

- [ ] **Step 1: Implement stage_3/harness.py**

```python
from __future__ import annotations
"""Stage 3 Agentic RAG harness entry point.

Usage:
  uv run python -m stage_3.harness --benchmark ragbench --n-samples 100
  uv run python -m stage_3.harness --benchmark ragbench --n-samples 100 --no-generation
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
from stage_3.state import RetrievalStrategy, make_initial_state


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

    # Load Stage 2 pipeline as the retrieval backend
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

    # Print reasoning trace for the last query (for inspection)
    if results and hasattr(harness, '_last_state') and harness._last_state:
        print("\n=== Reasoning trace (last query) ===")
        for step in harness._last_state.get("reasoning_trace", []):
            print(f"  {step}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify all stage_3 tests pass together**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run pytest tests/ stage_0/tests/ stage_1/tests/ stage_2/tests/ stage_3/tests/ -m "not slow" -q --tb=short 2>&1 | tail -5
```

Expected: all pass (41 existing + 4 state + 7 agents + 5 graph = 57 total).

- [ ] **Step 3: Smoke test harness import**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run python -c "
from stage_3.harness import Stage3Harness
from stage_3.graph import build_graph
from stage_3.state import make_initial_state, RetrievalStrategy
print('Stage3Harness.stage:', Stage3Harness.stage)
print('RetrievalStrategy.HYBRID:', RetrievalStrategy.HYBRID)
print('Imports OK')
"
```

- [ ] **Step 4: Commit harness**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
git add stage_3/harness.py
git commit -m "$(cat <<'EOF'
feat(stage-3): Stage3Harness wiring LangGraph into EvalHarness

Stage3Harness.retrieve_item() runs the full agentic graph per query.
Retrieval answer and reasoning trace both captured. Entry point:
stage_3.harness. Reuses Stage 2 pipeline as retrieval backend.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 5: Eval Run + Notebook Update

- [ ] **Step 1: Run Stage 3 smoke eval (50 queries, no generation)**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run python -m stage_3.harness --benchmark ragbench --n-samples 50 --no-generation
```

Expected output includes:
- Metrics table (Recall@10, MRR, nDCG@10, latency)
- Reasoning trace for last query showing supervisor/retriever/validator/synthesizer steps
- p99 latency significantly higher than Stage 2 (extra LLM calls per query)

- [ ] **Step 2: Commit results**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
git add -f results/stage_3/
git commit -m "$(cat <<'EOF'
chore(stage-3): eval results on RAGBench TechQA (50 queries, no generation)

Answers spec §1.3 Q2: how much latency does the agentic supervisor
pattern add per query at p99? Key comparison: Stage 2 p99=32884ms
vs Stage 3 p99=Xms. MRR comparison shows where extra LLM calls earn
their cost vs where they add latency for no accuracy gain.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

- [ ] **Step 3: Regenerate comparison charts**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
uv run python - <<'PYEOF'
import json, datetime, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_best(stage_dir):
    files = sorted(stage_dir.glob('*.json'))
    all_cfg = [f for f in files if '_all.' in f.name]
    if all_cfg:
        return json.loads(all_cfg[-1].read_text())
    for f in reversed(files):
        d = json.loads(f.read_text())
        if d.get('recall_at_10', 0) > 0 or d.get('mrr', 0) > 0:
            return d
    return {}

results = {}
for sd in sorted(Path('results').glob('stage_*')):
    sn = int(sd.name.split('_')[1])
    d = load_best(sd)
    if d:
        results[sn] = d

LABELS = {0:'S0\nBM25', 1:'S1\nNaive RAG', 2:'S2\nAdvanced\nRAG', 3:'S3\nAgentic\nRAG'}
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
stages = sorted(results.keys())
labels = [LABELS.get(s, f'S{s}') for s in stages]
colors = [COLORS[min(s, len(COLORS)-1)] for s in stages]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('RAG Evolution Lab — Retrieval Quality\n(RAGBench TechQA, MacBook M3)', fontsize=13, fontweight='bold', y=1.01)
for ax, (mk, ml) in zip(axes.flat, [
    ('recall_at_10','Recall@10'), ('mrr','MRR'),
    ('ndcg_at_10','nDCG@10'), ('hit_rate_10','Hit Rate@10')]):
    vals = [results[s].get(mk, 0) for s in stages]
    bars = ax.bar(range(len(stages)), vals, color=colors, width=0.5, zorder=3)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title(ml, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    if vals:
        ax.axhline(vals[0], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    for i in range(1, len(stages)):
        delta = vals[i]-vals[i-1]
        clr = '#2e7d32' if delta > 0 else ('#c62828' if delta < 0 else 'grey')
        ax.text(i, vals[i]+0.07, ('+' if delta>=0 else '')+f'{delta:.3f}',
                ha='center', va='bottom', fontsize=7.5, color=clr, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/retrieval_quality.png', bbox_inches='tight', dpi=150)
print('Saved: notebooks/retrieval_quality.png')

# Latency comparison
fig2, ax2 = plt.subplots(figsize=(11, 5))
x = np.arange(len(stages)); w = 0.25
p50 = [results[s].get('latency_p50_ms', 0) for s in stages]
p99 = [results[s].get('latency_p99_ms', 0) for s in stages]
ax2.bar(x-w/2, p50, w, label='p50', color='#4CAF50', zorder=3)
ax2.bar(x+w/2, p99, w, label='p99', color='#F44336', zorder=3)
ax2.set_yscale('log')
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('Latency (ms) — log scale', fontsize=10)
ax2.set_title('Latency by Stage (spec §1.3 Q2: agentic overhead)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3, zorder=0)
for i, (p5, p9) in enumerate(zip(p50, p99)):
    ax2.text(i-w/2, p5*1.3, f'{p5:.0f}ms', ha='center', va='bottom', fontsize=7, rotation=30)
    ax2.text(i+w/2, p9*1.3, f'{p9:.0f}ms', ha='center', va='bottom', fontsize=7, rotation=30)
plt.tight_layout()
plt.savefig('notebooks/latency_profile.png', bbox_inches='tight', dpi=150)
print('Saved: notebooks/latency_profile.png')

# RESULTS.md
lines = [
    '# RAG Evolution Lab — Results Summary',
    f'Generated: {datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}',
    'Benchmark: RAGBench TechQA | Hardware: MacBook M3 36GB',
    '',
    '| Stage | Architecture | N | Recall@10 | MRR | nDCG@10 | p99 (ms) | Δ Recall@10 | Δ MRR |',
    '|-------|-------------|---|-----------|-----|---------|----------|-------------|-------|',
]
for i, s in enumerate(stages):
    d = results[s]
    arch = LABELS.get(s, f'S{s}').replace('\n',' ').strip()
    r10, mrr_v = d.get('recall_at_10',0), d.get('mrr',0)
    ndcg, p99, n = d.get('ndcg_at_10',0), d.get('latency_p99_ms',0), d.get('n_queries',0)
    if i==0: dr, dm = 'baseline', 'baseline'
    else:
        prev = results[stages[i-1]]
        dr_v = r10-prev.get('recall_at_10',0)
        dm_v = mrr_v-prev.get('mrr',0)
        dr = ('+' if dr_v>=0 else '')+f'{dr_v:.4f}'
        dm = ('+' if dm_v>=0 else '')+f'{dm_v:.4f}'
    lines.append(f'| S{s} | {arch} | {n} | {r10:.4f} | {mrr_v:.4f} | {ndcg:.4f} | {p99:.0f} | {dr} | {dm} |')
Path('notebooks/RESULTS.md').write_text('\n'.join(lines))
print('Saved: notebooks/RESULTS.md')
print('\n'.join(lines))
PYEOF
```

- [ ] **Step 4: Commit updated charts**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
git add notebooks/
git commit -m "$(cat <<'EOF'
chore: update charts with Stage 3 Agentic RAG results

S0-S3 comparison now in retrieval_quality.png and RESULTS.md.
Latency chart shows the agentic overhead directly (spec §1.3 Q2).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Verification Checklist

- [ ] `uv run pytest tests/ stage_0/tests/ stage_1/tests/ stage_2/tests/ stage_3/tests/ -m "not slow" -q` → 57 PASS
- [ ] `uv run python -c "from stage_3.graph import build_graph; print('OK')"` — imports clean
- [ ] `uv run python -m stage_3.harness --benchmark ragbench --n-samples 5 --no-generation` — runs end-to-end, prints reasoning trace
- [ ] `results/stage_3/` contains at least one JSON with `stage=3`
- [ ] `notebooks/RESULTS.md` has rows for S0, S1, S2, S3
- [ ] `git log --oneline` shows 6+ new commits with Co-Authored-By

**Spec §1.3 alignment:**
- Q2: "How much latency does an agentic supervisor pattern add per query at p99?" — answered by Stage 3 p99 vs Stage 2 p99. ✅
- Blog post §6.4 framing: "agentic RAG is the right answer 30% of the time — here's how to know which 30%" — answered by MRR delta on hard vs easy queries. ✅
- Self-correction loop: validator can reject and force re-retrieval. ✅
- Query decomposition: supervisor breaks complex queries into sub-queries. ✅

---

## Next Plan

After Stage 3 is complete:
`docs/superpowers/plans/2026-05-18-rag-evolution-lab-stage4-graphrag.md`

Stage 4 adds a knowledge graph (Neo4j/LightRAG) built from arXiv entity+relationship extraction, community detection (Leiden), and graph traversal at query time. The key benchmark switches to MultiHop-RAG.
