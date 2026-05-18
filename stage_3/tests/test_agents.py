from __future__ import annotations

from unittest.mock import MagicMock, patch

from stage_3.state import AgentState, RetrievalStrategy, make_initial_state


def test_supervisor_appends_to_trace() -> None:
    from stage_3.agents import supervisor_node

    state = make_initial_state("What is attention in transformers?")
    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {
            "response": "strategy: hybrid\nsub_queries: What is attention? How do transformers work?",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        result = supervisor_node(state, MagicMock(), max_iterations=3)

    assert "reasoning_trace" in result
    assert len(result["reasoning_trace"]) > 0
    assert "supervisor" in result["reasoning_trace"][0].lower()


def test_supervisor_returns_valid_strategy() -> None:
    from stage_3.agents import supervisor_node

    state = make_initial_state("test query")
    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {
            "response": "strategy: hybrid\nsub_queries:",
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        result = supervisor_node(state, MagicMock(), max_iterations=3)

    assert result.get("strategy") in (
        RetrievalStrategy.HYBRID, RetrievalStrategy.DENSE, RetrievalStrategy.BM25,
        "hybrid", "dense", "bm25"
    )


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
