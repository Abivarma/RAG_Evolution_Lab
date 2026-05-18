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
    g = graph.get_graph()
    node_names = set(g.nodes.keys())
    assert "supervisor" in node_names
    assert "retriever" in node_names
    assert "validator" in node_names
    assert "synthesizer" in node_names


def test_graph_invoke_returns_state_with_answer() -> None:
    from stage_3.graph import build_graph

    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_from_passages.return_value = ["0", "1"]

    initial_state = make_initial_state(
        "What is RAG?",
        passages={"0": "RAG is retrieval augmented generation.", "1": "It uses LLMs."},
    )

    ollama_responses = [
        {"response": "strategy: hybrid\nsub_queries:", "prompt_eval_count": 10, "eval_count": 5},
        {"response": "YES. Passages answer the question. Confidence: 0.9", "prompt_eval_count": 10, "eval_count": 5},
        {"response": "RAG stands for Retrieval Augmented Generation.", "prompt_eval_count": 50, "eval_count": 30},
    ]

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.side_effect = ollama_responses
        graph = build_graph(mock_pipeline, max_iterations=3, top_k=5, top_k_context=3)
        final_state = graph.invoke(initial_state)

    assert final_state["answer"] is not None
    assert len(final_state["reasoning_trace"]) >= 3


def test_graph_self_correction_loop() -> None:
    from stage_3.graph import build_graph

    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_from_passages.return_value = ["0"]

    initial_state = make_initial_state(
        "Hard multi-hop question",
        passages={"0": "Some passage text here."},
    )

    ollama_responses = [
        {"response": "strategy: hybrid\nsub_queries:", "prompt_eval_count": 5, "eval_count": 5},
        {"response": "NO. Not enough info. Confidence: 0.3", "prompt_eval_count": 10, "eval_count": 5},
        {"response": "strategy: dense\nsub_queries:", "prompt_eval_count": 5, "eval_count": 5},
        {"response": "YES. Sufficient context. Confidence: 0.8", "prompt_eval_count": 10, "eval_count": 5},
        {"response": "The answer based on passage [0].", "prompt_eval_count": 50, "eval_count": 20},
    ]

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.side_effect = ollama_responses
        graph = build_graph(mock_pipeline, max_iterations=3, top_k=5, top_k_context=3)
        final_state = graph.invoke(initial_state)

    assert final_state["answer"] is not None
    assert final_state["iteration"] == 2


def test_graph_max_iterations_forces_synthesis() -> None:
    from stage_3.graph import build_graph

    mock_pipeline = MagicMock()
    mock_pipeline.retrieve_from_passages.return_value = ["0"]

    initial_state = make_initial_state("Impossible query", passages={"0": "irrelevant text"})

    reject = {"response": "NO. Confidence: 0.1", "prompt_eval_count": 5, "eval_count": 5}
    supervisor_resp = {"response": "strategy: hybrid\nsub_queries:", "prompt_eval_count": 5, "eval_count": 5}
    synth = {"response": "Could not find a definitive answer.", "prompt_eval_count": 20, "eval_count": 10}

    ollama_responses = [
        supervisor_resp, reject,
        supervisor_resp, reject,
        synth,
    ]

    with patch("stage_3.agents.ollama") as mock_ollama:
        mock_ollama.generate.side_effect = ollama_responses
        graph = build_graph(mock_pipeline, max_iterations=2, top_k=5, top_k_context=3)
        final_state = graph.invoke(initial_state)

    assert final_state["answer"] is not None
    assert final_state["iteration"] <= 2
