from __future__ import annotations

import pytest


def test_chunk_text_fixed_size() -> None:
    from stage_1.embedder import chunk_text

    text = " ".join([f"word{i}" for i in range(600)])
    chunks = chunk_text(text, chunk_size=512, overlap=0)
    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        assert len(chunk.split()) <= 512


def test_chunk_text_short_text() -> None:
    from stage_1.embedder import chunk_text

    text = "Short abstract here."
    chunks = chunk_text(text, chunk_size=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty() -> None:
    from stage_1.embedder import chunk_text

    assert chunk_text("", chunk_size=512, overlap=0) == []


def test_build_prompt_contains_query() -> None:
    from stage_1.generator import build_prompt

    query = "What is RAG?"
    passages = ["RAG stands for Retrieval Augmented Generation.", "It combines retrieval with LLMs."]
    prompt = build_prompt(query, passages)
    assert "What is RAG?" in prompt
    assert "Retrieval Augmented Generation" in prompt


def test_build_prompt_passage_ordering() -> None:
    from stage_1.generator import build_prompt

    prompt = build_prompt("q", ["passage A", "passage B"])
    assert prompt.index("passage A") < prompt.index("passage B")


@pytest.mark.slow
def test_embed_and_rank_passages_orders_correctly() -> None:
    """BGEEmbedder.embed_and_rank_passages should rank the most relevant passage first."""
    from stage_1.embedder import BGEEmbedder

    embedder = BGEEmbedder(device="mps")
    passages = {
        "0": "The Eiffel Tower is located in Paris, France.",
        "1": "Machine learning is a branch of artificial intelligence.",
        "2": "Paris is the capital of France and home to many landmarks.",
    }
    ranked = embedder.embed_and_rank_passages("famous landmarks in Paris", passages, top_k=3)
    # "0" or "2" should rank above "1" for this query
    assert ranked[0] in {"0", "2"}, f"Expected Paris-related doc first, got {ranked}"
