from __future__ import annotations

import pytest
from stage_2.chunker import SemanticChunker


@pytest.fixture
def chunker() -> SemanticChunker:
    return SemanticChunker(chunk_size=50, overlap=10, separators=["\n\n", "\n", ". ", " "])


def test_splits_on_paragraph_boundary(chunker: SemanticChunker) -> None:
    text = "First paragraph with some words here.\n\nSecond paragraph with different words here."
    chunks = chunker.split(text)
    assert len(chunks) >= 1
    assert any("First paragraph" in c for c in chunks)
    assert any("Second paragraph" in c for c in chunks)


def test_overlap_is_applied(chunker: SemanticChunker) -> None:
    words = [f"word{i}" for i in range(120)]
    text = " ".join(words)
    chunks = chunker.split(text)
    assert len(chunks) >= 2
    last_words = set(chunks[0].split()[-10:])
    first_words = set(chunks[1].split()[:10])
    assert len(last_words & first_words) > 0


def test_short_text_returns_single_chunk(chunker: SemanticChunker) -> None:
    assert chunker.split("Short text.") == ["Short text."]


def test_empty_text_returns_empty(chunker: SemanticChunker) -> None:
    assert chunker.split("") == []


def test_chunk_size_respected(chunker: SemanticChunker) -> None:
    words = [f"word{i}" for i in range(200)]
    chunks = chunker.split(" ".join(words))
    for chunk in chunks[:-1]:
        assert len(chunk.split()) <= chunker.chunk_size + chunker.overlap + 5
