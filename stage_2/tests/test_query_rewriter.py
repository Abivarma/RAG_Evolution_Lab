from __future__ import annotations

from unittest.mock import patch
import pytest
from stage_2.query_rewriter import QueryRewriter, parse_paraphrases


def test_parse_paraphrases_numbered_list() -> None:
    raw = "1. How does transformer attention work?\n2. What is self-attention in neural nets?"
    result = parse_paraphrases(raw)
    assert len(result) == 2
    assert not result[0][0].isdigit()


def test_parse_paraphrases_strips_numbering() -> None:
    raw = "1. First paraphrase\n2. Second paraphrase\n3. Third paraphrase"
    result = parse_paraphrases(raw)
    assert all(not r[0].isdigit() for r in result)


def test_parse_paraphrases_handles_empty() -> None:
    assert parse_paraphrases("") == []


def test_parse_paraphrases_single_line() -> None:
    result = parse_paraphrases("What is retrieval augmented generation?")
    assert len(result) >= 1


def test_rewriter_original_always_first() -> None:
    rewriter = QueryRewriter.__new__(QueryRewriter)
    rewriter.model = "qwen2.5:14b"
    rewriter.n_paraphrases = 2
    rewriter.temperature = 0.3

    mock_resp = "1. How does RAG retrieval work?\n2. What is retrieval in language models?"
    with patch("stage_2.query_rewriter.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {"response": mock_resp}
        queries = rewriter.rewrite("What is RAG?")

    assert queries[0] == "What is RAG?"
    assert len(queries) >= 2


def test_rewriter_deduplicates() -> None:
    rewriter = QueryRewriter.__new__(QueryRewriter)
    rewriter.model = "qwen2.5:14b"
    rewriter.n_paraphrases = 2
    rewriter.temperature = 0.3

    mock_resp = "1. What is RAG?\n2. RAG explanation please"
    with patch("stage_2.query_rewriter.ollama") as mock_ollama:
        mock_ollama.generate.return_value = {"response": mock_resp}
        queries = rewriter.rewrite("What is RAG?")

    assert queries.count("What is RAG?") == 1
