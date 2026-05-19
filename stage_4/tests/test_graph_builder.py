from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_prepare_corpus_json_returns_url_keyed_dict() -> None:
    from stage_4.graph_builder import prepare_corpus_json

    corpus = prepare_corpus_json(n_articles=3)
    assert len(corpus) == 3
    for url, text in corpus.items():
        assert url.startswith("http"), f"Key should be URL, got: {url[:40]}"
        assert len(text) > 20, f"Text too short for url {url}: {text[:40]}"


def test_prepare_corpus_json_deduplicates() -> None:
    from stage_4.graph_builder import prepare_corpus_json

    corpus = prepare_corpus_json(n_articles=None)
    assert len(corpus) <= 609
    assert len(set(corpus.keys())) == len(corpus)


def test_prepare_corpus_json_saves_to_file(tmp_path: Path) -> None:
    from stage_4.graph_builder import prepare_corpus_json

    output = tmp_path / "corpus.json"
    corpus = prepare_corpus_json(n_articles=3, output_path=output)
    assert output.exists()
    loaded = json.loads(output.read_text())
    assert len(loaded) == len(corpus)
