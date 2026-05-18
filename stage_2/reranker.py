from __future__ import annotations

import numpy as np
import torch


class BGEReranker:
    """BGE-reranker-v2-m3 cross-encoder. Uses transformers directly to avoid
    FlagEmbedding 1.4.0 / transformers 5.x incompatibility where
    XLMRobertaTokenizer lost prepare_for_model.

    Takes query + dict of {doc_id: text}, returns doc_ids sorted by
    cross-encoder score (highest first). Expensive but highly accurate.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "mps",
        top_k: int = 5,
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            if self.use_fp16 and self.device != "cpu":
                self._model = self._model.half()
            self._model = self._model.to(self.device)
            self._model.eval()

    @torch.no_grad()
    def _score_pairs(self, pairs: list[list[str]]) -> list[float]:
        """Score a list of [query, passage] pairs; return float scores."""
        all_scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]
            encoded = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self._model(**encoded, return_dict=True).logits
            scores = logits.view(-1).float().cpu().numpy().tolist()
            all_scores.extend(scores)
        return all_scores

    def rerank(
        self, query: str, passages: dict[str, str], top_k: int | None = None
    ) -> list[str]:
        """Score each (query, passage) pair; return doc_ids sorted by score."""
        self._load()
        k = top_k if top_k is not None else self.top_k
        doc_ids = list(passages.keys())
        pairs = [[query, text] for text in passages.values()]
        scores = self._score_pairs(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [doc_ids[i] for i in ranked[: min(k, len(doc_ids))]]
