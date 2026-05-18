from __future__ import annotations

import re

import ollama


def parse_paraphrases(raw: str) -> list[str]:
    """Extract paraphrase strings from a numbered or plain LLM response."""
    if not raw.strip():
        return []
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    results = []
    for line in lines:
        cleaned = re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", line).strip()
        if cleaned:
            results.append(cleaned)
    return results


class QueryRewriter:
    """Rewrites a query into N paraphrases using Qwen 14B.

    Always returns the original query first, followed by up to n_paraphrases
    model-generated variants. Deduplicates case-insensitively.
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        n_paraphrases: int = 2,
        temperature: float = 0.3,
    ) -> None:
        self.model = model
        self.n_paraphrases = n_paraphrases
        self.temperature = temperature

    def rewrite(self, query: str) -> list[str]:
        """Return [original] + paraphrases, deduplicated, original always first."""
        prompt = (
            f"Rewrite the following query into {self.n_paraphrases} different paraphrases "
            f"that preserve the meaning but use different words or structure. "
            f"Output only the paraphrases, one per line, numbered.\n\nQuery: {query}"
        )
        resp = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature, "num_predict": 256},
        )
        paraphrases = parse_paraphrases(resp["response"])

        seen: set[str] = {query.lower()}
        result = [query]
        for p in paraphrases:
            if p.lower() not in seen:
                seen.add(p.lower())
                result.append(p)

        return result[: self.n_paraphrases + 1]
