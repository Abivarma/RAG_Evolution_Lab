from __future__ import annotations

import ollama


def build_prompt(query: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        "Use the following passages to answer the question. "
        "If the passages don't contain the answer, say so.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


class OllamaGenerator:
    """Qwen 2.5 14B generator via Ollama."""

    def __init__(
        self, model: str = "qwen2.5:14b", temperature: float = 0.1, max_tokens: int = 512
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, passages: list[str]) -> tuple[str, int, int]:
        """Return (answer, input_tokens, output_tokens)."""
        prompt = build_prompt(query, passages)
        resp = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return (
            resp["response"],
            resp.get("prompt_eval_count", 0),
            resp.get("eval_count", 0),
        )
