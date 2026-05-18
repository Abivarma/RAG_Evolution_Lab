from __future__ import annotations


class SemanticChunker:
    """Recursive paragraph-aware text splitter with token overlap.

    Tries separators in order; falls back to the next when a part still
    exceeds chunk_size. Unlike Stage 1's fixed word splitter, this
    respects paragraph and sentence boundaries before whitespace.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        if len(text.split()) <= self.chunk_size:
            return [text]
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        parts = [p for p in text.split(sep) if p.strip()] if sep else list(text)

        chunks: list[str] = []
        current_words: list[str] = []

        for part in parts:
            part_words = part.split()
            if len(current_words) + len(part_words) <= self.chunk_size:
                current_words.extend(part_words)
            else:
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = current_words[-self.overlap:] if self.overlap else []
                if len(part_words) > self.chunk_size:
                    sub = self._recursive_split(part, separators[1:])
                    if current_words and sub:
                        sub[0] = " ".join(current_words) + " " + sub[0]
                        current_words = []
                    chunks.extend(sub[:-1])
                    current_words = sub[-1].split() if sub else []
                else:
                    current_words.extend(part_words)

        if current_words:
            chunks.append(" ".join(current_words))

        return [c for c in chunks if c.strip()]

    def _hard_split(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, len(words), step):
            chunks.append(" ".join(words[start : start + self.chunk_size]))
            if start + self.chunk_size >= len(words):
                break
        return chunks
