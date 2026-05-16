from __future__ import annotations
"""Download arXiv metadata + abstracts from HuggingFace, save as JSON list.

Each entry: {id, title, abstract, categories, update_date}
"""

import argparse
import json
from pathlib import Path


def download(n_samples: int | None, output_path: Path) -> None:
    from datasets import load_dataset

    print("Downloading arXiv abstracts from HuggingFace (ccdv/arxiv-summarization)...")
    # ccdv/arxiv-summarization uses standard Parquet format (no loading script required)
    # Fields: 'article' (full paper text), 'abstract'
    ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = []
    for i, row in enumerate(ds):
        if n_samples and i >= n_samples:
            break
        abstract = row.get("abstract", "")
        # Use first 120 chars of abstract as a synthetic title
        title = abstract[:120].strip().replace("\n", " ") if abstract else ""
        corpus.append({
            "id": f"arxiv_{i:07d}",
            "title": title,
            "abstract": abstract,
        })

    if n_samples:
        print(f"Using {len(corpus):,} samples (subset mode)")
    else:
        print(f"Full dataset: {len(corpus):,} papers")

    output_path.write_text(json.dumps(corpus))
    size_gb = output_path.stat().st_size / 1e9
    print(f"Saved {len(corpus):,} papers to {output_path} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit to N samples for quick tests. Omit for full 1.7M.")
    parser.add_argument("--output", type=Path, default=Path("data/arxiv_abstracts.json"))
    args = parser.parse_args()
    download(args.n_samples, args.output)
