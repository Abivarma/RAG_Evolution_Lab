# RAG Evolution Lab

## Project context
9-stage RAG architecture comparison on a MacBook M3 36GB.
See SPEC.md for the full design. We are currently working on Stage 0.

## Hardware constraints (HARD LIMITS)
- 36GB RAM total. One heavy service (Qdrant) + Qwen 14B + workspace must fit.
- 1TB SSD with warm tier on Box: /Users/abivarma/Library/CloudStorage/Box-Box/RAG_Evolution_Lab
- No GPU beyond Apple M3 Metal. Never suggest CUDA solutions.

## Stack
- Python 3.11, uv for env management (.venv/)
- Qdrant for vector store (single binary, port 6333)
- Ollama for local LLM serving (Qwen 2.5 14B Q4_K_M)
- LangGraph for agentic stages (3, 8)
- BGE-M3 embeddings, BGE-reranker-v2 cross-encoder
- RAGAS for generation metrics, custom harness for TRACe metrics

## Coding conventions
- Type hints on ALL functions. `from __future__ import annotations` at top of every file.
- All configs in `config.toml` per stage directory, never hardcoded values.
- Eval results saved to `results/stage_{N}/{timestamp}.json`.
- Every stage has identical eval harness CLI: `python -m stage_N.eval --benchmark NAME`.
- Run evals via uv: `uv run python -m stage_0.eval --benchmark ragbench`
- Index serialization: use JSON or numpy .npy formats only — never binary object serialization.

## Commit convention
- Author: Abivarma <Abivarma.Rs@ibm.com>
- Co-author every commit: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
- Commit after each logical unit of work (harness, retriever, indexer, eval run).
- Check alignment with spec goals §1.3 on every commit.

## Anti-patterns we explicitly avoid
- Never fine-tune models. Benchmarking off-the-shelf only.
- Never cloud-only solutions (Pinecone, OpenAI embeddings). Local-first always.
- Never conflate "agentic = better". Latency/cost must be measured, not assumed.
- Never use binary object serialization (e.g. Python's pickle). Use JSON or numpy .npy.

## What Claude Code should do in this project
- Lead with the architectural shift each change embodies.
- Always reference which metric the change affects.
- Suggest the minimal change before any refactor.
- Flag stage-coupling sins (stage N code importing stage M logic directly).
- On every commit: verify alignment with spec success criteria §1.3.
