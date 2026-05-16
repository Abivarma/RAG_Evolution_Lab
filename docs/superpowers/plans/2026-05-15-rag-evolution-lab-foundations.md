# RAG Evolution Lab — Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the complete RAG Evolution Lab repo — git init, connect to GitHub, set up Python 3.11 `uv` venv, write the shared eval harness skeleton, configure Claude Code hooks/subagents/slash commands, and implement Stage 0 (BM25 baseline) + Stage 1 (Naive RAG) with full benchmark runs.

**Architecture:** Each of the 9 RAG stages lives in `stage_N/` with isolated code but shares a common eval harness at `shared/eval/`. The eval harness has a fixed interface (`python -m stage_N.eval --benchmark NAME`) so every stage is directly comparable. Hooks enforce no-secrets policy and log session latency. Subagents run heavy batch jobs without blocking the main session.

**Tech Stack:** Python 3.11, uv, rank_bm25, sentence-transformers (BGE-M3), qdrant-client, ollama (Qwen 2.5 14B Q4_K_M), ragas, psutil, datasets (HuggingFace), TOML config per stage.

**Commit author:** Abivarma <Abivarma.Rs@ibm.com>  
**Co-author line (every commit):** `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`  
**Remote:** https://github.com/Abivarma/RAG_Evolution_Lab.git  
**Warm tier:** /Users/abivarma/Library/CloudStorage/Box-Box/RAG_Evolution_Lab

---

## Scope

This plan covers **Week 1 + Week 2** of the spec roadmap:
- Repo bootstrapping, CLAUDE.md, .claude/ infra
- Shared eval harness
- Stage 0: BM25 Baseline
- Stage 1: Naive RAG

Later stages (2-8) will each get their own plan file.

---

## File Structure

```
RAG Evolution Lab/
├── .claude/
│   ├── settings.json            # hooks config
│   ├── current_stage.txt        # tracks active stage
│   ├── agents/
│   │   ├── chunking-experimenter.md
│   │   ├── eval-runner.md
│   │   └── metrics-aggregator.md
│   └── commands/
│       ├── eval-stage.md
│       ├── compare-stages.md
│       ├── ingest.md
│       ├── promote-stage.md
│       └── blog-draft.md
├── CLAUDE.md                    # project context for Claude Code
├── SPEC.md                      # copy of spec
├── pyproject.toml               # uv project file
├── .python-version              # pins 3.11
├── .gitignore
├── scripts/
│   ├── verify_no_secrets.sh     # hook: PostToolUse
│   ├── log_latency.py           # hook: Stop
│   └── download_arxiv.py        # one-time dataset download
├── shared/
│   └── eval/
│       ├── __init__.py
│       ├── harness.py           # EvalHarness base class
│       ├── metrics.py           # recall@k, MRR, nDCG, hit_rate
│       ├── datasets.py          # dataset loaders (RAGBench, MultiHop, FinDER)
│       └── reporter.py          # JSON output, regression detection, print summary
├── stage_0/
│   ├── __init__.py
│   ├── config.toml
│   ├── retriever.py             # BM25Retriever (JSON-serialized index)
│   ├── indexer.py               # build + persist BM25 index
│   ├── eval.py                  # __main__ entry
│   └── tests/
│       └── test_retriever.py
├── stage_1/
│   ├── __init__.py
│   ├── config.toml
│   ├── embedder.py              # BGE-M3 wrapper + chunk_text
│   ├── indexer.py               # Qdrant collection builder
│   ├── retriever.py             # QdrantRetriever
│   ├── generator.py             # Ollama Qwen wrapper
│   ├── pipeline.py              # load_pipeline() factory
│   ├── eval.py                  # __main__ entry
│   └── tests/
│       └── test_pipeline.py
├── data/
│   └── .gitkeep
└── results/
    └── .gitkeep
```

---

## Task 0: Repo Bootstrap

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `CLAUDE.md`

- [ ] **Step 1: Initialize git and connect to remote**

```bash
cd "/Users/abivarma/Personal_projects/RAG Evolution Lab"
git init
git remote add origin https://github.com/Abivarma/RAG_Evolution_Lab.git
git config user.name "Abivarma"
git config user.email "Abivarma.Rs@ibm.com"
```

- [ ] **Step 2: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
*.pyc

# uv
.uv/

# Data (large files — gitignored, managed via tier strategy)
data/
!data/.gitkeep
results/*/
!results/.gitkeep

# Indexes
*.bin
qdrant_storage/

# Secrets / env
.env
*.key
secrets.toml

# macOS
.DS_Store

# Editor
.idea/
.vscode/
```

- [ ] **Step 3: Create .python-version**

Content:
```
3.11
```

- [ ] **Step 4: Initialize uv project**

```bash
pip install uv --quiet
uv init --python 3.11 --no-workspace
```

Expected: `pyproject.toml` created.

- [ ] **Step 5: Edit pyproject.toml to add all dependencies**

Replace the auto-generated `[project]` section in `pyproject.toml`:

```toml
[project]
name = "rag-evolution-lab"
version = "0.1.0"
description = "9-stage RAG architecture benchmark on MacBook M3 36GB"
requires-python = ">=3.11"
dependencies = [
    "rank-bm25>=0.2",
    "qdrant-client>=1.13",
    "sentence-transformers>=3.5",
    "FlagEmbedding>=1.4",
    "langgraph>=0.3",
    "langchain>=0.3",
    "ragas>=0.2",
    "ollama>=0.4",
    "datasets>=2.19",
    "psutil>=5.9",
    "numpy>=1.26",
    "scipy>=1.12",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

- [ ] **Step 6: Create the venv and install deps**

```bash
uv sync
```

Expected: `.venv/` created with all packages installed.

- [ ] **Step 7: Create CLAUDE.md**

```markdown
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
- Never use pickle for index serialization — use JSON or numpy .npy formats.

## Commit convention
- Author: Abivarma <Abivarma.Rs@ibm.com>
- Co-author every commit: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
- Commit after each logical unit of work (harness, retriever, indexer, eval run).
- Check alignment with spec goals §1.3 on every commit.

## Anti-patterns we explicitly avoid
- Never fine-tune models. Benchmarking off-the-shelf only.
- Never cloud-only solutions (Pinecone, OpenAI embeddings). Local-first always.
- Never conflate "agentic = better". Latency/cost must be measured, not assumed.
- Never use pickle for serialization. Use JSON or numpy .npy.

## What Claude Code should do in this project
- Lead with the architectural shift each change embodies.
- Always reference which metric the change affects.
- Suggest the minimal change before any refactor.
- Flag stage-coupling sins (stage N code importing stage M logic directly).
- On every commit: verify alignment with spec success criteria §1.3.
```

- [ ] **Step 8: Create data/ and results/ directories with .gitkeep**

```bash
mkdir -p data results
touch data/.gitkeep results/.gitkeep
```

- [ ] **Step 9: Copy spec into repo**

```bash
cp "/Users/abivarma/Personal_projects/RAG Evolution Lab/rag-evolution-lab-spec.md" \
   "/Users/abivarma/Personal_projects/RAG Evolution Lab/SPEC.md"
```

- [ ] **Step 10: Initial commit and push**

```bash
git add .gitignore pyproject.toml .python-version CLAUDE.md SPEC.md data/.gitkeep results/.gitkeep uv.lock
git commit -m "$(cat <<'EOF'
chore: bootstrap RAG Evolution Lab repo

Initialize 9-stage RAG architecture comparison project with uv env,
project config, and CLAUDE.md project context file.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git branch -M main
git push -u origin main
```

---

## Task 1: .claude/ Infrastructure (Hooks, Commands, Subagents)

**Files:**
- Create: `.claude/settings.json`
- Create: `.claude/current_stage.txt`
- Create: `.claude/commands/eval-stage.md`
- Create: `.claude/commands/compare-stages.md`
- Create: `.claude/commands/ingest.md`
- Create: `.claude/commands/promote-stage.md`
- Create: `.claude/commands/blog-draft.md`
- Create: `.claude/agents/chunking-experimenter.md`
- Create: `.claude/agents/eval-runner.md`
- Create: `.claude/agents/metrics-aggregator.md`
- Create: `scripts/verify_no_secrets.sh`
- Create: `scripts/log_latency.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p .claude/commands .claude/agents scripts docs/blog
```

- [ ] **Step 2: Create .claude/settings.json**

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash scripts/verify_no_secrets.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run python scripts/log_latency.py"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 3: Create .claude/current_stage.txt**

Content:
```
0
```

- [ ] **Step 4: Create scripts/verify_no_secrets.sh**

```bash
#!/usr/bin/env bash
# Scan git-modified files for common secret patterns.
# Runs after every Edit/Write via PostToolUse hook.

PATTERNS=(
  "sk-[a-zA-Z0-9]{20,}"
  "ANTHROPIC_API_KEY\s*="
  "ghp_[a-zA-Z0-9]{36}"
  "api_key\s*=\s*['\"][^'\"]{8,}"
)

EXIT=0
for f in $(git diff --name-only 2>/dev/null); do
  [ -f "$f" ] || continue
  for pat in "${PATTERNS[@]}"; do
    if grep -qE "$pat" "$f" 2>/dev/null; then
      echo "HOOK WARNING: possible secret in $f (pattern: $pat)" >&2
      EXIT=1
    fi
  done
done
exit $EXIT
```

```bash
chmod +x scripts/verify_no_secrets.sh
```

- [ ] **Step 5: Create scripts/log_latency.py**

```python
from __future__ import annotations
"""Append a session-end entry to results/session_log.jsonl on Stop hook."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")
log_path = Path("results/session_log.jsonl")
log_path.parent.mkdir(parents=True, exist_ok=True)

entry = {
    "session_id": session_id,
    "stopped_at": datetime.now(timezone.utc).isoformat(),
}
with log_path.open("a") as fh:
    fh.write(json.dumps(entry) + "\n")
```

- [ ] **Step 6: Create .claude/commands/eval-stage.md**

```markdown
Run the eval harness for stage $ARGUMENTS against the benchmark (default: ragbench).

Steps:
1. Parse $ARGUMENTS as: <stage_number> [--benchmark <name>]
   Default benchmark = ragbench
2. Verify Qdrant is running if stage >= 1: `curl -s http://localhost:6333/healthz`
   If not: `qdrant &` or `docker run -d -p 6333:6333 qdrant/qdrant`
3. Verify Ollama is running if stage >= 1: `ollama list`
4. Run: `uv run python -m stage_<N>.eval --benchmark <name> --output results/stage_<N>/$(date +%Y%m%d_%H%M%S).json`
5. Parse results JSON and print summary table:
   - Recall@5 / Recall@10 / Recall@20
   - MRR, nDCG@10
   - Faithfulness, Answer Relevance (if applicable)
   - p50 / p95 / p99 latency (ms)
   - Total tokens, estimated cost
6. Compare to the previous run JSON in results/stage_<N>/. Flag regressions >5% with ⚠.
```

- [ ] **Step 7: Create .claude/commands/compare-stages.md**

```markdown
Compare the most recent eval results from two stages.

Usage: /compare-stages <N> <M>
Parse $ARGUMENTS as: <stage_N> <stage_M>

Steps:
1. Load the most recent JSON from results/stage_<N>/ and results/stage_<M>/
2. Print a side-by-side table for every metric:
   - ✅ where stage M beats stage N
   - ❌ where stage M regresses vs stage N
   - ➖ where gap is within noise (<2% relative)
3. End with one paragraph: "Stage M buys [what improved] at the cost of [what got worse].
   Recall@10 moved from X to Y (+Z%), p99 latency from Xms to Yms (+Z%)."
```

- [ ] **Step 8: Create .claude/commands/ingest.md**

```markdown
Run corpus ingestion for the current active stage.

Usage: /ingest <corpus>
Valid corpus values: arxiv, sec

Steps:
1. Read .claude/current_stage.txt → N
2. If corpus=arxiv and stage in [0,1,2,3,4,7,8]:
   Run: `uv run python -m stage_<N>.indexer --corpus data/arxiv_abstracts.json`
3. If corpus=sec and stage in [5,6,7]:
   Run: `uv run python -m stage_<N>.indexer --corpus data/sec_10k/`
4. Report index size and time taken.
```

- [ ] **Step 9: Create .claude/commands/promote-stage.md**

```markdown
Archive the completed stage's data to warm tier storage and advance to the next stage.

Usage: /promote-stage

Steps:
1. Read .claude/current_stage.txt → N
2. Ask user to confirm before proceeding (destructive to hot-tier data).
3. WARM_TIER=/Users/abivarma/Library/CloudStorage/Box-Box/RAG_Evolution_Lab
4. Archive: `tar -czf $WARM_TIER/stage_${N}_$(date +%Y%m%d).tar.gz stage_${N}/data/ 2>/dev/null`
5. Verify: `tar -tzf $WARM_TIER/stage_${N}_$(date +%Y%m%d).tar.gz | wc -l`
6. Remove hot-tier data (NOT code): `rm -rf stage_${N}/data/`
7. Print disk freed: `df -h .`
8. Update .claude/current_stage.txt to $((N+1))
```

- [ ] **Step 10: Create .claude/commands/blog-draft.md**

```markdown
Generate a first draft of the blog post for stage N.

Usage: /blog-draft <N>

Steps:
1. Load the most recent eval JSON from results/stage_<N>/
2. Load the blog outline from SPEC.md section 6.<N+1>
3. Generate a draft with:
   - Hook (use exact LinkedIn hook from spec, substituting real numbers)
   - Problem section
   - Architecture diagram (ASCII from SPEC.md)
   - Key numbers table (real benchmark results)
   - Trade-off analysis vs previous stage
   - Where this architecture wins / loses
   - Teaser for next post
4. Save to docs/blog/post_<N+1>_draft.md
5. Print word count and flag any [NUMBER] placeholders still needing real values.
```

- [ ] **Step 11: Create .claude/agents/chunking-experimenter.md**

```markdown
---
name: chunking-experimenter
description: Runs chunk-size sweep experiments on a corpus subset to find optimal chunk size for Recall@10. Use when starting a new stage that involves chunking decisions.
---

You are a focused experimenter. Your only job:
1. Run `uv run python scripts/chunking_sweep.py --stage $STAGE --sizes 256,512,1024,2048 --sample 1000`
2. Parse the output JSON (one row per chunk size: Recall@10, latency_p95)
3. Report as markdown table: Chunk Size | Recall@10 | p95 Latency (ms)
4. Identify the elbow point (best Recall@10 per ms of added latency)
5. Return one recommendation: "Use chunk size X — Recall@10=Y with p95=Zms"

Do NOT refactor any code. Do NOT modify any stage files. Read-only except writing results JSON.
```

- [ ] **Step 12: Create .claude/agents/eval-runner.md**

```markdown
---
name: eval-runner
description: Runs long eval batches without blocking the main session. Use when running 1000+ query evaluations.
---

You are a background eval runner. Your only job:
1. Accept: stage number, benchmark name, sample size, output path
2. Run: `uv run python -m stage_<N>.eval --benchmark <name> --n-samples <size> --output <path>`
3. Monitor progress (print % complete every 100 queries)
4. On completion, print the full summary table
5. Flag any metric more than 5% below the previous run

Do NOT change code. Do NOT start services. Report errors verbatim and stop.
```

- [ ] **Step 13: Create .claude/agents/metrics-aggregator.md**

```markdown
---
name: metrics-aggregator
description: Walks results/ and produces the cross-stage comparison table and decision matrix. Use before writing the synthesis blog post.
---

You are a metrics aggregator. Your only job:
1. Walk results/stage_*/ and load the most recent JSON per stage
2. Build a cross-stage table:
   Recall@10 | Faithfulness | p99_latency_ms | cost_per_query_usd | Explainability | MultiHop_acc
3. Normalize each numeric column 0-1 (min=0, max=1 within the column)
4. Output: results/decision_matrix.csv and print as markdown table
5. For each stage pair (N, N+1): "Stage N+1 buys X at the cost of Y"
6. Save synthesis to results/synthesis_report.md

Do NOT run evals. Do NOT modify stage code. Read results/ only.
```

- [ ] **Step 14: Commit .claude/ infrastructure**

```bash
git add .claude/ scripts/ docs/
git commit -m "$(cat <<'EOF'
feat: add Claude Code infrastructure (.claude/, hooks, slash commands, subagents)

PostToolUse hook scans for secrets after every Edit/Write.
Stop hook logs session end time for latency profiling.
5 slash commands: eval-stage, compare-stages, ingest, promote-stage, blog-draft.
3 subagent definitions: chunking-experimenter, eval-runner, metrics-aggregator.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: Shared Eval Harness

**Files:**
- Create: `shared/__init__.py`
- Create: `shared/eval/__init__.py`
- Create: `shared/eval/metrics.py`
- Create: `shared/eval/harness.py`
- Create: `shared/eval/datasets.py`
- Create: `shared/eval/reporter.py`
- Test: `tests/shared/test_metrics.py`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p shared/eval tests/shared
touch shared/__init__.py shared/eval/__init__.py tests/__init__.py tests/shared/__init__.py
```

- [ ] **Step 2: Write failing tests for metrics.py**

Create `tests/shared/test_metrics.py`:

```python
from __future__ import annotations

import pytest
from shared.eval.metrics import recall_at_k, mrr, ndcg_at_k, hit_rate


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b", "c"}, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_zero():
    assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == pytest.approx(0.0)


def test_recall_at_k_truncates_to_k():
    # Only top-2 retrieved: [a, b]. Relevant = {a,b,c,d}. Hits=2, |relevant|=4
    assert recall_at_k(["a", "b", "c", "d"], {"a", "b", "c", "d"}, k=2) == pytest.approx(2 / 4)


def test_mrr_first_hit():
    assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_hit():
    assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)


def test_mrr_no_hit():
    assert mrr(["x", "y", "z"], {"a"}) == pytest.approx(0.0)


def test_ndcg_perfect():
    retrieved = ["a", "b", "c"]
    relevance = {"a": 2, "b": 1, "c": 1}
    assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)


def test_hit_rate_found():
    assert hit_rate(["x", "a", "y"], {"a"}, k=3) is True


def test_hit_rate_not_found():
    assert hit_rate(["x", "y", "z"], {"a"}, k=3) is False
```

- [ ] **Step 3: Run tests to verify failure**

```bash
uv run pytest tests/shared/test_metrics.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'shared'`

- [ ] **Step 4: Implement shared/eval/metrics.py**

```python
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit. 0 if no hit."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalized DCG at k. relevance maps doc_id → integer grade (0, 1, or 2)."""

    def dcg(ranked: list[str], grades: dict[str, int], cutoff: int) -> float:
        return sum(
            (2 ** grades.get(doc_id, 0) - 1) / math.log2(i + 2)
            for i, doc_id in enumerate(ranked[:cutoff])
        )

    actual = dcg(retrieved, relevance, k)
    ideal = dcg(sorted(relevance, key=relevance.get, reverse=True), relevance, k)
    return actual / ideal if ideal > 0 else 0.0


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> bool:
    """True if any relevant doc appears in top-k."""
    return any(doc_id in relevant for doc_id in retrieved[:k])


def compute_retrieval_metrics(
    retrieved: list[str],
    relevant: set[str],
    relevance_grades: dict[str, int] | None = None,
) -> dict[str, float]:
    """Full retrieval metrics bundle for one query."""
    grades = relevance_grades or {doc_id: 1 for doc_id in relevant}
    return {
        "recall_at_5": recall_at_k(retrieved, relevant, k=5),
        "recall_at_10": recall_at_k(retrieved, relevant, k=10),
        "recall_at_20": recall_at_k(retrieved, relevant, k=20),
        "mrr": mrr(retrieved, relevant),
        "ndcg_at_10": ndcg_at_k(retrieved, grades, k=10),
        "hit_rate_10": float(hit_rate(retrieved, relevant, k=10)),
    }
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
uv run pytest tests/shared/test_metrics.py -v
```

Expected: 10/10 PASS.

- [ ] **Step 6: Implement shared/eval/harness.py**

```python
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import psutil

from shared.eval.metrics import compute_retrieval_metrics


@dataclass
class QueryResult:
    query_id: str
    query: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    answer: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    ram_peak_mb: float = 0.0
    relevance_grades: dict[str, int] = field(default_factory=dict)


@dataclass
class StageMetrics:
    stage: int
    benchmark: str
    n_queries: int
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float
    hit_rate_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    total_input_tokens: int
    total_output_tokens: int
    peak_ram_mb: float
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class EvalHarness(ABC):
    """Base class every stage's eval.py inherits from."""

    stage: int

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Return a ranked list of doc_ids for the query."""
        ...

    @abstractmethod
    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        """Return (answer_text, input_tokens, output_tokens)."""
        ...

    def run(
        self,
        queries: list[dict[str, Any]],
        top_k: int = 10,
        include_generation: bool = False,
    ) -> list[QueryResult]:
        results: list[QueryResult] = []
        proc = psutil.Process()

        for item in queries:
            t0 = time.perf_counter()
            ram_before = proc.memory_info().rss / 1024 / 1024

            retrieved = self.retrieve(item["query"], top_k=top_k)

            answer = None
            in_toks = out_toks = 0
            if include_generation:
                answer, in_toks, out_toks = self.generate(item["query"], retrieved)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            ram_after = proc.memory_info().rss / 1024 / 1024

            results.append(QueryResult(
                query_id=str(item.get("id", item["query"][:16])),
                query=item["query"],
                retrieved_ids=retrieved,
                relevant_ids=set(item.get("relevant_ids", [])),
                answer=answer,
                latency_ms=elapsed_ms,
                input_tokens=in_toks,
                output_tokens=out_toks,
                ram_peak_mb=max(ram_before, ram_after),
                relevance_grades=item.get("relevance_grades", {}),
            ))

        return results
```

- [ ] **Step 7: Implement shared/eval/reporter.py**

```python
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from shared.eval.harness import QueryResult, StageMetrics
from shared.eval.metrics import compute_retrieval_metrics


def aggregate(results: list[QueryResult], stage: int, benchmark: str) -> StageMetrics:
    """Aggregate per-query results into StageMetrics."""
    all_metrics: list[dict[str, float]] = []
    latencies: list[float] = []
    total_in = total_out = 0
    peak_ram = 0.0

    for r in results:
        m = compute_retrieval_metrics(r.retrieved_ids, r.relevant_ids, r.relevance_grades or None)
        all_metrics.append(m)
        latencies.append(r.latency_ms)
        total_in += r.input_tokens
        total_out += r.output_tokens
        peak_ram = max(peak_ram, r.ram_peak_mb)

    def mean(key: str) -> float:
        return statistics.mean(row[key] for row in all_metrics)

    latencies.sort()
    n = len(latencies)

    def pct(p: float) -> float:
        return latencies[min(int(p * n), n - 1)]

    return StageMetrics(
        stage=stage,
        benchmark=benchmark,
        n_queries=n,
        recall_at_5=mean("recall_at_5"),
        recall_at_10=mean("recall_at_10"),
        recall_at_20=mean("recall_at_20"),
        mrr=mean("mrr"),
        ndcg_at_10=mean("ndcg_at_10"),
        hit_rate_10=mean("hit_rate_10"),
        latency_p50_ms=pct(0.50),
        latency_p95_ms=pct(0.95),
        latency_p99_ms=pct(0.99),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        peak_ram_mb=peak_ram,
    )


def save(metrics: StageMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {**metrics.__dict__, "saved_at": datetime.now(timezone.utc).isoformat()}
    output_path.write_text(json.dumps(data, indent=2))


def print_summary(metrics: StageMetrics) -> None:
    print(f"\n{'='*60}")
    print(f"Stage {metrics.stage} | Benchmark: {metrics.benchmark} | N={metrics.n_queries}")
    print(f"{'='*60}")
    print(f"Recall@5:    {metrics.recall_at_5:.4f}")
    print(f"Recall@10:   {metrics.recall_at_10:.4f}  ← primary metric")
    print(f"Recall@20:   {metrics.recall_at_20:.4f}")
    print(f"MRR:         {metrics.mrr:.4f}")
    print(f"nDCG@10:     {metrics.ndcg_at_10:.4f}")
    print(f"Hit Rate@10: {metrics.hit_rate_10:.4f}")
    print(f"---")
    print(f"p50 latency: {metrics.latency_p50_ms:.1f}ms")
    print(f"p95 latency: {metrics.latency_p95_ms:.1f}ms")
    print(f"p99 latency: {metrics.latency_p99_ms:.1f}ms")
    print(f"---")
    print(f"Input tokens:  {metrics.total_input_tokens:,}")
    print(f"Output tokens: {metrics.total_output_tokens:,}")
    print(f"Peak RAM:      {metrics.peak_ram_mb:.1f} MB")
    print(f"{'='*60}\n")


def check_regression(new: StageMetrics, prev_path: Path, threshold: float = 0.05) -> None:
    """Warn on metrics that regressed >threshold vs the previous run."""
    if not prev_path.exists():
        return
    prev = json.loads(prev_path.read_text())
    for key in ["recall_at_10", "mrr", "ndcg_at_10"]:
        old_val = prev.get(key, 0.0)
        new_val = getattr(new, key, 0.0)
        if old_val > 0 and (old_val - new_val) / old_val > threshold:
            pct = (old_val - new_val) / old_val * 100
            print(f"⚠ REGRESSION: {key} dropped {old_val:.4f} → {new_val:.4f} ({pct:.1f}%)")
```

- [ ] **Step 8: Implement shared/eval/datasets.py**

```python
from __future__ import annotations

from typing import Any


def load_ragbench(split: str = "test", n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load RAGBench queries. Returns list of {id, query, relevant_ids, answer}."""
    from datasets import load_dataset

    ds = load_dataset("rungalileo/ragbench", "techqa", split=split, trust_remote_code=True)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    return [
        {
            "id": f"ragbench_{i}",
            "query": row["question"],
            "relevant_ids": row.get("documents", []),
            "answer": row.get("response", ""),
        }
        for i, row in enumerate(ds)
    ]


def load_multihop_rag(split: str = "test", n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load MultiHop-RAG queries."""
    from datasets import load_dataset

    ds = load_dataset("yixuantt/MultiHopRAG", split=split, trust_remote_code=True)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    return [
        {
            "id": f"multihop_{i}",
            "query": row.get("query", row.get("question", "")),
            "relevant_ids": row.get("evidence_list", []),
            "answer": row.get("answer", ""),
        }
        for i, row in enumerate(ds)
    ]


def load_finder(split: str = "test", n_samples: int | None = None) -> list[dict[str, Any]]:
    """Load FinDER (SEC 10-K QA) queries."""
    from datasets import load_dataset

    ds = load_dataset("Linq-AI-Research/FinDER", split=split, trust_remote_code=True)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    return [
        {
            "id": f"finder_{i}",
            "query": row.get("question", ""),
            "relevant_ids": row.get("context_ids", []),
            "answer": row.get("answer", ""),
        }
        for i, row in enumerate(ds)
    ]


BENCHMARK_LOADERS = {
    "ragbench": load_ragbench,
    "multihop": load_multihop_rag,
    "finder": load_finder,
}
```

- [ ] **Step 9: Commit shared eval harness**

```bash
git add shared/ tests/
git commit -m "$(cat <<'EOF'
feat: add shared eval harness with Recall@K, MRR, nDCG, reporter

EvalHarness ABC defines retrieve/generate interface every stage
implements. Reporter aggregates per-query results into StageMetrics
and detects regressions >5%. Metrics: recall@5/10/20, MRR, nDCG@10.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 3: LLM Setup — Remove old models, pull Qwen 2.5 14B

- [ ] **Step 1: Remove existing Ollama models to free space**

```bash
ollama rm mistral:7b
ollama rm qwen2.5:7b
ollama rm granite4.1:8b
ollama rm llama3.1:8b
df -h ~ | head -3
```

- [ ] **Step 2: Pull Qwen 2.5 14B Q4_K_M**

```bash
ollama pull qwen2.5:14b
```

Verify:
```bash
ollama list | grep "qwen2.5:14b"
```

Expected: `qwen2.5:14b` listed with ~9GB size.

- [ ] **Step 3: Smoke test Qwen 14B**

```bash
ollama run qwen2.5:14b "Respond with exactly: OK" --nowordwrap
```

Expected: `OK`

- [ ] **Step 4: Install Qdrant binary**

```bash
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-aarch64-apple-darwin.tar.gz \
  | tar -xz -C /usr/local/bin/
chmod +x /usr/local/bin/qdrant
qdrant --version
```

If `/usr/local/bin/` requires sudo: `sudo mv qdrant /usr/local/bin/`

---

## Task 4: Stage 0 — BM25 Baseline

**Architectural shift:** Pre-AI keyword retrieval. No LLM, no embeddings.  
**Metric affected:** Establishes Recall@10 anchor — every later stage is judged against this.

**Files:**
- Create: `stage_0/__init__.py`
- Create: `stage_0/config.toml`
- Create: `stage_0/retriever.py`
- Create: `stage_0/indexer.py`
- Create: `stage_0/eval.py`
- Create: `stage_0/tests/test_retriever.py`
- Create: `scripts/download_arxiv.py`

- [ ] **Step 1: Create stage_0 structure**

```bash
mkdir -p stage_0/tests
touch stage_0/__init__.py stage_0/tests/__init__.py
```

- [ ] **Step 2: Create stage_0/config.toml**

```toml
[bm25]
k1 = 1.5
b = 0.75
top_k = 10

[data]
abstracts_path = "data/arxiv_abstracts.json"
index_path = "data/stage_0_bm25_index.json"

[eval]
benchmark = "ragbench"
n_samples = 1000
output_dir = "results/stage_0"
```

- [ ] **Step 3: Write failing tests for BM25Retriever**

Create `stage_0/tests/test_retriever.py`:

```python
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from stage_0.retriever import BM25Retriever

SMALL_CORPUS = [
    {"id": "doc1", "title": "neural network transformers attention", "abstract": "We study self-attention mechanisms in transformers."},
    {"id": "doc2", "title": "retrieval augmented generation RAG", "abstract": "RAG combines retrieval with LLM generation."},
    {"id": "doc3", "title": "BM25 keyword search baseline", "abstract": "BM25 is a classical keyword retrieval algorithm."},
]


@pytest.fixture
def retriever() -> BM25Retriever:
    r = BM25Retriever(k1=1.5, b=0.75)
    r.index(SMALL_CORPUS)
    return r


def test_retriever_returns_ids(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("transformers attention", top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(r, str) for r in results)


def test_retriever_ranks_by_relevance(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("BM25 keyword retrieval", top_k=3)
    assert results[0] == "doc3", f"Expected doc3 first, got {results}"


def test_retriever_handles_unknown_terms(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("xyzzy frobnicator", top_k=3)
    assert isinstance(results, list)


def test_retriever_save_load(retriever: BM25Retriever, tmp_path: Path) -> None:
    index_path = tmp_path / "bm25_index.json"
    retriever.save(index_path)
    loaded = BM25Retriever.load(index_path)
    assert retriever.retrieve("RAG generation", top_k=2) == loaded.retrieve("RAG generation", top_k=2)


def test_retriever_top_k_respected(retriever: BM25Retriever) -> None:
    results = retriever.retrieve("neural network", top_k=1)
    assert len(results) == 1
```

- [ ] **Step 4: Run tests to verify failure**

```bash
uv run pytest stage_0/tests/test_retriever.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError`

- [ ] **Step 5: Implement stage_0/retriever.py**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 retrieval over a corpus of documents.

    Serializes to/from JSON (not pickle) so index files are human-inspectable.
    Each document dict must have 'id' and 'abstract' keys; 'title' is optional.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def index(self, corpus: list[dict[str, Any]]) -> None:
        """Build BM25 index from corpus documents."""
        self._doc_ids = [doc["id"] for doc in corpus]
        self._tokenized_corpus = [
            self._tokenize(doc.get("title", "") + " " + doc.get("abstract", ""))
            for doc in corpus
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Return top_k doc_ids ranked by BM25 score."""
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve().")
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._doc_ids[i] for i in ranked[:top_k]]

    def save(self, path: Path) -> None:
        """Persist index state to JSON. Rebuilds BM25 on load (fast, avoids pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self._doc_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }
        path.write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        """Load from JSON and rebuild the BM25 index in memory."""
        state = json.loads(Path(path).read_text())
        obj = cls(k1=state["k1"], b=state["b"])
        obj._doc_ids = state["doc_ids"]
        obj._tokenized_corpus = state["tokenized_corpus"]
        obj._bm25 = BM25Okapi(obj._tokenized_corpus, k1=obj.k1, b=obj.b)
        return obj
```

- [ ] **Step 6: Run tests — expect all pass**

```bash
uv run pytest stage_0/tests/test_retriever.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 7: Implement stage_0/indexer.py**

```python
from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

from stage_0.retriever import BM25Retriever


def build_index(abstracts_path: Path, index_path: Path, k1: float, b: float) -> BM25Retriever:
    print(f"Loading abstracts from {abstracts_path}...")
    corpus = json.loads(abstracts_path.read_text())
    print(f"Loaded {len(corpus):,} documents.")

    retriever = BM25Retriever(k1=k1, b=b)
    t0 = time.perf_counter()
    retriever.index(corpus)
    elapsed = time.perf_counter() - t0
    print(f"Index built in {elapsed:.1f}s")

    retriever.save(index_path)
    size_mb = index_path.stat().st_size / 1e6
    print(f"Index saved to {index_path} ({size_mb:.1f} MB)")
    return retriever


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open("stage_0/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    build_index(
        abstracts_path=args.corpus or Path(cfg["data"]["abstracts_path"]),
        index_path=args.output or Path(cfg["data"]["index_path"]),
        k1=cfg["bm25"]["k1"],
        b=cfg["bm25"]["b"],
    )
```

- [ ] **Step 8: Create scripts/download_arxiv.py**

```python
from __future__ import annotations
"""Download arXiv metadata + abstracts from HuggingFace, save as JSON list.

Each entry: {id, title, abstract, categories, update_date}
"""

import argparse
import json
from pathlib import Path


def download(n_samples: int | None, output_path: Path) -> None:
    from datasets import load_dataset

    print("Downloading arXiv dataset from HuggingFace...")
    ds = load_dataset("arxiv-community/arxiv_dataset", split="train", trust_remote_code=True)

    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
        print(f"Using {n_samples:,} samples (subset mode)")
    else:
        print(f"Full dataset: {len(ds):,} papers")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = [
        {
            "id": row["id"],
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "categories": row.get("categories", ""),
            "update_date": row.get("update_date", ""),
        }
        for row in ds
    ]
    output_path.write_text(json.dumps(corpus))
    print(f"Saved {len(corpus):,} papers to {output_path} ({output_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit to N samples for quick tests. Omit for full 1.7M.")
    parser.add_argument("--output", type=Path, default=Path("data/arxiv_abstracts.json"))
    args = parser.parse_args()
    download(args.n_samples, args.output)
```

- [ ] **Step 9: Implement stage_0/eval.py**

```python
from __future__ import annotations
"""Stage 0 eval harness entry point.

Usage: uv run python -m stage_0.eval --benchmark ragbench [--n-samples 1000]
"""

import argparse
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from shared.eval.datasets import BENCHMARK_LOADERS
from shared.eval.harness import EvalHarness
from shared.eval.reporter import aggregate, check_regression, print_summary, save
from stage_0.retriever import BM25Retriever


class Stage0Harness(EvalHarness):
    stage = 0

    def __init__(self, retriever: BM25Retriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self.retriever.retrieve(query, top_k=top_k)

    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        # Stage 0 has no LLM — return passage IDs joined as the "answer"
        return " | ".join(retrieved_ids[:5]), 0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 — BM25 Baseline eval")
    parser.add_argument("--benchmark", default="ragbench", choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open("stage_0/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    index_path = Path(cfg["data"]["index_path"])
    if not index_path.exists():
        print(f"Index not found at {index_path}. Run: uv run python -m stage_0.indexer")
        raise SystemExit(1)

    print(f"Loading BM25 index from {index_path}...")
    retriever = BM25Retriever.load(index_path)

    harness = Stage0Harness(retriever)
    queries = BENCHMARK_LOADERS[args.benchmark](n_samples=args.n_samples)
    print(f"Running {len(queries)} queries...")

    results = harness.run(queries, top_k=cfg["bm25"]["top_k"])
    metrics = aggregate(results, stage=0, benchmark=args.benchmark)

    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or output_dir / f"{ts}.json"

    prev_runs = sorted(output_dir.glob("*.json"))
    if prev_runs:
        check_regression(metrics, prev_runs[-1])

    save(metrics, output_path)
    print_summary(metrics)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Download 10K arXiv sample for fast iteration**

```bash
uv run python scripts/download_arxiv.py --n-samples 10000 --output data/arxiv_abstracts_10k.json
```

Update `stage_0/config.toml` temporarily to use the 10K file:
```toml
abstracts_path = "data/arxiv_abstracts_10k.json"
index_path = "data/stage_0_bm25_index_10k.json"
```

- [ ] **Step 11: Build BM25 index on 10K sample**

```bash
uv run python -m stage_0.indexer --corpus data/arxiv_abstracts_10k.json --output data/stage_0_bm25_index_10k.json
```

Expected: completes in <60 seconds.

- [ ] **Step 12: Run Stage 0 smoke test eval**

```bash
uv run python -m stage_0.eval --benchmark ragbench --n-samples 50
```

Expected: metrics table printed, JSON saved to `results/stage_0/`.

- [ ] **Step 13: Commit Stage 0**

```bash
git add stage_0/ scripts/download_arxiv.py
git commit -m "$(cat <<'EOF'
feat(stage-0): BM25 baseline retriever and eval harness

BM25Okapi (k1=1.5, b=0.75) over arXiv abstracts. No LLM, no embeddings.
Index serialized as JSON (not pickle) for safety and inspectability.
Architectural shift: pre-AI keyword retrieval establishes the Recall@10
anchor that every subsequent stage is measured against (spec §1.3 Q1).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 5: Stage 1 — Naive RAG

**Architectural shift:** Dense embeddings (BGE-M3) + Qdrant ANN + Qwen 14B generation. Deliberately commits all "naive sins": fixed chunks, no rerank, no query rewriting.  
**Metric affected:** Recall@10 vs BM25 (semantic queries), Faithfulness (first generation stage).

**Files:**
- Create: `stage_1/__init__.py`
- Create: `stage_1/config.toml`
- Create: `stage_1/embedder.py`
- Create: `stage_1/indexer.py`
- Create: `stage_1/retriever.py`
- Create: `stage_1/generator.py`
- Create: `stage_1/pipeline.py`
- Create: `stage_1/eval.py`
- Create: `stage_1/tests/test_pipeline.py`

- [ ] **Step 1: Create stage_1 structure**

```bash
mkdir -p stage_1/tests
touch stage_1/__init__.py stage_1/tests/__init__.py
```

- [ ] **Step 2: Create stage_1/config.toml**

```toml
[embedding]
model = "BAAI/bge-m3"
batch_size = 64
dimension = 1024
device = "mps"

[qdrant]
host = "localhost"
port = 6333
collection = "arxiv_stage1"
hnsw_m = 16
hnsw_ef_construct = 100

[chunking]
chunk_size = 512
overlap = 0

[llm]
model = "qwen2.5:14b"
temperature = 0.1
max_tokens = 512
top_k_context = 5

[data]
abstracts_path = "data/arxiv_abstracts.json"

[eval]
benchmark = "ragbench"
n_samples = 1000
output_dir = "results/stage_1"
top_k = 10
```

- [ ] **Step 3: Write failing tests for Stage 1**

Create `stage_1/tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify failure**

```bash
uv run pytest stage_1/tests/test_pipeline.py -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError`

- [ ] **Step 5: Implement stage_1/embedder.py**

```python
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 0) -> list[str]:
    """Split text into fixed-size word chunks. No sentence awareness — deliberately naive."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


class BGEEmbedder:
    """BAAI/bge-m3 dense embedder. Lazy-loads model on first use."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "mps") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        self._load()
        vecs = self._model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, query: str) -> list[float]:
        self._load()
        return self._model.encode([query], normalize_embeddings=True)[0].tolist()
```

- [ ] **Step 6: Implement stage_1/retriever.py**

```python
from __future__ import annotations

from qdrant_client import QdrantClient

from stage_1.embedder import BGEEmbedder


class QdrantRetriever:
    """Dense retriever using Qdrant HNSW + BGE-M3 embeddings."""

    def __init__(self, collection: str, host: str, port: int, embedder: BGEEmbedder) -> None:
        self.collection = collection
        self.client = QdrantClient(host=host, port=port)
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Return top_k unique doc_ids ranked by cosine similarity."""
        query_vec = self.embedder.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k * 3,
        )
        seen: set[str] = set()
        doc_ids: list[str] = []
        for hit in hits:
            doc_id = hit.payload["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
            if len(doc_ids) >= top_k:
                break
        return doc_ids
```

- [ ] **Step 7: Implement stage_1/generator.py**

```python
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

    def __init__(self, model: str = "qwen2.5:14b", temperature: float = 0.1, max_tokens: int = 512) -> None:
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
        return resp["response"], resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)
```

- [ ] **Step 8: Implement stage_1/indexer.py**

```python
from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from stage_1.embedder import BGEEmbedder, chunk_text


def build_qdrant_index(
    abstracts_path: Path,
    collection: str,
    host: str,
    port: int,
    embedding_dim: int,
    chunk_size: int,
    batch_size: int,
    device: str,
) -> None:
    print(f"Loading abstracts from {abstracts_path}...")
    corpus = json.loads(abstracts_path.read_text())
    print(f"Loaded {len(corpus):,} documents.")

    client = QdrantClient(host=host, port=port)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    embedder = BGEEmbedder(device=device)
    all_chunks: list[dict] = []
    for doc in corpus:
        text = doc.get("title", "") + " " + doc.get("abstract", "")
        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=0)):
            all_chunks.append({"text": chunk, "doc_id": doc["id"], "chunk_idx": idx})

    print(f"Total chunks: {len(all_chunks):,}. Embedding...")
    t0 = time.perf_counter()

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors = embedder.embed([c["text"] for c in batch], batch_size=batch_size)
        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=str(uuid4()), vector=vectors[j], payload=batch[j])
                for j in range(len(batch))
            ],
        )
        print(f"  Indexed {min(i + batch_size, len(all_chunks)):,}/{len(all_chunks):,}")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=None)
    args = parser.parse_args()

    with open("stage_1/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    build_qdrant_index(
        abstracts_path=args.corpus or Path(cfg["data"]["abstracts_path"]),
        collection=cfg["qdrant"]["collection"],
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        embedding_dim=cfg["embedding"]["dimension"],
        chunk_size=cfg["chunking"]["chunk_size"],
        batch_size=cfg["embedding"]["batch_size"],
        device=cfg["embedding"]["device"],
    )
```

- [ ] **Step 9: Implement stage_1/pipeline.py**

```python
from __future__ import annotations

import tomllib

from stage_1.embedder import BGEEmbedder
from stage_1.generator import OllamaGenerator
from stage_1.retriever import QdrantRetriever


def load_pipeline(config_path: str = "stage_1/config.toml") -> tuple[QdrantRetriever, OllamaGenerator]:
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    embedder = BGEEmbedder(model_name=cfg["embedding"]["model"], device=cfg["embedding"]["device"])
    retriever = QdrantRetriever(
        collection=cfg["qdrant"]["collection"],
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        embedder=embedder,
    )
    generator = OllamaGenerator(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    return retriever, generator
```

- [ ] **Step 10: Implement stage_1/eval.py**

```python
from __future__ import annotations
"""Stage 1 eval harness entry point.

Usage: uv run python -m stage_1.eval --benchmark ragbench [--n-samples 1000] [--no-generation]
"""

import argparse
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from shared.eval.datasets import BENCHMARK_LOADERS
from shared.eval.harness import EvalHarness
from shared.eval.reporter import aggregate, check_regression, print_summary, save
from stage_1.generator import OllamaGenerator
from stage_1.retriever import QdrantRetriever


class Stage1Harness(EvalHarness):
    stage = 1

    def __init__(self, retriever: QdrantRetriever, generator: OllamaGenerator, top_k_context: int = 5) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k_context = top_k_context

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self.retriever.retrieve(query, top_k=top_k)

    def generate(self, query: str, retrieved_ids: list[str]) -> tuple[str, int, int]:
        return self.generator.generate(query, retrieved_ids[: self.top_k_context])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 — Naive RAG eval")
    parser.add_argument("--benchmark", default="ragbench", choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args()

    with open("stage_1/config.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    from stage_1.pipeline import load_pipeline
    retriever, generator = load_pipeline()

    harness = Stage1Harness(retriever, generator, top_k_context=cfg["llm"]["top_k_context"])
    queries = BENCHMARK_LOADERS[args.benchmark](n_samples=args.n_samples)
    print(f"Running {len(queries)} queries against Stage 1 Naive RAG...")

    results = harness.run(queries, top_k=cfg["eval"]["top_k"], include_generation=not args.no_generation)
    metrics = aggregate(results, stage=1, benchmark=args.benchmark)

    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or output_dir / f"{ts}.json"

    prev_runs = sorted(output_dir.glob("*.json"))
    if prev_runs:
        check_regression(metrics, prev_runs[-1])

    save(metrics, output_path)
    print_summary(metrics)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 11: Run Stage 1 unit tests — expect pass**

```bash
uv run pytest stage_1/tests/test_pipeline.py -v
```

Expected: 4/4 PASS.

- [ ] **Step 12: Start Qdrant and build Stage 1 index (10K sample first)**

```bash
# Start Qdrant
qdrant &
sleep 3
curl -s http://localhost:6333/healthz  # Expected: {"title":"qdrant","version":"..."}

# Build index on 10K sample (update config.toml to point at 10k file)
uv run python -m stage_1.indexer --corpus data/arxiv_abstracts_10k.json
```

Expected: ~5-15 minutes for 10K docs.

- [ ] **Step 13: Run Stage 1 smoke test eval**

```bash
uv run python -m stage_1.eval --benchmark ragbench --n-samples 50 --no-generation
```

Expected: metrics table. Different numbers from Stage 0.

- [ ] **Step 14: Commit Stage 1**

```bash
git add stage_1/
git commit -m "$(cat <<'EOF'
feat(stage-1): Naive RAG with BGE-M3 + Qdrant HNSW + Qwen 14B

Fixed 512-token chunks, no rerank, no query rewriting — the textbook
2023 RAG diagram. Deliberately commits all naive sins so Stage 2
improvements have a meaningful baseline (spec §2.2).
Architectural shift: dense embeddings close the semantic gap BM25
cannot cross on paraphrased and conceptual queries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 6: Full Dataset and Production Eval

- [ ] **Step 1: Download full 1.7M arXiv dataset (run overnight)**

```bash
nohup uv run python scripts/download_arxiv.py --output data/arxiv_abstracts.json > logs/arxiv_download.log 2>&1 &
```

Expected: ~2 hours, ~3GB.

- [ ] **Step 2: Update config.toml files for production paths**

In `stage_0/config.toml`:
```toml
abstracts_path = "data/arxiv_abstracts.json"
index_path = "data/stage_0_bm25_index.json"
```

In `stage_1/config.toml`:
```toml
abstracts_path = "data/arxiv_abstracts.json"
```

- [ ] **Step 3: Build Stage 0 BM25 full index**

```bash
uv run python -m stage_0.indexer
```

Expected: 1-2 hours, ~4GB JSON index file.

- [ ] **Step 4: Run Stage 0 production eval**

```bash
uv run python -m stage_0.eval --benchmark ragbench --n-samples 1000
```

Record the Recall@10 number. This is **the project anchor metric**.

- [ ] **Step 5: Build Stage 1 Qdrant full index**

```bash
uv run python -m stage_1.indexer
```

Expected: ~6 hours, ~12GB Qdrant storage.

- [ ] **Step 6: Run Stage 1 production eval**

```bash
uv run python -m stage_1.eval --benchmark ragbench --n-samples 1000 --no-generation
```

- [ ] **Step 7: Print cross-stage comparison**

```bash
uv run python -c "
import json
from pathlib import Path

s0 = json.loads(sorted(Path('results/stage_0').glob('*.json'))[-1].read_text())
s1 = json.loads(sorted(Path('results/stage_1').glob('*.json'))[-1].read_text())

for k in ['recall_at_5','recall_at_10','recall_at_20','mrr','ndcg_at_10']:
    delta = s1[k] - s0[k]
    arrow = '↑' if delta > 0 else '↓'
    print(f'{k:22s}: S0={s0[k]:.4f}  S1={s1[k]:.4f}  {arrow}{abs(delta):.4f}')
"
```

- [ ] **Step 8: Commit production results**

```bash
git add results/ stage_0/config.toml stage_1/config.toml
git commit -m "$(cat <<'EOF'
chore: production eval results for Stage 0 (BM25) and Stage 1 (Naive RAG)

Records baseline Recall@10 numbers on full 1.7M arXiv corpus with
1000 RAGBench queries. These anchor the entire comparison framework
per spec §1.3 success criteria questions 1 and 5.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Verification Checklist

After completing all tasks:

- [ ] `uv run pytest` — all tests pass (10 shared + 5 stage_0 + 4 stage_1)
- [ ] `uv run python -m stage_0.eval --benchmark ragbench --n-samples 10` — prints metrics table
- [ ] `uv run python -m stage_1.eval --benchmark ragbench --n-samples 10 --no-generation` — prints metrics table
- [ ] `results/stage_0/*.json` and `results/stage_1/*.json` exist with real numbers
- [ ] `git log --oneline` shows 6+ commits with "Co-Authored-By: Claude Sonnet 4.6" lines
- [ ] `gh repo view Abivarma/RAG_Evolution_Lab` — repo visible with all files
- [ ] `.claude/settings.json` hooks file present
- [ ] `.claude/commands/` has 5 slash command files
- [ ] `.claude/agents/` has 3 subagent definition files

**Spec alignment check (§1.3 success criteria):**
- Q1: "How much does hybrid retrieval improve Recall@10?" — Stage 0 and 1 numbers set the denominator. ✅
- Q5: "Cost-per-correct-answer for each stage?" — Token counting wired into harness from Stage 1. ✅
- All stages share identical CLI: `python -m stage_N.eval --benchmark NAME`. ✅

---

## Next Plan

Once Stage 0 and 1 have real production numbers, create:
`docs/superpowers/plans/2026-05-15-rag-evolution-lab-stage2-advanced-rag.md`

Stage 2 introduces: hybrid retrieval (BM25 + dense + RRF), BGE-reranker-v2 cross-encoder, query rewriting, and semantic chunking. Each component gets an ablation run. This is when Recall@10 numbers become interesting.
