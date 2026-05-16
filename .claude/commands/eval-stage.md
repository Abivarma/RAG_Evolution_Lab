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
