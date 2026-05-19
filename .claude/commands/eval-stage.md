Run the eval harness for stage $ARGUMENTS.

Benchmark defaults by stage (per SPEC.md §3.2):
  S0, S1, S2: scifact (open-corpus Recall@K) — use ragbench for ranking MRR/nDCG
  S3, S4, S8: multihop (multi-hop reasoning only)
  S5, S6:     finder

Steps:
1. Parse $ARGUMENTS as: <stage_number> [--benchmark <name>]
   Use stage-appropriate default above if not specified.
2. Always run a 3-query smoke test FIRST to confirm the harness works:
   `uv run python -m stage_<N>.harness --benchmark <name> --n-samples 3 --no-generation`
   If it fails, fix before running full eval.
3. Verify Qdrant running if stage >= 1 (open-corpus): `curl -s http://localhost:6333/healthz`
4. Verify Ollama: `ollama list` — confirm qwen2.5:14b is present.
5. Run full eval:
   `uv run python -m stage_<N>.harness --benchmark <name> --n-samples 50 --no-generation`
   Entry point is stage_N.harness for S2+, stage_N.eval for S0/S1.
6. Print summary table: Recall@5/10/20, MRR, nDCG@10, p50/p95/p99 latency.
7. Compare to previous run. Flag regressions >5% with ⚠.
8. Print reasoning trace for last query (S3+) to confirm agent worked correctly.
