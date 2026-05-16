Run corpus ingestion for the current active stage.

Usage: /ingest <corpus>
Valid corpus values: arxiv, sec

Steps:
1. Read .claude/current_stage.txt to determine active stage N.
2. If corpus=arxiv and stage in [0,1,2,3,4,7,8]:
   Run: `uv run python -m stage_<N>.indexer --corpus data/arxiv_abstracts.json`
3. If corpus=sec and stage in [5,6,7]:
   Run: `uv run python -m stage_<N>.indexer --corpus data/sec_10k/`
4. Report index size and time taken.
