Archive the completed stage's data to warm tier storage and advance to the next stage.

Usage: /promote-stage

Steps:
1. Read .claude/current_stage.txt to determine current stage N.
2. Ask user to confirm before proceeding (destructive to hot-tier data).
3. WARM_TIER=/Users/abivarma/Library/CloudStorage/Box-Box/RAG_Evolution_Lab
4. Archive: `tar -czf $WARM_TIER/stage_${N}_$(date +%Y%m%d).tar.gz stage_${N}/data/ 2>/dev/null`
5. Verify: `tar -tzf $WARM_TIER/stage_${N}_$(date +%Y%m%d).tar.gz | wc -l`
6. Remove hot-tier data (NOT code): `rm -rf stage_${N}/data/`
7. Print disk freed: `df -h .`
8. Update .claude/current_stage.txt to $((N+1))
