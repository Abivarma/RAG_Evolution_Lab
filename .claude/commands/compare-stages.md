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
