---
name: PoseNet Score Tracking — Keep Detailed Per-Pair Records
description: Always track PoseNet distortion per-pair, not just averages. The heavy tail dominates the score.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Always track PoseNet distortion at the PER-PAIR level, not just averages.

**Why:** The score formula uses sqrt(10 * MEAN(posenet_per_pair)). The mean is dominated by heavy-tail outlier pairs (5% of pairs can contribute 80%+ of the PoseNet term). Tracking only averages hides the real optimization opportunity.

**How to apply:**
- Every TTO run should save per-pair PoseNet distortion (600 values)
- The pair difficulty map (`experiments/pair_difficulty_map.py`) produces this — RUN IT
- When comparing experiments, compare the WORST 30 pairs, not just the mean
- The step curve showed pairs 100-129 have PoseNet=165 while the average is 0.031 — a 5000x disparity
- Adaptive TTO budget allocation depends on knowing WHICH pairs are hard
- Always record: mean, std, min, max, p95, p99, and the full distribution
