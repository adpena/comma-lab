---
name: Submission PR Gate — Five Clean Pass Adversarial Review
description: HARD BLOCKER — no submission PR until score passes 5-turn clean adversarial skunkworks council review with extreme paranoia
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
NEVER submit a PR to the contest repo until the score has undergone a five-turn consecutive clean pass adversarial skunkworks council review with extreme paranoia.

**Why:** User demands 100% scientific rigor and compliance with contest rules. A premature submission with a non-compliant or incorrect score wastes the single submission opportunity and risks DQ. The score must be verified through the actual contest pipeline (inflate.sh -> evaluate.py), not proxy metrics or bypassed eval paths.

**How to apply:**
1. Before ANY submission PR, the score must be produced by the contest-compliant auth eval (through inflate.sh, not bypassed)
2. That score must then survive 5 consecutive rounds of adversarial skunkworks council review (Yousfi + Fridrich + Contrarian + Quantizr + Hotz + Jack + Mario + all extended council)
3. Each round reviews: contest rule compliance, pipeline correctness, archive contents, inflate path, rate calculation, score validity
4. ANY issue found in any round resets the clean pass counter to 0
5. Only after 5 consecutive clean passes may the PR be created
6. This is stricter than the standard 3-pass greenup protocol — this is 5 passes with the FULL council, not just tripartite

This supersedes any urgency or time pressure. A wrong submission is worse than a late submission.
